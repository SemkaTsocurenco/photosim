"""Общие проверки входных данных photosim.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ROTATION_ATOL",
    "as_flat_points",
    "as_points",
    "as_time",
    "as_vec3",
    "check_rotation",
    "finite_scalar",
    "is_rotation_matrix",
    "readonly",
    "squeeze_time",
    "unit_vec3",
]

#: Допуск на ортогональность и определитель матрицы поворота.
ROTATION_ATOL = 1e-9

_NORM_EPS = 1e-12


def readonly(a: NDArray) -> NDArray:
    """Закрыть массив для записи и вернуть его."""
    a.flags.writeable = False
    return a


def finite_scalar(value: float, name: str) -> float:
    """Привести к ``float`` и проверить, что число конечно."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} должно быть числом, получено {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} должно быть конечным, получено {value!r}")
    return v


def as_points(points: ArrayLike, dim: int, name: str = "points") -> NDArray:
    """Массив точек float64 формы ``(..., dim)``.

    Копию не делает (это горячий путь рендера). Значения не проверяет:
    ``nan`` в точках допустим, его смысл решает вызывающий код.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 0 or pts.shape[-1] != dim:
        raise ValueError(
            f"{name}: ожидалась форма (..., {dim}), а пришла {pts.shape}"
        )
    return pts


def as_flat_points(
    points: ArrayLike, dim: int, name: str = "points"
) -> tuple[NDArray, tuple[int, ...]]:
    """Как ``as_points``, но вытянуть в ``(M, dim)`` и вернуть ведущие измерения.

    Нужна коду, который работает со списком точек (камера). Обратно:
    ``out.reshape(lead + (k,))``.
    """
    pts = as_points(points, dim, name)
    return pts.reshape(-1, dim), pts.shape[:-1]


def as_vec3(value: ArrayLike, name: str) -> NDArray:
    """Один вектор формы ``(3,)`` с конечными значениями, копия только для чтения."""
    v = np.array(value, dtype=np.float64)
    if v.shape != (3,):
        raise ValueError(f"{name} должен иметь форму (3,), получена {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} должен содержать конечные значения, получено {v}")
    return readonly(v)


def unit_vec3(value: ArrayLike, name: str) -> NDArray:
    """Как ``as_vec3``, но ненулевой и нормированный на единицу."""
    v = as_vec3(value, name)
    n = float(np.linalg.norm(v))
    if n <= _NORM_EPS:
        raise ValueError(f"{name} не должен быть нулевым, получено {v}")
    return readonly(v / n)


def as_time(t: ArrayLike, name: str = "t") -> tuple[NDArray, bool]:
    """Единое правило форм по времени.

    Число (или 0-мерный массив) даёт массив формы ``(1,)`` и флаг
    ``scalar=True``; массив формы ``(Nt,)`` возвращается как есть с флагом
    ``False``. Другие формы и нефинитные значения дают ``ValueError``.

    Методы считают всё на массиве ``(Nt,)``, а в конце вызывают
    ``squeeze_time``, чтобы для числа вернуть ``(3,)`` вместо ``(1, 3)``.
    """
    arr = np.asarray(t, dtype=np.float64)
    if arr.ndim == 0:
        scalar = True
        arr = arr.reshape(1)
    elif arr.ndim == 1:
        scalar = False
    else:
        raise ValueError(
            f"{name}: ожидается число или массив формы (Nt,), получена форма "
            f"{arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} должно содержать конечные значения")
    return arr, scalar


def squeeze_time(out: NDArray, scalar: bool) -> NDArray:
    """Убрать ось времени, если время было числом."""
    return out[0] if scalar else out


def is_rotation_matrix(R: ArrayLike, atol: float = ROTATION_ATOL) -> NDArray:
    """Поэлементная проверка матриц поворота: форма ``(..., 3, 3)`` → ``(...)``.

    Матрица считается поворотом, если она конечна, ``R Rᵀ = I`` и
    ``det R = +1`` с допуском ``atol``. Отражения (``det = -1``) не проходят.
    """
    m = np.asarray(R, dtype=np.float64)
    if m.ndim < 2 or m.shape[-2:] != (3, 3):
        raise ValueError(f"ожидаются матрицы формы (..., 3, 3), получена {m.shape}")
    finite = np.all(np.isfinite(m), axis=(-2, -1))
    safe = np.where(finite[..., None, None], m, 0.0)
    orth = np.abs(safe @ np.swapaxes(safe, -1, -2) - np.eye(3)).max(axis=(-2, -1))
    det = np.abs(np.linalg.det(safe) - 1.0)
    return finite & (orth <= atol) & (det <= atol)


def check_rotation(R: ArrayLike, name: str, atol: float = ROTATION_ATOL) -> NDArray:
    """Проверить одну матрицу поворота ``(3, 3)``; вернуть копию только для чтения."""
    m = np.array(R, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"{name} должна быть (3, 3), а пришла {m.shape}")
    # nan и inf отдельно не ловим: ошибка ортогональности тогда nan,
    # сравнение `nan <= atol` ложно, и матрица отбрасывается как неортогональная.
    with np.errstate(invalid="ignore", over="ignore"):
        orth = float(np.abs(m.T @ m - np.eye(3)).max())
    if not orth <= atol:
        raise ValueError(
            f"{name} не ортогональна (ошибка {orth:.2g}, допуск {atol:g}), "
            "это не поворот"
        )
    det = float(np.linalg.det(m))
    if not abs(det - 1.0) <= atol:
        raise ValueError(
            f"det {name} = {det:.3f}: это отражение, а не поворот "
            "(переставлены оси или перепутан знак)"
        )
    return readonly(m)
