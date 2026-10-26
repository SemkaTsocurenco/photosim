"""Методы оценки центра марки по кадру.

Все методы получают одинаково подготовленный кадр (вычтены уровень чёрного
и фон) и возвращают центр в координатах кадра, пикс. Новый метод
добавляется декоратором ``@register("name")`` без правки остального кода.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from skimage.measure import find_contours

from photosim._checks import finite_scalar, readonly

__all__ = [
    "BORDER_PX",
    "ESTIMATORS",
    "CenterEstimate",
    "ellipse_from_conic",
    "estimate",
    "fit_ellipse_direct",
    "register",
]

# Ширина рамки ROI для оценки фона, пикс.
BORDER_PX = 2

# Реестр методов
ESTIMATORS: dict[str, Callable[..., tuple[NDArray, NDArray, dict[str, NDArray]]]] = {}


_MIN_SIGMA = 0.3


@dataclass(frozen=True, eq=False)
class CenterEstimate:
    """Результат оценки центра.

    uv : (2,) или (n, 2)
        Центр в координатах кадра, пикс.; ``nan``, где ``ok = False``.
    ok : () или (n,), bool
        Метод отработал.
    info : dict
        Диагностика метода; у пачки массивы с первой осью ``n``. Всегда есть
        ``background`` — вычтенный фон, ед. АЦП.
    """

    uv: NDArray
    ok: NDArray
    info: dict[str, Any]


def register(name: str, **defaults: Any) -> Callable:
    """Декоратор: добавить метод в ``ESTIMATORS`` под именем ``name``.
    """

    def deco(fn: Callable) -> Callable:
        if name in ESTIMATORS:
            raise ValueError(f"метод {name!r} уже зарегистрирован")
        ESTIMATORS[name] = partial(fn, **defaults) if defaults else fn
        return fn

    return deco


def estimate(
    name: str,
    image: ArrayLike,
    roi: tuple[int, int, int, int],
    *,
    black_level_dn: float = 0.0,
    background: str | float | None = "border",
    **params: Any,
) -> CenterEstimate:
    """Оценить центр марки методом ``name``
    """
    try:
        method = ESTIMATORS[name]
    except KeyError:
        raise KeyError(
            f"неизвестный метод {name!r}, доступны: {', '.join(sorted(ESTIMATORS))}"
        ) from None
    images, single = _as_stack(image)
    origin = _check_roi(roi, images.shape[1:])
    black = finite_scalar(black_level_dn, "black_level_dn")
    prepared, bg = _preprocess(images - black, background)

    uv, ok, info = method(prepared, origin, **params)
    ok = np.asarray(ok, dtype=bool) & np.isfinite(uv).all(axis=1)
    uv = np.where(ok[:, None], uv, np.nan)
    info = {"background": bg, **info}
    if single:
        uv, ok = uv[0], ok[0, ...]
        info = {k: v[0] for k, v in info.items()}
    return CenterEstimate(readonly(uv), readonly(ok), info)


# Предобработка


def _as_stack(image: ArrayLike) -> tuple[NDArray, bool]:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"image: ожидалась форма (h, w) или (n, h, w), а пришла {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("image должен содержать конечные значения")
    return (arr[None], True) if arr.ndim == 2 else (arr, False)


def _check_roi(roi: tuple[int, int, int, int], shape: tuple[int, int]) -> NDArray:
    if roi is None:
        raise ValueError("roi = None: марка вне кадра, оценивать нечего")
    u0, v0, w, h = roi
    if (h, w) != tuple(shape):
        raise ValueError(f"roi {roi} не совпадает с формой кадра {tuple(shape)}")
    if h < 1 or w < 1:
        raise ValueError(f"пустой ROI {roi}")
    return np.array([u0, v0], dtype=np.float64)


def _border_mask(h: int, w: int) -> NDArray:
    mask = np.zeros((h, w), dtype=bool)
    b = BORDER_PX
    mask[:b] = mask[-b:] = True
    mask[:, :b] = mask[:, -b:] = True
    return mask


def _preprocess(
    images: NDArray, background: str | float | None
) -> tuple[NDArray, NDArray]:
    """Вычесть фон; ``images`` — новый массив, его можно менять на месте."""
    n, h, w = images.shape
    if isinstance(background, str):
        if background != "border":
            raise ValueError(
                f"background должен быть 'border', числом или None, "
                f"получено {background!r}"
            )
        bg = np.median(images[:, _border_mask(h, w)], axis=1)
    elif background is None:
        bg = np.zeros(n)
    else:
        bg = np.full(n, finite_scalar(background, "background"))
    images -= bg[:, None, None]
    return images, bg


# Центр тяжести


@register("centroid")
@register("centroid_thr50", threshold=0.5)
@register("centroid_sq", power=2)
def centroid(
    images: NDArray,
    origin: NDArray,
    *,
    threshold: float | None = None,
    power: float = 1,
) -> tuple[NDArray, NDArray, dict[str, NDArray]]:
    """Центр тяжести
    """
    p = finite_scalar(power, "power")
    if p <= 0:
        raise ValueError(f"power должно быть больше 0, получено {power!r}")
    weights = images
    if threshold is not None:
        t = finite_scalar(threshold, "threshold")
        if not 0.0 <= t < 1.0:
            raise ValueError(f"threshold должен быть в [0, 1), получено {threshold!r}")
        weights = images - t * images.max(axis=(1, 2), keepdims=True)
    weights = np.maximum(weights, 0.0)
    if p != 1:
        weights **= p

    _, h, w = images.shape
    total = weights.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = weights.sum(axis=1) @ np.arange(w) / total
        v = weights.sum(axis=2) @ np.arange(h) / total
    return np.stack([u, v], axis=1) + origin, total > 0, {}


# Двумерная гауссиана


def _gauss_parts(p: NDArray, x: NDArray, y: NDArray) -> tuple:
    _, _, u, v, sx, sy, th = p
    c, s = math.cos(th), math.sin(th)
    dx, dy = x - u, y - v
    xr = c * dx + s * dy
    yr = c * dy - s * dx
    g = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return g, xr, yr, c, s


def _gauss_residual(
    p: NDArray, x: NDArray, y: NDArray, z: NDArray, sw: NDArray | float = 1.0
) -> NDArray:
    """Невязка, умноженная на корень из веса пикселя ``sw``."""
    return sw * (p[0] * _gauss_parts(p, x, y)[0] + p[1] - z)


def _gauss_jac(
    p: NDArray, x: NDArray, y: NDArray, z: NDArray, sw: NDArray | float = 1.0
) -> NDArray:
    A, _, _, _, sx, sy, _ = p
    g, xr, yr, c, s = _gauss_parts(p, x, y)
    ag = A * g
    ax, ay = xr / sx**2, yr / sy**2
    J = np.empty((x.size, 7))
    J[:, 0] = g
    J[:, 1] = 1.0
    J[:, 2] = ag * (c * ax - s * ay)
    J[:, 3] = ag * (s * ax + c * ay)
    J[:, 4] = ag * xr**2 / sx**3
    J[:, 5] = ag * yr**2 / sy**3
    J[:, 6] = ag * xr * yr * (1 / sy**2 - 1 / sx**2)
    return J * np.reshape(sw, (-1, 1))


def _gauss_initial(z: NDArray, x: NDArray, y: NDArray, b0: float) -> NDArray | None:
    """Начальное приближение: центр тяжести и вторые моменты над фоном ``b0``."""
    wgt = np.maximum(z - b0, 0.0)
    total = wgt.sum()
    if not total > 0:
        return None
    u, v = wgt @ x / total, wgt @ y / total
    dx, dy = x - u, y - v
    cxy = wgt @ (dx * dy)
    cov = np.array([[wgt @ (dx * dx), cxy], [cxy, wgt @ (dy * dy)]]) / total
    evals, evecs = np.linalg.eigh(cov)  # по возрастанию: большая ось вторая
    sx, sy = np.sqrt(np.maximum(evals[::-1], _MIN_SIGMA**2))
    th = math.atan2(evecs[1, 1], evecs[0, 1])
    return np.array([z.max() - b0, b0, u, v, sx, sy, th])


def _window(center: float, n: int) -> NDArray:
    """Веса пикселей ``0..n-1`` в окне, симметричном вокруг ``center``.

    Полуширина — до ближайшего края ROI (края пикселей на -0,5 и n - 0,5);
    пиксель на границе окна входит долей, попавшей в окно. Окно сдвигается
    с центром непрерывно: без скачков, когда фаза переходит через целое.
    """
    half = min(center + 0.5, n - 0.5 - center)
    left = np.arange(n) - 0.5
    inside = np.minimum(left + 1, center + half) - np.maximum(left, center - half)
    return np.clip(inside, 0.0, 1.0)


@register("gauss2d")
def gauss2d(
    images: NDArray, origin: NDArray
) -> tuple[NDArray, NDArray, dict[str, NDArray]]:
    """Подгонка ``A * exp(-q/2) + B`` (7 параметров), цикл по кадрам.

    Подгонка идёт в окне, симметричном вокруг начального приближения (центр
    тяжести): иначе несимметричный ROI тянет оценку к своему центру, на
    0,02 пикс. при сдвиге ROI на пиксель. Модель не уточняется: смещение
    от несовпадения гауссианы с диском — изучаемый эффект. ``ok = False``,
    если подгонка не сошлась или центр вышел за ROI; в ``info`` — параметры,
    невязка, ``status`` и ``nfev`` из ``least_squares``.
    """
    n, h, w = images.shape
    y, x = (a.ravel().astype(np.float64) for a in np.mgrid[0:h, 0:w])
    border = _border_mask(h, w)
    params = np.full((n, 7), np.nan)
    residual = np.full(n, np.nan)
    status = np.full(n, np.nan)
    nfev = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    for k, img in enumerate(images):
        z = img.ravel()
        p0 = _gauss_initial(z, x, y, float(np.median(img[border])))
        if p0 is None:
            continue
        weight = np.outer(_window(p0[3], h), _window(p0[2], w)).ravel()
        sel = weight > 0
        if sel.sum() < 7:
            continue
        sw = np.sqrt(weight[sel])
        res = least_squares(
            _gauss_residual,
            p0,
            jac=_gauss_jac,
            args=(x[sel], y[sel], z[sel], sw),
            x_scale="jac",
        )
        A, B, u, v, sx, sy, th = res.x
        sx, sy = abs(sx), abs(sy)
        if sx < sy:
            sx, sy, th = sy, sx, th + math.pi / 2
        th = (th + math.pi / 2) % math.pi - math.pi / 2
        params[k] = A, B, u, v, sx, sy, th
        residual[k] = math.sqrt(2 * res.cost / weight[sel].sum())
        status[k], nfev[k] = res.status, res.nfev
        ok[k] = (
            res.success
            and A > 0
            and sy > 0
            and -0.5 <= u <= w - 0.5
            and -0.5 <= v <= h - 0.5
        )
    info = {
        "amplitude": params[:, 0],
        "offset": params[:, 1],
        "sigma_x": params[:, 4],
        "sigma_y": params[:, 5],
        "theta": params[:, 6],
        "residual": residual,
        "status": status,
        "nfev": nfev,
    }
    return params[:, 2:4] + origin, ok, info


# Эллипс по контуру половинного уровня


def fit_ellipse_direct(points: ArrayLike) -> NDArray | None:
    """Эллипс по точкам прямым методом Фитцгиббона
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 6:
        return None
    # нормировка точек для обусловленности: центр в 0, СКО радиуса 1
    m = pts.mean(axis=0)
    s = math.sqrt(((pts - m) ** 2).sum(axis=1).mean())
    if not s > 0:
        return None
    x, y = ((pts - m) / s).T
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    M = np.array([M[2] / 2, -M[1], M[0] / 2])  # C1^-1 M
    evecs = np.real(np.linalg.eig(M)[1])
    cond = 4 * evecs[0] * evecs[2] - evecs[1] ** 2  # 4ac - b**2 > 0: эллипс
    good = np.flatnonzero(cond > 0)
    if good.size != 1:
        return None
    a1 = evecs[:, good[0]]
    a, b, c = a1
    d, e, f = T @ a1
    Cn = np.array([[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]])
    N = np.array([[1 / s, 0, -m[0] / s], [0, 1 / s, -m[1] / s], [0, 0, 1]])
    C = N.T @ Cn @ N
    C /= np.linalg.norm(C)
    return -C if C[0, 0] < 0 else C


def ellipse_from_conic(C: ArrayLike) -> tuple[NDArray, NDArray, float]:

    C = np.asarray(C, dtype=np.float64)
    if C[0, 0] < 0:
        C = -C
    A, b = C[:2, :2], C[:2, 2]
    nan = (np.full(2, np.nan), np.full(2, np.nan), math.nan)
    if not np.linalg.det(A) > 0:
        return nan
    center = np.linalg.solve(A, -b)
    k = C[2, 2] + b @ center
    if not k < 0:
        return nan
    evals, evecs = np.linalg.eigh(A / -k)  # по возрастанию: большая ось первая
    axes = 1 / np.sqrt(evals)
    angle = math.atan2(evecs[1, 0], evecs[0, 0])
    return center, axes, (angle + math.pi / 2) % math.pi - math.pi / 2


def _contour_length(c: NDArray) -> float:
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


@register("ellipse")
def ellipse(
    images: NDArray, origin: NDArray, *, percentile: float = 95.0
) -> tuple[NDArray, NDArray, dict[str, NDArray]]:
    """Центр эллипса, вписанного в контур половинного уровня.
    """
    n = images.shape[0]
    shift = np.array([[1.0, 0, -origin[0]], [0, 1.0, -origin[1]], [0, 0, 1.0]])
    uv = np.full((n, 2), np.nan)
    axes = np.full((n, 2), np.nan)
    angle = np.full(n, np.nan)
    conic = np.full((n, 3, 3), np.nan)
    for k, img in enumerate(images):
        peak_max = img.max()
        if not peak_max > 0:
            continue
        peak = np.percentile(img[img > 0.5 * peak_max], percentile)
        closed = [
            c
            for c in find_contours(img, 0.5 * peak)
            if len(c) > 6 and np.array_equal(c[0], c[-1])
        ]
        if not closed:
            continue
        longest = max(closed, key=_contour_length)
        C = fit_ellipse_direct(longest[:-1, ::-1])  # (v, u) -> (u, v), без повтора
        if C is None:
            continue
        center, axes[k], angle[k] = ellipse_from_conic(C)
        uv[k] = center + origin
        Cf = shift.T @ C @ shift  # в координаты кадра
        conic[k] = Cf / np.linalg.norm(Cf)
    info = {
        "semi_major": axes[:, 0],
        "semi_minor": axes[:, 1],
        "angle": angle,
        "conic": conic,
    }
    return uv, np.isfinite(uv).all(axis=1), info
