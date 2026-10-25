"""Рендер: облучённость без шума в области ROI.

По камере, её позе, марке, движению и экспозиции строит массив ``(h, w)``
в относительных единицах марки.
Флагов эффектов здесь нет: рендер всегда считает честную физику, а
эффекты «выключаются» выбором ``Motion`` и ``nt``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from photosim._checks import finite_scalar, readonly
from photosim.camera import Camera, Pose
from photosim.scene import Motion, exposure_samples
from photosim.target import TargetShape

__all__ = [
    "RenderResult",
    "compute_roi",
    "intersect_plane",
    "render",
]

# Сколько элементов «луч × подшаг» обработать.
_BATCH_ELEMENTS = 1_000_000


@dataclass(frozen=True, eq=False)
class RenderResult:
    """Результат рендера.

    irradiance : (h, w), только для чтения
        Средняя по пикселю и по экспозиции яркость, отн. ед. марки.
        Для пустого кадра — массив формы ``(0, 0)``.
    roi : (u0, v0, w, h) или None
        Область кадра;
    ss, nt : int
        Субпикселей по оси и подшагов по времени.
    times : ndarray, только для чтения
        Использованные моменты: ``(nt,)`` для глобального затвора,
        ``(nt, h)`` для построчного (столбец ``j`` — строка ``v0 + j``).
    empty : bool
        Марка целиком вне кадра или позади камеры.
    clipped : bool
        ROI обрезана краем кадра хотя бы с одной стороны: часть марки или
        запаса вокруг неё не попала в кадр, и центр тяжести может быть
        смещён. При ``empty=True`` — ``False``.
    """

    irradiance: NDArray
    roi: tuple[int, int, int, int] | None
    ss: int
    nt: int
    times: NDArray
    empty: bool
    clipped: bool


def intersect_plane(o_l: ArrayLike, d_l: ArrayLike) -> NDArray:
    """Пересечение лучей с плоскостью марки.
    """
    o = np.asarray(o_l, dtype=np.float64)
    d = np.asarray(d_l, dtype=np.float64)
    return _intersect(o[..., 0], o[..., 1], o[..., 2], d[..., 0], d[..., 1], d[..., 2])


def _intersect(ox, oy, oz, dx, dy, dz) -> NDArray:
    """Пересечение по компонентам

    ``o*`` — компоненты центра камеры в системе марки
    ``d*`` — компоненты лучей
    Работа с отдельными непрерывными компонентами в несколько раз быстрее,
    чем со срезами ``[..., k]`` массива ``(..., 3)``.
    """
    # Марка видна спереди, только если o_z > 0; иначе o_z -> nan, и все s — nan.
    oz = np.where(oz > 0, oz, np.nan)
    shape = np.broadcast_shapes(np.shape(ox), np.shape(dz))
    xy = np.empty((*shape, 2))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # при o_z > 0 условие s > 0 равносильно d_z < 0; nan и d_z = 0 отсекаются
        s = np.where(dz < 0, oz / -dz, np.nan)
        np.multiply(dx, s, out=xy[..., 0])
        np.multiply(dy, s, out=xy[..., 1])
        xy[..., 0] += ox
        xy[..., 1] += oy
    return xy


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} должно быть целым, получено {value!r}")
    if value < 1:
        raise ValueError(f"{name} должно быть не меньше 1, получено {value}")
    return int(value)


def compute_roi(
    camera: Camera,
    pose: Pose,
    target: TargetShape,
    motion: Motion,
    t_k: float,
    tau: float,
    *,
    margin_px: float = 2,
    t_row: float = 0.0,
    n: int = 9,
    n_boundary: int = 64,
) -> tuple[tuple[int, int, int, int] | None, bool]:
    """Область кадра ``(u0, v0, w, h)``, в которой лежит марка за всю экспозицию.

    Возвращает ``(roi, clipped)``. ``roi`` — ``None``, если марка вне кадра
    (тогда ``clipped=False``). ``clipped=True``, если хотя бы одна граница
    ROI вместе с запасом ``margin_px`` вышла за кадр и была обрезана.
    """
    tk = finite_scalar(t_k, "t_k")
    ta = finite_scalar(tau, "tau")
    if ta <= 0:
        raise ValueError(f"tau должно быть больше 0, получено {tau!r}")
    margin = finite_scalar(margin_px, "margin_px")
    if margin < 0:
        raise ValueError(
            f"margin_px не может быть отрицательным, получено {margin_px!r}"
        )
    shift = camera.height * finite_scalar(t_row, "t_row")
    t_lo = tk + min(0.0, shift)
    t_hi = tk + ta + max(0.0, shift)

    times = np.concatenate([[t_lo], exposure_samples(t_lo, t_hi - t_lo, n), [t_hi]])
    contour = target.boundary(n_boundary)
    pts_w = motion.to_world(
        np.broadcast_to(contour, (times.size, *contour.shape)), times
    )
    uv, ok = camera.project(pose.to_camera(pts_w), return_mask=True)
    if not np.any(ok):
        return None, False
    if not np.all(ok):
        raise ValueError(
            "контур марки частично позади камеры или за r_max дисторсии: "
            "ROI не определена"
        )

    u0 = math.floor(float(uv[..., 0].min()) - margin)
    u1 = math.ceil(float(uv[..., 0].max()) + margin)
    v0 = math.floor(float(uv[..., 1].min()) - margin)
    v1 = math.ceil(float(uv[..., 1].max()) + margin)
    clipped = u0 < 0 or v0 < 0 or u1 > camera.width - 1 or v1 > camera.height - 1
    u0, v0 = max(u0, 0), max(v0, 0)
    u1, v1 = min(u1, camera.width - 1), min(v1, camera.height - 1)
    if u0 > u1 or v0 > v1:
        return None, False
    return (u0, v0, u1 - u0 + 1, v1 - v0 + 1), clipped


def _accumulate(
    d_rows: NDArray, C: NDArray, R: NDArray, X: NDArray, target: TargetShape
) -> NDArray:
    """Сумма ``target.value`` по подшагам.
    """
    nt, h = R.shape[:2]
    m = d_rows.shape[1]
    d_cols = np.ascontiguousarray(np.swapaxes(d_rows, 1, 2))  # (h, 3, M)
    chunk = max(1, _BATCH_ELEMENTS // max(1, h * m))
    acc = np.zeros((h, m))
    for s in range(0, nt, chunk):
        Rb = R[s : s + chunk]
        o_l = np.einsum("nhji,nhj->nhi", Rb, C - X[s : s + chunk])  # (n, h, 3)
        d_l = np.matmul(np.swapaxes(Rb, -1, -2), d_cols)  # (n, h, 3, M)
        o = o_l[..., None]  # (n, h, 3, 1)
        xy = _intersect(
            o[:, :, 0], o[:, :, 1], o[:, :, 2], d_l[:, :, 0], d_l[:, :, 1], d_l[:, :, 2]
        )
        acc += target.value(xy).sum(axis=0)
    return acc


def render(
    camera: Camera,
    pose: Pose,
    target: TargetShape,
    motion: Motion,
    t_k: float,
    tau: float,
    *,
    ss: int = 8,
    nt: int = 64,
    shutter: Literal["global", "rolling"] = "global",
    t_row: float = 0.0,
    margin_px: float = 2,
) -> RenderResult:
    """Облучённость без шума в ROI вокруг марки.

    Параметры
    ---------
    camera, pose
        Камера и её поза.
    target
        Марка (любой ``TargetShape``), ``motion`` — её движение.
    t_k, tau
        Начало и длительность экспозиции, с.
    ss, nt
        Субпикселей по каждой оси и подшагов по времени
    shutter, t_row
        ``"global"`` или ``"rolling"``. При построчном затворе экспозиция
        строки ``v`` (абсолютный номер в кадре) начинается в
        ``t_k + v · t_row``. При глобальном ``t_row`` должен быть 0.
    margin_px
        Запас вокруг марки при вычислении ROI, пикс. Если потом применяется
        PSF, запас должен быть не меньше ``SensorModel.required_margin_px``,
        иначе часть света уйдёт за ROI и центр тяжести сместится.
    """
    ss = _positive_int(ss, "ss")
    nt = _positive_int(nt, "nt")
    if shutter not in ("global", "rolling"):
        raise ValueError(
            f"shutter должен быть 'global' или 'rolling', получено {shutter!r}"
        )
    t_row = finite_scalar(t_row, "t_row")
    rolling = shutter == "rolling"
    if not rolling and t_row != 0.0:
        raise ValueError("t_row имеет смысл только при shutter='rolling'")

    samples = exposure_samples(t_k, tau, nt)
    roi, clipped = compute_roi(
        camera, pose, target, motion, t_k, tau, margin_px=margin_px, t_row=t_row
    )
    if roi is None:
        return RenderResult(
            irradiance=readonly(np.zeros((0, 0))),
            roi=None,
            ss=ss,
            nt=nt,
            times=readonly(samples),
            empty=True,
            clipped=False,
        )

    _, v0, w, h = roi
    offsets = (v0 + np.arange(h)) * t_row if rolling else np.zeros(h)
    times = samples[:, None] + offsets[None, :]  # (nt, h)
    flat = times.ravel()
    R = motion.orientation(flat).reshape(nt, h, 3, 3)
    X = motion.position(flat).reshape(nt, h, 3)

    rays_c, _ = camera.ray_grid(roi, ss)  # (h, w, ss, ss, 3), система камеры
    d_rows = (rays_c @ pose.R).reshape(h, w * ss * ss, 3)  # в мир
    acc = _accumulate(d_rows, pose.center, R, X, target)
    irradiance = acc.reshape(h, w, ss, ss).mean(axis=(2, 3)) / nt

    return RenderResult(
        irradiance=readonly(irradiance),
        roi=roi,
        ss=ss,
        nt=nt,
        times=readonly(times if rolling else samples),
        empty=False,
        clipped=clipped,
    )
