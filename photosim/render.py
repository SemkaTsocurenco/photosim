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
from scipy.special import ndtr

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

# Предел шагов уточнения ROI по строкам при построчном затворе.
_ROI_MAX_ITER = 10


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
    psf_sigma_px : float
        СКО PSF, применённой до усреднения по пикселю; 0 — без PSF.
    """

    irradiance: NDArray
    roi: tuple[int, int, int, int] | None
    ss: int
    nt: int
    times: NDArray
    empty: bool
    clipped: bool
    psf_sigma_px: float = 0.0


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

    При построчном затворе (``t_row != 0``) строка ``v`` экспонируется в
    ``[t_k + v t_row, t_k + v t_row + tau]``. Сначала берётся время всего
    кадра, затем — только время строк найденной ROI, пока строки не
    перестанут меняться. Каждый шаг даёт ROI, содержащую освещённую область:
    свет в строках ROI собирается только во время этих строк. Без этого ROI
    быстрой марки тянулась бы по всему пути за считывание кадра, а марка
    лежала бы у одного её края.
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
    t_row = finite_scalar(t_row, "t_row")
    contour = target.boundary(n_boundary)

    rows = (0, camera.height - 1)
    for _ in range(_ROI_MAX_ITER):
        lo, hi = sorted((rows[0] * t_row, rows[1] * t_row))
        t_lo, t_hi = tk + lo, tk + ta + hi
        times = np.concatenate(
            [[t_lo], exposure_samples(t_lo, t_hi - t_lo, n), [t_hi]]
        )
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
        clipped = (
            u0 < 0 or v0 < 0 or u1 > camera.width - 1 or v1 > camera.height - 1
        )
        u0, v0 = max(u0, 0), max(v0, 0)
        u1, v1 = min(u1, camera.width - 1), min(v1, camera.height - 1)
        if u0 > u1 or v0 > v1:
            return None, False
        if t_row == 0.0 or (v0, v1) == rows:
            break
        rows = (v0, v1)
    return (u0, v0, u1 - u0 + 1, v1 - v0 + 1), clipped


def _accumulate(
    rays_c: NDArray,
    pose_R: NDArray,
    C: NDArray,
    R: NDArray,
    X: NDArray,
    target: TargetShape,
) -> NDArray:
    """Сумма ``target.value`` по подшагам; ``rays_c`` — ``(h, M, 3)`` в камере.

    Пачки и по строкам ROI, и по времени: при больших ``ss`` (64: сотни тысяч
    лучей в строке ROI) массивы остаются порядка ``_BATCH_ELEMENTS``.
    """
    nt, h = R.shape[:2]
    m = rays_c.shape[1]
    rows = max(1, _BATCH_ELEMENTS // max(1, m))
    acc = np.zeros((h, m))
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        # в мир (= pose.Rᵀ d_c) и по компонентам: (rows, 3, M)
        d_cols = np.ascontiguousarray(np.swapaxes(rays_c[r0:r1] @ pose_R, 1, 2))
        chunk = max(1, _BATCH_ELEMENTS // ((r1 - r0) * m))
        for s in range(0, nt, chunk):
            Rb = R[s : s + chunk, r0:r1]
            o_l = np.einsum("nhji,nhj->nhi", Rb, C - X[s : s + chunk, r0:r1])
            d_l = np.matmul(np.swapaxes(Rb, -1, -2), d_cols)  # (n, rows, 3, M)
            o = o_l[..., None]  # (n, rows, 3, 1)
            xy = _intersect(
                o[:, :, 0],
                o[:, :, 1],
                o[:, :, 2],
                d_l[:, :, 0],
                d_l[:, :, 1],
                d_l[:, :, 2],
            )
            acc[r0:r1] += target.value(xy).sum(axis=0)
    return acc


def _pixel_weights(n: int, ss: int, sigma: float) -> NDArray:
    """Веса субпикселей в пиксели с гауссовой PSF до интегрирования, ``(n, n, ss)``.

    Субпиксель ``s`` (центр ``x_s``, площадь ``1/ss`` пикселя) размывается
    гауссианой ``sigma`` и попадает в пиксель ``j`` долей
    ``Φ((j + 1/2 - x_s)/σ) - Φ((j - 1/2 - x_s)/σ)``. Так PSF применяется до
    интегрирования по пикселю, как в оптике, и интеграл гауссианы по пикселю
    берётся точно. Строки нормируются на 1: свет, ушедший за край ROI,
    заменяется светом из-за края, где фон (это обеспечивает запас ROI).
    """
    x = (np.arange(n * ss) + 0.5) / ss - 0.5  # центры субпикселей, пикс.
    j = np.arange(n)[:, None]
    W = ndtr((j + 0.5 - x) / sigma) - ndtr((j - 0.5 - x) / sigma)
    W /= W.sum(axis=1, keepdims=True)
    return W.reshape(n, n, ss)


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
    psf_sigma_px: float = 0.0,
    roi: tuple[int, int, int, int] | None = None,
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
        Запас вокруг марки при вычислении ROI, пикс. При PSF запас должен
        быть не меньше ``SensorModel.required_margin_px``, иначе часть
        света уйдёт за ROI и центр тяжести сместится.
    psf_sigma_px
        СКО гауссовой PSF, пикс.: размывается субпиксельное изображение, и
        только потом оно усредняется по пикселю (``_pixel_weights``). Так
        PSF подавляет зависимость центра тяжести от фазы края; дискретная
        свёртка уже пиксельного изображения (этап ``psf`` сенсора) этого не
        делает. 0 — без PSF.
    roi
        Задать ROI ``(u0, v0, w, h)`` явно вместо ``compute_roi`` (например,
        чтобы сравнить кадры при малом сдвиге марки); ``clipped`` тогда
        ``False``.
    """
    ss = _positive_int(ss, "ss")
    nt = _positive_int(nt, "nt")
    sigma = finite_scalar(psf_sigma_px, "psf_sigma_px")
    if sigma < 0:
        raise ValueError(
            f"psf_sigma_px не может быть отрицательным, получено {psf_sigma_px!r}"
        )
    if shutter not in ("global", "rolling"):
        raise ValueError(
            f"shutter должен быть 'global' или 'rolling', получено {shutter!r}"
        )
    t_row = finite_scalar(t_row, "t_row")
    rolling = shutter == "rolling"
    if not rolling and t_row != 0.0:
        raise ValueError("t_row имеет смысл только при shutter='rolling'")

    samples = exposure_samples(t_k, tau, nt)
    if roi is None:
        roi, clipped = compute_roi(
            camera, pose, target, motion, t_k, tau, margin_px=margin_px, t_row=t_row
        )
    else:
        roi, clipped = tuple(int(a) for a in roi), False
    if roi is None:
        return RenderResult(
            irradiance=readonly(np.zeros((0, 0))),
            roi=None,
            ss=ss,
            nt=nt,
            times=readonly(samples),
            empty=True,
            clipped=False,
            psf_sigma_px=sigma,
        )

    _, v0, w, h = roi
    offsets = (v0 + np.arange(h)) * t_row if rolling else np.zeros(h)
    times = samples[:, None] + offsets[None, :]  # (nt, h)
    flat = times.ravel()
    R = motion.orientation(flat).reshape(nt, h, 3, 3)
    X = motion.position(flat).reshape(nt, h, 3)

    rays_c, _ = camera.ray_grid(roi, ss)  # (h, w, ss, ss, 3), система камеры
    acc = _accumulate(
        rays_c.reshape(h, w * ss * ss, 3), pose.R, pose.center, R, X, target
    )
    sub = acc.reshape(h, w, ss, ss)  # оси: строка, столбец, субстрока, субстолбец
    if sigma == 0.0:
        irradiance = sub.mean(axis=(2, 3)) / nt
    else:
        Wv, Wu = _pixel_weights(h, ss, sigma), _pixel_weights(w, ss, sigma)
        irradiance = np.einsum("iIa,IJab,jJb->ij", Wv, sub, Wu, optimize=True) / nt

    return RenderResult(
        irradiance=readonly(irradiance),
        roi=roi,
        ss=ss,
        nt=nt,
        times=readonly(times if rolling else samples),
        empty=False,
        clipped=clipped,
        psf_sigma_px=sigma,
    )
