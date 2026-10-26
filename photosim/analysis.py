"""Теоретические оценки для экспериментов и обработка их результатов.

Формулы здесь — приближения, с которыми сравнивается симулятор; точные
эталоны — в ``oracle``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation as _SciRotation

from photosim.camera import Camera, Pose
from photosim.oracle import conic_area, conic_center

__all__ = [
    "CLIP_VARIANCE",
    "along_across",
    "background_variance",
    "centroid_sigma",
    "centroid_signal_for_sigma",
    "eccentricity_first_order",
    "eccentricity_formula3",
    "eccentricity_pole",
    "eccentricity_tilt_sensitivity",
    "fit_variance_quadratic",
    "interp_log",
    "log_crossing",
    "method_own_bias",
    "point_at_pixel",
    "roi_s2",
    "rotation_offset_coefficients",
    "std_along_across",
]


def point_at_pixel(camera: Camera, u: float, v: float, z: float) -> tuple[float, ...]:
    """Точка на глубине ``z`` (система камеры), проецирующаяся в пиксель (u, v)."""
    ray = camera.backproject([u, v])
    return tuple(float(c) for c in ray / ray[2] * z)


def eccentricity_formula3(
    f_px: ArrayLike, radius: ArrayLike, normal_cam: ArrayLike, z: ArrayLike
) -> NDArray:
    """Эксцентриситет по формуле (3) обзора, пикс., форма ``(..., 2)``.

    ``eps = f * r**2 * n_z * (n_x, n_y) / Z**2``: ``f`` в пикселях, радиус
    марки ``r`` и глубина центра ``Z`` в метрах, ``n`` — единичная нормаль в
    системе камеры (знак не важен: произведение компонент от него не
    зависит). Первый порядок по ``r / Z``; точное значение —
    ``oracle.conic_center`` минус проекция центра.
    """
    n = np.asarray(normal_cam, dtype=np.float64)
    scale = np.asarray(f_px) * np.asarray(radius) ** 2 / np.asarray(z) ** 2
    return (scale * n[..., 2])[..., None] * n[..., :2]


def eccentricity_pole(
    camera: Camera, center_cam: ArrayLike, normal_cam: ArrayLike, radius: ArrayLike
) -> NDArray:
    """Точный эксцентриситет, пикс., форма ``(..., 2)``: замкнутая форма.

    Центр эллипса-образа — образ полюса прямой ``L`` (пересечение плоскости
    марки с плоскостью ``z_cam = 0``) относительно окружности марки: проекция
    сохраняет полярность и переводит ``L`` в бесконечно удалённую прямую
    изображения. Отсюда без приближений

        X_e = C - (r**2 / Z) * (z_hat - n_z * n),   центр = project(X_e),

    ``C`` — центр марки в системе камеры, ``Z = C_z``, ``n`` — единичная
    нормаль (знак не важен). Совпадает с ``oracle.conic_center`` до 1e-11
    пикс. Камера без дисторсии.
    """
    C = np.asarray(center_cam, dtype=np.float64)
    n = np.asarray(normal_cam, dtype=np.float64)
    r2_z = (np.asarray(radius, dtype=np.float64) ** 2 / C[..., 2])[..., None]
    shift = n[..., 2:3] * n
    shift[..., 2] -= 1.0  # n_z n - z_hat
    return camera.project(C + r2_z * shift) - camera.project(C)


def eccentricity_first_order(
    f_px: ArrayLike, radius: ArrayLike, normal_cam: ArrayLike, center_cam: ArrayLike
) -> NDArray:
    """Эксцентриситет в первом порядке по ``r**2``, пикс., ``(..., 2)``.

    ``eps = f r**2 / Z**2 * [n_z (n_x, n_y) + (1 - n_z**2) (X, Y) / Z]``.
    Первое слагаемое — формула (3) обзора, второе (член вне оси) она теряет;
    на оптической оси оно равно нулю. Относительная ошибка — порядка
    ``(r / Z)**2`` (2e-5 при r = 5 мм, Z = 1 м).
    """
    n = np.asarray(normal_cam, dtype=np.float64)
    C = np.asarray(center_cam, dtype=np.float64)
    z = C[..., 2:3]
    scale = np.asarray(f_px) * np.asarray(radius) ** 2 / z[..., 0] ** 2
    off_axis = (1 - n[..., 2:3] ** 2) * C[..., :2] / z
    return scale[..., None] * (n[..., 2:3] * n[..., :2] + off_axis)


def eccentricity_tilt_sensitivity(
    f_px: ArrayLike, radius: ArrayLike, theta: ArrayLike, z: ArrayLike
) -> NDArray:
    """``d|eps| / d theta`` на оптической оси, пикс./рад.

    ``f r**2 cos(2 theta) / Z**2``.

    На оси ``|eps| = f r**2 sin(2 theta) / (2 Z**2)``: при поправке
    эксцентриситета по известной позе ошибка наклона сказывается сильнее
    всего при малых наклонах и не сказывается при 45°.
    """
    return (
        np.asarray(f_px)
        * np.asarray(radius) ** 2
        * np.cos(2 * np.asarray(theta))
        / np.asarray(z) ** 2
    )


#: Доля дисперсии фона, остающаяся у центра тяжести после ``max(I - bg, 0)``
#: для нормального шума: ``1/2 - 1/(2 pi)`` (E1).
CLIP_VARIANCE = 0.5 - 1 / (2 * math.pi)


def background_variance(read_noise_e: ArrayLike, gain_e_per_dn: ArrayLike) -> NDArray:
    """Дисперсия фона в пикселе без сигнала, e⁻²: считывание и квантование."""
    return np.asarray(read_noise_e) ** 2 + np.asarray(gain_e_per_dn) ** 2 / 12


def roi_s2(u0: ArrayLike, w: ArrayLike, h: ArrayLike, u_c: ArrayLike) -> NDArray:
    """``sum((u - u_c)**2)`` по всем пикселям ROI ``(u0, w, h)``, пикс.²."""
    w = np.asarray(w, dtype=np.float64)
    mid = np.asarray(u0) + (w - 1) / 2
    return np.asarray(h) * w * ((mid - np.asarray(u_c)) ** 2 + (w**2 - 1) / 12)


def centroid_sigma(
    radius_px: ArrayLike,
    n_e: ArrayLike,
    var_bg_e: ArrayLike = 0.0,
    s2: ArrayLike = 0.0,
    extra_var_px: ArrayLike = 0.0,
) -> NDArray:
    """СКО центра тяжести по одной оси, пикс.

    ``sigma**2 = (R**2 / 4 + extra) / N_e + var_bg * S2 / N_e**2``.

    Первое слагаемое — фотонный шум однородного диска радиуса ``R`` с
    ``N_e`` электронами, второе — шум фона (``background_variance``) по
    всем пикселям ROI, ``S2 = roi_s2(...)``. ``extra_var_px`` добавляет к
    второму моменту диска размытие: PSF даёт ``sigma_psf**2``, отнесение
    света к центру пикселя — ``1/12``.
    """
    n = np.asarray(n_e, dtype=np.float64)
    moment = np.asarray(radius_px) ** 2 / 4 + np.asarray(extra_var_px)
    return np.sqrt(moment / n + np.asarray(var_bg_e) * np.asarray(s2) / n**2)


def centroid_signal_for_sigma(
    sigma_px: ArrayLike,
    radius_px: ArrayLike,
    var_bg_e: ArrayLike = 0.0,
    s2: ArrayLike = 0.0,
    extra_var_px: ArrayLike = 0.0,
) -> NDArray:
    """Сколько электронов нужно, чтобы ``centroid_sigma`` стало ``sigma_px``.

    Положительный корень ``sigma**2 N**2 - a N - b = 0``, где
    ``a = R**2 / 4 + extra``, ``b = var_bg * S2``.
    """
    s = np.asarray(sigma_px, dtype=np.float64) ** 2
    a = np.asarray(radius_px) ** 2 / 4 + np.asarray(extra_var_px)
    b = np.asarray(var_bg_e) * np.asarray(s2)
    return (a + np.sqrt(a**2 + 4 * s * b)) / (2 * s)


def log_crossing(x: ArrayLike, y: ArrayLike, level: float) -> tuple[float, bool]:
    """``x``, при котором ``y = level``, линейно в логарифмах обеих осей.

    ``x`` по возрастанию, ``y`` монотонна. Если ``level`` вне диапазона
    ``y``, продолжается крайний отрезок с той стороны; второй элемент
    результата — ``True`` (экстраполяция).
    """
    lx = np.log(np.asarray(x, dtype=np.float64))
    ly = np.log(np.asarray(y, dtype=np.float64))
    if lx.size < 2 or not (np.all(np.isfinite(lx)) and np.all(np.isfinite(ly))):
        return math.nan, False
    target = math.log(level)
    d = ly - target
    inside = np.flatnonzero(d[:-1] * d[1:] <= 0)
    if inside.size:
        i, extrapolated = int(inside[0]), False
    else:
        # level за первым концом, если y удаляется от него в другую сторону
        beyond_first = (target - ly[0]) * (ly[0] - ly[-1]) > 0
        i, extrapolated = (0 if beyond_first else lx.size - 2), True
    t = (target - ly[i]) / (ly[i + 1] - ly[i])
    return float(math.exp(lx[i] + t * (lx[i + 1] - lx[i]))), extrapolated


def along_across(
    du: ArrayLike, dv: ArrayLike, angle: ArrayLike
) -> tuple[NDArray, NDArray]:
    """Вектор ``(du, dv)`` в осях направления ``angle`` (рад от оси u к v).

    ``along = du cos + dv sin``, ``across = -du sin + dv cos``.
    """
    c, s = np.cos(angle), np.sin(angle)
    du, dv = np.asarray(du), np.asarray(dv)
    return du * c + dv * s, -du * s + dv * c


def std_along_across(
    std_u: ArrayLike, std_v: ArrayLike, corr_uv: ArrayLike, angle: ArrayLike
) -> tuple[NDArray, NDArray]:
    """СКО вдоль и поперёк направления ``angle`` по СКО u, v и их корреляции."""
    c, s = np.cos(angle), np.sin(angle)
    su, sv = np.asarray(std_u), np.asarray(std_v)
    cov = np.asarray(corr_uv) * su * sv
    along = su**2 * c**2 + sv**2 * s**2 + 2 * cov * c * s
    across = su**2 * s**2 + sv**2 * c**2 - 2 * cov * c * s
    return np.sqrt(along), np.sqrt(across)


def fit_variance_quadratic(x: ArrayLike, s: ArrayLike) -> tuple[float, float, float]:
    """Подгонка ``s**2 = s0**2 + k**2 x**2`` (МНК по ``s**2``, веса ``1/s**4``).

    Модель СКО от длины смаза ``L = x`` (E3, модель Sigma_xi статьи B):

        s(L)**2 = s0**2 + k**2 L**2 = s0**2 (1 + alpha (L / R)**2),
        alpha = (k R / s0)**2;

    у центра тяжести ``k**2 = 1 / (12 N_e)`` вдоль движения, 0 поперёк.

    Возвращает ``(s0, k, rms)``: ``rms`` — СКО относительного остатка
    ``s_fit / s - 1``. Если подгонка даёт ``k**2 < 0`` (рост не виден за
    шумом), ``k`` — со знаком минус: ``-sqrt(-k**2)``. Где модель не
    подходит, берётся таблица по ``L`` (``interp_log``).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(s, dtype=np.float64) ** 2
    A = np.column_stack([np.ones_like(x), x**2]) / y[:, None]
    (a, b), *_ = np.linalg.lstsq(A, np.ones_like(y), rcond=None)
    fit = np.sqrt(np.clip(a + b * x**2, 0.0, None))
    rms = float(np.sqrt(np.mean((fit / np.sqrt(y) - 1) ** 2)))
    return math.sqrt(max(a, 0.0)), math.copysign(math.sqrt(abs(b)), b), rms


def interp_log(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> NDArray:
    """Табличная интерполяция положительной величины: ``exp(np.interp(x, xp, ln fp))``.

    Линейно по ``x`` (``xp`` по возрастанию, может начинаться с 0), в
    логарифме по ``f``; за краями таблицы — крайнее значение. Так задаётся
    СКО от длины смаза ``s(L)`` там, где ``fit_variance_quadratic`` не
    подходит (эллипс в E3).
    """
    return np.exp(np.interp(x, xp, np.log(np.asarray(fp, dtype=np.float64))))


#: Шаг конечных разностей по углу в ``rotation_offset_coefficients``, рад.
ROTATION_FD_STEP = 1e-3


def rotation_offset_coefficients(
    camera: Camera,
    pose: Pose,
    center_w: ArrayLike,
    R_mid: ArrayLike,
    axis: ArrayLike,
    radius: float,
) -> tuple[NDArray, NDArray]:
    """Смещение центра тяжести от поворота за экспозицию во втором порядке.

    Марка поворачивается вокруг мировой оси ``axis`` равномерно на
    ``dtheta`` за экспозицию, ``R_mid`` — ориентация в её середине. Центр
    тяжести — среднее центров эллипса ``c`` с весом площади ``A``; до
    ``dtheta**2``

        c_bar - c_mid = (c'' / 24 + A' c' / (12 A)) dtheta**2.

    Возвращает ``(k_curv, k_area)``, пикс./рад², форма ``(2,)``: смещение —
    ``(k_curv + k_area) * dtheta**2``. Производные — центральные разности
    ``oracle.conic_center`` и ``oracle.conic_area`` с шагом
    ``ROTATION_FD_STEP``. Точный эталон — ``oracle.exposure_conic_center``.
    """
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    R0 = np.asarray(R_mid, dtype=np.float64)
    h = ROTATION_FD_STEP

    def at(angle: float) -> tuple[NDArray, float]:
        R = _SciRotation.from_rotvec(a * angle).as_matrix() @ R0
        return (
            conic_center(camera, pose, center_w, R, radius),
            conic_area(camera, pose, center_w, R, radius),
        )

    (cm, am), (c0, a0), (cp, ap) = at(-h), at(0.0), at(h)
    d1 = (cp - cm) / (2 * h)
    d2 = (cp - 2 * c0 + cm) / h**2
    area1 = (ap - am) / (2 * h)
    return d2 / 24, area1 * d1 / (12 * a0)


def method_own_bias(
    e0: pd.DataFrame, radius_px: float, psf: bool = True
) -> dict[str, float]:
    """Собственное смещение методов из таблицы E0, пикс.

    ``max |err0_u|`` по фазе с PSF (``psf=False`` — без неё) при радиусе
    образа, ближайшем к ``radius_px``.
    """
    d = e0[(e0["status"] == "ok") & (e0["effects.psf"].astype(bool) == psf)]
    radii = d["radius_px"].unique()
    r = radii[np.argmin(np.abs(radii - radius_px))]
    d = d[d["radius_px"] == r]
    return {
        m: float(d.loc[d["method"] == m, "err0_u_px"].abs().max())
        for m in sorted(d["method"].unique())
    }
