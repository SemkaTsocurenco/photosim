"""Модель погрешности наблюдения марки камерой (статья B).

Ковариация наблюдения — центра тяжести образа марки, пикс.² — для каждой
пары (точка, камера):

    Sigma_xi = kappa * M / N_e + Sigma_ecc + Sigma_sync,

``M`` — второй момент образа, ``N_e`` — сигнал марки, e⁻. Первое слагаемое —
фотонный шум центра тяжести (``analysis.centroid_sigma``; E1 и E3 статьи A
совпадают с ним до долей процента). Модель верна только для центра
тяжести — метода, который статья A рекомендует для подвижных марок.

Второй момент. Круг радиуса ``r`` в плоскости марки имеет момент
``(r**2 / 4) (I - n n^T)``. В первом порядке по ``r / Z`` проекция линейна
(``J`` — её якобиан), поэтому

    M = J (r**2 / 4) (I - n n^T) J^T + (L**2 / 12) e_v e_v^T + extra * I.

На оптической оси это эллипс с полуосями ``a = f r / Z`` и
``b = a cos(view)`` (``view`` — угол между нормалью и направлением на
камеру), малая ось — по проекции нормали на плоскость изображения:
``M0 = (a**2 / 4) e_maj e_maj^T + (b**2 / 4) e_min e_min^T``. Вне оси образ
ещё вытянут по радиусу в ``1 / cos(phi)`` раз (``phi`` — угол от
оптической оси; в углу кадра review_16mm phi = 19°, на 6 %), формула с
``J`` это учитывает. Смаз длиной ``L = |J V| tau`` вдоль направления
движения образа ``e_v = J V / |J V|`` добавляет ``L**2 / 12``.

Сигнал. Яркость марки от расстояния не зависит, заполнение пикселя
одинаково, и ``N_e`` пропорционален площади образа:
``N_e = fill * FW * pi a b``, ``pi a b = 4 pi sqrt(det M0)``. Отсюда без
смаза на оси

    sigma_maj**2 = 1 / (4 pi fill FW cos(view)),
    sigma_min**2 = cos(view) / (4 pi fill FW).

Случайная погрешность в пикселях не зависит от расстояния до камеры, а
только от формы образа (на оси — от угла визирования). Произведение
``sigma_maj sigma_min = sigma0**2`` не зависит и от неё: наклон лишь
перераспределяет дисперсию между осями. ``sigma0 = 1 / (2 sqrt(pi fill
FW))`` — марка, развёрнутая к камере (3,2e-3 пикс. при fill = 0,7 и
ёмкости review_16mm). Шум считывания и PSF: множитель ``kappa`` (при
fill >= 0,2 поправка меньше 15 %, E1) и добавка ``extra`` — ``sigma_psf**2
+ 1/12``, как в ``analysis.centroid_sigma``; по умолчанию 1 и 0.

Эксцентриситет ``eps``: центр эллипса — образ полюса
``X_e = X - (r**2 / Z) (z_hat - n_z n)`` (``analysis.eccentricity_pole``;
точно для камеры без дисторсии, с дисторсией — приближённо).
``"corrected"`` — поправка вводится по нормали, известной с ошибкой;
остаётся ``Sigma_ecc = J_n Sigma_n J_n^T``, ``J_n = d eps / d(два угла
нормали)``, ``Sigma_n = sigma_n**2 I``. Производная берётся по формуле
полюса аналитически: при повороте нормали на ``dn`` (``dn`` ⟂ ``n``)
``dX_e = (r**2 / Z) (dn_z n + n_z dn)``, поэтому

    Sigma_ecc = sigma_n**2 G (I - n n^T) G^T,
    G = (r**2 / Z) J(X_e) (n z_hat^T + n_z I).

Это точная производная того же ``eps`` (тест сверяет с центральными
разностями) и в несколько раз быстрее разностей — модель входит в целевую
функцию оптимизатора. ``"uncorrected"`` — ``eps`` в ковариацию не входит, а
возвращается смещением наблюдения ``bias``.

Рассинхронизация камер: ``Sigma_sync = (|J V| sigma_t)**2 e_v e_v^T``.

Режим ``"constant"`` для сравнения: ``Sigma_xi = sigma_c**2 I``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from photosim import presets
from photosim._checks import as_flat_points, finite_scalar
from photosim.camera import Camera, Pose

__all__ = [
    "COV_MODES",
    "ECC_MODES",
    "SIGMA_C_REVIEW",
    "ObsModel",
    "Observation",
    "observe",
    "sigma0",
]

#: Режимы ковариации: по модели или постоянная ``sigma_c**2 I``.
COV_MODES = ("model", "constant")

#: Учёт эксцентриситета, см. модуль.
ECC_MODES = ("corrected", "uncorrected")

#: Постоянная СКО наблюдения, как в обзоре, пикс.
SIGMA_C_REVIEW = 0.05


def sigma0(fill: float, full_well_e: float) -> float:
    """СКО центра тяжести марки, развёрнутой к камере, без смаза, пикс.

    ``1 / (2 sqrt(pi fill FW))``: по каждой оси, от размера образа не зависит.
    """
    return 1.0 / (2.0 * math.sqrt(math.pi * fill * full_well_e))


def _param(value: float, name: str, positive: bool) -> float:
    v = finite_scalar(value, name)
    if v < 0.0 or (positive and v == 0.0):
        sign = "больше 0" if positive else "не меньше 0"
        raise ValueError(f"{name} должно быть {sign}, получено {value!r}")
    return v


@dataclass(frozen=True)
class ObsModel:
    """Параметры модели погрешности наблюдения.

    radius
        Радиус марки, м.
    fill, full_well_e
        Заполнение ёмкости пикселя образом марки при экспозиции ``tau`` и
        ёмкость, e⁻ (по умолчанию — review_16mm).
    tau
        Экспозиция, с: длина смаза ``L = |J V| tau``.
    kappa, extra_var_px
        Поправка на шум считывания и PSF: множитель фотонного члена и
        добавка ко второму моменту образа, пикс.² (``sigma_psf**2 + 1/12``).
    eccentricity
        ``"corrected"`` или ``"uncorrected"``, см. модуль.
    sigma_n_deg
        СКО каждого из двух углов нормали, градусы (для ``"corrected"``).
    sigma_t
        СКО рассинхронизации камер, с (для построчного затвора с привязкой
        к строке — её остаток).
    mode, sigma_c
        ``"model"`` или ``"constant"``: ``Sigma_xi = sigma_c**2 I``, пикс.
    """

    radius: float = 0.005
    fill: float = 0.7
    full_well_e: float = presets.REVIEW_16MM["full_well_e"]
    tau: float = 1e-3
    kappa: float = 1.0
    extra_var_px: float = 0.0
    eccentricity: str = "corrected"
    sigma_n_deg: float = 2.0
    sigma_t: float = 10e-6
    mode: str = "model"
    sigma_c: float = SIGMA_C_REVIEW

    def __post_init__(self) -> None:
        set_ = object.__setattr__  # обход frozen, только здесь
        for name in ("radius", "fill", "full_well_e", "kappa", "sigma_c"):
            set_(self, name, _param(getattr(self, name), name, positive=True))
        for name in ("tau", "extra_var_px", "sigma_n_deg", "sigma_t"):
            set_(self, name, _param(getattr(self, name), name, positive=False))
        if self.mode not in COV_MODES:
            raise ValueError(f"mode = {self.mode!r}, допустимо: {', '.join(COV_MODES)}")
        if self.eccentricity not in ECC_MODES:
            raise ValueError(
                f"eccentricity = {self.eccentricity!r}, допустимо: {', '.join(ECC_MODES)}"
            )

    @property
    def sigma0(self) -> float:
        """СКО марки, развёрнутой к камере, без смаза и других членов, пикс."""
        return math.sqrt(self.kappa) * sigma0(self.fill, self.full_well_e)

    def as_constant(self, sigma_c: float | None = None) -> ObsModel:
        """Та же модель в режиме ``"constant"``.

        По умолчанию ``sigma_c = self.sigma0`` — уровень модели для марки,
        развёрнутой к камере: сравнение с моделью тогда честное (B3a).
        """
        return replace(
            self, mode="constant", sigma_c=self.sigma0 if sigma_c is None else sigma_c
        )


@dataclass(frozen=True, eq=False)
class Observation:
    """Наблюдения точек камерами: ведущие оси — как у точек, затем камера.

    Для точек ``(..., 3)`` и ``C`` камер: ``visible`` — ``(..., C)``,
    векторы — ``(..., C, 2)``, матрицы — ``(..., C, 2, 2)``, якобиан —
    ``(..., C, 2, 3)``. Члены ``cov_*`` считаются и в режиме ``"constant"``,
    но в ``cov`` тогда не входят.

    Невидимая пара (за камерой, марка видна сзади или ребром, образ не
    целиком в кадре) не даёт nan: в ``cov`` —
    ``inf`` на диагонали, ``weight``, ``jacobian`` и остальные величины —
    нули. Информация точки ``sum_c J^T W J`` поэтому сама пропускает
    невидимые камеры. ``uv`` — nan только там, где проекции нет.
    """

    visible: NDArray  # марка видна и образ в кадре
    uv: NDArray  # проекция центра, пикс.
    jacobian: NDArray  # d(u, v) / d(X мира), пикс./м
    cov: NDArray  # Sigma_xi, пикс.²
    weight: NDArray  # Sigma_xi**-1
    bias: NDArray  # смещение наблюдения: eps при "uncorrected", иначе 0
    cov_shot: NDArray  # kappa M / N_e
    cov_ecc: NDArray  # J_n Sigma_n J_n^T при "corrected", иначе 0
    cov_sync: NDArray  # sigma_t**2 (J V)(J V)^T
    ecc: NDArray  # эксцентриситет eps, пикс.
    cos_view: NDArray  # косинус угла между нормалью и направлением на камеру
    semi_axes_px: NDArray  # полуоси (a, b) образа без смаза
    n_e: NDArray  # сигнал марки, e⁻
    blur_px: NDArray  # длина смаза L


def _unit_rows(v: NDArray, name: str) -> NDArray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.all(norm > 0):
        raise ValueError(f"{name}: нужны ненулевые векторы")
    return v / norm


def _dot(a: NDArray, b: NDArray) -> NDArray:
    """Скалярные произведения строк ``(P, 3)``."""
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def _project(camera: Camera, Xc: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Проекция ``(P, 2)``, якобиан по координатам камеры ``(P, 2, 3)``, маска.

    То же, что ``Camera.project`` и ``Camera.jacobian`` (вне маски — nan),
    но для камеры без дисторсии в несколько раз быстрее: у них общий путь с
    дисторсией и проверки входа.
    """
    if camera.has_distortion:
        J = camera.jacobian(Xc)
        return camera.project(Xc), J, np.all(np.isfinite(J), axis=(1, 2))
    z = Xc[:, 2]
    valid = z > 0
    iz = np.divide(1.0, z, out=np.full_like(z, np.nan), where=valid)
    x, y = Xc[:, 0] * iz, Xc[:, 1] * iz
    uv = np.empty((len(z), 2))
    uv[:, 0] = camera.fx * x + camera.cx
    uv[:, 1] = camera.fy * y + camera.cy
    J = np.zeros((len(z), 2, 3))
    J[:, 0, 0] = camera.fx * iz
    J[:, 0, 2] = -camera.fx * x * iz
    J[:, 1, 1] = camera.fy * iz
    J[:, 1, 2] = -camera.fy * y * iz
    J[~valid] = np.nan
    return uv, J, valid


class _Terms(NamedTuple):
    """Величины одной камеры по компонентам, точки ``(P,)``.

    Вне ``vis`` — мусор (nan, inf): кто читает, тот маскирует. Матрицы 2x2 —
    кортежи ``(m11, m12, m22)``; в режиме ``"constant"`` ``cov`` и
    ``weight`` — числа.
    """

    vis: NDArray  # марка видна и образ в кадре
    uv: NDArray  # (P, 2)
    cos_view: NDArray
    jacobian: NDArray  # (P, 2, 3), по координатам мира
    cov: tuple
    weight: tuple
    shot: tuple
    ecc_cov: tuple
    sync: tuple
    ecc: NDArray  # (P, 2)
    n_e: NDArray
    semi_axes: tuple  # (a, b)
    blur: NDArray


def _camera_terms(
    model: ObsModel, camera: Camera, pose: Pose, X: NDArray, n: NDArray, V: NDArray
) -> _Terms:
    """Модель для одной камеры: точки, нормали, скорости в мире ``(P, 3)``.

    Входит в целевую функцию оптимизатора (``network.evaluate``), поэтому
    всё считается по компонентам на всех точках сразу: батчи крошечных
    matmul и выборки по индексам в numpy медленные. Где проекции нет, по
    дороге появляются nan и inf; маска ``vis`` отбрасывает их у читателя.
    """
    r, tau = model.radius, model.tau
    R = pose.R
    Xc = X @ R.T + pose.T
    nc, Vc = n @ R.T, V @ R.T
    uv, Jc, valid = _project(camera, Xc)
    dist = np.sqrt(_dot(Xc, Xc))
    cos_view = -_dot(nc, Xc) / np.where(dist > 0, dist, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        j1, j2 = Jc[:, 0], Jc[:, 1]
        jn1, jn2 = _dot(j1, nc), _dot(j2, nc)
        q = r**2 / 4
        m11 = q * (_dot(j1, j1) - jn1**2)
        m12 = q * (_dot(j1, j2) - jn1 * jn2)
        m22 = q * (_dot(j2, j2) - jn2**2)
        det0 = m11 * m22 - m12**2
        w1, w2 = _dot(j1, Vc), _dot(j2, Vc)  # скорость образа, пикс./с

        # образ (эллипс с полуширинами 2 sqrt(m_ii) и смаз) целиком в кадре
        ext_u = 2 * np.sqrt(m11) + tau * np.abs(w1) / 2
        ext_v = 2 * np.sqrt(m22) + tau * np.abs(w2) / 2
        vis = (
            valid
            & (dist > 0)
            & (cos_view > 0)  # видна спереди
            & (det0 > 0)  # 0 — марка ребром
            & (uv[:, 0] - ext_u >= -0.5)
            & (uv[:, 0] + ext_u <= camera.width - 0.5)
            & (uv[:, 1] - ext_v >= -0.5)
            & (uv[:, 1] + ext_v <= camera.height - 0.5)
        )

        n_e = model.fill * model.full_well_e * 4 * math.pi * np.sqrt(det0)
        half_tr = (m11 + m22) / 2
        disc = np.sqrt(np.clip(half_tr**2 - det0, 0.0, None))
        semi_axes = (
            2 * np.sqrt(half_tr + disc),
            2 * np.sqrt(np.clip(half_tr - disc, 0.0, None)),
        )
        # без батча matmul: (P, 2, 3) @ (3, 3) в 20 раз медленнее
        Jw = (Jc.reshape(-1, 3) @ R).reshape(Jc.shape)

        blur, extra, g = tau**2 / 12, model.extra_var_px, model.kappa / n_e
        shot = (
            g * (m11 + blur * w1**2 + extra),
            g * (m12 + blur * w1 * w2),
            g * (m22 + blur * w2**2 + extra),
        )
        st2 = model.sigma_t**2
        sync = (st2 * w1**2, st2 * w1 * w2, st2 * w2**2)

        # эксцентриситет: образ полюса X_e (analysis.eccentricity_pole)
        nz, s = nc[:, 2], r**2 / Xc[:, 2]
        Xe = Xc + (s * nz)[:, None] * nc
        Xe[:, 2] -= s
        uv_e, Je, _ = _project(camera, Xe)
        if model.eccentricity == "uncorrected":
            ecc_cov = (0.0, 0.0, 0.0)
        else:
            # строки G = (r**2 / Z) J(X_e) (n z_hat^T + n_z I)
            sz = (s * nz)[:, None]
            g1, g2 = sz * Je[:, 0], sz * Je[:, 1]
            g1[:, 2] += s * _dot(Je[:, 0], nc)
            g2[:, 2] += s * _dot(Je[:, 1], nc)
            gn1, gn2 = _dot(g1, nc), _dot(g2, nc)
            sn2 = math.radians(model.sigma_n_deg) ** 2
            ecc_cov = (
                sn2 * (_dot(g1, g1) - gn1**2),
                sn2 * (_dot(g1, g2) - gn1 * gn2),
                sn2 * (_dot(g2, g2) - gn2**2),
            )

        if model.mode == "model":
            cov = tuple(a + b + d for a, b, d in zip(shot, ecc_cov, sync))
        else:
            cov = (model.sigma_c**2, 0.0, model.sigma_c**2)
        c11, c12, c22 = cov
        det = c11 * c22 - c12**2
        weight = (c22 / det, -c12 / det, c11 / det)
    return _Terms(
        vis=vis,
        uv=uv,
        cos_view=cos_view,
        jacobian=Jw,
        cov=cov,
        weight=weight,
        shot=shot,
        ecc_cov=ecc_cov,
        sync=sync,
        ecc=uv_e - uv,
        n_e=n_e,
        semi_axes=semi_axes,
        blur=tau * np.hypot(w1, w2),
    )


def _put(
    a: NDArray,
    vis: NDArray,
    m11: NDArray,
    m12: NDArray,
    m22: NDArray,
    diag: float = 0.0,
) -> None:
    """Записать симметричные 2x2 ``(P, 2, 2)`` из компонент; вне ``vis`` —
    ``diag`` на диагонали и 0 вне неё."""
    a[:, 0, 0] = np.where(vis, m11, diag)
    a[:, 0, 1] = a[:, 1, 0] = np.where(vis, m12, 0.0)
    a[:, 1, 1] = np.where(vis, m22, diag)


def _store(model: ObsModel, t: _Terms, out: dict[str, NDArray]) -> None:
    """Записать величины камеры в её блоки ``out`` (``(P, ...)``) по маске."""
    vis, v1 = t.vis, t.vis[:, None]
    out["visible"][:] = vis
    out["uv"][:] = t.uv
    out["cos_view"][:] = t.cos_view
    out["jacobian"][:] = np.where(vis[:, None, None], t.jacobian, 0.0)
    _put(out["cov"], vis, *t.cov, diag=np.inf)
    _put(out["weight"], vis, *t.weight)
    _put(out["cov_shot"], vis, *t.shot)
    _put(out["cov_ecc"], vis, *t.ecc_cov)
    _put(out["cov_sync"], vis, *t.sync)
    out["ecc"][:] = np.where(v1, t.ecc, 0.0)
    if model.eccentricity == "uncorrected":
        out["bias"][:] = out["ecc"]
    out["n_e"][:] = np.where(vis, t.n_e, 0.0)
    out["semi_axes_px"][:, 0] = np.where(vis, t.semi_axes[0], 0.0)
    out["semi_axes_px"][:, 1] = np.where(vis, t.semi_axes[1], 0.0)
    out["blur_px"][:] = np.where(vis, t.blur, 0.0)


def _camera_pairs(cameras: Sequence[tuple[Camera, Pose]]) -> list[tuple[Camera, Pose]]:
    if (
        isinstance(cameras, tuple)
        and len(cameras) == 2
        and isinstance(cameras[0], Camera)
    ):
        cameras = [cameras]  # одна пара (Camera, Pose)
    pairs = list(cameras)
    if not pairs:
        raise ValueError("cameras: пустой список")
    for i, pair in enumerate(pairs):
        if not (
            len(pair) == 2 and isinstance(pair[0], Camera) and isinstance(pair[1], Pose)
        ):
            raise TypeError(
                f"cameras[{i}]: нужна пара (Camera, Pose), получено {pair!r}"
            )
    return pairs


def _prepare(
    points: ArrayLike, normals: ArrayLike, velocities: ArrayLike | None
) -> tuple[NDArray, NDArray, NDArray, tuple[int, ...]]:
    """Точки, единичные нормали и скорости ``(P, 3)`` и ведущие оси точек."""
    X, lead = as_flat_points(points, 3, "points")
    shape = X.shape
    n = np.broadcast_to(np.asarray(normals, dtype=np.float64), lead + (3,)).reshape(
        shape
    )
    n = _unit_rows(n, "normals")
    if velocities is None:
        V = np.zeros(shape)
    else:
        V = np.broadcast_to(
            np.asarray(velocities, dtype=np.float64), lead + (3,)
        ).reshape(shape)
    for name, a in (("points", X), ("normals", n), ("velocities", V)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} должны быть конечными")
    return X, n, V, lead


def observe(
    model: ObsModel,
    cameras: Sequence[tuple[Camera, Pose]] | tuple[Camera, Pose],
    points: ArrayLike,
    normals: ArrayLike,
    velocities: ArrayLike | None = None,
) -> Observation:
    """Модель наблюдения точек ``(..., 3)`` камерами ``[(Camera, Pose), ...]``.

    ``points`` — центры марок в мире, м; ``normals`` — нормали марок в мире
    (от марки в сторону, откуда её видно; нормируются здесь);
    ``velocities`` — скорости центров в мире, м/с (``None`` — неподвижны).
    Нормали и скорости транслируются к форме точек. Камеры обходятся по
    одной, точки — векторно.
    """
    X, n, V, lead = _prepare(points, normals, velocities)
    pairs = _camera_pairs(cameras)
    # память по камерам (C, P, ...): каждая камера пишет свой непрерывный
    # блок, наружу — виды (P, C, ...) без копирования
    P, C = len(X), len(pairs)
    out = {
        "visible": np.zeros((C, P), dtype=bool),
        "uv": np.empty((C, P, 2)),
        "jacobian": np.zeros((C, P, 2, 3)),
        "cov": np.full((C, P, 2, 2), np.inf),
        **{
            key: np.zeros((C, P, 2, 2))
            for key in ("weight", "cov_shot", "cov_ecc", "cov_sync")
        },
        **{key: np.zeros((C, P, 2)) for key in ("bias", "ecc", "semi_axes_px")},
        **{key: np.zeros((C, P)) for key in ("cos_view", "n_e", "blur_px")},
    }
    for c, (cam, pose) in enumerate(pairs):
        terms = _camera_terms(model, cam, pose, X, n, V)
        _store(model, terms, {k: a[c] for k, a in out.items()})
    views = {key: np.moveaxis(a, 0, 1) for key, a in out.items()}
    return Observation(**{k: v.reshape(lead + v.shape[1:]) for k, v in views.items()})
