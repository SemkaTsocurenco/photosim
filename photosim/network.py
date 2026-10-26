"""Сеть камер и поле неопределённости точек рабочего объёма (статья B).

Запись ``p`` — марка (точка, нормаль, скорость); у одной точки объёма может
быть несколько нормалей — это разные записи. Камера ``c`` видит запись,
если

- точка перед камерой и её проекция в кадре с отступом ``frame_margin``
  (доля ширины и высоты) от краёв;
- угол визирования ``view <= max_view_deg``;
- образ марки (эллипс и смаз) целиком в кадре и марка видна спереди
  (``obs_model``).

Загораживание не учитывается — допущение, его нужно указать в статье.

Ковариация точки по видимым камерам — МНК-триангуляция в первом порядке:

    I_p = sum_c J_c^T W_c J_c,   Sigma_X = I_p**-1,

``J_c = d(u, v) / d(X мира)`` (``Camera.jacobian(X_c) @ pose.R``),
``W_c = Sigma_xi**-1`` (``obs_model``). Точка с менее чем двумя видимыми
камерами (или с вырожденной ``I_p``: камеры и точка на одной прямой) не
измеряется: ``Sigma_X``, ``delta_X`` и все метрики — ``inf``.

Смещение от неисправленного эксцентриситета (``eccentricity="uncorrected"``,
``eps_c`` — смещение наблюдения):

    delta_X = Sigma_X sum_c J_c^T W_c eps_c,

полная погрешность ``err_total = sqrt(lambda_max(Sigma_X) + |delta_X|**2)``.

``evaluate`` — целевая функция оптимизатора: 2000 записей и 6 камер — не
дольше 10 мс. Поэтому она не собирает наблюдения ``(P, C, ...)`` (это
``observe``), а копит ``I_p`` по камерам: большие временные массивы на
каждом вызове стоили бы страничных отказов (~4 мс).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from photosim._checks import readonly
from photosim.camera import Camera, Pose
from photosim.obs_model import ObsModel, _camera_pairs, _camera_terms, _prepare

__all__ = [
    "FRAME_MARGIN",
    "MAX_VIEW_DEG",
    "CameraNetwork",
    "NetworkResult",
    "TargetSet",
    "evaluate",
]

#: Наибольший угол визирования, градусы.
MAX_VIEW_DEG = 70.0

#: Отступ проекции от краёв кадра, доля ширины и высоты.
FRAME_MARGIN = 0.05

#: Порог вырожденности ``I_p``: ``det <= RCOND * (trace / 3)**3``.
_RCOND = 1e-12


@dataclass(frozen=True, eq=False)
class CameraNetwork:
    """Камеры сети: внутренние параметры и позы, по одной на камеру."""

    cameras: Sequence[Camera]
    poses: Sequence[Pose]

    def __post_init__(self) -> None:
        set_ = object.__setattr__  # обход frozen, только здесь
        set_(self, "cameras", tuple(self.cameras))
        set_(self, "poses", tuple(self.poses))
        if len(self.cameras) != len(self.poses):
            raise ValueError(
                f"камер {len(self.cameras)}, а поз {len(self.poses)}: нужно поровну"
            )
        _camera_pairs(self.pairs)  # не пусто, типы

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[Camera, Pose]]) -> CameraNetwork:
        return cls([c for c, _ in pairs], [p for _, p in pairs])

    @property
    def pairs(self) -> list[tuple[Camera, Pose]]:
        return list(zip(self.cameras, self.poses, strict=True))

    def __len__(self) -> int:
        return len(self.cameras)

    def add(self, camera: Camera, pose: Pose) -> CameraNetwork:
        """Сеть с ещё одной камерой (последней)."""
        return CameraNetwork((*self.cameras, camera), (*self.poses, pose))


@dataclass(frozen=True, eq=False)
class TargetSet:
    """Записи рабочего объёма: точки, нормали и скорости в мире ``(P, 3)``.

    Нормали нормируются; нормаль или скорость одна на все записи
    транслируется; ``velocities=None`` — марки неподвижны.
    """

    points: ArrayLike
    normals: ArrayLike
    velocities: ArrayLike | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.points, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"points: нужна форма (P, 3), а пришла {X.shape}")
        X, n, V, _ = _prepare(X, self.normals, self.velocities)
        set_ = object.__setattr__  # обход frozen, только здесь
        for name, a in (("points", X), ("normals", n), ("velocities", V)):
            set_(self, name, readonly(np.array(a)))

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True, eq=False)
class NetworkResult:
    """Поле неопределённости: по записи ``(P, ...)``, видимость ``(P, C)``.

    Ковариация и смещение — в метрах, метрики — в микрометрах. У
    неизмеряемой записи (``measured = False``) ``cov``, ``delta`` и метрики —
    ``inf``, ``info`` — как есть.
    """

    visible: NDArray  # (P, C): камера видит запись по правилам сети
    n_visible: NDArray  # (P,)
    measured: NDArray  # (P,): >= 2 камер и I_p невырождена
    info: NDArray  # (P, 3, 3): I_p, 1/м²
    cov: NDArray  # (P, 3, 3): Sigma_X, м²
    delta: NDArray  # (P, 3): delta_X, м (0 при "corrected")
    sigma_max_um: NDArray  # sqrt(lambda_max(Sigma_X))
    sigma_trace_um: NDArray  # sqrt(trace(Sigma_X))
    bias_um: NDArray  # |delta_X|
    err_total_um: NDArray  # sqrt(lambda_max + |delta_X|**2)


#: Компоненты симметричной 3x3 по порядку: (00, 01, 02, 11, 12, 22).
_SYM = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _sym3(m: NDArray) -> NDArray:
    """Матрицы ``(P, 3, 3)`` из компонент ``(6, P)`` (порядок ``_SYM``)."""
    out = np.empty((m.shape[1], 3, 3))
    for k, (i, j) in enumerate(_SYM):
        out[:, i, j] = out[:, j, i] = m[k]
    return out


def _inverse_sym3(m: NDArray) -> tuple[NDArray, NDArray]:
    """Обратные к симметричным 3x3 по компонентам ``(6, P)`` и ``det``.

    Через присоединённую матрицу; где ``det = 0``, — inf и nan.
    """
    a, b, c, d, e, f = m
    adj = np.stack(
        [
            d * f - e * e,
            c * e - b * f,
            b * e - c * d,
            a * f - c * c,
            b * c - a * e,
            a * d - b * b,
        ]
    )
    det = a * adj[0] + b * adj[1] + c * adj[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return adj / det, det


def _lambda_max_sym3(m: NDArray) -> NDArray:
    """Наибольшее собственное число симметричных 3x3 по компонентам ``(6, P)``.

    Тригонометрическая формула (Смит, 1961): для наибольшего числа
    устойчива, в отличие от наименьшего; ``eigvalsh`` в 5 раз медленнее.
    """
    a, b, c, d, e, f = m
    q = (a + d + f) / 3
    p1 = b * b + c * c + e * e
    a, d, f = a - q, d - q, f - q
    p = np.sqrt((a * a + d * d + f * f + 2 * p1) / 6)
    det_b = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.clip(det_b / (2 * p**3), -1.0, 1.0)
    phi = np.arccos(np.where(p > 0, r, 1.0)) / 3
    return q + 2 * p * np.cos(phi)


def evaluate(
    network: CameraNetwork,
    targets: TargetSet,
    model: ObsModel,
    *,
    max_view_deg: float = MAX_VIEW_DEG,
    frame_margin: float = FRAME_MARGIN,
) -> NetworkResult:
    """Поле неопределённости записей ``targets`` для сети ``network``.

    ``I_p`` и ``sum J^T W eps`` копятся по камерам компонентами ``(P,)``:
    большие временные массивы на каждом вызове стоили бы страничных
    отказов.
    """
    if not 0.0 < max_view_deg <= 90.0:
        raise ValueError(f"max_view_deg должно быть в (0, 90], получено {max_view_deg}")
    if not 0.0 <= frame_margin < 0.5:
        raise ValueError(
            f"frame_margin должно быть в [0, 0,5), получено {frame_margin}"
        )
    X, n, V = targets.points, targets.normals, targets.velocities
    P, C = len(targets), len(network)
    cos_max = math.cos(math.radians(max_view_deg))
    uncorrected = model.eccentricity == "uncorrected"

    visible = np.zeros((C, P), dtype=bool)
    info = np.zeros((6, P))  # компоненты I_p, порядок _SYM
    g = np.zeros((3, P))  # sum_c J^T W eps
    for c, (cam, pose) in enumerate(network.pairs):
        t = _camera_terms(model, cam, pose, X, n, V)
        mu, mv = frame_margin * cam.width, frame_margin * cam.height
        with np.errstate(invalid="ignore"):  # uv = nan за камерой
            vis = (
                t.vis
                & (t.cos_view >= cos_max)
                & (t.uv[:, 0] >= mu - 0.5)
                & (t.uv[:, 0] <= cam.width - 0.5 - mu)
                & (t.uv[:, 1] >= mv - 0.5)
                & (t.uv[:, 1] <= cam.height - 0.5 - mv)
            )
        visible[c] = vis
        w11, w12, w22 = (np.where(vis, w, 0.0)[:, None] for w in t.weight)
        J = np.where(vis[:, None, None], t.jacobian, 0.0)
        j1, j2 = J[:, 0], J[:, 1]
        a1, a2 = w11 * j1 + w12 * j2, w12 * j1 + w22 * j2  # строки W J
        for k, (i, j) in enumerate(_SYM):
            info[k] += j1[:, i] * a1[:, j] + j2[:, i] * a2[:, j]
        if uncorrected:
            e1, e2 = np.where(vis, t.ecc[:, 0], 0.0), np.where(vis, t.ecc[:, 1], 0.0)
            g += a1.T * e1 + a2.T * e2  # J^T W eps

    n_visible = visible.sum(axis=0)
    inv, det = _inverse_sym3(info)
    trace = info[0] + info[3] + info[5]
    measured = (n_visible >= 2) & (det > _RCOND * (trace / 3) ** 3)
    inv = np.where(measured, inv, 0.0)  # без inf и nan у неизмеряемых
    s00, s01, s02, s11, s12, s22 = inv
    delta = np.stack(
        [
            s00 * g[0] + s01 * g[1] + s02 * g[2],
            s01 * g[0] + s11 * g[1] + s12 * g[2],
            s02 * g[0] + s12 * g[1] + s22 * g[2],
        ],
        axis=1,
    )
    delta[~measured] = np.inf
    lam = np.where(measured, _lambda_max_sym3(inv), np.inf)
    cov = _sym3(inv)
    cov[~measured] = np.diag([np.inf] * 3)
    bias2 = np.sum(delta**2, axis=1)
    return NetworkResult(
        visible=np.moveaxis(visible, 0, 1),
        n_visible=n_visible,
        measured=measured,
        info=_sym3(info),
        cov=cov,
        delta=delta,
        sigma_max_um=np.sqrt(lam) * 1e6,
        sigma_trace_um=np.sqrt(np.where(measured, s00 + s11 + s22, np.inf)) * 1e6,
        bias_um=np.sqrt(bias2) * 1e6,
        err_total_um=np.sqrt(lam + bias2) * 1e6,
    )
