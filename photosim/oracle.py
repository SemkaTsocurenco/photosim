"""Эталоны: точные значения без пикселей и подгонки.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from photosim._checks import as_vec3, check_rotation, finite_scalar
from photosim.camera import Camera, Pose
from photosim.scene import Motion, exposure_samples

__all__ = [
    "conic_area",
    "conic_center",
    "exposure_conic_center",
    "mean_projected_center",
]

_C0 = np.diag([1.0, 1.0, -1.0])  # единичная окружность в плоскости марки


def _image_conic(
    camera: Camera, pose: Pose, center_w: ArrayLike, R_obj: ArrayLike, radius: float
) -> NDArray:
    if camera.has_distortion:
        raise NotImplementedError(
            "conic_center работает только без дисторсии"
        )
    c = as_vec3(center_w, "center_w")
    R = check_rotation(R_obj, "R_obj")
    r = finite_scalar(radius, "radius")
    if r <= 0.0:
        raise ValueError(f"radius должен быть больше 0, получено {radius!r}")

    X_c = pose.to_camera(c)
    e1_c = pose.R @ R[:, 0]
    e2_c = pose.R @ R[:, 1]
    Hm = camera.K @ np.column_stack([r * e1_c, r * e2_c, X_c])

    scale = float(np.prod(np.linalg.norm(Hm, axis=0)))
    if abs(float(np.linalg.det(Hm))) <= 1e-12 * scale:
        raise ValueError("центр камеры лежит в плоскости марки: образ вырожден")
    Hi = np.linalg.inv(Hm)
    C = Hi.T @ _C0 @ Hi
    A = C[:2, :2]
    if not np.linalg.det(A) > 0:
        raise ValueError(
            "образ окружности не эллипс: часть марки лежит позади камеры"
        )
    return -C if A[0, 0] < 0 else C


def conic_center(
    camera: Camera, pose: Pose, center_w: ArrayLike, R_obj: ArrayLike, radius: float
) -> NDArray:
    """Точный центр эллипса — проекции круглой марки, в пикселях.
    """
    C = _image_conic(camera, pose, center_w, R_obj, radius)
    return np.linalg.solve(C[:2, :2], -C[:2, 2])


def conic_area(
    camera: Camera, pose: Pose, center_w: ArrayLike, R_obj: ArrayLike, radius: float
) -> float:
    """Площадь эллипса-проекции
    """
    C = _image_conic(camera, pose, center_w, R_obj, radius)
    A, b, c = C[:2, :2], C[:2, 2], C[2, 2]
    p0 = np.linalg.solve(A, -b)
    k = c + b @ p0
    return math.pi * float(-k) / math.sqrt(float(np.linalg.det(A)))


def mean_projected_center(
    camera: Camera, pose: Pose, motion: Motion, t_k: float, tau: float, nt: int
) -> NDArray:
    """Среднее по подшагам экспозиции от проекции центра марки, ``(2,)``.
    """
    ts = exposure_samples(t_k, tau, nt)
    uv, ok = camera.project(pose.to_camera(motion.position(ts)), return_mask=True)
    if not np.all(ok):
        raise ValueError("центр марки не проецируется в части подшагов")
    return uv.mean(axis=0)


def exposure_conic_center(
    camera: Camera,
    pose: Pose,
    motion: Motion,
    radius: float,
    t_k: float,
    tau: float,
    nt: int,
) -> NDArray:
    """Точный центр тяжести смазанного изображения круглой марки
    """
    ts = exposure_samples(t_k, tau, nt)
    X = motion.position(ts)
    R = motion.orientation(ts)
    centers = np.array(
        [conic_center(camera, pose, X[i], R[i], radius) for i in range(ts.size)]
    )
    areas = np.array(
        [conic_area(camera, pose, X[i], R[i], radius) for i in range(ts.size)]
    )
    return (areas[:, None] * centers).sum(axis=0) / areas.sum()