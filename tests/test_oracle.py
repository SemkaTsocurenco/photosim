import math

import numpy as np
import pytest

from photosim.camera import Camera, Pose
from photosim.oracle import (
    conic_area,
    conic_center,
    exposure_conic_center,
    mean_projected_center,
)
from photosim.scene import Motion, frame_from_normal

CAM = Camera.from_physical(16, 3.45, 2448, 2048)
POSE = Pose(np.eye(3), np.zeros(3))
F = CAM.fx


def test_facing_on_axis_is_principal_point():
    R = frame_from_normal([0, 0, -1])
    np.testing.assert_allclose(
        conic_center(CAM, POSE, [0, 0, 1], R, 0.005), [CAM.cx, CAM.cy], atol=1e-9
    )
    assert conic_area(CAM, POSE, [0, 0, 1], R, 0.005) == pytest.approx(
        math.pi * (F * 0.005) ** 2, rel=1e-9
    )


def test_fronto_parallel_center_is_projected_center():
    # плоскость марки параллельна сенсору: эксцентриситета нет
    X = np.array([0.15, -0.07, 1.3])
    c = conic_center(CAM, POSE, X, frame_from_normal([0, 0, -1]), 0.01)
    np.testing.assert_allclose(c, CAM.project(X), atol=1e-9)


def test_conic_center_independent_of_in_plane_rotation():
    n = [-0.5, 0.2, -0.8]
    X = [0.1, 0.05, 1.0]
    a = conic_center(CAM, POSE, X, frame_from_normal(n), 0.005)
    b = conic_center(CAM, POSE, X, frame_from_normal(n, ref=[1, 0, 0]), 0.005)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_tilted_center_matches_sampled_ellipse():
    # центр коники = среднее точек контура, равномерных по параметру эллипса;
    # здесь проверяем через площадь: проекция контура — замкнутая кривая,
    # её площадь (формула шнурования) совпадает с conic_area
    n = np.array([-math.sin(0.8), 0.0, -math.cos(0.8)])
    X = np.array([0.2, 0.1, 1.0])
    R = frame_from_normal(n)
    r = 0.005
    phi = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    pts = X + r * (np.outer(np.cos(phi), R[:, 0]) + np.outer(np.sin(phi), R[:, 1]))
    uv = CAM.project(pts)
    x, y = uv[:, 0], uv[:, 1]
    shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert conic_area(CAM, POSE, X, R, r) == pytest.approx(shoelace, rel=1e-6)
    # центр тяжести области по контуру (формула Грина)
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a2 = cross.sum()
    cx = ((x + np.roll(x, -1)) * cross).sum() / (3 * a2)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (3 * a2)
    np.testing.assert_allclose(conic_center(CAM, POSE, X, R, r), [cx, cy], atol=1e-6)


def test_distortion_not_implemented():
    cam = Camera(
        CAM.fx, CAM.fy, CAM.cx, CAM.cy, CAM.width, CAM.height, dist=[-0.1, 0, 0, 0]
    )
    with pytest.raises(NotImplementedError):
        conic_center(cam, POSE, [0, 0, 1], np.eye(3), 0.005)


def test_degenerate_cases():
    with pytest.raises(ValueError, match="плоскости"):
        conic_center(CAM, POSE, [0, 0, 1], frame_from_normal([1, 0, 0]), 0.005)
    with pytest.raises(ValueError, match="не эллипс"):
        conic_center(CAM, POSE, [0, 0, 0.002], frame_from_normal([1, 0, -0.1]), 0.005)
    with pytest.raises(ValueError):
        conic_center(CAM, POSE, [0, 0, 1], np.eye(3), -1.0)


def test_mean_projected_center_linear():
    m = Motion.linear([0, 0, 1], [0.1, 0, 0], [0, 0, -1])
    c = mean_projected_center(CAM, POSE, m, 0.0, 0.01, 8)
    # проекция линейна при постоянной глубине: среднее = проекция в середине
    np.testing.assert_allclose(c, CAM.project([0.1 * 0.005, 0, 1]), atol=1e-9)


def test_exposure_conic_center_reduces_for_static():
    m = Motion.static([0.2, 0.1, 1.0], [-0.7, 0, -0.7])
    a = exposure_conic_center(CAM, POSE, m, 0.005, 0.0, 0.01, 5)
    b = conic_center(CAM, POSE, m.position(0.0), m.orientation(0.0), 0.005)
    np.testing.assert_allclose(a, b, atol=1e-9)