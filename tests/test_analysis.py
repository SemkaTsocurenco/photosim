"""Тесты теоретических оценок для экспериментов."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as SR

from photosim.analysis import (
    along_across,
    background_variance,
    centroid_sigma,
    centroid_signal_for_sigma,
    eccentricity_first_order,
    eccentricity_formula3,
    eccentricity_point_shift,
    eccentricity_pole,
    eccentricity_tilt_sensitivity,
    fit_variance_quadratic,
    interp_log,
    log_crossing,
    method_own_bias,
    point_at_pixel,
    roi_s2,
    rotation_offset_coefficients,
    std_along_across,
)
from photosim.camera import Pose
from photosim.network import CameraNetwork, TargetSet, evaluate
from photosim.obs_model import ObsModel
from photosim.oracle import conic_center, exposure_conic_center
from photosim.presets import camera
from photosim.scene import AxisRotation, Motion, Static, frame_from_normal

CAM = camera("review_16mm")
POSE = Pose(np.eye(3), np.zeros(3))


def tilted(theta_deg):
    t = math.radians(theta_deg)
    return np.array([-math.sin(t), 0.0, -math.cos(t)])


@pytest.mark.parametrize("theta", [15.0, 45.0, 75.0])
@pytest.mark.parametrize("r, z", [(0.005, 1.0), (0.01, 0.8)])
def test_formula3_on_axis_matches_conic(theta, r, z):
    n = tilted(theta)
    X = np.array([0.0, 0.0, z])
    exact = conic_center(CAM, POSE, X, frame_from_normal(n), r) - CAM.project(X)
    eps3 = eccentricity_formula3(CAM.fx, r, n, z)
    # первый порядок по r/Z: замер 2e-6 ... 1,5e-4
    np.testing.assert_allclose(eps3, exact, rtol=2e-4, atol=1e-9)  # v: шум 1e-11
    np.testing.assert_array_equal(eccentricity_formula3(CAM.fx, r, -n, z), eps3)


def random_cases(n=40, seed=5):
    """Положения по кадру и нормали, видимые с камеры (n_z < 0)."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        u, v = rng.uniform(0, CAM.width - 1), rng.uniform(0, CAM.height - 1)
        C = np.array(point_at_pixel(CAM, u, v, rng.uniform(0.6, 1.5)))
        normal = rng.normal(size=3)
        normal[2] = -abs(normal[2]) - 0.2
        yield C, normal / np.linalg.norm(normal), rng.uniform(0.003, 0.012)


def test_pole_formula_is_exact():
    for C, n, r in random_cases():
        exact = conic_center(CAM, POSE, C, frame_from_normal(n), r) - CAM.project(C)
        np.testing.assert_allclose(eccentricity_pole(CAM, C, n, r), exact, atol=1e-9)
        np.testing.assert_allclose(eccentricity_pole(CAM, C, -n, r), exact, atol=1e-9)


def test_first_order_formula():
    for C, n, r in random_cases():
        exact = eccentricity_pole(CAM, C, n, r)
        first = eccentricity_first_order(CAM.fx, r, n, C)
        # ошибка второго порядка по r / Z
        bound = 5 * (r / C[2]) ** 2 * np.linalg.norm(exact)
        assert np.linalg.norm(first - exact) <= bound
    # на оси член вне оси равен нулю: совпадает с формулой (3)
    n, C = tilted(30.0), np.array([0.0, 0.0, 0.9])
    np.testing.assert_allclose(
        eccentricity_first_order(CAM.fx, 0.005, n, C),
        eccentricity_formula3(CAM.fx, 0.005, n, 0.9),
        rtol=1e-15,
    )
    batch = np.stack([C for C, _, _ in random_cases(5)])
    normals = np.stack([n for _, n, _ in random_cases(5)])
    assert eccentricity_first_order(CAM.fx, 0.005, normals, batch).shape == (5, 2)
    assert eccentricity_pole(CAM, batch, normals, 0.005).shape == (5, 2)


def test_tilt_sensitivity_on_axis():
    h = 1e-6
    for theta in (10.0, 45.0, 70.0):
        t = math.radians(theta)
        n_plus, n_minus = tilted(math.degrees(t + h)), tilted(math.degrees(t - h))
        C = np.array([0.0, 0.0, 1.0])
        d = (
            eccentricity_first_order(CAM.fx, 0.005, n_plus, C)[0]
            - eccentricity_first_order(CAM.fx, 0.005, n_minus, C)[0]
        ) / (2 * h)
        expected = eccentricity_tilt_sensitivity(CAM.fx, 0.005, t, 1.0)
        assert d == pytest.approx(expected, rel=1e-6, abs=1e-9)
    assert eccentricity_tilt_sensitivity(CAM.fx, 0.005, math.pi / 4, 1.0) == (
        pytest.approx(0.0, abs=1e-15)
    )


@pytest.mark.parametrize("v_deg", [15.0, 40.0, 60.0])
@pytest.mark.parametrize("d, r", [(0.8, 0.01), (1.5, 0.005), (4.0, 0.005)])
@pytest.mark.parametrize("mode", ["model", "constant"])
def test_point_shift_symmetric_pair(v_deg, d, r, mode):
    """Две камеры под ±v к нормали смотрят на марку: оценка верна до (r/D)**2."""
    rng = np.random.default_rng(int(v_deg * 10 + d * 100))
    X0 = np.array([0.1, -0.05, 2.0])
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    t = np.cross(n, rng.normal(size=3))
    t /= np.linalg.norm(t)
    v = math.radians(v_deg)
    centers = np.array(
        [X0 + d * (math.cos(v) * n + s * math.sin(v) * t) for s in (1, -1)]
    )
    poses = [Pose.look_at(c, X0, up=np.cross(n, t)) for c in centers]
    model = ObsModel(radius=r, eccentricity="uncorrected", mode=mode)
    res = evaluate(CameraNetwork([CAM, CAM], poses), TargetSet([X0], n), model)
    assert res.measured[0] and res.visible.all()
    est = eccentricity_point_shift(r, X0, n, centers, res.visible[0])
    np.testing.assert_allclose(est, -(r**2) * math.cos(v) / d * n, rtol=1e-12)
    # замер: 0,75 (r / D)**2 при v = 60°, меньше при меньших v
    rel = np.linalg.norm(res.delta[0] - est) / np.linalg.norm(est)
    assert rel <= (r / d) ** 2


def test_point_shift_shapes_and_no_camera():
    centers = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    X = np.array([[0.0, 0.0, 1.5], [0.1, 0.0, 1.4]])
    vis = np.array([[True, True, False], [False, False, False]])
    out = eccentricity_point_shift(0.005, X, [0.0, 0.0, -2.0], centers, vis)
    assert out.shape == (2, 3)
    # по нормали внутрь марки (нормаль -z смотрит на камеры, смещение — +z)
    assert out[0, 2] > 0 and np.allclose(out[0, :2], 0.0)
    D = math.hypot(0.5, 1.5)
    assert out[0, 2] == pytest.approx(0.005**2 * (1.5 / D) / D, rel=1e-12)
    assert np.isnan(out[1]).all()  # запись не видит ни одна камера
    batch = eccentricity_point_shift(
        0.005, X.reshape(1, 2, 3), [0, 0, -1.0], centers, vis[None]
    )
    assert batch.shape == (1, 2, 3)
    np.testing.assert_allclose(batch[0, 0], out[0], rtol=1e-15)


def test_formula3_vectorized():
    normals = np.stack([tilted(a) for a in (15.0, 45.0, 75.0)])
    out = eccentricity_formula3(CAM.fx, 0.005, normals, np.array([1.0, 0.9, 1.1]))
    assert out.shape == (3, 2)
    np.testing.assert_allclose(
        out[1], eccentricity_formula3(CAM.fx, 0.005, normals[1], 0.9)
    )


def test_point_at_pixel():
    X = point_at_pixel(CAM, 1224.3, 1024.3, 1.2)
    assert X[2] == 1.2
    np.testing.assert_allclose(CAM.project(np.array(X)), [1224.3, 1024.3], atol=1e-9)


@pytest.mark.parametrize(
    "u0, w, h, uc", [(1196, 57, 56, 1224.3), (0, 1, 3, 0.2), (10, 8, 5, 20.0)]
)
def test_roi_s2_closed_form(u0, w, h, uc):
    brute = h * np.sum((u0 + np.arange(w) - uc) ** 2)
    assert roi_s2(u0, w, h, uc) == pytest.approx(brute, rel=1e-12)


def test_centroid_sigma():
    assert centroid_sigma(23.0, 1e6) == pytest.approx(23.0 / (2 * 1e3))
    var_bg = background_variance(2.16, 2.72)
    assert var_bg == pytest.approx(2.16**2 + 2.72**2 / 12)
    with_bg = centroid_sigma(23.0, 1e6, var_bg, 1e6)
    assert with_bg == pytest.approx(
        math.sqrt((23.0**2 / 4) / 1e6 + var_bg * 1e6 / 1e12)
    )
    blurred = centroid_sigma(2.0, 1e4, extra_var_px=0.49 + 1 / 12)
    assert blurred == pytest.approx(math.sqrt((1 + 0.49 + 1 / 12) / 1e4))
    np.testing.assert_allclose(centroid_sigma(np.array([2.0, 4.0]), 1e4), [0.01, 0.02])


@pytest.mark.parametrize("var_bg, s2, extra", [(0.0, 0.0, 0.0), (5.3, 1e6, 0.57)])
def test_centroid_signal_for_sigma_inverts(var_bg, s2, extra):
    n = centroid_signal_for_sigma(0.05, 10.0, var_bg, s2, extra)
    assert centroid_sigma(10.0, n, var_bg, s2, extra) == pytest.approx(0.05)
    if var_bg == 0.0:
        assert n == pytest.approx((10.0 / (2 * 0.05)) ** 2)


def test_log_crossing():
    x = np.array([1e3, 1e4, 1e5, 1e6])
    y = 10 / np.sqrt(x)  # 0,316 ... 0,01
    xc, extrapolated = log_crossing(x, y, 0.05)
    assert xc == pytest.approx(4e4) and not extrapolated
    xc, extrapolated = log_crossing(x, y, 1.0)  # выше первой точки
    assert xc == pytest.approx(1e2) and extrapolated
    xc, extrapolated = log_crossing(x, y, 1e-3)  # ниже последней
    assert xc == pytest.approx(1e8) and extrapolated
    assert math.isnan(log_crossing([1.0], [1.0], 0.5)[0])


def test_along_across():
    along, across = along_across(1.0, 1.0, math.pi / 4)
    assert along == pytest.approx(math.sqrt(2)) and across == pytest.approx(0.0)
    along, across = along_across(1.0, 0.0, math.pi / 2)
    assert along == pytest.approx(0.0, abs=1e-15) and across == pytest.approx(-1.0)


def test_std_along_across_matches_samples():
    rng = np.random.default_rng(3)
    cov = np.array([[4.0, 1.5], [1.5, 1.0]])
    d = rng.multivariate_normal([0.0, 0.0], cov, size=200_000)
    su, sv = d.std(axis=0)
    corr = np.corrcoef(d.T)[0, 1]
    for angle in (0.0, 0.3, math.pi / 4, 2.0):
        along, across = along_across(d[:, 0], d[:, 1], angle)
        s_al, s_ac = std_along_across(su, sv, corr, angle)
        assert s_al == pytest.approx(along.std(), rel=1e-9)
        assert s_ac == pytest.approx(across.std(), rel=1e-9)


def test_fit_variance_quadratic_exact():
    x = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    s0, k, rms = fit_variance_quadratic(x, np.sqrt(0.003**2 + (2e-4 * x) ** 2))
    assert s0 == pytest.approx(0.003) and k == pytest.approx(2e-4) and rms < 1e-12
    # роста нет, а шум его «уменьшил»: k со знаком минус
    _, k, _ = fit_variance_quadratic(x, 0.003 * (1 - 1e-4 * x**2))
    assert k < 0


def test_interp_log():
    L = np.array([0.0, 2.0, 4.0])
    s = np.array([1e-3, 4e-3, 1.6e-2])
    np.testing.assert_allclose(interp_log(L, L, s), s)  # узлы — точно
    # между узлами — линейно по L в логарифме: среднее геометрическое
    assert interp_log(1.0, L, s) == pytest.approx(2e-3)
    # за краями — крайнее значение
    np.testing.assert_allclose(interp_log([-1.0, 9.0], L, s), [1e-3, 1.6e-2])


@pytest.mark.parametrize("theta", [15.0, 45.0, 75.0])
@pytest.mark.parametrize("uv", [(1223.5, 1023.5), (2203.0, 1842.0)], ids=["axis", "corner"])
def test_rotation_offset_second_order(theta, uv):
    X = np.array(point_at_pixel(CAM, *uv, 1.0))
    R_mid = frame_from_normal(tilted(theta))
    axis, r, tau = np.array([0.0, 1.0, 0.0]), 0.01, 1e-3
    k_curv, k_area = rotation_offset_coefficients(CAM, POSE, X, R_mid, axis, r)
    mid = conic_center(CAM, POSE, X, R_mid, r)
    for deg in (1.0, 5.0):
        dtheta = math.radians(deg)
        omega = dtheta / tau
        motion = Motion(Static(X), AxisRotation(
            SR.from_rotvec(axis * (-omega * tau / 2)).as_matrix() @ R_mid, axis, omega
        ))
        exact = exposure_conic_center(CAM, POSE, motion, r, 0.0, tau, 2048) - mid
        # четвёртый порядок: относительная ошибка ~ dtheta**2
        np.testing.assert_allclose(
            (k_curv + k_area) * dtheta**2, exact, rtol=0.02, atol=1e-9 * deg**2
        )


def test_method_own_bias():
    df = pd.DataFrame(
        {
            "status": "ok",
            "effects.psf": [True, True, True, False],
            "radius_px": [10.0, 10.0, 23.0, 23.0],
            "method": ["m", "m", "m", "m"],
            "err0_u_px": [1e-3, -3e-3, 5e-4, 9.0],
        }
    )
    assert method_own_bias(df, 12.0) == {"m": 3e-3}
    assert method_own_bias(df, 22.0) == {"m": 5e-4}
    assert method_own_bias(df, 22.0, psf=False) == {"m": 9.0}
