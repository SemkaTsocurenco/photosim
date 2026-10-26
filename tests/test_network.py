"""Тесты сети камер и поля неопределённости (статья B, этап 2)."""

import math
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SR

from photosim.camera import Camera, Pose
from photosim.network import (
    _SYM,
    CameraNetwork,
    TargetSet,
    _inverse_sym3,
    _lambda_max_sym3,
    evaluate,
)
from photosim.obs_model import ObsModel, observe
from photosim.presets import camera

CAM = camera("review_16mm")
CONST = ObsModel(mode="constant", sigma_c=0.05)
UP = (0.0, -1.0, 0.0)  # мир: z — вперёд от сети, y — вниз на снимке


def ring(k=6, radius=0.8, target=(0.0, 0.0, 1.5), cam=CAM):
    """``k`` камер по кругу в плоскости z = 0, все смотрят на ``target``."""
    poses = []
    for i in range(k):
        a = 2 * math.pi * i / k
        center = (radius * math.cos(a), radius * math.sin(a), 0.0)
        poses.append(Pose.look_at(center, target, up=UP))
    return CameraNetwork([cam] * k, poses)


def volume(P, seed=0, moving=True):
    """Записи в объёме около (0, 0, 1,5) м, нормали — к сети (к -z)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform([-0.25, -0.25, 1.3], [0.25, 0.25, 1.7], size=(P, 3))
    n = rng.normal(scale=0.4, size=(P, 3))
    n[:, 2] = -1.0
    V = rng.normal(scale=0.3, size=(P, 3)) if moving else None
    return TargetSet(X, n, V)


def parallel_stereo(B):
    """Нормальный случай: камеры в (±B/2, 0, 0), оси || +z."""
    poses = [Pose(np.eye(3), [-x, 0.0, 0.0]) for x in (-B / 2, B / 2)]
    return CameraNetwork([CAM, CAM], poses)


def test_jacobian_vs_finite_differences():
    distorted = Camera(
        CAM.fx, CAM.fy, CAM.cx, CAM.cy, CAM.width, CAM.height, (-0.1, 0.02, 1e-3, -1e-3)
    )
    ts = volume(30, moving=False)
    h = 1e-6
    for cam in (CAM, distorted):
        net = ring(3, cam=cam)
        o = observe(ObsModel(), net.pairs, ts.points, ts.normals)
        assert o.visible.any()
        for c, (cam_, pose) in enumerate(net.pairs):
            for p in np.flatnonzero(o.visible[:, c]):
                cols = []
                for k in range(3):
                    d = np.zeros(3)
                    d[k] = h
                    plus = cam_.project(pose.to_camera(ts.points[p] + d))
                    minus = cam_.project(pose.to_camera(ts.points[p] - d))
                    cols.append((plus - minus) / (2 * h))
                np.testing.assert_allclose(
                    o.jacobian[p, c], np.column_stack(cols), rtol=1e-6, atol=1e-3
                )


@pytest.mark.parametrize("B", [0.2, 0.5])
@pytest.mark.parametrize("z", [1.5, 2.5, 4.0])  # при B = 0,5, Z = 1 — в отступе 5 %
def test_normal_stereo_case(B, z):
    # sigma_Z = Z**2 sigma_p / (B f), sigma_p = sqrt(2) sigma — параллакс
    r = evaluate(
        parallel_stereo(B), TargetSet([[0.0, 0.0, z]], [0.0, 0.0, -1.0]), CONST
    )
    assert r.measured[0] and r.n_visible[0] == 2
    sigma_z = z**2 * math.sqrt(2) * 0.05 / (B * CAM.fx)
    assert math.sqrt(r.cov[0, 2, 2]) == pytest.approx(sigma_z, rel=0.01)
    assert r.sigma_max_um[0] == pytest.approx(
        sigma_z * 1e6, rel=0.01
    )  # глубина — худшая ось
    # поперёк: sigma_X = Z sigma / (f sqrt 2), sigma_Y так же
    assert math.sqrt(r.cov[0, 0, 0]) == pytest.approx(
        z * 0.05 / (CAM.fx * math.sqrt(2)), rel=0.01
    )
    assert r.bias_um[0] == 0.0 and r.err_total_um[0] == r.sigma_max_um[0]


def test_info_matches_explicit_sum():
    net, ts = ring(5), volume(200)
    for model in (ObsModel(), ObsModel(eccentricity="uncorrected"), CONST):
        r = evaluate(net, ts, model)
        o = observe(model, net.pairs, ts.points, ts.normals, ts.velocities)
        W = np.where(r.visible[..., None, None], o.weight, 0.0)
        info = np.einsum("pcji,pcjk,pckl->pil", o.jacobian, W, o.jacobian)
        np.testing.assert_allclose(r.info, info, rtol=1e-10, atol=1e-12 * info.max())
        m = r.measured
        assert m.mean() > 0.9
        np.testing.assert_allclose(r.cov[m], np.linalg.inv(info[m]), rtol=1e-8)
        g = np.einsum("pcji,pcjk,pck->pi", o.jacobian, W, o.bias)
        delta = np.einsum("pij,pj->pi", r.cov[m], g[m])
        np.testing.assert_allclose(r.delta[m], delta, rtol=1e-8, atol=1e-15)
        eig = np.linalg.eigvalsh(r.cov[m])
        np.testing.assert_allclose(
            r.sigma_max_um[m], np.sqrt(eig[:, -1]) * 1e6, rtol=1e-9
        )
        np.testing.assert_allclose(
            r.sigma_trace_um[m], np.sqrt(eig.sum(1)) * 1e6, rtol=1e-9
        )
        np.testing.assert_allclose(
            r.bias_um[m], np.linalg.norm(r.delta[m], axis=1) * 1e6
        )
        np.testing.assert_allclose(
            r.err_total_um[m], np.hypot(r.sigma_max_um[m], r.bias_um[m]), rtol=1e-12
        )
        if model.eccentricity == "uncorrected":
            assert np.all(r.bias_um[m] > 0)
        else:
            np.testing.assert_array_equal(r.delta[m], 0.0)


def test_adding_camera_never_worse():
    ts = volume(300, seed=4)
    rng = np.random.default_rng(9)
    for model in (ObsModel(), CONST):
        net = CameraNetwork.from_pairs(ring(6).pairs[::2])  # 3 камеры
        old = evaluate(net, ts, model)
        for _ in range(4):
            center = rng.uniform([-1.0, -1.0, -0.3], [1.0, 1.0, 0.3])
            net = net.add(
                CAM,
                Pose.look_at(center, rng.uniform(-0.2, 0.2, 3) + [0, 0, 1.5], up=UP),
            )
            new = evaluate(net, ts, model)
            assert np.all(new.measured[old.measured])
            m = old.measured
            diff = old.cov[m] - new.cov[m]
            scale = np.abs(old.cov[m]).max(axis=(1, 2))
            assert np.all(np.linalg.eigvalsh(diff).min(axis=1) >= -1e-9 * scale)
            assert np.all(new.sigma_max_um[m] <= old.sigma_max_um[m] * (1 + 1e-9))
            old = new


def test_one_or_no_visible_camera_is_inf():
    pts = [[0.0, 0.0, 1.5], [0.0, 0.0, 1.5], [0.0, 0.0, -1.0]]
    normals = [
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]  # видна; сзади; за камерами
    ts = TargetSet(pts, normals)
    for net in (
        CameraNetwork([CAM], [Pose(np.eye(3), np.zeros(3))]),
        parallel_stereo(0.3),
    ):
        r = evaluate(net, ts, ObsModel(eccentricity="uncorrected"))
        expected_visible = [len(net), 0, 0]
        np.testing.assert_array_equal(r.n_visible, expected_visible)
        unmeasured = [len(net) < 2, True, True]
        np.testing.assert_array_equal(~r.measured, unmeasured)
        u = np.array(unmeasured)
        for a in (r.sigma_max_um, r.sigma_trace_um, r.bias_um, r.err_total_um):
            assert np.all(np.isinf(a[u]))
        assert np.all(np.isinf(r.delta[u]))
        np.testing.assert_array_equal(
            r.cov[u], np.broadcast_to(np.diag([np.inf] * 3), (u.sum(), 3, 3))
        )
        for a in (r.cov, r.delta, r.info, r.sigma_max_um, r.err_total_um):
            assert not np.isnan(a).any()


def test_degenerate_pair_on_one_ray():
    # обе камеры видят точку по одному лучу: I_p вырождена
    poses = [Pose(np.eye(3), [0.0, 0.0, 0.0]), Pose(np.eye(3), [0.0, 0.0, 0.5])]
    r = evaluate(
        CameraNetwork([CAM, CAM], poses),
        TargetSet([[0.0, 0.0, 1.5]], [0, 0, -1.0]),
        CONST,
    )
    assert r.n_visible[0] == 2 and not r.measured[0] and np.isinf(r.sigma_max_um[0])


def rigid(net, ts, R_s, t_s):
    """Сеть и записи после X' = R_s X + t_s."""
    poses = [Pose(p.R @ R_s.T, p.T - p.R @ R_s.T @ t_s) for p in net.poses]
    moved = TargetSet(
        ts.points @ R_s.T + t_s, ts.normals @ R_s.T, ts.velocities @ R_s.T
    )
    return CameraNetwork(net.cameras, poses), moved


def test_rigid_motion_of_scene_keeps_eigenvalues():
    net, ts = ring(4), volume(150, seed=2)
    R_s = SR.from_euler("xyz", [30.0, -50.0, 110.0], degrees=True).as_matrix()
    t_s = np.array([3.0, -7.0, 2.5])
    net2, ts2 = rigid(net, ts, R_s, t_s)
    for model in (ObsModel(eccentricity="uncorrected"), CONST):
        a, b = evaluate(net, ts, model), evaluate(net2, ts2, model)
        np.testing.assert_array_equal(a.visible, b.visible)
        m = a.measured
        np.testing.assert_array_equal(m, b.measured)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(a.cov[m]), np.linalg.eigvalsh(b.cov[m]), rtol=1e-7
        )
        np.testing.assert_allclose(
            b.cov[m], R_s @ a.cov[m] @ R_s.T, rtol=1e-7, atol=1e-20
        )
        for name in ("sigma_max_um", "sigma_trace_um", "bias_um", "err_total_um"):
            np.testing.assert_allclose(
                getattr(a, name)[m], getattr(b, name)[m], rtol=1e-7
            )


def test_vectorized_matches_pointwise():
    net = ring(5)
    ts = volume(40, seed=6)
    extra = TargetSet(
        [[0.0, 0.0, -1.0], [3.0, 0.0, 1.5]], [0.0, 0.0, -1.0]
    )  # не измеряются
    ts = TargetSet(
        np.vstack([ts.points, extra.points]),
        np.vstack([ts.normals, extra.normals]),
        np.vstack([ts.velocities, extra.velocities]),
    )
    model = ObsModel(eccentricity="uncorrected")
    r = evaluate(net, ts, model)
    assert not r.measured.all() and r.measured.any()
    for p in range(len(ts)):
        one = evaluate(
            net, TargetSet(ts.points[[p]], ts.normals[[p]], ts.velocities[[p]]), model
        )
        for name in ("visible", "n_visible", "measured"):
            np.testing.assert_array_equal(getattr(r, name)[p], getattr(one, name)[0])
        for name in (
            "cov",
            "delta",
            "info",
            "sigma_max_um",
            "sigma_trace_um",
            "err_total_um",
        ):
            a, b = getattr(r, name)[p], getattr(one, name)[0]
            # малые компоненты вектора и матрицы — с допуском от наибольшей:
            # округление пачки и одной точки различается в последнем бите
            scale = np.abs(b).max() if np.all(np.isfinite(b)) else 0.0
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-9 * scale)


def test_visibility_rules():
    net = CameraNetwork([CAM], [Pose(np.eye(3), np.zeros(3))])
    model = ObsModel(sigma_n_deg=0.0)

    def visible(u, tilt_deg=0.0, **kw):
        z = 1.0
        X = [(u - CAM.cx) * z / CAM.fx, 0.0, z]
        t = math.radians(tilt_deg)
        n = [-math.sin(t), 0.0, -math.cos(t)]
        return bool(evaluate(net, TargetSet([X], n), model, **kw).visible[0, 0])

    W = CAM.width
    assert visible(0.06 * W) and not visible(0.04 * W)  # отступ 5 %
    assert visible(0.94 * W) and not visible(0.96 * W)
    assert visible(0.04 * W, frame_margin=0.0)
    assert visible(CAM.cx, tilt_deg=69.0) and not visible(CAM.cx, tilt_deg=71.0)
    assert visible(CAM.cx, tilt_deg=71.0, max_view_deg=80.0)
    with pytest.raises(ValueError):
        evaluate(net, TargetSet([[0, 0, 1.0]], [0, 0, -1.0]), model, max_view_deg=95.0)


def test_sym3_helpers():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(500, 3, 3))
    S = A @ A.transpose(0, 2, 1)
    # плохо обусловленные и почти вырожденные по собственным числам
    Q = SR.random(200, random_state=3).as_matrix()
    lam = np.column_stack(
        [np.full(200, 1.0), rng.uniform(1e-6, 1, 200), np.full(200, 1e-8)]
    )
    S = np.concatenate(
        [S, Q @ (lam[:, :, None] * Q.transpose(0, 2, 1)), np.tile(np.eye(3), (3, 1, 1))]
    )
    comps = np.stack([S[:, i, j] for i, j in _SYM])
    inv, det = _inverse_sym3(comps)
    np.testing.assert_allclose(det, np.linalg.det(S), rtol=1e-6, atol=1e-12)
    good = np.linalg.cond(S) < 1e6
    full = np.linalg.inv(S[good])
    np.testing.assert_allclose(
        np.stack([full[:, i, j] for i, j in _SYM]), inv[:, good], rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        _lambda_max_sym3(comps), np.linalg.eigvalsh(S)[:, -1], rtol=1e-9
    )


def test_target_set_and_network_checks():
    ts = TargetSet(np.zeros((4, 3)) + [0, 0, 1], [0.0, 0.0, -2.0])
    assert len(ts) == 4 and ts.normals.shape == (4, 3)
    np.testing.assert_array_equal(ts.normals[:, 2], -1.0)
    np.testing.assert_array_equal(ts.velocities, 0.0)
    assert not ts.points.flags.writeable
    with pytest.raises(ValueError):
        TargetSet([0.0, 0.0, 1.0], [0.0, 0.0, -1.0])  # нужна (P, 3)
    with pytest.raises(ValueError):
        CameraNetwork([CAM, CAM], [Pose(np.eye(3), np.zeros(3))])
    with pytest.raises(ValueError):
        CameraNetwork([], [])
    net = ring(2)
    assert len(net.add(CAM, Pose(np.eye(3), np.zeros(3)))) == 3 and len(net) == 2


@pytest.mark.slow
def test_speed_2000_records_6_cameras():
    net, ts = ring(6), volume(2000, seed=8)
    for model in (ObsModel(), ObsModel(eccentricity="uncorrected")):
        r = evaluate(net, ts, model)
        assert r.measured.mean() > 0.95
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            evaluate(net, ts, model)
            times.append(time.perf_counter() - t0)
        # требование плана — не больше 10 мс; минимум — без помех от системы
        assert min(times) < 10e-3, f"{min(times) * 1e3:.1f} мс"
