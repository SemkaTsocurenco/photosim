"""Тесты модели погрешности наблюдения (статья B, этап 1)."""

import math

import numpy as np
import pytest

from photosim.analysis import (
    eccentricity_pole,
    eccentricity_tilt_sensitivity,
    point_at_pixel,
    std_along_across,
)
from photosim.camera import Pose
from photosim.experiment import Config, build, run_point
from photosim.obs_model import ObsModel, observe, sigma0
from photosim.oracle import _image_conic, conic_area
from photosim.presets import REVIEW_16MM, camera
from photosim.scene import frame_from_normal

CAM = camera("review_16mm")
POSE = Pose(np.eye(3), np.zeros(3))
PAIR = (CAM, POSE)
FW = REVIEW_16MM["full_well_e"]
# только фотонный член: нормаль известна точно, камеры синхронны
SHOT = ObsModel(sigma_n_deg=0.0, sigma_t=0.0)
TAU = 1e-3


def tilted(theta_deg, axis="y"):
    """Нормаль к камере на оси, наклонённая на ``theta_deg`` (как в E1–E3)."""
    t = math.radians(theta_deg)
    if axis == "y":
        return np.array([-math.sin(t), 0.0, -math.cos(t)])
    return np.array([0.0, math.sin(t), -math.cos(t)])


def speed(length_px, z):
    """Скорость, при которой смаз за ``TAU`` на глубине ``z`` — ``length_px``."""
    return length_px * z / (CAM.fx * TAU)


def sqrt_eig(cov):
    return np.sqrt(np.linalg.eigvalsh(cov))


def test_sigma0():
    s0 = 1 / (2 * math.sqrt(math.pi * 0.7 * FW))
    assert sigma0(0.7, FW) == pytest.approx(s0, rel=1e-15)
    assert s0 == pytest.approx(3.2e-3, rel=1e-2)  # уровень E1
    assert ObsModel().sigma0 == pytest.approx(s0, rel=1e-15)
    assert ObsModel(kappa=1.44).sigma0 == pytest.approx(1.2 * s0, rel=1e-12)


@pytest.mark.parametrize("z", [0.5, 1.0, 3.0])
def test_facing_static_is_sigma0(z):
    o = observe(SHOT, PAIR, [0.0, 0.0, z], [0.0, 0.0, -1.0])
    assert o.visible[0]
    # от расстояния не зависит
    np.testing.assert_allclose(
        o.cov[0], SHOT.sigma0**2 * np.eye(2), rtol=1e-12, atol=1e-20
    )
    a = CAM.fx * SHOT.radius / z
    np.testing.assert_allclose(o.semi_axes_px[0], [a, a], rtol=1e-12)
    assert o.n_e[0] == pytest.approx(0.7 * FW * math.pi * a * a, rel=1e-12)
    assert o.blur_px[0] == 0.0
    np.testing.assert_array_equal(o.cov_sync[0], 0.0)


def test_tilt45_matches_e3():
    o = observe(SHOT, PAIR, [0.0, 0.0, 1.0], tilted(45.0))
    s_min, s_maj = sqrt_eig(o.cov[0])
    c = math.cos(math.radians(45.0))
    assert s_min == pytest.approx(SHOT.sigma0 * math.sqrt(c), rel=1e-12)
    assert s_maj == pytest.approx(SHOT.sigma0 / math.sqrt(c), rel=1e-12)
    # малая ось — по проекции нормали на плоскость изображения (u)
    assert o.cov[0, 0, 0] == pytest.approx(s_min**2, rel=1e-12)
    assert o.cov[0, 0, 1] == pytest.approx(0.0, abs=1e-20)
    # E3 (figures/e3_table.csv), центр тяжести, наклон 45°, L = 0, n_mc = 500:
    # std вдоль u 0,00263, вдоль v 0,00379 (в плане — «E1»: 0,0027 и 0,0038)
    assert s_min == pytest.approx(0.00263, rel=0.05)
    assert s_maj == pytest.approx(0.00379, rel=0.05)


def test_distance_independence_and_invariant_product():
    rng = np.random.default_rng(3)
    for _ in range(20):
        u, v = rng.uniform(100, CAM.width - 100), rng.uniform(100, CAM.height - 100)
        ray = np.array(point_at_pixel(CAM, u, v, 1.0))
        normal = rng.normal(size=3)
        normal -= 1.5 * np.linalg.norm(normal) * ray / np.linalg.norm(ray)  # к камере
        o = observe(SHOT, PAIR, np.stack([ray * z for z in (0.6, 1.0, 2.5)]), normal)
        assert o.visible.all()
        # форма образа вдоль луча та же — ковариация та же
        np.testing.assert_allclose(o.cov[1:], o.cov[[0, 0]], rtol=1e-9)
        # sigma_maj sigma_min = sigma0**2 при любом наклоне и положении
        assert np.prod(sqrt_eig(o.cov[0, 0])) == pytest.approx(SHOT.sigma0**2, rel=1e-9)


def exact_moment(X, normal, r):
    """Точный второй момент эллипса-образа, пикс.² (как в E3)."""
    C = _image_conic(CAM, POSE, X, frame_from_normal(normal), r)
    A, b = C[:2, :2], C[:2, 2]
    k = C[2, 2] + b @ np.linalg.solve(A, -b)
    return np.linalg.inv(A / -k) / 4


@pytest.mark.parametrize("uv", [(1224.3, 1024.3), (2300.0, 1950.0), (150.0, 900.0)])
@pytest.mark.parametrize("theta", [0.0, 40.0, 70.0])
def test_moment_and_signal_vs_exact_conic(uv, theta):
    X = np.array(point_at_pixel(CAM, *uv, 1.0))
    normal = tilted(theta)
    o = observe(SHOT, PAIR, X, normal)
    # M0 = cov * N_e без смаза и поправок; первый порядок по r / Z = 5e-3
    M0 = o.cov[0] * o.n_e[0]
    np.testing.assert_allclose(
        M0, exact_moment(X, normal, SHOT.radius), rtol=2e-4, atol=1e-3
    )
    area = conic_area(CAM, POSE, X, frame_from_normal(normal), SHOT.radius)
    assert o.n_e[0] == pytest.approx(0.7 * FW * area, rel=1e-4)


def test_off_axis_radial_stretch():
    # марка, развёрнутая к камере, в углу кадра: образ вытянут по радиусу
    # в 1 / cos(phi) раз; формула оси (a = b) этого не видит
    X = np.array(point_at_pixel(CAM, CAM.width - 60, CAM.height - 60, 1.0))
    o = observe(SHOT, PAIR, X, -X / np.linalg.norm(X))
    cos_phi = X[2] / np.linalg.norm(X)
    a, b = o.semi_axes_px[0]
    assert o.visible[0]
    assert a / b == pytest.approx(1 / cos_phi, rel=1e-9)
    assert 1 / cos_phi == pytest.approx(1.052, abs=1e-3)  # phi = 18°
    assert o.cos_view[0] == pytest.approx(1.0, rel=1e-12)


# E3 (figures/e3_table.csv), центр тяжести, theory_along / theory_across:
# (наклон, направление движения, L, along, across); фаза центра 0,3 пикс.
E3_THEORY = [
    (0.0, 0.0, 0.0, 0.00321057, 0.00321057),
    (0.0, 0.0, 20.0, 0.00358653, 0.00321099),
    (0.0, 45.0, 20.0, 0.00358654, 0.00321251),
    (45.0, 0.0, 20.0, 0.00330680, 0.00381862),
    (45.0, 45.0, 20.0, 0.00381627, 0.00331146),
]


@pytest.mark.parametrize("tilt, direction, length, along, across", E3_THEORY)
def test_blur_matches_e3_theory(tilt, direction, length, along, across):
    a = math.radians(direction)
    e = np.array([math.cos(a), math.sin(a)])
    V = speed(length, 1.0) * np.array([math.cos(a), math.sin(a), 0.0])
    X = point_at_pixel(CAM, 1224.3, 1024.3, 1.0)
    o = observe(SHOT, PAIR, X, tilted(tilt), V)
    assert o.blur_px[0] == pytest.approx(length, rel=1e-3)
    cov = o.cov[0]
    s_along = math.sqrt(e @ cov @ e)
    s_across = math.sqrt(np.array([-e[1], e[0]]) @ cov @ np.array([-e[1], e[0]]))
    # E3 учитывает ещё PSF, отнесение к центру пикселя и фон — 0,3 %
    assert s_along == pytest.approx(along, rel=0.01)
    assert s_across == pytest.approx(across, rel=0.01)
    # точно: s_along**2 = s0_along**2 + L**2 / (12 N_e)
    o0 = observe(SHOT, PAIR, X, tilted(tilt))
    s0_along2 = e @ o0.cov[0] @ e
    assert s_along**2 == pytest.approx(
        s0_along2 + length**2 / (12 * o.n_e[0]), rel=1e-6
    )


# Сравнение с моделированием статьи A (run_point, центр тяжести, все шумы и
# PSF до пикселя): замер отклонения модели — до 4,8 % (наклон 60°, v).
SIM_CASES = {
    "axis_facing": {"position": point_at_pixel(CAM, 1224.3, 1024.3, 1.0)},
    "axis_tilt60": {
        "position": point_at_pixel(CAM, 1224.3, 1024.3, 1.0),
        "tilt_deg": 60.0,
    },
    "axis_tilt45_blur16_u": {
        "position": point_at_pixel(CAM, 1224.3, 1024.3, 1.0),
        "tilt_deg": 45.0,
        "motion": {"type": "linear", "velocity": (speed(16, 1.0), 0.0, 0.0)},
    },
    "corner_tilt30x_blur10_diag": {
        "position": point_at_pixel(CAM, 2100.3, 300.3, 1.0),
        "tilt_deg": 30.0,
        "tilt_axis": "x",
        "motion": {"type": "linear", "velocity": (speed(7, 1.0), speed(7, 1.0), 0.0)},
    },
    "z2_los50_blur12_v_fill04": {
        "position": point_at_pixel(CAM, 600.3, 1500.3, 2.0),
        "tilt_deg": 50.0,
        "tilt_reference": "line_of_sight",
        "fill": 0.4,
        "motion": {"type": "linear", "velocity": (0.0, speed(12, 2.0), speed(4, 2.0))},
    },
}


@pytest.mark.parametrize("case", SIM_CASES)
def test_matches_simulation(case):
    cfg = Config(methods=("centroid",), n_mc=500, **SIM_CASES[case])
    s = build(cfg)
    model = ObsModel(
        radius=cfg.radius, fill=cfg.fill, tau=cfg.tau, sigma_n_deg=0.0, sigma_t=0.0
    )
    o = observe(
        model,
        (s.camera, s.pose),
        s.motion.position(s.t_ref),
        s.motion.normal(s.t_ref),
        s.motion.velocity(s.t_ref),
    )
    row = run_point(cfg)[0]
    assert row["status"] == "ok" and o.visible[0]
    assert o.n_e[0] == pytest.approx(row["signal_e"], rel=1e-3)
    su, sv, corr = row["std_u_px"], row["std_v_px"], row["corr_uv"]
    # СКО вдоль u, v и вдоль собственных осей модели
    _, vecs = np.linalg.eigh(o.cov[0])
    for angle in (0.0, math.pi / 2, *np.arctan2(vecs[1], vecs[0])):
        e = np.array([math.cos(angle), math.sin(angle)])
        model_std = math.sqrt(e @ o.cov[0] @ e)
        sim_std, _ = std_along_across(su, sv, corr, angle)
        assert sim_std == pytest.approx(model_std, rel=0.15)


def test_vectorized_matches_pointwise():
    rng = np.random.default_rng(11)
    P = 12
    X = np.column_stack(
        [rng.uniform(-0.4, 0.4, P), rng.uniform(-0.4, 0.4, P), rng.uniform(0.8, 2.5, P)]
    )
    normals = rng.normal(size=(P, 3))
    V = rng.normal(scale=0.5, size=(P, 3))
    cams = [
        PAIR,
        (CAM, Pose.look_at([1.0, 0.2, 0.0], [0.0, 0.0, 1.5], up=[0, -1, 0])),
        (
            camera("rpi_cm3_2x2"),
            Pose.look_at([-0.8, -0.5, 0.3], [0.0, 0.0, 1.5], up=[0, -1, 0]),
        ),
    ]
    model = ObsModel()
    o = observe(model, cams, X, normals, V)
    assert o.cov.shape == (P, 3, 2, 2) and o.jacobian.shape == (P, 3, 2, 3)
    assert o.visible.shape == (P, 3)
    assert 0 < o.visible.sum() < o.visible.size  # есть и видимые, и невидимые
    for p in range(P):
        for c, pair in enumerate(cams):
            one = observe(model, pair, X[p], normals[p], V[p])
            assert one.cov.shape == (1, 2, 2)
            assert o.visible[p, c] == one.visible[0]
            # BLAS округляет пачку и одну точку по-разному в последнем бите
            for name in ("cov", "weight", "jacobian", "bias", "cov_ecc", "n_e"):
                np.testing.assert_allclose(
                    getattr(o, name)[p, c], getattr(one, name)[0], rtol=1e-10, atol=0
                )
    # ведущие оси точек сохраняются, нормаль и скорость транслируются
    o2 = observe(model, cams, X.reshape(3, 4, 3), normals[0], V[0])
    assert o2.cov.shape == (3, 4, 3, 2, 2)
    np.testing.assert_allclose(
        o2.cov[0, 1], observe(model, cams, X[1], normals[0], V[0]).cov, rtol=1e-10
    )


def test_invisible_pairs_flagged_without_nan():
    X = np.array(
        [
            [0.0, 0.0, 1.0],  # видна сзади: cos(view) = -1
            [0.0, 0.0, 1.0],  # ребром: cos(view) = 0
            [0.0, 0.0, -1.0],  # за камерой
            [2.0, 0.0, 1.0],  # вне кадра
            [0.262, 0.0, 1.0],  # центр в кадре (u = 2439), образ выходит за край
            [0.0, 0.0, 1.0],  # видна
        ]
    )
    normals = np.array(
        [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 0, -1], [0, 0, -1.0]]
    )
    o = observe(ObsModel(eccentricity="uncorrected"), PAIR, X, normals, [0.3, 0.1, 0.0])
    np.testing.assert_array_equal(o.visible[:, 0], [False] * 5 + [True])
    assert o.cos_view[0, 0] == pytest.approx(-1.0) and o.cos_view[
        1, 0
    ] == pytest.approx(0.0)
    for name in (
        "weight",
        "jacobian",
        "bias",
        "cov_shot",
        "cov_ecc",
        "cov_sync",
        "ecc",
        "semi_axes_px",
        "n_e",
        "blur_px",
        "cos_view",
    ):
        a = getattr(o, name)
        assert not np.isnan(a).any(), name
        if name != "cos_view":
            assert np.all(a[:5] == 0), name
    np.testing.assert_array_equal(
        o.cov[:5, 0], np.broadcast_to(np.diag([np.inf, np.inf]), (5, 2, 2))
    )
    assert np.isnan(o.uv[2]).all()  # проекции нет только за камерой
    assert np.isfinite(o.cov[5]).all() and np.all(o.weight[5, 0].diagonal() > 0)
    # информация точки по камерам сама пропускает невидимые
    info = np.einsum("pcji,pcjk,pckl->pil", o.jacobian, o.weight, o.jacobian)
    assert np.isfinite(info).all() and np.all(info[:5] == 0)


@pytest.mark.parametrize("theta", [10.0, 45.0, 70.0])
def test_eccentricity_corrected_on_axis(theta):
    z, r = 1.0, SHOT.radius
    model = ObsModel(sigma_n_deg=2.0, sigma_t=0.0)
    o = observe(model, PAIR, [0.0, 0.0, z], tilted(theta))
    s_n = math.radians(2.0)
    # наклон вокруг y: |d eps_u / d theta| = f r**2 cos(2 theta) / Z**2;
    # вокруг x: eps_v = f r**2 n_z n_y / Z**2, производная f r**2 cos(theta) / Z**2
    du = eccentricity_tilt_sensitivity(CAM.fx, r, math.radians(theta), z)
    dv = CAM.fx * r**2 * math.cos(math.radians(theta)) / z**2
    expected = s_n**2 * np.diag([du**2, dv**2])
    # без члена второго порядка по r / Z; при 45° по u — 0 (замер ~1e-14)
    scale = (s_n * CAM.fx * r**2 / z**2) ** 2
    np.testing.assert_allclose(o.cov_ecc[0], expected, rtol=1e-3, atol=1e-6 * scale)
    np.testing.assert_allclose(o.cov[0], o.cov_shot[0] + o.cov_ecc[0], rtol=1e-15)
    np.testing.assert_array_equal(o.bias[0], 0.0)
    # eps ~ 0,05 пикс. — разность проекций ~1e3 пикс.: округление 1e-13
    np.testing.assert_allclose(
        o.ecc[0],
        eccentricity_pole(CAM, np.array([0.0, 0.0, z]), tilted(theta), r),
        rtol=0,
        atol=1e-11,
    )


def test_eccentricity_jacobian_vs_central_differences():
    """Аналитическая J_n = d eps / d(два угла нормали) против разностей."""
    rng = np.random.default_rng(7)
    sn = math.radians(2.0)
    model = ObsModel(sigma_n_deg=2.0, sigma_t=0.0)
    for _ in range(15):
        X = np.array(
            point_at_pixel(CAM, *rng.uniform(200, 1800, 2), rng.uniform(0.5, 2))
        )
        n = rng.normal(size=3)
        n -= 1.5 * np.linalg.norm(n) * X / np.linalg.norm(X)  # к камере
        n /= np.linalg.norm(n)
        t1 = np.cross(n, [0.0, 0.0, 1.0])
        t1 /= np.linalg.norm(t1)
        h, cols = 1e-4, []
        for t in (t1, np.cross(n, t1)):
            plus, minus = n + h * t, n - h * t
            cols.append(
                (
                    eccentricity_pole(CAM, X, plus / np.linalg.norm(plus), 0.005)
                    - eccentricity_pole(CAM, X, minus / np.linalg.norm(minus), 0.005)
                )
                / (2 * h)
            )
        D = np.column_stack(cols)
        o = observe(model, PAIR, X, n)
        # разности: округление проекций ~1e-13 / h и член h**2
        scale = (sn * CAM.fx * 0.005**2 / X[2] ** 2) ** 2
        np.testing.assert_allclose(
            o.cov_ecc[0], sn**2 * D @ D.T, rtol=1e-6, atol=1e-7 * scale
        )


def test_eccentricity_uncorrected_is_bias():
    X = np.array(point_at_pixel(CAM, 2000.0, 400.0, 0.8))
    normal = tilted(35.0, axis="x")
    o = observe(ObsModel(eccentricity="uncorrected"), PAIR, X, normal)
    np.testing.assert_array_equal(o.cov_ecc[0], 0.0)
    np.testing.assert_allclose(
        o.bias[0], eccentricity_pole(CAM, X, normal, 0.005), rtol=1e-12
    )
    np.testing.assert_array_equal(o.bias[0], o.ecc[0])
    assert np.linalg.norm(o.bias[0]) > 0.05  # пикс.: не пренебрежимо


def test_sync():
    V = np.array([0.5, -0.2, 0.0])
    model = ObsModel(sigma_n_deg=0.0, sigma_t=20e-6)
    o = observe(model, PAIR, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], V)
    w = CAM.fx * V[:2]  # скорость образа на оси, Z = 1 м
    np.testing.assert_allclose(o.cov_sync[0], 20e-6**2 * np.outer(w, w), rtol=1e-12)
    assert o.blur_px[0] == pytest.approx(np.linalg.norm(w) * TAU, rel=1e-12)
    # поперёк движения синхронизация не добавляет ничего
    e_perp = np.array([-w[1], w[0]]) / np.linalg.norm(w)
    scale = 20e-6**2 * (w @ w)
    assert e_perp @ o.cov_sync[0] @ e_perp == pytest.approx(0.0, abs=1e-14 * scale)
    np.testing.assert_allclose(o.cov[0], o.cov_shot[0] + o.cov_sync[0], rtol=1e-15)


def test_constant_mode():
    X, n = [0.1, -0.05, 1.2], tilted(50.0)
    model = ObsModel()
    review = observe(model.as_constant(0.05), PAIR, X, n, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(review.cov[0], 0.05**2 * np.eye(2), rtol=1e-15)
    np.testing.assert_allclose(review.weight[0], np.eye(2) / 0.05**2, rtol=1e-12)
    fair = model.as_constant()
    assert fair.mode == "constant" and fair.sigma_c == pytest.approx(3.2e-3, rel=1e-2)
    # на оси для марки, развёрнутой к камере, — та же ковариация, что у модели
    on_axis = [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]
    np.testing.assert_allclose(
        observe(fair, PAIR, *on_axis).cov, observe(SHOT, PAIR, *on_axis).cov, rtol=1e-12
    )
    # видимость та же, что у модели
    assert not observe(fair, PAIR, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]).visible[0]


def test_weight_is_inverse():
    o = observe(ObsModel(), PAIR, [0.05, 0.02, 1.0], tilted(30.0, "x"), [0.4, 0.3, 0.1])
    np.testing.assert_allclose(o.weight[0] @ o.cov[0], np.eye(2), atol=1e-10)


def test_kappa_and_extra():
    X, n = [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]
    base = observe(SHOT, PAIR, X, n)
    k = observe(ObsModel(sigma_n_deg=0.0, kappa=1.3), PAIR, X, n)
    np.testing.assert_allclose(k.cov, 1.3 * base.cov, rtol=1e-12)
    extra = 0.7**2 + 1 / 12
    e = observe(ObsModel(sigma_n_deg=0.0, extra_var_px=extra), PAIR, X, n)
    np.testing.assert_allclose(
        e.cov[0] - base.cov[0], extra / base.n_e[0] * np.eye(2), rtol=1e-9, atol=0
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"mode": "fixed"},
        {"eccentricity": "none"},
        {"fill": 0.0},
        {"radius": -0.005},
        {"sigma_t": -1e-6},
        {"kappa": math.nan},
    ],
)
def test_bad_parameters(kw):
    with pytest.raises(ValueError):
        ObsModel(**kw)


def test_bad_inputs():
    with pytest.raises(ValueError, match="normals"):
        observe(SHOT, PAIR, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    with pytest.raises(TypeError, match="Camera, Pose"):
        observe(SHOT, [(POSE, CAM)], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0])
    with pytest.raises(ValueError, match="пустой"):
        observe(SHOT, [], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0])
