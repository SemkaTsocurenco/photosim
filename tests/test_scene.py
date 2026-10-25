import dataclasses
import math

import numpy as np
import pytest

from photosim._checks import check_rotation, is_rotation_matrix
from photosim.scene import (
    Accel,
    Arc,
    AxisRotation,
    FixedOrientation,
    Linear,
    Motion,
    Static,
    exposure_mean_position,
    exposure_samples,
    frame_from_normal,
    reference_time,
)

X0 = np.array([0.1, -0.2, 1.5])
V = np.array([0.3, 0.05, -0.1])
A = np.array([-0.4, 0.2, 0.1])

CENTER = np.array([0.0, 0.0, 2.0])
AXIS = np.array([0.0, 0.0, 1.0])
START = np.array([0.3, 0.0, 2.0])
OMEGA = 2.0

T_GRID = np.linspace(-1.0, 3.0, 17)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def tilted_arc(omega=1.3):
    axis = _unit([1.0, 1.0, 2.0])
    e = _unit(np.cross(axis, [1.0, 0.0, 0.0]))
    center = np.array([0.2, -0.1, 1.8])
    return Arc(center, axis, center + 0.25 * e, omega)


LAWS = {
    "static": Static(X0),
    "linear": Linear(X0, V),
    "accel": Accel(X0, V, A),
    "arc": Arc(CENTER, AXIS, START, OMEGA),
    "arc_tilted": tilted_arc(),
    "arc_negative": Arc(CENTER, AXIS, START, -1.5),
}


def law_param():
    return pytest.mark.parametrize("law", list(LAWS.values()), ids=list(LAWS))


def skew_to_vec(W):
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


# ===================================================================== центр


# 1. X(0) == X0 (для дуги == start)
@law_param()
def test_initial_position(law):
    expected = law.start if isinstance(law, Arc) else X0
    np.testing.assert_allclose(law.position(0.0), expected, rtol=0, atol=1e-15)


@law_param()
def test_time_shapes(law):
    assert law.position(0.5).shape == (3,)
    assert law.velocity(0.5).shape == (3,)
    assert law.acceleration(0.5).shape == (3,)
    assert law.position(T_GRID).shape == (T_GRID.size, 3)
    assert law.velocity(T_GRID).shape == (T_GRID.size, 3)
    assert law.acceleration(T_GRID).shape == (T_GRID.size, 3)
    # число и массив из одного элемента согласованы
    np.testing.assert_array_equal(law.position(0.5), law.position([0.5])[0])


def test_bad_time():
    law = LAWS["linear"]
    with pytest.raises(ValueError):
        law.position(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        law.position([0.0, np.nan])


# 2. Linear и Accel: формулы точно
def test_linear_formulas():
    law = LAWS["linear"]
    t = T_GRID[:, None]
    np.testing.assert_allclose(law.position(T_GRID), X0 + V * t, rtol=1e-15)
    np.testing.assert_array_equal(law.velocity(T_GRID), np.broadcast_to(V, (17, 3)))
    np.testing.assert_array_equal(law.acceleration(T_GRID), 0.0)


def test_accel_formulas():
    law = LAWS["accel"]
    t = T_GRID[:, None]
    np.testing.assert_allclose(
        law.position(T_GRID), X0 + V * t + A * t**2 / 2, rtol=1e-14, atol=1e-15
    )
    np.testing.assert_allclose(law.velocity(T_GRID), V + A * t, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(law.acceleration(T_GRID), np.broadcast_to(A, (17, 3)))


def test_static():
    law = LAWS["static"]
    np.testing.assert_array_equal(law.position(T_GRID), np.broadcast_to(X0, (17, 3)))
    np.testing.assert_array_equal(law.velocity(T_GRID), 0.0)
    np.testing.assert_array_equal(law.acceleration(T_GRID), 0.0)


# 3. Скорость и ускорение против конечных разностей
@law_param()
def test_velocity_finite_difference(law):
    h = 1e-6
    fd = (law.position(T_GRID + h) - law.position(T_GRID - h)) / (2 * h)
    np.testing.assert_allclose(law.velocity(T_GRID), fd, rtol=0, atol=1e-8)


@law_param()
def test_acceleration_finite_difference(law):
    # Вторая центральная разность: ошибка аппроксимации ~ h²/12·|X''''|,
    # ошибка округления ~ 4·eps·|X|/h². При h = 1e-4 обе около 1e-8 … 1e-7.
    h = 1e-4
    p = law.position
    fd = (p(T_GRID + h) - 2 * p(T_GRID) + p(T_GRID - h)) / h**2
    np.testing.assert_allclose(law.acceleration(T_GRID), fd, rtol=0, atol=1e-6)


# 4. Геометрия дуги
@pytest.mark.parametrize("arc", [LAWS["arc"], LAWS["arc_tilted"], LAWS["arc_negative"]])
def test_arc_geometry(arc):
    X = arc.position(T_GRID)
    Vt = arc.velocity(T_GRID)
    At = arc.acceleration(T_GRID)
    d = X - arc.center
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), arc.radius, rtol=1e-14)
    np.testing.assert_allclose(d @ arc.axis, 0.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(Vt * d, axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(Vt, axis=1), arc.speed, rtol=1e-14)
    np.testing.assert_allclose(
        np.linalg.norm(At, axis=1), arc.omega**2 * arc.radius, rtol=1e-14
    )
    np.testing.assert_allclose(arc.normal_acceleration, arc.omega**2 * arc.radius)
    # ускорение направлено к центру
    cos = np.sum(At * -d, axis=1) / (
        np.linalg.norm(At, axis=1) * np.linalg.norm(d, axis=1)
    )
    np.testing.assert_allclose(cos, 1.0, atol=1e-14)


# 5. Направление обхода
@pytest.mark.parametrize("omega", [2.0, -0.7])
def test_arc_direction(omega):
    arc = Arc(CENTER, AXIS, START, omega)
    X = arc.position(T_GRID)
    Vt = arc.velocity(T_GRID)
    s = np.cross(X - arc.center, Vt) @ arc.axis
    assert np.all(np.sign(s) == np.sign(omega))


def test_arc_ccw_from_above():
    # Ось z, старт на +x, omega > 0: через четверть оборота точка на +y.
    arc = Arc([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0)
    np.testing.assert_allclose(arc.position(math.pi / 2), [0, 1, 0], atol=1e-15)


# 6. Полный оборот
@pytest.mark.parametrize("arc", [LAWS["arc"], LAWS["arc_tilted"], LAWS["arc_negative"]])
def test_arc_full_turn(arc):
    T = 2 * math.pi / abs(arc.omega)
    np.testing.assert_allclose(arc.position(T), arc.start, atol=1e-14)


# 7. Ошибки конструктора дуги
def test_arc_start_off_plane():
    with pytest.raises(ValueError, match="плоскости"):
        Arc(CENTER, AXIS, START + [0, 0, 1e-3], OMEGA)


def test_arc_zero_axis():
    with pytest.raises(ValueError, match="axis"):
        Arc(CENTER, [0, 0, 0], START, OMEGA)


def test_arc_start_equals_center():
    with pytest.raises(ValueError, match="center"):
        Arc(CENTER, AXIS, CENTER, OMEGA)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"center": [0, np.nan, 2]},
        {"axis": [0, 0, np.inf]},
        {"start": [0.3, 0, np.nan]},
        {"omega": np.nan},
        {"omega": np.inf},
        {"center": [0, 0]},
    ],
)
def test_arc_bad_numbers(kwargs):
    args = {"center": CENTER, "axis": AXIS, "start": START, "omega": OMEGA, **kwargs}
    with pytest.raises(ValueError):
        Arc(**args)


def test_arc_axis_normalized_and_from_speed():
    arc = Arc(CENTER, [0, 0, 5.0], START, OMEGA)
    np.testing.assert_allclose(arc.axis, AXIS)
    fast = Arc.from_speed(CENTER, AXIS, START, 1.5)
    assert fast.speed == pytest.approx(1.5)
    assert fast.omega == pytest.approx(1.5 / 0.3)
    assert fast.normal_acceleration == pytest.approx(1.5**2 / 0.3)
    assert Arc.from_speed(CENTER, AXIS, START, -1.5).omega < 0


def test_laws_frozen_and_readonly():
    arc = LAWS["arc"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        arc.omega = 1.0
    with pytest.raises(ValueError):
        arc.center[0] = 1.0
    src = X0.copy()
    law = Static(src)
    src[0] = 99.0  # копия, а не ссылка
    assert law.X0[0] == X0[0]


# ================================================================ ориентация

R_TILT = frame_from_normal([1.0, -0.5, 0.3])

ORIENTATIONS = {
    "fixed": FixedOrientation(R_TILT),
    "rot_z": AxisRotation(R_TILT, AXIS, OMEGA),
    "rot_tilted_neg": AxisRotation(R_TILT, _unit([0.3, -1.0, 0.5]), -0.8),
    "rot_about_normal": AxisRotation(np.eye(3), [0, 0, 1], 1.1),
}


def orient_param():
    return pytest.mark.parametrize(
        "law", list(ORIENTATIONS.values()), ids=list(ORIENTATIONS)
    )


# 8. R_obj(t) — матрица поворота при любых t
@orient_param()
def test_orientation_is_rotation(law):
    t = np.linspace(-5, 5, 101)
    R = law.orientation(t)
    assert R.shape == (101, 3, 3)
    assert np.all(is_rotation_matrix(R))
    assert law.orientation(0.3).shape == (3, 3)
    assert law.normal(t).shape == (101, 3)
    assert law.normal(0.3).shape == (3,)
    assert law.angular_velocity(t).shape == (101, 3)
    np.testing.assert_allclose(np.linalg.norm(law.normal(t), axis=1), 1.0, rtol=1e-14)


def test_rotation_checks():
    assert not is_rotation_matrix(np.diag([1.0, 1.0, -1.0]))  # отражение
    assert not is_rotation_matrix(2 * np.eye(3))
    assert not is_rotation_matrix(np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="R0"):
        FixedOrientation(np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError):
        check_rotation(np.eye(2), "R")


# 9. from_normal
@pytest.mark.parametrize(
    "n",
    [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, -2, 0], [0.3, 0.4, 0.95], [1, -0.5, 0.3]],
)
def test_from_normal(n):
    fo = FixedOrientation.from_normal(n)
    assert is_rotation_matrix(fo.R0)
    np.testing.assert_allclose(fo.R0[:, 2], _unit(n), atol=1e-15)
    np.testing.assert_allclose(fo.normal(1.7), _unit(n), atol=1e-15)
    # детерминизм
    np.testing.assert_array_equal(fo.R0, FixedOrientation.from_normal(n).R0)


def test_from_normal_up_is_identity_and_ref():
    np.testing.assert_allclose(frame_from_normal([0, 0, 1]), np.eye(3), atol=1e-15)
    R = frame_from_normal([1, 0, 0], ref=[0, 0, 1])
    np.testing.assert_allclose(R[:, 2], [1, 0, 0])
    with pytest.raises(ValueError, match="параллелен"):
        frame_from_normal([1, 0, 0], ref=[2, 0, 0])


# 10. AxisRotation: начало, полный оборот, единичная нормаль
@pytest.mark.parametrize("name", ["rot_z", "rot_tilted_neg", "rot_about_normal"])
def test_rotation_start_and_full_turn(name):
    law = ORIENTATIONS[name]
    np.testing.assert_allclose(law.orientation(0.0), law.R0, atol=1e-15)
    T = 2 * math.pi / abs(law.omega)
    np.testing.assert_allclose(law.orientation(T), law.R0, atol=1e-13)


# 11. Вращение вокруг нормали: нормаль стоит, оси в плоскости крутятся
def test_rotation_about_own_normal():
    law = ORIENTATIONS["rot_about_normal"]
    t = np.linspace(0, 4, 9)
    R = law.orientation(t)
    np.testing.assert_allclose(
        R[:, :, 2], np.broadcast_to([0, 0, 1], (9, 3)), atol=1e-15
    )
    phi = law.omega * t
    x_expected = np.stack([np.cos(phi), np.sin(phi), 0 * phi], axis=1)
    np.testing.assert_allclose(R[:, :, 0], x_expected, atol=1e-14)
    assert not np.allclose(R[1:, :, 0], R[0, :, 0])


def test_rotation_axis_is_world_axis():
    # Ось вращения задана в мире: нормаль вдоль x крутится вокруг мировой z.
    R0 = frame_from_normal([1, 0, 0])
    law = AxisRotation(R0, [0, 0, 1], 1.0)
    np.testing.assert_allclose(law.normal(math.pi / 2), [0, 1, 0], atol=1e-15)
    # Ось, понятая в системе марки (Rot справа), дала бы другой результат.
    from scipy.spatial.transform import Rotation as SR

    right = R0 @ SR.from_rotvec([0, 0, math.pi / 2]).as_matrix()
    np.testing.assert_allclose(right[:, 2], [1, 0, 0], atol=1e-15)


# 12. Угловая скорость против конечных разностей ориентации
@orient_param()
def test_angular_velocity_finite_difference(law):
    h = 1e-6
    for t in [-1.0, 0.0, 0.37, 2.5]:
        Rt = law.orientation(t)
        W = (law.orientation(t + h) @ Rt.T - law.orientation(t - h) @ Rt.T) / (2 * h)
        np.testing.assert_allclose(W, -W.T, atol=1e-8)
        np.testing.assert_allclose(law.angular_velocity(t), skew_to_vec(W), atol=1e-8)


# ===================================================================== Motion

MOTIONS = {
    "static": Motion.static(X0, [0.2, -0.3, 1.0]),
    "linear": Motion.linear(X0, V, [0, 0, 1]),
    "rigid_arc": Motion.rigid_arc(CENTER, AXIS, START, OMEGA, [1, 0, 0.2]),
    "carriage_arc": Motion.carriage_arc(CENTER, AXIS, START, OMEGA, [0, 1, 1]),
    "custom": Motion(tilted_arc(), ORIENTATIONS["rot_tilted_neg"]),
}


def motion_param():
    return pytest.mark.parametrize("motion", list(MOTIONS.values()), ids=list(MOTIONS))


# 13. Обратимость
@motion_param()
def test_roundtrip(motion):
    rng = np.random.default_rng(0)
    t = np.linspace(-0.5, 1.5, 6)
    x = rng.uniform(-0.1, 0.1, size=(6, 7, 3))
    np.testing.assert_allclose(motion.to_local(motion.to_world(x, t), t), x, atol=1e-15)
    Xw = rng.uniform(-2, 2, size=(6, 4, 3))
    np.testing.assert_allclose(
        motion.to_world(motion.to_local(Xw, t), t), Xw, atol=1e-14
    )


# 14. Формы и совпадение с поточечным циклом
@motion_param()
def test_shapes_match_loop(motion):
    rng = np.random.default_rng(1)
    nt, h, w, ss = 4, 3, 5, 2
    t = np.linspace(0.1, 0.9, nt)
    x = rng.uniform(-0.05, 0.05, size=(nt, h, w, ss, ss, 3))
    out = motion.to_world(x, t)
    loc = motion.to_local(out, t)
    assert out.shape == x.shape
    assert loc.shape == x.shape
    R = motion.orientation(t)
    X = motion.position(t)
    expected = np.empty_like(x)
    for idx in np.ndindex(x.shape[:-1]):
        n = idx[0]
        expected[idx] = R[n] @ x[idx] + X[n]
    np.testing.assert_allclose(out, expected, atol=1e-15)


@motion_param()
def test_scalar_time(motion):
    x = np.random.default_rng(2).uniform(-0.05, 0.05, size=(5, 2, 3))
    out = motion.to_world(x, 0.4)
    assert out.shape == x.shape
    np.testing.assert_allclose(
        out, x @ motion.orientation(0.4).T + motion.position(0.4), atol=1e-15
    )
    assert motion.to_world([0.0, 0.0, 0.0], 0.4).shape == (3,)
    np.testing.assert_allclose(
        motion.to_world([0.0, 0.0, 0.0], 0.4), motion.position(0.4), atol=1e-15
    )


def test_shape_mismatch():
    motion = MOTIONS["linear"]
    with pytest.raises(ValueError, match="форму"):
        motion.to_world(np.zeros((3, 4, 3)), np.zeros(5))
    with pytest.raises(ValueError):
        motion.to_world(np.zeros(3), np.zeros(5))
    with pytest.raises(ValueError):
        motion.to_local(np.zeros((5, 2)), np.zeros(5))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_nan_points_pass_through():
    motion = MOTIONS["rigid_arc"]
    t = np.array([0.0, 0.5])
    pts = np.array([[[np.nan, 0, 0], [1, 2, 3]], [[0, 0, 0], [np.nan] * 3]])
    loc = motion.to_local(pts, t)
    assert np.isnan(loc[0, 0]).all()
    assert np.isnan(loc[1, 1]).all()
    assert np.isfinite(loc[0, 1]).all()
    assert np.isfinite(loc[1, 0]).all()


def test_motion_delegates():
    m = MOTIONS["rigid_arc"]
    t = np.linspace(0, 1, 5)
    np.testing.assert_array_equal(m.velocity(t), m.translation_law.velocity(t))
    np.testing.assert_array_equal(m.acceleration(t), m.translation_law.acceleration(t))
    np.testing.assert_array_equal(
        m.angular_velocity(t), np.broadcast_to(OMEGA * AXIS, (5, 3))
    )
    np.testing.assert_array_equal(m.normal(t), m.orientation(t)[:, :, 2])


# 15. rigid_arc: радиус-вектор в системе марки постоянен
def test_rigid_arc_is_rigid():
    m = MOTIONS["rigid_arc"]
    t = np.linspace(-2, 5, 41)
    R = m.orientation(t)
    r_local = np.einsum("nji,nj->ni", R, m.position(t) - CENTER)
    np.testing.assert_allclose(
        r_local, np.broadcast_to(r_local[0], r_local.shape), atol=1e-14
    )
    # то же через to_local: центр звена неподвижен в системе марки
    c_local = m.to_local(np.broadcast_to(CENTER, (t.size, 3)), t)
    np.testing.assert_allclose(c_local, -r_local, atol=1e-14)


# 16. carriage_arc: нормаль постоянна, центр по дуге
def test_carriage_arc():
    m = MOTIONS["carriage_arc"]
    t = np.linspace(-2, 5, 41)
    n = m.normal(t)
    np.testing.assert_allclose(
        n, np.broadcast_to(_unit([0, 1, 1]), n.shape), atol=1e-15
    )
    arc = Arc(CENTER, AXIS, START, OMEGA)
    np.testing.assert_allclose(m.position(t), arc.position(t), atol=1e-15)
    np.testing.assert_array_equal(m.angular_velocity(t), 0.0)


# ====================================================== время и формула обзора


# 17. exposure_samples
@pytest.mark.parametrize("nt", [1, 2, 7, 100])
def test_exposure_samples(nt):
    t_k, tau = 0.25, 0.004
    s = exposure_samples(t_k, tau, nt)
    assert s.shape == (nt,)
    assert np.all((s > t_k) & (s < t_k + tau))
    assert s.mean() == pytest.approx(t_k + tau / 2, rel=1e-14)
    np.testing.assert_allclose(np.diff(s), tau / nt, rtol=1e-9)


def test_exposure_samples_single_is_mid():
    assert exposure_samples(0.25, 0.004, 1)[0] == 0.25 + 0.002


@pytest.mark.parametrize(
    ("args", "exc"),
    [
        ((0.0, 0.0, 3), ValueError),
        ((0.0, -1.0, 3), ValueError),
        ((0.0, np.nan, 3), ValueError),
        ((0.0, 1.0, 0), ValueError),
        ((0.0, 1.0, 2.5), TypeError),
        ((np.inf, 1.0, 3), ValueError),
    ],
)
def test_exposure_samples_bad(args, exc):
    with pytest.raises(exc):
        exposure_samples(*args)


# 18. reference_time
def test_reference_time():
    assert reference_time(1.0, 0.2, "start") == 1.0
    assert reference_time(1.0, 0.2, "mid") == pytest.approx(1.1)
    with pytest.raises(ValueError, match="mode"):
        reference_time(1.0, 0.2, "end")


# 19. Главный тест: δa = a τ²/24 при малом ωτ
DELTA_ARC = Arc([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.4, 0.0, 2.0], 3.0)


@pytest.mark.parametrize("omega_tau", [0.01, 0.03, 0.1])
@pytest.mark.parametrize(
    "law", [DELTA_ARC, Motion(DELTA_ARC, FixedOrientation(np.eye(3)))]
)
def test_delta_a_formula(law, omega_tau):
    arc = DELTA_ARC
    tau = omega_tau / arc.omega
    t_k = 0.37
    mean = exposure_mean_position(law, t_k, tau, 1000)
    mid = law.position(t_k + tau / 2)
    delta = mid - mean
    formula = arc.normal_acceleration * tau**2 / 24
    assert np.linalg.norm(delta) == pytest.approx(formula, rel=0.01)
    # среднее смещено к центру окружности
    to_center = _unit(arc.center - mid)
    assert np.dot(_unit(mean - mid), to_center) > 1 - 1e-6


# 20. Точное среднее для дуги и где формула перестаёт работать
OMEGA_TAU_RANGE = np.geomspace(0.01, 2.0, 15)


@pytest.mark.parametrize("omega_tau", OMEGA_TAU_RANGE)
def test_exact_mean_matches_numeric(omega_tau):
    arc = DELTA_ARC
    tau = omega_tau / arc.omega
    t_k = 0.37
    exact = arc.mean_position_exact(t_k, tau)
    numeric = exposure_mean_position(arc, t_k, tau, 4000)
    delta = arc.exposure_offset_exact(tau)
    # ошибка правила средней точки относительно самого смещения ~ 1/nt²
    assert np.linalg.norm(numeric - exact) < 1e-5 * delta
    mid = arc.position(t_k + tau / 2)
    assert np.linalg.norm(mid - exact) == pytest.approx(delta, rel=1e-9)


def test_formula_error_grows():
    arc = DELTA_ARC
    rel = []
    for wt in OMEGA_TAU_RANGE:
        tau = wt / arc.omega
        exact = arc.exposure_offset_exact(tau)
        formula = arc.normal_acceleration * tau**2 / 24
        rel.append(formula / exact - 1)
        # асимптотика относительной ошибки: (ωτ)²/80
        if wt <= 0.2:
            assert formula / exact - 1 == pytest.approx(wt**2 / 80, rel=0.01)
    rel = np.array(rel)
    assert np.all(np.diff(rel) > 0)
    assert rel[0] < 1e-5
    assert rel[-1] > 0.05  # при ωτ = 2 формула завышает примерно на 5 %


# 21. Linear: среднее за экспозицию совпадает с положением в середине
def test_linear_mean_is_mid():
    law = LAWS["linear"]
    for t_k, tau, nt in [(0.0, 0.01, 1), (0.37, 0.004, 7), (-1.2, 0.5, 100)]:
        mean = exposure_mean_position(law, t_k, tau, nt)
        np.testing.assert_allclose(mean, law.position(t_k + tau / 2), atol=1e-15)


def test_accel_offset_is_exact():
    # Для постоянного ускорения δa = A τ²/24 точно; правило средней точки
    # даёт множитель (1 − 1/nt²).
    law = LAWS["accel"]
    t_k, tau, nt = 0.37, 0.2, 10
    mean = exposure_mean_position(law, t_k, tau, nt)
    delta = mean - law.position(t_k + tau / 2)
    np.testing.assert_allclose(delta, A * tau**2 / 24 * (1 - 1 / nt**2), atol=1e-15)
