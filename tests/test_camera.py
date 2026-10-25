"""Тесты модели камеры.

Запуск из корня проекта:  python -m pytest -v tests/test_camera.py
"""
import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from photosim.camera import Camera, Pose

SEED = 12345

# Камера из обзора: объектив 16 мм, пиксель 3.45 мкм, сенсор 2448×2048
FOCAL_MM, PIXEL_UM, W, H = 16, 3.45, 2448, 2048


# ---------------------------------------------------------------- фикстуры

@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def cam():
    return Camera.from_physical(FOCAL_MM, PIXEL_UM, W, H)


def random_points_in_front(rng, n, z_min=0.5, z_max=20.0):
    """N случайных точек перед камерой (Z > 0), в метрах."""
    xy = rng.uniform(-5.0, 5.0, size=(n, 2))
    z = rng.uniform(z_min, z_max, size=(n, 1))
    return np.hstack([xy, z])


def random_rotations(rng, n):
    """Равномерно случайные повороты: нормированный гауссов кватернион."""
    return Rotation.from_quat(rng.normal(size=(n, 4))).as_matrix()


# ---------------------------------------------------------------- 1

@pytest.mark.parametrize("Z", [0.1, 1.0, 7.3, 1000.0])
def test_optical_axis_projects_to_principal_point(cam, Z):
    uv = cam.project([0.0, 0.0, Z])
    # 0/Z = 0 и 0·fx + cx = cx без ошибок округления, поэтому сравнение точное
    np.testing.assert_array_equal(uv, [cam.cx, cam.cy])


# ---------------------------------------------------------------- 2

def test_backproject_project_roundtrip(cam, rng):
    n = 10_000
    uv = rng.uniform([0.0, 0.0], [W, H], size=(n, 2))
    depth = rng.uniform(0.1, 100.0, size=(n, 1))

    rays = cam.backproject(uv)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0, rtol=0, atol=1e-12)

    uv_back = cam.project(rays * depth)
    max_err = np.abs(uv_back - uv).max()
    assert max_err < 1e-9, f"ошибка туда-обратно {max_err:.3g} пикс."


# ---------------------------------------------------------------- 3

def test_from_physical_review_camera(cam):
    f_expected = 16e-3 / 3.45e-6          # 4637.68… пикс.
    assert cam.fx == pytest.approx(f_expected, rel=1e-12)
    assert cam.fx == pytest.approx(4637.7, abs=0.05)   # в обзоре округлено до 4640
    assert cam.fy == cam.fx                            # пиксели квадратные
    assert (cam.width, cam.height) == (W, H)


def test_principal_point_convention(cam):
    # Соглашение OpenCV: центр пикселя (i, j) имеет целые координаты,
    # кадр занимает [-0.5, W-0.5], значит его середина — (W-1)/2.
    assert cam.cx == pytest.approx((W - 1) / 2)
    assert cam.cy == pytest.approx((H - 1) / 2)


# ---------------------------------------------------------------- 4

def test_matches_opencv(cam, rng):
    cv2 = pytest.importorskip("cv2")

    X = random_points_in_front(rng, 1000)

    # K для OpenCV собираем независимо от cam.K, прямо из физических параметров
    f = 16e-3 / 3.45e-6
    K_cv = np.array([[f, 0.0, (W - 1) / 2],
                     [0.0, f, (H - 1) / 2],
                     [0.0, 0.0, 1.0]])

    uv_cv, _ = cv2.projectPoints(
        X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_cv, np.zeros(5)
    )
    uv_cv = uv_cv.reshape(-1, 2)

    max_err = np.abs(cam.project(X) - uv_cv).max()
    assert max_err < 1e-6, f"расхождение с OpenCV {max_err:.3g}"


# ---------------------------------------------------------------- 5

def test_jacobian_matches_finite_differences(cam, rng):
    n = 500
    X = random_points_in_front(rng, n, z_min=1.0)
    J = cam.jacobian(X)
    assert J.shape == (n, 2, 3)

    h = 1e-6  # метров
    J_num = np.empty_like(J)
    for k in range(3):                      # цикл по осям, а не по точкам
        dX = np.zeros(3)
        dX[k] = h
        J_num[:, :, k] = (cam.project(X + dX) - cam.project(X - dX)) / (2 * h)

    # относительная ошибка по норме Фробениуса для каждой точки
    rel = np.linalg.norm(J - J_num, axis=(1, 2)) / np.linalg.norm(J, axis=(1, 2))
    assert rel.max() < 1e-6, f"макс. относительная ошибка якобиана {rel.max():.3g}"


# ---------------------------------------------------------------- 6

def test_vectorized_equals_pointwise(cam, rng):
    X = random_points_in_front(rng, 200)

    uv_batch = cam.project(X)
    uv_single = np.array([cam.project(p) for p in X])
    assert uv_batch.shape == (200, 2) and uv_single.shape == (200, 2)
    np.testing.assert_allclose(uv_batch, uv_single, rtol=0, atol=1e-12)

    rays_batch = cam.backproject(uv_batch)
    rays_single = np.array([cam.backproject(p) for p in uv_batch])
    np.testing.assert_allclose(rays_batch, rays_single, rtol=0, atol=1e-15)

    J_batch = cam.jacobian(X)
    J_single = np.array([cam.jacobian(p) for p in X])
    assert J_single.shape == (200, 2, 3)
    np.testing.assert_allclose(J_batch, J_single, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- 7

def test_pose_camera_origin_roundtrip(rng):
    Rs = random_rotations(rng, 20)
    Ts = rng.uniform(-10.0, 10.0, size=(20, 3))

    for R, T in zip(Rs, Ts):
        pose = Pose(R, T)
        origin_world = pose.to_world(np.zeros(3))
        # центр камеры в мире: C = -Rᵀ·T
        np.testing.assert_allclose(origin_world, -R.T @ T, atol=1e-12)
        np.testing.assert_allclose(pose.to_camera(origin_world), np.zeros(3), atol=1e-12)


def test_pose_roundtrip_many_points(rng):
    pose = Pose(random_rotations(rng, 1)[0], rng.uniform(-10.0, 10.0, size=3))
    X_world = rng.uniform(-50.0, 50.0, size=(1000, 3))
    np.testing.assert_allclose(pose.to_world(pose.to_camera(X_world)), X_world, atol=1e-10)


# ---------------------------------------------------------------- доп.: политика Z ≤ 0

def test_points_behind_camera(cam):
    X = np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 0.0], [1.0, 1.0, -5.0]])
    uv, mask = cam.project(X, return_mask=True)
    assert mask.tolist() == [True, False, False]
    assert np.all(np.isfinite(uv[0]))
    assert np.all(np.isnan(uv[1:]))
    assert np.all(np.isnan(cam.jacobian(X)[1:]))


# ================================================================ произвольные ведущие измерения

def make_grid_with_bad_points(rng):
    """Сетка (4, 5, 3), часть точек позади камеры и в её плоскости."""
    grid = random_points_in_front(rng, 20).reshape(4, 5, 3)
    grid[0, 1, 2] = -3.0
    grid[2, 4, 2] = 0.0
    grid[3, 0, 2] = -0.1
    return grid


def test_project_grid_equals_flat(cam, rng):
    grid = make_grid_with_bad_points(rng)
    flat = grid.reshape(20, 3)

    uv_g, mask_g = cam.project(grid, return_mask=True)
    uv_f, mask_f = cam.project(flat, return_mask=True)

    assert uv_g.shape == (4, 5, 2) and mask_g.shape == (4, 5)
    np.testing.assert_array_equal(uv_g.reshape(20, 2), uv_f)   # nan == nan здесь считается равным
    np.testing.assert_array_equal(mask_g.reshape(20), mask_f)
    assert mask_g.sum() == 17
    assert np.all(np.isnan(uv_g[~mask_g])) and np.all(np.isfinite(uv_g[mask_g]))


def test_backproject_grid_equals_flat(cam, rng):
    uv = rng.uniform([0.0, 0.0], [W, H], size=(4, 5, 2))
    rays_g = cam.backproject(uv)
    assert rays_g.shape == (4, 5, 3)
    np.testing.assert_array_equal(rays_g.reshape(20, 3), cam.backproject(uv.reshape(20, 2)))


def test_jacobian_grid_equals_flat(cam, rng):
    grid = make_grid_with_bad_points(rng)
    J_g = cam.jacobian(grid)
    assert J_g.shape == (4, 5, 2, 3)
    np.testing.assert_array_equal(J_g.reshape(20, 2, 3), cam.jacobian(grid.reshape(20, 3)))


def test_single_point_shapes(cam):
    uv, mask = cam.project([0.1, 0.2, 3.0], return_mask=True)
    assert uv.shape == (2,) and mask.shape == () and bool(mask)
    assert cam.backproject([10.0, 20.0]).shape == (3,)
    assert cam.jacobian([0.1, 0.2, 3.0]).shape == (2, 3)


def test_empty_input(cam):
    assert cam.project(np.empty((0, 3))).shape == (0, 2)
    assert cam.jacobian(np.empty((0, 3))).shape == (0, 2, 3)


@pytest.mark.parametrize("bad", [np.zeros((5, 2)), np.zeros((4, 5, 4)), 3.0])
def test_wrong_last_dim_raises(cam, bad):
    with pytest.raises(ValueError, match="ожидалась форма"):
        cam.project(bad)


def test_pose_grid(rng):
    pose = Pose(random_rotations(rng, 1)[0], rng.uniform(-5.0, 5.0, size=3))
    grid = rng.uniform(-10.0, 10.0, size=(4, 5, 3))
    np.testing.assert_allclose(
        pose.to_camera(grid).reshape(20, 3), pose.to_camera(grid.reshape(20, 3)), atol=1e-12
    )


# ================================================================ проверка R в Pose

def test_pose_accepts_valid_rotation(rng):
    for R in random_rotations(rng, 50):
        Pose(R, np.zeros(3))   # не должно падать


@pytest.mark.parametrize(
    "R, message",
    [
        (2.0 * np.eye(3), "не ортогональна"),                          # масштаб
        (np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "не ортогональна"),  # сдвиг
        (np.eye(3)[[1, 0, 2]], "отражение"),                           # переставлены строки
        (np.diag([1.0, 1.0, -1.0]), "отражение"),                      # перепутан знак оси
        (np.full((3, 3), np.nan), "не ортогональна"),
    ],
    ids=["scale", "shear", "swapped_rows", "flipped_axis", "nan"],
)
def test_pose_rejects_non_rotation(R, message):
    with pytest.raises(ValueError, match=message):
        Pose(R, np.zeros(3))


def test_pose_rejects_wrong_shapes():
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        Pose(np.eye(4), np.zeros(3))
    with pytest.raises(ValueError, match="3 числа"):
        Pose(np.eye(3), np.zeros(4))


# ================================================================ соглашения Pose

def test_from_euler_means_world_to_camera():
    # углы задают ровно ту матрицу R, которая хранится в позе
    angles = [10.0, -25.0, 40.0]
    pose = Pose.from_euler(angles, T=[1.0, 2.0, 3.0])
    expected = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    np.testing.assert_allclose(pose.R, expected, atol=1e-15)
    np.testing.assert_allclose(pose.T, [1.0, 2.0, 3.0])


def test_center_property(rng):
    R = random_rotations(rng, 1)[0]
    C = rng.uniform(-10.0, 10.0, size=3)
    pose = Pose(R, -R @ C)
    np.testing.assert_allclose(pose.center, C, atol=1e-12)
    np.testing.assert_allclose(pose.to_camera(C), np.zeros(3), atol=1e-12)


def test_look_at_geometry(cam, rng):
    for _ in range(20):
        C = rng.uniform(-10.0, 10.0, size=3)
        target = rng.uniform(-1.0, 1.0, size=3)
        pose = Pose.look_at(C, target, up=[0.0, 0.0, 1.0])

        np.testing.assert_allclose(pose.center, C, atol=1e-12)

        # цель лежит на оптической оси перед камерой → в главной точке кадра
        t_cam = pose.to_camera(target)
        assert t_cam[2] > 0
        np.testing.assert_allclose(cam.project(t_cam), [cam.cx, cam.cy], atol=1e-6)

        # точка чуть выше цели (по мировому up) видна выше центра кадра: v < cy
        above = cam.project(pose.to_camera(target + [0.0, 0.0, 0.1]))
        assert above[1] < cam.cy


def test_look_at_degenerate():
    with pytest.raises(ValueError, match="параллельно up"):
        Pose.look_at([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="совпадают"):
        Pose.look_at([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


# ================================================================ дисторсия пока не поддержана

def test_zero_distortion_is_no_distortion():
    cam0 = Camera(1000, 1000, 500, 400, 1001, 801, dist=np.zeros(5))
    assert not cam0.has_distortion
    assert cam0.jacobian([0.1, 0.2, 3.0]).shape == (2, 3)


# ================================================================ дисторсия Брауна-Конради

DIST = [-0.12, 0.03, 0.001, -0.0015, 0.004]   # порядок OpenCV: k1, k2, p1, p2, k3


@pytest.fixture
def cam_d():
    c = Camera.from_physical(FOCAL_MM, PIXEL_UM, W, H)
    return Camera(c.fx, c.fy, c.cx, c.cy, W, H, dist=DIST)


def test_distort_by_hand(cam_d):
    # одна точка, формула посчитана вручную
    k1, k2, p1, p2, k3 = DIST
    x, y = 0.2, -0.1
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    np.testing.assert_allclose(cam_d._distort(np.array([[x, y]])), [[xd, yd]], rtol=1e-14)


def test_distortion_matches_opencv(cam_d, rng):
    cv2 = pytest.importorskip("cv2")
    X = random_points_in_front(rng, 1000, z_min=5.0)
    uv_cv, _ = cv2.projectPoints(
        X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam_d.K, np.array(DIST)
    )
    np.testing.assert_allclose(cam_d.project(X), uv_cv.reshape(-1, 2), rtol=0, atol=1e-6)


def test_wrong_order_gives_different_pixels(cam_d, rng):
    # подвох: k3 на третьем месте вместо пятого -> другие числа
    k1, k2, p1, p2, k3 = DIST
    wrong = Camera(cam_d.fx, cam_d.fy, cam_d.cx, cam_d.cy, W, H, dist=[k1, k2, k3, p1, p2])
    X = random_points_in_front(rng, 100, z_min=5.0)
    assert np.abs(wrong.project(X) - cam_d.project(X)).max() > 1.0


def test_four_coeffs_means_k3_zero():
    c = Camera(1000, 1000, 500, 400, 1001, 801, dist=[-0.1, 0.01, 0.001, 0.002])
    np.testing.assert_array_equal(c.dist, [-0.1, 0.01, 0.001, 0.002, 0.0])


@pytest.mark.parametrize("bad", [[0.1], [0.1, 0.2, 0.3], np.zeros(6), np.zeros(8)],
                         ids=["1", "3", "6", "8"])
def test_dist_length_checked(bad):
    with pytest.raises(ValueError, match="4 или 5"):
        Camera(1000, 1000, 500, 400, 1001, 801, dist=bad)


# ---------------------------------------------------------------- обратная дисторсия

STRONG = [-0.5, 0.0, 0.0, 0.0, 0.0]   # сильная бочка: r_max = sqrt(2/3)


def test_distort_jacobian_numeric(cam_d, rng):
    xy = rng.uniform(-0.5, 0.5, size=(200, 2))
    J = cam_d._distort_jacobian(xy)
    h = 1e-6
    for k in range(2):
        d = np.zeros(2)
        d[k] = h
        num = (cam_d._distort(xy + d) - cam_d._distort(xy - d)) / (2 * h)
        np.testing.assert_allclose(J[:, :, k], num, rtol=1e-7, atol=1e-9)


def test_undistort_inverts_distort(cam_d, rng):
    xy = rng.uniform(-0.6, 0.6, size=(10_000, 2))
    xy_back, valid = cam_d._undistort(cam_d._distort(xy))
    assert valid.all()
    assert np.abs(xy_back - xy).max() < 1e-12


def test_backproject_project_roundtrip_with_distortion(cam_d, rng):
    uv = rng.uniform([0.0, 0.0], [W, H], size=(10_000, 2))
    rays, valid = cam_d.backproject(uv, return_mask=True)
    assert valid.all()
    depth = rng.uniform(0.1, 100.0, size=(10_000, 1))
    assert np.abs(cam_d.project(rays * depth) - uv).max() < 1e-9


def test_undistort_matches_opencv(cam_d, rng):
    cv2 = pytest.importorskip("cv2")
    uv = rng.uniform([0.0, 0.0], [W, H], size=(1000, 2))
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-14)
    src = uv.reshape(-1, 1, 2)
    if hasattr(cv2, "undistortPointsIter"):      # OpenCV 4.x
        xy_cv = cv2.undistortPointsIter(src, cam_d.K, np.array(DIST), None, None, criteria)
    else:                                         # OpenCV 5.x
        xy_cv = cv2.undistortPoints(src, cam_d.K, np.array(DIST), criteria=criteria)
    xy_cv = xy_cv.reshape(-1, 2)
    rays = cam_d.backproject(uv)
    np.testing.assert_allclose(rays[:, :2] / rays[:, 2:], xy_cv, rtol=0, atol=1e-9)


def test_r_max_strong_barrel():
    c = Camera(1000, 1000, 500, 500, 1001, 1001, dist=STRONG)
    # 1 + 3 k1 r^2 = 0  ->  r^2 = 2/3
    assert c.r_max == pytest.approx(np.sqrt(2 / 3), rel=1e-12)


def test_r_max_infinite_when_monotonic(cam_d):
    assert cam_d.r_max == np.inf          # наш DIST монотонен везде
    assert make_cam_plain().r_max == np.inf


def make_cam_plain():
    return Camera(1000, 1000, 500, 500, 1001, 1001)


def test_beyond_r_max_is_nan():
    c = Camera(1000, 1000, 500, 500, 1001, 1001, dist=STRONG)
    r_max = c.r_max
    rd_max = r_max * (1 - 0.5 * r_max**2)    # дальше этого искаженного радиуса решения нет

    # project: точки дальше r_max выкидываются
    X = np.array([[0.9 * r_max, 0, 1], [1.1 * r_max, 0, 1]])
    uv, mask = c.project(X, return_mask=True)
    assert mask.tolist() == [True, False]
    assert np.all(np.isnan(uv[1]))

    # backproject: пиксели дальше rd_max дают nan, внутри - нормальный луч
    uv = np.array([[500 + 1000 * 0.9 * rd_max, 500], [500 + 1000 * 1.1 * rd_max, 500]])
    rays, mask = c.backproject(uv, return_mask=True)
    assert mask.tolist() == [True, False]
    assert np.all(np.isfinite(rays[0])) and np.all(np.isnan(rays[1]))


def test_not_converged_is_nan(cam_d, rng):
    xy_d = cam_d._distort(rng.uniform(-0.6, 0.6, size=(100, 2)))
    xy, valid = cam_d._undistort(xy_d, max_iter=1)   # одной итерации мало
    assert not valid.all()
    assert np.all(np.isnan(xy[~valid]))


def test_backproject_grid_with_distortion(cam_d, rng):
    uv = rng.uniform([0.0, 0.0], [W, H], size=(4, 5, 2))
    rays, mask = cam_d.backproject(uv, return_mask=True)
    assert rays.shape == (4, 5, 3) and mask.shape == (4, 5)
    np.testing.assert_array_equal(rays.reshape(20, 3), cam_d.backproject(uv.reshape(20, 2)))


# ---------------------------------------------------------------- якобиан с дисторсией

def test_jacobian_with_distortion_numeric(cam_d, rng):
    X = random_points_in_front(rng, 500, z_min=1.0)
    J = cam_d.jacobian(X)
    h = 1e-6
    J_num = np.empty_like(J)
    for k in range(3):
        dX = np.zeros(3)
        dX[k] = h
        J_num[:, :, k] = (cam_d.project(X + dX) - cam_d.project(X - dX)) / (2 * h)
    rel = np.linalg.norm(J - J_num, axis=(1, 2)) / np.linalg.norm(J, axis=(1, 2))
    assert rel.max() < 1e-6, f"макс. относительная ошибка {rel.max():.3g}"


def test_jacobian_distortion_grid_and_nan():
    c = Camera(1000, 1000, 500, 500, 1001, 1001, dist=STRONG)
    r = c.r_max
    X = np.array([[0.5 * r, 0, 1], [1.1 * r, 0, 1], [0, 0, -1]]).reshape(1, 3, 3)
    J = c.jacobian(X)
    assert J.shape == (1, 3, 2, 3)
    assert np.all(np.isfinite(J[0, 0]))
    assert np.all(np.isnan(J[0, 1:]))


# ---------------------------------------------------------------- неизменяемость и проверки Camera

def test_camera_is_frozen(cam_d):
    with pytest.raises(dataclasses.FrozenInstanceError):
        cam_d.dist = np.zeros(5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cam_d.fx = 1.0
    with pytest.raises(ValueError):            # массив только для чтения
        cam_d.dist[0] = 0.0


def test_dist_is_copied():
    d = np.array(DIST)
    c = Camera(1000, 1000, 500, 500, 1001, 1001, dist=d)
    d[0] = 0.0                                 # меняем свой массив
    assert c.dist[0] == DIST[0]                # камера не изменилась


def test_r_max_computed_once(monkeypatch, cam_d, rng):
    calls = []
    real_roots = np.roots
    monkeypatch.setattr(np, "roots", lambda c: calls.append(1) or real_roots(c))
    c = Camera(1000, 1000, 500, 500, 1001, 1001, dist=STRONG)
    X = random_points_in_front(rng, 10)
    for _ in range(5):
        c.project(X)
        c.backproject(c.project(X))
        c.jacobian(X)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "args",
    [(0, 1000, 500, 400, 1001, 801), (1000, -1, 500, 400, 1001, 801),
     (1000, 1000, 500, 400, 0, 801), (1000, 1000, 500, 400, 1001, -5),
     (np.nan, 1000, 500, 400, 1001, 801)],
    ids=["fx0", "fy_neg", "w0", "h_neg", "fx_nan"],
)
def test_camera_rejects_bad_params(args):
    with pytest.raises(ValueError, match="> 0"):
        Camera(*args)


# ---------------------------------------------------------------- look_at: нормировка up

def test_look_at_up_scale_does_not_matter():
    a = Pose.look_at([3.0, -4.0, 2.0], [0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])
    b = Pose.look_at([3.0, -4.0, 2.0], [0.0, 0.0, 0.0], up=[0.0, 0.0, 1e-8])
    np.testing.assert_allclose(a.R, b.R, atol=1e-12)


def test_look_at_bad_up():
    with pytest.raises(ValueError, match="нулевым"):
        Pose.look_at([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], up=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="параллельно up"):   # угол ~1e-7 рад
        Pose.look_at([0.0, 0.0, 5.0], [5e-7, 0.0, 0.0], up=[0.0, 0.0, 1.0])


# ================================================================ сетка лучей

ROI = (100, 200, 8, 6)       # u0, v0, w, h


def test_ray_grid_shapes(cam_d):
    rays, mask = cam_d.ray_grid(ROI, 4)
    assert rays.shape == (6, 8, 4, 4, 3)
    assert mask.shape == (6, 8, 4, 4)
    assert rays.mean(axis=(2, 3)).shape == (6, 8, 3)   # усреднение по субпикселям


def test_ray_grid_ss1_equals_backproject(cam_d):
    rays, _ = cam_d.ray_grid(ROI, 1)
    u0, v0, w, h = ROI
    U, V = np.meshgrid(u0 + np.arange(w), v0 + np.arange(h))   # (h, w)
    expected = cam_d.backproject(np.stack([U, V], axis=-1))
    np.testing.assert_array_equal(rays[:, :, 0, 0], expected)


@pytest.mark.parametrize("ss", [1, 2, 7, 16])
def test_subpixel_centers_mean_is_pixel_center(cam, ss):
    uv = cam.subpixel_centers(ROI, ss)
    u0, v0, w, h = ROI
    U, V = np.meshgrid(u0 + np.arange(w), v0 + np.arange(h))
    np.testing.assert_allclose(uv.mean(axis=(2, 3)), np.stack([U, V], axis=-1),
                               rtol=0, atol=1e-12)
    # все центры внутри своего пикселя, шаг 1/ss
    d = uv - np.stack([U, V], axis=-1)[:, :, None, None, :]
    assert np.all(np.abs(d) < 0.5)
    if ss > 1:
        np.testing.assert_allclose(np.diff(uv[..., 0], axis=3), 1 / ss, atol=1e-12)
        np.testing.assert_allclose(np.diff(uv[..., 1], axis=2), 1 / ss, atol=1e-12)


def test_subpixel_centers_ss1_is_integer(cam):
    uv = cam.subpixel_centers((0, 0, 3, 2), 1)
    assert uv[0, 0, 0, 0].tolist() == [0.0, 0.0]
    assert uv[1, 2, 0, 0].tolist() == [2.0, 1.0]       # (u, v) = (столбец, строка)


def test_ray_grid_projects_back(cam_d, rng):
    ss = 7
    rays, mask = cam_d.ray_grid(ROI, ss)
    assert mask.all()
    depth = rng.uniform(0.1, 100.0, size=mask.shape + (1,))
    uv = cam_d.project(rays * depth)
    np.testing.assert_allclose(uv, cam_d.subpixel_centers(ROI, ss), rtol=0, atol=1e-9)


def test_ray_grid_cached(cam_d):
    a = cam_d.ray_grid(ROI, 4)
    assert cam_d.ray_grid(ROI, 4) is a
    assert cam_d.ray_grid(list(ROI), 4) is a           # тот же ключ в другом виде
    assert cam_d.ray_grid(ROI, 5) is not a
    assert cam_d.ray_grid(ROI, 4, dtype=np.float32) is not a


def test_ray_grid_cache_is_bounded(cam):
    for ss in range(1, Camera.RAY_CACHE_SIZE + 5):
        cam.ray_grid(ROI, ss)
    assert len(cam._ray_cache) == Camera.RAY_CACHE_SIZE


def test_ray_grid_readonly(cam_d):
    rays, mask = cam_d.ray_grid(ROI, 2)
    with pytest.raises(ValueError):
        rays[0, 0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        mask[0, 0, 0, 0] = False


def test_ray_grid_float32(cam_d):
    r64, _ = cam_d.ray_grid(ROI, 4)
    r32, _ = cam_d.ray_grid(ROI, 4, dtype=np.float32)
    assert r32.dtype == np.float32
    np.testing.assert_allclose(r32, r64, rtol=0, atol=1e-7)


def test_ray_grid_memory(cam):
    rays, mask = cam.ray_grid((0, 0, 64, 64), 16)
    assert rays.nbytes == 64 * 64 * 16 * 16 * 3 * 8          # 24 МиБ
    assert mask.nbytes == 64 * 64 * 16 * 16                   # 1 МиБ


def test_ray_grid_mask_beyond_r_max():
    c = Camera(1000, 1000, 500, 500, 1001, 1001, dist=STRONG)
    # угол кадра ~707 пикс. от центра, а складка ~544 пикс.: там лучей нет
    rays, mask = c.ray_grid((0, 0, 40, 40), 2)
    assert not mask[0, 0].any()
    assert np.all(np.isnan(rays[~mask]))
    assert np.all(np.isfinite(rays[mask]))
    _, mask_c = c.ray_grid((480, 480, 40, 40), 2)          # центр кадра
    assert mask_c.all()


def test_ray_grid_roi_at_edge_ok(cam):
    rays, _ = cam.ray_grid((W - 4, H - 3, 4, 3), 2)
    assert rays.shape == (3, 4, 2, 2, 3)


@pytest.mark.parametrize(
    "roi",
    [(W - 3, 0, 4, 4), (0, H - 1, 4, 2), (-1, 0, 4, 4), (0, -2, 4, 4)],
    ids=["right", "bottom", "left", "top"],
)
def test_ray_grid_roi_outside_rejected(cam, roi):
    with pytest.raises(ValueError, match="выходит за кадр"):
        cam.ray_grid(roi, 2)


@pytest.mark.parametrize(
    "roi, ss",
    [((0, 0, 0, 4), 2), ((0, 0, 4, 4), 0), ((0, 0, 4, 4), 1.5), ((0, 0, 4), 2)],
    ids=["w0", "ss0", "ss_float", "short_roi"],
)
def test_ray_grid_bad_args(cam, roi, ss):
    with pytest.raises(ValueError):
        cam.ray_grid(roi, ss)
