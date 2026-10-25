import dataclasses
import math

import numpy as np
import pytest

from photosim.target import Target, TargetShape

R = 0.05  # 5 см


@pytest.fixture
def target():
    return Target(R)


# 1. Внутри и снаружи
def test_inside_outside(target):
    assert target.value([0.0, 0.0]) == 1.0
    assert target.value([0.99 * R, 0.0]) == 1.0
    assert target.value([1.01 * R, 0.0]) == 0.0


def test_boundary_point_is_outside(target):
    # записанное в docstring решение: граница строгая, x² + y² < r²
    assert target.value([R, 0.0]) == 0.0


# 2. Симметрия
@pytest.mark.parametrize("rho", [0.3 * R, 0.99 * R, 1.01 * R, 2.0 * R])
def test_rotational_symmetry(target, rho):
    phi = np.linspace(0.0, 2.0 * np.pi, 37)
    pts = np.stack([rho * np.cos(phi), rho * np.sin(phi)], axis=-1)
    vals = target.value(pts)
    assert np.all(vals == vals[0])


# 3. Формы
@pytest.mark.parametrize(
    "shape",
    [(2,), (10, 2), (4, 5, 3, 3, 7, 2)],
)
def test_shapes(target, shape):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2 * R, 2 * R, size=shape)
    assert target.value(pts).shape == shape[:-1]


def test_grid_equals_flat(target):
    rng = np.random.default_rng(1)
    pts = rng.uniform(-2 * R, 2 * R, size=(4, 5, 3, 3, 7, 2))
    grid = target.value(pts)
    flat = target.value(pts.reshape(-1, 2))
    np.testing.assert_array_equal(grid.ravel(), flat)


def test_wrong_last_dim_raises(target):
    with pytest.raises(ValueError):
        target.value(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        target.value(1.0)


# 4. nan на входе
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_nan_gives_outside(target):
    pts = np.array(
        [[np.nan, 0.0], [0.0, np.nan], [np.nan, np.nan], [np.inf, 0.0], [0.0, 0.0]]
    )
    vals = target.value(pts)
    np.testing.assert_array_equal(vals, [0.0, 0.0, 0.0, 0.0, 1.0])
    assert not np.any(np.isnan(vals))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_huge_coordinates_no_overflow(target):
    assert target.value([1e200, 1e200]) == 0.0


# 5. Площадь методом Монте-Карло
def test_area_monte_carlo(target):
    rng = np.random.default_rng(42)
    n = 10**6
    pts = rng.uniform(-R, R, size=(n, 2))
    frac = target.value(pts).mean()  # value_inside = 1, value_outside = 0
    estimate = frac * (2 * R) ** 2
    assert target.area == pytest.approx(math.pi * R**2)
    assert estimate == pytest.approx(target.area, rel=0.01)


# 6. boundary
@pytest.mark.parametrize("n", [1, 3, 64, 1000])
def test_boundary(target, n):
    b = target.boundary(n)
    assert b.shape == (n, 3)
    np.testing.assert_allclose(np.hypot(b[:, 0], b[:, 1]), R, rtol=1e-12)
    np.testing.assert_array_equal(b[:, 2], 0.0)
    if n > 1:
        # первая точка в конце не повторяется
        assert not np.allclose(b[0], b[-1])
        # точки равномерны по углу
        ang = np.unwrap(np.arctan2(b[:, 1], b[:, 0]))
        np.testing.assert_allclose(np.diff(ang), 2 * np.pi / n, rtol=1e-9)


@pytest.mark.parametrize("n", [0, -1])
def test_boundary_bad_n(target, n):
    with pytest.raises(ValueError):
        target.boundary(n)


def test_boundary_non_int(target):
    with pytest.raises(TypeError):
        target.boundary(3.5)


# 7. Проверки конструктора
@pytest.mark.parametrize("radius", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_bad_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        Target(radius)


@pytest.mark.parametrize("field", ["value_inside", "value_outside"])
@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_bad_brightness(field, bad):
    with pytest.raises(ValueError, match=field):
        Target(R, **{field: bad})


def test_equal_brightness_allowed():
    t = Target(R, value_inside=0.5, value_outside=0.5)
    pts = np.random.default_rng(3).uniform(-2 * R, 2 * R, size=(100, 2))
    np.testing.assert_array_equal(t.value(pts), 0.5)


# 8. Неизменяемость
def test_frozen(target):
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.radius = 1.0


# 9. Свойства
def test_properties(target):
    assert target.bounding_radius == R
    assert target.is_rotationally_symmetric is True
    assert isinstance(target, TargetShape)


# 10. Тёмная марка на светлом фоне
def test_dark_on_light():
    t = Target(R, value_inside=0.2, value_outside=0.9)
    assert t.value([0.0, 0.0]) == 0.2
    assert t.value([0.99 * R, 0.0]) == 0.2
    assert t.value([1.01 * R, 0.0]) == 0.9
    assert t.value([np.nan, 0.0]) == 0.9