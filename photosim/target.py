"""Марка: форма и яркость в собственной системе координат.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from photosim._checks import as_points

__all__ = ["Target", "TargetShape"]


@runtime_checkable
class TargetShape(Protocol):
    """Интерфейс марки, которым пользуется рендер.
    """

    def value(self, xy_local: ArrayLike) -> NDArray[np.float64]:
        """Яркость в точках плоскости марки: ``(..., 2)`` → ``(...)``."""
        ...

    @property
    def bounding_radius(self) -> float:
        """Радиус круга с центром в начале координат, вмещающего марку, м."""
        ...

    @property
    def is_rotationally_symmetric(self) -> bool:
        """``True``, если поворот вокруг оси z не меняет марку."""
        ...

    def boundary(self, n: int) -> NDArray[np.float64]:
        """``n`` точек контура в системе марки, форма ``(n, 3)``, z = 0."""
        ...


def _check_brightness(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} должна быть конечной, получено {value!r}")
    if v < 0.0:
        raise ValueError(
            f"{name} не может быть отрицательной, получено {value!r}"
        )
    return v


@dataclass(frozen=True)
class Target:
    """Круглая марка радиуса ``radius`` на однородном фоне.

    Параметры
    ---------
    radius : float
        Радиус круга, м. Конечен и строго больше нуля.
    value_inside : float
        Относительная яркость марки, отн. ед., конечна и ≥ 0. По умолчанию 1.0.
    value_outside : float
        Относительная яркость фона вокруг марки, отн. ед., конечна и ≥ 0.
        По умолчанию 0.0.
    """

    radius: float
    value_inside: float = 1.0
    value_outside: float = 0.0

    def __post_init__(self) -> None:
        r = float(self.radius)
        if not math.isfinite(r) or r <= 0.0:
            raise ValueError(
                f"radius должен быть конечным и больше 0, получено {self.radius!r}"
            )
        object.__setattr__(self, "radius", r)
        object.__setattr__(
            self, "value_inside", _check_brightness("value_inside", self.value_inside)
        )
        object.__setattr__(
            self,
            "value_outside",
            _check_brightness("value_outside", self.value_outside),
        )

    def value(self, xy_local: ArrayLike) -> NDArray[np.float64]:
        """Яркость марки в точках её плоскости.
        """
        pts = as_points(xy_local, 2, "xy_local")
        # hypot не переполняется на больших координатах и тихо даёт nan
        # для nan; сравнение nan < r даёт False, то есть фон.
        rho = np.hypot(pts[..., 0], pts[..., 1])
        inside = rho < self.radius
        return np.where(inside, self.value_inside, self.value_outside)

    @property
    def bounding_radius(self) -> float:
        """Радиус вмещающего круга, м. Для круглой марки равен ``radius``."""
        return self.radius

    @property
    def is_rotationally_symmetric(self) -> bool:
        """Круглая марка симметрична: важна только нормаль, не поворот в плоскости."""
        return True

    @property
    def area(self) -> float:
        """Площадь марки, м²: π·r²."""
        return math.pi * self.radius**2

    def boundary(self, n: int) -> NDArray[np.float64]:
        """Точки контура в собственной системе марки.
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise TypeError(f"n должно быть целым, получено {n!r}")
        if n < 1:
            raise ValueError(f"n должно быть не меньше 1, получено {n}")
        theta = 2.0 * np.pi * np.arange(n) / n
        pts = np.zeros((int(n), 3))
        pts[:, 0] = self.radius * np.cos(theta)
        pts[:, 1] = self.radius * np.sin(theta)
        return pts