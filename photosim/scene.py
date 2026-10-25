"""Движение марки: где находится марка и как она повёрнута в момент t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation as _SciRotation

from photosim._checks import (
    as_points,
    as_time,
    as_vec3,
    check_rotation,
    finite_scalar,
    readonly,
    squeeze_time,
    unit_vec3,
)

__all__ = [
    "Accel",
    "Arc",
    "AxisRotation",
    "FacingCamera",
    "FixedOrientation",
    "Linear",
    "Motion",
    "OrientationLaw",
    "Static",
    "TranslationLaw",
    "exposure_mean_position",
    "exposure_samples",
    "frame_from_normal",
    "normal_from_orientation",
    "reference_time",
]

#: Допуск на перпендикулярность (start - center) к оси дуги, доля радиуса.
_PERP_RTOL = 1e-9

_ZERO3 = readonly(np.zeros(3))


# Интерфейсы


@runtime_checkable
class TranslationLaw(Protocol):
    """Закон движения центра марки. Всё в мировой системе."""

    def position(self, t: ArrayLike) -> NDArray:
        """Положение центра"""
        ...

    def velocity(self, t: ArrayLike) -> NDArray:
        """Скорость центра, м/с """
        ...

    def acceleration(self, t: ArrayLike) -> NDArray:
        """Ускорение центра, м/с²"""
        ...


@runtime_checkable
class OrientationLaw(Protocol):
    """Закон изменения ориентации марки."""

    def orientation(self, t: ArrayLike) -> NDArray:
        """``R_obj(t)`` из системы марки в мир"""
        ...

    def angular_velocity(self, t: ArrayLike) -> NDArray:
        """Угловая скорость в мировой системе, рад/с"""
        ...

    def normal(self, t: ArrayLike) -> NDArray:
        """Нормаль марки в мире"""
        ...


def normal_from_orientation(R: NDArray) -> NDArray:
    """Нормаль марки — третий столбе"""
    return R[..., :, 2]


def _const(vec: NDArray, t: ArrayLike) -> NDArray:
    """Постоянный вектор с правилом форм по времени."""
    ts, scalar = as_time(t)
    out = np.broadcast_to(vec, (ts.shape[0], *vec.shape)).copy()
    return squeeze_time(out, scalar)


class _NormalFromOrientation:
    """Общая реализация ``normal`` через ``orientation``."""

    def normal(self, t: ArrayLike) -> NDArray:
        return normal_from_orientation(self.orientation(t))  # type: ignore[attr-defined]


# Законы движения центра


@dataclass(frozen=True, eq=False)
class Static:
    """Неподвижный центр: ``X(t) = X0``, скорость и ускорение равны нулю."""

    X0: NDArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "X0", as_vec3(self.X0, "X0"))

    def position(self, t: ArrayLike) -> NDArray:
        return _const(self.X0, t)

    def velocity(self, t: ArrayLike) -> NDArray:
        return _const(_ZERO3, t)

    def acceleration(self, t: ArrayLike) -> NDArray:
        return _const(_ZERO3, t)


@dataclass(frozen=True, eq=False)
class Linear:
    """Равномерное движение: ``X(t) = X0 + V t``, ``V(t) = V``, ``A(t) = 0``."""

    X0: NDArray
    V: NDArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "X0", as_vec3(self.X0, "X0"))
        object.__setattr__(self, "V", as_vec3(self.V, "V"))

    def position(self, t: ArrayLike) -> NDArray:
        ts, scalar = as_time(t)
        return squeeze_time(self.X0 + self.V * ts[:, None], scalar)

    def velocity(self, t: ArrayLike) -> NDArray:
        return _const(self.V, t)

    def acceleration(self, t: ArrayLike) -> NDArray:
        return _const(_ZERO3, t)


@dataclass(frozen=True, eq=False)
class Accel:
    """Равноускоренное движение.

    ``X(t) = X0 + V0 t + A t²/2``, ``V(t) = V0 + A t``, ``A(t) = A``.
    """

    X0: NDArray
    V0: NDArray
    A: NDArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "X0", as_vec3(self.X0, "X0"))
        object.__setattr__(self, "V0", as_vec3(self.V0, "V0"))
        object.__setattr__(self, "A", as_vec3(self.A, "A"))

    def position(self, t: ArrayLike) -> NDArray:
        ts, scalar = as_time(t)
        tt = ts[:, None]
        return squeeze_time(self.X0 + self.V0 * tt + 0.5 * self.A * tt**2, scalar)

    def velocity(self, t: ArrayLike) -> NDArray:
        ts, scalar = as_time(t)
        return squeeze_time(self.V0 + self.A * ts[:, None], scalar)

    def acceleration(self, t: ArrayLike) -> NDArray:
        return _const(self.A, t)


def _one_minus_sinc(h: float) -> float:
    """``1 - sin(h)/h`` без потери точности при малых ``h``."""
    if abs(h) < 1e-2:
        h2 = h * h
        return h2 / 6.0 - h2 * h2 / 120.0 + h2 * h2 * h2 / 5040.0
    return 1.0 - math.sin(h) / h


@dataclass(frozen=True, eq=False)
class Arc:
    """Движение центра по окружности.

    Параметры
    ---------
    center : (3,)
        Центр окружности, м.
    axis : (3,)
        Ось вращения (нормаль плоскости траектории), ненулевая; внутри
        нормируется.
    start : (3,)
        Положение центра марки при ``t = 0``, м. Должно лежать в плоскости
        окружности: ``(start - center) ⊥ axis``. Если нет, это
        ``ValueError``, а не молчаливое проецирование: исправленная за
        пользователя геометрия хуже явной ошибки.
    omega : float
        Угловая скорость, рад/с. ``omega > 0`` — против часовой стрелки,
        если смотреть с конца ``axis``.
    """

    center: NDArray
    axis: NDArray
    start: NDArray
    omega: float
    radius: float = field(init=False)
    e1: NDArray = field(init=False, repr=False)
    e2: NDArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        c = as_vec3(self.center, "center")
        a = unit_vec3(self.axis, "axis")
        s = as_vec3(self.start, "start")
        w = finite_scalar(self.omega, "omega")
        d = s - c
        r = float(np.linalg.norm(d))
        scale = max(1.0, float(np.linalg.norm(c)))
        if r <= 1e-12 * scale:
            raise ValueError(
                f"start совпадает с center ({c}): радиус окружности равен нулю"
            )
        off = float(np.dot(d, a))
        if abs(off) > _PERP_RTOL * r:
            raise ValueError(
                "start не лежит в плоскости окружности: проекция (start - center) "
                f"на axis равна {off:.3e} м при радиусе {r:.3e} м. Задайте start "
                "в плоскости, которая проходит через center перпендикулярно axis."
            )
        e1 = d / r
        e2 = np.cross(a, e1)
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "axis", a)
        object.__setattr__(self, "start", s)
        object.__setattr__(self, "omega", w)
        object.__setattr__(self, "radius", r)
        object.__setattr__(self, "e1", readonly(e1))
        object.__setattr__(self, "e2", readonly(e2))

    @classmethod
    def from_speed(
        cls, center: ArrayLike, axis: ArrayLike, start: ArrayLike, speed: float
    ) -> Arc:
        """Дуга по линейной скорости, м/с. Знак ``speed`` задаёт направление."""
        d = as_vec3(start, "start") - as_vec3(center, "center")
        r = float(np.linalg.norm(d))
        v = finite_scalar(speed, "speed")
        if r == 0.0:
            raise ValueError("start совпадает с center: радиус окружности равен нулю")
        return cls(center, axis, start, v / r)

    @property
    def speed(self) -> float:
        """Модуль линейной скорости, м/с: ``|ω| r``."""
        return abs(self.omega) * self.radius

    @property
    def normal_acceleration(self) -> float:
        """Нормальное (центростремительное) ускорение, м/с²: ``ω² r``."""
        return self.omega**2 * self.radius

    def _phase(self, t: ArrayLike) -> tuple[NDArray, NDArray, bool]:
        ts, scalar = as_time(t)
        phi = self.omega * ts
        return np.cos(phi)[:, None], np.sin(phi)[:, None], scalar

    def position(self, t: ArrayLike) -> NDArray:
        c, s, scalar = self._phase(t)
        out = self.center + self.radius * (c * self.e1 + s * self.e2)
        return squeeze_time(out, scalar)

    def velocity(self, t: ArrayLike) -> NDArray:
        c, s, scalar = self._phase(t)
        out = self.radius * self.omega * (-s * self.e1 + c * self.e2)
        return squeeze_time(out, scalar)

    def acceleration(self, t: ArrayLike) -> NDArray:
        c, s, scalar = self._phase(t)
        out = -self.radius * self.omega**2 * (c * self.e1 + s * self.e2)
        return squeeze_time(out, scalar)

    def mean_position_exact(self, t_k: float, tau: float) -> NDArray:
        """Точное среднее положение центра за экспозицию ``[t_k, t_k + tau]``.
        """
        tk = finite_scalar(t_k, "t_k")
        ta = _positive(tau, "tau")
        h = 0.5 * self.omega * ta
        phim = self.omega * (tk + 0.5 * ta)
        k = 1.0 - _one_minus_sinc(h)
        return self.center + self.radius * k * (
            math.cos(phim) * self.e1 + math.sin(phim) * self.e2
        )

    def exposure_offset_exact(self, tau: float) -> float:
        """Точное смещение ``|X(t_k + τ/2) − <X>|`` за экспозицию, м.
        """
        ta = _positive(tau, "tau")
        return self.radius * _one_minus_sinc(0.5 * self.omega * ta)


# Законы ориентации

_REF_Z = readonly(np.array([0.0, 0.0, 1.0]))
_REF_Y = readonly(np.array([0.0, 1.0, 0.0]))


def frame_from_normal(normal: ArrayLike, ref: ArrayLike | None = None) -> NDArray:
    """Ориентация марки по нормали: ``R0 = [x | y | n]``, тройка правая.

    ``x = normalize(cross(ref, n))``, ``y = cross(n, x)``. По умолчанию
    ``ref`` — мировая ось z, а при ``|n_z| >= 0.9`` — мировая ось y (иначе
    ``ref`` почти параллелен ``n``). Из-за этого переключения при плавном
    изменении нормали оси x и y в плоскости марки на ``|n_z| = 0.9`` меняются
    скачком. Для круглой марки это не важно, для кодовой важно: тогда
    задавайте ``ref`` явно. ``ref``, параллельный ``n``, даёт ``ValueError``.
    """
    n = unit_vec3(normal, "normal")
    if ref is None:
        r = _REF_Z if abs(n[2]) < 0.9 else _REF_Y
    else:
        r = unit_vec3(ref, "ref")
    x = np.cross(r, n)
    nx = float(np.linalg.norm(x))
    if nx < 1e-9:
        raise ValueError(
            f"ref {r} параллелен normal {n}: оси в плоскости марки не определены"
        )
    x = x / nx
    y = np.cross(n, x)
    return readonly(np.column_stack([x, y, n]))


@dataclass(frozen=True, eq=False)
class FixedOrientation(_NormalFromOrientation):
    """Постоянная ориентация ``R_obj(t) = R0``, угловая скорость нулевая."""

    R0: NDArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "R0", check_rotation(self.R0, "R0"))

    @classmethod
    def from_normal(
        cls, normal: ArrayLike, ref: ArrayLike | None = None
    ) -> FixedOrientation:
        """Ориентация по нормали; оси в плоскости."""
        return cls(frame_from_normal(normal, ref))

    def orientation(self, t: ArrayLike) -> NDArray:
        return _const(self.R0, t)

    def angular_velocity(self, t: ArrayLike) -> NDArray:
        return _const(_ZERO3, t)


@dataclass(frozen=True, eq=False)
class AxisRotation(_NormalFromOrientation):
    """Вращение марки вокруг **мировой** оси ``axis`` с угловой скоростью ``omega``.

    Вращение меняет только ориентацию; центр марки задаёт закон движения.
    """

    R0: NDArray
    axis: NDArray
    omega: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "R0", check_rotation(self.R0, "R0"))
        object.__setattr__(self, "axis", unit_vec3(self.axis, "axis"))
        object.__setattr__(self, "omega", finite_scalar(self.omega, "omega"))

    def orientation(self, t: ArrayLike) -> NDArray:
        ts, scalar = as_time(t)
        rotvec = self.axis[None, :] * (self.omega * ts)[:, None]
        rot = _SciRotation.from_rotvec(rotvec).as_matrix()
        return squeeze_time(rot @ self.R0, scalar)

    def angular_velocity(self, t: ArrayLike) -> NDArray:
        return _const(readonly(self.omega * self.axis), t)


@dataclass(frozen=True, eq=False)
class FacingCamera(_NormalFromOrientation):
    """Марка всё время смотрит на центр камеры: ``n(t) = normalize(C - X(t))``.

    ``translation`` — закон движения центра марки, ``camera_center`` — ``C``
    в мире. Оси в плоскости — как в ``frame_from_normal``, но с постоянным
    ``ref``: если он не задан, выбирается один раз по нормали при ``t = 0``,
    чтобы оси не меняли скачком во времени. Угловая скорость аналитическая:
    ``omega = (x × x' + y × y' + n × n') / 2``.

    Вне оптической оси образ такой марки — не круг с центром в проекции
    центра: эксцентриситет остаётся, как у сферы.
    """

    translation: TranslationLaw
    camera_center: NDArray
    ref: NDArray | None = None

    def __post_init__(self) -> None:
        c = as_vec3(self.camera_center, "camera_center")
        object.__setattr__(self, "camera_center", c)
        if self.ref is None:
            d0 = c - self.translation.position(0.0)
            ref = _REF_Z if abs(d0[2]) < 0.9 * np.linalg.norm(d0) else _REF_Y
        else:
            ref = unit_vec3(self.ref, "ref")
        object.__setattr__(self, "ref", ref)
        self._frame(np.zeros(1))  # вырожденная геометрия при t = 0 — сразу ошибка

    def _frame(self, ts: NDArray) -> tuple[NDArray, ...]:
        """Расстояние до камеры ``(Nt, 1)``, оси ``n, x, y`` и ``|ref × n|``."""
        d = self.camera_center - self.translation.position(ts)
        r = np.linalg.norm(d, axis=-1, keepdims=True)
        if np.any(r == 0.0):
            raise ValueError(
                "центр марки совпал с центром камеры: нормаль не определена"
            )
        n = d / r
        xt = np.cross(self.ref, n)
        rho = np.linalg.norm(xt, axis=-1, keepdims=True)
        if np.any(rho < 1e-9):
            raise ValueError(
                f"нормаль стала параллельна ref {self.ref}: оси в плоскости марки "
                "не определены, задайте другой ref"
            )
        x = xt / rho
        return r, n, x, np.cross(n, x), rho

    def orientation(self, t: ArrayLike) -> NDArray:
        ts, scalar = as_time(t)
        _, n, x, y, _ = self._frame(ts)
        return squeeze_time(np.stack([x, y, n], axis=-1), scalar)

    def angular_velocity(self, t: ArrayLike) -> NDArray:
        ts, scalar = as_time(t)
        r, n, x, y, rho = self._frame(ts)
        dd = -self.translation.velocity(ts)  # d = C - X(t)
        dn = (dd - n * np.sum(n * dd, axis=-1, keepdims=True)) / r
        dxt = np.cross(self.ref, dn)
        dx = (dxt - x * np.sum(x * dxt, axis=-1, keepdims=True)) / rho
        dy = np.cross(dn, x) + np.cross(n, dx)
        w = 0.5 * (np.cross(x, dx) + np.cross(y, dy) + np.cross(n, dn))
        return squeeze_time(w, scalar)


# Motion


@dataclass(frozen=True, eq=False)
class Motion:
    """Движение марки: закон центра плюс закон ориентации.
    """

    translation_law: TranslationLaw
    orientation_law: OrientationLaw

    # -- центр
    def position(self, t: ArrayLike) -> NDArray:
        return self.translation_law.position(t)

    def velocity(self, t: ArrayLike) -> NDArray:
        return self.translation_law.velocity(t)

    def acceleration(self, t: ArrayLike) -> NDArray:
        return self.translation_law.acceleration(t)

    # -- ориентация
    def orientation(self, t: ArrayLike) -> NDArray:
        return self.orientation_law.orientation(t)

    def normal(self, t: ArrayLike) -> NDArray:
        return self.orientation_law.normal(t)

    def angular_velocity(self, t: ArrayLike) -> NDArray:
        return self.orientation_law.angular_velocity(t)

    # -- преобразования точек
    def _pose(
        self, points: ArrayLike, t: ArrayLike, name: str
    ) -> tuple[NDArray, NDArray, NDArray, bool]:
        ts, scalar = as_time(t)
        pts = as_points(points, 3, name)
        if not scalar and (pts.ndim < 2 or pts.shape[0] != ts.shape[0]):
            nt = ts.shape[0]
            raise ValueError(
                f"{name}: при t формы ({nt},) точки должны иметь форму "
                f"({nt}, ..., 3), получена {pts.shape}"
            )
        R = self.orientation_law.orientation(ts)
        X = self.translation_law.position(ts)
        if scalar:
            return pts, R[0], X[0], True
        # X (Nt, 3) -> (Nt, 1, ..., 1, 3) под форму точек
        X = X.reshape((X.shape[0],) + (1,) * (pts.ndim - 2) + (3,))
        return pts, R, X, False

    def to_world(self, x_local: ArrayLike, t: ArrayLike) -> NDArray:
        """Точки марки в мир: ``X = R_obj(t) @ x + X(t)``. Формы — см. класс."""
        pts, R, X, scalar = self._pose(x_local, t, "x_local")
        if scalar:
            return np.einsum("ij,...j->...i", R, pts) + X
        return np.einsum("nij,n...j->n...i", R, pts) + X

    def to_local(self, X_world: ArrayLike, t: ArrayLike) -> NDArray:
        """Мировые точки в систему марки: ``x = R_obj(t)ᵀ (X − X(t))``."""
        pts, R, X, scalar = self._pose(X_world, t, "X_world")
        if scalar:
            return np.einsum("ji,...j->...i", R, pts - X)
        return np.einsum("nji,n...j->n...i", R, pts - X)

    # -- фабрики ------------------------------------------------------------------
    @classmethod
    def static(cls, X0: ArrayLike, normal: ArrayLike) -> Motion:
        """Неподвижная марка."""
        return cls(Static(X0), FixedOrientation.from_normal(normal))

    @classmethod
    def linear(cls, X0: ArrayLike, V: ArrayLike, normal: ArrayLike) -> Motion:
        """Поступательное равномерное движение, ориентация постоянна."""
        return cls(Linear(X0, V), FixedOrientation.from_normal(normal))

    @classmethod
    def rigid_arc(
        cls,
        center: ArrayLike,
        axis: ArrayLike,
        start: ArrayLike,
        omega: float,
        normal0: ArrayLike,
    ) -> Motion:
        """Марка жёстко закреплена на звене, вращающемся вокруг ``axis``.
        """
        arc = Arc(center, axis, start, omega)
        R0 = frame_from_normal(normal0)
        return cls(arc, AxisRotation(R0, arc.axis, arc.omega))

    @classmethod
    def carriage_arc(
        cls,
        center: ArrayLike,
        axis: ArrayLike,
        start: ArrayLike,
        omega: float,
        normal0: ArrayLike,
    ) -> Motion:
        """Каретка на изогнутой направляющей: центр по дуге, ориентация постоянна."""
        return cls(
            Arc(center, axis, start, omega), FixedOrientation.from_normal(normal0)
        )


# Время экспозиции


def _positive(value: float, name: str) -> float:
    v = finite_scalar(value, name)
    if v <= 0.0:
        raise ValueError(f"{name} должно быть больше 0, получено {value!r}")
    return v


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} должно быть целым, получено {value!r}")
    if value < 1:
        raise ValueError(f"{name} должно быть не меньше 1, получено {value}")
    return int(value)


def exposure_samples(t_k: float, tau: float, nt: int) -> NDArray:
    """Моменты подшагов внутри экспозиции ``[t_k, t_k + tau]``, форма ``(nt,)``.
    """
    tk = finite_scalar(t_k, "t_k")
    ta = _positive(tau, "tau")
    n = _positive_int(nt, "nt")
    return tk + (np.arange(n) + 0.5) * (ta / n)


def reference_time(t_k: float, tau: float, mode: Literal["start", "mid"]) -> float:
    """Момент, к которому привязывается кадр.
    """
    tk = finite_scalar(t_k, "t_k")
    ta = _positive(tau, "tau")
    if mode == "start":
        return tk
    if mode == "mid":
        return tk + 0.5 * ta
    raise ValueError(f"mode должен быть 'start' или 'mid', получено {mode!r}")


def exposure_mean_position(
    motion: TranslationLaw, t_k: float, tau: float, nt: int
) -> NDArray:
    """Среднее положение центра за экспозицию
    """
    ts = exposure_samples(t_k, tau, nt)
    return np.asarray(motion.position(ts)).mean(axis=0)
