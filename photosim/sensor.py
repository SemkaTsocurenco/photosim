"""Сенсор: облучённость без шума → кадр в единицах АЦП.

Превращает результат рендера (облучённость в отн. ед. марки, усреднённую
по пикселю и по экспозиции) в кадр, каким его выдаёт камера. Геометрии
здесь нет: один рендер, затем сколько угодно реализаций шума.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter

from photosim._checks import finite_scalar

__all__ = ["ADC_BITS", "SensorModel"]

ADC_BITS = (8, 10, 12)

_PSF_TRUNCATE = 4.0

_FLAGS = ("psf", "shot_noise", "read_noise", "saturation", "quantization")


def _nonnegative(value: float, name: str) -> float:
    v = finite_scalar(value, name)
    if v < 0.0:
        raise ValueError(f"{name} не может быть отрицательным, получено {value!r}")
    return v


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


def _flag(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} должен быть bool, получено {value!r}")
    return bool(value)


def _as_irradiance(irradiance: ArrayLike) -> NDArray:
    irr = np.asarray(irradiance, dtype=np.float64)
    if irr.ndim != 2:
        raise ValueError(
            f"irradiance: ожидалась форма (h, w), а пришла {irr.shape}"
        )
    if not (np.isfinite(irr).all() and (irr >= 0.0).all()):
        raise ValueError("irradiance должна быть конечной и неотрицательной")
    return irr


def _check_rng(rng: np.random.Generator) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng должен быть np.random.Generator "
            f"(например, np.random.default_rng(seed)), получено {rng!r}"
        )
    return rng


@dataclass(frozen=True, kw_only=True)
class SensorModel:
    """Модель сенсора.

    Все параметры передаются по имени.

    flux_e_per_s : float
        Поток на пиксель при облучённости 1.0 (яркость марки), e⁻/с, >= 0.
        Подбирается через ``flux_for_fill``.
    dark_e_per_s : float
        Темновой ток, e⁻/с, >= 0. По умолчанию 0.
    read_noise_e : float
        Шум считывания, СКО, e⁻, >= 0.
    full_well_e : float
        Ёмкость пикселя, e⁻, > 0.
    gain_e_per_dn : float
        Коэффициент преобразования, e⁻ на единицу АЦП, > 0.
    bits : int
        Разрядность АЦП: 8, 10 или 12.
    black_level_dn : float
        Уровень чёрного, ед. АЦП, от 0 до ``2**bits − 1`` (не включая).
        По умолчанию 0.
    psf_sigma_px : float
        СКО гауссовой PSF, пикс., >= 0. Правильно её применяет рендер до
        усреднения по пикселю: ``render(psf_sigma_px=...)``, так делают
        эксперименты. Этап ``psf`` сенсора — приближение «после пикселя»:
        дискретная свёртка не меняет центр тяжести и не подавляет пиксельную
        фазу; с PSF в рендере его выключают (``psf=False``).
    psf, shot_noise, read_noise, saturation, quantization : bool
        Включают этапы конвейера по отдельности, по умолчанию все включены.
        При ``quantization=False`` выход — аналоговое значение в ед. АЦП:
        без округления и без ограничения диапазоном АЦП.
    """

    flux_e_per_s: float
    dark_e_per_s: float = 0.0
    read_noise_e: float
    full_well_e: float
    gain_e_per_dn: float
    bits: int
    black_level_dn: float = 0.0
    psf_sigma_px: float
    psf: bool = True
    shot_noise: bool = True
    read_noise: bool = True
    saturation: bool = True
    quantization: bool = True

    def __post_init__(self) -> None:
        set_ = object.__setattr__  # обход frozen, только здесь
        for name in (
            "flux_e_per_s",
            "dark_e_per_s",
            "read_noise_e",
            "black_level_dn",
            "psf_sigma_px",
        ):
            set_(self, name, _nonnegative(getattr(self, name), name))
        for name in ("full_well_e", "gain_e_per_dn"):
            set_(self, name, _positive(getattr(self, name), name))

        bits = self.bits
        if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)):
            raise TypeError(f"bits должно быть целым, получено {bits!r}")
        if bits not in ADC_BITS:
            raise ValueError(f"bits должно быть одним из {ADC_BITS}, получено {bits}")
        set_(self, "bits", int(bits))
        if self.black_level_dn >= self.max_dn:
            raise ValueError(
                f"black_level_dn = {self.black_level_dn} не меньше максимума "
                f"АЦП {self.max_dn}"
            )

        for name in _FLAGS:
            set_(self, name, _flag(getattr(self, name), name))

    @property
    def max_dn(self) -> int:
        """Наибольшее значение АЦП, ``2**bits − 1``."""
        return 2**self.bits - 1

    @property
    def required_margin_px(self) -> int:
        """Запас ROI для PSF: ``ceil(4 * sigma) + 1`` пикс., без PSF — 1.

        Ядро ``gaussian_filter`` обрезается на радиусе ``int(4 * sigma + 0.5)``,
        хвост гауссианы за ``4 sigma`` ~3e-5; ещё один пиксель — запас сверху.
        С таким ``render(margin_px=...)`` по краю ROI фон, и PSF не теряет свет.
        """
        return math.ceil(_PSF_TRUNCATE * self.psf_sigma_px) + 1 if self.psf else 1

    def flux_for_fill(self, fill: float, tau: float) -> float:
        """Поток, при котором облучённость 1.0 за ``tau`` даёт ``fill · full_well_e``.

        На практике экспозицию настраивают так, чтобы марка заполняла
        60–80 % ёмкости. Темновой ток входит в заряд пикселя, поэтому он
        вычитается.
        """
        f = _positive(fill, "fill")
        ta = _positive(tau, "tau")
        flux = f * self.full_well_e / ta - self.dark_e_per_s
        if flux <= 0.0:
            raise ValueError(
                f"темновой ток за tau = {ta:g} с уже даёт не меньше "
                f"{f:g} ёмкости: поток подобрать нельзя"
            )
        return flux

    def mean_electrons(self, irradiance: ArrayLike, tau: float) -> NDArray:
        """Среднее число электронов без шума (шаги 1–2), ``(h, w)``."""
        irr = _as_irradiance(irradiance)
        ta = _positive(tau, "tau")
        if self.psf:
            irr = gaussian_filter(
                irr, self.psf_sigma_px, mode="nearest", truncate=_PSF_TRUNCATE
            )
        return irr * self.flux_e_per_s * ta + self.dark_e_per_s * ta

    def expose(
        self, irradiance: ArrayLike, tau: float, rng: np.random.Generator
    ) -> NDArray:
        """Одна реализация кадра, ``(h, w)``, float64 в ед. АЦП."""
        rng = _check_rng(rng)
        mean_e = self.mean_electrons(irradiance, tau)
        return self._readout(mean_e, mean_e.shape, rng)

    def expose_many(
        self, irradiance: ArrayLike, tau: float, rng: np.random.Generator, n: int
    ) -> NDArray:
        """``n`` реализаций одним вызовом, ``(n, h, w)``, float64 в ед. АЦП.

        Шум генерируется сразу для всех реализаций, поэтому результат
        совпадает с ``n`` вызовами ``expose`` по распределению, но не бит
        в бит
        """
        rng = _check_rng(rng)
        n = _positive_int(n, "n")
        mean_e = self.mean_electrons(irradiance, tau)
        return self._readout(mean_e, (n, *mean_e.shape), rng)

    def _readout(
        self, mean_e: NDArray, shape: tuple[int, ...], rng: np.random.Generator
    ) -> NDArray:
        """Шаги 3–6; ``mean_e`` расширяется до ``shape`` по ведущим осям."""
        if self.shot_noise:
            e = rng.poisson(mean_e, size=shape).astype(np.float64)
        else:
            e = np.array(np.broadcast_to(mean_e, shape))
        if self.read_noise:
            e += rng.normal(0.0, self.read_noise_e, size=shape)
        if self.saturation:
            # только сверху: отрицательный шум считывания сохраняется над
            # уровнем чёрного, снизу обрезает АЦП
            np.minimum(e, self.full_well_e, out=e)
        e /= self.gain_e_per_dn
        e += self.black_level_dn
        if self.quantization:
            np.round(e, out=e)
            np.clip(e, 0.0, self.max_dn, out=e)
        return e
