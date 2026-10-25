"""Пресеты камер и сенсоров. Данные пока здесь, позже переедут в YAML.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from photosim.camera import Camera
from photosim.sensor import SensorModel

__all__ = [
    "CAMERAS",
    "REVIEW_16MM",
    "RPI_CM3",
    "SENSORS",
    "camera",
    "review_16mm",
    "rpi_cm3",
    "sensor",
]

#: Внутренние параметры камер (без дисторсии): фокус, мм; пиксель, мкм; кадр.
CAMERAS: dict[str, dict[str, Any]] = {
    # объектив 16 мм на IMX264
    "review_16mm": {"focal_mm": 16.0, "pixel_um": 3.45, "width": 2448, "height": 2048},
    # Camera Module 3, стандартный объектив 4,74 мм (у Wide — 2,75 мм);
    # IMX708, 4608×2592, пиксель 1,4 мкм — паспорт Raspberry Pi
    "rpi_cm3": {"focal_mm": 4.74, "pixel_um": 1.4, "width": 4608, "height": 2592},
}


def camera(name: str) -> Camera:
    """Камера из ``CAMERAS`` по имени."""
    return Camera.from_physical(**_lookup(CAMERAS, name, "камеры"))

#: Камера обзора: Sony IMX264
#: Источник: отчёт EMVA 1288 камеры FLIR Blackfly S BFS-U3-50S5M
#: (прошивка 2308.0.758.0, 20 °C, 525 нм),
#: https://softwareservices.flir.com/BFS-U3-50S5/latest/EMVA/EMVA.html
#: - full_well_e: saturation capacity 11 085 e⁻;
#: - read_noise_e: temporal dark noise 2,16 e⁻;
#: - gain_e_per_dn: 0,17 e⁻/ед. в 16-битном формате, АЦП 12 бит,
#:   0,17 * 16 = 2,72 e⁻/ед.
REVIEW_16MM: dict[str, Any] = {
    "full_well_e": 11_085.0,
    "read_noise_e": 2.16,
    "gain_e_per_dn": 2.72,
    "bits": 12,
    "black_level_dn": 0.0,  # смещение настраивается в камере
    # Оценка, не измерено. Уточнить по снимку наклонного края или точки.
    "psf_sigma_px": 0.7,
}

# Raspberry Pi Camera Module 3: Sony IMX708
RPI_CM3: dict[str, Any] = {
    "bits": 10,
    "black_level_dn": 64.0,
    # Оценка, не измерено.
    "psf_sigma_px": 0.7,
}


def review_16mm(tau_ref: float, fill: float = 0.7, **overrides: Any) -> SensorModel:
    return _with_fill({**REVIEW_16MM, **overrides}, tau_ref, fill)


def rpi_cm3(
    tau_ref: float,
    fill: float = 0.7,
    *,
    full_well_e: float,
    read_noise_e: float,
    **overrides: Any,
) -> SensorModel:
    """Сенсор Raspberry Pi Camera Module 3 (IMX708).
    """
    params = {
        **RPI_CM3,
        "full_well_e": full_well_e,
        "read_noise_e": read_noise_e,
        **overrides,
    }
    if "gain_e_per_dn" not in params:
        top = 2 ** params["bits"] - 1 - params["black_level_dn"]
        params["gain_e_per_dn"] = full_well_e / top
    return _with_fill(params, tau_ref, fill)


#: Пресеты сенсоров по имени: ``fn(tau_ref, fill, **overrides) -> SensorModel``.
SENSORS: dict[str, Callable[..., SensorModel]] = {
    "review_16mm": review_16mm,
    "rpi_cm3": rpi_cm3,
}


def sensor(
    name: str, tau_ref: float, fill: float = 0.7, **overrides: Any
) -> SensorModel:
    """Сенсор из ``SENSORS`` по имени."""
    return _lookup(SENSORS, name, "сенсора")(tau_ref, fill, **overrides)


def _lookup(table: dict[str, Any], name: str, what: str) -> Any:
    try:
        return table[name]
    except KeyError:
        raise KeyError(
            f"нет пресета {what} {name!r}, доступны: {', '.join(sorted(table))}"
        ) from None


def _with_fill(params: dict[str, Any], tau_ref: float, fill: float) -> SensorModel:
    if "flux_e_per_s" in params:
        raise TypeError("flux_e_per_s задаётся через fill и tau_ref, а не напрямую")
    sensor = SensorModel(flux_e_per_s=0.0, **params)
    return replace(sensor, flux_e_per_s=sensor.flux_for_fill(fill, tau_ref))
