import numpy as np
import pytest

from photosim.presets import (
    REVIEW_16MM,
    RPI_CM3_ASSUMED,
    camera,
    review_16mm,
    row_time_estimate,
    rpi_cm3,
    sensor,
)

TAU = 1e-3


@pytest.mark.parametrize(
    "name, f_px, size",
    [
        ("review_16mm", 16 / 3.45e-3, (2448, 2048)),
        ("rpi_cm3", 4.74 / 1.4e-3, (4608, 2592)),
        ("rpi_cm3_2x2", 4.74 / 2.8e-3, (2304, 1296)),
    ],
)
def test_camera_presets(name, f_px, size):
    cam = camera(name)
    assert cam.fx == pytest.approx(f_px) and cam.fy == cam.fx
    assert (cam.width, cam.height) == size
    assert not cam.has_distortion


def test_sensor_by_name():
    s = sensor("review_16mm", TAU, 0.6, psf=False)
    assert s.mean_electrons(np.ones((3, 3)), TAU) == pytest.approx(0.6 * s.full_well_e)
    assert not s.psf
    r = sensor("rpi_cm3", TAU, full_well_e=4000.0, read_noise_e=2.0)
    assert r.bits == 10
    with pytest.raises(KeyError, match="review_16mm"):
        sensor("nikon", TAU)
    with pytest.raises(KeyError, match="rpi_cm3"):
        camera("nikon")


def test_review_16mm_fill():
    s = review_16mm(TAU)
    assert s.bits == 12
    assert s.full_well_e == REVIEW_16MM["full_well_e"]
    # ёмкость попадает в шкалу 12-битного АЦП
    assert s.full_well_e / s.gain_e_per_dn <= s.max_dn
    ones = np.ones((9, 9))
    assert s.mean_electrons(ones, TAU) == pytest.approx(0.7 * s.full_well_e)
    s6 = review_16mm(TAU, fill=0.6)
    assert s6.mean_electrons(ones, TAU) == pytest.approx(0.6 * s.full_well_e)


def test_flux_fixed_when_tau_changes():
    s = review_16mm(TAU)
    ones = np.ones((3, 3))
    assert s.mean_electrons(ones, TAU / 2) == pytest.approx(0.35 * s.full_well_e)


def test_overrides():
    s = review_16mm(TAU, psf=False, psf_sigma_px=1.3)
    assert not s.psf and s.psf_sigma_px == 1.3
    assert s.required_margin_px == 1
    with pytest.raises(TypeError, match="flux_e_per_s"):
        review_16mm(TAU, flux_e_per_s=1e6)


def test_rpi_cm3_requires_noise_params():
    with pytest.raises(TypeError):
        rpi_cm3(TAU)
    with pytest.raises(TypeError):
        rpi_cm3(TAU, full_well_e=4000.0)


def test_rpi_cm3_full_well_at_adc_top():
    s = rpi_cm3(TAU, full_well_e=4000.0, read_noise_e=2.0)
    assert s.bits == 10 and s.black_level_dn == 64.0
    assert s.full_well_e / s.gain_e_per_dn + s.black_level_dn == pytest.approx(1023)
    # пересвет в 5 раз: все пиксели упираются в верх шкалы
    bright = rpi_cm3(TAU, fill=5.0, full_well_e=4000.0, read_noise_e=2.0)
    out = bright.expose(np.ones((5, 5)), TAU, np.random.default_rng(0))
    np.testing.assert_array_equal(out, 1023.0)
    # без сигнала шум считывания опускает значения и ниже уровня чёрного
    dark = s.expose(np.zeros((20, 20)), TAU, np.random.default_rng(0))
    assert dark.min() < 64.0


def test_rpi_cm3_binned_mode():
    assert camera("rpi_cm3_2x2").fx == pytest.approx(1692.86, abs=0.01)
    # 1 / (56 кадр/с * 1296 строк)
    assert row_time_estimate("rpi_cm3_2x2") == pytest.approx(13.78e-6, rel=1e-3)
    with pytest.raises(KeyError, match="rpi_cm3_2x2"):
        row_time_estimate("review_16mm")
    s = sensor("rpi_cm3", TAU, **RPI_CM3_ASSUMED)
    assert s.full_well_e == RPI_CM3_ASSUMED["full_well_e"]
    assert s.read_noise_e == RPI_CM3_ASSUMED["read_noise_e"]
