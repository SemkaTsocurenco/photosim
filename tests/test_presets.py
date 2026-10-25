import numpy as np
import pytest

from photosim.presets import REVIEW_16MM, review_16mm, rpi_cm3

TAU = 1e-3


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
