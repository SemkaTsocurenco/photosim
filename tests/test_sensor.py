"""Тесты сенсора.

Рендер — как в test_render: камера review_16mm без дисторсии, марка
r = 5 мм на Z = 1 м (≈ 23 пикс.), фон 0. Сенсор по умолчанию: ёмкость
10 000 e⁻, марка при τ = 10 мс даёт 70 % ёмкости.
"""

import dataclasses

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from photosim.camera import Camera, Pose
from photosim.render import render
from photosim.scene import Motion
from photosim.sensor import SensorModel
from photosim.target import Target

CAM = Camera.from_physical(16, 3.45, 2448, 2048)
POSE = Pose(np.eye(3), np.zeros(3))
F = CAM.fx
TARGET = Target(0.005)
TAU = 0.01
# центр марки в дробной точке кадра, нормаль к камере
MOTION = Motion.static([5.37 / F, -2.81 / F, 1.0], [0.0, 0.0, -1.0])

BASE = {
    "flux_e_per_s": 7e5,  # 7000 e⁻ за τ, 70 % ёмкости
    "read_noise_e": 2.5,
    "full_well_e": 10_000.0,
    "gain_e_per_dn": 2.5,
    "bits": 12,
    "psf_sigma_px": 0.7,
}
NO_EFFECTS = {
    "psf": False,
    "shot_noise": False,
    "read_noise": False,
    "saturation": False,
    "quantization": False,
}


def make(**kw):
    return SensorModel(**{**BASE, **kw})


def only(*flags, **kw):
    """Сенсор, у которого включены только этапы ``flags``."""
    return make(**{**NO_EFFECTS, **{f: True for f in flags}, **kw})


def rng(seed=0):
    return np.random.default_rng(seed)


def frame(margin_px):
    return render(CAM, POSE, TARGET, MOTION, 0.0, TAU, ss=8, nt=1, margin_px=margin_px)


def centroid(I):
    s = I.sum()
    h, w = I.shape
    return np.array(
        [(I.sum(axis=0) * np.arange(w)).sum() / s, (I.sum(axis=1) * np.arange(h)).sum() / s]
    )


@pytest.fixture(scope="module")
def irr():
    """Облучённость марки с запасом под PSF σ = 0,7."""
    return frame(make().required_margin_px).irradiance


# ============================================================= 1. все этапы выключены


def test_all_off_is_linear(irr):
    s = only()
    expected = irr * s.flux_e_per_s * TAU / s.gain_e_per_dn
    np.testing.assert_array_equal(s.expose(irr, TAU, rng()), expected)
    many = s.expose_many(irr, TAU, rng(), 3)
    np.testing.assert_array_equal(many, np.broadcast_to(expected, many.shape))
    np.testing.assert_array_equal(s.mean_electrons(irr, TAU), irr * s.flux_e_per_s * TAU)


def test_mean_electrons_dark_current(irr):
    s = only(dark_e_per_s=300.0)
    mean_e = s.mean_electrons(irr, TAU)
    np.testing.assert_allclose(mean_e, irr * s.flux_e_per_s * TAU + 3.0, rtol=1e-15)
    assert mean_e.min() == pytest.approx(3.0)  # фон: только темновой ток


def test_mean_electrons_psf_matches_filter(irr):
    s = only("psf")
    np.testing.assert_array_equal(
        s.mean_electrons(irr, TAU),
        gaussian_filter(irr, 0.7, mode="nearest") * s.flux_e_per_s * TAU,
    )


def test_input_not_modified(irr):
    before = irr.copy()
    make().expose_many(irr, TAU, rng(), 2)
    np.testing.assert_array_equal(irr, before)


# ============================================================= 2–3. PSF и запас ROI


@pytest.mark.parametrize("sigma", [0.5, 0.7, 1.3])
def test_psf_preserves_sum_and_centroid(sigma):
    # поток 1/τ: электроны равны облучённости
    s = only("psf", psf_sigma_px=sigma, flux_e_per_s=1 / TAU)
    I = frame(s.required_margin_px).irradiance
    blurred = s.mean_electrons(I, TAU)
    assert abs(blurred.sum() - I.sum()) < 1e-9
    assert np.abs(centroid(blurred) - centroid(I)).max() < 1e-6
    assert np.abs(blurred - I).max() > 0.01  # размытие действительно есть


# Замер при margin_px = 0: теряется 4,5e-6 (σ = 0,5), 1,7e-4 (σ = 0,7),
# 3,1e-3 (σ = 1,3) света, центр сдвигается до 1,4e-4 / 6,6e-4 / 5,8e-3 пикс.
@pytest.mark.parametrize("sigma", [0.5, 0.7, 1.3])
def test_psf_without_margin_loses_light(sigma):
    s = only("psf", psf_sigma_px=sigma, flux_e_per_s=1 / TAU)
    I = frame(0).irradiance
    loss = (I.sum() - s.mean_electrons(I, TAU).sum()) / I.sum()
    assert loss > 1e-6  # в тысячу раз больше допуска теста с запасом


# ============================================================= 4. фотонный шум


def test_shot_noise_is_poisson():
    n = 10_000
    # уровни от 0,7 до 7000 e⁻ и 1 e⁻ темнового тока: от единиц до тысяч
    s = only("shot_noise", dark_e_per_s=100.0)
    I = np.geomspace(1e-4, 1.0, 48).reshape(6, 8)
    mu = s.mean_electrons(I, TAU)
    e = s.expose_many(I, TAU, rng(4), n) * s.gain_e_per_dn
    m = e.mean(axis=0)
    v = e.var(axis=0, ddof=1)
    # дисперсия выборочной дисперсии для Пуассона ≈ (μ + 2μ²) / n
    z_mean = (m - mu) / np.sqrt(mu / n)
    z_var = (v - mu) / np.sqrt((mu + 2 * mu**2) / n)
    assert np.abs(z_mean).max() < 5
    assert np.abs(z_var).max() < 5
    np.testing.assert_allclose(e, np.round(e), atol=1e-9)  # число электронов целое


# ============================================================= 5. шум считывания


def test_read_noise_std():
    s = only("read_noise")
    e = s.expose_many(np.zeros((30, 30)), TAU, rng(5), 100) * s.gain_e_per_dn
    assert e.std() == pytest.approx(s.read_noise_e, rel=0.02)
    assert abs(e.mean()) < 5 * s.read_noise_e / np.sqrt(e.size)


# ============================================================= фон и уровень чёрного


def test_black_level_keeps_negative_read_noise():
    # насыщение режет только сверху: шум фона симметричен вокруг уровня чёрного
    s = make(black_level_dn=64.0, quantization=False)
    out = s.expose_many(np.zeros((30, 30)), TAU, rng(12), 100)
    se = out.std() / np.sqrt(out.size)
    assert abs(out.mean() - 64.0) < 5 * se
    assert out.min() < 64.0


def test_zero_black_level_adc_clips_at_zero():
    # без смещения отрицательный шум обрезает АЦП, как в реальной камере:
    # среднее = sum_k P(z > k - 0.5), z = e / gain ~ N(0, 1) при σ = gain
    s = make()
    assert s.read_noise_e == s.gain_e_per_dn
    out = s.expose_many(np.zeros((30, 30)), TAU, rng(13), 100)
    expected = norm.sf(np.arange(1, 10) - 0.5).sum()  # ≈ 0,382 ед.
    assert out.min() == 0.0
    assert out.mean() == pytest.approx(expected, abs=5 * out.std() / np.sqrt(out.size))


# ============================================================= 6. насыщение


@pytest.fixture(scope="module")
def bright():
    """Марка с потоком в 1000 раз больше ёмкости и маска её пикселей."""
    I = frame(make().required_margin_px).irradiance
    return I, I > 0


def test_saturation_at_full_well(bright):
    I, inside = bright
    s = make(quantization=False)
    s = dataclasses.replace(s, flux_e_per_s=s.flux_for_fill(1000.0, TAU))
    out = s.expose(I, TAU, rng(6))
    np.testing.assert_array_equal(out[inside], s.full_well_e / s.gain_e_per_dn)
    assert out.max() == s.full_well_e / s.gain_e_per_dn


@pytest.mark.parametrize("gain", [2.5, 2.0], ids=["fw_below_adc", "fw_above_adc"])
def test_saturation_after_adc(bright, gain):
    I, inside = bright
    s = make(gain_e_per_dn=gain)
    s = dataclasses.replace(s, flux_e_per_s=s.flux_for_fill(1000.0, TAU))
    out = s.expose_many(I, TAU, rng(7), 3)
    expected = min(round(s.full_well_e / gain), s.max_dn)  # 4000 или 4095
    assert np.all(out[:, inside] == expected)
    assert out.max() <= 2**s.bits - 1
    assert out.min() >= 0


# ============================================================= 7. квантование


@pytest.mark.parametrize("black", [0.0, 64.0])
def test_quantization_integer_values(irr, black):
    out = make(black_level_dn=black).expose_many(irr, TAU, rng(8), 4)
    np.testing.assert_array_equal(out, np.round(out))
    assert out.dtype == np.float64


def test_quantization_step_equals_gain():
    s = only("quantization", flux_e_per_s=1 / TAU)  # электроны = облучённость
    step = s.gain_e_per_dn / 100
    e = np.arange(0.0, 40 * s.gain_e_per_dn, step)[None, :]
    dn = s.expose(e, TAU, rng())
    np.testing.assert_array_equal(np.diff(np.unique(dn)), 1.0)  # уровни подряд
    assert np.abs(dn * s.gain_e_per_dn - e).max() <= s.gain_e_per_dn / 2 + 1e-12
    jumps = e[0, 1:][np.diff(dn[0]) > 0]  # где меняется код
    np.testing.assert_allclose(np.diff(jumps), s.gain_e_per_dn, atol=step + 1e-12)


# ============================================================= 8. воспроизводимость


def test_same_seed_bitwise(irr):
    s = make(dark_e_per_s=50.0)
    np.testing.assert_array_equal(s.expose(irr, TAU, rng(42)), s.expose(irr, TAU, rng(42)))
    np.testing.assert_array_equal(
        s.expose_many(irr, TAU, rng(42), 5), s.expose_many(irr, TAU, rng(42), 5)
    )
    assert not np.array_equal(s.expose(irr, TAU, rng(42)), s.expose(irr, TAU, rng(43)))


def test_expose_many_matches_expose_in_distribution(irr):
    n = 2000
    s = make()
    a = s.expose_many(irr, TAU, rng(10), n)
    g = rng(11)
    b = np.stack([s.expose(irr, TAU, g) for _ in range(n)])
    se = np.sqrt((a.var(axis=0) + b.var(axis=0)) / n)
    z = np.abs(a.mean(axis=0) - b.mean(axis=0)) / np.maximum(se, 1e-12)
    assert z.max() < 5
    # СКО по пикселям марки и фона отдельно: шумы там разные
    inside = irr > 0.5
    for mask in (inside, ~inside):
        sa = np.sqrt(a[:, mask].var(axis=0).mean())
        sb = np.sqrt(b[:, mask].var(axis=0).mean())
        assert sa == pytest.approx(sb, rel=0.01)


# ============================================================= 9. формы


@pytest.mark.parametrize("n", [1, 3])
def test_shapes(irr, n):
    s = make()
    assert s.mean_electrons(irr, TAU).shape == irr.shape
    assert s.expose(irr, TAU, rng()).shape == irr.shape
    assert s.expose_many(irr, TAU, rng(), n).shape == (n, *irr.shape)


def test_empty_roi():
    res = render(
        CAM, POSE, TARGET, Motion.static([1.0, 0.0, 1.0], [0, 0, -1.0]), 0.0, TAU
    )
    assert res.empty
    s = make()
    assert s.mean_electrons(res.irradiance, TAU).shape == (0, 0)
    assert s.expose(res.irradiance, TAU, rng()).shape == (0, 0)
    assert s.expose_many(res.irradiance, TAU, rng(), 4).shape == (4, 0, 0)


# ============================================================= 10. запас ROI


@pytest.mark.parametrize(
    "sigma, margin",
    [(0.0, 1), (0.5, 3), (0.7, 4), (0.75, 4), (1.3, 7), (1.5, 7)],  # 4 * 0.75 = 3 ровно
)
def test_required_margin_px(sigma, margin):
    assert make(psf_sigma_px=sigma).required_margin_px == margin


def test_required_margin_without_psf():
    assert make(psf_sigma_px=1.3, psf=False).required_margin_px == 1


# ============================================================= поток под заполнение


def test_flux_for_fill():
    s = make(dark_e_per_s=200.0)
    flux = s.flux_for_fill(0.7, TAU)
    s = dataclasses.replace(s, flux_e_per_s=flux)
    ones = np.ones((3, 3))
    assert s.mean_electrons(ones, TAU) == pytest.approx(0.7 * s.full_well_e)
    # поток не зависит от τ: вдвое длиннее экспозиция — вдвое больше сигнала
    assert s.mean_electrons(ones, 2 * TAU) == pytest.approx(1.4 * s.full_well_e)
    with pytest.raises(ValueError, match="темновой"):
        make(dark_e_per_s=1e6).flux_for_fill(0.7, TAU)
    with pytest.raises(ValueError):
        s.flux_for_fill(0.0, TAU)
    with pytest.raises(ValueError):
        s.flux_for_fill(0.7, 0.0)


# ============================================================= проверки параметров


@pytest.mark.parametrize(
    "kw",
    [
        {"flux_e_per_s": -1.0},
        {"dark_e_per_s": -1.0},
        {"read_noise_e": -0.1},
        {"full_well_e": 0.0},
        {"gain_e_per_dn": 0.0},
        {"psf_sigma_px": -0.1},
        {"black_level_dn": -1.0},
        {"black_level_dn": 255.0, "bits": 8},
        {"bits": 11},
        {"bits": 16},
        {"read_noise_e": float("nan")},
        {"full_well_e": float("inf")},
    ],
)
def test_bad_params_value_error(kw):
    with pytest.raises(ValueError):
        make(**kw)


@pytest.mark.parametrize(
    "kw",
    [
        {"bits": 12.0},
        {"bits": True},
        {"psf": 1},
        {"quantization": "yes"},
        {"gain_e_per_dn": None},
    ],
)
def test_bad_params_type_error(kw):
    with pytest.raises(TypeError):
        make(**kw)


def test_params_normalized_and_frozen():
    s = make(bits=np.int64(10), full_well_e=4000, psf=np.True_)
    assert type(s.bits) is int and s.bits == 10 and s.max_dn == 1023
    assert type(s.full_well_e) is float
    assert s.psf is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.gain_e_per_dn = 1.0
    with pytest.raises(TypeError):
        SensorModel(7e5, 2.5, 10_000.0, 2.5, 12, 0.7)  # только по имени


@pytest.mark.parametrize(
    "bad", [np.zeros(5), np.zeros((2, 3, 4)), np.full((3, 3), np.nan), -np.ones((3, 3))]
)
def test_bad_irradiance(bad):
    with pytest.raises(ValueError):
        make().expose(bad, TAU, rng())


def test_bad_call_args(irr):
    s = make()
    with pytest.raises(ValueError):
        s.mean_electrons(irr, 0.0)
    for bad_rng in (42, None, np.random.RandomState(0)):
        with pytest.raises(TypeError, match="Generator"):
            s.expose(irr, TAU, bad_rng)
    with pytest.raises(ValueError):
        s.expose_many(irr, TAU, rng(), 0)
    with pytest.raises(TypeError):
        s.expose_many(irr, TAU, rng(), 2.5)
