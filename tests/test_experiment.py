"""Тесты модуля экспериментов.

Маленькие конфигурации (ss = 8, nt = 8, n_mc = 20), чтобы тесты шли секунды.
Тест 5 (FacingCamera) — в test_scene.
"""

import json
import math
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from photosim.camera import Pose
from photosim.experiment import (
    EFFECTS,
    Config,
    Effects,
    _from_json,
    _to_json,
    ablation,
    additivity,
    build,
    load_run,
    run_point,
    save_run,
    sweep,
)

FAST = Config(ss=8, nt=8, n_mc=20, methods=("centroid",))
OFF_AXIS = (0.2, 0.1, 1.0)
NONE = Effects.none()


def run(cfg):
    return pd.DataFrame(run_point(cfg))


def only(*names):
    return replace(NONE, **dict.fromkeys(names, True))


def ref(row, name):
    return np.array([row[f"ref.{name}.u"], row[f"ref.{name}.v"]])


# ============================================================= 1. воспроизводимость


def test_reproducible_bitwise():
    base = replace(FAST, methods=("centroid", "ellipse"))
    grid = {"tau": [1e-3, 2e-3]}
    a, b = sweep(base, grid), sweep(base, grid)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    # точки перебора независимы: разные потомки seed дают разный шум
    assert a.loc[0, "bias_u_px"] != a.loc[2, "bias_u_px"]
    # другой seed — другой результат
    c = sweep(replace(base, seed=7), grid)
    assert not np.array_equal(a["bias_u_px"], c["bias_u_px"])


# ============================================================= 2. ничего не включено


def test_nothing_enabled_centroid_at_projection():
    cfg = replace(
        FAST, position=(0.0123, -0.0071, 1.0), effects=NONE, reference="proj_mid"
    )
    row = run(cfg).iloc[0]
    assert row["status"] == "ok"
    # замер 1,9e-3 / -1,5e-4: шум дискретизации рендера при ss = 8
    assert abs(row["bias_u_px"]) < 3e-3 and abs(row["bias_v_px"]) < 3e-3
    assert row["std_u_px"] == 0.0  # шумов нет: все реализации одинаковы
    assert row["bias_px"] == pytest.approx(row["err0_px"])


# ============================================================= 3. перспектива


def test_perspective_switch():
    cfg = replace(FAST, position=OFF_AXIS, tilt_deg=45.0, n_mc=2)
    off = run(replace(cfg, effects=NONE)).iloc[0]
    on = run(replace(cfg, effects=only("perspective"))).iloc[0]
    # выключена: плоскость марки || сенсору, образ — круг с центром в проекции
    assert np.abs(ref(off, "conic_mid") - ref(off, "proj_mid")).max() < 1e-9
    assert np.abs(ref(on, "conic_mid") - ref(on, "proj_mid")).max() > 0.01
    np.testing.assert_allclose(ref(on, "proj_mid"), ref(off, "proj_mid"), atol=1e-12)


# ============================================================= 4. экспозиция и движение


def test_exposure_motion_delta_t():
    v, tau, z = 0.5, 1e-3, 1.0
    cfg = replace(
        FAST,
        motion={"type": "linear", "velocity": (v, 0.0, 0.0)},  # поперёк луча
        tau=tau,
        effects=only("exposure", "motion"),
    )
    f = build(cfg).camera.fx
    delta_t = f * v * tau / (2 * z)  # ≈ 1,16 пикс.
    # на оси диск симметричен относительно сетки: замер совпадает до 1e-13
    start = run(replace(cfg, reference="proj_start")).iloc[0]
    assert start["bias_u_px"] == pytest.approx(delta_t, rel=0.01)
    assert abs(start["bias_v_px"]) < 3e-3
    mid = run(replace(cfg, reference="proj_mid")).iloc[0]
    assert abs(mid["bias_u_px"]) < 3e-3 and abs(mid["bias_v_px"]) < 3e-3


# ============================================================= 6. sweep


def test_sweep_rows_and_columns():
    base = replace(FAST, methods=("centroid", "centroid_sq"))
    grid = {"tau": [1e-3, 2e-3], "tilt_deg": [0.0, 30.0, 60.0]}
    df = sweep(base, grid)
    assert len(df) == 2 * 3 * 2
    assert df["point"].nunique() == 6
    assert set(df.groupby("point")["method"].apply(frozenset)) == {
        frozenset(base.methods)
    }
    combos = set(zip(df["tau"], df["tilt_deg"]))
    assert combos == {(t, a) for t in grid["tau"] for a in grid["tilt_deg"]}
    assert (df["status"] == "ok").all()


def test_sweep_nested_keys():
    base = replace(FAST, motion={"type": "linear", "velocity": (0.1, 0.0, 0.0)})
    df = sweep(
        base,
        {"effects.psf": [True, False], "motion.velocity": [(0.1, 0, 0), (0.2, 0, 0)]},
    )
    assert len(df) == 4
    assert set(df["effects.psf"]) == {True, False}
    speeds = {json.loads(m)["velocity"][0] for m in df["motion"]}
    assert speeds == {0.1, 0.2}
    with pytest.raises(ValueError, match="нет поля"):
        sweep(base, {"effects.blur": [True]})


# ============================================================= 7. ablation


def test_ablation_modes():
    names = ["psf", "motion", "exposure"]
    base = replace(FAST, motion={"type": "linear", "velocity": (0.3, 0.0, 0.0)})
    df = ablation(base, names)
    per_mode = df.groupby("mode")["effect"].apply(lambda s: sorted(s.dropna()))
    assert per_mode["single"] == sorted(names)
    assert per_mode["leave_one_out"] == sorted(names)
    assert (df["mode"] == "all").sum() == len(base.methods)
    assert (df["mode"] == "none").sum() == len(base.methods)
    # флаги эффектов в колонках соответствуют режиму
    single_psf = df[(df["mode"] == "single") & (df["effect"] == "psf")].iloc[0]
    assert single_psf["effects.psf"] and not single_psf["effects.motion"]
    loo_psf = df[(df["mode"] == "leave_one_out") & (df["effect"] == "psf")].iloc[0]
    assert not loo_psf["effects.psf"] and loo_psf["effects.motion"]
    # эффекты не из списка — как в base
    assert df["effects.read_noise"].all()

    add = additivity(df)
    assert list(add.index) == ["centroid"]
    assert np.isfinite(add.to_numpy()).all()


def test_ablation_unknown_effect():
    with pytest.raises(ValueError, match="effect"):
        ablation(FAST, ["blur"])


def test_additivity_arithmetic():
    rows = [
        ("none", None, 1.0),
        ("single", "a", 2.0),
        ("single", "b", 3.5),
        ("all", None, 4.0),
        ("leave_one_out", "a", 9.0),
    ]
    df = pd.DataFrame(
        [
            {"mode": m, "effect": e, "method": "c", "status": "ok", "bias_u_px": x}
            for m, e, x in rows
        ]
    )
    add = additivity(df, ["bias_u_px"]).loc["c"]
    assert add["bias_u_px.sum_single"] == pytest.approx(1.0 + 2.5)
    assert add["bias_u_px.all"] == pytest.approx(3.0)
    assert add["bias_u_px.residual"] == pytest.approx(0.5)


# ============================================================= 8. empty и clipped


def test_empty_and_clipped_rows():
    methods = ("centroid", "gauss2d")
    empty = run_point(replace(FAST, position=(1.0, 0.0, 1.0), methods=methods))
    f = build(FAST).camera.fx
    edge_u = (2448 - 10.5 - 1223.5) / f  # центр в 10 пикс. от правого края
    clipped = run_point(replace(FAST, position=(edge_u, 0.0, 1.0), methods=methods))
    for rows, status in ((empty, "empty"), (clipped, "clipped")):
        assert [r["method"] for r in rows] == list(methods)
        assert all(r["status"] == status for r in rows)
        assert all("bias_u_px" not in r for r in rows)
    df = sweep(FAST, {"position": [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]})
    assert list(df["status"]) == ["ok", "empty"]


# ============================================================= 9. сохранение


def test_save_load_roundtrip(tmp_path):
    base = replace(
        FAST,
        camera_pose=Pose.from_euler([0.0, 5.0, 0.0], [0.0, 0.0, 0.0]),
        motion={"type": "linear", "velocity": (0.1, 0.0, 0.0)},
        sensor_overrides={"read_noise_e": 3.0},
    )
    grid = {"tau": [1e-3, 2e-3], "position": [(0.0, 0.0, 1.0), (0.05, 0.0, 1.0)]}
    df = sweep(base, grid)
    path = save_run(df, {"base": base, "grid": grid}, "test", root=tmp_path)
    assert re.fullmatch(r"\d{8}_\d{6}_test", path.name)
    assert {p.name for p in path.iterdir()} == {
        "results.parquet",
        "config.json",
        "meta.json",
    }

    loaded = load_run(path)
    pd.testing.assert_frame_equal(loaded.df, df, check_exact=True)
    assert loaded.configs == {"base": base, "grid": grid}
    assert loaded.meta["seed"] == base.seed
    assert loaded.meta["rows"] == len(df)
    assert {"numpy", "scipy", "pandas"} <= set(loaded.meta["versions"])
    assert "git_commit" in loaded.meta

    again = save_run(df, base, "test", root=tmp_path)  # та же секунда
    assert again != path
    assert load_run(again).configs == {"base": base}


# ============================================================= build


@pytest.mark.parametrize(
    "motion",
    [
        {"type": "static"},
        {"type": "linear", "velocity": (0.3, -0.1, 0.05)},
        {"type": "accel", "velocity": (0.3, 0.0, 0.0), "acceleration": (0.0, 5.0, 0.0)},
        {"type": "arc", "center": (0.2, 0.3, 1.0), "axis": (0, 0, 1), "omega": 3.0},
    ],
    ids=["static", "linear", "accel", "arc"],
)
@pytest.mark.parametrize("t_ref", ["start", "mid"])
def test_build_state_at_t_ref(motion, t_ref):
    cfg = replace(
        FAST,
        position=OFF_AXIS,
        motion=motion,
        rotation={"axis": (1, 1, 0), "omega": 2.0},
        tilt_deg=30.0,
        t_ref=t_ref,
        tau=0.01,
    )
    s = build(cfg)
    np.testing.assert_allclose(s.motion.position(s.t_ref), OFF_AXIS, atol=1e-15)
    to_cam = -np.array(OFF_AXIS) / np.linalg.norm(OFF_AXIS)
    angle = math.degrees(math.acos(s.motion.normal(s.t_ref) @ to_cam))
    assert angle == pytest.approx(30.0, abs=1e-9)
    if "velocity" in motion:
        np.testing.assert_allclose(s.motion.velocity(s.t_ref), motion["velocity"])


def test_build_tilt_direction_matches_render_tests():
    # tilted_normal из test_render: [-sin a, 0, -cos a] на оси
    s = build(replace(FAST, tilt_deg=45.0))
    a = math.radians(45.0)
    np.testing.assert_allclose(
        s.motion.normal(0.0), [-math.sin(a), 0.0, -math.cos(a)], atol=1e-15
    )


def test_build_effects_off():
    cfg = replace(
        FAST,
        camera_pose=Pose.from_euler([10.0, -20.0, 5.0], [0.1, 0.0, 0.2]),
        position=OFF_AXIS,
        tilt_deg=40.0,
        motion={"type": "linear", "velocity": (0.3, 0.0, 0.0)},
        rotation={"axis": (0, 1, 0), "omega": 5.0},
        shutter="rolling",
        t_row=1e-5,
        tau=0.01,
    )
    on = build(cfg)
    assert (on.nt, on.shutter, on.t_row) == (8, "rolling", 1e-5)
    assert on.sensor.psf and on.sensor.shot_noise and on.sensor.quantization

    off = build(replace(cfg, effects=NONE))
    t = np.linspace(0.0, 0.01, 5)
    np.testing.assert_array_equal(off.motion.position(t), np.tile(OFF_AXIS, (5, 1)))
    z_cam = off.pose.R[2]
    np.testing.assert_allclose(
        off.motion.normal(t), np.tile(-z_cam, (5, 1)), atol=1e-15
    )
    assert (off.nt, off.shutter, off.t_row) == (1, "global", 0.0)
    assert not (off.sensor.psf or off.sensor.shot_noise or off.sensor.read_noise)
    assert not off.sensor.quantization

    still = build(replace(cfg, effects=replace(Effects(), rotation=False)))
    R = still.motion.orientation(t)
    np.testing.assert_allclose(R, np.broadcast_to(R[0], R.shape), atol=1e-15)
    np.testing.assert_allclose(R[0], on.motion.orientation(on.t_ref), atol=1e-15)


def test_build_tau_ref_keeps_flux():
    a = build(replace(FAST, tau=1e-3, tau_ref=1e-3)).sensor
    b = build(replace(FAST, tau=2e-3, tau_ref=1e-3)).sensor
    c = build(replace(FAST, tau=2e-3)).sensor  # tau_ref = tau: тот же fill
    assert a.flux_e_per_s == b.flux_e_per_s
    assert c.flux_e_per_s == pytest.approx(a.flux_e_per_s / 2)


def test_rolling_shutter_reference_shift():
    t_row = 2e-5
    cfg = replace(
        FAST,
        position=OFF_AXIS,
        motion={"type": "linear", "velocity": (0.2, 0.0, 0.0)},
        shutter="rolling",
        t_row=t_row,
        effects=only("exposure", "motion", "rolling_shutter"),
        reference="proj_mid",
    )
    row = run(cfg).iloc[0]
    # сдвиг — время дробной строки центра (не округлённой)
    assert row["ref_time_shift"] == pytest.approx(row["ref.proj_mid.v"] * t_row)
    assert row["ref_time_shift"] > 1000 * t_row  # марка ниже середины кадра
    # смазанный и срезанный центр тяжести совпадает со сдвинутой серединой
    # (замер 5,9e-4 / 1,3e-3 — шум дискретизации ss = 8); с округлённой
    # строкой по u было 5,6e-3
    assert abs(row["bias_u_px"]) < 2e-3 and abs(row["bias_v_px"]) < 2e-3


# ============================================================= Config


def test_config_normalized_and_json_roundtrip():
    cfg = Config(
        camera_pose=Pose(np.eye(3), [0.0, 0.1, 0.0]),
        position=[0.1, 0.0, 1.0],
        motion={"type": "linear", "velocity": np.array([0.1, 0.0, 0.0])},
        methods=["centroid"],
        effects={"psf": False},
    )
    assert cfg.position == (0.1, 0.0, 1.0)
    assert cfg.motion["velocity"] == (0.1, 0.0, 0.0)
    assert cfg.methods == ("centroid",)
    assert cfg.effects == replace(Effects(), psf=False)
    assert _from_json(json.loads(json.dumps(_to_json(cfg)))) == cfg
    np.testing.assert_allclose(build(cfg).pose.T, [0.0, 0.1, 0.0])


@pytest.mark.parametrize(
    "kw",
    [
        {"camera": "nikon"},
        {"sensor": "nikon"},
        {"tilt_axis": "z"},
        {"motion": {"type": "spiral"}},
        {"t_ref": "end"},
        {"shutter": "electronic"},
        {"reference": "centroid"},
        {"methods": ("centroid", "moments")},
        {"position": (0.0, 1.0)},
        {"sensor_overrides": {"psf": False}},
    ],
)
def test_config_validation(kw):
    with pytest.raises(ValueError):
        Config(**kw)


def test_build_motion_keys_checked():
    with pytest.raises(ValueError, match="velocity"):
        build(replace(FAST, motion={"type": "linear", "speed": 1.0}))
    arc = {"type": "arc", "center": (0, 0, 0), "axis": (0, 0, 1), "omega": 1.0}
    with pytest.raises(ValueError, match="плоскости"):  # position не в плоскости дуги
        build(replace(FAST, motion=arc))


def test_effects_names():
    assert EFFECTS == (
        "perspective",
        "exposure",
        "motion",
        "rotation",
        "rolling_shutter",
        "psf",
        "shot_noise",
        "read_noise",
        "quantization",
    )
    assert not any(getattr(NONE, name) for name in EFFECTS)
