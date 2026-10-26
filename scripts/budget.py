"""Бюджет погрешностей: вклад каждого эффекта в типовой конфигурации.

Апгрейд табл. 2 обзора: она была аналитической, здесь каждый вклад
проверен моделированием (experiment.ablation). Для каждой камеры одна
конфигурация: марка r = 5 мм, Z = 1 м, наклон 45° вокруг y, угол кадра
(0,9 по u и v, фаза 0,3 пикс.: там эксцентриситет при таком наклоне
наибольший), движение 0,5 м/с вдоль u, τ = 1 мс, t_ref = "mid", fill = 0,7,
n_mc = 500. Добавлено вращение звена 2 рад/с вокруг оси наклона (E5:
1–3 рад/с), иначе вклад «rotation» тождественно ноль.

- review_16mm — глобальный затвор (эффект rolling_shutter не перебирается);
- rpi_cm3 — режим 2304×1296 с бинированием (rpi_cm3_2x2), шумы —
  RPI_CM3_ASSUMED, построчный затвор с оценкой t_row (E6) и привязкой ко
  времени строки центра.

Эталон — proj_mid (проекция центра в середине экспозиции; у RPi — строки
центра): это истинное положение, которое нужно измерить. Вклад эффекта —
режим single (включён только он) минус режим none (все выключены):
систематический — |Δ err0| (кадр без шума, вектор), случайный — СКО
sqrt((std_u**2 + std_v**2) / 2). Режим none — резкий круг без эффектов:
его ошибка — пиксельная фаза метода, отдельная строка.

Экспозиция, движение, вращение и построчный затвор поодиночке ничего не
дают (смаз и перекос — совместные эффекты), поэтому они выключаются
группой GROUPS["motion_group"]: её вклад — all минус all без группы
(режим leave_group_out). Это строка «движение за экспозицию при верной
привязке ко времени». Все — всё включено; неаддитивность — сумма вкладов
(группа вместо своих эффектов поодиночке) минус all
(experiment.additivity). Вклад эффекта в присутствии остальных — столбец
marginal (all минус leave_one_out) в таблице.

Отдельно (штриховка) — ошибки, которых избегает привязка: привязка к началу
экспозиции вместо середины (δt = v τ / 2, самый большой член обзора) и, у
RPi, ко времени кадра вместо времени строки.

    python scripts/budget.py run [--quick] [--jobs 6]
    python scripts/budget.py plot runs/<...>_budget

Рисунок budget (одна колонка, рис. 6 статьи): центр тяжести, обе камеры
парами столбцов, мкм на объекте (у камер разный пиксель: 216 и 591
мкм/пикс.); на строке перспективы — значение по формуле (3) обзора.
Таблица budget_table.csv: все методы, эффект, вклад в пикс. и мкм на
объекте, marginal, аналитическая оценка там, где есть формула (формула (3)
обзора, второй порядок E5, СКО центра тяжести, exposure_conic для группы
движения, v τ / 2, v t_row v*).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # запуск без установки

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from photosim.analysis import (
    CLIP_VARIANCE,
    centroid_sigma,
    eccentricity_formula3,
    point_at_pixel,
    roi_s2,
    rotation_offset_coefficients,
)
from photosim.experiment import (
    EFFECTS,
    Config,
    Effects,
    Run,
    ablation,
    additivity,
    build,
    save_run,
)
from photosim.figures import style
from photosim.figures.cli import Output, main
from photosim.presets import RPI_CM3_ASSUMED, camera, row_time_estimate

NAME = "budget"
Z = 1.0
RADIUS = 0.005
TILT = 45.0
CORNER = 0.9
PHASE = 0.3
SPEED = 0.5  # м/с, вдоль u
TAU = 1e-3
OMEGA = 2.0  # рад/с, звено манипулятора
ROTATION_AXIS = (0.0, 1.0, 0.0)
N_MC = 500
FLOOR_UM = 1e-4  # меньше — на рисунке у оси, с подписью
SIM_LEVEL = 2e-3  # уровень симулятора при ss = 16, пикс. (E3, E6)
CAMERAS = {
    "review_16mm": {
        "label": "16 mm lens, IMX264 (global shutter)",
        "camera": "review_16mm",
        "sensor": "review_16mm",
        "sensor_overrides": {},
        "shutter": "global",
        "effects": tuple(e for e in EFFECTS if e != "rolling_shutter"),
    },
    "rpi_cm3": {
        "label": "Raspberry Pi CM3, 2304×1296 (rolling shutter)",
        "camera": "rpi_cm3_2x2",
        "sensor": "rpi_cm3",
        "sensor_overrides": RPI_CM3_ASSUMED,
        "shutter": "rolling",
        "effects": EFFECTS,
    },
}
#: Эффекты, которые поодиночке ничего не дают: выключаются разом.
GROUPS = {"motion_group": ["exposure", "motion", "rotation", "rolling_shutter"]}
#: Камеры на рисунке: цвет столбцов.
CAMERA_COLOR = {"review_16mm": "0.25", "rpi_cm3": "0.7"}
#: Маркер значения по распространённому приближению на строке перспективы.
APPROX_MARKER = {"marker": "D", "markersize": 4.5, "markerfacecolor": "white",
                 "markeredgecolor": "0.0", "markeredgewidth": 0.8}
CAMERA_SHORT = {
    "review_16mm": "16 mm, IMX264, global shutter",
    "rpi_cm3": "RPi CM3, IMX708, rolling shutter",
}


def config(key: str, n_mc: int) -> Config:
    spec = CAMERAS[key]
    cam = camera(spec["camera"])
    u = round(CORNER * (cam.width - 1)) + PHASE
    v = round(CORNER * (cam.height - 1)) + PHASE
    rolling = spec["shutter"] == "rolling"
    return Config(
        camera=spec["camera"],
        sensor=spec["sensor"],
        sensor_overrides=spec["sensor_overrides"],
        radius=RADIUS,
        position=point_at_pixel(cam, u, v, Z),
        tilt_deg=TILT,
        tilt_axis="y",
        motion={"type": "linear", "velocity": (SPEED, 0.0, 0.0)},
        rotation={"axis": ROTATION_AXIS, "omega": OMEGA},
        tau=TAU,
        t_ref="mid",
        shutter=spec["shutter"],
        t_row=row_time_estimate(spec["camera"]) if rolling else 0.0,
        effects=Effects(),
        n_mc=n_mc,
        reference="proj_mid",
        ss=16,
        nt=64,
    )


def run(args: argparse.Namespace) -> Path:
    n_mc = 50 if args.quick else N_MC
    parts, bases = [], {}
    for key, spec in CAMERAS.items():
        base = config(key, n_mc)
        bases[key] = base
        df = ablation(base, spec["effects"], groups=GROUPS, n_jobs=args.jobs)
        df.insert(0, "budget_camera", key)
        parts.append(df)
    configs = {"experiment": NAME, "quick": args.quick, "base": bases["review_16mm"],
               "bases": bases, "groups": GROUPS}
    name = NAME + ("_quick" if args.quick else "")
    return save_run(pd.concat(parts, ignore_index=True), configs, name, args.root)


# Обработка


def vec(row: pd.Series, prefix: str) -> np.ndarray:
    return np.array([row[f"{prefix}_u_px"], row[f"{prefix}_v_px"]])


def ref(row: pd.Series, name: str) -> np.ndarray:
    return np.array([row[f"ref.{name}.u"], row[f"ref.{name}.v"]])


#: СКО меньше этого — округление float в одинаковых кадрах, а не шум, пикс.
ZERO_STD_PX = 1e-9


def rms_std(row: pd.Series) -> float:
    """СКО на ось: sqrt((std_u**2 + std_v**2) / 2); без шума — 0."""
    su, sv = row["std_u_px"], row["std_v_px"]
    if not (np.isfinite(su) and np.isfinite(sv)):
        return 0.0
    std = math.sqrt((su**2 + sv**2) / 2)
    return std if std > ZERO_STD_PX else 0.0


def systematic(row: pd.Series) -> np.ndarray:
    """Систематическая ошибка режима, вектор, пикс.

    ``err0`` считается по кадру без шума и без квантования. Если кадры
    режима одинаковы (шумов нет), его ``bias`` точен и включает округление
    АЦП: так виден вклад квантования поодиночке.
    """
    return vec(row, "bias") if rms_std(row) == 0 else vec(row, "err0")


def analytic(key: str, base: Config, effect: str, single: pd.Series) -> float:
    """Аналитическая оценка вклада для центра тяжести, пикс.; nan — нет формулы."""
    if effect == "perspective":
        s = build(base)
        C = s.pose.to_camera(s.motion.position(s.t_ref))
        n = s.pose.R @ s.motion.normal(s.t_ref)
        return float(np.hypot(*eccentricity_formula3(s.camera.fx, RADIUS, n, C[2])))
    if effect == "rotation":
        s = build(base)
        k_curv, k_area = rotation_offset_coefficients(
            s.camera, s.pose, s.motion.position(s.t_ref),
            s.motion.orientation(s.t_ref), ROTATION_AXIS, RADIUS,
        )
        return float(np.hypot(*(k_curv + k_area))) * (OMEGA * TAU) ** 2
    if effect in ("exposure", "motion", "rolling_shutter", "psf"):
        return 0.0  # поодиночке ничего не сдвигают (t_ref = mid, нет движения)
    n_e = single["signal_e"]
    s2 = 0.5 * (
        roi_s2(single["roi.u0"], single["roi.w"], single["roi.h"], single["ref.proj_mid.u"])
        + roi_s2(single["roi.v0"], single["roi.h"], single["roi.w"], single["ref.proj_mid.v"])
    )
    if effect == "shot_noise":  # без PSF и фона: R**2 / 4 + 1/12
        return float(centroid_sigma(single["radius_px"], n_e, extra_var_px=1 / 12))
    if effect == "read_noise":  # только фон, с обрезкой max(I - bg, 0)
        var = CLIP_VARIANCE * single["sensor.read_noise_e"] ** 2
        return float(centroid_sigma(0.0, n_e, var, s2))
    return math.nan


def entry(
    head: dict, um: float, item: str, kind: str, sys_px: float, std_px: float, **extra
) -> dict:
    """Строка бюджета; ``head`` — камера и метод, ``um`` — мкм на пикс."""
    if sys_px < ZERO_STD_PX:  # тот же кадр: разница — округление float
        sys_px = 0.0
    return {
        **head,
        "item": item,
        "kind": kind,
        "sys_px": sys_px,
        "sys_um": sys_px * um,
        "std_px": std_px,
        "std_um": std_px * um,
        **extra,
    }


def contributions(run: Run) -> pd.DataFrame:
    """Длинная таблица: камера, метод, строка бюджета, вклад."""
    rows = []
    for key, d in run.df[run.df["status"] == "ok"].groupby("budget_camera", sort=False):
        base = run.configs["bases"][key]
        add = additivity(d)
        for method, dm in d.groupby("method", sort=False):
            rows += method_rows(key, base, add.loc[method], method, dm)
    return pd.DataFrame(rows)


def method_rows(
    key: str, base: Config, r: pd.Series, method: str, dm: pd.DataFrame
) -> list[dict]:
    """Строки бюджета одного метода; ``r`` — его строка ``additivity``."""
    none = dm[dm["mode"] == "none"].iloc[0]
    all_ = dm[dm["mode"] == "all"].iloc[0]
    um = float(all_["um_per_px"])
    head = {"camera": key, "method": method}
    rows: list[dict] = []

    def add_row(*args, **kwargs) -> None:
        rows.append(entry(head, um, *args, **kwargs))

    add_row("baseline", "baseline", float(np.hypot(*systematic(none))), 0.0)
    for effect in dm.loc[dm["mode"] == "single", "effect"]:
        single = dm[(dm["mode"] == "single") & (dm["effect"] == effect)].iloc[0]
        loo = dm[(dm["mode"] == "leave_one_out") & (dm["effect"] == effect)].iloc[0]
        a = analytic(key, base, effect, single) if method == "centroid" else math.nan
        add_row(
            effect, "effect",
            float(np.hypot(*(systematic(single) - systematic(none)))),
            rms_std(single),
            marginal_sys_px=float(np.hypot(*(vec(all_, "err0") - vec(loo, "err0")))),
            marginal_std_px=math.sqrt(abs(rms_std(all_) ** 2 - rms_std(loo) ** 2)),
            analytic_px=a,
            analytic_um=a * um,
        )
    for group, g in dm[dm["mode"] == "leave_group_out"].groupby("effect", sort=False):
        off = g.iloc[0]
        # центр тяжести без шума — центр эллипса, усреднённый за экспозицию;
        # без группы — эллипс неподвижной марки в середине экспозиции
        a = (
            float(np.linalg.norm(
                ref(all_, "exposure_conic") - ref(all_, "proj_mid")
                - (ref(off, "conic_mid") - ref(off, "proj_mid"))
            ))
            if method == "centroid"
            else math.nan
        )
        add_row(
            group, "group",
            float(np.hypot(*(vec(all_, "err0") - vec(off, "err0")))),
            math.sqrt(abs(rms_std(all_) ** 2 - rms_std(off) ** 2)),
            analytic_px=a,
            analytic_um=a * um,
        )
    add_row(
        "all", "all", float(np.hypot(*(vec(all_, "err0") - vec(none, "err0")))),
        rms_std(all_), bias_all_px=float(np.hypot(*vec(all_, "bias"))),
    )
    var_res = 0.5 * (r["var_u_px.residual"] + r["var_v_px.residual"])
    add_row(
        "non_additivity", "residual",
        float(np.hypot(r["err0_u_px.residual"], r["err0_v_px.residual"])),
        math.sqrt(abs(var_res)),
    )
    # чего избегает привязка (по эталонам режима all)
    dt = float(np.linalg.norm(ref(all_, "proj_mid") - ref(all_, "proj_start")))
    add_row("stamp_start", "avoided", dt, 0.0,
            analytic_px=all_["f_px"] * SPEED * TAU / (2 * all_["z_m"]),
            analytic_um=SPEED * TAU / 2 * 1e6)
    if base.shutter == "rolling":
        fr = float(np.linalg.norm(ref(all_, "proj_mid") - ref(all_, "proj_frame_mid")))
        add_row("stamp_frame", "avoided", fr, 0.0,
                analytic_px=all_["f_px"] * SPEED * all_["ref_time_shift"] / all_["z_m"],
                analytic_um=SPEED * all_["ref_time_shift"] * 1e6)
    return rows


#: Строки рисунка: (строка бюджета, систематика или СКО, подпись).
FIGURE_ROWS = [
    ("baseline", "sys", "Pixel phase (no effects)"),
    ("perspective", "sys", "Perspective (eccentricity)"),
    ("motion_group", "sys", "Motion during exposure,\ncorrect time stamp"),
    ("psf", "sys", "PSF"),
    ("shot_noise", "std", "Shot noise (std)"),
    ("read_noise", "std", "Read noise (std)"),
    ("quantization", "sys", "Quantization"),
    ("all", "sys", "All effects, offset"),
    ("all", "std", "All effects, std"),
    ("non_additivity", "sys", "Non-additivity"),
    ("stamp_start", "sys", "Start-of-exposure stamp"),
    ("stamp_frame", "sys", "Frame-time stamp"),
]


def plot(run: Run, out: Output, args: argparse.Namespace) -> None:
    table = contributions(run)
    out_table = table.copy()
    out_table["method"] = pd.Categorical(out_table["method"], style.ordered(out_table["method"]))
    out.table(out_table.sort_values(["camera", "method"], kind="stable"), NAME)
    out.figure(figure(table, run), NAME)
    check(table, int(run.df["n_mc"].min()), out)


def figure(table: pd.DataFrame, run: Run) -> plt.Figure:
    """Рис. 6: центр тяжести, обе камеры парами столбцов, мкм на объекте."""
    c = table[table["method"] == "centroid"]
    keys = [k for k in CAMERAS if k in set(c["camera"])]
    rows = [r for r in FIGURE_ROWS if r[0] in set(c["item"])]
    fig, ax = plt.subplots(figsize=(style.COLUMN, 0.27 * len(rows) + 1.25))
    y = np.arange(len(rows))[::-1].astype(float)
    h = 0.38
    left = FLOOR_UM / 10
    shown_max = 0.0
    for j, key in enumerate(keys):
        d = c[c["camera"] == key].set_index("item")
        dy = (0.5 - j) * h if len(keys) == 2 else 0.0
        for yi, (item, kind, _) in zip(y, rows, strict=True):
            if item not in d.index:
                continue
            row = d.loc[item]
            value = float(row[f"{kind}_um"])
            avoided = row["kind"] == "avoided"
            shown = max(value, FLOOR_UM)
            shown_max = max(shown_max, shown)
            # слои: линии уровня (2) < подложка подписи (2.5) < столбцы (3)
            ax.barh(yi + dy, shown - left, left=left, height=h, color=CAMERA_COLOR[key],
                    edgecolor="0.0", linewidth=0.4, hatch="////" if avoided else None,
                    zorder=3)
            text = plain(value) if value >= FLOOR_UM else f"<{FLOOR_UM:g}"
            # белая подложка: подпись не сливается с линиями уровня симулятора
            ax.annotate(text, (shown, yi + dy), xytext=(2.5, 0), textcoords="offset points",
                        va="center", fontsize=style.FONT_SIZE - 1,
                        bbox={"fc": "white", "ec": "none", "pad": 0.5}, zorder=2.5)
    # перспектива по формуле (3) обзора: одинакова в мкм для обеих камер
    # белый ромб между столбцами: виден на тёмном и светлом, не сливается с
    # концами столбцов (12,5 мкм почти у 15 и 20)
    persp = c[c["item"] == "perspective"]
    approx_um = math.nan
    if not persp.empty:
        yp = y[[r[0] for r in rows].index("perspective")]
        approx_um = float(persp["analytic_um"].iloc[0])
        ax.plot(approx_um, yp, linestyle="", **APPROX_MARKER, zorder=4)
    for key, ls in zip(keys, (":", "--"), strict=False):
        um = float(run.df.loc[run.df["budget_camera"] == key, "um_per_px"].iloc[0])
        ax.axvline(SIM_LEVEL * um, color="0.35", linewidth=0.6, linestyle=ls)
    # разделить вклады и то, чего избегает привязка
    n_avoided = sum(r[0].startswith("stamp_") for r in rows)
    if n_avoided:
        ax.axhline(y[len(rows) - n_avoided] + 0.5, color="0.0", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlim(left, 40 * shown_max)
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.set_yticks(y, [r[2] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribution, centroid (µm on the target)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CAMERA_COLOR[k], edgecolor="0.0",
                      linewidth=0.4)
        for k in keys
    ]
    labels = [f"{CAMERA_SHORT[k]}, {camera_um(k, run):.0f} µm/px" for k in keys]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.0",
                                 linewidth=0.4, hatch="////"))
    labels.append("Avoided by the time stamp")
    handles.append(plt.Line2D([], [], linestyle="", **APPROX_MARKER))
    labels.append(f"Common approximation, {approx_um:.1f} µm")
    for key, ls in zip(keys, (":", "--"), strict=False):
        handles.append(plt.Line2D([], [], color="0.35", linewidth=0.6, linestyle=ls))
        labels.append(f"Simulator level {SIM_LEVEL:g} px, {key_short(key)}")
    fig.legend(handles, labels, loc="outside lower center", ncols=1)
    return fig


def plain(value: float) -> str:
    """Две значащие цифры без экспоненты: 250, 8000, 0.0018."""
    return f"{float(f'{value:.2g}'):g}"


def camera_um(key: str, run: Run) -> float:
    return float(run.df.loc[run.df["budget_camera"] == key, "um_per_px"].iloc[0])


def key_short(key: str) -> str:
    return CAMERA_SHORT[key].split(",")[0]


def check(table: pd.DataFrame, run_n_mc: int, out: Output) -> None:
    c = table[table["method"] == "centroid"]
    for key, d in c.groupby("camera", sort=False):
        d = d.set_index("item")
        persp = d.loc["perspective"]
        print(
            f"  {key}: перспектива {persp['sys_px']:.3g} пикс. (формула (3) "
            f"{persp['analytic_px']:.3g}), все эффекты: смещение "
            f"{d.loc['all', 'sys_px']:.2g} пикс., СКО {d.loc['all', 'std_px']:.2g} пикс.; "
            f"неаддитивность {d.loc['non_additivity', 'sys_px']:.1e} пикс."
        )
        shot = d.loc["shot_noise"]
        rel = abs(shot["std_px"] / shot["analytic_px"] - 1)
        n_mc = int(run_n_mc)
        rtol = max(0.15, 3 / math.sqrt(2 * (n_mc - 1)))
        out.check(
            rel <= rtol,
            f"{key}: СКО центра тяжести от фотонного шума против теории: {rel:.1%} "
            f"(порог {rtol:.0%})",
        )
        read = d.loc["read_noise"]
        print(
            f"  {key}: шум считывания поодиночке — СКО {read['std_px']:.1e} пикс., "
            f"теория фона с обрезкой {read['analytic_px']:.1e} "
            f"(×{read['std_px'] / read['analytic_px']:.2f})"
        )
        if "motion_group" in d.index:
            g = d.loc["motion_group"]
            out.check(
                g["sys_px"] <= SIM_LEVEL,
                f"{key}: движение за экспозицию (группа) при верной привязке — "
                f"{g['sys_px']:.1e} пикс. = {g['sys_um']:.2f} мкм (ниже уровня "
                f"симулятора {SIM_LEVEL:g}); аналитически {g['analytic_px']:.1e}",
            )
        st = d.loc["stamp_start"]
        out.check(
            abs(st["sys_px"] / st["analytic_px"] - 1) < 1e-6,
            f"{key}: δt = v τ / 2 = {st['sys_um']:.0f} мкм ({st['sys_px']:.3g} пикс.) — "
            f"убирается привязкой к середине экспозиции",
        )


if __name__ == "__main__":
    sys.exit(main("Бюджет погрешностей", run, plot))
