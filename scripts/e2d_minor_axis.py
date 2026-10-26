"""E2d. Граница применимости методов по малой полуоси образа (дополнение к E2b).

В E2b узкий образ был один (θ = 75° в углу кадра, малая полуось 1,26
пикс.). Здесь — перебор: марка r = 5 мм (R ≈ 23 пикс.) на оптической оси
review_16mm, Z = 1 м, наклон θ = 60, 63, ..., 87° вокруг y камеры, малая
полуось образа b ≈ R cos θ от 11,6 до 1,2 пикс. Шумов нет, марка
неподвижна; PSF (sigma = 0,7 пикс., в рендере до пикселя) включена и, тем
же перебором, выключена. Все методы, эталон conic_mid (точный центр
эллипса), ss = 64, как в E0 (его уровни — пороги). Дискретизация рендера
проверяется повтором при ss = 128 в двух самых узких точках (θ = 84 и
87°) при фазе CHECK_PHASE.

ss = 16 узкому образу мало: его длинные стороны почти параллельны
столбцам субпикселей, и квантование края до 1/ss пикс. не усредняется
вдоль края, а складывается. При θ = 84° (b = 2,4 пикс.), фазе (0,3; 0,3)
центр тяжести с PSF ошибался на 6,6e-3 пикс. при ss = 16 и на 5,8e-5 при
ss = 32. Строго на оси ошибки левого и правого краёв взаимно гасятся.

Центр строго на оси (cx, cy полуцелые) — особая точка: сетка пикселей
зеркально симметрична относительно образа, и даже при b = 1,2 пикс. ни
один метод не выходит за свой уровень E0 (порог 50 % — 3e-3 пикс.), а в
E2b при сдвинутой фазе — 0,2 пикс. Поэтому, как в E0, перебирается
субпиксельная фаза центра: сдвиг du, dv = 0 ... 0,5 пикс. от оси (марка на
оси с точностью до пикселя), ошибка метода — худшая по фазам.

Ошибка метода — |err0| (кадр без шума) относительно conic_mid. Уровень
метода — его собственное смещение из E0 (R ≈ 23 пикс., с PSF или без, как
в точке; analysis.method_own_bias), но не ниже уровня симулятора: граница
b / sigma_psf — значение, ниже которого худшая по фазе ошибка устойчиво
больше уровня (как L в e3_threshold). Это граница применимости контурных и
пороговых методов (табл. II статьи).

    python scripts/e2d_minor_axis.py run [--jobs 10] --e0 runs/<...>_e0
    python scripts/e2d_minor_axis.py plot runs/<...>_e2d --e0 runs/<...>_e0

Рисунок e2d_minor_axis (одна колонка, в статью не идёт): ошибка методов от
b / sigma_psf с PSF (худшая по фазе), логарифмическая ось; тонкие
горизонтальные линии — уровни методов из E0, серая полоса — уровень
симулятора; одна линия без PSF — центр тяжести (по оси x тоже b / 0,7
пикс.).
Таблицы: e2d_table.csv — граница b / sigma_psf каждого метода с PSF и
без (b_over_sigma_limit; пометка "> x" — ошибка больше уровня во всём
диапазоне, "< x" — нигде) и второй критерий b_over_sigma_growth: ниже
него худшая ошибка больше GROWTH = 2 её значений у широкого образа
(θ <= 66°, err_wide_px); e2d_points_table.csv — все точки.

Итог (с PSF, худшая фаза): центр тяжести и квадрат весов — на уровне
симулятора при всех b (до 7e-4 пикс.), gauss2d — не выше уровня E0. У
порога 50 % и эллипса ошибка выше уровня E0 уже при θ = 60° (0,011 и
0,005 пикс.: двумерная фаза наклонного образа), а ниже b / sigma_psf ≈ 4
... 7 удваивается и при b / sigma_psf = 1,7 доходит до 0,075 и 0,079 пикс.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # запуск без установки

import argparse
import itertools
import math
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from photosim.analysis import method_own_bias, point_at_pixel
from photosim.estimators import ellipse_from_conic
from photosim.experiment import (
    Config,
    Effects,
    Run,
    build,
    load_run,
    run_configs,
    save_run,
)
from photosim.figures import style
from photosim.figures.cli import Output, main
from photosim.oracle import _image_conic
from photosim.presets import camera

NAME = "e2d"
CAMERA = "review_16mm"
Z = 1.0
RADIUS = 0.005
THETAS = tuple(range(60, 88, 3))  # 60, 63, ..., 87°
SS, SS_CHECK = 64, 128
CHECK_THETAS = (84, 87)  # самые узкие образы: повтор при SS_CHECK
PHASES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)  # сдвиг центра от оси, пикс., по u и v
CHECK_PHASE = (0.3, 0.3)
# Широкий образ — θ <= WIDE_MAX_DEG (b / sigma_psf >= 13): у порога 50 % и
# эллипса худшая по фазе ошибка выше уровня E0 (круг, фаза только по u) уже
# там, поэтому второй критерий — рост в GROWTH раз от широкого образа
WIDE_MAX_DEG = 66.0
GROWTH = 2.0
# Уровень симулятора, пикс.: как в E3 и E6 (при ss = 16, круг); у узкого
# образа при ss = 64 — не хуже (проверка ss = 128)
SIM_LEVEL = 2e-3


def semi_axes(config: Config) -> tuple[float, float]:
    """Точные полуоси образа (большая, малая), пикс."""
    s = build(config)
    C = _image_conic(
        s.camera,
        s.pose,
        s.motion.position(s.t_ref),
        s.motion.orientation(s.t_ref),
        s.target.radius,
    )
    major, minor = ellipse_from_conic(C)[1]
    return float(major), float(minor)


def run(args: argparse.Namespace) -> Path:
    cam = camera(CAMERA)
    thetas, phases = (THETAS[::3], PHASES[::2]) if args.quick else (THETAS, PHASES)
    base = Config(
        camera=CAMERA,
        radius=RADIUS,
        tilt_axis="y",
        effects=Effects(
            exposure=False,  # марка неподвижна
            motion=False,
            rotation=False,
            rolling_shutter=False,
            shot_noise=False,
            read_noise=False,
            quantization=False,
        ),
        n_mc=2,
        reference="conic_mid",
        ss=SS,
    )
    configs, labels = [], []
    for psf, theta, du, dv in itertools.product((True, False), thetas, phases, phases):
        check = theta in CHECK_THETAS and (du, dv) == CHECK_PHASE
        for ss in (SS, SS_CHECK) if check else (SS,):
            cfg = replace(
                base,
                position=point_at_pixel(cam, cam.cx + du, cam.cy + dv, Z),
                tilt_deg=float(theta),
                ss=ss,
                effects=replace(base.effects, psf=psf),
            )
            major, minor = semi_axes(cfg)
            configs.append(cfg)
            labels.append(
                {"du": du, "dv": dv, "semi_major_px": major, "semi_minor_px": minor}
            )
    df = run_configs(configs, labels=labels, seed=base.seed, n_jobs=args.jobs)
    configs_json = {
        "experiment": NAME,
        "quick": args.quick,
        "base": base,
        "thetas_deg": list(thetas),
        "phases_px": list(phases),
        "check_phase_px": list(CHECK_PHASE),
        "check_thetas_deg": list(CHECK_THETAS),
        "ss_check": SS_CHECK,
    }
    name = NAME + ("_quick" if args.quick else "")
    return save_run(df, configs_json, name, args.root)


# Обработка


def with_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["status"] == "ok"].copy()
    df["err_px"] = np.hypot(df["err0_u_px"], df["err0_v_px"])
    df["psf"] = df["effects.psf"].astype(bool)
    # sigma PSF — параметр сенсора: и у точек без PSF ось x та же, b / 0,7
    df["b_over_sigma"] = df["semi_minor_px"] / df["sensor.psf_sigma_px"]
    return df


def worst(df: pd.DataFrame) -> pd.DataFrame:
    """Худшая по фазам точка: строка на метод, PSF и наклон (ss = SS).

    Колонки ``du``, ``dv`` — фаза худшей точки; ``err_axis_px`` — ошибка
    строго на оси (du = dv = 0, симметричная сетка).
    """
    d = df[df["ss"] == SS]
    keys = ["method", "psf", "tilt_deg"]
    out = d.loc[d.groupby(keys)["err_px"].idxmax()].copy()
    axis = d[(d["du"] == 0) & (d["dv"] == 0)].set_index(keys)["err_px"]
    out["err_axis_px"] = axis.reindex(pd.MultiIndex.from_frame(out[keys])).to_numpy()
    return out


def levels(df: pd.DataFrame, e0: Run | None) -> dict[bool, dict[str, float]]:
    """Собственное смещение методов из E0 при радиусе образа точки, с PSF и без."""
    if e0 is None:
        return {True: {}, False: {}}
    R = float(df["radius_px"].iloc[0])
    return {psf: method_own_bias(e0.df, R, psf) for psf in (True, False)}


def limit(x: np.ndarray, y: np.ndarray, level: float) -> tuple[float, str]:
    """``x``, ниже которого ``y > level`` до конца сетки, и пометка.

    ``x`` перебирается по убыванию (образ сужается); граница — между
    последним узлом не выше ``level`` и следующим, линейно по ``x`` и в
    логарифме по ``y``. Отдельные выходы при широком образе не считаются.
    """
    order = np.argsort(x)[::-1]
    x, y = x[order], np.maximum(y[order], 1e-12)
    below = np.flatnonzero(y <= level)
    if below.size == 0:
        return math.nan, f"> {x[0]:.3g}"  # больше уровня при всех b
    i = int(below[-1])
    if i == len(x) - 1:
        return math.nan, f"< {x[-1]:.3g}"  # не выходит за уровень
    t = (math.log(level) - math.log(y[i])) / (math.log(y[i + 1]) - math.log(y[i]))
    return float(x[i] + t * (x[i + 1] - x[i])), ""


def table_limits(
    df: pd.DataFrame, own: dict[bool, dict[str, float]], out: Output
) -> pd.DataFrame:
    rows = []
    for (m, psf), d in worst(df).groupby(["method", "psf"], sort=False):
        d = d.sort_values("b_over_sigma")
        level_e0 = own[psf].get(m, math.nan)
        level = max(SIM_LEVEL, level_e0) if math.isfinite(level_e0) else SIM_LEVEL
        x, y = d["b_over_sigma"].to_numpy(), d["err_px"].to_numpy()
        x_lim, note = limit(x, y, level)
        wide = float(d.loc[d["tilt_deg"] <= WIDE_MAX_DEG, "err_px"].max())
        x_grow, note_grow = limit(x, y, GROWTH * wide)
        sigma = float(d["sensor.psf_sigma_px"].iloc[0])
        narrow = d.iloc[0]
        rows.append(
            {
                "method": m,
                "psf": psf,
                "own_level_e0_px": level_e0,
                "level_px": level,
                "b_over_sigma_limit": x_lim,
                "note": note,
                "b_limit_px": x_lim * sigma,
                "theta_limit_deg": float(
                    np.interp(x_lim, d["b_over_sigma"], d["tilt_deg"])
                )
                if note == ""
                else math.nan,
                "err_widest_px": float(d["err_px"].iloc[-1]),
                "err_wide_px": wide,
                "b_over_sigma_growth": x_grow,
                "note_growth": note_grow,
                "b_narrowest_px": float(narrow["semi_minor_px"]),
                "err_narrowest_px": float(narrow["err_px"]),
                "err_narrowest_on_axis_px": float(narrow["err_axis_px"]),
            }
        )
    table = pd.DataFrame(rows)
    table["method"] = pd.Categorical(table["method"], style.ordered(table["method"]))
    table = table.sort_values(["psf", "method"], ascending=[False, True])
    out.table(table, NAME)
    return table


def plot(run: Run, out: Output, args: argparse.Namespace) -> None:
    df = with_metrics(run.df)
    e0 = load_run(args.e0) if args.e0 is not None else None
    if e0 is None:
        print("  --e0 не указан: уровень методов — только уровень симулятора")
    own = levels(df, e0)
    methods = style.ordered(df["method"].unique())
    w = worst(df)
    on, off = w[w["psf"]], w[~w["psf"]]

    fig, ax = plt.subplots(figsize=(style.COLUMN, 3.6))
    ax.axhspan(1e-6, SIM_LEVEL, color="0.92", zorder=0, linewidth=0)
    for m in methods:
        st = style.method_style(m)
        d = on[on["method"] == m].sort_values("b_over_sigma")
        ax.plot(d["b_over_sigma"], d["err_px"], **st)
        if m in own[True]:
            ax.axhline(own[True][m], color=st["color"], linestyle=st["linestyle"],
                       linewidth=0.5, zorder=1)
    c = off[off["method"] == "centroid"].sort_values("b_over_sigma")
    ax.plot(c["b_over_sigma"], c["err_px"], color="0.6", linestyle="--", marker="o",
            markerfacecolor="white", label="Centroid, no PSF")
    ax.set_yscale("log")
    ax.set_ylim(max(1e-5, 0.3 * float(on["err_px"].min())), 2 * float(w["err_px"].max()))
    ax.set_xlim(0, 1.05 * float(w["b_over_sigma"].max()))
    ax.set_xlabel("Minor semi-axis of the image, $b / \\sigma_{PSF}$")
    ax.set_ylabel("Noise-free error vs conic center, worst phase (px)")
    # сверху — наклон: на оси b = k cos θ (k ≈ R / sigma_psf, по точкам)
    d = on[on["method"] == methods[0]]
    k = float((d["b_over_sigma"] / np.cos(np.radians(d["tilt_deg"]))).mean())
    top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda x: np.degrees(np.arccos(np.clip(np.asarray(x) / k, 0.0, 1.0))),
            lambda t: k * np.cos(np.radians(t)),
        ),
    )
    top.set_xticks([t for t in run.configs["thetas_deg"] if t % 6 == 0 or t == 87])
    top.set_xlabel("Tilt $\\theta$ (deg)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="0.0", linewidth=0.5))
    labels.append("Level of the method in E0")
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.92"))
    labels.append(f"Simulator level, {SIM_LEVEL:g} px")
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    out.figure(fig, "e2d_minor_axis")

    points = df[[
        "tilt_deg", "du", "dv", "psf", "ss", "method", "semi_major_px",
        "semi_minor_px", "b_over_sigma", "err0_u_px", "err0_v_px", "err_px",
    ]].copy()
    points["method"] = pd.Categorical(points["method"], methods)
    out.table(
        points.sort_values(["psf", "ss", "method", "tilt_deg", "du", "dv"]), "e2d_points"
    )
    table = table_limits(df, own, out)
    check(df, table, run, out)


def check(df: pd.DataFrame, table: pd.DataFrame, run: Run, out: Output) -> None:
    c = df[(df["method"] == "centroid") & df["psf"] & (df["ss"] == SS)]
    bad = c.loc[c["err_px"].idxmax()]
    out.check(
        bool((c["err_px"] <= SIM_LEVEL).all()),
        f"centroid с PSF на уровне симулятора при b = {c['semi_minor_px'].min():.2f} ... "
        f"{c['semi_minor_px'].max():.1f} пикс. и всех фазах: до {c['err_px'].max():.1e} "
        f"пикс. (b = {bad['semi_minor_px']:.2f}; порог {SIM_LEVEL:g})",
    )
    # ss = 32 в самых узких точках: дискретизация рендера узкого эллипса
    du, dv = run.configs["check_phase_px"]
    at = df[df["tilt_deg"].isin(CHECK_THETAS) & (df["du"] == du) & (df["dv"] == dv)]
    for (m, psf, theta), d in at.groupby(["method", "psf", "tilt_deg"], sort=False):
        e = d.set_index("ss")["err_px"]
        if SS_CHECK not in e:
            continue
        text = (
            f"{m}, {'с PSF' if psf else 'без PSF'}, θ = {theta:g}° "
            f"(b = {d['semi_minor_px'].iloc[0]:.2f} пикс.), фаза ({du:g}, {dv:g}): "
            f"ss = {SS} — {e[SS]:.2e}, ss = {SS_CHECK} — {e[SS_CHECK]:.2e} пикс."
        )
        if m == "centroid":
            out.check(abs(e[SS_CHECK] - e[SS]) <= SIM_LEVEL, text)
        else:
            print(f"  {text}")
    for _, row in table.iterrows():
        where = row["note"] or f"b / sigma_psf < {row['b_over_sigma_limit']:.2f}"
        print(
            f"  {row['method']}, {'с PSF' if row['psf'] else 'без PSF'}: уровень "
            f"{row['level_px']:.1e} пикс., ошибка больше уровня при {where}; "
            f"у самого узкого образа (b = {row['b_narrowest_px']:.2f}) "
            f"{row['err_narrowest_px']:.1e} пикс. (на оси {row['err_narrowest_on_axis_px']:.1e}); "
            f"рост в {GROWTH:g} раза от широкого ({row['err_wide_px']:.1e}) при "
            f"{row['note_growth'] or format(row['b_over_sigma_growth'], '.2f')}"
        )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--e0", type=Path, default=None,
        help="папка запуска E0: собственный уровень методов",
    )


if __name__ == "__main__":
    sys.exit(main("E2d: граница по малой полуоси", run, plot, add_arguments=add_arguments))
