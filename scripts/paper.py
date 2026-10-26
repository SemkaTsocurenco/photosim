"""Рисунки и таблицы статьи: из папок запусков в одну папку (paper/).

    python scripts/paper.py plot [--runs paper/runs.json] [--out paper]

Ничего не считает: читает папки запусков из runs.json (ключи e0, e1, e2,
e2d, e3, e6, budget) и строит всё функциями самих экспериментов. Только
рис. 1б заново рендерит одну точку E3 без шума по её строке таблицы
(experiment.config_from_row), как scripts/inspect_point.py.

В папке:

- fig1_simulator ... fig6_budget — PDF (вектор) и PNG 600 dpi, ширина
  style.COLUMN или style.DOUBLE, шрифт STIX 8 pt;
- table1_parameters, table2_methods — CSV и LaTeX (.tex);
- captions.tex — окружения figure с подписями, text.md — фразы для текста
  статьи; числа в них — из тех же запусков;
- README.md, runs.json, references.bib ведутся вручную.

Проверки (как в экспериментах) печатаются как [OK] / [FAIL]; при [FAIL]
код возврата 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # запуск без установки

import argparse
import json
import math
from dataclasses import replace

import budget
import e0_pixel_phase as e0
import e1_precision as e1
import e2_eccentricity as e2
import e2d_minor_axis as e2d
import e3_blur as e3
import e6_rolling as e6
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_point import ellipse_points
from matplotlib.figure import Figure

from photosim.analysis import method_own_bias
from photosim.estimators import ellipse_from_conic, estimate
from photosim.experiment import (
    Run,
    _render,
    build,
    config_from_row,
    load_run,
    references,
)
from photosim.figures import style
from photosim.oracle import _image_conic
from photosim.presets import CAMERAS

RUN_KEYS = ("e0", "e1", "e2", "e2d", "e3", "e6", "budget")
PAPER_DIR = Path("paper")
FIG1_POINT = {"tilt_deg": 45.0, "direction_deg": 0.0, "L_px": 8.0}  # рис. 1б, E3
R_REVIEW_PX = 23.19  # марка r = 5 мм на 1 м у review_16mm
E6_SPEED = 1.0  # м/с, остаток при построчной привязке в табл. II
# E2d при ss = 16 (худшая фаза, центр тяжести; docs/conventions.md, «Точность
# симулятора»): до ~2e-3 пикс. при b >= 4,8 пикс., ниже растёт до 6,6e-3
SS16_B_LIMIT_PX = 5.0
SS16_WORST_PX = 6.6e-3
# Приближение эксцентриситета (формула (3) обзора, ф. (11) Liebold & Maas)
EQ_APPROX = r"""\begin{equation}
  \boldsymbol{\varepsilon} \approx \frac{f\,r^2 n_z}{Z^2}
  \begin{bmatrix} n_x \\ n_y \end{bmatrix}
  \label{eq:ecc_approx}
\end{equation}"""


class Sink:
    """Вместо ``cli.Output``: таблицы экспериментов нужны как данные, без файлов."""

    def table(self, df: pd.DataFrame, name: str, header: str | None = None) -> None:
        pass

    def figure(self, fig: Figure, name: str) -> None:
        plt.close(fig)

    def check(self, ok: bool, message: str) -> bool:
        return bool(ok)


class Paper:
    """Файлы статьи и проверки."""

    def __init__(self, out: Path) -> None:
        self.dir = out
        self.failed: list[str] = []
        out.mkdir(parents=True, exist_ok=True)

    def figure(self, fig: Figure, name: str) -> None:
        for path in style.save(fig, name, self.dir):
            print(f"рисунок: {path}")
        plt.close(fig)

    def text(self, name: str, content: str) -> None:
        path = self.dir / name
        path.write_text(content)
        print(f"файл: {path}")

    def check(self, ok: bool, message: str) -> None:
        print(f"[{'OK' if ok else 'FAIL'}] {message}")
        if not ok:
            self.failed.append(message)


def fmt(x: float, digits: int = 2) -> str:
    """Число для таблицы: ``digits`` значащих цифр, без экспоненты при 1e-4 ... 1e4."""
    if not math.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    if 1e-4 <= abs(x) < 1e4:
        decimals = max(digits - 1 - math.floor(math.log10(abs(x))), 0)
        return f"{x:.{decimals}f}"
    return f"{x:.{digits - 1}e}"


def tex_number(text: str) -> str:
    """Число из ``fmt`` для LaTeX: экспонента и знаки сравнения — в математике."""
    if "e" in text and text[0].isdigit():
        mant, exp = text.split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    return text.replace("< ", "$<$").replace("> ", "$>$")


# Рис. 1


def sci(x: float) -> str:
    """Число для подписи на рисунке: 1.1×10⁻³."""
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}×10$^{{{int(exp)}}}$"


def span(ax, p, q, text: str, above: bool) -> None:
    """Двойная стрелка между точками ``p``, ``q`` и подпись над или под ней."""
    ax.annotate("", tuple(q), tuple(p),
                arrowprops={"arrowstyle": "<->", "color": "0.4", "linewidth": 0.6,
                            "shrinkA": 3, "shrinkB": 3})
    mid = (np.asarray(p) + np.asarray(q)) / 2
    ax.annotate(text, tuple(mid), xytext=(0, 6 if above else -6),
                textcoords="offset points", ha="center",
                va="bottom" if above else "top", fontsize=style.SMALL_FONT_SIZE)


def fig1(e3_run: Run) -> tuple[Figure, dict]:
    """Рис. 1 (одна колонка): кадр наклонной смазанной марки и эталоны центра;
    справа — центры относительно центра тяжести в двух масштабах. Структура
    симулятора — в тексте, параметры — в табл. I."""
    df = e3_run.df[e3_run.df["status"] == "ok"]
    sel = np.ones(len(df), dtype=bool)
    for key, value in FIG1_POINT.items():
        sel &= np.isclose(df[key], value)
    row = df[sel].iloc[0]
    s = build(config_from_row(row))
    res = _render(s, s.motion)
    clean = replace(
        s.sensor, psf=False, shot_noise=False, read_noise=False, quantization=False
    )
    img = clean.expose(res.irradiance, s.tau, np.random.default_rng(0))
    refs = references(s)[0]
    centroid = estimate(
        "centroid", img, res.roi, black_level_dn=s.sensor.black_level_dn
    ).uv
    u0, v0, w, h = res.roi

    small = style.SMALL_FONT_SIZE
    fig = plt.figure(figsize=(style.COLUMN, 3.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1.0], height_ratios=[1, 1, 0.42])
    ax_b = fig.add_subplot(gs[:2, 0])
    ax_z1, ax_z2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[2, :])

    extent = (u0 - 0.5, u0 + w - 0.5, v0 + h - 0.5, v0 - 0.5)
    ax_b.imshow(img, cmap="gray", extent=extent, interpolation="nearest")
    for t, ls in ((0.0, ":"), (s.tau / 2, "--"), (s.tau, ":")):
        C = _image_conic(s.camera, s.pose, s.motion.position(t),
                         s.motion.orientation(t), s.target.radius)
        center, axes, angle = ellipse_from_conic(C)
        ax_b.plot(*ellipse_points(center, axes, angle), color="0.5", linestyle=ls,
                  linewidth=0.7)
    ax_b.plot(*centroid, linestyle="", marker="o", markersize=3, color="0.0",
              markeredgecolor="white", markeredgewidth=0.5)
    ax_b.set_xlim(extent[:2])
    ax_b.set_ylim(extent[2:])
    ax_b.grid(False)
    ax_b.set_xlabel("$u$ (px)")
    ax_b.set_ylabel("$v$ (px)")
    ax_b.set_title(f"Tilt {row['tilt_deg']:g}°, blur L = {row['L_px']:g} px", loc="left")
    ax_b.tick_params(labelsize=small)

    # центры: смещения от центра тяжести в двух масштабах — эксцентриситет
    # (0,06 пикс.) и три почти совпадающих центра (1e-3 пикс.)
    rel = {
        "centroid": np.zeros(2),
        "exposure": refs["exposure_conic"] - centroid,
        "mid": refs["conic_mid"] - centroid,
        "proj": refs["proj_mid"] - centroid,
    }
    marks = {
        "centroid": ("Image centroid", {"marker": "o", "markerfacecolor": "0.0"}),
        "exposure": ("Exposure-averaged conic center", {"marker": "x", "markersize": 6}),
        "mid": ("Conic center, mid-exposure", {"marker": "+", "markersize": 8}),
        "proj": ("Projected center, mid-exposure",
                 {"marker": "s", "markerfacecolor": "white"}),
    }
    near = max(float(np.hypot(*rel[k])) for k in ("exposure", "mid"))
    for ax, scale, half, unit in (
        (ax_z1, 1.0, 1.3 * float(np.hypot(*rel["proj"])), "px"),
        (ax_z2, 1e3, 1.6 * near * 1e3, "$10^{-3}$ px"),
    ):
        for key, (label, st) in marks.items():
            d = rel[key] * scale
            ax.plot(d[0], d[1], linestyle="", color="0.0", label=label,
                    **{"markersize": 4, **st})
        ax.set_xlim(-half, half)
        ax.set_ylim(half, -half)  # v вниз, как в кадре
        ax.set_aspect("equal")
        ax.set_xlabel(f"$\\Delta u$ ({unit})", labelpad=1)
        ax.set_ylabel(f"$\\Delta v$ ({unit})", labelpad=1)
        ax.tick_params(labelsize=small)
    ax_z1.set_title("Centers vs centroid", loc="left")
    ax_z2.set_title("Same, zoomed", loc="left")
    ecc = float(np.hypot(*(refs["conic_mid"] - refs["proj_mid"])))
    blur = float(np.hypot(*(rel["exposure"] - rel["mid"])))
    sim = float(np.hypot(*rel["exposure"]))
    span(ax_z1, rel["proj"], rel["mid"], f"$\\varepsilon$ = {ecc:.3f} px", above=True)
    span(ax_z2, rel["mid"] * 1e3, rel["exposure"] * 1e3, f"{sci(blur)} px",
         above=True)
    span(ax_z2, rel["centroid"] * 1e3, rel["exposure"] * 1e3, f"{sci(sim)} px",
         above=False)
    ax_leg.set_axis_off()
    ax_leg.legend(*ax_z1.get_legend_handles_labels(), loc="center", ncols=2,
                  fontsize=small, handlelength=1.0, columnspacing=0.8)
    info = {
        "L_px": float(row["L_px"]),
        "tilt_deg": float(row["tilt_deg"]),
        "R_px": float(row["radius_px"]),
        "eccentricity_px": ecc,
        "centroid_vs_exposure_px": float(np.hypot(*(centroid - refs["exposure_conic"]))),
        "exposure_vs_mid_px": float(
            np.hypot(*(refs["exposure_conic"] - refs["conic_mid"]))
        ),
    }
    return fig, info


# Рис. 5


def fig5(e3_run: Run, e6_run: Run) -> Figure:
    """Рис. 5: (а, б) E3 — смаз, (в, г) E6 — построчный затвор."""
    d3 = e3.main_case(e3.with_metrics(e3_run.df))
    d6 = e6.with_metrics(e6_run.df)
    t_row0 = float(e6_run.configs["t_row_estimate_s"])
    methods = style.ordered(d3["method"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE, 5.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    e3.plot_bias(ax_a, d3, methods, "bias_along")
    ax_a.set_ylabel("Bias along the motion (px)")
    ax_a.set_title("(a) Blur: bias vs exposure-averaged center", loc="left")
    e3.plot_std(ax_b, d3, methods, "along")
    ax_b.set_ylabel("Std along the motion (px)")
    ax_b.set_title("(b) Blur: standard deviation along the motion", loc="left")
    for ax in (ax_a, ax_b):
        ax.set_xlabel("Blur length $L = f v \\tau / Z$ (px)")
        ax.set_xlim(-0.5, float(d3["L_px"].max()) + 0.5)
    e6.panel_frame(ax_c, d6, t_row0)
    ax_c.set_title("(c) Rolling shutter: frame-time stamp", loc="left")
    e6.panel_residual(ax_d, d6)
    ax_d.set_title("(d) Rolling shutter: row-time stamp", loc="left")
    handles, labels = ax_b.get_legend_handles_labels()
    h_d, l_d = ax_d.get_legend_handles_labels()
    extra = [(h, lab) for h, lab in zip(h_d, l_d, strict=True) if lab not in labels]
    handles += [h for h, _ in extra]
    labels += [lab for _, lab in extra]
    fig.legend(handles, labels, loc="outside lower center", ncols=4)
    return fig


# Таблицы


def table1(runs: dict[str, Run]) -> tuple[pd.DataFrame, str]:
    """Табл. I: параметры моделирования — из конфигураций запусков.

    Камеры, сенсоры и сценарий бюджета (рис. 6) — из запуска бюджета;
    исключения рендера (ss, n_mc) — из конфигураций E0, E1, E2, E2d.
    """
    bases = runs["budget"].configs["bases"]
    cols = {}
    for key, cfg in bases.items():
        s = build(cfg)
        cam = CAMERAS[cfg.camera]
        sensor = s.sensor
        rolling = cfg.shutter == "rolling"
        binned = cfg.camera.endswith("2x2")
        source = "assumed" if key == "rpi_cm3" else "EMVA 1288"
        u, v = s.camera.project(s.pose.to_camera(s.motion.position(s.t_ref)))
        velocity = np.asarray(cfg.motion["velocity"])
        cols[key] = [
            (f"{s.camera.fx:.0f}{' (2×2 binning)' if binned else ''}; "
             f"{cam['width']}×{cam['height']}"),
            f"{cam['pixel_um']:g}{' (after binning)' if binned else ''}",
            f"rolling, t_row = {cfg.t_row * 1e6:.1f} µs (estimate)" if rolling else "global",
            f"{sensor.full_well_e:.0f} ({source})",
            f"{sensor.read_noise_e:g} ({source})",
            f"{sensor.bits} bit"
            + (f", black level {sensor.black_level_dn:.0f} DN" if sensor.black_level_dn else ""),
            f"{sensor.gain_e_per_dn:.2f}",
            f"{sensor.psf_sigma_px:g} (assumed)",
            # сценарий бюджета: от него зависят числа рис. 6
            (f"τ = {cfg.tau * 1e3:g} ms, v = {np.linalg.norm(velocity):g} m/s along u, "
             f"ω = {cfg.rotation['omega']:g} rad/s, θ = {cfg.tilt_deg:g}°; frame "
             f"corner, (u, v) = ({u:.0f}, {v:.0f}) px"),
        ]
    base = bases["review_16mm"]
    radii = sorted({float(r) for r in runs["e2"].configs["a"]["radii_m"]})
    other = [f"{r * 1e3:g} mm" for r in radii if not np.isclose(r, base.radius)]
    target = (
        f"r = {base.radius * 1e3:g} mm{f' (and {other[0]})' if other else ''}, "
        f"Z = {budget.Z:g} m, fill {base.fill:g}"
    )
    render = (
        f"ss = {base.ss}, n_t = {base.nt}, n_mc = {base.n_mc} (unless stated otherwise)"
    )
    names = [
        "Focal length (px); frame (px)", "Pixel pitch (µm)", "Shutter",
        "Full well (e⁻)", "Read noise (e⁻)", "ADC", "Gain (e⁻/DN)", "PSF σ (px)",
        "Error budget scenario (Fig. 6)",
    ]
    table = pd.DataFrame({
        "parameter": [*names, "Target", "Rendering"],
        "16 mm / IMX264": [*cols["review_16mm"], target, render],
        "RPi CM3 / IMX708": [*cols["rpi_cm3"], "same", "same"],
    })
    # исключения рендера — по конфигурациям запусков
    ss_e0 = runs["e0"].configs["base"].ss
    ss_e2d = runs["e2d"].configs["base"].ss
    ss = f"{ss_e0}" if ss_e0 == ss_e2d else f"{ss_e0} and {ss_e2d}"
    n_free = {
        runs["e0"].configs["base"].n_mc, runs["e2"].configs["base_b"].n_mc,
        runs["e2d"].configs["base"].n_mc,
    }
    n_gauss = runs["e1"].configs["base_gauss2d"].n_mc
    note = (
        f"E0, E2 and E2d are noise-free (n_mc = {'/'.join(map(str, sorted(n_free)))}); "
        f"E0 and E2d: ss = {ss}; E1: n_mc = {n_gauss} for the 2D Gaussian fit."
    )
    return table, note


def table2(runs: dict[str, Run]) -> tuple[pd.DataFrame, dict]:
    """Табл. II: сравнение методов; числа — из тех же функций, что *_table.csv."""
    sink = Sink()
    own = method_own_bias(runs["e0"].df, R_REVIEW_PX)

    d1 = runs["e1"].df
    d1 = d1[(d1["status"] == "ok") & np.isclose(d1["fill"], 0.7)]
    r1 = d1["radius_px"].unique()
    d1 = d1[np.isclose(d1["radius_px"], r1[np.argmin(np.abs(r1 - R_REVIEW_PX))])]
    std = d1.groupby("method")["std_u_px"].first()

    thr = e3.table_threshold(e3.with_metrics(runs["e3"].df), runs["e0"], sink)
    thr = thr[np.isclose(thr["tilt_deg"], e3.MAIN["tilt_deg"])
              & np.isclose(thr["direction_deg"], e3.MAIN["direction_deg"])]
    thr = thr.set_index("method")

    d2 = e2d.with_metrics(runs["e2d"].df)
    lim = e2d.table_limits(d2, e2d.levels(d2, runs["e0"]), sink)
    lim = lim[lim["psf"]].set_index("method")

    d6 = e6.with_metrics(runs["e6"].df)
    est = d6[np.isclose(d6["t_row_factor"], 1.0) & np.isclose(d6["speed"], E6_SPEED)]
    resid = est.groupby("method")["resid_px"].max()

    contrib = budget.contributions(runs["budget"])
    nonadd = contrib[(contrib["camera"] == "review_16mm")
                     & (contrib["item"] == "non_additivity")].set_index("method")

    x_min, x_max = float(d2["b_over_sigma"].min()), float(d2["b_over_sigma"].max())

    def edge(note: str) -> str:
        """Пометка E2d по краю всей сетки: у методов край чуть разный (фаза)."""
        if note.startswith("<"):
            return f"< {x_min:.1f}"
        return f"> {x_max:.1f}" if note.startswith(">") else note

    rows = []
    for m in style.ordered(own):
        t = thr.loc[m]
        b = lim.loc[m]
        rows.append({
            "method": style.METHOD_STYLE[m]["label"],
            "own_bias_e0_px": own[m],
            "std_fill0.7_e1_px": float(std[m]),
            "L_limit_e3_px": float(t["L_exceed_noise_free_px"]),
            "L_limit_note": t["note_noise_free"],
            "b_over_sigma_limit_e2d": float(b["b_over_sigma_limit"]),
            "b_over_sigma_note": edge(b["note"]),
            "b_over_sigma_growth_e2d": float(b["b_over_sigma_growth"]),
            "err_at_narrowest_e2d_px": float(b["err_narrowest_px"]),
            "resid_row_stamp_1ms_e6_px": float(resid[m]),
            "non_additivity_16mm_um": float(nonadd.loc[m, "sys_um"]),
        })
    facts = {"x_min": x_min, "x_max": x_max, "lim": lim}
    return pd.DataFrame(rows), facts


def table2_tex(t: pd.DataFrame, x_min: float) -> str:
    def limit_cell(value: float, note: str) -> str:
        return tex_number(note) if note else fmt(value)

    lines = [
        "% табл. II, scripts/paper.py",
        "\\begin{table*}[t]",
        "\\centering",
        ("\\caption{Comparison of the center estimators. Own bias: noise-free, worst "
        "over the sub-pixel phase, R = 23 px, PSF $\\sigma$ = 0.7 px (E0). Std: fill "
        "0.7 (E1). $L$: blur length above which the noise-free bias along the motion "
        "exceeds the method level, tilt 45$^\\circ$ (E3). $b/\\sigma_{PSF}$: minor semi-axis "
        "below which the worst-phase error exceeds the method level (E2d). Residual: "
        "row-time stamp, worst over rows and directions (E6). Non-additivity: error "
        "budget of the 16 mm camera.}"),
        "\\label{tab:methods}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        ("Method & Own bias & Std, fill 0.7 & $L$ limit & $b/\\sigma_{PSF}$ limit "
        "& Residual, 1 m/s & Non-additivity \\\\"),
        " & (px) & (px) & (px) & & (px) & ($\\mu$m) \\\\",
        "\\hline",
    ]
    for _, r in t.iterrows():
        b = limit_cell(r["b_over_sigma_limit_e2d"], r["b_over_sigma_note"])
        if r["b_over_sigma_note"].startswith(">") and math.isfinite(
            r["b_over_sigma_growth_e2d"]
        ):
            b += f" ({fmt(r['b_over_sigma_growth_e2d'])})$^a$"
        cells = [
            r["method"].replace("%", "\\%"),
            tex_number(fmt(r["own_bias_e0_px"])),
            tex_number(fmt(r["std_fill0.7_e1_px"])),
            limit_cell(r["L_limit_e3_px"], r["L_limit_note"]),
            b,
            tex_number(fmt(r["resid_row_stamp_1ms_e6_px"])),
            tex_number(fmt(r["non_additivity_16mm_um"])),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\hline",
        "\\end{tabular}",
        "",
        "\\smallskip",
        "\\begin{minipage}{\\textwidth}\\footnotesize",
        ("Method level: max(own bias, simulator level 0.002 px). $>$: exceeded over "
         "the whole range; $<$: not exceeded down to the smallest value tested "
         f"($b/\\sigma_{{PSF}}$ = {x_min:.1f}). $^a$In parentheses: "
         "$b/\\sigma_{PSF}$ below which the worst-phase error doubles relative to "
         "wide images ($\\theta \\le 66^\\circ$)."),
        "\\end{minipage}",
        "\\end{table*}",
        "",
    ]
    return "\n".join(lines)


def table1_tex(t: pd.DataFrame, note: str) -> str:
    def esc(text: str) -> str:
        return (
            text.replace("µ", "$\\mu$").replace("e⁻", "e$^-$").replace("σ", "$\\sigma$")
            .replace("×", "$\\times$").replace("t_row", "$t_{row}$")
            .replace("n_t", "$n_t$").replace("n_mc", "$n_{mc}$")
            .replace("τ", "$\\tau$").replace("ω", "$\\omega$").replace("θ", "$\\theta$")
            .replace("°", "$^\\circ$")
        )

    lines = [
        "% табл. I, scripts/paper.py",
        "\\begin{table}[t]",
        "\\centering",
        ("\\caption{Simulation parameters. EMVA 1288: measured data of the camera "
         "\\cite{flir_bfs_u3_50s5_emva} according to \\cite{emva1288}; assumed: no "
         "measured data; Raspberry Pi mode and frame rate: \\cite{rpi_camera_docs}.}"),
        "\\label{tab:parameters}",
        "\\begin{tabular}{p{0.25\\columnwidth}p{0.28\\columnwidth}p{0.31\\columnwidth}}",
        "\\hline",
        "Parameter & 16 mm / IMX264 & RPi CM3 / IMX708 \\\\",
        "\\hline",
    ]
    for _, r in t.iterrows():
        # \\raggedright в ячейке p{} — без пакета array, конец строки — \\tabularnewline
        lines.append(
            " & ".join(f"\\raggedright {esc(str(v))}" for v in r) + " \\tabularnewline"
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        "",
        "\\smallskip",
        f"\\begin{{minipage}}{{\\columnwidth}}\\footnotesize {esc(note)}\\end{{minipage}}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


# Подписи и фразы


def tex_text(text: str) -> str:
    """Текст подписи для LaTeX: % — не комментарий, градус — в математике."""
    return text.replace("%", "\\%").replace("°", "$^\\circ$")


def figure_env(name: str, label: str, caption: str, wide: bool) -> str:
    caption = tex_text(caption)
    env = "figure*" if wide else "figure"
    width = "\\textwidth" if wide else "\\columnwidth"
    return (
        f"\\begin{{{env}}}[t]\n\\centering\n\\includegraphics[width={width}]{{{name}}}\n"
        f"\\caption{{{caption}}}\n\\label{{fig:{label}}}\n\\end{{{env}}}\n"
    )


def captions_and_text(runs: dict[str, Run], info1: dict, t2: pd.DataFrame,
                      facts2: dict, paper: Paper) -> tuple[str, str]:
    e2_df = runs["e2"].df
    sel = e2_df[(e2_df["case"] == "tilt") & np.isclose(e2_df["theta_deg"], e2.MAP_THETA)
                & np.isclose(e2_df["r_m"], e2.MAP_RADIUS)]
    k = (e2.MAP_RADIUS / e2.Z) ** 2 * math.sin(math.radians(e2.MAP_THETA)) ** 2
    worst1 = float(sel["rel_err1"].max())
    paper.check(
        worst1 <= 1.05 * k and k < 1e-4,
        f"рис. 4: ошибка eps1 по кадру до {worst1:.2e}, k = (r/Z)^2 sin^2 θ = {k:.2e}",
    )

    contrib = budget.contributions(runs["budget"])
    c = contrib[contrib["method"] == "centroid"].set_index(["camera", "item"])
    persp = {cam: float(c.loc[(cam, "perspective"), "sys_um"]) for cam in budget.CAMERAS}
    eq3 = float(c.loc[("review_16mm", "perspective"), "analytic_um"])
    start = float(c.loc[("review_16mm", "stamp_start"), "sys_um"])
    frame = float(c.loc[("rpi_cm3", "stamp_frame"), "sys_um"])
    group = {cam: float(c.loc[(cam, "motion_group"), "sys_um"]) for cam in budget.CAMERAS}
    um = {cam: float(runs["budget"].df.loc[runs["budget"].df["budget_camera"] == cam,
                                           "um_per_px"].iloc[0]) for cam in budget.CAMERAS}
    sim_um = {cam: budget.SIM_LEVEL * um[cam] for cam in budget.CAMERAS}

    e6_cfg = runs["e6"].configs
    t_row = float(e6_cfg["t_row_estimate_s"])
    factors = [f for f in e6_cfg["t_row_factors"] if f > 0]
    d6 = e6.with_metrics(runs["e6"].df)
    summ = d6[np.isclose(d6["t_row_factor"], 1.0) & np.isclose(d6["speed"], E6_SPEED)]
    ell_roll = float(summ.loc[summ["method"] == "ellipse", "resid_px"].max())
    glob = d6[(d6["t_row_factor"] == 0) & np.isclose(d6["speed"], E6_SPEED)]
    ell_glob = float(glob.loc[glob["method"] == "ellipse", "resid_px"].max())

    def desync_mm(rows: int, speed: float) -> float:
        return speed * rows * t_row * 1e3

    lim = facts2["lim"]
    # E1: СКО центра тяжести при fill 0,7 почти не зависит от R — фраза
    # вместо панели (а) рис. 3
    d1 = runs["e1"].df
    c1 = d1[(d1["status"] == "ok") & (d1["method"] == "centroid")
            & np.isclose(d1["fill"], 0.7)]
    s1_min, s1_max = float(c1["std_u_px"].min()), float(c1["std_u_px"].max())
    r1_min, r1_max = float(c1["radius_px"].min()), float(c1["radius_px"].max())
    thr50, ell = lim.loc["centroid_thr50"], lim.loc["ellipse"]

    captions = "% подписи рисунков, scripts/paper.py\n\n" + "\n".join([
        figure_env("fig1_simulator", "simulator", (
            "Rendered image of a tilted moving "
            f"target ($\\theta$ = {info1['tilt_deg']:g}°, blur $L$ = {info1['L_px']:g} "
            f"px, R = {info1['R_px']:.1f} px, noise-free) and the reference centers; "
            "dotted and dashed outlines: the exact ellipse at the start, middle and end "
            "of the exposure. Right: the centers relative to the image centroid at two "
            f"scales; the eccentricity is {info1['eccentricity_px']:.3f} px, blur moves "
            "the exposure-averaged conic center from the mid-exposure one by "
            f"{info1['exposure_vs_mid_px']:.1e} px, and the "
            "centroid coincides with the exposure-averaged conic center to "
            f"{info1['centroid_vs_exposure_px']:.0e} px."
        ), wide=False),
        figure_env("fig2_pixel_phase", "pixel_phase", (
            "Noise-free bias of the center estimators versus the sub-pixel phase of the "
            "target center, R = 10 px: (a) without PSF, (b) Gaussian PSF, "
            "$\\sigma$ = 0.7 px."
        ), wide=False),
        figure_env("fig3_precision", "precision", (
            "Random error of the center estimators versus the signal $N_e$ in the "
            f"target, R = {e1.PANEL_B_RADIUS_PX:g} px. Dashed: centroid theory "
            "\\eqref{eq:centroid_std}, $\\sigma = R / (2\\sqrt{N_e})$ with the PSF, "
            "pixel and background terms. "
            "Dotted horizontal line: 0.05 px. Cram\\'er--Rao bound: Fisher "
            "information of the noise-free image with shot, read and quantization "
            "noise per pixel; flux and background known."
        ), wide=False),
        figure_env("fig4_eccentricity", "eccentricity", (
            "Relative error of the common approximation "
            "\\eqref{eq:ecc_approx} (Eq. (11) in \\cite{liebold2024}) against the "
            "exact eccentricity over the frame: "
            f"$\\theta$ = {e2.MAP_THETA:g}°, r = {e2.MAP_RADIUS * 1e3:g} mm, "
            f"Z = {e2.Z:g} m; (a) 16 mm lens, (b) Raspberry Pi Camera Module 3 in the "
            "2304$\\times$1296 mode (2$\\times$2 binning), as in Figs. 5 and 6. White "
            f"line: {e2.REL_ERR_LIMIT:.0%} error boundary. (c) Eccentricity field of a "
            "target facing the camera (normal towards the camera center), 16 mm lens. "
            "The consistent first-order expression $\\varepsilon_1$ errs by "
            "$k = (r/Z)^2 \\sin^2\\theta < 10^{-4}$ over the whole frame."
        ), wide=True),
        figure_env("fig5_motion", "motion", (
            "Motion during the exposure. (a), (b) Linear blur (E3): tilt 45°, motion "
            "along $u$, R = 23.2 px, fill 0.7; (a) bias along the motion relative to "
            "the exposure-averaged conic center, bars $\\pm$3 s.e., grey band: "
            f"simulator level $\\pm${e3.SIM_LEVEL:g} px; (b) standard deviation along "
            "the motion, dashed: centroid theory. Jumps of the 2D Gaussian fit at small "
            "$L$ are the intrinsic pixel-phase bias of the fit (Fig. 2). The "
            "half-level contour of a blurred tilted image has flat ends (rows whose "
            "chord is shorter than $L/2$ never reach half level), so the ellipse fit "
            "and the 50% threshold carry a pixel-phase bias across the motion that "
            "changes with $L$ (up to 0.01 px at $L$ = 16 px). (c), (d) Rolling shutter "
            "of the Raspberry Pi Camera Module 3 (E6): $t_{row}$ = "
            f"{t_row * 1e6:.1f} $\\mu$s is estimated from the datasheet frame rate of "
            "the 2304$\\times$1296 mode (56 frame/s); bands "
            f"$\\times${min(factors):g}...{max(factors):g} show the sensitivity. (c) "
            "Error of the frame-time stamp on the target versus the image row, lines "
            "labelled with the speed: speed $\\times$ $t_{row}$ $\\times$ row, points: "
            "centroid; (d) residual "
            "error with the row-time stamp, worst over rows and directions; open grey "
            "markers: the same method with a global shutter."
        ), wide=True),
        figure_env("fig6_budget", "budget", (
            "Error budget of the centroid for the two cameras, $\\mu$m on the target "
            f"({um['review_16mm']:.0f} and {um['rpi_cm3']:.0f} $\\mu$m/px). Motion "
            "during exposure: blur, rotation and rolling shutter switched off together "
            "(all effects minus all without them), with the correct time stamp. "
            "Diamond on the perspective row: common approximation "
            f"\\eqref{{eq:ecc_approx}}, {eq3:.1f} $\\mu$m. Scenario: Table~I. Hatched: errors "
            "avoided by the time stamp. Vertical lines: simulator level 0.002 px."
        ), wide=False),
    ])

    text = "\n".join([
        "# Фразы для текста статьи (scripts/paper.py)",
        "",
        "## Метод (раздел II)",
        "",
        "Приближение эксцентриситета — своим номером в статье (нужен amsmath):",
        "",
        "```latex",
        EQ_APPROX,
        "```",
        "",
        ("where $f$ is the focal length in pixels, $r$ the target radius, $Z$ the "
         "depth of the target center and $(n_x, n_y, n_z)$ the unit normal of the "
         "target in the camera frame. This common first-order approximation is "
         "Eq. (11) in \\cite{liebold2024}."),
        "",
        ("Rendering resolution: for minor semi-axes below about "
         f"{SS16_B_LIMIT_PX:g} px the sub-pixel grid of ss = 16 gives errors up to "
         f"{SS16_WORST_PX:.1e} px (long edges nearly parallel to the pixel grid are "
         "quantized coherently), so such images (E2d) are rendered with ss = "
         f"{runs['e2d'].configs['base'].ss} (centroid error at most "
         f"{float(lim.loc['centroid', 'err_narrowest_px']):.1e} px, checked with "
         f"ss = {runs['e2d'].configs['ss_check']})."),
        "",
        "## E1, случайная погрешность (вместо панели (а) рис. 3)",
        "",
        (f"At fill {0.7:g} the standard deviation of the centroid is "
         f"{s1_min * 1e3:.1f}–{s1_max * 1e3:.1f}×10$^{{-3}}$ px for R = "
         f"{r1_min:g}–{r1_max:g} px, nearly independent of the image size: at the same "
         "fill the signal grows as $N_e \\propto R^2$, and $\\sigma = R/(2\\sqrt{N_e})$ "
         "stays constant."),
        "",
        "## E2, граница по малой полуоси (E2d)",
        "",
        (f"With the target on the optical axis (ss = {runs['e2d'].configs['base'].ss}) "
        "and the sub-pixel phase of its center "
        "varied, the centroid stays at the simulator level (at most "
        f"{float(lim.loc['centroid', 'err_narrowest_px']):.1e} px) down to "
        f"$b/\\sigma_{{PSF}}$ = {facts2['x_min']:.1f} ($b$ = "
        f"{float(lim.loc['centroid', 'b_narrowest_px']):.1f} px), whereas the "
        "worst-phase error of the 50% threshold and the ellipse fit is above their "
        f"E0 level already at $b/\\sigma_{{PSF}}$ = {facts2['x_max']:.1f} "
        f"({thr50['err_widest_px']:.3f} and {ell['err_widest_px']:.3f} px), doubles below "
        f"$b/\\sigma_{{PSF}}$ ≈ {thr50['b_over_sigma_growth']:.1f} and "
        f"{ell['b_over_sigma_growth']:.1f} and reaches {thr50['err_narrowest_px']:.3f} "
        f"and {ell['err_narrowest_px']:.3f} px at $b/\\sigma_{{PSF}}$ = "
        f"{facts2['x_min']:.1f}."),
        "",
        "## E6",
        "",
        (f"The row time $t_{{row}}$ = {t_row * 1e6:.1f} µs is estimated from the "
        "datasheet frame rate of the 2304×1296 mode (56 frame/s); the bands "
        f"×{min(factors):g}...{max(factors):g} in Fig. 5c show the sensitivity to this "
        "estimate."),
        "",
        (f"A difference of 10 rows between two cameras gives {desync_mm(10, 0.5):.2f} mm "
        f"at 0.5 m/s, half a frame ({e6.DESYNC_ROWS[3]} rows) gives "
        f"{desync_mm(e6.DESYNC_ROWS[3], 0.5):.1f} mm."),
        "",
        ("With the row-time stamp the residual of the ellipse fit grows with speed "
        f"({ell_roll:.4f} vs {ell_glob:.4f} px with a global shutter at {E6_SPEED:g} "
        "m/s): the skew of the image shape degrades the contour methods."),
        "",
        "## Бюджет (рис. 6)",
        "",
        ("Motion during the exposure (blur, rotation, rolling shutter) with the "
        f"correct time stamp contributes {group['review_16mm']:.2f} and "
        f"{group['rpi_cm3']:.2f} µm, below the simulator level "
        f"({sim_um['review_16mm']:.2f} and {sim_um['rpi_cm3']:.2f} µm). The time stamp "
        f"itself removes {start:.0f} µm (start of exposure) and {frame / 1e3:.1f} mm "
        f"(frame time, rolling shutter). The approximation \\eqref{{eq:ecc_approx}} "
        f"gives {eq3:.1f} µm for the "
        f"eccentricity, the simulation {persp['review_16mm']:.1f} and "
        f"{persp['rpi_cm3']:.1f} µm (frame corner)."),
        "",
    ])
    paper.check(
        all(group[cam] <= sim_um[cam] for cam in group),
        "рис. 6: движение за экспозицию при верной привязке ниже уровня симулятора: "
        + ", ".join(f"{cam} {group[cam]:.2f} мкм (уровень {sim_um[cam]:.2f})"
                    for cam in group),
    )
    return captions, text


def plot(args: argparse.Namespace) -> int:
    manifest = json.loads(args.runs.read_text())
    missing = [k for k in RUN_KEYS if k not in manifest]
    if missing:
        raise SystemExit(f"в {args.runs} нет запусков: {', '.join(missing)}")
    runs = {k: load_run(manifest[k]) for k in RUN_KEYS}
    style.apply()
    paper = Paper(args.out)

    fig, info1 = fig1(runs["e3"])
    paper.figure(fig, "fig1_simulator")
    paper.figure(e0.figure(e0.prepare(runs["e0"])), "fig2_pixel_phase")
    paper.figure(
        e1.methods_figure(e1.with_theory(runs["e1"].df[runs["e1"].df["status"] == "ok"])),
        "fig3_precision",
    )
    paper.figure(e2.formula_error_figure(runs["e2"].df, facing=True), "fig4_eccentricity")
    paper.figure(fig5(runs["e3"], runs["e6"]), "fig5_motion")
    paper.figure(
        budget.figure(budget.contributions(runs["budget"]), runs["budget"]), "fig6_budget"
    )

    t1, note1 = table1(runs)
    t1.to_csv(args.out / "table1_parameters.csv", index=False)
    paper.text("table1_parameters.tex", table1_tex(t1, note1))
    t2, facts2 = table2(runs)
    t2.to_csv(args.out / "table2_methods.csv", index=False, float_format="%.4g")
    paper.text("table2_methods.tex", table2_tex(t2, facts2["x_min"]))

    captions, text = captions_and_text(runs, info1, t2, facts2, paper)
    paper.text("captions.tex", captions)
    paper.text("text.md", text)
    if paper.failed:
        print(f"не прошло проверок: {len(paper.failed)}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="рисунки и таблицы статьи")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plot", help="построить всё из папок запусков")
    p.add_argument("--runs", type=Path, default=PAPER_DIR / "runs.json")
    p.add_argument("--out", type=Path, default=PAPER_DIR)
    return plot(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
