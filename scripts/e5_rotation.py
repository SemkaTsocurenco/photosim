"""E5. Поворот марки за экспозицию.

Эффект, для которого в обзоре нет формулы (назван в его выводах). Марка
неподвижна, у центра кадра и в углу (0,9 кадра по u и v: там
эксцентриситет при наклоне вокруг y наибольший), камера review_16mm,
Z = 1 м. Марка вращается вокруг оси y камеры, лежащей в её плоскости и
проходящей через центр, — меняется наклон. Наклон в середине экспозиции
θ = 15°, 45°, 75° (к оптической оси), изменение за экспозицию
Δθ = ω τ = 0 ... 20°, τ = 1 мс, r = 5 и 10 мм. Эталон — conic_mid (эллипс в
середине экспозиции).

Предсказание (до рендера). Центр тяжести смазанного изображения — среднее
центров эллипса c(θ) с весом площади A(θ). До Δθ**2

    c_bar - c_mid = (c'' / 24 + A' c' / (12 A)) Δθ**2

(analysis.rotation_offset_coefficients, производные — разностями по
oracle.conic_center и conic_area): первый член — кривизна c(θ), второй —
изменение площади (у одного края экспозиции эллипс больше и весит больше).
Эталон — то же среднее численно, oracle.exposure_conic_center по 512
подшагам, на плотной сетке Δθ от 0,03°. Рендер (ss = 32, PSF, без шумов) —
проверка в 6 точках с r = 10 мм, где эффект 3e-3 ... 1,5e-2 пикс.

    python scripts/e5_rotation.py run [--quick] [--jobs 3]
    python scripts/e5_rotation.py plot runs/<...>_e5

Память: ss = 32, r = 10 мм — сетка лучей до ~0,3 ГБ на процесс.

Рисунок e5_rotation: |c_bar - c_mid| от Δθ (логарифмы), панели — центр и
угол кадра, линии — θ и r; точки — рендер; горизонталь — уровень
симулятора 1e-3 пикс.; полоса — Δθ звена манипулятора (1–3 рад/с при
τ = 1 мс, 0,06–0,17°).
Таблицы: e5_table.csv — коэффициенты, Δθ, при котором эффект достигает
1e-3 пикс., эффект при реальных угловых скоростях; e5_render_table.csv —
проверка рендером.
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

from photosim.analysis import log_crossing, point_at_pixel, rotation_offset_coefficients
from photosim.experiment import T_K, Config, Effects, Run, build, run_configs, save_run
from photosim.figures import style
from photosim.figures.cli import Output, main
from photosim.oracle import conic_center, exposure_conic_center
from photosim.presets import camera

NAME = "e5"
CAMERA = "review_16mm"
Z = 1.0
TAU = 1e-3
AXIS = (0.0, 1.0, 0.0)  # ось y камеры = ось наклона, в плоскости марки
THETAS = (15.0, 45.0, 75.0)
DTHETAS_DEG = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
DENSE_DEG = np.geomspace(0.03, 20.0, 40)
RADII = (0.005, 0.010)
PHASE = 0.3
POSITIONS = {"center": (0.5, 0.5), "corner": (0.9, 0.9)}  # доля (W - 1, H - 1)
NT_EXACT = 512
SIM_LEVEL = 1e-3  # уровень симулятора, пикс.
LINK_OMEGA = (1.0, 3.0)  # звено манипулятора, рад/с
FORMULA_RTOL = 0.01  # второй порядок против точного при Δθ <= FORMULA_MAX_DEG
FORMULA_MAX_DEG = 5.0
# проверка рендером: (положение, r, θ, Δθ)
RENDER_POINTS = (
    ("center", 0.010, 75.0, 10.0),
    ("center", 0.010, 75.0, 20.0),
    ("center", 0.010, 45.0, 20.0),
    ("center", 0.010, 15.0, 20.0),
    ("corner", 0.010, 75.0, 20.0),
    ("corner", 0.010, 45.0, 20.0),
)
RENDER_SS = 32
QUICK_SS = 16  # --quick: уровень симулятора до 1,6e-3 пикс., порог проверки вдвое выше


def pixel(position: str) -> tuple[float, float]:
    cam = camera(CAMERA)
    fu, fv = POSITIONS[position]
    if position == "center":
        return cam.cx + PHASE, cam.cy + PHASE
    return round(fu * (cam.width - 1)) + PHASE, round(fv * (cam.height - 1)) + PHASE


def config(position: str, radius: float, theta: float, dtheta_deg: float) -> Config:
    cam = camera(CAMERA)
    return Config(
        camera=CAMERA,
        radius=radius,
        position=point_at_pixel(cam, *pixel(position), Z),
        tilt_deg=theta,
        tilt_axis="y",
        rotation={"axis": AXIS, "omega": math.radians(dtheta_deg) / TAU},
        tau=TAU,
        effects=Effects(
            rolling_shutter=False, shot_noise=False, read_noise=False,
            quantization=False,
        ),
        n_mc=2,
        reference="conic_mid",
        ss=RENDER_SS,
        nt=64,
    )


def analytic(quick: bool) -> pd.DataFrame:
    """Точное смещение (exposure_conic_center) и второй порядок на сетке Δθ."""
    dense = DENSE_DEG[::4] if quick else DENSE_DEG
    grid = np.unique(np.concatenate([DTHETAS_DEG, dense]))
    rows = []
    for position, r, theta in itertools.product(POSITIONS, RADII, THETAS):
        s = build(config(position, r, theta, 0.0))
        X, R_mid = s.motion.position(s.t_ref), s.motion.orientation(s.t_ref)
        mid = conic_center(s.camera, s.pose, X, R_mid, r)
        k_curv, k_area = rotation_offset_coefficients(
            s.camera, s.pose, X, R_mid, AXIS, r
        )
        for dtheta in grid:
            sd = build(config(position, r, theta, dtheta))
            exact = (
                exposure_conic_center(
                    sd.camera, sd.pose, sd.motion, r, T_K, TAU, NT_EXACT
                )
                - mid
            )
            d2 = math.radians(dtheta) ** 2
            formula = (k_curv + k_area) * d2
            rows.append(
                {
                    "position": position,
                    "r_m": r,
                    "theta_deg": theta,
                    "dtheta_deg": dtheta,
                    "on_grid": bool(np.isclose(dtheta, DTHETAS_DEG).any()),
                    "exact_u": exact[0],
                    "exact_v": exact[1],
                    "formula_u": formula[0],
                    "formula_v": formula[1],
                    "k_curv_u": k_curv[0],
                    "k_curv_v": k_curv[1],
                    "k_area_u": k_area[0],
                    "k_area_v": k_area[1],
                    "u_px": pixel(position)[0],
                    "v_px": pixel(position)[1],
                }
            )
    df = pd.DataFrame(rows)
    df["exact_abs"] = np.hypot(df["exact_u"], df["exact_v"])
    df["formula_abs"] = np.hypot(df["formula_u"], df["formula_v"])
    return df


def run(args: argparse.Namespace) -> Path:
    df = analytic(args.quick)
    points = RENDER_POINTS[:2] if args.quick else RENDER_POINTS
    configs = [config(*p) for p in points]
    if args.quick:
        configs = [replace(c, ss=QUICK_SS) for c in configs]
    labels = [
        {"position": p, "r_m": r, "theta_deg": t, "dtheta_deg": d}
        for p, r, t, d in points
    ]
    rendered = run_configs(configs, labels=labels, n_jobs=args.jobs)
    configs_json = {
        "experiment": NAME,
        "quick": args.quick,
        "base": configs[0],
        "render_points": [list(p) for p in points],
        "thetas_deg": list(THETAS),
        "radii_m": list(RADII),
        "positions": POSITIONS,
        "nt_exact": NT_EXACT,
    }
    name = NAME + ("_quick" if args.quick else "")
    return save_run(df, configs_json, name, args.root, tables={"render": rendered})


# Рисунок и таблицы


def render_table(run: Run) -> pd.DataFrame:
    """Рендер против точного смещения в 6 точках, все методы."""
    d = run.tables["render"]
    d = d[d["status"] == "ok"].copy()
    d["exact_u"] = d["ref.exposure_conic.u"] - d["ref.conic_mid.u"]
    d["exact_v"] = d["ref.exposure_conic.v"] - d["ref.conic_mid.v"]
    d["diff_u"] = d["err0_u_px"] - d["exact_u"]
    d["diff_v"] = d["err0_v_px"] - d["exact_v"]
    d["measured_abs"] = np.hypot(d["err0_u_px"], d["err0_v_px"])
    d["exact_abs"] = np.hypot(d["exact_u"], d["exact_v"])
    d["diff_abs"] = np.hypot(d["diff_u"], d["diff_v"])
    return d


def plot(run: Run, out: Output, args: argparse.Namespace) -> None:
    df = run.df
    rend = render_table(run)
    thetas = sorted(df["theta_deg"].unique())
    radii = sorted(df["r_m"].unique())
    shades = dict(zip(thetas, ("0.0", "0.4", "0.65"), strict=False))
    dashes = dict(zip(radii, ("-", "--"), strict=False))
    lo, hi = (math.degrees(w * TAU) for w in LINK_OMEGA)

    positions = [p for p in POSITIONS if p in set(df["position"])]
    fig, axes = plt.subplots(
        1, len(positions), figsize=(style.DOUBLE, 3.0), sharey=True, squeeze=False
    )
    for ax, position in zip(axes[0], positions, strict=True):
        ax.axvspan(lo, hi, color="0.9", linewidth=0, zorder=0)
        ax.axhline(SIM_LEVEL, color="0.3", linewidth=0.6, linestyle=":")
        for theta, r in itertools.product(thetas, radii):
            d = df[
                (df["position"] == position)
                & np.isclose(df["theta_deg"], theta)
                & np.isclose(df["r_m"], r)
                & (df["dtheta_deg"] > 0)
            ].sort_values("dtheta_deg")
            color = shades[theta]
            ax.plot(
                d["dtheta_deg"], d["exact_abs"], color=color, linestyle=dashes[r],
                marker="", label=f"$\\theta$ = {theta:g}°, r = {r * 1e3:g} mm",
            )
            g = d[d["on_grid"]]
            ax.plot(g["dtheta_deg"], g["formula_abs"], color=color, linestyle="",
                    marker="x", markersize=3)
        c = rend[(rend["position"] == position) & (rend["method"] == "centroid")]
        ax.plot(c["dtheta_deg"], c["measured_abs"], linestyle="", marker="o",
                markersize=5, markerfacecolor="none", color="0.0")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Tilt change during the exposure, $\\Delta\\theta$ (deg)")
        u, v = df.loc[df["position"] == position, ["u_px", "v_px"]].iloc[0]
        where = "Frame center" if position == "center" else "Frame corner"
        ax.set_title(f"({'ab'[positions.index(position)]}) {where}, "
                     f"$u$ = {u:.0f}, $v$ = {v:.0f} px")
        ax.annotate("simulator level", (0.3, SIM_LEVEL), xycoords=("axes fraction", "data"),
                    xytext=(0, 2), textcoords="offset points", ha="left", va="bottom")
        ax.annotate("1–3 rad/s\nlink", (math.sqrt(lo * hi), 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, -3),
                    textcoords="offset points", ha="center", va="top", fontsize=7)
    axes[0][0].set_ylabel("Offset of the centroid from the mid-exposure\n"
                          "ellipse center, $|\\bar c - c_{mid}|$ (px)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color="0.3", marker="x", linestyle="", markersize=3),
        plt.Line2D([], [], color="0.0", marker="o", markerfacecolor="none",
                   linestyle="", markersize=5),
    ]
    labels += ["2nd-order formula", "Rendered image, centroid"]
    fig.legend(handles, labels, loc="outside lower center", ncols=4)
    out.figure(fig, "e5_rotation")

    table = summary_table(df, out)
    keep = [
        "position", "r_m", "theta_deg", "dtheta_deg", "method", "err0_u_px",
        "err0_v_px", "exact_u", "exact_v", "diff_u", "diff_v", "measured_abs",
        "exact_abs", "diff_abs",
    ]
    rend_out = rend[keep].copy()
    rend_out["method"] = pd.Categorical(rend_out["method"], style.ordered(rend_out["method"]))
    out.table(rend_out.sort_values(["position", "theta_deg", "dtheta_deg", "method"]),
              "e5_render")
    check(df, rend, table, out)


def summary_table(df: pd.DataFrame, out: Output) -> pd.DataFrame:
    rows = []
    for (position, r, theta), d in df.groupby(["position", "r_m", "theta_deg"]):
        d = d[d["dtheta_deg"] > 0].sort_values("dtheta_deg")
        first = d.iloc[0]
        k = np.array([first["k_curv_u"] + first["k_area_u"],
                      first["k_curv_v"] + first["k_area_v"]])
        k_abs = float(np.hypot(*k))
        crossing, extrapolated = log_crossing(d["dtheta_deg"], d["exact_abs"], SIM_LEVEL)
        row = {
            "position": position,
            "r_mm": r * 1e3,
            "theta_deg": theta,
            "k_px_per_rad2": k_abs,
            "k_curv_px_per_rad2": float(np.hypot(first["k_curv_u"], first["k_curv_v"])),
            "k_area_px_per_rad2": float(np.hypot(first["k_area_u"], first["k_area_v"])),
            "dtheta_at_sim_level_deg": crossing,
            "dtheta_at_sim_level_formula_deg": math.degrees(math.sqrt(SIM_LEVEL / k_abs)),
            "extrapolated": extrapolated,
        }
        for w in LINK_OMEGA:
            row[f"offset_at_{w:g}rad_s_px"] = k_abs * (w * TAU) ** 2
        at20 = d[np.isclose(d["dtheta_deg"], 20.0)]
        if not at20.empty:
            row["offset_at_20deg_px"] = float(at20["exact_abs"].iloc[0])
            row["formula_over_exact_20deg"] = float(
                at20["formula_abs"].iloc[0] / at20["exact_abs"].iloc[0]
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    out.table(table, NAME)
    return table


def check(df: pd.DataFrame, rend: pd.DataFrame, table: pd.DataFrame, out: Output) -> None:
    small = df[(df["dtheta_deg"] > 0) & (df["dtheta_deg"] <= FORMULA_MAX_DEG)]
    err = np.hypot(small["formula_u"] - small["exact_u"], small["formula_v"] - small["exact_v"])
    rel = err / small["exact_abs"]
    out.check(
        bool((rel <= FORMULA_RTOL).all()),
        f"второй порядок против exposure_conic_center при Δθ <= {FORMULA_MAX_DEG:g}°: "
        f"до {rel.max():.1e} (порог {FORMULA_RTOL:g})",
    )
    c = rend[rend["method"] == "centroid"]
    level = SIM_LEVEL if c["ss"].min() >= RENDER_SS else 2 * SIM_LEVEL
    out.check(
        bool((c["diff_abs"] <= level).all()),
        f"рендер: центр тяжести против точного смещения в {len(c)} точках: "
        f"до {c['diff_abs'].max():.1e} пикс. при эффекте "
        f"{c['exact_abs'].min():.1e} ... {c['exact_abs'].max():.1e} (порог {level:g})",
    )
    for m, d in rend[rend["method"] != "centroid"].groupby("method"):
        print(f"  {m}: |рендер - точное| до {d['diff_abs'].max():.1e} пикс.")
    worst = table.loc[table["k_px_per_rad2"].idxmax()]
    hi = max(LINK_OMEGA)
    print(
        f"  Δθ, при котором эффект достигает {SIM_LEVEL:g} пикс.: от "
        f"{table['dtheta_at_sim_level_deg'].min():.1f}° до "
        f"{table['dtheta_at_sim_level_deg'].max():.1f}°"
    )
    at_link = float(table[f"offset_at_{hi:g}rad_s_px"].max())
    verdict = "эффектом можно пренебречь" if at_link < SIM_LEVEL / 100 else "учитывать"
    print(
        f"  при {hi:g} рад/с и {TAU * 1e3:g} мс (Δθ = {math.degrees(hi * TAU):.2f}°): "
        f"не больше {at_link:.1e} пикс. ({worst['position']}, r = "
        f"{worst['r_mm']:g} мм, θ = {worst['theta_deg']:g}°) — {verdict}"
    )


if __name__ == "__main__":
    sys.exit(main("E5: поворот марки за экспозицию", run, plot))
