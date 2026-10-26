"""E1. Случайная погрешность: сигнал и размер марки.

Неподвижная марка параллельна сенсору (review_16mm, Z = 1 м), фаза центра
0,3 пикс. по обеим осям, все шумы и PSF (в рендере, до пикселя) включены,
эталон — proj_mid. Перебор: заполнение 0,001 ... 0,7 (10 значений; малые —
чтобы пересечение с 0,05 пикс. было измерено, а не экстраполировано) ×
радиус образа 2, 3, 5, 8, 10, 12, 18, 25 пикс. и марка обзора r = 5 мм
(≈ 23,2 пикс.) — 90 точек. n_mc = 500; gauss2d отдельным прогоном с
n_mc = 200. В каждой точке — граница Крамера — Рао (experiment.crlb).

    python scripts/e1_precision.py run [--quick] [--jobs 6]
    python scripts/e1_precision.py plot runs/<...>_e1

Теория СКО центра тяжести (photosim.analysis.centroid_sigma):
sigma**2 = R**2 / (4 N_e) + var_bg * S2 / N_e**2, var_bg = read_noise**2 +
gain**2 / 12, S2 = sum((u - u_c)**2) по ROI. На рисунке и в проверке — с
двумя поправками. (1) К R**2 / 4 добавлено sigma_psf**2 + 1/12 (PSF и
отнесение света к центру пикселя): без неё при R = 2 пикс. теория занижает
СКО на 17–19 %. (2) Центр тяжести вычитает медиану рамки и обрезает
отрицательные остатки, max(I - bg, 0): у фона остаётся дисперсия
(1/2 - 1/(2 pi)) var_bg ≈ 0,34 var_bg. С полной var_bg теория завышает СКО
при малом сигнале в 1,2–1,7 раза.

Обрезка же даёт фону среднее sqrt(var_bg / (2 pi)) на пиксель и тянет
центр к середине ROI: смещение ≈ sqrt(var_bg / (2 pi)) * w * h *
(середина ROI - u_c) / N_e. Это систематическая ошибка метода из ТЗ; при
малом сигнале она сравнима со СКО, проверка смещения её показывает.

Граница Крамера — Рао (experiment.crlb) — по кадру без шума, в каждой
точке. mu_i — среднее число электронов в пикселе i (рендер с PSF до
пикселя, сенсор без шумов), var_i = mu_i + read_noise**2 + gain**2 / 12
(фотоны, считывание, квантование). Производные d mu_i / d(u, v) —
центральные разности: марка сдвигается вдоль осей x и y камеры на
±0,25 пикс. в пересчёте на объект, кадр строится на том же ROI, сдвиг
переводится в (u, v) по смещению проекции центра. Информация Фишера
F = sum_i grad mu_i grad mu_i^T / var_i, граница — sqrt((F^-1)_uu). Яркость
и фон считаются известными, поэтому это граница для одного положения;
обрезка фона и пороги методов в неё не входят. Для смещённых методов
(центр тяжести при малом сигнале) граница строго не применима.

Рисунок e1_precision (одна колонка, панели друг под другом): (а) СКО
центра тяжести от числа электронов при R = 2, 3, 5, 10 и 23,2 пикс.,
теория штрихом, 0,05 пикс.; (б) все методы при R = 10 пикс. и граница
Крамера — Рао (её определение — в подписи рисунка). Рис. 3 статьи — только
(б) (methods_figure); что СКО почти не зависит от R, — фразой в тексте.
Таблица e1_table.csv: число электронов, при котором СКО = 0,05 пикс.
(измеренное, интерполяция в логарифмах; за краем сетки — экстраполяция, и
по теории), СКО при fill = 0,7 и граница Крамера — Рао для каждого радиуса.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # запуск без установки

import argparse
import math
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import binom

from photosim.analysis import (
    CLIP_VARIANCE,
    background_variance,
    centroid_sigma,
    centroid_signal_for_sigma,
    log_crossing,
    point_at_pixel,
    roi_s2,
)
from photosim.experiment import Config, Effects, Run, save_run, sweep
from photosim.figures import style
from photosim.figures.cli import Output, main
from photosim.presets import camera

NAME = "e1"
CAMERA = "review_16mm"
Z = 1.0
U0, V0, PHASE = 1224, 1024, 0.3
FILLS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7)
RADII_PX = (2, 3, 5, 8, 10, 12, 18, 25)
REVIEW_RADIUS = 0.005  # марка обзора, м
MAIN_METHODS = ("centroid", "centroid_thr50", "centroid_sq", "ellipse")
N_MC, N_MC_GAUSS = 500, 200
LITERATURE_STD = 0.05  # пикс.
PANEL_B_RADIUS_PX = 10
# (а): пять радиусов — больше узкая панель не вмещает; 23,2 — марка 5 мм на 1 м
PANEL_A_RADII_PX = (2, 3, 5, 10, 23.2)
X_LIM = (1e2, 2e7)  # общие для (а) и (б), электроны
THEORY_RTOL, THEORY_MIN_FILL = 0.15, 0.2
# (а): у каждого радиуса свой маркер — линии различимы в оттенках серого
RADIUS_MARKERS = ("o", "s", "^", "D", "v", "<", ">", "p")
OPEN_MARKERS = ("s", "D", "<", "p")


def run(args: argparse.Namespace) -> Path:
    cam = camera(CAMERA)
    if args.quick:
        fills, radii_px, n_mc, n_mc_gauss = (0.02, 0.2, 0.7), (3, 10), 50, 50
    else:
        fills, radii_px, n_mc, n_mc_gauss = FILLS, RADII_PX, N_MC, N_MC_GAUSS
    base = Config(
        camera=CAMERA,
        position=point_at_pixel(cam, U0 + PHASE, V0 + PHASE, Z),
        effects=Effects(
            perspective=False,  # плоскость марки || сенсору
            exposure=False,  # марка неподвижна: nt = 1 даёт тот же кадр
            motion=False,
            rotation=False,
            rolling_shutter=False,
        ),
        reference="proj_mid",
        n_mc=n_mc,
        methods=MAIN_METHODS,
        crlb=True,
    )
    base_gauss = replace(base, n_mc=n_mc_gauss, methods=("gauss2d",), crlb=False)
    radii = sorted([r * Z / cam.fx for r in radii_px] + [REVIEW_RADIUS])
    grid = {"fill": list(fills), "radius": radii}
    df = pd.concat(
        [
            sweep(base, grid, n_jobs=args.jobs),
            sweep(base_gauss, grid, n_jobs=args.jobs),
        ],
        ignore_index=True,
    )
    configs = {
        "experiment": NAME,
        "quick": args.quick,
        "base": base,
        "base_gauss2d": base_gauss,
        "grid": grid,
    }
    return save_run(df, configs, NAME + ("_quick" if args.quick else ""), args.root)


def with_theory(df: pd.DataFrame) -> pd.DataFrame:
    """Теория СКО и смещения центра тяжести (см. описание модуля).

    ``theory_spec`` — формула ТЗ; ``theory_full`` — с PSF и пикселем;
    ``theory`` — ещё и с обрезанным фоном (на рисунке и в проверке);
    ``bias_clip`` — смещение от среднего обрезанного фона.
    """
    df = df.copy()
    df["R_px"] = df["radius_px"].round(2)
    df["var_bg"] = background_variance(
        df["sensor.read_noise_e"], df["sensor.gain_e_per_dn"]
    )
    df["s2"] = roi_s2(df["roi.u0"], df["roi.w"], df["roi.h"], df["ref.proj_mid.u"])
    df["extra"] = df["sensor.psf_sigma_px"] ** 2 + 1 / 12
    R, n_e = df["R_px"], df["signal_e"]
    df["theory_spec"] = centroid_sigma(R, n_e, df["var_bg"], df["s2"])
    df["theory_full"] = centroid_sigma(R, n_e, df["var_bg"], df["s2"], df["extra"])
    df["theory"] = centroid_sigma(
        R, n_e, CLIP_VARIANCE * df["var_bg"], df["s2"], df["extra"]
    )
    mid = df["roi.u0"] + (df["roi.w"] - 1) / 2
    df["bias_clip"] = (
        np.sqrt(df["var_bg"] / (2 * math.pi))
        * df["roi.w"]
        * df["roi.h"]
        * (mid - df["ref.proj_mid.u"])
        / n_e
    )
    return df


def figure(df: pd.DataFrame) -> Figure:
    """Рабочий рисунок E1 (одна колонка): (а) СКО центра тяжести при
    радиусах PANEL_A_RADII_PX, (б) все методы (``plot_methods``) — друг под
    другом, пределы по x общие. В статье (рис. 3) — только (б),
    ``methods_figure``; независимость от R — фразой в тексте.
    """
    radii = np.sort(df["R_px"].unique())
    cen = df[df["method"] == "centroid"]
    small = style.SMALL_FONT_SIZE

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(style.COLUMN, 4.6))
    radii_a = [r for r in radii if np.isclose(r, PANEL_A_RADII_PX, atol=0.05).any()]
    shades = plt.get_cmap("cividis")(np.linspace(0.0, 0.85, len(radii_a)))
    for r, color, marker in zip(radii_a, shades, RADIUS_MARKERS, strict=False):
        d = cen[cen["R_px"] == r].sort_values("signal_e")
        ax_a.plot(
            d["signal_e"], d["std_u_px"], marker=marker, color=color, label=f"{r:.3g}",
            markerfacecolor="white" if marker in OPEN_MARKERS else color,
        )
        ax_a.plot(d["signal_e"], d["theory"], linestyle="--", color=color, marker="")
    ax_a.set_title("(a) Centroid", loc="left")
    ax_a.legend(title="R (px)", ncols=3, loc="upper right", fontsize=small,
                title_fontsize=small, columnspacing=1.0)
    log_axes(ax_a)
    # над линиями — место для легенды радиусов (после логарифмической оси)
    ax_a.set_ylim(top=max(ax_a.get_ylim()[1], 6 * LITERATURE_STD))
    r_b = plot_methods(ax_b, df)
    ax_b.set_title(f"(b) All methods, R = {r_b:g} px", loc="left")
    for ax in (ax_a, ax_b):
        ax.set_xlim(*X_LIM)
    return fig


def methods_figure(df: pd.DataFrame) -> Figure:
    """Рис. 3 статьи (одна колонка, одна панель): все методы при R =
    PANEL_B_RADIUS_PX и граница Крамера — Рао. R и определение границы — в
    подписи рисунка."""
    fig, ax = plt.subplots(figsize=(style.COLUMN, 2.9))
    plot_methods(ax, df)
    return fig


def plot_methods(ax, df: pd.DataFrame) -> float:
    """СКО всех методов при R = PANEL_B_RADIUS_PX, теория центра тяжести и
    граница Крамера — Рао; легенда — под осями. Возвращает радиус панели."""
    radii = np.sort(df["R_px"].unique())
    r_b = radii[np.argmin(np.abs(radii - PANEL_B_RADIUS_PX))]
    sub = df[df["R_px"] == r_b]
    for m in style.ordered(df["method"].unique()):
        d = sub[sub["method"] == m].sort_values("signal_e")
        ax.plot(d["signal_e"], d["std_u_px"], **style.method_style(m))
    d = sub[sub["method"] == "centroid"].sort_values("signal_e")
    ax.plot(d["signal_e"], d["theory"], color="0.0", linestyle="--", label="Theory")
    if "crlb_u_px" in d:
        ax.plot(
            d["signal_e"], d["crlb_u_px"], color="0.0", linestyle=":",
            linewidth=1.2, label="Cramér–Rao bound",
        )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=3,
              fontsize=style.SMALL_FONT_SIZE, handlelength=2.2, columnspacing=1.0)
    log_axes(ax)
    return float(r_b)


def log_axes(ax) -> None:
    """Логарифмические оси, линия 0,05 пикс. с подписью у правого края, подписи осей."""
    small = style.SMALL_FONT_SIZE
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(LITERATURE_STD, color="0.4", linewidth=0.6, linestyle=":")
    ax.annotate(
        f"{LITERATURE_STD:g} px",
        (1.0, LITERATURE_STD),
        xycoords=("axes fraction", "data"),
        xytext=(-3, 2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=small,
    )
    ax.set_xlabel("Signal in the target, $N_e$ (electrons)")
    ax.set_ylabel("Std in $u$ (px)")
    ax.tick_params(labelsize=small)


def plot(run: Run, out: Output, args: argparse.Namespace) -> None:
    df = with_theory(run.df[run.df["status"] == "ok"])
    methods = style.ordered(df["method"].unique())
    radii = np.sort(df["R_px"].unique())
    cen = df[df["method"] == "centroid"]
    out.figure(figure(df), "e1_precision")

    rows = []
    for r in radii:
        d = cen[cen["R_px"] == r].sort_values("signal_e")
        n_e, extrapolated = log_crossing(d["signal_e"], d["std_u_px"], LITERATURE_STD)
        first = d.iloc[0]
        row = {
            "radius_px": r,
            "n_e_at_0.05px": n_e,
            "extrapolated": extrapolated,
            "n_e_at_0.05px_theory": float(
                centroid_signal_for_sigma(
                    LITERATURE_STD,
                    r,
                    CLIP_VARIANCE * first["var_bg"],
                    first["s2"],
                    first["extra"],
                )
            ),
        }
        at_07 = df[(df["R_px"] == r) & np.isclose(df["fill"], 0.7)]
        for m in methods:
            row[f"std_fill0.7_{m}"] = at_07.loc[
                at_07["method"] == m, "std_u_px"
            ].squeeze()
        cen_07 = at_07[at_07["method"] == "centroid"]
        row["theory_fill0.7"] = cen_07["theory"].squeeze()
        if "crlb_u_px" in cen_07:
            row["crlb_fill0.7"] = cen_07["crlb_u_px"].squeeze()
        rows.append(row)
    out.table(pd.DataFrame(rows), NAME)

    check(df, cen, out)


def check(df: pd.DataFrame, cen: pd.DataFrame, out: Output) -> None:
    # Шум не должен добавлять смещения: |bias - err0| <= 3 se. Само err0
    # центра тяжести — дискретизация рендера (до 7e-4 пикс. при ss = 16,
    # см. E0) — при fill 0,7 и n_mc 500 больше 3 se, поэтому |bias| <= 3 se
    # здесь не проверка. При многих точках пара выходов за 3 se случайна:
    # порог — квантиль 99,9 % биномиального распределения.
    n_out = int(
        ((cen["bias_u_px"] - cen["err0_u_px"]).abs() > 3 * cen["se_bias_u_px"]).sum()
        + ((cen["bias_v_px"] - cen["err0_v_px"]).abs() > 3 * cen["se_bias_v_px"]).sum()
    )
    n_tests = 2 * len(cen)
    allowed = int(binom.ppf(0.999, n_tests, 0.0027))
    noise_bias = cen["bias_u_px"] - cen["err0_u_px"]
    worst = cen.loc[(noise_bias / cen["se_bias_u_px"]).abs().idxmax()]
    out.check(
        n_out <= allowed,
        f"centroid: |bias - err0| > 3 se в {n_out} из {n_tests} проверок "
        f"(случайно допустимо до {allowed}); хуже всего R = {worst['R_px']:g} пикс., "
        f"fill = {worst['fill']:g}: {worst['bias_u_px'] - worst['err0_u_px']:+.1e} "
        f"пикс. при СКО {worst['std_u_px']:.1e}",
    )
    # модель: среднее обрезанного фона тянет центр к середине ROI
    big = cen[(noise_bias / cen["se_bias_u_px"]).abs() > 3]
    if not big.empty:
        ratio = (big["bias_u_px"] - big["err0_u_px"]) / big["bias_clip"]
        print(
            f"  смещение от обрезки фона max(I - bg, 0): измеренное / модель "
            f"= {ratio.median():.2f} (медиана), от {ratio.min():.2f} до "
            f"{ratio.max():.2f} по {len(big)} точкам"
        )
    for m in style.ordered(df["method"].unique()):
        d = df[df["method"] == m]
        se_u, se_v = 3 * d["se_bias_u_px"], 3 * d["se_bias_v_px"]
        raw = ((d["bias_u_px"].abs() > se_u) | (d["bias_v_px"].abs() > se_v)).sum()
        own = (
            ((d["bias_u_px"] - d["err0_u_px"]).abs() > se_u)
            | ((d["bias_v_px"] - d["err0_v_px"]).abs() > se_v)
        ).sum()
        print(
            f"  {m}: |bias| > 3 se в {raw} из {len(d)} точек, "
            f"|bias - err0| > 3 se в {own} (err0 — собственное смещение, E0)"
        )

    strong = cen[cen["fill"] >= THEORY_MIN_FILL]
    dev = (strong["std_u_px"] / strong["theory"] - 1).abs()
    worst = strong.loc[dev.idxmax()]
    # оценка СКО по n реализациям сама шумит на 1 / sqrt(2 (n - 1)): при
    # n_mc = 500 порог — 15 %, в быстром прогоне (n_mc = 50) — 3 sigma = 30 %
    rtol = max(THEORY_RTOL, 3 / math.sqrt(2 * (strong["n_mc"].min() - 1)))
    out.check(
        bool((dev <= rtol).all()),
        f"centroid, fill >= {THEORY_MIN_FILL}: СКО против теории до {dev.max():.1%} "
        f"при R = {worst['R_px']:g} пикс. (порог {rtol:.0%})",
    )
    dev = (strong["std_u_px"] / strong["theory_spec"] - 1).abs()
    worst = strong.loc[dev.idxmax()]
    print(
        f"  формула ТЗ без PSF и пикселя: до {dev.max():.1%} "
        f"при R = {worst['R_px']:g} пикс."
    )
    weak = cen[(cen["fill"] < THEORY_MIN_FILL) & (cen["fill"] >= 0.005)]
    for col, label in (("theory", "с обрезанным фоном"), ("theory_full", "полный фон")):
        ratio = weak["std_u_px"] / weak[col]
        print(
            f"  fill 0,005–0,1, СКО / теория ({label}): медиана {ratio.median():.2f}, "
            f"от {ratio.min():.2f} до {ratio.max():.2f}"
        )


if __name__ == "__main__":
    sys.exit(main("E1: случайная погрешность", run, plot))
