"""E0. Собственное смещение методов без движения и шума (нулевой уровень).

Неподвижная марка параллельна сенсору (камера review_16mm, Z = 1 м), шумов
нет, эталон — проекция центра (proj_mid). Перебор: дробная фаза центра по u
0 ... 1 пикс. с шагом 0,05, радиус образа 5, 10, 23 пикс., PSF выключена и
включена: 21 × 3 × 2 = 126 точек, ss = 64 (уровень 1e-4 пикс.). PSF
применяет рендер до усреднения по пикселю, как оптика. Ошибка без шума
err0_u_px — собственное смещение метода; с этим уровнем сравниваются все
следующие эффекты.

    python scripts/e0_pixel_phase.py run [--quick] [--jobs 4]
    python scripts/e0_pixel_phase.py plot runs/<...>_e0

Память: при ss = 64 сетка лучей для R = 23 — около 0,3 ГБ на процесс.

Рисунок e0_pixel_phase (рис. 2 статьи, одна колонка): две панели друг под
другом (без PSF и с PSF), радиус 10 пикс.
Таблица e0_table.csv: размах err0_u_px по фазе для метода, радиуса и PSF.
Проверка: у centroid с PSF размах не больше 3e-3 пикс. Без PSF размах
частично физический — пиксельная фаза идеально резкого края (при R = 5
пикс. ~4e-3 и с ростом ss не уходит), он печатается для сведения.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # запуск без установки

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from photosim.analysis import point_at_pixel
from photosim.experiment import Config, Effects, Run, save_run, sweep
from photosim.figures import style
from photosim.figures.cli import Output, main
from photosim.presets import camera

NAME = "e0"
CAMERA = "review_16mm"
Z = 1.0
U0, V0 = 1224, 1024  # целые пиксели: cx = 1223,5 полуцелое
PLOT_RADIUS_PX = 10
CENTROID_RANGE_MAX = 3e-3  # уровень симулятора, пикс.
SS = 64


def run(args: argparse.Namespace) -> Path:
    cam = camera(CAMERA)
    if args.quick:
        phases, radii_px, ss = np.linspace(0.0, 1.0, 5), (10,), 16
    else:
        phases, radii_px, ss = np.round(np.arange(21) * 0.05, 2), (5, 10, 23), SS
    base = Config(
        camera=CAMERA,
        effects=Effects(
            perspective=False,  # плоскость марки || сенсору
            exposure=False,  # марка неподвижна: nt = 1 даёт тот же кадр
            motion=False,
            rotation=False,
            rolling_shutter=False,
            shot_noise=False,
            read_noise=False,
            quantization=False,
        ),
        n_mc=2,
        reference="proj_mid",
        ss=ss,
    )
    grid = {
        "position": [point_at_pixel(cam, U0 + p, V0, Z) for p in phases],
        "radius": [r * Z / cam.fx for r in radii_px],
        "effects.psf": [False, True],
    }
    df = sweep(base, grid, n_jobs=args.jobs)
    configs = {
        "experiment": NAME,
        "quick": args.quick,
        "u0": U0,
        "base": base,
        "grid": grid,
    }
    return save_run(df, configs, NAME + ("_quick" if args.quick else ""), args.root)


def prepare(run: Run) -> pd.DataFrame:
    """Точки запуска с колонками ``phase`` (дробная фаза по u) и ``R_px``."""
    df = run.df[run.df["status"] == "ok"].copy()
    df["phase"] = (df["ref.proj_mid.u"] - run.configs["u0"]).round(6)
    df["R_px"] = df["radius_px"].round(2)
    return df


def figure(df: pd.DataFrame) -> Figure:
    """Рис. 2 статьи (одна колонка): смещение без шума от фазы при R =
    PLOT_RADIUS_PX, (а) без PSF и (б) с ней — друг под другом.

    Оси y у панелей свои: с PSF смещения в разы меньше. R и «без шума» — в
    подписи рисунка, на оси — коротко.
    """
    methods = style.ordered(df["method"].unique())
    radii = np.sort(df["R_px"].unique())
    r_plot = radii[np.argmin(np.abs(radii - PLOT_RADIUS_PX))]
    fig, axes = plt.subplots(2, 1, figsize=(style.COLUMN, 3.6), sharex=True)
    for ax, psf, letter in zip(axes, (False, True), "ab", strict=True):
        sub = df[(df["effects.psf"] == psf) & (df["R_px"] == r_plot)]
        for m in methods:
            d = sub[sub["method"] == m].sort_values("phase")
            # в узкой панели маркеры на каждой из 21 фазы сливаются
            ax.plot(d["phase"], d["err0_u_px"], markevery=2 if len(d) > 20 else 1,
                    **style.method_style(m))
        if psf:
            sigma = sub["sensor.psf_sigma_px"].iloc[0]
            ax.set_title(f"({letter}) Gaussian PSF, $\\sigma$ = {sigma:g} px", loc="left")
        else:
            ax.set_title(f"({letter}) No PSF", loc="left")
        ax.axhline(0.0, color="0.5", linewidth=0.5)
        ax.set_ylabel("Bias in $u$ (px)")
        ax.set_xlim(0.0, 1.0)
        ax.tick_params(labelsize=style.SMALL_FONT_SIZE)
    axes[1].set_xlabel("Sub-pixel phase of the center in $u$ (px)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3,
               fontsize=style.SMALL_FONT_SIZE, handlelength=2.2, columnspacing=1.0)
    return fig


def plot(run: Run, out: Output, args: argparse.Namespace) -> None:
    df = prepare(run)
    methods = style.ordered(df["method"].unique())
    out.figure(figure(df), "e0_pixel_phase")

    table = (
        df.groupby(["method", "R_px", "effects.psf"])["err0_u_px"]
        .agg(range_px=lambda s: s.max() - s.min(), max_abs_px=lambda s: s.abs().max())
        .reset_index()
        .rename(columns={"R_px": "radius_px", "effects.psf": "psf"})
    )
    table["method"] = table["method"].astype("category").cat.set_categories(methods)
    table = table.sort_values(["method", "psf", "radius_px"]).reset_index(drop=True)
    out.table(table, NAME)

    # Центр тяжести линеен, но облучённость резкого диска, усреднённая по
    # пикселю и отнесённая к его центру, зависит от фазы края (наложение
    # спектров): без PSF это частично физика идеально резкого изображения.
    # PSF до пикселя, как в оптике, подавляет это в десятки раз.
    c = table[table["method"] == "centroid"]
    for r, d in c[c["psf"]].groupby("radius_px"):
        out.check(
            bool((d["range_px"] <= CENTROID_RANGE_MAX).all()),
            f"centroid с PSF, R = {r:g} пикс.: размах err0_u по фазе "
            f"{d['range_px'].max():.1e} пикс. (порог {CENTROID_RANGE_MAX:g})",
        )
    for _, row in c[~c["psf"]].iterrows():
        print(
            f"  centroid без PSF, R = {row['radius_px']:g} пикс.: размах "
            f"{row['range_px']:.1e} пикс. (пиксельная фаза резкого края)"
        )


if __name__ == "__main__":
    sys.exit(main("E0: собственное смещение методов", run, plot))
