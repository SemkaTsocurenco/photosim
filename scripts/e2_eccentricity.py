"""E2. Эксцентриситет по полю зрения.

E2a — формула (3) против точного решения, без рендера: oracle.conic_center
на сетке 25 × 21 положений по кадру (отступ 5 %), Z = 1 м, наклон 15°, 45°,
75° к оптической оси вокруг y, r = 5 и 10 мм, камеры review_16mm и rpi_cm3_2x2
(режим 2304×1296 с бинированием, как в E6 и бюджете);
плюс марка, развёрнутая к камере (нормаль на её центр), и развёртка по
наклону 0–85° на оптической оси (Z = 0,8; 1,0; 1,2 м).
eps = conic_center - проекция центра; формула (3):
eps3 = f r**2 n_z (n_x, n_y) / Z**2 (photosim.analysis.eccentricity_formula3).

E2b — рендер воспроизводит эталон, а метод эксцентриситет не убирает:
sweep без шумов (PSF включена), эталон proj_mid, все методы; 5 положений
(центр и четыре точки у углов) × 3 наклона. err0 — измеренный методом
эксцентриситет; сравнивается с точным conic_mid - proj_mid.

Там же точная замкнутая форма (photosim.analysis.eccentricity_pole: центр
эллипса — образ полюса прямой пересечения плоскости марки с z_cam = 0) и
исправленная формула первого порядка (eccentricity_first_order): формула
(3) теряет член вне оси (1 - n_z**2) (X, Y) / Z. Таблица e2_sensitivity:
d|eps|/d theta на оси, ~ cos 2 theta.

E2c — эксцентриситет против случайной погрешности: |eps| из E2a (45°,
r = 5 мм, на оси и в углу), СКО центра тяжести из E1 при той же марке и
fill = 0,7 и собственное смещение методов из E0 (R ≈ 23 пикс., с PSF).
Нужны папки запуска: --e1 runs/<...>_e1 [--e0 runs/<...>_e0].

    python scripts/e2_eccentricity.py run [--quick] [--jobs 6] [--e1 ...] [--e0 ...]
    python scripts/e2_eccentricity.py plot runs/<...>_e2 [--e1 ...] [--e0 ...]

Папка запуска: results.parquet — E2a, methods.parquet — E2b.
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
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from photosim.analysis import (
    eccentricity_first_order,
    eccentricity_formula3,
    eccentricity_pole,
    eccentricity_tilt_sensitivity,
    method_own_bias,
    point_at_pixel,
)
from photosim.camera import Pose
from photosim.estimators import ellipse_from_conic
from photosim.experiment import Config, Effects, Run, build, load_run, save_run, sweep
from photosim.figures import style
from photosim.figures.cli import Output, main
from photosim.oracle import _image_conic, conic_center
from photosim.presets import camera
from photosim.scene import frame_from_normal

NAME = "e2"
# RPi — в режиме с бинированием, как в E6, бюджете и табл. I статьи
CAMERAS = ("review_16mm", "rpi_cm3_2x2")
# На рисунках формула (3) обзора — «распространённое приближение»: в статье
# это распространённое приближение (ф. (11) Liebold & Maas, 2024), с номером
# самой статьи
APPROX = "common approximation"
CAMERA_LABEL = {
    "review_16mm": "16 mm lens, 3.45 µm pixels",
    "rpi_cm3_2x2": "Raspberry Pi Camera Module 3, 2×2 binning",
}
THETAS = (15.0, 45.0, 75.0)
RADII = (0.005, 0.010)
Z = 1.0
GRID = (25, 21)  # точек по u и по v
EDGE = 0.05  # отступ сетки от краёв кадра, доля
AXIS_THETAS = np.arange(0.0, 86.0, 1.0)
AXIS_Z = (0.8, 1.0, 1.2)
MAP_THETA, MAP_RADIUS = 45.0, 0.005
FIELD_CAMERA = "review_16mm"
REL_ERR_LIMIT = 0.10
CORNER = 0.1  # E2b: точки у углов, доля кадра от края
CENTROID_LEVEL = 3e-3  # уровень симулятора для центра тяжести, пикс.
THIN_PX = 5.0  # образ уже — проверка центра тяжести не применяется
# Выброс E2b: |измеренное - точное| больше этого, пикс. Разбор худшей точки
# (θ = 75°, левые углы, малая полуось 1,26 пикс.): контур выбран верно и
# края ROI не касается; ось образа шириной ~2,5 пикс. слегка наклонена и
# проходит по разным фазам пикселей (u = 244,89 вверху, 244,53 внизу), и
# нелинейные веса (I - порог)+ и концы контура половинного уровня становятся
# несимметричными вдоль большой оси: centroid_thr50 0,21 пикс., ellipse
# 0,08 пикс. по v. Это ограничение методов на узком образе
# (scripts/inspect_point.py).
OUTLIER_PX = 0.01
LABEL_PX = 0.03  # на рисунке подписывается малая полуось выбросов больше этого
POSE = Pose(np.eye(3), np.zeros(3))


def tilted_normal(theta_deg: float) -> np.ndarray:
    """Нормаль, наклонённая на ``theta`` к оптической оси вокруг y камеры."""
    t = math.radians(theta_deg)
    return np.array([-math.sin(t), 0.0, -math.cos(t)])


def eccentricity_row(cam, X, normal, r, **labels) -> dict:
    """Строка E2a; ``labels`` содержат и номинальные ``u``, ``v`` узла сетки.

    eps — oracle.conic_center; pole — точная замкнутая форма (полюс прямой);
    eps1 — исправленная формула первого порядка; eps3 — формула (3) обзора.
    """
    X = np.asarray(X, dtype=np.float64)
    proj = cam.project(X)
    eps = conic_center(cam, POSE, X, frame_from_normal(normal), r) - proj
    pole = eccentricity_pole(cam, X, normal, r)
    eps1 = eccentricity_first_order(cam.fx, r, normal, X)
    eps3 = eccentricity_formula3(cam.fx, r, normal, X[2])
    return {
        **labels,
        "r_m": r,
        "z_m": X[2],
        "n_x": normal[0],
        "n_y": normal[1],
        "n_z": normal[2],
        "eps_u": eps[0],
        "eps_v": eps[1],
        "pole_u": pole[0],
        "pole_v": pole[1],
        "eps1_u": eps1[0],
        "eps1_v": eps1[1],
        "eps3_u": eps3[0],
        "eps3_v": eps3[1],
    }


def part_a(quick: bool) -> pd.DataFrame:
    """E2a: точный эксцентриситет и формула (3) без рендера."""
    nu, nv = (9, 7) if quick else GRID
    rows = []
    for cam_name in CAMERAS:
        cam = camera(cam_name)
        us = np.linspace(EDGE, 1 - EDGE, nu) * (cam.width - 1)
        vs = np.linspace(EDGE, 1 - EDGE, nv) * (cam.height - 1)
        for r in RADII:
            for iu, u in enumerate(us):
                for iv, v in enumerate(vs):
                    X = np.array(point_at_pixel(cam, u, v, Z))
                    node = {"camera": cam_name, "iu": iu, "iv": iv, "u": u, "v": v}
                    for theta in THETAS:
                        rows.append(
                            eccentricity_row(
                                cam,
                                X,
                                tilted_normal(theta),
                                r,
                                case="tilt",
                                theta_deg=theta,
                                **node,
                            )
                        )
                    rows.append(
                        eccentricity_row(
                            cam,
                            X,
                            -X / np.linalg.norm(X),
                            r,
                            case="facing",
                            theta_deg=0.0,
                            **node,
                        )
                    )
    cam = camera("review_16mm")
    on_axis = {"camera": "review_16mm", "iu": -1, "iv": -1, "u": cam.cx, "v": cam.cy}
    for r in RADII:
        for z in AXIS_Z:
            for theta in AXIS_THETAS:
                rows.append(
                    eccentricity_row(
                        cam,
                        (0.0, 0.0, z),
                        tilted_normal(theta),
                        r,
                        case="axis",
                        theta_deg=theta,
                        **on_axis,
                    )
                )
    df = pd.DataFrame(rows)
    df["eps_abs"] = np.hypot(df["eps_u"], df["eps_v"])
    df["eps3_abs"] = np.hypot(df["eps3_u"], df["eps3_v"])
    df["pole_err_px"] = np.hypot(df["pole_u"] - df["eps_u"], df["pole_v"] - df["eps_v"])
    # при |eps| ~ 0 (θ = 0 на оси) относительная ошибка не определена
    defined = df["eps_abs"] > 1e-9
    for key, col in (("3", "rel_err"), ("1", "rel_err1")):
        err = np.hypot(df[f"eps{key}_u"] - df["eps_u"], df[f"eps{key}_v"] - df["eps_v"])
        df[col] = np.where(defined, err / df["eps_abs"].clip(1e-9), np.nan)
    return df


def part_b(quick: bool, jobs: int) -> tuple[pd.DataFrame, Config, dict]:
    """E2b: рендер и все методы без шумов."""
    cam = camera("review_16mm")
    corners = [
        (CORNER * (cam.width - 1), CORNER * (cam.height - 1)),
        ((1 - CORNER) * (cam.width - 1), CORNER * (cam.height - 1)),
        (CORNER * (cam.width - 1), (1 - CORNER) * (cam.height - 1)),
        ((1 - CORNER) * (cam.width - 1), (1 - CORNER) * (cam.height - 1)),
    ]
    pixels = [(cam.cx, cam.cy), *corners]
    if quick:
        pixels, thetas = pixels[:2], [45.0]
    else:
        thetas = list(THETAS)
    base = Config(
        camera="review_16mm",
        radius=MAP_RADIUS,
        tilt_axis="y",
        tilt_reference="optical_axis",
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
        reference="proj_mid",
        ss=16,
    )
    grid = {
        "position": [point_at_pixel(cam, u, v, Z) for u, v in pixels],
        "tilt_deg": thetas,
    }
    df = sweep(base, grid, n_jobs=jobs)
    # точные полуоси образа: узкие эллипсы сильнее страдают от пиксельной фазы
    axes = []
    for pos, theta in itertools.product(grid["position"], grid["tilt_deg"]):
        s = build(replace(base, position=pos, tilt_deg=theta))
        C = _image_conic(
            s.camera,
            s.pose,
            s.motion.position(s.t_ref),
            s.motion.orientation(s.t_ref),
            s.target.radius,
        )
        axes.append(ellipse_from_conic(C)[1])
    df["semi_major_px"] = [axes[i][0] for i in df["point"]]
    df["semi_minor_px"] = [axes[i][1] for i in df["point"]]
    return df, base, grid


def run(args: argparse.Namespace) -> Path:
    df_a = part_a(args.quick)
    df_b, base_b, grid_b = part_b(args.quick, args.jobs)
    configs = {
        "experiment": NAME,
        "quick": args.quick,
        "a": {
            "cameras": list(CAMERAS),
            "thetas_deg": list(THETAS),
            "radii_m": list(RADII),
            "z_m": Z,
            "grid": list(GRID),
            "edge": EDGE,
        },
        "base_b": base_b,
        "grid_b": grid_b,
    }
    name = NAME + ("_quick" if args.quick else "")
    return save_run(df_a, configs, name, args.root, tables={"methods": df_b})


# Рисунки


def plot_axis(df: pd.DataFrame, out: Output) -> None:
    axis = df[df["case"] == "axis"]
    fig, ax = plt.subplots(figsize=(style.COLUMN, 3.0))
    for r, color in zip(RADII, ("0.0", "0.5"), strict=True):
        for z, ls in zip((0.8, 1.2), ("-", "--"), strict=True):
            d = axis[np.isclose(axis["r_m"], r) & np.isclose(axis["z_m"], z)]
            label = f"r = {r * 1e3:g} mm, Z = {z:g} m"
            ax.plot(
                d["theta_deg"], d["eps_abs"], color=color, linestyle=ls, label=label
            )
            every = d[d["theta_deg"] % 5 == 0]
            ax.plot(
                every["theta_deg"],
                every["eps3_abs"],
                linestyle="",
                marker="o",
                markerfacecolor="white",
                color=color,
            )
    ax.plot(
        [],
        [],
        linestyle="",
        marker="o",
        markerfacecolor="white",
        color="0.0",
        label="Approximation",
    )
    ax.set_xlabel("Tilt to the image plane, $\\theta$ (deg)")
    ax.set_ylabel("Eccentricity on the axis, $|\\varepsilon|$ (px)")
    ax.set_xlim(0, 85)
    ax.set_ylim(bottom=0)
    fig.legend(loc="outside lower center", ncols=2)  # под рисунком: не на кривых
    out.figure(fig, "e2_axis")

    ok = axis[axis["theta_deg"] > 0]
    worst = float(ok["rel_err"].max())
    out.check(worst < 1e-3, f"на оси формула (3) совпадает с точной: до {worst:.1e}")


def grid_of(d: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Значения на сетке кадра: (u, v, Z[v, u]) по индексам узлов iu, iv."""
    table = d.pivot(index="iv", columns="iu", values=value)
    u = d.groupby("iu")["u"].first().loc[table.columns].to_numpy()
    v = d.groupby("iv")["v"].first().loc[table.index].to_numpy()
    return u, v, table.to_numpy()


def plot_formula_error(df: pd.DataFrame, out: Output) -> None:
    out.figure(formula_error_figure(df), "e2_formula_error")


def formula_error_figure(df: pd.DataFrame, facing: bool = False) -> Figure:
    """Карты относительной ошибки формулы (3) по кадрам двух камер.

    ``facing`` — третья панель: поле эксцентриситета марки, развёрнутой к
    камере (review_16mm), стрелками (рис. 4 статьи).
    """
    sel = df[
        (df["case"] == "tilt")
        & np.isclose(df["theta_deg"], MAP_THETA)
        & np.isclose(df["r_m"], MAP_RADIUS)
    ]
    cams = [c for c in CAMERAS if c in set(sel["camera"])]
    # у центрального узла ошибка ~1e-4: шкала от 1e-2, чтобы было видно кадр
    norm = LogNorm(vmin=1e-2, vmax=max(float(sel["rel_err"].max()), 0.2), clip=True)
    # ширины панелей пропорциональны сторонам кадров: при равном масштабе
    # осей панели одной высоты
    ratios = [camera(c).width / camera(c).height for c in cams]
    if facing:
        ratios.append(camera(FIELD_CAMERA).width / camera(FIELD_CAMERA).height)
    fig, axes = plt.subplots(
        1, len(ratios), figsize=(style.DOUBLE, 2.5 if facing else 2.9), squeeze=False,
        width_ratios=ratios,
    )
    for ax, cam_name in zip(axes[0], cams, strict=False):
        d = sel[sel["camera"] == cam_name]
        u, v, err = grid_of(d, "rel_err")
        mesh = ax.pcolormesh(u, v, err, norm=norm, cmap="cividis", shading="nearest")
        if (err > REL_ERR_LIMIT).any() and (err < REL_ERR_LIMIT).any():
            ax.contour(
                u, v, err, levels=[REL_ERR_LIMIT], colors="white", linewidths=0.8
            )
        _, _, sign = grid_of(d.assign(s=d["eps_u"] * d["eps3_u"]), "s")
        if (sign < 0).any() and (sign > 0).any():
            ax.contour(
                u,
                v,
                sign,
                levels=[0.0],
                colors="white",
                linewidths=0.8,
                linestyles="--",
            )
        cam = camera(cam_name)
        ax.set_xlim(-0.5, cam.width - 0.5)
        ax.set_ylim(cam.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(CAMERA_LABEL.get(cam_name, cam_name))
        ax.set_xlabel("$u$ (px)")
        ax.set_ylabel("$v$ (px)")
    if facing:
        # шкала — под картами (а), (б), но место под неё берётся у всех трёх
        # панелей: иначе (в) выше остальных. anchor сдвигает её под (а), (б)
        ratios_ab = sum(ratios[: len(cams)]) / sum(ratios)
        shrink = 0.45
        cbar = fig.colorbar(
            mesh, ax=axes[0].tolist(), extend="min", location="bottom",
            shrink=shrink, aspect=40, label=f"Relative error of the {APPROX}",
            anchor=((ratios_ab / 2 - shrink / 2) / (1 - shrink), 1.0),
        )
        cbar.ax.axvline(REL_ERR_LIMIT, color="white", linewidth=1.5)
        for ax, letter in zip(axes[0], "ab", strict=False):
            title = ax.get_title()
            ax.set_title("")
            ax.set_title(f"({letter}) {title}", loc="left")
        facing_panel(axes[0][-1], df)
        axes[0][-1].set_title("(c) Facing the camera, 16 mm lens", loc="left")
        return fig
    cbar = fig.colorbar(
        mesh, ax=axes[0].tolist(), extend="min", label=f"Relative error of the {APPROX}"
    )
    cbar.ax.axhline(REL_ERR_LIMIT, color="white", linewidth=1.5)  # уровень контура
    sign_note = (
        "; dashed: sign of $\\varepsilon_u$ differs"
        if (sel["eps_u"] * sel["eps3_u"] < 0).any()
        else ""
    )
    fig.suptitle(
        f"The {APPROX} vs exact eccentricity: $\\theta$ = {MAP_THETA:g}°, "
        f"r = {MAP_RADIUS * 1e3:g} mm, Z = {Z:g} m. "
        f"White line: {REL_ERR_LIMIT:.0%} error boundary{sign_note}"
    )
    return fig


def facing_panel(ax, df: pd.DataFrame) -> None:
    """Поле эксцентриситета марки, развёрнутой к камере (нормаль на её центр)."""
    cam = camera(FIELD_CAMERA)
    d = df[
        (df["camera"] == FIELD_CAMERA)
        & np.isclose(df["r_m"], MAP_RADIUS)
        & (df["case"] == "facing")
    ]
    step = 3 if int(d["iu"].max()) + 1 > 12 else 1
    u_nodes = d.groupby("iu")["u"].first()
    spacing = float(u_nodes.loc[1] - u_nodes.loc[0]) * step
    eps_max = float(d["eps_abs"].max())
    gain = nice(0.9 * spacing / eps_max)
    key = nice(eps_max)
    d = d[(d["iu"] % step == 0) & (d["iv"] % step == 0)]
    q = ax.quiver(
        d["u"], d["v"], d["eps_u"] * gain, d["eps_v"] * gain, angles="xy",
        scale_units="xy", scale=1.0, width=0.008, color="0.0",
    )
    ax.quiverkey(q, 0.08, -0.3, key * gain, f"{key:g} px (arrows ×{gain:g})",
                 labelpos="E", coordinates="axes")
    ax.set_xlim(-0.5, cam.width - 0.5)
    ax.set_ylim(cam.height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$u$ (px)")
    ax.set_ylabel("$v$ (px)")


def nice(x: float) -> float:
    """Ближайшее снизу число вида 1, 2, 5 × 10^k."""
    k = math.floor(math.log10(x))
    return max(m * 10**k for m in (1, 2, 5) if m * 10**k <= x)


def plot_field(df: pd.DataFrame, out: Output) -> None:
    cam_name = "review_16mm"
    cam = camera(cam_name)
    base = df[(df["camera"] == cam_name) & np.isclose(df["r_m"], MAP_RADIUS)]
    panels = [
        (
            base[(base["case"] == "tilt") & np.isclose(base["theta_deg"], MAP_THETA)],
            f"(a) Tilt {MAP_THETA:g}°",
        ),
        (base[base["case"] == "facing"], "(b) Facing the camera"),
    ]
    n_u = int(base["iu"].max()) + 1
    step = 2 if n_u > 12 else 1
    u_nodes = base.groupby("iu")["u"].first()
    spacing = float(u_nodes.loc[1] - u_nodes.loc[0]) * step
    eps_max = max(float(d["eps_abs"].max()) for d, _ in panels)
    gain = nice(0.8 * spacing / eps_max)  # во сколько раз увеличены стрелки
    key = nice(eps_max / 2)
    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE, 2.7))
    for ax, (d, title) in zip(axes, panels, strict=True):
        d = d[(d["iu"] % step == 0) & (d["iv"] % step == 0)]
        q = ax.quiver(
            d["u"],
            d["v"],
            d["eps_u"] * gain,
            d["eps_v"] * gain,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            color="0.0",
        )
        ax.quiverkey(
            q,
            0.97,
            1.04,
            key * gain,
            f"{key:g} px (×{gain:g})",
            labelpos="W",
            coordinates="axes",
        )
        ax.set_xlim(-0.5, cam.width - 0.5)
        ax.set_ylim(cam.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left")
        ax.set_xlabel("$u$ (px)")
        ax.set_ylabel("$v$ (px)")
    out.figure(fig, "e2_field")


def table_a(df: pd.DataFrame, out: Output) -> pd.DataFrame:
    field = df[df["case"].isin(["tilt", "facing"])].copy()
    field["sign_differs_u"] = (field["eps_u"] * field["eps3_u"] < 0).astype(float)
    table = (
        field.groupby(["camera", "r_m", "case", "theta_deg"])
        .agg(
            # формула (3) обзора против точного conic_center
            rel_err_eq3_max=("rel_err", "max"),
            rel_err_eq3_median=("rel_err", "median"),
            frac_rel_err_eq3_gt_10pct=(
                "rel_err",
                lambda s: float((s > REL_ERR_LIMIT).mean()),
            ),
            eps_max_px=("eps_abs", "max"),
            eps_median_px=("eps_abs", "median"),
            frac_sign_differs_u=("sign_differs_u", "mean"),
            # исправленная формула первого порядка (с членом вне оси)
            rel_err_first_order_max=("rel_err1", "max"),
        )
        .reset_index()
    )
    table.insert(1, "r_mm", table.pop("r_m") * 1e3)
    out.table(table, NAME)

    pole = float(df["pole_err_px"].max())
    out.check(pole <= 1e-9, f"полюсная формула = conic_center: до {pole:.1e} пикс.")
    # ошибка первого порядка — второй порядок по r/Z: (r/Z)**2 = 2,5e-5 ... 1,6e-4
    first = df["rel_err1"] / (df["r_m"] / df["z_m"]) ** 2
    out.check(
        float(first.max()) <= 2.0,
        f"исправленная формула первого порядка: относительная ошибка до "
        f"{df['rel_err1'].max():.1e}, не больше {first.max():.2f} (r/Z)**2",
    )
    return table


def table_sensitivity(df: pd.DataFrame, out: Output) -> None:
    """d|eps|/d theta на оптической оси, пикс./град.

    ``deps_dtheta_exact`` — численная производная точного eps (conic_center,
    сетка 1°); ``deps_dtheta_eq3_axis`` — производная формулы (3) на оси:
    f r² cos 2θ / Z² (на оси член вне оси равен нулю).
    """
    axis = df[(df["case"] == "axis") & np.isclose(df["z_m"], Z)]
    f_px = camera("review_16mm").fx  # развёртка по наклону — на этой камере
    rows = []
    for r, d in axis.groupby("r_m"):
        d = d.sort_values("theta_deg")
        theta = d["theta_deg"].to_numpy()
        slope = np.gradient(d["eps_u"].to_numpy(), theta)  # пикс./град.
        for t in (5.0, 15.0, 30.0, 45.0, 60.0, 75.0):
            i = int(np.argmin(np.abs(theta - t)))
            rows.append(
                {
                    "r_mm": r * 1e3,
                    "theta_deg": t,
                    "eps_exact_px": float(d["eps_u"].iloc[i]),
                    "deps_dtheta_exact_px_per_deg": float(slope[i]),
                    "deps_dtheta_eq3_axis_px_per_deg": float(
                        eccentricity_tilt_sensitivity(f_px, r, math.radians(t), Z)
                    )
                    * math.pi
                    / 180,
                }
            )
    out.table(pd.DataFrame(rows), "e2_sensitivity")


def plot_methods(run: Run, out: Output) -> None:
    df = run.tables["methods"]
    df = df[df["status"] == "ok"].copy()
    df["exact_u"] = df["ref.conic_mid.u"] - df["ref.proj_mid.u"]
    df["exact_v"] = df["ref.conic_mid.v"] - df["ref.proj_mid.v"]
    df["diff_u"] = df["err0_u_px"] - df["exact_u"]
    df["diff_v"] = df["err0_v_px"] - df["exact_v"]
    methods = style.ordered(df["method"].unique())
    thetas = sorted(df["tilt_deg"].unique())

    # по панели на наклон: видно, откуда выбросы
    fig, axes = plt.subplots(
        1, len(thetas), figsize=(style.DOUBLE, 3.0), squeeze=False
    )
    for ax, theta in zip(axes[0], thetas, strict=True):
        sub = df[df["tilt_deg"] == theta]
        values = sub[["exact_u", "exact_v", "err0_u_px", "err0_v_px"]]
        lim = 1.1 * float(values.abs().max().max())
        ax.plot([-lim, lim], [-lim, lim], color="0.6", linewidth=0.6, label="$y = x$")
        for m in methods:
            d = sub[sub["method"] == m]
            st = style.method_style(m)
            st["linestyle"] = ""
            ax.plot(d["exact_u"], d["err0_u_px"], **st)
            st.pop("label")
            ax.plot(d["exact_v"], d["err0_v_px"], **st)
        # подписать малую полуось у самого большого выброса каждого образа
        far = sub[np.hypot(sub["diff_u"], sub["diff_v"]) > LABEL_PX]
        for _, group in far.groupby(far["semi_minor_px"].round(2)):
            r = group.loc[np.hypot(group["diff_u"], group["diff_v"]).idxmax()]
            comp = "u" if abs(r["diff_u"]) > abs(r["diff_v"]) else "v"
            ax.annotate(
                f"b = {r['semi_minor_px']:.1f} px",
                (r[f"exact_{comp}"], r[f"err0_{comp}_px"]),
                xytext=(5, 0), textcoords="offset points", va="center",
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        b_min = float(sub["semi_minor_px"].min())
        ax.set_title(f"$\\theta$ = {theta:g}° (minor semi-axis from {b_min:.1f} px)")
        ax.set_xlabel("Exact eccentricity, $u$ and $v$ (px)")
    axes[0][0].set_ylabel("Measured noise-free offset (px)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    out.figure(fig, "e2_methods")

    # выбросы: положение, наклон, полуоси
    big = df[np.hypot(df["diff_u"], df["diff_v"]) > OUTLIER_PX]
    out.table(
        big[
            [
                "method", "tilt_deg", "ref.proj_mid.u", "ref.proj_mid.v",
                "semi_major_px", "semi_minor_px", "diff_u", "diff_v",
            ]
        ]
        .rename(columns={"ref.proj_mid.u": "u_px", "ref.proj_mid.v": "v_px"})
        .sort_values(["tilt_deg", "method"]),
        "e2_outliers",
    )

    rows = []
    for m in methods:
        d = df[df["method"] == m]
        resid = np.concatenate(
            [d["err0_u_px"] - d["exact_u"], d["err0_v_px"] - d["exact_v"]]
        )
        row = {
            "method": m,
            "max_abs_diff_px": float(np.abs(resid).max()),
            "rms_diff_px": float(np.sqrt(np.mean(resid**2))),
            "max_abs_eccentricity_px": float(
                d[["exact_u", "exact_v"]].abs().max().max()
            ),
            "ok0_frac": float(d["ok0_frac"].mean()),
        }
        if "info0.status" in d and d["info0.status"].notna().any():
            # сходимость подгонки (least_squares: status > 0)
            row["converged_frac"] = float((d["info0.status"] > 0).mean())
            row["nfev_max"] = float(d["info0.nfev"].max())
        rows.append(row)
    table = pd.DataFrame(rows)
    out.table(table, "e2_methods")

    # У узкого образа (марка почти с ребра) центр тяжести сильнее зависит от
    # фазы края относительно пикселей — как маленькая марка в E0/E1.
    c = df[df["method"] == "centroid"]
    diff = np.maximum(
        (c["err0_u_px"] - c["exact_u"]).abs(), (c["err0_v_px"] - c["exact_v"]).abs()
    )
    wide = c["semi_minor_px"] >= THIN_PX
    if wide.any():
        out.check(
            float(diff[wide].max()) <= CENTROID_LEVEL,
            f"центр тяжести воспроизводит точный эксцентриситет при малой полуоси "
            f">= {THIN_PX:g} пикс.: до {diff[wide].max():.1e} пикс. "
            f"(порог {CENTROID_LEVEL:g})",
        )
    for (_, row), dd in zip(c[~wide].iterrows(), diff[~wide], strict=True):
        print(
            f"  узкий образ (малая полуось {row['semi_minor_px']:.1f} пикс., "
            f"наклон {row['tilt_deg']:g}°): центр тяжести {dd:.1e} пикс."
        )


def own_bias(e0: Run) -> dict[str, float]:
    """Собственное смещение методов из E0 (``analysis.method_own_bias``) при
    радиусе образа марки E2 (r = 5 мм, ≈ 23 пикс.), в порядке стиля."""
    own = method_own_bias(e0.df, camera("review_16mm").fx * MAP_RADIUS / Z)
    return {m: own[m] for m in style.ordered(own)}


def plot_vs_noise(df: pd.DataFrame, e1: Run, e0: Run | None, out: Output) -> None:
    axis = df[
        (df["case"] == "axis")
        & np.isclose(df["theta_deg"], MAP_THETA)
        & np.isclose(df["r_m"], MAP_RADIUS)
        & np.isclose(df["z_m"], Z)
    ]
    field = df[
        (df["case"] == "tilt")
        & (df["camera"] == "review_16mm")
        & np.isclose(df["theta_deg"], MAP_THETA)
        & np.isclose(df["r_m"], MAP_RADIUS)
    ]
    eps_axis = float(axis["eps_abs"].iloc[0])
    eps_corner = float(field["eps_abs"].max())
    noise = e1.df[
        (e1.df["status"] == "ok")
        & (e1.df["method"] == "centroid")
        & np.isclose(e1.df["radius"], MAP_RADIUS)
        & np.isclose(e1.df["fill"], 0.7)
    ]
    if noise.empty:
        print("в E1 нет точки r = 5 мм, fill = 0,7: E2c пропущен")
        return
    sigma = float(noise["std_u_px"].iloc[0])

    bars = [
        ("Eccentricity, on axis", eps_axis, "0.2"),
        ("Eccentricity, frame corner", eps_corner, "0.2"),
        ("Random error (std, E1)", sigma, "0.55"),
    ]
    own = own_bias(e0) if e0 is not None else {}
    for m, value in own.items():
        label = style.METHOD_STYLE.get(m, {}).get("label", m)
        bars.append((f"Own bias: {label}", value, "0.8"))
    labels, values, colors = zip(*bars, strict=True)
    fig, ax = plt.subplots(figsize=(style.COLUMN, 0.35 * len(bars) + 0.8))
    y = np.arange(len(bars))[::-1]
    ax.barh(y, values, color=colors, height=0.6)
    for yi, value in zip(y, values, strict=True):
        ax.annotate(f"{value:.2g}", (value, yi), xytext=(2, 0),
                    textcoords="offset points", va="center")
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlim(0.3 * min(values), 6 * max(values))
    ax.set_xlabel("Pixels")
    ax.grid(axis="y", visible=False)
    ax.set_title(
        f"r = {MAP_RADIUS * 1e3:g} mm, $\\theta$ = {MAP_THETA:g}°, fill 0.7", loc="left"
    )
    out.figure(fig, "e2_vs_noise")
    row = {
        "eps_axis_px": eps_axis,
        "eps_corner_px": eps_corner,
        "std_u_px": sigma,
        "ratio_axis": eps_axis / sigma,
        "ratio_corner": eps_corner / sigma,
        **{f"own_bias_{m}_px": v for m, v in own.items()},
        "e1_run": str(e1.path),
        "e0_run": str(e0.path) if e0 is not None else "",
    }
    out.table(pd.DataFrame([row]), "e2_vs_noise")
    print(
        f"эксцентриситет в {eps_axis / sigma:.0f} раз (на оси) и в "
        f"{eps_corner / sigma:.0f} раз (в углу) больше случайной погрешности"
    )


def plot(run: Run, out: Output, args: argparse.Namespace) -> None:
    df = run.df
    plot_axis(df, out)
    plot_formula_error(df, out)
    plot_field(df, out)
    table_a(df, out)
    table_sensitivity(df, out)
    plot_methods(run, out)
    if args.e1 is not None:
        e0 = load_run(args.e0) if args.e0 is not None else None
        plot_vs_noise(df, load_run(args.e1), e0, out)
    else:
        print("E2c пропущен: укажите папку запуска E1 через --e1")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--e1", type=Path, default=None, help="папка запуска E1 для E2c"
    )
    parser.add_argument(
        "--e0", type=Path, default=None,
        help="папка запуска E0: собственное смещение методов на рисунке E2c",
    )


if __name__ == "__main__":
    sys.exit(main("E2: эксцентриситет", run, plot, add_arguments=add_arguments))
