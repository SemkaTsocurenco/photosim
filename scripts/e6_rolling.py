"""E6. Построчный затвор.

Raspberry Pi Camera Module 3 в режиме 2304×1296 с бинированием 2×2
(rpi_cm3_2x2: f ≈ 1693 пикс.), шумы — допущение RPI_CM3_ASSUMED, PSF 0,7
пикс. в рендере. Марка r = 5 мм (R ≈ 8,5 пикс.), Z = 1 м, плоскость
параллельна сенсору, τ = 1 мс, fill = 0,7, n_mc = 500.

Время строки не измерено: оценка по паспортной частоте кадров режима,
t_row = 1 / (fps_max * rows) = 1 / (56 * 1296) ≈ 13,8 мкс
(presets.row_time_estimate). Это оценка сверху: в период кадра входит и
гашение по вертикали; точнее — из picamera2 sensor_modes. Результаты — при
t_row × 0,5; 1; 1,5, и при t_row = 0 (глобальный затвор, те же точки) —
база для перекоса: у нелинейных методов своё смещение пиксельной фазы.

Перебор: скорость 0,1; 0,25; 0,5; 1,0 м/с; направление вдоль строк (u),
поперёк (v, вниз) и под 45°; положение марки в середине экспозиции строки
0 — у верха, в центре и у низа кадра (0,1; 0,5; 0,9 высоты), по u — в
центре; 4 × 3 × 3 × 4 = 144 точки.

Метрики (колонки ref.* запуска; эталоны сдвинуты на время строки центра):

- привязка ко времени кадра — ошибка относительно proj_frame_mid (середина
  экспозиции строки 0), так делает обычная обработка. Теория: строка
  захвата v* = v_ref / (1 - nu_v t_row) (nu_v — скорость образа по v,
  пикс./с), ошибка = скорость * t_row * v*, мм на объекте; линейна по строке
  захвата при любом направлении;
- привязка ко времени строки центра — ошибка относительно exposure_conic
  (остаточный перекос формы: строки видят марку в разные моменты, образ
  срезан или растянут, но центрально-симметричен);
- рассинхронизация двух камер: одна марка в строках, отличающихся на Δrow,
  Δt = Δrow t_row, ошибка v Δt (без рендера).

    python scripts/e6_rolling.py run [--quick] [--jobs 6]
    python scripts/e6_rolling.py plot runs/<...>_e6

Рисунок e6_rolling: (а) ошибка привязки ко времени кадра, мм на объекте, от
строки марки, по скоростям (движение вдоль u; полосы — t_row ± 50 %);
(б) остаточная ошибка при построчной привязке, пикс., от скорости — все
методы, худшее по положениям и направлениям; серые открытые маркеры — тот
же метод при глобальном затворе.
Таблицы: e6_table.csv — все точки; e6_summary_table.csv — X и Y для
аннотации по t_row; e6_desync_table.csv — рассинхронизация камер.
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

from photosim.analysis import point_at_pixel
from photosim.experiment import Config, Effects, Run, run_configs, save_run
from photosim.figures import style
from photosim.figures.cli import Output, main
from photosim.presets import RPI_CM3_ASSUMED, camera, row_time_estimate

NAME = "e6"
CAMERA = "rpi_cm3_2x2"
SENSOR = "rpi_cm3"
Z = 1.0
RADIUS = 0.005
TAU = 1e-3
SPEEDS = (0.1, 0.25, 0.5, 1.0)  # м/с
DIRECTIONS = {"u": 0.0, "v": 90.0, "diag": 45.0}  # угол от u к v, градусы
ROWS = {"top": 0.1, "center": 0.5, "bottom": 0.9}  # доля (H - 1)
T_ROW_FACTORS = (0.0, 0.5, 1.0, 1.5)  # 0 — глобальный затвор
PHASE = 0.3
N_MC = 500
SIM_LEVEL = 2e-3  # уровень симулятора при ss = 16, пикс.
DESYNC_ROWS = (1, 10, 100, 648, 1296)
TEXT_SPEED = 0.5  # м/с, пример обзора


def t_row_estimate() -> float:
    return row_time_estimate(CAMERA)


def velocity(speed: float, direction_deg: float) -> tuple[float, float, float]:
    a = math.radians(direction_deg)
    return (speed * math.cos(a), speed * math.sin(a), 0.0)


def run(args: argparse.Namespace) -> Path:
    cam = camera(CAMERA)
    t_row0 = t_row_estimate()
    if args.quick:
        speeds, directions, rows, factors, n_mc = (0.1, 1.0), ("u", "v"), ROWS, (0.0, 1.0), 50
    else:
        speeds, directions, rows, factors, n_mc = SPEEDS, DIRECTIONS, ROWS, T_ROW_FACTORS, N_MC
    base = Config(
        camera=CAMERA,
        sensor=SENSOR,
        sensor_overrides=RPI_CM3_ASSUMED,
        radius=RADIUS,
        tau=TAU,
        shutter="rolling",
        effects=Effects(rotation=False),
        n_mc=n_mc,
        reference="exposure_conic",
        ss=16,
        nt=64,
    )
    configs, labels = [], []
    for speed, direction, row, factor in itertools.product(
        speeds, directions, rows, factors
    ):
        u = cam.cx + PHASE
        v = round(ROWS[row] * (cam.height - 1)) + PHASE
        configs.append(
            replace(
                base,
                position=point_at_pixel(cam, u, v, Z),
                motion={
                    "type": "linear",
                    "velocity": velocity(speed, DIRECTIONS[direction]),
                },
                t_row=factor * t_row0,
            )
        )
        labels.append(
            {
                "speed": speed,
                "direction": direction,
                "row_position": row,
                "t_row_factor": factor,
            }
        )
    df = run_configs(configs, labels=labels, seed=base.seed, n_jobs=args.jobs)
    configs_json = {
        "experiment": NAME,
        "quick": args.quick,
        "base": base,
        "t_row_estimate_s": t_row0,
        "speeds": list(speeds),
        "directions": {d: DIRECTIONS[d] for d in directions},
        "rows": {r: ROWS[r] for r in rows},
        "t_row_factors": list(factors),
    }
    name = NAME + ("_quick" if args.quick else "")
    return save_run(df, configs_json, name, args.root)


# Обработка


def with_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["status"] == "ok"].copy()
    mm = df["um_per_px"] / 1e3
    # ошибка относительно времени кадра: bias + (exposure_conic - proj_frame_mid)
    du = df["bias_u_px"] + df["ref.exposure_conic.u"] - df["ref.proj_frame_mid.u"]
    dv = df["bias_v_px"] + df["ref.exposure_conic.v"] - df["ref.proj_frame_mid.v"]
    df["frame_err_px"] = np.hypot(du, dv)
    df["frame_err_mm"] = df["frame_err_px"] * mm
    # теория: строка захвата v* = v_ref / (1 - nu_v t_row)
    angle = np.radians(df["direction"].map(DIRECTIONS))
    nu_v = df["f_px"] * df["speed"] * np.sin(angle) / df["z_m"]
    v_star = df["ref.proj_frame_mid.v"] / (1 - nu_v * df["t_row"])
    df["frame_err_theory_mm"] = df["speed"] * df["t_row"] * v_star * 1e3
    df["capture_row"] = df["ref.proj_mid.v"]
    # остаток при построчной привязке
    df["resid_px"] = np.hypot(df["bias_u_px"], df["bias_v_px"])
    df["resid0_px"] = np.hypot(df["err0_u_px"], df["err0_v_px"])
    df["resid_se_px"] = np.hypot(df["se_bias_u_px"], df["se_bias_v_px"])
    return df


def plot(run: Run, out: Output, args: argparse.Namespace) -> None:
    df = with_metrics(run.df)
    t_row0 = float(run.configs["t_row_estimate_s"])
    factors = sorted(df.loc[df["t_row_factor"] > 0, "t_row_factor"].unique())

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(style.DOUBLE, 3.0))
    panel_frame(ax_a, df, t_row0)
    ax_a.set_title("(a) Frame-time stamp, motion along rows")
    ax_a.annotate(
        f"lines: speed × $t_{{row}}$ × row, $t_{{row}}$ = {t_row0 * 1e6:.1f} µs\n"
        f"bands: $t_{{row}}$ × {min(factors):g} ... {max(factors):g}\npoints: centroid",
        (0.33, 0.97), xycoords="axes fraction", ha="left", va="top",
    )
    panel_residual(ax_b, df)
    ax_b.set_title("(b) Row-time stamp: worst over rows and directions")
    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    out.figure(fig, "e6_rolling")

    methods = style.ordered(df["method"].unique())
    keep = [
        "speed", "direction", "row_position", "t_row_factor", "t_row", "method",
        "capture_row", "ref_time_shift", "frame_err_mm", "frame_err_theory_mm",
        "frame_err_px", "resid_px", "resid0_px", "resid_se_px", "std_u_px", "std_v_px",
        "um_per_px",
    ]
    table = df[keep].copy()
    table["method"] = pd.Categorical(table["method"], methods)
    out.table(table.sort_values(["t_row_factor", "speed", "direction", "row_position",
                                 "method"]), NAME)
    summary = summary_table(df, out)
    desync_table(df, t_row0, out)
    check(df, summary, out)


def panel_frame(ax, df: pd.DataFrame, t_row0: float) -> None:
    """Ошибка привязки ко времени кадра, мм на объекте, от строки (движение вдоль
    u, центр тяжести): линия — скорость * t_row * строка, подписана скоростью у
    правого конца; полоса — t_row × min ... max множителя; точки —
    моделирование, маркером центра тяжести. Своих маркеров и типов линий у
    скоростей нет: на общей легенде рисунка они означают методы."""
    H = camera(CAMERA).height
    speeds = sorted(df["speed"].unique())
    rolling = df[df["t_row_factor"] > 0]
    est = df[np.isclose(df["t_row_factor"], 1.0)]
    rows = np.linspace(0, H - 1, 50)
    factors = sorted(rolling["t_row_factor"].unique())
    lo, hi = min(factors), max(factors)
    centroid = style.method_style("centroid")
    for speed in speeds:
        ax.fill_between(rows, speed * lo * t_row0 * rows * 1e3,
                        speed * hi * t_row0 * rows * 1e3, color="0.5", alpha=0.12,
                        linewidth=0)
        line = speed * t_row0 * rows * 1e3
        ax.plot(rows, line, color="0.3", linewidth=0.7, linestyle="-", marker="")
        ax.annotate(f"{speed:g} m/s", (rows[-1], line[-1]), xytext=(2, 0),
                    textcoords="offset points", ha="left", va="center")
        d = est[(est["speed"] == speed) & (est["direction"] == "u")
                & (est["method"] == "centroid")].sort_values("capture_row")
        ax.plot(d["capture_row"], d["frame_err_mm"], linestyle="",
                marker=centroid["marker"], color=centroid["color"],
                markerfacecolor=centroid["markerfacecolor"])
    ax.set_xlim(0, H - 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Image row of the target, $v$ (px)")
    ax.set_ylabel("Frame-time stamp error (mm on the target)")


def panel_residual(ax, df: pd.DataFrame) -> None:
    """Остаток при построчной привязке, пикс., от скорости: все методы, худшее
    по строкам и направлениям при t_row оценке; серые открытые маркеры — тот
    же метод при глобальном затворе."""
    est = df[np.isclose(df["t_row_factor"], 1.0)]
    glob = df[df["t_row_factor"] == 0]
    for m in style.ordered(df["method"].unique()):
        worst = est[est["method"] == m].groupby("speed")["resid_px"].max()
        ax.plot(worst.index, worst.to_numpy(), **style.method_style(m))
        if not glob.empty:
            base = glob[glob["method"] == m].groupby("speed")["resid_px"].max()
            st = style.method_style(m)
            ax.plot(base.index, base.to_numpy(), linestyle="", marker=st["marker"],
                    color="0.6", markerfacecolor="none")
    ax.axhline(SIM_LEVEL, color="0.3", linewidth=0.6, linestyle=":")
    ax.annotate("simulator level", (0.98, SIM_LEVEL), xycoords=("axes fraction", "data"),
                xytext=(0, -2), textcoords="offset points", ha="right", va="top")
    ax.plot([], [], linestyle="", marker="o", color="0.6", markerfacecolor="none",
            label="Same method, global shutter")
    ax.set_xlabel("Target speed (m/s)")
    ax.set_ylabel("Residual error, row-time stamp (px)")
    ax.set_yscale("log")


def summary_table(df: pd.DataFrame, out: Output) -> pd.DataFrame:
    """X (мм, привязка ко времени кадра) и Y (пикс., построчная) по t_row."""
    rows = []
    rolling = df[df["t_row_factor"] > 0]
    glob = df[df["t_row_factor"] == 0]
    for (factor, speed), d in rolling.groupby(["t_row_factor", "speed"]):
        c = d[d["method"] == "centroid"]
        row = {
            "t_row_factor": factor,
            "t_row_us": float(d["t_row"].iloc[0]) * 1e6,
            "speed": speed,
            "frame_err_max_mm": float(c["frame_err_mm"].max()),
            "frame_err_max_px": float(c["frame_err_px"].max()),
        }
        for m, dm in d.groupby("method"):
            row[f"resid_max_{m}_px"] = float(dm["resid_px"].max())
            g = glob[(glob["speed"] == speed) & (glob["method"] == m)]
            if not g.empty:
                row[f"global_max_{m}_px"] = float(g["resid_px"].max())
        rows.append(row)
    table = pd.DataFrame(rows)
    out.table(table, "e6_summary")
    return table


def desync_table(df: pd.DataFrame, t_row0: float, out: Output) -> None:
    f_px = float(df["f_px"].iloc[0])
    rows = []
    for d_rows, factor, speed in itertools.product(
        DESYNC_ROWS, sorted(df.loc[df["t_row_factor"] > 0, "t_row_factor"].unique()),
        SPEEDS,
    ):
        dt = d_rows * factor * t_row0
        rows.append(
            {
                "delta_rows": d_rows,
                "t_row_factor": factor,
                "speed": speed,
                "delta_t_us": dt * 1e6,
                "error_mm": speed * dt * 1e3,
                "error_px": f_px * speed * dt / Z,
            }
        )
    out.table(pd.DataFrame(rows), "e6_desync")


def check(df: pd.DataFrame, summary: pd.DataFrame, out: Output) -> None:
    rolling = df[(df["t_row_factor"] > 0) & (df["method"] == "centroid")]
    mm = rolling["um_per_px"] / 1e3
    tol = (SIM_LEVEL + 3 * rolling["resid_se_px"]) * mm
    dev = (rolling["frame_err_mm"] - rolling["frame_err_theory_mm"]).abs()
    out.check(
        bool((dev <= tol).all()),
        f"centroid: ошибка привязки ко времени кадра = скорость * t_row * v*, "
        f"v* = v_ref / (1 - nu_v t_row): отклонение до {dev.max() * 1e3:.2f} мкм",
    )
    lim = SIM_LEVEL + 3 * rolling["resid_se_px"]
    worst = rolling.loc[(rolling["resid_px"] - lim).idxmax()]
    out.check(
        bool((rolling["resid_px"] <= lim).all()),
        f"centroid: при построчной привязке остаток до {rolling['resid_px'].max():.1e} "
        f"пикс. (без шума {rolling['resid0_px'].max():.1e}); хуже всего "
        f"{worst['speed']:g} м/с, {worst['direction']}, {worst['row_position']}",
    )
    est = summary[np.isclose(summary["t_row_factor"], 1.0)].set_index("speed")
    band = summary.groupby("speed")["frame_err_max_mm"].agg(["min", "max"])
    for speed in est.index:
        print(
            f"  {speed:g} м/с: без построчной привязки до {est.loc[speed, 'frame_err_max_mm']:.2f} "
            f"мм на объекте (t_row ± 50 %: {band.loc[speed, 'min']:.2f}–"
            f"{band.loc[speed, 'max']:.2f} мм); с привязкой центр тяжести "
            f"{est.loc[speed, 'resid_max_centroid_px']:.1e} пикс."
        )
    if TEXT_SPEED in est.index:
        x = est.loc[TEXT_SPEED, "frame_err_max_mm"]
        y = summary.loc[summary["t_row_factor"] > 0, "resid_max_centroid_px"].max()
        print(
            f"  аннотация: построчный затвор бюджетной камеры даёт до {x:.1f} мм на "
            f"объекте при {TEXT_SPEED:g} м/с ({est['frame_err_max_mm'].max():.1f} мм при "
            f"{est.index.max():g} м/с), если не привязывать время к строке; при "
            f"построчной привязке остаётся {y:.1e} пикс. (центр тяжести)"
        )


if __name__ == "__main__":
    sys.exit(main("E6: построчный затвор", run, plot))
