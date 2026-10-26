"""E3. Линейный смаз.

Проверка δt = v τ / 2 по изображению и влияние смаза на смещение и разброс
каждого метода; калибровка модели Sigma_xi для статьи B.

Камера review_16mm, Z = 1 м, марка r = 5 мм (R ≈ 23 пикс.) у центра кадра
(фаза 0,3 пикс. по обеим осям, как в E1), все шумы и PSF (в рендере, до
пикселя) включены, fill = 0,7, n_mc = 500. Перебор: длина смаза
L = f v τ / Z = 0; 0,5; 1; 2; 4; 8; 12; 16; 20 пикс. — через скорость при
τ = 1 мс, сигнал не меняется; направление движения вдоль u и под 45°;
наклон 0° (плоскость || сенсору) и 45° вокруг y камеры (смаз вместе с
эксцентриситетом): 9 × 2 × 2 = 36 точек, ss = 16, nt = 64. Эталон —
exposure_conic; смещение относительно proj_start пересчитывается по
колонкам ref.*. Оси результата поворачиваются на направление движения
(analysis.along_across, std_along_across по corr_uv).

    python scripts/e3_blur.py run [--quick] [--jobs 6] [--e0 runs/<...>_e0]
    python scripts/e3_blur.py plot runs/<...>_e3 [--e0 runs/<...>_e0]

nt = 64 хватает при любом L: шаг подшагов L / 64 <= 0,31 пикс., гауссова
PSF 0,7 пикс. гасит «лесенку» с таким периодом в exp(-2 pi**2 sigma**2 /
step**2) ~ e**-99 раз, а эталон exposure_conic взят по тем же подшагам.

Теория СКО центра тяжести (как в E1, analysis.centroid_sigma): второй
момент образа вдоль направления — точный момент эллипса (1/4) A**-1 по
коническому сечению в середине экспозиции, плюс sigma_psf**2 + 1/12, плюс
L**2 / 12 вдоль движения (равномерный смаз); фон — с обрезкой
(CLIP_VARIANCE). Отсюда модель для статьи B:
s_along(L)**2 = s0**2 + k**2 L**2, k**2 = 1 / (12 N_e), поперёк k = 0;
то же в виде s(L) = s0 sqrt(1 + alpha (L / R)**2), alpha = (k R / s0)**2
(1/3 для круга при R >> sigma_psf).

δt = L/2 точно только для плоскости, параллельной сенсору. Площадь образа
наклонной марки меняется по пути (меняется угол зрения), центр тяжести
смещается к большей на (L**2 / 12) d ln A / ds: 7e-3 пикс. при L = 20,
наклоне 45° (1e-4 при L = 2,3 пикс., пример обзора). exposure_conic это
учитывает, поэтому проверка δt сравнивает с ним, а член выводится отдельно.

Рисунок e3_blur (наклон 45°, движение вдоль u — там нелинейные методы
смещаются со смазом; при наклоне 0° нет, образ центрально-симметричен):
(а) смещение вдоль движения относительно exposure_conic от L, все методы;
(б) СКО вдоль и поперёк движения. e3_blur_cases — смещение вдоль и поперёк
для всех четырёх сочетаний наклона и направления.

Таблицы: e3_table.csv — все точки и методы; e3_delta_t_table.csv —
отношение измеренного смещения от proj_start (за вычетом эксцентриситета)
к L/2, предсказание по exposure_conic и член веса площади;
e3_threshold_table.csv — L, начиная с которого |смещение вдоль движения|
метода больше max(его уровень из E0, уровень симулятора) (нужен --e0);
e3_fit_table.csv — модель СКО s_along(L), s_across(L) (формула — в
заголовке таблицы и в table_fit; у эллипса она не подходит, для него —
таблица по L, узлы в e3_std_knots_table.csv).

Скачок поперёк движения у эллипса и порога 50 % при наклоне 45°, L = 16
(~0,01 пикс., без шума тоже) — не ошибка рендера, а ограничение методов:
строки, где хорда эллипса короче L / 2, не доходят до половинного уровня,
и контур половинного уровня смазанного образа срезан плоскими концами
(при L = 16 — на 0,9 пикс. ниже полуоси). Где эти концы ложатся на строки
пикселей, зависит от L и фазы центра по v: при L = 16 смещение
антисимметрично по фазе (+0,011 при 0,3, -0,011 при 0,7, 0 при 0 и 0,5),
при ss = 32 то же; при L = 12 и 20 такие же значения на других фазах.
Разбор — scripts/inspect_point.py <e3> 25 (панель (d)).
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

from photosim.analysis import (
    CLIP_VARIANCE,
    along_across,
    background_variance,
    centroid_sigma,
    fit_variance_quadratic,
    interp_log,
    method_own_bias,
    point_at_pixel,
    roi_s2,
    std_along_across,
)
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

NAME = "e3"
CAMERA = "review_16mm"
Z = 1.0
U0, V0, PHASE = 1224, 1024, 0.3
RADIUS = 0.005
TAU = 1e-3
LENGTHS_PX = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0)
DIRECTIONS_DEG = (0.0, 45.0)
TILTS_DEG = (0.0, 45.0)
N_MC = 500
# рисунок e3_blur: на наклонной марке нелинейные методы смещаются со смазом,
# при наклоне 0° — нет (e3_blur_cases)
MAIN = {"tilt_deg": 45.0, "direction_deg": 0.0}
# Уровень симулятора при ss = 16: разброс центра тяжести по положениям —
# медиана 5,5e-4, максимум 1,6e-3 пикс. (docs/conventions.md)
SIM_LEVEL = 2e-3
THEORY_RTOL = 0.15
# Пример обзора: δt = v τ / 2 при 0,5 м/с и 1 мс
REVIEW_SPEED = 0.5
# e3_fit: модель СКО от длины смаза (table_fit). Квадратичная не подходит
# эллипсу: его СКО растёт со смазом в ~10 раз (alpha ~ 100), остатки
# систематические, от -4 % через +6 % до -7 % по L. У остальных методов
# остатки знакопеременные, на уровне шума оценки СКО 1 / sqrt(2 (n_mc - 1))
# = 3,2 % при n_mc = 500, поэтому правило FIT_RTOL — только для эллипса.
FIT_MODEL = "s(L)^2 = s0^2 + k^2 L^2 = s0^2 (1 + alpha (L/R)^2), alpha = (k R/s0)^2"
INTERP_MODEL = "s(L) = exp(np.interp(L, L_i, ln s_i)), L_i = knots"
FIT_RTOL = 0.03
INTERP_METHODS = ("ellipse",)


def velocity(length_px: float, direction_deg: float, f_px: float) -> tuple:
    """Скорость в мире, при которой смаз за ``TAU`` — ``length_px`` пикс."""
    speed = length_px * Z / (f_px * TAU)
    a = math.radians(direction_deg)
    return (speed * math.cos(a), speed * math.sin(a), 0.0)


def second_moments(config: Config) -> dict[str, float]:
    """Точный второй момент эллипса-образа в середине экспозиции, пикс.²:
    ``(1/4) A**-1``, где ``(x - c)^T A (x - c) = 1`` — его граница."""
    s = build(config)
    C = _image_conic(
        s.camera,
        s.pose,
        s.motion.position(s.t_ref),
        s.motion.orientation(s.t_ref),
        s.target.radius,
    )
    if C[0, 0] < 0:
        C = -C
    A, b = C[:2, :2], C[:2, 2]
    k = C[2, 2] + b @ np.linalg.solve(A, -b)
    M = np.linalg.inv(A / -k) / 4
    return {"m_uu": M[0, 0], "m_vv": M[1, 1], "m_uv": M[0, 1]}


def run(args: argparse.Namespace) -> Path:
    cam = camera(CAMERA)
    if args.quick:
        lengths, directions, tilts, n_mc = (0.0, 4.0, 20.0), DIRECTIONS_DEG, (0.0,), 50
    else:
        lengths, directions, tilts, n_mc = LENGTHS_PX, DIRECTIONS_DEG, TILTS_DEG, N_MC
    base = Config(
        camera=CAMERA,
        radius=RADIUS,
        position=point_at_pixel(cam, U0 + PHASE, V0 + PHASE, Z),
        tau=TAU,
        effects=Effects(rotation=False, rolling_shutter=False),
        n_mc=n_mc,
        reference="exposure_conic",
        ss=16,
        nt=64,
    )
    configs, labels = [], []
    for tilt, direction, length in itertools.product(tilts, directions, lengths):
        v = velocity(length, direction, cam.fx)
        cfg = replace(base, tilt_deg=tilt, motion={"type": "linear", "velocity": v})
        configs.append(cfg)
        labels.append(
            {"L_px": length, "direction_deg": direction, **second_moments(cfg)}
        )
    df = run_configs(configs, labels=labels, seed=base.seed, n_jobs=args.jobs)
    configs_json = {
        "experiment": NAME,
        "quick": args.quick,
        "base": base,
        "lengths_px": list(lengths),
        "directions_deg": list(directions),
        "tilts_deg": list(tilts),
    }
    name = NAME + ("_quick" if args.quick else "")
    return save_run(df, configs_json, name, args.root)


# Обработка


def ref(df: pd.DataFrame, name: str) -> tuple[pd.Series, pd.Series]:
    return df[f"ref.{name}.u"], df[f"ref.{name}.v"]


def with_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Оси движения, смещение от proj_start, δt и теория СКО (см. модуль)."""
    df = df[df["status"] == "ok"].copy()
    phi = np.radians(df["direction_deg"])
    c, s = np.cos(phi), np.sin(phi)
    df["bias_along"], df["bias_across"] = along_across(
        df["bias_u_px"], df["bias_v_px"], phi
    )
    df["err0_along"], df["err0_across"] = along_across(
        df["err0_u_px"], df["err0_v_px"], phi
    )
    df["se_along"], df["se_across"] = std_along_across(
        df["se_bias_u_px"], df["se_bias_v_px"], df["corr_uv"], phi
    )
    df["std_along"], df["std_across"] = std_along_across(
        df["std_u_px"], df["std_v_px"], df["corr_uv"], phi
    )
    # смещение от начала экспозиции: оценка - proj_start
    (eu, ev), (su, sv) = ref(df, "exposure_conic"), ref(df, "proj_start")
    start_u, start_v = df["bias_u_px"] + eu - su, df["bias_v_px"] + ev - sv
    df["start_along"], df["start_across"] = along_across(start_u, start_v, phi)
    start0_u, start0_v = df["err0_u_px"] + eu - su, df["err0_v_px"] + ev - sv
    df["start0_along"], _ = along_across(start0_u, start0_v, phi)
    # эксцентриситет не входит в δt: вычитается conic_mid - proj_mid
    (cu, cv), (pu, pv) = ref(df, "conic_mid"), ref(df, "proj_mid")
    df["ecc_along"], _ = along_across(cu - pu, cv - pv, phi)
    # точное предсказание: exposure_conic - proj_start; отличие от L/2 —
    # вес площади: площадь образа наклонной марки меняется по пути, и
    # центр тяжести смещается к большей, на (L**2 / 12) d ln A / ds
    df["exact_along"], _ = along_across(eu - su, ev - sv, phi)
    half = df["L_px"] / 2
    moving = half > 0

    def ratio(x: pd.Series) -> np.ndarray:
        return np.where(moving, (x - df["ecc_along"]) / half.where(moving, 1), np.nan)

    df["delta_t_ratio"] = ratio(df["start_along"])
    df["delta_t_ratio0"] = ratio(df["start0_along"])
    df["delta_t_ratio_exact"] = ratio(df["exact_along"])

    # теория СКО центра тяжести
    m_al = df["m_uu"] * c**2 + df["m_vv"] * s**2 + 2 * df["m_uv"] * c * s
    m_ac = df["m_uu"] * s**2 + df["m_vv"] * c**2 - 2 * df["m_uv"] * c * s
    extra = df["sensor.psf_sigma_px"] ** 2 + 1 / 12
    var_bg = CLIP_VARIANCE * background_variance(
        df["sensor.read_noise_e"], df["sensor.gain_e_per_dn"]
    )
    uc, vc = eu, ev
    s2u = roi_s2(df["roi.u0"], df["roi.w"], df["roi.h"], uc)
    s2v = roi_s2(df["roi.v0"], df["roi.h"], df["roi.w"], vc)
    mid_u = df["roi.u0"] + (df["roi.w"] - 1) / 2
    mid_v = df["roi.v0"] + (df["roi.h"] - 1) / 2
    suv = df["roi.w"] * df["roi.h"] * (mid_u - uc) * (mid_v - vc)
    s2_al = s2u * c**2 + s2v * s**2 + 2 * suv * c * s
    s2_ac = s2u * s**2 + s2v * c**2 - 2 * suv * c * s
    n_e = df["signal_e"]
    df["theory_along"] = centroid_sigma(
        0.0, n_e, var_bg, s2_al, m_al + extra + df["L_px"] ** 2 / 12
    )
    df["theory_across"] = centroid_sigma(0.0, n_e, var_bg, s2_ac, m_ac + extra)
    df["k_theory"] = 1 / np.sqrt(12 * n_e)
    return df


def case(df: pd.DataFrame, tilt: float, direction: float) -> pd.DataFrame:
    return df[
        np.isclose(df["tilt_deg"], tilt) & np.isclose(df["direction_deg"], direction)
    ]


def case_title(tilt: float, direction: float) -> str:
    where = "along $u$" if direction == 0 else f"at {direction:g}° to $u$"
    return f"tilt {tilt:g}°, motion {where}"


# Рисунки


def plot_bias(ax, d: pd.DataFrame, methods: list[str], column: str) -> None:
    for m in methods:
        dm = d[d["method"] == m].sort_values("L_px")
        se = "se_" + column.removeprefix("bias_")
        ax.errorbar(
            dm["L_px"], dm[column], yerr=3 * dm[se], capsize=1.5,
            elinewidth=0.5, **style.method_style(m),
        )
    ax.axhspan(-SIM_LEVEL, SIM_LEVEL, color="0.92", zorder=0, linewidth=0)
    ax.axhline(0.0, color="0.5", linewidth=0.5)


def plot_std(ax, d: pd.DataFrame, methods: list[str], axis: str) -> None:
    """СКО вдоль (``axis="along"``) или поперёк движения от L и теория центра тяжести."""
    for m in methods:
        dm = d[d["method"] == m].sort_values("L_px")
        ax.plot(dm["L_px"], dm[f"std_{axis}"], **style.method_style(m))
    c = d[d["method"] == "centroid"].sort_values("L_px")
    ax.plot(c["L_px"], c[f"theory_{axis}"], color="0.0", linestyle="--",
            marker="", linewidth=1.2, label="Centroid theory")
    ax.set_ylim(bottom=0)


def main_case(df: pd.DataFrame) -> pd.DataFrame:
    """Точки рисунка e3_blur (и рис. 5 статьи): наклон 45°, движение вдоль u."""
    return case(df, MAIN["tilt_deg"], MAIN["direction_deg"])


def plot_main(df: pd.DataFrame, out: Output) -> None:
    methods = style.ordered(df["method"].unique())
    d = case(df, MAIN["tilt_deg"], MAIN["direction_deg"])
    if d.empty:
        print("нет точек основной конфигурации: e3_blur пропущен")
        return
    fig = plt.figure(figsize=(style.DOUBLE, 3.6))
    grid = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_al = fig.add_subplot(grid[0, 1])
    ax_ac = fig.add_subplot(grid[1, 1], sharex=ax_al, sharey=ax_al)
    plot_bias(ax_a, d, methods, "bias_along")
    ax_a.set_title("(a) Bias along the motion vs exposure-averaged ellipse")
    ax_a.set_ylabel("Bias along the motion (px)")
    ax_a.annotate(
        f"grey band: simulator level ±{SIM_LEVEL:g} px; bars: ±3 s.e.",
        (0.02, 0.03), xycoords="axes fraction",
    )

    for ax, axis in ((ax_al, "along"), (ax_ac, "across")):
        plot_std(ax, d, methods, axis)
        ax.set_ylabel("Std (px)")
    ax_al.set_title("(b) Standard deviation along the motion")
    ax_ac.set_title("Standard deviation across the motion")
    ax_al.tick_params(labelbottom=False)
    for ax in (ax_a, ax_ac):
        ax.set_xlabel("Blur length $L = f v \\tau / Z$ (px)")
    ax_a.set_xlim(-0.5, max(d["L_px"]) + 0.5)
    ax_al.set_xlim(-0.5, max(d["L_px"]) + 0.5)
    handles, labels = ax_al.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    fig.suptitle(
        f"r = {RADIUS * 1e3:g} mm (R = {d['radius_px'].iloc[0]:.1f} px), "
        f"Z = {Z:g} m, $\\tau$ = {TAU * 1e3:g} ms, fill 0.7, "
        f"{case_title(MAIN['tilt_deg'], MAIN['direction_deg'])}"
    )
    out.figure(fig, "e3_blur")


def plot_cases(df: pd.DataFrame, out: Output) -> None:
    methods = style.ordered(df["method"].unique())
    tilts = sorted(df["tilt_deg"].unique())
    directions = sorted(df["direction_deg"].unique())
    pairs = list(itertools.product(tilts, directions))
    fig, axes = plt.subplots(
        2, len(pairs), figsize=(style.DOUBLE, 4.2), sharex=True, sharey="row",
        squeeze=False,
    )
    for j, (tilt, direction) in enumerate(pairs):
        d = case(df, tilt, direction)
        for i, column in enumerate(("bias_along", "bias_across")):
            plot_bias(axes[i][j], d, methods, column)
        axes[0][j].set_title(case_title(tilt, direction))
        axes[1][j].set_xlabel("Blur length $L$ (px)")
    axes[0][0].set_ylabel("Bias along the motion (px)")
    axes[1][0].set_ylabel("Bias across the motion (px)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(methods))
    out.figure(fig, "e3_blur_cases")


# Таблицы


TABLE_COLUMNS = [
    "tilt_deg", "direction_deg", "L_px", "method", "bias_along", "bias_across",
    "se_along", "se_across", "err0_along", "err0_across", "start_along",
    "delta_t_ratio", "std_along", "std_across", "theory_along", "theory_across",
    "um_per_px", "ok_frac",
]


def table_delta_t(df: pd.DataFrame, out: Output) -> pd.DataFrame:
    d = df[(df["method"] == "centroid") & (df["L_px"] > 0)]
    table = d[["tilt_deg", "direction_deg", "L_px"]].copy()
    table["half_L_um"] = d["L_px"] / 2 * d["um_per_px"]
    table["measured_um"] = (d["start_along"] - d["ecc_along"]) * d["um_per_px"]
    table["ratio"] = d["delta_t_ratio"]
    table["ratio_se"] = d["se_along"] / (d["L_px"] / 2)
    table["ratio_noise_free"] = d["delta_t_ratio0"]
    # предсказание по exposure_conic и его отличие от L/2 (вес площади)
    table["ratio_exact"] = d["delta_t_ratio_exact"]
    table["area_term_px"] = (d["delta_t_ratio_exact"] - 1) * d["L_px"] / 2
    table = table.sort_values(["tilt_deg", "direction_deg", "L_px"])
    out.table(table, "e3_delta_t")
    return table


def crossing(x: np.ndarray, y: np.ndarray, level: float) -> tuple[float, str]:
    """``x``, начиная с которого ``y > level`` до конца сетки, и пометка.

    Линейно между последним узлом не выше ``level`` и следующим: отдельные
    выбросы пиксельной фазы при малых ``x`` порогом не считаются.
    """
    below = np.flatnonzero(y <= level)
    if below.size == 0:
        return float(x[0]), "all L"
    i = int(below[-1])
    if i == len(x) - 1:
        return math.nan, f"> {x[-1]:g}"
    t = (level - y[i]) / (y[i + 1] - y[i])
    return float(x[i] + t * (x[i + 1] - x[i])), ""


def table_threshold(df: pd.DataFrame, e0: Run | None, out: Output) -> pd.DataFrame:
    """L, начиная с которого смещение вдоль движения выходит за уровень метода.

    Уровень — ``max(уровень E0, SIM_LEVEL)``; уровень E0 — ``max |err0_u|``
    по фазе с PSF при том же радиусе образа. Смещение — ``|bias_along|``
    относительно exposure_conic: по кадру без шума (``err0``, как в E0) и по
    среднему с шумом (там уровень не меньше 3 se). Учитывается только
    устойчивый выход (``crossing``): без смаза у метода своё смещение
    пиксельной фазы, при фазе 0,3 по обеим осям и наклоне оно бывает больше
    уровня E0 (E0 — фаза только по u, круг) и с ростом L усредняется.
    """
    own = {}
    if e0 is not None:
        own = method_own_bias(e0.df, float(df["radius_px"].iloc[0]))
    rows = []
    groups = df.groupby(["method", "tilt_deg", "direction_deg"], sort=False)
    for (m, tilt, direction), d in groups:
        d = d.sort_values("L_px")
        x = d["L_px"].to_numpy()
        y0, y = d["err0_along"].abs().to_numpy(), d["bias_along"].abs().to_numpy()
        level_e0 = own.get(m, math.nan)
        level = max(SIM_LEVEL, level_e0) if math.isfinite(level_e0) else SIM_LEVEL
        level_mc = np.maximum(level, 3 * d["se_along"].to_numpy())
        L0, note0 = crossing(x, y0, level)
        L1, note1 = crossing(x, y / level_mc, 1.0)
        rows.append(
            {
                "method": m,
                "tilt_deg": tilt,
                "direction_deg": direction,
                "own_level_e0_px": level_e0,
                "level_px": level,
                "L_exceed_noise_free_px": L0,
                "note_noise_free": note0,
                "L_exceed_with_noise_px": L1,
                "note_with_noise": note1,
                "err0_along_at_L0_px": float(d["err0_along"].iloc[0]),
                "err0_along_at_Lmax_px": float(d["err0_along"].iloc[-1]),
                "bias_along_at_Lmax_px": float(d["bias_along"].iloc[-1]),
            }
        )
    table = pd.DataFrame(rows)
    table["method"] = pd.Categorical(table["method"], style.ordered(table["method"]))
    table = table.sort_values(["method", "tilt_deg", "direction_deg"])
    out.table(table, "e3_threshold")
    return table


def table_fit(df: pd.DataFrame, out: Output) -> pd.DataFrame:
    """Модель СКО от длины смаза L для статьи B (модель Sigma_xi), ``FIT_MODEL``:

        s(L)**2 = s0**2 + k**2 L**2 = s0**2 (1 + alpha (L / R)**2),
        alpha = (k R / s0)**2

    (``analysis.fit_variance_quadratic``), вдоль и поперёк движения. Если у
    метода из ``INTERP_METHODS`` относительная ошибка подгонки больше
    ``FIT_RTOL``, для этой оси берётся таблица по L, ``INTERP_MODEL``
    (``analysis.interp_log``; узлы — e3_std_knots_table.csv), а ``k`` и
    ``alpha`` не даются. Колонки ``model_along``, ``model_across`` — какая
    модель действует.
    """
    rows, knots = [], []
    groups = df.groupby(["method", "tilt_deg", "direction_deg"], sort=False)
    for (m, tilt, direction), d in groups:
        d = d.sort_values("L_px")
        R = float(d["radius_px"].iloc[0])
        row = {"method": m, "tilt_deg": tilt, "direction_deg": direction, "R_px": R}
        for axis in ("along", "across"):
            s0, k, rms = fit_variance_quadratic(d["L_px"], d[f"std_{axis}"])
            interp = m in INTERP_METHODS and rms > FIT_RTOL
            row[f"model_{axis}"] = "interp" if interp else "quadratic"
            row[f"s0_{axis}_px"] = s0
            row[f"k_{axis}"] = math.nan if interp else k
            row[f"alpha_{axis}"] = (
                math.nan if interp else math.copysign((k * R / s0) ** 2, k)
            )
            # СКО относительного остатка квадратичной подгонки: почему interp
            row[f"rel_rms_{axis}"] = rms
            if interp:
                row[f"loo_rms_interp_{axis}"] = loo_interp_rms(
                    d["L_px"].to_numpy(), d[f"std_{axis}"].to_numpy()
                )
        if m == "centroid":
            row["k_along_theory"] = float(d["k_theory"].iloc[0])
            row["alpha_along_theory"] = float(
                (d["k_theory"].iloc[0] * R / d["theory_along"].iloc[0]) ** 2
            )
        rows.append(row)
        if "interp" in (row["model_along"], row["model_across"]):
            knots.append(
                d[["method", "tilt_deg", "direction_deg", "L_px"]].assign(
                    s_along_px=d["std_along"], s_across_px=d["std_across"],
                    model_along=row["model_along"], model_across=row["model_across"],
                )
            )
    table = pd.DataFrame(rows)
    table["method"] = pd.Categorical(table["method"], style.ordered(table["method"]))
    table = table.sort_values(["method", "tilt_deg", "direction_deg"])
    header = (
        f"quadratic: {FIT_MODEL}\n"
        f"interp: {INTERP_MODEL} (methods {', '.join(INTERP_METHODS)} "
        f"when rel_rms > {FIT_RTOL:g})"
    )
    out.table(table, "e3_fit", header=header)
    if knots:
        out.table(pd.concat(knots, ignore_index=True), "e3_std_knots",
                  header=f"knots of {INTERP_MODEL}")
    return table


def loo_interp_rms(L: np.ndarray, s: np.ndarray) -> float:
    """СКО относительной ошибки ``interp_log`` во внутренних узлах, если
    узел выбросить из таблицы (проверка, что интерполяции хватает сетки)."""
    errs = [
        interp_log(L[i], np.delete(L, i), np.delete(s, i)) / s[i] - 1
        for i in range(1, len(L) - 1)
    ]
    return float(np.sqrt(np.mean(np.square(errs))))


def plot(run: Run, out: Output, args: argparse.Namespace) -> None:
    df = with_metrics(run.df)
    plot_main(df, out)
    plot_cases(df, out)
    full = df[TABLE_COLUMNS].copy()
    full["method"] = pd.Categorical(full["method"], style.ordered(full["method"]))
    out.table(full.sort_values(["tilt_deg", "direction_deg", "method", "L_px"]), NAME)
    dt = table_delta_t(df, out)
    e0 = load_run(args.e0) if args.e0 is not None else None
    if e0 is None:
        print("  --e0 не указан: порог e3_threshold — только уровень симулятора")
    thr = table_threshold(df, e0, out)
    fit = table_fit(df, out)
    check(df, dt, thr, fit, out)


def check(
    df: pd.DataFrame,
    dt: pd.DataFrame,
    thr: pd.DataFrame,
    fit: pd.DataFrame,
    out: Output,
) -> None:
    # δt по изображению: измеренное отношение против точного предсказания
    # (exposure_conic); допуск — 3 se и уровень симулятора, отнесённые к L/2
    d = dt[dt["L_px"] >= 1.0]
    tol = (3 * d["ratio_se"] + SIM_LEVEL / (d["L_px"] / 2)).to_numpy()
    dev = (d["ratio"] - d["ratio_exact"]).abs().to_numpy()
    worst = d.iloc[int(np.argmax(dev - tol))]
    out.check(
        bool((dev <= tol).all()),
        f"centroid: смещение от proj_start (за вычетом эксцентриситета) / (L/2), "
        f"L >= 1 пикс.: от {d['ratio'].min():.4f} до {d['ratio'].max():.4f}, "
        f"от предсказания по exposure_conic — до {dev.max():.1e}; хуже всего "
        f"L = {worst['L_px']:g}, наклон {worst['tilt_deg']:g}°, направление "
        f"{worst['direction_deg']:g}°",
    )
    area = dt.loc[dt["area_term_px"].abs().idxmax()]
    print(
        f"  δt = L/2 точно только при плоскости || сенсору: вес площади даёт до "
        f"{area['area_term_px']:+.1e} пикс. (L = {area['L_px']:g}, наклон "
        f"{area['tilt_deg']:g}°, направление {area['direction_deg']:g}°)"
    )
    review_l = REVIEW_SPEED * TAU * float(df["f_px"].iloc[0]) / Z
    print(
        f"  δt при {REVIEW_SPEED:g} м/с и {TAU * 1e3:g} мс: L = {review_l:.2f} пикс., "
        f"δt = {review_l / 2:.2f} пикс. = "
        f"{REVIEW_SPEED * TAU / 2 * 1e3:.2f} мм на объекте"
    )

    c = df[df["method"] == "centroid"]
    lim_al = SIM_LEVEL + 3 * c["se_along"]
    lim_ac = SIM_LEVEL + 3 * c["se_across"]
    out.check(
        bool(((c["bias_along"].abs() <= lim_al) & (c["bias_across"].abs() <= lim_ac)).all()),
        f"centroid: смещение от exposure_conic не больше уровня симулятора + 3 se "
        f"при всех L: вдоль до {c['bias_along'].abs().max():.1e}, поперёк до "
        f"{c['bias_across'].abs().max():.1e} пикс.",
    )
    # оценка СКО по n реализациям шумит на 1 / sqrt(2 (n - 1))
    rtol = max(THEORY_RTOL, 3 / math.sqrt(2 * (c["n_mc"].min() - 1)))
    dev = np.maximum(
        (c["std_along"] / c["theory_along"] - 1).abs(),
        (c["std_across"] / c["theory_across"] - 1).abs(),
    )
    out.check(
        bool((dev <= rtol).all()),
        f"centroid: СКО вдоль и поперёк против теории до {dev.max():.1%} "
        f"(порог {rtol:.0%})",
    )
    for _, row in fit[fit["method"] == "centroid"].iterrows():
        print(
            f"  centroid, наклон {row['tilt_deg']:g}°, направление "
            f"{row['direction_deg']:g}°: k_along = {row['k_along']:.2e} "
            f"(теория {row['k_along_theory']:.2e}), alpha = {row['alpha_along']:.2f} "
            f"(теория {row['alpha_along_theory']:.2f}), k_across = {row['k_across']:.1e}"
        )
    switched = fit[(fit["model_along"] == "interp") | (fit["model_across"] == "interp")]
    for _, row in switched.iterrows():
        print(
            f"  {row['method']}, наклон {row['tilt_deg']:g}°, направление "
            f"{row['direction_deg']:g}°: квадратичная модель — ошибка "
            f"{row['rel_rms_along']:.1%} / {row['rel_rms_across']:.1%} (вдоль / "
            f"поперёк), таблица по L; интерполяция без узла ошибается на "
            f"{row.get('loo_rms_interp_along', math.nan):.1%} / "
            f"{row.get('loo_rms_interp_across', math.nan):.1%}"
        )
    for _, row in thr.iterrows():
        note = row["note_noise_free"] or f"{row['L_exceed_noise_free_px']:.1f} пикс."
        note_mc = row["note_with_noise"] or f"{row['L_exceed_with_noise_px']:.1f} пикс."
        print(
            f"  {row['method']}, наклон {row['tilt_deg']:g}°, направление "
            f"{row['direction_deg']:g}°: уровень {row['level_px']:.1e}, "
            f"без шума выход при L {note}, с шумом {note_mc}"
        )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--e0", type=Path, default=None,
        help="папка запуска E0: собственный уровень методов для e3_threshold",
    )


if __name__ == "__main__":
    sys.exit(main("E3: линейный смаз", run, plot, add_arguments=add_arguments))
