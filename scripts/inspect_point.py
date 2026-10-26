"""Панель экспертизы одной точки перебора: кадр, контуры и ошибки методов.

Это диагностика, а не рисунок результата: точка заново строится по
конфигурации из строки таблицы запуска (experiment.config_from_row, номер
point) и считается без шума. Подходит для любого запуска: sweep,
run_configs, ablation. Эталон — exposure_conic (у неподвижной марки он
равен conic_mid). Панель:

(a) кадр без шума; контур половинного уровня, по которому метод ellipse
    подгоняет эллипс, и сам подогнанный эллипс; граница порога 50 %
    (centroid_thr50); точный эллипс (oracle);
(b) максимум яркости по строкам кадра вдоль образа (и он же, отражённый
    относительно точного центра — видна асимметрия) и уровни: порог 50 %
    и половинный уровень эллипса;
(c) ошибки методов без шума относительно эталона.

    python scripts/inspect_point.py runs/<...>_e2 5
    python scripts/inspect_point.py runs/<...>_e3 79

Для E2 по умолчанию берётся таблица methods (E2b), иначе results
(``--table``). Рисунок — figures/inspect_<имя запуска>_point<N>.pdf/png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # запуск без установки

import argparse
import textwrap
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import find_contours

from photosim.estimators import ESTIMATORS, ellipse_from_conic, estimate
from photosim.experiment import (
    Config,
    _render,
    build,
    config_from_row,
    load_run,
    references,
)
from photosim.figures import style
from photosim.figures.cli import FIGURES_DIR
from photosim.oracle import _image_conic


def config_of_point(df: pd.DataFrame, point: int) -> tuple[Config, dict]:
    """Конфигурация точки ``point`` и её метки (колонки между ``point`` и
    параметрами конфигурации: L_px, mode, ...) для подписи."""
    rows = df[df["point"] == point]
    if rows.empty:
        raise SystemExit(f"нет точки {point}: есть 0 ... {df['point'].max()}")
    row = rows.iloc[0]
    cfg = config_from_row(row)
    cols = list(df.columns)
    labels = {k: row[k] for k in cols[cols.index("point") + 1 : cols.index("camera")]}
    labels["tilt_deg"] = cfg.tilt_deg
    labels["position"] = cfg.position
    if cfg.motion["type"] != "static":
        labels["motion"] = cfg.motion
    return cfg, labels


#: Панель (d): глубина концов контура, пикс.
CAP_DEPTH_PX = 3.0


def short(value) -> str:
    """Значение метки для подписи: числа — 4 значащие цифры, у движения — скорость."""
    if isinstance(value, dict):
        return short(value.get("velocity", value.get("type")))
    if isinstance(value, (tuple, list, np.ndarray)):
        return "(" + ", ".join(short(v) for v in value) + ")"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def ellipse_points(center, axes, angle, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    c, s = np.cos(angle), np.sin(angle)
    x, y = axes[0] * np.cos(t), axes[1] * np.sin(t)
    return center[0] + c * x - s * y, center[1] + s * x + c * y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("point", type=int)
    parser.add_argument(
        "--table", default=None, help="таблица запуска (по умолчанию results, у E2 — methods)"
    )
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    run = load_run(args.run_dir)
    table = args.table or ("methods" if "base_b" in run.configs else "results")
    df = run.df if table == "results" else run.tables[table]
    cfg, values = config_of_point(df, args.point)

    s = build(cfg)
    res = _render(s, s.motion)
    clean = replace(
        s.sensor, psf=False, shot_noise=False, read_noise=False, quantization=False
    )
    img = clean.expose(res.irradiance, s.tau, np.random.default_rng(0))
    u0, v0, w, h = res.roi
    refs = references(s)[0]
    exact = refs["exposure_conic"]
    C = _image_conic(
        s.camera, s.pose, s.motion.position(s.t_ref),
        s.motion.orientation(s.t_ref), s.target.radius,
    )
    _, axes_ex, angle_ex = ellipse_from_conic(C)
    methods = style.ordered(cfg.methods or ESTIMATORS)
    est = {m: estimate(m, img, res.roi) for m in methods}

    print(f"точка {args.point}: {values}")
    print(f"ROI {res.roi}; точный центр {exact}; полуоси {axes_ex}")
    # контур, как в estimators.ellipse
    peak = np.percentile(img[img > 0.5 * img.max()], 95)
    contours = find_contours(img, 0.5 * peak)
    closed = [c for c in contours if len(c) > 6 and np.array_equal(c[0], c[-1])]
    touches = [
        c[:, 0].min() <= 0 or c[:, 1].min() <= 0
        or c[:, 0].max() >= h - 1 or c[:, 1].max() >= w - 1
        for c in contours
    ]
    print(
        f"контуров половинного уровня {len(contours)}, замкнутых {len(closed)}, "
        f"касаются края ROI {sum(touches)}"
    )
    for m in methods:
        e = est[m]
        line = f"  {m}: ошибка {e.uv - exact}"
        if m == "ellipse":
            line += (
                f"; полуоси {e.info['semi_major']:.3f}, {e.info['semi_minor']:.3f}"
            )
        print(line)

    style.apply()
    fig = plt.figure(figsize=(style.DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[max(w / h, 0.35), 1.4, 1.4])
    ax_img, ax_prof = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    ax_err, ax_cap = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])

    extent = (u0 - 0.5, u0 + w - 0.5, v0 + h - 0.5, v0 - 0.5)
    ax_img.imshow(img, cmap="gray", extent=extent, interpolation="nearest")
    for c in closed:
        ax_img.plot(c[:, 1] + u0, c[:, 0] + v0, color="tab:cyan", linewidth=0.8)
    for c in find_contours(img, 0.5 * img.max()):
        ax_img.plot(c[:, 1] + u0, c[:, 0] + v0, color="tab:pink", linestyle=":")
    fit = est.get("ellipse")
    if fit is not None and fit.ok:
        _, ax_fit, ang_fit = ellipse_from_conic(fit.info["conic"])
        ax_img.plot(*ellipse_points(fit.uv, ax_fit, ang_fit), color="tab:olive",
                    linestyle="--", linewidth=0.8)
    ax_img.plot(*ellipse_points(refs["conic_mid"], axes_ex, angle_ex), color="tab:red",
                linestyle="--", linewidth=0.8)
    ax_img.set_xlim(extent[:2])
    ax_img.set_ylim(extent[2:])
    ax_img.grid(False)
    ax_img.set_title("(a) Frame", loc="left")
    ax_img.set_xlabel("$u$ (px)")
    ax_img.set_ylabel("$v$ (px)")

    rows = v0 + np.arange(h)
    ax_prof.plot(img.max(axis=1), rows, color="0.0", marker="o", markersize=2,
                 label="Row maximum")
    # отражение относительно точного центра: асимметрия верх/низ сразу видна
    ax_prof.plot(img.max(axis=1), 2 * exact[1] - rows, color="0.6",
                 linestyle="--", label="Mirrored about exact center")
    ax_prof.axvline(0.5 * img.max(), color="tab:pink", linestyle=":",
                    label="50% threshold")
    ax_prof.axvline(0.5 * peak, color="tab:cyan", label="Half level (ellipse)")
    ax_prof.set_ylim(extent[2:])
    ax_prof.set_title("(b) Brightest pixel of each row", loc="left")
    ax_prof.set_xlabel("DN")
    ax_prof.set_ylabel("$v$ (px)")
    ax_prof.legend(loc="center left")

    for m in methods:
        d = est[m].uv - exact
        st = style.method_style(m)
        st["linestyle"] = ""
        ax_err.plot(d[0], d[1], **st)
    ax_err.axhline(0.0, color="0.6", linewidth=0.5)
    ax_err.axvline(0.0, color="0.6", linewidth=0.5)
    lim = 1.2 * max(np.abs(est[m].uv - exact).max() for m in methods)
    ax_err.set_xlim(-lim, lim)
    ax_err.set_ylim(lim, -lim)  # v вниз, как в кадре
    ax_err.set_aspect("equal")
    ax_err.set_title("(c) Noise-free error", loc="left")
    ax_err.set_xlabel("$\\Delta u$ (px)")
    ax_err.set_ylabel("$\\Delta v$ (px)")
    ax_err.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)

    # (d) концы контура половинного уровня по v: низ отражён относительно
    # точного центра. У симметричного образа кривые совпадают; строки
    # пикселей у верха и низа лежат на разных фазах (тонкие линии)
    if closed:
        c = max(closed, key=len)
        cu, cv = c[:, 1] + u0 - exact[0], c[:, 0] + v0 - exact[1]
        depth = min(CAP_DEPTH_PX, 0.5 * float(cv.max() - cv.min()))
        for part, sign, ls, text in (
            (cv <= cv.min() + depth, -1, "-", "Top"),
            (cv >= cv.max() - depth, 1, "--", "Bottom, mirrored"),
        ):
            order = np.argsort(cu[part])
            ax_cap.plot(cu[part][order], sign * cv[part][order], color="0.0",
                        linestyle=ls, marker="", label=text)
            # центры строк пикселей в той же системе: sign * (v - v_c)
            dist = sign * (rows - exact[1])
            for y in dist[(dist > -cv.min() - depth - 1) & (dist < -cv.min() + 1)]:
                ax_cap.axhline(y, color="0.75", linewidth=0.4, linestyle=ls)
        ax_cap.set_title("(d) Ends of the half-level contour", loc="left")
        ax_cap.set_xlabel("$u - u_c$ (px)")
        ax_cap.set_ylabel("$|v - v_c|$ (px)")
        ax_cap.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    else:
        ax_cap.set_axis_off()

    label = ", ".join(f"{k} = {short(v)}" for k, v in values.items())
    fig.suptitle(
        textwrap.fill(
            f"Point {args.point}: {label}; semi-axes {axes_ex[0]:.1f} × "
            f"{axes_ex[1]:.2f} px",
            140,
        ),
        fontsize=7,
    )
    name = f"inspect_{run.path.name}_point{args.point}"
    for path in style.save(fig, name, args.out):
        print(f"рисунок: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
