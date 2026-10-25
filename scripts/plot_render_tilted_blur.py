"""Первая панель экспертизы: наклонённая смазанная марка.

(а) облучённость в ROI с отмеченными центрами;
(б) те же центры крупно, в долях пикселя относительно центра тяжести.

Эталоны:
* ``exposure_conic_center`` — точный центр тяжести смазанного изображения
  (среднее центров эллипсов, взвешенное площадями);
* ``conic_center`` в середине экспозиции — центр эллипса без смаза;
* проекция центра марки в середине экспозиции — без эксцентриситета.

Запуск из корня проекта::

    python scripts/plot_render_tilted_blur.py [папка_вывода]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from photosim.camera import Camera, Pose
from photosim.oracle import conic_center, exposure_conic_center
from photosim.render import render
from photosim.scene import FixedOrientation, Linear, Motion
from photosim.target import Target

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e4e3dc"


def main(out_dir: Path) -> None:
    cam = Camera.from_physical(16, 3.45, 2448, 2048)
    pose = Pose(np.eye(3), np.zeros(3))
    radius = 0.005
    tilt = math.radians(45)
    normal = [-math.sin(tilt), 0.0, -math.cos(tilt)]
    X0 = [0.2, 0.1, 1.0]
    V = [0.15, -0.06, 0.05]  # м/с; за τ = 10 мс ≈ 7 пикс. смаза
    t_k, tau, ss, nt = 0.0, 0.01, 16, 64
    motion = Motion(Linear(X0, V), FixedOrientation.from_normal(normal))

    res = render(cam, pose, Target(radius), motion, t_k, tau, ss=ss, nt=nt)
    I = res.irradiance
    u0, v0, w, h = res.roi
    s = I.sum()
    c_img = np.array(
        [
            (I.sum(axis=0) * (u0 + np.arange(w))).sum() / s,
            (I.sum(axis=1) * (v0 + np.arange(h))).sum() / s,
        ]
    )
    c_exp = exposure_conic_center(cam, pose, motion, radius, t_k, tau, nt)
    t_mid = t_k + tau / 2
    c_mid = conic_center(cam, pose, motion.position(t_mid), motion.orientation(t_mid), radius)
    p_mid = cam.project(pose.to_camera(motion.position(t_mid)))

    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), constrained_layout=True, width_ratios=[1.15, 1]
    )

    extent = (u0 - 0.5, u0 + w - 0.5, v0 + h - 0.5, v0 - 0.5)  # центры пикселей целые
    im = ax1.imshow(I, cmap="gray", extent=extent, interpolation="nearest", vmin=0, vmax=1)
    ax1.plot(*c_img, "+", color=ORANGE, ms=14, mew=2, label="центр тяжести изображения")
    ax1.plot(*c_exp, "x", color=BLUE, ms=10, mew=2, label="exposure_conic_center")
    ax1.set_xlabel("u, пикс.")
    ax1.set_ylabel("v, пикс.")
    ax1.set_title(
        f"(а) Облучённость, ROI {w}×{h}, ss={ss}, nt={nt}", loc="left", color=INK
    )
    fig.colorbar(im, ax=ax1, shrink=0.85, label="отн. ед.")

    pts = {
        "центр тяжести изображения": (c_img, "+", ORANGE, 14),
        "exposure_conic_center (эталон)": (c_exp, "x", BLUE, 11),
        "conic_center, середина экспозиции": (c_mid, "o", AQUA, 8),
        "проекция центра марки, середина": (p_mid, "s", MUTED, 7),
    }
    for label, (p, mk, col, ms) in pts.items():
        d = p - c_img
        kw = {"mfc": "white", "mec": col, "mew": 1.8} if mk in "os" else {"mew": 2}
        ax2.plot(d[0], d[1], mk, color=col, ms=ms, label=label, ls="none", **kw)
    span = max(0.05, 1.3 * max(np.abs(p - c_img).max() for p, *_ in pts.values()))
    ax2.set_xlim(-span, span)
    ax2.set_ylim(span, -span)  # v вниз, как на изображении
    ax2.set_aspect("equal")
    ax2.grid(True, color=GRID, lw=0.8)
    ax2.axhline(0, color=MUTED, lw=0.6)
    ax2.axvline(0, color=MUTED, lw=0.6)
    ax2.set_xlabel("Δu от центра тяжести, пикс.")
    ax2.set_ylabel("Δv от центра тяжести, пикс.")
    ax2.set_title("(б) Центры крупно", loc="left", color=INK)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=False)

    err = np.linalg.norm(c_img - c_exp)
    ecc = np.linalg.norm(c_mid - p_mid)
    smear = np.linalg.norm(c_exp - c_mid)
    ax2.text(
        0.98,
        0.98,
        f"|рендер − эталон| = {err:.1e} пикс.\n"
        f"эксцентриситет (середина) = {ecc:.3f} пикс.\n"
        f"эталон − conic_center(середина) = {smear:.1e} пикс.",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=INK,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"render_tilted_blur.{ext}", dpi=200)
    print(f"ROI {res.roi}")
    for label, (p, *_rest) in pts.items():
        print(f"{label:38s} u = {p[0]:.5f}  v = {p[1]:.5f}")
    print(f"|рендер − эталон| = {err:.2e} пикс.")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures"))
