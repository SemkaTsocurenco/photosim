"""Область применимости формулы δa = a·τ²/24 для движения по дуге.

Строит рисунок для статьи A (тест 20 из ТЗ на scene.py):

* (а) смещение среднего положения за экспозицию от положения в середине
  экспозиции, в долях радиуса: точное ``1 − sin(h)/h``, формула обзора
  ``(ωτ)²/24`` и численное ``exposure_mean_position``;
* (б) относительная ошибка формулы ``δ_формула/δ_точн − 1``; на всём
  диапазоне она совпадает с асимптотикой ``(ωτ)²/80`` в пределах толщины
  линии, поэтому асимптотика отдельно не рисуется.

Запуск из корня проекта::

    python scripts/plot_delta_a_validity.py [папка_вывода]

Сохраняет ``delta_a_validity.png`` и ``delta_a_validity.pdf``
(по умолчанию в ``figures/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from photosim.scene import Arc, exposure_mean_position

BLUE = "#2a78d6"  # точное решение
ORANGE = "#eb6834"  # формула обзора / численный расчёт
INK = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e4e3dc"


def compute(arc: Arc, omega_tau: np.ndarray, t_k: float = 0.37, nt: int = 4000):
    tau = omega_tau / arc.omega
    exact = np.array([arc.exposure_offset_exact(ta) for ta in tau])
    formula = arc.normal_acceleration * tau**2 / 24
    numeric = np.array(
        [
            np.linalg.norm(
                arc.position(t_k + ta / 2) - exposure_mean_position(arc, t_k, ta, nt)
            )
            for ta in tau
        ]
    )
    return exact, formula, numeric


def main(out_dir: Path) -> None:
    arc = Arc([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.4, 0.0, 2.0], 3.0)
    r = arc.radius

    wt = np.geomspace(1e-2, 2.0, 300)
    exact, formula, _ = compute(arc, wt)
    wt_pts = np.geomspace(1e-2, 2.0, 12)
    exact_pts, formula_pts, numeric_pts = compute(arc, wt_pts)

    rel = formula / exact - 1
    rel_pts_num = formula_pts / numeric_pts - 1
    # граница 1 % по точному решению
    wt_1pct = float(np.interp(0.01, rel, wt))

    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    # (а) смещение в долях радиуса
    ax1.loglog(wt, exact / r, color=BLUE, lw=2, label="точно: 1 − sin(h)/h, h = ωτ/2")
    ax1.loglog(wt, formula / r, color=ORANGE, lw=2, ls="--", label="формула: (ωτ)²/24")
    ax1.loglog(
        wt_pts,
        numeric_pts / r,
        ls="none",
        marker="o",
        ms=6,
        mfc="white",
        mec=BLUE,
        mew=1.5,
        label="численно: exposure_mean_position",
    )
    ax1.set_xlabel("ωτ, рад")
    ax1.set_ylabel("δ / r")
    ax1.set_title("(а) Смещение среднего за экспозицию", loc="left", color=INK)
    ax1.grid(True, which="major", color=GRID, lw=0.8)
    ax1.legend(frameon=False, fontsize=9)

    # (б) относительная ошибка формулы
    ax2.loglog(wt, rel * 100, color=BLUE, lw=2, label="δ_формула / δ_точн − 1")
    ax2.loglog(
        wt_pts,
        rel_pts_num * 100,
        ls="none",
        marker="o",
        ms=6,
        mfc="white",
        mec=ORANGE,
        mew=1.5,
        label="по численному среднему",
    )
    ax2.axhline(1.0, color=MUTED, lw=0.8)
    ax2.axvline(wt_1pct, color=MUTED, lw=0.8)
    ax2.annotate(
        f"1 % при ωτ ≈ {wt_1pct:.2f} рад",
        xy=(wt_1pct, 1.0),
        xytext=(0.17, 0.004),
        color=INK,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    ax2.set_xlabel("ωτ, рад")
    ax2.set_ylabel("относительная ошибка формулы, %")
    ax2.set_title("(б) Ошибка формулы δa = aτ²/24", loc="left", color=INK)
    ax2.grid(True, which="major", color=GRID, lw=0.8)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"delta_a_validity.{ext}", dpi=200)
    print(f"граница 1 %: ωτ ≈ {wt_1pct:.3f} рад")
    for w, e, f in zip(wt_pts, exact_pts, formula_pts, strict=True):
        print(f"ωτ = {w:7.4f}   ошибка формулы {100 * (f / e - 1):9.5f} %")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures"))
