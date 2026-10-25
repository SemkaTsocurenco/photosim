"""Единый стиль рисунков статьи: размеры, шрифт, вид методов.

Все подписи на рисунках — на английском. Ширина рисунка — ``COLUMN`` (в
одну колонку) или ``DOUBLE`` (на две), дюймы. Каждый метод оценки рисуется
стилем из ``METHOD_STYLE``: маркер, тип линии и оттенок серого у методов
разные, поэтому метод выглядит одинаково на всех рисунках и различим при
печати в оттенках серого.

    style.apply()
    fig, ax = plt.subplots(figsize=(style.COLUMN, 2.4))
    ax.plot(x, y, **style.method_style("centroid"))
    style.save(fig, "e1_std", run.path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.figure import Figure

__all__ = [
    "COLUMN",
    "DOUBLE",
    "EFFECT_LABEL",
    "FONT_SIZE",
    "METHOD_STYLE",
    "RC",
    "apply",
    "method_style",
    "save",
]

#: Ширина рисунка в одну колонку, дюймы.
COLUMN = 3.5
#: Ширина рисунка на две колонки, дюймы.
DOUBLE = 7.16
#: Размер всех надписей, pt.
FONT_SIZE = 8

#: Параметры matplotlib, которые ставит ``apply``.
RC: dict[str, Any] = {
    # STIX похож на Times и входит в matplotlib: рисунки одинаковы на любой
    # машине; Times New Roman — если STIX почему-то нет
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "figure.figsize": (COLUMN, 0.75 * COLUMN),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    # Ширина файла — ровно заданная: поля подгоняет constrained layout, а не
    # bbox="tight". Обрезанный рисунок LaTeX растянет до ширины колонки, и
    # шрифт станет крупнее 8 pt.
    "figure.constrained_layout.use": True,
    "savefig.bbox": None,
    "pdf.fonttype": 42,  # TrueType: текст в PDF остаётся текстом
    "ps.fonttype": 42,
    # тонкие линии
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.8,
    "lines.markersize": 3.0,
    "lines.markeredgewidth": 0.6,
    "patch.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    # светлая сетка под данными
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.4,
    "legend.frameon": False,
    "legend.handlelength": 2.8,  # чтобы в легенде был виден тип линии
}

#: Вид методов оценки: ``ax.plot(x, y, **method_style(name))``.
METHOD_STYLE: dict[str, dict[str, Any]] = {
    "centroid": {
        "label": "Centroid",
        "color": "0.0",
        "marker": "o",
        "markerfacecolor": "0.0",
        "linestyle": "-",
    },
    "centroid_thr50": {
        "label": "Centroid, 50% threshold",
        "color": "0.3",
        "marker": "s",
        "markerfacecolor": "white",
        "linestyle": "--",
    },
    "centroid_sq": {
        "label": "Squared-weight centroid",
        "color": "0.45",
        "marker": "^",
        "markerfacecolor": "0.45",
        "linestyle": "-.",
    },
    "gauss2d": {
        "label": "2D Gaussian fit",
        "color": "0.15",
        "marker": "D",
        "markerfacecolor": "white",
        "linestyle": ":",
    },
    "ellipse": {
        "label": "Ellipse fit",
        "color": "0.6",
        "marker": "v",
        "markerfacecolor": "0.6",
        "linestyle": (0, (5, 1.5, 1, 1.5, 1, 1.5)),  # штрих-пунктир с двумя точками
    },
}

#: Подписи эффектов для рисунков ``ablation``.
EFFECT_LABEL: dict[str, str] = {
    "perspective": "Perspective",
    "exposure": "Exposure",
    "motion": "Motion",
    "rotation": "Rotation",
    "rolling_shutter": "Rolling shutter",
    "psf": "PSF",
    "shot_noise": "Shot noise",
    "read_noise": "Read noise",
    "quantization": "Quantization",
}


def apply() -> None:
    """Установить стиль статьи в ``matplotlib.rcParams``."""
    mpl.rcParams.update(RC)


def method_style(name: str) -> dict[str, Any]:
    """Копия стиля метода; для метода не из ``METHOD_STYLE`` — нейтральный."""
    default = {"label": name, "color": "0.5", "marker": "x", "linestyle": "-"}
    return dict(METHOD_STYLE.get(name, default))


def save(fig: Figure, name: str, out_dir: str | Path) -> tuple[Path, Path]:
    """Сохранить ``out_dir/name.pdf`` и ``out_dir/name.png``; вернуть пути."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = (out / f"{name}.pdf", out / f"{name}.png")
    for path in paths:
        fig.savefig(path)
    return paths
