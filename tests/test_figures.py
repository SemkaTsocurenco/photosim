"""Тесты стиля рисунков.

``Figure`` без pyplot: бэкенд с окнами не нужен.
"""

import matplotlib as mpl
import pytest
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.image import imread

from photosim.estimators import ESTIMATORS
from photosim.experiment import EFFECTS
from photosim.figures import style


@pytest.fixture
def styled():
    with mpl.rc_context():
        style.apply()
        yield


def test_sizes():
    assert style.COLUMN == 3.5
    assert style.DOUBLE == 7.16


def test_apply(styled):
    rc = mpl.rcParams
    assert rc["font.size"] == rc["axes.labelsize"] == rc["legend.fontsize"] == 8
    assert rc["savefig.dpi"] == 600
    assert rc["axes.grid"]
    assert "STIX" in findfont(FontProperties(family="serif"), fallback_to_default=False)


def test_apply_is_scoped_by_rc_context():
    before = mpl.rcParams["savefig.dpi"]
    with mpl.rc_context():
        style.apply()
    assert mpl.rcParams["savefig.dpi"] == before


def test_every_method_has_distinct_style():
    assert set(style.METHOD_STYLE) == set(ESTIMATORS)
    styles = list(style.METHOD_STYLE.values())
    assert len({s["marker"] for s in styles}) == len(styles)
    assert len({str(s["linestyle"]) for s in styles}) == len(styles)
    assert len({s["color"] for s in styles}) == len(styles)
    for s in styles:
        r, g, b = to_rgb(s["color"])
        assert r == g == b  # оттенок серого
        assert s["label"].isascii()  # подписи на английском


def test_effect_labels():
    assert tuple(style.EFFECT_LABEL) == EFFECTS
    assert all(label.isascii() for label in style.EFFECT_LABEL.values())


def test_method_style_is_copy_and_has_fallback():
    s = style.method_style("centroid")
    s["color"] = "red"
    assert style.METHOD_STYLE["centroid"]["color"] == "0.0"
    assert style.method_style("custom")["label"] == "custom"


def test_save(styled, tmp_path):
    fig = Figure(figsize=(style.COLUMN, 2.0))
    ax = fig.subplots()
    for i, name in enumerate(style.METHOD_STYLE):
        ax.plot([0, 1, 2], [i, i + 0.5, i], **style.method_style(name))
    ax.set_xlabel("Fractional position (px)")
    ax.legend()
    pdf, png = style.save(fig, "check", tmp_path / "figs")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG")
    # ровно заданная ширина при 600 dpi: рисунок не обрезается
    assert imread(png).shape[:2] == (2.0 * 600, style.COLUMN * 600)
