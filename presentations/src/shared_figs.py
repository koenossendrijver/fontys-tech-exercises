"""Cross-deck shared figures for the M2 course decks.

Frame C - the FIT | CHOOSE | REPORT verbs bar (Harvard/ISLR motif) - is drawn
here ONCE and reused by decks 2, 3 and 4 so the three renderings are the same
diagram: pale FIT band, verb labels above the bar, hatched CHOOSE sub-band
INSIDE the FIT band with an arrow-callout below, dark REPORT block carrying
"sealed until the end", 712:179 proportions. Decks may only add a small tag
(e.g. a TODAY badge); colors, proportions, labels and motif are fixed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import deck_style as ds


def draw_frame_c(ax, tag=None):
    """Canonical Frame C rendering onto an existing axis.

    tag: None for the plain bar; "TODAY" for a badge pinned on the hatched
    CHOOSE band (deck 3); any other string for one small caption line below
    the CHOOSE label (deck 2). Everything else is identical everywhere.
    """
    pal = ds.mpl_theme()
    ax.barh(0, 712, left=0, height=0.62, color=pal["sky"],
            edgecolor=pal["blue"], lw=1.2)
    ax.barh(0, 179, left=712, height=0.62, color=pal["navy"],
            edgecolor=pal["navy"], lw=1.2)
    # CHOOSE: hatched sub-band inside the FIT segment
    ax.barh(0, 660, left=26, height=0.26, color="none",
            edgecolor=pal["blue"], lw=1.0, hatch="///")
    ax.text(0, 0.52, "FIT (train: 712 rows)", ha="left", va="bottom",
            fontsize=11, fontweight="bold", color=pal["navy"])
    ax.text(430, -0.62, "CHOOSE (validation: 5-fold CV inside the 712)",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=pal["blue"])
    ax.annotate("", xy=(430, -0.14), xytext=(430, -0.58),
                arrowprops=dict(arrowstyle="-|>", color=pal["blue"], lw=1.2))
    ax.text(891, 0.52, "REPORT (test: 179 rows, opened once)", ha="right",
            va="bottom", fontsize=11, fontweight="bold", color=pal["navy"])
    ax.text(712 + 89.5, 0, "sealed until\nthe end", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")
    if tag == "TODAY":
        ax.text(356, 0, "TODAY", ha="center", va="center", fontsize=9,
                fontweight="bold", color=pal["blue"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=pal["blue"], lw=1.0))
    elif tag:
        ax.text(430, -1.04, tag, ha="center", va="top", fontsize=9.5,
                fontweight="bold", color=pal["blue"])
    ax.set_xlim(-6, 897)
    ax.set_ylim(-1.45, 1.0)
    ax.axis("off")


def make_frame_c(out_path, tag=None, figsize=(7.5, 1.75)):
    """Save the canonical Frame C bar as a standalone image."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_frame_c(ax, tag=tag)
    fig.savefig(out_path)
    plt.close(fig)
