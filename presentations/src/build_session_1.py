"""Build deck: M2 Session 1 - From Analytics to Machine Learning.

All numbers from the notebook brief (M2 notebook 01 sections 1-3, notebook 02
for the baseline floor). Real-world examples from the research brief.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import deck_style as ds

FIGS = "/tmp/deck-workshop/figs1"
FIGS_STORY = "/tmp/deck-workshop/figs-story"
FIGS_CONCEPTS = "/tmp/deck-workshop/figs-concepts"
FIGS_THEORY = "/tmp/deck-workshop/figs-theory1"
os.makedirs(FIGS, exist_ok=True)
os.makedirs(FIGS_STORY, exist_ok=True)
os.makedirs(FIGS_CONCEPTS, exist_ok=True)
os.makedirs(FIGS_THEORY, exist_ok=True)
pal = ds.mpl_theme()


def notes(slide, text):
    """Speaker notes: a plain-text talk track for the teacher."""
    slide.notes_slide.notes_text_frame.text = text


def make_charts():
    # Fig 1: target balance (891 passengers; 549 died, 342 survived)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cats = ["Died", "Survived"]
    vals = [549, 342]
    colors = [pal["sky"], pal["blue"]]
    ax.barh(cats, vals, color=colors)
    ax.set_xlabel("Passengers")
    ax.set_title("Titanic target balance: 891 passengers")
    for i, v in enumerate(vals):
        ax.text(v + 8, i, f"{v} ({v / 891:.1%})", va="center",
                color=pal["ink"])
    ax.set_xlim(0, 660)
    fig.savefig(f"{FIGS}/fig1_target_balance.png")
    plt.close(fig)

    # Fig 2 (story): one manifest bar splits into 712 train / 179 test,
    # survivor share preserved (0.384 -> 0.383 / 0.385)
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    ax.barh(1.6, 891, height=0.42, color=pal["navy"])
    ax.text(445, 1.6, "All 891 passengers  -  survivor share 0.384",
            va="center", ha="center", color="white", fontweight="bold")
    ax.barh(0.55, 712, height=0.42, color=pal["blue"])
    ax.text(356, 0.55, "Train: 712 rows  -  survivor share 0.383",
            va="center", ha="center", color="white", fontweight="bold")
    ax.barh(0.55, 179, left=712, height=0.42, color=pal["sky"])
    ax.text(801, 0.55, "Test: 179", va="center", ha="center",
            color=pal["navy"], fontweight="bold", fontsize=11)
    ax.text(801, 0.18, "share 0.385", ha="center", color=pal["gray"],
            fontsize=10)
    ax.plot([712, 712], [0.82, 1.34], ls="--", color=pal["gray"], lw=1.2)
    ax.annotate("", xy=(300, 0.82), xytext=(300, 1.34),
                arrowprops=dict(arrowstyle="-|>", color=pal["gray"], lw=1.4))
    ax.text(720, 1.06, "80/20 cut, stratified, random_state=42",
            color=pal["gray"], fontsize=10, va="center")
    ax.set_xlim(-5, 905)
    ax.set_ylim(-0.15, 2.0)
    ax.axis("off")
    ax.set_title("Split first: one manifest becomes train and test")
    fig.savefig(f"{FIGS_STORY}/s1_split_story.png")
    plt.close(fig)


def make_error_stack_fig():
    """Frame B motif: stacked bar of total error - grey noise base,
    shrinkable colored top. Schematic, no numbers on either segment."""
    from matplotlib.patches import Rectangle, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    bx, bw = 0.30, 0.46  # bar x, width
    split = 0.40         # boundary between the two segments
    ax.add_patch(Rectangle((bx, 0.0), bw, split, facecolor=pal["gray"],
                           edgecolor="none"))
    ax.add_patch(Rectangle((bx, split), bw, 1.0 - split,
                           facecolor=pal["blue"], edgecolor="none"))
    ax.text(bx + bw / 2, split / 2, "noise\n(irreducible)", ha="center",
            va="center", color="white", fontweight="bold", fontsize=12)
    ax.text(bx + bw / 2, split + (1.0 - split) / 2, "reducible",
            ha="center", va="center", color="white", fontweight="bold",
            fontsize=13)
    # downward arrow on the reducible segment ONLY
    ax.add_patch(FancyArrowPatch((bx + bw + 0.10, 0.97),
                                 (bx + bw + 0.10, split + 0.05),
                                 arrowstyle="-|>", mutation_scale=20,
                                 color=pal["blue"], lw=2.2))
    # right-side annotations
    tx = bx + bw + 0.24
    ax.text(tx, 0.80, "REDUCIBLE - your job", color=pal["navy"],
            fontweight="bold", fontsize=13.5)
    ax.text(tx, 0.62, "your estimate of the hidden rule is off:\nbetter "
            "features, models, or data\nshrink this part", color=pal["gray"],
            fontsize=11.5, va="top")
    ax.text(tx, 0.28, "IRREDUCIBLE (noise ε) -\nnobody can remove this",
            color=pal["navy"], fontweight="bold", fontsize=13.5, va="top")
    ax.text(tx, 0.12, "randomness no manifest records -\nthe floor under "
            "any model, however perfect", color=pal["gray"], fontsize=11.5,
            va="top")
    # left bracket label
    ax.annotate("", xy=(bx - 0.10, 0.0), xytext=(bx - 0.10, 1.0),
                arrowprops=dict(arrowstyle="-", color=pal["gray"], lw=1.2))
    ax.text(bx - 0.17, 0.5, "total prediction error", rotation=90,
            ha="center", va="center", color=pal["ink"], fontsize=12)
    ax.text(bx - 0.17 + 0.0, 1.10, "Y = f(X) + ε", color=pal["navy"],
            fontweight="bold", fontsize=15)
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-0.04, 1.2)
    ax.axis("off")
    fig.savefig(f"{FIGS_THEORY}/s1_error_stack.png")
    plt.close(fig)


def make_recipe_fig():
    """Frame A motif: the three-slot recipe as three boxes -> best rule,
    with one worked filling (Logistic Regression) beneath."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    slots = [
        (0.30, "SLOT 1 - THE MODEL", "candidate rules",
         "a family of rules to\nchoose from (the\nhypothesis class)"),
        (3.35, "SLOT 2 - THE LOSS", "the scorecard",
         "how bad is each wrong\nprediction? 0-1 loss =\njust count mistakes"),
        (6.40, "SLOT 3 - THE OPTIMIZER", "the search",
         "finds the family member\nwith the lowest\naverage loss"),
    ]
    w, h, y0 = 2.55, 2.15, 2.30
    for x, head, sub, det in slots:
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.05",
                                    facecolor=pal["panel"],
                                    edgecolor=pal["blue"], lw=1.6))
        ax.text(x + w / 2, y0 + h - 0.32, head, ha="center", color=pal["navy"],
                fontweight="bold", fontsize=11.5)
        ax.text(x + w / 2, y0 + h - 0.68, sub, ha="center", color=pal["blue"],
                fontsize=11)
        ax.text(x + w / 2, y0 + 0.62, det, ha="center", color=pal["gray"],
                fontsize=9.8)
    for xp in (3.10, 6.15):
        ax.text(xp, y0 + h / 2, "+", ha="center", va="center",
                color=pal["navy"], fontsize=26, fontweight="bold")
    ax.add_patch(FancyArrowPatch((9.05, y0 + h / 2), (9.55, y0 + h / 2),
                                 arrowstyle="-|>", mutation_scale=20,
                                 color=pal["navy"], lw=2.0))
    ax.add_patch(FancyBboxPatch((9.62, y0), 2.45, h, boxstyle="round,pad=0.05",
                                facecolor=pal["blue"], edgecolor=pal["blue"]))
    ax.text(10.845, y0 + h - 0.55, "THE BEST RULE", ha="center",
            color="white", fontweight="bold", fontsize=12.5)
    ax.text(10.845, y0 + 0.75, "one fitted model,\nready to predict",
            ha="center", color=pal["sky"], fontsize=10.5)
    # worked filling lane: heading row on top, fillings row beneath
    ax.add_patch(FancyBboxPatch((0.30, 0.30), 11.77, 1.30,
                                boxstyle="round,pad=0.05",
                                facecolor="white", edgecolor=pal["sky"],
                                lw=1.4))
    ax.text(6.185, 1.30, "One worked filling - Session 3's champion, "
            "Logistic Regression:", ha="center", color=pal["navy"],
            fontweight="bold", fontsize=10.5)
    fillings = [(0.30 + w / 2, "weighted scorecard"),
                (3.35 + w / 2, "+  log loss"),
                (6.40 + w / 2, "+  gradient descent"),
                (10.845, "survival probability\nper passenger")]
    for x, label in fillings:
        ax.text(x, 0.72, label, ha="center", va="center", color=pal["blue"],
                fontsize=10.5, fontweight="bold")
    for x, _, _, _ in slots:
        ax.plot([x + w / 2, x + w / 2], [1.68, y0 - 0.05], ls=":",
                color=pal["sky"], lw=1.4)
    ax.plot([10.845, 10.845], [1.68, y0 - 0.05], ls=":", color=pal["sky"],
            lw=1.4)
    ax.set_xlim(0, 12.3)
    ax.set_ylim(0.10, 4.75)
    ax.axis("off")
    fig.savefig(f"{FIGS_THEORY}/s1_recipe_slots.png")
    plt.close(fig)


def make_cost_downhill_fig():
    """Cost + gradient descent intuition: three candidate rules with their
    J values (Ng's illustrative numbers) and a ball walking down a 1-D bowl.
    Synthetic data only - labeled as illustration on the slide."""
    import numpy as np
    rng = np.random.default_rng(42)
    x = np.linspace(0.5, 9.5, 12)
    y = 1.6 * x + 3 + rng.normal(0, 1.4, x.size)
    b, a = np.polynomial.polynomial.polyfit(x, y, 1)  # best line
    cands = [("A bad rule:  J = 480", lambda t: 16.5 - 1.1 * t),
             ("A decent rule:  J = 90", lambda t: 1.05 * t + 7.2),
             ("The best rule:  J = 12", lambda t: a * t + b)]
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.8))
    for ax, (title, f) in zip(axes.flat[:3], cands):
        xs = np.linspace(0, 10, 40)
        for xi, yi in zip(x, y):  # miss bars, point to line
            ax.plot([xi, xi], [yi, f(xi)], color=pal["sky"], lw=1.6)
        ax.scatter(x, y, s=26, color=pal["navy"], zorder=3)
        ax.plot(xs, f(xs), color=pal["blue"], lw=2.2)
        ax.set_title(title, fontsize=12.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, 22)
    axes.flat[0].set_ylabel("target y", fontsize=10)
    axes.flat[0].set_xlabel("feature x", fontsize=10)
    # panel 4: the downhill walk on J
    ax = axes.flat[3]
    wgrid = np.linspace(-3, 3, 120)
    ax.plot(wgrid, wgrid ** 2, color=pal["gray"], lw=2)
    steps = np.array([-2.6, -1.85, -1.25, -0.78, -0.44, -0.2, 0.0])
    ax.scatter(steps, steps ** 2, s=34, color=pal["blue"], zorder=3)
    for w0, w1 in zip(steps[:-1], steps[1:]):
        ax.annotate("", xy=(w1, w1 ** 2), xytext=(w0, w0 ** 2),
                    arrowprops=dict(arrowstyle="-|>", color=pal["blue"],
                                    lw=1.3, shrinkA=3, shrinkB=3))
    ax.annotate("start: any guess", (-2.6, 6.76), xytext=(-2.35, 8.6),
                color=pal["ink"], fontsize=10)
    ax.annotate("smallest J", (0, 0), xytext=(0.55, 0.7),
                color=pal["ink"], fontsize=10)
    ax.set_title("The search: walk downhill on J", fontsize=12.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("a model knob (parameter)", fontsize=10)
    ax.set_ylabel("cost J", fontsize=10)
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    fig.savefig(f"{FIGS_THEORY}/s1_cost_downhill.png")
    plt.close(fig)


def make_induction_fig():
    """The induction problem: finitely many labeled points -> a rule ->
    a question mark on a never-seen point. Synthetic - captioned as
    illustration on the slide."""
    import numpy as np
    from matplotlib.patches import FancyArrowPatch
    rng = np.random.default_rng(7)
    x = np.array([0.8, 1.5, 2.2, 2.9, 3.6, 4.3, 5.1, 5.9])
    y = 0.62 * x + 2.2 + rng.normal(0, 0.5, x.size)
    b, a = np.polynomial.polynomial.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    xs = np.linspace(0.2, 9.5, 40)
    ax.plot(xs, a * xs + b, color=pal["blue"], lw=2.4)
    ax.scatter(x, y, s=55, color=pal["navy"], zorder=3)
    ax.plot([0.6, 6.1], [1.35, 1.35], color=pal["gray"], lw=1.1)
    ax.text(3.35, 1.05, "the examples you have - finitely many",
            ha="center", va="top", color=pal["navy"], fontsize=11.5,
            fontweight="bold")
    ax.text(3.35, 0.45, "estimation: see through the noise\nin the data "
            "you have", ha="center", va="top", color=pal["gray"],
            fontsize=10.5)
    ax.text(7.0, a * 7.0 + b - 1.05, "the rule you leap to",
            color=pal["blue"], fontsize=11.5, fontweight="bold", rotation=10)
    xq = 8.7
    yq = a * xq + b
    ax.scatter([xq], [yq], s=600, facecolor="white", edgecolor=pal["blue"],
               lw=2.0, zorder=4)
    ax.text(xq, yq, "?", ha="center", va="center", color=pal["blue"],
            fontsize=19, fontweight="bold", zorder=5)
    ax.add_patch(FancyArrowPatch((3.3, 6.6), (xq - 0.35, yq + 0.8),
                                 connectionstyle="arc3,rad=-0.14",
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=pal["navy"], lw=1.8))
    ax.text(4.9, 9.65, "generalization: answer for a case never seen",
            ha="center", color=pal["navy"], fontsize=11.5, fontweight="bold")
    ax.text(4.9, 9.1, "possible only if new data looks like old (i.i.d.)",
            ha="center", color=pal["gray"], fontsize=10.5)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(-1.1, 10.3)
    ax.axis("off")
    fig.savefig(f"{FIGS_THEORY}/s1_induction.png")
    plt.close(fig)


def make_hypothesis_fig():
    """The hypothesis class: two candidate families through the same
    points - faint members are the class, the bold one is the member the
    search returns. Synthetic - captioned as illustration."""
    import numpy as np
    rng = np.random.default_rng(11)
    x = np.linspace(0.4, 9.6, 11)
    y = 2.0 + 1.1 * x - 0.055 * x ** 2 + rng.normal(0, 0.55, x.size)
    xs = np.linspace(0, 10, 120)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.9), sharey=True)
    ax = axes[0]
    for slope, icpt in [(0.2, 4.2), (0.45, 3.2), (0.95, 0.4),
                        (1.05, -0.5), (0.15, 1.6), (0.8, 2.6)]:
        ax.plot(xs, icpt + slope * xs, color=pal["sky"], lw=1.5)
    b, a = np.polynomial.polynomial.polyfit(x, y, 1)
    ax.plot(xs, a * xs + b, color=pal["blue"], lw=2.8, zorder=3)
    ax.set_title("A simple family: straight lines", fontsize=12.5)
    ax.text(5, 9.2, "faint = candidate rules the family allows",
            ha="center", color=pal["gray"], fontsize=10.5)
    ax.annotate("bold = the member\nthe search returns",
                xy=(7.6, a * 7.6 + b), xytext=(6.4, 0.3),
                color=pal["blue"], fontsize=10.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=pal["blue"],
                                lw=1.3))
    ax = axes[1]
    for seed in range(5):
        r2 = np.random.default_rng(seed + 30)
        coef = np.polynomial.polynomial.polyfit(
            x, y + r2.normal(0, 1.0, x.size), 4)
        ax.plot(xs, np.polynomial.polynomial.polyval(xs, coef),
                color=pal["sky"], lw=1.5)
    coef = np.polynomial.polynomial.polyfit(x, y, 3)
    ax.plot(xs, np.polynomial.polynomial.polyval(xs, coef),
            color=pal["blue"], lw=2.8, zorder=3)
    ax.set_title("A flexible family: curves", fontsize=12.5)
    ax.text(5, 9.2, "same points - a roomier family to search",
            ha="center", color=pal["gray"], fontsize=10.5)
    for ax in axes:
        ax.scatter(x, y, s=42, color=pal["navy"], zorder=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, 10)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(f"{FIGS_THEORY}/s1_hypothesis.png")
    plt.close(fig)


def make_loop_chart():
    """Fig 3 (story): the ML workflow drawn as a loop with one exit."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    steps = [("1. Prepare", "split, clean, encode\nSessions 1-2", 0.4),
             ("2. Train", "fit on train only\nSession 3", 2.9),
             ("3. Evaluate", "score on unseen data\nSession 3", 5.4),
             ("4. Improve", "tune, better features\nSession 4", 7.9)]
    for name, sub, x in steps:
        ax.add_patch(FancyBboxPatch((x, 2.0), 2.0, 1.15,
                                    boxstyle="round,pad=0.06",
                                    facecolor=pal["panel"],
                                    edgecolor=pal["blue"], lw=1.4))
        ax.text(x + 1.0, 2.86, name, ha="center", fontweight="bold",
                color=pal["navy"], fontsize=13)
        ax.text(x + 1.0, 2.38, sub, ha="center", color=pal["gray"],
                fontsize=9.5)
    for x0 in (2.48, 4.98, 7.48):  # forward arrows
        ax.add_patch(FancyArrowPatch((x0, 2.58), (x0 + 0.4, 2.58),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color=pal["blue"], lw=1.6))
    # the loop back: improve -> prepare (bulge below the boxes)
    ax.add_patch(FancyArrowPatch((8.9, 1.86), (1.4, 1.86),
                                 connectionstyle="arc3,rad=-0.26",
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=pal["navy"], lw=1.8))
    ax.text(5.15, 0.38, "evaluation sends you back to preparation "
            "more often than forward", ha="center", color=pal["navy"],
            fontsize=10.5)
    # the exit: evaluate -> ship
    ax.add_patch(FancyBboxPatch((5.3, 3.78), 2.2, 0.72,
                                boxstyle="round,pad=0.06",
                                facecolor=pal["blue"],
                                edgecolor=pal["blue"], lw=1.4))
    ax.text(6.4, 4.14, "5. Ship (Session 4)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=11)
    ax.add_patch(FancyArrowPatch((6.4, 3.26), (6.4, 3.73),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=pal["blue"], lw=1.6))
    ax.text(6.62, 3.48, "when the honest\nscore is good enough",
            color=pal["gray"], fontsize=8.5, va="center")
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0.1, 4.8)
    ax.axis("off")
    ax.set_title("The ML workflow: a loop with one exit")
    fig.savefig(f"{FIGS_STORY}/s1_workflow_loop.png")
    plt.close(fig)


def make_hero_fig():
    """Hero infographic: descriptive analytics vs machine learning, two lanes."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    heads = ["THE QUESTION", "THE INPUT", "THE OUTPUT", "ON THE TITANIC"]
    xs = [2.10, 4.85, 7.60, 10.35]
    bw, bh = 2.35, 1.5
    for head, x in zip(heads, xs):
        ax.text(x + bw / 2, 5.35, head, ha="center", color=pal["gray"],
                fontsize=10.5, fontweight="bold")
    lanes = [
        (3.55, "DESCRIPTIVE\nANALYTICS", "your M1 world", pal["navy"],
         [("What\nhappened?", ""),
          ("Past data", "the 891-passenger\nmanifest"),
          ("Charts and\nsummaries", "a human reads them,\nthen decides"),
          ("38.4% of passengers\nsurvived", "one number about\nthe whole crowd")]),
        (0.75, "MACHINE\nLEARNING", "this module (M2)", pal["blue"],
         [("What will happen\nto a NEW case?", ""),
          ("Past data + a new,\nunseen row", "891 labeled examples,\nthen one new passenger"),
          ("A prediction", "software can\nact on it"),
          ("Will a passenger like\nOwen survive?", "an answer per person,\non demand")]),
    ]
    for y0, label, sub, lab_fc, boxes in lanes:
        ml = lab_fc == pal["blue"]
        ax.add_patch(FancyBboxPatch((0.12, y0), 1.62, bh,
                                    boxstyle="round,pad=0.04",
                                    facecolor=lab_fc, edgecolor=lab_fc))
        ax.text(0.93, y0 + 0.95, label, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
        ax.text(0.93, y0 + 0.38, sub, ha="center", va="center",
                color=pal["sky"], fontsize=9)
        for i, (x, (head, det)) in enumerate(zip(xs, boxes)):
            if ml and i == 3:
                fc, ec, tc, dc = pal["blue"], pal["blue"], "white", pal["sky"]
            elif ml:
                fc, ec, tc, dc = "white", pal["blue"], pal["navy"], pal["gray"]
            else:
                fc, ec, tc, dc = pal["panel"], pal["sky"], pal["navy"], pal["gray"]
            ax.add_patch(FancyBboxPatch((x, y0), bw, bh,
                                        boxstyle="round,pad=0.04",
                                        facecolor=fc, edgecolor=ec, lw=1.6))
            hy = y0 + (0.95 if det else 0.75)
            ax.text(x + bw / 2, hy, head, ha="center", va="center",
                    color=tc, fontweight="bold", fontsize=10.5)
            if det:
                ax.text(x + bw / 2, y0 + 0.38, det, ha="center", va="center",
                        color=dc, fontsize=8.6)
            if i:
                ax.add_patch(FancyArrowPatch(
                    (x - 0.36, y0 + bh / 2), (x - 0.06, y0 + bh / 2),
                    arrowstyle="-|>", mutation_scale=13,
                    color=pal["blue"] if ml else pal["gray"], lw=1.4))
    ax.text(6.43, 2.9, "same past data - a different question, and an output "
            "a program can use", ha="center", color=pal["navy"],
            fontsize=11.5, fontweight="bold")
    ax.add_patch(FancyArrowPatch((0.93, 3.48), (0.93, 2.32), arrowstyle="-|>",
                                 mutation_scale=15, color=pal["navy"], lw=1.6))
    ax.set_xlim(0, 12.85)
    ax.set_ylim(0.35, 5.75)
    ax.axis("off")
    fig.savefig(f"{FIGS_CONCEPTS}/hero_da_vs_ml.png")
    plt.close(fig)


def make_ladder_fig():
    """The analytics ladder: descriptive, diagnostic, predictive, prescriptive."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    steps = [
        ("1. Descriptive", "What happened?",
         "means, charts,\nvalue_counts", 0.30, 1.80, "m1"),
        ("2. Diagnostic", "Why did it happen?",
         "groupby, crosstab,\ncorrelation", 2.95, 2.70, "m1"),
        ("3. Predictive", "What will happen?",
         "models learned from\nlabeled examples", 5.60, 3.60, "m2"),
        ("4. Prescriptive", "What should we do?",
         "decisions built on\ntrusted predictions", 8.25, 4.50, "next"),
    ]
    w, base = 2.45, 0.40
    for name, q, sub, x, top, kind in steps:
        if kind == "m2":
            fc, ec, tc, qc, ls = pal["blue"], pal["blue"], "white", pal["sky"], "-"
        elif kind == "m1":
            fc, ec, tc, qc, ls = pal["panel"], pal["sky"], pal["navy"], pal["gray"], "-"
        else:
            fc, ec, tc, qc, ls = "white", pal["gray"], pal["navy"], pal["gray"], "--"
        ax.add_patch(FancyBboxPatch((x, base), w, top - base,
                                    boxstyle="round,pad=0.03", facecolor=fc,
                                    edgecolor=ec, lw=1.5, linestyle=ls))
        ax.text(x + w / 2, top - 0.28, name, ha="center", color=tc,
                fontweight="bold", fontsize=12.5)
        ax.text(x + w / 2, top - 0.62, q, ha="center", color=tc, fontsize=10.5)
        ax.text(x + w / 2, top - 1.13, sub, ha="center", color=qc, fontsize=8.8)
    brackets = [
        (0.30, 5.40, "M1 - done: you built these skills last module",
         pal["gray"], False),
        (5.60, 8.05, "M2 - THIS MODULE", pal["blue"], True),
        (8.25, 10.70, "after a trusted model", pal["gray"], False)]
    for x0, x1, label, c, bold in brackets:
        ax.plot([x0, x0, x1, x1], [0.16, 0.04, 0.04, 0.16], color=c, lw=1.2)
        ax.text((x0 + x1) / 2, -0.24, label, ha="center", color=c,
                fontsize=10, fontweight="bold" if bold else "normal")
    ax.set_xlim(0, 10.95)
    ax.set_ylim(-0.55, 4.95)
    ax.axis("off")
    fig.savefig(f"{FIGS_CONCEPTS}/ladder_analytics.png")
    plt.close(fig)


def slides():
    prs = ds.new_deck()

    # 1. Title
    ds.title_slide(
        prs,
        "Session 1 of 4",
        "From Analytics to Machine Learning",
        "On the night of 14-15 April 1912 the Titanic went down; 549 of the "
        "891 passengers on our manifest died. One question for four "
        "sessions: could the data have told us who lives? First the ideas, "
        "then the manifest.",
    )

    # 2. Part 1 divider
    ds.section_slide(
        prs,
        "01",
        "Part 1 - The theory: what learning is",
        "Ideas before code: what a machine actually does when it learns, "
        "and why the leap to unseen cases can work at all.",
    )

    # 3. The DA-vs-ML hero infographic: the CONTRAST in one picture
    s = ds.image_slide(
        prs,
        "Descriptive analytics reads the past - machine learning answers "
        "about the future",
        f"{FIGS_CONCEPTS}/hero_da_vs_ml.png",
        kicker="The bridge from M1",
        caption="Same past data - a different question, and an output a "
                "program can use. M1 drew the line itself: describing is "
                "not predicting. M2 starts exactly there.",
    )
    notes(s, "Start from what they already know: in M1 they answered 'what "
             "happened' with charts a person reads. Machine learning answers "
             "a new question - 'what will happen to a NEW case' - and the "
             "answer is something a program can act on. Same data, different "
             "question. Ask the class: the app that predicts when your bus "
             "arrives - which of the two rows is it on?")

    # 4. The analytics ladder: the PROGRESSION (where M2 sits on the climb)
    s = ds.image_slide(
        prs,
        "Prediction is the next rung on a ladder you are already halfway up",
        f"{FIGS_CONCEPTS}/ladder_analytics.png",
        kicker="The analytics ladder",
        bullets=[
            "Descriptive and diagnostic: M1. You told what happened and dug "
            "into why - groupbys, crosstabs, correlations",
            "Predictive: M2, this module. What will happen to a new, "
            "unseen case?",
            "Prescriptive sits on top: what should we do - decisions built "
            "on predictions you can trust",
            "Your M1 skills still do most of the daily job: 80% of the work "
            "is still cleaning and describing data",
            "The climb, session by session: S1 what ML is and honest "
            "splits; S2 features; S3 the model tournament; S4 tune and ship",
        ],
        caption="MIT draws the same line by goal, not tools: statistics "
                "asks 'does the model explain?' - ML asks 'does the "
                "prediction hold up?' (MIT 6.390, Intro chapter). Ng, "
                "Harvard CS109A and MIT 6.390 all climb this same ladder.",
    )
    notes(s, "Reassure them: this module is one rung up, not a new ladder. "
             "They already climbed the first two rungs in M1, and those "
             "skills stay in daily use. Point at each rung and say its "
             "question out loud - what happened, why, what will happen, what "
             "should we do. Ask the class: where on this ladder does a "
             "weather forecast sit?")
    # 4. What machine learning is (absorbs old types table + everyday examples)
    s = ds.bullets_slide(
        prs,
        "ML learns the rules from examples instead of being told them",
        [
            ("Classic programming: a human writes the rules by hand",
             ["A hand-written spam rule list - block 'winner', block ALL "
              "CAPS, block unknown senders - never ends, and spammers adapt "
              "faster than you can type"]),
            ("Machine learning: you show labeled examples, the algorithm "
             "finds the rules",
             ["Arthur Samuel called it the \"field of study that gives "
              "computers the ability to learn without being explicitly "
              "programmed\" - his 1950s checkers program beat its own "
              "programmer after tens of thousands of self-played games"]),
            ("It is already in your day",
             ["Recommendations predict what you will want next; map "
              "arrival times predict a trip nobody has driven in exactly "
              "these conditions"]),
            "The same move on our data: 891 passengers with known outcomes "
            "go in; a pattern that predicts new passengers comes out",
            "A model is a function with adjustable knobs (parameters); "
            "training turns the knobs until predictions match the examples",
        ],
        kicker="What ML is",
        note="Samuel's phrase, quoted by Ng - Machine Learning "
             "Specialization C1W1.",
    )
    notes(s, "Two ways to get rules into a computer: write them yourself, or "
             "show labeled examples and let the algorithm find the rules. "
             "Spam is the perfect story - a hand-written rule list never "
             "ends, spammers adapt faster than you type. Ask the class: what "
             "would YOUR three spam rules be, and how would a spammer dodge "
             "each one within a week?")

    # Types of machine learning - the three families
    s = ds.bullets_slide(
        prs,
        "Machines learn in three ways - this module lives in the first",
        [
            ("Supervised learning: the examples include the right answer "
             "(the label)",
             ["A spam filter learns from mail already marked spam or not "
              "spam - ours: 891 passengers, each labeled Survived, 1 or 0"]),
            ("Unsupervised learning: no labels - find the structure "
             "yourself",
             ["Grouping shoppers into segments nobody named in advance: "
              "there is no right answer in the data to copy"]),
            ("Reinforcement learning: no examples at all - act, get "
             "rewarded or penalized, improve",
             ["Samuel's checkers program improving through wins and losses "
              "in self-play is the classic seed of the idea"]),
            "The families differ in what the data gives you: answers, "
            "no answers, or only consequences",
            "This module is supervised classification end to end - the "
            "other two families are a later course",
        ],
        kicker="Types of machine learning",
        note="Supervised learning comes first because it covers most ML "
             "used in industry today - Ng, Machine Learning Specialization "
             "C1W1.",
    )
    notes(s, "The three families differ only in what the data hands you: the "
             "right answers (supervised), no answers (unsupervised), or only "
             "consequences after acting (reinforcement). Everything in this "
             "module is supervised - our passengers come with the answer "
             "attached: survived or not. Ask the class: grouping customers "
             "into types nobody named in advance - which family is that?")

    # 5. Features and target
    s = ds.two_col_slide(
        prs,
        "Every supervised problem is a table: features X in, target y out",
        ("Features X = the inputs", [
            "The columns that describe each passenger",
            "Age, Sex, Pclass, Fare, SibSp, Parch, Embarked",
            "In code: X = everything except the answer",
            "Books also say predictors or covariates - "
            "different words, same idea: the input columns",
        ]),
        ("Target y = the answer to learn", [
            "One column: Survived, 1 or 0",
            "Predicting a category -> classification",
            "Predicting a number -> regression",
            "Survived is a category, so this module is "
            "a classification problem",
        ]),
        kicker="Framing",
        note="Same table, different y: predicting the Fare a passenger paid "
             "would make this a regression problem instead. Vocabulary: "
             "Harvard CS109A / ISLR ch. 2.",
    )
    notes(s, "Keep it physical: X is every column that describes a "
             "passenger, y is the one column we want to predict. Category "
             "out means classification, number out means regression - that "
             "is the whole distinction. Ask the class: same table, but now "
             "predict the Fare a passenger paid - classification or "
             "regression?")

    # 6. NEW - Frame B: Y = f(X) + epsilon, reducible vs irreducible error
    s = ds.image_slide(
        prs,
        "Every outcome is a hidden rule plus noise - only the rule part "
        "is learnable",
        f"{FIGS_THEORY}/s1_error_stack.png",
        kicker="The hidden rule and the noise",
        bullets=[
            "Y = f(X) + ε - in plain words: every outcome is pattern "
            "plus luck",
            "f is the pattern - learnable. ε (epsilon) is the luck - "
            "nothing predicts a coin flip",
            "On the Titanic: class, sex and age shape survival (the "
            "pattern); chaos that night is the luck",
            "A model estimates the pattern; the error you can still "
            "shrink is called reducible",
            "The luck is an irreducible floor - why 100% accuracy is "
            "never the goal; Session 3 pays this off",
        ],
        caption="Schematic illustration - no course numbers on this chart. "
                "Harvard CS109A / ISLR ch. 2.",
    )
    notes(s, "Say it as: every outcome is pattern plus luck. The pattern is "
             "learnable - class, sex and age really did shape survival. The "
             "luck is not: two identical passengers could meet different "
             "fates in the chaos. Better features and models shrink the "
             "pattern error; nothing shrinks the luck. Ask the class: why "
             "would a model claiming 100% accuracy on survival actually be "
             "suspicious?")

    # NEW - The induction problem (MIT)
    s = ds.image_slide(
        prs,
        "Learning is a leap from a few examples to cases never seen - one "
        "assumption makes it work",
        f"{FIGS_THEORY}/s1_induction.png",
        kicker="The induction problem",
        bullets=[
            "The induction problem - in plain words: after 10 dog photos "
            "you recognize an 11th you never saw",
            "Every learner makes that leap: a few examples in, answers "
            "for cases nobody has seen yet",
            "Nothing guarantees the future resembles the past - the leap "
            "rests on one assumption",
            "The assumption (called i.i.d.): new data looks like old - "
            "no assumption, no leap",
            "MIT splits the job in two: estimation - see through the "
            "noise; generalization - answer for inputs never seen",
            "Part 2 builds this exact leap: 712 passengers teach a rule; "
            "179 unseen ones judge it",
        ],
        caption="Illustration - synthetic data. MIT 6.390, Intro chapter "
                "('the problem of induction').",
    )
    notes(s, "Lead with the dog photos: after ten photos of dogs you "
             "recognize an eleventh you have never seen - that is all "
             "learning is. It works only because tomorrow resembles "
             "yesterday; that assumption is the bridge, not a proof. Ask the "
             "class: give me a case where tomorrow does NOT resemble "
             "yesterday - would a model trained on the past still work?")

    # NEW - The hypothesis class (MIT)
    s = ds.image_slide(
        prs,
        "You choose the space of candidate rules - the algorithm only "
        "finds its best member",
        f"{FIGS_THEORY}/s1_hypothesis.png",
        kicker="The hypothesis class",
        bullets=[
            "Hypothesis class - in plain words: you pick the shape of the "
            "answer (a line? a curve?)",
            "One hypothesis = one concrete rule, numbers filled in; the "
            "class = all rules of that shape",
            "Learning is a search: the algorithm scans the family and "
            "returns the member that fits best",
            "You pick the family before touching any data - that design "
            "call is yours, not the algorithm's",
            "A roomier family follows the pattern more closely - and "
            "follows the noise too; it cuts both ways",
            "MIT splits it: choose the class (Session 3's dial) vs fit "
            "the member (training's job)",
        ],
        caption="Illustration - synthetic data. MIT 6.390, Intro chapter.",
    )
    notes(s, "The one-liner: you decide the shape of the answer - a line or "
             "a curve - and training only finds the best rule of that shape. "
             "Point at the faint lines: those are the candidates the family "
             "allows; the bold one is what the search returns. Ask the "
             "class: if the true pattern is a curve and you only allow "
             "straight lines, can training ever fix that?")

    # The three-slot recipe - the chapter's synthesis
    s = ds.image_slide(
        prs,
        "Every learner you will meet is the same recipe: candidate rules "
        "+ a score + a search",
        f"{FIGS_THEORY}/s1_recipe_slots.png",
        kicker="The three-slot recipe",
        bullets=[
            "The three-slot recipe - in plain words: a menu of rules + a "
            "scorecard + a search",
            "Slot 1, the model: the hypothesis class you just met - your "
            "call, made before any data",
            "Slot 2, the loss: how bad is each wrong answer? Simplest "
            "scorecard: just count mistakes",
            "Slot 3, the optimizer: the search that finds the family "
            "member with the best score",
            "Every ML method = a different filling of the same slots; "
            "Session 3 refills slot 1 six times",
        ],
        caption="One recipe, every method in this module. MIT 6.390, Intro "
                "chapter & Appendix C 'Supervised learning in a nutshell'.",
    )
    notes(s, "This is the most reusable slide of the module: every learner "
             "they will ever meet is a menu of candidate rules, a scorecard "
             "for wrongness, and a search for the best-scoring rule. Walk "
             "the worked filling underneath: logistic regression is just one "
             "way to fill the three slots. Ask the class: which slot did the "
             "previous slide - the hypothesis class - cover?")

    # 8. NEW - cost function + downhill training (fills slots 2-3)
    s = ds.image_slide(
        prs,
        "Training is nothing mystical: make one number - the average miss "
        "- small",
        f"{FIGS_THEORY}/s1_cost_downhill.png",
        kicker="How training works",
        bullets=[
            "The cost J - in plain words: the average miss over all 891 "
            "examples - one number per rule",
            "Training = the search for the rule with the smallest J - "
            "nothing more mystical than that",
            "The search walks downhill in fog: nudge the knobs in "
            "whichever direction lowers J fastest",
            "Step size matters: too small crawls, too large overshoots "
            "the valley",
            "The same downhill walk trains Session 3's scorecard - no "
            "magic anywhere in this module",
        ],
        caption="Illustration - synthetic data; illustrative J values, not "
                "course results. Method: Ng, Machine Learning "
                "Specialization C1 / Stanford CS229 notes.",
    )
    notes(s, "Demystify training: score every candidate rule by its average "
             "miss - one number, J - and hunt for the rule with the smallest "
             "J. The hunt is a walk downhill in fog: feel the slope, step "
             "where it drops fastest, repeat. Point at the four panels: bad "
             "rule big J, decent rule smaller, best rule smallest. Ask the "
             "class: what goes wrong if your downhill steps are huge?")

    # Part 2 divider
    ds.section_slide(
        prs,
        "02",
        "Part 2 - Practice: the Titanic manifest",
        "The ideas land on 891 real passengers: one dataset, one honest "
        "split, and a floor every model must beat.",
    )

    # The dataset (chart)
    ds.image_slide(
        prs,
        "One dataset carries every concept: 891 passengers, 38.4% survived",
        f"{FIGS}/fig1_target_balance.png",
        kicker="The dataset",
        bullets=[
            "891 rows x 12 columns - small enough to read, real enough "
            "to be messy",
            "549 died, 342 survived: imbalanced (far from 50/50), but "
            "not extremely",
            "Missing values are real: Age 177, Cabin 687, Embarked 2",
            "One worked example carrying every new concept is Andrew Ng's "
            "teaching pattern - we reuse the Titanic the same way",
        ],
        caption="Class balance in the full dataset. Source: Module 2, "
                "notebook 01, section 2.",
    )

    # 7. Meet three passengers - the module's recurring characters
    ds.table_slide(
        prs,
        "Three real rows put faces on the data - they return every session",
        ["Passenger (as listed)", "Class", "Age", "Family aboard", "Fare",
         "Boarded", "Outcome"],
        [
            ["Braund, Mr. Owen Harris", "3rd", "22", "1 (SibSp 1)",
             "7.25", "Southampton", "Died"],
            ["Cumings, Mrs. John Bradley (Florence Briggs Thayer)", "1st",
             "38", "1 (SibSp 1)", "71.2833", "Cherbourg", "Survived"],
            ["Goldsmith, Master. Frank John William \"Frankie\"", "3rd",
             "9", "2 (Parch 2)", "20.525", "Southampton", "Survived"],
        ],
        kicker="The manifest",
        note="PassengerId 1, 2 and 166 of the dataset, fields verbatim. Owen, "
             "Florence and Frankie carry the whole module: their gaps in "
             "Session 2, their odds in Sessions 3-4.",
        col_widths=[3.4, 0.55, 0.5, 1.05, 0.75, 1.0, 0.85],
    )

    # 8. Which columns may be features
    ds.table_slide(
        prs,
        "Not every column becomes a feature - identifiers and free text sit out",
        ["Column", "Verdict", "Why"],
        [
            ["Pclass, Sex, Fare, SibSp, Parch", "Feature",
             "Numbers or clean categories, nearly complete"],
            ["Age", "Feature",
             "177 missing (19.9%) - we will impute, not drop"],
            ["Embarked", "Feature", "Only 2 missing (0.2%)"],
            ["Cabin", "Drop", "687 missing (77.1%) - mostly gaps"],
            ["Name", "Drop, for now",
             "Free text - Session 2 mines a Title feature from it"],
            ["Ticket", "Drop", "Arbitrary codes with no consistent meaning"],
            ["PassengerId", "Drop", "An identifier - a row number predicts "
             "nothing about survival"],
        ],
        kicker="Feature verdict",
        note="Rule of thumb: identifiers out, mostly-missing columns out, "
             "free text out until you can engineer something from it.",
        col_widths=[2.4, 1.1, 3.6],
    )

    # 9. The workflow as a loop (diagram)
    ds.image_slide(
        prs,
        "The ML workflow is a loop, not a line - you will walk it all module",
        f"{FIGS_STORY}/s1_workflow_loop.png",
        kicker="The workflow",
        bullets=[
            "Prepare: split, clean, encode, scale (Sessions 1-2) - today "
            "you started it on the 891-row manifest",
            "Train: fit models on the training set only (Session 3)",
            "Evaluate: score on data the model has never seen (Session 3)",
            "Improve, then ship: tune, engineer, deploy (Session 4)",
            "Evaluation sends you back to preparation more often than "
            "forward",
            "Harvard teaches it as a five-stage loop - ask, get, explore, "
            "model, communicate: modeling is one box of five, and answers "
            "routinely send you back a stage",
        ],
        caption="Harvard CS109A teaches the same shape: one workflow "
                "revisited all semester on messy real data. Source: Harvard "
                "CS109A, Lecture 1 (the data science process).",
    )

    # 10. The golden rule: split first (chart)
    s = ds.image_slide(
        prs,
        "Split before you prepare - even an innocent average can leak",
        f"{FIGS_STORY}/s1_split_story.png",
        kicker="The golden rule",
        bullets=[
            "80/20 split first: 712 training rows, 179 test rows (seed 42)",
            "The test set is the final exam - you sit it once; seeing the "
            "questions while studying proves nothing",
            "Data leakage - in plain words: exam answers sneaking into "
            "your study notes",
            "Even 'fill missing ages with the average' leaks, if that "
            "average was computed over all rows",
            "stratify=y - in plain words: keep the survivor share equal "
            "in both halves",
            "Leakage bites pros: KDD Cup 2008 - 'Patient ID' predicted "
            "cancer because the ID encoded the hospital, not the tumor",
            "A feature that works suspiciously well deserves suspicion, "
            "not celebration",
        ],
        caption="The split and the preserved survivor share. "
                "Source: Module 2, notebook 01, section 3. Leakage story: "
                "Kaufman, Rosset & Perlich, 'Leakage in Data Mining', "
                "KDD 2011.",
    )
    notes(s, "This is the module's exam metaphor - use it every session: the "
             "test set is the final exam, and you sit it exactly once. "
             "Leakage is any way the exam answers sneak into the study notes "
             "- even an average computed over all rows counts. Tell the KDD "
             "story slowly: the patient ID gave away the hospital, and the "
             "hospital gave away the diagnosis. Ask the class: why does a "
             "suspiciously good feature deserve suspicion first?")

    # 14. What "good" will mean
    s = ds.big_number_slide(
        prs,
        "In this module, 'good' means a number - and the floor is 0.617",
        "0.617",
        "Accuracy of a model that always predicts 'died' - the baseline "
        "floor every real model must beat",
        foot="Dummy classifier, 5-fold cross-validation on the training set. "
             "Source: Module 2, notebook 02. Beating the floor is the entry "
             "ticket, not the goal. Accuracy is 0-1 loss in disguise - the "
             "simplest filling of the recipe's loss slot (MIT 6.390, Intro "
             "chapter).",
        kicker="Honest evaluation preview",
    )
    notes(s, "A model that ignores everything and always answers 'died' is "
             "right 61.7% of the time, simply because most passengers died. "
             "That is the floor: any real model must clearly beat it, or it "
             "has learned nothing. Ask the class: why is 62% accuracy on "
             "this dataset nothing to celebrate?")

    # 15. Close (course mechanics folded in)
    ds.close_slide(
        prs,
        "You now predict, not describe - and every claim gets a number",
        [
            "Part 1, the ideas: ML learns rules from labeled examples - a "
            "leap from seen cases to unseen ones, allowed by one "
            "assumption (new data looks like old)",
            "You choose the hypothesis class; every learner = model + loss "
            "+ optimizer, and training = making one number (the cost J) "
            "small",
            "Y = f(X) + ε: we estimate the hidden rule, we never beat "
            "the noise",
            "Part 2, the manifest: 891 passengers, split first - 712 train "
            "/ 179 test, stratified; the test set is the exam you take once",
            "Every claim gets a number, and the floor is 0.617",
            "Practice now, zero setup: open notebook 01 in Colab (badge "
            "click), Runtime > Run all, work sections 1-3 - exercises at "
            "the end, solutions included, attempt them first",
        ],
    )

    return prs


if __name__ == "__main__":
    make_charts()
    make_error_stack_fig()
    make_recipe_fig()
    make_cost_downhill_fig()
    make_induction_fig()
    make_hypothesis_fig()
    make_loop_chart()
    make_hero_fig()
    make_ladder_fig()
    prs = slides()
    out = "../m2-session-1-machine-learning-fundamentals.pptx"
    ds.save_deck(prs, out, "M2 Session 1 - Machine Learning Fundamentals")
    print("Slides:", len(prs.slides._sldIdLst))
    print("Saved:", os.path.abspath(out))
