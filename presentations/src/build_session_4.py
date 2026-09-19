"""Build M2 Session 4 deck: Tuning and Shipping the Model."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import deck_style as ds

FIGS = "/tmp/deck-workshop/figs4"
FIGS_STORY = "/tmp/deck-workshop/figs-story"
FIGS_CONCEPTS = "/tmp/deck-workshop/figs-concepts"
FIGS_THEORY = "/tmp/deck-workshop/figs-theory4"
os.makedirs(FIGS, exist_ok=True)
os.makedirs(FIGS_STORY, exist_ok=True)
os.makedirs(FIGS_CONCEPTS, exist_ok=True)
os.makedirs(FIGS_THEORY, exist_ok=True)
pal = ds.mpl_theme()

RED = "#C0392B"  # used only for the two forbidden contamination arrows


def notes(slide, text):
    """Speaker notes: a plain-text talk track for the teacher."""
    slide.notes_slide.notes_text_frame.text = text

# ---- Chart A: the tuning arc (NB3 sec 2-4) ----
fig, ax = plt.subplots(figsize=(7.5, 4))
labels = ["Default forest\n(no tuning)", "Grid search\n(18 combinations)",
          "Random search\n(15 tries)"]
vals = [0.7937, 0.8259, 0.8274]
bars = ax.bar(labels, vals, color=[pal["sky"], pal["sky"], pal["blue"]], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0012, f"{v:.4f}", ha="center",
            fontsize=12, fontweight="bold", color=pal["navy"])
ax.set_ylim(0.75, 0.842)
ax.set_ylabel("5-fold CV accuracy\n(practice-test score)")
ax.set_title("The tuning arc: 0.7937 to 0.8259 to 0.8274")
ax.text(0.02, 0.97, "y-axis starts at 0.75, not 0 - gaps are zoomed",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        color=pal["gray"])
fig.savefig(f"{FIGS}/tuning_arc.png")
plt.close(fig)

# ---- Frame C: the FIT | CHOOSE | REPORT verbs bar (Harvard/ISLR motif) ----
# One canonical drawing, shared with Decks 2 and 3 via shared_figs. The bar
# now lives on the Part 1 selection-vs-assessment theory slide; the practice
# chart keeps only the real numbers.
from shared_figs import draw_frame_c

# ---- Chart B (practice): CV vs the one-shot test score ----
fig, ax = plt.subplots(figsize=(7.5, 4.2))
labels = ["Cross-validation\n(training data only)", "Test set\n(179 unseen passengers)"]
vals = [0.8274, 0.8268]
bars = ax.bar(labels, vals, color=[pal["sky"], pal["blue"]], width=0.4)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0012, f"{v:.4f}", ha="center",
            fontsize=12, fontweight="bold", color=pal["navy"])
ax.set_ylim(0.75, 0.842)
ax.set_ylabel("Accuracy")
ax.set_title("CV and test agree: a gap of only 0.0006")
ax.text(0.02, 0.97, "y-axis starts at 0.75, not 0 - gaps are zoomed",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        color=pal["gray"])
fig.savefig(f"{FIGS}/cv_vs_test_bar.png")
plt.close(fig)

# ---- Chart T1 (theory): two settings of one algorithm, one CV referee ----
fig, ax = plt.subplots(figsize=(7.5, 3.7))
contestants = [("Random Forest\nmax_depth = 4", 2.05), ("Random Forest\nmax_depth = 8", -0.05)]
for label, cy in contestants:
    ax.add_patch(FancyBboxPatch((0.15, cy), 2.5, 1.15, boxstyle="round,pad=0.05",
                                facecolor=pal["panel"], edgecolor=pal["blue"], lw=1.4))
    ax.text(1.4, cy + 0.575, label, ha="center", va="center",
            fontweight="bold", color=pal["navy"], fontsize=11)
    ax.add_patch(FancyArrowPatch((2.72, cy + 0.575), (4.03, 1.85 if cy > 1 else 1.3),
                                 arrowstyle="-|>", mutation_scale=15,
                                 color=pal["blue"], lw=1.6))
ax.text(1.4, 3.55, "two settings = two algorithms", ha="center", va="center",
        color=pal["gray"], fontsize=9.5, style="italic")
ax.add_patch(FancyBboxPatch((4.1, 0.95), 2.35, 1.25, boxstyle="round,pad=0.05",
                            facecolor=pal["blue"], edgecolor=pal["blue"], lw=1.4))
ax.text(5.275, 1.90, "5-fold CV", ha="center", va="center",
        fontweight="bold", color="white", fontsize=12)
ax.text(5.275, 1.38, "the same 5 practice\ntests as Session 3", ha="center",
        va="center", color=pal["sky"], fontsize=8.5)
ax.add_patch(FancyArrowPatch((6.52, 1.575), (7.28, 1.575), arrowstyle="-|>",
                             mutation_scale=15, color=pal["blue"], lw=1.6))
ax.add_patch(FancyBboxPatch((7.35, 0.95), 2.3, 1.25, boxstyle="round,pad=0.05",
                            facecolor=pal["navy"], edgecolor=pal["navy"], lw=1.4))
ax.text(8.5, 1.90, "Winner", ha="center", va="center",
        fontweight="bold", color="white", fontsize=12)
ax.text(8.5, 1.38, "the setting with the\nbetter CV score", ha="center", va="center",
        color=pal["sky"], fontsize=8.5)
ax.text(9.65, -0.35, "illustration", ha="right", va="center",
        color=pal["gray"], fontsize=8.5, style="italic")
ax.set_xlim(-0.1, 9.75)
ax.set_ylim(-0.55, 3.9)
ax.axis("off")
fig.savefig(f"{FIGS_THEORY}/two_settings_one_referee.png")
plt.close(fig)

# ---- Chart T2 (theory): grid vs random over an importance-skewed landscape ----
import numpy as np

rng = np.random.default_rng(42)
gx, gy = np.meshgrid(np.linspace(0, 1, 220), np.linspace(0, 1, 220))
# score depends strongly on x (the knob that matters), barely on y
score = np.exp(-((gx - 0.68) ** 2) / 0.045) * (0.92 + 0.08 * np.cos(3 * gy))
fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0))
grid_pts = [(x, y) for x in (0.17, 0.5, 0.83) for y in (0.17, 0.5, 0.83)]
rand_pts = list(zip(rng.uniform(0.04, 0.96, 9), rng.uniform(0.04, 0.96, 9)))
panels = [
    ("Grid: 9 trials,\n3 values of the knob that matters", grid_pts),
    ("Random: 9 trials,\n9 values of the knob that matters", rand_pts),
]
for axp, (ttl, pts) in zip(axes, panels):
    axp.imshow(score, extent=(0, 1, 0, 1), origin="lower", cmap="Blues",
               vmin=0, vmax=1.55, aspect="auto")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    axp.scatter(xs, ys, s=52, color=pal["navy"], edgecolor="white",
                lw=1.1, zorder=3)
    # projection ticks: what each trial learned about the important knob
    for x in xs:
        axp.plot([x, x], [-0.048, -0.012], color=pal["blue"], lw=1.8,
                 clip_on=False, zorder=3)
    axp.set_title(ttl, fontsize=10.5)
    axp.set_xlabel("knob that matters\n(ticks: distinct values tried)",
                   fontsize=9.5, labelpad=17)
    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1)
    axp.set_xticks([])
    axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_visible(True)
        sp.set_color(pal["gray"])
axes[0].set_ylabel("knob that barely matters", fontsize=9.5)
axes[1].text(1.0, -0.30, "darker background = better score - illustration",
             transform=axes[1].transAxes, ha="right", va="top",
             fontsize=8.5, style="italic", color=pal["gray"])
fig.subplots_adjust(bottom=0.24, wspace=0.14)
fig.savefig(f"{FIGS_THEORY}/grid_vs_random.png")
plt.close(fig)

# ---- Chart T3 (theory): the single-number scoreboard (Ng's A-vs-B device) ----
fig, ax = plt.subplots(figsize=(7.4, 4.1))
cols = ["", "Precision", "Recall", "Verdict", "F1"]
col_x, col_w = [0.0, 2.1, 3.7, 5.3, 7.3], [2.1, 1.6, 1.6, 2.0, 1.6]
rows = [("Classifier A", "0.95", "0.60", "wins precision", "0.74"),
        ("Classifier B", "0.85", "0.75", "wins recall", "0.80")]
top, row_h = 3.4, 0.78
for x, w, c in zip(col_x, col_w, cols):
    fc = pal["blue"] if c == "F1" else pal["navy"]
    ax.add_patch(plt.Rectangle((x, top), w, row_h, facecolor=fc,
                               edgecolor="white", lw=1.0, zorder=2))
    ax.text(x + w / 2, top + row_h / 2, c, ha="center", va="center",
            color="white", fontweight="bold", fontsize=11, zorder=3)
for r, vals in enumerate(rows, start=1):
    y = top - r * row_h
    for x, w, v in zip(col_x, col_w, vals):
        winner = (v == "0.80")
        fc = pal["panel"] if x == col_x[4] else "white"
        ax.add_patch(plt.Rectangle((x, y), w, row_h, facecolor=fc,
                                   edgecolor=pal["sky"], lw=0.8, zorder=1))
        ax.text(x + (0.15 if x == 0 else w / 2), y + row_h / 2, v,
                ha="left" if x == 0 else "center", va="center",
                color=pal["blue"] if winner else pal["ink"],
                fontweight="bold" if (winner or x == 0) else "normal",
                fontsize=10.5, zorder=3)
ax.text(6.3, top + row_h + 0.28, "deadlock", ha="center", va="center",
        color=pal["gray"], fontsize=9.5, style="italic")
ax.text(8.1, top + row_h + 0.28, "one number decides", ha="center", va="center",
        color=pal["blue"], fontsize=9.5, fontweight="bold")
# below: optimize one, satisfice the rest
by = top - 2 * row_h - 1.45
ax.add_patch(FancyBboxPatch((0.0, by), 4.25, 1.0, boxstyle="round,pad=0.05",
                            facecolor=pal["blue"], edgecolor=pal["blue"]))
ax.text(2.125, by + 0.68, "OPTIMIZE one metric", ha="center", va="center",
        color="white", fontweight="bold", fontsize=11)
ax.text(2.125, by + 0.30, "maximize F1", ha="center", va="center",
        color=pal["sky"], fontsize=9.5)
ax.add_patch(FancyBboxPatch((4.65, by), 4.25, 1.0, boxstyle="round,pad=0.05",
                            facecolor=pal["panel"], edgecolor=pal["blue"], lw=1.3))
ax.text(6.775, by + 0.68, "SATISFICE the rest", ha="center", va="center",
        color=pal["navy"], fontweight="bold", fontsize=11)
ax.text(6.775, by + 0.30, "constraints to clear, e.g. latency under 100 ms",
        ha="center", va="center", color=pal["gray"], fontsize=9.5)
ax.text(8.9, by - 0.35, "illustration", ha="right", va="center",
        color=pal["gray"], fontsize=8.5, style="italic")
ax.set_xlim(-0.2, 9.1)
ax.set_ylim(by - 0.6, top + row_h + 0.6)
ax.axis("off")
fig.savefig(f"{FIGS_THEORY}/single_number_scoreboard.png")
plt.close(fig)

# ---- Chart T4 (theory): selection vs assessment, Frame C bar beneath ----
# (canonical bar from shared_figs - same drawing as Decks 2 and 3)
fig, (axt, axb) = plt.subplots(2, 1, figsize=(7.5, 3.9),
                               gridspec_kw={"height_ratios": [1.15, 1.0],
                                            "hspace": 0.62})
jobs = [
    (0.0, pal["panel"], pal["navy"], "SELECTION - choosing", "compare settings on validation / CV\nrun it as many times as you like"),
    (4.65, pal["navy"], "white", "ASSESSMENT - grading", "one look at the untouched test set\na score you never tuned toward"),
]
for x, fc, hc, head, sub in jobs:
    axt.add_patch(FancyBboxPatch((x, 0.0), 4.25, 1.5, boxstyle="round,pad=0.05",
                                 facecolor=fc, edgecolor=pal["blue"], lw=1.4))
    axt.text(x + 2.125, 1.05, head, ha="center", va="center",
             color=hc, fontweight="bold", fontsize=12)
    axt.text(x + 2.125, 0.48, sub, ha="center", va="center",
             color=pal["gray"] if fc == pal["panel"] else pal["sky"], fontsize=9)
axt.text(2.125, 1.78, "different jobs, different data", ha="center", va="center",
         color=pal["gray"], fontsize=9, style="italic")
axt.text(6.775, 1.78, "spent the moment you use it", ha="center", va="center",
         color=pal["gray"], fontsize=9, style="italic")
axt.set_xlim(-0.2, 9.1)
axt.set_ylim(-0.25, 2.05)
axt.axis("off")
draw_frame_c(axb)
fig.savefig(f"{FIGS_THEORY}/select_vs_assess.png")
plt.close(fig)

# ---- Chart C: two passengers from the loaded pipeline (NB3 sec 7) ----
fig, ax = plt.subplots(figsize=(7.5, 3.6))
labels = ["Third-class man, 28\n(Fare 7.9, Southampton)\na passenger like Owen",
          "First-class woman, 30\n(Fare 80, Cherbourg)\na passenger like Florence"]
vals = [0.03, 1.00]
bars = ax.barh(labels, vals, color=[pal["sky"], pal["blue"]], height=0.5)
for b, v in zip(bars, vals):
    ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center",
            fontsize=13, fontweight="bold", color=pal["navy"])
ax.set_xlim(0, 1.14)
ax.set_xlabel("Predicted survival probability")
ax.set_title("Same model, two passengers")
fig.savefig(f"{FIGS}/two_passengers.png")
plt.close(fig)

# ---- Chart D (story): the production loop, circular (NB3 sec 6) ----
fig, ax = plt.subplots(figsize=(8.2, 4.9))
nodes = [
    ("1. Save", "joblib: the whole\npipeline, versioned v1", (0.0, 2.05), "blue"),
    ("2. Load", "fresh process,\nno retraining", (3.25, 1.02), "panel"),
    ("3. Wrap", "an interface people\ncan use: Gradio", (3.25, -1.02), "panel"),
    ("4. Host", "Hugging Face Spaces:\na public URL", (0.0, -2.05), "panel"),
    ("5. Monitor", "the world drifts,\nmodels go stale", (-3.25, -1.02), "panel"),
    ("6. Retrain", "new data, new\nversion, roll out", (-3.25, 1.02), "navy"),
]
bw, bh = 2.15, 1.10
for head, sub, (cx, cy), kind in nodes:
    fc = {"panel": pal["panel"], "blue": pal["blue"], "navy": pal["navy"]}[kind]
    txt = pal["navy"] if kind == "panel" else "white"
    sub_c = pal["gray"] if kind == "panel" else pal["sky"]
    ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                                boxstyle="round,pad=0.05", facecolor=fc,
                                edgecolor=pal["blue"], lw=1.3))
    ax.text(cx, cy + 0.22, head, ha="center", va="center",
            fontweight="bold", color=txt, fontsize=11.5)
    ax.text(cx, cy - 0.24, sub, ha="center", va="center",
            color=sub_c, fontsize=8.4)
# clockwise arrows anchored on box edges so every head stays visible
arrows = [((1.20, 1.92), (2.98, 1.70)),    # Save -> Load
          ((3.25, 0.34), (3.25, -0.34)),   # Load -> Wrap
          ((2.98, -1.70), (1.20, -1.92)),  # Wrap -> Host
          ((-1.20, -1.92), (-2.98, -1.70)),  # Host -> Monitor
          ((-3.25, -0.34), (-3.25, 0.34)),   # Monitor -> Retrain
          ((-2.98, 1.70), (-1.20, 1.92))]    # Retrain -> Save (loop closes)
for a, b in arrows:
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle="arc3,rad=-0.18",
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=pal["blue"], lw=1.6,
                                 shrinkA=2, shrinkB=2))
ax.text(0, 0.22, "The production loop", ha="center", va="center",
        fontweight="bold", color=pal["navy"], fontsize=13)
ax.text(0, -0.24, "a shipped model is never done -\nretraining starts the loop again",
        ha="center", va="center", color=pal["gray"], fontsize=9)
ax.set_xlim(-4.85, 4.85)
ax.set_ylim(-2.95, 2.95)
ax.axis("off")
fig.savefig(f"{FIGS_STORY}/s4_production_loop.png")
plt.close(fig)

# ---- Chart F (theory): Ng's orthogonalization ladder (DLS-C3) ----
fig, ax = plt.subplots(figsize=(7.4, 4.9))
rungs = [  # bottom to top: rung, its own knobs, TODAY marker
    ("1. Learn the training data", "knob: bigger / better model", False),
    ("2. Pass the practice tests (CV)", "knob: rein the model in or add data", True),
    ("3. Hold up on the final exam", "knob: a bigger, fresher batch of practice tests", False),
    ("4. Work in the real world", "knob: fix the data or the metric", True),
]
# ladder rails
for rx in (0.55, 6.05):
    ax.add_patch(FancyBboxPatch((rx, 0.15), 0.13, 5.45,
                                boxstyle="round,pad=0.02",
                                facecolor=pal["sky"], edgecolor="none"))
for i, (label, knob, today) in enumerate(rungs):
    y = 0.45 + i * 1.35
    fc = pal["blue"] if today else pal["panel"]
    txt = "white" if today else pal["navy"]
    sub = pal["sky"] if today else pal["gray"]
    ax.add_patch(FancyBboxPatch((0.75, y), 5.2, 0.95,
                                boxstyle="round,pad=0.04", facecolor=fc,
                                edgecolor=pal["blue"], lw=1.3))
    ax.text(3.35, y + 0.62, label, ha="center", va="center",
            fontweight="bold", color=txt, fontsize=12)
    ax.text(3.35, y + 0.27, knob, ha="center", va="center",
            color=sub, fontsize=9.5)
    if today:
        ax.add_patch(FancyBboxPatch((6.45, y + 0.28), 1.05, 0.42,
                                    boxstyle="round,pad=0.04",
                                    facecolor=pal["navy"], edgecolor="none"))
        ax.text(6.97, y + 0.49, "TODAY", ha="center", va="center",
                color="white", fontweight="bold", fontsize=9.5)
ax.add_patch(FancyArrowPatch((0.18, 0.55), (0.18, 5.55), arrowstyle="-|>",
                             mutation_scale=18, color=pal["blue"], lw=2.0))
ax.text(0.02, 3.05, "climb the rungs in order", rotation=90, ha="center",
        va="center", color=pal["gray"], fontsize=9.5)
ax.annotate("start here", xy=(6.25, 0.92), xytext=(6.6, 0.92),
            ha="left", va="center", color=pal["gray"], fontsize=9.5,
            arrowprops=dict(arrowstyle="-|>", color=pal["gray"], lw=1.2))
ax.text(3.35, 6.0, "One knob per rung - never turn them all at once",
        ha="center", va="center", fontweight="bold", color=pal["navy"],
        fontsize=12.5)
ax.set_xlim(-0.35, 7.75)
ax.set_ylim(0.0, 6.35)
ax.axis("off")
fig.savefig(f"{FIGS_THEORY}/orthogonalization_ladder.png")
plt.close(fig)

# ---- Chart G (theory): Ng's error-analysis spreadsheet, generic mock ----
fig, ax = plt.subplots(figsize=(7.2, 4.4))
cols = ["Misclassified\nexample", "Blurry", "Mislabeled", "Confusing\nsubclass"]
col_x = [0.0, 2.6, 4.4, 6.2]
col_w = [2.6, 1.8, 1.8, 1.9]
marks = [  # X = this error falls in that category (illustrative)
    ("error #1", True, False, False),
    ("error #2", False, True, False),
    ("error #3", True, False, False),
    ("error #4", True, True, False),
    ("error #5", False, False, True),
]
row_h, top = 0.62, 5.0
# highlighted category column (the 50% one - where the month goes);
# soft fill only - per-cell borders keep the grid clean
# header row
for x, w, c in zip(col_x, col_w, cols):
    ax.add_patch(plt.Rectangle((x, top), w, row_h, facecolor=pal["navy"],
                               edgecolor="white", lw=1.0, zorder=2))
    ax.text(x + w / 2, top + row_h / 2, c, ha="center", va="center",
            color="white", fontweight="bold", fontsize=10, zorder=3)
# example rows
for r, (name, *flags) in enumerate(marks, start=1):
    y = top - r * row_h
    for x, w in zip(col_x, col_w):
        cell_fc = pal["panel"] if x == col_x[1] else "white"
        ax.add_patch(plt.Rectangle((x, y), w, row_h, facecolor=cell_fc,
                                   edgecolor=pal["sky"], lw=0.8, zorder=1))
    ax.text(col_x[0] + 0.15, y + row_h / 2, name, ha="left", va="center",
            color=pal["ink"], fontsize=10, zorder=3)
    for flag, x, w in zip(flags, col_x[1:], col_w[1:]):
        if flag:
            ax.text(x + w / 2, y + row_h / 2, "X", ha="center", va="center",
                    color=pal["blue"], fontweight="bold", fontsize=12, zorder=3)
# continuation row: same cell borders as the example rows
y6 = top - 6 * row_h
for x, w in zip(col_x, col_w):
    cell_fc = pal["panel"] if x == col_x[1] else "white"
    ax.add_patch(plt.Rectangle((x, y6), w, row_h, facecolor=cell_fc,
                               edgecolor=pal["sky"], lw=0.8, zorder=1))
ax.text(col_x[0] + 0.15, top - 6 * row_h + row_h / 2,
        "... tally continues to ~100 ...", ha="left", va="center",
        color=pal["gray"], fontsize=9.5, style="italic", zorder=3)
# totals row: the bottom row of percentages IS the priority list
ty = top - 7 * row_h - 0.30
for x, w in zip(col_x, col_w):
    ax.add_patch(plt.Rectangle((x, ty), w, row_h, facecolor=pal["panel"],
                               edgecolor=pal["blue"], lw=1.0, zorder=2))
for x, w, v in zip(col_x, col_w, ["% of all errors", "50%", "30%", "5%"]):
    ax.text(x + w / 2 if v != "% of all errors" else x + 0.15,
            ty + row_h / 2, v, ha="center" if v != "% of all errors" else "left",
            va="center", color=pal["navy"], fontweight="bold", fontsize=11,
            zorder=3)
ax.annotate("where the month goes", xy=(col_x[1] + col_w[1] / 2, ty - 0.12),
            xytext=(col_x[1] + col_w[1] / 2 - 0.3, ty - 0.75), ha="center",
            color=pal["blue"], fontweight="bold", fontsize=10.5,
            arrowprops=dict(arrowstyle="-|>", color=pal["blue"], lw=1.3))
ax.annotate("payoff capped near zero - skip", xy=(col_x[3] + col_w[3] / 2, ty - 0.12),
            xytext=(col_x[3] + col_w[3] / 2 - 0.4, ty - 0.75), ha="center",
            color=pal["gray"], fontsize=10,
            arrowprops=dict(arrowstyle="-|>", color=pal["gray"], lw=1.1))
ax.text(0.0, top + row_h + 0.35, "The bottom row IS the priority list",
        ha="left", va="center", fontweight="bold", color=pal["navy"],
        fontsize=12.5)
ax.text(0.0, ty - 1.05, "sample rows shown; categories can overlap - "
        "totals from the full ~100", ha="left", va="top", fontsize=8.5,
        style="italic", color=pal["gray"])
ax.set_xlim(-0.2, 8.3)
ax.set_ylim(ty - 1.45, top + row_h + 0.7)
ax.axis("off")
fig.savefig(f"{FIGS_THEORY}/error_analysis_sheet.png")
plt.close(fig)

# ---- Chart H (theory): MIT's end-to-end evaluation pipeline ----
fig, ax = plt.subplots(figsize=(8.6, 4.3))
steps = [
    ("1. Compare", "variants by CV", "panel"),
    ("2. Pick", "the winner", "panel"),
    ("3. RETRAIN", "on ALL the data", "blue"),
    ("4. Ship", "the refit recipe", "panel"),
    ("5. Judged", "on unseen data", "navy"),
]
bw, bh, gap, y0 = 1.62, 1.05, 0.42, 1.6
for i, (head, sub, kind) in enumerate(steps):
    x = 0.2 + i * (bw + gap)
    fc = {"panel": pal["panel"], "blue": pal["blue"], "navy": pal["navy"]}[kind]
    txt = pal["navy"] if kind == "panel" else "white"
    sub_c = pal["gray"] if kind == "panel" else pal["sky"]
    lift = 0.16 if kind == "blue" else 0.0  # the surprise step stands proud
    ax.add_patch(FancyBboxPatch((x, y0 + lift), bw, bh,
                                boxstyle="round,pad=0.05", facecolor=fc,
                                edgecolor=pal["blue"],
                                lw=2.2 if kind == "blue" else 1.3))
    ax.text(x + bw / 2, y0 + lift + 0.70, head, ha="center", va="center",
            fontweight="bold", color=txt, fontsize=12)
    ax.text(x + bw / 2, y0 + lift + 0.33, sub, ha="center", va="center",
            color=sub_c, fontsize=9.5)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + bw + 0.06, y0 + 0.52),
                                     (x + bw + gap - 0.06, y0 + 0.52),
                                     arrowstyle="-|>", mutation_scale=15,
                                     color=pal["blue"], lw=1.6))
ax.text(0.2 + 2 * (bw + gap) + bw / 2, y0 + 1.55,
        "the fold models were scaffolding -\nmore data = a better final model",
        ha="center", va="center", color=pal["blue"], fontsize=9.5,
        fontweight="bold")
# the two forever-forbidden contamination arrows (red, crossed out):
# two nested arcs below the pipeline - the short validation arc in a shallow
# upper lane, the full-width test arc diving deeper; they never cross
x1c = 0.2 + bw / 2                      # Compare, center
x2c = 0.2 + 1 * (bw + gap) + bw / 2     # Pick, center
x5a = 0.2 + 4 * (bw + gap) + 0.43       # Judged, bottom-left anchor
x5b = 0.2 + 4 * (bw + gap) + bw - 0.43  # Judged, bottom-right anchor
# shallow arc: validation score (Pick) reused as the report card (Judged)
ax.add_patch(FancyArrowPatch((x2c, y0 - 0.08), (x5a, y0 - 0.08),
                             connectionstyle="arc3,rad=0.15",
                             arrowstyle="-|>", mutation_scale=15,
                             color=RED, lw=1.7, linestyle=(0, (5, 3))))
ax.text(x2c + 0.75 * (x5a - x2c), y0 - 0.40, "X", ha="center", va="center",
        color=RED, fontsize=17, fontweight="bold",
        bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=RED, lw=1.4))
ax.text(5.0, y0 - 0.76, "validation score reused as the report card",
        ha="center", va="center", color=RED, fontsize=10, fontweight="bold")
# deep arc: test data (Judged) flowing back into training (Compare)
ax.add_patch(FancyArrowPatch((x5b, y0 - 0.08), (x1c, y0 - 0.08),
                             connectionstyle="arc3,rad=-0.32",
                             arrowstyle="-|>", mutation_scale=15,
                             color=RED, lw=1.7, linestyle=(0, (5, 3))))
ax.text((x5b + x1c) / 2, y0 - 1.45, "X", ha="center", va="center", color=RED,
        fontsize=17, fontweight="bold",
        bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=RED, lw=1.4))
ax.text((x5b + x1c) / 2, y0 - 1.85, "test data flowing into training",
        ha="center", va="center", color=RED, fontsize=10, fontweight="bold")
ax.set_xlim(0, 10.5)
ax.set_ylim(y0 - 2.1, y0 + 2.15)
ax.axis("off")
fig.savefig(f"{FIGS_THEORY}/retrain_pipeline.png")
plt.close(fig)

# ---- Chart E (concepts): echo of the Session 1 hero contrast, compact
# two-line strip version (same motif, sized as a header above the table) ----
fig, ax = plt.subplots(figsize=(12.0, 1.3))
rows = [
    (1.05, pal["panel"], pal["navy"],
     "M1 - descriptive analytics: \"38.4% survived\" - a chart a person "
     "reads, then decides."),
    (0.10, pal["blue"], "white",
     "M2 - machine learning: \"3% for a passenger like Owen\" - answered "
     "on demand by a program."),
]
for y0, fc, hc, head in rows:
    ax.add_patch(FancyBboxPatch((0.9, y0), 11.0, 0.78,
                                boxstyle="round,pad=0.04",
                                facecolor=fc, edgecolor=pal["blue"], lw=1.4))
    ax.text(1.15, y0 + 0.39, head, ha="left", va="center", color=hc,
            fontweight="bold", fontsize=11)
ax.add_patch(FancyArrowPatch((0.55, 1.62), (0.55, 0.38), arrowstyle="-|>",
                             mutation_scale=16, color=pal["navy"], lw=1.8))
ax.text(0.32, 1.0, "same 891\npassengers", ha="right", va="center",
        color=pal["navy"], fontsize=8.5, fontweight="bold")
ax.set_xlim(-1.0, 12.1)
ax.set_ylim(0.0, 1.95)
ax.axis("off")
fig.savefig(f"{FIGS_CONCEPTS}/echo_da_vs_ml_strip.png")
plt.close(fig)

# === SLIDES ===
prs = ds.new_deck()

# 1. Title
ds.title_slide(prs, "Session 4 of 4", "Tuning and Shipping the Model",
               "Session 3's Random Forest gets tuned, the test set gets its "
               "one look, and the model leaves the notebook: a public app "
               "anyone can ask about a passenger like Owen or Florence.")

# 2. Where we are + Ng's orthogonalization ladder (REPLACED per plan)
s = ds.image_slide(
    prs, "Tune one knob at a time: four rungs, each with its own knobs",
    f"{FIGS_THEORY}/orthogonalization_ladder.png",
    kicker="Where we are - Ng's orthogonalization",
    bullets=[
        "Orthogonalization - in plain words: one knob does one job; a "
        "radio whose knobs all do everything is untunable.",
        "How to read the ladder: climb from the bottom rung; each rung is "
        "one problem, fixed by its OWN knobs - never someone else's.",
        "Where we are: a leak-proof pipeline, 712 training / 179 test "
        "rows, 18 features.",
        "LogReg won Session 3 (CV 0.819, test 0.838) but has no knobs "
        "left; the Random Forest (CV 0.797) has the most - today's "
        "candidate.",
        "Diagnose WHICH rung you are failing before touching any knob: "
        "compare the training score with the practice-test (CV) score.",
        "TODAY tags our rungs: pass the practice tests (tuning), then "
        "work in the real world (shipping).",
    ],
    caption="Ng's orthogonalization ladder: one job per knob. \"Practice "
            "tests\" = 5-fold cross-validation from Session 3. Ng, Deep "
            "Learning Specialization C3 \"Structuring ML Projects\".")
notes(s, "Orthogonalization is a long word for a simple demand: one knob, "
         "one job. A radio where every knob changes volume, station and "
         "tone at once cannot be tuned - and turning random ML knobs is "
         "that radio. Walk the ladder bottom-up: learn the training data, "
         "pass the practice tests (that is cross-validation from Session "
         "3), hold up on the final exam, work in the real world. First "
         "name the rung you are failing, then reach for that rung's knob "
         "only. Ask the class: your model aces training but flops on the "
         "practice tests - which rung, which knob?")

# 3. Section divider: Part 1 - theory
ds.section_slide(prs, "01", "Part 1 - The theory: tuning and judging "
                 "without fooling yourself",
                 "Six ideas that hold for every model you will ever tune - "
                 "the referee stays cross-validation, and the test set stays "
                 "sealed.")

# 4. Parameters vs hyperparameters
s = ds.two_col_slide(
    prs, "The model learns its parameters - you set the hyperparameters before training",
    ("Parameters: the model learns them",
     ["Learned from the data during training - fit() finds them.",
      "Example: the split rules inside every tree of the forest.",
      "There are thousands; you never set one by hand.",
      "In the kitchen: how the ingredients turn into cake."]),
    ("Hyperparameters: you choose them",
     ["Chosen by you before training starts.",
      "Examples: n_estimators (how many trees), max_depth (how deep).",
      "Tuning = try different knob positions, keep the best result.",
      "In the kitchen: the oven temperature - set before baking.",
      "MIT's sharper cut: they sit OUTSIDE Session 1's three-slot recipe "
      "(model, loss, optimizer).",
      "They define the game the learner plays - before it starts.",
      "Same algorithm, two settings = two different algorithms. "
      "MIT 6.390, Appendix C."]),
    kicker="Theory - two kinds of settings",
    note="Baking analogy: the oven temperature is a hyperparameter you set; how the "
         "ingredients turn into cake is the parameters.")
notes(s, "The kitchen carries this one: the oven temperature is a "
         "hyperparameter - you set it before baking; what happens to the "
         "ingredients inside is the parameters - the process finds those "
         "itself. You never hand-set a parameter, and the model never "
         "chooses a hyperparameter. Ask the class: max_depth of a tree - "
         "oven dial or cake batter?")

# 5. NEW theory: tuning IS comparing algorithms (MIT reframe)
s = ds.image_slide(
    prs, "Tuning is not a new skill - two settings of one algorithm are two "
         "different algorithms",
    f"{FIGS_THEORY}/two_settings_one_referee.png",
    kicker="Theory - tuning is comparing",
    bullets=[
        "Tuning - in plain words: Session 3's tournament again, now "
        "between versions of one model.",
        "How to read it: two settings walk in; one referee - the same 5 "
        "practice tests (cross-validation) - picks the winner.",
        "MIT's reframe: a forest capped at depth 4 and a forest allowed "
        "depth 8 are two different algorithms - same name, different game.",
        "So tuning is a problem you already solved: Session 3 compared "
        "six model families; tuning compares variants of one.",
        "The referee never changes: each variant is scored on rows it "
        "never trained on; the best score wins.",
        "Harvard files hyperparameter tuning under model selection, next "
        "to choosing predictors: one discipline, not two.",
    ],
    caption="Two settings of one forest, judged by the same referee - "
            "illustration. MIT 6.390, Appendix C; Harvard CS109A.")
notes(s, "Deflate the new word: tuning is nothing but Session 3's "
         "tournament run again, with versions of one model as the "
         "contestants instead of six different families. MIT's reframe "
         "makes it legitimate: two settings really are two algorithms. Same "
         "referee, same rules, new entrants. Ask the class: if you already "
         "trust the tournament from last week, what is actually new today?")

# 6. NEW theory: why random search beats a grid when knobs differ in importance
s = ds.image_slide(
    prs, "Knobs are not equally important - random search tests nine values "
         "of the one that matters, a grid tests three",
    f"{FIGS_THEORY}/grid_vs_random.png",
    kicker="Theory - search strategy",
    bullets=[
        "How to read it: each dot = one try; darker blue = better score; "
        "the ticks below count the distinct values tried.",
        "Grid vs random - in plain words: a grid re-tests the same few "
        "values; random tries a new value every time.",
        "Typically one or two knobs dominate the score - and you do not "
        "know in advance which.",
        "The grid's nine tries line up in three columns: only three "
        "values of the knob that matters.",
        "Nine random tries scatter: nine values of that same knob.",
        "Same budget, more information - that is why random search "
        "usually wins beyond a couple of knobs.",
        "The referee never changes: every try is scored by the same "
        "cross-validation.",
    ],
    caption="Nine trials spent two ways on a landscape where one knob "
            "matters - illustration.")
notes(s, "The picture does the arguing: both panels spend nine trials, but "
         "the grid lines its shots up in three columns, so the knob that "
         "matters only ever sees three values - the ticks under the axis "
         "count them. Random scatters, so the important knob gets nine "
         "different values for the same budget. Ask the class: you get 20 "
         "trials and five knobs - grid or random, and why?")

# 7. Ng's error analysis: read the mistakes before turning more knobs
s = ds.image_slide(
    prs, "Before turning more knobs, read the mistakes - an hour of counting "
         "beats a week of guessing",
    f"{FIGS_THEORY}/error_analysis_sheet.png",
    kicker="Theory - error analysis",
    bullets=[
        "How to read it: each row = one wrong prediction; each X = the "
        "category it falls into; the bottom row totals the percentages.",
        "The method: pull ~100 wrong predictions, look at them by hand, "
        "tally them in a spreadsheet.",
        "The bottom row IS the priority list: a 5% category cannot repay "
        "a month of work; the 50% one (shaded) is where the month goes.",
        "Guard: inspect only an \"Eyeball\" sample; keep a \"Blackbox\" "
        "part you never look at - you overfit whatever you stare at.",
        "Exercise: tally our champion's wrong test predictions (family "
        "size? fare band? title?) - count first, fix second.",
    ],
    caption="The method, on Ng's classic image-classifier example - "
            "illustration, not our data. Ng, Machine Learning Yearning "
            "chs. 14-19.")
notes(s, "Error analysis is deliberately low-tech: open the mistakes, look "
         "at them by hand, and tally them into categories in a spreadsheet. "
         "The bottom row of percentages is the priority list - a 5% "
         "category cannot pay for a month of work, a 50% category is the "
         "month. Warn them about the guard: you will overfit whatever you "
         "stare at, hence the Eyeball/Blackbox split. Ask the class: why "
         "count first and fix second, not the other way round?")

# 8. NEW theory: the single-number metric doctrine (Ng)
s = ds.image_slide(
    prs, "Agree on one evaluation number before you experiment - it makes "
         "every comparison instant",
    f"{FIGS_THEORY}/single_number_scoreboard.png",
    kicker="Theory - one number to steer by",
    bullets=[
        "The deadlock in the table: A wins precision, B wins recall - "
        "two numbers cannot rank two models.",
        "The blue F1 column combines both into one number - and B wins. "
        "Deadlock broken.",
        "The doctrine - in plain words: one scoreboard; teams argue "
        "forever with two scoreboards.",
        "F1 punishes imbalance: precision 0.9 with recall 0.1 scores "
        "near 0.18, not 0.5 - a useless side cannot hide.",
        "Something else matters too (safety, speed)? SATISFICE it: a bar "
        "to clear, not a score to chase. Maximize exactly ONE metric.",
        "Set the metric and the validation data at project start: they "
        "define \"better\" for the whole team.",
    ],
    caption="A deadlocked scoreboard resolved by one number - illustration. "
            "Ng, Machine Learning Yearning chs. 8-9.")
notes(s, "One scoreboard: with a single agreed number, ten experiments rank "
         "themselves; with three numbers, every comparison becomes a "
         "meeting. When something else genuinely matters - safety, speed - "
         "it becomes a bar to clear, not a second score to chase; exactly "
         "one number stays on the scoreboard. Ask the class: A wins "
         "precision, B wins recall - without a combiner, who ships?")

# 9. NEW theory: selection vs assessment - why REPORT opens exactly once
s = ds.image_slide(
    prs, "Choosing and grading are different jobs - CHOOSE as often as you "
         "like, REPORT opens once",
    f"{FIGS_THEORY}/select_vs_assess.png",
    kicker="Theory - selection vs assessment",
    bullets=[
        "In plain words: practice tests choose your strategy; the final "
        "exam grades it - you sit the final once.",
        "SELECTION picks the knob values - done on the practice tests "
        "(CV), repeatable a thousand times.",
        "ASSESSMENT estimates real-world performance - done exactly "
        "once, on the untouched test set.",
        "How to read the bar (from Session 2): pale = 712 training rows "
        "(FIT); hatched = practice tests (CHOOSE); dark = 179 sealed "
        "test rows (REPORT).",
        "Select on the test set and it silently becomes a practice test: "
        "its score flatters.",
        "A number you optimized against is a target you hit, not a "
        "forecast - so REPORT opens once.",
    ],
    caption="Two jobs over the same verbs bar: selection lives left of the "
            "seal; assessment spends it. Harvard CS109A / ISLR ch. 5.")
notes(s, "Close the theory on the exam metaphor: practice tests choose your "
         "strategy - retake them all you like; the final exam grades it - "
         "and you sit it once. The moment you pick your strategy using the "
         "final's questions, the final stops measuring anything. Ask the "
         "class: why does a score you optimized against stop being a "
         "forecast?")

# 10. Section divider: Part 2 - practice
ds.section_slide(prs, "02", "Part 2 - Practice: tune it, prove it, ship it",
                 "Grid and random search for real, one honest look at the "
                 "test set, then the model leaves the notebook - a public app "
                 "anyone can ask about a passenger like Owen or Florence.")

# 11. Practice: grid + random search around the tuning-arc chart
s = ds.image_slide(
    prs, "Grid and random search lift the forest from 0.7937 to 0.8274 - "
         "random wins with fewer tries",
    f"{FIGS}/tuning_arc.png",
    kicker="Practice - grid and random search",
    bullets=[
        "How to read it: each bar = the forest's practice-test (CV) score "
        "at one stage; the y-axis starts at 0.75, so gaps are zoomed.",
        "Grid search = CV plus for-loops, no magic: 18 combos x 5 folds = "
        "90 trainings. Best: CV 0.8259 (+0.0322 over the default).",
        "Grids explode: 5 knobs x 6 values = 7,776 combos - nearly "
        "39,000 trainings.",
        "15 random draws test up to 15 values of every knob: CV 0.8274 - "
        "beats the grid with fewer tries, as slide 6 predicted.",
        "All 33 candidates sat the same 5 practice tests; the test set "
        "stayed sealed.",
        "The winning knobs (capped depth, bigger leaves) are complexity "
        "brakes: less freedom to memorize, so it generalizes better.",
    ],
    caption="Random Forest 5-fold CV accuracy at each tuning stage. "
            "Module 2, notebook 3.")
notes(s, "Theory made real: 18 grid combos and 15 random draws all ran "
         "Session 3's tournament again, refereed by the same 5-fold CV, "
         "test set sealed. Grid best: max_depth 4, min_samples_leaf 5, 100 "
         "trees. Random found a better setting with fewer tries - exactly "
         "the slide-6 prediction. Read the winning knobs out loud: capped "
         "depth and bigger leaves are brakes - 'regularization' is the "
         "textbook word for such brakes. A little bias buys a big drop in "
         "variance: Session 3's U-curve, found automatically. Ask the "
         "class: why is it no surprise the winner has LESS freedom than "
         "the default forest?")

# 12. Practice: the one test look
s = ds.image_slide(
    prs, "One look at 179 unseen passengers: 0.8268 - within 0.0006 of CV, "
         "the tuning did not fool itself",
    f"{FIGS}/cv_vs_test_bar.png",
    kicker="Practice - the honest final exam",
    bullets=[
        "How to read it: left bar = what the practice tests (CV) "
        "promised; right bar = the one real exam. Y-axis starts at 0.75.",
        "Final exam: 179 passengers the model has never seen, scored one "
        "time: accuracy 0.8268, F1 0.7597.",
        "CV promised 0.8274; the sealed test answered 0.8268 - a search "
        "judged honestly generalizes.",
        "Peeking at test scores while tuning silently turns the test set "
        "into training data.",
        "Honest expectation: tuning buys a percentage point or two, not "
        "miracles.",
        "0.8268 is the only forecast of real-world performance we own.",
    ],
    caption="Tuned forest: CV score vs the single test-set look. Module 2, "
            "notebook 3.")
notes(s, "The final exam, sat once: the practice tests promised 0.8274 and "
         "the sealed final answered 0.8268 - a gap of 0.0006. That "
         "agreement is the reward for never peeking: a search judged "
         "honestly generalizes. Also set expectations: tuning bought about "
         "one percentage point, not a miracle. Ask the class: if we had "
         "checked the test score after every tuning try, what would this "
         "chart look like - and could we still trust it?")

# 13. MIT's retrain-on-all-data pipeline
s = ds.image_slide(
    prs, "You don't ship the CV copy: after choosing, retrain the winning "
         "recipe on all the data",
    f"{FIGS_THEORY}/retrain_pipeline.png",
    kicker="The last step before shipping",
    bullets=[
        "How to read it: five steps, left to right; the raised blue box "
        "is the surprise step; dashed red arrows = forbidden moves.",
        "MIT's five steps: compare by CV, pick the winner, RETRAIN on all "
        "data, ship, be judged on unseen data.",
        "Step 3 surprises beginners: the fold models were scaffolding; "
        "more data = a better final model, so refit everything.",
        "sklearn already does this on the training set: refit=True "
        "retrains the best settings on all 712 rows - the model we saved.",
        "After the one test look, trust the recipe, not the score: "
        "retraining on all 891 rows is legitimate; re-scoring is not - no "
        "honest data remains.",
        "Forbidden forever: test data flowing into training, and a "
        "validation score reused as the report card.",
    ],
    caption="MIT's end-to-end evaluation pipeline - the two red arrows are "
            "never allowed. MIT 6.390, Appendix C.")
notes(s, "Walk the boxes left to right; the raised blue one is the step "
         "beginners never guess: after CV has chosen the winner, throw the "
         "fold models away and retrain the winning RECIPE on every row you "
         "own - more data makes a better final model. Then the two red "
         "arrows: test data may never flow back into training, and a score "
         "you selected with may never be your report card. Ask the class: "
         "after retraining on all 891 rows, why can't we measure the new "
         "model's accuracy?")

# 14. What production means - the loop diagram
s = ds.image_slide(
    prs, "Production is a loop, not a finish line - the model runs where "
         "real users can reach it",
    f"{FIGS_STORY}/s4_production_loop.png",
    kicker="From notebook to production",
    bullets=[
        "How to read it: six steps, arrows run clockwise; after "
        "6. Retrain you are back at 1. Save - the loop never ends.",
        "Save the WHOLE pipeline (titanic_model_v1.joblib, 2,599 KB): raw "
        "data in, prediction out - re-coding the prep by hand invites "
        "silently wrong predictions.",
        "Load in a fresh process: no retraining, no notebook required.",
        "Version the file name (v1) so you always know what is running - "
        "and can roll back.",
        "Ng's rollout for steps 4-5: SHADOW (predictions logged, unused) "
        "-> CANARY (~5% of traffic) -> full swap, rollback ready.",
        "Monitoring closes the loop: drift detected means retrain and "
        "reship.",
    ],
    caption="The production loop: save, load, wrap, host, monitor, retrain. "
            "Module 2, notebook 3. Ng, Machine Learning Engineering for "
            "Production C1.")
notes(s, "Production, in plain words: the model runs somewhere real users "
         "can reach it. Follow the circle clockwise from 1. Save - the "
         "blue box, today's first step; 6. Retrain is dark because it "
         "closes the circle and starts it again. Stress Ng's rollout "
         "ladder: shadow first (the model watches, nobody acts on it), "
         "canary next (a sliver of real traffic), full swap last - "
         "deployment is a dial, not a switch. Ask the class: why is there "
         "no 'done' box anywhere on this diagram?")

# 15. Two passengers - the moment it becomes real
s = ds.image_slide(
    prs, "The loaded pipeline answers a real question - a survival gap of 100% vs 3%",
    f"{FIGS}/two_passengers.png",
    kicker="Sanity check on two passengers",
    bullets=[
        "How to read it: each bar = the predicted survival chance the "
        "loaded model gives one invented passenger.",
        "Loaded from disk, no retraining: raw passenger details in, "
        "verdict out.",
        "An invented first-class woman, 30 (Fare 80, Cherbourg) - a "
        "passenger like Florence: survives, 100%.",
        "An invented third-class man, 28 (Fare 7.9, Southampton) - a "
        "passenger like Owen: does not survive, 3%.",
        "A probability is confidence, not a guarantee about one "
        "individual.",
        "The gap reflects the Titanic's reality: class and sex decided "
        "lifeboat seats.",
    ],
    caption="Two invented sanity-check passengers, scored by the saved "
            "pipeline. Module 2, notebook 3.")
notes(s, "The moment it becomes real: a saved file, loaded in a fresh "
         "process, answers about people it has never seen. The two "
         "passengers are invented, but built to resemble the module's "
         "recurring ones - Owen, the third-class man, and Florence, the "
         "first-class woman. Read a probability honestly: 3% means 'of "
         "100 very similar passengers, about 3 survive' - confidence, not "
         "a verdict on one person. Ask the class: why test on invented "
         "passengers instead of rows from the test set?")

# 16. Gradio + Hugging Face Spaces
s = ds.two_col_slide(
    prs, "A few lines of Gradio turn the pipeline into a web app anyone can use",
    ("Gradio: a function becomes a web form",
     ["Gradio - in plain words: a Python library that turns a function "
      "into a web page.",
      "Write a normal function: inputs in, answer out.",
      "Gradio renders it as sliders, dropdowns and a text box.",
      "The app derives the Title feature by rule (boys under 13 -> "
      "Master, men -> Mr).",
      "On Colab, launch() gives a shareable temporary link."]),
    ("Hugging Face Spaces: free hosting",
     ["Spaces - in plain words: a free public host for ML apps.",
      "Four files: app.py, the model, requirements.txt, README.md.",
      "Pin the scikit-learn version in requirements.txt.",
      "Upload the files in the browser - that is the whole deploy.",
      "No tokens or keys in code, ever.",
      "Result: a public URL anyone can open."]),
    kicker="From model to app")
notes(s, "Demystify both names before the clicks: Gradio is just a Python "
         "library that draws a web form around a normal function - inputs "
         "in, answer out. Hugging Face Spaces is just a free public shelf "
         "that runs that form at a URL. Four files uploaded in the browser "
         "and the model is live - no server administration. Two habits to "
         "call out: pin the scikit-learn version so the loaded pipeline "
         "matches the saved one, and never paste tokens or keys into code.")

# 17. After launch: reality checks (Netflix Prize folded in)
s = ds.bullets_slide(
    prs, "After launch, reality bites: models go stale and accuracy alone does not ship",
    [
        ("The $1M Netflix Prize ensemble (about 800 models, a 10% RMSE gain) "
         "was never put into production.",
         ["Engineering cost outweighed the accuracy gain, and the business had "
          "moved to streaming. Netflix tech blog, 2012."]),
        ("Two kinds of drift make models go stale.",
         ["DATA drift: the inputs change - different passengers start arriving.",
          "CONCEPT drift: the hidden rule f itself moves - same inputs, new "
          "outcomes (Session 1's Y = f(X) + ε).",
          "Monitor inputs and outputs separately."]),
        ("Google Flu Trends overestimated flu by more than 50% in 2011-13.",
         ["It once claimed about 11% of the US had flu; CDC data said about 6%.",
          "It was essentially never retrained after 2009. Lazer et al., Science, 2014."]),
        "Version every saved model (model_v1, model_2026_09_06) so you can always roll back.",
        "Retrain on a schedule and re-score; monitoring is step 5 of the loop, not an afterthought.",
    ],
    kicker="After launch")
notes(s, "Two cautionary tales, both real. Netflix paid $1M for a 10% "
         "better model and never shipped it - accuracy alone does not "
         "ship; engineering cost and a business that had moved to "
         "streaming killed it. Google Flu Trends kept running an "
         "essentially untouched 2009 model and drifted to overestimating "
         "flu by more than 50% - claiming about 11% of the US had flu "
         "when CDC data said about 6%. Ask the class: which kind of drift "
         "was Flu Trends - the inputs changing, or the hidden rule "
         "moving?")

# 18. The whole module on one slide (MERGE: echo strip above the stage table)
s14 = ds.table_slide(
    prs, "M1 dashboards inform people - M2 models act inside products: "
         "the whole module in one table",
    ["Stage", "What you built", "The rule"],
    [
        ["Prep (NB1)", "Split 80/20 first; impute, encode, scale; Title + FamilySize; one Pipeline",
         "Split first. Fit on train only. Pipelines always."],
        ["Select (NB2)", "Six-model tournament, 5-fold CV; Dummy floor 0.617; LogReg wins 0.819",
         "Baseline first; pick the metric that fits the problem."],
        ["Tune (NB3)", "Grid + random search: 0.7937 to 0.8274; one test look: 0.8268",
         "Search honestly; the test set is used exactly once."],
        ["Ship (NB3)", "joblib save; Gradio app; Hugging Face Spaces; versioned v1",
         "Save the whole pipeline; monitor - models go stale."],
    ],
    kicker="Module 2 in one slide",
    note="M1 described the past for people to read; M2 learned from it so software "
         "can answer about the future. Module 3 adds the next layer: cloud ML APIs - "
         "consuming models other people have trained and deployed.",
    col_widths=[1.1, 3.4, 3.0])
# fold the Session 1 echo contrast in as a header strip above the table
# (strip is ~1.3in tall at CONTENT_W; table moves below it, no overlap)
for shp in s14.shapes:
    if shp.has_table:
        shp.top = ds.Inches(3.62)
        break
s14.shapes.add_picture(f"{FIGS_CONCEPTS}/echo_da_vs_ml_strip.png",
                       ds.MARGIN, ds.Inches(2.02), width=ds.CONTENT_W)

# 19. Close
ds.close_slide(
    prs, "Session 4 - what to remember",
    [
        "Part 1, the theory: every setting is a different algorithm - tune "
        "with CV, never the test set; random search spends trials where "
        "grids waste them.",
        "Diagnose the failing bar before turning any knob; when knobs stall, "
        "count errors; steer by ONE metric - CHOOSE often, REPORT once.",
        "Part 2, the practice: random search reached CV 0.8274; the one test "
        "look said 0.8268 - the tuning did not fool itself.",
        "Ship the recipe retrained on everything, then the loop: save, load, "
        "wrap, host, monitor - watch for data AND concept drift.",
        "Survived = 0 for Owen, 1 for Florence, 1 for Frankie (the "
        "9-year-old \"Master\") - the odds were never abstract.",
        "Practice now: notebook 03-model-optimization-and-deployment in Colab.",
    ])

out_path = "../m2-session-4-optimization-and-deployment.pptx"
ds.save_deck(prs, out_path, "M2 Session 4 - Optimization and Deployment")
print(f"Slides: {len(prs.slides)}")
print(f"Saved: {os.path.abspath(out_path)}")
