"""Build M2 Session 4 deck: Tuning and Shipping the Model."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ax.set_ylabel("5-fold CV accuracy")
ax.set_title("The tuning arc: 0.7937 to 0.8259 to 0.8274")
ax.text(0.02, 0.97, "y-axis starts at 0.75", transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5, color=pal["gray"])
fig.savefig(f"{FIGS}/tuning_arc.png")
plt.close(fig)

# ---- Frame C: the FIT | CHOOSE | REPORT verbs bar (Harvard/ISLR motif) ----
# One canonical drawing, shared with Decks 2 and 3 via shared_figs.
from shared_figs import draw_frame_c

# ---- Chart B: CV vs one-shot test score, Frame C bar re-shown beneath ----
fig, (ax, axb) = plt.subplots(2, 1, figsize=(7.5, 5.3),
                              gridspec_kw={"height_ratios": [2.9, 1.0],
                                           "hspace": 0.52})
labels = ["Cross-validation\n(training data only)", "Test set\n(179 unseen passengers)"]
vals = [0.8274, 0.8268]
bars = ax.bar(labels, vals, color=[pal["sky"], pal["blue"]], width=0.4)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0012, f"{v:.4f}", ha="center",
            fontsize=12, fontweight="bold", color=pal["navy"])
ax.set_ylim(0.75, 0.842)
ax.set_ylabel("Accuracy")
ax.set_title("CV and test agree: a gap of only 0.0006")
ax.text(0.02, 0.97, "y-axis starts at 0.75", transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5, color=pal["gray"])
draw_frame_c(axb)
fig.savefig(f"{FIGS_THEORY}/cv_vs_test_bar.png")
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
rungs = [  # bottom to top: bar, its own knobs, TODAY marker
    ("1. Fit the training set", "knob: bigger / better model", False),
    ("2. Generalize to validation", "knob: regularize or add data", True),
    ("3. Hold up on the test set", "knob: a bigger, fresher validation set", False),
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
ax.text(0.02, 3.05, "clear the bars in order", rotation=90, ha="center",
        va="center", color=pal["gray"], fontsize=9.5)
ax.text(3.35, 6.0, "One knob per bar - never turn them all at once",
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
     "M1 - descriptive analytics: 38.4% on a chart a person reads, "
     "then decides."),
    (0.10, pal["blue"], "white",
     "M2 - machine learning: 3% for an Owen-like passenger, answered "
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
               "The forest gets tuned, the test set gets its one look, and the "
               "model leaves the notebook: a public app anyone can ask about a "
               "passenger like Owen or Florence.")

# 2. Where we are + Ng's orthogonalization ladder (REPLACED per plan)
ds.image_slide(
    prs, "Tune one knob per problem: four bars, each with its own dials",
    f"{FIGS_THEORY}/orthogonalization_ladder.png",
    kicker="Where we are - Ng's orthogonalization",
    bullets=[
        "Where we are: leak-proof pipeline (712/179, 18 features); LogReg "
        "champion CV 0.819 / test 0.838, no knobs left; Random Forest 0.797 "
        "has the most knobs - today's candidate.",
        "Ng's tuning philosophy: clear four bars in order - fit the training "
        "set, generalize to validation, hold up on test, work in the real "
        "world - and each bar has its OWN knobs.",
        "Failing training: bigger/better model. Failing validation: "
        "regularize (= penalize complexity) or add data. Failing test: your "
        "validation was over-tuned. Failing the world: your data or metric "
        "misses reality.",
        "Diagnose WHICH bar you are failing (Session 3's two numbers) before "
        "touching any knob - turning random knobs at once is an untunable radio.",
        "Today: the validation bar's knobs, then the world bar.",
    ],
    caption="Ng's orthogonalization ladder: one job per knob. Ng, Deep "
            "Learning Specialization C3 \"Structuring ML Projects\".")

# 3. Section divider: tuning
ds.section_slide(prs, "01", "Squeeze more from the forest",
                 "Search the settings honestly - the referee stays 5-fold "
                 "cross-validation, and the test set stays sealed.")

# 4. Parameters vs hyperparameters
ds.two_col_slide(
    prs, "The model learns its parameters - you set the hyperparameters before training",
    ("Parameters: the model learns them",
     ["Learned from the data during training - fit() finds them.",
      "Example: the split rules inside every tree of the forest.",
      "There are thousands; you never set one by hand."]),
    ("Hyperparameters: you choose them",
     ["Chosen by you before training starts.",
      "Examples: n_estimators (how many trees), max_depth (how deep).",
      "Tuning = try different knob positions, keep the best result.",
      "MIT's sharper cut: hyperparameters are settings of the ALGORITHM "
      "itself - knobs OUTSIDE Session 1's three slots (model + loss + "
      "optimizer); they define the game the learner plays.",
      "The same algorithm with two settings is best thought of as two "
      "different algorithms. MIT 6.390, Appendix C."]),
    kicker="Tuning - two kinds of settings",
    note="Baking analogy: the oven temperature is a hyperparameter you set; how the "
         "ingredients turn into cake is the parameters.")

# 5. Grid + random search, merged around the tuning-arc chart (plan MERGE)
ds.image_slide(
    prs, "Tuning is Session 3's tournament again - every setting is a "
         "contestant, CV is the referee",
    f"{FIGS}/tuning_arc.png",
    kicker="Tuning - grid and random search",
    bullets=[
        "Grid search = CV plus nested for-loops, no magic: 2x3x3 = 18 combos "
        "x 5 folds = 90 trainings. Best (max_depth 4, min_samples_leaf 5, "
        "100 trees): CV 0.8259, +0.0322 over the 0.7937 default.",
        "Grids explode: 5 knobs x 6 values = 7,776 combos - nearly 39,000 "
        "trainings. 15 random draws instead test up to 15 distinct values of "
        "every knob: CV 0.8274, beating the grid with fewer tries.",
        "MIT's reframe: picking a hyperparameter value IS comparing learning "
        "algorithms - same referee, same rules; the contestants are settings "
        "instead of model families.",
        "Read the winning knobs: capped depth and bigger leaves are "
        "complexity BRAKES - regularization by another name: a little bias "
        "buys a big drop in variance. Session 3's U-curve, found automatically.",
    ],
    caption="Random Forest 5-fold CV accuracy at each tuning stage. "
            "Module 2, notebook 3. MIT 6.390, Appendix C.")

# 6. Ng's error analysis: read the mistakes before turning more knobs (NEW)
ds.image_slide(
    prs, "Before turning more knobs, read the mistakes - an hour of counting "
         "beats a week of guessing",
    f"{FIGS_THEORY}/error_analysis_sheet.png",
    kicker="Tuning - error analysis",
    bullets=[
        "Pull the misclassified validation examples and look at them by hand "
        "- about 100 when you have them.",
        "Tally them into categories in a spreadsheet; add new categories as "
        "they emerge.",
        "The bottom row of percentages IS your priority list: a category "
        "worth 5% of errors caps its payoff at almost nothing; one worth 50% "
        "is where the month goes.",
        "Guard: split a large validation set into an \"Eyeball\" part you "
        "inspect and a \"Blackbox\" part you never look at - you will overfit "
        "whatever you stare at.",
        "Exercise: pull our champion's misclassified test-set passengers in "
        "the notebook and tally your own categories (family size? fare band? "
        "title?) - no fixing allowed before the counting.",
    ],
    caption="The method, on Ng's classic image-classifier example - "
            "illustration, not our data. Ng, Machine Learning Yearning "
            "chs. 14-19.")

# 7. The honest final exam + Frame C closes (bar re-shown under the chart)
ds.image_slide(
    prs, "The test set is used exactly once - tune against it and it becomes training data",
    f"{FIGS_THEORY}/cv_vs_test_bar.png",
    kicker="The honest final exam",
    bullets=[
        "Final exam: 179 passengers the model has never seen, scored one "
        "time: accuracy 0.8268, F1 0.7597.",
        "Test is close to CV (0.8274): the tuning did not fool itself.",
        "Peeking at test scores while tuning silently turns the test set into training data.",
        "Honest expectation: tuning buys a percentage point or two, not miracles.",
        "The bar closes: we FIT on 712 rows, CHOSE by CV inside them - the "
        "REPORT band opens now, exactly once, and its score (0.8268) is the "
        "only forecast of real-world performance we own.",
    ],
    caption="Tuned forest: CV score vs the single test-set look. Module 2, "
            "notebook 3. Harvard CS109A / ISLR ch. 5 (model selection vs "
            "model assessment).")

# 8. MIT's retrain-on-all-data pipeline (NEW)
ds.image_slide(
    prs, "You don't ship the CV copy: after choosing, retrain the winning "
         "recipe on all the data",
    f"{FIGS_THEORY}/retrain_pipeline.png",
    kicker="The last step before shipping",
    bullets=[
        "MIT's five steps: compare variants by CV, pick the winner, RETRAIN "
        "the winning recipe on all available data, ship, and be judged on "
        "data nobody at the table has seen.",
        "Step 3 surprises beginners: the fold models were scaffolding; more "
        "data = a better final model, so refit everything.",
        "sklearn already does the training-set version: refit=True retrains "
        "the best settings on all 712 rows automatically - that is the model "
        "we saved.",
        "Once the single test look is spent, the recipe - not the score - is "
        "what you trust: retraining it on all 891 rows before shipping is "
        "legitimate; re-scoring it is not, because no honest data remains.",
        "Two contamination arrows are forever forbidden: test data into "
        "training, and validation scores reused as the final report card.",
    ],
    caption="MIT's end-to-end evaluation pipeline - the two red arrows are "
            "never allowed. MIT 6.390, Appendix C.")

# 9. Section divider: production
ds.section_slide(prs, "02", "Ship it",
                 "A model in a notebook helps nobody - ship it where anyone "
                 "can ask about a passenger like Owen or Florence.")

# 10. What production means - the loop diagram
ds.image_slide(
    prs, "Production is a loop, not a finish line - the model runs where "
         "real users can reach it",
    f"{FIGS_STORY}/s4_production_loop.png",
    kicker="From notebook to production",
    bullets=[
        "Save the WHOLE pipeline: titanic_model_v1.joblib, 2,599 KB - raw "
        "data in, prediction out; re-coding the prep by hand invites "
        "silently wrong predictions",
        "Load in a fresh process: no retraining, no notebook required",
        "Version the file name (v1) so you always know what is running - "
        "and can roll back",
        "Ng's rollout ladder for steps 4-5: SHADOW mode (model runs beside "
        "the human, predictions logged but unused), then CANARY (~5% of real "
        "traffic, watch, ramp), then full swap with a rollback path - "
        "deployment is a dial, not a switch",
        "Monitoring closes the loop: drift detected means retrain and reship",
    ],
    caption="The production loop: save, load, wrap, host, monitor, retrain. "
            "Module 2, notebook 3. Ng, Machine Learning Engineering for "
            "Production C1.")

# 11. Two passengers - the moment it becomes real
ds.image_slide(
    prs, "The loaded pipeline answers a real question - a survival gap of 100% vs 3%",
    f"{FIGS}/two_passengers.png",
    kicker="Sanity check on two passengers",
    bullets=[
        "Loaded from disk, no retraining: raw passenger details in, verdict out.",
        "An invented first-class woman, 30 (Fare 80, Cherbourg) - a passenger "
        "like Florence: survives, 100%.",
        "An invented third-class man, 28 (Fare 7.9, Southampton) - a passenger "
        "like Owen: does not survive, 3%.",
        "A probability is confidence, not a guarantee about one individual.",
        "The gap reflects the Titanic's reality: class and sex decided lifeboat seats.",
    ],
    caption="Two invented sanity-check passengers, scored by the saved "
            "pipeline. Module 2, notebook 3.")

# 12. Gradio + Hugging Face Spaces
ds.two_col_slide(
    prs, "A few lines of Gradio turn the pipeline into a web app anyone can use",
    ("Gradio: a function becomes a web form",
     ["Write a normal Python function: inputs in, answer out.",
      "Gradio renders it as sliders, dropdowns and a text box.",
      "Title is derived by rule (boys under 13 Master, men Mr).",
      "On Colab, launch() gives a shareable temporary link."]),
    ("Hugging Face Spaces: free hosting",
     ["Four files: app.py, the model, requirements.txt, README.md.",
      "Pin the scikit-learn version in requirements.txt.",
      "Upload the files in the browser - that is the whole deploy.",
      "No tokens or keys in code, ever - uploads stay in the browser.",
      "Result: a public URL anyone can open."]),
    kicker="From model to app")

# 13. After launch: reality checks (Netflix Prize folded in)
ds.bullets_slide(
    prs, "After launch, reality bites: models go stale and accuracy alone does not ship",
    [
        ("The $1M Netflix Prize ensemble (about 800 models, a 10% RMSE gain) "
         "was never put into production.",
         ["Engineering cost outweighed the accuracy gain, and the business had "
          "moved to streaming. Netflix tech blog, 2012."]),
        ("Two kinds of drift: DATA drift - the inputs change (different "
         "passengers start arriving); CONCEPT drift - the hidden rule f "
         "itself moves.",
         ["Same inputs, new outcomes: Session 1's Y = f(X) + ε, with f no "
          "longer standing still. Monitor inputs and outputs separately."]),
        ("Google Flu Trends overestimated flu by more than 50% in 2011-13.",
         ["It once claimed about 11% of the US had flu; CDC data said about 6%.",
          "It was essentially never retrained after 2009. Lazer et al., Science, 2014."]),
        "Version every saved model (model_v1, model_2026_09_06) so you can always roll back.",
        "Retrain on a schedule and re-score; monitoring is step 5 of the loop, not an afterthought.",
    ],
    kicker="After launch")

# 14. The whole module on one slide (MERGE: echo strip above the stage table)
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

# 15. Close
ds.close_slide(
    prs, "Session 4 - what to remember",
    [
        "Hyperparameters are the algorithm's own settings - tune them with CV, "
        "never the test set; every setting is a different algorithm.",
        "Diagnose the failing bar before turning any knob; when knobs stall, "
        "count your errors before guessing.",
        "Random search reached CV 0.8274; the one test look said 0.8268 - the tuning did not fool itself.",
        "Ship the recipe retrained on everything, then the loop: save, load, wrap, "
        "host, monitor - and watch for data drift AND concept drift.",
        "The manifest's last word: Survived = 0 for Owen, 1 for Florence, 1 for Frankie. The odds were never abstract.",
        "Practice now: notebook 03-model-optimization-and-deployment in Colab.",
    ])

out_path = "../m2-session-4-optimization-and-deployment.pptx"
ds.save_deck(prs, out_path, "M2 Session 4 - Optimization and Deployment")
print(f"Slides: {len(prs.slides)}")
print(f"Saved: {os.path.abspath(out_path)}")
