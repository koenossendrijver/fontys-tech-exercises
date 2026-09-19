"""Build M2 Session 3 deck: Choosing a Model You Can Trust.

All numbers come from the executed notebook M2/02-model-selection.ipynb.
The depth-sweep arrays below were recomputed with the notebook's exact code
(same prep, seed 42) and match its printed anchors: best CV 0.815 at depth 3;
depth 20 train 0.985 vs CV 0.749, gap 0.236. The learning-curve arrays were
recomputed the same way (notebook sec 8 code, champion pipeline, seed 42) and
match its printed anchors: CV 0.758 at 56 rows -> 0.819 at 569 rows.
Theory diagrams (Frame C bar, 2000-models device, regime sketches) use clearly
synthetic data and are captioned as illustrations.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import deck_style as ds

FIGS = "/tmp/deck-workshop/figs3"
FIGS_T = "/tmp/deck-workshop/figs-theory3"
os.makedirs(FIGS, exist_ok=True)
os.makedirs(FIGS_T, exist_ok=True)
pal = ds.mpl_theme()


def footnote(slide, text, y_in, size=12, color=ds.BLUE):
    """Extra footer line on a slide (for a second attribution/callback)."""
    box = ds._box(slide, ds.MARGIN, ds.Inches(y_in), ds.CONTENT_W, ds.Inches(0.55))
    ds._para(box.text_frame, text, size, color, first=True)

# ---------------------------------------------------------------- chart 1
# Unlimited decision tree: training accuracy vs honest 5-fold CV (NB2 sec 3)
fig, ax = plt.subplots(figsize=(7.5, 4))
labels = ["Training accuracy\n(graded on its own data)", "5-fold CV accuracy\n(the honest score)"]
vals = [0.985, 0.751]
bars = ax.bar(labels, vals, width=0.5, color=[pal["sky"], pal["blue"]])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
            ha="center", fontsize=14, fontweight="bold", color=pal["navy"])
ax.annotate("", xy=(1, 0.751), xytext=(1, 0.985),
            arrowprops=dict(arrowstyle="<->", color=pal["gray"], lw=1.2))
ax.text(1.08, 0.868, "gap 0.233\nmemorized,\nnot learned", fontsize=11, color=pal["gray"], va="center")
ax.set_ylim(0, 1.09)
ax.set_ylabel("Accuracy")
ax.set_title("One unlimited decision tree, two very different scores")
fig.savefig(f"{FIGS}/train_vs_cv.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 2
# 5-fold cross-validation schematic (NB2 sec 3 diagram)
fig, ax = plt.subplots(figsize=(7.5, 4))
for r in range(5):          # round r (top to bottom)
    for f in range(5):      # fold f (left to right)
        test = (f == r)
        color = pal["blue"] if test else pal["sky"]
        ax.add_patch(plt.Rectangle((f + 0.05, 4 - r + 0.08), 0.9, 0.84,
                                   facecolor=color, edgecolor="white", lw=2))
        if test:
            ax.text(f + 0.5, 4 - r + 0.5, "test", ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
for r in range(5):
    ax.text(-0.18, 4 - r + 0.5, f"Round {r + 1}", ha="right", va="center",
            fontsize=12, color=pal["ink"])
for f in range(5):
    ax.text(f + 0.5, 5.18, f"Fold {f + 1}", ha="center", fontsize=11, color=pal["gray"])
ax.text(2.5, -0.42, "Light = training folds, blue = held-out judge. Five rounds, five honest scores.",
        ha="center", fontsize=11, color=pal["gray"])
ax.set_xlim(-1.4, 5.2)
ax.set_ylim(-0.7, 5.55)
ax.axis("off")
ax.set_title("5-fold cross-validation: 712 training rows, 5 rounds")
fig.savefig(f"{FIGS}/cv_folds.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 3
# Tournament: mean 5-fold CV accuracy +/- 1 std (NB2 sec 5 table)
models = ["Dummy (baseline)", "Decision Tree", "Random Forest",
          "K-Nearest Neighbors", "Gradient Boosting", "Logistic Regression"]
acc = [0.617, 0.751, 0.797, 0.806, 0.816, 0.819]
std = [0.003, 0.022, 0.054, 0.026, 0.030, 0.020]
colors = [pal["sky"]] * 5 + [pal["blue"]]
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.barh(models, acc, xerr=std, color=colors, height=0.62,
        error_kw=dict(ecolor=pal["gray"], capsize=3, lw=1.2))
ax.axvline(0.617, color=pal["navy"], linestyle="--", lw=1.2)
ax.set_ylim(-1.05, 5.5)
ax.text(0.617, -0.85, "Dummy floor 0.617", fontsize=10.5, color=pal["navy"],
        ha="center", fontweight="bold")
for i, (a, s) in enumerate(zip(acc, std)):
    ax.text(a + s + 0.008, i, f"{a:.3f}", va="center", fontsize=11.5,
            fontweight="bold", color=pal["navy"])
ax.set_xlim(0.55, 0.92)
ax.set_xlabel("Mean 5-fold CV accuracy (error bars = +/- 1 std)")
ax.set_title("The tournament: same referee for all six models")
fig.savefig(f"{FIGS}/tournament.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 4
# Confusion matrix of the champion on the test set (NB2 sec 6): TN 99 FP 11 FN 18 TP 51
cm = np.array([[99, 11], [18, 51]])
cell_names = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
fig, ax = plt.subplots(figsize=(6.4, 4.4))
face = [[pal["blue"], pal["sky"]], [pal["sky"], pal["blue"]]]
for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=face[i][j],
                                   edgecolor="white", lw=3))
        txt_color = "white" if face[i][j] == pal["blue"] else pal["navy"]
        ax.text(j + 0.5, 1 - i + 0.58, str(cm[i][j]), ha="center", va="center",
                fontsize=26, fontweight="bold", color=txt_color)
        ax.text(j + 0.5, 1 - i + 0.26, cell_names[i][j], ha="center", va="center",
                fontsize=11, color=txt_color)
ax.text(0.5, 2.12, "Predicted: died", ha="center", fontsize=12, color=pal["ink"])
ax.text(1.5, 2.12, "Predicted: survived", ha="center", fontsize=12, color=pal["ink"])
ax.text(-0.1, 1.5, "Actually died", ha="right", va="center", fontsize=12, color=pal["ink"])
ax.text(-0.1, 0.5, "Actually survived", ha="right", va="center", fontsize=12, color=pal["ink"])
ax.set_xlim(-1.1, 2.2)
ax.set_ylim(-0.1, 2.45)
ax.axis("off")
ax.set_title("Champion on the 179 test passengers")
fig.savefig(f"{FIGS}/confusion.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 5
# Depth sweep (NB2 sec 7). Recomputed with the notebook's exact code + seed;
# matches printed anchors: CV peak 0.815 @ depth 3; depth 20 train 0.985 / CV 0.749.
depths = list(range(1, 21))
train_acc = [0.785, 0.795, 0.840, 0.851, 0.875, 0.890, 0.906, 0.914, 0.927, 0.937,
             0.952, 0.962, 0.972, 0.978, 0.980, 0.982, 0.982, 0.985, 0.985, 0.985]
cv_acc = [0.777, 0.794, 0.815, 0.802, 0.792, 0.795, 0.784, 0.791, 0.773, 0.780,
          0.758, 0.766, 0.756, 0.754, 0.753, 0.757, 0.754, 0.754, 0.750, 0.749]
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(depths, train_acc, marker="o", ms=4, color=pal["gray"],
        label="Training accuracy (the lying score)")
ax.plot(depths, cv_acc, marker="s", ms=4, color=pal["blue"], lw=2.2,
        label="5-fold CV accuracy (the honest score)")
ax.axvline(3, color=pal["navy"], linestyle="--", lw=1, ymax=0.55)
ax.axvspan(1, 2, color=pal["panel"])
ax.axvspan(8, 20, color=pal["panel"])
ax.text(1.15, 1.035, "underfitting - structural error:\nthe family is too rigid\nto contain a good rule",
        fontsize=9.5, color=pal["gray"], va="top")
ax.text(11.0, 0.905, "overfitting - estimation error:\nso flexible you can't reliably find\nthe right member from 712 rows",
        fontsize=9, color=pal["gray"], va="top")
ax.annotate("sweet spot: depth 3, CV 0.815", xy=(3, 0.815), xytext=(4.3, 0.832),
            fontsize=10.5, color=pal["navy"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=pal["navy"], lw=1))
ax.annotate("", xy=(20, 0.749), xytext=(20, 0.985),
            arrowprops=dict(arrowstyle="<->", color=pal["gray"], lw=1))
ax.text(19.4, 0.868, "gap\n0.236", fontsize=10, color=pal["gray"], ha="right")
ax.set_xlabel("max_depth of the decision tree")
ax.set_ylabel("Accuracy")
ax.set_xticks([1, 3, 5, 10, 15, 20])
ax.set_ylim(0.63, 1.04)
ax.legend(loc="lower right", fontsize=10.5, frameon=False)
ax.set_title("Train vs CV accuracy as the tree gets deeper")
fig.savefig(f"{FIGS}/depth_sweep.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 6
# Frame C: the FIT | CHOOSE | REPORT bar (Harvard/ISLR), re-shown from Deck 2.
# One canonical drawing shared with Decks 2 and 4; deck 3 adds the TODAY tag.
from shared_figs import make_frame_c

make_frame_c(f"{FIGS_T}/frame_c_bar.png", tag="TODAY")

# ---------------------------------------------------------------- chart 7
# Harvard's 2000-models device, small version: 60 refits per panel.
# Synthetic data ONLY - captioned as an illustration on the slide.
import warnings
rng = np.random.default_rng(42)
xg = np.linspace(0.02, 0.98, 200)


def hidden_rule(x):
    return 0.6 * np.sin(2.2 * np.pi * x) + 0.8 * x


fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), sharey=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for ax, deg in zip(axes, [1, 10]):
        for _ in range(60):
            xs = rng.uniform(0, 1, 20)
            ys = hidden_rule(xs) + rng.normal(0, 0.35, 20)
            ax.plot(xg, np.polyval(np.polyfit(xs, ys, deg), xg),
                    color=pal["blue"], alpha=0.10, lw=1)
        ax.plot(xg, hidden_rule(xg), color=pal["navy"], lw=2, ls="--")
        ax.set_ylim(-1.3, 2.1)
        ax.set_xticks([])
        ax.set_yticks([])
axes[0].set_title("Simple family: a tight bundle -\nall miss the curve the same way (bias)", fontsize=11)
axes[1].set_title("Flexible family: spaghetti -\nevery sample gives a new fit (variance)", fontsize=11)
axes[0].text(0.03, -1.15, "dashed = the hidden rule f", fontsize=9.5, color=pal["navy"])
fig.text(0.46, -0.03, "ERROR = BIAS² + VARIANCE", ha="right", fontsize=12.5,
         fontweight="bold", color=pal["ink"])
fig.text(0.46, -0.03, " + NOISE (ε - the floor nobody can shrink)", ha="left",
         fontsize=12.5, fontweight="bold", color=pal["gray"])
fig.savefig(f"{FIGS_T}/models_2000.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 8
# Learning curve (NB2 sec 8). Arrays recomputed with the notebook's exact code
# (champion pipeline, train_sizes 10%..100%, cv=5, shuffle, seed 42); matches
# printed anchors: CV 0.758 with 56 rows -> 0.819 with 569 rows.
lc_sizes = [56, 184, 312, 440, 569]
lc_tr = [0.8571, 0.8413, 0.8372, 0.8341, 0.8299]
lc_tr_std = [0.0160, 0.0234, 0.0150, 0.0079, 0.0064]
lc_cv = [0.7584, 0.8189, 0.8175, 0.8175, 0.8189]
lc_cv_std = [0.0192, 0.0227, 0.0241, 0.0216, 0.0200]
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.plot(lc_sizes, lc_tr, marker="o", ms=4, color=pal["gray"], label="Training accuracy")
ax.fill_between(lc_sizes, np.array(lc_tr) - np.array(lc_tr_std),
                np.array(lc_tr) + np.array(lc_tr_std), color=pal["gray"], alpha=0.15)
ax.plot(lc_sizes, lc_cv, marker="s", ms=5, lw=2.2, color=pal["blue"],
        label="5-fold CV accuracy (the honest score)")
ax.fill_between(lc_sizes, np.array(lc_cv) - np.array(lc_cv_std),
                np.array(lc_cv) + np.array(lc_cv_std), color=pal["blue"], alpha=0.15)
ax.annotate("0.758", xy=(56, 0.7584), xytext=(72, 0.737), fontsize=10.5,
            fontweight="bold", color=pal["navy"],
            arrowprops=dict(arrowstyle="->", color=pal["navy"], lw=1))
ax.annotate("0.819 - and flat", xy=(569, 0.8189), xytext=(452, 0.786), fontsize=10.5,
            fontweight="bold", color=pal["navy"],
            arrowprops=dict(arrowstyle="->", color=pal["navy"], lw=1))
ax.set_xlabel("Training rows used")
ax.set_ylabel("Accuracy")
ax.set_xlim(20, 640)
ax.set_ylim(0.70, 0.97)
ax.legend(loc="lower right", fontsize=10, frameon=False)
ax.set_title("The champion's learning curve: flat and close = more rows won't help")
# Ng's two regime sketches, boxed insets (synthetic - captioned as illustration)
GREEN = "#2E7D32"
sk_x = np.linspace(0, 1, 60)
in1 = ax.inset_axes([0.38, 0.66, 0.28, 0.24])
in1.plot(sk_x, 0.62 + 0.13 * np.exp(-4 * sk_x), color=pal["gray"], lw=1.4)
in1.plot(sk_x, 0.58 - 0.35 * np.exp(-4 * sk_x), color=pal["blue"], lw=1.4)
in1.axhline(0.85, color=GREEN, ls="--", lw=1)
in1.text(0.5, 0.88, "score you need", fontsize=6.5, color=GREEN, ha="center")
in1.set_title("high bias: converged, too low", fontsize=8, pad=2)
in2 = ax.inset_axes([0.70, 0.66, 0.28, 0.24])
in2.plot(sk_x, 0.93 - 0.03 * sk_x, color=pal["gray"], lw=1.4)
in2.plot(sk_x, 0.70 - 0.42 * np.exp(-2.5 * sk_x), color=pal["blue"], lw=1.4)
in2.axhline(0.85, color=GREEN, ls="--", lw=1)
in2.text(0.5, 0.82, "score you need", fontsize=6.5, color=GREEN,
         ha="center", va="top")
in2.set_title("high variance: wide gap", fontsize=8, pad=2)
for ins in (in1, in2):
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_ylim(0.15, 1.02)
    ins.set_facecolor(pal["panel"])
ax.text(0.66, 0.60, "regime sketches - illustration", transform=ax.transAxes,
        fontsize=8.5, color=pal["gray"], ha="center")
fig.savefig(f"{FIGS_T}/learning_curve.png")
plt.close(fig)

print("Charts written to", FIGS, "and", FIGS_T)

# ================================================================ slides
prs = ds.new_deck()

# 1 - title
ds.title_slide(
    prs, "Session 3 of 4", "Choosing a Model You Can Trust",
    "Six ways to guess who lives on the manifest, one fair referee to keep "
    "them honest - and a floor every contestant must beat.")

# 2 - Frame C callback: today we live in the CHOOSE band
ds.image_slide(
    prs, "Today we live in the CHOOSE band: selection and assessment are different jobs",
    f"{FIGS_T}/frame_c_bar.png",
    kicker="The map for today",
    bullets=[
        "Session 2 left a model-ready table: 712 training rows, 179 test rows, 18 features.",
        "Six algorithms could predict who survived - picking by name or fashion is guessing.",
        "Model SELECTION = choosing among models and settings. Done many times, "
        "in the CHOOSE band, by cross-validation.",
        "Model ASSESSMENT = grading the final choice. Done once, in the REPORT band. "
        "Select on the test set and its score stops being a forecast.",
        "The plan: a floor, a fair referee, the right metric - then one look at the test set.",
    ],
    caption="One bar rules the module, re-shown from Session 2. "
            "Harvard CS109A / ISLR ch. 5 (model selection vs model assessment).")

# 3 - the model zoo (+ Frame A callback footer)
zoo = ds.table_slide(
    prs, "Six contestants enter, from a deliberately dumb floor to committees of trees",
    ["Model", "Intuition in one line", "Needs scaling", "Interpretability"],
    [
        ["Dummy", 'Ignores the data, always predicts "died" - our floor', "No", "Nothing to read"],
        ["Logistic Regression", "A weighted scorecard summed into a probability", "Yes", "High - read the weights"],
        ["K-Nearest Neighbors", "Ask the 5 passengers most similar to Frankie", "Yes", "Medium - show the neighbors"],
        ["Decision Tree", "A flowchart of yes/no questions", "No", "High - read the flowchart"],
        ["Random Forest", "Hundreds of trees voting; the vote averages mistakes away", "No", "Low - too many trees to read"],
        ["Gradient Boosting", "Trees that learn from the previous tree's mistakes", "No", "Low"],
    ],
    kicker="The model zoo",
    note='A model is a function with adjustable knobs ("parameters"); training turns the knobs until predictions match the examples.',
    col_widths=[2.2, 5.2, 1.4, 3.2])
footnote(zoo, "Session 1's recipe, refilled six times: each contestant is a different "
              "filling of the MODEL slot — the loss and the referee stay the same for "
              "everyone. (MIT 6.390, Appendix C)", 5.95)

# 4 - training accuracy lies
ds.image_slide(
    prs, "Training accuracy lies: an unlimited tree scores 0.985 on itself, 0.751 honestly",
    f"{FIGS}/train_vs_cv.png",
    kicker="Why we need a referee",
    bullets=[
        "Let a decision tree grow without limits, then grade it on its own training "
        "data: 0.985. Grade it honestly with cross-validation: 0.751.",
        "It memorized the answers instead of learning the subject - like grading "
        "students on the exact questions they practiced at home.",
        "Ng's diagnostic: always read TWO numbers - training score and honest score. "
        "Poor training score = bias (too simple). Big gap = variance (memorizing). "
        "Our tree: gap 0.233 = variance.",
        "Ng's drill: train error 1% / dev error 11% = variance problem; train 15% / "
        "dev 16% = bias problem. (Ng's teaching examples, not course numbers.)",
        "Diagnose first - the cures are different and often opposite.",
    ],
    caption="Unlimited decision tree, training vs 5-fold CV accuracy. Module 2, notebook 2. "
            "Diagnostic: Ng, Machine Learning Yearning chs. 20-21.")

# 5 - cross-validation
ds.image_slide(
    prs, "Cross-validation gives every row a turn at being the test",
    f"{FIGS}/cv_folds.png",
    kicker="The fair referee",
    bullets=[
        "Split the 712 training rows into 5 equal folds; run 5 rounds - each fold "
        "sits out once as the judge.",
        "Five honest scores instead of one: we see the mean AND the wobble (std). "
        "One lucky split can flatter a model; five cannot.",
        "Subtle but crucial: CV evaluates the recipe (algorithm + settings), not one "
        "fitted model - CV scores the recipe, not the cake.",
        "Why 5 folds? Fewer folds starve training; leave-one-out folds are nearly "
        "identical and their average gets noisy - 5 or 10 is the empirical sweet spot.",
        "The real test set is never touched - CV lives entirely inside the CHOOSE band.",
    ],
    caption="5-fold cross-validation. Module 2, notebook 2. "
            "MIT 6.390, Appendix C; Harvard CS109A / ISLR §5.1.")

# 6 - baseline first
ds.big_number_slide(
    prs, 'Beat the floor first: always saying "died" already scores 0.617',
    "0.617",
    'Dummy baseline, 5-fold CV accuracy (+/- 0.003). It ignores the data and always predicts "died".',
    foot="Any model that cannot clearly beat this floor is useless, no matter how fancy its name. "
         "Andrew Ng's doctrine: build your first system quickly, then iterate (Machine Learning Yearning, ch. 13).",
    kicker="Baseline first")

# 7 - tournament results
ds.image_slide(
    prs, "The tournament verdict: the humble scorecard wins at 0.819",
    f"{FIGS}/tournament.png",
    kicker="Six models, one referee",
    bullets=[
        "Same referee for all six: 5-fold CV, preprocessing re-fit inside every fold - no leaks.",
        "Logistic Regression leads at 0.819. Gradient Boosting is a close second at 0.816.",
        "Those error bars are information: Random Forest wobbles most (+/- 0.054) - "
        "the same model refit on slightly different data gives noticeably different "
        "answers. Next slide shows why.",
        'Every real model clears the floor. The Dummy\'s F1 is 0.000 - it never predicts "survived".',
        "The surprise: fancy does not automatically mean better.",
    ],
    caption="Mean 5-fold CV accuracy, +/- 1 std. Module 2, notebook 2.")

# 8 - NEW: Harvard's 2000-models device - why flexible models wobble
ds.image_slide(
    prs, "Why the forest wobbles: flexible models change with every sample they see",
    f"{FIGS_T}/models_2000.png",
    kicker="Parallel universes",
    bullets=[
        "Thought experiment: refit the same model on many different samples from "
        "the same population - parallel universes - and watch the fits scatter.",
        "Simple models (a straight line) form a tight bundle: they miss the same "
        "way every time. Biased, but stable.",
        "Very flexible models form spaghetti: unbiased on average, but each sample "
        "yields a wildly different fit. That is variance.",
        "Our tournament ran a small version of this: the 5 folds are 5 mini-universes, "
        "and the fold-to-fold std is the wobble made visible - RF +/- 0.054 vs "
        "LogReg +/- 0.020.",
    ],
    caption="Illustration - synthetic data, 60 refits per panel. "
            "Device from Harvard CS109A L6 / ISLR §2.2.")

# 9 - section divider
ds.section_slide(
    prs, "02", "Beyond accuracy",
    "When 99% right is 100% useless: the mistakes have names, so count all "
    "four kinds before you trust a score.")

# 10 - confusion matrix
ds.image_slide(
    prs, "Accuracy can hide the mistakes that matter - the confusion matrix shows all four",
    f"{FIGS}/confusion.png",
    kicker="The 99% trap",
    bullets=[
        'A disease hits 1 in 100 people. A model that always says "no disease" is 99% accurate - and worthless.',
        "Confusion matrix = a count of the four ways a prediction can go right or "
        "wrong - and every metric you will ever meet is just arithmetic on these four counts.",
        'Our champion on the 179 test passengers: 99 correct "died" (TN), 51 correct "survived" (TP).',
        "11 false alarms (predicted survived, did not) and 18 missed survivors.",
        "One number became four. Now we can ask sharper questions.",
    ],
    caption="Logistic Regression, confusion matrix on the test set. Module 2, notebook 2. "
            "Harvard CS109A, Classification Metrics lecture.")

# 11 - precision vs recall
ds.two_col_slide(
    prs, "Precision and recall answer different questions - pick the one your problem asks",
    ("Precision 0.823 - trust our alarms", [
        '"Of everyone we said survives, how many actually did?"',
        "TP / (TP + FP) = 51 / (51 + 11) = 0.823",
        "High precision = few false alarms.",
        "Spam filter: a false alarm buries real mail. Precision first.",
    ]),
    ("Recall 0.739 - catch them all", [
        '"Of everyone who actually survived, how many did we catch?"',
        "TP / (TP + FN) = 51 / (51 + 18) = 0.739",
        "High recall = few missed cases.",
        "Cancer screening: a missed case is the disaster. Recall first.",
    ]),
    kicker="Two kinds of mistakes",
    note="F1 = harmonic mean of both = 0.779, high only when both are decent. "
         "Metric pairing after Andrew Ng's Machine Learning Specialization (C2, W3).")

# 12 - overfitting made visible, now with MIT's names + Frame B callback
ds.image_slide(
    prs, "Overfitting made visible: past depth 3, the tree memorizes instead of learning",
    f"{FIGS}/depth_sweep.png",
    kicker="Bias and variance, named",
    bullets=[
        "Grow the tree one level at a time. Training accuracy climbs to 0.985 and never looks back.",
        "The honest CV score peaks at 0.815 at depth 3, then slides. At depth 20 the "
        "gap is 0.236 - memorizing, not learning.",
        "MIT's names make the tradeoff mechanical: more flexibility lowers structural "
        "error and raises estimation error - every dial in Session 4 moves this dial.",
        "The outcome is a hidden rule plus noise: Y = f(X) + ε. Learning estimates the "
        "rule; nothing can predict the noise. Error you can shrink is reducible; the "
        "noise floor is irreducible.",
        "Why does the honest curve peak at 0.815, not 1.00? Session 1's grey bar: ε. "
        "The rest is noise nobody can predict.",
    ],
    caption="Decision tree, max_depth 1-20, train vs 5-fold CV accuracy. Module 2, notebook 2. "
            "MIT 6.390, ch. 2 & ch. 5; Harvard CS109A / ISLR ch. 2.")

# 13 - NEW: the learning curve, read with Ng's two regimes
ds.image_slide(
    prs, "The learning curve answers \"would more data help?\" - for us: no",
    f"{FIGS_T}/learning_curve.png",
    kicker="Would more data help?",
    bullets=[
        "Plot the honest score against training-set size: our CV accuracy climbs "
        "from 0.758 at 56 rows to 0.819 at 569 rows - then flattens.",
        "Regime 1, high bias: train and CV converge close together but below the "
        "score you need. The curve has flattened - more rows cannot get you there; "
        "change the model or features, not the dataset.",
        "Regime 2, high variance: training score is fine but a wide gap separates "
        "it from CV - the curves haven't converged; more data plausibly helps.",
        "Read ours: flat and close - more passengers would buy little. Better "
        "features or settings are the lever. That is exactly Session 4's job.",
    ],
    caption="Main panel: real curve, Module 2, notebook 2. Insets: regime sketches - "
            "illustration. Ng, Machine Learning Yearning chs. 28-32.")

# 14 - MERGE: checklist + verdict, with the pneumonia story as the footer
ds.two_col_slide(
    prs, "The verdict: the simplest model that beats the floor on the right metric - the scorecard",
    ("The checklist", [
        "Beats the baseline clearly? The floor is 0.617 - if not, stop.",
        "Metric matches the problem? Accuracy, precision, recall, or F1 - decide "
        "before you rank.",
        "Simple first: if the scorecard is within a hair of the forest, ship the scorecard.",
        "Stable across folds? A model that wobbles (Random Forest: +/- 0.054) is "
        "harder to trust.",
    ]),
    ("Verdict & handoff", [
        "Champion by CV: Logistic Regression, 0.819 (+/- 0.020).",
        "One look at the locked test set: accuracy 0.838.",
        'Precision 0.823, recall 0.739, F1 0.779 on "survived".',
        "AUC 0.878 = chance a random survivor ranks above a random non-survivor.",
        "Random Forest finished 0.797 but has the most knobs - Session 4 tunes it, "
        "then ships it.",
    ]),
    kicker="Decision and next step",
    note="Why interpretability is on the checklist: a real pneumonia model learned "
         '"asthma lowers risk" - true in the data (asthmatics were rushed to intensive '
         "care), fatal if acted on; caught only because the model was readable. "
         "Source: Caruana et al., KDD 2015.")

# 15 - close
ds.close_slide(
    prs, "The referee, the floor, and the right question",
    [
        "Two numbers, always: the training score lies, CV is honest - and the gap is variance.",
        "Baseline first: the floor is 0.617. Beat it clearly, or go home.",
        "Underfit = structural error, overfit = estimation error - the sweet spot is "
        "where the honest curve peaks, and ε keeps it under 1.00.",
        "The learning curve says more rows won't save us - better settings might (Session 4).",
        "Accuracy is one question of many. Precision, recall, and F1 ask sharper ones.",
        "Practice now: notebook 02-model-selection in Colab.",
    ])

out = "../m2-session-3-model-selection-and-evaluation.pptx"
ds.save_deck(prs, out, "M2 Session 3 - Model Selection and Evaluation")
print("Slides:", len(prs.slides._sldIdLst))
print("Saved:", os.path.abspath(out))
