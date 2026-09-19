"""Build M2 Session 3 deck: Choosing a Model You Can Trust.

Two-part architecture: Part 1 is a theory chapter (ERM, generalization and
Ng's two-number diagnostic, bias-variance, structural vs estimation error,
regularization, flexibility vs interpretability, cross-validation); Part 2
is the Titanic tournament practice.

All practice numbers come from the executed notebook M2/02-model-selection.ipynb.
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


def notes(slide, text):
    """Speaker notes: a plain-text talk track for the teacher."""
    slide.notes_slide.notes_text_frame.text = text


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
ax.text(2.5, -0.42, "Each block = one fold, ~142 of the 712 rows. Light = used for training, blue = that round's judge.",
        ha="center", fontsize=10.5, color=pal["gray"])
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
ax.set_xlabel("Mean 5-fold CV accuracy (error bars = +/- 1 std, "
              "the wobble across folds)")
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
ax.text(0.55, -0.22, "Dark cells = correct calls, light cells = mistakes.\n"
        "99 + 11 + 18 + 51 = 179: every test passenger lands in exactly one cell.",
        ha="center", va="top", fontsize=10, color=pal["gray"])
ax.set_xlim(-1.1, 2.2)
ax.set_ylim(-0.75, 2.45)
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
ax.set_xlabel("max_depth: how many yes/no questions deep the tree may grow (deeper = more flexible)")
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
axes[0].text(0.03, -1.15, "dashed = the hidden true pattern f", fontsize=9.5, color=pal["navy"])
fig.text(0.5, -0.03, "How to read it: each faint blue line = ONE model, refit on ONE fresh sample of 20 points.",
         ha="center", fontsize=10.5, color=pal["gray"])
fig.text(0.46, -0.15, "ERROR = BIAS² + VARIANCE", ha="right", fontsize=12.5,
         fontweight="bold", color=pal["ink"])
fig.text(0.46, -0.15, " + NOISE (ε - the floor nobody can shrink)", ha="left",
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

# ---------------------------------------------------------------- chart 9
# ERM: candidate rules ranked by average training loss (synthetic values).
fig, ax = plt.subplots(figsize=(7.4, 3.9))
rows = [
    ("Rule A - fixed answer", "rigid", "0.38"),
    ("Rule B - one threshold", "modest", "0.27"),
    ("Rule C - weighted scorecard", "medium", "0.19"),
    ("Rule D - unlimited flowchart", "maximal", "0.00"),
]
headers = ["Candidate rule", "Flexibility", "Train loss"]
col_x = [0.03, 0.56, 0.79]
ax.add_patch(plt.Rectangle((0, 4), 1.0, 0.75, facecolor=pal["navy"]))
for x, htxt in zip(col_x, headers):
    ax.text(x, 4.37, htxt, fontsize=11, color="white", fontweight="bold", va="center")
for i, (name, flex, loss) in enumerate(rows):
    y = 3 - i
    is_d = (i == 3)
    face = pal["panel"] if i % 2 else "white"
    ax.add_patch(plt.Rectangle((0, y), 1.0, 0.94, facecolor=face,
                               edgecolor=pal["blue"] if is_d else "none",
                               lw=2 if is_d else 0))
    ink = pal["navy"] if is_d else pal["ink"]
    ax.text(col_x[0], y + 0.47, name, fontsize=11, color=ink, va="center",
            fontweight="bold" if is_d else "normal")
    ax.text(col_x[1], y + 0.47, flex, fontsize=11, color=pal["gray"], va="center")
    ax.text(col_x[2], y + 0.47, loss, fontsize=12, color=ink, va="center",
            fontweight="bold" if is_d else "normal")
ax.annotate("the search keeps the\nlowest number - Rule D wins",
            xy=(0.965, 0.47), xytext=(1.04, 1.6), fontsize=10.5,
            color=pal["blue"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=pal["blue"], lw=1.2))
ax.text(1.04, 0.75, "suspiciously perfect:\nit memorized the rows.\nAbout the future it\nsays nothing.",
        fontsize=10.5, color=pal["gray"], va="top")
ax.text(0.0, -0.42, "Train loss = share of wrong answers on the past rows (0.00 = none).",
        fontsize=10, color=pal["gray"], va="top")
ax.set_xlim(0, 1.45)
ax.set_ylim(-0.75, 4.95)
ax.axis("off")
ax.set_title("Four candidates, one prize: the lowest average loss on the past")
fig.savefig(f"{FIGS_T}/erm_rules.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 10
# Ng's two-number diagnostic as a 2x2 grid (labeled teaching examples).
fig, ax = plt.subplots(figsize=(7.2, 4.3))
cells = [
    (0, 1, "Healthy", "both numbers small:\nlearned, and it generalizes", pal["panel"], None),
    (1, 1, "High variance", "memorized the quirks:\nsimplify or regularize",
     pal["sky"], "Ng's example:\ntrain 1% / validation 11%"),
    (0, 0, "High bias", "too simple even for the past:\nricher model or features",
     pal["sky"], "Ng's example:\ntrain 15% / validation 16%"),
    (1, 0, "Both", "the worst corner:\nfix bias first, then variance", pal["panel"], None),
]
for cx, cy, head, body, face, card in cells:
    ax.add_patch(plt.Rectangle((cx, cy), 0.97, 0.97, facecolor=face,
                               edgecolor="white", lw=3))
    ax.text(cx + 0.06, cy + 0.82, head, fontsize=14, fontweight="bold", color=pal["navy"])
    ax.text(cx + 0.06, cy + 0.60, body, fontsize=10.5, color=pal["ink"], va="top")
    if card:
        ax.add_patch(plt.Rectangle((cx + 0.06, cy + 0.05), 0.80, 0.26,
                                   facecolor="white", edgecolor=pal["blue"], lw=1.2))
        ax.text(cx + 0.46, cy + 0.18, card, fontsize=9, color=pal["blue"],
                ha="center", va="center", fontweight="bold")
ax.text(0.485, 2.10, "gap to validation SMALL", ha="center", fontsize=11.5,
        color=pal["gray"], fontweight="bold")
ax.text(1.455, 2.10, "gap to validation BIG", ha="center", fontsize=11.5,
        color=pal["gray"], fontweight="bold")
ax.text(1.455, 2.02, "(variance out of control)", ha="center", fontsize=9.5, color=pal["gray"])
ax.text(-0.05, 1.485, "training\nerror LOW", ha="right", va="center", fontsize=10.5,
        color=pal["gray"], fontweight="bold")
ax.text(-0.05, 0.485, "training\nerror HIGH", ha="right", va="center", fontsize=10.5,
        color=pal["gray"], fontweight="bold")
ax.text(-0.05, 0.28, "(bias)", ha="right", va="center", fontsize=9.5, color=pal["gray"])
ax.set_xlim(-0.55, 2.0)
ax.set_ylim(-0.12, 2.3)
ax.axis("off")
ax.set_title("Read two numbers, land in one of four boxes")
fig.savefig(f"{FIGS_T}/ng_grid.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 11
# Structural vs estimation error: schematic two-zone diagram (no real data).
fig, ax = plt.subplots(figsize=(7.4, 4.0))
xf = np.linspace(0, 1, 300)
structural = 0.52 * np.exp(-3.4 * xf) + 0.05
estimation = 0.03 + 0.46 * xf ** 2.1
total = structural + estimation
ax.plot(xf, structural, color=pal["gray"], lw=2.2, label="structural error")
ax.plot(xf, estimation, color=pal["blue"], lw=2.2, label="estimation error")
ax.plot(xf, total, color=pal["navy"], lw=1.8, ls="--", label="sum of the two")
best = xf[np.argmin(total)]
ax.axvspan(0, 0.22, color=pal["panel"])
ax.axvspan(0.62, 1.0, color=pal["panel"])
ax.text(0.11, 0.60, "TOO RIGID\nstructural error\ndominates", ha="center",
        fontsize=10, color=pal["gray"], fontweight="bold")
ax.text(0.81, 0.60, "TOO FLEXIBLE\nestimation error\ndominates", ha="center",
        fontsize=10, color=pal["gray"], fontweight="bold")
ax.annotate("sweet spot", xy=(best, total.min()), xytext=(best - 0.02, 0.33),
            fontsize=10.5, color=pal["navy"], fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=pal["navy"], lw=1))
ax.set_xlabel("model flexibility: rigid (left) to very flexible (right) - schematic")
ax.set_ylabel("error - LOWER is better (schematic)")
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(0, 0.72)
ax.legend(loc="upper center", fontsize=9.5, frameon=False)
ax.set_title("One error falls as flexibility grows, the other rises - the sum is a U")
fig.savefig(f"{FIGS_T}/structural_estimation.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 12
# Regularization: same synthetic points, penalty too low / right / too high.
rng2 = np.random.default_rng(7)
xr = np.sort(rng2.uniform(0, 1, 22))
yr = 0.5 * np.sin(2.1 * np.pi * xr) + 0.55 * xr + rng2.normal(0, 0.12, 22)
xg2 = np.linspace(0, 1, 250)


def ridge_fit(lam, deg=9):
    Xd = np.vander(xr, deg + 1, increasing=True)
    Xg = np.vander(xg2, deg + 1, increasing=True)
    w = np.linalg.solve(Xd.T @ Xd + lam * np.eye(deg + 1), Xd.T @ yr)
    return Xg @ w


fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.1), sharey=True)
panels = [(1e-9, "Penalty ~ zero:\nchases every point"),
          (2e-3, "Penalty right:\nfollows the trend"),
          (30.0, "Penalty huge:\nignores the data")]
for axp, (lam, label) in zip(axes, panels):
    axp.scatter(xr, yr, s=22, color=pal["navy"], zorder=3)
    axp.plot(xg2, ridge_fit(lam), color=pal["blue"], lw=2.2)
    axp.set_title(label, fontsize=11)
    axp.set_xticks([])
    axp.set_yticks([])
    axp.set_ylim(-0.9, 1.5)
fig.text(0.5, -0.02, "Same 22 dots in every panel - only λ, the price per wiggle, changes.",
         ha="center", fontsize=11.5, color=pal["ink"], fontweight="bold")
fig.text(0.5, -0.14, "The dial: minimize (average loss + λ × complexity)",
         ha="center", fontsize=10.5, color=pal["gray"])
fig.savefig(f"{FIGS_T}/regularization_leash.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 13
# Flexibility vs interpretability, ISLR fig 2.7 device, our six contestants.
fig, ax = plt.subplots(figsize=(7.3, 4.4))
contestants = [
    ("Dummy", 0.04, 0.96, "nothing to fit, nothing to read into"),
    ("Logistic Regression", 0.22, 0.82, "read the weights"),
    ("Decision Tree", 0.40, 0.68, "read the flowchart"),
    ("K-Nearest Neighbors", 0.58, 0.42, "show the neighbors"),
    ("Random Forest", 0.78, 0.22, "hundreds of trees"),
    ("Gradient Boosting", 0.90, 0.12, "hundreds of sequential trees"),
]
ax.add_patch(plt.Polygon([(0.14, 0.91), (0.52, 0.91), (0.52, 0.56), (0.14, 0.56)],
                         closed=True, facecolor=pal["panel"], edgecolor=pal["blue"],
                         lw=1.2, ls="--", zorder=1))
ax.text(0.33, 0.535, "the explainable point:\nwhat a bank or hospital may choose",
        fontsize=9.5, color=pal["blue"], ha="center", va="top", fontweight="bold")
for name, fx, iy, note in contestants:
    ax.scatter(fx, iy, s=110, color=pal["blue"], zorder=3,
               edgecolor="white", lw=1.5)
    ax.annotate(name, (fx, iy), xytext=(fx + 0.025, iy + 0.045),
                fontsize=11.5, fontweight="bold", color=pal["navy"])
    ax.annotate(note, (fx, iy), xytext=(fx + 0.025, iy - 0.01),
                fontsize=9, color=pal["gray"])
ax.annotate("", xy=(0.98, 0.04), xytext=(0.04, 0.98),
            arrowprops=dict(arrowstyle="-", color=pal["sky"], lw=14, alpha=0.5))
ax.text(1.26, 1.06, "everyone wants the top-right corner\n(flexible AND readable) - no model lives there",
        fontsize=9, color=pal["gray"], ha="right", va="top")
ax.set_xlabel("Flexibility: range of shapes the family can fit (low to high)")
ax.set_ylabel("Interpretability: how readable the fitted rule is")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-0.02, 1.28)
ax.set_ylim(-0.02, 1.12)
ax.set_title("The trade-off: flexibility up, readability down (schematic)")
fig.savefig(f"{FIGS_T}/flex_interp.png")
plt.close(fig)

print("Charts written to", FIGS, "and", FIGS_T)

# ================================================================ slides
prs = ds.new_deck()

# 1 - title
ds.title_slide(
    prs, "Session 3 of 4", "Choosing a Model You Can Trust",
    "Six models compete to predict who survived the Titanic, one fair referee "
    "keeps them honest - and there is a floor every contestant must beat.")

# 2 - Frame C callback: today we live in the CHOOSE band
s = ds.image_slide(
    prs, "Today we live in the CHOOSE band: selection and assessment are different jobs",
    f"{FIGS_T}/frame_c_bar.png",
    kicker="The map for today",
    bullets=[
        "Where we stand: a model-ready Titanic table - 712 training rows, "
        "179 test rows, 18 features.",
        "Six algorithms could predict who survived. Picking by name or "
        "fashion is guessing.",
        "How to read the bar: all 891 passengers, split once. Light left = "
        "712 training rows; dark right = 179 test rows, sealed.",
        "Model SELECTION = practice tests: try many models, always inside "
        "the training rows (the CHOOSE band).",
        "Model ASSESSMENT = the final exam: grade the winner once, on the "
        "sealed test rows (REPORT).",
        "Today: Part 1, the theory of judging honestly; Part 2, the "
        "tournament that puts it to work.",
    ],
    caption="The one bar that rules Module 2 (it returns in every session). "
            "Harvard CS109A / ISLR ch. 5 (model selection vs model assessment).")
notes(s, "Re-say the exam metaphor before the theory: practice tests choose "
         "your strategy - you can take as many as you like; the final exam "
         "grades it - and you sit it once. Selection is the practice-test "
         "job, assessment is the final. Ask the class: what would go wrong "
         "if you picked your strategy based on the final's questions?")

# 3 - PART 1 divider
ds.section_slide(
    prs, "01", "Part 1 - The theory: how to judge a learner honestly",
    "Seven ideas that explain every chart you will see in Part 2: what "
    "training really does, why it flatters itself, and how to measure "
    "without fooling yourself.")

# 4 - NEW theory: empirical risk minimization in plain words
s = ds.image_slide(
    prs, "Training just minimizes average loss on the past - and a flexible "
         "rule can always ace the past",
    f"{FIGS_T}/erm_rules.png",
    kicker="Empirical risk minimization",
    bullets=[
        "Empirical risk minimization - in plain words: pick the rule with "
        "the fewest mistakes on past exams.",
        "Loss = how wrong a prediction is. 'Empirical' = measured on the "
        "rows we already HAVE.",
        "That is all training does: scan a family of candidate rules, "
        "keep the lowest average loss.",
        "How to read the table: four candidate rules, ranked by train "
        "loss - their mistakes on the past.",
        "The catch: a flexible enough rule memorizes the answer sheet. "
        "Rule D scores a perfect 0.00.",
        "A perfect training score proves nothing about tomorrow. The rest "
        "of Part 1 closes that gap.",
    ],
    caption="Illustration - invented rules and loss values. "
            "MIT 6.390, ch. 1 and appendix C (empirical risk minimization).")
notes(s, "Strip the Latin off the term: empirical risk minimization is "
         "studying from past exams - pick whatever strategy made the fewest "
         "mistakes on the old exams. The catch sits in the table: Rule D "
         "scores a perfect zero because it memorized the answer sheet, and "
         "it says nothing about tomorrow. Ask the class: which classmate is "
         "Rule D - and how do they do on a fresh exam?")

# 5 - theory (promoted): generalization + Ng's two-number diagnostic
s = ds.image_slide(
    prs, "Two numbers diagnose any model: the training error and the gap "
         "to the honest score",
    f"{FIGS_T}/ng_grid.png",
    kicker="Generalization",
    bullets=[
        "Generalization - in plain words: doing well on rows the model "
        "has never seen. Only that counts.",
        "Validation score = a practice test: held-back rows the model "
        "did not train on.",
        "Ng's diagnostic: read TWO numbers - the training error, and the "
        "gap to the validation error.",
        "Training error too high = bias: too simple even for the past it "
        "studied.",
        "Gap too wide = variance: it memorized quirks that do not travel "
        "to new rows.",
        "How to read the grid: your two numbers pick one box - each box "
        "has a different cure. Bottom-right is worst.",
        "Ng's examples: train 1% / validation 11% = variance; train 15% "
        "/ validation 16% = bias.",
    ],
    caption="Grid cards: Ng's teaching examples, not course numbers. "
            "Ng, Machine Learning Yearning chs. 20-21.")
notes(s, "Two numbers, one diagnosis: how wrong on the material it studied "
         "(bias), and how much worse on the practice test (variance). If "
         "students read Ng later: he calls the validation set the 'dev "
         "set' - same thing. Walk the two example cards into their grid "
         "cells before naming the cures. Ask the class: train 2%, "
         "validation 3% - which box, and do you change anything?")

# 6 - theory (promoted): bias-variance via Harvard's 2000-models device
s = ds.image_slide(
    prs, "Refit on fresh samples and watch: simple models miss together, "
         "flexible models scatter",
    f"{FIGS_T}/models_2000.png",
    kicker="Bias-variance decomposition",
    bullets=[
        "Bias - in plain words: the same miss every time. Variance: a "
        "different answer every time.",
        "Thought experiment: refit the same model on many fresh samples - "
        "parallel universes - and watch.",
        "How to read it: one faint line = one model, fit on one fresh "
        "sample. Dashed = the true pattern.",
        "Left, the simple family: a tight bundle - all miss the curve "
        "the same way. That shared miss is bias.",
        "Right, the flexible family: spaghetti - a new fit for every "
        "sample. That instability is variance.",
        "Total error = bias² + variance + noise. The noise floor ε never "
        "shrinks - your work lives in the other two.",
        "Hold this image: later, five cross-validation folds act as five "
        "mini-universes.",
    ],
    caption="Illustration - synthetic data, 60 refits per panel. "
            "Device from Harvard CS109A L6 / ISLR §2.2.")
notes(s, "Give them the two plain words before the picture: bias means it "
         "misses the same way every time; variance means it gives a "
         "different answer every time you retrain it. Then let the panels "
         "do the work - tight bundle versus spaghetti, same hidden dashed "
         "rule behind both. Ask the class: which panel would you rather "
         "ship, and what would change your mind?")

# 7 - NEW theory: structural vs estimation error (MIT vocabulary)
s = ds.image_slide(
    prs, "Name the two errors: a family too rigid, or too little data to "
         "find its best member",
    f"{FIGS_T}/structural_estimation.png",
    kicker="Structural vs estimation error",
    bullets=[
        "Structural error - in plain words: shopping in the wrong shoe "
        "aisle. No shoe there fits, however long you look.",
        "That is underfitting: the true rule is not in your family of "
        "shapes, so even its best member misses.",
        "Estimation error: the right aisle, too little time to try pairs "
        "- you walk out with the wrong shoe.",
        "That is overfitting: the family is so flexible that 712 rows "
        "cannot point at its best member.",
        "How to read it: x = flexibility; y = ERROR, so LOWER is better. "
        "Grey falls, blue rises.",
        "Their dashed sum is a U - the sweet spot is its LOW point. In "
        "Part 2 a real tree draws this U.",
    ],
    caption="Illustration - schematic curves, no real data. "
            "MIT 6.390, ch. 2 and ch. 5 (structural vs estimation error).")
notes(s, "Use the shoe shop: structural error is shopping in the wrong "
         "aisle - no shoe there fits, no matter how carefully you search. "
         "Estimation error is the right aisle but too little time to try "
         "pairs - you grab the wrong shoe. Bigger aisle helps the first "
         "problem and worsens the second; that tension is the U. Ask the "
         "class: which error does more data fix - the aisle or the time?")

# 8 - NEW theory: regularization - complexity on a leash
s = ds.image_slide(
    prs, "Don't choose simple-or-flexible once: take the flexible family "
         "and make complexity pay",
    f"{FIGS_T}/regularization_leash.png",
    kicker="Complexity on a leash",
    bullets=[
        "Regularization - in plain words: every extra wiggle costs "
        "points. Flexibility on a budget.",
        "How to read it: same 22 dots in all three panels. Only the "
        "penalty λ changes; the blue curve is the fit.",
        "The new training goal: minimize (average loss + λ × complexity). "
        "λ (lambda) = the price per wiggle.",
        "λ near zero: wiggles free - memorizing allowed. λ huge: wiggles "
        "unaffordable - the data is ignored.",
        "The right λ (middle panel) follows the trend and lets the noise "
        "go.",
        "Ridge regression is this idea for scorecard models: a penalty "
        "on big weights.",
        "Foreshadow: Session 4's tree knobs (like max_depth) ARE this "
        "dial. A shallow tree is a short leash.",
    ],
    caption="Illustration - synthetic points, same data in all panels. "
            "MIT 6.390 ch. 2 (penalty framing); Harvard CS109A / ISLR ch. 6 (ridge).")
notes(s, "One sentence carries the slide: every extra wiggle costs points. "
         "The model may still buy wiggles - it just has to be worth the "
         "price, and lambda sets the price. Walk the three panels: free "
         "wiggles chase every point, fair-priced wiggles follow the trend, "
         "unaffordable wiggles ignore the data. Ask the class: which panel "
         "is the memorizing classmate from the ERM slide?")

# 9 - NEW theory: flexibility vs interpretability (ISLR fig 2.7 device)
s = ds.image_slide(
    prs, "Flexibility costs readability - and a bank or hospital may "
         "choose the explainable point",
    f"{FIGS_T}/flex_interp.png",
    kicker="The trade-off chart",
    bullets=[
        "The trade-off - in plain words: the more shapes a family can "
        "fit, the harder its fitted rule is to read.",
        "How to read it: each dot = one of today's six contestants. "
        "Right = more flexible; up = easier to read.",
        "Everyone wants the top-right corner - flexible AND readable. No "
        "model lives there; you must trade.",
        "Flexible is not automatically better: it needs more data and "
        "overfits more easily - the wrong-shoe trap.",
        "A bank explaining a rejected loan, a hospital a triage call: "
        "the explainable point wins, even at some accuracy cost.",
    ],
    caption="Schematic - our six models placed as a teaching judgment. "
            "Device from ISLR fig. 2.7.")
notes(s, "The axis trade: moving right buys shapes, moving down loses "
         "readability - you cannot have both ends. Make it concrete: a "
         "scorecard's weights can be read to a customer; a committee of "
         "500 trees cannot. Ask the class: your loan is rejected by a model "
         "nobody can read - what do you want the bank to be able to tell "
         "you?")

# 10 - theory (promoted): cross-validation
s = ds.image_slide(
    prs, "Cross-validation gives every row a turn at being the test",
    f"{FIGS}/cv_folds.png",
    kicker="Estimating the true error",
    bullets=[
        "Cross-validation - in plain words: five practice tests instead "
        "of one.",
        "How to read it: each row is one round; each block one fold, "
        "~142 of the 712 training rows.",
        "The blue block sits out as that round's judge - every row "
        "judges exactly once.",
        "Five honest scores give a mean AND a wobble (the spread) - "
        "stability becomes visible.",
        "Subtle but crucial: CV scores the recipe (algorithm + "
        "settings), not one baked cake.",
        "Why 5 folds? Fewer starve training; leave-one-out is noisy - 5 "
        "or 10 is the sweet spot.",
        "The real 179 test rows are never touched - CV lives entirely "
        "inside the CHOOSE band.",
    ],
    caption="5-fold cross-validation, the referee used throughout Part 2. "
            "MIT 6.390, Appendix C; Harvard CS109A / ISLR §5.1.")
notes(s, "Exam family again: cross-validation is five practice tests "
         "instead of one, and every row gets one turn as the judge. Then "
         "MIT's line students always miss: CV scores the recipe, not the "
         "cake - it rates the way of building models, not one finished "
         "model. Ask the class: why is an average of five practice tests "
         "harder to fool than the best of five?")

# 11 - PART 2 divider
ds.section_slide(
    prs, "02", "Part 2 - Practice: the tournament",
    "Six contestants, one fair referee, one locked test set - Part 1's "
    "theory, run for real on the Titanic manifest.")

# 12 - the model zoo (+ Frame A callback footer)
zoo = ds.table_slide(
    prs, "Six contestants enter, from a deliberately dumb floor to committees of trees",
    ["Model", "Intuition in one line", "Needs scaling", "Interpretability"],
    [
        ["Dummy", 'Ignores the data, always predicts "died" - our floor', "No", "Nothing to read"],
        ["Logistic Regression", "A weighted scorecard summed into a probability", "Yes", "High - read the weights"],
        ["K-Nearest Neighbors", "Ask the 5 most similar passengers, then let them vote", "Yes", "Medium - show the neighbors"],
        ["Decision Tree", "A flowchart of yes/no questions", "No", "High - read the flowchart"],
        ["Random Forest", "Hundreds of trees voting; the vote averages mistakes away", "No", "Low - too many trees to read"],
        ["Gradient Boosting", "Trees that learn from the previous tree's mistakes", "No", "Low"],
    ],
    kicker="The model zoo",
    note='A model = a function with adjustable knobs ("parameters"). "Needs scaling" = wants '
         "all features on one ruler first: Fare runs 0-512, Age 0-80 - Fare would shout, Age whisper.",
    col_widths=[2.2, 5.2, 1.4, 3.2])
footnote(zoo, "One recipe, six fillings: only the MODEL changes - the data prep, the "
              "loss (mistake counter) and the referee stay identical for all six. "
              "(MIT 6.390, Appendix C)", 5.95)

# 13 - training accuracy lies (ERM's warning, live)
s = ds.image_slide(
    prs, "Training accuracy lies: an unlimited tree scores 0.985 on itself, 0.751 honestly",
    f"{FIGS}/train_vs_cv.png",
    kicker="Part 1's warning, live",
    bullets=[
        "Let a decision tree grow with no depth limit, then grade it "
        "twice.",
        "Graded on its own training data: 0.985. Graded on five honest "
        "practice tests: 0.751.",
        "It memorized the answers - like grading students on the exact "
        "questions they practiced at home.",
        "This is Rule D from the ERM table, caught in the act.",
        "On Ng's grid: training error low, gap 0.233 wide - the "
        "high-variance box.",
        "That is why the tournament's referee is cross-validation, never "
        "training accuracy.",
    ],
    caption="Unlimited decision tree, training vs 5-fold CV accuracy. Module 2, notebook 2.")
notes(s, "Part 1's warning caught on camera: grade the tree on the exact "
         "questions it studied and it scores 0.985; grade it on practice "
         "tests it never saw and it drops to 0.751. The 0.233 gap is "
         "memorization, not learning - the student who copied the answer "
         "sheet. Ask the class: which of the two bars would you show your "
         "boss, and why is the other one a lie?")

# 14 - baseline first
ds.big_number_slide(
    prs, 'Beat the floor first: always saying "died" already scores 0.617',
    "0.617",
    'Dummy baseline, 5-fold CV accuracy (+/- 0.003). It always predicts "died" - '
    "and 61.7% of training passengers did die, so it is right 61.7% of the time.",
    foot="Any model that cannot clearly beat this floor is useless, no matter how fancy its name. "
         "Andrew Ng's doctrine: build your first system quickly, then iterate (Machine Learning Yearning, ch. 13).",
    kicker="Baseline first")

# 15 - tournament results
ds.image_slide(
    prs, "The tournament verdict: the humble scorecard wins at 0.819",
    f"{FIGS}/tournament.png",
    kicker="Six models, one referee",
    bullets=[
        "How to read it: each bar = one model's average score over the 5 "
        "practice tests.",
        "The whisker = the wobble across those 5 tests. Dashed line = "
        "the Dummy floor.",
        "Fair referee: preprocessing re-fit inside every fold, so no "
        "fold can leak.",
        "Logistic Regression leads at 0.819; Gradient Boosting is a "
        "close second at 0.816.",
        "Whiskers are Part 1's spaghetti, measured: Random Forest "
        "wobbles most (+/- 0.054), the scorecard least (+/- 0.020).",
        "All five real models clear the floor. The Dummy's F1 (a "
        'stricter metric, two slides on) is 0.000: it never says "survived".',
        "The surprise: the humble scorecard wins - fancy does not "
        "automatically mean better.",
    ],
    caption="Mean 5-fold CV accuracy, +/- 1 std; axis starts at 0.55 to magnify "
            "the differences. Module 2, notebook 2.")

# 16 - confusion matrix
s = ds.image_slide(
    prs, "Accuracy can hide the mistakes that matter - the confusion matrix shows all four",
    f"{FIGS}/confusion.png",
    kicker="The 99% trap",
    bullets=[
        'The trap: a disease hits 1 in 100. "Always say no disease" is '
        "99% accurate - and worthless.",
        "Confusion matrix = the four ways a prediction can go right or "
        "wrong, counted.",
        "How to read it: rows = what really happened, columns = the "
        "model's call. Dark cells = correct.",
        'Champion on the 179 test passengers: 99 correct "died" (TN), '
        '51 correct "survived" (TP).',
        "The mistakes: 11 false alarms (said survived, died) and 18 "
        "missed survivors (said died, survived).",
        "One number became four. Every metric you will ever meet is "
        "arithmetic on these four counts.",
    ],
    caption="Logistic Regression, confusion matrix on the test set. Module 2, notebook 2. "
            "Harvard CS109A, Classification Metrics lecture.")
notes(s, "Open with the 99% trap: a disease hits 1 in 100, so 'always say "
         "no disease' is 99% accurate and completely useless. The confusion "
         "matrix fixes that by splitting one number into four honest "
         "counts: right two ways, wrong two ways. Every metric they will "
         "ever meet is arithmetic on these four cells. Ask the class: which "
         "of our two mistake cells - 11 false alarms or 18 missed survivors "
         "- would matter more if this were a rescue list?")

# 17 - precision vs recall
s = ds.two_col_slide(
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
    note="TP, FP, FN are the confusion-matrix cells from the last slide. F1 = a "
         "strict average (harmonic mean) of precision and recall = 0.779 - high only "
         "when both are decent. Metric pairing after Ng's ML Specialization (C2, W3).")
notes(s, "Two questions, not one: precision asks 'when we raise an alarm, "
         "how often is it real?'; recall asks 'of the real cases, how many "
         "did we catch?'. The problem picks the metric - spam filters fear "
         "false alarms, cancer screening fears missed cases. F1 is a strict "
         "average of the two that a useless side drags down. Ask the class: "
         "for airport security screening, precision first or recall first?")

# 18 - the real U-curve, with MIT's zone labels
s = ds.image_slide(
    prs, "Overfitting made visible: past depth 3, the tree memorizes instead of learning",
    f"{FIGS}/depth_sweep.png",
    kicker="The U-curve, for real",
    bullets=[
        "How to read it: x = tree depth (deeper = more flexible); y = "
        "accuracy, so HIGHER is better.",
        "Grey = score on its own training data: climbs to 0.985 and "
        "never looks back.",
        "Blue = the honest CV score: peaks at 0.815 at depth 3, then "
        "slides.",
        "The gap between the curves is memorization. At depth 20 it is "
        "0.236.",
        "This is Part 1's U on real data, flipped: y is accuracy, so the "
        "sweet spot is a PEAK.",
        "Left of depth 3: too rigid (structural error). Right of it: "
        "memorizing (estimation error).",
        "Why peak at 0.815, not 1.00? The outcome is rule plus noise, "
        "Y = f(X) + ε - and nothing can predict ε.",
    ],
    caption="Decision tree, max_depth 1-20, train vs 5-fold CV accuracy. Module 2, notebook 2. "
            "MIT 6.390, ch. 2 & ch. 5; Harvard CS109A / ISLR ch. 2.")
notes(s, "This is the theory chapter drawn by real data. Depth is the "
         "aisle-size dial: too shallow and the family cannot hold a good "
         "rule (wrong aisle); past depth 3 the tree starts memorizing the "
         "712 training rows (right aisle, wrong shoe). The grey line only "
         "climbs because memorizing always improves the score on the "
         "questions you studied. Ask the class: the honest peak is 0.815, "
         "not 1.00 - which part of the outcome can no model ever predict? "
         "(The noise term ε on the slide.)")

# 19 - the learning curve, read with Ng's two regimes
s = ds.image_slide(
    prs, "The learning curve answers \"would more data help?\" - for us: no",
    f"{FIGS_T}/learning_curve.png",
    kicker="Would more data help?",
    bullets=[
        "How to read it: x = how many rows we trained on; y = accuracy; "
        "shaded band = wobble across folds.",
        "Grey = score on its own training data; blue = the honest CV "
        "score.",
        "Ours climbs from 0.758 (56 rows) to 0.819 (569 rows) - then "
        "goes flat.",
        "Insets, high bias: the two curves meet BELOW the score you need "
        "- more rows cannot fix that.",
        "Insets, high variance: a wide gap still separates the curves - "
        "more data plausibly helps.",
        "Our verdict: flat and close, so more passengers buy little. "
        "Better settings are the lever - Session 4's job.",
    ],
    caption="Main panel: real curve, Module 2, notebook 2. Insets: regime sketches - "
            "illustration. Ng, Machine Learning Yearning chs. 28-32.")
notes(s, "The learning curve answers one question managers always ask: "
         "would more data help? Read the shape - if the honest score has "
         "flattened and sits close to the training score, more rows buy "
         "almost nothing; if a wide gap remains, more data plausibly helps. "
         "Ours is flat and close, so the lever is better features or "
         "settings - which is Session 4. Ask the class: your boss offers "
         "10,000 more passenger records - based on this chart, do you take "
         "the deal?")

# 20 - checklist + verdict, with the pneumonia story as the footer
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

# 21 - close
ds.close_slide(
    prs, "The referee, the floor, and the right question",
    [
        "Part 1: training only minimizes loss on the past (ERM) - never "
        "trust the training score alone.",
        "Read two numbers: high training error = bias; a wide gap to the "
        "honest score = variance.",
        "Too rigid (structural error) vs too flexible (estimation error) "
        "trade off - that is the U-curve.",
        "Regularization is the leash, cross-validation the honest referee "
        "- and readability is a feature, not a bonus.",
        "Part 2: the floor was 0.617, the humble scorecard won at 0.819, "
        "the sealed test set was opened once: 0.838.",
        "The real U peaked at depth 3; the learning curve said more rows "
        "won't save us - better settings might (Session 4).",
        "Accuracy is one question of many. Precision, recall, and F1 ask "
        "sharper ones.",
        "Practice now: notebook 02-model-selection in Colab.",
    ])

out = "../m2-session-3-model-selection-and-evaluation.pptx"
ds.save_deck(prs, out, "M2 Session 3 - Model Selection and Evaluation")
print("Slides:", len(prs.slides._sldIdLst))
print("Saved:", os.path.abspath(out))
