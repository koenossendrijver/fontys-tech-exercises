"""Build M2 Session 3 deck: Choosing a Model You Can Trust.

All numbers come from the executed notebook M2/02-model-selection.ipynb.
The depth-sweep arrays below were recomputed with the notebook's exact code
(same prep, seed 42) and match its printed anchors: best CV 0.815 at depth 3;
depth 20 train 0.985 vs CV 0.749, gap 0.236.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import deck_style as ds

FIGS = "/tmp/deck-workshop/figs3"
os.makedirs(FIGS, exist_ok=True)
pal = ds.mpl_theme()

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
ax.axvline(3, color=pal["navy"], linestyle="--", lw=1)
ax.axvspan(1, 2, color=pal["panel"])
ax.axvspan(8, 20, color=pal["panel"])
ax.text(1.15, 0.995, "underfitting\n(too simple)", fontsize=10, color=pal["gray"])
ax.text(13.5, 0.70, "overfitting\n(memorizing)", fontsize=10, color=pal["gray"])
ax.annotate("sweet spot: depth 3, CV 0.815", xy=(3, 0.815), xytext=(4.6, 0.855),
            fontsize=10.5, color=pal["navy"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=pal["navy"], lw=1))
ax.annotate("", xy=(20, 0.749), xytext=(20, 0.985),
            arrowprops=dict(arrowstyle="<->", color=pal["gray"], lw=1))
ax.text(19.4, 0.868, "gap\n0.236", fontsize=10, color=pal["gray"], ha="right")
ax.set_xlabel("max_depth of the decision tree")
ax.set_ylabel("Accuracy")
ax.set_xticks([1, 3, 5, 10, 15, 20])
ax.set_ylim(0.63, 1.04)
ax.legend(loc="lower left", fontsize=10.5, frameon=False)
ax.set_title("Train vs CV accuracy as the tree gets deeper")
fig.savefig(f"{FIGS}/depth_sweep.png")
plt.close(fig)

print("Charts written to", FIGS)

# ================================================================ slides
prs = ds.new_deck()

# 1 - title
ds.title_slide(
    prs, "Session 3 of 4", "Choosing a Model You Can Trust",
    "Six ways to guess who lives on the manifest, one fair referee to keep "
    "them honest - and a floor every contestant must beat.")

# 2 - today's question
ds.bullets_slide(
    prs, '"Which model is best?" really asks: how do we compare fairly?',
    [
        "Session 2 left us a model-ready table: 712 training rows, 179 test rows, 18 features.",
        ("Six algorithms could predict who survived. Picking by name or fashion is guessing.",
         ["The scorecard weighs Florence's class and sex; the forest asks "
          "hundreds of yes/no flowcharts about Owen."]),
        ("The whole answer in one line: a baseline, a fair referee, and the right metric.",
         ["Baseline = the floor any real model must clearly beat.",
          "Fair referee = 5-fold cross-validation, the honest score.",
          "Right metric = accuracy is not always the question your problem asks."]),
        "The test set stays locked away until the very end. We open it exactly once.",
    ],
    kicker="Today's question",
    note="Plan: run the tournament first, then look at what accuracy hides.")

# 3 - the model zoo
ds.table_slide(
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

# 4 - training accuracy lies
ds.image_slide(
    prs, "Training accuracy lies: an unlimited tree scores 0.985 on itself, 0.751 honestly",
    f"{FIGS}/train_vs_cv.png",
    kicker="Why we need a referee",
    bullets=[
        "Let a decision tree grow without limits, then grade it on its own training data: 0.985.",
        "Grade the same tree honestly with cross-validation: 0.751. The gap is 0.233.",
        "It memorized the answers instead of learning the subject - passengers, noise included.",
        "Generalization = performing well on data you have never seen. It is the whole "
        "point of ML: the next passenger is always unseen.",
        "Overfitting = brilliant on seen data, stumbling on unseen - like grading students "
        "on the exact questions they practiced at home.",
    ],
    caption="Unlimited decision tree, training vs 5-fold CV accuracy. Module 2, notebook 2.")

# 5 - cross-validation
ds.image_slide(
    prs, "Cross-validation gives every row a turn at being the test",
    f"{FIGS}/cv_folds.png",
    kicker="The fair referee",
    bullets=[
        "Split the 712 training rows into 5 equal folds. Run 5 rounds.",
        "Each round, one fold sits out as the judge; the model trains on the other four.",
        "Five honest scores instead of one: we see the mean AND how much the model wobbles (std).",
        "One lucky split can flatter a model. Five splits cannot.",
        "The real test set is never touched - CV lives entirely inside the training data.",
    ],
    caption="5-fold cross-validation. Module 2, notebook 2.")

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
        "Random Forest wobbles the most between folds: 0.797 with std 0.054.",
        'Every real model clears the floor. The Dummy\'s F1 is 0.000 - it never predicts "survived".',
        "The surprise: fancy does not automatically mean better.",
    ],
    caption="Mean 5-fold CV accuracy, +/- 1 std. Module 2, notebook 2.")

# 8 - section divider
ds.section_slide(
    prs, "02", "Beyond accuracy",
    "When 99% right is 100% useless: the mistakes have names, so count all "
    "four kinds before you trust a score.")

# 9 - confusion matrix
ds.image_slide(
    prs, "Accuracy can hide the mistakes that matter - the confusion matrix shows all four",
    f"{FIGS}/confusion.png",
    kicker="The 99% trap",
    bullets=[
        'A disease hits 1 in 100 people. A model that always says "no disease" is 99% accurate - and worthless.',
        "Confusion matrix = a count of the four ways a prediction can go right or wrong.",
        'Our champion on the 179 test passengers: 99 correct "died" (TN), 51 correct "survived" (TP).',
        "11 false alarms (predicted survived, did not) and 18 missed survivors.",
        "One number became four. Now we can ask sharper questions.",
    ],
    caption="Logistic Regression, confusion matrix on the test set. Module 2, notebook 2.")

# 10 - precision vs recall
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

# 11 - real-world stakes: Caruana
ds.bullets_slide(
    prs, 'A pneumonia model learned "asthma lowers risk" - reading the model caught the trap',
    [
        "Real case: a model ranked pneumonia patients by death risk to decide who could go home.",
        "It learned that asthma predicts LOWER risk - and the data really does say that.",
        "Why: asthmatic pneumonia patients were rushed to intensive care, and aggressive care saved them.",
        "Deploy that rule and asthmatics get sent home. The pattern was true; the action would be fatal.",
        "It was caught only because the model was readable. Understand the mistakes before you ship.",
    ],
    kicker="Real-world stakes",
    note="Source: Caruana et al., KDD 2015 - pneumonia risk models in healthcare.")

# 12 - overfitting made visible
ds.image_slide(
    prs, "Overfitting made visible: past depth 3, the tree memorizes instead of learning",
    f"{FIGS}/depth_sweep.png",
    kicker="Bias and variance",
    bullets=[
        "Grow the tree one level at a time. Training accuracy climbs to 0.985 and never looks back.",
        "The honest CV score peaks at 0.815 at depth 3, then slides back down.",
        "At depth 20 the gap is 0.236. That gap is memorizing - generalization is what the CV curve measures.",
        "Bias = too simple to capture the real pattern: the model misses it everywhere (left zone).",
        "Variance = so flexible it memorizes the noise of its sample: every quirk reshapes it (right zone).",
        "Every model choice negotiates between the two - the sweet spot is where the honest score peaks.",
    ],
    caption="Decision tree, max_depth 1-20, train vs 5-fold CV accuracy. Module 2, notebook 2.")

# 13 - how to choose in real life
ds.bullets_slide(
    prs, "In real life, pick the simplest model that beats the baseline on the right metric",
    [
        "Does it clearly beat the baseline? If not, stop.",
        "Does the metric match the problem? Accuracy, precision, recall, or F1 - decide before you rank.",
        "Simple model first: if the scorecard is within a hair of the forest, ship the scorecard.",
        "Check the std across folds: a model that wobbles (Random Forest: +/- 0.054) is harder to trust.",
        "Weigh speed and explainability - you will have to retrain it and defend it.",
        "AUC 0.878 = chance a random survivor is ranked above a random non-survivor (0.5 coin flip, 0.8+ solid).",
    ],
    kicker="The checklist",
    note='The notebook\'s one-liner: "the best model is the simplest one that beats the baseline '
         'on the metric that matches your problem, reliably across folds."')

# 14 - verdict and handoff
ds.two_col_slide(
    prs, "Logistic Regression is the honest champion - the forest gets a second chance",
    ("The verdict", [
        "Champion by CV: Logistic Regression, 0.819 (+/- 0.020).",
        "One look at the locked test set: accuracy 0.838.",
        'Precision 0.823, recall 0.739, F1 0.779 on "survived".',
        "Test score close to CV: the referee did not fool itself.",
        "Simple, fast, and you can read its weights.",
    ]),
    ("The handoff to Session 4", [
        "Random Forest finished mid-table: 0.797 (+/- 0.054).",
        "But it has the most knobs left: depth, trees, leaf size.",
        "LogReg won by a whisker and has almost no knobs left to turn.",
        "Session 4: tune the forest with grid and random search.",
        "Then one look at the test set - and ship it as a real app.",
    ]),
    kicker="Decision and next step")

# 15 - close
ds.close_slide(
    prs, "The referee, the floor, and the right question",
    [
        "Training accuracy lies. Cross-validation is the honest score - mean and wobble.",
        "Baseline first: the floor is 0.617. Beat it clearly, or go home.",
        "Accuracy is one question of many. Precision, recall, and F1 ask sharper ones.",
        "The simplest model that beats the baseline on the right metric wins.",
        "Practice now: notebook 02-model-selection in Colab.",
    ])

out = "../m2-session-3-model-selection-and-evaluation.pptx"
ds.save_deck(prs, out, "M2 Session 3 - Model Selection and Evaluation")
print("Slides:", len(prs.slides._sldIdLst))
print("Saved:", os.path.abspath(out))
