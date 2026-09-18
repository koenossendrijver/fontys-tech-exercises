"""Build M2 Session 4 deck: Tuning and Shipping the Model."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import deck_style as ds

FIGS = "/tmp/deck-workshop/figs4"
FIGS_STORY = "/tmp/deck-workshop/figs-story"
FIGS_CONCEPTS = "/tmp/deck-workshop/figs-concepts"
os.makedirs(FIGS, exist_ok=True)
os.makedirs(FIGS_STORY, exist_ok=True)
os.makedirs(FIGS_CONCEPTS, exist_ok=True)
pal = ds.mpl_theme()

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
fig.savefig(f"{FIGS}/tuning_arc.png")
plt.close(fig)

# ---- Chart B: CV vs one-shot test score (NB3 sec 5) ----
fig, ax = plt.subplots(figsize=(7.5, 4))
labels = ["Cross-validation\n(training data only)", "Test set\n(179 unseen passengers)"]
vals = [0.8274, 0.8268]
bars = ax.bar(labels, vals, color=[pal["sky"], pal["blue"]], width=0.4)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0012, f"{v:.4f}", ha="center",
            fontsize=12, fontweight="bold", color=pal["navy"])
ax.set_ylim(0.75, 0.842)
ax.set_ylabel("Accuracy")
ax.set_title("CV and test agree: a gap of only 0.0006")
fig.savefig(f"{FIGS}/cv_vs_test.png")
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

# ---- Chart E (concepts): echo of the Session 1 hero contrast ----
fig, ax = plt.subplots(figsize=(12.0, 4.4))
rows = [
    (2.55, pal["panel"], pal["navy"], pal["ink"],
     "M1 - descriptive analytics: dashboards that inform people",
     "The manifest, described: 38.4% of 891 passengers survived - "
     "a chart a person reads, then decides."),
    (0.35, pal["blue"], "white", "white",
     "M2 - machine learning: models that act inside products",
     "The saved pipeline, live in a public web app: a passenger like "
     "Owen - 3% survival - answered on demand."),
]
for y0, fc, hc, bc, head, body in rows:
    ax.add_patch(FancyBboxPatch((0.5, y0), 11.0, 1.5,
                                boxstyle="round,pad=0.05",
                                facecolor=fc, edgecolor=pal["blue"], lw=1.4))
    ax.text(6.0, y0 + 1.02, head, ha="center", color=hc,
            fontweight="bold", fontsize=13.5)
    ax.text(6.0, y0 + 0.44, body, ha="center", color=bc, fontsize=10.5)
ax.add_patch(FancyArrowPatch((1.7, 2.48), (1.7, 1.92), arrowstyle="-|>",
                             mutation_scale=16, color=pal["navy"], lw=1.8))
ax.text(2.05, 2.20, "same 891 passengers - the question changed, and the "
        "reader became a program", ha="left", color=pal["navy"],
        fontsize=10.5, fontweight="bold")
ax.set_xlim(0, 12.0)
ax.set_ylim(0.1, 4.3)
ax.axis("off")
fig.savefig(f"{FIGS_CONCEPTS}/echo_da_vs_ml.png")
plt.close(fig)

# === SLIDES ===
prs = ds.new_deck()

# 1. Title
ds.title_slide(prs, "Session 4 of 4", "Tuning and Shipping the Model",
               "The forest gets tuned, the test set gets its one look, and the "
               "model leaves the notebook: a public app anyone can ask about a "
               "passenger like Owen or Florence.")

# 2. Where we are
ds.bullets_slide(
    prs, "Session 3 crowned a champion - today we tune the runner-up and ship it",
    [
        "Notebook 1 built a leak-proof pipeline: 712 train / 179 test rows, 18 model-ready features.",
        "Notebook 2 ran the tournament: Logistic Regression won with CV accuracy 0.819, test 0.838.",
        "The winning scorecard has almost no knobs left to turn - it is already at its ceiling.",
        "Random Forest (CV 0.797) has the most knobs and the most headroom: today's tuning candidate.",
        "Plan: tune the forest honestly, take one look at the test set, then ship it as a live app.",
    ],
    kicker="Where we are")

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
      "Tuning = try different knob positions, keep the best result."]),
    kicker="Tuning - two kinds of settings",
    note="Baking analogy: the oven temperature is a hyperparameter you set; how the "
         "ingredients turn into cake is the parameters.")

# 5. Grid search
ds.table_slide(
    prs, "Grid search is just cross-validation plus a for-loop - 18 combinations, best CV 0.8259",
    ["n_estimators", "max_depth", "min_samples_leaf", "Mean CV accuracy", "Std"],
    [
        ["100", "4", "5", "0.8259  (best)", "0.0138"],
        ["100", "None", "3", "0.8245", "0.0212"],
        ["100", "4", "3", "0.8245", "0.0143"],
        ["300", "8", "3", "0.8231", "0.0209"],
        ["300", "4", "1", "0.8231", "0.0174"],
    ],
    kicker="Tuning - grid search",
    note="GridSearchCV is nested for-loops around CV: 2x3x3 = 18 combos x 5 folds = 90 "
         "trainings; best gains +0.0322 over the 0.7937 default. Module 2, notebook 3.",
    col_widths=[1.0, 0.9, 1.2, 1.3, 0.7])

# 6. Random search + tuning arc chart
ds.image_slide(
    prs, "Fifteen random tries beat the whole grid - more distinct values per knob",
    f"{FIGS}/tuning_arc.png",
    kicker="Tuning - random search",
    bullets=[
        "Grids explode: 5 knobs x 6 values = 7,776 combos - nearly 39,000 trainings.",
        "15 random draws test up to 15 distinct values of every knob at once.",
        "If only one or two knobs really matter (typical), random explores them better.",
        "Best found: max_depth 8, max_features None, min_samples_leaf 3, n_estimators 314.",
        "CV 0.8274 - beats the grid's 0.8259 with fewer tries.",
    ],
    caption="Random Forest 5-fold CV accuracy at each tuning stage. Module 2, notebook 3.")

# 7. The honest final exam
ds.image_slide(
    prs, "The test set is used exactly once - tune against it and it becomes training data",
    f"{FIGS}/cv_vs_test.png",
    kicker="The honest final exam",
    bullets=[
        "Final exam: 179 passengers the model has never seen, scored one time.",
        "Test accuracy 0.8268, F1 0.7597.",
        "Test is close to CV (0.8274): the tuning did not fool itself.",
        "Peeking at test scores while tuning silently turns the test set into training data.",
        "Honest expectation: tuning buys a percentage point or two, not miracles.",
    ],
    caption="Tuned forest: CV score vs the single test-set look. Module 2, notebook 3.")

# 8. Section divider: production
ds.section_slide(prs, "02", "Ship it",
                 "A model in a notebook helps nobody - ship it where anyone "
                 "can ask about a passenger like Owen or Florence.")

# 9. What production means - the loop diagram
ds.image_slide(
    prs, "Production is a loop, not a finish line - the model runs where "
         "real users can reach it",
    f"{FIGS_STORY}/s4_production_loop.png",
    kicker="From notebook to production",
    bullets=[
        "Save the WHOLE pipeline: titanic_model_v1.joblib, 2,599 KB - "
        "raw data in, prediction out",
        "Re-coding the prep by hand invites silently wrong predictions",
        "Load in a fresh process: no retraining, no notebook required",
        "Version the file name (v1) so you always know what is running - "
        "and can roll back",
        "Monitoring closes the loop: drift detected means retrain and reship",
    ],
    caption="The production loop: save, load, wrap, host, monitor, retrain. "
            "Module 2, notebook 3.")

# 10. Two passengers - the moment it becomes real
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

# 11. Gradio + Hugging Face Spaces
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

# 12. After launch: reality checks (Netflix Prize folded in)
ds.bullets_slide(
    prs, "After launch, reality bites: models go stale and accuracy alone does not ship",
    [
        ("The $1M Netflix Prize ensemble (about 800 models, a 10% RMSE gain) "
         "was never put into production.",
         ["Engineering cost outweighed the accuracy gain, and the business had "
          "moved to streaming. Netflix tech blog, 2012."]),
        ("Data drift: the world changes and yesterday's model slowly goes stale.",
         ["A model trained on 2024 customers may misjudge 2026 customers."]),
        ("Google Flu Trends overestimated flu by more than 50% in 2011-13.",
         ["It once claimed about 11% of the US had flu; CDC data said about 6%.",
          "It was essentially never retrained after 2009. Lazer et al., Science, 2014."]),
        "Version every saved model (model_v1, model_2026_09_06) so you can always roll back.",
        "Retrain on a schedule and re-score; monitoring is step 5 of the loop, not an afterthought.",
    ],
    kicker="After launch")

# 13. The module's arc, closed: descriptive analytics vs machine learning
ds.image_slide(
    prs, "M1 dashboards inform people - M2 models act inside products",
    f"{FIGS_CONCEPTS}/echo_da_vs_ml.png",
    kicker="The module's arc, closed",
    caption="The contrast that opened Session 1, now with the module behind "
            "it: M1 described the crowd; M2 answers about the next person. "
            "Module 2, notebooks 1 and 3.")

# 14. The whole module on one slide
ds.table_slide(
    prs, "The whole module is one honest pipeline: prep, select, tune, ship",
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

# 15. Close
ds.close_slide(
    prs, "Session 4 - what to remember",
    [
        "Hyperparameters are knobs you set before training; search them with CV, never the test set.",
        "Random search reached CV 0.8274; the single test look scored 0.8268 - the tuning did not fool itself.",
        "Ship the whole pipeline: save, load, wrap in Gradio, host on Spaces, monitor.",
        "Models go stale: version them, watch for drift, retrain on a schedule.",
        "The manifest's last word: Survived = 0 for Owen, 1 for Florence, 1 for Frankie. The odds were never abstract.",
        "Practice now: notebook 03-model-optimization-and-deployment in Colab.",
    ])

out_path = "../m2-session-4-optimization-and-deployment.pptx"
ds.save_deck(prs, out_path, "M2 Session 4 - Optimization and Deployment")
print(f"Slides: {len(prs.slides)}")
print(f"Saved: {os.path.abspath(out_path)}")
