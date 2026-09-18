"""Build M2 Session 2 deck: Data Preparation and Feature Engineering.

All numbers come from M2/01-data-prep-and-feature-engineering.ipynb
(executed outputs, see notebook brief). Charts regenerated from those values.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import deck_style as ds

pal = ds.mpl_theme()
FIGS = "/tmp/deck-workshop/figs2"
FIGS_STORY = "/tmp/deck-workshop/figs-story"
os.makedirs(FIGS, exist_ok=True)
os.makedirs(FIGS_STORY, exist_ok=True)

# ---------------- Figure 1: imputation - fit on train only ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.5, 4))
ax = axes[0]
vals = [137, 40]
ax.bar(["Train\n(712 rows)", "Test\n(179 rows)"], vals,
       color=[pal["blue"], pal["sky"]])
ax.set_title("Missing Age rows")
for i, v in enumerate(vals):
    ax.text(i, v + 4, str(v), ha="center", fontweight="bold", color=pal["ink"])
ax.set_ylim(0, 160)
ax = axes[1]
meds = [28.5, 27.0]
ax.bar(["Train median\n(used for both)", "Test median\n(never used)"], meds,
       color=[pal["blue"], pal["sky"]])
ax.set_title("Median Age learned")
for i, v in enumerate(meds):
    ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontweight="bold",
            color=pal["ink"])
ax.set_ylim(0, 33)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_impute.png")
plt.close(fig)

# ---------------- Figure 2: raw scales - Fare dwarfs Age ----------------
fig, ax = plt.subplots(figsize=(7.5, 4))
feats = ["Age", "Fare"]
maxes = [80, 512.3]
labels = ["0 to 80", "0 to 512.3"]
bars = ax.barh(feats, maxes, color=[pal["sky"], pal["blue"]])
for bar, lab in zip(bars, labels):
    ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2, lab,
            va="center", fontweight="bold", color=pal["ink"])
ax.set_title("Raw value ranges before scaling")
ax.set_xlabel("Value")
ax.set_xlim(0, 610)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_scales.png")
plt.close(fig)

# ---------------- Figure 3: grouped Title counts (train) ----------------
fig, ax = plt.subplots(figsize=(7.5, 4))
titles = ["Rare", "Master", "Mrs", "Miss", "Mr"]  # bottom-up for barh
counts = [18, 31, 107, 144, 412]
colors = [pal["sky"], pal["blue"], pal["sky"], pal["sky"], pal["sky"]]
bars = ax.barh(titles, counts, color=colors)
for bar, v in zip(bars, counts):
    ax.text(bar.get_width() + 6, bar.get_y() + bar.get_height() / 2, str(v),
            va="center", fontweight="bold", color=pal["ink"])
ax.set_title("Title counts after grouping, training set")
ax.set_xlabel("Passengers")
ax.set_xlim(0, 470)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_titles.png")
plt.close(fig)

# ---------------- Figure 4: FamilySize distribution (train) ----------------
fig, ax = plt.subplots(figsize=(7.5, 4))
sizes = ["1", "2", "3", "4", "5", "6", "7", "8", "11"]
fam = [434, 129, 77, 22, 14, 16, 11, 4, 5]
colors = [pal["blue"] if s in ("2", "3", "4") else pal["sky"] for s in sizes]
bars = ax.bar(sizes, fam, color=colors)
for bar, v in zip(bars, fam):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 8, str(v), ha="center",
            fontweight="bold", color=pal["ink"], fontsize=11)
ax.set_title("FamilySize distribution, training set (sizes 2-4 highlighted)")
ax.set_xlabel("FamilySize = SibSp + Parch + 1")
ax.set_ylabel("Passengers")
ax.set_ylim(0, 490)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_family.png")
plt.close(fig)

# ------------- Figure 5 (story): the pipeline as a flow diagram -------------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9.2, 3.6))
boxes = [("Raw\npassengers", "12 columns,\ngaps and text", 0.30, "panel"),
         ("Impute", "median Age 28.5,\nEmbarked \"S\"\n(from train)", 2.45, "panel"),
         ("Encode", "one-hot:\n13 x 0/1 columns", 4.60, "panel"),
         ("Scale", "StandardScaler,\n5 numeric columns", 6.75, "panel"),
         ("Model-ready\nmatrix", "712 x 18 train\n179 x 18 test", 8.90, "blue"),
         ("Model", "any of Session\n3's contestants", 11.05, "navy")]
for name, sub, x, kind in boxes:
    fc = {"panel": pal["panel"], "blue": pal["blue"], "navy": pal["navy"]}[kind]
    txt = pal["navy"] if kind == "panel" else "white"
    sub_c = pal["gray"] if kind == "panel" else pal["sky"]
    ax.add_patch(FancyBboxPatch((x, 1.30), 1.75, 1.55,
                                boxstyle="round,pad=0.05", facecolor=fc,
                                edgecolor=pal["blue"], lw=1.3))
    ax.text(x + 0.875, 2.52, name, ha="center", va="center",
            fontweight="bold", color=txt, fontsize=11.5)
    ax.text(x + 0.875, 1.78, sub, ha="center", va="center",
            color=sub_c, fontsize=8.2)
for x0 in (2.10, 4.25, 6.40, 8.55, 10.70):
    ax.add_patch(FancyArrowPatch((x0, 2.07), (x0 + 0.32, 2.07),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=pal["blue"], lw=1.5))
# bracket: the three middle steps live inside one Pipeline object
ax.plot([2.45, 2.45, 8.50, 8.50], [3.18, 3.42, 3.42, 3.18],
        color=pal["navy"], lw=1.3)
ax.text(5.475, 3.62, "one Pipeline + ColumnTransformer: fit on train only, "
        "transform train and test", ha="center", color=pal["navy"],
        fontsize=10, fontweight="bold")
ax.text(1.175, 0.88, "Name, Ticket, Cabin, PassengerId:\ndropped automatically",
        ha="center", color=pal["gray"], fontsize=8.2)
ax.set_xlim(0, 13.1)
ax.set_ylim(0.4, 4.0)
ax.axis("off")
fig.savefig(f"{FIGS_STORY}/s2_pipeline_flow.png")
plt.close(fig)

# ------- Figure 6 (story): one passenger's row transforms (Frankie) -------
# All raw values verbatim from the Titanic CSV, PassengerId 166.
fig, ax = plt.subplots(figsize=(9.6, 4.0))
panels = [
    ("Raw row (PassengerId 166)",
     'Name: Goldsmith, Master. Frank\nJohn William "Frankie"\n'
     "Pclass 3    Sex male    Age 9\nSibSp 0    Parch 2    Fare 20.525\n"
     "Embarked S    Cabin (blank)",
     0.25, 3.30, "panel"),
    ("Engineered",
     'Title = "Master"\n(regex on Name)\nFamilySize = 0 + 2 + 1 = 3\n'
     "IsAlone = 0",
     4.75, 2.80, "blue"),
    ("18 numbers the model sees",
     "Title_Master, Pclass_3,\nSex_male, Embarked_S = 1\n"
     "(the other 9 one-hot\ncolumns stay 0)\n"
     "Age, Fare, SibSp, Parch,\nFamilySize: scaled",
     8.75, 3.30, "navy"),
]
for head, body, x, w, kind in panels:
    fc = {"panel": pal["panel"], "blue": pal["blue"], "navy": pal["navy"]}[kind]
    txt = pal["navy"] if kind == "panel" else "white"
    body_c = pal["ink"] if kind == "panel" else "white"
    ax.add_patch(FancyBboxPatch((x, 1.30), w, 2.10,
                                boxstyle="round,pad=0.06", facecolor=fc,
                                edgecolor=pal["blue"], lw=1.3))
    ax.text(x + w / 2, 3.12, head, ha="center", va="center",
            fontweight="bold", color=txt, fontsize=11.0)
    ax.text(x + w / 2, 2.18, body, ha="center", va="center",
            color=body_c, fontsize=9.0)
for x0, lab in ((3.72, "engineer\nnew columns"),
                (7.72, "impute, one-hot,\nscale")):
    ax.add_patch(FancyArrowPatch((x0, 2.30), (x0 + 0.86, 2.30),
                                 arrowstyle="-|>", mutation_scale=15,
                                 color=pal["blue"], lw=1.6))
    ax.text(x0 + 0.43, 1.02, lab, ha="center", va="top",
            color=pal["gray"], fontsize=8.0)
ax.text(6.1, 0.32, "IsAlone is computed for every row but not on the model's "
        "feature list - the ColumnTransformer keeps only listed columns.",
        ha="center", color=pal["gray"], fontsize=8.5)
ax.set_xlim(0, 12.2)
ax.set_ylim(0.1, 3.6)
ax.axis("off")
fig.savefig(f"{FIGS_STORY}/s2_row_transform.png")
plt.close(fig)

# ================================ DECK ================================
prs = ds.new_deck()

# Slide 1 - title
ds.title_slide(
    prs, "Session 2 of 4",
    "Data Preparation and Feature Engineering",
    "The manifest is messy: 177 missing ages, titles buried inside names, "
    "scales that disagree. Today we turn passengers into signals - "
    "without cheating.")

# Slide 2 - recap + today's map
ds.bullets_slide(
    prs,
    "Last session we split first - today we build everything else on that rule",
    [
        ("Where we left off: 891 names on the manifest - Owen, Florence, "
         "Frankie and 888 more - split 80/20 into 712 train / 179 test",
         ["Stratified with random_state=42: both halves keep roughly 38% "
          "survivors (0.383 / 0.385)"]),
        "Golden rule: the model must never see the test data during preparation",
        ("Data leakage = information from the test set sneaking into training",
         ["The test set is the final exam: seeing the questions while studying "
          "inflates the score and predicts nothing"]),
        ("Leakage bites professionals too: the KDD Cup 2008 breast-cancer "
         "challenge",
         ["Patient ID 'predicted' cancer - IDs traced to cancer-heavy hospitals "
          "(Kaufman, Rosset and Perlich, KDD 2011)"]),
        ("Today's route: gaps to fill, text to numbers, scales to align, new "
         "features to invent",
         ["Then one Pipeline that does all of it in the right order, every time"]),
    ],
    kicker="Recap and roadmap",
    note="Split first, prepare after. Splits and counts: Module 2, notebook 1.")

# Slide 3 - the three requirements
ds.table_slide(
    prs,
    "Models are picky eaters: they want numbers only, no gaps, sensible scales",
    ["The model demands", "Raw Titanic reality", "The fix"],
    [
        ["Numbers only",
         'Sex is "male"/"female"; Embarked is C/Q/S; Name is free text',
         "Encoding: turn categories into 0/1 columns"],
        ["No gaps",
         "Age missing in 177 of 891 rows (19.9%); Embarked missing in 2",
         "Imputation: fill gaps with a value learned from train"],
        ["Sensible scales",
         "Age runs 0 to 80 while Fare runs 0 to 512",
         "Scaling: put features on a comparable ruler"],
    ],
    kicker="Why preparation exists",
    note="Why so picky? A model sees each passenger as a point in space, one "
         "axis per column - words and gaps have no place on an axis. Impute = "
         "fill a gap with a learned substitute. Harvard's CS109A gives missing "
         "data its own lecture (Fall 2021). Counts: Module 2, notebook 1.",
    col_widths=[0.22, 0.44, 0.34])

# Slide 4 - imputation the ML way
ds.image_slide(
    prs,
    "Fit the imputer on train only - the test median is information we refuse to use",
    f"{FIGS}/fig_impute.png",
    kicker="Missing values the ML way",
    bullets=[
        'SimpleImputer(strategy="median"): fit(train) learns median Age 28.5',
        "transform(train) and transform(test) both fill gaps with that same 28.5",
        'The test-only median (27.0) is "exactly the information we refuse to use"',
        "Never call fit on the test set - treat it like the future",
        'Embarked (2 gaps): strategy="most_frequent" learns "S" from train',
    ],
    caption="Missing Age rows and learned medians. Module 2, notebook 1.")

# Slide 5 - one-hot encoding by hand
ds.table_slide(
    prs,
    "One-hot gives each category its own 0/1 column - exactly one is hot per row",
    ["Passenger", "Embarked", "Embarked_C", "Embarked_Q", "Embarked_S"],
    [
        ["1", "S", "0", "0", "1"],
        ["2", "C", "1", "0", "0"],
        ["3", "Q", "0", "1", "0"],
        ["4", "S", "0", "0", "1"],
    ],
    kicker="Encoding categorical features",
    note="Encoding = translating words into coordinates the model can plot. "
         'Why not C=1, Q=2, S=3? That would tell the model S is "three times" C '
         "- and models will try to use it. A fake order lies about distance. "
         "Module 2, notebook 1.",
    col_widths=[0.18, 0.22, 0.20, 0.20, 0.20])

# Slide 6 - the get_dummies trap
ds.two_col_slide(
    prs,
    "sklearn encoders remember the training columns - pd.get_dummies does not",
    ("pd.get_dummies: columns drift",
     [
         "Builds columns from whatever categories it happens to see",
         "Train and test batches can end up with different columns",
         "Notebook demo: encoding train and test separately misaligns them",
         "The model then crashes - or silently misreads features",
     ]),
    ("OneHotEncoder: train fixes the columns once",
     [
         "fit(train) memorizes the exact column layout",
         "transform(test) always produces those same columns",
         'handle_unknown="ignore": unseen category = all zeros, no crash',
         "Same contract as the imputer: learn on train, apply everywhere",
     ]),
    kicker="Encoding categorical features",
    note="Rule of thumb: real order (shirt sizes S < M < L) -> OrdinalEncoder; "
         "no real order -> OneHotEncoder; when in doubt, one-hot. A fake order "
         "hurts more than a few extra columns.")

# Slide 7 - scaling
ds.image_slide(
    prs,
    "Scaling re-labels the ruler: distance models care, trees never notice",
    f"{FIGS}/fig_scales.png",
    kicker="Feature scaling",
    bullets=[
        "Geometry first: each passenger is a point in space, one axis per "
        "feature - scaling makes the axes comparable",
        "Distance and equation models (KNN, Logistic Regression) let the "
        "big-number axis dominate: raw Fare gaps drown out Age gaps",
        'Trees only ask questions like "is Fare > 30?" - scale never matters',
        "StandardScaler: Fare becomes mean -0.00, std 1.00. MinMaxScaler: "
        "Fare squeezed into 0.0 to 1.0",
        "The histogram shape does not change - only the numbers on the axis do",
    ],
    caption="Raw training ranges: Age 0 to 80, Fare 0 to 512.3. "
            "Module 2, notebook 1.")

# Slide 8 - section divider
ds.section_slide(
    prs, "02", "Feature engineering",
    "Where humans still beat machines: the manifest already knows Frankie is "
    "a boy and Owen is not alone - our job is to put that into columns")

# Slide 9 - Title from Name
ds.image_slide(
    prs,
    "Title packs sex, age, and status into one column - and finds the boys",
    f"{FIGS}/fig_titles.png",
    kicker="Feature engineering: Title",
    bullets=[
        'Regex " ([A-Za-z]+)\\." grabs the word before the period - the title',
        '"Cumings, Mrs. John Bradley" yields Mrs for Florence; 14 raw titles '
        "group into 5 buckets: Mr, Miss, Mrs, Master, Rare",
        'Frankie, age 9, is a "Master" - the old title for young boys, '
        'secretly encoding "male child"',
        "To the Sex column, Owen (22) and Frankie (9) are both just male - "
        "Title tells the man from the boy",
        "Signal check: Mrs and Miss survived far more often than Mr; Master "
        "sits well above adult men",
    ],
    caption="Grouped Title counts, training set (712 rows). "
            "Module 2, notebook 1.")

# Slide 10 - FamilySize and IsAlone
ds.image_slide(
    prs,
    "FamilySize = SibSp + Parch + 1 - small families of 2 to 4 fared best",
    f"{FIGS}/fig_family.png",
    kicker="Feature engineering: FamilySize",
    bullets=[
        "FamilySize = SibSp + Parch + 1: everyone counts themselves",
        "Owen: SibSp 1, Parch 0 -> FamilySize 2. Frankie: SibSp 0, Parch 2 "
        "-> FamilySize 3",
        "IsAlone = 1 when FamilySize is 1: 434 of 712 training passengers - "
        "none of our three",
        "Notebook charts: solo travelers and very large families fared badly; "
        "small families of 2 to 4 did best",
        "More features is not automatically better - always check for signal",
    ],
    caption="FamilySize distribution, training set. Module 2, notebook 1.")

# Slide 11 - Ng's area feature
ds.two_col_slide(
    prs,
    "Multiply two weak columns and you can get one strong feature - Ng's area example",
    ("Andrew Ng - Machine Learning Specialization",
     [
         "A house lot has frontage (x1) and depth (x2)",
         "New feature: x3 = x1 * x2 = lot area",
         "Area is often more predictive than either raw column",
         "One multiplication - no new data collected",
     ]),
    ("The same move on the Titanic",
     [
         "SibSp + Parch + 1 = FamilySize: one column beats two",
         "Name -> Title: sex, age, and status compressed into one",
         "Master = young boy: a signal Sex and Age only hint at",
         "Good features often improve a model more than a fancier algorithm",
     ]),
    kicker="Feature engineering, attributed",
    note='Ng\'s rule for inventing features: "use knowledge or intuition about '
         'the problem to design new features."')

# Slide 12 - one passenger's row transforms (story visual)
ds.image_slide(
    prs,
    "The engineered row says what the raw row only hints: Frankie is a boy "
    "traveling with family",
    f"{FIGS_STORY}/s2_row_transform.png",
    kicker="Engineering in action",
    caption="PassengerId 166, raw fields verbatim from the dataset. "
            "Module 2, notebook 1.")

# Slide 13 - pipelines: the whole machine in one picture
ds.image_slide(
    prs,
    "One Pipeline remembers every step - leakage and forgotten steps become "
    "impossible",
    f"{FIGS_STORY}/s2_pipeline_flow.png",
    kicker="Putting it all together",
    bullets=[
        "Numeric (Age, Fare, SibSp, Parch, FamilySize): impute median, "
        "then StandardScaler",
        "Categorical (Pclass, Sex, Embarked, Title): impute most frequent, "
        'then OneHotEncoder(handle_unknown="ignore")',
        "Name, Ticket, Cabin, PassengerId: unlisted, dropped automatically",
        "fit_transform(train) learns; transform(test) only applies",
        "Three reasons: no leakage, no forgotten steps, portability",
    ],
    caption="The canonical prep, reused verbatim by Sessions 3 and 4. "
            "Module 2, notebook 1.")

# Slide 13 - proof it works
ds.big_number_slide(
    prs,
    "The smoke test passes: a plain Logistic Regression scores 0.838 on "
    "unseen data",
    "0.838",
    "Test accuracy on the 179 held-out passengers - train accuracy 0.829, "
    "so no memorization gap",
    foot="Input: the 712 x 18 prepared matrix. Just a smoke test - choosing "
         "the best model is next session's whole job. Module 2, notebook 1.",
    kicker="Sanity checkpoint")

# Slide 14 - close
ds.close_slide(
    prs,
    "Three golden rules",
    [
        "Split first: 80/20 before any preparation (seed 42, stratified)",
        "Fit on train only: imputers, encoders, and scalers learn from the "
        "712 training rows",
        "Pipelines always: one object that remembers every step and kills leakage",
        "Good features often improve a model more than a fancier algorithm does",
        "Practice now: notebook 01-data-prep-and-feature-engineering in Colab",
    ])

out = "../m2-session-2-data-prep-and-feature-engineering.pptx"
ds.save_deck(prs, out, "M2 Session 2 - Data Prep and Feature Engineering")
print(f"Slides: {len(prs.slides)}")
print(f"Saved: {os.path.abspath(out)}")
