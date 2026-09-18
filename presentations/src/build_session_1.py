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
os.makedirs(FIGS, exist_ok=True)
os.makedirs(FIGS_STORY, exist_ok=True)
os.makedirs(FIGS_CONCEPTS, exist_ok=True)
pal = ds.mpl_theme()


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
        "sessions: could the data have told us who lives?",
    )

    # 2. HERO: descriptive analytics vs machine learning
    ds.image_slide(
        prs,
        "Descriptive analytics reads the past - machine learning answers "
        "about the future",
        f"{FIGS_CONCEPTS}/hero_da_vs_ml.png",
        kicker="The bridge from M1",
        caption="M1 drew the line itself: describing is not predicting - "
                "'we stop here.' M2 starts exactly there, on the same 891 "
                "passengers.",
    )

    # 3. The analytics ladder
    ds.image_slide(
        prs,
        "Prediction is the next rung on a ladder you are already halfway up",
        f"{FIGS_CONCEPTS}/ladder_analytics.png",
        kicker="The analytics ladder",
        bullets=[
            "Descriptive and diagnostic: M1. You told what happened and dug "
            "into why - groupbys, crosstabs, correlations",
            "Predictive: M2, this module. What will happen to a new, unseen "
            "case?",
            "Prescriptive sits on top: what should we do - decisions built "
            "on predictions you can trust",
            "Your M1 skills stay load-bearing: 80% of the daily job is "
            "still cleaning and describing data",
            "The climb, session by session: S1 what ML is and honest "
            "splits; S2 features; S3 the model tournament; S4 tune and ship",
        ],
        caption="Reference points: Andrew Ng's specialization, Harvard "
                "CS109A and MIT 6.390 climb the same way - one worked "
                "example, honest validation early, features first-class.",
    )
    # 4. What machine learning is
    ds.bullets_slide(
        prs,
        "ML learns the rules from examples instead of being told them",
        [
            ("Classic programming: a human writes the rules by hand",
             ["A hand-written spam rule list - block 'winner', block ALL "
              "CAPS, block unknown senders - never ends, and spammers adapt "
              "faster than you can type"]),
            ("Machine learning: you show labeled examples, the algorithm "
             "finds the rules",
             ["Thousands of emails already flagged spam or not-spam go in; "
              "a filter that judges brand-new email comes out"]),
            ("The same move on our data",
             ["891 passengers with known outcomes go in; a pattern that "
              "predicts new passengers comes out"]),
            ("Supervised learning = learning from examples that include the "
             "right answer (the label)",
             ["Our label: Survived, 1 or 0"]),
            "A model is a function with adjustable knobs (parameters); "
            "training turns the knobs until predictions match the examples",
        ],
        kicker="What ML is",
        note="Spam filtering has been Andrew Ng's canonical everyday example "
             "since his original 2012 ML course; his specialization also "
             "opens with supervised learning taught through worked examples.",
    )

    # 5. Types of machine learning
    ds.table_slide(
        prs,
        "Machines learn in three ways - this module is supervised "
        "classification",
        ["Type", "How it learns", "Everyday example", "In this module?"],
        [
            ["Supervised",
             "From labeled examples: inputs plus the right answer",
             "A spam filter trained on mail users already flagged",
             "Yes - all of M2"],
            ["Unsupervised",
             "Finds structure in unlabeled data - no right answer is given",
             "Grouping a shop's customers by what they tend to buy",
             "No"],
            ["Reinforcement",
             "By trial and error: actions earn rewards or penalties",
             "Game-playing agents and robots improving with practice",
             "No"],
        ],
        kicker="Types of machine learning",
        note="M2 end to end: supervised classification - the label is "
             "Survived (1 or 0), a category. Predicting a number instead "
             "(say, Fare) would be supervised regression.",
        col_widths=[1.3, 3.2, 3.2, 1.4],
    )

    # 6. ML is already in your day
    ds.bullets_slide(
        prs,
        "You already used machine learning today - probably before breakfast",
        [
            ("Spam filter: classifies every new email as spam or not",
             ["learned from millions of messages users already flagged"]),
            "Music and film recommendations: predict what you will want "
            "next from what you (and people like you) played before",
            "Map arrival times: predict the duration of a trip nobody has "
            "driven in exactly these conditions",
            "Phone face unlock: predicts whether the face in a brand-new "
            "camera frame is the owner",
            "Translation apps: predict the most likely sentence in the "
            "other language",
        ],
        kicker="ML in your day",
        note="The common thread: each one answers a question about a NEW "
             "case, and software acts on the answer - the machine-learning "
             "lane of this session's opening contrast.",
    )

    # 5. Features and target
    ds.two_col_slide(
        prs,
        "Every supervised problem is a table: features X in, target y out",
        ("Features X = the inputs", [
            "The columns that describe each passenger",
            "Age, Sex, Pclass, Fare, SibSp, Parch, Embarked",
            "In code: X = everything except the answer",
            "Also called predictors or inputs",
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
             "would make this a regression problem instead.",
    )

    # 6. The dataset (chart)
    ds.image_slide(
        prs,
        "One dataset carries every concept: 891 passengers, 38.4% survived",
        f"{FIGS}/fig1_target_balance.png",
        kicker="The dataset",
        bullets=[
            "891 rows x 12 columns - small enough to read, real enough "
            "to be messy",
            "549 died, 342 survived: imbalanced, but not extremely",
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
            "Prepare: split, clean, encode, scale (Sessions 1-2)",
            "Train: fit models on the training set only (Session 3)",
            "Evaluate: score on data the model has never seen (Session 3)",
            "Improve, then ship: tune, engineer, deploy (Session 4)",
            "Evaluation sends you back to preparation more often than "
            "forward",
        ],
        caption="Harvard CS109A teaches the same shape: one workflow "
                "revisited all semester on messy real data.",
    )

    # 10. The golden rule: split first (chart)
    ds.image_slide(
        prs,
        "Split before you prepare - even an innocent average can leak",
        f"{FIGS_STORY}/s1_split_story.png",
        kicker="The golden rule",
        bullets=[
            "80/20 split first: 712 training rows, 179 test rows (seed 42)",
            "Data leakage = test-set information sneaking into training",
            "Analogy: the test set is the final exam - seeing the questions "
            "while studying inflates your score and proves nothing",
            "'Fill missing ages with the average' already leaks if that "
            "average was computed over all rows",
            "stratify=y keeps the survivor share equal in both halves",
        ],
        caption="The split and the preserved survivor share. "
                "Source: Module 2, notebook 01, section 3.",
    )

    # 10. Real-world leakage story
    ds.bullets_slide(
        prs,
        "Leakage happens to professionals: a patient ID predicted cancer",
        [
            ("KDD Cup 2008: teams competed to predict breast cancer from "
             "medical images",
             ["A serious research competition with experienced teams"]),
            ("The 'Patient ID' column turned out to be hugely predictive",
             ["IDs were assigned per source institution - and some sources "
              "treated far more cancer-heavy cases"]),
            "The ID leaked where the data came from, not anything about "
            "the tumor itself",
            "Lesson 1: a feature that works suspiciously well deserves "
            "suspicion, not celebration",
            "Lesson 2: this is exactly why identifiers left our feature "
            "list two slides ago",
        ],
        kicker="Real-world leakage",
        note="Source: Kaufman, Rosset & Perlich, 'Leakage in Data Mining', "
             "KDD 2011.",
    )
    # 14. What "good" will mean
    ds.big_number_slide(
        prs,
        "In this module, 'good' means a number - and the floor is 0.617",
        "0.617",
        "Accuracy of a model that always predicts 'died' - the baseline "
        "floor every real model must beat",
        foot="Dummy classifier, 5-fold cross-validation on the training set. "
             "Source: Module 2, notebook 02. Beating the floor is the entry "
             "ticket, not the goal.",
        kicker="Honest evaluation preview",
    )

    # 15. Close (course mechanics folded in)
    ds.close_slide(
        prs,
        "From describing to predicting",
        [
            "ML learns patterns from labeled examples; supervised learning "
            "needs features X and a target y",
            "The Titanic is our spine: 891 passengers - Owen, Florence, "
            "Frankie and 888 more - 38.4% survived",
            "Split first: 712 train / 179 test, stratified - the test set "
            "is the exam you take once",
            "Leakage fools professionals; identifiers and peeking are how "
            "it starts",
            "Every claim gets a number, and the floor is 0.617",
            "Practice now, zero setup: open notebook 01 in Colab (badge "
            "click), Runtime > Run all, work sections 1-3 - exercises at "
            "the end, solutions included, attempt them first",
        ],
    )

    return prs


if __name__ == "__main__":
    make_charts()
    make_loop_chart()
    make_hero_fig()
    make_ladder_fig()
    prs = slides()
    out = "../m2-session-1-machine-learning-fundamentals.pptx"
    ds.save_deck(prs, out, "M2 Session 1 - Machine Learning Fundamentals")
    print("Slides:", len(prs.slides._sldIdLst))
    print("Saved:", os.path.abspath(out))
