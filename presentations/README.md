# M2 Session Decks

Four teaching presentations for the "M2 - Machine Learning" module. Each session pairs one deck with one notebook. The decks introduce machine learning from zero (including the bridge from M1's descriptive analytics), follow one continuous story — the 891 passengers of the Titanic, with three real passengers from the manifest recurring across all sessions — and use only numbers actually computed in the course notebooks.

| Session | Deck | Companion notebook |
|---------|------|--------------------|
| 1 | `m2-session-1-machine-learning-fundamentals.pptx` | 01, sections 1-3 |
| 2 | `m2-session-2-data-prep-and-feature-engineering.pptx` | 01, sections 4-10 |
| 3 | `m2-session-3-model-selection-and-evaluation.pptx` | 02 |
| 4 | `m2-session-4-optimization-and-deployment.pptx` | 03 |

15 slides per deck. White and blue design, action titles, real charts.

## Regenerating the decks

The decks are fully generated from code in `src/` (shared design system in `src/deck_style.py`, one builder per deck). To rebuild:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install python-pptx matplotlib numpy pandas
cd src
python build_session_1.py   # and 2, 3, 4
```

Each builder regenerates its own charts and diagrams (to a temp folder) and writes the PPTX one level up. External references cited on slides (course syllabi, published papers) are attributed directly on the slides.
