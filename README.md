# Thyroid Triage AI

A confounder-aware, uncertainty-calibrated multi-agent clinical decision-support system for thyroid function triage.

**Author:** Sumashree Dornala

---

## Overview

Thyroid Triage AI takes a patient's thyroid hormone panel (TSH, T3, TT4, T4U, FTI) alongside demographic and clinical context, and produces an automated triage recommendation with two things most thyroid-classification systems don't offer: **explicit detection of lab-interference confounders** (biotin interference, macro-TSH, discordant assay patterns, non-thyroidal illness) and **statistically calibrated uncertainty** on every prediction via conformal prediction, rather than a bare, overconfident point estimate.

The system is built as a five-agent pipeline, each stage auditable independently:

```
Patient Data
   → Risk Scorer          (Random Forest, multi-class: Low / Medium / High)
   → Confounder Detector  (rule-based interference screening)
   → Conformal Wrapper    (calibrated prediction sets, coverage guarantee)
   → Retriever            (RAG over clinical guideline evidence)
   → Reasoner             (structured clinical impression + findings)
   → Summarizer           (doctor-facing and patient-facing reports)
```

---

## What's novel here

Most public thyroid-ML projects optimize for classifier accuracy on a static dataset and stop there. This system addresses three gaps identified in the current literature and in the original codebase this project builds on:

- **Confounder-Detection Agent** — operationalizes a published clinical interference-screening algorithm (Favresse et al., *Endocrine Reviews*, 2018) as an automated pre-triage check, flagging lab patterns inconsistent with true thyroid pathology (e.g., normal TSH with elevated free hormones — a biotin/assay interference signature) rather than letting the classifier triage them at face value. Validated on both synthetically injected interference cases and a real held-out set of natively discordant (`R`-class) patients from the UCI thyroid dataset.
- **Conformal Prediction Wrapper** — replaces the classifier's raw probability output with a calibrated prediction set carrying a formal coverage guarantee (e.g., "the true risk tier is in this set with 95% confidence"), computed via a held-out calibration split, distinct from bootstrap-based approaches used elsewhere in the literature.
- **Real diagnostic labels** — the project's underlying classification task was originally built on a synthetic, single-feature-derived label (a hard TSH threshold), which produced artificially perfect accuracy and masked any real uncertainty. This was identified and corrected by re-sourcing the full multi-class UCI thyroid diagnostic taxonomy (`thyroid0387`) and remapping it into a clinically grounded three-tier risk system, giving the classifier — and the conformal wrapper built on top of it — an actual, non-trivial task to learn.

---

## Architecture

| Component | Role |
|---|---|
| `src/agents/risk_scoring.py` | Random Forest classifier over the thyroid hormone panel, trained on real multi-class diagnostic labels |
| `src/agents/confounder.py` | Rule-based screen for biotin interference, macro-TSH, incoherent TSH/FT4 patterns, and non-thyroidal illness |
| `src/core/conformal.py` | Score-based conformal prediction wrapper producing calibrated, coverage-guaranteed prediction sets |
| `src/agents/retriever.py` | Retrieval-augmented evidence lookup against ATA/Endocrine Society clinical guidelines |
| `src/agents/reasoner.py` | Synthesizes risk score, confounder flags, conformal uncertainty, and retrieved evidence into a structured clinical impression |
| `src/agents/summarizer.py` | Generates separate doctor-facing (technical) and patient-facing (plain-language) reports |
| `api.py` | FastAPI service exposing `/triage`, `/batch-triage`, `/health`, and `/about` |
| `static/` | Web frontend for interactive triage input and report visualization |

---

## Evaluation

Evaluated on a held-out test split of the UCI Thyroid Disease dataset (~3,772 patients), with a separate calibration split for conformal prediction and a synthetic interference-injection set plus a natively-labeled real confounder set for validating the interference detector:

- **Baseline classifier accuracy:** 97.88% on real multi-class diagnostic labels
- **Confounder detection:** strong recall on synthetic interference patterns; ~15% recall on natively occurring discordant (`R`-class) real patient cases, with an 8.13% false-positive rate on genuine normals — reported honestly as a limitation motivating future learned or LLM-assisted detection, not oversold as a solved problem
- **Conformal prediction:** empirical coverage tracking the target confidence level, with explicit (not silently masked) handling of low-confidence "empty set" cases

Full evaluation methodology and results are in `results/evaluation_log.csv` and the accompanying build documentation.

---

## Tech Stack

Python · FastAPI · scikit-learn · pandas · conformal prediction (score-based / LAC method) · RAG-based evidence retrieval · HTML/CSS/JS frontend

---

## Getting Started

```bash
git clone https://github.com/sumashree29/Thyroid-Triage-AI.git
cd Thyroid-Triage-AI
pip install -r requirements.txt
python train_model.py                     # trains on real diagnostic labels
python scripts/calibrate_conformal.py      # computes conformal calibration threshold
uvicorn api:app --reload
```

Then open `http://localhost:8000` for the interactive triage interface, or POST to `/triage` directly.

---

## Limitations

- Confounder detection rules are threshold-based and calibrated primarily against synthetic interference patterns; real-world interference presents with more diversity than static rules currently capture.
- Evaluation is single-dataset (UCI Thyroid Disease); prospective clinical validation on an independent patient population has not been performed.
- This system is intended for clinical decision **support**, not autonomous diagnosis — all outputs require review by a qualified healthcare provider.

---

## Citation

If you use this work, please cite the associated paper (details to follow upon publication).

---

## Author

**Sumashree Dornala**
Final-year B.Tech Data Science, GRIET, Hyderabad

---

## License

MIT License
