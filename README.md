# Malware Detection — Reproduction \& Critical Evaluation

**Course:** Data Science in Cyber — Final Project (Dr. Uri Itai)
**Student:** Fairuz Khalil

\---

## Project description

This project reproduces and critically evaluates a published malware-detection tutorial that
classifies Windows PE (`.exe`) files as malware or legitimate using only static PE-header
metadata and three classifiers (KNN, Decision Tree, Random Forest).

We reproduce the source's exact methodology (**Pipeline A**), then identify and empirically
correct three significant methodological problems in **Pipeline B**:

1. **Train/test leakage** — 76% of rows are exact duplicate *feature vectors* (identical 8-value
PE-header signatures); the source splits *before* deduplication, so near-identical rows appear
in both train and test sets. **Important caveat:** the dataset has no file hash or unique
identifier, so "duplicate" here means duplicate feature vectors, not confirmed duplicate
*physical files* — two distinct binaries built by the same toolchain could legitimately share
an identical header signature. See `src/features.py` and Section 4 of the report for the full
discussion.
2. **Mislabelled metric** — the reported "F1 score" (`f1\_score(average='micro')`) is
mathematically identical to plain accuracy for a binary classification task.
3. **No feature scaling for KNN** — KNN is distance-based; without standardisation, large-scale
features dominate Euclidean distance by magnitude alone, not relevance.

**KNN k value:** The source's KNN notebooks call `KNeighborsClassifier()` with no arguments,
which uses sklearn's default `n\_neighbors=5`. The source's README ambiguously describes the
method as "k=1 KNN", but the executable code uses the sklearn default of k=5. Both pipelines
in this notebook therefore use **k=5** to faithfully reproduce the source's executable behaviour.

\---

## Selected article / tutorial

* **Source reproduced:**
https://github.com/emr4h/Malware-Detection-Using-Machine-Learning
* **Original GitHub repository:**
https://github.com/emr4h/Malware-Detection-Using-Machine-Learning

\---

## Repository structure

```
malware-detection-critical-evaluation/
├── README.md                                    <- this file
├── requirements.txt                             <- Python dependencies (incl. pytest)
├── report.pdf                                   <- full written report (PDF)
├── Malware_Detection_Critical_Evaluation.ipynb  <- complete, executable notebook (EDA, figures,
│                                                    error analysis, full discussion)
├── src/                                         <- reusable pipeline code, imported by both the
│   ├── __init__.py                                 notebook and scripts/tests
│   ├── data_loading.py                          <- download + load dataset, fix mislabeled column
│   ├── features.py                              <- DllCharacteristics decoding, duplicate stats,
│   │                                                deduplication (feature-vector caveat lives here)
│   ├── models.py                                <- Pipeline A (naive) / Pipeline B (corrected)
│   │                                                model builders
│   └── evaluation.py                            <- accuracy/precision/recall/F1/F2/MCC metric suite
├── scripts/
│   ├── download_data.py                         <- standalone, idempotent dataset download
│   └── run_pipeline.py                          <- non-interactive CLI reproduction of the
│                                                    headline experiment; writes to results/
├── tests/
│   ├── conftest.py
│   ├── test_data_loading.py
│   ├── test_features.py                         <- covers duplicate-vector counting + decoding
│   └── test_evaluation.py
├── results/                                      <- saved output artifacts (committed)
│   ├── README.md
│   ├── duplicate_stats.json                      <- duplicate-FEATURE-VECTOR counts + caveat
│   ├── metrics.csv / metrics.json                <- Pipeline A vs. B metrics per model
│   ├── cv_metrics.csv                             <- 5-fold CV MCC (corrected pipeline)
│   ├── roc_pr_auc.csv                             <- ROC-AUC / PR-AUC (average precision) per model
│   └── knn_scaling_ablation.csv                   <- KNN MCC with vs. without StandardScaler
├── data-set/
│   └── MalwareDataSet.csv  <- downloaded automatically on first run if not already present
└── figures/                                       <- PNG figures (committed)
    ├── 01_feature_distributions.png
    ├── 02_feature_correlation\_heatmap.png
    ├── 03_feature_distributions_by_class.png
    ├── 04_naive_vs_corrected.png
    ├── 05_roc_curves.png
    └── 06_confusion_matrices.png
```

> \*\*Dataset note:\*\* if `data-set/MalwareDataSet.csv` is not present, both the notebook and
> `scripts/download\_data.py` / `scripts/run\_pipeline.py` download it automatically from the
> source GitHub repository. No manual download is required.

## Reproducing outside the notebook

```bash
python scripts/download\_data.py     # fetch MalwareDataSet.csv (skips if already cached)
python scripts/run\_pipeline.py      # runs Pipeline A + B, writes results/\*.csv and \*.json
pytest tests/ -v                    # unit tests for the deduplication, feature-decoding,
                                     # and evaluation logic in src/
```

\---

## Dataset source

`MalwareDataSet.csv` — 137,444 rows × 9 columns (8 PE-header features + binary label).

Download URL (automatic):
https://raw.githubusercontent.com/emr4h/Malware-Detection-Using-Machine-Learning/main/data-set/MalwareDataSet.csv

**Label note:** The column is named `legitimate`, but per the source's data-generation code,
`legitimate = 1` means **malware** and `legitimate = 0` means **safe/legitimate**. The
notebook renames it to `is\_malware` at load time and documents this discovery.

\---

## Execution instructions

```bash
# 1. Clone this repository
git clone <https://github.com/fairuz23/Project\_Data\_Science\_in\_Cyber/tree/main>
cd malware-detection-critical-evaluation

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook — Kernel → Restart \& Run All
jupyter notebook Malware\_Detection\_Critical\_Evaluation.ipynb
```

On first run the notebook will:

1. Create `data-set/` and `figures/` directories if they do not exist.
2. Download `MalwareDataSet.csv` automatically.
3. Execute all analysis cells in order and save all figures to `figures/`.

> \*\*Note:\*\* The notebook is committed with its outputs already saved, so all figures, tables,
> and printed results are visible directly on GitHub without running anything. All results in
> `report.pdf` were produced by running this notebook with `RANDOM\_SEED = 42`; re-running it
> (Kernel → Restart \& Run All) will regenerate the same figures and, aside from minor
> scikit-learn-version-dependent tie-breaking in the naive (non-deduplicated) tree models, the
> same numbers.

\---

## Summary of findings

||Pipeline A (naive)|Pipeline B (corrected)|
|-|:-:|:-:|
|Deduplication before split|✗|✓|
|Stratified split|✗|✓|
|KNN scaling (inside Pipeline)|✗|✓|
|Random Forest accuracy|\~99.2 %|\~95.7 %|
|Random Forest MCC|\~0.981|\~0.891|

The \~3.5 % accuracy drop and \~0.09 MCC drop between pipelines quantify the leakage
artefact in the source's reported numbers. The core hypothesis — static PE header metadata
carries real signal for malware detection — survives correction at a lower but still
respectable level. Random Forest also leads on ranking metrics for the corrected pipeline
(ROC-AUC 0.989, PR-AUC 0.995 — see `results/roc\_pr\_auc.csv` and report Section 5).

Full details, metric definitions, and error analysis are in `report.pdf`.

