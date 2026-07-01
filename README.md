# Malware Detection — Reproduction & Critical Evaluation

**Course:** Data Science in Cyber — Final Project (Dr. Uri Itai)

---

## Project description

This project reproduces and critically evaluates a published malware-detection tutorial that
classifies Windows PE (`.exe`) files as malware or legitimate using only static PE-header
metadata and three classifiers (KNN, Decision Tree, Random Forest).

We reproduce the source's exact methodology (**Pipeline A**), then identify and empirically
correct three significant methodological problems in **Pipeline B**:

1. **Train/test leakage** — 76 % of rows are exact duplicates; the source splits *before*
   deduplication, so near-identical rows appear in both train and test sets.
2. **Mislabelled metric** — the reported "F1 score" (`f1_score(average='micro')`) is
   mathematically identical to plain accuracy for a binary classification task.
3. **No feature scaling for KNN** — KNN is distance-based; without standardisation, large-scale
   features dominate Euclidean distance by magnitude alone, not relevance.

**KNN k value:** The source's KNN notebooks call `KNeighborsClassifier()` with no arguments,
which uses sklearn's default `n_neighbors=5`. The source's README ambiguously describes the
method as "k=1 KNN", but the executable code uses the sklearn default of k=5. Both pipelines
in this notebook therefore use **k=5** to faithfully reproduce the source's executable behaviour.

---

## Selected article / tutorial

- **Source reproduced:**
  https://github.com/emr4h/Malware-Detection-Using-Machine-Learning
- **Original GitHub repository:**
  https://github.com/emr4h/Malware-Detection-Using-Machine-Learning

---

## Repository structure

```
malware-detection-critical-evaluation/
├── README.md                                    <- this file
├── requirements.txt                             <- Python dependencies
├── report.pdf                                   <- full written report (PDF)
├── Malware_Detection_Critical_Evaluation.ipynb  <- complete, executable notebook
├── data-set/
│   └── .gitkeep   <- placeholder; MalwareDataSet.csv is downloaded automatically
└── figures/
    └── .gitkeep   <- placeholder; PNG figures are saved here when the notebook runs
```

> **Dataset note:** `data-set/MalwareDataSet.csv` does **not** exist in this repository.
> The notebook creates it automatically on first run by downloading it from the source GitHub
> repository. No manual download is required.

---

## Dataset source

`MalwareDataSet.csv` — 137,444 rows × 9 columns (8 PE-header features + binary label).

Download URL (automatic):
https://raw.githubusercontent.com/emr4h/Malware-Detection-Using-Machine-Learning/main/data-set/MalwareDataSet.csv

**Label note:** The column is named `legitimate`, but per the source's data-generation code,
`legitimate = 1` means **malware** and `legitimate = 0` means **safe/legitimate**. The
notebook renames it to `is_malware` at load time and documents this discovery.

---

## Execution instructions

```bash
# 1. Clone this repository
git clone <your-repo-url>
cd malware-detection-critical-evaluation

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook — Kernel → Restart & Run All
jupyter notebook Malware_Detection_Critical_Evaluation.ipynb
```

On first run the notebook will:
1. Create `data-set/` and `figures/` directories if they do not exist.
2. Download `MalwareDataSet.csv` automatically.
3. Execute all analysis cells in order and save all figures to `figures/`.

> **Note:** The notebook was prepared in an offline environment and must be run locally
> to generate cell outputs. All results in `report.pdf` were produced by running this
> notebook with `RANDOM_SEED = 42`.

---

## Summary of findings

| | Pipeline A (naive) | Pipeline B (corrected) |
|---|:---:|:---:|
| Deduplication before split | ✗ | ✓ |
| Stratified split | ✗ | ✓ |
| KNN scaling (inside Pipeline) | ✗ | ✓ |
| Random Forest accuracy | ~99 % | ~95.7 % |
| Random Forest MCC | ~0.976 | ~0.891 |

The ~4 % accuracy drop and ~0.085 MCC drop between pipelines quantify the leakage
artefact in the source's reported numbers. The core hypothesis — static PE header metadata
carries real signal for malware detection — survives correction at a lower but still
respectable level.

Full details, metric definitions, and error analysis are in `report.pdf`.
