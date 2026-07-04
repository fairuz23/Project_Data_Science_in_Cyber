#!/usr/bin/env python3
"""
End-to-end command-line reproduction of the core experiment described in the
notebook and report: naive replication of the source's methodology
(Pipeline A) vs. the corrected methodology (Pipeline B), plus 5-fold
cross-validated MCC on the corrected/deduplicated data.

This script is a thin, scriptable wrapper around the ``src/`` modules — it
does not replace the notebook (which contains the full EDA, feature
engineering exploration, figures, and error analysis), but gives a fast,
non-interactive way to regenerate the headline numbers and save them to
``results/``.

Usage:
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data_loading import DEFAULT_CSV_PATH, download_dataset, feature_columns, load_dataset  # noqa: E402
from src.evaluation import compute_auc_metrics, evaluate, get_score  # noqa: E402
from src.features import deduplicate, get_duplicate_stats  # noqa: E402
from src.models import RANDOM_SEED, build_corrected_models, build_naive_models, train_and_predict  # noqa: E402

RESULTS_DIR = pathlib.Path("results")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    download_dataset(DEFAULT_CSV_PATH)
    df = load_dataset(DEFAULT_CSV_PATH)
    feat_cols = feature_columns(df)

    # --- Duplicate-vector statistics (see caveat in src/features.py) -------
    dup_stats = get_duplicate_stats(df, feat_cols)
    with open(RESULTS_DIR / "duplicate_stats.json", "w") as f:
        json.dump(dup_stats, f, indent=2)
    print("Duplicate feature-vector stats:", dup_stats)

    # --- Pipeline A: naive replication (no dedup, no stratify, no scaling) --
    X_raw = df[feat_cols].values
    y_raw = df["is_malware"].values
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_raw, y_raw, test_size=0.25, random_state=RANDOM_SEED
    )
    preds_naive = train_and_predict(Xr_train, Xr_test, yr_train, build_naive_models())

    # --- Pipeline B: corrected (dedup by feature vector -> stratify -> scale
    #     inside the KNN pipeline only) -------------------------------------
    df_clean = deduplicate(df, feat_cols)
    X = df_clean[feat_cols].values
    y = df_clean["is_malware"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    preds_fixed = train_and_predict(X_train, X_test, y_train, build_corrected_models())

    # --- ROC-AUC / PR-AUC (average precision) on the corrected pipeline ----
    # Threshold-independent ranking metrics; PR-AUC is the more informative of the
    # two under class imbalance. Refit here so we retain each fitted model's
    # score function (train_and_predict above only returns hard predictions).
    auc_rows = []
    for name, model in build_corrected_models().items():
        model.fit(X_train, y_train)
        y_score = get_score(model, X_test)
        auc_rows.append(compute_auc_metrics(y_test, y_score, label=name))
    auc_df = pd.DataFrame(auc_rows).round(4)
    auc_df.to_csv(RESULTS_DIR / "roc_pr_auc.csv", index=False)
    print(auc_df)


    rows = [evaluate(yr_test, pred, f"A: naive replication - {name}") for name, pred in preds_naive.items()]
    rows += [evaluate(y_test, pred, f"B: corrected pipeline - {name}") for name, pred in preds_fixed.items()]
    results_df = pd.DataFrame(rows).set_index("pipeline").round(4)
    results_df.to_csv(RESULTS_DIR / "metrics.csv")
    results_df.reset_index().to_json(RESULTS_DIR / "metrics.json", orient="records", indent=2)
    print(results_df)

    # --- 5-fold stratified CV MCC on the corrected/deduplicated data --------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_rows = []
    for name, model in build_corrected_models().items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="matthews_corrcoef", n_jobs=-1)
        cv_rows.append({"model": name, "mcc_mean": scores.mean(), "mcc_std": scores.std()})
        print(f"{name:15s}  MCC = {scores.mean():.4f} +/- {scores.std():.4f}  (5-fold CV)")
    pd.DataFrame(cv_rows).round(4).to_csv(RESULTS_DIR / "cv_metrics.csv", index=False)

    # --- KNN scaling ablation (backs the claim in report Section 3: Feature Engineering) ---
    knn_plain = KNeighborsClassifier(n_neighbors=5)
    knn_plain.fit(X_train, y_train)
    mcc_unscaled_split = matthews_corrcoef(y_test, knn_plain.predict(X_test))

    knn_scaled_pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
    knn_scaled_pipe.fit(X_train, y_train)
    mcc_scaled_split = matthews_corrcoef(y_test, knn_scaled_pipe.predict(X_test))

    scores_unscaled = cross_val_score(knn_plain, X, y, cv=cv, scoring="matthews_corrcoef", n_jobs=-1)
    scores_scaled = cross_val_score(knn_scaled_pipe, X, y, cv=cv, scoring="matthews_corrcoef", n_jobs=-1)

    ablation_df = pd.DataFrame([
        {"condition": "unscaled", "mcc_single_split": mcc_unscaled_split,
         "mcc_cv_mean": scores_unscaled.mean(), "mcc_cv_std": scores_unscaled.std()},
        {"condition": "scaled", "mcc_single_split": mcc_scaled_split,
         "mcc_cv_mean": scores_scaled.mean(), "mcc_cv_std": scores_scaled.std()},
    ]).round(4)
    ablation_df.to_csv(RESULTS_DIR / "knn_scaling_ablation.csv", index=False)
    print("\nKNN scaling ablation (backs report Section 3 claim):")
    print(ablation_df)

    print(f"\nSaved results to {RESULTS_DIR.resolve()}/")


if __name__ == "__main__":
    main()
