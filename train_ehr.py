"""
Train XGBoost and TabPFN classifiers on EHR features for 7 clinical outcomes.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, balanced_accuracy_score, f1_score,
)
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore")

# ── Paths ──
BASE = Path("/mnt/pool/datasets/CY/EchoJEPA/embeddings/mimic")
FEATURES_CSV = BASE / "experiments/nature_medicine/mimic/covariates/ehr_features.csv"
LABELS_DIR = BASE / "data/EHR_labels"
SPLIT_JSON = BASE / "experiments/nature_medicine/mimic/patient_split.json"
RESULTS_DIR = BASE / "experiments/nature_medicine/mimic/covariates/results"
RESULTS_DIR.mkdir(exist_ok=True)

TARGETS = [
    "mortality_30d", "mortality_90d", "mortality_1yr",
    "readmission_30d", "discharge_destination",
    "icu_transfer", "in_hospital_mortality",
]

# Features to drop for specific targets (leakage)
LEAK_COLS = {
    "icu_transfer": ["icu_during_admission", "icu_at_echo"],
}

ID_COLS = ["subject_id", "study_id"]


def load_data():
    features = pd.read_csv(FEATURES_CSV)
    with open(SPLIT_JSON) as f:
        patient_split = json.load(f)
    # map subject_id -> split
    features["split"] = features["subject_id"].astype(str).map(patient_split)
    return features, patient_split


def prepare_task(features: pd.DataFrame, target: str):
    """Join features with labels, add missingness indicators, return X/y per split."""
    labels = pd.read_csv(LABELS_DIR / f"{target}.csv")
    df = features.merge(labels, on=ID_COLS, how="inner")

    drop = ID_COLS + ["split", "label"]
    leak = LEAK_COLS.get(target, [])
    drop += leak

    feat_cols = [c for c in df.columns if c not in drop]

    # Add binary missingness indicators for columns with >10% missing in train
    train_mask = df["split"] == "train"
    miss_rate = df.loc[train_mask, feat_cols].isnull().mean()
    miss_cols = miss_rate[miss_rate > 0.10].index.tolist()
    for col in miss_cols:
        ind_name = f"{col}_missing"
        df[ind_name] = df[col].isnull().astype(int)
        feat_cols.append(ind_name)

    out = {}
    for split_name in ["train", "val", "test"]:
        mask = df["split"] == split_name
        out[split_name] = (
            df.loc[mask, feat_cols].values,
            df.loc[mask, "label"].values,
            df.loc[mask, "study_id"].values,
            df.loc[mask, "subject_id"].values,
        )

    print(f"  features={len(feat_cols)} ({len(miss_cols)} missingness indicators)")
    return out, feat_cols


def train_xgb(X_train, y_train, X_val, y_val):
    pos_rate = y_train.mean()
    scale = (1 - pos_rate) / max(pos_rate, 1e-6)
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def train_tabpfn(X_train, y_train):
    clf = TabPFNClassifier.create_default_for_version("v2")
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X, y, model_name):
    proba = clf.predict_proba(X)
    # handle both binary and multi-class output shapes
    if proba.ndim == 2:
        proba_pos = proba[:, 1]
    else:
        proba_pos = proba
    y_pred = (proba_pos >= 0.5).astype(int)
    auroc = roc_auc_score(y, proba_pos)
    auprc = average_precision_score(y, proba_pos)
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    f1_mac = f1_score(y, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y, y_pred, average="weighted", zero_division=0)
    return {
        "model": model_name, "auroc": auroc, "auprc": auprc,
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "f1_macro": f1_mac, "f1_weighted": f1_wt,
        "n": len(y), "pos_rate": y.mean(),
        "proba": proba_pos, "y_pred": y_pred,
    }


def main():
    features, _ = load_data()
    all_results = []

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"  {target}")
        print(f"{'='*60}")

        splits, feat_cols = prepare_task(features, target)
        X_train, y_train, _, _ = splits["train"]
        X_val, y_val, _, _ = splits["val"]
        X_test, y_test, study_ids_test, patient_ids_test = splits["test"]

        print(f"  train={len(y_train)} (pos={y_train.mean():.3f})  "
              f"val={len(y_val)} (pos={y_val.mean():.3f})  "
              f"test={len(y_test)} (pos={y_test.mean():.3f})")

        # ── XGBoost ──
        print("  Training XGBoost...")
        xgb_clf = train_xgb(X_train, y_train, X_val, y_val)
        for split_name, (X, y) in [("val", (X_val, y_val)), ("test", (X_test, y_test))]:
            res = evaluate(xgb_clf, X, y, "XGBoost")
            proba, y_pred = res.pop("proba"), res.pop("y_pred")
            res.update(target=target, split=split_name)
            all_results.append(res)
            print(f"    XGBoost {split_name}: AUROC={res['auroc']:.3f}  AUPRC={res['auprc']:.3f}  "
                  f"Acc={res['accuracy']:.3f}  BalAcc={res['balanced_accuracy']:.3f}  F1={res['f1_macro']:.3f}")
            if split_name == "test":
                pred_df = pd.DataFrame({
                    "study_id": study_ids_test,
                    "patient_id": patient_ids_test,
                    "true_label": y,
                    "predicted_class": y_pred,
                    "prediction_confidence": proba,
                })
                pred_dir = RESULTS_DIR / target / "XGBoost"
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred_df.to_csv(pred_dir / "predictions.csv", index=False)

        # ── TabPFN ──
        # TabPFN cannot handle NaN — impute with median from train
        print("  Training TabPFN...")
        medians = np.nanmedian(X_train, axis=0)
        X_train_imp = np.where(np.isnan(X_train), medians, X_train)
        X_val_imp = np.where(np.isnan(X_val), medians, X_val)
        X_test_imp = np.where(np.isnan(X_test), medians, X_test)

        tabpfn_clf = train_tabpfn(X_train_imp, y_train)
        for split_name, (X, y) in [("val", (X_val_imp, y_val)), ("test", (X_test_imp, y_test))]:
            res = evaluate(tabpfn_clf, X, y, "TabPFN")
            proba, y_pred = res.pop("proba"), res.pop("y_pred")
            res.update(target=target, split=split_name)
            all_results.append(res)
            print(f"    TabPFN  {split_name}: AUROC={res['auroc']:.3f}  AUPRC={res['auprc']:.3f}  "
                  f"Acc={res['accuracy']:.3f}  BalAcc={res['balanced_accuracy']:.3f}  F1={res['f1_macro']:.3f}")
            if split_name == "test":
                pred_df = pd.DataFrame({
                    "study_id": study_ids_test,
                    "patient_id": patient_ids_test,
                    "true_label": y,
                    "predicted_class": y_pred,
                    "prediction_confidence": proba,
                })
                pred_dir = RESULTS_DIR / target / "TabPFN"
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred_df.to_csv(pred_dir / "predictions.csv", index=False)

    # ── Save results ──
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "ehr_model_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'ehr_model_results.csv'}")

    # ── Print summary table ──
    print("\n" + "="*80)
    print("  SUMMARY (test set)")
    print("="*80)
    test_df = results_df[results_df["split"] == "test"].pivot_table(
        index="target", columns="model",
        values=["auroc", "auprc", "accuracy", "balanced_accuracy", "f1_macro"]
    )
    print(test_df.round(3).to_string())


if __name__ == "__main__":
    main()
