# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Train sklearn probes on precomputed NPZ embeddings.

Supports classification (LogisticRegression) and regression (Ridge/Lasso),
hyperparameter grid search, k-fold cross-validation, and multi-model comparison.

Usage:
    # Train/val mode
    python -m evals.train_probe \
        --train embeddings/views/echojepa_g_train.npz \
        --val   embeddings/views/echojepa_g_val.npz \
        --output_dir results/probes/views/echojepa_g

    # K-fold mode
    python -m evals.train_probe \
        --data embeddings/lvef/echojepa_g_embeddings.npz \
        --task regression --cv 5 \
        --output_dir results/probes/lvef/echojepa_g

    # Labels-only mode (master embeddings + task labels file from remap_embeddings)
    python -m evals.train_probe \
        --data embeddings/nature_medicine/mimic/echojepa_g_mimic_embeddings.npz \
        --labels embeddings/nature_medicine/mimic/labels/mortality_1yr.npz \
        --cv 5 --output_dir results/probes/nature_medicine/mortality_1yr

    # Multi-model comparison
    python -m evals.train_probe \
        --train embeddings/views/echojepa_g_embeddings.npz \
               embeddings/views/echoprime_embeddings.npz \
        --val   embeddings/test/echojepa_g_embeddings.npz \
               embeddings/test/echoprime_embeddings.npz \
        --model_names echojepa_g echoprime \
        --output_dir results/probes/views/comparison
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_npz(path, labels_path=None):
    """Load embeddings, labels, and paths from an NPZ file.

    If labels_path is provided, loads embeddings from path but uses indices/labels
    from labels_path (created by remap_embeddings.py). This avoids duplicating
    embeddings across tasks.
    """
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]

    if labels_path is not None:
        label_data = np.load(labels_path)
        indices = label_data["indices"]
        labels = label_data["labels"]
        embeddings = embeddings[indices]
        paths = np.array([f"sample_{i}" for i in range(len(labels))])
        logger.info(
            f"Loaded {path} ({data['embeddings'].shape[0]} total) + {labels_path} "
            f"-> {embeddings.shape[0]} samples, {embeddings.shape[1]}-dim embeddings"
        )
    else:
        labels = data["labels"]
        paths = data["paths"] if "paths" in data else np.array([f"sample_{i}" for i in range(len(labels))])
        logger.info(f"Loaded {path}: {embeddings.shape[0]} samples, {embeddings.shape[1]}-dim embeddings")

    return embeddings, labels, paths


def detect_task(labels):
    """Auto-detect classification vs regression from label dtype."""
    if np.issubdtype(labels.dtype, np.integer):
        n_unique = len(np.unique(labels))
        logger.info(f"Detected classification task ({n_unique} classes)")
        return "classification"
    else:
        logger.info(f"Detected regression task (labels range: {labels.min():.3f} to {labels.max():.3f})")
        return "regression"


def get_scaler_stats(scaler_path=None, target_mean=None, target_std=None):
    """Get denormalization stats from a scaler file or explicit values."""
    if scaler_path is not None:
        import joblib

        scaler = joblib.load(scaler_path)
        mean = float(scaler.mean_[0])
        std = float(scaler.scale_[0])
        logger.info(f"Loaded scaler from {scaler_path}: mean={mean:.4f}, std={std:.4f}")
        return mean, std
    if target_mean is not None and target_std is not None:
        return target_mean, target_std
    return None, None


def scale_embeddings(X_train, X_val=None):
    """StandardScaler on embeddings: fit on train, transform both."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val) if X_val is not None else None
    return X_train_sc, X_val_sc, scaler


def train_classification_probe(X_train, y_train, X_val, y_val, C_grid, seed=42):
    """Train LogisticRegression with HP search over C values."""
    best_score = -1
    best_C = None
    best_model = None
    hp_results = []

    for C in C_grid:
        model = LogisticRegression(C=C, max_iter=1000, random_state=seed, solver="lbfgs")
        model.fit(X_train, y_train)
        score = balanced_accuracy_score(y_val, model.predict(X_val))
        hp_results.append({"C": C, "balanced_accuracy": float(score)})
        logger.info(f"  C={C:.4g} -> balanced_accuracy={score:.4f}")
        if score > best_score:
            best_score = score
            best_C = C
            best_model = model

    return best_model, best_C, hp_results


def train_regression_probe(X_train, y_train, X_val, y_val, alpha_grid, regression_model="ridge", seed=42):
    """Train Ridge/Lasso with HP search over alpha values."""
    best_score = -np.inf
    best_alpha = None
    best_model = None
    hp_results = []

    ModelClass = Ridge if regression_model == "ridge" else Lasso

    for alpha in alpha_grid:
        model = ModelClass(alpha=alpha, random_state=seed) if regression_model == "lasso" else ModelClass(alpha=alpha)
        model.fit(X_train, y_train)
        score = r2_score(y_val, model.predict(X_val))
        hp_results.append({"alpha": alpha, "r2": float(score)})
        logger.info(f"  alpha={alpha:.4g} -> R²={score:.4f}")
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_model = model

    return best_model, best_alpha, hp_results


def compute_classification_metrics(y_true, y_pred, y_proba):
    """Compute comprehensive classification metrics."""
    n_classes = len(np.unique(y_true))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_classes": n_classes,
        "n_samples": len(y_true),
    }

    # AUC-ROC
    try:
        if n_classes == 2:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["auc_roc"] = None

    return metrics


def compute_regression_metrics(y_true, y_pred, target_mean=None, target_std=None, labels_are_zscored=False):
    """Compute regression metrics, optionally denormalizing to real units."""
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_r": float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman_rho": float(stats.spearmanr(y_true, y_pred)[0]),
        "n_samples": len(y_true),
    }

    # Denormalize and compute real-unit metrics
    if labels_are_zscored and target_mean is not None and target_std is not None:
        y_true_real = y_true * target_std + target_mean
        y_pred_real = y_pred * target_std + target_mean
        metrics["mae_real"] = float(mean_absolute_error(y_true_real, y_pred_real))
        metrics["rmse_real"] = float(np.sqrt(mean_squared_error(y_true_real, y_pred_real)))
        metrics["target_mean"] = target_mean
        metrics["target_std"] = target_std

    return metrics


def kfold_probe(
    embeddings,
    labels,
    paths,
    task,
    n_splits=5,
    C_grid=None,
    alpha_grid=None,
    regression_model="ridge",
    target_mean=None,
    target_std=None,
    labels_are_zscored=False,
    seed=42,
):
    """Run k-fold cross-validation and aggregate metrics."""
    if task == "classification":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics = []
    all_predictions = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(embeddings, labels)):
        logger.info(f"Fold {fold_idx + 1}/{n_splits}")
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        p_val = paths[val_idx]

        X_train_sc, X_val_sc, _ = scale_embeddings(X_train, X_val)

        if task == "classification":
            model, best_hp, _ = train_classification_probe(X_train_sc, y_train, X_val_sc, y_val, C_grid, seed=seed)
            y_pred = model.predict(X_val_sc)
            y_proba = model.predict_proba(X_val_sc)
            metrics = compute_classification_metrics(y_val, y_pred, y_proba)
            metrics["best_C"] = best_hp
            pred_classes = y_pred
            pred_conf = y_proba.max(axis=1)
            for i in range(len(p_val)):
                all_predictions.append(
                    {
                        "video_path": str(p_val[i]),
                        "true_label": int(y_val[i]),
                        "predicted_class": int(pred_classes[i]),
                        "prediction_confidence": float(pred_conf[i]),
                        "fold": fold_idx,
                    }
                )
        else:
            model, best_hp, _ = train_regression_probe(
                X_train_sc, y_train, X_val_sc, y_val, alpha_grid, regression_model, seed=seed
            )
            y_pred = model.predict(X_val_sc)
            metrics = compute_regression_metrics(y_val, y_pred, target_mean, target_std, labels_are_zscored)
            metrics["best_alpha"] = best_hp
            # Denormalize for CSV output if applicable
            if labels_are_zscored and target_mean is not None and target_std is not None:
                labels_real = y_val * target_std + target_mean
                preds_real = y_pred * target_std + target_mean
            else:
                labels_real = y_val
                preds_real = y_pred
            for i in range(len(p_val)):
                all_predictions.append(
                    {
                        "video_path": str(p_val[i]),
                        "label_real": float(labels_real[i]),
                        "pred_real": float(preds_real[i]),
                        "abs_error": float(abs(labels_real[i] - preds_real[i])),
                        "fold": fold_idx,
                    }
                )

        fold_metrics.append(metrics)
        logger.info(f"  Fold {fold_idx + 1} done")

    # Aggregate across folds
    agg = {}
    numeric_keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float)) and k != "n_samples"]
    for key in numeric_keys:
        values = [m[key] for m in fold_metrics if m.get(key) is not None]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    agg["n_samples"] = int(fold_metrics[0]["n_samples"])
    agg["n_folds"] = n_splits
    agg["per_fold"] = fold_metrics

    return agg, pd.DataFrame(all_predictions)


def save_results(output_dir, model_name, metrics, predictions_df, hp_results=None, model=None):
    """Save metrics, predictions, and optionally the trained model."""
    out_path = Path(output_dir) / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Metrics
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {out_path / 'metrics.json'}")

    # Predictions CSV
    predictions_df.to_csv(out_path / "predictions.csv", index=False)
    logger.info(f"Saved predictions to {out_path / 'predictions.csv'}")

    # HP search results
    if hp_results is not None:
        with open(out_path / "hp_search.json", "w") as f:
            json.dump(hp_results, f, indent=2)

    # Model
    if model is not None:
        import joblib

        joblib.dump(model, out_path / "probe.joblib")
        logger.info(f"Saved model to {out_path / 'probe.joblib'}")


def print_comparison_table(all_metrics, task):
    """Print a side-by-side comparison table for multiple models."""
    if not all_metrics:
        return

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    model_names = list(all_metrics.keys())

    if task == "classification":
        headers = ["Model", "Accuracy", "Bal. Acc.", "F1 (macro)", "AUC-ROC"]
        keys = ["accuracy", "balanced_accuracy", "f1_macro", "auc_roc"]
    else:
        headers = ["Model", "R²", "MAE", "RMSE", "Pearson r", "Spearman ρ"]
        keys = ["r2", "mae", "rmse", "pearson_r", "spearman_rho"]

    # Check if these are k-fold results (keys have _mean suffix)
    sample_metrics = all_metrics[model_names[0]]
    is_kfold = any(k.endswith("_mean") for k in sample_metrics)

    # Header
    col_widths = [max(20, len(h) + 2) for h in headers]
    col_widths[0] = max(col_widths[0], max(len(n) for n in model_names) + 2)
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for name in model_names:
        m = all_metrics[name]
        row = [name]
        for key in keys:
            if is_kfold:
                mean_key = f"{key}_mean"
                std_key = f"{key}_std"
                mean_val = m.get(mean_key)
                std_val = m.get(std_key)
                if mean_val is not None and std_val is not None:
                    row.append(f"{mean_val:.4f}±{std_val:.4f}")
                elif mean_val is not None:
                    row.append(f"{mean_val:.4f}")
                else:
                    row.append("N/A")
            else:
                val = m.get(key)
                row.append(f"{val:.4f}" if val is not None else "N/A")
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train sklearn probes on precomputed NPZ embeddings")

    # Data inputs (mutually exclusive modes)
    parser.add_argument("--train", nargs="+", help="Training NPZ file(s)")
    parser.add_argument("--val", nargs="+", help="Validation NPZ file(s)")
    parser.add_argument("--data", nargs="+", help="Single NPZ file(s) for k-fold CV")
    parser.add_argument(
        "--labels",
        help="Labels-only NPZ (indices + labels) from remap_embeddings.py. "
        "Embeddings loaded from --data/--train, labels and subset indices from this file.",
    )

    # Task
    parser.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    parser.add_argument("--regression_model", choices=["ridge", "lasso"], default="ridge")

    # Hyperparameters
    parser.add_argument(
        "--C",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        help="C values for LogisticRegression grid search",
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        help="Alpha values for Ridge/Lasso grid search",
    )

    # Denormalization
    parser.add_argument("--target_mean", type=float, default=None, help="Target mean for denormalization")
    parser.add_argument("--target_std", type=float, default=None, help="Target std for denormalization")
    parser.add_argument("--scaler", type=str, default=None, help="Path to sklearn scaler (joblib) for denorm")
    parser.add_argument(
        "--labels_are_zscored", action="store_true", help="If set, labels are z-scored and will be denormalized"
    )

    # Cross-validation
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds (used with --data)")

    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_names", nargs="+", help="Names for each model (used with multiple --train/--data)")
    parser.add_argument("--save_model", action="store_true", help="Save trained sklearn model (joblib)")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if args.data is None and (args.train is None or args.val is None):
        parser.error("Must provide either --data (for k-fold) or both --train and --val")

    if args.train is not None and args.val is not None and len(args.train) != len(args.val):
        parser.error("--train and --val must have the same number of files")

    # Determine mode
    is_kfold = args.data is not None
    npz_list = args.data if is_kfold else args.train
    n_models = len(npz_list)

    # Model names
    if args.model_names:
        if len(args.model_names) != n_models:
            parser.error(f"--model_names has {len(args.model_names)} entries but {n_models} NPZ files provided")
        model_names = args.model_names
    else:
        model_names = [Path(p).stem for p in npz_list]

    # Scaler stats for denorm
    target_mean, target_std = get_scaler_stats(args.scaler, args.target_mean, args.target_std)

    all_metrics = {}

    for idx in range(n_models):
        name = model_names[idx]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Model: {name}")
        logger.info(f"{'=' * 60}")

        if is_kfold:
            # K-fold mode
            embeddings, labels, paths = load_npz(npz_list[idx], labels_path=args.labels)
            task = args.task if args.task != "auto" else detect_task(labels)

            metrics, predictions_df = kfold_probe(
                embeddings,
                labels,
                paths,
                task,
                n_splits=args.cv,
                C_grid=args.C,
                alpha_grid=args.alpha,
                regression_model=args.regression_model,
                target_mean=target_mean,
                target_std=target_std,
                labels_are_zscored=args.labels_are_zscored,
                seed=args.seed,
            )
            metrics["mode"] = "kfold"
            metrics["task"] = task
            metrics["source_file"] = str(npz_list[idx])
            save_results(args.output_dir, name, metrics, predictions_df)
            all_metrics[name] = metrics

        else:
            # Train/val mode
            X_train, y_train, p_train = load_npz(args.train[idx], labels_path=args.labels)
            X_val, y_val, p_val = load_npz(args.val[idx], labels_path=args.labels)
            task = args.task if args.task != "auto" else detect_task(y_train)

            X_train_sc, X_val_sc, emb_scaler = scale_embeddings(X_train, X_val)

            if task == "classification":
                model, best_hp, hp_results = train_classification_probe(
                    X_train_sc, y_train, X_val_sc, y_val, args.C, seed=args.seed
                )
                y_pred = model.predict(X_val_sc)
                y_proba = model.predict_proba(X_val_sc)
                metrics = compute_classification_metrics(y_val, y_pred, y_proba)
                metrics["best_C"] = best_hp
                pred_conf = y_proba.max(axis=1)
                predictions_df = pd.DataFrame(
                    {
                        "video_path": [str(p) for p in p_val],
                        "true_label": y_val.astype(int),
                        "predicted_class": y_pred.astype(int),
                        "prediction_confidence": pred_conf,
                    }
                )
            else:
                model, best_hp, hp_results = train_regression_probe(
                    X_train_sc, y_train, X_val_sc, y_val, args.alpha, args.regression_model, seed=args.seed
                )
                y_pred = model.predict(X_val_sc)
                metrics = compute_regression_metrics(y_val, y_pred, target_mean, target_std, args.labels_are_zscored)
                metrics["best_alpha"] = best_hp
                # Denormalize for CSV
                if args.labels_are_zscored and target_mean is not None and target_std is not None:
                    labels_real = y_val * target_std + target_mean
                    preds_real = y_pred * target_std + target_mean
                else:
                    labels_real = y_val
                    preds_real = y_pred
                predictions_df = pd.DataFrame(
                    {
                        "video_path": [str(p) for p in p_val],
                        "label_real": labels_real,
                        "pred_real": preds_real,
                        "abs_error": np.abs(labels_real - preds_real),
                    }
                )

            metrics["mode"] = "train_val"
            metrics["task"] = task
            metrics["train_file"] = str(args.train[idx])
            metrics["val_file"] = str(args.val[idx])
            save_results(
                args.output_dir,
                name,
                metrics,
                predictions_df,
                hp_results=hp_results,
                model=model if args.save_model else None,
            )
            all_metrics[name] = metrics

    # Print comparison table
    if n_models > 1:
        print_comparison_table(all_metrics, task)
        comparison_path = Path(args.output_dir) / "comparison.json"
        # Build comparison-safe metrics (strip non-serializable items)
        comparison = {}
        for name, m in all_metrics.items():
            comparison[name] = {k: v for k, v in m.items() if k != "per_fold" and k != "confusion_matrix"}
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison to {comparison_path}")
    else:
        # Print single model summary
        name = model_names[0]
        m = all_metrics[name]
        print(f"\n--- {name} ---")
        if task == "classification":
            if is_kfold:
                print(f"Balanced Accuracy: {m['balanced_accuracy_mean']:.4f} ± {m['balanced_accuracy_std']:.4f}")
                print(f"F1 (macro):        {m['f1_macro_mean']:.4f} ± {m['f1_macro_std']:.4f}")
            else:
                print(f"Balanced Accuracy: {m['balanced_accuracy']:.4f}")
                print(f"F1 (macro):        {m['f1_macro']:.4f}")
                if m.get("auc_roc") is not None:
                    print(f"AUC-ROC:           {m['auc_roc']:.4f}")
        else:
            if is_kfold:
                print(f"R²:         {m['r2_mean']:.4f} ± {m['r2_std']:.4f}")
                print(f"MAE:        {m['mae_mean']:.4f} ± {m['mae_std']:.4f}")
                print(f"Pearson r:  {m['pearson_r_mean']:.4f} ± {m['pearson_r_std']:.4f}")
            else:
                print(f"R²:         {m['r2']:.4f}")
                print(f"MAE:        {m['mae']:.4f}")
                print(f"Pearson r:  {m['pearson_r']:.4f}")
        print()


if __name__ == "__main__":
    main()
