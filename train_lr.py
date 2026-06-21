"""
train_lr.py
-----------
Train Logistic Regression baselines (BoW and TF-IDF) for suicide risk detection.

Saves two models:
  Models/LR_model.pkl          + Models/Vectorizer_model.pkl   (TF-IDF — canonical)
  Models/BoW_LR_model.pkl      + Models/BoW_Vectorizer_model.pkl

The TF-IDF pair is the checkpoint used by evaluate_checkpoints.py and imbalance_eval.py.

Usage
-----
  python train_lr.py                     # skip if models exist
  python train_lr.py --force_retrain     # always retrain
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import utils


def parse_args():
    """Parse CLI arguments; see module docstring for full option list."""
    p = argparse.ArgumentParser(description="Train Logistic Regression baselines")
    p.add_argument("--dataset",       default="./Dataset/Suicide_Detection.csv")
    p.add_argument("--models_dir",    default="./Models")
    p.add_argument("--results_base",  default="./results")
    p.add_argument("--timestamp",     default=None)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--max_features",  type=int, default=15000,
                   help="Vocabulary size for BoW / TF-IDF vectorizer")
    p.add_argument("--cv_folds",      type=int, default=5,
                   help="Number of StratifiedKFold CV folds on the training set. "
                        "0 = disabled.")
    p.add_argument("--force_retrain", action="store_true",
                   help="Retrain even if saved models already exist")
    return p.parse_args()


def _save_pkl(obj, path: Path) -> None:
    """Serialise *obj* to *path* with pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pkl(path: Path):
    """Deserialise and return the object stored at *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)


def train_variant(
    label: str,
    VecClass,
    model_path: Path,
    vec_path: Path,
    X_train,
    X_test,
    y_train,
    max_features: int,
    seed: int,
    force: bool,
):
    """Train one LR variant (or load from disk) and return predictions.

    Skips training when both checkpoint files already exist and *force* is False,
    allowing subsequent scripts to reuse a single trained checkpoint.

    Args:
        label: Human-readable name used in log messages (e.g. "LR (TF-IDF)").
        VecClass: Vectorizer class — CountVectorizer or TfidfVectorizer.
        model_path: Destination path for the pickled LogisticRegression.
        vec_path: Destination path for the pickled vectorizer.
        X_train: Training text series (preprocessed with clean_text_linear).
        X_test: Test text series (same preprocessing as X_train).
        y_train: Integer label array for the training set.
        max_features: Vocabulary cap passed to the vectorizer.
        seed: Random state for LogisticRegression reproducibility.
        force: When True, always retrain even if checkpoints exist.

    Returns:
        Tuple (model, vectorizer, y_prob) where y_prob is a 1-D float array
        of positive-class probabilities on X_test.
    """
    if not force and model_path.exists() and vec_path.exists():
        print(f"  [SKIP] {label} — loading existing checkpoint.")
        vec   = _load_pkl(vec_path)
        model = _load_pkl(model_path)
    else:
        print(f"  Training {label} …")
        vec   = VecClass(max_features=max_features)
        X_tr  = vec.fit_transform(X_train)
        model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
        model.fit(X_tr, y_train)
        _save_pkl(vec,   vec_path)
        _save_pkl(model, model_path)
        print(f"  Saved → {model_path.name}  +  {vec_path.name}")

    y_prob = model.predict_proba(vec.transform(X_test))[:, 1]
    return model, vec, y_prob


def cv_variant(
    label: str,
    VecClass,
    X_train,
    y_train: np.ndarray,
    max_features: int,
    seed: int,
    n_folds: int,
) -> list[dict]:
    """
    Stratified K-fold CV on the training set.

    The vectorizer is fit fresh on each fold's training portion so no
    vocabulary leakage crosses fold boundaries.  Returns one metrics dict
    per fold (same keys as utils.compute_full_metrics).
    """
    print(f"  {n_folds}-fold CV  {label} …")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train[tr_idx]
        y_val = y_train[val_idx]

        vec   = VecClass(max_features=max_features)
        X_tr_vec  = vec.fit_transform(X_tr)
        X_val_vec = vec.transform(X_val)

        model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
        model.fit(X_tr_vec, y_tr)

        y_prob = model.predict_proba(X_val_vec)[:, 1]
        m      = utils.compute_full_metrics(y_val, y_prob)
        fold_metrics.append(m)
        print(f"    fold {fold_idx + 1}/{n_folds}  "
              f"F1={m['f1']:.4f}  AUROC={m['auroc']:.4f}  "
              f"FNR={m['fnr']:.4f}  ECE={m['ece']:.4f}")

    return fold_metrics


def _summarise_cv(fold_metrics: list[dict]) -> dict:
    """Aggregate per-fold metrics into mean ± std for every numeric key."""
    keys = [k for k, v in fold_metrics[0].items() if isinstance(v, float)]
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics])
        summary[k]          = float(vals.mean())
        summary[k + "_std"] = float(vals.std())
    return summary


def main():
    """Train both LR variants, run optional CV, evaluate on the held-out test set,
    save model checkpoints and a JSON results summary to the results directory."""
    args = parse_args()
    utils.set_seeds(args.seed)

    outdir = utils.make_results_dir(
        base=args.results_base,
        experiment_name="train_lr",
        timestamp=args.timestamp,
    )
    print(f"Results → {outdir}")

    print("Loading dataset …")
    df = utils.load_dataset(args.dataset)
    df_train, df_test, _, y_test_s = utils.make_splits(df, seed=args.seed)
    y_train = df_train["label"].values
    y_test  = y_test_s.values
    print(f"Train: {len(df_train):,}   Test: {len(df_test):,}")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        (
            "LR (BoW)",
            CountVectorizer,
            models_dir / "BoW_LR_model.pkl",
            models_dir / "BoW_Vectorizer_model.pkl",
        ),
        (
            "LR (TF-IDF)",
            TfidfVectorizer,
            models_dir / "LR_model.pkl",
            models_dir / "Vectorizer_model.pkl",
        ),
    ]

    summary    = {}
    cv_summary = {}

    for label, VecClass, model_path, vec_path in variants:
        key = label.replace(" ", "_").replace("(", "").replace(")", "")

        # ── Cross-validation on training set ──────────────────────────────────
        if args.cv_folds > 0:
            fold_metrics = cv_variant(
                label, VecClass,
                X_train=df_train["text_linear"],
                y_train=y_train,
                max_features=args.max_features,
                seed=args.seed,
                n_folds=args.cv_folds,
            )
            cv_summary[key] = {
                "fold_metrics": fold_metrics,
                "mean_std":     _summarise_cv(fold_metrics),
            }
            agg = cv_summary[key]["mean_std"]
            print(f"  {label} CV mean:  "
                  f"F1={agg['f1']:.4f}±{agg['f1_std']:.4f}  "
                  f"AUROC={agg['auroc']:.4f}±{agg['auroc_std']:.4f}  "
                  f"FNR={agg['fnr']:.4f}±{agg['fnr_std']:.4f}  "
                  f"ECE={agg['ece']:.4f}±{agg['ece_std']:.4f}")

        # ── Standard held-out test evaluation ─────────────────────────────────
        _, _, y_prob = train_variant(
            label, VecClass, model_path, vec_path,
            X_train=df_train["text_linear"],
            X_test=df_test["text_linear"],
            y_train=y_train,
            max_features=args.max_features,
            seed=args.seed,
            force=args.force_retrain,
        )
        metrics = utils.compute_full_metrics(y_test, y_prob)
        summary[key] = metrics
        y_pred = (y_prob >= 0.5).astype(int)
        utils.save_confusion_matrix(y_test, y_pred, label, outdir,
                                    filename=f"confusion_matrix_{key}.png")
        print(f"  {label} test:   F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
              f"FNR={metrics['fnr']:.4f}  ECE={metrics['ece']:.4f}")

    utils.save_metadata(outdir, {
        "seed":       args.seed,
        "cv_folds":   args.cv_folds,
        "models":     summary,
        "cv_results": cv_summary,
    })
    with open(outdir / "summary.json", "w") as f:
        json.dump({"test": summary, "cv": cv_summary}, f, indent=2)

    cols = ["accuracy", "precision", "recall", "f1", "auroc", "auprc", "fnr", "ece"]

    # ── Held-out test results ──────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"  Held-out test results")
    print(f"{'Model':<20}" + "".join(f"{c:>9}" for c in cols))
    print("-" * 84)
    for name, m in summary.items():
        print(f"{name:<20}" + "".join(f"{m[c]:>9.4f}" for c in cols))
    print("=" * 84)

    # ── CV mean ± std results ──────────────────────────────────────────────────
    if cv_summary:
        cv_cols = ["f1", "auroc", "auprc", "fnr", "ece"]
        print(f"\n  {args.cv_folds}-fold CV results (training set only)")
        header = f"{'Model':<20}" + "".join(f"{'  ' + c:>18}" for c in cv_cols)
        print(header)
        print("-" * (20 + 18 * len(cv_cols)))
        for name, cv in cv_summary.items():
            agg = cv["mean_std"]
            row = f"{name:<20}" + "".join(
                f"  {agg[c]:.4f}±{agg[c + '_std']:.4f}" for c in cv_cols
            )
            print(row)
        print("=" * (20 + 18 * len(cv_cols)))

    print(f"\nAll results saved to {outdir}")


if __name__ == "__main__":
    main()
