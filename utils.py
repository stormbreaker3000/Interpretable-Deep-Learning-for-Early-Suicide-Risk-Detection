"""
utils.py
--------
Shared utilities for the AdComSys 2026 camera-ready experiment suite.

Public API
----------
get_device()               – auto-select CUDA / MPS / CPU
set_seeds(seed)            – lock all RNG sources for reproducibility
make_results_dir()         – create ./results/{ts}/{name}/ and return its Path
make_splits(df, seed, …)  – canonical stratified 80/20 train/test split
compute_full_metrics()     – accuracy, precision, recall, specificity,
                             FNR, FPR, F1, AUROC, AUPRC, ECE
compute_ece()              – Expected Calibration Error (standalone)
save_metadata()            – write metadata.json into a results directory
load_dataset()             – load + dual-track-preprocess Suicide_Detection.csv
clean_text_neural()        – Track-2 light-normalisation preprocessing
clean_text_linear()        – Track-1 lemmatised / stop-word-stripped preprocessing
save_keras_tokenizer()     – persist a Keras Tokenizer as JSON (TF-version-stable)
load_keras_tokenizer()     – restore a Keras Tokenizer from JSON (pickle fallback)
"""

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    """
    Lock every RNG source for fully deterministic, paper-reproducible runs.

    Covers:
      - Python built-in hash randomisation (PYTHONHASHSEED)
      - Python random module
      - NumPy legacy and new-style RNGs
      - PyTorch CPU / CUDA / MPS
      - CuDNN algorithm selection  (deterministic=True, benchmark=False)
      - CuBLAS workspace           (CUBLAS_WORKSPACE_CONFIG)
      - PyTorch deterministic-algorithm enforcement (warn_only so unsupported
        ops emit a warning rather than crash)
      - TensorFlow / Keras         (TF_DETERMINISTIC_OPS)
    """
    # ── Environment variables (must be set before any CUDA / TF context) ──────
    os.environ["PYTHONHASHSEED"]          = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # required for deterministic CuBLAS
    os.environ["TF_DETERMINISTIC_OPS"]    = "1"         # Keras/TF GPU op determinism

    # ── Python + NumPy ────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)

    # ── PyTorch ───────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True   # reproducible CuDNN kernels
        torch.backends.cudnn.benchmark     = False  # no auto-tuning (non-deterministic)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # Warn (not error) if an op has no deterministic implementation — lets us
    # know without aborting training; address warnings if they appear.
    torch.use_deterministic_algorithms(True, warn_only=True)

    # ── TensorFlow / Keras ────────────────────────────────────────────────────
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Full op-level determinism (TF ≥ 2.9); must be called before any TF ops.
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass
        # Disable TF32 on Ampere+ GPUs — TF32 silently reduces float32 precision.
        try:
            tf.config.experimental.enable_tensor_float_32_execution(False)
        except AttributeError:
            pass
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Results directory management
# ---------------------------------------------------------------------------

def make_results_dir(
    base: str = "./results",
    experiment_name: str = "",
    timestamp: Optional[str] = None,
) -> Path:
    """
    Create and return ./results/{timestamp}/{experiment_name}/.

    If *timestamp* is omitted a new one is generated (YYYYMMDD_HHMMSS).
    Pass an explicit timestamp from run_all.sh so all three scripts
    share the same parent directory.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base) / ts / experiment_name if experiment_name else Path(base) / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Canonical train / test split
# ---------------------------------------------------------------------------

def make_splits(df, seed: int = 42, test_size: float = 0.2):
    """
    Stratified 80/20 train/test split — single source of truth for all scripts.

    Returns
    -------
    df_train, df_test, y_train_series, y_test_series
      (matches sklearn's train_test_split return signature)

    Usage
    -----
    df_train, df_test, _, y_test_s = utils.make_splits(df, seed=args.seed)
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(
        df, df["label"],
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )


# ---------------------------------------------------------------------------
# Text preprocessing — dual-track pipeline
#
# Different architectures require fundamentally different representations:
#   Track 1 (linear models): heavy normalisation — lemmatise + strip stop-words
#     so the TF-IDF vectorizer focuses on content-heavy root forms.
#   Track 2 (neural models): light normalisation — preserve word order, negations
#     ("not", "never") and structural markers that carry emotional meaning.
# Mixing tracks would give sequential models information they cannot exploit
# and deprive linear models of their signal (see paper §3.2).
# ---------------------------------------------------------------------------

_stop_words: Optional[set] = None
_lemmatizer = None


def _init_nlp() -> None:
    """Lazily initialise NLTK artefacts, downloading on first call.

    Populates the module-level _stop_words set and _lemmatizer singleton so
    repeated calls to clean_text_linear pay the import cost only once.
    """
    global _stop_words, _lemmatizer
    if _stop_words is not None:
        return
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    _stop_words = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()


def clean_text_neural(text: str) -> str:
    """
    Track 2 – light normalisation.
    Lowercase; remove URLs and punctuation characters. Word order,
    contractions, and negations (e.g. "n't", "can't") are retained,
    but sentence structure is simplified — not fully preserved.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', " ", text)
    return text.strip()


def clean_text_linear(text: str) -> str:
    """
    Track 1 – syntax destroyed.
    Lemmatise and drop stop-words so the TF-IDF vectorizer sees
    root-form content words only.
    """
    _init_nlp()
    words = text.split()
    words = [_lemmatizer.lemmatize(w) for w in words if w not in _stop_words]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str):
    """
    Load Suicide_Detection.csv, map class labels to binary integers, and
    add both preprocessed text columns.

    Returns a pandas DataFrame with columns:
        text, class, label (int), text_neural, text_linear
    """
    import pandas as pd

    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["class"] = df["class"].astype(str).str.lower().str.strip()
    df["label"] = df["class"].map({"suicide": 1, "non-suicide": 0})
    unexpected = df.loc[df["label"].isna(), "class"].unique().tolist()
    if unexpected:
        print(f"[WARN] load_dataset: unrecognised class values dropped: {unexpected}")
    df = df.dropna(subset=["label", "text"])
    df["label"]       = df["label"].astype(int)
    df["text_neural"] = df["text"].apply(clean_text_neural)
    df["text_linear"] = df["text_neural"].apply(clean_text_linear)
    return df


# ---------------------------------------------------------------------------
# Calibration metric
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) with equal-width probability bins.

        ECE = Σ_b (|B_b| / N) * |accuracy(B_b) − confidence(B_b)|

    Lower is better; a perfectly calibrated model has ECE = 0.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    n   = len(y_true)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        # Last bin uses <= so y_prob == 1.0 is not silently excluded.
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if not mask.any():
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Full metric suite
# ---------------------------------------------------------------------------

def compute_full_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute the complete metric suite for binary positive-class detection.

    Parameters
    ----------
    y_true    : ground-truth integer labels  (0 = non-suicide, 1 = suicide)
    y_prob    : predicted probability of the positive (suicide) class
    threshold : decision boundary (default 0.5)

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, specificity, fnr, fpr,
        f1, auroc, auprc, ece,
        tp, tn, fp, fn   (raw counts for confusion-matrix plots)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # labels=[0,1] guarantees a 2×2 matrix even when one class is absent.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall      = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # roc_auc_score / average_precision_score raise ValueError when only
    # one class is present in y_true (e.g. in heavily imbalanced sub-splits).
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        print("[WARN] compute_full_metrics: AUROC undefined (single class in y_true).")
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        print("[WARN] compute_full_metrics: AUPRC undefined (single class in y_true).")
        auprc = float("nan")

    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred,  zero_division=0)),
        "recall":      recall,
        "specificity": specificity,
        "fnr":         1.0 - recall,
        "fpr":         1.0 - specificity,
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc":       auroc,
        "auprc":       auprc,
        "ece":         compute_ece(y_true, y_prob),
        "tp":          int(tp),
        "tn":          int(tn),
        "fp":          int(fp),
        "fn":          int(fn),
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    outdir: Path,
    figsize: tuple = (4, 3.5),
    filename: str = "confusion_matrix.png",
) -> None:
    """Save a labelled seaborn heatmap confusion matrix to outdir/filename.

    Args:
        y_true: Ground-truth integer labels (0/1).
        y_pred: Hard predicted labels (0/1) at the chosen decision threshold.
        model_name: Title string embedded in the plot.
        outdir: Destination directory (must already exist).
        figsize: Matplotlib figure size in inches.
        filename: Output PNG filename within outdir.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as _cm

    cm = _cm(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        annot_kws={"size": 11},
        xticklabels=["Non-Suicidal", "Suicidal"],
        yticklabels=["Non-Suicidal", "Suicidal"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------------------------

def save_metadata(outdir: Path, metadata: dict) -> None:
    """Serialise *metadata* as JSON to outdir/metadata.json."""
    with open(outdir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Keras Tokenizer persistence (JSON — stable across TensorFlow versions)
# ---------------------------------------------------------------------------

def save_keras_tokenizer(tok, path) -> None:
    """
    Save a Keras Tokenizer as JSON.

    Prefer this over pickle: pickle embeds internal Keras class paths that
    break when the TensorFlow version changes; JSON is a stable text format.
    The file is always written with a .json extension regardless of *path*.
    """
    import tensorflow as tf  # noqa: F401 — needed for tok.to_json()
    out = Path(path).with_suffix(".json")
    with open(out, "w") as f:
        f.write(tok.to_json())


def load_keras_tokenizer(path):
    """
    Load a Keras Tokenizer.

    Tries *path* with a .json extension first (new format).  Falls back to
    loading *path* as-is via pickle for legacy .pkl files so that old
    checkpoints remain usable without retraining.

    Raises FileNotFoundError if neither file exists.
    """
    import tensorflow as tf
    path     = Path(path)
    json_path = path.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            return tf.keras.preprocessing.text.tokenizer_from_json(f.read())
    if path.exists():
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(
        f"Tokenizer not found at {json_path} (JSON) or {path} (legacy pickle)."
    )
