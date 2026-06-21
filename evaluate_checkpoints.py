"""
evaluate_checkpoints.py
-----------------------
Evaluate every trained model checkpoint on the held-out test set and
produce the full metric suite (accuracy, precision, recall, specificity,
FNR, FPR, F1, AUROC, AUPRC, ECE) together with ROC, PR, and calibration
curve plots saved per model.

Seven checkpoint types are discovered automatically:
  1. LR (BoW)              Models/BoW_LR_model.pkl      + BoW_Vectorizer_model.pkl
  2. LR (TF-IDF)           Models/LR_model.pkl           + Vectorizer_model.pkl
  3. Simple RNN            Models/RNN_model.keras         + Tokenizer_model.json
  4. LSTM                  Models/LSTM_model.keras        + Tokenizer_model.json
  5. BiLSTM                Models/BILSTM_model.keras      + Tokenizer_model.json
  6. Attention BiLSTM      Models/AttBiLSTM_model.keras   + Tokenizer_model.json
  7. DistilRoBERTa + LoRA  Models/distilroberta_lora_final/ (or --transformer_checkpoint)

Keras models are loaded via load_weights() into a pre-built architecture to
work around a known LSTM variable-loading bug in certain TF/Keras versions.

Usage
-----
python evaluate_checkpoints.py \\
    --dataset ./Dataset/Suicide_Detection.csv \\
    --models_dir ./Models \\
    --results_base ./results \\
    --seed 42 \\
    --batch_size 128

All outputs land in ./results/{timestamp}/evaluate_checkpoints/.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

import utils


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse and return CLI arguments for the checkpoint evaluation script."""
    p = argparse.ArgumentParser(
        description="Evaluate all model checkpoints on the test set"
    )
    p.add_argument("--dataset",      default="./Dataset/Suicide_Detection.csv")
    p.add_argument("--models_dir",   default="./Models")
    p.add_argument("--results_base", default="./results")
    p.add_argument("--timestamp",    default=None)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--batch_size",   type=int, default=128,
                   help="Batch size for transformer inference")
    p.add_argument("--max_len",      type=int, default=128,
                   help="Sequence length for all Keras models (must match training)")
    p.add_argument("--transformer_max_len", type=int, default=128,
                   help="Sequence length for transformer inference")
    p.add_argument("--transformer_checkpoint", default=None,
                   help="Override path to the DistilRoBERTa+LoRA checkpoint dir")
    p.add_argument("--calibration_bins", type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loaders
# ---------------------------------------------------------------------------

def load_lr(models_dir: Path, variant: str = "tfidf"):
    """Load a Logistic Regression model and its paired vectorizer from disk.

    Args:
        models_dir: Directory containing the serialised sklearn objects.
        variant: 'tfidf' loads LR_model.pkl + Vectorizer_model.pkl;
                 'bow' loads BoW_LR_model.pkl + BoW_Vectorizer_model.pkl.

    Returns:
        (lr_model, vectorizer) tuple, or (None, None) if either file is absent.
    """
    if variant == "bow":
        lr_path  = models_dir / "BoW_LR_model.pkl"
        vec_path = models_dir / "BoW_Vectorizer_model.pkl"
    else:
        lr_path  = models_dir / "LR_model.pkl"
        vec_path = models_dir / "Vectorizer_model.pkl"
    if not lr_path.exists() or not vec_path.exists():
        return None, None
    try:
        with open(lr_path,  "rb") as f:
            lr_model   = pickle.load(f)
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        return lr_model, vectorizer
    except Exception as exc:
        print(f"[WARN] Could not load LR ({variant}) checkpoint: {exc}")
        return None, None


def _load_keras_weights(model_path: Path, architecture):
    """
    Restore weights from a .keras checkpoint into a pre-built architecture.

    Using load_weights() instead of load_model() avoids a known LSTM
    variable-count bug in certain TF/Keras versions.
    """
    import tensorflow as tf
    architecture.load_weights(str(model_path))
    return architecture


def load_rnn(models_dir: Path, max_len: int = 128):
    """Rebuild the Simple RNN architecture and restore weights via load_weights().

    Returns:
        (model, keras_tokenizer) tuple, or (None, None) if checkpoints are absent.
    """
    import tensorflow as tf
    model_path = models_dir / "RNN_model.keras"
    tok_json   = models_dir / "Tokenizer_model.json"
    tok_pkl    = models_dir / "Tokenizer_model.pkl"
    if not model_path.exists() or (not tok_json.exists() and not tok_pkl.exists()):
        return None, None
    try:
        arch = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,)),
            tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
            tf.keras.layers.SimpleRNN(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model     = _load_keras_weights(model_path, arch)
        keras_tok = utils.load_keras_tokenizer(tok_json if tok_json.exists() else tok_pkl)
        return model, keras_tok
    except Exception as exc:
        print(f"[WARN] Could not load RNN checkpoint: {exc}")
        return None, None


def load_lstm(models_dir: Path, max_len: int = 128):
    """Rebuild the LSTM architecture and restore weights via load_weights().

    Returns:
        (model, keras_tokenizer) tuple, or (None, None) if checkpoints are absent.
    """
    import tensorflow as tf
    model_path = models_dir / "LSTM_model.keras"
    tok_json   = models_dir / "Tokenizer_model.json"
    tok_pkl    = models_dir / "Tokenizer_model.pkl"
    if not model_path.exists() or (not tok_json.exists() and not tok_pkl.exists()):
        return None, None
    try:
        arch = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,)),
            tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model     = _load_keras_weights(model_path, arch)
        keras_tok = utils.load_keras_tokenizer(tok_json if tok_json.exists() else tok_pkl)
        return model, keras_tok
    except Exception as exc:
        print(f"[WARN] Could not load LSTM checkpoint: {exc}")
        return None, None


def load_bilstm(models_dir: Path, max_len: int = 128):
    """Rebuild the BiLSTM architecture and restore weights via load_weights().

    Returns:
        (model, keras_tokenizer) tuple, or (None, None) if checkpoints are absent.
    """
    import tensorflow as tf
    model_path = models_dir / "BILSTM_model.keras"
    tok_json   = models_dir / "Tokenizer_model.json"
    tok_pkl    = models_dir / "Tokenizer_model.pkl"
    if not model_path.exists() or (not tok_json.exists() and not tok_pkl.exists()):
        return None, None
    try:
        arch = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,)),
            tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model     = _load_keras_weights(model_path, arch)
        keras_tok = utils.load_keras_tokenizer(tok_json if tok_json.exists() else tok_pkl)
        return model, keras_tok
    except Exception as exc:
        print(f"[WARN] Could not load BiLSTM checkpoint: {exc}")
        return None, None


def load_att_bilstm(models_dir: Path, max_len: int = 128):
    """Rebuild the Attention BiLSTM architecture and restore weights via load_weights().

    Returns:
        (model, keras_tokenizer) tuple, or (None, None) if checkpoints are absent.
    """
    import tensorflow as tf
    model_path = models_dir / "AttBiLSTM_model.keras"
    tok_json   = models_dir / "Tokenizer_model.json"
    tok_pkl    = models_dir / "Tokenizer_model.pkl"
    if not model_path.exists() or (not tok_json.exists() and not tok_pkl.exists()):
        return None, None
    try:
        inputs  = tf.keras.layers.Input(shape=(max_len,))
        emb     = tf.keras.layers.Embedding(input_dim=20000, output_dim=128)(inputs)
        bilstm  = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(emb)
        att     = tf.keras.layers.Attention()([bilstm, bilstm])
        pooled  = tf.keras.layers.GlobalAveragePooling1D()(att)
        dropped = tf.keras.layers.Dropout(0.3)(pooled)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(dropped)
        arch    = tf.keras.Model(inputs=inputs, outputs=outputs)
        model     = _load_keras_weights(model_path, arch)
        keras_tok = utils.load_keras_tokenizer(tok_json if tok_json.exists() else tok_pkl)
        return model, keras_tok
    except Exception as exc:
        print(f"[WARN] Could not load Attention BiLSTM checkpoint: {exc}")
        return None, None


def load_transformer(checkpoint_dir: Path, device):
    """Load DistilRoBERTa+LoRA for inference. Supports both PEFT adapter and merged formats.

    Detects whether checkpoint_dir contains adapter_config.json (PEFT LoRA adapter)
    or a fully merged model (config.json only), and loads accordingly.

    Args:
        checkpoint_dir: Path to the LoRA checkpoint or merged model directory.
        device: torch.device for model placement.

    Returns:
        (model, hf_tokenizer) tuple in eval mode, or (None, None) on failure.
    """
    if not checkpoint_dir.exists():
        return None, None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        if (checkpoint_dir / "adapter_config.json").exists():
            from peft import PeftConfig, PeftModel
            cfg   = PeftConfig.from_pretrained(str(checkpoint_dir))
            base  = AutoModelForSequenceClassification.from_pretrained(
                cfg.base_model_name_or_path, num_labels=2
            )
            model = PeftModel.from_pretrained(base, str(checkpoint_dir))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                str(checkpoint_dir)
            )
        model.to(device)
        model.eval()
        return model, hf_tokenizer
    except Exception as exc:
        print(f"[WARN] Could not load transformer from {checkpoint_dir}: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_lr(lr_model, vectorizer, texts) -> np.ndarray:
    """Return P(suicidal) scores by transforming texts with the vectorizer and calling predict_proba."""
    return lr_model.predict_proba(vectorizer.transform(texts))[:, 1]


def predict_keras(model, keras_tok, texts, max_len: int) -> np.ndarray:
    """Tokenise, pad, and run a forward pass through a Keras sequential model.

    Args:
        model: Loaded tf.keras.Model (RNN / LSTM / BiLSTM / AttBiLSTM).
        keras_tok: Shared Keras Tokenizer fitted on training text.
        texts: Iterable of preprocessed (Track 2) strings.
        max_len: Fixed pad length; must match the value used during training.

    Returns:
        1-D numpy array of shape (n_samples,) with P(suicidal).
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs   = keras_tok.texts_to_sequences(list(texts))
    padded = pad_sequences(seqs, maxlen=max_len, padding="post")
    return model.predict(padded, verbose=0).flatten()


def predict_transformer(
    model, hf_tokenizer, texts, device, batch_size: int, max_len: int
) -> np.ndarray:
    """Run batched inference through DistilRoBERTa+LoRA and return P(suicidal).

    Processes texts in chunks of batch_size to avoid OOM on the full test set.
    Softmax over two-class logits gives P(suicidal) at class index 1.

    Args:
        model: Eval-mode HuggingFace model (PeftModel or standard AutoModel).
        hf_tokenizer: Matching HuggingFace tokenizer.
        texts: List of preprocessed (Track 2) strings.
        device: torch.device for inference.
        batch_size: Number of samples per inference batch.
        max_len: Token sequence length for truncation/padding.

    Returns:
        1-D numpy array of shape (n_samples,) with P(suicidal).
    """
    import torch
    all_probs = []
    model.eval()
    text_list = list(texts)
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        enc   = hf_tokenizer(
            batch, truncation=True, max_length=max_len,
            padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        all_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Per-model plots
# ---------------------------------------------------------------------------

def _save_roc(y_true, y_prob, model_name: str, outdir: Path) -> None:
    """Plot and save a single-model ROC curve to outdir/roc_curve.png."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=150)
    plt.close()


def _save_pr(y_true, y_prob, model_name: str, outdir: Path) -> None:
    """Plot and save a single-model Precision-Recall curve to outdir/pr_curve.png."""
    from sklearn.metrics import average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap           = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(rec, prec, lw=2, color="darkorange", label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "pr_curve.png", dpi=150)
    plt.close()


def _save_calibration(y_true, y_prob, model_name: str, outdir: Path, n_bins: int) -> None:
    """Plot and save a reliability diagram to outdir/calibration_curve.png.

    Calibration is measured as ECE (Expected Calibration Error) over uniform bins.
    A perfectly calibrated model's curve follows the diagonal; lower ECE is better.
    """
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    ece = utils.compute_ece(np.array(y_true), np.array(y_prob))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(mean_pred, frac_pos, "s-", lw=2, color="mediumseagreen",
            label=f"Model  (ECE = {ece:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives",      fontsize=12)
    ax.set_title(f"Calibration Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "calibration_curve.png", dpi=150)
    plt.close()


def evaluate_one(name, y_true, y_prob, outdir, n_bins) -> dict:
    """Run the full evaluation suite for one model and return its metrics dict.

    Saves confusion matrix, ROC curve, PR curve, and calibration curve to outdir.

    Args:
        name: Display name used in plot titles (e.g. 'LR (BoW)', 'LSTM').
        y_true: Ground-truth binary labels (1 = suicidal).
        y_prob: Predicted positive-class probabilities, shape (n_samples,).
        outdir: Directory to write all output plots.
        n_bins: Number of uniform bins for the calibration curve.

    Returns:
        Dict with keys: accuracy, precision, recall, specificity, f1,
        auroc, auprc, ece, fnr, fpr, tp, tn, fp, fn.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = utils.compute_full_metrics(y_true, y_prob)
    utils.save_confusion_matrix(y_true, (y_prob >= 0.5).astype(int), name, outdir)
    _save_roc(y_true, y_prob, name, outdir)
    _save_pr(y_true, y_prob, name, outdir)
    _save_calibration(y_true, y_prob, name, outdir, n_bins=n_bins)
    return metrics


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def _save_comparison_roc(model_probs: dict, y_true: np.ndarray, outdir: Path) -> None:
    """Plot all-model ROC curves on a single axes and save to outdir/roc_comparison.png."""
    colors = ["steelblue", "darkorange", "mediumseagreen", "tomato",
              "mediumpurple", "saddlebrown", "crimson"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, y_prob), color in zip(model_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name}  (AUC={auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve Comparison — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "roc_comparison.png", dpi=150)
    plt.close()


def _save_comparison_pr(model_probs: dict, y_true: np.ndarray, outdir: Path) -> None:
    """Plot all-model PR curves on a single axes and save to outdir/pr_comparison.png."""
    from sklearn.metrics import average_precision_score
    colors = ["steelblue", "darkorange", "mediumseagreen", "tomato",
              "mediumpurple", "saddlebrown", "crimson"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, y_prob), color in zip(model_probs.items(), colors):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap           = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, lw=2, color=color, label=f"{name}  (AP={ap:.4f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve Comparison — All Models",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "pr_comparison.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Evaluate all seven model checkpoints on the test set and save results.

    Loads the dataset, constructs the 80/20 stratified test split (seed=42 by
    default), then evaluates each model type in sequence. Per-model outputs
    (confusion matrix, ROC, PR, calibration curve, metadata.json) land in
    outdir/{model_key}/; aggregated outputs (summary.json, roc_comparison.png,
    pr_comparison.png) land in outdir/.
    """
    args   = parse_args()
    utils.set_seeds(args.seed)
    device = utils.get_device()
    print(f"Device : {device}")

    outdir = utils.make_results_dir(
        base=args.results_base,
        experiment_name="evaluate_checkpoints",
        timestamp=args.timestamp,
    )
    print(f"Results → {outdir}")

    print("Loading dataset …")
    df = utils.load_dataset(args.dataset)
    _, X_test, _, y_test_series = train_test_split(
        df, df["label"],
        test_size=0.2, random_state=args.seed, stratify=df["label"],
    )
    y_test = y_test_series.values
    print(f"Test set : {len(y_test):,} samples  "
          f"(pos={y_test.sum():,}, neg={(y_test == 0).sum():,})")

    models_dir  = Path(args.models_dir)
    summary     = {}
    model_probs = {}

    def _record(key, display, y_prob):
        metrics = evaluate_one(display, y_test, y_prob, outdir / key,
                               n_bins=args.calibration_bins)
        utils.save_metadata(outdir / key, {
            "model": display, "n_test": int(len(y_test)), "metrics": metrics,
        })
        summary[key]         = metrics
        model_probs[display] = y_prob
        print(f"  F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
              f"ECE={metrics['ece']:.4f}  FNR={metrics['fnr']:.4f}")

    # ------------------------------------------------------------------
    # 1. LR (BoW)
    # ------------------------------------------------------------------
    lr, vec = load_lr(models_dir, variant="bow")
    if lr is not None:
        print("\nEvaluating  LR (BoW) …")
        _record("LR_BoW", "LR (BoW)",
                predict_lr(lr, vec, X_test["text_linear"]))
    else:
        print("\n[SKIP] LR (BoW) — checkpoint not found.")

    # ------------------------------------------------------------------
    # 2. LR (TF-IDF)
    # ------------------------------------------------------------------
    lr, vec = load_lr(models_dir, variant="tfidf")
    if lr is not None:
        print("\nEvaluating  LR (TF-IDF) …")
        _record("LR_TF-IDF", "LR (TF-IDF)",
                predict_lr(lr, vec, X_test["text_linear"]))
    else:
        print("\n[SKIP] LR (TF-IDF) — checkpoint not found.")

    # ------------------------------------------------------------------
    # 3. Simple RNN
    # ------------------------------------------------------------------
    rnn, tok = load_rnn(models_dir, args.max_len)
    if rnn is not None:
        print("\nEvaluating  Simple RNN …")
        _record("RNN", "Simple RNN",
                predict_keras(rnn, tok, X_test["text_neural"], args.max_len))
    else:
        print("\n[SKIP] Simple RNN — checkpoint not found.")

    # ------------------------------------------------------------------
    # 4. LSTM
    # ------------------------------------------------------------------
    lstm, tok = load_lstm(models_dir, args.max_len)
    if lstm is not None:
        print("\nEvaluating  LSTM …")
        _record("LSTM", "LSTM",
                predict_keras(lstm, tok, X_test["text_neural"], args.max_len))
    else:
        print("\n[SKIP] LSTM — checkpoint not found.")

    # ------------------------------------------------------------------
    # 5. BiLSTM
    # ------------------------------------------------------------------
    bilstm, tok = load_bilstm(models_dir, args.max_len)
    if bilstm is not None:
        print("\nEvaluating  BiLSTM …")
        _record("BiLSTM", "BiLSTM",
                predict_keras(bilstm, tok, X_test["text_neural"], args.max_len))
    else:
        print("\n[SKIP] BiLSTM — checkpoint not found.")

    # ------------------------------------------------------------------
    # 6. Attention BiLSTM
    # ------------------------------------------------------------------
    att, tok = load_att_bilstm(models_dir, args.max_len)
    if att is not None:
        print("\nEvaluating  Attention BiLSTM …")
        _record("AttBiLSTM", "Attention BiLSTM",
                predict_keras(att, tok, X_test["text_neural"], args.max_len))
    else:
        print("\n[SKIP] Attention BiLSTM — checkpoint not found.")

    # ------------------------------------------------------------------
    # 7. DistilRoBERTa + LoRA
    # ------------------------------------------------------------------
    transformer_ckpt = Path(
        args.transformer_checkpoint or str(models_dir / "distilroberta_lora_final")
    )
    tf_model, hf_tok = load_transformer(transformer_ckpt, device)
    if tf_model is not None:
        print("\nEvaluating  DistilRoBERTa + LoRA …")
        _record("DistilRoBERTa_LoRA", "DistilRoBERTa LoRA",
                predict_transformer(
                    tf_model, hf_tok, X_test["text_neural"].tolist(),
                    device=device, batch_size=args.batch_size,
                    max_len=args.transformer_max_len,
                ))
    else:
        print(f"\n[SKIP] Transformer checkpoint not found at {transformer_ckpt}.")

    if not summary:
        print("\nNo checkpoints found or loaded successfully. Exiting.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------
    if len(model_probs) >= 2:
        _save_comparison_roc(model_probs, y_test, outdir)
        _save_comparison_pr(model_probs,  y_test, outdir)

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ------------------------------------------------------------------
    # Console table
    # ------------------------------------------------------------------
    cols = ["accuracy", "precision", "recall", "f1", "auroc", "auprc", "ece", "fnr"]
    print("\n" + "=" * 96)
    print(f"{'Model':<24}" + "".join(f"{c:>9}" for c in cols))
    print("-" * 96)
    for mname, m in summary.items():
        print(f"{mname:<24}" + "".join(f"{m[c]:>9.4f}" for c in cols))
    print("=" * 96)
    print(f"\nAll results saved to {outdir}")


if __name__ == "__main__":
    main()
