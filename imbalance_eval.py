"""
imbalance_eval.py
-----------------
Evaluate the best trained model (default: DistilRoBERTa + LoRA) on
synthetically imbalanced test splits at positive-to-negative ratios of
1:1, 1:5, 1:10, and 1:20.

Imbalance construction
----------------------
All negatives from the balanced test set (~23 K) are retained.
Positives are downsampled to achieve the desired ratio:

    n_positive = n_negative // ratio

This simulates the real-world "needle-in-a-haystack" deployment scenario
discussed in the paper's Ethics section: the original 1:1 dataset does
not reflect social media prevalence rates.

Supported model types (--model_type)
--------------------------------------
  transformer  – DistilRoBERTa + LoRA checkpoint (HuggingFace format)
  rnn          – Simple RNN            (Models/RNN_model.keras)
  lstm         – LSTM                  (Models/LSTM_model.keras)
  bilstm       – BiLSTM                (Models/BILSTM_model.keras)
  att_bilstm   – Attention BiLSTM      (Models/AttBiLSTM_model.keras)
  lr           – sklearn LR + TF-IDF   (Models/LR_model.pkl)
  lr_bow       – sklearn LR + BoW      (Models/BoW_LR_model.pkl)

Outputs
-------
  {results_dir}/imbalance_eval/
    metadata.json           – full config + results for all ratios
    metrics_vs_ratio.png    – line chart of precision, recall, AUROC,
                              AUPRC, FNR against imbalance ratio
    confusion_per_ratio.png – TP/TN/FP/FN bar chart per ratio
    results.json            – machine-readable results dict

Usage
-----
python imbalance_eval.py \\
    --dataset ./Dataset/Suicide_Detection.csv \\
    --model_dir ./Models/distilroberta_lora_final \\
    --model_type transformer \\
    --results_base ./results \\
    --seed 42 \\
    --batch_size 128
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import utils


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse and return CLI arguments for the imbalance robustness evaluation script."""
    p = argparse.ArgumentParser(
        description="Evaluate best model on imbalanced positive-to-negative test splits"
    )
    p.add_argument("--dataset",      default="./Dataset/Suicide_Detection.csv",
                   help="Path to Suicide_Detection.csv")
    p.add_argument("--model_dir",    default="./Models",
                   help="For transformer: path to the LoRA checkpoint dir. "
                        "For all other types: path to the Models/ directory.")
    p.add_argument("--model_type",   default="transformer",
                   choices=["transformer", "rnn", "lstm", "bilstm",
                            "att_bilstm", "lr", "lr_bow"],
                   help="Which model architecture to load")
    p.add_argument("--results_base", default="./results")
    p.add_argument("--timestamp",    default=None,
                   help="Shared timestamp from run_all.sh")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--max_len",      type=int,   default=None,
                   help="Token sequence length. Defaults to 64 for transformer, "
                        "128 for bilstm/lr. Override only if you retrained at a "
                        "different max_len.")
    p.add_argument("--ratios",       nargs="+",  type=int,
                   default=[1, 5, 10, 20],
                   help="Neg-to-pos multipliers, e.g. '1 5 10 20' → 1:1 … 1:20")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_transformer(model_dir: Path, device):
    """Load a HuggingFace sequence-classification model, with LoRA adapter if present.

    Checks for adapter_config.json to distinguish a PEFT/LoRA checkpoint from a
    fully fine-tuned model; both formats are supported transparently.

    Args:
        model_dir: Directory containing tokenizer files and either a full model
            checkpoint or a LoRA adapter + base model reference.
        device: torch.device to move the model onto after loading.

    Returns:
        Tuple (model, tokenizer) — model in eval mode on the target device.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if (model_dir / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel
        cfg   = PeftConfig.from_pretrained(str(model_dir))
        base  = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model_name_or_path, num_labels=2
        )
        model = PeftModel.from_pretrained(base, str(model_dir))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return model, tokenizer


def _get_keras_tok(model_dir: Path):
    """Load the shared Keras tokenizer from model_dir (JSON preferred, pkl fallback)."""
    tok_json = model_dir / "Tokenizer_model.json"
    tok_pkl  = model_dir / "Tokenizer_model.pkl"
    return utils.load_keras_tokenizer(tok_json if tok_json.exists() else tok_pkl)


def _load_keras_weights(model_path: Path, arch):
    """Restore weights via load_weights() to avoid the LSTM variable-count bug."""
    arch.load_weights(str(model_path))
    return arch


def _load_rnn(model_dir: Path, max_len: int = 128):
    """Reconstruct the Simple RNN architecture and load saved weights.

    Uses load_weights() rather than load_model() to avoid a TensorFlow bug
    where LSTM variable counts differ between save and load contexts.

    Returns:
        Tuple (model, keras_tokenizer).
    """
    import tensorflow as tf
    arch = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len,)),
        tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
        tf.keras.layers.SimpleRNN(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model = _load_keras_weights(model_dir / "RNN_model.keras", arch)
    return model, _get_keras_tok(model_dir)


def _load_lstm(model_dir: Path, max_len: int = 128):
    """Reconstruct the LSTM architecture and load saved weights.

    Returns:
        Tuple (model, keras_tokenizer).
    """
    import tensorflow as tf
    arch = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len,)),
        tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model = _load_keras_weights(model_dir / "LSTM_model.keras", arch)
    return model, _get_keras_tok(model_dir)


def _load_bilstm(model_dir: Path, max_len: int = 128):
    """Reconstruct the BiLSTM architecture and load saved weights.

    Returns:
        Tuple (model, keras_tokenizer).
    """
    import tensorflow as tf
    arch = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len,)),
        tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model = _load_keras_weights(model_dir / "BILSTM_model.keras", arch)
    return model, _get_keras_tok(model_dir)


def _load_att_bilstm(model_dir: Path, max_len: int = 128):
    """Reconstruct the Attention BiLSTM architecture and load saved weights.

    The functional API graph must be rebuilt identically to how it was saved,
    including the self-attention call pattern Attention()([H, H]).

    Returns:
        Tuple (model, keras_tokenizer).
    """
    import tensorflow as tf
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
    model   = _load_keras_weights(model_dir / "AttBiLSTM_model.keras", arch)
    return model, _get_keras_tok(model_dir)


def _load_lr(model_dir: Path):
    """Load the TF-IDF Logistic Regression checkpoint.

    Returns:
        Tuple (LogisticRegression, TfidfVectorizer).
    """
    with open(model_dir / "LR_model.pkl",        "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "Vectorizer_model.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def _load_lr_bow(model_dir: Path):
    """Load the Bag-of-Words Logistic Regression checkpoint.

    Returns:
        Tuple (LogisticRegression, CountVectorizer).
    """
    with open(model_dir / "BoW_LR_model.pkl",        "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "BoW_Vectorizer_model.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def load_model(args, device):
    """Dispatch to the correct loader; return (model, auxiliary_object)."""
    model_dir = Path(args.model_dir)
    if args.model_type == "transformer":
        ckpt_dir = (model_dir / "distilroberta_lora_final"
                    if not (model_dir / "adapter_config.json").exists()
                    and not (model_dir / "config.json").exists()
                    else model_dir)
        return _load_transformer(ckpt_dir, device)
    if args.model_type == "rnn":
        return _load_rnn(model_dir, args.max_len)
    if args.model_type == "lstm":
        return _load_lstm(model_dir, args.max_len)
    if args.model_type == "bilstm":
        return _load_bilstm(model_dir, args.max_len)
    if args.model_type == "att_bilstm":
        return _load_att_bilstm(model_dir, args.max_len)
    if args.model_type == "lr_bow":
        return _load_lr_bow(model_dir)
    return _load_lr(model_dir)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_probabilities(model, aux, texts, args, device) -> np.ndarray:
    """Return positive-class probabilities for *texts*."""
    if args.model_type == "transformer":
        import torch
        tokenizer = aux
        all_probs = []
        model.eval()
        text_list = list(texts)
        for i in range(0, len(text_list), args.batch_size):
            batch = text_list[i : i + args.batch_size]
            enc   = tokenizer(
                batch, truncation=True, max_length=args.max_len,
                padding=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            all_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(all_probs)

    if args.model_type in ("rnn", "lstm", "bilstm", "att_bilstm"):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        keras_tok = aux
        seqs   = keras_tok.texts_to_sequences(list(texts))
        padded = pad_sequences(seqs, maxlen=args.max_len, padding="post")
        return model.predict(padded, verbose=0).flatten()

    # lr / lr_bow
    vectorizer = aux
    return model.predict_proba(vectorizer.transform(list(texts)))[:, 1]


# ---------------------------------------------------------------------------
# Imbalanced split construction
# ---------------------------------------------------------------------------

def make_imbalanced_split(df_test, y_test, neg_ratio: int, seed: int):
    """
    Keep all negatives; subsample positives so pos:neg = 1:neg_ratio.

    Parameters
    ----------
    df_test   : pd.DataFrame with reset integer index
    y_test    : pd.Series of labels with reset integer index
    neg_ratio : desired number of negatives per positive (1 → 1:1)
    seed      : RNG seed for reproducible subsampling

    Returns
    -------
    (df_sub, y_sub_array) with rows shuffled
    """
    pos_idx = np.where(y_test.values == 1)[0]
    neg_idx = np.where(y_test.values == 0)[0]

    n_neg        = len(neg_idx)
    n_pos_target = max(1, n_neg // neg_ratio)

    if n_pos_target < len(pos_idx):
        rng     = np.random.default_rng(seed)
        pos_idx = rng.choice(pos_idx, size=n_pos_target, replace=False)

    chosen = np.concatenate([pos_idx, neg_idx])
    rng2   = np.random.default_rng(seed + 1)
    chosen = rng2.permutation(chosen)

    return df_test.iloc[chosen], y_test.iloc[chosen].values


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_metrics_vs_ratio(ratio_results: dict, outdir: Path) -> None:
    """Line chart: 5 key metrics against imbalance ratio."""
    ratios   = sorted(ratio_results.keys())
    x_labels = [f"1:{r}" for r in ratios]

    metric_cfg = [
        ("precision", "Precision",  "steelblue",    "o-"),
        ("recall",    "Recall",     "darkorange",   "s-"),
        ("auroc",     "AUROC",      "mediumseagreen","^-"),
        ("auprc",     "AUPRC",      "purple",        "D-"),
        ("fnr",       "FNR (↓)",    "tomato",        "x-"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label, color, style in metric_cfg:
        values = [ratio_results[r][key] for r in ratios]
        ax.plot(x_labels, values, style, label=label,
                color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Positive : Negative Ratio", fontsize=12)
    ax.set_ylabel("Metric Value",              fontsize=12)
    ax.set_title("Model Performance under Class Imbalance",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(outdir / "metrics_vs_ratio.png", dpi=150)
    plt.close()


def _plot_confusion_per_ratio(ratio_results: dict, outdir: Path) -> None:
    """TP / TN / FP / FN bar chart, one panel per ratio."""
    ratios = sorted(ratio_results.keys())
    n      = len(ratios)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    bar_cats   = ["TP", "TN", "FP", "FN"]
    bar_colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

    for ax, ratio in zip(axes, ratios):
        m      = ratio_results[ratio]
        values = [m["tp"], m["tn"], m["fp"], m["fn"]]
        bars   = ax.bar(bar_cats, values, color=bar_colors, edgecolor="white")
        ax.set_title(f"1:{ratio}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{v:,}", ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle("Confusion Matrix Breakdown by Imbalance Ratio",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_per_ratio.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Build imbalanced test splits at each requested ratio, run inference,
    compute metrics, produce plots, and serialise all results to the results dir."""
    args   = parse_args()
    utils.set_seeds(args.seed)
    device = utils.get_device()
    print(f"Device : {device}")

    if args.max_len is None:
        args.max_len = 128

    outdir = utils.make_results_dir(
        base=args.results_base,
        experiment_name="imbalance_eval",
        timestamp=args.timestamp,
    )
    print(f"Results → {outdir}")

    # Reproduce the canonical stratified 80/20 test split used across all scripts.
    print("Loading dataset …")
    df = utils.load_dataset(args.dataset)
    _, df_test, _, y_test = train_test_split(
        df, df["label"],
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label"],
    )
    df_test = df_test.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    print(f"Full test set : {len(y_test):,}  "
          f"(pos={int((y_test==1).sum()):,}, neg={int((y_test==0).sum()):,})")

    print(f"Loading {args.model_type} from {args.model_dir} …")
    model, aux = load_model(args, device)
    print("Model loaded.")

    text_col = "text_linear" if args.model_type in ("lr", "lr_bow") else "text_neural"

    ratio_results: dict = {}

    for neg_ratio in args.ratios:
        label = f"1:{neg_ratio}"
        print(f"\nBuilding imbalanced split  {label} …")
        df_sub, y_sub = make_imbalanced_split(df_test, y_test, neg_ratio, seed=args.seed)

        n_pos = int((y_sub == 1).sum())
        n_neg = int((y_sub == 0).sum())
        actual_ratio = n_neg // max(n_pos, 1)
        print(f"  n={len(y_sub):,}   pos={n_pos:,}   neg={n_neg:,}   "
              f"actual ratio ≈ 1:{actual_ratio}")

        y_prob   = get_probabilities(model, aux, df_sub[text_col], args, device)
        metrics  = utils.compute_full_metrics(y_sub, y_prob)

        ratio_results[neg_ratio] = metrics
        print(f"  precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}  "
              f"AUROC={metrics['auroc']:.4f}  "
              f"AUPRC={metrics['auprc']:.4f}  "
              f"FNR={metrics['fnr']:.4f}")

    _plot_metrics_vs_ratio(ratio_results, outdir)
    _plot_confusion_per_ratio(ratio_results, outdir)
    results_export = {str(k): v for k, v in ratio_results.items()}
    with open(outdir / "results.json", "w") as fh:
        json.dump(results_export, fh, indent=2)

    utils.save_metadata(outdir, {
        "model_type":  args.model_type,
        "model_dir":   args.model_dir,
        "ratios":      args.ratios,
        "n_test_full": int(len(y_test)),
        "results":     results_export,
    })

    cols = ["precision", "recall", "auroc", "auprc", "fnr", "ece"]
    print("\n" + "=" * 70)
    print(f"{'Ratio':<10}" + "".join(f"{c:>10}" for c in cols))
    print("-" * 70)
    for ratio in sorted(ratio_results.keys()):
        m   = ratio_results[ratio]
        row = f"{'1:'+str(ratio):<10}" + "".join(f"{m[c]:>10.4f}" for c in cols)
        print(row)
    print("=" * 70)
    print(f"\nAll results saved to {outdir}")


if __name__ == "__main__":
    main()
