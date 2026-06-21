"""
train_bilstm.py
---------------
Train and evaluate BiLSTM and Attention BiLSTM models for suicide risk detection.

Two architectures are trained in this script to isolate the effect of
directionality (BiLSTM) and token-level attention (Attention BiLSTM):

  BiLSTM — processes each post in both forward and backward directions
  simultaneously, so each token's representation incorporates both prior
  and downstream context. This resolves a key limitation of the unidirectional
  LSTM: words like "anymore" and "end" gain stronger signal when the model
  already knows what follows them.

  Attention BiLSTM — adds a dot-product self-attention layer over the full
  sequence of BiLSTM hidden states, then global-average-pools the attended
  representations. This raises precision (fewer false positives) but lowers
  recall (more false negatives) compared to the plain BiLSTM — a trade-off
  that is clinically undesirable in a screening context where sensitivity
  is paramount.

Architectures:
  BiLSTM:
    Embedding(20000, 128, learned) → Bidirectional(LSTM(64))
      → Dropout(0.3) → Dense(1, sigmoid)

  Attention BiLSTM:
    Embedding(20000, 128, learned) → Bidirectional(LSTM(64, return_sequences=True))
      → Attention([H, H]) → GlobalAveragePooling1D → Dropout(0.3) → Dense(1, sigmoid)

Saves (canonical paths used by evaluate_checkpoints.py / imbalance_eval.py):
  Models/BILSTM_model.keras
  Models/AttBiLSTM_model.keras
  Models/Tokenizer_model.json

Usage
-----
  python train_bilstm.py
  python train_bilstm.py --epochs 5 --force_retrain
"""

import argparse
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import utils


def parse_args():
    """Parse and return CLI arguments for the BiLSTM / Attention BiLSTM training script."""
    p = argparse.ArgumentParser(description="Train BiLSTM and Attention BiLSTM")
    p.add_argument("--dataset",       default="./Dataset/Suicide_Detection.csv")
    p.add_argument("--models_dir",    default="./Models")
    p.add_argument("--results_base",  default="./results")
    p.add_argument("--timestamp",     default=None)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--max_vocab",     type=int, default=20000)
    p.add_argument("--max_len",       type=int, default=128)
    p.add_argument("--embed_dim",     type=int, default=128)
    p.add_argument("--hidden_units",  type=int, default=64)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--epochs",        type=int, default=3)
    p.add_argument("--batch_size",    type=int, default=128)
    p.add_argument("--es_patience",   type=int, default=2,
                   help="Early-stopping patience (epochs). 0 = disabled.")
    p.add_argument("--force_retrain", action="store_true",
                   help="Retrain even if saved models already exist")
    return p.parse_args()


def get_or_fit_tokenizer(texts_train, max_vocab: int, tok_path: Path, force: bool):
    """Load the shared Keras tokenizer from disk, or fit a new one on training text.

    The tokenizer is shared across RNN / LSTM / BiLSTM so that all sequential
    models see the same integer vocabulary mapping. It is only refit when the
    checkpoint does not exist or --force_retrain is set.

    Args:
        texts_train: Iterable of preprocessed (Track 2) training strings.
        max_vocab: Maximum vocabulary size (OOV tokens mapped to index 0).
        tok_path: Expected path of the tokenizer file (JSON preferred, pkl fallback).
        force: When True, always refit even if a checkpoint exists.

    Returns:
        A fitted tf.keras.preprocessing.text.Tokenizer instance.
    """
    import tensorflow as tf
    json_path = tok_path.with_suffix(".json")
    if not force and (json_path.exists() or tok_path.exists()):
        print("  [SKIP] Tokenizer — loading existing.")
        return utils.load_keras_tokenizer(tok_path)
    print("  Fitting tokenizer …")
    tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_vocab)
    tok.fit_on_texts(texts_train)
    utils.save_keras_tokenizer(tok, tok_path)
    print(f"  Saved → {json_path.name}")
    return tok


def build_bilstm(max_vocab: int, max_len: int, embed_dim: int,
                 hidden_units: int, dropout: float):
    """Build and return the BiLSTM model (uncompiled).

    Bidirectional wraps a single LSTM so a forward pass and a backward pass
    run in parallel; their outputs are concatenated, doubling the effective
    hidden dimension (2 * hidden_units) fed into the Dense head. This lets
    the model incorporate downstream context — e.g., knowing a sentence ends
    with "anymore" raises the weight of ambiguous earlier tokens.

    Args:
        max_vocab: Vocabulary size (must match the fitted tokenizer).
        max_len: Fixed sequence length after post-padding.
        embed_dim: Embedding output dimension.
        hidden_units: Number of LSTM units per direction (effective = 2×).
        dropout: Dropout rate applied before the output Dense layer.

    Returns:
        A tf.keras.Sequential model with sigmoid output for binary classification.
    """
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len,)),
        tf.keras.layers.Embedding(input_dim=max_vocab, output_dim=embed_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def build_att_bilstm(max_vocab: int, max_len: int, embed_dim: int,
                     hidden_units: int, dropout: float):
    """Build and return the Attention BiLSTM model (uncompiled).

    Extends the BiLSTM by returning the full hidden-state sequence
    (return_sequences=True) and passing it through a dot-product Attention
    layer. The attention scores are computed as:

        score(h_t, h_s) = h_t · h_s^T

    where both the query and value are the BiLSTM output sequence (self-attention).
    GlobalAveragePooling1D then collapses the attended sequence to a fixed-length
    vector, replacing the final hidden state used in the plain BiLSTM.

    This raises precision (the attention suppresses off-topic tokens) but
    empirically lowers recall — the mechanism over-corrects for false-positive
    triggers, missing genuine crises that use ambiguous language.

    Args:
        max_vocab: Vocabulary size (must match the fitted tokenizer).
        max_len: Fixed sequence length after post-padding.
        embed_dim: Embedding output dimension.
        hidden_units: Number of LSTM units per direction.
        dropout: Dropout rate applied before the output Dense layer.

    Returns:
        A tf.keras.Model (functional API) with sigmoid output.
    """
    import tensorflow as tf
    inputs     = tf.keras.layers.Input(shape=(max_len,))
    emb        = tf.keras.layers.Embedding(input_dim=max_vocab, output_dim=embed_dim)(inputs)
    bilstm_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_units, return_sequences=True)
    )(emb)
    att_out    = tf.keras.layers.Attention()([bilstm_out, bilstm_out])
    pooled     = tf.keras.layers.GlobalAveragePooling1D()(att_out)
    dropped    = tf.keras.layers.Dropout(dropout)(pooled)
    outputs    = tf.keras.layers.Dense(1, activation="sigmoid")(dropped)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _plot_loss_curves(hist_bilstm, hist_att, outdir: Path) -> None:
    """Val-loss curves — only produced when both models were freshly trained."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(hist_bilstm.history["val_loss"],
            label="BiLSTM val loss", color="steelblue", linewidth=2, linestyle="--")
    ax.plot(hist_att.history["val_loss"],
            label="Attention BiLSTM val loss", color="darkorange", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
    ax.set_title("Validation Loss: BiLSTM vs Attention BiLSTM",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "bilstm_loss_curves.png", dpi=150)
    plt.close()


def _plot_metrics_comparison(metrics_bilstm: dict, metrics_att: dict,
                              outdir: Path) -> None:
    """Side-by-side bar chart of key metrics — always produced."""
    cols   = ["accuracy", "precision", "recall", "f1", "auroc", "auprc", "fnr", "ece"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC", "FNR", "ECE"]
    v1 = [metrics_bilstm[c] for c in cols]
    v2 = [metrics_att[c]    for c in cols]

    x  = np.arange(len(cols))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x - w / 2, v1, w, label="BiLSTM",           color="steelblue",  edgecolor="white")
    ax.bar(x + w / 2, v2, w, label="Attention BiLSTM", color="darkorange", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("BiLSTM vs Attention BiLSTM — Test Metrics",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    plt.tight_layout()
    plt.savefig(outdir / "bilstm_metrics_comparison.png", dpi=150)
    plt.close()


def main():
    """Train (or load) both BiLSTM variants, evaluate on the held-out test set,
    produce loss-curve and metrics-comparison plots, and save all artefacts to
    the timestamped results directory."""
    args = parse_args()
    utils.set_seeds(args.seed)

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # Avoid pre-allocating the full GPU memory; grow as needed.
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TF device : {'GPU x' + str(len(gpus)) if gpus else 'CPU'}")

    outdir = utils.make_results_dir(
        base=args.results_base,
        experiment_name="train_bilstm",
        timestamp=args.timestamp,
    )
    print(f"Results → {outdir}")

    print("Loading dataset …")
    df = utils.load_dataset(args.dataset)
    df_train, df_test, _, y_test_s = utils.make_splits(df, seed=args.seed)
    y_train = df_train["label"].values
    y_test  = y_test_s.values
    print(f"Train: {len(df_train):,}   Test: {len(df_test):,}")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    print(f"Class weights: {class_weight}")

    models_dir     = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    tok_path       = models_dir / "Tokenizer_model.json"
    bilstm_path    = models_dir / "BILSTM_model.keras"
    att_path       = models_dir / "AttBiLSTM_model.keras"

    tok = get_or_fit_tokenizer(
        df_train["text_neural"].tolist(), args.max_vocab, tok_path, args.force_retrain
    )

    X_train = tf.keras.preprocessing.sequence.pad_sequences(
        tok.texts_to_sequences(df_train["text_neural"].tolist()),
        maxlen=args.max_len, padding="post",
    )
    X_test = tf.keras.preprocessing.sequence.pad_sequences(
        tok.texts_to_sequences(df_test["text_neural"].tolist()),
        maxlen=args.max_len, padding="post",
    )

    # ── BiLSTM ─────────────────────────────────────────────────────────────────
    if not args.force_retrain and bilstm_path.exists():
        print("  [SKIP] BiLSTM — loading existing checkpoint.")
        bilstm_model = tf.keras.models.load_model(str(bilstm_path))
        hist_bilstm  = None
    else:
        print("  Training BiLSTM …")
        bilstm_model = build_bilstm(
            args.max_vocab, args.max_len, args.embed_dim,
            args.hidden_units, args.dropout,
        )
        bilstm_model.compile(optimizer="adam", loss="binary_crossentropy",
                             metrics=["accuracy"])
        callbacks = []
        if args.es_patience > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.es_patience,
                restore_best_weights=True,
            ))
        hist_bilstm = bilstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs, batch_size=args.batch_size,
            class_weight=class_weight, callbacks=callbacks, verbose=1,
        )
        bilstm_model.save(str(bilstm_path))
        print(f"  Saved → {bilstm_path.name}")

    y_prob_bilstm   = bilstm_model.predict(X_test, verbose=0).flatten()
    metrics_bilstm  = utils.compute_full_metrics(y_test, y_prob_bilstm)
    utils.save_confusion_matrix(y_test, (y_prob_bilstm >= 0.5).astype(int),
                                "BiLSTM", outdir, filename="confusion_matrix_bilstm.png")
    print(f"\n  BiLSTM: F1={metrics_bilstm['f1']:.4f}  "
          f"AUROC={metrics_bilstm['auroc']:.4f}  "
          f"FNR={metrics_bilstm['fnr']:.4f}  "
          f"ECE={metrics_bilstm['ece']:.4f}")

    # ── Attention BiLSTM ───────────────────────────────────────────────────────
    if not args.force_retrain and att_path.exists():
        print("  [SKIP] Attention BiLSTM — loading existing checkpoint.")
        att_model  = tf.keras.models.load_model(str(att_path))
        hist_att   = None
    else:
        print("  Training Attention BiLSTM …")
        att_model = build_att_bilstm(
            args.max_vocab, args.max_len, args.embed_dim,
            args.hidden_units, args.dropout,
        )
        att_model.compile(optimizer="adam", loss="binary_crossentropy",
                          metrics=["accuracy"])
        callbacks = []
        if args.es_patience > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.es_patience,
                restore_best_weights=True,
            ))
        hist_att = att_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs, batch_size=args.batch_size,
            class_weight=class_weight, callbacks=callbacks, verbose=1,
        )
        att_model.save(str(att_path))
        print(f"  Saved → {att_path.name}")

    y_prob_att  = att_model.predict(X_test, verbose=0).flatten()
    metrics_att = utils.compute_full_metrics(y_test, y_prob_att)
    utils.save_confusion_matrix(y_test, (y_prob_att >= 0.5).astype(int),
                                "Attention BiLSTM", outdir,
                                filename="confusion_matrix_att_bilstm.png")
    print(f"\n  Attention BiLSTM: F1={metrics_att['f1']:.4f}  "
          f"AUROC={metrics_att['auroc']:.4f}  "
          f"FNR={metrics_att['fnr']:.4f}  "
          f"ECE={metrics_att['ece']:.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    # Loss curves: only available when both models were freshly trained this run.
    if hist_bilstm is not None and hist_att is not None:
        _plot_loss_curves(hist_bilstm, hist_att, outdir)
        print(f"  Loss curves saved → {outdir / 'bilstm_loss_curves.png'}")

    # Metrics comparison: always produced regardless of checkpoint state.
    _plot_metrics_comparison(metrics_bilstm, metrics_att, outdir)
    print(f"  Metrics comparison saved → {outdir / 'bilstm_metrics_comparison.png'}")

    summary = {
        "BiLSTM":            metrics_bilstm,
        "Attention_BiLSTM":  metrics_att,
    }
    utils.save_metadata(outdir, {"config": vars(args), "models": summary})

    import json
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    cols = ["accuracy", "precision", "recall", "f1", "auroc", "auprc", "fnr", "ece"]
    print("\n" + "=" * 88)
    print(f"{'Model':<24}" + "".join(f"{c:>9}" for c in cols))
    print("-" * 88)
    for name, m in summary.items():
        print(f"{name:<24}" + "".join(f"{m[c]:>9.4f}" for c in cols))
    print("=" * 88)
    print(f"\nAll results saved to {outdir}")


if __name__ == "__main__":
    main()
