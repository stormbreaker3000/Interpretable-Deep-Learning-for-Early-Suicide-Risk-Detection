"""
train_lstm.py
-------------
Train and evaluate a unidirectional LSTM for suicide risk detection.

The LSTM resolves the vanishing-gradient problem of the Simple RNN through
input, forget, and output gates that selectively retain long-range dependencies.
This allows contextual cues from the beginning of a post (e.g., "I'm fine but")
to influence the final prediction — a direct improvement over the RNN baseline.

Architecture:
  Embedding(vocab=20000, dim=128, learned) → LSTM(64) → Dropout(0.3)
    → Dense(1, sigmoid)

Hyperparameters (hidden units, dropout, embedding dim) are kept identical to
the RNN to isolate the effect of gating from capacity differences.

Saves:
  Models/LSTM_model.keras
  Models/Tokenizer_model.json  (shared tokenizer; fit only if absent)

Usage
-----
  python train_lstm.py
  python train_lstm.py --epochs 5 --force_retrain
"""

import argparse
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import utils


def parse_args():
    """Parse and return CLI arguments for the LSTM training script."""
    p = argparse.ArgumentParser(description="Train unidirectional LSTM baseline")
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
                   help="Retrain even if saved model already exists")
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


def build_lstm(max_vocab: int, max_len: int, embed_dim: int,
               hidden_units: int, dropout: float):
    """Build and return the LSTM model (uncompiled).

    The LSTM cell's forget gate defaults to bias=1 in Keras, which biases
    the network toward remembering context — important for long social media
    posts where distress signals may appear anywhere in the sequence.
    Dropout is applied after the LSTM output, not recurrently, to match
    the RNN baseline and keep the comparison architecturally consistent.

    Args:
        max_vocab: Vocabulary size (must match the fitted tokenizer).
        max_len: Fixed sequence length after post-padding.
        embed_dim: Embedding output dimension.
        hidden_units: Number of LSTM units.
        dropout: Dropout rate applied before the output Dense layer.

    Returns:
        A tf.keras.Sequential model with sigmoid output for binary classification.
    """
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len,)),
        tf.keras.layers.Embedding(input_dim=max_vocab, output_dim=embed_dim),
        tf.keras.layers.LSTM(hidden_units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def main():
    """Train (or load) the LSTM, evaluate on the held-out test set,
    save the checkpoint and a metadata JSON to the timestamped results directory."""
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
        experiment_name="train_lstm",
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

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "LSTM_model.keras"
    tok_path   = models_dir / "Tokenizer_model.json"

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

    if not args.force_retrain and model_path.exists():
        print("  [SKIP] LSTM model — loading existing checkpoint.")
        model = tf.keras.models.load_model(str(model_path))
    else:
        print("  Training LSTM …")
        model = build_lstm(
            args.max_vocab, args.max_len, args.embed_dim,
            args.hidden_units, args.dropout,
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        callbacks = []
        if args.es_patience > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.es_patience,
                restore_best_weights=True,
            ))
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        model.save(str(model_path))
        print(f"  Saved → {model_path.name}")

    y_prob  = model.predict(X_test, verbose=0).flatten()
    metrics = utils.compute_full_metrics(y_test, y_prob)
    utils.save_confusion_matrix(y_test, (y_prob >= 0.5).astype(int), "LSTM", outdir)
    print(f"\n  LSTM: F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
          f"FNR={metrics['fnr']:.4f}  ECE={metrics['ece']:.4f}")

    utils.save_metadata(outdir, {
        "model":   "LSTM",
        "config":  vars(args),
        "metrics": metrics,
    })
    print(f"All results saved to {outdir}")


if __name__ == "__main__":
    main()
