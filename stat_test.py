"""
stat_test.py
------------
Pairwise McNemar's test across all trained model checkpoints.

Compares every available pair from:
  LR (BoW) · LR (TF-IDF) · Simple RNN · LSTM · BiLSTM ·
  Attention BiLSTM · DistilRoBERTa+LoRA

For each pair the script builds the 2×2 contingency table of
correctly/incorrectly classified samples, then runs McNemar's test
(χ² approximation with Yates continuity correction, exact=False).

When ≥ 3 models are available, Bonferroni correction is applied:
    α_adj = 0.05 / n_pairs

Outputs
-------
  {results_dir}/stat_test/
    metadata.json   – config + full per-pair results with contingency tables
    results.json    – machine-readable pairwise statistics and p-values

Usage
-----
python stat_test.py \\
    --dataset     ./Dataset/Suicide_Detection.csv \\
    --models_dir  ./Models
"""

import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

import utils


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse and return CLI arguments for the McNemar significance test script."""
    p = argparse.ArgumentParser(description="Pairwise McNemar's significance test")
    p.add_argument("--dataset",                default="./Dataset/Suicide_Detection.csv")
    p.add_argument("--models_dir",             default="./Models")
    p.add_argument("--transformer_checkpoint", default=None,
                   help="Path to DistilRoBERTa+LoRA checkpoint. "
                        "Defaults to --models_dir/distilroberta_lora_final.")
    p.add_argument("--results_base",           default="./results")
    p.add_argument("--timestamp",              default=None)
    p.add_argument("--seed",                   type=int, default=42)
    p.add_argument("--batch_size",             type=int, default=128)
    p.add_argument("--max_len",                type=int, default=128,
                   help="Sequence length for all Keras and transformer models")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared Keras helpers
# ---------------------------------------------------------------------------

def _load_keras_weights(model_path: Path, arch):
    """Load weights into a pre-built Keras architecture and return it.

    Avoids tf.keras.models.load_model(), which fails with a variable-count
    mismatch when LSTM layers are loaded across different TF context states.
    """
    arch.load_weights(str(model_path))
    return arch


def _get_keras_tok(models_dir: Path):
    """Return the shared Keras tokenizer, or None if no checkpoint exists."""
    tok_json = models_dir / "Tokenizer_model.json"
    tok_pkl  = models_dir / "Tokenizer_model.pkl"
    if not tok_json.exists() and not tok_pkl.exists():
        return None
    return utils.load_keras_tokenizer(tok_json if tok_json.exists() else tok_pkl)


def _keras_predict(model, keras_tok, texts, max_len: int) -> np.ndarray:
    """Tokenise *texts*, pad to *max_len*, run inference, and return hard labels.

    Args:
        model: A loaded tf.keras.Model with sigmoid output.
        keras_tok: Fitted Keras Tokenizer (shared across all sequential models).
        texts: Iterable of preprocessed (Track 2) strings.
        max_len: Sequence length — must match the value used during training.

    Returns:
        1-D int array of binary predictions (0 or 1) at threshold 0.5.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs   = keras_tok.texts_to_sequences(list(texts))
    padded = pad_sequences(seqs, maxlen=max_len, padding="post")
    return (model.predict(padded, verbose=0).flatten() >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Per-model prediction helpers
# ---------------------------------------------------------------------------

def _predict_lr(models_dir: Path, texts, variant="tfidf") -> np.ndarray | None:
    """Load an LR checkpoint and return hard binary predictions, or None if missing.

    Args:
        models_dir: Directory containing model and vectorizer pkl files.
        texts: Preprocessed (Track 1 / linear) text series.
        variant: "tfidf" (default) or "bow" — selects which checkpoint to load.

    Returns:
        1-D int array of binary predictions, or None if the checkpoint is absent.
    """
    lr_path  = models_dir / ("BoW_LR_model.pkl"        if variant == "bow" else "LR_model.pkl")
    vec_path = models_dir / ("BoW_Vectorizer_model.pkl" if variant == "bow" else "Vectorizer_model.pkl")
    if not lr_path.exists() or not vec_path.exists():
        return None
    try:
        with open(lr_path,  "rb") as f: model      = pickle.load(f)
        with open(vec_path, "rb") as f: vectorizer = pickle.load(f)
        probs = model.predict_proba(vectorizer.transform(list(texts)))[:, 1]
        return (probs >= 0.5).astype(int)
    except Exception as exc:
        print(f"  [WARN] LR ({variant}) load failed: {exc}")
        return None


def _predict_rnn(models_dir: Path, texts, max_len: int) -> np.ndarray | None:
    """Load the Simple RNN checkpoint and return hard binary predictions, or None if missing."""
    import tensorflow as tf
    model_path = models_dir / "RNN_model.keras"
    if not model_path.exists():
        return None
    keras_tok = _get_keras_tok(models_dir)
    if keras_tok is None:
        return None
    try:
        arch = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,)),
            tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
            tf.keras.layers.SimpleRNN(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        return _keras_predict(_load_keras_weights(model_path, arch), keras_tok, texts, max_len)
    except Exception as exc:
        print(f"  [WARN] RNN load failed: {exc}")
        return None


def _predict_lstm(models_dir: Path, texts, max_len: int) -> np.ndarray | None:
    """Load the LSTM checkpoint and return hard binary predictions, or None if missing."""
    import tensorflow as tf
    model_path = models_dir / "LSTM_model.keras"
    if not model_path.exists():
        return None
    keras_tok = _get_keras_tok(models_dir)
    if keras_tok is None:
        return None
    try:
        arch = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,)),
            tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        return _keras_predict(_load_keras_weights(model_path, arch), keras_tok, texts, max_len)
    except Exception as exc:
        print(f"  [WARN] LSTM load failed: {exc}")
        return None


def _predict_bilstm(models_dir: Path, texts, max_len: int) -> np.ndarray | None:
    """Load the BiLSTM checkpoint and return hard binary predictions, or None if missing."""
    import tensorflow as tf
    model_path = models_dir / "BILSTM_model.keras"
    if not model_path.exists():
        return None
    keras_tok = _get_keras_tok(models_dir)
    if keras_tok is None:
        return None
    try:
        arch = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,)),
            tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        return _keras_predict(_load_keras_weights(model_path, arch), keras_tok, texts, max_len)
    except Exception as exc:
        print(f"  [WARN] BiLSTM load failed: {exc}")
        return None


def _predict_att_bilstm(models_dir: Path, texts, max_len: int) -> np.ndarray | None:
    """Load the Attention BiLSTM checkpoint and return hard binary predictions, or None if missing."""
    import tensorflow as tf
    model_path = models_dir / "AttBiLSTM_model.keras"
    if not model_path.exists():
        return None
    keras_tok = _get_keras_tok(models_dir)
    if keras_tok is None:
        return None
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
        return _keras_predict(_load_keras_weights(model_path, arch), keras_tok, texts, max_len)
    except Exception as exc:
        print(f"  [WARN] Attention BiLSTM load failed: {exc}")
        return None


def _predict_transformer(
    ckpt_dir: Path, texts, device, max_len: int, batch_size: int
) -> np.ndarray | None:
    """Load the DistilRoBERTa+LoRA checkpoint and return hard binary predictions.

    Supports both fully fine-tuned checkpoints and PEFT/LoRA adapters. Batches
    inference to avoid OOM on long text lists.

    Args:
        ckpt_dir: Path to the HuggingFace checkpoint directory.
        texts: Iterable of preprocessed (Track 2) strings.
        device: torch.device for inference.
        max_len: Maximum token sequence length (truncation applied).
        batch_size: Number of texts per inference batch.

    Returns:
        1-D int array of binary predictions, or None if the checkpoint is absent.
    """
    if not ckpt_dir.exists():
        return None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        if (ckpt_dir / "adapter_config.json").exists():
            from peft import PeftConfig, PeftModel
            cfg   = PeftConfig.from_pretrained(str(ckpt_dir))
            base  = AutoModelForSequenceClassification.from_pretrained(
                cfg.base_model_name_or_path, num_labels=2
            )
            model = PeftModel.from_pretrained(base, str(ckpt_dir))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
        model.to(device)
        model.eval()
        all_probs = []
        text_list = list(texts)
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            enc   = tokenizer(batch, truncation=True, max_length=max_len,
                              padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            all_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return (np.concatenate(all_probs) >= 0.5).astype(int)
    except Exception as exc:
        print(f"  [WARN] Transformer load failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def run_mcnemar(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    """Run McNemar's test on one pair of model predictions.

    Builds the 2×2 contingency table of discordant cases:
        a = both correct      b = A correct, B wrong
        c = A wrong, B correct  d = both wrong

    The test statistic uses Yates' continuity correction (exact=False) which
    is appropriate for large n — the balanced test set has ~46K samples.

    Args:
        y_true: Ground-truth integer labels (0/1).
        preds_a: Hard binary predictions from model A.
        preds_b: Hard binary predictions from model B.

    Returns:
        Dict with keys: contingency_table (a/b/c/d counts), statistic (χ²), pvalue.
    """
    a = int(np.sum((preds_a == y_true) & (preds_b == y_true)))
    b = int(np.sum((preds_a == y_true) & (preds_b != y_true)))
    c = int(np.sum((preds_a != y_true) & (preds_b == y_true)))
    d = int(np.sum((preds_a != y_true) & (preds_b != y_true)))
    result = mcnemar([[a, b], [c, d]], exact=False, correction=True)
    return {
        "contingency_table": {"a": a, "b": b, "c": c, "d": d},
        "statistic":         float(result.statistic),
        "pvalue":            float(result.pvalue),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load all available model checkpoints, run pairwise McNemar's tests with
    Bonferroni correction, print a formatted results table, and save JSON output."""
    args   = parse_args()
    utils.set_seeds(args.seed)
    device = utils.get_device()
    print(f"Device : {device}")

    outdir = utils.make_results_dir(
        base=args.results_base,
        experiment_name="stat_test",
        timestamp=args.timestamp,
    )
    print(f"Results → {outdir}")

    print("Loading dataset …")
    df = utils.load_dataset(args.dataset)
    _, df_test, _, y_test_s = utils.make_splits(df, seed=args.seed)
    df_test = df_test.reset_index(drop=True)
    y_test  = y_test_s.values
    print(f"Test set : {len(y_test):,}  "
          f"(pos={int(y_test.sum()):,}  neg={int((y_test == 0).sum()):,})")

    models_dir       = Path(args.models_dir)
    transformer_ckpt = Path(
        args.transformer_checkpoint or str(models_dir / "distilroberta_lora_final")
    )
    ml  = df_test["text_linear"]
    mnn = df_test["text_neural"]

    candidates = [
        ("LR_BoW",             lambda: _predict_lr(models_dir, ml, "bow")),
        ("LR_TF-IDF",          lambda: _predict_lr(models_dir, ml, "tfidf")),
        ("Simple_RNN",         lambda: _predict_rnn(models_dir, mnn, args.max_len)),
        ("LSTM",               lambda: _predict_lstm(models_dir, mnn, args.max_len)),
        ("BiLSTM",             lambda: _predict_bilstm(models_dir, mnn, args.max_len)),
        ("Attention_BiLSTM",   lambda: _predict_att_bilstm(models_dir, mnn, args.max_len)),
        ("DistilRoBERTa_LoRA", lambda: _predict_transformer(
            transformer_ckpt, mnn, device, args.max_len, args.batch_size)),
    ]

    model_preds: dict = {}
    for name, loader in candidates:
        print(f"\nLoading {name} …")
        preds = loader()
        if preds is not None:
            model_preds[name] = preds
            print(f"  accuracy = {(preds == y_test).mean():.4f}")
        else:
            print(f"  [SKIP] Checkpoint not found.")

    if len(model_preds) < 2:
        print("\n[ERROR] Need at least 2 loaded models to run McNemar's test. Exiting.")
        return

    pairs     = list(combinations(model_preds.keys(), 2))
    n_pairs   = len(pairs)
    alpha_adj = 0.05 / n_pairs

    print(f"\nRunning {n_pairs} pairwise McNemar's test(s)  "
          f"(Bonferroni α = 0.05 / {n_pairs} = {alpha_adj:.4f})\n")

    pair_results: dict = {}
    for name_a, name_b in pairs:
        key = f"{name_a}__vs__{name_b}"
        res = run_mcnemar(y_test, model_preds[name_a], model_preds[name_b])
        res["bonferroni_alpha"] = alpha_adj
        res["significant"]     = res["pvalue"] < alpha_adj
        pair_results[key]      = res

    col_w = max(len(k.replace("__vs__", " vs ")) for k in pair_results) + 2
    width = col_w + 56
    print("=" * width)
    print(f"{'Comparison':<{col_w}}{'Statistic':>12}{'p-value':>16}{'Significant':>14}")
    print("-" * width)
    for key, res in pair_results.items():
        label  = key.replace("__vs__", "  vs  ")
        sig    = "YES  ***" if res["significant"] else "no"
        pval   = res["pvalue"]
        pval_s = "< 1e-10" if pval < 1e-10 else f"{pval:.8f}"
        print(f"{label:<{col_w}}{res['statistic']:>12.4f}{pval_s:>16}{sig:>14}")
    print("=" * width)
    print(f"Bonferroni-corrected threshold : α = {alpha_adj:.4f}  (0.05 / {n_pairs} pairs)")

    print()
    for key, res in pair_results.items():
        name_a, name_b = key.split("__vs__")
        t = res["contingency_table"]
        print(f"  {name_a}  vs  {name_b}")
        print(f"    both correct={t['a']:,}  A✓B✗={t['b']:,}  "
              f"A✗B✓={t['c']:,}  both wrong={t['d']:,}")

    utils.save_metadata(outdir, {
        "models_compared":  list(model_preds.keys()),
        "n_test":           int(len(y_test)),
        "n_pairs":          n_pairs,
        "bonferroni_alpha": alpha_adj,
        "pair_results":     pair_results,
    })
    with open(outdir / "results.json", "w") as f:
        json.dump(pair_results, f, indent=2)

    print(f"\nAll results saved to {outdir}")


if __name__ == "__main__":
    main()
