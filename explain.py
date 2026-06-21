"""
explain.py
----------
Post-hoc explainability for trained models using SHAP and LIME.

SHAP strategy per model type
-----------------------------
  transformer / rnn / lstm / bilstm / att_bilstm
                   – shap.Explainer (PartitionExplainer via text masker)
  lr / lr_bow      – shap.LinearExplainer (exact, fast; uses feature names)

LIME is supported for all model types.

Single-model outputs  (explain/)
---------------------------------
  shap_global_importance.png   – top-N tokens/features by mean |SHAP|
  lime_top_features.png        – top-N features aggregated across LIME samples
  lime/sample_{i:02d}.html     – individual LIME explanation per sample
  metadata.json  /  results.json

Compare-all outputs  (explain/)
--------------------------------
  {model}/shap_global_importance.png   – per-model individual SHAP plots
  {model}/lime_top_features.png        – per-model individual LIME plots
  {model}/lime/sample_*.html
  shap_comparison.png                  – side-by-side SHAP panel (all models)
  lime_comparison.png                  – side-by-side LIME panel (all models)
  results.json

Dependencies
------------
  pip install shap lime

Note: SHAP PartitionExplainer performs O(n_tokens) model calls per sample.
With n_shap_samples=100 on CPU this takes ~5-20 min per model.
Reduce --n_shap_samples or run on GPU to speed up.

Usage
-----
# Single model
python explain.py --model_type transformer
python explain.py --model_type bilstm
python explain.py --model_type att_bilstm
python explain.py --model_type lr

# Joint comparison across all available models
python explain.py --compare_all
python explain.py --compare_all --n_shap_samples 50 --n_lime_samples 5
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse and return CLI arguments for the SHAP + LIME explainability script."""
    p = argparse.ArgumentParser(description="SHAP + LIME explainability")
    p.add_argument("--dataset",            default="./Dataset/Suicide_Detection.csv")
    p.add_argument("--results_base",       default="./results")
    p.add_argument("--timestamp",          default=None)
    p.add_argument("--seed",               type=int, default=42)
    p.add_argument("--batch_size",         type=int, default=32,
                   help="Inference batch size (lower reduces OOM risk during SHAP)")
    p.add_argument("--max_len",            type=int, default=None,
                   help="Sequence length override. Defaults: 64=transformer, 128=bilstm/lr")
    p.add_argument("--n_shap_samples",     type=int, default=100)
    p.add_argument("--n_lime_samples",     type=int, default=10)
    p.add_argument("--top_features",       type=int, default=20,
                   help="Top-N features in individual plots")
    p.add_argument("--n_compare_features", type=int, default=15,
                   help="Top-N features per panel in comparison plots")

    # ── Single-model mode ─────────────────────────────────────────────────────
    p.add_argument("--model_type",  default="transformer",
                   choices=["transformer", "rnn", "lstm", "bilstm",
                            "att_bilstm", "lr", "lr_bow"])
    p.add_argument("--model_dir",   default="./Models",
                   help="For transformer: LoRA checkpoint dir. "
                        "For all other types: Models/ directory.")

    # ── Compare-all mode ──────────────────────────────────────────────────────
    p.add_argument("--compare_all", action="store_true",
                   help="Run SHAP + LIME for all available models and produce joint plots")
    p.add_argument("--models_dir",  default="./Models",
                   help="Models directory used in --compare_all mode")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _get_keras_tok(model_dir: Path):
    """Load the shared Keras tokenizer from disk (JSON preferred, pkl fallback)."""
    tok_json = model_dir / "Tokenizer_model.json"
    tok_pkl  = model_dir / "Tokenizer_model.pkl"
    return utils.load_keras_tokenizer(tok_json if tok_json.exists() else tok_pkl)


def _load_keras_weights(model_path: Path, arch):
    """load_weights() avoids the LSTM variable-count bug in load_model()."""
    arch.load_weights(str(model_path))
    return arch


def _load_transformer(model_dir: Path, device):
    """Load DistilRoBERTa+LoRA and merge adapter weights for SHAP/LIME compatibility.

    Calls merge_and_unload() to fuse the LoRA rank-decomposition matrices back into
    the base weights, producing a standard HuggingFace model. This is required for
    SHAP's PartitionExplainer and LIME, which call the model as an opaque function
    and cannot navigate PEFT's adapter indirection.

    Args:
        model_dir: LoRA checkpoint directory (contains adapter_config.json)
                   or a fully merged checkpoint directory.
        device: torch.device for model placement.

    Returns:
        (model, tokenizer) tuple with model in eval mode.
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
        model = model.merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return model, tokenizer


def _load_rnn(model_dir: Path, max_len: int = 128):
    """Rebuild the Simple RNN architecture and restore weights from disk.

    Returns:
        (model, keras_tokenizer) tuple.
    """
    import tensorflow as tf
    arch = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len,)),
        tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
        tf.keras.layers.SimpleRNN(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return _load_keras_weights(model_dir / "RNN_model.keras", arch), _get_keras_tok(model_dir)


def _load_lstm(model_dir: Path, max_len: int = 128):
    """Rebuild the LSTM architecture and restore weights from disk.

    Returns:
        (model, keras_tokenizer) tuple.
    """
    import tensorflow as tf
    arch = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len,)),
        tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return _load_keras_weights(model_dir / "LSTM_model.keras", arch), _get_keras_tok(model_dir)


def _load_bilstm(model_dir: Path, max_len: int = 128):
    """Rebuild the BiLSTM architecture and restore weights from disk.

    Returns:
        (model, keras_tokenizer) tuple.
    """
    import tensorflow as tf
    arch = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len,)),
        tf.keras.layers.Embedding(input_dim=20000, output_dim=128),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return _load_keras_weights(model_dir / "BILSTM_model.keras", arch), _get_keras_tok(model_dir)


def _load_att_bilstm(model_dir: Path, max_len: int = 128):
    """Rebuild the Attention BiLSTM architecture and restore weights from disk.

    Returns:
        (model, keras_tokenizer) tuple.
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
    return _load_keras_weights(model_dir / "AttBiLSTM_model.keras", arch), _get_keras_tok(model_dir)


def _load_lr(model_dir: Path, variant="tfidf"):
    """Load an LR model and its paired vectorizer from disk.

    Args:
        model_dir: Directory containing the serialised sklearn objects.
        variant: 'tfidf' loads LR_model.pkl + Vectorizer_model.pkl;
                 'bow' loads BoW_LR_model.pkl + BoW_Vectorizer_model.pkl.

    Returns:
        (lr_model, vectorizer) tuple.
    """
    lr_key  = "BoW_LR_model.pkl"        if variant == "bow" else "LR_model.pkl"
    vec_key = "BoW_Vectorizer_model.pkl" if variant == "bow" else "Vectorizer_model.pkl"
    with open(model_dir / lr_key,  "rb") as f: model      = pickle.load(f)
    with open(model_dir / vec_key, "rb") as f: vectorizer = pickle.load(f)
    return model, vectorizer


# ---------------------------------------------------------------------------
# Predictor factories  →  [n, 2]: [P(non-suicide), P(suicide)]
# ---------------------------------------------------------------------------

def _make_transformer_predictor(model, tokenizer, device, max_len: int, batch_size: int):
    """Return a batched inference function for the DistilRoBERTa+LoRA model.

    The returned callable accepts a list of strings and returns an (n, 2) array of
    [P(Non-Suicidal), P(Suicidal)], which is the interface expected by both
    SHAP's PartitionExplainer and LIME's LimeTextExplainer.

    Args:
        model: Eval-mode HuggingFace model (merged LoRA or standard).
        tokenizer: Matching HuggingFace tokenizer.
        device: torch.device for inference.
        max_len: Truncation/padding length.
        batch_size: Inference batch size to manage GPU/CPU memory.

    Returns:
        Callable: List[str] → np.ndarray of shape (n, 2).
    """
    import torch
    def predict(texts):
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            enc   = tokenizer(batch, truncation=True, max_length=max_len,
                              padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs)
    return predict


def _make_keras_predictor(model, keras_tok, max_len: int):
    """Return a vectorised inference function for any Keras sequential model.

    Tokenises and pads texts using the shared Keras tokenizer, then runs a forward
    pass. Returns (n, 2) probabilities [P(Non-Suicidal), P(Suicidal)] to match
    the transformer predictor interface expected by SHAP and LIME.

    Args:
        model: Loaded tf.keras.Model (RNN / LSTM / BiLSTM / AttBiLSTM).
        keras_tok: Fitted Keras Tokenizer shared across all sequential models.
        max_len: Fixed pad/truncation length; must match training.

    Returns:
        Callable: List[str] → np.ndarray of shape (n, 2).
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    def predict(texts):
        seqs   = keras_tok.texts_to_sequences(list(texts))
        padded = pad_sequences(seqs, maxlen=max_len, padding="post")
        pos    = model.predict(padded, verbose=0).flatten()
        return np.column_stack([1 - pos, pos])
    return predict


def _make_lr_predictor(model, vectorizer):
    """Return an inference function for an LR + vectorizer pair.

    Passes texts through the vectorizer then calls predict_proba, returning
    (n, 2) class probabilities directly without any reshaping.

    Args:
        model: Fitted sklearn LogisticRegression.
        vectorizer: Fitted sklearn TfidfVectorizer or CountVectorizer.

    Returns:
        Callable: List[str] → np.ndarray of shape (n, 2).
    """
    def predict(texts):
        return model.predict_proba(vectorizer.transform(list(texts)))
    return predict


# ---------------------------------------------------------------------------
# SHAP — data collection (returns aggregated top features)
# ---------------------------------------------------------------------------

def _collect_shap_partition(predictor, texts: list, top_n: int) -> tuple[list, list]:
    """PartitionExplainer → (feat_list, imp_list) for top_n tokens."""
    import shap
    print(f"    Computing SHAP values for {len(texts)} samples …")
    masker    = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(predictor, masker,
                               output_names=["Non-Suicidal", "Suicidal"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer(texts, fixed_context=1)

    feature_names, feature_imps = [], []
    for sv in shap_values:
        vals = sv.values
        imps = np.abs(vals[:, 1]) if vals.ndim == 2 else np.abs(vals)
        feature_names.extend(sv.data)
        feature_imps.extend(imps.tolist())

    df  = pd.DataFrame({"token": feature_names, "importance": feature_imps})
    top = df.groupby("token")["importance"].mean().sort_values(ascending=False).head(top_n)
    return top.index.tolist(), top.values.tolist()


def _collect_shap_lr(model, vectorizer, texts: list, top_n: int) -> tuple[list, list]:
    """LinearExplainer → (feat_list, imp_list) for top_n TF-IDF terms."""
    import shap
    print(f"    Computing SHAP (LinearExplainer) for {len(texts)} samples …")
    X_vec      = vectorizer.transform(texts)
    explainer  = shap.LinearExplainer(model, X_vec, feature_perturbation="interventional")
    sv         = explainer.shap_values(X_vec)
    feat_names = vectorizer.get_feature_names_out()
    mean_abs   = np.abs(sv).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[-top_n:][::-1]
    return feat_names[top_idx].tolist(), mean_abs[top_idx].tolist()


# ---------------------------------------------------------------------------
# SHAP — plotting
# ---------------------------------------------------------------------------

def _plot_shap_single(feat_list: list, imp_list: list, top_n: int,
                       outdir: Path, title_suffix: str = "") -> None:
    """Save a horizontal bar chart of top-N mean |SHAP| importances to outdir/shap_global_importance.png."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(feat_list[:top_n][::-1], imp_list[:top_n][::-1],
            color="teal", edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value|  (impact on Suicidal prediction)", fontsize=12)
    ax.set_ylabel("Token / Feature", fontsize=12)
    ax.set_title(f"SHAP Global Feature Importance  (top {top_n}){title_suffix}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "shap_global_importance.png", dpi=150)
    plt.close()
    print(f"    SHAP plot saved → {outdir.name}/shap_global_importance.png")


def _plot_shap_comparison(model_results: dict, n_per_panel: int, outdir: Path) -> None:
    """Save a side-by-side SHAP importance panel (one subplot per model) to outdir/shap_comparison.png."""
    names = list(model_results.keys())
    n     = len(names)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        feats = model_results[name]["feats"][:n_per_panel]
        imps  = model_results[name]["imps"][:n_per_panel]
        ax.barh(feats[::-1], imps[::-1], color="teal", edgecolor="white")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean |SHAP|", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    fig.suptitle(f"SHAP Global Feature Importance — Model Comparison  (top {n_per_panel})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "shap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Joint SHAP comparison saved → shap_comparison.png")


# ---------------------------------------------------------------------------
# LIME — data collection + plotting
# ---------------------------------------------------------------------------

def _collect_lime(predictor, texts: list, top_n: int, lime_dir: Path) -> tuple[list, list]:
    """Run LIME, save HTML files, return (feat_list, imp_list) aggregated."""
    from lime.lime_text import LimeTextExplainer
    lime_dir.mkdir(parents=True, exist_ok=True)
    explainer   = LimeTextExplainer(class_names=["Non-Suicidal", "Suicidal"])
    all_weights: dict = {}

    for i, text in enumerate(texts):
        exp = explainer.explain_instance(text, predictor, num_features=top_n, labels=[1])
        exp.save_to_file(str(lime_dir / f"sample_{i:02d}.html"))
        for feat, weight in exp.as_list(label=1):
            all_weights.setdefault(feat, []).append(abs(weight))
        print(f"    LIME sample {i + 1}/{len(texts)} done")

    agg = sorted(
        [(f, float(np.mean(ws))) for f, ws in all_weights.items()],
        key=lambda x: x[1], reverse=True,
    )[:top_n]
    if not agg:
        return [], []
    feats, imps = zip(*agg)
    return list(feats), list(imps)


def _plot_lime_single(feat_list: list, imp_list: list, n_samples: int,
                       top_n: int, outdir: Path, title_suffix: str = "") -> None:
    """Save a horizontal bar chart of aggregated LIME importances to outdir/lime_top_features.png."""
    if not feat_list:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(feat_list[::-1], imp_list[::-1], color="darkorange", edgecolor="white")
    ax.set_xlabel("Mean |LIME Weight|  (impact on Suicidal prediction)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(
        f"LIME Aggregated Feature Importance  (top {top_n}, n={n_samples}){title_suffix}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "lime_top_features.png", dpi=150)
    plt.close()
    print(f"    LIME plot saved → {outdir.name}/lime_top_features.png")


def _plot_lime_comparison(model_results: dict, n_per_panel: int, outdir: Path) -> None:
    """Save a side-by-side LIME importance panel (one subplot per model) to outdir/lime_comparison.png."""
    names = list(model_results.keys())
    n     = len(names)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        feats = model_results[name]["feats"][:n_per_panel]
        imps  = model_results[name]["imps"][:n_per_panel]
        ax.barh(feats[::-1], imps[::-1], color="darkorange", edgecolor="white")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean |LIME Weight|", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    fig.suptitle(f"LIME Aggregated Feature Importance — Model Comparison  (top {n_per_panel})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "lime_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Joint LIME comparison saved → lime_comparison.png")


# ---------------------------------------------------------------------------
# High-level per-model runner (used by compare_all)
# ---------------------------------------------------------------------------

def _run_one_model(
    model_type: str,
    model_dir:  Path,
    device,
    df_test,
    shap_idxs:  np.ndarray,
    lime_idxs:  np.ndarray,
    args,
    subdir:     Path,
) -> dict:
    """Load one model, run SHAP + LIME, save individual plots, return results."""
    is_lr     = model_type in ("lr", "lr_bow")
    text_col  = "text_linear" if is_lr else "text_neural"
    texts_all  = df_test[text_col].tolist()
    texts_shap = [texts_all[i] for i in shap_idxs]
    texts_lime = [texts_all[i] for i in lime_idxs]
    max_len    = args.max_len if args.max_len is not None else 128

    print(f"\n  Loading {model_type} from {model_dir} …")
    if model_type == "transformer":
        model, tokenizer = _load_transformer(model_dir, device)
        predictor = _make_transformer_predictor(
            model, tokenizer, device, max_len, args.batch_size
        )
        shap_fn = lambda: _collect_shap_partition(predictor, texts_shap, args.top_features)
    elif model_type == "rnn":
        model, keras_tok = _load_rnn(model_dir, max_len)
        predictor = _make_keras_predictor(model, keras_tok, max_len)
        shap_fn = lambda: _collect_shap_partition(predictor, texts_shap, args.top_features)
    elif model_type == "lstm":
        model, keras_tok = _load_lstm(model_dir, max_len)
        predictor = _make_keras_predictor(model, keras_tok, max_len)
        shap_fn = lambda: _collect_shap_partition(predictor, texts_shap, args.top_features)
    elif model_type == "bilstm":
        model, keras_tok = _load_bilstm(model_dir, max_len)
        predictor = _make_keras_predictor(model, keras_tok, max_len)
        shap_fn = lambda: _collect_shap_partition(predictor, texts_shap, args.top_features)
    elif model_type == "att_bilstm":
        model, keras_tok = _load_att_bilstm(model_dir, max_len)
        predictor = _make_keras_predictor(model, keras_tok, max_len)
        shap_fn = lambda: _collect_shap_partition(predictor, texts_shap, args.top_features)
    elif model_type == "lr_bow":
        model, vectorizer = _load_lr(model_dir, variant="bow")
        predictor = _make_lr_predictor(model, vectorizer)
        shap_fn = lambda: _collect_shap_lr(model, vectorizer, texts_shap, args.top_features)
    else:  # lr (tfidf)
        model, vectorizer = _load_lr(model_dir, variant="tfidf")
        predictor = _make_lr_predictor(model, vectorizer)
        shap_fn = lambda: _collect_shap_lr(model, vectorizer, texts_shap, args.top_features)

    subdir.mkdir(parents=True, exist_ok=True)

    print(f"  ── SHAP ({model_type}) ──")
    shap_feats, shap_imps = shap_fn()
    _plot_shap_single(shap_feats, shap_imps, args.top_features, subdir)

    print(f"  ── LIME ({model_type}) ──")
    lime_feats, lime_imps = _collect_lime(
        predictor, texts_lime, args.top_features, subdir / "lime"
    )
    _plot_lime_single(lime_feats, lime_imps, len(texts_lime),
                      args.top_features, subdir)

    return {
        "shap": {
            "feats": shap_feats,
            "imps":  shap_imps,
            "top_features": [{"token": f, "mean_abs_shap": v}
                             for f, v in zip(shap_feats, shap_imps)],
        },
        "lime": {
            "feats": lime_feats,
            "imps":  lime_imps,
            "n_samples": len(texts_lime),
            "top_features": [{"feature": f, "mean_abs_weight": v}
                             for f, v in zip(lime_feats, lime_imps)],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run SHAP and LIME explainability in single-model or compare-all mode.

    Single-model mode (default): loads one model type, runs SHAP and LIME on
    fixed random samples from the test set, saves individual plots and results.json.

    Compare-all mode (--compare_all): iterates all seven model types, produces
    per-model subdirectories, then generates joint shap_comparison.png and
    lime_comparison.png panels. Sample indices are shared across all models
    so cross-model comparisons are fair.
    """
    args   = parse_args()
    utils.set_seeds(args.seed)
    device = utils.get_device()
    print(f"Device : {device}")

    outdir = utils.make_results_dir(
        base=args.results_base,
        experiment_name="explain",
        timestamp=args.timestamp,
    )
    print(f"Results → {outdir}")

    print("Loading dataset …")
    df = utils.load_dataset(args.dataset)
    _, df_test, _, _ = utils.make_splits(df, seed=args.seed)
    df_test = df_test.reset_index(drop=True)

    # Fixed sample indices — same across all models for fair comparison
    rng       = np.random.default_rng(args.seed)
    n         = len(df_test)
    shap_idxs = rng.choice(n, size=min(args.n_shap_samples, n), replace=False)
    lime_idxs = rng.choice(n, size=min(args.n_lime_samples, n),  replace=False)

    # ── Compare-all mode ──────────────────────────────────────────────────────
    if args.compare_all:
        models_dir = Path(args.models_dir)
        transformer_ckpt = models_dir / "distilroberta_lora_final"

        configs = [
            ("lr_bow",      models_dir,       models_dir / "BoW_LR_model.pkl"),
            ("lr",          models_dir,       models_dir / "LR_model.pkl"),
            ("rnn",         models_dir,       models_dir / "RNN_model.keras"),
            ("lstm",        models_dir,       models_dir / "LSTM_model.keras"),
            ("bilstm",      models_dir,       models_dir / "BILSTM_model.keras"),
            ("att_bilstm",  models_dir,       models_dir / "AttBiLSTM_model.keras"),
            ("transformer", transformer_ckpt, transformer_ckpt / "adapter_config.json"),
        ]
        labels = {
            "lr_bow":      "LR (BoW)",
            "lr":          "LR (TF-IDF)",
            "rnn":         "Simple RNN",
            "lstm":        "LSTM",
            "bilstm":      "BiLSTM",
            "att_bilstm":  "Attention BiLSTM",
            "transformer": "DistilRoBERTa+LoRA",
        }
        all_results  = {}
        shap_compare = {}
        lime_compare = {}

        for model_type, model_dir, ckpt_file in configs:
            if not ckpt_file.exists():
                print(f"\n  [SKIP] {model_type}: checkpoint not found.")
                continue

            label  = labels[model_type]
            subdir = outdir / model_type
            try:
                res = _run_one_model(
                    model_type, model_dir, device,
                    df_test, shap_idxs, lime_idxs, args, subdir,
                )
                all_results[label]  = res
                shap_compare[label] = res["shap"]
                lime_compare[label] = res["lime"]
            except Exception as exc:
                print(f"\n  [ERROR] {model_type} failed: {exc}")

        print("\n── Joint comparison plots ───────────────────────────────────────────")
        _plot_shap_comparison(shap_compare, args.n_compare_features, outdir)
        _plot_lime_comparison(lime_compare, args.n_compare_features, outdir)

        utils.save_metadata(outdir, {
            "mode":            "compare_all",
            "n_shap_samples":  args.n_shap_samples,
            "n_lime_samples":  args.n_lime_samples,
            "top_features":    args.top_features,
            "n_compare_features": args.n_compare_features,
        })
        with open(outdir / "results.json", "w") as f:
            json.dump(
                {k: {"shap": v["shap"]["top_features"],
                      "lime": v["lime"]["top_features"]}
                 for k, v in all_results.items()},
                f, indent=2, default=str,
            )

    # ── Single-model mode ─────────────────────────────────────────────────────
    else:
        if args.max_len is None:
            args.max_len = 128

        model_dir = Path(args.model_dir)
        # For transformer, resolve to the LoRA checkpoint subdir if needed
        if args.model_type == "transformer" and not (model_dir / "adapter_config.json").exists() \
                and not (model_dir / "config.json").exists():
            model_dir = model_dir / "distilroberta_lora_final"

        res = _run_one_model(
            args.model_type, model_dir, device,
            df_test, shap_idxs, lime_idxs, args, outdir,
        )

        utils.save_metadata(outdir, {
            "model_type":      args.model_type,
            "model_dir":       str(model_dir),
            "n_shap_samples":  args.n_shap_samples,
            "n_lime_samples":  args.n_lime_samples,
            "top_features":    args.top_features,
            "max_len":         args.max_len,
        })
        with open(outdir / "results.json", "w") as f:
            json.dump(
                {"shap": res["shap"]["top_features"],
                 "lime": res["lime"]["top_features"]},
                f, indent=2, default=str,
            )

    print(f"\nAll results saved to {outdir}")


if __name__ == "__main__":
    main()
