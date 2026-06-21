"""
lora_sweep.py
-------------
Hyperparameter sweep over LoRA rank (r) and scaling factor (α) on a 20 %
stratified sub-sample of the training set.  α can be less than, equal to,
or greater than r — there is no constraint.

Two modes
---------
1. Explicit pairs (recommended for one-at-a-time runs):
     --configs r:alpha [r:alpha ...]
   e.g.  --configs 4:2 8:8 8:16 16:32

2. Grid sweep (generates every rank × multiplier combination):
     --ranks 2 4 8 16 32 --alpha_multipliers 0.5 1.0 2.0 4.0

Full fine-tuning baseline
-------------------------
  --full_finetune          include full FT (default behaviour)
  --no_full_finetune       skip full FT
  --only_full_finetune     skip all LoRA configs, run only full FT

Each run is saved to {checkpoints_dir}/r{r}_a{alpha}/.
If that directory already contains an adapter_config.json or config.json
the training step is skipped and the existing checkpoint is evaluated
directly — making the script safe to re-run after interruptions.

Outputs per run
---------------
  {results_dir}/r{r}_a{alpha}/
    metadata.json      – config, trainable params, full metric suite
    loss_curve.png     – train & val loss trajectories

Outputs aggregated
------------------
  {results_dir}/
    sweep_results.json – all configs + metrics in one file
    sweep_summary.png  – four-panel line chart (F1, AUROC, FNR, params)

Usage
-----
# Run a single LoRA config (no full FT):
python lora_sweep.py --configs 8:16 --no_full_finetune

# Run only full fine-tuning:
python lora_sweep.py --only_full_finetune

# Run a few explicit pairs + full FT:
python lora_sweep.py --configs 4:2 4:8 8:4 16:32 --full_finetune

# Classic grid sweep over all ranks:
python lora_sweep.py \\
    --ranks 2 4 8 16 32 \\
    --alpha_multipliers 0.5 1.0 2.0 4.0 \\
    --no_full_finetune \\
    --dataset ./Dataset/Suicide_Detection.csv \\
    --base_model distilroberta-base \\
    --checkpoints_dir ./Models/lora_sweep \\
    --results_base ./results \\
    --seed 42 --epochs 3 --batch_size 128
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import utils


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse and return CLI arguments for the LoRA rank/alpha sweep."""
    p = argparse.ArgumentParser(description="LoRA rank / alpha sweep for DistilRoBERTa")
    p.add_argument("--dataset",           default="./Dataset/Suicide_Detection.csv")
    p.add_argument("--base_model",         default="distilroberta-base",
                   help="HuggingFace model ID for the base model")
    p.add_argument("--checkpoints_dir",    default="./Models/lora_sweep",
                   help="Root directory for per-config checkpoint saves")
    p.add_argument("--results_base",       default="./results")
    p.add_argument("--timestamp",          default=None,
                   help="Shared timestamp from run_all.sh")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--epochs",             type=int,   default=3)
    p.add_argument("--batch_size",         type=int,   default=128)
    p.add_argument("--max_len",            type=int,   default=128,
                   help="Token max length for transformer training")
    p.add_argument("--learning_rate",      type=float, default=5e-4)
    p.add_argument("--warmup_steps",            type=int,   default=500)
    p.add_argument("--lora_dropout",            type=float, default=0.1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Accumulate gradients over N steps before an optimizer update. "
                        "Effective batch = batch_size × N. Use when batch_size 128 "
                        "does not fit in GPU memory (e.g. set N=4 with batch_size=32).")
    p.add_argument("--configs",            nargs="+",  default=None,
                   metavar="r:alpha",
                   help="Explicit rank:alpha pairs to run, e.g. '4:2 8:16 16:8'. "
                        "Overrides --ranks / --alpha_multipliers / --alpha_fixed "
                        "when provided. Alpha may be less than, equal to, or "
                        "greater than rank.")
    p.add_argument("--ranks",              nargs="+",  type=int,
                   default=[2, 4, 8, 16, 32],
                   help="LoRA ranks for grid sweep (ignored when --configs is set)")
    p.add_argument("--alpha_multipliers",  nargs="+",  type=float,
                   default=[1.0, 2.0, 4.0],
                   help="Alpha as multiples of r for grid sweep, e.g. '0.5 1 2 4'. "
                        "Ignored when --configs is set.")
    p.add_argument("--alpha_fixed",        type=int,   default=0,
                   help="Extra fixed alpha added at every rank in grid sweep "
                        "(0 = disabled). Ignored when --configs is set.")
    p.add_argument("--target_modules",     nargs="+",
                   default=["query", "value"],
                   help="Attention weight matrices to inject LoRA into")

    ft_group = p.add_mutually_exclusive_group()
    ft_group.add_argument("--full_finetune",       action="store_true",  dest="full_finetune",
                          help="Include full fine-tuning comparison run (default)")
    ft_group.add_argument("--no_full_finetune",    action="store_false", dest="full_finetune",
                          help="Skip the full fine-tuning comparison run")
    ft_group.add_argument("--only_full_finetune",  action="store_true",  dest="only_full_finetune",
                          help="Skip all LoRA configs; run only full fine-tuning")
    p.set_defaults(full_finetune=True, only_full_finetune=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SuicideDataset(Dataset):
    """Tokenises text on-the-fly; returns HuggingFace-compatible dict."""

    def __init__(self, texts, labels, tokenizer, max_len: int = 64):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Trainer compute_metrics
# ---------------------------------------------------------------------------

def _make_compute_metrics():
    """Return a HuggingFace Trainer-compatible compute_metrics callback.

    Extracts accuracy, precision, recall, F1, and ECE from EvalPrediction
    objects. ECE uses a numerically stable softmax (row-wise shift before
    exponentiation) to avoid float overflow on large logit values.

    Returns:
        Callable matching the Trainer's compute_metrics signature.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    def compute_metrics(pred):
        labels = pred.label_ids
        preds  = pred.predictions.argmax(-1)

        # Numerically stable softmax → positive-class probability
        logits = pred.predictions
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_p   = np.exp(shifted)
        y_prob  = exp_p[:, 1] / exp_p.sum(axis=1)
        ece     = utils.compute_ece(labels, y_prob)

        return {
            "accuracy":  float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall":    float(recall_score(labels, preds,    zero_division=0)),
            "f1":        float(f1_score(labels, preds,        zero_division=0)),
            "ece":       float(ece),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Count trainable parameters for a given LoRA config (no GPU allocation)
# ---------------------------------------------------------------------------

def _count_trainable_params(base_model_name: str, r: int, alpha: int,
                              target_modules, lora_dropout: float) -> tuple[int, int]:
    """Instantiate a LoRA-wrapped model on CPU and count trainable vs total parameters.

    Used when a checkpoint already exists so we can populate the params column
    without re-running training. The model is constructed and immediately deleted
    to avoid holding GPU memory across sweep iterations.

    Args:
        base_model_name: HuggingFace model ID (e.g. 'distilroberta-base').
        r: LoRA rank.
        alpha: LoRA scaling factor (lora_alpha).
        target_modules: Attention projection matrices to inject adapters into.
        lora_dropout: Dropout rate inside the LoRA adapter layers.

    Returns:
        (trainable_params, total_params) integer tuple.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=2
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r, lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    m = get_peft_model(base, lora_cfg)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    del m
    return trainable, total


# ---------------------------------------------------------------------------
# Train one LoRA configuration
# ---------------------------------------------------------------------------

def _train_config(
    r: int,
    alpha: int,
    sweep_dataset: SuicideDataset,
    test_dataset:  SuicideDataset,
    args,
    checkpoint_dir: Path,
    device,
) -> dict:
    """Fine-tune DistilRoBERTa with the given LoRA config; return training metadata."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r, lora_alpha=alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"    trainable: {trainable:,}  /  total: {total:,}  "
          f"({100 * trainable / total:.3f} %)")

    model.to(device)

    tokenizer     = AutoTokenizer.from_pretrained(args.base_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            use_fp16 = False  # bf16 preferred for fine-tuning: same dynamic range as fp32,
            use_bf16 = True   # no loss scaling needed; hardware-accelerated on Ampere+
        else:
            use_fp16 = True   # pre-Ampere fallback (V100, T4): no bf16 hardware support
            use_bf16 = False
        use_pin_memory = True   # pinned memory → faster async host→GPU DMA transfers
    elif device.type == "mps":
        use_fp16       = False  # fp16 AMP not supported on MPS
        use_bf16       = True   # Apple Silicon supports bf16 natively
        use_pin_memory = False
    else:
        use_fp16       = False  # CPU has no reduced-precision hardware acceleration
        use_bf16       = False
        use_pin_memory = False

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=0,
        dataloader_pin_memory=use_pin_memory,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sweep_dataset,
        eval_dataset=test_dataset,
        compute_metrics=_make_compute_metrics(),
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    train_losses, eval_losses = _parse_log_history(trainer.state.log_history)

    return {
        "trainable_params": trainable,
        "total_params":     total,
        "trainable_pct":    100.0 * trainable / total,
        "train_loss_curve": train_losses,
        "eval_loss_curve":  eval_losses,
    }


# ---------------------------------------------------------------------------
# Full fine-tuning (all parameters unfrozen — upper-bound comparison)
# ---------------------------------------------------------------------------

def _train_full_finetune(
    sweep_dataset: SuicideDataset,
    test_dataset:  SuicideDataset,
    args,
    checkpoint_dir: Path,
    device,
) -> dict:
    """Fine-tune all parameters of the base model; return training metadata."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"    trainable: {trainable:,}  /  total: {total:,}  (100.000 %)")

    model.to(device)

    tokenizer     = AutoTokenizer.from_pretrained(args.base_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            use_fp16, use_bf16 = False, True
        else:
            use_fp16, use_bf16 = True, False
        use_pin_memory = True
    elif device.type == "mps":
        use_fp16, use_bf16, use_pin_memory = False, True, False
    else:
        use_fp16, use_bf16, use_pin_memory = False, False, False

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=2e-5,       # standard full fine-tuning LR (much lower than LoRA)
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=0,
        dataloader_pin_memory=use_pin_memory,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sweep_dataset,
        eval_dataset=test_dataset,
        compute_metrics=_make_compute_metrics(),
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    train_losses, eval_losses = _parse_log_history(trainer.state.log_history)

    return {
        "trainable_params": trainable,
        "total_params":     total,
        "trainable_pct":    100.0,
        "train_loss_curve": train_losses,
        "eval_loss_curve":  eval_losses,
    }


# ---------------------------------------------------------------------------
# Log-history parsing (robust against key variations across HF versions)
# ---------------------------------------------------------------------------

def _parse_log_history(log_history: list) -> tuple[list, list]:
    """Extract (train_losses, eval_losses) from Trainer.state.log_history."""
    train_losses, eval_losses = [], []
    for entry in log_history:
        try:
            if "eval_loss" in entry:
                eval_losses.append((entry["epoch"], entry["eval_loss"]))
            elif "loss" in entry and "epoch" in entry:
                train_losses.append((entry["epoch"], entry["loss"]))
        except (KeyError, TypeError):
            pass
    return train_losses, eval_losses


# ---------------------------------------------------------------------------
# Evaluate an existing checkpoint (no retraining)
# ---------------------------------------------------------------------------

def _eval_checkpoint(
    checkpoint_dir: Path,
    test_dataset:   SuicideDataset,
    y_true:         np.ndarray,
    device,
    batch_size:     int,
    tokenizer=None,
) -> dict:
    """
    Run inference from a saved checkpoint and return the full metric suite.

    Parameters
    ----------
    tokenizer : optional pre-loaded tokenizer (must match the one used to
                build test_dataset).  If None, loaded from checkpoint_dir.
                Passing the same instance avoids a redundant from_pretrained
                call and eliminates any risk of tokenizer mismatch.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    if (checkpoint_dir / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel
        cfg   = PeftConfig.from_pretrained(str(checkpoint_dir))
        base  = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model_name_or_path, num_labels=2
        )
        model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))

    model.to(device)
    model.eval()

    # Use the caller-supplied tokenizer when available so the collator is
    # guaranteed to match the tokenizer used to build test_dataset.
    collator_tok = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
        str(checkpoint_dir)
    )
    collator = DataCollatorWithPadding(tokenizer=collator_tok)
    loader   = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            enc    = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)

    y_prob = np.concatenate(all_probs)
    # free GPU memory before the next run
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return utils.compute_full_metrics(y_true, y_prob)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_loss_curve(
    train_losses: list,
    eval_losses:  list,
    run_name:     str,
    outdir:       Path,
) -> None:
    """Plot and save train/val loss trajectories for one sweep run to outdir/loss_curve.png."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if train_losses:
        steps, losses = zip(*train_losses)
        ax.plot(steps, losses, alpha=0.6, label="Train loss")
    if eval_losses:
        epochs, losses = zip(*eval_losses)
        ax.plot(epochs, losses, "o-", lw=2, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"Training Dynamics — {run_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "loss_curve.png", dpi=150)
    plt.close()


def _alpha_strategy_label(r: int, alpha: int) -> str:
    """Return a human-readable label for the alpha strategy."""
    for num, den in [(1, 4), (1, 2), (1, 1), (2, 1), (4, 1), (8, 1)]:
        if alpha * den == r * num:
            if den == 1:
                return "α=r" if num == 1 else f"α={num}r"
            return f"α=r/{den}"
    return f"α={alpha}"


def _plot_sweep_summary(sweep_results: dict, outdir: Path) -> None:
    """
    Line charts (F1, AUROC, FNR) with one line per alpha strategy and a
    horizontal dashed line for full fine-tuning, plus a params bar chart.
    Handles both the original 1D sweep and the 2D rank × alpha grid.
    """
    from collections import defaultdict

    lora    = {k: v for k, v in sweep_results.items() if v["r"] is not None}
    full_ft = sweep_results.get("full_finetune")

    if not lora:
        return

    ranks = sorted(set(v["r"] for v in lora.values()))

    # Group configs by alpha strategy label
    groups: dict = defaultdict(dict)   # strategy_label → {r: result_dict}
    for v in lora.values():
        label = _alpha_strategy_label(v["r"], v["alpha"])
        groups[label][v["r"]] = v

    line_colors = ["steelblue", "darkorange", "mediumseagreen", "tomato", "purple"]
    strategies  = sorted(groups.keys())

    metrics_cfg = [
        ("f1",    "F1 Score",       "higher →"),
        ("auroc", "AUROC",          "higher →"),
        ("fnr",   "FNR",            "← lower"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # --- Panels 0-2: line charts per metric ---
    for ax, (metric, title, direction) in zip(axes[:3], metrics_cfg):
        for strategy, color in zip(strategies, line_colors):
            grp = groups[strategy]
            xs  = sorted(grp.keys())
            ys  = [grp[r]["metrics"][metric] for r in xs]
            ax.plot(xs, ys, "o-", label=strategy, color=color, linewidth=2, markersize=7)
        if full_ft:
            ax.axhline(
                full_ft["metrics"][metric],
                color="black", linestyle="--", linewidth=1.5, label="Full FT",
            )
        ax.set_xlabel("LoRA Rank (r)", fontsize=11)
        ax.set_ylabel(f"{title}  ({direction})", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(ranks)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 3: trainable params vs rank (params depend on r, not alpha) ---
    ax = axes[3]
    # Use one representative alpha strategy to read params per rank
    rep_strategy = strategies[0]
    p_ranks = sorted(groups[rep_strategy].keys())
    p_vals  = [groups[rep_strategy][r]["trainable_params"] / 1_000 for r in p_ranks]

    bars = ax.bar(range(len(p_ranks)), p_vals, color="mediumseagreen", edgecolor="white")
    if full_ft:
        ax.axhline(
            full_ft["trainable_params"] / 1_000,
            color="black", linestyle="--", linewidth=1.5, label="Full FT",
        )
    ax.set_xticks(range(len(p_ranks)))
    ax.set_xticklabels([f"r={r}" for r in p_ranks], fontsize=10)
    ax.set_ylabel("Trainable Params (K)", fontsize=11)
    ax.set_title("Trainable Params (K)\nvs. Rank", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, p_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.1f}K", ha="center", va="bottom", fontsize=8,
        )

    fig.suptitle("LoRA Rank × Alpha Sweep — Performance vs. Efficiency",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(outdir / "sweep_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full LoRA rank/alpha sweep, evaluate each config, and aggregate results.

    Builds the (rank, alpha) list from --configs or the grid defined by
    --ranks × --alpha_multipliers. For each config, trains on a 20% stratified
    sub-sample of the training set (or skips if a checkpoint already exists),
    evaluates on the held-out test set, and saves per-config loss curves and
    metadata. Aggregated outputs (sweep_results.json, sweep_summary.png) land
    in the experiment results directory.
    """
    args = parse_args()
    utils.set_seeds(args.seed)
    device = utils.get_device()
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"Device : {device}")

    outdir = utils.make_results_dir(
        base=args.results_base,
        experiment_name="lora_sweep",
        timestamp=args.timestamp,
    )
    print(f"Results → {outdir}")

    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # --- dataset + splits ---
    print("Loading dataset …")
    df = utils.load_dataset(args.dataset)
    df_train, df_test, y_train_full, y_test = train_test_split(
        df, df["label"],
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label"],
    )

    # 20 % stratified sub-sample of the training set
    df_sweep, _, y_sweep, _ = train_test_split(
        df_train, y_train_full,
        train_size=0.2,
        random_state=args.seed,
        stratify=y_train_full,
    )
    y_test_arr = y_test.values
    print(f"Sweep train size : {len(df_sweep):,}  (20 % of {len(df_train):,})")
    print(f"Test  set size   : {len(y_test_arr):,}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    sweep_dataset = SuicideDataset(
        df_sweep["text_neural"].values, y_sweep.values,
        hf_tokenizer, max_len=args.max_len,
    )
    test_dataset = SuicideDataset(
        df_test["text_neural"].values, y_test_arr,
        hf_tokenizer, max_len=args.max_len,
    )

    # Build the (rank, alpha) list for this run
    if args.only_full_finetune:
        rank_alpha_configs: list[tuple[int, int]] = []
        print("\n[--only_full_finetune] Skipping all LoRA configs.")
    elif args.configs:
        rank_alpha_configs = []
        for spec in args.configs:
            try:
                r_str, a_str = spec.split(":")
                rank_alpha_configs.append((int(r_str), int(a_str)))
            except ValueError:
                raise ValueError(
                    f"--configs entries must be 'r:alpha' integers, got: '{spec}'"
                )
        seen: set = set()
        deduped = []
        for pair in rank_alpha_configs:
            if pair not in seen:
                deduped.append(pair)
                seen.add(pair)
        if len(deduped) < len(rank_alpha_configs):
            print(f"[WARN] Removed {len(rank_alpha_configs) - len(deduped)} duplicate "
                  f"r:alpha pair(s) from --configs.")
        rank_alpha_configs = deduped
        print(f"\nLoRA configs (explicit): {len(rank_alpha_configs)}")
        for r, a in rank_alpha_configs:
            print(f"  r={r:>2}  α={a:>3}  ({_alpha_strategy_label(r, a)})")
    else:
        rank_alpha_configs = []
        for r in args.ranks:
            seen_alphas: set = set()
            for mult in args.alpha_multipliers:
                a = max(1, int(round(mult * r)))
                if a not in seen_alphas:
                    rank_alpha_configs.append((r, a))
                    seen_alphas.add(a)
            if args.alpha_fixed > 0 and args.alpha_fixed not in seen_alphas:
                rank_alpha_configs.append((r, args.alpha_fixed))
        print(f"\nLoRA configs (grid): {len(rank_alpha_configs)}")
        for r, a in rank_alpha_configs:
            print(f"  r={r:>2}  α={a:>3}  ({_alpha_strategy_label(r, a)})")

    sweep_results: dict = {}

    for r, alpha in rank_alpha_configs:
        run_name = f"r{r}_a{alpha}"
        ckpt_dir = checkpoints_dir / run_name
        run_out  = outdir / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 62}")
        print(f"  Config: r={r}, α={alpha}  →  {ckpt_dir}")
        print(f"{'=' * 62}")

        already_trained = (
            (ckpt_dir / "adapter_config.json").exists() or
            (ckpt_dir / "config.json").exists()
        )

        if already_trained:
            meta_path = ckpt_dir / "metadata.json"
            if meta_path.exists():
                try:
                    stored = json.loads(meta_path.read_text())
                    stored_cfg = stored.get("config", {})
                    if stored_cfg.get("r") != r or stored_cfg.get("alpha") != alpha:
                        print(f"  [WARN] Checkpoint config mismatch: stored "
                              f"r={stored_cfg.get('r')}, α={stored_cfg.get('alpha')} "
                              f"vs requested r={r}, α={alpha}. "
                              f"Delete {ckpt_dir} to force a clean retrain.")
                except Exception:
                    pass
            print(f"  [SKIP] Checkpoint exists — evaluating without retraining.")
            trainable, total = _count_trainable_params(
                args.base_model, r, alpha, args.target_modules, args.lora_dropout
            )
            train_meta = {
                "trainable_params": trainable,
                "total_params":     total,
                "trainable_pct":    100.0 * trainable / total,
                "train_loss_curve": [],
                "eval_loss_curve":  [],
            }
        else:
            train_meta = _train_config(
                r=r, alpha=alpha,
                sweep_dataset=sweep_dataset,
                test_dataset=test_dataset,
                args=args,
                checkpoint_dir=ckpt_dir,
                device=device,
            )
            if train_meta["train_loss_curve"] or train_meta["eval_loss_curve"]:
                _plot_loss_curve(
                    train_meta["train_loss_curve"],
                    train_meta["eval_loss_curve"],
                    run_name, run_out,
                )

        # --- full metric evaluation ---
        print(f"  Evaluating {run_name} …")
        metrics = _eval_checkpoint(
            ckpt_dir, test_dataset, y_test_arr, device, args.batch_size,
            tokenizer=hf_tokenizer,
        )
        print(f"  F1={metrics['f1']:.4f}  "
              f"AUROC={metrics['auroc']:.4f}  "
              f"FNR={metrics['fnr']:.4f}  "
              f"ECE={metrics['ece']:.4f}  "
              f"params={train_meta['trainable_params']:,}")

        sweep_results[run_name] = {
            "r":               r,
            "alpha":           alpha,
            "trainable_params": train_meta["trainable_params"],
            "total_params":     train_meta["total_params"],
            "trainable_pct":    train_meta["trainable_pct"],
            "metrics":          metrics,
        }

        utils.save_metadata(run_out, {
            "config": {
                "r": r, "alpha": alpha,
                "base_model":     args.base_model,
                "target_modules": args.target_modules,
                "lora_dropout":   args.lora_dropout,
            },
            "trainable_params": train_meta["trainable_params"],
            "total_params":     train_meta["total_params"],
            "trainable_pct":    train_meta["trainable_pct"],
            "sweep_train_size": int(len(df_sweep)),
            "test_size":        int(len(y_test_arr)),
            "metrics":          metrics,
        })

    # ------------------------------------------------------------------
    # Full fine-tuning — upper-bound comparison point
    # ------------------------------------------------------------------
    if args.full_finetune:
        run_name = "full_finetune"
        ckpt_dir = checkpoints_dir / run_name
        run_out  = outdir / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 62}")
        print(f"  Config: Full Fine-Tuning  →  {ckpt_dir}")
        print(f"{'=' * 62}")

        already_trained = (ckpt_dir / "config.json").exists()

        if already_trained:
            print(f"  [SKIP] Checkpoint exists — evaluating without retraining.")
            base = AutoModelForSequenceClassification.from_pretrained(
                args.base_model, num_labels=2
            )
            trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in base.parameters())
            del base
            train_meta = {
                "trainable_params": trainable,
                "total_params":     total,
                "trainable_pct":    100.0,
                "train_loss_curve": [],
                "eval_loss_curve":  [],
            }
        else:
            train_meta = _train_full_finetune(
                sweep_dataset=sweep_dataset,
                test_dataset=test_dataset,
                args=args,
                checkpoint_dir=ckpt_dir,
                device=device,
            )
            if train_meta["train_loss_curve"] or train_meta["eval_loss_curve"]:
                _plot_loss_curve(
                    train_meta["train_loss_curve"],
                    train_meta["eval_loss_curve"],
                    run_name, run_out,
                )

        print(f"  Evaluating {run_name} …")
        metrics = _eval_checkpoint(
            ckpt_dir, test_dataset, y_test_arr, device, args.batch_size,
            tokenizer=hf_tokenizer,
        )
        print(f"  F1={metrics['f1']:.4f}  "
              f"AUROC={metrics['auroc']:.4f}  "
              f"FNR={metrics['fnr']:.4f}  "
              f"ECE={metrics['ece']:.4f}  "
              f"params={train_meta['trainable_params']:,}")

        sweep_results[run_name] = {
            "r":                None,
            "alpha":            None,
            "trainable_params": train_meta["trainable_params"],
            "total_params":     train_meta["total_params"],
            "trainable_pct":    100.0,
            "metrics":          metrics,
        }

        utils.save_metadata(run_out, {
            "config": {
                "type":       "full_finetune",
                "base_model": args.base_model,
            },
            "trainable_params": train_meta["trainable_params"],
            "total_params":     train_meta["total_params"],
            "trainable_pct":    100.0,
            "sweep_train_size": int(len(df_sweep)),
            "test_size":        int(len(y_test_arr)),
            "metrics":          metrics,
        })

    # --- aggregated outputs ---
    if sweep_results:
        _plot_sweep_summary(sweep_results, outdir)

    with open(outdir / "sweep_results.json", "w") as fh:
        json.dump(sweep_results, fh, indent=2)

    # --- console table ---
    cols = ["f1", "auroc", "auprc", "fnr", "ece"]
    print("\n" + "=" * 86)
    print(f"{'Config':<14}{'Strategy':<16}{'Params':>14}" +
          "".join(f"{c:>10}" for c in cols))
    print("-" * 86)
    for name, res in sweep_results.items():
        if res["r"] is None:
            strategy = "Full FT"
        else:
            strategy = _alpha_strategy_label(res["r"], res["alpha"])
        row = (
            f"{name:<14}{strategy:<16}{res['trainable_params']:>14,}" +
            "".join(f"{res['metrics'][c]:>10.4f}" for c in cols)
        )
        print(row)
    print("=" * 86)
    print(f"\nAll results saved to {outdir}")


if __name__ == "__main__":
    main()
