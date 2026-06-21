# From Frequencies to Self-Attention: Interpretable Deep Learning for Early Suicide Risk Detection

> **AdComSys 2026 Camera-Ready** — Subham Bagchi (NCI) · Debmalya Pal (UCSD)

---

## What & Why ?

Social media platforms have created an observable record of psychological distress: individuals experiencing suicidal ideation often articulate crisis states in public posts before any clinical contact occurs. Automated screening at scale could enable early, targeted intervention.

This codebase implements a **seven-model architectural ablation study** that traces the explicit evolution of NLP from simple frequency-based classifiers to parameter-efficient self-attention Transformers. Each step is motivated by the failure mode of its predecessor:

| Stage | Model | Why it was added ? |
|---|---|---|
| 1 | Logistic Regression (BoW) | Frequency-based baseline; establishes the "semantic gap" ceiling |
| 2 | Logistic Regression (TF-IDF) | Inverse-document weighting reduces stop-word dominance |
| 3 | Simple RNN | Introduces sequential memory; exposed by vanishing gradients on long posts |
| 4 | LSTM | Gating (input / forget / output gates) resolves vanishing gradients; context from the full post reaches the classifier |
| 5 | BiLSTM | Processes each post in both directions simultaneously |
| 6 | Attention BiLSTM | Adds token-level attention; reveals precision–recall trade-off |
| 7 | DistilRoBERTa + LoRA | Full-sequence self-attention with <1% trainable parameters |

The primary clinical safety metric is **False Negative Rate (FNR)**: every missed positive is a missed opportunity for intervention.

---

## Key Results

| Model | Accuracy | F1 | AUROC | AUPRC | FNR |
|---|---|---|---|---|---|
| LR (BoW) | 93.10% | 92.96% | 97.69% | 97.79% | 8.87% |
| LR (TF-IDF) | 93.87% | 93.82% | 98.26% | 98.21% | 7.05% |
| Simple RNN | 70.67% | 73.05% | 78.85% | 78.15% | 20.51% |
| LSTM | 93.77% | 93.82% | 98.24% | 98.06% | 5.31% |
| BiLSTM | 93.60% | 93.68% | 98.20% | 98.01% | 5.20% |
| Attention BiLSTM | 93.09% | 92.97% | 98.06% | 97.89% | 8.65% |
| **DistilRoBERTa+LoRA** | **95.95%** | **95.98%** | **99.27%** | **99.30%** | **3.47%** |

Test set: N = 46,415 (balanced, 20% stratified split, seed = 42).  
All pairwise differences crossing the Transformer boundary are statistically significant (McNemar's, Bonferroni α = 0.0024).

---

## Repository Structure

```
.
├── setup.sh                    # One-shot environment + dataset bootstrap
├── run_all.sh                  # Full 9-step experiment pipeline
├── requirements.txt
│
├── train_lr.py                 # Step 1 — Logistic Regression (BoW + TF-IDF)
├── train_rnn.py                # Step 2 — Simple RNN
├── train_lstm.py               # Step 3 — LSTM
├── train_bilstm.py             # Step 4 — BiLSTM + Attention BiLSTM
├── evaluate_checkpoints.py     # Step 5 — Unified checkpoint evaluation
├── lora_sweep.py               # Step 6 — LoRA rank × alpha grid search
├── imbalance_eval.py           # Step 7 — Class-imbalance robustness evaluation
├── explain.py                  # Step 8 — SHAP + LIME explainability
├── stat_test.py                # Step 9 — Pairwise McNemar's significance tests
├── utils.py                    # Shared data loading, preprocessing, seeding
│
├── Dataset/
│   └── Suicide_Detection.csv   # 232,074 Reddit posts (downloaded by setup.sh)
│
├── Models/                     # Saved checkpoints (populated by training steps)
│   ├── LR_model.pkl            # LR + TF-IDF
│   ├── Vectorizer_model.pkl    # TF-IDF vectorizer
│   ├── BoW_LR_model.pkl        # LR + BoW
│   ├── BoW_Vectorizer_model.pkl
│   ├── Tokenizer_model.json    # Keras tokenizer (shared by RNN/LSTM/BiLSTM)
│   ├── Tokenizer_model.pkl     # Legacy pickle fallback (auto-loaded if .json absent)
│   ├── RNN_model.keras
│   ├── LSTM_model.keras
│   ├── BILSTM_model.keras
│   ├── AttBiLSTM_model.keras
│   └── distilroberta_lora_final/   # LoRA-adapted DistilRoBERTa checkpoint
│
├── Plots/                      # Paper-ready figures
│   ├── 1_Class_Distribution.png
│   ├── 3_CM_LogReg_BoW.png
│   ├── 4_CM_LogReg_TFIDF.png
│   ├── 5_CM_RNN.png
│   ├── 6_CM_LSTM.png
│   ├── 7_CM_BiLSTM.png
│   ├── 7b_CM_Attention_BiLSTM.png
│   ├── 8_Loss_Comparison.png
│   ├── 9_CM_Transformer.png
│   ├── 10_LIME_1.png           # LIME prediction probability panel
│   ├── 10_LIME_2.png           # LIME feature importance panel
│   ├── 10_LIME_3.png           # LIME text-highlight panel
│   └── 11_SHAP_Summary.png
│
└── results/                    # Timestamped experiment outputs (git-ignored)
    └── YYYYMMDD_HHMMSS/
        ├── train_lr/
        ├── train_rnn/
        ├── train_lstm/
        ├── train_bilstm/
        ├── evaluate_checkpoints/   # Per-model ROC, PR, calibration, CM, summary.json
        ├── lora_sweep/             # Rank × alpha grid; sweep_results.json
        ├── imbalance_eval/         # Metrics at 1:1, 5:1, 10:1, 20:1 neg:pos
        ├── explain/                # SHAP global importance + LIME local explanations
        └── stat_test/              # McNemar contingency tables + p-values
```

---

## Replication Guide

### Prerequisites

- **Python 3.10+** — activate the correct environment before running `setup.sh` (pyenv, conda, and system Python all work).
- A Kaggle account with an API token (`~/.kaggle/kaggle.json`)  
  — or set `KAGGLE_USERNAME` / `KAGGLE_KEY` in a `.env` file at the project root
- A GPU is recommended for Steps 6–9 (LoRA training takes ~30 min on a single GPU). The pipeline auto-selects the best available device: **CUDA > MPS > CPU** — no code changes needed regardless of your hardware

---

### Step 0 — Clone and bootstrap

```bash
git clone https://github.com/stormbreaker3000/Interpretable-Deep-Learning-for-Early-Suicide-Risk-Detection.git
cd Interpretable-Deep-Learning-for-Early-Suicide-Risk-Detection
```

Create a `.env` file in the project root with your Kaggle credentials (required for the dataset download):

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

> Get your API key at kaggle.com → Account → Create New API Token. Alternatively place `kaggle.json` at `~/.kaggle/kaggle.json`.

```bash
bash setup.sh          # creates venv/, installs requirements, downloads dataset

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
# venv\Scripts\Activate.ps1
```

> **Note:** `setup.sh` is a bash script. Windows users should run it inside WSL or Git Bash, then activate with `venv\Scripts\activate`.

`setup.sh` will:
1. Verify Python 3.10+
2. Create `./venv` and install all dependencies from `requirements.txt`
3. Create `Dataset/`, `Models/`, `Plots/`, `results/` directories
4. Download `Suicide_Detection.csv` from Kaggle automatically

---

### Step 1 — Train Logistic Regression

```bash
python train_lr.py --dataset ./Dataset/Suicide_Detection.csv --force_retrain
```

**What it does:** Trains two LR classifiers — one on Bag-of-Words features, one on TF-IDF. Performs 5-fold cross-validation. Saves `LR_model.pkl`, `Vectorizer_model.pkl`, `BoW_LR_model.pkl`, `BoW_Vectorizer_model.pkl` to `Models/`.

**Why:** These are the frequency-based baselines. Their plateau in FNR (~7–9%) defines the "semantic gap" that motivates all subsequent architectures.

---

### Step 2 — Train Simple RNN

```bash
python train_rnn.py --dataset ./Dataset/Suicide_Detection.csv --epochs 3 --force_retrain
```

**What it does:** Trains a single-layer SimpleRNN with a learned embedding (vocab=20k, dim=128). Saves `RNN_model.keras` and the shared Keras tokenizer.

**Why:** Demonstrates the vanishing-gradient bottleneck. The model achieves only 70.67% accuracy (FNR 20.51%) because contextual cues at the start of long posts evaporate before reaching the output gate.

---

### Step 3 — Train LSTM

```bash
python train_lstm.py --dataset ./Dataset/Suicide_Detection.csv --epochs 3 --force_retrain
```

**What it does:** Trains an LSTM with the same embedding and capacity as the RNN. Saves `LSTM_model.keras`.

**Why:** LSTM gating (input, forget, output gates) resolves the vanishing-gradient problem, recovering FNR from 20.51% to 5.31% — a direct empirical proof of why gated architectures replaced vanilla RNNs.

---

### Step 4 — Train BiLSTM and Attention BiLSTM

```bash
python train_bilstm.py --dataset ./Dataset/Suicide_Detection.csv --epochs 3 --force_retrain
```

**What it does:** Trains two models: a standard BiLSTM (forward + backward passes) and an Attention BiLSTM (adds a dot-product attention layer over BiLSTM hidden states). Saves `BILSTM_model.keras`, `AttBiLSTM_model.keras`, and a loss-curve comparison plot.

**Why:** BiLSTM marginal improves FNR (5.20% vs 5.31%). Attention BiLSTM reveals an important trade-off: precision rises but recall falls (FNR 8.65%), showing that local attention without full-sequence context can hurt sensitivity — the wrong direction for a clinical screening task.

---

### Step 5 — Evaluate all checkpoints

```bash
python evaluate_checkpoints.py \
    --dataset    ./Dataset/Suicide_Detection.csv \
    --models_dir ./Models
```

**What it does:** Loads every saved checkpoint (LR, RNN, LSTM, BiLSTM, Attention BiLSTM, DistilRoBERTa+LoRA) and evaluates each on the held-out test set. Produces per-model confusion matrix, ROC curve, PR curve, calibration curve, and a unified `summary.json` with Accuracy, Precision, Recall, Specificity, F1, AUROC, AUPRC, ECE, and FNR. Any checkpoint that is missing is skipped gracefully.

---

### Step 6 — LoRA rank sensitivity sweep

> Runs on GPU (~30 min on a single GPU); falls back to MPS or CPU automatically.

```bash
python lora_sweep.py \
    --dataset ./Dataset/Suicide_Detection.csv \
    --base_model distilroberta-base \
    --ranks 2 4 8 16 32 64 \
    --epochs 3
```

**What it does:** Grid-searches rank r ∈ {2, 4, 8, 16, 32, 64} with α ∈ {1r, 2r, 4r} plus fixed α=16 at every rank. Saves each adapter to `Models/lora_sweep/r{r}_a{alpha}/` and writes aggregated metrics to `sweep_results.json`.

After the sweep, promote the paper-selected checkpoint (r=8, α=16) to the canonical path used by all downstream steps:

```bash
cp -r Models/lora_sweep/r8_a16 Models/distilroberta_lora_final
```

> When using `bash run_all.sh` this promotion is handled automatically (Step 6b). You only need the `cp` command when running individual scripts.

**Why:** Rather than fine-tuning all 82.1M parameters of DistilRoBERTa (which risks catastrophic forgetting), LoRA injects low-rank decomposition matrices into the query/value projections. Only the adapters are updated. `r=8, α=16` is the optimal configuration: highest recall (<1% trainable parameters at 0.89%). Ranks r≥16 exceed the 1% parameter budget with marginal FNR gains within run-to-run variance.

---

### Step 7 — Class-imbalance robustness evaluation

```bash
python imbalance_eval.py \
    --dataset ./Dataset/Suicide_Detection.csv \
    --model_dir ./Models/distilroberta_lora_final \
    --model_type transformer \
    --ratios 1 5 10 20
```

**What it does:** Constructs test sets at negative-to-positive ratios of 1:1, 5:1, 10:1, 20:1 by retaining all ~23K negatives and downsampling positives. Reports Precision, Recall, F1, AUPRC, and FNR per ratio.

**Why:** Balanced evaluation overstates real-world precision. At 20:1 the Transformer fires on roughly one in two alerts (Precision 51%). AUPRC — the recommended metric for imbalanced classification — degrades from 99.30% to 92.80%, confirming that deployment at realistic prevalence rates requires threshold recalibration and mandatory human-in-the-loop review.

---

### Step 8 — SHAP and LIME explainability

```bash
python explain.py \
    --dataset         ./Dataset/Suicide_Detection.csv \
    --model_dir       ./Models/distilroberta_lora_final \
    --model_type      transformer \
    --results_base    ./results \
    --seed            42 \
    --batch_size      32 \
    --max_len         128 \
    --n_shap_samples  100 \
    --n_lime_samples  10
```

> SHAP PartitionExplainer is slow. Use `--n_shap_samples 50` for a faster run, or `--model_type lr` to run SHAP on the linear baseline instead.

**What it does:** Applies SHAP (LinearExplainer for LR; PartitionExplainer for the Transformer) to produce global feature importance rankings. Applies LIME to borderline examples to show local decision logic — e.g., how "semester" and "sleep" suppress the false-positive trigger "killing" in *"this semester is killing me; i just want to sleep."*

**Why:** Clinical deployment requires transparent decision logic that practitioners can audit. SHAP confirms the Transformer's top signals are clinically meaningful crisis indicators; LIME demonstrates local disambiguation at the individual post level.

---

### Step 9 — Statistical significance (McNemar's test)

```bash
python stat_test.py \
    --dataset ./Dataset/Suicide_Detection.csv \
    --models_dir ./Models \
    --transformer_checkpoint ./Models/distilroberta_lora_final
```

**What it does:** Runs all C(7,2) = 21 pairwise McNemar's tests with Bonferroni correction (α_adj = 0.05/21 = 0.0024). Outputs contingency tables, χ² statistics, and p-values.

**Key finding:** LR (TF-IDF), LSTM, and BiLSTM are statistically indistinguishable from one another (p > 0.0024), confirming that gated sequential architectures offer no statistically meaningful advantage over well-tuned linear baselines. Every comparison crossing the Transformer boundary is significant (χ² up to 342.28, p < 1e-10).

---

### Run everything at once

```bash
bash run_all.sh                        # all defaults (epochs=3, seed=42)
bash run_all.sh --epochs 10 --seed 0   # custom settings
```

All outputs land under `./results/YYYYMMDD_HHMMSS/`.

---

## Dataset

- **Name**: Suicide and Depression Dataset (Komati, Kaggle 2023)
- **Source**: Reddit posts collected via the Pushshift API
- **Labels**: `r/SuicideWatch` posts → *suicidal*; `r/teenagers` posts → *non-suicidal*
- **Size**: 232,074 posts, balanced 1:1
- **Split**: 80% train / 20% test, stratified, seed = 42
- **Limitations**: English-language, Western-platform Reddit only. No expert clinical annotation was performed. Generalisation to other languages, platforms, or clinical populations has not been validated.

---

## Preprocessing: Dual-Track Pipeline

Different architectures impose different input requirements:

- **Track 1 (Linear models):** Aggressive lemmatisation, stop-word removal, TF-IDF / BoW vectorisation. Maximises signal in content-heavy root words.
- **Track 2 (Neural models):** Minimal cleaning — preserves syntax, punctuation, and negation markers (e.g., "NOT doing well"). The neural models learn what to attend to; removing structure harms them.

---

## Ethics and Responsible Use

This research uses publicly available Reddit data, processed in anonymised form with no attempt at re-identification.

**These models are intended exclusively as decision-support tools for qualified mental health professionals. They must not replace clinical judgment or trigger autonomous interventions.**

- At realistic deployment imbalance (10:1+), precision drops below 70% — human review is operationally mandatory, not optional.
- Automated flagging without human review risks stigmatisation and privacy violation.
- Posts from `r/SuicideWatch` were shared in a peer-support context; repurposing them for ML research raises contextual integrity concerns. Future data collection should operate under IRB guidance with appropriate opt-in mechanisms.

If you or someone you know is struggling, please contact a qualified mental health professional or a local crisis helpline.

---

## Citation

```
Bagchi, S., Pal, D.: From Frequencies to Self-Attention: Interpretable Deep Learning
for Early Suicide Risk Detection. AdComSys 2026.
```

---

## Authors

- **Subham Bagchi \*** — Dept. of Data Analytics, National College of Ireland (`sopam1998@gmail.com`)
- **Debmalya Pal \*** — Dept. of Computer Science and Engineering, UC San Diego (`d2pal@ucsd.edu`)

\* = Equal contribution.
