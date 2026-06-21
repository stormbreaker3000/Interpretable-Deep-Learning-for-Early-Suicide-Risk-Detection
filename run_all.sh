#!/usr/bin/env bash
# =============================================================================
# run_all.sh
# ----------
# Full experiment pipeline for the AdComSys 2026 camera-ready revision.
#
# Pipeline
# --------
#   Step 1  train_lr.py            Train LR (BoW + TF-IDF)  [+ 5-fold CV]
#   Step 2  train_rnn.py           Train Simple RNN
#   Step 3  train_lstm.py          Train LSTM
#   Step 4  train_bilstm.py        Train BiLSTM + Attention BiLSTM
#   Step 5  evaluate_checkpoints   Evaluate all saved checkpoints on test set
#   Step 6  lora_sweep.py          LoRA rank × alpha sweep + full fine-tuning
#   Step 6b (inline)               Promote paper-selected LoRA config (r8_a16)
#                                  to Models/distilroberta_lora_final/
#   Step 7  imbalance_eval.py      Class-imbalance robustness evaluation
#   Step 8  explain.py             SHAP + LIME explainability (best model)
#   Step 9  stat_test.py           Pairwise McNemar's statistical significance
#
# All outputs land under ./results/{TIMESTAMP}/.
# Training steps always retrain from scratch (--force_retrain).
# To skip retraining (use existing models), remove the --force_retrain flags.
#
# Usage
# -----
#   bash run_all.sh                          # all defaults
#   bash run_all.sh --epochs 5               # override epochs for all scripts
#   bash run_all.sh --seed 0                 # different random seed
#   bash run_all.sh --model_type bilstm \
#                   --model_dir ./Models     # BiLSTM for imbalance eval
#   bash run_all.sh --n_shap_samples 50      # fewer SHAP samples (faster)
#
# Prerequisites
# -------------
#   bash setup.sh && source venv/bin/activate
#   pip install shap lime                    # required for Step 8
# =============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
DATASET="./Dataset/Suicide_Detection.csv"
MODELS_DIR="./Models"
CHECKPOINTS_DIR="./Models/lora_sweep"
RESULTS_BASE="./results"
SEED=42
BATCH_SIZE=128
MAX_LEN=128                  # sequence length for all models
EPOCHS=3
TRANSFORMER_CKPT="./Models/distilroberta_lora_final"
MODEL_TYPE="transformer"     # model used for imbalance_eval
RANKS="2 4 8 16 32"
ALPHA_MULTIPLIERS="1.0 2.0 4.0"   # α = 1r, 2r, 4r
ALPHA_FIXED=16                     # additional fixed α=16 at every rank
BEST_LORA_CONFIG="r8_a16"          # paper-selected LoRA config → promoted to TRANSFORMER_CKPT
EXPLAIN_MODEL_TYPE="transformer"   # model used for Step 8 explainability
N_SHAP_SAMPLES=100                 # SHAP global importance samples
N_LIME_SAMPLES=10                  # LIME per-sample explanations

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)                DATASET="$2";           shift 2 ;;
        --models_dir)             MODELS_DIR="$2";        shift 2 ;;
        --checkpoints_dir)        CHECKPOINTS_DIR="$2";   shift 2 ;;
        --results_base)           RESULTS_BASE="$2";      shift 2 ;;
        --seed)                   SEED="$2";              shift 2 ;;
        --batch_size)             BATCH_SIZE="$2";        shift 2 ;;
        --epochs)                 EPOCHS="$2";            shift 2 ;;
        --max_len)                MAX_LEN="$2";           shift 2 ;;
        --transformer_checkpoint) TRANSFORMER_CKPT="$2";  shift 2 ;;
        --model_type)             MODEL_TYPE="$2";         shift 2 ;;
        --ranks)                  RANKS="$2";              shift 2 ;;
        --alpha_multipliers)      ALPHA_MULTIPLIERS="$2";  shift 2 ;;
        --alpha_fixed)            ALPHA_FIXED="$2";        shift 2 ;;
        --best_lora_config)       BEST_LORA_CONFIG="$2";  shift 2 ;;
        --explain_model_type)     EXPLAIN_MODEL_TYPE="$2"; shift 2 ;;
        --n_shap_samples)         N_SHAP_SAMPLES="$2";     shift 2 ;;
        --n_lime_samples)         N_LIME_SAMPLES="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,38p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1  (run with --help for usage)"
            exit 1
            ;;
    esac
done

# ── Shared timestamp ──────────────────────────────────────────────────────────
TIMESTAMP=$(python3 -c \
    "from datetime import datetime; print(datetime.now().strftime('%Y%m%d_%H%M%S'))")

# Export seed so Python processes inherit it from the environment.
# os.environ inside Python only propagates to child processes, not the calling
# shell, so the canonical place to set PYTHONHASHSEED is here.
export PYTHONHASHSEED="${SEED}"

echo "======================================================================"
echo "  AdComSys 2026 — Full Experiment Pipeline"
echo "  Timestamp : ${TIMESTAMP}"
echo "  Results   : ${RESULTS_BASE}/${TIMESTAMP}/"
echo "======================================================================"

# ── Step 1: Train Logistic Regression ────────────────────────────────────────
echo ""
echo "▶ [1 / 9]  Train Logistic Regression (BoW + TF-IDF)"
echo "----------------------------------------------------------------------"
python3 train_lr.py \
    --dataset       "${DATASET}" \
    --models_dir    "${MODELS_DIR}" \
    --results_base  "${RESULTS_BASE}" \
    --timestamp     "${TIMESTAMP}" \
    --seed          "${SEED}" \
    --force_retrain

# ── Step 2: Train Simple RNN ──────────────────────────────────────────────────
echo ""
echo "▶ [2 / 9]  Train Simple RNN"
echo "----------------------------------------------------------------------"
python3 train_rnn.py \
    --dataset       "${DATASET}" \
    --models_dir    "${MODELS_DIR}" \
    --results_base  "${RESULTS_BASE}" \
    --timestamp     "${TIMESTAMP}" \
    --seed          "${SEED}" \
    --max_len       "${MAX_LEN}" \
    --epochs        "${EPOCHS}" \
    --batch_size    "${BATCH_SIZE}" \
    --force_retrain

# ── Step 3: Train LSTM ────────────────────────────────────────────────────────
echo ""
echo "▶ [3 / 9]  Train LSTM"
echo "----------------------------------------------------------------------"
python3 train_lstm.py \
    --dataset       "${DATASET}" \
    --models_dir    "${MODELS_DIR}" \
    --results_base  "${RESULTS_BASE}" \
    --timestamp     "${TIMESTAMP}" \
    --seed          "${SEED}" \
    --max_len       "${MAX_LEN}" \
    --epochs        "${EPOCHS}" \
    --batch_size    "${BATCH_SIZE}" \
    --force_retrain

# ── Step 4: Train BiLSTM + Attention BiLSTM ───────────────────────────────────
echo ""
echo "▶ [4 / 9]  Train BiLSTM + Attention BiLSTM"
echo "----------------------------------------------------------------------"
python3 train_bilstm.py \
    --dataset       "${DATASET}" \
    --models_dir    "${MODELS_DIR}" \
    --results_base  "${RESULTS_BASE}" \
    --timestamp     "${TIMESTAMP}" \
    --seed          "${SEED}" \
    --max_len       "${MAX_LEN}" \
    --epochs        "${EPOCHS}" \
    --batch_size    "${BATCH_SIZE}" \
    --force_retrain

# ── Step 5: Evaluate all trained checkpoints ─────────────────────────────────
echo ""
echo "▶ [5 / 9]  Checkpoint Evaluation"
echo "----------------------------------------------------------------------"
[[ -f "evaluate_checkpoints.py" ]] || { echo "[ERROR] evaluate_checkpoints.py not found in project root. Ensure all scripts are present and retry."; exit 1; }
python3 evaluate_checkpoints.py \
    --dataset                 "${DATASET}" \
    --models_dir              "${MODELS_DIR}" \
    --results_base            "${RESULTS_BASE}" \
    --timestamp               "${TIMESTAMP}" \
    --seed                    "${SEED}" \
    --batch_size              "${BATCH_SIZE}" \
    --max_len                 "${MAX_LEN}" \
    --transformer_max_len     "${MAX_LEN}" \
    --transformer_checkpoint  "${TRANSFORMER_CKPT}"

# ── Step 6: LoRA rank × alpha sweep + full fine-tuning ───────────────────────
echo ""
echo "▶ [6 / 9]  LoRA Rank × Alpha Sweep  (+  Full Fine-Tuning)"
echo "----------------------------------------------------------------------"
[[ -f "lora_sweep.py" ]] || { echo "[ERROR] lora_sweep.py not found in project root. Ensure all scripts are present and retry."; exit 1; }
# shellcheck disable=SC2086
python3 lora_sweep.py \
    --dataset           "${DATASET}" \
    --base_model        "distilroberta-base" \
    --checkpoints_dir   "${CHECKPOINTS_DIR}" \
    --results_base      "${RESULTS_BASE}" \
    --timestamp         "${TIMESTAMP}" \
    --seed              "${SEED}" \
    --batch_size        "${BATCH_SIZE}" \
    --max_len           "${MAX_LEN}" \
    --epochs            "${EPOCHS}" \
    --ranks             ${RANKS} \
    --alpha_multipliers ${ALPHA_MULTIPLIERS} \
    --alpha_fixed       "${ALPHA_FIXED}"

# ── Step 6b: Promote paper-selected LoRA config → distilroberta_lora_final ───
echo ""
echo "▶ [6b/ 9]  Promote ${BEST_LORA_CONFIG} → ${TRANSFORMER_CKPT}"
echo "----------------------------------------------------------------------"
_LORA_SRC="${CHECKPOINTS_DIR}/${BEST_LORA_CONFIG}"
if [[ ! -d "${_LORA_SRC}" ]]; then
    echo "[ERROR] LoRA checkpoint not found: ${_LORA_SRC}"
    echo "        Run Step 6 (lora_sweep.py) first, or set --best_lora_config."
    exit 1
fi
rm -rf "${TRANSFORMER_CKPT}"
cp -r  "${_LORA_SRC}" "${TRANSFORMER_CKPT}"
echo "  Copied ${_LORA_SRC} → ${TRANSFORMER_CKPT}"

# ── Step 7: Imbalanced-split evaluation ──────────────────────────────────────
echo ""
echo "▶ [7 / 9]  Class-Imbalance Evaluation"
echo "----------------------------------------------------------------------"
python3 imbalance_eval.py \
    --dataset       "${DATASET}" \
    --model_dir     "${TRANSFORMER_CKPT}" \
    --model_type    "${MODEL_TYPE}" \
    --results_base  "${RESULTS_BASE}" \
    --timestamp     "${TIMESTAMP}" \
    --seed          "${SEED}" \
    --batch_size    "${BATCH_SIZE}" \
    --max_len       "${MAX_LEN}" \
    --ratios        1 5 10 20

# ── Step 8: SHAP + LIME explainability ───────────────────────────────────────
echo ""
echo "▶ [8 / 9]  SHAP + LIME Explainability"
echo "----------------------------------------------------------------------"
[[ -f "explain.py" ]] || { echo "[ERROR] explain.py not found in project root. Ensure all scripts are present and retry."; exit 1; }
python3 explain.py \
    --dataset          "${DATASET}" \
    --model_dir        "${TRANSFORMER_CKPT}" \
    --model_type       "${EXPLAIN_MODEL_TYPE}" \
    --results_base     "${RESULTS_BASE}" \
    --timestamp        "${TIMESTAMP}" \
    --seed             "${SEED}" \
    --batch_size       32 \
    --max_len          "${MAX_LEN}" \
    --n_shap_samples   "${N_SHAP_SAMPLES}" \
    --n_lime_samples   "${N_LIME_SAMPLES}"

# ── Step 9: McNemar's pairwise significance test ──────────────────────────────
echo ""
echo "▶ [9 / 9]  McNemar's Pairwise Statistical Significance Test"
echo "----------------------------------------------------------------------"
python3 stat_test.py \
    --dataset                  "${DATASET}" \
    --models_dir               "${MODELS_DIR}" \
    --transformer_checkpoint   "${TRANSFORMER_CKPT}" \
    --results_base             "${RESULTS_BASE}" \
    --timestamp                "${TIMESTAMP}" \
    --seed                     "${SEED}" \
    --batch_size               "${BATCH_SIZE}" \
    --max_len                  "${MAX_LEN}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  All experiments complete."
echo "  Results → ${RESULTS_BASE}/${TIMESTAMP}/"
echo ""
echo "  Layout:"
echo "    train_lr/             LR (BoW + TF-IDF) metrics + 5-fold CV"
echo "    train_rnn/            Simple RNN metrics"
echo "    train_lstm/           LSTM metrics"
echo "    train_bilstm/         BiLSTM + Attention BiLSTM metrics + loss plot"
echo "    evaluate_checkpoints/ ROC, PR, calibration curves + summary.json"
echo "    lora_sweep/           Per-config metadata + sweep_summary.png"
echo "    imbalance_eval/       Metrics vs imbalance ratio plots"
echo "    explain/              SHAP global importance + LIME explanations"
echo "    stat_test/            McNemar pairwise p-values + contingency tables"
echo "======================================================================"
