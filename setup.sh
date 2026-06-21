#!/usr/bin/env bash
# =============================================================================
# setup.sh
# --------
# One-shot environment bootstrap for the AdComSys 2026 experiment suite.
#
# What it does
# ------------
#   1. Checks Python 3.10+
#   2. Creates a virtual environment (./venv)
#   3. Installs all dependencies via requirements.txt
#   4. Creates project directories (Dataset/, Models/, Plots/, results/)
#   5. Downloads Suicide_Detection.csv from Kaggle
#
# Kaggle credentials
# ------------------
# The download requires a Kaggle API token at ~/.kaggle/kaggle.json.
# To get one:
#   1. Log in at https://www.kaggle.com → Account → Create New API Token
#   2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json
#   3. chmod 600 ~/.kaggle/kaggle.json
#
# Usage
# -----
#   bash setup.sh
#
# After setup, activate the environment and run experiments:
#   source venv/bin/activate
#   bash run_all.sh
# =============================================================================
set -euo pipefail

VENV_DIR="./venv"
DATASET_DIR="./Dataset"
DATASET_FILE="${DATASET_DIR}/Suicide_Detection.csv"
KAGGLE_DATASET="nikhileswarkomati/suicide-watch"

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }

echo "======================================================================"
echo "  AdComSys 2026 — Environment Setup"
echo "======================================================================"

# ── Load .env if present (exports KAGGLE_USERNAME and KAGGLE_KEY) ─────────────
if [[ -f ".env" ]]; then
    info "Loading credentials from .env …"
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

# ── 1. Python version check ───────────────────────────────────────────────────
info "Checking Python version …"
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
if [[ -z "$PYTHON" ]]; then
    error "Python not found. Install Python 3.10+ and retry."
fi
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
    error "Python 3.10+ required (found ${PY_VERSION}). Install it and retry."
fi
info "Python ${PY_VERSION} ✓"

# ── 2. Virtual environment ────────────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at ${VENV_DIR}. Skipping creation."
else
    info "Creating virtual environment at ${VENV_DIR} …"
    "$PYTHON" -m venv "$VENV_DIR"
    info "Virtual environment created ✓"
fi

VENV_PIP="${VENV_DIR}/bin/pip"

# ── 3. Install dependencies ───────────────────────────────────────────────────
[[ -f "requirements.txt" ]] || error "requirements.txt not found. Are you in the project root?"

info "Upgrading pip …"
"$VENV_PIP" install --upgrade pip

info "Installing dependencies from requirements.txt …"
"$VENV_PIP" install -r requirements.txt
info "Dependencies installed ✓"

# ── 4. Project directories ────────────────────────────────────────────────────
info "Creating project directories …"
mkdir -p Dataset Models Plots results
info "Directories ready ✓"

# ── 5. Kaggle dataset download ────────────────────────────────────────────────
if [[ -f "$DATASET_FILE" ]]; then
    info "Dataset already present at ${DATASET_FILE}. Skipping download."
else
    info "Downloading dataset from Kaggle …"

    # Check for kaggle credentials
    KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
    if [[ ! -f "$KAGGLE_JSON" ]]; then
        # Also accept credentials via environment variables
        if [[ -z "${KAGGLE_USERNAME:-}" ]] || [[ -z "${KAGGLE_KEY:-}" ]]; then
            echo ""
            warn "Kaggle credentials not found. Please do ONE of the following:"
            echo "  Option A — API token file:"
            echo "    1. Go to https://www.kaggle.com → Account → Create New API Token"
            echo "    2. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json"
            echo "    3. chmod 600 ~/.kaggle/kaggle.json"
            echo "    4. Re-run: bash setup.sh"
            echo ""
            echo "  Option B — environment variables:"
            echo "    export KAGGLE_USERNAME=your_username"
            echo "    export KAGGLE_KEY=your_api_key"
            echo "    bash setup.sh"
            echo ""
            echo "  Option C — manual download:"
            echo "    Download from: https://www.kaggle.com/datasets/${KAGGLE_DATASET}/data"
            echo "    Place Suicide_Detection.csv in: ${DATASET_DIR}/"
            echo ""
            warn "Skipping dataset download. Set up credentials and re-run, or place"
            warn "Suicide_Detection.csv in ${DATASET_DIR}/ manually."
        else
            info "Using KAGGLE_USERNAME / KAGGLE_KEY environment variables."
            mkdir -p "${HOME}/.kaggle"
            echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" \
                > "$KAGGLE_JSON"
            chmod 600 "$KAGGLE_JSON"
        fi
    fi

    # Attempt download if credentials are now available
    if [[ -f "$KAGGLE_JSON" ]] || { [[ -n "${KAGGLE_USERNAME:-}" ]] && [[ -n "${KAGGLE_KEY:-}" ]]; }; then
        # Ensure kaggle CLI is available inside the venv
        if ! "${VENV_DIR}/bin/kaggle" --version &>/dev/null 2>&1; then
            info "Installing kaggle CLI …"
            "$VENV_PIP" install --quiet kaggle
        fi

        info "Downloading ${KAGGLE_DATASET} …"
        "${VENV_DIR}/bin/kaggle" datasets download \
            -d "$KAGGLE_DATASET" \
            --path "$DATASET_DIR" \
            --unzip

        # The zip may extract to a subdirectory — find and move the CSV
        CSV_FOUND=$(find "$DATASET_DIR" -name "Suicide_Detection.csv" | head -1)
        if [[ -n "$CSV_FOUND" ]] && [[ "$CSV_FOUND" != "$DATASET_FILE" ]]; then
            mv "$CSV_FOUND" "$DATASET_FILE"
        fi

        if [[ -f "$DATASET_FILE" ]]; then
            info "Dataset downloaded → ${DATASET_FILE} ✓"
        else
            warn "Download completed but Suicide_Detection.csv not found."
            warn "Check ${DATASET_DIR}/ and rename the file if needed."
        fi
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "    1. Activate the environment:"
echo "         source ${VENV_DIR}/bin/activate"
echo ""
echo "    2. Run all experiments (trains all models + evaluations):"
echo "         bash run_all.sh"
echo ""
echo "    Note: The DistilRoBERTa+LoRA checkpoint is trained during"
echo "    run_all.sh (lora_sweep.py). No notebook step is required."
echo "======================================================================"
