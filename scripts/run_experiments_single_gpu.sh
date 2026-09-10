#!/usr/bin/env bash
# Run VulGuard Lite experiment configurations sequentially on one GPU.
# Intended to be started inside tmux; no scheduler or job array is required.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_PARENT="$(dirname -- "$REPO_DIR")"

DATASET="linux"
MODEL="jitfine"
LANGUAGE="C"
DEVICE="cuda"
GPU="0"
X_VALUES="0 1 2 3"
Y_VALUES="0 1 2 3"
SPLITS=""

OUTPUT_ROOT="$PROJECT_PARENT/output"
LOG_ROOT="$PROJECT_PARENT/vulguard_logs"
HF_REPO_ID="TheSyx/vulguard_lite"
HF_REVISION="main"
HF_SPLIT_PREFIX="dataset"
HF_UPLOAD_RESULT="False"
HF_UPLOAD_CHECKPOINT_ONLY="False"
HF_OUTPUT_REPO_ID=""

RUNS="3"
EPOCHS="30"
SAMPLING="True"
SAMPLING_SEEDS="1 2 3 4 5"
BUDGETS="0.05 0.075 0.1 0.15 0.2"
CALIBRATION_RANGE="0 1 10001"
RESUME="True"
DRY_RUN="False"

usage() {
    cat <<'EOF'
Usage: scripts/run_experiments_single_gpu.sh [options]

Runs configurations sequentially, which is suitable for a server with one GPU.
Start it inside tmux so it continues after disconnecting.

Configuration:
  --dataset NAME               Dataset/repository name (default: linux)
  --model NAME                 Model name (default: jitfine)
  --language LANGUAGE          Repository language (default: C)
  --device DEVICE              PyTorch device: cuda, cuda:0, or cpu (default: cuda)
  --gpu ID                     CUDA_VISIBLE_DEVICES value (default: 0)
  --x-values "LIST"            First split indices (default: "0 1 2 3")
  --y-values "LIST"            Second split indices (default: "0 1 2 3")
  --splits "LIST"              Explicit split names; overrides x/y generation

Experiment:
  --runs N                     Runs per sampling seed (default: 3)
  --epochs N                   Training epochs (default: 30)
  --sampling BOOL              True or False (default: True)
  --sampling-seeds "LIST"      Sampling seeds (default: "1 2 3 4 5")
  --budgets "LIST"             Calibration budgets
  --calibration-range "S E N" Threshold start, end, and steps
  --resume BOOL                Resume/skip existing work (default: True)

Paths and Hugging Face:
  --output-root PATH           Output base directory (default: <repo-parent>/output)
  --log-root PATH              Log base directory (default: <repo-parent>/vulguard_logs)
  --hf-repo-id ID             Input dataset repository
  --hf-revision REV            Input dataset revision (default: main)
  --hf-split-prefix PATH       Prefix before dataset/split (default: dataset)
  --upload-results BOOL        Upload completed experiment folders (default: False)
  --upload-checkpoints-only BOOL
                               Upload one run-1 checkpoint per seed and no results
  --hf-output-repo-id ID       Optional separate output dataset repository

Other:
  --dry-run                    Print commands without running experiments
  -h, --help                   Show this help

Examples:
  # All 16 Linux/JITFine configurations using the defaults:
  scripts/run_experiments_single_gpu.sh

  # One checkpoint for each explicit OpenSSL split:
  scripts/run_experiments_single_gpu.sh \
    --dataset openssl --model jitfine --splits "openssl_0_1 openssl_1_2" \
    --runs 1 --sampling-seeds "1"

  # Run a CPU model:
  scripts/run_experiments_single_gpu.sh --model lapredict --device cpu
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) require_value "$@"; DATASET="$2"; shift 2 ;;
        --model) require_value "$@"; MODEL="$2"; shift 2 ;;
        --language) require_value "$@"; LANGUAGE="$2"; shift 2 ;;
        --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
        --gpu) require_value "$@"; GPU="$2"; shift 2 ;;
        --x-values) require_value "$@"; X_VALUES="$2"; shift 2 ;;
        --y-values) require_value "$@"; Y_VALUES="$2"; shift 2 ;;
        --splits) require_value "$@"; SPLITS="$2"; shift 2 ;;
        --runs) require_value "$@"; RUNS="$2"; shift 2 ;;
        --epochs) require_value "$@"; EPOCHS="$2"; shift 2 ;;
        --sampling) require_value "$@"; SAMPLING="$2"; shift 2 ;;
        --sampling-seeds) require_value "$@"; SAMPLING_SEEDS="$2"; shift 2 ;;
        --budgets) require_value "$@"; BUDGETS="$2"; shift 2 ;;
        --calibration-range) require_value "$@"; CALIBRATION_RANGE="$2"; shift 2 ;;
        --resume) require_value "$@"; RESUME="$2"; shift 2 ;;
        --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
        --log-root) require_value "$@"; LOG_ROOT="$2"; shift 2 ;;
        --hf-repo-id) require_value "$@"; HF_REPO_ID="$2"; shift 2 ;;
        --hf-revision) require_value "$@"; HF_REVISION="$2"; shift 2 ;;
        --hf-split-prefix) require_value "$@"; HF_SPLIT_PREFIX="$2"; shift 2 ;;
        --upload-results) require_value "$@"; HF_UPLOAD_RESULT="$2"; shift 2 ;;
        --upload-checkpoints-only) require_value "$@"; HF_UPLOAD_CHECKPOINT_ONLY="$2"; shift 2 ;;
        --hf-output-repo-id) require_value "$@"; HF_OUTPUT_REPO_ID="$2"; shift 2 ;;
        --dry-run) DRY_RUN="True"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$DEVICE" == cuda* ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

read -r -a X_ARRAY <<< "$X_VALUES"
read -r -a Y_ARRAY <<< "$Y_VALUES"
read -r -a SEED_ARRAY <<< "$SAMPLING_SEEDS"
read -r -a BUDGET_ARRAY <<< "$BUDGETS"
read -r -a CALIBRATION_ARRAY <<< "$CALIBRATION_RANGE"

if [[ ${#CALIBRATION_ARRAY[@]} -ne 3 ]]; then
    echo "--calibration-range requires exactly three values" >&2
    exit 2
fi

if [[ -n "$SPLITS" ]]; then
    read -r -a SPLIT_ARRAY <<< "$SPLITS"
else
    SPLIT_ARRAY=()
    for x in "${X_ARRAY[@]}"; do
        for y in "${Y_ARRAY[@]}"; do
            SPLIT_ARRAY+=("${DATASET}_${x}_${y}")
        done
    done
fi

if [[ ${#SPLIT_ARRAY[@]} -eq 0 ]]; then
    echo "No splits were selected" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT/$DATASET/$MODEL"
cd "$PROJECT_PARENT"

echo "Repository       : $REPO_DIR"
echo "Dataset/model    : $DATASET / $MODEL"
echo "Device/GPU       : $DEVICE / ${GPU:-n/a}"
echo "Splits           : ${SPLIT_ARRAY[*]}"
echo "Runs/seeds       : $RUNS / ${SEED_ARRAY[*]}"
echo "Output root      : $OUTPUT_ROOT"
echo "Log directory    : $LOG_ROOT/$DATASET/$MODEL"

FAILED_SPLITS=()

for split in "${SPLIT_ARRAY[@]}"; do
    output_dir="$OUTPUT_ROOT/$DATASET/$MODEL/$split"
    split_log="$LOG_ROOT/$DATASET/$MODEL/${split}.log"
    hf_split_path="$HF_SPLIT_PREFIX/$DATASET/$split"

    mkdir -p "$output_dir"

    command=(
        python -m vulguard_lite experiment
        -repo_name "$DATASET"
        -repo_language "$LANGUAGE"
        -model "$MODEL"
        -device "$DEVICE"
        -dg_save_folder "$output_dir"
        -hf_repo_id "$HF_REPO_ID"
        -hf_revision "$HF_REVISION"
        -hf_split_path "$hf_split_path"
        -runs "$RUNS"
        -epochs "$EPOCHS"
        -sampling "$SAMPLING"
        -sampling_seeds "${SEED_ARRAY[@]}"
        -budget "${BUDGET_ARRAY[@]}"
        -calibration_range "${CALIBRATION_ARRAY[@]}"
        -resume_from_checkpoint "$RESUME"
        -hf_upload_result "$HF_UPLOAD_RESULT"
        -hf_upload_checkpoint_only "$HF_UPLOAD_CHECKPOINT_ONLY"
    )

    if [[ -n "$HF_OUTPUT_REPO_ID" ]]; then
        command+=(-hf_output_repo_id "$HF_OUTPUT_REPO_ID")
    fi

    {
        echo "============================================================"
        echo "Starting $split at $(date)"
        echo "Host: $(hostname)"
        printf 'Command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        echo "============================================================"
    } | tee -a "$split_log"

    if [[ "$DRY_RUN" == "True" ]]; then
        continue
    fi

    if "${command[@]}" 2>&1 | tee -a "$split_log"; then
        echo "Completed $split at $(date)" | tee -a "$split_log"
    else
        status=$?
        FAILED_SPLITS+=("$split")
        echo "Failed $split with status $status at $(date); continuing." | tee -a "$split_log"
    fi
done

if [[ ${#FAILED_SPLITS[@]} -gt 0 ]]; then
    echo "Failed splits: ${FAILED_SPLITS[*]}" >&2
    echo "Rerun the same command to resume them."
    exit 1
fi

echo "All selected configurations completed at $(date)."
