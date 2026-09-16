#!/usr/bin/env bash
# Shared inference-only worker for threshold 0.5 on Linux and OpenSSL.
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/scratch/congthanh_le/quan}"
REPO_DIR="${REPO_DIR:-$SERVER_ROOT/vulguard_lite}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SERVER_ROOT/output}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/output/threshold_0.5}"
LOG_ROOT="${LOG_ROOT:-$SERVER_ROOT/vulguard_logs/threshold_0.5}"
CONDA_SH="${CONDA_SH:-/app/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vulguard_lite}"
HF_REPO_ID="${HF_REPO_ID:-TheSyx/vulguard_lite}"
HF_REVISION="${HF_REVISION:-main}"
HF_OUTPUT_REPO_ID="${HF_OUTPUT_REPO_ID:-$HF_REPO_ID}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-True}"
TRAIN_MISSING_LINUX_CPU="${TRAIN_MISSING_LINUX_CPU:-True}"
SEEDS="${SEEDS:-1;2;3;4;5}"
X_VALUES="${X_VALUES:-0;1;2;3}"
Y_VALUES="${Y_VALUES:-0;1;2;3}"
CPU_THREADS="${CPU_THREADS:-1}"
TASK_STRIDE="${TASK_STRIDE:-1}"

if [[ -z "${DATASETS:-}" || -z "${MODELS:-}" || -z "${EXECUTION_KIND:-}" ]]; then
    echo "DATASETS, MODELS and EXECUTION_KIND must be set by a PBS wrapper." >&2
    exit 2
fi
if [[ ! -f "$REPO_DIR/scripts/reinfer_threshold_0_5.py" ]]; then
    echo "Inference script not found: $REPO_DIR/scripts/reinfer_threshold_0_5.py" >&2
    exit 2
fi
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda initialization script not found: $CONDA_SH" >&2
    exit 2
fi

IFS=';' read -r -a DATASET_ARRAY <<< "$DATASETS"
IFS=';' read -r -a MODEL_ARRAY <<< "$MODELS"
IFS=';' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=';' read -r -a X_ARRAY <<< "$X_VALUES"
IFS=';' read -r -a Y_ARRAY <<< "$Y_VALUES"
task_index="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-0}}"
config_count=$((${#X_ARRAY[@]} * ${#Y_ARRAY[@]}))
task_count=$((${#DATASET_ARRAY[@]} * ${#MODEL_ARRAY[@]} * config_count))
if ((task_index < 0 || task_index >= task_count)); then
    echo "Array index $task_index is outside 0-$((task_count - 1))" >&2
    exit 2
fi

dataset_index=$((task_index / (${#MODEL_ARRAY[@]} * config_count)))
remainder=$((task_index % (${#MODEL_ARRAY[@]} * config_count)))
model_index=$((remainder / config_count))
config_index=$((remainder % config_count))
x_index=$((config_index / ${#Y_ARRAY[@]}))
y_index=$((config_index % ${#Y_ARRAY[@]}))
dataset="${DATASET_ARRAY[$dataset_index]}"
model="${MODEL_ARRAY[$model_index]}"
config="${dataset}_${X_ARRAY[$x_index]}_${Y_ARRAY[$y_index]}"
case "$dataset" in linux|openssl) ;; *) echo "Invalid dataset: $dataset" >&2; exit 2 ;; esac
case "$EXECUTION_KIND:$model" in
    cpu:tlel|cpu:lapredict|cpu:lr) device="cpu" ;;
    gpu:deepjit|gpu:simcom|gpu:jitfine) device="cuda:0" ;;
    *) echo "Invalid $EXECUTION_KIND task: $dataset/$model" >&2; exit 2 ;;
esac

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/${EXECUTION_KIND}_${dataset}_${model}_${config}.log") 2>&1
echo "Started         : $(date --iso-8601=seconds)"
echo "Dataset/model   : $dataset / $model"
echo "Config          : $config"
echo "Mode            : inference-only (training is never called)"
echo "Threshold       : 0.5"
echo "Checkpoint order: local server, then Hugging Face fallback"
echo "Result root     : $RESULT_ROOT"
echo "HF output       : $HF_OUTPUT_REPO_ID/output/threshold_0.5"

# Recursive stride workers retain the activated environment. Avoid sourcing
# conda again for every task (the shared filesystem can transiently fail).
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi
project_parent="$(dirname "$REPO_DIR")"
export PYTHONPATH="$project_parent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
if [[ "$device" == "cpu" ]]; then export CUDA_VISIBLE_DEVICES=""; fi

job_id="${PBS_JOBID:-manual}"
stage_dir="${TMPDIR:-$SERVER_ROOT/tmp}/vulguard_threshold_0_5/${job_id}_${task_index}"
mkdir -p "$stage_dir"
command=(
    python "$REPO_DIR/scripts/reinfer_threshold_0_5.py"
    --repo-root "$REPO_DIR"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --result-root "$RESULT_ROOT"
    --stage-dir "$stage_dir"
    --datasets "$dataset"
    --models "$model"
    --configs "$config"
    --seeds "${SEED_ARRAY[@]}"
    --device "$device"
    --hf-repo-id "$HF_REPO_ID"
    --hf-revision "$HF_REVISION"
    --hf-output-repo-id "$HF_OUTPUT_REPO_ID"
    --skip-existing
)
case "${UPLOAD_RESULTS,,}" in
    true|1|yes) command+=(--upload-results) ;;
    false|0|no) ;;
    *) echo "UPLOAD_RESULTS must be True or False, got: $UPLOAD_RESULTS" >&2; exit 2 ;;
esac
if [[ "$dataset" == "linux" && "$EXECUTION_KIND" == "cpu" ]]; then
    case "${TRAIN_MISSING_LINUX_CPU,,}" in
        true|1|yes) command+=(--train-missing-linux-cpu) ;;
        false|0|no) ;;
        *) echo "TRAIN_MISSING_LINUX_CPU must be True or False" >&2; exit 2 ;;
    esac
fi
printf 'Command:'; printf ' %q' "${command[@]}"; printf '\n'
"${command[@]}"
echo "Completed task  : $task_index at $(date --iso-8601=seconds)"

# Keep the submitted PBS array small enough for per-user resource validation.
# Each live worker consumes the remaining matrix in a deterministic stride.
next_task=$((task_index + TASK_STRIDE))
if ((next_task < task_count)); then
    export PBS_ARRAY_INDEX="$next_task"
    exec bash "$REPO_DIR/scripts/reinfer_openssl_latest_sutd_worker.sh"
fi
echo "Completed worker: $(date --iso-8601=seconds)"
