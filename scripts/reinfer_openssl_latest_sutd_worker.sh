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
SEEDS="${SEEDS:-1;2;3;4;5}"
X_VALUES="${X_VALUES:-0;1;2;3}"
Y_VALUES="${Y_VALUES:-0;1;2;3}"
CPU_THREADS="${CPU_THREADS:-1}"

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

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
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
)
printf 'Command:'; printf ' %q' "${command[@]}"; printf '\n'
"${command[@]}"
echo "Completed       : $(date --iso-8601=seconds)"
