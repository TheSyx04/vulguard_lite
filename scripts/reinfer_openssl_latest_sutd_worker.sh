#!/usr/bin/env bash
# Shared worker for the CPU and GPU OpenSSL reinference PBS arrays.

set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/scratch/congthanh_le/quan}"
REPO_DIR="${REPO_DIR:-$SERVER_ROOT/vulguard_lite}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SERVER_ROOT/output}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/output/openssl_reinfer}"
LOG_ROOT="${LOG_ROOT:-$SERVER_ROOT/vulguard_logs/openssl/reinfer_latest}"
CONDA_SH="${CONDA_SH:-/app/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vulguard_lite}"
HF_REPO_ID="${HF_REPO_ID:-TheSyx/vulguard_lite}"
HF_REVISION="${HF_REVISION:-main}"
HF_OUTPUT_REPO_ID="${HF_OUTPUT_REPO_ID:-$HF_REPO_ID}"
X_VALUES="${X_VALUES:-0;1;2;3}"
Y_VALUES="${Y_VALUES:-1;2;3}"
SEEDS="${SEEDS:-1;2;3;4;5}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-True}"
CPU_THREADS="${CPU_THREADS:-1}"

if [[ -z "${TASKS:-}" && -z "${MODELS:-}" ]] || [[ -z "${EXECUTION_KIND:-}" ]]; then
    echo "TASKS (or MODELS) and EXECUTION_KIND must be set by a PBS wrapper." >&2
    exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required; submit the PBS file with qsub -v HF_TOKEN." >&2
    exit 2
fi
if [[ ! -d "$REPO_DIR" || ! -f "$REPO_DIR/scripts/reinfer_openssl_latest.py" ]]; then
    echo "Repository/script not found under REPO_DIR=$REPO_DIR" >&2
    exit 2
fi
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda initialization script not found: $CONDA_SH" >&2
    exit 2
fi

IFS=';' read -r -a SEED_ARRAY <<< "$SEEDS"

task_index="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-0}}"
if [[ -n "${TASKS:-}" ]]; then
    IFS=';' read -r -a TASK_ARRAY <<< "$TASKS"
    task_count=${#TASK_ARRAY[@]}
else
    IFS=';' read -r -a MODEL_ARRAY <<< "$MODELS"
    IFS=';' read -r -a X_ARRAY <<< "$X_VALUES"
    IFS=';' read -r -a Y_ARRAY <<< "$Y_VALUES"
    task_count=$((${#MODEL_ARRAY[@]} * ${#X_ARRAY[@]}))
fi
if ((task_index < 0 || task_index >= task_count)); then
    echo "Array index $task_index is outside the configured matrix 0-$((task_count - 1))"
    exit 0
fi

if [[ -n "${TASKS:-}" ]]; then
    task="${TASK_ARRAY[$task_index]}"
    model="${task%%:*}"
    config="${task#*:}"
    if [[ "$model" == "$task" || ! "$config" =~ ^openssl_[0-3]_[1-3]$ ]]; then
        echo "Invalid TASKS entry: $task (expected model:openssl_X_Y)" >&2
        exit 2
    fi
    configs=("$config")
else
    model_index=$((task_index / ${#X_ARRAY[@]}))
    x_index=$((task_index % ${#X_ARRAY[@]}))
    model="${MODEL_ARRAY[$model_index]}"
    x="${X_ARRAY[$x_index]}"
    configs=()
    for y in "${Y_ARRAY[@]}"; do
        configs+=("openssl_${x}_${y}")
    done
fi

case "$EXECUTION_KIND:$model" in
    cpu:tlel|cpu:lapredict|cpu:lr) device="cpu" ;;
    gpu:deepjit|gpu:simcom|gpu:jitfine) device="cuda:0" ;;
    *) echo "Model $model is invalid for the $EXECUTION_KIND array" >&2; exit 2 ;;
esac

mkdir -p "$LOG_ROOT"
config_label="${configs[0]}"
log_file="$LOG_ROOT/${EXECUTION_KIND}_${model}_${config_label}.log"
exec > >(tee -a "$log_file") 2>&1

echo "Started       : $(date --iso-8601=seconds)"
echo "Host          : $(hostname)"
echo "Execution     : $EXECUTION_KIND"
echo "Array task    : $task_index / $((task_count - 1))"
echo "Model         : $model"
echo "Configs       : ${configs[*]}"
echo "Seeds         : ${SEED_ARRAY[*]}"
echo "CPU threads   : $CPU_THREADS"
if [[ "$EXECUTION_KIND" == "cpu" ]]; then
    echo "Mode          : retrain + evaluate + test; checkpoint retained per run"
else
    echo "Mode          : inference-only from verified Hugging Face checkpoint"
fi
echo "Source root   : $OUTPUT_ROOT"
echo "Result root   : $RESULT_ROOT"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

project_parent="$(dirname "$REPO_DIR")"
export PYTHONPATH="$project_parent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export VULGUARD_HF_FORCE_DOWNLOAD="1"
if [[ "$device" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

job_id="${PBS_JOBID:-manual}"
stage_dir="${TMPDIR:-$SERVER_ROOT/tmp}/vulguard_openssl_reinfer/${job_id}_${task_index}"
mkdir -p "$stage_dir"

if [[ "$EXECUTION_KIND" == "cpu" ]]; then
    case "${UPLOAD_RESULTS,,}" in
        true|1|yes) upload_results="True" ;;
        false|0|no) upload_results="False" ;;
        *) echo "UPLOAD_RESULTS must be True or False, got: $UPLOAD_RESULTS" >&2; exit 2 ;;
    esac

    for config in "${configs[@]}"; do
        save_folder="$RESULT_ROOT/$model/$config"
        experiment_name="${model}_openssl_${config}_sampling"
        remote_folder="output/openssl_reinfer/$model/sampling/$experiment_name"
        mkdir -p "$save_folder"
        command=(
            python -m vulguard_lite experiment
            -repo_name openssl
            -repo_language C
            -model "$model"
            -device cpu
            -dg_save_folder "$save_folder"
            -hf_repo_id "$HF_REPO_ID"
            -hf_revision "$HF_REVISION"
            -hf_split_path "dataset/openssl/$config"
            -hf_upload_result "$upload_results"
            -hf_output_repo_id "$HF_OUTPUT_REPO_ID"
            -hf_output_folder "$remote_folder"
            -runs 3
            -epochs 30
            -sampling True
            -sampling_seeds "${SEED_ARRAY[@]}"
            -budget 0.05 0.075 0.1 0.15 0.2
            -calibration_range 0 1 10001
            -resume_from_checkpoint False
        )
        printf 'Command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        "${command[@]}"
    done
    echo "Completed     : $(date --iso-8601=seconds)"
    exit 0
fi

command=(
    python "$REPO_DIR/scripts/reinfer_openssl_latest.py"
    --repo-root "$REPO_DIR"
    --output-root "$OUTPUT_ROOT"
    --result-root "$RESULT_ROOT"
    --stage-dir "$stage_dir"
    --models "$model"
    --configs "${configs[@]}"
    --seeds "${SEED_ARRAY[@]}"
    --device "$device"
    --hf-repo-id "$HF_REPO_ID"
    --hf-revision "$HF_REVISION"
    --hf-output-repo-id "$HF_OUTPUT_REPO_ID"
)
case "${UPLOAD_RESULTS,,}" in
    true|1|yes) command+=(--upload-results) ;;
    false|0|no) ;;
    *) echo "UPLOAD_RESULTS must be True or False, got: $UPLOAD_RESULTS" >&2; exit 2 ;;
esac

printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"

echo "Completed     : $(date --iso-8601=seconds)"
