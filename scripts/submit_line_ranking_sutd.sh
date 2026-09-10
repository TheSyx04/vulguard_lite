#!/usr/bin/env bash
# Submit a configurable ground-truth line-ranking matrix to sutd_mega PBS.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PBS_SCRIPT="$SCRIPT_DIR/run_line_ranking_sutd.pbs"

DATASET=""
MODEL=""
X_VALUES=""
Y_VALUES=""
SEEDS="1 2 3 4 5"
RESUME="True"
HF_REPO_ID="TheSyx/vulguard_lite"
HF_OUTPUT_REPO_ID=""
UPLOAD_RESULTS="True"
MAX_CONCURRENT="4"
DRY_RUN="False"

usage() {
    cat <<'EOF'
Usage: scripts/submit_line_ranking_sutd.sh [options]

Required:
  --dataset NAME          Dataset/repository name, for example openssl or linux
  --model NAME            One model: jitfine, deepjit, or simcom
  --x-values "LIST"       First config indices, for example "0 1 2 3"
  --y-values "LIST"       Second config indices, for example "0 1 2 3"

Optional:
  --seeds "LIST"          Checkpoint seeds (default: "1 2 3 4 5")
  --resume BOOL           Resume existing ranking output (default: True)
  --hf-repo-id ID         HF dataset containing inputs/checkpoints
                           (default: TheSyx/vulguard_lite)
  --hf-output-repo-id ID  HF dataset receiving ranking results
                           (default: same as --hf-repo-id)
  --upload-results BOOL   Upload each completed ranking output (default: True)
  --max-concurrent N      Maximum simultaneous PBS array tasks (default: 4)
  --dry-run               Print the qsub command without submitting
  -h, --help              Show this help

Each array task runs one model/config/seed tuple exactly once. Configs are the
cross product <dataset>_<x>_<y>. HF_TOKEN must already be exported.

Examples:
  scripts/submit_line_ranking_sutd.sh \
    --dataset openssl --model jitfine \
    --x-values "0" --y-values "1" --seeds "1 2 3 4 5"

  scripts/submit_line_ranking_sutd.sh \
    --dataset linux --model deepjit \
    --x-values "0 1 2 3" --y-values "1 2 3" --resume True
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
        --x-values) require_value "$@"; X_VALUES="$2"; shift 2 ;;
        --y-values) require_value "$@"; Y_VALUES="$2"; shift 2 ;;
        --seeds) require_value "$@"; SEEDS="$2"; shift 2 ;;
        --resume) require_value "$@"; RESUME="$2"; shift 2 ;;
        --hf-repo-id) require_value "$@"; HF_REPO_ID="$2"; shift 2 ;;
        --hf-output-repo-id) require_value "$@"; HF_OUTPUT_REPO_ID="$2"; shift 2 ;;
        --upload-results) require_value "$@"; UPLOAD_RESULTS="$2"; shift 2 ;;
        --max-concurrent) require_value "$@"; MAX_CONCURRENT="$2"; shift 2 ;;
        --dry-run) DRY_RUN="True"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for required_name in DATASET MODEL X_VALUES Y_VALUES; do
    if [[ -z "${!required_name}" ]]; then
        required_flag="${required_name,,}"
        echo "--${required_flag//_/-} is required" >&2
        usage >&2
        exit 2
    fi
done

if [[ ! "$DATASET" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--dataset contains unsupported characters" >&2
    exit 2
fi
case "$MODEL" in
    jitfine|deepjit|simcom) ;;
    *) echo "--model must be jitfine, deepjit, or simcom" >&2; exit 2 ;;
esac
case "${RESUME,,}" in
    true|false|1|0|yes|no) ;;
    *) echo "--resume must be True or False" >&2; exit 2 ;;
esac
case "${UPLOAD_RESULTS,,}" in
    true|false|1|0|yes|no) ;;
    *) echo "--upload-results must be True or False" >&2; exit 2 ;;
esac
if [[ ! "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-concurrent must be a positive integer" >&2
    exit 2
fi
if [[ -z "${HF_TOKEN:-}" && "$DRY_RUN" != "True" ]]; then
    echo "HF_TOKEN is not set; export it before submitting." >&2
    exit 2
fi

read -r -a X_ARRAY <<< "$X_VALUES"
read -r -a Y_ARRAY <<< "$Y_VALUES"
read -r -a SEED_ARRAY <<< "$SEEDS"
if ((${#X_ARRAY[@]} == 0 || ${#Y_ARRAY[@]} == 0 || ${#SEED_ARRAY[@]} == 0)); then
    echo "Config and seed lists must not be empty" >&2
    exit 2
fi
for value in "${X_ARRAY[@]}" "${Y_ARRAY[@]}" "${SEED_ARRAY[@]}"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "Config indices and seeds must be non-negative integers: $value" >&2
        exit 2
    fi
done

task_count=$((${#X_ARRAY[@]} * ${#Y_ARRAY[@]} * ${#SEED_ARRAY[@]}))
last_index=$((task_count - 1))
config_x="${X_VALUES// /;}"
config_y="${Y_VALUES// /;}"
seed_list="${SEEDS// /;}"
HF_OUTPUT_REPO_ID="${HF_OUTPUT_REPO_ID:-$HF_REPO_ID}"
pbs_variables="HF_TOKEN,REPO_DIR=$REPO_DIR,DATASET=$DATASET,MODELS=$MODEL,CONFIG_X=$config_x,CONFIG_Y=$config_y,SEEDS=$seed_list,RESUME=$RESUME,HF_REPO_ID=$HF_REPO_ID,HF_OUTPUT_REPO_ID=$HF_OUTPUT_REPO_ID,HF_UPLOAD_RESULT=$UPLOAD_RESULTS"

command=(
    qsub
    -J "0-${last_index}%${MAX_CONCURRENT}"
    -v "$pbs_variables"
    "$PBS_SCRIPT"
)

echo "Dataset/model : $DATASET / $MODEL"
echo "Config X/Y    : ${X_ARRAY[*]} / ${Y_ARRAY[*]}"
echo "Seeds         : ${SEED_ARRAY[*]}"
echo "Array tasks   : $task_count (max concurrent: $MAX_CONCURRENT)"
echo "Resume        : $RESUME"
echo "HF upload     : $UPLOAD_RESULTS -> $HF_OUTPUT_REPO_ID"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "$DRY_RUN" != "True" ]]; then
    "${command[@]}"
fi
