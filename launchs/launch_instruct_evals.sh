#!/bin/bash
# Launch chat-template full-suite evals for the 5 instruct/chat models.
# Each goes to its own SLURM job (8 GPUs, 4h wall, batch=8 default).
#
# Output paths:
#   eval_results/<owner>__<model>__chat/metrics_full_suite_chat.json
#
# Usage:
#   bash launchs/launch_instruct_evals.sh
#   bash launchs/launch_instruct_evals.sh --skip Llama Qwen3-8B    # subset

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORK_DIR"

ACCOUNT="${ACCOUNT:-nvr_lpr_llm}"
BATCH_SIZE="${BATCH_SIZE:-8}"
HF_OFFLINE="${HF_OFFLINE:-0}"
SKIP_LIST=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip) shift; SKIP_LIST+=("$1"); shift ;;
        --account) shift; ACCOUNT="$1"; shift ;;
        --batch_size) shift; BATCH_SIZE="$1"; shift ;;
        --hf_offline) HF_OFFLINE=1; shift ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

# (label, HF repo id, needs_parallelize)
# Gemma-4-26B-A4B-it is large enough to need device_map='auto'; the others
# fit fine in 8x H100 80GB data-parallel.
declare -a MODELS=(
    "Llama-3.1-8B-Instruct    meta-llama/Llama-3.1-8B-Instruct  0"
    "Qwen3-8B                 Qwen/Qwen3-8B                     0"
    "phi-4                    microsoft/phi-4                   0"
    "Qwen3-14B                Qwen/Qwen3-14B                    0"
    "Gemma-4-26B-A4B-it       google/gemma-4-26B-A4B-it         1"
)

skip_match() {
    local q="$1"
    for s in "${SKIP_LIST[@]}"; do
        if [[ "$q" == *"$s"* ]]; then return 0; fi
    done
    return 1
}

echo "==================================================="
echo " Launching chat-template evals for instruct models"
echo "==================================================="
echo "  account     : $ACCOUNT"
echo "  batch_size  : $BATCH_SIZE"
echo "  hf_offline  : $HF_OFFLINE"
echo "  skip list   : ${SKIP_LIST[*]:-<none>}"
echo "==================================================="

mkdir -p "$WORK_DIR/logs"

for entry in "${MODELS[@]}"; do
    read -r label model needs_par <<< "$entry"
    if skip_match "$label"; then
        echo "[skip] $label"; continue
    fi
    EXPORT="ALL,MODEL=${model},BATCH_SIZE=${BATCH_SIZE},HF_OFFLINE=${HF_OFFLINE}"
    [[ "$needs_par" == "1" ]] && EXPORT+=",PARALLELIZE=1"
    JID=$(sbatch --parsable \
        --gpus-per-node=8 --time=4:00:00 \
        --account="$ACCOUNT" \
        --export="$EXPORT" \
        "$WORK_DIR/evaluate_instruct_full_suite.slurm")
    PAR_TAG=""
    [[ "$needs_par" == "1" ]] && PAR_TAG=", parallelize"
    echo "  $label  →  $JID  (${model}${PAR_TAG})"
done

echo ""
echo "Track:  squeue -u \$USER -n eval_instruct"
echo "Logs:   logs/eval_instruct_<JOBID>.out"
echo "JSONs:  eval_results/<owner>__<model>__chat/metrics_full_suite_chat.json"
