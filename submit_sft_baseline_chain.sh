#!/bin/bash
# Submit a chain of N sequential SLURM jobs for the CE-only SFT baseline.
# Mirrors submit_single_teacher_chain.sh in structure so the same training
# schedule (steps, batch, warmup/cosine, DP world) can be reproduced
# exactly, only swapping the loss/teacher signal for plain NLL CE.
#
# Token-level parity with the distillation run is preserved by:
#   - the SFT config uses dataset_name=arrow_text + same arrow_files glob
#   - same characters_per_sample (16384) for lazy text packing
#   - chat_template=null (passthrough) → raw packed text after rendering
#   - add_bos/add_eos/add_generation_prompt match distillation defaults
#   - same DP shape, same seed, same global batch
#
# Usage:
#   bash submit_sft_baseline_chain.sh [options]
#
# Options:
#   --config PATH          YAML config file
#                          (default: examples/configs/sft_baseline_llama8b_climb_arrow.yaml)
#   --n_jobs N             Number of chained jobs              (default: 5)
#   --nodes N              Number of SLURM nodes per job       (default: 32)
#   --time TIME            SLURM time limit per job            (default: 4:00:00)
#   --account ACCT         SLURM account                       (default: nvr_lpr_llm)
#   --partition PART       SLURM partition                     (default: batch)
#   --max_steps N          Max training steps                  (default: 10000)
#   --batch_size N         Global batch size                   (default: 768)
#   --run_name NAME        WandB/checkpoint run name (auto if omitted)
#   --extra_overrides STR  Extra Hydra overrides appended to the inner uv run

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Defaults ----
CONFIG="examples/configs/sft_baseline_llama8b_climb_arrow.yaml"
N_JOBS=5
NUM_NODES=32
TIME_LIMIT="4:00:00"
ACCOUNT="nvr_lpr_llm"
PARTITION="batch"
MAX_STEPS=10000
BATCH_SIZE=768
RUN_NAME=""
EXTRA_OVERRIDES=""

# ---- Parse named arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2";      shift 2 ;;
        --n_jobs)      N_JOBS="$2";      shift 2 ;;
        --nodes)       NUM_NODES="$2";   shift 2 ;;
        --time)        TIME_LIMIT="$2";  shift 2 ;;
        --account)     ACCOUNT="$2";     shift 2 ;;
        --partition)   PARTITION="$2";   shift 2 ;;
        --max_steps)   MAX_STEPS="$2";   shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        --run_name)    RUN_NAME="$2";    shift 2 ;;
        --extra_overrides) EXTRA_OVERRIDES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---- Auto-generate run name from config filename if not provided ----
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="$(basename "$CONFIG" .yaml | tr '_' '-')"
fi

# ---- Validate ----
CONFIG_PATH="${WORK_DIR}/${CONFIG}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: config not found: ${CONFIG_PATH}" >&2
    exit 1
fi
if [[ ! -f "${WORK_DIR}/ray.sub" ]]; then
    echo "Error: ray.sub not found in ${WORK_DIR}" >&2
    exit 1
fi

# ---- Compute scheduler shape: 5% linear warmup, 95% cosine decay ----
# Matches the LinearLR/CosineAnnealingLR/milestones layout in the SFT config
# and the distillation runs.
WARMUP_STEPS=$(( MAX_STEPS * 5 / 100 ))
[ "$WARMUP_STEPS" -lt 1 ] && WARMUP_STEPS=1
COSINE_T_MAX=$(( MAX_STEPS - WARMUP_STEPS ))

# ---- Build the COMMAND that ray.sub will execute on the head node ----
COMMAND=$(cat <<CMDEOF
export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN="\${HF_TOKEN:?HF_TOKEN required (export from shell before submission)}"
export HUGGINGFACE_HUB_TOKEN="\$HF_TOKEN"
export WANDB_API_KEY="\${WANDB_API_KEY:?WANDB_API_KEY required (export from shell before submission)}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Force ray actors to rebuild their per-actor venvs from pyproject.toml so
# transformers + tokenizers updates land inside the worker venv (same trick
# as the distillation submit).
export NRL_FORCE_REBUILD_VENVS=true
export NRL_IGNORE_VERSION_MISMATCH=1

cd ${WORK_DIR}

uv run ${WORK_DIR}/examples/run_sft.py \
  --config ${CONFIG_PATH} \
  cluster.num_nodes=${NUM_NODES} \
  policy.train_global_batch_size=${BATCH_SIZE} \
  sft.max_num_steps=${MAX_STEPS} \
  policy.scheduler.0.kwargs.total_iters=${WARMUP_STEPS} \
  policy.scheduler.1.kwargs.T_max=${COSINE_T_MAX} \
  "policy.scheduler.2.milestones=[${WARMUP_STEPS}]" \
  logger.wandb_enabled=true \
  logger.wandb.name=${RUN_NAME} \
  logger.log_dir=logs/${RUN_NAME} \
  checkpointing.enabled=true \
  checkpointing.checkpoint_dir=checkpoints/${RUN_NAME} ${EXTRA_OVERRIDES}
CMDEOF
)

# ---- Print summary ----
echo "Submitting CE-only SFT baseline chain of $N_JOBS jobs"
echo "  Config       : $CONFIG"
echo "  RUN_NAME     : $RUN_NAME"
echo "  Nodes/job    : $NUM_NODES"
echo "  Time/job     : $TIME_LIMIT"
echo "  Account      : $ACCOUNT"
echo "  Partition    : $PARTITION"
echo "  Max steps    : $MAX_STEPS"
echo "  Warmup       : $WARMUP_STEPS  (5% of max steps)"
echo "  Cosine T_max : $COSINE_T_MAX  (95% of max steps)"
echo "  Batch size   : $BATCH_SIZE"
echo "  extra        : ${EXTRA_OVERRIDES:-<none>}"
echo ""

# ---- Create logs directory ----
mkdir -p "${WORK_DIR}/logs"

LOG_PREFIX="${WORK_DIR}/logs/${RUN_NAME}"
JOB_NAME="sft-$(echo "$RUN_NAME" | cut -c1-28)"

# ---- Submit the chain ----
EXPORT_VARS="ALL,CONTAINER=/lustre/fsw/portfolios/coreai/users/mingyyang/containers/nemo_rl_0.6.0.sqsh,MOUNTS=/lustre:/lustre,BASE_LOG_DIR=${WORK_DIR}/x_token,COMMAND=${COMMAND}"

COMMON_SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --nodes="${NUM_NODES}"
    --ntasks-per-node=1
    --gpus-per-node=8
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --time="${TIME_LIMIT}"
    --output="${LOG_PREFIX}_%j.out"
    --error="${LOG_PREFIX}_%j.err"
    --export="${EXPORT_VARS}"
)

JOB_ID=$(sbatch --parsable \
    "${COMMON_SBATCH_ARGS[@]}" \
    "${WORK_DIR}/ray.sub")
echo "  Job 1/$N_JOBS submitted: $JOB_ID"

for i in $(seq 2 $N_JOBS); do
    JOB_ID=$(sbatch --parsable \
        "${COMMON_SBATCH_ARGS[@]}" \
        --dependency=afterany:$JOB_ID \
        "${WORK_DIR}/ray.sub")
    echo "  Job $i/$N_JOBS submitted: $JOB_ID (depends on previous)"
done

echo ""
echo "Chain submitted. Monitor with: squeue -u \$USER"
echo "Cancel entire chain with:      scancel --name=${JOB_NAME}"
