#!/bin/bash
# Submit a chain of N sequential SLURM jobs for single-teacher cross-tokenizer
# distillation in xtoken_nemorl_github_latest. Each job resumes from the latest
# checkpoint saved by the previous one (NeMo-RL auto-resume from
# checkpointing.checkpoint_dir).
#
# Usage:
#   bash submit_single_teacher_chain.sh [options]
#
# Options:
#   --config PATH          YAML config file (default: examples/configs/xtoken_distillation.yaml)
#   --n_jobs N             Number of chained jobs              (default: 1)
#   --nodes N              Number of SLURM nodes per job       (default: 4)
#   --time TIME            SLURM time limit per job            (default: 4:00:00)
#   --account ACCT         SLURM account                       (default: coreai_dlalgo_genai)
#   --partition PART       SLURM partition                     (default: batch)
#   --max_steps N          Max training steps                  (default: 5000)
#   --batch_size N         Global batch size                   (default: 64)
#   --teacher_load_precision PREC
#                          Teacher model load precision        (default: bfloat16)
#   --arrow_files GLOB     Arrow shard glob (required)         (default: climb 60-shard slice)
#   --run_name NAME        WandB/checkpoint run name (auto-generated if omitted)
#
# Examples:
#   bash submit_single_teacher_chain.sh
#   bash submit_single_teacher_chain.sh --n_jobs 5 --nodes 8 --max_steps 10000
#   bash submit_single_teacher_chain.sh \
#     --config examples/configs/xtoken_distillation.yaml \
#     --nodes 4 --n_jobs 3 --max_steps 8000 --batch_size 128

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Defaults ----
CONFIG="examples/configs/single_teacher_cross_tokenizer_llama8b_qwen3_32b_arrow.yaml"
N_JOBS=1
NUM_NODES=16
TIME_LIMIT="4:00:00"
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch"
MAX_STEPS=2000
BATCH_SIZE=768
TEACHER_LOAD_PRECISION="bfloat16"
ARROW_FILES="/lustre/fsw/portfolios/llmservice/users/sdiao/data/climb_nm5.5_phase3_400b_shuffled_text_only_global_shuffle/data-000[0-5][0-9]-of-02476.arrow"
RUN_NAME=""

# ---- Parse named arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";       shift 2 ;;
        --n_jobs)       N_JOBS="$2";       shift 2 ;;
        --nodes)        NUM_NODES="$2";    shift 2 ;;
        --time)         TIME_LIMIT="$2";   shift 2 ;;
        --account)      ACCOUNT="$2";      shift 2 ;;
        --partition)    PARTITION="$2";    shift 2 ;;
        --max_steps)    MAX_STEPS="$2";    shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
        --teacher_load_precision) TEACHER_LOAD_PRECISION="$2"; shift 2 ;;
        --arrow_files)  ARROW_FILES="$2";  shift 2 ;;
        --run_name)     RUN_NAME="$2";     shift 2 ;;
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
case "$TEACHER_LOAD_PRECISION" in
    float32|bfloat16|float16) ;;
    *) echo "Error: --teacher_load_precision must be one of: float32, bfloat16, float16" >&2; exit 1 ;;
esac

# ---- Compute scheduler shape: 5% linear warmup, 95% cosine decay ----
# Assumes the loaded config has a 3-entry policy.scheduler list:
#   [0] LinearLR  [1] CosineAnnealingLR  [2] ChainedScheduler milestones
# (xtoken_distillation.yaml matches this shape.)
WARMUP_STEPS=$(( MAX_STEPS * 5 / 100 ))
[ "$WARMUP_STEPS" -lt 1 ] && WARMUP_STEPS=1
COSINE_T_MAX=$(( MAX_STEPS - WARMUP_STEPS ))

# ---- Build the COMMAND that ray.sub will execute on the head node ----
COMMAND=$(cat <<CMDEOF
export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN=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
export HUGGINGFACE_HUB_TOKEN=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
export WANDB_API_KEY=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
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Ray workers run inside per-actor venvs baked into the container at
# /opt/ray_venvs/. Those were built from an older pyproject.toml and ship
# transformers without 'gemma4' in CONFIG_MAPPING. Force Ray to rebuild each
# worker venv from the current pyproject.toml so the transformers==5.3.0
# constraint actually lands inside the worker (otherwise teacher
# DTensorPolicyWorkerV2 actors die with "KeyError: 'gemma4'").
export NRL_FORCE_REBUILD_VENVS=true
export NRL_IGNORE_VERSION_MISMATCH=1

cd ${WORK_DIR}

uv run ${WORK_DIR}/examples/run_xtoken_distillation.py \
  --config ${CONFIG_PATH} \
  cluster.num_nodes=${NUM_NODES} \
  distillation.num_prompts_per_step=${BATCH_SIZE} \
  policy.train_global_batch_size=${BATCH_SIZE} \
  teacher.train_global_batch_size=${BATCH_SIZE} \
  ++teacher.dtensor_cfg.load_precision=${TEACHER_LOAD_PRECISION} \
  ++teacher.dtensor_cfg.shard_before_load=true \
  distillation.max_num_steps=${MAX_STEPS} \
  policy.scheduler.0.kwargs.total_iters=${WARMUP_STEPS} \
  policy.scheduler.1.kwargs.T_max=${COSINE_T_MAX} \
  "policy.scheduler.2.milestones=[${WARMUP_STEPS}]" \
  logger.wandb_enabled=true \
  logger.wandb.name=${RUN_NAME} \
  logger.log_dir=logs/${RUN_NAME} \
  checkpointing.enabled=true \
  checkpointing.checkpoint_dir=checkpoints/${RUN_NAME}
CMDEOF
)

# ---- Print summary ----
echo "Submitting chain of $N_JOBS jobs"
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
echo "  Teacher load : $TEACHER_LOAD_PRECISION"
echo "  Arrow files  : $ARROW_FILES"
echo ""

# ---- Create logs directory ----
mkdir -p "${WORK_DIR}/logs"

LOG_PREFIX="${WORK_DIR}/logs/${RUN_NAME}"
JOB_NAME="st-$(echo "$RUN_NAME" | cut -c1-28)"

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
