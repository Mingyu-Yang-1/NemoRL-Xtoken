#!/bin/bash
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_distill
#SBATCH --output=logs/eval_distill_%j.out
#SBATCH --error=logs/eval_distill_%j.err

# --gpus-per-node omitted intentionally — set via `sbatch --gpus-per-node=N`
# based on model size.

# Evaluate a single xtoken-distilled checkpoint. See evaluate_distill_checkpoint.sh
# for the full CLI; this slurm wrapper forwards positional + env args into the
# script inside a container with nemo_rl + lm-eval-harness available.
#
# Submit examples:
#   # Base Llama-3.1-8B
#   sbatch --gpus-per-node=1 evaluate_distill_checkpoint.slurm
#
#   # Latest step under a named run
#   sbatch --gpus-per-node=1 --export=ALL,ARG=single-teacher-cross-tokenizer-llama8b-qwen3-14b-arrow \
#       evaluate_distill_checkpoint.slurm
#
#   # Specific step
#   sbatch --gpus-per-node=1 --export=ALL,ARG=<run_name>,CHECKPOINT=step_500 \
#       evaluate_distill_checkpoint.slurm

set -euo pipefail

WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/xtoken_nemorl_github_latest

# ---- Knobs (overridable via sbatch --export=ALL,VAR=val) ----
ARG="${ARG:-base}"
CHECKPOINT="${CHECKPOINT:-latest}"
STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.1-8B}"
TRAINING_CONFIG="${TRAINING_CONFIG:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
USE_WANDB="${USE_WANDB:-1}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
TOKENALIGN_UPSTREAM="${TOKENALIGN_UPSTREAM:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/tokenalign_upstream}"

mkdir -p "${WORKDIR}/logs"

# Shared env passed to both srun steps.
export PYTHONPATH="${WORKDIR}:${TOKENALIGN_UPSTREAM}:${PYTHONPATH:-}"
export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

echo "=== eval_distill_checkpoint SLURM wrapper ==="
echo "  WORKDIR         : ${WORKDIR}"
echo "  ARG             : ${ARG}"
echo "  CHECKPOINT      : ${CHECKPOINT}"
echo "  STUDENT_MODEL   : ${STUDENT_MODEL}"
echo "  TRAINING_CONFIG : ${TRAINING_CONFIG:-<auto>}"
echo "  BATCH_SIZE      : ${BATCH_SIZE}"
echo "  NUM_GPUS        : ${NUM_GPUS}"
echo "  FORCE_CONVERT   : ${FORCE_CONVERT}"
echo "============================================="

# The eval container needs both nemo_rl (for DCP conversion) AND lm-eval-harness
# (for the eval itself). The nemo_rl container has both — same image used by
# the training jobs. Falls back to megatron container if nemo_rl is unavailable.
NEMO_RL_CONTAINER=/lustre/fsw/portfolios/coreai/users/mingyyang/containers/nemo_rl_0.6.0.sqsh
MEGATRON_CONTAINER=/lustre/fsw/portfolios/llmservice/users/sharatht/megatron_artifacts/containers/megatron_py25_fla_with_deps.sqsh
EVAL_CONTAINER="${EVAL_CONTAINER:-${MEGATRON_CONTAINER}}"

srun --ntasks-per-node=1 \
    --container-image="${EVAL_CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKDIR}" \
    bash -c "
        set -euo pipefail
        cd ${WORKDIR}
        # Container ships transformers 4.56.2; bump for gemma-4 model_type.
        # System site wins over --user, so install to --target and PYTHONPATH.
        TARGET=/tmp/eval_deps_\${SLURM_JOB_ID:-\$\$}
        mkdir -p \$TARGET
        # --no-deps so we don't reinstall torch (would break container's torchvision).
        pip install --quiet --upgrade --target \$TARGET --no-deps \\
            'transformers>=5.9.0,<6.0.0' 'huggingface_hub>=0.30' \\
            'tokenizers>=0.22.0,<=0.23.0' regex \\
            'lm-eval[math]==0.4.10' 'accelerate>=1.0' 2>&1 | tail -5 || true
        export PYTHONPATH=\$TARGET:\${PYTHONPATH:-}
        export PYTHONNOUSERSITE=0
        python -c 'import transformers, lm_eval; print(\"using transformers\", transformers.__version__, \"  lm_eval\", lm_eval.__version__)'
        bash ${WORKDIR}/evaluate_distill_checkpoint.sh \\
            '${ARG}' '${CHECKPOINT}'
    "
