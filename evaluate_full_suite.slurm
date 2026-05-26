#!/bin/bash
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=8:00:00
#SBATCH --job-name=eval_fullsuite
#SBATCH --output=logs/eval_fullsuite_%j.out
#SBATCH --error=logs/eval_fullsuite_%j.err

# --gpus-per-node intentionally omitted — set via `sbatch --gpus-per-node=N`
# based on model size. For Llama-8B: 1-2 GPUs is plenty.

# SLURM wrapper for the xt full-suite eval. Forwards positional + env args
# into the container. Mirrors evaluate_distill_checkpoint.slurm.
#
# Submit examples:
#   # Base
#   sbatch --gpus-per-node=1 evaluate_full_suite.slurm
#
#   # Latest checkpoint under a named run
#   sbatch --gpus-per-node=2 --export=ALL,RUN_NAME=<run_name> evaluate_full_suite.slurm
#
#   # Specific step
#   sbatch --gpus-per-node=2 --export=ALL,RUN_NAME=<run>,CHECKPOINT=step_500 \
#       evaluate_full_suite.slurm
#
#   # Sweep every nth checkpoint
#   sbatch --gpus-per-node=2 --time=24:00:00 \
#       --export=ALL,RUN_NAME=<run>,CHECKPOINT=all,STRIDE=2 \
#       evaluate_full_suite.slurm
#
#   # With code-eval (humaneval + mbpp)
#   sbatch --gpus-per-node=2 --export=ALL,RUN_NAME=<run>,INCLUDE_CODE=1 \
#       evaluate_full_suite.slurm

set -euo pipefail

WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/xtoken_nemorl_github_latest

# ---- Knobs (override via sbatch --export=ALL,VAR=val) ----
RUN_NAME="${RUN_NAME:-base}"
CHECKPOINT="${CHECKPOINT:-latest}"
STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.1-8B}"
TRAINING_CONFIG="${TRAINING_CONFIG:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-}"
SKIP_TASKS="${SKIP_TASKS:-}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
STRIDE="${STRIDE:-1}"
INCLUDE_CODE="${INCLUDE_CODE:-1}"   # default ON: humaneval + mbpp always evaluated
CODE_ONLY="${CODE_ONLY:-0}"
FORCE_PARALLELIZE="${FORCE_PARALLELIZE:-0}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
USE_WANDB="${USE_WANDB:-1}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
HF_CONVERTED_ROOT="${HF_CONVERTED_ROOT:-/lustre/fsw/portfolios/coreai/users/mingyyang/hf_converted}"
TOKENALIGN_UPSTREAM="${TOKENALIGN_UPSTREAM:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/tokenalign_upstream}"

mkdir -p "${WORKDIR}/logs"

export PYTHONPATH="${WORKDIR}:${TOKENALIGN_UPSTREAM}:${PYTHONPATH:-}"
export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
# Opt-in offline mode for datasets — set HF_OFFLINE=1 when datasets are pre-cached
# (avoids 429 storms when 8 ranks all hit cais/mmlu metadata API at once).
if [[ "${HF_OFFLINE:-0}" = "1" ]]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    echo "  HF_OFFLINE       : 1 (HF_HUB_OFFLINE + HF_DATASETS_OFFLINE)"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

echo "=== eval_full_suite SLURM wrapper ==="
echo "  WORKDIR          : ${WORKDIR}"
echo "  RUN_NAME         : ${RUN_NAME}"
echo "  CHECKPOINT       : ${CHECKPOINT}"
[[ "${CHECKPOINT}" = "all" && "${STRIDE}" -ne 1 ]] && echo "  STRIDE           : ${STRIDE}"
echo "  STUDENT_MODEL    : ${STUDENT_MODEL}"
echo "  TRAINING_CONFIG  : ${TRAINING_CONFIG:-<auto>}"
echo "  BATCH_SIZE       : ${BATCH_SIZE}"
echo "  NUM_GPUS         : ${NUM_GPUS}"
echo "  INCLUDE_CODE     : ${INCLUDE_CODE}"
echo "  CODE_ONLY        : ${CODE_ONLY}"
echo "  FORCE_PARALLELIZE: ${FORCE_PARALLELIZE}"
echo "  FORCE_CONVERT    : ${FORCE_CONVERT}"
echo "====================================="

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
        # The megatron container ships transformers 4.56.2 in system site-packages.
        # Even with PYTHONNOUSERSITE=0, system wins over --user, so we install to
        # an explicit --target dir and prepend PYTHONPATH so the newer transformers
        # takes precedence over BOTH system and user paths.
        TARGET=/tmp/eval_deps_\${SLURM_JOB_ID:-\$\$}
        mkdir -p \$TARGET
        # --no-deps so we don't co-install torch / numpy / etc. and break the
        # container's torch+torchvision combo. We install only the packages
        # whose container versions are too old for gemma-4 / current lm-eval.
        pip install --quiet --upgrade --target \$TARGET --no-deps \\
            'transformers>=5.9.0,<6.0.0' 'huggingface_hub>=0.30' \\
            'tokenizers>=0.22.0,<=0.23.0' regex \\
            'lm-eval[math]==0.4.10' 'accelerate>=1.0' \\
            sympy 'antlr4-python3-runtime==4.11' math_verify \\
            human_eval 2>&1 | tail -5 || true
        export PYTHONPATH=\$TARGET:\${PYTHONPATH:-}
        export PYTHONNOUSERSITE=0
        python -c 'import transformers, lm_eval; print(\"using transformers\", transformers.__version__, \"  lm_eval\", lm_eval.__version__)'
        # Pass the same target so evaluate_full_suite.sh skips its own redundant install.
        export EVAL_DEPS_DIR=\$TARGET
        export SKIP_PIP_INSTALL=1
        bash ${WORKDIR}/evaluate_full_suite.sh \\
            '${RUN_NAME}' '${CHECKPOINT}'
    "
