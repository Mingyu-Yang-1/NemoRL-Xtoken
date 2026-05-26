#!/bin/bash
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=2:00:00
#SBATCH --job-name=consolidate_eval
#SBATCH --output=logs/consolidate_eval_%j.out
#SBATCH --error=logs/consolidate_eval_%j.err

# Consolidate a NeMo-RL FSDP-sharded safetensors checkpoint, then evaluate.
#
# Required env (via sbatch --export=ALL,VAR=val):
#   RUN_NAME       run dir name under checkpoints/  (e.g. pkl-qwen14b-topk-global-100s)
#   STEP           step number                       (e.g. step_100)
#
# Optional:
#   BATCH_SIZE     (default 16)
#   INCLUDE_CODE   0|1   default 1 (humaneval + mbpp always evaluated)
#   CODE_ONLY      0|1
#   LIMIT          per-task sample cap
#   HF_OFFLINE     1 to set HF_HUB/DATASETS_OFFLINE

set -euo pipefail

WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/xtoken_nemorl_github_latest
HF_CONVERTED_ROOT=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_converted

: "${RUN_NAME:?RUN_NAME required}"
: "${STEP:?STEP required, e.g. step_100}"

CKPT="${WORKDIR}/checkpoints/${RUN_NAME}/${STEP}"
SRC="${CKPT}/policy/weights/model"
TOK="${CKPT}/policy/tokenizer"
HF_DIR="${HF_CONVERTED_ROOT}/${RUN_NAME}__${STEP}"

mkdir -p "${WORKDIR}/logs"

export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

BATCH_SIZE="${BATCH_SIZE:-16}"
INCLUDE_CODE="${INCLUDE_CODE:-1}"   # default ON: humaneval + mbpp always evaluated
CODE_ONLY="${CODE_ONLY:-0}"
LIMIT="${LIMIT:-}"
HF_OFFLINE="${HF_OFFLINE:-0}"
# When set, delete the heavy checkpoint files (FSDP shards + optimizer
# state) AND the HF-converted copy after eval completes successfully.
# Useful for long training runs to keep storage bounded. The light
# metadata (training_info.json, config.yaml, train_dataloader.pt,
# tokenizer/) is preserved so the checkpoint dir remains a marker.
#
# WARNING: enabling this races with chained-job auto-resume. If you delete
# the checkpoint that the next training job will resume from, that job
# crashes with FileNotFoundError on the load_model call. Default OFF.
# Storage is bounded by NeMo-RL's keep_top_k anyway.
DELETE_AFTER_EVAL="${DELETE_AFTER_EVAL:-0}"

echo "=== consolidate_and_eval ==="
echo "  RUN_NAME : ${RUN_NAME}"
echo "  STEP     : ${STEP}"
echo "  SRC      : ${SRC}"
echo "  HF_DIR   : ${HF_DIR}"
echo "============================"

MEGATRON_CONTAINER=/lustre/fsw/portfolios/llmservice/users/sharatht/megatron_artifacts/containers/megatron_py25_fla_with_deps.sqsh

srun --ntasks-per-node=1 \
    --container-image="${MEGATRON_CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKDIR}" \
    bash -c "
        set -euo pipefail
        cd ${WORKDIR}

        # Container's transformers is too old for newer lm-eval; install fresh
        # deps to a --target dir (the eval scripts use the same trick).
        TARGET=/tmp/eval_deps_\${SLURM_JOB_ID:-\$\$}
        mkdir -p \$TARGET
        pip install --quiet --upgrade --target \$TARGET --no-deps \\
            'transformers>=5.9.0,<6.0.0' 'huggingface_hub>=0.30' \\
            'tokenizers>=0.22.0,<=0.23.0' regex safetensors \\
            'lm-eval[math]==0.4.10' 'accelerate>=1.0' \\
            sympy 'antlr4-python3-runtime==4.11' math_verify \\
            human_eval 2>&1 | tail -5 || true
        export PYTHONPATH=\$TARGET:\${PYTHONPATH:-}
        export PYTHONNOUSERSITE=0

        # ---- 1. Consolidate (idempotent: skip if HF_DIR exists & non-empty) ----
        if [[ -f '${HF_DIR}/model.safetensors.index.json' ]]; then
            echo '[skip] Consolidated HF folder already exists at ${HF_DIR}'
        else
            mkdir -p '${HF_DIR}'
            python ${WORKDIR}/examples/converters/consolidate_sharded_safetensors.py \\
                --src '${SRC}' \\
                --dst '${HF_DIR}' \\
                --tokenizer-src '${TOK}' \\
                --overwrite
        fi

        # ---- 2. Eval the consolidated HF folder ----
        if [[ '${HF_OFFLINE}' == '1' ]]; then
            export HF_HUB_OFFLINE=1
            export HF_DATASETS_OFFLINE=1
        fi
        export EVAL_DEPS_DIR=\$TARGET
        export SKIP_PIP_INSTALL=1

        EXTRA_ENV=''
        [[ -n '${LIMIT}' ]] && EXTRA_ENV+=\"LIMIT=${LIMIT} \"
        [[ '${CODE_ONLY}' == '1' ]] && EXTRA_ENV+='CODE_ONLY=1 '

        env \$EXTRA_ENV \\
            INCLUDE_CODE=${INCLUDE_CODE} \\
            BATCH_SIZE=${BATCH_SIZE} \\
            NUM_GPUS=${SLURM_GPUS_ON_NODE:-1} \\
            STUDENT_MODEL='${HF_DIR}' \\
            bash ${WORKDIR}/evaluate_full_suite.sh \\
                '${HF_DIR}' 'latest'

        # ---- 3. Optional: prune heavy checkpoint files after successful eval ----
        # Only runs if eval completed (set -euo pipefail above gates this).
        if [[ '${DELETE_AFTER_EVAL}' == '1' ]]; then
            echo '[prune] eval succeeded; removing heavy checkpoint files'
            rm -rf '${SRC}'                         # FSDP-sharded safetensors (~16 GB)
            rm -rf '${CKPT}/policy/optimizer'       # optimizer state (~16-32 GB)
            rm -rf '${HF_DIR}'                      # HF-converted copy (~16 GB)
            # Keep training_info.json, config.yaml, train_dataloader.pt,
            # tokenizer/ — small metadata, useful as a marker that this
            # step was checkpointed and evaluated.
            echo '[prune] done. Kept metadata files at ${CKPT}'
        fi
    "
