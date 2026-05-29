#!/bin/bash
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_instruct
#SBATCH --output=logs/eval_instruct_%j.out
#SBATCH --error=logs/eval_instruct_%j.err

# Chat-template full-suite eval for instruct/chat models.
#
# Submit example:
#   sbatch --gpus-per-node=8 \
#     --export=ALL,MODEL=meta-llama/Llama-3.1-8B-Instruct,BATCH_SIZE=8 \
#     evaluate_instruct_full_suite.slurm
#
# Env knobs:
#   MODEL              required — HF repo id
#   BATCH_SIZE         default 8
#   LIMIT              optional per-task sample cap
#   SKIP_TASKS         space-separated names to drop
#   SKIP_CODE          1 to drop humaneval + mbpp
#   PARALLELIZE        1 to force HF device_map='auto' (needed for >24B)
#   HF_OFFLINE         1 to disable HF Hub metadata calls (uses cached datasets)
#   GEN_KWARGS         forwarded to lm-eval simple_evaluate

set -euo pipefail

WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/xtoken_nemorl_github_latest
HF_CONVERTED_ROOT=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_converted   # unused here, kept for parity

: "${MODEL:?MODEL required, e.g. meta-llama/Llama-3.1-8B-Instruct}"

BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-}"
SKIP_TASKS="${SKIP_TASKS:-}"
SKIP_CODE="${SKIP_CODE:-0}"
PARALLELIZE="${PARALLELIZE:-0}"
HF_OFFLINE="${HF_OFFLINE:-0}"
GEN_KWARGS="${GEN_KWARGS:-max_gen_toks=1024}"   # default ceiling so mmlu_pro doesn't ask for 2048 in a 2048-ctx model

mkdir -p "${WORKDIR}/logs"

export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN required (export from shell or pass via --export=ALL,HF_TOKEN=...)}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY required (export from shell or pass via --export=ALL,WANDB_API_KEY=...)}"
# HF's `code_eval` metric (used by humaneval / humaneval_plus / mbpp / mbpp_plus)
# requires this env var to acknowledge that model-generated code will be
# executed. Without it, code tasks crash with a ValueError before any
# requests are dispatched.
export HF_ALLOW_CODE_EVAL=1

SAFE_TAG="$(printf '%s' "${MODEL}" | sed 's|/|__|g')__chat"
OUTPUT_PATH="${WORKDIR}/eval_results/${SAFE_TAG}"

echo "=== eval_instruct (chat template) ==="
echo "  MODEL        : ${MODEL}"
echo "  BATCH_SIZE   : ${BATCH_SIZE}"
echo "  PARALLELIZE  : ${PARALLELIZE}"
echo "  SKIP_CODE    : ${SKIP_CODE}"
echo "  LIMIT        : ${LIMIT:-<full>}"
echo "  OUTPUT       : ${OUTPUT_PATH}"
echo "======================================"

MEGATRON_CONTAINER=/lustre/fsw/portfolios/llmservice/users/sharatht/megatron_artifacts/containers/megatron_py25_fla_with_deps.sqsh

srun --ntasks-per-node=1 \
    --container-image="${MEGATRON_CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKDIR}" \
    bash -c "
        set -euo pipefail
        cd ${WORKDIR}

        # Same --target dep install trick as evaluate_full_suite.slurm.
        TARGET=/tmp/eval_deps_\${SLURM_JOB_ID:-\$\$}
        mkdir -p \$TARGET
        pip install --quiet --upgrade --target \$TARGET --no-deps \
            'transformers>=5.9.0,<6.0.0' 'huggingface_hub>=0.30' \
            'tokenizers>=0.22.0,<=0.23.0' regex \
            'lm-eval[math]==0.4.10' 'accelerate>=1.0' \
            sympy 'antlr4-python3-runtime==4.11' math_verify \
            human_eval langdetect immutabledict 2>&1 | tail -5 || true
        export PYTHONPATH=\$TARGET:\${PYTHONPATH:-}
        export PYTHONNOUSERSITE=0

        if [[ '${HF_OFFLINE}' == '1' ]]; then
            export HF_HUB_OFFLINE=1
            export HF_DATASETS_OFFLINE=1
        fi

        EXTRA_ARGS=''
        [[ -n '${LIMIT}' ]]      && EXTRA_ARGS+=\" --limit ${LIMIT}\"
        [[ -n '${SKIP_TASKS}' ]] && EXTRA_ARGS+=\" --skip_tasks=${SKIP_TASKS}\"
        [[ '${SKIP_CODE}' == '1' ]] && EXTRA_ARGS+=\" --skip_code\"
        [[ '${PARALLELIZE}' == '1' ]] && EXTRA_ARGS+=\" --parallelize\"
        [[ -n '${GEN_KWARGS}' ]] && EXTRA_ARGS+=\" --gen_kwargs=${GEN_KWARGS}\"

        python ${WORKDIR}/examples/evaluate_instruct_full_suite.py \
            --model_name '${MODEL}' \
            --batch_size ${BATCH_SIZE} \
            --output_path '${OUTPUT_PATH}' \
            \$EXTRA_ARGS
    "
