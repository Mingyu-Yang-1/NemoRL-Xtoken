#!/bin/bash
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=2:00:00
#SBATCH --job-name=evalplus
#SBATCH --output=logs/evalplus_%j.out
#SBATCH --error=logs/evalplus_%j.err

# Direct EvalPlus runner — purpose-built for chat-model HumanEval+/MBPP+.
# EvalPlus handles its own chat templating and answer extraction, which is
# why lm-eval-harness's code_eval flow keeps giving 0% / weird numbers on
# chat models.
#
# Env knobs:
#   MODEL          required — HF repo id (e.g. meta-llama/Llama-3.1-8B-Instruct)
#   DATASET        humaneval | mbpp                    (default humaneval — run twice for both)
#   BS             generation batch size               (default 8)
#   GREEDY         "--greedy" or empty                 (default --greedy)
#   PARALLELIZE    1 for HF device_map='auto'          (needed for >24B)
#   HF_OFFLINE     1 to disable HF metadata calls

set -euo pipefail

WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/xtoken_nemorl_github_latest

: "${MODEL:?MODEL required}"
DATASET="${DATASET:-humaneval}"
BS="${BS:-8}"
GREEDY="${GREEDY:---greedy}"
PARALLELIZE="${PARALLELIZE:-0}"
HF_OFFLINE="${HF_OFFLINE:-0}"

mkdir -p "${WORKDIR}/logs"

export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN required (export from shell or pass via --export=ALL,HF_TOKEN=...)}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

SAFE_TAG="$(printf '%s' "${MODEL}" | sed 's|/|__|g')__evalplus"
OUTPUT_ROOT="${WORKDIR}/eval_results/${SAFE_TAG}"
mkdir -p "${OUTPUT_ROOT}"

echo "=== EvalPlus ==="
echo "  MODEL       : ${MODEL}"
echo "  DATASET     : ${DATASET}"
echo "  BS          : ${BS}"
echo "  GREEDY      : ${GREEDY}"
echo "  PARALLELIZE : ${PARALLELIZE}"
echo "  OUTPUT      : ${OUTPUT_ROOT}"
echo "================"

MEGATRON_CONTAINER=/lustre/fsw/portfolios/llmservice/users/sharatht/megatron_artifacts/containers/megatron_py25_fla_with_deps.sqsh

srun --ntasks-per-node=1 \
    --container-image="${MEGATRON_CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKDIR}" \
    bash -c "
        set -euo pipefail
        cd ${WORKDIR}

        TARGET=/tmp/evalplus_deps_\${SLURM_JOB_ID:-\$\$}
        mkdir -p \$TARGET
        # Pin the container's existing numpy + torch versions in a
        # constraint file so pip's resolver doesn't drag in an
        # ABI-incompatible numpy upgrade.
        NUMPY_VER=\$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '1.26.4')
        TORCH_VER=\$(python -c 'import torch; print(torch.__version__.split(\"+\")[0])' 2>/dev/null || echo '2.5.0')
        cat > /tmp/evalplus_constraint.txt <<C
numpy==\$NUMPY_VER
torch==\$TORCH_VER
C
        echo \"[constraint] numpy==\$NUMPY_VER torch==\$TORCH_VER\"
        # Install heavy ML deps with --no-deps to protect torch.
        pip install --quiet --upgrade --target \$TARGET --no-deps \
            'transformers>=5.9.0,<6.0.0' 'huggingface_hub>=0.30' \
            'tokenizers>=0.22.0,<=0.23.0' 'accelerate>=1.0' regex \
            2>&1 | tail -5 || true
        # Install evalplus with FULL deps but the constraint keeps
        # numpy/torch pinned. Should pull in every transitive dep
        # without breaking the ABI.
        pip install --quiet --upgrade --target \$TARGET \
            --constraint /tmp/evalplus_constraint.txt \
            evalplus 2>&1 | tail -5 || true
        export PYTHONPATH=\$TARGET:\${PYTHONPATH:-}
        export PYTHONNOUSERSITE=0
        # Sanity-check before model load.
        python -c 'import evalplus.evaluate; print(\"evalplus import OK\")' 2>&1 | head -3
        export PYTHONPATH=\$TARGET:\${PYTHONPATH:-}
        export PYTHONNOUSERSITE=0

        if [[ '${HF_OFFLINE}' == '1' ]]; then
            export HF_HUB_OFFLINE=1
            export HF_DATASETS_OFFLINE=1
        fi
        # EvalPlus's HF backend uses device_map='auto' which auto-shards
        # the model across all visible GPUs. For models that fit in one
        # GPU (≤14B in bf16 on 80GB), this triggers an accelerate device-
        # placement bug ('tensors on cuda:0 and cuda:1'). Limit to one
        # GPU for those models. Pass SINGLE_GPU=1 to opt in.
        if [[ '${SINGLE_GPU:-0}' == '1' ]]; then
            export CUDA_VISIBLE_DEVICES=0
            echo '[single-gpu mode] CUDA_VISIBLE_DEVICES=0'
        fi

        # EvalPlus auto-detects chat-mode for instruct models from their HF
        # config (uses tokenizer.apply_chat_template if a chat_template is
        # set). We just point at the dataset and let it handle prompting.
        echo
        echo '--- evalplus.evaluate --dataset=${DATASET} --model=${MODEL} ---'
        # EvalPlus's HF backend auto-uses device_map='auto' for >24B models.
        # No explicit --parallelize flag in the CLI; the PARALLELIZE env is
        # ignored here. Gemma-4-26B should still load by sharding across
        # the 8 GPUs automatically.

        python -m evalplus.evaluate \
            --model '${MODEL}' \
            --dataset '${DATASET}' \
            --backend hf \
            ${GREEDY} \
            --bs ${BS} \
            --root '${OUTPUT_ROOT}' \
            2>&1 | tee '${OUTPUT_ROOT}/${DATASET}.log'

        # Echo the scores line for easy grep
        echo
        echo '=== Final ${DATASET} scores ==='
        grep -E 'pass@1|HumanEval|MBPP' '${OUTPUT_ROOT}/${DATASET}.log' | tail -10 || true
    "
