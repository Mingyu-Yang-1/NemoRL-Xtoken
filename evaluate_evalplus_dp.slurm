#!/bin/bash
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=00:55:00
#SBATCH --job-name=evalplus_dp
#SBATCH --output=logs/evalplus_dp_%j.out
#SBATCH --error=logs/evalplus_dp_%j.err

# Data-parallel HumanEval+/MBPP+ for chat models.
# 8 workers, each pinned to one GPU, each handles 1/8 of problems.
# Targets the 36-40 min preemption window on coreai_dlalgo_genai.
#
# Env knobs:
#   MODEL          required — HF repo id
#   DATASET        humaneval | mbpp                  (default humaneval)
#   WORLD_SIZE     workers/GPUs                       (default 8)
#   MAX_NEW_TOKENS                                    (default 1024)
#   LIMIT          if >0, process only first N        (smoke test)
#   SKIP_GRADE     1 to stop after codegen            (default 0)

set -euo pipefail

WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/xtoken_nemorl_github_latest

: "${MODEL:?MODEL required}"
DATASET="${DATASET:-humaneval}"
WORLD_SIZE="${WORLD_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LIMIT="${LIMIT:-0}"
SKIP_GRADE="${SKIP_GRADE:-0}"

mkdir -p "${WORKDIR}/logs"

export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN required (export from shell or pass via --export=ALL,HF_TOKEN=...)}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_ALLOW_CODE_EVAL=1

SAFE_TAG="$(printf '%s' "${MODEL}" | sed 's|/|__|g')__evalplus_dp"
OUTPUT_ROOT="${WORKDIR}/eval_results/${SAFE_TAG}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}"
mkdir -p "${OUTPUT_DIR}"

echo "=== EvalPlus DP ==="
echo "  MODEL          : ${MODEL}"
echo "  DATASET        : ${DATASET}"
echo "  WORLD_SIZE     : ${WORLD_SIZE}"
echo "  MAX_NEW_TOKENS : ${MAX_NEW_TOKENS}"
echo "  LIMIT          : ${LIMIT}"
echo "  SKIP_GRADE     : ${SKIP_GRADE}"
echo "  OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "==================="

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
        NUMPY_VER=\$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '1.26.4')
        TORCH_VER=\$(python -c 'import torch; print(torch.__version__.split(\"+\")[0])' 2>/dev/null || echo '2.5.0')
        cat > /tmp/evalplus_constraint.txt <<C
numpy==\$NUMPY_VER
torch==\$TORCH_VER
C
        echo \"[constraint] numpy==\$NUMPY_VER torch==\$TORCH_VER\"
        pip install --quiet --upgrade --target \$TARGET --no-deps \
            'transformers>=5.9.0,<6.0.0' 'huggingface_hub>=0.30' \
            'tokenizers>=0.22.0,<=0.23.0' 'accelerate>=1.0' regex \
            2>&1 | tail -5 || true
        pip install --quiet --upgrade --target \$TARGET \
            --constraint /tmp/evalplus_constraint.txt \
            evalplus 2>&1 | tail -5 || true
        export PYTHONPATH=\$TARGET:\${PYTHONPATH:-}
        export PYTHONNOUSERSITE=0
        python -c 'import evalplus.evaluate; print(\"evalplus import OK\")' 2>&1 | head -3

        EXTRA=''
        [[ '${LIMIT}' != '0' ]]      && EXTRA+=\" --limit ${LIMIT}\"
        [[ '${SKIP_GRADE}' == '1' ]] && EXTRA+=\" --skip_grade\"

        python ${WORKDIR}/examples/evalplus_dp_codegen.py \
            --model '${MODEL}' \
            --dataset '${DATASET}' \
            --output_dir '${OUTPUT_DIR}' \
            --world_size ${WORLD_SIZE} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            \$EXTRA \
            2>&1 | tee '${OUTPUT_DIR}/codegen.log'

        echo
        echo '=== Final ${DATASET} pass@1 ==='
        grep -E 'pass@1|HumanEval|MBPP' '${OUTPUT_DIR}/codegen.log' | tail -10 || true
    "
