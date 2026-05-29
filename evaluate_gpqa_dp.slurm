#!/bin/bash
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=00:55:00
#SBATCH --job-name=gpqa_dp
#SBATCH --output=logs/gpqa_dp_%j.out
#SBATCH --error=logs/gpqa_dp_%j.err

# Data-parallel GPQA-diamond (CoT, 0-shot, chat-template) for instruct models.
# Uses lm-eval-harness's native DP via accelerate launch --num_processes 8.
# GPQA-diamond has 198 problems; 198/8 = ~25 per rank = ~5 min wall.
#
# Env knobs:
#   MODEL         required HF repo id
#   BATCH_SIZE    default 4
#   HF_TOKEN      default: the one with gated-dataset access
#   GEN_KWARGS    default: max_gen_toks=1024

set -euo pipefail
: "${MODEL:?MODEL required}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GEN_KWARGS="${GEN_KWARGS:-max_gen_toks=1024}"

WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/xtoken_nemorl_github_latest
mkdir -p ${WORKDIR}/logs

export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN required (export from shell; needs gated-dataset access for Idavidrein/gpqa)}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_DATASETS_TOKEN="$HF_TOKEN"

SAFE_TAG="$(printf '%s' "${MODEL}" | sed 's|/|__|g')__gpqa_dp"
OUTPUT_PATH="${WORKDIR}/eval_results/${SAFE_TAG}"
mkdir -p "${OUTPUT_PATH}"

echo "=== GPQA DP ==="
echo "  MODEL      : ${MODEL}"
echo "  BATCH_SIZE : ${BATCH_SIZE}"
echo "  HF_TOKEN   : ${HF_TOKEN:0:12}..."
echo "  OUTPUT     : ${OUTPUT_PATH}"
echo "================"

CONTAINER=/lustre/fsw/portfolios/llmservice/users/sharatht/megatron_artifacts/containers/megatron_py25_fla_with_deps.sqsh

srun --ntasks-per-node=1 \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKDIR}" \
    bash -c "
        set -euo pipefail
        cd ${WORKDIR}

        TARGET=/tmp/gpqa_deps_\${SLURM_JOB_ID:-\$\$}
        mkdir -p \$TARGET
        NUMPY_VER=\$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '1.26.4')
        TORCH_VER=\$(python -c 'import torch; print(torch.__version__.split(\"+\")[0])' 2>/dev/null || echo '2.5.0')
        cat > /tmp/gpqa_constraint.txt <<C
numpy==\$NUMPY_VER
torch==\$TORCH_VER
C
        echo \"[constraint] numpy==\$NUMPY_VER torch==\$TORCH_VER\"
        pip install --quiet --upgrade --target \$TARGET --no-deps \
            'transformers>=5.9.0,<6.0.0' 'huggingface_hub>=0.30' \
            'tokenizers>=0.22.0,<=0.23.0' 'accelerate>=1.0' regex \
            'lm-eval[math]==0.4.10' sympy 'antlr4-python3-runtime==4.11' \
            math_verify langdetect immutabledict datasets \
            2>&1 | tail -5 || true
        export PYTHONPATH=\$TARGET:\${PYTHONPATH:-}
        export PYTHONNOUSERSITE=0
        python -c 'import lm_eval, accelerate; print(\"lm_eval ok\", \"accelerate\", accelerate.__version__)'

        # Qwen3 enable_thinking=false tokenizer patch
        TOK_PATH='${MODEL}'
        if [[ '${MODEL}' == *Qwen3* || '${MODEL}' == *qwen3* ]]; then
            TOK_PATH=\$(python -c \"
import tempfile
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${MODEL}', trust_remote_code=True)
orig = tok.chat_template or ''
tok.chat_template = '{%- set enable_thinking = false %}\n' + orig
tmp = tempfile.mkdtemp(prefix='qwen3_no_think_')
tok.save_pretrained(tmp)
print(tmp)
\")
            echo \"[qwen3 patch] tokenizer at \$TOK_PATH\"
        fi

        # Pre-fetch GPQA dataset on driver to surface gated-access errors
        # quickly with a clear message, instead of buried in worker traces.
        python -c \"
from datasets import load_dataset
import os
ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', token=os.environ.get('HF_TOKEN'))
print('[gpqa] loaded', len(ds['train']), 'rows')
\" 2>&1 | tail -10

        accelerate launch --num_processes 8 --num_machines 1 --mixed_precision bf16 \
            -m lm_eval \
            --model hf \
            --model_args \"pretrained=${MODEL},tokenizer=\$TOK_PATH,trust_remote_code=True,parallelize=False,dtype=bfloat16\" \
            --tasks gpqa_diamond_cot_zeroshot \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --batch_size ${BATCH_SIZE} \
            --gen_kwargs ${GEN_KWARGS} \
            --output_path '${OUTPUT_PATH}' \
            --log_samples \
            2>&1 | tee '${OUTPUT_PATH}/gpqa.log'

        echo
        echo '=== Final GPQA-diamond ==='
        grep -E 'acc|exact_match' '${OUTPUT_PATH}/gpqa.log' | tail -10 || true
    "
