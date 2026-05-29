#!/bin/bash
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=00:55:00
#SBATCH --job-name=lmeval_dp
#SBATCH --output=logs/lmeval_dp_%j.out
#SBATCH --error=logs/lmeval_dp_%j.err

# Generic lm-eval-harness DP runner via accelerate launch --num_processes 8.
# Sized to land inside the ~40-min preemption window on coreai_dlalgo_genai.
#
# Env knobs:
#   MODEL         required HF repo id
#   TASK          required lm-eval task name (e.g. gsm8k_cot_zeroshot, ifeval)
#   BATCH_SIZE    default 4
#   GEN_KWARGS    default: max_gen_toks=1024
#   LIMIT         optional sample cap (debug)

set -euo pipefail
: "${MODEL:?MODEL required}"
: "${TASK:?TASK required (e.g. gsm8k_cot_zeroshot, ifeval)}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GEN_KWARGS="${GEN_KWARGS:-max_gen_toks=1024}"
LIMIT="${LIMIT:-}"

WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/xtoken_nemorl_github_latest
mkdir -p ${WORKDIR}/logs

export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN required (export from shell; some tasks need gated-dataset access)}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_DATASETS_TOKEN="$HF_TOKEN"

SAFE_MODEL="$(printf '%s' "${MODEL}" | sed 's|/|__|g')"
SAFE_TASK="$(printf '%s' "${TASK}" | sed 's|,|+|g')"
OUTPUT_PATH="${WORKDIR}/eval_results/${SAFE_MODEL}__${SAFE_TASK}__dp"
mkdir -p "${OUTPUT_PATH}"

echo "=== lm-eval DP ==="
echo "  MODEL      : ${MODEL}"
echo "  TASK       : ${TASK}"
echo "  BATCH_SIZE : ${BATCH_SIZE}"
echo "  GEN_KWARGS : ${GEN_KWARGS}"
echo "  LIMIT      : ${LIMIT:-<full>}"
echo "  OUTPUT     : ${OUTPUT_PATH}"
echo "==================="

CONTAINER=/lustre/fsw/portfolios/llmservice/users/sharatht/megatron_artifacts/containers/megatron_py25_fla_with_deps.sqsh

srun --ntasks-per-node=1 \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKDIR}" \
    bash -c "
        set -euo pipefail
        cd ${WORKDIR}

        TARGET=/tmp/lmeval_deps_\${SLURM_JOB_ID:-\$\$}
        mkdir -p \$TARGET
        NUMPY_VER=\$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '1.26.4')
        TORCH_VER=\$(python -c 'import torch; print(torch.__version__.split(\"+\")[0])' 2>/dev/null || echo '2.5.0')
        cat > /tmp/lmeval_constraint.txt <<C
numpy==\$NUMPY_VER
torch==\$TORCH_VER
C
        echo \"[constraint] numpy==\$NUMPY_VER torch==\$TORCH_VER\"
        pip install --quiet --upgrade --target \$TARGET --no-deps \
            'transformers>=5.9.0,<6.0.0' 'huggingface_hub>=0.30' \
            'tokenizers>=0.22.0,<=0.23.0' 'accelerate>=1.0' regex \
            'lm-eval[math,ifeval]==0.4.10' sympy 'antlr4-python3-runtime==4.11' \
            math_verify langdetect immutabledict datasets nltk \
            2>&1 | tail -5 || true
        export PYTHONPATH=\$TARGET:\${PYTHONPATH:-}
        export PYTHONNOUSERSITE=0
        python -c 'import lm_eval, accelerate; print(\"lm_eval ok\", \"accelerate\", accelerate.__version__)'

        # NLTK punkt for IFEval response parsing
        python -c 'import nltk; nltk.download(\"punkt\", quiet=True); nltk.download(\"punkt_tab\", quiet=True)' 2>&1 | tail -3 || true

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

        EXTRA=''
        [[ -n '${LIMIT}' ]] && EXTRA+=\" --limit ${LIMIT}\"

        accelerate launch --num_processes 8 --num_machines 1 --mixed_precision bf16 \
            -m lm_eval \
            --model hf \
            --model_args \"pretrained=${MODEL},tokenizer=\$TOK_PATH,trust_remote_code=True,parallelize=False,dtype=bfloat16\" \
            --tasks '${TASK}' \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --batch_size ${BATCH_SIZE} \
            --gen_kwargs ${GEN_KWARGS} \
            --output_path '${OUTPUT_PATH}' \
            --log_samples \
            \$EXTRA \
            2>&1 | tee '${OUTPUT_PATH}/run.log'

        echo
        echo '=== Final results table ==='
        tail -80 '${OUTPUT_PATH}/run.log' | grep -E 'exact_match|inst_level|prompt_level|acc|Tasks' | tail -20 || true
    "
