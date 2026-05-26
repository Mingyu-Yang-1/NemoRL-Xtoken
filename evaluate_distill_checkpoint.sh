#!/bin/bash
set -euo pipefail

# =============================================================================
# Evaluate a single xtoken-distilled checkpoint (basic suite via
# tokenalign_upstream/evaluate_models.py).
#
# xt's checkpoint layout is DCP-sharded:
#   checkpoints/<run_name>/step_<N>/policy/weights/model/shard-*.safetensors
# This script auto-converts it to an HF folder via the in-repo
# `examples/converters/convert_dcp_to_hf.py`, caches the result under
# ${HF_CONVERTED_ROOT}, and runs lm-eval-harness on the HF folder.
#
# Usage:
#   bash evaluate_distill_checkpoint.sh                                  # default base
#   bash evaluate_distill_checkpoint.sh base                             # base Llama-3.1-8B
#   bash evaluate_distill_checkpoint.sh <run_name>                       # latest step under that run
#   bash evaluate_distill_checkpoint.sh <run_name> step_100              # specific step
#   bash evaluate_distill_checkpoint.sh <run_name> 100                   # numeric shortcut
#   bash evaluate_distill_checkpoint.sh /abs/path/to/hf_folder           # arbitrary HF folder
#
# Or via env: CHECKPOINT=step_100 bash evaluate_distill_checkpoint.sh <run_name>
#
# Argument resolution:
#   "base"               → evaluate the raw STUDENT_MODEL (no --model_path)
#   path containing "/"  → if it's a folder, use as --model_path directly;
#                          if it's an HF repo id (no local copy), use as
#                          --model_name. CHECKPOINT ignored.
#   anything else        → look up checkpoints/<arg>/step_<CHECKPOINT> and
#                          convert DCP → HF on first use.
#
# Env overrides (all optional):
#   STUDENT_MODEL       base HF model id (default: meta-llama/Llama-3.1-8B).
#                       Only matters for "base" mode or for the HF tokenizer
#                       when --model_path is a folder without one.
#   CHECKPOINT          "latest" | "step_<N>" | "<N>"     (default: latest)
#   TRAINING_CONFIG     YAML used during training         (default: auto-pick
#                       under examples/configs/ matching the run name).
#   BATCH_SIZE          eval batch size                   (default: 8)
#   NUM_GPUS            number of GPUs                    (default: from nvidia-smi)
#   HF_CONVERTED_ROOT   where to cache HF-converted checkpoints
#                       (default: /lustre/fsw/portfolios/coreai/users/mingyyang/hf_converted)
#   FORCE_CONVERT       1 → re-run the DCP → HF conversion even if the cache
#                       already exists. Use after retraining the same step.
#   USE_WANDB           1 (default) | 0
#   GEN_KWARGS          forwarded to lm-eval simple_evaluate (see comments
#                       in tokenalign_upstream/evaluate_distill_checkpoint.sh).
#
# Must run inside a container that has nemo_rl (for DCP conversion) AND
# lm-eval-harness (for the eval itself). The matching SLURM wrapper
# (evaluate_distill_checkpoint.slurm) sets this up.
# =============================================================================

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"

ARG="${1:-base}"
CHECKPOINT="${2:-${CHECKPOINT:-latest}}"
STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.1-8B}"
BATCH_SIZE="${BATCH_SIZE:-8}"
USE_WANDB="${USE_WANDB:-1}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
TRAINING_CONFIG="${TRAINING_CONFIG:-}"
HF_CONVERTED_ROOT="${HF_CONVERTED_ROOT:-/lustre/fsw/portfolios/coreai/users/mingyyang/hf_converted}"
# Where the upstream eval scripts live (we re-use them rather than copying).
TOKENALIGN_UPSTREAM="${TOKENALIGN_UPSTREAM:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/tokenalign_upstream}"
GEN_KWARGS="${GEN_KWARGS:-do_sample=False,temperature=0.0,top_p=1.0,top_k=0,max_gen_toks=512}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-0}"

# Normalize CHECKPOINT: "latest" | "step_N" | "N" → "step_N"
case "${CHECKPOINT}" in
    latest) ;;
    step_*) ;;
    [0-9]*) CHECKPOINT="step_${CHECKPOINT}" ;;
    *)
        echo "Error: invalid CHECKPOINT '${CHECKPOINT}'." >&2
        echo "       Must be 'latest', 'step_<N>', or a bare number like 100." >&2
        exit 1
        ;;
esac

# Detect GPU count.
if [[ -n "${NUM_GPUS:-}" ]]; then
    :
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NUM_GPUS="${NUM_GPUS:-1}"
fi
PARALLELIZE_ARG=""
if [[ "${NUM_GPUS}" -gt 1 ]]; then PARALLELIZE_ARG="--parallelize"; fi

# ---- Resolve checkpoint source ----------------------------------------------
# Three paths: "base" / explicit path / named run.
HF_MODEL_DIR=""
MODEL_SOURCE=""

# Helper: resolve which training-config YAML to feed convert_dcp_to_hf.
resolve_training_config() {
    local run_name="$1"
    if [[ -n "${TRAINING_CONFIG}" ]]; then
        echo "${TRAINING_CONFIG}"; return 0
    fi
    # Try to find a config matching the run name (heuristic): match longest YAML
    # filename basename (without _arrow.yaml) that is a substring of run_name.
    local candidate
    for cfg in "${WORKDIR}/examples/configs/"single_teacher_cross_tokenizer*_arrow*.yaml; do
        [[ -f "$cfg" ]] || continue
        local base
        base="$(basename "$cfg" .yaml)"
        if [[ "${run_name}" == *"${base}"* || "${base}" == *"${run_name}"* ]]; then
            candidate="$cfg"
            break
        fi
    done
    if [[ -z "${candidate:-}" ]]; then
        # Fallback: just use the first config — convert_dcp_to_hf only needs the
        # model_name field, not the loss/scheduler.
        candidate="$(ls "${WORKDIR}/examples/configs/"single_teacher_cross_tokenizer*_arrow*.yaml 2>/dev/null | head -1)"
    fi
    if [[ -z "${candidate}" || ! -f "${candidate}" ]]; then
        echo "Error: could not auto-resolve a training config under examples/configs/ for run '${run_name}'." >&2
        echo "       Pass TRAINING_CONFIG=<path/to/train.yaml> explicitly." >&2
        return 1
    fi
    echo "${candidate}"
}

# Helper: convert a DCP step dir to HF, cache, echo the HF path.
convert_dcp_to_hf_cached() {
    local dcp_dir="$1"     # e.g. checkpoints/<run>/step_100
    local run_name="$2"
    local step_tag="$3"    # e.g. step_100
    local hf_dir="${HF_CONVERTED_ROOT}/${run_name}__${step_tag}"

    if [[ -d "${hf_dir}" && -f "${hf_dir}/config.json" && "${FORCE_CONVERT}" != "1" ]]; then
        echo "[convert] cache hit: ${hf_dir}" >&2
        echo "${hf_dir}"
        return 0
    fi
    if [[ "${FORCE_CONVERT}" = "1" && -d "${hf_dir}" ]]; then
        echo "[convert] FORCE_CONVERT=1 — removing stale cache ${hf_dir}" >&2
        rm -rf "${hf_dir}"
    fi

    local cfg
    cfg="$(resolve_training_config "${run_name}")" || return 1
    echo "[convert] converting DCP → HF" >&2
    echo "          dcp = ${dcp_dir}" >&2
    echo "          cfg = ${cfg}" >&2
    echo "          out = ${hf_dir}" >&2
    mkdir -p "$(dirname "${hf_dir}")"

    # convert_dcp_to_hf.py expects --dcp-ckpt-path to be the policy/weights dir
    # (sharded safetensors live one level under that). Provide both.
    local dcp_weights="${dcp_dir}/policy/weights"
    if [[ ! -d "${dcp_weights}/model" ]]; then
        echo "Error: expected DCP weights at ${dcp_weights}/model — not found." >&2
        return 1
    fi
    python "${WORKDIR}/examples/converters/convert_dcp_to_hf.py" \
        --config "${cfg}" \
        --dcp-ckpt-path "${dcp_weights}" \
        --hf-ckpt-path "${hf_dir}"

    # Copy the tokenizer dir into the HF folder if it's not already there
    # (convert_dcp_to_hf only writes the model weights + config).
    if [[ ! -f "${hf_dir}/tokenizer.json" && -d "${dcp_dir}/policy/tokenizer" ]]; then
        cp -r "${dcp_dir}/policy/tokenizer/"* "${hf_dir}/"
    fi
    echo "${hf_dir}"
}

if [[ "${ARG}" == "base" ]]; then
    MODEL_PATH_ARG=""
    MODEL_SOURCE="base model: ${STUDENT_MODEL}"
elif [[ "${ARG}" == */* || -e "${ARG}" ]]; then
    if [[ -d "${ARG}" ]]; then
        HF_MODEL_DIR="${ARG}"
        MODEL_PATH_ARG="--model_path ${HF_MODEL_DIR}"
        MODEL_SOURCE="explicit folder: ${HF_MODEL_DIR}"
    elif [[ "${ARG}" == /* || "${ARG}" == ./* || "${ARG}" == ../* ]]; then
        echo "Error: explicit path does not exist: ${ARG}" >&2
        exit 1
    else
        STUDENT_MODEL="${ARG}"
        MODEL_PATH_ARG=""
        MODEL_SOURCE="HuggingFace repo: ${ARG}"
    fi
    if [[ "${CHECKPOINT}" != "latest" ]]; then
        echo "[warn] CHECKPOINT='${CHECKPOINT}' ignored when an explicit path / HF repo is given" >&2
    fi
else
    # Named-run mode.
    CKPT_DIR="${WORKDIR}/checkpoints/${ARG}"
    if [[ "${CHECKPOINT}" == "latest" ]]; then
        # Pick the highest-numbered step_<N> directory.
        STEP_DIR=$(ls -d "${CKPT_DIR}"/step_* 2>/dev/null \
            | awk -F'step_' '{print $2 "\t" $0}' \
            | sort -n -k1,1 | tail -1 | cut -f2)
        if [[ -z "${STEP_DIR}" ]]; then
            echo "Error: no step_* directories in ${CKPT_DIR}" >&2
            ls -la "${CKPT_DIR}" 2>&1 >&2 || true
            exit 1
        fi
        STEP_TAG=$(basename "${STEP_DIR}")
    else
        STEP_DIR="${CKPT_DIR}/${CHECKPOINT}"
        STEP_TAG="${CHECKPOINT}"
        if [[ ! -d "${STEP_DIR}" ]]; then
            echo "Error: checkpoint not found: ${STEP_DIR}" >&2
            echo "Available steps under ${CKPT_DIR}:" >&2
            ls -d "${CKPT_DIR}"/step_* 2>/dev/null | xargs -n1 basename >&2 || echo "  (none)" >&2
            exit 1
        fi
    fi
    HF_MODEL_DIR="$(convert_dcp_to_hf_cached "${STEP_DIR}" "${ARG}" "${STEP_TAG}")"
    MODEL_PATH_ARG="--model_path ${HF_MODEL_DIR}"
    MODEL_SOURCE="named run: ${ARG}/${STEP_TAG} → ${HF_MODEL_DIR}"
fi

# Default OUTPUT_PATH to eval_results/<safe-tag> under WORKDIR when not set.
if [[ -z "${OUTPUT_PATH:-}" ]]; then
    SAFE_TAG="$(printf '%s' "${ARG}" | sed 's|/|__|g')"
    if [[ "${ARG}" != "base" && "${ARG}" != */* ]]; then
        SAFE_TAG="${SAFE_TAG}__${STEP_TAG:-latest}"
    fi
    OUTPUT_PATH="${WORKDIR}/eval_results/${SAFE_TAG}"
fi
mkdir -p "${OUTPUT_PATH}"

GEN_KWARGS_ARG=""
[[ -n "${GEN_KWARGS}" ]] && GEN_KWARGS_ARG="--gen_kwargs ${GEN_KWARGS}"
WANDB_ARGS=""
if [[ "${USE_WANDB}" = "1" && "${WANDB_MODE:-}" != "disabled" ]]; then
    WANDB_ARGS="--use_wandb --wandb_project x_token"
fi

echo "=== Eval config ==="
echo "  ARG         : ${ARG}"
[[ -n "${STEP_TAG:-}" ]] && echo "  CHECKPOINT  : ${STEP_TAG}"
echo "  MODEL       : ${MODEL_SOURCE}"
echo "  STUDENT     : ${STUDENT_MODEL}  (used for tokenizer fallback / base mode)"
echo "  NUM_GPUS    : ${NUM_GPUS}"
echo "  PARALLELIZE : ${PARALLELIZE_ARG:-off}"
echo "  BATCH_SIZE  : ${BATCH_SIZE}"
echo "  OUTPUT_PATH : ${OUTPUT_PATH}"
echo "==================="

# Run eval (re-use upstream's evaluator).
EVAL_SCRIPT="${TOKENALIGN_UPSTREAM}/evaluate_models.py"
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Error: ${EVAL_SCRIPT} not found." >&2
    echo "       Set TOKENALIGN_UPSTREAM to the tokenalign_upstream root." >&2
    exit 1
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    accelerate launch --num_processes="${NUM_GPUS}" "${EVAL_SCRIPT}" \
        --model_name "${STUDENT_MODEL}" \
        ${MODEL_PATH_ARG} \
        --batch_size "${BATCH_SIZE}" \
        ${GEN_KWARGS_ARG} \
        --output_path "${OUTPUT_PATH}" \
        ${WANDB_ARGS}
else
    python "${EVAL_SCRIPT}" \
        --model_name "${STUDENT_MODEL}" \
        ${MODEL_PATH_ARG} \
        ${PARALLELIZE_ARG} \
        --batch_size "${BATCH_SIZE}" \
        ${GEN_KWARGS_ARG} \
        --output_path "${OUTPUT_PATH}" \
        ${WANDB_ARGS}
fi
