#!/bin/bash
set -euo pipefail

# =============================================================================
# Full-suite evaluation of xtoken-distilled checkpoints
# (tokenalign_upstream/evaluate_full_suite.py covers MMLU, ARC, OBQA, PIQA,
# HellaSwag, Winogrande, GSM8K-CoT, Minerva MATH, optionally HumanEval+MBPP).
#
# Mirrors tokenalign_upstream/evaluate_full_suite.sh but reads xt-style DCP
# checkpoints from checkpoints/<run_name>/step_<N>/ and auto-converts them
# to HF format under ${HF_CONVERTED_ROOT} before invoking lm-eval-harness.
#
# Usage:
#   bash evaluate_full_suite.sh                                          # base
#   bash evaluate_full_suite.sh base                                     # base
#   bash evaluate_full_suite.sh <run_name>                               # latest step
#   bash evaluate_full_suite.sh <run_name> step_100                      # specific step
#   bash evaluate_full_suite.sh <run_name> 100                           # numeric shortcut
#   bash evaluate_full_suite.sh <run_name> all [STRIDE=N]                # sweep
#   bash evaluate_full_suite.sh /abs/path/to/hf_folder                   # explicit HF folder
#
# Env overrides:
#   STUDENT_MODEL          (default: meta-llama/Llama-3.1-8B)
#   TRAINING_CONFIG        (default: auto-pick under examples/configs/)
#   BATCH_SIZE             (default: 8)
#   LIMIT                  (default: unset → full)
#   SKIP_TASKS             (default: unset)
#   BOOTSTRAP_ITERS        (default: 1000)
#   CHECKPOINT             (default: latest)
#   STRIDE                 (default: 1, only used when CHECKPOINT=all)
#   INCLUDE_CODE=1         humaneval + mbpp at pass@1
#   CODE_ONLY=1            only humaneval + mbpp
#   FORCE_PARALLELIZE=1    use model parallelism for very large models
#   FORCE_CONVERT=1        re-run DCP→HF even if cache exists
#   HF_CONVERTED_ROOT      where to cache HF conversions
#                          (default: /lustre/fsw/.../mingyyang/hf_converted)
#   TOKENALIGN_UPSTREAM    path to tokenalign_upstream (default: sibling dir)
#   SKIP_PIP_INSTALL=1     skip minerva/code-eval extra deps install
# =============================================================================

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"

RUN_NAME="${1:-base}"
CHECKPOINT="${2:-${CHECKPOINT:-latest}}"
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
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"
USE_WANDB="${USE_WANDB:-1}"
HF_CONVERTED_ROOT="${HF_CONVERTED_ROOT:-/lustre/fsw/portfolios/coreai/users/mingyyang/hf_converted}"
TOKENALIGN_UPSTREAM="${TOKENALIGN_UPSTREAM:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/tokenalign_upstream}"

# NOTE: upstream sets PYTHONNOUSERSITE=1 to avoid an old /root/.local transformers
# shadowing the container stack. We INVERT that here — the slurm wrapper installs
# the right transformers into a target dir and prepends PYTHONPATH; user-site is
# unused but kept enabled in case the caller wants to layer their own packages.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-0}"
EVAL_DEPS_DIR="${EVAL_DEPS_DIR:-/tmp/fullsuite_deps_${SLURM_JOB_ID:-$$}}"

[[ "${CODE_ONLY}" = "1" ]] && INCLUDE_CODE=1

# Normalize CHECKPOINT: "latest" | "all" | "step_N" | "N" → "step_N"
case "${CHECKPOINT}" in
    latest|all) ;;
    step_*) ;;
    [0-9]*) CHECKPOINT="step_${CHECKPOINT}" ;;
    *)
        echo "Error: invalid CHECKPOINT '${CHECKPOINT}'." >&2
        echo "       Must be 'latest', 'all', 'step_<N>', or a number." >&2
        exit 1
        ;;
esac

# ---- Install eval deps (minerva_math, optionally human_eval) ---------------
if [[ "${SKIP_PIP_INSTALL}" != "1" ]]; then
    if [[ "${CODE_ONLY}" != "1" && "${SKIP_TASKS}" != *"minerva_math"* ]]; then
        echo "Installing minerva_math deps..."
        mkdir -p "${EVAL_DEPS_DIR}"
        python -m pip install --quiet --upgrade --target "${EVAL_DEPS_DIR}" \
            sympy math_verify "antlr4-python3-runtime==4.11" || {
            echo "[warn] dep install failed — minerva_math will be skipped"
            SKIP_TASKS="${SKIP_TASKS} minerva_math"
        }
        export PYTHONPATH="${EVAL_DEPS_DIR}:${PYTHONPATH:-}"
    fi
    if [[ "${INCLUDE_CODE}" = "1" ]]; then
        echo "Installing code-eval deps (human_eval)..."
        mkdir -p "${EVAL_DEPS_DIR}"
        python -m pip install --quiet --upgrade --target "${EVAL_DEPS_DIR}" human_eval || {
            echo "[warn] human_eval install failed — disabling INCLUDE_CODE"
            INCLUDE_CODE=0; CODE_ONLY=0
        }
        export PYTHONPATH="${EVAL_DEPS_DIR}:${PYTHONPATH:-}"
    fi
fi

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
USE_ACCELERATE=0
if [[ "${NUM_GPUS}" -gt 1 ]]; then
    if [[ "${FORCE_PARALLELIZE}" = "1" ]]; then
        PARALLELIZE_ARG="--parallelize"
        echo "[info] FORCE_PARALLELIZE=1 — using model parallelism"
    else
        USE_ACCELERATE=1
    fi
fi

if [[ "${INCLUDE_CODE}" = "1" || "${CODE_ONLY}" = "1" ]]; then
    export HF_ALLOW_CODE_EVAL=1
fi

# ---- Resolve training config for DCP→HF conversion --------------------------
resolve_training_config() {
    local run_name="$1"
    if [[ -n "${TRAINING_CONFIG}" ]]; then
        echo "${TRAINING_CONFIG}"; return 0
    fi
    local candidate
    for cfg in "${WORKDIR}/examples/configs/"single_teacher_cross_tokenizer*_arrow*.yaml; do
        [[ -f "$cfg" ]] || continue
        local base; base="$(basename "$cfg" .yaml)"
        if [[ "${run_name}" == *"${base}"* || "${base}" == *"${run_name}"* ]]; then
            candidate="$cfg"; break
        fi
    done
    if [[ -z "${candidate:-}" ]]; then
        candidate="$(ls "${WORKDIR}/examples/configs/"single_teacher_cross_tokenizer*_arrow*.yaml 2>/dev/null | head -1)"
    fi
    if [[ -z "${candidate}" || ! -f "${candidate}" ]]; then
        echo "Error: could not auto-resolve a training config under examples/configs/ for '${run_name}'." >&2
        echo "       Pass TRAINING_CONFIG=<path/to/train.yaml>." >&2
        return 1
    fi
    echo "${candidate}"
}

# ---- DCP → HF conversion (cached under ${HF_CONVERTED_ROOT}) ----------------
convert_dcp_to_hf_cached() {
    local dcp_dir="$1"; local run_name="$2"; local step_tag="$3"
    local hf_dir="${HF_CONVERTED_ROOT}/${run_name}__${step_tag}"
    if [[ -d "${hf_dir}" && -f "${hf_dir}/config.json" && "${FORCE_CONVERT}" != "1" ]]; then
        echo "[convert] cache hit: ${hf_dir}" >&2
        echo "${hf_dir}"
        return 0
    fi
    [[ "${FORCE_CONVERT}" = "1" && -d "${hf_dir}" ]] && rm -rf "${hf_dir}"
    local cfg; cfg="$(resolve_training_config "${run_name}")" || return 1
    echo "[convert] DCP → HF: ${dcp_dir} → ${hf_dir} (cfg=${cfg})" >&2
    mkdir -p "$(dirname "${hf_dir}")"
    local dcp_weights="${dcp_dir}/policy/weights"
    if [[ ! -d "${dcp_weights}/model" ]]; then
        echo "Error: expected DCP weights at ${dcp_weights}/model" >&2
        return 1
    fi
    python "${WORKDIR}/examples/converters/convert_dcp_to_hf.py" \
        --config "${cfg}" \
        --dcp-ckpt-path "${dcp_weights}" \
        --hf-ckpt-path "${hf_dir}"
    if [[ ! -f "${hf_dir}/tokenizer.json" && -d "${dcp_dir}/policy/tokenizer" ]]; then
        cp -r "${dcp_dir}/policy/tokenizer/"* "${hf_dir}/"
    fi
    echo "${hf_dir}"
}

# ---- Pre-flight summary -----------------------------------------------------
echo "=== Full-suite eval config ==="
echo "  RUN_NAME        : ${RUN_NAME}"
if [[ "${RUN_NAME}" == */* ]]; then
    echo "  MODE            : explicit path (CHECKPOINT ignored)"
else
    echo "  CHECKPOINT      : ${CHECKPOINT}"
fi
[[ "${CHECKPOINT}" = "all" && "${STRIDE}" -ne 1 && "${RUN_NAME}" != */* ]] && \
    echo "  STRIDE          : ${STRIDE}"
echo "  STUDENT_MODEL   : ${STUDENT_MODEL}"
echo "  TRAINING_CONFIG : ${TRAINING_CONFIG:-<auto>}"
echo "  NUM_GPUS        : ${NUM_GPUS}"
if [[ "${USE_ACCELERATE}" = "1" ]]; then
    echo "  GPU strategy    : data-parallel (accelerate launch -n=${NUM_GPUS})"
elif [[ -n "${PARALLELIZE_ARG}" ]]; then
    echo "  GPU strategy    : model-parallel (--parallelize)"
else
    echo "  GPU strategy    : single GPU"
fi
echo "  BATCH_SIZE      : ${BATCH_SIZE}"
echo "  LIMIT           : ${LIMIT:-<full>}"
echo "  SKIP_TASKS      : ${SKIP_TASKS:-<none>}"
echo "  BOOTSTRAP_ITERS : ${BOOTSTRAP_ITERS}"
echo "  INCLUDE_CODE    : ${INCLUDE_CODE}"
echo "  CODE_ONLY       : ${CODE_ONLY}"
echo "==============================="

EXTRA_ARGS=""
[[ -n "${LIMIT}" ]] && EXTRA_ARGS="${EXTRA_ARGS} --limit ${LIMIT}"
[[ -n "${SKIP_TASKS}" ]] && EXTRA_ARGS="${EXTRA_ARGS} --skip_tasks ${SKIP_TASKS}"
[[ "${INCLUDE_CODE}" = "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --include_code"
[[ "${CODE_ONLY}" = "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --code_only"
WANDB_ARGS=""
if [[ "${USE_WANDB}" = "1" && "${WANDB_MODE:-}" != "disabled" ]]; then
    WANDB_ARGS="--use_wandb --wandb_project x_token"
fi

if [[ -z "${OUTPUT_PATH:-}" ]]; then
    SAFE_TAG="$(printf '%s' "${RUN_NAME}" | sed 's|/|__|g')"
    OUTPUT_PATH="${WORKDIR}/eval_results/${SAFE_TAG}"
fi
mkdir -p "${OUTPUT_PATH}"
echo "  OUTPUT_PATH     : ${OUTPUT_PATH}"

EVAL_SCRIPT="${TOKENALIGN_UPSTREAM}/evaluate_full_suite.py"
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Error: ${EVAL_SCRIPT} not found." >&2
    echo "       Set TOKENALIGN_UPSTREAM to point at tokenalign_upstream/." >&2
    exit 1
fi

# ---- Helper: run one eval ---------------------------------------------------
run_one_eval() {
    local model_path_arg="$1"; local label="$2"
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo " ${label}"
    echo "──────────────────────────────────────────────────────────"
    if [[ "${USE_ACCELERATE}" = "1" ]]; then
        accelerate launch --num_processes="${NUM_GPUS}" "${EVAL_SCRIPT}" \
            --model_name "${STUDENT_MODEL}" \
            ${model_path_arg} \
            --batch_size "${BATCH_SIZE}" \
            --bootstrap_iters "${BOOTSTRAP_ITERS}" \
            --output_path "${OUTPUT_PATH}" \
            ${WANDB_ARGS} ${EXTRA_ARGS}
    else
        python "${EVAL_SCRIPT}" \
            --model_name "${STUDENT_MODEL}" \
            ${model_path_arg} \
            ${PARALLELIZE_ARG} \
            --batch_size "${BATCH_SIZE}" \
            --bootstrap_iters "${BOOTSTRAP_ITERS}" \
            --output_path "${OUTPUT_PATH}" \
            ${WANDB_ARGS} ${EXTRA_ARGS}
    fi
}

# ---- Main dispatch ----------------------------------------------------------
if [[ "${RUN_NAME}" == "base" ]]; then
    run_one_eval "" "Base model: ${STUDENT_MODEL}"
    exit 0
fi

if [[ "${RUN_NAME}" == */* ]]; then
    if [[ -e "${RUN_NAME}" ]]; then
        [[ "${CHECKPOINT}" != "latest" ]] && \
            echo "[warn] CHECKPOINT='${CHECKPOINT}' ignored for explicit path" >&2
        run_one_eval "--model_path ${RUN_NAME}" "Explicit path: $(basename "${RUN_NAME%/}")"
        exit 0
    elif [[ "${RUN_NAME}" == /* || "${RUN_NAME}" == ./* || "${RUN_NAME}" == ../* ]]; then
        echo "Error: model path does not exist: ${RUN_NAME}" >&2
        exit 1
    else
        STUDENT_MODEL="${RUN_NAME}"
        [[ "${CHECKPOINT}" != "latest" ]] && \
            echo "[warn] CHECKPOINT='${CHECKPOINT}' ignored for HF repo input" >&2
        run_one_eval "" "HuggingFace repo: ${RUN_NAME}"
        exit 0
    fi
fi

# Named run mode.
CKPT_DIR="${WORKDIR}/checkpoints/${RUN_NAME}"
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "Error: checkpoint dir not found: ${CKPT_DIR}" >&2
    exit 1
fi

case "${CHECKPOINT}" in
    latest)
        STEP_DIR=$(ls -d "${CKPT_DIR}"/step_* 2>/dev/null \
            | awk -F'step_' '{print $2 "\t" $0}' \
            | sort -n -k1,1 | tail -1 | cut -f2)
        if [[ -z "${STEP_DIR}" ]]; then
            echo "Error: no step_* directories under ${CKPT_DIR}" >&2; exit 1
        fi
        STEP_TAG=$(basename "${STEP_DIR}")
        HF_DIR="$(convert_dcp_to_hf_cached "${STEP_DIR}" "${RUN_NAME}" "${STEP_TAG}")"
        run_one_eval "--model_path ${HF_DIR}" \
            "Latest: ${RUN_NAME}/${STEP_TAG}"
        ;;

    step_*)
        STEP_DIR="${CKPT_DIR}/${CHECKPOINT}"
        if [[ ! -d "${STEP_DIR}" ]]; then
            echo "Error: checkpoint not found: ${STEP_DIR}" >&2
            ls -d "${CKPT_DIR}"/step_* 2>/dev/null | xargs -n1 basename >&2 || echo "  (none)" >&2
            exit 1
        fi
        HF_DIR="$(convert_dcp_to_hf_cached "${STEP_DIR}" "${RUN_NAME}" "${CHECKPOINT}")"
        run_one_eval "--model_path ${HF_DIR}" \
            "Checkpoint: ${RUN_NAME}/${CHECKPOINT}"
        ;;

    all)
        # Sort step_<N> directories numerically and apply STRIDE.
        mapfile -t ALL_STEPS < <(
            ls -d "${CKPT_DIR}"/step_* 2>/dev/null \
                | awk -F'step_' '{print $2}' \
                | sort -n
        )
        if [[ "${#ALL_STEPS[@]}" -eq 0 ]]; then
            echo "Error: no step_* dirs under ${CKPT_DIR}" >&2; exit 1
        fi
        echo "Found ${#ALL_STEPS[@]} checkpoint(s)"
        if [[ "${STRIDE}" -ne 1 ]]; then
            FILTERED=()
            for i in "${!ALL_STEPS[@]}"; do
                if (( i % STRIDE == 0 )); then
                    FILTERED+=("${ALL_STEPS[$i]}")
                fi
            done
            ALL_STEPS=("${FILTERED[@]}")
            echo "After STRIDE=${STRIDE}: ${#ALL_STEPS[@]} selected"
        fi
        echo "Steps to evaluate: ${ALL_STEPS[*]}"
        for n in "${ALL_STEPS[@]}"; do
            STEP_DIR="${CKPT_DIR}/step_${n}"
            HF_DIR="$(convert_dcp_to_hf_cached "${STEP_DIR}" "${RUN_NAME}" "step_${n}")"
            run_one_eval "--model_path ${HF_DIR}" \
                "Sweep [${n}]: ${RUN_NAME}/step_${n}"
        done
        echo ""
        echo "Sweep complete: ${#ALL_STEPS[@]} checkpoint(s) evaluated."
        ;;
esac
