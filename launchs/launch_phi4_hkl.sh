#!/bin/bash
# Launch Llama-3.1-8B ← Phi-4 (H-KL) distillation chain + auto-eval watcher.
#
# Submits 5 chained SLURM jobs (32 nodes each, 4h wall) and starts the
# auto-eval watcher in the background. The watcher fires a full-suite eval
# (incl. HumanEval + MBPP) at every multiple of 500 steps. Auto-deletion
# of checkpoint weights after eval is OFF (it races with chained-job
# auto-resume); NeMo-RL's keep_top_k handles storage.
#
# Usage:
#   bash launchs/launch_phi4_hkl.sh
#
# Edit the knobs below for one-off changes. For a per-run override of the
# loss-side flags, append to EXTRA_OVERRIDES (e.g. ++loss_fn.kl_chunk_shift=false).

set -euo pipefail

# ---- Resolve repo root ----
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORK_DIR"

# ---- Knobs ----
CONFIG=examples/configs/single_teacher_cross_tokenizer_llama8b_phi4_arrow_hkl.yaml
MAX_STEPS=10000
N_JOBS=5
NODES=32
ACCOUNT=nvr_lpr_llm
EXTRA_OVERRIDES=""   # e.g. "++loss_fn.kl_chunk_shift=false"

# ---- Derive run name (matches submit_single_teacher_chain.sh auto-derivation) ----
RUN_NAME="$(basename "$CONFIG" .yaml | tr '_' '-')"
TAG=phi4_hkl

echo "==================================================="
echo " Launching Llama-8B ← Phi-4 (H-KL) distillation"
echo "==================================================="
echo "  config       : $CONFIG"
echo "  run_name     : $RUN_NAME"
echo "  max_steps    : $MAX_STEPS"
echo "  n_jobs       : $N_JOBS"
echo "  nodes/job    : $NODES"
echo "  account      : $ACCOUNT"
echo "  extra_overrides : ${EXTRA_OVERRIDES:-<none>}"
echo "==================================================="

# ---- Submit chain ----
SUBMIT_ARGS=(
    --config "$CONFIG"
    --max_steps "$MAX_STEPS"
    --n_jobs "$N_JOBS"
    --nodes "$NODES"
    --account "$ACCOUNT"
)
[[ -n "$EXTRA_OVERRIDES" ]] && SUBMIT_ARGS+=(--extra_overrides "$EXTRA_OVERRIDES")

bash "$WORK_DIR/submit_single_teacher_chain.sh" "${SUBMIT_ARGS[@]}"

# ---- Launch auto-eval watcher in background ----
mkdir -p "$WORK_DIR/logs"
LOG="$WORK_DIR/logs/watcher_${TAG}.log"
PIDFILE="$WORK_DIR/logs/watcher_${TAG}.pid"

if [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo ""
    echo "[warn] auto-eval watcher already running (PID $(cat "$PIDFILE")) — leaving as-is."
else
    nohup bash "$WORK_DIR/auto_eval_watcher.sh" "$RUN_NAME" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 2
    if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo ""
        echo "Auto-eval watcher started:"
        echo "  PID  : $(cat "$PIDFILE")"
        echo "  log  : $LOG"
        echo "  state: $WORK_DIR/logs/auto_eval_watcher_${RUN_NAME}.state"
        echo "  stop : kill \$(cat $PIDFILE)"
    else
        echo ""
        echo "[error] watcher failed to start — see $LOG"
        exit 1
    fi
fi
