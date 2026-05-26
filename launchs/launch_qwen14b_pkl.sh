#!/bin/bash
# Launch Llama-3.1-8B ← Qwen3-14B-Base (P-KL) distillation chain + auto-eval watcher.
#
# See launchs/launch_phi4_hkl.sh for full docs — same pattern, different teacher/loss.

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORK_DIR"

CONFIG=examples/configs/single_teacher_cross_tokenizer_llama8b_qwen3_14b_arrow.yaml
MAX_STEPS=10000
N_JOBS=5
NODES=32
ACCOUNT=nvr_lpr_llm
EXTRA_OVERRIDES=""

RUN_NAME="$(basename "$CONFIG" .yaml | tr '_' '-')"
TAG=qwen14b_pkl

echo "==================================================="
echo " Launching Llama-8B ← Qwen3-14B-Base (P-KL) distillation"
echo "==================================================="
echo "  config       : $CONFIG"
echo "  run_name     : $RUN_NAME"
echo "  max_steps    : $MAX_STEPS"
echo "  n_jobs       : $N_JOBS"
echo "  nodes/job    : $NODES"
echo "  account      : $ACCOUNT"
echo "  extra_overrides : ${EXTRA_OVERRIDES:-<none>}"
echo "==================================================="

SUBMIT_ARGS=(
    --config "$CONFIG"
    --max_steps "$MAX_STEPS"
    --n_jobs "$N_JOBS"
    --nodes "$NODES"
    --account "$ACCOUNT"
)
[[ -n "$EXTRA_OVERRIDES" ]] && SUBMIT_ARGS+=(--extra_overrides "$EXTRA_OVERRIDES")

bash "$WORK_DIR/submit_single_teacher_chain.sh" "${SUBMIT_ARGS[@]}"

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
