#!/bin/bash
# Watch a training run's checkpoint dir and submit consolidate+eval SLURM jobs
# for new checkpoints at a configurable cadence.
#
# Each step that's a multiple of EVAL_EVERY (default 500) gets exactly one eval
# job; submissions are recorded in a state file so the watcher is restartable.
#
# Usage:
#   bash auto_eval_watcher.sh <run_name> [eval_every]
#   # or via env vars:
#   EVAL_EVERY=500 MAX_STEPS=10000 POLL_INTERVAL=120 \
#       bash auto_eval_watcher.sh <run_name>
#
# Recommended: launch in nohup so it survives login-node SSH disconnects:
#   mkdir -p logs
#   nohup bash auto_eval_watcher.sh pkl-qwen14b-base-10k > logs/watcher.log 2>&1 &
#   echo $!  # pid; kill with `kill <pid>` when done

set -euo pipefail

RUN_NAME="${1:?Usage: $0 <run_name> [eval_every]}"
EVAL_EVERY="${2:-${EVAL_EVERY:-500}}"
MAX_STEPS="${MAX_STEPS:-10000}"
POLL_INTERVAL="${POLL_INTERVAL:-120}"   # seconds between scans
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_GPUS_PER_NODE="${EVAL_GPUS_PER_NODE:-8}"
EVAL_TIME="${EVAL_TIME:-2:00:00}"
DELETE_AFTER_EVAL="${DELETE_AFTER_EVAL:-0}"  # default OFF — was racing with chained-job auto-resume

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="${WORK_DIR}/checkpoints/${RUN_NAME}"
STATE_FILE="${WORK_DIR}/logs/auto_eval_watcher_${RUN_NAME}.state"
LOG_FILE="${WORK_DIR}/logs/auto_eval_watcher_${RUN_NAME}.log"

mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

log "=== auto_eval_watcher started ==="
log "  RUN_NAME      : ${RUN_NAME}"
log "  EVAL_EVERY    : ${EVAL_EVERY}"
log "  MAX_STEPS     : ${MAX_STEPS}"
log "  POLL_INTERVAL : ${POLL_INTERVAL}s"
log "  BATCH_SIZE    : ${BATCH_SIZE}"
log "  EVAL_GPUS     : ${EVAL_GPUS_PER_NODE}"
log "  DELETE_AFTER_EVAL : ${DELETE_AFTER_EVAL}"
log "  CKPT_DIR      : ${CKPT_DIR}"
log "  STATE_FILE    : ${STATE_FILE}"

# Marker file inside each checkpoint that confirms it's fully written.
# NeMo-RL renames tmp_step_N -> step_N at the end of save, so existence of
# step_N is *usually* enough, but check for the metadata file inside too.
ckpt_complete() {
    local d="$1"
    [[ -f "${d}/policy/weights/model/.hf_metadata/config.json" ]]
}

last_submitted_step=-1

while true; do
    if [[ -d "${CKPT_DIR}" ]]; then
        # Iterate steps in numeric order so logs are sane on first scan.
        mapfile -t step_dirs < <(
            find "${CKPT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'step_*' \
                -printf '%f\n' 2>/dev/null \
            | sort -t'_' -k2 -n
        )

        for step in "${step_dirs[@]}"; do
            step_num="${step#step_}"
            # Only multiples of EVAL_EVERY
            if (( step_num % EVAL_EVERY != 0 )); then
                continue
            fi
            # Skip if already submitted
            if grep -qE "^${step}\s" "${STATE_FILE}"; then
                continue
            fi
            # Ensure checkpoint is complete
            if ! ckpt_complete "${CKPT_DIR}/${step}"; then
                log "  ${step}: not yet complete, waiting"
                continue
            fi
            # Submit
            log "Submitting eval for ${step}"
            JID=$(sbatch --parsable \
                --gpus-per-node="${EVAL_GPUS_PER_NODE}" \
                --time="${EVAL_TIME}" \
                --export=ALL,RUN_NAME=${RUN_NAME},STEP=${step},BATCH_SIZE=${BATCH_SIZE},HF_OFFLINE=1,DELETE_AFTER_EVAL=${DELETE_AFTER_EVAL} \
                "${WORK_DIR}/consolidate_and_eval.slurm")
            printf '%s\t%s\t%s\n' "${step}" "${JID}" "$(date +%s)" >> "${STATE_FILE}"
            log "  ${step} -> job ${JID}"
            last_submitted_step=$step_num

            if (( step_num >= MAX_STEPS )); then
                log "Reached MAX_STEPS=${MAX_STEPS} at ${step}. Watcher exiting."
                exit 0
            fi
        done
    fi
    sleep "${POLL_INTERVAL}"
done
