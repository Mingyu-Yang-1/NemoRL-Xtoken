#!/bin/bash
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00:30:00
#SBATCH --job-name=smoke_chat
#SBATCH --output=logs/smoke_chat_%j.out
#SBATCH --error=logs/smoke_chat_%j.err

# CPU-only smoke test for the new chat-mode collator + offset_cluster_decode_fix.
# Submits inside the same container all the eval slurms use.

set -euo pipefail

WORKDIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/mingyyang/xtoken_nemorl_github_latest
mkdir -p ${WORKDIR}/logs

export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN required (export from shell before submission)}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_DATASETS_TOKEN="$HF_TOKEN"

CONTAINER=/lustre/fsw/portfolios/coreai/users/mingyyang/containers/nemo_rl_0.6.0.sqsh

srun --ntasks-per-node=1 \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${WORKDIR}" \
    bash -c "
        set -euo pipefail
        cd ${WORKDIR}
        # The nemo_rl_0.6.0 container uses uv to manage the runtime venv.
        # Point UV's cache + venv build dir to lustre so we have ample
        # space (root tmpfs is only ~2 GB; uv pulls ~5 GB of nemo-rl deps).
        export UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/mingyyang/uv_cache_smoke
        export TMPDIR=/lustre/fsw/portfolios/coreai/users/mingyyang/uv_cache_smoke/tmp
        mkdir -p \$UV_CACHE_DIR \$TMPDIR
        uv run python ${WORKDIR}/scripts/smoke_chat_collator.py
    "
