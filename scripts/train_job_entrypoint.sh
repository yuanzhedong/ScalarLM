#!/bin/bash

# Thin shell wrapper: set env vars then exec mpirun. The `exec`
# replaces the batch shell with mpirun in place — the slurm batch
# shell's PID becomes mpirun's, so slurm's `--signal=B:TERM@N`
# (sent to the batch shell) lands on mpirun directly. mpirun's
# standard SIGTERM forwarding then propagates to each rank's python
# process, whose handler in main.py sets the stop_flag. No bash trap,
# no Python wrapper — keeping the sbatch → mpirun → main.py path
# stock avoids cross-system behavior drift.

set -Eeuoxa pipefail

export CRAY_TRAINING_JOB_CONFIG_PATH=REPLACE_CONFIG_PATH

# expandable_segments uses growable virtual address ranges so freed
# blocks of one size can satisfy a later allocation of a different
# size — without it, gradient checkpointing's recompute pattern
# fragments the caching allocator and reserved memory grows step
# over step (especially on Gemma-4-class models with alternating
# sliding/full-attention activation shapes). PyTorch 2.1+.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOCAL_DIRECTORY="$( cd "$( dirname "${CRAY_TRAINING_JOB_CONFIG_PATH}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="${LOCAL_DIRECTORY}/ml:${PYTHONPATH:-}"

# Slurm gres accounting does not know about the co-located vLLM server, so a
# job can be handed the GPU the inference engine already occupies and OOM.
# Re-pick the N GPUs with the most free memory (N = the job config gpus count).
# No-op on deployments without nvidia-smi (CPU, ROCm).
if command -v nvidia-smi >/dev/null 2>&1; then
    NGPUS=$(python -c "import yaml; print(yaml.safe_load(open('${CRAY_TRAINING_JOB_CONFIG_PATH}')).get('gpus', 1))")
    GPU_PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -rn | head -"${NGPUS}" | cut -d, -f1 | paste -sd, || true)
    # Only trust a comma-separated list of indices — nvidia-smi can emit
    # "Failed to initialize NVML: Unknown Error" on stdout (e.g. after the
    # host revokes container device access on a systemd reload), and
    # exporting that as CUDA_VISIBLE_DEVICES crashes the job.
    case "${GPU_PICK}" in
        *[!0-9,]*|"")
            echo "GPU auto-select skipped (nvidia-smi output unusable: ${GPU_PICK})"
            ;;
        *)
            export CUDA_VISIBLE_DEVICES="${GPU_PICK}"
            echo "Selected CUDA_VISIBLE_DEVICES=${GPU_PICK} (freest ${NGPUS} GPUs)"
            ;;
    esac
fi

exec mpirun --allow-run-as-root python "${LOCAL_DIRECTORY}/ml/cray_megatron/main.py" "$@"
