"""Optional Weights & Biases logging for training jobs.

Every function is a no-op unless WANDB_API_KEY is set in the environment and
the wandb package is importable, so deployments without W&B are unaffected.
Only the main rank logs (mpirun duplicates the process per rank).
"""

import logging
import os

logger = logging.getLogger(__name__)

_run = None


def _is_main_rank():
    return os.environ.get("OMPI_COMM_WORLD_RANK", "0") == "0"


def _enabled():
    return bool(os.environ.get("WANDB_API_KEY")) and _is_main_rank()


def wandb_init(job_config):
    global _run
    if not _enabled():
        return
    try:
        import wandb

        job_dir = job_config.get("job_directory", "")
        run_name = os.path.basename(job_dir)[:12] or None
        _run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "scalarlm"),
            name=run_name,
            config={
                k: v
                for k, v in job_config.items()
                if isinstance(v, (str, int, float, bool))
            },
            resume="allow",
            id=run_name,
        )
        logger.info(f"wandb run initialized: {run_name}")
    except Exception as e:  # never let telemetry break training
        logger.warning(f"wandb init failed (continuing without): {e}")
        _run = None


def wandb_log(metrics, step=None):
    if _run is None:
        return
    try:
        _run.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"wandb log failed: {e}")


def wandb_finish():
    global _run
    if _run is None:
        return
    try:
        _run.finish()
    except Exception as e:
        logger.warning(f"wandb finish failed: {e}")
    _run = None
