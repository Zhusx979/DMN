"""Shared training utilities: LR schedule, run paths, interruption-safe checkpointing."""

import math
import os
import random
import signal
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

REPO_ROOT = str(Path(__file__).resolve().parents[1])


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup, then cosine decay.

    Linearly increases the learning rate from 0 to 1 over `warmup_steps` steps,
    then decreases it from 1 to 0 over the remaining `t_total - warmup_steps`
    steps following a cosine curve (paper classification schedule: 30 warmup
    epochs over 600 total epochs).
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda,
                                                   last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def get_settings(dataset, network, suffix):
    """Return (log_path, checkpoint_folder) for a run, anchored at the repo root.

    The parent directory of the log file is created if missing; the caller is
    responsible for creating the checkpoint folder.
    """
    ckpt_folder = os.path.join(REPO_ROOT, 'checkpoint', dataset, network, suffix)
    log_path = os.path.join(REPO_ROOT, 'logs', dataset, network, '{}.log'.format(suffix))
    parent_path = os.path.dirname(log_path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    return log_path, ckpt_folder


# ---------------------------------------------------------------------------
# Interruption-safe checkpointing helpers, shared by the classification
# trainers.  These mirror the logic already proven in
# segmentation/train_segmentation.py (atomic tmp+rename writes, full resume
# state including optimizer/scheduler/RNG, a PID-liveness lock so two runs
# cannot write the same folder, and a graceful SIGINT/SIGTERM stop that
# finishes the current epoch before exiting).
#
# Every classification run keeps exactly TWO files in its checkpoint folder:
#   ckpt_best.pth  -- raw model state_dict of the best-val epoch (for eval)
#   ckpt_last.pth  -- full resume state, rewritten atomically every epoch
# Replacing the old "save epoch{N}.pth on every val increase" behaviour makes
# disk usage O(1) and lets a re-run continue instead of restarting.
# ---------------------------------------------------------------------------

_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True
    print("\n[signal {}] finishing the current epoch, then saving and exiting...".format(signum))


def install_graceful_stop():
    """Register SIGINT/SIGTERM handlers that set the module-level stop flag.

    The training loop checks stop_requested() at the epoch boundary, so an
    interrupt never corrupts a checkpoint mid-write.
    """
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)


def stop_requested():
    return _STOP


def atomic_save(obj, path):
    """torch.save to a .tmp sibling, then os.replace -> no torn checkpoints."""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, str(path))


def acquire_run_lock(ckpt_folder):
    """Single-writer lock: abort if a live process already owns this folder.

    A stale lock (owner pid no longer exists) is overwritten, so a crashed run
    can still be resumed.
    """
    ckpt_folder = Path(ckpt_folder)
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    lock = ckpt_folder / "run.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
            os.kill(pid, 0)  # raises ProcessLookupError if dead
            raise SystemExit(
                "another training process (pid {}) is already running in {}".format(pid, ckpt_folder))
        except (ProcessLookupError, ValueError):
            pass  # stale lock
    lock.write_text("{} {} {}".format(os.getpid(), socket.gethostname(),
                                      datetime.now().isoformat()))


def save_resume_state(ckpt_folder, model, optimizer, scheduler, epoch,
                      best, num_classes, extra=None):
    """Atomically write the full resume state to <ckpt_folder>/ckpt_last.pth.

    `best` is a dict of tracking scalars (e.g. best_val_acc / best_epoch /
    correspond_test_acc) round-tripped verbatim so the loop's reporting after a
    resume still reads correctly. RNG states are saved on CPU because
    torch.set_rng_state requires a CPU ByteTensor.
    """
    ckpt_folder = Path(ckpt_folder)
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,                     # last fully completed epoch
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best": dict(best),
        "num_classes": num_classes,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if extra:
        state["extra"] = dict(extra)
    atomic_save(state, ckpt_folder / "ckpt_last.pth")


def save_best(ckpt_folder, model, filename="ckpt_best.pth"):
    """Atomically write the raw model state_dict for downstream evaluation."""
    ckpt_folder = Path(ckpt_folder)
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    atomic_save(model.state_dict(), ckpt_folder / filename)


def load_resume_state(ckpt_folder, model, optimizer, scheduler, num_classes):
    """Restore a ckpt_last.pth run in place; no-op if the file is absent.

    Returns (start_epoch, best) where start_epoch = last completed epoch + 1.
    map_location="cpu" is required so the saved RNG ByteTensor can be fed back
    to torch.set_rng_state on a GPU machine.
    """
    path = Path(ckpt_folder) / "ckpt_last.pth"
    if not path.exists():
        return 0, {}
    # map_location="cpu": RNG state tensors must stay on CPU for set_rng_state
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["num_classes"] == num_classes, \
        "checkpoint num_classes {} != {}".format(state["num_classes"], num_classes)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["epoch"]) + 1, dict(state.get("best", {}))
