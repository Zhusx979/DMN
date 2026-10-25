#!/usr/bin/env python
# coding: utf-8
"""Exact 12-job Synthesizer queue after the breast-cancer runs.

Paper main comparison only:
    * default patch/depth: p16/d8;
    * variants: dense, random, factorized_random;
    * classification: tbx11k + bc x 3 = 6;
    * segmentation: BRISC + CardiacUDA-G x 3 = 6.

The launcher waits until all 10 base BC runs have summary.json, then runs the
available jobs on free GPUs.  A dataset whose files are not present is marked
BLOCKED; the launcher keeps polling and starts its jobs as soon as the data
appears, without touching the other jobs.  The CardiacUDA-R site is excluded
(methodology reference only, see docs/PAPER_MAPPING.md).
"""

import argparse
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_cls_queue as cls_q  # noqa: E402
import run_seg_queue as seg_q  # noqa: E402
from configs import dataset_paths as dp  # noqa: E402
from configs.experiments import EXPERIMENTS  # noqa: E402

QUEUE_DIR = REPO_ROOT / "classification" / "runs" / "_queue"
LOG_DIR = REPO_ROOT / "logs" / "syn"
STATE_PATH = QUEUE_DIR / "syn_jobs.json"
LAUNCHER_LOG = QUEUE_DIR / "syn_launcher.log"
LOCK_PATH = QUEUE_DIR / "syn_queue.lock"
VARIANTS = ["dense", "random", "factorized_random"]
PATCH, DEPTH = 16, 8


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg):
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    line = "[{}] {}".format(now(), msg)
    print(line, flush=True)
    with open(LAUNCHER_LOG, "a") as f:
        f.write(line + "\n")


def _has_files(path, suffixes):
    path = Path(path)
    if not path.is_dir():
        return False
    return any(p.is_file() and p.suffix.lower() in suffixes for p in path.iterdir())


def dataset_ready(key):
    """Return (ready, detail) for one of the four Synthesizer datasets."""
    if key == "bc":
        root = Path(dp.DEEPBC_ROOT)
        labels = root / "label_classification"
        need = [labels / "training.txt", labels / "val.txt", labels / "test.txt",
                root / "data"]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            return False, "missing " + ", ".join(missing)
        return True, str(root)

    if key == "tbx11k":
        root = Path(dp.TBX11K_ROOT)
        if not root.is_dir():
            return False, "missing {}".format(root)
        for cls in ("health", "sick", "tb"):
            cls_dir = root / cls
            if not cls_dir.is_dir():
                return False, "missing {}".format(cls_dir)
            n = sum(1 for p in cls_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if n < 800:
                return False, "{} has {} images (need >=800)".format(cls_dir, n)
        return True, str(root)

    if key == "BRISC":
        root = Path(dp.BRISC_ROOT)
        for split in ("train", "test"):
            for sub in ("images", "masks"):
                d = root / split / sub
                if not _has_files(d, {".jpg", ".jpeg", ".png"}):
                    return False, "missing/empty {}".format(d)
        n_train = sum(1 for p in (root / "train" / "images").iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        n_test = sum(1 for p in (root / "test" / "images").iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if (n_train, n_test) != (3933, 860):
            return False, "found train/test images {}/{}, expected 3933/860".format(
                n_train, n_test)
        return True, str(root)

    if key == "CardiacUDA":
        prepared = Path(dp.CARDIACUDC_PREPARED_ROOT)
        splits = EXPERIMENTS["CardiacUDA"]["site_splits"]["G"]
        for split_dir in splits.values():
            d = prepared / split_dir
            if not d.is_dir() or not any(d.glob("*.npz")):
                return False, "missing prepared G split {}".format(d)
        return True, str(prepared)

    raise KeyError(key)


def build_jobs():
    jobs = []
    for job in cls_q.make_jobs(with_syn=True, syn_variants=VARIANTS,
                               syn_configs="default"):
        if job.get("variant") is None:
            continue
        job = dict(job)
        job.update(kind="classification", script=cls_q.DATASETS[job["dataset"]]["script"],
                   run_dir=cls_q.run_dir_of(job))
        jobs.append(job)
    for job in seg_q.make_jobs(with_syn=True, syn_variants=VARIANTS,
                               syn_configs="default"):
        if job.get("variant") is None:
            continue
        job = dict(job)
        job.update(kind="segmentation", script="segmentation/train_segmentation.py",
                   run_dir=seg_q.run_dir_of(job))
        jobs.append(job)

    order = {"bc": 0, "tbx11k": 1, "BRISC": 2, "CardiacUDA": 3}
    jobs.sort(key=lambda j: (order[j["dataset"]], VARIANTS.index(j["variant"])))
    assert len(jobs) == 12, "expected 12 Synthesizer jobs, built {}".format(len(jobs))
    assert all(j["patch"] == PATCH and j["depth"] == DEPTH for j in jobs)
    assert all(j["variant"] in VARIANTS for j in jobs)
    return jobs


def is_done(job):
    return (job["run_dir"] / "summary.json").exists()


def is_running_externally(job):
    lock = job["run_dir"] / "run.lock"
    if not lock.exists():
        return False
    try:
        pid = int(lock.read_text().split()[0])
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, OSError):
        return False


def gpu_used_mb(gpu):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=15)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def gpu_compute_pids(gpu):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
             "--format=csv,noheader"], text=True, timeout=15)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return None


def gpu_ready(gpu, max_used_mb):
    """A GPU is usable only when it has no compute process at all.

    This keeps the queue off GPU1 while the base ViT d16 run is still
    training, even though that run uses only ~5.5 GB.
    """
    used = gpu_used_mb(gpu)
    pids = gpu_compute_pids(gpu)
    if used is None or pids is None:
        return False, "nvidia-smi unavailable"
    if pids:
        return False, "compute pids {}".format(",".join(pids))
    if used > max_used_mb:
        return False, "{} MiB already used".format(used)
    return True, ""


def bc_base_jobs():
    jobs = []
    for job in cls_q.make_jobs(with_syn=False):
        if job["dataset"] != "bc":
            continue
        job = dict(job)
        job["run_dir"] = cls_q.run_dir_of(job)
        jobs.append(job)
    return jobs


def wait_for_bc(poll):
    jobs = bc_base_jobs()
    log("waiting for 10 base BC runs before starting the 12 Synthesizer jobs")
    last = 0.0
    while True:
        missing = [j["run_name"] for j in jobs if not is_done(j)]
        running = [j["run_name"] for j in jobs if is_running_externally(j)]
        if not missing and not running:
            log("base BC runs complete; starting Synthesizer queue")
            return
        if time.time() - last > 600:
            log("BC wait: {} done, {} running, missing={}".format(
                len(jobs) - len(missing), len(running), missing))
            last = time.time()
        time.sleep(poll)


def dispatch(job, gpu):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "{}.out".format(job["run_name"])
    if job["kind"] == "classification":
        cmd = [sys.executable, "-u", job["script"],
               "--network", "SynthesizerViT", "--syn_variant", job["variant"],
               "--patch_size", str(PATCH), "--depth", str(DEPTH),
               "--gpu", str(gpu)]
    else:
        cmd = [sys.executable, "-u", job["script"],
               "--dataset", job["dataset"], "--model", "SynthesizerViTSeg",
               "--patch_size", str(PATCH), "--depth", str(DEPTH),
               "--gpu", str(gpu), "--variant", job["variant"]]
        if job.get("site"):
            cmd += ["--site", job["site"]]
    env = os.environ.copy()
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    log("GPU{} start {}: {}".format(gpu, job["run_name"], " ".join(cmd)))
    fh = open(log_path, "ab")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                            stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    fh.close()
    return proc


def save_state(jobs, slots, blocked, started_at):
    payload = {
        "updated": now(),
        "expected_jobs": 12,
        "started_at": started_at,
        "done": sum(is_done(j) for j in jobs),
        "jobs": [],
    }
    for job in jobs:
        item = {k: job.get(k) for k in (
            "run_name", "dataset", "site", "variant", "kind", "patch", "depth")}
        item["run_dir"] = str(job["run_dir"])
        item["status"] = ("DONE" if is_done(job) else
                          "RUNNING" if any(s and s[0]["run_name"] == job["run_name"]
                                           for s in slots.values()) else
                          "BLOCKED" if job["run_name"] in blocked else "PENDING")
        if job["run_name"] in blocked:
            item["reason"] = blocked[job["run_name"]]
        payload["jobs"].append(item)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, STATE_PATH)


def status_report(jobs):
    lines = ["", "{:<22} {:<10} {}".format("run_name", "status", "test_acc / detail"),
             "-" * 78]
    for job in jobs:
        s = job["run_dir"] / "summary.json"
        ready, detail = dataset_ready(job["dataset"])
        if s.exists():
            data = json.loads(s.read_text())
            if job["kind"] == "classification":
                val = "test_acc {}".format(data.get("test_acc"))
            else:
                val = "mDice {} / mIoU {}".format(
                    data.get("test", {}).get("mDice"),
                    data.get("test", {}).get("mIoU"))
            lines.append("{:<22} {:<10} {}".format(job["run_name"], "DONE", val))
        elif is_running_externally(job):
            lines.append("{:<22} {:<10} {}".format(job["run_name"], "RUNNING", ""))
        elif not ready:
            lines.append("{:<22} {:<10} {}".format(job["run_name"], "BLOCKED", detail))
        else:
            lines.append("{:<22} {:<10} {}".format(job["run_name"], "PENDING", detail))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--max-used-mb", type=int, default=20000,
                        help="do not dispatch on a GPU already using more than this")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--wait-for-bc", dest="wait_for_bc", action="store_true",
                        default=True)
    parser.add_argument("--no-wait-for-bc", dest="wait_for_bc", action="store_false")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs()
    if args.status:
        status_report(jobs)
        return

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    lock_fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another run_syn_queue.py instance owns {}".format(LOCK_PATH))
    lock_fh.write("{} {}\n".format(os.getpid(), now()))
    lock_fh.flush()

    if args.wait_for_bc:
        wait_for_bc(args.poll)

    pending = [j for j in jobs if not is_done(j)]
    failed = {}
    retry_counts = {j["run_name"]: 0 for j in jobs}
    slots = {gpu: None for gpu in args.gpus}
    blocked = {}
    started_at = now()
    last_block_log = 0.0

    log("synthesizer queue started: {}/{} already done, gpus {}".format(
        len(jobs) - len(pending), len(jobs), args.gpus))

    while pending or any(v is not None for v in slots.values()):
        progressed = False

        # resolve jobs finished by an earlier launcher / external trainer
        for job in list(pending):
            if is_done(job):
                log("DONE {} (external)".format(job["run_name"]))
                pending.remove(job)
                progressed = True

        # poll our trainers
        for gpu, busy in list(slots.items()):
            if busy is None:
                continue
            job, proc, t0 = busy
            rc = proc.poll()
            if rc is None:
                continue
            dur = (time.time() - t0) / 60.0
            slots[gpu] = None
            progressed = True
            if rc == 0 and is_done(job):
                log("GPU{} DONE {} ({:.1f} min)".format(gpu, job["run_name"], dur))
                continue
            if rc == 0:
                log("GPU{} exit 0 {} ({:.1f} min), no summary; resume".format(
                    gpu, job["run_name"], dur))
                pending.insert(0, job)
                continue
            retry_counts[job["run_name"]] += 1
            n = retry_counts[job["run_name"]]
            if n > args.retries:
                failed[job["run_name"]] = rc
                log("GPU{} FAILED {} rc={} after {} attempts".format(
                    gpu, job["run_name"], rc, n))
                with open(QUEUE_DIR / "failed_syn.txt", "a") as f:
                    f.write("{} rc={} {}\n".format(job["run_name"], rc, now()))
            else:
                log("GPU{} crashed {} rc={} ({:.1f} min); retry {}/{} (resume)".format(
                    gpu, job["run_name"], rc, dur, n, args.retries))
                pending.insert(0, job)

        # dispatch ready jobs; scan past blocked datasets
        for gpu in args.gpus:
            if slots[gpu] is not None or not pending:
                continue
            usable, gpu_detail = gpu_ready(gpu, args.max_used_mb)
            if not usable:
                continue
            chosen = None
            for job in pending:
                if is_running_externally(job):
                    continue
                ready, detail = dataset_ready(job["dataset"])
                if ready:
                    chosen = job
                    blocked.pop(job["run_name"], None)
                    break
                blocked[job["run_name"]] = detail
            if chosen is None:
                continue
            pending.remove(chosen)
            try:
                proc = dispatch(chosen, gpu)
            except Exception as e:
                log("dispatch failed {}: {}".format(chosen["run_name"], e))
                pending.insert(0, chosen)
                continue
            slots[gpu] = (chosen, proc, time.time())
            progressed = True

        save_state(jobs, slots, blocked, started_at)

        if not progressed:
            if blocked and time.time() - last_block_log > 300:
                detail = "; ".join("{}: {}".format(k, v) for k, v in sorted(blocked.items()))
                log("blocked on data: {}".format(detail))
                last_block_log = time.time()
            time.sleep(args.poll)

    log("synthesizer queue finished")
    status_report(jobs)
    save_state(jobs, slots, blocked, started_at)
    if failed:
        log("failed jobs: {}".format(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
