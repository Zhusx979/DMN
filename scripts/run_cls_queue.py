#!/usr/bin/env python
# coding: utf-8
"""Dual-GPU queue launcher for the draft-paper classification tables (2 & 3).

Experiment matrix (20 runs by default):
    datasets {tbx11k, bc} x models {DualMappingViT, ViT} x
    configs {(p16,d8) = Table 2 headline, (p8,d8), (p32,d8), (p16,d4),
    (p16,d16) = Table 3 ablations}

With --with_syn the Synthesizer family (SynthesizerViT, variants dense /
random / factorized_random) is appended as extra jobs: every config by
default, or only the p16/d8 headline with --syn_configs default.

Each run is executed by the dedicated trainer (classification/
train_tuberculosis.py / train_breast_cancer.py) with --patch_size/--depth
overrides; the default run suffix embeds depth+patch so every ablation gets
its own checkpoint folder and auto-resumes there (see utils/training.py).

Interruption safety (same design as run_seg_queue.py):
    * trainers checkpoint every epoch and resume automatically;
    * re-run this launcher any time -- finished runs are recognized by
      ``summary.json`` inside the trainer's ckpt folder and skipped;
    * crashed runs are retried up to --retries times, then skipped.

Usage (from the repository root):
    nohup python scripts/run_cls_queue.py > classification/runs/_queue/launcher.out 2>&1 &
    python scripts/run_cls_queue.py --status              # progress + draft deltas
    python scripts/run_cls_queue.py --only tbx11k        # restrict datasets
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

QUEUE_DIR = REPO_ROOT / "classification" / "runs" / "_queue"

DATASETS = {
    "tbx11k": dict(script="classification/train_tuberculosis.py",
                   ckpt_root=REPO_ROOT / "ckpts" / "tbx11k",
                   label="Tuberculosis"),
    "bc": dict(script="classification/train_breast_cancer.py",
               ckpt_root=REPO_ROOT / "ckpts" / "bc",
               label="Breast Cancer"),
}
MODELS = ["DualMappingViT", "ViT"]
CONFIGS = [(16, 8), (8, 8), (32, 8), (16, 4), (16, 16)]   # (patch, depth)
SYN_VARIANTS = ["dense", "random", "factorized_random"]
SYN_TAG = {"dense": "SynDense", "random": "SynRandom",
           "factorized_random": "SynFactorized", "random_fixed": "SynRandomFixed",
           "static": "SynStatic", "mix": "SynMix"}


def suffix_of(patch, depth, variant=None):
    # must match the trainer's deterministic default: optimizer_depth_patch
    # (synthesizer runs append _syn-<variant> on the trainer side)
    base = "AdamW_depth_{}_patch_{}".format(depth, patch)
    return base if variant is None else "{}_syn-{}".format(base, variant)


def run_dir_of(job):
    return DATASETS[job["dataset"]]["ckpt_root"] / job["model"] / suffix_of(
        job["patch"], job["depth"], job.get("variant"))


def make_jobs(with_syn=False, syn_variants=None, syn_configs="all"):
    jobs = []
    syn_variants = SYN_VARIANTS if syn_variants is None else syn_variants

    def add(dataset, model, patch, depth, table, variant=None):
        tag = SYN_TAG[variant] if variant else {"DualMappingViT": "DMViT", "ViT": "ViT"}[model]
        jobs.append(dict(run_name="{}_{}_p{}_d{}".format(dataset, tag, patch, depth),
                         dataset=dataset, model=model, patch=patch, depth=depth,
                         table=table, variant=variant))

    # dataset-major: all tbx11k rows (Table 2, then Table 3) before bc.  tb
    # runs are ~7x faster, so the fast sweep finishes first and its table can
    # be written while the slow bc runs continue.
    for ds in DATASETS:
        for model in MODELS:
            add(ds, model, 16, 8, "T2")
        for patch, depth in CONFIGS[1:]:
            for model in MODELS:
                add(ds, model, patch, depth, "T3")
    if with_syn:
        # synthesizer family: headline config first, then the remaining ones
        syn_cfgs = CONFIGS if syn_configs == "all" else [(16, 8)]
        for patch, depth in syn_cfgs:
            for ds in DATASETS:
                for variant in syn_variants:
                    add(ds, "SynthesizerViT", patch, depth, "SYN", variant=variant)
    return jobs


def is_done(job):
    return (run_dir_of(job) / "summary.json").exists()


def is_running_externally(job):
    """True if this job's run.lock is held by a live trainer process.

    Lets several launcher instances share the job list safely (and lets a new
    launcher adopt runs started by an older instance): a job whose trainer is
    already alive is never double-dispatched -- that would only crash against
    the single-writer lock and burn retries.
    """
    lock = run_dir_of(job) / "run.lock"
    if not lock.exists():
        return False
    try:
        pid = int(lock.read_text().split()[0])
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, OSError):
        return False


def draft_target(job):
    from configs.experiments import EXPERIMENTS, DEEPBC_CLASSIFICATION, TBX11K_CLASSIFICATION
    exp = TBX11K_CLASSIFICATION if job["dataset"] == "tbx11k" else DEEPBC_CLASSIFICATION
    if job["table"] == "T2":
        return exp.get("draft_acc", {}).get(job["model"])
    key = ("patch", job["patch"]) if job["depth"] == 8 else ("depth", job["depth"])
    return exp.get("draft_table3", {}).get(key, {}).get(job["model"])


def log(msg):
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(QUEUE_DIR / "launcher.log", "a") as f:
        f.write(line + "\n")


def dispatch(job, gpu):
    ds = DATASETS[job["dataset"]]
    cmd = [sys.executable, ds["script"], "--network", job["model"],
           "--patch_size", str(job["patch"]), "--depth", str(job["depth"]),
           "--gpu", str(gpu)]
    if job.get("variant") is not None:
        cmd += ["--syn_variant", job["variant"]]
    log(f"GPU{gpu} start {job['run_name']}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT))


def status_report(jobs):
    lines = ["", f"{'run_name':<22} {'status':<10} {'test acc':>9} {'draft':>7} {'delta':>7}",
             "-" * 60]
    for job in jobs:
        rd = run_dir_of(job)
        target = draft_target(job)
        t = f"{target:>7.4f}" if target else "      -"
        if (rd / "summary.json").exists():
            s = json.loads((rd / "summary.json").read_text())
            acc = s["test_acc"]
            d = f"{acc - target:>+7.4f}" if target else "      -"
            lines.append(f"{job['run_name']:<22} {'DONE':<10} {acc:>9.4f} {t} {d}")
        elif (rd / "run.lock").exists():
            lines.append(f"{job['run_name']:<22} {'RUNNING':<10} {'-':>9} {t}")
        else:
            lines.append(f"{job['run_name']:<22} {'PENDING':<10} {'-':>9} {t}")
    done = sum(is_done(j) for j in jobs)
    lines.append(f"\n{done}/{len(jobs)} runs finished")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--only", nargs="*", default=None, choices=list(DATASETS),
                        help="restrict to datasets (tbx11k, bc)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    parser.add_argument("--with_syn", action="store_true",
                        help="append the Synthesizer family (SynthesizerViT) jobs")
    parser.add_argument("--syn_variants", nargs="+", default=SYN_VARIANTS,
                        choices=list(SYN_TAG),
                        help="synthesizer variants to queue (with --with_syn)")
    parser.add_argument("--syn_configs", choices=["all", "default"], default="all",
                        help="syn configs: all five, or just the p16/d8 headline")
    args = parser.parse_args()

    jobs = make_jobs(args.with_syn, args.syn_variants, args.syn_configs)
    if args.only:
        jobs = [j for j in jobs if j["dataset"] in args.only]

    if args.status:
        status_report(jobs)
        return

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"launcher started: {len(jobs)} jobs, gpus {args.gpus}, "
        f"{sum(is_done(j) for j in jobs)} already done")

    pending = [j for j in jobs if not is_done(j)]
    failed = {}
    slots = {gpu: None for gpu in args.gpus}   # gpu -> (job, proc, started_at)
    retry_counts = {j["run_name"]: 0 for j in jobs}

    while pending or any(v is not None for v in slots.values()):
        progressed = False
        # adopt completions produced by trainers owned by another launcher
        still = []
        for j in pending:
            if is_done(j):
                log(f"external DONE {j['run_name']}")
            else:
                still.append(j)
        pending = still
        for gpu, busy in list(slots.items()):
            if busy is not None:
                job, proc, t0 = busy
                rc = proc.poll()
                if rc is None:
                    continue
                dur = (time.time() - t0) / 60.0
                if rc == 0 and not is_done(job):
                    log(f"GPU{gpu} exit 0 {job['run_name']} ({dur:.1f} min) -- "
                        f"not finished, will resume")
                    pending.insert(0, job)
                elif rc == 0:
                    log(f"GPU{gpu} DONE {job['run_name']} ({dur:.1f} min)")
                else:
                    retry_counts[job["run_name"]] += 1
                    n = retry_counts[job["run_name"]]
                    if n > args.retries:
                        failed[job["run_name"]] = rc
                        log(f"GPU{gpu} FAILED {job['run_name']} rc={rc} after {n} "
                            f"attempts -- skipped")
                        (QUEUE_DIR / "failed.txt").open("a").write(
                            f"{job['run_name']} rc={rc} {datetime.datetime.now()}\n")
                    else:
                        log(f"GPU{gpu} crashed {job['run_name']} rc={rc} "
                            f"({dur:.1f} min) -- retry {n}/{args.retries} (resumes)")
                        pending.insert(0, job)
                slots[gpu] = None
                progressed = True
            if slots[gpu] is None and pending:
                # skip jobs whose trainer is already alive (started by another
                # launcher or an earlier instance): double-dispatch would only
                # crash against the single-writer lock and burn retries
                idx = next((i for i, j in enumerate(pending)
                            if not is_running_externally(j)), None)
                if idx is not None:
                    job = pending.pop(idx)
                    proc = dispatch(job, gpu)
                    slots[gpu] = (job, proc, time.time())
                    progressed = True
        if not progressed:
            time.sleep(10)

    log("queue finished")
    status_report(jobs)
    if failed:
        log(f"failed jobs: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
