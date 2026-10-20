#!/usr/bin/env python
# coding: utf-8
"""Dual-GPU queue launcher for the draft-paper segmentation tables (5 & 6).

Runs the full comparison matrix -- datasets {BRISC, CardiacUDA-G, CardiacUDA-R}
x models {DualMappingViTSeg, ViTSeg} x configs {(p8,d8), (p16,d8) default,
(p32,d8), (p16,d4), (p16,d16)} -- as two parallel workers, one per GPU.

Interruption safety (server reboots / kills are expected):
    * every run checkpoints each epoch and resumes automatically
      (see segmentation/train_segmentation.py);
    * this launcher is re-entrant: simply run it again and it will
      re-dispatch unfinished runs (finished ones are recognized by
      ``summary.json``), so the whole experiment survives interruption;
    * a crashed run is retried up to --retries times, then marked failed
      (written to the queue log) and skipped so the queue keeps going.

Usage (from the repository root):
    nohup python scripts/run_seg_queue.py > segmentation/runs/_queue/launcher.out 2>&1 &
    python scripts/run_seg_queue.py --status          # progress report
    python scripts/run_seg_queue.py --only BRISC      # restrict datasets
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RUNS_ROOT = REPO_ROOT / "segmentation" / "runs"
QUEUE_DIR = RUNS_ROOT / "_queue"

# ---------------------------------------------------------------- job matrix
CONFIGS = [("p16", "d8"), ("p16", "d4"), ("p16", "d16"), ("p8", "d8"), ("p32", "d8")]
CONFIGS = [(16, 8), (16, 4), (16, 16), (8, 8), (32, 8)]
SYN_VARIANTS = ["dense", "random", "factorized_random"]
SYN_TAG = {"dense": "SynDense", "random": "SynRandom",
           "factorized_random": "SynFactorized", "random_fixed": "SynRandomFixed",
           "static": "SynStatic", "mix": "SynMix"}


def make_jobs(with_syn=False, syn_variants=None, syn_configs="all"):
    """Ordered job list: headline default rows first, then depth/patch variants."""
    dm, vit = "DualMappingViTSeg", "ViTSeg"
    syn_variants = SYN_VARIANTS if syn_variants is None else syn_variants
    jobs = []

    def add(dataset, site, model, patch, depth, priority, variant=None):
        tag = SYN_TAG[variant] if variant else \
            {"DualMappingViTSeg": "DMViT", "ViTSeg": "ViT", "Unet": "UNet"}[model]
        run_name = "{}{}_{}_p{}_d{}".format(
            dataset, "_{}".format(site) if site else "", tag, patch, depth)
        jobs.append(dict(run_name=run_name, dataset=dataset, site=site, model=model,
                         patch=patch, depth=depth, priority=priority, variant=variant))

    # headline rows (default p16/d8) from both tables first
    for i, (ds, site) in enumerate([("BRISC", None), ("CardiacUDA", "G"), ("CardiacUDA", "R")]):
        add(ds, site, dm, 16, 8, 0)
        add(ds, site, vit, 16, 8, 0)
    # remaining configs: cardiac (fast) interleaved before the long BRISC runs
    for ds, site in [("CardiacUDA", "G"), ("CardiacUDA", "R"), ("BRISC", None)]:
        for patch, depth in CONFIGS[1:]:
            add(ds, site, dm, patch, depth, 1)
            add(ds, site, vit, patch, depth, 1)
    if with_syn:
        # synthesizer family for the two table datasets (CardiacUDA-R stays
        # excluded: methodology reference only, see docs/PAPER_MAPPING.md)
        syn_cfgs = CONFIGS if syn_configs == "all" else [(16, 8)]
        for ds, site in [("BRISC", None), ("CardiacUDA", "G")]:
            for patch, depth in syn_cfgs:
                for variant in syn_variants:
                    add(ds, site, "SynthesizerViTSeg", patch, depth, 2, variant=variant)
    return jobs


def run_dir_of(job):
    return RUNS_ROOT / job["run_name"]


def is_done(job):
    return (run_dir_of(job) / "summary.json").exists()


def log(msg):
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(QUEUE_DIR / "launcher.log", "a") as f:
        f.write(line + "\n")


def dispatch(job, gpu):
    cmd = [sys.executable, "segmentation/train_segmentation.py",
           "--dataset", job["dataset"], "--model", job["model"],
           "--patch_size", str(job["patch"]), "--depth", str(job["depth"]),
           "--gpu", str(gpu)]
    if job["site"]:
        cmd += ["--site", job["site"]]
    if job.get("variant"):
        cmd += ["--variant", job["variant"]]
    log(f"GPU{gpu} start {job['run_name']}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT))


def status_report(jobs):
    lines = ["", f"{'run_name':<32} {'status':<12} best_epoch  test mDice/mIoU",
             "-" * 78]
    for job in jobs:
        rd = run_dir_of(job)
        if (rd / "summary.json").exists():
            s = json.loads((rd / "summary.json").read_text())
            t = s.get("test", {})
            lines.append(f"{job['run_name']:<32} {'DONE':<12} "
                         f"{s.get('best_epoch'):<11} "
                         f"{t.get('mDice', float('nan')):.4f} / {t.get('mIoU', float('nan')):.4f}")
        elif (rd / "ckpt_last.pth").exists():
            import torch
            try:
                ck = torch.load(rd / "ckpt_last.pth", map_location="cpu", weights_only=False)
                lines.append(f"{job['run_name']:<32} {'RUNNING':<12} "
                             f"{ck['epoch']} (best {ck.get('best_miou', -1):.4f})")
            except Exception:
                lines.append(f"{job['run_name']:<32} {'RUNNING':<12} ?")
        elif (rd / "run.lock").exists():
            lines.append(f"{job['run_name']:<32} {'STARTED':<12}")
        else:
            lines.append(f"{job['run_name']:<32} {'PENDING':<12}")
    print("\n".join(lines))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to datasets (BRISC, CardiacUDA)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    parser.add_argument("--with_syn", action="store_true",
                        help="append the Synthesizer family (SynthesizerViTSeg) jobs")
    parser.add_argument("--syn_variants", nargs="+", default=SYN_VARIANTS,
                        choices=list(SYN_TAG),
                        help="synthesizer variants to queue (with --with_syn)")
    parser.add_argument("--syn_configs", choices=["all", "default"], default="all",
                        help="syn configs: all five, or just the p16/d8 headline")
    parser.add_argument("--max_hours", type=float, default=None,
                        help="stop dispatching new jobs after this many hours")
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
    t_start = time.time()
    slots = {gpu: None for gpu in args.gpus}   # gpu -> (job, proc, started_at)
    retry_counts = {j["run_name"]: 0 for j in jobs}

    while pending or any(v is not None for v in slots.values()):
        progressed = False
        for gpu, busy in list(slots.items()):
            if busy is not None:
                job, proc, t0 = busy
                rc = proc.poll()
                if rc is None:
                    continue
                dur = (time.time() - t0) / 60.0
                if rc == 0 and not is_done(job):
                    # clean exit without summary (e.g. killed mid-run via signal)
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
                if args.max_hours and (time.time() - t_start) / 3600.0 > args.max_hours:
                    continue
                job = pending.pop(0)
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
