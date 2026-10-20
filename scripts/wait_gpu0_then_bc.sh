#!/bin/bash
# Wait until GPU0 is free (tbx11k queue launcher and its trainer have both
# exited), then dispatch the bc classification queue on GPU0.
#
# Started with setsid so it survives ZCode session cleanup (see the SIGTERM
# incident of 2026-10-18/19). Safe to kill any time: it only launches the
# queue, and the queue itself is re-entrant (summary.json markers + auto-
# resume + lock-aware dispatch).
cd /data1/zhushengxiang/DualMappingNetwork || exit 1
PY=/home/zhushengxiang/miniconda3/envs/deep_learning/bin/python
OUT=classification/runs/_queue/launcher_gpu0.out
mkdir -p classification/runs/_queue

echo "[$(date '+%F %T')] waiter started: will launch bc on GPU0 when tbx11k finishes" >> "$OUT"
# Do NOT poll with `pgrep -f "<script pattern>"` here: the shell command that
# STARTS this waiter can itself stay alive with that pattern in its cmdline
# (hit on 2026-10-19: the launching `bash -c` waited on this script as a
# background child for hours, pgrep matched it, the loop never exited and
# GPU0 idled 44 min after tbx11k finished). Poll the GPU directly instead.
while [ "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do
  sleep 60
done

echo "[$(date '+%F %T')] GPU0 free -- launching bc queue on GPU0" >> "$OUT"
exec "$PY" scripts/run_cls_queue.py --only bc --gpus 0 >> "$OUT" 2>&1
