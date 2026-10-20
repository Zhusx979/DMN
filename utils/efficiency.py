"""Efficiency profiling utilities: the five quantities reported for DM ViT.

For a given model and input resolution this module measures

    * total parameters (and trainable parameters)
    * FLOPs -- per-image forward FLOPs and per-image training-step FLOPs
      (forward + backward, counted at the operator level with
      torch.utils.flop_counter.FlopCounterMode)
    * GPU memory -- peak allocated memory during inference and during one
      full training step (forward + backward + AdamW step)
    * inference latency (milliseconds per image)
    * throughput (images per second)

CUDA timings use torch.cuda.Event with synchronisation; on CPU the module
falls back to wall-clock timing and returns None for the GPU-memory entries.
Dropout stays in eval mode during all measurements, dummy gradients are
zeroed before returning, so a model can be profiled between training epochs
without side effects.  Randomness is seeded for repeatable tables.

Usage
-----
    from utils import efficiency
    res = efficiency.benchmark_model(model, (3, 224, 224), device="cuda")
    efficiency.log_efficiency_summary(model, (3, 224, 224), device, logger)

or from the command line:  python benchmarks/benchmark_efficiency.py
"""

import time

import torch
from torch.utils.flop_counter import FlopCounterMode

__all__ = [
    "count_parameters",
    "count_flops",
    "measure_inference",
    "measure_training_step_memory",
    "benchmark_model",
    "format_table",
    "log_efficiency_summary",
]


def count_parameters(model):
    """Return {'total': int, 'trainable': int} parameter counts.

    'total' counts parameters AND non-learnable buffers (e.g. the frozen
    random matrices of SynthesizerViT_random_fixed -- the ICML-2021 paper
    counts its fixed N x N matrices as parameters too).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return {"total": trainable + buffers, "trainable": trainable}


def count_flops(model, input_shape, device="cuda", batch_size=1, mode="forward"):
    """Per-image FLOPs: mode='forward' (inference) or 'train' (fwd + bwd).

    FLOPs are counted for the ops actually executed (matmuls, convolutions,
    interpolations, ...), so the numbers are exact for the executed graph.
    """
    if mode not in ("forward", "train"):
        raise ValueError("mode must be 'forward' or 'train'")
    device = torch.device(device)
    was_training = model.training
    model.eval()
    if next(model.parameters()).device != device:
        model.to(device)
    x = torch.randn(batch_size, *input_shape, device=device)
    with FlopCounterMode(display=False) as counter:
        if mode == "train":
            model.train()
            out = model(x)
            out.sum().backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                model(x)
    model.train(was_training)
    return counter.get_total_flops() / batch_size


def measure_inference(model, input_shape, device="cuda", batch_size=1,
                      warmup=20, iters=100):
    """Latency per image, throughput and peak GPU memory during inference."""
    device = torch.device(device)
    model.eval()
    if next(model.parameters()).device != device:
        model.to(device)
    x = torch.randn(batch_size, *input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                model(x)
            end.record()
            torch.cuda.synchronize(device)
            total_s = start.elapsed_time(end) / 1e3
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                model(x)
            total_s = time.perf_counter() - t0
            peak_gb = None
    images = batch_size * iters
    return {
        "latency_ms_per_image": total_s * 1e3 / images,
        "throughput_img_per_s": images / total_s,
        "peak_gpu_mem_gb": peak_gb,
        "batch_size": batch_size,
    }


def measure_training_step_memory(model, input_shape, device="cuda", batch_size=16):
    """Peak GPU memory of one full training step (fwd + bwd + AdamW step).

    AdamW is used because it allocates the same two optimizer-state tensors
    per parameter as the paper's training runs.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return {"peak_gpu_mem_train_step_gb": None, "train_batch_size": batch_size}
    was_training = model.training
    model.train()
    if next(model.parameters()).device != device:
        model.to(device)
    x = torch.randn(batch_size, *input_shape, device=device)
    # lr=0: the dummy step allocates the AdamW state tensors (memory) without
    # changing any weight, so profiling never perturbs the model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = model(x)
    loss = out.float().sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return {"peak_gpu_mem_train_step_gb": peak_gb, "train_batch_size": batch_size}


def benchmark_model(model, input_shape, device="cuda", latency_batch=1,
                    throughput_batch=32, warmup=20, iters=100, train_batch=16,
                    flops_train=True):
    """All five quantities in one call; returns one flat dict.

    Keys: params_total, params_trainable, flops_forward, flops_train,
    latency_ms_per_image, peak_gpu_mem_infer_gb (latency_batch),
    throughput_img_per_s, throughput_batch_size, peak_gpu_mem_tp_gb,
    peak_gpu_mem_train_step_gb, train_batch_size.
    """
    params = count_parameters(model)
    res = {
        "params_total": params["total"],
        "params_trainable": params["trainable"],
        "flops_forward": count_flops(model, input_shape, device, mode="forward"),
    }
    if flops_train:
        res["flops_train"] = count_flops(model, input_shape, device, mode="train")

    lat = measure_inference(model, input_shape, device,
                            batch_size=latency_batch, warmup=warmup, iters=iters)
    res["latency_ms_per_image"] = lat["latency_ms_per_image"]
    res["peak_gpu_mem_infer_gb"] = lat["peak_gpu_mem_gb"]

    thr = measure_inference(model, input_shape, device,
                            batch_size=throughput_batch,
                            warmup=max(1, warmup // 2), iters=max(5, iters // 4))
    res["throughput_img_per_s"] = thr["throughput_img_per_s"]
    res["throughput_batch_size"] = throughput_batch
    res["peak_gpu_mem_tp_gb"] = thr["peak_gpu_mem_gb"]

    res.update(measure_training_step_memory(model, input_shape, device,
                                            batch_size=train_batch))
    return res


def _fmt(value, spec="{:.2f}"):
    return "-" if value is None else spec.format(value)


def format_table(results):
    """results: {name: dict from benchmark_model} -> markdown table string."""
    header = ("| Model | Params (M) | FLOPs fwd (G/img) | FLOPs train (G/img) | "
              "GPU mem infer (GB) | GPU mem train (GB) | Latency (ms/img) | "
              "Throughput (img/s) |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for name, r in results.items():
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name,
            _fmt(r["params_total"] / 1e6),
            _fmt(r.get("flops_forward", 0) / 1e9),
            _fmt(r["flops_train"] / 1e9 if r.get("flops_train") else None),
            _fmt(r.get("peak_gpu_mem_infer_gb")),
            _fmt(r.get("peak_gpu_mem_train_step_gb")),
            _fmt(r.get("latency_ms_per_image")),
            _fmt(r.get("throughput_img_per_s"), "{:.1f}"),
        ))
    return "\n".join(lines)


def log_efficiency_summary(model, input_shape, device="cuda", logger=None,
                           name="model", **benchmark_kwargs):
    """Compute the five quantities and log them line by line.

    Convenience helper for the training scripts: call it right after the
    model is built so that every training log starts with its efficiency
    fingerprint (usable next to the paper's results list).
    """
    res = benchmark_model(model, input_shape, device=device, **benchmark_kwargs)
    log = logger.info if logger is not None and hasattr(logger, "info") else print
    log(f"[efficiency] {name}: total parameters "
        f"{res['params_total'] / 1e6:.2f}M (trainable {res['params_trainable'] / 1e6:.2f}M)")
    log(f"[efficiency] {name}: FLOPs {res['flops_forward'] / 1e9:.2f}G fwd"
        + (f" / {res['flops_train'] / 1e9:.2f}G train" if res.get("flops_train") else "")
        + f" per image @ {tuple(input_shape)}")
    log(f"[efficiency] {name}: GPU memory {_fmt(res.get('peak_gpu_mem_infer_gb'))}GB infer "
        f"(batch {benchmark_kwargs.get('latency_batch', 1)}) / "
        f"{_fmt(res.get('peak_gpu_mem_train_step_gb'))}GB train "
        f"(batch {res.get('train_batch_size', '-')})")
    log(f"[efficiency] {name}: inference latency {_fmt(res.get('latency_ms_per_image'))}ms/image, "
        f"throughput {_fmt(res.get('throughput_img_per_s'), '{:.1f}')}img/s "
        f"(batch {res.get('throughput_batch_size', '-')})")
    return res
