"""Throughput measurement for paper reporting.

Run via: python src/core/train_test.py config_path=... ++TRAIN.THROUGHPUT_TEST=true
Requires single-GPU setup (WORLD_SIZE=1).
"""

from __future__ import annotations

import time

import torch

from utils.distributed import is_main_process


@torch.inference_mode()
def measure_throughput(trainer, val_loader) -> dict:
    """Measure model throughput (images/sec) for paper reporting.

    Runs a reduced forward pass with no metrics, no loss, and (for EoMT) a single
    decode step instead of one per query block.  Reports images/sec and
    latency after a warm-up phase.

    Requires single-GPU setup — call after verifying world_size == 1.
    """
    cfg = trainer.cfg
    warmup_batches = int(cfg.TRAIN.get("THROUGHPUT_WARMUP_BATCHES", 50))
    measure_batches = int(cfg.TRAIN.get("THROUGHPUT_MEASURE_BATCHES", 200))

    use_amp = trainer.precision in ("bf16", "fp16")
    from utils.device import precision_dtype
    dtype = precision_dtype(trainer.precision)
    is_cuda = trainer.device.type == "cuda"

    was_training = trainer.model.training
    trainer.model.eval()

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    # ── Warm-up ───────────────────────────────────────────────────────────────
    if is_main_process():
        print(f"[Throughput] Warming up ({warmup_batches} batches) ...")
    warmup_done = 0
    for batch in val_loader:
        if warmup_done >= warmup_batches:
            break
        batch = trainer._to_device(batch)
        with torch.autocast(device_type=trainer.device.type, dtype=dtype, enabled=use_amp):
            trainer.throughput_step(trainer.model, batch)
        warmup_done += 1
    _sync()

    # ── Timed measurement ─────────────────────────────────────────────────────
    if is_main_process():
        n_desc = f"{measure_batches} batches" if measure_batches > 0 else "full val loader"
        print(f"[Throughput] Measuring ({n_desc}) ...")

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    images_measured = 0
    batch_times: list[float] = []

    for batch in val_loader:
        if measure_batches > 0 and len(batch_times) >= measure_batches:
            break
        batch = trainer._to_device(batch)
        _sync()
        t0 = time.perf_counter()
        with torch.autocast(device_type=trainer.device.type, dtype=dtype, enabled=use_amp):
            trainer.throughput_step(trainer.model, batch)
        _sync()
        batch_times.append(time.perf_counter() - t0)
        images_measured += batch["image"].shape[0]

    elapsed = sum(batch_times)
    throughput = images_measured / elapsed
    latency_ms = elapsed / images_measured * 1000.0
    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if is_cuda
        else 0.0
    )

    results = {
        "throughput/images_per_sec": throughput,
        "throughput/latency_ms_per_image": latency_ms,
        "throughput/total_images": images_measured,
        "throughput/total_time_s": elapsed,
        "throughput/batch_size": cfg.TRAIN.BATCH_SIZE,
        "throughput/peak_mem_gb": peak_mem_gb,
    }
    from utils.device import unwrap_model
    token_stats_fn = getattr(unwrap_model(trainer.model), "get_token_stats", None)
    if callable(token_stats_fn):
        results.update({
            f"throughput/{k.removeprefix('token_stats/')}": v
            for k, v in token_stats_fn().items()
            if isinstance(v, (int, float))
        })

    if is_main_process():
        print(
            f"\n{'=' * 60}\n"
            f"  THROUGHPUT RESULTS\n"
            f"{'=' * 60}\n"
            f"  Images / sec:        {throughput:>10.1f}\n"
            f"  Latency ms / image:  {latency_ms:>10.3f}\n"
            f"  Peak memory (GB):     {peak_mem_gb:>10.3f}\n"
            f"  Batch size:          {cfg.TRAIN.BATCH_SIZE:>10d}\n"
            f"  Batches measured:    {len(batch_times):>10d}\n"
            f"  Images measured:     {images_measured:>10d}\n"
            f"  Total time (s):      {elapsed:>10.1f}\n"
            f"{'=' * 60}\n"
        )
    trainer.model.train(was_training)
    return results


def measure_flops(trainer, loader) -> dict:
    """Count one forward's multiply-adds via ``FlopCounterMode``.

    FLOPs are the regime-independent efficiency metric (CROPR reports
    "FLOPs / throughput optimal across batch sizes"): a physically-pruned
    sequence runs smaller matmuls, so this drops with the keep rate regardless
    of batch size, kernel-launch overhead, or whether the model is compiled.
    Must run on the EAGER model — a compiled graph executes as fused kernels
    that bypass the dispatch counter. Returns GFLOPs per image.
    """
    from torch.utils.flop_counter import FlopCounterMode

    use_amp = trainer.precision in ("bf16", "fp16")
    from utils.device import precision_dtype
    dtype = precision_dtype(trainer.precision)

    was_training = trainer.model.training
    trainer.model.eval()
    batch = trainer._to_device(next(iter(loader)))
    bs = max(1, int(batch["image"].shape[0]))
    try:
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad(), torch.autocast(
            device_type=trainer.device.type, dtype=dtype, enabled=use_amp
        ):
            trainer.throughput_step(trainer.model, batch)
        total = float(counter.get_total_flops())
    except Exception as exc:  # noqa: BLE001 — diagnostics must never hard-fail
        if is_main_process():
            print(f"[Throughput] FLOP counting skipped: {type(exc).__name__}: {exc}")
        trainer.model.train(was_training)
        return {"flops/error": f"{type(exc).__name__}: {exc}"}
    trainer.model.train(was_training)
    return {
        "flops/total": total,
        "flops/per_image": total / bs,
        "flops/gflops_per_image": total / bs / 1e9,
    }
