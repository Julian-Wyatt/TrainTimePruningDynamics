#!/usr/bin/env python3
"""Reproduce the paper's EoMT throughput and FLOPs CSV.

Each repeat runs in a fresh subprocess. The paper's ``backbone`` strategy is
intentional: it compiles stable backbone blocks and token scorers while routing
stays eager. This is the legacy strategy behind the existing CSV, not a generic
compile benchmark.

Example:
    python scripts/measure_throughput.py \
        --config-dir configs/experiments/FIVES/study/question_BC/cropr_vits \
        --batch-sizes 8 --repeats 3
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import io
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HYDRA_FULL_ERROR", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_ROOT = REPO_ROOT / "configs"
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CONFIG_DIR = CONFIGS_ROOT / "experiments/FIVES/study/question_BC/cropr_vits"
SCAFFOLD_CONFIGS = {"base", "step3_base"}
STRATEGY = "backbone"

CSV_FIELDS = [
    "results_group", "experiment_id", "config_leaf", "backbone", "train_mode",
    "reinject", "prune_layers", "keep_rates", "img_size",
    "throughput_batch_size", "throughput_strategy", "throughput_n_runs",
    "throughput_img_per_sec", "throughput_img_per_sec_std",
    "throughput_latency_ms_per_image", "gflops_per_image", "peak_mem_gb",
    "token_keep_ratio", "compile_note", "git_sha", "timestamp",
]
CSV_KEY = (
    "results_group", "experiment_id", "throughput_strategy",
    "throughput_batch_size", "img_size",
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", default="", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default="", help=argparse.SUPPRESS)
    parser.add_argument("--override-json", default="[]", help=argparse.SUPPRESS)
    parser.add_argument("--skip-flops", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Config stems; default is every leaf YAML in --config-dir.")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--registry-csv", default="results/throughput.csv")
    parser.add_argument("--loader", choices=("auto", "train", "val"), default="auto",
                        help="Keep the historical loader choice; auto is the legacy default.")
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measure-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=-1)
    args, overrides = parser.parse_known_args()
    return args, _normalise_overrides(overrides)


def _normalise_overrides(raw: list[str]) -> list[str]:
    overrides = []
    for item in raw:
        if item == "--":
            continue
        if "=" not in item:
            raise SystemExit(f"Config overrides must be KEY=VALUE, got {item!r}.")
        key, value = item.split("=", 1)
        if not key.startswith("~"):
            overrides.append(f"{key.lstrip('+')}={value}")
    return overrides


def _config_arg(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(CONFIGS_ROOT).with_suffix(""))
    except ValueError:
        return str(path)


def _discover_configs(config_dir: Path, names: list[str] | None) -> list[Path]:
    if names:
        return [
            (config_dir / (name if name.endswith(".yaml") else f"{name}.yaml")).resolve()
            for name in names
        ]
    return sorted(
        path for path in config_dir.glob("*.yaml")
        if path.stem not in SCAFFOLD_CONFIGS
    )


def _load_config(args: argparse.Namespace, overrides: list[str]):
    from omegaconf import OmegaConf

    from core.conf import Config, process_config

    base = OmegaConf.structured(Config())
    OmegaConf.set_struct(base, False)
    base.config_path = args.config
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = process_config(base, str(CONFIGS_ROOT))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(cfg, False)

    cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.VAL_BATCH_SIZE = int(args.batch_size)
    if args.num_workers >= 0:
        cfg.TRAIN.NUM_WORKERS = int(args.num_workers)
    cfg.TRAIN.EFFECTIVE_BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
    cfg.TRAIN.THROUGHPUT_TEST = True
    cfg.TRAIN.THROUGHPUT_WARMUP_BATCHES = int(args.warmup_batches)
    cfg.TRAIN.THROUGHPUT_MEASURE_BATCHES = int(args.measure_batches)
    cfg.TRAIN.LOG_TYPE = "none"
    cfg.TRAIN.SAVE_CHECKPOINTS = "false"
    cfg.TRAIN.RUN_TRAIN = cfg.TRAIN.RUN_TEST = False
    cfg.TRAIN.COMPILE = False
    cfg.MODEL.EMA_DECAY = 0.0
    return cfg


def _prepare_compile(trainer) -> bool:
    prepare = getattr(trainer, "_prepare_compile", None)
    if callable(prepare):
        return bool(prepare())

    import torch

    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.cache_size_limit = 64
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = max(
            64,
            int(getattr(torch._dynamo.config, "accumulated_cache_size_limit", 0)),
        )
    return True


def _compile_backbone(model) -> str:
    """Apply the historical ``backbone`` strategy used for the paper CSV."""
    import torch

    compiled = []
    blocks = getattr(
        getattr(getattr(model, "encoder", None), "backbone", None), "blocks", None)
    if hasattr(model, "_block") and blocks is not None and len(blocks) > 0:
        from models.EoMT.block import run_block

        attn_attr = getattr(model, "_attn_attr", "attn")
        compiled_blocks = {}
        for block in blocks:
            def _run_one(x, attn_mask, rope, policy=None, *, _block=block):
                return run_block(
                    _block,
                    x,
                    attn_mask,
                    rope,
                    attn_attr,
                    model.training,
                    policy,
                )

            compiled_blocks[id(block)] = torch.compile(
                _run_one, fullgraph=False, dynamic=False)
        original_block = model._block

        def _dispatch_compiled_block(block, x, attn_mask, rope, policy=None):
            compiled_block = compiled_blocks.get(id(block))
            if compiled_block is None:
                return original_block(block, x, attn_mask, rope, policy)
            return compiled_block(x, attn_mask, rope, policy)

        model._block = _dispatch_compiled_block
        compiled.append(f"blocks:{len(compiled_blocks)}")
    elif hasattr(model, "_block"):
        model._block = torch.compile(model._block, fullgraph=False, dynamic=False)
        compiled.append("_block")
    if bool(getattr(model, "aux_scorer", False)):
        mods = getattr(model, "aux_cross_attn", None)
        n_scorers = 0
        if mods is not None and len(mods) > 0:
            for sub in mods:
                scorer = getattr(sub, "forward_scorer", None)
                if callable(scorer):
                    sub.forward_scorer = torch.compile(
                        scorer, fullgraph=False, dynamic=False)
                    n_scorers += 1
        if n_scorers:
            compiled.append(f"aux_cross_attn.forward_scorer:{n_scorers}")
    else:
        selectors = getattr(model, "selectors", None)
        if selectors is not None and len(selectors) > 0:
            for sub in selectors:
                sub.compile(fullgraph=False, dynamic=False)
            compiled.append("selectors")
    return "backbone(" + "+".join(compiled or ["nothing"]) + ")"


def _apply_paper_compile(trainer) -> str:
    if not _prepare_compile(trainer):
        return "backbone(disabled)"
    from utils.device import unwrap_model

    return _compile_backbone(unwrap_model(trainer.model))


def _select_loader(cfg, data_module, requested: str):
    loader_name = requested
    if loader_name == "auto":
        loader_name = "train" if str(cfg.TRAIN.TASK).lower() == "semantic" else "val"
    return data_module.train_loader() if loader_name == "train" else data_module.val_loader()


def _identity(cfg) -> dict[str, str]:
    from omegaconf import OmegaConf

    def value(item) -> str:
        if OmegaConf.is_config(item):
            item = OmegaConf.to_container(item, resolve=True)
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(str(x) for x in item) + "]"
        return "" if item is None else str(item)

    model_cfg = cfg.MODEL
    pruning = bool(model_cfg.get("USE_EOMT_PRUNING", False))
    return {
        "config_leaf": Path(str(cfg.get("config_name", "") or "")).name,
        "backbone": value(model_cfg.get("BACKBONE", "")),
        "train_mode": value(model_cfg.get("EOMT_PRUNE_TRAIN_MODE", "") if pruning else "dense"),
        "reinject": value(model_cfg.get("EOMT_PRUNE_REINJECT", "") if pruning else ""),
        "prune_layers": value(model_cfg.get("EOMT_PRUNE_LAYERS", "") if pruning else ""),
        "keep_rates": value(model_cfg.get("EOMT_KEEP_RATES", "") if pruning else ""),
    }


def _worker(args: argparse.Namespace, overrides: list[str]) -> None:
    from data.datamodule import DataModule
    from data.dataset_factory import build_datasets
    from trainer.segmentation_trainer import EoMTTrainer
    from trainer.throughput import measure_flops, measure_throughput
    from utils.results_registry import resolve_experiment_id, resolve_results_group

    cfg = _load_config(args, overrides)
    train_ds, val_ds, test_ds = build_datasets(cfg)
    trainer = EoMTTrainer(cfg, DataModule(cfg, train_ds, val_ds, test_ds, local_rank=0))
    loader = _select_loader(cfg, trainer.data_module, args.loader)

    if trainer.device.type == "cuda":
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    results = {} if args.skip_flops else measure_flops(trainer, loader)
    compile_note = _apply_paper_compile(trainer)
    results.update(measure_throughput(trainer, loader))

    payload = {
        "config": Path(args.config).stem,
        "strategy": STRATEGY,
        "compile_note": compile_note,
        "batch_size": int(cfg.TRAIN.BATCH_SIZE),
        "img_size": str(list(cfg.DATASET.IMG_SIZE)),
        "experiment_id": resolve_experiment_id(cfg),
        "results_group": resolve_results_group(cfg),
        "identity": _identity(cfg),
        "results": results,
    }
    Path(args.result_json).write_text(json.dumps(payload, indent=2))


def _run_worker(
    config_path: Path,
    batch_size: int,
    args: argparse.Namespace,
    overrides: list[str],
    result_path: Path,
    *,
    skip_flops: bool,
) -> int:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--config", _config_arg(config_path),
        "--batch-size", str(batch_size),
        "--result-json", str(result_path),
        "--override-json", json.dumps(overrides),
        "--loader", args.loader,
        "--warmup-batches", str(args.warmup_batches),
        "--measure-batches", str(args.measure_batches),
        "--num-workers", str(args.num_workers),
    ]
    if skip_flops:
        command.append("--skip-flops")
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def _aggregate(payloads: list[dict]) -> dict:
    """Mean timed metrics across fresh-process repeats; preserve FLOPs from repeat 0."""
    aggregate = json.loads(json.dumps(payloads[0]))
    results = aggregate["results"]

    def values(key: str) -> list[float]:
        return [
            float(value)
            for payload in payloads
            if isinstance(value := payload["results"].get(key), (int, float))
        ]

    for key, reducer in {
        "throughput/images_per_sec": lambda xs: sum(xs) / len(xs),
        "throughput/latency_ms_per_image": lambda xs: sum(xs) / len(xs),
        "throughput/peak_mem_gb": max,
    }.items():
        if samples := values(key):
            results[key] = reducer(samples)
    ips = values("throughput/images_per_sec")
    results["throughput/images_per_sec_std"] = (
        statistics.stdev(ips) if len(ips) > 1 else 0.0)
    results["throughput/n_runs"] = len(ips)
    return aggregate


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _csv_row(payload: dict) -> dict[str, str]:
    results = payload["results"]
    identity = payload["identity"]

    def number(key: str):
        value = results.get(key)
        return value if isinstance(value, (int, float)) else ""

    row = {
        "results_group": payload["results_group"],
        "experiment_id": payload["experiment_id"],
        "config_leaf": identity["config_leaf"] or payload["config"],
        "backbone": identity["backbone"],
        "train_mode": identity["train_mode"],
        "reinject": identity["reinject"],
        "prune_layers": identity["prune_layers"],
        "keep_rates": identity["keep_rates"],
        "img_size": payload["img_size"],
        "throughput_batch_size": payload["batch_size"],
        "throughput_strategy": STRATEGY,
        "throughput_n_runs": results["throughput/n_runs"],
        "throughput_img_per_sec": number("throughput/images_per_sec"),
        "throughput_img_per_sec_std": number("throughput/images_per_sec_std"),
        "throughput_latency_ms_per_image": number("throughput/latency_ms_per_image"),
        "gflops_per_image": number("flops/gflops_per_image"),
        "peak_mem_gb": number("throughput/peak_mem_gb"),
        "token_keep_ratio": number("throughput/token_step_ratio"),
        "compile_note": payload["compile_note"],
        "git_sha": _git_sha(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return {key: f"{value:.6g}" if isinstance(value, float) else str(value)
            for key, value in row.items()}


def _upsert_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header, rows = CSV_FIELDS, []
    if path.exists():
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            header = list(dict.fromkeys(CSV_FIELDS + list(reader.fieldnames or [])))
            rows = list(reader)
    key = tuple(row[column] for column in CSV_KEY)
    rows = [old for old in rows if tuple(old.get(column, "") for column in CSV_KEY) != key]
    rows.append(row)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(file.name)
    temporary_path.replace(path)


def _print_summary(rows: list[dict]) -> None:
    print(f"\n{'config':<28}{'bs':>4}{'img/s':>10}{'GFLOPs':>10}{'peak GB':>10}")
    for payload in rows:
        results = payload["results"]
        print(
            f"{payload['config']:<28}{payload['batch_size']:>4}"
            f"{results.get('throughput/images_per_sec', 0.0):>10.1f}"
            f"{results.get('flops/gflops_per_image', 0.0):>10.1f}"
            f"{results.get('throughput/peak_mem_gb', 0.0):>10.2f}"
        )


def _parent(args: argparse.Namespace, overrides: list[str]) -> None:
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = REPO_ROOT / config_dir
    configs = _discover_configs(config_dir.resolve(), args.configs)
    if not configs:
        raise SystemExit(f"No configs found in {config_dir}")

    registry = Path(args.registry_csv)
    if not registry.is_absolute():
        registry = REPO_ROOT / registry
    rows, failures = [], []
    for config_path in configs:
        for batch_size in sorted(args.batch_sizes):
            print(f"\n=== {config_path.stem} / {STRATEGY} / bs{batch_size} ===", flush=True)
            payloads = []
            with tempfile.TemporaryDirectory(prefix="prunestudy-throughput-") as temp_dir:
                for repeat in range(max(1, args.repeats)):
                    result_path = Path(temp_dir) / f"repeat-{repeat}.json"
                    code = _run_worker(
                        config_path, batch_size, args, overrides, result_path,
                        skip_flops=repeat > 0,
                    )
                    if code != 0 or not result_path.exists():
                        failures.append(f"{config_path.stem}/bs{batch_size}/repeat-{repeat} (exit {code})")
                        break
                    payloads.append(json.loads(result_path.read_text()))
            if not payloads:
                break  # First repeat failed, commonly an OOM: stop the batch ladder.

            aggregate = _aggregate(payloads)
            rows.append(aggregate)
            if args.registry_csv:
                _upsert_csv(registry, _csv_row(aggregate))
            if len(payloads) < max(1, args.repeats):
                break

    _print_summary(rows)
    if args.registry_csv:
        print(f"\nthroughput CSV: {registry}")
    if failures:
        print(f"failures: {failures}")


def main() -> None:
    args, overrides = parse_args()
    if args.worker:
        _worker(args, json.loads(args.override_json))
    else:
        _parent(args, overrides)


if __name__ == "__main__":
    main()
