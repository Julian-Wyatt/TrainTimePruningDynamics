"""Append-only results registry: one CSV table per sweep axis.

Each finished training run appends a single row keyed by ``(EXPERIMENT_ID, SEED)``
to ``<RESULTS_TABLE_DIR>/<RESULTS_GROUP>.csv``. The identity columns mirror the
W&B run (``experiment_id`` == W&B group, ``run_id`` == W&B id, ``description`` ==
W&B name prefix) so a CSV row and its W&B plots cross-reference directly.

``has_result`` lets the entry point skip an ``(experiment, seed)`` cell that is
already recorded (unless ``TRAIN.OVERRIDE_RESULTS``), so re-queuing a sweep only
fills the gaps. ``collate_results.py`` pivots a table into a mean±std summary.
"""
from __future__ import annotations

import csv
import datetime
import glob
import os
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Tuple

from utils.distributed import is_main_process

# Identity/provenance columns, in display order. Metric columns are appended
# after these (in first-seen order) so new metrics never reorder the table.
_PREFERRED_COLUMNS = [
    "experiment_id", "results_group", "seed", "description", "run_id",
    "config_leaf", "backbone", "train_mode", "reinject", "prune_layers",
    "keep_rates", "img_size", "epochs", "best_val_mIoU", "git_sha", "timestamp",
]

# A test metric is "core" if its (std-stripped) name matches one of these and is
# not a per-class / per-query-block breakdown (those would explode the columns).
_CORE_METRIC_SUBSTRINGS = (
    "miou", "dice", "precision", "recall", "boundary", "cldice",
    "thin", "pixel_ap",
)
_METRIC_EXCLUDE_SUBSTRINGS = ("qblock", "per_image", "class", "iou_")


def _git_sha() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _normalise_dir_token(token: str) -> str:
    return (
        token.replace("question_A", "qA").replace("question_BC", "qBC")
        .replace("question_B", "qB").replace("question_C", "qC")
    )


def resolve_experiment_id(cfg) -> str:
    """Explicit ``TRAIN.EXPERIMENT_ID`` else the leaf of ``config_name``."""
    explicit = str(cfg.TRAIN.get("EXPERIMENT_ID", "") or "").strip()
    if explicit:
        return explicit
    config_name = str(cfg.get("config_name", "") or "")
    if config_name:
        return os.path.basename(config_name.rstrip("/"))
    return "run"


def resolve_results_group(cfg) -> str:
    """Explicit ``TRAIN.RESULTS_GROUP`` else derived from ``config_name`` dirs.

    ``.../question_A/core/dense`` -> ``qA_core``; ``.../question_BC/cropr_vits/x``
    -> ``qBC_cropr_vits``; ``.../question_A/keep_sweep/y`` -> ``qA_keep_sweep``.
    """
    explicit = str(cfg.TRAIN.get("RESULTS_GROUP", "") or "").strip()
    if explicit:
        return explicit
    config_name = str(cfg.get("config_name", "") or "")
    parts = [p for p in config_name.split("/") if p]
    dirs = [_normalise_dir_token(p) for p in parts[:-1]]
    for i, token in enumerate(dirs):
        if token.startswith("q") and token[1:2].isupper():  # qA/qB/qBC/qC
            return "_".join(dirs[i:])
    return dirs[-1] if dirs else "default"


def results_dir(cfg) -> str:
    base = str(cfg.TRAIN.get("RESULTS_TABLE_DIR", "") or "").strip()
    if not base:
        base = os.path.join(cfg.TRAIN.SAVING_ROOT_DIR, "results")
    return base


def results_csv_path(cfg) -> str:
    return os.path.join(results_dir(cfg), f"{resolve_results_group(cfg)}.csv")


def _iter_all_rows(directory: str) -> Iterable[Dict[str, str]]:
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        try:
            with open(path, newline="") as f:
                yield from csv.DictReader(f)
        except (OSError, csv.Error):
            continue


def has_result(cfg, experiment_id: str | None = None, seed: int | None = None) -> bool:
    """True if some ``results/*.csv`` already holds this ``(id, seed)`` cell.

    Scans every group CSV (not just this run's), so a thin-alias twin that maps
    to a different group is still recognised as already-run.
    """
    experiment_id = experiment_id or resolve_experiment_id(cfg)
    seed = int(cfg.TRAIN.SEED if seed is None else seed)
    directory = results_dir(cfg)
    if not os.path.isdir(directory):
        return False
    for row in _iter_all_rows(directory):
        if row.get("experiment_id") == experiment_id and str(row.get("seed")) == str(seed):
            return True
    return False


def _is_core_metric(key: str) -> bool:
    base = key[:-4].lower() if key.lower().endswith("_std") else key.lower()
    if any(x in base for x in _METRIC_EXCLUDE_SUBSTRINGS):
        return False
    return any(x in base for x in _CORE_METRIC_SUBSTRINGS)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fmt(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_row(
    cfg,
    test_metrics: Dict[str, float] | None,
    best_val: float | None = None,
) -> Dict[str, str]:
    m = cfg.MODEL
    pruning = bool(m.get("USE_EOMT_PRUNING", False))
    train_mode = m.get("EOMT_PRUNE_TRAIN_MODE", "") if pruning else "dense"
    row: Dict[str, Any] = {
        "experiment_id": resolve_experiment_id(cfg),
        "results_group": resolve_results_group(cfg),
        "seed": int(cfg.TRAIN.SEED),
        "description": cfg.TRAIN.get("DESCRIPTION", "") or "",
        "run_id": cfg.TRAIN.get("RUN_ID", "") or "",
        "config_leaf": os.path.basename(str(cfg.get("config_name", "") or "")),
        "backbone": m.get("BACKBONE", ""),
        "train_mode": train_mode,
        "reinject": m.get("EOMT_PRUNE_REINJECT", "") if pruning else "",
        "prune_layers": m.get("EOMT_PRUNE_LAYERS", "") if pruning else "",
        "keep_rates": m.get("EOMT_KEEP_RATES", "") if pruning else "",
        "img_size": cfg.DATASET.get("IMG_SIZE", ""),
        "epochs": int(cfg.TRAIN.EPOCHS),
        "git_sha": _git_sha(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if best_val is not None and best_val > -float("inf"):
        row["best_val_mIoU"] = float(best_val)
    for key, value in (test_metrics or {}).items():
        if _is_core_metric(key):
            row[f"test_{key}"] = value
    return {k: _fmt(v) for k, v in row.items()}


def _union_header(existing: List[str], new_keys: Iterable[str]) -> List[str]:
    all_keys = list(dict.fromkeys(list(existing) + list(new_keys)))
    ordered = [c for c in _PREFERRED_COLUMNS if c in all_keys]
    ordered += [k for k in all_keys if k not in ordered]
    return ordered


def _read_table(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    if not os.path.exists(path):
        return [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write_atomic(path: str, header: List[str], rows: List[Dict[str, str]]) -> None:
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def append_result(
    cfg,
    test_metrics: Dict[str, float] | None,
    best_val: float | None = None,
) -> str | None:
    """Append (or replace) this run's ``(id, seed)`` row in its group CSV.

    Rank-0 only. Concurrency-safe via an ``flock`` on a sibling lock file; the
    whole (tiny) table is re-read and rewritten so the header can grow and a
    stale same-key row is replaced (idempotent + honours OVERRIDE_RESULTS).
    """
    if not is_main_process():
        return None
    import fcntl

    row = build_row(cfg, test_metrics, best_val=best_val)
    path = results_csv_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            header, rows = _read_table(path)
            key = (row["experiment_id"], str(row["seed"]))
            rows = [
                r for r in rows
                if (r.get("experiment_id"), str(r.get("seed"))) != key
            ]
            rows.append(row)
            _write_atomic(path, _union_header(header, row.keys()), rows)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return path
