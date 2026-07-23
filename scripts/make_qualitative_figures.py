#!/usr/bin/env python
"""Build paper-ready qualitative segmentation figures from ordered checkpoints.

The main figure is a row-per-sample grid:

    image | ground truth | checkpoint 1 soft output | checkpoint 2 soft output | ...

By default samples are selected from the test set by the standard deviation of
per-sample mIoU across the supplied checkpoints.  This tends to surface cases
where methods genuinely disagree, which is useful for paper qualitative panels.

Example:
    PYTHONPATH=src python scripts/make_qualitative_figures.py \
        --candidate-batches 48 --rows 3
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Keep matplotlib/font caches out of the home directory.  This matters on the
# managed local sandbox and on some cluster nodes where $HOME is read-only.
_TMP_ROOT = os.environ.get("TMPDIR", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_TMP_ROOT, "prunestudy_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_TMP_ROOT, "prunestudy_xdg_cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# Figure column order (paper method names). Point these at your own trained
# checkpoints, or pass NAME=PATH via --checkpoint.
DEFAULT_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("Dense", "checkpoints/dense/best.pt"),
    ("Soft Mask", "checkpoints/soft_mask/best.pt"),
    ("Random_route", "checkpoints/random_route/best.pt"),
    ("Gumbel_penulti", "checkpoints/gumbel_route_penultimate/best.pt"),
    ("aux_route_penultimate", "checkpoints/aux_route_penultimate/best.pt"),
)

DEFAULT_PRUNE_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    DEFAULT_CHECKPOINTS[1],
    DEFAULT_CHECKPOINTS[3],
    DEFAULT_CHECKPOINTS[4],
)

STYLE_VERSION = "v6_input-left_soft-alpha-0.60_selected-pruning-policy"


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path


@dataclass
class Candidate:
    scan_index: int
    image: torch.Tensor
    mask: torch.Tensor
    meta: dict


def _parse_name_path(value: str) -> CheckpointSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH checkpoint spec, got {value!r}"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip().strip('"').strip("'")
    if not name or not path:
        raise argparse.ArgumentTypeError(
            f"Expected non-empty NAME=PATH checkpoint spec, got {value!r}"
        )
    return CheckpointSpec(name=name, path=Path(path))


def _default_specs() -> list[CheckpointSpec]:
    return [CheckpointSpec(name, Path(path)) for name, path in DEFAULT_CHECKPOINTS]


def _default_prune_specs() -> list[CheckpointSpec]:
    return [CheckpointSpec(name, Path(path)) for name, path in DEFAULT_PRUNE_CHECKPOINTS]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _torch_load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _base_cfg() -> DictConfig:
    from core.conf import Config

    cfg = OmegaConf.structured(Config)
    OmegaConf.set_struct(cfg, False)
    yaml_cfg = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
    cfg = OmegaConf.merge(cfg, yaml_cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _load_cfg_from_checkpoint(path: Path, fallback_config_path: str | None) -> DictConfig:
    ckpt = _torch_load(path)
    if "cfg" in ckpt:
        cfg = OmegaConf.merge(_base_cfg(), OmegaConf.create(ckpt["cfg"]))
        OmegaConf.set_struct(cfg, False)
        return cfg
    if not fallback_config_path:
        raise ValueError(
            f"Checkpoint {path} does not contain a saved cfg; pass --config-path."
        )

    from core.conf import process_config

    cfg = _base_cfg()
    cfg.config_path = fallback_config_path
    return process_config(cfg, str(REPO_ROOT / "configs"))


def _sanitise_cfg_for_plotting(cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(cfg, False)

    cfg.TRAIN.LOG_TYPE = "none"
    cfg.TRAIN.COMPILE = False
    cfg.TRAIN.NUM_WORKERS = 0
    cfg.TRAIN.VAL_BATCH_SIZE = int(args.eval_batch_size)
    cfg.TRAIN.BATCH_SIZE = int(args.eval_batch_size)
    cfg.TRAIN.RUN_TRAIN = False
    cfg.TRAIN.RUN_TEST = False
    cfg.TRAIN.SAVE_CHECKPOINTS = "false"
    cfg.TRAIN.SAVING_ROOT_DIR = str(args.output_dir)
    cfg.TRAIN.SEED = int(args.seed)

    # Checkpoints already contain the trained weights.  Avoid constructing an
    # EMA shadow copy and avoid timm attempting a pretrained download before
    # we immediately overwrite the state dict.
    cfg.MODEL.EMA_DECAY = 1.0
    cfg.MODEL.CKPT_PATH = "__paper_plot_skip_pretrained__"

    if args.img_size:
        cfg.DATASET.IMG_SIZE = [int(args.img_size), int(args.img_size)]

    return cfg


def _build_data_module(cfg: DictConfig):
    from data.datamodule import DataModule
    from data.dataset_factory import build_datasets

    train_ds, val_ds, test_ds = build_datasets(cfg)
    return DataModule(cfg, train_ds, val_ds, test_ds, local_rank=0)


def _build_trainer(cfg: DictConfig):
    from trainer.segmentation_trainer import EoMTTrainer

    return EoMTTrainer(cfg, data_module=None)


def _load_trainer_for_checkpoint(
    spec: CheckpointSpec,
    args: argparse.Namespace,
    device: torch.device,
):
    from utils.checkpoint import _adapt_state_dict_to_model

    cfg = _load_cfg_from_checkpoint(spec.path, args.config_path)
    cfg = _sanitise_cfg_for_plotting(cfg, args)
    trainer = _build_trainer(cfg)
    ckpt = _torch_load(spec.path)
    trainer.model.load_state_dict(_adapt_state_dict_to_model(ckpt["model"], trainer.model))
    trainer.model.to(device)
    trainer.model.eval()
    trainer.device = device
    trainer.current_epoch = int(ckpt.get("epoch", cfg.TRAIN.get("EPOCHS", 0)))
    return trainer


def _iter_candidate_batches(data_module, split: str):
    if split == "val":
        return data_module.val_loader()
    if split == "test":
        loader = data_module.test_loader()
        if loader is None:
            return data_module.val_loader()
        return loader
    raise ValueError(f"Unknown split {split!r}; expected val or test.")


def _collect_candidates(
    data_module,
    split: str,
    candidate_batches: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    scan_index = 0
    for batch_idx, batch in enumerate(_iter_candidate_batches(data_module, split)):
        if batch_idx >= candidate_batches:
            break
        images = batch["image"]
        masks = batch["mask"]
        metas = batch.get("meta", [{} for _ in range(len(images))])
        if isinstance(images, list):
            raise RuntimeError(
                "This plotting script currently expects fixed-size batched images. "
                "For variable-size semantic eval, pass a config that resizes val/test."
            )
        for i in range(images.shape[0]):
            meta = metas[i] if isinstance(metas, list) else {}
            candidates.append(
                Candidate(
                    scan_index=scan_index,
                    image=images[i].detach().cpu(),
                    mask=masks[i].detach().cpu(),
                    meta=dict(meta) if isinstance(meta, dict) else {},
                )
            )
            scan_index += 1
    return candidates


def _chunks(items: list[Candidate], chunk_size: int) -> Iterable[list[Candidate]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _qblock_pixel_scores(trainer, batch: dict, qblock: int) -> torch.Tensor:
    """Return pixel scores/probabilities in [B, C, H, W] when available."""
    out = trainer.shared_step(trainer.model, batch)
    if "mask_logits_per_layer" in out and "class_logits_per_layer" in out:
        masks_per = out["mask_logits_per_layer"]
        classes_per = out["class_logits_per_layer"]
        layer_idx = int(qblock) + len(masks_per) - 1
        if not 0 <= layer_idx < len(masks_per):
            rel = [i - (len(masks_per) - 1) for i in range(len(masks_per))]
            raise ValueError(
                f"Requested qblock={qblock}, but available qblocks are {rel}."
            )
        return trainer._derive_pixel_logits(masks_per[layer_idx], classes_per[layer_idx])

    pixel_logits = trainer.get_pixel_logits(trainer.model, batch)
    if pixel_logits.min().item() >= -1e-5 and pixel_logits.max().item() <= 1.0 + 1e-5:
        channel_sum = pixel_logits.sum(dim=1).mean().item()
        if 0.9 <= channel_sum <= 1.1:
            return pixel_logits
    return pixel_logits.softmax(dim=1)


def _fg_prob(scores: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    if int(cfg.DATASET.NUM_CLASSES) == 2 and scores.shape[1] >= 2:
        fg_channel = int(cfg.DATASET.get("BINARY_FG_CHANNEL", 1))
        return scores[:, fg_channel].clamp(0, 1)
    if scores.shape[1] >= 2:
        return scores[:, 1:].max(dim=1).values.clamp(0, 1)
    return scores[:, 0].clamp(0, 1)


def _compute_model_outputs(
    spec: CheckpointSpec,
    candidates: list[Candidate],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[np.ndarray], list[float]]:
    from utils.metrics import compute_seg_metrics

    _seed_everything(args.seed)
    trainer = _load_trainer_for_checkpoint(spec, args, device)
    probs: list[np.ndarray] = []
    mious: list[float] = []
    include_bg = not bool(trainer.cfg.DATASET.get("EXCLUDE_BACKGROUND_CLASS", False))

    with torch.no_grad():
        for chunk in _chunks(candidates, int(args.eval_batch_size)):
            images = torch.stack([c.image for c in chunk]).to(device)
            masks = torch.stack([c.mask for c in chunk]).to(device)
            batch = {
                "image": images,
                "mask": masks,
                "meta": [c.meta for c in chunk],
            }
            scores = _qblock_pixel_scores(trainer, batch, int(args.qblock))
            scores = trainer._resize_pixel_logits_to_masks(scores.float(), masks)
            pred = trainer._prediction_from_pixel_logits(scores)
            fg = _fg_prob(scores, trainer.cfg).detach().cpu()
            for i in range(fg.shape[0]):
                probs.append(fg[i].numpy().astype(np.float32))
                metric = compute_seg_metrics(
                    pred=pred[i : i + 1].detach().cpu(),
                    masks=masks[i : i + 1].detach().cpu(),
                    num_classes=int(trainer.cfg.DATASET.NUM_CLASSES),
                    ignore_index=int(trainer.cfg.DATASET.IGNORE_INDEX),
                    include_background=include_bg,
                )
                mious.append(float(metric["mIoU"]))

    del trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return probs, mious


def _denorm(img: torch.Tensor, mean, std) -> np.ndarray:
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    out = (img.float().cpu() * s + m).clamp(0, 1)
    return (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _resize_np(arr: np.ndarray, size: int | None, is_mask: bool = False) -> np.ndarray:
    if size is None:
        return arr
    pil = Image.fromarray(arr)
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    return np.asarray(pil.resize((size, size), resample=resample))


def _mask_panel(
    img_np: np.ndarray,
    mask: torch.Tensor,
    ignore_index: int,
    mode: str,
) -> np.ndarray:
    mask_np = mask.cpu().numpy()
    fg = (mask_np > 0) & (mask_np != ignore_index)
    if mode == "mask":
        out = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
        out[fg] = 255
        return out
    faded = (img_np.astype(np.float32) * 0.45).astype(np.uint8)
    out = faded.copy()
    green = np.array([40, 230, 80], dtype=np.float32)
    out[fg] = (0.65 * green + 0.35 * img_np[fg].astype(np.float32)).astype(np.uint8)
    return out


def _prob_panel(
    img_np: np.ndarray,
    prob: np.ndarray,
    cmap: str,
    mode: str,
    alpha: float,
) -> np.ndarray:
    prob = np.clip(prob, 0.0, 1.0)
    if prob.shape != img_np.shape[:2]:
        prob_t = torch.from_numpy(prob).float().reshape(1, 1, *prob.shape)
        prob = (
            F.interpolate(
                prob_t, size=img_np.shape[:2], mode="bilinear", align_corners=False
            )
            .reshape(img_np.shape[:2])
            .numpy()
        )
    coloured = (plt.get_cmap(cmap)(prob)[:, :, :3] * 255).astype(np.uint8)
    if mode == "heatmap":
        return coloured
    return (alpha * coloured + (1.0 - alpha) * img_np).clip(0, 255).astype(np.uint8)


def _compose_grid(
    rows: list[list[np.ndarray]],
    output_path: Path,
    pad: int,
    dpi: int,
    background: tuple[int, int, int] = (255, 255, 255),
    extra_col_gaps: dict[int, int] | None = None,
) -> None:
    if not rows or not rows[0]:
        raise ValueError("No panels to compose.")
    h, w = rows[0][0].shape[:2]
    n_rows = len(rows)
    n_cols = len(rows[0])
    extra_col_gaps = extra_col_gaps or {}
    canvas = Image.new(
        "RGB",
        (
            n_cols * w
            + (n_cols - 1) * pad
            + sum(max(0, int(v)) for k, v in extra_col_gaps.items() if 0 <= k < n_cols - 1),
            n_rows * h + (n_rows - 1) * pad,
        ),
        background,
    )
    for r, row in enumerate(rows):
        if len(row) != n_cols:
            raise ValueError("All rows must have the same number of panels.")
        x = 0
        for c, panel in enumerate(row):
            if panel.shape[:2] != (h, w):
                raise ValueError("All panels must share the same rendered size.")
            canvas.paste(Image.fromarray(panel), (x, r * (h + pad)))
            x += w
            if c < n_cols - 1:
                x += pad + max(0, int(extra_col_gaps.get(c, 0)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(dpi, dpi), compress_level=1)


def _first_method_col(image_column_position: str) -> int:
    # Rows are either:
    #   start: image | GT | method...
    #   end/none: GT | method... | image?
    return 2 if image_column_position == "start" else 1


def _dense_separator_gaps(args: argparse.Namespace, n_methods: int) -> dict[int, int]:
    if n_methods <= 1:
        return {}
    gap = int(args.dense_separator_extra_pad)
    if gap <= 0:
        return {}
    return {_first_method_col(args.image_column_position): gap}


def _column_order(
    method_names: list[str],
    image_column_position: str,
    include_ground_truth: bool,
) -> list[str]:
    columns: list[str] = []
    if image_column_position == "start":
        columns.append("input image")
    if include_ground_truth:
        columns.append("ground truth")
    columns.extend(method_names)
    if image_column_position == "end":
        columns.append("input image")
    return columns


def _prune_policy_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    return (args.prune_policy_checkpoint or _default_prune_specs())[:3]


def _load_selected_scan_indices_from_metadata(path: Path) -> list[list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected_plots = []
    for plot in payload.get("plots") or []:
        scan_indices = plot.get("scan_indices") or []
        if scan_indices:
            selected_plots.append([int(idx) for idx in scan_indices])
    if selected_plots:
        return selected_plots

    flat = [
        int(item["scan_index"])
        for item in payload.get("selected_samples") or []
        if "scan_index" in item
    ]
    return [flat] if flat else []


def _positions_from_scan_index_plots(
    candidates: list[Candidate],
    scan_index_plots: list[list[int]],
) -> list[list[int]]:
    index_to_pos = {c.scan_index: i for i, c in enumerate(candidates)}
    missing = sorted({
        idx
        for plot in scan_index_plots
        for idx in plot
        if idx not in index_to_pos
    })
    if missing:
        raise ValueError(
            "Selection metadata references scan indices that were not scanned: "
            f"{missing}. Increase --candidate-batches or lower --eval-batch-size."
        )
    return [[index_to_pos[idx] for idx in plot] for plot in scan_index_plots]


def _write_column_notes(
    specs: list[CheckpointSpec],
    prune_specs: list[CheckpointSpec],
    args: argparse.Namespace,
) -> Path:
    seg_columns = _column_order(
        [spec.name for spec in specs],
        args.image_column_position,
        include_ground_truth=True,
    )
    prune_columns = _column_order(
        [spec.name for spec in prune_specs],
        args.image_column_position,
        include_ground_truth=False,
    )
    lines = [
        f"style_version: {STYLE_VERSION}",
        "",
        "seg_soft columns:",
        *[f"  {i + 1}. {name}" for i, name in enumerate(seg_columns)],
        "",
        "selected_pruning_policy columns:",
        *[f"  {i + 1}. {name}" for i, name in enumerate(prune_columns)],
        "",
        "notes:",
        "  - seg_soft model columns follow the --checkpoint order.",
        "  - selected_pruning_policy defaults to Soft Mask, Gumbel_penulti, aux_route_penultimate.",
        "  - Black, cyan, and orange mark tokens first pruned at the three paper stages.",
        "  - Tokens retained through every stage preserve the input image colours.",
        "  - If a custom pruning-policy checkpoint exposes no selection maps, its column is omitted.",
    ]
    path = args.output_dir / "column_notes.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _select_positions(
    candidates: list[Candidate],
    model_mious: dict[str, list[float]],
    total_needed: int,
    fixed_indices: list[int] | None,
    selection_pool_size: int | None,
    shuffle_selected: bool,
    seed: int,
) -> list[int]:
    if fixed_indices:
        index_to_pos = {c.scan_index: i for i, c in enumerate(candidates)}
        missing = [idx for idx in fixed_indices if idx not in index_to_pos]
        if missing:
            raise ValueError(f"Requested --sample-index values not scanned: {missing}")
        selected = [index_to_pos[idx] for idx in fixed_indices[:total_needed]]
        if shuffle_selected:
            random.Random(seed).shuffle(selected)
        return selected

    scores = []
    for pos, cand in enumerate(candidates):
        vals = [mious[pos] for mious in model_mious.values()]
        fg_pixels = int(((cand.mask > 0) & (cand.mask != 255)).sum().item())
        scores.append((float(np.std(vals)), fg_pixels, pos))
    scores.sort(reverse=True)
    pool_size = int(selection_pool_size or total_needed)
    pool_size = max(total_needed, min(pool_size, len(scores)))
    selected = [pos for _, _, pos in scores[:pool_size]]
    if shuffle_selected:
        random.Random(seed).shuffle(selected)
    return selected[:total_needed]


def _split_selected_positions(
    selected: list[int],
    rows: int,
    num_plots: int,
) -> list[list[int]]:
    total_needed = rows * num_plots
    if len(selected) < total_needed:
        raise ValueError(
            f"Need {total_needed} selected samples for {num_plots} plot(s) "
            f"with {rows} rows each, but only have {len(selected)}."
        )
    return [
        selected[i * rows : (i + 1) * rows]
        for i in range(num_plots)
    ]


def _save_seg_soft_grid(
    selected: list[int],
    candidates: list[Candidate],
    model_probs: dict[str, list[np.ndarray]],
    ref_cfg: DictConfig,
    args: argparse.Namespace,
    plot_idx: int = 0,
    num_plots: int = 1,
) -> Path:
    mean = list(ref_cfg.DATASET.MEAN)
    std = list(ref_cfg.DATASET.STD)
    ignore = int(ref_cfg.DATASET.IGNORE_INDEX)
    rendered_rows = []
    for pos in selected:
        cand = candidates[pos]
        img_np = _denorm(cand.image, mean, std)
        image_panel = _resize_np(img_np, args.tile_size, is_mask=False)
        gt_panel = _resize_np(
            _mask_panel(img_np, cand.mask, ignore, args.gt_mode),
            args.tile_size,
            is_mask=args.gt_mode == "mask",
        )
        row = []
        if args.image_column_position == "start":
            row.append(image_panel)
        row.append(gt_panel)
        for probs in model_probs.values():
            row.append(
                _resize_np(
                    _prob_panel(
                        img_np,
                        probs[pos],
                        cmap=args.softmax_cmap,
                        mode=args.soft_panel,
                        alpha=float(args.alpha),
                    ),
                    args.tile_size,
                    is_mask=False,
                )
            )
        if args.image_column_position == "end":
            row.append(image_panel)
        rendered_rows.append(row)

    if num_plots == 1:
        filename = f"seg_soft_qblock{args.qblock}_top{len(selected)}.png"
    else:
        filename = (
            f"seg_soft_qblock{args.qblock}_plot{plot_idx + 1:02d}"
            f"_rows{len(selected)}.png"
        )
    out = args.output_dir / filename
    _compose_grid(
        rendered_rows,
        out,
        pad=int(args.pad),
        dpi=int(args.dpi),
        extra_col_gaps=_dense_separator_gaps(args, len(model_probs)),
    )
    return out


def _capture_prune_policy(
    spec: CheckpointSpec,
    selected_candidates: list[Candidate],
    args: argparse.Namespace,
    device: torch.device,
) -> list[np.ndarray] | None:
    _seed_everything(args.seed)
    trainer = _load_trainer_for_checkpoint(spec, args, device)
    raw_model = trainer.model
    if not (
        hasattr(raw_model, "set_selection_capture")
        and hasattr(raw_model, "get_selection_maps")
    ):
        del trainer
        return None

    out: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in _chunks(selected_candidates, int(args.eval_batch_size)):
            images = torch.stack([c.image for c in chunk]).to(device)
            was_training = raw_model.training
            raw_model.eval()
            raw_model.set_selection_capture(True)
            try:
                raw_model(images)
                selection_maps = raw_model.get_selection_maps()
            finally:
                raw_model.set_selection_capture(False)
                raw_model.train(was_training)
            if not selection_maps:
                out.extend([None for _ in chunk])
                continue
            keeps = torch.stack([s["keep"] for s in selection_maps], dim=0).bool()
            for i in range(keeps.shape[1]):
                out.append(keeps[:, i].cpu().numpy())

    del trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    if any(x is None for x in out):
        return None
    return out


def _save_prune_policy_grid(
    selected: list[int],
    candidates: list[Candidate],
    ref_cfg: DictConfig,
    args: argparse.Namespace,
    policy_by_model: dict[str, list[np.ndarray]],
    plot_idx: int = 0,
    num_plots: int = 1,
) -> Path | None:
    from data.visualisations import prune_policy_overlay

    selected = selected[: max(1, int(args.prune_policy_rows))]
    selected_candidates = [candidates[pos] for pos in selected]
    if not policy_by_model:
        return None

    mean = list(ref_cfg.DATASET.MEAN)
    std = list(ref_cfg.DATASET.STD)
    rendered_rows = []
    for row_i, cand in enumerate(selected_candidates):
        img_np = _denorm(cand.image, mean, std)
        image_panel = _resize_np(img_np, args.tile_size, is_mask=False)
        row = []
        if args.image_column_position == "start":
            row.append(image_panel)
        for policy_maps in policy_by_model.values():
            panel = prune_policy_overlay(
                img_np, torch.from_numpy(policy_maps[row_i])
            )
            row.append(_resize_np(panel, args.tile_size, is_mask=False))
        if args.image_column_position == "end":
            row.append(image_panel)
        rendered_rows.append(row)

    if num_plots == 1:
        filename = f"selected_pruning_policy_top{len(selected)}.png"
    else:
        filename = (
            "selected_pruning_policy"
            f"_plot{plot_idx + 1:02d}_rows{len(selected)}.png"
        )
    out = args.output_dir / filename
    _compose_grid(rendered_rows, out, pad=int(args.pad), dpi=int(args.dpi))
    return out


def _capture_prune_policy_by_model(
    selected: list[int],
    candidates: list[Candidate],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[np.ndarray]]:
    selected = selected[: max(1, int(args.prune_policy_rows))]
    selected_candidates = [candidates[pos] for pos in selected]
    policy_by_model: dict[str, list[np.ndarray]] = {}
    for spec in _prune_policy_specs(args):
        policy = _capture_prune_policy(spec, selected_candidates, args, device)
        if policy is not None:
            policy_by_model[spec.name] = policy
    return policy_by_model


def _write_metadata(
    selected_plots: list[list[int]],
    candidates: list[Candidate],
    specs: list[CheckpointSpec],
    prune_specs: list[CheckpointSpec],
    model_mious: dict[str, list[float]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    flat_selected = [pos for plot in selected_plots for pos in plot]
    rows = []
    for rank, pos in enumerate(flat_selected):
        vals = {name: model_mious[name][pos] for name in model_mious}
        miou_std = float(np.std(list(vals.values()))) if vals else None
        rows.append(
            {
                "rank": rank,
                "scan_index": candidates[pos].scan_index,
                "meta": candidates[pos].meta,
                "miou_std": miou_std,
                "miou_by_model": vals,
            }
        )
    payload = {
        "style_version": STYLE_VERSION,
        "style": {
            "image_column_position": args.image_column_position,
            "soft_panel": args.soft_panel,
            "softmax_cmap": args.softmax_cmap,
            "soft_alpha": float(args.alpha),
            "dense_separator_extra_pad": int(args.dense_separator_extra_pad),
            "prune_policy_rows": int(args.prune_policy_rows),
        },
        "column_order": {
            "seg_soft": _column_order(
                [spec.name for spec in specs],
                args.image_column_position,
                include_ground_truth=True,
            ),
            "selected_pruning_policy": _column_order(
                [spec.name for spec in prune_specs],
                args.image_column_position,
                include_ground_truth=False,
            ),
        },
        "checkpoint_order": [
            {"name": spec.name, "path": str(spec.path)} for spec in specs
        ],
        "prune_policy_checkpoint_order": [
            {"name": spec.name, "path": str(spec.path)} for spec in prune_specs
        ],
        "plots": [
            {
                "plot_index": plot_idx,
                "scan_indices": [candidates[pos].scan_index for pos in plot],
            }
            for plot_idx, plot in enumerate(selected_plots)
        ],
        "selected_samples": rows,
    }
    path = output_dir / "selection_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate high-resolution paper qualitative segmentation grids."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_parse_name_path,
        help=(
            "Ordered checkpoint spec NAME=PATH. Repeat to set columns. "
            "Defaults to the Question-A core checkpoints from this script."
        ),
    )
    parser.add_argument(
        "--prune-policy-checkpoint",
        action="append",
        type=_parse_name_path,
        help=(
            "Optional NAME=PATH checkpoint for the selected-pruning-policy grid. "
            "Repeat up to three times; defaults to Soft Mask, Gumbel, Aux."
        ),
    )
    parser.add_argument(
        "--only-prune-policy",
        action="store_true",
        help="Generate only selected-pruning-policy plots; skip segmentation inference.",
    )
    parser.add_argument(
        "--selection-metadata",
        type=Path,
        default=None,
        help="Reuse selected scan_indices from a previous selection_metadata.json.",
    )
    parser.add_argument("--config-path", default=None, help="Fallback config_path if checkpoint lacks cfg.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--num-plots", type=int, default=1, help="Number of separate grids to write.")
    parser.add_argument("--candidate-batches", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--sample-index",
        type=int,
        action="append",
        help=(
            "Fixed scanned sample index. Repeat rows*num-plots times to fully "
            "control multi-plot output."
        ),
    )
    parser.add_argument(
        "--selection-pool-size",
        type=int,
        default=None,
        help=(
            "Auto-selection pool size before shuffling. Defaults to rows*num-plots; "
            "use a larger value to randomise among the top-K disagreement cases."
        ),
    )
    parser.add_argument(
        "--shuffle-selected",
        action="store_true",
        help="Shuffle selected candidates reproducibly before splitting them into plots.",
    )
    parser.add_argument("--qblock", type=int, default=0, help="Existing qblock convention: 0=final, -1=penultimate, ...")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=None, help="Optional square DATASET.IMG_SIZE override.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "paper_figures" / "qA_core_qualitative",
    )
    parser.add_argument("--tile-size", type=int, default=None, help="Rendered square tile size; default keeps native size.")
    parser.add_argument("--pad", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--gt-mode", choices=["overlay", "mask"], default="overlay")
    parser.add_argument("--soft-panel", choices=["overlay", "heatmap"], default="overlay")
    parser.add_argument(
        "--image-column-position",
        choices=["start", "end", "none"],
        default="start",
        help="Where to place the raw input image reference column.",
    )
    parser.add_argument(
        "--dense-separator-extra-pad",
        type=int,
        default=27,
        help="Extra white gutter after the first method column, used to separate Dense from pruning/routing methods.",
    )
    parser.add_argument("--softmax-cmap", default="magma")
    parser.add_argument(
        "--prune-policy-rows",
        type=int,
        default=2,
        help="Rows in selected-pruning-policy grids; capped by --rows.",
    )
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument(
        "--skip-prune-policy",
        action="store_true",
        help="Only generate the Figure 3 segmentation-probability grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = Path(args.output_dir)
    rows = int(args.rows)
    num_plots = int(args.num_plots)
    if rows <= 0:
        raise ValueError("--rows must be positive.")
    if num_plots <= 0:
        raise ValueError("--num-plots must be positive.")
    total_needed = rows * num_plots
    specs = args.checkpoint or _default_specs()
    prune_specs = _prune_policy_specs(args)
    if not specs:
        raise ValueError("At least one checkpoint is required.")
    if args.only_prune_policy and args.skip_prune_policy:
        raise ValueError("--only-prune-policy conflicts with --skip-prune-policy.")

    metadata_scan_plots = None
    if args.selection_metadata is not None:
        metadata_scan_plots = _load_selected_scan_indices_from_metadata(args.selection_metadata)
        if not metadata_scan_plots:
            raise ValueError(f"No selected scan_indices found in {args.selection_metadata}.")
        num_plots = len(metadata_scan_plots)
        total_needed = sum(len(plot) for plot in metadata_scan_plots)

    if args.only_prune_policy and metadata_scan_plots is None and not args.sample_index:
        raise ValueError(
            "--only-prune-policy needs --selection-metadata or explicit --sample-index values; "
            "otherwise there is no mIoU-based selector to run."
        )

    _seed_everything(args.seed)
    device = _device(args.device)
    print(f"[paper-qual] device={device}")
    print("[paper-qual] checkpoint order:")
    for spec in specs:
        print(f"  - {spec.name}: {spec.path}")

    ref_cfg = _sanitise_cfg_for_plotting(
        _load_cfg_from_checkpoint(specs[0].path, args.config_path),
        args,
    )
    data_module = _build_data_module(ref_cfg)
    candidate_batches = int(args.candidate_batches)
    if metadata_scan_plots is not None:
        max_scan_index = max(idx for plot in metadata_scan_plots for idx in plot)
        candidate_batches = max(candidate_batches, max_scan_index + 1)
    candidates = _collect_candidates(data_module, args.split, candidate_batches)
    if metadata_scan_plots is None and len(candidates) < total_needed:
        raise RuntimeError(
            f"Only found {len(candidates)} candidate samples, need {total_needed} "
            f"for {num_plots} plot(s) × {rows} row(s)."
        )
    print(f"[paper-qual] scanned {len(candidates)} candidate samples from {args.split}.")

    model_probs: dict[str, list[np.ndarray]] = {}
    model_mious: dict[str, list[float]] = {}
    if not args.only_prune_policy:
        for spec in specs:
            print(f"[paper-qual] running {spec.name}")
            probs, mious = _compute_model_outputs(spec, candidates, args, device)
            model_probs[spec.name] = probs
            model_mious[spec.name] = mious

    if metadata_scan_plots is not None:
        selected_plots = _positions_from_scan_index_plots(candidates, metadata_scan_plots)
    else:
        selected = _select_positions(
            candidates,
            model_mious,
            total_needed=total_needed,
            fixed_indices=args.sample_index,
            selection_pool_size=args.selection_pool_size,
            shuffle_selected=bool(args.shuffle_selected),
            seed=int(args.seed),
        )
        selected_plots = _split_selected_positions(selected, rows, num_plots)
    print(
        "[paper-qual] selected scan indices:",
        [[candidates[pos].scan_index for pos in plot] for plot in selected_plots],
    )

    seg_paths = []
    if not args.only_prune_policy:
        for plot_idx, plot_selected in enumerate(selected_plots):
            seg_paths.append(
                _save_seg_soft_grid(
                    plot_selected,
                    candidates,
                    model_probs,
                    ref_cfg,
                    args,
                    plot_idx=plot_idx,
                    num_plots=num_plots,
                )
            )
    meta_path = _write_metadata(
        selected_plots,
        candidates,
        specs,
        prune_specs,
        model_mious,
        args.output_dir,
        args,
    )
    column_notes_path = _write_column_notes(specs, prune_specs, args)
    for seg_path in seg_paths:
        print(f"[paper-qual] wrote {seg_path}")
    print(f"[paper-qual] wrote {meta_path}")
    print(f"[paper-qual] wrote {column_notes_path}")

    if not args.skip_prune_policy:
        for plot_idx, plot_selected in enumerate(selected_plots):
            policy_by_model = _capture_prune_policy_by_model(
                plot_selected,
                candidates,
                args,
                device,
            )
            if not policy_by_model:
                print("[paper-qual] pruning policy skipped: no checkpoint captured keep maps.")
                continue
            prune_path = _save_prune_policy_grid(
                plot_selected,
                candidates,
                ref_cfg,
                args,
                policy_by_model,
                plot_idx=plot_idx,
                num_plots=num_plots,
            )
            if prune_path is not None:
                print(f"[paper-qual] wrote {prune_path}")


if __name__ == "__main__":
    main()
