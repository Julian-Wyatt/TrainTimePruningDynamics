import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# ── OmegaConf Resolvers ───────────────────────────────────────────────────

def _lowercase_resolver(x: Any) -> Any:
    return x.lower() if isinstance(x, str) else x


if not OmegaConf.has_resolver("lowercase"):
    OmegaConf.register_new_resolver("lowercase", _lowercase_resolver)


@dataclass
class Dataset:
    NAME: str = ""
    ROOT_DIR: str = ""
    IMG_SIZE: List[int] = field(default_factory=lambda: [512, 512])
    NUM_CLASSES: int = 150
    IGNORE_INDEX: int = 255
    # Retry train-time segmentation augmentation if a crop contains no valid labels.
    RESAMPLE_EMPTY_MASKS: bool = True
    EMPTY_MASK_MAX_RETRIES: int = 10
    # if True, exclude class 0 from mIoU (foreground-only, matches TP for binary datasets)
    EXCLUDE_BACKGROUND_CLASS: bool = False
    # foreground channel index for binary segmentation (used by binary_pred_from_logits)
    BINARY_FG_CHANNEL: int = 1
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    OVERFIT_BATCHES: int = 0

@dataclass
class Model:
    BACKBONE: str = "vit_base_patch16_dinov3.lvd1689m"
    CKPT_PATH: str = ""
    PATCH_SIZE: int = 16
    DROP_PATH_RATE: float = 0.0
    FREEZE_BACKBONE_N_BLOCKS: int = 0
    UNFREEZE_BACKBONE_EPOCH: int = -1

    EMA_DECAY: float = 0.9999
    # Use EMA weights during validation; disable if EMA causes collapse
    EMA_VALIDATE: bool = True

    # EoMT segmentation head
    NUM_QUERIES: int = 200
    NUM_QUERY_BLOCKS: int = 3
    MASKED_ATTN: bool = True
    # 2x ScaleBlock upscalers in the mask decode; <=0 => auto (log2(patch_size) - 2).
    # Lowering it trades mask resolution for decode throughput.
    EOMT_NUM_UPSCALE: int = -1

    # CROPR token pruning
    USE_CROPR: bool = False
    # tokens pruned per block (88 ≈ same keep ratio as large=40)
    CROPR_PRUNING_RATE: int = 88

    # Generic routing window — shared convention across CROPR/EoMT ablations
    # routing_end < 0 resolves from EOMT_PRUNE_REINJECT where supported.
    ROUTING_START: int = 2
    ROUTING_END: int = 10
    # CROPR ablation controls
    CROPR_STATIC_PRUNING: bool = False        # prune once upfront (same total budget as progressive)
    CROPR_STATIC_PRUNE_MULT: float = 1.0      # static prune rate multiplier (e.g. 6.0 for ~6x upfront budget)
    CROPR_SCHEDULE: str = "legacy"            # legacy|progressive_tokens|progressive_fraction|static_early|early_reinject
    CROPR_KEEP_FRACTION: Optional[float] = None
    CROPR_DERIVE_PRUNING_RATE: bool = False
    CROPR_REFERENCE_PRUNE_TOKENS: int = 40
    CROPR_REFERENCE_SPATIAL_TOKENS: int = 1024
    # Absolute block indices where a fraction-schedule prunes; empty => every routed
    # block in [ROUTING_START, ROUTING_END). [2, 5, 8] gives EoMT's 3-stage geometric keep.
    CROPR_PRUNE_BLOCKS: List[int] = field(default_factory=list)
    PRUNE_ROUND_MULTIPLE: int = 8

    # EoMT learned token-pruning ablations for FIVES research Question A
    USE_EOMT_PRUNING: bool = False
    EOMT_PRUNE_LAYERS: List[int] = field(default_factory=lambda: [2, 5, 8])
    # Per-stage *relative* keep rate; stages compound (0.7, 0.7^2, 0.7^3 -> ~34%).
    EOMT_KEEP_RATES: List[float] = field(default_factory=lambda: [0.7, 0.7, 0.7])
    # dense|soft_mask|gumbel_route|hard_route|random_route
    EOMT_PRUNE_TRAIN_MODE: str = "dense"
    EOMT_PRUNE_EVAL_MODE: str = "dense"   # dense|hard|random
    # routed-mode reinjection point: q_start|mask_pred|penultimate
    EOMT_PRUNE_REINJECT: str = "penultimate"
    # random eval mode: 0=learned top-k, 1=fully random, (0,1)=swap that fraction
    EOMT_PRUNE_RANDOM_RATIO: float = 1.0
    EOMT_PRUNE_LOSS_WEIGHT: float = 1.0
    EOMT_PRUNE_GUMBEL_TAU: float = 1.0
    EOMT_PRUNE_RAMP_EPOCHS: int = 0
    # CROPR scorer: learned queries directly score tokens and receive train-only BCE.
    EOMT_PRUNE_AUX_SCORER: bool = False
    EOMT_PRUNE_AUX_LOSS_WEIGHT: float = 1.0
    # Randomise routed-mode start/end within the configured fallback bounds.
    EOMT_PRUNE_RANDOM_WINDOW_BOUNDS: bool = False
    # Keep rate for routing-pruned tokens, separate from the learned-prune budget
    # in EOMT_KEEP_RATES. <0 => fall back to EOMT_KEEP_RATES.
    EOMT_PRUNE_ROUTING_KEEP_RATE: float = -1.0
    # Train-only random-route regulariser window for random_route.
    # The route starts after START block has run and ends after END block has run.
    EOMT_PRUNE_ROUTE_START_AFTER_BLOCK: int = -1
    EOMT_PRUNE_ROUTE_END_AFTER_BLOCK: int = -1

    # EoMT attention mask annealing (hard → soft over training steps)
    ATTN_MASK_ANNEALING_ENABLED: bool = True
    ATTN_MASK_ANNEALING_START_STEPS: List[int] = field(
        default_factory=lambda: [2, 4, 6, 8])
    ATTN_MASK_ANNEALING_END_STEPS: List[int] = field(
        default_factory=lambda: [4, 6, 8, 10])
    ATTN_MASK_POLY_POWER: float = 0.9


@dataclass
class Train:
    BATCH_SIZE: int = 4
    EFFECTIVE_BATCH_SIZE: Optional[int] = None
    # Val/test loader batch size; <=0 => int(1.5 * BATCH_SIZE). Set explicitly to
    # decouple eval throughput from the memory-bound train batch size.
    VAL_BATCH_SIZE: int = -1
    LR: float = 0.0002
    WEIGHT_DECAY: float = 0.05
    EPOCHS: int = 50
    NUM_WORKERS: int = 8
    SEED: int = 42
    COMPILE: bool = True

    OPTIMIZER: str = "adamw"
    ADAM_BETAS: List[float] = field(default_factory=lambda: [0.9, 0.999])
    MOMENTUM: float = 0.9
    LLRD: float = 1.0           # LLRD applied when LLRD != 1.0
    LR_MULT: float = 1.0        # Backbone LR multiplier; only used when LLRD=1.0
    CLIP_GRAD_VAL: float = 1.0
    MIN_LR_FRAC: float = 0.0
    LLRD_FULL_LR_LAST_N_BLOCKS: bool = True
    ORIGINAL_EOMT_LR_MULT_COMPAT: bool = False

    TASK: str = "segmentation"
    # explicit trainer selection (auto = existing logic)
    TRAINER_TYPE: str = "auto"
    LOSS_TYPE: str = "mask2former"
    DICE_WEIGHT: float = 1.0  # weight for Dice term in ce_dice loss

    # Mask2Former loss weights
    M2F_NUM_POINTS: int = 12544
    M2F_OVERSAMPLE_RATIO: float = 3.0
    M2F_IMPORTANCE_SAMPLE_RATIO: float = 0.75
    M2F_MASK_COEF: float = 5.0
    M2F_DICE_COEF: float = 5.0
    M2F_CLASS_COEF: float = 2.0
    M2F_NO_OBJECT_COEF: float = 0.1

    SCHEDULE_NAME: str = "cosine"
    WARMUP_RATIO: float = 0.05
    WARMDOWN_RATIO: float = 0.0

    # Two-stage warmup (head params warm first, backbone delayed)
    SPLIT_WARMUP: bool = True
    WARMUP_STEPS: List[int] = field(default_factory=lambda: [10, 50])

    VAL_EVERY_N_EPOCHS: int = 5
    CKPT_EVERY_N_EPOCHS: int = 1
    # log Boundary IoU (adds per-step CPU overhead)
    LOG_BOUNDARY_IOU: bool = False
    LOG_CLDICE: bool = False
    LOG_THIN_VESSEL_DICE: bool = False
    BOUNDARY_WIDTH: int = 5
    # Checkpoint retention
    MAX_SAVED_CHECKPOINTS: int = 1  # keep only top-K best checkpoints; 0 = keep all

    LOG_TYPE: str = "none"
    PROJECT: str = ""

    RUN_TRAIN: bool = True
    RUN_TEST: bool = True
    SAVING_ROOT_DIR: str = "saves"
    DESCRIPTION: str = ""
    CHECKPOINT_FILE: str = ""
    MAX_RESUBMIT: int = 0

    # Best-checkpoint metric: key must match what compute_metrics returns (e.g. "mIoU", "pq")
    BEST_METRIC: str = "mIoU"
    # "auto" = enabled on CUDA, disabled on MPS; "true" = always on; "false" = always off
    SAVE_CHECKPOINTS: str = "auto"
    # Fast local smoke-test: overrides epochs, batch size, img size, disables checkpointing
    RUN_LOCAL_TEST: bool = False
    # Populated at runtime — do not set manually
    RUN_ID: str = ""
    FRESH_RUN: bool = True   # False when RUN_ID is inherited by a Slurm requeue

    # Throughput test: skip training, measure images/sec on a single GPU only.
    # Uses deterministic benchmarking settings; requires WORLD_SIZE=1.
    THROUGHPUT_TEST: bool = False
    THROUGHPUT_WARMUP_BATCHES: int = 50
    THROUGHPUT_MEASURE_BATCHES: int = 200
    # Print resolved config and exit immediately — no datasets or trainer built.
    DRY_RUN: bool = False

    # --- Results registry --------------------------------------------------
    # Stable identity shared by every seed: results-table row key and W&B run group.
    # Empty -> derived from the config_name leaf.
    EXPERIMENT_ID: str = ""
    # Results CSV this run appends to, one per sweep axis (qA_core, qA_keep_sweep,
    # qBC_cropr_vits). Empty -> derived from config_name.
    RESULTS_GROUP: str = ""
    # Base dir for the CSV tables. Empty -> "<SAVING_ROOT_DIR>/results".
    RESULTS_TABLE_DIR: str = ""
    # Force a re-run + overwrite of an existing (EXPERIMENT_ID, SEED) row instead
    # of skipping it.
    OVERRIDE_RESULTS: bool = False
    # Master switch for the registry (auto-skipped for debug/throughput runs).
    LOG_RESULTS_TABLE: bool = True


@dataclass
class Augmentations:
    RANDOM_SCALE: List[float] = field(default_factory=lambda: [0.1, 2.0])
    # None = disabled; if set, must match IMG_SIZE or be larger
    RANDOM_CROP: Optional[List[int]] = None
    HORIZONTAL_FLIP: float = 0.5
    COLOR_JITTER: float = 0.4
    # defaults to COLOR_JITTER if None
    CONTRAST_JITTER: Optional[float] = None
    # defaults to COLOR_JITTER if None
    SATURATION_JITTER: Optional[float] = None
    # hue delta in [0, 0.5]; 0 = disabled
    HUE_JITTER: float = 0.0
    USE_LSJ: bool = False

    # Medical imaging augmentations (consumed by FIVESTransform, src/data/medical.py)
    # albumentations GaussNoise std_range (normalized [0, 1])
    GAUSS_NOISE_STD_RANGE: List[float] = field(
        default_factory=lambda: [0.01, 0.05])
    # 1.0 = plain Resize; <1.0 = RandomResizedCrop(scale=(SCALE_CROP_MIN, 1.0))
    SCALE_CROP_MIN: float = 1.0
    # elastic deformation probability; 0.0 = disabled
    ELASTIC_P: float = 0.0
    # coarse dropout probability; 0.0 = disabled
    COARSE_DROPOUT_P: float = 0.0


@dataclass
class Config:
    DATASET: Dataset = field(default_factory=Dataset)
    MODEL: Model = field(default_factory=Model)
    TRAIN: Train = field(default_factory=Train)
    AUGMENTATIONS: Augmentations = field(default_factory=Augmentations)

    # Run-specific overrides
    PATH: str = ""
    run_id: Optional[str] = None
    desc: str = ""
    saving_root_dir: str = "./"
    config_path: Any = None
    config_name: str = ""


def load_cfg(file_name="") -> Config:
    base = OmegaConf.structured(Config())
    exp = OmegaConf.load(file_name)
    cfg = OmegaConf.merge(base, exp)
    return cfg


def _normalise_default_path(default: Any) -> str | None:
    """Return a config path from a Hydra-style defaults entry.

    Only the subset used by local experiment configs is supported:
    string entries, ``_self_``, and one-item dict entries.
    """
    if isinstance(default, str):
        value = default.strip()
    elif isinstance(default, dict):
        if not default:
            return None
        key, value = next(iter(default.items()))
        if value is None:
            return None
        key = str(key).removeprefix(
            "override ").removeprefix("optional ").strip()
        value = str(value).strip()
        value = value if key in ("", "_self_") else f"{key}/{value}"
    else:
        return None

    if value in ("", "_self_"):
        return None
    return value


def _resolve_default_yaml(
    default: Any,
    current_dir: str,
    configs_root: str,
) -> str | None:
    value = _normalise_default_path(default)
    if value is None:
        return None

    root_relative = value.startswith("/")
    value = value[1:] if root_relative else value
    if not value.endswith(".yaml"):
        value += ".yaml"

    candidates = [os.path.join(configs_root, value)]
    if not root_relative:
        candidates.insert(0, os.path.join(current_dir, value))

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not resolve defaults entry {default!r} from {current_dir}.\n"
        "Searched:\n" + "\n".join(f"  {p}" for p in candidates)
    )


def _load_yaml_with_defaults(
    path: str,
    configs_root: str,
    seen: set[str] | None = None,
) -> DictConfig:
    """Load a YAML file and recursively compose its local ``defaults`` list.

    Experiment configs are merged through ``config_path`` after Hydra has
    already loaded ``configs/config.yaml``, so Hydra does not process defaults
    inside those experiment files. This small composer gives those files the
    inheritance style we use for research configs without changing the CLI.
    """
    path = os.path.abspath(path)
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Config defaults cycle detected at {path}")
    seen.add(path)

    cfg = OmegaConf.load(path)
    if not OmegaConf.is_config(cfg):
        return cfg

    defaults = cfg.get("defaults")
    if defaults is None:
        return cfg

    cfg = cfg.copy()
    del cfg["defaults"]
    if path == os.path.abspath(os.path.join(configs_root, "config.yaml")):
        cfg = OmegaConf.create({
            key: cfg[key]
            for key in ("DATASET", "MODEL", "TRAIN", "AUGMENTATIONS")
            if key in cfg
        })
    merged = OmegaConf.create({})
    self_merged = False
    current_dir = os.path.dirname(path)

    for item in defaults:
        if isinstance(item, str) and item.strip() == "_self_":
            merged = OmegaConf.merge(merged, cfg)
            self_merged = True
            continue
        parent_path = _resolve_default_yaml(item, current_dir, configs_root)
        if parent_path is None:
            continue
        parent = _load_yaml_with_defaults(
            parent_path, configs_root, seen.copy())
        merged = OmegaConf.merge(merged, parent)

    if not self_merged:
        merged = OmegaConf.merge(merged, cfg)
    return merged


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


def resolve_config_paths(cfg: DictConfig, configs_root: str):
    sections = ["DATASET", "MODEL", "AUGMENTATIONS", "TRAIN"]
    for section in sections:
        if section not in cfg:
            continue
        value = cfg.get(section)
        if isinstance(value, str):
            yaml_path = os.path.join(configs_root, value)
            if os.path.exists(yaml_path):
                loaded = OmegaConf.load(yaml_path)
                if section in loaded and len(loaded.keys()) == 1:
                    loaded = loaded[section]
                cfg[section] = loaded
            else:
                print(
                    f"Warning: Section {section} refers to missing file '{value}'. Blanking.")
                cfg[section] = {}


def process_config(cfg: DictConfig, configs_root: str) -> DictConfig:
    if OmegaConf.is_config(cfg):
        OmegaConf.set_struct(cfg, False)

    if cfg.get("config_path"):
        print(f"Merging experiment choice: {cfg.config_path}")
        requested_config_path = cfg.config_path
        exp_cfg = cfg.config_path
        if isinstance(exp_cfg, str):
            # Collapse doubled "experiments/experiments/" prefix (slurm wrappers prepend it).
            normalized = exp_cfg
            while normalized.startswith("experiments/experiments/"):
                normalized = normalized[len("experiments/"):]
            requested_config_path = normalized
            candidate_names = (
                [exp_cfg] if normalized == exp_cfg else [normalized, exp_cfg])
            possible_paths = []
            for name in candidate_names:
                possible_paths += [
                    os.path.join(configs_root, name),
                    os.path.join(configs_root, name + ".yaml"),
                    os.path.join(configs_root, "experiments", name),
                    os.path.join(configs_root, "experiments", name + ".yaml"),
                ]
            loaded = False
            for p in possible_paths:
                if os.path.exists(p):
                    print(f"Loading experiment config from: {p}")
                    exp_cfg = _load_yaml_with_defaults(p, configs_root)
                    loaded = True
                    break
            if not loaded:
                raise FileNotFoundError(
                    f"Could not find experiment config '{cfg.config_path}'.\n"
                    f"Searched paths:\n" +
                    "\n".join(f"  {p}" for p in possible_paths)
                )

        if OmegaConf.is_config(exp_cfg):
            OmegaConf.set_struct(exp_cfg, False)
            resolve_config_paths(exp_cfg, configs_root)
            cfg = OmegaConf.merge(cfg, exp_cfg)

            try:
                if hydra.core.hydra_config.HydraConfig.initialized():
                    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
                    overrides = hydra_cfg.overrides.task
                    if overrides:
                        print(f"Re-applying overrides: {overrides}")
                        dotlist_overrides = []
                        for o in overrides:
                            if o.startswith("config_path="):
                                continue
                            if "=" in o:
                                key, val = o.split("=", 1)
                                if key.startswith("++"):
                                    key = key[2:]
                                elif key.startswith("+"):
                                    key = key[1:]
                                elif key.startswith("~"):
                                    continue
                                dotlist_overrides.append(f"{key}={val}")
                            else:
                                dotlist_overrides.append(o)
                        if dotlist_overrides:
                            override_conf = OmegaConf.from_dotlist(
                                dotlist_overrides)
                            cfg = OmegaConf.merge(cfg, override_conf)
            except Exception as e:
                print(f"Failed to re-apply overrides: {e}")

            cfg.config_name = requested_config_path
            del cfg["config_path"]

    resolve_config_paths(cfg, configs_root)

    if cfg.get("desc"):
        cfg.TRAIN.DESCRIPTION = cfg.desc.replace("<<COMMA>>", ",")
    if "desc" in cfg:
        # `desc` is only a CLI shorthand; keep TRAIN.DESCRIPTION as the single source.
        del cfg["desc"]
    if cfg.get("saving_root_dir"):
        cfg.TRAIN.SAVING_ROOT_DIR = cfg.saving_root_dir

    # Normalise case-insensitive string fields
    _LOWER_FIELDS = [
        "TRAIN.LOSS_TYPE",
        "TRAIN.OPTIMIZER",
        "TRAIN.SCHEDULE_NAME",
        "TRAIN.TASK",
        "TRAIN.TRAINER_TYPE",
        "TRAIN.LOG_TYPE",
        "TRAIN.SAVE_CHECKPOINTS",
        "MODEL.CROPR_SCHEDULE",
        "MODEL.EOMT_PRUNE_TRAIN_MODE",
        "MODEL.EOMT_PRUNE_EVAL_MODE",
        "MODEL.EOMT_PRUNE_REINJECT",
    ]
    for key in _LOWER_FIELDS:
        val = OmegaConf.select(cfg, key)
        if isinstance(val, str):
            OmegaConf.update(cfg, key, val.lower(), merge=False)

    return cfg
