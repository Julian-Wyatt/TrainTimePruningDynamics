"""Main entry point for paper training and evaluation runs."""

import datetime
import os
import random
import string
import sys

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('HYDRA_FULL_ERROR', '1')

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from core.conf import Config, process_config
from utils.device import get_device
from utils.resubmit import ResubmitManager

ResubmitManager.register_early()

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


def _should_log_results(cfg, is_debug_run, throughput_mode, test_metrics, last_val_metrics):
    """Return True when this run should append a row to the results registry.

    Conditions:
      - LOG_RESULTS_TABLE is enabled (default True)
      - run involved training OR testing (eval-only runs with RUN_TEST=True count)
      - not a debug or throughput run
      - at least one metric was produced
    """
    return (
        cfg.TRAIN.get("LOG_RESULTS_TABLE", True)
        and (cfg.TRAIN.RUN_TRAIN or cfg.TRAIN.RUN_TEST)
        and not is_debug_run
        and not throughput_mode
        and bool(test_metrics or last_val_metrics)
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    configs_root = os.path.join(os.path.dirname(__file__), "../../configs")
    configs_root = os.path.abspath(configs_root)
    cfg = process_config(cfg, configs_root)

    # Resolve the results-registry identity once, up front, so the W&B logger
    # (group/tags), the skip-guard, and the end-of-run append all agree.
    from utils import results_registry
    cfg.TRAIN.EXPERIMENT_ID = results_registry.resolve_experiment_id(cfg)
    cfg.TRAIN.RESULTS_GROUP = results_registry.resolve_results_group(cfg)

    # Generate a run ID, preserving a caller-supplied value across a requeue.
    if not cfg.TRAIN.RUN_ID:
        cfg.TRAIN.RUN_ID = (
            datetime.datetime.now().strftime("%y/%m/%d-%H:%M:%S")
            + "-"
            + "".join(random.choices(string.ascii_lowercase, k=5))
        )
    else:
        cfg.TRAIN.FRESH_RUN = False  # RUN_ID was inherited — W&B run must be resumed

    # Redirect stdout/stderr to train.log inside the Hydra output dir (saves/slurm_outputs/...)
    from hydra.core.hydra_config import HydraConfig

    from utils.logging import setup_output_log
    setup_output_log(HydraConfig.get().runtime.output_dir)

    # Apply saving_root_dir CLI override if provided
    if cfg.get("saving_root_dir"):
        cfg.TRAIN.SAVING_ROOT_DIR = cfg.saving_root_dir

    # Resolve SAVE_CHECKPOINTS: "auto" = enabled on CUDA only
    ckpt_setting = str(cfg.TRAIN.SAVE_CHECKPOINTS).lower()
    if ckpt_setting == "auto":
        cfg.TRAIN.SAVE_CHECKPOINTS = "true" if get_device() == "cuda" else "false"
        if cfg.TRAIN.SAVE_CHECKPOINTS == "false":
            print("SAVE_CHECKPOINTS=auto: MPS/CPU detected — checkpointing disabled.")

    # Check if running in debug mode (local test or overfitting)
    is_debug_run = cfg.TRAIN.RUN_LOCAL_TEST or cfg.DATASET.OVERFIT_BATCHES > 0

    # Local smoke-test overrides
    if cfg.TRAIN.RUN_LOCAL_TEST:
        print("=" * 60)
        print("RUN_LOCAL_TEST: applying smoke-test overrides")
        print("=" * 60)
        cfg.DATASET.OVERFIT_BATCHES = 1
        cfg.TRAIN.EPOCHS = 1
        cfg.TRAIN.VAL_EVERY_N_EPOCHS = 1
        cfg.TRAIN.RUN_TEST = False
        cfg.TRAIN.BATCH_SIZE = 2
        cfg.TRAIN.SAVE_CHECKPOINTS = "false"
        cfg.TRAIN.LOG_TYPE = "none"
        cfg.DATASET.IMG_SIZE = [160, 160]

    # Clamp random crop to image size — prevents feed-size mismatch when IMG_SIZE is
    # overridden to a smaller value than the dataset's default RANDOM_CROP setting.
    img_h, img_w = cfg.DATASET.IMG_SIZE
    crop_h, crop_w = cfg.AUGMENTATIONS.RANDOM_CROP
    if crop_h > img_h or crop_w > img_w:
        cfg.AUGMENTATIONS.RANDOM_CROP = [min(crop_h, img_h), min(crop_w, img_w)]

    # Set visualization frequencies for debug runs (OVERFIT_BATCHES without RUN_LOCAL_TEST)
    if is_debug_run and not cfg.TRAIN.RUN_LOCAL_TEST:
        cfg.TRAIN.VAL_EVERY_N_EPOCHS = 1
    # Runtime policy: efficient + reproducible by default, deterministic benchmarking in throughput mode
    throughput_mode = bool(cfg.TRAIN.get("THROUGHPUT_TEST", False))
    if throughput_mode:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        runtime_policy = "throughput-mode (deterministic benchmarking)"
    else:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = False
        runtime_policy = "train-mode (efficient + reproducible)"
    deterministic = torch.are_deterministic_algorithms_enabled()
    print(f"[Runtime] Policy: {runtime_policy} | matmul_precision={torch.get_float32_matmul_precision()} "
          f"cudnn_benchmark={torch.backends.cudnn.benchmark} deterministic={deterministic}")



    # Seed
    seed = cfg.TRAIN.SEED + max(local_rank, 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if get_device() == "cuda":
        torch.cuda.manual_seed_all(seed)

    if cfg.TRAIN.get("DRY_RUN", False):
        print(
            f"[DRY RUN] Config validated. experiment_id={cfg.TRAIN.EXPERIMENT_ID} "
            f"results_group={cfg.TRAIN.RESULTS_GROUP} backbone={cfg.MODEL.BACKBONE} "
            f"use_prune={cfg.MODEL.get('USE_EOMT_PRUNING')} "
            f"reinject={cfg.MODEL.get('EOMT_PRUNE_REINJECT')}. "
            f"Exiting before dataset/trainer init."
        )
        return

    # Skip (experiment, seed) cells already in a results table. The read is
    # deterministic, so all ranks agree before DDP init; FRESH_RUN lets requeues resume.
    if (
        cfg.TRAIN.get("LOG_RESULTS_TABLE", True)
        and cfg.TRAIN.FRESH_RUN
        and cfg.TRAIN.RUN_TRAIN
        and not is_debug_run
        and not throughput_mode
        and not cfg.TRAIN.get("OVERRIDE_RESULTS", False)
        and results_registry.has_result(cfg)
    ):
        print(
            f"[results] {cfg.TRAIN.EXPERIMENT_ID} seed={cfg.TRAIN.SEED} already in "
            f"{results_registry.results_csv_path(cfg)} — skipping "
            f"(set ++TRAIN.OVERRIDE_RESULTS=true to re-run)."
        )
        return

    if local_rank >= 0 and world_size > 1:
        from utils.distributed import setup_ddp
        setup_ddp(local_rank, world_size)

    # Build data
    from data.datamodule import DataModule
    from data.dataset_factory import build_datasets

    train_ds, val_ds, test_ds = build_datasets(cfg)
    data_module = DataModule(cfg, train_ds, val_ds, test_ds, local_rank=local_rank)

    # Build trainer (imports trigger registration)
    from trainer.base_trainer import TRAINER_REGISTRY
    from trainer.segmentation_trainer import EoMTTrainer

    trainer_type = str(cfg.TRAIN.get("TRAINER_TYPE", "auto")).lower()
    if trainer_type == "auto":
        if cfg.TRAIN.TASK == "classification":
            raise ValueError(
                "Classification is not included in this paper-focused repository.")
        trainer = EoMTTrainer(cfg, data_module)
    else:
        trainer_cls = TRAINER_REGISTRY.get(trainer_type)
        if trainer_cls is None:
            raise ValueError(
                f"Unknown TRAIN.TRAINER_TYPE '{trainer_type}'. Choose from: {list(TRAINER_REGISTRY)}."
            )
        trainer = trainer_cls(cfg, data_module)

    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Wrap model in DDP after trainer (and model) are fully constructed
    if local_rank >= 0 and world_size > 1:
        trainer.setup_ddp(local_rank, world_size)

    # Throughput test: single-GPU only, skips training and metrics
    if cfg.TRAIN.get("THROUGHPUT_TEST", False):
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            raise RuntimeError(
                f"THROUGHPUT_TEST requires a single GPU, but WORLD_SIZE={world_size}. "
                "Run without torchrun (or set WORLD_SIZE=1)."
            )
        from trainer.throughput import measure_throughput
        measure_throughput(trainer, data_module.val_loader())
        return

    try:
        # Train
        if cfg.TRAIN.RUN_TRAIN:
            trainer.train()

        # Load the best checkpoint before testing. Only rank 0 sets best_ckpt_path,
        # so broadcast it before every rank loads.
        if cfg.TRAIN.RUN_TEST:
            import torch.distributed as dist

            from utils.checkpoint import resume_from_checkpoint
            path_holder = [trainer.state.best_ckpt_path]
            if dist.is_initialized():
                dist.broadcast_object_list(path_holder, src=0)
            best_path = path_holder[0] or cfg.TRAIN.CHECKPOINT_FILE
            if best_path:
                resume_from_checkpoint(
                    best_path,
                    trainer.model,
                    optimizer=None,
                    ema=trainer.ema,
                    scaler=None,
                )

        # Test
        test_metrics = {}
        if cfg.TRAIN.RUN_TEST and data_module.test_loader() is not None:
            test_metrics = trainer.test()

        # Final summary (main process only)
        from utils.distributed import is_main_process
        if is_main_process():
            def _fmt_table(rows: list[tuple[str, str]]) -> str:
                """Render (label, value) rows as a fixed-width table."""
                if not rows:
                    return ""
                col1 = max(len(r[0]) for r in rows)
                col2 = max(len(r[1]) for r in rows)
                sep = "+" + "-" * (col1 + 2) + "+" + "-" * (col2 + 2) + "+"
                lines = [sep]
                for label, value in rows:
                    lines.append(f"| {label:<{col1}} | {value:<{col2}} |")
                    lines.append(sep)
                return "\n".join(lines)

            def _metric_rows(metrics: dict) -> list[tuple[str, str]]:
                rows = []
                for k, v in metrics.items():
                    if k.endswith("_std"):
                        continue
                    std = metrics.get(f"{k}_std")
                    val_str = f"{v:.4f} ± {std:.4f}" if std is not None else f"{v:.4f}"
                    rows.append((k, val_str))
                return rows

            rows: list[tuple[str, str]] = []
            if trainer.state.last_val_metrics:
                rows += [("", "")] if rows else []
                rows += [("[ Val ] " + k, v) for k, v in _metric_rows(trainer.state.last_val_metrics)]
            if test_metrics:
                rows += [("", "")] if rows else []
                rows += [("[Test ] " + k, v) for k, v in _metric_rows(test_metrics)]
            if trainer.state.best_metric_val > -float("inf"):
                rows.append((f"Best val {cfg.TRAIN.BEST_METRIC}", f"{trainer.state.best_metric_val:.4f}"))
            if trainer.state.best_train_loss < float("inf"):
                rows.append(("Best train loss", f"{trainer.state.best_train_loss:.4f}"))
            if trainer.state.best_ckpt_path:
                rows.append(("Best ckpt", trainer.state.best_ckpt_path))
            if trainer.state.train_end_ckpt_path:
                rows.append(("Train-end ckpt", trainer.state.train_end_ckpt_path))

            print("\n" + "=" * 60)
            print("TRAINING COMPLETE — FINAL SUMMARY")
            print("=" * 60)
            print(_fmt_table(rows))
            print("=" * 60)

            # Append core metrics to this experiment's results table.
            if _should_log_results(
                cfg, is_debug_run, throughput_mode,
                test_metrics, trainer.state.last_val_metrics,
            ):
                table_path = results_registry.append_result(
                    cfg, test_metrics, best_val=trainer.state.best_metric_val,
                )
                if table_path:
                    print(f"[results] appended {cfg.TRAIN.EXPERIMENT_ID} "
                          f"seed={cfg.TRAIN.SEED} → {table_path}")

    finally:
        trainer.logger.finish()


if __name__ == "__main__":
    main()
