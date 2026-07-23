"""FIVES dataset construction for the paper experiments."""

from __future__ import annotations

from omegaconf import DictConfig


def build_datasets(cfg: DictConfig):
    """Build the resized FIVES train, validation, and test datasets."""
    if cfg.DATASET.NAME.upper() not in {"FIVES", "MEDICAL"}:
        raise ValueError("This paper repository supports only DATASET.NAME=FIVES.")

    from .medical import FIVESDataset, FIVESTransform

    root_dir = cfg.DATASET.ROOT_DIR
    return (
        FIVESDataset(root_dir, "train", FIVESTransform(cfg, "train")),
        FIVESDataset(root_dir, "val", FIVESTransform(cfg, "val")),
        FIVESDataset(root_dir, "test", FIVESTransform(cfg, "test")),
    )
