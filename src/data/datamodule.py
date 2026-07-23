from __future__ import annotations

import random

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def _worker_init_fn(_worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import cv2
        cv2.setRNGSeed(int(seed % (2**31)))
    except ImportError:
        pass


def _seg_collate(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    masks = [b["mask"] for b in batch]
    meta = [b["meta"] for b in batch]
    try:
        masks = torch.stack(masks)
    except RuntimeError:
        pass
    out = {"image": images, "mask": masks, "meta": meta}
    if "targets" in batch[0]:
        out["targets"] = [b["targets"] for b in batch]
    return out


class DataModule:
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset = None,
        local_rank: int = 0,
    ):
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self._distributed = dist.is_initialized()
        self._collate_fn = _seg_collate
        self._val_collate_fn = self._collate_fn
        # We add local_rank so that each GPU gets a different but deterministic
        # sequence of augmentations.
        self._generator = torch.Generator().manual_seed(cfg.TRAIN.SEED + max(local_rank, 0))

    def _get_num_workers(self):
        """Get number of workers, clamped to max 2 on MPS."""
        num_workers = self.cfg.TRAIN.NUM_WORKERS
        if torch.backends.mps.is_available():
            num_workers = min(num_workers, 2)
        return num_workers

    def _prefetch_factor(self):
        """Compute prefetch factor based on image size."""
        num_workers = self._get_num_workers()
        if num_workers == 0:
            return None
        return 4 if self.cfg.DATASET.IMG_SIZE[0] < 512 else 2

    def train_loader(self) -> DataLoader:
        sampler = (
            DistributedSampler(
                self.train_dataset, shuffle=True, seed=self.cfg.TRAIN.SEED
            )
            if self._distributed
            else None
        )
        num_workers = self._get_num_workers()
        loader_kwargs = dict(
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn,
            worker_init_fn=_worker_init_fn,
            persistent_workers=num_workers > 0,
            generator=self._generator,
        )
        pf = self._prefetch_factor()
        if pf is not None:
            loader_kwargs["prefetch_factor"] = pf
        return DataLoader(self.train_dataset, **loader_kwargs)

    def _eval_batch_size(self) -> int:
        """Val/test batch size: TRAIN.VAL_BATCH_SIZE if >0, else 1.5x train."""
        override = int(self.cfg.TRAIN.get("VAL_BATCH_SIZE", -1))
        if override > 0:
            return override
        return int(1.5 * self.cfg.TRAIN.BATCH_SIZE)

    def val_loader(self) -> DataLoader:
        sampler = DistributedSampler(
            self.val_dataset, shuffle=False, seed=self.cfg.TRAIN.SEED) if self._distributed else None
        val_batch_size = self._eval_batch_size()
        num_workers = self._get_num_workers()
        loader_kwargs = dict(
            batch_size=val_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._val_collate_fn,
            worker_init_fn=_worker_init_fn,
            persistent_workers=num_workers > 0,
        )
        pf = self._prefetch_factor()
        if pf is not None:
            loader_kwargs["prefetch_factor"] = pf
        return DataLoader(self.val_dataset, **loader_kwargs)

    def test_loader(self) -> DataLoader | None:
        if self.test_dataset is None:
            return None
        sampler = DistributedSampler(
            self.test_dataset, shuffle=False, seed=self.cfg.TRAIN.SEED) if self._distributed else None
        test_batch_size = self._eval_batch_size()
        # persistent workers disabled as only run once on train end
        num_workers = self._get_num_workers()
        loader_kwargs = dict(
            batch_size=test_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._val_collate_fn,
            worker_init_fn=_worker_init_fn,
        )
        pf = self._prefetch_factor()
        if pf is not None:
            loader_kwargs["prefetch_factor"] = pf
        return DataLoader(self.test_dataset, **loader_kwargs)
