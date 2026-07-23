"""Medical imaging segmentation datasets.

Supported datasets:
  - FIVES: Fundus Image Vessel Segmentation dataset (binary, 2 classes).
    Expected structure:
        ROOT_DIR/
            img_dir/
                train/   — *.png
                val/     — *.png
                test/    — *.png
            ann_dir/
                train/   — *.png  (0=background, 1=vessel)
                val/     — *.png
                test/    — *.png
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset


class FIVESTransform:
    """Albumentations-based augmentation pipeline for FIVES.

    Train: 90°-multiple rotations (valid for retinal quadrant images) + colour
           and texture augmentations.
    Val/Test: resize only.
    """

    def __init__(self, cfg: DictConfig, split: str):
        h, w = cfg.DATASET.IMG_SIZE[0], cfg.DATASET.IMG_SIZE[1]
        mean = list(cfg.DATASET.MEAN)
        std = list(cfg.DATASET.STD)

        if split == "train":
            jitter = float(cfg.AUGMENTATIONS.get("COLOR_JITTER", 0.3))
            sat_jitter = float(cfg.AUGMENTATIONS.get("SATURATION_JITTER", 0.2))
            hue_jitter = float(cfg.AUGMENTATIONS.get("HUE_JITTER", 0.05))
            scale_min = float(cfg.AUGMENTATIONS.get("SCALE_CROP_MIN", 1.0))
            elastic_p = float(cfg.AUGMENTATIONS.get("ELASTIC_P", 0.0))
            dropout_p = float(cfg.AUGMENTATIONS.get("COARSE_DROPOUT_P", 0.0))
            noise_range = tuple(
                cfg.AUGMENTATIONS.get("GAUSS_NOISE_STD_RANGE", [0.01, 0.05])
            )

            # RandomResizedCrop when scale_min < 1, else plain Resize.
            # ratio=(1,1) keeps the square fundus quadrant aspect ratio.
            if scale_min < 1.0:
                spatial_head = A.RandomResizedCrop(
                    size=(h, w), scale=(scale_min, 1.0), ratio=(1.0, 1.0)
                )
            else:
                spatial_head = A.Resize(h, w)

            ops = [
                spatial_head,
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.75),
                A.ColorJitter(brightness=jitter, contrast=jitter,
                              saturation=sat_jitter, hue=hue_jitter, p=0.8),
                A.OneOf(
                    [
                        A.Sharpen(alpha=(0.1, 0.25), lightness=(0.9, 1.05)),
                        A.UnsharpMask(blur_limit=(3, 5), sigma_limit=0.5),
                    ],
                    p=0.4,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(std_range=noise_range),
                        A.GaussianBlur(blur_limit=(3, 7)),
                    ],
                    p=0.4,
                ),
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(80, 120)),
                        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
                    ],
                    p=0.3,
                ),
            ]
            if elastic_p > 0.0:
                ops.append(A.ElasticTransform(
                    alpha=5.0, sigma=50.0,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=elastic_p,
                ))
            if dropout_p > 0.0:
                # fill_mask=None (default) leaves the segmentation mask intact —
                # the model must predict vessels through zeroed image patches.
                ops.append(A.CoarseDropout(
                    num_holes_range=(4, 8),
                    hole_height_range=(0.05, 0.08),
                    hole_width_range=(0.05, 0.08),
                    fill=0,
                    p=dropout_p,
                ))
            ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]

            self.pipeline = A.Compose(ops, seed=cfg.TRAIN.SEED)
        else:
            self.pipeline = A.Compose(
                [
                    A.Resize(h, w),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ],
            )

    def __call__(self, image, mask):
        """
        Args:
            image: PIL Image (RGB)
            mask:  PIL Image (single-channel uint8, values 0/1)
        Returns:
            image: float tensor [3, H, W]
            mask:  long tensor [H, W]
        """
        img_np = np.array(image)
        mask_np = np.array(mask)
        out = self.pipeline(image=img_np, mask=mask_np)
        # ToTensorV2 gives [C,H,W] for image and [H,W] for mask
        return out["image"].float(), out["mask"].long()


class FIVESDataset(Dataset):
    """FIVES fundus vessel segmentation dataset.

    Annotations are binary PNGs (pixel values 0 or 1).
    NUM_CLASSES=2, IGNORE_INDEX=255.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        img_dir = os.path.join(root_dir, "img_dir", split)
        ann_dir = os.path.join(root_dir, "ann_dir", split)

        self.images = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir) if f.endswith(".png")
        ])
        self.annotations = sorted([
            os.path.join(ann_dir, f)
            for f in os.listdir(ann_dir) if f.endswith(".png")
        ])
        assert len(self.images) == len(self.annotations), (
            f"Mismatch: {len(self.images)} images vs {len(self.annotations)} annotations"
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.images[idx]).convert("RGB")
        raw_mask = np.array(Image.open(self.annotations[idx]))
        # Binarise: any non-zero pixel → class 1 (handles masks stored as 0/255 or 0/219 etc.)
        mask = Image.fromarray((raw_mask > 0).astype(np.uint8))

        original_size = image.size[::-1]  # (H, W)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Pre-build Mask2Former targets here to keep mask.unique() off-device at train
        # time. Background is excluded — unmatched queries carry it as "no object".
        classes = mask.unique()
        classes = classes[(classes != 255) & (classes > 0)]
        if len(classes) == 0:
            H, W = mask.shape[-2:]
            binary_masks = torch.zeros((0, H, W), dtype=torch.float)
        else:
            binary_masks = torch.stack([(mask == c).float() for c in classes])
        targets = {"labels": classes.long(), "masks": binary_masks}

        return {
            "image": image,
            "mask": mask,
            "targets": targets,
            "meta": {
                "filename": os.path.basename(self.images[idx]),
                "original_size": original_size,
            },
        }
