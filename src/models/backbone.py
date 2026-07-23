from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

_DINOV3_HF_TO_TIMM = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": "vit_small_patch16_dinov3.lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m": "vit_base_patch16_dinov3.lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "vit_large_patch16_dinov3.lvd1689m",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": "vit_huge_plus_patch16_dinov3.lvd1689m",
}


class Backbone(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        num_classes=0,
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        original_backbone_name = backbone_name
        backbone_name = _DINOV3_HF_TO_TIMM.get(backbone_name, backbone_name)
        self.backbone_name = backbone_name

        if "/" in backbone_name:
            print(f"Loading model from HuggingFace Hub: {backbone_name}")
            # Imported lazily: transformers pulls sklearn/pandas/scipy, ~40s cold on
            # networked filesystems, and the timm path below never needs it.
            import os

            import dotenv
            from transformers import AutoModel

            dotenv.load_dotenv("./.env")
            token = os.getenv("HF_TOKEN")
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(backbone_name, token=token),
                img_size,
            )

        else:
            if original_backbone_name != backbone_name:
                print(
                    f"Loading backbone from timm: {backbone_name} "
                    f"(alias for {original_backbone_name})"
                )
            else:
                print(f"Loading backbone from timm: {backbone_name}")
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                dynamic_img_size=True,
            )
        if num_classes == 0:
            self._remove_classifier_params(self.backbone)

        # Normalisation handled in transforms.py

    @staticmethod
    def _remove_classifier_params(backbone: nn.Module) -> None:
        """Drop classifier-only parameters from feature-extractor backbones."""
        for attr in ("head", "fc", "classifier"):
            if isinstance(getattr(backbone, attr, None), nn.Module):
                setattr(backbone, attr, nn.Identity())
        if isinstance(getattr(backbone, "fc_norm", None), nn.Module):
            backbone.fc_norm = nn.Identity()

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        class _GammaScale(nn.Module):
            def __init__(self, gamma: nn.Parameter):
                super().__init__()
                self.gamma = gamma

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.gamma * x

        def _dinov3_pos_embed(x: torch.Tensor) -> torch.Tensor:
            """Compatibility shim for the EoMT timm-style interface.

            DINOv3 uses rotary position embeddings instead of timm's absolute
            pos_embed, but the surrounding code gates prefix-token handling on
            the *presence* of a `_pos_embed` attribute.
            """
            return x

        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer
        backbone._pos_embed = _dinov3_pos_embed
        # Align RoPE naming with timm EVA-02 (uses `rope`) for consistent handling.
        if hasattr(backbone, "rope_embeddings"):
            backbone.rope = backbone.rope_embeddings
        if hasattr(backbone.patch_embed, "cls_token") and not hasattr(
            backbone, "cls_token"
        ):
            backbone.cls_token = backbone.patch_embed.cls_token
        if hasattr(backbone.patch_embed, "register_tokens") and not hasattr(
            backbone, "reg_token"
        ):
            backbone.reg_token = backbone.patch_embed.register_tokens
        # Normalize block attribute names to match timm blocks.
        for blk in backbone.blocks:
            if hasattr(blk, "attention") and not hasattr(blk, "attn"):
                blk.attn = blk.attention
            if hasattr(blk, "attn") and not hasattr(blk.attn, "num_prefix_tokens"):
                blk.attn.num_prefix_tokens = backbone.num_prefix_tokens
            if hasattr(blk, "layer_scale1") and not hasattr(blk, "ls1"):
                blk.ls1 = blk.layer_scale1
            if hasattr(blk, "layer_scale2") and not hasattr(blk, "ls2"):
                blk.ls2 = blk.layer_scale2
            if hasattr(blk, "drop_path"):
                if not hasattr(blk, "drop_path1"):
                    blk.drop_path1 = blk.drop_path
                if not hasattr(blk, "drop_path2"):
                    blk.drop_path2 = blk.drop_path
            if not hasattr(blk, "ls1"):
                if hasattr(blk, "gamma_1"):
                    blk.ls1 = _GammaScale(blk.gamma_1)
                else:
                    blk.ls1 = nn.Identity()
            if not hasattr(blk, "ls2"):
                if hasattr(blk, "gamma_2"):
                    blk.ls2 = _GammaScale(blk.gamma_2)
                else:
                    blk.ls2 = nn.Identity()
            if not hasattr(blk, "drop_path1"):
                blk.drop_path1 = nn.Identity()
            if not hasattr(blk, "drop_path2"):
                blk.drop_path2 = nn.Identity()
            # Normalize attention submodule attrs to reduce hasattr/getattr in EoMT.
            attn = getattr(blk, "attn", None)
            if attn is not None:
                if not hasattr(attn, "qkv"):
                    attn.qkv = None
                if not hasattr(attn, "q_norm"):
                    attn.q_norm = nn.Identity()
                if not hasattr(attn, "k_norm"):
                    attn.k_norm = nn.Identity()
                if not hasattr(attn, "num_prefix_tokens"):
                    attn.num_prefix_tokens = 0
                if not hasattr(attn, "norm"):
                    attn.norm = nn.Identity()
                if hasattr(attn, "o_proj") and not hasattr(attn, "proj"):
                    attn.proj = attn.o_proj
                if not hasattr(attn, "proj_drop"):
                    attn.proj_drop = nn.Identity()
                if not hasattr(attn, "attn_drop"):
                    drop_p = getattr(attn, "dropout", 0.0)
                    attn.attn_drop = nn.Dropout(drop_p)

        del (
            backbone.patch_embed.mask_token,
            backbone.embeddings,
            backbone.layer,
        )

        return backbone


# Checkpoints

# DINOv2 - "vit_large_patch14_reg4_dinov2"
# DINOv3 - "facebook/dinov3-vits16-pretrain-lvd1689m"
# EVA 02 - "eva02_small_patch14_336.mim_in22k_ft_in1k"
