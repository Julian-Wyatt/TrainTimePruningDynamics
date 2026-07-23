# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Adapted, with substantial modifications, from the EoMT reference implementation:
# https://github.com/tue-mps/eomt
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import math
from typing import Optional

import torch
import torch.nn as nn

from optim.param_groups import build_backbone_head_param_groups

from .attn import build_attn_mask, disable_attn_mask, run_attn
from .block import run_block
from .ScaleBlock import ScaleBlock


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
        num_upscale: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        if hasattr(self.encoder.backbone, "head") and not isinstance(
            self.encoder.backbone.head, nn.Identity
        ):
            del self.encoder.backbone.head

        bb = self.encoder.backbone
        self._rope_attr = "rope" if hasattr(bb, "rope") else (
            "rope_embeddings" if hasattr(bb, "rope_embeddings") else None)
        self._has_rope = self._rope_attr is not None
        self._has_pos_embed = hasattr(bb, "_pos_embed")
        _blocks = list(bb.blocks)
        self._num_backbone_blocks: int = len(_blocks)
        self._attn_attr = "attn" if (_blocks and hasattr(
            _blocks[0], "attn")) else "attention"

        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

        self.class_head = nn.Linear(
            self.encoder.backbone.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim,
                      self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim,
                      self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim,
                      self.encoder.backbone.embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        # Auto: one 2x ScaleBlock per patch-size octave above 4px, upscaling to 1/4
        # of the input before the final bilinear. Override to trade resolution for speed.
        if num_upscale is None or int(num_upscale) <= 0:
            num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        num_upscale = max(1, int(num_upscale))

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim)
              for _ in range(num_upscale)],
        )
        self._current_grid_size = tuple(
            self.encoder.backbone.patch_embed.grid_size)

    @classmethod
    def from_cfg(cls, encoder, cfg) -> "EoMT":
        return cls(
            encoder=encoder,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_q=cfg.MODEL.NUM_QUERIES,
            num_blocks=cfg.MODEL.NUM_QUERY_BLOCKS,
            masked_attn_enabled=cfg.MODEL.MASKED_ATTN,
            num_upscale=cfg.MODEL.get("EOMT_NUM_UPSCALE", -1),
        )

    @staticmethod
    def resolve_patch_conv(patch_embed: nn.Module) -> nn.Conv2d:
        """Extract the underlying Conv2d from any backbone's patch embedding module.

        - timm (ViT, EVA-02, DINOv2): patch_embed.proj
        - HuggingFace DINOv3:          patch_embed.patch_embeddings
        - Raw Conv2d:                  returned as-is
        """
        if isinstance(patch_embed, nn.Conv2d):
            return patch_embed
        if hasattr(patch_embed, "proj") and isinstance(patch_embed.proj, nn.Conv2d):
            return patch_embed.proj
        if hasattr(patch_embed, "patch_embeddings") and isinstance(patch_embed.patch_embeddings, nn.Conv2d):
            return patch_embed.patch_embeddings
        raise ValueError(
            f"Cannot resolve Conv2d from patch_embed of type {type(patch_embed)}. "
            "Expected .proj or .patch_embeddings attribute."
        )

    def get_param_groups(
        self,
        embedding_lr: float,
        scalar_lr: float,
        weight_decay: float,
        adam_betas: tuple,
        llrd: float = 1.0,
        llrd_full_lr: bool = False,
        lr_mult: float = 1.0,
        original_eomt_lr_mult_compat: bool = False,
    ) -> list[dict]:
        """Return AdamW param groups tagged with ``group_kind`` (backbone/head).

        Backbone groups carry LLRD; the head group trains at ``scalar_lr``.
        """
        return build_backbone_head_param_groups(
            self,
            self.encoder.backbone,
            embedding_lr=embedding_lr,
            scalar_lr=scalar_lr,
            weight_decay=weight_decay,
            adam_betas=adam_betas,
            llrd=llrd,
            llrd_full_lr=llrd_full_lr,
            lr_mult=lr_mult,
            original_eomt_lr_mult_compat=original_eomt_lr_mult_compat,
            extra_backbone_named_params=self._extra_llrd_backbone_named_parameters(),
            head_group_kind="head",
        )

    def _extra_llrd_backbone_named_parameters(self):
        """Extra pretrained/stem-like params to optimize with backbone LLRD."""
        return ()

    def update_attn_mask_probs(self, step: int, cfg) -> dict:
        """Update attention mask probabilities for the current training step.

        Anneals per-block mask probabilities from 1 (hard) → 0 (soft) based on step ranges.

        Args:
            step: Current global training step.
            cfg: Config object containing ATTN_MASK_ANNEALING_* keys.

        Returns:
            Dict of logged probs for metrics, or empty dict if annealing disabled.
        """
        if not cfg.MODEL.get("ATTN_MASK_ANNEALING_ENABLED", False):
            return {}

        raw_start = list(cfg.MODEL.get("ATTN_MASK_ANNEALING_START_STEPS", []))
        raw_end = list(cfg.MODEL.get("ATTN_MASK_ANNEALING_END_STEPS", []))
        poly_power = float(cfg.MODEL.get("ATTN_MASK_POLY_POWER", 0.9))

        log_probs = {}
        for i in range(self.num_blocks):
            if i >= len(raw_start) or i >= len(raw_end):
                break
            s, e = raw_start[i], raw_end[i]
            if step < s:
                prob = 1.0
            elif step >= e:
                prob = 0.0
            else:
                progress = (step - s) / max(e - s, 1)
                prob = (1.0 - progress) ** poly_power
            self.attn_mask_probs[i] = prob
            rel_idx = i - (self.num_blocks - 1)
            log_probs[f"train/attn_mask_prob_{rel_idx}"] = prob

        return log_probs

    def _predict(self, x: torch.Tensor, predict_class: bool = True):
        q = x[:, : self.num_q, :]

        if predict_class:
            class_logits = self._class_head_forward(q)
        else:
            class_logits = q.new_empty(q.shape[0], q.shape[1], 0)

        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens:, :]
        grid_h, grid_w = self._get_current_grid_size()
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, grid_h, grid_w
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self._mask_head_forward(q), self._upscale_forward(x)
        )

        return mask_logits, class_logits

    def _get_current_grid_size(self) -> tuple[int, int]:
        """Return the patch-grid size corresponding to the last input image."""
        return getattr(self, "_current_grid_size", self.encoder.backbone.patch_embed.grid_size)

    def _compile_prediction_heads(
        self,
        dynamic: bool | None = None,
        fullgraph: bool = False,
        compile_mode: str | None = None,
    ) -> int:
        """Compile prediction-head call sites without replacing the modules.

        Keeping the original modules in-place preserves parameter names and
        diagnostics, while still letting Inductor optimize the stable dense
        class/mask/decoder work.
        """
        class_head = self.class_head
        mask_head = self.mask_head
        upscale = self.upscale

        def _class_head(q: torch.Tensor) -> torch.Tensor:
            return class_head(q)

        def _mask_head(q: torch.Tensor) -> torch.Tensor:
            return mask_head(q)

        def _upscale(x: torch.Tensor) -> torch.Tensor:
            return upscale(x)

        self._compiled_class_head = torch.compile(
            _class_head, fullgraph=fullgraph, dynamic=dynamic, mode=compile_mode)
        self._compiled_mask_head = torch.compile(
            _mask_head, fullgraph=fullgraph, dynamic=dynamic, mode=compile_mode)
        self._compiled_upscale = torch.compile(
            _upscale, fullgraph=fullgraph, dynamic=dynamic, mode=compile_mode)
        return 3

    def _compile_patch_embed(
        self,
        dynamic: bool | None = None,
        fullgraph: bool = False,
        compile_mode: str | None = None,
    ) -> int:
        patch_embed = self.encoder.backbone.patch_embed

        def _patch_embed(x: torch.Tensor) -> torch.Tensor:
            return patch_embed(x)

        self._compiled_patch_embed = torch.compile(
            _patch_embed, fullgraph=fullgraph, dynamic=dynamic, mode=compile_mode)
        return 1

    def compile_stable_submodules(
        self,
        dynamic: bool | None = None,
        fullgraph: bool = False,
        compile_mode: str | None = None,
    ) -> int:
        """Compile stable tensor-heavy regions while leaving forward orchestration eager."""
        return (
            self._compile_patch_embed(
                dynamic=dynamic,
                fullgraph=fullgraph,
                compile_mode=compile_mode,
            )
            + self._compile_prediction_heads(
                dynamic=dynamic,
                fullgraph=fullgraph,
                compile_mode=compile_mode,
            )
        )

    def _patch_embed_forward(self, x: torch.Tensor) -> torch.Tensor:
        compiled = getattr(self, "_compiled_patch_embed", None)
        if compiled is not None:
            return compiled(x)
        return self.encoder.backbone.patch_embed(x)

    def _class_head_forward(self, q: torch.Tensor) -> torch.Tensor:
        compiled = getattr(self, "_compiled_class_head", None)
        return compiled(q) if compiled is not None else self.class_head(q)

    def _mask_head_forward(self, q: torch.Tensor) -> torch.Tensor:
        compiled = getattr(self, "_compiled_mask_head", None)
        return compiled(q) if compiled is not None else self.mask_head(q)

    def _upscale_forward(self, x: torch.Tensor) -> torch.Tensor:
        compiled = getattr(self, "_compiled_upscale", None)
        return compiled(x) if compiled is not None else self.upscale(x)

    def _disable_attn_mask(self, attn_mask, prob):
        sp_start = self.num_q + self.encoder.backbone.num_prefix_tokens
        return disable_attn_mask(attn_mask, prob, self.num_q, sp_start)

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask,  # bool tensor | dense additive mask | None
        rope: Optional[torch.Tensor],
    ):
        return run_attn(
            module,
            x,
            mask,
            rope,
            self.training,
        )

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int):
        sp_start = self.num_q + self.encoder.backbone.num_prefix_tokens
        grid_h, grid_w = self._get_current_grid_size()
        # The query→spatial mask is defined over the full patch grid; while a
        # routed (shrunk) sequence is in flight it cannot be built, so skip it.
        if x.shape[1] != sp_start + grid_h * grid_w:
            return None
        block_idx = i - self._num_backbone_blocks + self.num_blocks
        return build_attn_mask(
            x, mask_logits, self.num_q, sp_start,
            self._get_current_grid_size(), block_idx, self.attn_mask_probs,
        )

    def _block(
        self,
        blk: nn.Module,
        x: torch.Tensor,
        attn_mask,
        rope,
        policy=None,
    ):
        return run_block(
            blk,
            x,
            attn_mask,
            rope,
            self._attn_attr,
            self.training,
            policy,
        )

    def _setup_rope(self, x: torch.Tensor):
        """Apply patch embedding + positional embedding, compute RoPE. Returns (x, rope)."""
        rope = None

        ph, pw = self.encoder.backbone.patch_embed.patch_size
        H, W = x.shape[2], x.shape[3]
        self._current_grid_size = (H // ph, W // pw)

        x = self._patch_embed_forward(x)
        if self._has_pos_embed:
            pos_out = self.encoder.backbone._pos_embed(x)
            if isinstance(pos_out, tuple):
                x, rope = pos_out
            else:
                x = pos_out
        return x, rope

    def _route_rope(
        self,
        rope,
        ids_keep: torch.Tensor | None,
        prefix_tokens: int,
    ):
        """Gather spatial RoPE entries to match a routed token sequence."""
        if rope is None or ids_keep is None:
            return rope

        num_spatial = ids_keep.shape[1] - int(prefix_tokens)
        if num_spatial <= 0:
            if isinstance(rope, tuple):
                return tuple(r[..., :0, :] for r in rope)
            return rope[..., :0, :]

        spatial_ids = ids_keep[:, -num_spatial:] - prefix_tokens

        def _gather(r: torch.Tensor) -> torch.Tensor:
            if r.dim() == 2:
                r = r.unsqueeze(0).expand(ids_keep.shape[0], -1, -1)
            gather_ids = spatial_ids.unsqueeze(-1).expand(-1, -1, r.shape[-1])
            return torch.gather(r, 1, gather_ids)

        if isinstance(rope, tuple):
            return tuple(_gather(r) for r in rope)
        return _gather(rope)

    @staticmethod
    def _route_rope_cached(route_rope_fn, rope, ids_keep, route_prefix, cache):
        """``_route_rope`` memoised across blocks within one forward pass.

        While a route is open and the kept-token set is unchanged, every block
        gathers the *same* spatial RoPE entries, so re-gathering per block is
        pure redundant work — one of the costs RoPE adds to physical token
        routing that an absolute-position backbone (e.g. DINOv2) does not pay.
        ``cache`` is a plain dict reused across blocks; the gather is recomputed
        only when ``ids_keep`` (by object identity) or ``route_prefix`` changes —
        i.e. once per prune stage, not once per block.
        """
        if cache.get("ids") is ids_keep and cache.get("prefix") == route_prefix:
            return cache["val"]
        val = route_rope_fn(rope, ids_keep, route_prefix)
        cache["ids"], cache["prefix"], cache["val"] = ids_keep, route_prefix, val
        return val

    def _collect_prediction(
        self,
        x,
        mask_logits_per_layer,
        class_logits_per_layer,
        predict_class: bool = True,
    ):
        """Run prediction head and append results to the per-layer lists. Returns mask_logits."""
        mask_logits, class_logits = self._predict(
            self.encoder.backbone.norm(x),
            predict_class=predict_class,
        )
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        return mask_logits

    def forward(
        self,
        x: torch.Tensor,
        throughput_mode: bool = False,
        predict_class: bool = True,
    ):
        # Normalisation is handled in dataset transforms
        x, rope = self._setup_rope(x)
        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        q_start = self._num_backbone_blocks - self.num_blocks
        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == q_start:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                not throughput_mode
                and self.masked_attn_enabled
                and i >= q_start
            ):
                mask_logits = self._collect_prediction(
                    x, mask_logits_per_layer, class_logits_per_layer)
                attn_mask = self._attn_mask(x, mask_logits, i)

            x = self._block(block, x, attn_mask, rope)

        self._collect_prediction(
            x,
            mask_logits_per_layer,
            class_logits_per_layer,
            predict_class=predict_class,
        )

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
