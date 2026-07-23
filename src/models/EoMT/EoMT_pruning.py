"""Token-pruned EoMT: the study's segmentation model."""
from __future__ import annotations

import random
import warnings

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.EoMT.cropr_scorer import CrossAttention
from models.pruning import (
    keep_score,
    keep_tokens_from_keep_rate,
    randomize_ids,
    ratio_loss,
    round_pruned_tokens_to_total_multiple,
    straight_through_gumbel_keep,
)

from .EoMT import EoMT
from .router import Router


class TokenSelector(nn.Module):
    """The shared MLP keep/drop scorer for DynamicViT, Gumbel, and Reg4Pru."""

    def __init__(self, embed_dim: int):
        super().__init__()
        hidden = max(embed_dim // 4, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PruneForward:
    """Mutable state threaded through ``EoMTPruning.forward``'s block loop.

    Holds the per-forward geometry/flags (set once) plus the routing, selection
    and accounting state that compounds across blocks. Keeping it on one object
    lets the loop body split into small, named steps that share state.
    """

    def __init__(self, *, batch, q_start, prefix, reinject_block, stage_of,
                 active_keep_rates, route_start, route_end, rope, throughput_mode,
                 capture, track_abs_ids, soft, gumbel_route, is_random, random_route):
        self.batch = batch
        self.q_start = q_start
        self.prefix = prefix
        self.reinject_block = reinject_block
        self.stage_of = stage_of
        self.active_keep_rates = active_keep_rates
        self.route_start = route_start
        self.route_end = route_end
        self.rope = rope
        self.throughput_mode = throughput_mode
        self.capture = capture
        self.track_abs_ids = track_abs_ids
        self.soft = soft
        self.gumbel_route = gumbel_route
        self.is_random = is_random
        self.random_route = random_route

        self.mask_logits: list[torch.Tensor] = []
        self.class_logits: list[torch.Tensor] = []

        # Soft-mask state (full sequence kept, dropped tokens attention-masked).
        self.attn_mask = None
        self.mask_policy: torch.Tensor | None = None
        self.prev_keep: torch.Tensor | None = None

        # Physical-routing state. Complement chunks accumulate in lists and are
        # concatenated once at consumption, avoiding per-stage growing cats.
        self.dropped_x: list[torch.Tensor] | None = None
        self.ids_drop: list[torch.Tensor] | None = None
        self.ids_keep: torch.Tensor | None = None
        self.full_len: int | None = None
        self.route_prefix: int | None = None
        self.rope_cache: dict = {}

        self.cur_prefix = prefix
        self.cur_pool: int | None = None      # selectable spatial tokens entering the next stage
        self.cumulative = 1.0                 # product of per-stage relative keep rates
        # Absolute grid index of each currently-active spatial token, for mapping
        # reduced per-stage keep masks back to the full grid (figure / aux scatter).
        self.abs_spatial_ids: torch.Tensor | None = None
        self.grid_spatial = 0                 # full spatial-token count (set at first stage)

        self.budget_losses: list[torch.Tensor] = []
        self.aux_preds: list[torch.Tensor] = []
        self.aux_spatial_ids: list[torch.Tensor] = []
        self.keep_means: list[torch.Tensor] = []
        self.processed: list[int] = []        # tokens actually run through each block
        self.dense: list[int] = []            # unpruned token count per block
        self.stage_keeps: list[int] = []      # kept spatial count after each stage
        self.captured_maps: list[torch.Tensor] = []

    def route_is_open(self) -> bool:
        return (self.dropped_x is not None and self.ids_drop is not None
                and self.ids_keep is not None and self.full_len is not None)

    def cat_complement(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cat(self.dropped_x, dim=1), torch.cat(self.ids_drop, dim=1)

    def close_route(self) -> None:
        self.dropped_x = self.ids_drop = self.ids_keep = None
        self.route_prefix = self.full_len = None
        self.abs_spatial_ids = None


class EoMTPruning(EoMT):
    """EoMT with the four pruning methods compared in the paper.

    The methods diverge at four deliberately small points:

    1. :meth:`_score_stage` selects the shared MLP scorer or CROPR's learned-query
       scorer.
    2. :meth:`_select_stage` chooses full-sequence masking or physical top-k
       routing.
    3. :meth:`_step_random_route` adds Reg4Pru's independent random window.
    4. :meth:`_accumulate_budget_loss` applies the MLP arms' ratio loss; CROPR
       instead has its direct-BCE loss assembled in ``EoMTTrainer``.

    Config-to-method map:

    - ``soft_mask``: **DynamicViT** — MLP + straight-through Gumbel policy masks
      attention, but all tokens remain in the sequence during training.
    - ``gumbel_route``: **Gumbel Pruning** — the same MLP policy is ranked and
      physically top-k routed through later blocks during training and evaluation.
    - ``random_route``: **Reg4Pru** — DynamicViT's full-sequence MLP arm plus a
      separate 50%-keep random route over one sampled block window.
    - ``hard_route`` with ``EOMT_PRUNE_AUX_SCORER=true``: **CROPR** — detached
      features feed learned-query scores, which are physically top-k routed and
      trained by direct, class-balanced BCE on pooled foreground targets.

    ``EOMT_PRUNE_EVAL_MODE=random`` is the paper's Random eval-only ablation: it
    reuses a Reg4Pru checkpoint and replaces the learned ranking with random ids.

    ``forward`` returns ``(mask_logits_per_layer, class_logits_per_layer)`` like
    ``EoMT``. Pruning diagnostics are exposed via ``get_pruning_loss`` /
    ``get_token_stats``.
    """

    VALID_TRAIN_MODES = {
        "dense", "soft_mask", "gumbel_route", "random_route", "hard_route",
    }
    # "random" eval routes a random keep set (same budget) instead of the learned
    # top-k, isolating the inference value of the selector from the representation
    # a given train mode produced.
    VALID_EVAL_MODES = {"dense", "hard", "random"}
    VALID_REINJECT = {"q_start", "mask_pred", "penultimate"}

    def __init__(self, encoder, cfg):
        self.cfg = cfg
        super().__init__(
            encoder=encoder,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_q=cfg.MODEL.NUM_QUERIES,
            num_blocks=cfg.MODEL.NUM_QUERY_BLOCKS,
            num_upscale=cfg.MODEL.EOMT_NUM_UPSCALE,
            masked_attn_enabled=cfg.MODEL.MASKED_ATTN,
        )
        if str(cfg.MODEL.EOMT_PRUNE_TRAIN_MODE).lower() not in self.VALID_TRAIN_MODES:
            raise ValueError(
                f"Unknown EOMT_PRUNE_TRAIN_MODE={cfg.MODEL.EOMT_PRUNE_TRAIN_MODE!r}")
        if str(cfg.MODEL.EOMT_PRUNE_EVAL_MODE).lower() not in self.VALID_EVAL_MODES:
            raise ValueError(
                f"Unknown EOMT_PRUNE_EVAL_MODE={cfg.MODEL.EOMT_PRUNE_EVAL_MODE!r}")
        if str(cfg.MODEL.EOMT_PRUNE_REINJECT).lower() not in self.VALID_REINJECT:
            raise ValueError(
                f"Unknown EOMT_PRUNE_REINJECT={cfg.MODEL.EOMT_PRUNE_REINJECT!r}")

        # EOMT_PRUNE_LAYERS is authoritative: a stage removes tokens wherever it
        # sits (including query blocks). EOMT_PRUNE_REINJECT independently controls
        # only where routed tokens are permanently scattered back. Out-of-range
        # indices are dropped with a warning.
        n_blocks = self._num_backbone_blocks
        requested = [int(i) for i in cfg.MODEL.EOMT_PRUNE_LAYERS]
        keep_rates = list(cfg.MODEL.EOMT_KEEP_RATES)
        if cfg.MODEL.USE_CROPR:
            initial_spatial_tokens = max(
                1,
                int(cfg.DATASET.IMG_SIZE[0]) // max(1, int(cfg.MODEL.PATCH_SIZE)),
            ) ** 2
            requested, keep_rates = self._cropr_schedule_from_cfg(
                encoder, cfg, initial_spatial_tokens)
        layers = [i for i in requested if 0 <= i < n_blocks]
        dropped = [i for i in requested if not (0 <= i < n_blocks)]
        if dropped:
            warnings.warn(
                f"EOMT_PRUNE_LAYERS contained out-of-range indices {dropped} "
                f"(valid range 0..{n_blocks - 1}); using prune_layers={layers}.",
                stacklevel=2,
            )
        self.prune_layers = layers
        if len(keep_rates) < len(layers):
            keep_rates = list(keep_rates) + [keep_rates[-1] if keep_rates else 1.0] * (len(layers) - len(keep_rates))
        self.target_keep_rates = [float(k) for k in keep_rates[: len(layers)]]
        self.keep_rates = list(self.target_keep_rates)
        self._prune_curriculum_alpha = 1.0

        # A stage landing on the reinjection block gathers the kept set then
        # immediately scatters it back, so no block runs the reduced sequence —
        # a routing no-op. Warn so it can be moved or the reinject point deepened.
        q_start = self._num_backbone_blocks - self.num_blocks
        if self._reinject_block(self._num_backbone_blocks, q_start) in self.prune_layers:
            warnings.warn(
                f"EOMT_PRUNE_LAYERS contains the reinjection block for "
                f"EOMT_PRUNE_REINJECT={cfg.MODEL.EOMT_PRUNE_REINJECT!r}: that stage gathers the "
                f"kept tokens and immediately scatters them back (a routing no-op). "
                f"Move the stage earlier or use a deeper reinjection point.",
                stacklevel=2,
            )

        embed_dim = self.encoder.backbone.embed_dim
        self.aux_scorer = bool(cfg.MODEL.EOMT_PRUNE_AUX_SCORER)
        img0 = cfg.DATASET.IMG_SIZE[0]
        grid = max(1, int(img0) // max(1, int(cfg.MODEL.PATCH_SIZE)))
        self.aux_num_queries = grid * grid
        if str(cfg.MODEL.EOMT_PRUNE_TRAIN_MODE).lower() == "random_route":
            if not cfg.MODEL.EOMT_PRUNE_RANDOM_WINDOW_BOUNDS:
                if not (
                    0 <= cfg.MODEL.EOMT_PRUNE_ROUTE_START_AFTER_BLOCK
                    <= cfg.MODEL.EOMT_PRUNE_ROUTE_END_AFTER_BLOCK < n_blocks
                ):
                    raise ValueError(
                        "random_route requires 0 <= EOMT_PRUNE_ROUTE_START_AFTER_BLOCK "
                        "<= EOMT_PRUNE_ROUTE_END_AFTER_BLOCK < num_blocks; got "
                        f"start={cfg.MODEL.EOMT_PRUNE_ROUTE_START_AFTER_BLOCK} "
                        f"end={cfg.MODEL.EOMT_PRUNE_ROUTE_END_AFTER_BLOCK} "
                        f"num_blocks={n_blocks}"
                    )
            elif n_blocks < 4:
                raise ValueError(
                    "random_route with EOMT_PRUNE_RANDOM_WINDOW_BOUNDS=True requires "
                    f"n_blocks >= 4 (l ~ U{{2..L/2}}, n ~ U{{l+1..L-2}}); got {n_blocks}"
                )
            if not (0.0 <= cfg.MODEL.EOMT_PRUNE_ROUTING_KEEP_RATE <= 1.0):
                raise ValueError("random_route requires 0 <= EOMT_PRUNE_ROUTING_KEEP_RATE <= 1")

        # MLP selectors are skipped when the aux scorer is active; their unused
        # params would trip strict DDP.
        self.selectors = nn.ModuleList(
            [] if self.aux_scorer else [TokenSelector(embed_dim) for _ in self.prune_layers]
        )
        if self.aux_scorer:
            self.aux_cross_attn = nn.ModuleList([
                CrossAttention(
                    embed_dim=embed_dim,
                    num_queries=self.aux_num_queries,
                )
                for _ in self.prune_layers
            ])
        self.router = Router()
        self._last_aux_preds: list[torch.Tensor] = []
        self._last_aux_spatial_ids: list[torch.Tensor] = []
        self._last_pruning_loss: torch.Tensor | None = None
        self._last_token_stats: dict[str, float] = {}
        self._keep_mean_tensors: list[torch.Tensor] = []
        # Figure capture is off by default, keeping the training hot path free
        # of host-side map construction.
        self._capture_selection = False
        self._selection_maps: list[dict[str, torch.Tensor]] = []

    @staticmethod
    def _cropr_schedule_from_cfg(encoder, cfg, initial_spatial_tokens: int) -> tuple[list[int], list[float]]:
        """Translate Question B/C CROPR schedule knobs into EoMT prune stages."""
        n_blocks = len(encoder.backbone.blocks)
        q_start = n_blocks - int(cfg.MODEL.NUM_QUERY_BLOCKS)
        routing_start = max(0, min(int(cfg.MODEL.ROUTING_START), n_blocks - 1))
        routing_end = int(cfg.MODEL.ROUTING_END)
        # The prune window ends before the EoMT query blocks; reinjection stays
        # controlled separately by EOMT_PRUNE_REINJECT.
        if routing_end < 0:
            routing_end = q_start
        routing_end = max(routing_start + 1, min(routing_end, n_blocks))
        schedule = str(cfg.MODEL.CROPR_SCHEDULE).lower()
        round_multiple = int(cfg.MODEL.PRUNE_ROUND_MULTIPLE)
        keep_fraction = cfg.MODEL.CROPR_KEEP_FRACTION
        keep_fraction = None if keep_fraction is None else float(keep_fraction)
        prune_blocks = sorted({
            int(b) for b in (cfg.MODEL.CROPR_PRUNE_BLOCKS or [])
            if routing_start <= int(b) < routing_end
        })
        static = schedule in {"static_early", "early_reinject"} or bool(
            cfg.MODEL.CROPR_STATIC_PRUNING)

        if static:
            layers = [routing_start]
        elif prune_blocks:
            layers = prune_blocks
        else:
            # Reference CROPR prunes every block up to the penultimate
            # (self.blocks[:-2] == range(0, n_blocks - 2)); the penultimate
            # reinjection point then runs the last two blocks on the full grid.
            layers = list(range(0, n_blocks - 2))

        if keep_fraction is not None and (static or schedule == "progressive_fraction"):
            return layers, [keep_fraction] * len(layers)

        prune_rate = int(cfg.MODEL.CROPR_PRUNING_RATE)
        if cfg.MODEL.CROPR_DERIVE_PRUNING_RATE:
            prune_rate = int(round(
                initial_spatial_tokens
                * float(cfg.MODEL.CROPR_REFERENCE_PRUNE_TOKENS)
                / max(1.0, float(cfg.MODEL.CROPR_REFERENCE_SPATIAL_TOKENS))
            ))
        if static:
            prune_rate = int(round(prune_rate * float(cfg.MODEL.CROPR_STATIC_PRUNE_MULT)))

        current = max(1, int(initial_spatial_tokens))
        prefix = int(getattr(encoder.backbone, "num_prefix_tokens", 0))
        num_q = int(cfg.MODEL.NUM_QUERIES)
        keep_rates = []
        for layer in layers:
            offset = prefix + (num_q if layer >= q_start else 0)
            pruned = round_pruned_tokens_to_total_multiple(
                prune_rate, current, total_tokens_offset=offset,
                multiple=round_multiple, min_keep=1)
            kept = max(1, current - pruned)
            keep_rates.append(kept / current)
            current = kept
        return layers, keep_rates

    # ── Diagnostics ──────────────────────────────────────────────────────────
    def get_pruning_loss(self) -> torch.Tensor | None:
        return self._last_pruning_loss

    def get_aux_spatial_ids(self) -> list[torch.Tensor]:
        """Per-stage absolute grid indices for BCE supervision of kept tokens."""
        return self._last_aux_spatial_ids

    def get_aux_predictions(self) -> list[torch.Tensor]:
        """CROPR auxiliary BCE logits ``[B, active_spatial, 1]`` per stage.
        Train-only (empty at eval, so inference pays no aux cost)."""
        return self._last_aux_preds

    def get_token_stats(self) -> dict[str, float]:
        stats = dict(self._last_token_stats)
        for j, t in enumerate(self._keep_mean_tensors):
            stats[f"token_stats/selector_keep_prob_{j}"] = float(t)
            if j < len(self.target_keep_rates):
                stats[f"token_stats/target_keep_rate_{j}"] = float(self.target_keep_rates[j])
            if j < len(self.keep_rates):
                stats[f"token_stats/current_keep_rate_{j}"] = float(self.keep_rates[j])
        return stats

    # ── Selection-map capture (paper figure) ─────────────────────────────────
    def set_selection_capture(self, enabled: bool) -> None:
        """Toggle per-stage keep-mask capture mapped to absolute grid positions."""
        self._capture_selection = bool(enabled)

    def get_selection_maps(self) -> list[dict[str, torch.Tensor]]:
        """Per-stage ``keep`` grids from the latest captured forward.

        Maps are ``[B, gh, gw]`` over the full patch grid and are consumed by
        the standalone qualitative-figure script. Empty when capture was off.
        """
        return self._selection_maps

    def set_pruning_epoch(self, epoch: int) -> None:
        """Ramp the training prune budget from dense to target over N epochs.

        ``EOMT_KEEP_RATES`` are per-stage relative rates, so the curriculum ramps
        each cumulative keep fraction linearly, then derives the active per-stage
        rates from those cumulative targets.
        """
        if self.cfg.MODEL.EOMT_PRUNE_RAMP_EPOCHS <= 0:
            self.keep_rates = list(self.target_keep_rates)
            self._prune_curriculum_alpha = 1.0
            return

        if self.cfg.MODEL.EOMT_PRUNE_RAMP_EPOCHS == 1:
            alpha = 1.0
        else:
            alpha = min(
                max(float(epoch), 0.0),
                self.cfg.MODEL.EOMT_PRUNE_RAMP_EPOCHS - 1,
            )
            alpha = alpha / float(self.cfg.MODEL.EOMT_PRUNE_RAMP_EPOCHS - 1)

        active_rates = []
        target_cumulative = 1.0
        prev_active_cumulative = 1.0
        for target_rate in self.target_keep_rates:
            target_cumulative *= target_rate
            active_cumulative = 1.0 - alpha * (1.0 - target_cumulative)
            active_rates.append(active_cumulative / max(prev_active_cumulative, 1e-12))
            prev_active_cumulative = active_cumulative

        self.keep_rates = active_rates
        self._prune_curriculum_alpha = alpha

    def _mode(self) -> str:
        if self.training:
            return str(self.cfg.MODEL.EOMT_PRUNE_TRAIN_MODE).lower()
        return str(self.cfg.MODEL.EOMT_PRUNE_EVAL_MODE).lower()

    def _reinject_block(self, n_blocks: int, q_start: int) -> int:
        """Block index after which an open route is permanently scattered back."""
        if self.cfg.MODEL.EOMT_PRUNE_REINJECT == "penultimate":
            return n_blocks - 2
        if self.cfg.MODEL.EOMT_PRUNE_REINJECT == "mask_pred":
            return n_blocks - 1
        return q_start - 1  # "q_start": close before query injection

    def _select_stage(
        self,
        logits: torch.Tensor | None,
        keep_rate: float,
        pool: int,
        total_tokens_offset: int,
        post_round_subtract: int,
        valid_mask: torch.Tensor | None,
        is_random: bool,
        score: torch.Tensor | None = None,
        soft: bool = False,
        gumbel_route: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
        """Return ``(policy, ids_keep, ids_drop, score, k)`` for one prune stage.

        ``policy`` is the ``[B, spatial]`` straight-through keep (hard forward, soft
        gradient) used only by the train-time budget loss; ``ids_keep``/``ids_drop``
        are the kept/dropped spatial indices; ``score`` is the soft keep score; ``k``
        is the kept count. ``keep_rate`` is the per-stage relative keep fraction
        applied to ``pool`` (currently-selectable tokens), so stages compound.

        ``score`` may be precomputed (CROPR cross-attention) or derived from
        ``logits`` via the MLP selector. Hard/gumbel routing take kept and dropped
        ids from a single argsort (reference CROPR ``prune()`` style).

        ``post_round_subtract`` reserves tokens inserted before the next block
        (the EoMT query token) so the rounded length already accounts for them.
        """
        def rounded_k() -> int:
            if pool <= 0:
                return 0
            k = keep_tokens_from_keep_rate(
                pool,
                keep_rate,
                multiple=max(1, int(self.cfg.MODEL.PRUNE_ROUND_MULTIPLE)),
                total_tokens_offset=total_tokens_offset)
            return max(1, k - max(0, int(post_round_subtract)))

        if soft:
            # Per-token straight-through Gumbel-argmax (no top-k); the realised
            # count is emergent and the budget is enforced by the ratio loss.
            policy, score = straight_through_gumbel_keep(
                logits,
                self.cfg.MODEL.EOMT_PRUNE_GUMBEL_TAU,
                self.training,
                valid_mask=valid_mask,
            )
            k = min(rounded_k(), score.shape[1])  # nominal budget, accounting only
            ids = score.new_empty(score.shape[0], 0, dtype=torch.long)
            return policy, ids, None, score, k
        if gumbel_route:
            # Same Gumbel-keep predictor as soft_mask, but a top-k over the score
            # physically routes the kept set; valid_mask is None (prior stages
            # already reduced the sequence, so survivors are the nested subset).
            policy, score = straight_through_gumbel_keep(
                logits,
                self.cfg.MODEL.EOMT_PRUNE_GUMBEL_TAU,
                self.training,
                valid_mask=valid_mask,
            )
            # int() keeps k concrete: under torch.compile score.shape[1] is a
            # symbolic dim, and a symbolic k propagates into gather/start_route and
            # trips the inductor tiling assertion in get_pw_red_splits.
            k = int(min(rounded_k(), score.shape[1]))
            ranked = score.argsort(dim=1, descending=True)
            return policy, ranked[:, :k], ranked[:, k:], score, k
        if score is None:
            score = keep_score(
                logits, self.cfg.MODEL.EOMT_PRUNE_GUMBEL_TAU, self.training)
        spatial = score.shape[1]
        k = int(min(rounded_k(), spatial))  # see gumbel_route branch for the int() reason
        # Policy feeds only the train-time budget loss. The aux scorer is
        # supervised by direct BCE (not the budget MSE) and the eval path consumes
        # ids alone, so skip the [B, spatial] scatter unless it is needed.
        need_policy = self.training and not self.aux_scorer
        ranked = score.argsort(dim=1, descending=True)
        ids = ranked[:, :k]
        ids_drop = ranked[:, k:]
        if need_policy:
            hard = torch.zeros_like(score).scatter_(1, ids, 1.0)
            policy = hard - score.detach() + score
        else:
            policy = None
        if is_random and ids.shape[1] > 0:
            # Route a (partly) random subset, keeping the straight-through path so
            # the budget loss still calibrates the soft score. Fires at train for
            # random_route and at eval for the "random" eval mode.
            ids = randomize_ids(
                ids, spatial, self.cfg.MODEL.EOMT_PRUNE_RANDOM_RATIO)
            if need_policy:
                hard = torch.zeros_like(score).scatter_(1, ids, 1.0)
                policy = hard - score.detach() + score
            keep_mask = torch.zeros(score.shape[0], spatial, dtype=torch.bool, device=score.device)
            keep_mask.scatter_(1, ids, True)
            ids_drop = (~keep_mask).nonzero(as_tuple=False)[:, 1].view(score.shape[0], spatial - k)
        return policy, ids, ids_drop, score, k

    def _collect_routed_prediction_checkpointed(
        self,
        x: torch.Tensor,
        ids_keep: torch.Tensor,
        dropped_x: torch.Tensor,
        ids_drop: torch.Tensor,
        full_len: int,
        mask_logits_per_layer: list[torch.Tensor],
        class_logits_per_layer: list[torch.Tensor],
    ) -> torch.Tensor:
        """Collect a routed prediction, rebuilding the full grid inside a
        checkpoint so the temporary dense head activations are not saved."""

        def _predict_route(cur_x, keep_ids, drop_x, drop_ids):
            full = self.router.merge_route(cur_x, keep_ids, drop_x, drop_ids, full_len)
            return self._predict(self.encoder.backbone.norm(full))

        if self.training and x.requires_grad:
            mask_logits, class_logits = checkpoint(
                _predict_route, x, ids_keep, dropped_x, ids_drop, use_reentrant=False)
        else:
            mask_logits, class_logits = _predict_route(x, ids_keep, dropped_x, ids_drop)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        return mask_logits

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, throughput_mode: bool = False, predict_class: bool = True):
        mode = self._mode()
        if mode == "dense" or not self.prune_layers:
            self._clear_diagnostics()
            return super().forward(x, throughput_mode=throughput_mode, predict_class=predict_class)

        x, st = self._init_forward(x, mode, throughput_mode)
        for i, block in enumerate(self.encoder.backbone.blocks):
            x = self._inject_queries(st, x, i)
            self._predict_query_block(st, x, i)
            block_policy, pending_ids, ids_drop = self._prune_stage(st, x, i)
            x = self._run_block(st, x, i, block, block_policy)
            x = self._apply_gather(st, x, pending_ids, ids_drop)
            x = self._step_random_route(st, x, i)
            x = self._maybe_reinject(st, x, i)

        if st.route_is_open():
            x = self._reconstruct_route(st, x)
        self._collect_prediction(x, st.mask_logits, st.class_logits, predict_class=predict_class)
        self._finalize(st)
        return st.mask_logits, st.class_logits

    def _init_forward(self, x: torch.Tensor, mode: str, throughput_mode: bool):
        """Map config modes to the four paper paths, then initialise loop state."""
        # DynamicViT remains full-sequence. Reg4Pru retains that same MLP policy
        # while separately routing a random window below. Gumbel and CROPR both
        # physically route selected tokens at each configured prune stage.
        dynamicvit = mode == "soft_mask"
        reg4pru = self.training and mode == "random_route"
        gumbel_pruning = mode == "gumbel_route"
        random_eval = mode == "random"
        soft = (dynamicvit or reg4pru) and not self.aux_scorer
        gumbel_route = gumbel_pruning and not self.aux_scorer

        x, rope = self._setup_rope(x)
        n_blocks = len(self.encoder.backbone.blocks)
        q_start = n_blocks - self.num_blocks

        route_start = self.cfg.MODEL.EOMT_PRUNE_ROUTE_START_AFTER_BLOCK
        route_end = self.cfg.MODEL.EOMT_PRUNE_ROUTE_END_AFTER_BLOCK
        if reg4pru and self.cfg.MODEL.EOMT_PRUNE_RANDOM_WINDOW_BOUNDS:
            # Reg4Pru: sample one independent route per forward, l ~ U{2..L/2}
            # and n ~ U{l+1..L-2}. The checked-in config sets this flag true.
            hi_l = max(2, min(n_blocks // 2, n_blocks - 2))
            route_start = random.randint(2, hi_l)
            route_end = random.randint(route_start + 1, max(route_start + 1, n_blocks - 2))

        st = _PruneForward(
            batch=x.shape[0], q_start=q_start,
            prefix=self.encoder.backbone.num_prefix_tokens,
            reinject_block=self._reinject_block(n_blocks, q_start),
            stage_of={layer: j for j, layer in enumerate(self.prune_layers)},
            active_keep_rates=self.keep_rates if self.training else self.target_keep_rates,
            route_start=route_start, route_end=route_end, rope=rope,
            throughput_mode=throughput_mode,
            capture=self._capture_selection and not throughput_mode,
            track_abs_ids=self.aux_scorer and self.training,
            soft=soft, gumbel_route=gumbel_route, is_random=random_eval,
            random_route=reg4pru,
        )
        return x, st

    def _inject_queries(self, st: _PruneForward, x: torch.Tensor, i: int) -> torch.Tensor:
        """(1) Prepend the learned query tokens at the first query block."""
        if i != st.q_start:
            return x
        q_tokens = self.q.weight[None, :, :].expand(st.batch, -1, -1)
        x = torch.cat((q_tokens, x), dim=1)
        st.cur_prefix += self.num_q
        if st.mask_policy is not None:
            ones = st.mask_policy.new_ones(st.batch, self.num_q)
            st.mask_policy = torch.cat((ones, st.mask_policy), dim=1)
        if st.route_is_open():
            query_ids = torch.arange(self.num_q, device=x.device).unsqueeze(0).expand(st.batch, -1)
            st.ids_keep = torch.cat((query_ids, st.ids_keep + self.num_q), dim=1)
            st.ids_drop[:] = [t + self.num_q for t in st.ids_drop]
            st.full_len = st.full_len + self.num_q
            st.route_prefix = self.num_q + st.prefix
        return x

    def _predict_query_block(self, st: _PruneForward, x: torch.Tensor, i: int) -> None:
        """(2) Mask prediction at each query block, building the attention mask."""
        if st.throughput_mode or not self.masked_attn_enabled or i < st.q_start:
            return
        if st.route_is_open():
            dropped_x, ids_drop = st.cat_complement()
            self._collect_routed_prediction_checkpointed(
                x, st.ids_keep, dropped_x, ids_drop, st.full_len, st.mask_logits, st.class_logits)
            st.attn_mask = None
        else:
            mask_logits = self._collect_prediction(x, st.mask_logits, st.class_logits)
            st.attn_mask = self._attn_mask(x, mask_logits, i)

    def _prune_stage(self, st: _PruneForward, x: torch.Tensor, i: int):
        """(3) Score and select at a prune stage. Returns
        ``(block_policy, pending_ids, ids_drop)``; ``pending_ids`` is ``None``
        when this block does not physically route."""
        if i not in st.stage_of:
            return st.mask_policy, None, None
        stage = st.stage_of[i]

        # DynamicViT/Reg4Pru keep the MLP selector on the full grid even while
        # Reg4Pru's independent random route is open.
        sel_src = self._reconstruct_route(st, x) if (st.soft or st.random_route) and st.route_is_open() else x
        spatial_x = sel_src[:, st.cur_prefix:, :]

        keep_rate = st.active_keep_rates[stage]
        if st.is_random and self.cfg.MODEL.EOMT_PRUNE_ROUTING_KEEP_RATE >= 0.0:
            # Random eval-only: retain the configured random-eval token budget.
            keep_rate = self.cfg.MODEL.EOMT_PRUNE_ROUTING_KEEP_RATE
        if st.cur_pool is None:
            st.grid_spatial = spatial_x.shape[1]
        pool = spatial_x.shape[1] if st.cur_pool is None else st.cur_pool

        logits, score = self._score_stage(st, stage, spatial_x)

        # A prune at the last pre-query block reserves the query token that the
        # next block prepends, so the query blocks see a rounded sequence length.
        post_round_subtract = (
            self.num_q
            if (not st.soft and not st.random_route
                and i == st.q_start - 1 and i < st.reinject_block)
            else 0
        )
        policy_sp, ids_sp, ids_drop_sp, score, k = self._select_stage(
            logits, keep_rate, pool, st.cur_prefix, post_round_subtract,
            valid_mask=st.prev_keep if st.soft else None,
            is_random=st.is_random, score=score,
            soft=st.soft, gumbel_route=st.gumbel_route,
        )
        st.cur_pool = k
        st.cumulative *= keep_rate
        st.stage_keeps.append(k)
        self._accumulate_budget_loss(st, policy_sp, score, k, spatial_x.shape[1], keep_rate)
        if not st.throughput_mode:
            # DynamicViT/Gumbel report the realised straight-through keep fraction;
            # CROPR and hard/random eval report the ranking score's mean.
            st.keep_means.append(
                (policy_sp.mean() if (st.soft or st.gumbel_route) else score.mean()).detach())
        self._capture_stage(st, score, policy_sp, ids_sp)

        block_policy = st.mask_policy
        pending_ids = None
        if st.soft:
            # DynamicViT (and the MLP arm of Reg4Pru): do not shorten x; only
            # the current block's attention receives the keep mask.
            st.prev_keep = policy_sp
            st.mask_policy = torch.cat((policy_sp.new_ones(st.batch, st.cur_prefix), policy_sp), dim=1)
            block_policy = st.mask_policy
        elif not st.random_route:
            prefix_ids = torch.arange(st.cur_prefix, device=x.device).unsqueeze(0).expand(st.batch, -1)
            pending_ids = torch.cat((prefix_ids, ids_sp + st.cur_prefix), dim=1)
            if self.training and st.gumbel_route:
                # Gumbel Pruning: the soft policy gives the MLP its budget gradient
                # at the decision block before the top-k set is physically gathered.
                block_policy = torch.cat((policy_sp.new_ones(st.batch, st.cur_prefix), policy_sp), dim=1)
        return block_policy, pending_ids, ids_drop_sp

    def _score_stage(self, st: _PruneForward, stage: int, spatial_x: torch.Tensor):
        """Use the MLP scorer (DynamicViT/Gumbel/Reg4Pru) or CROPR scorer."""
        if not self.aux_scorer:
            return self.selectors[stage](spatial_x), None
        if not self.training:
            return None, self.aux_cross_attn[stage].forward_scorer(spatial_x)
        # CROPR: detach features so this loss trains the learned-query scorer, not
        # the encoder. The trainer gathers these raw logits for direct BCE.
        score = self.aux_cross_attn[stage].forward_scorer(spatial_x.detach())
        if st.abs_spatial_ids is None:
            st.abs_spatial_ids = torch.arange(
                st.grid_spatial, device=spatial_x.device).unsqueeze(0).expand(st.batch, -1)
        st.aux_spatial_ids.append(st.abs_spatial_ids.clone())
        st.aux_preds.append(score.unsqueeze(-1))
        return None, score

    def _accumulate_budget_loss(self, st, policy_sp, score, k, denom, keep_rate):
        """Append the MLP-only budget loss for DynamicViT, Gumbel, or Reg4Pru."""
        if self.training and not self.aux_scorer:
            if st.soft:
                # Per-token argmax: drive the *realised* hard keep mask to the
                # cumulative target, else the ratio loss decouples (keep-prob → 1).
                st.budget_losses.append(ratio_loss(policy_sp, st.cumulative))
            elif st.gumbel_route:
                # Survivor set is already reduced by routing, so target the
                # per-stage relative rate.
                st.budget_losses.append(ratio_loss(policy_sp, keep_rate))
            else:
                # Reg4Pru's MLP arm uses ordinary score calibration; the random
                # window itself is introduced separately in _step_random_route.
                st.budget_losses.append(ratio_loss(score, k / max(denom, 1)))

    def _capture_stage(self, st: _PruneForward, score, policy_sp, ids_sp) -> None:
        """Record the full-grid keep map and advance the absolute-id map."""
        if not (st.capture or st.track_abs_ids):
            return
        if st.abs_spatial_ids is None:
            st.abs_spatial_ids = torch.arange(
                st.grid_spatial, device=score.device).unsqueeze(0).expand(st.batch, -1)
        if st.capture:
            # Capture the actually-kept set: the physical top-k for routed modes
            # (policy differs from the routed set for gumbel_route), the policy for
            # soft_mask (nothing is physically routed).
            if st.soft:
                keep_vec = policy_sp.detach()
            else:
                keep_vec = torch.zeros_like(score)
                if ids_sp.shape[1] > 0:
                    keep_vec.scatter_(1, ids_sp, 1.0)
            full_keep = score.new_zeros(st.batch, st.grid_spatial)
            full_keep.scatter_(1, st.abs_spatial_ids, keep_vec)
            st.captured_maps.append(full_keep)
        if not st.soft:
            st.abs_spatial_ids = st.abs_spatial_ids.gather(1, ids_sp)

    def _run_block(self, st: _PruneForward, x: torch.Tensor, i: int, block, block_policy) -> torch.Tensor:
        """(4) Run one transformer block, with reduced RoPE while routed."""
        routed = st.route_is_open()
        if routed:
            block_attn_mask = None  # query→spatial mask is undefined on a reduced sequence
            block_rope = self._route_rope_cached(
                self._route_rope, st.rope, st.ids_keep, st.route_prefix, st.rope_cache)
            if st.soft and block_policy is not None:
                # Soft mask runs on the routed subset; gather the full-grid policy.
                block_policy = block_policy.gather(1, st.ids_keep)
        else:
            block_attn_mask = st.attn_mask
            block_rope = st.rope
        st.dense.append(st.full_len if routed else x.shape[1])
        st.processed.append(x.shape[1])
        x = self._block(block, x, block_attn_mask, block_rope, block_policy)
        return x

    def _apply_gather(self, st: _PruneForward, x: torch.Tensor, pending_ids, ids_drop_sp) -> torch.Tensor:
        """(5) Physically gather the kept set, opening or extending the route."""
        if pending_ids is None:
            return x
        abs_drop_ids = ids_drop_sp + st.cur_prefix
        drop_x = x.gather(1, abs_drop_ids.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        if not st.route_is_open():
            st.ids_keep = pending_ids
            st.dropped_x = [drop_x]
            st.ids_drop = [abs_drop_ids]
            st.full_len = x.shape[1]
        else:
            # Map dropped local positions to absolute grid ids via the old ids_keep
            # before reassigning it.
            st.dropped_x.append(drop_x)
            st.ids_drop.append(st.ids_keep.gather(1, abs_drop_ids))
            st.ids_keep = st.ids_keep.gather(1, pending_ids)
        x = self.router.start_route(x, pending_ids)
        st.route_prefix = st.cur_prefix
        return x

    def _step_random_route(self, st: _PruneForward, x: torch.Tensor, i: int) -> torch.Tensor:
        """Reg4Pru's train-only independent random routing window.

        It is intentionally separate from the MLP's soft policy: this branch
        samples the window and 50%-keep ids, while _prune_stage still trains the
        full-sequence selector.
        """
        if st.random_route and not st.route_is_open() and i == st.route_start:
            st.route_prefix = st.cur_prefix
            st.ids_keep = self.router.get_mask(
                x,
                self.cfg.MODEL.EOMT_PRUNE_ROUTING_KEEP_RATE,
                st.route_prefix,
            )
            drop_x, ids_drop = self.router.dropped_tokens(x, st.ids_keep)
            st.dropped_x = [drop_x]
            st.ids_drop = [ids_drop]
            st.full_len = x.shape[1]
            x = self.router.start_route(x, st.ids_keep)
            if st.capture and not st.soft:
                if st.grid_spatial <= 0:
                    st.grid_spatial = x.shape[1] - st.route_prefix
                if st.abs_spatial_ids is None:
                    st.abs_spatial_ids = torch.arange(
                        st.grid_spatial, device=x.device).unsqueeze(0).expand(st.batch, -1)
                spatial_ids = st.ids_keep[:, st.route_prefix:] - st.route_prefix
                st.abs_spatial_ids = st.abs_spatial_ids.gather(1, spatial_ids)
        elif st.random_route and st.route_is_open() and i == st.route_end:
            x = self._reconstruct_route(st, x)
            st.close_route()
        return x

    def _maybe_reinject(self, st: _PruneForward, x: torch.Tensor, i: int) -> torch.Tensor:
        """(6) Permanently scatter an open route back at the reinjection block.
        The sequence returns to full grid order, so a later stage selects from all
        tokens again (the abs-id map restarts lazily)."""
        if st.route_is_open() and not st.random_route and i == st.reinject_block:
            x = self._reconstruct_route(st, x)
            st.close_route()
        return x

    def _reconstruct_route(self, st: _PruneForward, cur_x: torch.Tensor) -> torch.Tensor:
        dropped_x, ids_drop = st.cat_complement()
        return self.router.merge_route(cur_x, st.ids_keep, dropped_x, ids_drop, st.full_len)

    def _clear_diagnostics(self) -> None:
        """Reset all per-forward diagnostic buffers (used by the dense fast path)."""
        self._last_pruning_loss = None
        self._last_aux_preds = []
        self._last_aux_spatial_ids = []
        self._last_token_stats = {}
        self._keep_mean_tensors = []
        self._selection_maps = []

    def _finalize(self, st: _PruneForward) -> None:
        """Publish the loss, aux predictions, token stats and captures."""
        self._last_pruning_loss = (
            self.cfg.MODEL.EOMT_PRUNE_LOSS_WEIGHT * torch.stack(st.budget_losses).mean()
            if st.budget_losses else None)
        self._last_aux_preds = st.aux_preds
        self._last_aux_spatial_ids = st.aux_spatial_ids

        effective = float(sum(st.processed))
        dense_steps = float(sum(st.dense)) if st.dense else effective
        stats = {
            "token_stats/effective_token_steps": effective,
            "token_stats/dense_token_steps": dense_steps,
            "token_stats/token_step_ratio": effective / max(dense_steps, 1.0),
            "token_stats/prune_stage_count": float(len(self.prune_layers)),
            "token_stats/prune_round_multiple": float(
                self.cfg.MODEL.PRUNE_ROUND_MULTIPLE),
            "token_stats/prune_curriculum_alpha": float(
                self._prune_curriculum_alpha if self.training else 1.0),
        }
        if st.random_route:
            stats.update({
                "token_stats/random_route_keep_rate": float(
                    self.cfg.MODEL.EOMT_PRUNE_ROUTING_KEEP_RATE),
                "token_stats/random_route_start_after_block": float(st.route_start),
                "token_stats/random_route_end_after_block": float(st.route_end),
            })
        if st.grid_spatial > 0:
            for j, kept in enumerate(st.stage_keeps):
                stats[f"token_stats/remaining_after_prune_{j}"] = float(kept)
                stats[f"token_stats/keep_fraction_{j}"] = kept / st.grid_spatial
            stats["token_stats/final_keep_fraction"] = (
                st.stage_keeps[-1] / st.grid_spatial if st.stage_keeps else 1.0)
        self._last_token_stats = stats
        self._keep_mean_tensors = st.keep_means

        self._selection_maps = []
        if st.capture and st.captured_maps and st.grid_spatial > 0:
            gh, gw = self._get_current_grid_size()
            if gh * gw == st.grid_spatial:
                for keep in st.captured_maps:
                    self._selection_maps.append({
                        "keep": keep.reshape(st.batch, gh, gw).cpu(),
                        "grid": (gh, gw),
                    })
