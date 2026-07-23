"""Learning-rate scheduling utilities."""

from __future__ import annotations


class LRScheduleController:
    """Apply LR multipliers to optimizer param groups."""

    def __init__(self, cfg, schedule_fn):
        self.cfg = cfg
        self.schedule_fn = schedule_fn
        self._split_warmup: bool = cfg.TRAIN.get("SPLIT_WARMUP", False)
        _wu = list(cfg.TRAIN.get("WARMUP_STEPS", [10, 50]))
        self._non_vit_wu: int = _wu[0]
        self._vit_wu: int = _wu[1] if len(_wu) > 1 else 0

    def lr_mult(self, pg: dict, step: int, total_steps: int) -> float:
        """Return the LR multiplier for a param group at the given step.

        When SPLIT_WARMUP is enabled, backbone groups are frozen for
        WARMUP_STEPS[0] steps then warmed up over WARMUP_STEPS[1] steps;
        head/decoder groups warm up immediately over WARMUP_STEPS[0] steps.
        Otherwise all groups follow the unified schedule_fn.
        """
        if not self._split_warmup:
            return self.schedule_fn(step / max(total_steps, 1))

        non_vit_wu = self._non_vit_wu
        vit_wu = self._vit_wu

        if pg.get("group_kind") == "backbone":
            # Frozen during head warmup, then its own warmup, then main schedule
            if step < non_vit_wu:
                return 0.0
            adj = step - non_vit_wu
            if adj < vit_wu:
                return adj / vit_wu if vit_wu > 0 else 1.0
            tail_steps = max(total_steps - non_vit_wu - vit_wu, 1)
            return self.schedule_fn(min((adj - vit_wu) / tail_steps, 1.0))
        else:
            # Head params: warm up over non_vit_wu steps then follow main schedule
            if step < non_vit_wu:
                return step / non_vit_wu if non_vit_wu > 0 else 1.0
            tail_steps = max(total_steps - non_vit_wu, 1)
            return self.schedule_fn(min((step - non_vit_wu) / tail_steps, 1.0))

    def apply(self, optimizer, step: int, total_steps: int) -> None:
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", self.cfg.TRAIN.LR) * self.lr_mult(
                pg, step, total_steps
            )
