import contextlib

import torch
import torch.nn as nn
from timm.utils import ModelEmaV3


class ModelEMA:
    """EMA wrapper backed by timm.utils.ModelEmaV3.

    Public interface:
        update(model)          — EMA step after each optimiser update
        ema_scope(model)       — context manager: swap EMA→model, restore after
        state_dict()           — shadow model state dict (for checkpointing)
        load_state_dict(sd)    — restore shadow model from checkpoint
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        # foreach requires CUDA; disable on MPS/CPU
        foreach = torch.cuda.is_available()
        self._ema = ModelEmaV3(model, decay=decay, foreach=foreach)
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._ema.update(model)

    @contextlib.contextmanager
    def ema_scope(self, model: nn.Module):
        # Save live model parameters
        self._backup = {n: p.data.clone() for n, p in model.named_parameters()}
        # Overwrite live model with EMA (shadow) parameters
        for m_param, e_param in zip(model.parameters(), self._ema.module.parameters()):
            m_param.data.copy_(e_param.data)
        try:
            yield
        finally:
            # Restore live model parameters
            for n, p in model.named_parameters():
                if n in self._backup:
                    p.data.copy_(self._backup[n])
            self._backup = {}

    def state_dict(self):
        return self._ema.module.state_dict()

    def load_state_dict(self, state_dict):
        from .checkpoint import _adapt_state_dict_to_model
        self._ema.module.load_state_dict(
            _adapt_state_dict_to_model(state_dict, self._ema.module)
        )
