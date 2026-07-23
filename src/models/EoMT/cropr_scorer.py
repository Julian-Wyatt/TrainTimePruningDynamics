"""CROPR's learned-query token scorer used by the paper experiments."""

import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_tf_


class CrossAttention(nn.Module):
    """Scores spatial tokens by cross-attention from a set of learnable queries."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_queries: int = 1,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        self.init_weights()

    def init_weights(self):
        embed_dim = self.queries.shape[-1]
        trunc_normal_tf_(self.queries, std=embed_dim ** -0.5)

    def forward_scorer(self, x):
        """Return each spatial token's raw foreground logit."""
        query_sum = self.queries.sum(dim=1, keepdim=True)
        return torch.matmul(x, query_sum.transpose(-1, -2)).squeeze(-1)
