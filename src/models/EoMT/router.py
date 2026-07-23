"""Token routing (gather/scatter over a kept-index set).

Adapted, with modifications, from TREAD: Token Routing for Efficient
Architecture-agnostic Diffusion Training —
https://github.com/CompVis/tread (routing_module.py).
"""
import torch

from models.pruning import round_pruned_tokens_to_total_multiple


class Router:

    @torch.compiler.disable
    def get_mask(self, x, selection_rate=0.0, prefix_tokens=1):
        """Randomly sample a keep-index set for the route-once window.
        """
        B, N, _ = x.shape
        device = x.device

        spatial_len = N - prefix_tokens
        assert spatial_len >= 0

        # Round (prefix_tokens + kept_spatial) to a multiple of 8 for tensor-core
        # alignment, matching _select_stage's total_tokens_offset=cur_prefix.
        num_mask_spatial = round_pruned_tokens_to_total_multiple(
            spatial_len * (1 - float(selection_rate)),
            spatial_len,
            total_tokens_offset=prefix_tokens,
            min_keep=0,
        )
        num_keep_spatial = spatial_len - num_mask_spatial

        if spatial_len == 0 or num_keep_spatial <= 0:
            # nothing in the spatial to keep, only prefix survives
            spatial_ids_keep = torch.empty(B, 0, dtype=torch.long, device=device)
        else:
            noise = torch.rand(B, spatial_len, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            spatial_ids_keep = ids_shuffle[:, :num_keep_spatial]  # [B, num_keep_spatial]

            # shift into full sequence indices
            spatial_ids_keep = spatial_ids_keep + prefix_tokens

        if prefix_tokens > 0:
            prefix_ids = torch.arange(prefix_tokens, device=device)
            prefix_ids = prefix_ids.unsqueeze(0).expand(B, -1)  # [B, prefix_tokens]
            ids_keep = torch.cat([prefix_ids, spatial_ids_keep], dim=1)  # [B, prefix + spatial_keep]
        else:
            ids_keep = spatial_ids_keep
        return ids_keep

    def start_route(self, x, ids_keep):
        x_masked = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, x.size(2)))
        return x_masked

    def dropped_tokens(self, x, ids_keep):
        B, N, C = x.shape
        keep = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        keep.scatter_(1, ids_keep, True)
        ids_all = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        ids_drop = ids_all.masked_select(~keep).view(B, N - ids_keep.shape[1])
        dropped_x = x.gather(1, ids_drop.unsqueeze(-1).expand(-1, -1, C))
        return dropped_x, ids_drop

    def merge_route(self, masked_x, ids_keep, dropped_x, ids_drop, full_len):
        B, _, C = masked_x.shape
        x_unmasked = masked_x.new_empty(B, int(full_len), C)
        x_unmasked.scatter_(
            1, ids_keep.unsqueeze(-1).expand(-1, -1, C), masked_x
        )
        if dropped_x.shape[1] > 0:
            x_unmasked.scatter_(
                1, ids_drop.unsqueeze(-1).expand(-1, -1, C), dropped_x
            )
        return x_unmasked
