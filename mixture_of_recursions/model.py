"""Shared recurrent SiT blocks with nested, fixed-capacity expert-choice routing.

Hard Top-K indices are discrete. Selected sigmoid scores gate the next unit's
full output inside an outer recursion residual, supplying router gradients.
No SiTBlock, AdaLN, embedding, or final-layer internals are changed.
"""

from functools import partial
import math

import torch
from torch import nn

from models.sit_b1 import SiT as VanillaSiT


def gather_tokens(x, indices):
    """[B,N,D], [B,K] -> [B,K,D], with no sample loop."""
    return x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def scatter_tokens(base, indices, updates):
    """Out-of-place replacement; gradients reach updates and inactive base slots."""
    return base.scatter(1, indices.unsqueeze(-1).expand_as(updates), updates)


class SiT(VanillaSiT):
    def __init__(
        self, *, depth=28, use_mor=False, mor_prefix_blocks=3,
        mor_recurrent_blocks=2, mor_suffix_blocks=3, mor_max_recursions=3,
        mor_capacity_ratios=(1.0, 0.5, 0.25), use_recursion_conditioning=False,
        mor_global_topk=False, mor_gating=True,
        **kwargs,
    ):
        # The disabled path constructs exactly the baseline, including RNG use,
        # parameter names, zero initialization, and strict checkpoint compatibility.
        if not use_mor:
            super().__init__(depth=depth, **kwargs)
            self.use_mor = False
            return

        sizes = (mor_prefix_blocks, mor_recurrent_blocks, mor_suffix_blocks, mor_max_recursions)
        if any(type(value) is not int for value in sizes):
            raise ValueError("MoR block counts and max_recursions must be integers")
        if min(mor_prefix_blocks, mor_suffix_blocks) < 0 or min(mor_recurrent_blocks, mor_max_recursions) < 1:
            raise ValueError("prefix/suffix must be >= 0; recurrent_blocks/max_recursions must be >= 1")
        effective_depth = (
            mor_prefix_blocks + mor_recurrent_blocks * mor_max_recursions + mor_suffix_blocks
        )
        ratios = tuple(float(ratio) for ratio in mor_capacity_ratios)
        if (len(ratios) != mor_max_recursions or ratios[0] != 1.0
                or any(not math.isfinite(ratio) or not 0 < ratio <= 1 for ratio in ratios)
                or any(later > earlier for earlier, later in zip(ratios, ratios[1:]))):
            raise ValueError("capacity ratios must start at 1, be positive/nonincreasing, and match max_recursions")
        if not mor_gating and any(ratio != 1.0 for ratio in ratios):
            raise ValueError("--no-mor-gating requires all capacity ratios to be 1.0")

        # Construct only unique blocks, initialize through the original SiT, then
        # register each object in exactly one stage. No recursion-specific copies.
        super().__init__(depth=mor_prefix_blocks + mor_recurrent_blocks + mor_suffix_blocks, **kwargs)
        blocks = self.blocks
        del self.blocks
        self.prefix_blocks = nn.ModuleList(blocks[:mor_prefix_blocks])
        self.recurrent_blocks = nn.ModuleList(blocks[mor_prefix_blocks:mor_prefix_blocks + mor_recurrent_blocks])
        self.suffix_blocks = nn.ModuleList(blocks[mor_prefix_blocks + mor_recurrent_blocks:])
        self.use_mor = True
        self.base_depth = depth
        self.effective_depth = effective_depth
        # Retain this attribute for callers written before configurable
        # effective depth was supported. It denotes the named SiT base depth.
        self.original_depth = depth
        self.mor_max_recursions = mor_max_recursions
        self.mor_capacity_ratios = ratios
        self.use_recursion_conditioning = use_recursion_conditioning
        self.mor_global_topk = mor_global_topk
        self.mor_gating = mor_gating
        hidden_size = self.pos_embed.shape[-1]
        self.recursion_embed = nn.Embedding(mor_max_recursions, hidden_size) if use_recursion_conditioning else None
        if self.recursion_embed is not None:
            nn.init.normal_(self.recursion_embed.weight, mean=0.0, std=0.01)
        self.routers = nn.ModuleList([
            nn.Linear(hidden_size, 1)
            for _ in range(mor_max_recursions - 1)
        ])
        for router in self.routers:
            nn.init.normal_(router.weight, std=0.01)
            nn.init.zeros_(router.bias)
        if not mor_gating:
            # With full capacity and no gate, routers affect neither selection
            # nor values. Freeze them so DDP sees no unused trainable parameters
            # while keeping state-dict topology compatible with gated models.
            self.routers.requires_grad_(False)
        # Ratios are relative to original N, not the previous subset. Round down
        # and retain at least one token; capacities are identical across samples.
        tokens = self.x_embedder.num_patches
        self.active_token_counts = tuple(max(1, math.floor(tokens * ratio)) for ratio in ratios)

    def forward(self, x, t, y, return_logvar=False, force_drop_ids=None, return_aux=False):
        if not self.use_mor:
            output = super().forward(x, t, y, return_logvar=return_logvar, force_drop_ids=force_drop_ids)
            if return_aux:
                return output[0], {"router_scores": [], "selected_indices": [], "active_token_counts": []}
            return output

        x = self.x_embedder(x) + self.pos_embed
        c_base = self.t_embedder(t) + self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)
        for block in self.prefix_blocks:
            x = block(x, c_base)

        batch, tokens, _ = x.shape
        indices = torch.arange(tokens, device=x.device).unsqueeze(0).expand(batch, -1)
        active = x
        selected_scores = None
        aux = {"router_scores": [], "selected_indices": [], "active_token_counts": []} if return_aux else None
        for recursion in range(self.mor_max_recursions):
            if return_aux:
                aux["selected_indices"].append(indices.detach())
                aux["active_token_counts"].append(self.active_token_counts[recursion])
            c_r = c_base if self.recursion_embed is None else c_base + self.recursion_embed.weight[recursion]
            before = active
            for block in self.recurrent_blocks:
                active = block(active, c_r)

            if recursion > 0:
                # Intentionally retain both residual levels: SiTBlock's internal
                # residuals and this outer MoR residual on the full unit output.
                if self.mor_gating:
                    gate = selected_scores.to(active.dtype).unsqueeze(-1)
                    active = before + gate * active
                else:
                    active = before + active
            x = active if recursion == 0 else scatter_tokens(x, indices, active)

            if recursion + 1 < self.mor_max_recursions:
                if not self.mor_gating:
                    # Capacity is necessarily full, so no ranking/gather is
                    # needed before the next dense recurrence.
                    active = x
                    continue
                # Hierarchical mode ranks only the current subset. Global mode
                # ranks the restored full sequence, so later sets can include
                # tokens that skipped the immediately preceding recursion.
                routing_hidden = x if self.mor_global_topk else active
                logits = self.routers[recursion](routing_hidden).squeeze(-1)
                scores = logits.float().sigmoid()
                selected_scores, local_indices = scores.topk(self.active_token_counts[recursion + 1], dim=1)
                if return_aux:
                    aux["router_scores"].append(scores.detach())
                active = gather_tokens(routing_hidden, local_indices)
                indices = local_indices if self.mor_global_topk else indices.gather(1, local_indices)

        for block in self.suffix_blocks:
            x = block(x, c_base)
        output = self.unpatchify(self.final_layer(x, c_base))
        # Preserve the repository's sampler/loss contract (prediction, None).
        return (output, aux) if return_aux else (output, None)


def build_mor_sit(model_name="SiT-B/2", **kwargs):
    """Match the existing named sizes; L/XL MoR stage lengths are explicit."""
    size, patch = model_name.removeprefix("SiT-").split("/")
    depth, hidden_size, heads = {"S": (12, 384, 6), "B": (12, 768, 12),
                                 "L": (24, 1024, 16), "XL": (28, 1152, 16)}[size]
    if int(patch) not in (2, 4, 8):
        raise ValueError("patch size must be 2, 4, or 8")
    # Preserve even the original S factory's decoder default when MoR is off.
    decoder_hidden_size = 768 if size == "S" and not kwargs.get("use_mor", False) else hidden_size
    return SiT(depth=depth, hidden_size=hidden_size, decoder_hidden_size=decoder_hidden_size,
               num_heads=heads, patch_size=int(patch), **kwargs)


SiT_models = {f"SiT-{size}/{patch}": partial(build_mor_sit, f"SiT-{size}/{patch}")
              for size in ("S", "B", "L", "XL") for patch in (2, 4, 8)}
