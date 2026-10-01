"""SiT with a fixed random active-token subset across a routed block range."""

from functools import partial

from tread_routesync.attention_correction import configure_correction

import torch
import torch.distributed as dist

from models.sit_b1 import SiT as VanillaSiT


def gather_tokens(x, indices):
    """Gather per-sample [B,K] positions from [B,N,D]."""
    return x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def restore_token_order(grouped, permutation):
    """Restore active/bypass grouped states without an in-place update."""
    positions = torch.arange(permutation.shape[1], device=permutation.device)
    positions = positions.unsqueeze(0).expand_as(permutation)
    inverse = torch.zeros_like(permutation).scatter(1, permutation, positions)
    return gather_tokens(grouped, inverse)


def random_subset_indices(batch, tokens, active_tokens, device, generator=None):
    """Return a uniform random active subset and its disjoint complement."""
    if not 0 < active_tokens <= tokens:
        raise ValueError(
            f"active token count must be in [1, {tokens}], got {active_tokens}"
        )
    scores = torch.rand(batch, tokens, device=device, generator=generator)
    permutation = scores.argsort(dim=1)
    return permutation[:, :active_tokens], permutation[:, active_tokens:]


class SiT(VanillaSiT):
    def __init__(
        self, *, depth=28, use_tread_routing=False,
        tread_start_block=3, tread_end_block=9, tread_active_ratio=0.5,
        tread_recursive=False, tread_num_groups=3,
        tread_recursive_pattern="grouped",
        tread_eval_mode="sparse", tread_fp32_endpoint=True,
        tread_attn_correction=False, tread_attn_correction_strength=1.0,
        tread_attn_correction_backend="sdpa",
        tread_seed=None, tread_debug=False, **kwargs,
    ):
        super().__init__(depth=depth, **kwargs)
        self.use_tread_routing = use_tread_routing
        self.tread_start_block = tread_start_block
        self.tread_end_block = tread_end_block
        self.tread_active_ratio = tread_active_ratio
        self.tread_recursive = tread_recursive
        self.tread_num_groups = tread_num_groups
        self.tread_recursive_pattern = tread_recursive_pattern
        self.tread_eval_mode = tread_eval_mode
        self.tread_fp32_endpoint = tread_fp32_endpoint
        self.tread_seed = tread_seed
        self.tread_debug = tread_debug
        self._tread_debug_printed = False
        self.base_depth = depth

        if tread_eval_mode not in ("sparse", "dense"):
            raise ValueError("tread_eval_mode must be 'sparse' or 'dense'")
        if not 0 < tread_active_ratio <= 1:
            raise ValueError("tread_active_ratio must be in (0, 1]")
        if tread_recursive and not use_tread_routing:
            raise ValueError("--tread-recursive requires --use-tread-routing")
        if tread_recursive_pattern not in ("grouped", "interleaved"):
            raise ValueError(
                "tread_recursive_pattern must be 'grouped' or 'interleaved'"
            )
        if use_tread_routing and not 0 <= tread_start_block < tread_end_block <= depth:
            raise ValueError(
                f"route range must satisfy 0 <= start < end <= depth ({depth}), "
                f"got [{tread_start_block}, {tread_end_block})"
            )
        if use_tread_routing and tread_recursive:
            routed_blocks = tread_end_block - tread_start_block
            if tread_num_groups < 1 or routed_blocks % tread_num_groups:
                raise ValueError(
                    "recursive routed length must be divisible by tread_num_groups: "
                    f"{routed_blocks} vs {tread_num_groups}"
                )
            self.tread_blocks_per_group = routed_blocks // tread_num_groups
            original_blocks = list(self.blocks)
            del self.blocks
            self.prefix_blocks = torch.nn.ModuleList(
                original_blocks[:tread_start_block]
            )
            representative_offsets = (
                [group * self.tread_blocks_per_group for group in range(tread_num_groups)]
                if tread_recursive_pattern == "grouped"
                else list(range(tread_num_groups))
            )
            self.recurrent_group_blocks = torch.nn.ModuleList([
                original_blocks[tread_start_block + offset]
                for offset in representative_offsets
            ])
            self.suffix_blocks = torch.nn.ModuleList(
                original_blocks[tread_end_block:]
            )

        configure_correction(self, tread_attn_correction,
            tread_attn_correction_strength, tread_attn_correction_backend)

    def _sparse_for_forward(self, eval_mode):
        if not self.use_tread_routing:
            return False
        if self.training:
            return True
        mode = self.tread_eval_mode if eval_mode is None else eval_mode
        if mode not in ("sparse", "dense"):
            raise ValueError("tread_eval_mode must be 'sparse' or 'dense'")
        return mode == "sparse"

    def _subset_generator(self, device):
        if self.training or self.tread_seed is None:
            return None
        rank = dist.get_rank() if dist.is_initialized() else 0
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self.tread_seed) + rank)
        return generator

    def _run_routed_block(self, block, hidden, condition, *, endpoint):
        if not (self.training and self.tread_fp32_endpoint and endpoint):
            return block(hidden, condition)
        output_dtype = hidden.dtype
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            output = block(hidden.float(), condition.float())
        return output.to(dtype=output_dtype)

    def _recursive_group_for_offset(self, offset):
        if self.tread_recursive_pattern == "grouped":
            return offset // self.tread_blocks_per_group
        return offset % self.tread_num_groups

    def forward(
        self, x, t, y, return_logvar=False, force_drop_ids=None,
        tread_eval_mode=None, return_routing_info=False,
    ):
        sparse = self._sparse_for_forward(tread_eval_mode)
        if not sparse and not self.tread_recursive:
            output = super().forward(
                x, t, y, return_logvar=return_logvar,
                force_drop_ids=force_drop_ids,
            )
            if not return_routing_info:
                return output
            tokens = self.x_embedder.num_patches
            return output[0], output[1], {
                "routing_applied": False,
                "active_indices": None,
                "bypass_indices": None,
                "block_input_token_counts": [tokens] * self.base_depth,
                "post_block_token_counts": [tokens] * self.base_depth,
                "fp32_endpoint_applied": False,
                "recursive": False,
            }

        x = self.x_embedder(x) + self.pos_embed
        batch, tokens, _ = x.shape
        condition = self.t_embedder(t) + self.y_embedder(
            y, self.training, force_drop_ids=force_drop_ids,
        )
        prefix_blocks = (
            self.prefix_blocks if self.tread_recursive
            else self.blocks[:self.tread_start_block]
        )
        suffix_blocks = (
            self.suffix_blocks if self.tread_recursive
            else self.blocks[self.tread_end_block:]
        )
        for block in prefix_blocks:
            x = block(x, condition)

        if not sparse:
            routed_blocks = self.tread_end_block - self.tread_start_block
            for offset in range(routed_blocks):
                group = self._recursive_group_for_offset(offset)
                x = self.recurrent_group_blocks[group](x, condition)
            for block in suffix_blocks:
                x = block(x, condition)
            output = self.unpatchify(self.final_layer(x, condition))
            if not return_routing_info:
                return output, None
            return output, None, {
                "routing_applied": False,
                "active_indices": None,
                "bypass_indices": None,
                "block_input_token_counts": [tokens] * self.base_depth,
                "post_block_token_counts": [tokens] * self.base_depth,
                "fp32_endpoint_applied": False,
                "recursive": True,
            }

        active_tokens = int(round(tokens * self.tread_active_ratio))
        active_tokens = max(1, min(tokens, active_tokens))
        active_indices, bypass_indices = random_subset_indices(
            batch, tokens, active_tokens, x.device,
            generator=self._subset_generator(x.device),
        )
        permutation = torch.cat((active_indices, bypass_indices), dim=1)
        grouped = gather_tokens(x, permutation)
        active = grouped[:, :active_tokens]
        bypass = grouped[:, active_tokens:]
        input_counts = (
            [tokens] * self.tread_start_block if return_routing_info else None
        )
        output_counts = (
            [tokens] * self.tread_start_block if return_routing_info else None
        )

        if self.tread_debug and not self._tread_debug_printed:
            self._tread_debug_printed = True
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    "[TREAD Fixed-Subset Routing]\n"
                    f"tokens: {tokens}\n"
                    f"active tokens/block: {active_tokens} / {tokens} "
                    f"({active_tokens / tokens:.2%})\n"
                    f"bypass tokens: {tokens - active_tokens}\n"
                    f"routed blocks: {self.tread_start_block} -> "
                    f"{self.tread_end_block - 1}\n"
                    "train mode: sparse\n"
                    f"eval mode: {self.tread_eval_mode}\n"
                    f"training FP32 endpoint: {self.tread_fp32_endpoint}\n"
                    f"recursive group blocks: {self.tread_recursive}\n"
                    f"recursive pattern: {self.tread_recursive_pattern}"
                )

        for block_index in range(self.tread_start_block, self.tread_end_block):
            offset = block_index - self.tread_start_block
            block = (
                self.recurrent_group_blocks[
                    self._recursive_group_for_offset(offset)
                ]
                if self.tread_recursive else self.blocks[block_index]
            )
            active = self._run_routed_block(
                block, active, condition,
                endpoint=block_index == self.tread_end_block - 1,
            )
            if return_routing_info:
                input_counts.append(active.shape[1])
                output_counts.append(tokens)

        x = restore_token_order(torch.cat((active, bypass), dim=1), permutation)
        for block in suffix_blocks:
            x = block(x, condition)
            if return_routing_info:
                input_counts.append(tokens)
                output_counts.append(tokens)

        output = self.unpatchify(self.final_layer(x, condition))
        if not return_routing_info:
            return output, None
        return output, None, {
            "routing_applied": True,
            "active_indices": active_indices.detach(),
            "bypass_indices": bypass_indices.detach(),
            "block_input_token_counts": input_counts,
            "post_block_token_counts": output_counts,
            "fp32_endpoint_applied": bool(
                self.training and self.tread_fp32_endpoint
            ),
            "recursive": self.tread_recursive,
        }


def build_tread_sit(model_name="SiT-B/2", **kwargs):
    size, patch = model_name.removeprefix("SiT-").split("/")
    depth, hidden_size, heads = {
        "S": (12, 384, 6), "B": (12, 768, 12),
        "L": (24, 1024, 16), "XL": (28, 1152, 16),
    }[size]
    patch = int(patch)
    if patch not in (2, 4, 8):
        raise ValueError("patch size must be 2, 4, or 8")
    decoder_hidden_size = 768 if size == "S" else hidden_size
    return SiT(
        depth=depth, hidden_size=hidden_size,
        decoder_hidden_size=decoder_hidden_size, num_heads=heads,
        patch_size=patch, **kwargs,
    )


SiT_models = {
    f"SiT-{size}/{patch}": partial(build_tread_sit, f"SiT-{size}/{patch}")
    for size in ("S", "B", "L", "XL") for patch in (2, 4, 8)
}
