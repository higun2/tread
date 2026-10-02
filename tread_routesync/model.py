"""SiT with unchanged TREAD routing and training-only RouteSync features."""

from functools import partial
import math

from .attention_correction import configure_correction

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
        tread_active_ratios=None,
        tread_recursive=False, tread_num_groups=3,
        tread_recursive_pattern="grouped",
        tread_depth_embedding=False,
        tread_eval_mode="sparse", tread_fp32_endpoint=True,
        tread_attn_correction=False, tread_attn_correction_strength=1.0,
        tread_attn_correction_backend="sdpa",
        tread_seed=None, tread_debug=False,
        use_routesync=False, routesync_weight=0.0, routesync_loss_type="relational",
        routesync_sample_ratio=1.0, routesync_target_blocks=None,
        routesync_debug=False, use_dense_sparse_sync=False,
        dense_sparse_sync_ratio=0.1, dense_sparse_sync_weight=0.1,
        dense_sparse_sync_tokens="active", **kwargs,
    ):
        super().__init__(depth=depth, **kwargs)
        self.use_tread_routing = use_tread_routing
        self.tread_start_block = tread_start_block
        self.tread_end_block = tread_end_block
        self.tread_active_ratio = tread_active_ratio
        self.tread_active_ratios = (
            None if tread_active_ratios is None else tuple(tread_active_ratios)
        )
        self.last_tread_active_ratio = tread_active_ratio
        self.last_tread_active_fraction = tread_active_ratio
        self.tread_recursive = tread_recursive
        self.tread_num_groups = tread_num_groups
        self.tread_recursive_pattern = tread_recursive_pattern
        self.tread_depth_embedding = tread_depth_embedding
        self.tread_eval_mode = tread_eval_mode
        self.tread_fp32_endpoint = tread_fp32_endpoint
        self.tread_seed = tread_seed
        self.tread_debug = tread_debug
        self.use_dense_sparse_sync = use_dense_sparse_sync
        self.dense_sparse_sync_ratio = dense_sparse_sync_ratio
        self.dense_sparse_sync_weight = dense_sparse_sync_weight
        if dense_sparse_sync_tokens not in ("active", "routed", "all"):
            raise ValueError("dense_sparse_sync_tokens must be active, routed, or all")
        self.dense_sparse_sync_tokens = dense_sparse_sync_tokens
        if not 0 < dense_sparse_sync_ratio <= 1:
            raise ValueError("dense_sparse_sync_ratio must be in (0, 1]")
        if not math.isfinite(dense_sparse_sync_weight) or dense_sparse_sync_weight < 0:
            raise ValueError("dense_sparse_sync_weight must be finite and nonnegative")
        if use_dense_sparse_sync and not use_tread_routing:
            raise ValueError("--use-dense-sparse-sync requires --use-tread-routing")
        if use_dense_sparse_sync and tread_end_block >= depth:
            raise ValueError(
                "Dense-sparse sync requires a full-token block after merge; "
                "tread_end_block must be less than depth"
            )
        self.use_routesync = use_routesync
        self.routesync_weight = routesync_weight
        if routesync_loss_type not in ("relational", "feature-cosine"):
            raise ValueError("routesync_loss_type must be relational or feature-cosine")
        self.routesync_loss_type = routesync_loss_type
        self.routesync_sample_ratio = routesync_sample_ratio
        self.routesync_debug = routesync_debug
        self.routesync_target_blocks = (
            (tread_end_block,) if routesync_target_blocks is None
            else tuple(routesync_target_blocks)
        )
        self.last_routesync_target_block = None
        if routesync_target_blocks is not None or use_routesync:
            targets = self.routesync_target_blocks
            if not targets or any(type(b) is not int or not tread_end_block <= b < depth for b in targets):
                raise ValueError(f"routesync_target_blocks must be nonempty zero-based suffix indices in [{tread_end_block}, {depth})")
            if len(set(targets)) != len(targets):
                raise ValueError("routesync_target_blocks must contain distinct indices")
            if routesync_target_blocks is not None and not use_routesync:
                raise ValueError("--routesync-target-blocks requires --use-routesync")
        self._tread_debug_printed = False
        self.base_depth = depth

        if tread_eval_mode not in ("sparse", "dense"):
            raise ValueError("tread_eval_mode must be 'sparse' or 'dense'")
        if not 0 < tread_active_ratio <= 1:
            raise ValueError("tread_active_ratio must be in (0, 1]")
        if self.tread_active_ratios is not None:
            if not use_tread_routing:
                raise ValueError("--tread-active-ratios requires --use-tread-routing")
            if not self.tread_active_ratios or any(
                not 0 < ratio < 1 for ratio in self.tread_active_ratios
            ):
                raise ValueError("tread_active_ratios must be non-empty and all in (0, 1)")
            if len(set(self.tread_active_ratios)) != len(self.tread_active_ratios):
                raise ValueError("tread_active_ratios must contain distinct values")
        if tread_recursive and not use_tread_routing:
            raise ValueError("--tread-recursive requires --use-tread-routing")
        if tread_depth_embedding and not tread_recursive:
            raise ValueError("--tread-depth-embedding requires --tread-recursive")
        if tread_recursive_pattern not in ("grouped", "interleaved"):
            raise ValueError(
                "tread_recursive_pattern must be 'grouped' or 'interleaved'"
            )
        if use_tread_routing and not 0 <= tread_start_block < tread_end_block <= depth:
            raise ValueError(
                f"route range must satisfy 0 <= start < end <= depth ({depth}), "
                f"got [{tread_start_block}, {tread_end_block})"
            )
        if not 0.0 < routesync_sample_ratio <= 1.0:
            raise ValueError("routesync_sample_ratio must be in (0, 1]")
        if routesync_weight < 0:
            raise ValueError("routesync_weight must be non-negative")
        if use_routesync and not use_tread_routing:
            raise ValueError("--use-routesync requires --use-tread-routing")
        if use_routesync and tread_end_block >= depth:
            raise ValueError(
                "RouteSync requires an existing full-token block after "
                "reintegration, so tread_end_block must be less than depth"
            )
        if use_routesync and not 0 < tread_active_ratio < 1:
            raise ValueError(
                "RouteSync requires non-empty routed and processed token groups, "
                "so tread_active_ratio must be in (0, 1)"
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
            if tread_depth_embedding:
                self.tread_depth_embeddings = torch.nn.Parameter(
                    torch.empty(
                        routed_blocks,
                        self.y_embedder.embedding_table.embedding_dim,
                    )
                )
                torch.nn.init.normal_(self.tread_depth_embeddings, std=0.02)
            else:
                self.tread_depth_embeddings = None

        configure_correction(self, tread_attn_correction,
            tread_attn_correction_strength, tread_attn_correction_backend)

    def _routesync_target_for_forward(self, device):
        if len(self.routesync_target_blocks) == 1:
            return self.routesync_target_blocks[0]
        choice = torch.zeros((), dtype=torch.long, device=device)
        distributed = dist.is_available() and dist.is_initialized()
        if not distributed or dist.get_rank() == 0:
            choice.random_(len(self.routesync_target_blocks))
        if distributed:
            dist.broadcast(choice, src=0)
        return self.routesync_target_blocks[int(choice.item())]

    def _sparse_for_forward(self, eval_mode):
        if not self.use_tread_routing:
            return False
        if self.training:
            return True
        mode = self.tread_eval_mode if eval_mode is None else eval_mode
        if mode not in ("sparse", "dense"):
            raise ValueError("tread_eval_mode must be 'sparse' or 'dense'")
        return mode == "sparse"

    def _active_ratio_for_forward(self, device):
        if not self.training or self.tread_active_ratios is None:
            return self.tread_active_ratio
        # One selection for the whole forward; broadcast avoids rank-dependent
        # shapes/workload. Global RNG is covered by existing resume checkpoints.
        choice = torch.zeros((), dtype=torch.long, device=device)
        distributed = dist.is_available() and dist.is_initialized()
        if not distributed or dist.get_rank() == 0:
            choice.random_(len(self.tread_active_ratios))
        if distributed:
            dist.broadcast(choice, src=0)
        return self.tread_active_ratios[int(choice.item())]

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

    def _condition_for_logical_depth(self, condition, offset):
        if not self.tread_depth_embedding:
            return condition
        depth_embedding = self.tread_depth_embeddings[offset].to(
            dtype=condition.dtype,
        )
        return condition + depth_embedding.unsqueeze(0)

    def _recursive_block_for_offset(self, offset):
        return self.recurrent_group_blocks[
            self._recursive_group_for_offset(offset)
        ]

    def forward(
        self, x, t, y, return_logvar=False, force_drop_ids=None,
        tread_eval_mode=None, return_routing_info=False,
    ):
        sparse = self._sparse_for_forward(tread_eval_mode)
        capture_routesync = self.training and self.use_routesync and sparse
        capture_dense_sync = (self.training and sparse and self.use_dense_sparse_sync
                              and self.dense_sparse_sync_weight > 0)
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
                logical_condition = self._condition_for_logical_depth(
                    condition, offset,
                )
                x = self._recursive_block_for_offset(offset)(
                    x, logical_condition,
                )
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

        # This is the full state immediately before the existing TREAD split.
        # Keep it in the graph: RouteSync gradients must reach the prefix.
        h_pre = x if capture_routesync else None
        active_ratio = self._active_ratio_for_forward(x.device)
        active_tokens = int(round(tokens * active_ratio))
        active_tokens = max(1, min(tokens, active_tokens))
        if capture_routesync and active_tokens == tokens:
            raise ValueError("RouteSync needs at least one bypass token after ratio rounding")
        self.last_tread_active_ratio = active_ratio
        self.last_tread_active_fraction = active_tokens / tokens
        active_indices, bypass_indices = random_subset_indices(
            batch, tokens, active_tokens, x.device,
            generator=self._subset_generator(x.device),
        )
        dense_sync_features = None
        if capture_dense_sync:
            sync_indices = (active_indices if self.dense_sparse_sync_tokens == "active"
                            else bypass_indices if self.dense_sparse_sync_tokens == "routed"
                            else None)
            if sync_indices is not None and sync_indices.shape[1] == 0:
                raise ValueError("Dense sparse sync requires non-empty selected tokens; "
                                 "routed sync requires an active ratio below 1")
            count = max(1, int(batch * self.dense_sparse_sync_ratio))
            sample_indices = torch.randperm(batch, device=x.device)[:count]
            # A no-grad teacher must not cache detached autocast weight casts
            # that the subsequent student would reuse (silently losing grads).
            with torch.no_grad(), torch.autocast(
                device_type=x.device.type,
                enabled=torch.is_autocast_enabled(x.device.type),
                dtype=torch.get_autocast_dtype(x.device.type), cache_enabled=False,
            ):
                dense = x.detach().index_select(0, sample_indices)
                dense_condition = condition.detach().index_select(0, sample_indices)
                for block_index in range(self.tread_start_block, self.tread_end_block):
                    offset = block_index - self.tread_start_block
                    block = (self.recurrent_group_blocks[self._recursive_group_for_offset(offset)]
                             if self.tread_recursive else self.blocks[block_index])
                    dense = self._run_routed_block(
                        block, dense, self._condition_for_logical_depth(dense_condition, offset),
                        endpoint=block_index == self.tread_end_block - 1,
                    )
                # end is exclusive for routing. Align AFTER the first full-token
                # suffix block (zero-based end), not after routed block end-1.
                dense = suffix_blocks[0](dense, dense_condition)
                dense_target = (dense if sync_indices is None else gather_tokens(
                    dense, sync_indices.index_select(0, sample_indices)))
            dense_sync_features = {
                "teacher": dense_target.detach(),
                "sample_indices": sample_indices,
                "batch_size": batch,
                "target_block": self.tread_end_block,
            }

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
                    f"training ratio choices: {self.tread_active_ratios}\n"
                    f"eval mode: {self.tread_eval_mode}\n"
                    f"training FP32 endpoint: {self.tread_fp32_endpoint}\n"
                    f"recursive group blocks: {self.tread_recursive}\n"
                    f"recursive pattern: {self.tread_recursive_pattern}\n"
                    f"logical depth embedding: {self.tread_depth_embedding}"
                )

        for block_index in range(self.tread_start_block, self.tread_end_block):
            offset = block_index - self.tread_start_block
            block = (
                self._recursive_block_for_offset(offset)
                if self.tread_recursive else self.blocks[block_index]
            )
            active = self._run_routed_block(
                block, active,
                self._condition_for_logical_depth(condition, offset),
                endpoint=block_index == self.tread_end_block - 1,
            )
            if return_routing_info:
                input_counts.append(active.shape[1])
                output_counts.append(tokens)

        x = restore_token_order(torch.cat((active, bypass), dim=1), permutation)
        h_post = None
        target_block = self._routesync_target_for_forward(x.device) if capture_routesync else None
        self.last_routesync_target_block = target_block
        for suffix_offset, block in enumerate(suffix_blocks):
            x = block(x, condition)
            if dense_sync_features is not None and suffix_offset == 0:
                # Both branches use original spatial order after the first suffix block.
                student = x.index_select(0, sample_indices)
                dense_sync_features["student"] = (
                    student if sync_indices is None else gather_tokens(
                        student, sync_indices.index_select(0, sample_indices)))
            # Logical zero-based suffix index, also valid with recurrent routing.
            if capture_routesync and self.tread_end_block + suffix_offset == target_block:
                h_post = x
            if return_routing_info:
                input_counts.append(tokens)
                output_counts.append(tokens)

        output = self.unpatchify(self.final_layer(x, condition))
        route_features = None
        if capture_routesync:
            if h_post is None:
                raise RuntimeError(
                    "RouteSync could not capture the selected dense suffix block"
                )
            route_features = {
                "h_pre": h_pre,
                "h_post": h_post,
                "target_block": target_block,
                # TREAD calls P active and R bypass. Preserve the original
                # spatial indices rather than the temporary grouped ordering.
                "routed_idx": bypass_indices,
                "processed_idx": active_indices,
            }
        if dense_sync_features is not None:
            if route_features is None:
                route_features = {}
            route_features["dense_sparse_sync"] = dense_sync_features
        if not return_routing_info:
            return output, route_features
        return output, route_features, {
            "routing_applied": True,
            "active_ratio": active_ratio,
            "active_fraction": active_tokens / tokens,
            "active_indices": active_indices.detach(),
            "bypass_indices": bypass_indices.detach(),
            "block_input_token_counts": input_counts,
            "post_block_token_counts": output_counts,
            "fp32_endpoint_applied": bool(
                self.training and self.tread_fp32_endpoint
            ),
            "recursive": self.tread_recursive,
            "routesync_features": route_features,
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
