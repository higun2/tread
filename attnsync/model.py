"""SiT with optional TREAD routing and training-only attention sync (AttnSync).

Attention sync also runs without TREAD: then the training forward is the plain dense
SiT forward plus attention capture at the student and teacher blocks."""

from functools import partial
import math

import torch
import torch.distributed as dist

from models.sit_b1 import SiT as VanillaSiT, modulate


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


def attention_log_probs(block, hidden, condition):
    """Recompute a block's attention log-probabilities [B,H,K,K] from its own Q/K.

    Mirrors SiTBlock/Attention (adaLN shift/scale, qkv, q/k norm, scale); softmax runs in FP32.
    """
    shift, scale = block.adaLN_modulation(condition).chunk(6, dim=-1)[:2]
    x = modulate(block.norm1(hidden), shift, scale)
    attn = block.attn
    batch, count, _ = x.shape
    q, k, _ = attn.qkv(x).reshape(
        batch, count, 3, attn.num_heads, attn.head_dim,
    ).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    logits = (q.float() * attn.scale) @ k.float().transpose(-2, -1)
    return logits.log_softmax(dim=-1)


class SiT(VanillaSiT):
    def __init__(
        self, *, depth=28, use_tread_routing=False,
        tread_start_block=3, tread_end_block=9, tread_active_ratio=0.5,
        tread_eval_mode="sparse", tread_fp32_endpoint=True, tread_seed=None,
        use_attn_sync=False, attn_sync_student_blocks=None, attn_sync_teacher_block=None,
        attn_sync_heads="mean", attn_sync_loss="js", attn_sync_weight=0.1,
        attn_sync_ratio=1.0, attn_sync_mask_ratio=0.0, attn_sync_mask_unit="element",
        **kwargs,
    ):
        super().__init__(depth=depth, **kwargs)
        self.base_depth = depth
        self.use_tread_routing = use_tread_routing
        self.tread_start_block = tread_start_block
        self.tread_end_block = tread_end_block
        self.tread_active_ratio = tread_active_ratio
        self.tread_eval_mode = tread_eval_mode
        self.tread_fp32_endpoint = tread_fp32_endpoint
        self.tread_seed = tread_seed
        if tread_eval_mode not in ("sparse", "dense"):
            raise ValueError("tread_eval_mode must be 'sparse' or 'dense'")
        if not 0 < tread_active_ratio <= 1:
            raise ValueError("tread_active_ratio must be in (0, 1]")
        if use_tread_routing and not 0 <= tread_start_block < tread_end_block <= depth:
            raise ValueError(
                f"route range must satisfy 0 <= start < end <= depth ({depth}), "
                f"got [{tread_start_block}, {tread_end_block})"
            )

        self.use_attn_sync = use_attn_sync
        self.attn_sync_student_blocks = (
            () if attn_sync_student_blocks is None else tuple(attn_sync_student_blocks))
        self.attn_sync_teacher_block = attn_sync_teacher_block
        self.attn_sync_heads = attn_sync_heads
        self.attn_sync_loss = attn_sync_loss
        self.attn_sync_weight = attn_sync_weight
        self.attn_sync_ratio = attn_sync_ratio
        if attn_sync_heads not in ("mean", "per-head"):
            raise ValueError("attn_sync_heads must be mean or per-head")
        if attn_sync_loss not in ("l1", "js", "kl"):
            raise ValueError("attn_sync_loss must be l1, js, or kl")
        if not math.isfinite(attn_sync_weight) or attn_sync_weight < 0:
            raise ValueError("attn_sync_weight must be finite and nonnegative")
        if not 0 < attn_sync_ratio <= 1:
            raise ValueError("attn_sync_ratio must be in (0, 1]")
        if use_attn_sync:
            students = self.attn_sync_student_blocks
            if not students or attn_sync_teacher_block is None:
                raise ValueError("--use-attn-sync requires --attn-sync-student-blocks and --attn-sync-teacher-block")
            # With TREAD, rows only correspond inside the route (same active subset), so both
            # blocks must be routed. Without TREAD every block sees all tokens in the same order.
            allowed = range(tread_start_block, tread_end_block) if use_tread_routing else range(depth)
            if any(type(b) is not int or b not in allowed for b in (*students, attn_sync_teacher_block)):
                raise ValueError(
                    f"attention sync blocks must be zero-based "
                    f"{'routed ' if use_tread_routing else ''}blocks in [{allowed.start}, {allowed.stop})")
            if len(set(students)) != len(students) or attn_sync_teacher_block in students:
                raise ValueError("attention sync student blocks must be distinct and exclude the teacher")

    def _sparse_for_forward(self, eval_mode):
        if not self.use_tread_routing:
            return False
        if self.training:
            return True
        mode = self.tread_eval_mode if eval_mode is None else eval_mode
        if mode not in ("sparse", "dense"):
            raise ValueError("tread_eval_mode must be 'sparse' or 'dense'")
        return mode == "sparse"

    def _active_ratio_for_forward(self, override=None):
        if override is None:
            return self.tread_active_ratio
        if self.training:
            raise ValueError("tread_active_ratio override is inference-only")
        if not 0 < override <= 1:
            raise ValueError("tread_active_ratio override must be in (0, 1]")
        return override

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

    def _new_attn_sync_features(self, batch, device):
        attn_count = max(1, int(batch * self.attn_sync_ratio))
        samples = (torch.randperm(batch, device=device)[:attn_count]
                   if attn_count < batch else None)
        return {"students": {}, "teacher": None,
                "teacher_block": self.attn_sync_teacher_block}, samples

    def _capture_attn_sync(self, features, block, block_index, hidden, condition, samples):
        """Store the block's attention log-probs: teacher detached, students with grad."""
        if not (block_index == self.attn_sync_teacher_block
                or block_index in self.attn_sync_student_blocks):
            return
        attn_input = hidden if samples is None else hidden.index_select(0, samples)
        attn_condition = condition if samples is None else condition.index_select(0, samples)
        teacher = block_index == self.attn_sync_teacher_block
        # A no-grad teacher must not cache detached autocast weight casts
        # that this block's own forward would then reuse (losing grads).
        with torch.set_grad_enabled(torch.is_grad_enabled() and not teacher), torch.autocast(
            device_type=hidden.device.type,
            enabled=torch.is_autocast_enabled(hidden.device.type),
            dtype=torch.get_autocast_dtype(hidden.device.type), cache_enabled=False,
        ):
            log_probs = attention_log_probs(block, attn_input, attn_condition)
        if teacher:
            features["teacher"] = log_probs.detach()
        else:
            features["students"][block_index] = log_probs

    def _dense_attn_sync_forward(self, x, t, y, force_drop_ids, return_routing_info):
        """Plain SiT forward (as VanillaSiT.forward) that also captures attention sync maps."""
        x = self.x_embedder(x) + self.pos_embed
        batch, tokens, _ = x.shape
        condition = self.t_embedder(t) + self.y_embedder(
            y, self.training, force_drop_ids=force_drop_ids,
        )
        features, samples = self._new_attn_sync_features(batch, x.device)
        for block_index, block in enumerate(self.blocks):
            self._capture_attn_sync(features, block, block_index, x, condition, samples)
            x = block(x, condition)
        output = self.unpatchify(self.final_layer(x, condition))
        extras = {"attn_sync": features}
        if not return_routing_info:
            return output, extras
        return output, extras, {
            "routing_applied": False,
            "active_indices": None,
            "bypass_indices": None,
            "block_input_token_counts": [tokens] * self.base_depth,
            "post_block_token_counts": [tokens] * self.base_depth,
            "fp32_endpoint_applied": False,
        }

    def forward(
        self, x, t, y, return_logvar=False, force_drop_ids=None,
        tread_eval_mode=None, return_routing_info=False, tread_active_ratio=None,
    ):
        """Returns (output, extras[, routing_info]); extras holds the attention sync maps
        during training ({"attn_sync": ...}) and is None otherwise.

        tread_active_ratio: inference-only kept-token fraction for this call
        (Sparse Guidance runs one checkpoint at two sparsity levels).
        """
        sparse = self._sparse_for_forward(tread_eval_mode)
        if tread_active_ratio is not None and not sparse:
            raise ValueError("tread_active_ratio override requires the sparse TREAD path")
        capture_attn_sync = self.training and self.use_attn_sync and self.attn_sync_weight > 0
        if capture_attn_sync and not self.use_tread_routing:
            return self._dense_attn_sync_forward(x, t, y, force_drop_ids, return_routing_info)
        if not sparse:
            output = super().forward(
                x, t, y, return_logvar=return_logvar, force_drop_ids=force_drop_ids,
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
            }

        x = self.x_embedder(x) + self.pos_embed
        batch, tokens, _ = x.shape
        condition = self.t_embedder(t) + self.y_embedder(
            y, self.training, force_drop_ids=force_drop_ids,
        )
        for block in self.blocks[:self.tread_start_block]:
            x = block(x, condition)

        active_ratio = self._active_ratio_for_forward(tread_active_ratio)
        active_tokens = max(1, min(tokens, int(round(tokens * active_ratio))))
        active_indices, bypass_indices = random_subset_indices(
            batch, tokens, active_tokens, x.device,
            generator=self._subset_generator(x.device),
        )

        attn_sync_features = None
        if capture_attn_sync:
            attn_sync_features, attn_samples = self._new_attn_sync_features(batch, x.device)

        permutation = torch.cat((active_indices, bypass_indices), dim=1)
        grouped = gather_tokens(x, permutation)
        active = grouped[:, :active_tokens]
        bypass = grouped[:, active_tokens:]
        for block_index in range(self.tread_start_block, self.tread_end_block):
            block = self.blocks[block_index]
            if attn_sync_features is not None:
                # Same active subset at every routed block: query/key rows correspond exactly.
                self._capture_attn_sync(attn_sync_features, block, block_index,
                                        active, condition, attn_samples)
            active = self._run_routed_block(
                block, active, condition,
                endpoint=block_index == self.tread_end_block - 1,
            )

        x = restore_token_order(torch.cat((active, bypass), dim=1), permutation)
        for block in self.blocks[self.tread_end_block:]:
            x = block(x, condition)
        output = self.unpatchify(self.final_layer(x, condition))
        extras = None if attn_sync_features is None else {"attn_sync": attn_sync_features}
        if not return_routing_info:
            return output, extras
        routed = self.tread_end_block - self.tread_start_block
        suffix = self.base_depth - self.tread_end_block
        return output, extras, {
            "routing_applied": True,
            "active_ratio": active_ratio,
            "active_fraction": active_tokens / tokens,
            "active_indices": active_indices.detach(),
            "bypass_indices": bypass_indices.detach(),
            "block_input_token_counts": ([tokens] * self.tread_start_block
                                         + [active_tokens] * routed + [tokens] * suffix),
            "post_block_token_counts": [tokens] * self.base_depth,
            "fp32_endpoint_applied": bool(self.training and self.tread_fp32_endpoint),
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
