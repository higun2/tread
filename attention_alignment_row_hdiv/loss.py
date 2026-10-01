"""Attention alignment with selectable CKA or relational-L1 H-diversity."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from attention_alignment_hdiv.loss import (
    SILoss as CKASILoss,
    hdiv_margin_loss,
    linear_cka_per_sample,
)


def _validate_hidden_pair(h_shallow: torch.Tensor, h_deep: torch.Tensor) -> None:
    if h_shallow.ndim != 3 or h_deep.ndim != 3:
        raise ValueError(
            "row-L1 H-div expects [B, T, D], got "
            f"{tuple(h_shallow.shape)} and {tuple(h_deep.shape)}"
        )
    if h_shallow.shape != h_deep.shape:
        raise ValueError(
            "shallow/deep hidden shapes must match, got "
            f"{tuple(h_shallow.shape)} and {tuple(h_deep.shape)}"
        )
    if h_shallow.shape[1] <= 1:
        raise ValueError(f"row-L1 H-div requires T > 1, got T={h_shallow.shape[1]}")


def cosine_token_relation_matrix(
    hidden: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a per-image cosine token-relation matrix in FP32.

    This is the only L2 normalization in row-L1 mode. Relation rows are not
    normalized again.
    """
    if hidden.ndim != 3:
        raise ValueError(f"expected hidden [B, T, D], got {tuple(hidden.shape)}")
    if eps <= 0:
        raise ValueError(f"hdiv_eps must be positive, got {eps}")
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        normalized = F.normalize(hidden.float(), p=2, dim=-1, eps=eps)
        return normalized @ normalized.transpose(-1, -2)  # [B,T,T]


def row_relational_l1_distance(
    h_shallow: torch.Tensor,
    h_deep: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean off-diagonal L1 distance between token-relation rows, `[B,T]`.

    The deep reference is detached only for this auxiliary objective. All work
    is explicitly FP32 under AMP.
    """
    _validate_hidden_pair(h_shallow, h_deep)
    with torch.autocast(device_type=h_shallow.device.type, enabled=False):
        relation_shallow = cosine_token_relation_matrix(h_shallow, eps=eps)
        relation_deep = cosine_token_relation_matrix(h_deep.detach(), eps=eps)
        relation_diff = (relation_shallow - relation_deep).abs()

        tokens = h_shallow.shape[1]
        off_diagonal = 1.0 - torch.eye(
            tokens, device=h_shallow.device, dtype=relation_diff.dtype
        )
        relation_diff = relation_diff * off_diagonal.unsqueeze(0)
        return relation_diff.sum(dim=-1) / max(tokens - 1, 1)


def row_relational_l1_diversity_loss(
    h_shallow: torch.Tensor,
    h_deep: torch.Tensor,
    margin: float = 0.1,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Apply the per-token minimum-distance hinge `relu(margin-row_dist)`."""
    if not 0.0 <= margin <= 2.0:
        raise ValueError(f"row_hdiv_margin must be in [0, 2], got {margin}")
    row_distance = row_relational_l1_distance(h_shallow, h_deep, eps=eps)
    loss = F.relu(margin - row_distance)
    detached = row_distance.detach()
    return {
        "loss": loss,
        "row_distance": detached,
        "active": (detached < margin).to(torch.float32),
    }


class SILoss(CKASILoss):
    """Existing FM/attention/CKA loss plus independent row-L1 ablation."""

    MODES = ("none", "cka", "row_l1")

    def __init__(
        self,
        *args,
        hdiv_mode: str = "none",
        hdiv_cka_threshold: float = 0.8,
        row_hdiv_margin: float = 0.1,
        hdiv_eps: float = 1e-6,
        hdiv_detach_deep: bool = True,
        row_hdiv_log_quantiles: bool = False,
        **kwargs,
    ):
        if hdiv_mode not in self.MODES:
            raise ValueError(f"hdiv_mode must be one of {self.MODES}, got {hdiv_mode!r}")
        if not 0.0 <= row_hdiv_margin <= 2.0:
            raise ValueError("row_hdiv_margin must be in [0, 2]")
        self.hdiv_mode = hdiv_mode
        self.row_hdiv_margin = float(row_hdiv_margin)
        self.row_hdiv_log_quantiles = bool(row_hdiv_log_quantiles)
        # CKA mode calls the existing implementation without changing it.
        super().__init__(
            *args,
            enable_hdiv=hdiv_mode == "cka",
            hdiv_cka_threshold=hdiv_cka_threshold,
            hdiv_eps=hdiv_eps,
            hdiv_detach_deep=hdiv_detach_deep,
            **kwargs,
        )

    @staticmethod
    def _row_zero_stats(reference: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = reference.new_zeros((), dtype=torch.float32)
        return {
            "row_dist_mean": zero,
            "row_dist_std": zero,
            "row_dist_min": zero,
            "row_dist_max": zero,
            "row_active_ratio": zero,
            "row_dist_sum": zero,
            "row_dist_sq_sum": zero,
            "row_active_sum": zero,
            "row_dist_count": zero,
            "row_dist_values": None,
        }

    def __call__(self, *args, hidden_states_dict=None, **kwargs):
        denoising_loss, attn_loss, hdiv = super().__call__(
            *args, hidden_states_dict=hidden_states_dict, **kwargs
        )
        hdiv.update(self._row_zero_stats(denoising_loss))
        if self.hdiv_mode != "row_l1":
            return denoising_loss, attn_loss, hdiv

        if (
            hidden_states_dict is None
            or "shallow" not in hidden_states_dict
            or "deep" not in hidden_states_dict
        ):
            raise RuntimeError("row-L1 H-div is enabled but selected block outputs were not supplied")

        shallow_objects = self._as_list(hidden_states_dict["shallow"])
        deep_hidden = self._extract(hidden_states_dict["deep"], "saved_hidden")
        per_source = []
        for shallow_object in shallow_objects:
            shallow_hidden = self._extract(shallow_object, "saved_hidden")
            if shallow_hidden is None or deep_hidden is None:
                raise RuntimeError("a row-L1 H-div block-output hook did not capture a tensor")
            per_source.append(
                row_relational_l1_diversity_loss(
                    shallow_hidden,
                    deep_hidden,
                    margin=self.row_hdiv_margin,
                    eps=self.hdiv_eps,
                )
            )
        if not per_source:
            raise RuntimeError("row-L1 H-div requires at least one shallow block")

        # [sources,B,T] -> average sources for the optimized per-token loss.
        per_token_loss = torch.stack([item["loss"] for item in per_source]).mean(dim=0)
        all_distances = torch.cat(
            [item["row_distance"].reshape(-1) for item in per_source]
        ).float()
        all_active = torch.cat([item["active"].reshape(-1) for item in per_source]).float()
        count = all_distances.new_tensor(all_distances.numel())
        row_stats = {
            "row_dist_mean": all_distances.mean(),
            "row_dist_std": all_distances.std(unbiased=False),
            "row_dist_min": all_distances.min(),
            "row_dist_max": all_distances.max(),
            "row_active_ratio": all_active.mean(),
            # Sufficient statistics allow exact global DDP mean/std later.
            "row_dist_sum": all_distances.sum(),
            "row_dist_sq_sum": all_distances.square().sum(),
            "row_active_sum": all_active.sum(),
            "row_dist_count": count,
            "row_dist_values": all_distances if self.row_hdiv_log_quantiles else None,
        }
        hdiv = {
            "loss": per_token_loss,
            # CKA-only diagnostics are zero in the independent row-L1 ablation.
            "cka_mean": denoising_loss.new_zeros((), dtype=torch.float32),
            "distance_mean": denoising_loss.new_zeros((), dtype=torch.float32),
            "active_ratio": denoising_loss.new_zeros((), dtype=torch.float32),
            **row_stats,
        }
        return denoising_loss, attn_loss, hdiv


__all__ = [
    "SILoss",
    "cosine_token_relation_matrix",
    "hdiv_margin_loss",
    "linear_cka_per_sample",
    "row_relational_l1_distance",
    "row_relational_l1_diversity_loss",
]
