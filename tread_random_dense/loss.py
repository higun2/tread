"""Flow matching plus the training-only RouteSync R-P relational loss."""

import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .model import gather_tokens


def mean_flat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))


def _validate_partition(h_pre, h_post, routed_idx, processed_idx):
    if h_pre.ndim != 3 or h_post.shape != h_pre.shape:
        raise RuntimeError(
            f"RouteSync expects matching [B,N,D] states, got "
            f"{tuple(h_pre.shape)} and {tuple(h_post.shape)}"
        )
    batch, tokens, _ = h_pre.shape
    if routed_idx.ndim != 2 or processed_idx.ndim != 2:
        raise RuntimeError("RouteSync indices must have shape [B,K]")
    if routed_idx.shape[0] != batch or processed_idx.shape[0] != batch:
        raise RuntimeError("RouteSync index batch size does not match hidden states")
    if routed_idx.shape[1] + processed_idx.shape[1] != tokens:
        raise RuntimeError("RouteSync R and P sizes do not cover all tokens")
    if routed_idx.shape[1] == 0 or processed_idx.shape[1] == 0:
        raise RuntimeError("RouteSync requires non-empty R and P groups")
    union = torch.cat((routed_idx, processed_idx), dim=1).sort(dim=1).values
    expected = torch.arange(tokens, device=union.device).expand(batch, -1)
    if not torch.equal(union, expected):
        raise RuntimeError("RouteSync R and P indices must be disjoint and exhaustive")


def sample_group_indices(indices, ratio):
    """Independent per-image, fixed-size uniform sampling without replacement."""
    if ratio == 1.0:
        return indices
    count = max(1, int(indices.shape[1] * ratio))
    order = torch.rand(indices.shape, device=indices.device).argsort(dim=1)
    return indices.gather(1, order[:, :count])


def routesync_relational_loss(
    h_pre, h_post, routed_idx, processed_idx, *, sample_ratio=1.0, debug=False,
):
    """Match a random R/P submatrix to a stopped post-route target."""
    if not 0.0 < sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    if debug:
        _validate_partition(h_pre, h_post, routed_idx, processed_idx)

    # Sample before normalization/matmul; use identical original indices at
    # both depths. Ratio 1 takes the exact original path without RNG draws.
    routed_idx = sample_group_indices(routed_idx, sample_ratio)
    processed_idx = sample_group_indices(processed_idx, sample_ratio)
    pre_r = gather_tokens(h_pre, routed_idx)
    pre_p = gather_tokens(h_pre, processed_idx)
    # The post branch is always stop-gradient, including subset alignment.
    post_source = h_post.detach()
    post_r = gather_tokens(post_source, routed_idx)
    post_p = gather_tokens(post_source, processed_idx)

    pre_r = F.normalize(pre_r, p=2, dim=-1)
    pre_p = F.normalize(pre_p, p=2, dim=-1)
    post_r = F.normalize(post_r, p=2, dim=-1)
    post_p = F.normalize(post_p, p=2, dim=-1)
    relation_pre = torch.bmm(pre_r, pre_p.transpose(1, 2))
    relation_post = torch.bmm(post_r, post_p.transpose(1, 2))
    loss = F.l1_loss(relation_pre, relation_post)

    if debug:
        expected_shape = (
            h_pre.shape[0], routed_idx.shape[1], processed_idx.shape[1],
        )
        if relation_pre.shape != expected_shape or relation_post.shape != expected_shape:
            raise RuntimeError("RouteSync relation matrix shape is incorrect")
        if not relation_pre.requires_grad:
            raise RuntimeError("RouteSync pre relation is disconnected from autograd")
        if relation_post.requires_grad:
            raise RuntimeError("RouteSync post relation must be stop-gradient")
        tolerance = 2e-3
        for name, relation in (
            ("relation_pre", relation_pre), ("relation_post", relation_post),
        ):
            if not torch.isfinite(relation).all():
                raise RuntimeError(f"{name} contains non-finite values")
            if relation.detach().abs().max() > 1 + tolerance:
                raise RuntimeError(f"{name} contains values outside cosine range")

    stats = {
        "sampled_r_tokens": routed_idx.shape[1],
        "sampled_p_tokens": processed_idx.shape[1],
        "relation_pre_mean": relation_pre.detach().mean(),
        "relation_post_mean": relation_post.detach().mean(),
        "relation_abs_diff_mean": loss.detach(),
    }
    return loss, stats


def routesync_feature_cosine_loss(
    h_pre, h_post, routed_idx, processed_idx, *, sample_ratio=1.0, debug=False,
):
    """Negative mean cosine for corresponding spatial tokens, stopped late target.

    R and P are used only for stratified subset sampling, never as pairwise
    relations. Each selected token has equal weight, including unequal R/P sizes.
    """
    if not 0.0 < sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    if debug:
        _validate_partition(h_pre, h_post, routed_idx, processed_idx)
    routed_idx = sample_group_indices(routed_idx, sample_ratio)
    processed_idx = sample_group_indices(processed_idx, sample_ratio)
    indices = torch.cat((routed_idx, processed_idx), dim=1)
    pre = gather_tokens(h_pre, indices)
    post = gather_tokens(h_post.detach(), indices)
    # Reduce cosine in FP32 under mixed precision; retain float64 for checks.
    if pre.dtype in (torch.float16, torch.bfloat16):
        pre = pre.float()
    if post.dtype in (torch.float16, torch.bfloat16):
        post = post.float()
    cosine = F.cosine_similarity(pre, post, dim=-1, eps=1e-8)
    loss = -cosine.mean()
    if debug:
        if not pre.requires_grad or post.requires_grad:
            raise RuntimeError("Feature cosine must train pre features and stop post features")
        if not torch.isfinite(cosine).all():
            raise RuntimeError("Nonfinite feature cosine similarity")
    nr = routed_idx.shape[1]
    return loss, {
        "sampled_r_tokens": nr,
        "sampled_p_tokens": processed_idx.shape[1],
        "feature_cosine_mean": cosine.detach().mean(),
        "feature_cosine_r_mean": cosine[:, :nr].detach().mean(),
        "feature_cosine_p_mean": cosine[:, nr:].detach().mean(),
    }


def dense_sparse_cosine_loss(student, teacher):
    """Align corresponding active tokens; teacher never receives gradients."""
    if student.ndim != 3 or student.shape != teacher.shape:
        raise ValueError("Dense/sparse features must have matching [B,K,D] shapes")
    return -F.cosine_similarity(student.float(), teacher.detach().float(),
                                dim=-1, eps=1e-8).mean()


class FlowMatchingLoss:
    def __init__(
        self, prediction="v", path_type="linear", weighting="uniform", *,
        use_routesync=False, routesync_weight=0.0, routesync_debug=False,
        routesync_sample_ratio=1.0, routesync_loss_type="relational",
        use_dense_sparse_sync=False, dense_sparse_sync_weight=0.1,
    ):
        self.use_dense_sparse_sync = use_dense_sparse_sync
        self.dense_sparse_sync_weight = dense_sparse_sync_weight
        if not math.isfinite(dense_sparse_sync_weight) or dense_sparse_sync_weight < 0:
            raise ValueError("dense_sparse_sync_weight must be finite and nonnegative")
        self.prediction = prediction
        self.path_type = path_type
        self.weighting = weighting
        self.use_routesync = use_routesync
        self.routesync_weight = routesync_weight
        if routesync_loss_type not in ("relational", "feature-cosine"):
            raise ValueError("routesync_loss_type must be relational or feature-cosine")
        self.routesync_loss_type = routesync_loss_type
        self.routesync_sample_ratio = routesync_sample_ratio
        self.routesync_debug = routesync_debug
        self._debug_printed = False
        if not 0.0 < routesync_sample_ratio <= 1.0:
            raise ValueError("routesync_sample_ratio must be in (0, 1]")
        if routesync_weight < 0:
            raise ValueError("routesync_weight must be non-negative")

    def interpolant(self, t):
        if self.path_type == "linear":
            return 1 - t, t, -1, 1
        if self.path_type == "cosine":
            return (
                torch.cos(t * np.pi / 2), torch.sin(t * np.pi / 2),
                -np.pi / 2 * torch.sin(t * np.pi / 2),
                np.pi / 2 * torch.cos(t * np.pi / 2),
            )
        raise NotImplementedError(self.path_type)

    def sample_time(self, images):
        shape = (images.shape[0], 1, 1, 1)
        if self.weighting == "uniform":
            return torch.rand(shape, device=images.device, dtype=images.dtype)
        if self.weighting == "lognormal":
            sigma = torch.randn(shape, device=images.device, dtype=images.dtype).exp()
            if self.path_type == "linear":
                return sigma / (1 + sigma)
            return 2 / np.pi * torch.atan(sigma)
        raise NotImplementedError(self.weighting)

    def __call__(self, model, images, model_kwargs=None):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        t = self.sample_time(images)
        noise = torch.randn_like(images)
        alpha, sigma, d_alpha, d_sigma = self.interpolant(t)
        if self.prediction != "v":
            raise NotImplementedError("only velocity prediction is implemented")
        output, route_features = model(
            alpha * images + sigma * noise, t.flatten(), **model_kwargs,
        )
        target = d_alpha * images + d_sigma * noise
        per_sample = mean_flat((output.float() - target.float()).square())
        fm_loss = per_sample.mean()

        route_sync_loss = fm_loss.new_zeros(())
        stats = {
            "relation_pre_mean": fm_loss.new_zeros(()),
            "relation_post_mean": fm_loss.new_zeros(()),
            "relation_abs_diff_mean": fm_loss.new_zeros(()),
        }
        if self.use_routesync:
            if route_features is None:
                raise RuntimeError(
                    "RouteSync is enabled but the model returned no training features"
                )
            debug_now = self.routesync_debug and not self._debug_printed
            alignment = (
                routesync_relational_loss if self.routesync_loss_type == "relational"
                else routesync_feature_cosine_loss
            )
            route_sync_loss, stats = alignment(
                route_features["h_pre"], route_features["h_post"],
                route_features["routed_idx"], route_features["processed_idx"],
                sample_ratio=self.routesync_sample_ratio, debug=debug_now,
            )
            if debug_now:
                self._debug_printed = True
                if not dist.is_initialized() or dist.get_rank() == 0:
                    if self.routesync_loss_type == "feature-cosine":
                        print(
                            "[RouteSync Feature Cosine Loss]\n"
                            f"h_pre/h_post: {tuple(route_features['h_pre'].shape)}\n"
                            f"sampled tokens R/P: {stats['sampled_r_tokens']}/{stats['sampled_p_tokens']}\n"
                            f"sample ratio: {self.routesync_sample_ratio}; post target stop-gradient: True"
                        )
                    else:
                        print(
                            "[RouteSync R-P Relational Loss]\n"
                            f"h_pre/h_post: {tuple(route_features['h_pre'].shape)}\n"
                            f"R/P: {route_features['routed_idx'].shape[1]} / "
                            f"{route_features['processed_idx'].shape[1]}\n"
                            f"sampled relation: {(route_features['h_pre'].shape[0], stats['sampled_r_tokens'], stats['sampled_p_tokens'])}\n"
                            f"relation dtype: {stats['relation_pre_mean'].dtype}\n"
                            f"sample ratio: {self.routesync_sample_ratio}; post target stop-gradient: True"
                        )

        weighted = self.routesync_weight * route_sync_loss
        total = fm_loss if not self.use_routesync else fm_loss + weighted
        dense_loss = fm_loss.new_zeros(())
        dense_count = 0
        dense_fraction = 0.0
        if self.use_dense_sparse_sync and self.dense_sparse_sync_weight > 0:
            if route_features is None or "dense_sparse_sync" not in route_features:
                raise RuntimeError("Dense-sparse sync enabled but model returned no endpoint features")
            features = route_features["dense_sparse_sync"]
            dense_loss = dense_sparse_cosine_loss(features["student"], features["teacher"])
            dense_count = features["student"].shape[0]
            dense_fraction = dense_count / features["batch_size"]
            total = total + self.dense_sparse_sync_weight * dense_loss
        return {
            "dense_sparse_sync": dense_loss,
            "dense_sparse_sync_weighted": self.dense_sparse_sync_weight * dense_loss,
            "dense_sparse_sync_samples": dense_count,
            "dense_sparse_sync_fraction": dense_fraction,
            "total": total,
            "fm": fm_loss,
            "fm_per_sample": per_sample,
            "route_sync": route_sync_loss,
            "route_sync_weighted": weighted,
            **stats,
        }
