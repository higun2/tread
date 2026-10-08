"""Flow matching plus training-only routed attention sync and DensePush."""

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


def dense_push_loss(sparse, dense, margin=0.95, eps=1e-8):
    """Margin repulsion [cos(z_D, z_S) - tau]_+^2 on mean-pooled, L2-normalized paths.

    Returns the loss averaged over samples and per-sample cosines (detached).
    The dense side is only detached by the model (``dense_push_grad="sparse"``).
    """
    if sparse.ndim != 3 or sparse.shape != dense.shape:
        raise ValueError("Dense/sparse features must have matching [B,K,D] shapes")
    if not math.isfinite(margin) or not -1 <= margin < 1:
        raise ValueError("margin must be in [-1, 1)")
    # Pool and normalize in FP32 even with BF16 training; keep FP64 for checks.
    dtype = torch.float64 if sparse.dtype == torch.float64 or dense.dtype == torch.float64 else torch.float32
    z_sparse = F.normalize(sparse.to(dtype).mean(dim=1), dim=-1, eps=eps)
    z_dense = F.normalize(dense.to(dtype).mean(dim=1), dim=-1, eps=eps)
    cosine = (z_sparse * z_dense).sum(dim=-1)
    loss = F.relu(cosine - margin).square().mean()
    return loss, cosine.detach()


def attention_sync_loss(student, teacher, heads="mean", kind="js",
                        mask_ratio=0.0, mask_unit="element"):
    """Divergence between attention rows, teacher stop-gradient.

    ``student``/``teacher`` are log-probabilities [B,H,K,K] over the last dim.
    heads="mean" averages probabilities over heads before comparing rows;
    "per-head" compares head h with head h. Rows (queries) are weighted equally.
    l1 is sum_j |S - T| per row in [0, 2]; kl is KL(T || S); js is
    Jensen-Shannon in nats, in [0, log 2].

    ``mask_ratio`` randomly excludes that fraction of the per-entry divergence
    terms, resampled every call: ``element`` drops single (query, key) entries,
    ``query`` drops whole rows, ``key`` drops whole columns. The masked sum is
    divided by the realized kept fraction, so its scale matches the unmasked loss.
    """
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in [0, 1)")
    if mask_unit not in ("element", "query", "key"):
        raise ValueError("mask_unit must be element, query, or key")
    if student.ndim != 4 or student.shape != teacher.shape:
        raise ValueError("attention maps must have matching [B,H,K,K] shapes")
    if heads not in ("mean", "per-head"):
        raise ValueError("heads must be mean or per-head")
    log_s = student.float() if student.dtype in (torch.float16, torch.bfloat16) else student
    log_t = teacher.detach().to(log_s.dtype)
    if heads == "mean":
        # log of the head-averaged distribution, exact in log space.
        log_heads = math.log(log_s.shape[1])
        log_s = log_s.logsumexp(dim=1) - log_heads
        log_t = log_t.logsumexp(dim=1) - log_heads
    p_s, p_t = log_s.exp(), log_t.exp()
    if kind == "l1":
        terms = (p_s - p_t).abs()
    elif kind == "kl":
        # Generalized KL: per-entry nonnegative, same row sum as KL(T || S).
        terms = p_t * (log_t - log_s) - p_t + p_s
    elif kind == "js":
        log_m = torch.logaddexp(log_s, log_t) - math.log(2.0)
        terms = 0.5 * (p_s * (log_s - log_m) + p_t * (log_t - log_m))
    else:
        raise ValueError("kind must be l1, js, or kl")
    if mask_ratio == 0.0:
        return terms.sum(dim=-1).mean()
    shape = list(terms.shape)
    if mask_unit == "query":
        shape[-1] = 1
    elif mask_unit == "key":
        shape[-2] = 1
    keep = (torch.rand(shape, device=terms.device) >= mask_ratio).to(terms.dtype)
    kept_fraction = keep.mean().clamp_min(1.0 / keep.numel())
    return (terms * keep).sum(dim=-1).mean() / kept_fraction


class FlowMatchingLoss:
    def __init__(
        self, prediction="v", path_type="linear", weighting="uniform", *,
        use_routesync=False, routesync_weight=0.0, routesync_debug=False,
        routesync_sample_ratio=1.0, routesync_loss_type="relational",
        use_dense_push=False, dense_push_weight=0.1, dense_push_margin=0.95,
        use_attn_sync=False, attn_sync_weight=0.1, attn_sync_heads="mean",
        attn_sync_loss="js", attn_sync_mask_ratio=0.0, attn_sync_mask_unit="element",
        compile=False,
    ):
        # The divergence runs on large [B,H,K,K] FP32 maps: fusing it saves memory traffic.
        self._attention_sync_loss = (torch.compile(attention_sync_loss) if compile
                                     else attention_sync_loss)
        if not 0.0 <= attn_sync_mask_ratio < 1.0:
            raise ValueError("attn_sync_mask_ratio must be in [0, 1)")
        self.attn_sync_mask_ratio = attn_sync_mask_ratio
        self.attn_sync_mask_unit = attn_sync_mask_unit
        self.use_attn_sync = use_attn_sync
        self.attn_sync_weight = attn_sync_weight
        self.attn_sync_heads = attn_sync_heads
        self.attn_sync_loss = attn_sync_loss
        if not math.isfinite(attn_sync_weight) or attn_sync_weight < 0:
            raise ValueError("attn_sync_weight must be finite and nonnegative")
        if attn_sync_heads not in ("mean", "per-head"):
            raise ValueError("attn_sync_heads must be mean or per-head")
        if attn_sync_loss not in ("l1", "js", "kl"):
            raise ValueError("attn_sync_loss must be l1, js, or kl")
        self.use_dense_push = use_dense_push
        self.dense_push_weight = dense_push_weight
        self.dense_push_margin = dense_push_margin
        if not math.isfinite(dense_push_weight) or dense_push_weight < 0:
            raise ValueError("dense_push_weight must be finite and nonnegative")
        if not math.isfinite(dense_push_margin) or not -1 <= dense_push_margin < 1:
            raise ValueError("dense_push_margin must be in [-1, 1)")
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
        push_loss = fm_loss.new_zeros(())
        push_cosine = fm_loss.new_zeros(())
        push_active = fm_loss.new_zeros(())
        push_count = 0
        push_fraction = 0.0
        if self.use_dense_push and self.dense_push_weight > 0:
            if route_features is None or "dense_push" not in route_features:
                raise RuntimeError("DensePush enabled but model returned no path features")
            features = route_features["dense_push"]
            push_loss, cosine = dense_push_loss(
                features["sparse"], features["dense"], self.dense_push_margin)
            push_cosine = cosine.mean()
            push_active = (cosine > self.dense_push_margin).float().mean()
            push_count = cosine.shape[0]
            push_fraction = push_count / features["batch_size"]
            total = total + self.dense_push_weight * push_loss
        attn_loss = fm_loss.new_zeros(())
        attn_per_block = {}
        if self.use_attn_sync and self.attn_sync_weight > 0:
            if route_features is None or "attn_sync" not in route_features:
                raise RuntimeError("Attention sync enabled but model returned no attention maps")
            features = route_features["attn_sync"]
            if features["teacher"] is None or not features["students"]:
                raise RuntimeError("Attention sync is missing teacher or student maps")
            for block, student in features["students"].items():
                attn_per_block[block] = self._attention_sync_loss(
                    student, features["teacher"], self.attn_sync_heads, self.attn_sync_loss,
                    self.attn_sync_mask_ratio, self.attn_sync_mask_unit)
            # Mean over student blocks: the weight does not scale with their count.
            attn_loss = torch.stack(list(attn_per_block.values())).mean()
            total = total + self.attn_sync_weight * attn_loss
        return {
            "attn_sync": attn_loss,
            "attn_sync_weighted": self.attn_sync_weight * attn_loss,
            "attn_sync_per_block": {k: v.detach() for k, v in attn_per_block.items()},
            "dense_push": push_loss,
            "dense_push_weighted": self.dense_push_weight * push_loss,
            "dense_push_cosine": push_cosine,
            "dense_push_active_fraction": push_active,
            "dense_push_samples": push_count,
            "dense_push_fraction": push_fraction,
            "total": total,
            "fm": fm_loss,
            "fm_per_sample": per_sample,
            "route_sync": route_sync_loss,
            "route_sync_weighted": weighted,
            **stats,
        }
