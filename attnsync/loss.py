"""Flow matching plus training-only routed attention sync (AttnSync)."""

import math
import numpy as np
import torch


def mean_flat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))


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
        self.prediction = prediction
        self.path_type = path_type
        self.weighting = weighting

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
        output, extras = model(
            alpha * images + sigma * noise, t.flatten(), **model_kwargs,
        )
        target = d_alpha * images + d_sigma * noise
        per_sample = mean_flat((output.float() - target.float()).square())
        fm_loss = per_sample.mean()

        total = fm_loss
        attn_loss = fm_loss.new_zeros(())
        attn_per_block = {}
        if self.use_attn_sync and self.attn_sync_weight > 0:
            if extras is None or "attn_sync" not in extras:
                raise RuntimeError("Attention sync enabled but model returned no attention maps")
            features = extras["attn_sync"]
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
            "total": total,
            "fm": fm_loss,
            "fm_per_sample": per_sample,
        }
