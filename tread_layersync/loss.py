"""Flow matching plus LayerSync between selectable TREAD blocks.

For each (student, teacher) block pair, L = -mean_i cos(h_s[i], sg(h_t[i])) over
corresponding spatial tokens (LayerSync, tokenwise stopped deeper target). Only
tokens present in both blocks are compared: if either block lies inside the
routed interval, the active tokens; otherwise all tokens (or the ``token_set``
subset). The pair losses are averaged and added with ``weight``.
"""

import numpy as np
import torch
import torch.nn.functional as F

from tread_token_routing.model import gather_tokens

TOKEN_SETS = ("all", "active", "routed")


def mean_flat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))


def align_pair(student, teacher, token_set, routing):
    """Return [B,K,D] student/teacher features over common original positions."""
    (hs, idx_s), (ht, idx_t) = student, teacher
    if idx_s is not None and idx_t is not None:
        if not torch.equal(idx_s, idx_t):
            raise RuntimeError("routed blocks must hold the same active subset")
        return hs, ht, "active"
    if idx_s is not None:
        return hs, gather_tokens(ht, idx_s), "active"
    if idx_t is not None:
        return gather_tokens(hs, idx_t), ht, "active"
    if token_set == "all" or routing is None:
        return hs, ht, "all"
    idx = routing["active_indices"] if token_set == "active" else routing["bypass_indices"]
    return gather_tokens(hs, idx), gather_tokens(ht, idx), token_set


def layersync_loss(features, pairs, token_set="all", routing=None):
    losses, stats = [], {}
    for student, teacher in pairs:
        hs, ht, used = align_pair(features[student], features[teacher], token_set, routing)
        cosine = F.cosine_similarity(hs.float(), ht.detach().float(), dim=-1, eps=1e-8)
        losses.append(-cosine.mean())
        stats[f"layersync/cos_{student}_{teacher}"] = cosine.detach().mean()
        stats[f"layersync/tokens_{student}_{teacher}"] = used
    return torch.stack(losses).mean(), stats


class FlowMatchingLayerSyncLoss:
    def __init__(self, prediction="v", path_type="linear", weighting="uniform",
                 pairs=(), weight=0.0, token_set="all"):
        if prediction != "v":
            raise NotImplementedError("only velocity prediction is implemented")
        if token_set not in TOKEN_SETS:
            raise ValueError(f"token_set must be one of {TOKEN_SETS}")
        self.path_type, self.weighting = path_type, weighting
        self.pairs, self.weight, self.token_set = tuple(pairs), float(weight), token_set
        self.capture = sorted({b for pair in self.pairs for b in pair})

    def interpolant(self, t):
        if self.path_type == "linear":
            return 1 - t, t, -1, 1
        if self.path_type == "cosine":
            return (torch.cos(t * np.pi / 2), torch.sin(t * np.pi / 2),
                    -np.pi / 2 * torch.sin(t * np.pi / 2), np.pi / 2 * torch.cos(t * np.pi / 2))
        raise NotImplementedError(self.path_type)

    def sample_time(self, images):
        shape = (images.shape[0], 1, 1, 1)
        if self.weighting == "uniform":
            return torch.rand(shape, device=images.device, dtype=images.dtype)
        if self.weighting == "lognormal":
            sigma = torch.randn(shape, device=images.device, dtype=images.dtype).exp()
            return sigma / (1 + sigma) if self.path_type == "linear" else 2 / np.pi * torch.atan(sigma)
        raise NotImplementedError(self.weighting)

    def __call__(self, model, images, model_kwargs=None):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        t = self.sample_time(images)
        noise = torch.randn_like(images)
        alpha, sigma, d_alpha, d_sigma = self.interpolant(t)
        xt, target = alpha * images + sigma * noise, d_alpha * images + d_sigma * noise
        active = self.weight > 0 and self.pairs
        if not active:
            output = model(xt, t.flatten(), **model_kwargs)[0]
            fm = mean_flat((output.float() - target.float()).square())
            return {"total": fm.mean(), "fm": fm.mean(), "fm_per_sample": fm}
        routing = None
        if self.token_set != "all":
            output, _, features, routing = model(xt, t.flatten(), capture_blocks=self.capture,
                                                 return_routing=True, **model_kwargs)
        else:
            output, _, features = model(xt, t.flatten(), capture_blocks=self.capture, **model_kwargs)
        fm = mean_flat((output.float() - target.float()).square())
        sync, stats = layersync_loss(features, self.pairs, self.token_set, routing)
        result = {"total": fm.mean() + self.weight * sync, "fm": fm.mean(), "fm_per_sample": fm,
                  "layersync": sync}
        result.update(stats)
        return result
