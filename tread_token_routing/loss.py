"""Original SiT flow-matching objective without a routing auxiliary loss."""

import numpy as np
import torch


def mean_flat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))


class FlowMatchingLoss:
    def __init__(self, prediction="v", path_type="linear", weighting="uniform"):
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
        output, _ = model(alpha * images + sigma * noise, t.flatten(), **model_kwargs)
        target = d_alpha * images + d_sigma * noise
        per_sample = mean_flat((output.float() - target.float()).square())
        return {"total": per_sample.mean(), "fm": per_sample.mean(), "fm_per_sample": per_sample}
