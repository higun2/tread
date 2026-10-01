import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import smooth_l1_loss

# simple loss function
class Simpleloss(nn.Module):
    def __init__(self):
        super(Simpleloss, self).__init__()

    def forward(self, a, b, loss_type="sml1"):
        if loss_type == "sml1":
            align_loss = smooth_l1_loss(a, b, beta=0.05)
        elif loss_type == "l2":
            align_loss = F.mse_loss(a, b)
        elif loss_type == "l1":
            align_loss = F.l1_loss(a, b)
        else:
            raise NotImplementedError()
        return align_loss


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            loss_type="sml1",
            accelerator=None,
            latents_scale=None,
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.loss_type = loss_type
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

        self.criterion = Simpleloss()

    def _get_dropout_prob(self, model):
        model_ref = model.module if hasattr(model, "module") else model
        if hasattr(model_ref, "y_embedder") and hasattr(model_ref.y_embedder, "dropout_prob"):
            return float(model_ref.y_embedder.dropout_prob)
        return 0.0


    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, class_labels=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        labels = model_kwargs.get("y", None)
        if labels is None:
            raise ValueError("SILoss requires model_kwargs['y'] in this training setup.")

        # Use an explicit drop mask so we can reuse the dropped samples for follow-up alignment.
        time_flat = time_input.flatten()
        drop_prob = self._get_dropout_prob(model)
        drop_mask = (torch.rand(labels.shape[0], device=labels.device) < drop_prob)
        force_drop_ids = drop_mask.to(dtype=labels.dtype)
        drop_idx = torch.nonzero(drop_mask, as_tuple=False).squeeze(1)
        has_dropped = drop_idx.numel() > 0

        if has_dropped:
            subset_input = model_input[drop_idx]
            subset_time = time_flat[drop_idx]
            subset_labels = labels[drop_idx]

            aug_input = torch.cat([model_input, subset_input], dim=0)
            aug_time = torch.cat([time_flat, subset_time], dim=0)
            aug_labels = torch.cat([labels, subset_labels], dim=0)
            aug_force_drop_ids = torch.cat([force_drop_ids, torch.zeros_like(subset_labels)], dim=0)
        else:
            aug_input = model_input
            aug_time = time_flat
            aug_labels = labels
            aug_force_drop_ids = force_drop_ids

        aug_kwargs = dict(model_kwargs)
        aug_kwargs["y"] = aug_labels

        aug_output, aug_feat = model(
            aug_input,
            aug_time,
            force_drop_ids=aug_force_drop_ids,
            **aug_kwargs,
        )

        bsz = model_input.shape[0]
        model_output = aug_output[:bsz]
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        if (not has_dropped) or (aug_feat is None):
            return denoising_loss, denoising_loss.new_zeros(())

        uncond_feat_subset = aug_feat[1][:bsz][drop_idx]
        cond_feat_subset = aug_feat[1][bsz:]

        if cond_feat_subset.numel() == 0:
            return denoising_loss, denoising_loss.new_zeros(())

        cfg_scale = 3.0
        f_g = uncond_feat_subset + cfg_scale * (cond_feat_subset - uncond_feat_subset)
        f_g = f_g.detach()

        f_n = aug_feat[0][bsz:]

        f_g = F.normalize(f_g,  dim=-1)
        f_n = F.normalize(f_n, dim=-1)
        guided_loss = - (f_g * f_n).sum(dim=-1)

        # guided_loss = mean_flat((f_g - f_n) ** 2)

        return denoising_loss, guided_loss
