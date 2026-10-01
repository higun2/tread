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

    def __call__(self, model, images, model_kwargs=None, class_labels=None, attn_maps_dict=None):
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

        model_output, _ = model(model_input, time_input.flatten(), **model_kwargs)

        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # Compute KL-divergence loss for attention alignment
        kl_loss = torch.zeros_like(denoising_loss)
        if attn_maps_dict is not None and "shallow" in attn_maps_dict and "deep" in attn_maps_dict:
            shallow_obj = attn_maps_dict["shallow"]
            deep_obj = attn_maps_dict["deep"]

            # Accept either raw tensors or hooked attention modules.
            attn_shallow = shallow_obj.saved_attn if hasattr(shallow_obj, "saved_attn") else shallow_obj
            attn_deep = deep_obj.saved_attn if hasattr(deep_obj, "saved_attn") else deep_obj

            if attn_shallow is not None and attn_deep is not None:
                # Average over head dimension
                batch_size = attn_shallow.shape[0]
                avg_attn_shallow = attn_shallow.mean(dim=1).reshape(batch_size, -1)  # [B, N, N] -> [B, N*N]
                avg_attn_deep = attn_deep.mean(dim=1).reshape(batch_size, -1)        # [B, N, N] -> [B, N*N]

                # Apply log_softmax
                log_prob_shallow = F.log_softmax(avg_attn_shallow, dim=-1)
                log_prob_deep = F.log_softmax(avg_attn_deep.detach(), dim=-1)

                # Compute KL-divergence with log_target=True
                kl_loss = F.kl_div(
                    log_prob_shallow,
                    log_prob_deep,
                    reduction='none',
                    log_target=True
                ).sum(dim=-1)

        return denoising_loss, kl_loss
