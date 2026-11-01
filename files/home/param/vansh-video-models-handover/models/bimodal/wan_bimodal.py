import torch

from models.bimodal.stabiliser import StabiliserWidePipeline
from models.pose.wan_universal import UniversalWidePipeline


TAG = 'universal_wide_bimodal'


class BimodalWidePipeline(StabiliserWidePipeline):
    """
    The stabilising mode; self.resolving is the resolving mode over the same modules.
    """
    name = TAG

    def load_diffusion_model(self):
        super().load_diffusion_model()
        self.resolving = ResolvingView(self, **self.config['resolving'])


class ResolvingView(UniversalWidePipeline):
    """
    base's modules in the window [sink | num_history | num_present]: clean history; the sink and pads at t = 0.
    Present t in training: band k = (present_t_bands[k-1], present_t_bands[k]] (from 0) with probability
    proportional to present_t_weights[k], uniform within it; in eval, present_t_bands[-1].
    """

    def __init__(self, base, num_history, num_present, present_t_bands, present_t_weights):
        assert len(present_t_bands) == len(present_t_weights), (present_t_bands, present_t_weights)
        self.__dict__.update(base.__dict__)
        self.num_history, self.num_present = num_history, num_present
        self.num_frames, self.num_clean = 1 + num_history + num_present, 1 + num_history
        self.band_edges = torch.tensor([0.0, *present_t_bands])
        self.band_probs = torch.tensor(present_t_weights, dtype=torch.float32) / sum(present_t_weights)

    def training_t(self, latents, ofi, valid):
        B, device = latents.shape[0], latents.device
        band = torch.multinomial(self.band_probs.to(device), B * self.num_present, replacement=True).view(B, -1)
        edges = self.band_edges.to(device)
        quantiles = 1 - torch.rand(B, self.num_present, device=device)
        t = latents.new_zeros(B, latents.shape[2])
        t[:, self.num_clean:] = (edges[band] + quantiles * (edges[band + 1] - edges[band])) * valid
        return t

    def eval_t(self, ofi):
        t = torch.zeros(ofi.shape)
        t[:, self.num_clean:] = self.band_edges[-1]
        return t
