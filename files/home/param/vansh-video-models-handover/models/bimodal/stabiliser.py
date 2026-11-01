import torch

from models.pose.wan_universal import UniversalWidePipeline


TAG = 'universal_wide_stabiliser'


class StabiliserWidePipeline(UniversalWidePipeline):
    """
    Real history frames at t = history_t (out of the loss); present slot i of P over
    (history_t + (1 - history_t) i/P, history_t + (1 - history_t)(i+1)/P], so a frame leaves the present at exactly
    the t it is read at as history; the sink and pads at t = 0. In training, each real history frame but the newest
    is raised, with probability history_raise_p, to t ~ U(history_t, history_t_max].
    """
    name = TAG

    def __init__(self, config):
        super().__init__(config)
        self.history_t = config['history_t']
        self.history_raise_p = config['history_raise_p']
        self.history_t_max = config['history_t_max']

    def _t(self, t, ofi, present):
        """
        Fills t in place.

        present: [B, P] in (0, 1], mapped onto (history_t, 1]
        -> t [B, 1+H+P]
        """
        t[:, 1:self.num_clean] = self.history_t * (ofi[:, 1:self.num_clean] >= 0)
        t[:, self.num_clean:] = self.history_t + (1 - self.history_t) * present
        return t

    def training_t(self, latents, ofi, valid):
        quantiles = 1 - torch.rand(latents.shape[0], self.num_present, device=latents.device)
        buckets = torch.arange(self.num_present, device=latents.device).view(1, -1)
        t = self._t(latents.new_zeros(latents.shape[0], latents.shape[2]), ofi, (buckets + quantiles) / self.num_present)
        t[:, self.num_clean:] *= valid
        older = (slice(None), slice(1, self.num_history))
        raised = (torch.rand(t[older].shape, device=t.device) < self.history_raise_p) & (ofi[older] >= 0)
        high = self.history_t_max - (self.history_t_max - self.history_t) * torch.rand(t[older].shape, device=t.device)
        t[older] = torch.where(raised, high, t[older])
        return t

    def eval_t(self, ofi):
        return self._t(torch.zeros(ofi.shape), ofi, torch.arange(1, self.num_present + 1) / self.num_present)
