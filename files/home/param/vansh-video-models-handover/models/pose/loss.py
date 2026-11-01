import torch
import torch.distributed as dist
import torch.nn.functional as F


def make_loss_fn(self, num_present, num_clean):
    self._train_loss_bucket_sums = torch.zeros(num_present, dtype=torch.float64, device='cuda')
    self._train_loss_bucket_counts = torch.zeros(num_present, dtype=torch.float64, device='cuda')
    self._accumulate_loss_buckets = True

    def loss_fn(output, label):
        target, loss_mask = label
        with torch.autocast('cuda', enabled=False):
            output = output.float()
            target = target.to(output.device, torch.float32)
            loss_mask = loss_mask.to(output.device, torch.float32)
            frame_loss = F.mse_loss(output, target, reduction='none').mean(dim=(1, 3, 4))
            frame_valid = loss_mask[:, 0, :, 0, 0] > 0
            present_loss = frame_loss[:, num_clean:]
            present_valid = frame_valid[:, num_clean:]
            if self._accumulate_loss_buckets:
                self._train_loss_bucket_sums.add_(
                    (present_loss * present_valid.to(present_loss.dtype)).sum(dim=0).detach().double())
                self._train_loss_bucket_counts.add_(present_valid.sum(dim=0).double())
            valid_counts = present_valid.sum(dim=1).to(torch.float32).clamp(min=1)
            return ((present_loss * present_valid).sum(dim=1) / valid_counts).mean()

    return loss_fn


def reset_train_loss_buckets(self):
    self._train_loss_bucket_sums.zero_()
    self._train_loss_bucket_counts.zero_()


def get_train_loss_buckets(self):
    sums = self._train_loss_bucket_sums.clone()
    counts = self._train_loss_bucket_counts.clone()
    dist.all_reduce(sums)
    dist.all_reduce(counts)
    output = {}
    for index, (total, count) in enumerate(zip(sums.tolist(), counts.tolist())):
        if count:
            output[f'slot-{index + 1}'] = total / count
    return output
