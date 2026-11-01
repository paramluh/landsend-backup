# wan-vae-collection@588eaeb models/electron/encoder.py (trimmed: forward dropped, FastElectronEncoder runs the modules itself)
import torch.nn as nn

from .vae import AttentionBlock, AvgDown3D, CausalConv3d, RMS_norm, ResidualBlock


class _DownBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, downsample: bool):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(in_dim, out_dim),
            ResidualBlock(out_dim, out_dim),
        )
        self.downsample = downsample
        if downsample:
            self.spatial = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(out_dim, out_dim, 3, stride=(2, 2)),
            )
            self.time_conv = CausalConv3d(
                out_dim, out_dim, (3, 1, 1), stride=(2, 1, 1),
                padding=(0, 0, 0))
            self.shortcut = AvgDown3D(in_dim, out_dim, factor_t=2, factor_s=2)


class ElectronEncoder(nn.Module):
    """
    pixels [B, 3, 1 + 4k, H, W] in [-1, 1], k >= 1 ->
    z [B, 48, 1 + k, H/16, W/16] scaled latents (mu only, deterministic).
    """

    def __init__(self, base_dim: int):
        super().__init__()
        self.base_dim = base_dim
        self.conv1 = CausalConv3d(48, base_dim // 4, 3, padding=1)
        self.down0 = _DownBlock(base_dim // 4, base_dim // 4, downsample=True)
        self.down1 = _DownBlock(base_dim // 4, base_dim // 2, downsample=True)
        self.down2 = _DownBlock(base_dim // 2, base_dim, downsample=False)
        self.middle = nn.Sequential(
            ResidualBlock(base_dim, base_dim),
            AttentionBlock(base_dim),
            ResidualBlock(base_dim, base_dim),
        )
        self.head = nn.Sequential(
            RMS_norm(base_dim, images=False),
            nn.SiLU(),
            CausalConv3d(base_dim, 48, 3, padding=1),
        )
        self.quant = CausalConv3d(48, 48, 1)
