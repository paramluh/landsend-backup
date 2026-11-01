# HF InternalCan/sam-pose-aloe-v2pure-step636@cbd9c1e3 pose_tokenizer/model/tokenizer.py
from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from ..config import PoseTokenizerConfig
from ..data.kinematic import build_part_index, get_part_scheme
from .encoder import Encoder, compute_channel_progression
from .decoder import Decoder
from .layers import PermutedLayerNorm
from .seanet_transformer import SeanetTransformerEncoder, SeanetTransformerDecoder
from .vq import ResidualVectorQuantize, SharedResidualVectorQuantize


class AggregationCNN(nn.Module):
    """Small temporal 1D-CNN that fuses per-part reconstructions into the body.

    Input  (B, N*M*keypoint_dim, T): the per-part decoded coordinates stacked in
        fixed (part, slot, coord) channel order.
    Output (B, input_features, T): the whole-body reconstruction (channels-first).

    The output is added to a uniform-mean residual skip in the caller; this
    module's final projection is zero-initialised so training starts exactly at
    the PRQ uniform mean and learns corrections from there.
    """

    def __init__(self, in_channels, out_channels, hidden, n_layers, kernel):
        super().__init__()
        pad = kernel // 2
        blocks = []
        ch = in_channels
        for _ in range(max(1, n_layers)):
            blocks.append(nn.Conv1d(ch, hidden, kernel_size=kernel, padding=pad))
            blocks.append(PermutedLayerNorm(normalized_shape=hidden))
            blocks.append(nn.SiLU())
            ch = hidden
        self.body = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out_proj(self.body(x))


def _init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        # NOTE: zero-bias init on Conv1d turns out to act as an *implicit*
        # regulariser via the PermutedLayerNorm backward path on padded
        # frames. Removing it (using default PyTorch bias init) collapsed
        # codebook utilisation to <5% in early experiments, so we keep the
        # explicit zero-init even though it looks redundant.
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PoseTokenizerModel(
    nn.Module,
    PyTorchModelHubMixin,
    tags=["pose-tokenizer", "keypoints", "vqvae"],
    repo_url="https://github.com/canopy-labs-internal/pose-tokenizer.git",
    license="mit",
):
    """
    VQ-VAE that tokenises body-pose keypoints into discrete codes.

    Architecture ported from FaceTokenizer (TCN encoder/decoder + RVQ).

    HuggingFace integration via PyTorchModelHubMixin gives you:
        model = PoseTokenizerModel.from_pretrained("your-hf-repo")
        model.save_pretrained("./local-dir")
        model.push_to_hub("your-hf-repo")
    """

    def __init__(self, config: dict | PoseTokenizerConfig | None = None, **kwargs):
        super().__init__()

        if config is None:
            config = PoseTokenizerConfig(**kwargs)
        elif isinstance(config, dict):
            config = PoseTokenizerConfig(**{
                k: v for k, v in config.items()
                if k in PoseTokenizerConfig.__dataclass_fields__
            })

        self.config = config
        c = config

        self.input_features = c.input_features
        self.enable_quantization = c.enable_quantization
        self.model_type = getattr(c, "model_type", "conv")

        if self.model_type == "seanet_transformer":
            self._build_seanet_transformer(c)
        elif self.model_type == "conv":
            self._build_conv(c)
        elif self.model_type == "part_rvq":
            self._build_part_rvq(c)
        else:
            raise ValueError(
                f"Unknown model_type={self.model_type!r}; expected 'conv', "
                "'seanet_transformer' or 'part_rvq'."
            )

        if self.model_type == "part_rvq":
            # The (shared) quantizer is built inside _build_part_rvq; the default
            # per-codebook RVQ is not used in this mode.
            self.quantizer = None
        elif c.enable_quantization:
            self.quantizer = ResidualVectorQuantize(
                input_dim=self.latent_dim,
                n_codebooks=c.n_codebooks,
                codebook_size=c.codebook_size,
                codebook_dim=c.codebook_dim,
                vq_strides=c.vq_strides,
                apply_rotation_trick=c.apply_rotation_trick,
                quantizer_dropout=c.quantizer_dropout,
                normalize_codes=c.normalize_codes,
                dead_code_threshold=c.dead_code_threshold,
                dead_code_ema_decay=c.dead_code_ema_decay,
            )
        else:
            self.quantizer = None

        self._active_n_codebooks = c.n_codebooks

        self.apply(_init_weights)

        # Zero-init the aggregator output AFTER _init_weights (which would
        # otherwise re-randomise the Conv1d), so the part_rvq CNN aggregator
        # starts at the uniform-mean reconstruction and learns corrections.
        if self.model_type == "part_rvq" and self.aggregator is not None:
            nn.init.zeros_(self.aggregator.out_proj.weight)
            if self.aggregator.out_proj.bias is not None:
                nn.init.zeros_(self.aggregator.out_proj.bias)

    # ------------------------------------------------------------------
    # Backend builders
    # ------------------------------------------------------------------

    def _build_conv(self, c: PoseTokenizerConfig) -> None:
        """Original TCN encoder/decoder (default)."""
        self.downsampling_factor = 1
        for rate in c.downsampling_rates:
            self.downsampling_factor *= rate
        d_model = c.d_model if c.d_model is not None else c.first_channel_size * 2
        encoder_final_dim = compute_channel_progression(
            c.first_channel_size, d_model, c.downsampling_rates,
        )[-1]
        self.latent_dim = c.latent_dim if c.latent_dim is not None else encoder_final_dim

        self.encoder = Encoder(
            input_features=c.input_features,
            first_channel_size=c.first_channel_size,
            latent_dim=self.latent_dim,
            num_encoder_blocks=c.num_blocks,
            kernel_size=c.kernel_size,
            downsampling_rates=c.downsampling_rates,
            num_residual_blocks=c.num_residual_blocks,
            residual_units_per_block=c.residual_units_per_block,
            d_model=c.d_model,
            num_groups=c.num_groups,
            use_weight_norm=c.conv_weight_norm,
            causal=c.causal,
            dilation_mode="deep" if c.encoder_deep_dilation else c.encoder_dilation,
        )

        upsampling_rates = list(c.downsampling_rates)[::-1]

        self.decoder = Decoder(
            output_features=c.input_features,
            first_channel_size=c.first_channel_size,
            latent_dim=self.latent_dim,
            num_decoder_blocks=c.num_blocks,
            upsampling_rates=upsampling_rates,
            num_residual_blocks=c.num_residual_blocks,
            residual_units_per_block=c.residual_units_per_block,
            kernel_size=c.kernel_size,
            d_model=c.d_model,
            num_groups=c.num_groups,
            use_weight_norm=c.conv_weight_norm,
            causal=c.causal,
        )

    def _build_seanet_transformer(self, c: PoseTokenizerConfig) -> None:
        """Mimi-style SEANet conv front/back end + causal transformer bottleneck.

        Reuses the RVQ (built in ``__init__``) on the encoder output, so the
        codebook/bitrate match the conv tokenizer exactly. ``downsampling_factor``
        comes from the separate causal resample (``st_downsample``).
        """
        self.downsampling_factor = c.st_downsample
        # RVQ input width. Reuse the shared ``latent_dim`` field if set, else the
        # SEANet width. Keeps the bottleneck size explicit and serialisable.
        self.latent_dim = c.latent_dim if c.latent_dim is not None else c.st_seanet_dim

        st_kw = dict(
            seanet_dim=c.st_seanet_dim,
            n_filters=c.st_n_filters,
            n_residual_layers=c.st_n_residual_layers,
            ratios=list(c.st_ratios),
            seanet_kernel=c.st_seanet_kernel,
            residual_kernel=c.st_residual_kernel,
            compress=c.st_compress,
            d_model=c.st_d_model,
            num_heads=c.st_num_heads,
            num_layers=c.st_num_layers,
            dim_feedforward=c.st_dim_feedforward,
            layer_scale=c.st_layer_scale,
            rope_max_period=c.st_rope_max_period,
            causal=c.causal,
            use_weight_norm=c.conv_weight_norm,
        )
        self.encoder = SeanetTransformerEncoder(
            input_features=c.input_features,
            latent_dim=self.latent_dim,
            downsample=c.st_downsample,
            **st_kw,
        )
        self.decoder = SeanetTransformerDecoder(
            output_features=c.input_features,
            latent_dim=self.latent_dim,
            upsample=c.st_downsample,
            **st_kw,
        )

    def _build_part_rvq(self, c: PoseTokenizerConfig) -> None:
        """Body-part RVQ backend: shared per-part encoder/decoder + shared codebook.

        The 55-joint pose is split into ``part_scheme`` overlapping parts (each M
        joints). ONE Encoder/Decoder (sized like the conv baseline, but with M*kp
        I/O width) is applied to every part, ONE shared codebook is reused at
        ``part_n_layers`` residual depths per part, and a small learned CNN fuses
        the per-part reconstructions into the whole body.
        """
        part_names, part_joints = get_part_scheme(c.part_scheme)
        self.part_names = part_names
        self.n_parts = len(part_joints)
        self.part_m = len(part_joints[0])
        # Channels per joint = total features / joints. 2 = (y, x) coords-only;
        # 3 = (y, x, confidence) when confidence is baked into the features
        # (train_confidence.py). Derived (not keypoint_dim) so both layouts work.
        if c.input_features % c.num_keypoints != 0:
            raise ValueError(
                f"input_features={c.input_features} not divisible by "
                f"num_keypoints={c.num_keypoints} for part_rvq."
            )
        self.part_joint_channels = c.input_features // c.num_keypoints
        part_in_features = self.part_m * self.part_joint_channels

        part_joint_idx, joint_counts = build_part_index(part_joints, c.num_keypoints)
        self.register_buffer(
            "part_joint_idx",
            torch.as_tensor(part_joint_idx, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "part_flat_joint_idx",
            torch.as_tensor(part_joint_idx.reshape(-1), dtype=torch.long),
            persistent=False,
        )
        # (1, 1, num_keypoints, 1) float denominator for the uniform-mean skip.
        self.register_buffer(
            "part_joint_counts",
            torch.as_tensor(joint_counts, dtype=torch.float32)
            .clamp(min=1.0)
            .view(1, 1, c.num_keypoints, 1),
            persistent=False,
        )

        self.downsampling_factor = 1
        for rate in c.downsampling_rates:
            self.downsampling_factor *= rate

        d_model = c.d_model if c.d_model is not None else c.first_channel_size * 2
        encoder_final_dim = compute_channel_progression(
            c.first_channel_size, d_model, c.downsampling_rates,
        )[-1]
        self.latent_dim = c.latent_dim if c.latent_dim is not None else encoder_final_dim

        self.encoder = Encoder(
            input_features=part_in_features,
            first_channel_size=c.first_channel_size,
            latent_dim=self.latent_dim,
            num_encoder_blocks=c.num_blocks,
            kernel_size=c.kernel_size,
            downsampling_rates=c.downsampling_rates,
            num_residual_blocks=c.num_residual_blocks,
            residual_units_per_block=c.residual_units_per_block,
            d_model=c.d_model,
            num_groups=c.num_groups,
            use_weight_norm=c.conv_weight_norm,
            causal=c.causal,
            dilation_mode="deep" if c.encoder_deep_dilation else c.encoder_dilation,
        )

        upsampling_rates = list(c.downsampling_rates)[::-1]
        self.decoder = Decoder(
            output_features=part_in_features,
            first_channel_size=c.first_channel_size,
            latent_dim=self.latent_dim,
            num_decoder_blocks=c.num_blocks,
            upsampling_rates=upsampling_rates,
            num_residual_blocks=c.num_residual_blocks,
            residual_units_per_block=c.residual_units_per_block,
            kernel_size=c.kernel_size,
            d_model=c.d_model,
            num_groups=c.num_groups,
            use_weight_norm=c.conv_weight_norm,
            causal=c.causal,
        )

        self.part_quantizer = SharedResidualVectorQuantize(
            input_dim=self.latent_dim,
            n_layers=c.part_n_layers,
            codebook_size=c.part_codebook_size,
            codebook_dim=c.codebook_dim,
            apply_rotation_trick=c.apply_rotation_trick,
            normalize_codes=c.normalize_codes,
            dead_code_threshold=c.dead_code_threshold,
            dead_code_ema_decay=c.dead_code_ema_decay,
        )

        # Which residual layers each loss supervises (prefix counts, -1 = all).
        def _resolve_layers(v):
            return c.part_n_layers if v is None or v < 0 else min(int(v), c.part_n_layers)
        self.part_loss_layers = _resolve_layers(getattr(c, "part_loss_layers", -1))
        self.whole_loss_layers = max(1, _resolve_layers(getattr(c, "whole_loss_layers", -1)))

        self.part_aggregation = getattr(c, "part_aggregation", "mean")
        if self.part_aggregation == "cnn":
            self.aggregator = AggregationCNN(
                in_channels=self.n_parts * part_in_features,
                out_channels=c.input_features,
                hidden=c.part_agg_channels,
                n_layers=c.part_agg_layers,
                kernel=c.part_agg_kernel,
            )
        elif self.part_aggregation == "mean":
            self.aggregator = None
        else:
            raise ValueError(
                f"Unknown part_aggregation={self.part_aggregation!r}; "
                "expected 'mean' or 'cnn'."
            )

    def _decode_part_grid(self, z_q, B, T):
        """(B*N, D, T') quantized latent -> per-part grid (B, T, N, M, kp)."""
        N, M, kp = self.n_parts, self.part_m, self.part_joint_channels
        dec = self.decoder(z_q)  # (B*N, M*kp, T)
        return dec.reshape(B, N, M, kp, T).permute(0, 4, 1, 2, 3)

    def _aggregate(self, part_recon_grid, B, T):
        """Combine per-part grid (B, T, N, M, kp) -> whole body (B, T, input_features)."""
        c = self.config
        N, M, kp = self.n_parts, self.part_m, self.part_joint_channels
        parts_flat = part_recon_grid.reshape(B, T, N * M, kp)
        whole_sum = part_recon_grid.new_zeros(B, T, c.num_keypoints, kp)
        whole_sum.index_add_(2, self.part_flat_joint_idx, parts_flat)
        whole = whole_sum / self.part_joint_counts
        if self.aggregator is not None:
            agg_in = part_recon_grid.permute(0, 2, 3, 4, 1).reshape(B, N * M * kp, T)
            agg_out = self.aggregator(agg_in).permute(0, 2, 1).reshape(
                B, T, c.num_keypoints, kp
            )
            whole = whole + agg_out
        return whole.reshape(B, T, c.input_features)

    def _forward_part_rvq(self, keypoints: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-part encode -> shared RVQ -> per-part decode -> learned aggregate.

        Residual layers can be supervised by different losses (see config
        ``part_loss_layers`` / ``whole_loss_layers``): the returned
        ``reconstruction`` always uses ALL layers (for inference/metrics), while
        ``reconstruction_for_loss`` (term A) and ``part_recon`` (term B) use the
        configured leading-layer prefixes so e.g. the finest codes can be driven
        only by the whole-body loss.
        """
        c = self.config
        B, T, _ = keypoints.shape
        N, M, kp = self.n_parts, self.part_m, self.part_joint_channels
        L = c.part_n_layers

        kp_grid = keypoints.reshape(B, T, c.num_keypoints, kp)
        # Gather each part's joints: (B, T, N, M, kp).
        parts_in = kp_grid[:, :, self.part_joint_idx, :]
        # -> (B*N, M*kp, T) for the shared conv encoder (channels-first).
        enc_in = parts_in.permute(0, 2, 3, 4, 1).reshape(B * N, M * kp, T)

        z = self.encoder(enc_in)
        z_q, layer_codes, commitment_loss, codebook_loss, layer_zq = self.part_quantizer(z)

        def prefix_sum(k):
            # Sum the first k per-layer quantized contributions.
            zq = layer_zq[0]
            for i in range(1, k):
                zq = zq + layer_zq[i]
            return zq

        # Full-depth reconstruction (all layers) -> always the returned output.
        full_grid = self._decode_part_grid(z_q, B, T)
        reconstruction = self._aggregate(full_grid, B, T)

        out = {
            "reconstruction": reconstruction,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }

        # Whole-body loss target (term A): first `whole_loss_layers` layers.
        if self.whole_loss_layers >= L:
            out["reconstruction_for_loss"] = reconstruction
        else:
            whole_grid = self._decode_part_grid(prefix_sum(self.whole_loss_layers), B, T)
            out["reconstruction_for_loss"] = self._aggregate(whole_grid, B, T)

        # Per-part loss target (term B): first `part_loss_layers` layers.
        if self.part_loss_layers >= 1:
            if self.part_loss_layers >= L:
                part_grid = full_grid
            else:
                part_grid = self._decode_part_grid(prefix_sum(self.part_loss_layers), B, T)
            out["part_recon"] = [part_grid[:, :, j] for j in range(N)]  # N x (B,T,M,kp)

        # All N*L code streams, layer-major (coarse->fine), parts in scheme order:
        # [l0_p0, l0_p1, ..., l1_p0, ...]. Every layer's codes still feed the
        # full-depth `reconstruction`, regardless of which loss supervises them.
        codes = []
        for layer_idx in layer_codes:
            idx_bn = layer_idx.reshape(B, N, -1)  # (B, N, T')
            for p in range(N):
                codes.append(idx_bn[:, p, :].contiguous())
        out["codes"] = codes
        return out

    # ------------------------------------------------------------------
    # Preprocessing: (B, T, F) <-> (B, F, T)
    # ------------------------------------------------------------------

    def preprocess(self, x):
        return x.permute(0, 2, 1)

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Encode / Decode / Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor):
        """(B, T, F) -> quantised latent + codes + losses."""
        if self.model_type == "part_rvq":
            raise NotImplementedError(
                "part_rvq tokenizer does not expose a single-stream encode(); "
                "use forward() which returns per-part codes and reconstruction."
            )
        x = self.preprocess(x)
        z = self.encoder(x)

        if self.quantizer is not None:
            z_q, codes, commitment_loss, codebook_loss = self.quantizer(z)
            return z_q, codes, commitment_loss, codebook_loss

        zero = torch.tensor(0.0, device=z.device)
        return z, None, zero, zero

    def decode(self, codes: list[torch.Tensor], n_codebooks: int | None = None):
        """Reconstruct from discrete codes."""
        if n_codebooks is None:
            n_codebooks = self._active_n_codebooks
        z_q = self.quantizer.from_codes(codes, n_quantizers=n_codebooks)
        return self.postprocess(self.decoder(z_q))

    def forward(self, keypoints: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full encode -> (quantize) -> decode pass.

        Args:
            keypoints: (B, T, F) raw keypoint features.

        Returns:
            Dict with ``reconstruction``, ``codes``, ``commitment_loss``,
            ``codebook_loss``.
        """
        if self.model_type == "part_rvq":
            return self._forward_part_rvq(keypoints)

        x = self.preprocess(keypoints)
        z = self.encoder(x)

        if self.quantizer is not None:
            z_q, codes, commitment_loss, codebook_loss = self.quantizer(z)
        else:
            z_q = z
            codes = None
            commitment_loss = torch.tensor(0.0, device=z.device)
            codebook_loss = torch.tensor(0.0, device=z.device)

        reconstruction = self.postprocess(self.decoder(z_q))

        return {
            "reconstruction": reconstruction,
            "codes": codes,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    def set_active_codebooks(self, n: int):
        if n > self.config.n_codebooks:
            raise ValueError(
                f"Cannot set {n} active codebooks, model only has {self.config.n_codebooks}"
            )
        self._active_n_codebooks = n

    def tokenize(self, keypoints: torch.Tensor) -> list[torch.Tensor]:
        """keypoints -> discrete codes."""
        _, codes, _, _ = self.encode(keypoints)
        return codes

    def detokenize(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """discrete codes -> reconstructed keypoints."""
        return self.decode(codes)
