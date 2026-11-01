from pathlib import Path

import safetensors.torch
import torch
from flash_attn_interface import flash_attn_func
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers import MimiModel

from dataloading.sources import AUDIO_CODE_QUANTUM, NUM_REFS
from models.base import BasePipeline
from models.corruption import HistoryCorruption
from models.pose import helpers, loss, rope
from models.pose.audio import audio_latents_from_codes
from models.pose.settings import wan_5b_wide
from models.wan.model import sinusoidal_embedding_1d
from utils.common import AUTOCAST_DTYPE, is_main_process


TAG = 'universal_wide'
WINDOW_OFFSET = 10


class UniversalWidePipeline(BasePipeline):
    name = TAG

    def __init__(self, config):
        self.config = config
        self.model_config = config['model']
        assert self.model_config['arch'] == 'wan_5b_wide'

        self.num_history = config['num_history']
        self.num_present = config['num_present']
        self.num_frames = 1 + self.num_history + self.num_present
        self.num_clean = 1 + self.num_history
        # [refs] puts NUM_REFS clean reference frames before the sink.
        self.num_refs = NUM_REFS if 'refs' in config else 0
        self.model_dim = wan_5b_wide.dim
        self._debug_shapes = self.model_config['debug_shapes']
        self.history_corruption = HistoryCorruption(config['history_corruption']) if 'history_corruption' in config else None

        self.audio_config = self.model_config['audio']
        self.audio_num_codebooks = self.audio_config['num_codebooks']
        self.audio_min_codebooks = self.audio_config['min_codebooks']
        self.audio_cross_attention_codes = self.audio_config['cross_attention_codes']
        self.audio_dropout_p = self.audio_config['audio_dropout']
        self.audio_drop = self.audio_config['drop']
        self.audio_num_layers = self.audio_config['num_layers']
        self.audio_in_dim = self.audio_config['in_dim']
        self.audio_injection_offset = self.audio_config['injection_offset']
        self.audio_injection_interval = self.audio_config['injection_interval']
        self.use_hidden = 'hidden' in self.model_config

        self.model_arch = dict(
            patch_size=wan_5b_wide.patch_size,
            in_dim=wan_5b_wide.in_dim,
            dim=wan_5b_wide.dim,
            ffn_dim=wan_5b_wide.ffn_dim,
            freq_dim=wan_5b_wide.freq_dim,
            out_dim=wan_5b_wide.out_dim,
            num_heads=wan_5b_wide.num_heads,
            num_layers=wan_5b_wide.num_layers,
            qk_norm=wan_5b_wide.qk_norm,
            eps=wan_5b_wide.eps,
        )
        self.transformer_path = Path(self.model_config['checkpoint_path'])

        self.mimi = MimiModel.from_pretrained(self.audio_config['model_name'])
        self.mimi.requires_grad_(False)
        self.mimi.eval()
        self.mimi.to('cuda')

    def load_diffusion_model(self):
        helpers.load_diffusion_model(self)

    def _extract_audio_features(self, codes, training):
        if codes.ndim != 3 or codes.shape[1] < self.audio_num_codebooks or codes.shape[2] != AUDIO_CODE_QUANTUM:
            raise RuntimeError(
                f'expected codes [B, >={self.audio_num_codebooks}, {AUDIO_CODE_QUANTUM}], '
                f'got {tuple(codes.shape)}')
        num_codebooks = self.audio_num_codebooks
        if training and self.audio_min_codebooks < num_codebooks:
            num_codebooks = torch.randint(self.audio_min_codebooks, num_codebooks + 1, ()).item()
        return audio_latents_from_codes(self.mimi, codes, num_codebooks)

    def prepare_inputs(self, inputs):
        device = torch.device('cuda', torch.cuda.current_device())
        # Move first, cast on device: a dtype-changing .to() from pinned memory goes through an
        # unpinned CPU temporary and the H2D copy blocks until the GPU drains.
        latents = inputs['latents_low'].to(device, non_blocking=True).float()
        pose_latents = inputs['pose_latents_low'].to(device, non_blocking=True).float()
        ofi = inputs['original_frame_indices'].to(device, non_blocking=True)
        window_source = inputs['window_source'].to(device, non_blocking=True)
        valid = inputs['present_valid'].to(device, non_blocking=True)
        batch_size = latents.shape[0]
        training = 't' not in inputs

        # The data window is [sink | H history | P present]; loss_mask spans the model frames (subclasses may add some).
        if training:
            if self.history_corruption is not None:
                self.history_corruption(latents, pose_latents, ofi, self.num_history)
            t = self.training_t(latents, ofi, valid)
            noise = torch.randn_like(latents)
        else:
            t = inputs['t'].to(device, non_blocking=True).float()
            noise = inputs['noise'].to(device, non_blocking=True).float()

        t_expanded = t[:, None, :, None, None]
        x_t = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents
        loss_mask = latents.new_zeros(batch_size, 1, self.num_frames, 1, 1)
        loss_mask[:, :, self.num_clean:] = valid[:, None, :, None, None]
        audio_latents = self._extract_audio_features(inputs['codes'].to(device, non_blocking=True), training)

        features = (x_t, t * 1000, audio_latents, pose_latents, ofi, window_source, valid)
        if self.num_refs:
            features += (
                inputs['ref_latents'].to(device, non_blocking=True).float(),
                inputs['ref_keep'].to(device, non_blocking=True),
            )
        if self.use_hidden:
            features += (inputs['llm_hiddens'].to(device, non_blocking=True),)
        if training and 'bad_audio' in inputs:
            features += (inputs['bad_audio'].to(device, non_blocking=True),)

        if self._debug_shapes and is_main_process():
            print(
                f'[{TAG}] latents={tuple(latents.shape)} '
                f'audio={tuple(audio_latents.shape)} '
                f'pose={tuple(pose_latents.shape)}'
                + (f' refs={tuple(features[7].shape)}' if self.num_refs else '')
            )
            self._debug_shapes = False

        return features, (target, loss_mask)

    def training_t(self, latents, ofi, valid):
        """
        Present slot i at t ~ U(i/P, (i+1)/P] where valid; the sink, history and pads at 0.

        -> t [B, 1+H+P]
        """
        t = latents.new_zeros(latents.shape[0], latents.shape[2])
        quantiles = 1 - torch.rand(latents.shape[0], self.num_present, device=latents.device)
        buckets = torch.arange(self.num_present, device=latents.device).view(1, -1)
        t[:, self.num_clean:] = (buckets + quantiles) / self.num_present * valid
        return t

    def eval_t(self, ofi):
        """
        Present slot i at the top of its band, (i+1)/P; the sink, history and pads at 0.

        -> t [B, 1+H+P], on the CPU
        """
        t = torch.zeros(ofi.shape)
        t[:, self.num_clean:] = torch.arange(1, self.num_present + 1, dtype=t.dtype) / self.num_present
        return t

    def get_loss_fn(self):
        return loss.make_loss_fn(self, self.num_present, self.num_clean)

    def reset_train_loss_buckets(self):
        loss.reset_train_loss_buckets(self)

    def get_train_loss_buckets(self):
        return loss.get_train_loss_buckets(self)

    def get_param_groups(self, parameters):
        return helpers.get_param_groups(self, parameters)

    def save_model(self, save_dir, state_dict):
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def to_layers(self):
        freqs = self.transformer.freqs.to('cuda')
        compile_blocks = self.config['compile_blocks']
        layers = [
            InitialLayer(
                self.transformer, self.audio_encoder, self.pose_patch_embedding, self.pad_token,
                num_history=self.num_history,
                num_present=self.num_present,
                audio_cross_attention_codes=self.audio_cross_attention_codes,
                audio_dropout_p=self.audio_dropout_p,
                compile_block=compile_blocks,
                num_refs=self.num_refs,
                ref_patch_embedding=self.ref_patch_embedding,
                hidden_drop_token=self.hidden_adapter and self.hidden_adapter.drop_token,
            )
        ]
        skip = self.config['activation_checkpoint_skip']
        assert 0 <= skip <= len(self.transformer.blocks)
        for index, block in enumerate(self.transformer.blocks):
            layers.append(
                TransformerLayer(
                    block, self.audio_adapter.get_block(index), freqs,
                    num_history=self.num_history,
                    num_present=self.num_present,
                    compile_block=compile_blocks,
                    checkpoint=index >= skip,
                    num_refs=self.num_refs,
                    flex=self.config.get('flex_attention', False),
                    hidden_block=self.hidden_adapter and self.hidden_adapter.get_block(index),
                )
            )
        layers.append(FinalLayer(self.transformer, num_refs=self.num_refs))
        return layers


class InitialLayer(nn.Module):
    """
    With num_refs, the token sequence is [num_refs refs | sink | history | present]: the refs are clean (t = 0)
    and get no pose, and a ref slot with ref_keep False (dropped or missing) is the pad token, exactly like a
    dropped history frame. The grid handed on is the window's (sink | history | present) grid.
    With hidden_drop_token (HiddenAdapter.drop_token), the inputs carry the present frames' Orpheus hiddens
    [B, P, 2, 4096] after the refs, handed on with a window's hiddens replaced by the drop token wherever its audio is
    dropped.
    """

    def __init__(self, model, audio_encoder, pose_patch_embedding, pad_token, num_history, num_present,
                 audio_cross_attention_codes, audio_dropout_p, compile_block, num_refs=0, ref_patch_embedding=None,
                 hidden_drop_token=None):
        super().__init__()
        self.patch_embedding = model.patch_embedding
        self.ref_patch_embedding = ref_patch_embedding
        self.time_embedding = model.time_embedding
        self.time_projection = model.time_projection
        self.freq_dim = model.freq_dim
        self.dim = model.dim
        self.audio_encoder = audio_encoder
        self.pose_patch_embedding = pose_patch_embedding
        self.pad_token = pad_token
        self.num_history = num_history
        self.num_present = num_present
        self.num_frames = 1 + num_history + num_present
        self.num_clean = 1 + num_history
        self.num_refs = num_refs
        self.hidden_drop_token = hidden_drop_token
        self.hidden = hidden_drop_token is not None
        self.audio_cross_attention_codes = audio_cross_attention_codes
        self.audio_dropout_p = audio_dropout_p
        self.audio_dropped = torch.zeros((), dtype=torch.float64, device='cuda')
        self.audio_drop_seen = 0
        self._run = torch.compile(self._forward_impl) if compile_block else self._forward_impl

    def _window_audio(self, audio_tokens, ofi, window_source):
        K = self.audio_cross_attention_codes
        source = audio_tokens.index_select(0, window_source)
        indices = ofi[:, self.num_clean:]
        offsets = torch.arange(K, device=audio_tokens.device)
        positions = (K * (indices.clamp(min=1) - 1)).unsqueeze(-1) + offsets
        D = source.shape[-1]
        gather_idx = positions.unsqueeze(-1).expand(-1, -1, -1, D)
        expanded = source.unsqueeze(1).expand(-1, indices.shape[1], -1, -1)
        gathered = torch.gather(expanded, 2, gather_idx)
        null = self.audio_encoder.null_token.to(gathered.dtype)
        return torch.cat(
            [gathered, null.expand(gathered.shape[0], gathered.shape[1], 1, D)],
            dim=2,
        )

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        bad_audio = hiddens = None
        if len(inputs) > 7 + 2 * bool(self.num_refs) + self.hidden:
            *inputs, bad_audio = inputs
        if self.hidden:
            *inputs, hiddens = inputs
        x, e, e0, grid, audio_encoded, present_valid, hiddens, dropped = self._run(
            *inputs, bad_audio=bad_audio, hiddens=hiddens)
        if dropped is not None:
            self.audio_dropped += dropped
            self.audio_drop_seen += present_valid.shape[0]
        return x, e, e0, grid, audio_encoded, present_valid, hiddens

    def _forward_impl(self, x, t, audio_latents, pose_latents, ofi, window_source, present_valid, *refs, bad_audio=None,
                      hiddens=None):
        pad_frames = ofi < 0
        if refs:
            ref_latents, ref_keep = refs
            t = torch.cat((t.new_zeros(t.shape[0], self.num_refs), t), dim=1)
            pad_frames = torch.cat((~ref_keep, pad_frames), dim=1)
            x = torch.cat((self.ref_patch_embedding(ref_latents), self.patch_embedding(x)), dim=2)
        else:
            x = self.patch_embedding(x)
        grid = tuple(x.shape[2:])
        tokens_per_frame = grid[1] * grid[2]
        x = x.flatten(2).transpose(1, 2)
        pad = pad_frames.repeat_interleave(tokens_per_frame, dim=1)
        x = torch.where(pad[:, :, None], self.pad_token.to(x.dtype), x)

        pose_tokens = self.pose_patch_embedding(pose_latents.to(x.dtype)).flatten(2).transpose(1, 2)
        pose_tokens = pose_tokens.unflatten(1, (ofi.shape[1], tokens_per_frame))
        pose_valid = (ofi >= 1).to(pose_tokens.dtype)
        pose_tokens = (pose_tokens * pose_valid[:, :, None, None]).flatten(1, 2)
        if refs:
            grid = (grid[0] - self.num_refs, *grid[1:])
            pose_tokens = nn.functional.pad(pose_tokens, (0, 0, self.num_refs * tokens_per_frame, 0))
        x = x + pose_tokens

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).unflatten(0, t.shape)
        )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        audio_tokens = self.audio_encoder(audio_latents)
        audio_encoded = self._window_audio(audio_tokens, ofi, window_source)
        dropped = None
        if self.training and (self.audio_dropout_p or bad_audio is not None):
            drop = torch.rand(audio_encoded.shape[0], device=audio_encoded.device) < self.audio_dropout_p
            if bad_audio is not None:
                drop = drop | bad_audio
            audio_encoded = torch.where(
                drop.view(-1, 1, 1, 1),
                self.audio_encoder.no_audio_tokens.to(audio_encoded.dtype),
                audio_encoded,
            )
            if hiddens is not None:
                hiddens = torch.where(drop.view(-1, 1, 1, 1), self.hidden_drop_token.to(hiddens.dtype), hiddens)
            dropped = drop.sum()

        return x, e, e0, grid, audio_encoded, present_valid, hiddens, dropped


_GEOMETRY = {}


def sink_geometry(freqs, grid, num_refs, num_history, num_present, device):
    """
    The key sets of _fixed_attention as one flex_attention pass. Queries [refs | sink | history | present] with the
    history rotated at the anchor slot; keys [refs | sink | history at their true slots | present | history at the
    anchor slot] (the anchor copy is read only by the history frame itself, the true-slot copy only by present
    frames). Returns the query and key RoPE tables and the frame-level BlockMask, cached per geometry.
    """
    key = (grid, num_refs, num_history, num_present, device)
    if key in _GEOMETRY:
        return _GEOMETRY[key]
    R, H, P = num_refs, num_history, num_present
    f, h, w = grid
    N = h * w
    offset = WINDOW_OFFSET if R else 0
    sink = R
    Fq = R + 1 + H + P
    positions_q = [0] * R + [offset] + [offset + H] * H + [offset + H + 1 + i for i in range(P)]
    positions_k = positions_q[:R + 1] + [offset + 1 + j for j in range(H)] + positions_q[R + 1 + H:] + [offset + H] * H
    visible = torch.zeros(Fq, Fq + H, dtype=torch.bool, device=device)
    for r in range(R + 1):
        visible[r, r] = True
    for j in range(H):
        visible[sink + 1 + j, :sink + 1] = True
        visible[sink + 1 + j, Fq + j] = True
    for i in range(P):
        visible[sink + 1 + H + i, :sink + 1] = True
        visible[sink + 1 + H + i, sink + 1 + i:sink + 2 + H + i] = True

    # Token-level table: the kernel evaluates mask_mod per score in partial blocks (most blocks, 120-token frames
    # against 128-token blocks), and one load is far cheaper than two integer divisions by 120.
    dense = visible[torch.arange(Fq * N, device=device)[:, None] // N, torch.arange((Fq + H) * N, device=device)[None, :] // N]

    def mask_mod(b, head, q_idx, kv_idx):
        return dense[q_idx, kv_idx]

    block_mask = create_block_mask(mask_mod, None, None, Fq * N, (Fq + H) * N, device=device)
    table = lambda positions: torch.view_as_real(
        rope.build_freqs_block(freqs, torch.tensor(positions, device=device), h, w)).contiguous()
    _GEOMETRY[key] = table(positions_q), table(positions_k), block_mask
    return _GEOMETRY[key]


class TransformerLayer(nn.Module):
    def __init__(self, block, audio_block, freqs, num_history, num_present, compile_block, checkpoint, num_refs=0,
                 flex=False, hidden_block=None):
        super().__init__()
        self.block = block
        self.audio_block = audio_block
        self.hidden_block = hidden_block
        self.freqs = freqs
        self.checkpoint = checkpoint
        self.num_history = num_history
        self.num_present = num_present
        self.num_frames = 1 + num_history + num_present
        self.num_clean = 1 + num_history
        self.num_refs = num_refs
        self.flex = flex
        present_keys = [
            [0, *range(1 + index, 1 + num_history + index), 1 + num_history + index]
            for index in range(num_present)
        ]
        self.register_buffer('present_key_indices', torch.tensor(present_keys), persistent=False)
        self._rope_cache = {}
        # Running sum of per-slot injection ratios over training microbatches; the trainer averages and resets.
        self.audio_injection_ratio = 0
        self.audio_injection_count = 0
        self._run = torch.compile(self._forward_impl) if compile_block else self._forward_impl
        self._attend = flex_attention if compile_block else torch.compile(flex_attention, dynamic=False)

    def _causal_attention(self, x, tokens_per_frame, *attention_args):
        attention = self.block.self_attn
        B, S = x.shape[:2]
        n = attention.num_heads
        d = attention.head_dim

        q = attention.norm_q(attention.q(x)).view(B, S, n, d)
        k = attention.norm_k(attention.k(x)).view(B, S, n, d)
        v = attention.v(x).view(B, S, n, d)
        if self.flex:
            output = self._flex_attention(q, k, v, tokens_per_frame, *attention_args)
        else:
            freqs_i, *ref_freqs = attention_args
            output = self._fixed_attention(q, k, v, freqs_i, tokens_per_frame, *ref_freqs)
        return attention.o(output.flatten(2))

    def _flex_attention(self, q, k, v, N, freqs_q, freqs_k, block_mask):
        R, H = self.num_refs, self.num_history
        history = slice((R + 1) * N, (R + 1 + H) * N)
        k = torch.cat((k, k[:, history]), dim=1)
        v = torch.cat((v, v[:, history]), dim=1)
        output_dtype = q.dtype
        q = rope.apply_rope(q, freqs_q).bfloat16().transpose(1, 2)
        k = rope.apply_rope(k, freqs_k).bfloat16().transpose(1, 2)
        v = v.bfloat16().transpose(1, 2)
        return self._attend(q, k, v, block_mask=block_mask).transpose(1, 2).to(output_dtype)

    def _fixed_attention(self, q, k, v, freqs_i, N, ref_freqs=None):
        """
        Sink: itself. History frame: the sink and itself, with the sink read at the fixed offset
        num_history (the frame is rotated as if in the newest history slot), so a history frame's
        representation does not depend on the slot it currently occupies and can be cached once.
        Present slot i: sink, the num_history frames before it, itself, at their true slots.
        Refs (num_refs frames before the sink, all at temporal position 0 via ref_freqs, with freqs_i then
        starting at WINDOW_OFFSET so the sink sits WINDOW_OFFSET after them): each ref attends only to itself,
        like the sink, and every history and present frame also reads all refs, as extra sinks.
        """
        B, H, R = q.shape[0], self.num_history, self.num_refs
        if R:
            ref_q, ref_k, ref_v = (z[:, :R * N].unflatten(1, (R, N)).flatten(0, 1) for z in (q, k, v))
            q, k, v = (z[:, R * N:] for z in (q, k, v))
            ref_k = rope.apply_rope(ref_k, ref_freqs).bfloat16()
            ref_v = ref_v.bfloat16()
            refs = flash_attn_func(rope.apply_rope(ref_q, ref_freqs).bfloat16(), ref_k, ref_v).unflatten(0, (B, R))
            ref_k = ref_k.unflatten(0, (B, R)).flatten(1, 2)
            ref_v = ref_v.unflatten(0, (B, R)).flatten(1, 2)
            per_frame = lambda z, n: z.unsqueeze(1).expand(B, n, *z.shape[1:]).flatten(0, 1)
        q_frames = rope.apply_rope(q, freqs_i).unflatten(1, (self.num_frames, N))
        output_dtype = q_frames.dtype
        q_frames = q_frames.bfloat16()
        k_frames = rope.apply_rope(k, freqs_i).bfloat16().unflatten(1, (self.num_frames, N))
        v_frames = v.bfloat16().unflatten(1, (self.num_frames, N))

        sink = flash_attn_func(q_frames[:, 0], k_frames[:, 0], v_frames[:, 0])

        anchor = freqs_i[H * N:(H + 1) * N]
        history_q = rope.apply_rope(q.unflatten(1, (self.num_frames, N))[:, 1:1 + H].flatten(0, 1), anchor).bfloat16()
        history_k = rope.apply_rope(k.unflatten(1, (self.num_frames, N))[:, 1:1 + H].flatten(0, 1), anchor).bfloat16()
        sink_k = k_frames[:, :1].expand(B, H, N, *k_frames.shape[3:]).flatten(0, 1)
        sink_v = v_frames[:, :1].expand(B, H, N, *v_frames.shape[3:]).flatten(0, 1)
        history_k = torch.cat((sink_k, history_k), dim=1)
        history_v = torch.cat((sink_v, v_frames[:, 1:1 + H].flatten(0, 1)), dim=1)
        if R:
            history_k = torch.cat((per_frame(ref_k, H), history_k), dim=1)
            history_v = torch.cat((per_frame(ref_v, H), history_v), dim=1)
        history = flash_attn_func(history_q, history_k, history_v)

        present_q = q_frames[:, self.num_clean:].flatten(0, 1)
        present_k = k_frames[:, self.present_key_indices].flatten(2, 3).flatten(0, 1)
        present_v = v_frames[:, self.present_key_indices].flatten(2, 3).flatten(0, 1)
        if R:
            present_k = torch.cat((per_frame(ref_k, self.num_present), present_k), dim=1)
            present_v = torch.cat((per_frame(ref_v, self.num_present), present_v), dim=1)
        present = flash_attn_func(present_q, present_k, present_v)

        history = history.unflatten(0, (B, H))
        present = present.unflatten(0, (B, self.num_present))
        frames = (sink.unsqueeze(1), history, present)
        if R:
            frames = (refs, *frames)
        return torch.cat(frames, dim=1).flatten(1, 2).to(output_dtype)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, grid, audio_encoded, present_valid, hiddens = inputs
        if self.flex:
            attention_args = sink_geometry(self.freqs, grid, self.num_refs, self.num_history, self.num_present, x.device)
        else:
            freqs_i = rope.cached_freqs_i(self._rope_cache, grid, self.freqs, WINDOW_OFFSET if self.num_refs else 0)
            refs = (rope.cached_freqs_i(self._rope_cache, (1, *grid[1:]), self.freqs),) if self.num_refs else ()
            attention_args = (freqs_i, *refs)
        x, injection_ratio = self._run(x, e0, grid[1] * grid[2], audio_encoded, present_valid, hiddens, *attention_args)
        if injection_ratio is not None and self.training:
            self.audio_injection_ratio = self.audio_injection_ratio + injection_ratio
            self.audio_injection_count += 1
        return x, e, e0, grid, audio_encoded, present_valid, hiddens

    def _forward_impl(self, x, e0, tokens_per_frame, audio_encoded, present_valid, hiddens, *attention_args):
        frames, clean = self.num_refs + self.num_frames, self.num_refs + self.num_clean
        x = x.unflatten(1, (frames, tokens_per_frame))
        modulation = (self.block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
        normalized = self.block.norm1(x) * (1 + modulation[1]) + modulation[0]
        attention = self._causal_attention(normalized.flatten(1, 2), tokens_per_frame, *attention_args)
        x = x + attention.unflatten(1, (frames, tokens_per_frame)) * modulation[2]

        injection_ratio = None
        if self.audio_block is not None:
            present = x[:, clean:]
            audio = self.audio_block.forward_present(present, audio_encoded)
            audio = audio * present_valid[:, :, None, None].to(audio.dtype)
            # Per present slot, ||audio update|| / ||present residual stream||: how much the block injects.
            audio_norm = audio.detach().float().square().sum(dim=(0, 2, 3)).sqrt()
            present_norm = present.detach().float().square().sum(dim=(0, 2, 3)).sqrt()
            injection_ratio = audio_norm / present_norm.clamp_min(1e-6)
            x = torch.cat((x[:, :clean], present + audio), dim=1)

        if self.hidden_block is not None:
            present = x[:, clean:]
            text = self.hidden_block.forward_present(present, hiddens)
            x = torch.cat((x[:, :clean], present + text * present_valid[:, :, None, None].to(text.dtype)), dim=1)

        residual = self.block.ffn(
            self.block.norm2(x) * (1 + modulation[4]) + modulation[3]
        )
        return (x + residual * modulation[5]).flatten(1, 2), injection_ratio


class FinalLayer(nn.Module):
    """Refs are dropped before the head: the output is the window's frames only."""

    def __init__(self, model, num_refs=0):
        super().__init__()
        self.head = model.head
        self.unpatchify = model.unpatchify
        self.num_refs = num_refs

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, _, grid, *_ = inputs
        if self.num_refs:
            x = x[:, self.num_refs * grid[1] * grid[2]:]
            e = e[:, self.num_refs:]
        x = x.unflatten(1, (e.shape[1], grid[1] * grid[2]))
        with torch.autocast('cuda', dtype=torch.float32):
            modulation = (self.head.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head.head(self.head.norm(x) * (1 + modulation[1]) + modulation[0])
        return self.unpatchify(x.flatten(1, 2), grid)
