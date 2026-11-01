"""
The bimodal network as one functional forward over arbitrary frame layouts, for distillation.

A pass runs F frames through the shared weights at once: every Linear / FFN / AdaLN / audio op on the concatenated
tokens, attention per group of frames with the same key count. A frame is (latent, pose or None, t [B], slot,
valid [B]); its keys are listed per frame in the plan, each one of
  ('f', g)          this pass's frame g, K rotated at g's own slot,
  ('r', g, slot)    this pass's frame g, K rotated at `slot` (a history frame read by a present frame),
  ('e', k, slot)    external K before RoPE ext[2k] (rotated here at `slot`) and V ext[2k + 1], from an earlier pass.
Each pass returns every frame's per-layer K before RoPE and V, so later passes can read them (the rollout's cache,
kept differentiable). This one forward covers the student rollout (bimodal/student_rollout.py's fused schedule),
the stabilising window [sink | 3 history | 4 present] and the packed resolving windows used for scoring.
"""

import copy
import re

import torch
import torch.distributed as dist
from flash_attn_interface import flash_attn_func
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import MimiModel

from distillation.lora import fused_lora_qkv
from models.pose import rope
from models.pose.audio import AudioAdapter, MimiAudioEncoder, audio_latents_from_codes
from models.pose.settings import wan_5b_wide
from models.wan.model import WanModel, sinusoidal_embedding_1d
from utils.common import AUTOCAST_DTYPE, iterate_safetensors, load_weights_from_state_dict

MAX_SLOTS = 8  # [sink | 3 history | 4 present]
CHECKPOINT_SKIP_BLOCKS = ()
TORCH_COMPILE_MODE = 'default'


class BimodalNet(nn.Module):
    def __init__(self, transformer, encoder, adapter, pose_patch_embedding, pad_token, latent_hw, audio_codes):
        super().__init__()
        self.transformer, self.audio_encoder, self.audio_adapter = transformer, encoder, adapter
        self.pose_patch_embedding, self.pad_token = pose_patch_embedding, pad_token
        self.audio_blocks = [adapter.get_block(i) for i in range(len(transformer.blocks))]
        attention = transformer.blocks[0].self_attn
        self.n, self.d = attention.num_heads, attention.head_dim
        patch = transformer.patch_size
        self.grid = (1, latent_hw[0] // patch[1], latent_hw[1] // patch[2])
        self.N = self.grid[1] * self.grid[2]
        self.K = audio_codes
        freqs = rope.cached_freqs_i({}, (MAX_SLOTS, *self.grid[1:]), transformer.freqs, 0)
        self.register_buffer('slot_freqs', freqs.unflatten(0, (MAX_SLOTS, self.N)), persistent=False)
        self.block_fn = self._block
        self.checkpoint = 'block'
        self.checkpoint_skip_blocks = frozenset(CHECKPOINT_SKIP_BLOCKS)
        self.shards = None

    def original_named_parameters(self):
        for name, parameter in self.named_parameters():
            yield name.removeprefix('transformer.'), parameter

    @torch.no_grad()
    def shard_blocks(self, group):
        """
        Frozen nets only: each transformer block (+ its audio block) is kept as this rank's 1/group shard of a bf16
        flat; every block's parameters view one shared gather buffer, filled by an all-gather over `group` (one node,
        NVLink) right before the block runs.
        """
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        self.requires_grad_(False)  # block parameters become bf16 views (audio blocks too); the rest stays as loaded
        layers = []
        for i, block in enumerate(self.transformer.blocks):
            params = list(block.parameters()) + (list(self.audio_blocks[i].parameters()) if self.audio_blocks[i] else [])
            layers.append(params)
        size = max(sum(p.numel() for p in params) for params in layers)
        size = -(-size // world) * world
        device = self.slot_freqs.device
        self.gathered = torch.empty(size, dtype=torch.bfloat16, device=device)
        self.shards = []
        for params in layers:
            flat = torch.zeros(size, dtype=torch.bfloat16, device=device)
            offset = 0
            for p in params:
                flat[offset:offset + p.numel()].copy_(p.data.flatten())
                p.data = self.gathered[offset:offset + p.numel()].view_as(p)
                offset += p.numel()
            self.shards.append(flat.chunk(world)[rank].clone())
            del flat
        self.shard_group = group

    def gather_block(self, i):
        if self.shards is not None:
            dist.all_gather_into_tensor(self.gathered, self.shards[i], group=self.shard_group)

    def configure(self, compile_blocks, checkpoint_mode, compile_mode='parts', checkpoint_skip_blocks=CHECKPOINT_SKIP_BLOCKS):
        """compile_mode 'parts': the layout-independent pieces with dynamic shapes (any layout, few graphs);
        'block': whole static blocks; 'block_cudagraph': also CUDA graphs, requiring one
        torch.compiler.cudagraph_mark_step_begin() before each complete forward/backward iteration."""
        assert compile_mode in ('parts', 'block', 'block_cudagraph'), compile_mode
        if compile_mode == 'block_cudagraph' and not compile_blocks:
            raise ValueError('block_cudagraph requires compile_blocks')
        self.block_fn = self._block
        if compile_blocks and compile_mode == 'parts' and PRE is block_pre:
            use_compiled()
        if compile_blocks and compile_mode in ('block', 'block_cudagraph'):
            fn = compiled_whole_block('reduce-overhead' if compile_mode == 'block_cudagraph' else TORCH_COMPILE_MODE)
            self.block_fn = lambda *args: fn(self.slot_freqs, self.n, self.d, *args)
        assert checkpoint_mode in ('block', 'call', 'none'), checkpoint_mode
        if checkpoint_skip_blocks and checkpoint_mode != 'block':
            raise ValueError('checkpoint_skip_blocks requires block checkpointing')
        if any(type(i) is not int or not 0 <= i < len(self.transformer.blocks) for i in checkpoint_skip_blocks):
            raise ValueError('checkpoint_skip_blocks contains an invalid block index')
        self.checkpoint = checkpoint_mode
        self.checkpoint_skip_blocks = frozenset(checkpoint_skip_blocks)

    # ---- audio ----
    def audio_tokens(self, audio_latents):
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
            return self.audio_encoder(audio_latents)

    def frame_audio(self, tokens, ofi, drop):
        """tokens [B, T, D], ofi [B, F] (shifted frame indices, as training's _window_audio), drop [B] -> [B, F, K+1, D]."""
        K, D = self.K, tokens.shape[-1]
        positions = (K * (ofi.clamp(min=1) - 1)).unsqueeze(-1) + torch.arange(K, device=tokens.device)
        gathered = torch.gather(tokens[:, None].expand(-1, ofi.shape[1], -1, -1), 2,
                                positions.unsqueeze(-1).expand(-1, -1, -1, D))
        null = self.audio_encoder.null_token.to(gathered.dtype).reshape(1, 1, 1, D)
        audio = torch.cat((gathered, null.expand(gathered.shape[0], gathered.shape[1], 1, D)), dim=2)
        no_audio = self.audio_encoder.no_audio_tokens.to(audio.dtype)
        return torch.where(drop.view(-1, 1, 1, 1), no_audio, audio)

    # ---- one pass ----
    def embed(self, latents, poses, t, valid, posed):
        """latents / poses [B, C, F, h, w], t [B, F] (not * 1000), valid / posed [B, F] bool."""
        tr, N = self.transformer, self.N
        F = latents.shape[2]
        x = tr.patch_embedding(latents).flatten(2).transpose(1, 2).unflatten(1, (F, N))
        x = torch.where(valid[:, :, None, None], x, self.pad_token.to(x.dtype))
        pose = self.pose_patch_embedding(poses.to(x.dtype)).flatten(2).transpose(1, 2).unflatten(1, (F, N))
        x = x + pose * posed[:, :, None, None].to(pose.dtype)
        e = tr.time_embedding(sinusoidal_embedding_1d(tr.freq_dim, (t * 1000).flatten()).unflatten(0, t.shape))
        return x, e, tr.time_projection(e).unflatten(2, (6, tr.dim))

    def _block(self, *args):
        return block_forward(self.slot_freqs, self.n, self.d, *args)

    def _pass(self, latents, poses, t, valid, posed, slots, plan, heads, keep, audio, audio_mask, *ext_flat):
        num_ext = len(ext_flat) // len(self.transformer.blocks)
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
            x, e, e0 = self.embed(latents, poses, t, valid, posed)
            freqs = self.slot_freqs[list(slots)].flatten(0, 1)
            ks, vs = [], []
            for i, block in enumerate(self.transformer.blocks):
                self.gather_block(i)
                args = (block, self.audio_blocks[i], plan, heads, keep, x, e0, freqs, audio, audio_mask,
                        *ext_flat[i * num_ext:(i + 1) * num_ext])
                if self.checkpoint == 'block' and i not in self.checkpoint_skip_blocks and torch.is_grad_enabled():
                    outputs = list(checkpointed(self.block_fn, *args))
                    x, k, v = outputs if keep else (outputs[0], None, None)
                else:
                    x, k, v = self.block_fn(*args)
                ks.append(k)
                vs.append(v)
            velocity = None
            if heads:
                head = self.transformer.head
                with torch.autocast('cuda', dtype=torch.float32):
                    modulation = (head.modulation.unsqueeze(0) + e[:, :heads].unsqueeze(2)).chunk(2, dim=2)
                    y = head.head(head.norm(x[:, :heads]) * (1 + modulation[1]) + modulation[0])
                velocity = WanModel.unpatchify(self.transformer, y.flatten(1, 2), (heads, *self.grid[1:])).float()
        return velocity, ks, vs

    def forward_pass(self, frames, plan, ext, audio, audio_mask, heads, keep=()):
        """
        frames: list of (latent [B, C, h, w], pose [B, C, h, w] or None, t [B] or float, slot, valid [B] or bool),
        attention groups contiguous in plan order (plan: [(first frame, specs per frame)]), the first `heads` frames
        the ones with a velocity (and audio [B, heads, K+1, D], audio_mask [B, heads]).
        ext: per layer, a flat list [K raw, V, K raw, V, ...] of external frames.
        keep: the frames whose per-layer K/V to return.
        -> velocity [B, C, heads, h, w] or None, per-layer lists of K before RoPE and V [B, len(keep), N, n, d]
        """
        latent = frames[0][0]
        B, device = latent.shape[0], latent.device
        full = lambda value, dtype: (value if torch.is_tensor(value) else
                                     torch.full((B,), value, dtype=dtype, device=device))
        latents = torch.stack([f[0] for f in frames], dim=2)
        poses = torch.stack([f[1] if f[1] is not None else torch.zeros_like(f[0]) for f in frames], dim=2)
        t = torch.stack([full(f[2], torch.float32).float() for f in frames], dim=1)
        valid = torch.stack([full(f[4], torch.bool) for f in frames], dim=1)
        posed = valid & torch.tensor([f[1] is not None for f in frames], device=device)[None]
        slots = tuple(f[3] for f in frames)
        ext_flat = [tensor for layer in ext for tensor in layer] if ext and ext[0] else []
        args = (latents, poses, t, valid, posed, slots, plan, heads, tuple(keep), audio, audio_mask, *ext_flat)
        if self.checkpoint == 'call' and torch.is_grad_enabled():
            return checkpoint(self._pass, *args, use_reentrant=False)
        return self._pass(*args)


def block_pre(block, x, e0, freqs, n, d):
    """Layout-independent: AdaLN, q/k/v, RoPE at the frames' own slots. -> q, k (rotated), k before RoPE, v, gates."""
    B, F, N = x.shape[0], x.shape[1], x.shape[2]
    attention = block.self_attn
    modulation = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
    normalized = (block.norm1(x) * (1 + modulation[1]) + modulation[0]).flatten(1, 2)
    if getattr(attention, 'lora_fused_qkv', False):
        q, k, v = fused_lora_qkv(attention, normalized)
    else:
        q, k, v = attention.q(normalized), attention.k(normalized), attention.v(normalized)
    q = rope.apply_rope(attention.norm_q(q).view(B, F * N, n, d), freqs)
    k_raw = attention.norm_k(k).view(B, F * N, n, d)
    k = rope.apply_rope(k_raw, freqs).bfloat16().view(B, F, N, n, d)
    v = v.view(B, F, N, n, d).bfloat16()
    return (q.bfloat16().view(B, F, N, n, d), k, k_raw.view(B, F, N, n, d), v,
            modulation[2], modulation[3], modulation[4], modulation[5])


def block_post(block, audio_block, x, out, gate, shift, scale, gate_ffn, audio, audio_mask):
    """Layout-independent: output projection, audio cross-attention on the first audio.shape[1] frames, FFN."""
    x = x + block.self_attn.o(out.to(x.dtype).flatten(3)) * gate
    if audio_block is not None and audio is not None:
        heads = audio.shape[1]
        update = audio_block.forward_present(x[:, :heads], audio) * audio_mask[:, :, None, None].to(x.dtype)
        x = torch.cat((x[:, :heads] + update, x[:, heads:]), dim=1)
    return x + block.ffn(block.norm2(x) * (1 + scale) + shift) * gate_ffn


def rope_bf16(x, freqs):
    return rope.apply_rope(x, freqs).bfloat16()


PRE, POST, ROPE = block_pre, block_post, rope_bf16


def use_compiled():
    """One set of dynamic-shape graphs (token count, batch) shared by every layout, net and batch size."""
    global PRE, POST, ROPE
    torch._inductor.config.triton.coalesce_tiling_analysis = False  # asserts under dynamic shapes (torch 2.11)
    PRE = torch.compile(block_pre, dynamic=True)
    POST = torch.compile(block_post, dynamic=True)
    ROPE = torch.compile(rope_bf16, dynamic=True)


_WHOLE = {}


def compiled_whole_block(mode=TORCH_COMPILE_MODE):
    if mode not in _WHOLE:
        _WHOLE[mode] = torch.compile(block_forward, dynamic=False, mode=mode)
    return _WHOLE[mode]


def block_forward(slot_freqs, n, d, block, audio_block, plan, heads, keep, x, e0, freqs, audio, audio_mask, *ext):
    """One transformer block of a pass: compiled dense parts around the layout-specific attention (eager)."""
    B, F, N = x.shape[0], x.shape[1], x.shape[2]
    q, k, k_raw, v, gate, shift, scale, gate_ffn = PRE(block, x, e0, freqs, n, d)
    rotated = {}

    def key(s):
        if s[0] == 'f':
            return k[:, s[1]]
        if s not in rotated:
            raw = k_raw[:, s[1]] if s[0] == 'r' else ext[2 * s[1]]
            rotated[s] = ROPE(raw, slot_freqs[s[2]])
        return rotated[s]

    def value(s):
        return ext[2 * s[1] + 1] if s[0] == 'e' else v[:, s[1]]

    outputs = []
    for start, specs in plan:
        G = len(specs)
        keys = torch.stack([torch.cat([key(s) for s in spec], dim=1) for spec in specs], dim=1)
        values = torch.stack([torch.cat([value(s) for s in spec], dim=1) for spec in specs], dim=1)
        out = flash_attn_func(q[:, start:start + G].flatten(0, 1), keys.flatten(0, 1), values.flatten(0, 1))
        outputs.append(out.view(B, G, N, n, d))
    out = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
    x = POST(block, audio_block, x, out, gate, shift, scale, gate_ffn, audio if heads else None, audio_mask)
    # Only the frames later passes read (keep), as their own tensors: no gradient buffers for the rest.
    if not keep:
        return x, None, None
    return x, torch.stack([k_raw[:, f] for f in keep], dim=1), torch.stack([v[:, f] for f in keep], dim=1)


class _BlockCheckpoint(torch.autograd.Function):
    """
    Block recompute through the same (compiled) functions: forward without autograd, backward re-runs the block with
    autograd and backpropagates it on its own. Unlike torch.utils.checkpoint, the recompute is not run with Dynamo
    disabled (which saves different tensors than the compiled forward), and each block's weight gradients are
    accumulated as soon as that block is done instead of being summed over the whole rollout first.
    """

    @staticmethod
    def forward(ctx, run, autocast_dtype, anchor, *tensors):
        ctx.run, ctx.autocast_dtype = run, autocast_dtype
        ctx.save_for_backward(*tensors)
        with torch.no_grad():
            outputs = run(*tensors)
        ctx.present = [o is not None for o in outputs]
        return tuple(o for o in outputs if o is not None)

    @staticmethod
    def backward(ctx, *grads):
        inputs = [t.detach().requires_grad_(t.requires_grad) for t in ctx.saved_tensors]
        with torch.enable_grad(), torch.autocast('cuda', dtype=ctx.autocast_dtype):
            outputs = [o for o in ctx.run(*inputs) if o is not None]
        pairs = [(o, g) for o, g in zip(outputs, grads) if g is not None and o.requires_grad]
        torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        return (None, None, None) + tuple(i.grad if i.requires_grad else None for i in inputs)


def checkpointed(fn, *args):
    """fn(*args) -> tuple, recomputed in backward; non-tensor args are closed over."""
    positions = [i for i, a in enumerate(args) if torch.is_tensor(a)]

    def run(*tensors):
        full = list(args)
        for i, t in zip(positions, tensors):
            full[i] = t
        return fn(*full)

    tensors = [args[i] for i in positions]
    # Frozen embeddings can leave the first LoRA block without a differentiable input.
    anchor = None if any(t.requires_grad for t in tensors) else next(
        (p for arg in args if isinstance(arg, nn.Module) for p in arg.parameters() if p.requires_grad), None)
    return _BlockCheckpoint.apply(run, AUTOCAST_DTYPE, anchor, *tensors)


def load_state_dict(path):
    return {re.sub(r'^model\.diffusion_model\.', '', key): tensor for key, tensor in iterate_safetensors(path)}


def build_net(state_dict, audio_config, latent_hw, device, mimi_template):
    """The trained bimodal weights as one BimodalNet: transformer bf16, audio / pose / pad token fp32, as in training."""
    arch = wan_5b_wide
    K = audio_config['cross_attention_codes']
    with torch.device('meta'):
        transformer = WanModel(
            patch_size=arch.patch_size, in_dim=arch.in_dim, dim=arch.dim, ffn_dim=arch.ffn_dim,
            freq_dim=arch.freq_dim, out_dim=arch.out_dim, num_heads=arch.num_heads,
            num_layers=arch.num_layers, qk_norm=arch.qk_norm, eps=arch.eps,
        )
    load_weights_from_state_dict(transformer, state_dict, torch.bfloat16, torch.bfloat16)
    audio_transformer = copy.deepcopy(mimi_template)
    audio_transformer.layers = audio_transformer.layers[:audio_config['num_layers']]
    encoder = MimiAudioEncoder(transformer=audio_transformer, in_dim=audio_config['in_dim'], hidden_dim=arch.dim,
                               drop=audio_config['drop'], num_window_tokens=K + 1)
    adapter = AudioAdapter(hidden_dim=arch.dim, num_attn_heads=arch.num_heads, num_layers=arch.num_layers,
                           injection_offset=audio_config['injection_offset'],
                           injection_interval=audio_config['injection_interval'],
                           initialization='random', max_blocks=audio_config['max_blocks'])
    pose = nn.Conv3d(arch.in_dim, arch.dim, kernel_size=arch.patch_size, stride=arch.patch_size)
    for module, prefix in ((encoder, 'audio_encoder.'), (adapter, 'audio_adapter.'), (pose, 'pose_patch_embedding.')):
        module.load_state_dict({k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)})
    pad_token = nn.Parameter(state_dict['pad_token'].float().clone())
    transformer.to(device)
    for module in (encoder, adapter, pose):
        module.to(device)
    net = BimodalNet(transformer, encoder, adapter, pose, pad_token, latent_hw, K).to(device)
    names = {name for name, _ in net.original_named_parameters()}
    missing = names - set(state_dict)
    assert not missing, f'checkpoint lacks {sorted(missing)[:10]}'
    return net


def load_mimi(audio_config, device):
    """The frozen Mimi frontend (codes -> latents) and the decoder-transformer template of the audio encoders."""
    mimi = MimiModel.from_pretrained(audio_config['model_name']).eval().requires_grad_(False)
    template = copy.deepcopy(mimi.decoder_transformer)
    return mimi.to(device), template


@torch.no_grad()
def audio_latents(mimi, codes, num_codebooks):
    return audio_latents_from_codes(mimi, codes, num_codebooks)
