"""
The bimodal network as one functional forward over arbitrary frame layouts, for distillation.

A pass runs F frames through the shared weights at once: every Linear / FFN / AdaLN / audio op on the concatenated
tokens, attention per group of frames with the same key count. A frame is (latent, pose or None, t [B], slot,
valid [B]); its keys are listed per frame in the plan, each one of
  ('f', g)          this pass's frame g, K rotated at g's own slot,
  ('r', g, slot)    this pass's frame g, K rotated at `slot` (a history frame read by a present frame),
  ('e', k, slot)    external K before RoPE ext[2k] (rotated here at `slot`) and V ext[2k + 1], from an earlier pass.
  ('c', k)          external K already rotated, and V; used for one packed reference bank per layer.
  ('p', g, slot)    frame g in packed external K/V ext[0:2], rotated at `slot`.
  ('m', g, slot, k) choose local frame g or external frame k with mix_mask[:, g], rotating at `slot`.
Each pass returns every frame's per-layer K before RoPE and V, so later passes can read them (the rollout's cache,
kept differentiable). This one forward covers the student rollout (bimodal/student_rollout.py's fused schedule),
the stabilising window [sink | 3 history | 4 present] and the packed resolving windows used for scoring.
"""

import copy
from contextlib import contextmanager
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
REFERENCE_WINDOW_OFFSET = 10  # models.pose.wan_universal.WINDOW_OFFSET
CHECKPOINT_SKIP_BLOCKS = ()
TORCH_COMPILE_MODE = 'default'


class BimodalNet(nn.Module):
    def __init__(self, transformer, encoder, adapter, pose_patch_embedding, pad_token, latent_hw, audio_codes,
                 num_refs=0, ref_patch_embedding=None):
        super().__init__()
        self.transformer, self.audio_encoder, self.audio_adapter = transformer, encoder, adapter
        self.pose_patch_embedding, self.pad_token = pose_patch_embedding, pad_token
        self.num_refs, self.ref_patch_embedding = num_refs, ref_patch_embedding
        self._references = None
        self._reference_cache = None
        self.cache_references = True
        self.reference_checkpoint = None
        if bool(num_refs) != (ref_patch_embedding is not None):
            raise ValueError("Reference conditioning requires its trained patch embedding")
        self.audio_blocks = [adapter.get_block(i) for i in range(len(transformer.blocks))]
        attention = transformer.blocks[0].self_attn
        self.n, self.d = attention.num_heads, attention.head_dim
        patch = transformer.patch_size
        self.grid = (1, latent_hw[0] // patch[1], latent_hw[1] // patch[2])
        self.N = self.grid[1] * self.grid[2]
        self.K = audio_codes
        freqs = rope.cached_freqs_i({}, (MAX_SLOTS, *self.grid[1:]), transformer.freqs,
                                   REFERENCE_WINDOW_OFFSET if num_refs else 0)
        freqs = freqs.unflatten(0, (MAX_SLOTS, self.N))
        if num_refs:
            ref_freqs = rope.cached_freqs_i({}, self.grid, transformer.freqs, 0).unsqueeze(0)
            freqs = torch.cat((freqs, ref_freqs), dim=0)
        self.register_buffer('slot_freqs', freqs, persistent=False)
        self.block_fn = self._block
        self.checkpoint = 'block'
        self.checkpoint_skip_blocks = frozenset(CHECKPOINT_SKIP_BLOCKS)
        self.shards = None

    @contextmanager
    def references(self, latents, keep):
        """Bind reference inputs for a sequence of forwards; checkpoint inputs remain explicit.

        Nine-reference checkpoints retain their separate reference patch embedding,
        temporal RoPE zero for every reference, and window RoPE offset ten. Refs
        attend only themselves; history/present attend all refs; sink stays self-only.
        """
        if not self.num_refs:
            if latents is not None:
                raise ValueError('This network has no reference conditioning')
        elif (latents is None or latents.ndim != 5 or latents.shape[2] != self.num_refs or
              keep is None or keep.dtype != torch.bool or keep.shape != (latents.shape[0], self.num_refs)):
            raise ValueError('Expected reference latents [B,C,R,h,w] and boolean ref_keep [B,R]')
        previous, previous_cache = self._references, self._reference_cache
        self._references = (latents, keep) if self.num_refs else None
        self._reference_cache = None
        try:
            if self.num_refs and self.cache_references:
                # Build in the enclosing gradient mode, before any detached
                # bootstrap forwards. The explicit K/V inputs keep gradients
                # connected through every subsequent checkpointed target pass.
                # Nine reference frames can dominate activation memory even
                # when a small target pass fits without checkpointing. Reuse
                # the same compiled block, but choose prefill recomputation
                # independently. The checkpoint closure captures _run_block;
                # backward does not consult this mutable policy again.
                previous_checkpoint = self.checkpoint
                if self.reference_checkpoint not in (None, 'block', 'none'):
                    raise ValueError('reference_checkpoint must be block, none, or None')
                try:
                    if self.reference_checkpoint is not None:
                        self.checkpoint = self.reference_checkpoint
                    self._reference_cache = self._prefill_references(latents, keep)
                finally:
                    self.checkpoint = previous_checkpoint
            yield self
        finally:
            self._references = previous
            self._reference_cache = previous_cache

    def _prefill_references(self, latents, keep):
        B, _, R, _, _ = latents.shape
        times = torch.zeros((B, R), device=latents.device, dtype=torch.float32)
        plan = [(0, [[('f', index)] for index in range(R)])]
        _, keys, values = self._pass(
            latents, torch.zeros_like(latents), times, keep, torch.zeros_like(keep),
            (MAX_SLOTS,) * R, plan, 0, tuple(range(R)), None, None, R, None)
        freqs = self.slot_freqs[[MAX_SLOTS] * R].flatten(0, 1)
        return [[ROPE(key.flatten(1, 2), freqs), value.flatten(1, 2)] for key, value in zip(keys, values)]

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
    def embed(self, latents, poses, t, valid, posed, reference_count=0):
        """latents / poses [B, C, F, h, w], t [B, F] (not * 1000), valid / posed [B, F] bool."""
        tr, N = self.transformer, self.N
        F = latents.shape[2]
        if reference_count:
            refs = self.ref_patch_embedding(latents[:, :, -reference_count:])
            x = (refs if F == reference_count else
                 torch.cat((tr.patch_embedding(latents[:, :, :-reference_count]), refs), dim=2))
        else:
            x = tr.patch_embedding(latents)
        x = x.flatten(2).transpose(1, 2).unflatten(1, (F, N))
        x = torch.where(valid[:, :, None, None], x, self.pad_token.to(x.dtype))
        pose = self.pose_patch_embedding(poses.to(x.dtype)).flatten(2).transpose(1, 2).unflatten(1, (F, N))
        x = x + pose * posed[:, :, None, None].to(pose.dtype)
        e = tr.time_embedding(sinusoidal_embedding_1d(tr.freq_dim, (t * 1000).flatten()).unflatten(0, t.shape))
        return x, e, tr.time_projection(e).unflatten(2, (6, tr.dim))

    def _block(self, *args):
        return block_forward(self.slot_freqs, self.n, self.d, *args)

    def _run_block(self, block, audio_block, plan, heads, keep, x, e0, freqs, audio, audio_mask,
                   mix_mask, *ext):
        if mix_mask is not None:
            # Pack inside the checkpoint, immediately before the compiled
            # function. Saving these copies across the whole rollout would
            # duplicate eight temporal K/V frames per layer and target pass.
            # The checkpoint instead retains the original shared K/V tensors;
            # backward reconstructs the same stable, non-aliased packed inputs.
            ext = (torch.stack(ext[:2*MAX_SLOTS:2], dim=1),
                   torch.stack(ext[1:2*MAX_SLOTS:2], dim=1), *ext[2*MAX_SLOTS:])
        return self.block_fn(block, audio_block, plan, heads, keep, x, e0, freqs,
                             audio, audio_mask, mix_mask, *ext)

    def _pass(self, latents, poses, t, valid, posed, slots, plan, heads, keep, audio, audio_mask,
              reference_count, mix_mask, *ext_flat):
        num_ext = len(ext_flat) // len(self.transformer.blocks)
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
            x, e, e0 = self.embed(latents, poses, t, valid, posed, reference_count)
            freqs = self.slot_freqs[list(slots)].flatten(0, 1)
            ks, vs = [], []
            for i, block in enumerate(self.transformer.blocks):
                self.gather_block(i)
                args = (block, self.audio_blocks[i], plan, heads, keep, x, e0, freqs, audio, audio_mask, mix_mask,
                        *ext_flat[i * num_ext:(i + 1) * num_ext])
                if self.checkpoint == 'block' and i not in self.checkpoint_skip_blocks and torch.is_grad_enabled():
                    outputs = list(checkpointed(self._run_block, *args))
                    x, k, v = outputs if keep else (outputs[0], None, None)
                else:
                    x, k, v = self._run_block(*args)
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

    def forward_pass(self, frames, plan, ext, audio, audio_mask, heads, keep=(), mix_mask=None):
        """
        frames: list of (latent [B, C, h, w], pose [B, C, h, w] or None, t [B] or float, slot, valid [B] or bool),
        attention groups contiguous in plan order (plan: [(first frame, specs per frame)]), the first `heads` frames
        the ones with a velocity (and audio [B, heads, K+1, D], audio_mask [B, heads]).
        ext: per layer, a flat list [K raw, V, K raw, V, ...] of external frames.
        keep: the frames whose per-layer K/V to return.
        -> velocity [B, C, heads, h, w] or None, per-layer lists of K before RoPE and V [B, len(keep), N, n, d]
        """
        reference_count = 0
        if self.num_refs:
            if self._references is None:
                raise ValueError('Reference-conditioned forward requires net.references(...)')
            ref_latents, ref_keep = self._references
            if ref_latents.shape[0] != frames[0][0].shape[0] or ref_latents.shape[-2:] != frames[0][0].shape[-2:]:
                raise ValueError('Reference batch/spatial geometry does not match this forward')
            first_ref = len(frames)
            if self._reference_cache is not None:
                first_ext = 1 if mix_mask is not None else (len(ext[0]) // 2 if ext and ext[0] else 0)
                ext = [(list(layer) if ext else []) + cached
                       for layer, cached in zip(ext or [[]] * len(self._reference_cache), self._reference_cache)]
                refs = [('c', first_ext)]
            else:
                reference_count = self.num_refs
                refs = [('f', first_ref + r) for r in range(reference_count)]
            plan = [(start, [(refs + spec if frames[start + offset][3] != 0 else list(spec))
                             for offset, spec in enumerate(specs)]) for start, specs in plan]
            if reference_count:
                frames = list(frames) + [(ref_latents[:, :, r], None, 0., MAX_SLOTS, ref_keep[:, r])
                                        for r in range(reference_count)]
                plan.append((first_ref, [[ref] for ref in refs]))
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
        args = (latents, poses, t, valid, posed, slots, plan, heads, tuple(keep), audio, audio_mask,
                reference_count, mix_mask, *ext_flat)
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


def block_forward(slot_freqs, n, d, block, audio_block, plan, heads, keep, x, e0, freqs, audio, audio_mask,
                  mix_mask, *ext):
    """One transformer block of a pass: compiled dense parts around the layout-specific attention (eager)."""
    B, F, N = x.shape[0], x.shape[1], x.shape[2]
    q, k, k_raw, v, gate, shift, scale, gate_ffn = PRE(block, x, e0, freqs, n, d)
    rotated = {}

    def key(s):
        if s[0] == 'f':
            return k[:, s[1]]
        if s[0] == 'c':
            return ext[2 * s[1]]
        if s not in rotated:
            if s[0] == 'm':
                raw = torch.where(mix_mask[:, s[1], None, None, None], k_raw[:, s[1]], ext[0][:, s[3]])
            elif s[0] == 'p':
                raw = ext[0][:, s[1]]
            else:
                raw = k_raw[:, s[1]] if s[0] == 'r' else ext[2 * s[1]]
            rotated[s] = ROPE(raw, slot_freqs[s[2]])
        return rotated[s]

    def value(s):
        if s[0] == 'm':
            return torch.where(mix_mask[:, s[1], None, None, None], v[:, s[1]], ext[1][:, s[3]])
        if s[0] == 'p':
            return ext[1][:, s[1]]
        return ext[2 * s[1] + 1] if s[0] in ('e', 'c') else v[:, s[1]]

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
    template = tuple(None if torch.is_tensor(arg) else arg for arg in args)

    def run(*tensors):
        # ctx.saved_tensors must be the only owner of activation inputs. A
        # closure over the original args keeps whole upstream autograd graphs
        # alive after backward, even after saved_tensors have been released.
        full = list(template)
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


def build_net(state_dict, audio_config, latent_hw, device, mimi_template, num_refs=0):
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
    ref_patch = None
    if num_refs:
        if num_refs != 9:
            raise ValueError('The supported reference checkpoint has nine reference slots')
        ref_patch = nn.Conv3d(arch.in_dim, arch.dim, kernel_size=arch.patch_size, stride=arch.patch_size)
        ref_patch.load_state_dict({k.removeprefix('ref_patch_embedding.'): v for k, v in state_dict.items()
                                   if k.startswith('ref_patch_embedding.')})
        ref_patch.to(device)
    pad_token = nn.Parameter(state_dict['pad_token'].float().clone())
    transformer.to(device)
    for module in (encoder, adapter, pose):
        module.to(device)
    net = BimodalNet(transformer, encoder, adapter, pose, pad_token, latent_hw, K,
                     num_refs=num_refs, ref_patch_embedding=ref_patch).to(device)
    names = {name for name, _ in net.original_named_parameters()}
    missing = names - set(state_dict)
    assert not missing, f'checkpoint lacks {sorted(missing)[:10]}'
    if num_refs:
        # A matching reference convolution alone does not establish that this
        # loader implements every trained module (e.g. an Orpheus hidden
        # adapter or LoRA checkpoint would otherwise be silently ignored).
        # Include persistent buffers in the accepted set; nonpersistent RoPE
        # caches are reconstructed and should not occur in weight exports.
        supported = {name.removeprefix('transformer.') for name in net.state_dict()}
        unexpected = set(state_dict) - supported
        if unexpected:
            raise ValueError('Nine-reference checkpoint contains unsupported parameters/buffers: '
                             f'{sorted(unexpected)[:10]}. Use the matching dense audio/pose/reference '
                             'checkpoint or implement its additional conditioning modules.')
    return net


def load_mimi(audio_config, device):
    """The frozen Mimi frontend (codes -> latents) and the decoder-transformer template of the audio encoders."""
    mimi = MimiModel.from_pretrained(audio_config['model_name']).eval().requires_grad_(False)
    template = copy.deepcopy(mimi.decoder_transformer)
    return mimi.to(device), template


@torch.no_grad()
def audio_latents(mimi, codes, num_codebooks):
    return audio_latents_from_codes(mimi, codes, num_codebooks)
