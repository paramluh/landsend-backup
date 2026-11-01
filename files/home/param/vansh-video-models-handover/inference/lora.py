"""Explicit, unmerged inference adapters using the exact distillation operators."""

import torch
from torch import nn

from distillation.lora import fused_lora_qkv, inject_lora


def split_lora_state(state, lora):
    factors = {name: value for name, value in state.items() if name.endswith(('.lora_A', '.lora_B'))}
    if bool(factors) != bool(lora):
        raise ValueError('An unmerged LoRA checkpoint and explicit lora metadata are both required')
    return {name: value for name, value in state.items() if name not in factors}, factors


@torch.no_grad()
def load_lora(transformer, adapter, factors, lora):
    """Load factors after the ordinary BF16 transformer / FP32 audio initialization.

    The temporary holder gives inject_lora precisely the names used by BimodalNet.
    Saved factor dtypes must match that canonical initialization; no merge or cast
    of adapter factors is permitted here.
    """
    holder = nn.Module()
    holder.transformer, holder.audio_adapter = transformer, adapter
    names = inject_lora(holder, **lora)
    parameters = {name.removeprefix('transformer.'): parameter for name, parameter in holder.named_parameters()
                  if name.endswith(('.lora_A', '.lora_B'))}
    if parameters.keys() != factors.keys():
        raise ValueError(f'LoRA keys differ: missing={parameters.keys() - factors.keys()}, '
                         f'extra={factors.keys() - parameters.keys()}')
    for name, parameter in parameters.items():
        saved = factors[name]
        if saved.shape != parameter.shape or saved.dtype != parameter.dtype:
            raise ValueError(f'LoRA shape or dtype differs for {name}: '
                             f'{tuple(saved.shape)}/{saved.dtype} vs {tuple(parameter.shape)}/{parameter.dtype}')
        parameter.copy_(saved)
    holder.requires_grad_(False).eval()
    return names


def project_qkv(attention, x):
    if getattr(attention, 'lora_fused_qkv', False):
        return fused_lora_qkv(attention, x)
    return attention.q(x), attention.k(x), attention.v(x)


@torch.no_grad()
def cast_inference_weights(module):
    """Cache BF16 Linear/Conv operands while preserving factors and norm weights."""
    for sub in module.modules():
        if isinstance(sub, (nn.Linear, nn.Conv3d)):
            sub.weight.data = sub.weight.data.bfloat16()
            if sub.bias is not None:
                sub.bias.data = sub.bias.data.bfloat16()
