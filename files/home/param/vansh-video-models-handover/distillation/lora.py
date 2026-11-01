import math
import re

import torch
from torch import nn
from torch.nn import functional as F

BASE_DTYPE = None
FUSED_QKV = False


class LoRALinear(nn.Linear):
    def __init__(self, linear, rank, alpha, base_dtype=BASE_DTYPE):
        nn.Module.__init__(self)
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('LoRA rank and alpha must be positive')
        if base_dtype not in (None, 'bfloat16'):
            raise ValueError('LoRA base_dtype must be bfloat16 or None')
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight, self.bias = linear.weight, linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        self.lora_A = nn.Parameter(self.weight.new_empty(rank, self.in_features))
        self.lora_B = nn.Parameter(self.weight.new_zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if base_dtype == 'bfloat16':
            self.weight.data = self.weight.data.bfloat16()
            if self.bias is not None:
                self.bias.data = self.bias.data.bfloat16()
        self.train(linear.training)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

    @torch.no_grad()
    def merged_weight(self):
        update = self.lora_B.float() @ self.lora_A.float()
        return (self.weight.float() + update * self.scaling).to(self.weight.dtype)


def fused_lora_qkv(attention, x):
    layers = (attention.q, attention.k, attention.v)
    projected = F.linear(x, torch.cat([layer.lora_A for layer in layers]))
    hidden = projected.reshape(-1, 3, layers[0].rank).transpose(0, 1)
    updates = torch.bmm(hidden, torch.stack([layer.lora_B for layer in layers]).transpose(1, 2))
    return tuple(F.linear(x, layer.weight, layer.bias) + updates[i].reshape(*x.shape[:-1], layer.out_features) * layer.scaling
                 for i, layer in enumerate(layers))


def inject_lora(module, rank, alpha, targets, base_dtype=BASE_DTYPE, fused_qkv=FUSED_QKV):
    patterns = [re.compile(pattern) for pattern in targets]
    matches = [[name for name, layer in module.named_modules()
                if type(layer) is nn.Linear and pattern.fullmatch(name)] for pattern in patterns]
    if not patterns or any(not names for names in matches):
        raise ValueError(f'LoRA targets must each match a Linear: {targets}')
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('LoRA rank and alpha must be positive')
    if base_dtype not in (None, 'bfloat16'):
        raise ValueError('LoRA base_dtype must be bfloat16 or None')
    names = list(dict.fromkeys(name for matched in matches for name in matched))
    groups = []
    if fused_qkv:
        for name in names:
            parent_name, _, child_name = name.rpartition('.')
            if child_name == 'q' and all(f'{parent_name}.{key}'.lstrip('.') in names for key in ('k', 'v')):
                parent = module.get_submodule(parent_name)
                if len({(layer.in_features, layer.out_features) for layer in (parent.q, parent.k, parent.v)}) != 1:
                    raise ValueError('fused_qkv requires q/k/v projections with identical dimensions')
                groups.append(parent)
        if not groups:
            raise ValueError('fused_qkv requires targeted q/k/v Linear siblings')
    module.requires_grad_(False)
    for name in names:
        parent_name, _, child_name = name.rpartition('.')
        parent = module.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(getattr(parent, child_name), rank, alpha, base_dtype))
    for parent in groups:
        parent.lora_fused_qkv = True
    return names


@torch.no_grad()
def merged_state_dict(module):
    layers = {name: layer for name, layer in module.named_modules() if isinstance(layer, LoRALinear)}
    adapters = {f'{name}.{factor}'.lstrip('.') for name in layers for factor in ('lora_A', 'lora_B')}
    weights = {f'{name}.weight'.lstrip('.'): layer for name, layer in layers.items()}
    return {name: (weights[name].merged_weight() if name in weights else tensor).detach().cpu().clone()
            for name, tensor in module.state_dict().items() if name not in adapters}
