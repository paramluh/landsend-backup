import copy
import re

import torch
from torch import nn

from models.wan.model import WanModel
from utils.common import is_main_process, iterate_safetensors, load_weights_from_state_dict
from .audio import AudioAdapter, MimiAudioEncoder
from .hidden import HiddenAdapter


def _load_module(module, state_dict, prefix, allowed_missing=()):
    values = {
        key.removeprefix(prefix): tensor
        for key, tensor in state_dict.items()
        if key.startswith(prefix)
    }
    if not values:
        if is_main_process():
            print(f'[{module.__class__.__name__}] {prefix.rstrip(".")} training from scratch')
        return
    missing, unexpected = module.load_state_dict(values, strict=False)
    assert not unexpected
    assert set(missing) <= set(allowed_missing)


def load_diffusion_model(self):
    dtype = self.model_config['dtype']
    transformer_dtype = self.model_config['transformer_dtype']

    audio_transformer = copy.deepcopy(self.mimi.decoder_transformer)
    audio_transformer.layers = audio_transformer.layers[:self.audio_num_layers]
    from transformers.models.mimi.modeling_mimi import MimiLayerScale

    audio_transformer.apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)
    for module in audio_transformer.modules():
        if isinstance(module, MimiLayerScale):
            nn.init.constant_(module.scale, self.mimi.config.layer_scale_initial_scale)
    del self.mimi.decoder
    del self.mimi.decoder_transformer

    self.audio_encoder = MimiAudioEncoder(
        transformer=audio_transformer,
        in_dim=self.audio_in_dim,
        hidden_dim=self.model_dim,
        drop=self.audio_drop,
        num_window_tokens=self.audio_cross_attention_codes + 1,
    )
    self.audio_encoder.requires_grad_(True)
    self.audio_adapter = AudioAdapter(
        hidden_dim=self.model_dim,
        num_attn_heads=self.model_arch['num_heads'],
        num_layers=self.model_arch['num_layers'],
        injection_offset=self.audio_injection_offset,
        injection_interval=self.audio_injection_interval,
        initialization=self.audio_config['initialization'],
        max_blocks=self.audio_config['max_blocks'],
    )
    self.hidden_adapter = None
    if 'hidden' in self.model_config:
        self.hidden_adapter = HiddenAdapter(
            hidden_dim=self.model_dim,
            num_attn_heads=self.model_arch['num_heads'],
            num_layers=self.model_arch['num_layers'],
            **self.model_config['hidden'],
        )
    self.pose_patch_embedding = nn.Conv3d(
        self.model_arch['in_dim'], self.model_dim,
        kernel_size=self.model_arch['patch_size'], stride=self.model_arch['patch_size'],
    )
    self.pad_token = nn.Parameter(torch.zeros(self.model_dim))

    with torch.device('meta'):
        self.transformer = WanModel(**self.model_arch)

    state_dict = {}
    for key, tensor in iterate_safetensors(self.transformer_path):
        state_dict[re.sub(r'^model\.diffusion_model\.', '', key)] = tensor
    if 'patch_embedding.weight' not in state_dict:
        raise RuntimeError(f'[{self.name}] patch_embedding.weight not found in checkpoint')
    patch_shape = state_dict['patch_embedding.weight'].shape
    expected = (self.model_dim, self.model_arch['in_dim'])
    if patch_shape[:2] != expected:
        raise RuntimeError(f'[{self.name}] expected patch shape {expected + ("...",)}, got {tuple(patch_shape)}')

    load_weights_from_state_dict(self.transformer, state_dict, dtype, transformer_dtype)
    _load_module(self.audio_encoder, state_dict, 'audio_encoder.', ('no_audio_tokens',))
    _load_module(self.audio_adapter, state_dict, 'audio_adapter.')
    _load_module(self.pose_patch_embedding, state_dict, 'pose_patch_embedding.')
    if self.hidden_adapter is not None:
        _load_module(self.hidden_adapter, state_dict, 'hidden_adapter.')
    assert not self.num_refs or self.model_config.get('ref_patch_embedding'), '[refs] needs model.ref_patch_embedding'
    self.ref_patch_embedding = None
    if self.model_config.get('ref_patch_embedding', False):
        self.ref_patch_embedding = copy.deepcopy(self.transformer.patch_embedding)
        if 'ref_patch_embedding.weight' in state_dict:
            _load_module(self.ref_patch_embedding, state_dict, 'ref_patch_embedding.')
        elif is_main_process():
            print(f'[{self.name}] ref_patch_embedding copied from patch_embedding')
    if 'pad_token' in state_dict:
        self.pad_token.data.copy_(state_dict['pad_token'])
    elif is_main_process():
        print(f'[{self.name}] pad_token training from scratch')
    self.pad_token.original_name = 'pad_token'
    for prefix, module in (
        ('audio_encoder.', self.audio_encoder),
        ('audio_adapter.', self.audio_adapter),
        ('pose_patch_embedding.', self.pose_patch_embedding),
        ('ref_patch_embedding.', self.ref_patch_embedding or nn.Identity()),
        ('hidden_adapter.', self.hidden_adapter or nn.Identity()),
    ):
        for name, parameter in module.named_parameters():
            parameter.original_name = prefix + name

    self.transformer.train()
    if is_main_process():
        print(f'[{self.name}] Window: 1 sink + {self.num_history} history + {self.num_present} present')
        print(f'[{self.name}] AudioAdapter: {len(self.audio_adapter.blocks)} blocks')
        if self.hidden_adapter is not None:
            print(f'[{self.name}] HiddenAdapter: layers {self.hidden_adapter.injection_layers}')


def get_param_groups(self, parameters):
    groups_config = self.config['optimizer'].get('groups', [])
    buckets = [[] for _ in groups_config]
    default = []
    for parameter in parameters:
        name = getattr(parameter, 'original_name', '')
        for index, group in enumerate(groups_config):
            if name.startswith(group['prefix']):
                buckets[index].append(parameter)
                break
        else:
            default.append(parameter)

    output = [{'params': default}]
    for index, group in enumerate(groups_config):
        values = {'params': buckets[index]}
        for key in ('lr', 'weight_decay', 'betas'):
            if key in group:
                values[key] = group[key]
        output.append(values)
    return output
