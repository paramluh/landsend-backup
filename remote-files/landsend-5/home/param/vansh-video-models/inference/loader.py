"""
Builds the Universal Wide network for inference from an inference yaml: the same
module classes and layer wrappers training runs (WanModel, MimiAudioEncoder,
AudioAdapter, pose Conv3d, pad token; InitialLayer / TransformerLayer / FinalLayer),
loaded strictly from one model.safetensors, without the training pipeline or its config.
"""

import copy
import re
from dataclasses import dataclass, replace
from typing import Callable

import torch
from torch import nn
from transformers import MimiModel

from dataloading.sources import NUM_REFS
from inference.common import CKPT_PATH, DTYPE, resolve_path
from inference.lora import load_lora, split_lora_state
from models.autoencoders.wanvae import Wan2_2_VAE
from models.pose.audio import AudioAdapter, MimiAudioEncoder
from models.pose.hidden import HiddenAdapter
from models.pose.settings import wan_5b_wide
from models.pose.wan_universal import FinalLayer, InitialLayer, TransformerLayer
from models.wan.model import WanModel
from utils.common import iterate_safetensors, load_weights_from_state_dict

TAG = 'loader'
# Same names as model.type in the training toml.
TRANSFORMER_LAYERS = {
    'universal_wide': TransformerLayer, 'universal_wide_stabiliser': TransformerLayer,
    'universal_wide_bimodal': TransformerLayer,
}


@dataclass
class Models:
    net: nn.Module  # (x_t, t*1000, audio_latents, pose_latents, ofi, window_source, present_valid) -> velocity
    mimi: MimiModel  # frozen; codes -> encoder latents, and codes -> waveform for muxing
    vae: Wan2_2_VAE
    num_history: int
    num_present: int
    num_codebooks: int
    build_net: Callable  # (num_history, num_present) -> net over the same modules

    def view(self, num_history, num_present):
        """
        The same weights in another window geometry: another mode of one checkpoint.
        """
        return replace(self, net=self.build_net(num_history, num_present), num_history=num_history,
                       num_present=num_present)


class Layered(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def _load_prefix(module, state_dict, prefix):
    module.load_state_dict({k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)})


def build_models(config, base_dir, device):
    arch = wan_5b_wide
    audio = config['audio']
    H, P, K = config['num_history'], config['num_present'], audio['cross_attention_codes']
    R = NUM_REFS if 'refs' in config else 0
    transformer_layer = TRANSFORMER_LAYERS[config['type']]

    state_dict = {
        re.sub(r'^model\.diffusion_model\.', '', key): tensor
        for key, tensor in iterate_safetensors(resolve_path(config['checkpoint_path'], base_dir))
    }
    state_dict, lora_factors = split_lora_state(state_dict, config.get('lora'))

    with torch.device('meta'):
        transformer = WanModel(
            patch_size=arch.patch_size, in_dim=arch.in_dim, dim=arch.dim, ffn_dim=arch.ffn_dim,
            freq_dim=arch.freq_dim, out_dim=arch.out_dim, num_heads=arch.num_heads,
            num_layers=arch.num_layers, qk_norm=arch.qk_norm, eps=arch.eps,
        )
    load_weights_from_state_dict(transformer, state_dict, DTYPE, DTYPE)

    mimi = MimiModel.from_pretrained(audio['model_name']).eval().requires_grad_(False)
    audio_transformer = copy.deepcopy(mimi.decoder_transformer)
    audio_transformer.layers = audio_transformer.layers[:audio['num_layers']]
    encoder = MimiAudioEncoder(
        transformer=audio_transformer, in_dim=audio['in_dim'], hidden_dim=arch.dim,
        drop=audio['drop'], num_window_tokens=K + 1,
    )
    adapter = AudioAdapter(
        hidden_dim=arch.dim, num_attn_heads=arch.num_heads, num_layers=arch.num_layers,
        injection_offset=audio['injection_offset'], injection_interval=audio['injection_interval'],
        initialization='random', max_blocks=audio['max_blocks'],
    )
    assert ('hidden' in config) == any(k.startswith('hidden_adapter.') for k in state_dict), \
        'a [hidden] config needs a checkpoint with hidden_adapter weights and vice versa'
    hidden_adapter = None
    if 'hidden' in config:
        hidden_adapter = HiddenAdapter(
            hidden_dim=arch.dim, num_attn_heads=arch.num_heads, num_layers=arch.num_layers,
            **{**config['hidden'], 'initialization': 'random'},
        )
        _load_prefix(hidden_adapter, state_dict, 'hidden_adapter.')
    pose_patch_embedding = nn.Conv3d(arch.in_dim, arch.dim, kernel_size=arch.patch_size, stride=arch.patch_size)
    pad_token = nn.Parameter(state_dict['pad_token'].float().clone())  # fp32 as in training
    _load_prefix(encoder, state_dict, 'audio_encoder.')
    _load_prefix(adapter, state_dict, 'audio_adapter.')
    _load_prefix(pose_patch_embedding, state_dict, 'pose_patch_embedding.')
    ref_patch_embedding = None
    if R:
        ref_patch_embedding = copy.deepcopy(transformer.patch_embedding)
        _load_prefix(ref_patch_embedding, state_dict, 'ref_patch_embedding.')
    del state_dict

    transformer.to(device, DTYPE).eval()
    for module in (encoder, adapter, pose_patch_embedding, hidden_adapter or nn.Identity()):  # fp32 as in training; a bf16 cast rounds Mimi's rotary inv_freq
        module.to(device).eval()
    mimi.to(device)
    freqs = transformer.freqs.to(device)
    if config.get('lora'):
        names = load_lora(transformer, adapter, lora_factors, config['lora'])
        print(f'[{TAG}] loaded {len(names)} unmerged LoRA projections with original factor dtypes')
    del lora_factors

    def build_net(H, P):
        layers = [InitialLayer(
            transformer, encoder, pose_patch_embedding, pad_token, num_history=H, num_present=P,
            audio_cross_attention_codes=K, audio_dropout_p=0.0, compile_block=False,
            num_refs=R, ref_patch_embedding=ref_patch_embedding,
            hidden_drop_token=hidden_adapter and hidden_adapter.drop_token,
        )]
        for index, block in enumerate(transformer.blocks):
            layers.append(transformer_layer(
                block, adapter.get_block(index), freqs, num_history=H, num_present=P,
                compile_block=False, checkpoint=False, num_refs=R,
                hidden_block=hidden_adapter and hidden_adapter.get_block(index),
            ))
        layers.append(FinalLayer(transformer, num_refs=R))
        return Layered(layers).to(device).eval()  # the layer wrappers own index buffers (present/clean key indices)

    net = build_net(H, P)
    vae_path = config['vae_path'] if 'vae_path' in config else f'{CKPT_PATH}/{arch.vae_checkpoint}'
    vae = Wan2_2_VAE(vae_pth=resolve_path(vae_path, base_dir), dtype=torch.float32, device=device)
    print(f'[{TAG}] {config["type"]}: window 1 sink + {H} history + {P} present; '
          f'{len(adapter.blocks)} audio blocks; {audio["num_codebooks"]} codebooks')
    return Models(net, mimi, vae, H, P, audio['num_codebooks'], build_net)
