"""Optional eager inference with immutable CPU weights and one resident GPU block.

This backend is deliberately separate from training and the fast sampler. Weight
precision and the cached sampler's arithmetic are retained; compilation and CUDA
graphs are unsupported because every block reuses the same parameter storage.
"""

import copy
from types import SimpleNamespace

import torch
from torch.overrides import TorchFunctionMode

from bimodal.cached_resolver import ResolverNet
from bimodal.sampler import WavefrontNet
from inference.loader import build_models
from utils.common import AUTOCAST_DTYPE


class CPUConstruction(TorchFunctionMode):
    """Keep the existing loader CPU-only, including its unused training counter.

    InitialLayer currently creates one float64 scalar explicitly on CUDA even
    when the loader was asked for CPU. Redirect only that exact allocation. Any
    additional explicit CUDA factory fails instead of silently consuming VRAM.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else dict(kwargs)
        device = kwargs.get('device')
        if device is not None and torch.device(device).type == 'cuda':
            if func is torch.zeros and args == ((),) and kwargs.get('dtype') == torch.float64:
                kwargs['device'] = 'cpu'
            else:
                raise RuntimeError(f'Unexpected CUDA allocation during CPU model construction: {func}')
        return func(*args, **kwargs)


def build_models_cpu(config, base_dir):
    with CPUConstruction():
        models = build_models(config, base_dir, 'cpu')
    models.net.requires_grad_(False)
    for tensor in (*models.net.parameters(), *models.net.buffers(), *models.mimi.parameters()):
        if tensor.device.type != 'cpu':
            raise RuntimeError('CPU-first construction left a model tensor on another device')
    return models


def _tensors(module):
    return dict((*module.named_parameters(), *module.named_buffers()))


def _signature(module):
    return (tuple((name, type(child)) for name, child in module.named_modules()),
            tuple((name, tuple(value.shape), value.dtype) for name, value in _tensors(module).items()))


class ResidentLayers:
    """An inference-only iterable, with one reusable module for each block kind.

    CPU tensors are never modified or copied back. Copies and compute use the
    caller's current CUDA stream, so the next overwrite follows all reads from
    the previous layer. Keeping a yielded layer past the next iteration is not
    supported. Nested/concurrent iteration and gradient tracking are rejected.
    """

    def __init__(self, source_layers, device):
        self.sources = list(source_layers)
        if not self.sources:
            raise ValueError('Streaming requires at least one transformer block')
        self.device = torch.device(device)
        self.active = False
        self.closed = False
        self.resident = {}
        self.source_tensors = {}
        self.transferred_bytes = 0
        self.passes = 0
        for kind in ('block', 'audio_block', 'hidden_block'):
            modules = [getattr(layer, kind, None) for layer in self.sources]
            template = next((module for module in modules if module is not None), None)
            if template is None:
                self.resident[kind] = None
                self.source_tensors[kind] = [None] * len(modules)
                continue
            signature = _signature(template)
            for module in modules:
                if module is None:
                    continue
                if _signature(module) != signature:
                    raise ValueError(f'Incompatible {kind} structure, tensor geometry or precision')
                if any(value.device.type != 'cpu' for value in _tensors(module).values()):
                    raise ValueError(f'{kind} source weights must remain on CPU')
            self.resident[kind] = copy.deepcopy(template).requires_grad_(False).to(self.device).eval()
            self.source_tensors[kind] = [None if module is None else _tensors(module) for module in modules]
        self.resident_tensors = {kind: None if module is None else _tensors(module)
                                 for kind, module in self.resident.items()}

    def __len__(self):
        return len(self.sources)

    def __iter__(self):
        if self.closed:
            raise RuntimeError('Resident layers have been closed')
        if self.active:
            raise RuntimeError('Resident layers do not support concurrent or nested iteration')
        if torch.is_grad_enabled():
            raise RuntimeError('Resident layers are inference-only; use torch.no_grad()')
        self.active = True
        try:
            for index, source in enumerate(self.sources):
                parts = {}
                for kind, destination in self.resident_tensors.items():
                    tensors = self.source_tensors[kind][index]
                    if tensors is None:
                        parts[kind] = None
                        continue
                    for name, value in tensors.items():
                        destination[name].copy_(value, non_blocking=False)
                        self.transferred_bytes += value.numel() * value.element_size()
                    parts[kind] = self.resident[kind]
                # FP32 audio norms must stay FP32. More generally, an outer
                # autocast context may otherwise cache casts of overwritten
                # parameter storage and incorrectly reuse another layer's weights.
                with torch.autocast(self.device.type,
                                    enabled=torch.is_autocast_enabled(self.device.type),
                                    dtype=torch.get_autocast_dtype(self.device.type), cache_enabled=False):
                    yield SimpleNamespace(**parts, freqs=getattr(source, 'freqs', None))
            self.passes += 1
        finally:
            self.active = False

    def close(self):
        if self.active:
            raise RuntimeError('Cannot close resident layers during iteration')
        self.resident.clear()
        self.resident_tensors.clear()
        self.closed = True


def _prepare_edges(models, device):
    """Move only the small input/output modules and rotary table to the GPU."""
    layers = models.net.layers
    layers[0].to(device).eval().requires_grad_(False)
    layers[-1].to(device).eval().requires_grad_(False)
    freqs = layers[1].freqs.to(device)
    for layer in layers[1:-1]:
        layer.freqs = freqs  # freqs is an ordinary tensor attribute, not a buffer.
    # Conditioning needs only these two Mimi components. Keep its waveform
    # decoder on CPU until the separate audio/video decoding phase.
    for name in ('quantizer', 'upsample'):
        module = getattr(models.mimi, name, None)
        if module is not None:
            module.to(device)


def release_edges(models):
    layers = models.net.layers
    layers[0].cpu()
    layers[-1].cpu()
    freqs = layers[1].freqs.cpu()
    for layer in layers[1:-1]:
        layer.freqs = freqs
    models.mimi.cpu()


class StreamedWavefrontNet(WavefrontNet):
    def __init__(self, models, latent_hw, substeps, device, history_t):
        _prepare_edges(models, device)
        # The parent casts only audio Linear/Conv weights, retaining FP32
        # normalization, exactly as the ordinary production wavefront does.
        super().__init__(models, latent_hw, substeps, device, compile=False, batch=1, history_t=history_t)
        self.blocks = ResidentLayers(self.blocks, device)

    def start_clip(self, sink, cuda_graph=False):
        if cuda_graph:
            raise ValueError('CUDA graphs cannot capture CPU-to-GPU block streaming')
        return super().start_clip(sink, False)


class StreamedResolverNet(ResolverNet):
    def __init__(self, models, device):
        _prepare_edges(models, device)
        super().__init__(models)
        self.device = torch.device(device)
        self.blocks = ResidentLayers(self.blocks, device)

    def context(self, *args, **kwargs):
        with torch.autocast(self.device.type, dtype=AUTOCAST_DTYPE, cache_enabled=False):
            return super().context(*args, **kwargs)

    def velocity(self, *args, **kwargs):
        with torch.autocast(self.device.type, dtype=AUTOCAST_DTYPE, cache_enabled=False):
            return super().velocity(*args, **kwargs)

    def audio(self, *args, **kwargs):
        with torch.autocast(self.device.type, dtype=AUTOCAST_DTYPE, cache_enabled=False):
            return super().audio(*args, **kwargs)
