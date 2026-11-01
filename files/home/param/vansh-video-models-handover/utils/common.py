import gc
from pathlib import Path
import os

import torch
from safetensors import safe_open


DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float8': torch.float8_e4m3fn,
    'float8_e4m3fn': torch.float8_e4m3fn,
    'float8_e5m2': torch.float8_e5m2,
}
AUTOCAST_DTYPE = torch.bfloat16


def get_rank():
    return int(os.environ.get('RANK', 0))


def is_main_process():
    return get_rank() == 0


def empty_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_state_dict(path):
    path = str(path)
    if path.endswith('.safetensors'):
        sd = load_safetensors(path)
    else:
        sd = torch.load(path, weights_only=True)
    for key in sd:
        if key.endswith('scale_input') or key.endswith('scale_weight'):
            raise ValueError('fp8_scaled weights are not supported. Please use bf16 or normal fp8 weights.')
    return sd


def iterate_safetensors(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Safetensors path does not exist: {path}')
    if path.is_dir():
        safetensors_files = sorted(path.glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise FileNotFoundError(
                f'No .safetensors files found in directory: {path}\n'
                f'  Contents: {[f.name for f in sorted(path.iterdir())[:20]]}'
            )
    else:
        if path.suffix != '.safetensors':
            raise ValueError(f'Expected a .safetensors file, got: {path}')
        safetensors_files = [path]
    for filename in safetensors_files:
        with safe_open(str(filename), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head', 'modulation']


def load_weights_from_state_dict(model, state_dict, dtype, transformer_dtype):
    """Every parameter of `model` must be in `state_dict`; meta parameters are materialized in place."""
    model_keys = set(name for name, _ in model.named_parameters())
    missing = [name for name in model_keys if name not in state_dict]
    if missing:
        raise RuntimeError(
            f'Missing {len(missing)} base model weights from state_dict:\n'
            + '\n'.join(f'  {k}' for k in sorted(missing)[:20])
            + ('\n  ...' if len(missing) > 20 else '')
        )

    for name, param in list(model.named_parameters()):
        dt = dtype if any(kw in name for kw in KEEP_IN_HIGH_PRECISION) else transformer_dtype
        new_data = state_dict[name].to(dt)
        if param.is_meta:
            module_name, param_name = name.rsplit('.', 1)
            setattr(model.get_submodule(module_name), param_name,
                    torch.nn.Parameter(new_data, requires_grad=param.requires_grad))
        else:
            param.data = new_data

    if is_main_process():
        extra = [k for k in state_dict if k not in model_keys]
        print(f'[load_weights] Loaded {len(model_keys)} params from state_dict')
        if extra:
            print(f'  IGNORED: {len(extra)} extra keys in state_dict not in model')

    for name, p in model.named_parameters():
        p.original_name = name
