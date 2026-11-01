"""Contiguous, per-file atomic latent export without model or CUDA imports.

Call this for both model outputs before decoding either video. Filenames must
be unique across concurrent workers; an existing destination is replaced.
"""

import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def persist_latents(output, outputs, *, verify=False):
    """Save named tensors and return content-hash/shape/dtype receipts.

    ``outputs`` contains ``(name, tensor)`` pairs. Each final file is published
    only after its temporary file has been completely written (and optionally
    checked). Publication is atomic per file, not across the complete set.
    """
    output = Path(output)
    latent_receipts = []
    for name, values in outputs:
        if not torch.isfinite(values).all(): raise RuntimeError(f'Nonfinite output{name}')
        cpu = values.detach().cpu().contiguous()
        target = output/f'{name}_latents.safetensors'
        temporary = target.with_suffix('.safetensors.partial')
        save_file({'latents':cpu}, str(temporary))
        if verify:
            restored = load_file(str(temporary))['latents']
            assert torch.equal(cpu, restored), 'Serialized latent differs from model output'
        temporary.replace(target)
        with target.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        latent_receipts.append({'path':str(target), 'sha256':digest,
                                'shape':list(cpu.shape), 'dtype':str(cpu.dtype),
                                'finite':True, 'contiguous':cpu.is_contiguous()})
        del cpu
    return latent_receipts
