import argparse
import json
from pathlib import Path

import torch

from distillation.compare import sha256, snapshot


def merged_parameters(state, lora):
    factors = {name.rsplit('.', 1)[0] for name in state if name.endswith(('.lora_A', '.lora_B'))}
    if bool(factors) != bool(lora):
        raise ValueError('Adapter tensors and checkpoint LoRA metadata disagree')
    output = {name: value for name, value in state.items() if not name.endswith(('.lora_A', '.lora_B'))}
    for name in sorted(factors):
        a, b, weight = state[f'{name}.lora_A'], state[f'{name}.lora_B'], state[f'{name}.weight']
        if a.shape != (lora['rank'], weight.shape[1]) or b.shape != (weight.shape[0], lora['rank']):
            raise ValueError(f'Adapter rank or shape disagrees with checkpoint metadata: {name}')
        output[f'{name}.weight'] = (weight.float() + (b.float() @ a.float()) * (lora['alpha'] / lora['rank'])).to(weight.dtype)
    return output


def export_checkpoint(directory, output):
    from safetensors.torch import load_file, save_file

    source, meta = snapshot(directory)
    output = Path(output)
    provenance = output.with_suffix('.json')
    if output.exists() or provenance.exists():
        raise FileExistsError(f'Export already exists: {output}')
    state = load_file(str(source), device='cpu')
    merged = merged_parameters(state, meta.get('lora'))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output), metadata={'format': 'pt', 'source_checkpoint_id': meta['checkpoint_id']})
    details = {'source': str(source.resolve()), 'source_sha256': sha256(source), 'checkpoint_id': meta['checkpoint_id'],
               'lora': meta.get('lora'), 'output': str(output.resolve()), 'output_sha256': sha256(output),
               'merge': 'W + (alpha/rank) B A in fp32, rounded once to the stored base weight dtype'}
    from distillation.eval_bundle import MARKER
    bundle_marker = Path(directory) / MARKER
    if bundle_marker.exists():
        bundle = json.loads(bundle_marker.read_text())
        details['evaluation_bundle'] = {'marker_sha256': sha256(bundle_marker), 'source': bundle['source']}
    provenance.write_text(json.dumps(details, indent=2) + '\n')
    return details


def main():
    parser = argparse.ArgumentParser(description='Export a checked student checkpoint with LoRA explicitly merged for inference.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output)), flush=True)


if __name__ == '__main__':
    main()
