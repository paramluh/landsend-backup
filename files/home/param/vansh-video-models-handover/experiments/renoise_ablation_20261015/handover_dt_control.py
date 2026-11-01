import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

ROOT = Path('/mnt/paramfs/param/outputs/renoise-ablation-20261015')
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'code/production')]
from bimodal.cached_resolver import ResolverNet
from inference.loader import build_models


@torch.no_grad()
def main():
    cfg = yaml.safe_load((ROOT / 'configs/baseline.yaml').read_text())
    models = build_models({**cfg, **cfg['resolving']}, ROOT / 'code/bimodal', 'cuda')
    resolver = ResolverNet(models)
    report = json.loads((ROOT / 'handover_probe.json').read_text())
    results = []
    frame, epsilon = 256, .04
    for n in (0, 8):
        for kind in ('history', 'current'):
            for arm, t, steps in [('baseline', .4, 20), ('direct', .2, 10)]:
                source = ROOT / 'probe_tensors' / f'clip{n}-{kind}-{arm}-eps0.040.safetensors'
                tensors = load_file(source, device='cuda')
                C, h, w = tensors['present_initial'].shape[1:]
                out = torch.zeros(2, C, frame + 1, h, w, device='cuda')
                out[:, :, [0, frame - 2, frame - 1]] = tensors['context_latents']
                pose = torch.zeros_like(out)
                pose[:, :, [0, frame - 2, frame - 1, frame]] = tensors['context_and_present_pose']
                codes = tensors['mimi_codes'][None].expand(2, -1, -1)
                ts = torch.linspace(t, 0, steps + 1, device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    resolver.frame(out, pose, codes, frame, tensors['present_initial'], ts)
                resolved = out[:, :, frame]
                difference = (resolved[0] - resolved[1]).double()
                delta_norm = (tensors['direction_unit_rms'] * epsilon).double().norm()
                result = dict(index=n, kind=kind, arm=arm, t_start=t, steps=steps, dt=.02,
                              gain_l2=float(difference.norm() / (2 * delta_norm)),
                              gain_rms=float(difference.square().mean().sqrt() / (2 * epsilon)),
                              max_abs_output_difference_vs_10steps=float((resolved - tensors['resolved_plus_minus']).abs().max()),
                              source_inputs=str(source))
                output = ROOT / 'probe_tensors' / f'clip{n}-{kind}-{arm}-dt0.020.safetensors'
                save_file({'resolved_plus_minus': resolved.cpu().contiguous(), 'times': ts.cpu()}, output)
                result['outputs'] = str(output)
                results.append(result)
                print(json.dumps(result), flush=True)
    report['discretization_control'] = results
    (ROOT / 'history_probe_final.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
