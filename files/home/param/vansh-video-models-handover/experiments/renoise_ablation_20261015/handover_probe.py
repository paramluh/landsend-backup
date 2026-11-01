import json
import sys
import time
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

ROOT = Path('/mnt/paramfs/param/outputs/renoise-ablation-20261015')
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'code/production')]
from bimodal.cached_resolver import ResolverNet
from inference.loader import build_models
from production.sink_universal import load_chunk


@torch.no_grad()
def main():
    cfg = yaml.safe_load((ROOT / 'configs/baseline.yaml').read_text())
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    device, frame = 'cuda', 256
    models = build_models({**cfg, **cfg['resolving']}, ROOT / 'code/bimodal', device)
    resolver = ResolverNet(models)
    artifacts = ROOT / 'probe_tensors'
    artifacts.mkdir(exist_ok=True)
    results, derivatives = [], {}
    started = time.time()
    for n in manifest['gallery_indices']:
        clip = manifest['clips'][n]
        gt, pose, _, codes = load_chunk(str(Path(cfg['root_dir']) / clip['paths']), cfg['pose_file'], frame + 1, cfg['mimi_file'])
        gt, pose, codes = gt.to(device), pose.to(device), codes.to(device)
        seed = clip['seed']
        draw = lambda shape, seed: torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))
        noise = draw(gt[:, frame].shape, seed)
        renoise = draw(gt[:, frame].shape, seed * 1000003 + frame)
        x02 = .8 * gt[:, frame] + .2 * noise
        history_direction = draw((48, 2, 1, 1), 10000000 + seed).expand(48, 2, *gt.shape[-2:])
        current_direction = draw((48, 1, 1), 20000000 + seed).expand(48, *gt.shape[-2:])
        directions = {'history': history_direction / history_direction.square().mean().sqrt(),
                      'current': current_direction / current_direction.square().mean().sqrt()}
        for kind in ('history', 'current'):
            epsilons = [.01, .02, .04] if kind == 'history' and n in (0, 8) else [.04]
            if kind == 'history' and n == 0:
                epsilons = [0.] + epsilons
            for epsilon in epsilons:
                delta = directions[kind] * epsilon
                for arm, t in [('baseline', .4), ('middle', .3), ('direct', .2)]:
                    if arm == 'middle' and epsilon not in (0., .04):
                        continue
                    out = gt[None].repeat(2, 1, 1, 1, 1)
                    x02_pair = x02[None].repeat(2, 1, 1, 1)
                    if kind == 'history':
                        out[0, :, frame - 2:frame] += delta
                        out[1, :, frame - 2:frame] -= delta
                    else:
                        x02_pair[0] += delta
                        x02_pair[1] -= delta
                    poses = pose[None].expand(2, -1, -1, -1, -1)
                    code_rows = codes[None].expand(2, -1, -1)
                    a = (1 - t) / .8
                    sigma = max(0., t * t - (a * .2) ** 2) ** .5
                    x = a * x02_pair + sigma * renoise
                    ts = torch.linspace(t, 0, 11, device=device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        resolver.frame(out, poses, code_rows, frame, x, ts)
                    difference = (out[0, :, frame] - out[1, :, frame]).double()
                    result = dict(index=n, split=clip['split'], uuid=clip['uuid'], frame=frame, kind=kind, arm=arm,
                                  t_start=t, epsilon_rms=epsilon, input_delta_l2=float(delta.double().norm()),
                                  gain_l2=float(difference.norm() / (2 * delta.double().norm())) if epsilon else None,
                                  gain_rms=float(difference.square().mean().sqrt() / (2 * epsilon)) if epsilon else None,
                                  output_difference_rms=float(difference.square().mean().sqrt()),
                                  output_difference_max=float(difference.abs().max()),
                                  output_rms=float(out[:, :, frame].double().square().mean().sqrt()))
                    if epsilon:
                        derivative = difference / (2 * epsilon)
                        derivatives[n, kind, arm, epsilon] = derivative.cpu()
                        if epsilon == .04 and (n, kind, arm, .02) in derivatives:
                            previous = derivatives[n, kind, arm, .02].to(device)
                            result['derivative_relative_change_vs_002'] = float((derivative - previous).norm() / previous.norm())
                            result['derivative_cosine_vs_002'] = float(torch.nn.functional.cosine_similarity(derivative.flatten(), previous.flatten(), dim=0))
                    tensors = dict(context_latents=out[:, :, [0, frame - 2, frame - 1]],
                                   context_and_present_pose=poses[:, :, [0, frame - 2, frame - 1, frame]],
                                   gt_target=gt[:, frame], present_initial=x, x02_initial=x02_pair,
                                   resolved_plus_minus=out[:, :, frame], direction_unit_rms=directions[kind],
                                   mimi_codes=codes, times=ts)
                    name = f'clip{n}-{kind}-{arm}-eps{epsilon:.3f}.safetensors'
                    save_file({k: v.cpu().contiguous() for k, v in tensors.items()}, artifacts / name,
                              metadata={k: str(v) for k, v in result.items()})
                    result['tensors'] = str(artifacts / name)
                    results.append(result)
                    print(json.dumps(result), flush=True)
    ratios = []
    for kind in ('history', 'current'):
        for n in manifest['gallery_indices']:
            pair = {r['arm']: r['gain_l2'] for r in results if r['index'] == n and r['kind'] == kind and r['epsilon_rms'] == .04}
            ratios.append(dict(index=n, kind=kind, baseline_over_direct=pair['baseline'] / pair['direct'], **pair))
    report = dict(method='Fixed GT context; smooth colorlike offsets constant across spatial plane per latent channel. History perturbations affect two GT history frames; current perturbations affect x02 before marginal renoise scaling. Common noise and directions across all times and amplitudes. Ten Euler steps, bf16 teacher, centered finite differences. Saved exact model inputs and outputs.',
                  checkpoint=manifest['teacher'], elapsed_seconds=time.time() - started, results=results, paired_ratios=ratios)
    (ROOT / 'handover_probe.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(elapsed_seconds=report['elapsed_seconds'], paired_ratios=ratios)), flush=True)


if __name__ == '__main__':
    main()
