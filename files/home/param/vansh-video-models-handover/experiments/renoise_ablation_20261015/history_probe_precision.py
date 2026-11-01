import json
import sys
import time
from pathlib import Path

import torch
import yaml

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
    results, derivatives = [], {}
    started = time.time()
    for n in (0, 8):
        clip = manifest['clips'][n]
        gt, pose, _, codes = load_chunk(str(Path(cfg['root_dir']) / clip['paths']), cfg['pose_file'], frame + 1, cfg['mimi_file'])
        gt, pose, codes = gt.to(device), pose.to(device), codes.to(device)
        seed = clip['seed']
        noise = torch.randn(gt[:, frame].shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))
        renoise = torch.randn(gt[:, frame].shape, device=device, generator=torch.Generator(device=device).manual_seed(seed * 1000003 + frame))
        x02 = .8 * gt[:, frame] + .2 * noise
        direction = torch.randn(48, 2, 1, 1, device=device, generator=torch.Generator(device=device).manual_seed(10000000 + seed))
        direction = direction.expand(48, 2, *gt.shape[-2:])
        direction = direction / direction.square().mean().sqrt()
        for epsilon in (.01, .02, .04):
            delta = direction * epsilon
            for arm, t in [('baseline', .4), ('direct', .2)]:
                out = gt[None].repeat(2, 1, 1, 1, 1)
                out[0, :, frame - 2:frame] += delta
                out[1, :, frame - 2:frame] -= delta
                poses = pose[None].expand(2, -1, -1, -1, -1)
                code_rows = codes[None].expand(2, -1, -1)
                x = .75 * x02 + .1375 ** .5 * renoise if arm == 'baseline' else x02
                ts = torch.linspace(t, 0, 11, device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    resolver.frame(out, poses, code_rows, frame, x[None].repeat(2, 1, 1, 1), ts)
                difference = (out[0, :, frame] - out[1, :, frame]).double()
                derivative = difference / (2 * epsilon)
                result = dict(index=n, split=clip['split'], uuid=clip['uuid'], frame=frame, arm=arm,
                              epsilon_rms=epsilon, history_delta_l2=float(delta.double().norm()),
                              gain_l2=float(difference.norm() / (2 * delta.double().norm())),
                              output_difference_rms=float(difference.square().mean().sqrt()),
                              output_rms=float(out[:, :, frame].double().square().mean().sqrt()))
                if epsilon == .01:
                    derivatives[n, arm] = derivative.cpu()
                else:
                    previous = derivatives[n, arm].to(device)
                    result['derivative_relative_change_vs_001'] = float((derivative - previous).norm() / previous.norm())
                    result['derivative_cosine_vs_001'] = float(torch.nn.functional.cosine_similarity(derivative.flatten(), previous.flatten(), dim=0))
                results.append(result)
                print(json.dumps(result), flush=True)
    ratios = []
    for n in (0, 8):
        pair = {r['arm']: r['gain_l2'] for r in results if r['index'] == n and r['epsilon_rms'] == .01}
        ratios.append(dict(index=n, baseline_over_direct=pair['baseline'] / pair['direct'], **pair))
    report = dict(method='Fixed GT sink and two GT history frames; one spatially constant Gaussian channel-offset direction per clip, central finite difference; GT target independently noised to .2, then marginal renoise for baseline; 10 Euler steps per arm; bf16 teacher',
                  checkpoint=manifest['teacher'], elapsed_seconds=time.time() - started, results=results, paired_ratios=ratios)
    (ROOT / 'history_probe_precision.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(elapsed_seconds=report['elapsed_seconds'], paired_ratios=ratios)), flush=True)


if __name__ == '__main__':
    main()
