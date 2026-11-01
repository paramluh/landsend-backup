"""Check actual saved bank latents against their prepared RGB inputs through the matching VAE."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

from encode_bank import ENCODER_ROOT, VAE, VAE_SHA256, encoder_fingerprint, prepare, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--per-group', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    qa = args.output_dir / 'qa'
    qa.mkdir(exist_ok=True)
    (qa / 'decode-report.json').unlink(missing_ok=True)
    if sha256(VAE) != VAE_SHA256:
        raise ValueError('Wrong VAE checkpoint')
    fingerprint = encoder_fingerprint()
    rows = pd.read_parquet(args.selection).sort_values(['bank', 'property', 'origin', 'uuid', 'index'])
    sample = rows.groupby(['bank', 'property', 'origin'], sort=True).head(args.per_group)
    if sample.empty:
        raise ValueError('Cannot decode-check an empty bank')
    torch.set_num_threads(1)
    torch.cuda.set_device(args.gpu)
    sys.path.insert(0, str(ENCODER_ROOT / 'wan-vae-collection'))
    from optimisations.decoder import FastMuonDecoder
    decoder = FastMuonDecoder(str(VAE), 'float32', 'cuda')
    # The eager function is the same decode math; avoids compilation for a small QA sample.
    decoder._run_fn = decoder._run
    torch.backends.cudnn.benchmark = False
    result, latent_hashes = [], {}
    for row in sample.to_dict('records'):
        path = args.output_dir / row['bank'] / row['uuid'] / f'property_{row["property"]}' / 'latents.safetensors'
        with safe_open(path, framework='pt') as handle:
            latent = handle.get_tensor('video')[int(row['index'])]
        if path not in latent_hashes:
            latent_hashes[path] = sha256(path)
        with torch.no_grad():
            pixels = decoder.decode(latent[None, :, None].float().cuda().repeat(1, 1, 2, 1, 1))[0, :, 0].cpu()
        actual = ((pixels.clamp(-1, 1) + 1) * 127.5).round()
        expected = ((prepare(row, args.source_root)[:, 0] + 1) * 127.5).round()
        mse = float(((actual - expected) ** 2).mean())
        psnr = float(10 * np.log10(255 ** 2 / mse))
        result.append({'bank': row['bank'], 'uuid': row['uuid'], 'property': int(row['property']),
                       'index': int(row['index']), 'iid': int(row['iid']), 'origin': row['origin'],
                       'latent_sha256': latent_hashes[path], 'PSNR_dB': psnr})
        print(json.dumps(result[-1]), flush=True)
    values = [row['PSNR_dB'] for row in result]
    report = {'images': len(result), 'mean_PSNR_dB': float(np.mean(values)), 'min_PSNR_dB': min(values),
              'selection_sha256': sha256(args.selection), 'vae_sha256': VAE_SHA256,
              'encoder_fingerprint': fingerprint,
              'passed': bool(np.isfinite(values).all() and np.mean(values) >= 30),
              'criterion': 'Mean RGB reconstruction PSNR >=30dB; checks normalization/cropping/encoder integration, not historical identity',
              'samples': result}
    temporary = qa / 'decode-report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(qa / 'decode-report.json')
    if not report['passed']:
        raise ValueError('Saved bank reconstruction check failed')


if __name__ == '__main__':
    main()
