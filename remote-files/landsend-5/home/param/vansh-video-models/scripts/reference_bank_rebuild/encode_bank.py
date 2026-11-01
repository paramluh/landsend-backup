"""Encode a complete selection into the actor-bank contract, with atomic group checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from PIL import Image
import PIL
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from upstream import SLOTS

ENCODER_ROOT = Path('/mnt/paramfs/vansh/richmond-7/reference-latents/encode')
VAE = Path('/home/param/vansh-video-models/wan22-5b-weights/Wan2.2_VAE.pth')
VAE_SHA256 = '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_encoder_module(encoder_root):
    """Import the encoder implementation without loading weights or using CUDA."""
    spec = importlib.util.spec_from_file_location('refs_lib', Path(encoder_root) / 'refs_lib.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['refs_lib'] = module
    spec.loader.exec_module(module)
    return module


def encoder_source_files(module):
    """Use loaded module locations, so cached/shadowed imports cannot be mislabelled."""
    modules = {'refs_lib': module, 'models.muon.vae': module.muon}
    for name in ('models', 'models.muon', 'models.muon.constants', 'utils.kinematic'):
        modules[name] = sys.modules[name]
    files = {'encode_bank.py': Path(__file__).resolve()}
    for name, imported in modules.items():
        path = Path(imported.__file__).resolve()
        suffix = '/__init__.py' if path.name == '__init__.py' else '.py'
        files[name.replace('.', '/') + suffix] = path
    return files


def encoder_fingerprint(encoder_root=ENCODER_ROOT, *, module=None):
    module = load_encoder_module(encoder_root) if module is None else module
    einops = importlib.import_module('einops')
    inputs = {'schema': 1, 'files': {name: sha256(path) for name, path in encoder_source_files(module).items()},
              'versions': {'torch': str(torch.__version__), 'numpy': np.__version__,
                           'Pillow': PIL.__version__, 'einops': einops.__version__,
                           'cuda_runtime': torch.version.cuda},
              'slots': list(SLOTS)}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'sha256': digest, 'inputs': inputs}


def encoding_checkpoint_matches(receipt, destination, selection_hash, fingerprint, count):
    try:
        info = json.loads(Path(receipt).read_text())
        expected = {'selection_sha256': selection_hash, 'vae_sha256': VAE_SHA256,
                    'encoder_fingerprint': fingerprint, 'n': count}
        return (all(info.get(key) == value for key, value in expected.items())
                and info.get('sha256') == sha256(destination))
    except (OSError, ValueError):
        return False


def load_encoder(encoder_root, vae, device, *, module=None):
    if sha256(vae) != VAE_SHA256:
        raise ValueError('VAE does not match the documented Wan2.2 checkpoint')
    module = load_encoder_module(encoder_root) if module is None else module
    module.MUON = str(vae)
    return module.MuonImage(device)


def prepare(row, source_root):
    local = row.get('image_path')
    if not isinstance(local, str) or not local:
        if row['origin'] != 'ref-image':
            raise ValueError(f'Scanned image has no decoded image_path: {row["iid"]}')
        source = row['source']
        prefix = 'hf://buckets/InternalCan/aloe-conversationa-data/'
        if source.startswith(prefix):
            source = source[len(prefix):]
        if '://' in source or Path(source).is_absolute() or '..' in Path(source).parts:
            raise ValueError(f'Invalid source key: {source}')
        local = Path(source_root) / source
    with Image.open(local) as image:
        array = np.asarray(image.convert('RGB')).copy()
    prepared = row.get('prepared_image') is True or isinstance(row.get('prepared_image'), np.bool_) and bool(row['prepared_image'])
    if prepared:
        if array.shape != (480, 256, 3):
            raise ValueError(f'Prepared image shape {array.shape}: {local}')
        pixels = torch.from_numpy(array).permute(2, 0, 1).float()
    else:
        if array.shape != (960, 512, 3):
            raise ValueError(f'Source image shape {array.shape}: {local}')
        image = torch.from_numpy(array).permute(2, 0, 1).float()[None]
        if int(row['property']) >= 5:
            pixels = F.avg_pool2d(image, 2).round()[0]
        else:
            x, y, width, height = (int(row[key]) for key in ('cx0', 'cy0', 'cw', 'ch'))
            if not (0 <= x <= 512 - width and 0 <= y <= 960 - height and 256 <= width <= 282):
                raise ValueError(f'Invalid crop for iid={row["iid"]}')
            pixels = image[:, :, y:y + height, x:x + width]
            if width != 256:
                pixels = F.interpolate(pixels, size=(480, 256), mode='bilinear',
                                       antialias=True, align_corners=False).round().clamp(0, 255)
            pixels = pixels[0]
    if pixels.shape != (3, 480, 256):
        raise ValueError(f'Incorrect prepared shape: {pixels.shape}')
    return (pixels / 255 * 2 - 1)[:, None]


def selection_groups(selection):
    required = {'bank', 'uuid', 'property', 'slot', 'index', 'iid', 'origin', 'source'}
    if not required <= set(selection.columns) or selection.iid.duplicated().any():
        raise ValueError('Missing selection columns or nonunique iid')
    groups = []
    for (bank, actor, prop), frame in selection.groupby(['bank', 'uuid', 'property'], sort=True):
        frame = frame.sort_values('index')
        if bank not in ('bank', 'beatles-eval') or not 1 <= int(prop) <= 9:
            raise ValueError('Invalid bank/property')
        if not frame.slot.eq(SLOTS[int(prop) - 1]).all() or frame['index'].tolist() != list(range(len(frame))) or not 1 <= len(frame) <= 20:
            raise ValueError(f'Incorrect slot/index/cap: {bank}/{actor}/{prop}')
        groups.append((bank, actor, int(prop), frame.to_dict('records')))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--vae', type=Path, default=VAE)
    parser.add_argument('--encoder-root', type=Path, default=ENCODER_ROOT)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard')
    torch.set_num_threads(1)
    module = load_encoder_module(args.encoder_root)
    fingerprint = encoder_fingerprint(module=module)
    source_hash = sha256(args.selection)
    groups = selection_groups(pd.read_parquet(args.selection))[args.shard::args.shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / 'encoding-checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    pending = []
    for bank, actor, prop, rows in groups:
        destination = args.output_dir / bank / actor / f'property_{prop}' / 'latents.safetensors'
        receipt = checkpoint_dir / f'{bank}__{actor}__{prop}.json'
        if encoding_checkpoint_matches(receipt, destination, source_hash, fingerprint, len(rows)):
            continue
        pending.append((destination, receipt, bank, actor, prop, rows))
    print(json.dumps({'shard': args.shard, 'groups': len(groups), 'pending': len(pending)}), flush=True)
    if not pending:
        return
    torch.cuda.set_device(args.gpu)
    encoder = load_encoder(args.encoder_root, args.vae, 'cuda', module=module)
    started = time.monotonic()
    completed, images = 0, 0
    with ThreadPoolExecutor(8) as pool:
        while pending:
            batch, count = [], 0
            while pending and (not batch or count + len(pending[0][-1]) <= args.batch_size):
                group = pending.pop(0)
                batch.append(group)
                count += len(group[-1])
            rows = [row for group in batch for row in group[-1]]
            inputs = torch.stack(list(pool.map(lambda row: prepare(row, args.source_root), rows)))
            latents = encoder.encode(inputs.cuda()).to(torch.bfloat16).cpu()
            if latents.shape != (len(rows), 48, 30, 16) or not torch.isfinite(latents).all():
                raise ValueError('Invalid encoder output')
            offset = 0
            for destination, receipt, bank, actor, prop, group_rows in batch:
                tensor = latents[offset:offset + len(group_rows)].contiguous()
                offset += len(group_rows)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix('.tmp')
                save_file({'video': tensor}, temporary, metadata={
                    'uuid': actor, 'property': str(prop), 'slot': SLOTS[prop - 1], 'n': str(len(tensor))})
                temporary.replace(destination)
                info = {'selection_sha256': source_hash, 'sha256': sha256(destination),
                        'vae_sha256': VAE_SHA256, 'encoder_fingerprint': fingerprint, 'n': len(tensor)}
                temporary = receipt.with_suffix('.tmp')
                temporary.write_text(json.dumps(info) + '\n')
                temporary.replace(receipt)
            completed += len(batch)
            images += len(rows)
            print(json.dumps({'shard': args.shard, 'completed_groups': completed, 'images': images,
                              'seconds': round(time.monotonic() - started, 2)}), flush=True)


if __name__ == '__main__':
    main()
