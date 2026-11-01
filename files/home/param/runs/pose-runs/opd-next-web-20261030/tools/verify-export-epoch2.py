#!/usr/bin/env python3
"""Check final export identity and exact learned gain, without rehashing weights."""
import datetime
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

ROOT = Path('/home/param/pose-runs/opd-next-web-20261030')
EXPORT = ROOT / 'export-epoch-2'
SOURCE = Path('/home/param/pose-runs/opd-next-rosie-sft-20261030/training/epoch-2')

meta = json.loads((EXPORT / 'meta.json').read_text())
provenance = json.loads((EXPORT / 'PROVENANCE.json').read_text())
complete = json.loads((EXPORT / 'COMPLETE').read_text())
assert complete['provenance_sha256'] == hashlib.sha256((EXPORT / 'PROVENANCE.json').read_bytes()).hexdigest()
assert meta['checkpoint_id'] == 'rosie-epoch-2-step-54'
assert meta['step'] == 54 and meta['finetune_epoch'] == 2
assert provenance['learner_tensors'] == 1439 and provenance['frozen_prefixes'] == []
assert str(meta['opd_student_version']) == '110'
for item in provenance['files']:
    assert (EXPORT / item['path']).stat().st_size == item['bytes']
gain = torch.empty(1, dtype=torch.float32)
dcp.load({'model': {'vgain.theta': gain}}, checkpoint_id=SOURCE / 'model', no_dist=True)
export_gain = torch.load(EXPORT / 'rank0.pt', map_location='cpu', mmap=True,
                         weights_only=True)['vgain.theta']
assert export_gain.dtype == torch.float32 and torch.equal(export_gain, gain)
assert meta['vgain_theta_fp32'] == gain.item() == provenance['vgain_theta_fp32']
report = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              checkpoint_id=meta['checkpoint_id'], epoch=2, step=54,
              parent_opd_version=110, inference_parameters=1439,
              weight_files=len(provenance['files']),
              serialized_weight_bytes=sum(item['bytes'] for item in provenance['files']),
              vgain_theta_fp32=gain.item(), gain_matches_native_epoch2_exactly=True,
              verification='Exporter validated each serialized tensor against its intended cast; final identity and native FP32 gain independently checked.',
              ready_for_backend=True)
(ROOT / 'EXPORT_VERIFIED.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
