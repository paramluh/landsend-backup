#!/usr/bin/env python3
"""Run on landsend-1 after training succeeds; preserve every source checkpoint."""
import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

RUN = Path('/home/param/pose-runs/no-text-audio-pose-20261028')
SOURCE = RUN / 'exports/export-200'
DEST = Path('/home/param/models/no-text-audio-pose-20261028-step200')
REPO = Path('/home/param/pose-repo')

assert socket.gethostname().split('.')[0] == 'landsend-1'
assert os.getuid() == 1003
assert (RUN / 'TRAINING_COMPLETE').read_text().strip() == '200'
meta = json.loads((SOURCE / 'meta.json').read_text())
assert meta['step'] == 200 and meta['world'] == 32
assert set(meta['hosts']) == {'landsend-1', 'landsend-2', 'landsend-3', 'landsend-4'}
assert (SOURCE / 'COMPLETE').read_text().strip() == '200'
assert not (DEST / 'COMPLETE').exists(), 'Already assembled; inspect instead of replacing'
DEST.mkdir(parents=True, exist_ok=True)
config = json.loads((RUN / 'config.json').read_text())
assert config['recipe'] == 'no-text' and config['text_loss_enabled'] is False
shutil.copy2(RUN / 'config.json', DEST / 'config.json')
(DEST / 'README.md').write_text('''# No-text audio and pose fine-tune: step 200

This self-contained BF16 export contains all 32 model shards and the replicated
parameters from the completed 200-step Xabi no-text run. It starts from base
pretraining step 17850, uses the original Xabi production cache, and was trained
on landsend-1 through landsend-4 with 64 logical rows per update (MB2, GA1).
Audio and pose have equal loss weight; no text prediction loss was applied.
Each audio and pose residual decoder receives eight optimizer updates per main
update. The exact training settings are in `config.json`.

## Required inference contract

Preserve the training voice conditioning and its structural delimiters. Remove
lexical and word-timing text from that conditioning. Supply PAD throughout the
generation text input, emit constant PAD for the text output, and do not sample
the untouched vocabulary head. There is no transcript prefix. Preserve the
existing audio/pose delays, channel presence, and external initial geometry.

The tensor export does not persist the Python `head.media_only` runtime flag.
Loading these weights alone does not configure inference: callers must apply
the no-text contract above explicitly. Sampling text from this checkpoint would
not reproduce its training input distribution.

## Provenance and recovery

Training run: `no-text-audio-pose-20261028`; source commit
`2ee719b54b6cefe0aae73855e7451cfe9c9ac676`.
W&B: https://forge.coreweave.com/wandb/canopy-labs/param-pose-llm/runs/a5334709

This directory is an inference export, not a resumable optimizer checkpoint.
Original node-local exports and full resumable checkpoints remain under
`/home/param/pose-runs/no-text-audio-pose-20261028/` on each training host.
The final full save is `saves/step-00200`; optimizer and scheduler state are
distributed across those hosts. `SHA256SUMS` verifies all model shards and model
metadata; `assembly-audit.json` records additional documentation hashes.
''')
local_names = [f'rank{r}.pt' for r, host in enumerate(meta['hosts']) if host == 'landsend-1'] + ['rep.pt', 'meta.json']
methods = {}
for name in local_names:
    src, dst = SOURCE / name, DEST / name
    if dst.exists():
        methods[name] = 'existing; collector verifies SHA256'
        continue
    try:
        os.link(src, dst)
        methods[name] = 'hardlink'
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(src, dst)
        methods[name] = 'copy across filesystems'

# This existing collector verifies all local/remote source hashes and writes
# COMPLETE only after all 32 rank shards plus rep.pt/meta.json are verified.
subprocess.run([str(REPO / '.venv/bin/python'),
                str(REPO / 'infer/transcript_pose/collect_checkpoint.py'),
                str(SOURCE), str(DEST)], check=True)
expected = {f'rank{r}.pt' for r in range(32)} | {'rep.pt', 'meta.json'}
lines = (DEST / 'SHA256SUMS').read_text().splitlines()
assert {line.split()[1] for line in lines} == expected
assert len(lines) == 34
assert (DEST / 'COMPLETE').read_text().strip() == '200'
assert json.loads((DEST / 'config.json').read_text()) == config
report = {
    'status': 'pass',
    'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'host': 'landsend-1', 'path': str(DEST), 'source': str(SOURCE),
    'step': 200, 'world': 32, 'verified_model_files': 34,
    'model_bytes': sum((DEST / name).stat().st_size for name in expected),
    'local_file_methods': methods,
    'documentation_sha256': {name: hashlib.sha256((DEST / name).read_bytes()).hexdigest()
                             for name in ['config.json', 'README.md']},
    'source_preserved': True,
    'inference_contract': 'voice conditioning retained; no transcript; constant PAD text inputs and outputs; no vocabulary-head sampling',
}
(DEST / 'assembly-audit.json').write_text(json.dumps(report, indent=2) + '\n')
(RUN / 'final-export-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
