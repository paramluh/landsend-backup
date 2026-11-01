"""Bounded CPU checks for acquisition overlap, provenance, resume and cleanup."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from download_and_scan import VerifiedInputs, create_pass, file_identity, require_selected_complete, run
from hf_sources import BUCKET, write_json
from owned_subprocess import stop_group
from video_pipeline import plan_identity


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'scan'
        self.sources = self.root / 'sources'
        self.output.mkdir()
        self.sources.mkdir()
        self.receipts = self.root / 'receipts.jsonl'
        self.tracks = [self.track(n) for n in range(2)]
        write_json(self.output / 'tracks.json', self.tracks)
        # The fixture deliberately remains source-limited. Successful selected
        # work must not erase this flag or the original diagnostics.
        write_json(self.output / 'targets-summary.json', {'complete': False, 'missing_tracks': 1})
        write_json(self.output / 'targets-missing.json', [{'track': 'unavailable/speaker_0'}])
        lines = []
        for track in self.tracks:
            lines.extend([(track['video_source_metadata']['path'], track['video_path']),
                          (track['pose_sources']['0'].split(f'{BUCKET}/')[1], track['pose_paths']['0'])])
        (self.output / 'downloads.tsv').write_text(''.join(f'{key}\t{path}\n' for key, path in lines))

    def track(self, number):
        name = f'recording{number}/speaker_1'
        key = f'videos/{name}/full.mp4'
        return {'track': name, 'recording_id': f'recording{number}', 'speaker': 1,
                'uuid': f'actor{number}', 'session_id': 'session', 'usable_chunks': [0],
                'video_path': str(self.sources / f'video{number}.mp4'),
                'source': f'hf://buckets/{BUCKET}/{key}',
                'video_source_metadata': {'path': key, 'size': 5, 'xet_hash': 'pinned-video-hash'},
                'pose_paths': {'0': str(self.sources / f'pose{number}.safetensors')},
                'pose_sources': {'0': f'hf://buckets/{BUCKET}/poses/{name}/0'}, 'excluded_ranges': []}

    def receipt(self, track, video=True):
        path = Path(track['video_path'] if video else track['pose_paths']['0'])
        path.write_bytes(b'video' if video else b'pose')
        return {'status': 'reused', 'bucket': BUCKET,
                'source_key': (track['source'] if video else track['pose_sources']['0']).split(f'{BUCKET}/')[1],
                'local_path': str(path), 'size': path.stat().st_size,
                'xet_hash': 'pinned-video-hash' if video else 'pose-hash',
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'local_stat': list(file_identity(path)),
                'verified_at': datetime.now(timezone.utc).isoformat()}

    def test_old_receipts_and_partial_lines_cannot_authorize_work(self):
        first = self.receipt(self.tracks[0])
        self.receipts.write_text(json.dumps(first) + '\n')
        state = VerifiedInputs(self.receipts, self.tracks, self.receipts.stat().st_size)
        pose = self.receipt(self.tracks[0], video=False)
        with self.receipts.open('a') as handle:
            handle.write(json.dumps(pose))
        state.update()
        self.assertFalse(state.ready(self.tracks[0]))
        self.assertEqual(len(state.verified), 0)
        with self.receipts.open('a') as handle:
            handle.write('\n' + json.dumps(first) + '\n')
        state.update()
        self.assertTrue(state.ready(self.tracks[0]))
        self.assertFalse(state.ready(self.tracks[1]))

    def test_rejects_changed_pinned_hf_source_and_post_verification_bytes(self):
        record = self.receipt(self.tracks[0])
        changed = {**record, 'xet_hash': 'changed-remote-source'}
        self.receipts.write_text(json.dumps(changed) + '\n')
        with self.assertRaisesRegex(ValueError, 'changed since target'):
            VerifiedInputs(self.receipts, self.tracks).update()
        self.receipts.write_text(json.dumps(record) + '\n')
        state = VerifiedInputs(self.receipts, self.tracks)
        state.update()
        path = Path(self.tracks[0]['video_path'])
        original = path.stat()
        path.write_bytes(b'other')
        # Same-size writes can share a filesystem timestamp tick. Advance mtime
        # explicitly so this checks the recorded-stat contract deterministically.
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
        with self.assertRaisesRegex(ValueError, 'changed during'):
            state.assert_unchanged()
        with self.assertRaisesRegex(ValueError, 'changed after'):
            VerifiedInputs(self.receipts, self.tracks).update()

    def test_requires_all_poses_preserves_full_track_and_unavailable_metadata(self):
        track = deepcopy(self.tracks[0])
        track['usable_chunks'].append(1)
        track['pose_paths']['1'] = str(self.sources / 'second.safetensors')
        track['pose_sources']['1'] = f'hf://buckets/{BUCKET}/poses/second'
        self.receipts.write_text(''.join(json.dumps(self.receipt(track, flag)) + '\n' for flag in (True, False)))
        state = VerifiedInputs(self.receipts, [track])
        state.update()
        self.assertFalse(state.ready(track))
        destination = create_pass(self.output, self.root / 'pass', [track])
        self.assertEqual(json.loads((destination / 'tracks.json').read_text()), [track])
        for name in ('targets-summary.json', 'targets-missing.json'):
            self.assertEqual((destination / name).read_bytes(), (self.output / name).read_bytes())

    def test_final_check_requires_exact_selected_chunks_and_current_plan(self):
        identity = plan_identity(self.output)
        good = {'plan_sha256': identity, 'completed_chunks': 2, 'missing_sources': 0,
                'failures': 0, 'available_targets_complete': True, 'complete': False}
        require_selected_complete(good, self.tracks, identity)
        for key, value in [('completed_chunks', 1), ('missing_sources', 1), ('failures', 1),
                           ('available_targets_complete', False), ('plan_sha256', {})]:
            with self.assertRaises(ValueError, msg=key):
                require_selected_complete({**good, key: value}, self.tracks, identity)

    def test_real_process_overlap_and_final_checkpoint_import(self):
        scripts = self.root / 'fixture-scripts'
        scripts.mkdir()
        # The downloader will fail if a scan does not start before acquisition
        # finishes. No network, MediaPipe models, SSH, or GPUs are involved.
        (scripts / 'hf_sources.py').write_text('''
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root'); p.add_argument('--list'); p.add_argument('--receipts')
a,_=p.parse_known_args(); root=Path(a.root); pairs=[line.split('\\t') for line in Path(a.list).read_text().splitlines()]
for number,(key,name) in enumerate(pairs):
    if number==2:
        deadline=time.monotonic()+8
        while not (root/'early-scan-started').exists():
            if time.monotonic()>deadline: raise RuntimeError('No acquisition/scan overlap')
            time.sleep(.02)
    path=Path(name); video=key.startswith('videos/'); data=b'video' if video else b'pose'; path.write_bytes(data)
    row=dict(status='downloaded',bucket='InternalCan/aloe-conversationa-data',source_key=key,local_path=str(path),
             size=len(data),xet_hash='pinned-video-hash' if video else 'pose-hash',sha256=hashlib.sha256(data).hexdigest(),
             local_stat=[path.stat().st_dev,path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns],
             verified_at=datetime.now(timezone.utc).isoformat())
    with Path(a.receipts).open('a') as f: f.write(json.dumps(row)+'\\n')
Path(a.receipts+'.summary.json').write_text(json.dumps(dict(requested=len(pairs),missing=[],failures=[])))
(root/'download-finished').write_text(str(time.time()))
''')
        (scripts / 'distributed_scan.py').write_text('''
import argparse,json,time,sys,os
sys.path.insert(0, os.environ['PYTHONPATH'])
from pathlib import Path
import pandas as pd
from hf_sources import write_json
from video_pipeline import input_fingerprint,task_path,digest,plan_identity
p=argparse.ArgumentParser(); p.add_argument('--output-dir'); p.add_argument('--nodes',nargs='+')
a,_=p.parse_known_args(); out=Path(a.output_dir); tracks=json.loads((out/'tracks.json').read_text())
assert len(a.nodes)==5
root=Path(tracks[0]['video_path']).parent
if out.name.startswith('pass-'):
    assert not (root/'download-finished').exists()
    (root/'early-scan-started').write_text(str(time.time()))
count=resumed=0
for track in tracks:
    for chunk in track['usable_chunks']:
        path=task_path(out,track['track'],chunk); path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): resumed+=1
        else:
            pd.DataFrame().to_parquet(path,index=False)
            write_json(path.with_suffix('.complete.json'),dict(track=track['track'],chunk=chunk,rows=0,complete=True,
                sha256=digest(path),fingerprint=input_fingerprint(track,chunk,track['pose_paths'][str(chunk)],{'fixture':'hash'})))
        count+=1
write_json(out/'scan-summary.json',dict(plan_sha256=plan_identity(out),completed_chunks=count,candidate_rows=0,
    missing_sources=0,failures=0,available_targets_complete=True,complete=False,resumed=resumed))
''')
        args = SimpleNamespace(output_dir=self.output, source_root=self.sources, receipts=self.receipts,
                               python=sys.executable, token_name='fixture', nodes=['n0','n1','n2','n3','n4'],
                               workers=1, download_workers=1, download_processes=1, download_batch_size=1,
                               archive_root=self.root, model_root=self.root, cache=[], scan_interval=0,
                               poll_seconds=.02)
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parent), 'OPENBLAS_NUM_THREADS': '1'}
        with patch('download_and_scan.code_config', return_value={'fixture': 'hash'}), \
                patch('download_and_scan.snapshot_code', return_value=(scripts, {})), patch.dict(os.environ, env):
            self.assertEqual(run(args), 0)
        status = json.loads((self.output / 'download-scan-status.json').read_text())
        self.assertEqual(status['stage'], 'finished')
        self.assertFalse(status['complete'])
        self.assertTrue(status['available_targets_complete'])
        self.assertEqual(status['selected_inputs_verified'], 4)
        self.assertGreaterEqual(status['scan_summary']['resumed'], 1)
        self.assertLess(float((self.sources / 'early-scan-started').read_text()),
                        float((self.sources / 'download-finished').read_text()))


class OwnedProcessTests(unittest.TestCase):
    def test_disconnect_stops_only_owned_child_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lease = root / 'lease.json'
            write_json(lease, {'run_id': 'fixture', 'time': time.time()})
            code = ("import os,subprocess,sys,time; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                    f"Path({str(root / 'pids.json')!r}).write_text(str(os.getpid())+','+str(p.pid)); "
                    "time.sleep(60)")
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
            wrapper = subprocess.Popen([sys.executable, str(Path(__file__).with_name('owned_subprocess.py')),
                                        '--lease', str(lease), '--run-id', 'fixture', '--', sys.executable, '-c', code],
                                       stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not (root / 'pids.json').exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                pids = [int(value) for value in (root / 'pids.json').read_text().split(',')]
                wrapper.stdin.close()
                self.assertNotEqual(wrapper.wait(timeout=15), 0)
                for pid in pids:
                    path = Path(f'/proc/{pid}/stat')
                    self.assertTrue(not path.exists() or path.read_text().split()[2] == 'Z')
                self.assertIsNone(unrelated.poll())
            finally:
                stop_group(wrapper)
                stop_group(unrelated)


if __name__ == '__main__':
    unittest.main(verbosity=2)
