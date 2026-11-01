import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import collect_live

def payload(data):
    return {'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest(),
            'source_bytes':len(data),'copied_bytes':len(data),'truncated':False}

class CollectorTests(unittest.TestCase):
    def test_atomic_snapshots_and_waiting_state(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            health=collect_live.install({'source_exists':False,'remote_time':1,'files':{}},root)
            self.assertEqual(health['status'],'waiting_for_run')
            first=(root/'current').resolve();value=b'{"update":1,"loss/text":0.2}\n'
            health=collect_live.install({'source_exists':True,'remote_time':2,'files':{'events-rank0.jsonl':payload(value)}},root)
            self.assertEqual(health['status'],'live')
            self.assertNotEqual(first,(root/'current').resolve())
            self.assertEqual((root/'current/events-rank0.jsonl').read_bytes(),value)
            self.assertTrue(first.exists())
            terminal=payload(b'{"status":"completed"}');terminal['source_mtime']=1234.
            collect_live.install({'source_exists':True,'remote_time':3,'files':{'RESULT.json':terminal}},root)
            self.assertEqual((root/'current/RESULT.json').stat().st_mtime,1234.)

    def test_weights_and_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError,'Unapproved'):
                collect_live.install({'source_exists':True,'remote_time':1,'files':{'weights.pt':payload(b'anything')}},name)
            value=payload(b'{}');value['sha256']='bad'
            with self.assertRaisesRegex(ValueError,'hash'):
                collect_live.install({'source_exists':True,'remote_time':1,'files':{'config.json':value}},name)

    def test_connection_failure_preserves_last_snapshot(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);collect_live.install({'source_exists':False,'remote_time':1,'files':{}},root)
            before=(root/'current').resolve()
            with mock.patch.object(collect_live.subprocess,'run',side_effect=TimeoutError('test')):
                health=collect_live.collect_once(root)
            self.assertEqual(health['status'],'error');self.assertIn('last_success',health)
            self.assertEqual((root/'current').resolve(),before)

    def test_raw_samples_payloads_precede_manifests_and_exclude_weights(self):
        snapshot={'source_exists':True,'raw_exists':True,'remote_time':1,'files':{}}
        calls=[]
        def run(argv,**kwargs):
            calls.append(argv)
            return mock.Mock(stdout=json.dumps(snapshot) if argv[0]=='ssh' else '')
        with tempfile.TemporaryDirectory() as name, mock.patch.object(collect_live.subprocess,'run',side_effect=run):
            self.assertTrue(collect_live.collect_once(name)['raw_rollouts_mirrored'])
        self.assertEqual(len(calls),3)
        self.assertIn('--include=/update-??????/sample-*.npz',calls[1])
        self.assertIn('--include=/update-??????/manifest.json',calls[2])
        for call in calls[1:]:
            self.assertIn('--exclude=*',call);self.assertIn('--no-links',call)
            self.assertIn('--max-size=16m',call)

    def test_step_checkpoint_policy_comes_from_actual_metadata(self):
        from pose_doc import build_doc
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'config.json').write_text(json.dumps({'method':{'checkpoint_interval_steps':60,'checkpoint_interval_seconds':None,'learning_rate':3e-6}}))
            (root/'plan.json').write_text(json.dumps({'updates':[[[0,0,0]]]*129,'rows_per_epoch':43,'epochs':3}))
            doc=build_doc(root)
            self.assertEqual(doc['export_steps'],[60,120,129])
            self.assertEqual(doc['next_export_step'],60)
            self.assertIsNone(doc['save_every_min']);self.assertFalse(doc['save_wall'])
            self.assertEqual(doc['learning_rate_peak'],3e-6)
            self.assertIsNone(doc['learning_rate_current'])
            (root/'metrics.jsonl').write_text('{"update":1,"learning_rate":3e-7,"loss/direct":0.2}\n')
            doc=build_doc(root)
            self.assertEqual(doc['learning_rate_current'],3e-7)
            self.assertIn('learning_rate',doc['train_cols'])

    def test_failed_supervisor_never_remains_running(self):
        from pose_doc import build_doc
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'supervisor').mkdir()
            (root/'metrics.jsonl').write_text('{"update":1,"loss/direct":0.2}\n')
            (root/'events-rank0.jsonl').write_text('{"event":"phase","rank":0,"update":2,"phase":"actor_build_begin","time":100}\n')
            (root/'supervisor/finished.json').write_text('{"exit_code":1,"time":120}')
            doc=build_doc(root)
            self.assertEqual(doc['state'],'failed');self.assertEqual(doc['step'],1)
            self.assertEqual(doc['supervisor_exit_code'],1)

if __name__=='__main__':unittest.main()
