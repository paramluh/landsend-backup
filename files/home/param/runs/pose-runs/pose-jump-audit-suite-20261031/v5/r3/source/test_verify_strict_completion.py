"""Completion must fail on missing inputs, stale provenance or unrendered hits."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import strict_detection_r3 as strict
from test_detection import base_clip
from verify_strict_completion import DEFAULT_SUITE,sha,verify


class StrictCompletionControls(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.run=self.root/'runs/rollout-000000';self.run.mkdir(parents=True)
        self.controls=DEFAULT_SUITE/'v5/r3/controls.json'
        self.labels=DEFAULT_SUITE/'v5/ground-truth-update-243-rewatch.json'

    def fixture(self,jump=False):
        p=base_clip(200)
        if jump:p[80:,:,1]+=.03
        np.savez(self.run/'tokens.npz',pose=np.zeros((100,32),dtype=np.int16))
        summary={'duration':8.,'sample_index':0,'frames':100,'prefill_frames':0,'tokens_sha256':sha(self.run/'tokens.npz')}
        pose={'fps':25,'width':256,'height':480,'start_time':0,'frame_count':200,'frames':p.tolist()}
        for name,value in [('summary.json',summary),('pose_overlay.json',pose),('detection.json',{'events':[]}),('status.json',{'state':'done'})]:
            (self.run/name).write_text(json.dumps(value))
        with contextlib.redirect_stdout(io.StringIO()):
            strict.analyze_cohort(self.root,self.root/'strict-review-v5-r3.json',self.controls)

    def check(self,**kw):
        return verify(self.root,count=kw.pop('count',1),expected_duration=8.,controls_path=self.controls,
                      labels_path=kw.pop('labels_path',self.labels),**kw)

    def test_complete_zero_candidate_cohort_needs_no_old_pool_renders(self):
        self.fixture();r=self.check();self.assertEqual(r['state'],'passed',r['failures'])
        self.assertEqual(r['verified_rollouts'],1);self.assertEqual(r['strict_events_with_render_coverage'],0)
        labels={c['case']:c for c in r['controls']['control_pose_replays']}
        self.assertTrue(labels['243']['historical_human_label']);self.assertFalse(labels['243']['current_human_label'])

    def test_missing_expected_run_fails(self):
        self.fixture();r=self.check(count=2);self.assertEqual(r['state'],'failed')
        self.assertIn('Run IDs differ',str(r['failures']))

    def test_changed_pose_after_scan_fails_provenance(self):
        self.fixture();p=self.run/'pose_overlay.json';j=json.loads(p.read_text());j['frames'][40][0][0]+=.001;p.write_text(json.dumps(j))
        r=self.check();self.assertEqual(r['state'],'failed');self.assertIn('source hashes differ',str(r['failures']))

    def test_nonzero_strict_event_requires_actual_render_coverage(self):
        self.fixture(jump=True);r=self.check();self.assertEqual(r['state'],'failed')
        self.assertIn('Missing rendered coverage for strict event',str(r['failures']))

    def test_old_broad_pool_hits_do_not_create_render_requirement(self):
        self.fixture();(self.run/'detection.json').write_text(json.dumps({'events':[{'frame':80,'time_s':3.2}]}))
        with contextlib.redirect_stdout(io.StringIO()):strict.analyze_cohort(self.root,self.root/'strict-review-v5-r3.json',self.controls)
        r=self.check();self.assertEqual(r['state'],'passed',r['failures']);self.assertEqual(r['candidate_events_verified'],0)

    def test_historical243_positive_is_not_silently_used_as_current_truth(self):
        self.fixture();p=self.root/'stale-labels.json';p.write_text(json.dumps({'updates':[]}))
        r=self.check(labels_path=p);self.assertEqual(r['state'],'failed');self.assertIn('243 rewatch correction',str(r['failures']))


if __name__=='__main__':unittest.main()
