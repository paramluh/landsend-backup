"""Specificity and coverage controls for the conservative v5 pose scan."""
import json
from pathlib import Path
import unittest

import numpy as np

from strict_detection_r3 import Config, evaluate, prepare, scan
from test_detection import base_clip


def events(p):return scan(p,prefill_seconds=0)['events']
def has_event(p,f=250):return any(abs(e['frame']-f)<=1 for e in events(p))


class StrictControls(unittest.TestCase):
    def test_varied_large_single_frame_steps(self):
        for dx,dy,confidence in [(5,0,.8),(-8,0,.4),(0,6,.6),(0,-18,.9),(12,-9,.6)]:
            with self.subTest(dx=dx,dy=dy,confidence=confidence):
                p=base_clip();p[:,:,2]=confidence;p[250:,:,:2]+=[dx/256,dy/480]
                self.assertTrue(has_event(p))

    def test_nonrigid_persistent_body_shape_steps(self):
        for sign in (-1,1):
            p=base_clip();p[250:,:5,1]+=sign*.025
            p[250:,5,:2]+=sign*np.array([.012,-.008]);p[250:,6,:2]+=sign*np.array([-.008,-.004])
            self.assertTrue(has_event(p))

    def test_single_frame_step_then_smooth_motion(self):
        p=base_clip();p[250:,:,1]+=(.025+np.arange(len(p)-250)*.0001)[:,None]
        self.assertTrue(has_event(p))

    def test_two_equal_adjacent_steps_are_not_single_frame_dominant(self):
        p=base_clip();p[250:,:,1]+=.015;p[251:,:,1]+=.015
        self.assertFalse(has_event(p))

    def test_steps_on_smooth_background_varied_phases(self):
        for phase in (0,np.pi/2,np.pi):
            p=base_clip();p[:,:,1]+=(.005*np.sin(np.arange(len(p))*.035+phase))[:,None]
            p[250:,:,:2]+=[6/256,-4/480]
            self.assertTrue(has_event(p))

    def test_new_smooth_velocity_after_real_step(self):
        p=base_clip();t=np.arange(len(p));p[:,:,0]+=(t*.05/256)[:,None]
        p[250:,:,0]+=(6+np.arange(len(p)-250)*.08)[:,None]/256
        p[250:,:,1]-=4/480
        self.assertTrue(has_event(p))

    def test_small_coordinate_noise_does_not_hide_real_step(self):
        for seed in (11,37,109):
            p=base_clip();p[:,:,:2]+=np.random.default_rng(seed).normal(0,[.1/256,.1/480],(len(p),55,2))
            p[250:,:,:2]+=[6/256,-4/480]
            self.assertTrue(has_event(p))

    def test_visible_subset_is_not_diluted_by_static_invisible_anchors(self):
        p=base_clip();p[:,3:5,2]=0;p[250:,:3,1]+=4.5/480;p[250:,5:7,1]+=3/480
        self.assertTrue(evaluate(prepare(p),250)['passes'])
        self.assertTrue(has_event(p))

    def test_no_head_only_or_shoulder_only_jump(self):
        for ids in (slice(0,5),slice(5,7)):
            p=base_clip();p[250:,ids,1]+=.04;self.assertFalse(has_event(p))

    def test_head_jump_while_torso_continues_smoothly(self):
        p=base_clip();p[:,5:13,1]+=(np.arange(len(p))*.0004)[:,None];p[250:,:5,0]+=.04
        self.assertFalse(has_event(p))

    def test_abrupt_hand_entry_and_confidence_only_crossing(self):
        p=base_clip();p[:250,13:,2]=.05;p[250:,13:,1]-=.3
        self.assertFalse(has_event(p))
        p=base_clip();p[:250,:7,2]=.05
        self.assertFalse(has_event(p))

    def test_low_to_high_confidence_real_geometry_cut_is_allowed(self):
        p=base_clip();p[:250,:7,2]=.05;p[250:,:7,1]+=.04
        self.assertTrue(has_event(p))

    def test_low_destination_excluded_regardless_of_source_confidence(self):
        for source in (.05,.8):
            p=base_clip();p[:250,:7,2]=source;p[250:,:7,2]=.35;p[250:,:7,1]+=.04
            self.assertFalse(has_event(p))

    def test_both_shoulders_must_be_yellow_or_green_at_destination(self):
        p=base_clip();p[250:,:,1]+=.04;p[250:,6,2]=.39
        self.assertFalse(has_event(p))

    def test_fast_smooth_symmetric_and_asymmetric_gestures(self):
        t=np.arange(500)-250
        for rise,fall in ((1.5,1.5),(2.5,2.5),(4.,4.),(2.,5.),(5.,2.)):
            p=base_clip();sigma=np.where(t<0,rise,fall);pulse=np.exp(-.5*(t/sigma)**2)
            p[:,:5,0]+=(.065*pulse)[:,None];p[:,5:13,1]+=(.03*pulse)[:,None]
            self.assertEqual(events(p),[])

    def test_smooth_transition_to_new_persistent_pose(self):
        for width in (2.,3.,5.,8.):
            p=base_clip();q=(1+np.tanh((np.arange(len(p))-250)/width))/2
            p[:,:,1]+=(.06*q)[:,None];self.assertEqual(events(p),[])

    def test_impulse_and_brief_hold_return_are_not_persistent(self):
        for duration in (1,3,6):
            p=base_clip();p[250:250+duration,:,1]+=.04
            self.assertFalse(has_event(p))

    def test_persistent_jitter_and_larger_peak_are_rejected(self):
        for add_step in (False,True):
            p=base_clip();p[:,:,:2]+=np.random.default_rng(9).normal(0,[4/256,4/480],(len(p),1,2))
            if add_step:p[250:,:,1]+=.045
            self.assertEqual(events(p),[])

    def test_original_user_positive(self):
        path=Path('/home/param/runs/pose-runs/pose-sites-20261030/video-inspection/v1-19-21/pose_overlay.json')
        if not path.exists():self.skipTest('Original reference not present')
        self.assertTrue(has_event(json.loads(path.read_text())['frames'],512))

    def test_user_positive243_is_excluded_by_new_destination_confidence_rule(self):
        path=Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000243/pose_overlay.json')
        if not path.exists():self.skipTest('User positive243 not present')
        p=json.loads(path.read_text())['frames'];r=evaluate(prepare(p),292)
        self.assertFalse(r['passes'])
        self.assertIn('destination_not_medium_high_confidence_for_head_and_both_shoulders',r['rejection_reasons'])

    def test_user_confirmed_false_positive145(self):
        path=Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000145/pose_overlay.json')
        if not path.exists():self.skipTest('User negative145 not present')
        self.assertFalse(has_event(json.loads(path.read_text())['frames'],474))

    def test_same_transition_duration_must_explain_offset_and_all_fits(self):
        p=base_clip();p[250:,:,1]+=.03
        r=evaluate(prepare(p),250)
        for duration in r['passing_transition_intervals']:
            self.assertTrue(next(v for v in r['persistence_variants'] if v['transition_intervals']==duration)['passes_persistence'])
            self.assertTrue(all(f['v5_qualifies'] for f in r['trajectory_fits'] if f['transition_intervals']==duration))

    def test_multi_frame_variant_configuration_is_forbidden(self):
        with self.assertRaises(ValueError):Config(transition_intervals=(1,2))


if __name__=='__main__':unittest.main()
