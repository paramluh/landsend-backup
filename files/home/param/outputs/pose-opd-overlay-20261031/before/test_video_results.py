"""Generated video identity, cache isolation and publication regressions."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pose-preview'))
from video_results import merge_generated_videos
from make_demo import publish_samples
import make_demo


class VideoResultsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'decoder';self.source.mkdir()
        self.results=self.source/'video-results';self.results.mkdir()
        self.stage=self.root/'staging';self.stage.mkdir()
        self.sha='a'*64
        self.base={'schema_version':1,'synthetic':False,'rollouts':[{'id':'update-1','update':1,
          'source_manifest_sha256':self.sha,'samples':[{'id':f'u1-r0-s{i}','duration_seconds':120,'video_url':None} for i in range(4)]}]}
        self.raw=json.dumps(self.base).encode();(self.source/'manifest.json').write_bytes(self.raw)
        self.reset()

    def tearDown(self): self.tmp.cleanup()
    def reset(self): (self.stage/'manifest.json').write_bytes(self.raw)
    def publish(self, slot=0, **changes):
        data=b'fake mp4 test bytes '+str(slot).encode();name=f'video-{slot}.mp4';(self.results/name).write_bytes(data)
        d={'schema_version':1,'kind':'pose-opd-generated-video','complete':True,'update':1,'sample_id':f'u1-r0-s{slot}',
           'source_manifest_sha256':self.sha,'file':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
           'duration_seconds':120,'fps':25,'frame_count':3000,'width':270,'height':480,'video_offset_seconds':0}
        d.update(changes);p=self.results/f'u1-r0-s{slot}-{self.sha}.json';p.write_text(json.dumps(d));return d,p,data
    def sample(self, i=0): return json.loads((self.stage/'manifest.json').read_text())['rollouts'][0]['samples'][i]

    def test_partial_results_attach_independently_and_keep_decoder_manifest_unchanged(self):
        self.publish(2);r=merge_generated_videos(self.source,self.stage)
        self.assertEqual(r['attached'],['u1-r0-s2']);self.assertEqual(r['rejected'],[])
        self.assertIsNone(self.sample(0)['video_url']);self.assertTrue((self.stage/self.sample(2)['video_url']).exists())
        self.assertEqual((self.source/'manifest.json').read_bytes(),self.raw)

    def test_restart_and_new_decoder_snapshot_do_not_erase_video(self):
        self.publish();merge_generated_videos(self.source,self.stage);url=self.sample()['video_url']
        self.reset();merge_generated_videos(self.source,self.stage)
        self.assertEqual(self.sample()['video_url'],url)
        new=json.loads(self.raw);new['rollouts'].append({'update':2,'samples':[]})
        (self.stage/'manifest.json').write_text(json.dumps(new));merge_generated_videos(self.source,self.stage)
        self.assertEqual(self.sample()['video_url'],url)

    def test_symmetric_video_uses_received_audio_presentation_clock_only(self):
        self.publish()
        changed=json.loads(self.raw)
        changed['rollouts'][0]['samples'][0].update(conversation_mode='symmetric_streaming',pose_offset_seconds=.163)
        (self.stage/'manifest.json').write_text(json.dumps(changed))
        result=merge_generated_videos(self.source,self.stage)
        self.assertEqual(result['attached'],['u1-r0-s0'])
        self.assertEqual(self.sample()['video_offset_seconds'],.163)
        self.assertEqual(self.sample()['duration_seconds'],120)
        for offset in (None,0,-.163,.16,'invalid'):
            with self.subTest(offset=offset):
                changed['rollouts'][0]['samples'][0]['pose_offset_seconds']=offset
                (self.stage/'manifest.json').write_text(json.dumps(changed))
                result=merge_generated_videos(self.source,self.stage)
                self.assertEqual(result['attached'],[])
                self.assertEqual(len(result['rejected']),1)

    def test_regenerated_update_does_not_accept_stale_result(self):
        self.publish();new=json.loads(self.raw);new['rollouts'][0]['source_manifest_sha256']='b'*64
        (self.stage/'manifest.json').write_text(json.dumps(new));r=merge_generated_videos(self.source,self.stage)
        self.assertEqual(r['attached'],[]);self.assertIsNone(self.sample()['video_url'])

    def test_incomplete_corrupt_wrong_identity_and_wrong_clock_stay_skeleton(self):
        for changes in ({'complete':False},{'sha256':'b'*64},{'update':2},{'sample_id':'other'},
                        {'duration_seconds':119},{'frame_count':1},{'video_offset_seconds':.1},
                        {'file':'../escape.mp4'},{'width':0}):
            with self.subTest(changes=changes):
                self.reset();self.publish(**changes);r=merge_generated_videos(self.source,self.stage)
                self.assertEqual(r['attached'],[]);self.assertEqual(len(r['rejected']),1)
                self.assertIsNone(self.sample()['video_url'])

    def test_cached_public_video_is_not_changed_by_mutating_worker_output(self):
        d,_,data=self.publish();merge_generated_videos(self.source,self.stage)
        asset=self.stage/self.sample()['video_url'];(self.results/d['file']).write_bytes(b'x'*len(data))
        self.assertEqual(asset.read_bytes(),data)
        self.reset();merge_generated_videos(self.source,self.stage);self.assertEqual(asset.read_bytes(),data)

    def test_asset_is_published_before_manifest_and_old_video_remains_usable(self):
        self.publish();merge_generated_videos(self.source,self.stage);url=self.sample()['video_url']
        dest=self.root/'site';publish_samples(self.stage,dest)
        self.assertTrue((dest/url).is_file())
        self.stage.mkdir();self.reset();d,sidecar,data=self.publish();newdata=data+b' new render'
        (self.results/d['file']).write_bytes(newdata);d.update(sha256=hashlib.sha256(newdata).hexdigest(),bytes=len(newdata));sidecar.write_text(json.dumps(d))
        merge_generated_videos(self.source,self.stage);newurl=self.sample()['video_url'];self.assertNotEqual(newurl,url)
        publish_samples(self.stage,dest)
        self.assertTrue((dest/url).is_file());self.assertEqual((dest/newurl).read_bytes(),newdata)


    def hook_environment(self):
        control=self.root/'control';control.mkdir(exist_ok=True)
        (control/'settings.json').write_text(json.dumps({'mode':'live','samples_dir':str(self.source)}))
        site=self.root/'published';(site/'samples').mkdir(parents=True,exist_ok=True)
        (site/'samples/manifest.json').write_bytes(self.raw)
        self.stack=self.enterContext(mock.patch.multiple(make_demo,ROOT=control,DIST=site))
        return control,site

    def test_hook_publishes_video_without_rebuilding_audio_or_charts(self):
        self.publish();_,site=self.hook_environment()
        oldaudio=site/'samples/retained.wav';oldaudio.write_bytes(b'original received PCM')
        with mock.patch.object(make_demo,'_build',side_effect=AssertionError('unexpected full build')):
            report=make_demo.refresh_generated_videos(self.source)
            again=make_demo.refresh_generated_videos(self.source)
        self.assertTrue(report['published']);self.assertFalse(again['published'])
        self.assertEqual(oldaudio.read_bytes(),b'original received PCM')
        item=json.loads((site/'samples/manifest.json').read_text())['rollouts'][0]['samples'][0]
        self.assertTrue((site/'samples'/item['video_url']).is_file())

    def test_hook_uses_full_build_if_decoded_update_has_not_been_published(self):
        self.hook_environment();new=json.loads(self.raw);new['rollouts'].append({'update':2,'source_manifest_sha256':'b'*64})
        (self.source/'manifest.json').write_text(json.dumps(new))
        with mock.patch.object(make_demo,'_build') as build:
            result=make_demo.refresh_generated_videos(self.source)
        build.assert_called_once();self.assertTrue(result['full_rebuild'])

    def test_hook_rejects_a_source_that_is_not_currently_selected(self):
        self.hook_environment()
        with self.assertRaises(ValueError):make_demo.refresh_generated_videos(self.root/'other-run')

    def test_external_hook_waits_for_dashboard_writer_lock(self):
        self.hook_environment();entered=threading.Event();finished=threading.Event();errors=[]
        def writer():
            entered.set()
            try:make_demo.refresh_generated_videos(self.source)
            except Exception as error:errors.append(error)
            finally:finished.set()
        with make_demo.build_lock():
            worker=threading.Thread(target=writer);worker.start();self.assertTrue(entered.wait(2))
            self.assertFalse(finished.wait(.1))
        self.assertTrue(finished.wait(2));worker.join();self.assertEqual(errors,[])


if __name__=='__main__':unittest.main()
