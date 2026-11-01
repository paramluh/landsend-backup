"""Browser-facing media caching, range requests and atomic publication."""
from functools import partial
import gzip
import http.client
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pose-preview'))
from make_demo import compress_sample_json,publish_samples,prepare_motion_index,publish_motion_indexes,source_pose_normalization
from serve_preview import Handler

ASSET='assets/update-000008-a80fc2d2e65013c0'


class MediaDeliveryTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'decoder';folder=self.source/ASSET;folder.mkdir(parents=True)
        self.pose=json.dumps({'frames':[[[.123456789,-.987654321,0.0]]]*1000}).encode()
        (folder/'pose.json').write_bytes(self.pose)
        self.wav=bytes(range(256))*100
        (folder/'audio.wav').write_bytes(self.wav)
        self.site=self.root/'site';staging=self.site/'samples-next';(staging/ASSET).mkdir(parents=True)
        for path in folder.iterdir():(staging/ASSET/path.name).write_bytes(path.read_bytes())
        (staging/'manifest.json').write_text('{"synthetic":false,"rollouts":[]}')
        compress_sample_json(self.source,staging)
        publish_samples(staging,self.site/'samples')
        (self.site/'opd_prog.json').write_text('{"update":2}')
        self.server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(self.site)))
        self.worker=threading.Thread(target=self.server.serve_forever,daemon=True);self.worker.start()

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.worker.join();self.temp.cleanup()

    def request(self,path,headers=None,method='GET'):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port)
        connection.request(method,path,headers=headers or {})
        response=connection.getresponse();body=response.read()
        result=response.status,dict(response.getheaders()),body
        connection.close();return result

    def test_pose_is_losslessly_compressed_and_browser_cacheable(self):
        status,headers,body=self.request('/samples/'+ASSET+'/pose.json',{'Accept-Encoding':'br, gzip'})
        self.assertEqual(status,200)
        self.assertEqual(headers['Content-Encoding'],'gzip')
        self.assertEqual(headers['Vary'],'Accept-Encoding')
        self.assertIn('immutable',headers['Cache-Control'])
        self.assertEqual(int(headers['Content-Length']),len(body))
        self.assertEqual(gzip.decompress(body),self.pose)
        self.assertLess(len(body),len(self.pose)//2)
        _,identity,raw=self.request('/samples/'+ASSET+'/pose.json',{'Accept-Encoding':'gzip;q=0, *;q=1'})
        self.assertNotIn('Content-Encoding',identity)
        self.assertEqual(raw,self.pose)

    def test_media_seeking_keeps_exact_byte_ranges(self):
        path='/samples/'+ASSET+'/audio.wav'
        status,headers,body=self.request(path,{'Range':'bytes=100-399','Accept-Encoding':'gzip'})
        self.assertEqual(status,206)
        self.assertEqual(headers['Content-Range'],f'bytes 100-399/{len(self.wav)}')
        self.assertEqual(body,self.wav[100:400])
        self.assertNotIn('Content-Encoding',headers)
        self.assertIn('immutable',headers['Cache-Control'])
        status,_,body=self.request(path,{'Range':'bytes=-17'})
        self.assertEqual(status,206);self.assertEqual(body,self.wav[-17:])
        status,headers,body=self.request(path,method='HEAD')
        self.assertEqual(status,200);self.assertEqual(body,b'')
        self.assertEqual(int(headers['Content-Length']),len(self.wav))
        status,headers,_=self.request(path,{'Range':'bytes=999999999-'})
        self.assertEqual(status,416);self.assertEqual(headers['Cache-Control'],'no-store')

    def test_pose_ranges_use_identity_representation(self):
        status,headers,body=self.request('/samples/'+ASSET+'/pose.json',{'Range':'bytes=2-9','Accept-Encoding':'gzip'})
        self.assertEqual(status,206)
        self.assertNotIn('Content-Encoding',headers)
        self.assertEqual(body,self.pose[2:10])

    def test_mutable_feeds_and_missing_assets_never_get_immutable_cache(self):
        for path in ['/samples/manifest.json','/opd_prog.json','/samples/'+ASSET+'/missing.wav']:
            _,headers,_=self.request(path)
            self.assertEqual(headers['Cache-Control'],'no-store')

    def test_refresh_keeps_existing_media_directory_and_files(self):
        destination=self.site/'samples';inode=destination.stat().st_ino
        audio=destination/ASSET/'audio.wav'
        open_audio=audio.open('rb')
        self.assertEqual(open_audio.read(12),self.wav[:12])
        staging=self.site/'samples-next';other=staging/'assets/update-000009-0123456789abcdef'
        other.mkdir(parents=True);(other/'pose.json').write_text('{}')
        (staging/'manifest.json').write_text('{"synthetic":false,"rollouts":[{"update":9}]}')
        publish_samples(staging,destination)
        self.assertEqual(destination.stat().st_ino,inode)
        self.assertTrue(audio.is_file())
        self.assertEqual(open_audio.read(),self.wav[12:]);open_audio.close()
        self.assertEqual(json.loads((destination/'manifest.json').read_text())['rollouts'][0]['update'],9)

    def test_diagnostic_video_revalidates_and_rejects_stale_ranges(self):
        directory=self.site/'diagnostics/example';directory.mkdir(parents=True)
        (directory/'clip.mp4').write_bytes(self.wav)
        status,headers,body=self.request('/diagnostics/example/clip.mp4')
        self.assertEqual(status,200);self.assertEqual(body,self.wav)
        self.assertEqual(headers['Cache-Control'],'public, max-age=0, must-revalidate')
        etag=headers['ETag']
        status,_,body=self.request('/diagnostics/example/clip.mp4',{'If-None-Match':etag})
        self.assertEqual(status,304);self.assertEqual(body,b'')
        status,_,body=self.request('/diagnostics/example/clip.mp4',{'Range':'bytes=10-19','If-Range':etag})
        self.assertEqual(status,206);self.assertEqual(body,self.wav[10:20])
        status,_,body=self.request('/diagnostics/example/clip.mp4',{'Range':'bytes=10-19','If-Range':'"stale"'})
        self.assertEqual(status,200);self.assertEqual(body,self.wav)


class MotionIndexTest(unittest.TestCase):
    def test_source_normalization_uses_output_speaker_and_validates_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);session='48f6d2d3-919f-4397-a898-e6f2cf162a21'
            folder=root/session/'seg00';folder.mkdir(parents=True)
            for speaker,root_yx,width in [('SPEAKER_00',[.4,.5],.25),('SPEAKER_01',[.61,.47],.75)]:
                (folder/f'w00.{speaker}.pose.json').write_text(json.dumps({
                    'status':'ok','session_id':session,'seg':'seg00','speaker':speaker,
                    'normalization':{'root0_yx_normalized':root_yx,'shoulder_width0_normalized':width}}))
            row=f'ALOE/s2handover-none/{session}/seg00/c00'
            sample={'row_id':row,'orientation':'assistant → user','source_audio':{
                'row_key':row,'frame0':0,'side':'assistant','flac':f'{session}/seg00/SPEAKER_00.flac'}}
            self.assertEqual(source_pose_normalization(sample,root),{'root_yx':[.61,.47],'shoulder_width':.75})
            sample['orientation']='user → assistant'
            sample['source_audio'].update(side='user',flac=f'{session}/seg00/SPEAKER_01.flac')
            self.assertEqual(source_pose_normalization(sample,root),{'root_yx':[.4,.5],'shoulder_width':.25})
            sample['source_audio']['side']='assistant'
            self.assertIsNone(source_pose_normalization(sample,root))
            sample['source_audio'].update(side='user',frame0=1)
            self.assertIsNone(source_pose_normalization(sample,root))

    def test_normalization_changes_index_identity_without_altering_pose(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'pose.json'
            original={'fps':25,'joints':['a'],'parents':[-1],'frames':[[[1,2,0]]]}
            path.write_text(json.dumps(original));before=path.read_bytes()
            first={'root_yx':[.61,.47],'shoulder_width':.75}
            second={'root_yx':[.4,.5],'shoulder_width':.25}
            cache_a,name_a=prepare_motion_index(path,root/'cache',source_normalization=first)
            cache_b,name_b=prepare_motion_index(path,root/'cache',source_normalization=second)
            self.assertNotEqual(cache_a,cache_b);self.assertNotEqual(name_a,name_b)
            index=json.loads((cache_a/name_a).read_text())
            self.assertEqual(index['source_normalization'],first)
            self.assertEqual(json.loads((cache_a/index['chunks'][0]['url']).read_text())['frames'],original['frames'])
            self.assertEqual(path.read_bytes(),before)

    def test_progressive_chunks_reassemble_exact_values_and_last_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pose=root/'pose.json'
            frames=[[[i*.123456789,-i*.987654321,i*.314159265],[1.23456789+i,.123456789,0.0]] for i in range(253)]
            confidence=[[i/253,(253-i)/253] for i in range(253)]
            original={'fps':25,'joints':['a','b'],'parents':[-1,0],'up_axis':'y','units':'test',
                      'frames':frames,'confidence':confidence,'start_time_seconds':0}
            pose.write_text(json.dumps(original));original_bytes=pose.read_bytes()
            cached,index_name=prepare_motion_index(pose,root/'cache')
            index=json.loads((cached/index_name).read_text())
            self.assertNotIn('frames',index);self.assertNotIn('confidence',index)
            self.assertEqual(index['frame_count'],253)
            self.assertEqual([c['frame_count'] for c in index['chunks']],[125,125,3])
            self.assertEqual(index['joints'],original['joints'])
            got_frames=[];got_confidence=[]
            for entry in index['chunks']:
                body=(cached/entry['url']).read_bytes()
                self.assertEqual(gzip.decompress((cached/(entry['url']+'.gzip6.gz')).read_bytes()),body)
                chunk=json.loads(body)
                self.assertEqual(chunk['first_frame'],len(got_frames))
                got_frames.extend(chunk['frames']);got_confidence.extend(chunk['confidence'])
            self.assertEqual(got_frames,frames);self.assertEqual(got_confidence,confidence)
            projected=[(p[0]+p[2]*.25,p[1]) for f in frames for p in f]
            self.assertEqual(index['bounds'],[min(p[0] for p in projected),max(p[0] for p in projected),
                                             min(p[1] for p in projected),max(p[1] for p in projected)])
            self.assertEqual(pose.read_bytes(),original_bytes)
            before=(cached/index_name).stat().st_mtime_ns
            again,name=prepare_motion_index(pose,root/'cache')
            self.assertEqual(again,cached);self.assertEqual((again/name).stat().st_mtime_ns,before)

    def test_z_up_and_absent_confidence_remain_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'pose.json'
            path.write_text(json.dumps({'fps':25,'joints':['a'],'parents':[-1],'up_axis':'z',
                                       'frames':[[[4,8,12]],[[2,-4,-8]]]}))
            cache,name=prepare_motion_index(path,root/'cache')
            index=json.loads((cache/name).read_text())
            self.assertEqual(index['bounds'],[1,6,-8,12])
            chunk=json.loads((cache/index['chunks'][0]['url']).read_text())
            self.assertNotIn('confidence',chunk)

    def test_only_served_manifest_gets_progressive_url(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';staging=root/'staging';(source/ASSET).mkdir(parents=True);staging.mkdir()
            pose=source/ASSET/'sample-pose.json'
            pose.write_text(json.dumps({'fps':25,'joints':['a'],'parents':[-1],'frames':[[[1,2,0]]]}))
            original={'rollouts':[{'samples':[{'id':'sample','pose_url':ASSET+'/sample-pose.json'}]}]}
            body=json.dumps(original);(source/'manifest.json').write_text(body);(staging/'manifest.json').write_text(body)
            publish_motion_indexes(source,staging)
            sample=json.loads((staging/'manifest.json').read_text())['rollouts'][0]['samples'][0]
            self.assertEqual(sample['pose_url'],ASSET+'/sample-pose.json')
            self.assertEqual(sample['pose_index_url'],ASSET+'/sample-pose.motion-v2.json')
            self.assertTrue((staging/sample['pose_index_url']).is_file())
            self.assertTrue((staging/(sample['pose_index_url']+'.gzip6.gz')).is_file())
            self.assertEqual((source/'manifest.json').read_text(),body)


if __name__=='__main__':unittest.main()
