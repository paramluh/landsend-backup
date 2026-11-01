"""Browser regression tests for interrupted loading, stalls and repeated seeks.

Run with /tmp/pose-comparison-browser/bin/python -m unittest
tests.test_sample_playback from the dashboard directory. Media clocks are
controlled to reproduce network races reliably without timing a real CDN.
"""
import contextlib
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[1]
FAKE_MEDIA = r"""
window.testMedia = [];
class FakeMedia extends EventTarget {
  constructor() {
    super(); this._time=0; this.duration=12; this.readyState=4;
    this.paused=true; this.seeking=false; this.muted=false;
    this.seekCount=0; this.playCount=0; this.pauseCount=0; this.pending=[];
    window.testMedia.push(this);
    queueMicrotask(()=>this.dispatchEvent(new Event('loadedmetadata')));
  }
  get currentTime(){return this._time;}
  set currentTime(v){
    this._time=v; this.seekCount++; this.seeking=true;
    queueMicrotask(()=>{this.seeking=false;this.dispatchEvent(new Event('seeked'));});
  }
  play(){
    this.playCount++;this.paused=false;
    if(this.readyState>=3)return Promise.resolve();
    return new Promise((resolve,reject)=>this.pending.push({resolve,reject}));
  }
  pause(){
    this.pauseCount++; this.paused=true;
    for(const p of this.pending.splice(0))p.reject(new DOMException('Interrupted','AbortError'));
  }
  removeAttribute(){}
  load(){}
  stall(){this.readyState=2;this.dispatchEvent(new Event('waiting'));}
  ready(){
    this.readyState=4;this.dispatchEvent(new Event('canplay'));
    for(const p of this.pending.splice(0))p.resolve();
  }
}
window.Audio=FakeMedia;
const realCreate=document.createElement.bind(document);
document.createElement=(tag,...args)=>{
  const node=realCreate(tag,...args);
  if(String(tag).toLowerCase()!=='video')return node;
  const fake=new FakeMedia();node.fakeMedia=fake;
  for(const key of ['currentTime','duration','readyState','paused','seeking'])
    Object.defineProperty(node,key,{get:()=>fake[key],set:value=>{fake[key]=value;}});
  let src='';Object.defineProperty(node,'src',{get:()=>src,set:value=>{src=value;}});
  node.play=()=>fake.play();node.pause=()=>fake.pause();node.load=()=>{};
  for(const event of ['loadedmetadata','loadeddata','canplay','seeked','waiting','ended','error'])
    fake.addEventListener(event,()=>node.dispatchEvent(new Event(event)));
  queueMicrotask(()=>fake.dispatchEvent(new Event('canplay')));
  return node;
};
const realInterval=window.setInterval;
window.setInterval=(fn,delay,...args)=>{
  if(delay===5000)window.refreshSamples=fn;
  return realInterval(fn,delay,...args);
};
window.poseRequests=[]; window.abortedPoseRequests=[];
const realFetch=window.fetch;
window.fetch=(input,options={})=>{
  const target=String(input);
  if(target.includes('motion/')){
    window.poseRequests.push(target);
    options.signal?.addEventListener('abort',()=>window.abortedPoseRequests.push(target));
  }
  return realFetch(input,options);
};
setInterval(()=>{
  for(const m of window.testMedia)if(!window.freezeTestMedia&&!m.paused&&!m.seeking&&m.readyState>=3){
    m._time=Math.min(m.duration,m._time+.02);
    if(m._time===m.duration){m.paused=true;m.dispatchEvent(new Event('ended'));}
  }
},20);
"""


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args): pass


@unittest.skipUnless(sync_playwright, 'Playwright browser required')
class PlaybackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        for name in ('index.html', 'samples.css', 'samples.js'):
            shutil.copyfile(ROOT/'pose-preview/samples'/name, root/name)
        pose = dict(fps=25, joints=['root', 'joint'], parents=[-1, 0],
                    frames=[[[0, 0, 0], [1, 1, 0]]]*300,
                    confidence=[[.2, .8]]*300)
        (root/'pose.json').write_text(json.dumps(pose))
        (root/'motion').mkdir()
        chunks=[]
        for first in range(0,300,125):
            name=f'chunk-{first//125}.json'
            chunk=dict(first_frame=first,frames=pose['frames'][first:first+125],
                       confidence=pose['confidence'][first:first+125])
            (root/'motion'/name).write_text(json.dumps(chunk))
            chunks.append(dict(url=name,first_frame=first,frame_count=len(chunk['frames'])))
        index={key:value for key,value in pose.items() if key not in ('frames','confidence')}
        index.update(schema_version=1,frame_count=300,chunk_frames=125,bounds=[0,1,0,1],chunks=chunks,
                     source_normalization=dict(root_yx=[.25,.5],shoulder_width=.1))
        (root/'motion/index.json').write_text(json.dumps(index))
        cls.samples = [dict(id=f's{i}', duration_seconds=12,
                           input_audio_url=f'a{i}.wav', output_audio_url=f'b{i}.wav',
                           pose_url='pose.json',pose_index_url='motion/index.json',prefill_seconds=2) for i in range(4)]
        cls.manifest_path = root/'manifest.json'
        cls.write_manifest()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(root)))
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(headless=True, args=['--disable-gpu'])
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/'

    @classmethod
    def write_manifest(cls):
        cls.manifest_path.write_text(json.dumps(dict(synthetic=False, rollouts=[
            dict(id='update-1',update=1,samples=cls.samples),
            dict(id='update-2',update=2,samples=cls.samples)])))

    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close();cls.tmp.cleanup()

    def setUp(self):
        self.page = self.browser.new_page(viewport={'width':1280,'height':1000})
        self.errors=[]
        self.page.on('pageerror',lambda error:self.errors.append(str(error)))
        self.page.add_init_script(FAKE_MEDIA)
        self.page.goto(self.url)
        self.page.wait_for_function('window.testMedia.length===2')
        self.card=self.page.locator('article.sample').first
        self.button=self.card.locator('.transport button')

    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.page.close()

    def time(self):
        return float(self.card.locator('input[type=range]').input_value())

    def test_pause_resume_does_not_seek_again(self):
        self.button.click();self.page.wait_for_timeout(250)
        self.button.click();paused=self.time()
        self.page.wait_for_timeout(160)
        self.assertAlmostEqual(self.time(),paused,places=3)
        counts=self.page.evaluate('testMedia.slice(0,2).map(m=>m.seekCount)')
        self.button.click();self.page.wait_for_timeout(250)
        self.assertGreater(self.time(),paused+.1)
        self.assertEqual(self.page.evaluate('testMedia.slice(0,2).map(m=>m.seekCount)'),counts)
        self.assertEqual(self.button.inner_text(),'Pause')

    def test_buffering_freezes_clock_and_automatically_resumes(self):
        self.button.click();self.page.wait_for_timeout(160)
        self.page.evaluate('testMedia[1].stall()');self.page.wait_for_timeout(60)
        paused=self.time();self.page.wait_for_timeout(200)
        self.assertAlmostEqual(self.time(),paused,places=3)
        self.assertIn('Buffering',self.card.locator('.sample-status').inner_text())
        self.assertTrue(self.page.evaluate('testMedia.slice(0,2).every(m=>m.paused)'))
        self.page.evaluate('testMedia[1].ready()');self.page.wait_for_timeout(220)
        self.assertGreater(self.time(),paused+.1)
        self.assertEqual(self.button.inner_text(),'Pause')

    def test_pause_during_loading_then_immediate_resume_ignores_old_rejection(self):
        self.page.evaluate('testMedia[1].readyState=2')
        self.button.click();self.page.wait_for_timeout(50)
        self.assertFalse(self.button.is_disabled())
        self.button.click();self.button.click()
        self.page.evaluate('testMedia[1].ready()');self.page.wait_for_timeout(240)
        self.assertEqual(self.button.inner_text(),'Pause')
        self.assertGreater(self.time(),.1)

    def test_seek_while_playing_and_paused(self):
        self.button.click();self.page.wait_for_timeout(100)
        seek=self.card.locator('input[type=range]')
        seek.evaluate("el=>{el.value=6;el.dispatchEvent(new Event('input'));}")
        self.page.wait_for_timeout(220)
        self.assertGreater(self.time(),6.1)
        self.button.click()
        seek.evaluate("el=>{el.value=2;el.dispatchEvent(new Event('input'));}")
        self.page.wait_for_timeout(120)
        self.assertAlmostEqual(self.time(),2,places=3)
        self.button.click();self.page.wait_for_timeout(200)
        self.assertGreater(self.time(),2.1)

    def test_muting_both_lanes_does_not_stop_clocks(self):
        self.card.locator('.lane button').nth(0).click()
        self.card.locator('.lane button').nth(1).click()
        self.button.click();self.page.wait_for_timeout(220)
        self.assertGreater(self.time(),.1)
        self.assertTrue(self.page.evaluate('testMedia.slice(0,2).every(m=>m.muted&&!m.paused)'))

    def test_user_pause_during_buffering_does_not_autoresume(self):
        self.button.click();self.page.wait_for_timeout(100)
        self.page.evaluate('testMedia[1].stall()');self.button.click()
        paused=self.time();self.page.evaluate('testMedia[1].ready()');self.page.wait_for_timeout(160)
        self.assertAlmostEqual(self.time(),paused,places=3)
        self.assertEqual(self.button.inner_text(),'Play')

    def test_switching_sample_stops_previous_and_switching_rollout_disposes(self):
        self.button.click();self.page.wait_for_timeout(80)
        self.page.locator('#sample').select_option('1')
        self.assertEqual(self.page.locator('article.sample').count(),1)
        self.assertEqual(self.card.get_attribute('aria-label'),'Sample 2')
        self.assertTrue(self.page.evaluate('testMedia.slice(0,2).every(m=>m.paused)'))
        self.button.click()
        self.page.locator('#rollout').select_option('1')
        self.assertEqual(self.card.get_attribute('aria-label'),'Sample 2')
        self.assertTrue(self.page.evaluate('testMedia.slice(0,4).every(m=>m.paused)'))

    def test_end_can_restart(self):
        seek=self.card.locator('input[type=range]')
        seek.evaluate("el=>{el.value=11.9;el.dispatchEvent(new Event('input'));}")
        self.button.click();self.page.wait_for_timeout(240)
        self.assertEqual(self.button.inner_text(),'Play')
        self.assertAlmostEqual(self.time(),12,places=2)
        self.button.click();self.page.wait_for_timeout(220)
        self.assertLess(self.time(),1)
        self.assertGreater(self.time(),.1)

    def test_delayed_second_lane_does_not_skip_opening(self):
        self.page.evaluate('testMedia[1].readyState=2')
        self.button.click();self.page.wait_for_timeout(600)
        self.page.evaluate('testMedia[1].ready()');self.page.wait_for_timeout(180)
        self.assertLess(self.time(),.4)
        self.assertLess(self.page.evaluate('Math.abs(testMedia[0].currentTime-testMedia[1].currentTime)'),.1)

    def test_failed_lane_is_skipped_and_all_failed_is_recoverable_ui(self):
        self.button.click();self.page.wait_for_timeout(80)
        self.page.evaluate("testMedia[0].dispatchEvent(new Event('error'))")
        self.page.wait_for_timeout(180)
        self.assertEqual(self.button.inner_text(),'Pause')
        self.assertGreater(self.time(),.1)
        self.page.evaluate("testMedia[1].dispatchEvent(new Event('error'))")
        self.assertEqual(self.button.inner_text(),'Play')
        self.assertFalse(self.button.is_disabled())
        self.assertIn('Media could not load',self.card.locator('.sample-status').inner_text())

    def test_positive_and_negative_track_offsets(self):
        manifest=json.loads(self.manifest_path.read_text())
        manifest['rollouts'][0]['samples'][0].update(input_audio_offset_seconds=.4,output_audio_offset_seconds=.4)
        self.page.route('**/manifest.json',lambda route:route.fulfill(json=manifest))
        self.page.reload();self.page.wait_for_function('testMedia.length===2')
        self.button.click();self.page.wait_for_timeout(620)
        self.assertGreater(self.time(),.45)
        self.assertLess(self.page.evaluate('testMedia[0].currentTime'),.4)
        self.button.click()
        manifest['rollouts'][0]['samples'][0].update(input_audio_offset_seconds=-2,output_audio_offset_seconds=-2)
        self.page.reload();self.page.wait_for_function('testMedia.length===2')
        self.button.click();self.page.wait_for_timeout(180)
        self.assertGreater(self.page.evaluate('testMedia[0].currentTime'),2.1)
        self.assertLess(self.time(),.4)

    def test_progressive_pose_loads_only_first_chunk_until_play(self):
        self.page.wait_for_selector('canvas[data-ready=true]')
        self.page.wait_for_timeout(100)
        requested=self.page.evaluate('poseRequests')
        self.assertEqual(len(requested),2)
        self.assertTrue(requested[0].endswith('motion/index.json'))
        self.assertTrue(requested[1].endswith('motion/chunk-0.json'))
        self.button.click()
        self.page.wait_for_function("poseRequests.some(url=>url.endsWith('chunk-1.json'))")
        self.assertFalse(self.page.evaluate("poseRequests.some(url=>url.endsWith('chunk-2.json'))"))

    def test_progressive_pose_seek_and_chunk_boundary(self):
        self.page.wait_for_selector('canvas[data-ready=true]')
        seek=self.card.locator('input[type=range]')
        seek.evaluate("el=>{el.value=10.5;el.dispatchEvent(new Event('input'));}")
        self.page.wait_for_selector('canvas[data-frame="262"]')
        self.assertFalse(self.page.evaluate("poseRequests.some(url=>url.endsWith('chunk-1.json'))"))
        seek.evaluate("el=>{el.value=4.9;el.dispatchEvent(new Event('input'));}")
        self.button.click();self.page.wait_for_timeout(300)
        self.assertGreater(int(self.card.locator('canvas').get_attribute('data-frame')),125)
        self.assertTrue(self.card.locator('.pose-state').is_hidden())

    def test_changing_sample_aborts_pending_pose_requests(self):
        pending=[]
        self.page.route('**/motion/chunk-0.json',lambda route:pending.append(route))
        self.page.reload();self.page.wait_for_function('poseRequests.length===2')
        self.page.locator('#sample').select_option('2')
        self.page.wait_for_function("abortedPoseRequests.some(url=>url.endsWith('chunk-0.json'))")
        self.assertEqual(self.card.get_attribute('aria-label'),'Sample 3')
        for route in pending:
            with contextlib.suppress(Exception): route.abort()

    def test_camera_mapping_preserves_native_framing(self):
        self.page.add_init_script("""
            window.poseArcs=[];
            const originalArc=CanvasRenderingContext2D.prototype.arc;
            CanvasRenderingContext2D.prototype.arc=function(x,y,...args){
              window.poseArcs.push([x,y,this.canvas.clientWidth,this.canvas.clientHeight]);
              return originalArc.call(this,x,y,...args);
            };
        """)
        self.page.reload();self.page.wait_for_selector('canvas[data-ready=true]')
        root,joint=self.page.evaluate('poseArcs.slice(-2)')
        x,y,width,height=root
        content_width=min(width,height*9/16);content_height=content_width*16/9
        self.assertAlmostEqual(x,width/2,places=4)
        self.assertAlmostEqual(y,(height-content_height)/2+.25*content_height,places=4)
        self.assertAlmostEqual(joint[0],(width-content_width)/2+.6*content_width,places=4)
        self.assertAlmostEqual(joint[1],(height-content_height)/2+.15*content_height,places=4)
        self.assertAlmostEqual(width/height,8/15,places=3)

    def test_legacy_pose_still_loads_without_progressive_index(self):
        manifest=json.loads(self.manifest_path.read_text())
        del manifest['rollouts'][0]['samples'][0]['pose_index_url']
        self.page.route('**/manifest.json',lambda route:route.fulfill(json=manifest))
        self.page.reload();self.page.wait_for_selector('canvas[data-ready=true]')
        self.assertEqual(self.page.evaluate('poseRequests'),[])


    def video_manifest(self, selected=0):
        manifest=json.loads(self.manifest_path.read_text())
        manifest['rollouts'][0]['samples'][selected].update(video_url=f'generated-{selected}.mp4',video_offset_seconds=0)
        self.page.route('**/manifest.json',lambda route:route.fulfill(json=manifest))
        self.page.evaluate('window.refreshSamples()')

    def test_late_video_replaces_portrait_without_resetting_playback_or_mixer(self):
        self.card.locator('.lane button').nth(0).click()
        self.button.click();self.page.wait_for_timeout(160)
        before=self.time();pauses=self.page.evaluate('testMedia.slice(0,2).map(m=>m.pauseCount)')
        bounds=self.card.locator('.pose-wrap').bounding_box()
        self.video_manifest();self.page.wait_for_selector('.pose-wrap video:not([hidden])')
        self.page.wait_for_timeout(160)
        self.assertEqual(self.page.locator('article.sample').count(),1)
        self.assertEqual(self.page.locator('video').count(),1)
        self.assertEqual(self.card.locator('.pose-wrap').bounding_box(),bounds)
        self.assertEqual(self.page.evaluate('testMedia.length'),3)
        self.assertEqual(self.page.evaluate('testMedia.slice(0,2).map(m=>m.pauseCount)'),pauses)
        self.assertTrue(self.page.evaluate('testMedia[0].muted'))
        self.assertTrue(self.page.locator('video').evaluate('v=>v.muted'))
        self.assertGreater(self.time(),before+.1)
        self.assertEqual(self.button.inner_text(),'Pause')

    def test_other_samples_video_does_not_recreate_current_player(self):
        self.button.click();self.page.wait_for_timeout(140);before=self.time()
        self.video_manifest(selected=1);self.page.wait_for_timeout(140)
        self.assertEqual(self.page.evaluate('testMedia.length'),2)
        self.assertGreater(self.time(),before+.08);self.assertEqual(self.button.inner_text(),'Pause')

    def test_video_arrives_while_paused_at_seek_position_without_autoplay(self):
        seek=self.card.locator('input[type=range]');seek.evaluate("el=>{el.value=6;el.dispatchEvent(new Event('input'));}")
        self.video_manifest();self.page.wait_for_selector('.pose-wrap video:not([hidden])')
        self.assertAlmostEqual(self.time(),6,places=3)
        self.assertAlmostEqual(self.page.locator('video').evaluate('v=>v.currentTime'),6,places=3)
        self.assertTrue(self.page.locator('video').evaluate('v=>v.paused'))
        self.assertEqual(self.button.inner_text(),'Play')

    def test_received_audio_clock_delays_video_without_delaying_audio(self):
        self.page.evaluate('window.freezeTestMedia=true')
        self.button.click()
        manifest=json.loads(self.manifest_path.read_text())
        manifest['rollouts'][0]['samples'][0].update(video_url='generated.mp4',video_offset_seconds=.163)
        self.page.route('**/manifest.json',lambda route:route.fulfill(json=manifest))
        self.page.evaluate('window.refreshSamples()')
        self.page.wait_for_selector('.pose-wrap video:not([hidden])')
        self.assertEqual(self.page.evaluate('testMedia[2].playCount'),0)
        self.assertTrue(self.page.evaluate('testMedia.slice(0,2).every(m=>!m.paused)'))
        self.page.evaluate('window.freezeTestMedia=false')
        self.page.wait_for_timeout(400)
        self.assertGreater(self.page.evaluate('testMedia[2].currentTime'),.1)
        self.assertLess(self.page.evaluate('Math.abs(testMedia[2].currentTime+.163-testMedia[0].currentTime)'),.06)
        seek=self.card.locator('input[type=range]')
        seek.evaluate("el=>{el.value=6;el.dispatchEvent(new Event('input'));}")
        self.page.wait_for_timeout(80)
        self.assertLess(self.page.evaluate('Math.abs(testMedia[2].currentTime+.163-testMedia[0].currentTime)'),.06)

    def test_video_stall_or_failure_does_not_interrupt_audio_and_restores_skeleton(self):
        self.video_manifest();self.page.wait_for_selector('.pose-wrap video:not([hidden])')
        self.button.click();self.page.wait_for_timeout(120);before=self.time()
        self.page.evaluate('testMedia[2].stall()');self.page.wait_for_timeout(120)
        self.assertGreater(self.time(),before+.06);self.assertEqual(self.button.inner_text(),'Pause')
        self.page.evaluate("testMedia[2].dispatchEvent(new Event('error'))")
        self.assertTrue(self.page.locator('video').is_hidden())
        before=self.time();self.page.wait_for_timeout(100);self.assertGreater(self.time(),before+.04)


if __name__ == '__main__': unittest.main()
