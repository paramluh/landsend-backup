import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright

ROOT = Path('/home/param/pose-demo-website/comparison')
OUT = Path('/home/param/pose-runs/pose-overlay-ui-audit-20261029')
fixture = Path('/home/param/pose-runs/native-comparison-20261028/outputs/01/compact-all/video.mp4').read_bytes()
output = dict(status='complete', video='/fixture-original.mp4', pose_video='/fixture-pose.mp4',
              pose_overlay_video='/fixture-overlay.mp4', sha256='fixture', seconds=60., width=256, height=480, frames=1500, metadata={})
manifest = dict(updated_at='one', title='Pose view checks', note='Browser behavior fixture',
                models=[dict(id='compact-all', label='Compact', description='Fixture')],
                clips=[dict(id='clip-01', title='Clip 01', transcript='Hello world.', outputs={'compact-all': output})])
req = dict(recipe='compact-all', clip_id='clip-01', text_mode='native', audio_temp=1, pose_temp=1,
           audio_topk=0, pose_topk=0, text_temp=1, text_topk=0, seed=1, video_seed=2, duration_seconds=60)
job = dict(id='a'*32, status='complete', stage='complete', updated_at='one', created_at='2026-10-29T00:00:00Z',
           request=req, **{k:v for k,v in output.items() if k not in ('status','sha256')})
config = dict(csrf='test', models=manifest['models'], clips=manifest['clips'],
              options=dict(native_text_recipes=['compact-all','xabi-original'], seed=dict(defaults={'clip-01':1})))

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/home/param/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',
                              headless=True, args=['--disable-gpu', '--no-sandbox', '--autoplay-policy=no-user-gesture-required'])
    page = browser.new_page(viewport=dict(width=1280,height=1000))
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    def route(r):
        path=urlsplit(r.request.url).path
        if path.endswith('.mp4'):
            return r.fulfill(status=200,content_type='video/mp4',body=fixture)
        if path in ['/pose-player.js','/pose-player.css']:
            return r.fulfill(status=200,content_type='application/javascript' if path.endswith('.js') else 'text/css',body=(ROOT/path[1:]).read_bytes())
        if path in ['/comparison','/native-comparison','/']:
            return r.fulfill(status=200,content_type='text/html',body=(ROOT/('studio.html' if path=='/' else 'index.html')).read_bytes())
        data={'/manifest.json':manifest,'/native-manifest.json':manifest,'/api/config':config,
              '/api/requests':dict(requests=[job]),'/api/capacity':dict(available=False)}.get(path,{})
        r.fulfill(status=200,content_type='application/json',body=json.dumps(data))
    page.route('http://pose-ui.test/**',route)
    cases=[]
    for path in ['/comparison','/native-comparison','/']:
        page.goto('http://pose-ui.test'+path)
        page.wait_for_selector('.pose-player[data-view="overlay"]')
        page.locator('.pose-main-video').evaluate('(v)=>{v.load()}')
        page.wait_for_function('document.querySelector(".pose-main-video").readyState>=2')
        page.locator('.pose-main-video').evaluate('(v)=>{v.currentTime=3.2;v.muted=true;v.volume=.35;v.playbackRate=1.25;v.pause()}')
        page.wait_for_function('!document.querySelector(".pose-main-video").seeking')
        print('BEFORE',page.locator('.pose-main-video').evaluate('(v)=>({time:v.currentTime,ready:v.readyState,seeking:v.seeking,paused:v.paused,duration:v.duration})'),flush=True)
        for mode in ['pose','original','overlay']:
            page.locator(f'button[data-view="{mode}"]').click()
            page.wait_for_function('!document.querySelector(".pose-player").hasAttribute("aria-busy") && document.querySelector(".pose-main-video").readyState>=2 && !document.querySelector(".pose-main-video").seeking')
            state=page.locator('.pose-main-video').evaluate('(v)=>({time:v.currentTime,paused:v.paused,muted:v.muted,volume:v.volume,rate:v.playbackRate,src:v.currentSrc})')
            print('AFTER',state,flush=True)
            assert abs(state['time']-3.2)<.05 and state['paused'] and state['muted'] and abs(state['volume']-.35)<1e-9 and state['rate']==1.25,state
            assert mode in state['src'] if mode!='original' else 'original' in state['src']
        page.locator('button[data-view="pose"]').click()
        page.wait_for_function('!document.querySelector(".pose-player").hasAttribute("aria-busy")')
        if path=='/':
            job['updated_at']='two';page.evaluate('refresh()')
        else:
            manifest['updated_at']='two';page.evaluate('update()')
        page.wait_for_timeout(250)
        assert page.locator('.pose-player').get_attribute('data-view')=='pose'
        page.locator('.pose-main-video').evaluate('(v)=>v.play()')
        page.wait_for_timeout(200)
        page.locator('button[data-view="overlay"]').click()
        page.wait_for_function('!document.querySelector(".pose-player").hasAttribute("aria-busy") && !document.querySelector(".pose-main-video").paused')
        page.wait_for_timeout(200)
        assert page.locator('.pose-main-video').evaluate('(v)=>v.currentTime')>3.3
        page.locator('.pose-main-video').evaluate('(v)=>v.pause()')
        assert page.locator('.pose-view-footer a').get_attribute('href')=='/fixture-overlay.mp4'
        cases.append(dict(path=path,default_overlay=True,paused_seek_mute_volume_rate_preserved=True,poll_preserves_selection=True,playing_switch_resumes=True))
        page.screenshot(path=str(OUT/('studio-desktop.png' if path=='/' else path.strip('/')+'-desktop.png')),full_page=True)
        if path=='/':
            assert page.locator('#audio_temp').count()==page.locator('#pose_temp').count()==page.locator('#text_mode').count()==1
            page.set_viewport_size(dict(width=390,height=844));page.screenshot(path=str(OUT/'studio-mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'), 'Mobile horizontal overflow'
    browser.close()
    assert not errors,errors
    result=dict(status='pass',cases=cases,javascript_errors=errors,fixture='Existing video reused for all three source URLs; this tests player behavior, not raster appearance',gpu_disabled=True,new_jobs_submitted=0)
    (OUT/'browser-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
