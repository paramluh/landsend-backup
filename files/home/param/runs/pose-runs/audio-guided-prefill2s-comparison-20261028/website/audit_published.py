"""Read-only public gallery audit; credentials stay in the operator URL file."""
import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    access = urlsplit((ROOT / 'prefill-access-url.txt').read_text().strip())
    base = urlunsplit((access.scheme, access.netloc, '', '', ''))
    token = parse_qs(access.fragment)['token'][0]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path='/home/param/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',
                                    args=['--no-sandbox'])
        ctx = browser.new_context(viewport={'width': 1440, 'height': 1000})
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(base + '/guided-prefill-comparison', wait_until='domcontentloaded')
        assert ctx.request.get(base + '/guided-prefill-manifest.json').status == 401
        page.evaluate("token=>fetch('/auth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})}).then(r=>{if(!r.ok)throw Error('auth')})", token)
        result = ctx.request.get(base + '/guided-prefill-manifest.json')
        assert result.status == 200
        manifest = result.json()
        complete = [(c, m['id'], c['outputs'].get(m['id'], {})) for c in manifest['clips'] for m in manifest['models']
                    if c['outputs'].get(m['id'], {}).get('status') == 'complete']
        assert len(manifest['clips']) == 8 and len(manifest['models']) == 3
        assert complete, 'No complete cell yet'
        if args.require_all:
            assert len(complete) == 24, len(complete)
        checks = []
        for clip, recipe, output in complete:
            meta = output['metadata']
            assert meta['recipe'] == recipe and meta['force_text'] is (recipe == 'aligned-media')
            assert output['frames'] == 1500 and abs(output['seconds'] - 60) < .05
            sample = meta['sampling']
            for key, value in dict(text_temperature=1, text_top_k=0, audio_temperature=1, audio_top_k=30,
                                   audio_residual_temperature=1, audio_residual_top_k=30,
                                   pose_temperature=1, pose_top_k=30, pose_residual_temperature=1, pose_residual_top_k=50).items():
                assert sample[key] == value, (clip['id'], recipe, key, sample[key])
            g = meta['audio_guidance']
            assert g['strength'] == 3 and g['engine_scale'] == 4 and g['full_model_silence_pass']
            assert g['audit']['guided_frames'] == 727 and g['audit']['codebooks'] == list(range(32))
            assert meta['prefill_frames'] == 25 and meta['prefill_duration_seconds'] == 2
            marker = meta['prefill_marker']
            assert marker['flash']['first_frame'] == 50 and marker['flash']['frames'] == 2
            assert marker['beep']['frequency_hz'] == 880 and marker['beep']['start_seconds'] == 2
            assert marker['protected_inputs_sha256']
            for filename in ('video.mp4', 'pose.mp4', 'overlay.mp4'):
                assert marker['artifacts'][filename]['flash_validation']['min_rgb'] == 255
            for key in ('video', 'pose_video', 'pose_overlay_video'):
                url = output[key]
                assert url.startswith('/guided-prefill-media/')
                response = ctx.request.get(base + url, headers={'Range': 'bytes=0-1023'})
                assert response.status == 206 and len(response.body()) == 1024, (key, response.status)
                assert response.headers['content-range'].startswith('bytes 0-1023/')
            common = ctx.request.get(base + clip['conditioning_url'])
            assert common.status == 200
            assert len(common.json()['framewise_text_token_ids']) == 752
            assert common.json()['ground_truth_prefill']['sha256'] == meta['prefill_sha256']
            response = ctx.request.get(base + meta['conditioning_url'])
            assert response.status == 200
            prefix = response.json()
            assert prefix['recipe'] == recipe and prefix['framewise_text_forced'] is (recipe == 'aligned-media')
            assert prefix['ground_truth_prefill']['sha256'] == meta['prefill_sha256']
            assert len(prefix['prefix_text_token_ids']) == 2
            assert all(len(row) == prefix['prefix_positions'] for row in prefix['prefix_text_token_ids'])
            checks.append({'clip': clip['id'], 'recipe': recipe, 'media_range_status': 206, 'conditioning_status': 200})
        preserved = {}
        for path, expected in (('/manifest.json', 24), ('/native-manifest.json', 16), ('/guided-manifest.json', 24)):
            old = ctx.request.get(base + path).json()
            count = sum(o.get('status') == 'complete' for c in old['clips'] for o in c['outputs'].values())
            assert count == expected, (path, count)
            preserved[path] = count
        page.reload()
        page.wait_for_selector('.sampling-panel', timeout=30000)
        first = complete[0]
        cell = page.locator(f'section[data-clip="{first[0]["id"]}"] .cell').nth([m['id'] for m in manifest['models']].index(first[1]))
        cell.scroll_into_view_if_needed()
        assert 'CB1–31' in cell.inner_text() and '50' in cell.inner_text()
        assert '4 × audio logits − 3 × silence logits' in cell.inner_text()
        assert 'First 2 seconds: source audio, pose and text' in cell.inner_text()
        if first[1] == 'aligned-media':
            assert 'Not sampled' in cell.inner_text()
        else:
            assert 'Text: native' in cell.inner_text()
        tag = 'all-complete' if args.require_all else 'first-complete'
        page.screenshot(path=str(ROOT / f'prefill-{tag}.png'), full_page=False)
        flash_samples = []
        for key in ('video', 'pose_video', 'pose_overlay_video'):
            pixels = page.evaluate('''async url=>{
                const v=document.querySelector('video');v.muted=true;v.src=url;v.preload='auto';
                await new Promise((resolve,reject)=>{v.onloadeddata=resolve;v.onerror=()=>reject(Error('video load'));v.load();});
                await new Promise(resolve=>{v.onseeked=resolve;v.currentTime=2.04;});
                const c=document.createElement('canvas');c.width=4;c.height=4;const x=c.getContext('2d');x.drawImage(v,0,0,4,4);
                return Array.from(x.getImageData(0,0,4,4).data);
            }''', base + first[2][key])
            assert min(pixels) >= 245, (key, min(pixels))
            flash_samples.append(key)
        assert not errors, errors
        browser.close()
    report = dict(status='passed', completed_cells=len(complete), required_all=args.require_all,
                  media_and_conditioning=checks, older_complete_counts=preserved, page_errors=errors, actual_public_flash_views_checked=flash_samples,
                  url=base + '/guided-prefill-comparison')
    (ROOT / f'prefill-{tag}-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'completed_cells': len(complete), 'older_complete_counts': preserved}))


if __name__ == '__main__':
    main()
