#!/usr/bin/env python3
"""One browser microphone/audio round trip against the deployed site."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from playwright.async_api import async_playwright


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    events, errors, audio = [], [], []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=[
            '--disable-gpu', '--use-fake-device-for-media-stream',
            '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'])
        context = await browser.new_context(viewport={'width': 1440, 'height': 1000},
                                            permissions=['microphone'])
        page = await context.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))

        def on_frame(payload):
            if isinstance(payload, bytes):
                audio.append(len(payload))
            else:
                try:
                    item = json.loads(payload)
                    if item.get('type') in ('ready', 'done', 'error', 'status'):
                        events.append(item)
                except (ValueError, AttributeError):
                    pass

        page.on('websocket', lambda ws: ws.on('framereceived', on_frame))
        response = await page.goto(args.url, wait_until='domcontentloaded', timeout=60000)
        assert response and response.status == 200
        await page.wait_for_function("typeof startMsg === 'function' && SCHEMA && document.querySelector('#preset').options.length > 0", timeout=60000)
        request = await page.evaluate('startMsg()')
        await page.click('#go')
        deadline = time.monotonic() + 120
        while not any(e.get('type') == 'ready' for e in events):
            assert not errors, errors
            assert not any(e.get('type') == 'error' for e in events), events
            assert time.monotonic() < deadline, 'No ready event from model'
            await asyncio.sleep(.5)
        await asyncio.sleep(8)
        await page.click('#go')
        await asyncio.sleep(3)
        assert audio and sum(audio) > 0, 'No model PCM audio returned'
        assert not errors, errors
        assert not any(e.get('type') == 'error' for e in events), events
        screenshot = args.output.with_suffix('.png')
        await page.screenshot(path=str(screenshot), full_page=False)
        result = {'passed': True, 'url': args.url, 'browser': 'Chromium, fake microphone input',
                  'model_audio_chunks': len(audio), 'model_audio_bytes': sum(audio),
                  'browser_errors': errors, 'browser_start_request': request,
                  'events': events, 'screenshot': str(screenshot)}
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'passed': True, 'model_audio_chunks': len(audio),
                          'model_audio_bytes': sum(audio), 'output': str(args.output)}))
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
