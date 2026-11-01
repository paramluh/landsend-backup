"""Publish listening artifacts without touching the existing Samples viewer."""
import html
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).parent
destination = Path('/home/param/websites/pose-opd-credit-dashboard/pose-preview/dist/diagnostics/general-audio-trials-20261031')
sources = sorted((root / 'collected').glob('**/conversations-0/manifest.json'))
manifests = [json.loads(path.read_text()) for path in sources]
samples = [(path.parent, entry) for path, manifest in zip(sources, manifests) for entry in manifest['samples']]
if len(samples) != 4 or len({entry['id'] for _, entry in samples}) != 4:
    raise ValueError('Exactly four complete conversations are required')
destination.mkdir(parents=True, exist_ok=True)
cards = []
for folder, entry in sorted(samples, key=lambda pair: pair[1]['id']):
    stem = entry['id']
    wav = destination / entry['audio']
    shutil.copy2(folder / entry['audio'], wav)
    shutil.copy2(folder / entry['tokens'], destination / entry['tokens'])
    opus = destination / (stem + '.ogg')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(wav),
        '-c:a', 'libopus', '-b:a', '128k', '-vbr', 'on', str(opus)], check=True)
    metadata = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-show_entries', 'stream=channels,sample_rate', '-of', 'json', str(opus)], text=True)
    info = json.loads(metadata)
    if info['streams'][0]['channels'] != 2 or abs(float(info['format']['duration']) - 120) > .1:
        raise ValueError('Preview must retain the two-channel two-minute recording')
    entry['preview'] = opus.name
    cards.append(f'''<article><h2>Conversation {int(stem.rsplit('-', 1)[1])}</h2>
<p class="source">{html.escape(entry['row_key'])}</p>
<p>First {entry['prefill_seconds']:.2f} seconds: recorded prefill. Generated conversation follows.</p>
<audio controls preload="none" src="{html.escape(opus.name)}"></audio>
<p><a href="{html.escape(wav.name)}" download>Original two-track WAV</a></p></article>''')
combined = {key: value for key, value in manifests[0].items() if key != 'samples'}
combined['samples'] = [entry for _, entry in sorted(samples, key=lambda pair: pair[1]['id'])]
(destination / 'manifest.json').write_text(json.dumps(combined, indent=2) + '\n')
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Starting student and general Aloe+Zoom partner</title><style>
:root{color-scheme:dark}body{font:16px/1.5 system-ui,sans-serif;max-width:840px;margin:36px auto;padding:0 20px;background:#101217;color:#eee}h1{font-size:28px}h2{font-size:20px;margin:0}article{padding:22px;margin:20px 0;background:#191d25;border:1px solid #333a47;border-radius:10px}audio{width:100%}.source{font-size:13px;overflow-wrap:anywhere;color:#aeb8c9}a{color:#9ac3ff}p{margin:10px 0}</style>
<h1>Starting student and general Aloe+Zoom partner</h1><p>Four two-minute conversational rollouts. The student is on the left channel; the frozen audio-duplex partner is on the right. Headphones help separate them.</p>
<p>Starting checkpoints, before the new training run: pose-duplex student export-211 and general ppm-aloe-zoom-cotrain partner export-253. The partner is the general model that initialized the Rosie fine-tune.</p>
<p>The student uses natural temperature-one sampling for text, audio and pose. The partner uses the settings from your website link. They hear each other’s generated audio at the restored raw level. Each recording starts with its recorded 3–5-second prefill; pose was generated but is not rendered here.</p>
''' + '\n'.join(cards) + '''<details><summary>Partner sampling settings</summary><p>Factorized text; temperature 1; relative min-p 0.1; frequency penalty 0.3 over 50 word pieces; no text grammar. Adaptive PAD: −0.1 after 3 seconds silent, otherwise 0. Audio temperature 1, cb0 top-k 50, light top-k 100. Audio CFG, repetition penalties and n-gram penalty off. No extra loudness normalization in playback.</p></details><p><a href="manifest.json">Sample details</a></p><script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));</script></html>'''
(destination / 'index.html').write_text(page)
print(destination)
