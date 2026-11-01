"""Publish listening artifacts without touching the existing Samples viewer."""
import html
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).parent
destination = Path('/home/param/websites/pose-opd-credit-dashboard/pose-preview/dist/diagnostics/conversation-trials-20261031')
sources = sorted((root / 'collected').glob('**/conversations-0/manifest.json'))
manifests = [json.loads(path.read_text()) for path in sources]
samples = [(path.parent, entry) for path, manifest in zip(sources, manifests) for entry in manifest['samples']]
if len(samples) != 8 or len({entry['id'] for _, entry in samples}) != 8:
    raise ValueError('Exactly eight complete conversations are required')
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
<title>Student and partner conversations</title><style>
:root{color-scheme:dark}body{font:16px/1.5 system-ui,sans-serif;max-width:840px;margin:36px auto;padding:0 20px;background:#101217;color:#eee}h1{font-size:28px}h2{font-size:20px;margin:0}article{padding:22px;margin:20px 0;background:#191d25;border:1px solid #333a47;border-radius:10px}audio{width:100%}.source{font-size:13px;overflow-wrap:anywhere;color:#aeb8c9}a{color:#9ac3ff}p{margin:10px 0}</style>
<h1>Student and partner conversations</h1><p>Eight two-minute conversations. The student is on the left channel; the frozen audio-duplex partner is on the right. Headphones help separate them.</p>
<p>Starting checkpoints, before the new training run: pose-duplex student 211 and raw audio-duplex partner 253.</p>
<p>The student uses natural sampling. The partner uses Amu’s audio-duplex sampling settings. Pose was generated but is not rendered here. Each recording starts with its original 3–5-second prefill.</p>
''' + '\n'.join(cards) + '''<p><a href="manifest.json">Sample details</a></p><script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));</script></html>'''
(destination / 'index.html').write_text(page)
print(destination)
