#!/usr/bin/env python3
"""Extract twenty CPU-only listening references; never changes inference assets."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/param/datasets/rosie-stamp-conversations-v1')
FFMPEG = '/home/param/pose-repo/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
FIRST = Path('/home/param/pose-runs/opd-next-web-20261030/conditioning/review-candidate')
selected = json.loads((ROOT / 'private/selection.json').read_text())['candidates']
records = {r['row_id']: r for r in map(json.loads, (DATA / 'training-conversations.jsonl').read_text().splitlines())}
for relative in ('public/audio', 'private/codes', 'private/provenance'):
    (ROOT / relative).mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(candidate):
    ident = candidate['id']
    record = records[candidate['row_id']]
    directory = Path(record['conversation_dir'])
    first, last = candidate['frames']
    start, end = first / 12.5, last / 12.5
    assert last - first == 125 and record['has_sam']
    source_audio = directory / 'Rosie_Stamp.sam_isolated.flac'
    source_codes = directory / 'Rosie_Stamp.sam_isolated.codes.npy'
    wav = ROOT / f'public/audio/{ident}.wav'
    codes = ROOT / f'private/codes/{ident}.codes.npy'
    provenance = ROOT / f'private/provenance/{ident}.json'
    if any(path.exists() for path in (wav, codes, provenance)):
        raise FileExistsError(f'Refusing to replace existing gallery asset {ident}')
    if ident == '01':
        previous = json.loads((FIRST / 'PROVENANCE.json').read_text())
        assert previous['conversation_id'] == record['row_id'] and previous['frames'] == [first, last]
        shutil.copy2(FIRST / 'rosie-voice-10s.wav', wav)
        shutil.copy2(FIRST / 'rosie-voice-10s.codes.npy', codes)
    else:
        subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-threads', '1',
                        '-ss', str(start), '-i', str(source_audio), '-t', '10',
                        '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', str(wav)], check=True)
        array = np.load(source_codes, allow_pickle=False)[:, first:last].copy()
        assert array.shape == (32, 125) and array.dtype == np.int16
        np.save(codes, array, allow_pickle=False)
    with wave.open(str(wav)) as audio:
        audio_info = {'sample_rate': audio.getframerate(), 'channels': audio.getnchannels(),
                      'samples': audio.getnframes(), 'seconds': audio.getnframes() / audio.getframerate()}
    if audio_info != {'sample_rate': 24000, 'channels': 1, 'samples': 240000, 'seconds': 10.0}:
        raise ValueError(f'Unexpected extracted WAV format: {audio_info}')
    vad = json.loads((directory / 'vad.json').read_text())['intervals']
    coverage = lambda intervals: sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)
    own, other = coverage(vad['Rosie Stamp']), coverage(vad['user'])
    assert abs(own - 10) < 1e-6 and other == 0
    crossbleed = json.loads((directory / 'crossbleed.json').read_text())['summary']['Rosie Stamp']
    assert crossbleed['status'] == 'clean'
    words = [w for w in json.loads((directory / 'words.json').read_text())['words']
             if w['role'] == 'assistant' and start <= float(w['start']) < end]
    transcript = ' '.join(w['text'] for w in words)
    public = {'id': ident, 'title': f'Clip {ident}', 'src': f'audio/{ident}.wav',
              'duration': 10, 'transcript': transcript}
    private = {'schema': 'rosie-listening-gallery-reference-v1', **public,
               'status': 'candidate_for_listening_not_approved_or_active',
               'source_host': 'landsend-5', 'conversation_id': record['row_id'],
               'speaker': 'Rosie Stamp', 'role': 'assistant', 'source_lane': record['assistant_source_lane'],
               'source_stream': 'sam_isolated', 'source_audio': str(source_audio), 'source_codes': str(source_codes),
               'source_codes_sha256': sha(source_codes), 'frames': [first, last], 'seconds': [start, end],
               'conversation_duration_seconds': record['duration_seconds'],
               'in_latter_half': start >= record['duration_seconds'] / 2,
               'in_last_three_minutes': start >= record['duration_seconds'] - 180,
               'Rosie_VAD_speech_seconds': own, 'user_VAD_speech_seconds': other,
               'continuous_solo_interval': candidate['continuous_solo_interval'],
               'word_boundary_selection': {k: candidate[k] for k in ('whole_words', 'sentence_start', 'sentence_end', 'start_gap_seconds', 'end_gap_seconds')},
               'crossbleed_summary': crossbleed, 'words_for_review_only': words, 'waveform': audio_info,
               'wav_sha256': sha(wav), 'codes_sha256': sha(codes),
               'codes_file': f'private/codes/{ident}.codes.npy',
               'processing': 'Contiguous10secondcutfromSAMisolated24kFLAC; monoPCM16WAV; matching125frameMimislice; no gain/effects, no GPU or reencoding.',
               'quality_review': 'Metadata-selected only; awaiting user listening selection.'}
    provenance.write_text(json.dumps(private, indent=2) + '\n')
    return public, private


with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(extract, selected))
assert len(results) == len({r[1]['conversation_id'] for r in results}) == 20
catalog = {'schema': 'rosie-listening-gallery-v1', 'count': 20, 'clips': [r[0] for r in results]}
(ROOT / 'public/clips.json').write_text(json.dumps(catalog, indent=2) + '\n')
(ROOT / 'private/manifest.json').write_text(json.dumps({'schema': 'rosie-listening-gallery-manifest-v1',
    'count': 20, 'distinct_conversations': 20, 'all_sam_isolated': True,
    'all_continuous_Rosie_VAD_10s': True, 'all_zero_user_VAD': True,
    'all_crossbleed_labels_clean': True, 'clips': [r[1] for r in results]}, indent=2) + '\n')
print(json.dumps({'count': 20, 'distinct_conversations': 20, 'directory': str(ROOT),
                  'public_manifest': str(ROOT / 'public/clips.json'), 'bytes_audio': sum((ROOT / x['src'].replace('audio/', 'public/audio/')).stat().st_size for x in catalog['clips'])}))
