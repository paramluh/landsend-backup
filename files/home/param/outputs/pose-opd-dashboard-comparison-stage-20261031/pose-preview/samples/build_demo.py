"""Build clearly synthetic sample assets with no model, GPU or dataset access."""
import json
import math
from pathlib import Path
import shutil
import struct
import wave

SOURCE = Path(__file__).resolve().parent
JOINTS = ['pelvis', 'spine', 'neck', 'head', 'left_shoulder', 'left_elbow', 'left_hand',
          'right_shoulder', 'right_elbow', 'right_hand', 'left_hip', 'left_knee', 'left_foot',
          'right_hip', 'right_knee', 'right_foot']
PARENTS = [-1, 0, 1, 2, 2, 4, 5, 2, 7, 8, 0, 10, 11, 0, 13, 14]


def _audio(path, frequency, slot, output):
    rate, duration = 8000, 8
    values = []
    for index in range(rate * duration):
        t = index / rate
        active = (3.0 <= t < 7.4) if output else (.3 <= t < 2.8)
        envelope = max(0., math.sin(math.pi * ((t - (3 if output else .3)) / (4.4 if output else 2.5)))) if active else 0.
        pulse = .6 + .4 * math.sin(t * 13 + slot) ** 2
        signal = (math.sin(2 * math.pi * frequency * t) + .25 * math.sin(2 * math.pi * frequency * 1.5 * t)) * envelope * pulse * .08
        values.append(round(signal * 32767))
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate)
        handle.writeframes(struct.pack('<' + 'h' * len(values), *values))


def _pose(slot):
    frames = []
    for index in range(100):
        t = index / 12.5
        sway = .035 * math.sin(t * 1.8 + slot)
        gesture = max(0., math.sin((t - 3) * 1.8)) * (.5 if t >= 3 else .05)
        base = [[0, .95, 0], [0, 1.25, 0], [0, 1.52, 0], [0, 1.73, 0],
                [-.23, 1.48, 0], [-.36, 1.19 + gesture * .5, 0], [-.4 - gesture * .3, .96 + gesture, .1],
                [.23, 1.48, 0], [.36, 1.19 + gesture * .2, 0], [.4 + gesture * .15, .96 + gesture * .5, .1],
                [-.12, .92, 0], [-.12, .48, 0], [-.14, .05, .1],
                [.12, .92, 0], [.12, .48, 0], [.14, .05, .1]]
        frames.append([[round(x + sway * max(0, y - .8), 5), round(y, 5), z] for x, y, z in base])
    return dict(synthetic=True, fps=12.5, start_time_seconds=0, up_axis='y', units='meters',
                joints=JOINTS, parents=PARENTS, frames=frames)


def build_demo(output_dir):
    output = Path(output_dir)
    assets = output / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    for name in ('index.html', 'samples.css', 'samples.js'):
        shutil.copyfile(SOURCE / name, output / name)
    # Reuse four synthetic media sets across rollout selections. Their identity
    # is explicit; no files or outputs from a real training run are read.
    for slot in range(1, 5):
        for kind, frequency in [('input', 210 + 35 * slot), ('output', 310 + 45 * slot)]:
            path = assets / f'demo-{slot}-{kind}.wav'
            if not path.exists():
                _audio(path, frequency, slot, kind == 'output')
        path = assets / f'demo-{slot}-pose.json'
        path.write_text(json.dumps(_pose(slot), separators=(',', ':')))
    rollouts = []
    for update in (62, 61, 60):
        samples = [dict(id=f'demo-u{update}-s{slot}', row_id=f'DEMO-{slot:03}', orientation='A → B',
                        duration_seconds=8, fps=12.5, prefill_seconds=2,
                        input_audio_url=f'assets/demo-{slot}-input.wav',
                        output_audio_url=f'assets/demo-{slot}-output.wav',
                        pose_url=f'assets/demo-{slot}-pose.json', video_url=None,
                        text='Illustrative generated response. The audio above is a synthetic tone, not speech.')
                   for slot in range(1, 5)]
        rollouts.append(dict(id=f'demo-rollout-{update}', update=update, samples=samples))
    manifest = dict(schema_version=1, synthetic=True, rollouts=rollouts)
    temporary = output / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output / 'manifest.json')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    build_demo(parser.parse_args().output_dir)
