import argparse
import json
from pathlib import Path
import shutil
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--media', action='store_true')
parser.add_argument('--evaluation', action='store_true')
parser.add_argument('--probes', action='store_true')
parser.add_argument('--supplemental', action='store_true')
args = parser.parse_args()
root = Path('/mnt/paramfs/param/outputs/renoise-ablation-20261015')
dist = Path(__file__).parent / 'dist'
data_dir = dist / 'data'
data_dir.mkdir(exist_ok=True)
experiment = json.loads((dist / 'experiment.json').read_text())
manifest = json.loads((root / 'manifest.json').read_text())

if args.media:
    clips = []
    for index in manifest['gallery_indices']:
        source = manifest['clips'][index]
        clip = {'id': str(index), 'uuid': source['uuid'], 'title': f"{source['split'].title()} · sample {index:02d}",
                'shortLabel': source['split'].title(), 'seed': source['seed'], 'sourceRow': source['source_row'], 'split': source['split']}
        for arm in ('baseline', 'direct'):
            video = root / arm / f'sample_{index}.mp4'
            info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                              'stream=width,height,duration,r_frame_rate,nb_frames', '-of', 'json', str(video)], check=True, capture_output=True).stdout)['streams'][0]
            assert (info['width'], info['height'], info['r_frame_rate']) == (256, 480, '25/1'), (video, info)
            clip[arm] = {'src': f'/api/bimodal-resolver/media/{arm}/sample_{index}.mp4',
                         'original': f'/api/bimodal-resolver/media/{arm}/sample_{index}.mp4',
                         'width': info['width'], 'height': info['height'], 'durationSeconds': float(info['duration']), 'sourceBytes': video.stat().st_size}
        assert abs(clip['baseline']['durationSeconds'] - clip['direct']['durationSeconds']) < .001, clip
        poster = dist / 'media' / f'clip_{index:02d}.jpg'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-threads', '1', '-ss', '1', '-i', str(root / 'baseline' / f'sample_{index}.mp4'),
                        '-frames:v', '1', '-q:v', '2', '-threads', '1', str(poster)], check=True)
        clip['poster'] = f'media/{poster.name}'
        clips.append(clip)
    experiment['clips'] = clips
    experiment['durationSeconds'] = min(c['baseline']['durationSeconds'] for c in clips)
    experiment['mediaNote'] = 'Original CRF 18 videos · 256 × 480 · 25 fps · full duration. Playback and evaluation use the same encoded files.'
    shutil.copy2(root / 'manifest.json', data_dir / 'generation-manifest.json')

if args.probes:
    probe = json.loads((root / 'history_probe_final.json').read_text())
    summary = probe['summary']
    history, current = summary['history'], summary['current']
    experiment['probes'] = {
        'summary': f"Across eight fixed GT windows, history sensitivity rose with entry time while current-state preservation fell, monotonically in all 8. The median paired t = 0.4 / t = 0.2 gain ratio was {history['median_baseline_over_direct']:.3f} for history and {current['median_baseline_over_direct']:.3f} for the common current state. These sensitivities do not establish rollout quality.",
        'columns': [{'key': 'time', 'label': 'Resolver entry time', 'digits': 1}, {'key': 'history', 'label': 'History L2 gain · median', 'digits': 3}, {'key': 'current', 'label': 'Current-state L2 gain · median', 'digits': 3}],
        'rows': [{'time': t, 'history': history['median_gain_l2'][arm], 'current': current['median_gain_l2'][arm]} for t, arm in [(0.2, 'direct'), (0.3, 'middle'), (0.4, 'baseline')]],
        'method': 'Centered finite differences with smooth spatially constant latent-channel offsets, RMS amplitude 0.04; 10 Euler steps; bf16 teacher; fixed GT context and GT-corrupted target at latent frame 256. History offsets affect two history frames. Current-state offsets are applied to the same x₀.₂ before re-noise scaling. Compare gains across entry times within each perturbation type. Zero-offset controls were exactly zero. Amplitude 0.02 vs 0.04 direction cosines were 0.974–0.980; matching Euler step size on clips 0 and 8 retained the directions of both effects. These are finite perturbation measurements, not exact Jacobians or measured rollout drift.',
        'series': {kind: [{'clip': row['index'], 'points': [[t, row[arm] / row['direct']] for t, arm in [(0.2, 'direct'), (0.3, 'middle'), (0.4, 'baseline')]]} for row in probe['paired_ratios'] if row['kind'] == kind] for kind in ('history', 'current')},
        'download': 'data/handover-probes.json'
    }
    shutil.copy2(root / 'history_probe_final.json', data_dir / 'handover-probes.json')

if args.evaluation:
    evaluation = json.loads((root / 'evaluation/results.json').read_text())
    assert evaluation['complete_pairs'] == 16
    clip_curves = json.loads((root / 'evaluation/clip_curves.json').read_text())
    definitions = [
        ('m1_mean', 'Background color drift · full rollout', 'M1 ΔE, 1–299 s. Lower means less drift from the sink.'),
        ('m1_early', 'Background color drift · early', 'M1 ΔE, 1–59 s. Lower means less drift from the sink.'),
        ('m1_late', 'Background color drift · late', 'M1 ΔE, 240–299 s. Lower means less drift from the sink.'),
        ('m1_growth', 'Background color drift · growth', 'Late minus early M1. Lower means less accumulated drift.'),
        ('m2_early', 'Signed texture log-ratio · early', 'M2, 1–59 s. Positive means more texture energy than the sink; negative means smoothing.'),
        ('m2_late', 'Signed texture log-ratio · late', 'M2, 240–299 s. Compare against GT reconstruction; zero is not an ideal natural level.'),
        ('m2_growth', 'Signed texture log-ratio · growth', 'Late minus early M2. Measures texture-energy accumulation after the initial offset.')
    ]
    aggregate = []
    for metric, label, description in definitions:
        values = evaluation['aggregate']['all']['metrics'][metric]
        aggregate.append({'label': label, 'description': description, 'baseline': values['baseline'], 'direct': values['direct'], 'gt': values['gt'],
                          'delta': values['delta_mean'], 'ci': values['delta_ci95'], 'digits': 3})
    per_clip = {str(c['index']): [{'label': label, 'description': description, 'baseline': c['baseline'][metric], 'direct': c['direct'][metric], 'gt': c['gt'][metric],
                                 'delta': c['delta'][metric], 'digits': 3} for metric, label, description in definitions] for c in evaluation['clips']}
    curves = {str(i): {metric: {arm: list(zip(c['t'], c[arm][metric])) for arm in ('baseline', 'direct', 'gt')} for metric in ('m1', 'm2')} for i, c in clip_curves.items()}
    experiment['evaluation'] = {
        'state': 'complete',
        'scope': 'All 16 pairs (8 seen + 8 unseen). Gallery selection does not affect these averages.',
        'summary': 'Paired means across all 16 clips. Differences are direct minus re-noise. Color drift and texture-energy mismatch are diagnostic measurements; neither is a complete measure of perceptual quality.',
        'method': evaluation['ci_method'] + '. The clip is the resampling unit. AutoBench at 1 Hz on original CRF 18 outputs. Early window: 1–59 s; late: 240–299 s. These exploratory intervals are not adjusted for multiple metrics. GT reconstruction values use the dataset GT latents decoded with the same VAE and CRF 18 pipeline, not original camera RGB; zero M2 only means the same texture energy as the sink, not ideal quality. Full data include seen/unseen subgroup summaries and supplemental absolute-M2 measures.',
        'aggregate': aggregate,
        'perClip': per_clip,
        'curveMetrics': [{'id': 'm1', 'label': 'M1 · background color drift', 'description': 'Observed M1 ΔE at 1 Hz. Lower means less color drift relative to the sink.'},
                         {'id': 'm2', 'label': 'M2 · signed texture log-ratio', 'description': 'Observed signed M2 at 1 Hz. Zero matches sink energy, positive indicates excess texture, and negative indicates smoothing.'}],
        'curves': curves,
        'download': 'data/evaluation-results.json'
    }
    experiment['evaluatedPairs'] = evaluation['complete_pairs']
    for src, dst in [('results.json', 'evaluation-results.json'), ('paired_metrics.csv', 'paired-metrics.csv'), ('clip_curves.json', 'clip-curves.json'), ('preregistration.json', 'evaluation-protocol.json')]:
        shutil.copy2(root / 'evaluation' / src, data_dir / dst)

if args.supplemental:
    temporal = json.loads((root / 'evaluation/temporal/summary.json').read_text())
    reference = json.loads((root / 'evaluation/common_reference_summary.json').read_text())
    assert temporal['aggregate']['all']['n'] == 16
    metrics = temporal['aggregate']['all']['metrics']
    count = sum(c['mean']['direct']['hf_delta_rms'] > c['mean']['baseline']['hf_delta_rms'] for c in temporal['clips'])
    change = metrics['hf_delta_rms']
    experiment['supplemental'] = {
        'summary': f"Direct produced larger adjacent-frame changes in high-frequency background detail in {count}/16 clips: {change['direct']:.4f} versus {change['baseline']:.4f} RMS (+{100 * (change['direct'] / change['baseline'] - 1):.1f}%). The paired difference is only {change['delta_mean']:.4f} on the 0–255 luminance scale. This measures decoded-pixel temporal variation; perceptual flicker or a quality advantage has not been established.",
        'rows': [{'label': label, 'description': description, 'baseline': metrics[key]['baseline'], 'direct': metrics[key]['direct'], 'gt': metrics[key]['gt'], 'delta': metrics[key]['delta_mean'], 'ci': metrics[key]['delta_ci95'], 'digits': 4}
                 for key, label, description in [('hf_delta_rms', 'Adjacent-frame HF change · RMS', 'Change in spatial high-frequency luminance between adjacent frames, in 8-bit luminance units.'),
                                                  ('hf_rms', 'Spatial HF detail · RMS', 'Amount of high-frequency detail; read alongside temporal change so smoothing is visible.')]],
        'method': 'Original decoded 25 fps videos, with three 10-second windows starting at 10, 145, and 285 s. One common fixed background region per clip across both arms and GT reconstruction; 3,385–23,985 measured pixels. High-frequency signal is luminance minus a Gaussian blur (σ = 1.5 px). Paired clip bootstrap intervals. Matching encoder settings does not eliminate compression dependence. Background segmentation can admit moving hands; isolated native M1 excursions are not validated drift events.',
        'download': 'data/temporal-results.json',
        'protocolDownload': 'data/temporal-protocol.json',
        'reference': {
            'summary': 'Encoded first frames differ slightly despite identical conditioning latents. Recomputing against one common decoded GT sink and common background masks gives a small positive late M2 difference. The M2 growth interval still includes zero: this supports a texture-level offset, without establishing accumulated drift or a global quality advantage.',
            'method': reference['method'] + ' These are exploratory sensitivity results; the primary AutoBench table above is unchanged.',
            'rows': [{'label': label, 'description': description, 'baseline': reference['aggregate']['all'][key]['baseline'], 'direct': reference['aggregate']['all'][key]['direct'], 'gt': reference['aggregate']['all'][key]['gt'], 'delta': reference['aggregate']['all'][key]['delta_mean'], 'ci': reference['aggregate']['all'][key]['delta_ci95'], 'digits': 4}
                     for key, label, description in [('m1_late', 'Late M1 · shared reference', 'Background color drift, 240–299 s.'), ('m2_late', 'Late M2 · shared reference', 'Signed texture log-ratio, 240–299 s.'), ('m2_growth', 'M2 growth · shared reference', 'Late minus early signed texture log-ratio.')]],
            'download': 'data/common-reference-sensitivity.json'
        }
    }
    experiment['evaluation']['summary'] = experiment['evaluation']['summary'].replace('the reported M1 and M2 differences', 'the primary AutoBench M1 and M2 differences')
    for source, target in [('temporal/summary.json', 'temporal-results.json'), ('temporal_preregistration.json', 'temporal-protocol.json'), ('common_reference_summary.json', 'common-reference-sensitivity.json'), ('decoded_sink_qc.json', 'decoded-sink-checks.json')]:
        shutil.copy2(root / 'evaluation' / source, data_dir / target)
    raw = json.loads((root / 'evaluation/raw_temporal_control.json').read_text())
    pre, encoded = raw['aggregate']['raw']['hf_delta_rms'], raw['aggregate']['encoded']['hf_delta_rms']
    experiment['supplemental']['rawControl'] = {
        'summary': f"The same direction persists before H.264 encoding in all {pre['direct_higher']}/{pre['n_pairs']} predefined pairs (0, 1, 8, 9), using the same 285–295 s window and fixed masks. Pre-encoding adjacent-frame HF change RMS was {pre['baseline']:.4f} for re-noise and {pre['direct']:.4f} for direct (+{pre['relative_change_pct']:.2f}%). The matched encoded subset was {encoded['baseline']:.4f} versus {encoded['direct']:.4f} (+{encoded['relative_change_pct']:.2f}%). The effect is not solely a codec artifact; encoding changes its magnitude. This control still includes VAE reconstruction and 8-bit quantization. It does not establish visible flicker or confirm the pre-encoding result across all 16 pairs.",
        'download': 'data/pre-encoding-control.json'
    }
    shutil.copy2(root / 'evaluation/raw_temporal_control.json', data_dir / 'pre-encoding-control.json')

if experiment['clips'] and experiment['evaluation']['state'] == 'complete':
    experiment['state'] = 'complete'
    experiment['status'] = ''
elif experiment['clips']:
    experiment['status'] = 'All eight selected video pairs are ready. AutoBench evaluation is in progress.'
(dist / 'experiment.json').write_text(json.dumps(experiment, indent=2, allow_nan=False) + '\n')
print(json.dumps({'clips': len(experiment['clips']), 'evaluation': experiment['evaluation']['state'], 'probes': bool(experiment.get('probes'))}))
