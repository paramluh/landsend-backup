"""Read-only analysis of saved pose OPD telemetry; never imports training code."""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'current': Path('/home/param/outputs/pose-opd-symmetric-b336-website-pad0-live-20261031/current').resolve(),
    'fixed_audio': Path('/home/param/outputs/pose-opd-credit-live-attempt2/snapshots/1793465773953555733-553034'),
    'asymmetric': Path('/home/param/outputs/pose-opd-symmetric-production-launch-20261031/dashboard/reference-stopped-update13'),
}
rows, configs, receipts = {}, {}, {}
for name, source in SOURCES.items():
    dest = ROOT / 'inputs' / name
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for filename in ['metrics.jsonl', 'config.json', 'plan.json']:
        p = source / filename
        if p.exists():
            content = p.read_bytes()
            (dest / filename).write_bytes(content)
            hashes[filename] = hashlib.sha256(content).hexdigest()
    rows[name] = [json.loads(line) for line in (dest / 'metrics.jsonl').read_text().splitlines()]
    configs[name] = json.loads((dest / 'config.json').read_text())
    receipts[name] = {'source': str(source), 'hashes': hashes, 'updates': [r['update'] for r in rows[name]]}
(ROOT / 'input-provenance.json').write_text(json.dumps(receipts, indent=2) + '\n')

def vals(rs, key):
    return np.array([r.get(key, np.nan) for r in rs], dtype=float)

def corr(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[good], y[good])[0, 1])

def cosine(r, group='all'):
    d, c = r[f'grad_norm/direct/{group}'], r[f'grad_norm/credit/{group}']
    total = r['gradient_norm'] if group == 'all' else r[f'grad_norm/combined/{group}']
    return (total * total - d * d - c * c) / (2 * d * c)

cur, old = rows['current'], rows['fixed_audio']
x, xo = vals(cur, 'update'), vals(old, 'update')
probe = [r for r in cur if 'grad_norm/direct/all' in r]
xp = vals(probe, 'update')
audio, word = vals(cur, 'loss/audio'), vals(cur, 'conversation/student/sampled_fraction/word')
stats = {
    'current_updates': len(cur), 'fixed_updates': len(old),
    'audio_word_correlation': corr(audio, word),
    'audio_word_difference_correlation': corr(np.diff(audio), np.diff(word)),
    'audio_word_postwarmup_difference_correlation': corr(np.diff(audio[x >= 8]), np.diff(word[x >= 8])),
    'clip_steps': [{'step': r['update'], 'preclip_norm': r['gradient_norm']} for r in cur if r['gradient_norm'] > 1],
    'adam_memory': {str(b): {'e_folding_updates': -1/math.log(b), 'half_life_updates': math.log(.5)/math.log(b)} for b in [.9, .95]},
    'probes': [{'step': r['update'], 'direct_norm': r['grad_norm/direct/all'], 'credit_norm': r['grad_norm/credit/all'],
                'combined_norm': r['gradient_norm'], 'credit_direct_ratio': r['grad_norm/credit/all']/r['grad_norm/direct/all'],
                'cosine': {g: cosine(r, g) for g in ['all', 'layers', 'head', 'tower_s', 'dec_s', 'tower_p', 'dec_p']}}
               for r in probe],
}
(ROOT / 'key-statistics.json').write_text(json.dumps(stats, indent=2) + '\n')

with (ROOT / 'all-metric-summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['run', 'metric', 'n', 'first', 'last', 'minimum', 'maximum', 'mean', 'std'])
    writer.writeheader()
    for name, rs in rows.items():
        for key in sorted(set().union(*(r.keys() for r in rs))):
            if not all(isinstance(r[key], (int, float)) for r in rs if key in r):
                continue
            z = vals(rs, key); z = z[np.isfinite(z)]
            if len(z):
                writer.writerow(dict(run=name, metric=key, n=len(z), first=z[0], last=z[-1], minimum=z.min(), maximum=z.max(), mean=z.mean(), std=z.std()))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.grid': True, 'grid.alpha': .2, 'figure.facecolor': 'white'})
blue, orange, green, purple = '#1768ac', '#d97706', '#19806b', '#8c4cb0'
fig, axs = plt.subplots(3, 2, figsize=(13, 11), layout='constrained')
fig.suptitle('Pose OPD: what the raw training telemetry shows', fontsize=17, fontweight='bold')

ax = axs[0, 0]
for key, label, color in [('loss/text', 'Text', blue), ('loss/audio', 'Audio', orange), ('loss/pose', 'Pose', green)]:
    y = vals(cur, key); ax.plot(x, y / y[0], label=label, color=color)
ax.set(title='Loss families do not all follow the same wave', ylabel='Weighted loss / its value at update 1', xlabel='Optimizer update')
ax.legend(frameon=False, ncol=3)

ax = axs[0, 1]
ax.plot(x, audio, color=orange, label='Audio loss')
ax.set(ylabel='Weighted audio loss', xlabel='Optimizer update', title='Audio loss tracks generated word-token frequency')
ax2 = ax.twinx(); ax2.grid(False); ax2.spines['right'].set_visible(True)
ax2.plot(x, 100 * word, color=blue, label='Student WORD fraction')
ax2.set_ylabel('Student sampled WORD decisions (%)')
ax.tick_params(axis='y', colors=orange); ax2.tick_params(axis='y', colors=blue)
lines = ax.lines + ax2.lines
ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=9)
ax.text(.98, .07, f"Correlation of step-to-step changes: {stats['audio_word_difference_correlation']:.3f}", transform=ax.transAxes, ha='right', fontsize=9)

ax = axs[1, 0]
for key, label, color in [('kl/lexical', 'Conditional lexical KL', blue), ('kl/category', 'Category KL', purple)]:
    y = vals(cur, key); ax.plot(x, y / y[0], color=color, label=label)
ax.set(title='Early audio-loss trough is not uniform improvement', ylabel='KL / its value at update 1', xlabel='Optimizer update')
ax.legend(frameon=False)

ax = axs[1, 1]
ax.plot(xo, vals(old, 'gradient_norm'), color='#b6bcc5', label='Previous fixed-audio run')
ax.plot(x, vals(cur, 'gradient_norm'), color=blue, label='Current run')
ax.axhline(1, color='#a52b32', ls='--', label='Clipping threshold')
ax.set(title='Global gradient norms remain bounded', ylabel='Parameter gradient L2 norm, before clipping', xlabel='Optimizer update')
ax.legend(frameon=False, fontsize=9)

ax = axs[2, 0]
for key, label, color in [('grad_norm/direct/all', 'Direct KL', blue), ('grad_norm/credit/all', 'Credit', orange), ('gradient_norm', 'Combined', green)]:
    ax.plot(xp, vals(probe, key), marker='o', label=label, color=color)
ax.set(title='Credit is a substantial part of the parameter gradient', ylabel='Global parameter gradient L2 norm', xlabel='Diagnostic update (every 10 steps)')
ax.set_xticks(xp); ax.legend(frameon=False, ncol=3)

ax = axs[2, 1]
for group, label, color in [('all', 'All parameters', green), ('layers', 'Backbone', blue), ('head', 'Text head', purple)]:
    ax.plot(xp, [cosine(r, group) for r in probe], marker='o', label=label, color=color)
ax.axhline(0, color='#888888', lw=.8)
ax.set(title='Direct and credit gradients oppose each other', ylabel='Cosine inferred from measured gradient norms', xlabel='Diagnostic update (every 10 steps)', ylim=(-1.05, .2))
ax.set_xticks(xp); ax.legend(frameon=False, fontsize=9)
fig.text(.5, -.015, f'Observed data only: no forecasts or smoothing. Current: {len(cur)} updates; fixed-audio reference: {len(old)}.\n'
         'Cosines use synchronized parameter gradients before clipping; sparse probes may include replay-rounding error.',
         ha='center', fontsize=9)
fig.savefig(ROOT / 'diagnostic-overview.png', dpi=160, bbox_inches='tight')
fig.savefig(ROOT / 'diagnostic-overview.pdf', bbox_inches='tight')

fig, axs = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
fig.suptitle('All 32 codebooks: raw losses relative to their own first update', fontsize=16, fontweight='bold')
for row, modality in enumerate(['audio', 'pose']):
    for col, (tag, label) in enumerate([('current', 'Current run'), ('fixed_audio', 'Previous fixed-audio run')]):
        rs = rows[tag]; xx = vals(rs, 'update')
        yy = np.array([vals(rs, f'loss/{modality}_cb{cb}') for cb in range(32)])
        yy = yy / yy[:, :1]
        ax = axs[row, col]
        im = ax.imshow(yy, aspect='auto', origin='lower', cmap='viridis', vmin=.25, vmax=1.05,
                       extent=(xx[0]-.5, xx[-1]+.5, -.5, 31.5), interpolation='nearest')
        ax.set(title=f'{label}: {modality}', xlabel='Optimizer update', ylabel='Codebook'); ax.grid(False)
fig.colorbar(im, ax=axs, label='Loss / first-update loss', shrink=.85)
fig.savefig(ROOT / 'all-codebook-losses.png', dpi=160, bbox_inches='tight')
fig.savefig(ROOT / 'all-codebook-losses.pdf', bbox_inches='tight')
print(json.dumps(stats, indent=2))
