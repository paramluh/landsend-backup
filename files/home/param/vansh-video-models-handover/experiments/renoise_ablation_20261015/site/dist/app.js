'use strict';

const $ = (id) => document.getElementById(id);
const videos = {baseline: $('baseline-video'), direct: $('direct-video')};
const sides = Object.keys(videos);
let experiment;
let selected = 0;
let playing = false;
let wantsPlayback = false;
let starting = false;
let animationFrame;
let pendingPlay = 0;

const formatTime = (seconds) => `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const duration = () => {
  const values = sides.map((side) => videos[side].duration);
  return values.every((value) => Number.isFinite(value) && value > 0) ? Math.min(...values) : 0;
};
const ready = () => sides.every((side) => videos[side].readyState >= 1 && videos[side].getAttribute('src'));
const metricNumber = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';

function setText(id, text) {
  $(id).textContent = text ?? '';
}

function pausePair() {
  pendingPlay++;
  wantsPlayback = false;
  starting = false;
  sides.forEach((side) => videos[side].pause());
  playing = false;
  cancelAnimationFrame(animationFrame);
  setText('play-icon', '▶');
  setText('play-label', 'Play pair');
}

function updateTime() {
  const end = duration();
  const current = videos.baseline.currentTime || 0;
  $('timeline').max = end || 100;
  $('timeline').value = current;
  setText('time-display', `${formatTime(current)} / ${end ? formatTime(end) : '—'}`);
}

function tick() {
  updateTime();
  if (!playing) return;
  if (videos.baseline.ended || videos.direct.ended) {
    pausePair();
    return;
  }
  if (Math.abs(videos.direct.currentTime - videos.baseline.currentTime) > 0.10 && !videos.direct.seeking) videos.direct.currentTime = videos.baseline.currentTime;
  animationFrame = requestAnimationFrame(tick);
}

async function playPair() {
  if (!ready() || starting) return;
  wantsPlayback = true;
  starting = true;
  if (videos.baseline.currentTime >= duration() - 0.1) seekPair(0);
  videos.direct.currentTime = videos.baseline.currentTime;
  const request = ++pendingPlay;
  try {
    await Promise.all(sides.map((side) => videos[side].play()));
    if (request !== pendingPlay) return;
    starting = false;
    playing = true;
    setText('play-icon', 'Ⅱ');
    setText('play-label', 'Pause pair');
    tick();
  } catch (error) {
    if (request !== pendingPlay) return;
    pausePair();
    $('experiment-status').hidden = false;
    setText('experiment-status', `Playback could not start: ${error.message}. You can open either video directly.`);
  }
}

function bufferPair() {
  if (!wantsPlayback || starting) return;
  sides.forEach((side) => videos[side].pause());
  playing = false;
  cancelAnimationFrame(animationFrame);
  setText('play-icon', 'Ⅱ');
  setText('play-label', 'Buffering…');
}

function seekPair(seconds) {
  if (!ready()) return;
  const target = Math.min(Math.max(0, seconds), duration());
  sides.forEach((side) => { videos[side].currentTime = target; });
  updateTime();
}

function updateReady() {
  const enabled = ready();
  $('play-toggle').disabled = !enabled;
  $('timeline').disabled = !enabled;
  document.querySelectorAll('[data-seek]').forEach((button) => { button.disabled = !enabled; });
  updateTime();
}

function selectClip(index) {
  pausePair();
  selected = (index + experiment.clips.length) % experiment.clips.length;
  const clip = experiment.clips[selected];
  setText('clip-number', `CLIP ${(selected + 1).toString().padStart(2, '0')}`);
  setText('clip-title', clip.title || clip.id);
  setText('clip-id', clip.uuid ? `${clip.uuid} · seed ${clip.seed}` : clip.id);
  document.querySelectorAll('.clip-choice').forEach((button, i) => button.setAttribute('aria-current', i === selected ? 'true' : 'false'));
  sides.forEach((side) => {
    const media = clip[side];
    const video = videos[side];
    video.removeAttribute('src');
    video.removeAttribute('poster');
    if (media?.poster || clip.poster) video.poster = media?.poster || clip.poster;
    if (media?.src) video.src = media.src;
    video.load();
    video.playbackRate = Number($('playback-speed').value);
    $(`${side}-placeholder`).hidden = Boolean(media?.src);
    const link = $(`${side}-original`);
    link.hidden = !media?.src;
    if (media?.src) {
      link.href = media.original || media.src;
      link.textContent = media.original ? 'Full-quality video ↗' : 'Open full video ↗';
    }
  });
  updateReady();
  renderPerClip();
  renderCurve();
}

function renderClips() {
  const picker = $('clip-picker');
  experiment.clips.forEach((clip, i) => {
    const button = document.createElement('button');
    button.className = 'clip-choice';
    button.setAttribute('aria-label', `View clip ${i + 1}: ${clip.title || clip.id}`);
    button.setAttribute('aria-current', 'false');
    const poster = clip.poster || clip.baseline?.poster;
    const thumb = document.createElement(poster ? 'img' : 'div');
    thumb.className = 'clip-thumb';
    if (poster) { thumb.src = poster; thumb.alt = ''; thumb.loading = 'lazy'; }
    const label = document.createElement('span');
    label.className = 'clip-label';
    const number = document.createElement('span');
    number.textContent = (i + 1).toString().padStart(2, '0');
    const short = document.createElement('span');
    short.className = 'clip-short';
    short.textContent = clip.shortLabel || clip.title || clip.id;
    label.append(number, short);
    button.append(thumb, label);
    button.addEventListener('click', () => selectClip(i));
    picker.append(button);
  });
  $('previous-clip').disabled = experiment.clips.length < 2;
  $('next-clip').disabled = experiment.clips.length < 2;
  if (experiment.clips.length) selectClip(0);
}

function metricRow(row, confidence = false) {
  const tr = document.createElement('tr');
  const name = document.createElement('td');
  name.textContent = row.label;
  if (row.description) {
    const subtitle = document.createElement('span');
    subtitle.className = 'metric-subtitle';
    subtitle.textContent = row.description;
    name.append(subtitle);
  }
  tr.append(name);
  [row.baseline, row.direct, row.gt, row.delta ?? (row.direct - row.baseline)].forEach((value) => {
    const td = document.createElement('td');
    td.textContent = metricNumber(value, row.digits ?? 3);
    tr.append(td);
  });
  if (confidence) {
    const td = document.createElement('td');
    td.textContent = row.ci?.length === 2 ? `[${metricNumber(row.ci[0], row.digits ?? 3)}, ${metricNumber(row.ci[1], row.digits ?? 3)}]` : '—';
    tr.append(td);
  }
  return tr;
}

function renderPerClip() {
  if (!experiment?.clips?.length) return;
  const clip = experiment.clips[selected];
  setText('per-clip-title', clip.title || clip.id);
  $('per-clip-metrics').replaceChildren();
  const rows = experiment.evaluation?.perClip?.[clip.id] || [];
  rows.forEach((row) => $('per-clip-metrics').append(metricRow(row)));
  document.querySelector('.per-clip-details').hidden = !rows.length;
}

function svgElement(name, attributes = {}, content) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (content !== undefined) element.textContent = content;
  return element;
}

function renderCurve() {
  const id = experiment?.clips?.[selected]?.id;
  const metric = $('curve-metric').value;
  const curve = experiment.evaluation?.curves?.[id]?.[metric];
  const definition = experiment.evaluation?.curveMetrics?.find((item) => item.id === metric);
  $('curve-plot').replaceChildren();
  $('curve-panel').hidden = !curve;
  if (!curve) return;
  const points = [...(curve.baseline || []), ...(curve.direct || []), ...(curve.gt || [])].filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (!points.length) { $('curve-panel').hidden = true; return; }
  const width = 1100, height = 270, left = 68, right = 25, top = 14, bottom = 38;
  const end = Math.max(...points.map((p) => p[0]));
  let low = Math.min(0, ...points.map((p) => p[1]));
  let high = Math.max(0, ...points.map((p) => p[1]));
  if (high === low) high = low + 1;
  const padding = (high - low) * 0.06;
  low -= padding; high += padding;
  const x = (v) => left + v / (end || 1) * (width - left - right);
  const y = (v) => top + (high - v) / (high - low) * (height - top - bottom);
  const svg = svgElement('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `${definition?.label || metric} over time for ${id}; baseline and direct handoff`});
  for (let i = 0; i <= 4; i++) {
    const value = low + (high - low) * i / 4;
    svg.append(svgElement('line', {x1: left, y1: y(value), x2: width - right, y2: y(value), stroke: '#e0e4dc', 'stroke-width': 1}));
    svg.append(svgElement('text', {x: left - 10, y: y(value) + 4, 'text-anchor': 'end'}, value.toFixed(Math.abs(high - low) < 2 ? 2 : 1)));
    const t = end * i / 4;
    svg.append(svgElement('text', {x: x(t), y: height - 12, 'text-anchor': 'middle'}, formatTime(t)));
  }
  if (low < 0 && high > 0) svg.append(svgElement('line', {x1: left, y1: y(0), x2: width - right, y2: y(0), stroke: '#a7b3a5', 'stroke-dasharray': '4 4'}));
  ['baseline', 'direct', 'gt'].forEach((side) => {
    const valid = (curve[side] || []).filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    svg.append(svgElement('polyline', {points: valid.map((p) => `${x(p[0])},${y(p[1])}`).join(' '), fill: 'none', stroke: side === 'baseline' ? '#48705b' : side === 'direct' ? '#aa603d' : '#7e8591', 'stroke-width': 1.8, 'stroke-dasharray': side === 'gt' ? '5 4' : 'none', 'stroke-linejoin': 'round'}));
  });
  $('curve-plot').append(svg);
  setText('curve-description', definition?.description || 'Each point is an observed AutoBench measurement.');
}

function renderMetrics() {
  const evaluation = experiment.evaluation || {};
  const hasResults = evaluation.aggregate?.length > 0;
  $('metrics-pending').hidden = hasResults;
  $('metrics-results').hidden = !hasResults;
  if (!hasResults) {
    $('metrics-pending').querySelector('h3').textContent = evaluation.state === 'failed' ? 'Evaluation unavailable' : 'Measurements pending';
    $('metrics-pending').querySelector('p').textContent = evaluation.status || 'Only completed AutoBench results will be displayed here.';
    return;
  }
  setText('metrics-summary', evaluation.summary);
  setText('metrics-method', evaluation.method);
  setText('metrics-scope', evaluation.scope || 'All evaluated pairs; the gallery is a fixed subset.');
  evaluation.aggregate.forEach((row) => $('aggregate-metrics').append(metricRow(row, true)));
  (evaluation.curveMetrics || []).forEach((metric) => {
    const option = document.createElement('option');
    option.value = metric.id;
    option.textContent = metric.label;
    $('curve-metric').append(option);
  });
  if (evaluation.download) { $('metrics-download').href = evaluation.download; $('metrics-download').hidden = false; }
}

function renderProbeChart() {
  const series = experiment.probes.series;
  if (!series) return;
  const median = (values) => { const ordered = [...values].sort((a, b) => a - b); return (ordered[3] + ordered[4]) / 2; };
  const mobile = window.matchMedia('(max-width:760px)').matches;
  const width = mobile ? 360 : 1100, height = mobile ? 260 : 310, left = mobile ? 40 : 65, right = mobile ? 100 : 220, top = 20, bottom = 48;
  const values = Object.values(series).flat().flatMap((line) => line.points.map((p) => p[1]));
  const low = Math.floor(Math.min(...values) * 5) / 5, high = Math.ceil(Math.max(...values) * 5) / 5;
  const x = (v) => left + (v - 0.2) / 0.2 * (width - left - right);
  const y = (v) => top + (high - v) / (high - low) * (height - top - bottom);
  const svg = svgElement('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'Measured gain relative to t=0.2: history sensitivity increases and current-state sensitivity decreases for all eight clips as resolver entry time increases to0.4.'});
  for (let value = low; value <= high + 0.01; value += 0.2) {
    svg.append(svgElement('line', {x1: left, y1: y(value), x2: width - right, y2: y(value), stroke: Math.abs(value - 1) < 0.01 ? '#a7b3a5' : '#e0e4dc', 'stroke-dasharray': Math.abs(value - 1) < 0.01 ? '4 4' : 'none'}));
    svg.append(svgElement('text', {x: left - 12, y: y(value) + 4, 'text-anchor': 'end'}, `${value.toFixed(1)}×`));
  }
  [0.2, 0.3, 0.4].forEach((t) => svg.append(svgElement('text', {x: x(t), y: height - 20, 'text-anchor': 'middle'}, `t = ${t.toFixed(1)}`)));
  for (const [kind, lines] of Object.entries(series)) {
    const color = kind === 'history' ? '#48705b' : '#aa603d';
    lines.forEach((line) => svg.append(svgElement('polyline', {points: line.points.map((p) => `${x(p[0])},${y(p[1])}`).join(' '), fill: 'none', stroke: color, opacity: 0.24, 'stroke-width': 1.5})));
    const central = [0, 1, 2].map((i) => [lines[0].points[i][0], median(lines.map((line) => line.points[i][1]))]);
    svg.append(svgElement('polyline', {points: central.map((p) => `${x(p[0])},${y(p[1])}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 3}));
    central.forEach((p) => svg.append(svgElement('circle', {cx: x(p[0]), cy: y(p[1]), r: 4, fill: color})));
    const end = central[2];
    const label = svgElement('text', {x: x(end[0]) + (mobile ? 10 : 18), y: y(end[1]) + 4}, `${kind === 'history' ? 'History' : mobile ? 'Current' : 'Current state'} · ${end[1].toFixed(2)}×`);
    label.style.fill = color;
    svg.append(label);
  }
  if (mobile) svg.querySelectorAll('text').forEach((label) => { label.style.fontSize = '10px'; });
  $('probe-plot').replaceChildren(svg);
  $('probe-plot').hidden = false;
  $('probe-plot-note').hidden = false;
}

function renderSupplemental() {
  const supplemental = experiment.supplemental;
  if (!supplemental) return;
  $('supplemental-section').hidden = false;
  setText('supplemental-summary', supplemental.summary);
  setText('supplemental-method', supplemental.method);
  supplemental.rows.forEach((row) => $('supplemental-rows').append(metricRow(row, true)));
  $('supplemental-download').href = supplemental.download;
  if (supplemental.rawControl) {
    $('raw-control').hidden = false;
    setText('raw-control-summary', supplemental.rawControl.summary);
    $('raw-control-download').href = supplemental.rawControl.download;
  }
  if (supplemental.reference) {
    $('reference-details').hidden = false;
    setText('reference-summary', supplemental.reference.summary);
    setText('reference-method', supplemental.reference.method);
    supplemental.reference.rows.forEach((row) => $('reference-rows').append(metricRow(row, true)));
    $('reference-download').href = supplemental.reference.download;
  }
}

function renderExperiment() {
  setText('pair-count', experiment.evaluatedPairs ?? '—');
  setText('view-count', experiment.clips.length || '—');
  setText('run-length', experiment.durationSeconds ? `${Number((experiment.durationSeconds / 60).toFixed(1))}m` : '—');
  setText('selection-note', experiment.selectionNote);
  setText('media-note', experiment.mediaNote);
  setText('experiment-status', experiment.status);
  $('experiment-status').hidden = !experiment.status;
  sides.forEach((side) => {
    const strategy = experiment.strategies[side];
    setText(`${side}-title`, strategy.title);
    setText(`${side}-path`, strategy.path);
    setText(`${side}-description`, strategy.description);
  });
  $('protocol-details').replaceChildren();
  experiment.protocol.forEach((item) => {
    const div = document.createElement('div');
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = item.label;
    dd.textContent = item.value;
    div.append(dt, dd);
    $('protocol-details').append(div);
  });
  renderMetrics();
  renderSupplemental();
  if (experiment.probes?.rows?.length) {
    $('probe-section').hidden = false;
    setText('probe-summary', experiment.probes.summary);
    setText('probe-method', experiment.probes.method);
    renderProbeChart();
    experiment.probes.columns.forEach((column) => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = column.label;
      $('probe-columns').append(th);
    });
    experiment.probes.rows.forEach((row) => {
      const tr = document.createElement('tr');
      experiment.probes.columns.forEach((column) => {
        const td = document.createElement('td');
        td.textContent = typeof row[column.key] === 'number' ? metricNumber(row[column.key], column.digits ?? 3) : row[column.key];
        tr.append(td);
      });
      $('probe-rows').append(tr);
    });
    if (experiment.probes.download) { $('probe-download').href = experiment.probes.download; $('probe-download').hidden = false; }
  }
  renderClips();
}

$('play-toggle').addEventListener('click', () => wantsPlayback ? pausePair() : playPair());
$('timeline').addEventListener('input', (event) => seekPair(Number(event.target.value)));
$('previous-clip').addEventListener('click', () => selectClip(selected - 1));
$('next-clip').addEventListener('click', () => selectClip(selected + 1));
$('playback-speed').addEventListener('change', (event) => sides.forEach((side) => { videos[side].playbackRate = Number(event.target.value); }));
$('audio-source').addEventListener('change', (event) => sides.forEach((side) => { videos[side].muted = event.target.value !== side; }));
$('curve-metric').addEventListener('change', renderCurve);
document.querySelectorAll('[data-seek]').forEach((button) => button.addEventListener('click', () => {
  const end = duration();
  const points = {start: 0, early: Math.min(15, end * 0.1), middle: end * 0.5, late: end >= 240 ? 240 : end * 0.8};
  seekPair(points[button.dataset.seek]);
}));
document.querySelectorAll('[data-video]').forEach((button) => button.addEventListener('click', () => {
  const video = $(button.dataset.video);
  if (video.requestFullscreen) video.requestFullscreen();
  else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
}));
sides.forEach((side) => {
  videos[side].addEventListener('loadedmetadata', updateReady);
  videos[side].addEventListener('waiting', bufferPair);
  videos[side].addEventListener('canplay', () => {
    if (wantsPlayback && !playing && !starting && sides.every((name) => videos[name].readyState >= 3)) void playPair();
  });
  videos[side].addEventListener('ended', () => { pausePair(); updateTime(); });
  videos[side].addEventListener('error', () => {
    if (!videos[side].getAttribute('src')) return;
    pausePair();
    $(`${side}-placeholder`).hidden = false;
    setText(`${side}-placeholder`, 'Video could not load. Use the direct video link below.');
  });
  videos[side].addEventListener('seeked', updateTime);
});
document.addEventListener('keydown', (event) => {
  if (event.code !== 'Space' || /INPUT|SELECT|BUTTON|A|TEXTAREA/.test(event.target.tagName)) return;
  if (!ready()) return;
  event.preventDefault();
  wantsPlayback ? pausePair() : playPair();
});
document.addEventListener('visibilitychange', () => { if (document.hidden) pausePair(); });
window.addEventListener('resize', () => { if (experiment?.probes?.series) renderProbeChart(); });

fetch('experiment.json').then((response) => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then((data) => {
  experiment = data;
  renderExperiment();
}).catch((error) => setText('experiment-status', `Experiment manifest could not load: ${error.message}`));
