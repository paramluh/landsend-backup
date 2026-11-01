'use strict';
(() => {
  const $ = selector => document.querySelector(selector);
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const format = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  const text = (node, value) => { node.textContent = value; };
  const theme = () => {
    try { document.documentElement.dataset.theme = localStorage.getItem('klweb_theme') === 'light' ? 'light' : 'dark'; } catch (_) {}
  };
  theme();
  window.addEventListener('storage', theme);
  function url(value) {
    if (!value || typeof value !== 'string') return null;
    const resolved = new URL(value, location.href);
    return ['https:', 'http:'].includes(resolved.protocol) ? resolved.href : null;
  }
  function element(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value != null) text(node, value);
    return node;
  }
  let players = [];
  let manifestSignature = null, renderedSignature = null, refreshing = false;

  class Sample {
    constructor(data, slot, demo) {
      this.data = data || {};
      this.duration = finite(this.data.duration_seconds) ? Math.max(0, this.data.duration_seconds) : 0;
      this.time = 0;
      this.playing = false;
      this.tracks = [];
      this.pose = null;
      this.disposed = false;
      this.root = element('article', 'sample');
      this.root.setAttribute('aria-label', `Sample ${slot}`);
      const head = element('div', 'sample-head');
      const title = element('div');
      title.append(element('h2', '', `Sample ${slot}`));
      const identity = this.data.row_id == null ? 'Awaiting selected row' : `Row ${this.data.row_id} · ${this.data.orientation || 'orientation unspecified'}`;
      title.append(element('div', 'identity', identity));
      head.append(title, element('span', 'tag', demo ? 'SYNTHETIC' : 'ROLLOUT'));
      this.root.append(head);
      if (!data) {
        this.root.append(element('div', 'missing', 'This sample has not been published yet.'));
        return;
      }
      const wrap = element('div', 'pose-wrap');
      this.canvas = element('canvas');
      this.canvas.setAttribute('aria-label', `Generated pose for sample ${slot}`);
      wrap.append(this.canvas, element('span', 'pose-label', demo ? 'Procedural pose · demo' : 'Generated pose'));
      this.poseState = element('span', 'pose-state', 'Loading pose…');
      wrap.append(this.poseState);
      this.root.append(wrap);
      const lanes = element('div', 'lanes');
      this.addAudio(lanes, 'input', 'User audio · dataset', this.data.input_audio_url, this.data.input_audio_offset_seconds || 0, demo);
      this.addAudio(lanes, 'output', 'Generated audio', this.data.output_audio_url, this.data.output_audio_offset_seconds || 0, demo);
      this.root.append(lanes);
      const transport = element('div', 'transport');
      this.playButton = element('button', '', 'Play');
      this.playButton.type = 'button';
      this.playButton.setAttribute('aria-label', `Play synchronized sample ${slot}`);
      this.playButton.onclick = () => this.playing ? this.pause() : this.play();
      this.seek = element('input');
      Object.assign(this.seek, {type: 'range', min: '0', max: String(this.duration), step: '.01', value: '0'});
      this.seek.setAttribute('aria-label', `Seek sample ${slot}`);
      this.seek.oninput = () => this.setTime(Number(this.seek.value));
      this.clock = element('span', 'clock');
      transport.append(this.playButton, this.seek, this.clock);
      this.root.append(transport);
      this.status = element('p', 'sample-status', 'Both audio lanes and pose use this timeline.');
      const prefill = finite(this.data.prefill_seconds) ? this.data.prefill_seconds : null;
      this.root.append(element('p', 'prefill', prefill == null ? 'Full conversation timeline' : `${prefill.toFixed(2)} s prefill · full conversation timeline`), this.status);
      if(this.data.pose_representation)this.root.append(element('p','prefill',this.data.pose_representation));
      if (this.data.text) {
        const details = element('details', 'transcript');
        details.append(element('summary', '', demo ? 'Illustrative response text' : 'Generated response text'), element('p', '', this.data.text));
        this.root.append(details);
      }
      const videoUrl = url(this.data.video_url);
      if (videoUrl) {
        const video = element('video', 'video');
        Object.assign(video, {src: videoUrl, muted: true, playsInline: true, preload: 'metadata'});
        video.setAttribute('aria-label', `Rendered pose video for sample ${slot}`);
        this.root.append(video, element('p', 'video-note', 'Rendered video · synchronized above; video audio is muted.'));
        this.addTrack(video, this.data.video_offset_seconds || 0);
      } else {
        this.root.append(element('p', 'video-note', 'Rendered video: not attached'));
      }
      this.playButton.disabled = !this.duration;
      this.loadPose(this.data.pose_url);
      this.draw();
    }

    addAudio(parent, kind, label, source, offset, demo) {
      const lane = element('div', `lane ${kind}`);
      const name = element('span', `lane-title ${kind}`, label);
      const status = element('span', 'lane-status', source ? (demo ? 'Synthetic tone' : 'Loading…') : 'Not available');
      name.append(status);
      const rail = element('div', 'lane-track');
      const fill = element('div', 'lane-fill');
      rail.append(fill);
      const toggle = element('button', '', 'Listen');
      toggle.type = 'button';
      toggle.setAttribute('aria-pressed', 'true');
      toggle.setAttribute('aria-label', `Toggle ${label.toLowerCase()}`);
      lane.append(name, rail, toggle);
      parent.append(lane);
      const sourceUrl = url(source);
      if (!sourceUrl) { toggle.disabled = true; return; }
      const audio = new Audio();
      Object.assign(audio, {preload: 'metadata', src: sourceUrl});
      const track = this.addTrack(audio, offset, status, fill);
      audio.addEventListener('loadedmetadata', () => { if (!demo) text(status, 'Ready'); });
      toggle.onclick = () => {
        audio.muted = !audio.muted;
        toggle.setAttribute('aria-pressed', String(!audio.muted));
        text(toggle, audio.muted ? 'Muted' : 'Listen');
      };
      track.status = status;
    }

    addTrack(media, offset, status, fill) {
      const track = {media, offset: finite(offset) ? offset : 0, status, fill, started: false, failed: false};
      this.tracks.push(track);
      media.addEventListener('error', () => {
        track.failed = true;
        track.started = false;
        if (status) text(status, 'Could not load');
        this.pause();
        if (this.status) text(this.status, 'A media file could not load. Playback is paused; unavailable lanes are skipped on retry.');
      });
      media.addEventListener('waiting', () => {
        if (!this.playing || this.starting || this.time < .15) return;
        this.pause();
        if (this.status) text(this.status, 'Buffering media; all lanes are paused together. Press Play to resume.');
      });
      return track;
    }

    async loadPose(source) {
      const poseUrl = url(source);
      if (!poseUrl) { text(this.poseState, 'Pose not available'); return; }
      try {
        const response = await fetch(poseUrl, {cache: 'no-store'});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const pose = await response.json();
        if (!Array.isArray(pose.joints) || !Array.isArray(pose.parents) || !Array.isArray(pose.frames) || !pose.frames.length || pose.parents.length !== pose.joints.length || !finite(pose.fps) || pose.fps <= 0) throw new Error('Invalid pose');
        if (pose.frames.some(frame => !Array.isArray(frame) || frame.length !== pose.joints.length || frame.some(p => !Array.isArray(p) || p.length < 3 || !p.slice(0, 3).every(finite)))) throw new Error('Invalid joints');
        if (pose.parents.some((p, i) => !Number.isInteger(p) || p < -1 || p >= pose.joints.length || p === i)) throw new Error('Invalid hierarchy');
        if (pose.times && (!Array.isArray(pose.times) || pose.times.length !== pose.frames.length || pose.times.some((t, i) => !finite(t) || i && t <= pose.times[i - 1]))) throw new Error('Invalid timestamps');
        if (this.disposed) return;
        this.pose = pose;
        this.bounds = [Infinity, -Infinity, Infinity, -Infinity];
        for (const frame of pose.frames) for (const point of frame) {
          const [x, y] = this.project(point);
          this.bounds[0] = Math.min(this.bounds[0], x); this.bounds[1] = Math.max(this.bounds[1], x);
          this.bounds[2] = Math.min(this.bounds[2], y); this.bounds[3] = Math.max(this.bounds[3], y);
        }
        this.poseState.hidden = true;
        this.draw();
      } catch (_) { text(this.poseState, 'Pose could not load'); }
    }

    project(p) {
      // Z-up is supported explicitly; otherwise world Y is vertical.
      return this.pose?.up_axis === 'z' ? [p[0] + p[1] * .25, p[2]] : [p[0] + p[2] * .25, p[1]];
    }

    sync(force = false) {
      for (const track of this.tracks) {
        const {media, offset} = track;
        const target = this.time - offset;
        const active = !track.failed && target >= 0 && (!finite(media.duration) || target < media.duration);
        if (!active || !this.playing) {
          media.pause(); track.started = false;
        }
        if (active && media.readyState > 0 && (force || Math.abs(media.currentTime - target) > .12)) {
          try { media.currentTime = clamp(target, 0, finite(media.duration) ? media.duration : this.duration); } catch (_) {}
        }
        if (active && this.playing && !track.started) {
          track.started = true;
          media.play().catch(() => {
            if (this.disposed || !this.playing) return;
            this.pause();
            text(this.status, 'Playback could not start. Press Play to retry both audio lanes together.');
          });
        }
        if (track.fill) track.fill.style.width = `${clamp(this.time / Math.max(this.duration, .01), 0, 1) * 100}%`;
      }
    }

    async play() {
      if (this.disposed || !this.duration) return;
      // Keep listening focused: playing a new sample pauses the other cards.
      for (const player of players) if (player !== this) player.pause();
      if (this.time >= this.duration) this.time = 0;
      this.playing = true;
      this.starting = true;
      this.playButton.disabled = true;
      text(this.playButton, 'Loading');
      try {
        await Promise.all(this.tracks.filter(track => !track.failed && this.time >= track.offset &&
          (!finite(track.media.duration) || this.time - track.offset < track.media.duration)).map(track => {
          try { track.media.currentTime = Math.max(0, this.time - track.offset); } catch (_) {}
          track.started = true;
          return track.media.play();
        }));
      } catch (_) {
        this.pause();
        if (!this.disposed) text(this.status, 'Playback could not start. Press Play to retry both audio lanes together.');
        return;
      }
      if (this.disposed || !this.playing) return;
      this.starting = false;
      this.playButton.disabled = false;
      this.anchor = performance.now() - this.time * 1000;
      text(this.playButton, 'Pause');
      text(this.status, 'Synchronized playback · toggle audio lanes to compare.');
      this.sync(true);
      const tick = () => {
        if (!this.playing || this.disposed) return;
        this.time = clamp((performance.now() - this.anchor) / 1000, 0, this.duration);
        this.sync(); this.draw();
        if (this.time >= this.duration) { this.pause(); return; }
        this.animation = requestAnimationFrame(tick);
      };
      this.animation = requestAnimationFrame(tick);
    }

    pause() {
      this.playing = false;
      this.starting = false;
      cancelAnimationFrame(this.animation);
      if (this.playButton) { text(this.playButton, 'Play'); this.playButton.disabled = !this.duration; }
      for (const track of this.tracks) { track.media.pause(); track.started = false; }
    }

    setTime(time) {
      this.time = clamp(time, 0, this.duration);
      this.anchor = performance.now() - this.time * 1000;
      this.sync(true); this.draw();
    }

    draw() {
      if (!this.canvas) return;
      this.seek.value = String(this.time);
      text(this.clock, `${format(this.time)} / ${format(this.duration)}`);
      const width = this.canvas.clientWidth || 500, height = this.canvas.clientHeight || 240;
      const dpr = Math.min(2, devicePixelRatio || 1);
      if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
        this.canvas.width = Math.round(width * dpr); this.canvas.height = Math.round(height * dpr);
      }
      const ctx = this.canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
      if (!this.pose) return;
      const local = this.time - (this.pose.start_time_seconds || 0);
      if (local < 0) return;
      let index = Math.floor(local * this.pose.fps);
      if (this.pose.times) {
        let lo = 0, hi = this.pose.times.length;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (this.pose.times[mid] <= local) lo = mid + 1; else hi = mid; }
        index = Math.max(0, lo - 1);
      }
      index = clamp(index, 0, this.pose.frames.length - 1);
      const frame = this.pose.frames[index], [xmin, xmax, ymin, ymax] = this.bounds;
      const scale = Math.min((width - 60) / Math.max(.2, xmax - xmin), (height - 50) / Math.max(.2, ymax - ymin));
      const coords = frame.map(p => {
        const [x, y] = this.project(p);
        return [width / 2 + (x - (xmin + xmax) / 2) * scale, height - 20 - (y - ymin) * scale];
      });
      const style = getComputedStyle(document.documentElement);
      ctx.strokeStyle = style.getPropertyValue('--line').trim(); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(15, height - 19); ctx.lineTo(width - 15, height - 19); ctx.stroke();
      ctx.strokeStyle = style.getPropertyValue('--output').trim(); ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 2;
      for (let j = 0; j < coords.length; j++) {
        const parent = this.pose.parents[j];
        if (parent < 0) continue;
        ctx.beginPath(); ctx.moveTo(...coords[parent]); ctx.lineTo(...coords[j]); ctx.stroke();
      }
      for (const [x, y] of coords) { ctx.beginPath(); ctx.arc(x, y, 2.3, 0, Math.PI * 2); ctx.fill(); }
    }

    dispose() {
      this.pause(); this.disposed = true;
      for (const track of this.tracks) { track.media.removeAttribute('src'); track.media.load(); }
    }
  }

  async function init() {
    if(refreshing)return;refreshing=true;
    try {
      const response = await fetch('manifest.json', {cache: 'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const manifest = await response.json();
      if (!Array.isArray(manifest.rollouts)) throw new Error('Invalid manifest');
      const signature=JSON.stringify(manifest);
      if(signature===manifestSignature)return;
      const select = $('#rollout'),previousId=select.dataset.rolloutId;
      select.replaceChildren();select.disabled=false;
      const rollouts = manifest.rollouts;
      if (!rollouts.length) {
        select.append(new Option('No rollouts yet', '')); select.disabled = true;
        text($('#notice'), 'Waiting for the first rollout. Four sample slots will appear here.');
      }
      for (let i = 0; i < rollouts.length; i++) select.append(new Option(`Update ${rollouts[i].update} · ${rollouts[i].id}`, String(i)));
      const previousIndex=rollouts.findIndex(rollout=>rollout.id===previousId);
      if(previousIndex>=0)select.value=String(previousIndex);
      const render = () => {
        const rollout = rollouts[Number(select.value)] || {};
        const currentSignature=JSON.stringify([manifest.synthetic,rollout]);
        select.dataset.rolloutId=rollout.id||'';
        if(currentSignature===renderedSignature)return;
        for (const player of players) player.dispose();
        players = [];
        $('#samples').replaceChildren();
        const samples = Array.isArray(rollout.samples) ? rollout.samples.slice(0, 4) : [];
        const demo = manifest.synthetic === true;
        if (rollouts.length) text($('#notice'), demo ? 'SYNTHETIC PREVIEW · Tone waveforms and procedural stick figures. No dataset recordings or model outputs are loaded.' : `${samples.length} of 4 samples published · ${rollout.id || 'rollout'}`);
        for (let i = 0; i < 4; i++) {
          const sample = new Sample(samples[i], i + 1, demo);
          players.push(sample); $('#samples').append(sample.root); sample.draw();
        }
        renderedSignature=currentSignature;
      };
      select.onchange = render; render();
      manifestSignature=signature;
    } catch (error) {
      if(manifestSignature!==null){text($('#notice'),'Sample refresh temporarily unavailable; retaining published media.');return;}
      text($('#notice'), 'Sample manifest is not available yet. No media has been loaded.');
      $('#rollout').disabled = true;
      $('#samples').replaceChildren();
      for (let i = 0; i < 4; i++) $('#samples').append(new Sample(null, i + 1, false).root);
    } finally {
      refreshing=false;
    }
  }
  window.addEventListener('resize', () => players.forEach(player => player.draw()));
  document.addEventListener('visibilitychange', () => { if (document.hidden) players.forEach(player => player.pause()); });
  window.addEventListener('pagehide', () => players.forEach(player => player.dispose()));
  init();
  setInterval(init,10000);
})();
