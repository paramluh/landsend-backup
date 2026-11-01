'use strict';
(() => {
  const $ = selector => document.querySelector(selector);
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  // Match the monoplex playground's current PoseOverlay renderer.
  function poseColor(confidence) {
    const c = clamp(confidence, 0, 1);
    const t = clamp((c - .2) / .4, 0, 1);
    return `rgb(${Math.round(255 * Math.min(1, 2 * (1 - t)))},${Math.round(255 * Math.min(1, 2 * t))},70)`;
  }
  const handEdges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]];
  const monoEdges = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],
    ...handEdges.map(([a,b])=>[a+13,b+13]), [9,13], ...handEdges.map(([a,b])=>[a+34,b+34]), [10,34]];
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
  // Keep the user's overlay choice while switching steps/samples or refreshing.
  let poseOverlayEnabled = false;
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
      this.operation = 0;
      this.buffering = false;
      this.starting = false;
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
      this.poseWrap = wrap;
      this.canvas = element('canvas');
      this.canvas.setAttribute('aria-label', `Generated pose for sample ${slot}`);
      this.poseLabel = element('span', 'pose-label');
      this.demo = demo;
      wrap.append(this.canvas, this.poseLabel);
      this.poseState = element('span', 'pose-state', 'Loading pose…');
      wrap.append(this.poseState);
      this.overlayButton = element('button', 'pose-overlay-toggle', 'Pose overlay');
      this.overlayButton.type = 'button';
      this.overlayButton.hidden = true;
      this.overlayButton.setAttribute('aria-label', 'Superimpose pose on generated video');
      this.overlayButton.setAttribute('aria-pressed', String(poseOverlayEnabled));
      this.overlayButton.onclick = () => {
        poseOverlayEnabled = !poseOverlayEnabled;
        this.ensurePose(); this.draw();
      };
      wrap.append(this.overlayButton);
      this.root.append(wrap);
      this.poseLegend = element('div', 'pose-legend');
      this.poseLegend.hidden = true;
      this.root.append(this.poseLegend);
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
      if (this.data.text) {
        const details = element('details', 'transcript');
        details.append(element('summary', '', demo ? 'Illustrative response text' : 'Generated response text'), element('p', '', this.data.text));
        this.root.append(details);
      }
      this.attachVideo();
      const mediaPanel = element('div', 'sample-media');
      mediaPanel.append(wrap, transport, this.poseLegend, this.status, ...this.root.querySelectorAll(':scope > .prefill'));
      const detailPanel = element('div', 'sample-detail');
      detailPanel.append(...this.root.children);
      this.root.append(mediaPanel, detailPanel);
      this.playButton.disabled = !this.duration;
      // Load the selected sample when visible, or immediately on Play, and
      // cancel obsolete requests whenever the selected sample changes.
      if ('IntersectionObserver' in window) {
        this.poseObserver = new IntersectionObserver(entries => {
          if (entries.some(entry => entry.isIntersecting)) this.ensurePose();
        }, {rootMargin: '150px'});
        this.poseObserver.observe(this.root);
      } else this.ensurePose();
      this.draw();
    }

    attachVideo() {
      const videoUrl = url(this.data.video_url);
      if (!videoUrl || this.video?.src === videoUrl) return;
      if (this.videoTrack) {
        this.videoTrack.failed = true; this.video.pause();
        this.video.removeAttribute('src'); this.video.load(); this.video.remove();
        this.tracks = this.tracks.filter(track => track !== this.videoTrack);
      }
      this.videoVisible = false;
      const video = element('video', 'video');
      this.video = video;
      Object.assign(video, {muted: true, playsInline: true, preload: 'auto', hidden: true});
      video.setAttribute('aria-label', 'Generated video for this sample');
      this.poseWrap.append(video);
      const track = this.addTrack(video, this.data.video_offset_seconds || 0);
      track.optional = true; this.videoTrack = track;
      video.addEventListener('loadedmetadata', () => {
        if (this.disposed || this.video !== video) return;
        const target = clamp(this.time - track.offset, 0, this.duration);
        if (Math.abs(video.currentTime - target) > .04) video.currentTime = target;
      });
      const ready = () => {
        if (this.disposed || this.video !== video || track.failed || video.seeking || video.readyState < 2) return;
        this.videoVisible = true; video.hidden = false;
        this.poseState.hidden = true;
        this.draw();
        if (this.playing) this.startOptionalTrack(track);
      };
      for (const name of ['loadeddata', 'canplay', 'seeked']) video.addEventListener(name, ready);
      video.addEventListener('error', () => {
        if (this.video !== video) return;
        video.hidden = true; this.videoVisible = false;
        if (this.pose) this.poseLegend.hidden = false;
        this.ensurePose(); this.draw();
      });
      video.src = videoUrl;
    }

    updateVideo(data) {
      this.data = data; this.attachVideo(); this.draw();
    }

    videoPoseNormalization() {
      const normal = this.data.video_pose_normalization;
      return normal && Array.isArray(normal.root_yx) && normal.root_yx.length === 2 &&
        normal.root_yx.every(finite) && finite(normal.shoulder_width) && normal.shoulder_width > 0 &&
        finite(normal.width) && normal.width > 0 && finite(normal.height) && normal.height > 0 ? normal : null;
    }

    startOptionalTrack(track) {
      if (!this.playing || this.disposed || track.failed || track.media.seeking || track.media.readyState < 3) return;
      if (this.time < track.offset) return;
      track.started = true;
      const operation = this.operation;
      Promise.resolve(track.media.play()).then(() => {
        if (this.disposed || !this.playing || operation !== this.operation) track.media.pause();
      }).catch(() => { track.started = false; });
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
        if (this.playing && !track.optional) this.startPlayback();
      });
      media.addEventListener('waiting', () => {
        if (!this.playing || this.starting || track.optional) return;
        this.holdForBuffering();
      });
      for (const event of ['canplay', 'seeked', 'loadedmetadata']) {
        media.addEventListener(event, () => this.resumeWhenReady());
      }
      media.addEventListener('ended', () => {
        if (this.playing) this.readClock();
        track.started = false;
      });
      return track;
    }

    ensurePose() {
      if (this.poseRequested || this.disposed) return;
      this.poseRequested = true;
      this.poseObserver?.disconnect();
      this.loadPose(this.data.pose_index_url || this.data.pose_url, Boolean(this.data.pose_index_url));
    }

    poseLegendFor(hasConfidence) {
      this.poseLegend.replaceChildren();
      if (hasConfidence) {
        const ramp = element('span', 'confidence-ramp');
        ramp.setAttribute('aria-hidden', 'true');
        this.poseLegend.append(ramp, element('span', '', 'Pose confidence: red 0.2 · yellow 0.4 · green ≥0.6; fades below 0.2'));
      } else {
        this.poseLegend.append(element('span', '', 'Pose confidence unavailable · neutral gray'));
      }
      this.poseLegend.hidden = Boolean(this.videoVisible && !poseOverlayEnabled);
    }

    validFrames(frames, confidence) {
      const joints = this.pose.joints.length;
      if (!Array.isArray(frames) || frames.some(frame => !Array.isArray(frame) || frame.length !== joints ||
          frame.some(point => !Array.isArray(point) || point.length < 3 || !point.slice(0, 3).every(finite)))) return false;
      return confidence == null || (Array.isArray(confidence) && confidence.length === frames.length &&
        confidence.every(frame => Array.isArray(frame) && frame.length === joints && frame.every(finite)));
    }

    async loadPose(source, indexed = false) {
      const poseUrl = url(source);
      if (!poseUrl) { text(this.poseState, 'Pose not available'); return; }
      try {
        this.poseRequest = new AbortController();
        const response = await fetch(poseUrl, {signal: this.poseRequest.signal});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const pose = await response.json();
        if (!Array.isArray(pose.joints) || !pose.joints.length || !Array.isArray(pose.parents) ||
            pose.parents.length !== pose.joints.length || !finite(pose.fps) || pose.fps <= 0) throw new Error('Invalid pose');
        if (pose.parents.some((p, i) => !Number.isInteger(p) || p < -1 || p >= pose.joints.length || p === i)) throw new Error('Invalid hierarchy');
        const count = indexed ? pose.frame_count : pose.frames?.length;
        if (!Number.isInteger(count) || count < 1) throw new Error('Invalid pose length');
        if (pose.times && (!Array.isArray(pose.times) || pose.times.length !== count || pose.times.some((t, i) => !finite(t) || i && t <= pose.times[i - 1]))) throw new Error('Invalid timestamps');
        if (this.disposed) return;
        this.pose = pose;
        const normalization = pose.source_normalization;
        if (normalization && (!Array.isArray(normalization.root_yx) || normalization.root_yx.length !== 2 ||
            !normalization.root_yx.every(finite) || !finite(normalization.shoulder_width) || normalization.shoulder_width <= 0))
          throw new Error('Invalid source normalization');
        if (indexed) {
          if (pose.schema_version !== 1 || !Number.isInteger(pose.chunk_frames) || pose.chunk_frames < 1 ||
              !Array.isArray(pose.bounds) || pose.bounds.length !== 4 || !pose.bounds.every(finite) ||
              pose.bounds[0] > pose.bounds[1] || pose.bounds[2] > pose.bounds[3] ||
              !Array.isArray(pose.chunks) || pose.chunks.length !== Math.ceil(count / pose.chunk_frames) ||
              pose.chunks.some((chunk, i) => chunk.first_frame !== i * pose.chunk_frames ||
                chunk.frame_count !== Math.min(pose.chunk_frames, count - chunk.first_frame) || !url(chunk.url)))
            throw new Error('Invalid pose index');
          this.poseIndexUrl = poseUrl;
          this.poseChunks = new Map(); this.chunkRequests = new Map(); this.chunkErrors = new Map();
          this.bounds = pose.bounds;
          await this.requestPoseChunk(this.poseFrameIndex());
        } else {
          if (!this.validFrames(pose.frames, pose.confidence)) throw new Error('Invalid pose frames');
          this.poseLegendFor(Boolean(pose.confidence));
          this.bounds = [Infinity, -Infinity, Infinity, -Infinity];
          for (const frame of pose.frames) for (const point of frame) {
            const [x, y] = this.project(point);
            this.bounds[0] = Math.min(this.bounds[0], x); this.bounds[1] = Math.max(this.bounds[1], x);
            this.bounds[2] = Math.min(this.bounds[2], y); this.bounds[3] = Math.max(this.bounds[3], y);
          }
          this.poseState.hidden = true;
        }
        if (this.disposed) return;
        this.draw();
      } catch (error) {
        if (!this.disposed && error.name !== 'AbortError') {
          this.pose = null;
          text(this.poseState, 'Pose could not load'); this.poseState.hidden = false;
        }
      }
    }

    requestPoseChunk(frameIndex) {
      if (!this.poseChunks || this.disposed) return Promise.resolve();
      const number = Math.floor(frameIndex / this.pose.chunk_frames);
      if (!this.pose.chunks[number]) return Promise.resolve();
      if (this.poseChunks.has(number)) return Promise.resolve(this.poseChunks.get(number));
      if (this.chunkRequests.has(number)) return this.chunkRequests.get(number);
      if (performance.now() - (this.chunkErrors.get(number) ?? -Infinity) < 3000) return Promise.resolve();
      const metadata = this.pose.chunks[number];
      const pending = (async () => {
        try {
          const response = await fetch(new URL(metadata.url, this.poseIndexUrl), {signal: this.poseRequest.signal});
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const chunk = await response.json();
          if (this.disposed) return;
          if (chunk.first_frame !== metadata.first_frame || chunk.frames?.length !== metadata.frame_count ||
              !this.validFrames(chunk.frames, chunk.confidence)) throw new Error('Invalid pose chunk');
          this.poseChunks.set(number, chunk); this.chunkErrors.delete(number);
          this.poseLegendFor(Boolean(chunk.confidence));
          this.draw();
          return chunk;
        } catch (error) {
          if (!this.disposed && error.name !== 'AbortError') {
            this.chunkErrors.set(number, performance.now());
            if (Math.floor(this.poseFrameIndex() / this.pose.chunk_frames) === number) {
              text(this.poseState, 'Pose is still loading; retrying…'); this.poseState.hidden = false;
            }
            clearTimeout(this.poseRetry);
            this.poseRetry = setTimeout(() => { if (!this.disposed) this.draw(); }, 3100);
          }
        } finally { this.chunkRequests.delete(number); }
      })();
      this.chunkRequests.set(number, pending);
      return pending;
    }

    poseFrameIndex(time = this.time) {
      const local = time - (this.pose.start_time_seconds || 0);
      let index = Math.floor(local * this.pose.fps);
      if (this.pose.times) {
        let lo = 0, hi = this.pose.times.length;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (this.pose.times[mid] <= local) lo = mid + 1; else hi = mid; }
        index = Math.max(0, lo - 1);
      }
      return clamp(index, 0, (this.pose.frame_count ?? this.pose.frames.length) - 1);
    }

    project(p) {
      // Z-up is supported explicitly; otherwise world Y is vertical.
      return this.pose?.up_axis === 'z' ? [p[0] + p[1] * .25, p[2]] : [p[0] + p[2] * .25, p[1]];
    }

    activeTracks() {
      return this.tracks.filter(track => !track.failed && (!track.optional || (track.media.readyState >= 3 && !track.media.seeking)) && this.time >= track.offset &&
        (!finite(track.media.duration) || this.time - track.offset < track.media.duration - .005));
    }

    readClock() {
      const master = this.tracks.find(track => !track.failed && track.started &&
        !track.media.seeking && this.time >= track.offset);
      if (master) this.time = clamp(master.media.currentTime + master.offset, 0, this.duration);
      return master;
    }

    positionTracks() {
      for (const track of this.tracks) {
        const {media, offset} = track;
        const target = clamp(this.time - offset, 0, finite(media.duration) ? media.duration : this.duration);
        // Reassigning currentTime even to its existing value starts a browser
        // seek. In particular, do not seek again after play() has resolved.
        if (!track.failed && Math.abs(media.currentTime - target) > .04) {
          try { media.currentTime = target; } catch (_) {}
        }
      }
    }

    sync() {
      const active = this.activeTracks();
      if (active.some(track => track.media.readyState < 3 || track.media.seeking)) {
        this.holdForBuffering(); return;
      }
      for (const track of this.tracks) {
        const {media, offset} = track;
        if (!active.includes(track)) {
          media.pause(); track.started = false;
        } else if (!track.started || media.paused) {
          if (track.optional) this.startOptionalTrack(track);
          else { this.startPlayback(); return; }
        } else if (Math.abs(media.currentTime + offset - this.time) > .3) {
          // Correct only real drift against the playing master, never against a
          // wall clock that continues to advance while the network is stalled.
          try { media.currentTime = Math.max(0, this.time - offset); } catch (_) {}
        }
        if (track.fill) track.fill.style.width = `${clamp(this.time / Math.max(this.duration, .01), 0, 1) * 100}%`;
      }
    }

    play() {
      if (this.disposed || !this.duration) return;
      // Keep listening focused: playing a new sample pauses the other cards.
      for (const player of players) if (player !== this) player.pause();
      if (this.time >= this.duration) this.time = 0;
      this.ensurePose();
      this.playing = true;
      this.positionTracks();
      this.startPlayback();
      cancelAnimationFrame(this.animation);
      const tick = () => {
        if (!this.playing || this.disposed) return;
        if (this.buffering) this.resumeWhenReady();
        else if (!this.starting) {
          const master = this.readClock();
          // Only a genuinely silent gap (e.g. a positive track offset) uses
          // wall time. Any media clock takes precedence as soon as it starts.
          if (!master && !this.activeTracks().length)
            this.time = clamp(this.gapTime + (performance.now() - this.gapAnchor) / 1000, 0, this.duration);
          else { this.gapTime = this.time; this.gapAnchor = performance.now(); }
          this.sync(); this.draw();
          if (this.time >= this.duration) { this.pause(); return; }
        }
        this.animation = requestAnimationFrame(tick);
      };
      this.animation = requestAnimationFrame(tick);
    }

    async startPlayback() {
      if (!this.playing || this.disposed) return;
      const operation = ++this.operation;
      if (this.tracks.length && this.tracks.every(track => track.failed)) {
        this.pause(); text(this.status, 'Media could not load. Select this rollout again to retry.'); return;
      }
      this.buffering = false;
      this.starting = true;
      // Pause remains available even if a native play promise is waiting for
      // network data. A stale promise may never change a newer user action.
      this.playButton.disabled = false;
      text(this.playButton, 'Pause');
      text(this.status, 'Loading media… Playback will start automatically.');
      const active = this.activeTracks();
      try {
        await Promise.all(active.filter(track => !track.optional).map(track => {
          track.media.preload = 'auto';
          track.started = true;
          return track.media.play();
        }));
      } catch (_) {
        if (operation !== this.operation || this.disposed || !this.playing) return;
        this.pause();
        text(this.status, 'Playback could not start. Press Play to retry.');
        return;
      }
      if (operation !== this.operation || this.disposed || !this.playing) return;
      // One stream may have started while another was still loading. Rewind
      // only that genuine startup drift; ordinary pause/resume performs no seek.
      if (active.some(track => track.media.currentTime + track.offset - this.time > .25))
        this.positionTracks();
      this.starting = false;
      for (const track of active) if (track.optional) this.startOptionalTrack(track);
      this.gapTime = this.time; this.gapAnchor = performance.now();
      text(this.status, 'Synchronized playback · toggle audio lanes to compare.');
    }

    holdForBuffering() {
      if (!this.playing || this.starting || this.buffering) return;
      this.readClock();
      ++this.operation;
      this.buffering = true;
      for (const track of this.tracks) { track.media.pause(); track.started = false; }
      text(this.status, 'Buffering… Playback will resume automatically.');
      this.draw();
      queueMicrotask(() => this.resumeWhenReady());
    }

    resumeWhenReady() {
      if (!this.playing || !this.buffering || this.starting || this.disposed) return;
      const active = this.activeTracks();
      if (active.every(track => track.media.readyState >= 3 && !track.media.seeking)) {
        this.positionTracks();
        this.startPlayback();
      }
    }

    pause() {
      if (this.playing && !this.starting && !this.buffering) this.readClock();
      ++this.operation;
      this.playing = false;
      this.starting = false;
      this.buffering = false;
      cancelAnimationFrame(this.animation);
      if (this.playButton) { text(this.playButton, 'Play'); this.playButton.disabled = !this.duration; }
      for (const track of this.tracks) { track.media.pause(); track.started = false; }
      if (this.canvas) this.draw();
    }

    setTime(time) {
      ++this.operation;
      this.time = clamp(time, 0, this.duration);
      this.starting = false; this.buffering = false;
      for (const track of this.tracks) { track.media.pause(); track.started = false; }
      this.positionTracks();
      if (this.playing) this.startPlayback();
      this.draw();
    }

    draw() {
      if (!this.canvas) return;
      this.seek.value = String(this.time);
      text(this.clock, `${format(this.time)} / ${format(this.duration)}`);
      text(this.poseLabel, this.demo ? 'Procedural pose · demo' :
        this.time < (this.data.prefill_seconds || 0) ? 'PREFILL · real pose · unscored' : 'GENERATED');
      const videoNormalization = this.videoPoseNormalization();
      const overlay = Boolean(this.videoVisible && poseOverlayEnabled && videoNormalization);
      this.overlayButton.hidden = !this.video;
      this.overlayButton.disabled = !this.videoVisible || !videoNormalization || !this.pose;
      this.overlayButton.setAttribute('aria-pressed', String(poseOverlayEnabled));
      this.overlayButton.title = !videoNormalization ? 'Pose overlay unavailable: video actor framing is not published' :
        !this.videoVisible || !this.pose ? 'Waiting for video and pose' : 'Superimpose the pose used to condition this video';
      this.canvas.hidden = Boolean(this.videoVisible && !overlay);
      this.canvas.classList.toggle('pose-over-video', overlay);
      this.poseLegend.hidden = !this.pose || Boolean(this.videoVisible && !overlay);
      if (this.videoVisible && !overlay) { this.poseState.hidden = true; return; }
      const width = this.canvas.clientWidth || 500, height = this.canvas.clientHeight || 240;
      const dpr = Math.min(2, devicePixelRatio || 1);
      if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
        this.canvas.width = Math.round(width * dpr); this.canvas.height = Math.round(height * dpr);
      }
      const ctx = this.canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
      if (!this.pose) return;
      // Follow the displayed video during buffering/drift correction. Both
      // assets retain their explicit presentation offsets (163 ms in the
      // symmetric run); never add the delay twice or draw across a pending seek.
      if (overlay && (this.video.seeking || this.time < this.videoTrack.offset)) return;
      const poseTime = overlay ? this.video.currentTime + this.videoTrack.offset : this.time;
      const local = poseTime - (this.pose.start_time_seconds || 0);
      if (local < 0) return;
      const index = this.poseFrameIndex(poseTime);
      let frame, confidence;
      if (this.poseChunks) {
        const number = Math.floor(index / this.pose.chunk_frames), chunk = this.poseChunks.get(number);
        if (!chunk) {
          text(this.poseState, `Loading pose at ${format(this.time)}…`); this.poseState.hidden = false;
          this.requestPoseChunk(index); return;
        }
        const within = index - chunk.first_frame;
        frame = chunk.frames[within]; confidence = chunk.confidence?.[within];
        this.poseState.hidden = true;
        if (this.playing) this.requestPoseChunk((number + 1) * this.pose.chunk_frames);
      } else {
        frame = this.pose.frames[index]; confidence = this.pose.confidence?.[index];
      }
      const normalization = overlay ? videoNormalization : this.pose.source_normalization;
      // The native mono video is 270x480, contained in an 8:15 display frame.
      // Restore the source camera coordinates; never fit a person to the
      // extrema of an entire generated trajectory.
      const aspect = overlay ? videoNormalization.width / videoNormalization.height : 9 / 16;
      const contentWidth = normalization ? Math.min(width, height * aspect) : width;
      const contentHeight = normalization ? contentWidth / aspect : height;
      const ox = (width - contentWidth) / 2, oy = (height - contentHeight) / 2;
      const [xmin, xmax, ymin, ymax] = this.bounds;
      const scale = Math.min((width - 60) / Math.max(.2, xmax - xmin), (height - 50) / Math.max(.2, ymax - ymin));
      const coords = frame.map(p => {
        if (normalization) return [ox + (p[0] * normalization.shoulder_width + normalization.root_yx[1]) * contentWidth,
          oy + (-p[1] * normalization.shoulder_width + normalization.root_yx[0]) * contentHeight];
        const [x, y] = this.project(p);
        return [width / 2 + (x - (xmin + xmax) / 2) * scale, height - 20 - (y - ymin) * scale];
      });
      const style = getComputedStyle(document.documentElement);
      const neutral = style.getPropertyValue('--muted').trim();
      const edges = coords.length === 55 ? monoEdges : this.pose.parents.flatMap((parent, j) => parent < 0 ? [] : [[parent,j]]);
      const alpha = c => c == null ? 1 : clamp((c - .05) / .15, 0, 1);
      const line = Math.max(1.5, contentWidth / 160);
      ctx.save(); ctx.beginPath(); ctx.rect(ox, oy, contentWidth, contentHeight); ctx.clip();
      if (normalization && !overlay) { ctx.fillStyle = '#181818'; ctx.fillRect(ox, oy, contentWidth, contentHeight); }
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      for (const [a,b] of edges) {
        const c = confidence ? Math.min(confidence[a], confidence[b]) : null;
        if (c != null && c <= .05) continue;
        ctx.globalAlpha = .85 * alpha(c);
        ctx.beginPath(); ctx.moveTo(...coords[a]); ctx.lineTo(...coords[b]);
        ctx.strokeStyle = 'rgba(0,0,0,.65)'; ctx.lineWidth = line + 1.5; ctx.stroke();
        ctx.strokeStyle = c == null ? neutral : poseColor(c); ctx.lineWidth = line; ctx.stroke();
      }
      for (let j = 0; j < coords.length; j++) {
        const c = confidence?.[j];
        if (c != null && c <= .05) continue;
        const [x, y] = coords[j];
        ctx.globalAlpha = .9 * alpha(c);
        ctx.fillStyle = c == null ? neutral : poseColor(c);
        ctx.beginPath(); ctx.arc(x, y, Math.max(1.4, contentWidth / 145), 0, Math.PI * 2); ctx.fill();
      }
      ctx.restore();
      this.canvas.dataset.frame = String(index); this.canvas.dataset.ready = 'true';
    }

    dispose() {
      this.pause(); this.disposed = true;
      this.poseObserver?.disconnect();
      this.poseRequest?.abort();
      clearTimeout(this.poseRetry);
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
      for (let i = 0; i < rollouts.length; i++) select.append(new Option(`Step ${rollouts[i].update}`, String(i)));
      const previousIndex=rollouts.findIndex(rollout=>rollout.id===previousId);
      if(previousIndex>=0)select.value=String(previousIndex);
      const render = () => {
        const rollout = rollouts[Number(select.value)] || {};
        const sampleSelect = $('#sample');
        const selectedSample = clamp(Number(sampleSelect?.value || 0), 0, 3);
        const selected = rollout.samples?.[selectedSample];
        const currentSignature=JSON.stringify([manifest.synthetic,rollout.id,selectedSample,selected]);
        select.dataset.rolloutId=rollout.id||'';
        if(currentSignature===renderedSignature)return;
        const withoutVideo = sample => {
          const copy = {...sample};
          for (const key of ['video_url','video_offset_seconds','video_metadata','video_pose_normalization']) delete copy[key];
          return JSON.stringify(copy);
        };
        if (players.length === 1 && selected && players[0].rolloutId === rollout.id &&
            withoutVideo(players[0].data) === withoutVideo(selected)) {
          players[0].updateVideo(selected); renderedSignature=currentSignature; return;
        }
        for (const player of players) player.dispose();
        players = [];
        $('#samples').replaceChildren();
        const samples = Array.isArray(rollout.samples) ? rollout.samples.slice(0, 4) : [];
        const demo = manifest.synthetic === true;
        if (rollouts.length) text($('#notice'), demo ? 'SYNTHETIC PREVIEW · Tone waveforms and procedural stick figures. No dataset recordings or model outputs are loaded.' : `${samples.length} of 4 samples published · ${rollout.id || 'rollout'}`);
        if (sampleSelect) {
          sampleSelect.disabled = !rollouts.length;
          sampleSelect.onchange = render;
        }
        const sample = new Sample(samples[selectedSample], selectedSample + 1, demo);
        sample.rolloutId = rollout.id;
        players.push(sample); $('#samples').append(sample.root); sample.draw();
        renderedSignature=currentSignature;
      };
      select.onchange = render; render();
      manifestSignature=signature;
    } catch (error) {
      if(manifestSignature!==null){text($('#notice'),'Sample refresh temporarily unavailable; retaining published media.');return;}
      text($('#notice'), 'Sample manifest is not available yet. No media has been loaded.');
      $('#rollout').disabled = true;
      $('#samples').replaceChildren();
      $('#samples').append(new Sample(null, 1, false).root);
    } finally {
      refreshing=false;
    }
  }
  window.addEventListener('resize', () => players.forEach(player => player.draw()));
  document.addEventListener('visibilitychange', () => { if (document.hidden) players.forEach(player => player.pause()); });
  window.addEventListener('pagehide', () => players.forEach(player => player.dispose()));
  init();
  setInterval(init,5000);
})();
