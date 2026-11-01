"""STAGE-4 audio plumbing between the data (DataLoader workers) and the frozen Voxtral front end (trainer, GPU, outside FSDP).

  WavProvider(cfg).load(ref) -> float32 [n x 1920] 24 kHz waveform of channel A's conversation frames 0..n-1 (ref = data_s4q.audio_ref,
      n = the conversation frames kept after the prefix crop). The audio is resolved by row_key in the AUDIO MANIFEST (S4_AUDIO_MANIFEST;
      the test rows: stage4/testdata/manifest.json) and loaded by voxtral_front.load_row_audio with the RECORDED per-(episode, speaker)
      gain lufs.lufs_gain_db (the -23 LUFS / peak-limited gain the Mimi codes were made with; a chunk alone cannot recompute it).
      A row without a manifest entry is an ERROR (no fallback). The production stage-4 dataset must carry the same fields per row / side.
  make_front(cfg, dev) -> .encode(wavs: [n_r x 1920] per row, refs) -> latents [1, 4 sum n_r, 1280] bf16 (rows concatenated in time; each
      row encoded on its own, so no audio crosses rows):
        flac : voxtral_front.VoxtralFront (frozen Voxtral Realtime encoder, causal, bf16, not an nn.Module)
        stub : deterministic N(0, 1) latents per row_key - TESTS / SMOKES ONLY (make_cfg refuses it otherwise)."""
import hashlib
import json
import os

import numpy as np
import torch

SAMPLES_PER_FRAME_24K = 1920
ENC_PER_FRAME = 4
# 10-09 (model 2, stage2-handover rows; additive: only manifest entries with decode == FFMPEG_DECODE take this path). The entry points at the
# speaker's whole-conversation SOURCE flac (48 kHz stereo L == R, or mono 48 / 44.1 kHz); it is decoded EXACTLY as the Mimi encoder's
# preprocessing_spec ("ffmpeg:-f f32le -ac 1 -ar 24000"; checked: pyloudnorm of this decode == the recorded lufs_in to 5e-4 LU), the row is
# samples [1920 frame0, 1920 (frame0 + n_frames)) (zero-padded past the end, as the encoder padded), times the RECORDED gain.
FFMPEG_DECODE = "ffmpeg_f32le_ac1_ar24000"
FFMPEG_EXE = os.environ.get("S4_FFMPEG", "/data/home/amartyavarma/venvs/qwen-s2/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")


def ffmpeg_decode_24k(path):
    import subprocess
    r = subprocess.run([FFMPEG_EXE, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "24000", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32)


def ffmpeg_row_audio(e, gain_db):
    x = ffmpeg_decode_24k(e["flac"])
    s0, n = int(e["frame0"]) * SAMPLES_PER_FRAME_24K, int(e["n_frames"]) * SAMPLES_PER_FRAME_24K
    # the encoder zero-padded a lane shorter than the clip (apad; e.g. f43d6d21 SPEAKER_01 is 0.096 s shorter than SPEAKER_00, both coded
    # to the same T): pad the same way; a shortfall above 1 s is a data error
    assert len(x) > s0 + n - 6000, ("source flac shorter than the row by > 0.25 s", e["row_key"], len(x), s0, n)
    if len(x) < s0 + n - SAMPLES_PER_FRAME_24K:
        print(f"[audio] WARNING {e['row_key']}: source flac {(s0 + n - len(x)) / 24000:.3f} s shorter than the row; zero-padded (as the Mimi encoder did)", flush=True)
    w = np.zeros(n, np.float32)
    seg = x[s0:s0 + n]
    w[:len(seg)] = seg
    return (w * np.float32(10.0 ** (float(gain_db) / 20.0))).astype(np.float32)


def _vf():
    import voxtral_front as VF        # stage4/voxtral_front.py (the front-end port)
    return VF


class WavProvider:
    def __init__(self, cfg):
        self.mode, self.root, self.man_path = cfg["s4_audio"], cfg["s4_audio_root"], cfg.get("s4_audio_manifest")
        self.by_key = None

    def entry(self, row_key, side=None):
        """S4_AUDIO_MANIFEST = one or more manifests (comma-separated); an entry's relative flac path is joined with ITS manifest's directory.
        Entries WITH a 'side' field (validation manifests, s4-audio-manifest-v2) are keyed by (row_key, side); entries without one are the
        row's USER side (the older manifests) and only serve side None / 'user'."""
        if self.by_key is None:
            self.by_key = {}
            for mp in [x for x in str(self.man_path).replace("+", ",").split(",") if x]:     # ',' or '+' (sbatch --export splits on ',')
                m = json.load(open(mp))
                for e in (m["rows"] if isinstance(m, dict) else m):
                    k = f"{e['row_key']}|{e.get('side', 'user')}"
                    assert k not in self.by_key, ("row_key / side in two audio manifests", k)
                    self.by_key[k] = dict(e, _root=os.path.dirname(os.path.abspath(mp)))
        k = f"{row_key}|{side or 'user'}"
        assert k in self.by_key, ("no raw audio for this row / side in the audio manifest (no fallback)", k, self.man_path)
        return self.by_key[k]

    def load(self, ref):
        if self.mode != "flac":
            return None
        if ref.get("kind") == "mls_stream":                                   # stage4-asr2: ONE streamed shard arrow (stream_mls.py); same record
            import sys                                                       # format / decode as 'mls'; small LRU so deleted shards get unmapped
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter_data"))
            import mlsclips
            import pyarrow as pa
            import pyarrow.ipc as ipc
            lru = self.__dict__.setdefault("_stream_lru", {})
            p_ = ref["arrow"]
            if p_ not in lru:
                while len(lru) >= 3:
                    lru.pop(next(iter(lru)))
                lru[p_] = ipc.open_file(pa.memory_map(p_, "r")).read_all()
            else:
                lru[p_] = lru.pop(p_)
            rec = lru[p_].slice(int(ref["i"]), 1).to_pylist()[0]
            assert rec["clip_id"] == ref["row_key"], ("stream shard index -> another clip", rec["clip_id"], ref["row_key"])
            w = mlsclips.load_wav24(rec)
            assert w.shape[0] == int(ref["n"]) * SAMPLES_PER_FRAME_24K, (ref["row_key"], w.shape, ref["n"])
            return w
        if ref.get("kind") == "mls":                                          # an MLS clip shard: agent 5's mlsclips.load_wav24 (gain included)
            if not hasattr(self, "_shards"):
                self._shards = {}
            if ref["shard"] not in self._shards:
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter_data"))
                import mlsclips
                self._shards[ref["shard"]] = mlsclips
                self._recs = getattr(self, "_recs", {})
                self._recs[ref["shard"]] = mlsclips.Shard(ref["shard"])
            rec = self._recs[ref["shard"]][int(ref["i"])]
            assert rec["clip_id"] == ref["row_key"], ("MLS shard index -> another clip", rec["clip_id"], ref["row_key"])
            w = self._shards[ref["shard"]].load_wav24(rec)
            assert w.shape[0] == int(ref["n"]) * SAMPLES_PER_FRAME_24K, (ref["row_key"], w.shape, ref["n"])
            return w
        e = self.entry(ref["row_key"], ref.get("side"))
        assert e.get("dataset_folder", ref.get("dataset_folder")) == ref.get("dataset_folder", e.get("dataset_folder")), ("manifest row differs", ref["row_key"])
        if e.get("speaker_dir") and ref.get("speaker_dir"):
            assert e["speaker_dir"] == ref["speaker_dir"], ("manifest speaker differs from the row's side", ref["row_key"], ref.get("side"), e["speaker_dir"], ref["speaker_dir"])
        gain = float(e["lufs"]["lufs_gain_db"])
        if e.get("decode") == FFMPEG_DECODE:                                  # model-2 handover rows: the SOURCE flac, decoded as the Mimi encoder did
            w = ffmpeg_row_audio(e, gain)
        else:
            w = np.asarray(_vf().load_row_audio(e, e["_root"], gain_db=gain), np.float32).reshape(-1)
        f0 = int(ref.get("f0", 0)) * SAMPLES_PER_FRAME_24K                  # a clip cut from a row: its first frame in the row
        need = int(ref["n"]) * SAMPLES_PER_FRAME_24K
        assert w.shape[0] >= f0 + need, ("A audio shorter than the conversation / clip", ref["row_key"], w.shape[0], f0, need)
        return w[f0:f0 + need].copy()


class WithAudio:
    """wraps a micro dataset: item['wav'] = [float32 tensor per row] (S4_AUDIO=flac) or absent."""

    def __init__(self, ds, cfg):
        self.ds, self.pv = ds, WavProvider(cfg)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        b = self.ds[i]
        if self.pv.mode == "flac":
            for d in ([b["s"]] if "s" in b else [b]):                          # a KL pair: only the student hears audio
                d["wav"] = [torch.from_numpy(self.pv.load(r)) for r in json.loads(d["audio"])]
        return b


class StubFront:
    """random latents, deterministic per row_key: N(0, 1), bf16. Tests / smokes only."""

    def __init__(self, dev, hidden=1280):
        self.dev, self.hidden = dev, hidden

    @torch.no_grad()
    def encode(self, wavs, refs):
        out = []
        for r in refs:
            seed = int.from_bytes(hashlib.blake2b(r["row_key"].encode(), digest_size=8).digest(), "little") % (2 ** 63)
            g = torch.Generator(device="cpu").manual_seed(seed)
            out.append(torch.randn(ENC_PER_FRAME * int(r["n"]), self.hidden, generator=g))
        return torch.cat(out, 0)[None].to(self.dev, torch.bfloat16)


class VoxFront:
    def __init__(self, cfg, dev):
        self.fe = _vf().VoxtralFront(path=cfg["s4_vox_model"], device=dev)
        self.dev = dev

    @torch.no_grad()
    def encode(self, wavs, refs):
        assert wavs is not None and len(wavs) == len(refs), "S4_AUDIO=flac: every row needs its waveform"
        out = []
        for w, r in zip(wavs, refs):
            n = int(r["n"])
            assert w.shape[-1] == n * SAMPLES_PER_FRAME_24K, (r["row_key"], tuple(w.shape), n)
            lat = self.fe.encode(w.to(self.dev, torch.float32), n)
            assert lat.shape == (1, ENC_PER_FRAME * n, 1280), (tuple(lat.shape), n)
            out.append(lat[0])
        return torch.cat(out, 0)[None].to(torch.bfloat16)


def make_front(cfg, dev):
    return StubFront(dev, int(cfg["s4_enc_hidden"])) if cfg["s4_audio"] == "stub" else VoxFront(cfg, dev)
