# Speech and pose comparison / request studio

This CPU-only gallery publishes completed inference files without changing the
interactive demo, its queue, or its existing gallery. The publisher accepts the
three PPM variants `aligned-media`, `compact-all`, and `xabi-original`.

```bash
ROOT=/home/param/pose-runs/controlled-comparison-20261028/gallery
python3 comparison/publish.py --data-root "$ROOT" init --clips clips.json
python3 comparison/server.py --data-root "$ROOT" --port 7880
python3 comparison/publish.py --data-root "$ROOT" publish \
  --clip-id clip-01 --model aligned-media --video /path/to/completed/video.mp4 \
  --metadata /path/to/public-run-details.json
```

`clips.json` is a list (or an object with a `clips` list). Each entry has `id`,
`title`, `transcript`, and optional `recording_id` / `provenance`. Use public
sample metadata only. Initialization preserves existing published entries.
Publish only after the inference status is `complete`. The publisher checks
audio/video streams, copies and hashes the completed MP4, extracts a poster,
then atomically updates the manifest under a file lock. Concurrent publishers
cannot overwrite each other's manifest updates. Content-addressed media avoids
stale browser caches. The page polls every ten seconds without resetting videos
already playing.

Each completed gallery video shows its recorded sampling settings beside the
player: text mode, audio/pose temperature and top-k, sampled-text settings,
residual-codebook settings, and available seeds. Top-k zero is labeled
unfiltered; forced text is labeled not sampled. These labels come from that
video's public metadata, not from the request form defaults. Missing settings
are shown as not recorded. Original, overlay, and pose-only views share these
settings because they are views of the same generation.

`publish.py ... status --clip-id ID --model MODEL --status generating` updates
live progress; use `pending` or `failed --message ...` when appropriate.

The server binds to localhost by default. Its read-only routes expose only the
page, manifest, and the gallery's media files, with byte-range playback support.
Expose port 7880 through the existing Cloudflare quick-tunnel workflow when
publishing. The tunnel URL changes if the tunnel process restarts.

The server creates a mode-0600 access token at `DATA_ROOT/.access-token`.
Transcripts, manifests, and videos require its session cookie. Share the URL
with `#token=TOKEN` as the fragment; the browser exchanges it for a Secure,
HttpOnly cookie and removes the fragment. Fragments are not sent to the tunnel
or server access logs. Keep tokens out of checked-in files and command output.

## Persistent request studio

`request_server.py` adds a request form, FIFO queue display, and completed video
players while serving the same comparison at `/comparison`. The allowed clips
and models come from the controlled comparison; custom text and arbitrary paths
are not accepted. Controlled mode forces the prepared framewise text. Native mode is available
for compact transcript and original Xabi: the compact model receives its plain
upfront transcript and samples framewise text, while original Xabi receives no
transcript and generates text freely. Aligned transcript remains controlled only.
Every mode keeps the same eight-second voice, initial pose geometry, and visual
references. Native words and timing can differ from the source clip. Defaults
are text/audio/pose temperature 1 and top-k 0
(unfiltered), with separate controls for each modality.

```bash
python3 comparison/request_server.py \
  --gallery-root /home/param/pose-runs/controlled-comparison-20261028/gallery \
  --queue-root /home/param/pose-runs/inference-requests-20261028 \
  --schema-dir /home/param/pose-repo/infer/transcript_pose --port 7880
```

The server imports the inference repository's `request_schema.py`. Authenticated
`POST /api/requests` also requires the same Origin and an `X-CSRF-Token` from
`GET /api/config`. Validation rejects unknown fields, duplicate JSON keys,
nonfinite values, out-of-range numbers, and booleans in numeric fields. Requests
are written and fsynced in a staging directory, then atomically renamed to
`jobs/<32-lowercase-hex-UUID>`. `GET /api/requests` reads actual worker states;
media is available only after that worker marks the request complete.

The central dispatcher on landsend-1 assigns independent requests across the
32 GPUs on landsend-1 through landsend-4. Audio/pose uses two GPUs;
video retains one and releases the other for another queued request. Completed
media is copied back before the central request becomes playable.
The app and dispatcher share the queue root and request schema.
Landsend-0's existing comparison stays untouched.
The old authenticated gallery manifest can set `request_studio_url` to link to
the new studio. Store token-bearing URLs only in that protected manifest and
mode-0600 operator files, never public HTML or source control.

## Native comparison

The request server also serves `/native-comparison` from
`/home/param/pose-runs/native-comparison-20261028/gallery` (override with
`--native-gallery-root`). It uses the same session and tunnel as the controlled
comparison. `/native-manifest.json` rewrites media URLs to the separate,
authenticated `/native-media/` namespace. Initialize with `publish.py init
--mode native` to select compact and original Xabi columns; controlled remains
the default. Each completed cell displays generated text and, for the compact
model, its actual supplied upfront transcript.

The inference repository's `publish_native.py` validates the native sampling
and video receipt, publishes locally, then copies only the completed cell to
landsend-1 and merges its manifest there under the publisher lock. This avoids
overwriting another finished cell while multiple videos arrive. The existing
24 controlled videos remain in their original gallery.

`GET /api/capacity` reads the dispatcher's atomic `dispatcher-status.json` from
the queue root and exposes only capacity counts, node/pair assignments, stages,
and the heartbeat timestamp. Missing or corrupt status is shown as unavailable;
a heartbeat older than sixty seconds is marked stale. Request cards show their
assigned node and active GPUs, including staging and output-transfer phases.
This endpoint is read-only and requires the same authenticated session.

## Audio-guided native comparison

`/guided-comparison` is a separate 24-video gallery for all three checkpoints on
the same eight 60-second clips. Its root is
`/home/param/pose-runs/audio-guided-native-comparison-20261028/gallery`, configurable
with `--guided-gallery-root`. Initialize it with `publish.py init --mode guided-native`.
The protected `/guided-manifest.json` and `/guided-media/` routes use the existing
session; the controlled and previous native comparisons keep their own files.

Aligned receives the full aligned prefix and forced framewise text. Compact
receives its full plain transcript and samples framewise text. Original Xabi
receives only the common eight-second voice reference and samples its text.
All temperatures are 1; text is unfiltered, audio top-k is 30 for every codebook,
pose cb0 top-k is 30, and pose cb1–31 top-k is 50. Every pose codebook uses
`audio + 3 * (audio - silence)` logits, with an independent full-model silence
pass and shared text/pose history. The gallery shows these settings per video.

`infer/transcript_pose/publish_guided.py` validates actual inference metadata,
silence-code provenance, video receipts and hashes before atomically publishing
each completed cell locally and on landsend-1. It preserves Original, Pose
overlay, and Pose only views and records each model's supplied/generated text.

The follow-up `/guided-prefill-comparison` uses the separate
`audio-guided-prefill2s-comparison-20261028/gallery` root. The first 25 codec
frames (2 seconds) are exact source audio, pose and framewise text. Its
conditioning downloads include these arrays and hashes. The continuation uses
the same native formats and guidance settings; `/guided-comparison` preserves
all earlier results without prefill.

At 2.000 seconds, all three presentation views show two white frames (80 ms)
and share a quiet 880 Hz beep lasting 100 ms with 8 ms fades. The marker runs
after diffusion and pose compositing. Clean views remain in `unmarked/`; raw PPM
audio, codes and model-input latents remain untouched and their hashes are
checked. `prefill-marker.json` records the marker and identical AAC/PCM hashes
across the three views. New studio requests default to 25 prefill frames;
previously stored requests without that field retain their original behavior.

## Pose views

Every completed comparison and request can expose three synchronized views:
**Original**, **Pose overlay**, and **Pose only**. The player defaults to the
overlay when available and preserves the playback position, play/pause state,
volume, mute and speed when switching. Polling preserves a manually chosen view.
The overlay is precomposed, so it also works with native video fullscreen and
the **Open view** link.

`pose-player.js` and `pose-player.css` are shared by the controlled gallery,
native gallery and request studio. Gallery entries use `pose_video` and
`pose_overlay_video` URLs alongside their SHA-256 values. Publish both together:

```bash
python3 comparison/publish.py --data-root "$ROOT" publish \
  --clip-id clip-01 --model aligned-media --video /path/to/video.mp4 \
  --pose-video /path/to/pose.mp4 --pose-overlay-video /path/to/overlay.mp4
```

The inference repository's `infer/transcript_pose/pose_views.py` creates these
views on CPU, reusing the production raster or reconstructing missing older
previews from the pinned pose tokenizer. It keeps the original frame grid and
soundtrack, verifies compressed audio packets and decoded PCM, and publishes a
`pose-views.json` receipt last. Request URLs `/requests/<id>/pose.mp4` and
`/requests/<id>/overlay.mp4` require authentication, a completed request, and a
matching receipt and metadata. Neither preview changes the original video.
