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

The separate inference worker runs one job at a time on its assigned GPU pair.
It must use the same queue root and shared schema. For this deployment the app
and worker live on landsend-1; landsend-0's existing comparison stays untouched.
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
