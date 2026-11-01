# livedup: Duplex playground = the generation playground, but LIVE from the mic (from main amartyavarma-26, 10-26 06:25 PT)

The user, verbatim:
> "can you make the duplex palyground excactly like the generatoin playground except with the duplex model and whe i press start intsead of generate it starts streaming my audio"

## Goal
The Duplex view of the playground (${DEMO_URL}/?model=duplex) must look and behave EXACTLY like the monoplex generation playground. The one difference: the big button is **Start / Stop** instead of Generate.
- Start opens the mic and streams the user's audio live to the duplex model (Aloe pose-duplex export-498, the live-demo model). It's a real-time call:
  - the model's speech plays back;
  - the avatar renders live on the seq engine (Param student EMA, as on the live demo).
- Stop ends the call.

Keep everything from the monoplex playground, wired to the live call:
- **Tabs and layout.** Output tab (live avatar + reel at the top), Generation tab (every setting incl. Prefill), Batches, History. Same layout, dark mode default, version badge + CHANGELOG.
- **Generation tab.** Every engine knob the live duplex engine accepts: DUPLEX_SCHEMA / the live server's knobs, incl. pose CFG + "only while speaking", adaptive rep-pen, the cb0/light split and prefill person/seconds/source.
  - Plus Copy settings / Save file / Load settings (the new feature).
  - Settings are applied per call at Start.
- **History.** Every call is recorded: mic audio (A), model audio (B), words, pose and the avatar mp4. Each entry has settings + copy-settings, replay with speed/volume/skip-prefill, pose stats (cb0 repeat, gestures/min, hands moving %), the finish chime when the post-call render is done, and no autoplay.
- **Video weights.** Param (live, default). Vansh / Both cannot stream in real time, so for a finished call they appear as a post-call re-render of the recorded pose (render-only, 20 steps), side by side for Both. GT toggle default OFF.
- **The old offline "generate both sides" duplex.** Keep it reachable as a secondary mode in the Generation tab ("Mode: Live mic (default) | Generate both sides"), unchanged.

## Backend
- Reuse the live demo's exact streaming engine and protocol: ppm/aloe-posedup-live server.py / proxy.py / index.html call path; mic → server → model → audio + videort feed.
- Same maths and the same defaults as the live preset: Default Person 01, 2 s conversation-start prefill, pose light top-k 151, exact light decoder, CFG off, NO pose post-processing.
- Capacity:
  - Use the live LB's backends only if they can take a per-call knob set without changing the live demo's preset. Otherwise give the playground its own backend (borrow a GPU pair from aloedemo's pool).
  - NEVER preempt or take over a user session on the live demo (memory demo-no-test-preempt). Test clients must be non-preempting.

## Rules
- **Maths.** USER HARD RULE: never change the maths. UI and plumbing only; same engine forward and sampling as the live demo.
- **Compute.** Always --mem. NEVER slinky-9. NEVER the 16 MoT hosts. NEVER slinky-4 GPUs 6-7. No login-node compute. No `cd … &&`. No GCS.
- **Restarts.** Idle units only; no downtime on either public link. Every deploy appends a CHANGELOG line.
- **Speed.** Max-parallelise with forks:
  - (a) page: copy the monoplex shell, swap in Start/Stop + live output;
  - (b) streaming backend wiring + per-call knobs;
  - (c) call recording → History + post-call Vansh/Both render;
  - (d) checks.
  First usable version (Start streams the mic, the model talks back, the avatar shows, all settings apply) within ~30 min, then the rest.

## Owners
- **dupui (amartyavarma-1b)** leads: page + wiring.
- **aloedemo (amartyavarma-2c)** owns genweb + the live demo backends: agree file boundaries first, provide the backend/GPU, roll idle-only.
- **videort (amartyavarma-75)** for the live avatar feed if needed.

## Verify
On the public link, with a synthetic non-preempting mic client (a WAV fed as mic):
- the model replies with speech;
- the avatar streams;
- a changed knob (e.g. pose CFG on + only-while-speaking) shows in the call summary;
- the History entry has A/B audio + mp4 + stats;
- Save/Load settings round-trips;
- no sideways scroll at 1280 / 1440 / 1920 / 390.

## Reporting
- Log progress in genweb/LIVEDUP_STATUS.md.
- dupui messages main (name on line 3 of ppm/control/OPS_PLAYBOOK.md, now amartyavarma-26):
  - with an ETA now;
  - then ONCE at first-usable (with link);
  - then ONCE when everything is done;
  - or on a blocker only the user can resolve.
- No interim chatter.
