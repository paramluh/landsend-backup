# Universal Wide: A Relative-Time Model of Video

## 1. Video as latent motion

Let a video be a finite sequence of latent visual states

```text
(x₀, x₁, …, x_(T−1)).
```

The first state x₀ has a distinguished role. It is a permanent clean
reference, called the **sink**, and remains at relative position 0. It is
not ordinary history: it is always present, never corrupted, and never moves.
All other states are arranged relative to the sampled training state or the
current inference wavefront. Their positions do not encode absolute clip time.

For fixed history width H and frontier width P, a training state is

```text
S = [s | h₁, …, h_H | p₀, …, p_(P−1)]
```

where s = x₀, the h_j are clean historical states, and the p_i form
the present frontier. The sink is excluded from the count H. A useful
relative arrangement is

```text
relative position | 0 |  1  2  3  4  5 |  6  7  8  9 10
role              | s | h₁ h₂ h₃ h₄ h₅ | p₀ p₁ p₂ p₃ p₄
```

for H = P = 5. Shifting the sampled temporal region through a clip changes the
states occupying these positions but not the geometry itself. This is the
translation symmetry of the construction.

## 2. A monotone uncertainty frontier

The sink and every valid history state have diffusion time t = 0. Thus they
are clean. Frontier slot i independently samples

```text
t_i ~ Uniform((i/P, (i+1)/P]),    i = 0, …, P−1.
```

The convention is t = 0 for clean data and t = 1 for pure noise. Hence
uncertainty increases from left to right across the frontier. For P = 5, the
five inference boundary phases are

```text
0.2,  0.4,  0.6,  0.8,  1.0.
```

Training samples within the corresponding right-closed intervals. Their upper
boundaries are exactly the distilled five-step phases, including t = 1. Phase,
relative position, and uncertainty are therefore aligned: as a state becomes
cleaner, it moves left.

For a clean latent state x and independent noise
ε ~ N(0, I), the interpolation is

```text
x_t = (1−t)x + tε
v   = d x_t / dt = ε−x
```

The learned conditional law predicts this velocity from the visible state and
its permitted information. Its objective applies only to valid frontier
states. If m_(b,i,e) indicates that element e of frontier slot i in
training window b is valid, then

```text
      Σ_(i=0)^(P−1) Σ_e m_(b,i,e) ‖v̂_(b,i,e) − v_(b,i,e)‖²
L_b = ─────────────────────────────────────────────────────────
                 Σ_(i=0)^(P−1) Σ_e m_(b,i,e)

L = (1/B) Σ_(b=1)^B L_b
```

Each window is normalized by its exact valid element count, then a fixed B
windows are averaged on each rank. A startup window therefore has the same
sample weight as a full window. This is an equal-window objective, not a
frame-uniform objective. It has P distinct loss strata, one for each frontier
phase interval.

## 3. The information law

The modeled conditional law is defined by a per-layer visual information
relation. Let Prev_H(q) denote the preceding H relative non-sink slots of a
present state q. These slots may contain clean history, earlier less uncertain
frontier states, or ordinary zero padding. The direct inputs to one layer are

```text
I(s)   = {s}
I(h_j) = {s, h_j}
I(p_i) = {s, p_i} ∪ Prev_H(p_i)
```

Thus the sink depends only on itself. Each history slot is represented from
the sink and itself, without any older temporal state. The sink is read at a
fixed relative offset of H — every history frame is rotated as if it sat in the
newest history slot — so a history representation does not depend on the slot
the frame currently occupies and is computed once per frame. Each present state
receives the sink, itself, and exactly the preceding H relative non-sink slots
at their true offsets.
Absent history or startup slots keep their positions and hold a learned pad
token: one vector standing in for every token of the frame, at t = 0, with
pose and audio masked. They are not replaced by older valid states.

H limits direct lookback in each layer, not end-to-end information flow.
Across layers, an earlier present state can pass information that it received
from still earlier present states, so present-state information may propagate
transitively beyond H direct edges. History states see only the sink and
themselves in every layer. Their representations therefore do not recursively
carry cross-round cached memory.

Pose is given for every valid non-sink state, whether historical or present.
Audio is given only for valid present states. Neither signal is given for the
sink or for absent states. These exogenous signals help define which evolution
is being requested; visual context alone does not define the task.

## 4. Dense non-overlapping training states

On each pass, independently draw a startup width

```text
s ~ Uniform{1, …, P}.
```

Frames 1, …, s form the clip's first window. They occupy the rightmost s
frontier slots, preserving chronological order. The other P−s slots are pad
states at t = 0 and contribute no loss (none when s = P: the first window is
then already full). For example, with P = 5 and s = 2,

```text
[p₀, p₁, p₂, p₃, p₄]
    =
[pad_(t=0), pad_(t=0), pad_(t=0), (x₁)_(t₃), (x₂)_(t₄)].
```

Starting at frame 1+s, frames are divided consecutively into full groups of
size P:

```text
(x_(1+s), …, x_(s+P)),
(x_(1+s+P), …, x_(s+2P)), …
```

For each group, history consists of the latest H preceding clean non-sink
states, with pad states wherever fewer than H exist. The final incomplete
tail is dropped. Present targets are non-overlapping within a pass. The random
s changes the slot assignment for most retained frames, but the final source
frame can only be retained in the final slot. There are no ending or draining
training states.

Each loader epoch is one seeded source pass. Epoch and clip identity determine
the clip's offset without mutable state. Every clip contributes its minimum
⌊(T−1)/P⌋ windows: if an offset would create one extra full window, the
last one is omitted. Each rank microbatch contains exactly B windows.
Complete B × DP × GA groups define the epoch geometry; only the final
incomplete global group is omitted. Epoch and microbatch position reconstruct
the same samples after resume.

**History dropout.** Independently per window, with probability π_d the d
oldest available history frames are removed (d = 1, …, H, Σ_d π_d ≤ 1); the
survivors stay right-aligned against the present, so the
window keeps the exact layout of a startup or short-clip window with fewer
frames. Dropping the oldest frames is the only admissible form: history at
inference is always a contiguous suffix of the past, and temporal RoPE indexes
slots, so a gap adjacent to the present would be invisible to the model and
never occur. The expected fraction of available history frames removed is
Σ_d π_d · d / H. Eval windows are never dropped.

Audio sources are local to each microbatch. Its distinct source clips occupy
rows of a fixed B-row code table, with unused rows padded and never selected.
A clip crossing a microbatch boundary is represented independently on both
sides. Every table has exactly one 750-code time quantum.

## 5. Inference as a causal wavefront

Consider the distilled case with P frontier states and P phase steps.
Inference begins with one pure-noise state in the rightmost phase. During each
round, every active state advances one phase toward clean, moves one position
left, and a new pure-noise state enters on the right. With P = 5, startup is

```text
  [ ·,     ·,     ·,     ·,     x_1.0]
→ [ ·,     ·,     ·,     x_0.8, x_1.0]
→ [ ·,     ·,     x_0.6, x_0.8, x_1.0]
→ [ ·,     x_0.4, x_0.6, x_0.8, x_1.0]
→ [ x_0.2, x_0.4, x_0.6, x_0.8, x_1.0].
```

On the next advance, the leftmost state reaches t = 0, is emitted as a clean
state, and becomes the newest member of history. After this warmup, one clean
state is emitted per round. The sink remains fixed at relative position 0;
history and frontier continue translating around that reference geometry.

Inference stops once the requested number of clean outputs has been emitted.
Any unfinished frontier is discarded. Since unfinished states are not desired
outputs, training has no corresponding draining state.

The intended anti-drift effect follows from the wavefront's uncertainty
geometry. A generated visual state can enter the causal context of a later
state while still carrying graded uncertainty, rather than appearing only as
unquestioned clean truth. Errors, accidental repetitions, and copied details
are therefore presented with uncertainty appropriate to their phase. This can
weaken their authority and increase reliance on the sink, pose, and audio. It
is a motivation and expected effect, not a theorem that drift is eliminated.

## 6. Two evaluation windows

Eval scores one full-frontier window per clip in two layouts, each present
slot at the top of its band, t = i/P, where the wavefront first evaluates it.
For P = 5 the present slots hold

```text
[x_0.2, x_0.4, x_0.6, x_0.8, x_1.0].
```

**window-0** is the first full window of a clip: sink, no history, frames
1, …, P in the present slots. **window-1** is the window immediately after
history has filled: sink, frames 1, …, H as history, frames H+1, …, H+P in
the present slots. Loss is reported per window and present slot (slot 1 =
lowest noise, slot P = pure noise). Each source frame reuses noise determined by its source and frame identity, so
the numbers are deterministic across evaluations.

## 7. Scope and limitations

H bounds each layer's direct lookback but does not impose a strict end-to-end
information horizon because present states communicate transitively across
layers. The sink is a stable reference, not a complete memory of the world.
Sink-plus-self history prevents hidden-state memory from accumulating recursively
across rounds, but generated visual states still carry their content forward.
Pose and audio remain essential to the requested evolution. Monotone
uncertainty and direct causal context can mitigate error amplification,
repetition, and drift, but cannot remove model error or guarantee long-horizon
consistency.