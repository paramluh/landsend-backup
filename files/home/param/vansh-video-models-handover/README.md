# Agent guidelines for this repo

The ideas explored here are conceptually deep and demand precision and rigor —
attention budget belongs on the math and the model, not on engineering
ceremony. Keep the code minimal and direct so the ideas stay legible. This is
not "experimental, so the code can be sloppy"; it is "the concepts are
complex, so the code must not add complexity of its own".

## Working with the user

- **Do exactly what was asked, at the granularity asked.** "Get run 1 going" is
not run 2; "give me the command" is the bare command, not a tmux wrapper; "guess
how video enters" is a guess in chat, not a built module. Scope is the user's to
widen.
- **Implementation is not execution permission.** No GPU or CPU jobs, smoke
tests, downloads, corpus sweeps, detached processes, or shared-state mutations
without explicit approval of the command, hosts, scope, and expected outputs. A
question, code request, or implementation request is never permission to run.
- **Ground before you speak.** Answer from the code and the tensors, not from
memory or from what is generally true of models like this. After a context reset,
re-read before re-asserting. If the answer depends on a file, open it first.
- **State what is known.** Distinguish code-proven facts, measurements,
assumptions, and hypotheses. Lead with the best-supported conclusion; if the
evidence does not isolate one cause, say so instead of turning a plausible story
into a diagnosis.
- **Diagnose, don't fiddle.** An OOM is not fixed by trying a smaller batch and a
wedge is not fixed by relaunching. Find the tensor, the process, the handshake
that is wrong, show the evidence, then propose the change.
- **Corrections fix the analysis, not the premises.** When told you're wrong,
re-check the reasoning against the code before changing your model of the design and 
never invent a design detail the user did not state. Stop related actions, re-read the 
relevant sources, restate the corrected invariant, and identify which conclusions,  
patches, or runs it invalidates before continuing. Never preserve a premise to  
defend prior work or invent a replacement detail the user did not state.
- **When in doubt, ask.** Ambiguity about intent, permission, or training
behaviour is resolved by a question, never by a guess.



## Code

**Configs are the source of truth.** A config fully describes a run. The script
consuming it never substitutes a value the user didn't write — if a key is
missing and matters, fail loudly or forward its absence; when a block maps to a
constructor, splat it (`Cls(**cfg)`) rather than enumerating keys. A default is
warranted only when many configs share the value; then it is a named
module-level constant in the signature, never an inline literal.

```python
audio_length = config.get('audio_length', 48000)   # NO — the config lied about runtime
if 'audio_length' in config:                        # OK — let the constructor decide
    kwargs['audio_length'] = config['audio_length']
```

**No unrequested layers.** No try/except around things that don't fail here, no
back-compat shims, no re-implementing existing logic "just in case", no drive-by
renames or refactors. If extra robustness seems warranted, ask first.

**Don't narrate code.** Comment only what the code can't say: a shape contract, a
calibration the model was trained against, an upstream invariant. Most files need
no docstring. A comment describes only what its own module owns — a dataset
documents its on-disk layout, not which model consumes it; a model documents its
shape contracts, not the trainer feeding it. A comment reaching across that
boundary may reveal coupling; do not refactor it unless asked.
`models/pose/audio.py` is the standard.

**Style.** Keep Python compact. Short calls, indexing,
assignments, conditions, comprehensions, and signatures stay on one line when
readable. Only constructs that actually need multiple lines use hanging
brackets: open at the end of the line, contents one indent in, close at the
opener's indent; never align continuation lines to the bracket.

```python
pose = latents[:, :clip_len].contiguous()  # good
result = build_model(
    config=config,
    checkpoint=checkpoint,
)

pose = latents[
    :, :clip_len
]  # bad: a short expression expanded mechanically
```



## Operating rules

- **Configs are edited with extreme care** (`*.yaml`, `*.toml`, run scripts):
the minimal change that does the job, and every change reported exactly in chat.
No comments in configs — a config is values, not prose; context goes in chat.
- **No commit or push unless asked.** "Can I", "should we", "ready", "done", or a
clean tree is not permission. Propose the message and wait.
- **Never change state on a shared or multi-node machine** without explicit,
host-scoped permission: no killing processes, freeing GPUs, launching, or moving
files, and never fan out (`pdsh`, `ssh` loops). Inspecting is fine; anything that
mutates is proposed first. "Stop" or "off limits" ends all access to that host,
including cleanup; report any known leftovers instead.
- **Leave remote repos exactly as found.** No untracked files, logs, scratch
configs, or outputs left on any node while access is authorized. If authorization
is withdrawn, report what remains without reconnecting.
- **Run output streams to the terminal.** Launchers never swallow logs into
files. Follow `run_multinode.sh`: master in the foreground, peers over attached
ssh, every node's output on the one terminal.



## Repo facts

- **Launchers are ephemeral pointers.** `run_multinode.sh`, `run_llm.sh` and the
like are re-edited per launch. Do not treat their current target or GPU count as
a property of a config; inspect and report launcher state only for launch
readiness.
- **Pose v1 and v2 are the same latents.** The suffix marks which
data-processing round encoded the clip — time, not a different model or
distribution. Never flag v1/v2 mixing as a mismatch.

