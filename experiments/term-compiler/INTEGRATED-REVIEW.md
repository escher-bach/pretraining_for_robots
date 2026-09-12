# Integrated review, 2026-09-13

Path row: Compiler-backed procedural GPU system.

The compiler/runtime semantic path passes the bounded integrated review. This
is a restricted typed operator compiler: reaching with every legal combination
of actuator lag, goal switching, and disturbance. It does not establish
arbitrary graph execution, the full capability portfolio, or transfer.

Reviewed the Rust graph lowering, Python import/execution boundary, public
event rendering, and public-history teacher. The initial implementation
discarded compiled plant gain, goal-switch magnitude, and embodiment widths;
these are now propagated and checked. Runtime parameters are checked against
the node graph. Unknown operators, invalid connections, and private process
ports are rejected. Lag identification uses only the public calibration
prefix, and its bounded per-teacher cache is keyed by public data.

Executed `cargo run -q -p pretraining-term-compiler --bin process-export --
--seed 20260912 --count 32`, imported each generated program into `ProcessIR`,
and generated episodes with seed `314159 + generator_index`, both absolute
and relative goals, and each program's declared embodiment. This covers eight
operator compositions crossed with four sensor/body combinations and two goal
modes: 64 episodes. All 64 teacher trajectories succeeded; maximum final
physical error was 0.004655737284935148. Public event JSON excluded plant gain,
process IR, body/sensor matrices, seed, and target state. Event times were
monotonic and action target counts matched each compiled horizon. This is
semantic evidence, not learner acquisition evidence; no CPU training was run.

Training integration findings sent to the implementation agent for repair:

- Select from the actual Rust manifest and cover all eight compositions;
  remove ignored legacy factorial controls from the active profile.
- Evaluate those same process contracts and persist checkpoint evidence with
  truthful world/teacher versions, dimensions, horizons, and seed accounting.
- Save periodically, expose standard Trainer checkpoint resume, and make the
  subprocess timeout consistent with the declared three-hour budget.

The parent must confirm those active training edits before launch. The
semantic review introduces no additional user approval gate.

Follow-up review found and repaired a dispatch whitelist that omitted five
configured compiled families. Compiled family names now also reject missing
Rust manifests rather than entering the historical Python fallback.

Two active findings still prevent an integrated launch GO: the compiled
closed-loop evaluator initially covered only eight sensor-8/body-2 absolute
episodes despite declaring 256 crossed episodes, and a later teacher speed
shortcut fixed identified lag response to 1.0 during generation. The latter
was introduced after the 64-episode semantic receipt above and is not covered
by it. Both findings were sent for immediate repair, together with periodic
checkpoint/resume and persistent evaluation evidence.

Final readiness review: those launch blockers are resolved in the current
source. Generation again uses the identified public-history bounded teacher;
`artifacts/composed-process-audit-current.json` records the current eight-family
semantic audit. Compiled evaluation now crosses all 64 family/embodiment/goal
cells over 256 held-out episode seeds with matched policy comparisons. The
review corrected compiled success scoring to use the configured physical
tolerance and made requested ablations obey their enable flag. Standard
Trainer checkpoints are saved every 256 updates, partial receipts are written
atomically after evaluation/save, and scientific checkpoint resume is exposed
and validated. Python compilation passed for the three integrated modules.

GO for the authorized GPU acquisition run, subject to its existing executable
preflight checks. This is code and semantic readiness, not evidence that the
learner has acquired the capabilities. The wider 64-episode receipt above is
historical; the current-generation audit is the explicitly named artifact.
