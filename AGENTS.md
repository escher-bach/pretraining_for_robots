# Repository Working Agreement

## Current user authorization — 2026-09-12

The user explicitly directs end-to-end implementation of a stronger compiler-backed
procedural training system and meaningful larger GPU training. This supersedes
the older smallest-step, CPU-first, fixed-20-update and per-stage user-review
restrictions below. Build and integrate the required compiler, composed trajectory
worlds, shared learner/data path, and GPU execution without additional approval
gates. Use CPU for build/semantic checks, GPU for training. Preserve scientific
honesty and public/private boundaries; repair defects instead of bypassing them.
Use Luna implementation agents and Astra integrated review. Stop safely if the
five-hour account usage reaches 95% used, preserving exact continuation state.

Read `GOAL.md`, `LEARNER-INPUT-ARCHITECTURE.md`, `META-PROCESS.md`,
`EMBODIED-PROCESS.md`, `DEVELOPMENT-PATH.md`, and `CARDS.md` before selecting
work. Use `APPARATUS.md` for code and run commands, and `HANDOFF.md` for where
the current work stopped and what building it found.

- Map every action to one progress-chart row and update that row when its state
  changes.
- Prefer the smallest decision-changing action. Do not create speculative cards,
  broad test matrices, or infrastructure without a live chart need.
- Do not launch a GPU run without explicit user authorization.
- Use local CPU for work expected to finish within ten minutes; otherwise use
  the declared Kaggle path after authorization.
- Use maintained libraries for model, training, checkpoint, distributed, and
  platform functionality. Keep custom code at world semantics, information
  boundaries, learner adapters, audits, and scientific comparisons.
- Keep learner-visible public data, privileged evaluation data, and generator
  metadata structurally separate.
- Treat abstract source score, learner progress, transfer, and grounding as
  different evidence levels.
- Keep the repository self-contained. Do not add references to external project
  documents or import historical document hierarchies.

## Current project objective

The active objective is the first viable abstract robot pretraining system.
Work proceeds through a small sampled trajectory system before any scheduler or
grounding work:

1. make the common content boundary and public/private trajectory schema
   executable;
2. let the user review one fixed CPU configuration and its evidence;
3. run one fixed-mixture abstract pretraining experiment only after review;
4. diagnose acquisition, scale, or scheduling from that evidence; and
5. open matched grounding only when the abstract result justifies it.

The old R3/R11 gate language and the closed R5--R10 receipts describe
historical apparatus. They remain useful evidence and must not be deleted, but
they do not block the first-system implementation. The current trajectory
world is deliberately limited to public-history identification, goal
conditioning, and regulation. It does not establish the full capability
portfolio, visual grounding, demonstrations, or transfer.

The active world is `CalibratedReachV1` in
`python/pretraining_experiments/trajectory_world.py`. It uses public temporal
sensor/goal/action histories, independently crossed two- and four-actuator
bodies and eight- and ten-channel sensors, public basis calibration, and a
minimum-effort bounded teacher. A transition table, latent state, private map,
seed, or generator metadata may never enter learner input. GPU work and the
first scientific run require explicit user verification of the fixed
configuration; CPU boundary checks are the preparation step.
