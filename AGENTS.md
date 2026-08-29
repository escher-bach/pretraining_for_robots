# Repository Working Agreement

Read `GOAL.md`, `META-PROCESS.md`, `DEVELOPMENT-PATH.md`, and `CARDS.md` before
selecting work. Use `APPARATUS.md` for code and run commands.

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
