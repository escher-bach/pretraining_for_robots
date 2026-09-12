# Pretraining for Robots

**Goal:** abstract pretraining to induce transferable embodied capabilities via
capability decomposition, a process algebra, and procedurally generated worlds.

The repository has one route through the project:

1. [GOAL.md](GOAL.md) defines the claim and the capability target.
2. [LEARNER-INPUT-ARCHITECTURE.md](LEARNER-INPUT-ARCHITECTURE.md) defines the
   binding common content path for symbolic, visual, linguistic,
   proprioceptive, demonstration, and action realizations, and records the
   rejected cold-sidecar design.
3. [META-PROCESS.md](META-PROCESS.md) defines how work is selected.
4. [EMBODIED-PROCESS.md](EMBODIED-PROCESS.md) defines the world model,
   process algebra, capability graph, and transfer boundary.
5. [DEVELOPMENT-PATH.md](DEVELOPMENT-PATH.md) defines the worlds, cards,
   admission gates, current position, and next path.
6. [CARDS.md](CARDS.md) specifies every card's witness, controls, information
   boundary, baselines, admission rule, and transfer falsifier.
7. [APPARATUS.md](APPARATUS.md) explains the code and exact commands.

The executable surface is deliberately small:

```text
crates/       Rust world semantics, audits, event records, and Python bridge
python/       maintained-model training adapters and tests
configs/      declared run configurations
kaggle/       experiment registry
artifacts/    checked-in model configuration
tools/        Kaggle CLI control plane
```

Current position: R5–R10 are closed, but only Card 02 was frontier-admitted, so
R11 remains blocked. The experimental R3c compiler can generate audited finite
G0 corpora. The 2026-09-05 architecture review added R3d: the existing
eight-float symbolic boundary is one modality adapter, not the universal
learner input, and no new learner, grounding, or demonstration work may treat
the dormant additive sidecar as sufficient. See `DEVELOPMENT-PATH.md` and
`LEARNER-INPUT-ARCHITECTURE.md`.

Quick verification:

```powershell
cargo test --workspace --locked
$env:PYTHONPATH = (Resolve-Path python).Path
python -m unittest discover -s python/tests -v
```

No GPU run is launched without explicit user authorization.
