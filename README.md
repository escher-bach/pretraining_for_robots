# Pretraining for Robots

**Goal:** abstract pretraining to induce transferable embodied capabilities via
capability decomposition, a process algebra, and procedurally generated worlds.

The repository has one route through the project:

1. [GOAL.md](GOAL.md) defines the claim and the capability target.
2. [META-PROCESS.md](META-PROCESS.md) defines how work is selected.
3. [DEVELOPMENT-PATH.md](DEVELOPMENT-PATH.md) defines the worlds, cards,
   admission gates, current position, and next path.
4. [APPARATUS.md](APPARATUS.md) explains the code and exact commands.

The executable surface is deliberately small:

```text
crates/       Rust world semantics, audits, event records, and Python bridge
python/       maintained-model training adapters and tests
configs/      declared run configurations
kaggle/       experiment registry
artifacts/    checked-in model configuration
tools/        Kaggle CLI control plane
```

Current position: the common Rust apparatus, finite G0 audit layer, three
prototype worlds, learner event boundary, and Card 04 semantic audit are built.
Card 04 still needs a learner-facing profile adapter. Cards 03, 02, 05, and 06
then complete the proposed four-trunk seed portfolio. Multi-world pretraining
starts only after that portfolio passes the gate in `DEVELOPMENT-PATH.md`.

Quick verification:

```powershell
cargo test --workspace --locked
$env:PYTHONPATH = (Resolve-Path python).Path
python -m unittest discover -s python/tests -v
```

No GPU run is launched without explicit user authorization.
