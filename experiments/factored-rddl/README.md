# Factored RDDL source experiment

This isolated R3c experiment keeps `pyRDDLGym` as the maintained RDDL parser,
grounder, and simulator. It does not integrate RDDL with Rust, a card, a
learner renderer, or a training profile.

The source model has two exchangeable hidden sources and two public channels.
`pulse(channel)` binds the source currently at that channel, increments that
source, and then fires the public reassignment boundary. A `noop` does not fire
the boundary. A later pulse must follow the public channel values to act on the
same hidden source. The four instances are a preserving source-permutation
pair, a meaning-changing target-relation edit, and a hidden-only non-fluent
variant.

## Boundary artifact

The runner exhausts all nine legal two-action traces. It emits four addressed
RLDS-shaped `public_episode` records through the
`prompted-interface/0.2` boundary—one for every reachable nonterminal
privileged state. A public record contains only learner-visible events,
target-scoped executable action references, and an empty declared native-RDDL
alignment manifest. Its public contract follows the shared 0.2 profile:
nonempty source/target embodiments and an explicit `goal_publication` that
binds the public goal carrier. Its RLDS `reward` field is always `null`.

Each public episode has separate loss-only `supervision` and a
`private_receipt`. The exact labels are recomputed from exhaustive simulator
continuations: an action is correct precisely when its best cumulative
simulator return equals the best available continuation. The public symbolic
carrier says `maximize cumulative return`; the private denotation and graph
goal semantics use the same cumulative-return criterion. The private receipt
holds source hashes/provenance, the privileged state and source/channel
assignment, per-query action outcomes, and a goal-grounding receipt. Grounding
uses exhaustive pyRDDLGym simulation and content hashing, ties the private
receipt to the public-contract `goal_publication` and addressed `GoalCarrier`, and does
not import the prompted-interface fixture's bay/dock text-marker heuristic.

The public profile follows `prompted-interface/0.2`, but this experiment does
not import that fixture validator: its private evaluator and grounding schema
are deliberately fixture-specific. Instead the private evaluator is explicitly
versioned as `factored-rddl-exhaustive-evaluator/0.2`, and the local validator
checks the public 0.2 contract plus RDDL-specific exhaustive evidence.

The artifact also carries a private canonical graph with every *reachable*
privileged state and legal action through this fixed horizon. This is an
independent `finite-rddl-graph/0.1` simulator receipt, not a Rust Graph IR
lowering or a claim of an unbounded total RDDL transition system. Its replay
receipt checks each legal complete trace against graph legality, reward,
terminal/truncation, public observation, and privileged destination state.

`source-value(source)` remains an RDDL `int`, so this source syntax does not
claim an arbitrary finite-valued fluent domain. The admitted graph is finite
because the horizon is fixed at two and every reachable source value is in
`{0, 1, 2}`; the conformance test checks that bound. A future general factored
source profile must declare finite source-value domains before it can claim
finite-state coverage beyond this finite-horizon witness.

## Run

From the repository root, on CPU:

```powershell
python -m venv .venv-rddl
.\.venv-rddl\Scripts\python.exe -m pip install -r experiments\factored-rddl\requirements.txt
.\.venv-rddl\Scripts\python.exe -m unittest discover -s experiments\factored-rddl -p 'test_*.py' -v
.\.venv-rddl\Scripts\python.exe experiments\factored-rddl\conformance.py --output "$env:TEMP\factored-rddl-prompted-interface-v02.json"
```

The focused tests retain the preserving-permutation, relation-edit, and
hidden-only controls across the full exhaustive trace matrix; enforce the
intervention-triggered boundary; check the exact public observation schema;
and validate privacy, actor/body action scope, exact supervision, canonical
reachable-graph semantics, and replay equivalence.

Candidate enumeration is checked against the maintained pyRDDLGym model: this
instance declares `max-nondef-actions = 1` and boolean action fluents, so the
adapter exhausts all four Boolean assignments, proves the joint-true assignment
illegal, and proves explicit-false assignments replay exactly as omitted noop.
The resulting complete legal candidate surface is `noop` plus each singleton
grounded action; another action-space shape is rejected rather than silently
treated as complete.
