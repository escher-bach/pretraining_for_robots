# Factored RDDL source experiment

This isolated R3c experiment checks a maintained factored-world source format;
it does **not** integrate RDDL with Rust, the learner renderer, a card, or a
training profile.

`visible-reassignment-domain.rddl` declares two object types, a hidden
source-to-channel assignment, hidden source values and bound-source state, a
public channel-value observation, a public reassignment boundary, a relational
action precondition, and deterministic conditional transition functions. The
only public action is `pulse(channel)`: it binds the source reached at the
first channel, then the declared boundary reassigns sources. A successful
second pulse must follow that same hidden source to its new public channel.
The four instances form one preserving source-permutation pair, one
meaning-changing target-relation edit, and one hidden-only boundary variant.

Run on CPU from the repository root:

```powershell
python -m venv .venv-rddl
.\.venv-rddl\Scripts\python.exe -m pip install -r experiments\factored-rddl\requirements.txt
.\.venv-rddl\Scripts\python.exe experiments\factored-rddl\conformance.py --output "$env:TEMP\factored-rddl-grounded-traces.json"
```

The runner loads and grounds all files with pyRDDLGym, exhausts the 9 legal
two-action traces of the base instance, checks the preserving and
meaning-changing transformations, checks that `private-token` has no public
effect, and emits deterministic canonical JSON.  Each trace has separate
`public_observation` and `privileged_state` fields so a later adapter cannot
mistake the artifact for learner-visible data.

The public observation has channel tokens only; source tokens, assignment,
source values, and the bound source are checked to be absent. This remains a
small source/grounding conformance witness, not a Card 06 migration, learner
rendering, or independent verifier of future Rust lowering. pyRDDLGym is only
the maintained parser, grounder, and simulator for this gate.
