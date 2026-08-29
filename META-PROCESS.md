# Meta Process

## Velocity prior

Prefer the smallest action that changes the next project decision. Velocity
means moving an item on the progress chart toward an executable world, an
admission decision, a learner checkpoint, or a transfer conclusion. More
documents, tests, metrics, and runs are not progress unless they cause one of
those state changes.

When several actions are valid, prefer in this order:

1. remove a blocker on the current path;
2. make an existing claim executable;
3. obtain evidence that selects between live alternatives;
4. improve reusable apparatus required by more than one imminent path item;
5. open a new path item only when current evidence creates the need.

## Every action maps to the progress chart

Before acting, name a row from `DEVELOPMENT-PATH.md` and record:

```text
Path row:
State change sought:
Decision unlocked:
Cheapest sufficient verification:
Expected local/GPU cost:
```

If an action cannot name a row and a state change, it does not enter the work
queue. New evidence may add or revise a row, but the chart must be updated in
the same change.

## Tests are purchased only when needed

Run a test when its result can change at least one of:

- whether a family is valid or admitted;
- whether an implementation is safe to use;
- which learner action is selected next;
- whether a checkpoint is retained, replayed, compared, or stopped;
- whether a capability edge or transfer claim is kept; or
- whether the next higher-cost test is justified.

Use the cheapest sufficient level:

- **Build checks:** formatting, unit tests, type checks, deterministic replay.
- **Semantic checks:** exact enumeration, information leakage, baselines,
  ambiguity, and invariance. Run once per semantic family version.
- **Checkpoint sentinels:** small learner-facing checks needed to choose the next
  action.
- **Matched branches:** adaptation comparisons only when two live actions cannot
  otherwise be ranked.
- **Transfer milestones:** broad transfer and retention only when a checkpoint
  can support the project claim.

Do not run a broad suite merely because it exists. Do not repeat a semantic
audit for newly sampled instances of an unchanged admitted family.

## Compute rule

- Use local CPU when the expected test is at most ten minutes.
- Use Kaggle when the CPU estimate exceeds ten minutes and include launch/setup
  time in the cost.
- Any GPU launch requires explicit user authorization, a declared path row,
  fixed configuration, budget, stop rule, and decision interpretation.
- The source commit used by Kaggle must be committed, remotely reachable, and
  identified by its full SHA.
- Checkpoints and recovery payloads remain on Kaggle. Only compact verified
  evidence is collected locally.

## Engineering rule

Use maintained libraries for model bodies, training, optimization,
checkpointing, distributed execution, serialization, and platform access.
Project code owns only world semantics, public/privileged information
boundaries, narrow learner adapters, audits, and scientific comparisons.

## Reporting rule

Report in this order:

1. the path row and capability/world relation affected;
2. the state before and after;
3. whether the result is apparatus evidence, world-validity evidence, learner
   evidence, or transfer evidence;
4. the decision now enabled; and
5. the next chart row.

An apparatus failure says the measurement could not be made. It is not evidence
against a world or capability. A source-world learner result is not transfer.
