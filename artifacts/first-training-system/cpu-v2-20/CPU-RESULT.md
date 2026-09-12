# CPU V2 fixed pilot

The authorized fixed profile completed 20 updates on CPU with one thread,
batch size 2, 64 generated training episodes, 32 held-out episodes, and 40
training presentations. The source config is
`configs/first_training_system_scientific_cpu.toml`, whose world version is
`calibrated-reach-v2`.

Validation before the run: 40 targeted Python tests passed, including the
width-6 generated path, independent sensor/body realization checks, the V2
teacher/reactive/inaction bracket, common-content gradients, and checkpoint
resume evidence.

Held-out teacher-forced action loss was 0.025209 at update 0, 0.015564 at
update 5, 0.016654 at update 10, and 0.016088 at update 20. Closed-loop
evaluation used all 32 factorial cells and the same seed cohort at updates 0
and 20:

| policy | update 0 | update 20 | mean physical error at 20 |
| --- | ---: | ---: | ---: |
| learner | 2/32 | 2/32 | 0.143723 |
| learner without goal | 2/32 | 2/32 | 0.144035 |
| learner without calibration | 2/32 | 2/32 | 0.141643 |
| learner without action history | 2/32 | 2/32 | 0.143362 |
| teacher | 32/32 | 32/32 | 0.000688 |
| fixed reactive | 7/32 | 7/32 | 0.354527 |
| inaction | 4/32 | 4/32 | 0.122437 |

The regenerated reviewer trace in `../episode-trace-world-only.txt` is
coherent: it begins with reset, public observation and goal, shows the full
sequence of four signed calibration pulse/observation pairs, then emits
bounded action queries and executions whose sensor error falls from 0.147676
to approximately 4.1e-9 before episode end. Its sections label reviewer
metadata and teacher supervision separately from public event content. Use
`../arm-probe-trace-world-only.txt` when inspecting the public event stream
alone; private maps, latent state, seed, and transition tables are absent from
that stream.

This is apparatus and within-family acquisition evidence only. Twenty updates
cover 40 of 64 training examples; the unchanged 2/32 success rate means the
pilot does not yet establish learned closed-loop reaching, scheduling, scale,
compositional generation, robust disturbance rejection, transfer, or
grounding. The receipt is `scientific_receipt.json` and the final checkpoint
is `checkpoint-20`.
