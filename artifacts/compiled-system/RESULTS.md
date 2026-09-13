# Compiled-process GPU run

This report analyzes the verified final receipt at
`audit/runs/pretraining-first-large-93a62b5/pretraining-results/first-training/scientific_receipt.json`.
The run is source-acquisition evidence from one visible Tesla T4, source
`93a62b54ce149de729066e6122860153c443d072`, configuration SHA
`c1fe759f4c4edd7ec64b6f05dfe7788ed579bfda3497ef4cd92a577fbcd14d17`, root
seed `20260912`, 4,096 updates, batch 8, and 32,768 episode presentations.
The Rust manifest contains 32 programs and has SHA
`b9ada4053262f2d021f07195334a0a2a6fa60c108f6bd661503a98411c75d94f`.

The masked action loss fell monotonically from `0.03442` at update 0 to
`0.01644` at update 4096 (52.2% lower), with most of the reduction before
update 1024. This is learner-fitting evidence only; it does not imply useful
closed-loop control.

The final closed-loop evaluation has 256 matched episodes and 1,792 rows
(eight compiled families × two sensor widths × two body widths × two goal
modes, with the configured repeated held-out support). Overall results are:

| policy | physical error | success | action effort | early action MSE | post-switch MSE |
|---|---:|---:|---:|---:|---:|
| learner | 0.17139 | 13/256 (5.1%) | 0.0000124 | 0.18374 | 0.17607 |
| inaction | 0.17128 | 12/256 (4.7%) | 0 | 0.18380 | 0.17622 |
| teacher | 0.00231 | 254/256 (99.2%) | 0.18008 | 0 | 0 |
| reactive | 0.39039 | 22/256 (8.6%) | 0.21103 | 0.33070 | 0.32325 |

The learner is effectively inaction. Its effort is about four orders of
magnitude below the teacher's, and its physical error is indistinguishable
from inaction (`+0.00011` learner minus inaction). Early and post-switch
action error are also essentially identical to inaction. Removing the goal,
calibration, or action history changes physical error by only `+0.000006`,
`-0.000093`, and `-0.000453`, respectively. The model therefore did not
demonstrate use of calibration, goal binding, action history, or switching.

Per-family learner physical error / success versus inaction is:

| compiled family | learner | inaction |
|---|---:|---:|
| reaching | 0.1784 / 9.4% | 0.1780 / 9.4% |
| actuator lag | 0.1818 / 6.2% | 0.1818 / 3.1% |
| goal switch | 0.1554 / 3.1% | 0.1553 / 3.1% |
| disturbance | 0.2054 / 0% | 0.2052 / 0% |
| lag + goal switch | 0.1790 / 3.1% | 0.1793 / 0% |
| lag + disturbance | 0.1705 / 3.1% | 0.1697 / 6.2% |
| goal switch + disturbance | 0.1526 / 9.4% | 0.1527 / 12.5% |
| lag + goal switch + disturbance | 0.1482 / 6.2% | 0.1483 / 3.1% |

The failure is therefore policy-relevant action collapse, rather than a
family-specific failure or a mere lack of final loss reduction. The public
teacher baseline is strong on matched worlds, while the reactive baseline is
worse, so the world and public-teacher path provide a meaningful control
contrast. The model reduced regression loss without acquiring the early
calibration and control decisions needed for rollout. Small late teacher actions
are one plausible contributor; these results do not isolate their contribution
from representation difficulty, conditional-mean prediction, or the difference
between demonstrated and learner-induced histories.

The receipt contains closed-loop records at updates 0, 1024, 2048, and 4096;
this report uses the final update 4096 record. No checkpoint was downloaded and
no additional GPU run was launched.

The next work should address action collapse directly, preserving the same
matched cohort. Weighting early/post-switch decisions and training on
learner-induced histories are concrete candidates; neither is established as a
fix by this receipt. Additional scale alone has not been shown to address the
observed failure: the final policy tracks inaction despite a 52% loss reduction.
