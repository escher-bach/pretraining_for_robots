# Fixed GPU20 result

The reviewed fixed-mixture abstract run completed on 2026-09-12 and its
receipt is verified at
`audit/runs/pretraining-first-system-f8bcb79/receipt.json`. Kaggle completed
the run at
`https://www.kaggle.com/code/aniruddhavarma/pretraining-first-system-f8bcb79/versions/1`.

The exact committed source was
`f8bcb7977ba77f9c5fb8f6c9f2d008efed1a7b9a`. The exact configuration was
`configs/first_training_system_gpu_proposal.toml` with SHA256
`c9c1776db9a2b87f6ffa0e99602d3f844dae103ab0a8a903fff3c1cef203a183`.
The run used one visible NVIDIA T4, world size 1, 20 updates, and 40 episode
presentations. The receipt status is `complete`; Kaggle's Rust and Python test
suites passed (115 Python tests in the collected run).

Held-out action loss improved from `0.025209` at update 0 to `0.016089` at
update 20. Mean physical error improved from `0.245424` to `0.143723`.
Evaluator success remained `2/32`; the fixed baselines were inaction `4/32`,
reactive `7/32`, and public teacher `32/32`. This is a learner/support-fit
diagnostic within one sampled family, not evidence of acquired reaching,
robust disturbance rejection, transfer, or justification for autonomous
scaling.

The supplied calibration prefix is sequential in event order, but its actual
physical timestamps collapse to zero. Timing-sensitive calibration evidence is
therefore not established. The next smallest evidence question is whether the
policy learns informative early decisions or mainly predicts small late teacher
actions; that diagnostic remains proposed rather than launched.
