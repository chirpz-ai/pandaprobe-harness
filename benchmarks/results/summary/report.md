# PandaBench results

## Headline (eval phase)

| benchmark      | model   | arm      |   n_tasks |   pass_at_1 |   pass_hat_k |   mean_cost_usd |   mean_input_tokens |   n_error |
|:---------------|:--------|:---------|----------:|------------:|-------------:|----------------:|--------------------:|----------:|
| appworld       | mock    | baseline |         2 |           0 |            0 |               0 |                  10 |         0 |
| appworld       | mock    | harness  |         2 |           0 |            0 |               0 |                  10 |         0 |
| tau2           | mock    | baseline |         2 |           0 |            0 |               0 |                  10 |         0 |
| tau2           | mock    | harness  |         2 |           0 |            0 |               0 |                  10 |         0 |
| terminal_bench | mock    | baseline |         2 |           0 |            0 |               0 |                  10 |         0 |
| terminal_bench | mock    | harness  |         2 |           0 |            0 |               0 |                  10 |         0 |

## Harness vs baseline (paired pass@1)

| benchmark      | model   |   n_pairs |   rate_a |   rate_b |   delta |   ci_low |   ci_high |   p_value | underpowered   |
|:---------------|:--------|----------:|---------:|---------:|--------:|---------:|----------:|----------:|:---------------|
| appworld       | mock    |         2 |        0 |        0 |       0 |        0 |         0 |         1 | True           |
| tau2           | mock    |         2 |        0 |        0 |       0 |        0 |         0 |         1 | True           |
| terminal_bench | mock    |         2 |        0 |        0 |       0 |        0 |         0 |         1 | True           |

## Harness telemetry

| benchmark      | model   | phase    |   trials |   rules_active_max |   rules_candidate_max |   rules_retired_max |   notices_total |   breach_rate |
|:---------------|:--------|:---------|---------:|-------------------:|----------------------:|--------------------:|----------------:|--------------:|
| appworld       | mock    | eval     |        2 |                  0 |                     0 |                   0 |               0 |             0 |
| appworld       | mock    | learning |        1 |                  0 |                     0 |                   0 |               0 |             0 |
| tau2           | mock    | eval     |        2 |                  0 |                     0 |                   0 |               0 |             0 |
| tau2           | mock    | learning |        1 |                  0 |                     0 |                   0 |               0 |             0 |
| terminal_bench | mock    | eval     |        2 |                  0 |                     0 |                   0 |               0 |             0 |
| terminal_bench | mock    | learning |        1 |                  0 |                     0 |                   0 |               0 |             0 |

## Cost / overhead

| benchmark      | model   |   baseline_input_tokens |   harness_input_tokens |   overhead_tokens |   mean_cost_baseline |   mean_cost_harness |
|:---------------|:--------|------------------------:|-----------------------:|------------------:|---------------------:|--------------------:|
| appworld       | mock    |                      10 |                     10 |                 0 |                    0 |                   0 |
| tau2           | mock    |                      10 |                     10 |                 0 |                    0 |                   0 |
| terminal_bench | mock    |                      10 |                     10 |                 0 |                    0 |                   0 |

## Methodology notes

- **Power caveat.** At ~30-40 eval tasks, McNemar detects only large deltas (~10+ points); small effects are underpowered even pooling seeds. Results are directional — read the bootstrap CIs, not just point deltas.
- **Nondeterminism.** Current Claude models reject `temperature`, so trial-to-trial variance comes from natural model nondeterminism; no sampler seed is forced.
- **Preamble confound.** The arm-B harness preamble + 14 tools cost context/tokens every turn (see cost/overhead), which can depress arm B on long tasks independent of rule quality.
- **Checkpoints.** Checkpoint 1 (metric<->failure calibration) and Checkpoint 2 (rule promotion; `learning_outcome` in each manifest) gate the full matrix; see IMPLEMENTATION_NOTES.md.
