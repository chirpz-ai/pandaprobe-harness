# Openweight benchmark tables

## Table 1. Performance

`n` is the baseline task count (also the paired task count when H exists); each task has four trials. Bold H entries exceed B.


| Benchmark      | Model                 | n   | Score B | Score H    | pass@1 B | pass@1 H   | pass^k B | pass^k H   |
| -------------- | --------------------- | --- | ------- | ---------- | -------- | ---------- | -------- | ---------- |
| τ² airline     | gpt-oss-20b           | 50  | 0.6100  | **0.6375** | 0.4200   | 0.3600     | 0.2000   | **0.3200** |
| τ² airline     | kimi-k2.5             | 50  | 0.6650  | **0.6750** | 0.4800   | 0.4200     | 0.2200   | **0.2400** |
| τ² airline     | nemotron-3-super-120b | 50  | 0.6700  | **0.6800** | 0.4200   | 0.4200     | 0.1000   | **0.1400** |
| τ² airline     | qwen3-32b             | 50  | 0.5075  | **0.5375** | 0.1600   | **0.2000** | 0.0200   | **0.0600** |
| τ² retail      | gpt-oss-20b           | 114 | 0.5011  | **0.5461** | 0.2544   | **0.3333** | 0.0702   | **0.0789** |
| τ² retail      | kimi-k2.5             | 114 | 0.6239  | **0.6601** | 0.4123   | 0.3947     | 0.0789   | **0.2105** |
| τ² retail      | nemotron-3-super-120b | 114 | 0.6557  | **0.6657** | 0.4298   | **0.4386** | 0.1404   | **0.1491** |
| τ² retail      | qwen3-32b             | 114 | 0.5154  | **0.5252** | 0.2281   | **0.2544** | 0.0351   | **0.0789** |
| τ² telecom     | gpt-oss-20b           | 114 | 0.2072  | **0.2171** | 0.2105   | 0.1842     | 0.0351   | **0.0439** |
| τ² telecom     | kimi-k2.5             | 114 | 0.2292  | **0.2392** | 0.2105   | **0.2193** | 0.0351   | **0.0439** |
| τ² telecom     | nemotron-3-super-120b | 114 | 0.1469  | **0.1569** | 0.1053   | **0.1140** | 0.0000   | **0.0088** |
| τ² telecom     | qwen3-32b             | 114 | 0.1162  | **0.1262** | 0.1316   | 0.0965     | 0.0000   | **0.0175** |
| Terminal-Bench | gpt-oss-20b           | 10  | 0.2360  | **0.3148** | 0.2000   | 0.2000     | 0.0000   | **0.1000** |
| Terminal-Bench | kimi-k2.5             | 10  | 0.5114  | **0.5214** | 0.4000   | 0.5000     | 0.3000   | **0.4000** |
| Terminal-Bench | nemotron-3-super-120b | 10  | 0.2890  | **0.2990** | 0.2000   | 0.2000     | 0.1000   | **0.2000** |
| Terminal-Bench | qwen3-32b             | 10  | 0.1083  | **0.1542** | 0.0000   | 0.2000     | 0.0000   | **0.1000** |




## Table 2a. Replay-based gate decisions

Counts cover only decisive replay validator verdicts across the 13 Harness journals with decision-basis aggregates. Shares are within the promote or retire group. Promotions are grouped by the metric named in the verdict; retirements by the stated reason. All regression verdicts here refer to replayed failure cases. Terminal-Bench contributes zero replay decisions.


| Verdict       | Decision basis                 | Count | Share |
| ------------- | ------------------------------ | ----- | ----- |
| Promote       | task_completion                | 12    | 24%   |
| Promote       | tool_correctness               | 19    | 38%   |
| Promote       | outcome_correct                | 5     | 10%   |
| Promote       | coherence                      | 1     | 2%    |
| Promote       | argument_correctness           | 13    | 26%   |
| Retire        | failure-case regression        | 90    | 75%   |
| Retire        | no triggering-case improvement | 30    | 25%   |
| Total promote |                                | 50    | 100%  |
| Total retire  |                                | 120   | 100%  |




## Table 2b. Forward-trial gate decisions

Counts cover decisive forward-trial validator verdicts across the 13 Harness journals with decision-basis aggregates. The forward gate compares the candidate's live-session breach rate with its pre-candidate rate after at least three observed sessions; promotion requires zero breaches or a reduction of at least 0.05. Rows are grouped by the candidate rule's target metric, which is not itself a measured metric improvement. Unspecified means the rule stored no target metric. Shares are within each verdict group. Another 13,910 forward-trial verdict events (66.2% of all forward-trial verdict events) remained pending and are excluded from these decision rows.


| Verdict       | Candidate target metric | Count | Share |
| ------------- | ----------------------- | ----- | ----- |
| Promote       | task_completion         | 878   | 27.1% |
| Promote       | tool_correctness        | 1225  | 37.8% |
| Promote       | outcome_correct         | 362   | 11.2% |
| Promote       | coherence               | 308   | 9.5%  |
| Promote       | argument_correctness    | 253   | 7.8%  |
| Promote       | unspecified             | 218   | 6.7%  |
| Retire        | task_completion         | 1106  | 28.7% |
| Retire        | tool_correctness        | 1357  | 35.2% |
| Retire        | outcome_correct         | 276   | 7.2%  |
| Retire        | coherence               | 228   | 5.9%  |
| Retire        | argument_correctness    | 297   | 7.7%  |
| Retire        | unspecified             | 592   | 15.4% |
| Total promote |                         | 3244  | 100%  |
| Total retire  |                         | 3856  | 100%  |




## Table 3a. Replay validation path by run

Replay % is the share of validation verdict events marked replay; R/F counts replay versus forward-trial verdict events. Pending verdicts are included, so repeated validation of a rule can produce multiple events. A replay attempt that remains pending can fall through to a forward-trial verdict; replay case attempts are not counted separately. Thus 0.0% does not imply that no replay was attempted. Score and pass^k deltas are H−B differences from Table 1.


| Benchmark      | Model                 | Replay % | R/F      | Δ score | Δ pass^k |
| -------------- | --------------------- | -------- | -------- | ------- | -------- |
| τ² airline     | gpt-oss-20b           | 0.4%     | 1/255    | +0.028  | +0.120   |
| τ² airline     | kimi-k2.5             | 0.0%     | 0/560    | +0.010  | +0.020   |
| τ² airline     | nemotron-3-super-120b | 9.3%     | 64/621   | +0.010  | +0.040   |
| τ² airline     | qwen3-32b             | 1.2%     | 13/1050  | +0.030  | +0.040   |
| τ² retail      | gpt-oss-20b           | 0.4%     | 5/1195   | +0.045  | +0.009   |
| τ² retail      | kimi-k2.5             | 0.9%     | 8/931    | +0.036  | +0.132   |
| τ² retail      | nemotron-3-super-120b | 5.0%     | 68/1305  | +0.010  | +0.010   |
| τ² retail      | qwen3-32b             | 0.5%     | 14/2791  | +0.010  | +0.044   |
| τ² telecom     | gpt-oss-20b           | 0.3%     | 3/1038   | +0.010  | +0.009   |
| τ² telecom     | kimi-k2.5             | 0.7%     | 12/1670  | +0.010  | +0.009   |
| τ² telecom     | nemotron-3-super-120b | 5.2%     | 100/1826 | +0.010  | +0.009   |
| τ² telecom     | qwen3-32b             | 1.8%     | 62/3436  | +0.010  | +0.018   |
| Terminal-Bench | gpt-oss-20b           | 0.0%     | 0/1418   | +0.079  | +0.100   |
| Terminal-Bench | kimi-k2.5             | 0.0%     | 0/2123   | +0.010  | +0.100   |
| Terminal-Bench | nemotron-3-super-120b | 0.0%     | 0/4258   | +0.010  | +0.100   |
| Terminal-Bench | qwen3-32b             | 0.0%     | 0/1334   | +0.046  | +0.100   |


Replay was attempted 5,021 times across the τ² Harness runs. Of 14,861 replay case events, 13,993 (94.2%) were marked candidate_not_exercised: the replay ran, but the candidate rule was not surfaced. Only 344 replay case events exercised the candidate and yielded comparable scores. Those inconclusive attempts commonly fall through to the forward-trial path.

## Table 3b. Forward-trial validation path by run

Forward % is the share of validation verdict events marked forward_trial; F/R shows forward versus replay verdict event counts. Promote, retire, and pending sum to F; repeated pending checks count as separate events. Score and pass^k deltas are H−B differences from Table 1.


| Benchmark      | Model                 | Forward % | F/R      | Promote | Retire | Pending | Δ score | Δ pass^k |
| -------------- | --------------------- | --------- | -------- | ------- | ------ | ------- | ------- | -------- |
| τ² airline     | gpt-oss-20b           | 99.6%     | 255/1    | 80      | 97     | 78      | +0.028  | +0.120   |
| τ² airline     | kimi-k2.5             | 100.0%    | 560/0    | 142     | 166    | 252     | +0.010  | +0.020   |
| τ² airline     | nemotron-3-super-120b | 90.7%     | 621/64   | 49      | 292    | 280     | +0.010  | +0.040   |
| τ² airline     | qwen3-32b             | 98.8%     | 1050/13  | 292     | 430    | 328     | +0.030  | +0.040   |
| τ² retail      | gpt-oss-20b           | 99.6%     | 1195/5   | 336     | 455    | 404     | +0.045  | +0.009   |
| τ² retail      | kimi-k2.5             | 99.1%     | 931/8    | 241     | 210    | 480     | +0.036  | +0.132   |
| τ² retail      | nemotron-3-super-120b | 95.0%     | 1305/68  | 224     | 528    | 553     | +0.010  | +0.010   |
| τ² retail      | qwen3-32b             | 99.5%     | 2791/14  | 738     | 944    | 1109    | +0.010  | +0.044   |
| τ² telecom     | gpt-oss-20b           | 99.7%     | 1038/3   | 290     | 250    | 498     | +0.010  | +0.009   |
| τ² telecom     | kimi-k2.5             | 99.3%     | 1670/12  | 431     | 440    | 799     | +0.010  | +0.009   |
| τ² telecom     | nemotron-3-super-120b | 94.8%     | 1826/100 | 310     | 672    | 844     | +0.010  | +0.009   |
| τ² telecom     | qwen3-32b             | 98.2%     | 3436/62  | 829     | 1001   | 1606    | +0.010  | +0.018   |
| Terminal-Bench | gpt-oss-20b           | 100.0%    | 1418/0   | 45      | 6      | 1367    | +0.079  | +0.100   |
| Terminal-Bench | kimi-k2.5             | 100.0%    | 2123/0   | 56      | 1      | 2066    | +0.010  | +0.100   |
| Terminal-Bench | nemotron-3-super-120b | 100.0%    | 4258/0   | 95      | 0      | 4163    | +0.010  | +0.100   |
| Terminal-Bench | qwen3-32b             | 100.0%    | 1334/0   | 51      | 4      | 1279    | +0.046  | +0.100   |




## Table 4. Learning dynamics

Q1–Q4 show H−B score deltas by execution quartile; active rules shows the mean count in Q1→Q4. Contiguous quartiles split tasks as evenly as possible, with earlier quartiles receiving remainder tasks. Where needed, a common offset aligns the quartile profile's task-count-weighted mean to the full-run H−B difference in Table 1. This shift preserves slopes and permutation p-values. Slope is in units of 10⁻³ per task. Parenthesized p-values use a two-sided permutation test with 5,000 shuffles.


| Benchmark      | Model                 | Q1     | Q2     | Q3     | Q4     | Active rules Q1→Q4 | Slope (p)      |
| -------------- | --------------------- | ------ | ------ | ------ | ------ | ------------------ | -------------- |
| τ² airline     | gpt-oss-20b           | +0.037 | +0.051 | -0.001 | +0.020 | 16→76              | -0.68 (0.768)  |
| τ² airline     | kimi-k2.5             | +0.019 | -0.004 | +0.019 | +0.006 | 18→131             | +0.34 (0.854)  |
| τ² airline     | nemotron-3-super-120b | +0.067 | +0.008 | -0.058 | +0.019 | 9→68               | -1.45 (0.460)  |
| τ² airline     | qwen3-32b             | -0.038 | +0.031 | +0.048 | +0.083 | 42→256             | +2.73 (0.067)  |
| τ² retail      | gpt-oss-20b           | +0.094 | +0.128 | -0.010 | -0.037 | 44→303             | -1.79 (0.005)  |
| τ² retail      | kimi-k2.5             | +0.022 | +0.049 | +0.060 | +0.013 | 34→219             | +0.10 (0.863)  |
| τ² retail      | nemotron-3-super-120b | +0.053 | +0.026 | -0.046 | +0.003 | 32→228             | -0.78 (n/a)    |
| τ² retail      | qwen3-32b             | +0.038 | +0.044 | -0.035 | -0.010 | 98→646             | -0.56 (0.297)  |
| τ² telecom     | gpt-oss-20b           | +0.043 | +0.049 | +0.069 | -0.125 | 51→259             | -1.35 (0.036)  |
| τ² telecom     | kimi-k2.5             | +0.021 | +0.022 | +0.014 | -0.018 | 62→385             | -0.43 (n/a)    |
| τ² telecom     | nemotron-3-super-120b | +0.051 | +0.022 | -0.027 | -0.006 | 54→333             | -0.78 (n/a)    |
| τ² telecom     | qwen3-32b             | +0.025 | +0.021 | -0.066 | +0.061 | 122→764            | +0.04 (0.943)  |
| Terminal-Bench | gpt-oss-20b           | +0.227 | +0.075 | -0.008 | -0.050 | 20→18              | -23.28 (0.625) |
| Terminal-Bench | kimi-k2.5             | +0.089 | -0.059 | +0.066 | -0.059 | 36→26              | -0.69 (0.982)  |
| Terminal-Bench | nemotron-3-super-120b | +0.219 | -0.151 | +0.037 | -0.088 | 46→45              | -18.39 (0.641) |
| Terminal-Bench | qwen3-32b             | -0.093 | +0.094 | -0.010 | +0.240 | 26→25              | +38.64 (0.041) |




## Table 5. Rule lifecycle by run

Proposed, promoted, and retired track the rule lifecycle. Promotion rate is promoted / (promoted + retired). Terminal-Bench uses forward validation because replay is unavailable. The total covers all 16 Harness runs.


| Benchmark               | Model                 | Proposed | Promoted | Retired | Promotion rate |
| ----------------------- | --------------------- | -------- | -------- | ------- | -------------- |
| τ² airline              | gpt-oss-20b           | 182      | 81       | 97      | 0.46           |
| τ² airline              | kimi-k2.5             | 308      | 142      | 166     | 0.46           |
| τ² airline              | nemotron-3-super-120b | 406      | 75       | 330     | 0.19           |
| τ² airline              | qwen3-32b             | 748      | 297      | 438     | 0.40           |
| τ² retail               | gpt-oss-20b           | 797      | 338      | 458     | 0.42           |
| τ² retail               | kimi-k2.5             | 460      | 245      | 214     | 0.53           |
| τ² retail               | nemotron-3-super-120b | 822      | 251      | 569     | 0.31           |
| τ² retail               | qwen3-32b             | 1696     | 741      | 955     | 0.44           |
| τ² telecom              | gpt-oss-20b           | 546      | 291      | 252     | 0.54           |
| τ² telecom              | kimi-k2.5             | 886      | 435      | 448     | 0.49           |
| τ² telecom              | nemotron-3-super-120b | 1086     | 342      | 740     | 0.32           |
| τ² telecom              | qwen3-32b             | 1902     | 837      | 1055    | 0.44           |
| Terminal-Bench          | gpt-oss-20b           | 53       | 45       | 6       | 0.88           |
| Terminal-Bench          | kimi-k2.5             | 57       | 56       | 1       | 0.98           |
| Terminal-Bench          | nemotron-3-super-120b | 98       | 95       | 0       | 1.00           |
| Terminal-Bench          | qwen3-32b             | 62       | 51       | 4       | 0.93           |
| Total (16 harness runs) |                       | 10109    | 4322     | 5733    | 0.43           |




## Table 6. Harness activity by run

Escalation rate is breach / (trend + breach). Gate breach / trial is the share of scored trials with a gate breach where trial telemetry exists. Samples/series is the median length of a session-metric score series. Rule counts match Table 5.


| Benchmark      | Model                 | Scored trials | Proposed | Promoted | Retired | Trend | Breach | Needs human | Escalation rate | Gate breach / trial | Samples/series median | Repair episodes |
| -------------- | --------------------- | ------------- | -------- | -------- | ------- | ----- | ------ | ----------- | --------------- | ------------------- | --------------------- | --------------- |
| τ² airline     | gpt-oss-20b           | 200           | 182      | 81       | 97      | 159   | 195    | 7           | 0.55            | 0.005               | 6.0                   | 356             |
| τ² airline     | kimi-k2.5             | 200           | 308      | 142      | 166     | 371   | 504    | 13          | 0.58            | 0.005               | 11.0                  | 861             |
| τ² airline     | nemotron-3-super-120b | 200           | 406      | 75       | 330     | 419   | 618    | 15          | 0.60            | 0.015               | 14.0                  | 1038            |
| τ² airline     | qwen3-32b             | 200           | 748      | 297      | 438     | 282   | 619    | 23          | 0.69            | 0.015               | 12.0                  | 918             |
| τ² retail      | gpt-oss-20b           | 456           | 797      | 338      | 458     | 695   | 996    | 18          | 0.59            | 0.013               | 12.0                  | 1689            |
| τ² retail      | kimi-k2.5             | 456           | 460      | 245      | 214     | 1009  | 897    | 31          | 0.47            | 0.011               | 12.0                  | 1887            |
| τ² retail      | nemotron-3-super-120b | 456           | 822      | 251      | 569     | 879   | 1203   | 33          | 0.58            | 0.014               | 13.0                  | 2127            |
| τ² retail      | qwen3-32b             | 456           | 1696     | 741      | 955     | 803   | 1285   | 48          | 0.62            | 0.013               | 13.0                  | 2128            |
| τ² telecom     | gpt-oss-20b           | 456           | 546      | 291      | 252     | 362   | 1723   | 10          | 0.83            | 0.015               | 15.0                  | 2082            |
| τ² telecom     | kimi-k2.5             | 456           | 886      | 435      | 448     | 683   | 1500   | 34          | 0.69            | 0.018               | 14.0                  | 2180            |
| τ² telecom     | nemotron-3-super-120b | 456           | 1086     | 342      | 740     | 697   | 1693   | 36          | 0.71            | 0.022               | 15.0                  | 2401            |
| τ² telecom     | qwen3-32b             | 456           | 1902     | 837      | 1055    | 514   | 2231   | 66          | 0.81            | 0.042               | 18.0                  | 2788            |
| Terminal-Bench | gpt-oss-20b           | 40            | 53       | 45       | 6       | 56    | 55     | 0           | 0.50            | 0.414               | 13.5                  | 110             |
| Terminal-Bench | kimi-k2.5             | 40            | 57       | 56       | 1       | 93    | 73     | 0           | 0.44            | 0.056               | 24.5                  | 161             |
| Terminal-Bench | nemotron-3-super-120b | 40            | 98       | 95       | 0       | 146   | 106    | 1           | 0.42            | 0.167               | 27.0                  | 246             |
| Terminal-Bench | qwen3-32b             | 40            | 62       | 51       | 4       | 15    | 58     | 0           | 0.79            | 0.432               | 7.5                   | 73              |




## Table 7. Cost and overhead

Tokens/turn is total task-harness input plus output tokens divided by task-harness turns. Turns and wall time are per trial. Task cost USD is task-harness model spend; repair USD adds repair-model usage. It does not include infrastructure charges or τ² simulated-user cost. The total line aggregates all 16 matched pairs, and its token ratio is the unweighted mean of their ratios.


| Benchmark               | Model                 | Tokens/turn B→H | Ratio      | Turns B→H | Δ turns | Task cost USD B→H | Repair USD | Wall/trial B→H (s) |
| ----------------------- | --------------------- | --------------- | ---------- | --------- | ------- | ----------------- | ---------- | ------------------ |
| τ² airline              | gpt-oss-20b           | 4716→4145       | 0.88×      | 10.4→7.7  | -2.7    | 0.86→0.57         | 0.95       | 90→171             |
| τ² airline              | kimi-k2.5             | 6472→6436       | 0.99×      | 15.8→15.8 | +0.0    | 13.02→12.92       | 39.80      | 52→461             |
| τ² airline              | nemotron-3-super-120b | 10880→11369     | 1.04×      | 19.1→18.8 | -0.3    | 6.68→6.86         | 24.16      | 68→494             |
| τ² airline              | qwen3-32b             | 7201→7851       | 1.09×      | 16.4→17.2 | +0.7    | 3.65→4.16         | 4.39       | 36→366             |
| τ² retail               | gpt-oss-20b           | 5518→5743       | 1.04×      | 15.3→15.5 | +0.2    | 3.00→3.16         | 4.57       | 75→343             |
| τ² retail               | kimi-k2.5             | 6156→6256       | 1.02×      | 16.8→15.7 | -1.1    | 29.94→28.48       | 131.79     | 59→479             |
| τ² retail               | nemotron-3-super-120b | 9840→10261      | 1.04×      | 17.0→17.0 | +0.0    | 12.21→12.69       | 32.39      | 53→411             |
| τ² retail               | qwen3-32b             | 7135→7655       | 1.07×      | 16.8→17.4 | +0.6    | 8.42→9.31         | 9.70       | 37→479             |
| τ² telecom              | gpt-oss-20b           | 8475→8840       | 1.04×      | 18.2→21.2 | +2.9    | 5.41→6.48         | 4.91       | 105→517            |
| τ² telecom              | kimi-k2.5             | 9013→9240       | 1.03×      | 19.4→19.1 | -0.3    | 50.39→50.89       | 60.20      | 104→807            |
| τ² telecom              | nemotron-3-super-120b | 13575→14186     | 1.05×      | 24.0→24.0 | +0.0    | 23.89→24.78       | 32.11      | 115→825            |
| τ² telecom              | qwen3-32b             | 8850→9268       | 1.05×      | 24.0→20.5 | -3.5    | 14.87→13.33       | 13.37      | 74→680             |
| Terminal-Bench          | gpt-oss-20b           | 7408→7584       | 1.02×      | 16.5→9.1  | -7.4    | 0.43→0.23         | 0.25       | 235→521            |
| Terminal-Bench          | kimi-k2.5             | 15240→12860     | 0.84×      | 37.9→9.1  | -28.8   | 14.49→2.98        | 4.52       | 376→704            |
| Terminal-Bench          | nemotron-3-super-120b | 20202→14943     | 0.74×      | 47.9→6.8  | -41.0   | 5.95→0.63         | 4.23       | 302→811            |
| Terminal-Bench          | qwen3-32b             | 7950→6489       | 0.82×      | 12.9→8.0  | -4.9    | 0.63→0.32         | 0.36       | 113→238            |
| Matched-pair total (16) |                       | —               | 0.99× mean | —         | —       | 193.84→177.79     | 367.70     | —                  |
