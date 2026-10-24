# Ascend 910C dynamic EPLB: delivery self-test report

Final packaged single-node and disaggregated P/D switch acceptance passed on 2026-10-24. Production code commit: `3683421a24e3917332923ddbab37490f849e7f49`.

## Changes and correctness baseline

The NPU adaptation was committed independently on `feat/dev_ep`, using planner and migration scheduling from fixed community commit `611a70306197239593959cb47a9fa92a9913dd18`. It adds valid sender TopK counts, padding-statistics masks, draft/target isolation, an independent HCCL migration group, ND staging for W8A8/NZ tensors, and exact batched statistics across graph-capture boundaries. Eligible NPU single-node trivial/static initialization uses a 64-step first learning period only; subsequent intervals and windows are unchanged.

`static` describes use of the dispatch map; dynamic EPLB still updates the map when ON. Baselines started in fresh processes with trivial placement and no offline optimized layout. The next OFF process did not inherit ON's learned placement. Redundant-expert budgets matched between arms.

All 39 CPU cases passed. NPU graph and migration evidence is scoped in the scenario screenshot. The final packaged model passed 16 original AIS inputs with fixed 64-token output, 16 successes/0 failures, SOURCE 16/16 and real rebalance completions on every rank. The 10 production files' executable ASTs matched the measured packaged version after formatting; source/archive hashes are recorded in [SOURCE_BINDING.json](ascend_dynamic_eplb_delivery_20261024/SOURCE_BINDING.json). AST equality is not additional GPU hardware or general-accuracy validation.

Applicable pre-commit, isort, ruff and codespell checks passed. Three system hooks depending on a python3 alias unavailable on Windows were run with the actual Python interpreter and unchanged scripts; all exited 0. The later CI follow-up only synchronizes the target branch, fixes full-tree import/format issues and translates publication artifacts; it does not establish new hardware measurements.

## Completed single-node performance confirmation

B103 / CANN 9.1.0 / PyTorch 2.10.0 / torch_npu 2.10.0.post6, DeepSeek-V4-Flash W8A8. Sixteen logical devices, TP/EP/attention-DP=16 and DP1. Original AIS fixed-copy workload: N320, concurrency cap 128, target 6 req/s, actual input 16388/output 1024 tokens, cache hit 0%. DSpark/Compact disabled. Initial learning 64, steady interval 640, statistics window 320, redundant experts 16.

Order: first ON, fresh OFF, fresh ON. Every run used a fresh process, matching source/environment and the same 16x64 preparation requests; only `--enable-eplb` differed. ON learns during preparation, so placement at the formal start is not guaranteed to match OFF.

| Run | Output tokens/s | Mean TTFT (s) | Mean TPOT (ms) | Mean E2E (s) | Success / failed |
|---|---:|---:|---:|---:|---:|
| First ON | 1397.65 | 19.79 | 55.10 | 76.19 | 320 / 0 |
| Fresh OFF | 1311.01 | 25.62 | 55.80 | 82.67 | 320 / 0 |
| Fresh ON | 1385.85 | 20.45 | 55.20 | 76.90 | 320 / 0 |

The ON arms improved throughput 6.61%/5.71% versus fresh OFF. Latest TTFT/TPOT/E2E changed -20.19%/-1.08%/-6.98%. All 320 input/output triplets matched exactly, with SOURCE 320/320 per run. Preparation plus formal time decreased 3.17%, excluding model loading, AIS initialization, orchestration and collection; this is not a full lifecycle benefit.

Performance belongs to the frozen `singlecapture_v1` candidate. Later package naming, logging and formatting changes have equivalence evidence and the full-model functional validation above, but were not rebenchmarked and do not produce a new speedup claim. Two ON runs and one OFF reproduce one fixed workload; they do not provide a confidence interval or a general workload guarantee.

![Single-node performance, tails and preparation cost](ascend_dynamic_eplb_delivery_20261024/performance.png)

## Test scenarios and screenshots

![Final-package self-test scenarios and results](ascend_dynamic_eplb_delivery_20261024/delivery_checks.png)

Both images are browser-rendered English report pages generated from the preserved measured results, rather than terminal screenshots. Translation does not change measurements.

PD acceptance used 16 ranks per side, matching OFF/ON configuration, trivial initialization and disabled radix cache/DSpark/Compact. P/D intervals were 4/64 to expose real migrations. Each arm used two 16-request preparations with output 16, then 64 formal requests at C16 and output 128. Inputs were 2048-character SOURCE prefixes, with actual token lengths taken from responses. This short functional test is separate from the 16K AIS benchmark above.

Actual formal input was 689-708 tokens with fixed output 128. Runtime receipts matched source manifest, entry, image, relevant environment and arguments; only `--enable-eplb` differed. ON was a new process that reloaded weights and started from trivial placement. `server_info` verified the actual switch on all 16 P/D ranks.

| Switch | Preparation success | Formal success / failed | SOURCE | P updates per rank | D updates per rank | P/D nonzero placement records |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 16 + 16 | 64 / 0 | 64/64 | 0 | 0 | 0 / 0 |
| ON | 16 + 16 | 64 / 0 | 64/64 | 4 | 10 | 4 / 10 |

Counts cover the request window including both preparations. Second-resolution timestamps use a 1-second start/3-second end tolerance; whole-service startup or health-probe updates are not substituted for request-window counts. All 64 formal request/output pairs matched exactly and passed the independent SOURCE-prefix oracle. All 192 planned requests succeeded. Successful preparation/formal logs had no ERROR, Traceback or scheduler exception. Placement-change records are rank-0 log entries, rather than all-weight equality or utilization proof. See [PD_ACCEPTANCE.json](ascend_dynamic_eplb_delivery_20261024/PD_ACCEPTANCE.json) for archive hashes and per-rank counts.

Preparation plus formal PD time was 31.867 s OFF/36.623 s ON. P4/D64 deliberately exercises frequent migration; these times are retained, not presented as steady PD net benefit or combined with single-node AIS results. Stable PD performance benefit for this optimized version has not been established.

The `feat/dev_ep` target advanced to `8e69805c` after hardware acceptance. A clean merge preview reran the same 39 CPU cases in an isolated source copy without NPU initialization or hot-service changes. This does not make the target's later changes hardware-validated; see [BASE_COMPATIBILITY.json](ascend_dynamic_eplb_delivery_20261024/BASE_COMPATIBILITY.json).

## Startup and collection fixes

D's first OFF startup failed the KV budget while another compute container occupied memory. The second attempt increased the memory fraction and passed that check, but found the default communication port occupied. After clearing the competing container and restoring 0.71/default port, the third startup succeeded. A terminated HTTP child was automatically replaced; the original service was fully ready when connectivity returned, so the model was not reloaded for that event. These failures preceded PD client requests; original logs are preserved.

The OFF-to-ON entry was initially rejected by a renamed router process and an old candidate-path guard on D. Neither rejection stopped the original model. After checking PID, start time, executable and listening port, the orchestration entry was fixed while retaining identity guards, and switching succeeded. Model code was unchanged. The first sealed-log summary missed one line with a progress-dot prefix. Recounting the same raw log by time/TP rank established P4/D10 on every rank without new requests or replacing archives. A CPU-mock cleanup error was also fixed and the 39 cases passed in a fresh process. These events are not mixed into model performance results.

## Historical results and limits

The [pre-optimization PD report](ascend_dynamic_eplb_b103.md) preserves forward-order +7.39% and reverse-order -7.35% results. They belong to an older version; they are not pooled with the single-node optimization or substituted for current final-package PD acceptance.

Full task accuracy, GPU hardware regression, all-43-layer/all-rank weight export, actual token balancedness and hardware utilization in the main performance workload were not observed. SOURCE-copy equality is not full model accuracy. Speculation was disabled; AL, measured strict draft acceptance and gamma/verify/proposed/accepted denominators are NOT_APPLICABLE. Failures and negative results are retained. Functional acceptance is complete within the stated source scope; performance claims remain limited to the fixed single-node workload.
