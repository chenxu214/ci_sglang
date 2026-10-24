> Historical report: PD comparisons before the later single-node optimization. See the [delivery self-test report](ascend_dynamic_eplb_delivery_20261024.md) for subsequent optimization and final-package verification. Historical results, including regressions, remain below.

# Ascend 910C dynamic EPLB: B103 self-test report

Ascend 910C / DeepSeek-V4-Flash W8A8 / TP/EP/attention-DP=16 on each P/D side / 2026-10-23.

B103 / CANN 9.1.0 / torch_npu 2.10.0.post6; N320/C128/6 req/s; actual input 16388 and fixed output 1024 tokens, cache hit 0%. DSpark and Compact disabled. No offline optimized placement; trivial/static initialization; redundant-expert budgets P16/D0.

## Execution and measurement scope

The orders were R1 OFF to ON and R2 ON to OFF. Every arm started in a fresh process with trivial placement and ran the same two PD preparations. ON preparation could already change placement; identical placement at the formal start is not claimed. Actual argv differed only by `--enable-eplb`; environment, 7 functional source files and dependency entry matched.

The baseline used no offline optimized placement. `static` describes dispatch; dynamic EPLB updates its mapping table. Regular model graph settings matched: D used graph batch sizes 1/2/4/6/8/12/16; P graphs were disabled.

## Scenarios and results

| Scenario | Coverage | Result |
|---|---|---|
| CPU unit regression | 31 independent cases, including incremental revalidation | PASS; no NPU initialization |
| Two-rank CPU/Gloo | Actual W8A8 postprocessing, shared experts, 12 whole-layer updates | PASS; weights, quantization parameters and pointers checked |
| Two-rank NPU/HCCL | NZ/W8A8 without fused shared experts, 12 whole-layer updates | PASS after fix; initial failure retained |
| NPU online loop | Synthetic routes; 113 steps and 6 updates on each P/D side, two hotspot changes | Peak route count 82 to 43; 23 skips after balancing |
| Actual 16-rank communication | P normal: T4/8192; D low_latency: T1/16 | Dispatch/combine, shape and zero-value oracle PASS |
| HcPre / NPU graph | DeepEP loaded first; hidden size 4096, 1/16 tokens | Eager/capture/replay values identical |
| Two PD preparations | 2x16 per arm; 4 arms validated | 128 requests passed; 0 new P-side IPC entries per round |
| Formal PD workload | N320 per arm; 4 arms validated | 1280 successful / 0 failed; all SOURCE checks passed |

## Whole-request interval

| Order | EPLB | Success/failed | Output tokens/s | Mean TTFT (s) | Mean TPOT (ms) | Mean E2E (s) |
|---|---|---|---|---|---|---|
| R1 | OFF | 320/0 | 1632.85 | 28.91 | 37.3 | 67.08 |
| R1 | ON | 320/0 | 1753.57 | 20.04 | 38.6 | 59.55 |
| R2 | ON | 320/0 | 1795.66 | 19.44 | 39.0 | 59.34 |
| R2 | OFF | 320/0 | 1938.08 | 20.27 | 34.9 | 55.97 |

| Pair | Order | Throughput change | TTFT change | TPOT change | E2E change | Identical outputs |
|---|---|---|---|---|---|---|
| R1 | OFF to ON | +7.39% | -30.70% | +3.49% | -11.23% | 320/320 |
| R2 | ON to OFF | -7.35% | -4.08% | +11.75% | +6.01% | 320/320 |

The comparison isolates the EPLB switch on matching source/configuration. Whole-request timing includes statistics, planning, communication, migration and draining. Negative latency changes mean improvement; positive TPOT changes mean regression. Results do not establish benefits for other data or maximum capacity.

## Original AIS native steady state

| Arm | Native steady-state N | Mean TTFT (s) | Mean TPOT (ms) | Mean E2E (s) |
|---|---|---|---|---|
| R1_OFF | 192 | 21.99 | 33.7 | 56.44 |
| R1_ON | 190 | 16.48 | 38.2 | 55.61 |
| R2_ON | 192 | 14.01 | 39.0 | 53.91 |
| R2_OFF | 191 | 15.59 | 35.0 | 51.39 |

Each native steady-state N was selected by the original AIS algorithm. Whole-request conclusions are not replaced by these subsets. Native statistics reused existing artifacts without new inference.

## Screenshots

These images are browser-rendered English report pages derived from the archived results. They replace the earlier report-page images without changing measurements; they are not terminal screenshots.

![Historical PD performance and regressions](ascend_dynamic_eplb_b103/performance.png)

![Historical functional scenarios and results](ascend_dynamic_eplb_b103/scenarios.png)

## Environment and retained failures

Image tag `cann9.1.0-a3-B103`; digest `sha256:45519e8921b54ae6424922d9934b734f9f7d126c1dc19ff0c10951f9c122dbbd`. Python 3.12.13/aarch64, PyTorch 2.10.0+cpu, torch_npu 2.10.0.post6, CANN 9.1.0. Measured functional commit `2949a9e58b31eeed591589ecf609af777552f994`, based on `feat/dev_ep@d3c76503`. Subsequent formatting affected only 3 test files; executable ASTs and functional source bytes were unchanged.

Missing vendor OPP setup, role-specific DeepEP buffer settings and import ordering caused initial startup/small-probe failures. After correction, the image's DeepEP OPP was loaded before NPU initialization. Actual HcPre graph and 16-rank communication probes passed before the four formal arms in this report. Failure records are retained. Port/router admission and the local python3 entry problem are classified as orchestration/check failures, rather than model performance regressions.

Applicable local pre-commit formatting, isort, ruff and codespell checks passed. Three system hooks requiring a python3 executable were run with the actual installed Python interpreter and unchanged scripts; all exited 0.

## Interpretation limits

This is a functional/performance test on a fixed synthetic SOURCE-copy workload. AIS artifacts retained gsm8k filenames, but this is not GSM8K mathematical accuracy or a full accuracy benchmark. All 320 nonempty outputs per arm were exact prefixes of their independent SOURCE; fixed 1024-token output permits truncation.

DSpark/Compact were disabled; speculative integration in this image and cross-host EP migration were not covered. AL, measured strict draft acceptance and gamma/verify denominators are NOT_APPLICABLE. Hardware utilization and actual token balancedness were NOT_OBSERVED in the main performance run. Placement logs demonstrate updates but do not replace device profiling. TPOT regressions are retained.

Raw responses, logs, source hashes, whole/native statistics and independent output oracles are archived with per-file verification. This repository contains redacted reports and screenshots, without internal addresses, accounts, weight paths or large raw artifacts.

## Percentiles and conclusions

| Arm | Metric | Mean | p50 | p99 | Maximum |
|---|---|---|---|---|---|
| R1_OFF | TTFT | 28913.0 ms | 26767.7 ms | 50229.8 ms | 50793.7 ms |
| R1_OFF | TPOT | 37.3 ms | 35.4 ms | 49.3 ms | 49.3 ms |
| R1_OFF | E2EL | 67079.1 ms | 61820.5 ms | 90550.0 ms | 90798.7 ms |
| R1_OFF | ITL | 37.2 ms | 36.5 ms | 48.2 ms | 2867.3 ms |
| R1_ON | TTFT | 20035.6 ms | 19272.2 ms | 33892.4 ms | 34263.2 ms |
| R1_ON | TPOT | 38.6 ms | 38.6 ms | 41.7 ms | 41.7 ms |
| R1_ON | E2EL | 59546.7 ms | 59488.9 ms | 73563.4 ms | 74268.6 ms |
| R1_ON | ITL | 38.5 ms | 37.8 ms | 51.9 ms | 2037.1 ms |
| R2_ON | TTFT | 19442.5 ms | 17917.7 ms | 36685.3 ms | 37165.9 ms |
| R2_ON | TPOT | 39.0 ms | 39.1 ms | 40.9 ms | 41.1 ms |
| R2_ON | E2EL | 59340.5 ms | 57594.9 ms | 77289.1 ms | 77968.1 ms |
| R2_ON | ITL | 38.9 ms | 38.3 ms | 92.0 ms | 1941.4 ms |
| R2_OFF | TTFT | 20269.6 ms | 17892.5 ms | 37765.1 ms | 38573.0 ms |
| R2_OFF | TPOT | 34.9 ms | 35.0 ms | 37.4 ms | 37.5 ms |
| R2_OFF | E2EL | 55974.8 ms | 53768.6 ms | 74246.8 ms | 75039.6 ms |
| R2_OFF | ITL | 34.8 ms | 37.1 ms | 47.4 ms | 342.7 ms |

The original report does not provide p95; it is NOT_OBSERVED and is not inferred from other percentiles.

R1 throughput improved 7.39%, while R2 throughput regressed 7.35%. Stable net benefit was not established for this historical configuration. ON TPOT regressed in both rounds; R2 E2E also regressed. The first positive pair cannot stand alone. The first OFF was slower than the later OFF; cold compilation, scheduling and runtime state remain hypotheses, rather than established causes.

In the formal controller window, both ON arms completed P1/D13 rebalances per rank, consistently across 16 ranks. Rank-0 placement diffs were nonzero. Exact all-rank map equality was not collected. Second-resolution log timestamps are not device migration costs. OFF had no periodic rebalances on either side.

## Bounded manual reproduction

Run at the repository root in the same B103 NPU environment. These commands do not load the full model:

```bash
export PYTHONPATH=python:${PYTHONPATH}
python -m torch.distributed.run --standalone --nproc-per-node=2 test/manual/ep/test_eplb_weight_migration.py --device npu --weight-format nz --w8a8-postprocess --no-fused-shared --all-layers-per-update
python -m torch.distributed.run --standalone --nproc-per-node=2 test/manual/ep/test_eplb_online_loop.py --device npu --reuse-tp-group --rebalance-threshold 0.9
```

## Key service configuration

| Parameter | P | D |
|---|---|---|
| TP / EP / attention-DP / independent replicas | 16 / 16 / 16 / 1 | 16 / 16 / 16 / 1 |
| DeepEP / initial placement / dispatch | normal / trivial / static | low_latency / trivial / static |
| Redundant experts / static memory fraction | 16 / 0.62 | 0 / 0.71 |
| EPLB interval / statistics window | 48 / 48 | 320 / 320 |
| Layers per rebalance chunk / utilization threshold | 64 / 1.0 | 64 / 1.0 |
| Chunked-prefill CLI / actual per rank | 131072 / 8192 | default 8192 / 512 |
| Prefill max requests / regular graphs | 2 / disabled | default / full, bs=1,2,4,6,8,12,16 |
| HCCL_BUFFSIZE / DEEPEP_HCCL_BUFFSIZE (MB) | 200 / 2600 | 200 / 1200 |
| DeepEP tokens per round / long-sequence rounds | 8192 / 1 | 2048 / 8 |
| Max dispatch tokens per rank | 128 | 128 |
| Expert-balancedness reporting / placement logs | off / enabled | off / enabled |
| Quantization / KV dtype / page size | modelslim W8A8 / bfloat16 / 128 | modelslim W8A8 / bfloat16 / 128 |
| use_index_cache / index_topk_freq | true / 2 | true / 2 |

Placement logging was fixed in both arms; actual ON logging cost is included in the whole-request interval. CANN customize/custom_transformer vendor setup and the bundled DeepEP OPP must be ready before their first operator initialization. The entry only imports DeepEP early and executes the original SGLang launcher; it does not inject the old WOA/Compact/PD helpers.
