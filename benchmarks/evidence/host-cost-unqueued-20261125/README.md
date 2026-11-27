# Unqueued host-cost entries, B300 (2026-11-25)

Source of the two `host-cost-unqueued` entries published to `ai-dynamo/aisim-data` at
`e81092f6d675cc20152707bc8da1f2c04a053707` (the revision in `aisim_core/data/host_cost_pin.json`). Host names, job
IDs and the raw SGLang seed-42 inputs are in `aisim-data-internal` under
`evidence/data/host-cost/<framework>/<version>/b300_sxm/<tokenizer>/unqueued-chat/`.

| Entry | Runs | Outside engine (mean, stored) | Tokenizer entry | Unqueued (replay) |
|---|---|---|---|---|
| SGLang 0.5.20, GLM-5.3-NVFP4, chat | seed 42 (exact config), seed 7 | 12.58 ms + 1.559 µs/token | SGLang 0.5.21 `glm`, 0.29 ms + 1.217 µs/token | 12.29 ms + 0.342 µs/token |
| vLLM 0.31.0, Qwen3.5-397B-A17B, chat | l1/l4 midpoint | 29.5 ms + 1.681 µs/token | vLLM 0.31.0 `qwen2`, 1.532 µs/token | 29.5 ms + 0.149 µs/token |

The entries store the outside-engine totals. Replay subtracts the paired tokenizer's clamped `base_ms` and slope from
them, which gives the last column; the two stages add back up to the stored total.

## SGLang

- Lane-1 AgentX captures on SGLang 0.5.20, B300 TP8, MTP on; not the scored E2E points (out of sample).
- `measure.py` (the script used, verbatim) on each run: client TTFT from the AIPerf export minus the engine
  prefill span from the FPM step hooks, fitted against total input tokens. The skill copy is
  `.agents/skills/aisim-cpu-overhead-collect/scripts/measure_unqueued.py`.
- Seed 42: 163 requests, 19.46 ms + 1.530 µs/token, R² 0.81 (`sglang-glm-seed42.txt`, `.json`).
- Seed 7: 166 requests, 5.7 ms + 1.587 µs/token, R² 0.92. Its step file was not retained; the fit is recorded as
  the measurement printed it.
- After the engine is about 2 ms; almost all of it is before the engine.

## vLLM

- Client TTFT minus engine time on the AgentX Weka l1 and l4 E2E runs: 25–34 ms + 1.65–1.72 µs/token in total
  (`benchmarks/evidence/vllm-tokenization-host-cpu-20261116/`, commit `0e87f7b35`). The per-run pairs were not
  retained, so one row holds the midpoint: 29.5 ms + 1.681 µs/token (1.436 + 0.245, the total replay was validated
  with). In sample for l1 and l4.
- That evidence split the total against a 1.436 µs/token tokenizer slope from a single-pass benchmark. The
  published tokenizer entry (3 repeats, 2 merged runs) fits 1.532 µs/token, and replay pairs the unqueued term with
  it, so the unqueued per-token term is 0.149 µs/token and the total is unchanged.

## Reproduce

```bash
python benchmarks/evidence/host-cost-unqueued-20261125/publish.py <copy of aisim-data with catalog/ and data/host-cost/>
```
