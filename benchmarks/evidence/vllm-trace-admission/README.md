# Kimi K3 admission CPU benchmark

This benchmark measures the CPU time that AISim spends replaying AgentX on `moonshotai/Kimi-K3`. It compares two release builds and checks the simulated report after every pair.

## Workload

Use the first 32 complete plays whose requests fit 262,144 tokens, including child requests. The resulting graph contains 2,065 requests. The benchmark keeps that context limit and uses 120-second windows, seed 42, and the recorded token counts.

Download the pinned [AgentX trace](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/blob/23f152f6f0f9399a85901b89a6458def0ef16729/traces.jsonl). Generate the sample once and use the same file for both builds.

```sh
python benchmark.py select traces.jsonl agentx-32.jsonl
sha256sum agentx-32.jsonl
# 32f48684cd46688f082b4379d22b28b29cfdc2718536b680bcae70a72a0ac60b
```

## Replay configuration

AISim uses its packaged Kimi K3 checkpoint configuration, the vLLM scheduler, and analytical op-level GB200 timing. Tensor parallelism spans 16 GPUs. The replay models text generation with one replica, no speculative decoding, and a fixed pool of 4,096 physical KV blocks per rank.

The K3 cache model resolves 27,648 token-KV bytes per token per rank. One padded KDA (Kimi Delta Attention) recurrent state occupies 30,523,392 bytes per request per rank. A physical KV block holds 384 tokens. The 384-token prefix unit reuses its physical block hash. With a 64-token prefix unit, the engine computes a separate hash chain. The source trace uses 64-token blocks in both cases. [K3 cache sizing](../../../python/aisim/src/aisim_core/sdk/memory/kimi_k3.py)

## Reproduction

Build both revisions in release mode with the same compiler flags. Set `BASE_PYTHON` and `PR_PYTHON` to the respective Python executables. Choose one logical CPU available to both processes.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
for case in 1-384 16-384 16-64; do
    taskset -c "$CPU" python benchmark.py compare agentx-32.jsonl \
        --baseline-python "$BASE_PYTHON" --candidate-python "$PR_PYTHON" \
        --lanes "${case%-*}" --prefix-unit "${case#*-}" --seconds 120 --repeats 12
done
```

The comparison keeps each revision in a separate process. Each process warms its trace graph once. Each subsequent pair runs sequentially on the selected CPU. The first revision alternates between pairs. Timing covers the native replay call, including report generation.

Every pair must match a SHA-256 digest of the replay report. The digest includes per-request records, cache hits, simulated timestamps, and AgentX lifecycle events. It excludes host wall time and the two rates that use host wall time.
