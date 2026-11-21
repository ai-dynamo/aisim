# AgentX replay on Qwen

This benchmark measures CPU time for one warm native replay call. It includes request admission, scheduling, model timing, and report generation.

Use the first 32 complete plays that fit the 262,144-token context limit, including all child requests. The sample contains 2,065 requests. Each run replays a 120-second window with seed 42. The number of completed requests depends on the model and lane count.

Download the pinned [AgentX trace](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/blob/23f152f6f0f9399a85901b89a6458def0ef16729/traces.jsonl). Use the same sample and CPU affinity for both builds.

```sh
python benchmark.py select traces.jsonl agentx-32.jsonl
sha256sum agentx-32.jsonl
# 32f48684cd46688f082b4379d22b28b29cfdc2718536b680bcae70a72a0ac60b
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
python benchmark.py run agentx-32.jsonl --model coder --lanes 1 --seconds 120
python benchmark.py run agentx-32.jsonl --model coder --lanes 16 --seconds 120
python benchmark.py run agentx-32.jsonl --model hybrid --lanes 16 --seconds 120
```

Run the commands with each checkout's release build of AISim. Use baseline, candidate, candidate, baseline order. Each process loads the graph on its first run and measures three warm runs. Compare medians across the six warm runs per revision.

The WEKA trace uses 64-token blocks. Both models use 16-token KV blocks, the vLLM scheduler, and analytical op-level timing on modeled GB200 GPUs. Qwen3-Coder-480B-A35B-Instruct uses TP8. Qwen3.5-27B uses TP4 and a 16-token prefix match unit. Its 16 attention layers use 16,384 KV bytes per token per rank. Its 48 recurrent layers use 38,486,016 state bytes per request per rank. These sizes follow the [Qwen3.5 checkpoint](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json) and [vLLM's state layout](https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/model_executor/layers/mamba/mamba_utils.py#L213).

Each output line includes a SHA-256 digest of the replay report. Compare digests between builds for each case. The digest includes request records, cache reuse, simulated times, and AgentX lifecycle events. It excludes host wall time and the two rates that use host wall time.
