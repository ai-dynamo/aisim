# Request path of each framework (what the harness times)

Audited 2026-11-18 from the installed source in the pinned images, on a CPU node with
`collector/host_cpu/audit.py` (read-only, no GPU). Line numbers are for these versions only.

| Framework | Image | Version |
|---|---|---|
| vLLM | `vllm-0310-cu130-a4a4c043` | 0.31.0 |
| SGLang | `sglang-0521-2dbe4c7f` | 0.5.21 |
| TRT-LLM | `trtllm-130rc29-e9dab9fc` | 1.3.0rc29 |

Requests are chat completions. Every adapter times the same normalized chat messages (tool calls folded
to text, same-role messages merged) so tokenizers see the same text. Each step is a median of 3 passes.

## vLLM 0.31.0

- `renderers/base.py:1181,1206`: `arrival_time = time.time()` is stamped **before** chat-template rendering and
  tokenization, so both fall inside the server's TTFT. Rendering is `render_chat` (`renderers/base.py:1225`).
- `config/model.py:378`: `--renderer-num-workers` defaults to 1.
- `v1/core/kv_cache_utils.py:684`: `hash_block_tokens` (prefix-cache block hashing, 16-token blocks) is a
  pre-scheduling cost; the harness times it as `hash`.
- Timed: `json`, `validate` (`ChatCompletionRequest`), `template` (`apply_chat_template`), `encode`, `hash`.

## SGLang 0.5.21

- `entrypoints/openai/serving_base.py:78`: `received_time = monotonic_time()` at the start of `handle_request`,
  after FastAPI has parsed and validated the body, so `json` and `validate` are outside SGLang's own TTFT clock
  but are still host time before the engine.
- `entrypoints/openai/serving_chat.py:1211`: `_process_messages` renders the template, then
  `_render_and_encode_chat_template` (`:1742`) calls `tokenizer.apply_chat_template(tokenize=False)` and
  `tokenizer.encode(rendered, add_special_tokens=False when the tokenizer auto-adds specials)` (`:1789-1799`).
- `use_cache=is_multimodal` (`:1688`): the template cache is off for text-only models, so the harness times
  the cold path every request takes.
- `managers/tokenizer_manager.py:1048`: `_tokenize_one_request` does not tokenize again when `input_ids` are present.
- `--tokenizer-worker-num` defaults to 1 (`arg_groups/fields/serving.py:50`).
- No prompt block hashing happens here; the radix-cache match runs in the scheduler.
- Timed: `json`, `validate` (`ChatCompletionRequest`), `template`, `encode`.

## TRT-LLM 1.3.0rc29

- `serve/openai_server.py:1981`: `openai_chat` renders the template with `async_apply_chat_template`
  (`inputs/utils.py:819`), which runs `apply_chat_template` (`inputs/utils.py:728`) on a worker thread
  (`asyncio.to_thread`). The thread hop is not timed.
- `inputs/registry.py:157`: `DefaultInputProcessor.__call__` encodes with
  `tokenizer.encode(prompt, add_special_tokens=sampling_params.add_special_tokens)` (`:212`). The tokenization
  prefix cache (`enable_tokenization_cache`) is off by default.
- `num_input_processor_workers` does **not** exist in this version, so the tokenizer worker count is not verified;
  entries record `tokenizer_workers: null` and replay uses 1.
- `ChatCompletionRequest` forbids unknown fields; the client's `return_token_ids` is dropped (recorded per entry).
- No prompt block hashing happens before the request reaches the executor.
- Timed: `json`, `validate` (`ChatCompletionRequest`), `template`, `encode`.

## What this does not cover

- Idle single-thread cost only; contention under load, the HTTP receive, streaming and detokenization are not measured.
- The fixed (unqueued) overhead is a separate collection. `json` and `validate` are recorded but not fitted.
