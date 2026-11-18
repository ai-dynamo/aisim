# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The accessors follow SGLang's argument, engine and tokenizer interfaces at v0.5.22 and v0.5.23.
# See THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90

"""SGLang accessors for the launcher of AISim's SGLang self-benchmark.

This module is the only place where the launcher's process calls SGLang. It parses and resolves SGLang's
server arguments, builds the in-process engine with another scheduler entry, sends one batch of prompts and
returns what each request reported, and reads the vocabulary, the child processes and the model's expert
keys. It also turns texts into token ids with the served tokenizer and names the class of that tokenizer. It
decides nothing about the run: it rejects no setup, plans no point and reads no log.

Public functions: ``sglang_version``, ``load_plugins``, ``parse_server_args``, ``resolve``, ``resolved_dict``,
``platform_facts``, ``model_expert_keys``, ``build_engine``, ``generate_batch``, ``vocabulary``,
``model_vocabulary_size``, ``encode_texts``, ``tokenizer_class``, ``child_pids`` and ``shutdown``.

It runs inside the SGLang pod, where AISim is not installed. At module level it imports the standard library
only. Every function imports SGLang inside its body, so the module imports on a machine without SGLang, torch
or numpy. It imports no other ``sglbench_*`` module and reads no environment variable.

Two releases
------------
``TESTED_VERSIONS`` names the two SGLang releases the accessors were read against. Every SGLang name used
here has the same module, the same spelling and the same meaning at both, so no function branches on the
version. The caller compares ``sglang_version()`` with ``TESTED_VERSIONS`` and refuses any other release.

Checked against
---------------
SGLang v0.5.22 (baa9befa381511cd4d56f83980951e5a45fffc6d) and
v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90) of ``sgl-project/sglang`` (Apache-2.0), files under
``python/sglang``, read in the source at both tags:

* ``__init__.py`` and ``version.py``: ``__version__``; ``Engine`` in ``__init__.py`` is a ``LazyImport``
  instance (``utils.py``), which is not a class;
* ``srt/plugins/__init__.py``, ``load_plugins``; ``launch_server.py`` calls it before
  ``prepare_server_args``;
* ``srt/server_args.py``: ``prepare_server_args``, ``ServerArgs.resolve_once``, ``ServerArgs.resolved_dict``
  and ``ServerArgs.__setattr__`` (a field cannot be assigned during or after resolution);
* ``srt/arg_groups/model_override_base.py``, ``resolving_view`` and ``ResolvingConfig``, imported through
  ``srt/arg_groups/overrides.py`` as ``launch_server.py`` (``run_server``) does;
* ``srt/runtime_context.py``, ``get_platform`` and ``PlatformContext``; ``srt/utils/common.py``, ``is_cuda``
  and ``is_hip``;
* ``srt/configs/model_config.py``, ``ModelConfig.from_server_args`` and ``ModelConfig.__init__``
  (``hf_config``, ``hf_text_config``, ``vocab_size``); ``srt/model_loader/utils.py``, ``_is_moe_model``;
  ``srt/models/step3_vl.py``, ``Step3TextMoEMLP``; ``srt/models/glm_image_vl.py``, ``GlmImageTextModel``
  (``embed_tokens``) and ``GlmImageForConditionalGeneration`` (the size of its output head);
* ``srt/entrypoints/engine.py``, ``Engine``: ``run_scheduler_process_func``, ``__init__``,
  ``_ensure_event_loop``, ``async_generate``, ``get_all_child_pids`` and ``shutdown``;
* ``srt/managers/io_struct.py``, ``GenerateReqInput._determine_batch_size`` and ``_normalize_rid``;
* ``srt/managers/tokenizer_manager.py``, ``TokenizerManager``: ``generate_request``,
  ``auto_create_handle_loop``, ``_handle_batch_request``, ``_send_batch_request``,
  ``_stream_batch_responses``, ``_stream_one_response``, ``_coalesce_streaming_chunks``,
  ``_handle_batch_output``, ``_handle_abort_req``, ``_release_req_states_on_failure``,
  ``_validate_one_request``, ``sigterm_watchdog``, ``init_model_config``,
  ``init_tokenizer_and_processor`` and ``_tokenize_texts`` (SGLang's own batch call of the tokenizer);
* ``srt/managers/schedule_batch.py``: the ``to_json`` forms of ``FINISH_LENGTH``, ``FINISH_MATCHED_TOKEN``
  and ``FINISH_ABORT``, and ``Req._check_token_based_finish``;
* ``srt/managers/scheduler_components/output_streamer.py``, ``_GenerationStreamAccumulator.accept``;
* ``srt/sampling/sampling_params.py``, ``SamplingParams`` and its ``__post_init__``;
* ``srt/utils/hf_transformers/common.py``, ``attach_additional_stop_token_ids``.

Seen on running engines
-----------------------
``benchmarks/evidence/collector/sglbench-engine-probe-20261113`` holds result files of engines that report
SGLang 0.5.22 and 0.5.23, and the probe driver saved with them. Two of its checks bear on this module:

* check_1: every scheduler process wrote a recorder header, one process at TP 1 and two at TP 2. The probe
  driver parses with ``prepare_server_args`` and builds its engine from a subclass of
  ``sglang.srt.entrypoints.engine.Engine`` with a ``staticmethod`` scheduler entry and ``server_args=``; only
  that entry installs the recorder that writes the header;
* check_6: streamed batch calls ran to their end. The probe driver sent them through ``Engine.generate`` and
  kept the last chunk of each ``index``; the result files hold only the timing of those runs.

Run on engines on 2026-11-16 through sglbench_launcher.py, with SGLang 0.5.22 and 0.5.23 on H200 GPUs, at
TP 1 and TP 2: ``load_plugins``, ``parse_server_args``, ``resolve``, ``resolved_dict``, ``platform_facts``,
``model_expert_keys``, ``build_engine``, the call through ``Engine.async_generate`` under ``asyncio.wait_for``,
``vocabulary`` and ``shutdown``, which raised nothing. Four launches ran to a complete result. The files of
those launches are not kept in this repository.

Run on engines on 2026-11-17 through sglbench_launcher.py, in text runs with SGLang 0.5.22 and 0.5.23 on H200
GPUs: ``encode_texts`` and ``tokenizer_class``, with the tokenizers of ``Qwen/Qwen1.5-MoE-A2.7B`` and
``Qwen/Qwen3-0.6B``. ``model_vocabulary_size`` ran in two of them, smoke runs of the first model with SGLang
0.5.23 at TP 2, whose text pools were built with the value it returned. The result file of the first does
not hold the value: the launcher began to write it afterwards, as the upper end of
``producer.content.pool.id_range``, wherever a result file has the ``pool`` record. The result file of the
second holds it. At 0.5.22 it rests on the source, where it reads the attribute that ``vocabulary`` read on
those engines.

Not run on any engine, and so resting on the source alone: an abort or a timeout of a call and
``child_pids``. The tokenizer interface (``vocab_size``, ``all_special_ids``,
``get_added_vocab`` and the batch call ``tokenizer(texts, add_special_tokens=False)``) belongs to the
transformers library, whose source was not read; SGLang's use of the three names and of the batch call was.

No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the header of this
file and recorded in THIRD_PARTY_NOTICES.md. For this module that is SGLang alone: the names it imports and
calls, the request fields it sets, the fields of a stream chunk it reads and the config attributes of
``EXPERT_KEYS``.
"""

from __future__ import annotations

import asyncio
import inspect

# The SGLang releases whose source the accessors were read against (the commits are in the module docstring).
# ``sglang.__version__`` of a build at either tag is expected to be one of these strings: the recorder headers
# of the engine probe (benchmarks/evidence/collector/sglbench-engine-probe-20261113, check_1) hold exactly
# "0.5.22" and "0.5.23". The string is generated at build time (``version.py``), so another build of the same
# source can report another string; the caller then refuses it.
TESTED_VERSIONS = ("0.5.22", "0.5.23")

# Every SGLang name this module reaches by an import or as an attribute of an imported class, as far as the
# name exists before an engine is built: (module, name, kind). sglbench_preflight.py imports each module and
# looks each name up, so an image whose SGLang lacks one is refused before the model is loaded. The kinds:
# "name" is a module-level name; "class" is a module-level name that must be a class, because ``build_engine``
# subclasses it; "attribute" is ``Owner.attribute``, an attribute of a module-level class.
# Each row was read in the source at both tags, in the files the module docstring lists. Two notes on them:
# - ``ServerArgs`` is a class statement with the two methods, and the module then rebinds the name to
#   ``msgspec.defstruct(...)`` built with the namespace of that class (srt/server_args.py), so both methods
#   are attributes of the class whose records ``prepare_server_args`` returns;
# - ``async_generate``, ``get_all_child_pids`` and ``shutdown`` are called on the engine object and are
#   methods of ``Engine`` itself.
# Not listed, because no class holds them:
# - ``is_cuda`` and ``is_hip`` of ``get_platform()``. ``PlatformContext.__getattr__`` answers both from a
#   table of probes (srt/runtime_context.py), and reading ``is_cuda`` calls ``torch.cuda.is_available()``
#   (srt/utils/common.py), which asks the device;
# - what an engine and a model config set on themselves while they are built (``loop``,
#   ``tokenizer_manager``, ``hf_config``, ``hf_text_config``, ``vocab_size``), the tokenizer's attributes and
#   its batch call, the request fields ``generate_batch`` sets and the chunk fields it reads. A change of one
#   of these shows at the call that uses it.
SGLANG_SEAM_NAMES = (
    ("sglang", "__version__", "name"),
    ("sglang.srt.plugins", "load_plugins", "name"),
    ("sglang.srt.server_args", "prepare_server_args", "name"),
    ("sglang.srt.server_args", "ServerArgs.resolve_once", "attribute"),
    ("sglang.srt.server_args", "ServerArgs.resolved_dict", "attribute"),
    ("sglang.srt.arg_groups.overrides", "resolving_view", "name"),
    ("sglang.srt.runtime_context", "get_platform", "name"),
    ("sglang.srt.configs.model_config", "ModelConfig.from_server_args", "attribute"),
    ("sglang.srt.entrypoints.engine", "Engine", "class"),
    ("sglang.srt.entrypoints.engine", "Engine.run_scheduler_process_func", "attribute"),
    ("sglang.srt.entrypoints.engine", "Engine.async_generate", "attribute"),
    ("sglang.srt.entrypoints.engine", "Engine.get_all_child_pids", "attribute"),
    ("sglang.srt.entrypoints.engine", "Engine.shutdown", "attribute"),
)

# Config attributes that say a model has expert layers. A truthy value of any of them on the top config or on
# the text config counts. Where SGLang reads them, at both tags:
# - ``num_local_experts``, ``num_experts``, ``n_routed_experts``, ``num_experts_per_tok`` and
#   ``moe_intermediate_size``: srt/model_loader/utils.py ``_is_moe_model``, SGLang's own test, on the text
#   config;
# - ``moe_num_experts``: srt/models/step3_vl.py ``Step3TextMoEMLP`` (declared in srt/configs/step3_vl.py
#   ``Step3TextConfig``).
# This test is stricter than SGLang's in two ways: it also looks at the top config, and any truthy value counts
# (``_is_moe_model`` wants a count above 1, or an intermediate size above 0). It does not use SGLang's test of
# the architecture name ("moe" or "mixtral"), and it cannot see a count that a model keeps in a nested config
# (srt/models/dbrx.py reads ``config.ffn_config.moe_num_experts``).
EXPERT_KEYS = (
    "num_local_experts",
    "num_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
)

# The two configs of SGLang's ``ModelConfig`` (srt/configs/model_config.py ``ModelConfig.__init__``):
# ``hf_config`` is the config of the checkpoint and ``hf_text_config`` its language-model part
# (srt/utils/hf_transformers/common.py ``get_hf_text_config``), the same object when the checkpoint has no
# such part.
_MODEL_CONFIG_PARTS = ("hf_config", "hf_text_config")

# Counts in the ``meta_info`` of a finished request. srt/managers/tokenizer_manager.py
# ``TokenizerManager._handle_batch_output`` sets all four on every chunk of a generation request:
# ``prompt_tokens`` (the length of the prompt), ``cached_tokens`` (prompt tokens found in the prefix cache),
# ``completion_tokens`` (output tokens so far, a running total and not a delta) and ``num_retractions`` (how
# often the scheduler pushed the request out and ran it again).
_COUNT_KEYS = ("prompt_tokens", "cached_tokens", "completion_tokens", "num_retractions")

# The two counts that the echo of an abort lacks. ``TokenizerManager._handle_abort_req`` builds that
# ``meta_info`` without them (it sets ``id``, ``finish_reason``, ``weight_version``, ``e2e_latency`` and
# ``completion_tokens``), so a final chunk without them is an abort whatever its finish reason says.
_ABORT_ECHO_ABSENT_KEYS = ("prompt_tokens", "cached_tokens")


class EngineCallError(RuntimeError):
    """One engine call did not give a complete answer for every request.

    Raised by ``generate_batch`` for a timeout, an aborted request, a stream that ended without the final
    chunk of a request, a chunk that cannot be placed, a missing key and any exception of the call. The call
    is never sent again.
    """


def _is_count(value) -> bool:
    """True for a plain int. A bool is not a count although Python makes it an int."""
    return isinstance(value, int) and not isinstance(value, bool)


def sglang_version() -> str:
    """The version string of the installed SGLang package.

    ``sglang.__version__`` (``__init__.py``, taken from ``version.py``). The string is fixed when the package
    is built, so it names a release only for a build made at that release's tag.
    """
    import sglang

    version = sglang.__version__
    if not isinstance(version, str):
        raise TypeError(f"sglang.__version__ is {version!r}, not a string")
    return version


def load_plugins() -> None:
    """Load SGLang's plugins, as SGLang's own launcher does before it parses the arguments.

    ``sglang.srt.plugins.load_plugins`` (srt/plugins/__init__.py). ``launch_server.py`` calls it before
    ``prepare_server_args`` and ``Engine.__init__`` calls it first. Resolution steps run through hooks that a
    plugin may replace, so without this call the launcher could resolve other values than the engine. The
    function is idempotent in SGLang.
    """
    from sglang.srt.plugins import load_plugins as sglang_load_plugins

    sglang_load_plugins()


def parse_server_args(argv: list[str]):
    """Parse SGLang's own flags into an unresolved ``ServerArgs`` record.

    ``sglang.srt.server_args.prepare_server_args`` (srt/server_args.py), the parser of ``sglang serve``. It
    parses with ``parse_args``: an unknown flag or a missing ``--model-path`` prints the usage and raises
    ``SystemExit(2)``, which passes through. It also resets the root logging handlers
    (``logging.basicConfig(..., force=True)``). The record it returns holds the operator's input only;
    ``resolve`` fills in what SGLang decides.
    """
    from sglang.srt.server_args import prepare_server_args

    return prepare_server_args(list(argv))


def resolve(server_args):
    """Resolve the record once and return the live view of the resolved values.

    ``ServerArgs.resolve_once`` (srt/server_args.py) runs SGLang's resolution pipeline in this process;
    ``launch_server.py`` (``run_server``) does the same before it starts anything, and the engine's own call
    is then a no-op. Resolution never writes a field: ``server_args.page_size`` stays the operator's input
    (``None`` unless the flag was passed). The resolved value is ``resolving_view(server_args).page_size``
    (srt/arg_groups/model_override_base.py ``resolving_view``, imported through srt/arg_groups/overrides.py as
    ``launch_server.py`` does). The view raises ``AttributeError`` for a name that is not a field.

    Resolution probes the device, reads the model config and sets some of SGLang's environment variables
    (srt/arg_groups/serving_hook.py ``handle_environment_variables``). For a setup SGLang refuses its hooks
    raise ``ValueError``, ``AssertionError``, ``NotImplementedError`` or ``RuntimeError``, and a record whose
    resolution failed cannot be resolved again. More checks (``ServerArgs.check_server_args``) run only when
    the engine is built. After resolution the record is sealed: an assignment to a field raises
    ``AttributeError`` (``ServerArgs.__setattr__``).
    """
    from sglang.srt.arg_groups.overrides import resolving_view

    server_args.resolve_once()
    return resolving_view(server_args)


def resolved_dict(server_args) -> dict:
    """Every field of the record with its resolved value, nested records as dicts.

    ``ServerArgs.resolved_dict`` (srt/server_args.py). Values are not all JSON: SGLang leaves callables and
    unknown objects as they are, so a writer needs a fallback encoder.
    """
    resolved = server_args.resolved_dict()
    if not isinstance(resolved, dict):
        raise TypeError(f"ServerArgs.resolved_dict() returned {type(resolved).__name__}, not a dict")
    return resolved


def platform_facts() -> dict:
    """Whether this process sees an NVIDIA CUDA build of torch: ``{"is_cuda": bool, "is_hip": bool}``.

    ``sglang.srt.runtime_context.get_platform()`` (srt/runtime_context.py ``PlatformContext``), whose
    ``is_cuda`` and ``is_hip`` call the probes of the same names in srt/utils/common.py: ``is_cuda`` is true
    when torch has CUDA and a device is available, ``is_hip`` when torch is a ROCm build. A ROCm build also
    reports the device name "cuda", so the device name alone does not tell the two apart.
    """
    from sglang.srt.runtime_context import get_platform

    platform = get_platform()
    facts = {"is_cuda": platform.is_cuda, "is_hip": platform.is_hip}
    for name, value in facts.items():
        if not isinstance(value, bool):
            raise TypeError(f"get_platform().{name} is {value!r}, not a bool")
    return facts


def model_expert_keys(server_args) -> dict:
    """The truthy expert keys of the model config, ``{}`` for a dense model.

    Builds ``ModelConfig.from_server_args(server_args)`` (srt/configs/model_config.py), the call the tokenizer
    manager makes for the same record (``TokenizerManager.init_model_config``). It loads the config of the
    checkpoint and never the weights; it may download the config. It reads the record through SGLang's
    resolved view, so call it after ``resolve``: before that it would see the model path and revision as the
    operator typed them. Each key of ``EXPERT_KEYS`` is looked up on ``hf_config`` and then on
    ``hf_text_config``; the result maps a key to its first truthy value, in the order of ``EXPERT_KEYS``. A
    key that a config does not have is not an error: these are optional attributes of a Hugging Face config.
    """
    from sglang.srt.configs.model_config import ModelConfig

    model_config = ModelConfig.from_server_args(server_args)
    configs = []
    for part in _MODEL_CONFIG_PARTS:
        config = getattr(model_config, part)
        if config is None:
            raise ValueError(f"ModelConfig.{part} is None; the expert keys cannot be read")
        configs.append(config)
    found: dict = {}
    for key in EXPERT_KEYS:
        for config in configs:
            value = getattr(config, key, None)
            if value:
                found[key] = value
                break
    return found


def build_engine(server_args, scheduler_entry):
    """Build SGLang's in-process engine with ``scheduler_entry`` as the entry of every scheduler process.

    The engine is a subclass of the real class ``sglang.srt.entrypoints.engine.Engine``. ``sglang.Engine`` is
    a ``LazyImport`` instance (``__init__.py``, ``utils.py``) and cannot be a base class. The subclass sets
    the class attribute ``run_scheduler_process_func``, the seam ``Engine.__init__`` reads and passes to
    ``_launch_subprocesses``. It must be a ``staticmethod``: a plain function in a class body would be bound
    to the engine, and the engine would then have to be pickled for every spawned child. ``scheduler_entry``
    must be a module-level function of an importable file, because a spawned child finds it by name; it
    receives the seven or ten positional arguments SGLang gives its own entry.

    With ``server_args=`` the engine uses the record as given and ignores every other keyword. The record
    comes from SGLang's own parser, so the engine is built as ``sglang serve`` would build it apart from the
    entry, and the entry is expected to call the stock one.

    Call it outside any running event loop. ``Engine._ensure_event_loop`` would take a running loop as the
    engine's loop, and ``generate_batch`` could not run a call on it. The build blocks until every scheduler
    process is ready; it raises what SGLang raises for a setup it refuses.

    Seen on running engines: benchmarks/evidence/collector/sglbench-engine-probe-20261113 (check_1).
    """
    if not callable(scheduler_entry):
        raise TypeError(f"scheduler_entry is {scheduler_entry!r}, not a callable")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("build_engine was called inside a running event loop; build the engine outside any loop")

    from sglang.srt.entrypoints.engine import Engine

    if not isinstance(Engine, type):
        raise TypeError(f"sglang.srt.entrypoints.engine.Engine is {Engine!r}, not a class")
    if not hasattr(Engine, "run_scheduler_process_func"):
        raise AttributeError(
            "sglang.srt.entrypoints.engine.Engine has no run_scheduler_process_func; "
            "the scheduler entry cannot be replaced"
        )

    class SglbenchEngine(Engine):
        run_scheduler_process_func = staticmethod(scheduler_entry)

    return SglbenchEngine(server_args=server_args)


def _check_call_arguments(prompts, rids, max_new_tokens, timeout_seconds) -> None:
    """Refuse arguments that SGLang would read as another kind of call, before anything is sent."""
    if not isinstance(prompts, list) or not prompts:
        raise EngineCallError(f"prompts must be a non-empty list of token-id lists, got {type(prompts).__name__}")
    for position, prompt in enumerate(prompts):
        # A flat list of ints is ONE request for SGLang (srt/managers/io_struct.py
        # ``GenerateReqInput._determine_batch_size`` tests ``isinstance(input_ids[0], int)``), and the stream
        # of a single request has no ``index``.
        if not isinstance(prompt, list) or not prompt:
            raise EngineCallError(f"prompt {position} must be a non-empty list of token ids")
    if not isinstance(rids, list) or len(rids) != len(prompts):
        raise EngineCallError(f"rids must be a list of {len(prompts)} request ids, one per prompt")
    for position, rid in enumerate(rids):
        if not isinstance(rid, str) or not rid:
            raise EngineCallError(f"request id {position} is {rid!r}, not a non-empty string")
    if len(set(rids)) != len(rids):
        raise EngineCallError("request ids repeat inside one call")
    if not _is_count(max_new_tokens) or max_new_tokens < 1:
        raise EngineCallError(f"max_new_tokens is {max_new_tokens!r}, not an int of at least 1")
    # ``asyncio.wait_for`` waits without limit for ``None``; a call must always have a finite limit.
    is_number = isinstance(timeout_seconds, (int, float)) and not isinstance(timeout_seconds, bool)
    if not is_number or not 0 < timeout_seconds < float("inf"):
        raise EngineCallError(f"timeout_seconds is {timeout_seconds!r}, not a finite number above 0")


def _chunk_position(chunk, count: int) -> int:
    """The position of the request a stream chunk belongs to.

    ``TokenizerManager._stream_batch_responses`` sets ``index`` on every chunk of a batch stream, also for a
    batch of one, from the request id in ``meta_info``. Only the stream of a single request has none, so a
    chunk without it is placed when the call has one request and refused otherwise.
    """
    if not isinstance(chunk, dict):
        raise EngineCallError(f"a stream chunk is {type(chunk).__name__}, not a dict")
    if "index" not in chunk:
        if count == 1:
            return 0
        raise EngineCallError(f"a stream chunk of a call with {count} requests has no index")
    position = chunk["index"]
    if not _is_count(position) or not 0 <= position < count:
        raise EngineCallError(f"a stream chunk has index {position!r}; the call has {count} requests")
    return position


def _final_record(rid: str, meta_info: dict) -> dict:
    """The normalized answer of one finished request, from the ``meta_info`` of its last chunk."""
    finish_reason = meta_info["finish_reason"]
    # The scheduler sends ``finished_reason.to_json()`` (srt/managers/schedule_batch.py): ``{"type": "length",
    # "length": n}``, ``{"type": "stop", "matched": ...}`` or ``{"type": "abort", "message", "status_code",
    # "err_type"}``.
    if not isinstance(finish_reason, dict):
        raise EngineCallError(f"request {rid}: finish_reason is {finish_reason!r}, not a dict")
    finish_type = finish_reason.get("type")
    if finish_type == "abort":
        raise EngineCallError(
            f"request {rid} was aborted: {finish_reason.get('message')!r} "
            f"(status_code={finish_reason.get('status_code')!r})"
        )
    absent = [key for key in _ABORT_ECHO_ABSENT_KEYS if key not in meta_info]
    if absent:
        raise EngineCallError(f"request {rid} was aborted: its final meta_info has no {' and no '.join(absent)}")
    if finish_type is not None and not isinstance(finish_type, str):
        raise EngineCallError(f"request {rid}: finish_reason type is {finish_type!r}, not a string")
    for key in _COUNT_KEYS:
        if key not in meta_info:
            raise EngineCallError(f"request {rid}: its final meta_info has no {key}")
        if not _is_count(meta_info[key]) or meta_info[key] < 0:
            raise EngineCallError(f"request {rid}: {key} is {meta_info[key]!r}, not a count")
    return {
        "rid": rid,
        "prompt_tokens": meta_info["prompt_tokens"],
        "cached_tokens": meta_info["cached_tokens"],
        "completion_tokens": meta_info["completion_tokens"],
        "finish_type": finish_type,
        "num_retractions": meta_info["num_retractions"],
    }


async def _collect_stream(engine, prompts, rids, sampling_params) -> list[dict]:
    """Send one batch call on the running loop and return the normalized answer of every request.

    How a batch stream delivers its chunks (``TokenizerManager._stream_batch_responses``, the same code at
    both tags, and ``_stream_one_response``, which differs at v0.5.23 only in which customized-info keys it
    copies into a chunk):

    - chunks of different requests interleave; each carries ``index``, the position of its request in the
      call, and a ``meta_info`` built for that output message, whose ``id`` is the request id;
    - ``meta_info["finish_reason"]`` is ``None`` until the request is finished. The chunk that carries a
      finish reason is the last chunk of its request, also for an abort;
    - the number of chunks says nothing. Chunks that queued up are merged into one (incremental streaming,
      ``_coalesce_streaming_chunks``: token ids are joined, ``meta_info`` is the last one) or all but the last
      are dropped (cumulative streaming). So only the last chunk is read, and only its ``meta_info``, whose
      counts are totals in both modes.

    An error is raised at the chunk that shows it, without waiting for the other requests; the stream is then
    closed, which makes SGLang abort them.

    The first call of a process also starts SGLang's reader of scheduler output as a task on the running loop
    (``TokenizerManager.auto_create_handle_loop``). That is why every call must run on the engine's own loop.
    """
    count = len(rids)
    final: dict[int, dict] = {}
    stream = None
    try:
        # ``Engine.async_generate`` is a coroutine function; with ``stream=True`` (it tests ``stream is True``)
        # awaiting it returns the async generator ``TokenizerManager.generate_request``. Nothing is sent
        # before the first chunk is asked for. ``input_ids`` as a list of lists makes the call a batch, and
        # ``rid`` as a list of the same length gives each prompt its id (srt/managers/io_struct.py
        # ``GenerateReqInput._determine_batch_size`` and ``_normalize_rid``).
        stream = await engine.async_generate(input_ids=prompts, sampling_params=sampling_params, rid=rids, stream=True)
        async for chunk in stream:
            position = _chunk_position(chunk, count)
            rid = rids[position]
            meta_info = chunk.get("meta_info")
            if not isinstance(meta_info, dict):
                raise EngineCallError(f"request {rid}: a stream chunk has no meta_info dict")
            if meta_info.get("id") != rid:
                raise EngineCallError(f"request {rid}: a chunk at its index carries the id {meta_info.get('id')!r}")
            if position in final:
                raise EngineCallError(f"request {rid}: a chunk arrived after its final chunk")
            if "finish_reason" not in meta_info:
                raise EngineCallError(f"request {rid}: the meta_info of a chunk has no finish_reason")
            if meta_info["finish_reason"] is not None:
                final[position] = _final_record(rid, meta_info)
    except EngineCallError:
        raise
    except Exception as error:
        # A ``TimeoutError`` raised inside the call is turned into this error here, so the one that leaves
        # ``asyncio.wait_for`` can only be the limit of the call. A cancellation is not an ``Exception`` and
        # passes, as ``wait_for`` needs it to.
        raise EngineCallError(f"the engine call raised {type(error).__name__}: {error}") from error
    finally:
        # A stream that is left half read keeps its requests running. Closing it makes SGLang abort at
        # once what it had dispatched (``TokenizerManager.generate_request``,
        # ``_release_req_states_on_failure``) and not when the generator happens to be collected. It does
        # not close SGLang's inner stream (``_stream_batch_responses``): its waiting tasks, one per
        # unfinished request, stay pending on the loop until the loop runs again, and a failed call is
        # the last one of a run. A stream that ended, or that a cancellation went through, is closed
        # already and this does nothing.
        if stream is not None and hasattr(stream, "aclose"):
            try:
                await stream.aclose()
            except Exception as error:
                raise EngineCallError(f"closing the stream raised {type(error).__name__}: {error}") from error
    missing = [rids[position] for position in range(count) if position not in final]
    if missing:
        raise EngineCallError(
            f"the stream ended without the final chunk of {len(missing)} of {count} requests (first: {missing[0]})"
        )
    return [final[position] for position in range(count)]


# Deviation note (collector rule "measure what serving runs"): how a benchmark call differs from served
# requests. All four follow from the fields ``generate_batch`` sets; nothing else is set, so every other
# request field has SGLang's default.
# - One call carries every prompt of a stage. Served requests arrive one by one and the scheduler batches what
#   is waiting. Here ``input_ids`` is a list of prompts, which the tokenizer manager sends to the scheduler as
#   ONE message (srt/managers/tokenizer_manager.py ``TokenizerManager._handle_batch_request`` and
#   ``_send_batch_request``; srt/managers/scheduler.py ``Scheduler.handle_batch_generate_request``), so all
#   requests of a stage are waiting in the same scheduler pass. What differs is the arrival; the batch is then
#   admitted, built and run by the stock scheduler. Direction of the error: none known for the forward pass
#   of that batch. Seen on running engines: one call of 2 and one of 128 prompts each ran as one extend batch
#   (benchmarks/evidence/collector/sglbench-engine-probe-20261113, check_2; 128 prompts in nine of its ten
#   runs, 32 in the other).
#   Retire by: staging the requests of a batch through SGLang's serving frontend.
# - ``temperature`` 0.0. ``SamplingParams.__post_init__`` (srt/sampling/sampling_params.py) turns it into
#   greedy sampling (``top_k = 1``). A served request samples as its client asks. Direction of the error: not
#   measured; the model's forward pass is the same, the sampling step after it is the greedy one.
#   Retire by: sending the sampling settings of the served workload.
# - ``ignore_eos``. ``Req._check_token_based_finish`` (srt/managers/schedule_batch.py) then never ends a
#   request at a stop token, so it runs to exactly ``max_new_tokens`` and a decode run has a known number of
#   steps. A served request can end earlier. Direction of the error: none on a step that is timed; the run
#   length is fixed by it. Retire by: nothing to retire while a point is a fixed number of steps.
# - ``stream=True``. The scheduler then sends the output of every request after every step
#   (srt/managers/scheduler_components/output_streamer.py ``_GenerationStreamAccumulator.accept``, when the
#   server's ``stream_interval`` is 1, its default), so a timed decode step includes that per-step output
#   work. Without streaming it sends at the finish and every ``SGLANG_FORCE_STREAM_INTERVAL`` output tokens
#   (50 unless set). Served clients choose either. Direction of the error: streaming adds output work to a
#   step, but no increase was seen. On running engines the decode sample with streaming was 0.9626 to 1.0006
#   times the sample without it, at batch 8 and context 1024
#   (benchmarks/evidence/collector/sglbench-engine-probe-20261113, check_6; six runs, none with incremental
#   streaming output). Retire by: nothing to retire; streaming is the serving path of a streaming client.
def generate_batch(engine, prompts, rids, max_new_tokens, timeout_seconds) -> list[dict]:
    """Send ``prompts`` as one batch call and return one dict per request, in input order.

    ``prompts`` is a list of lists of token ids; a batch of one is ``[[...]]``. ``rids`` is a list of as many
    distinct request ids. Each returned dict is::

        {"rid": str, "prompt_tokens": int, "cached_tokens": int, "completion_tokens": int,
         "finish_type": str | None, "num_retractions": int}

    taken from the ``meta_info`` of the request's last stream chunk (``_collect_stream`` says how a batch
    stream delivers chunks). ``finish_type`` is ``finish_reason["type"]``: ``"length"`` for a request that
    produced ``max_new_tokens`` tokens, ``"stop"`` for one that ended earlier.

    The call is ``engine.async_generate(input_ids=prompts, sampling_params={"max_new_tokens": ...,
    "temperature": 0.0, "ignore_eos": True}, rid=rids, stream=True)`` (srt/entrypoints/engine.py
    ``Engine.async_generate``) and nothing else: no ``return_logprob``, ``logprob_start_len``, ``priority``,
    ``cache_salt``, ``n`` or ``stream_interval``. It runs as
    ``engine.loop.run_until_complete(asyncio.wait_for(..., timeout_seconds))``:

    - on ``engine.loop``, the loop the engine took at its build (``Engine._ensure_event_loop``). SGLang binds
      its reader of scheduler output to the loop of the first request. A call on any other loop, such as a
      new ``asyncio.run`` for each call, finds no reader and waits for ever;
    - under ``asyncio.wait_for``, because ``Engine.generate`` and the wait inside SGLang have no time limit
      for an engine request. At the limit the call is cancelled, and SGLang aborts the requests it had
      dispatched (``TokenizerManager.generate_request``).

    Raises ``EngineCallError`` for: the time limit; an aborted request (a finish reason of type ``"abort"``,
    or a final ``meta_info`` without ``prompt_tokens`` or ``cached_tokens``, which is the echo of an abort); a
    stream that ended without the final chunk of a request; a chunk without ``index`` when the call has more
    than one request, or with an index or a request id that does not belong to the call; a missing key; a
    cancellation; and any other exception of the call. The call is never sent again.

    ``SystemExit`` and ``KeyboardInterrupt`` pass through; the launcher's stop on a signal is a ``SystemExit``
    (sglbench_launcher.py ``_StopRequested``). SGLang's own SIGTERM handling ends with ``sys.exit(0)`` inside a
    task of the loop (``TokenizerManager.sigterm_watchdog``). By asyncio's rules that would leave the running
    call as ``SystemExit`` too; that path was read and not run.
    """
    _check_call_arguments(prompts, rids, max_new_tokens, timeout_seconds)
    # The fields of a request the benchmark sets. ``SamplingParams`` (srt/sampling/sampling_params.py) has
    # ``max_new_tokens``, ``temperature`` and ``ignore_eos``; the deviation note above says what each does.
    sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0.0, "ignore_eos": True}
    collect = _collect_stream(engine, prompts, rids, sampling_params)
    bounded = asyncio.wait_for(collect, timeout_seconds)
    try:
        return engine.loop.run_until_complete(bounded)
    except EngineCallError:
        raise
    except TimeoutError as error:
        # What ``asyncio.wait_for`` raises at its limit. ``asyncio.TimeoutError`` is this builtin class from
        # Python 3.11 on.
        raise EngineCallError(
            f"the engine call timed out after {timeout_seconds} s ({len(rids)} requests, first: {rids[0]})"
        ) from error
    except asyncio.CancelledError as error:
        raise EngineCallError(f"the engine call was cancelled ({len(rids)} requests, first: {rids[0]})") from error
    except Exception as error:
        raise EngineCallError(f"the engine call failed: {type(error).__name__}: {error}") from error
    finally:
        # A loop that is closed or already running refuses the call before it starts it. The two coroutine
        # objects were then never run; closing them here keeps Python from reporting them as never awaited.
        # A call that did start is left alone, also when it is still pending after a stop.
        for pending in (bounded, collect):
            if inspect.getcoroutinestate(pending) == inspect.CORO_CREATED:
                pending.close()


def _token_ids(values, source: str) -> set[int]:
    """The ids of one tokenizer attribute as a set; anything that is not a plain int is refused."""
    try:
        values = iter(values)
    except TypeError as error:
        raise TypeError(f"the tokenizer's {source} is {values!r}, not a collection of token ids") from error
    ids = set()
    for value in values:
        if not _is_count(value) or value < 0:
            raise TypeError(f"the tokenizer's {source} holds {value!r}, which is not a token id")
        ids.add(value)
    return ids


def _tokenizer_attribute(tokenizer, name: str):
    try:
        return getattr(tokenizer, name)
    except AttributeError as error:
        raise AttributeError(
            f"the tokenizer {type(tokenizer).__name__} has no {name}; the benchmark needs it to keep special "
            "ids out of the token ids it makes up"
        ) from error


def _served_tokenizer(engine):
    """The tokenizer the engine serves with; ``ValueError`` when SGLang built none."""
    tokenizer = engine.tokenizer_manager.tokenizer
    if tokenizer is None:
        raise ValueError("engine.tokenizer_manager.tokenizer is None; SGLang builds none with --skip-tokenizer-init")
    return tokenizer


def vocabulary(engine) -> dict:
    """The id range of the served vocabulary and the ids the benchmark must not make up.

    Returns ``{"size": int, "excluded": sorted list of ids}``. Ordinary token ids are those in ``[0, size)``
    that are not in ``excluded``. The seeded ids and the allocated first tokens of the prompts are ordinary
    ids. The ids of dataset text are not held to them: a text keeps what the tokenizer returned
    (``model_vocabulary_size``).

    The tokenizer is ``engine.tokenizer_manager.tokenizer`` (srt/managers/tokenizer_manager.py
    ``TokenizerManager.init_tokenizer_and_processor``), ``None`` with ``--skip-tokenizer-init``.

    - ``size`` is the smaller of ``tokenizer.vocab_size`` and
      ``engine.tokenizer_manager.model_config.vocab_size``. The second is the model's bound, the
      ``vocab_size`` of the text config (srt/configs/model_config.py ``ModelConfig``), and it can differ from
      the tokenizer's. SGLang does not check input ids against either on this path: the tokenizer manager
      checks the length of a prompt only (``TokenizerManager._validate_one_request``). What the model does
      with an id at or above its bound was not read.
    - ``excluded`` holds ``tokenizer.all_special_ids``, the ids of ``tokenizer.get_added_vocab()`` and
      ``tokenizer.additional_stop_token_ids`` when it is not ``None``. The last one is SGLang's own attribute
      (srt/utils/hf_transformers/common.py ``attach_additional_stop_token_ids``), a set of ids or ``None``;
      SGLang's scheduler reads it without a default (srt/managers/schedule_batch.py
      ``Req._check_token_based_finish``), so every tokenizer SGLang serves with has it. Excluded ids are not
      limited to ``[0, size)``: added tokens usually lie above the tokenizer's ``vocab_size``.

    Raises ``ValueError`` when there is no tokenizer or a size is not positive, ``AttributeError`` when the
    tokenizer lacks ``vocab_size``, ``all_special_ids``, ``get_added_vocab`` or
    ``additional_stop_token_ids`` (SGLang can also serve tokenizers that are not Hugging Face ones), and
    ``TypeError`` when a value is not what the names promise. No value is guessed.
    """
    manager = engine.tokenizer_manager
    tokenizer = _served_tokenizer(engine)
    sizes = {
        "tokenizer.vocab_size": _tokenizer_attribute(tokenizer, "vocab_size"),
        "tokenizer_manager.model_config.vocab_size": manager.model_config.vocab_size,
    }
    for name, value in sizes.items():
        if not _is_count(value):
            raise TypeError(f"{name} is {value!r}, not an int")
        if value <= 0:
            raise ValueError(f"{name} is {value}, not a positive size")
    excluded = _token_ids(_tokenizer_attribute(tokenizer, "all_special_ids"), "all_special_ids")
    added = _tokenizer_attribute(tokenizer, "get_added_vocab")()
    if not isinstance(added, dict):
        raise TypeError(f"the tokenizer's get_added_vocab() returned {type(added).__name__}, not a dict")
    excluded |= _token_ids(added.values(), "get_added_vocab()")
    stop_ids = _tokenizer_attribute(tokenizer, "additional_stop_token_ids")
    if stop_ids is not None:
        excluded |= _token_ids(stop_ids, "additional_stop_token_ids")
    return {"size": min(sizes.values()), "excluded": sorted(excluded)}


def model_vocabulary_size(engine) -> int:
    """The number of token ids the served model counts: ``engine.tokenizer_manager.model_config.vocab_size``.

    It is the model's bound that ``vocabulary`` reads, here without the tokenizer's: SGLang takes it from the
    ``vocab_size`` of the text config (srt/configs/model_config.py ``ModelConfig``). One architecture differs:
    for ``GlmImageForConditionalGeneration`` SGLang takes the ``vision_vocab_size`` of the text config, which
    is the size of that model's output head and not of its input embedding (srt/models/glm_image_vl.py), so
    for that model the value is not the bound of the input ids. The ids of a tokenized text are compared
    with it, because a tokenizer can hold an id at or above the model's bound, and what the model does with
    such an id was not read. ``TypeError`` when the value is not an int, ``ValueError`` when it is not
    positive.
    """
    value = engine.tokenizer_manager.model_config.vocab_size
    if not _is_count(value):
        raise TypeError(f"tokenizer_manager.model_config.vocab_size is {value!r}, not an int")
    if value <= 0:
        raise ValueError(f"tokenizer_manager.model_config.vocab_size is {value}, not a positive size")
    return value


def encode_texts(engine, texts: list[str]) -> list[list[int]]:
    """The token ids of each text: one list of ids per text, in the order of ``texts``.

    ``texts`` is a non-empty list of ``str``. The tokenizer is ``engine.tokenizer_manager.tokenizer``, the one
    ``vocabulary`` reads. The call is ``tokenizer(texts, add_special_tokens=False)`` and the result is its
    ``"input_ids"``: one batch call for all texts, which a fast tokenizer spreads over the CPUs. SGLang's own
    text path makes the same batch call and reads the same key for a fast tokenizer
    (srt/managers/tokenizer_manager.py ``TokenizerManager._tokenize_texts``), there without the keyword; for a
    tokenizer that is not a fast one it calls ``encode`` once for each text. ``add_special_tokens=False`` keeps the
    tokenizer from putting a BOS or EOS id around a text. It does not keep special ids out: a tokenizer turns
    the written form of a special token inside a text into that token's id, so the ids of a text can lie
    outside the range ``vocabulary`` describes. Nothing is filtered here. The launcher keeps such an id, as
    Dynamo's vLLM self-benchmark does, and leaves out only an id at or above ``model_vocabulary_size``.

    Raises ``TypeError`` or ``ValueError`` when ``texts`` is not such a list, ``ValueError`` when there is no
    tokenizer, and ``TypeError`` or ``ValueError`` that says what came back when the result is not a list of
    ``len(texts)`` lists of plain ints. A tokenizer that cannot be called this way raises whatever it raises
    (SGLang can also serve tokenizers that are not Hugging Face ones); that error passes through.
    """
    if not isinstance(texts, list):
        raise TypeError(f"texts must be a list of str, got {type(texts).__name__}")
    if not texts:
        raise ValueError("texts must not be empty")
    for position, text in enumerate(texts):
        if type(text) is not str:
            raise TypeError(f"text {position} is {type(text).__name__}, not a str")
    tokenizer = _served_tokenizer(engine)
    name = type(tokenizer).__name__
    encoded = tokenizer(texts, add_special_tokens=False)
    try:
        batch = encoded["input_ids"]
    except (KeyError, IndexError, TypeError) as error:
        raise TypeError(f"the tokenizer {name} returned {type(encoded).__name__} without input_ids") from error
    if not isinstance(batch, list):
        raise TypeError(f"the tokenizer {name} returned input_ids as {type(batch).__name__}, not as a list")
    if len(batch) != len(texts):
        raise ValueError(f"the tokenizer {name} returned {len(batch)} id lists for {len(texts)} texts")
    for position, ids in enumerate(batch):
        if not isinstance(ids, list):
            raise TypeError(f"the tokenizer {name} returned {type(ids).__name__} for text {position}, not a list")
        # One pass in C over the types: a text has thousands of ids, and a bool is not a plain int.
        if not set(map(type, ids)) <= {int}:
            wrong = next(token for token in ids if type(token) is not int)
            raise TypeError(f"the tokenizer {name} returned {wrong!r} among the ids of text {position}")
    return batch


def tokenizer_class(engine) -> str:
    """The module and the class name of the served tokenizer, as ``"module.ClassName"``.

    The tokenizer is ``engine.tokenizer_manager.tokenizer``; ``ValueError`` when there is none. The name says
    which kind of tokenizer made the ids of ``encode_texts``. It is a record and decides nothing.
    """
    kind = type(_served_tokenizer(engine))
    return f"{kind.__module__}.{kind.__qualname__}"


def child_pids(engine) -> list[int]:
    """The process ids of the engine's children: schedulers, the detokenizer and, if any, the DP controller.

    ``Engine.get_all_child_pids`` (srt/entrypoints/engine.py), which returns the list SGLang collected while
    it launched them.
    """
    pids = list(engine.get_all_child_pids())
    for pid in pids:
        if not _is_count(pid):
            raise TypeError(f"Engine.get_all_child_pids() holds {pid!r}, which is not a process id")
    return pids


def shutdown(engine) -> None:
    """Shut the engine down.

    ``Engine.shutdown`` (srt/entrypoints/engine.py). It stops SGLang's watchdog of its children and then
    sends SIGKILL to every descendant of this process (``kill_process_tree`` with ``include_parent=False``),
    waiting up to 60 s for them; it raises ``RuntimeError`` when they are not gone by then, and that error
    passes through. Schedulers are not stopped gracefully. SGLang also registers the same method with
    ``atexit``; a second call finds nothing to kill.
    """
    engine.shutdown()
