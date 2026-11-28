# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vLLM MSA collector lays each cache out in the layout serving resolves and
times the indexer on written history and index projections of its own."""

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_msa_module.py"


class _Tensor:
    """The slice of a torch tensor _layer_kv_cache touches, over numpy."""

    def __init__(self, array):
        self.array = array

    def permute(self, *dims):
        return _Tensor(self.array.transpose(dims))

    def view(self, _dtype):
        return self


def _layer_kv_cache(monkeypatch):
    """Load only _layer_kv_cache: the module itself imports vllm and torch."""
    tree = ast.parse(SRC.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_layer_kv_cache")
    fake_torch = SimpleNamespace(uint8=np.uint8, zeros=lambda shape, dtype, device: _Tensor(np.zeros(shape, dtype)))
    interface = types.ModuleType("vllm.v1.kv_cache_interface")
    interface.compute_layer_kv_cache_shape_bytes = lambda spec, num_blocks: (num_blocks, spec.heads, spec.tokens, 512)
    for name in ("vllm", "vllm.v1"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", interface)
    namespace = {"torch": fake_torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), namespace)
    return namespace["_layer_kv_cache"]


@pytest.mark.parametrize(
    "layer_view_order,strides",
    [
        # LBHNC (SM100 family): each head's 128 tokens are one contiguous run.
        ((0, 1, 2, 3), (4 * 128 * 512, 128 * 512, 512, 1)),
        # LBNHC (elsewhere): a token's heads sit side by side inside the block.
        ((0, 2, 1, 3), (128 * 4 * 512, 512, 4 * 512, 1)),
    ],
)
def test_cache_keeps_the_logical_view_over_the_resolved_physical_order(monkeypatch, layer_view_order, strides):
    layer_kv_cache = _layer_kv_cache(monkeypatch)
    spec = SimpleNamespace(heads=4, tokens=128, dtype="bfloat16")

    cache = layer_kv_cache(spec, 3, SimpleNamespace(layer_view_order=layer_view_order), "cuda:0").array

    assert cache.shape == (3, 4, 128, 512)
    assert cache.strides == strides


# numpy stand-ins for the torch dtypes the collector tells apart.
_FP8, _BF16 = np.float16, np.float64


class _Array(np.ndarray):
    """The slice of a torch tensor the input-preparation helpers touch, over numpy."""

    device = "cuda:0"

    def copy_(self, other):
        self[...] = other
        return self

    def to(self, *args, dtype=None, device=None):
        dtype = next((arg for arg in args if not isinstance(arg, str)), dtype)
        return self if dtype is None else self.astype(dtype)


def _array(shape, dtype, value=0.0):
    return np.full(shape, value, dtype=dtype).view(_Array)


class _Generator:
    def __init__(self, device):
        assert device == "cpu"

    def manual_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        return self


def _helpers(*names, **constants):
    """Load the named helpers over numpy: the module itself imports vllm and torch."""
    from contextlib import nullcontext

    tree = ast.parse(SRC.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    fake_torch = SimpleNamespace(
        Tensor=_Array,
        Generator=_Generator,
        no_grad=nullcontext,
        device=lambda value: value,
        uint8=np.uint8,
        float32=np.float32,
        randn=lambda shape, generator, dtype, device: (
            generator.rng.standard_normal(tuple(shape)).astype(dtype).view(_Array)
        ),
        randint=lambda low, high, shape, generator, dtype, device: (
            generator.rng.integers(low, high, tuple(shape)).astype(dtype).view(_Array)
        ),
        ones=lambda shape, dtype, device: _array(tuple(shape), dtype, 1.0),
        empty=lambda shape, dtype, device: _array(tuple(shape), dtype),
    )
    namespace = {"torch": fake_torch, **constants}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("num_blocks", [3, 5, 12])
def test_index_key_history_fills_the_cache_with_one_repeated_block_of_keys(num_blocks):
    # 5 distinct blocks of 4 tokens, drawn 2 blocks at a time.
    helpers = _helpers("_write_index_key_history", _HISTORY_TOKENS=20, _SAMPLE_CHUNK_ELEMENTS=64, _HISTORY_KEYS={})
    cache = _array((num_blocks, 1, 4, 8), np.float32)

    helpers["_write_index_key_history"](cache)

    assert np.all(cache != 0) and 0.5 < cache.std() < 1.5
    distinct = min(num_blocks, 5)
    assert len({block.tobytes() for block in cache[:distinct]}) == distinct
    assert len({row.tobytes() for row in cache[:distinct].reshape(-1, 8)}) == distinct * 4
    for block in range(distinct, num_blocks):
        assert np.array_equal(cache[block], cache[block % 5])


def test_index_key_history_is_seeded():
    first, second, third = (_array((6, 1, 4, 8), np.float32) for _ in range(3))
    helpers = _helpers(
        "_write_index_key_history", _HISTORY_TOKENS=1 << 18, _SAMPLE_CHUNK_ELEMENTS=1 << 20, _HISTORY_KEYS={}
    )
    helpers["_write_index_key_history"](first)
    helpers["_write_index_key_history"](second)  # the worker's kept block
    again = _helpers(
        "_write_index_key_history", _HISTORY_TOKENS=1 << 18, _SAMPLE_CHUNK_ELEMENTS=1 << 20, _HISTORY_KEYS={}
    )
    again["_write_index_key_history"](third)  # another worker samples the same keys

    assert np.array_equal(first, second) and np.array_equal(first, third)
    assert len(helpers["_HISTORY_KEYS"]) == 1


def _projection(weight_dtype, *, packed=False):
    """A fused [q | k | v | index_q | index_k] projection of 2 + 1 + 1 + 2 + 1 heads of 2 rows."""
    width = 4 if packed else 8
    projection = SimpleNamespace(weight=_array((14, width), weight_dtype))
    if packed:
        projection.weight_scale = _array((14, 1), _FP8)
    return SimpleNamespace(qkv_proj=projection, index_q_size=4, idx_head_dim=2)


@pytest.mark.parametrize("weight_dtype", [_BF16, _FP8])
def test_index_projection_rows_are_seeded_and_differ_between_heads(weight_dtype):
    helpers = _helpers("_give_index_projection_values")
    module, again = _projection(weight_dtype), _projection(weight_dtype)
    module.qkv_proj.weight[:8] = 0.01

    helpers["_give_index_projection_values"](module)
    helpers["_give_index_projection_values"](again)

    weight = module.qkv_proj.weight
    assert weight.dtype == weight_dtype
    # q, k and v keep the dummy fill; the index_q and index_k rows are the last six.
    assert np.all(weight[:8] == weight_dtype(0.01))
    assert np.all(weight[8:] != 0) and np.abs(weight[8:]).max() < 0.2
    assert len({row.tobytes() for row in weight[8:]}) == 6
    assert np.array_equal(weight[8:], again.qkv_proj.weight[8:])


def test_packed_nvfp4_index_rows_take_seeded_bytes_and_a_unit_block_scale():
    helpers = _helpers("_give_index_projection_values")
    module = _projection(np.uint8, packed=True)

    helpers["_give_index_projection_values"](module)

    weight, scale = module.qkv_proj.weight, module.qkv_proj.weight_scale
    assert weight.dtype == np.uint8 and scale.dtype == _FP8
    assert not weight[:8].any() and not scale[:8].any()
    assert len({row.tobytes() for row in weight[8:]}) == 6
    assert np.all(scale[8:] == 1.0)


@pytest.mark.parametrize("meta", [False, True])
def test_move_keeps_the_rotary_table_when_to_empty_reallocates_buffers(meta):
    helpers = _helpers("_move_to_device")
    table = np.arange(6, dtype=np.float32).reshape(3, 2)

    class Module:
        def __init__(self):
            self.rotary = SimpleNamespace(cos_sin_cache=table.copy().view(_Array))
            self.moves = []

        def parameters(self):
            return [SimpleNamespace(is_meta=meta)]

        def modules(self):
            return [self, self.rotary]

        def to(self, device):
            self.moves.append(("to", device))
            return self

        def to_empty(self, *, device):
            self.moves.append(("to_empty", device))
            self.rotary.cos_sin_cache = _array((3, 2), np.float32, np.nan)
            return self

    module = Module()

    assert helpers["_move_to_device"](module, "cuda:0") is module
    assert module.moves == [("to_empty" if meta else "to", "cuda:0")]
    assert np.array_equal(module.rotary.cos_sin_cache, table)


def _function(name):
    tree = ast.parse(SRC.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _call_lines(function, name):
    return [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == name
    ]


def test_dummy_fill_skips_the_rotary_table_and_seeds_before_processing():
    # The move/fill/seed/process sequence is shared by the MSA and QSA builders
    # (_finalize_module_weights), so the rotary-table contract is pinned there.
    # The index seeder is an architecture hook: MSA passes MiniMax's fused
    # qkv+index seeder, QSA passes none (its layout differs and its accepted
    # rows were measured with the constant fill).
    finalize = _function("_finalize_module_weights")
    fill = next(
        node for node in ast.walk(finalize) if isinstance(node, ast.For) and "named_buffers" in ast.unparse(node.iter)
    )
    # Both parameter and buffer names are walked: the RoPE table is a buffer.
    assert "named_parameters" in ast.unparse(fill.iter)
    # The table is passed over before any branch can overwrite it.
    skip = fill.body[1]
    assert ast.unparse(skip.test) == "name.endswith('cos_sin_cache')"
    assert isinstance(skip.body[-1], ast.Continue)

    (moved,) = _call_lines(finalize, "_move_to_device")
    (seeded,) = _call_lines(finalize, "seed_index")
    (processed,) = _call_lines(finalize, "process_weights_after_loading")
    assert moved < fill.lineno < seeded < processed

    def finalize_kwargs(builder):
        (call,) = [
            node
            for node in ast.walk(_function(builder))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "_finalize_module_weights"
        ]
        return {kw.arg: ast.unparse(kw.value) for kw in call.keywords}

    assert finalize_kwargs("_create_msa_attention_module") == {"seed_index": "_give_index_projection_values"}
    assert finalize_kwargs("_create_qsa_attention_module") == {}


def test_qsa_builder_installs_servings_skinny_gemm_plans_before_processing_weights():
    """Serving swaps in its measured GEMM plans at the end of model __init__.

    ``Qwen4ExpForCausalLM.__init__`` (and ``Qwen4ExpMTP``) call
    ``enable_qwen4_exp_low_latency_gemm(self, self.model_config.dtype)``
    (model.py:747, mtp.py @v0.32.0). That replaces the quant method of every
    unquantized linear whose ``(N, K)`` has a plan, so those shapes run the
    skinny kernel rather than ``F.linear``. The plans key on exact widths (the
    TP4 fused QKV among them), so a module built without the call times a
    different kernel than serving and labels the row as serving's.
    """
    build = _function("_create_qsa_attention_module")
    (constructed,) = _call_lines(build, "Qwen4ExpQSAAttention")
    (selected,) = _call_lines(build, "enable_qwen4_exp_low_latency_gemm")
    (finalized,) = _call_lines(build, "_finalize_module_weights")
    assert constructed < selected < finalized

    (call,) = [
        node
        for node in ast.walk(build)
        if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == "enable_qwen4_exp_low_latency_gemm"
    ]
    # Same module and the dtype serving passes.
    assert ast.unparse(call.args[0]) == "attn_module"
    assert ast.unparse(call.args[1]) == "vllm_config.model_config.dtype"


def test_history_is_written_after_the_dtype_check_and_before_the_first_forward():
    run = _function("run_msa_module")
    (checked,) = _call_lines(run, "_check_msa_kv_precision")
    (written,) = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_write_index_key_history"
    ]
    assert ast.unparse(written.args[0]) == "index_kv_cache"
    # The raises live in the shared helpers both builders call.
    check_fn = _function("_check_msa_kv_precision")
    raises = [node for node in ast.walk(check_fn) if isinstance(node, ast.Raise)]
    assert any("precision mismatch" in ast.unparse(node) for node in raises)
    assert checked < written.lineno < min(_call_lines(run, "forward"))

    # The M3 index-key history is MSA-only: QSA has two side caches and no M3
    # index layout, so it keeps the zero fill its accepted rows were measured
    # with. The write must sit in the branch that is NOT the QSA route.
    (branch,) = [
        node
        for node in run.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "attention_type == 'qsa'"
        and any(
            isinstance(call, ast.Call) and ast.unparse(call.func) == "_check_msa_kv_precision"
            for call in ast.walk(ast.Module(body=node.orelse, type_ignores=[]))
        )
    ]
    assert any(node is written for node in ast.walk(ast.Module(body=branch.orelse, type_ignores=[])))
    assert not any(node is written for node in ast.walk(ast.Module(body=branch.body, type_ignores=[])))


class _TensorPath:
    """Chainable stand-in for the `arange(...).unsqueeze().expand()...` shape."""

    def unsqueeze(self, _axis):
        return self

    def expand(self, *_shape):
        return self

    def reshape(self, *_shape):
        return self

    def contiguous(self):
        return self


def _run_msa_module(*, attention_type, calls):
    """Exec the collector's real ``run_msa_module`` over stubbed framework/GPU boundaries.

    The module imports vllm and torch at import time, so the point runner is
    executed from its own AST with every boundary replaced: the framework
    module/cache builders, the distributed and workspace setup, the timing
    harness and the row writer. Everything between them — the dispatcher, the
    branch split, the tensor preparation — is the shipped code.
    """
    import copy
    from contextlib import nullcontext

    forward_ctx = {
        "attn": SimpleNamespace(bind_kv_cache=lambda cache: calls["bound"].append(cache)),
        "index": SimpleNamespace(bind_kv_cache=lambda cache: calls["bound"].append(cache)),
    }
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=forward_ctx),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=2560), uses_mrope=False),
    )
    attn_module = SimpleNamespace(
        forward=lambda *_a: calls["forwards"].append(1),
        impl=SimpleNamespace(__class__=type("Impl", (), {})),
        kv_cache=SimpleNamespace(dtype="bfloat16"),
    )

    def create_msa(**kwargs):
        calls["create"] = kwargs
        return attn_module, vllm_config, "MiniMaxM3ForCausalLM"

    def create_qsa(**kwargs):
        calls["create"] = kwargs
        return attn_module, vllm_config, "Qwen4ExpForConditionalGeneration"

    main_cache, index_cache = SimpleNamespace(dtype="bfloat16"), SimpleNamespace(dtype="bfloat16")

    def create_msa_caches(**_kwargs):
        calls["caches"] = "msa"
        return (
            main_cache,
            index_cache,
            object(),
            object(),
            SimpleNamespace(slot_mapping=object()),
        )

    def create_qsa_caches(**_kwargs):
        calls["caches"] = "qsa"
        return {"qsa": object()}, SimpleNamespace(slot_mapping=object())

    @_nullcontext_manager
    def bench(**_kwargs):
        return {"latency_ms": 3.5, "power_stats": {"power": 1.0}}

    fake_torch = SimpleNamespace(
        inference_mode=nullcontext,
        device=lambda value: value,
        bfloat16="bfloat16",
        long="long",
        uint8="uint8",
        arange=lambda *_a, **_k: _TensorPath(),
        full=lambda *_a, **_k: object(),
        cuda=SimpleNamespace(
            set_device=lambda _d: None,
            get_device_name=lambda _d: "GPU",
            get_device_properties=lambda _d: SimpleNamespace(total_memory=1 << 40),
            OutOfMemoryError=MemoryError,
        ),
    )
    namespace = {
        "torch": fake_torch,
        "traceback": SimpleNamespace(print_exc=lambda: None),
        "vllm_version": "0.32.0",
        "_MSA_MODULE_ATTENTION_TYPES": ("msa", "qsa"),
        "_ATTN_LAYER_NAME": "attn",
        "_INDEX_CACHE_LAYER_NAME": "index",
        "_attention_type_for_model": lambda _path: attention_type,
        "_qsa_generation_kv_footprint_bytes": lambda _path: None,
        "_kv_does_not_fit": lambda *_a, **_k: None,
        "setup_distributed": lambda _d: None,
        "enable_engine_fused_ops": lambda: None,
        "init_workspace_manager": lambda _d: None,
        "_create_msa_attention_module": create_msa,
        "_create_qsa_attention_module": create_qsa,
        "_create_kv_caches_and_metadata": create_msa_caches,
        "_create_qsa_kv_caches_and_metadata": create_qsa_caches,
        "_check_msa_kv_precision": lambda *_a: None,
        "_check_qsa_kv_precision": lambda *_a: None,
        "_write_index_key_history": lambda cache: calls["written"].append(cache),
        "_qsa_kernel_source": lambda _module, context: "qsa_prefill" if context else "qsa_decode",
        "_qsa_forward_positions": lambda *_a, **_k: object(),
        "set_current_vllm_config": lambda _config: nullcontext(),
        "set_forward_context": lambda *_a, **_k: nullcontext(),
        "benchmark_with_power": bench,
        "_persist_msa_row": lambda **kwargs: calls["rows"].append(kwargs),
        "_cleanup": lambda: None,
    }
    tree = ast.parse(SRC.read_text())
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_msa_module"))
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), namespace)
    return namespace["run_msa_module"]


def _nullcontext_manager(func):
    from contextlib import contextmanager

    @contextmanager
    def wrapped(*args, **kwargs):
        yield func(*args, **kwargs)

    return wrapped


@pytest.mark.parametrize("attention_type", ["msa", "qsa"])
def test_run_msa_module_prices_both_builders_from_the_new_token_budget(attention_type):
    """Both builders size their per-step buffers from ``max_query_len``.

    The serving scheduler budgets NEW tokens per step, so this is ``seq_len``
    for a context point and ``1`` for generation — never the cached prefix.
    """
    calls = {"bound": [], "forwards": [], "written": [], "rows": [], "create": None, "caches": None}
    run = _run_msa_module(attention_type=attention_type, calls=calls)

    latency = run(
        SimpleNamespace(enter_context=lambda ctx: ctx),
        seq_len=1024,
        batch_size=2,
        num_heads=3,
        kv_cache_dtype="bfloat16",
        compute_dtype="bfloat16",
        gemm_type="bfloat16",
        perf_filename="/t/msa_context_module_perf.txt",
        prefix_len=512,
        model_path="Qwen/Qwen3.8-Flash-Next",
        device="cuda:0",
    )

    assert latency == 3.5
    assert calls["create"]["max_seq_len"] == 512 + 1024
    assert calls["create"]["max_batch_size"] == 2
    assert calls["create"]["max_query_len"] == 1024
    assert calls["caches"] == attention_type
    assert calls["rows"] and calls["rows"][0]["item"]["isl"] == 1024
    assert calls["forwards"], "the dry run and the timed kernel must run"


@pytest.mark.parametrize("builder", ["_create_msa_attention_module", "_create_qsa_attention_module"])
def test_builder_sizes_the_step_budget_from_new_tokens_not_the_prefix(builder):
    """Both builders take ``max_query_len`` and size the step budget from it.

    The serving scheduler budgets NEW tokens per step, and each module
    allocates its persistent selection/top-k buffer at
    ``max_num_batched_tokens``. Sizing that from ``max_seq_len`` made a
    1M-token QSA prefix reserve a ~4 GB packed selection buffer and the M3
    SM100 indexer up to ~140 GB of score rows, none of which serving allocates.
    """
    tree = ast.parse(SRC.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == builder)
    assert "max_query_len" in [a.arg for a in node.args.args]

    config_calls = [
        call for call in ast.walk(node) if isinstance(call, ast.Call) and ast.unparse(call.func) == "create_vllm_config"
    ]
    assert len(config_calls) == 1
    options = {kw.arg: kw.value for kw in config_calls[0].keywords}
    budget = ast.unparse(options["max_num_batched_tokens"])
    assert "max_query_len" in budget
    assert "max_seq_len" not in budget
    assert "131072" in budget
    # The KV bound still sizes the cache and the model limit.
    assert "max_seq_len" in ast.unparse(options["num_gpu_blocks"])
    local = {
        target.id: ast.unparse(assign.value)
        for assign in ast.walk(node)
        if isinstance(assign, ast.Assign)
        for target in assign.targets
        if isinstance(target, ast.Name)
    }
    assert "max_seq_len" in local["max_model_len"]


@pytest.mark.parametrize("attention_type", ["msa", "qsa"])
def test_run_msa_module_writes_m3_history_only_on_the_msa_branch(attention_type):
    """QSA's ``index_kv_cache`` is unbound; the shared history write is MSA-only."""
    calls = {"bound": [], "forwards": [], "written": [], "rows": [], "create": None, "caches": None}
    run = _run_msa_module(attention_type=attention_type, calls=calls)

    run(
        SimpleNamespace(enter_context=lambda ctx: ctx),
        seq_len=1024,
        batch_size=2,
        num_heads=3,
        kv_cache_dtype="bfloat16",
        compute_dtype="bfloat16",
        gemm_type="bfloat16",
        perf_filename="/t/msa_context_module_perf.txt",
        prefix_len=0,
        model_path="Qwen/Qwen3.8-Flash-Next",
        device="cuda:0",
    )

    if attention_type == "qsa":
        assert calls["written"] == [], "QSA must keep the constructor's zero fill"
    else:
        assert len(calls["written"]) == 1


def test_standalone_cli_offers_only_the_sparse_attention_families():
    """``--model``'s choices (and a bare ``--quick``) must not list MLA/DSA rows.

    ``get_mla_module_model_specs()`` without an ``attention_type`` returns every
    ``mla_module`` row, so the unfiltered list offered ``deepseek-ai/DeepSeek-V3``
    (``mla``) as a model for this collector. ``run_msa_module`` rejects that
    type, and a bare ``--mode context --quick`` walks the same list, so the run
    stopped on the first MLA row.
    """
    rows = [
        SimpleNamespace(model_path="MiniMax/MiniMax-M3", attention_type="msa"),
        SimpleNamespace(model_path="deepseek-ai/DeepSeek-V3", attention_type="mla"),
        SimpleNamespace(model_path="deepseek-ai/DeepSeek-V3.2", attention_type="dsa"),
        SimpleNamespace(model_path="Qwen/Qwen3.8-Flash-Next", attention_type="qsa"),
    ]
    namespace = {
        "MLAModuleModelSpec": SimpleNamespace,
        "_MSA_MODULE_ATTENTION_TYPES": ("msa", "qsa"),
        "get_mla_module_model_specs": lambda **_kwargs: rows,
    }
    exec(
        compile(ast.Module(body=[_function("_standalone_model_specs")], type_ignores=[]), str(SRC), "exec"),
        namespace,
    )

    kept = namespace["_standalone_model_specs"]()

    assert [spec.attention_type for spec in kept] == ["msa", "qsa"]
    assert [spec.model_path for spec in kept] == ["MiniMax/MiniMax-M3", "Qwen/Qwen3.8-Flash-Next"]
