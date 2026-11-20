# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load probe: what SGLang builds when the MLA module collector loads a ModelOpt mixed export.

Run inside an SGLang image on one Blackwell GPU, with ``PYTHONPATH=<repo>/python/aisim``::

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=<repo>/python/aisim \\
        python3 probe_mixed_load.py --model nvidia/GLM-5.3-NVFP4 --kv-cache-dtype fp8 --output probe.json

The probe changes no collector code. It takes the ``dsa_context_module`` plan row
``[0, 256, 64, '<kv-cache-dtype>', ...]`` of the model on this SM (the row that
``collect.py --case-filter "[0, 256, 64, "`` runs) and calls
``collector.sglang.collect_mla_module.load_model_runner`` with the arguments that
``run_mla_module`` passes for that row: 64 heads, the row's KV dtype, ``attention_backend``
``"dsa"``, the row's pinned DSA backend, ``gemm_type`` ``"bfloat16"``, target TP 1 and
``max_total_tokens`` from the row's points. The loader uses dummy weights and two layers unless
``SGLANG_LOAD_FORMAT`` or ``SGLANG_TEST_NUM_LAYERS`` say otherwise. As in ``run_mla_module``, the
probe sets the CUDA device and clears the distributed state first. ``run_mla_module`` also refuses
points over the prefill chunk before it sizes the pool. The probe sizes the pool from all points
of the row. The pool size changes no quantization method and no KV dtype.

It writes one JSON object (``--output``) and then exits:

* 0 when every check passes;
* 1 when ``model_config.quantization`` is not ``modelopt_mixed``, when it finds no attention
  projection, when an attention projection has a quantization method other than
  ``UnquantizedLinearMethod`` or a weight dtype other than ``torch.bfloat16``, or when the KV pool
  dtype does not correspond to ``--kv-cache-dtype``. The JSON key ``failures`` names each one;
* 1 without JSON when the plan has not exactly one such row, or the row does not match the request;
* 2 for a usage error; any other error is an uncaught exception (nonzero exit, no JSON).

Linear modules are the ``sglang.srt.layers.linear.LinearBase`` instances in
``model_runner.model.named_modules()``. ``LinearBase.__init__`` stores the method in the attribute
``quant_method`` at both SGLang 0.5.21 and 0.5.23 (``layers/linear.py``). A linear module without
that attribute is a failure.

Attention projections are found by their position in the module tree, not by name: every
``LinearBase`` inside ``model_runner.model.model.layers[i].self_attn`` (the DSA projections and the
DSA indexer projections). The qualified name pattern is ``model.layers.<i>.self_attn.<...>``. It
comes from ``GlmMoeDsaForCausalLM`` (``python/sglang/srt/models/glm4_moe.py``), a subclass of
``DeepseekV2ForCausalLM`` (``python/sglang/srt/models/deepseek_v2.py``): ``self.model =
DeepseekV2Model(..., prefix=add_prefix("model", prefix))``; ``DeepseekV2Model.layers`` with prefix
``"layers"`` and one ``add_prefix(idx, prefix)`` per layer (``make_layers`` at 0.5.21,
``make_pp_layers`` at 0.5.23, ``utils/common.py``); ``DeepseekV2DecoderLayer.self_attn =
DeepseekV2AttentionMLA(..., prefix=add_prefix("self_attn", prefix))``. The probe raises when the
tree position and the name pattern disagree.

The KV dtype is ``model_runner.token_to_kv_pool.dtype``, the logical dtype of the pool
(``KVCache.__init__`` in ``mem_cache/memory_pool.py``). The pool stores an FP8 cache as
``torch.uint8``, so the JSON also gives ``store_dtype`` and the dtype of the first buffer tensor.
The expected dtype maps ``--kv-cache-dtype`` through the collector's ``SGLANG_KV_DTYPE``, as
``load_model_runner`` does.
"""

import argparse
import json
import os
import re
import sys
from importlib.metadata import version as package_version

# The collector's SGLang KV dtype strings (collect_mla_module.SGLANG_KV_DTYPE values) as torch dtype
# names. A value the collector adds later and this table does not know is a failure, not a guess.
_TORCH_KV_DTYPE_NAMES = {"bfloat16": "bfloat16", "fp8_e4m3": "float8_e4m3fn"}
_ROW_FILTER = "[0, 256, 64, '{kv}', "
_ATTENTION_NAME = re.compile(r"model\.layers\.\d+\.self_attn\.")


def _dtype_name(dtype) -> str | None:
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _plan_row(cmm, model: str, kv_cache_dtype: str, sm_version: int):
    """Return the one plan row that ``collect.py --case-filter "[0, 256, 64, "`` runs for this KV dtype."""
    from collector.op_plans import plan_cases

    previous = os.environ.get("COLLECTOR_MODEL_PATH")
    os.environ["COLLECTOR_MODEL_PATH"] = model
    try:
        rows = plan_cases("dsa_context_module", "sglang", sm_version)
    finally:
        if previous is None:
            os.environ.pop("COLLECTOR_MODEL_PATH", None)
        else:
            os.environ["COLLECTOR_MODEL_PATH"] = previous
    wanted = _ROW_FILTER.format(kv=kv_cache_dtype)
    matches = [row for row in rows if str(row).startswith(wanted)]
    if len(matches) != 1:
        raise SystemExit(
            f"probe: expected exactly one dsa_context_module plan row starting with {wanted!r} for {model} "
            f"on SM{sm_version}, found {len(matches)} of {len(rows)} rows"
        )
    return matches[0]


def _linear_modules(model_runner):
    from sglang.srt.layers.linear import LinearBase

    model = model_runner.model
    try:
        layers = model.model.layers
    except AttributeError as exc:
        raise RuntimeError(f"probe: {type(model).__name__} has no model.layers") from exc

    attention_ids = {}
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise RuntimeError(f"probe: model.layers[{index}] ({type(layer).__name__}) has no self_attn")
        for module in attention.modules():
            attention_ids[id(module)] = index

    records = []
    failures = []
    for name, module in model.named_modules():
        if not isinstance(module, LinearBase):
            continue
        is_attention = id(module) in attention_ids
        if is_attention != bool(_ATTENTION_NAME.match(name)):
            raise RuntimeError(f"probe: tree position and name pattern disagree for {name}")
        if not hasattr(module, "quant_method"):
            failures.append(f"{name}: LinearBase without a quant_method attribute")
            method_class = None
        else:
            method_class = type(module.quant_method).__name__
        weight = getattr(module, "weight", None)
        weight_dtype = _dtype_name(getattr(weight, "dtype", None))
        records.append(
            {
                "name": name,
                "module_class": type(module).__name__,
                "quant_method_class": method_class,
                "weight_dtype": weight_dtype,
                "attention_projection": is_attention,
            }
        )
        if is_attention and method_class is not None and method_class != "UnquantizedLinearMethod":
            failures.append(f"{name}: attention projection with {method_class}")
        if is_attention and weight_dtype != "bfloat16":
            failures.append(f"{name}: attention projection with weight dtype {weight_dtype}")
    return records, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hugging Face model ID, for example nvidia/GLM-5.3-NVFP4")
    parser.add_argument("--kv-cache-dtype", required=True, choices=["fp8", "bfloat16"])
    parser.add_argument("--output", required=True, help="Path of the JSON result")
    args = parser.parse_args()

    import collector.sglang.collect_mla_module as cmm
    import torch

    sglang_kv_dtype = cmm.SGLANG_KV_DTYPE[args.kv_cache_dtype]
    expected_kv_dtype = _TORCH_KV_DTYPE_NAMES.get(sglang_kv_dtype)

    sm_version = cmm.get_sm_version()
    row = _plan_row(cmm, args.model, args.kv_cache_dtype, sm_version)
    _seq, _batch, heads, kv_dtype, _compute, gemm, model_path, _attn, backend, target_tp, pinned, points = row
    if model_path != args.model or kv_dtype != args.kv_cache_dtype or heads != 64:
        raise SystemExit(f"probe: plan row {str(row)[:120]} does not match the request")
    attention_backend = backend or cmm._get_backends("dsa")
    # run_mla_module: required_kv_alloc_tokens per point, then a margin of max(1024, 5%).
    max_total_tokens = max(
        cmm.required_kv_alloc_tokens(int(b), int(s), int(p), cmm.SGLANG_DSA_PAGE_SIZE, is_prefill=True)
        for b, s, p in points
    )
    max_total_tokens += max(1024, max_total_tokens // 20)

    device = "cuda:0"
    torch.cuda.set_device(device)
    cmm.cleanup_distributed()
    torch.cuda.empty_cache()
    try:
        model_runner = cmm.load_model_runner(
            model_path=args.model,
            head_num=heads,
            kv_cache_dtype=kv_dtype,
            attention_backend=attention_backend,
            dsa_prefill_backend=pinned,
            device=device,
            gemm_type=gemm,
            target_tp_size=target_tp,
            enable_piecewise_cuda_graph=False,
            max_total_tokens=max_total_tokens,
            chunked_prefill_size=None,
        )

        failures = []
        resolved_quantization = model_runner.model_config.quantization
        if resolved_quantization != "modelopt_mixed":
            failures.append(f"model_config.quantization is {resolved_quantization!r}, not 'modelopt_mixed'")

        linear_modules, linear_failures = _linear_modules(model_runner)
        failures.extend(linear_failures)
        attention_count = sum(record["attention_projection"] for record in linear_modules)
        if attention_count == 0:
            failures.append("no attention projection found under model.layers[*].self_attn")

        pool = model_runner.token_to_kv_pool
        pool_dtype = _dtype_name(getattr(pool, "dtype", None))
        buffers = getattr(pool, "kv_buffer", None)
        if isinstance(buffers, (list, tuple)):
            buffers = buffers[0] if buffers else None
        buffer_dtype = _dtype_name(getattr(buffers, "dtype", None))
        if expected_kv_dtype is None:
            failures.append(f"no torch dtype known for the collector KV dtype {sglang_kv_dtype!r}")
        elif pool_dtype != expected_kv_dtype:
            failures.append(
                f"KV pool dtype is {pool_dtype}, but --kv-cache-dtype {args.kv_cache_dtype} "
                f"({sglang_kv_dtype}) expects {expected_kv_dtype}"
            )

        result = {
            "sglang_version": package_version("sglang"),
            "collector_compat": cmm.__compat__,
            "gpu_name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "model": args.model,
            "plan_row": str(row),
            "num_layers": int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2")),
            "load_format": os.environ.get("SGLANG_LOAD_FORMAT", "dummy"),
            "checkpoint_quant": cmm._model_native_gemm_quant(args.model),
            "requested_quantization": getattr(model_runner.server_args, "quantization", None),
            "resolved_quantization": resolved_quantization,
            "requested_kv_cache_dtype": args.kv_cache_dtype,
            "server_args_kv_cache_dtype": getattr(model_runner.server_args, "kv_cache_dtype", None),
            "expected_kv_cache_dtype": expected_kv_dtype,
            "kv_cache_dtype": pool_dtype,
            "kv_pool": {
                "class": type(pool).__name__,
                "dtype": pool_dtype,
                "store_dtype": _dtype_name(getattr(pool, "store_dtype", None)),
                "buffer_dtype": buffer_dtype,
                "model_runner_kv_cache_dtype": _dtype_name(getattr(model_runner, "kv_cache_dtype", None)),
            },
            "attention_projection_count": attention_count,
            "linear_modules": linear_modules,
            "failures": failures,
        }
    finally:
        cmm.cleanup_distributed()

    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    for failure in failures:
        print(f"probe FAILED: {failure}", file=sys.stderr)
    print(
        f"probe: sglang {result['sglang_version']}, resolved {resolved_quantization!r}, KV pool {pool_dtype}, "
        f"{attention_count} attention projections, {len(failures)} failures; wrote {args.output}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
