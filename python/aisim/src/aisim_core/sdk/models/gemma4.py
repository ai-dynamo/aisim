# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from packaging.version import Version

import aisim_core.sdk.operations as ops
from aisim_core.decoder_config import decoder_config_key
from aisim_core.sdk import common, perf_database
from aisim_core.sdk.config import validate_parallel_size
from aisim_core.sdk.models.base import BaseModel, register_model
from aisim_core.sdk.models.blocks.vit import build_gemma4_vision_encoder_ops
from aisim_core.sdk.models.helpers import mtp_scale_factor, quant_exclude_patterns
from aisim_core.sdk.utils import gemma4_cache_shapes

# Official release index (first) and its Linux platform manifest digests for the vLLM
# releases with supported Transformers readers that disagree on a Gemma configuration.
# Pin the image index, then the amd64 and (for CUDA) arm64 manifests, in that order.
_GEMMA4_ADAPTER_IMAGES = {
    ("0.28.0", "vllm"): (
        "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14",
        "sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635",
        "sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce",
    ),
    ("0.28.0", "vllm_xpu"): (
        "sha256:4756b66a077627133cee653b551f6f5eaa1b9a981b5eea13edd33fcd3b0d3ca3",
        "sha256:d3e9c5f146a8251c7489107e6353cf512fcd1bf14a16cab17148125a62a537c8",
    ),
    ("0.29.0", "vllm"): (
        "sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1",
        "sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b",
        "sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a",
    ),
    ("0.29.0", "vllm_xpu"): (
        "sha256:96db42e248d48760a4937eb3d04c4878b39d13a9814efea95d510393e097a901",
        "sha256:1db27a8b75ae1d6b3cbf16ebbb88310c2b5548221c8df1335082b2a54dba209a",
    ),
    ("0.30.0", "vllm"): (
        "sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90",
        "sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40",
        "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56",
    ),
    ("0.30.0", "vllm_xpu"): (
        "sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d",
        "sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9",
    ),
    ("0.31.0", "vllm"): (
        "sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3",
        "sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b",
        "sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971",
    ),
    ("0.31.0", "vllm_xpu"): (
        "sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7",
        "sha256:0229393589110073085ffb558d3da4ce1114b93e985019a8811433b5fc793bac",
    ),
    ("0.32.0", "vllm"): (
        "sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34",
        "sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9",
        "sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc",
    ),
    ("0.32.0", "vllm_xpu"): (
        "sha256:20509fabe50dda62c09c0786d1a6effa299d3dbdddcc49331b93e65144652a8c",
        "sha256:9dd913e9036633d55ace17265cbfc29f48b9b85fe4bc8b129debcf7e1b269ce7",
    ),
    ("0.33.0", "vllm"): (
        "sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
        "sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3",
        "sha256:e3481489818d3e563cae517b84ab959cc8b69f85a9fbb771c189140131ffc65c",
    ),
    ("0.33.0", "vllm_xpu"): (
        "sha256:be3a9aea234be64ffc2816ac561c597e0f0b1f44a61a7f4912780e4d77450086",
        "sha256:010bdc441c72ee222f459d3b688bb41317525f1bd7223afe6218490b0c526832",
    ),
}


@lru_cache(maxsize=256)
def _table_rows(path: Path, revision: tuple[int, ...]) -> int:
    if path.suffix == ".parquet":
        from pyarrow.parquet import read_metadata

        return read_metadata(path).num_rows
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(line for line in stream if line.strip() and not line.lstrip().startswith("#"))
        next(rows, None)  # CSV header
        return sum(1 for _ in rows)


@lru_cache(maxsize=256)
def _frozen_table_evidence(path: Path, revision: tuple[int, ...]) -> tuple[int, str | None]:
    from pyarrow.parquet import ParquetFile

    retained = ParquetFile(path)
    if "runtime_digest" in retained.schema_arrow.names:
        digests = retained.read(columns=["runtime_digest"])["runtime_digest"].unique().to_pylist()
        if len(digests) == 1 and isinstance(digests[0], str):
            return retained.metadata.num_rows, digests[0]
    return retained.metadata.num_rows, None


def _valid_gemma4_v1_record(collection: dict, table: str, event: object, path: Path) -> bool:
    """Check the collector's published v1 forms before using a retained table as runtime evidence."""
    if not isinstance(event, dict) or "runtime" in event or event.get("status") not in ("complete", "partial"):
        return False
    try:
        runtime = collection.get("runtime")
        fields = event.keys()
        # The frozen sparse-attention collector records a concrete platform with its image manifest.
        frozen = (
            collection.get("provenance") is None
            and table == "glm53_attention_module_perf"
            and event.get("collector") == "collector.vllm.glm53flash_attention_runner"
            and isinstance(runtime, dict)
            and ("image_arch" not in runtime or runtime["image_arch"] in ("amd64", "arm64"))
        )
        perf_database._validate_collection_runtime(
            {k: v for k, v in runtime.items() if k != "image_arch"} if frozen else runtime,
            field="runtime",
            path=str(path),
        )
        if frozen:
            digests = _GEMMA4_ADAPTER_IMAGES.get((runtime["version"], runtime["framework"]), ())
            platform = {"amd64": 1, "arm64": 2}.get(runtime.get("image_arch"))
            perf_database._validate_non_negative_row_count(
                event.get("rows"), field=f"tables.{table}.rows", path=str(path)
            )
            table_path = path.parent / f"{table}.parquet"
            if not table_path.is_file() or (
                platform is not None and (len(digests) <= platform or runtime.get("image_digest") != digests[platform])
            ):
                return False
            # The catalog re-encodes Parquet after collection. Its logical rows retain the exact
            # runtime manifest, even when two releases have the same number of measurements.
            digest = event.get("data_sha256")
            return (
                isinstance(digest, str)
                and re.fullmatch("[0-9a-f]{64}", digest) is not None
                and isinstance(runtime.get("image_digest"), str)
                and _frozen_table_evidence(table_path, perf_database._file_revision(table_path))
                == (event["rows"], runtime["image_digest"])
            )
        if collection.get("provenance") is None and fields in (
            {"collector_ref", "rows", "status"},
            {"collector_ref", "collector_hash", "case_plan_hash", "rows", "status"},
        ):
            # The published B60 v1 tables recorded the collector identity before collected_at was required.
            perf_database._validate_non_negative_row_count(event["rows"], field=f"tables.{table}.rows", path=str(path))
            return all(isinstance(event[k], str) and event[k].strip() for k in fields - {"rows", "status"})
        if "schema_version" not in collection and fields != {"rows", "status"}:
            return False
        if fields == {"status"}:
            return collection.get("provenance") in ("local", "collected")
        if fields in ({"rows", "status"}, {"rows", "status", "collected_at"}):
            # Earlier measured rows/status summaries predate the provenance tier.
            perf_database._validate_non_negative_row_count(event["rows"], field=f"tables.{table}.rows", path=str(path))
            if "collected_at" not in event:
                return collection.get("provenance") is None
            return (
                collection.get("provenance") in ("local", "collected")
                and isinstance(event["collected_at"], str)
                and bool(event["collected_at"].strip())
            )
        required = perf_database._COLLECTION_EVENT_REQUIRED_KEYS
        if fields - {*required, "classified_failures"}:
            return False
        perf_database._validate_collection_event(event, table=table, index=0, path=str(path))
        return True
    except (TypeError, ValueError):
        return False


def _gemma4_collection_images(directory: Path, version: str, framework: str, *, cache: bool) -> Iterator[bool]:
    """Yield whether each retained measurement of this framework used the pinned official image."""
    path = directory / perf_database.COLLECTION_META_MARKER
    digests = _GEMMA4_ADAPTER_IMAGES.get((version, framework))
    if not digests:
        return
    collection = perf_database._load_collection_meta_yaml(str(path), cache=cache) if path.is_file() else {}
    schema = collection.get("schema_version", 1)
    tier = collection.get("provenance")
    valid_header = (
        type(schema) is int
        and not (collection.keys() - {"schema_version", "provenance", "runtime", "tables"})
        and (tier is None or (isinstance(tier, str) and bool(tier.strip())))
    )
    tables = collection.get("tables")
    if not isinstance(tables, dict):
        tables = {}
    # FPM has its own metadata sidecar; only op measurements qualify the serving image here.
    files = {
        Path(name).stem
        for name in perf_database._version_dir_data_filenames(str(directory))
        if name.endswith((".parquet", ".txt")) and name != "fpm_forward_perf.parquet"
    }
    repository = "vllm/vllm-openai-xpu" if framework == "vllm_xpu" else "vllm/vllm-openai"
    image = f"{repository}:v{version}"
    for table in files:
        table_path = directory / f"{table}.parquet"
        if not table_path.is_file():
            table_path = directory / f"{table}.txt"
        rows = _table_rows(table_path, perf_database._file_revision(table_path))
        if not rows:
            continue
        # A table with physical measurements needs a record that accounts for them.
        data = tables.get(table)
        if not valid_header or not isinstance(data, dict) or data.get("rows") == 0:
            yield False
            continue
        # A v2 summary and a full or frozen v1 record describe the finalized file. Reduced
        # historical summaries establish row presence and can omit the count altogether.
        if (schema == 2 or "collector_ref" in data or "data_sha256" in data) and data.get("rows") != rows:
            yield False
            continue
        if schema == 1:
            if not _valid_gemma4_v1_record(collection, table, data, path):
                yield False
                continue
            runtimes = (collection["runtime"],)
        else:
            # v2 records the rows contributed by each event, including zero-row attempts.
            # Contributions can overlap on merge, but must cover the retained measurements.
            events = data["collections"]
            if sum(event["rows"] for event in events) < rows:
                yield False
                continue
            runtimes = (event.get("runtime", collection["runtime"]) for event in events if event["rows"])
        for runtime in runtimes:
            if runtime["framework"] != framework:
                continue
            reference = runtime.get("image", "")
            reference = reference if isinstance(reference, str) else ""
            registry, _, rest = reference.partition("/")
            if registry in ("docker.io", "index.docker.io", "registry-1.docker.io"):
                reference = rest
            name, _, embedded_digest = reference.partition("@")
            digest = runtime.get("image_digest", embedded_digest)
            yield (
                runtime.get("version") == version
                and isinstance(digest, str)
                and digest in digests
                and name in (repository, image)
                and (not embedded_digest or embedded_digest == digest or embedded_digest == digests[0])
            )


def _has_stock_gemma4_collection(
    version: str,
    database: perf_database.PerfDatabase | None,
    system: str | None,
    systems_paths: str | list[str] | None,
) -> bool:
    """Use the selected database, or the first usable catalog in the deployment's search order."""
    if database is not None:
        recorded_dir = database.system_spec.get("data_dir")
        if not recorded_dir:
            return False
        data_dir = str(Path(database.systems_root) / recorded_dir)
        paths = (
            (directory, data_dir)
            for release, directory in perf_database._iter_backend_version_dirs(data_dir, "vllm")
            if release == version
        )
        catalogs = ((database.systems_root, paths),)
        strict = database.strict_provenance
    elif system:
        roots = perf_database.resolve_deployment_systems_paths(systems_paths)
        catalogs = ((root, perf_database._iter_database_version_paths(system, "vllm", version, root)) for root in roots)
        strict = perf_database._strict_provenance_enabled(None)
    else:
        return False
    for root, paths in catalogs:
        cells = [
            directory
            for directory, data_dir in paths
            if not perf_database._version_dir_unusable_for_request(
                directory, data_dir, strict=strict, cache_collection_meta=database is not None
            )
        ]
        if not cells:
            continue
        spec = database.system_spec if database is not None else perf_database.load_system_spec(system, [root])
        if spec.get("gpu"):
            framework = "vllm_xpu" if perf_database.is_xpu_system_spec(spec) else "vllm"
            images = {
                stock
                for directory in cells
                for stock in _gemma4_collection_images(Path(directory), version, framework, cache=database is not None)
            }
            return images == {True}
        break
    return False


class UnverifiedGemmaRuntimeError(ValueError):
    """The selected native database cannot identify the Gemma attention layout."""


def check_gemma4_adapter_runtime(
    document: dict,
    version: str | None,
    tp_size: int,
    *,
    database: perf_database.PerfDatabase | None = None,
    system: str | None = None,
    systems_paths: str | list[str] | None = None,
    validate_inherited_attention: bool = True,
) -> None:
    """Require an observed image when supported Transformers readers select different Gemma geometry."""
    release = common.parse_support_matrix_version(version)
    if release is None or release < Version("0.28.0rc1"):
        return
    decoder = document.get(decoder_config_key(document), document)
    if not isinstance(decoder, dict) or "per_layer_config" not in decoder:
        return
    names = document.get("architectures")
    if decoder.get("model_type") != "gemma4_text" and not (
        isinstance(names, list) and names[:1] == ["Gemma4ForConditionalGeneration"]
    ):
        return
    if release >= Version("0.32.0rc0"):
        # All supported readers have per-layer geometry, but only Transformers 5.17+
        # inherits default_rope_type. An explicit global value removes the difference.
        overrides = decoder.get("per_layer_config")
        if (
            "default_rope_type" in decoder
            or not isinstance(overrides, dict)
            or not any(isinstance(row, dict) and "default_rope_type" in row for row in overrides.values())
        ):
            return
    tp_size = validate_parallel_size("tp_size", tp_size)
    try:
        gemma4_cache_shapes(
            decoder,
            "vllm",
            version,
            agreement_tp_size=tp_size,
            compare_adapter_readers=True,
            validate_inherited_attention=validate_inherited_attention,
        )
        return
    except ValueError:
        # An invalid per-layer layout still fails even with a known stock runtime.
        gemma4_cache_shapes(
            decoder,
            "vllm",
            version,
            agreement_tp_size=tp_size,
            validate_inherited_attention=validate_inherited_attention,
        )

    try:
        if _has_stock_gemma4_collection(version, database, system, systems_paths):
            return
    except (OSError, ValueError, csv.Error) as error:
        if database is not None:
            raise UnverifiedGemmaRuntimeError(str(error)) from error
        raise
    message = f"vLLM {version} Gemma 4 per_layer_config needs a stock collection or a runtime profile"
    if database is not None:
        raise UnverifiedGemmaRuntimeError(message)
    raise ValueError(message)


@register_model("GEMMA4MIX")
class Gemma4MixModel(BaseModel):
    """
    Google Gemma 4 (gemma4_text): hybrid SWA/global attention + shared dense MLP
    running in parallel with routed top-k MoE on every layer.

    Two layer-type recipes are emitted, driven by ``Gemma4MixConfig.layer_types``:

    - **sliding_attention (SWA)**: 16 Q heads x ``swa_head_dim``, ``swa_num_kv_heads``
      KV heads, separate K and V projections (standard GQA), token window =
      ``sliding_window_size``.
    - **full_attention (global)**: 16 Q heads x ``global_head_dim``, ``global_num_kv_heads``
      KV heads, no window. When ``attention_k_eq_v`` is set, V is reused from the K
      projection output -- there is no v_proj weight or v_proj GEMM on these layers,
      but K and V are still distinct tensors in the KV cache (post-norm/post-RoPE).

    Every layer (gated on global ``enable_moe_block`` at parse time) runs both:

    - a shared dense MLP at intermediate_size ``inter_size``, gated SwiGLU; and
    - a routed top-k MoE at expert intermediate ``moe_inter_size`` with ``num_experts``
      experts and a router GEMM (the router GEMM is emitted only when num_experts >= 128,
      mirroring the MoEModel/HybridMoEModel convention).

    Outputs of the two FFN branches are summed before the post-feedforward norm.
    """

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        # Dense SWA/global GQA prefill CP: SGLang AllGather (zigzag FMHA).
        return backend_name == "sglang"

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        model = cls(
            model_info["topk"],
            model_info["num_experts"],
            model_info["moe_inter_size"],
            model_info["model_path"],
            model_info["model_family"],
            model_info["architecture"],
            model_info["layers"],
            model_info["n"],
            model_info["n_kv"],
            model_info["d"],
            model_info["hidden_size"],
            model_info["inter_size"],
            model_info["vocab"],
            model_info["context"],
            model_config,
            backend_name=backend_name,
            raw_config=model_info["raw_config"],
            allow_checkpoint_split=not model_info["gemm_quant_mode_is_explicit"],
        )
        cfg = model_info["extra_params"]
        raw = model_info["raw_config"]
        text = raw.get("text_config", raw)
        version = model_info.get("backend_version")
        if backend_name == "vllm":
            check_gemma4_adapter_runtime(
                raw,
                version,
                model_config.tp_size,
                database=model_info.get("database"),
                validate_inherited_attention=False,
            )
        (swa_heads, swa_dim), (full_heads, full_dim) = gemma4_cache_shapes(
            text,
            backend_name,
            version,
            agreement_tp_size=model_config.tp_size,
            # The SDK model parser accepts inherited window/chunk settings. Config-only
            # estimates also validate them against the selected serving constructor.
            validate_inherited_attention=False,
        )
        cfg = replace(
            cfg,
            swa_num_kv_heads=swa_heads,
            swa_head_dim=swa_dim,
            global_num_kv_heads=full_heads,
            global_head_dim=full_dim,
        )
        model.set_gemma4_config(cfg)
        return model

    def __init__(
        self,
        topk: int,
        num_experts: int,
        moe_inter_size: int,
        *args,
        backend_name: str = "",
        raw_config: dict | None = None,
        allow_checkpoint_split: bool = True,
    ) -> None:
        super().__init__(*args)
        exclusions = tuple(str(pattern).lower() for pattern in quant_exclude_patterns(raw_config or {}))
        self._projection_gemm_quant_mode = self.config.gemm_quant_mode
        self._shared_mlp_gemm_quant_mode = self.config.gemm_quant_mode
        self._routed_moe_quant_mode = self.config.moe_quant_mode
        if allow_checkpoint_split:
            if any("self_attn" in pattern for pattern in exclusions):
                self._projection_gemm_quant_mode = common.GEMMQuantMode.bfloat16
            if any(".mlp" in pattern for pattern in exclusions):
                self._shared_mlp_gemm_quant_mode = common.GEMMQuantMode.bfloat16
                self._routed_moe_quant_mode = common.MoEQuantMode.bfloat16
        # Framework backend, threaded to MoEDispatch (its comm flavor is a
        # constructor field since the pyo3 op unification).
        self._backend_name = backend_name
        # Gemma 4 family includes both MoE variants (e.g. gemma-4-26B-A4B-it,
        # topk=8/num_experts=128) and dense variants (e.g. gemma-4-31B-it,
        # gemma-4-E2B-it, gemma-4-E4B-it: topk=0/num_experts=None). For dense
        # variants every layer is just the shared dense MLP -- there is no
        # routed-MoE block, so MoE-related parallelism constraints don't apply.
        self._is_dense = not topk or not num_experts
        if not self._is_dense:
            assert (
                self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
                == self.config.moe_tp_size * self.config.moe_ep_size
            ), (
                f"tp_size ({self.config.tp_size}) * attention_dp_size "
                f"({self.config.attention_dp_size}) * cp_size ({self.config.cp_size}) should be equal to "
                f"moe_tp_size ({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
            )
            assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"
        else:
            # Dense Gemma 4: collapse the MoE search-space to ep=1 so pareto
            # iteration doesn't enumerate equivalent dense configurations.
            assert self.config.moe_ep_size == 1, (
                f"dense Gemma 4 variants require moe_ep_size=1, got {self.config.moe_ep_size}"
            )
        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
        self._gemma4_config: common.Gemma4MixConfig | None = None
        # These run during LM prefill but use one independent vision-token
        # block per image, rather than the full text+vision sequence shape.
        self.visual_context_ops: list = []
        self._power_law_alpha = 1.01

    def set_gemma4_config(self, cfg: common.Gemma4MixConfig) -> None:
        """Apply Gemma4MixConfig and rebuild text plus optional vision ops.

        Validates that ``layer_types`` length matches ``num_layers`` and contains only
        recognized values before accepting the config.
        """
        if cfg is None or not isinstance(cfg, common.Gemma4MixConfig):
            raise ValueError(f"Gemma4MixModel requires a Gemma4MixConfig, got {type(cfg).__name__}")
        if len(cfg.layer_types) != self._num_layers:
            raise ValueError(
                f"Gemma4MixConfig.layer_types length ({len(cfg.layer_types)}) "
                f"does not match num_layers ({self._num_layers})"
            )
        for i, lt in enumerate(cfg.layer_types):
            if lt not in ("sliding_attention", "full_attention"):
                raise ValueError(f"Gemma4MixConfig layer {i} has invalid type {lt!r}")
        self._gemma4_config = cfg
        self._build_context_ops()
        self._build_generation_ops()
        self.visual_context_ops = []
        self.encoder_ops = []
        self.encoder_config = cfg.vision_config
        if cfg.vision_config is not None:
            # A language-only worker still attends to visual embeddings, but
            # the vision tower itself is hosted on the encoder worker.
            if not self.config.language_only:
                self.encoder_ops.extend(
                    build_gemma4_vision_encoder_ops(
                        cfg.vision_config,
                        self.config.tp_size,
                        self.config.enable_encoder_dp,
                    )
                )
            if cfg.use_bidirectional_vision_attention:
                # The ordinary SWA ContextAttention already accounts for the
                # causal half of each visual block. Gemma's blockwise overlay
                # additionally unmasks the upper triangle for every SWA layer.
                # BaseBackend supplies the per-image pooled length; the op
                # scales Gemma's collected causal curve to the added pairs.
                # Independent image blocks use an average layer split under CP;
                # unlike the language path they do not use zigzag representative-rank sizing.
                # The compiled kernel-only evaluator omits fused RoPE and KV
                # writes already charged by the ordinary language graph.
                # Mask source: Transformers commit
                # cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55.
                d = self._resolve_dims(self.config.tp_size)
                self.visual_context_ops.append(
                    ops.ContextAttention(
                        "context_swa_visual_block_attention",
                        self._count_layer_types()["swa"] / self.config.cp_size,
                        self._num_heads // self.config.tp_size,
                        d["swa_n_kv_per_gpu"],
                        self.config.kvcache_quant_mode,
                        self.config.fmha_quant_mode,
                        window_size=cfg.sliding_window_size,
                        head_size=cfg.swa_head_dim,
                    )
                )
        if self.config.cp_size > 1:
            # decode never runs CP. Route the generation MoEDispatch ops to their
            # decode-CP comm path (pre=0 / post=all_reduce) rather than prefill's
            # all_gather/reduce_scatter -- attn_cp_size>1 + is_context=False. The
            # context loop in _build_context_ops handles the prefill side.
            cp = self.config.cp_size
            for op in self.generation_ops:
                if isinstance(op, ops.MoEDispatch):
                    op._attn_cp_size = cp
                    op._is_context = False

    def _count_layer_types(self) -> dict[str, int]:
        cfg = self._gemma4_config
        return {
            "swa": cfg.layer_types.count("sliding_attention"),
            "global": cfg.layer_types.count("full_attention"),
        }

    def _resolve_dims(self, tp_size: int) -> dict:
        """Resolve per-layer-type attention dims and dense-MLP intermediate per TP shard."""
        cfg = self._gemma4_config
        swa_n_kv = cfg.swa_num_kv_heads
        global_n_kv = cfg.global_num_kv_heads
        swa_hd = cfg.swa_head_dim
        global_hd = cfg.global_head_dim
        swa_n_kv_per_gpu = (swa_n_kv + tp_size - 1) // tp_size
        global_n_kv_per_gpu = (global_n_kv + tp_size - 1) // tp_size

        # QKV-projection output width. SWA always has separate K and V buffers.
        swa_qkv_out = self._num_heads * swa_hd // tp_size + swa_n_kv_per_gpu * swa_hd * 2
        # Global: when attention_k_eq_v, V reuses K's projection output -> only Q + K.
        if cfg.attention_k_eq_v:
            global_qkv_out = self._num_heads * global_hd // tp_size + global_n_kv_per_gpu * global_hd
        else:
            global_qkv_out = self._num_heads * global_hd // tp_size + global_n_kv_per_gpu * global_hd * 2

        return {
            "swa_n_kv_per_gpu": swa_n_kv_per_gpu,
            "global_n_kv_per_gpu": global_n_kv_per_gpu,
            "swa_qkv_out": swa_qkv_out,
            "global_qkv_out": global_qkv_out,
            "swa_proj_in": self._num_heads * swa_hd // tp_size,
            "global_proj_in": self._num_heads * global_hd // tp_size,
            "swa_hd": swa_hd,
            "global_hd": global_hd,
            "dense_inter_per_tp": self._inter_size // tp_size,
        }

    def _shared_mlp_ops(self, prefix: str, count: float, h: int, dense_inter_per_tp: int) -> list:
        """Shared dense MLP (Gemma4TextMLP): gated SwiGLU. Runs on every layer."""
        gemm_q = self._shared_mlp_gemm_quant_mode
        return [
            ops.GEMM(f"{prefix}_shared_mlp_gate_up_gemm", count, 2 * dense_inter_per_tp, h, gemm_q),
            ops.ElementWise(f"{prefix}_shared_mlp_act", count, 2 * dense_inter_per_tp, dense_inter_per_tp, 0.8),
            ops.GEMM(f"{prefix}_shared_mlp_down_gemm", count, h, dense_inter_per_tp, gemm_q, low_precision_input=True),
        ]

    def _moe_ops(
        self,
        prefix: str,
        count: float,
        h: int,
        moe_tp: int,
        moe_ep: int,
        attn_dp: int,
        moe_q: common.MoEQuantMode,
        wl_dist: str,
    ) -> list:
        """Routed-MoE ops: router GEMM (when num_experts ≥ 128), pre-dispatch, MoE, post-dispatch.

        Returns an empty list for dense Gemma 4 variants (no routed-MoE block).
        """
        if self._is_dense:
            return []
        router_ops = (
            [ops.GEMM(f"{prefix}_router_gemm", count, self._num_experts, h, common.GEMMQuantMode.bfloat16)]
            if self._num_experts >= 128
            else []
        )
        return router_ops + [
            ops.MoEDispatch(
                f"{prefix}_moe_pre_dispatch",
                count,
                h,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                attn_dp,
                True,
                quant_mode=moe_q,
                backend=self._backend_name,
            ),
            ops.MoE(
                f"{prefix}_moe",
                count,
                h,
                self._moe_inter_size,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                moe_q,
                wl_dist,
                attn_dp,
                moe_kernel_source=self.config.moe_kernel_source,
            ),
            ops.MoEDispatch(
                f"{prefix}_moe_post_dispatch",
                count,
                h,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                attn_dp,
                False,
                quant_mode=moe_q,
                backend=self._backend_name,
            ),
        ]

    def _build_context_ops(self) -> None:
        if not self._gemma4_config:
            return

        cfg = self._gemma4_config
        counts = self._count_layer_types()
        h = self._hidden_size
        tp = self.config.tp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        pp = self.config.pp_size
        gemm_q = self._projection_gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        fmha_q = self.config.fmha_quant_mode
        moe_q = self._routed_moe_quant_mode
        wl_dist = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        d = self._resolve_dims(tp)

        self.context_ops = [ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3)]

        # --- sliding_attention layers ---
        if counts["swa"] > 0:
            c = counts["swa"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_swa_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_swa_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                    ),
                    ops.GEMM("context_swa_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("context_swa_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._shared_mlp_ops("context_swa", c, h, d["dense_inter_per_tp"])
                + self._moe_ops("context_swa", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        # --- full_attention (global) layers ---
        if counts["global"] > 0:
            c = counts["global"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_global_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_global_qkv_gemm", c, d["global_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        d["global_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=0,
                        head_size=d["global_hd"],
                    ),
                    ops.GEMM("context_global_proj_gemm", c, h, d["global_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("context_global_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._shared_mlp_ops("context_global", c, h, d["dense_inter_per_tp"])
                + self._moe_ops("context_global", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        self.context_ops.extend(
            [
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("context_p2p", pp - 1, h, pp),
            ]
        )

        # cp (SGLang prefill AllGather CP). Gemma4 has heterogeneous KV per
        # layer-type (SWA vs global) -> emit one NCCL all_gather per type,
        # weighted by its layer count, so total comm matches the runtime.
        # Dense FMHA uses zigzag (``cp_size`` on the attention op, balanced
        # full/cp work); token-major ops shrink the M-axis via ``seq_split``
        # (DB lookup at per-rank M). This bypasses the BaseModel CP helper,
        # which assumes one uniform per-token KV size.
        # NOTE: the SWA all_gather is sized by the full new-token count (not the
        # window) on purpose -- this matches sglang v0.5.13
        # ``cp_allgather_and_save_kv_cache`` / ``cp_all_gather_rerange_kv_cache``,
        # which gather the FULL per-layer new-token KV across CP ranks; the
        # sliding window only caps the SWA write target / stored KV (handled in
        # get_kvcache_bytes_per_sequence), not this per-layer comm volume.
        if self.config.cp_size > 1:
            cp = self.config.cp_size
            kvcache_bytes = self.config.kvcache_quant_mode.value.memory
            comm_bytes = self.config.comm_quant_mode.value.memory
            # Post-construction CP wiring (not the __init__ _CP_AWARE gate): this
            # family has heterogeneous layer types (SWA vs global / dense vs MoE)
            # built across separate passes, so CP is applied here once every op
            # exists. The per-op _CP_AWARE opt-in is re-asserted in the loop so an
            # un-audited op still fails loud instead of silently skipping CP.
            for op in self.context_ops:
                if isinstance(op, ops.ContextAttention):
                    op._cp_size = cp
                elif isinstance(op, ops.MoEDispatch):
                    # MoEDispatch keys CP off attn_cp_size (AG pre / RS post),
                    # NOT seq_split; with moe_ep=cp its attention_tp_size>1 would
                    # otherwise wrongly take the TP all-reduce path.
                    op._attn_cp_size = cp
                elif op._CP_AWARE:
                    # Token-major op: shrink the M-axis. This post-construction
                    # mutation bypasses the constructor's _CP_AWARE gate, so
                    # re-assert the opt-in here -- an un-audited op in a
                    # CP-enabled pipeline must fail loud, not silently skip CP.
                    op._seq_split = cp
                else:
                    raise NotImplementedError(
                        f"{type(op).__name__} ('{op._name}') has not been audited for "
                        f"context parallelism but appears in a CP-enabled context pipeline."
                    )
            for ctx_key, n_kv_per_gpu, head_dim in (
                ("swa", d["swa_n_kv_per_gpu"], d["swa_hd"]),
                ("global", d["global_n_kv_per_gpu"], d["global_hd"]),
            ):
                if counts[ctx_key] <= 0:
                    continue
                kv_bytes_per_token = n_kv_per_gpu * head_dim * 2 * kvcache_bytes
                self.context_ops.append(
                    ops.NCCL(
                        f"context_cp_all_gather_{ctx_key}",
                        counts[ctx_key],
                        "all_gather",
                        num_elements_per_token=kv_bytes_per_token / comm_bytes,
                        num_gpus=cp,
                        comm_quant_mode=self.config.comm_quant_mode,
                    )
                )

    def _build_generation_ops(self) -> None:
        if not self._gemma4_config:
            return

        cfg = self._gemma4_config
        counts = self._count_layer_types()
        sf = self._mtp_scale_factor
        h = self._hidden_size
        tp = self.config.tp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        pp = self.config.pp_size
        gemm_q = self._projection_gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        moe_q = self._routed_moe_quant_mode
        wl_dist = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        d = self._resolve_dims(tp)

        self.generation_ops = [ops.Embedding("generation_embedding", 1 * sf, self._vocab_size, h, 0.3)]

        # --- sliding_attention layers ---
        if counts["swa"] > 0:
            c = counts["swa"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_swa_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_swa_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        self._num_heads // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                    ),
                    ops.GEMM("generation_swa_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("generation_swa_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._shared_mlp_ops("generation_swa", c, h, d["dense_inter_per_tp"])
                + self._moe_ops("generation_swa", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        # --- full_attention (global) layers ---
        if counts["global"] > 0:
            c = counts["global"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_global_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_global_qkv_gemm", c, d["global_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        self._num_heads // tp,
                        d["global_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=0,
                        head_size=d["global_hd"],
                    ),
                    ops.GEMM(
                        "generation_global_proj_gemm", c, h, d["global_proj_in"], gemm_q, low_precision_input=True
                    ),
                    ops.ElementWise("generation_global_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._shared_mlp_ops("generation_global", c, h, d["dense_inter_per_tp"])
                + self._moe_ops("generation_global", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        self.generation_ops.extend(
            [
                ops.GEMM("generation_logits_gemm", 1 * sf, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("generation_p2p", (pp - 1) * sf, h, pp),
            ]
        )

    def get_kvcache_elements_per_token(self) -> int:
        """Per-token KV-cache element count (per GPU), summed over all layers.

        Both K and V tensors are stored even on global layers where K=V at the
        projection — by the time the cache is written, K has had RoPE applied and V
        has not, so they are distinct tensors.

        This count does NOT account for the SWA window cap; callers that need a
        sequence-length-aware byte count should use ``get_kvcache_bytes_per_sequence``.
        """
        if not self._gemma4_config:
            return super().get_kvcache_elements_per_token()
        cfg = self._gemma4_config
        tp = self.config.tp_size
        swa_kv_per_gpu = (cfg.swa_num_kv_heads + tp - 1) // tp
        global_kv_per_gpu = (cfg.global_num_kv_heads + tp - 1) // tp
        num_swa = cfg.layer_types.count("sliding_attention")
        num_global = cfg.layer_types.count("full_attention")
        return 2 * (num_swa * swa_kv_per_gpu * cfg.swa_head_dim + num_global * global_kv_per_gpu * cfg.global_head_dim)

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """KV-cache bytes for one sequence on one GPU.

        SWA layers cap at ``sliding_window_size`` tokens; global layers grow with
        ``seq_len``. This per-layer-type math is what makes Gemma 4's 256K-context
        memory footprint tractable.
        """
        if not self._gemma4_config:
            return super().get_kvcache_bytes_per_sequence(seq_len)
        seq_len = max(0, seq_len)
        cfg = self._gemma4_config
        bytes_per_elem = self.config.kvcache_quant_mode.value.memory
        tp = self.config.tp_size
        swa_kv_per_gpu = (cfg.swa_num_kv_heads + tp - 1) // tp
        global_kv_per_gpu = (cfg.global_num_kv_heads + tp - 1) // tp
        num_swa = cfg.layer_types.count("sliding_attention")
        num_global = cfg.layer_types.count("full_attention")
        swa_seq = min(seq_len, cfg.sliding_window_size) if cfg.sliding_window_size > 0 else seq_len
        swa_bytes = num_swa * swa_kv_per_gpu * cfg.swa_head_dim * 2 * bytes_per_elem * swa_seq
        global_bytes = num_global * global_kv_per_gpu * cfg.global_head_dim * 2 * bytes_per_elem * seq_len
        return float(swa_bytes + global_bytes)

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Capacity inverse over the window-capped KV curve (non-linear past the window)."""
        if not self._gemma4_config:
            return super().get_kvcache_max_tokens(kv_budget_bytes)
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)
