<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notices

AISim contains source material derived from the projects identified below.
These notices apply only to the identified third-party material. Dependencies
installed separately by package managers are governed by the license material
distributed with those packages.

Notices for model-configuration files removed from release 0.14.0 are retained conservatively for historical attribution. The current `scripts/ci/bundled_configs.json` inventory identifies the configurations included in the source tree and validated wheel. Retained notices do not mean a removed configuration is still packaged.

Unless otherwise stated, AISim file paths in this document are relative
to `python/aisim/` in the repository source tree.

## AIPerf AgentX snapshot oracle

The reference sampling vectors and recycling contract in repository-root
`crates/core/src/replay/loadgen/snapshot_tests.rs` and
`docs/replay/workloads.md`, plus the invocation contract in
`scripts/prediction_regression/export_agentx_snapshots.py`, are derived from
`src/aiperf/timing/trajectory_source.py` and
`src/aiperf/timing/strategies/agentic_replay.py` in SemiAnalysisAI/aiperf at
commit `754356e9a39acc6cc6afb242d123bb57c3fb6f75`:
https://github.com/SemiAnalysisAI/aiperf/tree/754356e9a39acc6cc6afb242d123bb57c3fb6f75

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under Apache-2.0. The upstream license is at
https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/LICENSE
and the full license is included in this repository as `LICENSE`.
Modified by NVIDIA: executed the pinned sampler on a small synthetic corpus
to freeze expected cuts, and documented its sequential recycling behavior.
No upstream implementation is vendored; AISim consumes explicit cuts.

The live warmup policy in repository-root
`crates/core/src/replay/loadgen/phase.rs` and
`crates/core/src/replay/loadgen/driver.rs`, its regression tests in those files,
and `docs/replay/agentic/warmup.md`, `docs/replay/agentic/continuous-profiles.md`
and `docs/replay/workloads.md` are modified re-implementations of the same
pinned `agentic_replay.py` behavior: globally aligned primers, per-lane
admission quotas, compressed dependency waits, shared-cursor recycling, and
source-clock frontier handoff. NVIDIA modified these algorithms to use the
native deterministic dependency executor and its transactional server-settlement
barrier. The original NVIDIA copyright and Apache-2.0 license apply; no
upstream serving implementation is vendored.

The continuous-profile release rules in the same `driver.rs`, their regression
tests there, and the release descriptions in
`docs/replay/agentic/continuous-profiles.md` and `docs/replay/workloads.md` are
modified re-implementations of the pinned `agentic_replay.py`,
`src/aiperf/timing/branch_orchestrator.py`,
`src/aiperf/timing/replay_dependencies.py` and
`src/aiperf/common/loop_scheduler.py` behavior: dependency-relative release of
continuations, spawns and joins, joins gated by the parent's own replay
deadline, detected chains released when their replay barrier clears, and
uniform idle shifts of every pending replay timer, repeated when a shift lands
on a request still held by its replay barrier. NVIDIA modified these rules to
operate on the native dependency graph. The original NVIDIA copyright and
Apache-2.0 license apply; no upstream implementation is vendored.

## AIConfigurator

The repository-root `.coderabbit.yaml` is adapted and modified from
AIConfigurator's `.coderabbit.yaml` at commit
`13b5cf2697876692b0a52098266c81162add11fc`. The original imported policy is
preserved at `python/aisim/.coderabbit.yaml` as migration provenance.

Upstream source:
https://github.com/ai-dynamo/aiconfigurator/blob/13b5cf2697876692b0a52098266c81162add11fc/.coderabbit.yaml

Pinned upstream collection and preserved-file SHA-256:
`5fb7a61a53f71f476169fa8e2419d3073c8d7e96206d8986b7d4fbb0f11fbcdd`

AISim-modified root overlay SHA-256:
`a70a21425533b419feaafcec67b176b3048a0893c4833bb802f46dc6e9a8f57e`

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This material is licensed under the Apache License 2.0. The upstream license
at the identified revision is available at:
https://github.com/ai-dynamo/aiconfigurator/blob/13b5cf2697876692b0a52098266c81162add11fc/LICENSE

The 18 B200 TensorRT-LLM 1.3.0rc20 performance tables under
`src/aisim_core/systems/data/b200_sxm/*/trtllm/1.3.0rc20/` include
power measurements derived from AIConfigurator commit
`915f590680d8a79fe9c39f6f3a9ff13bc267fcce` (PR #1584). Fourteen tables remain
unmodified, byte-identical copies after the September 17-18 refresh. The
context-MLA table was replaced with locally collected measurements, and the
MoE table has locally refreshed MXFP4 rows with unavailable-power sentinels.
The context-attention and
generation-attention tables are modified derivatives: AISim preserves
newer local timing rows and adds the typed `0.0` / `0.0` unavailable sentinel
to those local-only identities. Source paths, row counts, measured coverage,
and merge details are recorded in
`src/aisim_core/systems/data/b200_sxm/README.md`. The two unmodified
upstream attention copies under
`src/aisim_core/systems/data/b200_sxm/power_upstream/` support focused
import regression tests, which pin source and packaged SHA-256 digests.

The corresponding energy expectations in the repository-root file
`crates/core/parity_tests/perfmodel/goldens/per_op.json` are modified generated
derivatives of those measurements. AISim's native pinning workflow at
commit `36dcc8f3afe9e6e2e9de976737b6337fad8c4d74` produced the two case updates;
the adjacent parity README records the reviewed energy-only delta.

Upstream source:
https://github.com/ai-dynamo/aiconfigurator/tree/915f590680d8a79fe9c39f6f3a9ff13bc267fcce/aic-core/src/aiconfigurator_core/systems/data/b200_sxm

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This material is licensed under the Apache License 2.0. The upstream license
at the identified revision is available at:
https://github.com/ai-dynamo/aiconfigurator/blob/915f590680d8a79fe9c39f6f3a9ff13bc267fcce/LICENSE

## Dynamo V4.1 FPM collection adapter

`collector/fpm_forward/runtime/dsv41/dsv41_scheduler.py` is modified code
adapted from `components/src/dynamo/vllm/instrumented_scheduler.py` in
https://github.com/ai-dynamo/dynamo/tree/54960177085413259859c88bd34ed0734d4c2ea9.
It adds bounded same-request real-KV collection while preserving the native
benchmark and FPM contracts. Copyright (c) 2025-2026 NVIDIA CORPORATION &
AFFILIATES. All rights reserved. Licensed under Apache-2.0; the upstream
license is preserved in the adapter's adjacent `LICENSE`. The adjacent README
records the inspected vLLM API revision and immutable runtime image/source
hashes. vLLM implementation files are not vendored. The text fixture and
lifecycle tests are original work for this change, with no external corpus.

## Dynamo GLM-5.3-Flash default-serving (prefix-cache) FPM adapter

`python/aisim/collector/fpm_forward/runtime/glm53flash/glm53flash_prefix_scheduler.py`
is modified code adapted from `components/src/dynamo/vllm/instrumented_scheduler.py`
in https://github.com/ai-dynamo/dynamo/tree/54960177085413259859c88bd34ed0734d4c2ea9
(benchmark state machine on the native scheduler). Its prefix-cache real-seed staging
(seed shot, untimed warm shot, validated measured shot; `_bench_realseed_stage_point` /
`_bench_realseed_pending_step`) is ported from the same path at
https://github.com/ai-dynamo/dynamo/tree/99dae1f53e5a0534c274223ff9daa97f0b1fc2ea, the revision
shipped as ai-dynamo 1.5.0.dev20260917 in
`lmsysorg/sglang@sha256:b0d8718a4424bb22e448e04407ab3ce5f7399a4c5fc702d6fbe36c3772ec8862`
(file SHA256 `76cdd291e92c3c41f259279c316bcdedcc6a9aa615ad61e29ca6997105950047`, git blob
`cbc2a12947ce18304721ac0549274fbeac51ddef`). Its queued-request classification for vLLM
v0.31.0 (`_compute_queued` over `waiting` and `kv_holding_waiting`, replacing the removed
`skipped_waiting`) is ported from the same path at
https://github.com/ai-dynamo/dynamo/tree/395f02405c1dd1835ab73a39f7a5984ae5fd4552 (file SHA256
`06fcb26ffcfeb4d58a01704d8d7842d9d838d4a34170a1086f09e5617442da8e`, git blob
`c19f8174b2e232cf8c0f5a26da83b8085ef71776`). Changes: GLM-5.3-Flash
default serving state (prefix caching on, Mamba align mode), seeded token inputs, executed
per-request geometries on the 4-token IndexPool grid with recorded prefill chunk starts,
5+10 repetitions with native-FPM validation, and compact deferred evidence. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under Apache-2.0; the upstream license is preserved in the adjacent `LICENSE`.
No upstream root NOTICE exists. vLLM and SGLang implementation files are not vendored.

## NVIDIA AIConfigurator speculative decoding

The speculation SDK, compatibility exports, CLI/task integration, attention and whole-forward FPM operation changes, native bindings, and their tests are adapted and modified from AIConfigurator PR #1563, pinned at commit `6290c161a354da5250c391bd43372b2e9c6f4a51`. Original paths are under `aic-core/src/aiconfigurator_core/sdk/`, `src/aiconfigurator/`, `aic-core/rust/aiconfigurator-core/`, `aic-core/rust/tests/public-api/`, and `tests/`.

Derived AISim paths are under `python/aisim/src/aisim_core/sdk/`, `python/aisim/src/aisim/sdk/speculation/`, `python/aisim/src/aisim_core/sdk/speculation/`, `python/aisim/src/aisim/legacy_cli/`, `python/aisim/src/aisim/sdk/{speculative,task_v2}.py`, and `python/aisim/tests/`; repository-root Rust paths are under `crates/core/src/perfmodel/`, `crates/core/parity_tests/perfmodel/`, and `crates/tests/public-api/`. The repository's [synchronization provenance](https://github.com/ai-dynamo/aisim/blob/main/docs/ci/aic-sync.md#source-provenance) records import boundaries and links the current path mapping. Changes preserve AISim's current native contracts and strengthen configuration validation and regression coverage.

Upstream source:
https://github.com/ai-dynamo/aiconfigurator/tree/6290c161a354da5250c391bd43372b2e9c6f4a51

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. Licensed under Apache-2.0; the original license is at:
https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/LICENSE

This records provenance for NVIDIA-authored predecessor code, without implying ownership by an unaffiliated third party.

## NVIDIA AIConfigurator CI provenance

Selected CI definitions and a recommendation test were adapted from NVIDIA's
AIConfigurator repository at commit
`77fd0773407b3683d8a671fe24a30a7110651b64` and modified for AISim's
unified package layout and Fast CI / Full CI execution model:

- repository-root `.github/workflows/ci.yml` (selected jobs)
- repository-root `.github/actions/build-platform-wheel/action.yml`
- repository-root `.github/actions/setup-python-rust/action.yml`
- repository-root `.github/workflows/collector-check.yml`
- repository-root `.github/workflows/prediction-regression-gate.yml`
- repository-root `.github/workflows/validate-platform-wheels.yml`
- `tests/e2e/cli/test_cli_recommend.py`

Upstream source:
https://github.com/ai-dynamo/AIConfigurator/tree/77fd0773407b3683d8a671fe24a30a7110651b64

Copyright (c) NVIDIA CORPORATION & AFFILIATES.

AIConfigurator is licensed under the Apache License, Version 2.0. The full
Apache-2.0 license text is reproduced in `LICENSE`. This section records
cross-repository provenance for NVIDIA-authored predecessor code; it is not a
claim that AIConfigurator is owned by an unaffiliated third party.

## NVIDIA Dynamo Weka regression provenance

The `weka_seam_rekey_never_uses_a_future_parent` regression in repository-root
`crates/core/src/replay/loadgen/weka.rs` is adapted and modified from NVIDIA
Dynamo's `lib/mocker/src/loadgen/tests.rs` at immutable commit
`b113ceae74da3f6754429ade750b0f5b0c8de5d6`. The adaptation uses AISim's public
graph loader and adds assertions that the later continuation was selected while
the earlier fork retains its original parent.

Upstream source:
https://github.com/ai-dynamo/dynamo/blob/b113ceae74da3f6754429ade750b0f5b0c8de5d6/lib/mocker/src/loadgen/tests.rs

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This material is licensed under the Apache License, Version 2.0. The upstream
license at the identified revision is available at:
https://github.com/ai-dynamo/dynamo/blob/b113ceae74da3f6754429ade750b0f5b0c8de5d6/LICENSE

The full Apache-2.0 license text is reproduced in `LICENSE`. This section
records provenance for NVIDIA-authored predecessor code; it is not a claim
that Dynamo is owned by an unaffiliated third party.

## SGLang feed-forward/decode composition and quantization exclusions

The dense-prefix composition and exclusion resolver in `src/aisim_core/sdk/models/deepseek_v32.py` and their regression fixtures in `tests/unit/sdk/models/test_deepseek_v32_dense.py` and `tests/unit/sdk/models/test_large_ep_model_graphs.py` adapt and modify the tensor-parallel communication and packed-linear selection behavior from SGLang revision `02c5a855aceb968c310e6fbc6632270e26edc84b`. Original source paths are `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, and `python/sglang/srt/layers/quantization/utils.py`. The adaptation models operator composition and projection precision without importing the serving runtime or its GPU dependencies.

The vLLM copyright in `src/aisim_core/sdk/models/deepseek_v32.py` is retained from those SGLang sources. At the pinned SGLang revision, the header of `python/sglang/srt/models/deepseek_v2.py` identifies its own upstream as [`vllm/model_executor/models/deepseek_v2.py` at vLLM commit `fb6af8bc086328ca6659e72d11ffd4309ce4de22`](https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py), licensed under Apache-2.0. This records the inherited attribution chain; the SGLang revision above is the source adapted for AISim.

Upstream source:
https://gitlab-master.nvidia.com/dl/sglang/sglang/-/tree/02c5a855aceb968c310e6fbc6632270e26edc84b/python/sglang/srt

Copyright contributors to the vLLM project. Copyright 2023-2024 SGLang Team. These source files are licensed under the Apache License, Version 2.0. The upstream license is at:
https://gitlab-master.nvidia.com/dl/sglang/sglang/-/blob/02c5a855aceb968c310e6fbc6632270e26edc84b/LICENSE

## vLLM

The grouped-cache config derivation and metadata regression tests in
`src/aisim/support/config_profile.py`,
`tests/unit/test_support_config_profile.py`, and
`tests/unit/test_support_topology.py` are adapted and modified from vLLM's
Inkling cache layout, decoder state construction, and configuration defaults at
commit `98dff2a81d747d1dba01a47f939f48c3526d4206`:

- [`vllm/models/inkling/nvidia/sconv_swa_attn.py`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/models/inkling/nvidia/sconv_swa_attn.py)
- [`vllm/models/inkling/nvidia/model.py`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/models/inkling/nvidia/model.py)
- [`vllm/models/inkling/configs.py`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/models/inkling/configs.py)

Copyright contributors to the vLLM project.

This material is licensed under Apache-2.0; the upstream license at the
identified revision is available at:
https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/LICENSE

AISim modifies these contracts for metadata-only rank-local aggregate
cache sizing and validation, without constructing runtime tensors or a model.
The derived tests use original synthetic geometries and do not copy a
checkpoint fixture. The derived files retain the upstream copyright and
identify the modified source material.

The repository-root files `crates/core/src/engine/scheduler/vllm/core.rs`,
`crates/core/src/engine/kv_manager/{state_cache_manager,vllm_backend}.rs`,
`crates/core/src/engine/cache/vllm_block_pool.rs`,
`crates/core/src/engine/kv_manager/g1_manager.rs`,
and `crates/core/src/engine/kv_manager/state_cache_tests.rs` adapt the align-mode
prefill split, state-slot turnover/copy lifetimes and default checkpoint-retention
behavior from vLLM v0.29.0,
immutable commit `98dff2a81d747d1dba01a47f939f48c3526d4206`:

- https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/core/sched/scheduler.py
- https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/core/single_type_kv_cache_manager.py
- https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/core/kv_cache_manager.py
- https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/worker/mamba_utils.py

Copyright contributors to the vLLM project. Licensed under Apache-2.0:
https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/LICENSE

These are modified Rust simulation adaptations, restricted to no internal
prefill checkpoints, no speculative decoding and no periodic retention.
The identified upstream revision has no root NOTICE file. The repository's
Apache-2.0 LICENSE supplies the license text; no upstream source file is vendored.


The inference-mode scope and MSA query-position metadata integration in
`collector/vllm/collect_mla_module.py` and
`collector/vllm/collect_msa_module.py` are adapted (modified) from serving
behavior at vLLM commit `dd10e03f95f94edbea1975c67ace3a35ec9a8a40`:

- `vllm/v1/worker/gpu_model_runner.py` (query positions, common attention metadata,
  and the inference-mode model execution boundary).
- `vllm/models/minimax_m3/nvidia/indexer_msa.py` (MSA positions metadata contract).
- `vllm/models/minimax_m3/nvidia/model.py` (versioned shared top-k buffer layout).

Upstream source:
https://github.com/vllm-project/vllm/tree/dd10e03f95f94edbea1975c67ace3a35ec9a8a40

Copyright contributors to the vLLM project. Licensed under Apache-2.0;
modifications adapt the serving contracts to synthetic collector batches.


The independently written Rust deferred-queue and post-lookup-touch behavior in
`crates/core/src/engine/scheduler/vllm/{core,host_offload}.rs`
and related G3 fixtures in `crates/core/src/replay/agg_tests.rs` reference vLLM at
immutable revision `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`:
[`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/core/sched/scheduler.py),
`vllm/v1/core/sched/request_queue.py` and `vllm/v1/kv_offload/tiering/manager.py`.
The upstream project is copyright contributors to the vLLM project and licensed
under Apache-2.0. These repository-root-relative files adapt behavioral contracts;
no upstream method bodies are copied. The Rust implementation and fixtures are
modified for simulation and do not reproduce native filesystem timing.

The speculative scheduling/progress corrections in repository-root files
`crates/core/src/engine/scheduler/vllm/{core,policy,tests}.rs` and
`crates/core/src/engine/common/speculative.rs` are independently written,
modified Rust simulation adaptations of behavioral contracts at vLLM commit
`b3563fc65ae0f5359802593d78e7ea097e1fed31`:

- https://github.com/vllm-project/vllm/blob/b3563fc65ae0f5359802593d78e7ea097e1fed31/vllm/v1/core/sched/scheduler.py
- https://github.com/vllm-project/vllm/blob/b3563fc65ae0f5359802593d78e7ea097e1fed31/vllm/v1/core/sched/async_scheduler.py

Copyright contributors to the vLLM project. Licensed under Apache-2.0:
https://github.com/vllm-project/vllm/blob/b3563fc65ae0f5359802593d78e7ea097e1fed31/LICENSE

No upstream method bodies or test fixtures are copied. The adaptation models
scheduled draft positions, first-output and preemption state in virtual time;
it does not reproduce asynchronous worker execution. That revision has no
root NOTICE file. The repository LICENSE contains the Apache-2.0 license text.

The async-scheduling admission in repository-root files
`crates/core/src/engine/scheduler/vllm/{core,tests}.rs` is an independently
written, modified Rust simulation adaptation of behavioral contracts at vLLM
tag `v0.28.0` (commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`):

- https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/engine/core.py
- https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/config/vllm.py

Copyright contributors to the vLLM project. Licensed under Apache-2.0:
https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/LICENSE

No upstream method bodies or test fixtures are copied. The adaptation models,
in virtual time, which committed batch an arriving request can join while the
engine core keeps a queue of `max_concurrent_batches`; it does not reproduce
asynchronous worker execution. That revision has no root NOTICE file.

The following files are derived from vLLM's attention test utilities at tag
`v0.11.0` (commit `b8b302cde434df8c9289a2b465406b47ebab1c2d`):

- `collector/vllm/utils.py`
- `collector/vllm/utils_xpu.py`

Upstream source:
https://github.com/vllm-project/vllm/blob/v0.11.0/tests/v1/attention/utils.py

The Gemma 4 visual-mask behavior and replicated multimodal adapter in
`src/aiconfigurator_core/sdk/models/gemma4.py` and
`src/aiconfigurator_core/sdk/models/blocks/vit.py` are adapted and modified
for performance modeling from vLLM at commit
`d2906091bfc579cebefe3d8e8fb9077397ce9882`:

https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/gemma4.py
https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/gemma4_mm.py

The Gemma 4 source preserves these upstream notices:

Copyright contributors to the vLLM project.
Copyright 2025 The vLLM team.
Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.

This material is licensed under the Apache License 2.0.

The Mistral3 projector topology modeled in
`src/aisim_core/sdk/models/blocks/vit.py` and parsed in
`src/aisim_core/sdk/utils.py`, and Pixtral image geometry and prompt-token
accounting in `src/aisim_core/sdk/backends/base_backend.py`, with regression
coverage in `tests/unit/sdk/models/test_mistral3.py`, are adapted and modified for
performance modeling from vLLM v0.24.0 at immutable commit
`ee0da84ab9e04ac7610e28580af62c365e898389`:

https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/model_executor/models/mistral3.py
https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/vllm/model_executor/models/pixtral.py

Copyright contributors to the vLLM project. This material is licensed under
the Apache License 2.0. The upstream license is available at:

https://github.com/vllm-project/vllm/blob/ee0da84ab9e04ac7610e28580af62c365e898389/LICENSE

The Kimi K2.5 and Kimi K3 vision-tower topology, pooled PatchMerger, and
PatchMergerV2 adapters modeled in
`src/aiconfigurator_core/sdk/models/blocks/vit.py` and parsed in
`src/aiconfigurator_core/sdk/utils.py`, with Kimi K3 rotary-grid validation in
`src/aiconfigurator_core/sdk/backends/base_backend.py` and regression derivatives in
`tests/unit/sdk/models/test_kimi_k25_vision.py` and
`tests/unit/sdk/models/test_kimi_k3_vision.py`, are modified adaptations of the
MoonViT3D implementation that Kimi K3 reuses from Kimi K2.5 in vLLM at commit
`d2906091bfc579cebefe3d8e8fb9077397ce9882`:

- https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/kimi_k25.py
- https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/kimi_k25_vit.py
- https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/layers/quantization/modelopt.py

The upstream license at that revision is available at:
https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/LICENSE

The Llama 4 encoder operation topology in
`src/aiconfigurator_core/sdk/models/blocks/vit.py` is adapted (modified) from vLLM's Llama 4
implementation at tag `v0.8.5` (commit
`ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`):

https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/vllm/model_executor/models/mllama4.py

Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team. All rights reserved.

The upstream license at that revision is available at:
https://github.com/vllm-project/vllm/blob/ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d/LICENSE

Copyright contributors to the vLLM project.

The vLLM `LICENSE` file at commit
`b8b302cde434df8c9289a2b465406b47ebab1c2d` is reproduced verbatim below:

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## DeepEP

The following files are derived from DeepEP test utilities at commit
`73b6ea4a439ba03a695563f9fd242c8e4b02b37c` and contain NVIDIA modifications:

- `collector/wideep/sglang/deepep/test_internode.py`
- `collector/wideep/sglang/deepep/test_intranode.py`
- `collector/wideep/sglang/deepep/test_low_latency.py`
- `collector/wideep/sglang/deepep/utils.py`

The patch `collector/wideep/vllm/patches/deepep_73b_nvl4.patch` is also a
modification of DeepEP source at that commit.

Upstream source:
https://github.com/deepseek-ai/DeepEP/tree/73b6ea4a439ba03a695563f9fd242c8e4b02b37c

```text
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SGLang DeepSeek-V4.1 serving contracts

The V4.1 execution and performance-model logic in
`src/aisim_core/sdk/deepseek_v41.py`, `sdk/models/deepseek_v41.py`
(with the same `src/aisim_core/` prefix), and repository-root
`crates/core/src/perfmodel/operators/dsv41.rs` is informed by and modified from
SGLang's serving architecture at immutable commit
`1aa0e962b206102b7c439a4a0c4981cfec6e87bc`:

- `python/sglang/srt/models/deepseek_v4.py` and `deepseek_v2.py`
- `python/sglang/srt/layers/engram.py`
- `python/sglang/srt/layers/attention/dsv4/compressor.py`
- `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py`
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py`
- `python/sglang/kernels/ops/attention/dsv4/sm90_fp4_indexer.py`
- `python/sglang/benchmark/one_batch.py`

The original integration adapter
`collector/sglang/dsv41_native_runner.py` calls that pinned benchmark's model
builder and request lifecycle. Its component boundaries are modified from
the serving contracts above; it does not copy framework metadata builders.
The original attention-only adapter `collector/sglang/dsv41_attention_runner.py`,
its `tests/unit/collector/test_dsv41_attention.py` fixtures and the attention
section of `collector/sglang/README.dsv41.md` also adapt those API boundaries
(modified), including native dummy-loader post-load order from
`python/sglang/srt/model_loader/loader.py` and random initialization from
`python/sglang/srt/model_loader/weight_utils.py` at the same immutable revision.
They call native attention, pool and request builders; they do not copy those
implementations or represent the isolated stack as the complete model.
The matching loaded-dimension guards in `collector/sglang/dsv41_contract.py`
and their CPU fixtures in `tests/unit/collector/test_dsv41_contract.py` are
modified analytical adaptations of the indexer layout in `dsv41_sparse.py`.
The SM90 index-score arithmetic and query-width formulas in
`crates/core/src/perfmodel/operators/dsv41.rs`, their Rust regression tests,
and `docs/perf-model/models/deepseek-v41.md` are independently expressed, modified
analytical adaptations of those same pinned BF16 indexer contracts.

The measured operator databases and adjacent documentation under
`src/aisim_core/systems/profiles/dsv41/` contain AISim timings
and geometry adapted from the same serving contracts (modified). Their
README and adjacent provenance identify the immutable collection runtime
and source identities. The restored GB300 tables preserve their historical
timings and documented source-audited indexer metadata correction. The
B300 TP4 and TP2 tables are new native measurements with loaded-module dimension
validation; they do not inherit that historical correction.

The FPM table execution identities, geometry, collection sidecars and accompanying
README under `src/aisim_core/systems/profiles/dsv41_fpm/`, including
B300 TP2/TP4 and GB300 TP2 full/bounded measurements, are modified AISim
adaptations of the SGLang serving-contract paths and immutable commit
listed above (Copyright 2023-2024 SGLang Team and SGLang contributors,
Apache-2.0), and of `config.json`, `inference/model.py` and
`DeepSeek_V41_Tech_Report.pdf` from
`deepseek-ai/DeepSeek-V4.1-Flash@fb2764a5cf321eaa5070ca8f9e892818f477c16d`
(Copyright (c) 2023 DeepSeek, MIT; source and license below). The latencies are
new AISim measurements; these files contain no upstream model execution
code. The SGLang source revision identifies the serving-contract reference,
not the entire measured runtime image; the immutable image and captured source
identities remain collection provenance.

Source: https://github.com/sgl-project/sglang/tree/1aa0e962b206102b7c439a4a0c4981cfec6e87bc
Copyright 2023-2024 SGLang Team and SGLang contributors. Licensed under Apache-2.0; its terms are
reproduced in the repository `LICENSE`. These are analytical adaptations,
not a copy of the model execution implementation. The modified analytical
scoring/storage adaptations and their independently written regression cases
also appear in `python/aisim/tests/unit/sdk/models/test_deepseek_v41.py`,
Rust operator/spec unit tests, `docs/perf-model/models/deepseek-v41.md`. They distinguish candidate masking from scoring
and physical FlashMLA cache payload from logical FP4 values.

## SGLang 0.5.19 multimodal frontend and scheduler behavior

The native VL replay path models sgl-project/sglang v0.5.19 at immutable commit
`0bcd822377da7b5718e674eaf9c870d349424dd1`. Repository-root paths:

- `crates/core/src/engine/scheduler/sglang/{host_loop,vision,frontend}.rs` and
  their tests re-implement the observed behavior of `Scheduler.event_loop_overlap`,
  `run_batch`, `recv_requests`, `process_batch_result_prefill`,
  `_batch_encode_per_image_misses`, `MultiModalStaticCache`, `TokenizerManager`,
  `BaseMultimodalProcessor`, the scheduler's receive preparation
  (`managers/scheduler_components/request_receiver.py`) and the Rust multimodal workers'
  feature transport (`RustMmProcessor._use_feature_shm`) as performance models.
  They do not execute or copy the upstream implementation.
- `crates/core/src/engine/scheduler/sglang/core.rs` (disaggregated prefill) and
  `crates/core/src/replay/components/encoder.rs` likewise re-implement the
  observed behavior of `disaggregation/prefill.py` (`SchedulerDisaggregationPrefillMixin`)
  and of `disaggregation/encoder/{runtime,server,receiver,preprocessor}.py`
  (`EncoderScheduler`, `_assign_items_by_modality`, `zmq_to_scheduler`) as
  performance models, without copying the upstream implementation.
- `python/aisim/src/aisim/vl/collect/` instantiates the pinned
  upstream frontend objects unmodified on the serving host and times them from
  the outside; it does not copy or modify their implementations.

Source: https://github.com/sgl-project/sglang/tree/0bcd822377da7b5718e674eaf9c870d349424dd1
Copyright 2023-2024 SGLang Team and SGLang contributors. Licensed under Apache-2.0; its terms are
at https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/LICENSE.

## DeepSeek model configuration files

SPDX license expression for the configuration files identified in this section: `MIT`.

The following model configuration files are copied from, or formatting-only
adaptations of, the named DeepSeek model repositories:

| AISim file | Upstream revision |
| --- | --- |
| `src/aisim_core/model_configs/deepseek-ai--DeepSeek-R1_config.json` | `deepseek-ai/DeepSeek-R1@56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad` |
| `src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json` | `deepseek-ai/DeepSeek-V3.2@c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3` |
| `src/aisim_core/model_configs/deepseek-ai--DeepSeek-V4-Flash_config.json` | `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1` |
| `src/aisim_core/model_configs/deepseek-ai--DeepSeek-V4.1-Flash_config.json` | `deepseek-ai/DeepSeek-V4.1-Flash@fb2764a5cf321eaa5070ca8f9e892818f477c16d` |
| `src/aisim_core/model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json` | `deepseek-ai/DeepSeek-V4-Pro@b5968e9190ef611bbf34a7229255be88a0e937c1` |

The V4.1 descriptor and performance formulas in `src/aisim_core/sdk/deepseek_v41.py`,
`src/aisim_core/sdk/models/deepseek_v41.py`, and repository-root
`crates/core/src/perfmodel/operators/dsv41.rs` are AISim performance-model
adaptations of the architecture described by `inference/model.py` and
`DeepSeek_V41_Tech_Report.pdf` at the same V4.1 revision (modified; no model execution code).
Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/fb2764a5cf321eaa5070ca8f9e892818f477c16d

Upstream repositories:
https://huggingface.co/deepseek-ai

```text
MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Meta Muse Glimmer model configuration

SPDX license expression for the configuration files identified in this section: `Apache-2.0`.

`src/aisim_core/model_configs/meta-models--Muse-Glimmer-30B_config.json`
is an unmodified copy of `config.json` from the Meta Muse Glimmer model
repository at immutable revision
`f84ecc3a0ea984a4c04542a84269e3d065350a6e`:

https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/f84ecc3a0ea984a4c04542a84269e3d065350a6e/config.json

Copyright owner: Meta Platforms, Inc. and affiliates.

License: Apache License 2.0. The upstream license is available at:
https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/f84ecc3a0ea984a4c04542a84269e3d065350a6e/LICENSE

The Apache License 2.0 terms are reproduced in this distribution's `LICENSE`
file.

## Qwen3.8-Max model configuration files

SPDX license expression for the configuration files identified in this section: `LicenseRef-Qwen3.8-Max`.

The following model configuration files are copied byte-for-byte from the
named Qwen model repositories at the immutable revisions shown:

| Packaged file | Upstream revision |
| --- | --- |
| `aisim_core/model_configs/Qwen--Qwen3.8-2.4T-A95B_config.json` | `Qwen/Qwen3.8-2.4T-A95B@207bd685a7e3696cfaff12ded7c6a7ea0f88c996` |
| `aisim_core/model_configs/Qwen--Qwen3.8-2.4T-A95B-FP8_config.json` | `Qwen/Qwen3.8-2.4T-A95B-FP8@d2dc35658bcf77e66643428cb52e774cc3b5bd29` |

Upstream repositories:
https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B and
https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8

Copyright owner: Qwen. The copied files are unmodified. Both upstream
revisions apply the same custom Qwen3.8-Max License, reproduced in full
below:

```text
Qwen3.8-Max License

Copyright (c) 2026 Qwen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software, including the model weights, parameters, configuration files, inference code and associated documentation files (collectively, the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, sell, deploy, host, fine-tune, and create derivative works from (collectively, "Use" or "Using") copies of the Software; and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. If the Software (or any derivative works thereof) is Used for any of the licensee's commercial products or services that have more than 100,000,000 monthly active users or US$ 20,000,000 (or equivalent in other currencies) monthly revenue, respective model name must be prominently displayed on the user interface of such product or service; and,

2. If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant business, and the aggregate revenue of the licensee and its affiliates exceeds US$50,000,000 (or the equivalent amount in any other currencies) during any consecutive twelve (12) months, the licensee shall obtain a separate license from Qwen before Using the Software or its derivative works for any commercial purpose. The foregoing requirement shall not apply to the licensee's internal Use of the Software, provided that such Use does not make the Software, its outputs, or its underlying model capabilities available to any third party.

"Model as a Service" means giving a third party access to language model inference or fine-tuning (e.g., via API or a hosted endpoint) in a manner that allows such third parties to exercise meaningful control over the inputs, parameters, or training data. This does not include the mere relaying of requests to models hosted by other third parties.
“AI Work Assistant” means an independent AI-powered product primarily designed for AI-assisted coding or office productivity (e.g., Qoder and QwenWork). It does not include: (a) a single-purpose AI tool (such as an AI translation tool); (b) an AI assistant primarily designed for a domain other than coding or office productivity (such as Taobao AI Shopping Assistant or AMap AI Chat); or (c) an AI assistant that is a feature of a product whose primary purpose is not AI-assisted coding or office productivity.

THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL QWEN, ITS AFFILIATES OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. THE USE OF THE SOFTWARE MUST COMPLY WITH APPLICABLE LAWS AND REGULATIONS, AND MUST NOT INFRINGE THE INTELLECTUAL PROPERTY RIGHTS OF ANY THIRD PARTY.

For any questions regarding this license, please contact model-business@notice.qwencloud.com.
```

## Meta Llama 4

SPDX license expression for the configuration files identified in this section: `LicenseRef-Llama-4-Community`.

The following bundled model configs are modified copies of Meta Llama 4
checkpoint configuration files:

- `src/aisim_core/model_configs/meta-llama--Llama-4-Scout-17B-16E-Instruct_config.json`
  from revision `92f3b1597a195b523d8d9e5700e57e4fbb8f20d3`:
  https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/92f3b1597a195b523d8d9e5700e57e4fbb8f20d3/config.json
- `src/aisim_core/model_configs/meta-llama--Llama-4-Maverick-17B-128E-Instruct_config.json`
  from revision `73d14711bcc77c16df3470856949c3764056b617`:
  https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/blob/73d14711bcc77c16df3470856949c3764056b617/config.json

Required attribution: Llama 4 is licensed under the Llama 4 Community License,
Copyright © Meta Platforms, Inc. All Rights Reserved.

The Llama 4 Community License below is reproduced from the official Meta Llama
repository at commit `0e0b8c519242d5833d8c11bffc1232b77ad7f301`:

https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama4/LICENSE

```text
LLAMA 4 COMMUNITY LICENSE AGREEMENT

Llama 4 Version Effective Date: April 5, 2025

“Agreement” means the terms and conditions for use, reproduction, distribution and modification of the Llama Materials set forth herein.

“Documentation” means the specifications, manuals and documentation accompanying Llama 4 distributed by Meta at https://www.llama.com/docs/overview.

“Licensee” or “you” means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

“Llama 4” means the foundational large language models and software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code and other elements of the foregoing distributed by Meta at https://www.llama.com/llama-downloads.

“Llama Materials” means, collectively, Meta’s proprietary Llama 4 and Documentation (and any portion thereof) made available under this Agreement.

“Meta” or “we” means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).

By clicking “I Accept” below or by using or distributing any portion or element of the Llama Materials, you agree to be bound by this Agreement.

1. License Rights and Redistribution.

a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Llama Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Llama Materials.

b. Redistribution and Use.

i. If you distribute or make available the Llama Materials (or any derivative works thereof), or a product or service (including another AI model) that contains any of them, you shall (A) provide a copy of this Agreement with any such Llama Materials; and (B) prominently display “Built with Llama” on a related website, user interface, blogpost, about page, or product documentation. If you use the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include “Llama” at the beginning of any such AI model name.

ii. If you receive Llama Materials, or any derivative works thereof, from a Licensee as part of an integrated end user product, then Section 2 of this Agreement will not apply to you.

iii. You must retain in all copies of the Llama Materials that you distribute the following attribution notice within a “Notice” text file distributed as a part of such copies: “Llama 4 is licensed under the Llama 4 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

iv. Your use of the Llama Materials must comply with applicable laws and regulations (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Llama Materials (available at https://www.llama.com/llama4/use-policy), which is hereby incorporated by reference into this Agreement.

2. Additional Commercial Terms. If, on the Llama 4 version release date, the monthly active users of the products or services made available by or for Licensee, or Licensee’s affiliates, is greater than 700 million monthly active users in the preceding calendar month, you must request a license from Meta, which Meta may grant to you in its sole discretion, and you are not authorized to exercise any of the rights under this Agreement unless or until Meta otherwise expressly grants you such rights.

3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE LLAMA MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS.

4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.

5. Intellectual Property.

a. No trademark licenses are granted under this Agreement, and in connection with the Llama Materials, neither Meta nor Licensee may use any name or mark owned by or associated with the other or any of its affiliates, except as required for reasonable and customary use in describing and redistributing the Llama Materials or as set forth in this Section 5(a). Meta hereby grants you a license to use “Llama” (the “Mark”) solely as required to comply with the last sentence of Section 1.b.i. You will comply with Meta’s brand guidelines (currently accessible at https://about.meta.com/brand/resources/meta/company-brand/). All goodwill arising out of your use of the Mark will inure to the benefit of Meta.

b. Subject to Meta’s ownership of Llama Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the Llama Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.

c. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Llama Materials or Llama 4 outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the Llama Materials.

6. Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Llama Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Llama Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement.

7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement.
```

## Hugging Face Transformers

The `rotate_half` function in `collector/trtllm/collect_mla.py` is copied from
`transformers.models.llama.modeling_llama.rotate_half` in the Hugging Face
Transformers project at tag `v4.57.1`:

https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py#L109-L113

The Gemma 4 vision-tower graph, bidirectional visual-block mask behavior, and
aspect-ratio-preserving processor math in these files are adapted and modified
for performance modeling from Hugging Face
Transformers at commit `cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55`:

- `src/aiconfigurator_core/sdk/models/blocks/vit.py`
- `src/aiconfigurator_core/sdk/models/gemma4.py`
- `src/aiconfigurator_core/sdk/backends/base_backend.py`

Upstream sources:

- https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/gemma4/modeling_gemma4.py
- https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/gemma4/image_processing_gemma4.py

Copyright 2026 the HuggingFace Team. All rights reserved.
This Gemma 4 material is licensed under the Apache License 2.0.

The Kimi K2.5 and Kimi K3 spatial-temporal vision-tower, processor, and temporal-pooling
behavior in `src/aiconfigurator_core/sdk/models/blocks/vit.py`,
`src/aiconfigurator_core/sdk/utils.py`, and `src/aiconfigurator_core/sdk/backends/base_backend.py`,
with regression derivatives in `tests/unit/sdk/models/test_kimi_k25_vision.py`
and `tests/unit/sdk/models/test_kimi_k3_vision.py`,
are modified adaptations of Hugging Face Transformers at
commit `cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55`:

- https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/modeling_kimi_k25.py
- https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/image_processing_kimi_k25.py
- https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/video_processing_kimi_k25.py

Copyright 2026 the HuggingFace Inc. team. All rights reserved.
Copyright 2026 the HuggingFace Team. All rights reserved.
This material is licensed under the Apache License 2.0.

The upstream license at that revision is available at:
https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/LICENSE

The Qwen VL `smart_resize` pixel-budget rule in
`python/aisim/src/aisim_core/sdk/backends/base_backend.py`
(`_qwen_smart_resize`) is a modified adaptation of the Qwen2-VL image processor
in Hugging Face Transformers at tag `v4.57.0`:

- https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py

Copyright 2024 the HuggingFace Inc. team. All rights reserved.
This material is licensed under the Apache License 2.0.

The fixed-tile canvas and global-tile logic in
`src/aiconfigurator_core/sdk/backends/base_backend.py`, and the processor metadata
normalization in `src/aiconfigurator_core/sdk/utils.py`, are modified adaptations
of the Llama 4 image processor (Copyright 2025 HuggingFace Inc. team. All rights
reserved.) at tag `v4.51.0` (commit
`0720e206c6ba28887e4d60ef60a6a089f6c1cc76`):

https://github.com/huggingface/transformers/blob/0720e206c6ba28887e4d60ef60a6a089f6c1cc76/src/transformers/models/llama4/image_processing_llama4_fast.py

The upstream license at that revision is available at:
https://github.com/huggingface/transformers/blob/0720e206c6ba28887e4d60ef60a6a089f6c1cc76/LICENSE

The Transformers `LICENSE` file at tag `v4.57.1` is reproduced verbatim below:

```text
Copyright 2018- The Hugging Face team. All rights reserved.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## Published FPM dataset pin manifests

The `hf_dataset.json` manifests and adjacent attribution README files under
`python/aisim/src/aisim_core/systems/profiles/` in
`minimax_m27_fpm/`, `dsv4pro_fpm/`, `dsv4flash_fpm/`, `glm52_fpm/`,
`kimi_k3_fpm/` and `inkling_fpm/` adapt published NVIDIA collection metadata
into hash-pinned download indexes (modified). No external timing tables,
model weights or collection implementation are bundled in these directories.

- Source: https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/dbfcdd9eb62ea733efbc3f0cbd78277f2960061a
- Immutable revision: `dbfcdd9eb62ea733efbc3f0cbd78277f2960061a`.
- Original paths: configuration `manifest.json` files and `fpm/*.metadata.json`
  under `data/MiniMaxAI--MiniMax-M2.7/`, `data/deepseek-ai--DeepSeek-V4-Pro/`,
  `data/deepseek-ai--DeepSeek-V4-Flash-0731/`, `data/nvidia--GLM-5.2-NVFP4/`,
  `data/moonshotai--Kimi-K3/` and `data/thinkingmachines--Inkling-NVFP4/`.
  Each derived pin records the exact artifact paths and SHA-256 values.
- Copyright: NVIDIA CORPORATION & AFFILIATES.
- License: Apache-2.0, as declared in the dataset card at the pinned revision;
  the Apache-2.0 terms are reproduced in this repository's `LICENSE`.
- Modifications: separate deployment roots, explicit development/diagnostic
  admission, canonical loader targets and explicit references to installed
  AISim hardware configurations. Source artifact bytes are unchanged when
  downloaded.

The dataset's root `THIRD_PARTY_NOTICES.md` is explicitly scoped to separate
DeepSeek V4.1 source archives; it is not a license grant for these six models.
This attribution relies on the dataset card for the indexed collection metadata
and does not relicense model weights, producer code or their upstream projects.

## AISim FPM Gym

- Source: https://gitlab-master.nvidia.com/dl/ai-dynamo/aisim-fpm-gym
- Revision: `e8221729db2802e822f6919fd68bc2941743385b`.
- Original paths: `src/aisim_fpm/{hf,types,models,evals/fpt}`,
  `dashboard/index.html`, `dashboard/assets/gym.css`, and `tests/test_hf_dataset.py`.
- Derived files: `scripts/fpm_accuracy/`, `pages/fpm-accuracy/`, and
  `tests/fpm_accuracy/test_hf_dataset.py`.
- Copyright: NVIDIA CORPORATION & AFFILIATES.
- License: Apache-2.0; NVIDIA maintainer confirmed permission to migrate and
  publish this code under Apache-2.0.
- Modified: development-only two-predictor evaluation, public overview export,
  GitHub Pages presentation, local import paths, and canonical estimator API
  adaptation with older-wheel compatibility.
- Additional revision: `f934c030afc3a03cb04d8f3ff4709194f7445c98`.
  Original paths: `src/aisim_fpm/dashboard/{data,measurement_heatmaps,visualization,visualization_diagnostics}.py`,
  `dashboard/3d-visualization.html`, `dashboard/assets/visualization.{js,css}`,
  and `tests/test_visualization.py`.
  Derived files: `scripts/fpm_accuracy/dashboard/`,
  `pages/fpm-accuracy/3d-visualization.html`, `pages/fpm-accuracy/assets/visualization.{js,css}`,
  `tests/fpm_accuracy/test_visualization.py`, and its synthetic outputs in
  `tests/fpm_accuracy/fixtures/dashboard/`.
  Modified imports, reduced contracts, GitHub artifact publication, navigation,
  theme integration, and tests. Same NVIDIA copyright and Apache-2.0 license.

## Plotly.js (FPM visualization)

- Upstream: https://github.com/plotly/plotly.js/tree/v3.4.0
- Immutable release tag: `v3.4.0`; original path: `dist/plotly.min.js`.
- Distributed unmodified as `pages/fpm-accuracy/assets/plotly.min.js`.
- Copyright 2012–2026 Plotly, Inc.; MIT license preserved in the bundled header
  and `pages/fpm-accuracy/assets/PLOTLY-LICENSE.txt`. The Pages build copies both.

## Plotly.js (replay dashboard)

- The replay HTML renderer embeds the unmodified Plotly.js bundle supplied by
  the installed `plotly` Python dependency, preserving its version and copyright
  header. No Plotly.js source is vendored in the Python renderer.
- `python/aisim/src/aisim/replay/plotly-js-LICENSE.txt` is an unmodified
  copy of the MIT license from https://github.com/plotly/plotly.js/blob/v3.4.0/LICENSE
  (immutable release tag `v3.4.0`, original path `LICENSE`). It is packaged and
  embedded in generated `report.html` files.
- Copyright (c) 2012–2026 Plotly, Inc. The dashboard template and telemetry
  transformations are AISim code; they do not modify Plotly.js.


## vLLM Kimi KDA state sizing

`src/aisim_core/sdk/memory/kimi_k3.py` and the cache-dtype controls in
`src/aisim/config/engine.py` are adapted and modified from vLLM commit
`a474da28131f61684849b31e29af0eebaaedc383`. Original paths:
`vllm/model_executor/layers/mamba/mamba_utils.py`,
`vllm/model_executor/models/kimi_linear.py`,
`vllm/model_executor/layers/kda.py`,
`vllm/platforms/interface.py`, `vllm/v1/kv_cache_interface.py`,
`vllm/v1/attention/backends/mla/triton_mla.py`, and `vllm/config/cache.py`.
The modified implementation reuses AISim's Kimi model for token KV geometry,
resolves the requested block granularity to fit one state using CPU arithmetic,
and adds AISim validation and diagnostics.

Source: https://github.com/vllm-project/vllm/tree/a474da28131f61684849b31e29af0eebaaedc383

Copyright contributors to the vLLM project.
Licensed under Apache-2.0;
the license is included at the repository root as `LICENSE` and in the Python
distribution. Upstream license:
https://github.com/vllm-project/vllm/blob/a474da28131f61684849b31e29af0eebaaedc383/LICENSE

## GLM-5.3 model configuration files

SPDX license expression for the configuration files identified in this section: `LicenseRef-GLM-5.3`.

The same base license also governs the NVIDIA-derived configuration files `nvidia--GLM-5.3-NVFP4_config.json` and `nvidia--GLM-5.3-NVFP4_hf_quant_config.json`, whose source revision is `e3b8f9f2fad29727dfa95e766c5cac18ea1414bf`: https://huggingface.co/nvidia/GLM-5.3-NVFP4/tree/e3b8f9f2fad29727dfa95e766c5cac18ea1414bf . NVIDIA CORPORATION & AFFILIATES contributes its export/quantization material; Z.AI retains the base rights. This cross-reference binds the base license separately from NVIDIA's own contribution; the combined recorded terms remain in the NVIDIA GLM-5.3-NVFP4 section.

The following files are unmodified copies of the named Hugging Face
configuration files at the immutable revisions shown:

| Packaged file | Upstream file and revision |
| --- | --- |
| `src/aisim_core/model_configs/zai-org--GLM-5.3_config.json` | `config.json` from `zai-org/GLM-5.3@ba32f8c19c8738e2c57c71798c601d477a7bf844` |
| `src/aisim_core/model_configs/zai-org--GLM-5.3-BF16_config.json` | `config.json` from `zai-org/GLM-5.3-BF16@abf42e6c0e53bc29a875511301e7b7fcd220315b` |

Upstream repositories:
https://huggingface.co/zai-org/GLM-5.3 and
https://huggingface.co/zai-org/GLM-5.3-BF16

Copyright owner: Z.AI. The copied files are unmodified. Both upstream
revisions ship the GLM-5.3 License, reproduced in full below:

```text
GLM-5.3 License

Copyright (c) 2026 Z.AI

Permission is hereby granted, free of charge, to any person or entity (the "Licensee") obtaining a copy of this software — including the model weights, parameters, configuration files, inference and training code, and associated documentation (collectively, the "Software") — to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software; to run, deploy, fine-tune, or otherwise modify the Software and create derivative works from it; and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. The Licensee's use of the Software must comply with applicable laws and regulations.

2. "Model as a Service" means giving a third party access to language model inference or fine-tuning (e.g., via API) in a manner that allows such third party to exercise meaningful control over the inputs, parameters, or training data. This does not include (a) end-user products with model capabilities solely embedded within specific features or harnesses, or (b) mere relaying of requests to models hosted by others.
If the Licensee or any of its affiliates operates a Model as a Service business, and the aggregate revenue of the Licensee and its affiliates exceeds 10 billion US dollars (or the equivalent in other currencies) in total over any consecutive 12 months, the Licensee must pass Z.AI's security review before using the Software or its derivative works for any commercial purpose. The scope and method of the security review shall be reasonably determined by Z.AI.

3. THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL Z.AI OR ITS AFFILIATES OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For any questions regarding this license, please contact glmlicense@z.ai.

-----

版权所有 (c) 2026 Z.AI

特此免费授予任何获得本软件副本的个人或实体（"被许可方"）——包括模型权重、参数、配置文件、推理和训练代码及相关文档（统称"软件"）——不受限制地处理本软件的权利，包括但不限于：使用、复制、修改、合并、发布、分发、再许可和/或销售软件副本；运行、部署、微调或以其他方式修改软件并创建衍生作品；以及允许获得软件的其他人行使上述权利，但须遵守以下条件：

1. 上述版权声明和本许可声明应包含在软件的所有副本或实质性部分中。被许可方对软件的使用必须符合适用法律法规。

2. "模型即服务"指以允许第三方对输入、参数或训练数据行使实质性控制的方式，向第三方提供语言模型推理或微调服务（如通过API）。不包括：(a) 模型能力仅嵌入特定功能或框架中的终端用户产品，或(b) 单纯转发请求至他人托管的模型。
若被许可方或其关联方运营"模型即服务"业务，且被许可方及关联方在任意连续12个月内累计总收入超过100亿美元（或等值其他货币），则被许可方在使用软件或其衍生作品进行任何商业用途之前，须通过Z.AI的安全审查。安全审查的范围和方式由Z.AI合理确定。

3. 软件及其任何输出和结果均按"现状"提供，不附带任何形式的保证，无论是明示还是暗示，包括但不限于适销性、特定用途适用性和不侵权的保证。在任何情况下，Z.AI或其关联方或版权持有人均不对任何索赔、损害或其他责任承担责任，无论该责任是基于合同、侵权或其他方式，因软件或使用软件而产生或与之相关。

如对本许可有任何疑问，请联系 glmlicense@z.ai。
```

## GLM-5.3-Flash configurations and analytical model

Unmodified configurations are included at
`src/aisim_core/model_configs/zai-org--GLM-5.3-Flash_config.json`,
`nvidia--GLM-5.3-Flash-NVFP4_config.json`, and
`nvidia--GLM-5.3-Flash-NVFP4_hf_quant_config.json` in the same directory.
Sources are Z.AI's `config.json` at
https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
and NVIDIA's `config.json` / `hf_quant_config.json` at
https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/09b04e5e74bca08ca8549fc736d4cdd8624bfde3.
Copyright (c) 2026 Z.AI Co., Ltd. MIT license; the full license is preserved
in the adjacent distributed `GLM53FLASH_PROVENANCE.md`.

The geometry in `src/aisim_core/sdk/glm53flash.py`,
`src/aisim_core/sdk/models/glm53flash.py`, repository-root
`crates/core/src/perfmodel/operators/glm53flash.rs`, and their GLM tests is a
modified analytical adaptation of those configurations. Execution boundaries
and precision partitions are independently expressed adaptations of
`vllm/models/glm5next/common/{model,attention,kda}.py` and
`vllm/models/glm5next/{nvidia,common}/sparse_indexer.py`,
`vllm/model_executor/layers/logits_processor.py`, and
`vllm/platforms/interface.py` from vLLM at
https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6 (v0.31.0)
(Copyright vLLM contributors, Apache-2.0), and
`python/sglang/srt/models/glm5_next.py`, `python/sglang/srt/models/deepseek_v2.py`,
`python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py`, and
`python/sglang/srt/layers/communicator_mhc.py` and
`python/sglang/srt/layers/logits_processor.py` from SGLang at
https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e
(Copyright SGLang contributors, Apache-2.0). No serving implementation is
vendored. Upstream licenses are at `LICENSE` under those immutable revisions;
the repository Apache-2.0 license text applies to these adaptations.

The Hopper SGLang FP8-KV attention precision in
`crates/core/src/perfmodel/operators/glm53flash.rs` and its arithmetic tests
is a modified, independently expressed analytical adaptation of
`python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py` at
https://github.com/sgl-project/sglang/tree/9336681310c7a7153c4fbd7c2b9ae62011899014
(Copyright SGLang contributors, Apache-2.0). Changes: express the raw-FP8
TileLang QK/PV precision as a tensor-core roofline rate on SM90 and verify
the mathematical work with hand-calculated examples; no kernel source is
vendored. The upstream license is preserved by the repository and packaged
Apache-2.0 `LICENSE` text.

## NVIDIA GLM-5.3-NVFP4 model configuration files

SPDX license expression for the configuration files identified in this section: `LicenseRef-GLM-5.3 AND LicenseRef-NVIDIA-Open-Model-Agreement`.

The following files are unmodified copies of the named Hugging Face
configuration files at the immutable revision shown:

| Packaged file | Upstream file and revision |
| --- | --- |
| `src/aisim_core/model_configs/nvidia--GLM-5.3-NVFP4_config.json` | `config.json` from `nvidia/GLM-5.3-NVFP4@e3b8f9f2fad29727dfa95e766c5cac18ea1414bf` |
| `src/aisim_core/model_configs/nvidia--GLM-5.3-NVFP4_hf_quant_config.json` | `hf_quant_config.json` from `nvidia/GLM-5.3-NVFP4@e3b8f9f2fad29727dfa95e766c5cac18ea1414bf` |

Upstream repository:
https://huggingface.co/nvidia/GLM-5.3-NVFP4/tree/e3b8f9f2fad29727dfa95e766c5cac18ea1414bf

Copyright owners: NVIDIA CORPORATION & AFFILIATES (the quantized export and
its quantization metadata) and Z.AI (the base GLM-5.3 configuration). The
copied files are unmodified.

License, as declared by the upstream model card at that revision: "License
type: NVIDIA Proprietary", with use governed by the NVIDIA Open Model
Agreement,
https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
(the card links the unversioned page; the agreement text preserved in the
provenance file named below was retrieved 2026-10-27 and is v. March 9,
2026, page release date April 2, 2026). The card metadata declares
`license_name: nvidia-open-model-license`. The original GLM-5.3 model is
licensed under the GLM-5.3 License (Copyright (c) 2026 Z.AI), which ships in
the upstream repository as `LICENSE`; the card states that the NVIDIA Open
Model Agreement does not modify or supersede it. The full texts of both are
preserved in the adjacent distributed
`src/aisim_core/model_configs/GLM53_NVFP4_PROVENANCE.md`.

The two license texts are also preserved together here so the canonical
wheel notice and the bundled-config inventory bind both sets of terms.
The wording is unchanged from the adjacent provenance file above.

```text
GLM-5.3 License

Copyright (c) 2026 Z.AI

Permission is hereby granted, free of charge, to any person or entity (the "Licensee") obtaining a copy of this software — including the model weights, parameters, configuration files, inference and training code, and associated documentation (collectively, the "Software") — to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software; to run, deploy, fine-tune, or otherwise modify the Software and create derivative works from it; and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. The Licensee's use of the Software must comply with applicable laws and regulations.

2. "Model as a Service" means giving a third party access to language model inference or fine-tuning (e.g., via API) in a manner that allows such third party to exercise meaningful control over the inputs, parameters, or training data. This does not include (a) end-user products with model capabilities solely embedded within specific features or harnesses, or (b) mere relaying of requests to models hosted by others.
If the Licensee or any of its affiliates operates a Model as a Service business, and the aggregate revenue of the Licensee and its affiliates exceeds 10 billion US dollars (or the equivalent in other currencies) in total over any consecutive 12 months, the Licensee must pass Z.AI's security review before using the Software or its derivative works for any commercial purpose. The scope and method of the security review shall be reasonably determined by Z.AI.

3. THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL Z.AI OR ITS AFFILIATES OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For any questions regarding this license, please contact glmlicense@z.ai.

-----

版权所有 (c) 2026 Z.AI

特此免费授予任何获得本软件副本的个人或实体（"被许可方"）——包括模型权重、参数、配置文件、推理和训练代码及相关文档（统称"软件"）——不受限制地处理本软件的权利，包括但不限于：使用、复制、修改、合并、发布、分发、再许可和/或销售软件副本；运行、部署、微调或以其他方式修改软件并创建衍生作品；以及允许获得软件的其他人行使上述权利，但须遵守以下条件：

1. 上述版权声明和本许可声明应包含在软件的所有副本或实质性部分中。被许可方对软件的使用必须符合适用法律法规。

2. "模型即服务"指以允许第三方对输入、参数或训练数据行使实质性控制的方式，向第三方提供语言模型推理或微调服务（如通过API）。不包括：(a) 模型能力仅嵌入特定功能或框架中的终端用户产品，或(b) 单纯转发请求至他人托管的模型。
若被许可方或其关联方运营"模型即服务"业务，且被许可方及关联方在任意连续12个月内累计总收入超过100亿美元（或等值其他货币），则被许可方在使用软件或其衍生作品进行任何商业用途之前，须通过Z.AI的安全审查。安全审查的范围和方式由Z.AI合理确定。

3. 软件及其任何输出和结果均按"现状"提供，不附带任何形式的保证，无论是明示还是暗示，包括但不限于适销性、特定用途适用性和不侵权的保证。在任何情况下，Z.AI或其关联方或版权持有人均不对任何索赔、损害或其他责任承担责任，无论该责任是基于合同、侵权或其他方式，因软件或使用软件而产生或与之相关。

如对本许可有任何疑问，请联系 glmlicense@z.ai。

NVIDIA Open Model Agreement

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

Release Date: April 2, 2026

NVIDIA Works released under this License are intended to be used permissively and enable the further development of AI technologies. Subject to the terms of this License, NVIDIA confirms that:

- Works are commercially usable.

- You are free to create and distribute Derivative Works.

- NVIDIA does not claim ownership to any outputs generated using the Works or Derivative Works.

By using, reproducing, modifying, distributing, performing or displaying any portion or element of the Works or Derivative Works, or otherwise accepting the terms of this License, you agree to be bound by this License.

1. Definitions.

   “License” shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 10 of this document.

   “Legal Entity” shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, “control” means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

   “You” (or “Your”) shall mean an individual or Legal Entity exercising permissions granted by this License.

   “Work” shall mean the work of authorship, including machine learning model, software, checkpoints, learnt weights, algorithms, parameters, configuration files and documentation, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).

   “Derivative Works” shall mean any work, whether in source or object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

2. Grant of License. Subject to the terms and conditions of this License, NVIDIA hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in source or object form.

   If You institute patent or copyright litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or an output from the Work constitutes direct or contributory patent or copyright infringement, then any licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.

3. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in source or object form, provided that You meet the following conditions:

      a. You must give any other recipients of the Work a copy of this License; and

      b. You must retain, in the source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and

      c. If the Work includes a “NOTICE” text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the following attribution notice within a “Notice” text file with such copies and the following statement: “Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement.”

   You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.

4. Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of NVIDIA, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.

5. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, NVIDIA provides the Work on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

6. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall NVIDIA be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work, a Derivative Work or an output from the Work or Derivative Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if NVIDIA has been advised of the possibility of such damages.

7. Accepting Warranty or Additional Liability. While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of NVIDIA, and only if You agree to indemnify, defend, and hold NVIDIA harmless for any liability incurred by, or claims asserted against, NVIDIA by reason of your accepting any such warranty or additional liability. You will indemnify and hold harmless NVIDIA from and against any claim by any third party arising out of or related to your use or distribution of the Works, Derivative Works thereof, or output from the Works or Derivative Works.

8. Feedback. NVIDIA appreciates your feedback, and You agree that NVIDIA may use it without restriction or compensation to You.

9. Governing Law. This Agreement will be governed in all respects by the laws of the United States and the laws of the State of Delaware, without regard to conflict of laws principles or the United Nations Convention on Contracts for the International Sale of Goods. The state and federal courts residing in Santa Clara County, California will have exclusive jurisdiction over any dispute or claim arising out of or related to this Agreement, and the parties irrevocably consent to personal jurisdiction and venue in those courts; except that, either party may apply for injunctive remedies or an equivalent type of urgent legal relief in any jurisdiction.

10. Trade and Compliance. You agree to comply with all applicable export, import, trade and economic sanctions laws and regulations, as amended, including without limitation U.S. Export Administration Regulations and Office of Foreign Assets Control regulations. These laws include restrictions on destinations, end-users and end-use.

(v. March 9, 2026)

End of terms and conditions
```

## vLLM DeepSeek V4 cache sizing

`src/aisim_core/sdk/memory/deepseek_v4.py` and the indexer cache-dtype
control in `src/aisim/config/engine.py` are adapted and modified from vLLM
commit `c5c116138267ac738bc262495326d28ffff834a2`. Original paths:
`vllm/models/deepseek_v4/attention.py`, `vllm/models/deepseek_v4/compressor.py`,
`vllm/v1/attention/backends/mla/sparse_swa.py`, `vllm/v1/kv_cache_interface.py`,
`vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/single_type_kv_cache_manager.py`,
and `vllm/config/attention.py`. The modified implementation computes DeepSeek V4
cache pages, packed cache groups and resident sliding-window blocks with CPU
arithmetic, maps them onto AISim's state cache, and adds AISim
validation and diagnostics.

Source: https://github.com/vllm-project/vllm/tree/c5c116138267ac738bc262495326d28ffff834a2

Copyright contributors to the vLLM project.
Licensed under Apache-2.0;
the license is included at the repository root as `LICENSE` and in the Python
distribution. Upstream license:
https://github.com/vllm-project/vllm/blob/c5c116138267ac738bc262495326d28ffff834a2/LICENSE

## SemiAnalysis AgentX simulation-performance fixture

`python/aisim/tools/simulation_perf_gate/fixtures/agentx.jsonl` is derived
from `traces.jsonl` in the SemiAnalysis dataset
https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/tree/8fecd2fc56694469f758f0afbbb6335ad3043740
at revision `8fecd2fc56694469f758f0afbbb6335ad3043740`.
`python/aisim/tools/simulation_perf_gate/fixtures/DATASET_CARD.md` is adapted
from upstream `README.md` at the same revision. Modified by NVIDIA: omitted the
distribution-plot section and added a source and scope note. The remaining
upstream text and license declaration are retained.

Upstream authors: SemiAnalysis. The upstream card supplies no separate
copyright or NOTICE statement. License: Apache License 2.0, declared in the
pinned dataset card and reproduced in `fixtures/LICENSE`.

Modified by NVIDIA: selected four complete plays and normalized JSON whitespace.
The original play retains its rows-API representation; the three added plays
retain their raw JSONL request values, dependencies, hashes, and timestamps.
See the adjacent fixture README and `agentx.json` manifest for the source,
play identities, counts, and content checksum.

## NVIDIA Dynamo CI handoff compatibility

The exact-source rewrite in `scripts/performance/build_simulation_dynamo.py`
is adapted and modified from NVIDIA Dynamo's `lib/mocker/src/common/handoff.rs`
at commit `def3b79b15c266805540a678dd400aeb6ccada1d`:
https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/common/handoff.rs

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under Apache-2.0; the full license is reproduced in `LICENSE`.
Upstream license: https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/LICENSE
The adaptation adds zero recurrent-state bytes to the legacy token-only DTO
conversion for CI builds against the newer AISim handoff API. It preserves
the upstream copyright header and marks the modified conversion in the output.

## NVIDIA Dynamo CI struct-literal compatibility

The exact-source rewrites in `scripts/performance/build_simulation_dynamo.py`
and their regression fixtures in `tests/test_ci_workflow_contracts.py`
are adapted and modified from NVIDIA Dynamo's
`lib/mocker/src/grouped_scheduler/compatibility.rs` and
`lib/mocker/src/replay/offline/extensions/kv_router/mod.rs`,
`lib/mocker/src/replay/offline/entrypoints.rs`,
`lib/mocker/src/replay/offline/extensions/kv_events/mod.rs`, and
`lib/bindings/python/rust/llm/replay.rs` at commit
`def3b79b15c266805540a678dd400aeb6ccada1d`:
https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/grouped_scheduler/compatibility.rs
https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/replay/offline/extensions/kv_router/mod.rs
https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/replay/offline/entrypoints.rs
https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/replay/offline/extensions/kv_events/mod.rs
https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/bindings/python/rust/llm/replay.rs

Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under Apache-2.0; the full license is reproduced in `LICENSE`.
Upstream license: https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/LICENSE
The adaptation adds `None` for new optional AISim fields to the native
request and offline KV-router placement struct literals for CI builds against
newer AISim APIs. It also sets an empty image list for the pinned text-only
request when AISim requires `images: Vec<ImageSpec>`, rejecting image inputs
instead of adding multimodal support to Dynamo's adapter. Both offline replay
specifications disable the encoder pool; the Python binding's synthetic trace
explicitly retains text-only traffic with `images: None`. The lifecycle conversion explicitly
rejects host-loop TTFT milestones, which its legacy DTO cannot represent.
The Weka import adapter retains the default recorded client when the AISim
API adds the client selection field.
It preserves the upstream copyright headers and marks each modification in the output.

The replay compatibility rewrite in the same script adapts
`lib/mocker/src/engine_adapter.rs` at that same immutable revision:
https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/engine_adapter.rs
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under Apache-2.0 as above. The modified adapter explicitly sets prefill
context parallelism to 1 in four replay descriptors, preserving the pinned
adapter's topology. It also disables the SGLang host loop and takes the vision
cache capacity from the canonical SGLang default, preserving text-only behavior.
Its upstream copyright header is retained and each modified literal is marked
in the generated output.

## NVIDIA Dynamo CI speculation compatibility

The exact-source rewrite in `scripts/performance/build_simulation_dynamo.py`
and its regression fixture in `tests/test_ci_workflow_contracts.py` are adapted
and modified from NVIDIA Dynamo's
`lib/bindings/python/rust/llm/ais_callback.rs` at commit
`def3b79b15c266805540a678dd400aeb6ccada1d`:
https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/bindings/python/rust/llm/ais_callback.rs

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under Apache-2.0; the full license is reproduced in `LICENSE`.
Upstream license: https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/LICENSE
The adaptation reads the serialized Ngram identity instead of the removed
enum accessor, retaining legacy `nextn`, accepting disabled speculation, and
rejecting other replay schemes.
It preserves the upstream copyright header and marks the modified conversion
in the output.

The callback contract harness in `scripts/performance/check_dynamo_callback.py`
compiles that entire adapted callback with the real AISim core. Its minimal
trait interfaces are adapted from `lib/mocker/src/common/perf_model.rs` and
`lib/kv-router/src/scheduling/prefill_load.rs` at the same immutable commit:
https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/common/perf_model.rs
https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/kv-router/src/scheduling/prefill_load.rs

These files carry the same NVIDIA copyright and Apache-2.0 license above.
The harness modifies their module placement to test the callback independently
of the Dynamo scheduler; it does not replace callback or estimator behavior.

## SGLang simulator and HiSim feature definitions

The per-request feature set of the learned forward-pass model
(`REQUEST_FEATURE_NAMES` in `python/aisim/src/aisim_core/sdk/fpm_learned.py` and
`REQUEST_FEATURE_NAMES` in `crates/core/src/perfmodel/fpm/learned.rs`,
relative to the repository root) is a re-implementation of the 18 batch
features defined by `MLTimePredictor.FEATURE_NAMES` in the SGLang simulator,
and the 32 `(present, past, extend)` request slots follow the feature layout
of HiSim's XGBoost predictor. No upstream source code is copied; the feature
names, ordering, and formulas were adapted so that models trained for either
tool line up with this one.

Upstream sources:
- https://github.com/sgl-project/sglang/blob/59799a368793b9f795baf59b067233a65ad8e38e/tools/sglang-simulator/src/sglang_simulator/time_predictor/ml.py
  (SGLang, Apache License 2.0; copyright the SGLang project contributors)
- https://github.com/alibaba/tair-kvcache/blob/85402f1a012fb4279be5bb4b57613bb0a067d361/hisim/src/hisim/time_predictor/aiconfigurator.py
  (HiSim, Alibaba tair-kvcache, Apache License 2.0; copyright Alibaba Group)

Both upstream licenses at the identified revisions:
https://github.com/sgl-project/sglang/blob/59799a368793b9f795baf59b067233a65ad8e38e/LICENSE
https://github.com/alibaba/tair-kvcache/blob/85402f1a012fb4279be5bb4b57613bb0a067d361/LICENSE

The AISim files are modified re-implementations, not verbatim copies.
Both derived files carry the source URLs and revisions in their headers.

## SGLang v0.5.20 transferred-prompt publication at running admission and release on ready-queue cancellation

The following repository-root files independently model transferred-prompt publication at running admission and release on ready-queue cancellation:

- `crates/core/src/engine/kv_manager/sglang_backend.rs`
- `crates/core/src/engine/scheduler/sglang/core.rs`
- `crates/core/src/engine/scheduler/sglang/tests.rs`
- `crates/core/src/replay/disagg_tests.rs`
- `crates/core/src/replay/handoff.rs`

The modified Rust behavior and synthetic regression fixtures adapt the contract
observed in SGLang v0.5.20, commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.
Original source paths: python/sglang/srt/disaggregation/decode.py (get_new_prebuilt_batch), python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py (process_prebuilt), and python/sglang/srt/managers/scheduler.py (waiting-queue cancellation).

Source: https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e

Copyright 2023-2024 SGLang Team and SGLang contributors. Apache-2.0;
the upstream license is at
https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/LICENSE
and its full text is included in this repository as `LICENSE`.
Modified for AISim request leases, physical page accounting and replay.
No serving implementation, production request content or upstream test fixture is vendored.

## SGLang v0.5.20 partial-edge access recency and reinsertion of an already cached prefix

The following repository-root files independently model partial-edge access recency and reinsertion of an already cached prefix:

- `crates/core/src/engine/cache/radix_cache.rs`

The modified Rust behavior and synthetic regression fixtures adapt the contract
observed in SGLang v0.5.20, commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.
Original source paths: python/sglang/srt/mem_cache/radix_cache.py (_match_prefix_helper and _insert_helper).

Source: https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e

Copyright 2023-2024 SGLang Team and SGLang contributors. Apache-2.0;
the upstream license is at
https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/LICENSE
and its full text is included in this repository as `LICENSE`.
Modified for AISim request leases, physical page accounting and replay.
No serving implementation, production request content or upstream test fixture is vendored.

## SGLang v0.5.20 stable length-policy retraction order and post-release cache eviction toward the survivor reserve

The following repository-root files independently model stable length-policy retraction order and post-release cache eviction toward the survivor reserve:

- `crates/core/src/engine/scheduler/sglang/core.rs`
- `crates/core/src/engine/scheduler/sglang/decode.rs`
- `crates/core/src/engine/scheduler/sglang/tests.rs`

The modified Rust behavior and synthetic regression fixtures adapt the contract
observed in SGLang v0.5.20, commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.
Original source paths: python/sglang/srt/managers/schedule_batch.py (_get_decode_retraction_order, retract_decode and release_req).

Source: https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e

Copyright 2023-2024 SGLang Team and SGLang contributors. Apache-2.0;
the upstream license is at
https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/LICENSE
and its full text is included in this repository as `LICENSE`.
Modified for AISim request leases, physical page accounting and replay.
No serving implementation, production request content or upstream test fixture is vendored.

## SGLang v0.5.20 completed source-prompt publication before the transfer hold

The following repository-root files independently model completed source-prompt publication before the transfer hold:

- `crates/core/src/engine/scheduler/sglang/core.rs`
- `crates/core/src/engine/scheduler/sglang/tests.rs`

The modified Rust behavior and synthetic regression fixtures adapt the contract
observed in SGLang v0.5.20, commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.
Original source paths: python/sglang/srt/disaggregation/prefill.py (process_batch_result_disagg_prefill).

Source: https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e

Copyright 2023-2024 SGLang Team and SGLang contributors. Apache-2.0;
the upstream license is at
https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/LICENSE
and its full text is included in this repository as `LICENSE`.
Modified for AISim request leases, physical page accounting and replay.
No serving implementation, production request content or upstream test fixture is vendored.

## SGLang cache publication and KV event cadence

The modified behavior and synthetic tests in `crates/core/src/engine/kv_manager/sglang_backend.rs`, `crates/core/src/engine/scheduler/sglang/core.rs`, `crates/core/src/engine/scheduler/sglang/decode.rs`, `crates/core/src/engine/scheduler/sglang/host_loop.rs`, and `crates/core/src/engine/scheduler/sglang/tests.rs` adapt the cache insert points and event queue/publish contract from SGLang v0.5.20, immutable commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`: `python/sglang/srt/mem_cache/radix_cache.py`, `python/sglang/srt/mem_cache/events.py`, `python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`, and `python/sglang/srt/managers/scheduler_components/metrics_reporter.py`. Source: https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e . Copyright 2023-2024 SGLang Team and contributors, Apache-2.0. The license text is included as `LICENSE`. Modified for AISim request leases, overlap observation, and physical page accounting. No production fixture is included.

Running-request cancellation retains its KV lease until the in-flight forward
is observed, adapting `Scheduler.abort_request` and result-time
`release_kv_cache` from the same pinned scheduler and batch-result sources.

## DeepSeek V3 configuration source-material scope

The exact `deepseek-ai--DeepSeek-V3_config.json` source, copyright, complete terms and modification status are now recorded in the previously uncited configuration section below. This is source-side, untrained architecture metadata. The DeepSeek source repository identifies its code as MIT; the separate model agreement additionally grants distribution of Complementary Material in section II.2 and exempts it from the model-use restriction flow-down in III.4.a. This records the grant and source-material classification; it does not assert OSRB approval or distribute trained model weights. The upstream model/code texts are at https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/e815299b0bcbac849fa540c768ef21845365c9eb . Source terms: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/LICENSE-CODE and https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/LICENSE-MODEL .

## StepFun Step-3.7-Flash model configuration files

SPDX license expression for the configuration files identified in this section: `Apache-2.0`.

The following files are modified adaptations of the named Hugging Face
configuration files at the immutable revisions shown:

| Packaged file | Upstream file and revision |
| --- | --- |
| `src/aisim_core/model_configs/stepfun-ai--Step-3.7-Flash_config.json` | `config.json` from `stepfun-ai/Step-3.7-Flash@5f6244077ac62e04eec3f320501ff8c2b293373a` |
| `src/aisim_core/model_configs/stepfun-ai--Step-3.7-Flash-FP8_config.json` | `config.json` from `stepfun-ai/Step-3.7-Flash-FP8@b3d7916fccac844cca050d7520f2aaa513f9a84f` |

Upstream repositories:
https://huggingface.co/stepfun-ai/Step-3.7-Flash and
https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8

Copyright owner: StepFun. Both repositories declare the Apache-2.0 license in
their model cards. Its terms are in this repository's `LICENSE` and at
https://www.apache.org/licenses/LICENSE-2.0.

The packaged files are modified. They keep the decoder (`text_config`) fields
that AISim reads, moved to the top level, with these changes:

- The architecture is named `Step3p7FlashForCausalLM` and the model type
  `step3p7`.
- `num_attention_groups` is written as `num_key_value_heads`.
- The MoE fields are written as `num_experts`, `num_experts_per_tok`,
  `first_k_dense_replace`, `n_shared_experts` and `scoring_func`.
- The `layer_types` entries of the MTP layers are dropped.
- In the FP8 file, `quantization_config` keeps `quant_method`,
  `weight_block_size` and `activation_scheme`.

The release/0.14.0 adaptations retain `num_attention_heads: 64` without a separate `attention_other_setting` block and set `max_position_embeddings: 65536` instead of the upstream 262144. The bundled bytes are recorded in `scripts/ci/bundled_configs.json`.

## Previously uncited bundled model configurations

These 108 files were previously byte-pinned under the legacy UNKNOWN exception. Their exact immediate-source bytes are now identified: 105 match NVIDIA AIConfigurator's final sync boundary `c8aee02f0887547a334c3d6cd192c42757e4b40e`, and three GLM-5.3-Flash files match their separately recorded immutable Hugging Face sources. AIConfigurator's complete Apache-2.0 license is at https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/LICENSE and is reproduced in this distribution's `LICENSE`.

Each row identifies the packaged file, original model publisher/copyright owner, all recorded license grants and immutable immediate source path/revision. NVIDIA copyright applies to NVIDIA's own predecessor/export contributions; it does not replace the original model owner's rights. “Copied unchanged” is relative to that cited immediate source. It does not claim that the predecessor never adapted a model's original configuration. These are config/quantization JSON files, not model weights or a model inference service. Ownership, cited grants, notice preservation and actual approval remain separate records.

| Packaged file | Base copyright owner / publisher | Recorded licenses | Immutable immediate source | Modification status |
| --- | --- | --- | --- | --- |
| `aisim_core/model_configs/MiniMaxAI--MiniMax-M2.5_config.json` | MiniMax | `Apache-2.0 AND LicenseRef-MiniMax-M2.5-Model` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/MiniMaxAI--MiniMax-M2.5_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/MiniMaxAI--MiniMax-M2.7_config.json` | MiniMax | `Apache-2.0 AND LicenseRef-MiniMax-M2.7-Noncommercial` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/MiniMaxAI--MiniMax-M2.7_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/MiniMaxAI--MiniMax-M3_config.json` | MiniMax | `Apache-2.0 AND LicenseRef-MiniMax-M3-Community` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/MiniMaxAI--MiniMax-M3_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-0.6B_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-0.6B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-1.7B_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-1.7B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-235B-A22B-FP8_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-235B-A22B-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-235B-A22B_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-235B-A22B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-30B-A3B-FP8_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-30B-A3B-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-30B-A3B_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-30B-A3B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-32B-FP8-Static-PerTensor_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-32B-FP8-Static-PerTensor_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-32B-FP8-Static-PerTensor_hf_quant_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-32B-FP8-Static-PerTensor_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-32B-FP8_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-32B-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-32B_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-32B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-8B_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-8B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-Coder-480B-A35B-Instruct_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-Coder-480B-A35B-Instruct_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-VL-235B-A22B-Instruct_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-VL-235B-A22B-Instruct_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-VL-2B-Instruct_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-VL-2B-Instruct_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-VL-30B-A3B-Instruct_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-VL-30B-A3B-Instruct_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-VL-32B-Instruct_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-VL-32B-Instruct_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-VL-32B-Thinking_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-VL-32B-Thinking_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-VL-4B-Instruct_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-VL-4B-Instruct_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3-VL-8B-Instruct_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3-VL-8B-Instruct_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3.5-122B-A10B_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3.5-122B-A10B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3.5-27B_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3.5-27B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3.5-35B-A3B_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3.5-35B-A3B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/Qwen--Qwen3.5-397B-A17B_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3.5-397B-A17B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/XiaomiMiMo--MiMo-7B-Base_config.json` | XiaomiMiMo | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/XiaomiMiMo--MiMo-7B-Base_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/XiaomiMiMo--MiMo-V2-Flash_config.json` | XiaomiMiMo | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/XiaomiMiMo--MiMo-V2-Flash_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/deepseek-ai--DeepSeek-V3_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/deepseek-ai--DeepSeek-V3_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/google--gemma-4-26B-A4B_config.json` | Google | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/google--gemma-4-26B-A4B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/meta-llama--Meta-Llama-3.1-405B_config.json` | Meta Platforms, Inc. | `Apache-2.0 AND LicenseRef-Llama-3.1-Community` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/meta-llama--Meta-Llama-3.1-405B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/meta-llama--Meta-Llama-3.1-70B_config.json` | Meta Platforms, Inc. | `Apache-2.0 AND LicenseRef-Llama-3.1-Community` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/meta-llama--Meta-Llama-3.1-70B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json` | Meta Platforms, Inc. | `Apache-2.0 AND LicenseRef-Llama-3.1-Community` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/mistralai--Mixtral-8x22B-v0.1_config.json` | Mistral AI | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/mistralai--Mixtral-8x22B-v0.1_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/mistralai--Mixtral-8x7B-v0.1_config.json` | Mistral AI | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/mistralai--Mixtral-8x7B-v0.1_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/moonshotai--Kimi-K2-Instruct_config.json` | Moonshot AI | `Apache-2.0 AND LicenseRef-Kimi-K2-Instruct-Modified-MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/moonshotai--Kimi-K2-Instruct_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/moonshotai--Kimi-K2.5_config.json` | Moonshot AI | `Apache-2.0 AND LicenseRef-Kimi-K2.5-Modified-MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/moonshotai--Kimi-K2.5_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/moonshotai--Kimi-K3_config.json` | Moonshot AI | `Apache-2.0 AND LicenseRef-Kimi-K3` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/moonshotai--Kimi-K3_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--DeepSeek-V3.1-NVFP4_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--DeepSeek-V3.1-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--DeepSeek-V3.1-NVFP4_hf_quant_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--DeepSeek-V3.1-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--DeepSeek-V3.2-NVFP4_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--DeepSeek-V3.2-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--DeepSeek-V3.2-NVFP4_hf_quant_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--DeepSeek-V3.2-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--DeepSeek-V4-Flash-NVFP4_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--DeepSeek-V4-Flash-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--DeepSeek-V4-Flash-NVFP4_hf_quant_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--DeepSeek-V4-Flash-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--DeepSeek-V4-Pro-NVFP4_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--DeepSeek-V4-Pro-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--DeepSeek-V4-Pro-NVFP4_hf_quant_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--DeepSeek-V4-Pro-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--GLM-5-NVFP4_config.json` | Z.AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--GLM-5-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--GLM-5-NVFP4_hf_quant_config.json` | Z.AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--GLM-5-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--GLM-5.1-NVFP4_config.json` | Zhipu AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--GLM-5.1-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--GLM-5.1-NVFP4_hf_quant_config.json` | Zhipu AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--GLM-5.1-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--GLM-5.2-NVFP4_config.json` | Zhipu AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--GLM-5.2-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--GLM-5.2-NVFP4_hf_quant_config.json` | Zhipu AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--GLM-5.2-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--GLM-5.3-Flash-NVFP4_config.json` | Z.AI Co., Ltd. | `MIT` | [source](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/09b04e5e74bca08ca8549fc736d4cdd8624bfde3/config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--GLM-5.3-Flash-NVFP4_hf_quant_config.json` | Z.AI Co., Ltd. | `MIT` | [source](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/09b04e5e74bca08ca8549fc736d4cdd8624bfde3/hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Gemma-4-26B-A4B-NVFP4_config.json` | Google | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Gemma-4-26B-A4B-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Gemma-4-26B-A4B-NVFP4_hf_quant_config.json` | Google | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Gemma-4-26B-A4B-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Gemma-4-31B-IT-NVFP4_config.json` | Google | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Gemma-4-31B-IT-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Gemma-4-31B-IT-NVFP4_hf_quant_config.json` | Google | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Gemma-4-31B-IT-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Kimi-K2.5-NVFP4_config.json` | Moonshot AI | `Apache-2.0 AND LicenseRef-Kimi-K2.5-Modified-MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Kimi-K2.5-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Kimi-K2.6-NVFP4_config.json` | Moonshot AI | `Apache-2.0 AND LicenseRef-Kimi-K2.6-Modified-MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Kimi-K2.6-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Kimi-K2.6-NVFP4_hf_quant_config.json` | Moonshot AI | `Apache-2.0 AND LicenseRef-Kimi-K2.6-Modified-MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Kimi-K2.6-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Kimi-K2.7-Code-NVFP4_config.json` | Moonshot AI | `Apache-2.0 AND LicenseRef-Kimi-K2.7-Code-Modified-MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Kimi-K2.7-Code-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Kimi-K2.7-Code-NVFP4_hf_quant_config.json` | Moonshot AI | `Apache-2.0 AND LicenseRef-Kimi-K2.7-Code-Modified-MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Kimi-K2.7-Code-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Llama-3.1-70B-Instruct-FP8_config.json` | Meta Platforms, Inc. | `Apache-2.0 AND LicenseRef-Llama-3.1-Community` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Llama-3.1-70B-Instruct-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Llama-3.1-70B-Instruct-FP8_hf_quant_config.json` | Meta Platforms, Inc. | `Apache-2.0 AND LicenseRef-Llama-3.1-Community` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Llama-3.1-70B-Instruct-FP8_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Llama-3_3-Nemotron-Super-49B-v1_config.json` | Meta Platforms, Inc. | `Apache-2.0 AND LicenseRef-Llama-3.3-Community` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Llama-3_3-Nemotron-Super-49B-v1_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--MiniMax-M2.5-NVFP4_config.json` | MiniMax | `Apache-2.0 AND LicenseRef-MiniMax-M2.5-Model` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--MiniMax-M2.5-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--MiniMax-M2.5-NVFP4_hf_quant_config.json` | MiniMax | `Apache-2.0 AND LicenseRef-MiniMax-M2.5-Model` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--MiniMax-M2.5-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--MiniMax-M2.7-NVFP4_config.json` | MiniMax | `Apache-2.0 AND LicenseRef-MiniMax-M2.7-Noncommercial` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--MiniMax-M2.7-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--MiniMax-M2.7-NVFP4_hf_quant_config.json` | MiniMax | `Apache-2.0 AND LicenseRef-MiniMax-M2.7-Noncommercial` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--MiniMax-M2.7-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--MiniMax-M3-NVFP4_config.json` | MiniMax | `Apache-2.0 AND LicenseRef-MiniMax-M3-Community` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--MiniMax-M3-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--MiniMax-M3-NVFP4_hf_quant_config.json` | MiniMax | `Apache-2.0 AND LicenseRef-MiniMax-M3-Community` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--MiniMax-M3-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4_hf_quant_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-FP8_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-FP8_hf_quant_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-FP8_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_hf_quant_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8_hf_quant_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4_hf_quant_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4_hf_quant_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Nemotron-H-56B-Base-8K_config.json` | NVIDIA CORPORATION & AFFILIATES | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Nemotron-H-56B-Base-8K_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Qwen3-235B-A22B-NVFP4_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Qwen3-235B-A22B-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Qwen3-235B-A22B-NVFP4_hf_quant_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Qwen3-235B-A22B-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Qwen3.5-122B-A10B-NVFP4_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Qwen3.5-122B-A10B-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Qwen3.5-122B-A10B-NVFP4_hf_quant_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Qwen3.5-122B-A10B-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Qwen3.5-397B-A17B-NVFP4_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Qwen3.5-397B-A17B-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Qwen3.5-397B-A17B-NVFP4_hf_quant_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Qwen3.5-397B-A17B-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Qwen3.6-27B-NVFP4_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Qwen3.6-27B-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Qwen3.6-27B-NVFP4_hf_quant_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Qwen3.6-27B-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Qwen3.6-35B-A3B-NVFP4_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Qwen3.6-35B-A3B-NVFP4_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/nvidia--Qwen3.6-35B-A3B-NVFP4_hf_quant_config.json` | Qwen | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/nvidia--Qwen3.6-35B-A3B-NVFP4_hf_quant_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/openai--gpt-oss-120b_config.json` | OpenAI | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/openai--gpt-oss-120b_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/openai--gpt-oss-20b_config.json` | OpenAI | `Apache-2.0` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/openai--gpt-oss-20b_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/sgl-project--DeepSeek-V4-Flash-FP8_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/sgl-project--DeepSeek-V4-Flash-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/sgl-project--DeepSeek-V4-Pro-FP8_config.json` | DeepSeek | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/sgl-project--DeepSeek-V4-Pro-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/zai-org--GLM-5-FP8_config.json` | Z.AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/zai-org--GLM-5-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/zai-org--GLM-5.1-FP8_config.json` | Zhipu AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/zai-org--GLM-5.1-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/zai-org--GLM-5.1_config.json` | Zhipu AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/zai-org--GLM-5.1_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/zai-org--GLM-5.2-FP8_config.json` | Zhipu AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/zai-org--GLM-5.2-FP8_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/zai-org--GLM-5.2_config.json` | Zhipu AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/zai-org--GLM-5.2_config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/zai-org--GLM-5.3-Flash_config.json` | Z.AI Co., Ltd. | `MIT` | [source](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json) | Copied unchanged from cited immediate source |
| `aisim_core/model_configs/zai-org--GLM-5_config.json` | Z.AI | `Apache-2.0 AND MIT` | [source](https://github.com/ai-dynamo/aiconfigurator/blob/c8aee02f0887547a334c3d6cd192c42757e4b40e/aic-core/src/aiconfigurator_core/model_configs/zai-org--GLM-5_config.json) | Copied unchanged from cited immediate source |

Required base attribution retained outside the license quotations:

- Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
- Llama 3.3 is licensed under the Llama 3.3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
- MiniMax AI model is licensed under the MiniMax Model License, Copyright © MiniMax. All Rights Reserved.

Llama distribution branding is a separate condition: Built with Llama. This attribution record is not a claim that a prominent related website/documentation display has already been verified. MiniMax noncommercial terms remain applicable to base material within the release owner's stated noncommercial scope. Conditional commercial/MaaS conditions are not automatically deemed triggered or waived. Pending approval records are not changed to approved by adding these citations.

The Qwen Static-PerTensor files originate in NVIDIA AIConfigurator PR #338, real upstream merge commit `ae7ece35aff3224d428150d7d43b752d4e414680`, at `src/aiconfigurator/model_configs/`; their exact bytes also match the final boundary above. The Nemotron Ultra FP8 files were renamed/adapted in AIConfigurator PR #1197, merge commit `c3ef575cce61edf1e32c54d6f909f8775f380658`; the config moved from `nvidia--nemotron-ultra-rl-050826_config.json`, and the quantization config explicitly changed `NVFP4` to `FP8`. Inaccessible inferred HF URLs do not negate these verifiable NVIDIA source records.

The MiMo and GLM-5 model cards declare MIT but did not supply a model MIT license file: https://huggingface.co/XiaomiMiMo/MiMo-7B-Base/blob/d892dcc601b8a242e7b3e7d3bb9f55abbe72c0ad/README.md , https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/8b60b288e8325b7890fcdfe32bbb1d172c591a2d/README.md , and https://huggingface.co/zai-org/GLM-5/blob/c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2/README.md . Their separate GitHub code repositories use Apache-2.0; that code license is not substituted for the model-card MIT grant. The standard MIT permission/warranty terms below are adapted from SPDX license-list-data (CC0-1.0), https://github.com/spdx/license-list-data/blob/279b7e997a87c09be00dd6bdda189519ade502cf/text/MIT.txt ; the unused copyright-template line is omitted rather than inventing an upstream notice or year. The permission/warranty terms are otherwise unchanged, and the table supplies publisher attribution.

MiniMax M2.5's model card links an unavailable modified-MIT reference, while the pinned HF tree includes LICENSE-MODEL with an express redistribution grant. This record conservatively preserves that available full grant and flags the discrepancy for scoped review; it does not relabel it standard MIT. All full third-party terms, including the MiniMax use-policy appendices, remain below.

### Legal-text source references

- `LicenseRef-MiniMax-M2.5-Model`: https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/2515a3ea0afec1dd22d0c7794b3a91557804d195/LICENSE-MODEL
- `LicenseRef-MiniMax-M2.7-Noncommercial`: https://huggingface.co/MiniMaxAI/MiniMax-M2.7/blob/0a6536713b0b03f3be97494e5cf8d22f668a3164/LICENSE
- `LicenseRef-MiniMax-M3-Community`: https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/d95e30cb912fe3021bb9c2ba66a8ce60b6c937a9/LICENSE
- `MIT`: https://github.com/spdx/license-list-data/blob/279b7e997a87c09be00dd6bdda189519ade502cf/text/MIT.txt
- `MIT`: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/LICENSE-CODE
- `LicenseRef-Llama-3.1-Community`: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/LICENSE
- `LicenseRef-Kimi-K2-Instruct-Modified-MIT`: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/LICENSE
- `LicenseRef-Kimi-K2.5-Modified-MIT`: https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/LICENSE
- `LicenseRef-Kimi-K3`: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/LICENSE
- `MIT`: https://huggingface.co/zai-org/GLM-5.1/blob/26e1bd6e011feb778d25ae34b09b07074139d92d/LICENSE
- `MIT`: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/LICENSE
- `LicenseRef-Kimi-K2.6-Modified-MIT`: https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/LICENSE
- `LicenseRef-Kimi-K2.7-Code-Modified-MIT`: https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/LICENSE
- `LicenseRef-Llama-3.3-Community`: https://github.com/meta-llama/llama-models/blob/c908f5b7bd2c2fbdd89a48f6bc252efd4ebc293f/models/llama3_3/LICENSE

### Complete referenced terms

Legal-text SHA256 `afcfcb3aab51377faef0defdc8ae4499cc15efeedc0293f6fd9eb47de58e521b` for `Apache-2.0 AND LicenseRef-MiniMax-M2.5-Model`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

MINIMAX MODEL LICENSE
MiniMax-M2.5 Version Release Date: 2026-02-13

1. Definitions

"Agreement" means the terms and conditions for use, reproduction, distribution and modification of the Model Materials set forth herein.

"Licensee" or "you" means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity's behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

"Model" means the foundational large language models and software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code and other elements of the foregoing distributed by MiniMax.

"Model Materials" means, collectively, the Model and any source code, scripts, specifications, manuals and documentation accompanying the Model (and any portion thereof) made available under this Agreement.

"MiniMax" or "we" means MiniMax AI.

2. License Rights and Redistribution

a. Grant of Rights. You are granted a non-exclusive, worldwide and royalty-free limited license under MiniMax's intellectual property or other rights owned by MiniMax embodied in the Model Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Model Materials.

b. Redistribution and Use.

   i. If you distribute or make available the Model Materials (or any derivative works thereof), or a product or service that uses any of them, including another AI model, you shall provide a copy of this Agreement with any such the Model Materials or derivative works and cause any modified files to carry prominent notices stating that you changed the files. You may add your own copyright statement to your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of your modifications, or for any such derivative works as a whole, provided your use, reproduction, and distribution of the work otherwise complies with the terms and conditions of this Agreement.

   ii. You must retain in all copies of the Model Materials that you distribute the following attribution notice within a "Notice" text file distributed as a part of such copies: "MiniMax AI model is licensed under the MiniMax Model License, Copyright © MiniMax. All Rights Reserved."

   iii. Your use of the Model Materials must comply with applicable laws and regulations (including trade compliance laws and regulations) and adhere to the Prohibited Uses Policy for the Model Materials, which is hereby incorporated by reference into this Agreement.

3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE MODEL MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, AND MINIMAX DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE MODEL MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE MODEL MATERIALS AND ANY OUTPUT AND RESULTS.

4. Limitation of Liability. IN NO EVENT WILL MINIMAX OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF MINIMAX OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.

5. Intellectual Property.

a. No trademark license is granted to use the trade names, trademarks, service marks, or product names of MiniMax, except as required to fulfill notice requirements under this Agreement.

b. Subject to MiniMax's ownership of the Model Materials and derivatives made by or for MiniMax, with respect to any derivative works and modifications of the Model Materials that are made by you, as between you and MiniMax, you are and will be the owner of such derivative works and modifications.

c. If you institute litigation or other proceedings against MiniMax or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Model Materials or outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless MiniMax from and against any claim by any third party arising out of or related to your use or distribution of the Model Materials.

6. Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Model Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. MiniMax may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Model Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement.

7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of Singapore without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. Any dispute arising out of or in connection with this Agreement, including any question regarding its existence, validity or termination, shall be referred to and finally resolved by arbitration administered by the Singapore International Arbitration Centre ("SIAC") in accordance with the Arbitration Rules of the Singapore International Arbitration Centre ("SIAC Rules") for the time being in force, which rules are deemed to be incorporated by reference in this clause.

Prohibited Uses Policy

You agree you will not use, or allow others to use, the Models or any derivatives of the Models to:

1. Violate any applicable federal, state, local, or international law or regulation, or infringe upon the lawful rights or interests of any third party.
2. Assist with, engage in or otherwise support any military purpose.
3. Exploit, harm, or attempt to exploit or harm minors in any way.
4. Generate or disseminate false or misleading information with the intent to cause harm.
5. Generate or disseminate content prohibited by applicable laws or regulations.
6. Generate or disseminate personally identifiable information without proper authorization or for unlawful or unreasonable purposes.
7. Defame, disparage, harass, or cause harm to any individual or entity.
8. Conduct fully automated decision-making that adversely affects an individual's legal rights or creates or modifies a binding, enforceable obligation.
9. Promote discrimination, hate speech, or harmful behavior against individuals or groups based on race or ethnic origin, religion, disability, age, nationality and national origin, veteran status, sexual orientation, gender or gender identity, caste, immigration status, or any other characteristic that is associated with systemic discrimination or marginalization.
```

Legal-text SHA256 `be2c470faf49a31a57dc43322b85d7161a7781b8795b9fb69585b93adb971179` for `LicenseRef-MiniMax-M2.5-Model`:

```text
MINIMAX MODEL LICENSE
MiniMax-M2.5 Version Release Date: 2026-02-13

1. Definitions

"Agreement" means the terms and conditions for use, reproduction, distribution and modification of the Model Materials set forth herein.

"Licensee" or "you" means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity's behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

"Model" means the foundational large language models and software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code and other elements of the foregoing distributed by MiniMax.

"Model Materials" means, collectively, the Model and any source code, scripts, specifications, manuals and documentation accompanying the Model (and any portion thereof) made available under this Agreement.

"MiniMax" or "we" means MiniMax AI.

2. License Rights and Redistribution

a. Grant of Rights. You are granted a non-exclusive, worldwide and royalty-free limited license under MiniMax's intellectual property or other rights owned by MiniMax embodied in the Model Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Model Materials.

b. Redistribution and Use.

   i. If you distribute or make available the Model Materials (or any derivative works thereof), or a product or service that uses any of them, including another AI model, you shall provide a copy of this Agreement with any such the Model Materials or derivative works and cause any modified files to carry prominent notices stating that you changed the files. You may add your own copyright statement to your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of your modifications, or for any such derivative works as a whole, provided your use, reproduction, and distribution of the work otherwise complies with the terms and conditions of this Agreement.

   ii. You must retain in all copies of the Model Materials that you distribute the following attribution notice within a "Notice" text file distributed as a part of such copies: "MiniMax AI model is licensed under the MiniMax Model License, Copyright © MiniMax. All Rights Reserved."

   iii. Your use of the Model Materials must comply with applicable laws and regulations (including trade compliance laws and regulations) and adhere to the Prohibited Uses Policy for the Model Materials, which is hereby incorporated by reference into this Agreement.

3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE MODEL MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, AND MINIMAX DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE MODEL MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE MODEL MATERIALS AND ANY OUTPUT AND RESULTS.

4. Limitation of Liability. IN NO EVENT WILL MINIMAX OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF MINIMAX OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.

5. Intellectual Property.

a. No trademark license is granted to use the trade names, trademarks, service marks, or product names of MiniMax, except as required to fulfill notice requirements under this Agreement.

b. Subject to MiniMax's ownership of the Model Materials and derivatives made by or for MiniMax, with respect to any derivative works and modifications of the Model Materials that are made by you, as between you and MiniMax, you are and will be the owner of such derivative works and modifications.

c. If you institute litigation or other proceedings against MiniMax or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Model Materials or outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless MiniMax from and against any claim by any third party arising out of or related to your use or distribution of the Model Materials.

6. Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Model Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. MiniMax may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Model Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement.

7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of Singapore without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. Any dispute arising out of or in connection with this Agreement, including any question regarding its existence, validity or termination, shall be referred to and finally resolved by arbitration administered by the Singapore International Arbitration Centre ("SIAC") in accordance with the Arbitration Rules of the Singapore International Arbitration Centre ("SIAC Rules") for the time being in force, which rules are deemed to be incorporated by reference in this clause.

Prohibited Uses Policy

You agree you will not use, or allow others to use, the Models or any derivatives of the Models to:

1. Violate any applicable federal, state, local, or international law or regulation, or infringe upon the lawful rights or interests of any third party.
2. Assist with, engage in or otherwise support any military purpose.
3. Exploit, harm, or attempt to exploit or harm minors in any way.
4. Generate or disseminate false or misleading information with the intent to cause harm.
5. Generate or disseminate content prohibited by applicable laws or regulations.
6. Generate or disseminate personally identifiable information without proper authorization or for unlawful or unreasonable purposes.
7. Defame, disparage, harass, or cause harm to any individual or entity.
8. Conduct fully automated decision-making that adversely affects an individual's legal rights or creates or modifies a binding, enforceable obligation.
9. Promote discrimination, hate speech, or harmful behavior against individuals or groups based on race or ethnic origin, religion, disability, age, nationality and national origin, veteran status, sexual orientation, gender or gender identity, caste, immigration status, or any other characteristic that is associated with systemic discrimination or marginalization.
```

Legal-text SHA256 `02863ccc6e457c24602ef7d0860f45934f7e69866c183d4bc21e0374ad62cb9a` for `Apache-2.0 AND LicenseRef-MiniMax-M2.7-Noncommercial`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

NON-COMMERCIAL LICENSE
Non-commercial use permitted based on MIT-style terms; commercial use requires prior written authorization.
Copyright (c) 2026 MiniMax
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software for non-commercial purposes, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or provide copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
2. If the Software (or any derivative works thereof) is used for any Commercial Use, you shall prominently display "Built with MiniMax M2.7" on a related website, user interface, blogpost, about page or product documentation.
3. Any Commercial Use of the Software or any derivative work thereof is prohibited without obtaining a separate, prior written authorization from MiniMax.  To request such authorization, please contact api@minimax.io with the subject line "M2.7 licensing".
4. "Commercial Use" means any use of the Software or any derivative work thereof that is primarily intended for commercial advantage or monetary compensation, which includes, without limitation: (i) offering products or services to third parties for a fee, which utilize, incorporate, or rely on the Software or its derivatives, (ii) the commercial use of APIs provided by or for the Software or its derivatives, including to support or enable commercial products, services, or operations, whether in a cloud-based, hosted, or other similar environment, and (iii) the deployment or provision of the Software or its derivatives that have been subjected to post-training, fine-tuning, instruction-tuning, or any other form of modification, for any commercial purpose.
5. Permitted Free Uses. The following uses are expressly permitted free of charge: (a) personal use, including self-hosted deployment for coding, development of applications, agents, tools, integrations, research, experimentation, or other personal purposes; (b) use by non-profit organizations, academic institutions, and researchers for non-commercial research or educational purposes; (c) modification of the Software solely for the uses described in (a) or (b) above.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Appendix: Prohibited Uses
You agree you will not use, or allow others to use, the Software or any derivatives of the Software to:
1. Generate or disseminate content prohibited by applicable laws or regulations.
2. Assist with, engage in or otherwise support any military purpose.
3. Exploit, harm, or attempt to exploit or harm minors.
4. Generate or disseminate false or misleading information with the intent to cause harm.
5. Promote discrimination, hate speech, or harmful behavior against individuals or groups based on race or ethnic origin, religion, disability, age, nationality and national origin, veteran status, sexual orientation, gender or gender identity, caste, immigration status, or any other characteristic that is associated with systemic discrimination or marginalization.
```

Legal-text SHA256 `ae0f00538d47e6894db9452498083b896a60676a26a06dbaa73a59d9aa7de2b9` for `LicenseRef-MiniMax-M2.7-Noncommercial`:

```text
NON-COMMERCIAL LICENSE
Non-commercial use permitted based on MIT-style terms; commercial use requires prior written authorization.
Copyright (c) 2026 MiniMax
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software for non-commercial purposes, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or provide copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
2. If the Software (or any derivative works thereof) is used for any Commercial Use, you shall prominently display "Built with MiniMax M2.7" on a related website, user interface, blogpost, about page or product documentation.
3. Any Commercial Use of the Software or any derivative work thereof is prohibited without obtaining a separate, prior written authorization from MiniMax.  To request such authorization, please contact api@minimax.io with the subject line "M2.7 licensing".
4. "Commercial Use" means any use of the Software or any derivative work thereof that is primarily intended for commercial advantage or monetary compensation, which includes, without limitation: (i) offering products or services to third parties for a fee, which utilize, incorporate, or rely on the Software or its derivatives, (ii) the commercial use of APIs provided by or for the Software or its derivatives, including to support or enable commercial products, services, or operations, whether in a cloud-based, hosted, or other similar environment, and (iii) the deployment or provision of the Software or its derivatives that have been subjected to post-training, fine-tuning, instruction-tuning, or any other form of modification, for any commercial purpose.
5. Permitted Free Uses. The following uses are expressly permitted free of charge: (a) personal use, including self-hosted deployment for coding, development of applications, agents, tools, integrations, research, experimentation, or other personal purposes; (b) use by non-profit organizations, academic institutions, and researchers for non-commercial research or educational purposes; (c) modification of the Software solely for the uses described in (a) or (b) above.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Appendix: Prohibited Uses
You agree you will not use, or allow others to use, the Software or any derivatives of the Software to:
1. Generate or disseminate content prohibited by applicable laws or regulations.
2. Assist with, engage in or otherwise support any military purpose.
3. Exploit, harm, or attempt to exploit or harm minors.
4. Generate or disseminate false or misleading information with the intent to cause harm.
5. Promote discrimination, hate speech, or harmful behavior against individuals or groups based on race or ethnic origin, religion, disability, age, nationality and national origin, veteran status, sexual orientation, gender or gender identity, caste, immigration status, or any other characteristic that is associated with systemic discrimination or marginalization.
```

Legal-text SHA256 `9df48aec80f44b8f9027c0bcd13141201c04666aeeb9da6fbae25b4a5d48f9cd` for `Apache-2.0 AND LicenseRef-MiniMax-M3-Community`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

MINIMAX COMMUNITY LICENSE
Copyright (c) 2026 MiniMax
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software for non-commercial purposes, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or provide copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
2. If the Software (or any derivative works thereof) is used for any Commercial Use for your products or services:
  1. you shall prominently display “Built with MiniMax M3” on a related website, user interface, blogpost, about page or product documentation.
  2. you shall obtain a separate, prior written authorization from MiniMax by contacting api@minimax.io with the subject line “M3 licensing - authorization request”, if such products and services generate more than 20 million US dollars (or equivalent in other currencies) in yearly revenue; otherwise, you only need to send a one-time notice to api@minimax.io with the subject “M3 licensing — notice”.
3. “Commercial Use” means any use of the Software or any derivative work thereof that is primarily intended for commercial advantage or monetary compensation, which includes, without limitation: (i) offering products or services to third parties for a fee, which utilize, incorporate, or rely on the Software or its derivatives, (ii) the commercial use of APIs provided by or for the Software or its derivatives, including to support or enable commercial products, services, or operations, whether in a cloud-based, hosted, or other similar environment, and (iii) the deployment or provision of the Software or its derivatives that have been subjected to post-training, fine-tuning, instruction-tuning, or any other form of modification, for any commercial purpose.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Appendix: Prohibited Uses
You agree you will not use, or allow others to use, the Software or any derivatives of the Software to:
1. Generate or disseminate content prohibited by applicable laws or regulations.
2. Assist with, engage in or otherwise support any military purpose.
3. Exploit, harm, or attempt to exploit or harm minors.
4. Generate or disseminate false or misleading information with the intent to cause harm.
5. Promote discrimination, hate speech, or harmful behavior against individuals or groups based on race or ethnic origin, religion, disability, age, nationality and national origin, veteran status, sexual orientation, gender or gender identity, caste, immigration status, or any other characteristic that is associated with systemic discrimination or marginalization.
```

Legal-text SHA256 `a42faa56fb186ca55475fad45ddee0073da4fe0ad9db517676e4fad209a32029` for `LicenseRef-MiniMax-M3-Community`:

```text
MINIMAX COMMUNITY LICENSE
Copyright (c) 2026 MiniMax
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software for non-commercial purposes, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or provide copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
2. If the Software (or any derivative works thereof) is used for any Commercial Use for your products or services:
  1. you shall prominently display “Built with MiniMax M3” on a related website, user interface, blogpost, about page or product documentation.
  2. you shall obtain a separate, prior written authorization from MiniMax by contacting api@minimax.io with the subject line “M3 licensing - authorization request”, if such products and services generate more than 20 million US dollars (or equivalent in other currencies) in yearly revenue; otherwise, you only need to send a one-time notice to api@minimax.io with the subject “M3 licensing — notice”.
3. “Commercial Use” means any use of the Software or any derivative work thereof that is primarily intended for commercial advantage or monetary compensation, which includes, without limitation: (i) offering products or services to third parties for a fee, which utilize, incorporate, or rely on the Software or its derivatives, (ii) the commercial use of APIs provided by or for the Software or its derivatives, including to support or enable commercial products, services, or operations, whether in a cloud-based, hosted, or other similar environment, and (iii) the deployment or provision of the Software or its derivatives that have been subjected to post-training, fine-tuning, instruction-tuning, or any other form of modification, for any commercial purpose.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Appendix: Prohibited Uses
You agree you will not use, or allow others to use, the Software or any derivatives of the Software to:
1. Generate or disseminate content prohibited by applicable laws or regulations.
2. Assist with, engage in or otherwise support any military purpose.
3. Exploit, harm, or attempt to exploit or harm minors.
4. Generate or disseminate false or misleading information with the intent to cause harm.
5. Promote discrimination, hate speech, or harmful behavior against individuals or groups based on race or ethnic origin, religion, disability, age, nationality and national origin, veteran status, sexual orientation, gender or gender identity, caste, immigration status, or any other characteristic that is associated with systemic discrimination or marginalization.
```

Legal-text SHA256 `981d2ffffb81106c6970f9f6d7f96581b12c873e17e503e89de15d1c69e4ad60` for `Apache-2.0 AND MIT`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
```

Legal-text SHA256 `7c9b48b52decb9837c70f608678129e1ac79e056829c8d1e82e8cdd8aed562f8` for `MIT`:

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
```

Legal-text SHA256 `91a5e7f6977b366f9deab8b02366e79ad2ecd4500f501a740c1601237fddf4ad` for `Apache-2.0 AND MIT`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Legal-text SHA256 `789377a8cfe3a67c77bd2bd9f9cb9b0142a9a905906c91bcaba52092a8d03169` for `MIT`:

```text
MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Legal-text SHA256 `1c779f92a3da59c85945bbaddcc81ba9a4eed6d41dd028302bf9937076cddee9` for `Apache-2.0 AND LicenseRef-Llama-3.1-Community`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

LLAMA 3.1 COMMUNITY LICENSE AGREEMENT
Llama 3.1 Version Release Date: July 23, 2024

“Agreement” means the terms and conditions for use, reproduction, distribution and modification of the
Llama Materials set forth herein.

“Documentation” means the specifications, manuals and documentation accompanying Llama 3.1
distributed by Meta at https://llama.meta.com/doc/overview.

“Licensee” or “you” means you, or your employer or any other person or entity (if you are entering into
this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or
regulations to provide legal consent and that has legal authority to bind your employer or such other
person or entity if you are entering in this Agreement on their behalf.

“Llama 3.1” means the foundational large language models and software and algorithms, including
machine-learning model code, trained model weights, inference-enabling code, training-enabling code,
fine-tuning enabling code and other elements of the foregoing distributed by Meta at
https://llama.meta.com/llama-downloads.

“Llama Materials” means, collectively, Meta’s proprietary Llama 3.1 and Documentation (and any
portion thereof) made available under this Agreement.

“Meta” or “we” means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your
principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located
outside of the EEA or Switzerland).

By clicking “I Accept” below or by using or distributing any portion or element of the Llama Materials,
you agree to be bound by this Agreement.

1. License Rights and Redistribution.

  a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free
limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Llama
Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the
Llama Materials.

  b. Redistribution and Use.

      i. If you distribute or make available the Llama Materials (or any derivative works
thereof), or a product or service (including another AI model) that contains any of them, you shall (A)
provide a copy of this Agreement with any such Llama Materials; and (B) prominently display “Built with
Llama” on a related website, user interface, blogpost, about page, or product documentation. If you use
the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or
otherwise improve an AI model, which is distributed or made available, you shall also include “Llama” at
the beginning of any such AI model name.

      ii. If you receive Llama Materials, or any derivative works thereof, from a Licensee as part
of an integrated end user product, then Section 2 of this Agreement will not apply to you.

      iii. You must retain in all copies of the Llama Materials that you distribute the following
attribution notice within a “Notice” text file distributed as a part of such copies: “Llama 3.1 is
licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights
Reserved.”

      iv. Your use of the Llama Materials must comply with applicable laws and regulations
(including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Llama
Materials (available at https://llama.meta.com/llama3_1/use-policy), which is hereby incorporated by
reference into this Agreement.

2. Additional Commercial Terms. If, on the Llama 3.1 version release date, the monthly active users
of the products or services made available by or for Licensee, or Licensee’s affiliates, is greater than 700
million monthly active users in the preceding calendar month, you must request a license from Meta,
which Meta may grant to you in its sole discretion, and you are not authorized to exercise any of the
rights under this Agreement unless or until Meta otherwise expressly grants you such rights.

3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY
OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF
ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED,
INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR
DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE LLAMA MATERIALS AND
ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE LLAMA MATERIALS AND ANY OUTPUT AND
RESULTS.

4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING
OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL,
INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED
OF THE POSSIBILITY OF ANY OF THE FOREGOING.

5. Intellectual Property.

  a. No trademark licenses are granted under this Agreement, and in connection with the Llama
Materials, neither Meta nor Licensee may use any name or mark owned by or associated with the other
or any of its affiliates, except as required for reasonable and customary use in describing and
redistributing the Llama Materials or as set forth in this Section 5(a). Meta hereby grants you a license to
use “Llama” (the “Mark”) solely as required to comply with the last sentence of Section 1.b.i. You will
comply with Meta’s brand guidelines (currently accessible at
https://about.meta.com/brand/resources/meta/company-brand/ ). All goodwill arising out of your use
of the Mark will inure to the benefit of Meta.

  b. Subject to Meta’s ownership of Llama Materials and derivatives made by or for Meta, with
respect to any derivative works and modifications of the Llama Materials that are made by you, as
between you and Meta, you are and will be the owner of such derivative works and modifications.

  c. If you institute litigation or other proceedings against Meta or any entity (including a
cross-claim or counterclaim in a lawsuit) alleging that the Llama Materials or Llama 3.1 outputs or
results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other
rights owned or licensable by you, then any licenses granted to you under this Agreement shall
terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold
harmless Meta from and against any claim by any third party arising out of or related to your use or
distribution of the Llama Materials.

6. Term and Termination. The term of this Agreement will commence upon your acceptance of this
Agreement or access to the Llama Materials and will continue in full force and effect until terminated in
accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in
breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete
and cease use of the Llama Materials. Sections 3, 4 and 7 shall survive the termination of this
Agreement.

7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of
the State of California without regard to choice of law principles, and the UN Convention on Contracts
for the International Sale of Goods does not apply to this Agreement. The courts of California shall have
exclusive jurisdiction of any dispute arising out of this Agreement.
```

Legal-text SHA256 `f247b541404c402d906d71f47818179d4fe79d426f3a13514759680669f710a2` for `LicenseRef-Llama-3.1-Community`:

```text
LLAMA 3.1 COMMUNITY LICENSE AGREEMENT
Llama 3.1 Version Release Date: July 23, 2024

“Agreement” means the terms and conditions for use, reproduction, distribution and modification of the
Llama Materials set forth herein.

“Documentation” means the specifications, manuals and documentation accompanying Llama 3.1
distributed by Meta at https://llama.meta.com/doc/overview.

“Licensee” or “you” means you, or your employer or any other person or entity (if you are entering into
this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or
regulations to provide legal consent and that has legal authority to bind your employer or such other
person or entity if you are entering in this Agreement on their behalf.

“Llama 3.1” means the foundational large language models and software and algorithms, including
machine-learning model code, trained model weights, inference-enabling code, training-enabling code,
fine-tuning enabling code and other elements of the foregoing distributed by Meta at
https://llama.meta.com/llama-downloads.

“Llama Materials” means, collectively, Meta’s proprietary Llama 3.1 and Documentation (and any
portion thereof) made available under this Agreement.

“Meta” or “we” means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your
principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located
outside of the EEA or Switzerland).

By clicking “I Accept” below or by using or distributing any portion or element of the Llama Materials,
you agree to be bound by this Agreement.

1. License Rights and Redistribution.

  a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free
limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Llama
Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the
Llama Materials.

  b. Redistribution and Use.

      i. If you distribute or make available the Llama Materials (or any derivative works
thereof), or a product or service (including another AI model) that contains any of them, you shall (A)
provide a copy of this Agreement with any such Llama Materials; and (B) prominently display “Built with
Llama” on a related website, user interface, blogpost, about page, or product documentation. If you use
the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or
otherwise improve an AI model, which is distributed or made available, you shall also include “Llama” at
the beginning of any such AI model name.

      ii. If you receive Llama Materials, or any derivative works thereof, from a Licensee as part
of an integrated end user product, then Section 2 of this Agreement will not apply to you.

      iii. You must retain in all copies of the Llama Materials that you distribute the following
attribution notice within a “Notice” text file distributed as a part of such copies: “Llama 3.1 is
licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights
Reserved.”

      iv. Your use of the Llama Materials must comply with applicable laws and regulations
(including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Llama
Materials (available at https://llama.meta.com/llama3_1/use-policy), which is hereby incorporated by
reference into this Agreement.

2. Additional Commercial Terms. If, on the Llama 3.1 version release date, the monthly active users
of the products or services made available by or for Licensee, or Licensee’s affiliates, is greater than 700
million monthly active users in the preceding calendar month, you must request a license from Meta,
which Meta may grant to you in its sole discretion, and you are not authorized to exercise any of the
rights under this Agreement unless or until Meta otherwise expressly grants you such rights.

3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY
OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF
ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED,
INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR
DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE LLAMA MATERIALS AND
ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE LLAMA MATERIALS AND ANY OUTPUT AND
RESULTS.

4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING
OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL,
INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED
OF THE POSSIBILITY OF ANY OF THE FOREGOING.

5. Intellectual Property.

  a. No trademark licenses are granted under this Agreement, and in connection with the Llama
Materials, neither Meta nor Licensee may use any name or mark owned by or associated with the other
or any of its affiliates, except as required for reasonable and customary use in describing and
redistributing the Llama Materials or as set forth in this Section 5(a). Meta hereby grants you a license to
use “Llama” (the “Mark”) solely as required to comply with the last sentence of Section 1.b.i. You will
comply with Meta’s brand guidelines (currently accessible at
https://about.meta.com/brand/resources/meta/company-brand/ ). All goodwill arising out of your use
of the Mark will inure to the benefit of Meta.

  b. Subject to Meta’s ownership of Llama Materials and derivatives made by or for Meta, with
respect to any derivative works and modifications of the Llama Materials that are made by you, as
between you and Meta, you are and will be the owner of such derivative works and modifications.

  c. If you institute litigation or other proceedings against Meta or any entity (including a
cross-claim or counterclaim in a lawsuit) alleging that the Llama Materials or Llama 3.1 outputs or
results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other
rights owned or licensable by you, then any licenses granted to you under this Agreement shall
terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold
harmless Meta from and against any claim by any third party arising out of or related to your use or
distribution of the Llama Materials.

6. Term and Termination. The term of this Agreement will commence upon your acceptance of this
Agreement or access to the Llama Materials and will continue in full force and effect until terminated in
accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in
breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete
and cease use of the Llama Materials. Sections 3, 4 and 7 shall survive the termination of this
Agreement.

7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of
the State of California without regard to choice of law principles, and the UN Convention on Contracts
for the International Sale of Goods does not apply to this Agreement. The courts of California shall have
exclusive jurisdiction of any dispute arising out of this Agreement.
```

Legal-text SHA256 `93663f288a42860b69743a59ba301a2b7fc9e73792ab6d27e26225a183d90e5b` for `Apache-2.0 AND LicenseRef-Kimi-K2-Instruct-Modified-MIT`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Modified MIT License

Copyright (c) 2025 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Our only modification part is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services that have
more than 100 million monthly active users, or more than 20 million US dollars
(or equivalent in other currencies) in monthly revenue, you shall prominently
display "Kimi K2" on the user interface of such product or service.
```

Legal-text SHA256 `7aadc2c3b398bd1801247b4baa24d353fc6c7a3ce4d52ed19122bded6e33d1fa` for `LicenseRef-Kimi-K2-Instruct-Modified-MIT`:

```text
Modified MIT License

Copyright (c) 2025 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Our only modification part is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services that have
more than 100 million monthly active users, or more than 20 million US dollars
(or equivalent in other currencies) in monthly revenue, you shall prominently
display "Kimi K2" on the user interface of such product or service.
```

Legal-text SHA256 `b9187cd88244972781dccb574056705994b92c3de30fd6500eaf25765dd70438` for `Apache-2.0 AND LicenseRef-Kimi-K2.5-Modified-MIT`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Modified MIT License

Copyright (c) 2026 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Our only modification part is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services that have
more than 100 million monthly active users, or more than 20 million US dollars
(or equivalent in other currencies) in monthly revenue, you shall prominently
display "Kimi K2.5" on the user interface of such product or service.
```

Legal-text SHA256 `8b22e3918250a78292ec17e8b2596808318d044684e866719e9d8a1a928ea46d` for `LicenseRef-Kimi-K2.5-Modified-MIT`:

```text
Modified MIT License

Copyright (c) 2026 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Our only modification part is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services that have
more than 100 million monthly active users, or more than 20 million US dollars
(or equivalent in other currencies) in monthly revenue, you shall prominently
display "Kimi K2.5" on the user interface of such product or service.
```

Legal-text SHA256 `a270c9a6e214d3f09f332a11ee6ef513292fac94a19928e913940f4d837e2a39` for `Apache-2.0 AND LicenseRef-Kimi-K3`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Kimi K3 License

Copyright (c) 2026 Moonshot AI

Permission is hereby granted, free of charge, to any person (the "Licensee")
obtaining a copy of this software — including the model weights, parameters,
configuration files, inference and training code, and associated documentation
(collectively, the "Software") — to deal in the Software without restriction.
This includes, without limitation, the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software; to run,
deploy, fine-tune, or otherwise modify the Software and create derivative works
from it; and to permit persons to whom the Software is furnished to do so, in
each case subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. Licensee's use of the
Software must comply with applicable laws and regulations.

2. "Model as a Service" means giving a third party access to language model
inference or fine-tuning (e.g., via API) in a manner that allows such third
party to exercise meaningful control over the inputs, parameters, or training
data. This does not include (a) end-user products with model capabilities solely
embedded within specific features or harnesses, or (b) mere relaying of requests
to models hosted by others.

If the Licensee or any of its affiliates operates a Model as a Service business,
and the aggregate revenue of the Licensee and its affiliates exceeds 20 million
US dollars (or the equivalent in other currencies) in total over any consecutive
12 months, the Licensee must enter into a separate agreement with Moonshot AI
before using the Software or its derivative works for any commercial purpose.

3. If the Software (or any derivative works thereof) is used for any of the
Licensee's commercial products or services that have more than 100 million
monthly active users, or more than 20 million US dollars (or equivalent in other
currencies) in monthly revenue, "Kimi K3" must be prominently displayed on the
user interface of such product or service.

4. The requirements set forth in Sections 2 and 3 do not apply to: (a) internal
use of the Software, defined as any use that does not make the Software, its
outputs, or its underlying capabilities available to third parties; or (b) any
use of the Software accessed through Moonshot AI's official products or
certified inference partners.

5. THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS”
BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
AND NONINFRINGEMENT. IN NO EVENT SHALL MOONSHOT AI OR ITS AFFILIATES OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For any questions regarding this license, please contact <license@moonshot.ai>.
```

Legal-text SHA256 `2867133c376aa3deb3a95732e30fa77a9dcc31686bdf9191f405de7e46e84652` for `LicenseRef-Kimi-K3`:

```text
Kimi K3 License

Copyright (c) 2026 Moonshot AI

Permission is hereby granted, free of charge, to any person (the "Licensee")
obtaining a copy of this software — including the model weights, parameters,
configuration files, inference and training code, and associated documentation
(collectively, the "Software") — to deal in the Software without restriction.
This includes, without limitation, the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software; to run,
deploy, fine-tune, or otherwise modify the Software and create derivative works
from it; and to permit persons to whom the Software is furnished to do so, in
each case subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. Licensee's use of the
Software must comply with applicable laws and regulations.

2. "Model as a Service" means giving a third party access to language model
inference or fine-tuning (e.g., via API) in a manner that allows such third
party to exercise meaningful control over the inputs, parameters, or training
data. This does not include (a) end-user products with model capabilities solely
embedded within specific features or harnesses, or (b) mere relaying of requests
to models hosted by others.

If the Licensee or any of its affiliates operates a Model as a Service business,
and the aggregate revenue of the Licensee and its affiliates exceeds 20 million
US dollars (or the equivalent in other currencies) in total over any consecutive
12 months, the Licensee must enter into a separate agreement with Moonshot AI
before using the Software or its derivative works for any commercial purpose.

3. If the Software (or any derivative works thereof) is used for any of the
Licensee's commercial products or services that have more than 100 million
monthly active users, or more than 20 million US dollars (or equivalent in other
currencies) in monthly revenue, "Kimi K3" must be prominently displayed on the
user interface of such product or service.

4. The requirements set forth in Sections 2 and 3 do not apply to: (a) internal
use of the Software, defined as any use that does not make the Software, its
outputs, or its underlying capabilities available to third parties; or (b) any
use of the Software accessed through Moonshot AI's official products or
certified inference partners.

5. THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS”
BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
AND NONINFRINGEMENT. IN NO EVENT SHALL MOONSHOT AI OR ITS AFFILIATES OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For any questions regarding this license, please contact <license@moonshot.ai>.
```

Legal-text SHA256 `0a3e345a54ab255dd511d35ac2404684ad8c60fdb8c1bab25356464d86afc226` for `Apache-2.0 AND MIT`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

MIT License

Copyright (c) 2026 Zhipu AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Legal-text SHA256 `c3b8527b08443c6fc04be79d8da2e7c37679c57f0564269af8b47a6ba30b1d7c` for `MIT`:

```text
MIT License

Copyright (c) 2026 Zhipu AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Legal-text SHA256 `7da938f77330680af8e9b70f0c7a0817f2ecc0fa1c212d333e71550755a46b5e` for `MIT`:

```text
MIT License

Copyright (c) 2026 Z.AI Co., Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Legal-text SHA256 `a6c205937203b51cf56acfed485534b4a1a738faa09d510da3d571a3c75c2a36` for `Apache-2.0 AND LicenseRef-Kimi-K2.6-Modified-MIT`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Modified MIT License

Copyright (c) 2026 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Our only modification part is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services that have
more than 100 million monthly active users, or more than 20 million US dollars
(or equivalent in other currencies) in monthly revenue, you shall prominently
display "Kimi K2.6" on the user interface of such product or service.
```

Legal-text SHA256 `941ddde5de6c074eac890c67856db540e48bd4653e34d4d9e38d1a9fae4344b8` for `LicenseRef-Kimi-K2.6-Modified-MIT`:

```text
Modified MIT License

Copyright (c) 2026 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Our only modification part is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services that have
more than 100 million monthly active users, or more than 20 million US dollars
(or equivalent in other currencies) in monthly revenue, you shall prominently
display "Kimi K2.6" on the user interface of such product or service.
```

Legal-text SHA256 `86c40b0e4d23e36595522ec3cd98d355518bfdbfeb53aa8d161139e93bf6e58c` for `Apache-2.0 AND LicenseRef-Kimi-K2.7-Code-Modified-MIT`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Modified MIT License

Copyright (c) 2026 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Our only modification part is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services that have
more than 100 million monthly active users, or more than 20 million US dollars
(or equivalent in other currencies) in monthly revenue, you shall prominently
display "Kimi K2.7 Code" on the user interface of such product or service.
```

Legal-text SHA256 `0bc1a42b295528b3d531d8bac9d831836c9b22bf43fc40733530da9186b753d8` for `LicenseRef-Kimi-K2.7-Code-Modified-MIT`:

```text
Modified MIT License

Copyright (c) 2026 Moonshot AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Our only modification part is that, if the Software (or any derivative works
thereof) is used for any of your commercial products or services that have
more than 100 million monthly active users, or more than 20 million US dollars
(or equivalent in other currencies) in monthly revenue, you shall prominently
display "Kimi K2.7 Code" on the user interface of such product or service.
```

Legal-text SHA256 `b5c7be2418fefa3283f75e19d8a85f6429f82035e15c4affe3a549469cc5669c` for `Apache-2.0 AND LicenseRef-Llama-3.3-Community`:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTICE: The file
python/aisim/src/aisim_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json
is a verbatim copy of config.json from the DeepSeek-V3.2 model repository at
commit c69397ecfd1fd142e90e3fbad51f4c7e40b9f3d3 and is licensed under the MIT
License. Its license text, and attribution for other third-party material, is
provided in THIRD_PARTY_NOTICES.md.

Except for the third-party material identified in that notice file, this
codebase is licensed under the Apache License 2.0 as described below.

                                Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

**LLAMA 3.3 COMMUNITY LICENSE AGREEMENT**

 Llama 3.3 Version Release Date: December 6, 2024

“**Agreement**” means the terms and conditions for use, reproduction, distribution and modification of the Llama Materials set forth herein.

“**Documentation**” means the specifications, manuals and documentation accompanying Llama 3.3 distributed by Meta at [https://www.llama.com/docs/overview](https://llama.com/docs/overview).

“**Licensee**” or “**you**” means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

“**Llama 3.3**” means the foundational large language models and software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code and other elements of the foregoing distributed by Meta at [https://www.llama.com/llama-downloads](https://www.llama.com/llama-downloads).

“**Llama Materials**” means, collectively, Meta’s proprietary Llama 3.3 and Documentation (and any portion thereof) made available under this Agreement.

“**Meta**” or “**we**” means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).

By clicking “I Accept” below or by using or distributing any portion or element of the Llama Materials, you agree to be bound by this Agreement.

1\. **License Rights and Redistribution**.

a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Llama Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Llama Materials.

b. Redistribution and Use.

i. If you distribute or make available the Llama Materials (or any derivative works thereof), or a product or service (including another AI model) that contains any of them, you shall (A) provide a copy of this Agreement with any such Llama Materials; and (B) prominently display “Built with Llama” on a related website, user interface, blogpost, about page, or product documentation. If you use the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include “Llama” at the beginning of any such AI model name.

ii. If you receive Llama Materials, or any derivative works thereof, from a Licensee as part of an integrated end user product, then Section 2 of this Agreement will not apply to you.

iii. You must retain in all copies of the Llama Materials that you distribute the following attribution notice within a “Notice” text file distributed as a part of such copies: “Llama 3.3 is licensed under the Llama 3.3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

iv. Your use of the Llama Materials must comply with applicable laws and regulations (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Llama Materials (available at [https://www.llama.com/llama3\_3/use-policy](https://www.llama.com/llama3_3/use-policy)), which is hereby incorporated by reference into this Agreement.

2\. **Additional Commercial Terms**. If, on the Llama 3.3 version release date, the monthly active users of the products or services made available by or for Licensee, or Licensee’s affiliates, is greater than 700 million monthly active users in the preceding calendar month, you must request a license from Meta, which Meta may grant to you in its sole discretion, and you are not authorized to exercise any of the rights under this Agreement unless or until Meta otherwise expressly grants you such rights.

3**. Disclaimer of Warranty**. UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE LLAMA MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS.

4\. **Limitation of Liability**. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.

5\. **Intellectual Property**.

a. No trademark licenses are granted under this Agreement, and in connection with the Llama Materials, neither Meta nor Licensee may use any name or mark owned by or associated with the other or any of its affiliates, except as required for reasonable and customary use in describing and redistributing the Llama Materials or as set forth in this Section 5(a). Meta hereby grants you a license to use “Llama” (the “Mark”) solely as required to comply with the last sentence of Section 1.b.i. You will comply with Meta’s brand guidelines (currently accessible at [https://about.meta.com/brand/resources/meta/company-brand/](https://about.meta.com/brand/resources/meta/company-brand/)[)](https://en.facebookbrand.com/). All goodwill arising out of your use of the Mark will inure to the benefit of Meta.

b. Subject to Meta’s ownership of Llama Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the Llama Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.

c. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Llama Materials or Llama 3.3 outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the Llama Materials.

6\. **Term and Termination**. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Llama Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Llama Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement.

7\. **Governing Law and Jurisdiction**. This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement.
```

Legal-text SHA256 `1beaed4a9925af5f6d4d07eebcc351c2c59b00bbb3cdeac59d780adffadcfcdb` for `LicenseRef-Llama-3.3-Community`:

```text
**LLAMA 3.3 COMMUNITY LICENSE AGREEMENT**

 Llama 3.3 Version Release Date: December 6, 2024

“**Agreement**” means the terms and conditions for use, reproduction, distribution and modification of the Llama Materials set forth herein.

“**Documentation**” means the specifications, manuals and documentation accompanying Llama 3.3 distributed by Meta at [https://www.llama.com/docs/overview](https://llama.com/docs/overview).

“**Licensee**” or “**you**” means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

“**Llama 3.3**” means the foundational large language models and software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code and other elements of the foregoing distributed by Meta at [https://www.llama.com/llama-downloads](https://www.llama.com/llama-downloads).

“**Llama Materials**” means, collectively, Meta’s proprietary Llama 3.3 and Documentation (and any portion thereof) made available under this Agreement.

“**Meta**” or “**we**” means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).

By clicking “I Accept” below or by using or distributing any portion or element of the Llama Materials, you agree to be bound by this Agreement.

1\. **License Rights and Redistribution**.

a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Llama Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Llama Materials.

b. Redistribution and Use.

i. If you distribute or make available the Llama Materials (or any derivative works thereof), or a product or service (including another AI model) that contains any of them, you shall (A) provide a copy of this Agreement with any such Llama Materials; and (B) prominently display “Built with Llama” on a related website, user interface, blogpost, about page, or product documentation. If you use the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include “Llama” at the beginning of any such AI model name.

ii. If you receive Llama Materials, or any derivative works thereof, from a Licensee as part of an integrated end user product, then Section 2 of this Agreement will not apply to you.

iii. You must retain in all copies of the Llama Materials that you distribute the following attribution notice within a “Notice” text file distributed as a part of such copies: “Llama 3.3 is licensed under the Llama 3.3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”

iv. Your use of the Llama Materials must comply with applicable laws and regulations (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Llama Materials (available at [https://www.llama.com/llama3\_3/use-policy](https://www.llama.com/llama3_3/use-policy)), which is hereby incorporated by reference into this Agreement.

2\. **Additional Commercial Terms**. If, on the Llama 3.3 version release date, the monthly active users of the products or services made available by or for Licensee, or Licensee’s affiliates, is greater than 700 million monthly active users in the preceding calendar month, you must request a license from Meta, which Meta may grant to you in its sole discretion, and you are not authorized to exercise any of the rights under this Agreement unless or until Meta otherwise expressly grants you such rights.

3**. Disclaimer of Warranty**. UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE LLAMA MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS.

4\. **Limitation of Liability**. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.

5\. **Intellectual Property**.

a. No trademark licenses are granted under this Agreement, and in connection with the Llama Materials, neither Meta nor Licensee may use any name or mark owned by or associated with the other or any of its affiliates, except as required for reasonable and customary use in describing and redistributing the Llama Materials or as set forth in this Section 5(a). Meta hereby grants you a license to use “Llama” (the “Mark”) solely as required to comply with the last sentence of Section 1.b.i. You will comply with Meta’s brand guidelines (currently accessible at [https://about.meta.com/brand/resources/meta/company-brand/](https://about.meta.com/brand/resources/meta/company-brand/)[)](https://en.facebookbrand.com/). All goodwill arising out of your use of the Mark will inure to the benefit of Meta.

b. Subject to Meta’s ownership of Llama Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the Llama Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.

c. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Llama Materials or Llama 3.3 outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the Llama Materials.

6\. **Term and Termination**. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Llama Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Llama Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement.

7\. **Governing Law and Jurisdiction**. This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement.
```
