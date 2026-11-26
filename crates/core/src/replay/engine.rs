// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0
// Modified grouped handoff validation follows vLLM's NIXL handshake at v0.31.0, v0.32.0 and v0.33.0.
// https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py
// https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py
// https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py

//! Default generalized-engine construction for offline replay.
//!
//! This module contains configuration and conversion helpers only. Scheduler
//! state lives in `aisim_core::engine`; virtual time and worker lifecycle live
//! in the moved aggregated/disaggregated replay runtimes.

use std::num::NonZeroU32;
use std::sync::Arc;

use crate::engine::belady::BeladyOracle;
use crate::engine::generalized::EngineIdentity;
use crate::engine::{
    Backend, Engine, EngineConfig, EngineFactory, G2Binding, G2Registry, G2Scope, KvEvictionPolicy,
    TimingModel, TimingModelConfig, WorkerType,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::replay::OfflineDisaggReplayConfig;
use crate::replay::components::{
    AdmissionQueue, NoReplayMetadata, ReplayEngineObservation, ReplayMode,
};
use crate::replay::core::EngineEventBatch;
use crate::replay::core::round_robin::PoolRoundRobinPlacement;
use crate::replay::disagg::DisaggRuntimeImpl;
use crate::replay::error::runtime_error;
use crate::replay::protocol::DirectRequest;
use crate::replay::{
    ReplayError, ReplayReport, ReplayResult, ReplaySpec, ReplayTopology, Replayer, WorkerStage,
};

fn default_dp_size() -> u32 {
    1
}

fn default_parallel_size() -> u32 {
    1
}

/// Serializable execution-time descriptor for the AISim engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReplayEngineConfig {
    /// Belady ranks worker-local victims by global future input demand. It does
    /// not forecast routing, output reuse, later chunks, or recomputation, and
    /// therefore promises neither worker-local optimality nor higher throughput.
    /// Lookahead changes eviction priority only; execution remains causal.
    pub kv_eviction_policy: KvEvictionPolicy,
    #[serde(default = "default_dp_size")]
    pub dp_size: u32,
    #[serde(default = "default_parallel_size")]
    pub tensor_parallel_size: u32,
    /// Prefill CP adds physical ranks; decode CP partitions existing TP ranks.
    #[serde(default = "default_parallel_size")]
    pub prefill_context_parallel_size: u32,
    /// Whether `rank.num_gpu_blocks` came from the user rather than an
    /// upstream capacity estimator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_gpu_blocks_is_explicit: Option<bool>,
    pub rank: EngineConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill: Option<ReplayRoleConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode: Option<ReplayRoleConfig>,
}

impl Default for ReplayEngineConfig {
    fn default() -> Self {
        Self {
            kv_eviction_policy: KvEvictionPolicy::Lru,
            dp_size: 1,
            tensor_parallel_size: 1,
            prefill_context_parallel_size: 1,
            num_gpu_blocks_is_explicit: None,
            rank: EngineConfig::default(),
            prefill: None,
            decode: None,
        }
    }
}

/// Rank-group descriptor for one disaggregated role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReplayRoleConfig {
    #[serde(default = "default_dp_size")]
    pub dp_size: u32,
    #[serde(default = "default_parallel_size")]
    pub tensor_parallel_size: u32,
    /// Prefill CP adds physical ranks; decode CP partitions existing TP ranks.
    #[serde(default = "default_parallel_size")]
    pub prefill_context_parallel_size: u32,
    /// Whether `rank.num_gpu_blocks` came from the user rather than an
    /// upstream capacity estimator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_gpu_blocks_is_explicit: Option<bool>,
    pub rank: EngineConfig,
}

impl Default for ReplayRoleConfig {
    fn default() -> Self {
        Self {
            dp_size: 1,
            tensor_parallel_size: 1,
            prefill_context_parallel_size: 1,
            num_gpu_blocks_is_explicit: None,
            rank: EngineConfig::default(),
        }
    }
}

/// An injected timing model can carry either spelling from the canonical timing schema.
fn timing_alias<'a>(
    config: &'a Value,
    name: &str,
    legacy: &str,
) -> anyhow::Result<Option<&'a Value>> {
    let value = config.get(name).filter(|v| !v.is_null());
    let fallback = config.get(legacy).filter(|v| !v.is_null());
    if let (Some(a), Some(b)) = (value, fallback) {
        let equal = a == b
            || name.ends_with("_quant_mode")
                && a.as_str()
                    .zip(b.as_str())
                    .is_some_and(|(x, y)| x.eq_ignore_ascii_case(y));
        anyhow::ensure!(equal, "conflicting timing aliases {name} and {legacy}");
    }
    Ok(value.or(fallback))
}

impl ReplayRoleConfig {
    /// Match the rank, estimator, and selected deployment before comparing NIXL's
    /// model ID, KV dtype, attention backend and model dtype across the pair.
    fn grouped_transfer_identity(
        &self,
    ) -> anyhow::Result<[crate::perfmodel::fpm::NixlIdentity; 2]> {
        use crate::perfmodel::fpm::NixlIdentity;
        use crate::perfmodel::{ForwardPassPerfModelConfig, FpmCacheGroup, FpmRuntimeMemoryConfig};
        use anyhow::{Context, bail, ensure};

        let empty = || {
            let identity = || NixlIdentity {
                phase_width: Some(u64::from(self.tensor_parallel_size) * u64::from(self.dp_size)),
                tp_size: Some(u64::from(self.tensor_parallel_size)),
                modeled_version: self.rank.vllm_version.clone(),
                ..Default::default()
            };
            Ok([identity(), identity()])
        };
        let TimingModelConfig::External { provider, config } = &self.rank.timing_model else {
            return empty();
        };
        if !matches!(provider.as_str(), "aic" | "ais") {
            return empty();
        }
        // Duration may be supplied by the caller, bypassing the ordinary timing descriptor reader.
        for (name, legacy) in [
            ("gemm_quant_mode", "gemm_dtype"),
            ("moe_quant_mode", "moe_dtype"),
            ("fmha_quant_mode", "fmha_dtype"),
            ("fpm_fmha_quant_mode", "fpm_fmha_dtype"),
            ("comm_quant_mode", "comm_dtype"),
            ("enable_shared_layer", "shared_layer"),
        ] {
            timing_alias(config, name, legacy)?;
        }
        let known = |value: &&str| {
            !value.eq_ignore_ascii_case("auto") && !value.eq_ignore_ascii_case("unknown")
        };
        let role = serde_json::json!(self.rank.worker_type);
        ensure!(
            config["worker_type"].is_null() || config["worker_type"] == role,
            "estimator worker_type does not match replay role"
        );
        // Injected providers may omit identity fields. Use the rank's geometry for
        // those defaults; parse recorded sizes and their aliases with the FPM types.
        let model = timing_alias(config, "model", "model_path")?.and_then(Value::as_str);
        let mut selector = serde_json::json!({
            "model": model.unwrap_or_default(), "system": config["system"].as_str().unwrap_or_default(),
            "backend": config["backend"].as_str().unwrap_or("vllm"), "worker_type": role,
            "backend_version": config["backend_version"], "tp": self.tensor_parallel_size,
            "attention_dp": self.dp_size, "cp_size": config["cp_size"], "dcp": config["dcp"],
            "moe_tp_size": config["moe_tp_size"], "moe_ep_size": config["moe_ep_size"]
        });
        for (key, legacy) in [
            ("tp", "tp_size"),
            ("pp", "pp_size"),
            ("attention_dp", "attention_dp_size"),
        ] {
            if let Some(value) = timing_alias(config, key, legacy)? {
                selector[key] = value.clone();
            }
        }
        let timing: ForwardPassPerfModelConfig =
            serde_json::from_value(selector).context("invalid grouped timing identity")?;
        ensure!(
            timing.backend == crate::BackendKind::Vllm,
            "grouped rank requires timing backend=vllm"
        );
        for (field, recorded, rank) in [
            ("tp", timing.tp, self.tensor_parallel_size),
            ("pp", timing.pp, 1),
            ("attention_dp", timing.attention_dp, self.dp_size),
            (
                "cp_size",
                timing.cp_size.unwrap_or(1),
                self.prefill_context_parallel_size,
            ),
        ] {
            let unknown = field != "cp_size" && recorded == crate::UNKNOWN_SIZE;
            ensure!(
                unknown || recorded == rank,
                "grouped timing {field} ({recorded}) must match the replay rank ({rank})"
            );
        }
        let profile = config.get("fpm_profile").filter(|v| !v.is_null());
        ensure!(
            timing.dcp.is_none_or(|size| size == 1 && profile.is_none()),
            "grouped replay does not model timing dcp for an FPM profile or a striped cache"
        );
        let deployment = match profile {
            Some(value) => Some(timing.matching_fpm_deployment(
                value.as_object().context("fpm_profile must be an object")?,
            )?),
            None => None,
        };
        let memory = deployment.and_then(|d| d.pointer("/resources/runtime_memory"));
        for (field, version) in [
            ("timing backend_version", timing.backend_version.as_deref()),
            (
                "runtime_memory.runtime_framework_version",
                memory.and_then(|m| m["runtime_framework_version"].as_str()),
            ),
        ] {
            if let Some(version) = version.filter(known)
                && let Some(rank_version) = &self.rank.vllm_version
                && !FpmRuntimeMemoryConfig::matches_grouped_version(
                    version,
                    rank_version,
                    self.rank.enable_prefix_caching || field == "timing backend_version",
                )
            {
                bail!("grouped rank vllm_version ({rank_version}) must match {field} ({version})");
            }
        }
        if let Some(resources) = deployment.and_then(|d| d.get("resources")) {
            ensure!(
                resources["cache_layout"]
                    .as_str()
                    .is_none_or(|layout| layout == "grouped"),
                "grouped rank requires a grouped selected deployment"
            );
            if let Some(groups) = resources.get("cache_groups") {
                let measured: Vec<FpmCacheGroup> = serde_json::from_value(groups.clone())
                    .context("invalid selected cache groups")?;
                ensure!(
                    measured == self.rank.kv_cache_groups,
                    "grouped rank kv_cache_groups must match the selected deployment"
                );
            }
        }
        let source = |field, declared: Option<&str>, observed: Option<&str>| -> anyhow::Result<_> {
            let declared = declared.filter(known);
            let observed = observed.filter(known);
            if let (Some(a), Some(b)) = (declared, observed)
                && !a.eq_ignore_ascii_case(b)
            {
                bail!("grouped timing {field} ({a}) must match the selected deployment ({b})");
            }
            Ok(declared.or(observed).map(str::to_owned))
        };
        let model_dtype = config["model_dtype"]
            .as_str()
            .and_then(FpmRuntimeMemoryConfig::normalize_model_dtype);
        let mut observed = NixlIdentity::observed(
            memory.and_then(|m| m["model_dtype"].as_str()),
            memory
                .and_then(|m| m["provenance"].as_str())
                .unwrap_or_default(),
            Some(u64::from(self.tensor_parallel_size) * u64::from(self.dp_size)),
            Some(u64::from(self.tensor_parallel_size)),
        )?;
        let backend = source(
            "attention_backend",
            config["attention_backend"].as_str(),
            deployment.and_then(|d| d["attention_backend"].as_str()),
        )?;
        let labels = NixlIdentity {
            model: model.map(str::to_owned),
            kv_cache_dtype: source(
                "kv_cache_dtype",
                timing_alias(config, "kvcache_quant_mode", "kv_cache_dtype")?
                    .and_then(Value::as_str),
                deployment.and_then(|d| d["kv_cache_dtype"].as_str()),
            )?
            .map(|s| s.to_ascii_lowercase()),
            attention_backend: backend.map(|s| s.to_ascii_lowercase()),
            model_dtype,
            ..NixlIdentity::default()
        };
        observed.model_dtype = source(
            "model_dtype",
            labels.model_dtype.as_deref(),
            observed.model_dtype.as_deref(),
        )?;
        observed.select(
            role.as_str().unwrap_or_default(),
            labels.attention_backend.as_deref(),
            labels.kv_cache_dtype.as_deref(),
            self.rank.vllm_version.as_deref(),
            memory.and_then(|m| m["runtime_framework_version"].as_str()),
            self.rank.enable_prefix_caching,
        )?;
        Ok([labels, observed])
    }
}

impl ReplayEngineConfig {
    pub(crate) fn parse(value: &Value) -> ReplayResult<Self> {
        if value.is_null() {
            return Ok(Self::default());
        }
        serde_json::from_value(value.clone()).map_err(|error| {
            ReplayError::InvalidSpec(format!("invalid native engine descriptor: {error}"))
        })
    }

    pub(crate) fn role(&self, stage: WorkerStage) -> ReplayRoleConfig {
        let mut role = match stage {
            WorkerStage::Aggregated => ReplayRoleConfig {
                dp_size: self.dp_size,
                tensor_parallel_size: self.tensor_parallel_size,
                prefill_context_parallel_size: self.prefill_context_parallel_size,
                num_gpu_blocks_is_explicit: self.num_gpu_blocks_is_explicit,
                rank: self.rank.clone(),
            },
            WorkerStage::Prefill => self.prefill.clone().unwrap_or_else(|| ReplayRoleConfig {
                dp_size: self.dp_size,
                tensor_parallel_size: self.tensor_parallel_size,
                prefill_context_parallel_size: self.prefill_context_parallel_size,
                num_gpu_blocks_is_explicit: self.num_gpu_blocks_is_explicit,
                rank: self.rank.clone(),
            }),
            WorkerStage::Decode => self.decode.clone().unwrap_or_else(|| ReplayRoleConfig {
                dp_size: self.dp_size,
                tensor_parallel_size: self.tensor_parallel_size,
                prefill_context_parallel_size: self.prefill_context_parallel_size,
                num_gpu_blocks_is_explicit: self.num_gpu_blocks_is_explicit,
                rank: self.rank.clone(),
            }),
        };
        role.rank.worker_type = match stage {
            WorkerStage::Aggregated => WorkerType::Aggregated,
            WorkerStage::Prefill => WorkerType::Prefill,
            WorkerStage::Decode => WorkerType::Decode,
        };
        role
    }

    fn validate_eviction_geometry(&self, rank: &EngineConfig) -> ReplayResult<()> {
        if self.kv_eviction_policy == KvEvictionPolicy::Belady && rank.sglang.mamba_cache.is_some()
        {
            return Err(ReplayError::InvalidSpec(
                "belady does not support sglang.mamba_cache checkpoint ownership".into(),
            ));
        }
        if self.kv_eviction_policy == KvEvictionPolicy::Belady && rank.prefix_match_unit.is_some() {
            // The oracle hashes complete physical pages. Fine state keys and
            // token-page aliases require a different future-demand index.
            return Err(ReplayError::InvalidSpec(
                "belady does not support prefix_match_unit; use lru for fine-grained state caching"
                    .into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_topology(&self, topology: &ReplayTopology) -> ReplayResult<()> {
        self.validate_eviction_geometry(&self.rank)?;
        if self.kv_eviction_policy == KvEvictionPolicy::Belady {
            if !matches!(topology, ReplayTopology::Aggregated { .. }) || self.dp_size != 1 {
                return Err(ReplayError::InvalidSpec(
                    "belady requires aggregated replay with dp_size=1 per worker".into(),
                ));
            }
            if self.rank.native_host_offload.is_some()
                || self.rank.g3_offload.is_some()
                || self.rank.sglang.hicache.is_some()
            {
                return Err(ReplayError::InvalidSpec(
                    "belady does not support native_host_offload, g3_offload or sglang.hicache"
                        .into(),
                ));
            }
            if !self.rank.enable_prefix_caching {
                return Err(ReplayError::InvalidSpec(
                    "belady requires enable_prefix_caching=true".into(),
                ));
            }
        }
        match topology {
            ReplayTopology::Aggregated { .. } => {
                if self.rank.g3_offload.is_some() && self.dp_size != 1 {
                    return Err(ReplayError::InvalidSpec(
                        "g3_offload supports only dp_size=1".to_string(),
                    ));
                }
                // vLLM returns a pass's outputs after launching the next one
                // only with its uniprocess executor: one GPU per engine.
                if self.rank.launch.is_some()
                    && (self.dp_size != 1
                        || self.tensor_parallel_size != 1
                        || self.prefill_context_parallel_size != 1)
                {
                    return Err(ReplayError::InvalidSpec(
                        "launch models vLLM's uniprocess executor and requires dp_size, \
                         tensor_parallel_size and prefill_context_parallel_size of 1"
                            .to_string(),
                    ));
                }
            }
            ReplayTopology::Disaggregated { .. } => {
                let [prefill_role, decode_role] =
                    [WorkerStage::Prefill, WorkerStage::Decode].map(|stage| self.role(stage));
                let (prefill, decode) = (&prefill_role.rank, &decode_role.rank);
                if prefill.g3_offload.is_some() || decode.g3_offload.is_some() {
                    return Err(ReplayError::InvalidSpec(
                        "g3_offload supports only aggregated replay".to_string(),
                    ));
                }
                // Decode resumes from the transferred state, so both roles model it.
                if prefill.state_cache.is_some() != decode.state_cache.is_some() {
                    return Err(ReplayError::InvalidSpec(
                        "state_cache must be set on both prefill and decode roles".to_string(),
                    ));
                }
                // A grouped rank counts the handoff in full-attention blocks of
                // its groups and a linear rank in tokens, so both roles need
                // the same groups. Two linear roles have none.
                if prefill.kv_cache_groups != decode.kv_cache_groups {
                    return Err(ReplayError::InvalidSpec(
                        "kv_cache_groups must be identical on the prefill and decode roles"
                            .to_string(),
                    ));
                }
                if !prefill.kv_cache_groups.is_empty() {
                    if prefill_role.prefill_context_parallel_size != 1
                        || decode_role.prefill_context_parallel_size != 1
                    {
                        return Err(ReplayError::InvalidSpec(
                            "grouped prefill and decode roles require prefill_context_parallel_size=1".into(),
                        ));
                    }
                    // The handoff moves one TP rank's cache with one vLLM rule set.
                    if prefill_role.tensor_parallel_size != decode_role.tensor_parallel_size {
                        return Err(ReplayError::InvalidSpec(
                            "grouped prefill and decode roles require equal tensor_parallel_size"
                                .into(),
                        ));
                    }
                    if prefill.vllm_version != decode.vllm_version {
                        return Err(ReplayError::InvalidSpec(
                            "vllm_version must be identical on the prefill and decode roles".into(),
                        ));
                    }
                    let identity = |role: &ReplayRoleConfig| {
                        role.grouped_transfer_identity()
                            .map_err(|e| ReplayError::InvalidSpec(format!("{e:#}")))
                    };
                    let prefill_identity = identity(&prefill_role)?;
                    let decode_identity = identity(&decode_role)?;
                    for (prefill, decode) in prefill_identity.iter().zip(&decode_identity) {
                        if let Some((field, p, d)) = prefill.mismatch(decode) {
                            return Err(ReplayError::InvalidSpec(format!(
                                "grouped prefill and decode roles require matching {field} ({p} and {d})"
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Reusable construction state for one worker role.
#[doc(hidden)]
#[derive(Clone)]
pub struct ReplayRoleFactory {
    factory: EngineFactory,
    needs_belady_oracle: bool,
    dp_size: NonZeroU32,
    tensor_parallel_size: u32,
    prefill_context_parallel_size: u32,
    /// Pipeline stages of an SGLang PP prefill worker; one elsewhere.
    pipeline_parallel_size: u32,
    backend: Backend,
    total_blocks: u64,
    kv_cache_capacity_bytes: Option<u64>,
    /// Per-rank P/D KV transfer link bandwidth in decimal GB/s, if priced.
    kv_transfer_bandwidth: Option<f64>,
    /// AFD pool GPU total (`n_a_nodes * gpus_per_node +
    /// n_f_nodes * effective_f_gpus_per_node()`),
    /// overriding the monolithic `dp * tp` accounting.
    afd_total_gpus: Option<usize>,
    pub(crate) g3_config: Option<crate::engine::G3OffloadConfig>,
    pub(crate) g3_block_bytes: usize,
    pub(crate) g3_tier: Option<crate::engine::g3_offload::SharedG3Tier>,
    /// Replay-owned KVCR timing, taken out of the rank config like `g3_config`.
    pub(crate) kvcr_config: Option<crate::engine::KvcrConfig>,
    /// The pool's KVCR registry, which its `EngineComponent` creates.
    pub(crate) kvcr: Option<crate::engine::kvcr::SharedKvcrRegistry>,
    /// Deployment pool of a cluster-shared G2 role, once bound.
    g2_registry: Option<Arc<G2Registry>>,
    shared_g2: bool,
}

impl ReplayRoleFactory {
    /// Bind a cluster-shared G2 role to its deployment pool. Other roles and
    /// already-bound factories are unchanged.
    pub(crate) fn bind_g2_registry(&mut self, registry: &Arc<G2Registry>) {
        if self.shared_g2 && self.g2_registry.is_none() {
            self.factory = self.factory.clone().with_g2_binding(G2Binding {
                registry: Arc::clone(registry),
                tensor_parallel_size: self.tensor_parallel_size,
            });
            self.g2_registry = Some(Arc::clone(registry));
        }
    }

    pub(crate) fn g2_registry(&self) -> Option<&Arc<G2Registry>> {
        self.g2_registry.as_ref()
    }

    pub(crate) fn can_have_internal_deadlines(&self) -> bool {
        self.factory.can_have_internal_deadlines()
    }

    pub(crate) fn can_hold_outputs(&self) -> bool {
        self.factory.can_hold_outputs()
    }

    pub(crate) fn reset_timing_evidence(&self) -> anyhow::Result<()> {
        self.factory.reset_timing_evidence()
    }

    pub(crate) fn with_belady_oracle(mut self, oracle: BeladyOracle) -> Self {
        self.factory = self.factory.with_belady_oracle(oracle);
        self.needs_belady_oracle = false;
        self
    }

    #[doc(hidden)]
    pub fn build(&self, worker_id: usize) -> ReplayResult<Engine> {
        if self.needs_belady_oracle {
            return Err(ReplayError::InvalidSpec(
                "belady requires a prepared input forecast; construct workers through Replayer"
                    .into(),
            ));
        }
        if self.g3_config.is_some() && self.g3_tier.is_none() {
            return Err(ReplayError::InvalidSpec(
                "g3_offload requires a Replay deployment registry".into(),
            ));
        }
        if self.kvcr_config.is_some() && self.kvcr.is_none() {
            return Err(ReplayError::InvalidSpec(
                "kvcr requires a Replay deployment registry".into(),
            ));
        }
        let worker_id = u64::try_from(worker_id).map_err(|_| {
            ReplayError::Engine(format!(
                "worker id {worker_id} exceeds the native engine range"
            ))
        })?;
        let mut engine = self
            .factory
            .build(EngineIdentity::new(worker_id), self.dp_size)
            .map_err(|error| ReplayError::Engine(format!("{error:#}")))?;
        if let Some(registry) = &self.g3_tier {
            registry
                .lock()
                .unwrap()
                .register_worker(worker_id as usize)
                .map_err(|error| ReplayError::Engine(error.to_string()))?;
            for rank in engine.ranks_mut() {
                rank.set_g3_offload(Arc::clone(registry), worker_id as usize);
            }
        }
        if let Some(registry) = &self.kvcr {
            let tensor_parallel_size =
                usize::try_from(self.tensor_parallel_size).map_err(|_| {
                    ReplayError::InvalidSpec("tensor_parallel_size overflows usize".into())
                })?;
            let mut handles = Vec::new();
            for (dp_rank, rank) in engine.ranks_mut().enumerate() {
                let key = crate::engine::KvHintSource {
                    worker_id,
                    dp_rank: u32::try_from(dp_rank).expect("dp_size fits u32"),
                };
                let g2 = rank
                    .set_kvcr(Arc::clone(registry), key, tensor_parallel_size)
                    .ok_or_else(|| {
                        ReplayError::InvalidSpec(
                            "kvcr requires vLLM ranks with host_offload".into(),
                        )
                    })?;
                handles.push(g2.map_err(|error| ReplayError::InvalidSpec(error.to_string()))?);
            }
            registry
                .lock()
                .unwrap()
                .register_worker(worker_id, handles, tensor_parallel_size)
                .map_err(|error| ReplayError::Engine(error.to_string()))?;
        }
        Ok(engine)
    }

    #[doc(hidden)]
    pub fn dp_size(&self) -> u32 {
        self.dp_size.get()
    }

    #[doc(hidden)]
    pub fn gpus_per_worker(&self) -> ReplayResult<usize> {
        if let Some(total) = self.afd_total_gpus {
            return Ok(total);
        }
        usize::try_from(self.dp_size.get())
            .ok()
            .and_then(|dp| {
                usize::try_from(self.tensor_parallel_size)
                    .ok()
                    .and_then(|tp| dp.checked_mul(tp))
            })
            .and_then(|gpus| {
                usize::try_from(self.prefill_context_parallel_size)
                    .ok()
                    .and_then(|cp| gpus.checked_mul(cp))
            })
            .and_then(|gpus| {
                usize::try_from(self.pipeline_parallel_size)
                    .ok()
                    .and_then(|pp| gpus.checked_mul(pp))
            })
            .ok_or_else(|| ReplayError::InvalidSpec("engine GPU count overflows usize".into()))
    }

    #[doc(hidden)]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    #[doc(hidden)]
    pub fn total_blocks(&self) -> u64 {
        self.total_blocks
    }

    pub(crate) fn kv_cache_capacity_bytes(&self) -> Option<u64> {
        self.kv_cache_capacity_bytes
    }

    pub(crate) fn kv_transfer_bandwidth(&self) -> Option<f64> {
        self.kv_transfer_bandwidth
    }
}

/// Resolves built-in or Runner-provided timing once, then creates role factories.
#[derive(Clone, Default)]
pub struct ReplayEngineFactory {
    timing: Option<Arc<dyn TimingModel>>,
    prefill_timing: Option<Arc<dyn TimingModel>>,
    decode_timing: Option<Arc<dyn TimingModel>>,
    /// Prices the encoder pool's forwards; resolved like a rank's model.
    encoder_timing: Option<Arc<dyn TimingModel>>,
}

impl ReplayEngineFactory {
    pub const fn new() -> Self {
        Self {
            timing: None,
            prefill_timing: None,
            decode_timing: None,
            encoder_timing: None,
        }
    }

    pub fn with_timing_model(timing: Arc<dyn TimingModel>) -> Self {
        Self {
            timing: Some(timing),
            prefill_timing: None,
            decode_timing: None,
            encoder_timing: None,
        }
    }

    pub fn with_optional_role_timing_models(
        prefill: Option<Arc<dyn TimingModel>>,
        decode: Option<Arc<dyn TimingModel>>,
    ) -> Self {
        Self {
            timing: None,
            prefill_timing: prefill,
            decode_timing: decode,
            encoder_timing: None,
        }
    }

    pub fn with_encoder_timing(mut self, timing: Arc<dyn TimingModel>) -> Self {
        self.encoder_timing = Some(timing);
        self
    }

    /// The model pricing the encoder pool; it never falls back to a rank's.
    pub(crate) fn encoder_timing(&self) -> Option<&Arc<dyn TimingModel>> {
        self.encoder_timing.as_ref()
    }

    #[doc(hidden)]
    pub fn role_factory(
        &self,
        config: &ReplayEngineConfig,
        stage: WorkerStage,
        emit_kv_events: bool,
    ) -> ReplayResult<ReplayRoleFactory> {
        let mut role = config.role(stage);
        if role.rank.afd.is_some() && role.dp_size != 1 {
            return Err(ReplayError::InvalidSpec(
                "AFD uses one logical scheduler (dp_size must be 1); configure A-replica parallelism in rank.afd".into(),
            ));
        }
        config.validate_eviction_geometry(&role.rank)?;
        if role.rank.sglang.mamba_cache.is_some()
            && (role.dp_size != 1
                || role.prefill_context_parallel_size != 1
                || pipeline_parallel_size(&role.rank.timing_model)? != 1)
        {
            return Err(ReplayError::InvalidSpec(
                "sglang.mamba_cache requires attention DP, prefill context parallelism and pipeline parallelism of one".into(),
            ));
        }
        role.rank.emit_kv_events = emit_kv_events;
        let dp_size = NonZeroU32::new(role.dp_size).ok_or_else(|| {
            ReplayError::InvalidSpec("native engine dp_size must be positive".into())
        })?;
        if role.tensor_parallel_size == 0 {
            return Err(ReplayError::InvalidSpec(
                "native tensor_parallel_size must be positive".into(),
            ));
        }
        if role.prefill_context_parallel_size == 0 {
            return Err(ReplayError::InvalidSpec(
                "native prefill_context_parallel_size must be positive".into(),
            ));
        }
        let timing = match stage {
            WorkerStage::Aggregated => self.timing.as_ref(),
            WorkerStage::Prefill => self.prefill_timing.as_ref().or(self.timing.as_ref()),
            WorkerStage::Decode => self.decode_timing.as_ref().or(self.timing.as_ref()),
        };
        let backend = role.rank.backend;
        let kv_cache_capacity_bytes = role.rank.kv_cache_capacity_bytes;
        let kv_transfer_bandwidth = role.rank.kv_transfer_bandwidth;
        let total_blocks = if role.rank.kv_cache_groups.is_empty() {
            u64::try_from(role.rank.num_gpu_blocks).map_err(|_| {
                ReplayError::InvalidSpec("engine KV block count exceeds the metrics range".into())
            })?
        } else {
            0
        };
        role.rank.validate().map_err(engine_error)?;
        let shared_g2 = role
            .rank
            .native_host_offload
            .as_ref()
            .is_some_and(|host| host.scope == G2Scope::ClusterShared);
        let afd_total_gpus = role.rank.afd.as_ref().map(|afd| {
            crate::engine::afd_total_gpus(
                &afd.a_pool,
                &afd.f_pool,
                afd.gpus_per_node,
                afd.effective_f_gpus_per_node(),
            )
        });
        let g3_config = role.rank.g3_offload.take();
        let kvcr_config = role.rank.kvcr.take();
        let pipeline_size = pipeline_parallel_size(&role.rank.timing_model)?;
        if kvcr_config.is_some() && pipeline_size > 1 {
            return Err(ReplayError::InvalidSpec(
                "kvcr does not support pipeline parallelism (pp must be 1)".into(),
            ));
        }
        // A prefill worker with pp stages keeps pp chunks in flight (SGLang's PP event loop feeds the
        // next chunk of the same prompt into stage 0 while the previous one runs on later stages), so
        // its passes retire at the per-stage rate. The timing provider prices the whole forward over
        // every stage; the prompt's last forward still crosses the later stages before its KV is handed off.
        if pipeline_size > 1 && (stage != WorkerStage::Prefill || backend != Backend::Sglang) {
            return Err(ReplayError::InvalidSpec(
                "engine replay prices pipeline parallelism only on an SGLang disaggregated prefill worker"
                    .into(),
            ));
        }
        if pipeline_size > 1 {
            let fill_passes = (pipeline_size - 1) as f64;
            if fill_passes > crate::engine::MAX_PIPELINE_FILL_PASSES {
                return Err(ReplayError::InvalidSpec(format!(
                    "pp {pipeline_size} exceeds the supported pipeline stage count"
                )));
            }
            // The stage count sets the fill; zero is the unset default.
            let supplied = role.rank.pipeline_fill_passes;
            if supplied != 0.0 && supplied != fill_passes {
                return Err(ReplayError::InvalidSpec(format!(
                    "pipeline_fill_passes {supplied} conflicts with pp {pipeline_size}; expected 0 or {fill_passes}"
                )));
            }
            role.rank.speedup_ratio = pipeline_speedup_ratio(&role.rank)?;
            role.rank.pipeline_fill_passes = fill_passes;
        }
        // A pull moves G2 bytes times the tensor-parallel size; it does not
        // model how prefill context parallelism shards the prompt's KV.
        if kvcr_config.is_some() && role.prefill_context_parallel_size > 1 {
            return Err(ReplayError::InvalidSpec(
                "kvcr does not support prefill context parallelism (pcp must be 1)".into(),
            ));
        }
        if g3_config.is_some() && dp_size.get() != 1 {
            return Err(ReplayError::InvalidSpec(
                "g3_offload supports only dp_size=1".into(),
            ));
        }
        let g3_block_bytes = role
            .rank
            .block_size
            .checked_mul(role.rank.kv_cache_bytes_per_token.unwrap_or(0))
            .ok_or_else(|| ReplayError::InvalidSpec("G3 block bytes overflow".into()))?;
        let factory = match timing {
            Some(timing) => EngineFactory::with_timing_model(role.rank, Arc::clone(timing)),
            None => EngineFactory::new(role.rank),
        }
        .map_err(engine_error)?;
        Ok(ReplayRoleFactory {
            factory,
            needs_belady_oracle: config.kv_eviction_policy == KvEvictionPolicy::Belady,
            dp_size,
            tensor_parallel_size: role.tensor_parallel_size,
            prefill_context_parallel_size: role.prefill_context_parallel_size,
            pipeline_parallel_size: u32::try_from(pipeline_size).map_err(|_| {
                ReplayError::InvalidSpec("pipeline parallel size exceeds u32".into())
            })?,
            backend,
            total_blocks,
            kv_cache_capacity_bytes,
            kv_transfer_bandwidth,
            afd_total_gpus,
            g3_config,
            g3_block_bytes,
            g3_tier: None,
            kvcr_config,
            kvcr: None,
            g2_registry: None,
            shared_g2,
        })
    }
}

pub fn run_engine_replay(spec: ReplaySpec) -> ReplayResult<ReplayReport> {
    Replayer::new(spec, ReplayEngineFactory::new())?.run()
}

pub fn run_engine_replay_with_timing(
    spec: ReplaySpec,
    timing: Arc<dyn TimingModel>,
) -> ReplayResult<ReplayReport> {
    Replayer::new(spec, ReplayEngineFactory::with_timing_model(timing))?.run()
}

pub fn run_engine_replay_with_optional_role_timing(
    spec: ReplaySpec,
    prefill: Option<Arc<dyn TimingModel>>,
    decode: Option<Arc<dyn TimingModel>>,
) -> ReplayResult<ReplayReport> {
    Replayer::new(
        spec,
        ReplayEngineFactory::with_optional_role_timing_models(prefill, decode),
    )?
    .run()
}

#[derive(Debug, Default)]
struct KvEventBatch(Vec<crate::engine::KvEvent>);

impl EngineEventBatch for KvEventBatch {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}

#[derive(Debug, Default)]
struct KvEventObservation;

impl ReplayEngineObservation for KvEventObservation {
    type Batch = KvEventBatch;

    const CAPTURE_ENGINE_KV_EVENTS: bool = true;

    fn observe_engine_events(
        _stage: WorkerStage,
        _worker_id: usize,
        _dp_rank: u32,
        events: Vec<crate::engine::KvEvent>,
    ) -> Self::Batch {
        KvEventBatch(events)
    }

    fn stored_hashes(batch: &Self::Batch) -> Vec<u64> {
        batch
            .0
            .iter()
            .filter(|event| event.tier == crate::engine::KvEventTier::Device)
            .flat_map(|event| match &event.data {
                crate::engine::KvEventData::Stored(stored) => stored.blocks.as_slice(),
                crate::engine::KvEventData::Removed { .. } => &[],
            })
            .map(|block| block.tokens_hash)
            .collect()
    }
}

/// Run the engine-neutral half of Dynamo's live/offline handoff conformance
/// fixture without importing or recompiling Replay implementation sources.
#[doc(hidden)]
pub fn run_engine_handoff_conformance(
    config: ReplayEngineConfig,
    factory: ReplayEngineFactory,
    request: DirectRequest,
) -> ReplayResult<crate::replay::NormalizedHandoffConformance> {
    let prefill_factory = factory.role_factory(&config, WorkerStage::Prefill, true)?;
    let decode_factory = factory.role_factory(&config, WorkerStage::Decode, true)?;
    let backend = prefill_factory.backend();
    if backend != decode_factory.backend() {
        return Err(ReplayError::InvalidSpec(
            "handoff conformance requires matching prefill/decode backends".into(),
        ));
    }
    let runtime_config = OfflineDisaggReplayConfig {
        prefill_factory,
        decode_factory,
        prefill_startup_time_ms: None,
        decode_startup_time_ms: None,
        num_prefill_workers: 1,
        num_decode_workers: 1,
        handoff_latency_ms: 0.0,
    };
    DisaggRuntimeImpl::<
        PoolRoundRobinPlacement<KvEventBatch>,
        KvEventObservation,
        NoReplayMetadata,
    >::new_composed(
        &runtime_config,
        AdmissionQueue::new_requests(
            std::collections::VecDeque::from([request]),
            ReplayMode::Trace,
        ),
        true,
        |_, prefill_topology, _, decode_topology| {
            Ok((
                PoolRoundRobinPlacement::new(prefill_topology),
                PoolRoundRobinPlacement::new(decode_topology),
            ))
        },
    )
    .map_err(runtime_error)?
    .run_handoff_conformance(backend)
    .map_err(runtime_error)
}

/// Pipeline-parallel size of an `aic`/`ais` timing provider (`pp`, alias
/// `pp_size`, default 1). Other providers model one stage. A regression or
/// learned identity that records `pp` as `"unknown"` takes the engine's
/// topology, which replays one stage. Any other present value that is not a
/// positive integer is rejected rather than read as 1.
fn pipeline_parallel_size(timing: &crate::engine::TimingModelConfig) -> Result<u64, ReplayError> {
    let crate::engine::TimingModelConfig::External { provider, config } = timing else {
        return Ok(1);
    };
    if !matches!(provider.as_str(), "aic" | "ais") {
        return Ok(1);
    }
    let Some(value) = timing_alias(config, "pp", "pp_size")
        .map_err(|e| ReplayError::InvalidSpec(e.to_string()))?
    else {
        return Ok(1);
    };
    let key = if config["pp"].is_null() {
        "pp_size"
    } else {
        "pp"
    };
    if value.as_str() == Some(crate::UNKNOWN_IDENTITY) {
        return Ok(1);
    }
    value.as_u64().filter(|size| *size > 0).ok_or_else(|| {
        ReplayError::InvalidSpec(format!(
            "timing provider `{key}` must be a positive integer, got {value}"
        ))
    })
}

/// The prefill speedup ratio a rank replays with: its own ratio times its pipeline stages, whose
/// passes retire at the per-stage rate. Diagnostics scale the timing evidence by the same value.
pub(crate) fn pipeline_speedup_ratio(rank: &EngineConfig) -> Result<f64, ReplayError> {
    let pipeline_size = pipeline_parallel_size(&rank.timing_model)?;
    if pipeline_size == 1 {
        return Ok(rank.speedup_ratio);
    }
    // A zero ratio means unscaled; normalize it before dividing by the stages.
    let ratio = if rank.speedup_ratio == 0.0 {
        1.0
    } else {
        rank.speedup_ratio
    };
    Ok(ratio * pipeline_size as f64)
}

fn engine_error(error: impl std::fmt::Display) -> ReplayError {
    ReplayError::Engine(error.to_string())
}

#[cfg(test)]
mod disaggregated_tests {
    use super::*;

    #[test]
    fn mamba_replay_rejects_dp_cp_pp_and_belady() {
        let rank: EngineConfig = serde_json::from_value(serde_json::json!({
            "backend": "sglang", "sglang": {"mamba_cache": {"max_states": 16}}
        }))
        .unwrap();
        let base = ReplayEngineConfig {
            rank,
            ..Default::default()
        };
        let factory = ReplayEngineFactory::new();
        assert!(
            factory
                .role_factory(&base, WorkerStage::Aggregated, false)
                .is_ok()
        );
        for dimension in ["dp", "cp", "pp", "pp_size", "belady"] {
            let mut config = base.clone();
            match dimension {
                "dp" => config.dp_size = 2,
                "cp" => config.prefill_context_parallel_size = 2,
                "belady" => config.kv_eviction_policy = KvEvictionPolicy::Belady,
                pp => {
                    config.rank.timing_model = crate::engine::TimingModelConfig::External {
                        provider: "aic".into(),
                        config: serde_json::json!({pp: 2}),
                    }
                }
            }
            let Err(error) = factory.role_factory(&config, WorkerStage::Aggregated, false) else {
                panic!("Mamba cache must reject {dimension}");
            };
            assert!(matches!(error, ReplayError::InvalidSpec(_)), "{error}");
            assert!(error.to_string().contains("sglang.mamba_cache"), "{error}");
        }
    }

    #[test]
    fn grouped_handoff_compares_only_the_known_timing_identity_of_each_role() {
        let topology: ReplayTopology = serde_json::from_value(serde_json::json!({
            "kind": "disaggregated", "prefill": {"initial_workers": 1}, "decode": {"initial_workers": 1}
        })).unwrap();
        let rank: EngineConfig = serde_json::from_value(serde_json::json!({
            "vllm_version": "0.32.0", "kv_cache_groups": [{
                "name": "full", "kind": "attention", "num_layers": 1, "block_size_tokens": 16,
                "page_size_bytes": 64, "prefix_cacheable": true, "transfer_bytes_per_block": 64
            }]
        }))
        .unwrap();
        let role = |timing_model| {
            Some(ReplayRoleConfig {
                rank: EngineConfig {
                    timing_model,
                    ..rank.clone()
                },
                ..Default::default()
            })
        };
        let aic = |config| TimingModelConfig::External {
            provider: "aic".into(),
            config,
        };
        let mut config = ReplayEngineConfig {
            prefill: role(aic(
                serde_json::json!({"model_path":"org/Model", "kv_cache_dtype":"fp8", "attention_backend":"FLASHINFER"}),
            )),
            ..Default::default()
        };
        for timing in [
            TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            },
            aic(
                serde_json::json!({"model":"org/Model", "kvcache_quant_mode":"fp8", "attention_backend":"flashinfer"}),
            ),
            aic(serde_json::json!({"model":"org/Model", "attention_backend":"auto"})),
        ] {
            config.decode = role(timing);
            config.validate_topology(&topology).unwrap();
        }
        config.decode = role(aic(serde_json::json!({"model":"org/model"})));
        assert!(
            config
                .validate_topology(&topology)
                .unwrap_err()
                .to_string()
                .contains("require matching model")
        );
        for role in [&mut config.prefill, &mut config.decode] {
            role.as_mut().unwrap().rank.kv_cache_groups.clear();
        }
        config.validate_topology(&topology).unwrap();
    }

    #[test]
    fn grouped_injected_providers_check_timing_aliases_before_selecting_a_spelling() {
        let cases = [
            ("tp", "tp_size", serde_json::json!(1), serde_json::json!(4)),
            ("pp", "pp_size", serde_json::json!(1), serde_json::json!(2)),
            (
                "attention_dp",
                "attention_dp_size",
                serde_json::json!(1),
                serde_json::json!(2),
            ),
            (
                "model",
                "model_path",
                serde_json::json!("/mnt/M"),
                serde_json::json!("/mnt/m"),
            ),
            (
                "kvcache_quant_mode",
                "kv_cache_dtype",
                serde_json::json!("fp8"),
                serde_json::json!("bf16"),
            ),
            (
                "fmha_quant_mode",
                "fmha_dtype",
                serde_json::json!("fp8"),
                serde_json::json!("bf16"),
            ),
        ];
        for provider in ["ais", "aic"] {
            for (name, alias, a, b) in &cases {
                let read = |first: &Value, second: &Value| {
                    let role = ReplayRoleConfig {
                        rank: EngineConfig {
                            timing_model: TimingModelConfig::External {
                                provider: provider.to_owned(),
                                config: serde_json::json!({ *name: first, *alias: second }),
                            },
                            ..EngineConfig::default()
                        },
                        ..ReplayRoleConfig::default()
                    };
                    role.grouped_transfer_identity().map(|_| ())
                };
                assert!(
                    read(a, b)
                        .unwrap_err()
                        .to_string()
                        .contains("conflicting timing aliases")
                );
                assert!(
                    read(b, a)
                        .unwrap_err()
                        .to_string()
                        .contains("conflicting timing aliases")
                );
                read(a, a).unwrap();
                read(a, &Value::Null).unwrap();
                read(&Value::Null, a).unwrap();
            }
        }
    }

    #[test]
    fn gpu_accounting_rejects_zero_cp_and_overflow() {
        let mut config = ReplayEngineConfig {
            prefill_context_parallel_size: 0,
            ..Default::default()
        };
        let factory = ReplayEngineFactory::new();
        assert!(
            factory
                .role_factory(&config, WorkerStage::Aggregated, false)
                .is_err()
        );
        config.prefill_context_parallel_size = u32::MAX;
        config.tensor_parallel_size = u32::MAX;
        config.dp_size = u32::MAX;
        let role = factory
            .role_factory(&config, WorkerStage::Aggregated, false)
            .unwrap();
        assert!(
            role.gpus_per_worker()
                .unwrap_err()
                .to_string()
                .contains("overflows")
        );
    }

    #[test]
    fn disaggregated_state_cache_requires_both_roles() {
        let state: EngineConfig = serde_json::from_value(serde_json::json!({
            "num_gpu_blocks": 64, "kv_cache_bytes_per_token": 16, "block_size": 16,
            "state_cache": {"bytes_per_request": 256}
        }))
        .unwrap();
        let role = |rank: EngineConfig| {
            Some(ReplayRoleConfig {
                dp_size: 1,
                tensor_parallel_size: 1,
                prefill_context_parallel_size: 1,
                num_gpu_blocks_is_explicit: None,
                rank,
            })
        };
        let topology: ReplayTopology = serde_json::from_value(serde_json::json!({
            "kind": "disaggregated",
            "prefill": {"initial_workers": 1},
            "decode": {"initial_workers": 1}
        }))
        .unwrap();
        let mut config = ReplayEngineConfig {
            prefill: role(state.clone()),
            decode: role(EngineConfig::default()),
            ..Default::default()
        };
        let error = config.validate_topology(&topology).unwrap_err();
        assert!(error.to_string().contains("both prefill and decode"));
        config.decode = role(state);
        config.validate_topology(&topology).unwrap();
    }
}

#[cfg(test)]
mod belady_tests {
    use super::*;

    #[test]
    fn fine_prefix_caching_rejects_belady_at_topology_and_role_entrypoints() {
        let rank: EngineConfig = serde_json::from_value(serde_json::json!({
            "block_size": 1536, "prefix_match_unit": 128,
            "num_gpu_blocks": 64, "kv_cache_bytes_per_token": 16,
            "state_cache": {"bytes_per_request": 24576}
        }))
        .unwrap();
        let mut config = ReplayEngineConfig {
            rank,
            kv_eviction_policy: KvEvictionPolicy::Belady,
            ..Default::default()
        };
        let error = config
            .validate_topology(&ReplayTopology::aggregated(1))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("belady does not support prefix_match_unit")
        );
        let error = match ReplayEngineFactory::new().role_factory(
            &config,
            WorkerStage::Aggregated,
            false,
        ) {
            Ok(_) => panic!("direct role construction must reject incompatible hash geometry"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("belady does not support prefix_match_unit")
        );
        config.kv_eviction_policy = KvEvictionPolicy::Lru;
        config
            .validate_topology(&ReplayTopology::aggregated(1))
            .unwrap();
        assert!(
            ReplayEngineFactory::new()
                .role_factory(&config, WorkerStage::Aggregated, false)
                .is_ok()
        );
    }

    #[test]
    fn belady_direct_role_factory_cannot_silently_build_an_lru_engine() {
        let config = ReplayEngineConfig {
            kv_eviction_policy: KvEvictionPolicy::Belady,
            ..Default::default()
        };
        let factory = ReplayEngineFactory::new()
            .role_factory(&config, WorkerStage::Aggregated, false)
            .unwrap();
        let error = match factory.build(0) {
            Ok(_) => panic!("Belady must not construct a worker without its input forecast"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("through Replayer"));
        assert!(
            factory
                .with_belady_oracle(BeladyOracle::new(Vec::new()).unwrap())
                .build(0)
                .is_ok()
        );
    }
}

#[cfg(test)]
mod kvcr_tests {
    use super::*;
    use crate::engine::kvcr::KvcrRegistry;
    use crate::engine::{KvcrConfig, NativeHostOffloadConfig};

    fn rank(kvcr: Option<KvcrConfig>) -> EngineConfig {
        EngineConfig {
            kv_cache_bytes_per_token: Some(250_000),
            native_host_offload: Some(NativeHostOffloadConfig::new(4)),
            kvcr,
            ..Default::default()
        }
    }

    fn role_factory(
        config: &ReplayEngineConfig,
        stage: WorkerStage,
    ) -> ReplayResult<ReplayRoleFactory> {
        ReplayEngineFactory::new().role_factory(config, stage, false)
    }

    #[test]
    fn role_factory_takes_kvcr_out_of_the_rank_config() {
        let config = ReplayEngineConfig {
            rank: rank(Some(KvcrConfig::default())),
            ..Default::default()
        };
        for stage in [WorkerStage::Aggregated, WorkerStage::Prefill] {
            let mut factory = role_factory(&config, stage).unwrap();
            assert_eq!(factory.kvcr_config, Some(KvcrConfig::default()));
            let Err(error) = factory.build(0) else {
                panic!("a KVCR worker builds only with its pool's registry");
            };
            assert!(
                error
                    .to_string()
                    .contains("kvcr requires a Replay deployment registry")
            );
            // Replay owns it: the rank factory no longer sees the block.
            factory.kvcr = Some(KvcrRegistry::new(KvcrConfig::default()));
            assert!(factory.build(0).is_ok());
        }

        let off = ReplayEngineConfig {
            rank: rank(None),
            ..Default::default()
        };
        let factory = role_factory(&off, WorkerStage::Aggregated).unwrap();
        assert_eq!(factory.kvcr_config, None);
    }

    #[test]
    fn kvcr_rejects_a_pipeline_parallel_timing_provider() {
        let config_with = |kvcr, pp: &str, value: serde_json::Value| {
            let mut rank = rank(kvcr);
            rank.timing_model = crate::engine::TimingModelConfig::External {
                provider: "aic".into(),
                config: serde_json::json!({"model": "m", "system": "s", pp: value}),
            };
            ReplayEngineConfig {
                rank,
                ..Default::default()
            }
        };
        let config = |kvcr, pp| config_with(kvcr, pp, serde_json::json!(2));
        for pp in ["pp", "pp_size"] {
            let Err(error) = role_factory(
                &config(Some(KvcrConfig::default()), pp),
                WorkerStage::Aggregated,
            ) else {
                panic!("kvcr with {pp}=2 must be rejected");
            };
            assert!(matches!(error, ReplayError::InvalidSpec(_)), "{error}");
            assert!(
                error
                    .to_string()
                    .contains("kvcr does not support pipeline parallelism"),
                "{error}"
            );
        }
        // A size KVCR cannot read is rejected rather than taken as one stage.
        for value in [
            serde_json::json!("2"),
            serde_json::json!(2.0),
            serde_json::json!(0),
        ] {
            let Err(error) = role_factory(
                &config_with(Some(KvcrConfig::default()), "pp", value.clone()),
                WorkerStage::Aggregated,
            ) else {
                panic!("kvcr with pp={value} must be rejected");
            };
            assert!(matches!(error, ReplayError::InvalidSpec(_)), "{error}");
            assert!(
                error
                    .to_string()
                    .contains("`pp` must be a positive integer"),
                "{error}"
            );
        }
        // A regression or learned identity that does not know its pipeline
        // width takes the engine's single stage.
        let timing = EngineConfig::default().built_in_timing_model().unwrap();
        for pp in ["pp", "pp_size"] {
            let unknown = config_with(
                Some(KvcrConfig::default()),
                pp,
                serde_json::json!("unknown"),
            );
            let factory = ReplayEngineFactory::with_timing_model(Arc::clone(&timing))
                .role_factory(&unknown, WorkerStage::Aggregated, false)
                .unwrap_or_else(|error| panic!("kvcr with {pp}=unknown: {error}"));
            assert_eq!(factory.kvcr_config, Some(KvcrConfig::default()));
        }
        // Pipeline parallelism alone is not KVCR's to reject.
        let plain = role_factory(&config(None, "pp"), WorkerStage::Aggregated);
        assert!(
            !matches!(&plain, Err(error) if error.to_string().contains("kvcr")),
            "{:?}",
            plain.err()
        );
    }

    #[test]
    fn pipeline_stages_are_priced_only_on_an_sglang_prefill_worker() {
        let config = |backend| ReplayEngineConfig {
            rank: EngineConfig {
                backend,
                timing_model: crate::engine::TimingModelConfig::External {
                    provider: "aic".into(),
                    config: serde_json::json!({"model": "m", "system": "s", "pp": 4}),
                },
                ..Default::default()
            },
            ..Default::default()
        };
        struct Unit;
        impl crate::engine::TimingModel for Unit {
            fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
                Ok(1.0)
            }
            fn predict_decode_ms(
                &self,
                _: usize,
                _: usize,
                _: usize,
                _: usize,
            ) -> anyhow::Result<f64> {
                Ok(1.0)
            }
        }
        // Its passes retire at the per-stage rate on four stages' GPUs.
        let prefill = ReplayEngineFactory::with_timing_model(Arc::new(Unit))
            .role_factory(&config(Backend::Sglang), WorkerStage::Prefill, false)
            .unwrap();
        assert_eq!(prefill.gpus_per_worker().unwrap(), 4);
        let rank = prefill.factory.config();
        assert_eq!((rank.speedup_ratio, rank.pipeline_fill_passes), (4.0, 3.0));
        // Zero means unscaled, so it is one before the stages divide it.
        let mut unscaled = config(Backend::Sglang);
        unscaled.rank.speedup_ratio = 0.0;
        let prefill = ReplayEngineFactory::with_timing_model(Arc::new(Unit))
            .role_factory(&unscaled, WorkerStage::Prefill, false)
            .unwrap();
        assert_eq!(prefill.factory.config().speedup_ratio, 4.0);
        // A supplied fill must agree with the stage count; an oversized stage count is rejected.
        let mut matching = config(Backend::Sglang);
        matching.rank.pipeline_fill_passes = 3.0;
        ReplayEngineFactory::with_timing_model(Arc::new(Unit))
            .role_factory(&matching, WorkerStage::Prefill, false)
            .unwrap();
        let mut conflicting = config(Backend::Sglang);
        conflicting.rank.pipeline_fill_passes = 1.0;
        let error = ReplayEngineFactory::with_timing_model(Arc::new(Unit))
            .role_factory(&conflicting, WorkerStage::Prefill, false)
            .err()
            .unwrap();
        assert!(error.to_string().contains("conflicts with pp 4"), "{error}");
        let mut oversized = config(Backend::Sglang);
        oversized.rank.timing_model = crate::engine::TimingModelConfig::External {
            provider: "aic".into(),
            config: serde_json::json!({"model": "m", "system": "s", "pp": 4096}),
        };
        let error = ReplayEngineFactory::with_timing_model(Arc::new(Unit))
            .role_factory(&oversized, WorkerStage::Prefill, false)
            .err()
            .unwrap();
        assert!(
            error.to_string().contains("exceeds the supported"),
            "{error}"
        );
        // Diagnostics scale the timing evidence by the ratio replay charged.
        assert_eq!(pipeline_speedup_ratio(&unscaled.rank).unwrap(), 4.0);
        let mut doubled = config(Backend::Sglang);
        doubled.rank.speedup_ratio = 2.0;
        assert_eq!(pipeline_speedup_ratio(&doubled.rank).unwrap(), 8.0);
        assert_eq!(
            pipeline_speedup_ratio(&EngineConfig::default()).unwrap(),
            EngineConfig::default().speedup_ratio
        );
        for (backend, stage) in [
            (Backend::Vllm, WorkerStage::Prefill),
            (Backend::Sglang, WorkerStage::Decode),
            (Backend::Sglang, WorkerStage::Aggregated),
        ] {
            let Err(error) = role_factory(&config(backend), stage) else {
                panic!("pp=4 on {backend:?} {stage:?} must be rejected");
            };
            assert!(
                error
                    .to_string()
                    .contains("only on an SGLang disaggregated prefill worker"),
                "{error}"
            );
        }
    }

    #[test]
    fn kvcr_rejects_prefill_context_parallelism() {
        let config = |kvcr, pcp| ReplayEngineConfig {
            rank: rank(kvcr),
            prefill_context_parallel_size: pcp,
            ..Default::default()
        };
        for stage in [WorkerStage::Aggregated, WorkerStage::Prefill] {
            let Err(error) = role_factory(&config(Some(KvcrConfig::default()), 2), stage) else {
                panic!("kvcr with pcp=2 must be rejected");
            };
            assert!(matches!(error, ReplayError::InvalidSpec(_)), "{error}");
            assert!(
                error
                    .to_string()
                    .contains("kvcr does not support prefill context parallelism"),
                "{error}"
            );
            assert!(role_factory(&config(Some(KvcrConfig::default()), 1), stage).is_ok());
        }
        // Prefill context parallelism alone is not KVCR's to reject.
        assert!(role_factory(&config(None, 2), WorkerStage::Prefill).is_ok());
    }
}

#[cfg(test)]
mod observation_tests {
    use std::sync::Mutex;

    use uuid::Uuid;

    use super::*;
    use crate::engine::{KvEventData, KvEventTier, NativeHostOffloadConfig, TimingModelConfig};
    use crate::replay::agg::AggRuntimeImpl;
    use crate::replay::core::{
        Placement, PlacementDecision, PlacementEffects, PlacementPolicy, WorkerTopology,
    };
    use crate::replay::loadgen::ReplayRequestPayload;

    /// Ingested batch: (time, router-visible device hashes, host-pinned stored
    /// hashes, total events).
    type Ingested = Arc<Mutex<Vec<(f64, Vec<u64>, Vec<u64>, usize)>>>;

    /// Places request 1 on worker 0 and holds request 2 until an ingested
    /// batch carries a host-pinned store, then places it on worker 1.
    struct PlaceAfterHostCopy {
        held: Option<Uuid>,
        ingested: Ingested,
    }

    fn placement(request_id: Uuid, scheduler_id: usize) -> Placement {
        Placement {
            request_id,
            scheduler_id,
            reported_overlap_tokens: 0,
            cache_sample: None,
            kv_hints: None,
            placement_replica_id: None,
        }
    }

    impl PlacementPolicy<ReplayRequestPayload> for PlaceAfterHostCopy {
        type Metadata = NoReplayMetadata;
        type Observation = KvEventBatch;

        fn place(
            &mut self,
            request: &ReplayRequestPayload,
            _: NoReplayMetadata,
            _: Option<String>,
            _: f64,
        ) -> anyhow::Result<PlacementEffects> {
            let request_id = request.metadata().uuid.unwrap();
            let decision = if request_id == Uuid::from_u128(1) {
                PlacementDecision::Immediate(placement(request_id, 0))
            } else {
                assert!(self.held.replace(request_id).is_none());
                PlacementDecision::Queued
            };
            Ok(PlacementEffects {
                decision,
                released: Vec::new(),
            })
        }

        fn observe(&mut self, batch: KvEventBatch, now_ms: f64) -> anyhow::Result<Vec<Placement>> {
            let host_pinned = batch
                .0
                .iter()
                .filter(|event| event.tier == KvEventTier::HostPinned)
                .flat_map(|event| match &event.data {
                    KvEventData::Stored(stored) => stored.blocks.as_slice(),
                    KvEventData::Removed { .. } => &[],
                })
                .map(|block| block.tokens_hash)
                .collect::<Vec<_>>();
            let released = if host_pinned.is_empty() {
                None
            } else {
                self.held.take()
            };
            if !batch.0.is_empty() {
                self.ingested.lock().unwrap().push((
                    now_ms,
                    KvEventObservation::stored_hashes(&batch),
                    host_pinned,
                    batch.0.len(),
                ));
            }
            Ok(released.map(|id| placement(id, 1)).into_iter().collect())
        }

        fn cancel_pending(&mut self, request_id: Uuid) -> bool {
            self.held.take_if(|held| *held == request_id).is_some()
        }

        fn request_terminal(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn prefill_completed(&mut self, _: Uuid, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn pending_count(&self) -> usize {
            usize::from(self.held.is_some())
        }

        fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }

        fn topology_settled(&mut self, _: f64) -> anyhow::Result<Vec<Placement>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn aggregated_replay_ingests_host_pinned_stores_outside_device_residency() {
        let config = ReplayEngineConfig {
            rank: EngineConfig {
                num_gpu_blocks: 4,
                block_size: 4,
                kv_cache_bytes_per_token: Some(250_000),
                native_host_offload: Some(
                    NativeHostOffloadConfig::new(4)
                        .with_bandwidths(1.0, 1.0)
                        .cluster_shared("agg-host-pinned"),
                ),
                timing_model: TimingModelConfig::Fixed {
                    prefill_ms: 1.0,
                    decode_ms: 0.0,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let factory = ReplayEngineFactory::new()
            .role_factory(&config, WorkerStage::Aggregated, true)
            .unwrap();
        let requests = [(1, 0.0), (2, 0.5)]
            .into_iter()
            .map(|(id, arrival_ms)| DirectRequest {
                tokens: vec![1, 2, 3, 4, 5],
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(id)),
                arrival_timestamp_ms: Some(arrival_ms),
                ..Default::default()
            })
            .collect();
        let ingested = Ingested::default();
        let policy = PlaceAfterHostCopy {
            held: None,
            ingested: Arc::clone(&ingested),
        };
        let report = AggRuntimeImpl::<_, KvEventObservation, NoReplayMetadata>::new_composed(
            factory,
            AdmissionQueue::new_requests(requests, ReplayMode::Trace),
            2,
            None,
            |_, _| Ok(policy),
        )
        .unwrap()
        .with_per_request_records(true)
        .run()
        .unwrap()
        .0
        .finish();

        let consumer = report
            .per_request
            .iter()
            .find(|row| row.arrival_time_ms == 0.5)
            .unwrap();
        assert_eq!(
            (
                consumer.decode_worker_idx,
                consumer.first_admission_g1_reused_input_tokens,
                consumer.first_admission_host_reused_input_tokens,
            ),
            (Some(1), Some(0), Some(4))
        );
        let ingested = ingested.lock().unwrap();
        let hash = ingested[0].1[0];
        assert_eq!(
            *ingested,
            vec![
                // Worker 0's prefill stores the prompt block in its G1.
                (1.0, vec![hash], vec![], 1),
                // The shared D2H completes: each subscribed worker ingests a
                // host-pinned store that is not router-visible residency.
                (2.0, vec![], vec![hash, hash], 2),
                // Worker 1's completed H2D makes the block resident in its G1.
                (3.0, vec![hash], vec![], 1),
            ]
        );
    }
}

#[cfg(test)]
mod afd_role_tests {
    use super::*;

    #[test]
    fn afd_pool_cannot_be_duplicated_by_outer_dp() {
        let rank: EngineConfig = serde_json::from_value(serde_json::json!({
            "backend": "sglang", "block_size": 1,
            "afd": {
                "a_pool": {"system": "h200_sxm", "a_tp_size": 1, "n_a_nodes": 1, "batch_size_per_a_replica": 18},
                "f_pool": {"system": "h200_sxm", "f_moe_ep_size": 8, "n_f_nodes": 1},
                "gpus_per_node": 8, "num_microbatches": 3, "comm_overhead_factor": 1.0
            }
        })).unwrap();
        for dp_size in [2, 8] {
            let config = ReplayEngineConfig {
                rank: rank.clone(),
                dp_size,
                ..Default::default()
            };
            for stage in [
                WorkerStage::Aggregated,
                WorkerStage::Prefill,
                WorkerStage::Decode,
            ] {
                let error = ReplayEngineFactory::new()
                    .role_factory(&config, stage, false)
                    .err()
                    .expect("outer geometry must not duplicate an AFD pool");
                assert!(error.to_string().contains("one logical scheduler"));
            }
        }
    }

    /// The afd+pd decode role carries the AFD pool-pair topology as rank
    /// metadata; its per-worker GPU accounting must report the A/F pool
    /// total instead of the monolithic dp*tp product, while the monolithic
    /// companion keeps the standard accounting.
    #[test]
    fn afd_role_factory_reports_pool_pair_gpu_total() {
        let decode_rank: EngineConfig = serde_json::from_value(serde_json::json!({
            "backend": "sglang",
            "block_size": 1,
            "num_gpu_blocks": 1024,
            "worker_type": "decode",
            "kv_transfer_bandwidth": 50.0,
            "afd": {
                "a_pool": {"system": "h200_sxm", "a_tp_size": 1, "n_a_nodes": 1, "batch_size_per_a_replica": 18},
                "f_pool": {"system": "h200_sxm", "f_moe_ep_size": 16, "n_f_nodes": 2},
                "gpus_per_node": 8,
                "num_microbatches": 3,
                "comm_overhead_factor": 1.0,
            },
        }))
        .unwrap();
        let prefill_rank: EngineConfig = serde_json::from_value(serde_json::json!({
            "backend": "sglang",
            "block_size": 1,
            "num_gpu_blocks": 1024,
            "worker_type": "prefill",
            "kv_transfer_bandwidth": 100.0,
            "kv_transfer_bytes_per_token": 16,
        }))
        .unwrap();
        let config = ReplayEngineConfig {
            rank: EngineConfig::default(),
            prefill: Some(ReplayRoleConfig {
                dp_size: 1,
                tensor_parallel_size: 1,
                num_gpu_blocks_is_explicit: None,
                prefill_context_parallel_size: 1,
                rank: prefill_rank,
            }),
            decode: Some(ReplayRoleConfig {
                dp_size: 1,
                tensor_parallel_size: 1,
                num_gpu_blocks_is_explicit: None,
                prefill_context_parallel_size: 1,
                rank: decode_rank,
            }),
            ..Default::default()
        };
        let factory = ReplayEngineFactory::new();
        let decode_factory = factory
            .role_factory(&config, WorkerStage::Decode, false)
            .unwrap();
        assert_eq!(decode_factory.gpus_per_worker().unwrap(), 24);
        assert_eq!(decode_factory.kv_transfer_bandwidth(), Some(50.0));
        let prefill_factory = factory
            .role_factory(&config, WorkerStage::Prefill, false)
            .unwrap();
        assert_eq!(prefill_factory.gpus_per_worker().unwrap(), 1);
        assert_eq!(prefill_factory.kv_transfer_bandwidth(), Some(100.0));
    }
}
