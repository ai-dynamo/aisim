// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Observable NIXL handshake fields for a selected grouped prefill or decode deployment.
//! Modified from the vLLM NIXL worker and metadata checks. Source revisions:
//! <https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6>
//! <https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be>
//! <https://github.com/vllm-project/vllm/tree/4ba787ec9959e566d3b587a796dff07673324da5>
//! See THIRD_PARTY_NOTICES.md for upstream paths and the Apache-2.0 license.

mod layout;
mod roster;

use std::collections::BTreeMap;

use layout::Layouts;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::FpmRuntimeMemoryConfig;
use crate::AicError;

const PROBE_FAMILY: &str = "aisimulate-onboarding-runtime-probe/";

fn conflict(field: &str) -> AicError {
    AicError::InvalidEngineConfig(format!(
        "grouped {field} conflicts with observed runtime settings"
    ))
}

fn known(value: &str) -> bool {
    !value.trim().is_empty() && !value.eq_ignore_ascii_case("unknown")
}

fn record(slot: &mut Option<String>, value: Option<&str>, field: &str) -> Result<(), AicError> {
    if let Some(value) = value.filter(|s| known(s)) {
        if slot.as_deref().is_some_and(|other| other != value) {
            return Err(conflict(field));
        }
        if slot.is_none() {
            *slot = Some(value.to_owned());
        }
    }
    Ok(())
}

#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub(crate) struct Worker {
    tp_rank: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dp_rank: Option<u64>,
    phase: Option<String>,
    backends: Vec<String>,
    kv_cache_layout: Option<Layouts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_cache_layout: Option<Layouts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    host_buffer: Option<bool>,
    use_mla: Option<bool>,
    permit_permute: Option<bool>,
    requires_hma: bool,
}

impl Worker {
    fn names(value: &Value) -> Vec<String> {
        let values = value
            .as_array()
            .map_or(std::slice::from_ref(value), Vec::as_slice);
        let mut names = Vec::new();
        for value in values {
            let Some(name) = value
                .as_str()
                .filter(|s| known(s) && !s.eq_ignore_ascii_case("auto"))
            else {
                return Vec::new();
            };
            names.push(name.to_ascii_lowercase());
        }
        names.sort_unstable();
        names.dedup();
        names
    }

    fn observed(
        rank: &Value,
        physical_strides: bool,
        keep_choices: bool,
        engine_layout: bool,
    ) -> Result<Self, AicError> {
        let config = &rank["resolved_config"];
        let groups = rank.pointer("/cache/groups").and_then(Value::as_array);
        // Older probes record the concrete per-layer classes. Only recognize an ordinary attention
        // stack when every attention group identifies its layers; custom or incomplete classes stay unknown.
        let mut ordinary = None;
        let mut hybrid = false;
        let mut mamba = false;
        if let Some(groups) = groups {
            for group in groups {
                mamba |= group["spec_type"] == "vllm.v1.kv_cache_interface.MambaSpec"
                    || (engine_layout && Self::contains_mamba(group));
                hybrid |= group["kind"] == "state" || group["sliding_window"].as_u64().is_some();
                if group["kind"] == "attention" {
                    let standard = group["layer_classes"].as_object().is_some_and(|classes| {
                        !classes.is_empty() && classes.values().all(|name| {
                            matches!(
                                name.as_str(),
                                Some(
                                    "vllm.model_executor.layers.attention.Attention"
                                        | "vllm.model_executor.layers.attention.attention.Attention"
                                )
                            )
                        })
                    });
                    ordinary = Some(ordinary.unwrap_or(true) && standard);
                }
            }
        }
        let permute = config
            .pointer("/kv_transfer_config/enable_permute_local_kv")
            .and_then(Value::as_bool);
        let host = config
            .pointer("/kv_transfer_config/kv_buffer_device")
            .and_then(Value::as_str)
            == Some("cpu");
        let use_mla = config
            .pointer("/model_config/use_mla")
            .and_then(Value::as_bool)
            .or_else(|| (ordinary == Some(true)).then_some(false));
        let gpu_attention = if engine_layout {
            // The stride reader checks each layer's class and excludes state views.
            use_mla == Some(false) && groups.is_some()
        } else {
            ordinary == Some(true) && !mamba && use_mla == Some(false)
        };
        let physical = if gpu_attention {
            layout::from_strides(rank, true, engine_layout)?
        } else {
            None
        };
        let named = Self::named_layout(rank, engine_layout)?;
        if let (Some(name), Some(possible)) = (&named, &physical)
            && !possible.names().contains(&name.names()[0])
        {
            return Err(conflict("kv_cache_layout"));
        }
        let layout = if named.is_some() || !physical_strides {
            named
        } else if keep_choices {
            physical
        } else if gpu_attention {
            layout::from_strides(rank, false, false)?.filter(|v| matches!(v, Layouts::Named(_)))
        } else {
            None
        };
        Ok(Self {
            tp_rank: rank["tp_rank"].as_u64(),
            dp_rank: rank["dp_rank"].as_u64(),
            phase: rank["phase"].as_str().map(str::to_owned),
            backends: Self::names(&rank["attention_backend"]),
            gpu_cache_layout: if engine_layout { layout.clone() } else { None },
            host_buffer: engine_layout.then_some(host),
            kv_cache_layout: if host && (!engine_layout || use_mla != Some(true)) {
                layout.and_then(|v| {
                    // Unknown MLA status also permits the unchanged GPU layout.
                    let permute = if engine_layout && use_mla.is_none() && permute == Some(true) {
                        None
                    } else {
                        permute
                    };
                    v.host(permute, keep_choices)
                })
            } else {
                layout
            },
            use_mla,
            permit_permute: permute,
            // Standard full attention and Mamba cannot be promoted to one cache spec when
            // HMA is disabled. Their separate initialized groups also establish this for old probes.
            requires_hma: (mamba && (engine_layout || ordinary == Some(true)))
                || (hybrid
                    && config
                        .pointer("/scheduler_config/disable_hybrid_kv_cache_manager")
                        .and_then(Value::as_bool)
                        == Some(false)),
        })
    }

    fn named_layout(rank: &Value, engine_layout: bool) -> Result<Option<Layouts>, AicError> {
        let mut named = None;
        let paths = [
            "/cache/kv_cache_layout",
            "/raw_cache_config/kv_cache_layout",
        ];
        for path in paths
            .into_iter()
            .chain(engine_layout.then_some("/resolved_config/cache_config/kv_cache_layout"))
        {
            if let Some(name) = rank
                .pointer(path)
                .and_then(Value::as_str)
                .and_then(Layouts::named)
            {
                if named.as_ref().is_some_and(|other| other != &name) {
                    return Err(conflict("kv_cache_layout"));
                }
                named = Some(name);
            }
        }
        Ok(named)
    }

    fn contains_mamba(spec: &Value) -> bool {
        match spec["spec_type"].as_str() {
            Some("vllm.v1.kv_cache_interface.MambaSpec") => true,
            Some("vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs") => spec["layer_specs"]
                .as_object()
                .is_some_and(|layers| layers.values().any(Self::contains_mamba)),
            _ => false,
        }
    }

    fn coordinate(&self, single_dp: bool) -> Option<(bool, u64, u64)> {
        let decode = match self.phase.as_deref()? {
            "prefill" => false,
            "decode" => true,
            _ => return None,
        };
        Some((
            decode,
            self.dp_rank.or(single_dp.then_some(0))?,
            self.tp_rank?,
        ))
    }

    fn combine(&mut self, other: &Self) -> Result<(), AicError> {
        if self.backends.is_empty() {
            self.backends.clone_from(&other.backends);
        } else if !other.backends.is_empty() {
            self.backends.retain(|name| other.backends.contains(name));
            if self.backends.is_empty() {
                return Err(conflict("attention_backend"));
            }
        }
        for (ours, theirs) in [
            (&mut self.kv_cache_layout, &other.kv_cache_layout),
            (&mut self.gpu_cache_layout, &other.gpu_cache_layout),
        ] {
            if let Some(theirs) = theirs {
                *ours = Some(match ours {
                    None => theirs.clone(),
                    Some(ours) => Layouts::from_names(
                        ours.names()
                            .iter()
                            .filter(|n| theirs.names().contains(n))
                            .cloned()
                            .collect(),
                    )
                    .ok_or_else(|| conflict("kv_cache_layout"))?,
                });
            }
        }
        for (field, old, new) in [
            ("kv_buffer_device", &mut self.host_buffer, other.host_buffer),
            ("use_mla", &mut self.use_mla, other.use_mla),
            (
                "enable_permute_local_kv",
                &mut self.permit_permute,
                other.permit_permute,
            ),
        ] {
            if let Some(value) = new {
                if old.is_some_and(|v| v != value) {
                    return Err(conflict(field));
                }
                *old = Some(value);
            }
        }
        self.requires_hma |= other.requires_hma;
        Ok(())
    }

    fn excludes_backend(&self, peer: &Self) -> bool {
        // The probe sorts mixed layers, losing vLLM's first registered backend. Disjoint names
        // exclude both a common first backend and the packed connector's sorted, joined name.
        !self.backends.is_empty()
            && !peer.backends.is_empty()
            && !self
                .backends
                .iter()
                .any(|name| peer.backends.contains(name))
    }

    fn accepts_layout(&self, remote: &str, local: &str, version: Option<&str>) -> bool {
        // 0.31 only tests the remote layout in its permute branch. A probe with no connector
        // does not determine a later serving deployment's permute setting.
        remote == local
            || (remote == "LBHNC"
                && (local == "LBNHC" || version == Some("0.31.0") || version.is_none())
                && self.permit_permute != Some(false)
                && !self.requires_hma)
    }

    fn excludes_layout(&self, decode: &Self, version: Option<&str>) -> bool {
        let (Some(remote), Some(local)) = (&self.kv_cache_layout, &decode.kv_cache_layout) else {
            return false;
        };
        decode.use_mla == Some(false)
            && !remote.names().iter().any(|r| {
                local
                    .names()
                    .iter()
                    .any(|l| decode.accepts_layout(r, l, version))
            })
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Capture {
    Full,
    References,
    TensorRank,
    Ranked,
    LayoutChoices,
    EngineLayout,
}

impl Capture {
    fn ranked(self) -> bool {
        matches!(
            self,
            Self::TensorRank | Self::Ranked | Self::LayoutChoices | Self::EngineLayout
        )
    }
}

#[derive(Default)]
pub(crate) struct NixlIdentity {
    pub model: Option<String>,
    pub kv_cache_dtype: Option<String>,
    pub attention_backend: Option<String>,
    pub model_dtype: Option<String>,
    pub version: Option<String>,
    pub(crate) modeled_version: Option<String>,
    pub(crate) disable_hma: Option<String>,
    pub(crate) workers: Vec<Worker>,
    pub(crate) engines: layout::Deployment,
    pub(crate) rosters: Vec<Vec<Worker>>,
    pub(crate) phase_width: Option<u64>,
    pub(crate) tp_size: Option<u64>,
}

impl NixlIdentity {
    /// Reproducing an older snapshot preserves its omitted DP coordinate and named-only layout.
    pub(crate) fn compact_workers(
        records: &[Value],
        ranked: bool,
        keep_choices: bool,
        engine_layout: bool,
    ) -> Result<Value, AicError> {
        Ok(serde_json::json!(Self::project_workers(
            records.iter(),
            ranked,
            keep_choices,
            engine_layout
        )?))
    }

    fn project_workers<'a>(
        records: impl Iterator<Item = &'a Value> + Clone,
        ranked: bool,
        keep_choices: bool,
        engine_layout: bool,
    ) -> Result<Vec<Worker>, AicError> {
        let mut schedulers = BTreeMap::new();
        if engine_layout {
            for row in records.clone().filter(|r| r["kind"] == "scheduler") {
                if let Some(layout) = Worker::named_layout(row, true)? {
                    let pool = (row["phase"].as_str(), row["dp_rank"].as_u64());
                    if schedulers
                        .insert(pool, layout.clone())
                        .is_some_and(|old| old != layout)
                    {
                        return Err(conflict("kv_cache_layout"));
                    }
                }
            }
        }
        records
            .filter(|r| r["kind"] != "scheduler")
            .map(|row| {
                let mut worker =
                    Worker::observed(row, ranked, ranked && keep_choices, engine_layout)?;
                if let Some(layout) = schedulers.get(&(row["phase"].as_str(), worker.dp_rank)) {
                    if worker
                        .gpu_cache_layout
                        .as_ref()
                        .is_some_and(|gpu| !gpu.names().contains(&layout.names()[0]))
                    {
                        return Err(conflict("kv_cache_layout"));
                    }
                    worker.gpu_cache_layout = Some(layout.clone());
                }
                if !ranked {
                    worker.dp_rank = None;
                }
                Ok(worker)
            })
            .collect()
    }

    pub(crate) fn observation_digest(
        entry: &Value,
        source: &str,
        bind_references: bool,
    ) -> Result<String, AicError> {
        roster::digest(entry, Some(source), bind_references)
    }

    fn settings(&mut self, config: &Value) -> Result<(), AicError> {
        record(
            &mut self.model,
            config
                .pointer("/model_config/model")
                .and_then(Value::as_str),
            "model",
        )?;
        record(
            &mut self.kv_cache_dtype,
            config
                .pointer("/cache_config/cache_dtype")
                .and_then(Value::as_str),
            "kv_cache_dtype",
        )?;
        let dtype = config
            .pointer("/model_config/dtype")
            .and_then(Value::as_str)
            .and_then(FpmRuntimeMemoryConfig::normalize_model_dtype);
        record(&mut self.model_dtype, dtype.as_deref(), "model_dtype")?;
        let hma = config
            .pointer("/scheduler_config/disable_hybrid_kv_cache_manager")
            .and_then(Value::as_bool);
        record(
            &mut self.disable_hma,
            hma.map(|v| if v { "true" } else { "false" }),
            "disable_hybrid_kv_cache_manager",
        )
    }

    fn memory(&mut self, data: &Value) -> Result<(), AicError> {
        let dtype = data["model_dtype"]
            .as_str()
            .and_then(FpmRuntimeMemoryConfig::normalize_model_dtype);
        record(&mut self.model_dtype, dtype.as_deref(), "model_dtype")?;
        record(
            &mut self.version,
            data["runtime_framework_version"].as_str(),
            "vllm_version",
        )
    }

    fn observation(
        &mut self,
        entry: &Value,
        v1_phase_width: Option<u64>,
        capture: Capture,
    ) -> Result<bool, AicError> {
        if capture == Capture::References
            && (entry.get("worker_observations").is_some()
                || entry.get("worker_observations_sha256").is_some()
                || entry.get("worker_attention_backends").is_some_and(|v| {
                    v.as_array().is_none_or(|names| {
                        names.iter().any(|n| {
                            !(n.is_null()
                                || n.is_string()
                                || n.as_array().is_some_and(|a| a.iter().all(Value::is_string)))
                        })
                    })
                }))
        {
            return Err(roster::invalid());
        }
        self.settings(&entry["runtime_settings"])?;
        record(
            &mut self.version,
            entry.pointer("/runtime/version").and_then(Value::as_str),
            "vllm_version",
        )?;
        if entry["source"] == "vllm_initialized_cache" {
            record(
                &mut self.version,
                entry["backend_version"].as_str(),
                "vllm_version",
            )?;
        }
        let mut complete = roster::validate_source(entry, capture, self.phase_width, self.tp_size)?
            || capture.ranked();
        let start = self.workers.len();
        if let Some(workers) = entry["worker_observations"].as_array() {
            for worker in workers {
                // New snapshots spell unavailable optional fields as null or an empty name list.
                // A missing key means a stored observation was lost.
                if [
                    "tp_rank",
                    "phase",
                    "backends",
                    "kv_cache_layout",
                    "use_mla",
                    "permit_permute",
                    "requires_hma",
                ]
                .iter()
                .any(|field| worker.get(*field).is_none())
                {
                    return Err(AicError::InvalidEngineConfig(
                        "incomplete grouped worker observation".into(),
                    ));
                }
                if capture == Capture::TensorRank && worker.get("dp_rank").is_some()
                    || capture != Capture::EngineLayout
                        && (worker.get("host_buffer").is_some()
                            || worker.get("gpu_cache_layout").is_some())
                    || capture == Capture::EngineLayout && !worker["host_buffer"].is_boolean()
                {
                    return Err(roster::invalid());
                }
                for field in ["kv_cache_layout", "gpu_cache_layout"] {
                    match &worker[field] {
                        Value::Null => {}
                        Value::String(s) if Layouts::named(s).is_some() => {}
                        Value::Array(names)
                            if matches!(
                                capture,
                                Capture::LayoutChoices | Capture::EngineLayout
                            ) && names.len() > 1
                                && names
                                    .iter()
                                    .all(|n| n.as_str().and_then(Layouts::named).is_some())
                                && names.windows(2).all(|n| n[0].as_str() < n[1].as_str()) => {}
                        _ => return Err(roster::invalid()),
                    }
                }
                self.workers
                    .push(serde_json::from_value(worker.clone()).map_err(|e| {
                        AicError::InvalidEngineConfig(format!(
                            "invalid grouped worker observation: {e}"
                        ))
                    })?);
            }
        } else if let Some(names) = entry["worker_attention_backends"].as_array() {
            // The v2 compact list has no TP coordinates. The verified importer emits the
            // workers and capacities one complete phase at a time, but sorts only the
            // capacities by rank. Its phase boundaries survive, its rank ordering does not.
            let capacities = entry["rank_capacities"].as_array().filter(|rows| {
                entry["source"] == "runtime_observations"
                    && rows.len() == names.len()
                    && rows
                        .iter()
                        .all(|r| matches!(r["phase"].as_str(), Some("prefill" | "decode")))
            });
            if let Some(rows) = capacities {
                complete = true;
                let prefill = rows.iter().filter(|row| row["phase"] == "prefill").count();
                self.workers
                    .extend(names.iter().enumerate().map(|(i, name)| Worker {
                        phase: Some(if i < prefill { "prefill" } else { "decode" }.to_owned()),
                        backends: Worker::names(name),
                        ..Default::default()
                    }));
            } else if let Some(width) = v1_phase_width.and_then(|n| usize::try_from(n).ok()) {
                if width == 0
                    || (names.len() != width && names.len().checked_sub(width) != Some(width))
                {
                    return Err(roster::invalid());
                }
                complete = true;
                // A v1 import verifies every TP × DP worker before retaining its slot, including
                // nulls. A combined import writes one complete prefill block, then one decode block.
                // The rank order inside each phase is lost.
                self.workers
                    .extend(names.iter().enumerate().map(|(i, name)| {
                        Worker {
                            phase: (names.len() > width)
                                .then(|| if i < width { "prefill" } else { "decode" }.to_owned()),
                            backends: Worker::names(name),
                            ..Default::default()
                        }
                    }));
            } else {
                // Without phase boundaries a worker may belong to another phase. Only an
                // exclusion shared by all possible workers is definitive.
                let mut possibilities = Vec::new();
                for name in names {
                    let recorded = Worker::names(name);
                    if recorded.is_empty() {
                        possibilities.clear();
                        break;
                    }
                    possibilities.extend(recorded);
                }
                possibilities.sort_unstable();
                possibilities.dedup();
                self.workers.push(Worker {
                    backends: possibilities,
                    ..Default::default()
                });
            }
        }
        if let Some(artifacts) = entry["artifacts"].as_array() {
            for artifact in artifacts {
                let rank = &artifact["evidence"];
                if rank["kind"].as_str() == Some("worker") {
                    self.settings(&rank["resolved_config"])?;
                    if entry["source"] == "vllm_initialized_cache" {
                        record(
                            &mut self.version,
                            rank["backend_version"].as_str(),
                            "vllm_version",
                        )?;
                    }
                    record(
                        &mut self.version,
                        rank.pointer("/runtime/version").and_then(Value::as_str),
                        "vllm_version",
                    )?;
                }
            }
            self.workers.extend(Self::project_workers(
                artifacts
                    .iter()
                    .map(|a| &a["evidence"])
                    .filter(|r| matches!(r["kind"].as_str(), Some("worker" | "scheduler"))),
                true,
                true,
                true,
            )?);
        }
        let count = self.workers.len() - start;
        if count > 0
            && (complete
                || (entry["source"] == "runtime_observations"
                    && entry["rank_capacities"]
                        .as_array()
                        .is_some_and(|r| r.len() == count)))
        {
            self.rosters.push(self.workers[start..].to_vec());
        }
        if entry["rank_capacities"]
            .as_array()
            .is_some_and(|ranks| ranks.len() > count)
            || (count == 0 && entry["artifacts"].is_array())
        {
            self.workers.push(Worker::default());
        }
        Ok(self.workers.len() != start)
    }

    fn collection(&mut self, entry: &Value, capture: Capture) -> Result<bool, AicError> {
        let mut found = self.observation(entry, None, capture)?;
        if let Some(phases) = entry["phases"].as_object() {
            for (phase, snapshot) in phases {
                let before = self.workers.len();
                let rosters_before = self.rosters.len();
                if !self.observation(snapshot, None, capture)? {
                    self.workers.push(Worker::default());
                }
                for rank in self.workers[before..]
                    .iter_mut()
                    .chain(self.rosters[rosters_before..].iter_mut().flatten())
                {
                    rank.phase.get_or_insert_with(|| phase.clone());
                }
                found = true;
            }
        }
        Ok(found)
    }

    fn probe(&mut self, entry: &Value) -> Result<(), AicError> {
        self.collection(entry, Capture::Full)?;
        let version = entry["schema_version"].as_str();
        let engines = version == Some("aisimulate-onboarding-runtime-probe/v5");
        let choices = engines || version == Some("aisimulate-onboarding-runtime-probe/v4");
        let bound = matches!(
            version,
            Some(
                "aisimulate-onboarding-runtime-probe/v3"
                    | "aisimulate-onboarding-runtime-probe/v4"
                    | "aisimulate-onboarding-runtime-probe/v5"
            )
        );
        if bound || version == Some("aisimulate-onboarding-runtime-probe/v2") {
            let snapshot = &entry["nixl_observation"];
            if !bound && snapshot.get("worker_observations_sha256").is_some() {
                return Err(roster::invalid());
            }
            let recorded = [
                "/runtime_settings/model_config/model",
                "/runtime_settings/model_config/dtype",
                "/runtime_settings/cache_config/cache_dtype",
                "/runtime/version",
            ];
            if recorded.iter().any(|path| {
                !snapshot
                    .pointer(path)
                    .and_then(Value::as_str)
                    .is_some_and(known)
            }) || snapshot["worker_observations"]
                .as_array()
                .is_none_or(Vec::is_empty)
            {
                return Err(AicError::InvalidEngineConfig(
                    "onboarding runtime probe is missing its NIXL observation".into(),
                ));
            }
            roster::validate(
                snapshot,
                snapshot["worker_observations"]
                    .as_array()
                    .ok_or_else(roster::invalid)?,
                self.phase_width,
                self.tp_size,
                None,
                bound,
            )?;
            if bound {
                let source = entry["source_artifacts"]
                    .as_array()
                    .and_then(|refs| {
                        refs.iter()
                            .find(|r| r["path"] == entry["observations_index"])
                    })
                    .and_then(|r| r["sha256"].as_str());
                if snapshot["worker_observations_sha256"]
                    != roster::digest(snapshot, source, false)?
                {
                    return Err(roster::invalid());
                }
            }
        }
        match version {
            Some(
                "aisimulate-onboarding-runtime-probe/v1"
                | "aisimulate-onboarding-runtime-probe/v2"
                | "aisimulate-onboarding-runtime-probe/v3"
                | "aisimulate-onboarding-runtime-probe/v4"
                | "aisimulate-onboarding-runtime-probe/v5",
            ) => {
                let memory = &entry["observed_resources"]["runtime_memory"];
                self.memory(memory)?;
                let legacy = entry["schema_version"] == "aisimulate-onboarding-runtime-probe/v1";
                let snapshot = &entry["nixl_observation"];
                if legacy && !snapshot.is_null() && !snapshot.is_object() {
                    return Err(roster::invalid());
                }
                self.observation(
                    snapshot,
                    self.phase_width.filter(|_| legacy),
                    if legacy {
                        Capture::References
                    } else if engines {
                        Capture::EngineLayout
                    } else if choices {
                        Capture::LayoutChoices
                    } else if bound {
                        Capture::Ranked
                    } else {
                        Capture::TensorRank
                    },
                )?;
                if let Some(nested) = memory["provenance"].as_str()
                    && let Ok(direct) = serde_json::from_str(nested)
                {
                    self.probe(&direct)?;
                }
            }
            Some(other) if other.starts_with(PROBE_FAMILY) => {
                return Err(AicError::InvalidEngineConfig(format!(
                    "unsupported onboarding runtime probe version: {other}"
                )));
            }
            _ => {}
        }
        Ok(())
    }

    pub fn observed(
        recorded_dtype: Option<&str>,
        provenance: &str,
        phase_width: Option<u64>,
        tp_size: Option<u64>,
    ) -> Result<Self, AicError> {
        let mut identity = Self {
            phase_width,
            tp_size,
            model_dtype: recorded_dtype.and_then(FpmRuntimeMemoryConfig::normalize_model_dtype),
            ..Self::default()
        };
        let Ok(evidence) = serde_json::from_str::<Value>(provenance) else {
            return Ok(identity);
        };
        identity.probe(&evidence)?;
        if let Some(probe) = evidence.get("runtime_probe")
            && probe["schema_version"]
                .as_str()
                .is_some_and(|s| s.starts_with(PROBE_FAMILY))
        {
            identity.probe(probe)?;
        }
        let format = evidence["observation_provenance"].as_str();
        if format.is_some_and(|s| {
            s.starts_with("source_references_v")
                && !matches!(
                    s,
                    "source_references_v2"
                        | "source_references_v3"
                        | "source_references_v4"
                        | "source_references_v5"
                        | "source_references_v6"
                )
        }) {
            return Err(AicError::InvalidEngineConfig(
                "unsupported finalized observation provenance format".into(),
            ));
        }
        let engines = format == Some("source_references_v6");
        let current = engines || format == Some("source_references_v5");
        let bound = current || format == Some("source_references_v4");
        let compact = bound || format == Some("source_references_v3");
        if format.is_some() && !bound && evidence.get("worker_observation_digests").is_some() {
            return Err(roster::invalid());
        }
        let capture = if engines {
            Capture::EngineLayout
        } else if current {
            Capture::LayoutChoices
        } else if bound {
            Capture::Ranked
        } else if compact {
            Capture::TensorRank
        } else if matches!(format, Some("source_references" | "source_references_v2")) {
            Capture::References
        } else {
            Capture::Full
        };
        let mut any_workers = false;
        let mut digests = Vec::new();
        if let Some(cells) = evidence["cell_observations"].as_array() {
            for cell in cells {
                let memory = &cell["runtime_memory"];
                identity.memory(memory)?;
                let nested = memory["provenance"]
                    .as_str()
                    .and_then(|s| serde_json::from_str(s).ok());
                let nested: Value = nested.unwrap_or_default();
                if current || (compact && nested.get("worker_observations").is_some()) {
                    let capacities = nested["rank_capacities"]
                        .as_array()
                        .ok_or_else(roster::invalid)?;
                    roster::validate(
                        &nested,
                        nested["worker_observations"]
                            .as_array()
                            .ok_or_else(roster::invalid)?,
                        phase_width,
                        tp_size,
                        Some(capacities),
                        bound,
                    )?;
                    any_workers = true;
                }
                if bound {
                    digests.push(roster::digest(
                        &nested,
                        roster::cell_source(&nested),
                        engines,
                    )?);
                }
                if nested["schema_version"]
                    .as_str()
                    .is_some_and(|s| s.starts_with(PROBE_FAMILY))
                {
                    identity.probe(&nested)?;
                } else if !identity.collection(&nested, capture)? {
                    identity.workers.push(Worker::default());
                }
            }
        }
        if (compact && !bound && !any_workers) || (bound && digests.is_empty()) {
            return Err(roster::invalid());
        }
        if bound {
            let mut expected = evidence["worker_observation_digests"]
                .as_array()
                .ok_or_else(roster::invalid)?
                .iter()
                .map(|v| v.as_str().ok_or_else(roster::invalid))
                .collect::<Result<Vec<_>, _>>()?;
            expected.sort_unstable();
            digests.sort_unstable();
            if expected != digests {
                return Err(roster::invalid());
            }
        }
        // Only repeated observations of the same measured worker constrain one another. A
        // historical record without DP coordinates can identify its worker on a DP=1 row.
        let single_dp = phase_width.is_some() && phase_width == tp_size;
        let mut by_rank = BTreeMap::new();
        for worker in &identity.workers {
            if let Some(key) = worker.coordinate(single_dp) {
                match by_rank.entry(key) {
                    std::collections::btree_map::Entry::Vacant(slot) => {
                        slot.insert(worker.clone());
                    }
                    std::collections::btree_map::Entry::Occupied(mut slot) => {
                        slot.get_mut().combine(worker)?
                    }
                }
            }
        }
        for worker in identity
            .workers
            .iter_mut()
            .chain(identity.rosters.iter_mut().flatten())
        {
            if let Some(matched) = worker.coordinate(single_dp).and_then(|k| by_rank.get(&k)) {
                worker.clone_from(matched);
            }
        }
        Ok(identity)
    }

    pub fn select(
        &mut self,
        phase: &str,
        backend: Option<&str>,
        precision: Option<&str>,
        modeled_version: Option<&str>,
        recorded_version: Option<&str>,
        prefix_caching: bool,
    ) -> Result<(), AicError> {
        if self
            .workers
            .iter()
            .any(|w| w.phase.as_deref() == Some(phase))
        {
            let selected = |w: &Worker| w.phase.as_deref().is_none_or(|p| p == phase);
            self.workers.retain(selected);
            for roster in &mut self.rosters {
                roster.retain(selected);
            }
        }
        self.rosters
            .retain(|r| !r.is_empty() && self.phase_width.is_some_and(|n| r.len() as u64 == n));
        if self.rosters.len() > 1
            && self
                .rosters
                .iter()
                .flatten()
                .any(|w| w.tp_rank.is_none() || w.dp_rank.is_none())
        {
            let slots = self.rosters[0].len();
            let tp = self.tp_size.unwrap_or(slots as u64).max(1);
            let rosters: Vec<_> = self.rosters.iter().map(Vec::as_slice).collect();
            if !roster::backends_fit(&rosters, slots, |w, i| {
                w.tp_rank.is_none_or(|t| t == i as u64 % tp)
                    && w.dp_rank.is_none_or(|d| d == i as u64 / tp)
            }) {
                return Err(conflict("attention_backend"));
            }
            if rosters.iter().flat_map(|r| r.iter()).any(|w| {
                w.kv_cache_layout.is_some()
                    || w.gpu_cache_layout.is_some()
                    || w.use_mla.is_some()
                    || w.permit_permute.is_some()
                    || w.host_buffer.is_some()
            }) && !roster::workers_fit(&rosters)
            {
                return Err(conflict("kv_cache_layout or worker settings"));
            }
        }
        self.engines = layout::Deployment::from_workers(
            &self.workers,
            &self.rosters,
            self.phase_width,
            self.tp_size,
        )?;
        self.workers.sort_unstable();
        self.workers.dedup();
        if let Some(name) = backend.filter(|s| known(s) && !s.eq_ignore_ascii_case("auto")) {
            let name = name.to_ascii_lowercase();
            if self
                .workers
                .iter()
                .any(|w| !w.backends.is_empty() && !w.backends.contains(&name))
            {
                return Err(conflict("attention_backend"));
            }
            self.attention_backend = Some(name);
        }
        self.modeled_version = modeled_version.filter(|s| known(s)).map(str::to_owned);
        record(&mut self.version, recorded_version, "vllm_version")?;
        if let (Some(actual), Some(modeled)) = (
            self.version.as_deref(),
            modeled_version.filter(|s| known(s)),
        ) && !FpmRuntimeMemoryConfig::matches_grouped_version(actual, modeled, prefix_caching)
        {
            let field = if recorded_version.is_some() {
                "runtime_memory.runtime_framework_version"
            } else {
                "observed vllm_version"
            };
            return Err(AicError::InvalidEngineConfig(format!(
                "grouped FPM cache P/D requires {field} ({actual}) to equal the deployment's backend_version ({modeled})"
            )));
        }
        if let (Some(raw), Some(label)) = (
            self.kv_cache_dtype.as_deref(),
            precision.filter(|s| known(s) && *s != "auto"),
        ) {
            let physical = if raw == "auto" {
                self.model_dtype.as_deref()
            } else {
                Some(raw)
            };
            let canonical = |s: &str| match s {
                "fp8_e4m3" | "fp8_e4m3fn" => "fp8".to_owned(),
                _ => FpmRuntimeMemoryConfig::normalize_model_dtype(s).unwrap_or_default(),
            };
            if physical.is_some_and(|value| canonical(value) != canonical(label)) {
                return Err(conflict("kv_cache_dtype"));
            }
        }
        Ok(())
    }

    pub fn mismatch(&self, decode: &Self) -> Option<(&'static str, String, String)> {
        for (field, a, b) in [
            ("model", &self.model, &decode.model),
            (
                "kv_cache_dtype",
                &self.kv_cache_dtype,
                &decode.kv_cache_dtype,
            ),
            (
                "attention_backend",
                &self.attention_backend,
                &decode.attention_backend,
            ),
            ("model_dtype", &self.model_dtype, &decode.model_dtype),
            ("vllm_version", &self.version, &decode.version),
            (
                "disable_hybrid_kv_cache_manager",
                &self.disable_hma,
                &decode.disable_hma,
            ),
        ] {
            if let (Some(a), Some(b)) = (a, b)
                && a != b
            {
                return Some((field, a.clone(), b.clone()));
            }
        }
        for (role, peer) in [(self, decode), (decode, self)] {
            if let Some(name) = &role.attention_backend
                && let Some(rank) = peer
                    .workers
                    .iter()
                    .find(|w| !w.backends.is_empty() && !w.backends.contains(name))
            {
                return Some((
                    "attention_backend",
                    name.clone(),
                    rank.backends.join(" or "),
                ));
            }
        }
        // Equal-TP transfer maps rank i to rank i. For old summaries with no coordinates,
        // a conflict still follows when an observed worker cannot match any peer, in either direction.
        fn has_no_peer(
            local: &[Worker],
            remote: &[Worker],
            excludes: impl Fn(&Worker, &Worker) -> bool,
        ) -> bool {
            local.iter().any(|worker| {
                let mut compared = false;
                for peer in remote {
                    if worker.tp_rank.is_some()
                        && peer.tp_rank.is_some()
                        && worker.tp_rank != peer.tp_rank
                    {
                        continue;
                    }
                    compared = true;
                    if !excludes(worker, peer) {
                        return false;
                    }
                }
                compared
            })
        }
        let incompatible = |test: &dyn Fn(&Worker, &Worker) -> bool| {
            self.workers.iter().any(|p| {
                decode
                    .workers
                    .iter()
                    .any(|d| p.tp_rank.is_some() && p.tp_rank == d.tp_rank && test(p, d))
            }) || has_no_peer(&self.workers, &decode.workers, test)
                || has_no_peer(&decode.workers, &self.workers, |d, p| test(p, d))
        };
        if incompatible(&Worker::excludes_backend) {
            return Some((
                "attention_backend",
                "prefill workers".into(),
                "decode workers".into(),
            ));
        }
        let version = decode
            .version
            .as_deref()
            .or(decode.modeled_version.as_deref())
            .map(FpmRuntimeMemoryConfig::modeled_release);
        if incompatible(&|p, d| p.excludes_layout(d, version)) {
            return Some((
                "kv_cache_layout",
                "prefill workers".into(),
                "decode workers".into(),
            ));
        }
        let tp = self
            .tp_size
            .filter(|t| Some(*t) == decode.tp_size && *t > 0);
        if tp.is_some() && !layout::engines_routable(&self.engines, &decode.engines, version) {
            return Some((
                "kv_cache_layout",
                "prefill engines".into(),
                "decode engines".into(),
            ));
        }
        if let Some(tp) = tp.filter(|_| !(self.rosters.is_empty() && decode.rosters.is_empty())) {
            for slot in 0..tp {
                let at_slot =
                    |w: &&Worker| w.tp_rank == Some(slot) || (tp == 1 && w.tp_rank.is_none());
                let p: Vec<_> = self.workers.iter().filter(at_slot).collect();
                let d: Vec<_> = decode.workers.iter().filter(at_slot).collect();
                // A missing prefill observation leaves one freely chosen, immutable layout.
                // A missing decode observation also leaves the MLA exemption possible.
                if (p.len() > 1 || d.len() > 1) && !layout::routable(&p, &d, version) {
                    return Some((
                        "kv_cache_layout",
                        "prefill workers".into(),
                        "decode workers".into(),
                    ));
                }
            }
            let rosters: Vec<_> = self
                .rosters
                .iter()
                .chain(&decode.rosters)
                .map(Vec::as_slice)
                .collect();
            if !roster::backends_fit(&rosters, tp as usize, |worker, slot| {
                worker.tp_rank.is_none_or(|rank| rank == slot as u64)
            }) {
                return Some((
                    "attention_backend",
                    "prefill workers".into(),
                    "decode workers".into(),
                ));
            }
        }
        for p in &self.rosters {
            for d in &decode.rosters {
                // With unequal DP populations there is no one-to-one mapping of the full worker
                // sets. Exact TP coordinates have already been checked above.
                if p.len() != d.len() || p.iter().chain(d).all(|w| w.tp_rank.is_some()) {
                    continue;
                }
                let compatible = |a: &Worker, b: &Worker| {
                    (a.tp_rank.is_none() || b.tp_rank.is_none() || a.tp_rank == b.tp_rank)
                        && !a.excludes_backend(b)
                };
                if !roster::can_pair(p, d, |a, b| {
                    compatible(a, b) && !a.excludes_layout(b, version)
                }) {
                    let field = if roster::can_pair(p, d, compatible) {
                        "kv_cache_layout"
                    } else {
                        "attention_backend"
                    };
                    return Some((field, "prefill workers".into(), "decode workers".into()));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn observe(evidence: Value, phase: &str) -> NixlIdentity {
        let mut identity = NixlIdentity::observed(None, &evidence.to_string(), None, None).unwrap();
        identity
            .select(phase, None, None, None, None, false)
            .unwrap();
        identity
    }

    fn artifacts(names: &[Value], phase: &str) -> Value {
        json!({"artifacts": names.iter().enumerate().map(|(i,n)|
            json!({"evidence":{"kind":"worker","phase":phase,"tp_rank":i,"attention_backend":n}})).collect::<Vec<_>>()})
    }

    #[test]
    fn observed_ranks_exclude_only_proven_backend_conflicts() {
        let a = json!("FLASH_ATTN");
        let b = json!("FLASHINFER");
        let n = Value::Null;
        let mixed = json!(["GDN_ATTN", "FLASHINFER"]);
        for (p, d, compatible) in [
            (
                vec![a.clone(), a.clone()],
                vec![b.clone(), n.clone()],
                false,
            ),
            (
                vec![a.clone(), b.clone()],
                vec![b.clone(), a.clone()],
                false,
            ),
            (vec![a.clone(), n.clone()], vec![n.clone(), b.clone()], true),
            (vec![a.clone(), n.clone()], vec![a.clone(), b.clone()], true),
            (vec![mixed.clone()], vec![a.clone()], false),
            (vec![mixed.clone()], vec![b.clone()], true),
            (
                vec![mixed.clone()],
                vec![json!(["FLASH_ATTN", "GDN_ATTN"])],
                true,
            ),
        ] {
            let p = observe(artifacts(&p, "prefill"), "prefill");
            let d = observe(artifacts(&d, "decode"), "decode");
            assert_eq!(p.mismatch(&d).is_none(), compatible);
            assert_eq!(d.mismatch(&p).is_none(), compatible);
        }
        let legacy = |v: Value| observe(json!({"worker_attention_backends":v}), "prefill");
        assert!(
            legacy(json!([a, b]))
                .mismatch(&legacy(json!([b, a])))
                .is_none()
        );
        assert!(
            legacy(json!([a, a]))
                .mismatch(&legacy(json!([b, null])))
                .is_none()
        );
        assert!(
            legacy(json!([a, null]))
                .mismatch(&legacy(json!([b, null])))
                .is_none()
        );
        let both_phases = json!({"source":"runtime_observations","worker_attention_backends":[a,b],
            "rank_capacities":[{"phase":"prefill","tp_rank":0},{"phase":"decode","tp_rank":0}]});
        assert!(
            observe(both_phases.clone(), "prefill")
                .mismatch(&legacy(json!([a])))
                .is_none()
        );
        assert!(
            observe(both_phases, "decode")
                .mismatch(&legacy(json!([a])))
                .is_some()
        );
        // Missing disjoint TP coordinates say nothing about their unobserved peers.
        let a0 = observe(
            json!({"artifacts":[{"evidence":{"kind":"worker","tp_rank":0,"attention_backend":a}}]}),
            "prefill",
        );
        let b1 = observe(
            json!({"artifacts":[{"evidence":{"kind":"worker","tp_rank":1,"attention_backend":b}}]}),
            "decode",
        );
        assert!(a0.mismatch(&b1).is_none());
        let phases = json!({"phases":{"prefill":artifacts(&[a.clone()],"prefill"),"decode":artifacts(&[b.clone()],"decode")}});
        assert!(
            observe(phases.clone(), "prefill")
                .mismatch(&observe(phases, "decode"))
                .is_some()
        );
        let mut identity = observe(artifacts(&[mixed, n], "prefill"), "prefill");
        assert!(
            identity
                .select("prefill", Some("FLASH_ATTN"), None, None, None, false)
                .is_err()
        );
        identity
            .select("prefill", Some("flashinfer"), None, None, None, false)
            .unwrap();
    }

    #[test]
    fn only_formal_collector_versions_are_installed_versions() {
        let mut evidence = json!({"backend_version":"0.32.0+cu130", "audited_backend_version":"0.32.0",
            "artifacts":[{"evidence":{"kind":"worker","backend_version":"0.32.0+cu130"}}]});
        assert!(observe(evidence.clone(), "prefill").version.is_none());
        evidence["source"] = json!("vllm_initialized_cache");
        let mut measured = observe(evidence.clone(), "prefill");
        assert_eq!(measured.version.as_deref(), Some("0.32.0+cu130"));
        measured
            .select("prefill", None, None, Some("0.32.0"), None, false)
            .unwrap();
        assert!(
            measured
                .select("prefill", None, None, Some("0.33.0"), None, false)
                .is_err()
        );
        evidence["artifacts"][0]["evidence"]["backend_version"] = json!("0.33.0");
        assert!(NixlIdentity::observed(None, &evidence.to_string(), None, None).is_err());
    }

    #[test]
    fn repeated_rank_observations_must_admit_the_same_backend() {
        let cell = |phase, dp, names: Value| {
            let rank = json!({"kind":"worker","phase":phase,"dp_rank":dp,"tp_rank":0,"attention_backend":names});
            json!({"runtime_memory":{"provenance":json!({"artifacts":[{"evidence":rank}]}).to_string()}})
        };
        let read = |cells: Vec<Value>| {
            NixlIdentity::observed(
                None,
                &json!({"cell_observations":cells}).to_string(),
                Some(4),
                Some(2),
            )
        };
        let first = cell("prefill", 0, json!(["a", "b"]));
        let second = cell("prefill", 0, json!(["b", "c"]));
        let narrowed = read(vec![first.clone(), second.clone()]).unwrap();
        assert!(narrowed.workers.iter().all(|w| w.backends == ["b"]));
        assert!(
            read(vec![
                first.clone(),
                second,
                cell("prefill", 0, json!(["a", "c"]))
            ])
            .is_err()
        );
        assert!(read(vec![first.clone(), cell("prefill", 1, json!("c"))]).is_ok());
        assert!(read(vec![first, cell("decode", 0, json!("c"))]).is_ok());
    }

    #[test]
    fn legacy_draft_retains_complete_phase_blocks_without_assigning_tp_ranks() {
        let draft = |names: Value, phase, width| {
            let saved = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v1",
                "nixl_observation":{"worker_attention_backends":names}});
            let mut identity =
                NixlIdentity::observed(None, &saved.to_string(), width, None).unwrap();
            identity
                .select(phase, None, None, None, None, false)
                .unwrap();
            identity
        };
        let a = json!("FLASH_ATTN");
        let b = json!("FLASHINFER");
        let three_a = draft(json!([a, a, a, b]), "prefill", Some(4));
        let three_b = draft(json!([b, a, b, b]), "decode", Some(4));
        assert!(three_a.mismatch(&three_b).is_some());
        assert!(three_b.mismatch(&three_a).is_some());
        // A manually recorded list of unrelated observations has no complete worker roster.
        let unscoped = |names: &[Value], phase| {
            let raw = names
                .iter()
                .map(|n| json!({"attention_backend":n}))
                .collect::<Vec<_>>();
            let saved = json!({"worker_observations": NixlIdentity::compact_workers(&raw, false, false, false).unwrap()});
            let mut id =
                NixlIdentity::observed(None, &saved.to_string(), Some(4), Some(4)).unwrap();
            id.select(phase, None, None, None, None, false).unwrap();
            id
        };
        assert!(
            unscoped(&[a.clone(), a.clone(), a.clone(), b.clone()], "prefill")
                .mismatch(&unscoped(
                    &[a.clone(), b.clone(), b.clone(), b.clone()],
                    "decode"
                ))
                .is_none()
        );
        for possible in [json!([b, a, a, a]), json!([b, null, a, null])] {
            assert!(
                three_a
                    .mismatch(&draft(possible, "decode", Some(4)))
                    .is_none()
            );
        }
        let uniform = draft(json!([a, a]), "prefill", Some(2));
        let two_phases = json!([a, a, b, null]);
        assert!(
            uniform
                .mismatch(&draft(two_phases.clone(), "prefill", Some(2)))
                .is_none()
        );
        assert!(
            uniform
                .mismatch(&draft(two_phases.clone(), "decode", Some(2)))
                .is_some()
        );
        let impossible = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v1",
            "nixl_observation":{"worker_attention_backends":two_phases}});
        assert!(NixlIdentity::observed(None, &impossible.to_string(), Some(3), None).is_err());
        assert!(NixlIdentity::observed(None, &impossible.to_string(), None, None).is_ok());
        for names in [json!([b, null]), json!([null, b])] {
            let partial = draft(names.clone(), "decode", Some(2));
            assert!(uniform.mismatch(&partial).is_some());
            assert!(partial.mismatch(&uniform).is_some());
            assert!(uniform.mismatch(&draft(names, "decode", None)).is_none());
        }
        assert!(
            uniform
                .mismatch(&draft(json!([null, a]), "decode", Some(2)))
                .is_none()
        );
        assert!(
            draft(json!([a, b]), "prefill", Some(2))
                .mismatch(&draft(json!([b, a]), "decode", Some(2)))
                .is_none()
        );
    }

    #[test]
    fn initialized_settings_agree_with_workers_and_selected_row() {
        let settings = json!({"model_config":{"model":"/p/M","dtype":"torch.bfloat16"},"cache_config":{"cache_dtype":"auto"}});
        let raw = json!({"runtime":{"version":"0.32.0+build"},"runtime_settings":settings,
            "artifacts":[{"evidence":{"kind":"worker","resolved_config":settings}}]});
        let mut p = observe(raw.clone(), "prefill");
        p.select(
            "prefill",
            None,
            Some("bfloat16"),
            Some("0.32.0"),
            None,
            false,
        )
        .unwrap();
        assert!(
            p.select("prefill", None, Some("fp8"), None, None, false)
                .is_err()
        );
        assert!(
            p.select("prefill", None, None, Some("0.33.0"), None, false)
                .is_err()
        );
        let mut diff = raw.clone();
        diff["runtime"]["version"] = json!("0.32.0+other");
        assert_eq!(
            p.mismatch(&observe(diff, "decode")).unwrap().0,
            "vllm_version"
        );
        for (section, key, value) in [
            ("cache_config", "cache_dtype", "bfloat16"),
            ("model_config", "model", "/d/M"),
            ("model_config", "dtype", "half"),
        ] {
            let mut corrupt = raw.clone();
            corrupt["artifacts"][0]["evidence"]["resolved_config"][section][key] = json!(value);
            assert!(NixlIdentity::observed(None, &corrupt.to_string(), None, None).is_err());
        }
    }

    #[test]
    fn imported_snapshot_requires_recorded_fields_but_accepts_sparse_workers() {
        let mut probe = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v2", "nixl_observation":{
            "runtime":{"version":"0.33.0"},"runtime_settings":{"model_config":{"model":"/M","dtype":"torch.bfloat16"},
            "cache_config":{"cache_dtype":"auto"}},"worker_observations":NixlIdentity::compact_workers(&[json!({"tp_rank":0,"phase":"prefill"})], false, false, false).unwrap()}});
        assert!(NixlIdentity::observed(None, &probe.to_string(), None, None).is_ok());
        for (parent, key) in [
            ("/nixl_observation/runtime_settings/model_config", "model"),
            ("/nixl_observation/runtime_settings/model_config", "dtype"),
            (
                "/nixl_observation/runtime_settings/cache_config",
                "cache_dtype",
            ),
            ("/nixl_observation/runtime", "version"),
            ("/nixl_observation/worker_observations/0", "backends"),
            ("/nixl_observation/worker_observations/0", "kv_cache_layout"),
        ] {
            let mut damaged = probe.clone();
            damaged
                .pointer_mut(parent)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove(key);
            assert!(
                NixlIdentity::observed(None, &damaged.to_string(), None, None).is_err(),
                "{parent}/{key}"
            );
        }
        probe.as_object_mut().unwrap().remove("nixl_observation");
        assert!(NixlIdentity::observed(None, &probe.to_string(), None, None).is_err());
        probe["schema_version"] = json!("aisimulate-onboarding-runtime-probe/v1");
        assert!(NixlIdentity::observed(None, &probe.to_string(), None, None).is_ok());
    }

    #[test]
    fn preceding_compact_formats_validate_every_phase_before_selection() {
        let raw = [
            json!({"phase":"prefill", "tp_rank":1}),
            json!({"phase":"prefill", "tp_rank":0}),
        ];
        let old = NixlIdentity::compact_workers(&raw, false, false, false).unwrap();
        let snapshot = |rows: Value| {
            json!({"schema_version":"aisimulate-onboarding-runtime-probe/v2",
            "nixl_observation":{"runtime":{"version":"0.33.0"},"runtime_settings":{"model_config":{"model":"M","dtype":"half"},
                "cache_config":{"cache_dtype":"auto"}},"worker_observations":rows}})
        };
        let capacities = json!([{"phase":"prefill","tp_rank":0,"dp_rank":0}, {"phase":"prefill","tp_rank":1,"dp_rank":0}]);
        let cell = |data: Value| json!({"runtime_memory":{"provenance":data.to_string()}});
        let finalized = |rows: Value| {
            json!({"observation_provenance":"source_references_v3", "cell_observations":[
            cell(json!({"rank_capacities":capacities})), cell(json!({"rank_capacities":capacities,"worker_observations":rows}))]})
        };
        let parse = |data: Value| NixlIdentity::observed(None, &data.to_string(), Some(2), Some(2));
        assert!(parse(snapshot(old.clone())).is_ok());
        assert!(parse(snapshot(json!([old[0], old[0]]))).is_err());
        assert!(parse(snapshot(json!([old[0]]))).is_err());
        assert!(parse(finalized(old.clone())).is_ok());
        assert!(parse(finalized(json!([old[0], old[0]]))).is_err());
        let mut foreign = old.clone();
        foreign[0]["phase"] = json!("decode");
        assert!(parse(snapshot(foreign.clone())).is_err());
        assert!(parse(finalized(foreign)).is_err());
        let mut missing = finalized(old);
        missing["cell_observations"][1] = cell(json!({"rank_capacities":capacities}));
        assert!(parse(missing).is_err());
    }

    #[test]
    fn imported_anonymous_workers_must_match_every_dp_peer_at_their_tp_rank() {
        fn imported(names: Value, tp: u64, phase: &str) -> NixlIdentity {
            let width = names.as_array().unwrap().len() as u64;
            let probe = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v1",
                "nixl_observation":{"worker_attention_backends":names}});
            let mut id =
                NixlIdentity::observed(None, &probe.to_string(), Some(width), Some(tp)).unwrap();
            id.select(phase, None, None, None, None, false).unwrap();
            id
        }
        let mut missing = NixlIdentity::observed(None, "", Some(2), Some(1)).unwrap();
        missing
            .select("prefill", None, None, None, None, false)
            .unwrap();
        let mixed = imported(json!(["a", "b"]), 1, "decode");
        assert!(missing.mismatch(&mixed).is_some());
        assert!(mixed.mismatch(&missing).is_some());
        assert!(
            missing
                .mismatch(&imported(json!(["a", null]), 1, "decode"))
                .is_none()
        );
        for (tp, p, d, legal) in [
            (1, json!(["a", "b"]), json!(["a", "b"]), false),
            (1, json!(["a", null]), json!([null, "a"]), true),
            (2, json!(["a", "a", "b", "b"]), json!(["a", "b"]), true),
            (2, json!(["a", "a", "a", "b"]), json!(["a", "b"]), false),
            (
                2,
                json!(["a", "a", "b", null]),
                json!([["b", "c"], "a"]),
                true,
            ),
        ] {
            let p = imported(p, tp, "prefill");
            let d = imported(d, tp, "decode");
            assert_eq!(p.mismatch(&d).is_none(), legal);
            assert_eq!(d.mismatch(&p).is_none(), legal);
        }
    }

    #[test]
    fn complete_anonymous_cells_must_describe_the_same_physical_workers() {
        let cell = |names: Value| {
            let captures = json!({"source":"runtime_observations","worker_attention_backends":names,
                "rank_capacities":[{"phase":"prefill","tp_rank":0,"dp_rank":0},{"phase":"prefill","tp_rank":1,"dp_rank":0}]});
            json!({"runtime_memory":{"provenance":captures.to_string()}})
        };
        let select = |cells: Vec<Value>| {
            let profile =
                json!({"observation_provenance":"source_references_v2","cell_observations":cells});
            let mut id =
                NixlIdentity::observed(None, &profile.to_string(), Some(2), Some(2)).unwrap();
            id.select("prefill", None, None, None, None, false)
        };
        assert!(select(vec![cell(json!(["a", "b"])), cell(json!(["b", "a"]))]).is_ok());
        assert!(select(vec![cell(json!(["a", null])), cell(json!(["b", null]))]).is_ok());
        assert!(select(vec![cell(json!(["a", "a"])), cell(json!(["b", "b"]))]).is_err());
        // Every two cells share a possibility. All three together still have no consistent assignment.
        assert!(
            select(vec![
                cell(json!([["a", "b"], "x"])),
                cell(json!([["b", "c"], "x"])),
                cell(json!([["a", "c"], "x"]))
            ])
            .is_err()
        );
    }

    #[test]
    fn unsupported_runtime_probe_versions_are_errors_at_a_recognized_location() {
        let future = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v99"});
        let cell =
            json!({"cell_observations":[{"runtime_memory":{"provenance":future.to_string()}}]});
        for value in [future.clone(), json!({"runtime_probe":future}), cell] {
            assert!(NixlIdentity::observed(None, &value.to_string(), None, None).is_err());
        }
        assert!(
            NixlIdentity::observed(None, r#"{"schema_version":"unrelated/v99"}"#, None, None)
                .is_ok()
        );
    }

    #[test]
    fn gpu_layout_is_common_to_one_engine_and_registration_is_local_to_a_worker() {
        let identity = |phase: &str,
                        layouts: &[(u64, u64, Option<&str>, bool)],
                        scheduler: Option<&str>| {
            let mut records = layouts.iter().map(|(dp,tp,layout,host)| json!({"kind":"worker","phase":phase,
                "dp_rank":dp,"tp_rank":tp,"raw_cache_config":{"kv_cache_layout":layout},
                "resolved_config":{"model_config":{"use_mla": false},"kv_transfer_config":{
                    "kv_buffer_device": if *host { "cpu" } else { "cuda" }, "enable_permute_local_kv":true}}})).collect::<Vec<_>>();
            records.push(json!({"kind":"scheduler","phase":phase,"dp_rank":0,"raw_cache_config":{"kv_cache_layout":scheduler}}));
            let artifacts = records
                .iter()
                .map(|r| json!({"evidence":r}))
                .collect::<Vec<_>>();
            let mut id = NixlIdentity::observed(
                None,
                &json!({"artifacts":artifacts}).to_string(),
                Some(layouts.len() as u64),
                Some(2),
            )?;
            id.select(phase, None, None, Some("0.33.0"), None, false)?;
            Ok::<_, AicError>(id)
        };
        let p = identity(
            "prefill",
            &[(0, 0, Some("BLHNC"), false), (0, 1, None, false)],
            None,
        )
        .unwrap();
        let d = identity(
            "decode",
            &[(0, 0, None, false), (0, 1, Some("BLNHC"), false)],
            None,
        )
        .unwrap();
        assert_eq!(p.mismatch(&d).unwrap().0, "kv_cache_layout");
        let none = [(0, 0, None, false), (0, 1, None, false)];
        assert!(
            p.mismatch(&identity("decode", &none, Some("BLNHC")).unwrap())
                .is_some()
        );
        assert!(
            p.mismatch(&identity("decode", &none, Some("BLHNC")).unwrap())
                .is_none()
        );
        assert!(
            identity(
                "prefill",
                &[(0, 0, Some("BLNHC"), false), (0, 1, None, false)],
                Some("BLHNC")
            )
            .is_err()
        );
        let host = [(0, 0, Some("LBNHC"), true), (0, 1, Some("LBNHC"), false)];
        let a = identity("prefill", &host, None).unwrap();
        assert!(
            a.mismatch(&identity("decode", &host, None).unwrap())
                .is_none()
        );
        assert!(
            identity(
                "prefill",
                &[
                    host[0],
                    host[1],
                    (1, 0, Some("BLHNC"), false),
                    (1, 1, None, false)
                ],
                None
            )
            .is_ok()
        );
    }

    #[test]
    fn hybrid_attention_views_and_wrapped_state_specs_constrain_the_handoff() {
        let view = json!({"shape":[10,2,16,4],"stride_bytes":[4096,128,8,2],"block_axis":0});
        let standard = "vllm.model_executor.layers.attention.Attention";
        let wrapped = json!({"kind":"state","spec_type":"vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs",
            "layer_specs":{"s":{"spec_type":"vllm.v1.kv_cache_interface.MambaSpec"}}});
        let mut row = json!({"schema_version":"aisimulate-runtime-observation/v1","runtime":{"version":"0.33.0"},
            "cache":{"groups":[{"kind":"attention","layer_classes":{"a":standard,"b":standard}},wrapped],
            "tensor_allocations":[{"layers":["a","b"],"host_resident":false,"layer_stride":1024}],
            "layer_tensors":{"a":[view],"b":[view]}}});
        let w = Worker::observed(&row, true, true, true).unwrap();
        assert_eq!(w.gpu_cache_layout, Layouts::named("BLHNC"));
        assert!(w.requires_hma);
        row["raw_cache_config"] = json!({"kv_cache_layout":"LBNHC"});
        assert!(Worker::observed(&row, true, true, true).is_err());
        row["cache"]["layer_tensors"] = json!({});
        let w = Worker::observed(&row, true, true, true).unwrap();
        assert!(!w.accepts_layout("LBHNC", "LBNHC", Some("0.33.0")));
        row["resolved_config"] = json!({"model_config":{"use_mla":true},"kv_transfer_config":{
            "kv_buffer_device":"cpu","enable_permute_local_kv":true}});
        assert_eq!(
            Worker::observed(&row, true, true, true)
                .unwrap()
                .kv_cache_layout,
            Layouts::named("LBNHC")
        );
    }

    #[test]
    fn anonymous_dp_coordinates_preserve_common_gpu_layouts_and_their_counts() {
        let archive = |phase: &str, ranks: &[&[Option<&str>]]| {
            let mut workers = Vec::new();
            let mut capacities = Vec::new();
            for (tp, engines) in ranks.iter().enumerate() {
                for (dp, name) in engines.iter().enumerate() {
                    workers.push(json!({"phase":phase,"tp_rank":tp,"dp_rank":dp,
                        "raw_cache_config":{"kv_cache_layout":name},"resolved_config":{"model_config":{"use_mla":false}}}));
                    capacities.push(json!({"phase":phase,"tp_rank":tp,"dp_rank":dp}));
                }
            }
            let projected = NixlIdentity::compact_workers(&workers, false, false, false)?;
            let cell = json!({"source":"runtime_observations","worker_observations":projected,"rank_capacities":capacities});
            let portable = json!({"observation_provenance":"source_references_v3","cell_observations":[{"runtime_memory":{"provenance":cell.to_string()}}]});
            let mut id = NixlIdentity::observed(
                None,
                &portable.to_string(),
                Some(workers.len() as u64),
                Some(ranks.len() as u64),
            )?;
            id.select(phase, None, None, Some("0.33.0"), None, false)?;
            Ok::<_, AicError>(id)
        };
        let a = Some("BLHNC");
        let b = Some("BLNHC");
        let p = archive("prefill", &[&[a, a], &[None, None]]).unwrap();
        assert!(
            p.mismatch(&archive("decode", &[&[None, None], &[b, b]]).unwrap())
                .is_some()
        );
        assert!(
            p.mismatch(&archive("decode", &[&[None, None], &[a, a]]).unwrap())
                .is_none()
        );
        assert!(archive("prefill", &[&[a, b, a], &[b, a, a]]).is_ok());
        assert!(archive("prefill", &[&[a, b, a], &[b, a, b]]).is_err());
    }

    #[test]
    fn archive_writer_controls_the_observation_schema() {
        for snapshot in [
            json!({"worker_observations":[]}),
            json!({"worker_observations_sha256":"hash"}),
            json!({"worker_attention_backends":"FLASH_ATTN"}),
            json!({"worker_attention_backends":[[null]]}),
        ] {
            let probe = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v1","nixl_observation":snapshot});
            assert!(NixlIdentity::observed(None, &probe.to_string(), Some(1), Some(1)).is_err());
        }
        let row = json!({"tp_rank":0,"dp_rank":0,"phase":"prefill"});
        let mut worker =
            NixlIdentity::compact_workers(std::slice::from_ref(&row), false, false, false).unwrap();
        worker[0]["kv_cache_layout"] = json!(["LBHNC", "LBNHC"]);
        let entry = json!({"rank_capacities":[row],"worker_observations":worker});
        let archive = |format, entry: &Value| {
            json!({"observation_provenance":format,
            "cell_observations":[{"runtime_memory":{"provenance":entry.to_string()}}]})
        };
        assert!(
            NixlIdentity::observed(
                None,
                &archive("source_references_v3", &entry).to_string(),
                Some(1),
                Some(1)
            )
            .is_err()
        );
        let mut entry = json!({"source":"runtime_observations","launch":{"topology":{"tp":1,"dp":1}},
            "launch_artifacts":[{}],"worker_attention_backends":["FLASH_ATTN"]});
        for missing in [None, Some(json!("missing"))] {
            if let Some(value) = missing {
                entry["rank_capacities"] = value;
            }
            assert!(
                NixlIdentity::observed(
                    None,
                    &archive("source_references_v2", &entry).to_string(),
                    Some(1),
                    Some(1)
                )
                .is_err()
            );
        }
    }

    #[test]
    fn ambiguous_physical_layouts_require_a_permitted_pair() {
        let worker = |layouts: &[&str]| Worker {
            kv_cache_layout: Layouts::from_names(
                layouts.iter().map(|name| name.to_string()).collect(),
            ),
            use_mla: Some(false),
            ..Default::default()
        };
        let layer_outer = worker(&["LBHNC", "LBNHC", "LHBNC"]);
        let block_outer = worker(&["BLHNC", "BLNHC", "BHLNC"]);
        for (p, d) in [(&layer_outer, &block_outer), (&block_outer, &layer_outer)] {
            assert!(p.excludes_layout(d, Some("0.32.0")));
            assert!(!p.excludes_layout(p, Some("0.32.0")));
        }
        let p = worker(&["LBHNC", "BLHNC"]);
        let mut d = worker(&["LBNHC", "BLNHC"]);
        assert!(!p.excludes_layout(&d, Some("0.32.0")));
        assert!(d.excludes_layout(&p, Some("0.32.0")));
        d.permit_permute = Some(false);
        assert!(p.excludes_layout(&d, Some("0.32.0")));
        d.use_mla = Some(true);
        assert!(!p.excludes_layout(&d, Some("0.32.0")));
        let mut captured = layer_outer.clone();
        captured.combine(&worker(&["BLNHC", "LHBNC"])).unwrap();
        assert_eq!(captured.kv_cache_layout, Layouts::named("LHBNC"));
        assert!(captured.combine(&block_outer).is_err());
    }

    #[test]
    fn one_physical_layout_has_to_serve_all_routable_dp_peers() {
        let p = Worker {
            kv_cache_layout: Layouts::from_names(
                ["BHLNC", "BLHNC", "BLNHC"].map(str::to_owned).to_vec(),
            ),
            use_mla: Some(false),
            ..Default::default()
        };
        let local = |name| Worker {
            kv_cache_layout: Layouts::named(name),
            ..p.clone()
        };
        let a = local("BLHNC");
        let b = local("BLNHC");
        let mla = Worker {
            use_mla: Some(true),
            ..b.clone()
        };
        assert!(!layout::routable(&[], &[&a, &b], Some("0.32.0")));
        assert!(layout::routable(&[&a, &b], &[], Some("0.32.0")));
        assert!(!layout::routable(&[&p, &p], &[&a, &b], Some("0.32.0")));
        assert!(!layout::routable(&[&a, &b], &[&p, &p], Some("0.32.0")));
        assert!(layout::routable(&[&p, &p], &[&a, &mla], Some("0.32.0")));
        assert!(layout::routable(&[&p, &p], &[&a, &p], Some("0.32.0")));
        let remote = local("LBHNC");
        let permuted = local("LBNHC");
        assert!(layout::routable(
            &[&remote, &permuted],
            &[&permuted, &permuted],
            Some("0.32.0")
        ));
        assert!(!layout::routable(
            &[&remote, &permuted],
            &[&remote, &permuted],
            Some("0.32.0")
        ));
    }

    #[test]
    fn physical_layout_mismatch_follows_the_decoder_transfer_direction() {
        let w = |layout: &str| Worker {
            kv_cache_layout: Layouts::named(layout),
            use_mla: Some(false),
            ..Default::default()
        };
        let (p, mut d) = (w("LBHNC"), w("LBNHC"));
        assert!(!p.excludes_layout(&d, Some("0.33.0")));
        assert!(d.excludes_layout(&p, Some("0.33.0")));
        assert!(!p.excludes_layout(&w("BLNHC"), Some("0.31.0")));
        assert!(p.excludes_layout(&w("BLNHC"), Some("0.32.0")));
        d.permit_permute = Some(false);
        assert!(p.excludes_layout(&d, Some("0.33.0")));
        d.permit_permute = Some(true);
        d.requires_hma = true;
        assert!(p.excludes_layout(&d, Some("0.33.0")));
        for mla in [Some(true), None] {
            d.use_mla = mla;
            assert!(!p.excludes_layout(&d, Some("0.33.0")));
        }
    }
}
