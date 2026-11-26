// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Observable NIXL handshake fields for a selected grouped prefill or decode deployment.
//! Modified from the vLLM NIXL worker and metadata checks. Source revisions:
//! <https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6>
//! <https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be>
//! <https://github.com/vllm-project/vllm/tree/4ba787ec9959e566d3b587a796dff07673324da5>
//! See THIRD_PARTY_NOTICES.md for upstream paths and the Apache-2.0 license.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::FpmRuntimeMemoryConfig;
use crate::AicError;

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
    phase: Option<String>,
    backends: Vec<String>,
    kv_cache_layout: Option<String>,
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

    fn observed(rank: &Value) -> Self {
        let config = &rank["resolved_config"];
        let groups = rank.pointer("/cache/groups").and_then(Value::as_array);
        // Older probes record the concrete per-layer classes. Only recognize an ordinary attention
        // stack when every attention group identifies its layers; custom or incomplete classes stay unknown.
        let mut ordinary = None;
        let mut hybrid = false;
        let mut mamba = false;
        if let Some(groups) = groups {
            for group in groups {
                mamba |= group["spec_type"] == "vllm.v1.kv_cache_interface.MambaSpec";
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
        let mut layout = rank
            .pointer("/cache/kv_cache_layout")
            .and_then(Value::as_str)
            .or_else(|| {
                rank.pointer("/raw_cache_config/kv_cache_layout")
                    .and_then(Value::as_str)
            });
        if host && layout == Some("LBNHC") {
            layout = permute.map(|enabled| if enabled { "LBHNC" } else { "LBNHC" });
        }
        Self {
            tp_rank: rank["tp_rank"].as_u64(),
            phase: rank["phase"].as_str().map(str::to_owned),
            backends: Self::names(&rank["attention_backend"]),
            kv_cache_layout: layout
                .filter(|s| {
                    matches!(
                        *s,
                        "LBHNC" | "LBNHC" | "LHBNC" | "BLHNC" | "BLNHC" | "BHLNC"
                    )
                })
                .map(str::to_owned),
            use_mla: config
                .pointer("/model_config/use_mla")
                .and_then(Value::as_bool)
                .or_else(|| (ordinary == Some(true)).then_some(false)),
            permit_permute: permute,
            // Standard full attention and Mamba cannot be promoted to one cache spec when
            // HMA is disabled. Their separate initialized groups also establish this for old probes.
            requires_hma: (mamba && ordinary == Some(true))
                || (hybrid
                    && config
                        .pointer("/scheduler_config/disable_hybrid_kv_cache_manager")
                        .and_then(Value::as_bool)
                        == Some(false)),
        }
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

    fn excludes_layout(&self, decode: &Self, version: Option<&str>) -> bool {
        let (Some(remote), Some(local)) = (&self.kv_cache_layout, &decode.kv_cache_layout) else {
            return false;
        };
        if decode.use_mla != Some(false) || remote == local {
            return false;
        }
        // 0.31 only tests the remote layout in its permute branch. The later tags test both.
        // A memory probe without a connector does not determine the serving permute setting.
        !(remote == "LBHNC"
            && (local == "LBNHC" || version == Some("0.31.0") || version.is_none())
            && decode.permit_permute != Some(false)
            && !decode.requires_hma)
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
}

impl NixlIdentity {
    pub(crate) fn compact_workers(records: &[Value]) -> Value {
        serde_json::json!(records.iter().map(Worker::observed).collect::<Vec<_>>())
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
    ) -> Result<bool, AicError> {
        self.settings(&entry["runtime_settings"])?;
        record(
            &mut self.version,
            entry.pointer("/runtime/version").and_then(Value::as_str),
            "vllm_version",
        )?;
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
                self.workers
                    .extend(names.iter().zip(rows).map(|(name, row)| Worker {
                        phase: row["phase"].as_str().map(str::to_owned),
                        backends: Worker::names(name),
                        ..Default::default()
                    }));
            } else if let Some(width) = v1_phase_width
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0 && (names.len() == n || names.len().checked_sub(n) == Some(n)))
            {
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
                    record(
                        &mut self.version,
                        rank.pointer("/runtime/version").and_then(Value::as_str),
                        "vllm_version",
                    )?;
                    self.workers.push(Worker::observed(rank));
                }
            }
        }
        let count = self.workers.len() - start;
        if entry["rank_capacities"]
            .as_array()
            .is_some_and(|ranks| ranks.len() > count)
            || (count == 0 && entry["artifacts"].is_array())
        {
            self.workers.push(Worker::default());
        }
        Ok(self.workers.len() != start)
    }

    fn collection(&mut self, entry: &Value) -> Result<bool, AicError> {
        let mut found = self.observation(entry, None)?;
        if let Some(phases) = entry["phases"].as_object() {
            for (phase, snapshot) in phases {
                let before = self.workers.len();
                if !self.observation(snapshot, None)? {
                    self.workers.push(Worker::default());
                }
                for rank in &mut self.workers[before..] {
                    rank.phase.get_or_insert_with(|| phase.clone());
                }
                found = true;
            }
        }
        Ok(found)
    }

    fn probe(&mut self, entry: &Value, phase_width: Option<u64>) -> Result<(), AicError> {
        self.collection(entry)?;
        if entry["schema_version"] == "aisimulate-onboarding-runtime-probe/v2" {
            let snapshot = &entry["nixl_observation"];
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
        }
        match entry["schema_version"].as_str() {
            Some(
                "aisimulate-onboarding-runtime-probe/v1" | "aisimulate-onboarding-runtime-probe/v2",
            ) => {
                let memory = &entry["observed_resources"]["runtime_memory"];
                self.memory(memory)?;
                let legacy = entry["schema_version"] == "aisimulate-onboarding-runtime-probe/v1";
                self.observation(&entry["nixl_observation"], phase_width.filter(|_| legacy))?;
                if let Some(nested) = memory["provenance"].as_str()
                    && let Ok(direct) = serde_json::from_str(nested)
                {
                    self.collection(&direct)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn observed(
        recorded_dtype: Option<&str>,
        provenance: &str,
        phase_width: Option<u64>,
    ) -> Result<Self, AicError> {
        let mut identity = Self {
            model_dtype: recorded_dtype.and_then(FpmRuntimeMemoryConfig::normalize_model_dtype),
            ..Self::default()
        };
        let Ok(evidence) = serde_json::from_str::<Value>(provenance) else {
            return Ok(identity);
        };
        identity.probe(&evidence, phase_width)?;
        if let Some(probe) = evidence.get("runtime_probe")
            && matches!(
                probe["schema_version"].as_str(),
                Some(
                    "aisimulate-onboarding-runtime-probe/v1"
                        | "aisimulate-onboarding-runtime-probe/v2"
                )
            )
        {
            identity.probe(probe, phase_width)?;
        }
        if let Some(cells) = evidence["cell_observations"].as_array() {
            for cell in cells {
                let memory = &cell["runtime_memory"];
                identity.memory(memory)?;
                let nested = memory["provenance"]
                    .as_str()
                    .and_then(|s| serde_json::from_str(s).ok());
                if !identity.collection(&nested.unwrap_or_default())? {
                    identity.workers.push(Worker::default());
                }
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
    ) -> Result<(), AicError> {
        if self
            .workers
            .iter()
            .any(|w| w.phase.as_deref() == Some(phase))
        {
            self.workers
                .retain(|w| w.phase.as_deref().is_none_or(|p| p == phase));
        }
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
        ) && actual.split('+').next() != Some(modeled)
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
            .and_then(|s| s.split('+').next());
        if incompatible(&|p, d| p.excludes_layout(d, version)) {
            return Some((
                "kv_cache_layout",
                "prefill workers".into(),
                "decode workers".into(),
            ));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn observe(evidence: Value, phase: &str) -> NixlIdentity {
        let mut identity = NixlIdentity::observed(None, &evidence.to_string(), None).unwrap();
        identity.select(phase, None, None, None, None).unwrap();
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
                .select("prefill", Some("FLASH_ATTN"), None, None, None)
                .is_err()
        );
        identity
            .select("prefill", Some("flashinfer"), None, None, None)
            .unwrap();
    }

    #[test]
    fn legacy_draft_retains_complete_phase_blocks_without_assigning_tp_ranks() {
        let draft = |names: Value, phase, width| {
            let saved = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v1",
                "nixl_observation":{"worker_attention_backends":names}});
            let mut identity = NixlIdentity::observed(None, &saved.to_string(), width).unwrap();
            identity.select(phase, None, None, None, None).unwrap();
            identity
        };
        let a = json!("FLASH_ATTN");
        let b = json!("FLASHINFER");
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
        assert!(
            uniform
                .mismatch(&draft(two_phases, "decode", Some(3)))
                .is_none()
        );
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
        p.select("prefill", None, Some("bfloat16"), Some("0.32.0"), None)
            .unwrap();
        assert!(p.select("prefill", None, Some("fp8"), None, None).is_err());
        assert!(
            p.select("prefill", None, None, Some("0.33.0"), None)
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
            assert!(NixlIdentity::observed(None, &corrupt.to_string(), None).is_err());
        }
    }

    #[test]
    fn imported_snapshot_requires_recorded_fields_but_accepts_sparse_workers() {
        let mut probe = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v2", "nixl_observation":{
            "runtime":{"version":"0.33.0"},"runtime_settings":{"model_config":{"model":"/M","dtype":"torch.bfloat16"},
            "cache_config":{"cache_dtype":"auto"}},"worker_observations":NixlIdentity::compact_workers(&[json!({"tp_rank":0,"phase":"prefill"})])}});
        assert!(NixlIdentity::observed(None, &probe.to_string(), None).is_ok());
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
                NixlIdentity::observed(None, &damaged.to_string(), None).is_err(),
                "{parent}/{key}"
            );
        }
        probe.as_object_mut().unwrap().remove("nixl_observation");
        assert!(NixlIdentity::observed(None, &probe.to_string(), None).is_err());
        probe["schema_version"] = json!("aisimulate-onboarding-runtime-probe/v1");
        assert!(NixlIdentity::observed(None, &probe.to_string(), None).is_ok());
    }

    #[test]
    fn physical_layout_mismatch_follows_the_decoder_transfer_direction() {
        let w = |layout: &str| Worker {
            kv_cache_layout: Some(layout.into()),
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
