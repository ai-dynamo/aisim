// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Observable NIXL handshake fields for a selected grouped prefill or decode deployment.
//! Modified from the vLLM NIXL worker and metadata checks. Source revisions:
//! <https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6>
//! <https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be>
//! <https://github.com/vllm-project/vllm/tree/4ba787ec9959e566d3b587a796dff07673324da5>
//! See THIRD_PARTY_NOTICES.md for upstream paths and the Apache-2.0 license.

mod heads;
mod layout;
mod registry;
mod roster;

use std::collections::{BTreeMap, BTreeSet};

use heads::{Decoder, HeadGeometry, UniformAttention};
use layout::Layouts;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::FpmRuntimeMemoryConfig;
use crate::AicError;

const FINALIZATION: &str = "aisimulate-onboarding-finalization/v1";
const PROBE_FAMILY: &str = "aisimulate-onboarding-runtime-probe/";
const UNIFORM_SPEC: &str = "vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs";

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

fn merge_observed<T: Copy + Eq>(
    stored: &mut Option<T>,
    value: Option<T>,
    field: &str,
) -> Result<(), AicError> {
    if let Some(value) = value {
        if stored.is_some_and(|s| s != value) {
            return Err(conflict(field));
        }
        *stored = Some(value);
    }
    Ok(())
}

fn positive(value: &Value) -> Option<u64> {
    value.as_u64().filter(|&n| n > 0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum TransferMode {
    Pull,
    Push,
}

impl TransferMode {
    fn configured(config: &Value) -> Option<Self> {
        let transfer = &config["kv_transfer_config"];
        // An explicitly configured external module overrides vLLM's connector registry.
        if !matches!(
            transfer["kv_connector_module_path"].as_str(),
            None | Some(
                "vllm.distributed.kv_transfer.kv_connector.v1.nixl"
                    | "vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector"
            )
        ) {
            return None;
        }
        match transfer["kv_connector"].as_str()? {
            "NixlConnector" | "NixlPullConnector" => Some(Self::Pull),
            "NixlPushConnector" => Some(Self::Push),
            _ => None,
        }
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub(crate) struct TransferMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    transfer_mode: Option<TransferMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_hidden_layers: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speculative: Option<bool>,
    // The model reports global KV heads. A cache spec or a view describes one layer on one TP rank.
    #[serde(default, skip_serializing_if = "HeadGeometry::is_empty")]
    model_heads: HeadGeometry,
    #[serde(skip_serializing_if = "Option::is_none")]
    uniform_attention: Option<UniformAttention>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    layer_heads: BTreeMap<String, HeadGeometry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    registered_layers: Option<BTreeSet<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nemotron_attention_layers: Option<Vec<usize>>,
    // A compact writer can retain the top-level override without the checkpoint architecture.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    hybrid_attention_positions: BTreeSet<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nemotron_state_regions: Option<layout::LayerRegions>,
}

impl TransferMetadata {
    fn configured(
        config: &Value,
        use_mla: Option<bool>,
        architecture: Option<&str>,
        source: &Value,
    ) -> Self {
        let model = &config["model_config"];
        let resolved = &model["model_arch_config"];
        let checkpoint = heads::checkpoint(model, source);
        let top = heads::HfConfig::observed(model, checkpoint.as_ref());
        let names = resolved
            .get("architectures")
            .or_else(|| top.fields.get("architectures"));
        let release = source
            .pointer("/runtime/version")
            .and_then(Value::as_str)
            .or_else(|| {
                (source["source"] == "vllm_initialized_cache")
                    .then(|| source["backend_version"].as_str())
                    .flatten()
            });
        let model_impl = model["model_impl"].as_str().unwrap_or("auto");
        let native = matches!(model_impl, "auto" | "vllm");
        let registered = names
            .filter(|_| native)
            .and_then(Value::as_array)
            .and_then(|names| {
                registry::select(
                    names.iter().map(Value::as_str),
                    release,
                    model_impl == "vllm",
                    top.has_stock_model_lookup(),
                )
            });
        let decoder = if native && names.is_none_or(Value::is_null) {
            Decoder::inferred(top.fields, architecture)
        } else {
            registered.and_then(registry::Registration::decoder)
        };
        let text = top.text_config(
            model,
            source["launch"]["model_config"].is_object(),
            registered,
        );
        let family = resolved["model_type"]
            .as_str()
            .or_else(|| top.fields["model_type"].as_str())
            .or_else(|| decoder.map(Decoder::family));
        let (configured_heads, configured_layers) =
            text.model_geometry(family, use_mla, resolved, top);
        let mut facts = Self {
            transfer_mode: TransferMode::configured(config),
            num_hidden_layers: resolved["total_num_hidden_layers"]
                .as_u64()
                .or(configured_layers)
                .or_else(|| {
                    let pattern = heads::hybrid_pattern(top.fields)?;
                    (family.is_none()
                        && architecture.is_none()
                        && top.fields["num_hidden_layers"].as_u64() == Some(pattern.len() as u64))
                    .then_some(pattern.len() as u64)
                }),
            speculative: config["speculative_config"].is_object().then_some(true),
            model_heads: HeadGeometry {
                num_kv_heads: resolved["total_num_kv_heads"]
                    .as_u64()
                    .or(configured_heads.num_kv_heads),
                head_size: resolved["head_size"]
                    .as_u64()
                    .or(configured_heads.head_size),
                ..Default::default()
            },
            ..Self::default()
        };
        if use_mla != Some(true) {
            let ordered = names.is_some_and(Value::is_array);
            facts.uniform_attention = decoder.and_then(|d| d.physical_heads(top, text, ordered));
        }
        // NemotronHModel constructs its layers from the top-level HF pattern. A nested text
        // config and the separate hidden-layer count do not select its attention mixers.
        let top = top.fields;
        let top_family = top["model_type"].as_str();
        if native
            && top_family.is_none_or(|name| name == "nemotron_h")
            && (decoder == Some(Decoder::NemotronH) || (names.is_none() && decoder.is_none()))
            && let Some(pattern) = heads::hybrid_pattern(top)
        {
            let positions = pattern
                .match_indices('*')
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            facts.nemotron_state_regions = Some(layout::LayerRegions::new(
                pattern.match_indices('M').map(|(i, _)| i).collect(),
                positions.first().copied(),
            ));
            if top_family == Some("nemotron_h") || names.is_some() {
                facts.nemotron_attention_layers = Some(positions);
            } else {
                facts.hybrid_attention_positions.insert(positions);
            }
        }
        facts
    }

    fn merge(&mut self, other: &Self) -> Result<(), AicError> {
        merge_observed(
            &mut self.transfer_mode,
            other.transfer_mode,
            "transfer_mode",
        )?;
        merge_observed(
            &mut self.num_hidden_layers,
            other.num_hidden_layers,
            "num_hidden_layers",
        )?;
        merge_observed(
            &mut self.speculative,
            other.speculative,
            "speculative_config",
        )?;
        self.model_heads.merge(&other.model_heads)?;
        if let Some(attention) = &other.uniform_attention {
            let saved = self
                .uniform_attention
                .get_or_insert_with(|| attention.clone());
            record(
                &mut saved.architecture,
                attention.architecture.as_deref(),
                "attention architecture",
            )?;
            merge_observed(
                &mut saved.divided_head_size,
                attention.divided_head_size,
                "head_size",
            )?;
            saved.heads.merge(&attention.heads)?;
            saved.ordered_architecture |= attention.ordered_architecture;
        }
        if let Some(names) = &other.registered_layers {
            if self.registered_layers.get_or_insert_with(|| names.clone()) != names {
                return Err(conflict("cache region names"));
            }
        }
        self.hybrid_attention_positions
            .extend(other.hybrid_attention_positions.iter().cloned());
        if let Some(other) = &other.nemotron_state_regions {
            self.nemotron_state_regions
                .get_or_insert_with(|| other.clone())
                .merge(other);
        }
        if let Some(layers) = &other.nemotron_attention_layers
            && self
                .nemotron_attention_layers
                .get_or_insert_with(|| layers.clone())
                != layers
        {
            return Err(conflict("attention layer positions"));
        }
        for (name, heads) in &other.layer_heads {
            self.layer_heads
                .entry(name.clone())
                .or_default()
                .merge(heads)?;
        }
        Ok(())
    }
}

#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub(crate) struct Worker {
    #[serde(flatten)]
    metadata: TransferMetadata,
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
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    single_head_mla: bool,
    permit_permute: Option<bool>,
    requires_hma: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    requires_gpu_buffer: bool,
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

    fn observed(rank: &Value, architecture: Option<&str>) -> Result<Self, AicError> {
        let config = &rank["resolved_config"];
        let groups = rank.pointer("/cache/groups").and_then(Value::as_array);
        // Only an entirely identified ordinary attention stack establishes non-MLA operation.
        let mut ordinary = None;
        let mut hybrid = false;
        let mut mamba = false;
        if let Some(groups) = groups {
            for group in groups {
                layout::check_layer_specs(group)?;
                mamba |= Self::contains_spec(group, "vllm.v1.kv_cache_interface.MambaSpec");
                hybrid |= group["kind"] == "state" || group["sliding_window"].as_u64().is_some();
                if group["kind"] == "attention" {
                    let mut layers = layout::decoder_layers(group).peekable();
                    let standard = layers.peek().is_some() && layers.all(|(_, ordinary)| ordinary);
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
        let raw_groups = rank["raw_cache_config"]["groups"].as_array();
        let mla_groups = groups.is_some_and(|groups| {
            !groups.is_empty()
                && groups
                    .iter()
                    .all(|g| g["kind"] == "attention" && layout::single_head_mla(g))
        }) && raw_groups.is_none_or(|raw| {
            !raw.is_empty()
                && groups.is_some_and(|normalized| {
                    raw.len() == normalized.len()
                        && raw.iter().all(|g| layout::raw_mla_group(g, normalized))
                })
        });
        let use_mla = config
            .pointer("/model_config/use_mla")
            .and_then(Value::as_bool)
            .or_else(|| (ordinary == Some(true)).then_some(false))
            .or_else(|| {
                (mla_groups
                    && groups.is_some_and(|groups| {
                        groups.iter().all(layout::records_one_head)
                            || raw_groups.is_some_and(|raw| {
                                raw.len() == groups.len()
                                    && raw.iter().all(layout::records_one_head)
                            })
                    }))
                .then_some(true)
            });
        let single_head_mla = mla_groups && use_mla == Some(true);
        let mut metadata = TransferMetadata::configured(config, use_mla, architecture, rank);
        metadata.registered_layers = layout::registered_layers(rank);
        if let Some(regions) = groups.and_then(|g| {
            layout::nemotron_state_regions(rank, g, host, metadata.registered_layers.as_ref())
        }) {
            metadata
                .nemotron_state_regions
                .get_or_insert_with(|| regions.clone())
                .merge(&regions);
        }

        let physical = if groups.is_some() {
            layout::from_strides(rank, &mut metadata.layer_heads)?
        } else {
            None
        };
        let named = Self::named_layout(rank)?;
        if let (Some(name), Some(possible)) = (&named, &physical)
            && !possible.names().contains(&name.names()[0])
        {
            return Err(conflict("kv_cache_layout"));
        }
        let layout = named.or(physical);
        Ok(Self {
            metadata,
            tp_rank: rank["tp_rank"].as_u64(),
            dp_rank: rank["dp_rank"].as_u64(),
            phase: rank["phase"].as_str().map(str::to_owned),
            backends: Self::names(&rank["attention_backend"]),
            gpu_cache_layout: layout.clone(),
            host_buffer: Some(host),
            kv_cache_layout: if host && use_mla != Some(true) {
                layout.and_then(|v| {
                    // Unknown MLA status also permits the unchanged GPU layout.
                    let permute = if use_mla.is_none() && permute == Some(true) {
                        None
                    } else {
                        permute
                    };
                    v.host(permute)
                })
            } else {
                layout
            },
            use_mla,
            single_head_mla,
            permit_permute: permute,
            requires_gpu_buffer: groups.is_some_and(|groups| {
                groups.iter().any(|g| {
                    Self::contains_spec(g, "vllm.v1.kv_cache_interface.CircularBufferSpec")
                })
            }),
            // Mamba requires HMA. A runtime that initialized hybrid groups with HMA enabled also
            // establishes the requirement for that capture.
            requires_hma: mamba
                || (hybrid
                    && config
                        .pointer("/scheduler_config/disable_hybrid_kv_cache_manager")
                        .and_then(Value::as_bool)
                        == Some(false)),
        })
    }

    fn named_layout(rank: &Value) -> Result<Option<Layouts>, AicError> {
        let mut named = None;
        for path in [
            "/cache/kv_cache_layout",
            "/raw_cache_config/kv_cache_layout",
            "/resolved_config/cache_config/kv_cache_layout",
        ] {
            if let Some(value) = rank.pointer(path).filter(|v| !v.is_null()) {
                let name = match value.as_str() {
                    Some("NHD") => Layouts::named("LBNHC"),
                    Some("HND") => Layouts::named("LBHNC"),
                    Some(name) => Layouts::named(name),
                    None => None,
                }
                .ok_or_else(|| conflict("kv_cache_layout"))?;
                if named.as_ref().is_some_and(|other| other != &name) {
                    return Err(conflict("kv_cache_layout"));
                }
                named = Some(name);
            }
        }
        Ok(named)
    }

    fn contains_spec(spec: &Value, wanted: &str) -> bool {
        match spec["spec_type"].as_str() {
            Some(name) if name == wanted => true,
            Some(UNIFORM_SPEC) => spec["layer_specs"]
                .as_object()
                .is_some_and(|layers| layers.values().any(|s| Self::contains_spec(s, wanted))),
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
        self.metadata.merge(&other.metadata)?;
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
        self.requires_gpu_buffer |= other.requires_gpu_buffer;
        self.single_head_mla |= other.single_head_mla;
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

    fn accepts_layout(
        &self,
        remote: &str,
        local: &str,
        version: Option<&str>,
        no_repack: bool,
    ) -> bool {
        // 0.31 only tests the remote layout in its permute branch. A probe with no connector
        // does not determine a later serving deployment's permute setting.
        remote == local
            || (remote == "LBHNC"
                && (local == "LBNHC" || version == Some("0.31.0") || version.is_none())
                && self.permit_permute != Some(false)
                && !self.requires_hma
                && !no_repack)
    }

    fn excludes_layout(&self, decode: &Self, version: Option<&str>, no_repack: bool) -> bool {
        let (Some(remote), Some(local)) = (&self.kv_cache_layout, &decode.kv_cache_layout) else {
            return false;
        };
        decode.use_mla == Some(false)
            && !remote.names().iter().any(|r| {
                local
                    .names()
                    .iter()
                    .any(|l| decode.accepts_layout(r, l, version, no_repack))
            })
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Capture {
    Full,
    References,
    BoundReferences,
}

#[derive(Default)]
pub(crate) struct NixlIdentity {
    pub(crate) metadata: TransferMetadata,
    pub model: Option<String>,
    pub architecture: Option<String>,
    pub kv_cache_dtype: Option<String>,
    pub attention_backend: Option<String>,
    pub model_dtype: Option<String>,
    pub version: Option<String>,
    pub(crate) modeled_version: Option<String>,
    pub(crate) disable_hma: Option<String>,
    pub(crate) measured_mamba: bool,
    pub(crate) recorded_groups: Option<Vec<Value>>,
    pub(crate) recorded_memory: Value,
    pub(crate) hybrid_groups: bool,
    pub(crate) workers: Vec<Worker>,
    pub(crate) engines: layout::Deployment,
    pub(crate) rosters: Vec<Vec<Worker>>,
    pub(crate) phase_width: Option<u64>,
    pub(crate) tp_size: Option<u64>,
}

impl NixlIdentity {
    pub(crate) fn compact_workers(
        records: &[Value],
        architecture: Option<&str>,
    ) -> Result<Value, AicError> {
        Ok(serde_json::json!(Self::project_workers(
            records.iter(),
            architecture
        )?))
    }

    fn project_workers<'a>(
        records: impl Iterator<Item = &'a Value> + Clone,
        architecture: Option<&str>,
    ) -> Result<Vec<Worker>, AicError> {
        let mut schedulers = BTreeMap::new();
        for row in records.clone().filter(|r| r["kind"] == "scheduler") {
            if let Some(layout) = Worker::named_layout(row)? {
                let pool = (row["phase"].as_str(), row["dp_rank"].as_u64());
                if schedulers
                    .insert(pool, layout.clone())
                    .is_some_and(|old| old != layout)
                {
                    return Err(conflict("kv_cache_layout"));
                }
            }
        }
        records
            .filter(|r| r["kind"] != "scheduler")
            .map(|row| {
                let mut worker = Worker::observed(row, architecture)?;
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
                Ok(worker)
            })
            .collect()
    }

    pub(crate) fn bound_cell_digests(digests: Vec<String>) -> Result<Vec<String>, AicError> {
        roster::bound_cell_digests(digests)
    }

    pub(crate) fn observation_digest(
        entry: &Value,
        source: &str,
        inventory: &Value,
        draft: bool,
        cell: Option<&Value>,
    ) -> Result<String, AicError> {
        roster::digest(entry, Some(source), inventory, draft, cell)
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

    fn common_settings(&mut self, config: &Value, source: &Value) -> Result<(), AicError> {
        self.settings(config)?;
        self.metadata.merge(&TransferMetadata::configured(
            config,
            config["model_config"]["use_mla"].as_bool(),
            self.architecture.as_deref(),
            source,
        ))
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
        if capture != Capture::Full
            && entry["artifacts"]
                .as_array()
                .is_some_and(|files| files.iter().any(|file| file.get("evidence").is_some()))
        {
            return Err(roster::invalid());
        }
        self.common_settings(&entry["runtime_settings"], entry)?;
        let formal = entry["source"] == "vllm_initialized_cache";
        if formal {
            self.common_settings(&entry["resolved_config"], entry)?;
        }
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
            || capture == Capture::BoundReferences;
        let start = self.workers.len();
        // The original finalizer kept the initialized common settings and capacity coordinates,
        // but omitted every worker view. A resolved cache layout still applies to those ranks.
        if capture == Capture::References && complete {
            let mut config = entry["runtime_settings"].clone();
            if formal {
                // The formal writer retained both copies of these initialized worker settings.
                // Its normalized copy deliberately omits phase-specific cache policy.
                for (section, key) in [
                    ("cache_config", "kv_cache_layout"),
                    ("model_config", "use_mla"),
                    ("kv_transfer_config", "enable_permute_local_kv"),
                    ("kv_transfer_config", "kv_buffer_device"),
                ] {
                    let other = &entry["resolved_config"][section][key];
                    if !other.is_null() {
                        if !config[section][key].is_null() && config[section][key] != *other {
                            return Err(conflict(key));
                        }
                        if !config[section].is_null() && !config[section].is_object() {
                            return Err(conflict(key));
                        }
                        config[section][key] = other.clone();
                    }
                }
            }
            let common = Worker::observed(
                &serde_json::json!({"resolved_config":config,"launch":entry["launch"],
                    "runtime":entry["runtime"], "source": entry["source"], "backend_version":entry["backend_version"]}),
                self.architecture.as_deref(),
            )?;
            if common.gpu_cache_layout.is_some()
                || common.use_mla.is_some()
                || common.permit_permute.is_some()
                || common.host_buffer == Some(true)
            {
                for rank in entry["rank_capacities"].as_array().into_iter().flatten() {
                    self.workers.push(Worker {
                        phase: rank["phase"].as_str().map(str::to_owned),
                        dp_rank: rank["dp_rank"].as_u64(),
                        tp_rank: rank["tp_rank"].as_u64(),
                        ..common.clone()
                    });
                }
            }
        }
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
                if capture == Capture::BoundReferences && !worker["host_buffer"].is_boolean() {
                    return Err(roster::invalid());
                }
                for field in ["kv_cache_layout", "gpu_cache_layout"] {
                    match &worker[field] {
                        Value::Null => {}
                        Value::String(s) if Layouts::named(s).is_some() => {}
                        Value::Array(names)
                            if names.len() > 1
                                && names
                                    .iter()
                                    .all(|n| n.as_str().and_then(Layouts::named).is_some())
                                && names.windows(2).all(|n| n[0].as_str() < n[1].as_str()) => {}
                        _ => return Err(roster::invalid()),
                    }
                }
                let mut worker: Worker = serde_json::from_value(worker.clone()).map_err(|e| {
                    AicError::InvalidEngineConfig(format!(
                        "invalid grouped worker observation: {e}"
                    ))
                })?;
                worker.metadata.uniform_attention = worker
                    .metadata
                    .uniform_attention
                    .take()
                    .and_then(|attention| attention.for_selected(self.architecture.as_deref()));
                self.workers.push(worker);
            }
        } else if let Some(names) = entry["worker_attention_backends"].as_array() {
            // An optional backend list has no TP coordinates. The verified importer emits the
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
            let mut phase_groups: BTreeMap<Option<&str>, Self> = BTreeMap::new();
            for artifact in artifacts {
                let rank = &artifact["evidence"];
                if rank["kind"].as_str() == Some("worker") {
                    self.settings(&rank["resolved_config"])?;
                    if roster::identified_source(entry) {
                        phase_groups
                            .entry(rank["phase"].as_str())
                            .or_default()
                            .worker_groups(
                                &rank["cache"],
                                rank["runtime"]["version"]
                                    .as_str()
                                    .or(rank["backend_version"].as_str()),
                            )?;
                    } else {
                        self.measured_groups(&rank["cache"]["groups"])?;
                    }
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
            // The importer keeps prefill geometry first in a capture of both phases. Mamba
            // block size follows the phase's cache mode. Ranks inside each phase still agree.
            let different_modes = entry
                .pointer("/prefix_caching/prefill/mamba_cache_mode")
                .and_then(Value::as_str)
                .zip(
                    entry
                        .pointer("/prefix_caching/decode/mamba_cache_mode")
                        .and_then(Value::as_str),
                )
                .is_some_and(|(a, b)| a != b);
            let has_prefill = phase_groups.contains_key(&Some("prefill"));
            for (phase, identity) in phase_groups {
                if let Some(mut groups) = identity.recorded_groups {
                    if has_prefill && phase == Some("decode") && different_modes {
                        for group in &mut groups {
                            if group["kind"] == "state" {
                                group.as_object_mut().unwrap().remove("block_size_tokens");
                            }
                        }
                    }
                    self.normalized_groups(&Value::Array(groups))?;
                }
            }
            self.workers.extend(Self::project_workers(
                artifacts
                    .iter()
                    .map(|a| &a["evidence"])
                    .filter(|r| matches!(r["kind"].as_str(), Some("worker" | "scheduler"))),
                self.architecture.as_deref(),
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

    fn worker_groups(&mut self, cache: &Value, version: Option<&str>) -> Result<(), AicError> {
        let Some(groups) = cache["groups"].as_array() else {
            return Ok(());
        };
        // The importer uses the allocation of one shared pool block for every group.
        // spec_page_size_bytes is the page of a single layer or spec, not that allocation.
        let page = cache["allocated_cache_bytes"]
            .as_u64()
            .zip(cache["num_blocks"].as_u64())
            .and_then(|(bytes, count)| {
                (count > 0 && bytes > 0 && bytes % count == 0).then(|| bytes / count)
            });
        let normalized: Vec<_> = groups.iter().map(|g| serde_json::json!({
            "kind":g["kind"], "num_layers":g["layer_names"].as_array().map(Vec::len),
            "block_size_tokens":g["block_size_tokens"], "page_size_bytes":page,
            "sliding_window":g["sliding_window"], "blocks_per_request":g["state"]["blocks_per_request"],
            "prefix_cacheable":version.filter(|v| v.split('+').next() == Some("0.33.0")).and_then(|_| layout::prefix_cacheable(g)),
            "prefill_checkpoint_blocks":g["state"]["num_prefill_checkpoint_blocks"],
            "prefill_checkpoint_alignment":g["state"]["prefill_checkpoint_alignment"],
        })).collect();
        self.normalized_groups(&Value::Array(normalized))
    }

    // A normalized cell or probe records shared-pool geometry. Keep optional fields
    // that its writer actually recorded; older receipts can be augmented with markers.
    fn normalized_groups(&mut self, groups: &Value) -> Result<(), AicError> {
        self.measured_groups(groups)?;
        let Some(groups) = groups.as_array().filter(|g| !g.is_empty()) else {
            return Ok(());
        };
        let saved = self
            .recorded_groups
            .get_or_insert_with(|| vec![Value::Null; groups.len()]);
        if saved.len() != groups.len() {
            return Err(conflict("cache_groups"));
        }
        for (recorded, group) in saved.iter_mut().zip(groups) {
            // A missing sliding window denotes full attention in the normalized profile.
            for field in [
                "kind",
                "num_layers",
                "block_size_tokens",
                "page_size_bytes",
                "sliding_window",
                "blocks_per_request",
                "prefix_cacheable",
                "transfer_bytes_per_block",
                "prefill_checkpoint_blocks",
                "prefill_checkpoint_alignment",
            ] {
                let value = &group[field];
                if value.is_null() && !(field == "sliding_window" && group["kind"] == "attention") {
                    continue;
                }
                if recorded.get(field).is_some_and(|v| v != value) {
                    return Err(conflict(field));
                }
                recorded[field] = value.clone();
            }
        }
        Ok(())
    }

    pub(crate) fn selected_memory(
        &self,
        selected: &Value,
        limits: Option<(Option<u64>, Option<u64>)>,
    ) -> Result<(), AicError> {
        if let Some(fields) = self.recorded_memory.as_object() {
            for (field, value) in fields {
                let limit = match field.as_str() {
                    "max_num_tokens" => limits.map(|v| v.0),
                    "max_batch_size" => limits.map(|v| v.1),
                    _ => None,
                };
                if matches!(field.as_str(), "max_num_tokens" | "max_batch_size") {
                    if limit.is_some_and(|n| n != value.as_u64()) {
                        return Err(conflict(field));
                    }
                } else if !roster::same_memory_setting(field, &selected[field], value) {
                    return Err(conflict(field));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn selected_groups(&mut self, groups: &Value) -> Result<(), AicError> {
        if let Some(saved) = &self.recorded_groups {
            let selected = groups
                .as_array()
                .filter(|g| g.len() == saved.len())
                .ok_or_else(|| conflict("cache_groups"))?;
            for (recorded, selected) in saved.iter().zip(selected) {
                if let Some(fields) = recorded.as_object() {
                    for (field, value) in fields {
                        if selected[field] != *value {
                            return Err(conflict(field));
                        }
                    }
                }
            }
        }
        self.measured_groups(groups)
    }

    fn measured_groups(&mut self, groups: &Value) -> Result<(), AicError> {
        self.hybrid_groups |= groups.as_array().is_some_and(|groups| {
            groups
                .iter()
                .any(|g| g["kind"] == "state" || g["sliding_window"].is_u64())
        });
        // The importer records a prefill checkpoint count (including zero) only for Mamba.
        if groups.as_array().is_some_and(|groups| {
            groups
                .iter()
                .any(|g| g["kind"] == "state" && g["prefill_checkpoint_blocks"].is_u64())
        }) {
            self.measured_mamba = true;
            record(
                &mut self.disable_hma,
                Some("false"),
                "disable_hybrid_kv_cache_manager",
            )?;
        }
        Ok(())
    }

    fn probe(&mut self, entry: &Value) -> Result<(), AicError> {
        self.collection(entry, Capture::Full)?;
        let version = entry["schema_version"].as_str();
        let current = version == Some("aisimulate-onboarding-runtime-probe/v2");
        let snapshot = &entry["nixl_observation"];
        if !current
            && (snapshot.get("worker_observations").is_some()
                || snapshot.get("worker_observations_sha256").is_some())
        {
            return Err(roster::invalid());
        }
        if current {
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
                true,
            )?;
            {
                let source = entry["source_artifacts"]
                    .as_array()
                    .and_then(|refs| {
                        refs.iter()
                            .find(|r| r["path"] == entry["observations_index"])
                    })
                    .and_then(|r| r["sha256"].as_str());
                if snapshot["worker_observations_sha256"]
                    != roster::digest(
                        snapshot,
                        source,
                        entry,
                        true,
                        entry.get("observed_resources"),
                    )?
                {
                    return Err(roster::invalid());
                }
            }
        }
        match version {
            Some(
                "aisimulate-onboarding-runtime-probe/v1" | "aisimulate-onboarding-runtime-probe/v2",
            ) => {
                roster::validate_probe(entry)?;
                self.normalized_groups(&entry["observed_resources"]["cache_groups"])?;
                let memory = &entry["observed_resources"]["runtime_memory"];
                self.memory(memory)?;
                let legacy = entry["schema_version"] == "aisimulate-onboarding-runtime-probe/v1";
                let snapshot = &entry["nixl_observation"];
                if legacy && entry.get("nixl_observation").is_some() && !snapshot.is_object() {
                    return Err(roster::invalid());
                }
                self.observation(
                    snapshot,
                    self.phase_width.filter(|_| legacy),
                    if legacy {
                        Capture::References
                    } else {
                        Capture::BoundReferences
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
        architecture: Option<&str>,
    ) -> Result<Self, AicError> {
        let mut identity = Self {
            phase_width,
            tp_size,
            architecture: architecture.map(str::to_owned),
            model_dtype: recorded_dtype.and_then(FpmRuntimeMemoryConfig::normalize_model_dtype),
            ..Self::default()
        };
        let Ok(evidence) = serde_json::from_str::<Value>(provenance) else {
            return Ok(identity);
        };
        identity.probe(&evidence)?;
        identity.recorded_memory = roster::selected_memory(&evidence)?;
        let schema = evidence["schema_version"].as_str().unwrap_or_default();
        let finalized = schema == FINALIZATION;
        if !finalized
            && (schema.starts_with("aisimulate-onboarding-finalization/")
                || (evidence["source_request_id"].is_string()
                    && evidence["source_collection_plan_sha256"].is_string()
                    && evidence["cell_observations"].is_array()))
        {
            return Err(AicError::InvalidEngineConfig(
                "invalid indexed finalization schema".into(),
            ));
        }
        if let Some(probe) = evidence.get("runtime_probe") {
            identity.probe(probe)?;
        }
        let format = evidence["observation_provenance"].as_str();
        let capture = match format {
            Some("source_references_v2") => Capture::BoundReferences,
            Some("source_references") => Capture::References,
            Some(other) if other.starts_with("source_references_v") => {
                return Err(AicError::InvalidEngineConfig(
                    "unsupported finalized observation provenance format".into(),
                ));
            }
            _ if finalized && evidence.get("observation_provenance").is_some() => {
                return Err(AicError::InvalidEngineConfig(
                    "unsupported finalized observation provenance format".into(),
                ));
            }
            _ => Capture::Full,
        };
        let bound = capture == Capture::BoundReferences;
        if !bound && evidence.get("worker_observation_digests").is_some() {
            return Err(roster::invalid());
        }
        if capture != Capture::Full
            && evidence["cell_observations"]
                .as_array()
                .is_none_or(Vec::is_empty)
        {
            return Err(roster::invalid());
        }
        let mut digests = Vec::new();
        let mut formal_cells = std::collections::BTreeSet::new();
        let mut indexed_cell = false;
        if let Some(cells) = evidence["cell_observations"].as_array() {
            let measured = finalized
                || bound
                || evidence["capacity_policy"]
                    == "minimum observed request-usable rank-local capacity across compatible cells";
            let mut capacities = Vec::new();
            for cell in cells {
                let memory = &cell["runtime_memory"];
                identity.memory(memory)?;
                let nested = memory["provenance"]
                    .as_str()
                    .and_then(|s| serde_json::from_str(s).ok());
                let nested: Value = nested.unwrap_or_default();
                if capture != Capture::Full
                    && (!roster::identified_source(&nested)
                        || nested["runtime_settings"]
                            .as_object()
                            .is_none_or(|s| s.is_empty())
                        || nested["artifacts"].as_array().is_none_or(Vec::is_empty))
                {
                    return Err(roster::invalid());
                }
                indexed_cell |= nested["source"] == "runtime_observations";
                if roster::identified_source(&nested) {
                    if measured
                        && let Some(capacity) = roster::record_cell_memory(
                            cell,
                            &nested,
                            if finalized {
                                &evidence["runtime_compatibility"]
                            } else {
                                &Value::Null
                            },
                            &mut identity.recorded_memory,
                        )?
                    {
                        capacities.push(capacity);
                    }
                    if finalized && nested["source"] == "runtime_observations" {
                        roster::validate_formal_index(&nested, &evidence)?;
                    }
                    if capture != Capture::Full
                        && (bound || finalized || evidence["capacity_policy"].is_string())
                    {
                        let groups: Vec<super::FpmCacheGroup> =
                            serde_json::from_value(cell["cache_groups"].clone())
                                .map_err(|_| roster::invalid())?;
                        if groups.is_empty() {
                            return Err(roster::invalid());
                        }
                        for group in groups {
                            group.validate()?;
                        }
                    }
                    identity.normalized_groups(&cell["cache_groups"])?;
                }
                if bound {
                    if nested["source"] == "vllm_initialized_cache" {
                        let id = nested["cell_id"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .ok_or_else(roster::invalid)?;
                        if !formal_cells.insert(id.to_owned()) {
                            return Err(roster::invalid());
                        }
                    }
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
                        true,
                    )?;
                    digests.push(roster::digest(
                        &nested,
                        roster::cell_source(&nested),
                        &evidence,
                        false,
                        Some(cell),
                    )?);
                }
                if nested.get("nixl_observation").is_some()
                    || nested["schema_version"]
                        .as_str()
                        .is_some_and(|s| s.starts_with(PROBE_FAMILY))
                {
                    identity.probe(&nested)?;
                } else if !identity.collection(&nested, capture)? {
                    identity.workers.push(Worker::default());
                }
            }
            if capacities.len() == cells.len()
                && let Some(minimum) = capacities.into_iter().min()
            {
                identity.recorded_memory["kv_cache_bytes"] = Value::from(minimum);
            }
        }
        if finalized
            && (indexed_cell
                || evidence.get("runtime_compatibility").is_some()
                || evidence.get("runtime_probe").is_some())
        {
            let source = evidence["runtime_compatibility"]["source_observations_index"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(roster::invalid)?;
            let probe = &evidence["runtime_probe"];
            if !matches!(
                probe["schema_version"].as_str(),
                Some(
                    "aisimulate-onboarding-runtime-probe/v1"
                        | "aisimulate-onboarding-runtime-probe/v2"
                )
            ) || probe["observations_index"].as_str() != Some(source)
            {
                return Err(roster::invalid());
            }
        }
        if bound {
            if digests.is_empty() {
                return Err(roster::invalid());
            }
            let mut expected = evidence["worker_observation_digests"]
                .as_array()
                .ok_or_else(roster::invalid)?
                .iter()
                .map(|v| v.as_str().ok_or_else(roster::invalid))
                .collect::<Result<Vec<_>, _>>()?;
            expected.sort_unstable();
            if expected != Self::bound_cell_digests(digests)? {
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

    fn nemotron_attention_regions(&self) -> Option<layout::LayerRegions> {
        let measured = self
            .metadata
            .registered_layers
            .as_ref()
            .and_then(|names| layout::nemotron_owners(names, "mixer.attn"));
        let Some(logical) = self.metadata.nemotron_attention_layers.as_ref() else {
            return measured;
        };
        let states = self.metadata.nemotron_state_regions.as_ref();
        let single_full_group = self.recorded_groups.as_deref().is_some_and(|groups| {
            matches!(groups, [g] if g["kind"] == "attention" && g["sliding_window"].is_null()
                && g["num_layers"].as_u64() == Some(logical.len() as u64))
        });
        let layer_major = self.workers.iter().any(|worker| {
            worker
                .gpu_cache_layout
                .as_ref()
                .is_some_and(|layout| layout.names().iter().all(|name| name.starts_with('L')))
        });
        let positions = logical.iter().copied().collect();
        let mut owners = if single_full_group
            && layer_major
            && states.is_some_and(|s| s.first_possible().is_none())
        {
            // The only full-attention group allocates a distinct layer-major region per layer.
            layout::LayerRegions::exact(positions)
        } else {
            layout::LayerRegions::new(positions, states.and_then(|r| r.first_possible()))
        };
        if let Some(measured) = measured {
            owners.merge(&measured);
        }
        Some(owners)
    }

    fn nemotron_decoder(&self) -> bool {
        self.metadata
            .uniform_attention
            .as_ref()
            .filter(|heads| heads.ordered_architecture)
            .and_then(|heads| heads.architecture.as_deref())
            .or(self.architecture.as_deref())
            .and_then(Decoder::from_name)
            == Some(Decoder::NemotronH)
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
        if self.measured_mamba {
            for worker in self
                .workers
                .iter_mut()
                .chain(self.rosters.iter_mut().flatten())
            {
                worker.requires_hma = true;
            }
        }
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
        for worker in &self.workers {
            self.metadata.merge(&worker.metadata)?;
        }
        if self.nemotron_decoder() {
            if self
                .metadata
                .nemotron_state_regions
                .as_ref()
                .is_some_and(|r| r.missing(r).is_some())
            {
                return Err(conflict("Mamba layer positions"));
            }
            let positions = std::mem::take(&mut self.metadata.hybrid_attention_positions);
            for layers in positions {
                if self
                    .metadata
                    .nemotron_attention_layers
                    .get_or_insert_with(|| layers.clone())
                    != &layers
                {
                    return Err(conflict("attention layer positions"));
                }
            }
            if self
                .nemotron_attention_regions()
                .is_some_and(|owners| owners.missing(&owners).is_some())
            {
                return Err(conflict("attention layer positions"));
            }
        }
        if let Some(configured) = self.tp_size.and_then(|tp| {
            self.metadata
                .uniform_attention
                .as_ref()?
                .rank_heads(tp, self.architecture.as_deref())
        }) {
            for layer in self.metadata.layer_heads.values() {
                if let Some((field, _, _)) = configured.mismatch(layer) {
                    return Err(conflict(field));
                }
            }
        }
        if matches!(phase, "prefill" | "decode") && self.metadata.speculative == Some(true) {
            return Err(AicError::InvalidEngineConfig(
                "grouped FPM cache requires vLLM without speculative decoding".into(),
            ));
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
        if self.workers.iter().any(|w| w.requires_hma) {
            record(
                &mut self.disable_hma,
                Some("false"),
                "disable_hybrid_kv_cache_manager",
            )?;
        }
        if self
            .workers
            .iter()
            .any(|w| w.requires_gpu_buffer && w.host_buffer == Some(true))
        {
            return Err(conflict("CSA-linear NIXL host buffer"));
        }
        self.engines =
            layout::Deployment::from_workers(&self.workers, self.phase_width, self.tp_size)?;
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
        self.check_version(recorded_version, prefix_caching)?;
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

    pub(crate) fn check_version(
        &self,
        recorded_version: Option<&str>,
        prefix_caching: bool,
    ) -> Result<(), AicError> {
        if let (Some(actual), Some(modeled)) =
            (self.version.as_deref(), self.modeled_version.as_deref())
            && !FpmRuntimeMemoryConfig::matches_grouped_version(actual, modeled, prefix_caching)
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
        Ok(())
    }

    pub fn mismatch(&self, decode: &Self) -> Option<(&'static str, String, String)> {
        if let (Some(a), Some(b)) = (self.metadata.transfer_mode, decode.metadata.transfer_mode)
            && a != b
        {
            return Some(("transfer_mode", format!("{a:?}"), format!("{b:?}")));
        }
        if let (Some(a), Some(b)) = (
            self.metadata.num_hidden_layers,
            decode.metadata.num_hidden_layers,
        ) && a != b
        {
            return Some(("num_hidden_layers", a.to_string(), b.to_string()));
        }
        if let Some(difference) = self
            .metadata
            .model_heads
            .mismatch(&decode.metadata.model_heads)
        {
            return Some(difference);
        }
        if let (Some(a), Some(b)) = (
            &self.metadata.registered_layers,
            &decode.metadata.registered_layers,
        ) && a != b
        {
            return Some(("cache region names", format!("{a:?}"), format!("{b:?}")));
        }
        if self.nemotron_decoder() && decode.nemotron_decoder() {
            let attention = (
                self.nemotron_attention_regions(),
                decode.nemotron_attention_regions(),
            );
            for (field, suffix, local, remote) in [
                (
                    "attention layer positions",
                    "mixer.attn",
                    attention.0.as_ref(),
                    attention.1.as_ref(),
                ),
                (
                    "Mamba layer positions",
                    "mixer",
                    self.metadata.nemotron_state_regions.as_ref(),
                    decode.metadata.nemotron_state_regions.as_ref(),
                ),
            ] {
                if let (Some(a), Some(b)) = (local, remote) {
                    for (required, possible, transpose) in [(a, b, false), (b, a, true)] {
                        if let Some(i) = required.missing(possible) {
                            let name = format!("model.layers.{i}.{suffix}");
                            let absent = "absent".to_owned();
                            return Some(if transpose {
                                (field, absent, name)
                            } else {
                                (field, name, absent)
                            });
                        }
                    }
                }
            }
        }
        if let Some(tp) = self.tp_size.filter(|size| Some(*size) == decode.tp_size) {
            let configured = |rank: &Self| {
                rank.metadata
                    .uniform_attention
                    .as_ref()?
                    .rank_heads(tp, rank.architecture.as_deref())
            };
            let prefill_heads = configured(self);
            let decode_heads = configured(decode);
            if let Some(difference) = prefill_heads
                .as_ref()
                .zip(decode_heads.as_ref())
                .and_then(|(a, b)| a.mismatch(b))
            {
                return Some(difference);
            }
            for (expected, peer, transpose) in
                [(prefill_heads, decode, false), (decode_heads, self, true)]
            {
                if let Some(expected) = expected {
                    for observed in peer.metadata.layer_heads.values() {
                        if let Some((field, a, b)) = expected.mismatch(observed) {
                            return Some(if transpose {
                                (field, b, a)
                            } else {
                                (field, a, b)
                            });
                        }
                    }
                }
            }
            for (layer, a) in &self.metadata.layer_heads {
                if let Some(b) = decode.metadata.layer_heads.get(layer)
                    && let Some(difference) = a.mismatch(b)
                {
                    return Some(difference);
                }
            }
        }
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
        // Pull performs the full handshake on decode. Push performs it on prefill and does
        // not run Pull's receive-side cache permutation. Model Push with one common GPU layout.
        let push = self
            .metadata
            .transfer_mode
            .or(decode.metadata.transfer_mode)
            == Some(TransferMode::Push);
        let (remote, local) = if push { (decode, self) } else { (self, decode) };
        let common_gpu = || {
            let workers = self.workers.iter().chain(&decode.workers);
            let only_mla = workers
                .clone()
                .filter(|w| w.gpu_cache_layout.is_some() || w.kv_cache_layout.is_some())
                .all(|w| w.single_head_mla);
            workers.fold(u8::MAX, |domain, w| domain & layout::push_gpu(w, only_mla)) != 0
        };
        let no_repack = push
            || local.hybrid_groups
                && [self.disable_hma.as_deref(), decode.disable_hma.as_deref()]
                    .contains(&Some("false"));
        let excludes = |p: &Worker, d: &Worker| {
            if push {
                d.excludes_layout(p, version, no_repack)
            } else {
                p.excludes_layout(d, version, no_repack)
            }
        };
        if (push && !common_gpu()) || incompatible(&excludes) {
            return Some((
                "kv_cache_layout",
                "prefill workers".into(),
                "decode workers".into(),
            ));
        }
        let tp = self
            .tp_size
            .filter(|t| Some(*t) == decode.tp_size && *t > 0);
        if tp.is_some()
            && !layout::engines_routable(&remote.engines, &local.engines, version, no_repack)
        {
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
                let p: Vec<_> = remote.workers.iter().filter(at_slot).collect();
                let d: Vec<_> = local.workers.iter().filter(at_slot).collect();
                // A missing prefill observation leaves one freely chosen, immutable layout.
                // A missing decode observation also leaves the MLA exemption possible.
                if (p.len() > 1 || d.len() > 1) && !layout::routable(&p, &d, version, no_repack) {
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
                if !roster::can_pair(p, d, |a, b| compatible(a, b) && !excludes(a, b)) {
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
        let mut identity =
            NixlIdentity::observed(None, &evidence.to_string(), None, None, None).unwrap();
        identity
            .select(phase, None, None, None, None, false)
            .unwrap();
        identity
    }

    fn legacy_probe(names: Value) -> Value {
        let sources = ["observations.json", "launch.json", "model-config.json"]
            .map(|file| json!({"path":format!("x/{file}"),"sha256":"0".repeat(64)}));
        json!({"schema_version":"aisimulate-onboarding-runtime-probe/v1",
            "checkpoint":"checkpoint","configuration":"tp1","observations_index":"x/observations.json",
            "source_artifacts":sources,
            "model_metadata":{},"user_overrides":{},
            "observed_resources":{"cache_layout":"grouped","runtime_memory":{"kv_cache_bytes":64},
                "cache_groups":[{"name":"full","kind":"attention","num_layers":1,"block_size_tokens":16,"page_size_bytes":64}]},
            "nixl_observation":{"worker_attention_backends":names}})
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
        assert!(NixlIdentity::observed(None, &evidence.to_string(), None, None, None).is_err());
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
                None,
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
            let saved = legacy_probe(names);
            let mut identity =
                NixlIdentity::observed(None, &saved.to_string(), width, None, None).unwrap();
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
            let saved =
                json!({"worker_observations": NixlIdentity::compact_workers(&raw, None).unwrap()});
            let mut id =
                NixlIdentity::observed(None, &saved.to_string(), Some(4), Some(4), None).unwrap();
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
        let impossible = legacy_probe(two_phases);
        assert!(
            NixlIdentity::observed(None, &impossible.to_string(), Some(3), None, None).is_err()
        );
        assert!(NixlIdentity::observed(None, &impossible.to_string(), None, None, None).is_ok());
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
    fn nemotron_registered_names_identify_attention_positions_across_formats() {
        let role = |names: &[&str], pattern: Option<Vec<usize>>, phase| {
            let mut id = NixlIdentity {
                architecture: Some("NemotronHForCausalLM".into()),
                metadata: TransferMetadata {
                    registered_layers: (!names.is_empty())
                        .then(|| names.iter().map(|name| (*name).to_owned()).collect()),
                    nemotron_attention_layers: pattern,
                    ..Default::default()
                },
                ..Default::default()
            };
            id.select(phase, None, None, None, None, false).unwrap();
            id
        };
        let raw = role(
            &["model.layers.0.mixer.attn", "model.layers.1.mixer.attn"],
            None,
            "prefill",
        );
        let same = role(&[], Some(vec![0, 1]), "decode");
        let different = role(&[], Some(vec![2, 3]), "decode");
        assert!(raw.mismatch(&same).is_none());
        assert_eq!(
            raw.mismatch(&different).unwrap().0,
            "attention layer positions"
        );
        assert_eq!(
            different.mismatch(&raw).unwrap().0,
            "attention layer positions"
        );
        assert!(
            role(&["other.layers.0"], None, "prefill")
                .mismatch(&different)
                .is_none()
        );
    }

    #[test]
    fn selected_nemotron_interprets_the_saved_top_level_layer_pattern() {
        let selected = |model: Value, arch: &str| {
            let mut id = observe(json!({"runtime_settings":{"model_config":model}}), "decode");
            id.architecture = Some(arch.into());
            id.select("decode", None, None, None, None, false).unwrap();
            id.metadata.nemotron_attention_layers
        };
        let arch = "NemotronHForCausalLM";
        let compact = json!({"hf_overrides":{"hybrid_override_pattern":"--**"}});
        let role = |p, arch: &str| {
            let evidence = json!({"runtime_settings":{"model_config":{"hf_overrides":{"hybrid_override_pattern":p}}}});
            let mut id =
                NixlIdentity::observed(None, &evidence.to_string(), None, None, Some(arch))
                    .unwrap();
            id.select("prefill", None, None, None, None, false).unwrap();
            id
        };
        let puzzle = "NemotronHPuzzleForCausalLM";
        assert!(role("**--", puzzle).mismatch(&role("**--", arch)).is_none());
        assert_eq!(
            role("**--", puzzle)
                .mismatch(&role("--**", puzzle))
                .unwrap()
                .0,
            "attention layer positions"
        );
        assert_eq!(selected(compact.clone(), arch), Some(vec![2, 3]));
        assert!(selected(compact, "LlamaForCausalLM").is_none());
        assert!(selected(json!({"hf_overrides":{}}), arch).is_none());
        assert!(
            selected(
                json!({"hf_overrides":{"text_config":{"hybrid_override_pattern":"**--"}}}),
                arch
            )
            .is_none()
        );
        assert_eq!(
            selected(
                json!({"hf_config":{"hybrid_override_pattern":"**--"},
            "hf_overrides":{"hybrid_override_pattern":"--**"}}),
                arch
            ),
            Some(vec![0, 1])
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
            assert!(NixlIdentity::observed(None, &corrupt.to_string(), None, None, None).is_err());
        }
    }

    #[test]
    fn bound_drafts_require_ranked_observations_and_their_source_digest() {
        let raw = json!({"tp_rank":0,"dp_rank":0,"phase":"prefill"});
        let mut probe = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v2",
            "nixl_observation":{"runtime":{"version":"0.33.0"},"runtime_settings":{"model_config":{"model":"example","dtype":"bfloat16"},"cache_config":{"cache_dtype":"auto"}},
            "worker_observations":NixlIdentity::compact_workers(&[raw], None).unwrap()}});
        assert!(NixlIdentity::observed(None, &probe.to_string(), Some(1), Some(1), None).is_err());
        probe["nixl_observation"]["worker_observations"][0]["dp_rank"] = Value::Null;
        assert!(NixlIdentity::observed(None, &probe.to_string(), Some(1), Some(1), None).is_err());
    }

    #[test]
    fn imported_anonymous_workers_must_match_every_dp_peer_at_their_tp_rank() {
        fn imported(names: Value, tp: u64, phase: &str) -> NixlIdentity {
            let width = names.as_array().unwrap().len() as u64;
            let probe = legacy_probe(names);
            let mut id =
                NixlIdentity::observed(None, &probe.to_string(), Some(width), Some(tp), None)
                    .unwrap();
            id.select(phase, None, None, None, None, false).unwrap();
            id
        }
        let mut missing = NixlIdentity::observed(None, "", Some(2), Some(1), None).unwrap();
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
            let captures = json!({"source":"runtime_observations","observation_schema":"aisimulate-runtime-observation/v1",
                "observations_index":{"path":"observations.json","sha256":"0".repeat(64)},"runtime_settings":{"parallel_config":{"tensor_parallel_size":2}},"artifacts":(["worker-a","worker-b","scheduler"].map(|p| json!({"path":p,"sha256":"0".repeat(64)}))),"worker_attention_backends":names,
                "rank_capacities":[{"phase":"prefill","tp_rank":0,"dp_rank":0},{"phase":"prefill","tp_rank":1,"dp_rank":0}]});
            json!({"runtime_memory":{"provenance":captures.to_string()}})
        };
        let select = |cells: Vec<Value>| {
            let profile =
                json!({"observation_provenance":"source_references","cell_observations":cells});
            let mut id =
                NixlIdentity::observed(None, &profile.to_string(), Some(2), Some(2), None).unwrap();
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
            assert!(NixlIdentity::observed(None, &value.to_string(), None, None, None).is_err());
        }
        assert!(
            NixlIdentity::observed(
                None,
                r#"{"schema_version":"unrelated/v99"}"#,
                None,
                None,
                None
            )
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
                None,
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
        let w = Worker::observed(&row, None).unwrap();
        assert_eq!(w.gpu_cache_layout, Layouts::named("BLHNC"));
        assert!(w.requires_hma);
        let mut mixed = row.clone();
        mixed["cache"]["groups"][0]["layer_classes"]["b"] = json!("custom.UnknownAttention");
        let w = Worker::observed(&mixed, None).unwrap();
        assert_eq!(w.use_mla, None);
        assert_eq!(w.gpu_cache_layout, Layouts::named("BLHNC"));
        row["raw_cache_config"] = json!({"kv_cache_layout":"LBNHC"});
        assert!(Worker::observed(&row, None).is_err());
        row["cache"]["layer_tensors"] = json!({});
        let w = Worker::observed(&row, None).unwrap();
        assert!(!w.accepts_layout("LBHNC", "LBNHC", Some("0.33.0"), false));
        row["resolved_config"] = json!({"model_config":{"use_mla":true},"kv_transfer_config":{
            "kv_buffer_device":"cpu","enable_permute_local_kv":true}});
        assert_eq!(
            Worker::observed(&row, None).unwrap().kv_cache_layout,
            Layouts::named("LBNHC")
        );
    }

    #[test]
    fn attention_specs_identify_decoder_layouts_for_subclasses() {
        let subclass = "vllm.model_executor.layers.attention.prefill_prefix_lm_attention.PrefillPrefixLMAttention";
        let view = json!({"shape":[10,2,64,16],"stride_bytes":[10240,2048,32,2],"block_axis":0});
        let mut rank = json!({"schema_version":"aisimulate-runtime-observation/v1","runtime":{"version":"0.33.0"},
            "cache":{"groups":[{"kind":"attention","layer_names":["a","b"],"layer_classes":{"a":subclass,"b":subclass}}],
                "tensor_allocations":[{"layers":["a","b"],"host_resident":false,"layer_stride":4096}],
                "layer_tensors":{"a":[view],"b":[view]}}});
        let class_only = Worker::observed(&rank, None).unwrap();
        assert_eq!(class_only.use_mla, Some(false));
        assert_eq!(class_only.gpu_cache_layout, Layouts::named("BLHNC"));
        for (spec, ordinary) in [
            ("FullAttentionSpec", true),
            ("SlidingWindowSpec", true),
            ("SinkFullAttentionSpec", true),
            ("MLAAttentionSpec", false),
            ("SlidingWindowMLASpec", false),
            ("HiddenStateCacheSpec", false),
            ("UniformTypeKVCacheSpecs", false),
        ] {
            let spec = format!("vllm.v1.kv_cache_interface.{spec}");
            for wrapped in [false, true] {
                let group = &mut rank["cache"]["groups"][0];
                if wrapped {
                    group["spec_type"] =
                        json!("vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs");
                    group["layer_specs"] = json!({"a":{"spec_type":spec},"b":{"spec_type":spec}});
                } else {
                    group["spec_type"] = json!(spec);
                    group.as_object_mut().unwrap().remove("layer_specs");
                }
                let worker = Worker::observed(&rank, None).unwrap();
                assert_eq!(worker.use_mla, ordinary.then_some(false));
                if ordinary {
                    assert_eq!(worker.gpu_cache_layout, Layouts::named("BLHNC"));
                } else {
                    assert_eq!(worker.gpu_cache_layout, None);
                }
            }
        }
    }

    #[test]
    fn inkling_windows_constrain_layout_without_reporting_ordinary_head_dimensions() {
        let class = "vllm.models.inkling.nvidia.sconv_swa_attn.InklingConvState";
        let full = "vllm.v1.kv_cache_interface.FullAttentionSpec";
        let sliding = "vllm.v1.kv_cache_interface.SlidingWindowSpec";
        let view = json!({"shape":[8,2,4,8],"stride_bytes":[256,64,16,2],"block_axis":0});
        let mut row = json!({"schema_version":"aisimulate-runtime-observation/v1","runtime":{"version":"0.32.0"},
            "cache":{"groups":[{"kind":"attention","spec_type":full,"layer_names":["a"]},
                {"kind":"convolution","spec_type":sliding,"layer_classes":{"b":class,"c":class}}],
                "tensor_allocations":[{"layers":["a"],"host_resident":false,"layer_stride":0},
                    {"layers":["b","c"],"host_resident":false,"layer_stride":128}],
                "layer_tensors":{"a":view,"b":view,"c":view}}});
        let worker = Worker::observed(&row, None).unwrap();
        assert_eq!(worker.gpu_cache_layout, Layouts::named("BLHNC"));
        assert_eq!(
            worker.metadata.layer_heads.keys().collect::<Vec<_>>(),
            ["a"]
        );
        row["cache"]["groups"][0]["layer_specs"] = json!({"a":{"spec_type":"unknown"}});
        assert_eq!(
            Worker::observed(&row, None).unwrap().metadata.layer_heads["a"].num_kv_heads,
            Some(2)
        );
        row["cache"]["groups"][0]["layer_specs"] = json!({"a":{"spec_type":sliding}});
        assert!(Worker::observed(&row, None).is_err());
        row["cache"]["groups"][0]["spec_type"] =
            json!("vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs");
        assert!(Worker::observed(&row, None).is_ok());
    }

    #[test]
    fn observed_head_geometry_and_connector_mode_survive_worker_compaction() {
        let identity = |heads: u64, dim: u64, compact: bool, connector: &str| {
            let spec = "vllm.v1.kv_cache_interface.FullAttentionSpec";
            let row = json!({"schema_version":"aisimulate-runtime-observation/v1","kind":"worker",
                "resolved_config":{"kv_transfer_config":{"kv_connector":connector}},
                "raw_cache_config":{"groups":[{"spec_type":spec,"layer_names":["a"],"num_kv_heads":heads,"head_size":dim}]},
                "cache":{"groups":[{"kind":"attention","spec_type":spec,"layer_names":["a"]}]}});
            let evidence = if compact {
                json!({"worker_observations":NixlIdentity::compact_workers(&[row], None).unwrap()})
            } else {
                json!({"artifacts":[{"evidence":row}]})
            };
            let mut id =
                NixlIdentity::observed(None, &evidence.to_string(), Some(2), Some(2), None)
                    .unwrap();
            id.select("decode", None, None, None, None, false).unwrap();
            id
        };
        let prefill = identity(2, 128, false, "NixlPullConnector");
        for compact in [false, true] {
            for (heads, width, connector, expected) in [
                (2, 128, "NixlConnector", None),
                (4, 128, "NixlConnector", Some("num_kv_heads")),
                (2, 64, "NixlConnector", Some("head_size")),
                (2, 128, "NixlPushConnector", Some("transfer_mode")),
            ] {
                assert_eq!(
                    prefill
                        .mismatch(&identity(heads, width, compact, connector))
                        .map(|d| d.0),
                    expected
                );
            }
        }
        let model = |count, width| {
            observe(
                json!({"runtime_settings":{"model_config":{"use_mla":false,
            "hf_config":{"model_type":"llama"},
            "hf_text_config":{"model_type":"llama","num_key_value_heads":count,"head_dim":width}}}}),
                "decode",
            )
        };
        assert_eq!(
            model(4, 128).mismatch(&model(8, 128)).unwrap().0,
            "num_kv_heads"
        );
        assert_eq!(
            model(4, 128).mismatch(&model(4, 64)).unwrap().0,
            "head_size"
        );
        assert!(model(4, 128).mismatch(&NixlIdentity::default()).is_none());
    }

    #[test]
    fn effective_model_geometry_depends_on_the_recorded_converter() {
        let model = |over| {
            observe(
                json!({"runtime_settings":{"model_config":{
            "use_mla":false,"hf_overrides":over}}}),
                "decode",
            )
        };
        for family in ["falcon", "unknown"] {
            let first = model(
                json!({"model_type":family,"n_head_kv":4,"multi_query":true,"new_decoder_architecture":false}),
            );
            let second = model(
                json!({"model_type":family,"n_head_kv":8,"multi_query":true,"new_decoder_architecture":false}),
            );
            assert!(first.mismatch(&second).is_none());
        }
        let partial = |n| model(json!({"model_type":"qwen2","num_attention_groups":n}));
        assert!(partial(4).mismatch(&partial(8)).is_none());
        let shadowed = json!({"model_type":"qwen2","num_key_value_heads":4});
        let resolved = json!({"model_type":"qwen2","num_key_value_heads":4,"n_head_kv":8});
        assert!(
            model(shadowed.clone())
                .mismatch(&model(resolved.clone()))
                .is_none()
        );
        for (overlay, effective, expected) in [
            (shadowed.clone(), Value::Null, None),
            (shadowed, resolved.clone(), Some(8)),
            (resolved, Value::Null, Some(8)),
            (
                json!({"model_type":"qwen2","n_head_kv":null,"num_kv_heads":null,"num_key_value_heads":4}),
                Value::Null,
                None,
            ),
        ] {
            let config =
                json!({"model_config":{"hf_overrides":overlay,"hf_text_config":effective}});
            assert_eq!(
                TransferMetadata::configured(&config, Some(false), None, &Value::Null)
                    .model_heads
                    .num_kv_heads,
                expected
            );
        }
        let layers = |n| model(json!({"model_type":"nemotron_h","num_hidden_layers":n}));
        // NemotronH's setter ignores this scalar. A converter result records the actual count.
        assert!(layers(3).mismatch(&layers(4)).is_none());
        let converted = |n| {
            observe(
                json!({"runtime_settings":{"model_config":{"model_arch_config":{
                    "model_type":"nemotron_h","total_num_hidden_layers":n}}}}),
                "decode",
            )
        };
        assert_eq!(
            converted(3).mismatch(&converted(4)).unwrap().0,
            "num_hidden_layers"
        );
        assert!(layers(3).mismatch(&model(json!({}))).is_none());
    }

    #[test]
    fn compact_model_hash_and_rank_geometry_follow_the_selected_decoder() {
        let role = |arch, model: Value, measured: Option<(u64, u64)>| {
            let mut evidence = json!({"runtime_settings":{"model_config":model}});
            if let Some((heads, dim)) = measured {
                let spec = "vllm.v1.kv_cache_interface.FullAttentionSpec";
                evidence["artifacts"] = json!([{"evidence":{"kind":"worker","schema_version":"aisimulate-runtime-observation/v1",
                    "raw_cache_config":{"groups":[{"spec_type":spec,"layer_names":["a"],"num_kv_heads":heads,"head_size":dim}]},
                    "cache":{"groups":[{"kind":"attention","spec_type":spec,"layer_names":["a"]}]}}}]);
            }
            let mut id =
                NixlIdentity::observed(None, &evidence.to_string(), Some(2), Some(2), Some(arch))
                    .unwrap();
            id.select("decode", None, None, None, None, false).unwrap();
            id
        };
        let qwen = |width, measured| {
            role(
                "Qwen2ForCausalLM",
                json!({
                    "use_mla":false,"hf_overrides":{"head_dim":width}
                }),
                measured,
            )
        };
        for measured in [None, Some((4, 64))] {
            // The checkpoint's 512 hidden channels / 8 query heads give both physical caches
            // width 64. NIXL hashes the selected model converter's explicit 64 or 128.
            let a = qwen(64, measured);
            assert!(a.mismatch(&qwen(64, None)).is_none());
            assert_eq!(
                a.mismatch(&qwen(128, Some((4, 64)))).unwrap().0,
                "head_size"
            );
        }
        let nemotron = |query, kv, dim| {
            role(
                "NemotronHForCausalLM",
                json!({"hf_overrides":{
                    "num_attention_heads":query,"num_key_value_heads":kv,"head_dim":dim
                }}),
                None,
            )
        };
        let a = nemotron(8, 4, 128);
        let b = nemotron(16, 8, 64);
        // Projection shapes and cache page bytes agree. At TP2 this is 2 x 128 versus 4 x 64.
        assert!(a.mismatch(&nemotron(8, 4, 128)).is_none());
        assert!(a.mismatch(&b).is_some());
        assert!(b.mismatch(&a).is_some());
        let sparse = |observed| role("NemotronHForCausalLM", json!({}), observed);
        assert!(a.mismatch(&sparse(None)).is_none());
        assert!(a.mismatch(&sparse(Some((2, 128)))).is_none());
        assert_eq!(
            a.mismatch(&sparse(Some((4, 64)))).unwrap().0,
            "num_kv_heads"
        );
        assert_eq!(
            sparse(Some((4, 64))).mismatch(&a).unwrap().0,
            "num_kv_heads"
        );
    }

    #[test]
    fn converter_and_constructor_read_their_own_hf_configuration() {
        let configured = |arch, model: Value| {
            TransferMetadata::configured(
                &json!({"model_config":model}),
                None,
                Some(arch),
                &Value::Null,
            )
        };
        let arch = "NemotronHForCausalLM";
        let settings = json!({"hf_config":{"model_type":"nemotron_h","head_dim":128,"num_key_value_heads":4,
            "text_config":{"model_type":"qwen2","n_head_kv":8,"head_dim":96,"num_hidden_layers":16}},
            "hf_overrides":{"num_key_value_heads":16,"head_dim":32}});
        let facts = configured(arch, settings.clone());
        assert_eq!(facts.model_heads.num_kv_heads, Some(8));
        assert_eq!(facts.model_heads.head_size, Some(96));
        assert_eq!(facts.num_hidden_layers, Some(16));
        let rank = facts
            .uniform_attention
            .unwrap()
            .rank_heads(2, Some(arch))
            .unwrap();
        assert_eq!(rank.num_kv_heads, Some(2));
        assert_eq!(rank.head_size, Some(128));
        let mut resolved = settings;
        resolved["model_arch_config"] = json!({"head_size":112,"total_num_kv_heads":12});
        let direct = configured(arch, resolved);
        assert_eq!(direct.model_heads.head_size, Some(112));
        assert_eq!(direct.model_heads.num_kv_heads, Some(12));

        // Rank heads use num_key_value_heads. The ordinary converter's higher-priority
        // KV aliases and head-size fallback can remain on an unobserved checkpoint.
        let sparse = |hf| configured("Qwen2ForCausalLM", json!({"hf_overrides":hf}));
        assert!(
            sparse(json!({"num_key_value_heads":4,"hidden_size_per_head":64}))
                .model_heads
                .is_empty()
        );
        let explicit = sparse(json!({"n_head_kv":8,"head_dim":null,"hidden_size_per_head":96}));
        assert_eq!(explicit.model_heads.num_kv_heads, Some(8));
        assert_eq!(explicit.model_heads.head_size, Some(96));
        for family in ["falcon", "RefinedWeb", "RefinedWebModel"] {
            let falcon = configured(
                "FalconForCausalLM",
                json!({"hf_overrides":{"model_type":family,
                "new_decoder_architecture":false,"multi_query":true,"n_head_kv":8}}),
            );
            assert_eq!(falcon.model_heads.num_kv_heads, Some(1));
        }
        let base = configured(
            "OtherArchitecture",
            json!({"hf_overrides":{"model_type":"gpt_neox","n_head_kv":8,"head_dim":96}}),
        );
        assert_eq!(base.model_heads.num_kv_heads, Some(8));
        assert_eq!(base.model_heads.head_size, Some(96));
        assert!(base.uniform_attention.is_none());
        let mla = configured(
            "DeepseekV3ForCausalLM",
            json!({"use_mla":false,"hf_overrides":{
            "model_type":"deepseek_v3","kv_lora_rank":512,"head_dim":128}}),
        );
        assert_eq!(mla.model_heads.head_size, None);
        let unidentified = configured(
            "OtherArchitecture",
            json!({"hf_text_config":{"model_type":"qwen2","head_dim":96}}),
        );
        assert!(unidentified.model_heads.is_empty());
    }

    #[test]
    fn source_scope_and_native_architecture_order_survive_compaction() {
        use sha2::{Digest, Sha256};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let outer = json!({"model_type":"qwen2_vl","architectures":["Qwen2VLForConditionalGeneration"],
                          "text_config":{"model_type":"qwen2_vl_text","architectures":["Qwen2ForCausalLM"],"num_key_value_heads":8}}).to_string();
        std::fs::write(&path, &outer).unwrap();
        let source = json!({"launch":{"model_config":{"path":path, "sha256": format!("{:x}", Sha256::digest(outer.as_bytes()))}}});
        let settings = |overrides| json!({"model_config":{"hf_overrides":overrides}});
        let arch = Some("Qwen2ForCausalLM");
        let project = |v| TransferMetadata::configured(&settings(v), Some(false), arch, &source);
        assert_eq!(
            project(json!({"head_dim":96})).model_heads.head_size,
            Some(128)
        );
        assert_eq!(
            project(json!({"text_config":{"head_dim":128}}))
                .model_heads
                .head_size,
            Some(128)
        );
        let local = project(json!({"num_key_value_heads":2,"text_config":{"hidden_size":512,"num_attention_heads":8}}))
            .uniform_attention.unwrap().rank_heads(2, arch).unwrap();
        assert_eq!((local.num_kv_heads, local.head_size), (Some(4), Some(64)));
        std::fs::write(&path, r#"{"model_type":"qwen2"}"#).unwrap();
        assert_eq!(project(json!({"head_dim":128})).model_heads.head_size, None);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(project(json!({"head_dim":128})).model_heads.head_size, None);

        let base = json!({"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","n_head_kv":4,
            "head_dim":64,"num_key_value_heads":8,"num_attention_heads":8,"hidden_size":512,"num_hidden_layers":12}).to_string();
        std::fs::write(&path, &base).unwrap();
        let source = json!({"launch":{"model_config":{"path":path,"sha256":format!("{:x}",Sha256::digest(base.as_bytes()))}}});
        let read = |overrides| {
            TransferMetadata::configured(&settings(overrides), Some(false), arch, &source)
        };
        let a = read(json!({"n_head_kv":8,"head_dim":128}));
        let b = read(json!({"num_kv_heads":8,"hidden_size_per_head":128}));
        assert_eq!(a.model_heads.num_kv_heads, Some(8));
        assert_eq!(b.model_heads.num_kv_heads, Some(4));
        assert_eq!(b.model_heads.head_size, Some(64));
        assert_eq!(b.num_hidden_layers, Some(12));
        // vLLM selects the registered RWConfig before loading this checkpoint.
        assert_eq!(
            read(json!({"model_type":"RefinedWeb"})).num_hidden_layers,
            None
        );
        assert_eq!(
            read(json!({"n_head_kv":null})).model_heads.num_kv_heads,
            None
        );
        assert_eq!(
            read(json!({"head_dim":null,"hidden_size_per_head":96}))
                .model_heads
                .head_size,
            Some(96)
        );

        let effective = json!({"resolved_config":settings(json!({"architectures":["Qwen2ForCausalLM","LlamaForCausalLM"],
                           "hidden_size":512,"num_attention_heads":8,"num_key_value_heads":8,"head_dim":128}))});
        let projected =
            NixlIdentity::compact_workers(&[effective], Some("LlamaForCausalLM")).unwrap();
        let worker: Worker = serde_json::from_value(projected[0].clone()).unwrap();
        assert_eq!(worker.metadata.model_heads.head_size, Some(128));
        let physical = worker
            .metadata
            .uniform_attention
            .unwrap()
            .for_selected(Some("LlamaForCausalLM"))
            .unwrap();
        assert_eq!(
            physical
                .rank_heads(2, Some("LlamaForCausalLM"))
                .unwrap()
                .head_size,
            Some(64)
        );
    }

    #[test]
    fn qwen_vl_initializes_the_text_model_before_serving_overrides() {
        use sha2::{Digest, Sha256};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        for (family, architecture) in [
            ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
            ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
        ] {
            // A sparse draft can retain the ordered names and an outer scalar alone.
            let sparse = |names: &[&str]| {
                let settings =
                    json!({"model_config":{"hf_overrides":{"architectures":names,"head_dim":96}}});
                TransferMetadata::configured(
                    &settings,
                    Some(false),
                    Some(architecture),
                    &Value::Null,
                )
                .model_heads
                .head_size
            };
            assert_eq!(sparse(&["UnregisteredModel", architecture]), None);
            assert_eq!(sparse(&["Qwen2ForCausalLM", architecture]), Some(96));
            let flat = json!({"model_type":family,"architectures":[architecture],
                "hidden_size":1536,"num_attention_heads":12,"num_key_value_heads":2,
                "num_hidden_layers":28,"head_dim":192,"n_head_kv":1});
            let nested = json!({"hidden_size":512,"num_attention_heads":8,"num_key_value_heads":8});
            let mut typed = nested.clone();
            typed["model_type"] = json!(format!("{family}_text"));
            for (text, kv_heads, width) in
                [(None, 2, 128), (Some(nested), 8, 64), (Some(typed), 8, 64)]
            {
                let mut checkpoint = flat.clone();
                if let Some(text) = text {
                    checkpoint["text_config"] = text;
                }
                let data = serde_json::to_vec(&checkpoint).unwrap();
                std::fs::write(&path, &data).unwrap();
                let source = json!({"launch":{"model_config":{"path":path,"sha256":format!("{:x}",Sha256::digest(&data))}}});
                let facts = |overrides| {
                    TransferMetadata::configured(
                        &json!({"model_config":{"hf_overrides":overrides}}),
                        Some(false),
                        Some(architecture),
                        &source,
                    )
                };
                let observed = |overrides, hash_heads, hash_size| {
                    let projected = facts(overrides);
                    assert_eq!(projected.model_heads.num_kv_heads, Some(hash_heads));
                    assert_eq!(projected.model_heads.head_size, Some(hash_size));
                    let rank = projected
                        .uniform_attention
                        .unwrap()
                        .rank_heads(2, Some(architecture))
                        .unwrap();
                    assert_eq!(
                        (rank.num_kv_heads, rank.head_size),
                        (Some(kv_heads / 2), Some(width))
                    );
                };
                observed(json!({}), kv_heads, width);
                observed(
                    json!({"head_dim":256,"num_key_value_heads":16}),
                    kv_heads,
                    width,
                );
                observed(json!({"text_config":{"head_dim":96,"n_head_kv":4}}), 4, 96);
                observed(
                    json!({"text_config":{"head_dim":null,"hidden_size_per_head":112}}),
                    kv_heads,
                    112,
                );
            }
        }
    }

    #[test]
    fn configured_heads_and_attention_owners_must_match_the_selected_role() {
        let qwen = |heads| {
            let mut id = NixlIdentity {
                architecture: Some("Qwen2ForCausalLM".into()), tp_size: Some(2),
                metadata: serde_json::from_value(json!({
                    "uniform_attention":{"architecture":"Qwen2ForCausalLM","ordered_architecture":true,
                                         "num_kv_heads":heads,"divided_head_size":64},
                    "layer_heads":{"model.layers.0.self_attn.attn":{"num_kv_heads":4,"head_size":64}}
                })).unwrap(), ..Default::default()};
            id.select("prefill", None, None, None, None, false)
        };
        assert!(qwen(8).is_ok());
        assert!(qwen(4).is_err());

        let nemotron = |second, layout, measured: &[usize]| {
            let mut id = NixlIdentity {
                architecture: Some("NemotronHForCausalLM".into()),
                metadata: TransferMetadata {
                    nemotron_attention_layers: Some(vec![0, second]),
                    nemotron_state_regions: Some(layout::LayerRegions::exact(BTreeSet::new())),
                    registered_layers: (!measured.is_empty()).then(|| {
                        measured
                            .iter()
                            .map(|i| format!("model.layers.{i}.mixer.attn"))
                            .collect()
                    }),
                    ..Default::default()
                },
                recorded_groups: Some(vec![json!({"kind":"attention","num_layers":2})]),
                workers: vec![Worker {
                    gpu_cache_layout: Layouts::named(layout),
                    ..Default::default()
                }],
                ..Default::default()
            };
            id.select("prefill", None, None, None, None, false)?;
            Ok::<_, AicError>(id)
        };
        assert!(nemotron(1, "LBHNC", &[0, 2]).is_err());
        let layer_major = nemotron(1, "LBHNC", &[]).unwrap();
        assert!(
            layer_major
                .mismatch(&nemotron(1, "LBHNC", &[0, 1]).unwrap())
                .is_none()
        );
        assert!(
            layer_major
                .mismatch(&nemotron(2, "LBHNC", &[]).unwrap())
                .is_some()
        );
        let block_major = nemotron(1, "BLHNC", &[]).unwrap();
        assert!(
            block_major
                .mismatch(&nemotron(2, "BLHNC", &[]).unwrap())
                .is_none()
        );
    }

    #[test]
    fn raw_scheduler_limits_are_bound_to_the_launch_and_rank_records() {
        let settings = json!({"scheduler_config":{"max_num_seqs":8,"max_num_batched_tokens":128},
                              "model_config":{"max_model_len":1024}});
        let launch = json!({"collection":{"max_num_seqs":8,"max_num_batched_tokens":128,"max_model_len":1024}});
        let raw = json!({"source":"runtime_observations","launch":launch,"runtime_settings":settings,
                         "artifacts":[{"evidence":{"kind":"worker","launch":launch,"resolved_config":settings}}]});
        let validate = |entry: &Value| {
            roster::record_cell_memory(&Value::Null, entry, &Value::Null, &mut json!({}))
        };
        assert!(validate(&raw).is_ok());
        for path in [
            "/launch/collection/max_num_batched_tokens",
            "/artifacts/0/evidence/resolved_config/scheduler_config/max_num_seqs",
            "/artifacts/0/evidence/launch/collection/max_model_len",
        ] {
            let mut changed = raw.clone();
            *changed.pointer_mut(path).unwrap() = json!(1);
            assert!(validate(&changed).is_err());
        }
    }

    #[test]
    fn ordered_native_fallback_depends_on_the_first_transformers_class() {
        let select = |first, stock| {
            registry::select(
                [Some(first), Some("NemotronHForConditionalGeneration")].into_iter(),
                Some("0.32.0"),
                false,
                stock,
            )
        };
        assert!(
            select("UnregisteredGenerationName", true).and_then(registry::Registration::decoder)
                == Some(Decoder::NemotronH)
        );
        assert!(select("UnregisteredGenerationName", false).is_none());
        assert!(select("AlbertModel", true).is_none());
        assert!(
            select("ReformerModel", true).and_then(registry::Registration::decoder)
                == Some(Decoder::NemotronH)
        );
    }

    #[test]
    fn authenticated_decoder_constructor_precedes_recorded_updates() {
        use sha2::{Digest, Sha256};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        for (family, arch, absent_kv, absent_width) in [
            ("qwen2", "Qwen2ForCausalLM", 32, 64),
            ("qwen3", "Qwen3ForCausalLM", 32, 128),
            ("llama", "LlamaForCausalLM", 16, 64),
            ("mistral", "MistralForCausalLM", 8, 64),
            ("nemotron_h", "NemotronHForCausalLM", 8, 128),
            ("falcon", "FalconForCausalLM", 16, 64),
        ] {
            let kv_field = if family == "falcon" {
                "num_kv_heads"
            } else {
                "num_key_value_heads"
            };
            for null in [false, true] {
                // NemotronH rejects both null fields. Mistral accepts a null width only.
                if null && family == "nemotron_h" {
                    continue;
                }
                let mut config = json!({"model_type":family,"architectures":[arch],
                    "num_attention_heads":16,"hidden_size":1024});
                if null {
                    if family != "mistral" {
                        config[kv_field] = Value::Null;
                    }
                    if matches!(family, "llama" | "mistral") {
                        config["head_dim"] = Value::Null;
                    }
                }
                if family == "falcon" {
                    config["new_decoder_architecture"] = json!(true);
                    config["hidden_size"] = json!(512);
                    config["n_embed"] = json!(1024);
                }
                let data = serde_json::to_vec(&config).unwrap();
                std::fs::write(&path, &data).unwrap();
                let source = json!({"launch":{"model_config":{"path":path,"sha256":format!("{:x}",Sha256::digest(&data))}}});
                let project = |overrides| {
                    TransferMetadata::configured(
                        &json!({"model_config":{"hf_overrides":overrides}}),
                        Some(false),
                        Some(arch),
                        &source,
                    )
                };
                let expected_kv = if null && family != "mistral" {
                    16
                } else {
                    absent_kv
                };
                let observed = |overrides, hash_heads, hash_width, rank_width| {
                    let facts = project(overrides);
                    let rank = facts
                        .uniform_attention
                        .unwrap()
                        .rank_heads(2, Some(arch))
                        .unwrap();
                    assert_eq!(
                        (facts.model_heads.num_kv_heads, facts.model_heads.head_size),
                        (Some(hash_heads), Some(hash_width)),
                        "{family} null={null}"
                    );
                    assert_eq!(
                        (rank.num_kv_heads, rank.head_size),
                        (Some(expected_kv / 2), Some(rank_width)),
                        "{family} null={null}"
                    );
                };
                observed(json!({}), expected_kv, absent_width, absent_width);
                // n_head_kv changes the connector hash while the actual cache stays the same.
                let hash_width = if family == "qwen2" { 96 } else { absent_width };
                observed(
                    json!({"n_head_kv":4,"hidden_size_per_head":96}),
                    4,
                    hash_width,
                    absent_width,
                );
                assert_eq!(
                    project(json!({kv_field:null})).model_heads.num_kv_heads,
                    None
                );
                if matches!(family, "llama" | "mistral") {
                    observed(
                        json!({"head_dim":null,"hidden_size_per_head":96}),
                        expected_kv,
                        96,
                        64,
                    );
                }
                if family == "falcon" {
                    observed(json!({"n_embed":2048}), expected_kv, 64, 64);
                    assert_eq!(
                        project(json!({"new_decoder_architecture":null}))
                            .model_heads
                            .num_kv_heads,
                        Some(1)
                    );
                    assert_eq!(
                        project(json!({"new_decoder_architecture":null,"multi_query":null}))
                            .model_heads
                            .num_kv_heads,
                        Some(expected_kv)
                    );
                }
                let updated_width = if matches!(family, "qwen2" | "falcon") {
                    128
                } else {
                    absent_width
                };
                for update in [
                    json!({"hidden_size":2048}),
                    json!({"num_attention_heads":8}),
                ] {
                    observed(update, expected_kv, updated_width, updated_width);
                }
            }
        }
    }

    #[test]
    fn serialized_falcon_config_omits_its_computed_width() {
        let mut config = json!({"model_type":"falcon","num_attention_heads":8,"hidden_size":512,"hidden_size_per_head":128});
        let width = |config: &Value| {
            TransferMetadata::configured(
                &json!({"model_config":{"hf_config":config}}),
                Some(false),
                Some("FalconForCausalLM"),
                &Value::Null,
            )
            .model_heads
            .head_size
        };
        assert_eq!(width(&config), Some(64));
        config["new_decoder_architecture"] = Value::Null;
        config["multi_query"] = json!(true);
        let saved = json!({"model_config":{"hf_config":config}});
        assert_eq!(
            TransferMetadata::configured(
                &saved,
                Some(false),
                Some("FalconForCausalLM"),
                &Value::Null
            )
            .model_heads
            .num_kv_heads,
            Some(1)
        );
        config["auto_map"] = json!({"AutoConfig":"custom.FalconConfig"});
        assert_eq!(width(&config), Some(128));
        config["head_dim"] = json!(96);
        assert_eq!(width(&config), Some(96));
    }

    #[test]
    fn verified_nemotron_layers_drive_both_positions_and_count() {
        use sha2::{Digest, Sha256};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let data = br#"{"model_type":"nemotron_h","architectures":["NemotronHForCausalLM"],"hybrid_override_pattern":"**--","num_hidden_layers":4}"#;
        std::fs::write(&path, data).unwrap();
        let source = json!({"launch":{"model_config":{"path":path,"sha256":format!("{:x}",Sha256::digest(data))}}});
        let project = |overrides| {
            TransferMetadata::configured(
                &json!({"model_config":{"hf_overrides":overrides}}),
                None,
                Some("NemotronHForCausalLM"),
                &source,
            )
        };
        let baseline = project(json!({"num_hidden_layers":5}));
        assert_eq!(
            (
                baseline.nemotron_attention_layers,
                baseline.num_hidden_layers
            ),
            (Some(vec![0, 1]), Some(4))
        );
        for alias in ["layer_types", "layers_block_type"] {
            let changed = project(json!({alias:["mlp","mlp","full_attention","full_attention"]}));
            assert_eq!(
                (changed.nemotron_attention_layers, changed.num_hidden_layers),
                (Some(vec![2, 3]), Some(4))
            );
        }
    }

    #[test]
    fn nemotron_regions_require_the_first_registered_generation_class() {
        let nemotron = "NemotronHForCausalLM";
        let foreign = "Qwen2MoeForCausalLM";
        let regions = |version, names: &[&str], model_impl| {
            let model = json!({"hf_overrides":{"model_type":"nemotron_h","architectures":names,
                "hybrid_override_pattern":"**--"}, "model_impl":model_impl});
            TransferMetadata::configured(
                &json!({"model_config":model}),
                None,
                Some(nemotron),
                &json!({"runtime":{"version":version}}),
            )
            .nemotron_attention_layers
        };
        for version in ["0.31.0", "0.32.0", "0.33.0"] {
            for names in [
                vec!["NemotronHModel", nemotron],
                vec!["UnknownModel", nemotron],
                vec![nemotron, foreign],
                vec!["NemotronHPuzzleForCausalLM", foreign],
                vec![nemotron, nemotron],
            ] {
                assert_eq!(regions(version, &names, "auto"), Some(vec![0, 1]));
            }
            assert_eq!(regions(version, &[foreign, nemotron], "auto"), None);
            assert_eq!(regions(version, &[nemotron], "transformers"), None);
            assert_eq!(
                regions(
                    version,
                    &["NemotronHForConditionalGeneration", foreign],
                    "auto"
                ),
                Some(vec![0, 1])
            );
            assert_eq!(
                regions(version, &["NemotronHForConditionalGeneration"], "vllm"),
                Some(vec![0, 1])
            );
        }
        let normalized = |auto_map: Value| {
            let model = json!({"hf_config":{"model_type":"qwen2","architectures":["Qwen2LMHeadModel"],
                "auto_map":auto_map,"num_attention_heads":8,"num_key_value_heads":4,"hidden_size":512}});
            TransferMetadata::configured(
                &json!({"model_config":model}),
                Some(false),
                None,
                &Value::Null,
            )
            .uniform_attention
        };
        assert!(normalized(Value::Null).is_some());
        assert!(normalized(json!({"AutoModel":"custom.Model"})).is_none());
        let names = ["DomynEdgeForCausalLM", nemotron];
        assert_eq!(regions("0.31.0", &names, "auto"), Some(vec![0, 1]));
        assert_eq!(regions("0.32.0", &names, "auto"), Some(vec![0, 1]));
        assert_eq!(regions("0.33.0", &names, "auto"), None);
        assert_eq!(regions("", &names, "auto"), None);
    }

    #[test]
    fn nemotron_state_registration_respects_aliased_mixers() {
        let role = |pattern| {
            let fields = json!({"hf_overrides":{"architectures":["NemotronHForCausalLM","LlamaForCausalLM"],
              "hybrid_override_pattern":pattern}});
            let arch = "NemotronHForCausalLM";
            let mut identity = NixlIdentity::observed(
                None,
                &json!({"runtime_settings":{"model_config":fields}}).to_string(),
                None,
                Some(2),
                Some(arch),
            )
            .unwrap();
            identity
                .select("prefill", None, None, None, None, false)
                .unwrap();
            identity
        };
        let a = role("MMMM-**");
        assert!(a.mismatch(&role("MMM-M**")).is_none());
        assert_eq!(
            a.mismatch(&role("-MMMM**")).unwrap().0,
            "Mamba layer positions"
        );
        assert_eq!(
            role("-MMMM**").mismatch(&a).unwrap().0,
            "Mamba layer positions"
        );
    }

    #[test]
    fn compact_uniform_attention_compares_with_measured_logical_heads_in_either_direction() {
        let config = |family, width| {
            json!({"model_config":{"use_mla":false,"hf_overrides":{
            "model_type":family,"num_attention_heads":8,"hidden_size":1024,"head_dim":width,"num_key_value_heads":4}}})
        };
        for (family, arch, local_width) in [
            ("qwen2", "Qwen2ForCausalLM", 128),
            ("qwen3", "Qwen3ForCausalLM", 64),
        ] {
            let mut compact = observe(json!({"runtime_settings":config(family,64)}), "prefill");
            compact.architecture = Some(arch.to_owned());
            compact.tp_size = Some(8);
            let mut other = config(family, 32);
            other["model_config"]["hf_overrides"]["hidden_size"] = json!(512);
            let mut peer = observe(json!({"runtime_settings":other}), "decode");
            peer.architecture = Some(arch.to_owned());
            peer.tp_size = Some(8);
            assert_eq!(compact.mismatch(&peer).unwrap().0, "head_size");
            assert_eq!(peer.mismatch(&compact).unwrap().0, "head_size");
            assert!(compact.mismatch(&compact).is_none());
            let mut measured = NixlIdentity {
                tp_size: Some(8),
                ..Default::default()
            };
            let heads = |dim| HeadGeometry {
                num_kv_heads: Some(1),
                head_size: Some(dim),
                content_size: Some(512),
            };
            measured
                .metadata
                .layer_heads
                .insert("attention".into(), heads(local_width));
            assert!(compact.mismatch(&measured).is_none());
            measured
                .metadata
                .layer_heads
                .insert("attention".into(), heads(local_width * 2));
            assert_eq!(compact.mismatch(&measured).unwrap().0, "head_size");
            assert_eq!(measured.mismatch(&compact).unwrap().0, "head_size");
            compact.architecture = Some("CustomModel".into());
            assert!(compact.mismatch(&measured).is_none());
        }
    }

    #[test]
    fn selected_groups_preserve_observed_geometry_and_allow_unmeasured_transfer_bytes() {
        let group =
            json!({"kind":"attention","num_layers":1,"block_size_tokens":64,"page_size_bytes":256});
        let mut id = NixlIdentity::default();
        id.normalized_groups(&json!([group])).unwrap();
        let mut selected = group.clone();
        selected["transfer_bytes_per_block"] = json!(128);
        id.selected_groups(&json!([selected])).unwrap();
        let cache = json!({"groups":[{"kind":"attention","block_size_tokens":64,"layer_names":["a"],
            "spec_page_size_bytes":128}],"allocated_cache_bytes":1024,"num_blocks":4});
        let mut raw = NixlIdentity::default();
        raw.worker_groups(&cache, None).unwrap();
        raw.selected_groups(&json!([selected])).unwrap();
        for (field, value) in [
            ("block_size_tokens", 128),
            ("num_layers", 2),
            ("page_size_bytes", 512),
            ("sliding_window", 32),
        ] {
            let mut changed = selected.clone();
            changed[field] = json!(value);
            assert!(raw.selected_groups(&json!([changed])).is_err());
            assert!(
                id.selected_groups(&json!([changed]))
                    .unwrap_err()
                    .to_string()
                    .contains(field)
            );
        }
    }

    #[test]
    fn explicit_push_requires_a_common_layout_across_peers() {
        let worker = |connector, layout, permute, role| {
            observe(
                json!({"artifacts":[{"evidence":{
                    "kind":"worker","phase":role,"tp_rank":0,"dp_rank":0,"raw_cache_config":{"kv_cache_layout":layout},
                    "resolved_config":{"model_config":{"use_mla":false},"kv_transfer_config":{
                        "kv_connector":connector,"enable_permute_local_kv":permute,"kv_buffer_device":"cuda"}}
                }}]}),
                role,
            )
        };
        for (connector, can_permute) in [
            ("NixlConnector", true),
            ("NixlPullConnector", true),
            ("NixlPushConnector", false),
        ] {
            let p = worker(connector, "LBHNC", false, "prefill");
            assert!(
                p.mismatch(&worker(connector, "LBHNC", false, "decode"))
                    .is_none()
            );
            assert_eq!(
                p.mismatch(&worker(connector, "LBNHC", true, "decode"))
                    .is_none(),
                can_permute
            );
        }
    }

    #[test]
    fn mla_push_compares_physical_orders_after_removing_the_singleton_head() {
        let worker = |name, compact, flag: Option<bool>, heads: Option<u64>| {
            let row = json!({"kind":"worker","phase":"decode","tp_rank":0,"dp_rank":0,
                "raw_cache_config":{"kv_cache_layout":name},
                "cache":{"groups":[{"kind":"attention","layer_names":["a"],"spec_type":"vllm.v1.kv_cache_interface.MLAAttentionSpec","num_kv_heads":heads}]},
                "resolved_config":{"model_config":{"use_mla":flag},"kv_transfer_config":{"kv_connector":"NixlPushConnector"}}});
            let evidence = if compact {
                json!({"worker_observations":NixlIdentity::compact_workers(&[row], None).unwrap()})
            } else {
                json!({"artifacts":[{"evidence":row}]})
            };
            observe(evidence, "decode")
        };
        for compact in [false, true] {
            let a = worker("LBHNC", compact, Some(true), None);
            for (flag, heads) in [(Some(true), None), (None, Some(1))] {
                let b = worker("LBNHC", !compact, flag, heads);
                assert!(a.mismatch(&b).is_none() && b.mismatch(&a).is_none());
                assert_eq!(
                    a.mismatch(&worker("BLHNC", !compact, flag, heads))
                        .unwrap()
                        .0,
                    "kv_cache_layout"
                );
            }
            for heads in [None, Some(2)] {
                assert_eq!(
                    a.mismatch(&worker("LBNHC", !compact, None, heads))
                        .unwrap()
                        .0,
                    "kv_cache_layout"
                );
            }
        }
    }

    #[test]
    fn normalized_uniform_mla_specs_identify_the_singleton_head_for_push() {
        let regions = |layout, raw_heads| {
            let uniform = "vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs";
            let mla = "vllm.v1.kv_cache_interface.MLAAttentionSpec";
            let names = ["latent", "indexer"];
            let raw = json!({"spec_type":uniform,"layer_names":names,"num_kv_heads":raw_heads});
            let group = json!({"kind":"attention","spec_type":uniform,"layer_names":names,
                "layer_specs":{"latent":{"spec_type":mla,"num_kv_heads":1},"indexer":{"spec_type":mla,"num_kv_heads":1}}});
            let row = json!({"kind":"worker","phase":"decode","tp_rank":0,"dp_rank":0,
                "raw_cache_config":{"kv_cache_layout":layout,"groups":[raw]}, "cache":{"groups":[group]},
                "resolved_config":{"model_config":{"use_mla":true},"kv_transfer_config":{"kv_connector":"NixlPushConnector"}}});
            observe(json!({"artifacts":[{"evidence":row}]}), "decode")
        };
        let outer = regions("BLHNC", None);
        assert!(outer.mismatch(&regions("BLNHC", None)).is_none());
        assert_eq!(
            outer.mismatch(&regions("BLNHC", Some(2))).unwrap().0,
            "kv_cache_layout"
        );
    }

    #[test]
    fn grouped_workers_preserve_their_initialized_layout_and_buffer_restrictions() {
        let wrap = |name| {
            json!({"kind":"state","spec_type":"vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs",
            "layer_specs":{"model.layer":{"spec_type":name}}})
        };
        let mut row = json!({"kind":"worker","phase":"decode","dp_rank":0,"tp_rank":0,
            "cache":{"groups":[wrap("vllm.v1.kv_cache_interface.CircularBufferSpec")]},
            "resolved_config":{"kv_transfer_config":{"kv_buffer_device":"cpu"},"cache_config":{"kv_cache_layout":"NHD"}}});
        assert_eq!(
            Worker::observed(&row, None).unwrap().gpu_cache_layout,
            Layouts::named("LBNHC")
        );
        let selected = |row: &Value| {
            let mut id = NixlIdentity::observed(
                None,
                &json!({"artifacts":[{"evidence":row}]}).to_string(),
                Some(1),
                Some(1),
                None,
            )?;
            id.select("decode", None, None, Some("0.33.0"), None, false)
        };
        assert!(selected(&row).is_err());
        row["resolved_config"]["kv_transfer_config"]["kv_buffer_device"] = json!("cuda");
        assert!(selected(&row).is_ok());
        row["cache"]["groups"] = json!([wrap("vllm.v1.kv_cache_interface.MambaSpec")]);
        assert!(selected(&row).is_ok());
        row["resolved_config"]["scheduler_config"] =
            json!({"disable_hybrid_kv_cache_manager":true});
        assert!(selected(&row).is_err());
        row["resolved_config"]["cache_config"]["kv_cache_layout"] = json!("HND");
        assert_eq!(
            Worker::observed(&row, None).unwrap().gpu_cache_layout,
            Layouts::named("LBHNC")
        );
        row["resolved_config"]["cache_config"]["kv_cache_layout"] = json!("LBLNC");
        assert!(Worker::observed(&row, None).is_err());
    }

    #[test]
    fn recorded_dp_coordinates_check_the_physical_layout_across_tp_ranks() {
        let archive = |phase: &str, ranks: &[&[Option<&str>]]| {
            let mut workers = Vec::new();
            let mut capacities = Vec::new();
            for (tp, engines) in ranks.iter().enumerate() {
                for (dp, name) in engines.iter().enumerate() {
                    workers.push(json!({"kind":"worker","phase":phase,"tp_rank":tp,"dp_rank":dp,
                        "raw_cache_config":{"kv_cache_layout":name},"resolved_config":{"model_config":{"use_mla":false}}}));
                    capacities.push(json!({"phase":phase,"tp_rank":tp,"dp_rank":dp}));
                }
            }
            workers.reverse();
            let portable = json!({"artifacts":workers.into_iter().map(|e|json!({"evidence":e})).collect::<Vec<_>>(),
                "rank_capacities":capacities});
            let mut id = NixlIdentity::observed(
                None,
                &portable.to_string(),
                Some(capacities.len() as u64),
                Some(ranks.len() as u64),
                None,
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
        assert!(archive("prefill", &[&[a, b, a], &[a, b, a]]).is_ok());
        assert!(archive("prefill", &[&[a, b, a], &[b, a, b]]).is_err());
        let mut large = [a, b].repeat(64);
        assert!(archive("prefill", &[&large]).is_ok());
        let row = &large[..24];
        let reordered = row.to_vec();
        assert!(archive("prefill", &[row, &reordered]).is_ok());
        large[0] = b;
        assert!(archive("prefill", &[&large[..24], &reordered]).is_err());
    }

    #[test]
    fn a_bound_receipt_cannot_be_read_as_an_original_receipt() {
        let probe = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v1","nixl_observation":{"worker_observations_sha256":"edited"}});
        let compact =
            json!({"observation_provenance":"source_references","worker_observation_digests":[]});
        let null_snapshot = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v1","nixl_observation":null});
        for record in [probe, compact, null_snapshot] {
            assert!(
                NixlIdentity::observed(None, &record.to_string(), Some(1), Some(1), None).is_err()
            );
        }
    }

    #[test]
    fn original_compact_cells_keep_initialized_layout_and_mamba_requirements() {
        let compact = |layout: &str, groups: Value| {
            let entry = json!({
                "source":"vllm_initialized_cache", "observer_schema_version":"v1",
                "runtime_settings":{"cache_config":{"kv_cache_layout":layout},"model_config":{"use_mla":false}},
                "artifacts":(["worker","scheduler"].map(|p| json!({"path":p,"sha256":"0".repeat(64)}))),
                "rank_capacities":[{"dp_rank":0,"tp_rank":0}]
            });
            json!({"observation_provenance":"source_references","cell_observations":[{
                "cache_groups":groups, "runtime_memory":{"provenance":entry.to_string()}
            }]})
        };
        let left = observe(compact("LBNHC", json!([])), "prefill");
        let right = observe(compact("LBHNC", json!([])), "decode");
        assert_eq!(left.mismatch(&right).unwrap().0, "kv_cache_layout");
        assert!(left.mismatch(&left).is_none());
        let disabled = observe(
            json!({"runtime_settings":{"scheduler_config":{
                "disable_hybrid_kv_cache_manager":true
            }}}),
            "prefill",
        );
        assert!(disabled.mismatch(&right).is_none());
        let window = json!([{"kind":"attention", "sliding_window":128}]);
        let decoder = observe(compact("LBNHC", window), "decode");
        let mut sender = compact("LBHNC", json!([]));
        assert!(
            observe(sender.clone(), "prefill")
                .mismatch(&decoder)
                .is_none()
        );
        sender["runtime_settings"] =
            json!({"scheduler_config":{"disable_hybrid_kv_cache_manager":false}});
        let sender = observe(sender, "prefill");
        assert_eq!(sender.mismatch(&decoder).unwrap().0, "kv_cache_layout");
        assert!(sender.mismatch(&right).is_none());
        for count in [0, 1] {
            let mamba = observe(
                compact(
                    "LBHNC",
                    json!([{"kind":"state","prefill_checkpoint_blocks":count}]),
                ),
                "decode",
            );
            assert!(mamba.workers.iter().all(|w| w.requires_hma));
            assert_eq!(
                disabled.mismatch(&mamba).unwrap().0,
                "disable_hybrid_kv_cache_manager"
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
            assert!(p.excludes_layout(d, Some("0.32.0"), false));
            assert!(!p.excludes_layout(p, Some("0.32.0"), false));
        }
        let p = worker(&["LBHNC", "BLHNC"]);
        let mut d = worker(&["LBNHC", "BLNHC"]);
        assert!(!p.excludes_layout(&d, Some("0.32.0"), false));
        assert!(d.excludes_layout(&p, Some("0.32.0"), false));
        d.permit_permute = Some(false);
        assert!(p.excludes_layout(&d, Some("0.32.0"), false));
        d.use_mla = Some(true);
        assert!(!p.excludes_layout(&d, Some("0.32.0"), false));
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
        let mut receiver = local("LBNHC");
        receiver.permit_permute = Some(true);
        for version in ["0.31.0", "0.32.0", "0.33.0"] {
            let native = local("LBNHC");
            let converted = local("LBHNC");
            assert!(layout::routable(
                &[&native],
                &[&receiver],
                Some(version),
                false
            ));
            assert!(layout::routable(
                &[&converted],
                &[&receiver],
                Some(version),
                false
            ));
            assert!(!layout::routable(
                &[&native, &converted],
                &[&receiver],
                Some(version),
                false
            ));
        }
        let a = local("BLHNC");
        let b = local("BLNHC");
        let mla = Worker {
            use_mla: Some(true),
            ..b.clone()
        };
        assert!(!layout::routable(&[], &[&a, &b], Some("0.32.0"), false));
        assert!(layout::routable(&[&a, &b], &[], Some("0.32.0"), false));
        assert!(!layout::routable(
            &[&p, &p],
            &[&a, &b],
            Some("0.32.0"),
            false
        ));
        assert!(!layout::routable(
            &[&a, &b],
            &[&p, &p],
            Some("0.32.0"),
            false
        ));
        assert!(layout::routable(
            &[&p, &p],
            &[&a, &mla],
            Some("0.32.0"),
            false
        ));
        assert!(layout::routable(
            &[&p, &p],
            &[&a, &p],
            Some("0.32.0"),
            false
        ));
        let remote = local("LBHNC");
        let permuted = local("LBNHC");
        assert!(!layout::routable(
            &[&remote, &permuted],
            &[&permuted, &permuted],
            Some("0.32.0"),
            false
        ));
        assert!(!layout::routable(
            &[&remote, &permuted],
            &[&remote, &permuted],
            Some("0.32.0"),
            false
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
        assert!(!p.excludes_layout(&d, Some("0.33.0"), false));
        assert!(d.excludes_layout(&p, Some("0.33.0"), false));
        assert!(!p.excludes_layout(&w("BLNHC"), Some("0.31.0"), false));
        assert!(p.excludes_layout(&w("BLNHC"), Some("0.32.0"), false));
        d.permit_permute = Some(false);
        assert!(p.excludes_layout(&d, Some("0.33.0"), false));
        d.permit_permute = Some(true);
        d.requires_hma = true;
        assert!(p.excludes_layout(&d, Some("0.33.0"), false));
        for mla in [Some(true), None] {
            d.use_mla = mla;
            assert!(!p.excludes_layout(&d, Some("0.33.0"), false));
        }
    }
}
