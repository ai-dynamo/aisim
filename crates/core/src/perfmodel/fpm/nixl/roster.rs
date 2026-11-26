// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{Capture, Worker};
use crate::AicError;

pub(super) fn invalid() -> AicError {
    AicError::InvalidEngineConfig("incomplete or conflicting grouped worker roster".into())
}

pub(super) fn cell_source(entry: &Value) -> Option<&str> {
    if entry["source"] == "vllm_initialized_cache" {
        entry["plan_sha256"].as_str()
    } else {
        entry
            .pointer("/observations_index/sha256")
            .and_then(Value::as_str)
    }
}

pub(super) fn sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit())
}

/// A finalized indexed collection records the formal index separately from the draft probe.
pub(super) fn validate_formal_index(cell: &Value, manifest: &Value) -> Result<(), AicError> {
    let Some(index) = cell.get("observations_index") else {
        return Ok(());
    };
    let invalid_index = || {
        AicError::InvalidEngineConfig(
            "conflicting formal observations index in grouped worker roster".into(),
        )
    };
    let compatibility = &manifest["runtime_compatibility"];
    for (field, key) in [
        ("observations_index", "path"),
        ("observations_sha256", "sha256"),
    ] {
        match compatibility.get(field) {
            Some(value) if *value == index[key] => {}
            None if manifest.get("source_artifacts").is_none() => {}
            _ => return Err(invalid_index()),
        }
    }
    if let Some(inventory) = manifest.get("source_artifacts") {
        let mut matches = inventory
            .as_array()
            .ok_or_else(invalid_index)?
            .iter()
            .filter(|file| file["path"] == index["path"]);
        if matches
            .next()
            .is_none_or(|file| file["sha256"] != index["sha256"])
            || matches.next().is_some()
        {
            return Err(invalid_index());
        }
    }
    Ok(())
}

pub(super) fn validate_probe(probe: &Value) -> Result<(), AicError> {
    let index = std::path::Path::new(
        probe["observations_index"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(invalid)?,
    );
    let sources = probe["source_artifacts"].as_array().ok_or_else(invalid)?;
    let resources = &probe["observed_resources"];
    if ["checkpoint", "configuration"]
        .iter()
        .any(|key| probe[key].as_str().is_none_or(str::is_empty))
        || ["model_metadata", "user_overrides"]
            .iter()
            .any(|key| !probe[key].is_object())
        || resources["runtime_memory"]["kv_cache_bytes"]
            .as_u64()
            .is_none_or(|n| n == 0)
        || resources["cache_groups"]
            .as_array()
            .is_none_or(Vec::is_empty)
        || sources.iter().any(|r| {
            r["path"].as_str().is_none_or(str::is_empty)
                || !r["sha256"].as_str().is_some_and(sha256)
        })
        || ![
            index.to_path_buf(),
            index.with_file_name("launch.json"),
            index.with_file_name("model-config.json"),
        ]
        .iter()
        .all(|path| {
            sources
                .iter()
                .any(|r| r["path"].as_str().map(std::path::Path::new) == Some(path.as_path()))
        })
    {
        return Err(invalid());
    }
    Ok(())
}

fn rank_capacity(entry: &Value) -> Result<Option<u64>, AicError> {
    let Some(rows) = entry["rank_capacities"].as_array() else {
        return Ok(None);
    };
    let mut minimum = None;
    let mut complete = true;
    for row in rows {
        let bytes = row["kv_cache_bytes"].as_u64();
        let allocated = row["allocated_cache_bytes"].as_u64();
        let page = row["page_size_bytes"].as_u64();
        for (total, blocks) in [(bytes, "initial_free_blocks"), (allocated, "num_blocks")] {
            if let (Some(total), Some(blocks), Some(page)) = (total, row[blocks].as_u64(), page)
                && blocks.checked_mul(page) != Some(total)
            {
                return Err(super::conflict("kv_cache_bytes"));
            }
        }
        if bytes.zip(allocated).is_some_and(|(b, a)| b > a) {
            return Err(super::conflict("kv_cache_bytes"));
        }
        if let Some(bytes) = bytes.filter(|&n| n > 0) {
            minimum = Some(minimum.map_or(bytes, |m: u64| m.min(bytes)));
        } else {
            complete = false;
        }
    }
    Ok(minimum.filter(|_| complete))
}

fn first_phase_settings(entry: &Value) -> Option<&Value> {
    let phases = entry["prefix_caching"].as_object()?;
    phases.get("prefill").or_else(|| phases.get("decode"))
}

pub(super) fn same_memory_setting(key: &str, lhs: &Value, rhs: &Value) -> bool {
    lhs == rhs
        || key == "gpu_memory_utilization"
            && lhs.as_f64().zip(rhs.as_f64()).is_some_and(|(a, b)| a == b)
}

fn record_memory_field(expected: &mut Value, memory: &Value, key: &str) -> Result<(), AicError> {
    record_field(expected, key, &memory[key])
}

fn record_field(expected: &mut Value, key: &str, value: &Value) -> Result<(), AicError> {
    if !value.is_null() {
        if expected
            .get(key)
            .is_some_and(|old| !same_memory_setting(key, old, value))
        {
            return Err(super::conflict(key));
        }
        expected[key] = value.clone();
    }
    Ok(())
}

fn record_memory_settings(expected: &mut Value, memory: &Value) -> Result<(), AicError> {
    for key in [
        "enable_prefix_caching",
        "hash_block_size_tokens",
        "mamba_cache_mode",
        "gpu_memory_utilization",
    ] {
        record_memory_field(expected, memory, key)?;
    }
    Ok(())
}

fn record_collection_settings(expected: &mut Value, entry: &Value) -> Result<(), AicError> {
    let limits = [
        ("max_num_tokens", "max_num_batched_tokens"),
        ("max_batch_size", "max_num_seqs"),
    ];
    let artifacts = entry["artifacts"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|r| &r["evidence"])
        .filter(|r| matches!(r["kind"].as_str(), Some("worker" | "scheduler")));
    for settings in [&entry["runtime_settings"], &entry["resolved_config"]]
        .into_iter()
        .chain(artifacts.clone().map(|r| &r["resolved_config"]))
    {
        record_memory_field(expected, &settings["model_config"], "max_model_len")?;
        record_memory_field(
            expected,
            &settings["cache_config"],
            "gpu_memory_utilization",
        )?;
        for (key, serving) in limits {
            record_field(expected, key, &settings["scheduler_config"][serving])?;
        }
    }
    for launch in std::iter::once(&entry["launch"]).chain(artifacts.map(|r| &r["launch"])) {
        let collection = &launch["collection"];
        record_memory_field(expected, collection, "max_model_len")?;
        record_memory_field(expected, collection, "gpu_memory_utilization")?;
        for (key, serving) in limits {
            record_field(expected, key, &collection[serving])?;
        }
    }
    Ok(())
}

fn record_resource_limits(expected: &mut Value, resources: &Value) -> Result<(), AicError> {
    record_memory_field(expected, &resources["runtime_memory"], "max_model_len")?;
    for key in ["max_num_tokens", "max_batch_size"] {
        record_memory_field(expected, resources, key)?;
    }
    Ok(())
}

pub(super) fn record_cell_memory(
    cell: &Value,
    entry: &Value,
    compatibility: &Value,
    expected: &mut Value,
) -> Result<Option<u64>, AicError> {
    let memory = &cell["runtime_memory"];
    let observed = rank_capacity(entry)?;
    let saved = memory["kv_cache_bytes"].as_u64();
    let formal = (entry["source"] == "runtime_observations")
        .then(|| compatibility["observed_capacity_bytes"].as_u64())
        .flatten();
    for (a, b) in [(observed, saved), (formal, observed.or(saved))] {
        if a.zip(b).is_some_and(|(a, b)| a != b) {
            return Err(super::conflict("kv_cache_bytes"));
        }
    }
    if formal
        .zip(compatibility["accepted_capacity_bytes"].as_u64())
        .is_some_and(|(seen, accepted)| seen < accepted)
    {
        return Err(super::conflict("kv_cache_bytes"));
    }
    record_memory_settings(expected, memory)?;
    record_resource_limits(expected, cell)?;
    record_collection_settings(expected, entry)?;
    if let Some(phase) = first_phase_settings(entry) {
        record_memory_settings(expected, phase)?;
    }
    Ok(observed.or(saved).or(formal))
}

/// The importer keeps the first collected phase's settings and the minimum usable rank
/// capacity. A registered draft can carry a separately labelled capacity override.
pub(super) fn selected_memory(evidence: &Value) -> Result<Value, AicError> {
    let mut expected = serde_json::json!({});
    let capacity = if matches!(
        evidence["schema_version"].as_str(),
        Some("aisimulate-onboarding-runtime-probe/v1" | "aisimulate-onboarding-runtime-probe/v2")
    ) {
        let resources = &evidence["observed_resources"];
        let memory = &resources["runtime_memory"];
        record_memory_settings(&mut expected, memory)?;
        record_resource_limits(&mut expected, resources)?;
        record_collection_settings(&mut expected, &evidence["nixl_observation"])?;
        let recorded = memory["kv_cache_bytes"].as_u64();
        if let Some(change) = evidence["user_overrides"].get("runtime_memory.kv_cache_bytes") {
            if change["observed"].as_u64() != recorded
                || change["source"] != "user override; assumption, not a runtime measurement"
            {
                return Err(super::conflict("kv_cache_bytes"));
            }
            Some(
                change["value"]
                    .as_u64()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| super::conflict("kv_cache_bytes"))?,
            )
        } else {
            recorded
        }
    } else if evidence["source"] == "bound_collection_phases"
        && let Some(phases) = evidence["phases"].as_object()
    {
        // The writer retains each phase's own rank capacities and chooses their minimum.
        let mut capacities = Vec::new();
        for entry in phases.values() {
            record_collection_settings(&mut expected, entry)?;
            capacities.extend(rank_capacity(entry)?);
        }
        if let Some(phase) = phases.get("prefill").or_else(|| phases.get("decode"))
            && let Some(settings) = first_phase_settings(phase)
        {
            record_memory_settings(&mut expected, settings)?;
        }
        (capacities.len() == phases.len())
            .then(|| capacities.into_iter().min())
            .flatten()
    } else if identified_source(evidence) {
        record_collection_settings(&mut expected, evidence)?;
        if let Some(phase) = first_phase_settings(evidence) {
            record_memory_settings(&mut expected, phase)?;
        }
        rank_capacity(evidence)?
    } else {
        None
    };
    if let Some(bytes) = capacity {
        expected["kv_cache_bytes"] = Value::from(bytes);
    }
    Ok(expected)
}

fn file_references(files: &Value, path_key: &str) -> Result<Value, AicError> {
    let mut sorted = BTreeMap::new();
    for file in files.as_array().ok_or_else(invalid)? {
        let path = file[path_key]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(invalid)?;
        if !file["sha256"].as_str().is_some_and(sha256)
            || file
                .get("kind")
                .is_some_and(|v| !v.is_null() && v.as_str().is_none_or(str::is_empty))
            || sorted.insert(path, file.clone()).is_some()
        {
            return Err(invalid());
        }
    }
    Ok(Value::Array(sorted.into_values().collect()))
}

/// Bind the entire portable observation and its recorded file associations. Only the order
/// of file references and worker/capacity rows is immaterial.
pub(crate) fn digest(
    entry: &Value,
    source: Option<&str>,
    inventory: &Value,
    draft: bool,
    cell: Option<&Value>,
) -> Result<String, AicError> {
    let formal = entry["source"] == "vllm_initialized_cache";
    let source = source
        .filter(|s| if formal { !s.is_empty() } else { sha256(s) })
        .ok_or_else(invalid)?;
    let mut observation = entry.as_object().cloned().ok_or_else(invalid)?;
    if draft {
        observation.remove("worker_observations_sha256");
    }
    for key in ["worker_observations", "rank_capacities"] {
        if let Some(rows) = observation.get_mut(key).and_then(Value::as_array_mut) {
            rows.sort_by_cached_key(Value::to_string);
        }
    }
    let mut files = serde_json::Map::new();
    if draft {
        let index = inventory["observations_index"]
            .as_str()
            .ok_or_else(invalid)?;
        let sources = inventory["source_artifacts"]
            .as_array()
            .ok_or_else(invalid)?;
        if !sources.iter().any(|r| {
            r["kind"] == "observations_index" && r["path"] == index && r["sha256"] == source
        }) || !["launch", "model_config"]
            .iter()
            .all(|kind| sources.iter().any(|r| r["kind"] == *kind))
        {
            return Err(invalid());
        }
        files.insert("observations_index".into(), Value::from(index));
    } else {
        if !formal {
            let index = &entry["observations_index"];
            if index["path"].as_str().is_none_or(str::is_empty) || index["sha256"] != source {
                return Err(invalid());
            }
        }
        if entry["artifacts"].as_array().is_none_or(Vec::is_empty) {
            return Err(invalid());
        }
        for key in ["artifacts", "runtime_artifacts", "launch_artifacts"] {
            if let Some(refs) = entry.get(key) {
                observation.insert(key.into(), file_references(refs, "path")?);
            }
        }
    }
    for (key, path_key) in [
        ("source_artifacts", "path"),
        ("formal_data", "relative_path"),
    ] {
        if let Some(refs) = inventory.get(key) {
            files.insert(key.into(), file_references(refs, path_key)?);
        }
    }
    if let Some(resources) = cell {
        let mut resources = resources.as_object().cloned().ok_or_else(invalid)?;
        resources.remove("provenance");
        resources
            .get_mut("runtime_memory")
            .and_then(Value::as_object_mut)
            .ok_or_else(invalid)?
            .remove("provenance");
        files.insert("resources".into(), Value::Object(resources));
    }
    let mut contents = serde_json::json!([source, observation, files]);
    contents.sort_all_objects();
    Ok(format!("{:x}", Sha256::digest(contents.to_string())))
}

/// Each multi-cell receipt also binds the complete set. Removing a cell and its own
/// receipt leaves every surviving receipt inconsistent, even when the source files are offline.
pub(super) fn bound_cell_digests(mut digests: Vec<String>) -> Result<Vec<String>, AicError> {
    digests.sort_unstable();
    if digests.is_empty()
        || digests.iter().any(|d| !sha256(d))
        || digests.windows(2).any(|w| w[0] == w[1])
    {
        return Err(invalid());
    }
    if digests.len() > 1 {
        let set = Sha256::digest(digests.concat());
        for digest in &mut digests {
            *digest = format!(
                "{:x}",
                Sha256::new()
                    .chain_update(set)
                    .chain_update(digest.as_bytes())
                    .finalize()
            );
        }
        digests.sort_unstable();
    }
    Ok(digests)
}

pub(super) fn identified_source(entry: &Value) -> bool {
    let formal = entry["source"] == "vllm_initialized_cache"
        && (entry["observer_schema_version"].is_string() || entry["plan_sha256"].is_string());
    let probe = entry["source"] == "runtime_observations"
        && (entry["observation_schema"] == "aisimulate-runtime-observation/v1"
            || entry.pointer("/launch/topology").is_some()
            || entry.pointer("/observations_index/sha256").is_some());
    formal || probe
}

/// A full capture retains raw files for every worker and scheduler. A historical compact cell
/// may legitimately retain only file references and capacities. Validate each against the
/// recorded and selected topology, including when just one independent roster is available.
pub(super) fn validate_source(
    entry: &Value,
    capture: Capture,
    selected_width: Option<u64>,
    selected_tp: Option<u64>,
) -> Result<bool, AicError> {
    if !identified_source(entry) {
        return Ok(false);
    }
    let parallel = &entry["runtime_settings"]["parallel_config"];
    let topology = &entry["launch"]["topology"];
    let axis = |saved: &str, initialized: &str| {
        let a = topology[saved].as_u64();
        let b = parallel[initialized].as_u64();
        if a.zip(b).is_some_and(|(x, y)| x != y) {
            Err(invalid())
        } else {
            Ok(a.or(b))
        }
    };
    let tp = axis("tp", "tensor_parallel_size")?;
    let width = tp
        .zip(axis("dp", "data_parallel_size")?)
        .and_then(|(t, d)| t.checked_mul(d));
    if tp.zip(selected_tp).is_some_and(|(a, b)| a != b)
        || width.zip(selected_width).is_some_and(|(a, b)| a != b)
    {
        return Err(invalid());
    }
    let tp = tp.or(selected_tp);
    let width = width.or(selected_width);
    let capacities = entry["rank_capacities"].as_array();
    if capture != Capture::Full {
        let capacities = capacities.ok_or_else(invalid)?;
        if capture == Capture::References {
            if entry["source"] == "runtime_observations" {
                let index = &entry["observations_index"];
                if index["path"].as_str().is_none_or(str::is_empty)
                    || !index["sha256"].as_str().is_some_and(sha256)
                {
                    return Err(invalid());
                }
            }
            for key in ["artifacts", "runtime_artifacts", "launch_artifacts"] {
                if let Some(refs) = entry.get(key) {
                    file_references(refs, "path")?;
                }
            }
            // Both original writers retained one observation per worker and one scheduler per DP pool.
            let pools = capacities
                .iter()
                .map(|r| (r["phase"].as_str(), r["dp_rank"].as_u64()))
                .collect::<BTreeSet<_>>();
            if entry["artifacts"]
                .as_array()
                .is_none_or(|files| files.len() != capacities.len() + pools.len())
            {
                return Err(invalid());
            }
        }
        if entry.get("worker_observations").is_none() {
            validate(entry, capacities, width, tp, None, true)?;
            if entry["worker_attention_backends"]
                .as_array()
                .is_some_and(|names| names.len() != capacities.len())
            {
                return Err(invalid());
            }
        }
        return Ok(true);
    }
    let files = entry["artifacts"].as_array().ok_or_else(invalid)?;
    if files.iter().any(|f| f["evidence"].is_null()) {
        return Err(invalid());
    }
    let workers: Vec<_> = files
        .iter()
        .map(|f| &f["evidence"])
        .filter(|r| r["kind"] == "worker")
        .collect();
    validate(
        entry,
        workers.iter().copied(),
        width,
        tp,
        capacities.map(Vec::as_slice),
        true,
    )?;
    let pool = |row: &Value| -> Result<_, AicError> {
        Ok((
            phase(entry, &row["phase"])?,
            row["dp_rank"].as_u64().ok_or_else(invalid)?,
        ))
    };
    let worker_pools: BTreeSet<_> = workers.iter().map(|r| pool(r)).collect::<Result<_, _>>()?;
    let mut schedulers = BTreeSet::new();
    for row in files
        .iter()
        .map(|f| &f["evidence"])
        .filter(|r| r["kind"] == "scheduler")
    {
        if !schedulers.insert(pool(row)?) {
            return Err(invalid());
        }
    }
    if schedulers != worker_pools {
        return Err(invalid());
    }
    Ok(true)
}

// The original formal memory collector measured one cell and omitted its phase name.
fn phase(entry: &Value, value: &Value) -> Result<&'static str, AicError> {
    match value.as_str() {
        Some("prefill") => Ok("prefill"),
        Some("decode") => Ok("decode"),
        None if value.is_null() && entry["source"] == "vllm_initialized_cache" => Ok(""),
        _ => Err(invalid()),
    }
}

/// The older compact projection retained TP and phase but omitted DP coordinates. Its per-TP
/// multiplicity is still known. Current projections retain all three coordinates.
pub(super) fn validate<'a>(
    entry: &Value,
    workers: impl IntoIterator<Item = &'a Value>,
    width: Option<u64>,
    tp: Option<u64>,
    capacities: Option<&[Value]>,
    require_dp: bool,
) -> Result<(), AicError> {
    let mut slots = BTreeMap::new();
    let mut full_slots = BTreeSet::new();
    let mut sizes = BTreeMap::new();
    for worker in workers {
        let phase = phase(entry, &worker["phase"])?;
        let rank = worker["tp_rank"].as_u64().ok_or_else(invalid)?;
        if tp.is_some_and(|t| rank >= t) {
            return Err(invalid());
        }
        *slots.entry((phase, rank)).or_insert(0_u64) += 1;
        *sizes.entry(phase).or_insert(0_u64) += 1;
        if require_dp || !worker["dp_rank"].is_null() {
            let dp = worker["dp_rank"].as_u64().ok_or_else(invalid)?;
            if !full_slots.insert((phase, rank, dp))
                || width.zip(tp).is_some_and(|(w, t)| t == 0 || dp >= w / t)
            {
                return Err(invalid());
            }
        }
    }
    if entry["source"] == "runtime_observations" {
        // The importer stores one phase-bound launch receipt per captured phase, including
        // for single-role imports. Some v0.33 phases also retained prefix-caching evidence.
        if entry["launch_artifacts"]
            .as_array()
            .is_some_and(|files| files.len() != sizes.len())
            || entry["prefix_caching"].as_object().is_some_and(|policies| {
                policies.keys().any(|p| {
                    matches!(p.as_str(), "prefill" | "decode") && !sizes.contains_key(p.as_str())
                })
            })
        {
            return Err(invalid());
        }
    }
    if slots.is_empty()
        || width.is_some_and(|w| sizes.values().any(|&n| n != w))
        || width.zip(tp).is_some_and(|(w, t)| {
            t == 0 || !w.is_multiple_of(t) || slots.values().any(|&n| n != w / t)
        })
    {
        return Err(invalid());
    }
    if let Some(capacities) = capacities {
        let mut expected = BTreeMap::new();
        let mut coordinates = BTreeSet::new();
        for row in capacities {
            let phase = phase(entry, &row["phase"])?;
            let rank = row["tp_rank"].as_u64().ok_or_else(invalid)?;
            let dp = row["dp_rank"].as_u64().ok_or_else(invalid)?;
            *expected.entry((phase, rank)).or_insert(0_u64) += 1;
            if !coordinates.insert((phase, rank, dp))
                || width.zip(tp).is_some_and(|(w, t)| t == 0 || dp >= w / t)
            {
                return Err(invalid());
            }
        }
        if expected != slots || !full_slots.is_subset(&coordinates) {
            return Err(invalid());
        }
    }
    Ok(())
}

/// An augmenting path lets a previously matched worker take a different peer. Every successful
/// iteration assigns one additional physical slot. Anonymous slots never acquire a guessed TP rank.
pub(super) fn can_pair<L, R>(
    prefill: &[L],
    decode: &[R],
    compatible: impl Fn(&L, &R) -> bool,
) -> bool {
    let mut left_peer = vec![None; prefill.len()];
    let mut right_peer = vec![None; decode.len()];
    for start in 0..prefill.len() {
        let mut parent = vec![None; decode.len()];
        let mut queue = VecDeque::from([start]);
        let free = 'search: loop {
            let Some(left) = queue.pop_front() else {
                return false;
            };
            for (right, peer) in decode.iter().enumerate() {
                if parent[right].is_some() || !compatible(&prefill[left], peer) {
                    continue;
                }
                parent[right] = Some(left);
                match right_peer[right] {
                    None => break 'search right,
                    Some(matched) => queue.push_back(matched),
                }
            }
        };
        let mut next = Some(free);
        while let Some(right) = next {
            let left = parent[right].expect("augmenting path reaches matched worker");
            right_peer[right] = Some(left);
            next = left_peer[left].replace(right);
        }
    }
    true
}

/// Assign one worker from each complete cell to each physical slot, retaining all their
/// common settings. The optional physical-layout constraint can depend on the slot's DP rank.
fn assign_workers(
    slots: &mut [Worker],
    cells: &[Vec<Worker>],
    allowed: &impl Fn(&Worker) -> bool,
) -> bool {
    fn join(a: &Worker, b: &Worker, allowed: &impl Fn(&Worker) -> bool) -> Option<Worker> {
        if a.tp_rank.zip(b.tp_rank).is_some_and(|(x, y)| x != y)
            || a.dp_rank.zip(b.dp_rank).is_some_and(|(x, y)| x != y)
        {
            return None;
        }
        let mut worker = a.clone();
        worker.combine(b).ok()?;
        worker.tp_rank = a.tp_rank.or(b.tp_rank);
        worker.dp_rank = a.dp_rank.or(b.dp_rank);
        allowed(&worker).then_some(worker)
    }
    fn assign(
        slots: &mut [Worker],
        cells: &[Vec<Worker>],
        used: &mut [bool],
        i: usize,
        allowed: &impl Fn(&Worker) -> bool,
    ) -> bool {
        let Some((cell, later)) = cells.split_first() else {
            return true;
        };
        if cell.len() != slots.len() {
            return false;
        }
        if i == 0 {
            if !can_pair(slots, cell, |a, b| join(a, b, allowed).is_some()) {
                return false;
            }
            if later.is_empty() {
                return true;
            }
        }
        let Some(worker) = cell.get(i) else {
            return assign(slots, later, &mut vec![false; slots.len()], 0, allowed);
        };
        let mut tried = BTreeSet::new();
        for j in 0..slots.len() {
            if used[j] || !tried.insert(slots[j].clone()) {
                continue;
            }
            if let Some(merged) = join(&slots[j], worker, allowed) {
                let old = std::mem::replace(&mut slots[j], merged);
                used[j] = true;
                if assign(slots, cells, used, i + 1, allowed) {
                    return true;
                }
                slots[j] = old;
                used[j] = false;
            }
        }
        false
    }
    assign(slots, cells, &mut vec![false; slots.len()], 0, allowed)
}

pub(super) fn workers_fit(rosters: &[&[Worker]]) -> bool {
    fn fit(mut cells: Vec<Vec<Worker>>) -> bool {
        for cell in &mut cells {
            cell.sort_unstable();
        }
        cells.sort_unstable();
        cells.dedup();
        let Some(mut first) = cells.pop() else {
            return true;
        };
        assign_workers(&mut first, &cells, &|w| super::layout::possible_gpu(w) != 0)
    }

    if rosters
        .iter()
        .flat_map(|r| r.iter())
        .all(|w| w.tp_rank.is_some())
    {
        let ranks: BTreeSet<_> = rosters[0].iter().map(|w| w.tp_rank).collect();
        ranks.into_iter().all(|rank| {
            fit(rosters
                .iter()
                .map(|r| r.iter().filter(|w| w.tp_rank == rank).cloned().collect())
                .collect())
        })
    } else {
        fit(rosters.iter().map(|r| r.to_vec()).collect())
    }
}

/// Each slot needs one possible initialized backend. On one deployment a slot is a physical
/// TP/DP worker. Across a handoff it is a TP rank shared by every routable DP peer.
pub(super) fn backends_fit(
    rosters: &[&[Worker]],
    slots: usize,
    allowed: impl Fn(&Worker, usize) -> bool,
) -> bool {
    if slots == 0
        || rosters
            .iter()
            .any(|r| r.is_empty() || !r.len().is_multiple_of(slots))
    {
        return false;
    }
    if rosters.len() == 1 && rosters[0].len() == slots {
        return true;
    }
    let names: BTreeSet<_> = rosters
        .iter()
        .flat_map(|r| r.iter())
        .flat_map(|w| &w.backends)
        .collect();
    if names.len() <= 1
        || names.iter().any(|name| {
            rosters.iter().all(|r| {
                r.iter()
                    .all(|w| w.backends.is_empty() || w.backends.contains(name))
            })
        })
    {
        return true;
    }
    // Names that occur in exactly the same observed sets are interchangeable for the handshake.
    // Keep one representative even when hundreds of workers repeated a mixed per-layer set.
    let recorded: BTreeSet<_> = rosters
        .iter()
        .flat_map(|r| r.iter())
        .map(|w| &w.backends)
        .collect();
    let mut signatures = BTreeSet::new();
    let names: Vec<_> = names
        .into_iter()
        .filter(|name| {
            signatures.insert(
                recorded
                    .iter()
                    .map(|set| set.contains(name))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    // Count forced names first. For anonymous singleton observations this is the entire
    // feasibility test: remaining slots in every roster have unobserved names.
    let mut required = BTreeMap::new();
    let mut anonymous_singletons = true;
    for roster in rosters {
        let copies = roster.len() / slots;
        let mut counts = BTreeMap::new();
        for worker in *roster {
            anonymous_singletons &= worker.tp_rank.is_none() && worker.dp_rank.is_none();
            if !worker.backends.is_empty() {
                let mut options = names
                    .iter()
                    .copied()
                    .filter(|n| worker.backends.contains(n));
                let first = options
                    .next()
                    .expect("each observed class has a representative");
                if options.next().is_none() {
                    *counts.entry(first).or_insert(0usize) += 1;
                } else {
                    anonymous_singletons = false;
                }
            }
        }
        for (name, n) in counts {
            let needed = required.entry(name).or_insert(0usize);
            *needed = (*needed).max(n.div_ceil(copies));
        }
    }
    if required.values().sum::<usize>() > slots || slots == 1 {
        return false;
    }
    if anonymous_singletons {
        return true;
    }

    // Equal-sized feasible rosters must have a perfect bipartite matching. When each
    // slot holds just one worker and one roster has no coordinates, this test is also sufficient.
    for (i, a) in rosters.iter().enumerate() {
        for b in &rosters[..i] {
            if a.len() != b.len() {
                continue;
            }
            if !can_pair(a, b, |x, y| {
                (x.tp_rank.is_none() || y.tp_rank.is_none() || x.tp_rank == y.tp_rank)
                    && !x.excludes_backend(y)
            }) {
                return false;
            }
            if rosters.len() == 2
                && a.len() == slots
                && [*a, *b].iter().any(|rows| {
                    rows.iter()
                        .all(|w| w.tp_rank.is_none() && w.dp_rank.is_none())
                })
            {
                return true;
            }
        }
    }
    let placements: Vec<_> = rosters
        .iter()
        .map(|roster| {
            roster
                .iter()
                .map(|worker| {
                    let positions: Vec<_> = (0..slots).filter(|&i| allowed(worker, i)).collect();
                    let labels: Vec<_> = (0..names.len())
                        .filter(|&i| {
                            worker.backends.is_empty() || worker.backends.contains(names[i])
                        })
                        .collect();
                    (positions, labels)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let mut choices = vec![(0..names.len()).collect::<Vec<_>>(); slots];
    for (positions, labels) in placements.iter().flatten() {
        if let [slot] = positions[..] {
            choices[slot].retain(|i| labels.contains(i));
        }
    }
    let fits = |choices: &[Vec<usize>]| {
        placements.iter().all(|roster| {
            let copies = roster.len() / slots;
            let positions: Vec<_> = (0..slots)
                .flat_map(|i| std::iter::repeat_n(i, copies))
                .collect();
            can_pair(roster, &positions, |(positions, labels), slot| {
                positions.contains(slot) && choices[*slot].iter().any(|i| labels.contains(i))
            })
        })
    };
    // Equivalent positions differ only in their lost coordinates. Explore distinct backend
    // selections in sorted order in such slots, never permutations of the original workers.
    let mut previous = BTreeMap::new();
    let prev: Vec<_> = (0..slots)
        .map(|i| {
            let mask: Vec<_> = placements
                .iter()
                .flatten()
                .map(|(positions, _)| positions.contains(&i))
                .collect();
            previous.insert(mask, i)
        })
        .collect();
    fn assign(
        choices: &mut [Vec<usize>],
        prev: &[Option<usize>],
        start: usize,
        fits: &impl Fn(&[Vec<usize>]) -> bool,
    ) -> bool {
        if !fits(choices) {
            return false;
        }
        let Some(i) = (start..choices.len()).find(|&i| choices[i].len() > 1) else {
            return true;
        };
        let lower = prev[i].map_or(0, |j| choices[j][0]);
        let labels = std::mem::take(&mut choices[i]);
        for &name in labels.iter().filter(|&&n| n >= lower) {
            choices[i] = vec![name];
            if assign(choices, prev, i + 1, fits) {
                return true;
            }
        }
        choices[i] = labels;
        false
    }
    assign(&mut choices, &prev, 0, &fits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn finalized_pool_keeps_its_captured_scheduler_and_context_limits() {
        let cell = json!({"max_num_tokens":1024,"max_batch_size":64,
            "runtime_memory":{"max_model_len":4096,"kv_cache_bytes":1622016}});
        let entry = json!({"runtime_settings":{
            "model_config":{"max_model_len":4096}, "scheduler_config":{"max_num_batched_tokens":1024,"max_num_seqs":64}}});
        let mut identity = super::super::NixlIdentity::default();
        assert_eq!(
            record_cell_memory(&cell, &entry, &Value::Null, &mut identity.recorded_memory).unwrap(),
            Some(1622016)
        );
        let memory = &cell["runtime_memory"];
        let caps = Some((Some(1024), Some(64)));
        assert!(identity.selected_memory(memory, caps).is_ok());
        assert!(
            identity
                .selected_memory(memory, Some((Some(512), Some(64))))
                .is_err()
        );
        assert!(
            identity
                .selected_memory(memory, Some((Some(1024), Some(32))))
                .is_err()
        );
        assert!(
            identity
                .selected_memory(&json!({"max_model_len":8192}), caps)
                .is_err()
        );
        let mut edited = cell.clone();
        edited["max_num_tokens"] = json!(512);
        assert!(record_cell_memory(&edited, &entry, &Value::Null, &mut json!({})).is_err());
    }

    #[test]
    fn selected_memory_retains_the_measured_limits_in_raw_and_draft_records() {
        let settings = json!({"model_config":{"max_model_len":4096},"scheduler_config":{"max_num_batched_tokens":1024,"max_num_seqs":64}});
        let limits = json!({"max_model_len":4096,"max_num_tokens":1024,"max_batch_size":64});
        let raw = json!({"source":"runtime_observations","observation_schema":"aisimulate-runtime-observation/v1","runtime_settings":settings});
        let draft = json!({"schema_version":"aisimulate-onboarding-runtime-probe/v2","observed_resources":{
            "max_num_tokens":1024,"max_batch_size":64,"runtime_memory":{"max_model_len":4096}}});
        assert_eq!(selected_memory(&raw).unwrap(), limits);
        assert_eq!(selected_memory(&draft).unwrap(), limits);
        assert_eq!(
            selected_memory(&json!({"source":"runtime_observations"})).unwrap(),
            json!({})
        );
    }

    #[test]
    fn measured_pool_retains_its_gpu_memory_fraction() {
        let mut expected = json!({});
        let cell = json!({"runtime_memory":{"kv_cache_bytes":1024,"gpu_memory_utilization":0.9}});
        let mut source =
            json!({"runtime_settings":{"cache_config":{"gpu_memory_utilization":0.9}}});
        assert_eq!(
            record_cell_memory(&cell, &source, &Value::Null, &mut expected).unwrap(),
            Some(1024)
        );
        assert_eq!(expected["gpu_memory_utilization"], 0.9);
        source["runtime_settings"]["cache_config"]["gpu_memory_utilization"] = json!(0.5);
        assert!(record_cell_memory(&cell, &source, &Value::Null, &mut expected).is_err());
        let mut full = json!({"gpu_memory_utilization":1});
        record_memory_field(
            &mut full,
            &json!({"gpu_memory_utilization":1.0}),
            "gpu_memory_utilization",
        )
        .unwrap();
    }

    #[test]
    fn bound_collection_retains_each_phases_limits_and_minimum_capacity() {
        let phase = |bytes| {
            json!({"source":"runtime_observations",
            "runtime_settings":{"scheduler_config":{"max_num_batched_tokens":1024}},
            "artifacts":[{"evidence":{"kind":"scheduler","resolved_config":{
                "cache_config":{"gpu_memory_utilization":0.9}}}}],
            "rank_capacities":[{"kv_cache_bytes":bytes}]})
        };
        let mut entry = json!({"source":"bound_collection_phases",
            "phases":{"prefill":phase(4096),"decode":phase(2048)}});
        assert_eq!(
            selected_memory(&entry).unwrap(),
            json!({
            "max_num_tokens":1024,"gpu_memory_utilization":0.9,"kv_cache_bytes":2048})
        );
        entry["phases"]["decode"]["launch"] = json!({"collection":{"gpu_memory_utilization":0.8}});
        assert!(selected_memory(&entry).is_err());
    }

    #[test]
    fn measured_rank_bytes_agree_with_retained_block_accounting() {
        let mut rows = json!({"rank_capacities":[{"initial_free_blocks":98,"num_blocks":100,
            "page_size_bytes":32768,"kv_cache_bytes":3211264,"allocated_cache_bytes":3276800}]});
        assert_eq!(rank_capacity(&rows).unwrap(), Some(3211264));
        for bytes in [0, 6422528] {
            rows["rank_capacities"][0]["kv_cache_bytes"] = json!(bytes);
            assert!(rank_capacity(&rows).is_err());
        }
        rows["rank_capacities"][0]["page_size_bytes"] = Value::Null;
        assert!(rank_capacity(&rows).is_err());
        assert_eq!(
            rank_capacity(&json!({"rank_capacities":[{}]})).unwrap(),
            None
        );
    }

    #[test]
    fn inventoried_formal_index_requires_the_recorded_path_and_digest() {
        let index = json!({"path":"index","sha256":"0".repeat(64)});
        let cell = json!({"observations_index":index});
        let mut manifest = json!({"source_artifacts":[index],"runtime_compatibility":{
            "observations_index":"index","observations_sha256":"0".repeat(64)}});
        assert!(validate_formal_index(&cell, &manifest).is_ok());
        for key in ["observations_index", "observations_sha256"] {
            let saved = manifest["runtime_compatibility"]
                .as_object_mut()
                .unwrap()
                .remove(key)
                .unwrap();
            assert!(validate_formal_index(&cell, &manifest).is_err());
            manifest["runtime_compatibility"][key] = saved;
        }
        assert!(validate_formal_index(&cell, &json!({})).is_ok());
    }

    #[test]
    fn large_anonymous_rosters_keep_multiplicity_of_disjoint_backend_classes() {
        let class = |names: &[&str]| Worker {
            backends: names.iter().map(|n| n.to_string()).collect(),
            ..Default::default()
        };
        let a = class(&["flash_attn", "flashinfer"]);
        let b = class(&["flex_attention", "triton_attn"]);
        let roster = |size, left| {
            std::iter::repeat_n(a.clone(), left)
                .chain(std::iter::repeat_n(b.clone(), size - left))
                .collect::<Vec<_>>()
        };
        for size in [64, 128] {
            let p = roster(size, size * 3 / 4);
            let mut d = roster(size, size / 4);
            assert!(!backends_fit(&[&p, &d], size, |_, _| true));
            d.clone_from(&p);
            d.reverse();
            assert!(backends_fit(&[&p, &d], size, |_, _| true));
            d[0] = Worker::default();
            assert!(backends_fit(&[&p, &d], size, |_, _| true));
        }
    }

    #[test]
    fn captured_topology_checks_raw_workers_without_an_independent_capacity_list() {
        let workers = json!([{"evidence":{"kind":"worker","phase":"prefill","dp_rank":0,"tp_rank":0}},
            {"evidence":{"kind":"worker","phase":"prefill","dp_rank":0,"tp_rank":1}},
            {"evidence":{"kind":"scheduler","phase":"prefill","dp_rank":0}}]);
        let mut entry = json!({"source":"runtime_observations","launch":{"topology":{"tp":2,"dp":1}},
            "artifacts":workers});
        assert!(validate_source(&entry, Capture::Full, None, None).unwrap());
        entry["launch_artifacts"] = json!([{"path":"a"}, {"path":"b"}]);
        assert!(validate_source(&entry, Capture::Full, None, None).is_err());
        entry["launch_artifacts"] = json!([{"path":"a"}]);
        assert!(validate_source(&entry, Capture::Full, None, None).unwrap());
        entry["artifacts"][0]["evidence"]["tp_rank"] = json!(1);
        assert!(validate_source(&entry, Capture::Full, None, None).is_err());
        entry.as_object_mut().unwrap().remove("artifacts");
        assert!(validate_source(&entry, Capture::Full, None, None).is_err());
        entry["artifacts"] = json!([{"path":"worker.json","sha256":"old"}]);
        assert!(validate_source(&entry, Capture::References, None, None).is_err());
        entry["rank_capacities"] = json!([
            {"phase":"prefill","dp_rank":0,"tp_rank":0}, {"phase":"prefill","dp_rank":0,"tp_rank":1}
        ]);
        assert!(validate_source(&entry, Capture::References, None, None).is_err());
        entry["launch_artifacts"][0]["sha256"] = json!("0".repeat(64));
        entry["artifacts"] = json!(
            ["worker-a", "worker-b", "scheduler"]
                .map(|path| json!({"path":path, "sha256":"0".repeat(64)}))
        );
        assert!(validate_source(&entry, Capture::References, None, None).is_err());
        entry["observations_index"] = json!({"path":"observations.json","sha256":"0".repeat(64)});
        assert!(validate_source(&entry, Capture::References, None, None).unwrap());
    }

    #[test]
    fn anonymous_cells_need_one_assignment_of_backend_and_layout_together() {
        let w = |backend: &str, layout: &str| Worker {
            tp_rank: Some(0),
            backends: vec![backend.to_string()],
            kv_cache_layout: super::super::Layouts::named(layout),
            ..Default::default()
        };
        let a = vec![w("a", "BLHNC"), w("b", "BLNHC")];
        let mut b = a.clone();
        b.reverse();
        assert!(workers_fit(&[&a, &b]));
        let c = vec![w("a", "BLNHC"), w("b", "BLHNC")];
        assert!(!workers_fit(&[&a, &c]));
        b[0].kv_cache_layout = None;
        assert!(workers_fit(&[&a, &b]));
        b[0].kv_cache_layout = super::super::Layouts::named("BLHNC");
        assert!(!workers_fit(&[&a, &b]));
    }

    #[test]
    fn formal_memory_requires_each_dp_scheduler_once_without_a_phase_label() {
        let worker =
            |dp| serde_json::json!({"evidence":{"kind":"worker","tp_rank":0,"dp_rank":dp}});
        let scheduler = |dp| serde_json::json!({"evidence":{"kind":"scheduler","dp_rank":dp}});
        let mut entry = serde_json::json!({"source":"vllm_initialized_cache","plan_sha256":"plan",
            "artifacts":[worker(0),worker(1),scheduler(0),scheduler(1)]});
        assert!(validate_source(&entry, Capture::Full, Some(2), Some(1)).unwrap());
        entry["artifacts"][3] = scheduler(0);
        assert!(validate_source(&entry, Capture::Full, Some(2), Some(1)).is_err());
        entry["artifacts"].as_array_mut().unwrap().pop();
        assert!(validate_source(&entry, Capture::Full, Some(2), Some(1)).is_err());
    }

    #[test]
    fn a_compact_cell_with_no_worker_identity_still_binds_its_rank_capacities() {
        let source = "a".repeat(64);
        let first = json!({"phase":"decode","tp_rank":0,"dp_rank":0,"kv_cache_bytes":256});
        let second = json!({"phase":"decode","tp_rank":1,"dp_rank":0,"kv_cache_bytes":128});
        let mut cell = json!({"rank_capacities":[first,second],"observations_index":{"path":"index","sha256":source},
            "artifacts":[{"path":"worker","kind":"worker","sha256":source}]});
        let original = digest(&cell, Some(&source), &json!({}), false, None).unwrap();
        cell["rank_capacities"].as_array_mut().unwrap().reverse();
        assert_eq!(
            digest(&cell, Some(&source), &json!({}), false, None).unwrap(),
            original
        );
        cell["rank_capacities"][0]["phase"] = json!("prefill");
        assert_ne!(
            digest(&cell, Some(&source), &json!({}), false, None).unwrap(),
            original
        );
    }
}
