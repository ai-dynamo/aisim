// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Modified vLLM scheduler group and hash block checks. See THIRD_PARTY_NOTICES.md
//! for the original paths, immutable revision, and Apache-2.0 license.

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

fn capacity_conflict() -> AicError {
    super::conflict("kv_cache_bytes")
}

// Older formal summaries stored only usable bytes. Compare every retained measurement
// of a field and derive the remaining fields only where the collector's inputs exist.
fn pool_field(key: &str, records: &[&Value]) -> Result<Option<u64>, AicError> {
    let mut result = None;
    for record in records {
        let Some(field) = record.get(key).filter(|v| !v.is_null()) else {
            continue;
        };
        let value = field.as_u64().ok_or_else(capacity_conflict)?;
        if value > 1 << 53 || result.is_some_and(|n| n != value) {
            return Err(capacity_conflict());
        }
        result = Some(value);
    }
    Ok(result)
}

struct RankCapacity {
    limit: Option<u64>,
    exact: bool,
}

fn measured_rank_capacity(
    row: &Value,
    worker: &Value,
    scheduler: &Value,
    recorded_page: Option<u64>,
    pool: &mut (Option<u64>, Option<u64>),
) -> Result<RankCapacity, AicError> {
    super::merge_observed(
        &mut pool.0,
        pool_field("num_blocks", &[row, worker, scheduler])?,
        "kv_cache_bytes",
    )?;
    super::merge_observed(
        &mut pool.1,
        pool_field("initial_free_blocks", &[row, scheduler])?,
        "kv_cache_bytes",
    )?;
    let (mut count, mut free) = *pool;
    let reserved = pool_field("reserved_blocks", &[row, scheduler])?;
    let mut allocated = pool_field("allocated_cache_bytes", &[row, worker])?;
    let mut page = recorded_page;
    super::merge_observed(
        &mut page,
        pool_field("page_size_bytes", &[row])?,
        "page_size_bytes",
    )?;
    let mut bytes = pool_field("kv_cache_bytes", &[row])?;
    if [free, allocated, page, bytes, reserved].contains(&Some(0))
        || count.is_some_and(|n| n < 2)
        || free.zip(count).is_some_and(|(f, n)| f >= n)
    {
        return Err(capacity_conflict());
    }
    if let Some(storage) = worker.get("storages") {
        let storage = storage
            .as_array()
            .filter(|s| !s.is_empty())
            .ok_or_else(capacity_conflict)?;
        let mut pointers = BTreeSet::new();
        let mut device = None;
        let mut total = 0_u64;
        for region in storage {
            let name = region["device"]
                .as_str()
                .filter(|d| d.starts_with("cuda:"))
                .ok_or_else(capacity_conflict)?;
            let pointer = super::positive(&region["pointer"]).ok_or_else(capacity_conflict)?;
            let size = pool_field("size_bytes", &[region])?
                .filter(|n| *n > 0)
                .ok_or_else(capacity_conflict)?;
            if device.replace(name).is_some_and(|d| d != name) || !pointers.insert(pointer) {
                return Err(capacity_conflict());
            }
            total = total.checked_add(size).ok_or_else(capacity_conflict)?;
        }
        super::merge_observed(&mut allocated, Some(total), "kv_cache_bytes")?;
    }
    if let (Some(n), Some(r)) = (free, reserved) {
        super::merge_observed(
            &mut count,
            Some(n.checked_add(r).ok_or_else(capacity_conflict)?),
            "kv_cache_bytes",
        )?;
    }
    if let (Some(bytes), Some(free)) = (bytes, free) {
        if !bytes.is_multiple_of(free) {
            return Err(capacity_conflict());
        }
        super::merge_observed(&mut page, Some(bytes / free), "kv_cache_bytes")?;
    }
    if let (Some(allocated), Some(count)) = (allocated, count) {
        if !allocated.is_multiple_of(count) {
            return Err(capacity_conflict());
        }
        super::merge_observed(&mut page, Some(allocated / count), "kv_cache_bytes")?;
    }
    if let (Some(physical), Some(page)) = (allocated, page) {
        if !physical.is_multiple_of(page) {
            return Err(capacity_conflict());
        }
        super::merge_observed(&mut count, Some(physical / page), "kv_cache_bytes")?;
    }
    if let (Some(n), Some(r)) = (count, reserved) {
        super::merge_observed(
            &mut free,
            Some(n.checked_sub(r).ok_or_else(capacity_conflict)?),
            "kv_cache_bytes",
        )?;
    }
    if let (Some(count), Some(page)) = (count, page) {
        let physical = count.checked_mul(page).ok_or_else(capacity_conflict)?;
        super::merge_observed(&mut allocated, Some(physical), "kv_cache_bytes")?;
    }
    if let (Some(bytes), Some(page)) = (bytes, page) {
        if !bytes.is_multiple_of(page) {
            return Err(capacity_conflict());
        }
        super::merge_observed(&mut free, Some(bytes / page), "kv_cache_bytes")?;
    }
    if let (Some(free), Some(page)) = (free, page) {
        let usable = free.checked_mul(page).ok_or_else(capacity_conflict)?;
        super::merge_observed(&mut bytes, Some(usable), "kv_cache_bytes")?;
    }
    if count.is_some_and(|n| n < 2)
        || free == Some(0)
        || free.zip(count).is_some_and(|(f, n)| f >= n)
    {
        return Err(capacity_conflict());
    }
    super::merge_observed(&mut pool.0, count, "kv_cache_bytes")?;
    super::merge_observed(&mut pool.1, free, "kv_cache_bytes")?;
    let budget = pool_field("available_cache_bytes", &[row, worker])?;
    if budget == Some(0) || allocated.zip(budget).is_some_and(|(n, b)| n > b) {
        return Err(capacity_conflict());
    }
    if bytes.is_some_and(|n| n > 1 << 53)
        || allocated.is_some_and(|n| n > 1 << 53)
        || bytes.zip(allocated).is_some_and(|(b, a)| b >= a)
    {
        return Err(capacity_conflict());
    }
    let null = pool_field("null_block_id", &[scheduler])?;
    if pool_field("pool_count", &[scheduler])?.is_some_and(|n| n != 1)
        || pool_field("watermark_blocks", &[scheduler])?.is_some_and(|n| n != 0)
        || null.zip(count).is_some_and(|(id, n)| id >= n)
        || reserved == Some(0)
        || count
            .zip(free)
            .zip(reserved)
            .is_some_and(|((n, f), r)| n - f != r)
    {
        return Err(capacity_conflict());
    }
    if let Some(ids) = scheduler.get("permanent_reserved_block_ids") {
        let mut blocks = BTreeSet::new();
        for id in ids.as_array().ok_or_else(capacity_conflict)? {
            let id = id.as_u64().ok_or_else(capacity_conflict)?;
            if !blocks.insert(id) || count.is_some_and(|n| id >= n) {
                return Err(capacity_conflict());
            }
        }
        let hybrid = worker["groups"]
            .as_array()
            .is_some_and(|groups| groups.iter().any(|g| g.get("layer_specs").is_some()));
        if blocks.is_empty()
            || hybrid && blocks.len() != 1
            || reserved.is_some_and(|r| r != blocks.len() as u64)
            || null.is_some_and(|id| !blocks.contains(&id))
        {
            return Err(capacity_conflict());
        }
    }
    // A sparse row can still bound the usable allocation. At least the null block is
    // reserved. Use its measured page size when that survived the original writer.
    let physical = allocated.or(budget);
    let mut limit = budget;
    if let Some(physical) = physical {
        let reserved = reserved.unwrap_or(1);
        let page = page.unwrap_or(1);
        let usable = (physical / page)
            .checked_sub(reserved)
            .and_then(|n| n.checked_mul(page))
            .filter(|n| *n > 0)
            .ok_or_else(capacity_conflict)?;
        limit = Some(usable);
    }
    if bytes.zip(limit).is_some_and(|(n, l)| n > l) {
        return Err(capacity_conflict());
    }
    Ok(RankCapacity {
        limit: bytes.or(limit),
        exact: bytes.is_some(),
    })
}

fn indexed_pool(worker: &Value, scheduler: &Value) -> Result<(), AicError> {
    let pool = worker["pool_id"]
        .as_str()
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(capacity_conflict)?;
    let groups = worker["groups"].as_array().ok_or_else(capacity_conflict)?;
    let assigned = scheduler["group_pool_ids"]
        .as_array()
        .ok_or_else(capacity_conflict)?;
    if scheduler["pool_id"] != pool
        || groups.len() != assigned.len()
        || assigned.iter().any(|id| id != pool)
        || groups.iter().any(|g| g["pool_id"] != pool)
    {
        return Err(capacity_conflict());
    }
    Ok(())
}

fn check_scheduler_groups(groups: &[Value], scheduler: &Value) -> Result<(), AicError> {
    let scheduled = scheduler["groups"]
        .as_array()
        .filter(|s| s.len() == groups.len())
        .ok_or_else(capacity_conflict)?;
    for (group, saved) in groups.iter().zip(scheduled) {
        // The scheduler omits worker-only labels. For a uniform group vLLM
        // broadcasts the first named layer's concrete spec to the group.
        let saved = saved.as_object().ok_or_else(capacity_conflict)?;
        let mut expected: BTreeMap<_, _> = group
            .as_object()
            .ok_or_else(capacity_conflict)?
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        expected.remove("kind");
        expected.remove("layer_classes");
        if group["spec_type"]
            .as_str()
            .is_some_and(|s| s.ends_with(".UniformTypeKVCacheSpecs"))
        {
            let names = group["layer_names"]
                .as_array()
                .ok_or_else(capacity_conflict)?;
            let first = names
                .first()
                .and_then(Value::as_str)
                .and_then(|name| group["layer_specs"].get(name))
                .filter(|s| s.is_object())
                .ok_or_else(capacity_conflict)?;
            for (key, source) in [
                ("spec_type", "spec_type"),
                ("spec_page_size_bytes", "page_size_bytes"),
                ("dtype", "dtype"),
            ] {
                expected.insert(key, &first[source]);
            }
            let layers = saved
                .get("layer_specs")
                .and_then(Value::as_object)
                .ok_or_else(capacity_conflict)?;
            if layers.len() != names.len()
                || !names
                    .iter()
                    .all(|name| name.as_str().and_then(|n| layers.get(n)) == Some(first))
            {
                return Err(super::conflict("scheduler cache_groups"));
            }
            expected.insert("layer_specs", &saved["layer_specs"]);
        }
        if saved.len() != expected.len()
            || expected
                .into_iter()
                .any(|(key, value)| saved.get(key) != Some(value))
        {
            return Err(super::conflict("scheduler cache_groups"));
        }
    }
    Ok(())
}

fn check_tensor_blocks(worker: &Value, groups: &[Value]) -> Result<(), AicError> {
    if let Some(tensors) = worker["layer_tensors"].as_object() {
        let count = super::positive(&worker["num_blocks"]).ok_or_else(capacity_conflict)?;
        let mut strides = BTreeMap::new();
        for allocation in worker["tensor_allocations"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if let Some(stride) = super::positive(&allocation["block_stride"]) {
                for name in allocation["layers"]
                    .as_array()
                    .or(allocation["shared_by"].as_array())
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    strides.insert(name, stride);
                }
            }
        }
        for group in groups {
            let block =
                super::positive(&group["block_size_tokens"]).ok_or_else(capacity_conflict)?;
            for name in group["layer_names"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                let Some(saved) = tensors.get(name) else {
                    continue;
                };
                let views = saved
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or(std::slice::from_ref(saved));
                for view in views {
                    let kernel = super::positive(&view["kernel_block_size_tokens"])
                        .ok_or_else(capacity_conflict)?;
                    let axis = view["block_axis"]
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(capacity_conflict)?;
                    if !block.is_multiple_of(kernel) {
                        return Err(capacity_conflict());
                    }
                    let ratio = block / kernel;
                    let step = if saved.is_array() {
                        pool_field("kernel_block_stride", &[view])?
                            .filter(|&s| s > 0)
                            .unwrap_or(ratio)
                    } else {
                        ratio
                    };
                    let blocks = (count - 1)
                        .checked_mul(step)
                        .and_then(|n| n.checked_add(ratio))
                        .ok_or_else(capacity_conflict)?;
                    if step < ratio
                        || view["shape"].get(axis).and_then(Value::as_u64) != Some(blocks)
                        || strides.get(name).is_some_and(|&stride| {
                            view["stride_bytes"]
                                .get(axis)
                                .and_then(Value::as_u64)
                                .and_then(|s| s.checked_mul(step))
                                != Some(stride)
                        })
                    {
                        return Err(capacity_conflict());
                    }
                }
            }
        }
    }
    Ok(())
}

// When a historical writer omits one rank, measured ranks still bound the minimum.
fn rank_limits(entry: &Value, mut page: Option<u64>) -> Result<(Option<u64>, bool), AicError> {
    fn pool(r: &Value) -> (Option<&str>, Option<u64>) {
        (r["phase"].as_str(), r["dp_rank"].as_u64())
    }
    let mut workers = BTreeMap::new();
    let mut schedulers = BTreeMap::new();
    for record in entry["artifacts"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|f| &f["evidence"])
    {
        let cache = &record["cache"];
        let indexed = record["schema_version"] == "aisimulate-runtime-observation/v1";
        if indexed || record["schema_name"] == "aisimulate_fpm_runtime_memory" {
            let required: &[&str] = match record["kind"].as_str() {
                Some("worker") => &[
                    "num_blocks",
                    "available_cache_bytes",
                    "allocated_cache_bytes",
                    "storages",
                ],
                Some("scheduler") => &[
                    "num_blocks",
                    "initial_free_blocks",
                    "reserved_blocks",
                    "pool_count",
                    "watermark_blocks",
                    "null_block_id",
                ],
                _ => continue,
            };
            if required.iter().any(|key| cache[key].is_null())
                || indexed
                    && record["kind"] == "scheduler"
                    && cache["permanent_reserved_block_ids"].is_null()
            {
                return Err(capacity_conflict());
            }
        }
        match record["kind"].as_str() {
            Some("worker") => {
                workers.insert((pool(record), record["tp_rank"].as_u64()), record);
            }
            Some("scheduler") => {
                schedulers.insert(pool(record), &record["cache"]);
            }
            _ => {}
        }
    }
    for ((key, _), record) in &workers {
        let worker = &record["cache"];
        let indexed = record["schema_version"] == "aisimulate-runtime-observation/v1";
        if indexed || record["schema_name"] == "aisimulate_fpm_runtime_memory" {
            let groups = worker["groups"].as_array().ok_or_else(capacity_conflict)?;
            if indexed {
                check_tensor_blocks(worker, groups)?;
            }
            if let Some(scheduler) = schedulers.get(key) {
                check_scheduler_groups(groups, scheduler)?;
                if indexed {
                    indexed_pool(worker, scheduler)?;
                }
            }
        }
    }
    let rows = entry["rank_capacities"].as_array();
    if identified_source(entry) {
        // These collectors normalize every group against the same shared pool page and
        // require the normalized groups to agree across workers and collected phases.
        for row in rows.into_iter().flatten() {
            super::merge_observed(
                &mut page,
                pool_field("page_size_bytes", &[row])?,
                "page_size_bytes",
            )?;
        }
    }
    // The formal memory collector identifies a single cell without writing a phase.
    let shared_pool = |(phase, dp): (Option<&str>, Option<u64>)| {
        dp.is_some()
            && (matches!(phase, Some("prefill" | "decode"))
                || phase.is_none() && entry["source"] == "vllm_initialized_cache")
    };
    let mut pools: BTreeMap<_, (Option<u64>, Option<u64>)> = BTreeMap::new();
    // Complete rank summaries already carry both scheduler counts.
    for row in rows.into_iter().flatten() {
        let key = pool(row);
        if shared_pool(key) {
            let shared = pools.entry(key).or_default();
            super::merge_observed(
                &mut shared.0,
                pool_field("num_blocks", &[row])?,
                "kv_cache_bytes",
            )?;
            super::merge_observed(
                &mut shared.1,
                pool_field("initial_free_blocks", &[row])?,
                "kv_cache_bytes",
            )?;
        }
    }
    let ranks = rows
        .into_iter()
        .flatten()
        .chain(workers.values().copied().filter(|_| rows.is_none()));
    // A sparse rank can establish a count from its byte measurements. Recheck
    // earlier ranks against it. Each phase/DP pool can gain only two counts.
    loop {
        let mut changed = false;
        let mut minimum = None;
        let mut complete = true;
        for row in ranks.clone() {
            let key = pool(row);
            let scheduler = schedulers.get(&key).copied().unwrap_or(&Value::Null);
            let mut unassigned = (None, None);
            let shared = if shared_pool(key) {
                pools.entry(key).or_default()
            } else {
                &mut unassigned
            };
            let before = *shared;
            let record = workers
                .get(&(key, row["tp_rank"].as_u64()))
                .copied()
                .unwrap_or(&Value::Null);
            let worker = &record["cache"];
            let capacity = measured_rank_capacity(row, worker, scheduler, page, shared)?;
            changed |= shared_pool(key) && *shared != before;
            if let Some(limit) = capacity.limit {
                minimum = Some(minimum.map_or(limit, |n: u64| n.min(limit)));
            }
            complete &= capacity.exact;
        }
        if !changed {
            return Ok((minimum, complete));
        }
    }
}

#[cfg(test)]
fn rank_capacity(entry: &Value) -> Result<Option<u64>, AicError> {
    let (minimum, complete) = rank_limits(entry, None)?;
    Ok(minimum.filter(|_| complete))
}

pub(super) fn retain_bound(expected: &mut Value, limit: Option<u64>) {
    if let Some(limit) = limit {
        expected["kv_cache_bytes_limit"] = Value::from(
            expected["kv_cache_bytes_limit"]
                .as_u64()
                .map_or(limit, |n| n.min(limit)),
        );
    }
}

pub(super) fn bound_phases(
    entry: &Value,
) -> Result<Option<&serde_json::Map<String, Value>>, AicError> {
    if entry["source"] != "bound_collection_phases" {
        return Ok(entry["phases"].as_object());
    }
    let phases = entry["phases"]
        .as_object()
        .filter(|p| !p.is_empty())
        .ok_or_else(invalid)?;
    for (name, snapshot) in phases {
        if !matches!(name.as_str(), "prefill" | "decode")
            || snapshot["source"] != "runtime_observations"
            || !identified_source(snapshot)
            || std::iter::once(&snapshot["launch"])
                .chain(
                    snapshot["artifacts"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|r| &r["evidence"]["launch"]),
                )
                .filter_map(Value::as_object)
                .any(|launch| match launch.get("worker_type") {
                    None | Some(Value::Null) => phases.len() != 2,
                    Some(role) if role == "aggregated" => phases.len() != 2,
                    Some(role) => phases.len() != 1 || role != name.as_str(),
                })
        {
            return Err(invalid());
        }
        let mut identified = false;
        for marker in snapshot["rank_capacities"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|r| &r["phase"])
            .chain(
                snapshot["artifacts"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|r| &r["evidence"]["phase"]),
            )
        {
            if !marker.is_null() {
                if marker != name.as_str() {
                    return Err(invalid());
                }
                identified = true;
            }
        }
        if let Some(settings) = snapshot["prefix_caching"].as_object() {
            if settings.keys().any(|phase| phase != name) {
                return Err(invalid());
            }
        }
        if !identified || rank_limits(snapshot, None)?.0.is_none() {
            return Err(invalid());
        }
    }
    Ok(Some(phases))
}

fn check_scheduler_hash(record: &Value) -> Result<(), AicError> {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let invalid = || super::conflict("hash_block_size_tokens");
    let cache = &record["cache"];
    let settings = &record["resolved_config"]["cache_config"];
    let groups = cache["groups"]
        .as_array()
        .filter(|g| !g.is_empty())
        .ok_or_else(invalid)?;
    let hash = super::positive(&cache["hash_block_size"]).ok_or_else(invalid)?;
    let scheduler_block = super::positive(&cache["block_size"]).ok_or_else(invalid)?;
    let unit = settings.get("prefix_match_unit").filter(|v| !v.is_null());
    if cache["prefix_caching"]["block_size"] != scheduler_block {
        return Err(invalid());
    }
    let mamba = |g: &Value| g["spec_type"] == "vllm.v1.kv_cache_interface.MambaSpec";
    let caching = settings["enable_prefix_caching"] == true;
    let split = groups.len() > 1
        && caching
        && !groups
            .iter()
            .any(|g| mamba(g) && g["state"]["mamba_cache_mode"] != "align");
    let alignment = if split
        && groups
            .iter()
            .any(|g| mamba(g) && g["block_size_tokens"].as_u64().is_some_and(|b| b > hash))
    {
        hash
    } else {
        scheduler_block
    };
    let mut lcm = 1;
    let mut all = 0;
    let mut participating = 0;
    let mut mandatory = false;
    for group in groups {
        let block = super::positive(&group["block_size_tokens"]).ok_or_else(invalid)?;
        lcm = (lcm / gcd(lcm, block))
            .checked_mul(block)
            .ok_or_else(invalid)?;
        all = gcd(all, block);
        if !split || group["spec_type"] == "vllm.v1.kv_cache_interface.CircularBufferSpec" {
            continue;
        }
        // A sliding MLA group may opt out. Include every eligible optional group
        // to test whether their GCD can reach the recorded hash size.
        let optional = group["spec_type"] == "vllm.v1.kv_cache_interface.SlidingWindowMLASpec";
        let fits = block.is_multiple_of(hash)
            && group["layer_specs"]
                .as_object()
                .into_iter()
                .flat_map(|layers| layers.values())
                .filter_map(|s| s["tokens_per_state"].as_u64().filter(|&n| n > 1))
                .all(|n| alignment.is_multiple_of(n));
        if !optional && !fits {
            return Err(invalid());
        }
        mandatory |= !optional;
        if fits {
            participating = gcd(participating, block);
        }
    }
    let valid_hash = if !split {
        hash == lcm && (groups.len() != 1 || !caching || unit.is_none_or(|v| *v == hash))
    } else if let Some(unit) = unit {
        *unit == hash && (participating > 0 || all.is_multiple_of(hash))
    } else {
        participating == hash || !mandatory && all == hash
    };
    if scheduler_block != lcm || !valid_hash {
        return Err(invalid());
    }
    Ok(())
}

// Importers retain the first collected phase's cache policy. Workers and schedulers
// from the other phase can legitimately use a different Mamba mode and hash size.
fn record_prefix_settings(expected: &mut Value, entry: &Value) -> Result<(), AicError> {
    let mut phases = entry["prefix_caching"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let records = entry["artifacts"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|f| &f["evidence"]);
    // The indexed importer reads the per-process prefix record only for hybrid captures.
    // Older adapters can carry an identically named extension with different semantics.
    let hybrid: BTreeSet<_> = records
        .clone()
        .filter(|r| {
            entry["source"] == "runtime_observations"
                && r["kind"] == "worker"
                && r["cache"]["groups"]
                    .as_array()
                    .is_some_and(|gs| gs.iter().any(|g| g.get("layer_specs").is_some()))
        })
        .filter_map(|r| r["phase"].as_str())
        .collect();
    for record in records {
        let Some(phase) = record["phase"].as_str().filter(|p| hybrid.contains(p)) else {
            continue;
        };
        let cache = &record["cache"];
        let prefix = &cache["prefix_caching"];
        if !prefix.is_object() {
            return Err(super::conflict("prefix_caching"));
        }
        let settings = phases.entry(phase).or_insert_with(|| serde_json::json!({}));
        if !settings.is_object() {
            *settings = serde_json::json!({});
        }
        for key in ["enable_prefix_caching", "mamba_cache_mode"] {
            record_memory_field(settings, prefix, key)?;
            record_memory_field(settings, &record["resolved_config"]["cache_config"], key)?;
        }
        for group in cache["groups"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|g| g["kind"] == "state")
        {
            if record["kind"] == "worker"
                && Worker::contains_spec(group, "vllm.v1.kv_cache_interface.MambaSpec")
            {
                let block = &record["resolved_config"]["cache_config"]["mamba_block_size"];
                if group["state"]["mamba_cache_mode"].is_null()
                    || block.is_null()
                    || &group["block_size_tokens"] != block
                {
                    return Err(super::conflict("block_size_tokens"));
                }
            }
            record_memory_field(settings, &group["state"], "mamba_cache_mode")?;
        }
        if record["kind"] == "scheduler" {
            check_scheduler_hash(record)?;
            record_memory_field(settings, cache, "mamba_cache_mode")?;
            for source in [prefix, cache] {
                record_field(
                    settings,
                    "hash_block_size_tokens",
                    &source["hash_block_size"],
                )?;
            }
        }
    }
    for settings in phases.values() {
        if let Some(mode) = settings.get("mamba_cache_mode").filter(|v| !v.is_null())
            && (!matches!(mode.as_str(), Some("none" | "align"))
                || mode == "align" && settings["enable_prefix_caching"] == false)
        {
            return Err(super::conflict("mamba_cache_mode"));
        }
    }
    if let Some(settings) = phases.get("prefill").or_else(|| phases.get("decode")) {
        record_memory_settings(expected, settings)?;
    }
    Ok(())
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
    let mut page = None;
    if identified_source(entry) {
        for group in cell["cache_groups"].as_array().into_iter().flatten() {
            super::merge_observed(
                &mut page,
                pool_field("page_size_bytes", &[group])?,
                "page_size_bytes",
            )?;
        }
    }
    let (limit, complete) = rank_limits(entry, page)?;
    let observed = limit.filter(|_| complete);
    let saved = memory["kv_cache_bytes"].as_u64();
    let formal = (entry["source"] == "runtime_observations")
        .then(|| compatibility["observed_capacity_bytes"].as_u64())
        .flatten();
    if !complete {
        if saved.or(formal).zip(limit).is_some_and(|(s, l)| s > l) {
            return Err(capacity_conflict());
        }
        retain_bound(expected, limit);
    }
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
    record_prefix_settings(expected, entry)?;
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
        // The released draft writers strip raw provenance from observed_resources. If an
        // archive also carries it, validate that measurement before applying a user override.
        if let Some(direct) = memory["provenance"]
            .as_str()
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
        {
            let measured = selected_memory(&direct)?;
            for (key, value) in measured.as_object().into_iter().flatten() {
                if !matches!(key.as_str(), "kv_cache_bytes" | "kv_cache_bytes_limit") {
                    record_field(&mut expected, key, value)?;
                }
            }
            if measured["kv_cache_bytes"]
                .as_u64()
                .is_some_and(|n| recorded != Some(n))
                || measured["kv_cache_bytes_limit"]
                    .as_u64()
                    .zip(recorded)
                    .is_some_and(|(l, n)| n > l)
            {
                return Err(capacity_conflict());
            }
        }
        let (limit, complete) = rank_limits(&evidence["nixl_observation"], None)?;
        if limit
            .zip(recorded)
            .is_some_and(|(l, n)| n > l || complete && n != l)
        {
            return Err(capacity_conflict());
        }
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
    } else if evidence["source"] == "bound_collection_phases" {
        let phases = bound_phases(evidence)?.ok_or_else(invalid)?;
        let mut minimum = None;
        let mut complete = true;
        for (name, entry) in phases {
            record_collection_settings(&mut expected, entry)?;
            let mut settings = serde_json::json!({});
            record_prefix_settings(&mut settings, entry)?;
            if name == "prefill" || !phases.contains_key("prefill") {
                record_memory_settings(&mut expected, &settings)?;
            }
            let (limit, exact) = rank_limits(entry, None)?;
            if let Some(limit) = limit {
                minimum = Some(minimum.map_or(limit, |m: u64| m.min(limit)));
            }
            complete &= exact && limit.is_some();
        }
        if !complete {
            retain_bound(&mut expected, minimum);
        }
        minimum.filter(|_| complete)
    } else if identified_source(evidence) {
        record_collection_settings(&mut expected, evidence)?;
        record_prefix_settings(&mut expected, evidence)?;
        let (limit, complete) = rank_limits(evidence, None)?;
        if !complete {
            retain_bound(&mut expected, limit);
        }
        limit.filter(|_| complete)
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
        let phase = |bytes, phase| {
            json!({"source":"runtime_observations","observation_schema":"aisimulate-runtime-observation/v1",
            "runtime_settings":{"scheduler_config":{"max_num_batched_tokens":1024}},
            "artifacts":[{"evidence":{"kind":"scheduler","resolved_config":{
                "cache_config":{"gpu_memory_utilization":0.9}}}}],
            "rank_capacities":[{"kv_cache_bytes":bytes,"phase":phase}]})
        };
        let mut entry = json!({"source":"bound_collection_phases",
            "phases":{"prefill":phase(4096, "prefill"),"decode":phase(2048, "decode")}});
        assert_eq!(
            selected_memory(&entry).unwrap(),
            json!({
            "max_num_tokens":1024,"gpu_memory_utilization":0.9,"kv_cache_bytes":2048})
        );
        entry["phases"]["decode"]["launch"] = json!({"collection":{"gpu_memory_utilization":0.8}});
        assert!(selected_memory(&entry).is_err());
        entry["phases"]["decode"] = phase(2048, "decode");
        entry["phases"]["prefill"]["launch"] = json!({"worker_type":"aggregated"});
        assert!(selected_memory(&entry).is_ok());
        entry["phases"].as_object_mut().unwrap().remove("decode");
        assert!(selected_memory(&entry).is_err());
        entry["phases"]["prefill"]["launch"] = json!({});
        assert!(selected_memory(&entry).is_err());
        entry["phases"]["prefill"]["launch"]["worker_type"] = json!("prefill");
        assert!(selected_memory(&entry).is_ok());
        entry["phases"]["prefill"] = json!({"source":"runtime_observations","observation_schema":"aisimulate-runtime-observation/v1",
            "prefix_caching":{"prefill":{}},"rank_capacities":[{"phase":"prefill"}]});
        assert!(selected_memory(&entry).is_err());
    }

    #[test]
    fn partially_recorded_ranks_preserve_usable_upper_bounds() {
        let raw = |partial| {
            json!({"rank_capacities":[{"phase":"prefill","dp_rank":0,"tp_rank":0,
            "kv_cache_bytes":3211264},partial]})
        };
        assert_eq!(
            rank_limits(
                &raw(json!({"phase":"decode","dp_rank":0,"tp_rank":0,
            "num_blocks":80,"page_size_bytes":32768,"allocated_cache_bytes":2621440})),
                None
            )
            .unwrap(),
            (Some(2588672), false)
        );
        let invalid =
            raw(json!({"phase":"decode","dp_rank":0,"tp_rank":0,"allocated_cache_bytes":2621441}));
        assert!(rank_limits(&invalid, Some(32768)).is_err());
        let entry =
            raw(json!({"phase":"decode","dp_rank":0,"tp_rank":0,"available_cache_bytes":2621440}));
        assert_eq!(rank_limits(&entry, None).unwrap(), (Some(2_621_439), false));
    }

    #[test]
    fn sparse_rank_byte_measurements_check_each_budget_in_the_same_pool() {
        const PAGE: u64 = 32_768;
        // TP1 retained 98 usable pages. Only ranks sharing its scheduler inherit
        // that count and its physical allocation.
        for (budget, physical, phase, dp, exact) in [
            (99, 100, "decode", Some(0), None),
            (99, 99, "decode", Some(0), Some(true)),
            (100, 100, "decode", Some(0), Some(true)),
            (99, 100, "prefill", Some(0), Some(false)),
            (99, 100, "decode", Some(1), Some(false)),
            (99, 100, "decode", None, Some(false)),
        ] {
            let tp0 = json!({"phase":"decode","dp_rank":0,"tp_rank":0,
                "available_cache_bytes":budget * PAGE});
            let tp1 = json!({"phase":phase,"dp_rank":dp,"tp_rank":1,
                "allocated_cache_bytes":physical * PAGE,"kv_cache_bytes":98 * PAGE});
            for ranks in [[&tp0, &tp1], [&tp1, &tp0]] {
                let entry = json!({"rank_capacities":ranks});
                assert_eq!(
                    rank_limits(&entry, Some(PAGE)).ok(),
                    exact.map(|e| (Some(98 * PAGE), e)),
                    "{entry}"
                );
            }
        }
    }

    #[test]
    fn the_formal_collector_identifies_an_unnamed_single_phase_pool() {
        let mut entry = json!({"rank_capacities":[
            {"dp_rank":0,"tp_rank":0,"available_cache_bytes":99*32},
            {"dp_rank":0,"tp_rank":1,"allocated_cache_bytes":100*32,"kv_cache_bytes":98*32}
        ]});
        assert_eq!(
            rank_limits(&entry, Some(32)).unwrap(),
            (Some(98 * 32), false)
        );
        entry["source"] = json!("vllm_initialized_cache");
        assert!(rank_limits(&entry, Some(32)).is_err());
    }

    #[test]
    fn capacity_agrees_with_the_scheduler_for_each_rank_before_taking_the_minimum() {
        let sizes = |free| {
            json!({"dp_rank":0,"tp_rank":1,"phase":"prefill","num_blocks":100,
            "page_size_bytes":32,"allocated_cache_bytes":3200,"available_cache_bytes":3200,
            "initial_free_blocks":free,"kv_cache_bytes":free * 32})
        };
        let mut evidence = json!({"rank_capacities":[sizes(98),sizes(98)], "artifacts":[{"evidence":{
            "kind":"scheduler","dp_rank":0,"phase":"prefill","cache":{"num_blocks":100,"initial_free_blocks":98,
            "reserved_blocks":2,"null_block_id":0,"permanent_reserved_block_ids":[0,1]}}}]});
        evidence["rank_capacities"][0]["tp_rank"] = json!(0);
        assert_eq!(rank_capacity(&evidence).unwrap(), Some(3136));
        for (field, bad) in [
            ("available_cache_bytes", 3199),
            ("initial_free_blocks", 100),
        ] {
            let mut changed = evidence.clone();
            changed["rank_capacities"][1][field] = json!(bad);
            assert!(rank_capacity(&changed).is_err());
        }
        evidence["rank_capacities"][1] = sizes(99);
        assert!(rank_capacity(&evidence).is_err());
    }

    #[test]
    fn indexed_worker_measurements_match_their_scheduler_and_tensor_block_axis() {
        let spec = json!({"spec_type":"vllm.v1.kv_cache_interface.FullAttentionSpec","dtype":"torch.bfloat16","page_size_bytes":96});
        let group = json!({"kind":"attention","spec_type":super::super::UNIFORM_SPEC,"dtype":null,
            "spec_page_size_bytes":192,"pool_id":"pool","block_size_tokens":8,"layer_names":["a","b"],
            "layer_classes":{"a":"Attention","b":"Attention"},"layer_specs":{"a":spec,"b":spec}});
        let mut scheduler = json!({"groups":[{"spec_type":spec["spec_type"],"dtype":spec["dtype"],
            "spec_page_size_bytes":96,"pool_id":"pool","block_size_tokens":8,"layer_names":["a","b"],
            "layer_specs":{"a":spec,"b":spec}}]});
        let mut worker = json!({"num_blocks":100,"layer_tensors":{"a":[{"block_axis":0,
            "kernel_block_size_tokens":4,"kernel_block_stride":3,"shape":[299,2],"stride_bytes":[32,2]}]},
            "tensor_allocations":[{"layers":["a"],"block_stride":96}]});
        let groups = [group];
        check_scheduler_groups(&groups, &scheduler).unwrap();
        let measured = [json!({"block_size_tokens":8,"layer_names":["a"]})];
        check_tensor_blocks(&worker, &measured).unwrap();
        scheduler["groups"][0]["block_size_tokens"] = json!(16);
        assert!(check_scheduler_groups(&groups, &scheduler).is_err());
        worker["num_blocks"] = json!(200);
        assert!(check_tensor_blocks(&worker, &measured).is_err());
        assert!(
            measured_rank_capacity(
                &json!({"available_cache_bytes":3201}),
                &json!({"available_cache_bytes":3202}),
                &Value::Null,
                None,
                &mut (None, None)
            )
            .is_err()
        );
    }

    #[test]
    fn scheduler_hash_matches_its_groups_and_explicit_prefix_match_unit() {
        let mut record = json!({"resolved_config":{"cache_config":{"enable_prefix_caching":true}},
            "cache":{"groups":[{"block_size_tokens":64,"spec_type":"vllm.v1.kv_cache_interface.FullAttentionSpec"},
                {"block_size_tokens":64,"spec_type":"vllm.v1.kv_cache_interface.MambaSpec","state":{"mamba_cache_mode":"align"}}],
                "block_size":64,"hash_block_size":64,"prefix_caching":{"block_size":64}}});
        check_scheduler_hash(&record).unwrap();
        record["cache"]["hash_block_size"] = json!(32);
        assert!(check_scheduler_hash(&record).is_err());
        record["resolved_config"]["cache_config"]["prefix_match_unit"] = json!(32);
        check_scheduler_hash(&record).unwrap();
        record["cache"]["groups"][1]["state"]["mamba_cache_mode"] = json!("none");
        assert!(check_scheduler_hash(&record).is_err());
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
