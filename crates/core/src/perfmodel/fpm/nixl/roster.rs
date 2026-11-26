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

fn sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit())
}

fn file_references(files: &Value, path_key: &str) -> Result<Value, AicError> {
    let mut paths = BTreeSet::new();
    let mut references = Vec::new();
    for file in files.as_array().ok_or_else(invalid)? {
        let path = file[path_key]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(invalid)?;
        let hash = file["sha256"]
            .as_str()
            .filter(|s| sha256(s))
            .ok_or_else(invalid)?;
        let kind = file
            .get("kind")
            .filter(|v| !v.is_null())
            .map(|v| v.as_str().filter(|s| !s.is_empty()).ok_or_else(invalid))
            .transpose()?;
        if !paths.insert(path) {
            return Err(invalid());
        }
        references.push((kind, path, hash));
    }
    references.sort_unstable();
    Ok(serde_json::json!(references))
}

/// Bind the portable projection to its recorded sources, independently of worker order.
/// The onboarding verifier separately rehashes the original files.
pub(crate) fn digest(
    entry: &Value,
    source: Option<&str>,
    inventory: &Value,
    draft: bool,
) -> Result<String, AicError> {
    let formal = entry["source"] == "vllm_initialized_cache";
    let source = source
        .filter(|s| if formal { !s.is_empty() } else { sha256(s) })
        .ok_or_else(invalid)?;
    let sorted_rows = |field: &str| {
        entry[field].as_array().map(|records| {
            let mut rows = records.iter().map(Value::to_string).collect::<Vec<_>>();
            rows.sort_unstable();
            rows
        })
    };
    let mut contents = vec![
        Value::from(source),
        if formal {
            entry["backend_version"].clone()
        } else {
            Value::Null
        },
        entry["runtime"].clone(),
        entry["runtime_settings"].clone(),
        serde_json::json!(sorted_rows("worker_observations")),
        serde_json::json!(sorted_rows("rank_capacities")),
    ];
    let mut files = serde_json::Map::new();
    if draft {
        let index = inventory["observations_index"]
            .as_str()
            .ok_or_else(invalid)?;
        // A draft retains its indexed capture and the frozen launch and model inputs.
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
            files.insert("observations_index".into(), index.clone());
        }
        for key in ["artifacts", "runtime_artifacts", "launch_artifacts"] {
            if key == "artifacts" && entry[key].as_array().is_none_or(Vec::is_empty) {
                return Err(invalid());
            }
            if let Some(refs) = entry.get(key) {
                files.insert(key.into(), file_references(refs, "path")?);
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
    contents.push(Value::Object(files));
    Ok(format!(
        "{:x}",
        Sha256::digest(Value::Array(contents).to_string())
    ))
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
    let formal = entry["source"] == "vllm_initialized_cache"
        && (entry["observer_schema_version"].is_string() || entry["plan_sha256"].is_string());
    let probe = entry["source"] == "runtime_observations"
        && (entry["observation_schema"] == "aisimulate-runtime-observation/v1"
            || entry.pointer("/launch/topology").is_some()
            || entry.pointer("/observations_index/sha256").is_some());
    if !formal && !probe {
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
        if probe
            && entry.get("worker_attention_backends").is_some()
            && entry.pointer("/launch/topology").is_some()
            && entry.get("launch_artifacts").is_some()
            && capacities.is_none()
        {
            return Err(invalid());
        }
        if let Some(capacities) = capacities
            && entry.get("worker_observations").is_none()
        {
            validate(entry, capacities, width, tp, None, true)?;
            if entry["worker_attention_backends"]
                .as_array()
                .is_some_and(|names| names.len() != capacities.len())
            {
                return Err(invalid());
            }
        }
        return Ok(false);
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
        assert!(!validate_source(&entry, Capture::References, None, None).unwrap());
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
        let original = digest(&cell, Some(&source), &json!({}), false).unwrap();
        cell["rank_capacities"].as_array_mut().unwrap().reverse();
        assert_eq!(
            digest(&cell, Some(&source), &json!({}), false).unwrap(),
            original
        );
        cell["rank_capacities"][0]["phase"] = json!("prefill");
        assert_ne!(
            digest(&cell, Some(&source), &json!({}), false).unwrap(),
            original
        );
    }
}
