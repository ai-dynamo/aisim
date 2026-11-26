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

/// Bind the portable projection to its recorded sources, independently of worker order.
/// The onboarding verifier separately rehashes the original files.
pub(crate) fn digest(entry: &Value, source: Option<&str>) -> Result<String, AicError> {
    let formal = entry["source"] == "vllm_initialized_cache";
    let sha256 = |s: &str| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit());
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
    // The formal memory collector saves a plan hash and individual file hashes in place of
    // a runtime-observation index. Bind those exact source files and the installed version.
    let mut artifacts = Vec::new();
    if formal {
        for artifact in entry["artifacts"]
            .as_array()
            .filter(|files| !files.is_empty())
            .ok_or_else(invalid)?
        {
            artifacts.push(
                artifact["sha256"]
                    .as_str()
                    .filter(|s| sha256(s))
                    .ok_or_else(invalid)?,
            );
        }
        artifacts.sort_unstable();
    }
    let mut contents = vec![
        Value::from(source),
        Value::from(artifacts),
        if formal {
            entry["backend_version"].clone()
        } else {
            Value::Null
        },
        entry["runtime"].clone(),
        entry["runtime_settings"].clone(),
        serde_json::json!(sorted_rows("worker_observations")),
    ];
    // Compact finalizations retain the measured roster even for a cell with no observed names or layout.
    if let Some(capacities) = sorted_rows("rank_capacities") {
        contents.push(serde_json::json!(capacities));
    }
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
    if probe {
        fn pool(row: &Value) -> Result<(&str, u64), AicError> {
            Ok((
                row["phase"].as_str().ok_or_else(invalid)?,
                row["dp_rank"].as_u64().ok_or_else(invalid)?,
            ))
        }
        let worker_pools: BTreeSet<_> =
            workers.iter().map(|r| pool(r)).collect::<Result<_, _>>()?;
        let mut schedulers = BTreeSet::new();
        for file in files
            .iter()
            .map(|f| &f["evidence"])
            .filter(|r| r["kind"] == "scheduler")
        {
            if !schedulers.insert(pool(file)?) {
                return Err(invalid());
            }
        }
        if schedulers != worker_pools {
            return Err(invalid());
        }
    }
    Ok(true)
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
    // The original formal memory collector measured one cell and omitted its phase name.
    let phase = |v: &Value| match v.as_str() {
        Some("prefill") => Ok("prefill"),
        Some("decode") => Ok("decode"),
        None if v.is_null() && entry["source"] == "vllm_initialized_cache" => Ok(""),
        _ => Err(invalid()),
    };
    let mut slots = BTreeMap::new();
    let mut full_slots = BTreeSet::new();
    let mut sizes = BTreeMap::new();
    for worker in workers {
        let phase = phase(&worker["phase"])?;
        let rank = worker["tp_rank"].as_u64().ok_or_else(invalid)?;
        if tp.is_some_and(|t| rank >= t) {
            return Err(invalid());
        }
        *slots.entry((phase, rank)).or_insert(0_u64) += 1;
        *sizes.entry(phase).or_insert(0_u64) += 1;
        if require_dp {
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
            let phase = phase(&row["phase"])?;
            let rank = row["tp_rank"].as_u64().ok_or_else(invalid)?;
            let dp = row["dp_rank"].as_u64().ok_or_else(invalid)?;
            *expected.entry((phase, rank)).or_insert(0_u64) += 1;
            if !coordinates.insert((phase, rank, dp))
                || width.zip(tp).is_some_and(|(w, t)| t == 0 || dp >= w / t)
            {
                return Err(invalid());
            }
        }
        if expected != slots || (require_dp && coordinates != full_slots) {
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
    fn a_compact_cell_with_no_worker_identity_still_binds_its_rank_capacities() {
        let source = "a".repeat(64);
        let first = json!({"phase":"decode","tp_rank":0,"dp_rank":0,"kv_cache_bytes":256});
        let second = json!({"phase":"decode","tp_rank":1,"dp_rank":0,"kv_cache_bytes":128});
        let mut cell = json!({"rank_capacities":[first,second]});
        let original = digest(&cell, Some(&source)).unwrap();
        cell["rank_capacities"].as_array_mut().unwrap().reverse();
        assert_eq!(digest(&cell, Some(&source)).unwrap(), original);
        cell["rank_capacities"][0]["phase"] = json!("prefill");
        assert_ne!(digest(&cell, Some(&source)).unwrap(), original);
    }
}
