// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::Worker;
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

/// The older compact projection retained TP and phase but omitted DP coordinates. Its per-TP
/// multiplicity is still known. Current projections retain all three coordinates.
pub(super) fn validate(
    entry: &Value,
    width: Option<u64>,
    tp: Option<u64>,
    capacities: Option<&[Value]>,
    require_dp: bool,
) -> Result<(), AicError> {
    let workers = entry["worker_observations"]
        .as_array()
        .filter(|r| !r.is_empty())
        .ok_or_else(invalid)?;
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
    if width.is_some_and(|w| sizes.values().any(|&n| n != w))
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
pub(super) fn can_pair(
    prefill: &[Worker],
    decode: &[Worker],
    compatible: impl Fn(&Worker, &Worker) -> bool,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
