// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The same test probe runs on both sides of the native-radix integration.
//! Physical page IDs are comparable; native and simulator node IDs are not.

use super::*;
use crate::engine::belady::input_sequence_hashes;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

#[derive(Deserialize)]
struct Scenarios {
    seed: u32,
    scenarios: Vec<Scenario>,
}

#[derive(Deserialize)]
struct Scenario {
    id: String,
    policy: String,
    page_size: usize,
    capacity_pages: usize,
    forecast: Vec<Vec<u32>>,
    operations: Vec<Operation>,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Operation {
    Insert { pages: Vec<u32> },
    Touch { pages: Vec<u32> },
    Lock { pages: Vec<u32> },
    Unlock { pages: Vec<u32> },
    Evict { tokens: usize },
    Retire { request: usize },
}

fn observe(seed: u32, scenario: &Scenario) -> Value {
    let page_size = scenario.page_size;
    let tokens = |pages: &[u32]| -> Vec<u32> {
        pages
            .iter()
            .flat_map(|page| {
                (0..page_size as u32)
                    .map(move |offset| seed * 65536 + page * page_size as u32 + offset)
            })
            .collect()
    };
    let oracle = BeladyOracle::new(
        scenario
            .forecast
            .iter()
            .enumerate()
            .map(|(index, pages)| {
                (
                    Uuid::from_u128(index as u128 + 1),
                    input_sequence_hashes(&tokens(pages), page_size),
                )
            })
            .collect(),
    )
    .unwrap();
    let mut cache = RadixCache::new(scenario.capacity_pages * page_size, page_size);
    if scenario.policy == "belady" {
        cache.set_belady_oracle(oracle.clone());
    } else {
        assert_eq!(scenario.policy, "lru");
    }
    let mut physical_prefixes = BTreeMap::new();
    let mut locks = BTreeMap::new();
    let mut probes = BTreeSet::new();
    let mut evictions = Vec::new();
    for (step, operation) in scenario.operations.iter().enumerate() {
        match operation {
            Operation::Insert { pages } => {
                let key = tokens(pages);
                let matched = cache.prefix_match_len(&key) / page_size;
                let mut indices = Vec::new();
                for end in 1..=matched {
                    let start = physical_prefixes[&pages[..end]] * page_size;
                    indices.extend(start..start + page_size);
                }
                let allocated = cache
                    .page_pool
                    .allocate((pages.len() - matched) * page_size)
                    .unwrap();
                for (index, chunk) in allocated.chunks_exact(page_size).enumerate() {
                    physical_prefixes
                        .insert(pages[..matched + index + 1].to_vec(), chunk[0] / page_size);
                }
                indices.extend(allocated);
                cache.insert(&key, &indices);
                probes.insert(pages.clone());
                assert_eq!(cache.prefix_match_len(&key), key.len());
            }
            Operation::Touch { pages } | Operation::Lock { pages } => {
                let key = tokens(pages);
                let (matched, node) = cache.match_prefix(&key);
                assert_eq!(matched, key.len());
                if matches!(operation, Operation::Lock { .. }) {
                    cache.inc_lock_ref(node);
                    assert!(locks.insert(pages.clone(), node).is_none());
                }
            }
            Operation::Unlock { pages } => cache.dec_lock_ref(locks.remove(pages).unwrap()),
            Operation::Retire { request } => {
                oracle.retire_requests([Uuid::from_u128(*request as u128 + 1)]);
            }
            Operation::Evict { tokens: requested } => {
                let before = cache.available_tokens();
                let (freed_tokens, victims) = cache.evict(*requested);
                let victim_page_ids: Vec<_> = victims.iter().map(|page| page.index()).collect();
                assert_eq!(freed_tokens, victims.len() * page_size);
                assert_eq!(cache.available_tokens(), before + freed_tokens);
                assert_eq!(victims.len(), victims.iter().collect::<BTreeSet<_>>().len());
                let requested_pages = requested.div_ceil(page_size);
                evictions.push(json!({
                    "operation_index": step,
                    "requested_tokens": requested,
                    "requested_pages": requested_pages,
                    "freed_tokens": freed_tokens,
                    "freed_pages": victims.len(),
                    "overshoot_pages": victims.len().saturating_sub(requested_pages),
                    "victim_page_ids": victim_page_ids,
                    "available_tokens": cache.available_tokens(),
                    "prefix_matches": probes.iter().map(|pages| json!({
                        "pages": pages,
                        "matched_tokens": cache.prefix_match_len(&tokens(pages)),
                    })).collect::<Vec<_>>(),
                }));
            }
        }
    }
    assert!(locks.is_empty(), "fixture left a live lock");
    assert_eq!(cache.available_tokens(), cache.total_tokens());
    json!({"id": scenario.id, "evictions": evictions})
}

#[test]
fn eviction_selection_matches_recorded_evidence() {
    let scenarios: Scenarios =
        serde_json::from_str(include_str!("eviction-selection-scenarios.json")).unwrap();
    let observed: Vec<_> = scenarios
        .scenarios
        .iter()
        .map(|scenario| observe(scenarios.seed, scenario))
        .collect();
    if let Ok(path) = std::env::var("AISIM_EVICTION_EVIDENCE_OUTPUT") {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&observed).unwrap() + "\n",
        )
        .unwrap();
    }
    let expected: Value =
        serde_json::from_str(include_str!("eviction-selection-comparison.json")).unwrap();
    let side = std::env::var("AISIM_EVICTION_EVIDENCE_SIDE").unwrap_or("candidate".into());
    assert!(matches!(side.as_str(), "baseline" | "candidate"));
    assert_eq!(expected["seed"], json!(scenarios.seed));
    assert_eq!(json!(observed), expected[side]["scenarios"]);
}
