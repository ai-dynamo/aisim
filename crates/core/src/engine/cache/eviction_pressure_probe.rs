// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identical probe for pinned adapter revisions. Setup is outside each timer;
//! assertions, native eviction, allocator reuse, and replacement inserts are in it.

use super::*;
use crate::engine::belady::input_sequence_hashes;
use std::time::Instant;
use uuid::Uuid;

fn insert_one(cache: &mut RadixCache, token: u32) {
    let page = cache.page_pool.allocate(1).unwrap();
    cache.insert(&[token], &page);
}

#[test]
#[ignore = "manual paired optimized eviction measurement"]
fn eviction_pressure_measurement() {
    let mut records = Vec::new();
    for leaves in [64, 1024, 8192] {
        for policy in ["lru", "belady", "belady_retirement"] {
            let iterations = 256.min(leaves - 1);
            let mut cache = RadixCache::new(leaves, 1);
            let forecast = match policy {
                "lru" => None,
                "belady" => Some(BeladyOracle::new(Vec::new()).unwrap()),
                _ => Some(
                    BeladyOracle::new(
                        (0..leaves + iterations + 1)
                            .map(|token| {
                                (
                                    Uuid::from_u128(token as u128 + 1),
                                    input_sequence_hashes(&[token as u32], 1),
                                )
                            })
                            .collect(),
                    )
                    .unwrap(),
                ),
            };
            if let Some(forecast) = &forecast {
                cache.set_belady_oracle(forecast.clone());
            }
            for token in 0..leaves as u32 {
                insert_one(&mut cache, token);
            }
            // Establish a prior selection before a remote retirement changes
            // the next victim from the farthest-ranked tail to a buried entry.
            if policy == "belady_retirement" {
                assert_eq!(cache.evict(1).1, vec![KvPageId(leaves - 1)]);
                insert_one(&mut cache, leaves as u32);
            }
            let started = Instant::now();
            for offset in 0..iterations {
                if policy == "belady_retirement" {
                    forecast
                        .as_ref()
                        .unwrap()
                        .retire_requests([Uuid::from_u128(offset as u128 + 1)]);
                }
                assert_eq!(cache.evict(1), (1, vec![KvPageId(offset)]));
                insert_one(&mut cache, (leaves + offset + 1) as u32);
                assert_eq!(cache.available_tokens(), 0);
            }
            records.push(serde_json::json!({
                "case": policy,
                "resident_leaves": leaves,
                "iterations": iterations,
                "evicted_pages": iterations,
                "replacement_inserts": iterations,
                "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
            }));
        }
    }
    for pages in [256, 2048, 8192] {
        for checkpoint in [false, true] {
            let mut cache = RadixCache::new(pages, 1);
            cache.set_belady_oracle(BeladyOracle::new(Vec::new()).unwrap());
            let tokens: Vec<_> = (0..pages as u32).collect();
            let indices = cache.page_pool.allocate(pages).unwrap();
            cache.insert(&tokens, &indices);
            let saved = checkpoint.then(|| cache.admission_checkpoint());
            let started = Instant::now();
            let (freed, victims) = cache.evict(pages);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(freed, pages);
            assert_eq!(victims, (0..pages).rev().map(KvPageId).collect::<Vec<_>>());
            assert_eq!(cache.available_tokens(), pages);
            if let Some(saved) = saved {
                assert_eq!(saved.prefix_match_len(&tokens), pages);
                assert_eq!(saved.available_tokens(), 0);
            }
            records.push(serde_json::json!({
                "case": "belady_compressed_drain",
                "pages": pages,
                "checkpoint": checkpoint,
                "evicted_pages": victims.len(),
                "elapsed_ms": elapsed_ms,
            }));
        }
    }
    let output = std::env::var_os("AISIM_PRESSURE_OUTPUT").expect("set AISIM_PRESSURE_OUTPUT");
    std::fs::write(output, serde_json::to_vec_pretty(&records).unwrap()).unwrap();
}
