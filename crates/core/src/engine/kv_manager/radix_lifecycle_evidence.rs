// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identical lifecycle probes run against the pre-integration and native caches.

use super::*;
use crate::engine::common::hashing::compute_seq_hash_for_block;
use crate::engine::common::protocols::KvCacheEventSink;
use serde_json::json;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Sink(Mutex<Vec<KvEvent>>);

impl KvCacheEventSink for Sink {
    fn publish(&self, event: KvEvent) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(event);
        Ok(())
    }
}

fn admit(mgr: &mut SglangKvManager, tokens: &[u32]) -> (usize, RadixRequestLease) {
    let mut lease = RadixRequestLease::default();
    let reused = mgr
        .allocate_for_request_lease_capped(tokens, &mut lease, tokens.len() - 1)
        .unwrap();
    (reused, lease)
}

#[test]
fn radix_lifecycle_matches_recorded_behavior() {
    let baseline = std::env::var("AISIM_EVICTION_EVIDENCE_SIDE").as_deref() == Ok("baseline");
    let mut observations = Vec::new();
    for completion in [false, true] {
        let sink = Arc::new(Sink::default());
        let mut mgr = SglangKvManager::new(24, 4, KvEventPublishers::new(Some(sink.clone())), 0);
        let prompt: &[u32] = if completion {
            &[1, 2, 3, 4, 5]
        } else {
            &[1, 2, 3, 4, 5, 6, 7, 8]
        };
        let (_, initial) = admit(&mut mgr, prompt);
        mgr.finish(prompt, initial);
        if completion {
            let (reused, active) = admit(&mut mgr, prompt);
            assert_eq!(reused, 4);
            let other = [9, 10, 11, 12, 13];
            let (_, newer) = admit(&mut mgr, &other);
            mgr.finish(&other, newer);
            mgr.finish(prompt, active);
        }
        let first_event = sink.0.lock().unwrap().len();
        let available = mgr.cache.available_tokens();
        mgr.evict(4);
        let freed_tokens = mgr.cache.available_tokens() - available;
        let removed: Vec<_> = sink.0.lock().unwrap()[first_event..]
            .iter()
            .flat_map(|event| match &event.data {
                KvEventData::Removed { block_hashes } => block_hashes.clone(),
                _ => panic!("eviction must publish only removals"),
            })
            .collect();
        let (reused_tokens, repeat) = admit(&mut mgr, prompt);
        mgr.abort(repeat);
        let hashes = compute_seq_hash_for_block(&compute_block_hash_for_seq(prompt, 4));
        let expected = if completion {
            if baseline {
                vec![hashes[0]]
            } else {
                compute_seq_hash_for_block(&compute_block_hash_for_seq(&[9, 10, 11, 12], 4))
            }
        } else if baseline {
            vec![hashes[1]]
        } else {
            hashes
        };
        assert_eq!(removed, expected);
        assert_eq!(freed_tokens, removed.len() * 4);
        assert_eq!(reused_tokens, if baseline != completion { 4 } else { 0 });
        observations.push(json!({
            "scenario": if completion { "complete_without_new_page" } else { "two_page_leaf" },
            "page_size": 4,
            "capacity_tokens": 24,
            "prompt": prompt,
            "requested_tokens": 4,
            "freed_tokens": freed_tokens,
            "removed_block_hashes": removed,
            "repeat_reused_tokens": reused_tokens,
        }));
        mgr.evict(24);
        assert_eq!(mgr.cache.available_tokens(), 24);
    }
    if let Ok(path) = std::env::var("AISIM_EVICTION_EVIDENCE_OUTPUT") {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&observations).unwrap() + "\n",
        )
        .unwrap();
    }
}

#[test]
#[ignore = "explicit optimized cache-management measurement"]
fn radix_growth_measurement() {
    let mut observations = Vec::new();
    for prompt_len in [64, 4096, 32768] {
        for overlap in [false, true] {
            let growth_pages = if prompt_len == 64 { 2 } else { 128 };
            let repetitions = 50;
            let started = std::time::Instant::now();
            for run in 0..repetitions {
                let mut mgr = SglangKvManager::new(131072, 32, KvEventPublishers::default(), 0);
                let mut tokens: Vec<u32> = (0..prompt_len).map(|i| i + run * 262144).collect();
                let mut alloc = mgr.allocate_for_request(&tokens).unwrap();
                alloc
                    .lease
                    .reserve_page_hashes(prompt_len as usize / 32 + growth_pages);
                mgr.extend_cached_prefix(&tokens, &mut alloc.lease);
                for page in 0..growth_pages {
                    let start = tokens.len() as u32 + run * 262144;
                    tokens.extend(start..start + 32);
                    mgr.extend_allocation(&tokens, &mut alloc.lease).unwrap();
                    if overlap && page == growth_pages / 2 {
                        let other = mgr.allocate_for_request(&tokens).unwrap();
                        let canonical = other.lease.pages().to_vec();
                        mgr.finish(&tokens, other.lease);
                        mgr.extend_cached_prefix(&tokens, &mut alloc.lease);
                        assert_eq!(alloc.lease.pages(), canonical);
                    } else {
                        mgr.extend_cached_prefix(&tokens, &mut alloc.lease);
                    }
                }
                assert_eq!(alloc.lease.cached_tokens(), tokens.len());
                mgr.finish(&tokens, alloc.lease);
                assert_eq!(mgr.cache().protected_size(), 0);
                assert_eq!(mgr.cache().prefix_match_len(&tokens), tokens.len());
                std::hint::black_box(mgr);
            }
            observations.push(json!({
                "prompt_tokens": prompt_len,
                "growth_pages": growth_pages,
                "page_size": 32,
                "overlap": overlap,
                "repetitions": repetitions,
                "elapsed_ns": started.elapsed().as_nanos() as u64,
            }));
        }
    }
    let output = serde_json::to_string_pretty(&observations).unwrap() + "\n";
    if let Ok(path) = std::env::var("AISIM_EVICTION_EVIDENCE_OUTPUT") {
        std::fs::write(path, &output).unwrap();
    }
    println!("{output}");
}
