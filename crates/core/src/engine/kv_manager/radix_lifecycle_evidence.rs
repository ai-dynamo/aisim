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
    // Cache A1/A2, then B; admitting A1 splits and locks only the first page.
    // Keep the active lease alive while applying one page of eviction pressure.
    let sink = Arc::new(Sink::default());
    let mut mgr = SglangKvManager::new(24, 4, KvEventPublishers::new(Some(sink.clone())), 0);
    let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let b = [10, 11, 12, 13, 14];
    let (_, initial_a) = admit(&mut mgr, &a);
    mgr.finish(&a, initial_a);
    let (_, initial_b) = admit(&mut mgr, &b);
    mgr.finish(&b, initial_b);
    let (reused_prefix, active) = admit(&mut mgr, &[1, 2, 3, 4, 99]);
    assert_eq!(reused_prefix, 4);
    assert_eq!(active.cached_tokens(), 4);
    assert_eq!(mgr.cache.node(active.last_node()).lock_ref, 1);
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
    let (repeat_a, repeat) = admit(&mut mgr, &a);
    mgr.abort(repeat);
    let (repeat_b, repeat) = admit(&mut mgr, &b);
    mgr.abort(repeat);
    let a_hashes = compute_seq_hash_for_block(&compute_block_hash_for_seq(&a, 4));
    let b_hashes = compute_seq_hash_for_block(&compute_block_hash_for_seq(&b, 4));
    assert_eq!(
        removed,
        if baseline {
            vec![a_hashes[1]]
        } else {
            b_hashes
        }
    );
    assert_eq!(freed_tokens, 4);
    assert_eq!(repeat_a, if baseline { 4 } else { 8 });
    assert_eq!(repeat_b, if baseline { 4 } else { 0 });
    observations.push(json!({
        "scenario": "match_and_lock_compressed_prefix",
        "page_size": 4,
        "capacity_tokens": 24,
        "prompt_a": a,
        "prompt_b": b,
        "active_prompt": [1, 2, 3, 4, 99],
        "active_reused_tokens": reused_prefix,
        "requested_tokens": 4,
        "freed_tokens": freed_tokens,
        "removed_block_hashes": removed,
        "repeat_a_reused_tokens": repeat_a,
        "repeat_b_reused_tokens": repeat_b,
    }));
    mgr.abort(active);
    mgr.evict(24);
    assert_eq!(mgr.cache.available_tokens(), 24);
    if let Ok(path) = std::env::var("AISIM_EVICTION_EVIDENCE_OUTPUT") {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&observations).unwrap() + "\n",
        )
        .unwrap();
    }
}

#[test]
fn radix_completion_ancestor_refresh_matches_recorded_behavior() {
    let baseline = std::env::var("AISIM_EVICTION_EVIDENCE_SIDE").as_deref() == Ok("baseline");
    let mut observations = Vec::new();
    // Completing P -> Q with a private page X refreshes the retained path.
    // Keep a newer, unrelated B locked while removing Q/X, then compare P/B.
    let sink = Arc::new(Sink::default());
    let mut mgr = SglangKvManager::new(24, 4, KvEventPublishers::new(Some(sink.clone())), 0);
    let p = [1, 2, 3, 4];
    let pq = [1, 2, 3, 4, 5, 6, 7, 8];
    let pqx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let b = [20, 21, 22, 23];
    for prompt in [&p[..], &pq[..]] {
        let (_, lease) = admit(&mut mgr, prompt);
        mgr.finish(prompt, lease);
    }
    // Legacy growth coalesces P/Q. A real partial match exposes the requested
    // P -> Q topology before admitting the lease with its private X page.
    let mut prefix = RadixRequestLease::default();
    assert_eq!(
        mgr.allocate_for_request_lease_capped(&p, &mut prefix, p.len())
            .unwrap(),
        4
    );
    let p_node = prefix.last_node();
    mgr.abort(prefix);
    let (reused, active) = admit(&mut mgr, &pqx);
    assert_eq!(reused, 8);
    assert_eq!(mgr.cache.node(active.last_node()).parent, Some(p_node));
    let (_, initial_b) = admit(&mut mgr, &b);
    mgr.finish(&b, initial_b);
    let mut locked_b = RadixRequestLease::default();
    assert_eq!(
        mgr.allocate_for_request_lease_capped(&b, &mut locked_b, b.len())
            .unwrap(),
        4
    );
    mgr.finish(&pqx, active);
    let first_event = sink.0.lock().unwrap().len();
    mgr.evict(8);
    let qx_removed: Vec<_> = sink.0.lock().unwrap()[first_event..]
        .iter()
        .flat_map(|event| match &event.data {
            KvEventData::Removed { block_hashes } => block_hashes.clone(),
            _ => panic!("eviction must publish only removals"),
        })
        .collect();
    let hashes = compute_seq_hash_for_block(&compute_block_hash_for_seq(&pqx, 4));
    assert_eq!(
        qx_removed,
        if baseline {
            vec![hashes[1], hashes[2]]
        } else {
            vec![hashes[2], hashes[1]]
        }
    );
    mgr.abort(locked_b);
    let first_event = sink.0.lock().unwrap().len();
    mgr.evict(4);
    let removed: Vec<_> = sink.0.lock().unwrap()[first_event..]
        .iter()
        .flat_map(|event| match &event.data {
            KvEventData::Removed { block_hashes } => block_hashes.clone(),
            _ => panic!("eviction must publish only removals"),
        })
        .collect();
    let b_hashes = compute_seq_hash_for_block(&compute_block_hash_for_seq(&b, 4));
    assert_eq!(removed, b_hashes);
    let mut repeat = RadixRequestLease::default();
    let reused = mgr
        .allocate_for_request_lease_capped(&p, &mut repeat, p.len())
        .unwrap();
    assert_eq!(reused, 4);
    mgr.abort(repeat);
    observations.push(json!({
        "scenario": "complete_with_new_page_refreshes_ancestors",
        "page_size": 4,
        "capacity_tokens": 24,
        "prompt_p": p,
        "prompt_pq": pq,
        "prompt_pqx": pqx,
        "prompt_b": b,
        "locked_b_eviction_hashes": qx_removed,
        "removed_block_hashes": removed,
        "repeat_reused_tokens": reused,
    }));
    mgr.evict(24);
    assert_eq!(mgr.cache.available_tokens(), 24);
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
    for prompt_len in [64, 4096] {
        for growth_pages in [2, 128, 512, 2048] {
            for overlap in [false, true] {
                let repetitions = 20;
                let mut growth_elapsed = std::time::Duration::ZERO;
                let started = std::time::Instant::now();
                for run in 0..repetitions {
                    let mut mgr = SglangKvManager::new(131072, 32, KvEventPublishers::default(), 0);
                    let mut tokens: Vec<u32> = (0..prompt_len).map(|i| i + run * 262144).collect();
                    let mut alloc = mgr.allocate_for_request(&tokens).unwrap();
                    alloc
                        .lease
                        .reserve_page_hashes(prompt_len as usize / 32 + growth_pages);
                    mgr.extend_cached_prefix(&tokens, &mut alloc.lease);
                    let growth_started = std::time::Instant::now();
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
                    growth_elapsed += growth_started.elapsed();
                    assert_eq!(alloc.lease.cached_tokens(), tokens.len());
                    mgr.finish(&tokens, alloc.lease);
                    assert_eq!(mgr.cache().prefix_match_len(&tokens), tokens.len());
                    mgr.evict(131072);
                    assert_eq!(mgr.cache().available_tokens(), 131072);
                    std::hint::black_box(mgr);
                }
                observations.push(json!({
                    "prompt_tokens": prompt_len,
                    "growth_pages": growth_pages,
                    "page_size": 32,
                    "overlap": overlap,
                    "repetitions": repetitions,
                    "elapsed_ns": started.elapsed().as_nanos() as u64,
                    "growth_elapsed_ns": growth_elapsed.as_nanos() as u64,
                }));
            }
        }
    }
    let output = serde_json::to_string_pretty(&observations).unwrap() + "\n";
    if let Ok(path) = std::env::var("AISIM_EVICTION_EVIDENCE_OUTPUT") {
        std::fs::write(path, &output).unwrap();
    }
    println!("{output}");
}

#[test]
#[ignore = "explicit optimized warm-readmission and checkpoint measurement"]
fn radix_warm_readmission_measurement() {
    let mut observations = Vec::new();
    for growth_pages in [2, 128, 512, 2048] {
        let capacity = 131072;
        let repetitions = 100;
        let mut mgr = SglangKvManager::new(capacity, 32, KvEventPublishers::default(), 0);
        let mut tokens: Vec<u32> = (0..64).collect();
        let mut alloc = mgr.allocate_for_request(&tokens).unwrap();
        mgr.extend_cached_prefix(&tokens, &mut alloc.lease);
        for _ in 0..growth_pages {
            let start = tokens.len() as u32;
            tokens.extend(start..start + 32);
            mgr.extend_allocation(&tokens, &mut alloc.lease).unwrap();
            mgr.extend_cached_prefix(&tokens, &mut alloc.lease);
        }
        mgr.finish(&tokens, alloc.lease);
        let nodes = mgr.cache().num_nodes();
        let checkpoint_started = std::time::Instant::now();
        for _ in 0..repetitions {
            std::hint::black_box(mgr.cache.admission_checkpoint());
        }
        let checkpoint_ns = checkpoint_started.elapsed().as_nanos() as u64;
        let mut warm_admission_ns = 0;
        for _ in 0..repetitions {
            let started = std::time::Instant::now();
            let admission = mgr.begin_admission();
            let alloc = mgr.allocate_for_request(&tokens).unwrap();
            mgr.commit_admission(admission);
            warm_admission_ns += started.elapsed().as_nanos() as u64;
            assert_eq!(alloc.prefix_len, tokens.len());
            mgr.finish(&tokens, alloc.lease);
        }
        assert_eq!(mgr.cache().num_nodes(), nodes);
        mgr.evict(capacity);
        assert_eq!(mgr.cache().available_tokens(), capacity);
        observations.push(json!({"growth_pages": growth_pages, "page_size": 32,
            "prompt_tokens": tokens.len(), "nodes": nodes, "repetitions": repetitions,
            "checkpoint_ns": checkpoint_ns, "warm_admission_ns": warm_admission_ns}));
    }
    let output = serde_json::to_string_pretty(&observations).unwrap() + "\n";
    if let Ok(path) = std::env::var("AISIM_EVICTION_EVIDENCE_OUTPUT") {
        std::fs::write(path, &output).unwrap();
    }
    println!("{output}");
}
