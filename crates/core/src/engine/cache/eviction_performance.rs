// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::engine::belady::input_sequence_hashes;
use uuid::Uuid;

fn oracle(prompts: &[Vec<u32>]) -> BeladyOracle {
    BeladyOracle::new(
        prompts
            .iter()
            .enumerate()
            .map(|(rank, tokens)| {
                (
                    Uuid::from_u128(rank as u128 + 1),
                    input_sequence_hashes(tokens, 1),
                )
            })
            .collect(),
    )
    .unwrap()
}

fn insert(cache: &mut RadixCache, tokens: &[u32]) -> NodeId {
    let prefix = cache.prefix_match_len(tokens);
    let (_, anchor) = cache.match_prefix(tokens);
    let mut pages = cache.collect_path_pages(anchor);
    pages.extend(
        cache
            .page_pool
            .allocate_pages(tokens.len() - prefix)
            .unwrap(),
    );
    cache
        .insert_page_hashes_from_node(
            anchor,
            prefix,
            cache.page_hashes(tokens).into(),
            &pages,
            false,
        )
        .last_node
}

fn assert_winner(cache: &mut RadixCache) {
    let expected = cache
        .tree
        .full_device_eviction_candidates()
        .map(|candidate| match &cache.belady {
            Some(belady) => belady.priority(candidate).0,
            None => (Reverse(0), candidate.last_access_counter, candidate.node_id),
        })
        .min();
    cache.refresh_forecast();
    let actual = cache.next_victim();
    assert_eq!(actual, expected.map(|priority| (priority.2, priority.1)));
    if let Some((node, _)) = actual {
        cache.refresh_candidate(node);
    }
}

#[rstest::rstest]
fn pressure_does_not_visit_unchanged_leaves(
    #[values(false, true)] belady: bool,
    #[values(64, 1024, 8192)] leaves: usize,
) {
    let mut cache = RadixCache::new(leaves, 1);
    if belady {
        cache.set_belady_oracle(oracle(&[]));
    }
    for token in 0..leaves as u32 {
        insert(&mut cache, &[token]);
    }
    let before = cache.candidates.candidate_queries;
    for offset in 0..256.min(leaves) {
        assert_eq!(cache.evict(1), (1, vec![KvPageId(offset)]));
        insert(&mut cache, &[(leaves + offset) as u32]);
        assert_eq!(cache.available_tokens(), 0);
    }
    assert!(cache.candidates.candidate_queries - before <= 4 * 256.min(leaves));
    assert_eq!(cache.candidates.by_node.len(), leaves);
    assert_winner(&mut cache);
}

#[rstest::rstest]
fn candidate_index_follows_native_mutations_and_snapshots(#[values(false, true)] belady: bool) {
    let mut cache = RadixCache::new(32, 1);
    if belady {
        cache.set_belady_oracle(oracle(&[vec![1], vec![9], vec![1, 2, 4], vec![1, 2, 3]]));
    }
    let old_leaf = insert(&mut cache, &[1, 2, 3]);
    insert(&mut cache, &[9]);
    assert_winner(&mut cache);
    cache.match_prefix(&[1, 2, 3]);
    assert_winner(&mut cache);
    // A partial match splits a leaf; the original handle now names its suffix.
    let (_, prefix) = cache.match_prefix(&[1, 2]);
    cache.inc_lock_ref(prefix);
    insert(&mut cache, &[1, 2, 4]);
    insert(&mut cache, &[1, 2, 3, 5]);
    cache.inc_lock_ref(old_leaf);
    assert_winner(&mut cache);
    let mut snapshot = cache.admission_checkpoint();
    for branch in [&mut cache, &mut snapshot] {
        while branch.evictable_size() > 0 {
            assert_winner(branch);
            assert!(branch.evict(1).0 > 0);
        }
        assert_winner(branch);
        // Parents exposed while locked become eligible when their anchor unlocks.
        branch.dec_lock_ref(old_leaf);
        assert_winner(branch);
        branch.evict(32);
        branch.dec_lock_ref(prefix);
        assert_winner(branch);
        branch.evict(32);
        assert_eq!(branch.available_tokens(), 32);
        assert!(branch.candidates.by_node.is_empty());
        assert!(branch.candidates.by_hash.is_empty());
    }
}

#[test]
fn forecast_retirement_refreshes_buried_victims_in_each_checkpoint() {
    let prompts: Vec<_> = (0..1024).map(|token| vec![token]).collect();
    let forecast = oracle(&prompts);
    let mut cache = RadixCache::new(1024, 1);
    cache.set_belady_oracle(forecast.clone());
    for prompt in &prompts {
        insert(&mut cache, prompt);
    }
    assert_eq!(cache.evict(1).1, vec![KvPageId(1023)]);
    let mut checkpoint = cache.admission_checkpoint();
    forecast.retire_requests([Uuid::from_u128(1)]);
    for branch in [&mut cache, &mut checkpoint] {
        let before = branch.candidates.candidate_queries;
        assert_eq!(branch.evict(1).1, vec![KvPageId(0)]);
        assert_eq!(branch.candidates.candidate_queries - before, 2);
        assert_winner(branch);
    }
}

#[rstest::rstest]
fn compressed_drain_copies_linear_buffers(
    #[values(false, true)] checkpoint: bool,
    #[values(33, 1024, 8192)] pages: usize,
) {
    let mut cache = RadixCache::new(pages, 1);
    cache.set_belady_oracle(oracle(&[]));
    let tokens: Vec<_> = (0..pages as u32).collect();
    insert(&mut cache, &tokens);
    let saved = checkpoint.then(|| cache.admission_checkpoint());
    PageKey::take_owned_key_atoms();
    SPLIT_VALUE_COPIES.with(|copies| copies.set(0));
    for remaining in (0..pages).rev() {
        assert_eq!(cache.evict(1), (1, vec![KvPageId(remaining)]));
        if remaining > 0 {
            let candidate = cache.tree.full_device_eviction_candidates().next().unwrap();
            assert_eq!(candidate.value.as_slice().len(), remaining);
            assert!(candidate.value.backing_len <= 2 * remaining);
        }
    }
    // Per split: one suffix atom and three one-atom edge keys (two inserts,
    // one removal). Geometric prefix compactions copy fewer than N atoms.
    let copied = PageKey::take_owned_key_atoms();
    assert!(
        copied <= 5 * pages,
        "copied {copied} key atoms while draining {pages} pages"
    );
    assert!(SPLIT_VALUE_COPIES.with(|copies| copies.get()) <= 2 * pages);
    assert_eq!(cache.available_tokens(), pages);
    if let Some(saved) = saved {
        assert_eq!(saved.prefix_match_len(&tokens), pages);
        assert_eq!(saved.available_tokens(), 0);
    }
}

#[test]
fn shared_split_keys_compact_without_pinning_freed_tails() {
    let mut key: PageKey = (0..1025).map(LocalBlockHash).collect::<Vec<_>>().into();
    let saved = key.clone();
    for remaining in (1..1025).rev() {
        let (head, tail) = key.split_at(remaining);
        assert_eq!(tail.as_ref(), &[LocalBlockHash(remaining as u64)]);
        assert!(head.hashes.len() <= 2 * head.end);
        key = head;
    }
    assert_eq!(saved.as_ref().len(), 1025);
    assert_eq!(key.as_ref(), &[LocalBlockHash(0)]);
}
