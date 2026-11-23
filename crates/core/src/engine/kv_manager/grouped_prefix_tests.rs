// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rule and coordinator cases of the grouped prefix cache, with the vLLM
//! 0.31.0, 0.32.0 and 0.33.0 literals re-derived from the pinned trees, the
//! 0.33.0 ones at `4ba787ec9959e566d3b587a796dff07673324da5`. Some cases are
//! ported from the vLLM tests at `db9527a46873454610df6dbedf79a36d6bf1a7f6` and
//! `5df63a7203cdb9328d824a27b682e77c7b9928be`. Copyright contributors to the
//! vLLM project. See THIRD_PARTY_NOTICES.md. Apache-2.0.
//!
//! - https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/core/test_mamba_align_chunk_split.py

use super::*;
use crate::engine::cache::vllm_block_pool::CacheKey;
use crate::engine::common::hashing::SequenceHash;
use crate::engine::kv_manager::grouped::{GroupLease, GroupedBlockPool, GroupedKvPool};
use crate::engine::kv_manager::vllm_backend::BlockRequestLease;
use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};
use uuid::Uuid;

const BUDGET: usize = 8192;
const GLM_BLOCK: u32 = 2176;

fn group(
    name: &str,
    kind: FpmCacheKind,
    block: u32,
    window: Option<u32>,
    state: Option<(bool, u32)>,
) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind,
        num_layers: 1,
        block_size_tokens: block,
        page_size_bytes: 1,
        sliding_window: window,
        blocks_per_request: state.map(|_| 1),
        prefix_cacheable: Some(state.is_none_or(|(cacheable, _)| cacheable)),
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: state.and_then(|(cacheable, ckpt)| cacheable.then_some(ckpt)),
        prefill_checkpoint_alignment: state
            .and_then(|(cacheable, ckpt)| (cacheable && ckpt == 1).then_some(16)),
    }
}

fn attention(name: &str, block: u32, window: Option<u32>) -> FpmCacheGroup {
    group(name, FpmCacheKind::Attention, block, window, None)
}

fn state(name: &str, block: u32, cacheable: bool, checkpoint: bool) -> FpmCacheGroup {
    group(
        name,
        FpmCacheKind::State,
        block,
        None,
        Some((cacheable, u32::from(checkpoint))),
    )
}

/// The six GLM-5.3-Flash TP 4 cache groups that the vLLM 0.33.0 runtime
/// probe records (fixture `glm53flash_vllm_0330_probe`): MLA attention and
/// four KDA `MambaSpec` states with 2,176-token blocks, and a 4-token
/// `CircularBufferSpec` ring. The tests use them as hand-made profiles of
/// vLLM 0.31.0, 0.32.0 and 0.33.0, the versions that the rule table holds. The
/// helper lists them as MLA, ring, KDA. The record lists the four Mamba
/// groups first, then MLA, then the ring. With `checkpoint`, G2 to G5
/// carry `prefill_checkpoint_blocks: Some(1)` and alignment 16.
fn groups_glm(checkpoint: bool) -> Vec<FpmCacheGroup> {
    let mut groups = vec![
        attention("mla", GLM_BLOCK, None),
        state("kpool_tail", 4, false, false),
    ];
    for i in 0..4 {
        groups.push(state(&format!("kda_{i}"), GLM_BLOCK, true, checkpoint));
    }
    groups
}

fn rules_glm_hash(
    version: VllmRuleVersion,
    checkpoint: bool,
    hash_block: usize,
) -> GroupedPrefixRules {
    GroupedPrefixRules::new(&groups_glm(checkpoint), hash_block, version).unwrap()
}

fn rules_glm(version: VllmRuleVersion, checkpoint: bool) -> GroupedPrefixRules {
    rules_glm_hash(version, checkpoint, GLM_BLOCK as usize)
}

fn rules(groups: &[FpmCacheGroup], hash_block: usize) -> GroupedPrefixRules {
    GroupedPrefixRules::new(groups, hash_block, VllmRuleVersion::V0_31_0).unwrap()
}

/// A block pool with `blocks` pool blocks for `groups`.
fn kv_pool(groups: &[FpmCacheGroup], blocks: u64) -> GroupedKvPool {
    GroupedKvPool::with_prefix_cache(groups.to_vec(), blocks, 1)
}

/// A lease with chained hashes of the tokens `0..tokens` at `hash_block`.
fn lease(tokens: usize, hash_block: usize) -> BlockRequestLease {
    let tokens: Vec<u32> = (0..tokens as u32).collect();
    let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
    lease.configure_prefix_hashes(&tokens, hash_block);
    lease
}

fn hash_at(hash_block: usize, tokens: usize) -> SequenceHash {
    lease(tokens, hash_block).prefix_hash_at(tokens).unwrap()
}

fn key(group: u16, hash_block: usize, tokens: usize) -> CacheKey {
    CacheKey::Grouped {
        group,
        hash: hash_at(hash_block, tokens),
    }
}

/// Publish one block under `key` and leave it in the inactive LRU, as a
/// finished request leaves its cached blocks.
fn seed(pool: &mut GroupedBlockPool, key: CacheKey) {
    let mut reservation = pool.reserve(&[], 1).unwrap();
    let id = pool.take_private(&mut reservation);
    assert!(pool.cache(id, key));
    pool.release(id);
}

fn seed_many(pool: &mut GroupedBlockPool, group: u16, hash_block: usize, ends: &[usize]) {
    for &end in ends {
        seed(pool, key(group, hash_block, end));
    }
}

fn seed_states(pool: &mut GroupedBlockPool, groups: &[u16], hash_block: usize, end: usize) {
    for &group in groups {
        seed(pool, key(group, hash_block, end));
    }
}

/// Seed the key at `end`, inside a block of `block` tokens, in each of
/// `groups`, listed under its block as `finalize` lists it.
fn seed_inside(
    pool: &mut GroupedBlockPool,
    groups: &[u16],
    hash_block: usize,
    block: usize,
    end: usize,
) {
    let in_block = key_in_block(&lease(end, hash_block), hash_block, block, end);
    assert!(in_block.is_some(), "{end} is a block end");
    for &group in groups {
        let mut reservation = pool.reserve(&[], 1).unwrap();
        let id = pool.take_private(&mut reservation);
        assert!(pool.cache_at(id, key(group, hash_block, end), in_block));
        pool.release(id);
    }
}

fn hit_exists(pool: &GroupedBlockPool, group: u16, hash_block: usize, tokens: usize) -> bool {
    pool.key_hit(group, hash_at(hash_block, tokens)).is_some()
}

fn state_lease(lease: &GroupedBlockLease, group: usize) -> &GroupStateLease {
    match &lease.groups[group] {
        GroupLease::State(state) => state,
        other => panic!("group {group} is not a state lease: {other:?}"),
    }
}

/// The release rules of the previous blocks of a state group, oldest first.
fn previous_rules(lease: &GroupedBlockLease, group: usize) -> Vec<PreviousRelease> {
    state_lease(lease, group)
        .previous
        .iter()
        .map(|&(_, rule)| rule)
        .collect()
}

/// The blocks that every group but the first holds: the ring and the
/// state groups of `groups_glm`.
fn state_and_ring_blocks(lease: &GroupedBlockLease) -> usize {
    lease.groups[1..]
        .iter()
        .map(|g| match g {
            GroupLease::Attention { .. } => 0,
            GroupLease::Ring { copy } => usize::from(copy.is_some()),
            GroupLease::State(s) => {
                [
                    s.running,
                    s.hit.map(|h| h.0),
                    s.copy.map(|c| c.0),
                    s.checkpoint.map(|c| c.0),
                ]
                .iter()
                .flatten()
                .count()
                    + s.previous.len()
            }
        })
        .sum()
}

fn attention_copies(lease: &GroupedBlockLease, group: usize) -> usize {
    match &lease.groups[group] {
        GroupLease::Attention { copies, .. } => copies.iter().flatten().count(),
        other => panic!("group {group} is not an attention lease: {other:?}"),
    }
}

fn split(
    rules: &GroupedPrefixRules,
    start: usize,
    prompt: usize,
    known: usize,
    desired: usize,
    junction: usize,
) -> usize {
    rules.align_split(start, prompt, known, desired, junction, BUDGET)
}

/// Run a whole prefill of `prompt` tokens from `hit`: the chunks of
/// `align_split` at `budget`, one step each. Returns the chunk ends.
fn prefill(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
    lease: &mut GroupedBlockLease,
    request: &BlockRequestLease,
    prompt: usize,
    hit: &GroupedHit,
    budget: usize,
) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut computed = hit.tokens;
    while computed < prompt {
        coordinator.begin_step();
        let chunk = coordinator.rules().align_split(
            computed,
            prompt,
            prompt,
            budget.min(prompt - computed),
            hit.tokens + hit.junction_tokens,
            budget,
        );
        assert!(chunk > 0, "the split must make progress at {computed}");
        let target = computed + chunk;
        assert!(coordinator.allocate(pool, lease, prompt, computed, target, computed));
        coordinator.finalize(pool, lease, request, prompt, target, hit);
        coordinator.release_copy_sources(pool);
        computed = target;
        ends.push(target);
    }
    coordinator.begin_step();
    ends
}

const GLM_STATES: [u16; 4] = [2, 3, 4, 5];

/// One batch of `BUDGET` tokens in flight and no model-length cap.
const LIMITS: AdmissionLimits = AdmissionLimits {
    max_in_flight_tokens: BUDGET,
    max_model_len: usize::MAX,
};

fn with_limits(rules: GroupedPrefixRules) -> GroupedPrefixCoordinator {
    GroupedPrefixCoordinator::new(rules, LIMITS)
}

#[test]
fn version_rows_differ_in_the_last_cacheable_stop_and_the_partial_tail() {
    // v0.31 sched:444 `num_tokens - num_tokens % block_size`; v0.32
    // sched:476-480 with iface:1110-1122 `(n - 1) // B * B`.
    let v31 = rules_glm(VllmRuleVersion::V0_31_0, false);
    let v32 = rules_glm(VllmRuleVersion::V0_32_0, false);
    assert_eq!(v31.last_cacheable_stop(9000), 8704);
    assert_eq!(v32.last_cacheable_stop(9000), 8704);
    assert_eq!(v31.last_cacheable_stop(8704), 8704);
    assert_eq!(v32.last_cacheable_stop(8704), 6528);
    assert_eq!(v32.last_cacheable_stop(0), 0);
    // v0.31 sched:482-487 `prompt // hash * hash`; v0.32 sched:517-525
    // `(prompt - 1) // hash * hash`.
    let v31 = rules_glm_hash(VllmRuleVersion::V0_31_0, false, 4);
    let v32 = rules_glm_hash(VllmRuleVersion::V0_32_0, false, 4);
    assert_eq!(v31.partial_tail(9000), 9000);
    assert_eq!(v32.partial_tail(9000), 8996);
    assert_eq!(v31.partial_tail(4356), 4356);
    assert_eq!(v32.partial_tail(4356), 4352);
    assert_eq!(
        VllmRuleVersion::parse("0.31.0"),
        Some(VllmRuleVersion::V0_31_0)
    );
    assert_eq!(
        VllmRuleVersion::parse("0.32.0"),
        Some(VllmRuleVersion::V0_32_0)
    );
    assert_eq!(VllmRuleVersion::parse("0.30.0"), None);
}

#[test]
fn vllm_0330_row_takes_the_0320_arm_in_every_version_branch() {
    assert_eq!(
        VllmRuleVersion::parse("0.33.0"),
        Some(VllmRuleVersion::V0_33_0)
    );
    assert_eq!(VllmRuleVersion::parse("0.33.1"), None);
    assert_eq!(VllmRuleVersion::V0_31_0.label(), "0.31.0");
    assert_eq!(VllmRuleVersion::V0_32_0.label(), "0.32.0");
    assert_eq!(VllmRuleVersion::V0_33_0.label(), "0.33.0");
    // v0.33 sched:579-584 with iface:1169-1186: `(n - max(1, L)) // B * B`.
    // The hash lookahead L is 0 without speculative decoding (v0.33
    // `vllm/config/vllm.py:677-692`), which grouped execution refuses.
    let v32 = rules_glm(VllmRuleVersion::V0_32_0, false);
    let v33 = rules_glm(VllmRuleVersion::V0_33_0, false);
    for (n, stop) in [(9000, 8704), (8704, 6528), (0, 0)] {
        assert_eq!(v33.last_cacheable_stop(n), stop);
        assert_eq!(v32.last_cacheable_stop(n), stop);
    }
    // v0.33 sched:622-631: `(prompt - 1) // hash * hash`.
    let fine = rules_glm_hash(VllmRuleVersion::V0_33_0, false, 4);
    assert_eq!(fine.partial_tail(9000), 8996);
    assert_eq!(fine.partial_tail(4356), 4352);
    // v0.33 sched:437-449: checkpoint mode only when every Mamba group has
    // a positive count.
    assert!(v33.checkpoint.is_none());
    let ckpt = rules_glm(VllmRuleVersion::V0_33_0, true);
    assert_eq!(ckpt.checkpoint, Some(CheckpointRule { alignment: 16 }));
    // The align split (v0.33 sched:546-662, iface:1189-1210).
    assert_eq!(split(&v33, 0, 9000, 9000, 8192, 0), 6528);
    assert_eq!(split(&v33, 6528, 9000, 9000, 2472, 0), 2176);
    assert_eq!(split(&v33, 4352, 8704, 8704, 4352, 0), 2176);
    assert_eq!(ckpt.checkpoint_position(9000), 8704);
    assert!(ckpt.checkpoint_valid(6528, 9000, 8704));
    assert_eq!(split(&ckpt, 6528, 9000, 9000, 2472, 0), 2472);
    assert_eq!(split(&ckpt, 4352, 8704, 8704, 4352, 0), 4352);
}

#[test]
fn vllm_0330_row_refuses_checkpoint_fields_with_its_own_texts() {
    let refusal = |groups: &[FpmCacheGroup], hash: usize| {
        GroupedPrefixRules::new(groups, hash, VllmRuleVersion::V0_33_0)
            .unwrap_err()
            .to_string()
    };
    let mut groups = groups_glm(false);
    groups[2].prefill_checkpoint_blocks = None;
    assert_eq!(
        refusal(&groups, 2176),
        "kv_cache_groups with vllm_version=0.33.0 require prefill_checkpoint_blocks on every prefix-cacheable state group; set the field (0 or 1) on each such group"
    );
    let mut groups = groups_glm(true);
    groups[3].prefill_checkpoint_blocks = Some(0);
    groups[3].prefill_checkpoint_alignment = None;
    assert_eq!(
        refusal(&groups, 2176),
        "kv_cache_groups with vllm_version=0.33.0 require the same prefill_checkpoint_blocks across state groups"
    );
    let mut groups = groups_glm(true);
    groups[3].prefill_checkpoint_alignment = Some(32);
    assert_eq!(
        refusal(&groups, 2176),
        "kv_cache_groups with vllm_version=0.33.0 require one prefill_checkpoint_alignment across state groups"
    );
    let small_attention = [attention("full", 16, None), state("s", 64, true, true)];
    assert_eq!(
        refusal(&small_attention, 16),
        "kv_cache_groups with vllm_version=0.33.0 require every state group with prefill checkpoint blocks to have the smallest block_size_tokens of the prefix_cacheable groups"
    );
}

#[test]
fn second_cache_step_of_a_keyed_state_block_defers_no_waiting_request_on_vllm_0330() {
    // vLLM 0.33.0 does not publish again a state block that already has the
    // key of its position (v0.33 single:2112-2115), so the key does not go
    // into the set of the step (v0.33 single:1994-1999) and a request that
    // hits it does not wait (v0.33 single:1704-1712). Replay publishes a
    // state key once, so the second cache step of the producer marks nothing.
    let groups = [attention("full", 16, None), state("s", 16, true, false)];
    let hash = 16;
    let rules = GroupedPrefixRules::new(&groups, hash, VllmRuleVersion::V0_33_0).unwrap();
    let mut coordinator = with_limits(rules);
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let producer = lease(40, hash);
    let hit = coordinator.find_longest_hit(pool, &producer, 40);
    let mut block_lease = coordinator.new_lease();
    // Step 1: the chunk ends at the last cacheable stop (40 - 1) / 16 * 16.
    coordinator.begin_step();
    assert_eq!(
        coordinator
            .rules()
            .align_split(0, 40, 40, BUDGET, 0, BUDGET),
        32
    );
    assert!(coordinator.allocate(pool, &mut block_lease, 40, 0, 32, 0));
    coordinator.finalize(pool, &mut block_lease, &producer, 40, 32, &hit);
    // In the publishing step the key at 32 is no hit.
    assert_eq!(
        coordinator
            .find_longest_hit(pool, &lease(48, hash), 48)
            .tokens,
        0
    );
    // Step 2: the producer computes 32..40. Its cache step is still in the
    // block that ends at 32.
    coordinator.begin_step();
    assert_eq!(
        coordinator
            .rules()
            .align_split(32, 40, 40, BUDGET, 0, BUDGET),
        8
    );
    assert!(coordinator.allocate(pool, &mut block_lease, 40, 32, 40, 32));
    coordinator.finalize(pool, &mut block_lease, &producer, 40, 40, &hit);
    // A request that waits in step 2 gets the hit of 32 tokens in step 2.
    assert_eq!(
        coordinator
            .find_longest_hit(pool, &lease(48, hash), 48)
            .tokens,
        32
    );
}

#[test]
fn rules_refuse_unsupported_checkpoint_fields() {
    // v0.32 sched:390-394 runs checkpoint mode only when every Mamba
    // group has a positive count.
    let err = GroupedPrefixRules::new(&groups_glm(true), 2176, VllmRuleVersion::V0_31_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.31.0 do not model prefill checkpoint blocks"
    );
    let mut groups = groups_glm(false);
    groups[2].prefill_checkpoint_blocks = None;
    let err = GroupedPrefixRules::new(&groups, 2176, VllmRuleVersion::V0_32_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.32.0 require prefill_checkpoint_blocks on every prefix-cacheable state group; set the field (0 or 1) on each such group"
    );
    let mut groups = groups_glm(true);
    groups[3].prefill_checkpoint_alignment = Some(32);
    let err = GroupedPrefixRules::new(&groups, 2176, VllmRuleVersion::V0_32_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.32.0 require one prefill_checkpoint_alignment across state groups"
    );
    let zero = rules_glm(VllmRuleVersion::V0_32_0, false);
    assert!(zero.checkpoint.is_none());
    let one = rules_glm(VllmRuleVersion::V0_32_0, true);
    assert_eq!(one.checkpoint, Some(CheckpointRule { alignment: 16 }));
    // Block mode holds one running block per state group, so another count
    // is refused.
    let mut groups = groups_glm(false);
    groups[4].blocks_per_request = Some(2);
    let err = GroupedPrefixRules::new(&groups, 2176, VllmRuleVersion::V0_31_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with enable_prefix_caching=true model one state block per request; blocks_per_request must be 1"
    );
    // utils:799-805 and coord:87-92.
    assert!(GroupedPrefixRules::new(&groups_glm(false), 3, VllmRuleVersion::V0_31_0).is_err());
}

#[test]
fn rules_refuse_a_checkpoint_block_on_a_ring() {
    // The count is a field of `MambaSpec` alone (v0.31 iface:1049-1050,
    // v0.32 iface:1042-1043, v0.33 iface:1101-1102). A ring that declares a
    // block is refused on every row. A ring with no count or a zero count
    // stays a ring.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut groups = vec![attention("full", 16, None), state("ring", 4, false, false)];
        for count in [None, Some(0)] {
            groups[1].prefill_checkpoint_blocks = count;
            let rules = GroupedPrefixRules::new(&groups, 16, version).unwrap();
            assert_eq!(rules.groups[1].kind, GroupRuleKind::Ring, "{version:?}");
        }
        groups[1].prefill_checkpoint_blocks = Some(1);
        groups[1].prefill_checkpoint_alignment = Some(16);
        assert!(groups[1].validate().is_ok());
        let err = GroupedPrefixRules::new(&groups, 16, version)
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "kv_cache_groups with enable_prefix_caching=true do not model a prefill checkpoint block on a state group with prefix_cacheable=false; set prefix_cacheable: true on a Mamba state group, or remove prefill_checkpoint_blocks and prefill_checkpoint_alignment from a ring buffer",
            "{version:?}"
        );
    }
}

#[test]
fn rules_refuse_missing_prefix_cacheability() {
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for missing in 0..2 {
            let mut groups = [attention("full", 16, None), state("s", 16, true, false)];
            groups[missing].prefix_cacheable = None;
            // Legacy profiles permit absence, but it cannot identify a
            // state group as a ring when prefix caching is enabled.
            assert!(groups[missing].validate().is_ok());
            let err = GroupedPrefixRules::new(&groups, 16, version)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("require prefix_cacheable on every group"),
                "{err}"
            );
        }
    }
}

#[test]
fn rules_refuse_a_non_cacheable_sliding_window_group() {
    // The window specs that opt out of prefix caching: the v0.31 ring
    // `KpoolTailSpec` (iface:1015-1038), which GLM-5.3-Flash builds with
    // `block_size=ring` and `sliding_window=ring`
    // (glm5next/common/attention.py:205-212; v0.32 and v0.33 have no such
    // spec), and a bounded replay window (`SlidingWindowMLASpec`, v0.31
    // iface:949-955, v0.32 :968-970, v0.33 :1027-1029).
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut groups = [
            attention("full", 16, None),
            attention("other", 16, Some(17)),
        ];
        groups[1].prefix_cacheable = Some(false);
        let err = GroupedPrefixRules::new(&groups, 16, version)
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "kv_cache_groups with enable_prefix_caching=true do not model a sliding-window attention group with prefix_cacheable=false; describe a ring buffer as a state group with blocks_per_request: 1 and prefix_cacheable: false, and turn prefix caching off for a bounded replay window, which replay does not model",
            "{version:?}"
        );
        // The same ring as a state group is accepted.
        let ring = [attention("full", 16, None), state("ring", 16, false, false)];
        assert!(GroupedPrefixRules::new(&ring, 16, version).is_ok());
    }
}

#[test]
fn non_cacheable_full_attention_group_takes_no_part_in_the_hit() {
    // Two full-attention groups of block 16, the second with
    // `prefix_cacheable: false`. vLLM's hybrid coordinator skips a group
    // that opts out of prefix caching in the lookup, so its slot of the
    // hit stays empty (v0.31 coord:743-749, v0.32 coord:713-720, v0.33
    // coord:723-730). Its manager then counts `cdiv(17, 16) = 2` new blocks
    // with no computed block (v0.31 single:216-268, v0.32 single:213-265,
    // v0.33 single:214-266). The first group hits 16 tokens and needs one
    // new block and its inactive hit block.
    let mut groups = [attention("full", 16, None), attention("other", 16, None)];
    groups[1].prefix_cacheable = Some(false);
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 16, version).unwrap());
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16]);
        let request = lease(17, 16);
        let hit = coordinator.find_longest_hit(pool, &request, 17);
        assert_eq!(hit.tokens, 16, "{version:?}");
        assert_eq!(hit.per_group[1], GroupHitBlocks::None, "{version:?}");
        assert_eq!(coordinator.admission_blocks(pool, &hit, 17), 2 + 2);
        let mut block_lease = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
        assert!(coordinator.allocate(pool, &mut block_lease, 17, 16, 17, 16));
        assert_eq!(attention_copies(&block_lease, 0), 2);
        assert_eq!(attention_copies(&block_lease, 1), 2);
        coordinator.release_request(pool, &mut block_lease);
        assert_eq!(pool.active_blocks(), 0);
    }
}

#[test]
fn rules_map_a_convolution_group_to_a_sliding_window() {
    // vLLM's convolution state is a plain `SlidingWindowSpec` whose block and
    // window are the kernel size (v0.31.0, v0.32.0 and v0.33.0
    // vllm/models/inkling/nvidia/sconv_swa_attn.py:158-159, :185-193).
    let mut convolution = attention("convolution", 4, Some(4));
    convolution.kind = FpmCacheKind::Convolution;
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for groups in [
            vec![convolution.clone()],
            vec![attention("full", 16, None), convolution.clone()],
        ] {
            let rules = GroupedPrefixRules::new(&groups, 4, version).unwrap();
            assert_eq!(
                rules.groups.last().unwrap().kind,
                GroupRuleKind::SlidingWindow { window: 4 },
                "{version:?}"
            );
        }
        // With a full-attention group of 16-token blocks the scheduler block
        // is lcm(16, 4) = 16. No state group allows hits inside a block, so
        // hits land on multiples of 16. The startup check of a max_model_len
        // of 256 counts 256 / 16 = 16 full-attention blocks and the admission
        // cap of the window, cdiv(min(3 + in flight, 256), 4) + 1 (v0.31.0
        // kv_cache_interface.py:859-866, v0.32.0 :878-885, v0.33.0 :937-944):
        // 18, 10 and 65 blocks for 64, 32 and 512 or 256 tokens in flight.
        let rules = GroupedPrefixRules::new(
            &[attention("full", 16, None), convolution.clone()],
            4,
            version,
        )
        .unwrap();
        assert_eq!(rules.alignment, 16, "{version:?}");
        for (in_flight, startup) in [(64, 34), (32, 26), (512, 81), (256, 81)] {
            assert_eq!(
                rules.startup_blocks(256, in_flight),
                startup,
                "{version:?} at {in_flight}"
            );
        }
        // The profile can give a convolution group a window that is not its
        // block. The window is then the `sliding_window` of the group: with a
        // block of 4 and a window of 8 the admission cap is
        // cdiv(min(7 + in flight, 256), 4) + 1, with the term
        // `extra_retained_tokens` at its default 0 (v0.31.0 iface:838, v0.32.0
        // :857, v0.33.0 :908), so the startup check counts 16 + 10 + 1 = 27
        // blocks for 32 tokens in flight and 16 + 18 + 1 = 35 for 64.
        let mut wide = attention("convolution", 4, Some(8));
        wide.kind = FpmCacheKind::Convolution;
        let rules =
            GroupedPrefixRules::new(&[attention("full", 16, None), wide], 4, version).unwrap();
        assert_eq!(
            rules.groups.last().unwrap().kind,
            GroupRuleKind::SlidingWindow { window: 8 },
            "{version:?}"
        );
        for (in_flight, startup) in [(32, 27), (64, 35)] {
            assert_eq!(
                rules.startup_blocks(256, in_flight),
                startup,
                "{version:?} at {in_flight}"
            );
        }
        // The profile keeps the kind, so a convolution group and an attention
        // group with the same block and window stay two spec groups. vLLM
        // joins two groups only when their specs are equal in every field
        // (v0.31.0 kv_cache_coordinator.py:755-761, v0.32.0 :726-732,
        // v0.33.0 :736-742), and the profile does not hold the other fields
        // of a spec, such as the head size.
        let rules = GroupedPrefixRules::new(
            &[attention("window", 4, Some(4)), convolution.clone()],
            4,
            version,
        )
        .unwrap();
        assert_eq!(rules.specs.len(), 2, "{version:?}");

        let mut opted_out = convolution.clone();
        opted_out.prefix_cacheable = Some(false);
        let mut no_window = convolution.clone();
        no_window.sliding_window = None;
        for (group, text) in [
            (
                opted_out,
                "kv_cache_groups with enable_prefix_caching=true do not model a convolution group with prefix_cacheable=false; turn prefix caching off to replay it cold",
            ),
            (
                no_window,
                "convolution cache groups require a sliding_window",
            ),
        ] {
            let err = GroupedPrefixRules::new(&[attention("full", 16, None), group], 4, version)
                .unwrap_err()
                .to_string();
            assert_eq!(err, text, "{version:?}");
        }
    }
}

// The values below fit a 64-bit `usize` only.
#[cfg(target_pointer_width = "64")]
#[test]
fn rules_refuse_a_scheduler_block_above_usize() {
    // The group validation has no upper bound for block_size_tokens. The
    // LCM of these three blocks is 23,058,430,081,399,521,280, above
    // `usize::MAX`. vLLM computes `math.lcm` on Python integers, which
    // cannot overflow.
    let groups = [
        attention("a", 1 << 31, None),
        attention("b", (1 << 31) - 1, None),
        attention("c", 5, None),
    ];
    assert!(groups.iter().all(|group| group.validate().is_ok()));
    let err = GroupedPrefixRules::new(&groups, 1, VllmRuleVersion::V0_31_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with enable_prefix_caching=true require the least common multiple of every group's block_size_tokens (the scheduler block) to be at most 18446744073709551615; choose block_size_tokens with a smaller least common multiple"
    );
    // Two u32 blocks cannot overflow, and keep their exact LCM.
    let groups = [
        attention("a", u32::MAX, None),
        attention("b", u32::MAX - 1, None),
    ];
    assert_eq!(rules(&groups, 1).alignment, 18_446_744_060_824_649_730);
}

/// `count` full-attention groups of one block size, which form one spec.
fn equal_groups(count: usize) -> Vec<FpmCacheGroup> {
    (0..count)
        .map(|i| attention(&format!("g{i}"), 16, None))
        .collect()
}

#[test]
fn rules_refuse_more_groups_than_a_cache_key_can_name() {
    // `CacheKey::Grouped` names a group by its position among all groups of
    // the rank, in a `u16`, so it can name the positions 0 to 65,535. The
    // bound counts every group, also one that opts out of prefix caching:
    // placed first, such a group moves the last caching group to position
    // 65,536. So 65,536 caching groups plus one that opts out are refused
    // in either order. The check is the same for each rule version.
    let mut opted_out = attention("off", 16, None);
    opted_out.prefix_cacheable = Some(false);
    let mut opted_out_first = vec![opted_out.clone()];
    opted_out_first.extend(equal_groups(65_536));
    let mut opted_out_last = equal_groups(65_536);
    opted_out_last.push(opted_out);
    let too_many = [equal_groups(65_537), opted_out_first, opted_out_last];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        assert!(
            GroupedPrefixRules::new(&equal_groups(65_536), 16, version).is_ok(),
            "{version:?}"
        );
        for (row, groups) in too_many.iter().enumerate() {
            // `err()` keeps a failure of this test from printing 65,537 rules.
            let refusal = GroupedPrefixRules::new(groups, 16, version)
                .err()
                .map(|err| err.to_string());
            assert_eq!(
                refusal.as_deref(),
                Some(
                    "kv_cache_groups with enable_prefix_caching=true support at most 65536 groups; use fewer groups, or turn prefix caching off to replay them cold"
                ),
                "{version:?} row {row}"
            );
        }
    }
}

#[test]
fn last_of_65536_groups_has_its_own_cache_key() {
    // The groups have equal fields, so they form one spec, and a block of a
    // spec hits only when every group of the spec holds it, as vLLM's
    // `get_cached_block` asks (v0.31.0 and v0.32.0 `pool:198-224`, v0.33.0
    // `pool:202-228`). The last group has the largest position that a key
    // can hold, 65,535.
    let groups = equal_groups(65_536);
    let coordinator = with_limits(rules(&groups, 16));
    let mut kv = kv_pool(&groups, 65_536);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(32, 16);
    let hash = request.prefix_hash_at(16).unwrap();
    for group in 0..u16::MAX {
        seed(pool, CacheKey::Grouped { group, hash });
    }
    assert_eq!(coordinator.find_longest_hit(pool, &request, 32).tokens, 0);
    seed(
        pool,
        CacheKey::Grouped {
            group: u16::MAX,
            hash,
        },
    );
    let hit = coordinator.find_longest_hit(pool, &request, 32);
    assert_eq!(hit.tokens, 16);
    assert!(matches!(
        &hit.per_group[65_535],
        GroupHitBlocks::Attention(keys)
            if keys[..] == [Some(CacheKey::Grouped { group: u16::MAX, hash })]
    ));
}

#[test]
fn state_group_at_the_last_position_is_found_inside_its_block() {
    // 65,535 full-attention groups of block 16 form one spec, and a state
    // group of block 32 at position 65,535 forms another. The hash block is
    // 16, so a state hit below the block end 32 lies inside a block, and
    // the lookup finds it only through the keys that the pool lists under
    // the block's first hash for the state group's own position. The hit is
    // the length that every spec accepts, as vLLM's hybrid lookup lowers it
    // (v0.31.0 `coord:883-953`, v0.32.0 `coord:837-907`, v0.33.0
    // `coord:847-917`). With a limit of 18 the attention spec hits 16 and
    // the state spec hits only where its key sits, so the hit is 0 before
    // the state key at 16 is cached and 16 after.
    let mut groups = equal_groups(65_535);
    groups.push(state("s", 32, true, false));
    let coordinator = with_limits(rules(&groups, 16));
    let mut kv = kv_pool(&groups, 70_000);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(32, 16);
    let hash = request.prefix_hash_at(16).unwrap();
    for group in 0..u16::MAX {
        seed(pool, CacheKey::Grouped { group, hash });
    }
    assert_eq!(coordinator.find_longest_hit(pool, &request, 18).tokens, 0);
    seed_inside(pool, &[u16::MAX], 16, 32, 16);
    let hit = coordinator.find_longest_hit(pool, &request, 18);
    assert_eq!(hit.tokens, 16);
    assert_eq!(
        hit.per_group[65_535],
        GroupHitBlocks::State(Some(CacheKey::Grouped {
            group: u16::MAX,
            hash
        }))
    );
}

#[test]
fn alignment_is_the_hash_block_only_with_fine_grained_hits() {
    // coord:687-735. G2 to G5 are state rules with block 2176 and G0 is
    // full attention. At hash 2176 no state block exceeds the hash block.
    let glm = rules_glm(VllmRuleVersion::V0_31_0, false);
    assert_eq!(glm.alignment, 2176);
    assert_eq!(glm.cacheable_block, 2176);
    let fine = rules_glm_hash(VllmRuleVersion::V0_31_0, false, 4);
    assert_eq!(fine.alignment, 4);
    // Fine-grained hits need a Mamba group in align mode whose block exceeds
    // the hash block. Two attention groups have none, so they give the
    // scheduler block, with or without a window (coord:691-704).
    let windowed = [attention("a", 64, Some(100)), attention("b", 16, None)];
    assert_eq!(rules(&windowed, 16).alignment, 64);
    let plain = [attention("a", 64, None), attention("b", 16, None)];
    assert_eq!(rules(&plain, 16).alignment, 64);
    // A state group above the hash block turns fine-grained hits on when
    // every other cacheable rule is full attention or at the hash block.
    let with_state = [
        attention("a", 64, None),
        attention("b", 16, None),
        state("s", 64, true, false),
    ];
    assert_eq!(rules(&with_state, 16).alignment, 16);
    // A window at another block than the hash block turns them off again
    // (single:960 has no sub-block lookup).
    let with_window = [
        attention("a", 64, Some(100)),
        attention("b", 16, None),
        state("s", 64, true, false),
    ];
    assert_eq!(rules(&with_window, 16).alignment, 64);
}

#[test]
fn full_attention_hit_is_capped_below_the_prompt() {
    // mgr:289-295 caps the lookup at `num_tokens - 1`.
    let groups = [attention("full", 16, None)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 16, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16, 32, 48, 64]);
        let hit = coordinator.find_longest_hit(pool, &lease(64, 16), 64);
        assert_eq!(hit.tokens, 48);
        assert_eq!(hit.junction_tokens, 0);
        assert_eq!(hit.per_group.len(), 1);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 3));
        let hit = coordinator.find_longest_hit(pool, &lease(65, 16), 65);
        assert_eq!(hit.tokens, 64);
    }
}

#[test]
fn decode_side_hit_of_a_state_model_is_its_full_attention_hit_floored_to_its_block() {
    // A state group above the hash block turns fine-grained hits on, so the
    // alignment is 4 while the attention block is 16. The state key at 36 is
    // an own hit of 36, no deeper than the full-attention hit (mgr:360-364),
    // so the local prefix is that hit, floored to whole attention blocks
    // (V32 `sched:971-982`). The state's hit of 36 is cdiv(36, 16) = 3
    // blocks, above the 2 that the truncation keeps (V32 `mgr:826-827`), so
    // it is dropped.
    let groups = [attention("a", 16, None), state("s", 16, true, false)];
    let coordinator = with_limits(rules(&groups, 4));
    assert_eq!(coordinator.rules.alignment, 4);
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, 4, &[16, 32]);
    seed_inside(pool, &[0, 1], 4, 16, 36);
    let request = lease(64, 4);
    // The fine-grained lookup reaches the sub-block tail at 36.
    let (ends, tokens) =
        coordinator.full_attention_hit(pool, &request, &coordinator.rules.specs[0], 60);
    assert_eq!((ends, tokens), (vec![16, 32, 36], 36));
    let hit = coordinator.find_destination_hit(pool, &request, 64, 63);
    assert_eq!(hit.tokens, 32);
    assert_eq!(hit.junction_tokens, 0);
    assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 2));
    assert_eq!(hit.per_group[1], GroupHitBlocks::None);
    // The cap at `num_tokens - 1` applies before the floor.
    assert_eq!(
        coordinator
            .find_destination_hit(pool, &lease(33, 4), 33, 32)
            .tokens,
        32
    );
    assert_eq!(
        coordinator
            .find_destination_hit(pool, &lease(32, 4), 32, 31)
            .tokens,
        16
    );
}

#[test]
fn decode_side_hit_without_state_layers_is_the_hit_every_group_agrees_on() {
    // mgr:346-351: without Mamba layers the local prefix is the fixed point.
    // The window needs one cached block before an aligned end and has one
    // only at 16, so the hit is 16 with a junction of 48 - 16 = 32.
    let groups = [
        attention("full", 16, None),
        attention("window", 16, Some(17)),
    ];
    let coordinator = with_limits(rules(&groups, 16));
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, 16, &[16, 32, 48]);
    seed(pool, key(1, 16, 16));
    let hit = coordinator.find_destination_hit(pool, &lease(64, 16), 64, 64);
    assert_eq!(hit.tokens, 16);
    assert_eq!(hit.junction_tokens, 32);
    assert_eq!(
        hit.per_group[0],
        GroupHitBlocks::Attention(vec![Some(key(0, 16, 16))])
    );
    assert_eq!(hit.per_group[1], GroupHitBlocks::None);
}

#[test]
fn decode_side_hit_falls_back_when_another_group_hits_deeper() {
    // mgr:360-364: a deeper own hit of another group means the fixed point.
    // The own hits are 32 and 48, and the fixed point finds no state at 32 or
    // 16, so the local prefix is 0.
    let groups = [attention("full", 16, None), state("s", 16, true, false)];
    let coordinator = with_limits(rules(&groups, 16));
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, 16, &[16, 32]);
    seed(pool, key(1, 16, 48));
    assert_eq!(
        coordinator
            .find_destination_hit(pool, &lease(64, 16), 64, 63)
            .tokens,
        0
    );
    // A state at 32 is not deeper, so the local prefix is the full-attention
    // hit of 32 (mgr:366-369).
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, 16, &[16, 32]);
    seed(pool, key(1, 16, 32));
    let hit = coordinator.find_destination_hit(pool, &lease(64, 16), 64, 63);
    assert_eq!(hit.tokens, 32);
    assert_eq!(hit.junction_tokens, 0);
}

#[test]
fn decode_side_hit_that_falls_back_keeps_the_agreed_hit_and_its_junction() {
    // The own hits are 32 and 48, so the finder falls back (mgr:360-364). The
    // fixed point is 16 and the longest lookup 32, so the junction is 16, at
    // vLLM's boundary 16 + 16 = 32 (mgr:312-321). Only full attention gets
    // hit blocks to pin. The state's hit is its block that ends at 16. The
    // hit of 16 ends on a block, so the scheduler keeps it (V32
    // `sched:1007-1018`). Both tests count it, and the load drops it (V32
    // `single:1724-1778`, `:295-300`).
    let groups = [attention("full", 16, None), state("s", 16, true, false)];
    let coordinator = with_limits(rules(&groups, 16));
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, 16, &[16, 32]);
    seed_many(pool, 1, 16, &[16, 48]);
    let hit = coordinator.find_destination_hit(pool, &lease(64, 16), 64, 63);
    assert_eq!(hit.tokens, 16);
    assert_eq!(hit.junction_tokens, 16);
    assert_eq!(
        hit.per_group[0],
        GroupHitBlocks::Attention(vec![Some(key(0, 16, 16))])
    );
    assert_eq!(
        hit.per_group[1],
        GroupHitBlocks::DroppedState {
            key: key(1, 16, 16),
            tokens: 16
        }
    );
}

#[test]
fn decode_side_hit_floors_a_fallback_inside_a_block_and_keeps_the_junction_position() {
    // Fine-grained hits at hash 4 with 16-token blocks. The own hits are 32
    // and 48, so the finder falls back (mgr:360-364). The fixed point is 20
    // and the longest lookup 32, so vLLM's boundary is 32 (mgr:312-321). A
    // load past 20 truncates the hit to the floor 16 (V32 `sched:1007-1018`),
    // and the junction grows from 12 to 16 to keep the boundary at 32. The
    // state's hit of 20 is cdiv(20, 16) = 2 blocks, above the 1 that the
    // truncation keeps (V32 `mgr:826-827`), so it is dropped.
    let groups = [attention("a", 16, None), state("s", 16, true, false)];
    let coordinator = with_limits(rules(&groups, 4));
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, 4, &[16, 32]);
    seed_inside(pool, &[1], 4, 16, 20);
    seed_many(pool, 1, 4, &[48]);
    let hit = coordinator.find_destination_hit(pool, &lease(64, 4), 64, 63);
    assert_eq!(hit.tokens, 16);
    assert_eq!(hit.junction_tokens, 16);
    assert_eq!(
        hit.per_group[0],
        GroupHitBlocks::Attention(vec![Some(key(0, 4, 16))])
    );
    assert_eq!(hit.per_group[1], GroupHitBlocks::None);
}

#[test]
fn decode_side_hit_without_a_full_attention_group_is_the_hit_every_group_agrees_on() {
    // mgr:346-351: without a full-attention group there is no per-group path.
    // The window and the state both hit at 16, the fixed point below the cap
    // of 32. The window gets no hit block. The state's hit of 16 ends on its
    // block, below the 32 loaded tokens, so both tests count it and the load
    // drops it.
    let groups = [
        attention("window", 16, Some(17)),
        state("s", 16, true, false),
    ];
    let coordinator = with_limits(rules(&groups, 16));
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed(pool, key(0, 16, 16));
    seed(pool, key(1, 16, 16));
    let hit = coordinator.find_destination_hit(pool, &lease(33, 16), 33, 32);
    assert_eq!(hit.tokens, 16);
    assert_eq!(hit.per_group[0], GroupHitBlocks::None);
    assert_eq!(
        hit.per_group[1],
        GroupHitBlocks::DroppedState {
            key: key(1, 16, 16),
            tokens: 16
        }
    );
}

#[test]
fn decode_side_hit_without_a_full_attention_group_drops_a_state_hit_inside_its_block() {
    // A window at the hash block 4 and a state of 16-token blocks make the
    // hits fine-grained. Both groups hit at 20, so the agreed hit is 20. A
    // load past 20 truncates it to the scheduler block 16, the LCM of the
    // group blocks (V31 `sched:974-985`, V32 `sched:1007-1018`, v0.33
    // `sched:1167-1178`). The state's hit of 20 is cdiv(20, 16) = 2 blocks,
    // above the 1 that the truncation keeps (V31 `mgr:852-853`, V32
    // `mgr:826-827`, v0.33 `mgr:835-836`), so it is dropped.
    let groups = [attention("window", 4, Some(5)), state("s", 16, true, false)];
    let coordinator = with_limits(rules(&groups, 4));
    assert_eq!(coordinator.rules.alignment, 4);
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed(pool, key(0, 4, 20));
    seed_inside(pool, &[1], 4, 16, 20);
    let hit = coordinator.find_destination_hit(pool, &lease(33, 4), 33, 32);
    // 20 - 20 % 16.
    assert_eq!(hit.tokens, 16);
    assert_eq!(hit.per_group[0], GroupHitBlocks::None);
    assert_eq!(hit.per_group[1], GroupHitBlocks::None);
    // An agreed hit of 16 ends on the state block, so the state's hit stays.
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed(pool, key(0, 4, 16));
    seed(pool, key(1, 4, 16));
    let hit = coordinator.find_destination_hit(pool, &lease(33, 4), 33, 32);
    assert_eq!(hit.tokens, 16);
    assert_eq!(
        hit.per_group[1],
        GroupHitBlocks::DroppedState {
            key: key(1, 4, 16),
            tokens: 16
        }
    );
}

#[test]
fn decode_side_admission_counts_each_state_groups_own_hit() {
    // The local prefix is 48, and the state's own hit of 32 is not deeper
    // (mgr:366-369). The whole prompt of 65 counts full attention 5 - 3 = 2
    // new and 3 free hit blocks, and the state min(5 - 2, 1) = 1 new and its
    // free hit block: 7. The load of 64 counts 4 - 3 = 1 and 3, and
    // min(4 - 2, 1) = 1 and 1: 6 (V32 `single:1724-1778`).
    let groups = [attention("full", 16, None), state("s", 16, true, false)];
    let coordinator = with_limits(rules(&groups, 16));
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, 16, &[16, 32, 48]);
    seed(pool, key(1, 16, 32));
    let hit = coordinator.find_destination_hit(pool, &lease(65, 16), 65, 64);
    assert_eq!(hit.tokens, 48);
    assert_eq!(
        hit.per_group[1],
        GroupHitBlocks::DroppedState {
            key: key(1, 16, 32),
            tokens: 32
        }
    );
    assert_eq!(
        coordinator.destination_admission_blocks(pool, &hit, 64, 65),
        7
    );
    assert_eq!(coordinator.destination_load_blocks(pool, &hit, 64, 65), 6);
    // A state block that another request holds costs no capacity.
    let held = GroupedHit {
        tokens: 32,
        junction_tokens: 0,
        per_group: vec![
            GroupHitBlocks::Attention(Vec::new()),
            GroupHitBlocks::State(Some(key(1, 16, 32))),
        ],
    };
    coordinator
        .pin_hit(pool, &mut coordinator.new_lease(), &held)
        .unwrap();
    assert_eq!(
        coordinator.destination_admission_blocks(pool, &hit, 64, 65),
        6
    );
    assert_eq!(coordinator.destination_load_blocks(pool, &hit, 64, 65), 5);
    // The decode rank pins the full-attention blocks only.
    let mut blocks = coordinator.new_lease();
    coordinator.pin_hit(pool, &mut blocks, &hit).unwrap();
    assert!(state_lease(&blocks, 1).hit.is_none());
}

#[test]
fn decode_side_hit_drops_a_state_hit_above_the_floored_prefix() {
    // Hash block 4 makes the hits fine-grained. The full-attention hit of 36
    // floors to 32, and a Mamba hit of 36 is cdiv(36, 16) = 3 blocks, above
    // the 2 that the truncation keeps (V32 sched:1007-1018, mgr:826-827).
    let groups = [attention("a", 16, None), state("s", 16, true, false)];
    let coordinator = with_limits(rules(&groups, 4));
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, 4, &[16, 32]);
    seed_inside(pool, &[0, 1], 4, 16, 36);
    let hit = coordinator.find_destination_hit(pool, &lease(64, 4), 64, 63);
    assert_eq!(hit.tokens, 32);
    assert_eq!(hit.per_group[1], GroupHitBlocks::None);
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, 4, &[16, 32]);
    seed_inside(pool, &[0], 4, 16, 36);
    seed(pool, key(1, 4, 32));
    let hit = coordinator.find_destination_hit(pool, &lease(64, 4), 64, 63);
    assert_eq!(
        hit.per_group[1],
        GroupHitBlocks::DroppedState {
            key: key(1, 4, 32),
            tokens: 32
        }
    );
}

#[test]
fn decode_side_hit_floors_to_the_scheduler_block_above_the_full_attention_block() {
    // Full attention with 16-token blocks and a state with 32-token blocks at
    // hash block 4: the hits are fine-grained and the scheduler block is
    // lcm(16, 32) = 32. The full-attention hit is 48 and the state's own hit
    // 32, not deeper, so the local prefix is 48 (V31, V32 `mgr:356-369`, v0.33
    // `mgr:365-378`). The scheduler offers 48 - 48 % 32 = 32 to the connector
    // (V31 `sched:944-952`, V32 `sched:974-982`, v0.33 `sched:1134-1142`). The
    // load of 96 - 32 = 64 tokens exceeds the tail of 16, so the hit is
    // truncated to 32 (V31 `sched:974-985`, V32 `sched:1007-1018`, v0.33
    // `sched:1167-1178`): full attention keeps 32 / 16 = 2 blocks and the
    // state min(32 / 32, 1) = 1 (V31 `mgr:850-856`, V32 `mgr:824-830`, v0.33
    // `mgr:833-839`).
    let groups = [attention("a", 16, None), state("s", 32, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        assert_eq!(coordinator.rules.alignment, 4);
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 4, &[16, 32, 48]);
        seed(pool, key(1, 4, 32));
        let hit = coordinator.find_destination_hit(pool, &lease(97, 4), 97, 96);
        // 48 - 48 % 32.
        assert_eq!(hit.tokens, 32);
        assert_eq!(hit.junction_tokens, 0);
        // The first 32 / 16 blocks.
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::Attention(vec![Some(key(0, 4, 16)), Some(key(0, 4, 32))])
        );
        // The Mamba list of cdiv(32, 32) = 1 block stays.
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::DroppedState {
                key: key(1, 4, 32),
                tokens: 32
            }
        );
        // The whole prompt of 97: full attention cdiv(97, 16) - 2 = 5 new and
        // 2 free hit blocks. The state cdiv(97, 32) - 1 = 3 new, capped at 1
        // with no copy block, as the local prefix of 32 ends on its block,
        // and its free hit block: 2 (V31 `single:1762-1777`, `:1806-1811`, V32 `single:1724-1739`,
        // `:1768-1773`, v0.33 `single:1740-1755`, `:1785-1790`). 7 + 2 = 9.
        assert_eq!(
            coordinator.destination_admission_blocks(pool, &hit, 96, 97),
            9
        );
        // The load of 96: cdiv(96, 16) - 2 = 4 and 2, and
        // min(cdiv(96, 32) - 1, 1) = 1 and 1: 8.
        assert_eq!(coordinator.destination_load_blocks(pool, &hit, 96, 97), 8);
    }
}

#[test]
fn decode_side_hit_without_a_full_attention_group_floors_to_the_scheduler_block() {
    // A window at the hash block 4 and a state with 32-token blocks: the hits
    // are fine-grained and the scheduler block is lcm(4, 32) = 32. Without a
    // full-attention group the local prefix is the hit that every group
    // agrees on, 48 (V31, V32 `mgr:346-351`, v0.33 `mgr:353-358`). The
    // scheduler offers 48 - 48 % 32 = 32, and the load of 96 - 32 = 64 tokens
    // exceeds the tail of 16, so the hit is truncated to 32 (V31
    // `sched:944-952`, `:974-985`, V32 `sched:974-982`, `:1007-1018`, v0.33
    // `sched:1134-1142`, `:1167-1178`). The state's hit of 48 is
    // cdiv(48, 32) = 2 blocks, above the 32 / 32 = 1 that the truncation keeps
    // (V31 `mgr:850-853`, V32 `mgr:824-827`, v0.33 `mgr:833-836`), so it is
    // dropped.
    let groups = [attention("window", 4, Some(5)), state("s", 32, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        assert_eq!(coordinator.rules.alignment, 4);
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed(pool, key(0, 4, 48));
        seed_inside(pool, &[1], 4, 32, 48);
        let request = lease(97, 4);
        assert_eq!(coordinator.find_longest_hit(pool, &request, 97).tokens, 48);
        let hit = coordinator.find_destination_hit(pool, &request, 97, 96);
        // 48 - 48 % 32.
        assert_eq!(hit.tokens, 32);
        assert_eq!(hit.junction_tokens, 0);
        assert_eq!(hit.per_group[0], GroupHitBlocks::None);
        assert_eq!(hit.per_group[1], GroupHitBlocks::None);
    }
}

#[test]
fn decode_side_hit_drops_a_state_hit_above_the_scheduler_block_floor() {
    // Two states with 16-token and 32-token blocks at hash block 4: the
    // scheduler block is lcm(16, 32) = 32. Both hold the key at 48, so the
    // agreed hit is 48, and the scheduler truncates it to 48 - 48 % 32 = 32
    // (V31 `sched:944-952`, `:974-985`, V32 `sched:974-982`, `:1007-1018`,
    // v0.33 `sched:1134-1142`, `:1167-1178`). The hit of the 16-token state
    // ends on its block, but its cdiv(48, 16) = 3 blocks are above the
    // 32 / 16 = 2 that the truncation keeps (V31 `mgr:850-853`, V32
    // `mgr:824-827`, v0.33 `mgr:833-836`), so both state hits are dropped.
    let groups = [
        state("small", 16, true, false),
        state("large", 32, true, false),
    ];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed(pool, key(0, 4, 48));
        seed_inside(pool, &[1], 4, 32, 48);
        let request = lease(97, 4);
        assert_eq!(coordinator.find_longest_hit(pool, &request, 97).tokens, 48);
        let hit = coordinator.find_destination_hit(pool, &request, 97, 96);
        // 48 - 48 % 32.
        assert_eq!(hit.tokens, 32);
        assert_eq!(hit.per_group[0], GroupHitBlocks::None);
        assert_eq!(hit.per_group[1], GroupHitBlocks::None);
        // No group keeps a hit block, so each state counts
        // min(cdiv(97, B), 1) = 1 new block: 2.
        assert_eq!(
            coordinator.destination_admission_blocks(pool, &hit, 96, 97),
            2
        );
    }
}

#[test]
fn decode_side_hit_floors_to_the_least_common_multiple_of_every_group_block() {
    // vLLM's scheduler block is the least common multiple of the blocks of
    // all cache groups (v0.31.0 kv_cache_utils.py:767-770, v0.32.0 :848-851,
    // v0.33.0 :860-863), and the scheduler floors the local prefix to it
    // (V31 sched:941-952, V32 sched:971-982, v0.33 sched:1131-1142). A load
    // past the local prefix truncates the hit blocks of every group to the
    // floor (V31 sched:974-985, V32 sched:1007-1018, v0.33 sched:1167-1178),
    // and a Mamba list keeps min(floor / B, len) blocks (V31 mgr:852-853,
    // V32 mgr:826-827, v0.33 mgr:835-836).
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        // Blocks that are not multiples of each other: full attention 16
        // and a state group of 24 at a hash block of 8 have a scheduler
        // block of 48. The full-attention keys reach 80 and the state key
        // lies at 48, so the local prefix is 80 and its floor 48: 3
        // full-attention keys, and the state list of cdiv(48, 24) = 2
        // entries keeps min(48 / 24, 2) = 2, so the state hit stays. The
        // largest block, 24, would give 72.
        let groups = [attention("a", 16, None), state("s", 24, true, false)];
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 8, version).unwrap());
        assert_eq!(coordinator.rules().scheduler_block, 48, "{version:?}");
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 8, &[16, 32, 48, 64, 80]);
        seed(pool, key(1, 8, 48));
        let hit = coordinator.find_destination_hit(pool, &lease(161, 8), 161, 160);
        assert_eq!((hit.tokens, hit.junction_tokens), (48, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::Attention(
                [16, 32, 48]
                    .iter()
                    .map(|&end| Some(key(0, 8, end)))
                    .collect()
            ),
            "{version:?}"
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::DroppedState {
                key: key(1, 8, 48),
                tokens: 48
            },
            "{version:?}"
        );

        // A group that is not prefix-cacheable enters the scheduler block:
        // full attention 16, a state group of 32 and a ring of 24 at a hash
        // block of 4 have a scheduler block of 96. The full-attention keys
        // reach 176 and the state key lies at 96, so the local prefix is
        // 176 and its floor 96: 6 full-attention keys, and the state list
        // of 3 entries keeps min(96 / 32, 3) = 3. The blocks of the
        // prefix-cacheable groups alone, 16 and 32, would give 160.
        let groups = [
            attention("a", 16, None),
            state("s", 32, true, false),
            state("ring", 24, false, false),
        ];
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        assert_eq!(coordinator.rules().scheduler_block, 96, "{version:?}");
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(
            pool,
            0,
            4,
            &[16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176],
        );
        seed(pool, key(1, 4, 96));
        let hit = coordinator.find_destination_hit(pool, &lease(257, 4), 257, 256);
        assert_eq!((hit.tokens, hit.junction_tokens), (96, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::Attention(
                [16, 32, 48, 64, 80, 96]
                    .iter()
                    .map(|&end| Some(key(0, 4, end)))
                    .collect()
            ),
            "{version:?}"
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::DroppedState {
                key: key(1, 4, 96),
                tokens: 96
            },
            "{version:?}"
        );
        assert_eq!(hit.per_group[2], GroupHitBlocks::None, "{version:?}");
    }
}

#[test]
fn decode_side_full_hit_leaves_the_state_groups_without_a_hit() {
    // A local prefix that reaches the loaded tokens loads nothing in vLLM
    // (nixl/pull_scheduler.py:59-65). Replay still moves the state then.
    let groups = [attention("full", 16, None), state("s", 16, true, false)];
    let coordinator = with_limits(rules(&groups, 16));
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, 16, &[16, 32, 48, 64]);
    seed(pool, key(1, 16, 64));
    let hit = coordinator.find_destination_hit(pool, &lease(65, 16), 65, 64);
    assert_eq!(hit.tokens, 64);
    assert_eq!(hit.per_group[1], GroupHitBlocks::None);
}

#[test]
fn sliding_window_hits_need_a_contiguous_run_before_an_aligned_end() {
    // single:960-1069: `cdiv(32, 16) == 2` contiguous blocks before an
    // aligned end, or a run from block 0.
    let groups = [
        attention("full", 16, None),
        attention("window", 16, Some(33)),
    ];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let r = GroupedPrefixRules::new(&groups, 16, version).unwrap();
        assert_eq!(r.alignment, 16);
        let coordinator = with_limits(r);
        let mut kv = kv_pool(&groups, 32);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16, 32, 48, 64, 80, 96]);
        seed_many(pool, 1, 16, &[32, 48, 64, 96]);
        let hit = coordinator.find_longest_hit(pool, &lease(96, 16), 96);
        // The end 64 has blocks 2 and 3 before it. The end 80 lacks block 4.
        assert_eq!(hit.tokens, 64);
        match &hit.per_group[1] {
            GroupHitBlocks::Attention(keys) => {
                assert_eq!(keys.len(), 4);
                assert!(keys[0].is_none() && keys[1].is_none());
                assert_eq!(keys[2], Some(key(1, 16, 48)));
                assert_eq!(keys[3], Some(key(1, 16, 64)));
            }
            other => panic!("{other:?}"),
        }
        // Junction: the full-attention hit before the loop (80) minus 64.
        assert_eq!(hit.junction_tokens, 16);

        let mut kv = kv_pool(&groups, 32);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16, 32, 48, 64, 80, 96]);
        seed_many(pool, 1, 16, &[16]);
        let hit = coordinator.find_longest_hit(pool, &lease(96, 16), 96);
        assert_eq!(hit.tokens, 16);
    }
}

#[test]
fn one_token_window_hits_only_at_a_cached_block() {
    // A window of one token needs `cdiv(0, 4) == 0` contiguous blocks
    // (v0.31 single:961-971, v0.32 :943-953). vLLM counts a block only
    // after it finds the block cached, so a hit still needs the cached
    // block at its end (v0.31 single:1021-1044, v0.32 :1003-1026).
    let groups = [attention("window", 4, Some(1))];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 32);
        let pool = kv.block_pool_mut().unwrap();
        let hit = coordinator.find_longest_hit(pool, &lease(100, 4), 100);
        assert_eq!((hit.tokens, hit.junction_tokens), (0, 0), "{version:?}");
        seed_many(pool, 0, 4, &[48]);
        let hit = coordinator.find_longest_hit(pool, &lease(100, 4), 100);
        assert_eq!((hit.tokens, hit.junction_tokens), (48, 0), "{version:?}");
        match &hit.per_group[0] {
            GroupHitBlocks::Attention(keys) => {
                assert_eq!(keys.len(), 12, "{version:?}");
                assert!(keys[..11].iter().all(Option::is_none), "{version:?}");
                assert_eq!(keys[11], Some(key(0, 4, 48)), "{version:?}");
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn sliding_window_admission_reserves_the_window_plus_the_in_flight_tokens() {
    // iface:859-866: `cdiv(min(window - 1 + max_in_flight_tokens,
    // max_model_len), block) + 1`, with `max_in_flight_tokens =
    // max_concurrent_batches * max_num_batched_tokens` (vllm.py:605-613).
    let groups = [
        attention("full", 16, None),
        attention("window", 16, Some(33)),
    ];
    let prompt: usize = 100_000;
    let hit = GroupedHit {
        tokens: 0,
        junction_tokens: 0,
        per_group: vec![
            GroupHitBlocks::Attention(Vec::new()),
            GroupHitBlocks::Attention(Vec::new()),
        ],
    };
    let full_blocks = prompt.div_ceil(16);
    let kv = kv_pool(&groups, 16);
    let pool = kv.block_pool().unwrap();
    // One batch of 8,192: (32 + 8192) / 16 + 1 = 515.
    let one_batch = with_limits(rules(&groups, 16));
    assert_eq!(
        one_batch.admission_blocks(pool, &hit, prompt),
        full_blocks + 515
    );
    // Two batches of 8,192: (32 + 16384) / 16 + 1 = 1027.
    let two_batches = GroupedPrefixCoordinator::new(
        rules(&groups, 16),
        AdmissionLimits {
            max_in_flight_tokens: 2 * BUDGET,
            max_model_len: usize::MAX,
        },
    );
    assert_eq!(
        two_batches.admission_blocks(pool, &hit, prompt),
        full_blocks + 1027
    );
    // The model length caps it: cdiv(4096, 16) + 1 = 257.
    let capped = GroupedPrefixCoordinator::new(
        rules(&groups, 16),
        AdmissionLimits {
            max_in_flight_tokens: 2 * BUDGET,
            max_model_len: 4096,
        },
    );
    assert_eq!(
        capped.admission_blocks(pool, &hit, prompt),
        full_blocks + 257
    );
    // A short prompt needs fewer blocks than the cap.
    assert_eq!(one_batch.admission_blocks(pool, &hit, 100), 7 + 7);
}

#[test]
fn sliding_window_admission_after_a_hit_is_vllms_whole_prompt_count() {
    // vLLM's whole-prompt test (v0.31 mgr:519-536, v0.32 mgr:522-539,
    // v0.33 mgr:531-548) for a sliding-window group (v0.31
    // single:216-268, v0.32 single:213-265, v0.33 single:214-266):
    //   required = min(cdiv(prompt, B), cap), with
    //   cap = cdiv(min(window - 1 + in_flight, max_model_len), B) + 1
    //         (v0.31 iface:859-866, v0.32 iface:878-885, v0.33
    //         iface:937-944);
    //   skipped = max(0, hit - window + 1) // B
    //         (v0.31 single:1185-1188, v0.32 single:1167-1170, v0.33
    //         single:1180-1183);
    //   new = max(required - max(skipped, len(hit blocks)), 0);
    //   plus the hit blocks from `skipped` on that no request holds.
    // The hit list has one entry per hit block, null blocks included.
    // vLLM compares the capped count with absolute block indices, so a
    // long hit gives no new block here. vLLM's second test, on the first
    // chunk and without the cap (v0.31 mgr:558-576, v0.32 mgr:561-579,
    // v0.33 mgr:570-588), is a separate count.
    //
    // C1, window 8, block 4, in flight 8, model length 200, prompt 108,
    // hit 96: cap = cdiv(min(7 + 8, 200), 4) + 1 = 5; required =
    // min(27, 5) = 5; skipped = 89 // 4 = 22; the hit list has 24
    // entries, so new = max(5 - 24, 0) = 0; the two hit blocks of the
    // window run count when no request holds them: 2, else 0.
    // (window, block, in flight, model length, prompt, hit, the cached
    // blocks that end at the hit, not held, held)
    type Case = (u32, u32, usize, usize, usize, usize, usize, usize, usize);
    let cases: [Case; 6] = [
        (8, 4, 8, 200, 108, 96, 2, 2, 0),
        (8, 4, 8, 200, 200, 96, 2, 2, 0),
        // C3: skipped = 5 // 4 = 1, 3 entries, new = 5 - 3 = 2.
        (8, 4, 8, 200, 108, 12, 2, 4, 2),
        // C4, cold: new = min(27, 5) = 5.
        (8, 4, 8, 200, 108, 0, 0, 5, 5),
        // C5: cap = cdiv(13, 4) + 1 = 5, skipped = 91 // 4 = 22.
        (6, 4, 8, 200, 108, 96, 2, 2, 0),
        // C6: cap = cdiv(min(127 + 16384, 131072), 16) + 1 = 1,033;
        // skipped = 49,873 // 16 = 3,117; 3,125 entries; new = 0; the
        // eight blocks of the window run.
        (128, 16, 16384, 131_072, 100_000, 50_000, 8, 8, 0),
    ];
    for (window, block, in_flight, model_len, prompt, hit_tokens, run, idle, held) in cases {
        let groups = [attention("window", block, Some(window))];
        for version in [
            VllmRuleVersion::V0_31_0,
            VllmRuleVersion::V0_32_0,
            VllmRuleVersion::V0_33_0,
        ] {
            let coordinator = GroupedPrefixCoordinator::new(
                GroupedPrefixRules::new(&groups, block as usize, version).unwrap(),
                AdmissionLimits {
                    max_in_flight_tokens: in_flight,
                    max_model_len: model_len,
                },
            );
            let mut kv = kv_pool(&groups, 16);
            let pool = kv.block_pool_mut().unwrap();
            let ends: Vec<usize> = (0..run).map(|i| hit_tokens - i * block as usize).collect();
            seed_many(pool, 0, block as usize, &ends);
            let request = lease(prompt, block as usize);
            let hit = coordinator.find_longest_hit(pool, &request, prompt);
            let case = format!("{version:?}, window {window}, prompt {prompt}");
            assert_eq!(hit.tokens, hit_tokens, "{case}");
            assert_eq!(
                coordinator.admission_blocks(pool, &hit, prompt),
                idle,
                "{case}"
            );
            let mut block_lease = coordinator.new_lease();
            coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
            assert_eq!(
                coordinator.admission_blocks(pool, &hit, prompt),
                held,
                "{case}"
            );
            coordinator.release_request(pool, &mut block_lease);
        }
    }
}

#[test]
fn remaining_blocks_follow_vllm_for_every_group_kind() {
    // GLM at 4,352 of 10,000 tokens: full attention 5 - 2 = 3, the ring 0,
    // and each KDA group 1, or 2 with a valid checkpoint at 8,704.
    // The checkpoint passes the five conditions of V32 `iface:1125-1146`
    // for the query 4,352..10,000 at 9,999 / 2,176 * 2,176 = 8,704, and
    // vLLM counts min(5 - 2, 1 + 1) = 2 per KDA group (V32
    // `single:1724-1773`).
    for (version, checkpoint, expected) in [
        (VllmRuleVersion::V0_31_0, false, 7),
        (VllmRuleVersion::V0_32_0, false, 7),
        (VllmRuleVersion::V0_32_0, true, 11),
    ] {
        let groups = groups_glm(checkpoint);
        let mut coordinator = with_limits(rules_glm(version, checkpoint));
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let mut lease = coordinator.new_lease();
        assert!(coordinator.allocate(pool, &mut lease, 10_000, 0, 4_352, 0));
        assert_eq!(
            coordinator.remaining_blocks(&lease, 10_000, 4_352, 10_000),
            expected
        );
    }
    // A 16-token window at 64 of 65 tokens: 5 blocks. One batch of 32 tokens
    // caps it at cdiv(15 + 32, 16) + 1 = 4, equal to the 4 held, so 0.
    // One batch of 8,192 tokens caps it at 514, so 5 - 4 = 1 (V31
    // `single:216-238`, `iface:859-866`).
    let groups = [attention("window", 16, Some(16))];
    let limits = AdmissionLimits {
        max_in_flight_tokens: 32,
        max_model_len: usize::MAX,
    };
    for (mut coordinator, expected) in [
        (GroupedPrefixCoordinator::new(rules(&groups, 16), limits), 0),
        (with_limits(rules(&groups, 16)), 1),
    ] {
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        let mut lease = coordinator.new_lease();
        assert!(coordinator.allocate(pool, &mut lease, 65, 0, 64, 0));
        assert_eq!(coordinator.remaining_blocks(&lease, 65, 64, 65), expected);
    }
}

#[test]
fn remaining_blocks_count_the_producer_copy_that_is_still_to_come() {
    // Attention and state blocks of 64, hash block 4, a cold prompt of 50.
    // vLLM's whole-prompt test counts 2 blocks. The split stops at the tail
    // 48, where the state is published inside its block, and the write
    // 48..50 takes a third block for the copy. Until then vLLM counts that
    // block in what the request still needs: the request is a producer in
    // `_partial_hit_reqs` (V31 single:1770-1777, V32 single:1732-1739), and
    // `_request_remaining_blocks` counts through the same function (V31
    // sched:3003-3017, V32 sched:3060-3076).
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let groups = [attention("full", 64, None), state("s", 64, true, false)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 3);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        assert_eq!(
            coordinator.admission_blocks(pool, &hit, 50),
            2,
            "{version:?}"
        );
        let mut block_lease = coordinator.new_lease();
        assert_eq!(
            coordinator.remaining_blocks(&block_lease, 50, 0, 50),
            2,
            "{version:?}"
        );
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 50, 0, 48, 0));
        coordinator.finalize(pool, &mut block_lease, &request, 50, 48, &hit);
        assert!(
            state_lease(&block_lease, 1).published.is_some(),
            "{version:?}"
        );
        assert_eq!(
            coordinator.remaining_blocks(&block_lease, 50, 48, 50),
            1,
            "{version:?}"
        );
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 50, 48, 50, 48));
        assert_eq!(pool.active_blocks(), 3, "{version:?}");
        assert_eq!(
            coordinator.remaining_blocks(&block_lease, 50, 50, 50),
            0,
            "{version:?}"
        );
        coordinator.release_request(pool, &mut block_lease);
    }
}

#[test]
fn remaining_bytes_saturate_without_limits() {
    // A window of 1-token blocks with no model length and no budget: the cap
    // saturates at usize::MAX and does not bind, so 12 - 10 = 2 blocks.
    let window = FpmCacheGroup {
        page_size_bytes: 10,
        ..attention("w", 1, Some(4))
    };
    let limits = AdmissionLimits {
        max_in_flight_tokens: usize::MAX,
        max_model_len: usize::MAX,
    };
    assert_eq!(remaining_bytes(&[window], 10, 12, limits), 20);
}

#[test]
fn destination_admission_bytes_equal_the_blocks_of_block_mode() {
    // Full attention, a window of 33 tokens and a state, with 16-token blocks
    // and one batch of 32 tokens: the window cap is cdiv(32 + 32, 16) + 1 = 5
    // blocks (V31 `iface:859-866`, V32 `iface:878-885`, v0.33
    // `iface:937-944`). A decode rank loads 160 of 161 prompt tokens with no
    // local hit. vLLM's whole-prompt test counts cdiv(161, 16) = 11
    // full-attention blocks, max(min(11, 5) - floor((160 - 32) / 16), 0) =
    // max(5 - 8, 0) = 0 window blocks (V31 `single:216-251`, V32
    // `single:213-248`, v0.33 `single:214-249`) and 1 state block: 12.
    let page = 100;
    let groups = [
        attention("full", 16, None),
        attention("window", 16, Some(33)),
        state("s", 16, true, false),
    ]
    .map(|group| FpmCacheGroup {
        page_size_bytes: page,
        ..group
    });
    let limits = AdmissionLimits {
        max_in_flight_tokens: 32,
        max_model_len: 200_000,
    };
    let bytes = |loaded: usize, tokens: usize| {
        // Byte mode keeps the state groups apart, as one fixed charge.
        let pool = GroupedKvPool::new(groups.to_vec(), 64 * page);
        destination_admission_bytes(
            pool.groups(),
            pool.fixed_request_bytes(),
            loaded,
            tokens,
            limits,
        )
    };
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let rules = GroupedPrefixRules::new(&groups, 16, version).unwrap();
        let coordinator = GroupedPrefixCoordinator::new(rules, limits);
        let mut kv = GroupedKvPool::with_prefix_cache(groups.to_vec(), 64 * page, page);
        let pool = kv.block_pool_mut().unwrap();
        let blocks = |loaded: usize, tokens: usize| {
            coordinator.destination_admission_blocks(pool, &coordinator.empty_hit(), loaded, tokens)
        };
        // 11 + 0 + 1 blocks of 100 bytes.
        assert_eq!(bytes(160, 161), 1_200);
        assert_eq!(blocks(160, 161), 12);
        // The cap binds and the window is not skipped past it: loaded 96 of
        // 97 counts cdiv(97, 16) = 7 full-attention blocks, min(7, 5) -
        // floor((96 - 32) / 16) = 5 - 4 = 1 window block and 1 state block:
        // 9.
        assert_eq!(bytes(96, 97), 900);
        assert_eq!(blocks(96, 97), 9);
        // Below the cap: loaded 32 of 33 counts cdiv(33, 16) = 3
        // full-attention blocks, min(3, 5) - 0 = 3 window blocks and 1 state
        // block: 7.
        assert_eq!(bytes(32, 33), 700);
        assert_eq!(blocks(32, 33), 7);
        // The window of the loaded tokens, not of the prompt, sets the
        // skipped blocks: loaded 47 of 48 counts cdiv(48, 16) = 3
        // full-attention blocks, min(3, 5) - floor((47 - 32) / 16) = 3 - 0 =
        // 3 window blocks and 1 state block: 7.
        assert_eq!(bytes(47, 48), 700);
        assert_eq!(blocks(47, 48), 7);
    }
    // A request of no tokens holds no block, so it counts no state bytes.
    assert_eq!(bytes(0, 0), 0);
    // A state group in the list counts only through its fixed bytes.
    assert_eq!(
        destination_admission_bytes(&groups, page, 160, 161, limits),
        1_200
    );
}

#[test]
fn state_group_hits_only_at_a_kept_boundary() {
    // single:1487-1564 scans aligned ends right to left for one cached
    // state block. pool:198-224: every group of the spec group must hold
    // the key.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &GLM_STATES, hash, 8704);
        let request = lease(9000, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        assert_eq!(hit.tokens, 8704);
        assert_eq!(hit.junction_tokens, 0);
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, hash, 8704)))
        );
        assert_eq!(hit.per_group[1], GroupHitBlocks::None);

        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &[2, 4, 5], hash, 8704);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        assert_eq!(hit.tokens, 0);
        assert_eq!(hit.per_group[0], GroupHitBlocks::Attention(Vec::new()));
        assert_eq!(hit.per_group[3], GroupHitBlocks::State(None));

        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &GLM_STATES, hash, 6528);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        assert_eq!(hit.tokens, 6528);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 3));
    }
}

#[test]
fn ring_group_never_hits_and_holds_one_block() {
    // single:1214-1230 claims one block per request; single:498-499
    // caches nothing for a group that is not prefix cacheable.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let mut coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(2500, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 2500);
        assert_eq!(hit.tokens, 0);
        let mut block_lease = coordinator.new_lease();
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 2500, 0, 2176, 0));
        assert_eq!(
            block_lease.groups[1],
            GroupLease::Ring {
                copy: Some(ring_copy(&block_lease))
            }
        );
        coordinator.finalize(pool, &mut block_lease, &request, 2500, 2176, &hit);
        assert_eq!(pool.inactive_cached_blocks(), 0);
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 2500, 2176, 2500, 2176));
        coordinator.finalize(pool, &mut block_lease, &request, 2500, 2500, &hit);
        coordinator.release_request(pool, &mut block_lease);
        coordinator.begin_step();

        let hit = coordinator.find_longest_hit(pool, &lease(2500, hash), 2500);
        assert_eq!(hit.tokens, 2176);
        assert_eq!(hit.per_group[1], GroupHitBlocks::None);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 1));
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, hash, 2176)))
        );
    }
}

fn ring_copy(lease: &GroupedBlockLease) -> crate::engine::cache::vllm_block_pool::BlockCopyId {
    match &lease.groups[1] {
        GroupLease::Ring { copy: Some(copy) } => *copy,
        other => panic!("{other:?}"),
    }
}

#[test]
fn hybrid_hit_is_the_longest_prefix_every_cacheable_group_serves() {
    // coord:842-977: the fixed point shortens to 4,352; coord:970-977
    // reports the uncached common prefix.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704, 10880]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        let hit = coordinator.find_longest_hit(pool, &lease(11000, hash), 11000);
        assert_eq!(hit.tokens, 4352);
        assert_eq!(hit.junction_tokens, 10880 - 4352);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 2));
    }
}

#[test]
fn hybrid_partial_hash_truncates_full_attention_hit_length() {
    // Ported from v0.31 partial_test:1596. Values hand-derived: the
    // fine-grained probe of single:806-825 extends the two full blocks to
    // the partial entry at 4,356, and the state hit there keeps it.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm_hash(version, false, 4));
        let hash = 4;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352]);
        seed_inside(pool, &[0], hash, GLM_BLOCK as usize, 4356);
        seed_inside(pool, &GLM_STATES, hash, GLM_BLOCK as usize, 4356);
        let request = lease(4500, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 4500);
        assert_eq!(hit.tokens, 4356);
        assert_eq!(hit.junction_tokens, 0);
        match &hit.per_group[0] {
            GroupHitBlocks::Attention(keys) => {
                assert_eq!(keys.len(), 3);
                assert_eq!(keys[2], Some(key(0, hash, 4356)));
            }
            other => panic!("{other:?}"),
        }

        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352]);
        seed_inside(pool, &[0], hash, GLM_BLOCK as usize, 4356);
        seed_states(pool, &GLM_STATES, hash, 4352);
        let hit = coordinator.find_longest_hit(pool, &request, 4500);
        assert_eq!(hit.tokens, 4352);
        // coord:970-977: the full-attention hit of 4,356 minus 4,352.
        assert_eq!(hit.junction_tokens, 4);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 2));

        // The state search goes down through the blocks to the one position
        // where all four state groups have a key, 2,180, inside the second
        // block, and passes 3,000, where one group has a key.
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352]);
        seed_inside(pool, &[0], hash, GLM_BLOCK as usize, 4356);
        seed_inside(pool, &GLM_STATES, hash, GLM_BLOCK as usize, 2180);
        seed_inside(pool, &GLM_STATES[..1], hash, GLM_BLOCK as usize, 3000);
        let hit = coordinator.find_longest_hit(pool, &request, 4500);
        assert_eq!(hit.tokens, 2180);
        assert_eq!(hit.junction_tokens, 4356 - 2180);
    }
}

#[test]
fn shared_first_hash_miss_probes_each_distinct_offset_once() {
    use std::cell::Cell;

    let groups = [attention("full", 64, None), state("s", 64, true, false)];
    let hash_block = 4;
    let coordinator = with_limits(rules(&groups, hash_block));
    let spec = &coordinator.rules.specs[0];
    let request = lease(49, hash_block);
    let mut kv = kv_pool(&groups, 1024);
    let pool = kv.block_pool_mut().unwrap();
    for i in 0..1024 {
        // All prompts share the first hash, but diverge at the fifth token.
        let mut tokens: Vec<u32> = (0..49).collect();
        tokens[4] = 1000 + i;
        let mut cached = BlockRequestLease::new(Uuid::nil(), Vec::new());
        cached.configure_prefix_hashes(&tokens, hash_block);
        let mut reservation = pool.reserve(&[], 1).unwrap();
        let id = pool.take_private(&mut reservation);
        assert!(pool.cache_at(
            id,
            grouped_key(0, cached.prefix_hash_at(48)).unwrap(),
            key_in_block(&cached, hash_block, 64, 48),
        ));
        pool.release(id);

        if i == 0 || i == 1023 {
            let probes = Cell::new(0);
            let found = coordinator.hit_in_block(pool, &request, spec, 0, 48, |end| {
                probes.set(probes.get() + 1);
                GroupedPrefixCoordinator::spec_resident(pool, &request, spec, end)
            });
            assert_eq!(found, None);
            // Debug builds also run the independent linear-scan oracle.
            let oracle_probes = if cfg!(debug_assertions) { 48 / 4 } else { 0 };
            assert_eq!(probes.get(), 1 + oracle_probes, "{} cached keys", i + 1);
            assert_eq!(coordinator.find_longest_hit(pool, &request, 49).tokens, 0);
            assert_eq!(
                coordinator.full_attention_hit(pool, &cached, spec, 48).1,
                48,
            );
        }
    }
}

/// Two full-attention groups of one spec and a state group, all at block 64.
fn two_full_groups_and_a_state() -> [FpmCacheGroup; 3] {
    [
        attention("full_a", 64, None),
        attention("full_b", 64, None),
        state("state", 64, true, false),
    ]
}

#[test]
fn groups_of_one_spec_hit_jointly_and_set_the_junction_together() {
    // coord:737-779 (v0.32 coord:708-750) looks up the groups of one spec
    // together, and pool:198-224 finds a block only when every group of the
    // spec holds it. The longest hit is the largest result of one spec's
    // lookup (coord:947, :970-977). Group a holds 192 tokens, group b 64 and
    // the state group the state at 64, so vLLM hits 64 with no junction.
    let groups = two_full_groups_and_a_state();
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 64, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 64, &[64, 128, 192]);
        seed_many(pool, 1, 64, &[64]);
        seed_many(pool, 2, 64, &[64]);
        let hit = coordinator.find_longest_hit(pool, &lease(401, 64), 401);
        assert_eq!((hit.tokens, hit.junction_tokens), (64, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::Attention(vec![Some(key(0, 64, 64))])
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::Attention(vec![Some(key(1, 64, 64))])
        );
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, 64, 64)))
        );
        // So the first chunk runs to the last cacheable stop 384 (v0.31
        // sched:444, v0.32 sched:476-480). A junction at 192 would end it
        // there.
        let junction = hit.tokens + hit.junction_tokens;
        assert_eq!(
            split(coordinator.rules(), 64, 401, 401, BUDGET, junction),
            320,
            "{version:?}"
        );
    }
}

#[test]
fn groups_of_one_spec_hit_inside_a_block_only_jointly() {
    // With fine-grained hits, single:794-828 probes the hash boundaries of the
    // first partial block for every group of the spec at once. Group a holds
    // two full blocks, group b one full block and the partial entry at 96.
    // No boundary inside the second block is cached in both, so vLLM hits 64.
    let groups = two_full_groups_and_a_state();
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let rules = GroupedPrefixRules::new(&groups, 16, version).unwrap();
        assert_eq!(rules.alignment, 16);
        let coordinator = with_limits(rules);
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[64, 128]);
        seed_many(pool, 1, 16, &[64, 96]);
        seed_many(pool, 2, 16, &[64, 96]);
        let hit = coordinator.find_longest_hit(pool, &lease(200, 16), 200);
        assert_eq!((hit.tokens, hit.junction_tokens), (64, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::Attention(vec![Some(key(0, 16, 64))])
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::Attention(vec![Some(key(1, 16, 64))])
        );
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, 16, 64)))
        );
    }
}

#[test]
fn state_groups_of_one_spec_hit_jointly() {
    // Without a full-attention group, the first spec looked up is the first
    // one in group order (coord:775-779), here the two state groups. They
    // hold the states at 64 and 192, and at 64. The joint lookup
    // (single:1540-1564) hits 64 and leaves no junction.
    let groups = [
        state("state_a", 64, true, false),
        state("state_b", 64, true, false),
        attention("window", 64, Some(65)),
    ];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 64, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 64, &[64, 192]);
        seed_many(pool, 1, 64, &[64]);
        seed_many(pool, 2, 64, &[64]);
        let hit = coordinator.find_longest_hit(pool, &lease(401, 64), 401);
        assert_eq!((hit.tokens, hit.junction_tokens), (64, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::State(Some(key(0, 64, 64)))
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::State(Some(key(1, 64, 64)))
        );
    }
    // In this helper the full-attention group is G0. vLLM looks up the
    // full-attention spec first in any group order (`coord:775-779`), so
    // its hit sets the junction. The extra state of G2 at 8,704 changes
    // nothing, and the four KDA groups hit 4,352 together.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        seed_states(pool, &[2], hash, 8704);
        let hit = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(
            (hit.tokens, hit.junction_tokens),
            (4352, 8704 - 4352),
            "{version:?}"
        );
        for group in GLM_STATES {
            assert_eq!(
                hit.per_group[usize::from(group)],
                GroupHitBlocks::State(Some(key(group, hash, 4352)))
            );
        }
    }
}

#[test]
fn admission_counts_inactive_hits_and_state_copies() {
    // single:216-268 counts inactive hit blocks again. The keys are seeded
    // directly, because a finished request of 4,352 tokens keeps its state
    // key only at its replay boundary, 2,176.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        let hit = coordinator.find_longest_hit(pool, &lease(4400, hash), 4400);
        assert_eq!(hit.tokens, 4352);
        // 2 inactive G0 blocks, 1 new G0 block, 1 ring, 4 x (running +
        // inactive cached state block).
        assert_eq!(coordinator.admission_blocks(pool, &hit, 4400), 12);
        // Another running request holds the two G0 blocks active.
        let other = [key(0, hash, 2176), key(0, hash, 4352)];
        let mut reservation = pool.reserve(&other, 0).unwrap();
        let held = pool.activate(&mut reservation);
        assert_eq!(held.len(), 2);
        assert_eq!(coordinator.admission_blocks(pool, &hit, 4400), 10);
        pool.cancel(reservation);
    }
}

#[test]
fn resumed_state_holds_cached_and_running_blocks_during_the_first_step() {
    // single:1858-1868: a resumed request holds the hit block and the
    // running block. vLLM frees the hit block at the first `allocate_slots`
    // of the request (mgr:552-556, single:1626-1645) with
    // `cdiv(processed, block) - 1 > hit_index` (single:1667-1671). With one
    // batch the processed count of the second allocation is 6,000:
    // `cdiv(6000, 2176) - 1 == 2 > 1`.
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let hash = 2176;
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[2176, 4352]);
    seed_states(pool, &GLM_STATES, hash, 4352);
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!(hit.tokens, 4352);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert_eq!(state.hit.map(|(_, index)| index), Some(1));
        assert!(state.running.is_none());
    }
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 4352, 6000, 4352));
    let state_blocks: usize = GLM_STATES
        .iter()
        .map(|&group| {
            let state = state_lease(&block_lease, usize::from(group));
            assert!(state.hit.is_some() && state.running.is_some());
            assert!(state.previous.is_empty() && state.copy.is_none());
            2
        })
        .sum();
    assert_eq!(state_blocks, 8);
    // G0: 2 hits + 1 new, ring 1, states 8.
    assert_eq!(pool.active_blocks(), 12);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 6000, &hit);
    // The hit blocks stay through the first step.
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.hit.is_some() && state.running.is_some());
    }
    assert_eq!(pool.active_blocks(), 12);
    assert_eq!(pool.inactive_cached_blocks(), 0);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 6000, 6001, 6000));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.hit.is_none() && state.running.is_some());
    }
    assert_eq!(pool.active_blocks(), 8);
    assert_eq!(pool.inactive_cached_blocks(), 4);
}

#[test]
fn boundary_crossing_keeps_the_previous_running_block_for_one_step() {
    // single:1647-1675: the block of the earlier boundary goes at the first
    // allocation whose processed count lies past the end of that block. With
    // one batch the processed count is the computed count, so it goes at the
    // next allocation.
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let hash = 2176;
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 0, 2176, 0));
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 2176, &hit);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 2176, 4352, 2176));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.running.is_some());
        assert_eq!(
            previous_rules(&block_lease, usize::from(group)),
            [PreviousRelease::Above(2176)]
        );
    }
    assert_eq!(state_and_ring_blocks(&block_lease), 9);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 4352, &hit);
    assert_eq!(state_and_ring_blocks(&block_lease), 9);
    coordinator.begin_step();
    let before = pool.active_blocks();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 4352, 6528, 4352));
    // Four previous blocks go, four new running blocks and one G0 block come.
    assert_eq!(pool.active_blocks(), before + 1);
    for group in GLM_STATES {
        assert_eq!(
            previous_rules(&block_lease, usize::from(group)),
            [PreviousRelease::Above(4352)]
        );
    }
    assert_eq!(state_and_ring_blocks(&block_lease), 9);
}

#[test]
fn partial_hit_takes_a_copy_block_for_the_shared_block() {
    // single:387-409 and single:1770-1777: a hit that ends inside a block
    // takes one copy block. single:800-805: the full-attention probe needs
    // the full blocks below the partial entry, so blocks 0 and 1 are
    // cached too.
    let groups = [attention("full", 16, None), state("s", 32, true, false)];
    let r = rules(&groups, 4);
    assert_eq!(r.alignment, 4);
    let mut coordinator = with_limits(r);
    let hash = 4;
    let mut kv = kv_pool(&groups, 32);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[16, 32]);
    seed_inside(pool, &[0], hash, 16, 36);
    seed_inside(pool, &[1], hash, 32, 36);
    let request = lease(50, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 50);
    assert_eq!(hit.tokens, 36);
    // G0: 4 live - 3 hits + 3 inactive + 1 copy = 5. State
    // (single:1757-1777): `cdiv(50, 32) - cdiv(36, 32) == 0` new blocks
    // plus the copy block, plus the inactive hit = 2.
    assert_eq!(coordinator.admission_blocks(pool, &hit, 50), 7);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert_eq!(pool.active_blocks(), 4);
    assert!(coordinator.allocate(pool, &mut block_lease, 50, 36, 50, 36));
    // G0 holds blocks 0 to 3, with block 2 as a private copy. The target
    // 50 stays in the hit's block, so the state's copy-on-write block is
    // its running block (single:1891-1895, `_apply_cow` at
    // single:454-474). The shared block 2 and the state's hit block are
    // the copy sources. They leave the lease and stay active until the
    // scheduling pass ends (sched:1434-1442).
    assert_eq!(attention_copies(&block_lease, 0), 4);
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_none() && state.running.is_some() && state.copy.is_none());
    assert_eq!(coordinator.copy_source_count(), 2);
    assert_eq!(pool.active_blocks(), 4 + 1 + 2);
    assert_eq!(pool.inactive_cached_blocks(), 0);
    coordinator.finalize(pool, &mut block_lease, &request, 50, 50, &hit);
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_none() && state.running.is_some());
    assert_eq!(coordinator.copy_source_count(), 2);
    assert_eq!(pool.active_blocks(), 4 + 1 + 2);
    assert_eq!(pool.inactive_cached_blocks(), 0);
    assert_eq!(coordinator.release_copy_sources(pool), 2);
    assert_eq!(pool.active_blocks(), 4 + 1);
    assert_eq!(pool.inactive_cached_blocks(), 2);

    // A target beyond the hit's block: the copy sits at the hit's index
    // and a running block follows. `remove_skipped_blocks` frees the
    // copy at the next allocation (single:1658-1675, index set at :1868).
    let mut kv = kv_pool(&groups, 32);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[16, 32]);
    seed_inside(pool, &[0], hash, 16, 36);
    seed_inside(pool, &[1], hash, 32, 36);
    let request = lease(70, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 70);
    assert_eq!(hit.tokens, 36);
    // G0: 5 live - 3 hits + 3 inactive + 1 copy = 6. State: `cdiv(70, 32)
    // - cdiv(36, 32) == 1` new block plus the copy block plus the
    // inactive hit = 3.
    assert_eq!(coordinator.admission_blocks(pool, &hit, 70), 9);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 70, 36, 70, 36));
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_none() && state.running.is_some() && state.copy.is_some());
    assert_eq!(coordinator.copy_source_count(), 2);
    assert_eq!(pool.active_blocks(), 5 + 2 + 2);
    coordinator.finalize(pool, &mut block_lease, &request, 70, 70, &hit);
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_none() && state.copy.is_some());
    assert_eq!(pool.active_blocks(), 5 + 2 + 2);
    assert_eq!(coordinator.release_copy_sources(pool), 2);
    assert_eq!(pool.active_blocks(), 5 + 2);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 70, 70, 71, 70));
    let state = state_lease(&block_lease, 1);
    assert!(state.copy.is_none() && state.running.is_some());
    assert_eq!(pool.active_blocks(), 5 + 1);
}

#[test]
fn two_previous_state_blocks_stay_while_the_step_before_is_in_flight() {
    // Rule of v0.33 single:1608-1657 with the skipped tokens of
    // single:1971-1976: the block with index `i` goes at the first
    // allocation with `processed > (i + 1) * B`, where processed is
    // `computed - num_in_flight_tokens` (v0.33 mgr:561-568). With two
    // batches the step before is in flight, so each allocation sees the
    // computed count before that step. Previous blocks are listed by the
    // token count above which they go; the crossing from `c` stores
    // `((c - 1) / B + 1) * B`.
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let hash = 2176;
    let above = |counts: &[usize]| -> Vec<PreviousRelease> {
        counts.iter().map(|&t| PreviousRelease::Above(t)).collect()
    };
    // (c, t, processed with two batches, previous blocks, state groups + ring)
    let two_batches: [(usize, usize, usize, &[usize], usize); 7] = [
        (0, 2176, 0, &[], 5),
        (2176, 4352, 0, &[2176], 9),
        (4352, 6528, 2176, &[2176, 4352], 13),
        (6528, 8704, 4352, &[4352, 6528], 13),
        (8704, 9000, 6528, &[6528, 8704], 13),
        (9000, 9001, 8704, &[8704], 9),
        (9001, 9002, 9000, &[], 5),
    ];
    for batches in [2, 1] {
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(9000, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        let mut block_lease = coordinator.new_lease();
        let mut counts = Vec::new();
        for (row, &(c, t, processed, previous, held)) in two_batches.iter().enumerate() {
            let step = row as u64 + 1;
            // With one batch nothing is in flight: processed = computed.
            let processed = if batches == 2 { processed } else { c };
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 9000, c, t, processed));
            if batches == 2 {
                for group in GLM_STATES {
                    let rules = previous_rules(&block_lease, usize::from(group));
                    assert_eq!(rules, above(previous), "step {step}, group {group}");
                }
                assert_eq!(state_and_ring_blocks(&block_lease), held, "step {step}");
            }
            counts.push(state_and_ring_blocks(&block_lease));
            coordinator.finalize(pool, &mut block_lease, &request, 9000, t, &hit);
        }
        let expected: &[usize] = if batches == 2 {
            &[5, 9, 13, 13, 13, 9, 5]
        } else {
            // One previous block at most: the crossing of step `s` frees the
            // block of step `s - 1`, and the decode steps 6 and 7 free the
            // last one (9000 > 8704).
            &[5, 9, 9, 9, 9, 5, 5]
        };
        assert_eq!(counts, expected, "{batches} batches");
    }
}

#[test]
fn checkpoint_block_stays_while_its_step_is_in_flight() {
    // The checkpoint block of a final chunk from `c` to `t` has index
    // `cdiv(t, B) - 2` (v0.33 single:1797-1923) and goes by the rule of
    // single:1608-1657: at the first allocation with processed past its
    // position. With two batches, the allocation after its step still sees
    // the computed count before that step.
    let groups = groups_glm(true);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_33_0, true));
    let hash = 2176;
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    // (c, t, processed, previous blocks, checkpoint, blocks of one state group)
    type Row = (usize, usize, usize, &'static [usize], bool, usize);
    let rows: [Row; 6] = [
        (0, 2176, 0, &[], false, 1),
        (2176, 4352, 0, &[2176], false, 2),
        (4352, 6528, 2176, &[2176, 4352], false, 3),
        // Index 0 goes (4352 > 2176). The chunk reaches the prefill end and
        // spans two blocks, so it takes the checkpoint block at 8704. vLLM:
        // `[null, b1, b2, checkpoint, running]`.
        (6528, 9000, 4352, &[4352, 6528], true, 4),
        // 6528 > 4352 frees one. The checkpoint stays: 6528 > 8704 is false.
        (9000, 9001, 6528, &[6528], true, 3),
        // 9000 > 8704 and 9000 > 6528.
        (9001, 9002, 9000, &[], false, 1),
    ];
    let mut cached = 0;
    for (row, &(c, t, processed, previous, checkpoint, blocks)) in rows.iter().enumerate() {
        let step = row as u64 + 1;
        coordinator.begin_step();
        cached = pool.inactive_cached_blocks();
        assert!(coordinator.allocate(pool, &mut block_lease, 9000, c, t, processed));
        for group in GLM_STATES {
            let state = state_lease(&block_lease, usize::from(group));
            let expected: Vec<_> = previous
                .iter()
                .map(|&p| PreviousRelease::Above(p))
                .collect();
            assert_eq!(
                previous_rules(&block_lease, usize::from(group)),
                expected,
                "step {step}"
            );
            assert!(state.running.is_some());
            assert_eq!(
                state.checkpoint.map(|(_, pos)| pos),
                checkpoint.then_some(8704),
                "step {step}"
            );
            let held = 1 + state.previous.len() + usize::from(state.checkpoint.is_some());
            assert_eq!(held, blocks, "step {step}");
        }
        coordinator.finalize(pool, &mut block_lease, &request, 9000, t, &hit);
    }
    // The finalize of step 4 published the prompt checkpoint at 8704. The
    // four checkpoint blocks enter the LRU in step 6; the previous blocks
    // hold no key in this run.
    assert_eq!(pool.inactive_cached_blocks(), cached + 4);
}

#[test]
fn boundary_hit_block_stays_while_the_first_step_is_in_flight() {
    // The hit block of a hit on a block boundary at `h` tokens has index
    // `h / B - 1` and goes at the first allocation with
    // `cdiv(processed, B) - 1 > index` (v0.33 single:1629-1657).
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let hash = 2176;
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[2176, 4352]);
    seed_states(pool, &GLM_STATES, hash, 4352);
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!(hit.tokens, 4352);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 4352, 6000, 4352));
    // G0: 2 hits + 1 new, ring 1, states 4 x (hit + running).
    assert_eq!(pool.active_blocks(), 12);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 6000, &hit);
    // Step 1 is in flight: processed 4352, and
    // `cdiv(4352, 2176) - 1 = 1 > 1` is false.
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 6000, 6001, 4352));
    for group in GLM_STATES {
        assert!(state_lease(&block_lease, usize::from(group)).hit.is_some());
    }
    assert_eq!(pool.active_blocks(), 12);
    assert_eq!(pool.inactive_cached_blocks(), 0);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 6001, &hit);
    // Processed 6000: `cdiv(6000, 2176) - 1 = 2 > 1`, the hits go.
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 6001, 6002, 6000));
    for group in GLM_STATES {
        assert!(state_lease(&block_lease, usize::from(group)).hit.is_none());
    }
    assert_eq!(pool.active_blocks(), 8);
    assert_eq!(pool.inactive_cached_blocks(), 4);
}

#[test]
fn copy_block_stays_while_the_first_step_is_in_flight() {
    // The copy block of a hit inside a block at 36 tokens replaces the hit
    // block at index `36 / 32 = 1` (v0.33 single:455-475), and
    // `last_state_block_idx` is 1 (v0.33 single:1844-1847). It goes at the
    // first allocation with `1 < cdiv(processed, 32) - 1`, that is
    // `processed > (1 + 1) * 32 = 64` (v0.33 single:1649-1657).
    let groups = [attention("full", 16, None), state("s", 32, true, false)];
    let mut coordinator = with_limits(rules(&groups, 4));
    let hash = 4;
    let mut kv = kv_pool(&groups, 32);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[16, 32]);
    seed_inside(pool, &[0], hash, 16, 36);
    seed_inside(pool, &[1], hash, 32, 36);
    let request = lease(70, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 70);
    assert_eq!(hit.tokens, 36);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 70, 36, 70, 36));
    coordinator.finalize(pool, &mut block_lease, &request, 70, 70, &hit);
    assert_eq!(coordinator.release_copy_sources(pool), 2);
    assert_eq!(pool.active_blocks(), 5 + 2);
    // Step 1 is in flight: processed 36, and 36 > 64 is false.
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 70, 70, 71, 36));
    assert!(state_lease(&block_lease, 1).copy.is_some());
    assert_eq!(pool.active_blocks(), 5 + 2);
    coordinator.finalize(pool, &mut block_lease, &request, 70, 71, &hit);
    // Processed 70 > 64.
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 70, 71, 72, 70));
    assert!(state_lease(&block_lease, 1).copy.is_none());
    assert_eq!(pool.active_blocks(), 5 + 1);
}

#[test]
fn published_state_block_goes_at_the_end_of_the_pass_that_writes_past_it() {
    // A write past a state that the request published inside its running
    // block keeps that block and takes a fresh one (v0.31 single:1903-1936,
    // v0.32 single:1867-1900, v0.33 single:1884-1917). The kept block holds
    // the key and goes at the end of the pass, as vLLM frees the copy block
    // at the end of `schedule()` (v0.31 sched:1434-1442, v0.32
    // sched:1474-1482, v0.33 sched:1698-1706), also with a step in flight.
    // The block of the crossing at 32 waits for processed > 32.
    let groups = [attention("full", 32, None), state("s", 32, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 5);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        // (step, c, t, processed with two batches, previous blocks, active
        // in the pass, active after the pass): the two attention blocks and,
        // in the state group, the running block, the block of index 0 until
        // processed > 32 and, in the pass of step 3 only, the published
        // block. So the state group holds 1, 2, 2 and 1 blocks after the
        // four passes, as vLLM does with async scheduling.
        for (step, c, t, processed, previous, active, after) in [
            (1, 0, 32, 0, 0, 2, 2),
            (2, 32, 48, 0, 1, 4, 4),
            (3, 48, 50, 32, 1, 5, 4),
            (4, 50, 51, 48, 0, 3, 3),
        ] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, c, t, processed));
            assert_eq!(
                previous_rules(&block_lease, 1),
                [PreviousRelease::Above(32)][..previous],
                "{version:?} step {step}"
            );
            assert_eq!(pool.active_blocks(), active, "{version:?} step {step}");
            coordinator.finalize(pool, &mut block_lease, &request, 50, t, &hit);
            // The published block is the one copy source, of step 3.
            assert_eq!(
                coordinator.release_copy_sources(pool),
                usize::from(step == 3),
                "{version:?} step {step}"
            );
            assert_eq!(pool.active_blocks(), after, "{version:?} step {step}");
            // The published state is a hit from the step after its copy.
            assert_eq!(
                pool.inactive_cached_blocks(),
                [0, 0, 1, 2][step - 1],
                "{version:?} step {step}"
            );
        }
        // The state at 32 and the published state at 48.
        assert!(hit_exists(pool, 1, 4, 32) && hit_exists(pool, 1, 4, 48));
        // The block with the key at 48 left at the end of step 3, and the
        // block of index 0 with the key at 32 in step 4, whose processed
        // count 48 lies above 32 (v0.33 single:682-720, :1608-1627). So the
        // key at 48 is the older one in the free queue, and an eviction takes
        // it first.
        // 5 blocks, 3 active and 2 cached: a reservation of 1 block evicts
        // the oldest cached block.
        let reservation = pool.reserve(&[], 1).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 1, 4, 48), "{version:?}");
        assert!(hit_exists(pool, 1, 4, 32), "{version:?}");
    }
}

#[test]
fn keyed_state_blocks_of_a_released_request_enter_the_lru_in_vllms_order() {
    // Two blocks of the state group hold keys after step 3: R0, the block of
    // index 0 with the state at 32 (the replay boundary of the prompt, left
    // by the crossing of step 2), and R1, the block of index 1 with the
    // state at 48 that the request published inside it. Step 3 wrote past
    // that state, so R1 left at the end of its pass, as vLLM frees the copy
    // block with the key at 48 at the end of the schedule of step 3 (v0.31
    // sched:1434-1442, v0.32 sched:1474-1482, v0.33 sched:1698-1706). A
    // preemption then releases the request through `release_request`, and
    // `free` frees each group tail first (v0.31 and v0.32 single:581-589,
    // v0.33 single:591-599): the state block of index 0 with the key at 32
    // comes last. So the state key at 48 is older than every block of the
    // released request in the free queue.
    let groups = [attention("full", 32, None), state("s", 32, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 5);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        // (step, c, t, processed with two batches)
        for (step, c, t, processed) in [(1, 0, 32, 0), (2, 32, 48, 0), (3, 48, 50, 32)] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, c, t, processed));
            coordinator.finalize(pool, &mut block_lease, &request, 50, t, &hit);
            // R1 is the one copy source, of step 3.
            assert_eq!(
                coordinator.release_copy_sources(pool),
                usize::from(step == 3),
                "{version:?}"
            );
        }
        assert_eq!(
            previous_rules(&block_lease, 1),
            [PreviousRelease::Above(32)],
            "{version:?}"
        );
        coordinator.release_request(pool, &mut block_lease);
        assert_eq!(pool.active_blocks(), 0, "{version:?}");
        // The attention blocks with the key at 32 and the partial-tail entry
        // at 48, R1 and R0. Only the running state block has no key.
        assert_eq!(pool.inactive_cached_blocks(), 4, "{version:?}");
        // The free queue holds R1, the attention block of
        // the tail, the attention block of 32, then R0. One free block and
        // the oldest cached block: R1 goes alone.
        let reservation = pool.reserve(&[], 2).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 1, 4, 48), "{version:?}");
        assert!(hit_exists(pool, 0, 4, 48) && hit_exists(pool, 0, 4, 32));
        // The two attention blocks go next, and R0 stays.
        let reservation = pool.reserve(&[], 4).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 0, 4, 32), "{version:?}");
        assert!(hit_exists(pool, 1, 4, 32), "{version:?}");
    }
}

#[test]
fn crossing_past_a_published_state_keeps_the_block_by_the_crossing_rule() {
    // A resumed request (70 known tokens, prefill end `max(50, 70 - 1)`,
    // v0.33 mgr:529) whose step 3 crosses the block boundary at 64 and also
    // writes past the state at 48 that step 2 published inside the block of
    // index 1. The test calls the coordinator directly: the split stops a
    // chunk that starts inside a block at the next block boundary unless it
    // uses the internal checkpoint (v0.33 sched:644-649), so a scheduler
    // forms such a step only with a checkpoint chunk. vLLM keeps the block of
    // index 1 in the table and moves the key at 48 to a copy block (v0.33
    // single:1837-1847, :1872-1919). The block of index 1 goes at the first
    // allocation with processed > (1 + 1) * 32 = 64 (v0.33 single:1608-1657),
    // which is the crossing rule with c = 48, not the rule of a published
    // state.
    let groups = [attention("full", 32, None), state("s", 32, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(70, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        let above = |counts: &[usize]| -> Vec<PreviousRelease> {
            counts.iter().map(|&t| PreviousRelease::Above(t)).collect()
        };
        // (step, c, t, processed with two batches, previous blocks, active)
        type Row = (u64, usize, usize, usize, &'static [usize], usize);
        let rows: [Row; 5] = [
            (1, 0, 32, 0, &[], 2),
            (2, 32, 48, 0, &[32], 4),
            (3, 48, 69, 32, &[32, 64], 6),
            (4, 69, 70, 48, &[64], 5),
            (5, 70, 71, 69, &[], 4),
        ];
        for (step, c, t, processed, previous, active) in rows {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 69, c, t, processed));
            assert_eq!(
                previous_rules(&block_lease, 1),
                above(previous),
                "{version:?} step {step}"
            );
            assert_eq!(pool.active_blocks(), active, "{version:?} step {step}");
            coordinator.finalize(pool, &mut block_lease, &request, 50, t, &hit);
        }
    }
}

#[test]
fn two_keyed_state_blocks_of_one_allocation_enter_the_lru_highest_index_first() {
    // vLLM frees the range of skipped blocks from the top (v0.33
    // single:1618-1626). With the step before in flight, step 3 frees
    // `[0, 6599 // 2176 = 3)`: index 2 (the state at 6528), the null block
    // of index 1, then index 0 (the hit at 2176).
    let groups = groups_glm(false);
    let hash = 2176;
    for batches in [2, 1] {
        let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176]);
        seed_states(pool, &GLM_STATES, hash, 2176);
        let request = lease(6600, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 6600);
        assert_eq!(hit.tokens, 2176);
        let mut block_lease = coordinator.new_lease();
        coordinator.begin_step();
        assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
        // (step, c, t, processed with two batches)
        for (step, c, t, processed) in [
            (1, 2176, 6528, 2176),
            (2, 6528, 6600, 2176),
            (3, 6600, 6601, 6600),
        ] {
            let processed = if batches == 2 { processed } else { c };
            if step > 1 {
                coordinator.begin_step();
            }
            assert!(coordinator.allocate(pool, &mut block_lease, 6600, c, t, processed));
            coordinator.finalize(pool, &mut block_lease, &request, 6600, t, &hit);
        }
        // 4 attention, the ring and 4 running blocks; the previous and the
        // hit block of each state group are cached and held by no request.
        assert_eq!(pool.active_blocks(), 9, "{batches} batches");
        assert_eq!(pool.inactive_cached_blocks(), 8, "{batches} batches");
        let reservation = pool.reserve(&[], 64 - 9 - 8 + 1).unwrap();
        pool.cancel(reservation);
        if batches == 2 {
            // Both blocks left in step 3, the block of 6528 first.
            assert!(!hit_exists(pool, 2, 2176, 6528));
            assert!(hit_exists(pool, 2, 2176, 2176));
        } else {
            // Step 2 freed the hit (`cdiv(6528, 2176) - 1 = 2 > 0`), step 3
            // the block of 6528.
            assert!(!hit_exists(pool, 2, 2176, 2176));
            assert!(hit_exists(pool, 2, 2176, 6528));
        }
    }
}

#[test]
fn admission_counts_vllms_whole_prompt_blocks_without_the_producer_copy() {
    // vLLM's whole-prompt test (v0.31 mgr:519-536, v0.32 mgr:522-539,
    // v0.33 mgr:531-548) counts an attention group as `cdiv(prompt, A)`
    // less its hit blocks, plus the hit blocks that no request holds and a
    // copy block for a hit inside a block (v0.31 single:216-268, v0.32
    // single:213-265, v0.33 single:214-266). It counts a state group as
    // `cdiv(prompt, S) - len(computed)`, plus one for a hit inside a block
    // that the request consumes, capped at `1 + partial hit + checkpoint`
    // (v0.31 single:1762-1777, :1806-1811; v0.32 single:1724-1739,
    // :1768-1773; v0.33 single:1740-1755, :1785-1790). A new request is
    // not yet the producer of a tail, so the copy that its own published
    // tail takes later is not counted.
    // (attention block, hash, prompt, checkpoint, vLLM's count)
    let cold = [
        // 1 + 1 for every prompt inside one block of 64.
        (64, 4, 4, false, 2),
        (64, 4, 48, false, 2),
        (64, 4, 50, false, 2),
        (64, 4, 64, false, 2),
        (64, 64, 50, false, 2),
        (64, 4, 48, true, 2),
        (64, 4, 50, true, 2),
        (64, 4, 64, true, 2),
        // cdiv(prompt, 4) + 1.
        (4, 4, 4, false, 1 + 1),
        (4, 4, 48, false, 12 + 1),
        (4, 4, 50, false, 13 + 1),
        (4, 4, 64, false, 16 + 1),
    ];
    for (attention_block, hash, prompt, checkpoint, count) in cold {
        for version in [
            VllmRuleVersion::V0_31_0,
            VllmRuleVersion::V0_32_0,
            VllmRuleVersion::V0_33_0,
        ] {
            if checkpoint && version == VllmRuleVersion::V0_31_0 {
                continue;
            }
            let groups = [
                attention("full", attention_block, None),
                state("s", 64, true, checkpoint),
            ];
            let coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
            let kv = kv_pool(&groups, 32);
            let pool = kv.block_pool().unwrap();
            let hit = coordinator.find_longest_hit(pool, &lease(prompt, hash), prompt);
            assert_eq!(hit.tokens, 0);
            assert_eq!(
                coordinator.admission_blocks(pool, &hit, prompt),
                count,
                "{version:?}, attention {attention_block}, hash {hash}, prompt {prompt}, checkpoint={checkpoint}"
            );
        }
    }
    // A hit at 32 or at the tail 48 inside the blocks of 64: one copy
    // block per group, and the two hit blocks when no request holds
    // them: 4, else 2.
    let groups = [attention("full", 64, None), state("s", 64, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for hit_tokens in [32, 48] {
            for held in [false, true] {
                let coordinator =
                    with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
                let mut kv = kv_pool(&groups, 8);
                let pool = kv.block_pool_mut().unwrap();
                let keys = [key(0, 4, hit_tokens), key(1, 4, hit_tokens)];
                seed_inside(pool, &[0, 1], 4, 64, hit_tokens);
                let other = if held {
                    let mut reservation = pool.reserve(&keys, 0).unwrap();
                    let copies = pool.activate(&mut reservation);
                    pool.cancel(reservation);
                    copies
                } else {
                    Vec::new()
                };
                let hit = coordinator.find_longest_hit(pool, &lease(50, 4), 50);
                assert_eq!(hit.tokens, hit_tokens);
                assert_eq!(
                    coordinator.admission_blocks(pool, &hit, 50),
                    if held { 2 } else { 4 },
                    "{version:?}, hit {hit_tokens}, held={held}"
                );
                pool.release_all(other);
            }
        }
    }
}

#[test]
fn producer_copy_needs_a_block_that_admission_does_not_count() {
    // Blocks of 64, a hash block of 4, a cold prompt of 50 tokens. The
    // split stops at the tail 48, and the write 48..50 takes the copy.
    // vLLM's admission counts 2 blocks; the allocation of the copy then
    // needs a third (v0.31 single:1891-1936, v0.32 single:1855-1900,
    // v0.33 single:1872-1917). When it does not fit, vLLM preempts the
    // last running request, the request itself included, and tries again
    // (v0.31 sched:745-813, v0.32 sched:776-844, v0.33 sched:912-980).
    // vLLM counts the pending copy in the blocks that a prefill in flight
    // still needs (v0.31 sched:3003-3017, v0.32 sched:3060-3076, v0.33
    // sched:3404-3420, with single:1770-1777, v0.32 :1732-1739 and v0.33
    // :1748-1755); that count is not part of this coordinator.
    let groups = [attention("full", 64, None), state("s", 64, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for pool_blocks in [2, 3] {
            let mut coordinator =
                with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
            let mut kv = kv_pool(&groups, pool_blocks);
            let pool = kv.block_pool_mut().unwrap();
            let request = lease(50, 4);
            let hit = coordinator.find_longest_hit(pool, &request, 50);
            assert_eq!(coordinator.admission_blocks(pool, &hit, 50), 2);
            let mut block_lease = coordinator.new_lease();
            coordinator.begin_step();
            let chunk = coordinator
                .rules()
                .align_split(0, 50, 50, BUDGET, 0, BUDGET);
            assert_eq!(chunk, 48, "{version:?}");
            assert!(coordinator.allocate(pool, &mut block_lease, 50, 0, 48, 0));
            coordinator.finalize(pool, &mut block_lease, &request, 50, 48, &hit);
            assert!(hit_exists(pool, 1, 4, 48));
            coordinator.begin_step();
            let fits = coordinator.allocate(pool, &mut block_lease, 50, 48, 50, 48);
            assert_eq!(fits, pool_blocks == 3, "{version:?}, pool {pool_blocks}");
            if fits {
                assert_eq!(pool.active_blocks(), 3);
            } else {
                assert_eq!(pool.active_blocks(), 2);
            }
            coordinator.release_request(pool, &mut block_lease);
        }
    }
}

#[test]
fn producer_partial_state_keeps_separate_capacity_until_eviction() {
    // Producer-tail CoW: v0.31 single:1903-1936, v0.32 single:1867-1900
    // and v0.33 single:1884-1917 preserve the cached boundary before
    // another write.
    let groups = [attention("full", 32, None), state("s", 32, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 5);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for target in [32, 48, 50] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, before, target, before));
            coordinator.finalize(pool, &mut block_lease, &request, 50, target, &hit);
            before = target;
        }
        // Two attention blocks, the retained state at 48, and the
        // writable state at 50. The state at 32 is already inactive.
        assert_eq!(pool.active_blocks(), 4, "{version:?}");
        let state = state_lease(&block_lease, 1);
        assert!(pool.is_private(state.running.unwrap()));
        // The retained state is the copy of the pass, not a block of the
        // request, and it leaves at the end of the pass (v0.31
        // sched:1434-1442, v0.32 sched:1474-1482, v0.33 sched:1698-1706).
        assert!(state.previous.is_empty());
        assert_eq!(pool.inactive_cached_blocks(), 1);
        assert_eq!(coordinator.release_copy_sources(pool), 1);
        assert_eq!(pool.active_blocks(), 3, "{version:?}");
        assert_eq!(pool.inactive_cached_blocks(), 2);
        coordinator.release_request(pool, &mut block_lease);
        assert_eq!(pool.active_blocks(), 0);
        assert_eq!(pool.inactive_cached_blocks(), 4);
        assert!(hit_exists(pool, 1, 4, 48));
        // The free queue holds the state at 32 (freed in
        // the allocation of the last step), the preserved partial state
        // (freed at the end of that pass), then the two attention blocks of
        // the released request. One free block and the state at 32: the
        // preserved partial state is still independently resident. One block
        // more takes it, before any attention block.
        let pressure = pool.reserve(&[], 2).unwrap();
        assert!(!hit_exists(pool, 1, 4, 32));
        assert!(hit_exists(pool, 1, 4, 48));
        let last = pool.reserve(&[], 1).unwrap();
        assert!(!hit_exists(pool, 1, 4, 48));
        assert!(hit_exists(pool, 0, 4, 32) && hit_exists(pool, 0, 4, 48));
        pool.cancel(last);
        pool.cancel(pressure);
    }
}

#[test]
fn producer_copy_is_taken_only_with_hits_inside_a_block() {
    // With hits inside a block on, a chunk that ends at the prompt's
    // partial tail inside a state block publishes the running block, and
    // the next allocation takes a fresh block and keeps the published
    // copy (v0.31 single:2094-2147, :1891-1936; v0.32 single:2067-2112,
    // :1855-1900; v0.33 single:2085-2135, :1872-1917). With checkpoint
    // blocks, a final chunk without a valid checkpoint follows the same
    // rule (v0.32 single:2019-2066, v0.33 single:2036-2084). The
    // chunk 48..50 cannot export the checkpoint at 48.
    for (version, checkpoint) in [
        (VllmRuleVersion::V0_31_0, false),
        (VllmRuleVersion::V0_32_0, false),
        (VllmRuleVersion::V0_32_0, true),
        (VllmRuleVersion::V0_33_0, false),
        (VllmRuleVersion::V0_33_0, true),
    ] {
        let groups = [
            attention("full", 32, None),
            state("s", 32, true, checkpoint),
        ];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        assert_eq!(coordinator.rules().alignment, 4);
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for target in [32, 48, 50] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, before, target, before));
            coordinator.finalize(pool, &mut block_lease, &request, 50, target, &hit);
            before = target;
        }
        let case = format!("{version:?}, checkpoint={checkpoint}");
        assert!(hit_exists(pool, 1, 4, 48), "{case}");
        // Two attention blocks, the fresh running block and the published
        // copy, which is in use until the end of the pass.
        assert_eq!(pool.active_blocks(), 2 + 2, "{case}");
        assert!(state_lease(&block_lease, 1).previous.is_empty(), "{case}");
        assert_eq!(coordinator.release_copy_sources(pool), 1, "{case}");
        assert_eq!(pool.active_blocks(), 2 + 1, "{case}");
        coordinator.release_request(pool, &mut block_lease);
    }
    // With them off, vLLM rounds the cached length down to the scheduler
    // block before the state manager sees it (v0.31 coord:796-806,
    // :808-840; v0.32 coord:767-794, :354-377; v0.33 coord:777-804,
    // :358-381). The rounded length is a multiple of the state block, so
    // vLLM registers no tail and takes no copy (v0.31 single:2096-2097;
    // v0.32 single:2101; v0.33 single:2124). A window group of
    // block 8 has no sub-block lookup, so the alignment is 8 at a hash
    // block of 4. A budget of 4 makes a chunk end at the tail 12: the
    // prompt end on v0.31, one token before it on v0.32 and v0.33.
    for (version, prompt) in [
        (VllmRuleVersion::V0_31_0, 12),
        (VllmRuleVersion::V0_32_0, 13),
        (VllmRuleVersion::V0_33_0, 13),
    ] {
        let groups = [attention("window", 8, Some(8)), state("s", 8, true, false)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        assert_eq!(coordinator.rules().alignment, 8);
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(prompt + 1, 4);
        let hit = coordinator.find_longest_hit(pool, &request, prompt);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for target in [4, 8, 12, 13] {
            if before < prompt {
                let chunk = coordinator
                    .rules()
                    .align_split(before, prompt, prompt, 4, 0, 4);
                assert_eq!(before + chunk, target, "{version:?}");
            }
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, prompt, before, target, before));
            if target < 13 {
                coordinator.finalize(pool, &mut block_lease, &request, prompt, target, &hit);
            }
            before = target;
        }
        assert!(hit_exists(pool, 1, 4, 8), "{version:?}");
        assert!(!hit_exists(pool, 1, 4, 12), "{version:?}");
        // The write 12..13, the last prompt token on v0.32 and v0.33 or the
        // first decode token on v0.31, stays on the running state block: two
        // window blocks and one state block.
        assert_eq!(pool.active_blocks(), 2 + 1, "{version:?}");
        assert!(state_lease(&block_lease, 1).previous.is_empty());
        assert_eq!(coordinator.release_copy_sources(pool), 0, "{version:?}");
        coordinator.release_request(pool, &mut block_lease);
    }
}

#[test]
fn producer_copy_defers_hits_on_its_state_in_the_step_of_the_copy() {
    // vLLM moves the cache entry of the published state to the copy block
    // and adds its key to the keys of the step, so a waiting request hits
    // it only from the next step on (v0.31 single:1931-1934, v0.32
    // single:1895-1898, v0.33 single:1912-1915). Replay keeps the key on
    // the published block, which is the copy of the pass, and hides it for
    // the rest of the step of the copy.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = [attention("full", 32, None), state("s", 32, true, false)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for target in [32, 48] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, before, target, before));
            coordinator.finalize(pool, &mut block_lease, &request, 50, target, &hit);
            before = target;
        }
        coordinator.begin_step();
        let sibling = lease(60, 4);
        assert_eq!(
            coordinator.find_longest_hit(pool, &sibling, 60).tokens,
            48,
            "{version:?}"
        );
        assert!(coordinator.allocate(pool, &mut block_lease, 50, 48, 50, 48));
        assert_eq!(coordinator.copy_source_count(), 1, "{version:?}");
        assert!(
            coordinator.find_longest_hit(pool, &sibling, 60).tokens < 48,
            "{version:?}"
        );
        // The end of the pass frees the copy and does not show its key.
        assert_eq!(coordinator.release_copy_sources(pool), 1, "{version:?}");
        assert!(
            coordinator.find_longest_hit(pool, &sibling, 60).tokens < 48,
            "{version:?}"
        );
        coordinator.begin_step();
        assert_eq!(
            coordinator.find_longest_hit(pool, &sibling, 60).tokens,
            48,
            "{version:?}"
        );
        coordinator.release_request(pool, &mut block_lease);
    }
}

#[test]
fn the_plan_of_an_allocation_counts_the_producer_copy() {
    // The plan reads from the lease that the running state block is
    // published inside the block, so the planned fresh blocks equal what
    // the allocation takes: one block for the copy (v0.31
    // single:1891-1936, v0.32 single:1855-1900).
    for version in [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0] {
        let groups = [attention("full", 32, None), state("s", 32, true, false)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for target in [32, 48] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, before, target, before));
            coordinator.finalize(pool, &mut block_lease, &request, 50, target, &hit);
            before = target;
        }
        assert!(
            state_lease(&block_lease, 1).published.is_some(),
            "{version:?}"
        );
        let planned = GroupedPrefixCoordinator::plan_allocation(
            &coordinator.rules,
            &mut block_lease.clone(),
            50,
            48,
            50,
            48,
            &mut Vec::new(),
            &mut Vec::new(),
        );
        assert_eq!(planned, 1, "{version:?}");
        let active = pool.active_blocks();
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 50, 48, 50, 48));
        // The previous block of the boundary at 32 goes first.
        assert_eq!(pool.active_blocks(), active - 1 + planned, "{version:?}");
        assert!(
            state_lease(&block_lease, 1).published.is_none(),
            "{version:?}"
        );
        coordinator.release_request(pool, &mut block_lease);
    }
}

/// For each prompt in order on one pool: the hit, the junction, the
/// admission count, then each chunk end with the active blocks after its
/// allocation, the active blocks after one decode token, and the
/// inactive cached blocks after the release.
fn glm_trace(version: VllmRuleVersion, checkpoint: bool) -> Vec<Vec<usize>> {
    let mut coordinator = with_limits(rules_glm(version, checkpoint));
    let groups = groups_glm(checkpoint);
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let block = GLM_BLOCK as usize;
    let mut trace = Vec::new();
    // Cold, warm on the first prompt's prefix, and a block boundary.
    for prompt in [9000, 12_000, 4 * block] {
        let request = lease(prompt, block);
        let hit = coordinator.find_longest_hit(pool, &request, prompt);
        let junction = hit.tokens + hit.junction_tokens;
        let mut row = vec![
            hit.tokens,
            hit.junction_tokens,
            coordinator.admission_blocks(pool, &hit, prompt),
        ];
        let mut block_lease = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
        let mut computed = hit.tokens;
        while computed <= prompt {
            coordinator.begin_step();
            let target = if computed < prompt {
                computed
                    + coordinator
                        .rules()
                        .align_split(computed, prompt, prompt, BUDGET, junction, BUDGET)
            } else {
                prompt + 1
            };
            assert!(coordinator.allocate(
                pool,
                &mut block_lease,
                prompt,
                computed,
                target,
                computed
            ));
            row.extend([target, pool.active_blocks()]);
            if target <= prompt {
                coordinator.finalize(pool, &mut block_lease, &request, prompt, target, &hit);
            }
            computed = target;
        }
        coordinator.release_request(pool, &mut block_lease);
        row.push(pool.inactive_cached_blocks());
        trace.push(row);
    }
    trace
}

#[test]
fn glm_groups_at_the_hash_block_keep_their_counts() {
    // The six GLM-5.3-Flash groups with the hash block equal to the
    // block. A state is published only at a block boundary, where the
    // next write already takes a new block, so the producer copy never
    // adds one, and no group has a window or a tail inside a block. The
    // rows give the hit, the junction, the admission count, each chunk
    // end with the active blocks, one decode token, and the inactive
    // cached blocks after the release, for a cold prompt, a warm prompt
    // and a prompt that ends on a block boundary.
    let no_checkpoint = vec![
        vec![0, 0, 10, 6528, 8, 8704, 13, 9000, 14, 9001, 10, 8],
        vec![8704, 0, 15, 10880, 14, 12000, 15, 12001, 11, 13],
        vec![0, 6528, 9, 6528, 8, 8704, 13, 8705, 14, 21],
    ];
    let checkpoint = vec![
        vec![0, 0, 14, 6528, 8, 9000, 18, 9001, 10, 8],
        vec![8704, 0, 19, 12000, 19, 12001, 11, 13],
        vec![0, 6528, 13, 6528, 8, 8704, 13, 8705, 14, 21],
    ];
    let expected: [(VllmRuleVersion, bool, Vec<Vec<usize>>); 5] = [
        (VllmRuleVersion::V0_31_0, false, no_checkpoint.clone()),
        (VllmRuleVersion::V0_32_0, false, no_checkpoint.clone()),
        (VllmRuleVersion::V0_32_0, true, checkpoint.clone()),
        (VllmRuleVersion::V0_33_0, false, no_checkpoint),
        (VllmRuleVersion::V0_33_0, true, checkpoint),
    ];
    for (version, checkpoint, rows) in expected {
        assert_eq!(
            glm_trace(version, checkpoint),
            rows,
            "{version:?}, checkpoint={checkpoint}"
        );
    }
}

#[test]
fn finalize_publishes_states_at_the_replay_boundary_and_the_junction() {
    // single:1614-1622 with retention 0 (cache:160-166): the replay
    // boundary `(9000 - 1) // 2176 * 2176 == 8704` and the junction.
    let groups = groups_glm(false);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        9000,
        &hit,
        2176,
    );
    assert_eq!(ends, vec![2176, 4352, 6528, 8704, 9000]);
    coordinator.release_request(pool, &mut block_lease);
    for group in GLM_STATES {
        for end in [2176, 4352, 6528] {
            assert!(
                !hit_exists(pool, group, hash, end),
                "group {group} at {end}"
            );
        }
        assert!(hit_exists(pool, group, hash, 8704));
    }
    // G0 blocks 0 to 3 and four state blocks stay cached.
    assert_eq!(pool.inactive_cached_blocks(), 8);
    assert_eq!(pool.active_blocks(), 0);

    // A hit of 4,352 with a junction of 2,176.
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[2176, 4352, 6528]);
    seed_states(pool, &GLM_STATES, hash, 4352);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!((hit.tokens, hit.junction_tokens), (4352, 2176));
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        9000,
        &hit,
        2176,
    );
    assert_eq!(ends, vec![6528, 8704, 9000]);
    coordinator.release_request(pool, &mut block_lease);
    for group in GLM_STATES {
        assert!(hit_exists(pool, group, hash, 6528), "group {group}");
        assert!(hit_exists(pool, group, hash, 8704), "group {group}");
        assert!(!hit_exists(pool, group, hash, 2176), "group {group}");
    }
}

#[test]
fn align_split_follows_vllm_0310() {
    // sched:414-527 at v0.31.
    let r = rules_glm(VllmRuleVersion::V0_31_0, false);
    // The floor of 8,192 to 2,176 is 6,528 (sched:460-468).
    assert_eq!(split(&r, 0, 9000, 9000, 8192, 0), 6528);
    // The last cacheable stop `9000 - 9000 % 2176 == 8704` (sched:444).
    assert_eq!(split(&r, 6528, 9000, 9000, 8192, 0), 2176);
    assert_eq!(split(&r, 8704, 9000, 9000, 8192, 0), 296);
    // A prompt of exactly 8,704 tokens has no stop at 6,528, because its
    // last cacheable stop `8704 - 8704 % 2176` is the prompt itself
    // (sched:444).
    assert_eq!(split(&r, 6528, 8704, 8704, 8192, 0), 2176);
    // With hash 4 the tail 9,000 equals the prompt and adds no stop
    // (sched:482-487, :512-514). With prompt 9,002 the tail is 9,000.
    let fine = rules_glm_hash(VllmRuleVersion::V0_31_0, false, 4);
    assert_eq!(split(&fine, 8704, 9000, 9000, 8192, 0), 296);
    assert_eq!(split(&fine, 8704, 9002, 9002, 8192, 0), 296);
    // The junction stop (sched:488-519).
    assert_eq!(split(&r, 4352, 9000, 9000, 8192, 6528), 2176);
    // Past the prefill the split returns the request (sched:434-436).
    assert_eq!(split(&r, 9000, 9000, 9001, 1, 0), 1);
}

#[test]
fn align_split_follows_vllm_0320() {
    // v0.32 sched:443-556.
    let r = rules_glm(VllmRuleVersion::V0_32_0, false);
    assert_eq!(split(&r, 0, 9000, 9000, 8192, 0), 6528);
    assert_eq!(split(&r, 6528, 9000, 9000, 2472, 0), 2176);
    // The last stop of 8,704 tokens is `(8704 - 1) // 2176 * 2176 == 6528`
    // (v0.32 sched:476-480).
    assert_eq!(split(&r, 4352, 8704, 8704, 4352, 0), 2176);
    let v31 = rules_glm(VllmRuleVersion::V0_31_0, false);
    assert_eq!(split(&v31, 4352, 8704, 8704, 4352, 0), 4352);
    // Checkpoint mode (v0.32 sched:482-501, iface:1125-1146). For
    // (6528, 9000, pos 8704): 6528 % 2176 == 0; cdiv(9000, 2176) - 2 == 3
    // > 6527 / 2176 == 2; 6528 + 2176 <= 8704; 6528 < 8704 < 9000;
    // 2176 % 16 == 0.
    let ckpt = rules_glm(VllmRuleVersion::V0_32_0, true);
    assert_eq!(ckpt.checkpoint_position(9000), 8704);
    assert!(ckpt.checkpoint_valid(6528, 9000, 8704));
    assert_eq!(split(&ckpt, 6528, 9000, 9000, 2472, 0), 2472);
    // For (4352, 8704, pos 6528): 4352 % 2176 == 0; cdiv(8704, 2176) - 2
    // == 2 > 4351 / 2176 == 1; 4352 + 2176 <= 6528; 4352 < 6528 < 8704;
    // 2176 % 16 == 0.
    assert_eq!(ckpt.checkpoint_position(8704), 6528);
    assert!(ckpt.checkpoint_valid(4352, 8704, 6528));
    assert_eq!(split(&ckpt, 4352, 8704, 8704, 4352, 0), 4352);
    // Without a checkpoint rule the check is false.
    assert!(!r.checkpoint_valid(6528, 9000, 8704));
}

#[test]
fn align_split_with_a_checkpoint_block_ports_vllm_0320() {
    // Ported from v0.32 split_test:127-180, which stubs `block_size` to
    // 1,600 and `hash_block_size` to 16 (split_test:139-153). The engine
    // shape with the same numbers: one full-attention rule and one state
    // rule at block 1,600, hash 16, checkpoint 1 at alignment 16. The
    // state block is above the hash block and the other rule is full
    // attention, so fine-grained hits are on and A is 16.
    let groups = [attention("full", 1600, None), state("s", 1600, true, true)];
    let r = GroupedPrefixRules::new(&groups, 16, VllmRuleVersion::V0_32_0).unwrap();
    assert_eq!(r.alignment, 16);
    assert_eq!(r.cacheable_block, 1600);
    // The engine's checkpoint position is `(2002 - 1) / 16 * 16 == 2000`.
    // The vLLM test stubs partial hits off and gets 1,600. Both positions
    // pass the five conditions (0 % 16 == 0; cdiv(2002, 1600) - 2 == 0 >
    // -1; 0 + 16 <= pos; 0 < pos < 2002; pos % 16 == 0), so both give
    // 2,002.
    assert_eq!(r.checkpoint_position(2002), 2000);
    assert!(r.checkpoint_valid(0, 2002, 2000));
    assert!(r.checkpoint_valid(0, 2002, 1600));
    assert_eq!(r.align_split(0, 2002, 2002, 2002, 0, 16384), 2002);
    // The end 2,000 is below the prompt, so the floor to 1,600 applies
    // and no checkpoint is taken.
    assert_eq!(r.align_split(0, 3602, 3602, 2000, 0, 16384), 1600);
    let groups = [attention("full", 1600, None), state("s", 1600, true, false)];
    let v31 = GroupedPrefixRules::new(&groups, 16, VllmRuleVersion::V0_31_0).unwrap();
    assert_eq!(v31.align_split(0, 2002, 2002, 2002, 0, 16384), 1600);
    assert_eq!(v31.align_split(0, 3602, 3602, 2000, 0, 16384), 1600);
}

#[test]
fn align_split_needs_a_mamba_state_group() {
    // vLLM splits only with Mamba layers in align mode (v0.31 `sched:354-356`,
    // `:695-697`, v0.32 `sched:377-379`, `:724-726`). Without a state rule
    // every chunk is the desired count, on every row.
    let full = [attention("full", 64, None)];
    let windowed = [attention("full", 64, None), attention("swa", 64, Some(128))];
    // A tail ring, such as the GLM `KpoolTailSpec`, is not a MambaSpec.
    let ring = [attention("full", 64, None), state("ring", 4, false, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for groups in [&full[..], &windowed[..], &ring[..]] {
            let r = GroupedPrefixRules::new(groups, 64, version).unwrap();
            // The last block boundary of the prompt is no stop.
            assert_eq!(split(&r, 0, 300, 300, 300, 0), 300, "{version:?}");
            // A chunk that ends below the prompt keeps its length.
            assert_eq!(split(&r, 0, 1000, 1000, 500, 0), 500, "{version:?}");
            // A start inside a block runs past the next boundary.
            assert_eq!(split(&r, 100, 1000, 1000, 500, 0), 500, "{version:?}");
            // The junction is no stop.
            assert_eq!(split(&r, 64, 1000, 1000, 800, 640), 800, "{version:?}");
        }
        // A state group keeps the split: the floor of 8,192 to 2,176.
        assert_eq!(
            split(&rules_glm(version, false), 0, 9000, 9000, 8192, 0),
            6528,
            "{version:?}"
        );
    }
}

#[test]
fn partial_tail_moves_one_unit_down_on_vllm_0320() {
    // At hash 4 the 0.32.0 tail, `(prompt - 1) / hash * hash`, is 8,996,
    // between the last stop 8,704 and the prompt (v0.32 sched:517-525,
    // single:860-868, single:2069-2073). The 0.31.0 tail is 9,000.
    let groups = groups_glm(false);
    let hash = 4;
    for (version, expected_ends, expected_hit) in [
        (VllmRuleVersion::V0_32_0, vec![6528, 8704, 8996, 9000], 8996),
        (VllmRuleVersion::V0_33_0, vec![6528, 8704, 8996, 9000], 8996),
        (VllmRuleVersion::V0_31_0, vec![6528, 8704, 9000], 8704),
    ] {
        let mut coordinator = with_limits(rules_glm_hash(version, false, hash));
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(9000, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        let mut block_lease = coordinator.new_lease();
        let ends = prefill(
            &mut coordinator,
            pool,
            &mut block_lease,
            &request,
            9000,
            &hit,
            BUDGET,
        );
        assert_eq!(ends, expected_ends, "{version:?}");
        if version != VllmRuleVersion::V0_31_0 {
            for group in GLM_STATES {
                assert!(hit_exists(pool, group, hash, 8704));
                assert!(hit_exists(pool, group, hash, 8996));
            }
        }
        let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(resend.tokens, expected_hit, "{version:?}");
        coordinator.release_request(pool, &mut block_lease);
        let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(resend.tokens, expected_hit, "{version:?}");
    }
}

#[test]
fn checkpoint_block_is_taken_in_the_final_chunk_and_published_at_the_prompt_checkpoint() {
    // v0.32 single:1643-1674, :1745-1773 (one more block in the chunk
    // that reaches the prompt), :2017-2066 (published at the prompt
    // checkpoint under retention 0).
    let groups = groups_glm(true);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_32_0, true));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!(hit.tokens, 0);
    // 5 for G0, 1 for the ring, 2 for each KDA group.
    assert_eq!(coordinator.admission_blocks(pool, &hit, 9000), 14);
    let v31 = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    assert_eq!(v31.admission_blocks(pool, &hit, 9000), 10);

    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert_eq!(split(coordinator.rules(), 0, 9000, 9000, BUDGET, 0), 6528);
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 0, 6528, 0));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.running.is_some());
        assert!(state.previous.is_empty() && state.checkpoint.is_none() && state.hit.is_none());
    }
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 6528, &hit);
    coordinator.begin_step();
    assert_eq!(
        split(coordinator.rules(), 6528, 9000, 9000, BUDGET, 0),
        2472
    );
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 6528, 9000, 6528));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.running.is_some() && !state.previous.is_empty());
        assert_eq!(state.checkpoint.map(|(_, position)| position), Some(8704));
    }
    // G0 5, ring 1, 4 x (running + previous + checkpoint).
    assert_eq!(pool.active_blocks(), 5 + 1 + 12);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 9000, &hit);
    for group in GLM_STATES {
        assert!(hit_exists(pool, group, hash, 8704), "group {group}");
        let state = state_lease(&block_lease, usize::from(group));
        let (checkpoint, _) = state.checkpoint.unwrap();
        assert!(!pool.is_private(checkpoint));
        assert!(pool.is_private(state.running.unwrap()));
    }
    coordinator.begin_step();
    let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
    assert_eq!(resend.tokens, 8704);
    coordinator.release_request(pool, &mut block_lease);
    // Four G0 blocks and the four checkpoint copies stay in the LRU.
    assert_eq!(pool.inactive_cached_blocks(), 8);
    assert_eq!(pool.active_blocks(), 0);
    let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
    assert_eq!(resend.tokens, 8704);

    // At hash 4 the checkpoint of a 4,500-token prompt lies inside a block,
    // at (4,500 - 1) / 4 * 4 = 4,496, and a resend hits it there.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_33_0, true, 4));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(4500, 4);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 4500, 0, 4500, 0));
    coordinator.finalize(pool, &mut block_lease, &request, 4500, 4500, &hit);
    coordinator.begin_step();
    let resend = coordinator.find_longest_hit(pool, &request, 4500);
    assert_eq!(resend.tokens, 4496);
}

#[test]
fn checkpoint_block_goes_at_the_next_step_and_stays_cached_until_eviction() {
    // A Mamba manager skips every token before the last computed one (v0.32
    // single:1954-1959). So the request's next allocation first frees every
    // block below its last block, the checkpoint block included (v0.32
    // kv_cache_manager.py:546-559, single:672-710, :1592-1641). A freed
    // block keeps its key in the LRU (v0.32 block_pool.py:774-803).
    let groups = groups_glm(true);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_32_0, true));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        9000,
        &hit,
        BUDGET,
    );
    assert_eq!(ends, [6528, 9000]);
    // G0 5, ring 1, 4 x (running + previous + checkpoint).
    assert_eq!(pool.active_blocks(), 5 + 1 + 12);
    // The next allocation releases blocks, so no step is quiet yet.
    assert_eq!(block_lease.quiet_until, 0);
    // The prefill ran steps 1 and 2. Each decode step holds G0 5, the ring
    // and the four running blocks only, and a resend still hits the
    // checkpoint keys.
    for (step, target) in [(3, 9001), (4, 9002)] {
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 9000, target - 1, target, target - 1));
        assert_eq!(pool.active_blocks(), 5 + 1 + 4, "step {step}");
        for group in GLM_STATES {
            let state = state_lease(&block_lease, usize::from(group));
            assert!(state.checkpoint.is_none() && state.previous.is_empty());
            let cached = pool.key_hit(group, hash_at(hash, 8704));
            assert!(cached.is_some_and(|hit| !hit.is_active), "group {group}");
        }
        coordinator.finalize(pool, &mut block_lease, &request, 9000, target, &hit);
        // Quiet up to the end of the fifth G0 block, which the request holds.
        assert_eq!(block_lease.quiet_until, 10880, "step {step}");
        let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(resend.tokens, 8704, "step {step}");
    }
    // The checkpoint keys go only when a new block evicts them, oldest first
    // in the order the groups freed them (v0.32 coord:447-450).
    assert_eq!(pool.inactive_cached_blocks(), 4);
    let free = pool.capacity_blocks() - pool.active_blocks() - 4;
    let mut reservation = pool.reserve(&[], free + 1).unwrap();
    for _ in 0..=free {
        pool.take_private(&mut reservation);
    }
    pool.cancel(reservation);
    assert!(!hit_exists(pool, 2, hash, 8704));
    for group in [3, 4, 5] {
        assert!(hit_exists(pool, group, hash, 8704), "group {group}");
    }
}

#[test]
fn checkpoint_block_publishes_the_prompt_checkpoint_under_retention_zero() {
    // Ported from v0.32 partial_test:775-807: the prompt-end checkpoint
    // `(prompt - 1) / 32 * 32` stays published under retention 0.
    let groups = [attention("full", 32, None), state("s", 32, true, true)];
    let hash = 32;
    for (prompt, expected) in [(120, 96), (104, 96), (96, 64)] {
        let mut coordinator =
            with_limits(GroupedPrefixRules::new(&groups, hash, VllmRuleVersion::V0_32_0).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(prompt, hash);
        let hit = coordinator.find_longest_hit(pool, &request, prompt);
        let mut block_lease = coordinator.new_lease();
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, prompt, 0, prompt, 0));
        assert_eq!(
            state_lease(&block_lease, 1).checkpoint.map(|(_, p)| p),
            Some(expected)
        );
        coordinator.finalize(pool, &mut block_lease, &request, prompt, prompt, &hit);
        assert!(hit_exists(pool, 1, hash, expected), "prompt {prompt}");
        coordinator.release_request(pool, &mut block_lease);
        coordinator.begin_step();
        let resend = coordinator.find_longest_hit(pool, &lease(prompt, hash), prompt);
        assert_eq!(resend.tokens, expected, "prompt {prompt}");
    }
}

#[test]
fn checkpoint_beyond_the_prompt_checkpoint_is_not_published_under_retention_zero() {
    // A request with a prompt of 32 tokens resumes with 40 known tokens. Its
    // prefill ends at `max(32, 40 - 1) = 39` (v0.32 mgr:518-520). From a hit
    // at 16, the chunk 16..40 reaches it and reserves the checkpoint at 32
    // (v0.32 single:1643-1674, :1745-1773). Retention 0 keeps a checkpoint
    // only at the prompt checkpoint `(32 - 1) / 16 * 16 = 16` or at the
    // shared-prefix junction, and drops this one (v0.32 single:2025-2056,
    // cache.py:159-165). So a later prompt hits 16, not 32.
    let groups = [attention("full", 16, None), state("s", 16, true, true)];
    let hash = 16;
    let mut coordinator =
        with_limits(GroupedPrefixRules::new(&groups, hash, VllmRuleVersion::V0_32_0).unwrap());
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[16]);
    seed_states(pool, &[1], hash, 16);
    let request = lease(32, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 40);
    assert_eq!(hit.tokens, 16);
    assert_eq!(split(coordinator.rules(), 16, 32, 40, BUDGET, 0), 24);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
    assert!(coordinator.allocate(pool, &mut block_lease, 39, 16, 40, 16));
    assert_eq!(
        state_lease(&block_lease, 1).checkpoint.map(|(_, p)| p),
        Some(32)
    );
    coordinator.finalize(pool, &mut block_lease, &request, 32, 40, &hit);
    assert!(!hit_exists(pool, 1, hash, 32));
    coordinator.release_request(pool, &mut block_lease);
    coordinator.begin_step();
    let later = coordinator.find_longest_hit(pool, &lease(48, hash), 48);
    assert_eq!(later.tokens, 16);
}

#[test]
fn state_key_published_in_a_step_is_not_a_hit_in_that_step() {
    // single:1726-1734: a state block cached in this step is not a hit
    // until the next step. The block pool stores no step, so the coordinator
    // keeps the set.
    let groups = groups_glm(false);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(2500, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 2500);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 2500, 0, 2176, 0));
    coordinator.finalize(pool, &mut block_lease, &request, 2500, 2176, &hit);
    for group in GLM_STATES {
        assert!(hit_exists(pool, group, hash, 2176));
    }
    let same_step = coordinator.find_longest_hit(pool, &lease(2500, hash), 2500);
    assert_eq!(same_step.tokens, 0);
    coordinator.begin_step();
    let next_step = coordinator.find_longest_hit(pool, &lease(2500, hash), 2500);
    assert_eq!(next_step.tokens, 2176);
}

#[test]
fn allocate_fails_atomically() {
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    // Room for the first chunk (G0 1, ring 1, 4 running) but not the
    // second (G0 1 more, 4 new running blocks while 4 previous stay).
    let mut kv = kv_pool(&groups, 9);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, 2176);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 0, 2176, 0));
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 2176, &hit);
    assert_eq!(pool.active_blocks(), 6);
    coordinator.begin_step();
    let before = format!("{block_lease:?}");
    assert!(!coordinator.allocate(pool, &mut block_lease, 9000, 2176, 4352, 2176));
    assert_eq!(pool.active_blocks(), 6);
    assert_eq!(format!("{block_lease:?}"), before);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
}

#[test]
fn release_frees_groups_in_profile_order_each_tail_first() {
    // coord:387-395 frees the managers in group order, and each manager frees
    // its blocks tail first (single:581-589). Cached blocks join the LRU tail
    // in that order (pool:777-808). So the cached block of a state group that
    // the profile lists first is evicted before the attention blocks.
    let groups = [state("s", 64, true, false), attention("full", 64, None)];
    let hash = 64;
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 6);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(200, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 200);
        let mut block_lease = coordinator.new_lease();
        let ends = prefill(
            &mut coordinator,
            pool,
            &mut block_lease,
            &request,
            200,
            &hit,
            256,
        );
        assert_eq!(ends, vec![192, 200], "{version:?}");
        coordinator.release_request(pool, &mut block_lease);
        // The state at the replay boundary 192 and the attention blocks that
        // end at 64, 128 and 192 stay cached. Two blocks are free.
        assert_eq!(pool.inactive_cached_blocks(), 4, "{version:?}");
        assert_eq!(pool.active_blocks(), 0, "{version:?}");
        // Three fresh blocks evict one cached block: the state block.
        let reservation = pool.reserve(&[], 3).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 0, hash, 192), "{version:?}");
        for end in [64, 128, 192] {
            assert!(hit_exists(pool, 1, hash, end), "{version:?} at {end}");
        }
        // Four fresh blocks evict one more: the attention tail block.
        let reservation = pool.reserve(&[], 4).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 1, hash, 192), "{version:?}");
        for end in [64, 128] {
            assert!(hit_exists(pool, 1, hash, end), "{version:?} at {end}");
        }
    }
}

#[test]
fn checkpoint_rule_needs_state_groups_at_the_smallest_cacheable_block() {
    // vLLM 0.32.0 tests a checkpoint against `cache_config.block_size`, the
    // smallest prefix-cacheable block, in the scheduler (v0.32 sched:469,
    // :496) and against the Mamba block in the manager (v0.32 single:1652,
    // :1662). The rule table holds one block for both.
    let small_attention = [attention("full", 16, None), state("s", 64, true, true)];
    let err = GroupedPrefixRules::new(&small_attention, 16, VllmRuleVersion::V0_32_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.32.0 require every state group with prefill checkpoint blocks to have the smallest block_size_tokens of the prefix_cacheable groups"
    );
    // Without checkpoint blocks the same shape is modeled on every row.
    let no_checkpoint = [attention("full", 16, None), state("s", 64, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        assert!(GroupedPrefixRules::new(&no_checkpoint, 16, version).is_ok());
    }
    // A larger attention block keeps the state block the smallest. The
    // checkpoint position then moves in units of the scheduler block, the LCM
    // 128, in the manager (v0.32 coord:699-706) and in the scheduler (v0.32
    // sched:473-475, :209): `(300 - 1) / 128 * 128 == 256`.
    let large_attention = [attention("full", 128, None), state("s", 64, true, true)];
    let r = GroupedPrefixRules::new(&large_attention, 64, VllmRuleVersion::V0_32_0).unwrap();
    assert_eq!((r.alignment, r.cacheable_block), (128, 64));
    assert_eq!(r.checkpoint_position(300), 256);
}

#[test]
fn retired_window_blocks_are_released_tail_first() {
    // v0.31 single:660-685 and v0.32 single:645-670 free a retired window
    // range backward, and free_blocks appends its cached blocks to the free
    // queue in that order (v0.31 pool:777-806, v0.32 pool:774-803). So the
    // tail block of the range is evicted first.
    let groups = [attention("window", 4, Some(8))];
    let hash = 4;
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        // The run that a finished request of 100 tokens keeps before its
        // replay boundary 96: the blocks that end at 92 and 96.
        seed_many(pool, 0, hash, &[92, 96]);
        let request = lease(108, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 108);
        assert_eq!(hit.tokens, 96, "{version:?}");
        let mut block_lease = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
        assert!(coordinator.allocate(pool, &mut block_lease, 108, 96, 104, 96));
        coordinator.finalize(pool, &mut block_lease, &request, 108, 104, &hit);
        // The next allocation retires both hit blocks: (104 - 7) / 4 = 24.
        assert!(coordinator.allocate(pool, &mut block_lease, 108, 104, 108, 104));
        assert_eq!(pool.inactive_cached_blocks(), 2, "{version:?}");
        assert_eq!(pool.active_blocks(), 3, "{version:?}");
        // Three blocks are free, so four fresh blocks evict one cached block:
        // the tail block of the retired range, which ends at 96.
        let reservation = pool.reserve(&[], 4).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 0, hash, 96), "{version:?}");
        assert!(hit_exists(pool, 0, hash, 92), "{version:?}");
    }
}

#[test]
fn retired_window_blocks_wait_for_the_step_in_flight() {
    // vLLM frees the blocks below the window on the processed count,
    // computed minus the tokens in flight (v0.31 mgr:549-556, v0.32
    // mgr:552-559, v0.33 mgr:561-568). The window skips
    // `processed - window + 1` tokens (v0.33 single:1148-1183, where
    // `extra_retained_tokens` is 0 without speculative decoding) and frees
    // the blocks `[0, skipped // B)` (v0.33 single:682-720). Window 4, block
    // 4: the first live block is `(processed - 3) / 4`, and the live blocks
    // after a step to `t` tokens are `cdiv(t, 4) - first`. With two batches
    // the step before is in flight and processed is its computed count
    // before it; with one batch processed is the computed count.
    //
    // The last three steps decode one token each from 34, 35 and 36 tokens,
    // so that the processed count passes 35, where `(processed - 3) / 4`
    // moves from 7 to 8. Two batches, processed 33, 34 and 35: first 7, 7
    // and 8, `cdiv(t, 4)` 9, 9 and 10, live 2, 2 and 2. One batch, processed
    // 34, 35 and 36: first 7, 8 and 8, live 2, 1 and 2. These steps pin the
    // offset `window - 1`: a rule that skips `processed - window` tokens
    // holds 3 blocks in the last step with two batches, and a rule that
    // skips `processed - window + 2` holds 1 in the step before it.
    let groups = [attention("window", 4, Some(4))];
    let hash = 4;
    // (c, t, processed with two batches)
    let steps = [
        (0, 8, 0),
        (8, 16, 0),
        (16, 24, 8),
        (24, 32, 16),
        (32, 33, 24),
        (33, 34, 32),
        (34, 35, 33),
        (35, 36, 34),
        (36, 37, 35),
    ];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for (two_batches, expected) in [
            (true, [2, 4, 5, 5, 4, 2, 2, 2, 2]),
            (false, [2, 3, 3, 3, 2, 2, 2, 1, 2]),
        ] {
            let mut coordinator =
                with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
            let mut kv = kv_pool(&groups, 9);
            let pool = kv.block_pool_mut().unwrap();
            let request = lease(32, hash);
            let hit = coordinator.find_longest_hit(pool, &request, 32);
            assert_eq!(hit.tokens, 0);
            let mut block_lease = coordinator.new_lease();
            let mut live = Vec::new();
            for (step, (c, t, processed_two_batches)) in (1..).zip(steps) {
                let processed = if two_batches {
                    processed_two_batches
                } else {
                    c
                };
                coordinator.begin_step();
                assert!(
                    coordinator.allocate(pool, &mut block_lease, 32, c, t, processed),
                    "{version:?} step {step}"
                );
                live.push(block_lease.groups[0].live_copies());
                coordinator.finalize(pool, &mut block_lease, &request, 32, t, &hit);
            }
            assert_eq!(live, expected, "{version:?} two batches {two_batches}");
        }
    }
}

/// The sliding-window search before the single traversal: every candidate
/// end scans its own run again.
fn rescanning_window_run(
    max: usize,
    block: usize,
    step: usize,
    needed: usize,
    mut resident: impl FnMut(usize) -> bool,
) -> (usize, usize) {
    let mut end = max / step * step;
    while end > 0 {
        let blocks = end / block;
        let mut run = 0;
        while run < blocks && run < needed && resident(blocks - run - 1) {
            run += 1;
        }
        if run >= needed || run == blocks {
            return (end, run);
        }
        end -= step;
    }
    (0, 0)
}

#[test]
fn window_search_looks_up_each_block_at_most_once() {
    // single:1018-1044 finds a sliding-window hit in one backward traversal.
    // 4,096 candidate blocks of which the last 2,048 are resident, and a
    // window that needs 4,096 contiguous blocks: no hit.
    let (block, needed) = (16, 4096);
    let resident = |i: usize| i >= 2048;
    let mut lookups = 0;
    let found = window_run(4096 * block, block, block, needed, |i| {
        lookups += 1;
        resident(i)
    });
    let mut rescans = 0;
    let rescanned = rescanning_window_run(4096 * block, block, block, needed, |i| {
        rescans += 1;
        resident(i)
    });
    assert_eq!(found, (0, 0));
    assert_eq!(rescanned, found);
    assert_eq!(rescans, 2_102_272);
    assert_eq!(lookups, 4096);
}

#[test]
fn window_search_finds_the_hits_of_the_rescanning_search() {
    // A fixed xorshift sequence: random residency, blocks, alignments,
    // windows and lengths. Each block is looked up at most once.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    for case in 0..20_000 {
        let block = 1 + next(8) as usize;
        let step = block * (1 + next(4) as usize);
        let needed = next(7) as usize;
        let max = next(400) as usize;
        let density = next(101);
        let blocks = max / block;
        let resident: Vec<bool> = (0..blocks).map(|_| next(100) < density).collect();
        let mut lookups = vec![0; blocks];
        let found = window_run(max, block, step, needed, |i| {
            lookups[i] += 1;
            resident[i]
        });
        let expected = rescanning_window_run(max, block, step, needed, |i| resident[i]);
        assert_eq!(
            found, expected,
            "case {case}: max {max} block {block} step {step} needed {needed} {resident:?}"
        );
        assert!(lookups.iter().all(|&count| count <= 1), "case {case}");
    }
}

/// Whether a scan of every completed attention block, as finalization did
/// before it kept a cursor, would still find a block to publish.
fn rescan_finds_a_block(
    coordinator: &GroupedPrefixCoordinator,
    pool: &GroupedBlockPool,
    lease: &GroupedBlockLease,
    request: &BlockRequestLease,
    prompt: usize,
    computed_after: usize,
    hit: &GroupedHit,
) -> bool {
    use crate::engine::cache::vllm_block_pool::BlockCopyId;
    let rules = coordinator.rules();
    let junction = (hit.junction_tokens > 0).then_some(hit.tokens + hit.junction_tokens);
    let publishable = |g: usize, copy: Option<&Option<BlockCopyId>>, end: usize| {
        copy.is_some_and(|copy| copy.is_some_and(|id| pool.is_private(id)))
            && grouped_key(g, request.prefix_hash_at(end)).is_some()
    };
    rules.groups.iter().enumerate().any(|(g, rule)| {
        let GroupLease::Attention { copies, .. } = &lease.groups[g] else {
            return false;
        };
        let block = rule.block;
        match rule.kind {
            // The test has two groups and no hits inside a block, so only the
            // blocks up to the last multiple of the scheduler block get a key
            // (v0.31.0 coord:796-806, v0.32.0 coord:767-777, v0.33.0
            // coord:777-787).
            GroupRuleKind::FullAttention => (0..computed_after / rules.alignment * rules.alignment
                / block)
                .any(|i| publishable(g, copies.get(i), (i + 1) * block)),
            GroupRuleKind::SlidingWindow { window } => [
                Some(rules.replay_boundary(prompt, block)),
                junction.map(|junction| rules.block_floor(junction, block)),
            ]
            .into_iter()
            .flatten()
            .filter(|&pos| pos > 0 && pos <= computed_after)
            .any(|pos| {
                let top = pos / block;
                (top.saturating_sub((window - 1).div_ceil(block))..top)
                    .any(|i| publishable(g, copies.get(i), (i + 1) * block))
            }),
            _ => false,
        }
    })
}

#[test]
fn finalize_publishes_what_a_full_rescan_would() {
    // Two requests share a prefix on a full-attention and a sliding-window
    // group, with chunked prefills and decode steps. The two groups and no
    // hits inside a block make vLLM cache the computed tokens up to the last
    // multiple of the scheduler block (v0.31.0 coord:796-806, v0.32.0
    // coord:767-777, v0.33.0 coord:777-787). After every step a scan of the
    // full-attention blocks up to that count, and of the window blocks below
    // the published positions, finds nothing more to publish, so the cursor
    // skips only blocks that are cached or have no key. No full-attention
    // block past that count holds a key.
    let mut state: u64 = 0xD1B5_4A32_D192_ED03;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound) as usize
    };
    let mut hits = 0;
    for case in 0..400 {
        let hash = [4, 8][next(2)];
        let groups = [
            attention("full", (hash * (1 + next(2))) as u32, None),
            attention(
                "window",
                (hash * (1 + next(2))) as u32,
                Some(2 + next(40) as u32),
            ),
        ];
        let version = [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0][next(2)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 4096);
        let pool = kv.block_pool_mut().unwrap();
        let first = 1 + next(300);
        // Half of the cases resend the whole first prompt, which reaches its
        // replay boundary.
        let shared = if next(2) == 0 {
            first
        } else {
            next(first as u64 + 1)
        };
        let second: Vec<u32> = (0..shared as u32)
            .chain(10_000..10_000 + (1 + next(300)) as u32)
            .collect();
        let first: Vec<u32> = (0..first as u32).collect();
        for tokens in [first, second] {
            let mut request = BlockRequestLease::new(Uuid::nil(), Vec::new());
            request.configure_prefix_hashes(&tokens, hash);
            let prompt = tokens.len();
            let hit = coordinator.find_longest_hit(pool, &request, prompt);
            hits += usize::from(hit.tokens > 0);
            let mut block_lease = coordinator.new_lease();
            coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
            // A chunked prefill, then up to eight decode steps.
            let (budget, end) = (1 + next(64), prompt + 1 + next(8));
            let mut computed = hit.tokens;
            while computed < end {
                coordinator.begin_step();
                let target = (computed + budget).min(prompt).max(computed + 1);
                assert!(coordinator.allocate(
                    pool,
                    &mut block_lease,
                    prompt,
                    computed,
                    target,
                    computed
                ));
                coordinator.finalize(pool, &mut block_lease, &request, prompt, target, &hit);
                assert!(
                    !rescan_finds_a_block(
                        &coordinator,
                        pool,
                        &block_lease,
                        &request,
                        prompt,
                        target,
                        &hit
                    ),
                    "case {case} {groups:?} prompt {prompt} hit {} at {target}",
                    hit.tokens
                );
                let rules = coordinator.rules();
                let GroupLease::Attention { copies, .. } = &block_lease.groups[0] else {
                    unreachable!("group 0 is a full-attention group");
                };
                let cached = target / rules.alignment * rules.alignment / rules.groups[0].block;
                assert!(
                    copies
                        .iter()
                        .skip(cached)
                        .flatten()
                        .all(|id| pool.is_private(*id)),
                    "case {case} {groups:?} prompt {prompt}: a block past the cached count holds a key at {target}"
                );
                computed = target;
            }
            coordinator.release_request(pool, &mut block_lease);
        }
    }
    assert!(hits > 150, "{hits} hits");
}

#[test]
fn first_allocation_test_counts_the_blocks_that_the_first_allocation_takes() {
    // The test before a first allocation (mgr:558-576) counts the inactive
    // hit blocks and the fresh blocks of the chunk. pin_hit and the first
    // allocate hold exactly these at their peak, so the test never refuses a
    // chunk that fits and never admits one that does not.
    let layouts = [
        (
            vec![
                attention("full", 16, None),
                attention("window", 16, Some(40)),
            ],
            16,
            VllmRuleVersion::V0_31_0,
        ),
        (
            vec![attention("full", 64, None), state("s", 64, true, false)],
            16,
            VllmRuleVersion::V0_31_0,
        ),
        (
            vec![attention("full", 64, None), state("s", 64, true, false)],
            16,
            VllmRuleVersion::V0_32_0,
        ),
        (
            vec![
                attention("full", 64, None),
                state("ring", 4, false, false),
                state("s", 64, true, true),
            ],
            64,
            VllmRuleVersion::V0_32_0,
        ),
    ];
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound) as usize
    };
    let mut hits = 0;
    for (groups, hash, version) in &layouts {
        for case in 0..150 {
            // A small step budget, so that the admission cap of the window
            // binds. The allocation test leaves it out.
            let mut coordinator = GroupedPrefixCoordinator::new(
                GroupedPrefixRules::new(groups, *hash, *version).unwrap(),
                AdmissionLimits {
                    max_in_flight_tokens: 16,
                    max_model_len: usize::MAX,
                },
            );
            let mut kv = kv_pool(groups, 512);
            let pool = kv.block_pool_mut().unwrap();
            let first = 1 + next(400);
            let request = lease(first, *hash);
            let hit = coordinator.find_longest_hit(pool, &request, first);
            let mut block_lease = coordinator.new_lease();
            let budget = 64 * (1 + next(4));
            prefill(
                &mut coordinator,
                pool,
                &mut block_lease,
                &request,
                first,
                &hit,
                budget,
            );
            coordinator.release_request(pool, &mut block_lease);
            // A second request shares a prefix of the first one.
            let shared = if next(2) == 0 {
                first
            } else {
                next(first as u64 + 1)
            };
            let tokens: Vec<u32> = (0..shared as u32)
                .chain(10_000..10_000 + (1 + next(300)) as u32)
                .collect();
            let mut request = BlockRequestLease::new(Uuid::nil(), Vec::new());
            request.configure_prefix_hashes(&tokens, *hash);
            let prompt = tokens.len();
            coordinator.begin_step();
            let hit = coordinator.find_longest_hit(pool, &request, prompt);
            hits += usize::from(hit.tokens > 0);
            let target = if next(2) == 0 {
                prompt
            } else {
                hit.tokens + 1 + next((prompt - hit.tokens) as u64)
            };
            let needed = coordinator.first_allocation_blocks(pool, &hit, prompt, target);
            let before = pool.active_blocks();
            pool.reset_peak_active_blocks();
            let mut block_lease = coordinator.new_lease();
            coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
            assert!(coordinator.allocate(
                pool,
                &mut block_lease,
                prompt,
                hit.tokens,
                target,
                hit.tokens
            ));
            // The copy of a shared block is taken while the hit block is
            // still held, so the peak counts both.
            assert_eq!(
                pool.peak_active_blocks() - before,
                needed,
                "{version:?} case {case}: prompt {prompt} hit {} target {target}",
                hit.tokens
            );
            coordinator.release_request(pool, &mut block_lease);
        }
    }
    assert!(hits > 200, "{hits} hits");
}

#[test]
fn hit_stamp_changes_exactly_when_a_search_input_changes() {
    // Full attention and a cacheable state, both of 4-token blocks, at hash
    // 4, in a pool of 3 blocks.
    let groups = [attention("full", 4, None), state("s", 4, true, false)];
    let mut coordinator = with_limits(rules(&groups, 4));
    let mut kv = kv_pool(&groups, 3);
    let pool = kv.block_pool_mut().unwrap();
    let start = coordinator.hit_stamp(pool, 8);
    assert_eq!(coordinator.hit_stamp(pool, 8), start);
    // The token count.
    assert_ne!(coordinator.hit_stamp(pool, 9), start);
    // Another pool with the same resident keys.
    let other = kv_pool(&groups, 3);
    assert_ne!(coordinator.hit_stamp(other.block_pool().unwrap(), 8), start);
    // Clearing an empty published set changes nothing.
    coordinator.begin_step();
    assert_eq!(coordinator.hit_stamp(pool, 8), start);

    // A key enters the index.
    seed(pool, key(0, 4, 4));
    let one = coordinator.hit_stamp(pool, 8);
    assert_ne!(one.residency, start.residency);
    // Pinning, activating and releasing it changes no residency.
    let mut reservation = pool.reserve(&[key(0, 4, 4)], 0).unwrap();
    let copies = pool.activate(&mut reservation);
    pool.cancel(reservation);
    pool.release_all(copies);
    assert_eq!(coordinator.hit_stamp(pool, 8), one);
    // Nor does a second copy of the same key, or the loss of one of two
    // copies.
    let mut reservation = pool.reserve(&[], 1).unwrap();
    let id = pool.take_private(&mut reservation);
    pool.cancel(reservation);
    assert!(!pool.cache(id, key(0, 4, 4)));
    pool.release(id);
    assert_eq!(coordinator.hit_stamp(pool, 8), one);
    let reservation = pool.reserve(&[], 2).unwrap();
    assert!(hit_exists(pool, 0, 4, 4));
    assert_eq!(coordinator.hit_stamp(pool, 8), one);
    pool.cancel(reservation);
    // The key leaves the index with its last copy.
    let reservation = pool.reserve(&[], 3).unwrap();
    assert!(!hit_exists(pool, 0, 4, 4));
    let gone = coordinator.hit_stamp(pool, 8);
    assert_ne!(gone.residency, one.residency);
    pool.cancel(reservation);

    // A state key published in the step, then the clear of the set.
    let request = lease(9, 4);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.allocate(pool, &mut block_lease, 9, 0, 8, 0));
    let before = coordinator.hit_stamp(pool, 9);
    let no_hit = coordinator.empty_hit();
    coordinator.finalize(pool, &mut block_lease, &request, 9, 8, &no_hit);
    let published = coordinator.hit_stamp(pool, 9);
    assert!(coordinator.published_this_step.contains(&key(1, 4, 8)));
    assert_ne!(published.published_epoch, before.published_epoch);
    coordinator.begin_step();
    let cleared = coordinator.hit_stamp(pool, 9);
    assert_ne!(cleared.published_epoch, published.published_epoch);
    assert_eq!(cleared.residency, published.residency);
    coordinator.release_request(pool, &mut block_lease);
}

/// Publish in this step, through `finalize`, a second copy of a state key that
/// is already resident, and return whether a hit stored before that publish
/// would still be reused. The residency of every key stays the same, so only
/// the published set tells the waiting request that its hit changed.
fn reused_after_a_duplicate_state_publish(
    groups: &[FpmCacheGroup],
    version: VllmRuleVersion,
    hash: usize,
    seeded: usize,
    waiting_tokens: usize,
    running_prompt: usize,
    running_target: usize,
) -> bool {
    let mut coordinator = with_limits(GroupedPrefixRules::new(groups, hash, version).unwrap());
    let mut kv = kv_pool(groups, 32);
    let pool = kv.block_pool_mut().unwrap();
    // An earlier request left the keys of its first `seeded` tokens.
    let ends: Vec<usize> = (1..=seeded / hash).map(|i| i * hash).collect();
    seed_many(pool, 0, hash, &ends);
    seed(pool, key(1, hash, seeded));
    let waiting = lease(waiting_tokens, hash);
    let first = coordinator.find_longest_hit(pool, &waiting, waiting_tokens);
    assert_eq!(first.tokens, seeded);
    let stored =
        StoredGroupedHit::searched(first.clone(), coordinator.hit_stamp(pool, waiting_tokens));
    let residency = pool.residency();
    // A running request with the same tokens started cold. Its chunk up to
    // `running_target` publishes its state at `seeded` in this step.
    let request = lease(running_prompt, hash);
    let mut block_lease = coordinator.new_lease();
    let target = running_target;
    assert!(coordinator.allocate(pool, &mut block_lease, running_prompt, 0, target, 0));
    let no_hit = coordinator.empty_hit();
    coordinator.finalize(
        pool,
        &mut block_lease,
        &request,
        running_prompt,
        target,
        &no_hit,
    );
    assert!(
        coordinator
            .published_this_step
            .contains(&key(1, hash, seeded))
    );
    assert_eq!(pool.residency(), residency, "a second copy keeps every key");
    // A state key published in this step is no hit (single:1726-1734).
    let fresh = coordinator.find_longest_hit(pool, &waiting, waiting_tokens);
    assert_ne!(fresh, first);
    let reused = stored
        .reusable(&coordinator.hit_stamp(pool, waiting_tokens))
        .is_some();
    coordinator.release_request(pool, &mut block_lease);
    reused
}

#[test]
fn a_published_running_state_key_ends_the_reuse_of_a_stored_hit() {
    // vLLM 0.31.0, no checkpoint: the running state of a 9-token request is
    // kept at its replay boundary 8, which a waiting 9-token request hits.
    let groups = [attention("full", 4, None), state("s", 4, true, false)];
    assert!(!reused_after_a_duplicate_state_publish(
        &groups,
        VllmRuleVersion::V0_31_0,
        4,
        8,
        9,
        9,
        8
    ));
}

#[test]
fn a_published_checkpoint_state_key_ends_the_reuse_of_a_stored_hit() {
    // vLLM 0.32.0 with a checkpoint block: a 40-token prompt that runs in one
    // chunk exports its checkpoint at 32 (v0.32 iface:1110-1146), which a
    // waiting 41-token request hits.
    let groups = [attention("full", 16, None), state("s", 16, true, true)];
    assert!(!reused_after_a_duplicate_state_publish(
        &groups,
        VllmRuleVersion::V0_32_0,
        16,
        32,
        41,
        40,
        40
    ));
}

#[test]
fn a_moved_producer_state_key_ends_the_reuse_of_a_stored_hit() {
    // The copy of a state that the request published inside its running
    // block hides the key for the rest of the step (v0.31
    // single:1931-1934, v0.32 single:1895-1898, v0.33 single:1912-1915).
    // The residency of every key stays the same, so only the published set
    // tells the waiting request that its hit changed.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = [attention("full", 32, None), state("s", 32, true, false)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for target in [32, 48] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, before, target, before));
            coordinator.finalize(pool, &mut block_lease, &request, 50, target, &hit);
            before = target;
        }
        coordinator.begin_step();
        let waiting = lease(60, 4);
        let first = coordinator.find_longest_hit(pool, &waiting, 60);
        assert_eq!(first.tokens, 48, "{version:?}");
        let stored = StoredGroupedHit::searched(first, coordinator.hit_stamp(pool, 60));
        let residency = pool.residency();
        assert!(coordinator.allocate(pool, &mut block_lease, 50, 48, 50, 48));
        assert_eq!(pool.residency(), residency, "{version:?}");
        assert!(
            stored.reusable(&coordinator.hit_stamp(pool, 60)).is_none(),
            "{version:?}"
        );
        coordinator.release_request(pool, &mut block_lease);
    }
}

/// The hash block of the copy-source tests. A hit can then end inside a
/// 2,176-token block of the GLM groups.
const FINE: usize = 1088;

/// A request over `tokens` tokens that start at `base`. Two bases give
/// two prefixes that share no hash.
fn lease_from(base: u32, tokens: usize) -> BlockRequestLease {
    let tokens: Vec<u32> = (base..base + tokens as u32).collect();
    let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
    lease.configure_prefix_hashes(&tokens, FINE);
    lease
}

/// Prefill a request of 3,500 tokens from no hit and release it. Its
/// chunks end at 2,176, 3,264 and 3,500, so it publishes a state at
/// 3,264, inside the second block of each 2,176-token group.
fn produce(coordinator: &mut GroupedPrefixCoordinator, pool: &mut GroupedBlockPool, base: u32) {
    let request = lease_from(base, 3500);
    let hit = coordinator.find_longest_hit(pool, &request, 3500);
    assert_eq!(hit.tokens, 0);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    let ends = prefill(
        coordinator,
        pool,
        &mut block_lease,
        &request,
        3500,
        &hit,
        BUDGET,
    );
    assert_eq!(ends, [2176, 3264, 3500]);
    coordinator.release_request(pool, &mut block_lease);
}

/// Admit a request of 3,800 tokens on the prefix at `base` and allocate
/// its whole prompt. It hits 3,264 tokens, inside a block.
fn consume(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
    base: u32,
) -> (BlockRequestLease, GroupedBlockLease, GroupedHit) {
    let request = lease_from(base, 3800);
    let hit = coordinator.find_longest_hit(pool, &request, 3800);
    assert_eq!(hit.tokens, 3264);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 3800, 3264, 3800, 3264));
    (request, block_lease, hit)
}

/// `None` when the key of `tokens` in `group` is not resident, else
/// whether a request holds its block.
fn key_state(
    pool: &GroupedBlockPool,
    group: u16,
    request: &BlockRequestLease,
    tokens: usize,
) -> Option<bool> {
    pool.key_hit(group, request.prefix_hash_at(tokens).unwrap())
        .map(|hit| hit.is_active)
}

#[test]
fn copy_sources_of_a_hit_inside_a_block_stay_pinned_until_the_pass_ends() {
    // A hit inside a block copies the cached block of the full-attention
    // group (v0.31 and v0.32 single:387-397, `_apply_cow` at :454-474,
    // v0.33 :388-398, :455-475) and the hit block of each state group
    // (v0.31 single:1935-1937, v0.32 :1899-1901, v0.33 :1916-1918). Each
    // source keeps the reference that `touch` gave it (v0.31 single:307,
    // v0.32 :304, v0.33 :305) until `schedule()` frees the copy endpoints
    // at its end (v0.31 sched:1434-1442, v0.32 :1474-1482, v0.33
    // :1698-1706). Until then a later request of the same step cannot
    // take or evict it.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(rules_glm_hash(version, false, FINE));
        let mut kv = kv_pool(&groups_glm(false), 32);
        let pool = kv.block_pool_mut().unwrap();
        produce(&mut coordinator, pool, 0);
        assert_eq!(pool.active_blocks(), 0);
        assert_eq!(pool.inactive_cached_blocks(), 10);
        let (request, mut block_lease, hit) = consume(&mut coordinator, pool, 0);
        // The lease holds 2 full-attention blocks, the ring block and a
        // running block in each state group. The 5 sources are active
        // too: 7 + 5 = 12.
        assert_eq!(coordinator.copy_source_count(), 5);
        assert_eq!(pool.active_blocks(), 12);
        assert_eq!(pool.inactive_cached_blocks(), 4);
        assert_eq!(attention_copies(&block_lease, 0), 2);
        for group in GLM_STATES {
            let state = state_lease(&block_lease, usize::from(group));
            assert!(state.hit.is_none() && state.running.is_some() && state.copy.is_none());
        }
        for group in [0, 2, 3, 4, 5] {
            assert_eq!(key_state(pool, group, &request, 3264), Some(true));
        }
        // Only the blocks that are not active count as free.
        let capacity = pool.capacity_blocks();
        assert!(pool.reserve(&[], capacity - 12 + 1).is_none());
        assert_eq!(pool.active_blocks(), 12);
        assert_eq!(pool.inactive_cached_blocks(), 4);
        let reservation = pool.reserve(&[], capacity - 12).unwrap();
        pool.cancel(reservation);
        assert_eq!(pool.inactive_cached_blocks(), 0);
        assert_eq!(key_state(pool, 0, &request, 3264), Some(true));
        coordinator.finalize(pool, &mut block_lease, &request, 3800, 3800, &hit);
        assert_eq!(coordinator.copy_source_count(), 5);
        assert_eq!(pool.active_blocks(), 12);
        assert_eq!(coordinator.release_copy_sources(pool), 5);
        assert_eq!(coordinator.copy_source_count(), 0);
        assert_eq!(pool.active_blocks(), 7);
        for group in [0, 2, 3, 4, 5] {
            assert_eq!(key_state(pool, group, &request, 3264), Some(false));
        }
    }
}

#[test]
fn copy_sources_leave_in_group_order_then_in_allocation_order() {
    // vLLM collects the copies manager by manager, in group order, and
    // each manager in the order of its allocations (v0.31
    // kv_cache_manager.py:894-909). `free_blocks` puts a cached block
    // that no request holds at the tail of the free queue, in that order
    // (v0.31 block_pool.py:777-808). So the sources of two requests of
    // one step leave group by group.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 40);
    let pool = kv.block_pool_mut().unwrap();
    produce(&mut coordinator, pool, 0);
    produce(&mut coordinator, pool, 1_000_000);
    let (first, mut first_lease, first_hit) = consume(&mut coordinator, pool, 0);
    coordinator.finalize(pool, &mut first_lease, &first, 3800, 3800, &first_hit);
    let (second, mut second_lease, second_hit) = consume(&mut coordinator, pool, 1_000_000);
    coordinator.finalize(pool, &mut second_lease, &second, 3800, 3800, &second_hit);
    assert_eq!(coordinator.copy_source_count(), 10);
    assert_eq!(pool.active_blocks(), 24);
    assert_eq!(pool.inactive_cached_blocks(), 8);
    assert_eq!(coordinator.release_copy_sources(pool), 10);
    assert_eq!(pool.active_blocks(), 14);
    assert_eq!(pool.inactive_cached_blocks(), 18);
    let free = pool.capacity_blocks() - 14 - 18;
    assert_eq!(free, 8);
    // The 8 free blocks and 11 evictions: the 8 states at 2,176 of the
    // two producers, then the full-attention sources of both requests,
    // then the source of group 2 of the first request, which allocated
    // first.
    let mut reservation = pool.reserve(&[], free + 11).unwrap();
    let mut taken: Vec<_> = (0..free + 11)
        .map(|_| pool.take_private(&mut reservation))
        .collect();
    pool.cancel(reservation);
    assert_eq!(
        GLM_STATES.map(|group| key_state(pool, group, &first, 3264).is_some()),
        [false, true, true, true]
    );
    assert_eq!(
        GLM_STATES.map(|group| key_state(pool, group, &second, 3264).is_some()),
        [true, true, true, true]
    );
    // The 12th eviction takes the source of group 2 of the second request.
    let mut reservation = pool.reserve(&[], 1).unwrap();
    taken.push(pool.take_private(&mut reservation));
    pool.cancel(reservation);
    for request in [&first, &second] {
        assert_eq!(
            GLM_STATES.map(|group| key_state(pool, group, request, 3264).is_some()),
            [false, true, true, true]
        );
    }
    pool.release_all(taken);
}

#[test]
fn copy_sources_outlive_a_request_that_leaves_in_its_admission_pass() {
    // Freeing a request leaves the pending copies alone (v0.31
    // single:562-579, :1971-1988), so its sources stay referenced until
    // `schedule()` ends (v0.31 sched:1434-1442).
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 32);
    let pool = kv.block_pool_mut().unwrap();
    produce(&mut coordinator, pool, 0);
    let (request, mut block_lease, hit) = consume(&mut coordinator, pool, 0);
    coordinator.finalize(pool, &mut block_lease, &request, 3800, 3800, &hit);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(coordinator.copy_source_count(), 5);
    assert_eq!(pool.active_blocks(), 5);
    assert_eq!(key_state(pool, 0, &request, 3264), Some(true));
    let capacity = pool.capacity_blocks();
    assert!(pool.reserve(&[], capacity - 5 + 1).is_none());
    assert_eq!(coordinator.release_copy_sources(pool), 5);
    assert_eq!(pool.active_blocks(), 0);
}

#[test]
fn a_hit_on_a_block_boundary_pins_no_copy_source() {
    // A hit on a block boundary takes no copy (v0.31 single:321-327), so
    // the hit block of a state group stays in the request until its next
    // allocation (v0.31 mgr:552-556, single:1626-1645).
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 32);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease_from(0, 4400);
    let hit = coordinator.find_longest_hit(pool, &request, 4400);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        4400,
        &hit,
        BUDGET,
    );
    assert_eq!(ends, [4352, 4400]);
    coordinator.release_request(pool, &mut block_lease);
    let request = lease_from(0, 4700);
    let hit = coordinator.find_longest_hit(pool, &request, 4700);
    assert_eq!(hit.tokens, 4352);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert_eq!(pool.active_blocks(), 6);
    assert!(coordinator.allocate(pool, &mut block_lease, 4700, 4352, 4700, 4352));
    assert_eq!(coordinator.copy_source_count(), 0);
    assert_eq!(pool.active_blocks(), 12);
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.hit.is_some() && state.running.is_some() && state.copy.is_none());
    }
    coordinator.finalize(pool, &mut block_lease, &request, 4700, 4700, &hit);
    assert_eq!(coordinator.copy_source_count(), 0);
    assert_eq!(pool.active_blocks(), 12);
    assert_eq!(pool.inactive_cached_blocks(), 0);
    assert_eq!(coordinator.release_copy_sources(pool), 0);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 4700, 4700, 4701, 4700));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.hit.is_none() && state.running.is_some());
    }
    assert_eq!(pool.active_blocks(), 8);
    assert_eq!(pool.inactive_cached_blocks(), 4);
}

#[test]
fn a_failed_first_allocation_pins_no_copy_source() {
    // vLLM tests the free blocks before it touches the hit
    // (v0.31 kv_cache_manager.py:572-589), so a refused request pins
    // nothing.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 11);
    let pool = kv.block_pool_mut().unwrap();
    produce(&mut coordinator, pool, 0);
    assert_eq!(pool.active_blocks(), 0);
    assert_eq!(pool.inactive_cached_blocks(), 6);
    let request = lease_from(0, 3800);
    let hit = coordinator.find_longest_hit(pool, &request, 3800);
    assert_eq!(hit.tokens, 3264);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert_eq!(pool.active_blocks(), 6);
    // 6 fresh blocks are needed and 5 are left.
    assert!(!coordinator.allocate(pool, &mut block_lease, 3800, 3264, 3800, 3264));
    assert_eq!(coordinator.copy_source_count(), 0);
    assert_eq!(pool.active_blocks(), 6);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
    assert_eq!(pool.inactive_cached_blocks(), 6);
}

/// Prefill a request of 4,400 tokens from no hit and release it. Its chunks
/// end at 4,352 and 4,400, so it publishes a state at 4,352, on a block
/// boundary of each 2,176-token group.
fn produce_on_a_boundary(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
    base: u32,
) {
    let request = lease_from(base, 4400);
    let hit = coordinator.find_longest_hit(pool, &request, 4400);
    assert_eq!(hit.tokens, 0);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    let ends = prefill(
        coordinator,
        pool,
        &mut block_lease,
        &request,
        4400,
        &hit,
        BUDGET,
    );
    assert_eq!(ends, [4352, 4400]);
    coordinator.release_request(pool, &mut block_lease);
}

/// Evict the oldest cached block: take every free block and one more, and
/// give them back.
fn evict_one(pool: &mut GroupedBlockPool) {
    let free = pool.capacity_blocks() - pool.active_blocks() - pool.inactive_cached_blocks();
    let mut reservation = pool.reserve(&[], free + 1).unwrap();
    let taken: Vec<_> = (0..free + 1)
        .map(|_| pool.take_private(&mut reservation))
        .collect();
    pool.cancel(reservation);
    pool.release_all(taken);
}

#[test]
fn a_boundary_hit_block_leaves_after_the_copy_sources_of_its_pass() {
    // A hits 3,264 tokens, inside a block, and B hits 4,352, on a block
    // boundary. vLLM frees A's copy sources at the end of the step
    // (v0.31 sched:1434-1442) and B's hit blocks at B's next
    // `allocate_slots` (v0.31 mgr:552-556, single:1626-1645), in a later
    // step. So A's sources enter the free queue first, in either
    // admission order.
    for a_first in [true, false] {
        let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
        let mut kv = kv_pool(&groups_glm(false), 64);
        let pool = kv.block_pool_mut().unwrap();
        produce(&mut coordinator, pool, 0);
        produce_on_a_boundary(&mut coordinator, pool, 1_000_000);
        let b_request = lease_from(1_000_000, 4700);
        let admit_b = |coordinator: &mut GroupedPrefixCoordinator, pool: &mut GroupedBlockPool| {
            let hit = coordinator.find_longest_hit(pool, &b_request, 4700);
            assert_eq!(hit.tokens, 4352);
            let mut lease = coordinator.new_lease();
            assert!(coordinator.pin_hit(pool, &mut lease, &hit).is_ok());
            assert!(coordinator.allocate(pool, &mut lease, 4700, 4352, 4700, 4352));
            coordinator.finalize(pool, &mut lease, &b_request, 4700, 4700, &hit);
            (lease, hit)
        };
        let admit_a = |coordinator: &mut GroupedPrefixCoordinator, pool: &mut GroupedBlockPool| {
            let (request, mut lease, hit) = consume(coordinator, pool, 0);
            coordinator.finalize(pool, &mut lease, &request, 3800, 3800, &hit);
            (request, lease, hit)
        };
        let ((a_request, mut a_lease, a_hit), (mut b_lease, b_hit)) = if a_first {
            let a = admit_a(&mut coordinator, pool);
            (a, admit_b(&mut coordinator, pool))
        } else {
            let b = admit_b(&mut coordinator, pool);
            (admit_a(&mut coordinator, pool), b)
        };
        // A: 7 blocks and 5 sources. B: 8 blocks and 4 hit blocks.
        assert_eq!(pool.active_blocks(), 24, "{a_first}");
        assert_eq!(coordinator.release_copy_sources(pool), 5);
        assert_eq!(pool.active_blocks(), 19, "{a_first}");
        // The next pass: both requests compute one token, in running order.
        coordinator.begin_step();
        for is_a in if a_first {
            [true, false]
        } else {
            [false, true]
        } {
            if is_a {
                assert!(coordinator.allocate(pool, &mut a_lease, 3800, 3800, 3801, 3800));
                coordinator.finalize(pool, &mut a_lease, &a_request, 3800, 3801, &a_hit);
            } else {
                assert!(coordinator.allocate(pool, &mut b_lease, 4700, 4700, 4701, 4700));
                coordinator.finalize(pool, &mut b_lease, &b_request, 4700, 4701, &b_hit);
            }
        }
        assert_eq!(coordinator.release_copy_sources(pool), 0);
        assert_eq!(pool.active_blocks(), 15, "{a_first}");
        assert_eq!(pool.inactive_cached_blocks(), 13, "{a_first}");
        // The four producer states at 2,176 go first, then A's sources in
        // group order, then B's hit blocks.
        let (mut a_gone, mut b_gone) = (None, None);
        for eviction in 1..=13 {
            evict_one(pool);
            if a_gone.is_none() && key_state(pool, 2, &a_request, 3264).is_none() {
                a_gone = Some(eviction);
            }
            if b_gone.is_none() && key_state(pool, 2, &b_request, 4352).is_none() {
                b_gone = Some(eviction);
            }
        }
        assert_eq!((a_gone, b_gone), (Some(6), Some(10)), "{a_first}");
    }
}

#[test]
fn a_request_that_ends_first_frees_its_boundary_hit_block_after_its_attention_blocks() {
    // A request that ends frees its blocks in group order, each group tail
    // first (v0.31 coord:387-395, single:581-589). Its hit blocks are still
    // in its state groups, so they go after its full-attention blocks.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 32);
    let pool = kv.block_pool_mut().unwrap();
    produce_on_a_boundary(&mut coordinator, pool, 0);
    let request = lease_from(0, 4700);
    let hit = coordinator.find_longest_hit(pool, &request, 4700);
    assert_eq!(hit.tokens, 4352);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 4700, 4352, 4700, 4352));
    coordinator.finalize(pool, &mut block_lease, &request, 4700, 4700, &hit);
    assert_eq!(pool.active_blocks(), 12);
    assert_eq!(coordinator.release_copy_sources(pool), 0);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
    assert_eq!(pool.inactive_cached_blocks(), 6);
    evict_one(pool);
    assert_eq!(key_state(pool, 0, &request, 4352), None);
    assert_eq!(key_state(pool, 0, &request, 2176), Some(false));
    evict_one(pool);
    assert_eq!(key_state(pool, 0, &request, 2176), None);
    for group in GLM_STATES {
        assert_eq!(key_state(pool, group, &request, 4352), Some(false));
    }
}

#[test]
fn a_request_with_a_copy_source_and_a_boundary_hit_releases_the_source_first() {
    // The full-attention group (block 64) hits 48 tokens inside a block and
    // takes a copy. The state group (block 16) hits on its block boundary
    // at 48 and takes none. The copy source leaves at the end of the pass
    // (v0.31 sched:1434-1442), the state hit block at the request's next
    // allocation (v0.31 mgr:552-556).
    let groups = [attention("full", 64, None), state("kda", 16, true, false)];
    let hash = 8;
    let mut coordinator =
        with_limits(GroupedPrefixRules::new(&groups, hash, VllmRuleVersion::V0_31_0).unwrap());
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_inside(pool, &[0], hash, 64, 48);
    seed_states(pool, &[1], hash, 48);
    let request = lease(100, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 100);
    assert_eq!(hit.tokens, 48);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 100, 48, 100, 48));
    assert_eq!(coordinator.copy_source_count(), 1);
    assert!(state_lease(&block_lease, 1).hit.is_some());
    coordinator.finalize(pool, &mut block_lease, &request, 100, 100, &hit);
    assert!(state_lease(&block_lease, 1).hit.is_some());
    assert_eq!(coordinator.release_copy_sources(pool), 1);
    evict_one(pool);
    assert!(!hit_exists(pool, 0, hash, 48));
    assert!(hit_exists(pool, 1, hash, 48));
}

/// A request of 9,000 tokens on the prefix of `produce_on_a_boundary`: it
/// hits 4,352 tokens, on a block boundary, and its first chunk computes to
/// 8,704. It then holds 13 blocks: 4 full-attention blocks, the ring block,
/// and a hit block and a running block in each state group.
fn admit_past_a_boundary_hit(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
) -> GroupedBlockLease {
    let request = lease_from(0, 9000);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!(hit.tokens, 4352);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 4352, 8704, 4352));
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 8704, &hit);
    assert_eq!(coordinator.release_copy_sources(pool), 0);
    assert_eq!(pool.active_blocks(), 13);
    block_lease
}

#[test]
fn a_next_allocation_can_take_the_blocks_of_its_boundary_hit() {
    // vLLM frees the skipped blocks before the free-block test of
    // `allocate_slots` (v0.31 kv_cache_manager.py:552-556, :570-576). The
    // next chunk, to 9,000, crosses the block boundary at 8,704 and needs 5
    // fresh blocks: one full-attention block and a running block in each
    // state group. On a pool of 14 blocks one block is free and the 4 hit
    // blocks leave first, so the allocation succeeds.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 14);
    let pool = kv.block_pool_mut().unwrap();
    produce_on_a_boundary(&mut coordinator, pool, 0);
    let mut block_lease = admit_past_a_boundary_hit(&mut coordinator, pool);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 8704, 9000, 8704));
    // 13 - 4 hit blocks + 5 fresh blocks.
    assert_eq!(pool.active_blocks(), 14);
    for group in GLM_STATES {
        assert!(state_lease(&block_lease, usize::from(group)).hit.is_none());
    }
    coordinator.release_request(pool, &mut block_lease);
}

#[test]
fn a_refused_next_allocation_still_releases_the_boundary_hit_block() {
    // On a pool of 13 blocks no block is free, and the 4 released hit blocks
    // do not cover the 5 fresh blocks, so the allocation is refused. vLLM
    // has freed the skipped blocks before its test, so they stay released.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 13);
    let pool = kv.block_pool_mut().unwrap();
    produce_on_a_boundary(&mut coordinator, pool, 0);
    let mut block_lease = admit_past_a_boundary_hit(&mut coordinator, pool);
    coordinator.begin_step();
    assert!(!coordinator.allocate(pool, &mut block_lease, 9000, 8704, 9000, 8704));
    assert_eq!(pool.active_blocks(), 9);
    assert_eq!(pool.inactive_cached_blocks(), 4);
    for group in GLM_STATES {
        assert!(state_lease(&block_lease, usize::from(group)).hit.is_none());
    }
    // A second try changes nothing, and the request leaves cleanly.
    assert!(!coordinator.allocate(pool, &mut block_lease, 9000, 8704, 9000, 8704));
    assert_eq!(pool.active_blocks(), 9);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
}

/// A request of 13,000 tokens with checkpoint blocks: it hits 8,704 tokens,
/// on a block boundary, and computes its whole prompt in one chunk. It then
/// holds 19 blocks: 6 full-attention blocks, the ring block, and a hit, a
/// running and a checkpoint block in each state group. The checkpoint block
/// holds the state at 10,880 and is cached.
fn admit_with_a_checkpoint_block(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
    version: VllmRuleVersion,
) -> GroupedBlockLease {
    let hash = 2176;
    seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
    seed_states(pool, &GLM_STATES, hash, 8704);
    let request = lease(13000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 13000);
    assert_eq!(hit.tokens, 8704, "{version:?}");
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 13000, 8704, 13000, 8704));
    coordinator.finalize(pool, &mut block_lease, &request, 13000, 13000, &hit);
    assert_eq!(coordinator.release_copy_sources(pool), 0);
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(
            state.hit.is_some() && state.checkpoint.is_some(),
            "{version:?}"
        );
    }
    assert_eq!(pool.active_blocks(), 19, "{version:?}");
    assert_eq!(pool.inactive_cached_blocks(), 0, "{version:?}");
    block_lease
}

#[test]
fn the_checkpoint_block_leaves_before_the_boundary_hit_block_of_its_group() {
    // vLLM frees the skipped blocks manager by manager, in group order (v0.32
    // kv_cache_coordinator.py:447-450, v0.33 :451-454), and the blocks of a
    // state group in descending index order (v0.32
    // single_type_kv_cache_manager.py:1604-1608, v0.33 :1620-1624). So the
    // checkpoint block of group 2 enters the free queue first, then its hit
    // block, then the blocks of group 3.
    for version in [VllmRuleVersion::V0_32_0, VllmRuleVersion::V0_33_0] {
        let mut coordinator = with_limits(rules_glm(version, true));
        let mut kv = kv_pool(&groups_glm(true), 64);
        let pool = kv.block_pool_mut().unwrap();
        let mut block_lease = admit_with_a_checkpoint_block(&mut coordinator, pool, version);
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 13000, 13000, 13001, 13000));
        // The 4 hit and the 4 checkpoint blocks are cached now.
        assert_eq!(pool.active_blocks(), 11, "{version:?}");
        assert_eq!(pool.inactive_cached_blocks(), 8, "{version:?}");
        evict_one(pool);
        assert!(!hit_exists(pool, 2, 2176, 10880), "{version:?}");
        assert!(hit_exists(pool, 2, 2176, 8704), "{version:?}");
        evict_one(pool);
        assert!(!hit_exists(pool, 2, 2176, 8704), "{version:?}");
        assert!(hit_exists(pool, 3, 2176, 10880), "{version:?}");
        coordinator.release_request(pool, &mut block_lease);
    }
}

#[test]
fn a_request_that_ends_first_frees_its_checkpoint_block_before_its_boundary_hit_block() {
    // A request that ends frees its blocks manager by manager, in group order
    // (v0.32 kv_cache_coordinator.py:407-408, v0.33 :411-412), and each group's
    // table in reverse order (v0.32 single_type_kv_cache_manager.py:581-589,
    // v0.33 :591-599): in a state group the running block, the checkpoint
    // block, then the hit block, and then the next group.
    for version in [VllmRuleVersion::V0_32_0, VllmRuleVersion::V0_33_0] {
        let mut coordinator = with_limits(rules_glm(version, true));
        let mut kv = kv_pool(&groups_glm(true), 64);
        let pool = kv.block_pool_mut().unwrap();
        let mut block_lease = admit_with_a_checkpoint_block(&mut coordinator, pool, version);
        coordinator.release_request(pool, &mut block_lease);
        assert_eq!(pool.active_blocks(), 0, "{version:?}");
        // The five cached full-attention blocks go first, then the
        // checkpoint block of group 2.
        for _ in 0..6 {
            evict_one(pool);
        }
        assert!(!hit_exists(pool, 2, 2176, 10880), "{version:?}");
        assert!(hit_exists(pool, 2, 2176, 8704), "{version:?}");
        // Then the hit block of group 2, before the blocks of group 3.
        evict_one(pool);
        assert!(!hit_exists(pool, 2, 2176, 8704), "{version:?}");
        assert!(hit_exists(pool, 3, 2176, 10880), "{version:?}");
    }
}

#[test]
fn a_request_that_ends_first_frees_its_running_state_block_before_its_boundary_hit_block() {
    // The first chunk ends on the block boundary at 8,704, so the running
    // block of each state group holds the cached state at 8,704. A request
    // that ends frees each group's table in reverse order (v0.31
    // single_type_kv_cache_manager.py:581-589): the running block, at index
    // 3, before the hit block, at index 1. `release_request` has no rule that
    // depends on the vLLM version, so one rule row is enough.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 64);
    let pool = kv.block_pool_mut().unwrap();
    produce_on_a_boundary(&mut coordinator, pool, 0);
    let request = lease_from(0, 9000);
    let mut block_lease = admit_past_a_boundary_hit(&mut coordinator, pool);
    for group in GLM_STATES {
        assert_eq!(key_state(pool, group, &request, 8704), Some(true));
        assert_eq!(key_state(pool, group, &request, 4352), Some(true));
    }
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
    // The four full-attention blocks, tail first, then the running block of
    // group 2.
    for _ in 0..5 {
        evict_one(pool);
    }
    for at in [2176, 4352, 6528, 8704] {
        assert_eq!(key_state(pool, 0, &request, at), None, "{at}");
    }
    assert_eq!(key_state(pool, 2, &request, 8704), None);
    assert_eq!(key_state(pool, 2, &request, 4352), Some(false));
}

#[test]
fn startup_blocks_equal_the_pages_of_the_profile_peak_bound() {
    // The startup check of replay and the peak bound of planning count the
    // same blocks for one request of the context length. Every page of
    // `groups_glm` is one byte. By hand: 8,192 tokens fill cdiv(8,192, 2,176)
    // = 4 attention blocks, the ring holds 1, and each of the four KDA groups
    // holds `2 + c` blocks in align mode (v0.33 `iface:1135-1141`): 4 + 1 +
    // 4 * 2 = 13 without checkpoint blocks and 4 + 1 + 4 * 3 = 17 with them.
    // Replay models no checkpoint block on the v0.31.0 row
    // (`GroupedPrefixRules::new` refuses it), so that row has no case with
    // one.
    for (version, checkpoints) in [
        (VllmRuleVersion::V0_31_0, &[(false, 13)][..]),
        (VllmRuleVersion::V0_32_0, &[(false, 13), (true, 17)][..]),
        (VllmRuleVersion::V0_33_0, &[(false, 13), (true, 17)][..]),
    ] {
        for &(checkpoint, blocks) in checkpoints {
            let peak: u64 = groups_glm(checkpoint)
                .iter()
                .map(|group| {
                    group.startup_request_bytes(8192, 8192, true).unwrap() / group.page_size_bytes
                })
                .sum();
            assert_eq!(peak, blocks, "{version:?} {checkpoint}");
            assert_eq!(
                rules_glm(version, checkpoint).startup_blocks(8192, 16384),
                blocks as usize,
                "{version:?} {checkpoint}"
            );
        }
        // A window group in blocks of 4, with chunks of 8 tokens in two
        // batches (v0.33 `iface:918-955`, `vllm/config/vllm.py:588-609`), on
        // both sides. Window 8, context 64: cdiv(min(7 + 16, 64), 4) + 1 = 7.
        // Window 9, context 64: cdiv(min(8 + 16, 64), 4) + 1 = 7, where a
        // window without its `- 1` would give 8. Window 8, context 16:
        // cdiv(min(7 + 16, 16), 4) + 1 = 5. Window 8, context 17, one token
        // above a block: cdiv(min(7 + 16, 17), 4) + 1 = 6.
        for (window, context, blocks) in [(8, 64, 7), (9, 64, 7), (8, 16, 5), (8, 17, 6)] {
            let groups = [attention("window", 4, Some(window))];
            assert_eq!(
                groups[0].startup_request_bytes(context, 8, true).unwrap()
                    / groups[0].page_size_bytes,
                blocks,
                "{version:?} {window} {context}"
            );
            assert_eq!(
                GroupedPrefixRules::new(&groups, 4, version)
                    .unwrap()
                    .startup_blocks(context as usize, 2 * 8),
                blocks as usize,
                "{version:?} {window} {context}"
            );
        }
    }
    // The window branch refuses what the other branches refuse: a window of
    // 0, a context of 0 and a chunk of 0.
    let window = attention("window", 4, Some(8));
    for (group, context, chunk, refusal) in [
        (
            attention("window", 4, Some(0)),
            64,
            8,
            "cache group requires a name, positive layers/block/page/window",
        ),
        (
            window.clone(),
            0,
            8,
            "cache context and max_num_tokens must be positive",
        ),
        (
            window,
            64,
            0,
            "cache context and max_num_tokens must be positive",
        ),
    ] {
        let error = group
            .startup_request_bytes(context, chunk, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains(refusal), "{error}");
    }
}

/// The hash boundaries of `request`, up to `tokens`, whose key is cached in
/// `group`.
fn cached_positions(
    pool: &GroupedBlockPool,
    group: u16,
    request: &BlockRequestLease,
    hash_block: usize,
    tokens: usize,
) -> Vec<usize> {
    (hash_block..=tokens)
        .step_by(hash_block)
        .filter(|&end| {
            pool.key_hit(group, request.prefix_hash_at(end).unwrap())
                .is_some()
        })
        .collect()
}

/// Full attention, a ring and one state group, the first and the last with
/// blocks of 32 tokens. With `checkpoint`, the state group has one prefill
/// checkpoint block at alignment 4.
fn groups_of_32(checkpoint: bool) -> Vec<FpmCacheGroup> {
    let mut kda = state("kda", 32, true, checkpoint);
    if checkpoint {
        kda.prefill_checkpoint_alignment = Some(4);
    }
    vec![
        attention("mla", 32, None),
        state("kpool_tail", 4, false, false),
        kda,
    ]
}

#[test]
fn block_that_one_chunk_fills_past_the_prompt_tail_takes_both_keys() {
    // A prompt of 64 tokens at a hash block of 4, then the same prompt again.
    // vLLM 0.32.0 and 0.33.0 put the partial tail at 60, inside the second
    // block, which the same cache step has given its full key (v0.32.0
    // single:830-884, block_pool.py:449-523, :626-646, v0.33.0
    // single:840-897, block_pool.py:486-560, :666-686). vLLM 0.31.0 puts the
    // tail at 64, a block end, and registers none (v0.31.0 single:876-880).
    // Without a checkpoint block the same versions compute the prompt in
    // three chunks, so the block gets the tail key first and its full key
    // when the last chunk fills it (v0.32.0 block_pool.py:286-299, v0.33.0
    // :290-303), and the same cache step adds the tail key back. A cached
    // block counts once, whatever number of keys it holds.
    let hash = 4;
    for (
        version,
        checkpoint,
        a_chunks,
        a_keys,
        a_states,
        hit,
        b_chunks,
        b_blocks,
        b_states,
        cached,
    ) in [
        // v0.31.0: A is one chunk and leaves g0{32 64} g2{-}; B hits 0 with
        // the junction 32, computes 32 and 32, takes 3 blocks first and
        // leaves g2{32}.
        (
            VllmRuleVersion::V0_31_0,
            false,
            vec![64],
            vec![32, 64],
            vec![],
            (0, 32),
            vec![32, 64],
            3,
            vec![32],
            // g0{32x2 64x2} g2{32} are 5 cached blocks.
            5,
        ),
        // v0.32.0 and v0.33.0 with the checkpoint block: A is one chunk and
        // leaves g0{32 64+60} g2{60}; B hits 60 with no junction, computes 4,
        // takes 6 blocks first and leaves g2{60}.
        (
            VllmRuleVersion::V0_32_0,
            true,
            vec![64],
            vec![32, 60, 64],
            vec![60],
            (60, 0),
            vec![64],
            6,
            vec![60],
            // g0{32 64+60x2} g2{60} are 4 cached blocks.
            4,
        ),
        (
            VllmRuleVersion::V0_33_0,
            true,
            vec![64],
            vec![32, 60, 64],
            vec![60],
            (60, 0),
            vec![64],
            6,
            vec![60],
            4,
        ),
        // v0.32.0 and v0.33.0 without the checkpoint block: A computes 32, 28
        // and 4 and leaves g0{32 64+60} g2{32 60}; B hits 60 with no junction,
        // computes 4, takes 6 blocks first and leaves g2{32 60}.
        (
            VllmRuleVersion::V0_32_0,
            false,
            vec![32, 60, 64],
            vec![32, 60, 64],
            vec![32, 60],
            (60, 0),
            vec![64],
            6,
            vec![32, 60],
            // g0{32 64+60x2} g2{32 60} are 5 cached blocks.
            5,
        ),
        (
            VllmRuleVersion::V0_33_0,
            false,
            vec![32, 60, 64],
            vec![32, 60, 64],
            vec![32, 60],
            (60, 0),
            vec![64],
            6,
            vec![32, 60],
            5,
        ),
    ] {
        let groups = groups_of_32(checkpoint);
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(64, hash);
        let cold = coordinator.find_longest_hit(pool, &request, 64);
        let mut a = coordinator.new_lease();
        let ends = prefill(&mut coordinator, pool, &mut a, &request, 64, &cold, BUDGET);
        assert_eq!(ends, a_chunks, "{version:?}");
        assert_eq!(
            cached_positions(pool, 0, &request, hash, 64),
            a_keys,
            "{version:?}"
        );
        assert_eq!(
            cached_positions(pool, 2, &request, hash, 64),
            a_states,
            "{version:?}"
        );
        // The keys of group 0 lie on two blocks.
        assert_eq!(attention_copies(&a, 0), 2, "{version:?}");
        coordinator.release_request(pool, &mut a);

        let b_hit = coordinator.find_longest_hit(pool, &request, 64);
        assert_eq!((b_hit.tokens, b_hit.junction_tokens), hit, "{version:?}");
        assert_eq!(
            coordinator.first_allocation_blocks(pool, &b_hit, 64, b_chunks[0]),
            b_blocks,
            "{version:?}"
        );
        let mut b = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut b, &b_hit).unwrap();
        let ends = prefill(&mut coordinator, pool, &mut b, &request, 64, &b_hit, BUDGET);
        assert_eq!(ends, b_chunks, "{version:?}");
        assert_eq!(
            cached_positions(pool, 0, &request, hash, 64),
            a_keys,
            "{version:?}"
        );
        assert_eq!(
            cached_positions(pool, 2, &request, hash, 64),
            b_states,
            "{version:?}"
        );
        coordinator.release_request(pool, &mut b);
        assert_eq!(pool.active_blocks(), 0, "{version:?}");
        assert_eq!(pool.inactive_cached_blocks(), cached, "{version:?}");
    }
}

#[test]
fn tail_key_goes_to_a_cached_block_when_one_step_ends_past_that_block() {
    // A request of 64 prompt tokens that is admitted again after 35 output
    // tokens knows 99 tokens. Without a hit its first step runs from 0 to
    // 96, past the prompt end: 96 is the last block end below its 99 known
    // tokens, above the partial tail 60, so the split has no stop at the
    // tail. The cache step of 96 tokens gives the second block its full key,
    // and vLLM then adds the tail entry at 60 to that block, because 60 is
    // not above the cached count (v0.32.0 single:860-884, v0.33.0
    // single:870-897). The lease of this test holds the hashes of the prompt
    // only, so the block of the output tokens gets no key.
    let hash = 4;
    for version in [VllmRuleVersion::V0_32_0, VllmRuleVersion::V0_33_0] {
        let groups = groups_of_32(false);
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        assert_eq!(
            coordinator.rules().align_split(0, 64, 99, 99, 0, BUDGET),
            96
        );
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(64, hash);
        let cold = coordinator.find_longest_hit(pool, &request, 64);
        let mut a = coordinator.new_lease();
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut a, 98, 0, 96, 0));
        coordinator.finalize(pool, &mut a, &request, 64, 96, &cold);
        coordinator.release_copy_sources(pool);
        assert_eq!(
            cached_positions(pool, 0, &request, hash, 64),
            [32, 60, 64],
            "{version:?}"
        );
    }
}

#[test]
fn two_key_block_serves_a_hit_through_either_key_and_leaves_with_both() {
    // The second block of a 64-token prompt holds its full key 64 and the
    // key of the prompt tail 60 on vLLM 0.32.0 and 0.33.0. A repeat of the
    // prompt takes the block through the tail key and leaves a second block
    // with both keys, g0{32 64+60x2} (v0.32.0 single:830-884,
    // block_pool.py:449-523, :626-646, v0.33.0 single:840-897,
    // block_pool.py:486-560, :666-686). A longer prompt takes it through the
    // full key.
    let hash = 4;
    for version in [VllmRuleVersion::V0_32_0, VllmRuleVersion::V0_33_0] {
        let groups = groups_of_32(true);
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(64, hash);
        let key_state = |pool: &GroupedBlockPool, tokens| {
            pool.key_hit(0, request.prefix_hash_at(tokens).unwrap())
                .map(|hit| hit.is_active)
        };
        let cold = coordinator.find_longest_hit(pool, &request, 64);
        let mut a = coordinator.new_lease();
        prefill(&mut coordinator, pool, &mut a, &request, 64, &cold, BUDGET);
        coordinator.release_request(pool, &mut a);
        // A's cached blocks, oldest first: the block of 64 and 60, the block
        // of 32, the checkpoint block.
        assert_eq!(pool.inactive_cached_blocks(), 3, "{version:?}");

        // A prompt of 100 tokens finds the full blocks 32 and 64 in group 0
        // and the state at 60, so it hits 60 with the junction 4 (v0.32.0
        // coord:796-931, v0.33.0 coord:806-941) and takes the second block
        // through its full key.
        let long = lease(100, hash);
        let long_hit = coordinator.find_longest_hit(pool, &long, 100);
        assert_eq!((long_hit.tokens, long_hit.junction_tokens), (60, 4));
        assert_eq!(
            long_hit.per_group[0],
            GroupHitBlocks::Attention(vec![Some(key(0, hash, 32)), Some(key(0, hash, 64))])
        );
        let mut held = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut held, &long_hit).unwrap();
        assert_eq!(key_state(pool, 60), Some(true), "{version:?}");
        assert_eq!(key_state(pool, 64), Some(true), "{version:?}");
        coordinator.release_request(pool, &mut held);

        // The repeat takes the same block through the tail key, copies it
        // and gives its copy both keys.
        let b_hit = coordinator.find_longest_hit(pool, &request, 64);
        assert_eq!(
            b_hit.per_group[0],
            GroupHitBlocks::Attention(vec![Some(key(0, hash, 32)), Some(key(0, hash, 60))])
        );
        let mut b = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut b, &b_hit).unwrap();
        prefill(&mut coordinator, pool, &mut b, &request, 64, &b_hit, BUDGET);
        coordinator.release_request(pool, &mut b);
        // Cached, oldest first: A's block of 64 and 60 and A's checkpoint
        // block (the copy sources of B's pass), then B's block of 64 and 60,
        // then the block of 32.
        assert_eq!(pool.inactive_cached_blocks(), 4, "{version:?}");
        // A block leaves with every key that it holds
        // (v0.32.0 block_pool.py:590-609, :744, v0.33.0 :630-649, :784), and
        // a key stays visible while another block holds it.
        evict_one(pool);
        assert_eq!(
            cached_positions(pool, 0, &request, hash, 64),
            [32, 60, 64],
            "{version:?}"
        );
        evict_one(pool);
        assert!(cached_positions(pool, 2, &request, hash, 64).is_empty());
        evict_one(pool);
        assert_eq!(
            cached_positions(pool, 0, &request, hash, 64),
            [32],
            "{version:?}"
        );
        assert_eq!(pool.inactive_cached_blocks(), 1, "{version:?}");
    }
}

#[test]
fn block_with_the_key_of_the_prompt_tail_takes_its_full_key_when_a_later_chunk_fills_it() {
    // Two full-attention groups with blocks of 64 and 16 tokens, so a hash
    // block of 16 and hits on multiples of 64. A prompt of 128 tokens has its
    // partial tail at 112 on vLLM 0.32.0 and 0.33.0, inside the second
    // 64-token block, and at 128 on vLLM 0.31.0, which registers none. Then
    // a prompt of 200 tokens with the same first 128 (v0.31.0
    // single:845-892, v0.32.0 single:830-884, v0.33.0 single:840-897). At a
    // budget of 120 replay gives the block the tail key in the chunk that
    // ends at 120 and its full key in the chunk that fills it, the order of
    // v0.32.0 block_pool.py:286-299 and v0.33.0 :290-303. vLLM caches only
    // up to the scheduler block of 64 tokens here (v0.31.0
    // kv_cache_coordinator.py:796-806, v0.32.0 :767-777, v0.33.0 :777-787),
    // so it gives the block both keys in the second chunk. Both end with the
    // same keys. At a budget of 512 one chunk gives the block its full key
    // and then the tail key (v0.32.0 block_pool.py:449-523, :626-646,
    // v0.33.0 :486-560, :666-686).
    let hash = 16;
    let groups = [attention("wide", 64, None), attention("narrow", 16, None)];
    let narrow_keys: Vec<usize> = (16..=128).step_by(16).collect();
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        // Keys after A: g0{64 128} on v0.31.0 and g0{64 128+112} on v0.32.0
        // and v0.33.0, and g1{16 .. 128} on all three.
        let wide_keys = match version {
            VllmRuleVersion::V0_31_0 => vec![64, 128],
            VllmRuleVersion::V0_32_0 | VllmRuleVersion::V0_33_0 => vec![64, 112, 128],
        };
        // A computes 120 and 8 at a budget of 120, and 128 at a budget of
        // 512.
        for (budget, chunks) in [(120, vec![120, 128]), (512, vec![128])] {
            let mut coordinator =
                with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
            let mut kv = kv_pool(&groups, 64);
            let pool = kv.block_pool_mut().unwrap();
            let request = lease(128, hash);
            let cold = coordinator.find_longest_hit(pool, &request, 128);
            let mut a = coordinator.new_lease();
            let ends = prefill(&mut coordinator, pool, &mut a, &request, 128, &cold, budget);
            assert_eq!(ends, chunks, "{version:?} at {budget}");
            assert_eq!(
                cached_positions(pool, 0, &request, hash, 128),
                wide_keys,
                "{version:?} at {budget}"
            );
            assert_eq!(
                cached_positions(pool, 1, &request, hash, 128),
                narrow_keys,
                "{version:?} at {budget}"
            );
            // The keys of the 64-token group lie on two blocks.
            assert_eq!(attention_copies(&a, 0), 2, "{version:?} at {budget}");
            coordinator.release_request(pool, &mut a);

            let tokens: Vec<u32> = (0..128).chain(5000..5072).collect();
            let mut longer = BlockRequestLease::new(Uuid::nil(), Vec::new());
            longer.configure_prefix_hashes(&tokens, hash);
            let b_hit = coordinator.find_longest_hit(pool, &longer, 200);
            // B finds the full key 128 in both groups, so it hits 128 with
            // no junction. Its first allocation counts the blocks of 200
            // tokens, cdiv(200, 64) + cdiv(200, 16) = 4 + 13 = 17, the hit
            // blocks among them because no request holds them (v0.31.0
            // single:216-268, v0.32.0 single:213-265, v0.33.0
            // single:214-266).
            assert_eq!(
                (b_hit.tokens, b_hit.junction_tokens),
                (128, 0),
                "{version:?} at {budget}"
            );
            assert_eq!(
                coordinator.first_allocation_blocks(pool, &b_hit, 200, 200),
                17,
                "{version:?} at {budget}"
            );
            let mut b = coordinator.new_lease();
            coordinator.pin_hit(pool, &mut b, &b_hit).unwrap();
            let ends = prefill(&mut coordinator, pool, &mut b, &longer, 200, &b_hit, budget);
            // B computes its 72 new tokens in one chunk.
            assert_eq!(ends, [200], "{version:?} at {budget}");
            // Keys after B: g0{64 128(+112) 192} g1{16 .. 192}. The tail of
            // B, (200 - 1) // 16 * 16 = 192 at v0.32.0 and v0.33.0 and
            // 200 // 16 * 16 = 192 at v0.31.0, is a block end and gets no
            // entry.
            let mut wide_after = wide_keys.clone();
            wide_after.push(192);
            assert_eq!(
                cached_positions(pool, 0, &longer, hash, 200),
                wide_after,
                "{version:?} at {budget}"
            );
            assert_eq!(
                cached_positions(pool, 1, &longer, hash, 200),
                (16..=192).step_by(16).collect::<Vec<_>>(),
                "{version:?} at {budget}"
            );
            coordinator.release_request(pool, &mut b);

            // B held every block of A and released its blocks tail first,
            // the 64-token group first: the block of 192, A's second
            // 64-token block, the block of 64. So the first eviction takes
            // the block of 192 and its one key, and the second takes A's
            // second block with every key that it holds (v0.31.0
            // block_pool.py:593-612, :747, v0.32.0 :590-609, :744, v0.33.0
            // :630-649, :784).
            let before = cached_positions(pool, 0, &longer, hash, 200);
            evict_one(pool);
            assert_eq!(
                cached_positions(pool, 0, &longer, hash, 200).len() + 1,
                before.len()
            );
            evict_one(pool);
            assert_eq!(
                cached_positions(pool, 0, &longer, hash, 200),
                [64],
                "{version:?} at {budget}"
            );
        }
    }
}

#[test]
fn published_state_block_leaves_among_the_copy_sources_of_its_group_in_allocation_order() {
    // A running request P writes past the state that it published at 48,
    // inside its running state block, and a waiting request C starts on a hit
    // of 48 tokens inside a block of another prefix, in one pass. vLLM
    // collects the copies manager by manager, in group order, and each
    // manager in the order of its allocations, and frees them at the end of
    // `schedule()` (v0.31 kv_cache_manager.py:894-909, sched:1434-1442,
    // v0.32 kv_cache_manager.py:854-869, sched:1474-1482, v0.33
    // kv_cache_manager.py:863-878, sched:1698-1706). Of each pair one block
    // gets free: the copy that holds P's key, and the hit block of C.
    // So the free queue gets C's full-attention hit block,
    // then the two state blocks in the order of the two allocations.
    let hash = 4;
    let groups = [attention("full", 32, None), state("s", 32, true, false)];
    let tokens = |base: u32, count: u32| {
        let tokens: Vec<u32> = (base..base + count).collect();
        let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
        lease.configure_prefix_hashes(&tokens, hash);
        lease
    };
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for producer_first in [true, false] {
            let label = format!("{version:?} producer_first={producer_first}");
            let mut coordinator =
                with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
            let mut kv = kv_pool(&groups, 32);
            let pool = kv.block_pool_mut().unwrap();
            // F leaves the prefix that C hits: its state at 48 and its
            // full-attention blocks, the second with the key of 48.
            let finished = tokens(1000, 50);
            let cold = coordinator.find_longest_hit(pool, &finished, 50);
            let mut f = coordinator.new_lease();
            let ends = prefill(&mut coordinator, pool, &mut f, &finished, 50, &cold, BUDGET);
            assert_eq!(ends, [32, 48, 50], "{label}");
            coordinator.release_request(pool, &mut f);
            // P stands at its published state.
            let producer = tokens(0, 50);
            let cold = coordinator.find_longest_hit(pool, &producer, 50);
            let mut p = coordinator.new_lease();
            let mut before = 0;
            for target in [32, 48] {
                coordinator.begin_step();
                assert!(coordinator.allocate(pool, &mut p, 50, before, target, before));
                coordinator.finalize(pool, &mut p, &producer, 50, target, &cold);
                assert_eq!(coordinator.release_copy_sources(pool), 0, "{label}");
                before = target;
            }
            // One pass: P's write past 48 and C's first allocation.
            coordinator.begin_step();
            let consumer = tokens(1000, 60);
            let c_hit = coordinator.find_longest_hit(pool, &consumer, 60);
            assert_eq!(c_hit.tokens, 48, "{label}");
            let mut c = coordinator.new_lease();
            let write_past = |coordinator: &mut GroupedPrefixCoordinator,
                              pool: &mut GroupedBlockPool,
                              p: &mut GroupedBlockLease| {
                assert!(coordinator.allocate(pool, p, 50, 48, 50, 48));
            };
            if producer_first {
                write_past(&mut coordinator, pool, &mut p);
            }
            coordinator.pin_hit(pool, &mut c, &c_hit).unwrap();
            assert!(coordinator.allocate(pool, &mut c, 60, 48, 60, 48));
            if !producer_first {
                write_past(&mut coordinator, pool, &mut p);
            }
            // C's full-attention block and state block, and P's published
            // block.
            assert_eq!(coordinator.copy_source_count(), 3, "{label}");
            coordinator.finalize(pool, &mut p, &producer, 50, 50, &cold);
            coordinator.finalize(pool, &mut c, &consumer, 60, 60, &c_hit);
            let cached = pool.inactive_cached_blocks();
            assert_eq!(coordinator.release_copy_sources(pool), 3, "{label}");
            assert_eq!(pool.inactive_cached_blocks(), cached + 3, "{label}");

            // The cached blocks that were free before the pass go first.
            for _ in 0..cached {
                evict_one(pool);
            }
            let resident = |pool: &GroupedBlockPool| {
                [
                    hit_exists_in(pool, 0, &finished, 48),
                    hit_exists_in(pool, 1, &producer, 48),
                    hit_exists_in(pool, 1, &finished, 48),
                ]
            };
            assert_eq!(resident(pool), [true, true, true], "{label}");
            evict_one(pool);
            assert_eq!(resident(pool), [false, true, true], "{label}");
            evict_one(pool);
            assert_eq!(
                resident(pool),
                [false, !producer_first, producer_first],
                "{label}"
            );
            evict_one(pool);
            assert_eq!(resident(pool), [false, false, false], "{label}");
            coordinator.release_request(pool, &mut p);
            coordinator.release_request(pool, &mut c);
        }
    }
}

/// Whether the key of the first `tokens` tokens of `request` is cached in
/// `group`.
fn hit_exists_in(
    pool: &GroupedBlockPool,
    group: u16,
    request: &BlockRequestLease,
    tokens: usize,
) -> bool {
    pool.key_hit(group, request.prefix_hash_at(tokens).unwrap())
        .is_some()
}

#[test]
fn full_attention_keys_of_several_groups_wait_for_the_scheduler_block() {
    // With more than one cache group and no hits inside a block, vLLM caches
    // only the computed tokens up to the last multiple of the scheduler block
    // (v0.31.0 coord:796-806, :808-840, v0.32.0 coord:767-794, v0.33.0
    // coord:777-804). Two full-attention groups with blocks of 64 and 16
    // tokens have a scheduler block of 64 and a hash block of 16, and no
    // state group, so hits stay on multiples of 64.
    let hash = 16;
    let groups = [attention("wide", 64, None), attention("narrow", 16, None)];
    let narrow = |end: usize| (16..=end).step_by(16).collect::<Vec<_>>();
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        // A prompt of 120 tokens in one chunk. The cached count is 120
        // rounded down to 64: one block of the 64-token group and four of
        // the 16-token group. The partial tail is 112 at every version (120
        // / 16 * 16 and 119 / 16 * 16), above 64, so it gets no key
        // (v0.31.0 single:845-893, v0.32.0 single:830-884, v0.33.0
        // single:840-897).
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(120, hash);
        let cold = coordinator.find_longest_hit(pool, &request, 120);
        let mut a = coordinator.new_lease();
        let ends = prefill(&mut coordinator, pool, &mut a, &request, 120, &cold, 512);
        assert_eq!(ends, [120], "{version:?}");
        assert_eq!(
            cached_positions(pool, 0, &request, hash, 120),
            [64],
            "{version:?}"
        );
        assert_eq!(
            cached_positions(pool, 1, &request, hash, 120),
            narrow(64),
            "{version:?}"
        );
        coordinator.release_request(pool, &mut a);
        // Each of those 5 keys is on its own block, so 5 blocks stay cached.
        assert_eq!(pool.inactive_cached_blocks(), 5, "{version:?}");
        // A prompt of 200 tokens with the same first 120 hits 64 in both
        // groups, with no junction. Its first allocation counts the 5 hit
        // blocks and the fresh blocks of 200 tokens: 4 - 1 of the 64-token
        // group and 13 - 4 of the 16-token group, 17 in all.
        let tokens: Vec<u32> = (0..120).chain(5000..5080).collect();
        let mut longer = BlockRequestLease::new(Uuid::nil(), Vec::new());
        longer.configure_prefix_hashes(&tokens, hash);
        let b_hit = coordinator.find_longest_hit(pool, &longer, 200);
        assert_eq!(
            (b_hit.tokens, b_hit.junction_tokens),
            (64, 0),
            "{version:?}"
        );
        assert_eq!(
            coordinator.first_allocation_blocks(pool, &b_hit, 200, 200),
            17,
            "{version:?}"
        );
        let mut b = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut b, &b_hit).unwrap();
        let ends = prefill(&mut coordinator, pool, &mut b, &longer, 200, &b_hit, 512);
        assert_eq!(ends, [200], "{version:?}");
        // 200 rounded down to 64 is 192, a block end of both groups, and the
        // partial tail of 200 tokens is 192 at every version, so it adds no
        // entry.
        assert_eq!(
            cached_positions(pool, 0, &longer, hash, 200),
            [64, 128, 192],
            "{version:?}"
        );
        assert_eq!(
            cached_positions(pool, 1, &longer, hash, 200),
            narrow(192),
            "{version:?}"
        );
        coordinator.release_request(pool, &mut b);

        // A prompt of 128 tokens in chunks that end at 120 and 128. After
        // the first chunk the cached count is 64. The second chunk reaches
        // 128, a multiple of 64: the second 64-token block gets its full key
        // 128, and the 16-token group its keys from 80 to 128. vLLM 0.31.0
        // puts the partial tail at 128, a block end, with no entry. vLLM
        // 0.32.0 and 0.33.0 put it at 127 / 16 * 16 = 112, inside the second
        // block, which gets that key beside its full key (v0.32.0
        // block_pool.py:286-299, :449-523, v0.33.0 block_pool.py:290-303,
        // :486-560).
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(128, hash);
        let cold = coordinator.find_longest_hit(pool, &request, 128);
        let mut a = coordinator.new_lease();
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut a, 128, 0, 120, 0));
        coordinator.finalize(pool, &mut a, &request, 128, 120, &cold);
        assert_eq!(
            cached_positions(pool, 0, &request, hash, 128),
            [64],
            "{version:?}"
        );
        assert_eq!(
            cached_positions(pool, 1, &request, hash, 128),
            narrow(64),
            "{version:?}"
        );
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut a, 128, 120, 128, 120));
        coordinator.finalize(pool, &mut a, &request, 128, 128, &cold);
        let wide = match version {
            VllmRuleVersion::V0_31_0 => vec![64, 128],
            VllmRuleVersion::V0_32_0 | VllmRuleVersion::V0_33_0 => vec![64, 112, 128],
        };
        assert_eq!(
            cached_positions(pool, 0, &request, hash, 128),
            wide,
            "{version:?}"
        );
        assert_eq!(
            cached_positions(pool, 1, &request, hash, 128),
            narrow(128),
            "{version:?}"
        );
        coordinator.release_request(pool, &mut a);

        // A group that takes no part in prefix caching still counts: vLLM
        // picks its coordinator by the number of all cache groups (v0.31.0
        // coord:1043-1044, v0.32.0 :1009-1010, v0.33.0 :1021-1022). A
        // full-attention group of 16 tokens and a ring of 4 tokens have a
        // scheduler block of 16. No state group has a block above the hash
        // block of 4, so hits stay on multiples of 16, and a prompt of 22
        // tokens caches 16 tokens: the full key 16 and no entry at the
        // partial tail, 20. A longer prompt that shares the 22 tokens hits 16.
        // These keys and this hit are derived from the rules above and from
        // vLLM's lookup, which skips a group that is not prefix-cacheable
        // (v0.31.0 coord:748-749, v0.32.0 :719-720, v0.33.0 :729-730).
        let ringed = [attention("full", 16, None), state("ring", 4, false, false)];
        let (request, longer) = (lease(22, 4), extended(22, 20, 4));
        let (keys, hit) = keys_and_hit(&ringed, 4, version, (&request, 22), (&longer, 42));
        assert_eq!(keys, [vec![16]], "{version:?}");
        assert_eq!(hit, 16, "{version:?}");

        // A prompt one token below a multiple of the scheduler block: 127
        // tokens on the groups of 64 and 16 tokens cache 64 tokens, so the
        // keys are 64 and 16 to 64, and the partial tail, 112 at every
        // version (127 / 16 * 16 and 126 / 16 * 16), gets no entry. A longer
        // prompt that shares the 127 tokens hits 64. These keys and this hit
        // are derived from the same rules.
        let (request, longer) = (lease(127, hash), extended(127, 73, hash));
        let (keys, hit) = keys_and_hit(&groups, hash, version, (&request, 127), (&longer, 200));
        assert_eq!(keys, [vec![64], narrow(64)], "{version:?}");
        assert_eq!(hit, 64, "{version:?}");

        // A scheduler block above the largest group block: a full-attention
        // group of 8 tokens and a ring of 3 tokens have a scheduler block of
        // 24, the least common multiple of the two blocks (v0.31.0
        // kv_cache_utils.py:767-770, v0.32.0 :848-851, v0.33.0 :860-863).
        // A prompt of 22 tokens caches 0 tokens and leaves no key, and a
        // longer prompt that shares it hits 0. A prompt of 30 tokens caches
        // 24 tokens and leaves the keys 8, 16 and 24; its partial tail, 28
        // at every version (30 / 4 * 4 and 29 / 4 * 4), lies above 24 and
        // gets no entry. A longer prompt that shares the 30 tokens hits 24.
        // These keys and these hits are derived from the same rules and the
        // scheduler block of 24.
        let ringed = [attention("full", 8, None), state("ring", 3, false, false)];
        for (prompt, expected, expected_hit) in [(22, vec![], 0), (30, vec![8, 16, 24], 24)] {
            let (request, longer) = (lease(prompt, 4), extended(prompt, 20, 4));
            let (keys, hit) = keys_and_hit(
                &ringed,
                4,
                version,
                (&request, prompt),
                (&longer, prompt + 20),
            );
            assert_eq!(keys, [expected], "{version:?} prompt {prompt}");
            assert_eq!(hit, expected_hit, "{version:?} prompt {prompt}");
        }

        // A sliding-window group as the second group: a full-attention group
        // of 16 tokens and a window of 4 tokens on blocks of 4 tokens, at a
        // hash block of 4, have a scheduler block of 16. A prompt of 22
        // tokens caches 16 tokens: the full-attention key 16, and no entry at
        // the partial tail, 20 at every version. Under the default retention
        // the window keeps the block below the replay boundary, the prompt
        // length minus 1 (v0.31.0 kv_cache_coordinator.py:319-338, v0.32.0
        // :65-86, v0.33.0 :65-86) rounded down to the scheduler block, here
        // 16, so its key is 16 too. A prompt of 70 tokens caches 64 tokens: the
        // full-attention keys 16 to 64 and no entry at the partial tail, 68,
        // and the window key at the replay boundary 64. These keys and the hits
        // 16 and 64 of a longer prompt are derived from the rules above and
        // from the retention of a window group: at the default retention
        // interval 0 (v0.31.0 config/cache.py:160, v0.32.0 :159, v0.33.0 :275)
        // it keeps only the cdiv(4 - 1, 4) = 1 block that ends at the replay
        // boundary rounded down to the scheduler block (v0.31.0 single:961-971,
        // :1072-1151, v0.32.0 :943-953, :1054-1133, v0.33.0 :956-966,
        // :1067-1146).
        let windowed = [attention("full", 16, None), attention("window", 4, Some(4))];
        for (prompt, full, window) in [
            (22, vec![16], vec![16]),
            (70, vec![16, 32, 48, 64], vec![64]),
        ] {
            let (request, longer) = (lease(prompt, 4), extended(prompt, 20, 4));
            let (keys, hit) = keys_and_hit(
                &windowed,
                4,
                version,
                (&request, prompt),
                (&longer, prompt + 20),
            );
            assert_eq!(keys, [full, window], "{version:?} prompt {prompt}");
            assert_eq!(hit, prompt / 16 * 16, "{version:?} prompt {prompt}");
        }
    }
}

/// A lease of the tokens `0..prompt` followed by `extra` other tokens.
fn extended(prompt: usize, extra: usize, hash_block: usize) -> BlockRequestLease {
    let tokens: Vec<u32> = (0..prompt as u32)
        .chain(5000..5000 + extra as u32)
        .collect();
    let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
    lease.configure_prefix_hashes(&tokens, hash_block);
    lease
}

/// Prefill the `prompt` tokens of `request` cold in one chunk on a fresh
/// pool of `groups` and release it. Returns the cached hash boundaries of
/// each prefix-cacheable group, up to the prompt, and the hit of the
/// `longer_len` tokens of `longer` after the release.
fn keys_and_hit(
    groups: &[FpmCacheGroup],
    hash_block: usize,
    version: VllmRuleVersion,
    (request, prompt): (&BlockRequestLease, usize),
    (longer, longer_len): (&BlockRequestLease, usize),
) -> (Vec<Vec<usize>>, usize) {
    let mut coordinator =
        with_limits(GroupedPrefixRules::new(groups, hash_block, version).unwrap());
    let mut kv = kv_pool(groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let cold = coordinator.find_longest_hit(pool, request, prompt);
    let mut a = coordinator.new_lease();
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut a,
        request,
        prompt,
        &cold,
        BUDGET,
    );
    assert_eq!(ends, [prompt], "{version:?}");
    coordinator.release_request(pool, &mut a);
    let keys = coordinator
        .rules()
        .groups
        .iter()
        .enumerate()
        .filter(|(_, rule)| rule.cacheable)
        .map(|(g, _)| cached_positions(pool, g as u16, request, hash_block, prompt))
        .collect();
    let hit = coordinator
        .find_longest_hit(pool, longer, longer_len)
        .tokens;
    (keys, hit)
}

#[test]
fn rounding_to_the_scheduler_block_changes_no_key_where_it_is_the_full_attention_block() {
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        // The GLM groups at a hash block equal to their block: several cache
        // groups and no hits inside a block, so vLLM rounds the cached count
        // down to the scheduler block (v0.31.0 coord:796-806, v0.32.0
        // coord:767-777, v0.33.0 coord:777-787), which is the block of the
        // full-attention group: the ring block of 4 tokens divides it. Every
        // block end of that group is a multiple of the scheduler block, so
        // the rounding changes no key. After each chunk the group holds the
        // keys of its full blocks, and the hash block equals the block, so
        // there is no partial tail entry.
        let hash = GLM_BLOCK as usize;
        let groups = groups_glm(false);
        let mut coordinator = with_limits(rules_glm(version, false));
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(9000, hash);
        let cold = coordinator.find_longest_hit(pool, &request, 9000);
        let mut a = coordinator.new_lease();
        let mut computed = 0;
        for target in [6528, 8704, 9000] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut a, 9000, computed, target, computed));
            coordinator.finalize(pool, &mut a, &request, 9000, target, &cold);
            assert_eq!(
                cached_positions(pool, 0, &request, hash, 9000),
                (hash..=target).step_by(hash).collect::<Vec<_>>(),
                "{version:?} at {target}"
            );
            computed = target;
        }
        coordinator.release_request(pool, &mut a);
    }
}

#[test]
fn cached_count_is_the_computed_count_with_hits_inside_a_block_or_one_group() {
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        // Hits inside a block: vLLM caches the computed count as it is
        // (v0.31.0 coord:804-805, v0.32.0 coord:775-776, v0.33.0
        // coord:785-786). A prompt of 45 tokens at a hash block of 4 on the
        // 32-token layout computes 32, 44 and 45. Its partial tail is 44 at
        // every version (45 / 4 * 4 and 44 / 4 * 4), inside the second
        // block, so the full-attention group has the key 32 after the first
        // chunk and the keys 32 and 44 from the chunk to 44 on.
        let hash = 4;
        let groups = groups_of_32(false);
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(45, hash);
        let cold = coordinator.find_longest_hit(pool, &request, 45);
        let mut a = coordinator.new_lease();
        let mut computed = 0;
        for (target, keys) in [(32, vec![32]), (44, vec![32, 44]), (45, vec![32, 44])] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut a, 45, computed, target, computed));
            coordinator.finalize(pool, &mut a, &request, 45, target, &cold);
            coordinator.release_copy_sources(pool);
            assert_eq!(
                cached_positions(pool, 0, &request, hash, 45),
                keys,
                "{version:?} at {target}"
            );
            computed = target;
        }
        coordinator.release_request(pool, &mut a);

        // One cache group with a hash block below its block is a layout of
        // replay alone. vLLM takes the block as the hash block of one group
        // at v0.31.0 and ignores the unit (v0.31.0 kv_cache_utils.py:761-765),
        // so its scheduler keeps only the key 16 for this prompt, and it
        // refuses the unit at v0.32.0 and v0.33.0 (v0.32.0
        // kv_cache_utils.py:828-846, v0.33.0 :840-858). Its coordinator of
        // one group requires the hash block to equal the block (v0.31.0
        // kv_cache_coordinator.py:572-574, v0.32.0 :565-567, v0.33.0
        // :573-575). Replay accepts the layout and applies to it the rule of
        // that coordinator, which does not round the cached count (v0.31.0
        // coord:340-364, :1043-1044, v0.32.0 coord:346-377, :1009-1010,
        // v0.33.0 coord:350-381, :1021-1022), and the partial-tail entry of
        // a hash block below the block (v0.31.0 single:845-893, v0.32.0
        // single:830-884, v0.33.0 single:840-897). So a prompt of 22 tokens
        // at a hash block of 4 has its partial tail at 20 at every version
        // (22 / 4 * 4 and 21 / 4 * 4), inside the second 16-token block, and
        // the key is there when the 22 tokens are computed, beside the full
        // key 16.
        let groups = [attention("full", 16, None)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(22, hash);
        let cold = coordinator.find_longest_hit(pool, &request, 22);
        let mut a = coordinator.new_lease();
        prefill(&mut coordinator, pool, &mut a, &request, 22, &cold, BUDGET);
        assert_eq!(
            cached_positions(pool, 0, &request, hash, 22),
            [16, 20],
            "{version:?}"
        );
        coordinator.release_request(pool, &mut a);
    }
}

#[test]
fn convolution_group_keeps_one_block_before_the_replay_boundary() {
    // A full-attention group of 16-token blocks and a convolution group with
    // a block and a window of 4 tokens, at a hash block of 4: a scheduler
    // block of 16, and hits on multiples of 16. A prompt of 64 tokens runs in
    // two chunks of 32. The full-attention group keys its four full blocks,
    // and at v0.32.0 and v0.33.0 the prompt tail (64 - 1) / 4 * 4 = 60 on the
    // block that ends at 64 (v0.32.0 single_type_kv_cache_manager.py:860-868,
    // v0.33.0 :870-881); v0.31.0 takes 64 / 4 * 4 = 64, a block end (v0.31.0
    // :876). Under the default retention a window keeps only the block below
    // the replay boundary, num_prompt_tokens - 1 = 63, rounded down to the
    // alignment: the block that ends at 48 (v0.31.0 kv_cache_coordinator.py
    // :319-338, v0.32.0 :65-86, v0.33.0 :65-86, and the window mask v0.31.0
    // single_type_kv_cache_manager.py:1072-1151, v0.32.0 :1054-1133, v0.33.0
    // :1067-1146). So 4 + 1 blocks stay cached.
    //
    // A prompt of 74 tokens with the same first 64 then hits 48: the window
    // has no key at 64, and a hit is the shortest over the groups. The
    // junction is 64. It holds 3 hit and 2 new full-attention blocks, and
    // the convolution block that ends at 48 plus cdiv(74, 4) - 12 = 7 new
    // ones: 13 blocks. It computes the block that ends at 64 again, which
    // gets the key 64 a second time, and its window keeps the block below 64,
    // the replay boundary 73 and the junction rounded down to 16. Its tail
    // 72 lies above its last scheduler block, 64, and gets no key. So 7
    // blocks stay cached, two of them in the convolution group.
    let hash = 4;
    let mut convolution = attention("convolution", 4, Some(4));
    convolution.kind = FpmCacheKind::Convolution;
    let groups = [attention("full", 16, None), convolution];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 200);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(64, hash);
        let cold = coordinator.find_longest_hit(pool, &request, 64);
        let mut a = coordinator.new_lease();
        let ends = prefill(&mut coordinator, pool, &mut a, &request, 64, &cold, 32);
        assert_eq!(ends, [32, 64], "{version:?}");
        let full_keys = match version {
            VllmRuleVersion::V0_31_0 => vec![16, 32, 48, 64],
            VllmRuleVersion::V0_32_0 | VllmRuleVersion::V0_33_0 => vec![16, 32, 48, 60, 64],
        };
        assert_eq!(
            cached_positions(pool, 0, &request, hash, 64),
            full_keys,
            "{version:?}"
        );
        assert_eq!(
            cached_positions(pool, 1, &request, hash, 64),
            [48],
            "{version:?}"
        );
        coordinator.release_request(pool, &mut a);
        assert_eq!(pool.inactive_cached_blocks(), 5, "{version:?}");

        let tokens: Vec<u32> = (0..64).chain(2000..2010).collect();
        let mut longer = BlockRequestLease::new(Uuid::nil(), Vec::new());
        longer.configure_prefix_hashes(&tokens, hash);
        let hit = coordinator.find_longest_hit(pool, &longer, 74);
        assert_eq!(
            (hit.tokens, hit.tokens + hit.junction_tokens),
            (48, 64),
            "{version:?}"
        );
        let mut b = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut b, &hit).unwrap();
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut b, 74, 48, 74, 48));
        assert_eq!(pool.active_blocks(), 13, "{version:?}");
        assert_eq!(attention_copies(&b, 0), 5, "{version:?}");
        assert_eq!(attention_copies(&b, 1), 8, "{version:?}");
        coordinator.finalize(pool, &mut b, &longer, 74, 74, &hit);
        coordinator.release_request(pool, &mut b);
        assert_eq!(pool.inactive_cached_blocks(), 7, "{version:?}");
        assert_eq!(
            cached_positions(pool, 1, &longer, hash, 74),
            [48, 64],
            "{version:?}"
        );
    }
}

#[test]
fn keeps_in_flight_hold_in_each_kind_of_layout() {
    // A layout without a prefix-cacheable state group keeps the hold. A
    // layout with one keeps it only if it has a prefix-cacheable
    // full-attention group and every prefix-cacheable group has a block
    // equal to the scheduler block, the least common multiple of the blocks
    // of all groups. Each layout lists its groups, its hash block, its
    // scheduler block in the label, and the answer.
    let opted_out = |name: &str, block: u32| {
        let mut group = attention(name, block, None);
        group.prefix_cacheable = Some(false);
        group
    };
    let mut convolution = attention("conv", 8, Some(8));
    convolution.kind = FpmCacheKind::Convolution;
    let cases: [(&str, Vec<FpmCacheGroup>, usize, bool); 21] = [
        (
            "no state group (8)",
            vec![attention("full_8", 8, None), attention("full_4", 4, None)],
            4,
            true,
        ),
        (
            "no state group, a window group only (8)",
            vec![attention("window", 8, Some(16))],
            8,
            true,
        ),
        (
            "equal blocks (8)",
            vec![attention("full", 8, None), state("s", 8, true, false)],
            8,
            true,
        ),
        (
            "state block above the full-attention block (16)",
            vec![attention("full", 8, None), state("s", 16, true, false)],
            8,
            false,
        ),
        (
            "two state blocks above the full-attention block (16)",
            vec![
                attention("full", 4, None),
                state("s_0", 16, true, false),
                state("s_1", 16, true, false),
            ],
            4,
            false,
        ),
        (
            "two full-attention groups and a state group at one block (8)",
            vec![
                attention("full_0", 8, None),
                attention("full_1", 8, None),
                state("s", 8, true, false),
            ],
            8,
            true,
        ),
        (
            "a window group at the state block (8)",
            vec![
                attention("full", 8, None),
                attention("window", 8, Some(16)),
                state("s", 8, true, false),
            ],
            8,
            true,
        ),
        (
            "a window group below the state block (8)",
            vec![
                attention("full", 8, None),
                attention("window", 4, Some(8)),
                state("s", 8, true, false),
            ],
            4,
            false,
        ),
        (
            "an opted-out full-attention group that divides the state block (8)",
            vec![
                attention("full", 8, None),
                opted_out("full_4", 4),
                state("s", 8, true, false),
            ],
            8,
            true,
        ),
        (
            "ring below equal blocks (8)",
            vec![
                attention("full", 8, None),
                state("ring", 4, false, false),
                state("s", 8, true, false),
            ],
            8,
            true,
        ),
        (
            "state block below the full-attention block (8)",
            vec![attention("full", 8, None), state("s", 2, true, false)],
            2,
            false,
        ),
        (
            "state block below the larger of two full-attention blocks (8)",
            vec![
                attention("full_4", 4, None),
                state("s", 4, true, false),
                attention("full_8", 8, None),
            ],
            4,
            false,
        ),
        (
            "two state blocks, one below the other (8)",
            vec![
                attention("full", 4, None),
                state("s_4", 4, true, false),
                state("s_8", 8, true, false),
            ],
            4,
            false,
        ),
        (
            "state and ring only (2)",
            vec![state("ring", 2, false, false), state("s", 2, true, false)],
            2,
            false,
        ),
        (
            "no prefix-cacheable full-attention group, a window group (8)",
            vec![attention("window", 8, Some(16)), state("s", 2, true, false)],
            2,
            false,
        ),
        (
            "no prefix-cacheable full-attention group, a convolution group (8)",
            vec![convolution.clone(), state("s", 2, true, false)],
            2,
            false,
        ),
        (
            "an opted-out full-attention group only, at the state block (4)",
            vec![opted_out("full", 4), state("s", 4, true, false)],
            4,
            false,
        ),
        (
            "an opted-out full-attention group only, below the state block (12)",
            vec![opted_out("full", 3), state("s", 4, true, false)],
            4,
            false,
        ),
        (
            "state block below the scheduler block by a ring of 12 (12)",
            vec![
                attention("full", 4, None),
                state("ring", 12, false, false),
                state("s", 4, true, false),
            ],
            4,
            false,
        ),
        (
            "state block below the scheduler block by a window group (8)",
            vec![
                attention("full", 2, None),
                attention("window", 8, Some(16)),
                state("s", 2, true, false),
            ],
            2,
            false,
        ),
        (
            "state block below the scheduler block by an opted-out group (16)",
            vec![
                opted_out("full_16", 16),
                attention("full_4", 4, None),
                state("s", 4, true, false),
            ],
            4,
            false,
        ),
    ];
    for (label, groups, hash, expected) in cases {
        assert_eq!(
            rules(&groups, hash).keeps_in_flight_hold(),
            expected,
            "{label}"
        );
    }
    // A state block below the scheduler block by a convolution group, beside
    // a full-attention group at the state block (8).
    let groups = vec![
        attention("full", 2, None),
        convolution.clone(),
        state("s", 2, true, false),
    ];
    assert!(!rules(&groups, 2).keeps_in_flight_hold());
    // A convolution group at the full-attention and state block (8).
    let groups = vec![
        attention("full", 8, None),
        convolution,
        state("s", 8, true, false),
    ];
    assert!(rules(&groups, 8).keeps_in_flight_hold());
    // The GLM-5.3-Flash layout: MLA and four state groups of 2,176-token
    // blocks and a 4-token ring, so the scheduler block is 2,176, at the hash
    // block 2,176 and at a hash block of 4, with and without a checkpoint
    // block where the version takes one.
    for (version, checkpoint) in [
        (VllmRuleVersion::V0_31_0, false),
        (VllmRuleVersion::V0_32_0, false),
        (VllmRuleVersion::V0_32_0, true),
        (VllmRuleVersion::V0_33_0, false),
        (VllmRuleVersion::V0_33_0, true),
    ] {
        for hash in [GLM_BLOCK as usize, 4] {
            assert!(
                rules_glm_hash(version, checkpoint, hash).keeps_in_flight_hold(),
                "{} {checkpoint} {hash}",
                version.label()
            );
        }
    }
}
