// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

// Modified block-mode semantics derived from vLLM v0.31.0, v0.32.0 and v0.33.0:
// https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6
// https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be
// https://github.com/vllm-project/vllm/tree/4ba787ec9959e566d3b587a796dff07673324da5
// Exact upstream source paths and Apache-2.0 attribution: THIRD_PARTY_NOTICES.md.

//! Request-owned cache groups sharing one rank-local physical budget. Byte
//! mode replays them cold. Block mode keeps the blocks of every group in one
//! block pool, so that cached blocks stay until eviction.

use crate::engine::cache::vllm_block_pool::{
    BlockCopyId, BlockReservation, CacheKey, KeyInBlock, PrefixHit, VllmBlockPool,
};
use crate::engine::common::hashing::SequenceHash;
use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Default)]
pub(crate) struct GroupedLease {
    /// In byte mode, the blocks that the request holds across every
    /// configured group. Expired window blocks are not counted. A state group
    /// counts its fixed blocks.
    blocks: usize,
    bytes: u64,
    /// The copies of every group in block mode.
    pub(crate) block_lease: Option<GroupedBlockLease>,
    /// What a destination load still needs for its whole prompt, as vLLM's
    /// `_request_remaining_blocks` counts it: blocks in block mode, bytes in
    /// byte mode. 0 once its first pass has allocated, and on every other
    /// lease.
    pub(crate) first_pass_need: u64,
}

impl GroupedLease {
    pub(crate) fn blocks(&self) -> usize {
        match &self.block_lease {
            Some(lease) => lease.groups.iter().map(GroupLease::live_copies).sum(),
            None => self.blocks,
        }
    }

    /// Adds the blocks each group holds for a forward from `computed` to
    /// `target` tokens, or `None` when the bytes overflow. A window starts
    /// from `computed`, which `GroupedKvPool::allocate` gets as the processed
    /// count.
    fn charge(mut self, groups: &[FpmCacheGroup], computed: u64, target: u64) -> Option<Self> {
        for group in groups {
            // A forward must retain the history needed by its first query,
            // together with every token in the scheduled chunk. Freeing to
            // the final query's window here would undercount prefill storage.
            let range = group.validated_block_range(computed, target);
            let blocks = range.end - range.start;
            self.bytes = self
                .bytes
                .checked_add(blocks.checked_mul(group.page_size_bytes)?)?;
            self.blocks += blocks as usize;
        }
        Some(self)
    }
}

/// One entry per cache group of the profile, in profile order, state groups
/// included.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupedBlockLease {
    pub(crate) groups: Vec<GroupLease>,
    /// Whether the request has allocated once. The copy blocks of a partial
    /// hit come with the first allocation only.
    pub(crate) allocated: bool,
    /// The token count below which a step of the request finds nothing to
    /// allocate, publish or release, so that its `allocate` and `finalize`
    /// can be skipped (see `GroupedPrefixRules::quiet_until`). 0 when the
    /// next step has to run both.
    pub(crate) quiet_until: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GroupLease {
    /// `copies[i]` holds logical block `i`. It is `None` for an expired
    /// window block or a null block below a sliding-window hit. Such blocks
    /// lie below `first_live`. `finalized` counts the blocks that the request
    /// had completed at its last finalization. The next finalization visits
    /// only the full-attention blocks above it, as vLLM's `num_cached_block`
    /// does, and only the sliding-window retention positions above
    /// `finalized * block_size`. The window run that it publishes before such
    /// a position can include blocks below `finalized`.
    Attention {
        copies: Vec<Option<BlockCopyId>>,
        first_live: usize,
        finalized: usize,
    },
    State(GroupStateLease),
    /// One block for the whole request. A ring never caches it.
    Ring {
        copy: Option<BlockCopyId>,
    },
}

impl GroupLease {
    pub(super) fn live_copies(&self) -> usize {
        match self {
            // The `None` copies lie exactly below `first_live`, so a count per
            // step does not scan the whole context.
            Self::Attention {
                copies, first_live, ..
            } => {
                debug_assert_eq!(copies.iter().flatten().count(), copies.len() - first_live);
                copies.len() - first_live
            }
            Self::State(state) => {
                [
                    state.running,
                    state.hit.map(|(id, _)| id),
                    state.copy.map(|(id, _)| id),
                    state.checkpoint.map(|(id, _)| id),
                ]
                .into_iter()
                .flatten()
                .count()
                    + state.previous.len()
            }
            Self::Ring { copy } => usize::from(copy.is_some()),
        }
    }
}

/// When a previous state block goes. vLLM frees the state block of index
/// `i` at the first allocation of its request with a processed token count
/// above `(i + 1) * B` (v0.31 `mgr:549-556`, `single:1626-1675`,
/// `:1990-1995`, v0.32 `mgr:552-559`, `single:1592-1641`, `:1954-1959`,
/// v0.33 `mgr:561-568`, `single:1608-1657`, `:1971-1976`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreviousRelease {
    /// The block goes at the first allocation with a processed token count
    /// above this one.
    Above(usize),
    /// The block goes at the first allocation of a step after this one.
    AfterStep(u64),
}

impl PreviousRelease {
    pub(crate) fn is_met(self, processed: usize, step: u64) -> bool {
        match self {
            Self::Above(tokens) => processed > tokens,
            Self::AfterStep(after) => step > after,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupStateLease {
    /// The block that the request writes.
    pub(crate) running: Option<BlockCopyId>,
    /// A cached hit block, and the index of the block whose end it covers.
    /// The block of a hit on a block boundary is held until an allocation
    /// whose processed token count lies past that block.
    pub(crate) hit: Option<(BlockCopyId, usize)>,
    /// The running blocks before a boundary crossing or a producer
    /// copy-on-write, oldest first, each with its release rule. A crossing
    /// from `c` tokens leaves the block of index `(c - 1) / B`, which goes
    /// once the processed count passes its end. With the step before in
    /// flight, the blocks of two crossings can be held at once. A block kept
    /// for a state that the request published inside it goes at the next
    /// step of the request.
    pub(crate) previous: Vec<(BlockCopyId, PreviousRelease)>,
    /// The copy block of a partial hit, and the token count above which the
    /// processed count frees it: the end of its block.
    pub(crate) copy: Option<(BlockCopyId, usize)>,
    /// A vLLM 0.32.0 prefill checkpoint block, and its token position. It
    /// goes once the processed count passes that position.
    pub(crate) checkpoint: Option<(BlockCopyId, usize)>,
    /// The key of a state that the request published inside its running
    /// block. The next allocation keeps that block and takes a fresh one.
    pub(crate) published: Option<CacheKey>,
}

/// The block pool of block mode. Every group takes blocks of one page size
/// from it, and every cached block has a `CacheKey::Grouped` key.
pub(crate) struct GroupedBlockPool {
    pool: VllmBlockPool,
    page_size_bytes: u64,
    /// Unique among the pools of the process, so that a residency stamp of
    /// one pool never matches another pool.
    id: u64,
    /// See `GroupedKvPool::retry_epoch`.
    retry_epoch: u64,
    /// The most active blocks seen since the last reset, for tests that
    /// observe a step's peak after the step.
    #[cfg(test)]
    peak_active_blocks: std::cell::Cell<usize>,
}

impl GroupedBlockPool {
    /// The pool and the generation of its resident keys. Two equal values
    /// mean the same pool with the same set of resident keys, so every
    /// `key_hit(..).is_some()` gives the same answer.
    pub(crate) fn residency(&self) -> (u64, u64) {
        (self.id, self.pool.grouped_key_generation())
    }

    pub(crate) fn capacity_blocks(&self) -> usize {
        self.pool.capacity()
    }

    #[cfg(test)]
    pub(crate) fn peak_active_blocks(&self) -> usize {
        self.peak_active_blocks.get()
    }

    #[cfg(test)]
    pub(crate) fn reset_peak_active_blocks(&self) {
        self.peak_active_blocks.set(self.pool.num_active());
    }

    fn note_peak(&self) {
        #[cfg(test)]
        self.peak_active_blocks
            .set(self.peak_active_blocks.get().max(self.pool.num_active()));
    }

    pub(crate) fn key_hit(&self, group: u16, hash: SequenceHash) -> Option<PrefixHit> {
        self.pool.key_hit(CacheKey::Grouped { group, hash })
    }

    /// Pin `hits` and reserve `fresh` blocks, or change nothing when they do
    /// not fit. Every hit must be resident. A hit block in the inactive LRU
    /// leaves it and costs capacity again. An active hit block costs nothing.
    pub(crate) fn reserve(&mut self, hits: &[CacheKey], fresh: usize) -> Option<BlockReservation> {
        self.pool
            .reserve_keys(hits.iter().copied(), hits.len() + fresh)
            .map(|outcome| outcome.reservation)
    }

    /// Turn the pinned hits into request references. The copies follow the
    /// order of `hits`.
    pub(crate) fn activate(&mut self, reservation: &mut BlockReservation) -> Vec<BlockCopyId> {
        let copies = self
            .pool
            .activate_keys(reservation)
            .map(|(_, id)| id)
            .collect();
        self.note_peak();
        copies
    }

    pub(crate) fn take_private(&mut self, reservation: &mut BlockReservation) -> BlockCopyId {
        let id = self.pool.allocate_private(reservation);
        self.note_peak();
        id
    }

    /// Publish a private block. Returns whether `key` became visible.
    pub(crate) fn cache(&mut self, id: BlockCopyId, key: CacheKey) -> bool {
        debug_assert!(
            matches!(key, CacheKey::Grouped { .. }),
            "a grouped pool caches grouped keys only"
        );
        let visible = self.pool.cache_private_key(id, key);
        self.retry_epoch += u64::from(visible);
        visible
    }

    /// [`Self::cache`] for a key that can lie inside a block of its group, at
    /// `at`. A hit search finds such a key through [`Self::keys_in_block`] only.
    pub(crate) fn cache_at(
        &mut self,
        id: BlockCopyId,
        key: CacheKey,
        at: Option<KeyInBlock>,
    ) -> bool {
        let visible = self.pool.cache_private_key_at(id, key, at);
        self.retry_epoch += u64::from(visible);
        visible
    }

    /// Give the cached block `id` one more key of its group, at `at` when the
    /// key lies inside a block of the group. Nothing changes when the block
    /// holds `key`. Returns whether `key` became visible.
    pub(crate) fn cache_alias(
        &mut self,
        id: BlockCopyId,
        key: CacheKey,
        at: Option<KeyInBlock>,
    ) -> bool {
        let visible = self.pool.cache_grouped_alias(id, key, at);
        self.retry_epoch += u64::from(visible);
        visible
    }

    /// See [`VllmBlockPool::grouped_keys_in_block`].
    pub(crate) fn keys_in_block(
        &self,
        group: u16,
        first: SequenceHash,
    ) -> impl DoubleEndedIterator<Item = u32> + '_ {
        self.pool.grouped_keys_in_block(group, first)
    }

    pub(crate) fn release(&mut self, id: BlockCopyId) {
        self.pool.release(id);
        self.retry_epoch += 1;
    }

    /// Whether `id` is a private block, which no key has published yet.
    pub(crate) fn is_private(&self, id: BlockCopyId) -> bool {
        self.pool.is_private(id)
    }

    /// Release in the given order, which is the eviction order of the cached
    /// blocks among them. Pass a request's blocks tail first, as vLLM frees
    /// them.
    pub(crate) fn release_all(&mut self, ids: impl IntoIterator<Item = BlockCopyId>) {
        for id in ids {
            self.release(id);
        }
    }

    pub(crate) fn cancel(&mut self, reservation: BlockReservation) {
        self.pool.cancel(reservation);
    }

    pub(crate) fn active_blocks(&self) -> usize {
        self.pool.num_active()
    }

    pub(crate) fn inactive_cached_blocks(&self) -> usize {
        self.pool.num_inactive()
    }

    pub(crate) fn bytes(&self, blocks: usize) -> u64 {
        blocks as u64 * self.page_size_bytes
    }
}

/// The bytes in use and the active blocks of a grouped pool, the cached
/// blocks that no request holds, and the count and the order-free
/// fingerprint of the visible keys. In byte mode the groups can differ in
/// page size, so the bytes in use can change while the active blocks stay
/// equal. In block mode the bytes in use follow the active blocks. Byte mode
/// caches nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupedPoolState {
    used_bytes: u64,
    active_blocks: usize,
    cached_blocks: usize,
    visible_keys: (usize, u64),
}

/// Byte mode backs `required_bytes`, `fixed_request_bytes`, `allocate` and
/// `release`, which assert the mode in debug builds. Block mode keeps the
/// blocks of every request in `block_pool`. The two modes share
/// `projected_blocks` and `fixed_request_blocks`, the test that one forward
/// fits the whole pool.
pub(crate) enum GroupedMode {
    Bytes,
    Blocks(GroupedBlockPool),
}

pub(crate) struct GroupedKvPool {
    /// Attention and convolution groups. Their blocks follow the request length.
    groups: Vec<FpmCacheGroup>,
    /// What the state groups hold for every positive target, or `None` when
    /// their bytes overflow.
    state: Option<GroupedLease>,
    capacity_bytes: u64,
    used_bytes: u64,
    active_blocks: usize,
    mode: GroupedMode,
    /// The byte-mode count of `retry_epoch`.
    retry_epoch: u64,
}

impl GroupedKvPool {
    /// `groups` must have passed [`FpmCacheGroup::validate`], as
    /// `EngineConfig::validate` checks once. Projections do not repeat it.
    pub(crate) fn new(groups: Vec<FpmCacheGroup>, capacity_bytes: u64) -> Self {
        debug_assert!(groups.iter().all(|group| group.validate().is_ok()));
        let (state_groups, groups): (Vec<_>, Vec<_>) = groups
            .into_iter()
            .partition(|group| group.kind == FpmCacheKind::State);
        Self {
            groups,
            state: GroupedLease::default().charge(&state_groups, 0, 1),
            capacity_bytes,
            used_bytes: 0,
            active_blocks: 0,
            mode: GroupedMode::Bytes,
            retry_epoch: 0,
        }
    }

    /// Block mode for prefix caching. The caller checks that every group has
    /// pages of `page_size_bytes`. Its caller holds every group, state groups
    /// included, in one block lease, and admits and allocates blocks of the
    /// block pool only. The attention groups and the state charge that byte
    /// mode keeps are read only by `projected_blocks` and
    /// `fixed_request_blocks`, which count each state group once.
    pub(crate) fn with_prefix_cache(
        groups: Vec<FpmCacheGroup>,
        capacity_bytes: u64,
        page_size_bytes: u64,
    ) -> Self {
        debug_assert!(
            groups
                .iter()
                .all(|group| group.page_size_bytes == page_size_bytes),
            "block mode needs one page size"
        );
        let pool = VllmBlockPool::new((capacity_bytes / page_size_bytes) as usize);
        Self {
            mode: GroupedMode::Blocks(GroupedBlockPool {
                pool,
                page_size_bytes,
                id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
                retry_epoch: 0,
                #[cfg(test)]
                peak_active_blocks: std::cell::Cell::new(0),
            }),
            ..Self::new(groups, capacity_bytes)
        }
    }

    fn debug_assert_byte_mode(&self) {
        debug_assert!(
            matches!(self.mode, GroupedMode::Bytes),
            "the byte-mode API is not called in block mode"
        );
    }

    /// Pool blocks that the groups hold for a forward over
    /// `computed_before..computed_after`, in either mode.
    pub(crate) fn projected_blocks(
        &self,
        computed_before: usize,
        computed_after: usize,
    ) -> Option<usize> {
        self.project(computed_before, computed_after)
            .map(|lease| lease.blocks())
    }

    /// Pool blocks that the state groups hold for one running request: the
    /// blocks of the state charge, or `usize::MAX` when its bytes overflow.
    pub(crate) fn fixed_request_blocks(&self) -> usize {
        self.state.as_ref().map_or(usize::MAX, GroupedLease::blocks)
    }

    fn project(&self, computed_before: usize, computed_after: usize) -> Option<GroupedLease> {
        if computed_after < computed_before {
            return None;
        }
        let held = if computed_after == 0 {
            GroupedLease::default()
        } else {
            self.state.clone()?
        };
        held.charge(&self.groups, computed_before as u64, computed_after as u64)
    }

    pub(crate) fn required_bytes(
        &self,
        computed_before: usize,
        computed_after: usize,
    ) -> Option<u64> {
        self.debug_assert_byte_mode();
        self.project(computed_before, computed_after)
            .map(|lease| lease.bytes)
    }

    /// Bytes that the state groups hold for one running request, whatever
    /// its context length: the state charge computed when the pool is built,
    /// or `u64::MAX` when those bytes overflow.
    pub(crate) fn fixed_request_bytes(&self) -> u64 {
        self.debug_assert_byte_mode();
        self.state.as_ref().map_or(u64::MAX, |state| state.bytes)
    }

    /// Reserve every group atomically. Expired blocks become reusable in this
    /// transition, but failed attempts leave both the lease and pool unchanged.
    /// `computed_before` is the count that a window starts from: the caller
    /// passes the processed count, `computed - num_in_flight_tokens` (v0.31
    /// `mgr:549-556`, v0.32 `mgr:552-559`, v0.33 `mgr:561-568`). The lease
    /// then holds the window of the step in flight and every token up to
    /// `computed_after`.
    pub(crate) fn allocate(
        &mut self,
        lease: &mut Option<GroupedLease>,
        computed_before: usize,
        computed_after: usize,
    ) -> bool {
        self.debug_assert_byte_mode();
        let Some(projected) = self.project(computed_before, computed_after) else {
            return false;
        };
        let previous_bytes = lease.as_ref().map_or(0, |lease| lease.bytes);
        let other_bytes = self.used_bytes - previous_bytes;
        if projected.bytes > self.capacity_bytes - other_bytes {
            return false;
        }
        self.used_bytes = other_bytes + projected.bytes;
        self.active_blocks = self.active_blocks - lease.as_ref().map_or(0, GroupedLease::blocks)
            + projected.blocks();
        // Window blocks that leave the window free bytes.
        self.retry_epoch += u64::from(projected.bytes < previous_bytes);
        *lease = Some(projected);
        true
    }

    pub(crate) fn release(&mut self, lease: &mut Option<GroupedLease>) {
        self.debug_assert_byte_mode();
        if let Some(lease) = lease.take() {
            self.used_bytes -= lease.bytes;
            self.active_blocks -= lease.blocks();
            self.retry_epoch += 1;
        }
    }

    /// Grows whenever the pool releases a block, a request's held bytes
    /// fall, the pool publishes a key or, in block mode, it evicts the last
    /// copy of a key. These are the pool's changes that can let a waiting
    /// reservation succeed. A pass frees blocks without completing a
    /// request, and an activation publishes blocks that it keeps, so the
    /// active count alone misses both. An eviction can change the hit of a
    /// waiting load and lower the blocks that its tests count, so block mode
    /// adds the generation of the resident keys. In byte mode, a request
    /// whose bytes grow can still free a window page, with no move of this
    /// count. The scheduler also retries a waiting reservation when the
    /// local prefills or the loads in flight need less.
    pub(crate) fn retry_epoch(&self) -> u64 {
        match &self.mode {
            GroupedMode::Bytes => self.retry_epoch,
            GroupedMode::Blocks(blocks) => {
                blocks.retry_epoch + blocks.pool.grouped_key_generation()
            }
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        match &self.mode {
            GroupedMode::Bytes => self.capacity_bytes,
            GroupedMode::Blocks(blocks) => blocks.bytes(blocks.capacity_blocks()),
        }
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        match &self.mode {
            GroupedMode::Bytes => self.used_bytes,
            GroupedMode::Blocks(blocks) => blocks.bytes(blocks.active_blocks()),
        }
    }

    pub(crate) fn active_blocks(&self) -> usize {
        match &self.mode {
            GroupedMode::Bytes => self.active_blocks,
            GroupedMode::Blocks(blocks) => blocks.active_blocks(),
        }
    }

    pub(crate) fn state(&self) -> GroupedPoolState {
        let used_bytes = self.used_bytes();
        match &self.mode {
            GroupedMode::Bytes => GroupedPoolState {
                used_bytes,
                active_blocks: self.active_blocks,
                cached_blocks: 0,
                visible_keys: (0, 0),
            },
            GroupedMode::Blocks(blocks) => GroupedPoolState {
                used_bytes,
                active_blocks: blocks.active_blocks(),
                cached_blocks: blocks.inactive_cached_blocks(),
                visible_keys: blocks.pool.grouped_keys(),
            },
        }
    }

    /// Bytes of cached blocks that no request holds. Byte mode caches nothing.
    pub(crate) fn cached_bytes(&self) -> u64 {
        match &self.mode {
            GroupedMode::Bytes => 0,
            GroupedMode::Blocks(blocks) => blocks.bytes(blocks.inactive_cached_blocks()),
        }
    }

    pub(crate) fn groups(&self) -> &[FpmCacheGroup] {
        &self.groups
    }

    pub(crate) fn block_pool(&self) -> Option<&GroupedBlockPool> {
        match &self.mode {
            GroupedMode::Bytes => None,
            GroupedMode::Blocks(blocks) => Some(blocks),
        }
    }

    pub(crate) fn block_pool_mut(&mut self) -> Option<&mut GroupedBlockPool> {
        match &mut self.mode {
            GroupedMode::Bytes => None,
            GroupedMode::Blocks(blocks) => Some(blocks),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full attention, a ring and a prefix-cacheable state, with equal pages.
    fn three_groups() -> Vec<FpmCacheGroup> {
        [
            ("full", FpmCacheKind::Attention, None, true),
            ("ring", FpmCacheKind::State, Some(1), false),
            ("kda", FpmCacheKind::State, Some(1), true),
        ]
        .into_iter()
        .map(
            |(name, kind, blocks_per_request, prefix_cacheable)| FpmCacheGroup {
                name: name.into(),
                kind,
                num_layers: 1,
                block_size_tokens: 4,
                page_size_bytes: 8,
                sliding_window: None,
                blocks_per_request,
                prefix_cacheable: Some(prefix_cacheable),
                transfer_bytes_per_block: None,
                prefill_checkpoint_blocks: None,
                prefill_checkpoint_alignment: None,
            },
        )
        .collect()
    }

    #[test]
    fn mixed_groups_reserve_atomically_and_reuse_expired_pages() {
        let mut pool = GroupedKvPool::new(
            vec![
                FpmCacheGroup {
                    name: "full".into(),
                    kind: FpmCacheKind::Attention,
                    num_layers: 1,
                    block_size_tokens: 4,
                    page_size_bytes: 8,
                    sliding_window: None,
                    blocks_per_request: None,
                    prefix_cacheable: None,
                    transfer_bytes_per_block: None,
                    prefill_checkpoint_blocks: None,
                    prefill_checkpoint_alignment: None,
                },
                FpmCacheGroup {
                    name: "window".into(),
                    kind: FpmCacheKind::Attention,
                    num_layers: 1,
                    block_size_tokens: 2,
                    page_size_bytes: 6,
                    sliding_window: Some(3),
                    blocks_per_request: None,
                    prefix_cacheable: None,
                    transfer_bytes_per_block: None,
                    prefill_checkpoint_blocks: None,
                    prefill_checkpoint_alignment: None,
                },
            ],
            42,
        );
        assert!(pool.block_pool().is_none());
        assert_eq!(pool.capacity_bytes(), 42);
        assert_eq!(pool.fixed_request_bytes(), 0);
        let mut first = None;
        let mut second = None;
        assert!(pool.allocate(&mut first, 0, 4)); // 8 + 2 * 6 = 20
        assert!(pool.allocate(&mut second, 0, 2)); // 8 + 6 = 14
        assert!(!pool.allocate(&mut first, 4, 8)); // 16 + 3 * 6 = 34, plus 14
        assert_eq!(first.as_ref().unwrap().blocks(), 3); // 0..1 and 0..2
        assert_eq!(pool.used_bytes(), 34);
        pool.release(&mut second);
        assert!(pool.allocate(&mut first, 4, 8));
        assert_eq!(first.as_ref().unwrap().blocks(), 5); // 0..2 and 1..4
        assert_eq!(pool.used_bytes(), 34);
        assert!(pool.allocate(&mut first, 8, 9));
        assert_eq!(first.as_ref().unwrap().blocks(), 5); // 0..3 and 3..5
        assert_eq!(pool.used_bytes(), 36);
        pool.release(&mut first);
        pool.release(&mut first);
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.active_blocks(), 0);
        assert_eq!(pool.cached_bytes(), 0);
    }

    #[test]
    fn state_group_reserves_constant_bytes_and_releases_them() {
        let mut pool = GroupedKvPool::new(
            vec![
                FpmCacheGroup {
                    name: "full".into(),
                    kind: FpmCacheKind::Attention,
                    num_layers: 1,
                    block_size_tokens: 4,
                    page_size_bytes: 8,
                    sliding_window: None,
                    blocks_per_request: None,
                    prefix_cacheable: None,
                    transfer_bytes_per_block: None,
                    prefill_checkpoint_blocks: None,
                    prefill_checkpoint_alignment: None,
                },
                FpmCacheGroup {
                    name: "kda".into(),
                    kind: FpmCacheKind::State,
                    num_layers: 1,
                    block_size_tokens: 4,
                    page_size_bytes: 10,
                    sliding_window: None,
                    blocks_per_request: Some(2),
                    prefix_cacheable: None,
                    transfer_bytes_per_block: None,
                    prefill_checkpoint_blocks: None,
                    prefill_checkpoint_alignment: None,
                },
            ],
            100,
        );
        assert!(pool.block_pool().is_none());
        assert_eq!(pool.capacity_bytes(), 100);
        // Only the state group counts: 2 blocks of 10 bytes.
        assert_eq!(pool.fixed_request_bytes(), 20);
        let mut lease = None;
        assert!(pool.allocate(&mut lease, 0, 4)); // 8 + 2 * 10 = 28
        assert_eq!(lease.as_ref().unwrap().blocks(), 3); // 0..1 and the state's 0..2
        assert_eq!(pool.used_bytes(), 28);
        assert!(pool.allocate(&mut lease, 4, 8)); // 2 * 8 + 2 * 10 = 36
        assert_eq!(lease.as_ref().unwrap().blocks(), 4); // 0..2 and the state's 0..2
        assert_eq!(pool.used_bytes(), 36);
        assert_eq!(pool.active_blocks(), 4);
        pool.release(&mut lease);
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.active_blocks(), 0);
        assert_eq!(pool.cached_bytes(), 0);
    }

    #[test]
    fn byte_pool_holds_the_window_of_the_step_in_flight() {
        // The caller passes the processed count as the window start. With
        // the step before in flight, that is its computed count before it
        // (v0.33 mgr:561-568). Window 4, block 4, page 8: the window keeps
        // the blocks from `(processed - 3) / 4` (v0.33 single:1148-1183,
        // :682-720, where `extra_retained_tokens` is 0 without speculative
        // decoding) to `cdiv(target, 4)`.
        let mut pool = GroupedKvPool::new(
            vec![FpmCacheGroup {
                name: "window".into(),
                kind: FpmCacheKind::Attention,
                num_layers: 1,
                block_size_tokens: 4,
                page_size_bytes: 8,
                sliding_window: Some(4),
                blocks_per_request: None,
                prefix_cacheable: None,
                transfer_bytes_per_block: None,
                prefill_checkpoint_blocks: None,
                prefill_checkpoint_alignment: None,
            }],
            48,
        );
        let mut lease = None;
        assert!(pool.allocate(&mut lease, 0, 8)); // blocks 0..2
        assert_eq!(pool.used_bytes(), 16);
        assert!(pool.allocate(&mut lease, 0, 16)); // step 1 in flight: 0..4
        assert_eq!(pool.used_bytes(), 32);
        assert!(pool.allocate(&mut lease, 8, 24)); // 5 / 4 = 1: 1..6
        assert_eq!(pool.used_bytes(), 40);
        assert!(pool.allocate(&mut lease, 16, 32)); // 13 / 4 = 3: 3..8
        assert_eq!(pool.used_bytes(), 40);
        assert!(pool.allocate(&mut lease, 24, 33)); // 21 / 4 = 5: 5..9
        assert_eq!(pool.used_bytes(), 32);
        assert!(pool.allocate(&mut lease, 32, 34)); // 29 / 4 = 7: 7..9
        assert_eq!(pool.used_bytes(), 16);
    }

    #[test]
    fn block_mode_pins_hits_and_evicts_cached_blocks_in_lru_order() {
        let (h0, h1) = (10, 11);
        let hits = [
            CacheKey::Grouped { group: 0, hash: h0 },
            CacheKey::Grouped { group: 0, hash: h1 },
        ];
        let released_hits = || {
            let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 48, 8);
            assert_eq!(pool.capacity_bytes(), 48);
            let blocks = pool.block_pool_mut().unwrap();
            assert_eq!(blocks.capacity_blocks(), 6);
            // Request A publishes two blocks and frees them tail first, as
            // vLLM frees a request's blocks.
            let mut reservation = blocks.reserve(&[], 2).unwrap();
            let a = [
                blocks.take_private(&mut reservation),
                blocks.take_private(&mut reservation),
            ];
            assert!(blocks.cache(a[0], hits[0]));
            assert!(blocks.cache(a[1], hits[1]));
            blocks.release_all(a.into_iter().rev());
            assert_eq!(blocks.inactive_cached_blocks(), 2);
            assert!(!blocks.key_hit(0, h1).unwrap().is_active);
            assert_eq!(pool.cached_bytes(), 16);
            assert_eq!(pool.used_bytes(), 0);

            // Request B hits both. Leaving the LRU, they cost capacity again.
            let blocks = pool.block_pool_mut().unwrap();
            let mut reservation = blocks.reserve(&hits, 0).unwrap();
            let b = blocks.activate(&mut reservation);
            assert_eq!(b, a);
            assert_eq!(blocks.active_blocks(), 2);
            assert_eq!(blocks.inactive_cached_blocks(), 0);
            // Another hit on the same blocks, now active, costs nothing.
            let shared = blocks.reserve(&hits, 0).unwrap();
            assert_eq!(blocks.active_blocks(), 2);
            blocks.cancel(shared);
            assert_eq!(pool.used_bytes(), 16);
            assert_eq!(pool.active_blocks(), 2);
            assert_eq!(pool.cached_bytes(), 0);
            // B frees tail first too, so h1 is the oldest cached block.
            pool.block_pool_mut()
                .unwrap()
                .release_all(b.into_iter().rev());
            pool
        };

        // Six fresh blocks take both cached ones.
        let mut pool = released_hits();
        let blocks = pool.block_pool_mut().unwrap();
        let reservation = blocks.reserve(&[], 6).unwrap();
        assert!(blocks.key_hit(0, h0).is_none());
        assert!(blocks.key_hit(0, h1).is_none());
        blocks.cancel(reservation);
        assert_eq!(blocks.active_blocks(), 0);
        assert_eq!(blocks.inactive_cached_blocks(), 0);

        // Four blocks are free, so a fifth evicts only the oldest cached
        // block, h1.
        let mut pool = released_hits();
        let blocks = pool.block_pool_mut().unwrap();
        assert!(blocks.reserve(&[], 5).is_some());
        assert!(blocks.key_hit(0, h1).is_none());
        assert!(!blocks.key_hit(0, h0).unwrap().is_active);
    }

    #[test]
    fn block_mode_duplicate_key_survives_primary_eviction_only() {
        let hash = 7;
        let key = CacheKey::Grouped { group: 0, hash };
        let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 24, 8);
        let blocks = pool.block_pool_mut().unwrap();
        let mut reservation = blocks.reserve(&[], 3).unwrap();
        let primary = blocks.take_private(&mut reservation);
        let duplicate = blocks.take_private(&mut reservation);
        let other_group = blocks.take_private(&mut reservation);
        assert!(blocks.cache(primary, key));
        assert!(!blocks.cache(duplicate, key));
        assert!(blocks.cache(other_group, CacheKey::Grouped { group: 2, hash }));
        blocks.cancel(reservation);
        // Primary first in the LRU, so pressure must promote the duplicate.
        blocks.release_all([primary, duplicate]);
        let first = blocks.reserve(&[], 1).unwrap();
        assert!(!blocks.key_hit(0, hash).unwrap().is_active);
        assert!(blocks.key_hit(2, hash).unwrap().is_active);
        assert_eq!(blocks.inactive_cached_blocks(), 1);
        let mut hit = blocks.reserve(&[key], 0).unwrap();
        let held = blocks.activate(&mut hit);
        assert_eq!(held, vec![duplicate]);
        blocks.cancel(hit);
        blocks.release(held[0]);
        let last = blocks.reserve(&[], 1).unwrap();
        assert!(blocks.key_hit(0, hash).is_none());
        assert!(blocks.key_hit(2, hash).unwrap().is_active);
        blocks.cancel(last);
        blocks.cancel(first);
        blocks.release(other_group);
    }

    #[test]
    fn block_mode_keys_are_separate_per_group() {
        let hash = 7;
        let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 48, 8);
        let blocks = pool.block_pool_mut().unwrap();
        let mut reservation = blocks.reserve(&[], 2).unwrap();
        let full = blocks.take_private(&mut reservation);
        let state = blocks.take_private(&mut reservation);
        assert!(blocks.cache(full, CacheKey::Grouped { group: 0, hash }));
        // The same hash in another group is a new key, not a second copy.
        assert!(blocks.cache(state, CacheKey::Grouped { group: 2, hash }));
        assert!(blocks.key_hit(0, hash).is_some());
        assert!(blocks.key_hit(2, hash).is_some());
        assert!(blocks.key_hit(1, hash).is_none());
        blocks.release(full);
        // Four blocks are free, so a fifth evicts the one inactive block.
        assert!(blocks.reserve(&[], 5).is_some());
        assert!(blocks.key_hit(0, hash).is_none());
        assert!(blocks.key_hit(2, hash).unwrap().is_active);
    }

    #[test]
    fn block_mode_state_follows_the_set_of_visible_keys() {
        let key = |hash| CacheKey::Grouped { group: 0, hash };
        // Publish `hashes` on fresh blocks, in order, and free them.
        let publish = |pool: &mut GroupedKvPool, hashes: &[u64]| {
            let blocks = pool.block_pool_mut().unwrap();
            let mut reservation = blocks.reserve(&[], hashes.len()).unwrap();
            for &hash in hashes {
                let id = blocks.take_private(&mut reservation);
                assert!(blocks.cache(id, key(hash)));
                blocks.release(id);
            }
        };
        let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 32, 8);
        publish(&mut pool, &[10, 11]);
        let first = pool.state();
        // The order in which the keys became visible does not count.
        let mut reversed = GroupedKvPool::with_prefix_cache(three_groups(), 32, 8);
        publish(&mut reversed, &[11, 10]);
        assert_eq!(reversed.state(), first);
        // Three fresh blocks evict the older cached block, key 10. Publishing
        // key 10 again restores the same set, the same counts and the same
        // state.
        let blocks = pool.block_pool_mut().unwrap();
        let reservation = blocks.reserve(&[], 3).unwrap();
        assert!(blocks.key_hit(0, 10).is_none());
        blocks.cancel(reservation);
        assert_ne!(pool.state(), first);
        publish(&mut pool, &[10]);
        assert_eq!(pool.state(), first);
        // Another key in the place of key 11 keeps both counts but not the
        // state.
        let mut other = GroupedKvPool::with_prefix_cache(three_groups(), 32, 8);
        publish(&mut other, &[10, 12]);
        assert_ne!(other.state(), first);
    }

    #[test]
    fn block_mode_publish_inside_a_block_moves_the_retry_epoch_as_any_publish() {
        // A waiting load is tried again when a key becomes visible, also a key
        // inside a block, such as a state at the prompt's partial tail.
        let key = |hash| CacheKey::Grouped { group: 0, hash };
        let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 32, 8);
        let blocks = pool.block_pool_mut().unwrap();
        let mut reservation = blocks.reserve(&[], 2).unwrap();
        let (at_end, inside) = (
            blocks.take_private(&mut reservation),
            blocks.take_private(&mut reservation),
        );
        blocks.cancel(reservation);
        let start = pool.retry_epoch();
        assert!(pool.block_pool_mut().unwrap().cache(at_end, key(10)));
        let after_end = pool.retry_epoch();
        let at = Some(KeyInBlock {
            first: 7,
            offset: 4,
        });
        assert!(pool.block_pool_mut().unwrap().cache_at(inside, key(11), at));
        assert!(after_end > start);
        assert_eq!(pool.retry_epoch() - after_end, after_end - start);
    }

    #[test]
    fn block_mode_key_added_to_a_cached_block_moves_the_retry_epoch_as_any_publish() {
        // A waiting load is tried again when a key becomes visible, also a
        // key that a cached block gets beside its first key.
        let key = |hash| CacheKey::Grouped { group: 0, hash };
        let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 32, 8);
        let blocks = pool.block_pool_mut().unwrap();
        let mut reservation = blocks.reserve(&[], 1).unwrap();
        let id = blocks.take_private(&mut reservation);
        blocks.cancel(reservation);
        let start = pool.retry_epoch();
        assert!(pool.block_pool_mut().unwrap().cache(id, key(10)));
        let after_first = pool.retry_epoch();
        let at = Some(KeyInBlock {
            first: 7,
            offset: 4,
        });
        assert!(pool.block_pool_mut().unwrap().cache_alias(id, key(11), at));
        assert!(after_first > start);
        // Derived, not run: a key that becomes visible moves the count by the
        // same amount, whichever way it was published.
        assert_eq!(pool.retry_epoch() - after_first, after_first - start);
        // Derived, not run: a key that the block holds changes nothing.
        let after_alias = pool.retry_epoch();
        assert!(!pool.block_pool_mut().unwrap().cache_alias(id, key(11), at));
        assert!(
            !pool
                .block_pool_mut()
                .unwrap()
                .cache_alias(id, key(10), None)
        );
        assert_eq!(pool.retry_epoch(), after_alias);
        assert!(pool.block_pool().unwrap().key_hit(0, 11).is_some());
    }

    #[test]
    fn block_lease_counts_the_live_copies_of_every_group() {
        let mut pool = GroupedKvPool::with_prefix_cache(three_groups(), 64, 8);
        let blocks = pool.block_pool_mut().unwrap();
        let mut reservation = blocks.reserve(&[], 8).unwrap();
        let ids: Vec<_> = (0..8)
            .map(|_| blocks.take_private(&mut reservation))
            .collect();
        let lease = GroupedLease {
            block_lease: Some(GroupedBlockLease {
                groups: vec![
                    // A window group whose block 0 has expired.
                    GroupLease::Attention {
                        copies: vec![None, Some(ids[0]), Some(ids[1])],
                        first_live: 1,
                        finalized: 0,
                    },
                    GroupLease::Ring { copy: Some(ids[2]) },
                    GroupLease::State(GroupStateLease {
                        running: Some(ids[3]),
                        hit: Some((ids[4], 1)),
                        previous: vec![(ids[5], PreviousRelease::AfterStep(2))],
                        copy: Some((ids[6], 8)),
                        checkpoint: Some((ids[7], 8)),
                        published: None,
                    }),
                ],
                allocated: true,
                quiet_until: 0,
            }),
            ..GroupedLease::default()
        };
        assert_eq!(lease.blocks(), 8);
    }

    #[test]
    fn state_groups_charge_their_sum_only_for_a_positive_target() {
        let state = |name: &str, page_size_bytes, blocks| FpmCacheGroup {
            name: name.into(),
            kind: FpmCacheKind::State,
            num_layers: 1,
            block_size_tokens: 4,
            page_size_bytes,
            sliding_window: None,
            blocks_per_request: Some(blocks),
            prefix_cacheable: None,
            transfer_bytes_per_block: None,
            prefill_checkpoint_blocks: None,
            prefill_checkpoint_alignment: None,
        };
        let mut pool = GroupedKvPool::new(vec![state("kda", 10, 2), state("tail", 6, 1)], 100);
        assert_eq!(pool.required_bytes(0, 0), Some(0));
        assert_eq!(pool.required_bytes(0, 1), Some(26)); // 2 * 10 + 6
        assert_eq!(pool.required_bytes(7, 9), Some(26));
        let mut lease = None;
        assert!(pool.allocate(&mut lease, 0, 0));
        assert_eq!(pool.active_blocks(), 0);
        assert!(pool.allocate(&mut lease, 0, 1));
        assert_eq!(pool.active_blocks(), 3);

        // State bytes beyond u64 fail every positive target: one group's
        // product (2^64), or a sum of products that each fit (2^63 + 2^63).
        for groups in [
            vec![state("huge", 1 << 53, 1 << 11)],
            vec![state("a", 1 << 53, 1 << 10), state("b", 1 << 53, 1 << 10)],
        ] {
            let mut pool = GroupedKvPool::new(groups, 100);
            assert_eq!(pool.required_bytes(0, 0), Some(0));
            assert_eq!(pool.required_bytes(0, 1), None);
            assert!(!pool.allocate(&mut None, 0, 1));
        }
    }

    #[test]
    fn block_mode_forward_test_counts_each_state_group_once() {
        // The full-attention group, then the ring and the KDA state with one
        // block each. A forward to 10 tokens holds 3 attention blocks and 2
        // state blocks in either mode.
        let bytes = GroupedKvPool::new(three_groups(), 12 * 8);
        let blocks = GroupedKvPool::with_prefix_cache(three_groups(), 12 * 8, 8);
        for pool in [&bytes, &blocks] {
            assert_eq!(pool.projected_blocks(0, 0), Some(0));
            assert_eq!(pool.projected_blocks(0, 10), Some(5));
            assert_eq!(pool.projected_blocks(0, 4), Some(3));
            assert_eq!(pool.fixed_request_blocks(), 2);
        }
        assert_eq!(bytes.required_bytes(0, 10), Some(5 * 8));
        assert_eq!(bytes.fixed_request_bytes(), 2 * 8);
        // The block pool is untouched by the test.
        assert_eq!(blocks.active_blocks(), 0);
        assert_eq!(blocks.block_pool().unwrap().capacity_blocks(), 12);
    }
}
