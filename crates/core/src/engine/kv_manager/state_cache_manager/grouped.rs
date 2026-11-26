// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Grouped MambaManager rules, modified from the vLLM v0.31.0, v0.32.0 and
//! v0.33.0 revisions of single_type_kv_cache_manager.py, kv_cache_manager.py
//! and scheduler.py listed in THIRD_PARTY_NOTICES.md.
//!
//! - <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/single_type_kv_cache_manager.py>
//! - <https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py>
//! - <https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py>

use super::{StateAlignRules, StateCacheManager, StateLease, StatePlan, StateWrite};
use crate::engine::cache::vllm_block_pool::{
    BlockCopyId, BlockReservation, CacheKey, KeyInBlock, VllmBlockPool,
};
use crate::engine::common::hashing::SequenceHash;
use crate::engine::kv_manager::{BlockRequestLease, grouped::GroupedBlockPool};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupStateLease {
    pub(in crate::engine::kv_manager) blocks: StateLease<Option<BlockCopyId>>,
    /// An aligned hit retains its reference until compute advances beyond
    /// its block. A partial hit transfers the reference to the pass.
    pub(in crate::engine::kv_manager) hit: Option<(BlockCopyId, usize)>,
    /// Separate destination when a partial-hit copy is below the new working block.
    pub(in crate::engine::kv_manager) copy: Option<BlockCopyId>,
    /// Valid while the checkpoint slot is occupied.
    pub(in crate::engine::kv_manager) checkpoint_position: usize,
    /// Replacing a published partial block hides its key for this pass.
    pub(in crate::engine::kv_manager) partial_hash: Option<SequenceHash>,
}

impl GroupStateLease {
    pub(in crate::engine::kv_manager) fn live_copies(&self) -> usize {
        usize::from(self.blocks.working.is_some())
            + usize::from(self.blocks.previous.is_some())
            + usize::from(self.blocks.checkpoint.is_some())
            + usize::from(self.hit.is_some())
            + usize::from(self.copy.is_some())
    }

    pub(in crate::engine::kv_manager) fn quiet_until(
        &self,
        computed: usize,
        block: usize,
    ) -> usize {
        if self.live_copies() == 1 && self.blocks.working.is_some() && !self.blocks.working_cached {
            computed.div_ceil(block) * block + 1
        } else {
            0
        }
    }

    #[cfg(test)]
    pub(in crate::engine::kv_manager) fn running(&self) -> Option<BlockCopyId> {
        self.blocks.working
    }

    #[cfg(test)]
    pub(in crate::engine::kv_manager) fn previous(&self) -> Option<(BlockCopyId, u64)> {
        self.blocks
            .previous
            .map(|id| (id, self.blocks.previous_step))
    }

    pub(in crate::engine::kv_manager) fn checkpoint(&self) -> Option<(BlockCopyId, usize)> {
        self.blocks
            .checkpoint
            .map(|id| (id, self.checkpoint_position))
    }

    pub(in crate::engine::kv_manager) fn release(&mut self, releases: &mut Vec<BlockCopyId>) {
        // A grouped manager frees in block-table order (tail first).
        releases.extend(self.blocks.working.take());
        releases.extend(self.blocks.checkpoint.take());
        releases.extend(self.blocks.previous.take());
        releases.extend(self.hit.take().map(|(id, _)| id));
        releases.extend(self.copy.take());
        self.blocks.working_cached = false;
        self.partial_hash = None;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StateGroup {
    pub(in crate::engine::kv_manager) id: usize,
    pub(in crate::engine::kv_manager) block: usize,
    pub(in crate::engine::kv_manager) align: StateAlignRules,
}

impl StateGroup {
    pub(in crate::engine::kv_manager) fn startup_blocks(checkpoint: bool) -> usize {
        2 + usize::from(checkpoint)
    }

    pub(in crate::engine::kv_manager) fn first_allocation_blocks(
        self,
        pool: &VllmBlockPool,
        hit: Option<CacheKey>,
        computed: usize,
        target: usize,
        prefill_end: usize,
    ) -> usize {
        let inactive = hit.is_some_and(|key| pool.key_hit(key).is_some_and(|hit| !hit.is_active));
        usize::from(inactive)
            + self
                .plan_chunk(None, hit.is_some(), computed, target, prefill_end)
                .fresh_states()
    }

    pub(in crate::engine::kv_manager) fn pin(
        self,
        lease: &mut GroupStateLease,
        id: BlockCopyId,
        tokens: usize,
    ) {
        lease.hit = Some((id, tokens.div_ceil(self.block) - 1));
    }

    /// Release in main's aggregate eviction order: internal checkpoint,
    /// separate copy destination, completed boundary hit, previous working block.
    pub(in crate::engine::kv_manager) fn release_before_allocation(
        self,
        lease: &mut GroupStateLease,
        first: bool,
        computed: usize,
        step: u64,
        releases: &mut Vec<BlockCopyId>,
    ) {
        releases.extend(lease.blocks.checkpoint.take());
        if !first {
            releases.extend(lease.copy.take());
        }
        if let Some((id, index)) = lease.hit
            && computed.div_ceil(self.block).saturating_sub(1) > index
        {
            releases.push(id);
            lease.hit = None;
        }
        lease.blocks.release_previous(step, |id| releases.push(id));
    }

    pub(in crate::engine::kv_manager) fn plan(
        self,
        lease: &GroupStateLease,
        first: bool,
        computed: usize,
        target: usize,
        prefill_end: usize,
    ) -> StatePlan {
        self.plan_chunk(
            Some(lease),
            first && lease.hit.is_some(),
            computed,
            target,
            prefill_end,
        )
    }

    fn plan_chunk(
        self,
        lease: Option<&GroupStateLease>,
        first_hit: bool,
        computed: usize,
        target: usize,
        prefill_end: usize,
    ) -> StatePlan {
        let crossing = target.div_ceil(self.block) > computed.div_ceil(self.block);
        let partial_hit = first_hit && !computed.is_multiple_of(self.block);
        let write = match lease {
            Some(lease) => lease
                .blocks
                .write(computed > 0 && crossing, target > computed),
            // Before pinning, a covered aligned block costs only the hit.
            None if crossing || partial_hit => StateWrite::Start,
            None => StateWrite::Keep,
        };
        StatePlan {
            write,
            copy_pinned_hit: partial_hit,
            separate_copy: partial_hit && crossing,
            checkpoint: self.align.checkpoint(computed, target, prefill_end, target),
        }
    }
}

impl StateCacheManager {
    pub(in crate::engine::kv_manager) fn for_groups(
        blocks: impl Iterator<Item = Option<usize>>,
        align: StateAlignRules,
    ) -> Self {
        let groups = blocks
            .enumerate()
            .map(|(id, block)| block.map(|block| StateGroup { id, block, align }))
            .collect();
        Self {
            groups,
            ..Self::new(1, false)
        }
    }

    pub(in crate::engine::kv_manager) fn group(&self, id: usize) -> StateGroup {
        self.groups[id].expect("cacheable state group")
    }

    /// A state spec hits at the last position visible in every member. The
    /// pool's index skips empty positions inside a physical block.
    pub(in crate::engine::kv_manager) fn grouped_hit(
        &self,
        pool: &GroupedBlockPool,
        request: &BlockRequestLease,
        group: StateGroup,
        groups: &[usize],
        max: usize,
    ) -> usize {
        let block = group.block;
        let alignment = group.align.alignment;
        let hits = |pos| {
            request.prefix_hash_at(pos).is_some_and(|hash| {
                groups.iter().all(|&g| {
                    let key = CacheKey::Grouped {
                        group: g as u16,
                        hash,
                    };
                    self.visible(key) && pool.pool.key_hit(key).is_some()
                })
            })
        };
        let mut end = max / alignment * alignment;
        while end > 0 {
            if alignment < block && !end.is_multiple_of(block) {
                let floor = end / block * block;
                if let Some(found) = pool.hit_in_block(
                    group.id,
                    request.prefix_hash_at(floor + group.align.hash),
                    floor,
                    end,
                    alignment,
                    hits,
                ) {
                    return found;
                }
                end = floor;
            } else if hits(end) {
                return end;
            } else {
                end -= alignment;
            }
        }
        0
    }

    pub(in crate::engine::kv_manager) fn apply_group(
        &mut self,
        pool: &mut VllmBlockPool,
        group: usize,
        lease: &mut GroupStateLease,
        plan: StatePlan,
        step: u64,
        reservation: &mut BlockReservation,
    ) {
        if plan.copy_pinned_hit {
            let (id, _) = lease.hit.take().expect("pinned partial state");
            self.retain_copy(group, id);
        }
        if plan.write != StateWrite::Keep {
            if let Some(hash) = lease.partial_hash.take() {
                self.mark_published(CacheKey::Grouped {
                    group: group as u16,
                    hash,
                });
            }
            debug_assert!(
                lease.blocks.previous.is_none(),
                "one state crossing per pass"
            );
            lease.blocks.rotate(pool, reservation, 1, step);
        }
        if plan.separate_copy {
            lease.copy = Some(pool.allocate_private(reservation));
        }
        if let Some(pos) = plan.checkpoint {
            lease.blocks.allocate_checkpoint(pool, reservation, 1);
            lease.checkpoint_position = pos;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::engine::kv_manager) fn finalize_group(
        &mut self,
        pool: &mut VllmBlockPool,
        group: StateGroup,
        lease: &mut GroupStateLease,
        prompt: usize,
        computed: usize,
        junction: Option<usize>,
        step: u64,
        key_at: impl Fn(usize) -> Option<(CacheKey, Option<KeyInBlock>)>,
    ) {
        if group.align.retains(prompt, computed, group.block, junction)
            && lease.blocks.working.is_some()
            && !lease.blocks.working_cached
            && let Some((key, at)) = key_at(computed)
        {
            self.publish_destination(pool, &mut lease.blocks, std::iter::once(key), at);
            if !computed.is_multiple_of(group.block) {
                lease.partial_hash = Some(key.prefix_hash());
            }
        }
        if let Some((id, pos)) = lease.checkpoint()
            && (pos == group.align.checkpoint_position(prompt) || junction == Some(pos))
            && pool.is_private(id)
            && let Some((key, at)) = key_at(pos)
        {
            self.publish(
                pool,
                lease.blocks.checkpoint.as_slice(),
                std::iter::once(key),
                at,
            );
        }
        lease.blocks.release_previous(step, |id| pool.release(id));
    }
}
