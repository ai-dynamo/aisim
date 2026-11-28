// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Modified vLLM align-mode scheduling and retention. See THIRD_PARTY_NOTICES.md
//! for the pinned scheduler.py, kv_cache_interface.py and MambaManager revisions.
//!
//! - <https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/core/sched/scheduler.py>
//! - <https://github.com/vllm-project/vllm/blob/43b4aaea3e40e20ef53ce2bfecfa1b72060bf2b5/vllm/v1/kv_cache_interface.py>
//! - <https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py>
//! - <https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py>
//! - <https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/kv_cache_interface.py>

use super::super::grouped_prefix::VllmRuleVersion;

#[derive(Clone, Copy)]
enum Layout {
    StateCache { drop_eagle_block: bool },
    Grouped(VllmRuleVersion),
}

/// The scheduler splits at the smallest cacheable block. Checkpoints use the
/// reconciled hit alignment, which can be coarser than the prefix hash.
#[derive(Clone, Copy)]
pub(crate) struct StateAlignRules {
    layout: Layout,
    block: usize,
    pub(super) hash: usize,
    pub(super) alignment: usize,
    checkpoint_alignment: Option<usize>,
}

impl StateAlignRules {
    pub(crate) fn state_cache(
        block: usize,
        hash: usize,
        checkpoint: bool,
        drop_eagle_block: bool,
    ) -> Self {
        Self {
            layout: Layout::StateCache { drop_eagle_block },
            block,
            hash,
            alignment: hash,
            checkpoint_alignment: checkpoint.then_some(1),
        }
    }

    pub(crate) fn grouped(
        version: VllmRuleVersion,
        block: usize,
        hash: usize,
        alignment: usize,
        checkpoint_alignment: Option<usize>,
    ) -> Self {
        Self {
            layout: Layout::Grouped(version),
            block,
            hash,
            alignment,
            checkpoint_alignment,
        }
    }

    fn drop_eagle_block(self) -> bool {
        matches!(
            self.layout,
            Layout::StateCache {
                drop_eagle_block: true
            }
        )
    }

    pub(crate) fn last_cacheable_stop(self, tokens: usize) -> usize {
        let block = self.block;
        match self.layout {
            Layout::Grouped(VllmRuleVersion::V0_31_0) => tokens / block * block,
            Layout::StateCache {
                drop_eagle_block: true,
            } => (tokens / block * block).saturating_sub(block),
            _ => tokens.saturating_sub(1) / block * block,
        }
    }

    pub(crate) fn partial_tail(self, prompt: usize) -> usize {
        let tokens = match self.layout {
            Layout::Grouped(VllmRuleVersion::V0_32_0 | VllmRuleVersion::V0_33_0) => {
                prompt.saturating_sub(1)
            }
            _ => prompt,
        };
        let tail = tokens / self.hash * self.hash;
        if self.drop_eagle_block() {
            tail.saturating_sub(self.hash)
        } else {
            tail
        }
    }

    pub(crate) fn block_floor(self, tokens: usize, block: usize) -> usize {
        tokens / self.alignment * self.alignment / block * block
    }

    pub(crate) fn replay_boundary(self, prompt: usize, block: usize) -> usize {
        let replay = self.block_floor(prompt.saturating_sub(1), block);
        if self.drop_eagle_block() {
            replay.saturating_sub(block)
        } else {
            replay
        }
    }

    /// Default retention, using the state group's own block size. A grouped
    /// junction is retained at its block floor. The state_cache caller supplies
    /// a junction only when its junction checkpoint option is enabled.
    pub(crate) fn retains(
        self,
        prompt: usize,
        tokens: usize,
        block: usize,
        junction: Option<usize>,
    ) -> bool {
        let tail = self.partial_tail(prompt);
        tokens == self.replay_boundary(prompt, block)
            || (matches!(self.layout, Layout::StateCache { .. })
                && tokens == self.last_cacheable_stop(prompt))
            || (self.alignment == self.hash
                && self.hash < block
                && !tail.is_multiple_of(block)
                && tokens == tail)
            || junction.is_some_and(|pos| {
                tokens
                    == match self.layout {
                        Layout::StateCache { .. } => pos,
                        Layout::Grouped(_) => self.block_floor(pos, block),
                    }
            })
    }

    pub(crate) fn checkpoint_position(self, end: usize) -> usize {
        let pos = end.saturating_sub(1) / self.alignment * self.alignment;
        if self.drop_eagle_block() {
            pos.saturating_sub(self.hash)
        } else {
            pos
        }
    }

    pub(crate) fn checkpoint_valid(self, start: usize, end: usize, pos: usize) -> bool {
        self.checkpoint_alignment.is_some_and(|alignment| {
            start.is_multiple_of(self.hash)
                && end
                    .div_ceil(self.block)
                    .saturating_sub(start.div_ceil(self.block))
                    > 1
                && pos.saturating_sub(start) >= self.hash
                && pos < end
                && (pos - start).is_multiple_of(alignment)
        })
    }

    /// The scheduler tests the prompt end to choose a split. Allocation
    /// reserves the checkpoint at the end of the resulting chunk instead.
    pub(crate) fn checkpoint(
        self,
        start: usize,
        end: usize,
        prefill_end: usize,
        position_tokens: usize,
    ) -> Option<usize> {
        if end < prefill_end {
            return None;
        }
        let pos = self.checkpoint_position(position_tokens);
        self.checkpoint_valid(start, end, pos).then_some(pos)
    }

    pub(crate) fn split(
        self,
        start: usize,
        prompt: usize,
        known: usize,
        desired: usize,
        junction: Option<usize>,
        budget: usize,
    ) -> (usize, Option<usize>) {
        let prefill_end = prompt.max(known.saturating_sub(1));
        if start >= prefill_end {
            return (desired, None);
        }
        let (desired, stop_tokens) = match self.layout {
            Layout::StateCache { .. } => (desired, prompt),
            Layout::Grouped(_) => (desired.min(known.saturating_sub(start)), known),
        };
        let mut end = start + desired;
        let exports = self
            .checkpoint(start, end, prefill_end, prefill_end)
            .is_some();
        if end < prefill_end {
            let aligned = end / self.block * self.block;
            if aligned > start || self.block <= budget {
                end = aligned;
            }
        }
        let last = self.last_cacheable_stop(stop_tokens);
        let tail = self.partial_tail(prompt);
        let resend = matches!(self.layout, Layout::StateCache { .. })
            .then(|| self.replay_boundary(prompt, self.block));
        let stops = [
            start
                .checked_add(self.block - start % self.block)
                .filter(|_| !start.is_multiple_of(self.block)),
            Some(last),
            resend,
            (self.alignment == self.hash && self.hash < self.block && last < tail && tail < prompt)
                .then_some(tail),
        ];
        let junction = junction
            .filter(|&pos| start < pos && pos < end)
            .map(|pos| start + (pos - start) / self.block * self.block);
        for stop in stops
            .into_iter()
            .filter(|_| !exports)
            .chain([junction])
            .flatten()
        {
            if start < stop && stop < end {
                end = stop;
            }
        }
        (
            end.saturating_sub(start),
            self.checkpoint(start, end, prefill_end, end),
        )
    }
}
