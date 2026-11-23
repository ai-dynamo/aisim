// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use uuid::Uuid;

use crate::engine::common::sequence::{PromptTokens, RequestSequence};
use crate::engine::kv_manager::{BlockRequestLease, PrefixHashes};

/// Request-local token state and its native-G1 physical lease.
pub(crate) struct RequestKvState {
    pub(super) sequence: RequestSequence,
    pub(super) lease: BlockRequestLease,
}

impl RequestKvState {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn native(
        owner: Uuid,
        tokens: Vec<u32>,
        max_output_tokens: usize,
        output_capacity_hint: usize,
        block_size: usize,
        enable_prefix_caching: bool,
        retain_local_hashes: bool,
        emit_token_ids: bool,
        planned_output_ids: Option<Vec<u32>>,
    ) -> Self {
        Self::from_prompt(
            owner,
            PromptTokens::Dense(tokens),
            max_output_tokens,
            output_capacity_hint,
            block_size,
            enable_prefix_caching,
            retain_local_hashes,
            emit_token_ids,
            planned_output_ids,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_prompt(
        owner: Uuid,
        tokens: PromptTokens,
        max_output_tokens: usize,
        output_capacity_hint: usize,
        block_size: usize,
        enable_prefix_caching: bool,
        retain_local_hashes: bool,
        emit_token_ids: bool,
        planned_output_ids: Option<Vec<u32>>,
        prefix_unit: Option<usize>,
    ) -> Self {
        let prompt_len = tokens.len();
        let prefix_hashes = prefix_unit.map(|unit| {
            let hashes = if unit == block_size && enable_prefix_caching {
                PrefixHashes::Identities
            } else {
                let mut hashes = Vec::with_capacity(prompt_len / unit);
                tokens.visit_blocks(unit, |_, hash| hashes.push(hash));
                PrefixHashes::Separate(hashes)
            };
            (unit, hashes)
        });
        let (sequence, identities) = RequestSequence::from_prompt(
            owner,
            tokens,
            max_output_tokens,
            output_capacity_hint,
            block_size,
            enable_prefix_caching,
            retain_local_hashes,
            emit_token_ids,
            planned_output_ids,
        );
        let mut lease = BlockRequestLease::new(owner, identities);
        if let Some((unit, hashes)) = prefix_hashes {
            lease.set_prefix_hashes(prompt_len, unit, hashes);
        }
        Self { sequence, lease }
    }

    #[cfg(test)]
    pub(super) fn native_parts(&self) -> (&RequestSequence, &BlockRequestLease) {
        (&self.sequence, &self.lease)
    }

    pub(super) fn len(&self) -> usize {
        self.sequence.len()
    }

    pub(super) fn max_output_tokens(&self) -> usize {
        self.sequence.max_output_tokens()
    }

    pub(super) fn generated_tokens(&self) -> usize {
        self.sequence.generated_tokens()
    }

    pub(super) fn num_input_tokens(&self) -> usize {
        self.sequence.num_input_tokens()
    }

    pub(super) fn num_allocated_tokens(&self) -> usize {
        self.lease.allocated_tokens()
    }

    pub(super) fn current_known_blocks(&self) -> usize {
        self.sequence.current_known_blocks()
    }

    pub(super) fn to_completion_blocks(&self) -> usize {
        self.sequence.to_completion_blocks()
    }

    pub(super) fn generate_token(&mut self) -> u32 {
        let (token, opened_partial) = self.sequence.generate_token();
        if opened_partial {
            self.lease.append_partial();
        }
        token
    }

    #[cfg(test)]
    pub(super) const fn uses_flat_tokens(&self) -> bool {
        true
    }

    #[cfg(test)]
    pub(super) fn native_storage_capacities(&self) -> (usize, usize) {
        (self.sequence.token_capacity(), self.lease.entry_capacity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TracePrompt;
    use crate::engine::common::hashing::{block_hashes, compute_next_sequence_hash};

    #[test]
    fn trace_admission_preserves_prefixes_and_generated_blocks() {
        let ids = [7, 2, 7, 11, 0];
        let block = 4;
        for trace_block in [2, 3, 4, 8] {
            for len in [0, 3, 4, 7, 8, 9] {
                let tokens: Vec<_> = (0..len).map(|i| ids[i / trace_block]).collect();
                for unit in [2, block] {
                    for (cache, emit_ids) in [(true, false), (true, true), (false, false)] {
                        let mut state = RequestKvState::from_prompt(
                            Uuid::nil(),
                            PromptTokens::Trace(TracePrompt {
                                len,
                                ids: ids.to_vec(),
                                block_size: trace_block,
                            }),
                            block,
                            block,
                            block,
                            cache,
                            true,
                            emit_ids,
                            Some(vec![42; block]),
                            Some(unit),
                        );
                        let mut parent = None;
                        for (i, local) in block_hashes(&tokens, unit).enumerate() {
                            let hash =
                                parent.map_or(local, |p| compute_next_sequence_hash(p, local));
                            parent = Some(hash);
                            assert_eq!(state.lease.prefix_hash_at((i + 1) * unit), Some(hash));
                        }
                        assert_eq!(state.lease.prefix_hash_at(0), None);
                        assert_eq!(state.lease.prefix_hash_at(1), None);
                        let mut expected = tokens.clone();
                        for _ in 0..block {
                            assert_eq!(state.generate_token(), 42);
                            expected.push(42);
                        }
                        let mut parent = None;
                        for (i, local) in block_hashes(&expected, block).enumerate() {
                            let hash =
                                parent.map_or(local, |p| compute_next_sequence_hash(p, local));
                            if cache && i < len / block {
                                assert_eq!(state.lease.sequence_hash(i), Some(hash));
                                assert_eq!(state.lease.local_hash(i), Some(local));
                            } else if cache {
                                assert_eq!(
                                    state
                                        .sequence
                                        .complete_block_identity(i, parent)
                                        .sequence_hash,
                                    Some(hash)
                                );
                            }
                            if emit_ids {
                                assert_eq!(
                                    state.sequence.block_token_ids(i).as_deref(),
                                    Some(&expected[i * block..(i + 1) * block])
                                );
                            }
                            parent = Some(hash);
                        }
                        assert_eq!(state.lease.prefix_hash_at((len / unit + 1) * unit), None);
                    }
                }
            }
        }
    }
}
