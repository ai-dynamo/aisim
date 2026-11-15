// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Modified simulation of the vLLM v0.31.0 and v0.32.0 NIXL connector rules
//! that decide the P/D transfer payload of each KV cache group, with the
//! in-window block count of a decode rank from the KV cache manager. No
//! speculative decoding, no HiSparse, no bounded replay window and no `all`
//! Mamba cache mode. Copyright contributors to the vLLM project. Sources: the
//! files below at `db9527a46873454610df6dbedf79a36d6bf1a7f6` and at
//! `5df63a7203cdb9328d824a27b682e77c7b9928be`. See THIRD_PARTY_NOTICES.md.
//! Apache-2.0.
//!
//! - https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py
//! - https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py
//! - https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py
//! - https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py
//! - https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/kv_cache_manager.py
//! - https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/single_type_kv_cache_manager.py
//! - https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/kv_cache_interface.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/utils.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_manager.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py
//!
//! The payload of a request is, for each group, the blocks of the group that
//! vLLM's NIXL connector transfers for one request on one TP rank times the
//! group's `transfer_bytes_per_block`: the bytes that the connector reads for
//! one block of that group, for the vLLM version of the deployment. Each TP
//! rank moves its own copy (`nixl/tp_mapping.py:88-98`, the same in V31 and
//! V32). The connector makes one region for each distinct layer address, and
//! layers of different groups at one address share a region (V31
//! `nixl/base_worker.py:1702-1724`, V32 `:1623-1645`). In a model with a Mamba
//! state, it reads a block of an attention group in every region, a block of a
//! `CircularBufferSpec` ring in its own regions only, and a block of a Mamba
//! group in every Mamba region (V31 `nixl/base_worker.py:274-304`, V32
//! `:236-266`), except that a `tp_replicated` state group reads its own region
//! only (V31 `:616-633`, V32 `:581-598`). So the bytes of one block depend on the addresses that the
//! layers of all groups share, and they can be more than the page of the group.
//! In V31 the ring of GLM-5.3-Flash is a `KpoolTailSpec` and not a
//! `CircularBufferSpec` (V31 `kvi:1015-1016`), so its block is read in every
//! region. In V32 that model runs a block-outer layout (V32
//! `vllm/v1/attention/backends/mla/indexer.py:312-314` and `utils.py:372-374`),
//! where the layers of other groups have addresses of their own in a block row,
//! so a full-attention block is read in those regions too. The rules compute no
//! bytes: they take them from the profile. For GLM-5.3-Flash at TP 4 the test
//! profiles hold the values of a live run on both versions, and the rules give
//! the bytes of every request of that run
//! (`benchmarks/evidence/nixl-pd-transfer-bytes-20261115/`).
//!
//! The rules hold for vLLM 0.31.0 (V31) and vLLM 0.32.0 (V32). They count the
//! same blocks in both versions and take the bytes of a block from the
//! profile, so they take no version. Two rules differ in code and give the
//! same value for every request that replay sends. The V32
//! backoff adds a term for a bounded replay window (V31
//! `nixl/base_scheduler.py:385-398`, V32 `:379-394`), and GLM-5.3-Flash has
//! none. The V32 truncation skips a request that skips reading the prefix
//! cache (V32 `nixl/base_scheduler.py:421-423`), and replay sends no such
//! request. A model with a bounded replay window, or a replay that skips the
//! prefix cache, needs a version row here.
//!
//! Grouped P/D uses four different blocks. These rules read two of them. The
//! transfer unit of a full-attention group and the window count read the
//! group's own `block_size_tokens`, because vLLM clips the block list of each
//! group with that group's spec (V31 `nixl/base_scheduler.py:133-146`,
//! `:282-297`, V32 `:138-151`, `:280-291`), and the decode rank allocates it
//! with that group's manager (see `fixed_bytes`). The timing unit is the one
//! block of the full-attention groups. `GroupedPrefixRules` reads the other
//! two: the decode-side hit reads the hit alignment, and the checkpoint rule
//! reads the cacheable block, `cache_config.block_size` in vLLM 0.32.0. For
//! GLM-5.3-Flash at TP 4, the scheduler block (the least common multiple of
//! the group blocks), `cache_config.block_size`, the hash block and the
//! alignment are all 2,176 tokens.
//!
//! Paths: `nixl/` is `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`,
//! `kvi` is `vllm/v1/kv_cache_interface.py` and `utils.py` is
//! `vllm/distributed/kv_transfer/kv_connector/utils.py`.

use anyhow::{bail, ensure};

use crate::engine::HandoffTransferTiming;
use crate::engine::common::protocols::KvTransferTimingMode;
use crate::engine::common::utils::prefill_handoff_transfer_timing;
use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};

/// How vLLM clips the block list of one group for a transfer
/// (`get_exchange_clipped_blocks`, V31 `nixl/base_scheduler.py:250-297`, V32
/// `:250-291`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransferKind {
    /// Full attention or MLA, without a clip. Every block of the transferred
    /// tokens moves as a whole block, also a partly filled last block (V31
    /// `nixl/base_worker.py:2033-2063`, V32 `:1973-2003`).
    Full,
    /// A sliding window or a convolution: the blocks of the transferred
    /// tokens inside the window. The prefill clips its list to the last
    /// `cdiv(window, block) + 1` blocks (V31 `nixl/base_scheduler.py:143-146`,
    /// `:284-285`, V32 `:148-151`, `:282-283`), and the worker trims that
    /// list to the in-window blocks of the decode rank, as
    /// `GroupedTransferRules::fixed_bytes` counts them.
    Window { window: u32 },
    /// A Mamba state or a ring: one block. A Mamba group moves its running
    /// state block in the cache modes that AISim models (V31
    /// `nixl/base_scheduler.py:156-160`, `:286-295`, V32 `:284-289`,
    /// `utils.py:37-44`). A ring holds one block per request (V31
    /// `kvi:1015-1025`, V32 `kvi:928-930`).
    State,
}

/// The transfer rule of one group, its `block_size_tokens` and the payload of
/// one of its blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TransferGroup {
    kind: TransferKind,
    block: u32,
    bytes_per_block: u64,
}

/// The transfer payload rules of one grouped cache.
#[derive(Clone, Debug)]
pub(crate) struct GroupedTransferRules {
    groups: Vec<TransferGroup>,
    /// The `block_size_tokens` of the full-attention groups, the unit of the
    /// handoff timing. `None` without a full-attention group.
    unit_block: Option<u32>,
    backoff_tokens: usize,
}

impl GroupedTransferRules {
    /// Classify the groups. Every group takes part: `enable_kv_transfer`
    /// defaults to true, and only HiSparse clears it (V31 `kvi:1461-1464` and
    /// `vllm/v1/hisparse/layout.py:182`, V32 `kvi:1452-1455` and
    /// `vllm/v1/hisparse/layout.py:183`). A prefix-cacheable state group is a
    /// Mamba state. A Mamba spec keeps the default `prefix_cacheable` (V31
    /// `kvi:172-174`, V32 `kvi:196-198`), and the ring specs opt out (V31
    /// `kvi:1032-1034`, V32 `kvi:939-941`). A Mamba state sets the backoff to
    /// one token (V31 `nixl/base_scheduler.py:102`, `:385-398`, V32 `:103`,
    /// `:379-394`). Grouped execution has no speculation, so the lookahead is
    /// 0 (V31 `vllm/config/vllm.py:668-670`, V32 `:655-657`) and its term
    /// does not count. Fails when a group fails `FpmCacheGroup::validate`,
    /// when a group lacks `prefix_cacheable` or `transfer_bytes_per_block`,
    /// and when two full-attention groups differ in block, because the timing
    /// has one unit.
    pub(crate) fn new(groups: &[FpmCacheGroup]) -> anyhow::Result<Self> {
        // A zero block would divide by zero in `full_blocks` and `fixed_bytes`.
        for group in groups {
            group.validate()?;
        }
        let Some(bytes) = groups
            .iter()
            .map(|group| group.prefix_cacheable.and(group.transfer_bytes_per_block))
            .collect::<Option<Vec<u64>>>()
        else {
            bail!("every group must declare prefix_cacheable and transfer_bytes_per_block");
        };
        let mut unit_block = None;
        let mut rules = Vec::with_capacity(groups.len());
        for (group, bytes_per_block) in groups.iter().zip(bytes) {
            let block = group.block_size_tokens;
            let kind = match (group.kind, group.sliding_window) {
                (FpmCacheKind::State, _) => TransferKind::State,
                (FpmCacheKind::Attention | FpmCacheKind::Convolution, Some(window)) => {
                    TransferKind::Window { window }
                }
                (FpmCacheKind::Attention, None) => {
                    ensure!(
                        unit_block.is_none_or(|unit| unit == block),
                        "full-attention groups must share one block_size_tokens"
                    );
                    unit_block = Some(block);
                    TransferKind::Full
                }
                // `validate` above refuses this group.
                (FpmCacheKind::Convolution, None) => {
                    bail!("convolution cache groups require a sliding_window")
                }
            };
            rules.push(TransferGroup {
                kind,
                block,
                bytes_per_block,
            });
        }
        let mamba = groups
            .iter()
            .any(|group| group.kind == FpmCacheKind::State && group.prefix_cacheable == Some(true));
        Ok(Self {
            groups: rules,
            unit_block,
            backoff_tokens: usize::from(mamba),
        })
    }

    /// The trailing prompt tokens that the prefill rank does not compute and
    /// the decode rank recomputes: 1 with a Mamba state, else 0 (V31
    /// `nixl/base_scheduler.py:385-398`, V32 `:379-394`).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "ReservedVllmDecode::activate reads it when a decode rank activates a grouped handoff"
        )
    )]
    pub(crate) fn backoff_tokens(&self) -> usize {
        self.backoff_tokens
    }

    /// The prompt tokens that the prefill rank computes and the decode rank
    /// loads. Only a prompt longer than the backoff loses the backoff tokens
    /// (V31 `nixl/base_scheduler.py:400-407`, `:424`, V32 `:396-403`, `:420`).
    pub(crate) fn transfer_tokens(&self, prompt_tokens: usize) -> usize {
        if prompt_tokens > self.backoff_tokens {
            prompt_tokens - self.backoff_tokens
        } else {
            prompt_tokens
        }
    }

    /// The full-attention blocks of `transfer_tokens` tokens, a partly filled
    /// last block included, or 0 without a full-attention group (V31
    /// `nixl/base_worker.py:2033-2063`, V32 `:1973-2003`).
    pub(crate) fn full_blocks(&self, transfer_tokens: usize) -> usize {
        self.unit_block
            .map_or(0, |block| transfer_tokens.div_ceil(block as usize))
    }

    /// The transfer bytes of one full-attention block, summed over the
    /// full-attention groups. vLLM does not clip the block list of a
    /// full-attention group (V31 `nixl/base_scheduler.py:282-297`, V32
    /// `:280-291`).
    pub(crate) fn bytes_per_full_block(&self) -> u64 {
        self.groups
            .iter()
            .filter(|group| group.kind == TransferKind::Full)
            .map(|group| group.bytes_per_block)
            .sum()
    }

    /// The bytes that do not grow with the full-attention blocks. Of `T`
    /// transferred tokens, a window group moves the
    /// `cdiv(T, block) - floor(max(0, T - window + 1) / block)` blocks inside
    /// the window of the next token. The decode rank skips the tokens before
    /// that window (V31
    /// `vllm/v1/core/single_type_kv_cache_manager.py:1185-1188`, V32
    /// `:1167-1170`), allocates only the blocks after them (V31 `:240-251`,
    /// V32 `:237-248`) and pads the skipped blocks with null blocks (V31
    /// `:313-314`, V32 `:310-311`), which the connector leaves out
    /// (`vllm/v1/core/kv_cache_manager.py:114-124`, the same in V31 and V32).
    /// The worker trims the prefill list to the decode count (V31
    /// `nixl/base_worker.py:3485-3490`, `:3529-3534`, V32 `:3450-3455`,
    /// `:3493-3498`). A state group moves one block (V31
    /// `nixl/base_scheduler.py:286-295`, V32 `:284-289`).
    pub(crate) fn fixed_bytes(&self, transfer_tokens: usize) -> u64 {
        self.groups
            .iter()
            .map(|group| {
                let blocks = match group.kind {
                    TransferKind::Full => 0,
                    TransferKind::Window { window } => {
                        let block = group.block as usize;
                        let skipped = transfer_tokens.saturating_sub(window as usize - 1) / block;
                        let blocks = transfer_tokens.div_ceil(block) - skipped;
                        // vLLM asserts that the decode list is no longer than
                        // the prefill clip (V31 `nixl/base_worker.py:3488`,
                        // V32 `:3453`).
                        debug_assert!(blocks <= (window as usize).div_ceil(block) + 1);
                        blocks
                    }
                    TransferKind::State => 1,
                };
                blocks as u64 * group.bytes_per_block
            })
            .sum()
    }

    /// The payload of one request when the decode rank already holds
    /// `local_full_blocks` full-attention blocks: the missing full-attention
    /// blocks and the fixed bytes. The decode rank pulls only the blocks that
    /// its prefix cache lacks (`nixl/pull_scheduler.py:149-156`, the same in
    /// V31 and V32), and the state moves with every transfer that moves a
    /// token (V31 `nixl/base_worker.py:3507-3528`, V32 `:3472-3492`). When
    /// the decode rank holds every block, vLLM moves nothing
    /// (`nixl/pull_scheduler.py:59-65`), but this formula still counts the
    /// fixed bytes, as the handoff timing does.
    #[cfg(test)]
    pub(crate) fn payload_bytes(&self, transfer_tokens: usize, local_full_blocks: usize) -> u64 {
        let missing = self
            .full_blocks(transfer_tokens)
            .saturating_sub(local_full_blocks);
        missing as u64 * self.bytes_per_full_block() + self.fixed_bytes(transfer_tokens)
    }

    /// The handoff timing of one request, counted in full-attention blocks,
    /// not tokens. `full_prompt_tokens` is `full_blocks(transfer_tokens)`,
    /// `kv_bytes_per_token` holds the bytes of one full-attention block and
    /// `state_bytes` holds the fixed bytes. The destination reports its
    /// missing full-attention blocks in the same unit, as vLLM pulls only the
    /// blocks that the decode rank lacks (`nixl/pull_scheduler.py:149-156`,
    /// the same in V31 and V32). So the delay is the `payload_bytes` over the
    /// bandwidth in both timing modes. The struct comes from the builder of
    /// the linear path, which also maps the mode.
    pub(crate) fn handoff_transfer_timing(
        &self,
        transfer_tokens: usize,
        bandwidth_gb_s: Option<f64>,
        mode: KvTransferTimingMode,
    ) -> HandoffTransferTiming {
        prefill_handoff_transfer_timing(
            self.full_blocks(transfer_tokens),
            bandwidth_gb_s,
            Some(self.bytes_per_full_block() as usize),
            self.fixed_bytes(transfer_tokens) as usize,
            mode,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLM_PAGE: u64 = 13_045_120;
    const GLM_BLOCK: u32 = 2176;

    /// One group whose page and transfer bytes are both `bytes`, with one
    /// running block for a state. A prefix-cacheable state records 0 prefill
    /// checkpoint blocks and no alignment, the `MambaSpec` defaults (V31
    /// `kvi:1049-1050`).
    fn group(
        kind: FpmCacheKind,
        block: u32,
        bytes: u64,
        window: Option<u32>,
        marker: bool,
    ) -> FpmCacheGroup {
        let state = kind == FpmCacheKind::State;
        FpmCacheGroup {
            name: "group".into(),
            kind,
            num_layers: 1,
            block_size_tokens: block,
            page_size_bytes: bytes,
            sliding_window: window,
            blocks_per_request: state.then_some(1),
            prefix_cacheable: Some(marker),
            transfer_bytes_per_block: Some(bytes),
            prefill_checkpoint_blocks: (state && marker).then_some(0),
            prefill_checkpoint_alignment: None,
        }
    }

    /// The bytes that the connector reads for one block of each of the six
    /// GLM-5.3-Flash groups at TP 4 with an fp8 cache: attention, ring and four
    /// KDA states of 9, 9, 8 and 8 layers. A live run measured, with
    /// `vllm:nixl_bytes_transferred_sum` and `_count`
    /// (`benchmarks/evidence/nixl-pd-transfer-bytes-20261115/`), the attention
    /// value and the sum of the other five groups. The split of that sum is a
    /// reading of the connector code. vLLM 0.31.0 reads 22 regions:
    /// `11 * 1_114_112 + 11 * 71_808` for an attention block and for the ring
    /// block, a `KpoolTailSpec`. vLLM 0.32.0 reads 40 regions for an attention
    /// block, `11 * 71_808 + 19 * 1_114_112 + 10 * 2_048`, and the 11 own
    /// regions of the `CircularBufferSpec` ring, `71_808 + 10 * 2_048`. A KDA
    /// block is read in the 9 state regions on both versions:
    /// `9 * (36_864 + 1_048_576)`, also for a group of 8 layers.
    fn glm_transfer_bytes(version: &str) -> [u64; 6] {
        match version {
            "0.31.0" => [
                13_045_120, 13_045_120, 9_768_960, 9_768_960, 9_768_960, 9_768_960,
            ],
            "0.32.0" => [
                21_978_496, 92_288, 9_768_960, 9_768_960, 9_768_960, 9_768_960,
            ],
            _ => unreachable!("no measured GLM-5.3-Flash transfer bytes"),
        }
    }

    /// The six GLM-5.3-Flash groups at TP 4 as a hand-made profile of
    /// `version`, with the transfer bytes of `glm_transfer_bytes`.
    /// Every page is 13,045,120 bytes, the block row of vLLM 0.31.0, which
    /// its pool of 14,351 blocks in 174.36 GiB agrees with. The run measured
    /// no page of vLLM 0.32.0: its pool held 13,888 blocks in 174.83 GiB,
    /// about 13,517,000 bytes for each block row, and the rules do not read
    /// the page.
    fn glm_groups(version: &str) -> Vec<FpmCacheGroup> {
        let mut groups = vec![
            group(FpmCacheKind::Attention, GLM_BLOCK, 0, None, true),
            group(FpmCacheKind::State, 4, 0, None, false),
            group(FpmCacheKind::State, GLM_BLOCK, 0, None, true),
            group(FpmCacheKind::State, GLM_BLOCK, 0, None, true),
            group(FpmCacheKind::State, GLM_BLOCK, 0, None, true),
            group(FpmCacheKind::State, GLM_BLOCK, 0, None, true),
        ];
        for (group, bytes) in groups.iter_mut().zip(glm_transfer_bytes(version)) {
            group.page_size_bytes = GLM_PAGE;
            group.transfer_bytes_per_block = Some(bytes);
        }
        groups
    }

    fn rules(groups: &[FpmCacheGroup]) -> GroupedTransferRules {
        GroupedTransferRules::new(groups).unwrap()
    }

    fn refusal(groups: &[FpmCacheGroup]) -> String {
        GroupedTransferRules::new(groups).unwrap_err().to_string()
    }

    fn close(delay: Option<f64>, expected: f64) -> bool {
        delay.is_some_and(|delay| (delay - expected).abs() < 1e-9)
    }

    /// The measured rows of
    /// `benchmarks/evidence/nixl-pd-transfer-bytes-20261115/`: prompt tokens, the full-attention
    /// blocks that the decode rank already holds, and the bytes of one rank.
    /// The decode rank holds 2 blocks when the same prompt of 5,000 tokens
    /// comes again.
    fn measured_rows(version: &str) -> [(usize, usize, u64); 7] {
        match version {
            "0.31.0" => [
                (1_000, 0, 65_166_080),
                (2_176, 0, 65_166_080),
                (2_177, 0, 65_166_080),
                (5_000, 0, 91_256_320),
                (5_000, 2, 65_166_080),
                (30_000, 0, 234_752_640),
                (130_000, 0, 834_828_160),
            ],
            "0.32.0" => [
                (1_000, 0, 61_146_624),
                (2_176, 0, 61_146_624),
                (2_177, 0, 61_146_624),
                (5_000, 0, 105_103_616),
                (5_000, 2, 61_146_624),
                (30_000, 0, 346_867_072),
                (130_000, 0, 1_357_877_888),
            ],
            _ => unreachable!("no measured GLM-5.3-Flash rows"),
        }
    }

    #[test]
    fn glm_payload_matches_the_vllm_oracle() {
        // The payload of a prompt of N tokens is the bytes of its N - 1
        // transferred tokens: `ceil((N - 1) / 2_176)` attention blocks less
        // the held ones, and one block of each state group. Each case is a
        // measured row of the evidence folder. A state group moves its
        // running block only (V32 `utils.py:37-44`), never the vLLM 0.32.0
        // FlashKDA checkpoint block, so the profile with it gives the same
        // rows.
        for (version, checkpoints) in [("0.31.0", &[false][..]), ("0.32.0", &[false, true])] {
            for &checkpoint in checkpoints {
                let mut groups = glm_groups(version);
                if checkpoint {
                    for group in &mut groups[2..] {
                        group.prefill_checkpoint_blocks = Some(1);
                        group.prefill_checkpoint_alignment = Some(16);
                    }
                }
                let rules = rules(&groups);
                assert_eq!(rules.backoff_tokens(), 1);
                for (n, held, bytes) in measured_rows(version) {
                    assert_eq!(rules.transfer_tokens(n), n - 1, "{version} N {n}");
                    assert_eq!(
                        rules.payload_bytes(n - 1, held),
                        bytes,
                        "{version} N {n} held {held}"
                    );
                }
            }
        }

        // `delay_ms` divides by 10^6 at 1 GB/s. The fixed bytes are the
        // measured row of 1,000 tokens less one attention block. Their split
        // into one block of each state group, `13_045_120 + 4 * 9_768_960` on
        // vLLM 0.31.0 and `92_288 + 4 * 9_768_960` on vLLM 0.32.0, is a
        // reading of the connector code. The full-prompt delay is the measured
        // row of 1,000 tokens.
        for (version, block, fixed) in [
            ("0.31.0", 13_045_120, 52_120_960),
            ("0.32.0", 21_978_496, 39_168_128),
        ] {
            let rules = rules(&glm_groups(version));
            // T 999 has 1 block and 5 are held: the missing count floors at 0.
            assert_eq!(rules.payload_bytes(999, 5), fixed, "{version}");
            let full =
                rules.handoff_transfer_timing(999, Some(1.0), KvTransferTimingMode::FullPrompt);
            assert_eq!(full.full_prompt_tokens, 1);
            assert_eq!(full.kv_bytes_per_token, Some(block));
            assert_eq!(full.state_bytes, fixed as usize);
            let row = measured_rows(version)[0].2;
            assert!(close(full.delay_ms(0), row as f64 / 1e6), "{version}");
            let missing = rules.handoff_transfer_timing(
                999,
                Some(1.0),
                KvTransferTimingMode::DestinationMissing,
            );
            // This pins the arithmetic of the timing struct only. When no
            // block is missing, vLLM moves nothing (`nixl/pull_scheduler.py:59-65`,
            // `:149-153`), while this timing still charges the fixed bytes: a
            // known difference that the run did not measure.
            assert!(close(missing.delay_ms(0), fixed as f64 / 1e6), "{version}");
            assert!(close(missing.delay_ms(1), row as f64 / 1e6), "{version}");
        }
    }

    #[test]
    fn page_sizes_are_not_the_payload() {
        // With the hand-made page as the bytes of a block, the six groups give
        // `6 * 13_045_120 = 78_270_720` bytes for 999 tokens (computed), not
        // the measured 65,166,080 of vLLM 0.31.0 or 61,146,624 of vLLM 0.32.0.
        for (version, measured) in [("0.31.0", 65_166_080), ("0.32.0", 61_146_624)] {
            let pages: Vec<_> = glm_groups(version)
                .into_iter()
                .map(|mut group| {
                    group.transfer_bytes_per_block = Some(group.page_size_bytes);
                    group
                })
                .collect();
            assert_eq!(rules(&pages).payload_bytes(999, 0), 78_270_720);
            assert_eq!(rules(&glm_groups(version)).payload_bytes(999, 0), measured);
        }
    }

    #[test]
    fn window_groups_move_their_last_blocks_only() {
        for kind in [FpmCacheKind::Attention, FpmCacheKind::Convolution] {
            let window = rules(&[group(kind, 16, 100, Some(33), true)]);
            // 1,000 tokens fill 63 blocks. The window of the next token skips
            // `1_000 - 33 + 1 = 968` tokens, 60 whole blocks, so 3 blocks of
            // 100 bytes move, not the prefill clip of `cdiv(33, 16) + 1 = 4`.
            assert_eq!(window.fixed_bytes(1_000), 300);
            // 40 tokens fill 3 blocks, and the window skips 8 tokens, no
            // whole block: all 3 move.
            assert_eq!(window.fixed_bytes(40), 300);
            assert_eq!(window.full_blocks(1_000), 0);
            assert_eq!(window.backoff_tokens(), 0);
            // A window of 32 and 1,024 tokens both end on a block. The window
            // of the next token skips `1_024 - 32 + 1 = 993` tokens, 62 whole
            // blocks, so blocks 62 and 63 move: 2, where the prefill clip
            // `cdiv(32, 16) + 1` is 3.
            let aligned = rules(&[group(kind, 16, 100, Some(32), true)]);
            assert_eq!(aligned.fixed_bytes(1_024), 200);
        }
    }

    #[test]
    fn full_and_window_groups_add_up_in_one_payload() {
        let rules = rules(&[
            group(FpmCacheKind::Attention, 64, 1_000, None, true),
            group(FpmCacheKind::Attention, 16, 100, Some(33), true),
        ]);
        // T 1_000: `ceil(1_000 / 64) = 16` full blocks of 1_000 bytes, and
        // the 3 window blocks of 100 bytes inside the window: 16_300.
        assert_eq!(rules.payload_bytes(1_000, 0), 16_300);
        // 10 full blocks held: `6 * 1_000 + 3 * 100 = 6_300`.
        assert_eq!(rules.payload_bytes(1_000, 10), 6_300);
    }

    #[test]
    fn state_groups_move_one_block_each_and_set_the_backoff() {
        let states = |mamba: bool| {
            let mut groups = vec![
                group(FpmCacheKind::State, 16, 50, None, mamba),
                group(FpmCacheKind::State, 16, 50, None, false),
            ];
            for group in &mut groups {
                group.blocks_per_request = Some(2);
            }
            rules(&groups)
        };
        let mamba = states(true);
        // Each group moves its running block, not its second block.
        assert_eq!(mamba.fixed_bytes(1_000), 100);
        assert_eq!(mamba.backoff_tokens(), 1);
        assert_eq!(mamba.transfer_tokens(2), 1);
        // A prompt of one token is not truncated.
        assert_eq!(mamba.transfer_tokens(1), 1);
        // Rings alone do not truncate.
        let rings = states(false);
        assert_eq!(rings.backoff_tokens(), 0);
        assert_eq!(rings.transfer_tokens(1_000), 1_000);
    }

    #[test]
    fn rules_refuse_missing_markers_and_mixed_full_blocks() {
        let missing = "every group must declare prefix_cacheable and transfer_bytes_per_block";
        let mut groups = glm_groups("0.32.0");
        groups[1].transfer_bytes_per_block = None;
        assert_eq!(refusal(&groups), missing);
        let mut groups = glm_groups("0.32.0");
        groups[2].prefix_cacheable = None;
        assert_eq!(refusal(&groups), missing);

        let full = |block| group(FpmCacheKind::Attention, block, 100, None, true);
        assert_eq!(
            refusal(&[full(64), full(16)]),
            "full-attention groups must share one block_size_tokens"
        );
        // A missing marker is reported before mixed full blocks.
        let mut unmarked = full(16);
        unmarked.prefix_cacheable = None;
        assert_eq!(refusal(&[full(64), unmarked]), missing);

        let window = group(FpmCacheKind::Attention, 16, 100, Some(32), true);
        assert_eq!(rules(&[full(64), window]).unit_block, Some(64));
    }

    #[test]
    fn rules_refuse_a_group_that_fails_validation() {
        // A zero block would divide by zero in `full_blocks` or `fixed_bytes`.
        for window in [None, Some(33)] {
            let mut zero = group(FpmCacheKind::Attention, 16, 100, window, true);
            zero.block_size_tokens = 0;
            assert_eq!(
                refusal(&[zero]),
                "invalid engine config: cache group requires a name, positive layers/block/page/window, and page_size_bytes <= 2**53"
            );
        }
    }

    #[test]
    fn no_full_group_gives_a_fixed_payload() {
        // The ring and one KDA group of `glm_groups("0.32.0")`, without the
        // attention group: `92_288 + 9_768_960 = 9_861_248` bytes, computed
        // from the split of `glm_transfer_bytes`, a reading of the connector
        // code.
        let glm = glm_groups("0.32.0");
        let rules = rules(&glm[1..3]);
        assert_eq!(rules.unit_block, None);
        assert_eq!(rules.full_blocks(5_000), 0);
        assert_eq!(rules.bytes_per_full_block(), 0);
        assert_eq!(rules.fixed_bytes(5_000), 9_861_248);
        // 9,861,248 bytes at 1 GB/s.
        let timing =
            rules.handoff_transfer_timing(5_000, Some(1.0), KvTransferTimingMode::FullPrompt);
        assert!(close(timing.delay_ms(0), 9.861248));
    }
}
