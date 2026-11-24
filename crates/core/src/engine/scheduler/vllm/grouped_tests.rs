// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project.
// Tests of a modified Rust simulation of the vLLM v0.31.0
// (db9527a46873454610df6dbedf79a36d6bf1a7f6), v0.32.0
// (5df63a7203cdb9328d824a27b682e77c7b9928be) and v0.33.0
// (4ba787ec9959e566d3b587a796dff07673324da5) grouped KV rules and grouped KV
// handoff rules.
// Immutable sources and license: THIRD_PARTY_NOTICES.md.

use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::engine::common::perf_model::PerfModel;
use crate::engine::common::protocols::{
    DirectRequest, KvTransferTimingMode, MockEngineArgs, WorkerType,
};
use crate::engine::kv_manager::BlockRequestLease;
use crate::engine::scheduler::{SchedulerCommand, SchedulerCommandResult, SchedulerLifecycleEvent};
use crate::engine::trace::TraceCollector;
use crate::engine::{HandoffId, HandoffTransferTiming};
use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};

use super::core::{RequestStatus, VllmCore};
use super::request::RequestKvState;

/// Decode queries `(active_tokens, mean_context, token_bound)` and prefill
/// queries `(batch, new_tokens, prefix_tokens)`.
#[derive(Default)]
struct CapturedDecodeQueries(
    Mutex<Vec<(usize, usize, usize)>>,
    Mutex<Vec<(usize, usize, usize)>>,
);

impl crate::engine::TimingModel for CapturedDecodeQueries {
    fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
        Ok(0.0)
    }

    fn predict_prefill_totals_ms(
        &self,
        batch_size: usize,
        new_tokens: usize,
        prefix_tokens: usize,
    ) -> anyhow::Result<f64> {
        self.1
            .lock()
            .unwrap()
            .push((batch_size, new_tokens, prefix_tokens));
        Ok(0.0)
    }

    fn predict_decode_ms(
        &self,
        _: usize,
        active_tokens: usize,
        mean_context: usize,
        token_bound: usize,
    ) -> anyhow::Result<f64> {
        self.0
            .lock()
            .unwrap()
            .push((active_tokens, mean_context, token_bound));
        Ok(0.0)
    }
}

fn group(name: &str, block: u32, bytes: u64, window: Option<u32>) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind: FpmCacheKind::Attention,
        num_layers: 1,
        block_size_tokens: block,
        page_size_bytes: bytes,
        sliding_window: window,
        blocks_per_request: None,
        prefix_cacheable: None,
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: None,
        prefill_checkpoint_alignment: None,
    }
}

fn state(name: &str, bytes: u64, blocks: u32) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind: FpmCacheKind::State,
        num_layers: 1,
        block_size_tokens: 4,
        page_size_bytes: bytes,
        sliding_window: None,
        blocks_per_request: Some(blocks),
        prefix_cacheable: None,
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: None,
        prefill_checkpoint_alignment: None,
    }
}

fn core(groups: Vec<FpmCacheGroup>, capacity: u64, chunk: usize) -> VllmCore {
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            // Deliberately smaller than the prompts: this linear capacity
            // must not decide admission or accounting for grouped caches.
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(chunk))
            .max_num_seqs(Some(3))
            .enable_prefix_caching(false)
            .kv_cache_groups(groups)
            .kv_cache_capacity_bytes(Some(capacity))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
}

/// `core` for a prefill or decode rank, which times a full-prompt transfer at
/// `bandwidth` decimal GB/s.
fn pd_core(
    groups: Vec<FpmCacheGroup>,
    capacity: u64,
    chunk: usize,
    worker_type: WorkerType,
    bandwidth: Option<f64>,
) -> VllmCore {
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(chunk))
            .max_num_seqs(Some(3))
            .max_model_len(Some(200_000))
            .enable_prefix_caching(false)
            .worker_type(worker_type)
            .kv_transfer_bandwidth(bandwidth)
            .kv_transfer_timing_mode(KvTransferTimingMode::FullPrompt)
            .kv_cache_groups(groups)
            .kv_cache_capacity_bytes(Some(capacity))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
}

fn submit(core: &mut VllmCore, id: u128, prompt: usize, output: usize) -> Uuid {
    submit_tokens(core, id, (0..prompt as u32).collect(), output)
}

fn submit_tokens(core: &mut VllmCore, id: u128, tokens: Vec<u32>, output: usize) -> Uuid {
    core.receive(DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens,
        max_output_tokens: output,
        ..Default::default()
    })
}

/// A handoff prefill as Replay sends it, with one output token.
fn source_request(id: u128, prompt: usize) -> DirectRequest {
    DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens: (0..prompt as u32).collect(),
        max_output_tokens: 1,
        ..Default::default()
    }
}

/// Step until the prefill rank holds its source, and return the timing that
/// the held source reports.
fn held_timing(core: &mut VllmCore) -> HandoffTransferTiming {
    for _ in 0..8 {
        let pass = core.try_execute_hidden_pass(0.0).unwrap();
        assert!(pass.output_signals.iter().all(|output| !output.rejected));
        if let Some(timing) = pass.lifecycle_events.iter().find_map(|event| match event {
            SchedulerLifecycleEvent::SourceHeld {
                transfer_timing, ..
            } => Some(*transfer_timing),
            _ => None,
        }) {
            return timing;
        }
    }
    panic!("source must be held");
}

/// Prefill a source of `prompt` tokens and return its held timing, like
/// `hold_source` of `core.rs`.
fn hold_grouped_source(
    core: &mut VllmCore,
    handoff_id: HandoffId,
    id: u128,
    prompt: usize,
) -> HandoffTransferTiming {
    core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
        handoff_id,
        request: source_request(id, prompt),
    })
    .unwrap();
    held_timing(core)
}

const GLM_PAGE: u64 = 13_045_120;
const GLM_BLOCK: u32 = 2176;

fn glm_state(name: &str, block: u32, cacheable: bool, checkpoint: bool) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind: FpmCacheKind::State,
        num_layers: 1,
        block_size_tokens: block,
        page_size_bytes: GLM_PAGE,
        sliding_window: None,
        blocks_per_request: Some(1),
        prefix_cacheable: Some(cacheable),
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: cacheable.then_some(u32::from(checkpoint)),
        prefill_checkpoint_alignment: (cacheable && checkpoint).then_some(16),
    }
}

/// Six GLM-5.3-Flash TP 4 cache groups as a hand-made profile of vLLM 0.31.0,
/// 0.32.0 or 0.33.0, at one page size. Their kinds and blocks are those that
/// the vLLM 0.33.0 runtime probe records (fixture
/// `glm53flash_vllm_0330_probe`): MLA attention and four KDA `MambaSpec` states
/// with 2,176-token blocks, and a 4-token `CircularBufferSpec` ring. The layer
/// counts and the page size are hand-made. With `checkpoint`, G2 to G5 carry
/// one FlashKDA checkpoint block at alignment 16.
/// The page is the sum of the pages of the attention layers, 13,045,120 bytes,
/// the block row of vLLM 0.31.0. vLLM 0.32.0 rounds the row up to the least
/// common multiple of the block-stride alignments of all layers (v0.32.0
/// `vllm/v1/core/kv_cache_utils.py:1533-1551`): 13,516,800 bytes for this
/// model at TP 4 on GB300, 25 times the least common multiple of the
/// 16,384-byte alignment of the MLA layers and the 4,224-byte alignment of the
/// indexer layers. The fixture keeps the page of the attention layers. The
/// tests count blocks and tokens, which do not depend on the page, and their
/// pool byte values are multiples of this page, not the pool of a vLLM 0.32.0
/// deployment.
fn glm_groups(checkpoint: bool) -> Vec<FpmCacheGroup> {
    let mut mla = group("mla", GLM_BLOCK, GLM_PAGE, None);
    mla.prefix_cacheable = Some(true);
    let mut groups = vec![mla, glm_state("kpool_tail", 4, false, false)];
    for i in 0..4 {
        groups.push(glm_state(&format!("kda_{i}"), GLM_BLOCK, true, checkpoint));
    }
    groups
}

/// `glm_groups` with the bytes that the connector reads for one block of each
/// group on `version`, for GLM-5.3-Flash at TP 4: attention, ring and four KDA
/// states. A live run measured the attention value and the sum of the other
/// five groups, the fixed bytes of a request: 52,120,960 on vLLM 0.31.0 and
/// 39,168,128 on vLLM 0.32.0
/// (`benchmarks/evidence/nixl-pd-transfer-bytes-20261115/`). The split of that
/// sum among the groups is a reading of the connector code. On vLLM 0.33.0 a
/// log of each group measured the six values, and the run moved the rows of
/// vLLM 0.32.0 (`benchmarks/evidence/nixl-pd-transfer-bytes-20261117/`).
fn glm_transfer_groups(version: &str, checkpoint: bool) -> Vec<FpmCacheGroup> {
    let mut groups = glm_groups(checkpoint);
    let bytes = match version {
        "0.31.0" => [
            13_045_120, 13_045_120, 9_768_960, 9_768_960, 9_768_960, 9_768_960,
        ],
        "0.32.0" | "0.33.0" => [
            21_978_496, 92_288, 9_768_960, 9_768_960, 9_768_960, 9_768_960,
        ],
        _ => unreachable!("no measured GLM-5.3-Flash transfer bytes"),
    };
    for (group, bytes) in groups.iter_mut().zip(bytes) {
        group.transfer_bytes_per_block = Some(bytes);
    }
    groups
}

/// Five GLM cores: vLLM 0.31.0, and vLLM 0.32.0 and 0.33.0 each without and
/// with checkpoint blocks, which GLM-5.3-Flash takes only under the FlashKDA
/// prefill backend. The P/D transfer rules cover the same versions.
const GLM_CORES: [(&str, bool); 5] = [
    ("0.31.0", false),
    ("0.32.0", false),
    ("0.32.0", true),
    ("0.33.0", false),
    ("0.33.0", true),
];

fn prefix_core(
    groups: Vec<FpmCacheGroup>,
    capacity: u64,
    chunk: usize,
    unit: usize,
    version: &str,
) -> VllmCore {
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(chunk))
            .max_num_seqs(Some(3))
            .max_model_len(Some(131_072))
            .enable_prefix_caching(true)
            .prefix_match_unit(Some(unit))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(groups)
            .kv_cache_capacity_bytes(Some(capacity))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
}

/// Step until the core drains. Returns `(uuid, reused_input_tokens)` of every
/// admission in order.
fn drain(
    core: &mut VllmCore,
    trace: &mut TraceCollector,
    from_step: &mut usize,
) -> Vec<(Uuid, usize)> {
    let mut admissions = Vec::new();
    for _ in 0..200 {
        let pass = core.execute_pass(trace, *from_step as f64);
        *from_step += 1;
        admissions.extend(
            pass.admissions
                .iter()
                .map(|admission| (admission.uuid, admission.reused_input_tokens)),
        );
        for output in pass.output_signals {
            assert!(!output.rejected, "{} rejected", output.uuid);
        }
        if core.is_drained() {
            return admissions;
        }
    }
    panic!("the core did not drain");
}

fn block_pool(core: &VllmCore) -> &crate::engine::kv_manager::GroupedBlockPool {
    core.kv_manager.grouped().unwrap().block_pool().unwrap()
}

#[test]
fn grouped_resumed_request_takes_a_checkpoint_block_only_at_its_prefill_end() {
    // vLLM 0.32.0 ends the prefill of a request at `max(prompt, num_tokens -
    // 1)`, which extends it to a resumed request that replays its output
    // tokens (v0.32 kv_cache_manager.py:518-520, sched:463-465,
    // sched:3075). Mamba reserves a checkpoint block only on the chunk that
    // reaches that end (v0.32 single:1643-1674, :1750-1758). A prompt of 33
    // tokens grows to 113 known tokens and is preempted. Its re-admission
    // hits the 32 tokens that its prefill published, and the first chunk of
    // 32 tokens ends at 64 < 112, so it takes no checkpoint block.
    let mut full = group("full", 16, GLM_PAGE, None);
    full.prefix_cacheable = Some(true);
    let groups = vec![full, glm_state("s", 16, true, true)];
    let mut core = prefix_core(groups, 8_200 * GLM_PAGE, 32, 16, "0.32.0");
    let id = submit(&mut core, 1700, 33, 200);
    let mut trace = TraceCollector::default();
    let mut step = 0.0;
    for _ in 0..200 {
        if core.state.requests[&id].sequence.len() >= 113 {
            break;
        }
        core.execute_pass(&mut trace, step);
        step += 1.0;
    }
    assert_eq!(core.state.requests[&id].sequence.len(), 113);
    assert!(core.policy_preempt(step).is_some());
    assert_eq!(block_pool(&core).active_blocks(), 0);
    let pass = core.execute_pass(&mut trace, step);
    let admissions: Vec<_> = pass
        .admissions
        .iter()
        .map(|admission| (admission.uuid, admission.reused_input_tokens))
        .collect();
    assert_eq!(admissions, [(id, 32)]);
    assert_eq!(core.state.requests[&id].num_computed_tokens, 64);
    // Four full-attention blocks, the running state block and the hit state
    // block, which stays until the request's next allocation, as in vLLM
    // (v0.32 kv_cache_manager.py:555-559).
    assert_eq!(block_pool(&core).active_blocks(), 4 + 1 + 1);
}

#[test]
fn grouped_window_replays_long_context_with_bounded_storage_and_logical_fpm_lengths() {
    let mut core = core(vec![group("window", 4, 8, Some(4))], 24, 8);
    let captured = Arc::new(CapturedDecodeQueries::default());
    core.args.perf_model = Arc::new(PerfModel::External {
        timing: captured.clone(),
    });
    let id = submit(&mut core, 101, 32, 5);
    let mut trace = TraceCollector::default();
    let mut outputs = 0;
    let mut prefixes = Vec::new();
    let mut decode_contexts = Vec::new();
    let mut bytes = Vec::new();
    for step in 0..20 {
        let pass = core.execute_pass(&mut trace, step as f64);
        let fpm = pass.fpm.unwrap();
        if fpm.num_prefill_requests > 0 {
            prefixes.push(fpm.sum_prefill_kv_tokens);
            assert_eq!(fpm.sum_prefill_tokens, 8);
        }
        if fpm.num_decode_requests > 0 {
            decode_contexts.push(fpm.sum_decode_kv_tokens);
        }
        bytes.push(pass.mocker_metrics.kv_cache_used_bytes.unwrap());
        assert_eq!(pass.mocker_metrics.total_blocks, 0);
        assert_eq!(pass.mocker_metrics.kv_cache_capacity_bytes, Some(24));
        assert!(pass.mocker_metrics.physical_gpu_cache_usage_perc <= 1.0);
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
        for output in pass.output_signals {
            assert_eq!(output.uuid, id);
            assert!(!output.rejected);
            outputs += usize::from(output.token_id.is_some());
        }
        if core.is_drained() {
            break;
        }
    }
    assert!(core.is_drained());
    assert_eq!(outputs, 5);
    assert_eq!(prefixes, vec![0, 8, 16, 24]);
    assert_eq!(decode_contexts, vec![33, 34, 35, 36]);
    assert_eq!(
        *captured.0.lock().unwrap(),
        // Prefill produces output token one. The four decode forwards include
        // their current input token and retain the full logical context.
        vec![(33, 33, 33), (34, 34, 34), (35, 35, 35), (36, 36, 36)]
    );
    assert_eq!(&bytes[..4], &[16, 24, 24, 24]);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_full_window_and_convolution_charge_distinct_padded_pages() {
    let mut convolution = group("convolution", 2, 2, Some(2));
    convolution.kind = FpmCacheKind::Convolution;
    let mut core = core(
        vec![
            group("full", 4, 8, None),
            group("window", 4, 4, Some(4)),
            convolution,
        ],
        100,
        4,
    );
    let id = submit(&mut core, 102, 16, 5);
    let mut trace = TraceCollector::default();
    let mut bytes = Vec::new();
    for step in 0..5 {
        core.execute_pass(&mut trace, step as f64);
        bytes.push(core.mocker_metrics().kv_cache_used_bytes.unwrap());
    }
    assert_eq!(bytes, vec![16, 30, 38, 46, 52]);
    assert_eq!(core.state.requests[&id].num_computed_tokens, 17);
    assert_eq!(core.state.requests[&id].sequence.num_allocated_tokens(), 17);
    assert_eq!(core.state.requests[&id].sequence.len(), 18);
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_contention_preempts_and_retries_without_leaking_capacity() {
    let mut core = core(
        vec![group("full", 4, 4, None), group("window", 4, 4, Some(4))],
        24,
        8,
    );
    let first = submit(&mut core, 103, 4, 6);
    let second = submit(&mut core, 104, 4, 6);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    assert_eq!(initial.admissions.len(), 2);
    assert_eq!(initial.mocker_metrics.kv_cache_used_bytes, Some(16));
    let pressure = core.execute_pass(&mut trace, 1.0);
    assert_eq!(pressure.mocker_metrics.vllm_preemptions_total, 1);
    assert_eq!(pressure.mocker_metrics.kv_cache_used_bytes, Some(16));
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    assert_eq!(
        core.state.requests[&second].sequence.num_allocated_tokens(),
        0
    );
    assert_eq!(core.state.requests[&first].num_computed_tokens, 5);
    let mut completed = Vec::new();
    for step in 2..40 {
        let pass = core.execute_pass(&mut trace, step as f64);
        assert!(pass.mocker_metrics.kv_cache_used_bytes.unwrap() <= 24);
        for output in pass.output_signals {
            assert!(!output.rejected);
            if output.completed {
                completed.push(output.uuid);
            }
        }
        if core.is_drained() {
            break;
        }
    }
    assert_eq!(completed, vec![first, second]);
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_waiting_request_cannot_preempt_running_work() {
    let mut core = core(vec![group("window", 4, 4, Some(4))], 12, 8);
    let first = submit(&mut core, 105, 8, 2);
    let second = submit(&mut core, 106, 8, 2);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    assert_eq!(initial.admissions[0].uuid, first);
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    let next = core.execute_pass(&mut trace, 1.0);
    assert!(next.admissions.is_empty());
    assert_eq!(next.mocker_metrics.vllm_preemptions_total, 0);
    assert_eq!(next.mocker_metrics.kv_cache_used_bytes, Some(0));
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    let admitted = core.execute_pass(&mut trace, 2.0);
    assert_eq!(admitted.admissions[0].uuid, second);
}

#[test]
fn grouped_oversized_chunk_fails_without_partial_allocation_or_endless_preemption() {
    let mut core = core(vec![group("window", 4, 4, Some(4))], 8, 8);
    let id = submit(&mut core, 107, 16, 2);
    let mut trace = TraceCollector::default();
    core.execute_pass(&mut trace, 0.0);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(8));
    let error = core.try_execute_pass(&mut trace, 1.0).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("exceeding the entire 8-byte cache pool")
    );
    assert!(error.to_string().contains("reduce max_num_batched_tokens"));
    // Without a state group the message has no sentence about fixed bytes.
    assert!(!error.to_string().contains("fixed state blocks"));
    assert_eq!(core.state.requests[&id].num_computed_tokens, 8);
    assert_eq!(core.state.requests[&id].sequence.num_allocated_tokens(), 8);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(8));
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 0);
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_state_blocks_exceeding_the_pool_reject_at_admission() {
    // The 16-byte state block alone is larger than the 12-byte pool, so even
    // the one-token forward cannot fit and the request never reaches a forward.
    let mut core = core(vec![group("full", 4, 4, None), state("kda", 16, 1)], 12, 8);
    let id = submit(&mut core, 115, 4, 2);
    let mut trace = TraceCollector::default();
    let pass = core.execute_pass(&mut trace, 0.0);
    assert!(pass.admissions.is_empty());
    assert!(
        pass.output_signals
            .iter()
            .any(|output| output.uuid == id && output.rejected)
    );
    assert_eq!(pass.mocker_metrics.kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_oversized_chunk_names_the_fixed_state_bytes() {
    let mut core = core(
        vec![group("window", 4, 4, Some(4)), state("kda", 2, 1)],
        9,
        8,
    );
    submit(&mut core, 116, 16, 2);
    let mut trace = TraceCollector::default();
    // The one-token forward needs one 4-byte window page and the 2-byte state
    // block, 6 bytes, so admission passes. The first chunk of 8 tokens needs
    // two window pages (8 bytes) and the state block, 10 bytes, above the pool.
    let error = core
        .try_execute_pass(&mut trace, 0.0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("needs 10 bytes for 8 scheduled tokens after 0 computed tokens"));
    assert!(error.contains("exceeding the entire 9-byte cache pool"));
    assert!(error.contains("the fixed state blocks of one request take 2 bytes"));
    assert!(error.contains("reduce max_num_batched_tokens"));
    // The sentence comes before the remedy.
    assert!(error.contains("2 bytes of this pool; reduce max_num_batched_tokens"));
}

#[test]
fn grouped_handoff_commands_fail_before_acquiring_state() {
    let mut core = core(vec![group("window", 4, 4, Some(4))], 24, 8);
    let handoff_id = HandoffId::new(Uuid::from_u128(108));
    let error = core
        .apply_command(SchedulerCommand::ReserveDestination {
            handoff_id,
            request: DirectRequest {
                tokens: vec![0, 1, 2, 3],
                max_output_tokens: 2,
                ..Default::default()
            },
        })
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "grouped KV caches support KV handoff commands only on prefill and decode ranks"
    );
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_aggregated_rank_refuses_source_commands() {
    // An aggregated rank builds no transfer rules, so it cannot price a
    // held source.
    let mut core = core(glm_transfer_groups("0.32.0", false), 128 * GLM_PAGE, 8192);
    let handoff_id = HandoffId::new(Uuid::from_u128(1624));
    for command in [
        SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: source_request(1623, 1_000),
        },
        SchedulerCommand::ReleaseSource { handoff_id },
        SchedulerCommand::CancelSource { handoff_id },
    ] {
        let error = core.apply_command(command).unwrap_err();
        assert_eq!(
            error.to_string(),
            "grouped KV caches support KV handoff commands only on prefill and decode ranks"
        );
    }
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_prefill_source_truncation_limits_the_output_budget() {
    for version in ["0.31.0", "0.32.0"] {
        for planned in [false, true] {
            // A one-token prompt does not truncate and keeps both outputs.
            for (prompt, expected_prompt, expected_outputs) in [(2177, 2176, 1), (1, 1, 2)] {
                let mut core = pd_core(
                    glm_transfer_groups(version, false),
                    128 * GLM_PAGE,
                    8192,
                    WorkerType::Prefill,
                    Some(1.0),
                );
                let handoff_id = HandoffId::new(Uuid::from_u128(1630));
                core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                    handoff_id,
                    request: DirectRequest {
                        max_output_tokens: 2,
                        output_token_ids: planned.then(|| vec![90_000, 90_001]),
                        ..source_request(1629, prompt)
                    },
                })
                .unwrap();
                let request = &core.state.requests[&Uuid::from_u128(1629)];
                assert_eq!(request.sequence.num_input_tokens(), expected_prompt);
                assert_eq!(request.sequence.max_output_tokens(), expected_outputs);
                let first = core.try_execute_hidden_pass(0.0).unwrap();
                assert_eq!(
                    first
                        .lifecycle_events
                        .iter()
                        .any(|event| matches!(event, SchedulerLifecycleEvent::SourceHeld { .. })),
                    expected_outputs == 1,
                    "{version} planned={planned} prompt={prompt}"
                );
                if expected_outputs > 1 {
                    held_timing(&mut core);
                }
                core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
                    .unwrap();
                assert!(core.is_drained());
                assert_eq!(core.kv_manager.grouped().unwrap().active_blocks(), 0);
            }
        }
    }
}

#[test]
fn grouped_prefill_source_rejects_an_empty_prompt_before_registering_the_hold() {
    for version in ["0.31.0", "0.32.0"] {
        let mut core = pd_core(
            glm_transfer_groups(version, false),
            128 * GLM_PAGE,
            8192,
            WorkerType::Prefill,
            Some(1.0),
        );
        let handoff_id = HandoffId::new(Uuid::from_u128(1632));
        let error = core
            .apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request: DirectRequest {
                    max_output_tokens: 0,
                    ..source_request(1631, 0)
                },
            })
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "grouped KV sources require a nonempty prompt"
        );
        assert!(core.is_drained());
        assert!(
            core.try_execute_hidden_pass(0.0)
                .unwrap()
                .lifecycle_events
                .is_empty()
        );
        assert_eq!(core.kv_manager.grouped().unwrap().active_blocks(), 0);
        // The rejected command must leave both IDs available, and a single
        // token still creates a valid source without backoff.
        let timing = hold_grouped_source(&mut core, handoff_id, 1631, 1);
        assert_eq!(timing.full_prompt_tokens, 1);
        assert!(timing.state_bytes > 0);
        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert!(core.is_drained());
    }
}

#[test]
fn grouped_decode_rank_refuses_source_commands() {
    // A decode rank has transfer rules but holds no source: only a prefill
    // rank frees the skipped blocks of a source that it holds.
    for version in ["0.32.0", "0.33.0"] {
        let byte_mode = pd_core(
            glm_transfer_groups(version, false),
            128 * GLM_PAGE,
            8192,
            WorkerType::Decode,
            None,
        );
        for (mode, mut core) in [
            ("byte", byte_mode),
            (
                "block",
                block_mode_decode_core(128 * GLM_PAGE, version, false),
            ),
        ] {
            let handoff_id = HandoffId::new(Uuid::from_u128(1640));
            for command in [
                SchedulerCommand::SubmitHandoffPrefill {
                    handoff_id,
                    request: source_request(1641, 1_000),
                },
                SchedulerCommand::ReleaseSource { handoff_id },
                SchedulerCommand::CancelSource { handoff_id },
            ] {
                let error = core.apply_command(command).unwrap_err();
                assert_eq!(
                    error.to_string(),
                    "a grouped decode rank holds no KV handoff source",
                    "{mode} mode"
                );
            }
            assert!(core.is_drained(), "{mode} mode");
            assert_eq!(grouped_pool(&core).active_blocks(), 0, "{mode} mode");
        }
    }
}

#[test]
fn grouped_prefill_rank_refuses_a_destination() {
    // A prefill rank has transfer rules but loads no handoff: only a decode
    // rank reserves a destination.
    for version in ["0.32.0", "0.33.0"] {
        let byte_mode = pd_core(
            glm_transfer_groups(version, false),
            128 * GLM_PAGE,
            8192,
            WorkerType::Prefill,
            None,
        );
        let mut args = prefix_core(
            glm_transfer_groups(version, false),
            128 * GLM_PAGE,
            8192,
            2176,
            version,
        )
        .args;
        args.worker_type = WorkerType::Prefill;
        for (mode, mut core) in [("byte", byte_mode), ("block", VllmCore::new(args))] {
            let error = core
                .apply_command(SchedulerCommand::ReserveDestination {
                    handoff_id: HandoffId::new(Uuid::from_u128(1642)),
                    request: pd_request(1643, 1_000, 3),
                })
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                "a grouped prefill rank reserves no KV handoff destination",
                "{mode} mode"
            );
            assert!(core.is_drained(), "{mode} mode");
            assert_eq!(grouped_pool(&core).active_blocks(), 0, "{mode} mode");
        }
    }
}

#[test]
fn grouped_prefill_source_keeps_only_the_blocks_inside_its_window() {
    // A prompt of 1,024 tokens in one chunk, with a window of 32 tokens and
    // blocks of 16. The forward keeps the 64 blocks that its first query
    // reads. When a source completes, vLLM frees the blocks that the next
    // query skips (V32 `vllm/v1/core/sched/scheduler.py:3023-3031`):
    // 1,024 - 32 + 1 = 993 tokens, so the 62 blocks below 992 go and 2 stay.
    for version in ["0.32.0", "0.33.0"] {
        let mut window = group("window", 16, 100, Some(32));
        window.prefix_cacheable = Some(true);
        window.transfer_bytes_per_block = Some(100);
        let byte_mode = pd_core(
            vec![window.clone()],
            67 * 100,
            1024,
            WorkerType::Prefill,
            None,
        );
        let mut args = prefix_core(vec![window], 67 * 100, 1024, 16, version).args;
        args.worker_type = WorkerType::Prefill;
        for (mode, mut core) in [("byte", byte_mode), ("block", VllmCore::new(args))] {
            let handoff_id = HandoffId::new(Uuid::from_u128(1626));
            hold_grouped_source(&mut core, handoff_id, 1625, 1024);
            let held = |core: &VllmCore| core.kv_manager.grouped().unwrap().active_blocks();
            assert_eq!(held(&core), 2, "{mode}");
            core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
                .unwrap();
            assert_eq!(held(&core), 0, "{mode}");
            assert!(core.is_drained(), "{mode}");
        }
    }
}

#[test]
fn grouped_prefill_source_truncates_one_token_and_prices_the_payload_in_blocks() {
    // N, the N - 1 tokens that the prefill computes, their attention blocks
    // and the time at 1 GB/s of the bytes of one rank that a live run
    // measured for this prompt on each version: vLLM 0.31.0 and 0.32.0 in
    // `benchmarks/evidence/nixl-pd-transfer-bytes-20261115/`, and vLLM 0.33.0
    // in `benchmarks/evidence/nixl-pd-transfer-bytes-20261117/`.
    for (version, block, fixed, rows) in [
        (
            "0.31.0",
            13_045_120,
            52_120_960,
            [
                (1_000, 999, 1, 65.166080),
                (2_177, 2_176, 1, 65.166080),
                (5_000, 4_999, 3, 91.256320),
            ],
        ),
        (
            "0.32.0",
            21_978_496,
            39_168_128,
            [
                (1_000, 999, 1, 61.146624),
                (2_177, 2_176, 1, 61.146624),
                (5_000, 4_999, 3, 105.103616),
            ],
        ),
        (
            "0.33.0",
            21_978_496,
            39_168_128,
            [
                (1_000, 999, 1, 61.146624),
                (2_177, 2_176, 1, 61.146624),
                (5_000, 4_999, 3, 105.103616),
            ],
        ),
    ] {
        for (prompt, computed, blocks, delay_ms) in rows {
            let mut core = pd_core(
                glm_transfer_groups(version, false),
                128 * GLM_PAGE,
                8192,
                WorkerType::Prefill,
                Some(1.0),
            );
            let captured = Arc::new(CapturedDecodeQueries::default());
            core.args.perf_model = Arc::new(PerfModel::External {
                timing: captured.clone(),
            });
            let handoff_id = HandoffId::new(Uuid::from_u128(1602));
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request: source_request(1601, prompt),
            })
            .unwrap();
            // The request that the source will hold lacks the last prompt token.
            let id = Uuid::from_u128(1601);
            assert_eq!(
                core.state.requests[&id].sequence.num_input_tokens(),
                computed,
                "N {prompt}"
            );
            let timing = held_timing(&mut core);
            assert_eq!(timing.full_prompt_tokens, blocks, "N {prompt}");
            assert_eq!(timing.kv_bytes_per_token, Some(block));
            assert_eq!(timing.state_bytes, fixed);
            assert!(
                timing
                    .delay_ms(0)
                    .is_some_and(|delay| (delay - delay_ms).abs() < 1e-9),
                "N {prompt}: {:?}",
                timing.delay_ms(0)
            );
            // The prefill computes and prices N - 1 tokens in one chunk.
            assert_eq!(
                *captured.1.lock().unwrap(),
                vec![(1, computed, 0)],
                "N {prompt}"
            );
            // The held source keeps its attention blocks, the ring block and the
            // one block of each KDA group.
            assert!(core.source_is_held(handoff_id));
            assert!(!core.state.requests.contains_key(&id));
            assert_eq!(
                core.kv_manager.grouped().unwrap().active_blocks(),
                blocks + 5,
                "N {prompt}"
            );
            core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
                .unwrap();
            assert_eq!(core.kv_manager.grouped().unwrap().used_bytes(), 0);
            assert!(core.is_drained(), "N {prompt}");
        }
    }
}

#[test]
fn grouped_prefill_without_state_layers_keeps_the_whole_prompt() {
    // Without a Mamba state the backoff is 0: the prefill computes all 130
    // tokens, which take three 64-token blocks, the last one partly filled.
    let mut full = group("full", 64, 100, None);
    full.prefix_cacheable = Some(true);
    full.transfer_bytes_per_block = Some(100);
    let mut core = pd_core(vec![full], 1_000, 8192, WorkerType::Prefill, Some(1.0));
    let captured = Arc::new(CapturedDecodeQueries::default());
    core.args.perf_model = Arc::new(PerfModel::External {
        timing: captured.clone(),
    });
    let handoff_id = HandoffId::new(Uuid::from_u128(1604));
    let timing = hold_grouped_source(&mut core, handoff_id, 1603, 130);
    assert_eq!(*captured.1.lock().unwrap(), vec![(1, 130, 0)]);
    assert_eq!(timing.full_prompt_tokens, 3);
    assert_eq!(timing.kv_bytes_per_token, Some(100));
    assert_eq!(timing.state_bytes, 0);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(300));
    // Cancelling a held source frees it as a release does.
    core.apply_command(SchedulerCommand::CancelSource { handoff_id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert!(core.is_drained());
}

#[test]
fn grouped_prefill_source_in_block_mode_frees_its_checkpoint_blocks_at_completion() {
    for version in ["0.32.0", "0.33.0"] {
        let handoff_id = HandoffId::new(Uuid::from_u128(1606));
        // `prefix_core` on `version` as a prefill rank that holds a source of
        // `prompt` tokens. Returns the core, its prefill chunks and the timing.
        let hold = |checkpoint: bool, prompt: usize| {
            let mut args = prefix_core(
                glm_transfer_groups(version, checkpoint),
                128 * GLM_PAGE,
                8192,
                2176,
                version,
            )
            .args;
            args.worker_type = WorkerType::Prefill;
            args.kv_transfer_bandwidth = Some(1.0);
            let mut core = VllmCore::new(args);
            let captured = Arc::new(CapturedDecodeQueries::default());
            core.args.perf_model = Arc::new(PerfModel::External {
                timing: captured.clone(),
            });
            let timing = hold_grouped_source(&mut core, handoff_id, 1605, prompt);
            let chunks = captured.1.lock().unwrap().clone();
            (core, chunks, timing)
        };
        // The block of each KDA group keyed at `at` tokens of the prompt
        // `0..prompt`: active while a request holds it, inactive once only the
        // cache does.
        let kda_keys = |core: &VllmCore, prompt: u32, at: usize| -> Vec<Option<bool>> {
            let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
            lease.configure_prefix_hashes(&(0..prompt).collect::<Vec<u32>>(), 2176);
            let hash = lease.prefix_hash_at(at).unwrap();
            (2..6)
                .map(|g| block_pool(core).key_hit(g, hash).map(|hit| hit.is_active))
                .collect()
        };
        let release = |core: &mut VllmCore| {
            core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
                .unwrap();
            assert_eq!(block_pool(core).active_blocks(), 0);
            assert!(core.is_drained());
        };
        // When a source completes, vLLM frees in each KDA group every block below
        // the state of the last computed token (V32
        // `vllm/v1/core/sched/scheduler.py:3023-3031`,
        // `vllm/v1/core/single_type_kv_cache_manager.py:1613-1641`). So the held
        // source keeps only the running block there. A freed block stays cached.

        // With one FlashKDA checkpoint block per KDA group, the truncated prompt
        // of 8,191 tokens runs in one chunk. Its allocation reaches the end of
        // that prompt, so each KDA group takes a checkpoint block. Finalize keys
        // it at 6,528, the last block boundary below the end. A first allocation
        // from 0 replaces no running block, so no previous block holds that key.
        let (mut core, chunks, timing) = hold(true, 8_192);
        assert_eq!(chunks, vec![(1, 8191, 0)]);
        // The payload is that of byte mode, with the measured bytes of one block
        // of `version` and the fixed bytes computed from them. The checkpoint
        // block does not move.
        assert_eq!(timing.full_prompt_tokens, 4);
        assert_eq!(timing.kv_bytes_per_token, Some(21_978_496));
        assert_eq!(timing.state_bytes, 39_168_128);
        // Four attention blocks, the ring block and the running block of each
        // KDA group. The checkpoint blocks left at completion.
        assert_eq!(block_pool(&core).active_blocks(), 9);
        assert_eq!(kda_keys(&core, 8191, 6528), vec![Some(false); 4]);
        release(&mut core);
        assert_eq!(kda_keys(&core, 8191, 6528), vec![Some(false); 4]);

        // Without checkpoint blocks the split stops at 6,528, where finalize
        // keys the running block. The next chunk replaces it, and the previous
        // block with that key leaves at completion. The block count and the key
        // are the same, so only the chunks tell the two cores apart.
        let (core, chunks, _) = hold(false, 8_192);
        assert_eq!(chunks, vec![(1, 6528, 0), (1, 1663, 6528)]);
        assert_eq!(block_pool(&core).active_blocks(), 9);
        assert_eq!(kda_keys(&core, 8191, 6528), vec![Some(false); 4]);

        // A source of 6,529 tokens ties the checkpoint to the truncated prompt
        // end. Its prompt of 6,528 tokens puts the checkpoint at 4,352. An
        // untruncated prompt of 6,529 tokens would put it at 6,528.
        let (mut core, chunks, timing) = hold(true, 6_529);
        assert_eq!(chunks, vec![(1, 6528, 0)]);
        assert_eq!(timing.full_prompt_tokens, 3);
        // Three attention blocks, the ring block and the running block of each
        // KDA group.
        assert_eq!(block_pool(&core).active_blocks(), 8);
        assert_eq!(kda_keys(&core, 6528, 4352), vec![Some(false); 4]);
        assert_eq!(kda_keys(&core, 6528, 6528), vec![None; 4]);
        release(&mut core);
        assert_eq!(kda_keys(&core, 6528, 4352), vec![Some(false); 4]);
    }
}

#[test]
fn grouped_prefill_source_preempted_in_its_prefill_stays_truncated_once() {
    // Two sources of 4,354 tokens on 17 pages. The split ends the first
    // chunk of the first source at 4,352, which leaves budget to the second
    // source. The last token of the first source then needs one attention
    // block and a new running block in each KDA group, 5 blocks, while 4 are
    // free, so the second source is preempted.
    for version in ["0.32.0", "0.33.0"] {
        let mut args = prefix_core(
            glm_transfer_groups(version, false),
            17 * GLM_PAGE,
            8192,
            2176,
            version,
        )
        .args;
        args.worker_type = WorkerType::Prefill;
        let mut core = VllmCore::new(args);
        let first = HandoffId::new(Uuid::from_u128(1610));
        let second = HandoffId::new(Uuid::from_u128(1612));
        for (handoff_id, request) in [
            (first, source_request(1609, 4_354)),
            (
                second,
                DirectRequest {
                    tokens: (100_000..104_354).collect(),
                    ..source_request(1611, 0)
                },
            ),
        ] {
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request,
            })
            .unwrap();
        }
        assert_eq!(held_timing(&mut core).full_prompt_tokens, 3);
        assert_eq!(core.mocker_metrics().vllm_preemptions_total, 1);
        // The preempted source restarts from 0 on the prompt that its submission
        // truncated, as vLLM's `_p_side_truncated` guard keeps a rescheduled
        // request from losing a second token (V31
        // `nixl/base_scheduler.py:415-423`, V32 `:411-419`).
        let preempted = &core.state.requests[&Uuid::from_u128(1611)];
        assert_eq!(preempted.sequence.num_input_tokens(), 4_353);
        assert_eq!(preempted.num_computed_tokens, 0);
        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id: first })
            .unwrap();
        // 4,353 tokens take 3 attention blocks. A second truncation would give 2.
        assert_eq!(held_timing(&mut core).full_prompt_tokens, 3);
    }
}

/// A destination request as Replay sends it, with its full output budget.
fn pd_request(id: u128, prompt: usize, output: usize) -> DirectRequest {
    DirectRequest {
        max_output_tokens: output,
        ..source_request(id, prompt)
    }
}

/// Reserve a destination and return its transfer units, or `None` while it
/// waits in `pending_destinations`.
fn reserve_grouped(
    core: &mut VllmCore,
    handoff_id: HandoffId,
    request: DirectRequest,
) -> Option<usize> {
    let effects = core
        .apply_command_effects(
            SchedulerCommand::ReserveDestination {
                handoff_id,
                request,
            },
            true,
        )
        .unwrap();
    assert!(matches!(
        effects.result,
        SchedulerCommandResult::DestinationAccepted { .. }
    ));
    reserved_units(&effects.lifecycle_events, handoff_id)
}

fn reserved_units(events: &[SchedulerLifecycleEvent], handoff_id: HandoffId) -> Option<usize> {
    events.iter().find_map(|event| match event {
        SchedulerLifecycleEvent::DestinationReserved {
            handoff_id: reserved,
            transferable_prompt_tokens,
            ..
        } if *reserved == handoff_id => Some(*transferable_prompt_tokens),
        _ => None,
    })
}

fn activate(core: &mut VllmCore, handoff_id: HandoffId) {
    core.apply_command(SchedulerCommand::ActivateDestination { handoff_id })
        .unwrap();
}

/// One pass of a decode rank. Returns the output tokens it emits.
fn step_pd(core: &mut VllmCore, trace: &mut TraceCollector, from_step: &mut usize) -> usize {
    let pass = core.execute_pass(trace, *from_step as f64);
    *from_step += 1;
    for output in &pass.output_signals {
        assert!(!output.rejected, "{} rejected", output.uuid);
    }
    pass.output_signals.len()
}

/// Step until no request is owned by the scheduler state, with the pool never
/// above `peak` blocks. Held and pending destinations may remain. Returns the
/// output tokens emitted.
fn drain_pd(
    core: &mut VllmCore,
    trace: &mut TraceCollector,
    from_step: &mut usize,
    peak: usize,
) -> usize {
    let mut outputs = 0;
    for _ in 0..64 {
        outputs += step_pd(core, trace, from_step);
        assert!(core.kv_manager.grouped().unwrap().active_blocks() <= peak);
        if core.state.requests.is_empty() {
            return outputs;
        }
    }
    panic!("the decode rank did not drain");
}

fn grouped_pool(core: &VllmCore) -> &crate::engine::kv_manager::GroupedKvPool {
    core.kv_manager.grouped().unwrap()
}

/// `prefix_core` as a decode rank with the GLM-5.3-Flash transfer bytes.
fn block_mode_decode_core(capacity: u64, version: &str, checkpoint: bool) -> VllmCore {
    let mut args = prefix_core(
        glm_transfer_groups(version, checkpoint),
        capacity,
        8192,
        2176,
        version,
    )
    .args;
    args.worker_type = WorkerType::Decode;
    VllmCore::new(args)
}

/// The chained hash of the first `at` tokens of the prompt `0..prompt`.
fn prompt_hash(prompt: usize, at: usize) -> crate::engine::common::hashing::SequenceHash {
    let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
    lease.configure_prefix_hashes(&(0..prompt as u32).collect::<Vec<u32>>(), 2176);
    lease.prefix_hash_at(at).unwrap()
}

#[test]
fn grouped_prefill_source_takes_no_copy_of_a_state_published_at_its_end() {
    // With a hash block below the state block, vLLM 0.31.0 caches the state
    // at the prompt's partial tail, `prompt // hash * hash` (V31
    // single:2094-2102). The prefill of a source stops one token short, so
    // it ends at that tail when the computed tokens are a multiple of the
    // hash block. vLLM copies such a block only when it allocates for the
    // request again (V31 single:1891-1936, V32 :1855-1900). A source
    // allocates no more: vLLM frees its skipped blocks and later the
    // request, which drops the producer (V31 sched:2650-2658, :2966-2974,
    // single:562-579, :1971-1977). So the held source takes no copy block,
    // and a pool with no free block holds it too. On vLLM 0.32.0 the tail
    // lies before the end of the prefill (V32 single:2069-2073), and its last
    // chunk takes the copy.
    let glm = |version: &str, pages: u64| {
        prefix_core(
            glm_transfer_groups(version, false),
            pages * GLM_PAGE,
            8192,
            16,
            version,
        )
        .args
    };
    let full_and_state = |version: &str, pages: u64| {
        let mut full = group("full", 32, 100, None);
        full.prefix_cacheable = Some(true);
        full.transfer_bytes_per_block = Some(100);
        let mut recurrent = state("state", 100, 1);
        recurrent.block_size_tokens = 32;
        recurrent.prefix_cacheable = Some(true);
        recurrent.transfer_bytes_per_block = Some(100);
        recurrent.prefill_checkpoint_blocks = Some(0);
        prefix_core(vec![full, recurrent], pages * 100, 8192, 4, version).args
    };
    // GLM-5.3-Flash groups at a hash block of 16 hold one attention block,
    // the ring and four KDA states. Six pages leave no free block on vLLM
    // 0.31.0. Attention and state blocks of 32 at a hash block of 4 hold two
    // attention blocks and the running state block.
    let cases = [
        ("0.31.0", glm("0.31.0", 128), 6),
        ("0.31.0", glm("0.31.0", 6), 6),
        ("0.31.0", full_and_state("0.31.0", 16), 3),
        ("0.32.0", glm("0.32.0", 128), 6),
        ("0.32.0", full_and_state("0.32.0", 16), 3),
        ("0.33.0", glm("0.33.0", 128), 6),
        ("0.33.0", full_and_state("0.33.0", 16), 3),
    ];
    for (index, (version, mut args, held)) in cases.into_iter().enumerate() {
        args.worker_type = WorkerType::Prefill;
        args.kv_transfer_bandwidth = Some(1.0);
        let mut core = VllmCore::new(args);
        let handoff_id = HandoffId::new(Uuid::from_u128(1_871));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: source_request(1_870, 49),
        })
        .unwrap();
        let first = core.try_execute_hidden_pass(0.0).unwrap();
        let state_groups: Vec<_> = core
            .args
            .kv_cache_groups
            .iter()
            .enumerate()
            .filter(|(_, group)| {
                group.kind == FpmCacheKind::State && group.prefix_cacheable == Some(true)
            })
            .map(|(index, _)| index as u16)
            .collect();
        if version == "0.32.0" {
            assert!(first.lifecycle_events.is_empty(), "{version} case {index}");
            let tail = 48 - core.args.prefix_match_unit.unwrap();
            for _ in 0..8 {
                if core.state.requests[&Uuid::from_u128(1_870)].num_computed_tokens == tail {
                    break;
                }
                let pass = core.try_execute_hidden_pass(0.0).unwrap();
                assert!(pass.lifecycle_events.is_empty(), "{version} case {index}");
            }
            assert_eq!(
                core.state.requests[&Uuid::from_u128(1_870)].num_computed_tokens,
                tail
            );
            let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
            lease.configure_prefix_hashes(
                &(0..48).collect::<Vec<u32>>(),
                core.args.prefix_match_unit.unwrap(),
            );
            let hash = lease.prefix_hash_at(tail).unwrap();
            for &group in &state_groups {
                assert!(block_pool(&core).key_hit(group, hash).unwrap().is_active);
            }
            block_pool(&core).reset_peak_active_blocks();
            held_timing(&mut core);
            // The final chunk needs a new writable block in every state
            // group while the published blocks are still active. This peak
            // observes allocation before completion trims the old blocks.
            assert_eq!(
                block_pool(&core).peak_active_blocks(),
                held + state_groups.len(),
                "{version} case {index}"
            );
            for &group in &state_groups {
                assert!(!block_pool(&core).key_hit(group, hash).unwrap().is_active);
            }
        } else if index < 2 {
            assert!(
                first
                    .lifecycle_events
                    .iter()
                    .any(|event| matches!(event, SchedulerLifecycleEvent::SourceHeld { .. })),
                "{version} case {index}"
            );
            assert_eq!(
                block_pool(&core).peak_active_blocks(),
                held,
                "{version} case {index}"
            );
        } else {
            held_timing(&mut core);
        }
        assert_eq!(
            block_pool(&core).active_blocks(),
            held,
            "{version} case {index}"
        );
    }
}

/// A prefill or decode rank with a full-attention group and a sliding-window
/// group of 4-token blocks, a window of 8 tokens and room for 40 blocks, in
/// block mode on `version` or in byte mode.
fn window_pd_core(
    block_mode: bool,
    version: &str,
    worker_type: WorkerType,
    max_concurrent_batches: usize,
) -> VllmCore {
    let mut full = group("full", 4, 100, None);
    full.prefix_cacheable = Some(true);
    full.transfer_bytes_per_block = Some(100);
    let mut window = group("window", 4, 100, Some(8));
    window.prefix_cacheable = Some(true);
    window.transfer_bytes_per_block = Some(100);
    let groups = vec![full, window];
    let mut args = if block_mode {
        prefix_core(groups, 40 * 100, 64, 4, version).args
    } else {
        pd_core(groups, 40 * 100, 64, worker_type, None).args
    };
    args.worker_type = worker_type;
    args.max_concurrent_batches = max_concurrent_batches;
    VllmCore::new(args)
}

/// The rank modes of `window_pd_core`: byte mode and block mode on the three
/// versions.
const WINDOW_PD_MODES: [(bool, &str); 4] = [
    (false, "0.32.0"),
    (true, "0.31.0"),
    (true, "0.32.0"),
    (true, "0.33.0"),
];

/// The blocks that the rank holds for every request but `busy`, which is
/// running.
fn blocks_besides(core: &VllmCore, busy: Uuid) -> usize {
    core.kv_manager.grouped().unwrap().active_blocks() - core.request_block_count(busy)
}

#[test]
fn grouped_prefill_rank_frees_skipped_blocks_after_the_next_pass_with_async_scheduling() {
    // X (4 tokens, 6 outputs) keeps a pass in flight. S, a source of 20
    // tokens, and L, a local request of 20 tokens with one output, each
    // prefill in pass 0 and end there with 5 full-attention and 5 window
    // blocks. The next query of 20 tokens reads the window from token
    // 20 - 8 + 1 = 13, so the 3 window blocks of tokens 0 to 11 are skipped.
    // vLLM frees them in `_connector_finished`, which `_free_request` runs
    // before it frees the other blocks or hands them to the connector (V31
    // `sched:2650-2679`, `:2966-2974`, V32 `sched:2714-2744`, `:3023-3031`).
    // `update_from_output` calls it (V31 `sched:2171`, V32 `sched:2230`), and
    // with async scheduling the engine forms the next step before it processes
    // these outputs (V31 `engine/core.py:697`, `:740-755`, V32
    // `engine/core.py:725`, `:768-783`). So S holds 7 blocks after pass 0
    // without async scheduling, and 10 with it until pass 1 is formed, then
    // 7. L holds 0 after pass 0 without async scheduling, and 10 with it, then
    // 0.
    for (block_mode, version) in WINDOW_PD_MODES {
        for (max_concurrent_batches, after_finish) in [(1, [7, 0]), (2, [10, 10])] {
            let label = format!("{block_mode} {version} {max_concurrent_batches}");
            let mut core = window_pd_core(
                block_mode,
                version,
                WorkerType::Prefill,
                max_concurrent_batches,
            );
            let mut now = 0.0;
            let x = submit_tokens(&mut core, 1, (1000..1004).collect(), 6);
            let handoff_id = HandoffId::new(Uuid::from_u128(1002));
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request: DirectRequest {
                    tokens: (2000..2020).collect(),
                    ..source_request(2, 20)
                },
            })
            .unwrap();
            let held = async_step(&mut core, &mut now);
            assert!(
                held.lifecycle_events
                    .iter()
                    .any(|event| matches!(event, SchedulerLifecycleEvent::SourceHeld { .. }))
            );
            assert_eq!(blocks_besides(&core, x), after_finish[0], "{label}");
            async_step(&mut core, &mut now);
            assert_eq!(blocks_besides(&core, x), 7, "{label}");

            let mut core = window_pd_core(
                block_mode,
                version,
                WorkerType::Prefill,
                max_concurrent_batches,
            );
            let mut now = 0.0;
            let x = submit_tokens(&mut core, 1, (1000..1004).collect(), 6);
            let l = submit_tokens(&mut core, 3, (3000..3020).collect(), 1);
            let ended = async_step(&mut core, &mut now);
            assert!(
                ended
                    .output_signals
                    .iter()
                    .any(|signal| signal.uuid == l && signal.completed)
            );
            assert_eq!(blocks_besides(&core, x), after_finish[1], "{label}");
            async_step(&mut core, &mut now);
            assert_eq!(blocks_besides(&core, x), 0, "{label}");
        }
    }
}

#[test]
fn grouped_prefill_source_released_before_its_deferred_free_frees_every_block_once() {
    // As above, with async scheduling: S ends in pass 0 and holds 10 blocks,
    // of which 3 are skipped, until pass 1 is formed. A release or a cancel of
    // the source in between frees all 10. The deferred free of pass 1 then
    // finds no held source and frees nothing more.
    for (block_mode, version) in WINDOW_PD_MODES {
        for cancel in [false, true] {
            let label = format!("{block_mode} {version} {cancel}");
            let mut core = window_pd_core(block_mode, version, WorkerType::Prefill, 2);
            let mut now = 0.0;
            let x = submit_tokens(&mut core, 1, (1000..1004).collect(), 6);
            let handoff_id = HandoffId::new(Uuid::from_u128(1002));
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request: DirectRequest {
                    tokens: (2000..2020).collect(),
                    ..source_request(2, 20)
                },
            })
            .unwrap();
            async_step(&mut core, &mut now);
            assert_eq!(blocks_besides(&core, x), 10, "{label}");
            let command = if cancel {
                SchedulerCommand::CancelSource { handoff_id }
            } else {
                SchedulerCommand::ReleaseSource { handoff_id }
            };
            core.apply_command(command).unwrap();
            assert_eq!(blocks_besides(&core, x), 0, "{label}");
            async_step(&mut core, &mut now);
            assert_eq!(blocks_besides(&core, x), 0, "{label}");
            drain_async(&mut core, &mut now, &label);
            assert_eq!(
                core.kv_manager.grouped().unwrap().active_blocks(),
                0,
                "{label}"
            );
        }
    }
}

/// The cached key of `group` at `end` tokens of the prompt `2000..2020`.
fn source_key(core: &VllmCore, group: u16, end: usize) -> bool {
    let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
    lease.configure_prefix_hashes(&(2000..2020).collect::<Vec<u32>>(), 4);
    block_pool(core)
        .key_hit(group, lease.prefix_hash_at(end).unwrap())
        .is_some()
}

/// A block-mode prefill rank of `blocks` blocks with a full-attention group
/// and a window of 8 tokens, at 4-token blocks and a hash block of 4.
fn window_prefill_core(version: &str, blocks: u64, max_concurrent_batches: usize) -> VllmCore {
    let mut full = group("full", 4, 100, None);
    full.prefix_cacheable = Some(true);
    full.transfer_bytes_per_block = Some(100);
    let mut window = group("window", 4, 100, Some(8));
    window.prefix_cacheable = Some(true);
    window.transfer_bytes_per_block = Some(100);
    let mut args = prefix_core(vec![full, window], blocks * 100, 64, 4, version).args;
    args.worker_type = WorkerType::Prefill;
    args.max_concurrent_batches = max_concurrent_batches;
    VllmCore::new(args)
}

#[test]
fn grouped_prefill_source_that_leaves_before_its_deferred_free_frees_its_skipped_blocks_first() {
    // A rank of 14 blocks. X (3 tokens, 8 outputs) and the source S (20
    // tokens) fill 12 blocks in pass 0, and S ends there with 5
    // full-attention and 5 window blocks, of which the 3 window blocks below
    // 20 - 8 + 1 = 13 tokens are skipped. vLLM frees the skipped blocks in
    // `_connector_finished` when it processes the outputs of the step (V31
    // `sched:2171`, `:2650-2679`, `:2966-2974`, V32 `sched:2230`,
    // `:2714-2744`, `:3023-3031`), and the rest when the transfer ends (V31
    // `sched:3155-3158`, V32 `sched:3232-3235`). So the skipped blocks enter
    // the free queue first, whether the release or the cancel of S comes
    // before or after the deferred free of async scheduling. The freed
    // blocks keep their keys, and the pool evicts them in the order of the
    // free. X's growth and W (8 tokens, submitted before pass 2) then evict
    // one cached block of S: the window block of tokens 8 to 11. The
    // full-attention blocks keep their keys. A later prompt that starts with
    // S's 20 tokens then hits nothing: a hit at 16 tokens needs the window
    // blocks of tokens 8 to 15. Freed in group order, a full-attention block
    // goes instead, the window block stays, and that prompt hits 16 tokens.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for cancel in [false, true] {
            // One batch, the source leaves after pass 0; async scheduling,
            // after pass 1 (the deferred free first) and after pass 0 (before
            // it).
            for (max_concurrent_batches, leaves_after) in [(1, 0), (2, 1), (2, 0)] {
                let label = format!("{version} {cancel} {max_concurrent_batches} {leaves_after}");
                let mut core = window_prefill_core(version, 14, max_concurrent_batches);
                let mut now = 0.0;
                submit_tokens(&mut core, 1, (1000..1003).collect(), 8);
                let handoff_id = HandoffId::new(Uuid::from_u128(1002));
                core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                    handoff_id,
                    request: DirectRequest {
                        tokens: (2000..2020).collect(),
                        ..source_request(2, 20)
                    },
                })
                .unwrap();
                for pass in 0..6 {
                    if pass == 2 {
                        submit_tokens(&mut core, 3, (3000..3008).collect(), 1);
                    }
                    async_step(&mut core, &mut now);
                    if pass == leaves_after {
                        let command = if cancel {
                            SchedulerCommand::CancelSource { handoff_id }
                        } else {
                            SchedulerCommand::ReleaseSource { handoff_id }
                        };
                        core.apply_command(command).unwrap();
                    }
                }
                let full_keys = [4, 8, 12, 16, 20].map(|end| source_key(&core, 0, end));
                assert_eq!(full_keys, [true; 5], "{label}");
                assert_eq!(
                    [source_key(&core, 1, 12), source_key(&core, 1, 16)],
                    [false, true],
                    "{label}"
                );
                let mut probe = waiting_probe(&core, 7, (2000..2024).collect());
                assert_eq!(waiting_hit(&core, &mut probe), 0, "{label}");
            }
        }
    }
}

#[test]
fn grouped_prefill_rank_evicts_the_skipped_blocks_of_a_request_that_it_does_not_hold_first() {
    // A rank of 12 blocks. X (3 tokens, 8 outputs) and L, a local request of
    // 20 tokens with one output that the rank does not hold, fill it in pass
    // 0, and L ends there. vLLM frees L's 3 skipped window blocks below 13
    // tokens first, in `_connector_finished`, and then the rest, each group
    // tail first, in the same `_free_request` (V31 `sched:2650-2679`,
    // `:2966-2974`, `coord:387-395`, `single:581-589`, V32 `sched:2714-2744`,
    // `:3023-3031`, `coord:400-408`, `single:581-589`), with async scheduling
    // after it has scheduled step 1. X's growth and W (8 tokens, submitted
    // before pass 2) then evict three cached blocks of L: the window block of
    // tokens 8 to 11, and then the full-attention blocks of tokens 16 to 19
    // and 12 to 15. The full-attention block of tokens 8 to 11 stays. Freed
    // in group order, the full-attention blocks of tokens 8 to 19 would go
    // and the window block would stay.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for max_concurrent_batches in [1, 2] {
            let label = format!("{version} {max_concurrent_batches}");
            let mut core = window_prefill_core(version, 12, max_concurrent_batches);
            let mut now = 0.0;
            submit_tokens(&mut core, 1, (1000..1003).collect(), 8);
            submit_tokens(&mut core, 2, (2000..2020).collect(), 1);
            for pass in 0..6 {
                if pass == 2 {
                    submit_tokens(&mut core, 3, (3000..3008).collect(), 1);
                }
                async_step(&mut core, &mut now);
            }
            assert_eq!(
                [
                    source_key(&core, 1, 12),
                    source_key(&core, 0, 12),
                    source_key(&core, 0, 16),
                    source_key(&core, 0, 20)
                ],
                [false, true, false, false],
                "{label}"
            );
        }
    }
}

/// Whether `group` of the rank holds the key of RP's first `end` tokens, at
/// a hash block of 1,088 tokens.
fn rp_key(core: &VllmCore, group: u16, end: usize) -> bool {
    let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
    lease.configure_prefix_hashes(&(44_000_000..44_004_400).collect::<Vec<u32>>(), 1088);
    block_pool(core)
        .key_hit(group, lease.prefix_hash_at(end).unwrap())
        .is_some()
}

#[test]
fn grouped_prefill_rank_frees_the_boundary_hit_block_of_an_ending_request_with_its_skipped_blocks()
{
    // A GLM prefill rank of 18 blocks with transfer rules and a hash block of
    // 1,088 tokens. RP (4,400 tokens) runs and ends. X (100 tokens, 8
    // outputs) keeps a pass in flight. R, RP's tokens and 300 more with one
    // output, hits 4,352 tokens, two whole blocks, in pass 0 and ends there.
    // In that pass R holds 12 blocks: 2 full-attention hit blocks and a new
    // one, the ring block, and the cached hit block and a running block in
    // each of the four state groups. A state group skips every token before
    // the last one, so the state hit blocks are skipped blocks of R. vLLM
    // frees the state hit blocks in `_connector_finished`, with the other
    // skipped blocks, and the rest after them, each group tail first, when
    // it processes the outputs of the pass (V31 `sched:2650-2679`,
    // `:2966-2974`, `single:1626-1645`, `coord:387-395`, V32
    // `sched:2714-2744`, `:3023-3031`, `single:1592-1611`, `coord:400-408`),
    // with async scheduling after it has scheduled the next step. So the
    // free queue holds the four state hit blocks before R's full-attention
    // blocks of 2,176 and 4,352 tokens. W (10,980 new tokens) then takes 10
    // blocks with 6 free, and the pool evicts the four state hit blocks and
    // keeps the two full-attention blocks. Freed with R's other blocks in
    // group order, the full-attention blocks would go first.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for max_concurrent_batches in [1, 2] {
            let label = format!("{version} {max_concurrent_batches}");
            let mut args = pool_args(version, 18, max_concurrent_batches, 16_384);
            args.max_num_batched_tokens = Some(16_384);
            args.kv_cache_groups = glm_transfer_groups(version, false);
            args.worker_type = WorkerType::Prefill;
            args.kv_transfer_bandwidth = Some(1.0);
            let mut core = VllmCore::new(args);
            let mut now = 0.0;
            let rp: Vec<u32> = (44_000_000..44_004_400).collect();
            submit_tokens(&mut core, 1, rp.clone(), 1);
            drain_async(&mut core, &mut now, &label);
            submit_tokens(&mut core, 2, (45_000_000..45_000_100).collect(), 8);
            let mut tokens = rp;
            tokens.extend(44_100_000..44_100_300);
            let r = submit_tokens(&mut core, 3, tokens, 1);
            block_pool(&core).reset_peak_active_blocks();
            let pass = async_step(&mut core, &mut now);
            let admission = pass
                .admissions
                .iter()
                .find(|admission| admission.uuid == r)
                .expect("R starts in pass 0");
            assert_eq!(admission.reused_input_tokens, 4352, "{label}");
            assert!(
                pass.output_signals
                    .iter()
                    .any(|output| output.uuid == r && output.completed),
                "{label}"
            );
            // X's 6 blocks and R's 12.
            assert_eq!(block_pool(&core).peak_active_blocks(), 18, "{label}");
            async_step(&mut core, &mut now);
            assert_eq!(block_pool(&core).active_blocks(), 6, "{label}");
            assert_eq!(block_pool(&core).inactive_cached_blocks(), 6, "{label}");
            let w = submit_tokens(&mut core, 4, (46_000_000..46_010_980).collect(), 1);
            let mut admitted = false;
            for _ in 0..3 {
                let pass = async_step(&mut core, &mut now);
                if pass.admissions.iter().any(|admission| admission.uuid == w) {
                    admitted = true;
                    break;
                }
            }
            assert!(admitted, "{label}");
            assert_eq!(block_pool(&core).active_blocks(), 16, "{label}");
            assert_eq!(
                [
                    rp_key(&core, 0, 2176),
                    rp_key(&core, 0, 4352),
                    rp_key(&core, 2, 4352),
                    rp_key(&core, 3, 4352),
                    rp_key(&core, 4, 4352),
                    rp_key(&core, 5, 4352),
                ],
                [true, true, false, false, false, false],
                "{label}"
            );
        }
    }
}

#[test]
fn grouped_prefill_source_with_the_request_id_of_a_cancelled_source_holds_its_own_blocks() {
    // With async scheduling, S (20 tokens) ends in pass 0 while W (12
    // tokens, 6 blocks) waits on a rank of 14 blocks, and S is cancelled at
    // the boundary, before its deferred free. A second source T of 12 tokens
    // under a new handoff ID then joins pass 1 and ends there with 3
    // full-attention and 3 window blocks, of which the window block below
    // 12 - 8 + 1 = 5 tokens is skipped. T frees that block in its own
    // deferred free, after pass 2 is formed. So T holds 6 blocks after pass 1
    // and 5 after pass 2, also when it takes S's request ID: the cancel of S
    // drops S's queued deferred free.
    for (block_mode, version) in WINDOW_PD_MODES {
        for second_id in [2, 4] {
            let label = format!("{block_mode} {version} {second_id}");
            let mut full = group("full", 4, 100, None);
            full.prefix_cacheable = Some(true);
            full.transfer_bytes_per_block = Some(100);
            let mut window = group("window", 4, 100, Some(8));
            window.prefix_cacheable = Some(true);
            window.transfer_bytes_per_block = Some(100);
            let groups = vec![full, window];
            let mut args = if block_mode {
                prefix_core(groups, 14 * 100, 64, 4, version).args
            } else {
                pd_core(groups, 14 * 100, 64, WorkerType::Prefill, None).args
            };
            args.worker_type = WorkerType::Prefill;
            args.max_concurrent_batches = 2;
            let mut core = VllmCore::new(args);
            let mut now = 0.0;
            let first = HandoffId::new(Uuid::from_u128(1002));
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id: first,
                request: DirectRequest {
                    tokens: (2000..2020).collect(),
                    ..source_request(2, 20)
                },
            })
            .unwrap();
            let w = submit_tokens(&mut core, 9, (9000..9012).collect(), 6);
            async_step(&mut core, &mut now);
            core.apply_command(SchedulerCommand::CancelSource { handoff_id: first })
                .unwrap();
            let second = HandoffId::new(Uuid::from_u128(1003));
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id: second,
                request: DirectRequest {
                    tokens: (5000..5012).collect(),
                    ..source_request(second_id, 12)
                },
            })
            .unwrap();
            let pass1 = async_step(&mut core, &mut now);
            assert!(pass1.lifecycle_events.iter().any(|event| matches!(
                event,
                SchedulerLifecycleEvent::SourceHeld { handoff_id, .. } if *handoff_id == second
            )));
            let held = |core: &VllmCore| {
                core.kv_manager.grouped().unwrap().active_blocks() - core.request_block_count(w)
            };
            assert_eq!(held(&core), 6, "{label}");
            async_step(&mut core, &mut now);
            assert_eq!(held(&core), 5, "{label}");
        }
    }
}

#[test]
fn grouped_destination_reserves_every_group_and_releases_on_cancel() {
    let mut core = pd_core(
        glm_transfer_groups("0.32.0", false),
        128 * GLM_PAGE,
        8192,
        WorkerType::Decode,
        None,
    );
    let handoff_id = HandoffId::new(Uuid::from_u128(1608));
    // The 999 transferred tokens fill one attention block, so one unit moves.
    assert_eq!(
        reserve_grouped(&mut core, handoff_id, pd_request(1607, 1_000, 3)),
        Some(1)
    );
    // One attention block, one ring block and four state blocks.
    assert_eq!(grouped_pool(&core).used_bytes(), 6 * GLM_PAGE);
    assert_eq!(grouped_pool(&core).active_blocks(), 6);
    assert_eq!(core.destination_block_count(handoff_id), 6);
    core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
        .unwrap();
    assert_eq!(grouped_pool(&core).used_bytes(), 0);
    assert!(core.is_drained());

    assert_eq!(
        reserve_grouped(&mut core, handoff_id, pd_request(1607, 1_000, 3)),
        Some(1)
    );
    activate(&mut core, handoff_id);
    let request = &core.state.requests[&Uuid::from_u128(1607)];
    assert_eq!(request.num_computed_tokens, 999);
    assert_eq!(request.recompute_tokens, 1);
    assert_eq!(request.sequence.num_allocated_tokens(), 999);
    assert_eq!(grouped_pool(&core).used_bytes(), 6 * GLM_PAGE);
}

#[test]
fn grouped_hybrid_destination_recomputes_the_last_prompt_token() {
    // N, the blocks after the first step and the blocks after activation.
    // 1,000 tokens fit one attention block. The last token of 2,177 opens a
    // second one.
    for (prompt, after_first_step) in [(1_000, 6), (2_177, 7)] {
        let mut core = pd_core(
            glm_transfer_groups("0.32.0", false),
            128 * GLM_PAGE,
            8192,
            WorkerType::Decode,
            None,
        );
        let captured = Arc::new(CapturedDecodeQueries::default());
        core.args.perf_model = Arc::new(PerfModel::External {
            timing: captured.clone(),
        });
        let handoff_id = HandoffId::new(Uuid::from_u128(1610));
        let id = Uuid::from_u128(1609);
        assert_eq!(
            reserve_grouped(&mut core, handoff_id, pd_request(1609, prompt, 3)),
            Some(1),
            "N {prompt}"
        );
        activate(&mut core, handoff_id);
        assert_eq!(grouped_pool(&core).active_blocks(), 6, "N {prompt}");
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // The first pass computes the last prompt token and emits the first
        // output token. It is priced as one decode query of the prompt
        // length: the decode rank prices no prefill.
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.output_signals.len(), 1, "N {prompt}");
        assert_eq!(pass.output_signals[0].uuid, id);
        // The admission reports no reused tokens, as a linear handoff does.
        assert_eq!(
            pass.admissions
                .iter()
                .map(|admission| (admission.uuid, admission.reused_input_tokens))
                .collect::<Vec<_>>(),
            vec![(id, 0)],
            "N {prompt}"
        );
        assert_eq!(
            *captured.0.lock().unwrap(),
            vec![(prompt, prompt, prompt)],
            "N {prompt}"
        );
        assert!(captured.1.lock().unwrap().is_empty(), "N {prompt}");
        assert_eq!(
            grouped_pool(&core).active_blocks(),
            after_first_step,
            "N {prompt}"
        );
        assert_eq!(
            drain_pd(&mut core, &mut trace, &mut step, after_first_step),
            2,
            "N {prompt}"
        );
        assert_eq!(grouped_pool(&core).used_bytes(), 0);
        assert!(core.is_drained(), "N {prompt}");
    }
}

#[test]
fn grouped_hybrid_destination_preempted_after_activation_recomputes_its_prompt() {
    let mut core = pd_core(
        glm_transfer_groups("0.32.0", false),
        128 * GLM_PAGE,
        8192,
        WorkerType::Decode,
        None,
    );
    let handoff_id = HandoffId::new(Uuid::from_u128(1636));
    let id = Uuid::from_u128(1635);
    assert_eq!(
        reserve_grouped(&mut core, handoff_id, pd_request(1635, 1_000, 3)),
        Some(1)
    );
    activate(&mut core, handoff_id);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    // Preemption frees the grouped lease and forgets the handoff: the prompt
    // is recomputed from 0 and no token is left to recompute, so the request
    // is not a prebuilt prompt any more.
    assert!(core.policy_preempt(0.0).is_some());
    let request = &core.state.requests[&id];
    assert_eq!(request.num_computed_tokens, 0);
    assert_eq!(request.recompute_tokens, 0);
    assert_eq!(request.sequence.num_allocated_tokens(), 0);
    assert_eq!(grouped_pool(&core).used_bytes(), 0);
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 1);
    // The re-admission prefills the whole prompt and emits the remaining
    // tokens.
    assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 6), 2);
    assert_eq!(grouped_pool(&core).used_bytes(), 0);
    assert!(core.is_drained());
}

#[test]
fn grouped_decode_rank_readmits_a_held_request_when_the_bytes_in_use_fall_at_one_time() {
    // A decode rank of 700 bytes in byte mode, with a budget of 4 tokens, a
    // full-attention group of 4-token blocks of 100 bytes and a group with a
    // window of 5 tokens and 2-token blocks of 1 byte. A destination of 10
    // tokens that is not active holds 3 full blocks and the window blocks 3
    // and 4: 302 bytes in 5 blocks.
    let mut groups = vec![group("full", 4, 100, None), group("window", 2, 1, Some(5))];
    for group in &mut groups {
        group.prefix_cacheable = Some(true);
        group.transfer_bytes_per_block = Some(group.page_size_bytes);
    }
    let mut core = pd_core(groups, 700, 4, WorkerType::Decode, None);
    let usage = |core: &VllmCore| {
        let pool = grouped_pool(core);
        (pool.used_bytes(), pool.active_blocks())
    };
    let blocker = HandoffId::new(Uuid::from_u128(1638));
    assert_eq!(
        reserve_grouped(&mut core, blocker, pd_request(1637, 10, 1)),
        Some(3)
    );
    assert_eq!(usage(&core), (302, 5));
    // A local prompt of 13 tokens computes 4 tokens in each pass, at 0 ms.
    // Its last token needs 4 full blocks and the window blocks 4 to 6, 403
    // bytes, and 398 are free. So it preempts itself in pass 4, is admitted
    // again in pass 5 and preempts itself in pass 8 with the same pool. From
    // pass 9 on it waits.
    let local = submit(&mut core, 1639, 13, 1);
    let mut trace = TraceCollector::default();
    let mut admissions = Vec::new();
    for pass in 1..=12 {
        let result = core.execute_pass(&mut trace, 0.0);
        assert_eq!(result.end_ms, 0.0, "pass {pass}");
        admissions.extend(result.admissions.iter().map(|_| pass));
    }
    assert_eq!(admissions, [1, 5]);
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
    assert_eq!(usage(&core), (302, 5));
    // The destination is cancelled, and a destination of 7 tokens is
    // reserved at the same time. It holds 2 full blocks and the window
    // blocks 1 to 3: 203 bytes, again in 5 blocks. So 497 bytes are free,
    // and the prompt fits. The next pass at that time admits it, and its
    // fourth pass computes the last token and emits the output token.
    core.apply_command(SchedulerCommand::CancelDestination {
        handoff_id: blocker,
    })
    .unwrap();
    let other = HandoffId::new(Uuid::from_u128(1641));
    assert_eq!(
        reserve_grouped(&mut core, other, pd_request(1640, 7, 1)),
        Some(2)
    );
    assert_eq!(usage(&core), (203, 5));
    let result = core.execute_pass(&mut trace, 0.0);
    assert_eq!(
        result
            .admissions
            .iter()
            .map(|admission| admission.uuid)
            .collect::<Vec<_>>(),
        [local]
    );
    assert_eq!(usage(&core), (305, 8));
    let outputs: Vec<usize> = (0..3)
        .map(|_| core.execute_pass(&mut trace, 0.0).output_signals.len())
        .collect();
    assert_eq!(outputs, [0, 0, 1]);
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
    assert!(core.state.requests.is_empty());
    assert_eq!(usage(&core), (203, 5));
}

#[test]
fn grouped_decode_rank_readmits_a_held_request_when_a_load_activates_at_that_time() {
    // A decode rank of 627 bytes in byte mode with a budget of 7 tokens: a
    // full-attention group of 4-token blocks of 100 bytes, a group with a
    // window of 5 tokens and 2-token blocks of 1 byte, and a state group of
    // 1 block of 10 bytes, with which a load computes its last prompt token
    // again. A destination of 6 tokens holds its 5 loaded tokens in 2 full
    // blocks, the window blocks 0 to 2 and the state block: 213 bytes in 6
    // blocks, so 414 bytes are free.
    let mut groups = vec![
        group("full", 4, 100, None),
        group("window", 2, 1, Some(5)),
        state("state", 10, 1),
    ];
    for group in &mut groups {
        group.prefix_cacheable = Some(true);
        group.transfer_bytes_per_block = Some(group.page_size_bytes);
    }
    let mut core = pd_core(groups, 627, 7, WorkerType::Decode, None);
    let usage = |core: &VllmCore| {
        let pool = grouped_pool(core);
        (pool.used_bytes(), pool.active_blocks())
    };
    let handoff_id = HandoffId::new(Uuid::from_u128(1643));
    let load = Uuid::from_u128(1642);
    assert_eq!(
        reserve_grouped(&mut core, handoff_id, pd_request(1642, 6, 8)),
        Some(2)
    );
    assert_eq!(usage(&core), (213, 6));
    // A local prompt of 13 tokens computes 7 tokens in pass 1. Its other 6
    // tokens need 4 full blocks, the window blocks 1 to 6 and the state
    // block: 416 bytes. So it preempts itself in pass 2 and, with the same
    // pool, in pass 4, and then it waits.
    let local = submit(&mut core, 1644, 13, 1);
    let mut trace = TraceCollector::default();
    let mut admissions = Vec::new();
    for pass in 1..=6 {
        let result = core.execute_pass(&mut trace, 0.0);
        admissions.extend(result.admissions.iter().map(|_| pass));
    }
    assert_eq!(admissions, [1, 3]);
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
    // The destination activates at the same time. Its first pass computes
    // its sixth token inside the blocks that it holds, so the pool stays as
    // it is. It runs before the local prompt and takes 1 token of the
    // budget. The prompt then computes 6 tokens in a pass, which need 213
    // and 315 bytes, and its last token needs 4 full blocks, the window
    // blocks 4 to 6 and the state block: 413 bytes. So the prompt fits now.
    // The pass of the activation admits it, and its third pass ends it.
    activate(&mut core, handoff_id);
    assert_eq!(usage(&core), (213, 6));
    let result = core.execute_pass(&mut trace, 0.0);
    assert_eq!(
        result
            .admissions
            .iter()
            .map(|admission| admission.uuid)
            .collect::<Vec<_>>(),
        [load, local]
    );
    assert_eq!(usage(&core), (426, 12));
    core.execute_pass(&mut trace, 0.0);
    assert_eq!(usage(&core), (528, 15));
    let result = core.execute_pass(&mut trace, 0.0);
    assert_eq!(
        result
            .output_signals
            .iter()
            .map(|output| (output.uuid, output.completed))
            .collect::<Vec<_>>(),
        [(load, false), (local, true)]
    );
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
    assert_eq!(usage(&core), (213, 6));
}

/// Price every pass of `core` at 0 ms.
fn price_passes_at_0_ms(core: &mut VllmCore) {
    core.args.perf_model = Arc::new(PerfModel::External {
        timing: Arc::new(CapturedDecodeQueries::default()),
    });
}

/// A prefill rank of 500 bytes in byte mode with a budget of 4 tokens and
/// passes at 0 ms: a full-attention group of 4-token blocks of 100 bytes and
/// a state group of 1 block of 10 bytes, with which a prefill stops one token
/// short. A source of 9 tokens holds its 8 computed tokens in 2 full blocks
/// and the state block: 210 bytes in 3 blocks, so 290 bytes are free. Returns
/// the rank and the handoff of that source.
fn byte_mode_prefill_rank_with_a_held_source() -> (VllmCore, HandoffId) {
    let mut groups = vec![group("full", 4, 100, None), state("state", 10, 1)];
    for group in &mut groups {
        group.prefix_cacheable = Some(true);
        group.transfer_bytes_per_block = Some(group.page_size_bytes);
    }
    let mut core = pd_core(groups, 500, 4, WorkerType::Prefill, None);
    price_passes_at_0_ms(&mut core);
    let blocker = HandoffId::new(Uuid::from_u128(1646));
    hold_grouped_source(&mut core, blocker, 1645, 9);
    let pool = grouped_pool(&core);
    assert_eq!((pool.used_bytes(), pool.active_blocks()), (210, 3));
    (core, blocker)
}

/// Run `passes` passes at 0 ms and return the passes that admit a request.
fn admitting_passes_at_0_ms(core: &mut VllmCore, passes: usize) -> Vec<usize> {
    let mut trace = TraceCollector::default();
    let mut admissions = Vec::new();
    for pass in 1..=passes {
        let result = core.execute_pass(&mut trace, 0.0);
        assert_eq!(result.end_ms, 0.0, "pass {pass}");
        admissions.extend(result.admissions.iter().map(|_| pass));
    }
    admissions
}

#[test]
fn grouped_prefill_rank_readmits_a_held_request_when_a_held_source_leaves_at_that_time() {
    for cancel in [false, true] {
        let (mut core, blocker) = byte_mode_prefill_rank_with_a_held_source();
        let usage = |core: &VllmCore| {
            let pool = grouped_pool(core);
            (pool.used_bytes(), pool.active_blocks())
        };
        // A second source of 13 tokens computes 4 of its 12 tokens in each
        // pass: 110 and 210 bytes. Its last 4 tokens need 3 full blocks and
        // the state block, 310 bytes, and 290 are free. So it preempts
        // itself in pass 3, is admitted again in pass 4 and preempts itself
        // in pass 6 with the same pool. From pass 7 on it waits.
        let waiting = HandoffId::new(Uuid::from_u128(1648));
        let second = Uuid::from_u128(1647);
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: waiting,
            request: source_request(1647, 13),
        })
        .unwrap();
        assert_eq!(admitting_passes_at_0_ms(&mut core, 10), [1, 4], "{cancel}");
        assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
        assert_eq!(usage(&core), (210, 3));
        // The held source is released or cancelled at the same time, so the
        // 500 bytes are free. The next pass at that time admits the second
        // source, and its third pass computes its last tokens and holds it.
        let command = if cancel {
            SchedulerCommand::CancelSource {
                handoff_id: blocker,
            }
        } else {
            SchedulerCommand::ReleaseSource {
                handoff_id: blocker,
            }
        };
        core.apply_command(command).unwrap();
        assert_eq!(usage(&core), (0, 0));
        let mut trace = TraceCollector::default();
        let result = core.execute_pass(&mut trace, 0.0);
        assert_eq!(
            result
                .admissions
                .iter()
                .map(|admission| admission.uuid)
                .collect::<Vec<_>>(),
            [second],
            "{cancel}"
        );
        assert_eq!(usage(&core), (110, 2));
        core.execute_pass(&mut trace, 0.0);
        assert!(!core.source_is_held(waiting));
        core.execute_pass(&mut trace, 0.0);
        assert!(core.source_is_held(waiting), "{cancel}");
        assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
        assert_eq!(usage(&core), (310, 4));
    }
}

#[test]
fn grouped_prefill_rank_keeps_the_request_behind_a_held_request_waiting() {
    // The rank and the second source of the test above, and behind it a
    // third source of 5 tokens, whose 4 computed tokens need 110 bytes.
    let (mut core, blocker) = byte_mode_prefill_rank_with_a_held_source();
    let usage = |core: &VllmCore| {
        let pool = grouped_pool(core);
        (pool.used_bytes(), pool.active_blocks())
    };
    let (waiting, behind) = (
        HandoffId::new(Uuid::from_u128(1648)),
        HandoffId::new(Uuid::from_u128(1650)),
    );
    let (second, third) = (Uuid::from_u128(1647), Uuid::from_u128(1649));
    for (handoff_id, request) in [
        (waiting, source_request(1647, 13)),
        (behind, source_request(1649, 5)),
    ] {
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request,
        })
        .unwrap();
    }
    // Each pass in which the second source runs gives it the whole budget,
    // and a pass that preempts admits no request. From pass 7 on the second
    // source waits at the head of the queue. The hold stops the waiting
    // loop, as on a decode rank, so the third source waits too, while its
    // 110 bytes fit the 290 free bytes and the budget is free.
    assert_eq!(admitting_passes_at_0_ms(&mut core, 10), [1, 4]);
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
    assert_eq!(usage(&core), (210, 3));
    assert_eq!(
        [second, third].map(|uuid| core.state.requests[&uuid].status),
        [RequestStatus::Preempted, RequestStatus::Waiting]
    );
    // The held source is released at the same time. The second source runs
    // in the next three passes and is held with 310 bytes. The fourth pass
    // admits the third source, which computes its 4 tokens and is held too.
    core.apply_command(SchedulerCommand::ReleaseSource {
        handoff_id: blocker,
    })
    .unwrap();
    let mut trace = TraceCollector::default();
    let mut admissions = Vec::new();
    for _ in 0..4 {
        let result = core.execute_pass(&mut trace, 0.0);
        admissions.push(
            result
                .admissions
                .iter()
                .map(|admission| admission.uuid)
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(admissions, [vec![second], vec![], vec![], vec![third]]);
    assert!(core.source_is_held(waiting) && core.source_is_held(behind));
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
    assert_eq!(usage(&core), (420, 6));
}

#[test]
fn grouped_prefill_rank_in_block_mode_readmits_a_held_request_when_a_held_source_leaves() {
    // A prefill rank of 5 pages in block mode on vLLM 0.31.0 with a budget of
    // 16 tokens and passes at 0 ms: a full-attention group and a state group
    // of 16-token blocks, at a hash block of 16. A source of 17 tokens holds
    // its 16 computed tokens in a full block and a state block, so 3 blocks
    // are free.
    for cancel in [false, true] {
        let mut full = group("full", 16, 100, None);
        full.prefix_cacheable = Some(true);
        full.transfer_bytes_per_block = Some(100);
        let mut recurrent = state("state", 100, 1);
        recurrent.block_size_tokens = 16;
        recurrent.prefix_cacheable = Some(true);
        recurrent.transfer_bytes_per_block = Some(100);
        let mut args = prefix_core(vec![full, recurrent], 5 * 100, 16, 16, "0.31.0").args;
        args.worker_type = WorkerType::Prefill;
        let mut core = VllmCore::new(args);
        price_passes_at_0_ms(&mut core);
        let blocker = HandoffId::new(Uuid::from_u128(1652));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: blocker,
            request: DirectRequest {
                tokens: (1000..1017).collect(),
                ..source_request(1651, 0)
            },
        })
        .unwrap();
        held_timing(&mut core);
        assert_eq!(block_pool(&core).active_blocks(), 2);
        // A second request of 15 tokens with 3 output tokens, sent as a
        // plain request. Pass 1 computes its 15 tokens in a full block and a
        // state block and emits the first output token, and pass 2 computes
        // that token and emits the second. The 17th token then needs a second full block and, past
        // the block boundary, a new state block beside the previous one: 2
        // blocks, and 1 is free. So the request preempts itself in pass 3.
        // Its readmission in pass 4 counts 2 full blocks and a state block
        // for its 17 tokens and finds 3. It computes 16 tokens again, and
        // pass 5 preempts it with the same pool. From pass 6 on it waits.
        let second = Uuid::from_u128(1653);
        core.apply_command(SchedulerCommand::Submit(DirectRequest {
            tokens: (2000..2015).collect(),
            max_output_tokens: 3,
            ..source_request(1653, 0)
        }))
        .unwrap();
        assert_eq!(admitting_passes_at_0_ms(&mut core, 10), [1, 4], "{cancel}");
        assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
        assert_eq!(block_pool(&core).active_blocks(), 2);
        // The held source is released or cancelled at the same time. The
        // next pass at that time admits the second request, and the pass
        // after it computes the 17th token in 4 blocks, emits the third
        // output token and finishes the request.
        let command = if cancel {
            SchedulerCommand::CancelSource {
                handoff_id: blocker,
            }
        } else {
            SchedulerCommand::ReleaseSource {
                handoff_id: blocker,
            }
        };
        core.apply_command(command).unwrap();
        assert_eq!(block_pool(&core).active_blocks(), 0);
        let mut trace = TraceCollector::default();
        let result = core.execute_pass(&mut trace, 0.0);
        assert_eq!(
            result
                .admissions
                .iter()
                .map(|admission| admission.uuid)
                .collect::<Vec<_>>(),
            [second],
            "{cancel}"
        );
        assert_eq!(block_pool(&core).active_blocks(), 2);
        assert_eq!(core.state.requests[&second].status, RequestStatus::Running);
        block_pool(&core).reset_peak_active_blocks();
        core.execute_pass(&mut trace, 0.0);
        assert!(core.is_drained(), "{cancel}");
        assert_eq!(block_pool(&core).peak_active_blocks(), 4);
        assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
    }
}

#[test]
fn grouped_destination_without_state_layers_activates_prebuilt() {
    // Without a Mamba state the backoff is 0: the decode rank loads all 130
    // tokens, three 64-token blocks, and recomputes none.
    let mut full = group("full", 64, 100, None);
    full.prefix_cacheable = Some(true);
    full.transfer_bytes_per_block = Some(100);
    let mut core = pd_core(vec![full], 1_000, 8192, WorkerType::Decode, None);
    let handoff_id = HandoffId::new(Uuid::from_u128(1612));
    assert_eq!(
        reserve_grouped(&mut core, handoff_id, pd_request(1611, 130, 2)),
        Some(3)
    );
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(300));
    activate(&mut core, handoff_id);
    let request = &core.state.requests[&Uuid::from_u128(1611)];
    assert_eq!(request.num_computed_tokens, 130);
    assert_eq!(request.recompute_tokens, 0);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 3), 1);
    assert!(core.is_drained());
}

#[test]
fn grouped_destination_counts_the_copy_of_a_state_loaded_to_its_partial_tail() {
    // Attention and state blocks of 32 at a hash block of 4, so hits inside a
    // block are on. A 49-token prompt transfers 48 tokens, the prompt's
    // partial tail. vLLM caches the loaded state there when the load
    // completes (V31 sched:3083-3085, V32 :3155-3157), so the request is a
    // producer, and what it still needs counts the block of the copy (V31
    // single:1770-1777 with sched:3003-3017, V32 single:1732-1739 with
    // sched:3060-3076). Its first step takes that block.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut full = group("full", 32, 100, None);
        full.prefix_cacheable = Some(true);
        full.transfer_bytes_per_block = Some(100);
        let mut recurrent = state("state", 100, 1);
        recurrent.block_size_tokens = 32;
        recurrent.prefix_cacheable = Some(true);
        recurrent.transfer_bytes_per_block = Some(100);
        recurrent.prefill_checkpoint_blocks = Some(0);
        let mut args = prefix_core(vec![full, recurrent], 16 * 100, 8192, 4, version).args;
        args.worker_type = WorkerType::Decode;
        let mut core = VllmCore::new(args);
        let handoff_id = HandoffId::new(Uuid::from_u128(1977));
        assert_eq!(
            reserve_grouped(&mut core, handoff_id, pd_request(1976, 49, 2)),
            Some(2),
            "{version}"
        );
        assert_eq!(core.kv_manager.grouped_inflight_need(), 0, "{version}");
        activate(&mut core, handoff_id);
        assert_eq!(block_pool(&core).active_blocks(), 3, "{version}");
        assert_eq!(core.kv_manager.grouped_inflight_need(), 1, "{version}");
        let mut trace = TraceCollector::default();
        let mut step = 0;
        step_pd(&mut core, &mut trace, &mut step);
        assert_eq!(core.kv_manager.grouped_inflight_need(), 0, "{version}");
        // Two attention blocks, the new running block and the published one.
        assert_eq!(block_pool(&core).active_blocks(), 4, "{version}");
    }
}

/// A 16-token sliding window of 16-token blocks and one cacheable state
/// block, both at 100 bytes a page, with the transfer fields set.
fn window_transfer_groups() -> Vec<FpmCacheGroup> {
    let mut window = group("window", 16, 100, Some(16));
    window.prefix_cacheable = Some(true);
    window.transfer_bytes_per_block = Some(100);
    let mut recurrent = state("state", 100, 1);
    recurrent.block_size_tokens = 16;
    recurrent.prefix_cacheable = Some(true);
    recurrent.transfer_bytes_per_block = Some(100);
    vec![window, recurrent]
}

#[test]
fn grouped_destination_with_a_window_group_holds_only_the_window_blocks() {
    // The 129 loaded tokens of a 130-token prompt take nine 16-token blocks,
    // and the window skips the first seven. A fresh destination holds no
    // blocks to free: vLLM counts the two blocks in the window, pads the
    // skipped ones with null blocks and adds the state block, so the request
    // fits three pages (V32 `single_type_kv_cache_manager.py:245-248`,
    // `:310-311`).
    let mut core = pd_core(
        window_transfer_groups(),
        300,
        8192,
        WorkerType::Decode,
        None,
    );
    let handoff_id = HandoffId::new(Uuid::from_u128(1638));
    assert_eq!(
        reserve_grouped(&mut core, handoff_id, pd_request(1637, 130, 3)),
        Some(0)
    );
    assert_eq!(grouped_pool(&core).active_blocks(), 3);
    assert_eq!(grouped_pool(&core).used_bytes(), 300);
    activate(&mut core, handoff_id);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    // The first pass computes the last prompt token in the same window.
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert_eq!(grouped_pool(&core).active_blocks(), 3);
    assert_eq!(grouped_pool(&core).used_bytes(), 300);
    assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 3), 2);
    assert_eq!(grouped_pool(&core).used_bytes(), 0);
    assert!(core.is_drained());
}

#[test]
fn grouped_destination_with_a_window_group_in_block_mode_holds_only_the_window_blocks() {
    let mut args = prefix_core(window_transfer_groups(), 300, 8192, 16, "0.31.0").args;
    args.worker_type = WorkerType::Decode;
    let mut core = VllmCore::new(args);
    let handoff_id = HandoffId::new(Uuid::from_u128(1640));
    assert_eq!(
        reserve_grouped(&mut core, handoff_id, pd_request(1639, 130, 3)),
        Some(0)
    );
    // Two window blocks and the running state block. The seven blocks below
    // the window are null blocks.
    assert_eq!(block_pool(&core).active_blocks(), 3);
    activate(&mut core, handoff_id);
    // The activation keys the window block that ends at the replay boundary
    // 128. The state after 129 tokens sits at no boundary.
    assert_eq!(block_pool(&core).active_blocks(), 3);
    assert_eq!(block_pool(&core).inactive_cached_blocks(), 0);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert_eq!(block_pool(&core).active_blocks(), 3);
    assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 3), 2);
    assert_eq!(block_pool(&core).active_blocks(), 0);
    assert_eq!(block_pool(&core).inactive_cached_blocks(), 1);
    assert_eq!(core.mocker_metrics().kv_cache_cached_bytes, Some(100));
    assert!(core.is_drained());
}

#[test]
fn grouped_destination_rejects_a_prompt_that_cannot_fit() {
    // 4,999 transferred tokens need three attention blocks and five fixed
    // blocks, and the pool holds four.
    let mut core = pd_core(
        glm_transfer_groups("0.32.0", false),
        4 * GLM_PAGE,
        8192,
        WorkerType::Decode,
        None,
    );
    let handoff_id = HandoffId::new(Uuid::from_u128(1614));
    let error = core
        .apply_command(SchedulerCommand::ReserveDestination {
            handoff_id,
            request: pd_request(1613, 5_000, 2),
        })
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("destination prompt and state exceed the grouped KV pool capacity"),
        "{error}"
    );
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_destination_capacity_counts_the_whole_prompt() {
    // vLLM's admission check needs free blocks for the whole prompt with the
    // loaded tokens computed (V32 `vllm/v1/core/kv_cache_manager.py:522-539`).
    // The 2,176 loaded tokens of a 2,177-token prompt fill one attention
    // block, and the last prompt token opens a second one: seven blocks with
    // the five state blocks. A pool of six pages never admits the request.
    for version in ["0.32.0", "0.33.0"] {
        let byte_mode = pd_core(
            glm_transfer_groups(version, false),
            6 * GLM_PAGE,
            8192,
            WorkerType::Decode,
            None,
        );
        let block_mode = block_mode_decode_core(6 * GLM_PAGE, version, false);
        for mut core in [byte_mode, block_mode] {
            let error = core
                .apply_command(SchedulerCommand::ReserveDestination {
                    handoff_id: HandoffId::new(Uuid::from_u128(1642)),
                    request: pd_request(1641, 2_177, 2),
                })
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                "destination prompt and state exceed the grouped KV pool capacity"
            );
            assert!(core.is_drained());
            assert_eq!(grouped_pool(&core).used_bytes(), 0);
        }
    }
}

#[test]
fn grouped_destination_waits_in_fifo_while_the_pool_is_full() {
    let mut core = pd_core(
        glm_transfer_groups("0.32.0", false),
        12 * GLM_PAGE,
        8192,
        WorkerType::Decode,
        None,
    );
    let first = HandoffId::new(Uuid::from_u128(1616));
    let second = HandoffId::new(Uuid::from_u128(1618));
    let third = HandoffId::new(Uuid::from_u128(1620));
    assert_eq!(
        reserve_grouped(&mut core, first, pd_request(1615, 1_000, 3)),
        Some(1)
    );
    activate(&mut core, first);
    // The second reservation fills the pool. The third waits at the head of
    // the FIFO: accepted, not reserved.
    assert_eq!(
        reserve_grouped(&mut core, second, pd_request(1617, 1_000, 3)),
        Some(1)
    );
    assert_eq!(grouped_pool(&core).used_bytes(), 12 * GLM_PAGE);
    assert_eq!(
        reserve_grouped(&mut core, third, pd_request(1619, 1_000, 3)),
        None
    );
    assert!(core.destination_is_held(third));
    assert_eq!(grouped_pool(&core).used_bytes(), 12 * GLM_PAGE);

    // The first request decodes within its blocks and finishes. Nothing is
    // preempted while the pool is full.
    let mut trace = TraceCollector::default();
    let mut step = 0;
    assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 12), 3);
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 0);
    assert_eq!(grouped_pool(&core).used_bytes(), 6 * GLM_PAGE);
    // The retry serves the third request from the freed blocks.
    let events = core.retry_pending_destinations();
    assert_eq!(reserved_units(&events, third), Some(1));
    assert_eq!(core.destination_block_count(third), 6);
    assert_eq!(grouped_pool(&core).used_bytes(), 12 * GLM_PAGE);
}

#[test]
fn grouped_destination_in_block_mode_pins_the_attention_hit_and_moves_the_state() {
    for (version, checkpoint) in GLM_CORES {
        let mut core = block_mode_decode_core(128 * GLM_PAGE, version, checkpoint);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let first = HandoffId::new(Uuid::from_u128(1622));
        let id = Uuid::from_u128(1621);
        // The 2,176 transferred tokens fill one attention block: one unit.
        assert_eq!(
            reserve_grouped(&mut core, first, pd_request(1621, 2_177, 3)),
            Some(1),
            "{version} {checkpoint}"
        );
        assert_eq!(block_pool(&core).active_blocks(), 6);
        activate(&mut core, first);
        // The activation publishes what vLLM caches when the load completes
        // (V32 `sched:3155-3157`, V31 `:3083-3085`): the full attention
        // block, and the state at the replay boundary 2,176 in each KDA group.
        // The ring caches nothing.
        let hash = prompt_hash(2_177, 2_176);
        assert!(block_pool(&core).key_hit(0, hash).is_some());
        assert!(block_pool(&core).key_hit(1, hash).is_none());
        for g in 2..6 {
            assert!(
                block_pool(&core).key_hit(g, hash).is_some(),
                "{version} {checkpoint} group {g}"
            );
        }
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 0);
        assert_eq!(block_pool(&core).active_blocks(), 6);
        let request = &core.state.requests[&id];
        assert_eq!(request.num_computed_tokens, 2_176);
        assert_eq!(request.recompute_tokens, 1);
        assert_eq!(request.sequence.lease.grouped_hit().unwrap().tokens, 0);
        // The first step opens a second attention block and crosses a state
        // block boundary in each KDA group, whose previous running block
        // stays one step.
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
        assert_eq!(block_pool(&core).active_blocks(), 11);
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 11), 2);
        // The attention block and the four KDA states stay cached.
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 5);
        assert_eq!(
            core.mocker_metrics().kv_cache_cached_bytes,
            Some(5 * GLM_PAGE),
            "{version} {checkpoint}"
        );

        // The same prompt again. The decode rank holds its one full block,
        // so the hit covers every transferable token and no unit moves. vLLM
        // then moves nothing at all, while the handoff timing still charges
        // the fixed state bytes: a known difference that the measured run did
        // not cover.
        let second = HandoffId::new(Uuid::from_u128(1624));
        assert_eq!(
            reserve_grouped(&mut core, second, pd_request(1623, 2_177, 3)),
            Some(0),
            "{version} {checkpoint}"
        );
        // The pinned attention block, a fresh ring block and four fresh
        // state blocks. The cached KDA states are not the hit: the state
        // arrives with the transfer.
        assert_eq!(block_pool(&core).active_blocks(), 6);
        assert!(
            block_pool(&core)
                .key_hit(0, hash)
                .is_some_and(|hit| hit.is_active)
        );
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 4);
        activate(&mut core, second);
        let request = &core.state.requests[&Uuid::from_u128(1623)];
        assert_eq!(request.num_computed_tokens, 2_176);
        let hit = request.sequence.lease.grouped_hit().unwrap();
        assert_eq!(hit.tokens, 2_176);
        assert_eq!(hit.junction_tokens, 0);
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 11), 2);

        // A prompt that shares the first 2,177 tokens transfers 2,177: the
        // hit covers one block, and the partly filled second block moves
        // with the state.
        let third = HandoffId::new(Uuid::from_u128(1626));
        assert_eq!(
            reserve_grouped(&mut core, third, pd_request(1625, 2_178, 2)),
            Some(1),
            "{version} {checkpoint}"
        );
        assert_eq!(block_pool(&core).active_blocks(), 7);
        activate(&mut core, third);
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 12), 2);
        assert!(core.is_drained(), "{version} {checkpoint}");
    }
}

#[test]
fn grouped_destination_in_block_mode_publishes_only_full_blocks_at_a_boundary() {
    // 8,191 transferred tokens fill three attention blocks and a fourth in
    // part. The KDA state after 8,191 tokens sits at no replay boundary, so
    // no state is keyed.
    for (version, checkpoint) in GLM_CORES {
        let mut core = block_mode_decode_core(128 * GLM_PAGE, version, checkpoint);
        let handoff_id = HandoffId::new(Uuid::from_u128(1628));
        assert_eq!(
            reserve_grouped(&mut core, handoff_id, pd_request(1627, 8_192, 2)),
            Some(4),
            "{version} {checkpoint}"
        );
        activate(&mut core, handoff_id);
        for at in [2_176, 4_352, 6_528] {
            assert!(
                block_pool(&core)
                    .key_hit(0, prompt_hash(8_192, at))
                    .is_some(),
                "{version} {checkpoint} at {at}"
            );
        }
        // 8,191 is no hash position, so only the boundary below it can hold a
        // KDA key, and it holds none.
        for g in 2..6 {
            assert!(
                block_pool(&core)
                    .key_hit(g, prompt_hash(8_192, 6_528))
                    .is_none(),
                "{version} {checkpoint} group {g}"
            );
        }
        assert_eq!(block_pool(&core).active_blocks(), 9);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 13), 2);
        assert!(core.is_drained());
    }
}

#[test]
fn grouped_destination_in_block_mode_waits_in_fifo_when_its_blocks_do_not_fit() {
    // Eleven pages. A finished request leaves five cached blocks. A running
    // request holds six. A repeat of the first prompt passes the capacity
    // check, which counts seven blocks against the eleven of the pool. Its
    // reservation then needs free blocks for the whole prompt: the same
    // seven, the inactive cached attention block of its hit among them,
    // against the five that are free. So it waits before it pins the hit.
    for version in ["0.32.0", "0.33.0"] {
        let mut core = block_mode_decode_core(11 * GLM_PAGE, version, false);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let first = HandoffId::new(Uuid::from_u128(1630));
        assert_eq!(
            reserve_grouped(&mut core, first, pd_request(1629, 2_177, 3)),
            Some(1)
        );
        activate(&mut core, first);
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 11), 3);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 5);

        let second = HandoffId::new(Uuid::from_u128(1632));
        assert_eq!(
            reserve_grouped(
                &mut core,
                second,
                DirectRequest {
                    tokens: (100_000..101_000).collect(),
                    ..pd_request(1631, 0, 3)
                }
            ),
            Some(1)
        );
        activate(&mut core, second);
        assert_eq!(block_pool(&core).active_blocks(), 6);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 5);

        let third = HandoffId::new(Uuid::from_u128(1634));
        assert_eq!(
            reserve_grouped(&mut core, third, pd_request(1633, 2_177, 3)),
            None
        );
        // The failed reservation pinned nothing: the attention block stays in the
        // inactive LRU, the pool is as it was, and nothing was preempted.
        assert!(core.destination_is_held(third));
        assert_eq!(block_pool(&core).active_blocks(), 6);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 5);
        let hash = prompt_hash(2_177, 2_176);
        assert!(
            block_pool(&core)
                .key_hit(0, hash)
                .is_some_and(|hit| !hit.is_active)
        );
        assert_eq!(core.mocker_metrics().vllm_preemptions_total, 0);

        // The running request finishes, and the retry serves the head of the
        // FIFO with its hit pinned: no unit moves.
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 11), 3);
        let events = core.retry_pending_destinations();
        assert_eq!(reserved_units(&events, third), Some(0));
        assert_eq!(block_pool(&core).active_blocks(), 6);
        assert!(
            block_pool(&core)
                .key_hit(0, hash)
                .is_some_and(|hit| hit.is_active)
        );
        activate(&mut core, third);
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 11), 3);
        assert!(core.is_drained());
    }
}

/// A full-attention group and a cacheable state group of 16-token blocks,
/// both at 100 bytes a page, with the transfer fields set.
fn full_and_state_transfer_groups() -> Vec<FpmCacheGroup> {
    let mut full = group("full", 16, 100, None);
    full.prefix_cacheable = Some(true);
    full.transfer_bytes_per_block = Some(100);
    let mut recurrent = state("state", 100, 1);
    recurrent.block_size_tokens = 16;
    recurrent.prefix_cacheable = Some(true);
    recurrent.transfer_bytes_per_block = Some(100);
    vec![full, recurrent]
}

#[test]
fn grouped_destination_reserves_one_pass_later_with_async_scheduling() {
    // A decode rank with one full-attention group of 4-token blocks and room
    // for 10 blocks, in byte mode and in block mode. X (4 tokens, 20 outputs)
    // keeps a pass in flight. R (16 tokens, 2 outputs) prefills in pass 0 and
    // ends in pass 1. After pass 0, X holds 1 block and R 4, so the load D of
    // 24 tokens, 6 blocks, waits with 5 blocks free. Pass 1 takes X's second
    // block and R's fifth. Without async scheduling R frees its 5 blocks when
    // it ends, and the retry at the boundary of pass 1 reserves D in the 8
    // free blocks. vLLM frees R when it processes the outputs of step 1,
    // which with async scheduling follows the scheduling of step 2 (V31
    // `engine/core.py:697`, `:740-755`, `sched:2171`, V32 `engine/core.py:725`,
    // `:768-783`, `sched:2230`), and tests D again in step 3. So with async
    // scheduling replay frees R when it forms pass 2, and the retry at the
    // boundary of pass 2 reserves D.
    for (block_mode, version) in WINDOW_PD_MODES {
        for (max_concurrent_batches, reserving_pass) in [(1, 1), (2, 2)] {
            let label = format!("{block_mode} {version} {max_concurrent_batches}");
            let mut full = group("full", 4, 100, None);
            full.prefix_cacheable = Some(true);
            full.transfer_bytes_per_block = Some(100);
            let mut args = if block_mode {
                prefix_core(vec![full], 10 * 100, 64, 4, version).args
            } else {
                pd_core(vec![full], 10 * 100, 64, WorkerType::Decode, None).args
            };
            args.worker_type = WorkerType::Decode;
            args.max_concurrent_batches = max_concurrent_batches;
            let mut core = VllmCore::new(args);
            let mut now = 0.0;
            submit_tokens(&mut core, 1, (1000..1004).collect(), 20);
            submit_tokens(&mut core, 2, (2000..2016).collect(), 2);
            async_step(&mut core, &mut now);
            let handoff_id = HandoffId::new(Uuid::from_u128(1004));
            assert_eq!(
                reserve_grouped(&mut core, handoff_id, pd_request(1003, 24, 2)),
                None,
                "{label}"
            );
            let reserved = (1..4).find(|_| {
                async_step(&mut core, &mut now);
                reserved_units(&core.retry_pending_destinations(), handoff_id).is_some()
            });
            assert_eq!(reserved, Some(reserving_pass), "{label}");
        }
    }
}

#[test]
fn grouped_destination_retries_when_a_pass_releases_a_state_block() {
    // Eight pages in block mode. The first request loads 32 of its 33 tokens
    // into 2 attention blocks and 1 state block. Its first pass opens a third
    // attention block and crosses a state block boundary, and the previous
    // state block stays for one step: 5 blocks. A distinct second prompt of
    // 33 tokens needs 4 free blocks for its whole prompt and finds 3, so it
    // waits. The next pass releases the previous state block, and the retry
    // after that pass reserves the second load in the 4 free blocks.
    let mut args = prefix_core(full_and_state_transfer_groups(), 800, 8192, 16, "0.31.0").args;
    args.worker_type = WorkerType::Decode;
    let mut core = VllmCore::new(args);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let first = HandoffId::new(Uuid::from_u128(1644));
    assert_eq!(
        reserve_grouped(&mut core, first, pd_request(1643, 33, 3)),
        Some(2)
    );
    activate(&mut core, first);
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert_eq!(block_pool(&core).active_blocks(), 5);
    let second = HandoffId::new(Uuid::from_u128(1646));
    let request = DirectRequest {
        tokens: (1000..1033).collect(),
        ..pd_request(1645, 0, 3)
    };
    assert_eq!(reserve_grouped(&mut core, second, request), None);
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert_eq!(block_pool(&core).active_blocks(), 4);
    let events = core.retry_pending_destinations();
    assert_eq!(reserved_units(&events, second), Some(2));
    activate(&mut core, second);
    assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 8), 4);
    assert!(core.is_drained());
}

#[test]
fn grouped_destination_retries_when_a_pass_releases_the_cached_block_of_a_boundary_hit() {
    // A decode rank in block mode with full-attention and state blocks of 16
    // tokens and a hash block of 16. P (33 tokens) runs and ends, and leaves
    // 3 cached blocks: two full-attention blocks and the state block keyed
    // at 32 tokens. L is P's first 32 tokens and 10 more. It hits 32 tokens,
    // a hit on a block boundary, and computes its prompt in pass 0. It then
    // holds 5 blocks: the 2 hit blocks and a new full-attention block, and
    // the cached state block of its hit and a running state block. The load
    // B of 33 tokens needs 4 blocks for its whole prompt and finds 3, so it
    // waits. In pass 1 L allocates again, has computed past the hit block,
    // and releases it with its other skipped blocks. The retry after pass 1
    // reserves B in the 4 free blocks. vLLM frees that block in the running
    // loop of the step and can reserve B in the waiting loop of the same
    // step, one forward pass earlier. On 7 pages B fits only once L ends, in
    // pass 5, and the retry after pass 5 reserves it.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for max_concurrent_batches in [1, 2] {
            for (pages, reserving_pass) in [(8u64, 1), (7, 5)] {
                let label = format!("{version} {max_concurrent_batches} {pages}");
                let mut groups = full_and_state_transfer_groups();
                if version != "0.31.0" {
                    groups[1].prefill_checkpoint_blocks = Some(0);
                }
                let mut args = prefix_core(groups, pages * 100, 8192, 16, version).args;
                args.worker_type = WorkerType::Decode;
                args.max_concurrent_batches = max_concurrent_batches;
                let mut core = VllmCore::new(args);
                let mut now = 0.0;
                let prompt: Vec<u32> = (5000..5033).collect();
                submit_tokens(&mut core, 1, prompt.clone(), 1);
                while !core.is_drained() {
                    async_step(&mut core, &mut now);
                }
                assert_eq!(block_pool(&core).inactive_cached_blocks(), 3, "{label}");
                let mut tokens = prompt[..32].to_vec();
                tokens.extend(6000..6010);
                let l = submit_tokens(&mut core, 2, tokens, 6);
                let pass = async_step(&mut core, &mut now);
                let hit = pass
                    .admissions
                    .iter()
                    .find(|admission| admission.uuid == l)
                    .map(|admission| admission.reused_input_tokens);
                assert_eq!(hit, Some(32), "{label}");
                assert_eq!(block_pool(&core).active_blocks(), 5, "{label}");
                let load = HandoffId::new(Uuid::from_u128(3001));
                let request = DirectRequest {
                    tokens: (8000..8033).collect(),
                    ..pd_request(3000, 33, 2)
                };
                assert_eq!(reserve_grouped(&mut core, load, request), None, "{label}");
                let reserved = (1..7).find(|_| {
                    async_step(&mut core, &mut now);
                    reserved_units(&core.retry_pending_destinations(), load).is_some()
                });
                assert_eq!(reserved, Some(reserving_pass), "{label}");
            }
        }
    }
}

/// One arrival of a scripted decode rank: before pass `pass`, a local
/// request (`load: false`) or a KV load `id` whose first `shared` tokens are
/// the shared prompt `0..` and whose other tokens are its own.
struct Arrival {
    pass: usize,
    load: bool,
    id: u128,
    shared: u32,
    prompt: u32,
    output: usize,
}

/// A decode rank of 16 pages of the full-attention and state groups on
/// `version`, with prefix caching on a hash block of 16 tokens and up to 8
/// running requests. Each pass gets its arrivals, runs, and then retries the
/// waiting loads, as replay does at a pass boundary. A load that reserves is
/// activated at once. Returns the core before the retry after the last pass.
fn scripted_decode_rank(arrivals: &[Arrival], passes: usize, version: &str) -> VllmCore {
    scripted_decode_rank_with(arrivals, passes, 1, version)
}

/// `scripted_decode_rank` with `max_concurrent_batches` batches. With more
/// than one, each pass ends at its boundary and the next pass starts at that
/// end, so each pass that follows a pass with work is formed while that pass
/// is in flight, as a rank does.
fn scripted_decode_rank_with(
    arrivals: &[Arrival],
    passes: usize,
    max_concurrent_batches: usize,
    version: &str,
) -> VllmCore {
    let mut groups = full_and_state_transfer_groups();
    groups[1].prefill_checkpoint_blocks = Some(0);
    let mut args = prefix_core(groups, 1_600, 8192, 16, version).args;
    args.worker_type = WorkerType::Decode;
    args.max_num_seqs = Some(8);
    args.max_concurrent_batches = max_concurrent_batches;
    let mut core = VllmCore::new(args);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let mut now = 0.0;
    for pass in 0..passes {
        for arrival in arrivals.iter().filter(|arrival| arrival.pass == pass) {
            let id = arrival.id;
            let tokens = (0..arrival.prompt)
                .map(|i| {
                    if i < arrival.shared {
                        i
                    } else {
                        100_000 * id as u32 + i
                    }
                })
                .collect();
            if arrival.load {
                let handoff_id = HandoffId::new(Uuid::from_u128(1_000_000 + id));
                let request = DirectRequest {
                    tokens,
                    ..pd_request(id, 0, arrival.output)
                };
                if reserve_grouped(&mut core, handoff_id, request).is_some() {
                    activate(&mut core, handoff_id);
                }
            } else {
                submit_tokens(&mut core, id, tokens, arrival.output);
            }
        }
        if max_concurrent_batches > 1 {
            let result = core.execute_pass(&mut trace, now);
            assert!(result.output_signals.iter().all(|output| !output.rejected));
            core.complete_engine_boundary(result.end_ms);
            now = result.end_ms;
        } else {
            step_pd(&mut core, &mut trace, &mut step);
        }
        if pass + 1 == passes {
            break;
        }
        for event in core.retry_pending_destinations() {
            if let SchedulerLifecycleEvent::DestinationReserved { handoff_id, .. } = event {
                activate(&mut core, handoff_id);
            }
        }
    }
    core
}

#[test]
fn grouped_destination_retries_when_the_loads_in_flight_need_less() {
    let arrival = |pass, load, id, shared, prompt, output| Arrival {
        pass,
        load,
        id,
        shared,
        prompt,
        output,
    };
    let arrivals = [
        arrival(4, true, 1, 96, 87, 10),
        arrival(9, true, 2, 0, 40, 25),
        arrival(10, true, 3, 64, 46, 12),
        arrival(15, true, 4, 0, 72, 17),
        arrival(19, true, 5, 96, 69, 24),
        arrival(22, true, 6, 64, 65, 20),
        arrival(25, true, 7, 80, 70, 12),
        arrival(25, true, 8, 64, 65, 27),
        arrival(29, true, 9, 80, 78, 25),
        // A load of the first 73 tokens of the shared prefix. It waits.
        arrival(29, true, 10, 96, 73, 25),
        arrival(29, true, 11, 80, 65, 10),
        arrival(32, true, 12, 0, 97, 23),
        arrival(33, true, 13, 0, 63, 21),
    ];
    for version in ["0.32.0", "0.33.0"] {
        let mut core = scripted_decode_rank(&arrivals, 35, version);
        let head = HandoffId::new(Uuid::from_u128(1_000_010));
        // Load 10 waits while the loads in flight need 2 blocks. The last pass
        // releases no block, publishes and evicts no key and leaves the published
        // epoch, so the retry epoch stays. It is the first pass of a load in
        // flight, which takes that load out of the need: the need falls from 2
        // blocks to 0. Load 10 then fits. vLLM sums that need again at each
        // attempt (V32 `sched:1241-1251`, `:3088-3092`), so the retry after this
        // pass reserves it.
        assert_eq!(
            reserved_units(&core.retry_pending_destinations(), head),
            Some(1)
        );
    }
}

#[test]
fn grouped_destination_retries_when_a_step_start_makes_a_state_key_a_hit() {
    // A decode rank of the GLM-5.3-Flash groups on vLLM 0.32.0 and on vLLM
    // 0.33.0, 22 pages and a hash block of 16.
    for version in ["0.32.0", "0.33.0"] {
        let mut args = prefix_core(
            glm_transfer_groups(version, false),
            22 * GLM_PAGE,
            8192,
            16,
            version,
        )
        .args;
        args.worker_type = WorkerType::Decode;
        args.max_num_seqs = Some(8);
        let mut core = VllmCore::new(args);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // (pass, load, id, shared, prompt, output)
        let arrivals: [(usize, bool, u128, u32, u32, usize); 4] = [
            (4, false, 1, 3264, 1546, 28),
            (7, false, 2, 3264, 2658, 1),
            (8, true, 3, 2720, 3446, 5),
            // A load of the first 2,926 tokens of the shared prefix. It waits.
            (9, true, 4, 3264, 2926, 8),
        ];
        let passes = 11;
        for pass in 0..passes {
            for &(_, load, id, shared, prompt, output) in arrivals.iter().filter(|a| a.0 == pass) {
                let tokens = (0..prompt)
                    .map(|i| {
                        if i < shared {
                            i
                        } else {
                            1_000_000 * id as u32 + i
                        }
                    })
                    .collect();
                if load {
                    let handoff_id = HandoffId::new(Uuid::from_u128(1_000_000 + id));
                    let request = DirectRequest {
                        tokens,
                        ..pd_request(id, 0, output)
                    };
                    if reserve_grouped(&mut core, handoff_id, request).is_some() {
                        activate(&mut core, handoff_id);
                    }
                } else {
                    submit_tokens(&mut core, id, tokens, output);
                }
            }
            step_pd(&mut core, &mut trace, &mut step);
            if pass + 1 == passes {
                break;
            }
            for event in core.retry_pending_destinations() {
                if let SchedulerLifecycleEvent::DestinationReserved { handoff_id, .. } = event {
                    activate(&mut core, handoff_id);
                }
            }
        }
        let head = HandoffId::new(Uuid::from_u128(1_000_004));
        // The state keys that the step before the last one published are no hit
        // in that step, as in vLLM (V31 `single:1726-1734`, V32
        // `single:1688-1696`), and load 4 waits. The step start of the last pass
        // makes them hits (V32 `sched:632`, `single:2008-2009`). The last pass
        // releases no block, publishes and evicts no key and leaves the need of
        // the loads in flight at 0. Only the published epoch of the coordinator
        // moves, and load 4 now fits, so the retry after this pass reserves it.
        assert_eq!(
            reserved_units(&core.retry_pending_destinations(), head),
            Some(1)
        );
    }
}

#[test]
fn grouped_destination_retries_when_a_pass_evicts_a_key() {
    let arrival = |pass, load, id, shared, prompt, output| Arrival {
        pass,
        load,
        id,
        shared,
        prompt,
        output,
    };
    let arrivals = [
        arrival(2, false, 1, 0, 80, 3),
        arrival(7, false, 2, 64, 41, 16),
        arrival(8, false, 3, 0, 54, 18),
        arrival(10, false, 4, 80, 75, 8),
        arrival(12, false, 5, 96, 99, 24),
        arrival(12, false, 6, 0, 61, 2),
        arrival(15, true, 7, 64, 74, 27),
        arrival(20, false, 8, 80, 66, 17),
        arrival(23, true, 9, 96, 59, 11),
        arrival(24, true, 10, 0, 76, 26),
        arrival(28, false, 11, 64, 39, 1),
        // A load of the first 71 tokens of the shared prefix. It waits.
        arrival(29, true, 12, 80, 71, 15),
    ];
    for version in ["0.32.0", "0.33.0"] {
        let mut core = scripted_decode_rank(&arrivals, 30, version);
        let head = HandoffId::new(Uuid::from_u128(1_000_012));
        // Load 12 waits after its arrival. The last pass releases no block and
        // publishes no key. It takes two of the five inactive cached blocks, and
        // one of them holds the last copy of a key, which moves the generation of
        // the resident keys. A decode rank counts each state group's own hit, so
        // that eviction can change what load 12 counts. The need of the loads in
        // flight stays 0, and the published epoch stays. Load 12 now fits, so
        // the retry after this pass reserves it.
        assert_eq!(core.kv_manager.grouped_inflight_need(), 0);
        assert_eq!(
            reserved_units(&core.retry_pending_destinations(), head),
            Some(2)
        );
    }
}

#[test]
fn grouped_destination_retries_when_a_pass_evicts_a_key_with_async_scheduling() {
    let arrival = |pass, load, id, shared, prompt, output| Arrival {
        pass,
        load,
        id,
        shared,
        prompt,
        output,
    };
    let arrivals = [
        arrival(5, false, 1, 80, 86, 5),
        arrival(5, false, 2, 64, 91, 2),
        arrival(9, true, 3, 96, 42, 11),
        arrival(10, false, 4, 64, 52, 19),
        arrival(12, false, 5, 80, 99, 13),
        arrival(17, false, 6, 0, 47, 11),
        arrival(17, false, 7, 64, 45, 4),
        arrival(17, true, 8, 0, 86, 4),
        arrival(19, false, 9, 80, 93, 5),
        arrival(20, false, 10, 0, 56, 20),
        arrival(22, true, 11, 96, 69, 25),
        arrival(22, true, 12, 96, 81, 15),
    ];
    for version in ["0.32.0", "0.33.0"] {
        let mut core = scripted_decode_rank_with(&arrivals, 27, 2, version);
        let head = HandoffId::new(Uuid::from_u128(1_000_012));
        // With async scheduling. Load 12 waits after its arrival. The last pass
        // releases no block and publishes no key, so the pool's own epoch stays.
        // It takes two of the six inactive cached blocks, and both hold the last
        // copy of a key, which moves the generation of the resident keys by 2.
        // The need of the loads in flight stays 0, and the published epoch
        // stays. Load 12 now fits, so the retry after this pass reserves it with
        // one transfer unit. Without the key generation in the retry epoch, that
        // retry skips it.
        assert_eq!(core.kv_manager.grouped_inflight_need(), 0);
        assert_eq!(
            reserved_units(&core.retry_pending_destinations(), head),
            Some(1)
        );
    }
}

#[test]
fn grouped_destination_keeps_the_first_pass_need_of_each_load() {
    // Seven pages in block mode. The first load holds 3 blocks, and its first
    // pass will take 2 more. A distinct second prompt needs 4 free blocks for
    // its whole prompt and finds 4, but its load of 3 blocks would leave 1 for
    // the first pass, so it waits. A cancel of the first load frees its blocks
    // and its need, and the retry reserves the second load.
    let mut args = prefix_core(full_and_state_transfer_groups(), 700, 8192, 16, "0.31.0").args;
    args.worker_type = WorkerType::Decode;
    let mut core = VllmCore::new(args);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let need = |core: &VllmCore| core.kv_manager.grouped_inflight_need();
    let first = HandoffId::new(Uuid::from_u128(1702));
    assert_eq!(
        reserve_grouped(&mut core, first, pd_request(1701, 33, 3)),
        Some(2)
    );
    assert_eq!(need(&core), 2);
    let second = HandoffId::new(Uuid::from_u128(1704));
    let request = DirectRequest {
        tokens: (1000..1033).collect(),
        ..pd_request(1703, 0, 3)
    };
    assert_eq!(reserve_grouped(&mut core, second, request), None);
    assert_eq!(need(&core), 2);
    let effects = core
        .apply_command_effects(
            SchedulerCommand::CancelDestination { handoff_id: first },
            true,
        )
        .unwrap();
    assert_eq!(reserved_units(&effects.lifecycle_events, second), Some(2));
    assert_eq!(need(&core), 2);
    // A load that activates and is dropped before its first pass gives its
    // need back too.
    activate(&mut core, second);
    assert_eq!(need(&core), 2);
    core.apply_command(SchedulerCommand::CancelDestination { handoff_id: second })
        .unwrap();
    assert_eq!(need(&core), 0);
    assert_eq!(block_pool(&core).active_blocks(), 0);
    let third = HandoffId::new(Uuid::from_u128(1706));
    let request = DirectRequest {
        tokens: (2000..2033).collect(),
        ..pd_request(1705, 0, 3)
    };
    assert_eq!(reserve_grouped(&mut core, third, request), Some(2));
    activate(&mut core, third);
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert_eq!(need(&core), 0);
    assert_eq!(block_pool(&core).active_blocks(), 5);
    drain_pd(&mut core, &mut trace, &mut step, 5);
    assert!(core.is_drained());
}

#[test]
fn grouped_destination_in_byte_mode_gives_back_the_need_of_a_dropped_load() {
    // The byte-mode twin of the drop in
    // `grouped_destination_keeps_the_first_pass_need_of_each_load`, on five
    // pages of 100 bytes. A load of 33 tokens holds 2 attention pages and the
    // state page, and its first pass needs one more attention page. A drop
    // after the activation and before the first pass gives that need back,
    // by a cancel of the request or of the handoff, as vLLM drops the request
    // from its prefills in flight (V31 `sched:2657`, V32 `sched:2721`). Then a
    // distinct load of 64 tokens, which fills the empty pool, is reserved.
    for cancel_handoff in [false, true] {
        let mut core = pd_core(
            full_and_state_transfer_groups(),
            500,
            32,
            WorkerType::Decode,
            None,
        );
        let need = |core: &VllmCore| core.kv_manager.grouped_inflight_need();
        let first = HandoffId::new(Uuid::from_u128(1871));
        assert_eq!(
            reserve_grouped(&mut core, first, pd_request(1870, 33, 3)),
            Some(2)
        );
        activate(&mut core, first);
        assert_eq!(need(&core), 100, "{cancel_handoff}");
        let command = if cancel_handoff {
            SchedulerCommand::CancelDestination { handoff_id: first }
        } else {
            SchedulerCommand::CancelRequest {
                request_id: Uuid::from_u128(1870),
            }
        };
        core.apply_command(command).unwrap();
        assert_eq!(need(&core), 0, "{cancel_handoff}");
        assert_eq!(grouped_pool(&core).used_bytes(), 0, "{cancel_handoff}");
        let second = HandoffId::new(Uuid::from_u128(1873));
        let request = DirectRequest {
            tokens: (1_000..1_064).collect(),
            ..pd_request(1872, 0, 2)
        };
        assert_eq!(
            reserve_grouped(&mut core, second, request),
            Some(4),
            "{cancel_handoff}"
        );
    }
}

#[test]
fn grouped_destination_caps_the_first_pass_need_of_a_window_group() {
    // Five pages and one batch of 32 tokens, so the window cap is
    // cdiv(15 + 32, 16) + 1 = 4 blocks in both modes (V31
    // `vllm/v1/kv_cache_interface.py:859-866`). A load of 65 tokens loads 64,
    // and its window holds cdiv(64, 16) = 4 blocks, null blocks included. So
    // vLLM counts min(5, 4) - 4 = 0 window blocks for the rest of the prompt
    // (V31 `vllm/v1/core/single_type_kv_cache_manager.py:216-238`), though
    // the first pass opens window block 4. The state counts 1 block in block
    // mode, as token 65 opens a new state block, and 0 bytes in byte mode,
    // which holds its state block already. So a distinct second load passes
    // the load test at once: 2 + 1 <= 3 free blocks in block mode, and
    // 300 + 0 <= 300 free bytes for a prompt of 66 tokens in byte mode.
    let block = {
        let mut args = prefix_core(window_transfer_groups(), 500, 32, 16, "0.31.0").args;
        args.worker_type = WorkerType::Decode;
        VllmCore::new(args)
    };
    let byte = pd_core(window_transfer_groups(), 500, 32, WorkerType::Decode, None);
    for (mut core, need, second_prompt, held) in [(block, 1, 65, 4), (byte, 0, 66, 5)] {
        let first = HandoffId::new(Uuid::from_u128(1781));
        assert_eq!(
            reserve_grouped(&mut core, first, pd_request(1780, 65, 2)),
            Some(0)
        );
        assert_eq!(core.kv_manager.grouped_inflight_need(), need);
        let second = HandoffId::new(Uuid::from_u128(1783));
        let request = DirectRequest {
            tokens: (1_000..1_000 + second_prompt).collect(),
            ..pd_request(1782, 0, 2)
        };
        assert_eq!(reserve_grouped(&mut core, second, request), Some(0));
        // Then the first load cannot run its first pass, as in vLLM. In block
        // mode each first pass needs a window block and a state block, and 1
        // block is free. In byte mode the first load needs a window page, and
        // no byte is free. A waiting request does not preempt, and the
        // waiting loop stops at the first request that does not fit (V31
        // `vllm/v1/core/sched/scheduler.py:1223-1230`). So neither load runs,
        // and with no request running the rank stops.
        activate(&mut core, first);
        activate(&mut core, second);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        for _ in 0..3 {
            assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
        }
        assert!(core.state.running.is_empty());
        for id in [1780, 1782] {
            assert_eq!(
                core.state.requests[&Uuid::from_u128(id)].status,
                RequestStatus::Waiting
            );
        }
        assert_eq!(grouped_pool(&core).active_blocks(), held);
    }
}

#[test]
fn grouped_destination_in_byte_mode_waits_for_its_whole_prompt_with_no_load_in_flight() {
    // Six pages of 100 bytes in byte mode. The first load of 32 tokens holds
    // 3 pages, and its first pass takes a third attention page. Then nothing
    // is in flight, and 2 pages are free. A distinct prompt of 17 tokens loads
    // 16 tokens into 2 pages, which fit, but its whole prompt needs 3, so it
    // waits, as vLLM's whole-prompt test refuses it (V32
    // `vllm/v1/core/kv_cache_manager.py:522-539`).
    let mut core = pd_core(
        full_and_state_transfer_groups(),
        600,
        8192,
        WorkerType::Decode,
        None,
    );
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let first = HandoffId::new(Uuid::from_u128(1712));
    assert_eq!(
        reserve_grouped(&mut core, first, pd_request(1711, 33, 3)),
        Some(2)
    );
    activate(&mut core, first);
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert_eq!(grouped_pool(&core).active_blocks(), 4);
    let second = HandoffId::new(Uuid::from_u128(1714));
    let request = DirectRequest {
        tokens: (1000..1017).collect(),
        ..pd_request(1713, 0, 3)
    };
    assert_eq!(reserve_grouped(&mut core, second, request), None);
    drain_pd(&mut core, &mut trace, &mut step, 4);
    let events = core.retry_pending_destinations();
    assert_eq!(reserved_units(&events, second), Some(1));
    activate(&mut core, second);
    drain_pd(&mut core, &mut trace, &mut step, 3);
    assert!(core.is_drained());
}

#[test]
fn grouped_destination_in_byte_mode_retries_when_a_window_page_leaves() {
    // Five pages in byte mode, with a window of 16 tokens and blocks of 16.
    // The first load of 30 of 31 tokens holds 2 window pages and a state
    // page, and its first pass keeps them. A distinct second prompt of 31
    // tokens needs 3 free pages and finds 2, so it waits. The next pass
    // computes token 32, whose window starts at token 16, so the first window
    // page leaves. The retry after that pass reserves the second load.
    let mut core = pd_core(
        window_transfer_groups(),
        500,
        8192,
        WorkerType::Decode,
        None,
    );
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let first = HandoffId::new(Uuid::from_u128(1716));
    assert_eq!(
        reserve_grouped(&mut core, first, pd_request(1715, 31, 8)),
        Some(0)
    );
    let second = HandoffId::new(Uuid::from_u128(1718));
    let request = DirectRequest {
        tokens: (1000..1031).collect(),
        ..pd_request(1717, 0, 8)
    };
    assert_eq!(reserve_grouped(&mut core, second, request), None);
    activate(&mut core, first);
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert_eq!(grouped_pool(&core).active_blocks(), 3);
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert_eq!(grouped_pool(&core).active_blocks(), 2);
    let events = core.retry_pending_destinations();
    assert_eq!(reserved_units(&events, second), Some(0));
    activate(&mut core, second);
    drain_pd(&mut core, &mut trace, &mut step, 5);
    assert!(core.is_drained());
}

#[test]
fn grouped_destination_on_vllm_0_32_counts_no_checkpoint_block_for_its_load() {
    // A cold destination of 2,177 tokens on vLLM 0.32.0 or 0.33.0 with
    // checkpoint blocks, on 10 pages. The loaded tokens count as computed, so
    // the query of the whole-prompt test runs from 2,176 to 2,177, where no
    // checkpoint position fits (V32 `vllm/v1/kv_cache_interface.py:1125-1146`).
    // The test counts 7 blocks, not the 11 of a query from 0, so the load fits.
    for version in ["0.32.0", "0.33.0"] {
        let mut core = block_mode_decode_core(10 * GLM_PAGE, version, true);
        let handoff_id = HandoffId::new(Uuid::from_u128(1720));
        assert_eq!(
            reserve_grouped(&mut core, handoff_id, pd_request(1719, 2_177, 3)),
            Some(1)
        );
        assert_eq!(block_pool(&core).active_blocks(), 6);
        core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
            .unwrap();
        assert!(core.is_drained());
    }
}

/// A decode rank with full-attention and state blocks of 16 tokens on
/// `pages` pages, at a hash block of `unit` tokens.
fn one_token_load_core(
    version: &str,
    unit: usize,
    pages: u64,
    max_concurrent_batches: usize,
) -> VllmCore {
    let mut groups = full_and_state_transfer_groups();
    if version != "0.31.0" {
        groups[1].prefill_checkpoint_blocks = Some(0);
    }
    let mut args = prefix_core(groups, pages * 100, 8192, unit, version).args;
    args.worker_type = WorkerType::Decode;
    args.max_concurrent_batches = max_concurrent_batches;
    VllmCore::new(args)
}

fn one_token_load(id: u128, token: u32) -> DirectRequest {
    DirectRequest {
        tokens: vec![token],
        ..pd_request(id, 1, 2)
    }
}

#[test]
fn grouped_decode_rank_local_hit_inside_a_block_keeps_its_copy_sources() {
    // The tight-pool case on a grouped decode rank with transfer rules, as
    // local requests and with one batch, where vLLM frees the copy sources
    // of RC's hit inside a block at the end of `schedule()` (v0.31
    // sched:1434-1442, v0.32 sched:1474-1482). RC holds its 7 blocks and the
    // 5 sources in its pass, and RN waits one pass on pools of 13 to 17
    // blocks, as on an aggregated rank.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for blocks in 12..=18 {
            let expected = match blocks {
                12 => (0, 4, vec![7, 7, 7, 0, 6, 6, 6, 0], 12),
                18 => (0, 0, vec![13, 13, 13, 0], 18),
                _ => (0, 1, vec![7, 13, 13, 6, 0], 12),
            };
            let mut args = pool_args(version, blocks, 1, 4352);
            args.kv_cache_groups = glm_transfer_groups(version, false);
            args.worker_type = WorkerType::Decode;
            assert_eq!(
                pool_run(VllmCore::new(args), 3500, 3264),
                expected,
                "{version} on {blocks} blocks"
            );
        }
    }
}

#[test]
fn grouped_destination_with_a_local_prefix_inside_a_block_takes_no_copy_source() {
    // A decode rank with full-attention and state blocks of 16 tokens at a
    // hash block of 4. A local request P leaves its blocks cached, with a
    // full-attention key at 36 tokens, inside the third block. vLLM 0.31.0
    // keys a prompt's partial tail at its last hash boundary and vLLM 0.32.0
    // at the last one below its last token, so P has 36 tokens on 0.31.0 and
    // 40 on 0.32.0. A load of P's first 36 tokens and one more transfers 36
    // tokens and would hit all of them as a local request. Its local prefix
    // is the full-attention hit cut to whole blocks, 32 tokens. So the first
    // allocation of its reservation has no partial hit block to copy: it
    // pins the two hit blocks, takes fresh blocks for the rest (one
    // full-attention block and the running state block) and holds no copy
    // source. Its activation holds none either. Its first pass takes one
    // block for the copy of the state that the load published at 36 tokens,
    // so the pass holds 5 blocks at its peak.
    for (version, cached) in [("0.31.0", 36), ("0.32.0", 40), ("0.33.0", 40)] {
        let mut groups = full_and_state_transfer_groups();
        if version != "0.31.0" {
            groups[1].prefill_checkpoint_blocks = Some(0);
        }
        let mut args = prefix_core(groups, 32 * 100, 8192, 4, version).args;
        args.worker_type = WorkerType::Decode;
        let mut core = VllmCore::new(args);
        let mut now = 0.0;
        let prefix: Vec<u32> = (5000..5000 + cached).collect();
        submit_tokens(&mut core, 1, prefix.clone(), 1);
        while !core.is_drained() {
            async_step(&mut core, &mut now);
        }
        let mut tokens = prefix[..36].to_vec();
        tokens.push(9);
        let handoff_id = HandoffId::new(Uuid::from_u128(3001));
        let request = DirectRequest {
            tokens,
            ..pd_request(3000, 37, 2)
        };
        // 36 transferred tokens in 3 blocks, the first two local.
        assert_eq!(
            reserve_grouped(&mut core, handoff_id, request),
            Some(1),
            "{version}"
        );
        assert_eq!(core.kv_manager.grouped_copy_source_count(), 0, "{version}");
        activate(&mut core, handoff_id);
        assert_eq!(block_pool(&core).active_blocks(), 4, "{version}");
        block_pool(&core).reset_peak_active_blocks();
        let pass = async_step(&mut core, &mut now);
        assert_eq!(pass.output_signals.len(), 1, "{version}");
        assert_eq!(block_pool(&core).peak_active_blocks(), 5, "{version}");
    }
}

#[test]
fn grouped_decode_one_token_load_takes_the_copy_of_its_state_in_its_first_pass() {
    // A prompt of one token is not longer than the backoff, so the decode
    // rank loads the whole prompt (V31 `nixl/base_scheduler.py:400-407`, V32
    // `:396-403`) and computes its token again (V31 `sched:3095-3096`, V32
    // `sched:3171-3172`). With a hash block of 1 token, vLLM 0.31.0 caches
    // the loaded state at token 1, the prompt's last hash boundary, inside
    // its 16-token block (V31 `single:2094-2138`), and the first step takes
    // one block for its copy (V31 `single:1770-1777`, `:1891-1940`). vLLM
    // 0.32.0 keeps that boundary below the last prompt token (V32
    // `kv_cache_interface.py:1110-1122`, `single:2067-2084`), and a hash
    // block of 16 has no boundary at 1 token, so these take no copy.
    for (version, unit, copy) in [
        ("0.31.0", 1, 1),
        ("0.31.0", 16, 0),
        ("0.32.0", 1, 0),
        ("0.33.0", 1, 0),
    ] {
        for max_concurrent_batches in [1, 2] {
            let label = format!("{version} {unit} {max_concurrent_batches}");
            let mut core = one_token_load_core(version, unit, 16, max_concurrent_batches);
            let handoff_id = HandoffId::new(Uuid::from_u128(2001));
            assert_eq!(
                reserve_grouped(&mut core, handoff_id, one_token_load(1001, 7)),
                Some(1),
                "{label}"
            );
            assert_eq!(core.kv_manager.grouped_inflight_need(), 0, "{label}");
            activate(&mut core, handoff_id);
            // One attention block and one state block.
            assert_eq!(block_pool(&core).active_blocks(), 2, "{label}");
            assert_eq!(core.kv_manager.grouped_inflight_need(), copy, "{label}");
            let mut now = 0.0;
            let first = async_step(&mut core, &mut now);
            assert_eq!(first.output_signals.len(), 1, "{label}");
            assert_eq!(core.kv_manager.grouped_inflight_need(), 0, "{label}");
            assert_eq!(
                block_pool(&core).active_blocks(),
                2 + copy as usize,
                "{label}"
            );
        }
    }
}

#[test]
fn grouped_decode_one_token_loads_wait_for_the_block_of_the_copy() {
    // vLLM 0.31.0 at a hash block of 1 token. Two loads of one token are
    // reserved before either is activated: each holds 2 blocks, and each
    // needs one more block for the copy of its state in its first step.
    // On 6 pages both run in pass 0 with 6 blocks. On 5 pages only the first
    // one fits its copy; the second one runs after the first one ended. On
    // 4 pages no block is free, so neither runs: vLLM stops its waiting loop
    // at a request that it cannot allocate (V31 `sched:1223-1230`).
    for max_concurrent_batches in [1, 2] {
        for (pages, first_pass, active) in [
            (6u64, vec![1001u128, 1002], 6usize),
            (5, vec![1001], 5),
            (4, vec![], 4),
        ] {
            let label = format!("{max_concurrent_batches} {pages}");
            let mut core = one_token_load_core("0.31.0", 1, pages, max_concurrent_batches);
            let a = HandoffId::new(Uuid::from_u128(2001));
            let b = HandoffId::new(Uuid::from_u128(2002));
            assert_eq!(
                reserve_grouped(&mut core, a, one_token_load(1001, 7)),
                Some(1),
                "{label}"
            );
            assert_eq!(
                reserve_grouped(&mut core, b, one_token_load(1002, 9)),
                Some(1),
                "{label}"
            );
            activate(&mut core, a);
            activate(&mut core, b);
            assert_eq!(block_pool(&core).active_blocks(), 4, "{label}");
            assert_eq!(core.kv_manager.grouped_inflight_need(), 2, "{label}");
            let mut now = 0.0;
            let pass = async_step(&mut core, &mut now);
            let outputs: Vec<u128> = pass
                .output_signals
                .iter()
                .map(|output| output.uuid.as_u128())
                .collect();
            assert_eq!(outputs, first_pass, "{label}");
            assert_eq!(block_pool(&core).active_blocks(), active, "{label}");
            assert_eq!(
                core.kv_manager.grouped_inflight_need(),
                2 - first_pass.len() as u64,
                "{label}"
            );
            assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{label}");
            if pages == 4 {
                // Nothing changes in the next passes either.
                for _ in 0..3 {
                    let pass = async_step(&mut core, &mut now);
                    assert!(pass.output_signals.is_empty(), "{label}");
                }
                assert_eq!(block_pool(&core).active_blocks(), 4, "{label}");
            } else {
                let mut completed = 0;
                for _ in 0..8 {
                    let pass = async_step(&mut core, &mut now);
                    assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{label}");
                    completed += pass
                        .output_signals
                        .iter()
                        .filter(|output| output.completed)
                        .count();
                }
                assert_eq!(completed, 2, "{label}");
                assert_eq!(block_pool(&core).active_blocks(), 0, "{label}");
            }
        }
    }
}

/// A full-attention group and a 17-token sliding window, both of 16-token
/// blocks at 100 bytes a page, with the transfer fields set.
fn full_and_window_transfer_groups() -> Vec<FpmCacheGroup> {
    let mut full = group("full", 16, 100, None);
    full.prefix_cacheable = Some(true);
    full.transfer_bytes_per_block = Some(100);
    let mut window = group("window", 16, 100, Some(17));
    window.prefix_cacheable = Some(true);
    window.transfer_bytes_per_block = Some(100);
    vec![full, window]
}

#[test]
fn grouped_destination_without_state_layers_loads_past_the_hit_every_group_agrees_on() {
    let mut args = prefix_core(full_and_window_transfer_groups(), 4_000, 8192, 16, "0.31.0").args;
    args.worker_type = WorkerType::Decode;
    let mut core = VllmCore::new(args);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    // Without a state group the load is the whole prompt: cdiv(100, 16) = 7 units.
    let first = HandoffId::new(Uuid::from_u128(1761));
    assert_eq!(
        reserve_grouped(&mut core, first, pd_request(1760, 100, 2)),
        Some(7)
    );
    activate(&mut core, first);
    // 7 full-attention blocks and the 2 window blocks inside the window of 100.
    drain_pd(&mut core, &mut trace, &mut step, 9);
    // The first request cached one window block, the one that ends at its
    // replay boundary 96, and the full-attention block that ends at 64.
    let mut shared = BlockRequestLease::new(Uuid::nil(), Vec::new());
    shared.configure_prefix_hashes(&(0..64).collect::<Vec<u32>>(), 16);
    assert!(
        block_pool(&core)
            .key_hit(0, shared.prefix_hash_at(64).unwrap())
            .is_some()
    );
    // The second prompt shares 64 tokens. Its full-attention hit is 64, but
    // the window has no cached block before 64, so the hit that every group
    // agrees on is 0 (`vllm/v1/core/kv_cache_manager.py:346-351`) and all 7
    // units move.
    let second = HandoffId::new(Uuid::from_u128(1763));
    let request = DirectRequest {
        tokens: (0..64).chain(1_000..1_036).collect(),
        ..pd_request(1762, 0, 2)
    };
    assert_eq!(reserve_grouped(&mut core, second, request), Some(7));
}

#[test]
fn grouped_destination_waits_for_the_cached_block_of_a_lagging_state_hit() {
    let mut args = prefix_core(full_and_state_transfer_groups(), 600, 8192, 16, "0.31.0").args;
    args.worker_type = WorkerType::Decode;
    let mut core = VllmCore::new(args);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    // A caches full attention at 16 and 32 and its state at 32, its replay boundary.
    let a = HandoffId::new(Uuid::from_u128(1771));
    assert_eq!(
        reserve_grouped(&mut core, a, pd_request(1770, 33, 1)),
        Some(2)
    );
    activate(&mut core, a);
    drain_pd(&mut core, &mut trace, &mut step, 5);
    // B adds full attention at 48. Its state after 54 tokens sits at no boundary.
    // B also counts the state hit of A at 32: its whole prompt needs
    // 2 + 2 + 1 + 1 = 6 blocks of 6 free, so B still gets 2 units.
    let b = HandoffId::new(Uuid::from_u128(1773));
    assert_eq!(
        reserve_grouped(&mut core, b, pd_request(1772, 55, 1)),
        Some(2)
    );
    activate(&mut core, b);
    drain_pd(&mut core, &mut trace, &mut step, 5);
    assert_eq!(block_pool(&core).inactive_cached_blocks(), 4);
    // C: local prefix 48, state hit 32. Whole prompt 7 > 6 free blocks.
    let c = HandoffId::new(Uuid::from_u128(1775));
    let request = DirectRequest {
        tokens: (0..48).chain(1_000..1_017).collect(),
        ..pd_request(1774, 0, 2)
    };
    assert_eq!(reserve_grouped(&mut core, c, request), None);
}

#[test]
fn grouped_destination_transfers_every_block_of_an_uncacheable_full_attention_group() {
    // A full-attention group that is not prefix-cacheable and a cacheable
    // state group, both of 16-token blocks, in block mode on 20 pages. The
    // full-attention group never caches a block (V31 and V32
    // `vllm/v1/core/single_type_kv_cache_manager.py:498-499`). vLLM pulls
    // the blocks without a hash (`nixl/pull_scheduler.py:149-156`, the same
    // in V31 and V32), so a decode rank pulls every block of that group.
    let mut groups = full_and_state_transfer_groups();
    groups[0].prefix_cacheable = Some(false);
    let mut args = prefix_core(groups, 2_000, 32, 16, "0.31.0").args;
    args.worker_type = WorkerType::Decode;
    let mut core = VllmCore::new(args);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    // A load of 33 tokens loads 32, as the state group backs off one token:
    // cdiv(32, 16) = 2 units. Its first pass computes token 32, so it opens a
    // third full-attention block and a new state block, and the previous
    // state block stays for one step: 5 blocks. Of its blocks, only the state
    // at its replay boundary 32 stays cached.
    let first = HandoffId::new(Uuid::from_u128(1901));
    assert_eq!(
        reserve_grouped(&mut core, first, pd_request(1900, 33, 2)),
        Some(2)
    );
    activate(&mut core, first);
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert_eq!(block_pool(&core).active_blocks(), 5);
    assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 5), 1);
    assert_eq!(block_pool(&core).inactive_cached_blocks(), 1);
    // A prompt of 49 tokens that shares those 33 tokens loads 48: cdiv(48,
    // 16) = 3 units. Without a cacheable full-attention group, its local
    // prefix is the hit that every group agrees on, the state hit at 32. That
    // hit keeps no full-attention block, so all 3 units move.
    let second = HandoffId::new(Uuid::from_u128(1903));
    assert_eq!(
        reserve_grouped(&mut core, second, pd_request(1902, 49, 2)),
        Some(3)
    );
}

#[test]
fn grouped_destination_transfers_every_unit_when_one_of_two_full_attention_groups_is_uncacheable() {
    // Two full-attention groups of 16-token blocks at 100 bytes a page, one
    // prefix-cacheable and one not, in block mode on 20 pages. A transfer
    // unit is one block of every full-attention group, and the group that is
    // not prefix-cacheable keeps no hit block. So every unit moves, whatever
    // the cacheable group keeps.
    let mut groups = full_transfer_group();
    let mut uncached = group("uncached", 16, 100, None);
    uncached.prefix_cacheable = Some(false);
    uncached.transfer_bytes_per_block = Some(100);
    groups.push(uncached);
    let mut args = prefix_core(groups, 2_000, 8192, 16, "0.31.0").args;
    args.worker_type = WorkerType::Decode;
    let mut core = VllmCore::new(args);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    // Without a state group the load is the whole prompt: cdiv(33, 16) = 3
    // units. The request holds 3 blocks in each group.
    let first = HandoffId::new(Uuid::from_u128(1951));
    assert_eq!(
        reserve_grouped(&mut core, first, pd_request(1950, 33, 2)),
        Some(3)
    );
    activate(&mut core, first);
    drain_pd(&mut core, &mut trace, &mut step, 6);
    // The cacheable group keeps its blocks that end at 16 and 32.
    let mut shared = BlockRequestLease::new(Uuid::nil(), Vec::new());
    shared.configure_prefix_hashes(&(0..32).collect::<Vec<u32>>(), 16);
    for end in [16, 32] {
        assert!(
            block_pool(&core)
                .key_hit(0, shared.prefix_hash_at(end).unwrap())
                .is_some()
        );
    }
    // A prompt of 49 tokens that shares those 33 tokens loads cdiv(49, 16) =
    // 4 units. Its local prefix is the hit of 32 tokens of the cacheable
    // group, which keeps 2 blocks. The other group keeps none, so all 4
    // units move, not 4 - 32 / 16 = 2.
    let second = HandoffId::new(Uuid::from_u128(1953));
    assert_eq!(
        reserve_grouped(&mut core, second, pd_request(1952, 49, 2)),
        Some(4)
    );
}

/// One full-attention group of 16-token blocks at 100 bytes a page, with
/// the transfer fields set. No state group, so a load is the whole prompt.
fn full_transfer_group() -> Vec<FpmCacheGroup> {
    let mut full = group("full", 16, 100, None);
    full.prefix_cacheable = Some(true);
    full.transfer_bytes_per_block = Some(100);
    vec![full]
}

#[test]
fn grouped_destination_keeps_free_what_a_local_prefill_still_needs() {
    let block = {
        let mut args = prefix_core(full_transfer_group(), 1_000, 32, 16, "0.31.0").args;
        args.worker_type = WorkerType::Decode;
        VllmCore::new(args)
    };
    let byte = pd_core(full_transfer_group(), 1_000, 32, WorkerType::Decode, None);
    for mut core in [block, byte] {
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // A local prompt of 100 tokens holds 2 blocks after its first chunk
        // and still needs cdiv(100, 16) - 2 = 5 (V32
        // `single_type_kv_cache_manager.py:229-235`).
        submit_tokens(&mut core, 1811, (0..100).collect(), 1);
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
        // A load of 64 tokens needs 4 blocks, and 4 > 8 free - 5.
        let handoff_id = HandoffId::new(Uuid::from_u128(1813));
        let request = DirectRequest {
            tokens: (1_000..1_064).collect(),
            ..pd_request(1812, 0, 2)
        };
        assert_eq!(reserve_grouped(&mut core, handoff_id, request), None);
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 7), 1);
        assert_eq!(
            reserved_units(&core.retry_pending_destinations(), handoff_id),
            Some(4)
        );
    }
}

#[test]
fn grouped_decode_rank_takes_a_local_prefill_out_of_the_set_at_preemption_drop_and_last_chunk() {
    // A decode rank with one full-attention group of 16-token blocks on 10
    // pages and one batch of 32 tokens, in block and in byte mode. A local
    // prompt of 100 tokens runs a first chunk of 32, which ends before its
    // known tokens, so the request joins the local prefills in flight
    // (V32 `sched:1376-1378`). It leaves them when it is preempted (V32
    // `sched:1591`) and when it is dropped (V32 `sched:2721`), with no chunk
    // that reaches its known tokens.
    let block = || {
        let mut args = prefix_core(full_transfer_group(), 1_000, 32, 16, "0.31.0").args;
        args.worker_type = WorkerType::Decode;
        VllmCore::new(args)
    };
    let byte = || pd_core(full_transfer_group(), 1_000, 32, WorkerType::Decode, None);
    let id = Uuid::from_u128(1841);
    for (mode, make) in [("block", &block as &dyn Fn() -> VllmCore), ("byte", &byte)] {
        // Preemption.
        let mut core = make();
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1841, (0..100).collect(), 1);
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
        assert!(core.grouped_local_prefills().contains(&id), "{mode}");
        assert!(core.policy_preempt(0.0).is_some());
        assert!(core.grouped_local_prefills().is_empty(), "{mode}");
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 7), 1);
        assert!(core.grouped_local_prefills().is_empty(), "{mode}");

        // Drop in the middle of the prompt.
        let mut core = make();
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1841, (0..100).collect(), 1);
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
        assert!(core.grouped_local_prefills().contains(&id), "{mode}");
        core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
            .unwrap();
        assert!(core.grouped_local_prefills().is_empty(), "{mode}");
        assert!(core.is_drained(), "{mode}");

        // The last chunk. The second chunk of 16 tokens of a 48-token prompt
        // reaches its known tokens, so the request leaves the set there (V32
        // `sched:1640-1642`), before it emits its first token. It finishes
        // with the second token outside the set.
        let mut core = make();
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1841, (0..48).collect(), 2);
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
        assert!(core.grouped_local_prefills().contains(&id), "{mode}");
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
        assert!(core.grouped_local_prefills().is_empty(), "{mode}");
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 4), 1);
        assert!(core.grouped_local_prefills().is_empty(), "{mode}");
    }
}

#[test]
fn grouped_destination_keeps_free_what_a_preempted_recompute_still_needs() {
    // Byte mode, the sweeper default. A load of 100 tokens activates, emits
    // one token and is preempted. Its recompute of 101 tokens runs a first
    // chunk of 32 and still needs cdiv(101, 16) - 2 = 5 pages.
    let mut core = pd_core(full_transfer_group(), 1_000, 32, WorkerType::Decode, None);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let first = HandoffId::new(Uuid::from_u128(1821));
    assert_eq!(
        reserve_grouped(&mut core, first, pd_request(1820, 100, 3)),
        Some(7)
    );
    activate(&mut core, first);
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
    assert!(core.policy_preempt(0.0).is_some());
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
    let second = HandoffId::new(Uuid::from_u128(1823));
    let request = DirectRequest {
        tokens: (1_000..1_064).collect(),
        ..pd_request(1822, 0, 2)
    };
    assert_eq!(reserve_grouped(&mut core, second, request), None);
    assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 7), 2);
    assert_eq!(
        reserved_units(&core.retry_pending_destinations(), second),
        Some(4)
    );
}

#[test]
fn grouped_destination_counts_nothing_for_a_local_request_that_decodes() {
    // A local prompt of 48 tokens runs chunks of 32 and 16 tokens. The second
    // chunk reaches its known tokens, so the request leaves the prefills in
    // flight (V32 `sched:1640-1642`) and emits a token. Its next decode step
    // opens a fourth block, which vLLM does not keep free for a load. So a
    // load of 64 tokens takes the last 4 of 7 blocks.
    let block = {
        let mut args = prefix_core(full_transfer_group(), 700, 32, 16, "0.31.0").args;
        args.worker_type = WorkerType::Decode;
        VllmCore::new(args)
    };
    let byte = pd_core(full_transfer_group(), 700, 32, WorkerType::Decode, None);
    for mut core in [block, byte] {
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1831, (0..48).collect(), 4);
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
        let handoff_id = HandoffId::new(Uuid::from_u128(1833));
        let request = DirectRequest {
            tokens: (1_000..1_064).collect(),
            ..pd_request(1832, 0, 2)
        };
        assert_eq!(reserve_grouped(&mut core, handoff_id, request), Some(4));
        core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
            .unwrap();
        assert_eq!(drain_pd(&mut core, &mut trace, &mut step, 4), 3);
        assert!(core.is_drained());
    }
}

#[test]
fn grouped_destination_retries_when_the_local_prefills_need_less() {
    // Byte mode on 14 pages of 100 bytes, with one batch of 32 tokens. So the
    // cap of the 17-token window is cdiv(16 + 32, 16) + 1 = 4 blocks (V31
    // `vllm/v1/kv_cache_interface.py:859-866`).
    let mut core = pd_core(
        full_and_window_transfer_groups(),
        1_400,
        32,
        WorkerType::Decode,
        None,
    );
    let mut trace = TraceCollector::default();
    let mut step = 0;
    // A local prompt of 100 tokens. Its first chunk of 32 tokens holds 2
    // full-attention pages and 2 window pages: 400 bytes. It still needs
    // cdiv(100, 16) - 2 = 5 full-attention pages and min(7, 4) - 2 = 2 window
    // pages: 700 bytes.
    let local = submit_tokens(&mut core, 1880, (0..100).collect(), 1);
    let need = |core: &VllmCore| {
        let lease = &core.state.requests[&local].sequence.lease;
        core.kv_manager.grouped_remaining_need(lease, 100)
    };
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
    assert_eq!(grouped_pool(&core).used_bytes(), 400);
    assert_eq!(need(&core), 700);
    // Without a state group, a load of 48 tokens loads the whole prompt in
    // cdiv(48, 16) = 3 units. It holds 3 full-attention pages and the window
    // page of tokens 32 to 47: 400 bytes. It waits, as 400 + 700 > 1,000
    // free bytes.
    let handoff_id = HandoffId::new(Uuid::from_u128(1882));
    let request = DirectRequest {
        tokens: (1_000..1_048).collect(),
        ..pd_request(1881, 0, 2)
    };
    assert_eq!(reserve_grouped(&mut core, handoff_id, request), None);
    // The second chunk adds 2 full-attention pages, and the window moves
    // from pages 0 and 1 to pages 1 to 3: 700 bytes. So the pool grows by
    // 300 bytes, and the retry epoch stays, as the request's bytes grow.
    // The need falls by 400 bytes, to 7 - 4 = 3 full-attention pages and
    // min(7, 4) - 4 = 0 window pages. It counts cdiv(64, 16) = 4 window
    // pages as held, page 0 included, though page 0 left the window and
    // freed its 100 bytes. The cap sets the level of 300 bytes, so the load
    // now fits, as 400 + 300 <= 700 free bytes. vLLM stops its waiting loop
    // at a load that does not fit (V31 `sched:1223-1230`, V32
    // `sched:1275-1282`) and tests the load again at the next step that
    // preempted no request and has token budget left (V31 `sched:873`,
    // `:885`, V32 `sched:900`, `:912`). Replay tests it again after this
    // pass, because the need fell.
    assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
    assert_eq!(grouped_pool(&core).used_bytes(), 700);
    assert_eq!(need(&core), 300);
    assert_eq!(
        reserved_units(&core.retry_pending_destinations(), handoff_id),
        Some(3)
    );
}

#[test]
fn grouped_destination_in_block_mode_retries_when_the_local_prefills_need_less() {
    // The GLM-5.3-Flash groups on vLLM 0.32.0 or 0.33.0 with checkpoint
    // blocks, in block mode on 20 pages, with a hash block of 16 and one batch
    // of 600 tokens. A local prompt of 5,000 tokens runs chunks of 600
    // tokens. Below token 2,176 it holds 1 MLA block, the ring block and 1
    // block of each of the 4 KDA groups: 6 blocks, so 14 are free.
    for version in ["0.32.0", "0.33.0"] {
        let mut args = prefix_core(
            glm_transfer_groups(version, true),
            20 * GLM_PAGE,
            600,
            16,
            version,
        )
        .args;
        args.worker_type = WorkerType::Decode;
        let mut core = VllmCore::new(args);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // The prompt still needs cdiv(5,000, 2,176) - 1 = 2 MLA blocks. The ring
        // holds its block and needs none. Each KDA group needs 1 block, plus a
        // checkpoint block while the rest of the prompt can export one (V32
        // `vllm/v1/core/single_type_kv_cache_manager.py:1724-1773`). The
        // checkpoint sits at (5,000 - 1) / 16 * 16 = 4,992, and a query exports
        // it only from a start on a hash block, among other conditions (V32
        // `vllm/v1/kv_cache_interface.py:1110-1146`). 600 is not on one, so the
        // need is 2 + 4 = 6 blocks.
        let local = submit_tokens(&mut core, 1890, (0..5_000).collect(), 1);
        let need = |core: &VllmCore| {
            let lease = &core.state.requests[&local].sequence.lease;
            core.kv_manager.grouped_remaining_need(lease, 5_000)
        };
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
        assert_eq!(need(&core), 6);
        // 1,200 is on a hash block, so the need is 2 + 4 * 2 = 10 blocks. A
        // distinct load of 1,000 tokens loads 999. Both of its tests count 6
        // blocks: 1 MLA block, the ring block and 1 block per KDA group. It
        // waits, as 6 + 10 > 14.
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
        assert_eq!(block_pool(&core).active_blocks(), 6);
        assert_eq!(need(&core), 10);
        let handoff_id = HandoffId::new(Uuid::from_u128(1892));
        let request = DirectRequest {
            tokens: (100_000..101_000).collect(),
            ..pd_request(1891, 0, 2)
        };
        assert_eq!(reserve_grouped(&mut core, handoff_id, request), None);
        // The third chunk ends at 1,800, still inside the first MLA and KDA
        // blocks. It takes and frees no block, so the pool's retry epoch is still
        // 0. But 1,800 is not on a hash block, so the need falls to 6 blocks, and
        // the load fits, as 6 + 6 <= 14.
        assert_eq!(step_pd(&mut core, &mut trace, &mut step), 0);
        assert_eq!(block_pool(&core).active_blocks(), 6);
        assert_eq!(grouped_pool(&core).retry_epoch(), 0);
        assert_eq!(need(&core), 6);
        assert_eq!(
            reserved_units(&core.retry_pending_destinations(), handoff_id),
            Some(1)
        );
    }
}

#[test]
fn grouped_decode_rank_frees_the_skipped_blocks_of_an_ending_request_first() {
    let hash = prompt_hash(2_177, 2_176);
    for (version, checkpoint) in GLM_CORES {
        for drop in [false, true] {
            let mut core = block_mode_decode_core(11 * GLM_PAGE, version, checkpoint);
            let mut trace = TraceCollector::default();
            let mut step = 0;
            let first = HandoffId::new(Uuid::from_u128(1851));
            let output = if drop { 3 } else { 1 };
            assert_eq!(
                reserve_grouped(&mut core, first, pd_request(1850, 2_177, output)),
                Some(1)
            );
            activate(&mut core, first);
            // The first pass keeps each KDA group's previous block, keyed at 2,176.
            assert_eq!(step_pd(&mut core, &mut trace, &mut step), 1);
            if drop {
                core.apply_command(SchedulerCommand::CancelRequest {
                    request_id: Uuid::from_u128(1850),
                })
                .unwrap();
            }
            assert_eq!(
                block_pool(&core).inactive_cached_blocks(),
                5,
                "{version} {checkpoint} {drop}"
            );
            // 7 fresh blocks from 6 free ones evict the oldest cached block.
            let second = HandoffId::new(Uuid::from_u128(1853));
            let request = DirectRequest {
                tokens: (100_000..104_353).collect(),
                ..pd_request(1852, 0, 2)
            };
            assert_eq!(reserve_grouped(&mut core, second, request), Some(2));
            assert!(
                block_pool(&core).key_hit(0, hash).is_some(),
                "{version} {checkpoint} {drop}"
            );
            assert!(
                block_pool(&core).key_hit(2, hash).is_none(),
                "{version} {checkpoint} {drop}"
            );
            for g in 3..6 {
                assert!(
                    block_pool(&core).key_hit(g, hash).is_some(),
                    "{version} {checkpoint} {drop}"
                );
            }
        }
    }
}

/// An 8-token sliding window of 4-token blocks and a ring of one 4-token
/// block per request, both at 8 bytes a page, with the transfer fields set.
fn window_and_ring_transfer_groups() -> Vec<FpmCacheGroup> {
    let mut window = group("window", 4, 8, Some(8));
    window.prefix_cacheable = Some(true);
    window.transfer_bytes_per_block = Some(8);
    let mut ring = state("ring", 8, 1);
    ring.prefix_cacheable = Some(false);
    ring.transfer_bytes_per_block = Some(8);
    vec![window, ring]
}

#[test]
fn grouped_decode_rank_frees_skipped_blocks_with_the_rest_after_the_next_pass_with_async_scheduling()
 {
    // A decode rank with a full-attention group and a window of 8 tokens, at
    // 4-token blocks. X (4 tokens, 6 outputs) keeps a pass in flight. L, a
    // local prompt of 20 tokens with one output, ends in pass 0 with 5
    // full-attention and 5 window blocks. vLLM frees all of them when it
    // processes the outputs of step 0 (V31 `sched:2171`, V32 `sched:2230`):
    // first the 3 window blocks below 20 - 8 + 1 = 13 tokens in
    // `_connector_finished`, then the rest, each group tail first (V31
    // `sched:2650-2679`, `:2966-2974`, `coord:387-395`, `single:581-589`,
    // V32 `sched:2714-2744`, `:3023-3031`, `coord:400-408`,
    // `single:581-589`). With async scheduling that comes after the
    // scheduling of step 1 (V31 `engine/core.py:697`, `:740-755`, V32
    // `engine/core.py:725`, `:768-783`). So L holds 0 blocks after pass 0
    // without async scheduling, and 10 with it until pass 1 is formed, then 0.
    for (block_mode, version) in WINDOW_PD_MODES {
        for (max_concurrent_batches, after_finish) in [(1, 0), (2, 10)] {
            let label = format!("{block_mode} {version} {max_concurrent_batches}");
            let mut core = window_pd_core(
                block_mode,
                version,
                WorkerType::Decode,
                max_concurrent_batches,
            );
            let mut now = 0.0;
            let x = submit_tokens(&mut core, 1, (1000..1004).collect(), 6);
            let l = submit_tokens(&mut core, 2, (2000..2020).collect(), 1);
            let ended = async_step(&mut core, &mut now);
            assert!(
                ended
                    .output_signals
                    .iter()
                    .any(|signal| signal.uuid == l && signal.completed)
            );
            assert_eq!(blocks_besides(&core, x), after_finish, "{label}");
            async_step(&mut core, &mut now);
            assert_eq!(blocks_besides(&core, x), 0, "{label}");
        }
    }
}

#[test]
fn grouped_decode_rank_with_async_scheduling_evicts_the_skipped_blocks_of_an_ending_request_first()
{
    // A block-mode decode rank of 12 blocks with a full-attention group and a
    // window of 8 tokens, at 4-token blocks and a hash block of 4. X (3
    // tokens, 8 outputs) and L (20 tokens, 1 output) fill the pool in pass 0,
    // and L ends there. vLLM frees L first in `_connector_finished`, the
    // skipped window blocks below 20 - 8 + 1 = 13 tokens, and then the rest,
    // each group tail first (V31 `sched:2650-2679`, `:2966-2974`,
    // `coord:387-395`, `single:581-589`, V32 `sched:2714-2744`,
    // `:3023-3031`, `coord:400-408`, `single:581-589`), with async
    // scheduling after it has scheduled step 1. The freed blocks keep their
    // keys, and the pool evicts them in the order of the free. X's growth and
    // W (8 tokens, from pass 2) then take blocks until three cached blocks of
    // L are evicted: the window block of tokens 8 to 11, and then the
    // full-attention blocks of tokens 16 to 19 and 12 to 15. The
    // full-attention block of tokens 8 to 11 stays. Freed in group order, the
    // full-attention blocks of tokens 8 to 19 would go and the window block
    // would stay.
    let key = |core: &VllmCore, group: u16, end: usize| {
        let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
        lease.configure_prefix_hashes(&(2000..2020).collect::<Vec<u32>>(), 4);
        block_pool(core)
            .key_hit(group, lease.prefix_hash_at(end).unwrap())
            .is_some()
    };
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for max_concurrent_batches in [1, 2] {
            let label = format!("{version} {max_concurrent_batches}");
            let mut full = group("full", 4, 100, None);
            full.prefix_cacheable = Some(true);
            full.transfer_bytes_per_block = Some(100);
            let mut window = group("window", 4, 100, Some(8));
            window.prefix_cacheable = Some(true);
            window.transfer_bytes_per_block = Some(100);
            let mut args = prefix_core(vec![full, window], 12 * 100, 64, 4, version).args;
            args.worker_type = WorkerType::Decode;
            args.max_concurrent_batches = max_concurrent_batches;
            let mut core = VllmCore::new(args);
            let mut now = 0.0;
            submit_tokens(&mut core, 1, (1000..1003).collect(), 8);
            submit_tokens(&mut core, 2, (2000..2020).collect(), 1);
            for pass in 0..6 {
                if pass == 2 {
                    submit_tokens(&mut core, 3, (3000..3008).collect(), 1);
                }
                async_step(&mut core, &mut now);
            }
            assert_eq!(
                [
                    key(&core, 1, 12),
                    key(&core, 0, 12),
                    key(&core, 0, 16),
                    key(&core, 0, 20)
                ],
                [false, true, false, false],
                "{label}"
            );
        }
    }
}

#[test]
fn grouped_decode_rank_drop_before_the_first_allocation_takes_no_block() {
    // The shape of `grouped_prefix_waiting_hit_waits_when_its_first_chunk_does_not_fit`
    // on a decode rank of 12 pages, with a ring that every request holds.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut args = prefix_core(window_and_ring_transfer_groups(), 12 * 8, 8, 4, version).args;
        args.worker_type = WorkerType::Decode;
        let mut core = VllmCore::new(args);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // Two finished prompts of 100 tokens each keep the two window blocks
        // before their replay boundary 96 cached: 4 blocks.
        submit_tokens(&mut core, 1861, (0..100).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        submit_tokens(&mut core, 1862, (5_000..5_100).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 4, "{version}");
        // After 3 passes, two prompts of 8 tokens hold 3 and 2 window blocks,
        // each with its ring block.
        submit_tokens(&mut core, 1863, (1_000..1_008).collect(), 30);
        submit_tokens(&mut core, 1864, (2_000..2_008).collect(), 30);
        for _ in 0..3 {
            core.execute_pass(&mut trace, step as f64);
            step += 1;
        }
        assert_eq!(block_pool(&core).active_blocks(), 7, "{version}");
        // A prompt of 104 tokens hits the first prompt at 96. Under the window
        // cap, its whole prompt counts the 2 inactive hit blocks and the ring.
        let waiting = submit_tokens(&mut core, 1865, (0..104).collect(), 1);
        let request = &core.state.requests[&waiting].sequence;
        let cost = core
            .kv_manager
            .get_native_prefill_cost(&request.sequence, &request.lease);
        assert_eq!((cost.new_blocks, cost.cached_tokens), (3, 96), "{version}");
        // The decode steps leave 4 blocks, all cached. The whole prompt fits,
        // and the first chunk of 6 tokens needs 5. Its first allocation fails
        // and leaves an unallocated block lease.
        let pass = core.execute_pass(&mut trace, step as f64);
        assert!(pass.admissions.is_empty(), "{version}");
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{version}");
        assert_eq!(
            core.state.requests[&waiting].num_computed_tokens, 0,
            "{version}"
        );
        assert_eq!(block_pool(&core).active_blocks(), 8, "{version}");
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 4, "{version}");
        // The drop takes no ring block, so it evicts no cached block.
        core.apply_command(SchedulerCommand::CancelRequest {
            request_id: waiting,
        })
        .unwrap();
        assert_eq!(block_pool(&core).active_blocks(), 8, "{version}");
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 4, "{version}");
    }
}

#[test]
fn grouped_decode_rank_ending_request_takes_no_copy_of_its_published_state() {
    // Attention and state blocks of 32 at a hash block of 4, so hits inside a
    // block are on. A request whose state is published inside its running
    // block takes the copy only at its next allocation (V31
    // single:1892-1936, V32 :1856-1900). A request that ends allocates no
    // more: vLLM frees its skipped blocks and then the request, which drops
    // the producer (V31 sched:2650-2658, :2966-2974, single:562-579,
    // :1971-1977, V32 sched:2714-2723, :3023-3031, single:562-579,
    // :1935-1941). So the release of an ending request takes no block, also
    // in a pool with no free block.
    let core_of = |version: &str, pages: u64| {
        let mut full = group("full", 32, 100, None);
        full.prefix_cacheable = Some(true);
        full.transfer_bytes_per_block = Some(100);
        let mut recurrent = state("state", 100, 1);
        recurrent.block_size_tokens = 32;
        recurrent.prefix_cacheable = Some(true);
        recurrent.transfer_bytes_per_block = Some(100);
        recurrent.prefill_checkpoint_blocks = Some(0);
        let mut args = prefix_core(vec![full, recurrent], pages * 100, 8192, 4, version).args;
        args.worker_type = WorkerType::Decode;
        VllmCore::new(args)
    };
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        // Two distinct loads of 49 tokens fill 6 pages: each holds 2
        // attention blocks and its running state block. Each activation
        // publishes the loaded state at 48, inside the running block. A
        // cancel of one load before its first pass frees its 3 blocks.
        let mut core = core_of(version, 6);
        for (id, base) in [(1880u128, 10_000u32), (1882, 20_000)] {
            let handoff_id = HandoffId::new(Uuid::from_u128(id + 1));
            let request = DirectRequest {
                tokens: (base..base + 49).collect(),
                ..pd_request(id, 0, 2)
            };
            assert_eq!(
                reserve_grouped(&mut core, handoff_id, request),
                Some(2),
                "{version}"
            );
        }
        for id in [1881u128, 1883] {
            activate(&mut core, HandoffId::new(Uuid::from_u128(id)));
        }
        assert_eq!(block_pool(&core).active_blocks(), 6, "{version}");
        core.apply_command(SchedulerCommand::CancelRequest {
            request_id: Uuid::from_u128(1880),
        })
        .unwrap();
        assert_eq!(block_pool(&core).active_blocks(), 3, "{version}");
        // A local prompt of 28 tokens with one output token ends in its
        // prefill pass. On vLLM 0.31.0 its partial tail is 28, the prompt end,
        // inside its running state block, where the state is published. Its
        // 2 blocks fill 2 pages. On vLLM 0.32.0 the tail lies before the
        // prompt end (V32 single:2069-2073), so its last chunk takes the copy.
        if version == "0.31.0" {
            let mut core = core_of(version, 2);
            let mut trace = TraceCollector::default();
            let mut step = 0;
            submit(&mut core, 1884, 28, 1);
            drain(&mut core, &mut trace, &mut step);
            assert_eq!(block_pool(&core).active_blocks(), 0, "{version}");
        }
    }
}

#[test]
fn grouped_aggregated_rank_builds_no_transfer_rules() {
    // Groups that carry every transfer field, in byte and in block mode.
    let byte_mode = core(glm_transfer_groups("0.32.0", false), 128 * GLM_PAGE, 8192);
    assert!(byte_mode.kv_manager.grouped_transfer().is_none());
    let block_mode = prefix_core(
        glm_transfer_groups("0.32.0", true),
        128 * GLM_PAGE,
        8192,
        2176,
        "0.32.0",
    );
    assert!(block_mode.kv_manager.grouped_transfer().is_none());
}

#[test]
fn grouped_state_blocks_stay_constant_through_prefill_and_decode() {
    let mut core = core(vec![group("full", 4, 8, None), state("kda", 10, 2)], 200, 4);
    let id = submit(&mut core, 109, 16, 5);
    let mut trace = TraceCollector::default();
    let mut bytes = Vec::new();
    for step in 0..5 {
        core.execute_pass(&mut trace, step as f64);
        bytes.push(core.mocker_metrics().kv_cache_used_bytes.unwrap());
    }
    // Every pass adds one 8-byte page; the two 10-byte state blocks stay.
    assert_eq!(bytes, vec![28, 36, 44, 52, 60]);
    assert!(core.kv_manager.grouped().unwrap().block_pool().is_none());
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_state_bytes_stay_constant_while_window_pages_expire() {
    let run = |groups: Vec<FpmCacheGroup>, capacity: u64| {
        let mut core = core(groups, capacity, 8);
        let id = submit(&mut core, 110, 32, 5);
        let mut trace = TraceCollector::default();
        let mut outputs = 0;
        let mut bytes = Vec::new();
        for step in 0..20 {
            let pass = core.execute_pass(&mut trace, step as f64);
            bytes.push(pass.mocker_metrics.kv_cache_used_bytes.unwrap());
            assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
            for output in pass.output_signals {
                assert_eq!(output.uuid, id);
                assert!(!output.rejected);
                outputs += usize::from(output.token_id.is_some());
            }
            if core.is_drained() {
                break;
            }
        }
        assert!(core.is_drained());
        assert_eq!(outputs, 5);
        assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        bytes
    };
    let without = run(vec![group("window", 4, 8, Some(4))], 24);
    let with = run(
        vec![group("window", 4, 8, Some(4)), state("kda", 10, 2)],
        44,
    );
    // Window pages expire as the decode moves on (24 to 16 bytes), while the
    // two state blocks add a constant 20 bytes. The last pass completes the
    // request, which frees every group before the pass reports its bytes.
    assert_eq!(without, vec![16, 24, 24, 24, 16, 16, 16, 0]);
    assert_eq!(with, vec![36, 44, 44, 44, 36, 36, 36, 0]);
}

#[test]
fn grouped_state_preemption_frees_and_rereserves_state_blocks() {
    let mut core = core(
        vec![
            group("full", 4, 4, None),
            group("window", 4, 4, Some(4)),
            state("kda", 4, 1),
        ],
        36,
        8,
    );
    let first = submit(&mut core, 111, 4, 6);
    let second = submit(&mut core, 112, 4, 6);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    // Each prefill holds a full page, a window page and its state block.
    assert_eq!(initial.admissions.len(), 2);
    assert_eq!(initial.mocker_metrics.kv_cache_used_bytes, Some(24));
    let pressure = core.execute_pass(&mut trace, 1.0);
    // The first request grows to 20 bytes. The preempted second request
    // returns all 12 of its bytes, its state block included.
    assert_eq!(pressure.mocker_metrics.vllm_preemptions_total, 1);
    assert_eq!(pressure.mocker_metrics.kv_cache_used_bytes, Some(20));
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    assert_eq!(
        core.state.requests[&second].sequence.num_allocated_tokens(),
        0
    );
    let mut completed = Vec::new();
    let mut resumed = Vec::new();
    for step in 2..40 {
        let pass = core.execute_pass(&mut trace, step as f64);
        let bytes = pass.mocker_metrics.kv_cache_used_bytes.unwrap();
        assert!(bytes <= 36);
        if pass
            .admissions
            .iter()
            .any(|admission| admission.uuid == second)
        {
            resumed.push(bytes);
        }
        for output in pass.output_signals {
            assert!(!output.rejected);
            if output.completed {
                completed.push(output.uuid);
            }
        }
        if core.is_drained() {
            break;
        }
    }
    // Each resume reserves the state block once more: two full pages, two
    // window pages and one state block, 20 bytes. The first resume fills the
    // pool next to the first request's 16 bytes. The first request's next
    // pages then preempt the second request again, and it resumes alone.
    assert_eq!(resumed, vec![36, 20]);
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
    assert_eq!(completed, vec![first, second]);
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_state_request_waits_when_only_attention_pages_fit() {
    let full = group("full", 4, 4, None);
    let mut core = core(vec![full.clone(), state("kda", 8, 1)], 28, 8);
    let first = submit(&mut core, 113, 8, 2);
    let second = submit(&mut core, 114, 8, 2);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    assert_eq!(initial.admissions.len(), 1);
    assert_eq!(initial.admissions[0].uuid, first);
    assert_eq!(initial.mocker_metrics.kv_cache_used_bytes, Some(16));
    // The next pass first grows the running request to 9 tokens: three pages
    // and its state block hold 12 + 8 = 20 bytes, so 8 of 28 are free. The
    // waiting request's chunk, the 7 tokens left in the 8-token batch, would
    // fit its two pages (8 bytes) there, but not its 8-byte state block too.
    let pool = core.kv_manager.grouped().unwrap();
    let free = pool.capacity_bytes() - pool.required_bytes(8, 9).unwrap();
    assert_eq!(free, 8);
    assert!(full.bytes_for_forward(0, 7).unwrap() <= free);
    assert!(pool.required_bytes(0, 7).unwrap() > free);
    let next = core.execute_pass(&mut trace, 1.0);
    assert!(next.admissions.is_empty());
    assert_eq!(next.mocker_metrics.vllm_preemptions_total, 0);
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    assert!(
        next.output_signals
            .iter()
            .any(|output| output.uuid == first && output.completed)
    );
    let admitted = core.execute_pass(&mut trace, 2.0);
    assert_eq!(admitted.admissions[0].uuid, second);
    for pass in [&initial, &next, &admitted] {
        assert!(pass.output_signals.iter().all(|output| !output.rejected));
    }
}

#[test]
fn grouped_prefix_caching_off_keeps_byte_mode_accounting() {
    let mut core = core(glm_groups(false), 64 * GLM_PAGE, 8192);
    assert!(core.kv_manager.grouped().unwrap().block_pool().is_none());
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let first = submit(&mut core, 301, 9000, 4);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(first, 0)]);
    // A resend of the same prompt replays cold: nothing is cached in byte mode.
    let second = submit(&mut core, 302, 9000, 4);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(second, 0)]);
    let metrics = core.mocker_metrics();
    assert_eq!(metrics.kv_cache_used_bytes, Some(0));
    assert_eq!(metrics.kv_cache_cached_bytes, None);
    assert_eq!(core.kv_manager.grouped().unwrap().cached_bytes(), 0);
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_glm_shaped_multi_turn_replay_reports_hit_tokens_and_block_counts() {
    for (version, checkpoint) in GLM_CORES {
        let mut core = prefix_core(glm_groups(checkpoint), 64 * GLM_PAGE, 8192, 2176, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let turn_one: Vec<u32> = (0..9000).collect();
        let first = submit_tokens(&mut core, 401, turn_one.clone(), 4);
        assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(first, 0)]);
        assert_eq!(block_pool(&core).active_blocks(), 0);

        // Turn two: the first 9,000 tokens again plus 4,000 new ones. The hit
        // is the state at request one's replay boundary (9000 - 1) / 2176 *
        // 2176 = 8704, where G0 holds four cached blocks too.
        let mut turn_two = turn_one.clone();
        turn_two.extend(100_000..104_000);
        let second = submit_tokens(&mut core, 402, turn_two.clone(), 4);
        block_pool(&core).reset_peak_active_blocks();
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.admissions.len(), 1, "{version} {checkpoint}");
        assert_eq!(pass.admissions[0].uuid, second);
        assert_eq!(
            pass.admissions[0].reused_input_tokens, 8704,
            "{version} {checkpoint}"
        );
        // During the first step: G0 holds the four pinned blocks and the fresh
        // ones of the chunk, the ring one, each KDA group the cached and the
        // running block, plus the checkpoint block on the cores with checkpoint
        // blocks. The cached state blocks stay until the request's next
        // allocation, as in vLLM (v0.31 kv_cache_manager.py:552-556, v0.32
        // :555-559, v0.33 :564-568).
        let (peak, after) = if checkpoint { (19, 19) } else { (14, 14) };
        assert_eq!(
            block_pool(&core).peak_active_blocks(),
            peak,
            "{version} {checkpoint}"
        );
        assert_eq!(
            block_pool(&core).active_blocks(),
            after,
            "{version} {checkpoint}"
        );
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 0);
        assert_eq!(
            pass.mocker_metrics.kv_cache_used_bytes,
            Some(after as u64 * GLM_PAGE)
        );
        assert_eq!(pass.mocker_metrics.kv_cache_cached_bytes, Some(0));
        // The next pass releases the four cached state blocks, and on the
        // third core the checkpoint block, before it takes the blocks of the
        // next chunk.
        core.execute_pass(&mut trace, step as f64);
        step += 1;
        let (active, cached) = if checkpoint { (11, 8) } else { (15, 4) };
        assert_eq!(
            block_pool(&core).active_blocks(),
            active,
            "{version} {checkpoint}"
        );
        assert_eq!(
            block_pool(&core).inactive_cached_blocks(),
            cached,
            "{version} {checkpoint}"
        );
        assert!(drain(&mut core, &mut trace, &mut step).is_empty());

        // A request that shares only the first 2,176 tokens finds a G0 block
        // but no state key at 2,176 (retention 0), so it hits nothing.
        let mut other: Vec<u32> = (0..2176).collect();
        other.extend(200_000..202_824);
        let third = submit_tokens(&mut core, 403, other, 4);
        assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(third, 0)]);

        // The whole second turn plus 100 tokens hits request two's replay
        // boundary (13000 - 1) / 2176 * 2176 = 10880.
        let mut turn_three = turn_two;
        turn_three.extend(300_000..300_100);
        let fourth = submit_tokens(&mut core, 404, turn_three, 4);
        assert_eq!(
            drain(&mut core, &mut trace, &mut step),
            vec![(fourth, 10880)]
        );
        assert_eq!(block_pool(&core).active_blocks(), 0);
        assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    }
}

#[test]
fn grouped_prefix_hit_lowers_prefill_tokens_and_raises_prefix_tokens() {
    for (version, checkpoint) in GLM_CORES {
        let mut core = prefix_core(glm_groups(checkpoint), 64 * GLM_PAGE, 8192, 2176, version);
        let captured = Arc::new(CapturedDecodeQueries::default());
        core.args.perf_model = Arc::new(PerfModel::External {
            timing: captured.clone(),
        });
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let turn_one: Vec<u32> = (0..9000).collect();
        submit_tokens(&mut core, 501, turn_one.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        // The first prefill stops at the block boundary 6528 inside the 8192
        // budget, then at the last cacheable stop 8704, then at the prompt.
        // In checkpoint mode the final chunk runs from 6528 to the prompt.
        let first_turn: Vec<(usize, usize, usize)> = if checkpoint {
            vec![(1, 6528, 0), (1, 2472, 6528)]
        } else {
            vec![(1, 6528, 0), (1, 2176, 6528), (1, 296, 8704)]
        };
        assert_eq!(
            *captured.1.lock().unwrap(),
            first_turn,
            "{version} {checkpoint}"
        );
        captured.1.lock().unwrap().clear();

        let mut turn_two = turn_one;
        turn_two.extend(100_000..104_000);
        submit_tokens(&mut core, 502, turn_two, 1);
        drain(&mut core, &mut trace, &mut step);
        // From the hit at 8704 the split ends the first chunk at the last
        // cacheable stop 10880 and the second at the prompt. The checkpoint
        // branch runs to the prompt end in one chunk.
        let second_turn: Vec<(usize, usize, usize)> = if checkpoint {
            vec![(1, 4296, 8704)]
        } else {
            vec![(1, 2176, 8704), (1, 2120, 10880)]
        };
        assert_eq!(
            *captured.1.lock().unwrap(),
            second_turn,
            "{version} {checkpoint}"
        );
    }
}

#[test]
fn grouped_prefix_admission_tests_the_whole_prompt() {
    for (version, checkpoint) in GLM_CORES {
        // A prompt of 40,000 tokens needs 19 G0 blocks, 1 ring block and one
        // running block per KDA group, plus one checkpoint block each on the
        // cores with checkpoint blocks: 24 or 28, above the 20-block pool.
        let mut core = prefix_core(glm_groups(checkpoint), 20 * GLM_PAGE, 8192, 2176, version);
        let id = submit(&mut core, 601, 40_000, 2);
        let request = &core.state.requests[&id].sequence;
        let cost = core
            .kv_manager
            .get_native_prefill_cost(&request.sequence, &request.lease);
        assert_eq!(
            cost.new_blocks,
            if checkpoint { 28 } else { 24 },
            "{version}"
        );
        assert_eq!(cost.cached_tokens, 0);
        let mut trace = TraceCollector::default();
        let pass = core.execute_pass(&mut trace, 0.0);
        assert!(pass.admissions.is_empty(), "{version} {checkpoint}");
        assert!(
            pass.output_signals
                .iter()
                .any(|output| output.uuid == id && output.rejected)
        );
        assert_eq!(block_pool(&core).active_blocks(), 0);
        assert!(core.is_drained());

        // Under contention the whole prompt decides, as in vLLM's default
        // admission (mgr:519-536). A running request of 13,000 tokens holds
        // 3 G0 pages, the ring and four state blocks after its first chunk,
        // which the split ends at 6,528, so 22 of 30 blocks are free. The
        // 40,000-token prompt needs 24 and waits without preempting anything.
        let mut core = prefix_core(glm_groups(checkpoint), 30 * GLM_PAGE, 8192, 2176, version);
        let running = submit(&mut core, 603, 13_000, 2);
        let waiting = submit_tokens(&mut core, 604, (100_000..140_000).collect(), 2);
        let pass = core.execute_pass(&mut trace, 0.0);
        assert_eq!(pass.admissions.len(), 1, "{version} {checkpoint}");
        assert_eq!(pass.admissions[0].uuid, running);
        assert_eq!(block_pool(&core).active_blocks(), 8);
        // The second pass has budget left after the running request's
        // chunk, and the waiting request still waits for the whole prompt.
        let pass = core.execute_pass(&mut trace, 1.0);
        assert!(pass.admissions.is_empty(), "{version} {checkpoint}");
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
        assert_eq!(core.state.requests[&waiting].num_computed_tokens, 0);
    }
    // Prefix caching off keeps the first-chunk test of byte mode. The first
    // pass spends its budget on the running request. In the second pass the
    // running request takes its last 4,808 tokens (6 G0 pages, the ring and
    // the states: 11 pages) and the waiting request's first chunk of the
    // 3,384 tokens left (2 pages, the ring and the states: 7 pages) fits.
    let mut off = core(glm_groups(false), 30 * GLM_PAGE, 8192);
    let running = submit(&mut off, 605, 13_000, 2);
    let waiting = submit_tokens(&mut off, 606, (100_000..140_000).collect(), 2);
    let mut trace = TraceCollector::default();
    let pass = off.execute_pass(&mut trace, 0.0);
    assert_eq!(pass.admissions.len(), 1);
    assert_eq!(pass.admissions[0].uuid, running);
    let pass = off.execute_pass(&mut trace, 1.0);
    assert_eq!(pass.admissions.len(), 1);
    assert_eq!(pass.admissions[0].uuid, waiting);
    assert_eq!(pass.mocker_metrics.kv_cache_used_bytes, Some(18 * GLM_PAGE));
    // Byte mode also refuses a final forward that can never fit, as
    // grouped_state_blocks_exceeding_the_pool_reject_at_admission shows: the
    // 40,000-token prompt's last forward needs 24 pages of a 20-page pool.
    let mut off = core(glm_groups(false), 20 * GLM_PAGE, 8192);
    let id = submit(&mut off, 602, 40_000, 2);
    let pass = off.execute_pass(&mut trace, 0.0);
    assert!(pass.admissions.is_empty());
    assert!(
        pass.output_signals
            .iter()
            .any(|output| output.uuid == id && output.rejected)
    );
}

#[test]
fn grouped_prefix_waiting_request_keeps_fifo() {
    // Each 20,000-token prompt needs 10 G0 blocks, the ring block and four
    // running blocks: 15 of 20. The second request waits for the first one
    // to finish and preempts nothing.
    let mut core = prefix_core(glm_groups(false), 20 * GLM_PAGE, 8192, 2176, "0.31.0");
    let first = submit_tokens(&mut core, 701, (0..20_000).collect(), 2);
    let second = submit_tokens(&mut core, 702, (100_000..120_000).collect(), 2);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let initial = core.execute_pass(&mut trace, step as f64);
    step += 1;
    assert_eq!(initial.admissions.len(), 1);
    assert_eq!(initial.admissions[0].uuid, first);
    let mut saw_first_complete = false;
    let mut admitted_second_at = None;
    for _ in 0..40 {
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
        if pass
            .admissions
            .iter()
            .any(|admission| admission.uuid == second)
        {
            admitted_second_at = Some(step);
            assert!(
                saw_first_complete,
                "the second request waited for the first"
            );
        }
        for output in pass.output_signals {
            assert!(!output.rejected);
            saw_first_complete |= output.uuid == first && output.completed;
        }
        if core.is_drained() {
            break;
        }
    }
    assert!(admitted_second_at.is_some());
    assert!(core.is_drained());
    assert_eq!(block_pool(&core).active_blocks(), 0);
}

#[test]
fn grouped_prefix_waiting_hit_waits_when_its_first_chunk_does_not_fit() {
    // One sliding-window group of 4-token blocks with an 8-token window, a
    // 9-block pool and a step budget of 8 tokens.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut window = group("window", 4, 8, Some(8));
        window.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![window], 9 * 8, 8, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // A finished request of 100 tokens keeps the two blocks before its
        // replay boundary 96 cached.
        submit_tokens(&mut core, 1101, (0..100).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 2, "{version}");
        submit_tokens(&mut core, 1102, (1000..1008).collect(), 30);
        submit_tokens(&mut core, 1103, (2000..2008).collect(), 30);
        for _ in 0..3 {
            core.execute_pass(&mut trace, step as f64);
            step += 1;
        }
        assert_eq!(block_pool(&core).active_blocks(), 5, "{version}");
        // A request of 104 tokens hits those blocks at 96. The whole-prompt
        // test counts only the two inactive hit blocks, because the window
        // caps it (mgr:519-536). After their decode steps the running
        // requests hold 6 of the 9 blocks, which leaves 3, but the first
        // chunk of 6 tokens needs the two hit blocks and two fresh blocks
        // (mgr:558-576). vLLM then stops its waiting loop (v0.31
        // sched:1223-1230, v0.32 sched:1275-1282), so the request waits and
        // preempts nothing.
        let waiting = submit_tokens(&mut core, 1104, (0..104).collect(), 1);
        let request = &core.state.requests[&waiting].sequence;
        let cost = core
            .kv_manager
            .get_native_prefill_cost(&request.sequence, &request.lease);
        assert_eq!((cost.new_blocks, cost.cached_tokens), (2, 96), "{version}");
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{version}");
        assert!(pass.admissions.is_empty(), "{version}");
        assert_eq!(
            core.state.requests[&waiting].num_computed_tokens, 0,
            "{version}"
        );
        assert_eq!(core.state.running.len(), 2, "{version}");
        // The waiting request starts with its hit once the decode steps of the
        // running requests leave room.
        let admissions = drain(&mut core, &mut trace, &mut step);
        assert_eq!(admissions.first(), Some(&(waiting, 96)), "{version}");
    }
}

#[test]
fn grouped_prefix_waiting_hit_keeps_its_lru_place_when_its_first_chunk_does_not_fit() {
    // vLLM tests the blocks of the first chunk before it touches the hit
    // (mgr:558-589). So a waiting request that fails that test leaves its hit
    // blocks where they are in the LRU, ahead of a block cached after them.
    // Pinning and then releasing them would move them behind that block.
    let cached = |core: &VllmCore, tokens: &[u32], end: usize| {
        let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
        lease.configure_prefix_hashes(tokens, 4);
        block_pool(core)
            .key_hit(0, lease.prefix_hash_at(end).unwrap())
            .is_some()
    };
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut window = group("window", 4, 8, Some(8));
        window.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![window], 9 * 8, 8, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // The LRU holds the blocks that end at 96 and 92 of a finished request
        // of 100 tokens, then the block that ends at 4 of a finished request
        // of 8 tokens.
        let first: Vec<u32> = (0..100).collect();
        let second: Vec<u32> = (3000..3008).collect();
        submit_tokens(&mut core, 1401, first.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        submit_tokens(&mut core, 1402, second.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 3, "{version}");
        // As in grouped_prefix_waiting_hit_waits_when_its_first_chunk_does_not_fit,
        // two running requests hold 6 of the 9 blocks, and the request of 104
        // tokens fails the first-chunk test of its hit at 96.
        submit_tokens(&mut core, 1403, (1000..1008).collect(), 30);
        submit_tokens(&mut core, 1404, (2000..2008).collect(), 30);
        for _ in 0..3 {
            core.execute_pass(&mut trace, step as f64);
            step += 1;
        }
        let waiting = submit_tokens(&mut core, 1405, (0..104).collect(), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert!(pass.admissions.is_empty(), "{version}");
        assert_eq!(block_pool(&core).active_blocks(), 6, "{version}");
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 3, "{version}");
        core.apply_command(SchedulerCommand::CancelRequest {
            request_id: waiting,
        })
        .unwrap();
        // No block is free. A request of 4 new tokens takes one fresh block,
        // which evicts the oldest cached block: the one that ends at 96.
        let fresh = submit_tokens(&mut core, 1406, (4000..4004).collect(), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        assert_eq!(pass.admissions.len(), 1, "{version}");
        assert_eq!(pass.admissions[0].uuid, fresh, "{version}");
        assert!(!cached(&core, &first, 96), "{version}");
        assert!(cached(&core, &first, 92), "{version}");
        assert!(cached(&core, &second, 4), "{version}");
    }
}

#[test]
fn grouped_prefix_without_chunked_prefill_budgets_the_uncached_tokens() {
    // With chunked prefill off, vLLM compares the tokens after the prefix hit
    // with the step budget (v0.31 sched:1084, :1115-1121, v0.32 sched:1129,
    // :1160-1167), as the linear pool does. Two running requests decode one
    // token each and leave 14 of the 16 tokens. A prompt of 15 tokens with 12
    // cached tokens needs 3 of them.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let args = || {
            MockEngineArgs::builder()
                .block_size(4)
                .max_num_batched_tokens(Some(16))
                .max_num_seqs(Some(4))
                .max_model_len(Some(4096))
                .enable_prefix_caching(true)
                .enable_chunked_prefill(false)
                .speedup_ratio(0.0)
        };
        let linear = VllmCore::new(args().num_gpu_blocks(64).build().unwrap());
        let mut full = group("full", 4, 1, None);
        full.prefix_cacheable = Some(true);
        let grouped = VllmCore::new(
            args()
                .num_gpu_blocks(1)
                .prefix_match_unit(Some(4))
                .vllm_version(Some(version.to_string()))
                .kv_cache_groups(vec![full])
                .kv_cache_capacity_bytes(Some(64))
                .build()
                .unwrap(),
        );
        for (pool, mut core) in [("linear", linear), ("grouped", grouped)] {
            let mut trace = TraceCollector::default();
            let mut step = 0;
            submit_tokens(&mut core, 1301, (0..15).collect(), 1);
            drain(&mut core, &mut trace, &mut step);
            submit_tokens(&mut core, 1302, (1000..1004).collect(), 10);
            submit_tokens(&mut core, 1303, (2000..2004).collect(), 10);
            let pass = core.execute_pass(&mut trace, step as f64);
            step += 1;
            assert_eq!(pass.admissions.len(), 2, "{version} {pool}");
            let waiting = submit_tokens(&mut core, 1304, (0..15).collect(), 1);
            let pass = core.execute_pass(&mut trace, step as f64);
            let admitted: Vec<(Uuid, usize)> = pass
                .admissions
                .iter()
                .map(|admission| (admission.uuid, admission.reused_input_tokens))
                .collect();
            assert_eq!(admitted, vec![(waiting, 12)], "{version} {pool}");
        }
    }
}

/// Full-attention and state groups of 16-token blocks, a 4-token hash unit
/// and a pool of five blocks: the blocks of one 34-token request, as vLLM's
/// startup check counts them (3 + 2).
fn fine_grained_core(version: &str) -> VllmCore {
    let mut full = group("full", 16, 8, None);
    full.prefix_cacheable = Some(true);
    let mut state = state("state", 8, 1);
    state.block_size_tokens = 16;
    state.prefix_cacheable = Some(true);
    state.prefill_checkpoint_blocks = (version != "0.31.0").then_some(0);
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .max_model_len(Some(34))
            .enable_prefix_caching(true)
            .prefix_match_unit(Some(4))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(vec![full, state])
            .kv_cache_capacity_bytes(Some(5 * 8))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
}

#[test]
fn grouped_prefix_hit_that_needs_more_blocks_than_the_pool_waits() {
    // A finished 13-token request leaves its full-attention block and its
    // state keyed at the hash boundary 12. A 33-token prompt that repeats it
    // hits 12 tokens inside a block. That hit needs 7 blocks: 2 new, the
    // inactive hit block and a copy for full attention (single:216-268), and
    // the running block, a copy and the inactive hit block for the state
    // (single:1716-1816). Cold the prompt needs 4 of the 5 blocks. vLLM keeps
    // such a request waiting and searches its hit again at the next step
    // (v0.31 sched:930-937, :1223-1230, v0.32 sched:960-967, :1275-1282).
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        // With nothing running, nothing changes the cache, so the request
        // waits in every pass. The replay reports that as a livelock
        // (runtime_contract.rs grouped_prefix_hit_above_the_pool_is_a_livelock).
        let mut core = fine_grained_core(version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1501, (0..13).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 2, "{version}");
        let waiting = submit_tokens(&mut core, 1502, (0..33).collect(), 1);
        for _ in 0..3 {
            let pass = core.execute_pass(&mut trace, step as f64);
            step += 1;
            assert!(pass.admissions.is_empty(), "{version}");
            assert!(pass.output_signals.is_empty(), "{version}");
            assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{version}");
        }
        assert!(core.state.requests.contains_key(&waiting), "{version}");
        // Read after a pass: the last step of the finished request copied its
        // state at 12, which hides that key until the next step.
        let request = &core.state.requests[&waiting].sequence;
        let cost = core
            .kv_manager
            .get_native_prefill_cost(&request.sequence, &request.lease);
        assert_eq!((cost.new_blocks, cost.cached_tokens), (7, 12), "{version}");

        // A running request can evict the hit blocks. Its decode crosses a
        // block boundary at 16 tokens and takes two blocks while one is free,
        // which evicts the cached full-attention block of the hit. The
        // waiting request then hits nothing and starts cold once the running
        // request finishes.
        let mut core = fine_grained_core(version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1503, (0..13).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        let running = submit_tokens(&mut core, 1504, (1000..1004).collect(), 20);
        let waiting = submit_tokens(&mut core, 1505, (0..33).collect(), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.admissions.len(), 1, "{version}");
        assert_eq!(pass.admissions[0].uuid, running, "{version}");
        assert!(pass.output_signals.iter().all(|output| !output.rejected));
        let mut finished = Vec::new();
        let mut admissions = Vec::new();
        for _ in 0..40 {
            let pass = core.execute_pass(&mut trace, step as f64);
            step += 1;
            assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{version}");
            admissions.extend(
                pass.admissions
                    .iter()
                    .map(|admission| (admission.uuid, admission.reused_input_tokens)),
            );
            for output in pass.output_signals {
                assert!(!output.rejected, "{version}");
                if output.completed {
                    finished.push(output.uuid);
                }
            }
            if core.is_drained() {
                break;
            }
        }
        assert_eq!(admissions, vec![(waiting, 0)], "{version}");
        assert_eq!(finished, vec![running, waiting], "{version}");
    }
}

#[test]
fn grouped_prefix_physical_usage_counts_cached_blocks() {
    // Metrics::physical_cache_usage includes the inactive reusable blocks and
    // Metrics::inactive_blocks counts them (protocol.rs). Active usage counts
    // only the blocks that requests hold.
    for (version, checkpoint) in GLM_CORES {
        let mut core = prefix_core(glm_groups(checkpoint), 64 * GLM_PAGE, 8192, 2176, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1201, (0..9000).collect(), 4);
        drain(&mut core, &mut trace, &mut step);
        // G0 keeps its four full blocks, and each KDA group the state at the
        // replay boundary 8704.
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 8, "{version}");
        let metrics = core.mocker_metrics();
        assert_eq!(metrics.kv_cache_used_bytes, Some(0), "{version}");
        assert_eq!(
            metrics.kv_cache_cached_bytes,
            Some(8 * GLM_PAGE),
            "{version}"
        );
        assert_eq!(metrics.gpu_cache_usage_perc, 0.0, "{version}");
        assert_eq!(
            metrics.physical_gpu_cache_usage_perc,
            8.0 / 64.0,
            "{version}"
        );
        assert_eq!(metrics.active_decode_blocks, 0, "{version}");
        assert_eq!(metrics.inactive_decode_blocks, 8, "{version}");
    }
}

/// The tight-pool case: RC hits 3,264 tokens, inside a block.
fn tight_pool_case(
    version: &str,
    blocks: u64,
    max_concurrent_batches: usize,
) -> (usize, usize, Vec<usize>, usize) {
    pool_case(version, blocks, max_concurrent_batches, 3500, 4352, 3264)
}

/// A GLM rank of `blocks` pool blocks with a hash block of 1,088 tokens. RP
/// (`producer` tokens) runs alone and finishes. RC (RP's tokens and 300
/// more) and RN (1,000 new tokens) then arrive together, RC first, and RC
/// hits `hit` tokens. Returns the passes that admit RC and RN, counted from
/// their arrival, the active blocks after each of those passes and the peak
/// of active blocks in RC's pass.
fn pool_case(
    version: &str,
    blocks: u64,
    max_concurrent_batches: usize,
    producer: u32,
    max_model_len: usize,
    hit: usize,
) -> (usize, usize, Vec<usize>, usize) {
    pool_run(
        VllmCore::new(pool_args(
            version,
            blocks,
            max_concurrent_batches,
            max_model_len,
        )),
        producer,
        hit,
    )
}

/// The engine arguments of the rank of [`pool_case`].
fn pool_args(
    version: &str,
    blocks: u64,
    max_concurrent_batches: usize,
    max_model_len: usize,
) -> MockEngineArgs {
    MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(1)
        .max_num_batched_tokens(Some(8192))
        .max_num_seqs(Some(4))
        .max_model_len(Some(max_model_len))
        .enable_prefix_caching(true)
        .prefix_match_unit(Some(1088))
        .vllm_version(Some(version.to_string()))
        .kv_cache_groups(glm_groups(false))
        .kv_cache_capacity_bytes(Some(blocks * GLM_PAGE))
        .speedup_ratio(0.0)
        .max_concurrent_batches(max_concurrent_batches)
        .build()
        .unwrap()
}

/// [`pool_case`] on the rank of `core`.
fn pool_run(mut core: VllmCore, producer: u32, hit: usize) -> (usize, usize, Vec<usize>, usize) {
    let mut now = 0.0;
    let rp: Vec<u32> = (44_000_000..44_000_000 + producer).collect();
    submit_tokens(&mut core, 1, rp.clone(), 4);
    drain_async(&mut core, &mut now, "pool producer");
    let mut rc_tokens = rp;
    rc_tokens.extend(44_100_000..44_100_300);
    let rc = submit_tokens(&mut core, 2, rc_tokens, 4);
    let rn = submit_tokens(&mut core, 3, (44_200_000..44_201_000).collect(), 4);
    let (mut rc_pass, mut rn_pass, mut active, mut peak) = (None, None, Vec::new(), 0);
    for pass in 0..50 {
        block_pool(&core).reset_peak_active_blocks();
        let result = async_step(&mut core, &mut now);
        assert_eq!(result.mocker_metrics.vllm_preemptions_total, 0);
        for admission in &result.admissions {
            if admission.uuid == rc {
                rc_pass = Some(pass);
                peak = block_pool(&core).peak_active_blocks();
                assert_eq!(admission.reused_input_tokens, hit);
            }
            if admission.uuid == rn {
                rn_pass = Some(pass);
            }
        }
        active.push(block_pool(&core).active_blocks());
        if core.is_drained() {
            break;
        }
    }
    (rc_pass.unwrap(), rn_pass.unwrap(), active, peak)
}

#[test]
fn grouped_prefix_copy_sources_block_a_later_admission_of_the_pass() {
    // RC's hit inside a block copies the cached full-attention block and the
    // hit block of each of the four state groups. vLLM keeps these 5 sources
    // in use until the end of `schedule()` (v0.31 sched:1434-1442, v0.32
    // sched:1474-1482, v0.33 sched:1698-1706). So RC holds 12 blocks in its
    // step: 2 full-attention blocks, the ring block, 4 running state blocks
    // and the 5 sources. RN needs 6 blocks. On a pool of 15 blocks vLLM
    // refused RN in RC's step and admitted it in the next one. Replay
    // releases the sources at the end of the pass, so RN waits one pass on
    // pools of 13 to 17 blocks. On 18 blocks both fit in one pass. On 12
    // blocks RN waits until RC ends.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for blocks in 12..=18 {
            let expected = match blocks {
                12 => (0, 4, vec![7, 7, 7, 0, 6, 6, 6, 0], 12),
                18 => (0, 0, vec![13, 13, 13, 0], 18),
                _ => (0, 1, vec![7, 13, 13, 6, 0], 12),
            };
            assert_eq!(
                tight_pool_case(version, blocks, 1),
                expected,
                "{version} on {blocks} blocks"
            );
        }
    }
}

#[test]
fn grouped_prefix_boundary_hit_holds_its_state_blocks_until_the_next_pass() {
    // RP has 4,400 tokens, so its chunks end at 4,352 and 4,400. RC hits
    // 4,352 tokens, on a block boundary, and takes no copy. vLLM keeps the
    // hit block of each state group in RC's table until RC's next
    // `allocate_slots` (v0.31 kv_cache_manager.py:552-556,
    // single_type_kv_cache_manager.py:1626-1645, v0.32 :555-559,
    // :1592-1611, v0.33 :564-568, :1608-1627). So RC holds 12 blocks in its
    // step: 3 full-attention blocks, the ring block, and a hit block and a
    // running block in each state group. RN needs 6 blocks. Run on vLLM's
    // cache manager, this case refuses RN in RC's step and admits it in the
    // next one on pools of 14 to 17 blocks. Replay does the same. On 18
    // blocks both fit in one pass. On 12 and 13 blocks RN waits until RC
    // ends.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for blocks in 12..=18 {
            let expected = match blocks {
                12 | 13 => (0, 4, vec![12, 8, 8, 0, 6, 6, 6, 0], 12),
                18 => (0, 0, vec![18, 14, 14, 0], 18),
                _ => (0, 1, vec![12, 14, 14, 6, 0], 12),
            };
            assert_eq!(
                pool_case(version, blocks, 1, 4400, 6528, 4352),
                expected,
                "{version} on {blocks} blocks"
            );
        }
    }
}

#[test]
fn grouped_prefix_steps_that_the_kv_manager_skips_change_nothing() {
    // The KV manager skips the block walks of a step below a lease's quiet
    // position. Debug builds run them and assert that they change nothing.
    // Each request here takes a step that a wrong position would skip.
    let glm = |checkpoint, chunk, unit, version| {
        prefix_core(glm_groups(checkpoint), 64 * GLM_PAGE, chunk, unit, version)
    };
    let mut window = group("window", 4, 8, Some(6));
    window.prefix_cacheable = Some(true);
    let cases = [
        // Decode crosses a block boundary: the previous state block of the
        // last chunk goes, the attention block completes, new blocks come.
        (glm(false, 8192, 1088, "0.31.0"), 2170, 10),
        // A prompt of one chunk decodes with its checkpoint block and no
        // previous state block.
        (glm(true, 8192, 2176, "0.33.0"), 5000, 4),
        // A step budget below the block: the prefill advances inside a block
        // that the request holds, and publishes at its tail, 5,440.
        (glm(false, 512, 1088, "0.31.0"), 5600, 4),
        // The prompt ends at its tail, so its running state block is
        // published and the first decode step takes a fresh one.
        (glm(false, 8192, 1088, "0.31.0"), 1088, 4),
        // A window that is no multiple of the block retires a block inside
        // the next one.
        (prefix_core(vec![window], 9 * 8, 8, 4, "0.33.0"), 8, 8),
    ];
    for (mut core, prompt, output) in cases {
        let request = submit(&mut core, 1, prompt, output);
        let admissions = drain(&mut core, &mut TraceCollector::default(), &mut 0);
        assert_eq!(admissions, vec![(request, 0)], "a {prompt}-token prompt");
    }
}

#[test]
fn grouped_prefix_copy_sources_leave_in_group_order_at_the_end_of_the_pass() {
    // Two consumers with hits inside a block in one pass. vLLM frees their
    // copy sources at the end of `schedule()`, manager by manager in group
    // order (v0.31 kv_cache_manager.py:894-909, sched:1434-1442, v0.32
    // kv_cache_manager.py:854-869, sched:1474-1482, v0.33
    // kv_cache_manager.py:863-878, sched:1698-1706), and each cached source
    // goes to the tail of the free queue (v0.31 block_pool.py:777-808, v0.32
    // :774-805, v0.33 :816-849). So the full-attention sources of both
    // consumers enter the LRU first, then the sources of group 2 of both, and
    // so on. A cold request X then evicts the oldest cached blocks, and a
    // later request on either prefix finds the same hit. After the consumers
    // end, 22 blocks are cached: the 8 producer states at 2,176, the 10
    // sources, and the 4 full-attention blocks of the consumers, whose copies
    // keep the key of 3,264. 18 blocks are free. On the 0.31.0 rules X (26
    // blocks) computes its whole prompt in one chunk, since its last
    // cacheable stop is its length: 26 + 1 + 4 = 31 blocks, 13 evictions.
    // They take the 8 states, both full-attention sources, both sources of
    // group 2 and one of group 3, so neither prefix hits. X ends in that
    // pass, and its 26 full-attention blocks join the 9 cached blocks left:
    // 35. On the 0.32.0 and 0.33.0 rules the first chunk of X (24 blocks)
    // stops at (52,224 - 1) / 2,176 * 2,176 = 50,048: 23 + 1 + 4 = 28 blocks,
    // 10 evictions. They end with both full-attention sources, every state
    // source stays, both prefixes hit 3,264, and 12 blocks stay cached.
    // (version, X's blocks, inactive blocks after X's pass, the two hits)
    for (version, x_blocks, inactive_after_x, hits) in [
        ("0.31.0", 26, 35, (0, 0)),
        ("0.32.0", 24, 12, (3264, 3264)),
        ("0.33.0", 24, 12, (3264, 3264)),
    ] {
        let mut core = VllmCore::new(
            MockEngineArgs::builder()
                .block_size(4)
                .num_gpu_blocks(1)
                .max_num_batched_tokens(Some(65_536))
                .max_num_seqs(Some(4))
                .max_model_len(Some(131_072))
                .enable_prefix_caching(true)
                .prefix_match_unit(Some(1088))
                .vllm_version(Some(version.to_string()))
                .kv_cache_groups(glm_groups(false))
                .kv_cache_capacity_bytes(Some(40 * GLM_PAGE))
                .speedup_ratio(0.0)
                .build()
                .unwrap(),
        );
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let first: Vec<u32> = (44_000_000..44_003_500).collect();
        let second: Vec<u32> = (45_000_000..45_003_500).collect();
        submit_tokens(&mut core, 1, first.clone(), 1);
        submit_tokens(&mut core, 2, second.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        let extend = |tokens: &[u32], base: u32| {
            let mut tokens = tokens.to_vec();
            tokens.extend(base..base + 300);
            tokens
        };
        let c1 = submit_tokens(&mut core, 3, extend(&first, 46_000_000), 1);
        let c2 = submit_tokens(&mut core, 4, extend(&second, 47_000_000), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        let admitted: Vec<(Uuid, usize)> = pass
            .admissions
            .iter()
            .map(|admission| (admission.uuid, admission.reused_input_tokens))
            .collect();
        assert_eq!(admitted, [(c1, 3264), (c2, 3264)], "{version}");
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 22, "{version}");
        let x_tokens = x_blocks * GLM_BLOCK;
        submit_tokens(
            &mut core,
            5,
            (48_000_000..48_000_000 + x_tokens).collect(),
            1,
        );
        core.execute_pass(&mut trace, step as f64);
        assert_eq!(
            block_pool(&core).inactive_cached_blocks(),
            inactive_after_x,
            "{version}"
        );
        let hit = |tokens: Vec<u32>| {
            let mut probe = super::request::RequestKvState::native(
                Uuid::from_u128(6),
                tokens.clone(),
                4,
                4,
                core.args.block_size,
                true,
                false,
                false,
                None,
            );
            probe.lease.configure_prefix_hashes(&tokens, 1088);
            core.kv_manager
                .get_native_prefill_cost(&probe.sequence, &probe.lease)
                .cached_tokens
        };
        assert_eq!(
            (
                hit(extend(&first, 49_000_000)),
                hit(extend(&second, 49_500_000))
            ),
            hits,
            "{version}"
        );
    }
}

#[test]
fn grouped_prefix_eviction_frees_cached_blocks_oldest_first() {
    // One full-attention group of 4-token blocks in a pool of 16 blocks.
    let mut full = group("full", 4, 8, None);
    full.prefix_cacheable = Some(true);
    let mut core = prefix_core(vec![full], 16 * 8, 64, 4, "0.31.0");
    let mut trace = TraceCollector::default();
    let mut step = 0;
    // Three finished requests of 16 tokens leave 12 cached blocks.
    for (id, base) in [(801, 0), (802, 100), (803, 200)] {
        submit_tokens(&mut core, id, (base..base + 16).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
    }
    assert_eq!(block_pool(&core).inactive_cached_blocks(), 12);
    // A fourth request of 40 tokens needs 10 fresh blocks: the 4 free ones
    // and the 6 evicted oldest first, in release order (tail first): all of
    // request one and the two last blocks of request two.
    let fourth = submit_tokens(&mut core, 804, (300..340).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(fourth, 0)]);
    assert_eq!(block_pool(&core).inactive_cached_blocks(), 16);
    // A fifth request equal to the third one still hits, capped below the
    // prompt: (16 - 1) / 4 * 4 = 12.
    let fifth = submit_tokens(&mut core, 805, (200..216).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(fifth, 12)]);
    // The fifth request's one fresh block evicted the older of request
    // two's two surviving blocks, so request two hits its first block only.
    let sixth = submit_tokens(&mut core, 806, (100..116).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(sixth, 4)]);
    // Request one is gone.
    let seventh = submit_tokens(&mut core, 807, (0..16).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(seventh, 0)]);
}

/// A rank with a full-attention group and a prefix-cacheable state group of
/// 4-token blocks, a hash block of 4 and `blocks` pool blocks.
fn async_prefix_core(version: &str, blocks: u64, max_concurrent_batches: usize) -> VllmCore {
    let mut attention = group("attention", 4, 8, None);
    attention.prefix_cacheable = Some(true);
    let mut kda = state("kda", 8, 1);
    kda.prefix_cacheable = Some(true);
    kda.prefill_checkpoint_blocks = (version != "0.31.0").then_some(0);
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(8))
            .max_model_len(Some(64))
            .enable_prefix_caching(true)
            .prefix_match_unit(Some(4))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(vec![attention, kda])
            .kv_cache_capacity_bytes(Some(blocks * 8))
            .speedup_ratio(0.0)
            .max_concurrent_batches(max_concurrent_batches)
            .build()
            .unwrap(),
    )
}

/// One pass at `now_ms` and its boundary. `now_ms` moves to the end of the
/// pass, so the next pass is formed while this one is in flight.
fn async_step(core: &mut VllmCore, now_ms: &mut f64) -> crate::engine::scheduler::EnginePassResult {
    let mut trace = TraceCollector::default();
    let result = core.execute_pass(&mut trace, *now_ms);
    core.complete_engine_boundary(result.end_ms);
    *now_ms = result.end_ms;
    result
}

fn drain_async(core: &mut VllmCore, now_ms: &mut f64, label: &str) {
    for _ in 0..50 {
        if core.is_drained() {
            return;
        }
        async_step(core, now_ms);
    }
    assert!(
        core.is_drained(),
        "{label}: drain exceeded 50 passes; {} active blocks",
        core.kv_manager.grouped().unwrap().active_blocks()
    );
}

#[test]
fn grouped_prefix_blocks_of_a_finished_request_first_serve_pass_k_plus_2_with_async_scheduling() {
    // X (4 tokens) keeps the rank busy. R (17 tokens) prefills in passes 0
    // and 1 and finishes in pass 1 with 5 attention blocks and 2 state blocks.
    // W (21 tokens) needs 7 blocks of the 12 and fits only in R's blocks.
    // vLLM frees R's blocks when it processes the outputs of step 1 (v0.31
    // sched:2171, :2650-2679, v0.32 sched:2230, :2714-2744, v0.33
    // sched:2476, :3031-3062). Without async scheduling that comes before the
    // scheduling of step 2, so W starts in step 2. With async scheduling the
    // engine schedules step 2 before it processes the outputs of step 1 (v0.31
    // engine/core.py:697, :740-755, v0.32 engine/core.py:725, :768-783, v0.33
    // engine/core.py:735, :778-793), so W starts in step 3. The block counts
    // are replay's. After R's last pass X holds 2 attention blocks, its
    // running state block and the previous state block of its crossing at 4:
    // 4 blocks, and 11 with async scheduling, where R still holds its 7. Pass
    // 2 releases X's previous state block: 3 left with async scheduling.
    // Without async scheduling pass 2 also admits W, whose first chunk of 20
    // tokens takes 5 attention blocks and 1 state block: 9. With async
    // scheduling vLLM also holds X's previous state block for one more step
    // (v0.31 mgr:552-556, v0.32 mgr:555-559, v0.33 mgr:564-568), a known
    // difference.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for (max_concurrent_batches, admitting_pass, after_finish, after_next) in
            [(1, 2, 4, 9), (2, 3, 11, 3)]
        {
            let mut core = async_prefix_core(version, 12, max_concurrent_batches);
            let mut now = 0.0;
            submit_tokens(&mut core, 1, (1000..1004).collect(), 30);
            let r = submit_tokens(&mut core, 2, (2000..2017).collect(), 1);
            let w = submit_tokens(&mut core, 3, (3000..3021).collect(), 1);
            async_step(&mut core, &mut now);
            let finished = async_step(&mut core, &mut now);
            assert!(
                finished
                    .output_signals
                    .iter()
                    .any(|signal| signal.uuid == r && signal.completed),
                "{version}"
            );
            // With async scheduling R still holds its 7 blocks.
            assert_eq!(block_pool(&core).active_blocks(), after_finish, "{version}");
            let mut active = Vec::new();
            let admitted = (2..6).find(|_| {
                let result = async_step(&mut core, &mut now);
                active.push(block_pool(&core).active_blocks());
                assert_eq!(result.mocker_metrics.vllm_preemptions_total, 0, "{version}");
                result
                    .admissions
                    .iter()
                    .any(|admission| admission.uuid == w)
            });
            assert_eq!(admitted, Some(admitting_pass), "{version}");
            assert_eq!(active[0], after_next, "{version}");
        }
    }
}

#[test]
fn grouped_prefix_async_scheduling_frees_a_finished_request_after_a_running_request_releases_its_previous_state()
 {
    // This test pins a known difference from vLLM. G (10 tokens) publishes
    // its state at 8 in pass 0, crosses that boundary in pass 1 and releases
    // the block of that state one pass later, in pass 2. F (9 tokens)
    // prefills 8 tokens in pass 0 and 1 in pass 1, where it finishes with 3
    // attention blocks and 2 state blocks; its cached blocks are attention
    // at 4 and 8 and its state at 8. Without async scheduling F's blocks
    // enter the LRU in pass 1, tail first, before G's block. With async
    // scheduling replay frees F's blocks after it has formed pass 2, so
    // after G's block. After pass 2 G holds 4 blocks, 4 blocks are cached
    // and 3 are free. In pass 4 H (3 tokens) takes 2 blocks and G 2 more (its
    // fourth attention block and its next state block), so the pool evicts
    // its oldest cached block: F's attention block at 8 without async
    // scheduling, G's state block at 8 with it. A later prompt that starts
    // with G's tokens then hits 8 tokens against 0, and one that starts with
    // F's tokens 0 against 8.
    // By a reading of vLLM, F goes first in both modes. vLLM frees F when it
    // processes the outputs of step 1 (v0.31 sched:2171, :2650-2679, v0.32
    // sched:2230, :2714-2744, v0.33 sched:2476, :3031-3062). It frees G's
    // block in `allocate_slots` on the processed-token basis (v0.31
    // mgr:552-556, v0.32 mgr:555-559, v0.33 mgr:564-568), through the base
    // path of `MambaManager.remove_skipped_blocks` (v0.31 single:1647-1675,
    // base path :1655-1657, :687-725, :1626-1645, :1990-1995; v0.32
    // single:1613-1641, base path :1621-1623, :672-710, :1592-1611,
    // :1954-1959; v0.33 single:1629-1657, base path :1637-1639, :682-720,
    // :1608-1627, :1971-1976). With async scheduling the tokens of the step
    // in flight do not count, so vLLM frees G's block in the scheduling of
    // step 3, one step later than replay. So the outcome without async
    // scheduling equals vLLM's, and the outcome with async scheduling is
    // replay's own.
    let cached = |core: &VllmCore, group: u16, tokens: &[u32], end: usize| {
        let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
        lease.configure_prefix_hashes(tokens, 4);
        block_pool(core)
            .key_hit(group, lease.prefix_hash_at(end).unwrap())
            .is_some()
    };
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for max_concurrent_batches in [1usize, 2] {
            let mut core = async_prefix_core(version, 11, max_concurrent_batches);
            let mut now = 0.0;
            let g: Vec<u32> = (1000..1010).collect();
            let f: Vec<u32> = (2000..2009).collect();
            submit_tokens(&mut core, 1, g.clone(), 30);
            let f_id = submit_tokens(&mut core, 2, f.clone(), 1);
            // H arrives so that pass 4 admits it in both modes: without
            // async scheduling after pass 3, and with async scheduling while
            // pass 2 is in flight, when pass 3 is already queued behind it.
            let arrival = if max_concurrent_batches == 1 { 3 } else { 2 };
            let mut admitted = None;
            let mut f_finished = None;
            for pass in 0..5 {
                let result = async_step(&mut core, &mut now);
                if !result.admissions.is_empty() && pass > 0 {
                    admitted = Some(pass);
                }
                if result
                    .output_signals
                    .iter()
                    .any(|signal| signal.uuid == f_id && signal.completed)
                {
                    f_finished = Some(pass);
                }
                if pass == 2 {
                    // After pass 2: G holds 4 blocks and 4 are cached.
                    assert_eq!(block_pool(&core).active_blocks(), 4, "{version}");
                    assert_eq!(block_pool(&core).inactive_cached_blocks(), 4, "{version}");
                }
                if pass == arrival {
                    core.apply_command_effects(
                        SchedulerCommand::Submit(DirectRequest {
                            uuid: Some(Uuid::from_u128(3)),
                            tokens: (3000..3003).collect(),
                            max_output_tokens: 1,
                            ..Default::default()
                        }),
                        false,
                    )
                    .unwrap();
                }
            }
            assert_eq!(f_finished, Some(1), "{version}");
            assert_eq!(admitted, Some(4), "{version}");
            // [F attention at 4, F attention at 8, F state at 8, G state at 8]
            let left = [
                cached(&core, 0, &f, 4),
                cached(&core, 0, &f, 8),
                cached(&core, 1, &f, 8),
                cached(&core, 1, &g, 8),
            ];
            // Later prompts that start with G's and with F's tokens.
            let mut g_probe = waiting_probe(&core, 4, (1000..1016).collect());
            let mut f_probe = waiting_probe(&core, 5, (2000..2016).collect());
            let hits = (
                waiting_hit(&core, &mut g_probe),
                waiting_hit(&core, &mut f_probe),
            );
            if max_concurrent_batches == 1 {
                // vLLM's order in both modes: F's attention block at 8 goes
                // first, G's state at 8 stays.
                assert_eq!(left, [true, false, true, true], "{version}");
                assert_eq!(hits, (8, 0), "{version}");
            } else {
                // Replay with async scheduling: G's state at 8 goes first, so
                // the G prompt hits less and the F prompt more than in vLLM.
                assert_eq!(left, [true, true, true, false], "{version}");
                assert_eq!(hits, (0, 8), "{version}");
            }
        }
    }
}

#[test]
fn grouped_prefix_copy_sources_leave_at_the_end_of_the_pass_with_async_scheduling() {
    // The tight-pool case with async scheduling. Without a KV connector in a
    // consumer role vLLM does not defer its block frees (v0.31
    // sched:156, :175-181, v0.32 sched:167, :186-192, v0.33 sched:169,
    // :189-195), so the copy sources still leave at the end of `schedule()`
    // (v0.31 sched:1434-1442, v0.32 sched:1474-1482, v0.33 sched:1698-1706).
    // RN waits one pass on pools of 13 to 17 blocks, as
    // with one batch. RC's blocks leave one pass later, so on 12 blocks RN
    // starts one pass later than with one batch.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for blocks in 12..=18 {
            let expected = match blocks {
                12 => (0, 5, vec![7, 7, 7, 7, 0, 6, 6, 6, 0], 12),
                18 => (0, 0, vec![13, 13, 13, 0], 18),
                _ => (0, 1, vec![7, 13, 13, 13, 0], 12),
            };
            assert_eq!(
                tight_pool_case(version, blocks, 2),
                expected,
                "{version} on {blocks} blocks"
            );
        }
    }
}

#[test]
fn grouped_prefix_boundary_hit_holds_its_state_blocks_until_the_next_pass_with_async_scheduling() {
    // The boundary-hit case with async scheduling. Replay holds RC's hit
    // blocks until RC's next allocation, one pass, as with one batch. vLLM
    // frees them on the processed token count, so with a step in flight it
    // holds them one more step (v0.31 kv_cache_manager.py:552-556, v0.32
    // :555-559, v0.33 :564-568). Replay does not model that step. RC's
    // blocks leave one pass later than with one batch, so on 12 and 13 blocks
    // RN starts one pass later.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for blocks in 12..=18 {
            let expected = match blocks {
                12 | 13 => (0, 5, vec![12, 8, 8, 8, 0, 6, 6, 6, 0], 12),
                18 => (0, 0, vec![18, 14, 14, 0], 18),
                _ => (0, 1, vec![12, 14, 14, 14, 0], 12),
            };
            assert_eq!(
                pool_case(version, blocks, 2, 4400, 6528, 4352),
                expected,
                "{version} on {blocks} blocks"
            );
        }
    }
}

#[test]
fn grouped_prefill_rank_keeps_copy_sources_until_the_scheduling_pass_ends() {
    // The tight-pool case on a grouped prefill rank with transfer rules, as
    // local requests. vLLM frees the copy sources of RC's hit inside a block
    // at the end of `schedule()` (v0.31 sched:1434-1442, v0.32
    // sched:1474-1482). It defers that free by one step only with async
    // scheduling and a connector in a consumer role, `kv_consumer` or
    // `kv_both` (v0.31 sched:156, :175-181, v0.32 sched:167, :186-192), and
    // replay does not model a connector's role. So with one batch and with
    // async scheduling RC holds its 7 blocks and the 5 sources in its pass,
    // and RN waits one pass on pools of 13 to 17 blocks, as on an aggregated
    // rank.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for max_concurrent_batches in [1, 2] {
            for blocks in 12..=18 {
                let expected = match (blocks, max_concurrent_batches) {
                    (12, 1) => (0, 4, vec![7, 7, 7, 0, 6, 6, 6, 0], 12),
                    (12, _) => (0, 5, vec![7, 7, 7, 7, 0, 6, 6, 6, 0], 12),
                    (18, _) => (0, 0, vec![13, 13, 13, 0], 18),
                    (_, 1) => (0, 1, vec![7, 13, 13, 6, 0], 12),
                    _ => (0, 1, vec![7, 13, 13, 13, 0], 12),
                };
                let mut args = pool_args(version, blocks, max_concurrent_batches, 4352);
                args.kv_cache_groups = glm_transfer_groups(version, false);
                args.worker_type = WorkerType::Prefill;
                args.kv_transfer_bandwidth = Some(1.0);
                assert_eq!(
                    pool_run(VllmCore::new(args), 3500, 3264),
                    expected,
                    "{version} {max_concurrent_batches} on {blocks} blocks"
                );
            }
        }
    }
}

/// A waiting request outside the core's queues, with the prefix hashes that
/// the core gives a request it receives.
fn waiting_probe(core: &VllmCore, id: u128, tokens: Vec<u32>) -> RequestKvState {
    let mut probe = RequestKvState::native(
        Uuid::from_u128(id),
        tokens.clone(),
        4,
        4,
        core.args.block_size,
        true,
        false,
        false,
        None,
    );
    probe
        .lease
        .configure_prefix_hashes(&tokens, core.args.prefix_match_unit.unwrap());
    probe
}

/// The cached tokens that the waiting-loop lookup gives `probe`, checked
/// against a fresh search of the same pool. The lookup that `schedule_request`
/// uses must give the same cost from the hit that the waiting lookup stored.
fn waiting_hit(core: &VllmCore, probe: &mut RequestKvState) -> usize {
    let lookup = core
        .kv_manager
        .get_native_waiting_prefill_cost(&probe.sequence, &mut probe.lease);
    let fresh = core
        .kv_manager
        .get_native_prefill_cost(&probe.sequence, &probe.lease);
    assert_eq!(
        (
            lookup.new_blocks,
            lookup.cached_tokens,
            lookup.active_cached_tokens
        ),
        (
            fresh.new_blocks,
            fresh.cached_tokens,
            fresh.active_cached_tokens
        )
    );
    let scheduled = core
        .kv_manager
        .lookup_native_prefill_cost(&probe.sequence, &mut probe.lease);
    assert_eq!(
        (
            scheduled.new_blocks,
            scheduled.cached_tokens,
            scheduled.active_cached_tokens
        ),
        (
            lookup.new_blocks,
            lookup.cached_tokens,
            lookup.active_cached_tokens
        )
    );
    lookup.cached_tokens
}

#[test]
fn grouped_prefix_waiting_hit_follows_keys_that_enter_and_leave_the_pool() {
    // One full-attention group of 4-token blocks in a pool of 16 blocks. A
    // waiting request of 16 tokens is looked up in every step. Its hit must
    // follow the pool: none, then the keys of a finished equal request, then
    // none again once a long request evicts them.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut full = group("full", 4, 8, None);
        full.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![full], 16 * 8, 64, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let mut probe = waiting_probe(&core, 2001, (0..16).collect());
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
        submit_tokens(&mut core, 2002, (0..16).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        // Capped below the prompt: (16 - 1) / 4 * 4 = 12.
        assert_eq!(waiting_hit(&core, &mut probe), 12, "{version}");
        assert_eq!(waiting_hit(&core, &mut probe), 12, "{version}");
        // 64 new tokens take all 16 blocks and evict the four cached ones.
        submit_tokens(&mut core, 2003, (1000..1064).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
    }
}

#[test]
fn grouped_prefix_waiting_hit_follows_the_state_keys_published_in_the_step() {
    // A state key published in a step is no hit in that step
    // (single:1726-1734). A 13-token request publishes its state at 12 in
    // its one pass. A waiting 33-token request repeats it: between that pass
    // and the next step its state lookup misses at 12, so it hits nothing.
    // The next step clears the set, and the same request hits 12 tokens with
    // no change of the resident keys.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut core = fine_grained_core(version);
        let mut trace = TraceCollector::default();
        let mut probe = waiting_probe(&core, 2101, (0..33).collect());
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
        submit_tokens(&mut core, 2102, (0..13).collect(), 1);
        let pass = core.execute_pass(&mut trace, 0.0);
        assert_eq!(pass.admissions.len(), 1, "{version}");
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
        core.kv_manager.begin_step();
        assert_eq!(waiting_hit(&core, &mut probe), 12, "{version}");
    }
}

#[test]
fn grouped_pd_rank_on_vllm_0330_holds_no_waiting_request_after_the_step_that_publishes_a_state_key()
{
    // vLLM 0.33.0 does not publish again a state block that already has the
    // key of its position (v0.33 single:2112-2115). So the key does not go
    // into the set of the step again (v0.33 single:1994-1999), and a request
    // that hits it waits for the publishing step only (v0.33
    // single:1704-1712). Replay publishes a state key once. P (40 tokens)
    // ends its first chunk at the last cacheable stop, (40 - 1) / 16 * 16 =
    // 32, and publishes its state there. Its second chunk, 32..40, is a
    // second cache step for that key. W (48 tokens) repeats the prompt of P.
    for (role, worker_type) in [
        ("prefill", WorkerType::Prefill),
        ("decode", WorkerType::Decode),
    ] {
        let mut groups = full_and_state_transfer_groups();
        groups[1].prefill_checkpoint_blocks = Some(0);
        let mut args = prefix_core(groups, 1_600, 8192, 16, "0.33.0").args;
        args.worker_type = worker_type;
        let mut core = VllmCore::new(args);
        let mut trace = TraceCollector::default();
        let mut probe = waiting_probe(&core, 2402, (0..48).collect());
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{role}");
        submit_tokens(&mut core, 2401, (0..40).collect(), 1);
        // The publishing step: the key at 32 is no hit.
        let pass = core.execute_pass(&mut trace, 0.0);
        assert_eq!(pass.admissions.len(), 1, "{role}");
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{role}");
        // The next step: P computes 32..40, and W hits 32 tokens.
        let pass = core.execute_pass(&mut trace, 1.0);
        assert!(
            pass.output_signals.iter().all(|output| !output.rejected),
            "{role}"
        );
        assert_eq!(waiting_hit(&core, &mut probe), 32, "{role}");
    }
}

#[test]
fn grouped_prefix_waiting_hit_follows_the_token_count_of_the_request() {
    // The hit stops below the last known token (mgr:289-295). A waiting
    // request of 8 tokens hits 4 of a cached 16-token prefix; one more token
    // lifts the cap, and the same pool gives 8.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut full = group("full", 4, 8, None);
        full.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![full], 16 * 8, 64, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 2201, (0..16).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        let mut probe = waiting_probe(&core, 2202, (0..8).collect());
        assert_eq!(waiting_hit(&core, &mut probe), 4, "{version}");
        probe.generate_token();
        assert_eq!(probe.len(), 9, "{version}");
        assert_eq!(waiting_hit(&core, &mut probe), 8, "{version}");
    }
}

#[test]
fn grouped_prefix_waiting_hit_follows_new_prefix_hashes_of_the_request() {
    // The prefix hashes are an input of the search. A waiting request that
    // hits 12 tokens of a cached prefix hits nothing once its hashes are set
    // from other tokens, with no change of the pool.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut full = group("full", 4, 8, None);
        full.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![full], 16 * 8, 64, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 2301, (0..16).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        let mut probe = waiting_probe(&core, 2302, (0..16).collect());
        assert_eq!(waiting_hit(&core, &mut probe), 12, "{version}");
        let other: Vec<u32> = (5000..5016).collect();
        probe.lease.configure_prefix_hashes(&other, 4);
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
    }
}

/// `(id, submit pass, prompt token ranges, output tokens)`.
type PlannedRequest = (u128, usize, &'static [(u32, u32)], usize);

/// Twelve requests with shared prefixes, chunked prefill and eviction.
const SINGLE_GROUP_PLAN: [PlannedRequest; 12] = [
    (901, 0, &[(0, 300)], 3),
    (902, 0, &[(0, 300)], 3),
    (903, 3, &[(0, 256)], 2),
    (904, 6, &[(0, 100), (1000, 1150)], 4),
    (905, 8, &[(5000, 6600)], 5),
    (906, 9, &[(7000, 9000)], 6),
    (907, 10, &[(10_000, 12_500)], 3),
    (908, 30, &[(13_000, 14_600)], 2),
    (909, 45, &[(5000, 6600), (30_000, 30_030)], 3),
    (910, 60, &[(40_000, 40_192), (50_000, 50_040)], 2),
    (911, 60, &[(40_000, 40_192), (51_000, 51_010)], 2),
    (912, 70, &[(0, 300), (60_000, 60_500)], 4),
];

/// A linear core and a grouped core with one full-attention group of the
/// same block and capacity, with every other setting equal.
fn single_group_cores(version: &str) -> (VllmCore, VllmCore) {
    let args = || {
        MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(256))
            .max_num_seqs(Some(4))
            .max_model_len(Some(4096))
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
    };
    let linear = VllmCore::new(args().num_gpu_blocks(128).build().unwrap());
    let mut full = group("full", 64, 1, None);
    full.prefix_cacheable = Some(true);
    let grouped = VllmCore::new(
        args()
            .num_gpu_blocks(1)
            .prefix_match_unit(Some(64))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(vec![full])
            .kv_cache_capacity_bytes(Some(128))
            .build()
            .unwrap(),
    );
    (linear, grouped)
}

#[test]
fn single_full_attention_group_matches_the_linear_pool() {
    // Every rule row. One full-attention group at the hash block leaves every
    // row difference inert (the last cacheable stop, the partial tail and the
    // checkpoint block), so every row matches the linear pool.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let (mut linear, mut grouped) = single_group_cores(version);
        let mut linear_trace = TraceCollector::default();
        let mut grouped_trace = TraceCollector::default();
        let computed = |core: &VllmCore| {
            let mut computed: Vec<(Uuid, usize)> = core
                .state
                .requests
                .iter()
                .map(|(id, request)| (*id, request.num_computed_tokens))
                .collect();
            computed.sort();
            computed
        };
        let mut next = 0;
        let mut hits = Vec::new();
        for pass in 0..400 {
            while let Some(&(id, at, ranges, output)) = SINGLE_GROUP_PLAN.get(next)
                && at == pass
            {
                let tokens: Vec<u32> = ranges.iter().flat_map(|&(a, b)| a..b).collect();
                submit_tokens(&mut linear, id, tokens.clone(), output);
                submit_tokens(&mut grouped, id, tokens, output);
                next += 1;
            }
            let expected = linear.execute_pass(&mut linear_trace, pass as f64);
            let actual = grouped.execute_pass(&mut grouped_trace, pass as f64);
            let reused = |admissions: &[crate::engine::scheduler::AdmissionEvent]| {
                admissions
                    .iter()
                    .map(|admission| (admission.uuid, admission.reused_input_tokens))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                reused(&actual.admissions),
                reused(&expected.admissions),
                "{version} pass {pass}"
            );
            assert_eq!(
                computed(&grouped),
                computed(&linear),
                "{version} pass {pass}"
            );
            assert_eq!(
                grouped.kv_manager.num_active_blocks(),
                linear.kv_manager.num_active_blocks(),
                "{version} pass {pass}"
            );
            assert!(
                expected
                    .output_signals
                    .iter()
                    .chain(&actual.output_signals)
                    .all(|output| !output.rejected)
            );
            hits.extend(reused(&actual.admissions));
            if next == SINGLE_GROUP_PLAN.len() && linear.is_drained() && grouped.is_drained() {
                break;
            }
        }
        assert!(linear.is_drained() && grouped.is_drained(), "{version}");
        // The linear cap is `known - 1` floored to the block
        // (`apply_prefix_recompute`), the grouped one `num_tokens - 1` floored to
        // the alignment (`mgr:289-295`). With one group both are the block.
        let id = Uuid::from_u128;
        assert_eq!(
            hits,
            vec![
                (id(901), 0),
                // The four blocks that request 901 holds.
                (id(902), 256),
                // Four cached blocks, capped below the prompt: (256 - 1) / 64 * 64.
                (id(903), 192),
                (id(904), 64),
                (id(905), 0),
                (id(906), 0),
                (id(907), 0),
                (id(908), 0),
                (id(909), 1600),
                (id(910), 0),
                // Request 910's three blocks, in the pass that computes them.
                (id(911), 192),
                // Eviction left block 0 of request 901 only.
                (id(912), 64),
            ],
            "{version}"
        );
    }
}
