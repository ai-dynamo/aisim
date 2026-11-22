// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hybrid (attention + recurrent state) prefix-cache rules in align mode: vLLM
//! v0.29.0's shared-prefix junction and same-pass wait, vLLM 43b4aaea's
//! prefill checkpoint and drop switch, and the hold of a previous state copy
//! for one step. `B` is both the block size and the hash unit. See
//! THIRD_PARTY_NOTICES.md for the vLLM sources.

use uuid::Uuid;

use crate::engine::StateCacheConfig;
use crate::engine::common::protocols::{DirectRequest, MockEngineArgs, WorkerType};
use crate::engine::scheduler::EnginePassResult;
use crate::engine::trace::TraceCollector;

use super::core::VllmCore;

const B: usize = 4;

/// One 64-byte block per state at 16 bytes per token.
fn args(prefill_checkpoint: bool, junction_checkpoints: bool) -> MockEngineArgs {
    MockEngineArgs::builder()
        .num_gpu_blocks(256)
        .block_size(B)
        .max_num_batched_tokens(Some(1024))
        .max_num_seqs(Some(16))
        .enable_prefix_caching(true)
        .speedup_ratio(0.0)
        .kv_cache_bytes_per_token(Some(16))
        .state_cache(Some(StateCacheConfig {
            bytes_per_request: 64,
            sliding_window: None,
            num_speculative_states: None,
            prefill_checkpoint,
            junction_checkpoints,
        }))
        .prefix_match_unit(Some(B))
        .build()
        .unwrap()
}

fn submit(core: &mut VllmCore, id: u128, tokens: Vec<u32>, output: usize) -> Uuid {
    core.receive(DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens,
        max_output_tokens: output,
        ..Default::default()
    })
}

fn step(core: &mut VllmCore) -> EnginePassResult {
    core.try_execute_hidden_pass(0.0).unwrap()
}

fn run_to_completion(core: &mut VllmCore) {
    for _ in 0..256 {
        if core.state.requests.is_empty() {
            return;
        }
        step(core);
    }
    panic!("requests did not finish");
}

fn reused(pass: &EnginePassResult, id: Uuid) -> usize {
    pass.admissions
        .iter()
        .find(|admission| admission.uuid == id)
        .unwrap_or_else(|| panic!("{id} was not admitted"))
        .reused_input_tokens
}

fn tokens(prefix: std::ops::Range<u32>, suffix: std::ops::Range<u32>) -> Vec<u32> {
    prefix.chain(suffix).collect()
}

#[test]
fn prefill_checkpoint_finishes_the_prompt_in_one_pass() {
    // The final chunk runs to the prompt end and exports the state at
    // (P - 1) // B * B = 12 from inside its forward, in one extra state
    // reserved for that pass. Without it the chunk stops at 12.
    for (prefill_checkpoint, first_pass) in [(true, 14), (false, 12)] {
        let mut core = VllmCore::new(args(prefill_checkpoint, false));
        let id = submit(&mut core, 1, (0..14).collect(), 4);
        step(&mut core);
        assert_eq!(core.state.requests[&id].num_computed_tokens, first_pass);
        if prefill_checkpoint {
            // Four token blocks, the working state and the checkpoint.
            assert_eq!(core.kv_manager.num_active_blocks(), 6);
        }
        // Either way the state at 12 is hittable from the next pass.
        let reader = submit(&mut core, 2, (0..20).collect(), 1);
        assert_eq!(reused(&step(&mut core), reader), 12);
        run_to_completion(&mut core);
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
    }
}

#[test]
fn junction_checkpoint_keeps_the_shared_prefix_state_for_siblings() {
    // The parent caches the shared 12 tokens as attention only; its state sits
    // at its own last full block, 28. A child's attention hit runs deeper than
    // its state hit, so with junctions its prefill stops at 12 and keeps the
    // state there for the next child.
    for (junction_checkpoints, first_pass, sibling_reuse) in [(true, 12, 12), (false, 28, 0)] {
        let mut core = VllmCore::new(args(true, junction_checkpoints));
        submit(&mut core, 1, tokens(0..12, 1000..1020), 2);
        run_to_completion(&mut core);
        let child = submit(&mut core, 2, tokens(0..12, 2000..2016), 2);
        assert_eq!(reused(&step(&mut core), child), 0);
        assert_eq!(core.state.requests[&child].num_computed_tokens, first_pass);
        let sibling = submit(&mut core, 3, tokens(0..12, 3000..3016), 2);
        assert_eq!(reused(&step(&mut core), sibling), sibling_reuse);
        run_to_completion(&mut core);
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
    }
}

#[test]
fn lookup_waits_one_pass_for_a_state_published_in_the_same_pass() {
    // vLLM's lookup sees a state cached by this pass, but cannot allocate a
    // request that hits it. Two identical prompts arrive together; the first
    // exports its checkpoint at 12, and with junction_checkpoints the second
    // waits one pass and resumes there instead of recomputing the prompt.
    let mut core = VllmCore::new(args(true, true));
    let first = submit(&mut core, 1, (0..14).collect(), 2);
    let second = submit(&mut core, 2, (0..14).collect(), 2);
    let pass = step(&mut core);
    assert_eq!(reused(&pass, first), 0);
    assert!(
        pass.admissions
            .iter()
            .all(|admission| admission.uuid != second)
    );
    assert_eq!(reused(&step(&mut core), second), 12);
    run_to_completion(&mut core);
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn disable_eagle_block_drop_keeps_the_last_matched_block_under_mtp() {
    // The previous prompt ends mid-block (14 = 3B + 2). Without the drop its
    // follow-up resumes at that prompt's last full block, 12; with the drop,
    // one block earlier.
    for (disable_eagle_block_drop, expected) in [(true, 12), (false, 8)] {
        let mut config = args(false, false);
        config.aic_nextn = Some(5);
        config.aic_nextn_accept_rates = Some("1,1,1,1,1".to_string());
        config.disable_eagle_block_drop = disable_eagle_block_drop;
        config.state_cache.as_mut().unwrap().num_speculative_states = Some(5);
        let mut core = VllmCore::new(config);
        submit(&mut core, 1, (0..14).collect(), 3);
        run_to_completion(&mut core);
        let follow_up = submit(&mut core, 2, (0..22).collect(), 1);
        assert_eq!(reused(&step(&mut core), follow_up), expected);
    }
}

#[test]
fn linear_state_cache_holds_a_previous_copy_for_one_step_in_both_scheduling_modes() {
    // A budget of one block per pass prefills the 13-token prompt in chunks
    // that end at 4, 8, 12 and 13 tokens. A chunk that ends at `t` writes the
    // state into slot (t - 1) / B: slots 0, 1, 2 and 3, so every chunk moves
    // the state to a new block. The request then takes a new working copy and
    // holds the copy before it for the step of that chunk, and its next
    // allocation releases the held copy, because the copy was held in an
    // earlier step. After each pass the request has 1, 2, 3 and 4 token
    // blocks, the working copy, and from the second chunk on one held copy:
    // 2, 4, 5 and 6 blocks. The release reads the step and no token count, so
    // a pass in flight does not change it, and the counts are the same with
    // one batch and with two.
    //
    // With async scheduling vLLM frees a state block only when the processed
    // token count of the request, the computed count minus the tokens of the
    // step in flight, lies past it (v0.31 kv_cache_manager.py:549-556,
    // single_type_kv_cache_manager.py:1647-1675; v0.32
    // kv_cache_manager.py:552-559, single_type_kv_cache_manager.py:1613-1641;
    // v0.33 kv_cache_manager.py:561-568,
    // single_type_kv_cache_manager.py:1629-1657). So vLLM holds one state
    // copy more for one step: the third chunk still holds the block of the
    // first, and the fourth the block of the second, 6 and 7 blocks. Replay
    // keeps the hold of one step for a state cache in both scheduling modes,
    // and the State cache section of docs/replay/engine/kv-cache.md lists this
    // difference.
    for (max_concurrent_batches, in_flight) in [
        (1, [false, false, false, false]),
        (2, [false, true, true, true]),
    ] {
        let mut config = args(false, false);
        config.max_num_batched_tokens = Some(B);
        config.max_concurrent_batches = max_concurrent_batches;
        config.worker_type = WorkerType::Aggregated;
        let mut core = VllmCore::new(config);
        let id = submit(&mut core, 1, (0..13).collect(), 3);
        let mut now = 0.0;
        let seen = [(); 4].map(|()| {
            let mut trace = TraceCollector::default();
            let pass = core.execute_pass(&mut trace, now);
            core.complete_engine_boundary(pass.end_ms);
            now = pass.end_ms;
            (
                core.kv_manager.previous_pass_in_flight(),
                core.state.requests[&id].num_computed_tokens,
                core.kv_manager.num_active_blocks(),
            )
        });
        assert_eq!(
            seen.map(|pass| pass.0),
            in_flight,
            "{max_concurrent_batches}"
        );
        assert_eq!(
            seen.map(|pass| pass.1),
            [4, 8, 12, 13],
            "{max_concurrent_batches}"
        );
        assert_eq!(
            seen.map(|pass| pass.2),
            [2, 4, 5, 6],
            "{max_concurrent_batches}"
        );
    }
}

/// The in-flight flag, the computed tokens of `id` and the active blocks
/// after each of `count` passes that follow one another back to back.
fn back_to_back_passes(core: &mut VllmCore, id: Uuid, count: usize) -> Vec<(bool, usize, usize)> {
    let mut now = 0.0;
    (0..count)
        .map(|_| {
            let mut trace = TraceCollector::default();
            let pass = core.execute_pass(&mut trace, now);
            core.complete_engine_boundary(pass.end_ms);
            now = pass.end_ms;
            (
                core.kv_manager.previous_pass_in_flight(),
                core.state.requests[&id].num_computed_tokens,
                core.kv_manager.num_active_blocks(),
            )
        })
        .collect()
}

#[test]
fn linear_state_cache_holds_the_copy_of_a_hit_for_one_step_in_both_scheduling_modes() {
    // A request that hits a cached state on a block boundary holds that
    // state as the source of its first pass, and its next allocation
    // releases it, because the copy was held in an earlier step. The release
    // reads the step and no token count, so the counts are the same with one
    // batch and with two.
    //
    // A 13-token prompt completes and leaves its state at 12 tokens. A
    // 21-token prompt with the same first 13 tokens hits that state and runs
    // with a budget of one block per pass: 12 to 16, 16 to 20, 20 to 21 and
    // the decode step 21 to 22. A pass that ends at `t` writes the state
    // into slot (t - 1) / B: slots 3, 4, 5 and 5. Pass 1: 3 reused and 1 new
    // token block, the working copy and the cached state of the hit, 6.
    // Pass 2 releases the hit copy: 5 token blocks, the working copy of slot
    // 4 and the copy of slot 3, 7. Pass 3: 6 token blocks, the working copy
    // of slot 5 and the copy of slot 4, 8. Pass 4 stays in slot 5: 6 token
    // blocks and the working copy, 7.
    //
    // vLLM frees such a block at the first allocation whose processed token
    // count lies past it (v0.31 kv_cache_manager.py:549-556, v0.32 :552-559,
    // v0.33 :561-568), which is one step later with async scheduling. The
    // State cache section of docs/replay/engine/kv-cache.md lists this
    // difference.
    for max_concurrent_batches in [1, 2] {
        let in_flight = max_concurrent_batches == 2;
        let mut config = args(false, false);
        config.max_num_batched_tokens = Some(B);
        config.max_concurrent_batches = max_concurrent_batches;
        config.worker_type = WorkerType::Aggregated;
        let mut core = VllmCore::new(config);
        submit(&mut core, 1, (0..13).collect(), 1);
        run_to_completion(&mut core);
        let id = submit(&mut core, 2, (0..21).collect(), 3);
        assert_eq!(
            back_to_back_passes(&mut core, id, 4),
            [
                (false, 16, 6),
                (in_flight, 20, 7),
                (in_flight, 21, 8),
                (in_flight, 22, 7),
            ],
            "{max_concurrent_batches}"
        );
    }
}

#[test]
fn linear_state_cache_holds_a_checkpoint_copy_for_one_step_in_both_scheduling_modes() {
    // With `prefill_checkpoint` the final prefill chunk keeps a checkpoint
    // copy of the state at the last block boundary before the prompt end.
    // The request holds that copy for the step of the chunk, and its next
    // allocation releases it, with one batch and with two.
    //
    // A 14-token prompt with a budget of two blocks per pass runs 0 to 8 and
    // the final chunk 8 to 14, which keeps the checkpoint copy at 12, then
    // decodes 14 to 15, 15 to 16 and 16 to 17: slots 1, 3, 3, 3 and 4. Pass
    // 1: 2 token blocks and the working copy, 3. Pass 2: 4 token blocks, the
    // working copy of slot 3, the copy of slot 1 and the checkpoint copy, 7.
    // Pass 3 releases both held copies: 4 token blocks and the working copy,
    // 5. Pass 4: the same 5. Pass 5: 5 token blocks, the working copy of
    // slot 4 and the copy of slot 3, 7.
    //
    // vLLM frees such a block at the first allocation whose processed token
    // count lies past it (v0.31 kv_cache_manager.py:549-556, v0.32 :552-559,
    // v0.33 :561-568), which is one step later with async scheduling. The
    // State cache section of docs/replay/engine/kv-cache.md lists this
    // difference.
    for max_concurrent_batches in [1, 2] {
        let in_flight = max_concurrent_batches == 2;
        let mut config = args(true, false);
        config.max_num_batched_tokens = Some(2 * B);
        config.max_concurrent_batches = max_concurrent_batches;
        config.worker_type = WorkerType::Aggregated;
        let mut core = VllmCore::new(config);
        let id = submit(&mut core, 1, (0..14).collect(), 6);
        assert_eq!(
            back_to_back_passes(&mut core, id, 5),
            [
                (false, 8, 3),
                (in_flight, 14, 7),
                (in_flight, 15, 5),
                (in_flight, 16, 5),
                (in_flight, 17, 7),
            ],
            "{max_concurrent_batches}"
        );
    }
}
