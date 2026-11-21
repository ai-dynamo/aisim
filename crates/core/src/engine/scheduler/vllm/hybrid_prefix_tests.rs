// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hybrid (attention + recurrent state) prefix-cache rules in align mode: vLLM
//! v0.29.0's shared-prefix junction and same-pass wait, and vLLM 43b4aaea's
//! prefill checkpoint and drop switch. `B` is both the block size and the hash
//! unit. See THIRD_PARTY_NOTICES.md for the vLLM sources.

use uuid::Uuid;

use crate::engine::StateCacheConfig;
use crate::engine::common::protocols::{DirectRequest, MockEngineArgs};
use crate::engine::scheduler::EnginePassResult;

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
fn cold_aligned_admission_requires_the_state_rotation_peak() {
    #[derive(Clone, Default)]
    struct Warnings(
        std::sync::Arc<std::sync::Mutex<Vec<std::collections::BTreeMap<String, String>>>>,
    );
    struct Fields(std::collections::BTreeMap<String, String>);
    impl tracing::field::Visit for Fields {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0.insert(field.name().into(), format!("{value:?}"));
        }
    }
    impl tracing::Subscriber for Warnings {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            let mut fields = Fields(Default::default());
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields.0);
        }
    }

    for capacity in [4, 6] {
        let mut config = args(false, false);
        config.block_size = 64;
        config.prefix_match_unit = Some(64);
        config.num_gpu_blocks = capacity;
        config.state_cache.as_mut().unwrap().bytes_per_request = 1500;
        let mut core = VllmCore::new(config);
        let id = submit(&mut core, 900, (0..65).collect(), 1);
        let warnings = Warnings::default();
        let first = tracing::subscriber::with_default(warnings.clone(), || step(&mut core));
        if capacity == 4 {
            assert!(
                first.admissions.is_empty(),
                "reject before computing 64 tokens"
            );
            assert!(
                first
                    .output_signals
                    .iter()
                    .any(|signal| { signal.uuid == id && signal.completed && signal.rejected })
            );
            assert!(
                warnings.0.lock().unwrap().iter().any(|fields| {
                    fields.get("required_blocks").map(String::as_str) == Some("6")
                        && fields.get("available_blocks").map(String::as_str) == Some("4")
                }),
                "admission must report required and available capacity"
            );
        } else {
            assert_eq!(first.admissions.len(), 1);
            assert_eq!(core.state.requests[&id].num_computed_tokens, 64);
            let last = step(&mut core);
            assert!(
                last.output_signals
                    .iter()
                    .any(|signal| { signal.uuid == id && signal.completed && !signal.rejected })
            );
        }
        assert!(core.state.requests.is_empty());
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert_eq!(core.state.preemptions_total, 0);
    }
}

#[test]
fn cold_one_pass_prefill_under_the_eagle_drop_needs_one_working_state() {
    // The drop keeps no state at token 64, so the 65-token prompt runs in one
    // pass: two token blocks, one working state and one scratch state.
    let mut config = args(false, false);
    config.block_size = 64;
    config.prefix_match_unit = Some(64);
    config.num_gpu_blocks = 6;
    config.aic_nextn = Some(1);
    config.aic_nextn_accept_rates = Some("1".to_string());
    let state = config.state_cache.as_mut().unwrap();
    state.bytes_per_request = 1500;
    state.num_speculative_states = Some(1);
    let mut core = VllmCore::new(config);
    let id = submit(&mut core, 901, (0..65).collect(), 1);
    let mut passes = vec![step(&mut core)];
    assert_eq!(passes[0].admissions.len(), 1);
    while !core.state.requests.is_empty() && passes.len() < 8 {
        passes.push(step(&mut core));
    }
    assert!(
        passes
            .iter()
            .flat_map(|pass| &pass.output_signals)
            .any(|signal| signal.uuid == id && signal.completed && !signal.rejected)
    );
    assert_eq!(core.state.preemptions_total, 0);
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn cold_budget_split_prefill_requires_the_state_rotation_peak() {
    // No retained boundary splits this EAGLE prompt, but the step budget
    // does: two token blocks + two working states + one scratch state = 8.
    for capacity in [6, 8] {
        let mut config = args(false, false);
        config.block_size = 64;
        config.prefix_match_unit = Some(64);
        config.max_num_batched_tokens = Some(64);
        config.num_gpu_blocks = capacity;
        config.aic_nextn = Some(1);
        config.aic_nextn_accept_rates = Some("1".to_string());
        let state = config.state_cache.as_mut().unwrap();
        state.bytes_per_request = 1500;
        state.num_speculative_states = Some(1);
        let mut core = VllmCore::new(config);
        let id = submit(&mut core, 902, (0..65).collect(), 1);
        let first = step(&mut core);
        if capacity == 6 {
            assert!(first.admissions.is_empty(), "reject before partial prefill");
            assert!(
                first
                    .output_signals
                    .iter()
                    .any(|signal| { signal.uuid == id && signal.completed && signal.rejected })
            );
        } else {
            assert_eq!(first.admissions.len(), 1);
            assert_eq!(core.state.requests[&id].num_computed_tokens, 64);
            let last = step(&mut core);
            assert!(
                last.output_signals
                    .iter()
                    .any(|signal| { signal.uuid == id && signal.completed && !signal.rejected })
            );
        }
        assert!(core.state.requests.is_empty());
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert_eq!(core.state.preemptions_total, 0);
    }
}

#[test]
fn cold_budget_split_within_one_state_slot_does_not_rotate() {
    let mut config = args(false, false);
    config.block_size = 64;
    config.prefix_match_unit = Some(64);
    config.max_num_batched_tokens = Some(16);
    // One token block, one two-block working state, one two-block scratch.
    config.num_gpu_blocks = 5;
    config.aic_nextn = Some(1);
    config.aic_nextn_accept_rates = Some("1".to_string());
    let state = config.state_cache.as_mut().unwrap();
    state.bytes_per_request = 1500;
    state.num_speculative_states = Some(1);
    let mut core = VllmCore::new(config);
    let id = submit(&mut core, 904, (0..32).collect(), 1);
    let first = step(&mut core);
    assert_eq!(first.admissions.len(), 1);
    assert_eq!(core.state.requests[&id].num_computed_tokens, 16);
    let last = step(&mut core);
    assert!(
        last.output_signals
            .iter()
            .any(|signal| { signal.uuid == id && signal.completed && !signal.rejected })
    );
    assert!(core.state.requests.is_empty());
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
    assert_eq!(core.state.preemptions_total, 0);
}

#[test]
fn cold_prefill_checkpoint_only_replaces_rotation_in_one_pass() {
    // 193 tokens occupy four blocks. A one-pass prefill uses a two-block
    // working state and checkpoint (8 total). A 128-token budget also holds
    // the previous working state while exporting the checkpoint (10 total).
    // A 64-token budget exports no internal checkpoint, so two working states
    // suffice (8 total), with chunks ending at 64, 128, 192 and 193.
    for (budget, capacity, admitted) in [
        (64, 8, true),
        (128, 8, false),
        (1024, 8, true),
        (128, 10, true),
        (192, 8, true),
    ] {
        let mut config = args(true, false);
        config.block_size = 64;
        config.prefix_match_unit = Some(64);
        config.max_num_batched_tokens = Some(budget);
        config.num_gpu_blocks = capacity;
        config.state_cache.as_mut().unwrap().bytes_per_request = 1500;
        let mut core = VllmCore::new(config);
        let id = submit(&mut core, 903, (0..193).collect(), 1);
        let first = step(&mut core);
        if admitted {
            assert_eq!(first.admissions.len(), 1);
            if budget < 193 {
                assert_eq!(core.state.requests[&id].num_computed_tokens, budget);
                if budget == 64 {
                    for (expected_end, active_blocks) in [(128, 6), (192, 7)] {
                        let pass = step(&mut core);
                        assert!(pass.output_signals.is_empty());
                        assert_eq!(core.state.requests[&id].num_computed_tokens, expected_end);
                        assert_eq!(core.kv_manager.num_active_blocks(), active_blocks);
                    }
                }
                let last = step(&mut core);
                assert!(
                    last.output_signals.iter().any(|signal| {
                        signal.uuid == id && signal.completed && !signal.rejected
                    })
                );
            } else {
                assert!(
                    first.output_signals.iter().any(|signal| {
                        signal.uuid == id && signal.completed && !signal.rejected
                    })
                );
            }
        } else {
            assert!(first.admissions.is_empty(), "reject before partial prefill");
            assert!(
                first
                    .output_signals
                    .iter()
                    .any(|signal| { signal.uuid == id && signal.completed && signal.rejected })
            );
        }
        assert!(core.state.requests.is_empty());
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert_eq!(core.state.preemptions_total, 0);
    }
}

#[test]
fn warm_junction_checkpoint_admission_accounts_for_the_final_chunk() {
    // Four token blocks and two two-block working states fit in eight blocks.
    // A warm attention-only prefix forces a stop at 64, so the final chunk
    // [64, 193) also exports checkpoint 192 and needs ten blocks. The cold
    // budget split at 192 exports no checkpoint and still fits in eight.
    // The junction also rotates the state when the whole prompt fits in the
    // token budget; it is still a two-pass prefill.
    for (warm, budget, capacity) in [
        (false, 192, 8),
        (true, 192, 8),
        (true, 192, 10),
        (true, 1024, 8),
        (true, 1024, 10),
    ] {
        let mut config = args(true, true);
        config.block_size = 64;
        config.prefix_match_unit = Some(64);
        config.max_num_batched_tokens = Some(budget);
        config.num_gpu_blocks = capacity;
        config.state_cache.as_mut().unwrap().bytes_per_request = 1500;
        let mut core = VllmCore::new(config);
        if warm {
            let parent = submit(&mut core, 910, (0..64).collect(), 1);
            let pass = step(&mut core);
            assert!(
                pass.output_signals.iter().any(|signal| {
                    signal.uuid == parent && signal.completed && !signal.rejected
                })
            );
            assert!(core.state.requests.is_empty());
            assert_eq!(core.kv_manager.num_active_blocks(), 0);
        }

        let id = submit(&mut core, 911, (0..193).collect(), 1);
        let first = step(&mut core);
        if warm && capacity == 8 {
            assert!(first.admissions.is_empty(), "reject before partial prefill");
            assert!(
                first
                    .output_signals
                    .iter()
                    .any(|signal| { signal.uuid == id && signal.completed && signal.rejected })
            );
        } else {
            assert_eq!(reused(&first, id), 0);
            let request = &core.state.requests[&id];
            assert_eq!(request.sequence.lease.junction(), warm.then_some(64));
            assert_eq!(request.num_computed_tokens, if warm { 64 } else { 192 });
            let last = step(&mut core);
            assert!(
                last.output_signals
                    .iter()
                    .any(|signal| { signal.uuid == id && signal.completed && !signal.rejected })
            );
        }
        assert!(core.state.requests.is_empty());
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert_eq!(core.state.preemptions_total, 0);
    }
}

#[test]
fn warm_junction_budget_split_does_not_charge_an_unexported_checkpoint() {
    // The attention-only junction changes the chunks from 128 + 65 to
    // 64 + 128 + 1. None exports an internal checkpoint, so four token
    // blocks plus two two-block working states fit in eight blocks.
    let mut config = args(true, true);
    config.block_size = 64;
    config.prefix_match_unit = Some(64);
    config.max_num_batched_tokens = Some(128);
    config.num_gpu_blocks = 8;
    config.state_cache.as_mut().unwrap().bytes_per_request = 1500;
    let mut core = VllmCore::new(config);

    let parent = submit(&mut core, 912, (0..64).collect(), 1);
    let warmup = step(&mut core);
    assert!(
        warmup
            .output_signals
            .iter()
            .any(|signal| signal.uuid == parent && signal.completed && !signal.rejected)
    );
    assert!(core.state.requests.is_empty());
    assert_eq!(core.kv_manager.num_active_blocks(), 0);

    let id = submit(&mut core, 913, (0..193).collect(), 1);
    for expected_end in [64, 192] {
        let pass = step(&mut core);
        assert!(pass.output_signals.is_empty());
        if expected_end == 64 {
            assert_eq!(reused(&pass, id), 0);
        }
        let request = &core.state.requests[&id];
        assert_eq!(request.sequence.lease.junction(), Some(64));
        assert_eq!(request.num_computed_tokens, expected_end);
    }
    let last = step(&mut core);
    assert!(
        last.output_signals
            .iter()
            .any(|signal| signal.uuid == id && signal.completed && !signal.rejected)
    );
    assert!(core.state.requests.is_empty());
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
    assert_eq!(core.state.preemptions_total, 0);
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
