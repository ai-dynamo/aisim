// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM async scheduling in offline replay.

use std::collections::VecDeque;

use aisim_core::engine::{
    EngineConfig, G2Connector, NativeHostOffloadConfig, PassLaunchConfig, TimingModelConfig,
};
use aisim_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, DirectRequest, PerRequestRecord, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer,
    WorkerPoolSpec,
};
use uuid::Uuid;

/// One vLLM worker with `num_gpu_blocks` 4-token blocks. A pass with prefill
/// takes 100 ms and a decode pass 10 ms.
fn engine(num_gpu_blocks: usize) -> EngineConfig {
    EngineConfig {
        block_size: 4,
        num_gpu_blocks,
        max_num_batched_tokens: 64,
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 100.0,
            decode_ms: 10.0,
        },
        ..EngineConfig::default()
    }
}

/// `engine` with `launch`: launching a pass takes `ms` without prefill, with up
/// to 16 scheduled tokens and above.
fn launch(engine: EngineConfig, ms: [f64; 3], eager_queue_fraction: f64) -> EngineConfig {
    let [decode_ms, piecewise_ms, eager_ms] = ms;
    EngineConfig {
        launch: Some(PassLaunchConfig {
            decode_ms,
            piecewise_ms,
            eager_ms,
            max_cudagraph_capture_size: 16,
            eager_queue_fraction,
        }),
        ..engine
    }
}

fn spec(engine: ReplayEngineConfig) -> ReplaySpec {
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        encoder: None,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: 1,
                ..WorkerPoolSpec::default()
            },
        },
        engine: serde_json::to_value(engine).unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    }
}

/// A request with `(id, prompt tokens, outputs, arrival)`.
fn request(&(id, prompt, outputs, arrival_ms): &(u128, u32, usize, f64)) -> DirectRequest {
    DirectRequest {
        tokens: (0..prompt).map(|token| token + id as u32 * 1000).collect(),
        max_output_tokens: outputs,
        uuid: Some(Uuid::from_u128(id)),
        arrival_timestamp_ms: Some(arrival_ms),
        ..Default::default()
    }
}

fn replay_requests(engine: ReplayEngineConfig, requests: VecDeque<DirectRequest>) -> Replayer {
    Replayer::new(spec(engine), ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Requests(requests))
}

/// A replayer of `(id, prompt tokens, outputs, arrival)` requests.
fn replayer(engine: EngineConfig, requests: &[(u128, u32, usize, f64)]) -> Replayer {
    replay_requests(
        ReplayEngineConfig {
            rank: engine,
            ..Default::default()
        },
        requests.iter().map(request).collect(),
    )
}

fn run(engine: EngineConfig, requests: &[(u128, u32, usize, f64)]) -> ReplayReport {
    replayer(engine, requests).run().unwrap()
}

fn record(report: &ReplayReport, id: u128) -> &PerRequestRecord {
    let uuid = Uuid::from_u128(id).to_string();
    report
        .per_request
        .iter()
        .find(|record| record.uuid == uuid)
        .unwrap()
}

#[test]
fn arrivals_after_the_last_running_request_get_its_blocks_at_once() {
    // R (24 tokens) prefills until 100 and finishes in its decode pass at 110.
    // Requests 2 and 3 arrive during that pass, so vLLM had already decided
    // the pass at 110, which has nothing to run. That empty step frees R's
    // blocks, and vLLM then admits both at 110, though request 3 (16 tokens)
    // fits only in R's blocks.
    let report = run(
        engine(10),
        &[(1, 24, 2, 0.0), (2, 4, 2, 105.0), (3, 16, 2, 106.0)],
    );
    assert_eq!(
        [2, 3].map(|id| record(&report, id).first_admit_ms),
        [Some(110.0); 2]
    );
}

#[test]
fn every_attention_dp_rank_frees_finished_blocks_after_the_next_decision() {
    // On rank 0, X and R prefill until 100, and W (16 tokens), arriving at 50,
    // fits only in R's blocks. R finishes in its decode pass, which ends at
    // 110, but rank 1's prefill of Y holds the group pass until 200. Rank 0
    // decides its pass at 200 before it frees R's blocks, as a single rank
    // would, so that pass decodes only X and W is admitted at 210.
    let requests = [
        ((1, 8, 4, 0.0), 0),
        ((2, 24, 2, 0.0), 0),
        ((3, 16, 2, 50.0), 0),
        ((4, 8, 1, 50.0), 1),
    ]
    .iter()
    .map(|(fields, rank)| DirectRequest {
        preferred_dp_rank: Some(*rank),
        ..request(fields)
    })
    .collect();
    let two_ranks = ReplayEngineConfig {
        dp_size: 2,
        rank: engine(10),
        ..Default::default()
    };
    let report = replay_requests(two_ranks, requests).run().unwrap();
    assert_eq!(record(&report, 3).first_admit_ms, Some(210.0));
}

/// A prefills from 0 to 100 and decodes until 110 and 120; B (32 tokens)
/// arrives while the first decode runs, so the pass from 120 is eager.
const EAGER_AFTER_DECODE: [(u128, u32, usize, f64); 2] = [(1, 8, 3, 0.0), (2, 32, 1, 105.0)];

#[test]
fn a_pass_outputs_leave_once_the_next_pass_is_launched() {
    // vLLM decided the eager pass at 110, once the decode before it had
    // launched and the pass before that had ended, so it launches it until
    // 110 + 50. A's last token and its completion leave then, instead of at
    // 120. If the launch waits until the GPU has run 0.8 of that 100 ms pass,
    // they leave at 200. Passes do not move: B runs from 120 to 220.
    let plain = run(engine(64), &EAGER_AFTER_DECODE);
    let a = record(&plain, 1);
    assert_eq!(
        (a.first_token_ms, a.last_token_ms),
        (Some(100.0), Some(120.0))
    );
    for (fraction, returned_ms) in [(0.0, 160.0), (0.8, 200.0)] {
        let report = run(
            launch(engine(64), [2.0, 20.0, 50.0], fraction),
            &EAGER_AFTER_DECODE,
        );
        let a = record(&report, 1);
        assert_eq!(a.first_token_ms, Some(100.0));
        assert_eq!(a.last_token_ms, Some(returned_ms));
        assert_eq!(a.terminal_time_ms, returned_ms);
        let b = record(&report, 2);
        assert_eq!(
            (b.first_admit_ms, b.first_token_ms),
            (Some(120.0), Some(220.0))
        );
    }
}

#[test]
fn held_outputs_leave_before_those_of_the_pass_that_releases_them() {
    // A decodes until 120, and its last token comes from the pass that also
    // prefills B, from 120 to 230. vLLM decided that eager pass at 110, and
    // its 150 ms launch lasts until the pass ends, so A's third token, held
    // since 120, and its last token both leave at 230, in that order.
    let report = run(
        launch(engine(64), [2.0, 20.0, 150.0], 0.0),
        &[(1, 8, 4, 0.0), (2, 32, 1, 105.0)],
    );
    assert_eq!(record(&report, 1).last_token_ms, Some(230.0));
}

#[test]
fn each_worker_returns_its_held_outputs_at_its_own_launch() {
    // Round-robin placement gives worker 0 the requests of EAGER_AFTER_DECODE,
    // whose A leaves at 160, and worker 1 the same pair 3 ms later with an
    // 8-token B. Worker 1 decides that pass at 113 and launches it piecewise
    // for 20 ms, so its A leaves at 133: after worker 0 starts to hold its
    // A, at 120, and before worker 0 returns it.
    let requests = [
        (1, 8, 3, 0.0),
        (3, 8, 3, 3.0),
        (2, 32, 1, 105.0),
        (4, 8, 1, 108.0),
    ];
    let two_workers = ReplaySpec {
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: 2,
                ..WorkerPoolSpec::default()
            },
        },
        ..spec(ReplayEngineConfig {
            rank: launch(engine(64), [2.0, 20.0, 50.0], 0.0),
            ..Default::default()
        })
    };
    let report = Replayer::new(two_workers, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Requests(
            requests.iter().map(request).collect(),
        ))
        .run()
        .unwrap();
    assert_eq!(
        [1, 3].map(|id| record(&report, id).last_token_ms),
        [Some(160.0), Some(133.0)]
    );
}

#[test]
fn a_decode_pass_above_the_graph_limit_launches_eager() {
    // A, B and C prefill until 100, and A's last token comes from the decode
    // pass that ends at 110. With a one-token graph limit the two-token decode
    // pass after it runs eager: vLLM decided it at 100 and launches it for
    // 50 ms, so A's last token leaves when that pass ends at 120, not at 110.
    let mut config = launch(engine(64), [2.0, 20.0, 50.0], 0.0);
    config.launch.as_mut().unwrap().max_cudagraph_capture_size = 1;
    let report = run(config, &[(1, 3, 2, 0.0), (2, 3, 3, 0.0), (3, 3, 3, 0.0)]);
    assert_eq!(record(&report, 1).last_token_ms, Some(120.0));
}

#[test]
fn detokenization_starts_when_vllm_returns_the_outputs() {
    // A 1 ms detokenizer takes A's last token when vLLM returns it at 160.
    let launched = replayer(
        launch(engine(64), [2.0, 20.0, 50.0], 0.0),
        &EAGER_AFTER_DECODE,
    )
    .with_detokenization(1, 1.0, 0.0)
    .unwrap()
    .run()
    .unwrap();
    assert_eq!(record(&launched, 1).last_token_ms, Some(161.0));
}

#[test]
fn an_arrival_joins_the_first_pass_decided_after_it() {
    // A's prefill runs from 0 to 100 and vLLM decides the next pass once it
    // has launched it, at 20. B arrives before that and joins the next pass,
    // which prefills B and decodes A until 210; C arrives after it and waits
    // for the pass after. Counting passes, both would skip the pass vLLM had
    // already queued and join the one from 110.
    let requests = [(1, 8, 2, 0.0), (2, 8, 1, 10.0), (3, 8, 1, 30.0)];
    let admitted =
        |report: &ReplayReport| [2, 3].map(|id| record(report, id).first_admit_ms.unwrap());
    assert_eq!(admitted(&run(engine(64), &requests)), [110.0, 110.0]);
    assert_eq!(
        admitted(&run(launch(engine(64), [2.0, 20.0, 50.0], 0.0), &requests)),
        [100.0, 210.0]
    );
    // With no launch cost vLLM decides each pass when the one before it
    // starts, so the rule counts passes as before: B, arriving mid-pass at
    // 105, skips one pass; C, arriving as a pass ends at 110, skips two.
    let requests = [(1, 8, 10, 0.0), (2, 8, 1, 105.0), (3, 8, 1, 110.0)];
    assert_eq!(
        admitted(&run(launch(engine(64), [0.0; 3], 0.0), &requests)),
        admitted(&run(engine(64), &requests))
    );
}

#[test]
fn launch_needs_a_one_gpu_engine() {
    // With more GPUs vLLM's multiprocess executor returns a pass's outputs
    // independently of the next launch.
    let two_gpus = ReplayEngineConfig {
        tensor_parallel_size: 2,
        rank: launch(engine(64), [2.0, 20.0, 50.0], 0.0),
        ..Default::default()
    };
    let error = Replayer::new(spec(two_gpus), ReplayEngineFactory::new())
        .unwrap()
        .run()
        .unwrap_err();
    assert!(error.to_string().contains("uniprocess executor"), "{error}");
}

/// `launch` with prefix caching and a rank-local `offloading_v0_32` CPU tier
/// of 64 blocks: stores copy at once, and loads copy 16 B per token at
/// `h2d_gbps`.
fn polled_cpu_tier(num_gpu_blocks: usize, h2d_gbps: f64) -> EngineConfig {
    let mut config = launch(engine(num_gpu_blocks), [2.0, 20.0, 50.0], 0.0);
    config.enable_prefix_caching = true;
    config.kv_cache_bytes_per_token = Some(16);
    config.native_host_offload = Some(
        NativeHostOffloadConfig::new(64)
            .with_bandwidths(0.0, h2d_gbps)
            .with_connector(G2Connector::OffloadingV032),
    );
    config
}

/// A request with `(id, tokens, outputs, arrival)`.
fn tokens_request(id: u128, tokens: Vec<u32>, outputs: usize, arrival_ms: f64) -> DirectRequest {
    DirectRequest {
        tokens,
        max_output_tokens: outputs,
        uuid: Some(Uuid::from_u128(id)),
        arrival_timestamp_ms: Some(arrival_ms),
        ..Default::default()
    }
}

fn run_tokens(engine: EngineConfig, requests: Vec<DirectRequest>) -> ReplayReport {
    replay_requests(
        ReplayEngineConfig {
            rank: engine,
            ..Default::default()
        },
        requests.into(),
    )
    .run()
    .unwrap()
}

#[test]
fn a_load_awaiting_vllms_poll_does_not_stall_an_engine_with_only_a_blocked_waiter() {
    // Request 1 stores its five prompt blocks in the CPU tier, and request 2
    // fills G1 and evicts them. R (request 3) prefills from 600 and decodes
    // until 750. A (request 4) arrives at 605, joins R's first decode pass,
    // which vLLM decides at 620 and replay forms at 700, and loads request 1's
    // prefix: the copy waits for that pass's GPU work, until 710, and takes
    // 300 ms, so it ends at 1010. B (request 5, 24 tokens) does not fit beside
    // A's reservation in the 10-block G1. Once R finishes nothing runs, and
    // vLLM takes empty steps that poll the connector: it sees A's copy at 1010
    // and admits A then. A finishes at 1120, and B is admitted.
    let five_blocks_in_300_ms = 320.0 / (300.0 * 1e6);
    let report = run_tokens(
        polled_cpu_tier(10, five_blocks_in_300_ms),
        vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (500..536).collect(), 1, 300.0),
            tokens_request(3, (900..904).collect(), 6, 600.0),
            tokens_request(4, (0..24).collect(), 2, 605.0),
            tokens_request(5, (2000..2024).collect(), 2, 606.0),
        ],
    );
    let a = record(&report, 4);
    assert_eq!(
        (a.first_admit_ms, a.first_admission_host_reused_input_tokens),
        (Some(1010.0), Some(20))
    );
    assert_eq!(record(&report, 5).first_admit_ms, Some(1120.0));
}

#[test]
fn a_request_sharing_a_load_waits_for_it_until_vllm_has_seen_the_copy() {
    // As above with a 20-block G1, and five blocks copying in 50 ms: A
    // (request 4) loads request 1's prefix from the pass formed at 700, and
    // the copy runs from 710 to 760. C (request 5) shares that prefix and waits
    // for A's load. R's decode passes start every 10 ms, and each polls 8 ms
    // before it starts: it is decided at the previous pass's start and
    // launched in 2 ms. The pass starting at 770 polls at 762, the first poll
    // after the copy, so A is schedulable two decisions later, at 790. C waits
    // until then too and reuses A's blocks from G1 instead of copying the
    // prefix again.
    let five_blocks_in_50_ms = 320.0 / (50.0 * 1e6);
    let report = run_tokens(
        polled_cpu_tier(20, five_blocks_in_50_ms),
        vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (500..576).collect(), 1, 300.0),
            tokens_request(3, (900..904).collect(), 12, 600.0),
            tokens_request(4, (0..24).collect(), 2, 605.0),
            tokens_request(5, (0..20).chain(3000..3008).collect(), 2, 606.0),
        ],
    );
    let [a, c] = [4, 5].map(|id| record(&report, id));
    assert_eq!(
        (a.first_admit_ms, a.first_admission_host_reused_input_tokens),
        (Some(790.0), Some(20))
    );
    assert_eq!(
        (
            c.first_admit_ms,
            c.first_admission_reused_input_tokens,
            c.first_admission_host_reused_input_tokens
        ),
        (Some(790.0), 20, Some(0))
    );
}

#[test]
fn work_that_starts_before_an_idle_engine_sees_a_load_runs_before_it() {
    // The sharing run's G1 with five blocks copying in 8 ms, and R (request
    // 3) decoding only three tokens: its last decode runs from 710 to 720. A
    // (request 4) loads request 1's prefix from the pass formed at 700, and
    // the copy runs from 710 to 718. While R's last decode runs, vLLM takes
    // an empty step that polls at 710, before the copy ends, then waits for
    // R. With nothing else to run, it then steps empty and sees the copy:
    // replay admits A at 720. If N (request 6) arrives at 715, vLLM schedules
    // N at 720 instead; N's prefill launch polls at 740 and sees the copy, so
    // A is schedulable two decisions later, at 820, and its pass starts at
    // 830.
    let five_blocks_in_8_ms = 320.0 / (8.0 * 1e6);
    let run = |new_work: bool| {
        let mut requests = vec![
            tokens_request(1, (0..20).collect(), 1, 0.0),
            tokens_request(2, (500..576).collect(), 1, 300.0),
            tokens_request(3, (900..904).collect(), 3, 600.0),
            tokens_request(4, (0..24).collect(), 2, 605.0),
        ];
        if new_work {
            requests.push(tokens_request(6, (3000..3024).collect(), 2, 715.0));
        }
        let report = run_tokens(polled_cpu_tier(20, five_blocks_in_8_ms), requests);
        let admitted = |id| record(&report, id).first_admit_ms;
        (admitted(4), new_work.then(|| admitted(6)).flatten())
    };
    assert_eq!(run(false), (Some(720.0), None));
    assert_eq!(run(true), (Some(830.0), Some(720.0)));
}
