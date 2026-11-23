// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builds the replay workload for one traffic configuration.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;

use super::loadgen::{
    AgenticProfileOptions, AgenticSnapshotOptions, AgenticWarmupPolicy, ArrivalSpec, DelaySpec,
    DynamoRequestTrace, LengthSpec, SyntheticImages, SyntheticTraceSpec, Trace,
    ValidatedAgenticGraph, WekaChatTemplate, WekaClient, WekaImportOptions,
    WekaNestedTimestampBasis, WekaResolvedTimestampBasis, WorkloadDriver, load_agentic_mooncake,
    load_weka_agentic_graph_cached,
};
use super::{ReplayRoleConfig, ReplayRuntimeInput};

/// One traffic configuration: where requests come from and how they arrive.
///
/// Built only by deserializing the traffic payload, so new fields do not break
/// callers in other crates.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrafficSpec {
    pub(crate) source_type: String,
    /// Configured deployment model used to time every agentic request. Source
    /// model labels remain provenance on the validated graph.
    #[serde(default)]
    pub(crate) execution_model: Option<String>,
    #[serde(default)]
    pub(crate) load_type: Option<String>,
    #[serde(default)]
    pub(crate) trace_path: Option<String>,
    #[serde(default)]
    pub(crate) trace_paths: Vec<String>,
    #[serde(default)]
    pub(crate) trace_format: Option<String>,
    #[serde(default)]
    pub(crate) trace_block_size: Option<usize>,
    #[serde(default)]
    pub(crate) weka_nested_timestamp_basis: Option<WekaNestedTimestampBasis>,
    #[serde(default)]
    pub(crate) weka_client: Option<WekaClient>,
    #[serde(default)]
    pub(crate) weka_chat_template: Option<WekaChatTemplate>,
    /// Keep the loaded Weka graph for later runs in this process (sweeps).
    #[serde(default)]
    pub(crate) trace_graph_cache: bool,
    #[serde(default)]
    pub(crate) arrival_speedup_ratio: Option<f64>,
    #[serde(default)]
    pub(crate) replay_concurrency: Option<usize>,
    #[serde(default)]
    pub(crate) agentic_lanes: Option<usize>,
    #[serde(default)]
    pub(crate) agentic_snapshot: Option<AgenticSnapshotOptions>,
    #[serde(default)]
    pub(crate) agentic_warmup: bool,
    #[serde(default)]
    pub(crate) agentic_warmup_policy: AgenticWarmupPolicy,
    #[serde(default)]
    pub(crate) agentic_profile: Option<AgenticProfileOptions>,
    #[serde(default)]
    pub(crate) isl: Option<usize>,
    #[serde(default)]
    pub(crate) osl: Option<usize>,
    #[serde(default)]
    pub(crate) cached_prefix_tokens: Option<usize>,
    // Fixed per-request image workload, already resolved from the image geometry.
    #[serde(default)]
    pub(crate) image_count: Option<usize>,
    #[serde(default)]
    pub(crate) image_visual_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) image_sequences: Option<u32>,
    #[serde(default)]
    pub(crate) image_patch_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) image_transformer_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) image_output_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) image_embedding_bytes: Option<u64>,
    #[serde(default)]
    pub(crate) image_identity_pool: Option<u64>,
    #[serde(default)]
    pub(crate) request_count: Option<usize>,
    #[serde(default)]
    pub(crate) turns_per_session: Option<usize>,
    #[serde(default)]
    pub(crate) shared_prefix_ratio: Option<f64>,
    #[serde(default)]
    pub(crate) num_prefix_groups: Option<usize>,
    #[serde(default)]
    pub(crate) inter_turn_delay_ms: Option<f64>,
    #[serde(default)]
    pub(crate) request_rate: Option<f64>,
    #[serde(default)]
    pub(crate) arrival_interval_ms: Option<f64>,
    #[serde(default)]
    pub(crate) arrival_seed: Option<u64>,
    #[serde(default)]
    pub(crate) concurrency: Option<usize>,
    // Relative load: a request count per unit of load, and a share of KV capacity.
    #[serde(default)]
    pub(crate) num_request_ratio: Option<f64>,
    #[serde(default)]
    pub(crate) kv_load_ratio: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) max_sim_time_ms: Option<f64>,
    /// For `synthetic-session`: whether a turn's prompt starts with the
    /// session's earlier turns. Defaults to true.
    #[serde(default)]
    pub(crate) accumulate_session_history: Option<bool>,
    /// Share of each `applied_compute_agentic` prompt drawn from a prefix
    /// shared within a prefix group. Defaults to 0.
    #[serde(default)]
    pub(crate) trace_shared_prefix_ratio: Option<f64>,
    /// Number of prefix groups for `trace_shared_prefix_ratio`.
    #[serde(default)]
    pub(crate) trace_num_prefix_groups: Option<usize>,
    /// Use at most one agentic lane per play in the trace. Requires
    /// `agentic_lanes` and does not combine with `agentic_snapshot`.
    #[serde(default)]
    pub(crate) cap_agentic_lanes_to_plays: bool,
    /// Divides synthetic arrival times and inter-turn delays. Defaults to 1.
    /// `arrival_speedup_ratio` stays trace-only: synthetic input has always
    /// accepted and ignored it.
    #[serde(default)]
    pub(crate) synthetic_arrival_speedup_ratio: Option<f64>,
}

/// Options the caller chooses when building a workload.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TrafficBuildOptions {
    /// Block size the engine hashes prompts at.
    pub engine_block_size: usize,
    /// Whether requests carry replay hashes. Callers that never route on
    /// prefix overlap can turn them off to skip the hashing. Without hashes a
    /// trace must also pass the driver's load-mode validation, such as a first
    /// arrival timestamp for every session under open-loop load.
    pub replay_hashes: bool,
}

impl TrafficBuildOptions {
    pub fn new(engine_block_size: usize) -> Self {
        Self {
            engine_block_size,
            replay_hashes: true,
        }
    }

    pub fn without_replay_hashes(mut self) -> Self {
        self.replay_hashes = false;
        self
    }
}

/// The workload built for one traffic configuration.
#[non_exhaustive]
pub struct BuiltTraffic {
    pub input: ReplayRuntimeInput,
    /// The timestamp basis a Weka trace resolved to, for Weka input.
    pub weka_nested_timestamp_basis: Option<WekaResolvedTimestampBasis>,
}

impl BuiltTraffic {
    fn without_weka_basis(input: ReplayRuntimeInput) -> Self {
        Self {
            input,
            weka_nested_timestamp_basis: None,
        }
    }
}

fn require_agentic_execution_model(traffic: &TrafficSpec) -> Result<&str> {
    traffic
        .execution_model
        .as_deref()
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .context("agentic execution requires a configured target model")
}

fn runtime_paths(traffic: &TrafficSpec) -> Result<Vec<PathBuf>> {
    let mut paths = traffic
        .trace_paths
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.is_empty()
        && let Some(path) = traffic.trace_path.as_deref()
    {
        paths.push(PathBuf::from(path));
    }
    ensure!(
        !paths.is_empty(),
        "trace traffic requires at least one path"
    );
    Ok(paths)
}

/// The largest trace block that divides both the cached prefix and the
/// engine's scheduler block (the engine block when there is no prefix).
/// Requests then share exactly the engine blocks inside the prefix, as with
/// one-token trace blocks, without one hash ID per token. Grouped prefixes are
/// sized in whole trace blocks, so they keep one-token blocks with a cached
/// prefix. The driver still hashes prompts at `engine_block_size`.
fn synthetic_trace_block_size(
    cached_prefix_tokens: usize,
    shared_prefix_ratio: f64,
    engine_block_size: usize,
) -> usize {
    if cached_prefix_tokens > 0 && shared_prefix_ratio > 0.0 {
        return 1;
    }
    let (mut a, mut b) = (engine_block_size, cached_prefix_tokens);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn synthetic_arrivals(traffic: &TrafficSpec) -> Result<ArrivalSpec> {
    match traffic.load_type.as_deref().unwrap_or("concurrency") {
        "concurrency" | "kv_capacity_fraction" => Ok(ArrivalSpec::Burst),
        "poisson" => {
            let qps = traffic
                .request_rate
                .context("poisson traffic requires request_rate")?;
            ensure!(
                qps.is_finite() && qps > 0.0,
                "request_rate must be positive"
            );
            Ok(ArrivalSpec::PoissonQps { qps })
        }
        "constant_rate" => {
            let qps = if let Some(qps) = traffic.request_rate {
                qps
            } else {
                let interval = traffic
                    .arrival_interval_ms
                    .context("constant_rate traffic requires an interval or rate")?;
                ensure!(
                    interval.is_finite() && interval > 0.0,
                    "arrival_interval_ms must be positive"
                );
                1_000.0 / interval
            };
            ensure!(
                qps.is_finite() && qps > 0.0,
                "request_rate must be positive"
            );
            Ok(ArrivalSpec::ConstantQps { qps })
        }
        other => Err(anyhow!("unsupported synthetic load type {other:?}")),
    }
}

fn concrete_session_count(traffic: &TrafficSpec) -> Result<usize> {
    if let Some(count) = traffic.request_count {
        ensure!(count > 0, "request_count must be positive");
        return Ok(count);
    }
    let ratio = traffic
        .num_request_ratio
        .context("synthetic traffic requires request_count or num_request_ratio")?;
    ensure!(
        ratio.is_finite() && ratio > 0.0,
        "num_request_ratio must be positive"
    );
    let load = traffic
        .concurrency
        .map(|value| value as f64)
        .or(traffic.request_rate)
        .or_else(|| {
            traffic
                .arrival_interval_ms
                .filter(|value| *value > 0.0)
                .map(|value| 1_000.0 / value)
        })
        .context("relative synthetic stop requires a concrete load")?;
    Ok(((ratio * load).round() as usize).max(1))
}

/// Per-request image workload declared by the traffic payload, if any.
fn synthetic_images(traffic: &TrafficSpec) -> Result<Option<SyntheticImages>> {
    let Some(count) = traffic.image_count else {
        return Ok(None);
    };
    Ok(Some(SyntheticImages {
        count,
        visual_tokens: traffic
            .image_visual_tokens
            .context("image traffic requires image_visual_tokens")?,
        encoder: crate::engine::EncoderShape {
            sequences: traffic
                .image_sequences
                .context("image traffic requires image_sequences")?,
            patch_tokens: traffic
                .image_patch_tokens
                .context("image traffic requires image_patch_tokens")?,
            transformer_tokens: traffic
                .image_transformer_tokens
                .context("image traffic requires image_transformer_tokens")?,
            output_tokens: traffic
                .image_output_tokens
                .context("image traffic requires image_output_tokens")?,
        },
        embedding_bytes: traffic
            .image_embedding_bytes
            .context("image traffic requires image_embedding_bytes")?,
        identity_pool: traffic.image_identity_pool,
    }))
}

/// For `kv_capacity_fraction` load, converts the configured fraction of the
/// role's KV capacity into a concurrency and stores it in `traffic`.
pub fn resolve_kv_capacity_concurrency(
    traffic: &mut TrafficSpec,
    role: &ReplayRoleConfig,
    replicas: usize,
) -> Result<Option<usize>> {
    if traffic.load_type.as_deref() != Some("kv_capacity_fraction") {
        return Ok(None);
    }
    let ratio = traffic
        .kv_load_ratio
        .as_ref()
        .and_then(serde_json::Value::as_f64)
        .context("kv_capacity_fraction requires one concrete ratio")?;
    ensure!(
        ratio.is_finite() && ratio > 0.0,
        "KV load ratio must be positive"
    );
    let isl = traffic.isl.context("KV load requires isl")?;
    let osl = traffic.osl.context("KV load requires osl")?;
    let visual_tokens = synthetic_images(traffic)?.map_or(Some(0), |images| {
        images.count.checked_mul(images.visual_tokens)
    });
    let expected_tokens = visual_tokens
        .and_then(|visual_tokens| isl.checked_add(visual_tokens))
        .and_then(|tokens| tokens.checked_add(osl / 2))
        .context("KV-load expected token count overflow")?;
    ensure!(
        expected_tokens > 0,
        "KV load requires positive token lengths"
    );
    let capacity = if !role.rank.kv_cache_groups.is_empty() {
        let request_bytes = role
            .rank
            .kv_cache_groups
            .iter()
            .try_fold(0u64, |sum, group| {
                sum.checked_add(group.peak_request_bytes(expected_tokens as u64, 1)?)
                    .context("grouped KV request bytes overflow")
            })?;
        let per_rank = role
            .rank
            .kv_cache_capacity_bytes
            .context("grouped KV load requires byte capacity")?
            / request_bytes;
        usize::try_from(per_rank)?
            .checked_mul(role.dp_size as usize)
            .and_then(|value| value.checked_mul(replicas))
            .context("aggregate grouped KV concurrency overflow")?
    } else {
        let per_rank_tokens = role
            .rank
            .num_gpu_blocks
            .checked_mul(role.rank.block_size)
            .context("KV capacity overflow")?;
        let total_tokens = per_rank_tokens
            .checked_mul(role.dp_size as usize)
            .and_then(|value| value.checked_mul(replicas))
            .context("aggregate KV capacity overflow")?;
        total_tokens / expected_tokens
    };
    ensure!(
        capacity > 0,
        "candidate KV capacity cannot hold one request"
    );
    let concurrency = ((ratio * capacity as f64) as usize).max(1);
    traffic.concurrency = Some(concurrency);
    Ok(Some(concurrency))
}

/// The driver for a trace of ordinary sessions: closed-loop when `concurrency`
/// is set, open-loop otherwise. With `accumulate`, a turn's prompt starts with
/// its session's earlier prompts and outputs.
fn trace_driver(
    trace: Trace,
    engine_block_size: usize,
    concurrency: Option<usize>,
    accumulate: bool,
    replay_hashes: bool,
) -> Result<WorkloadDriver> {
    match (concurrency, replay_hashes, accumulate) {
        (Some(cap), false, _) => WorkloadDriver::new_concurrency_without_replay_hashes(
            trace,
            engine_block_size,
            cap,
            accumulate,
        ),
        (None, false, _) => {
            WorkloadDriver::new_trace_without_replay_hashes(trace, engine_block_size, accumulate)
        }
        (Some(cap), true, true) => {
            WorkloadDriver::new_concurrency_accumulating_deltas(trace, engine_block_size, cap)
        }
        (Some(cap), true, false) => WorkloadDriver::new_concurrency(trace, engine_block_size, cap),
        (None, true, true) => {
            WorkloadDriver::new_trace_accumulating_deltas(trace, engine_block_size)
        }
        (None, true, false) => WorkloadDriver::new_trace(trace, engine_block_size),
    }
}

fn build_agentic_driver(
    graph: ValidatedAgenticGraph,
    traffic: &TrafficSpec,
    engine_block_size: usize,
    replay_hashes: bool,
    speedup: f64,
) -> Result<WorkloadDriver> {
    if let Some(options) = &traffic.agentic_snapshot {
        let lanes = traffic
            .agentic_lanes
            .context("agentic_snapshot requires positive agentic_lanes")?;
        // Sample recorded time before applying speedup to remaining timers.
        let prepared = graph.into_prepared_snapshots(lanes, options.clone())?;
        let mut driver = if traffic.agentic_warmup {
            WorkloadDriver::new_agentic_warmup_with_policy(
                prepared,
                engine_block_size,
                replay_hashes,
                speedup,
                traffic.agentic_warmup_policy,
                traffic
                    .agentic_profile
                    .as_ref()
                    .map_or(10_000.0, |profile| profile.global_idle_cap_seconds * 1000.0),
            )
        } else {
            WorkloadDriver::new_agentic_snapshots(
                prepared,
                engine_block_size,
                replay_hashes,
                speedup,
            )
        }?;
        if let Some(profile) = &traffic.agentic_profile {
            driver.enable_agentic_profile(profile.clone())?;
        }
        Ok(driver)
    } else {
        let lanes = match traffic.agentic_lanes {
            Some(lanes) if traffic.cap_agentic_lanes_to_plays => {
                Some(lanes.min(graph.play_count()))
            }
            lanes => lanes,
        };
        WorkloadDriver::new_agentic_trace_with_options(
            graph.normalize_starts().speed_up_timing(speedup)?,
            engine_block_size,
            replay_hashes,
            lanes,
        )
    }
}

/// Builds the replay input for `traffic`.
pub fn build_traffic(traffic: TrafficSpec, options: &TrafficBuildOptions) -> Result<BuiltTraffic> {
    let engine_block_size = options.engine_block_size;
    let replay_hashes = options.replay_hashes;
    ensure!(engine_block_size > 0, "engine block size must be positive");
    ensure!(
        !traffic.cap_agentic_lanes_to_plays
            || (traffic.agentic_lanes.is_some() && traffic.agentic_snapshot.is_none()),
        "cap_agentic_lanes_to_plays requires agentic_lanes without agentic_snapshot"
    );
    ensure!(
        traffic.agentic_warmup_policy != AgenticWarmupPolicy::LiveTrajectoryV1
            || (traffic.agentic_warmup && traffic.agentic_profile.is_some()),
        "live_trajectory_v1 requires agentic_warmup and agentic_profile"
    );
    ensure!(
        traffic.agentic_warmup_policy != AgenticWarmupPolicy::PrimersOnly || traffic.agentic_warmup,
        "primers_only requires agentic_warmup"
    );
    if let Some(profile) = &traffic.agentic_profile {
        profile.validate()?;
        ensure!(
            traffic.agentic_snapshot.is_some(),
            "agentic_profile requires agentic_snapshot"
        );
        ensure!(
            traffic.max_sim_time_ms.is_none(),
            "agentic_profile cannot be combined with max virtual time"
        );
    }
    ensure!(
        !traffic.agentic_warmup || traffic.agentic_snapshot.is_some(),
        "agentic_warmup requires agentic_snapshot"
    );
    if traffic.agentic_snapshot.is_some() {
        ensure!(
            traffic.source_type == "trace"
                && traffic.load_type.as_deref() == Some("trace_timestamps")
                && traffic.agentic_lanes.is_some_and(|lanes| lanes > 0)
                && traffic.replay_concurrency.is_none(),
            "agentic_snapshot requires trace_timestamps agentic input with positive agentic_lanes"
        );
    }
    ensure!(
        traffic.source_type == "trace" || traffic.agentic_lanes.is_none(),
        "agentic_lanes requires agentic trace input"
    );
    ensure!(
        traffic.source_type == "trace" || traffic.weka_nested_timestamp_basis.is_none(),
        "weka_nested_timestamp_basis requires Weka trace input"
    );
    ensure!(
        traffic.weka_client.is_none()
            || (traffic.source_type == "trace" && traffic.trace_format.as_deref() == Some("weka")),
        "weka_client requires Weka trace input"
    );
    ensure!(
        traffic.source_type == "trace" || traffic.weka_chat_template.is_none(),
        "weka_chat_template requires Weka trace input"
    );
    if traffic.source_type == "trace" {
        let paths = runtime_paths(&traffic)?;
        let trace_block_size = traffic.trace_block_size.unwrap_or(512);
        let format = traffic.trace_format.as_deref().unwrap_or("mooncake");
        let speedup = traffic.arrival_speedup_ratio.unwrap_or(1.0);
        ensure!(
            traffic.accumulate_session_history.is_none(),
            "accumulate_session_history requires synthetic-session input"
        );
        ensure!(
            traffic.synthetic_arrival_speedup_ratio.is_none(),
            "synthetic_arrival_speedup_ratio requires synthetic input"
        );
        ensure!(
            format == "applied_compute_agentic"
                || (traffic.trace_shared_prefix_ratio.is_none()
                    && traffic.trace_num_prefix_groups.is_none()),
            "trace_shared_prefix_ratio and trace_num_prefix_groups require \
             applied_compute_agentic input"
        );
        ensure!(
            format == "weka" || traffic.weka_nested_timestamp_basis.is_none(),
            "weka_nested_timestamp_basis requires Weka input"
        );
        ensure!(
            format == "weka" || traffic.weka_chat_template.is_none(),
            "weka_chat_template requires Weka input"
        );
        ensure!(
            traffic.agentic_lanes != Some(0),
            "agentic_lanes must be greater than 0"
        );
        if traffic.agentic_lanes.is_some() {
            ensure!(
                matches!(format, "weka" | "agentic_mooncake" | "dynamo"),
                "agentic_lanes requires weka, agentic_mooncake, or agentic Dynamo input"
            );
        }
        if format == "agentic_mooncake" {
            require_agentic_execution_model(&traffic)?;
            ensure!(
                traffic.load_type.as_deref() == Some("trace_timestamps"),
                "agentic_mooncake requires trace_timestamps load"
            );
            ensure!(
                traffic.max_sim_time_ms.is_none(),
                "agentic trace does not support max virtual time"
            );
            ensure!(
                paths.len() == 1,
                "agentic_mooncake requires exactly one path"
            );
            ensure!(
                traffic.replay_concurrency.is_none(),
                "agentic_mooncake does not support concurrency load"
            );
            let trace = load_agentic_mooncake(&paths[0], trace_block_size)?;
            return Ok(BuiltTraffic::without_weka_basis(
                ReplayRuntimeInput::Workload(build_agentic_driver(
                    trace,
                    &traffic,
                    engine_block_size,
                    replay_hashes,
                    speedup,
                )?),
            ));
        }
        if format == "weka" {
            require_agentic_execution_model(&traffic)?;
            ensure!(
                traffic.load_type.as_deref() == Some("trace_timestamps"),
                "weka requires trace_timestamps load"
            );
            ensure!(
                traffic.max_sim_time_ms.is_none(),
                "Weka agentic trace does not support max virtual time"
            );
            ensure!(paths.len() == 1, "weka requires exactly one path");
            ensure!(
                traffic.replay_concurrency.is_none(),
                "Weka agentic trace does not support concurrency load"
            );
            let requested_basis = traffic.weka_nested_timestamp_basis.unwrap_or_default();
            let (trace, resolved_basis, _cache_hit) = load_weka_agentic_graph_cached(
                &paths[0],
                traffic.trace_block_size,
                WekaImportOptions {
                    nested_timestamp_basis: requested_basis,
                    client: traffic.weka_client.unwrap_or_default(),
                    chat_template: traffic.weka_chat_template,
                },
                traffic.trace_graph_cache,
            )?;
            return Ok(BuiltTraffic {
                input: ReplayRuntimeInput::Workload(build_agentic_driver(
                    trace,
                    &traffic,
                    engine_block_size,
                    replay_hashes,
                    speedup,
                )?),
                weka_nested_timestamp_basis: Some(resolved_basis),
            });
        }
        if format == "dynamo" {
            let loaded =
                DynamoRequestTrace::from_request_trace_files(&paths, traffic.trace_block_size)?;
            let driver = match loaded {
                DynamoRequestTrace::Standard(trace) => {
                    ensure!(
                        traffic.agentic_lanes.is_none(),
                        "agentic_lanes requires an agentic Dynamo trace"
                    );
                    let trace = trace.normalize_session_starts()?.speed_up_timing(speedup)?;
                    trace_driver(
                        trace,
                        engine_block_size,
                        traffic.replay_concurrency,
                        false,
                        replay_hashes,
                    )?
                }
                DynamoRequestTrace::Agentic(trace) => {
                    require_agentic_execution_model(&traffic)?;
                    ensure!(
                        traffic.replay_concurrency.is_none(),
                        "agentic Dynamo trace does not support concurrency load"
                    );
                    ensure!(
                        traffic.max_sim_time_ms.is_none(),
                        "agentic Dynamo trace does not support max virtual time"
                    );
                    build_agentic_driver(
                        trace,
                        &traffic,
                        engine_block_size,
                        replay_hashes,
                        speedup,
                    )?
                }
            };
            return Ok(BuiltTraffic::without_weka_basis(
                ReplayRuntimeInput::Workload(driver),
            ));
        }
        ensure!(
            paths.len() == 1,
            "trace format {format:?} requires exactly one path"
        );
        let mut trace = match format {
            "mooncake" | "mooncake-delta" => Trace::from_mooncake(&paths[0], trace_block_size)?,
            "applied_compute_agentic" => Trace::from_applied_compute_agentic(
                &paths[0],
                trace_block_size,
                traffic.trace_shared_prefix_ratio.unwrap_or(0.0),
                traffic.trace_num_prefix_groups.unwrap_or(0),
            )?,
            other => return Err(anyhow!("unsupported trace format {other:?}")),
        };
        trace = trace.normalize_session_starts()?.speed_up_timing(speedup)?;
        let driver = trace_driver(
            trace,
            engine_block_size,
            traffic.replay_concurrency,
            format == "mooncake-delta",
            replay_hashes,
        )?;
        return Ok(BuiltTraffic::without_weka_basis(
            ReplayRuntimeInput::Workload(driver),
        ));
    }

    ensure!(
        matches!(
            traffic.source_type.as_str(),
            "synthetic" | "synthetic-session"
        ),
        "unsupported synthetic source type {:?}",
        traffic.source_type
    );
    let is_session = traffic.source_type == "synthetic-session";
    ensure!(
        is_session || traffic.accumulate_session_history.is_none(),
        "accumulate_session_history requires synthetic-session input"
    );
    ensure!(
        traffic.trace_shared_prefix_ratio.is_none() && traffic.trace_num_prefix_groups.is_none(),
        "trace_shared_prefix_ratio and trace_num_prefix_groups require \
         applied_compute_agentic input"
    );
    let sessions = concrete_session_count(&traffic)?;
    let turns = if is_session {
        traffic.turns_per_session.unwrap_or(4)
    } else {
        1
    };
    let cached_prefix_tokens = traffic.cached_prefix_tokens.unwrap_or(0);
    let images = synthetic_images(&traffic)?;
    let shared_prefix_ratio = traffic.shared_prefix_ratio.unwrap_or(0.0);
    let trace = Trace::synthetic(SyntheticTraceSpec {
        // Image placeholder spans stay token-exact on a one-token trace block; the
        // driver still hashes them at `engine_block_size` when it constructs
        // replay requests.
        block_size: if images.is_some() {
            1
        } else {
            synthetic_trace_block_size(cached_prefix_tokens, shared_prefix_ratio, engine_block_size)
        },
        num_sessions: sessions,
        turns_per_session: turns,
        input_tokens: LengthSpec {
            mean: traffic.isl.context("synthetic traffic requires isl")?,
            stddev: 0.0,
        },
        output_tokens: LengthSpec {
            mean: traffic.osl.context("synthetic traffic requires osl")?,
            stddev: 0.0,
        },
        cached_prefix_tokens,
        shared_prefix_ratio,
        num_prefix_groups: traffic.num_prefix_groups.unwrap_or(0),
        first_turn_arrivals: synthetic_arrivals(&traffic)?,
        inter_turn_delays: traffic
            .inter_turn_delay_ms
            .filter(|delay| *delay > 0.0)
            .map_or(DelaySpec::None, DelaySpec::ConstantMs),
        seed: 0,
        arrival_seed: traffic.arrival_seed.unwrap_or(42),
        images,
    })?;
    let trace = match traffic.synthetic_arrival_speedup_ratio {
        Some(ratio) => trace.speed_up_timing(ratio)?,
        None => trace,
    };
    let driver = trace_driver(
        trace,
        engine_block_size,
        traffic.concurrency,
        traffic.accumulate_session_history.unwrap_or(is_session),
        replay_hashes,
    )?;
    Ok(BuiltTraffic::without_weka_basis(
        ReplayRuntimeInput::Workload(driver),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_prefix_trace_blocks_share_the_same_engine_blocks_as_one_token_blocks() {
        let engine_block_size = 64;
        // Number each request's engine-block sequence hashes in first-seen
        // order, so traces with different token values compare by sharing.
        let shared_blocks = |cached_prefix_tokens, shared_prefix_ratio, block_size| {
            let trace = Trace::synthetic(SyntheticTraceSpec {
                block_size,
                num_sessions: 3,
                turns_per_session: 1,
                input_tokens: LengthSpec {
                    mean: 1000,
                    stddev: 0.0,
                },
                output_tokens: LengthSpec {
                    mean: 1,
                    stddev: 0.0,
                },
                cached_prefix_tokens,
                shared_prefix_ratio,
                // Three sessions in two groups put at least two in one group.
                num_prefix_groups: 2,
                first_turn_arrivals: ArrivalSpec::Burst,
                inter_turn_delays: DelaySpec::None,
                seed: 0,
                arrival_seed: 42,
                images: None,
            })
            .unwrap();
            let mut ids = std::collections::HashMap::new();
            trace
                .sessions
                .iter()
                .map(|session| {
                    let tokens = session.turns[0].synthesize_tokens(block_size).unwrap();
                    crate::replay::loadgen::ReplayRequestHashes::from_tokens(
                        &tokens,
                        engine_block_size as u32,
                    )
                    .sequence_hashes
                    .into_iter()
                    .map(|hash| {
                        let next = ids.len();
                        *ids.entry(hash).or_insert(next)
                    })
                    .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        for (cached_prefix_tokens, shared_prefix_ratio, expected_block_size) in
            [(512, 0.0, 64), (500, 0.0, 4), (0, 0.0, 64), (512, 0.72, 1)]
        {
            let block_size = synthetic_trace_block_size(
                cached_prefix_tokens,
                shared_prefix_ratio,
                engine_block_size,
            );
            assert_eq!(block_size, expected_block_size);
            assert_eq!(
                shared_blocks(cached_prefix_tokens, shared_prefix_ratio, block_size),
                shared_blocks(cached_prefix_tokens, shared_prefix_ratio, 1),
                "cached_prefix_tokens={cached_prefix_tokens} \
                 shared_prefix_ratio={shared_prefix_ratio}"
            );
        }
    }

    #[test]
    fn warmup_requires_a_boolean_and_seeded_snapshot_at_the_native_boundary() {
        let base = serde_json::json!({
            "source_type": "trace", "load_type": "trace_timestamps",
            "trace_path": "unused", "trace_format": "weka", "agentic_lanes": 1,
        });
        assert!(
            !serde_json::from_value::<TrafficSpec>(base.clone())
                .unwrap()
                .agentic_warmup
        );
        for invalid in [
            serde_json::Value::Null,
            serde_json::json!(0),
            serde_json::json!(1),
            serde_json::json!("true"),
            serde_json::json!({}),
        ] {
            let mut traffic = base.clone();
            traffic["agentic_warmup"] = invalid;
            assert!(serde_json::from_value::<TrafficSpec>(traffic).is_err());
        }
        let mut traffic = base;
        traffic["agentic_warmup"] = serde_json::json!(true);
        let traffic = serde_json::from_value::<TrafficSpec>(traffic).unwrap();
        assert!(
            build_traffic(traffic, &TrafficBuildOptions::new(64))
                .err()
                .unwrap()
                .to_string()
                .contains("agentic_warmup requires agentic_snapshot")
        );
    }

    #[test]
    fn snapshot_options_are_strict_at_the_native_json_boundary() {
        let base = serde_json::json!({
            "source_type": "trace", "load_type": "trace_timestamps",
            "trace_path": "unused", "trace_format": "weka", "agentic_lanes": 1,
        });
        for invalid in [
            serde_json::json!({}),
            serde_json::json!({"seed": true}),
            serde_json::json!({"seed": -1}),
            serde_json::json!({"seed": 1.0}),
            serde_json::json!({"seed": "42"}),
            serde_json::json!({"seed": 42, "extra": 0}),
        ] {
            let mut traffic = base.clone();
            traffic["agentic_snapshot"] = invalid;
            assert!(serde_json::from_value::<TrafficSpec>(traffic).is_err());
        }
        for (field, invalid) in [
            ("source_type", serde_json::json!("synthetic")),
            ("load_type", serde_json::json!("concurrency")),
            ("agentic_lanes", serde_json::json!(0)),
            ("agentic_lanes", serde_json::Value::Null),
            ("replay_concurrency", serde_json::json!(1)),
        ] {
            let mut traffic = base.clone();
            traffic["agentic_snapshot"] = serde_json::json!({"seed": u64::MAX});
            traffic[field] = invalid;
            let traffic = serde_json::from_value::<TrafficSpec>(traffic).unwrap();
            assert!(
                build_traffic(traffic, &TrafficBuildOptions::new(64))
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("agentic_snapshot requires")
            );
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Replayed {
        session: String,
        turn: usize,
        prompt: Vec<u32>,
        output_tokens: usize,
        ready_at_ms: f64,
        hashed: bool,
    }

    /// Every request the built workload issues, completing each as it is issued.
    fn try_replay(spec: serde_json::Value, options: TrafficBuildOptions) -> Result<Vec<Replayed>> {
        let spec: TrafficSpec = serde_json::from_value(spec)?;
        let ReplayRuntimeInput::Workload(mut driver) = build_traffic(spec, &options)?.input else {
            panic!("traffic builds a workload driver");
        };
        let mut replayed = Vec::new();
        while let Some(now_ms) = driver.next_ready_time_ms() {
            for ready in driver.pop_ready(now_ms, usize::MAX) {
                driver.on_complete(ready.request_uuid, now_ms)?;
                replayed.push(Replayed {
                    session: ready.session_id,
                    turn: ready.turn_index,
                    prompt: ready.request.tokens,
                    output_tokens: ready.request.max_output_tokens,
                    ready_at_ms: ready.scheduled_ready_at_ms,
                    hashed: ready.replay_hashes.is_some(),
                });
            }
        }
        Ok(replayed)
    }

    fn replay(spec: serde_json::Value) -> Vec<Replayed> {
        try_replay(spec, TrafficBuildOptions::new(64)).unwrap()
    }

    fn replay_error(spec: serde_json::Value) -> String {
        try_replay(spec, TrafficBuildOptions::new(64))
            .unwrap_err()
            .to_string()
    }

    fn prompt_lengths(replayed: &[Replayed]) -> Vec<usize> {
        replayed
            .iter()
            .map(|request| request.prompt.len())
            .collect()
    }

    fn write_trace(dir: &tempfile::TempDir, rows: &[&str]) -> String {
        let path = dir.path().join("trace.jsonl");
        std::fs::write(&path, rows.join("\n")).unwrap();
        path.to_str().unwrap().to_owned()
    }

    const MOONCAKE_SESSION: [&str; 2] = [
        r#"{"session_id":"s","timestamp":0,"input_length":16,"output_length":4,"hash_ids":[201]}"#,
        r#"{"session_id":"s","timestamp":10,"delay":5,"input_length":8,"output_length":4,"hash_ids":[202]}"#,
    ];

    const AGENTIC_PLAY: [&str; 3] = [
        r#"{"schema":"dynamo.agentic_mooncake","version":2,"block_size":4,"hash_id_scope":"local","source":{"format":"test","digest":"v1"}}"#,
        r#"{"request_id":"root","play_id":"play","session_id":"root","model":"m","input_length":4,"output_length":2,"hash_ids":[301],"not_before_ms":0,"dependencies":[]}"#,
        r#"{"request_id":"next","play_id":"play","session_id":"root","model":"m","input_length":4,"output_length":2,"hash_ids":[302],"not_before_ms":0,"dependencies":[{"request_id":"root","trigger":"completion","delay_ms":1,"relation":"sequence"}]}"#,
    ];

    fn agentic_traffic(trace: &str) -> serde_json::Value {
        serde_json::json!({"source_type": "trace", "load_type": "trace_timestamps",
            "trace_path": trace, "trace_format": "agentic_mooncake", "trace_block_size": 4,
            "execution_model": "m"})
    }

    #[test]
    fn replay_hashes_off_builds_the_same_requests_for_valid_traces() {
        let dir = tempfile::tempdir().unwrap();
        let trace = write_trace(&dir, &MOONCAKE_SESSION);
        // (traffic, whether a session's second turn carries its first)
        let cases = [
            (
                serde_json::json!({"source_type": "synthetic", "load_type": "concurrency",
                    "isl": 64, "osl": 8, "request_count": 3, "concurrency": 1}),
                false,
            ),
            (
                serde_json::json!({"source_type": "synthetic-session", "load_type": "poisson",
                    "request_rate": 2.0, "isl": 64, "osl": 8, "request_count": 2,
                    "turns_per_session": 3}),
                true,
            ),
            (
                serde_json::json!({"source_type": "trace", "load_type": "trace_timestamps",
                    "trace_path": trace, "trace_format": "mooncake", "trace_block_size": 512}),
                false,
            ),
            (
                serde_json::json!({"source_type": "trace", "load_type": "concurrency",
                    "trace_path": trace, "trace_format": "mooncake-delta",
                    "trace_block_size": 512, "replay_concurrency": 1}),
                true,
            ),
        ];
        for (spec, accumulates) in cases {
            let hashed = replay(spec.clone());
            let options = TrafficBuildOptions::new(64).without_replay_hashes();
            let mut plain = try_replay(spec.clone(), options).unwrap();
            assert!(hashed.iter().all(|request| request.hashed), "{spec}");
            assert!(plain.iter().all(|request| !request.hashed), "{spec}");
            plain.iter_mut().for_each(|request| request.hashed = true);
            assert_eq!(plain, hashed, "{spec}");
            let first = &plain[0];
            let grows = plain.iter().any(|request| {
                request.session == first.session
                    && request.turn == 1
                    && request.prompt.len() > first.prompt.len()
            });
            assert_eq!(grows, accumulates, "{spec}");
        }
    }

    #[test]
    fn replay_hashes_off_reaches_agentic_workloads() {
        let dir = tempfile::tempdir().unwrap();
        let spec = agentic_traffic(&write_trace(&dir, &AGENTIC_PLAY));
        let hashed = try_replay(spec.clone(), TrafficBuildOptions::new(4)).unwrap();
        let options = TrafficBuildOptions::new(4).without_replay_hashes();
        let plain = try_replay(spec, options).unwrap();
        assert_eq!(hashed.len(), 2);
        assert!(hashed.iter().all(|request| request.hashed));
        assert_eq!(plain.len(), 2);
        assert!(plain.iter().all(|request| !request.hashed));
    }

    #[test]
    fn session_history_can_be_turned_off_for_synthetic_sessions_only() {
        let mut spec = serde_json::json!({"source_type": "synthetic-session",
            "load_type": "concurrency", "isl": 64, "osl": 8, "request_count": 1,
            "turns_per_session": 3, "concurrency": 1});
        assert_eq!(prompt_lengths(&replay(spec.clone())), [64, 128, 192]);
        spec["accumulate_session_history"] = false.into();
        assert_eq!(prompt_lengths(&replay(spec.clone())), [64, 64, 64]);

        spec["source_type"] = "synthetic".into();
        assert!(
            replay_error(spec).contains("accumulate_session_history requires synthetic-session")
        );
        let dir = tempfile::tempdir().unwrap();
        let trace = write_trace(&dir, &MOONCAKE_SESSION);
        let spec = serde_json::json!({"source_type": "trace", "load_type": "trace_timestamps",
            "trace_path": trace, "trace_format": "mooncake-delta", "trace_block_size": 512,
            "accumulate_session_history": false});
        assert!(
            replay_error(spec).contains("accumulate_session_history requires synthetic-session")
        );
    }

    #[test]
    fn trace_prefix_sharing_applies_to_applied_compute_traces_only() {
        let dir = tempfile::tempdir().unwrap();
        let row = r#"{"num_turns":1,"input_prompt_length":1024,"assistant_response_length":[4],"tool_call_output_length":[8],"tool_call_latency":[0.01],"final_assistant_response_length":4}"#;
        let trace = write_trace(&dir, &[row, row]);
        let mut spec = serde_json::json!({"source_type": "trace", "load_type": "concurrency",
            "trace_path": trace, "trace_format": "applied_compute_agentic",
            "trace_block_size": 64, "replay_concurrency": 2});
        // Tokens the two sessions' first prompts have in common from the start.
        let shared = |spec: &serde_json::Value| {
            let replayed = replay(spec.clone());
            let firsts: Vec<_> = replayed
                .iter()
                .filter(|request| request.turn == 0)
                .collect();
            assert_eq!(firsts.len(), 2);
            firsts[0]
                .prompt
                .iter()
                .zip(&firsts[1].prompt)
                .take_while(|(left, right)| left == right)
                .count()
        };
        assert_eq!(shared(&spec), 0);
        spec["trace_shared_prefix_ratio"] = 0.5.into();
        spec["trace_num_prefix_groups"] = 1.into();
        assert_eq!(shared(&spec), 512);

        spec["trace_format"] = "mooncake".into();
        assert!(replay_error(spec).contains("require applied_compute_agentic"));
    }

    #[test]
    fn agentic_lanes_can_be_capped_to_the_trace_plays() {
        let dir = tempfile::tempdir().unwrap();
        let mut spec = agentic_traffic(&write_trace(&dir, &AGENTIC_PLAY));
        let uncapped = try_replay(spec.clone(), TrafficBuildOptions::new(4)).unwrap();
        spec["cap_agentic_lanes_to_plays"] = true.into();
        assert!(
            try_replay(spec.clone(), TrafficBuildOptions::new(4))
                .unwrap_err()
                .to_string()
                .contains("cap_agentic_lanes_to_plays requires agentic_lanes")
        );
        // One lane per play at most: this many lanes could not be allocated.
        spec["agentic_lanes"] = u64::MAX.into();
        let capped = try_replay(spec, TrafficBuildOptions::new(4)).unwrap();
        assert_eq!(capped.len(), uncapped.len());
    }

    #[test]
    fn synthetic_arrivals_can_be_sped_up() {
        let mut spec = serde_json::json!({"source_type": "synthetic", "load_type": "constant_rate",
            "request_rate": 1.0, "isl": 64, "osl": 8, "request_count": 3});
        let ready_at = |spec: &serde_json::Value| -> Vec<f64> {
            replay(spec.clone())
                .iter()
                .map(|request| request.ready_at_ms)
                .collect()
        };
        assert_eq!(ready_at(&spec), [0.0, 1000.0, 2000.0]);
        spec["synthetic_arrival_speedup_ratio"] = 2.0.into();
        assert_eq!(ready_at(&spec), [0.0, 500.0, 1000.0]);
    }
}
