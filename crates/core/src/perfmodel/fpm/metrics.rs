// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forward-pass metrics telemetry types.
//!
//! These mirror Dynamo's scheduler `ForwardPassMetrics` v1 wire schema. AIC owns
//! this Rust copy so it does not depend on Dynamo crates. The forward-pass perf
//! model consumes the scheduled portion as estimator input and uses `wall_time`
//! as the tuning target.

use serde::{Deserialize, Serialize};

use crate::AicError;

/// Wire-schema version for [`ForwardPassMetrics`] (Dynamo's scheduler
/// telemetry). The FPM input is rejected with [`AicError::UnsupportedSchemaVersion`]
/// when `version` does not match.
pub const FPM_VERSION: u32 = 1;

/// Metrics for requests scheduled in one forward-pass iteration.
///
/// This mirrors Dynamo ForwardPassMetrics v1 scheduled request telemetry.
/// AIC owns this Rust copy so AIC does not depend on Dynamo crates.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ScheduledRequestMetrics {
    /// Exact prefill groups `[requests, new_tokens_per_request, prefix_tokens_per_request]`.
    /// Optional: op-level qualified Mamba uses these instead of balanced totals.
    /// Supplying groups also makes token-major ops use the full `sum_prefill_tokens`,
    /// including scheduled padding, while qualified Mamba uses only the prompt chunks.
    /// Together with explicit query padding, these must agree with prefill totals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_chunks: Option<Vec<[u32; 3]>>,
    /// Scheduled query positions beyond the actual prompt chunks (e.g. vLLM MTP
    /// padding). Included in `sum_prefill_tokens`, excluded from `prefill_chunks`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub sum_prefill_padding_tokens: u32,
    /// Optional regression features: freshly computed tokens for each scheduled
    /// request (one for ordinary decode), aligned with `past_kv_lengths`.
    /// These additive fields are unnecessary for the default attention/MoE fit.
    /// Their sums may differ from the backend's aggregate token counters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extend_lengths: Option<Vec<u64>>,
    /// Past KV lengths before this iteration, in the same request order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub past_kv_lengths: Option<Vec<u64>>,
    /// Number of prefill requests, including new requests and chunked-prefill
    /// continuations.
    #[serde(default)]
    pub num_prefill_requests: u32,
    /// Total tokens freshly computed for prefill requests in this iteration.
    #[serde(default)]
    pub sum_prefill_tokens: u32,
    /// Population variance of total prompt lengths across prefill requests.
    #[serde(default)]
    pub var_prefill_length: f64,
    /// Total KV tokens read for prefill requests, including prefix cache hits
    /// and previously computed chunks.
    #[serde(default)]
    pub sum_prefill_kv_tokens: u32,
    /// Number of decode requests.
    #[serde(default)]
    pub num_decode_requests: u32,
    /// Total KV context length across decode requests.
    #[serde(default)]
    pub sum_decode_kv_tokens: u32,
    /// Population variance of KV context lengths across decode requests.
    #[serde(default)]
    pub var_decode_kv_tokens: f64,
}

/// Metrics for requests queued but not scheduled in one iteration.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct QueuedRequestMetrics {
    #[serde(default)]
    pub num_prefill_requests: u32,
    #[serde(default)]
    pub sum_prefill_tokens: u32,
    #[serde(default)]
    pub var_prefill_length: f64,
    #[serde(default)]
    pub num_decode_requests: u32,
    #[serde(default)]
    pub sum_decode_kv_tokens: u32,
    #[serde(default)]
    pub var_decode_kv_tokens: f64,
}

/// Per-iteration forward-pass metrics.
///
/// In Dynamo this struct is telemetry emitted after an engine iteration. In
/// AIC the scheduled portion is also the estimator input. `wall_time` and
/// `queued_requests` are accepted for schema parity but ignored by the
/// latency estimator (the forward-pass perf model uses `wall_time` as the
/// tuning target, never as an estimation input).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForwardPassMetrics {
    #[serde(default = "default_fpm_version")]
    pub version: u32,
    #[serde(default)]
    pub worker_id: String,
    #[serde(default)]
    pub dp_rank: u32,
    #[serde(default)]
    pub counter_id: u64,
    #[serde(default)]
    pub wall_time: f64,
    #[serde(default)]
    pub scheduled_requests: ScheduledRequestMetrics,
    #[serde(default)]
    pub queued_requests: QueuedRequestMetrics,
}

impl Default for ForwardPassMetrics {
    fn default() -> Self {
        Self {
            version: FPM_VERSION,
            worker_id: String::new(),
            dp_rank: 0,
            counter_id: 0,
            wall_time: 0.0,
            scheduled_requests: ScheduledRequestMetrics::default(),
            queued_requests: QueuedRequestMetrics::default(),
        }
    }
}

/// Validate one [`ForwardPassMetrics`] entry. Rejects mismatched schema
/// versions and inconsistent scheduled-request sums (token sums without a
/// matching request count). Shared by the forward-pass perf model and
/// [`crate::perfmodel::engine::Engine::forward_pass_time_ms`].
pub(crate) fn validate_forward_pass_metrics(metrics: &ForwardPassMetrics) -> Result<(), AicError> {
    if metrics.version != FPM_VERSION {
        return Err(AicError::UnsupportedSchemaVersion {
            kind: "ForwardPassMetrics",
            got: metrics.version,
            expected: FPM_VERSION,
        });
    }
    let scheduled = &metrics.scheduled_requests;
    if scheduled.sum_prefill_padding_tokens > scheduled.sum_prefill_tokens
        || (scheduled.sum_prefill_padding_tokens > 0 && scheduled.prefill_chunks.is_none())
    {
        return Err(AicError::InvalidForwardPassMetrics(
            "prefill padding requires exact chunks and cannot exceed total prefill tokens".into(),
        ));
    }
    if scheduled.num_prefill_requests == 0
        && (scheduled.sum_prefill_tokens > 0 || scheduled.sum_prefill_kv_tokens > 0)
    {
        return Err(AicError::InvalidForwardPassMetrics(
            "prefill token sums require num_prefill_requests > 0".to_string(),
        ));
    }
    if scheduled.num_decode_requests == 0 && scheduled.sum_decode_kv_tokens > 0 {
        return Err(AicError::InvalidForwardPassMetrics(
            "decode KV token sum requires num_decode_requests > 0".to_string(),
        ));
    }
    if let Some(groups) = &scheduled.prefill_chunks {
        let mut totals = [0_u64; 3];
        for &[count, new, prefix] in groups {
            if count == 0 || new == 0 {
                return Err(AicError::InvalidForwardPassMetrics(
                    "prefill_chunks groups need positive request and new-token counts".into(),
                ));
            }
            for (total, value) in totals.iter_mut().zip([
                u64::from(count),
                u64::from(count) * u64::from(new),
                u64::from(count) * u64::from(prefix),
            ]) {
                *total = total.checked_add(value).ok_or_else(|| {
                    AicError::InvalidForwardPassMetrics("prefill_chunks totals overflow".into())
                })?;
            }
        }
        if totals
            != [
                u64::from(scheduled.num_prefill_requests),
                u64::from(scheduled.sum_prefill_tokens - scheduled.sum_prefill_padding_tokens),
                u64::from(scheduled.sum_prefill_kv_tokens),
            ]
        {
            return Err(AicError::InvalidForwardPassMetrics(
                "prefill_chunks must match prefill request, new-token and prefix totals".into(),
            ));
        }
    }
    match (&scheduled.extend_lengths, &scheduled.past_kv_lengths) {
        (None, None) => {}
        (Some(extend), Some(past)) => {
            let count = u64::from(scheduled.num_prefill_requests)
                + u64::from(scheduled.num_decode_requests);
            // Request-level lengths and aggregate counters can use different
            // backend conventions (for example padded prefill token counts).
            if extend.len() as u64 != count || past.len() as u64 != count {
                return Err(AicError::InvalidForwardPassMetrics(
                    "request feature lists must match scheduled request counts".into(),
                ));
            }
        }
        _ => {
            return Err(AicError::InvalidForwardPassMetrics(
                "extend_lengths and past_kv_lengths must be supplied together".into(),
            ));
        }
    }
    Ok(())
}

fn default_fpm_version() -> u32 {
    FPM_VERSION
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

#[cfg(test)]
mod padding_tests {
    use super::*;

    #[test]
    fn prefill_padding_keeps_exact_prompt_validation() {
        let mut metrics = ForwardPassMetrics {
            scheduled_requests: ScheduledRequestMetrics {
                num_prefill_requests: 1,
                sum_prefill_tokens: 4,
                sum_prefill_kv_tokens: 16,
                prefill_chunks: Some(vec![[1, 1, 16]]),
                sum_prefill_padding_tokens: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        validate_forward_pass_metrics(&metrics).unwrap();
        let restored = serde_json::from_str(&serde_json::to_string(&metrics).unwrap()).unwrap();
        assert_eq!(metrics, restored);
        for padding in [0, 2, 4, 5] {
            metrics.scheduled_requests.sum_prefill_padding_tokens = padding;
            assert!(validate_forward_pass_metrics(&metrics).is_err());
        }
        metrics.scheduled_requests.sum_prefill_padding_tokens = 3;
        metrics.scheduled_requests.prefill_chunks = None;
        assert!(validate_forward_pass_metrics(&metrics).is_err());
    }
}
