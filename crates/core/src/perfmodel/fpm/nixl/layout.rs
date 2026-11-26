// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Modified from vLLM's `v1/kv_cache_layout.py`, pinned in THIRD_PARTY_NOTICES.md.

use super::{Worker, conflict};
use crate::AicError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

// Physical axes in descending stride order. The initialized layer view's logical axes are
// [blocks, heads, tokens, content], with the layer axis supplied by its allocation.
const LAYOUTS: [(&str, [usize; 5]); 6] = [
    ("LBHNC", [0, 1, 2, 3, 4]),
    ("LBNHC", [0, 1, 3, 2, 4]),
    ("LHBNC", [0, 2, 1, 3, 4]),
    ("BLHNC", [1, 0, 2, 3, 4]),
    ("BLNHC", [1, 0, 3, 2, 4]),
    ("BHLNC", [1, 2, 0, 3, 4]),
];

/// A compact scalar preserves previously recorded names. An array retains measured alternatives
/// when unit dimensions make more than one physical order possible.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(untagged)]
pub(super) enum Layouts {
    Named(String),
    Possible(Vec<String>),
}

impl Layouts {
    pub(super) fn named(name: &str) -> Option<Self> {
        LAYOUTS
            .iter()
            .any(|(n, _)| *n == name)
            .then(|| Self::Named(name.to_owned()))
    }

    pub(super) fn names(&self) -> &[String] {
        match self {
            Self::Named(name) => std::slice::from_ref(name),
            Self::Possible(names) => names,
        }
    }

    pub(super) fn from_names(mut names: Vec<String>) -> Option<Self> {
        names.sort_unstable();
        names.dedup();
        match names.len() {
            0 => None,
            1 => names.pop().map(Self::Named),
            _ => Some(Self::Possible(names)),
        }
    }

    pub(super) fn host(self, permute: Option<bool>) -> Option<Self> {
        if !self.names().iter().any(|n| n == "LBNHC") {
            return Some(self);
        }
        let mut names = self.names().to_vec();
        if permute != Some(false) {
            names.push("LBHNC".to_owned());
        }
        if permute == Some(true) {
            names.retain(|name| name != "LBNHC");
        }
        Self::from_names(names)
    }
}

// Indices into LAYOUTS. Only these two orders participate in the NIXL permutation.
const LBHNC: usize = 0;
const LBNHC: usize = 1;
const ALL: u8 = (1 << LAYOUTS.len()) - 1;

fn routes_fit(senders: &[u8], receivers: &[[u8; 6]]) -> bool {
    // NIXL's repack flag belongs to the decoder worker. Native and repacked inputs
    // cannot share that decoder. Only MLA accepts several remote layouts unchanged.
    (1..=ALL).any(|set| {
        senders.iter().all(|s| s & set != 0)
            && receivers.iter().all(|opts| {
                opts.iter()
                    .any(|&r| set & !r == 0 && (set.count_ones() == 1 || r == ALL))
            })
    })
}

fn announced(received: &[u8; 6]) -> u8 {
    received
        .iter()
        .enumerate()
        .fold(0, |bits, (i, &v)| bits | (u8::from(v != 0) << i))
}

// A partially observed worker may have a TP rank without a DP engine identifier.
pub(super) fn routable(
    prefill: &[&Worker],
    decode: &[&Worker],
    version: Option<&str>,
    hybrid_hma: bool,
) -> bool {
    let allowed = |w: &Worker| {
        let choices = std::array::from_fn(|gpu| transfer(w, gpu, version, hybrid_hma));
        receive_options(&choices, ALL)
    };
    let senders = prefill
        .iter()
        .map(|w| announced(&allowed(w)))
        .collect::<Vec<_>>();
    let receivers = decode.iter().map(|w| allowed(w)).collect::<Vec<_>>();
    routes_fit(&senders, &receivers)
}

fn mask(layout: Option<&Layouts>) -> u8 {
    layout.map_or(ALL, |values| {
        LAYOUTS.iter().enumerate().fold(0, |bits, (i, (name, _))| {
            bits | (u8::from(values.names().iter().any(|value| value == name)) << i)
        })
    })
}

/// For each announced layout, the set of remote layouts one decoder configuration can accept.
/// The host buffer changes LBNHC to LBHNC only for an ordinary attention worker with permutation enabled.
/// Older receipts retain only the announced layout, so either registration device can explain those observations.
fn transfer(worker: &Worker, gpu: usize, version: Option<&str>, hybrid_hma: bool) -> [u8; 6] {
    let mut allowed = [0; 6];
    let effective = mask(worker.kv_cache_layout.as_ref());
    if mask(worker.gpu_cache_layout.as_ref()) & (1 << gpu) == 0 {
        return allowed;
    }
    for host in [false, true]
        .into_iter()
        .filter(|v| worker.host_buffer.is_none_or(|x| x == *v))
    {
        for mla in [false, true]
            .into_iter()
            .filter(|v| worker.use_mla.is_none_or(|x| x == *v))
        {
            for permute in [false, true]
                .into_iter()
                .filter(|v| worker.permit_permute.is_none_or(|x| x == *v))
            {
                let local = if host && !mla && permute && gpu == LBNHC {
                    LBHNC
                } else {
                    gpu
                };
                if effective & (1 << local) == 0 {
                    continue;
                }
                let can_repack = permute
                    && !worker.requires_hma
                    && !hybrid_hma
                    && (local == LBNHC || version.is_none_or(|v| v == "0.31.0"));
                allowed[local] |= if mla {
                    ALL
                } else {
                    (1 << local) | (u8::from(can_repack) << LBHNC)
                };
            }
        }
    }
    allowed
}

fn receive_options(choices: &[[u8; 6]; 6], domain: u8) -> [u8; 6] {
    let mut received = [0; 6];
    for (_, local) in choices
        .iter()
        .enumerate()
        .filter(|(i, _)| domain & (1 << i) != 0)
    {
        for (to, options) in received.iter_mut().zip(local) {
            *to |= options;
        }
    }
    received
}

pub(super) fn possible_gpu(worker: &Worker) -> u8 {
    if worker.kv_cache_layout.is_none() {
        return mask(worker.gpu_cache_layout.as_ref());
    }
    (0..LAYOUTS.len()).fold(0, |bits, gpu| {
        bits | (u8::from(transfer(worker, gpu, None, false) != [0; 6]) << gpu)
    })
}

#[derive(Default)]
struct Engine {
    workers: BTreeMap<u64, Worker>,
    layouts: u8,
}

#[derive(Default)]
pub(crate) struct Deployment {
    engines: Vec<Engine>,
}

impl Deployment {
    pub(super) fn from_workers(
        workers: &[Worker],
        width: Option<u64>,
        tp: Option<u64>,
    ) -> Result<Self, AicError> {
        let mut engines = BTreeMap::<u64, Engine>::new();
        for worker in workers {
            let (Some(dp), Some(rank)) = (
                worker
                    .dp_rank
                    .or((tp.is_some() && tp == width).then_some(0)),
                worker.tp_rank.or((tp == Some(1)).then_some(0)),
            ) else {
                continue;
            };
            let entry = engines
                .entry(dp)
                .or_default()
                .workers
                .entry(rank)
                .or_default();
            entry.combine(worker)?;
        }
        for engine in engines.values_mut() {
            engine.layouts = engine
                .workers
                .values()
                .fold(ALL, |layouts, w| layouts & possible_gpu(w));
            if engine.layouts == 0 {
                return Err(conflict("kv_cache_layout"));
            }
        }
        Ok(Self {
            engines: engines.into_values().collect(),
        })
    }
}

/// A DP engine advertises one physical GPU layout across TP ranks. Registration and receive-side
/// conversion remain worker-local. Try a common physical choice per engine and enforce the
/// all-to-all DP routes separately at each TP rank under that same choice.
pub(super) fn engines_routable(
    prefill: &Deployment,
    decode: &Deployment,
    version: Option<&str>,
    hybrid_hma: bool,
) -> bool {
    let observed = |w: &Worker| w.kv_cache_layout.is_some() || w.gpu_cache_layout.is_some();
    if ![prefill, decode].iter().any(|r| {
        r.engines
            .iter()
            .flat_map(|e| e.workers.values())
            .any(observed)
    }) {
        return true;
    }
    let unknown = Engine {
        workers: BTreeMap::new(),
        layouts: ALL,
    };
    let mut engines: Vec<_> = prefill.engines.iter().collect();
    if engines.is_empty() {
        engines.push(&unknown);
    }
    let senders_count = engines.len();
    if decode.engines.is_empty() {
        engines.push(&unknown);
    } else {
        engines.extend(&decode.engines);
    }
    let ranks: BTreeSet<_> = engines
        .iter()
        .flat_map(|e| e.workers.keys())
        .copied()
        .collect();
    let unknown_worker = Worker::default();
    let mut domains = engines.iter().map(|e| e.layouts).collect::<Vec<_>>();
    // Cache the six hardware choices once, including the unknown choice at an unmeasured rank.
    let options: Vec<Vec<[[u8; 6]; 6]>> = ranks
        .into_iter()
        .map(|rank| {
            engines
                .iter()
                .map(|e| {
                    let worker = e.workers.get(&rank).unwrap_or(&unknown_worker);
                    std::array::from_fn(|gpu| transfer(worker, gpu, version, hybrid_hma))
                })
                .collect()
        })
        .collect();
    let fits = |domains: &[u8]| {
        options.iter().all(|choices| {
            let mut senders = Vec::new();
            let mut receivers = Vec::new();
            for (index, (gpus, domain)) in choices.iter().zip(domains).enumerate() {
                let received = receive_options(gpus, *domain);
                if index < senders_count {
                    senders.push(announced(&received));
                } else {
                    receivers.push(received);
                }
            }
            routes_fit(&senders, &receivers)
        })
    };
    // A single TP row has no second worker to constrain the same engine's physical layout.
    if options.len() == 1 {
        return fits(&domains);
    }
    fn choose(domains: &mut [u8], fits: &impl Fn(&[u8]) -> bool) -> bool {
        if !fits(domains) {
            return false;
        }
        let narrowest = domains
            .iter()
            .enumerate()
            .filter(|(_, d)| d.count_ones() > 1)
            .min_by_key(|(_, d)| d.count_ones());
        let Some((i, &choices)) = narrowest else {
            return true;
        };
        for gpu in 0..LAYOUTS.len() {
            if choices & (1 << gpu) != 0 {
                domains[i] = 1 << gpu;
                if choose(domains, fits) {
                    return true;
                }
            }
        }
        domains[i] = choices;
        false
    }
    // Measured GPU names and a common feasible first choice need no search.
    let first: Vec<_> = domains.iter().map(|d| d & d.wrapping_neg()).collect();
    fits(&first) || choose(&mut domains, &fits)
}

fn view_dimensions(
    cache: &Value,
    allocation: &Value,
    layers: usize,
    view: &Value,
    block_first: bool,
) -> Option<[(u64, u64); 5]> {
    match allocation.get("host_resident") {
        Some(Value::Bool(false)) => {}
        None if block_first => {
            let storage = view["storage_id"].as_str()?;
            if !cache["storages"].as_array()?.iter().any(|r| {
                r["storage_id"] == storage
                    && r["device"]
                        .as_str()
                        .is_some_and(|v| v == "cuda" || v.starts_with("cuda:"))
            }) {
                return None;
            }
        }
        _ => return None,
    }
    let shape = view["shape"].as_array()?;
    let strides = view["stride_bytes"].as_array()?;
    if shape.len() != 4 || strides.len() != 4 || view["block_axis"] != 0 {
        return None;
    }
    let mut dims = [(layers as u64, allocation["layer_stride"].as_u64()?); 5];
    for (i, (size, stride)) in shape.iter().zip(strides).enumerate() {
        dims[i + 1] = (size.as_u64()?, stride.as_u64()?);
    }
    dims.iter()
        .all(|&(size, stride)| size > 0 && (size == 1 || stride > 0))
        .then_some(dims)
}

/// UniformTypeKVCacheSpecs records each layer's spec. Older probes can record a shared
/// attention spec only on the group, or name a stock decoder class without its spec.
pub(super) fn decoder_layers(group: &Value) -> impl Iterator<Item = (&str, bool)> {
    let names = group["layer_names"].as_array();
    let legacy = group["layer_classes"]
        .as_object()
        .or_else(|| group["layer_specs"].as_object())
        .filter(|_| names.is_none());
    names
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .chain(
            legacy
                .into_iter()
                .flat_map(|layers| layers.keys().map(String::as_str)),
        )
        .map(move |name| {
            let spec = group["layer_specs"][name]["spec_type"]
                .as_str()
                .or_else(|| group["spec_type"].as_str());
            let ordinary = match spec {
                Some(spec) => matches!(
                    spec,
                    "vllm.v1.kv_cache_interface.FullAttentionSpec"
                        | "vllm.v1.kv_cache_interface.SlidingWindowSpec"
                        | "vllm.v1.kv_cache_interface.SinkFullAttentionSpec"
                ),
                None => matches!(
                    group["layer_classes"][name].as_str(),
                    Some(
                        "vllm.model_executor.layers.attention.Attention"
                            | "vllm.model_executor.layers.attention.attention.Attention"
                            | "vllm.model_executor.layers.attention.prefill_prefix_lm_attention.PrefillPrefixLMAttention"
                    )
                ),
            };
            (name, ordinary)
        })
}

pub(super) fn from_strides(rank: &Value) -> Result<Option<Layouts>, AicError> {
    if rank["schema_version"] != "aisimulate-runtime-observation/v1" {
        return Ok(None);
    }
    // v0.33 records every block-first view of a layer. Its backing storage identifies the device.
    let block_first = rank
        .pointer("/runtime/version")
        .and_then(Value::as_str)
        .is_some_and(|v| v.split('+').next() == Some("0.33.0"));
    let cache = &rank["cache"];
    let (Some(allocations), Some(tensors)) = (
        cache["tensor_allocations"].as_array(),
        cache["layer_tensors"].as_object(),
    ) else {
        return Ok(None);
    };
    // A separate state allocation can have four axes with different meanings.
    let attention_layers = cache["groups"].as_array().map(|groups| {
        groups
            .iter()
            .filter(|g| g["kind"] == "attention")
            .flat_map(decoder_layers)
            .filter_map(|(name, ordinary)| ordinary.then_some(name))
            .collect::<BTreeSet<_>>()
    });
    let mut candidates = LAYOUTS.to_vec();
    let mut measured = false;
    for allocation in allocations {
        let layers = allocation["layers"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default();
        for name in layers {
            let name = name.as_str().unwrap_or_default();
            if attention_layers
                .as_ref()
                .is_some_and(|names| !names.contains(name))
            {
                continue;
            }
            let saved = tensors.get(name).unwrap_or(&Value::Null);
            let views = match saved.as_array() {
                Some(views) if block_first => views.as_slice(),
                _ => std::slice::from_ref(saved),
            };
            for view in views {
                let Some(dims) =
                    view_dimensions(cache, allocation, layers.len(), view, block_first)
                else {
                    // Opaque views contribute no constraint on the initialized layout.
                    continue;
                };
                measured = true;
                candidates.retain(|(_, order)| {
                    let mut previous = None;
                    for &axis in order {
                        let (size, stride) = dims[axis];
                        if size > 1 {
                            if previous.is_some_and(|p| p <= stride) {
                                return false;
                            }
                            previous = Some(stride);
                        }
                    }
                    true
                });
                if candidates.is_empty() {
                    return Err(conflict("kv_cache_layout"));
                }
            }
        }
    }
    if !measured {
        return Ok(None);
    }
    Ok(Layouts::from_names(
        candidates
            .into_iter()
            .map(|(name, _)| name.to_owned())
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tensor_strides_preserve_every_feasible_physical_order() {
        let capture = |shape: [u64; 4], strides: [u64; 4], layer_stride| {
            let view = json!({"shape":shape,"stride_bytes":strides,"block_axis":0});
            json!({"schema_version":"aisimulate-runtime-observation/v1", "cache":{
                "tensor_allocations":[{"layers":["a","b"],"host_resident":false,"layer_stride":layer_stride}],
                "layer_tensors":{"a":view,"b":view}
            }})
        };
        let inferred = |rank: &Value| from_strides(rank).unwrap();
        let shape = [100, 2, 16, 4];
        let compact = capture(shape, [256, 128, 8, 2], 25600);
        assert_eq!(inferred(&compact), Layouts::named("LBHNC"));
        assert_eq!(
            inferred(&capture(shape, [256, 8, 16, 2], 25600)),
            Layouts::named("LBNHC")
        );
        assert_eq!(
            inferred(&capture(shape, [512, 8, 16, 2], 256)),
            Layouts::named("BLNHC")
        );
        let one_head = capture([100, 1, 16, 4], [128, 128, 8, 2], 12800);
        assert_eq!(
            inferred(&one_head).unwrap().names(),
            ["LBHNC", "LBNHC", "LHBNC"]
        );
        let mut block_first = capture([10, 1, 64, 32], [10240, 4096, 64, 2], 4096);
        block_first["runtime"] = json!({"version":"0.33.0"});
        let cache = &mut block_first["cache"];
        cache["tensor_allocations"][0]
            .as_object_mut()
            .unwrap()
            .remove("host_resident");
        cache["storages"] = json!([{"storage_id":"s0","device":"cuda:0"}]);
        for view in cache["layer_tensors"].as_object_mut().unwrap().values_mut() {
            view["storage_id"] = json!("s0");
            *view = json!([view]);
        }
        assert_eq!(
            inferred(&block_first).unwrap().names(),
            ["BHLNC", "BLHNC", "BLNHC"]
        );
        block_first["cache"]["storages"][0]["device"] = json!("cpu");
        assert_eq!(inferred(&block_first), None);
        let mut empty = compact.clone();
        empty["cache"]["tensor_allocations"] = json!([]);
        assert_eq!(inferred(&empty), None);
        let mut incomplete = compact.clone();
        incomplete["cache"]["layer_tensors"]
            .as_object_mut()
            .unwrap()
            .remove("b");
        assert_eq!(inferred(&incomplete), Layouts::named("LBHNC"));
        let mut unrelated = compact;
        unrelated["schema_version"] = json!("other");
        assert_eq!(inferred(&unrelated), None);
    }
    #[test]
    fn opaque_or_separate_state_views_do_not_erase_attention_layouts() {
        let head_first = json!({"shape":[10,2,16,4],"stride_bytes":[4096,128,8,2],"block_axis":0});
        let token_first = json!({"shape":[10,2,16,4],"stride_bytes":[4096,8,16,2],"block_axis":0});
        let opaque = json!({"shape":[10,512],"stride_bytes":[4096,2],"block_axis":0});
        for block_first in [false, true] {
            let view = |v: &Value| if block_first { json!([v]) } else { v.clone() };
            for (a, b, layout) in [
                (&head_first, &opaque, Some("BLHNC")),
                (&opaque, &head_first, Some("BLHNC")),
                (&token_first, &token_first, Some("BLNHC")),
                (&opaque, &opaque, None),
                (&head_first, &token_first, Some("conflict")),
            ] {
                let rank = json!({
                    "schema_version":"aisimulate-runtime-observation/v1",
                    "runtime":{"version": if block_first { "0.33.0" } else { "0.32.0" }},
                    "cache":{
                        "groups":[{"kind":"attention","layer_classes":{"a":"vllm.model_executor.layers.attention.Attention","b":"vllm.model_executor.layers.attention.Attention"}},
                                  {"kind":"state","layer_classes":{"s":"State"}}],
                        "tensor_allocations":[{"layers":["a","b"],"host_resident":false,"layer_stride":1024},
                                              {"layers":["s"],"host_resident":false,"layer_stride":1024}],
                        "layer_tensors":{"a":view(a),"b":view(b),"s":view(&token_first)}
                    }
                });
                let physical = from_strides(&rank);
                if layout == Some("conflict") {
                    assert!(
                        physical
                            .unwrap_err()
                            .to_string()
                            .contains("kv_cache_layout")
                    );
                } else {
                    assert_eq!(physical.unwrap(), layout.and_then(Layouts::named));
                }
            }
        }
    }
}
