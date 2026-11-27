// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Modified from vLLM's `v1/kv_cache_layout.py`, pinned in THIRD_PARTY_NOTICES.md.

use super::{HeadGeometry, UNIFORM_SPEC, Worker, conflict, known};
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
    no_repack: bool,
) -> bool {
    let allowed = |w: &Worker| {
        let choices = std::array::from_fn(|gpu| transfer(w, gpu, version, no_repack));
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
fn transfer(worker: &Worker, gpu: usize, version: Option<&str>, no_repack: bool) -> [u8; 6] {
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
                    && !no_repack
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

pub(super) fn push_gpu(worker: &Worker, single_head_mla: bool) -> u8 {
    let layouts = possible_gpu(worker);
    if !single_head_mla {
        return layouts;
    }
    // The one latent KV head has no stride effect. Moving H preserves the order of L, B, N and C.
    const LAYER_FIRST: u8 = 0b000111;
    const BLOCK_FIRST: u8 = 0b111000;
    [LAYER_FIRST, BLOCK_FIRST]
        .into_iter()
        .filter(|class| class & layouts != 0)
        .fold(0, |all, class| all | class)
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
    no_repack: bool,
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
                    std::array::from_fn(|gpu| transfer(worker, gpu, version, no_repack))
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

// The 0.31 and 0.33 probes record a list of block-first views for each layer.
fn block_first_probe(rank: &Value) -> bool {
    rank.pointer("/runtime/version")
        .and_then(Value::as_str)
        .is_some_and(|v| matches!(v.split('+').next(), Some("0.31.0" | "0.33.0")))
}

fn layer_views(saved: &Value, block_first: bool) -> &[Value] {
    match saved.as_array() {
        Some(views) if block_first => views,
        _ => std::slice::from_ref(saved),
    }
}

fn on_gpu(cache: &Value, allocation: &Value, view: &Value, block_first: bool) -> bool {
    match allocation.get("host_resident") {
        Some(Value::Bool(host)) => !host,
        None if block_first => view["storage_id"].as_str().is_some_and(|id| {
            cache["storages"].as_array().is_some_and(|storages| {
                storages.iter().any(|s| {
                    s["storage_id"] == id
                        && s["device"]
                            .as_str()
                            .is_some_and(|d| d == "cuda" || d.starts_with("cuda:"))
                })
            })
        }),
        _ => false,
    }
}

fn view_span(dims: &[(u64, u64)], item_bytes: u64) -> Option<u128> {
    dims.iter()
        .try_fold(u128::from(item_bytes), |span, &(count, stride)| {
            span.checked_add(u128::from(count.checked_sub(1)?) * u128::from(stride))
        })
}

fn view_dimensions(
    cache: &Value,
    allocation: &Value,
    layers: usize,
    view: &Value,
    block_first: bool,
) -> Option<Result<[(u64, u64); 5], AicError>> {
    if !on_gpu(cache, allocation, view, block_first) {
        return None;
    }
    let shape = view["shape"].as_array()?;
    let strides = view["stride_bytes"].as_array()?;
    if shape.len() != 4 || strides.len() != 4 || view["block_axis"] != 0 {
        return None;
    }
    // A negative stride on a non-singleton axis cannot describe these GPU views.
    let stride = |v: &Value| v.as_u64().or_else(|| v.as_i64().map(|_| 0));
    let layer_stride = stride(&allocation["layer_stride"]).or((layers == 1).then_some(0))?;
    let mut dims = [(layers as u64, layer_stride); 5];
    for (i, (size, step)) in shape.iter().zip(strides).enumerate() {
        let Some(size) = size.as_u64().filter(|&n| n > 0) else {
            return Some(Err(conflict("kv_cache_layout tensor dimensions")));
        };
        dims[i + 1] = (size, stride(step).or((size == 1).then_some(0))?);
    }
    Some(Ok(dims))
}

fn shared_spec(group: &Value) -> Option<&str> {
    group["spec_type"]
        .as_str()
        .filter(|name| known(name) && *name != UNIFORM_SPEC)
}

fn concrete_spec<'a>(group: &'a Value, name: &str) -> Option<&'a str> {
    shared_spec(group).or_else(|| shared_spec(&group["layer_specs"][name]))
}

/// The hybrid collector records this marker from every concrete layer spec in the group.
pub(super) fn prefix_cacheable(group: &Value) -> Option<bool> {
    let specs = group["layer_specs"].as_object()?;
    let layers = group["layer_names"]
        .as_array()
        .filter(|names| !names.is_empty() && names.len() == specs.len())?;
    let mut cacheable = true;
    for layer in layers {
        match specs.get(layer.as_str()?)?["spec_type"]
            .as_str()?
            .strip_prefix("vllm.v1.kv_cache_interface.")?
        {
            "CircularBufferSpec" => cacheable = false,
            "FullAttentionSpec"
            | "MLAAttentionSpec"
            | "SinkFullAttentionSpec"
            | "HiddenStateCacheSpec"
            | "SlidingWindowSpec"
            | "MambaSpec" => {}
            _ => return None,
        }
    }
    Some(cacheable)
}

pub(super) fn records_one_head(group: &Value) -> bool {
    group["num_kv_heads"] == 1
        || group["layer_names"].as_array().is_some_and(|names| {
            !names.is_empty()
                && names.iter().all(|n| {
                    n.as_str()
                        .is_some_and(|n| group["layer_specs"][n]["num_kv_heads"] == 1)
                })
        })
}

pub(super) fn single_head_mla(group: &Value) -> bool {
    let mla = |spec: &Value| {
        matches!(
            spec["spec_type"].as_str(),
            Some(
                "vllm.v1.kv_cache_interface.MLAAttentionSpec"
                    | "vllm.v1.kv_cache_interface.SlidingWindowMLASpec"
            )
        ) && spec
            .get("num_kv_heads")
            .is_none_or(|heads| heads.is_null() || *heads == 1)
    };
    if group["spec_type"] == UNIFORM_SPEC {
        return group["num_kv_heads"]
            .as_u64()
            .is_none_or(|heads| heads == 1)
            && group["layer_names"].as_array().is_some_and(|names| {
                !names.is_empty()
                    && names
                        .iter()
                        .all(|name| name.as_str().is_some_and(|n| mla(&group["layer_specs"][n])))
            });
    }
    mla(group)
        && group["layer_specs"]
            .as_object()
            .is_none_or(|layers| layers.values().all(mla))
}

/// The raw uniform wrapper names its layers. The normalized observation supplies
/// their concrete specs; some instrumented releases record those specs only there.
pub(super) fn raw_mla_group(raw: &Value, normalized: &[Value]) -> bool {
    if raw["spec_type"] != UNIFORM_SPEC {
        return single_head_mla(raw);
    }
    let Some(names) = raw["layer_names"]
        .as_array()
        .filter(|names| !names.is_empty())
    else {
        return false;
    };
    normalized.iter().any(|g| {
        g["spec_type"] == UNIFORM_SPEC
            && single_head_mla(g)
            && raw["num_kv_heads"].as_u64().is_none_or(|heads| heads == 1)
            && g["layer_names"].as_array().is_some_and(|other| {
                names.len() == other.len() && names.iter().all(|name| other.contains(name))
            })
            && raw["layer_specs"].as_object().is_none_or(|specs| {
                specs.iter().all(|(name, spec)| {
                    let measured = &g["layer_specs"][name];
                    !measured.is_null()
                        && spec["spec_type"]
                            .as_str()
                            .is_none_or(|t| measured["spec_type"] == t)
                        && spec["num_kv_heads"].as_u64().is_none_or(|h| h == 1)
                })
            })
    })
}

pub(super) fn check_layer_specs(group: &Value) -> Result<(), AicError> {
    if let Some(shared) = shared_spec(group)
        && group["layer_specs"].as_object().is_some_and(|layers| {
            layers.values().any(|layer| {
                layer["spec_type"]
                    .as_str()
                    .is_some_and(|specific| known(specific) && specific != shared)
            })
        })
    {
        return Err(conflict("cache group spec_type"));
    }
    Ok(())
}

/// UniformTypeKVCacheSpecs records each layer's spec. Older probes can record a shared
/// attention spec only on the group, or name a stock decoder class without its spec.
fn ordinary_spec(spec: &str) -> bool {
    matches!(
        spec,
        "vllm.v1.kv_cache_interface.FullAttentionSpec"
            | "vllm.v1.kv_cache_interface.SlidingWindowSpec"
            | "vllm.v1.kv_cache_interface.SinkFullAttentionSpec"
    )
}

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
            let inner = group["layer_specs"][name]["spec_type"].as_str().filter(|name| known(name));
            let spec = shared_spec(group).or(inner).or_else(|| group["spec_type"].as_str());
            let ordinary = if group["kind"] == "convolution" {
                spec == Some("vllm.v1.kv_cache_interface.SlidingWindowSpec")
                    && group["layer_classes"][name] == "vllm.models.inkling.nvidia.sconv_swa_attn.InklingConvState"
            } else { match spec {
                Some(spec) => ordinary_spec(spec),
                None => matches!(
                    group["layer_classes"][name].as_str(),
                    Some(
                        "vllm.model_executor.layers.attention.Attention"
                            | "vllm.model_executor.layers.attention.attention.Attention"
                            | "vllm.model_executor.layers.attention.prefill_prefix_lm_attention.PrefillPrefixLMAttention"
                    )
                ),
            }};
            (name, ordinary)
        })
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub(super) struct LayerRegions {
    possible: BTreeSet<usize>,
    required: BTreeSet<usize>,
}

impl LayerRegions {
    pub(super) fn new(possible: BTreeSet<usize>, first_attention: Option<usize>) -> Self {
        // NIXL visits the native mixers in layer order and keeps the first name at each address.
        let required = possible
            .first()
            .copied()
            .filter(|i| first_attention.is_none_or(|a| *i < a))
            .into_iter()
            .collect();
        Self { possible, required }
    }

    pub(super) fn first_possible(&self) -> Option<usize> {
        self.possible.first().copied()
    }

    pub(super) fn exact(names: BTreeSet<usize>) -> Self {
        Self {
            required: names.clone(),
            possible: names,
        }
    }

    pub(super) fn merge(&mut self, other: &Self) {
        self.possible.retain(|name| other.possible.contains(name));
        self.required.extend(&other.required);
    }

    pub(super) fn missing(&self, peer: &Self) -> Option<usize> {
        self.required.difference(&peer.possible).next().copied()
    }
}

pub(super) fn nemotron_owners(names: &BTreeSet<String>, suffix: &str) -> Option<LayerRegions> {
    let mut indices = BTreeSet::new();
    for name in names {
        let (index, mixer) = name.strip_prefix("model.layers.")?.split_once('.')?;
        if !matches!(mixer, "mixer" | "mixer.attn") {
            return None;
        }
        if mixer == suffix {
            indices.insert(index.parse().ok()?);
        }
    }
    Some(LayerRegions::exact(indices))
}

/// A Mamba position is a possible region name. An aliased state can never replace an
/// earlier mixer at its address. The GPU probe retains the starts of its state views.
pub(super) fn nemotron_state_regions(
    rank: &Value,
    groups: &[Value],
    host: bool,
    owners: Option<&BTreeSet<String>>,
    observed_layers: &mut BTreeSet<usize>,
) -> Option<LayerRegions> {
    let mut names = BTreeMap::new();
    let mut first_attention = None;
    let mut complete = true;
    for g in groups {
        let state = match g["kind"].as_str() {
            Some("state") => true,
            Some("attention") => false,
            _ => {
                complete = false;
                continue;
            }
        };
        let Some(layers) = g["layer_names"].as_array() else {
            complete = false;
            continue;
        };
        for layer in layers {
            let suffix = if state { ".mixer" } else { ".mixer.attn" };
            let index = layer
                .as_str()
                .and_then(|n| n.strip_prefix("model.layers."))
                .and_then(|n| n.strip_suffix(suffix))
                .and_then(|n| n.parse::<usize>().ok());
            let (Some(index), Some(name)) = (index, layer.as_str()) else {
                complete = false;
                continue;
            };
            if state {
                if concrete_spec(g, name) == Some("vllm.v1.kv_cache_interface.MambaSpec") {
                    observed_layers.insert(index);
                    names.insert(index, name);
                } else {
                    complete = false;
                }
            } else {
                first_attention = Some(first_attention.map_or(index, |i: usize| i.min(index)));
            }
        }
    }
    if !complete || names.is_empty() {
        return None;
    }
    if let Some(exact) = owners.and_then(|n| nemotron_owners(n, "mixer")) {
        return Some(exact);
    }
    let mut regions = LayerRegions::new(names.keys().copied().collect(), first_attention);
    if host {
        return Some(regions);
    }
    let cache = &rank["cache"];
    let mut addresses = BTreeSet::new();
    for (&i, &name) in names
        .iter()
        .take_while(|(i, _)| first_attention.is_none_or(|a| **i < a))
    {
        let views = layer_views(&cache["layer_tensors"][name], block_first_probe(rank));
        let Some((id, offset)) = views.first().and_then(|v| {
            let id = v["storage_id"].as_str()?;
            let allocation = cache["tensor_allocations"]
                .as_array()?
                .iter()
                .find(|allocation| {
                    allocation["storage_id"] == id
                        && allocation["layers"]
                            .as_array()
                            .is_some_and(|ns| ns.iter().any(|n| *n == name))
                })?;
            let start = v["storage_offset_bytes"].as_u64()?;
            (on_gpu(cache, allocation, v, block_first_probe(rank))
                && views.iter().all(|v| {
                    v["storage_id"] == id
                        && v["storage_offset_bytes"]
                            .as_u64()
                            .is_some_and(|offset| offset >= start)
                }))
            .then_some((id, start))
        }) else {
            break; // Later aliases can refer to this unobserved address.
        };
        if addresses.insert((id, offset)) {
            regions.required.insert(i);
        } else {
            regions.possible.remove(&i);
        }
    }
    Some(regions)
}

/// For a block-outer allocation NIXL registers per-layer pointers only when each
/// tensor row is contiguous and has exactly one physical cache page.
fn contiguous_page(
    group: &Value,
    name: &str,
    view: &Value,
    axes: &[(u64, u64)],
    item: u64,
) -> Option<u128> {
    let pool_block = group["block_size_tokens"].as_u64()?;
    let kernel_block = view["kernel_block_size_tokens"].as_u64()?;
    if kernel_block == 0 || !pool_block.is_multiple_of(kernel_block) {
        return None;
    }
    let page = group["layer_specs"][name]["page_size_bytes"]
        .as_u64()
        .or_else(|| shared_spec(group).and(group["spec_page_size_bytes"].as_u64()))?;
    let bytes = axes.iter().rev().try_fold(item, |width, &(n, stride)| {
        (n > 0 && (n == 1 || stride == width))
            .then(|| width.checked_mul(n))
            .flatten()
    })?;
    (bytes > 0 && bytes.checked_mul(pool_block / kernel_block) == Some(page))
        .then_some(u128::from(bytes))
}

/// The tagged NIXL connector retains the first layer name at each registered
/// address. Numeric substrings sort as integers, including double-digit layers.
fn layer_order(left: &str, right: &str) -> std::cmp::Ordering {
    fn parts(name: &str) -> impl Iterator<Item = (bool, usize, &[u8])> {
        name.as_bytes()
            .chunk_by(|a, b| a.is_ascii_digit() == b.is_ascii_digit())
            .map(|part| {
                if part[0].is_ascii_digit() {
                    let digits = &part[part.iter().take_while(|&&b| b == b'0').count()..];
                    (false, digits.len(), digits)
                } else {
                    (true, 0, part)
                }
            })
    }
    parts(left).cmp(parts(right))
}

fn attention_address(
    group: &Value,
    name: &str,
    view: &Value,
    bytes: u64,
    blocks: u64,
) -> Option<u64> {
    let start = view["storage_offset_bytes"].as_u64()?;
    let shape = view["shape"].as_array()?;
    let steps = view["stride_bytes"].as_array()?;
    let item = view["element_size_bytes"].as_u64().filter(|&b| b > 0)?;
    if view["block_axis"] != 0 || shape.len() <= 1 || shape.len() != steps.len() {
        return None;
    }
    let dims = shape
        .iter()
        .zip(steps)
        .map(|(d, s)| Some((d.as_u64()?, s.as_u64()?)))
        .collect::<Option<Vec<_>>>()?;
    if u128::from(start).checked_add(view_span(&dims, item)?)? > u128::from(bytes) {
        return None;
    }
    let (num, stride) = dims[0];
    let kernel = view["kernel_block_size_tokens"]
        .as_u64()
        .filter(|&n| n > 0)?;
    let pool = group["block_size_tokens"].as_u64()?;
    if stride == 0 || !pool.is_multiple_of(kernel) || blocks.checked_mul(pool / kernel)? != num {
        return None;
    }
    let page = group["layer_specs"][name]["page_size_bytes"]
        .as_u64()
        .or_else(|| shared_spec(group).and(group["spec_page_size_bytes"].as_u64()))?;
    if page == 0 || pool == 0 || !page.is_multiple_of(pool / kernel) {
        return None;
    }
    let segment = num.checked_mul(stride)?;
    let block_major = bytes.checked_sub(segment).is_some_and(|tail| tail < stride);
    Some(if block_major {
        if contiguous_page(group, name, view, &dims[1..], item).is_some() {
            start
        } else {
            0
        }
    } else {
        // The connector can split a plane-major cache into several regions.
        // A single observed segment determines its sole registration address.
        if dims.iter().try_fold(item, |n, (d, _)| n.checked_mul(*d))? != segment {
            return None;
        }
        start
    })
}

fn gpu_region_owners(rank: &Value) -> Option<BTreeSet<String>> {
    let cache = &rank["cache"];
    let block_first = block_first_probe(rank);
    let blocks = cache["num_blocks"].as_u64().filter(|&n| n > 0)?;
    let mut groups = BTreeMap::new();
    for group in cache["groups"].as_array()? {
        for name in group["layer_names"].as_array()? {
            if groups.insert(name.as_str()?, group).is_some() {
                return None;
            }
        }
    }
    if groups.is_empty() || groups.len() != cache["layer_tensors"].as_object()?.len() {
        return None;
    }
    let count = groups.len();
    let has_state = groups.values().any(|g| g["kind"] == "state");
    let mut owners: BTreeMap<(&str, u64), &str> = BTreeMap::new();
    for allocation in cache["tensor_allocations"].as_array()? {
        let storage_id = allocation["storage_id"].as_str()?;
        let bytes = cache["storages"]
            .as_array()?
            .iter()
            .find(|s| s["storage_id"] == storage_id)?["size_bytes"]
            .as_u64()?;
        for layer in allocation["layers"].as_array()? {
            let name = layer.as_str()?;
            let group = groups.remove(name)?;
            let views = layer_views(&cache["layer_tensors"][name], block_first);
            let first = views.first()?;
            if first["storage_id"] != storage_id || !on_gpu(cache, allocation, first, block_first) {
                return None;
            }
            let start = first["storage_offset_bytes"].as_u64()?;
            let address = match group["kind"].as_str()? {
                "state"
                    if concrete_spec(group, name)? == "vllm.v1.kv_cache_interface.MambaSpec" =>
                {
                    // The observer retains the typed state views of the same raw cache.
                    if !views.iter().all(|v| {
                        v["storage_id"] == storage_id
                            && v["storage_offset_bytes"]
                                .as_u64()
                                .is_some_and(|n| n >= start && n < bytes)
                    }) {
                        return None;
                    }
                    start
                }
                "attention" if views.len() == 1 && ordinary_spec(concrete_spec(group, name)?) => {
                    attention_address(group, name, first, bytes, blocks)?
                }
                _ => return None,
            };
            let owner = owners.entry((storage_id, address)).or_insert(name);
            match layer_order(name, owner) {
                std::cmp::Ordering::Less => *owner = name,
                std::cmp::Ordering::Equal if name != *owner => return None,
                _ => {}
            }
        }
    }
    (groups.is_empty() && (has_state || owners.len() < count))
        .then(|| owners.into_values().map(str::to_owned).collect())
}

pub(super) fn registered_layers(rank: &Value) -> Option<BTreeSet<String>> {
    disjoint_layers(rank).or_else(|| {
        if rank
            .pointer("/resolved_config/kv_transfer_config/kv_buffer_device")
            .and_then(Value::as_str)
            == Some("cpu")
        {
            return None;
        }
        gpu_region_owners(rank)
    })
}

/// Disjoint ordinary cache pages register each layer under its own name. Older
/// probes can establish this even when the full shaped view is unavailable.
fn disjoint_layers(rank: &Value) -> Option<BTreeSet<String>> {
    let cache = &rank["cache"];
    let block_first = block_first_probe(rank);
    let groups = cache["groups"].as_array()?;
    let mut by_name = BTreeMap::new();
    for group in groups {
        let layers = group["layer_names"].as_array()?;
        if group["kind"] != "attention"
            || layers.is_empty()
            || !decoder_layers(group).all(|(_, ordinary)| ordinary)
        {
            return None;
        }
        for layer in layers {
            if by_name.insert(layer.as_str()?, group).is_some() {
                return None;
            }
        }
    }
    if cache["layer_tensors"].as_object()?.len() != by_name.len() {
        return None;
    }
    let blocks = cache["num_blocks"].as_u64().filter(|&n| n > 0)?;
    let mut placed = BTreeSet::new();
    let mut ranges = Vec::new();
    for allocation in cache["tensor_allocations"].as_array()? {
        let layers = allocation["layers"].as_array()?;
        let block_stride = allocation["block_stride"].as_u64().filter(|&n| n > 0)?;
        if layers.is_empty() {
            return None;
        }
        for layer in layers {
            let name = layer.as_str()?;
            let group = by_name.get(name)?;
            let [view] = layer_views(&cache["layer_tensors"][name], block_first) else {
                return None;
            };
            let storage = allocation["storage_id"].as_str()?;
            if view["block_axis"] != 0
                || view["storage_id"] != storage
                || !on_gpu(cache, allocation, view, block_first)
                || !placed.insert(name.to_owned())
            {
                return None;
            }
            let start = view["storage_offset_bytes"].as_u64()?;
            let stride = view["stride_bytes"][0].as_u64().filter(|&n| n > 0)?;
            let mut span = u128::from(blocks) * u128::from(block_stride);
            let mut page = None;
            if let Some(shape) = view["shape"].as_array() {
                let steps = view["stride_bytes"].as_array()?;
                let dims = shape
                    .iter()
                    .zip(steps)
                    .map(|(n, step)| Some((n.as_u64()?, step.as_u64()?)))
                    .collect::<Option<Vec<_>>>()?;
                let physical = u128::from(dims.first()?.0);
                // The probe records a whole pool-block stride on the allocation and kernel-block
                // strides on the view. The final logical block can have unused trailing slots.
                if shape.len() != steps.len()
                    || !block_stride.is_multiple_of(stride)
                    || physical * u128::from(stride) > span
                    || physical * u128::from(stride) <= span - u128::from(block_stride)
                {
                    return None;
                }
                // Head- or plane-major views can extend beyond the first run of blocks.
                let item = view["element_size_bytes"].as_u64()?;
                span = span.max(view_span(&dims, item)?);
                page = contiguous_page(group, name, view, &dims[1..], item)
                    .filter(|width| *width <= u128::from(stride));
            } else if block_first || stride != block_stride {
                return None;
            }
            ranges.push((
                storage,
                u128::from(start),
                u128::from(start).checked_add(span)?,
                u128::from(stride),
                page,
            ));
        }
    }
    ranges.sort_unstable();
    for (i, &(storage, start, end, stride, page)) in ranges.iter().enumerate() {
        for &(peer, next, _, step, width) in &ranges[i + 1..] {
            if peer != storage || next >= end {
                break;
            }
            // BL* can interleave per-layer pages inside every physical block.
            // Only equal strides and disjoint occupied bytes prove this layout.
            let delta = (next - start) % stride;
            if stride != step
                || !page
                    .zip(width)
                    .is_some_and(|(a, b)| a <= delta && b <= stride - delta)
            {
                return None;
            }
        }
    }
    (!by_name.is_empty() && by_name.len() == placed.len()).then_some(placed)
}

pub(super) fn from_strides(
    rank: &Value,
    heads: &mut BTreeMap<String, HeadGeometry>,
) -> Result<Option<Layouts>, AicError> {
    if rank["schema_version"] != "aisimulate-runtime-observation/v1" {
        return Ok(None);
    }
    let block_first = block_first_probe(rank);
    let cache = &rank["cache"];
    // A separate state allocation can have four axes with different meanings.
    let attention_layers = cache["groups"].as_array().map(|groups| {
        groups
            .iter()
            .filter(|g| matches!(g["kind"].as_str(), Some("attention" | "convolution")))
            .flat_map(|g| {
                decoder_layers(g)
                    .filter(|(_, ordinary)| *ordinary)
                    .map(move |(name, _)| (name, g["kind"] == "attention"))
            })
            .collect::<BTreeMap<_, _>>()
    });
    let recorded_specs: BTreeMap<_, _> = cache["groups"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|g| {
            decoder_layers(g).filter_map(move |(name, _)| Some((name, concrete_spec(g, name)?)))
        })
        .collect();
    // Only a decoder cache spec supplies logical KV geometry. An Inkling window or an MLA
    // indexer may carry similarly named dimensions for different data.
    for group in rank["raw_cache_config"]["groups"]
        .as_array()
        .into_iter()
        .flatten()
    {
        check_layer_specs(group)?;
        for (name, ordinary) in decoder_layers(group) {
            if let Some(raw) = concrete_spec(group, name)
                && recorded_specs.get(name).is_some_and(|saved| *saved != raw)
            {
                return Err(conflict("cache group spec_type"));
            }
            if !ordinary
                || attention_layers
                    .as_ref()
                    .and_then(|layers| layers.get(name))
                    != Some(&true)
            {
                continue;
            }
            for spec in shared_spec(group)
                .map(|_| group)
                .into_iter()
                .chain(group["layer_specs"].get(name))
            {
                let saved = HeadGeometry {
                    num_kv_heads: spec["num_kv_heads"].as_u64().filter(|v| *v > 0),
                    head_size: spec["head_size"].as_u64().filter(|v| *v > 0),
                    ..Default::default()
                };
                if !saved.is_empty() {
                    heads.entry(name.to_owned()).or_default().merge(&saved)?;
                }
            }
        }
    }
    let (Some(allocations), Some(tensors)) = (
        cache["tensor_allocations"].as_array(),
        cache["layer_tensors"].as_object(),
    ) else {
        return Ok(None);
    };
    let storage_sizes: BTreeMap<_, _> = cache["storages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|storage| {
            Some((
                storage["storage_id"].as_str()?,
                storage["size_bytes"].as_u64()?,
            ))
        })
        .collect();
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
                .is_some_and(|names| !names.contains_key(name))
            {
                continue;
            }
            let saved = tensors.get(name).unwrap_or(&Value::Null);
            let views = layer_views(saved, block_first);
            for view in views {
                let Some(dims) =
                    view_dimensions(cache, allocation, layers.len(), view, block_first)
                else {
                    // Opaque views contribute no constraint on the initialized layout.
                    continue;
                };
                let dims = dims?;
                if let (Some(size), Some(start), Some(item)) = (
                    view["storage_id"]
                        .as_str()
                        .and_then(|id| storage_sizes.get(id)),
                    view["storage_offset_bytes"].as_u64(),
                    view["element_size_bytes"].as_u64(),
                ) {
                    let end = view_span(&dims[1..], item)
                        .and_then(|span| u128::from(start).checked_add(span));
                    if end.is_none_or(|end| end > u128::from(*size)) {
                        return Err(conflict("kv_cache_layout tensor storage bounds"));
                    }
                }
                measured = true;
                // Inkling packs several streams into C. Only an ordinary attention view describes KV heads.
                if attention_layers
                    .as_ref()
                    .and_then(|layers| layers.get(name))
                    == Some(&true)
                {
                    heads
                        .entry(name.to_owned())
                        .or_default()
                        .merge(&HeadGeometry {
                            num_kv_heads: Some(dims[2].0),
                            // C is the packed storage width, which need not equal the logical head size.
                            content_size: (views.len() == 1).then_some(dims[4].0),
                            ..Default::default()
                        })?;
                }
                candidates.retain(|(_, order)| {
                    let mut span = u128::from(view["element_size_bytes"].as_u64().unwrap_or(1));
                    for &axis in order.iter().rev() {
                        let (size, stride) = dims[axis];
                        if size == 0 || (size > 1 && u128::from(stride) < span) {
                            return false;
                        }
                        if size > 1 {
                            span += u128::from(size - 1) * u128::from(stride);
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
    fn layer_names_require_complete_nonoverlapping_ordinary_buffers() {
        let view = |start| json!({"storage_id":"s0","storage_offset_bytes":start,"block_axis":0,"stride_bytes":[16]});
        let mut row = json!({"cache":{"num_blocks":2,
            "groups":[{"kind":"attention","layer_names":["a","b"],"spec_type":"vllm.v1.kv_cache_interface.FullAttentionSpec"}],
            "tensor_allocations":[{"storage_id":"s0","layers":["b","a"],"block_stride":16,"host_resident":false}],
            "layer_tensors":{"a":view(0),"b":view(32)}}});
        let names = BTreeSet::from(["a".to_owned(), "b".to_owned()]);
        assert_eq!(registered_layers(&row), Some(names));
        row["cache"]["layer_tensors"]["b"] = view(16);
        assert!(registered_layers(&row).is_none());
        row["cache"]["tensor_allocations"][0]["layers"] = json!(["a"]);
        assert!(registered_layers(&row).is_none());
    }

    #[test]
    fn registered_layers_follow_the_cuda_views_of_each_probe() {
        let names = BTreeSet::from(["a".to_owned(), "b".to_owned()]);
        for version in ["0.31.0", "0.32.0", "0.33.0"] {
            for per_block in [1, 4] {
                let pool_stride = 16_384 * per_block;
                let layer_bytes = 40 * pool_stride;
                let view = |start| {
                    let tensor = json!({"storage_id":"s0", "storage_offset_bytes":start, "block_axis":0,
                        "shape":[40 * per_block, 2, 16, 256],
                        "stride_bytes":[16_384, 8192, 512, 2],
                        "element_size_bytes":2, "kernel_block_size_tokens":16});
                    if version == "0.32.0" {
                        tensor
                    } else {
                        json!([tensor])
                    }
                };
                let mut rank = json!({"schema_version":"aisimulate-runtime-observation/v1",
                    "runtime":{"version":version}, "cache":{"num_blocks":40,
                        "groups":[{"kind":"attention", "layer_names":["a","b"], "block_size_tokens":16 * per_block,"spec_page_size_bytes":pool_stride,
                                   "spec_type":"vllm.v1.kv_cache_interface.FullAttentionSpec"}],
                        "storages":[{"storage_id":"s0", "size_bytes":2 * layer_bytes, "device":"cuda:0"}],
                        "tensor_allocations":[{"storage_id":"s0", "layers":["a","b"],
                            "offset":0, "size":2 * layer_bytes, "layer_stride":layer_bytes, "block_stride":pool_stride}],
                        "layer_tensors":{"a":view(0),"b":view(layer_bytes)}}});
                if version == "0.32.0" {
                    rank["cache"]["tensor_allocations"][0]["host_resident"] = json!(false);
                }
                assert_eq!(
                    registered_layers(&rank).as_ref(),
                    Some(&names),
                    "{version} {per_block}"
                );
                let mut interleaved = rank.clone();
                let al = &mut interleaved["cache"]["tensor_allocations"][0];
                al["block_stride"] = json!(2 * pool_stride);
                al["layer_stride"] = json!(16_384);
                for (i, name) in ["a", "b"].iter().enumerate() {
                    let layer = &mut interleaved["cache"]["layer_tensors"][name];
                    let tensor = if version == "0.32.0" {
                        layer
                    } else {
                        &mut layer[0]
                    };
                    tensor["stride_bytes"][0] = json!(32_768);
                    tensor["storage_offset_bytes"] = json!(i * 16_384);
                }
                assert_eq!(registered_layers(&interleaved).as_ref(), Some(&names));
                let layer = &mut interleaved["cache"]["layer_tensors"]["b"];
                let tensor = if version == "0.32.0" {
                    layer
                } else {
                    &mut layer[0]
                };
                tensor["storage_offset_bytes"] = json!(8_192);
                assert!(registered_layers(&interleaved).is_none());
                let layer = &mut interleaved["cache"]["layer_tensors"]["b"];
                let tensor = if version == "0.32.0" {
                    layer
                } else {
                    &mut layer[0]
                };
                tensor["storage_offset_bytes"] = json!(16_384);
                tensor["stride_bytes"] = json!([32_768, 512, 1024, 2]);
                assert_eq!(
                    registered_layers(&interleaved),
                    Some(BTreeSet::from(["a".to_owned()])),
                    "the shared row keeps its first registration"
                );
                rank["cache"]["layer_tensors"]["b"] = view(0);
                assert_eq!(
                    registered_layers(&rank),
                    Some(BTreeSet::from(["a".to_owned()])),
                    "aliased {version}"
                );
                rank["cache"]["layer_tensors"]["b"] = view(layer_bytes);
                let first = &mut rank["cache"]["layer_tensors"]["a"];
                let first = if version == "0.32.0" {
                    first
                } else {
                    &mut first[0]
                };
                first["stride_bytes"][1] = json!(layer_bytes);
                assert!(
                    registered_layers(&rank).is_none(),
                    "overlapping planes {version}"
                );
            }
        }
    }

    #[test]
    fn shared_attention_and_mamba_buffers_keep_the_first_name_at_each_address() {
        let attention = "model.layers.0.mixer.attn";
        let earlier = "model.layers.3.mixer";
        let later = "model.layers.11.mixer";
        let alias = "model.layers.4.mixer.attn";
        let state = |start| {
            json!([{"storage_id":"s", "storage_offset_bytes":start},
                                   {"storage_id":"s", "storage_offset_bytes":start + 4}])
        };
        let page = |start| {
            json!([{"storage_id":"s", "storage_offset_bytes":start,
            "shape":[4,1,1,8], "stride_bytes":[16,16,16,2], "block_axis":0,
            "element_size_bytes":2, "kernel_block_size_tokens":1}])
        };
        let mut rank = json!({"runtime":{"version":"0.33.0"}, "cache":{"num_blocks":4,
            "groups":[{"kind":"attention", "layer_names":[attention,alias], "block_size_tokens":1,
                "spec_type":"vllm.v1.kv_cache_interface.FullAttentionSpec", "spec_page_size_bytes":16},
                {"kind":"state", "layer_names":[earlier,later],
                "spec_type":"vllm.v1.kv_cache_interface.MambaSpec"}],
            "storages":[{"storage_id":"s", "size_bytes":128, "device":"cuda:0"}],
            "tensor_allocations":[{"layers":[later,alias,attention,earlier], "storage_id":"s"}],
            "layer_tensors":{(attention):page(0),(earlier):state(64),(later):state(64),(alias):page(64)}}});
        assert_eq!(
            registered_layers(&rank),
            Some(BTreeSet::from([attention.to_owned(), earlier.to_owned()]))
        );
        rank["cache"]["layer_tensors"][earlier] = state(0);
        assert_eq!(
            registered_layers(&rank),
            Some(BTreeSet::from([attention.to_owned(), alias.to_owned()]))
        );
        rank["cache"]["layer_tensors"][later][1]["storage_id"] = json!("unobserved");
        assert!(registered_layers(&rank).is_none());
    }

    #[test]
    fn ordinary_spec_and_view_must_agree_with_the_recorded_storage() {
        let full = "vllm.v1.kv_cache_interface.FullAttentionSpec";
        let mut rank = json!({"schema_version":"aisimulate-runtime-observation/v1",
            "raw_cache_config":{"groups":[{"layer_names":["a"],"spec_type":full}]},
            "cache":{"groups":[{"kind":"attention","layer_names":["a"],"spec_type":full}],
            "tensor_allocations":[{"layers":["a"],"host_resident":false}],
            "storages":[{"storage_id":"s","size_bytes":32}],
            "layer_tensors":{"a":{"storage_id":"s","storage_offset_bytes":8,"element_size_bytes":2,
                "block_axis":0,"shape":[2,1,2,2],"stride_bytes":[16,8,4,2]}}}});
        let project = |rank: &Value| from_strides(rank, &mut BTreeMap::new());
        assert!(project(&rank).is_ok());
        rank["cache"]["groups"][0]["spec_type"] =
            json!("vllm.v1.kv_cache_interface.MLAAttentionSpec");
        assert!(project(&rank).is_err());
        rank["cache"]["groups"][0]["spec_type"] = json!(full);
        rank["cache"]["layer_tensors"]["a"]["shape"][2] = json!(-2);
        assert!(project(&rank).is_err());
        rank["cache"]["layer_tensors"]["a"]["shape"][2] = json!(2);
        rank["cache"]["layer_tensors"]["a"]["stride_bytes"][0] = json!(32);
        assert!(project(&rank).is_err());
        rank["cache"].as_object_mut().unwrap().remove("storages");
        assert!(project(&rank).is_ok());
    }

    #[test]
    fn tensor_strides_preserve_every_feasible_physical_order() {
        let capture = |shape: [u64; 4], strides: [u64; 4], layer_stride| {
            let view = json!({"shape":shape,"stride_bytes":strides,"block_axis":0});
            json!({"schema_version":"aisimulate-runtime-observation/v1", "cache":{
                "tensor_allocations":[{"layers":["a","b"],"host_resident":false,"layer_stride":layer_stride}],
                "layer_tensors":{"a":view,"b":view}
            }})
        };
        let inferred = |rank: &Value| from_strides(rank, &mut BTreeMap::new()).unwrap();
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
        for invalid in [[256, 0, 8, 2], [256, 8, 8, 2], [256, 64, 8, 2]] {
            assert!(from_strides(&capture(shape, invalid, 25600), &mut BTreeMap::new()).is_err());
        }
        let one_head = capture([100, 1, 16, 4], [128, 0, 8, 2], 12800);
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
        for version in ["0.31.0", "0.33.0"] {
            block_first["runtime"]["version"] = json!(version);
            assert_eq!(
                inferred(&block_first).unwrap().names(),
                ["BHLNC", "BLHNC", "BLNHC"]
            );
        }
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
                let physical = from_strides(&rank, &mut BTreeMap::new());
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
