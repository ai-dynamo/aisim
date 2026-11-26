// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Modified from vLLM's `v1/kv_cache_layout.py`, pinned in THIRD_PARTY_NOTICES.md.

use super::{Worker, conflict};
use crate::AicError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

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

    pub(super) fn host(self, permute: Option<bool>, keep_choices: bool) -> Option<Self> {
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
        Self::from_names(names).filter(|v| keep_choices || matches!(v, Self::Named(_)))
    }
}

/// Each prefill worker has one immutable physical layout. A decoder must select one
/// local layout that can receive from every routable prefill worker with the same TP rank.
pub(super) fn routable(prefill: &[&Worker], decode: &[&Worker], version: Option<&str>) -> bool {
    const ALL: u8 = (1 << LAYOUTS.len()) - 1;
    let possible = |worker: &Worker| {
        worker.kv_cache_layout.as_ref().map_or(ALL, |layouts| {
            LAYOUTS.iter().enumerate().fold(0, |bits, (i, (name, _))| {
                bits | (u8::from(layouts.names().iter().any(|n| n == name)) << i)
            })
        })
    };
    let remote: Vec<_> = prefill.iter().map(|w| possible(w)).collect();
    let receivers: Vec<_> = decode
        .iter()
        .filter(|d| d.use_mla == Some(false))
        .map(|d| {
            let local = possible(d);
            LAYOUTS
                .iter()
                .enumerate()
                .filter(|(i, _)| local & (1 << i) != 0)
                .map(|(_, (name, _))| {
                    LAYOUTS.iter().enumerate().fold(0, |bits, (i, (peer, _))| {
                        bits | (u8::from(d.accepts_layout(peer, name, version)) << i)
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect();
    (1..=ALL).any(|set| {
        remote.iter().all(|mask| mask & set != 0)
            && receivers
                .iter()
                .all(|options| options.iter().any(|mask| set & !mask == 0))
    })
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

pub(super) fn from_strides(
    rank: &Value,
    current_format: bool,
) -> Result<Option<Layouts>, AicError> {
    if rank["schema_version"] != "aisimulate-runtime-observation/v1" {
        return Ok(None);
    }
    // v0.33 records every block-first view of a layer. Its backing storage identifies the device.
    let block_first = current_format
        && rank
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
    let attention_layers = cache["groups"]
        .as_array()
        .filter(|_| current_format)
        .map(|groups| {
            groups
                .iter()
                .filter(|g| g["kind"] == "attention")
                .filter_map(|g| g["layer_classes"].as_object())
                .flat_map(|classes| classes.keys().map(String::as_str))
                .collect::<BTreeSet<_>>()
        });
    let mut candidates = LAYOUTS.to_vec();
    let mut measured = false;
    for allocation in allocations {
        let layers = allocation["layers"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default();
        if layers.is_empty() && !current_format {
            return Ok(None);
        }
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
                    if current_format {
                        // Opaque views contribute no constraint on the worker's initialized layout.
                        continue;
                    }
                    return Ok(None); // Preserve the historical all-views projection.
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
                if current_format && candidates.is_empty() {
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
        let inferred = |rank: &Value, current| from_strides(rank, current).unwrap();
        let shape = [100, 2, 16, 4];
        let compact = capture(shape, [256, 128, 8, 2], 25600);
        assert_eq!(inferred(&compact, true), Layouts::named("LBHNC"));
        assert_eq!(
            inferred(&capture(shape, [256, 8, 16, 2], 25600), true),
            Layouts::named("LBNHC")
        );
        assert_eq!(
            inferred(&capture(shape, [512, 8, 16, 2], 256), true),
            Layouts::named("BLNHC")
        );
        let one_head = capture([100, 1, 16, 4], [128, 128, 8, 2], 12800);
        assert_eq!(
            inferred(&one_head, true).unwrap().names(),
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
            inferred(&block_first, true).unwrap().names(),
            ["BHLNC", "BLHNC", "BLNHC"]
        );
        assert_eq!(inferred(&block_first, false), None);
        block_first["cache"]["storages"][0]["device"] = json!("cpu");
        assert_eq!(inferred(&block_first, true), None);
        let mut empty = compact.clone();
        empty["cache"]["tensor_allocations"] = json!([]);
        assert_eq!(inferred(&empty, true), None);
        let mut incomplete = compact.clone();
        incomplete["cache"]["layer_tensors"]
            .as_object_mut()
            .unwrap()
            .remove("b");
        assert_eq!(inferred(&incomplete, true), Layouts::named("LBHNC"));
        assert_eq!(inferred(&incomplete, false), None);
        let mut unrelated = compact;
        unrelated["schema_version"] = json!("other");
        assert_eq!(inferred(&unrelated, true), None);
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
                        "groups":[{"kind":"attention","layer_classes":{"a":"Attention","b":"Attention"}},
                                  {"kind":"state","layer_classes":{"s":"State"}}],
                        "tensor_allocations":[{"layers":["a","b"],"host_resident":false,"layer_stride":1024},
                                              {"layers":["s"],"host_resident":false,"layer_stride":1024}],
                        "layer_tensors":{"a":view(a),"b":view(b),"s":view(&token_first)}
                    }
                });
                let physical = from_strides(&rank, true);
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
