// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Modified from vLLM's `v1/kv_cache_layout.py`, pinned in THIRD_PARTY_NOTICES.md.

use serde_json::Value;

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

pub(super) fn from_strides(rank: &Value) -> Option<&'static str> {
    if rank["schema_version"] != "aisimulate-runtime-observation/v1" {
        return None;
    }
    let cache = &rank["cache"];
    let allocations = cache["tensor_allocations"].as_array()?;
    let tensors = cache["layer_tensors"].as_object()?;
    let mut candidates = LAYOUTS.to_vec();
    for allocation in allocations {
        let layers = allocation["layers"].as_array()?;
        if layers.is_empty() || allocation["host_resident"] != false {
            return None;
        }
        for name in layers {
            let view = tensors.get(name.as_str()?)?;
            let shape = view["shape"].as_array()?;
            let strides = view["stride_bytes"].as_array()?;
            if shape.len() != 4 || strides.len() != 4 || view["block_axis"] != 0 {
                return None;
            }
            let mut dims = [(layers.len() as u64, allocation["layer_stride"].as_u64()?); 5];
            for (i, (size, stride)) in shape.iter().zip(strides).enumerate() {
                dims[i + 1] = (size.as_u64()?, stride.as_u64()?);
            }
            if dims
                .iter()
                .any(|&(size, stride)| size == 0 || (size > 1 && stride == 0))
            {
                return None;
            }
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
        }
    }
    match candidates.as_slice() {
        [(name, _)] => Some(*name),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tensor_strides_identify_only_unique_physical_orders() {
        let capture = |shape: [u64; 4], strides: [u64; 4], layer_stride| {
            let view = json!({"shape":shape,"stride_bytes":strides,"block_axis":0});
            json!({"schema_version":"aisimulate-runtime-observation/v1", "cache":{
                "tensor_allocations":[{"layers":["a","b"],"host_resident":false,"layer_stride":layer_stride}],
                "layer_tensors":{"a":view,"b":view}
            }})
        };
        let shape = [100, 2, 16, 4];
        let compact = capture(shape, [256, 128, 8, 2], 25600);
        assert_eq!(from_strides(&compact), Some("LBHNC"));
        assert_eq!(
            from_strides(&capture(shape, [256, 8, 16, 2], 25600)),
            Some("LBNHC")
        );
        assert_eq!(
            from_strides(&capture(shape, [512, 8, 16, 2], 256)),
            Some("BLNHC")
        );
        assert_eq!(
            from_strides(&capture([100, 1, 16, 4], [128, 128, 8, 2], 12800)),
            None
        );
        let mut incomplete = compact.clone();
        incomplete["cache"]["layer_tensors"]
            .as_object_mut()
            .unwrap()
            .remove("b");
        assert_eq!(from_strides(&incomplete), None);
        let mut unrelated = compact;
        unrelated["schema_version"] = json!("other");
        assert_eq!(from_strides(&unrelated), None);
    }
}
