// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! NIXL hashes the model converter's global geometry. Registered caches use the
//! attention constructor's rank-local geometry. See THIRD_PARTY_NOTICES.md for sources.

use std::{
    fs,
    io::{BufReader, Seek},
    path::Path,
    sync::LazyLock,
};

use quick_cache::sync::Cache;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{AicError, known, merge_observed, positive};

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub(super) struct HeadGeometry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) num_kv_heads: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) head_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) content_size: Option<u64>,
}

impl HeadGeometry {
    pub(super) fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    pub(super) fn mismatch(&self, peer: &Self) -> Option<(&'static str, String, String)> {
        [
            ("num_kv_heads", self.num_kv_heads, peer.num_kv_heads),
            ("head_size", self.head_size, peer.head_size),
            ("kv_cache_head_width", self.content_size, peer.content_size),
        ]
        .into_iter()
        .find_map(|(field, a, b)| match (a, b) {
            (Some(a), Some(b)) if a != b => Some((field, a.to_string(), b.to_string())),
            _ => None,
        })
    }

    pub(super) fn merge(&mut self, other: &Self) -> Result<(), AicError> {
        merge_observed(&mut self.num_kv_heads, other.num_kv_heads, "num_kv_heads")?;
        merge_observed(&mut self.head_size, other.head_size, "head_size")?;
        merge_observed(
            &mut self.content_size,
            other.content_size,
            "kv_cache_head_width",
        )
    }
}

/// An override contains only the changed fields. A missing higher-priority field
/// can still come from the checkpoint. A saved effective HF config is complete.
#[derive(Clone, Copy)]
pub(super) struct HfConfig<'a> {
    pub(super) fields: &'a Value,
    complete: bool,
    verified: bool,
}

/// Keep only configuration fields consumed by the tagged converters and native
/// constructors. An authenticated checkpoint supplies the baseline before overlays.
fn geometry_config(config: &Value) -> serde_json::Map<String, Value> {
    let mut selected = serde_json::Map::new();
    for key in [
        "model_type",
        "architectures",
        "n_head_kv",
        "num_kv_heads",
        "num_key_value_heads",
        "multi_query_group_num",
        "num_attention_groups",
        "num_attention_heads",
        "head_dim",
        "hidden_size_per_head",
        "hidden_size",
        "num_hidden_layers",
        "new_decoder_architecture",
        "multi_query",
        "attention_head_dim",
        "is_heterogeneous",
        "hybrid_override_pattern",
    ] {
        if let Some(value) = config.get(key) {
            selected.insert(key.to_owned(), value.clone());
        }
    }
    if let Some(fields) = config.get("attn_config") {
        selected.insert(
            "attn_config".to_owned(),
            serde_json::json!({"kv_n_heads":fields["kv_n_heads"]}),
        );
    }
    selected
}

fn apply_geometry(base: &mut Value, overrides: &Value) {
    if let Some(fields) = base.as_object_mut() {
        fields.extend(geometry_config(overrides));
    }
}

fn text_field(config: &Value) -> Option<&'static str> {
    // The tagged vLLM loader calls PreTrainedConfig.get_text_config().
    ["text_config", "decoder", "generator", "text_encoder"]
        .into_iter()
        .find(|&key| config[key].is_object())
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct CheckpointKey {
    path: std::path::PathBuf,
    digest: String,
    size: u64,
    modified: std::time::SystemTime,
    #[cfg(unix)]
    file_id: (u64, u64, i64, i64),
}

impl CheckpointKey {
    fn new(path: &Path, digest: &str, metadata: &fs::Metadata) -> Option<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Some(Self {
            path: path.to_owned(),
            digest: digest.to_owned(),
            size: metadata.len(),
            modified: metadata.modified().ok()?,
            #[cfg(unix)]
            file_id: (
                metadata.dev(),
                metadata.ino(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            ),
        })
    }
}

pub(super) fn checkpoint(model: &Value, source: &Value) -> Option<Value> {
    // An omitted override table is different from an explicitly empty one:
    // a historical writer may not have captured the serving overrides.
    let overrides = model.get("hf_overrides").filter(|v| v.is_object())?;
    if model["hf_config"].is_object() {
        return None;
    }
    let reference = &source["launch"]["model_config"];
    let digest = reference["sha256"]
        .as_str()
        .filter(|s| super::roster::sha256(s))?;
    let path = Path::new(reference["path"].as_str()?);
    let metadata = fs::metadata(path).ok()?;
    if !metadata.is_file() {
        return None;
    }
    // The same content-bound reference is used by many ranks and search candidates.
    // Keep only serving geometry, and revalidate when the referenced file changes.
    static SCOPES: LazyLock<Cache<CheckpointKey, Option<std::sync::Arc<Value>>>> =
        LazyLock::new(|| Cache::new(128));
    let key = CheckpointKey::new(path, digest, &metadata)?;
    let saved = SCOPES
        .get_or_insert_with(&key, || -> std::io::Result<_> {
            let mut options = fs::OpenOptions::new();
            options.read(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                // A previously regular path can be replaced with a FIFO before open.
                options.custom_flags(libc::O_NONBLOCK);
            }
            let mut file = options.open(path)?;
            let opened = file.metadata()?;
            if !opened.is_file() || CheckpointKey::new(path, digest, &opened).as_ref() != Some(&key)
            {
                return Ok(None);
            }
            let mut hash = Sha256::new();
            std::io::copy(&mut file, &mut hash)?;
            if format!("{:x}", hash.finalize()) != digest {
                return Ok(None);
            }
            file.rewind()?;
            let config = serde_json::from_reader::<_, Value>(BufReader::new(&file)).ok();
            // Keep the scope only if the same descriptor still identifies the hashed file.
            if CheckpointKey::new(path, digest, &file.metadata()?).as_ref() != Some(&key) {
                return Ok(None);
            }
            Ok(config.filter(Value::is_object).map(|config| {
                let mut fields = Value::Object(geometry_config(&config));
                if let Some(key) = text_field(&config) {
                    fields[key] = Value::Object(geometry_config(&config[key]));
                }
                std::sync::Arc::new(fields)
            }))
        })
        .ok()
        .flatten()?;
    let mut effective = (*saved).clone();
    if let Some(key) = text_field(&saved).or_else(|| text_field(overrides)) {
        if effective[key].is_null() {
            effective[key] = serde_json::json!({});
        }
        apply_geometry(&mut effective[key], &overrides[key]);
    }
    apply_geometry(&mut effective, overrides);
    Some(effective)
}

impl<'a> HfConfig<'a> {
    pub(super) fn observed(
        model: &'a Value,
        checkpoint: Option<&'a Value>,
        referenced: bool,
    ) -> (Self, Self) {
        let saved = model.get("hf_config").filter(|v| v.is_object());
        let top = Self {
            fields: saved.or(checkpoint).unwrap_or(&model["hf_overrides"]),
            complete: saved.is_some(),
            verified: checkpoint.is_some(),
        };
        let nested = text_field(top.fields).or_else(|| {
            // Qwen2-VL creates its text sub-config even for older flat checkpoint files.
            (top.fields["model_type"] == "qwen2_vl"
                || top.fields["architectures"][0] == "Qwen2VLForConditionalGeneration")
                .then_some("text_config")
        });
        let text = if let Some(fields) = model.get("hf_text_config").filter(|v| v.is_object()) {
            Self {
                fields,
                complete: true,
                verified: false,
            }
        } else if let Some(key) = nested {
            Self {
                fields: &top.fields[key],
                complete: top.complete,
                verified: top.verified,
            }
        } else if referenced
            && saved.is_none()
            && checkpoint.is_none()
            && top.fields["model_type"].as_str().is_none()
        {
            // A missing archive cannot identify whether an outer override reaches the text model.
            Self {
                fields: &Value::Null,
                complete: false,
                verified: false,
            }
        } else {
            top
        };
        (top, text)
    }

    fn divided_head_size(self) -> Option<u64> {
        positive(&self.fields["hidden_size"])?
            .checked_div(positive(&self.fields["num_attention_heads"])?)
            .filter(|&size| size > 0)
    }

    fn base_kv_heads(self) -> Option<u64> {
        for key in [
            "n_head_kv",
            "num_kv_heads",
            "num_key_value_heads",
            "multi_query_group_num",
            "num_attention_groups",
        ] {
            match self.fields.get(key) {
                // getattr_iter returns the first present attribute, including None.
                Some(value) => return value.as_u64(),
                // These ordinary HF configs do not manufacture either higher-priority alias.
                None if self.verified
                    && matches!(key, "n_head_kv" | "num_kv_heads")
                    && matches!(
                        self.fields["model_type"].as_str(),
                        Some(
                            "qwen2"
                                | "qwen2_vl_text"
                                | "qwen3"
                                | "llama"
                                | "mistral"
                                | "nemotron_h"
                        )
                    ) => {}
                None if !self.complete => return None,
                None => {}
            }
        }
        self.fields["num_attention_heads"].as_u64()
    }

    fn base_head_size(self) -> Option<u64> {
        if let Some(size) = positive(&self.fields["head_dim"]) {
            return Some(size);
        }
        let qwen2_config = self.verified && self.fields["model_type"] == "qwen2";
        if !self.complete && !qwen2_config && self.fields.get("head_dim").is_none() {
            return None;
        }
        match self.fields.get("hidden_size_per_head") {
            Some(Value::Null) => self.divided_head_size(),
            Some(value) => value.as_u64(),
            None if self.complete || qwen2_config => self.divided_head_size(),
            None => None,
        }
    }

    fn flag(self, name: &str) -> Option<bool> {
        self.fields[name]
            .as_bool()
            .or_else(|| (self.complete && self.fields.get(name).is_none()).then_some(false))
    }

    pub(super) fn model_geometry(
        self,
        family: Option<&str>,
        use_mla: Option<bool>,
        resolved: &Value,
        top: Self,
    ) -> (HeadGeometry, Option<u64>) {
        let mut heads = HeadGeometry::default();
        let Some(family) = family.filter(|s| known(s)) else {
            return (heads, None);
        };
        // These converters can obtain their geometry from another stack, individual
        // layers or a draft backbone. Their emitted model_arch_config is authoritative.
        if family.ends_with("_mtp")
            || matches!(
                family,
                "cohere_asr"
                    | "falcon_mamba"
                    | "mamba"
                    | "medusa"
                    | "moss_audio"
                    | "timm_wrapper"
                    | "gemma4"
                    | "gemma4_text"
                    | "gemma4_unified"
                    | "gemma4_unified_text"
                    | "diffusion_gemma"
                    | "diffusion_gemma_text"
            )
            || [top.fields, self.fields]
                .iter()
                .any(|c| c["is_heterogeneous"] == true)
        {
            return (heads, None);
        }
        let layers = (family != "longcat_flash")
            .then(|| positive(&self.fields["num_hidden_layers"]))
            .flatten();
        heads.num_kv_heads = match family {
            "falcon" | "RefinedWeb" | "RefinedWebModel" => {
                match (
                    self.flag("new_decoder_architecture"),
                    self.flag("multi_query"),
                ) {
                    (Some(false), Some(true)) => Some(1),
                    (Some(true), _) | (_, Some(false)) => self.base_kv_heads(),
                    _ => None,
                }
            }
            "dbrx" => match self.fields["attn_config"].get("kv_n_heads") {
                Some(value) => positive(value),
                None if self.complete => positive(&self.fields["num_attention_heads"]),
                None => None,
            },
            "nemotron-nas" => None,
            _ => self.base_kv_heads(),
        };
        // is_deepseek_mla() uses the text model type, whereas the enclosing HF
        // model type selects the converter. use_mla=false alone can mean that MLA
        // was disabled by the environment on a DeepSeek-family checkpoint.
        let text_family = resolved["text_model_type"]
            .as_str()
            .or_else(|| self.fields["model_type"].as_str())
            .or_else(|| std::ptr::eq(self.fields, top.fields).then_some(family));
        let ordinary = resolved["is_deepseek_mla"]
            .as_bool()
            .map(|mla| !mla)
            .unwrap_or_else(|| {
                if self.complete && self.fields.get("model_type").is_none() {
                    return true; // The tagged converter checks for this attribute before testing MLA.
                }
                text_family.filter(|s| known(s)).is_some_and(|name| {
                    !matches!(
                        name,
                        "axk1"
                            | "deepseek_v2"
                            | "deepseek_v3"
                            | "deepseek_v32"
                            | "deepseek_v4"
                            | "deepseek_v41"
                            | "dots3_note"
                            | "deepseek_mtp"
                            | "k3_dspark"
                            | "glm5_next"
                            | "glm5_next_text"
                            | "glm_moe_dsa"
                            | "glm4_moe_lite"
                            | "glm4_moe_lite_mtp"
                            | "kimi_k2"
                            | "kimi_linear"
                            | "longcat_flash"
                            | "longcat_flash_ngram"
                            | "pangu_ultra_moe"
                            | "pangu_ultra_moe_mtp"
                            | "bailing_hybrid"
                            | "bailing_hybrid_mtp"
                            | "bailing_hybrid_v3_mtp"
                            | "hy_v4"
                            | "hy_v4_mtp"
                            | "eagle"
                    )
                })
            });
        heads.head_size = match family {
            "zamba2" => positive(&self.fields["attention_head_dim"]),
            _ if ordinary && use_mla != Some(true) => self.base_head_size(),
            _ => None,
        };
        (heads, layers)
    }
}

/// Tagged native layer constructors whose ordinary attention layers share KV geometry.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Decoder {
    Llama,
    Mistral,
    Qwen2,
    Qwen3,
    NemotronH,
    Falcon,
}

impl Decoder {
    pub(super) fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "LlamaForCausalLM" => Self::Llama,
            "MistralForCausalLM" => Self::Mistral,
            // Qwen2-VL pins this native language model in init_vllm_registered_model.
            "Qwen2ForCausalLM" | "Qwen2Model" | "Qwen2VLForConditionalGeneration" => Self::Qwen2,
            "Qwen3ForCausalLM" => Self::Qwen3,
            "NemotronHForCausalLM" | "NemotronHPuzzleForCausalLM" => Self::NemotronH,
            "FalconForCausalLM" => Self::Falcon,
            _ => return None,
        })
    }

    fn from_family(family: &str) -> Option<Self> {
        Some(match family {
            "llama" => Self::Llama,
            "mistral" => Self::Mistral,
            "qwen2" | "qwen2_vl" => Self::Qwen2,
            "qwen3" => Self::Qwen3,
            "nemotron_h" => Self::NemotronH,
            "falcon" | "RefinedWeb" | "RefinedWebModel" => Self::Falcon,
            _ => return None,
        })
    }

    pub(super) fn family(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::NemotronH => "nemotron_h",
            Self::Falcon => "falcon",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Llama => "LlamaForCausalLM",
            Self::Mistral => "MistralForCausalLM",
            Self::Qwen2 => "Qwen2ForCausalLM",
            Self::Qwen3 => "Qwen3ForCausalLM",
            Self::NemotronH => "NemotronHForCausalLM",
            Self::Falcon => "FalconForCausalLM",
        }
    }

    pub(super) fn observed(top: &Value, selected: Option<&str>) -> Option<Self> {
        if let Some(names) = top.get("architectures").filter(|v| !v.is_null()) {
            // Native dispatch tries registered classes in order. A checkpoint's
            // advertised architecture can be a fallback behind a different loaded class.
            return Self::from_name(names.as_array()?.first()?.as_str()?);
        }
        let selected = match selected {
            Some(s) => Some(Self::from_name(s)?),
            None => None,
        };
        if let Some(name) = top["model_type"].as_str() {
            let family = Self::from_family(name)?;
            return selected.is_none_or(|d| d == family).then_some(family);
        }
        selected
    }

    pub(super) fn physical_heads(
        self,
        top: HfConfig<'_>,
        text: HfConfig<'_>,
        ordered: bool,
    ) -> Option<UniformAttention> {
        let config = match self {
            Self::Qwen2 | Self::Qwen3 => text, // Qwen3Model inherits Qwen2Model's text selection.
            _ => top,
        };
        let divided = config.divided_head_size();
        let num_kv_heads = match self {
            Self::Falcon => match (
                config.flag("new_decoder_architecture"),
                config.flag("multi_query"),
            ) {
                (Some(true), _) => positive(&config.fields["num_kv_heads"]),
                (Some(false), Some(true)) => Some(1),
                (Some(false), Some(false)) => positive(&config.fields["num_attention_heads"]),
                _ => None,
            },
            Self::Llama | Self::Mistral
                if config.complete && config.fields.get("num_key_value_heads").is_none() =>
            {
                positive(&config.fields["num_attention_heads"])
            }
            _ => positive(&config.fields["num_key_value_heads"]),
        };
        let divided_only = matches!(self, Self::Qwen2 | Self::Falcon);
        let head_size = (!divided_only)
            .then(|| {
                positive(&config.fields["head_dim"]).or_else(|| {
                    (config.complete || config.fields.get("head_dim").is_some())
                        .then_some(divided)
                        .flatten()
                })
            })
            .flatten();
        let heads = HeadGeometry {
            num_kv_heads,
            head_size,
            ..Default::default()
        };
        let divided_head_size = divided.filter(|_| divided_only);
        (!heads.is_empty() || divided_head_size.is_some()).then(|| UniformAttention {
            architecture: Some(self.name().to_owned()),
            ordered_architecture: ordered,
            heads,
            divided_head_size,
        })
    }
}

/// The constructor's KV head count stays global until the selected TP is known.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub(super) struct UniformAttention {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) architecture: Option<String>,
    // A constructor selected from the ordered registry list can differ from the profile checkpoint.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) ordered_architecture: bool,
    #[serde(flatten)]
    pub(super) heads: HeadGeometry,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) divided_head_size: Option<u64>,
}

impl UniformAttention {
    fn decoder(&self, selected: Option<&str>) -> Option<Decoder> {
        let recorded = self.architecture.as_deref().and_then(Decoder::from_name);
        if self.ordered_architecture {
            return recorded;
        }
        let requested = Decoder::from_name(selected?)?;
        recorded
            .is_none_or(|original| original == requested)
            .then_some(requested)
    }

    pub(super) fn for_selected(mut self, architecture: Option<&str>) -> Option<Self> {
        let Some(name) = architecture else {
            return Some(self);
        };
        let decoder = self.decoder(Some(name))?;
        if self.architecture.is_none() && !matches!(decoder, Decoder::Qwen2 | Decoder::Qwen3) {
            // Previously projected anonymous constructor fields came from the text
            // config. They do not establish the top-level decoder's physical dimensions.
            return None;
        }
        if matches!(decoder, Decoder::Qwen2 | Decoder::Falcon) {
            self.heads.head_size = None;
        } else {
            self.divided_head_size = None;
        }
        self.architecture = Some(decoder.name().to_owned());
        Some(self)
    }

    pub(super) fn rank_heads(&self, tp: u64, architecture: Option<&str>) -> Option<HeadGeometry> {
        let decoder = self.decoder(architecture)?;
        if tp == 0 {
            return None;
        }
        let head_size = if matches!(decoder, Decoder::Qwen2 | Decoder::Falcon) {
            self.divided_head_size
        } else {
            self.heads.head_size
        };
        let num_kv_heads = self.heads.num_kv_heads.and_then(|heads| {
            if heads.is_multiple_of(tp) {
                Some(heads / tp)
            } else if tp.is_multiple_of(heads) {
                Some(1)
            } else {
                None
            }
        });
        Some(HeadGeometry {
            num_kv_heads,
            head_size,
            ..Default::default()
        })
    }
}
