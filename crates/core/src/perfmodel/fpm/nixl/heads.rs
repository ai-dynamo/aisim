// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023 the Falcon authors and HuggingFace Inc. team. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 The HuggingFace Inc. team. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023 The vLLM team
// SPDX-FileCopyrightText: Copyright 2022 the Big Science Workshop and HuggingFace Inc. team. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL hashes the model converter's global geometry. Registered caches use the
//! attention constructor's rank-local geometry. See THIRD_PARTY_NOTICES.md for sources.

use std::{
    borrow::Cow,
    fs,
    io::{self, BufReader, Read, Seek},
    path::Path,
    sync::LazyLock,
};

use quick_cache::sync::Cache;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{AicError, known, merge_observed, positive, registry};

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
/// can still come from the checkpoint. A saved effective config or a known
/// stock constructor establishes the complete set of geometry fields.
#[derive(Clone, Copy)]
pub(super) struct HfConfig<'a> {
    pub(super) fields: &'a Value,
    complete: bool,
    verified: bool,
    class: Option<ConfigClass>,
    text_class: Option<ConfigClass>,
    stock_only: bool,
}

/// Keep only configuration fields consumed by the tagged converters and native
/// constructors. An authenticated checkpoint supplies the baseline before overlays.
fn geometry_config(config: &Value) -> serde_json::Map<String, Value> {
    let mut selected = serde_json::Map::new();
    for key in [
        "model_type",
        "architectures",
        "auto_map",
        "layers_block_type",
        "layer_types",
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

fn qwen2_vl_text_type(config: &Value, stock_only: bool) -> Option<&'static str> {
    if !stock_only && config["auto_map"].get("AutoConfig").is_some() {
        return None;
    }
    match config["model_type"].as_str() {
        Some("qwen2_vl") => Some("qwen2_vl_text"),
        Some("qwen2_5_vl") => Some("qwen2_5_vl_text"),
        _ => None,
    }
}

pub(super) fn hybrid_pattern(config: &Value) -> Option<Cow<'_, str>> {
    if let Some(layers) = config
        .get("layers_block_type")
        .or_else(|| config.get("layer_types"))
        .and_then(Value::as_array)
    {
        let pattern = layers
            .iter()
            .map(|layer| {
                Some(match layer.as_str()? {
                    "linear_attention" | "mamba" => 'M',
                    "full_attention" | "attention" => '*',
                    "moe" => 'E',
                    "mlp" => '-',
                    _ => return None,
                })
            })
            .collect::<Option<String>>()?;
        return (!pattern.is_empty()).then_some(Cow::Owned(pattern));
    }
    config["hybrid_override_pattern"]
        .as_str()
        .filter(|p| !p.is_empty() && p.bytes().all(|c| matches!(c, b'*' | b'M' | b'-' | b'E')))
        .map(Cow::Borrowed)
}

fn stock_decoder(config: &Value, stock_only: bool) -> Option<Decoder> {
    if !stock_only && config["auto_map"].get("AutoConfig").is_some() {
        return None;
    }
    let family = config["model_type"].as_str()?;
    Decoder::from_family(family).filter(|decoder| decoder.family() == family)
}

#[derive(Clone, Copy, PartialEq)]
enum ConfigClass {
    Decoder(Decoder),
    RefinedWeb,
}

impl ConfigClass {
    fn observed(config: &Value, stock_only: bool) -> Option<Self> {
        if refined_web(config) {
            Some(Self::RefinedWeb)
        } else {
            stock_decoder(config, stock_only).map(Self::Decoder)
        }
    }
}

fn wrapper_text_default(config: &Value, stock_only: bool) -> Option<&'static str> {
    if !stock_only && config["auto_map"].get("AutoConfig").is_some() {
        return None;
    }
    match config["model_type"].as_str()? {
        "llava" | "llava_next" => Some("llama"),
        "idefics2" => Some("mistral"),
        _ => None,
    }
}

fn refined_web(config: &Value) -> bool {
    matches!(
        config["model_type"].as_str(),
        Some("RefinedWeb" | "RefinedWebModel")
    )
}

// RWConfig stores its public attributes under these names, including after config.update().
const RW_ATTRIBUTES: [(&str, &str, u64); 3] = [
    ("num_attention_heads", "n_head", 8),
    ("num_hidden_layers", "n_layer", 2),
    ("num_kv_heads", "n_head_kv", 1),
];

fn apply_rw_attributes(fields: &mut serde_json::Map<String, Value>, values: &Value, init: bool) {
    for (public, stored, default) in RW_ATTRIBUTES {
        let value = match (values.get(public), values.get(stored)) {
            // Initial aliases are passed to the base class after the RW constructor.
            // Archived override maps do not preserve the order of two conflicting setters.
            (Some(a), Some(b)) if !init && a != b => Value::Null,
            (Some(a), _) => a.clone(),
            (None, Some(Value::Null)) if init && stored == "n_head_kv" => Value::from(1),
            (None, Some(b)) => b.clone(),
            (None, None) if init => Value::from(default),
            _ => continue,
        };
        fields.remove(public);
        fields.insert(stored.into(), value);
    }
}

/// Ordinary HF constructors distinguish omitted arguments from explicit nulls.
/// Values assigned in __post_init__ retain their initial value across later updates.
fn initialize_decoder(config: &Value, top: &mut serde_json::Map<String, Value>, decoder: Decoder) {
    let falcon = decoder == Decoder::Falcon;
    if falcon && let Some(size) = config.get("n_embed").filter(|v| !v.is_null()) {
        // Falcon consumes this legacy checkpoint argument only during construction.
        top.insert("hidden_size".into(), size.clone());
    }
    top.entry("hidden_size")
        .or_insert_with(|| Value::from(if falcon { 4544 } else { 4096 }));
    top.entry("num_attention_heads")
        .or_insert_with(|| Value::from(if falcon { 71 } else { 32 }));
    if decoder != Decoder::NemotronH {
        top.entry("num_hidden_layers").or_insert(Value::from(32));
    }
    let query_heads = top["num_attention_heads"].clone();
    let (kv_field, default) = match decoder {
        Decoder::Falcon => ("num_kv_heads", Value::Null),
        Decoder::Llama => ("num_key_value_heads", Value::Null),
        Decoder::Qwen2 | Decoder::Qwen3 => ("num_key_value_heads", Value::from(32)),
        Decoder::Mistral | Decoder::NemotronH => ("num_key_value_heads", Value::from(8)),
    };
    let kv_heads = top.entry(kv_field).or_insert(default);
    if kv_heads.is_null() && !matches!(decoder, Decoder::Mistral | Decoder::NemotronH) {
        *kv_heads = query_heads;
    }
    match decoder {
        Decoder::Llama | Decoder::Mistral if top.get("head_dim").is_none_or(Value::is_null) => {
            if let Some(width) = divided_head_size(&top["hidden_size"], &top["num_attention_heads"])
            {
                top.insert("head_dim".into(), Value::from(width));
            }
        }
        Decoder::Qwen3 | Decoder::NemotronH => {
            top.entry("head_dim").or_insert(Value::from(128));
        }
        Decoder::Falcon => {
            top.entry("new_decoder_architecture")
                .or_insert(Value::Bool(false));
            top.entry("multi_query").or_insert(Value::Bool(true));
        }
        _ => {}
    }
}

fn divided_head_size(hidden_size: &Value, query_heads: &Value) -> Option<u64> {
    positive(hidden_size)?
        .checked_div(positive(query_heads)?)
        .filter(|&size| size > 0)
}

/// Project the authenticated HF checkpoint before vLLM applies its serving overrides.
fn checkpoint_geometry(config: &Value, stock_only: bool) -> Value {
    let mut top = geometry_config(config);
    if refined_web(config) {
        apply_rw_attributes(&mut top, config, true);
        let hidden = config
            .get("n_embed")
            .filter(|v| !v.is_null())
            .or_else(|| config.get("hidden_size"))
            .cloned()
            .unwrap_or(Value::from(64));
        if hidden == 8192 {
            top.insert("new_decoder_architecture".into(), Value::Bool(true));
        }
        top.insert("hidden_size".into(), hidden);
        top.entry("new_decoder_architecture")
            .or_insert(Value::Bool(false));
        top.entry("multi_query").or_insert(Value::Bool(true));
    } else if let Some(decoder) = stock_decoder(config, stock_only) {
        initialize_decoder(config, &mut top, decoder);
    }
    if let Some(family) = qwen2_vl_text_type(config, stock_only) {
        // Both Qwen2 VL constructors consume these flat kwargs, even when an
        // explicit text_config supplies the actual values. Other flat attributes
        // (including head_dim and n_head_kv) stay on the enclosing config.
        let declared = config["text_config"].is_object();
        let mut text = geometry_config(&config["text_config"]);
        for (key, default) in [
            ("hidden_size", 8192),
            ("num_hidden_layers", 80),
            ("num_attention_heads", 64),
            ("num_key_value_heads", 8),
        ] {
            let flat = top.remove(key);
            text.entry(key)
                .or_insert_with(|| flat.filter(|_| !declared).unwrap_or(Value::from(default)));
        }
        if text["num_key_value_heads"].is_null() {
            text.insert(
                "num_key_value_heads".into(),
                text["num_attention_heads"].clone(),
            );
        }
        text.entry("model_type")
            .or_insert_with(|| Value::from(family));
        top.insert("text_config".into(), Value::Object(text));
    } else if let Some(default) = wrapper_text_default(config, stock_only) {
        // These stock wrappers instantiate CONFIG_MAPPING[text_config.model_type].
        let mut text = geometry_config(&config["text_config"]);
        let family = text
            .entry("model_type")
            .or_insert(Value::from(default))
            .as_str();
        if let Some(decoder) = family.and_then(Decoder::from_family)
            && Some(decoder.family()) == family
        {
            initialize_decoder(&config["text_config"], &mut text, decoder);
        }
        top.insert("text_config".into(), Value::Object(text));
    } else if let Some(key) = text_field(config) {
        top.insert(key.into(), Value::Object(geometry_config(&config[key])));
    }
    Value::Object(top)
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct CheckpointKey {
    stock_only: bool,
    path: std::path::PathBuf,
    digest: String,
    size: u64,
    modified: std::time::SystemTime,
    #[cfg(unix)]
    file_id: (u64, u64, i64, i64),
}

impl CheckpointKey {
    fn new(path: &Path, digest: &str, metadata: &fs::Metadata, stock_only: bool) -> Option<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Some(Self {
            stock_only,
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

pub(super) struct Checkpoint {
    fields: Value,
    class: Option<ConfigClass>,
    text_class: Option<ConfigClass>,
}

/// Python's JSON reader also accepts NaN and infinities in checkpoint settings.
/// Preserve strings and stream the file. A nonfinite number cannot be a KV dimension.
struct CheckpointReader<R: Read> {
    bytes: std::iter::Peekable<io::Bytes<R>>,
    string: bool,
    escaped: bool,
    pending: &'static [u8],
}

impl<R: Read> Read for CheckpointReader<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        for (i, slot) in output.iter_mut().enumerate() {
            if let Some((first, rest)) = self.pending.split_first() {
                *slot = *first;
                self.pending = rest;
                continue;
            }
            let Some(byte) = self.bytes.next().transpose()? else {
                return Ok(i);
            };
            *slot = byte;
            if self.string {
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == b'"' {
                    self.string = false;
                }
                continue;
            }
            let rest: &[u8] = match byte {
                b'N' => b"aN",
                b'I' => b"nfinity",
                b'-' if matches!(self.bytes.peek(), Some(Ok(b'I'))) => b"Infinity",
                _ => {
                    self.string = byte == b'"';
                    continue;
                }
            };
            for &expected in rest {
                if self.bytes.next().transpose()? != Some(expected) {
                    return Err(io::ErrorKind::InvalidData.into());
                }
            }
            *slot = b' ';
            self.pending = b"-1 ";
        }
        Ok(output.len())
    }
}

pub(super) fn checkpoint(
    model: &Value,
    source: &Value,
    release: Option<&str>,
) -> Option<Checkpoint> {
    // An omitted override table is different from an explicitly empty one:
    // a historical writer may not have captured the serving overrides.
    let overrides = model.get("hf_overrides").filter(|v| v.is_object())?;
    if model["hf_config"].is_object()
        && !overrides["model_type"].is_string()
        && !overrides["auto_map"].is_object()
    {
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
    let stock_only = model["trust_remote_code"] == false;
    let key = CheckpointKey::new(path, digest, &metadata, stock_only)?;
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
            if !opened.is_file()
                || CheckpointKey::new(path, digest, &opened, stock_only).as_ref() != Some(&key)
            {
                return Ok(None);
            }
            let mut hash = Sha256::new();
            std::io::copy(&mut file, &mut hash)?;
            if format!("{:x}", hash.finalize()) != digest {
                return Ok(None);
            }
            file.rewind()?;
            let reader = CheckpointReader {
                bytes: BufReader::new(&file).bytes().peekable(),
                string: false,
                escaped: false,
                pending: &[],
            };
            let config = serde_json::from_reader::<_, Value>(reader).ok();
            // Keep the scope only if the same descriptor still identifies the hashed file.
            if CheckpointKey::new(path, digest, &file.metadata()?, stock_only).as_ref()
                != Some(&key)
            {
                return Ok(None);
            }
            Ok(config
                .filter(Value::is_object)
                .map(|config| std::sync::Arc::new(checkpoint_geometry(&config, stock_only))))
        })
        .ok()
        .flatten()?;
    let decoder = stock_decoder(&saved, stock_only);
    let rw = refined_web(&saved);
    if let Some(family) = overrides["model_type"].as_str()
        && saved["model_type"] != family
        && (rw || registry::substitutes_config(family, release))
    {
        // An ordinary model_type update does not change the class AutoConfig loaded.
        // Registered parser substitutions can instead select a different constructor.
        return None;
    }
    let mut effective = (*saved).clone();
    if let Some(key) = text_field(&saved).or_else(|| text_field(overrides)) {
        if effective[key].is_null() {
            effective[key] = serde_json::json!({});
        }
        apply_geometry(&mut effective[key], &overrides[key]);
    }
    apply_geometry(&mut effective, overrides);
    if rw {
        apply_rw_attributes(effective.as_object_mut()?, overrides, false);
    }
    if decoder == Some(Decoder::Falcon) {
        // Falcon exposes head_dim as a read-only property of the current scalars.
        effective["head_dim"] =
            divided_head_size(&effective["hidden_size"], &effective["num_attention_heads"]).into();
    }
    if decoder == Some(Decoder::NemotronH) {
        // Stock NemotronHConfig computes both properties from the same list. Its
        // compatibility setter for num_hidden_layers ignores scalar updates.
        let pattern = hybrid_pattern(overrides)
            .or_else(|| hybrid_pattern(&saved))
            .unwrap_or(Cow::Borrowed("ME*-"));
        let fields = effective.as_object_mut()?;
        fields.remove("layers_block_type");
        fields.remove("layer_types");
        fields.insert("num_hidden_layers".into(), Value::from(pattern.len()));
        fields.insert(
            "hybrid_override_pattern".into(),
            Value::from(pattern.as_ref()),
        );
    }
    Some(Checkpoint {
        fields: effective,
        class: ConfigClass::observed(&saved, stock_only),
        text_class: if qwen2_vl_text_type(&saved, stock_only).is_some() {
            Some(ConfigClass::Decoder(Decoder::Qwen2))
        } else if wrapper_text_default(&saved, stock_only).is_some() {
            stock_decoder(&saved["text_config"], true).map(ConfigClass::Decoder)
        } else {
            None
        },
    })
}

impl<'a> HfConfig<'a> {
    pub(super) fn observed(model: &'a Value, checkpoint: Option<&'a Checkpoint>) -> Self {
        let saved = model.get("hf_config").filter(|v| v.is_object());
        let fields = saved
            .or_else(|| checkpoint.map(|c| &c.fields))
            .unwrap_or(&model["hf_overrides"]);
        let stock_only = model["trust_remote_code"] == false;
        Self {
            fields,
            complete: saved.is_some() || checkpoint.is_some_and(|c| c.class.is_some()),
            verified: saved.is_none() && checkpoint.is_some(),
            class: checkpoint
                .map_or_else(|| ConfigClass::observed(fields, stock_only), |c| c.class),
            text_class: checkpoint.and_then(|c| c.text_class),
            stock_only,
        }
    }

    pub(super) fn model_type(self, model: &'a Value) -> Option<&'a str> {
        // Stock to_dict() serializes the class's model_type even after config.update()
        // sets a different instance value. vLLM selects its converter from the instance.
        if (self.class.is_some() || self.text_class.is_some())
            && model["hf_config"].is_object()
            && let Some(family) = model["hf_overrides"]["model_type"].as_str()
        {
            return Some(family);
        }
        self.fields["model_type"].as_str()
    }

    pub(super) fn text_config(
        self,
        model: &'a Value,
        referenced: bool,
        registered: Option<registry::Registration>,
    ) -> Self {
        if let Some(fields) = model.get("hf_text_config").filter(|v| v.is_object()) {
            return Self {
                fields,
                complete: true,
                verified: false,
                class: self
                    .text_class
                    .or_else(|| ConfigClass::observed(fields, self.stock_only)),
                ..self
            };
        }
        let qwen_vl = qwen2_vl_text_type(self.fields, self.stock_only).is_some();
        let nested = text_field(self.fields).or_else(|| {
            // A recorded wrapper can establish that its outer override is not a text field.
            (qwen_vl || registered == Some(registry::Registration::Qwen2Vl))
                .then_some("text_config")
        });
        if let Some(key) = nested {
            Self {
                fields: &self.fields[key],
                complete: (self.complete && !self.verified) || self.text_class.is_some(),
                class: self
                    .text_class
                    .or_else(|| ConfigClass::observed(&self.fields[key], self.stock_only)),
                ..self
            }
        } else if referenced
            && !self.complete
            && !self.verified
            && self.fields["model_type"].as_str().is_none()
        {
            // A missing archive cannot identify whether an outer override reaches the text model.
            Self {
                fields: &Value::Null,
                complete: false,
                verified: false,
                ..self
            }
        } else {
            self
        }
    }

    pub(super) fn may_load_dynamic_model(self, model: &Value) -> bool {
        if model["trust_remote_code"] == false {
            return false;
        }
        match self.fields.get("auto_map") {
            Some(Value::Object(m)) => m.keys().any(|k| k.starts_with("AutoModel")),
            Some(Value::Null) => false,
            None => !(self.complete || self.verified),
            _ => true,
        }
    }

    fn attribute(self, name: &str) -> &'a Value {
        if self.class == Some(ConfigClass::RefinedWeb)
            && let Some((_, stored, _)) =
                RW_ATTRIBUTES.iter().find(|(public, _, _)| *public == name)
            && let Some(value) = self.fields.get(*stored)
        {
            return value;
        }
        &self.fields[name]
    }

    fn divided_head_size(self) -> Option<u64> {
        divided_head_size(
            &self.fields["hidden_size"],
            self.attribute("num_attention_heads"),
        )
    }

    fn base_kv_heads(self) -> Option<u64> {
        if self.class == Some(ConfigClass::RefinedWeb) {
            // The first two converter aliases are the same RWConfig attribute.
            return self.attribute("num_kv_heads").as_u64();
        }
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
        self.attribute("num_attention_heads").as_u64()
    }

    fn base_head_size(self) -> Option<u64> {
        // FalconConfig and RWConfig omit their read-only head_dim property from to_dict().
        if self.complete
            && self.fields.get("head_dim").is_none()
            && matches!(
                self.class,
                Some(ConfigClass::RefinedWeb | ConfigClass::Decoder(Decoder::Falcon))
            )
        {
            return self.divided_head_size();
        }
        if let Some(size) = positive(&self.fields["head_dim"]) {
            return Some(size);
        }
        let qwen2_config =
            self.verified && self.class == Some(ConfigClass::Decoder(Decoder::Qwen2));
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
        match self.fields.get(name) {
            Some(Value::Null) if self.complete => Some(false),
            Some(value) => value.as_bool(),
            None => self.complete.then_some(false),
        }
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
        let stored_layers = positive(self.attribute("num_hidden_layers"));
        let layers = match family {
            "longcat_flash" => None,
            "nemotron_h" if std::ptr::eq(self.fields, top.fields) => hybrid_pattern(self.fields)
                .map(|p| p.len() as u64)
                .or_else(|| {
                    (self.complete || self.verified)
                        .then_some(stored_layers)
                        .flatten()
                }),
            _ => stored_layers,
        };
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
                text_family
                    .filter(|s| self.complete || known(s))
                    .is_some_and(|name| {
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
        registry::at_release(None, |minor| registry::lookup(name, minor)).and_then(|r| r.decoder())
    }

    fn from_family(family: &str) -> Option<Self> {
        Some(match family {
            "llama" => Self::Llama,
            "mistral" => Self::Mistral,
            "qwen2" | "qwen2_vl" | "qwen2_5_vl" => Self::Qwen2,
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

    pub(super) fn inferred(top: &Value, selected: Option<&str>) -> Option<Self> {
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
                (Some(true), _) => positive(config.attribute("num_kv_heads")),
                (Some(false), Some(true)) => Some(1),
                (Some(false), Some(false)) => positive(config.attribute("num_attention_heads")),
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
