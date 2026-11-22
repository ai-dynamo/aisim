// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact measured MoE-stage table. Native loading consumes Parquet typed keys
//! and capture provenance; Python imports legacy JSON before this boundary.

use super::parquet_loader::PerfReader;
use crate::{
    AicError, EstimationMode, ForwardPassFallbackPolicy, ForwardPassPerfModelConfig,
    ForwardPassWorkerType,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    path::PathBuf,
};

fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidEngineConfig(message.into())
}

/// Immutable identity of one measured split-stage configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastAfdIdentity {
    pub logical_batch_per_source_rank: u32,
    pub attention_gpus: u32,
    pub ffn_gpus: u32,
    pub f_moe_ep_size: u32,
    pub microbatches: u32,
    pub moe_layers: u32,
    pub routed_topk: u32,
    pub boundary_on_attn: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastAfdConfig {
    pub profile_path: PathBuf,
}

pub(crate) fn validate_topology(
    worker_type: ForwardPassWorkerType,
    tp: u32,
    attention_gpus: u32,
    ffn_gpus: u32,
    f_moe_ep_size: u32,
    microbatches: u32,
    boundary_on_attn: bool,
) -> Result<(), AicError> {
    if worker_type != ForwardPassWorkerType::Decode {
        return Err(invalid(
            "FastAFD split MoE stage profiles measure decode only",
        ));
    }
    if tp != 1 {
        return Err(invalid(format!(
            "FastAFD requires tp_a={tp}; only tp_a=1 is supported"
        )));
    }
    if ffn_gpus == 0 || f_moe_ep_size != ffn_gpus {
        return Err(invalid("FastAFD experts must span every F GPU"));
    }
    if !boundary_on_attn {
        return Err(invalid("FastAFD requires boundary_on_attn=true"));
    }
    if attention_gpus == 0 || microbatches == 0 {
        return Err(invalid(
            "FastAFD pool sizes and microbatches must be positive",
        ));
    }
    Ok(())
}

pub(crate) fn validate_config(config: &ForwardPassPerfModelConfig) -> Result<(), AicError> {
    let enabled = config.estimation_mode == EstimationMode::FastafdStage;
    if enabled != config.afd.is_some() || enabled != config.estimator_config.fastafd_stage.is_some()
    {
        return Err(invalid(
            "fastafd_stage requires afd identity and estimator_config.fastafd_stage together",
        ));
    }
    if !enabled {
        return Ok(());
    }
    let identity = config.afd.as_ref().unwrap();
    validate_topology(
        config.worker_type,
        config.tp,
        identity.attention_gpus,
        identity.ffn_gpus,
        identity.f_moe_ep_size,
        identity.microbatches,
        identity.boundary_on_attn,
    )?;
    if identity.logical_batch_per_source_rank == 0
        || identity.moe_layers == 0
        || identity.routed_topk == 0
    {
        return Err(invalid(
            "FastAFD batch, layers and routed top-k must be positive",
        ));
    }
    if config.fallback_policy != ForwardPassFallbackPolicy::Deny
        || config.backend_version.is_none()
        || config.pp != 1
        || config.attention_dp != 1
        || config.cp_size.is_some()
        || config.dcp.is_some()
        || config.moe_tp_size.is_some()
        || config.moe_ep_size.is_some()
        || config.fpm_profile.is_some()
        || config.speculation.is_some()
        || config.decoder_replay
        || config.enable_eplb
        || config.wideep_num_slots.is_some()
        || config.moe_kernel_source.is_some()
        || config.enable_shared_layer.is_some()
        || config.gemm_quant_mode.is_some()
        || config.fmha_quant_mode.is_some()
        || config.kvcache_quant_mode.is_some()
        || config.comm_quant_mode.is_some()
        || config.attention_backend.is_some()
        || config.database_mode != crate::DatabaseMode::Silicon
        || config.transfer_policy.is_some()
        || !config.systems_paths.is_empty()
        || config.strict_provenance
        || config.kv_block_size.is_some()
    {
        return Err(invalid(
            "FastAFD stage timing requires a literal backend version, deny fallback, and no unrelated estimator/graph controls",
        ));
    }
    if !matches!(
        config.moe_backend.as_deref(),
        Some("megamoe" | "deepep_deepgemm")
    ) {
        return Err(invalid(
            "FastAFD requires moe_backend='megamoe' or 'deepep_deepgemm'",
        ));
    }
    let controls = &config.estimator_config;
    let mut supported = crate::EstimatorConfig::default();
    supported.fastafd_stage = controls.fastafd_stage.clone();
    // Canonical resolution expands the unrelated Auto default to Sol. Accept
    // that same default before and after serialization, without permitting a
    // direct interpolation override on the stage estimator.
    if controls.fpm_interpolation.method == crate::FpmInterpolationMethod::Sol {
        supported.fpm_interpolation.method = crate::FpmInterpolationMethod::Sol;
    }

    if controls != &supported {
        return Err(invalid(
            "FastAFD stage timing does not support controls for other estimators",
        ));
    }
    if controls
        .fastafd_stage
        .as_ref()
        .unwrap()
        .profile_path
        .as_os_str()
        .is_empty()
    {
        return Err(invalid("FastAFD profile_path cannot be empty"));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastAfdKey {
    pub model_path: String,
    pub system: String,
    pub stage: String,
    pub topology: String,
    pub logical_batch_per_source_rank: u32,
    pub mtp_nextn: u32,
    pub microbatches: u32,
    pub moe_layers: u32,
    pub routed_topk: u32,
    pub moe_precision: String,
    pub moe_backend: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    repository: String,
    commit: String,
    benchmark: String,
    boundary: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Capture {
    scope: String,
    method: String,
    method_version: String,
    statistic: String,
    sample_count: u32,
    cv_percent: f64,
    routing: String,
    source_tree_sha256: String,
    raw_artifact: String,
    raw_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Validation {
    stable: bool,
    correctness: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    key: FastAfdKey,
    model_profile: String,
    latency_ms: f64,
    measurement: Capture,
    validation: Validation,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ParquetCapture {
    model_profile: String,
    measurement: Capture,
    validation: Validation,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    schema: String,
    lookup_policy: String,
    source: Source,
    entries: Vec<Entry>,
}

fn digest(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn text(value: &str) -> bool {
    !value.trim().is_empty()
}
fn topology(value: &str, stage: &str) -> bool {
    let positive = |s: &str| s.parse::<u32>().is_ok_and(|n| n > 0 && n.to_string() == s);
    match stage {
        "agg" => value.strip_prefix("ep").is_some_and(positive),
        "afd" => value
            .strip_suffix('F')
            .and_then(|s| s.split_once('A'))
            .is_some_and(|(a, f)| positive(a) && positive(f)),
        _ => false,
    }
}
impl Document {
    fn normalize(&mut self) {
        // Match the legacy import's text normalization at the data boundary.
        fn trim(values: &mut [&mut String]) {
            for value in values {
                **value = value.trim().to_owned();
            }
        }
        let s = &mut self.source;
        trim(&mut [
            &mut s.repository,
            &mut s.commit,
            &mut s.benchmark,
            &mut s.boundary,
        ]);
        for e in &mut self.entries {
            let k = &mut e.key;
            trim(&mut [
                &mut k.model_path,
                &mut k.system,
                &mut k.stage,
                &mut k.topology,
                &mut k.moe_precision,
                &mut k.moe_backend,
                &mut e.model_profile,
            ]);
            let c = &mut e.measurement;
            trim(&mut [
                &mut c.method,
                &mut c.method_version,
                &mut c.routing,
                &mut c.source_tree_sha256,
                &mut c.raw_artifact,
                &mut c.raw_sha256,
            ]);
        }
    }
    fn validate(&self) -> Result<(), AicError> {
        if self.schema != "aisimulate.fastafd-moe-stage.v1"
            || self.lookup_policy != "exact-only"
            || self.entries.is_empty()
        {
            return Err(invalid(
                "invalid FastAFD schema, lookup policy or empty table",
            ));
        }
        let s = &self.source;
        if !text(&s.repository)
            || !digest(&s.commit, 40)
            || !text(&s.benchmark)
            || !text(&s.boundary)
        {
            return Err(invalid("invalid FastAFD source provenance"));
        }
        let mut keys = HashSet::new();
        for e in &self.entries {
            let k = &e.key;
            let c = &e.measurement;
            if !keys.insert(&e.key) {
                return Err(invalid("duplicate FastAFD MoE stage key"));
            }
            if !text(&k.model_path)
                || !text(&k.system)
                || !topology(&k.topology, &k.stage)
                || k.logical_batch_per_source_rank == 0
                || k.microbatches == 0
                || k.moe_layers == 0
                || k.routed_topk == 0
                || !text(&k.moe_precision)
                || !text(&k.moe_backend)
                || !text(&e.model_profile)
                || !e.latency_ms.is_finite()
                || e.latency_ms <= 0.0
                || !e.validation.stable
                || !matches!(e.validation.correctness, Value::Null | Value::Bool(true))
                || c.scope != "complete_moe_stage"
                || c.statistic != "p50"
                || c.sample_count == 0
                || !c.cv_percent.is_finite()
                || c.cv_percent < 0.0
                || !text(&c.method)
                || !text(&c.method_version)
                || !text(&c.routing)
                || !text(&c.raw_artifact)
                || !digest(&c.source_tree_sha256, 64)
                || !digest(&c.raw_sha256, 64)
            {
                return Err(invalid("invalid or unqualified FastAFD measurement"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FastAfdTable {
    entries: BTreeMap<u32, Vec<Entry>>,
    source: Source,
    path: PathBuf,
    sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FastAfdStageEstimate {
    pub ffn_ms: f64,
    pub num_layers: u32,
    pub stage: Value,
}

impl FastAfdTable {
    pub(crate) fn load(config: &ForwardPassPerfModelConfig) -> Result<Self, AicError> {
        let path = &config
            .estimator_config
            .fastafd_stage
            .as_ref()
            .unwrap()
            .profile_path;
        let raw = std::fs::read(path).map_err(|source| AicError::Io {
            path: path.clone(),
            source,
        })?;
        Self::from_snapshot(config, raw.into())
    }

    fn from_snapshot(
        config: &ForwardPassPerfModelConfig,
        raw: bytes::Bytes,
    ) -> Result<Self, AicError> {
        let digest = format!("{:x}", Sha256::digest(&raw));
        Self::from_parquet_snapshot(config, raw, digest)
    }

    pub(crate) fn from_parquet_snapshot(
        config: &ForwardPassPerfModelConfig,
        raw: bytes::Bytes,
        source_sha256: String,
    ) -> Result<Self, AicError> {
        let path = &config
            .estimator_config
            .fastafd_stage
            .as_ref()
            .unwrap()
            .profile_path;
        let mut document = {
            let reader = PerfReader::from_bytes(path, raw.clone())?;
            let key_names = [
                "model_path",
                "system",
                "stage",
                "topology",
                "logical_batch_per_source_rank",
                "mtp_nextn",
                "microbatches",
                "moe_layers",
                "routed_topk",
                "moe_precision",
                "moe_backend",
            ];
            let columns = key_names
                .iter()
                .map(|name| reader.col(name))
                .collect::<Result<Vec<_>, _>>()?;
            let latency = reader.col("latency_ms")?;
            let metadata = reader.col("capture_json")?;
            let source = reader.col("source_json")?;
            let mut entries = Vec::new();
            let mut origin = None;
            for row in reader.rows()? {
                let row = row?;
                let current = row.str_owned(source)?;
                if origin.as_ref().is_some_and(|s| s != &current) {
                    return Err(invalid("mixed FastAFD source provenance"));
                }
                origin = Some(current);
                let mut key = serde_json::Map::new();
                for (index, name) in key_names.iter().enumerate() {
                    key.insert(
                        (*name).into(),
                        if (4..=8).contains(&index) {
                            json!(row.u32(columns[index])?)
                        } else {
                            json!(row.str_owned(columns[index])?)
                        },
                    );
                }
                let capture: ParquetCapture = serde_json::from_str(&row.str_owned(metadata)?)
                    .map_err(|e| invalid(e.to_string()))?;
                entries.push(Entry {
                    key: serde_json::from_value(Value::Object(key))
                        .map_err(|e| invalid(e.to_string()))?,
                    latency_ms: row.f64(latency)?,
                    model_profile: capture.model_profile,
                    measurement: capture.measurement,
                    validation: capture.validation,
                });
            }
            Document {
                schema: "aisimulate.fastafd-moe-stage.v1".into(),
                lookup_policy: "exact-only".into(),
                source: serde_json::from_str(
                    &origin.ok_or_else(|| invalid("empty FastAFD parquet table"))?,
                )
                .map_err(|e| invalid(e.to_string()))?,
                entries,
            }
        };
        document.normalize();
        document.validate()?;
        let id = config.afd.as_ref().unwrap();
        let shape = format!("{}A{}F", id.attention_gpus, id.ffn_gpus);
        let mut entries: BTreeMap<u32, Vec<Entry>> = BTreeMap::new();
        for entry in document.entries {
            let k = &entry.key;
            if k.model_path == config.model
                && k.system == config.system
                && k.stage == "afd"
                && k.topology == shape
                && k.mtp_nextn == config.nextn
                && k.logical_batch_per_source_rank == id.logical_batch_per_source_rank
                && k.microbatches == id.microbatches
                && k.moe_layers == id.moe_layers
                && k.routed_topk == id.routed_topk
                && Some(&k.moe_backend) == config.moe_backend.as_ref()
                && config
                    .moe_quant_mode
                    .as_ref()
                    .is_none_or(|q| q == &k.moe_precision)
            {
                entries
                    .entry(k.logical_batch_per_source_rank)
                    .or_default()
                    .push(entry);
            }
        }
        if entries.is_empty() {
            return Err(invalid(
                "no FastAFD split MoE stage measurement matches the canonical identity",
            ));
        }
        if entries.values().any(|variants| variants.len() != 1) {
            return Err(invalid(
                "several FastAFD split MoE stage measurements match; set engine.moe_quant_mode to pick the profile precision",
            ));
        }
        Ok(Self {
            entries,
            source: document.source,
            path: path.clone(),
            sha256: source_sha256,
        })
    }

    pub(crate) fn precision(&self) -> &str {
        // Construction retains exactly one batch and one qualified capture.
        &self.entries.first_key_value().unwrap().1[0]
            .key
            .moe_precision
    }

    pub(crate) fn estimate(&self, batch: u32) -> Result<FastAfdStageEstimate, AicError> {
        if batch == 0 {
            return Err(invalid("FastAFD logical batch must be positive"));
        }
        let matches=self.entries.get(&batch).filter(|v| !v.is_empty()).ok_or_else(|| invalid("no FastAFD split MoE stage measurement matches the canonical identity and batch"))?;
        if matches.len() != 1 {
            return Err(invalid(
                "several FastAFD split MoE stage measurements match; set engine.moe_quant_mode to pick the profile precision",
            ));
        }
        let e = &matches[0];
        let c = &e.measurement;
        let s = &self.source;
        Ok(FastAfdStageEstimate {
            ffn_ms: e.latency_ms / (f64::from(e.key.moe_layers) * f64::from(e.key.microbatches)),
            num_layers: e.key.moe_layers,
            stage: json!({"profile_path":self.path,"profile_sha256":self.sha256,"key":e.key,"latency_ms":e.latency_ms,
                "provider":"fastafd","schema":"aisimulate.fastafd-moe-stage.v1","repository":s.repository,"source_commit":s.commit,"benchmark":s.benchmark,"boundary":s.boundary,
                "model_profile":e.model_profile,"method":c.method,"method_version":c.method_version,"statistic":c.statistic,"sample_count":c.sample_count,
                "cv_percent":c.cv_percent,"routing":c.routing,"source_tree_sha256":c.source_tree_sha256,"raw_artifact":c.raw_artifact,"raw_sha256":c.raw_sha256,"correctness":e.validation.correctness}),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendKind, ForwardPassPerfModel, ForwardPassPerfSource};
    use std::io::Write;

    fn fixture_document() -> Value {
        // Synthetic capture: 12 ms for three layers and two microbatches.
        // Each of the six pipeline units therefore costs exactly 2 ms.
        let document = json!({
            "schema":"aisimulate.fastafd-moe-stage.v1", "lookup_policy":"exact-only",
            "source":{"repository":"https://example.invalid/fixture","commit":"0".repeat(40),"benchmark":"synthetic","boundary":"complete stage"},
            "entries":[{"key":{"model_path":"synthetic","system":"gb200","stage":"afd","topology":"4A8F","logical_batch_per_source_rank":128,"mtp_nextn":0,"microbatches":2,"moe_layers":3,"routed_topk":8,"moe_precision":"fp8","moe_backend":"megamoe"},
                "latency_ms":12.0,"model_profile":"synthetic",
                "measurement":{"scope":"complete_moe_stage","method":"synthetic","method_version":"1","statistic":"p50","sample_count":40,"cv_percent":0.5,"routing":"balanced","source_tree_sha256":"a".repeat(64),"raw_artifact":"fixture.json","raw_sha256":"b".repeat(64)},
                "validation":{"stable":true,"correctness":null}}]
        });
        document
    }

    fn fixture() -> (tempfile::NamedTempFile, ForwardPassPerfModelConfig) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&parquet_snapshot(&fixture_document()))
            .unwrap();
        let mut config = ForwardPassPerfModelConfig::new(
            "synthetic",
            "gb200",
            BackendKind::Sglang,
            ForwardPassWorkerType::Decode,
        );
        config.backend_version = Some("0.5.14".into());
        config.estimation_mode = EstimationMode::FastafdStage;
        config.moe_backend = Some("megamoe".into());
        config.afd = Some(FastAfdIdentity {
            logical_batch_per_source_rank: 128,
            attention_gpus: 4,
            ffn_gpus: 8,
            f_moe_ep_size: 8,
            microbatches: 2,
            moe_layers: 3,
            routed_topk: 8,
            boundary_on_attn: true,
        });
        config.estimator_config.fastafd_stage = Some(FastAfdConfig {
            profile_path: file.path().into(),
        });
        (file, config)
    }

    #[test]
    fn native_loading_rejects_legacy_json() {
        let (mut file, config) = fixture();
        file.as_file_mut().set_len(0).unwrap();
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(0)).unwrap();
        write!(file, "{}", fixture_document()).unwrap();
        let error = ForwardPassPerfModel::best_available(config)
            .unwrap_err()
            .to_string();
        assert!(error.contains("parquet") || error.contains("Parquet"));
    }

    #[test]
    fn canonical_stage_oracle_and_saved_identity() {
        let (_file, config) = fixture();
        let model = ForwardPassPerfModel::best_available(config).unwrap();
        let estimate = model.estimate_fastafd_stage(128).unwrap();
        assert_eq!(estimate.ffn_ms, 2.0);
        assert_eq!(estimate.num_layers, 3);
        assert_eq!(estimate.stage["latency_ms"], 12.0);
        assert_eq!(
            model.diagnostics().source,
            ForwardPassPerfSource::FastafdStage
        );
        assert!(model.estimate_forward_pass_time_ms(&[]).is_err());
        let saved = model.diagnostics().provenance.unwrap().config;
        let restored = serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
        let mut model = ForwardPassPerfModel::best_available(restored).unwrap();
        assert_eq!(model.estimate_fastafd_stage(128).unwrap().ffn_ms, 2.0);
        assert!(model.estimate_fastafd_stage(127).is_err());
        assert!(model.estimate_fastafd_stage(0).is_err());
        assert!(model.tune_with_fpms(&[Vec::new()]).is_err());
    }

    fn parquet_snapshot(document: &Value) -> bytes::Bytes {
        use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;
        use std::sync::Arc;

        fn fields(document: &Value, entry: &Value) -> serde_json::Map<String, Value> {
            let mut fields = entry["key"].as_object().unwrap().clone();
            let mut capture = entry.as_object().unwrap().clone();
            capture.remove("key");
            capture.remove("latency_ms");
            fields.insert("latency_ms".into(), entry["latency_ms"].clone());
            fields.insert(
                "capture_json".into(),
                json!(Value::Object(capture).to_string()),
            );
            fields.insert("source_json".into(), json!(document["source"].to_string()));
            fields
        }
        let first = fields(document, &document["entries"][0]);
        let columns = first
            .iter()
            .map(|(name, value)| {
                let data_type = if name == "latency_ms" {
                    "DOUBLE"
                } else if value.is_number() {
                    "INT64"
                } else {
                    "BINARY"
                };
                let logical_type = if value.is_string() { " (UTF8)" } else { "" };
                format!("REQUIRED {data_type} {name}{logical_type};")
            })
            .collect::<String>();
        let schema = Arc::new(parse_message_type(&format!("message stage {{{columns}}}")).unwrap());
        let mut writer = SerializedFileWriter::new(
            Vec::new(),
            schema,
            Arc::new(WriterProperties::builder().build()),
        )
        .unwrap();
        for entry in document["entries"].as_array().unwrap() {
            let mut group = writer.next_row_group().unwrap();
            for (name, value) in fields(document, entry) {
                let mut column = group.next_column().unwrap().unwrap();
                if name == "latency_ms" {
                    column
                        .typed::<DoubleType>()
                        .write_batch(&[value.as_f64().unwrap()], None, None)
                        .unwrap();
                } else if let Some(value) = value.as_i64() {
                    column
                        .typed::<Int64Type>()
                        .write_batch(&[value], None, None)
                        .unwrap();
                } else {
                    column
                        .typed::<ByteArrayType>()
                        .write_batch(&[ByteArray::from(value.as_str().unwrap())], None, None)
                        .unwrap();
                }
                column.close().unwrap();
            }
            group.close().unwrap();
        }
        writer.into_inner().unwrap().into()
    }

    #[test]
    fn parquet_replacement_keeps_snapshot_latency_and_digest_together() {
        let (_fixture, mut config) = fixture();
        let mut document = fixture_document();
        let file = tempfile::Builder::new()
            .suffix(".parquet")
            .tempfile()
            .unwrap();
        config
            .estimator_config
            .fastafd_stage
            .as_mut()
            .unwrap()
            .profile_path = file.path().into();
        let original = parquet_snapshot(&document);
        std::fs::write(file.path(), &original).unwrap();
        let snapshot = std::fs::read(file.path()).unwrap().into();
        // Replace the path after the loader has captured its immutable input.
        document["entries"][0]["latency_ms"] = json!(24.0);
        let replacement = parquet_snapshot(&document);
        let replacement_file =
            tempfile::NamedTempFile::new_in(file.path().parent().unwrap()).unwrap();
        std::fs::write(replacement_file.path(), &replacement).unwrap();
        replacement_file.persist(file.path()).unwrap();

        let captured = FastAfdTable::from_snapshot(&config, snapshot)
            .unwrap()
            .estimate(128)
            .unwrap();
        assert_eq!(captured.ffn_ms, 2.0); // 12 ms / six pipeline units
        assert_eq!(captured.stage["latency_ms"], 12.0);
        assert_eq!(
            captured.stage["profile_sha256"],
            format!("{:x}", Sha256::digest(&original))
        );
        let refreshed = ForwardPassPerfModel::best_available(config)
            .unwrap()
            .estimate_fastafd_stage(128)
            .unwrap();
        assert_eq!(refreshed.ffn_ms, 4.0); // 24 ms / six pipeline units
        assert_eq!(
            refreshed.stage["profile_sha256"],
            format!("{:x}", Sha256::digest(&replacement))
        );
    }

    #[test]
    fn snapshot_reader_preserves_lfs_and_path_diagnostics() {
        let path = std::path::Path::new("unresolved-stage.parquet");
        let error = PerfReader::from_bytes(
            path,
            bytes::Bytes::from_static(b"version https://git-lfs/spec/v1\n"),
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("unresolved git-lfs pointer"));
        assert!(error.contains("unresolved-stage.parquet"));
        let error = PerfReader::from_bytes(path, bytes::Bytes::from_static(b"not parquet"))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("unresolved-stage.parquet"));
    }

    #[test]
    fn stage_selection_never_falls_back_or_discards_controls() {
        let (_file, base) = fixture();
        let mut invalids = Vec::new();
        let mut cfg = base.clone();
        cfg.worker_type = ForwardPassWorkerType::Prefill;
        invalids.push(cfg);
        let mut cfg = base.clone();
        cfg.fallback_policy = ForwardPassFallbackPolicy::Allow;
        invalids.push(cfg);
        let mut cfg = base.clone();
        cfg.tp = 2;
        invalids.push(cfg);
        let mut cfg = base.clone();
        cfg.afd.as_mut().unwrap().f_moe_ep_size = 4;
        invalids.push(cfg);
        let mut cfg = base.clone();
        cfg.afd.as_mut().unwrap().boundary_on_attn = false;
        invalids.push(cfg);
        let mut cfg = base.clone();
        cfg.estimation_mode = EstimationMode::Auto;
        invalids.push(cfg);
        let mut cfg = base.clone();
        cfg.estimator_config.correction.enabled = false;
        invalids.push(cfg);
        let mut cfg = base.clone();
        cfg.estimator_config.fastafd_stage = None;
        invalids.push(cfg);
        for cfg in invalids {
            assert!(ForwardPassPerfModel::best_available(cfg).is_err());
        }
        // Full-key mismatches are errors rather than alternate precision/model selection.
        for field in ["model", "system", "nextn", "moe_quant_mode"] {
            let mut cfg = base.clone();
            match field {
                "model" => cfg.model = "other".into(),
                "system" => cfg.system = "other".into(),
                "nextn" => cfg.nextn = 1,
                _ => cfg.moe_quant_mode = Some("other".into()),
            }
            assert!(
                ForwardPassPerfModel::best_available(cfg)
                    .and_then(|m| m.estimate_fastafd_stage(128))
                    .is_err()
            );
        }
    }

    #[test]
    fn unqualified_and_duplicate_native_imports_fail_loudly() {
        let (mut file, config) = fixture();
        let original = fixture_document();
        for kind in ["duplicate", "unstable", "bad_digest", "wrong_key"] {
            let mut value = original.clone();
            match kind {
                "duplicate" => {
                    let entry = value["entries"][0].clone();
                    value["entries"].as_array_mut().unwrap().push(entry);
                }
                "unstable" => value["entries"][0]["validation"]["stable"] = json!(false),
                "bad_digest" => value["entries"][0]["measurement"]["raw_sha256"] = json!("bad"),
                _ => value["entries"][0]["key"]["routed_topk"] = json!(0),
            }
            file.as_file_mut().set_len(0).unwrap();
            use std::io::{Seek, SeekFrom};
            file.seek(SeekFrom::Start(0)).unwrap();
            file.write_all(&parquet_snapshot(&value)).unwrap();
            assert!(
                ForwardPassPerfModel::best_available(config.clone()).is_err(),
                "{kind}"
            );
        }
    }
}
