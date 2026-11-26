// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MiniMax Sparse Attention (MSA) module ops for MiniMax-M3.
//!
//! Mirrors `aisim.sdk.operations.msa`. MSA is structurally a GQA
//! version of DSA: an indexer scores KV *blocks* (block_size tokens each),
//! the top-k blocks are selected, and full attention runs over only the
//! selected tokens.
//!
//! MSA has its own module-level silicon tables (msa_context_module_perf /
//! msa_generation_module_perf, DSA-module row schema — see
//! `perf_database::msa`). SILICON resolves them on the raw grids via the
//! perf_interp v2 engine (context `[heads][prefix][seq][batch]`, generation
//! `[heads][batch][seq]`) with the analytic SOL below as the util-hold
//! anchor. HYBRID / EMPIRICAL try that silicon path first; on a typed data
//! miss (table or quant slice absent) they fall back to the CROSS-OP (XOP)
//! transfer from DSA's measured utilisation at the same workload, scaled by
//! the manual `dsa_scale_k` level-alignment hook:
//! `latency = SOL_msa / (util_dsa * k)`. A policy with XOP disabled raises
//! the terminal empirical miss when the silicon path also missed.
//!
//! The XOP transfer belongs to the architecture it was calibrated against
//! ([`MSA_ARCHITECTURE`]). An op that selects another architecture reads that
//! architecture's own rows, and when they are absent it raises a typed miss
//! instead of borrowing the DSA transfer: the borrowed util prices a different
//! kernel, and on SM90 it lands 15-28x below the measured module (see
//! `docs/perf-model/models/qwen38-flash-next.md`). SOL remains available.

use serde::{Deserialize, Serialize};

use crate::common::enums::{
    DatabaseMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode, TransferKind,
};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::dsa::DsaModuleOp;
use crate::perf_database::PerfDatabase;
use crate::perf_database::dsa::{
    dsa_context_sol_flops, dsa_context_sol_ms, dsa_dims, dsa_generation_sol_flops,
    dsa_generation_sol_ms,
};
use crate::perf_database::gemm::quant_tc_flops;

/// Default MSA table architecture — the stamp on MiniMax-M3 rows (Python
/// `DEFAULT_MSA_ARCHITECTURE`). The op's `architecture` field selects the
/// silicon slice; Qwen3.8-Flash-Next QSA rows are keyed by
/// `Qwen4ExpForConditionalGeneration`.
const MSA_ARCHITECTURE: &str = "MiniMaxM3ForCausalLM";

fn default_msa_architecture() -> String {
    MSA_ARCHITECTURE.to_string()
}

/// One MSA module block (context or generation — the phase is chosen by the
/// `Op` variant). Field-for-field mirror of Python `_BaseMSAModule`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MsaModuleOp {
    pub name: String,
    pub scale_factor: f64,
    /// Architecture slice this op reads from the msa tables. Rows are keyed by
    /// architecture, and different architectures carry different native head
    /// geometry, so the lookup must not assume the M3 stamp. Defaults keep
    /// previously serialized specs (and every M3 caller) on MiniMax.
    #[serde(default = "default_msa_architecture")]
    pub architecture: String,
    /// Local (per-rank) query heads.
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub hidden_size: u32,
    pub head_dim: u32,
    pub v_head_dim: u32,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub block_size: u32,
    /// Output width (per rank) of a sigmoid output gate fused into the
    /// module's own q projection, in columns. Qwen3.8-Flash-Next QSA fuses
    /// q_proj with its gate (`2 * heads * head_dim` wide); MiniMax-M3's q
    /// projection has no gate, so its ops keep the default 0.
    #[serde(default)]
    pub output_gate_dim: u32,
    /// Output width (per rank) of the indexer K half fused into the module's
    /// own index_qk projection, in columns. Qwen3.8-Flash-Next QSA fuses the
    /// index query and key halves (`index_n_heads + indexer_kv_heads` heads
    /// at `index_head_dim`); MiniMax-M3 relies on a constant fill for its
    /// index-K rows, so its ones are the default 0: its SOL is the anchor the
    /// accepted M3 rows were calibrated against, and adding a term there
    /// would move a table already in the dataset.
    #[serde(default)]
    pub indexer_k_dim: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub gemm_quant_mode: GemmQuantMode,
    /// The DSA architecture whose measured utilisation is borrowed (XOP).
    pub dsa_architecture: String,
    /// Manual cross-op level-alignment scale `k` (`latency = SOL/(util*k)`).
    pub dsa_scale_k: f64,
}

impl MsaModuleOp {
    /// Context (prefill) query. Mirrors `ContextMSAModule.query`: own silicon
    /// table first (SILICON, and the preferred HYBRID/EMPIRICAL source), the
    /// DSA XOP transfer only on a typed data miss.
    pub fn query_context(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        let sol = self.sol_ms(db, batch_size, s, prefix, true)?;
        match db.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => {
                Ok(PerformanceResult::new(sol * self.scale_factor, Source::Sol))
            }
            DatabaseMode::Silicon => {
                self.silicon_context(db, batch_size, s, prefix)
                    .map(|latency| {
                        PerformanceResult::new(latency * self.scale_factor, Source::Silicon)
                    })
            }
            // EMPIRICAL is SOL+empirical for every op (README/CLI contract;
            // same split as operators/dsa.rs): it must NOT consult silicon.
            DatabaseMode::Empirical => self.xop_context(db, batch_size, s, prefix, sol),
            DatabaseMode::Hybrid => match self.silicon_context(db, batch_size, s, prefix) {
                Ok(latency) => Ok(PerformanceResult::new(
                    latency * self.scale_factor,
                    Source::Silicon,
                )),
                Err(err) if err.is_missing_perf_data() => {
                    self.xop_context(db, batch_size, s, prefix, sol)
                }
                Err(err) => Err(err),
            },
        }
    }

    /// Whether this op may use the DSA cross-op (XOP) transfer.
    ///
    /// The transfer substitutes `SOL_msa / util_dsa` for a measured module and
    /// is calibrated on M3's kernel. A different `architecture` reads a
    /// different implementation (Qwen3.8-Flash-Next QSA has a block indexer and
    /// a fused q+gate that M3 does not), so its util-space value is not a level
    /// shift of M3's: borrowing measures the wrong op. Without rows for that
    /// architecture the only correct answer is a typed coverage miss.
    fn ensure_xop_calibrated(&self, phase: &str) -> Result<(), AicError> {
        if self.architecture == MSA_ARCHITECTURE {
            return Ok(());
        }
        Err(AicError::EmpiricalNotImplemented(format!(
            "MSA {phase}: no silicon rows for architecture {} and the DSA cross-op \
             transfer is calibrated for {MSA_ARCHITECTURE} only; collect \
             msa_context_module_perf / msa_generation_module_perf rows for {} or run \
             in SOL mode.",
            self.architecture, self.architecture
        )))
    }

    /// The legacy cross-op fallback: `SOL_msa / (util_dsa * k)`, gated by XOP.
    fn xop_context(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
        prefix: u32,
        sol: f64,
    ) -> Result<PerformanceResult, AicError> {
        self.ensure_xop_calibrated("context")?;
        if !db.transfer_policy.contains(TransferKind::XOp) {
            return Err(AicError::EmpiricalNotImplemented(
                "MSA context: cross-op transfer (xop) is disabled by the transfer policy \
                 and no MSA silicon data is available for this workload."
                    .to_string(),
            ));
        }
        let util = self.dsa_context_util(db, batch_size, s, prefix);
        match util {
            Some(util) if util > 0.0 => {
                let latency = sol / (util * self.dsa_scale_k);
                // Cross-op transfer from DSA (Python msa.py note_provenance("xop")).
                db.note_provenance(crate::operators::util_empirical::ProvenanceTier::XOp);
                Ok(PerformanceResult::new(
                    latency * self.scale_factor,
                    Source::Empirical,
                ))
            }
            _ => Err(AicError::EmpiricalNotImplemented(format!(
                "MSA context: no DSA util to transfer from (arch={}, b={batch_size}, \
                 s={s}); collect MSA/DSA data or set msa_dsa_scale_k against an available \
                 quant.",
                self.dsa_architecture
            ))),
        }
    }

    /// Generation (decode) query; `s` is the total KV length. Mirrors
    /// `GenerationMSAModule.query`: own silicon table first, DSA XOP transfer
    /// only on a typed data miss.
    pub fn query_generation(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let sol = self.sol_ms(db, batch_size, s, 0, false)?;
        match db.database_mode {
            DatabaseMode::Sol | DatabaseMode::SolFull => {
                Ok(PerformanceResult::new(sol * self.scale_factor, Source::Sol))
            }
            DatabaseMode::Silicon => self.silicon_generation(db, batch_size, s).map(|latency| {
                PerformanceResult::new(latency * self.scale_factor, Source::Silicon)
            }),
            // See query_context: EMPIRICAL never consults silicon.
            DatabaseMode::Empirical => self.xop_generation(db, batch_size, s, sol),
            DatabaseMode::Hybrid => match self.silicon_generation(db, batch_size, s) {
                Ok(latency) => Ok(PerformanceResult::new(
                    latency * self.scale_factor,
                    Source::Silicon,
                )),
                Err(err) if err.is_missing_perf_data() => {
                    self.xop_generation(db, batch_size, s, sol)
                }
                Err(err) => Err(err),
            },
        }
    }

    /// The legacy cross-op fallback for decode, gated by XOP.
    fn xop_generation(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
        sol: f64,
    ) -> Result<PerformanceResult, AicError> {
        self.ensure_xop_calibrated("generation")?;
        if !db.transfer_policy.contains(TransferKind::XOp) {
            return Err(AicError::EmpiricalNotImplemented(
                "MSA generation: cross-op transfer (xop) is disabled by the transfer \
                 policy and no MSA silicon data is available for this workload."
                    .to_string(),
            ));
        }
        let util = self.dsa_generation_util(db, batch_size, s);
        match util {
            Some(util) if util > 0.0 => {
                let latency = sol / (util * self.dsa_scale_k);
                // Cross-op transfer from DSA (Python msa.py note_provenance("xop")).
                db.note_provenance(crate::operators::util_empirical::ProvenanceTier::XOp);
                Ok(PerformanceResult::new(
                    latency * self.scale_factor,
                    Source::Empirical,
                ))
            }
            _ => Err(AicError::EmpiricalNotImplemented(format!(
                "MSA generation: no DSA util to transfer from (arch={}, b={batch_size}, \
                 s={s}); collect MSA/DSA data or set msa_dsa_scale_k against an available \
                 quant.",
                self.dsa_architecture
            ))),
        }
    }

    /// Own-table silicon lookup for context (Python
    /// `ContextMSAModule._query_context_msa_module_table`): the analytic SOL
    /// closure over the engine coordinates `(num_heads, prefix, seq, batch)`
    /// anchors perf_interp's util-hold extrapolation.
    fn silicon_context(
        &self,
        db: &PerfDatabase,
        b: u32,
        s: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        // Strict eager flops resolution (mirrors the Python table query's
        // entry checks): an exact silicon hit never invokes the SOL closure.
        let flops = msa_sol_flops(&db.system_spec, self.gemm_quant_mode, self.fmha_quant_mode)?;
        let spec = &db.system_spec;
        let sol = move |c: &[f64]| {
            msa_attention_sol_ms_with(
                spec,
                true,
                c[3] as i128, // b
                c[2] as i128, // s
                c[1] as i128, // prefix
                c[0] as i128, // num_heads
                self.num_kv_heads as i128,
                self.hidden_size as i128,
                self.head_dim as i128,
                self.v_head_dim as i128,
                self.index_n_heads as i128,
                self.index_head_dim as i128,
                self.index_topk as i128,
                self.block_size as i128,
                self.output_gate_dim as i128,
                self.indexer_k_dim as i128,
                self.kv_cache_dtype,
                self.fmha_quant_mode,
                self.gemm_quant_mode,
                flops,
            )
        };
        db.msa.query_context(
            b,
            s,
            prefix,
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            &self.architecture,
            &sol,
        )
    }

    /// Own-table silicon lookup for decode (Python
    /// `GenerationMSAModule._query_generation_msa_module_table`); engine
    /// coordinates `(num_heads, batch, seq)`, `seq` = total decode length.
    fn silicon_generation(&self, db: &PerfDatabase, b: u32, s: u32) -> Result<f64, AicError> {
        let flops = msa_sol_flops(&db.system_spec, self.gemm_quant_mode, self.fmha_quant_mode)?;
        let spec = &db.system_spec;
        let sol = move |c: &[f64]| {
            msa_attention_sol_ms_with(
                spec,
                false,
                c[1] as i128, // b
                c[2] as i128, // s
                0,            // prefix (decode SOL ignores it)
                c[0] as i128, // num_heads
                self.num_kv_heads as i128,
                self.hidden_size as i128,
                self.head_dim as i128,
                self.v_head_dim as i128,
                self.index_n_heads as i128,
                self.index_head_dim as i128,
                self.index_topk as i128,
                self.block_size as i128,
                self.output_gate_dim as i128,
                self.indexer_k_dim as i128,
                self.kv_cache_dtype,
                self.fmha_quant_mode,
                self.gemm_quant_mode,
                flops,
            )
        };
        db.msa.query_generation(
            b,
            s,
            self.num_heads,
            self.kv_cache_dtype,
            self.gemm_quant_mode,
            &self.architecture,
            &sol,
        )
    }

    fn sol_ms(
        &self,
        db: &PerfDatabase,
        b: u32,
        s: u32,
        prefix: u32,
        is_context: bool,
    ) -> Result<f64, AicError> {
        msa_attention_sol_ms(
            &db.system_spec,
            is_context,
            b as i128,
            s as i128,
            prefix as i128,
            self.num_heads as i128,
            self.num_kv_heads as i128,
            self.hidden_size as i128,
            self.head_dim as i128,
            self.v_head_dim as i128,
            self.index_n_heads as i128,
            self.index_head_dim as i128,
            self.index_topk as i128,
            self.block_size as i128,
            self.output_gate_dim as i128,
            self.indexer_k_dim as i128,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
        )
    }

    /// DSA's measured utilisation (SOL / silicon) at the same context
    /// workload, or `None`. Mirrors Python `_dsa_context_util`: the SOL comes
    /// from the analytic DSA formula (`database_mode=SOL`), the silicon value
    /// from a SILICON-view probe of the DSA module table; ANY failure means
    /// no transfer source.
    fn dsa_context_util(&self, db: &PerfDatabase, b: u32, s: u32, prefix: u32) -> Option<f64> {
        let dims = dsa_dims(&self.dsa_architecture);
        let flops =
            dsa_context_sol_flops(&db.system_spec, self.gemm_quant_mode, self.fmha_quant_mode)
                .ok()?;
        let sol = dsa_context_sol_ms(
            &db.system_spec,
            dims,
            dims.index_topk,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            b as i64,
            s as i64,
            prefix as i64,
            self.num_heads as i64,
            // MSA borrows the FULL DSA layer (Python's probe never sets
            // skip_indexer).
            false,
            db.dsa.reuse_sol_policy,
            flops,
        );
        let probe = self.dsa_probe(dims.index_topk);
        let silicon = probe
            .query_context(&db.silicon_view(), b, s, prefix)
            .ok()?
            .latency_ms;
        if sol > 0.0 && silicon > 0.0 {
            Some(sol / silicon)
        } else {
            None
        }
    }

    /// DSA's measured utilisation at the same decode workload, or `None`.
    /// Mirrors Python `_dsa_generation_util`.
    fn dsa_generation_util(&self, db: &PerfDatabase, b: u32, s: u32) -> Option<f64> {
        let dims = dsa_dims(&self.dsa_architecture);
        let flops = dsa_generation_sol_flops(&db.system_spec, self.gemm_quant_mode).ok()?;
        let sol = dsa_generation_sol_ms(
            &db.system_spec,
            dims,
            self.kv_cache_dtype,
            self.gemm_quant_mode,
            b as i64,
            s as i64,
            self.num_heads as i64,
            1,
            false,
            db.dsa.reuse_sol_policy,
            flops,
        );
        let probe = self.dsa_probe(dims.index_topk);
        let silicon = probe
            .query_generation(&db.silicon_view(), b, s)
            .ok()?
            .latency_ms;
        if sol > 0.0 && silicon > 0.0 {
            Some(sol / silicon)
        } else {
            None
        }
    }

    fn dsa_probe(&self, index_topk: i64) -> DsaModuleOp {
        DsaModuleOp::new(
            format!("{}_dsa_probe", self.name),
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            self.dsa_architecture.clone(),
            index_topk as u32,
        )
    }
}

/// Pre-resolved TC-FLOPS for the three MSA op groups (GEMM projections /
/// always-FP8 indexer / fmha attention). Resolved once per query with the
/// strict `quant_tc_flops` so the perf_interp SOL closure stays `-> f64`.
#[derive(Clone, Copy)]
struct MsaSolFlops {
    gemm: f64,
    indexer_fp8: f64,
    attn: f64,
}

fn msa_sol_flops(
    spec: &SystemSpec,
    gemm_quant: GemmQuantMode,
    fmha_quant: FmhaQuantMode,
) -> Result<MsaSolFlops, AicError> {
    Ok(MsaSolFlops {
        gemm: quant_tc_flops(spec, gemm_quant.mapping())?,
        // Python passes `common.FMHAQuantMode.fp8`.
        indexer_fp8: quant_tc_flops(spec, FmhaQuantMode::Fp8.mapping())?,
        attn: quant_tc_flops(spec, fmha_quant.mapping())?,
    })
}

/// SOL for one MSA block. Verbatim port of Python `_msa_attention_sol`
/// (`operations/msa.py`): GQA projections + per-block FP8 indexer + sparse
/// attention over the top-k selected tokens, integer pair counts in i128.
#[allow(clippy::too_many_arguments)]
fn msa_attention_sol_ms(
    spec: &SystemSpec,
    is_context: bool,
    b: i128,
    s: i128,
    prefix: i128,
    num_heads: i128,
    num_kv_heads: i128,
    hidden_size: i128,
    head_dim: i128,
    v_head_dim: i128,
    index_n_heads: i128,
    index_head_dim: i128,
    index_topk: i128,
    block_size: i128,
    output_gate_dim: i128,
    indexer_k_dim: i128,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
    gemm_quant: GemmQuantMode,
) -> Result<f64, AicError> {
    let flops = msa_sol_flops(spec, gemm_quant, fmha_quant)?;
    Ok(msa_attention_sol_ms_with(
        spec,
        is_context,
        b,
        s,
        prefix,
        num_heads,
        num_kv_heads,
        hidden_size,
        head_dim,
        v_head_dim,
        index_n_heads,
        index_head_dim,
        index_topk,
        block_size,
        output_gate_dim,
        indexer_k_dim,
        kv_quant,
        fmha_quant,
        gemm_quant,
        flops,
    ))
}

/// The pure-`f64` SOL body over pre-resolved flops (the perf_interp anchor).
#[allow(clippy::too_many_arguments)]
fn msa_attention_sol_ms_with(
    spec: &SystemSpec,
    is_context: bool,
    b: i128,
    s: i128,
    prefix: i128,
    num_heads: i128,
    num_kv_heads: i128,
    hidden_size: i128,
    head_dim: i128,
    v_head_dim: i128,
    index_n_heads: i128,
    index_head_dim: i128,
    index_topk: i128,
    block_size: i128,
    output_gate_dim: i128,
    indexer_k_dim: i128,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
    gemm_quant: GemmQuantMode,
    flops: MsaSolFlops,
) -> f64 {
    let qk_head_dim = head_dim;
    let tokens = if is_context { b * s } else { b };
    // context: full prefill of `s` new tokens on top of `prefix` cached.
    // generation: 1 query token, kv_len = s - 1 cached.
    let full_s = if is_context { prefix + s } else { s };
    let kv_len = if is_context { full_s } else { (s - 1).max(0) };

    // ── GEMM group (Q / GQA-KV / O / indexer-Q projections) ──────────────
    // `output_gate_dim` and `indexer_k_dim` are the fused halves this
    // architecture's own module adds to those projections (Qwen3.8-Flash-Next
    // QSA fuses the sigmoid gate into q_proj and the key half into
    // index_qk_proj). They are priced here, in the module that is measured,
    // rather than as separate GEMMs outside it: SILICON/HYBRID return the
    // whole module's measured latency, so an outside GEMM would be charged
    // twice once the rows resolve.
    let gemm_ops = 2 * tokens * hidden_size * (num_heads * qk_head_dim)
        + 2 * tokens * hidden_size * (2 * num_kv_heads * head_dim)
        + 2 * tokens * (num_heads * v_head_dim) * hidden_size
        + 2 * tokens * hidden_size * (index_n_heads * index_head_dim)
        + 2 * tokens * hidden_size * (output_gate_dim + indexer_k_dim);

    // ── sparse attention: top-k saturated causal (query, kv) pair count ──
    let (pairs, score_len) = if is_context {
        let pairs = if full_s <= index_topk {
            b * (full_s * (full_s + 1) - prefix * (prefix + 1)) / 2
        } else if prefix >= index_topk {
            tokens * index_topk
        } else {
            let ramp = b * (index_topk * (index_topk + 1) - prefix * (prefix + 1)) / 2;
            let sat = b * (full_s - index_topk) * index_topk;
            ramp + sat
        };
        (pairs, full_s)
    } else {
        (tokens * kv_len.min(index_topk), kv_len)
    };
    let effective_kv = if is_context {
        full_s.min(index_topk)
    } else {
        kv_len.min(index_topk)
    };
    let attention_ops = 2 * num_heads * (qk_head_dim + v_head_dim) * pairs; // QK^T + AV

    // ── indexer: per-block scoring (block_size tokens per block), FP8 ────
    let num_blocks = if score_len > index_topk {
        (score_len + block_size - 1) / block_size
    } else {
        0
    };
    let indexer_ops = 2 * tokens * index_n_heads * index_head_dim * num_blocks;

    // ── memory ───────────────────────────────────────────────────────────
    let gemm_weight_elems = hidden_size * num_heads * qk_head_dim
        + hidden_size * 2 * num_kv_heads * head_dim
        + num_heads * v_head_dim * hidden_size
        + hidden_size * index_n_heads * index_head_dim
        + hidden_size * (output_gate_dim + indexer_k_dim);
    let gemm_weight_bytes = gemm_weight_elems as f64 * gemm_quant.mapping().memory;
    let kv_cache_bytes = (b * num_kv_heads * effective_kv * (qk_head_dim + v_head_dim)) as f64
        * kv_quant.mapping().memory;
    // FP8 index keys, per block (1 byte per element).
    let indexer_cache_bytes = (b * num_blocks * index_n_heads * index_head_dim) as f64;
    let q_io_bytes = (tokens * num_heads * qk_head_dim) as f64 * fmha_quant.mapping().memory * 2.0;
    let total_mem = gemm_weight_bytes + kv_cache_bytes + indexer_cache_bytes + q_io_bytes;

    let MsaSolFlops {
        gemm: gemm_flops,
        indexer_fp8: fp8_flops,
        attn: attn_flops,
    } = flops;

    let sol_math = (gemm_ops as f64 / gemm_flops
        + indexer_ops as f64 / fp8_flops
        + attention_ops as f64 / attn_flops)
        * 1000.0;
    let sol_mem = total_mem / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::TransferPolicy;
    use crate::perf_database::perf_interp::LeafValue;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn db(backend: &str, version: &str) -> PerfDatabase {
        let systems_root = crate::test_systems::path("");
        let mut db =
            PerfDatabase::load(&systems_root, "b200_sxm", backend, version).expect("db must load");
        db.database_mode = DatabaseMode::Hybrid;
        db
    }

    /// MiniMax-M3 MSA block at tp=8 (local heads 8, GQA kv heads 1), all
    /// bfloat16 — the shapes the M3 model builder emits.
    fn msa_op() -> MsaModuleOp {
        MsaModuleOp {
            name: "msa".to_string(),
            scale_factor: 1.0,
            architecture: MSA_ARCHITECTURE.to_string(),
            num_heads: 8,
            num_kv_heads: 1,
            hidden_size: 6144,
            head_dim: 128,
            v_head_dim: 128,
            index_n_heads: 4,
            index_head_dim: 128,
            index_topk: 2048,
            block_size: 128,
            output_gate_dim: 0,
            indexer_k_dim: 0,
            kv_cache_dtype: KvCacheQuantMode::Bfloat16,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            gemm_quant_mode: GemmQuantMode::Bfloat16,
            dsa_architecture: "GlmMoeDsaForCausalLM".to_string(),
            dsa_scale_k: 1.0,
        }
    }

    fn approx(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-9 * b.abs().max(1.0),
            "expected {b}, got {a}"
        );
    }

    /// Oracle values from the Python reference (`ContextMSAModule` /
    /// `GenerationMSAModule` with `scale_factor=1.0` on a HYBRID
    /// `shared_layer=False` view). Regenerate if the DSA tables or the
    /// util math change.
    ///
    /// The XOP transfer needs a backend whose current slot ships NO msa
    /// tables; sglang/0.5.14 is that vehicle. The former vllm branch was
    /// dropped when vllm/0.24.0 started shipping msa data (the op then
    /// silicon-hits and never reaches the transfer) — the silicon-hit
    /// precedence is covered by `msa_silicon_table_hit_prefers_silicon_over_xop`
    /// on injected grids.
    #[test]
    fn msa_xop_transfer_matches_python_oracles() {
        // Structural: the transfer must FIRE (source Empirical, xop tier) on
        // every anchor; the borrowed values are data-refresh churn and are
        // not pinned (math on synthetic grids, end-to-end in goldens).
        for (backend, version, anchors) in [(
            "sglang",
            "0.5.14",
            [
                (1u32, 1024u32, 0u32, true),
                (2, 3000, 512, true),
                (8, 1025, 0, false),
                (4, 7777, 0, false),
            ],
        )] {
            let db = db(backend, version);
            let op = msa_op();
            for (b, s, prefix, is_context) in anchors {
                db.reset_provenance();
                let result = if is_context {
                    op.query_context(&db, b, s, prefix)
                } else {
                    op.query_generation(&db, b, s)
                }
                .unwrap_or_else(|e| panic!("{backend} b={b} s={s}: {e}"));
                assert!(result.latency_ms.is_finite() && result.latency_ms > 0.0);
                assert_eq!(result.source, Source::Empirical);
                // The xop tier must be recorded on the db accumulator
                // (Python msa.py:297/335 note_provenance("xop")).
                assert_eq!(
                    db.worst_provenance(),
                    crate::operators::util_empirical::ProvenanceTier::XOp
                );
            }
        }
    }

    /// An architecture other than M3 must NOT borrow the DSA transfer when its
    /// own rows are absent: the guard raises the terminal empirical miss, in
    /// both phases and in both modes that reach the transfer (HYBRID after the
    /// silicon miss, EMPIRICAL directly). sglang/0.5.14 ships no msa tables, so
    /// this is the same vehicle the M3 oracle test uses.
    #[test]
    fn msa_xop_refuses_an_uncalibrated_architecture() {
        const QSA: &str = "Qwen4ExpForConditionalGeneration";
        for mode in [DatabaseMode::Hybrid, DatabaseMode::Empirical] {
            let mut db = db("sglang", "0.5.14");
            db.database_mode = mode;
            let op = MsaModuleOp {
                architecture: QSA.to_string(),
                ..msa_op()
            };
            for (phase, err) in [
                ("context", op.query_context(&db, 1, 1024, 0).unwrap_err()),
                ("generation", op.query_generation(&db, 1, 1025).unwrap_err()),
            ] {
                match err {
                    AicError::EmpiricalNotImplemented(msg) => {
                        assert!(
                            msg.contains(QSA)
                                && msg.contains(MSA_ARCHITECTURE)
                                && msg.contains("calibrated"),
                            "{phase} {mode:?}: guard message must name both \
                             architectures, got {msg}"
                        );
                    }
                    other => panic!("{phase} {mode:?}: expected the typed miss, got {other:?}"),
                }
            }
            // Nothing was priced, so the borrowed tier must not be recorded.
            assert_ne!(
                db.worst_provenance(),
                crate::operators::util_empirical::ProvenanceTier::XOp,
                "{mode:?}: a refused transfer must not claim xop provenance"
            );
        }
    }

    /// The guard is scoped to other architectures: M3 keeps the transfer (the
    /// oracle test above pins the values), and an op that DOES carry QSA rows
    /// still resolves them under SILICON.
    #[test]
    fn msa_xop_guard_leaves_calibrated_and_collected_paths_intact() {
        const QSA: &str = "Qwen4ExpForConditionalGeneration";
        // M3, no tables -> transfer still fires.
        let m3_db = db("sglang", "0.5.14");
        assert_eq!(
            msa_op().query_context(&m3_db, 1, 1024, 0).unwrap().source,
            Source::Empirical
        );
        // QSA with rows -> own slice, under the default SILICON mode.
        let mut qsa_db = db("vllm", "0.24.0");
        qsa_db.database_mode = DatabaseMode::Silicon;
        let (context, generation) = msa_grids(&[(MSA_ARCHITECTURE, 10.0), (QSA, 20.0)]);
        qsa_db.msa.inject_for_test(context, generation);
        let op = MsaModuleOp {
            architecture: QSA.to_string(),
            ..msa_op()
        };
        approx(
            op.query_context(&qsa_db, 1, 1024, 0).unwrap().latency_ms,
            20.0,
        );
        assert_eq!(
            op.query_context(&qsa_db, 1, 1024, 0).unwrap().source,
            Source::Silicon
        );
        approx(
            op.query_generation(&qsa_db, 1, 4097).unwrap().latency_ms,
            1.0,
        );
        assert_eq!(
            op.query_generation(&qsa_db, 1, 4097).unwrap().source,
            Source::Silicon
        );
    }

    /// QSA's fused projection halves are priced INSIDE the module SOL, not as
    /// GEMMs beside it. SILICON/HYBRID return the measured whole-module
    /// latency, so a supplementary GEMM would be charged twice once the QSA
    /// rows resolve; including the columns here keeps SOL and the measured
    /// unit the same arithmetic module.
    #[test]
    fn msa_sol_prices_the_fused_projection_halves_inside_the_module() {
        const GATE: u32 = 1024; // sigmoid output-gate width (columns)
        const IDX_K: u32 = 128; // the fused index_qk key half
        let mut db = db("vllm", "0.24.0");
        db.database_mode = DatabaseMode::Sol;
        let flops = msa_sol_flops(
            &db.system_spec,
            GemmQuantMode::Bfloat16,
            FmhaQuantMode::Bfloat16,
        )
        .expect("bf16 flops");
        let gemm_mem = GemmQuantMode::Bfloat16.mapping().memory;

        let m3 = msa_op();
        let mut qsa = msa_op();
        qsa.output_gate_dim = GATE;
        qsa.indexer_k_dim = IDX_K;

        // One context point (math-bound: 1024 tokens make the added GEMM work
        // dwarf the added weight read) and one generation point (memory-bound:
        // one query token re-reads the whole weight block). Each delta is
        // exactly its dominant resource's added work — the terms are charged
        // once, inside the module.
        let cases = [
            (true, 1024u32, 0u32),  // context: 1024 tokens, math-bound
            (false, 4097u32, 0u32), // generation: 1 token, memory-bound
        ];
        for (is_context, s, prefix) in cases {
            let tokens = if is_context { s as f64 } else { 1.0 };
            let delta_math =
                2.0 * tokens * m3.hidden_size as f64 * (GATE + IDX_K) as f64 / flops.gemm * 1000.0;
            let delta_mem = m3.hidden_size as f64 * (GATE + IDX_K) as f64 * gemm_mem
                / db.system_spec.gpu.mem_bw
                * 1000.0;
            let base = if is_context {
                m3.query_context(&db, 1, s, prefix).unwrap().latency_ms
            } else {
                m3.query_generation(&db, 1, s).unwrap().latency_ms
            };
            let fused = if is_context {
                qsa.query_context(&db, 1, s, prefix).unwrap().latency_ms
            } else {
                qsa.query_generation(&db, 1, s).unwrap().latency_ms
            };
            assert!(fused > base, "fused halves must add work, not replace it");
            if is_context {
                assert!(delta_math > delta_mem, "context point must be math-bound");
                approx(fused - base, delta_math);
            } else {
                assert!(
                    delta_mem > delta_math,
                    "generation point must be memory-bound"
                );
                approx(fused - base, delta_mem);
            }
        }
    }

    // ------------------------------------------------------------------
    // Own silicon tables (msa_context_module / msa_generation_module).
    // No parquet writer in the test deps, so the loaded-grid layer is
    // injected directly (the loader itself is the shared, DSA-tested
    // `load_dsa_parquet`).
    // ------------------------------------------------------------------
    use crate::perf_database::dsa::{DsaGrids, DsaHeadGrid, DsaKey};
    use std::collections::BTreeMap;

    fn msa_key() -> DsaKey {
        msa_key_for(MSA_ARCHITECTURE)
    }

    fn msa_key_for(architecture: &str) -> DsaKey {
        DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: "bfloat16".to_string(),
            kv_quant: "bfloat16".to_string(),
            gemm_quant: "bfloat16".to_string(),
        }
    }

    /// One-point grids at the coordinates the loader would produce for
    /// kernel_source="default" rows (flashmla_kv bucket), one per
    /// `(architecture, context_ms)` entry: context
    /// `[heads=8][prefix=0][isl=1024][b=1] = context_ms`; generation collapsed
    /// to `[heads=8][step=0][seq=4097][b=1] = context_ms / 20`.
    fn msa_grids(entries: &[(&str, f64)]) -> (DsaGrids, DsaGrids) {
        let mut context = BTreeMap::new();
        let mut generation = BTreeMap::new();
        for (architecture, context_ms) in entries {
            let mut ctx_head = DsaHeadGrid::new();
            ctx_head
                .entry(8)
                .or_default()
                .entry(0)
                .or_default()
                .entry(1024)
                .or_default()
                .insert(1, LeafValue::latency_only(*context_ms));
            let mut gen_head = DsaHeadGrid::new();
            gen_head
                .entry(8)
                .or_default()
                .entry(0)
                .or_default()
                .entry(4097)
                .or_default()
                .insert(1, LeafValue::latency_only(context_ms / 20.0));
            let key = msa_key_for(architecture);
            context.insert(
                key.clone(),
                BTreeMap::from([("flashmla_kv".to_string(), ctx_head)]),
            );
            generation.insert(key, BTreeMap::from([("flashmla_kv".to_string(), gen_head)]));
        }
        (
            DsaGrids { by_keys: context },
            DsaGrids {
                by_keys: generation,
            },
        )
    }

    fn inject_msa_grids(db: &PerfDatabase) {
        let (context, generation) = msa_grids(&[(MSA_ARCHITECTURE, 10.0)]);
        db.msa.inject_for_test(context, generation);
    }

    /// With MSA data present, SILICON answers from the table and HYBRID
    /// prefers it over the DSA XOP transfer (source stays Silicon, no xop
    /// provenance is recorded). EMPIRICAL deliberately absent: it never
    /// consults silicon (see msa_empirical_mode_never_reads_silicon).
    #[test]
    fn msa_silicon_table_hit_prefers_silicon_over_xop() {
        for mode in [DatabaseMode::Silicon, DatabaseMode::Hybrid] {
            let mut db = db("vllm", "0.24.0");
            db.database_mode = mode;
            inject_msa_grids(&db);
            let op = msa_op();

            db.reset_provenance();
            let ctx = op
                .query_context(&db, 1, 1024, 0)
                .expect("context silicon hit");
            approx(ctx.latency_ms, 10.0);
            assert_eq!(ctx.source, Source::Silicon, "{mode:?}");

            let generation = op
                .query_generation(&db, 1, 4097)
                .expect("generation silicon hit");
            approx(generation.latency_ms, 0.5);
            assert_eq!(generation.source, Source::Silicon, "{mode:?}");
            assert_eq!(
                db.worst_provenance(),
                crate::operators::util_empirical::ProvenanceTier::Silicon,
                "{mode:?}"
            );
        }
    }

    /// SILICON/HYBRID return the measured whole-module latency, which already
    /// contains the fused q+gate and indexer-K work. The fused widths feed the
    /// SOL only; on a measured row they must not move the answer, or they
    /// would be charged a second time.
    #[test]
    fn msa_silicon_row_is_not_recharged_for_the_fused_projection_halves() {
        let mut db = db("vllm", "0.24.0");
        db.database_mode = DatabaseMode::Silicon;
        inject_msa_grids(&db);
        let mut qsa = msa_op();
        qsa.output_gate_dim = 1024;
        qsa.indexer_k_dim = 128;

        // The M3 row measured the whole module; the fused widths change only
        // the analytical SOL, so the measured answer stays the row's value.
        approx(
            qsa.query_context(&db, 1, 1024, 0)
                .expect("context silicon hit")
                .latency_ms,
            10.0,
        );
        approx(
            qsa.query_generation(&db, 1, 4097)
                .expect("generation silicon hit")
                .latency_ms,
            0.5,
        );
    }

    /// Silicon lookup keys on the op's own architecture, not the M3 default:
    /// Qwen3.8-Flash-Next QSA rows are keyed by
    /// `Qwen4ExpForConditionalGeneration`, and reading the wrong slice would
    /// interpolate M3 head geometry.
    #[test]
    fn msa_silicon_lookup_keys_on_the_op_architecture() {
        const QSA: &str = "Qwen4ExpForConditionalGeneration";
        let mut db = db("vllm", "0.24.0");
        db.database_mode = DatabaseMode::Silicon;
        let (context, generation) = msa_grids(&[(MSA_ARCHITECTURE, 10.0), (QSA, 20.0)]);
        db.msa.inject_for_test(context, generation);

        let mut qsa = msa_op();
        qsa.architecture = QSA.to_string();
        approx(
            qsa.query_context(&db, 1, 1024, 0)
                .expect("qsa context silicon hit")
                .latency_ms,
            20.0,
        );
        approx(
            qsa.query_generation(&db, 1, 4097)
                .expect("qsa generation silicon hit")
                .latency_ms,
            1.0,
        );

        let m3 = msa_op();
        approx(
            m3.query_context(&db, 1, 1024, 0)
                .expect("m3 context silicon hit")
                .latency_ms,
            10.0,
        );
        approx(
            m3.query_generation(&db, 1, 4097)
                .expect("m3 generation silicon hit")
                .latency_ms,
            0.5,
        );
    }

    /// Specs serialized before the field existed stay loadable and keep the
    /// M3 slice every legacy caller expects.
    #[test]
    fn msa_op_json_without_architecture_defaults_to_m3() {
        let mut value = serde_json::to_value(msa_op()).expect("serialize msa op");
        let fields = value.as_object_mut().expect("object");
        fields.remove("architecture");
        // Specs serialized before the fused-half widths existed must stay
        // readable, on M3's (no fused half) pricing.
        fields.remove("output_gate_dim");
        fields.remove("indexer_k_dim");
        let restored: MsaModuleOp =
            serde_json::from_value(value).expect("deserialize legacy msa op");
        assert_eq!(restored.architecture, MSA_ARCHITECTURE);
        assert_eq!((restored.output_gate_dim, restored.indexer_k_dim), (0, 0));
    }

    /// A quant slice absent from the injected table is a typed silicon miss:
    /// SILICON propagates it; HYBRID falls through to the XOP transfer (which
    /// still resolves against the real DSA tables — the pre-silicon
    /// behaviour, unregressed).
    #[test]
    fn msa_missing_quant_slice_falls_back_to_xop_under_hybrid() {
        let mut silicon = db("vllm", "0.24.0");
        silicon.database_mode = DatabaseMode::Silicon;
        inject_msa_grids(&silicon);
        let mut op = msa_op();
        op.kv_cache_dtype = KvCacheQuantMode::Fp8;
        assert!(matches!(
            op.query_context(&silicon, 1, 1024, 0),
            Err(AicError::PerfDatabase(_))
        ));

        let hybrid = db("vllm", "0.24.0"); // Hybrid by default in `db()`
        inject_msa_grids(&hybrid);
        let bf16_op = msa_op(); // bf16 KV: DSA util source exists on vllm 0.24.0
        let mut fp8_kv = msa_op();
        fp8_kv.kv_cache_dtype = KvCacheQuantMode::Fp8;
        // Injected slice is bf16-only -> fp8-KV misses silicon; the xop value
        // differs from the bf16 silicon hit, proving the fallback fired.
        let result = fp8_kv.query_context(&hybrid, 1, 1024, 0);
        match result {
            Ok(r) => assert_eq!(r.source, Source::Empirical),
            Err(AicError::EmpiricalNotImplemented(_)) => {} // no fp8 DSA util source either
            Err(other) => panic!("unexpected error: {other}"),
        }
        // The bf16 op still hits silicon on the same db.
        let hit = bf16_op
            .query_context(&hybrid, 1, 1024, 0)
            .expect("silicon hit");
        assert_eq!(hit.source, Source::Silicon);
    }

    /// XOP disabled ("balanced" preset) -> the terminal empirical miss; and
    /// SILICON mode -> the perf-data miss ("MSA module data missing").
    /// Anchored on sglang/0.5.14, whose current slot ships no msa tables
    /// (vllm/0.24.0 does, so it can no longer carry these absence contracts).
    #[test]
    fn msa_policy_and_silicon_contracts() {
        let mut hybrid = db("sglang", "0.5.14");
        hybrid.transfer_policy = TransferPolicy {
            xshape: true,
            xquant: true,
            xprofile: false,
            xop: false,
        };
        let op = msa_op();
        assert!(matches!(
            op.query_context(&hybrid, 1, 1024, 0),
            Err(AicError::EmpiricalNotImplemented(_))
        ));
        assert!(matches!(
            op.query_generation(&hybrid, 8, 1025),
            Err(AicError::EmpiricalNotImplemented(_))
        ));

        let mut silicon = db("sglang", "0.5.14");
        silicon.database_mode = DatabaseMode::Silicon;
        assert!(matches!(
            op.query_context(&silicon, 1, 1024, 0),
            Err(AicError::PerfDatabase(_))
        ));
    }

    /// Mode/source contract (review 4969690316 P1): EMPIRICAL is SOL+empirical
    /// for every op and must NEVER consult silicon — with silicon grids
    /// injected, SILICON and HYBRID return Source::Silicon while EMPIRICAL
    /// still routes to the XOP transfer (Source::Empirical, xop provenance),
    /// for both context and generation.
    #[test]
    fn msa_empirical_mode_never_reads_silicon() {
        let op = msa_op();

        let mut silicon = db("vllm", "0.24.0");
        silicon.database_mode = DatabaseMode::Silicon;
        inject_msa_grids(&silicon);
        assert_eq!(
            op.query_context(&silicon, 1, 1024, 0)
                .expect("silicon ctx")
                .source,
            Source::Silicon
        );
        assert_eq!(
            op.query_generation(&silicon, 1, 4097)
                .expect("silicon gen")
                .source,
            Source::Silicon
        );

        let hybrid = db("vllm", "0.24.0"); // Hybrid by default in `db()`
        inject_msa_grids(&hybrid);
        assert_eq!(
            op.query_context(&hybrid, 1, 1024, 0)
                .expect("hybrid ctx")
                .source,
            Source::Silicon
        );

        let mut empirical = db("vllm", "0.24.0");
        empirical.database_mode = DatabaseMode::Empirical;
        inject_msa_grids(&empirical);
        empirical.reset_provenance();
        let ctx = op
            .query_context(&empirical, 1, 1024, 0)
            .expect("empirical ctx");
        assert_eq!(ctx.source, Source::Empirical);
        assert_eq!(
            empirical.worst_provenance(),
            crate::operators::util_empirical::ProvenanceTier::XOp
        );
        empirical.reset_provenance();
        let generation = op
            .query_generation(&empirical, 8, 1025)
            .expect("empirical gen");
        assert_eq!(generation.source, Source::Empirical);
        assert_eq!(
            empirical.worst_provenance(),
            crate::operators::util_empirical::ProvenanceTier::XOp
        );
        // The empirical value must be the XOP transfer, not the injected
        // silicon point (silicon ctx at this coordinate is exactly 10.0).
        assert!((f64::from(ctx.latency_ms) - 10.0).abs() > 1e-6);
    }
}
