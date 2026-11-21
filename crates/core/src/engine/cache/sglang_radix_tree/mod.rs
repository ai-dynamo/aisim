// SPDX-FileCopyrightText: SGLang contributors
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Derived from sgl-project/sglang at f0fda25fc95c357368b714f90a13a69c1b67f50a.
// See README.md for original paths, license, and documented modifications.

//! Native-only Rust radix-tree core for SGLang's KV cache.
//! The bundle omits Tensor implementations and Torch/PyO3 modules; enabling
//! AISimulate's Python feature does not provide those backends.
// TODO(Jialin): Replace recoverable panics with explicit Rust errors and map
// them to PyErr at the Python boundary.
#![allow(
    dead_code,
    unused_imports,
    unsafe_op_in_unsafe_fn,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::owned_cow,
    clippy::panic,
    clippy::print_stdout,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::useless_conversion
)]

mod components;
mod node;
mod unified_lru_list;
mod unified_tree_core;
mod value;

pub use components::{ComponentSet, ComponentType, FULL, MAMBA, SWA};
pub use node::{
    ChildKeyType, KeyNamespace, KeyNamespaceRef, NodeAccessError, NodeId, TreeCoreRuntimeError,
};
pub use unified_tree_core::{
    BackupKV, BufferBackupSnapshot, BufferBackupState, CacheAction, CacheInitParams,
    CacheTransferPhase, DecLockRefParams, DecLockRefResult, EvictLayer, EvictionStepResult,
    FullDeviceEvictionCandidate, FullInsertContinuation, IncLockRefResult, InsertParams,
    InsertResult, InsertStepResult, KvCacheEvent, KvCanaryWalkResult, MatchPrefixParams,
    MatchResult, PoolHitPolicy, PoolName, PoolTransfer, PoolTransferResult, Req, StorageBackupSpec,
    StorageMedium, UnifiedTreeCore,
};
pub use value::{PageValue, RadixValue};

#[cfg(test)]
mod page_value_core;

#[cfg(test)]
#[path = "../sglang_radix_regressions.rs"]
mod repair_regressions;
