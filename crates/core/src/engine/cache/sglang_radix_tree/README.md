# Bundled SGLang native radix core

Source: <https://github.com/sgl-project/sglang/tree/f0fda25fc95c357368b714f90a13a69c1b67f50a/rust/sglang-radix-tree>

Revision: `f0fda25fc95c357368b714f90a13a69c1b67f50a`.
Copyright SGLang contributors. Licensed under Apache-2.0; the upstream
repository's license is retained in [LICENSE](LICENSE). The upstream repository
has no root NOTICE file. NVIDIA modifications are identified in each source
header and listed below. The canonical repository notices also identify this
copy in `THIRD_PARTY_NOTICES.md`.

The upstream native core is bundled as a private module of `aisimulate-core`.
There is no separately published or buildable vendor crate. Cargo cannot package
the original Git-only dependency, even when that dependency is optional. Keeping
the sources inside the existing crate preserves its registry release path and
the two-artifact product contract while the upstream crate is unpublished.

## Original paths and modifications

All paths below are relative to upstream `rust/sglang-radix-tree/`:

| Bundled file | Original file |
| --- | --- |
| `mod.rs` | `src/lib.rs` |
| `node.rs` | `src/node.rs` |
| `unified_lru_list.rs` | `src/unified_lru_list.rs` |
| `unified_tree_core.rs` | `src/unified_tree_core.rs` |
| `value.rs` | `src/value.rs` |
| `components/*.rs` | `src/components/*.rs` |
| `page_value_core.rs` | `tests/page_value_core.rs` |

The mechanical modifications are:

- Add attribution headers and qualify upstream `crate::` paths with
  `crate::engine::cache::sglang_radix_tree::`.
- Alias upstream's `thiserror` 1.x as `thiserror1`, preserving the existing
  AISimulate dependency on `thiserror` 2.x. Keep upstream's `hashbrown` 0.16
  and Linux/aarch64 SHA acceleration dependencies.
- Omit the disabled Torch/PyO3 modules, Tensor implementations, and Torch-only
  test module declarations. Compile upstream's native integration tests as a
  unit-test module instead; retain its inline native value tests.
- Change inspection's `cfg(any(test, feature = "inspection"))` to `cfg(test)`.
  It remains absent from production builds and available to adapter tests.
- Allow unused upstream exports within the private module and apply Rustfmt.

No tree algorithms, data representations, runtime checks, or test assertions
are changed. This packaging adaptation adds no work to cache operations.

## Updating

Start from a verified immutable upstream checkout, preserve its license and
notices, and use `scripts/sync_sglang_radix_core.py` to reproduce the documented
mechanical adaptations. The script reads committed source at the pinned revision
without changing the upstream checkout. Verify an existing copy with:

```bash
python3 scripts/sync_sglang_radix_core.py --source-checkout /path/to/sglang --check
```

For a deliberate version update, update the script's revision, regenerate without
`--check`, and update this revision and both canonical notice copies together.
Keep the native tests, AISimulate cache/manager/scheduler regressions, and
`cargo package --manifest-path crates/core/Cargo.toml --locked` passing.
Changes to upstream behavior require separate behavioral qualification.
