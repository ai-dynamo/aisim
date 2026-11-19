// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

fn rust_sources(root: &Path, sources: &mut Vec<PathBuf>) {
    assert!(
        root.is_dir(),
        "source firewall root does not exist: {}",
        root.display()
    );
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, sources);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            sources.push(path);
        }
    }
}

#[test]
fn replay_core_has_no_runtime_adapter_dependencies() {
    const FORBIDDEN: &[&str] = &[
        "crate::replay::loadgen",
        "crate::engine::scheduler",
        "OfflineWorkerState",
        "WorkloadDriver",
        "ReplayRouterMode",
        "extensions::",
    ];

    let core = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/replay/core");
    let mut sources = Vec::new();
    rust_sources(&core, &mut sources);
    assert!(
        !sources.is_empty(),
        "source firewall found no Rust sources under {}",
        core.display()
    );
    for path in sources {
        let source = fs::read_to_string(&path).unwrap();
        for forbidden in FORBIDDEN {
            assert!(
                !source.contains(forbidden),
                "{} crosses the replay core firewall with `{forbidden}`",
                path.display()
            );
        }
    }
}

#[test]
fn engine_has_no_replay_dependencies() {
    let engine = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/engine");
    let mut sources = Vec::new();
    rust_sources(&engine, &mut sources);
    assert!(
        !sources.is_empty(),
        "engine firewall found no Rust sources under {}",
        engine.display()
    );
    for path in sources {
        let source = fs::read_to_string(&path).unwrap();
        assert!(
            !source.contains("crate::replay"),
            "{} violates the engine <- replay dependency direction",
            path.display()
        );
    }
}

/// Existing `src/perfmodel` references into `crate::engine`, per file. Lower an
/// entry, or drop it at zero, in the same change that removes a reference.
const PERFMODEL_ENGINE_REFERENCES: &[(&str, usize)] = &[
    ("afd/model.rs", 5),
    ("fpm/snapshot/tests.rs", 1),
    ("py.rs", 1),
];

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn leading_token(path: &str) -> &str {
    let path = path.trim_start();
    if path.starts_with('*') {
        return "*";
    }
    &path[..path.find(|c| !is_ident(c)).unwrap_or(path.len())]
}

fn group_items(group: &str) -> Vec<&str> {
    let (mut items, mut depth, mut start) = (Vec::new(), 0, 0);
    for (at, c) in group.char_indices() {
        match c {
            '{' => depth += 1,
            '}' if depth == 0 => {
                items.push(&group[start..at]);
                break;
            }
            '}' => depth -= 1,
            ',' if depth == 0 => {
                items.push(&group[start..at]);
                start = at + 1;
            }
            _ => {}
        }
    }
    items
}

fn crate_root_engine_reexports(lib: &str) -> Vec<String> {
    let mut names = Vec::new();
    for statement in lib.split(';') {
        let Some((_, path)) = statement.split_once("use ") else {
            continue;
        };
        let path = path.trim().trim_start_matches("crate::");
        let Some(items) = path.strip_prefix("engine::") else {
            continue;
        };
        for item in items.split([',', '{', '}']) {
            let name = item.rsplit_once(" as ").map_or(item, |(_, alias)| alias);
            let name = name.rsplit("::").next().unwrap().trim();
            if !name.is_empty() {
                names.push(name.to_string());
            }
        }
    }
    names
}

/// Counts `crate::engine`, grouped `crate::{engine…}`, `crate::*`, crate-root
/// re-exports of engine items, and every `super::…::engine` chain.
fn engine_references(code: &str, crate_root_names: &[String]) -> usize {
    let mut count = 0;
    for prefix in ["crate::", "super::"] {
        let named = |path: &str| {
            let token = leading_token(path);
            token == "engine"
                || (prefix == "crate::"
                    && (token == "*" || crate_root_names.iter().any(|name| name == token)))
        };
        let mut rest = code;
        while let Some(at) = rest.find(prefix) {
            let bounded = !rest[..at].ends_with(is_ident);
            rest = &rest[at + prefix.len()..];
            while let Some(tail) = rest.strip_prefix(prefix) {
                rest = tail;
            }
            if bounded {
                count += match rest.strip_prefix('{') {
                    Some(group) => group_items(group)
                        .into_iter()
                        .filter(|item| named(item))
                        .count(),
                    None => usize::from(named(rest)),
                };
            }
        }
    }
    count
}

fn without_line_comments(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn perfmodel_engine_references_only_shrink() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let lib = without_line_comments(&fs::read_to_string(src.join("lib.rs")).unwrap());
    let crate_root_names = crate_root_engine_reexports(&lib);
    let perfmodel = src.join("perfmodel");
    let mut sources = Vec::new();
    rust_sources(&perfmodel, &mut sources);
    assert!(
        !sources.is_empty(),
        "perfmodel firewall found no Rust sources under {}",
        perfmodel.display()
    );

    let mut found = BTreeMap::new();
    for path in sources {
        let code = without_line_comments(&fs::read_to_string(&path).unwrap());
        let count = engine_references(&code, &crate_root_names);
        if count > 0 {
            let file = path.strip_prefix(&perfmodel).unwrap().to_string_lossy();
            found.insert(file.replace('\\', "/"), count);
        }
    }

    let pinned: BTreeMap<String, usize> = PERFMODEL_ENGINE_REFERENCES
        .iter()
        .map(|&(file, count)| (file.to_string(), count))
        .collect();
    if found != pinned {
        let files: BTreeSet<&String> = pinned.keys().chain(found.keys()).collect();
        let drift: Vec<String> = files
            .into_iter()
            .filter_map(|file| {
                let before = pinned.get(file).copied().unwrap_or(0);
                let after = found.get(file).copied().unwrap_or(0);
                (before != after)
                    .then(|| format!("  src/perfmodel/{file}: pinned {before}, found {after}"))
            })
            .collect();
        panic!(
            "perfmodel references into crate::engine changed:\n{}\n\
             Dependencies in crates/core/src point one way: replay -> engine -> perfmodel, \
             so perfmodel must not add references to crate::engine. \
             These counts may only shrink: when you remove a reference, lower or drop its \
             entry in PERFMODEL_ENGINE_REFERENCES in the same change. \
             Crate-root re-exports of engine items (such as crate::TimingModel) count as \
             engine references, as does every super::...::engine path; refer to perfmodel's \
             own compiled engine as crate::perfmodel::engine.",
            drift.join("\n")
        );
    }
}
