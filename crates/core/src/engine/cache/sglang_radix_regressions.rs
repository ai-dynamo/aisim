// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AISimulate regressions for the documented local corrections to the pinned core.

use super::*;
use std::collections::HashMap;

type Core = UnifiedTreeCore<Vec<i64>, PageValue<u32>>;

fn params<'a>(key: &'a Vec<i64>, values: &[u32]) -> InsertParams<'a, Vec<i64>, PageValue<u32>> {
    InsertParams {
        rotation_base: None,
        key,
        namespace: KeyNamespaceRef::default(),
        session_id: None,
        swa_branching_seqlen: None,
        value: PageValue::from_vec(values.to_vec()),
        prev_prefix_len: 0,
        // Fixtures start with Full KV tombstones; tests attach explicit SWA
        // pool rows afterward. In-window rebuild tests use the resumable API.
        swa_evicted_seqlen: key.len(),
        mamba_value: None,
        chunked: false,
        priority: 0,
        track_adopted_ranges: false,
    }
}

fn insert(tree: &mut Core, key: Vec<i64>, values: &[u32]) -> NodeId {
    tree.insert(&params(&key, values))
        .last_device_node_id
        .unwrap()
}

fn complete_backup(tree: &mut Core, leaf: NodeId) {
    tree.commit_backup(leaf, PageValue::from_vec(vec![101, 102]), HashMap::new())
        .unwrap();
    let nodes = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
    tree.finish_write_through(nodes, leaf).unwrap();
}

fn commit_staged_backup(
    tree: &mut Core,
    leaf: NodeId,
    full_rows: Vec<u32>,
    mut auxiliary: HashMap<ComponentType, Vec<PoolTransfer<PageValue<u32>>>>,
    generic_entrypoint: bool,
) -> Result<(), TreeCoreRuntimeError> {
    if generic_entrypoint {
        if !full_rows.is_empty() {
            auxiliary.insert(
                FULL,
                vec![PoolTransfer {
                    host_indices: Some(PageValue::from_vec(full_rows)),
                    ..Default::default()
                }],
            );
        }
        tree.commit_hicache_transfers(
            leaf,
            CacheTransferPhase::BackupHost,
            auxiliary,
            &mut Vec::new(),
            None,
            None,
        )
    } else {
        tree.commit_backup(leaf, PageValue::from_vec(full_rows), auxiliary)
    }
}

#[test]
fn scoped_review_full_backup_rejects_an_occupied_host_slot_before_auxiliary_adoption() {
    for generic_entrypoint in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                enable_kv_cache_events: true,
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let key = vec![10, 20];
        let mut p = params(&key, &[1, 2]);
        p.mamba_value = Some(PageValue::from_vec(vec![11]));
        let leaf = tree.insert(&p).last_device_node_id.unwrap();
        let (first, _) = tree.build_backup_spec(leaf).unwrap();
        let (second, mut auxiliary) = tree.build_backup_spec(leaf).unwrap();
        assert_eq!(first.as_slice(), second.as_slice());
        auxiliary.get_mut(&MAMBA).unwrap()[0].host_indices = Some(PageValue::from_vec(vec![201]));
        tree.commit_backup(leaf, PageValue::from_vec(vec![101, 102]), HashMap::new())
            .unwrap();
        tree.take_events();
        let nodes = tree.inspect_get_all_node_ids();
        let sizes = (
            tree.total_size(),
            tree.evictable_size(),
            tree.protected_size(),
        );

        assert!(matches!(
            commit_staged_backup(
                &mut tree,
                leaf,
                vec![301, 302],
                auxiliary,
                generic_entrypoint,
            ),
            Err(TreeCoreRuntimeError::InvalidBackupSpec {
                component_type: FULL,
                ..
            })
        ));
        let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
        assert_eq!(node.host_value(FULL).as_slice(), &[101, 102]);
        assert!(!node.has_host_value(MAMBA));
        assert_eq!(tree.inspect_get_all_node_ids(), nodes);
        assert_eq!(
            (
                tree.total_size(),
                tree.evictable_size(),
                tree.protected_size()
            ),
            sizes
        );
        assert!(tree.take_events().is_empty());

        // Replanning skips Full while allowing the auxiliary-only backup.
        let (full, mut auxiliary) = tree.build_backup_spec(leaf).unwrap();
        assert!(full.is_empty());
        auxiliary.get_mut(&MAMBA).unwrap()[0].host_indices = Some(PageValue::from_vec(vec![401]));
        commit_staged_backup(&mut tree, leaf, vec![], auxiliary, generic_entrypoint).unwrap();
        let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
        assert_eq!(node.host_value(FULL).as_slice(), &[101, 102]);
        assert_eq!(node.host_value(MAMBA).as_slice(), &[401]);
        let pending = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
        tree.finish_write_through(pending, leaf).unwrap();
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_mamba_backup_validates_slot_count_before_adopting_full_rows() {
    for generic_entrypoint in [false, true] {
        for row_count in [0, 1, 2] {
            let mut tree = Core::new(
                CacheInitParams {
                    enable_hicache: true,
                    enable_kv_cache_events: true,
                    mamba_cache_chunk_size: Some(1),
                    ..Default::default()
                },
                vec![FULL, MAMBA],
            );
            let key = vec![10, 20];
            let mut p = params(&key, &[1, 2]);
            p.mamba_value = Some(PageValue::from_vec(vec![11]));
            let leaf = tree.insert(&p).last_device_node_id.unwrap();
            let (_, mut auxiliary) = tree.build_backup_spec(leaf).unwrap();
            auxiliary.get_mut(&MAMBA).unwrap()[0].host_indices =
                Some(PageValue::from_vec(vec![201; row_count]));
            tree.take_events();
            let nodes = tree.inspect_get_all_node_ids();
            let sizes = (
                tree.total_size(),
                tree.evictable_size(),
                tree.protected_size(),
            );
            let result = commit_staged_backup(
                &mut tree,
                leaf,
                vec![101, 102],
                auxiliary,
                generic_entrypoint,
            );
            if row_count != 1 {
                assert!(matches!(
                    result,
                    Err(TreeCoreRuntimeError::InvalidBackupSpec {
                        component_type: MAMBA,
                        ..
                    })
                ));
                let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
                assert!(!node.has_host_value(FULL));
                assert!(!node.has_host_value(MAMBA));
                assert_eq!(tree.inspect_get_all_node_ids(), nodes);
                assert_eq!(
                    (
                        tree.total_size(),
                        tree.evictable_size(),
                        tree.protected_size()
                    ),
                    sizes
                );
                assert!(tree.take_events().is_empty());
                backup_components(&mut tree, leaf, MAMBA);
            } else {
                result.unwrap();
                let pending = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
                tree.finish_write_through(pending, leaf).unwrap();
            }
            let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
            assert_eq!(node.host_value(FULL).as_slice(), &[101, 102]);
            assert_eq!(node.host_value(MAMBA).as_slice(), &[201]);
            tree.try_sanity_check(&[], &[]).unwrap();
        }
    }
}

#[test]
fn scoped_review_host_digest_validation_precedes_mutation_and_events() {
    for events in [false, true] {
        for namespace in [
            KeyNamespaceRef::default(),
            KeyNamespaceRef::new(Some("tenant"), Some("salt")),
        ] {
            for digest in [
                "abc".into(),
                "a".repeat(63),
                "a".repeat(65),
                "g".repeat(64),
                "é".repeat(32),
            ] {
                let mut tree = Core::new(
                    CacheInitParams {
                        enable_hicache: true,
                        enable_kv_cache_events: events,
                        ..Default::default()
                    },
                    vec![FULL],
                );
                let root = tree.root_node_handle(None);
                let nodes = tree.inspect_get_all_node_ids();
                let tick = tree.inspect_get_node_access_counter(root).unwrap();
                let result = tree.insert_host_in_namespace(
                    root,
                    namespace,
                    vec![10],
                    PageValue::from_vec(vec![101]),
                    vec![digest],
                );
                assert!(matches!(
                    result,
                    Err(TreeCoreRuntimeError::InsertHostInvalidHash { page_index: 0 })
                ));
                assert_eq!(tree.inspect_get_all_node_ids(), nodes);
                assert_eq!(tree.inspect_get_node_access_counter(root).unwrap(), tick);
                assert!(tree.take_events().is_empty());

                // Mixed-case ASCII hex remains a valid event hash and chain seed.
                let digest = "aBcD".repeat(16);
                let leaf = tree
                    .insert_host_in_namespace(
                        root,
                        namespace,
                        vec![10],
                        PageValue::from_vec(vec![201]),
                        vec![digest.clone()],
                    )
                    .unwrap()
                    .inserted_host_node
                    .unwrap();
                let idx = tree.arena.resolve(leaf).unwrap();
                assert_eq!(tree.arena.node(idx).host_value(FULL).as_slice(), &[201]);
                assert_eq!(tree.get_hash_values(leaf).unwrap(), vec![digest.clone()]);
                assert_eq!(!tree.take_events().is_empty(), events);
                let child = tree
                    .insert_host_in_namespace(
                        leaf,
                        namespace,
                        vec![20],
                        PageValue::from_vec(vec![202]),
                        vec!["11".repeat(32)],
                    )
                    .unwrap()
                    .inserted_host_node
                    .unwrap();
                let idx = tree.arena.resolve(child).unwrap();
                assert_eq!(
                    tree.arena.compute_node_hash_values(idx, 1),
                    super::node::get_hash_str::<Vec<i64>>(&vec![20], Some(&digest), 1),
                );
                tree.try_sanity_check(&[], &[]).unwrap();
            }
        }
    }
}

#[test]
fn scoped_review_full_backup_rejects_a_split_before_any_component_commits() {
    for generic_entrypoint in [false, true] {
        for split in [false, true] {
            let mut tree = Core::new(
                CacheInitParams {
                    enable_hicache: true,
                    mamba_cache_chunk_size: Some(1),
                    ..Default::default()
                },
                vec![FULL, MAMBA],
            );
            let key = vec![10, 20];
            let mut p = params(&key, &[1, 2]);
            p.mamba_value = Some(PageValue::from_vec(vec![11]));
            let leaf = tree.insert(&p).last_device_node_id.unwrap();
            let (planned, mut auxiliary) = tree.build_backup_spec(leaf).unwrap();
            assert_eq!(planned.as_slice(), &[1, 2]);
            auxiliary.get_mut(&MAMBA).unwrap()[0].host_indices =
                Some(PageValue::from_vec(vec![201]));
            if split {
                tree.match_prefix(&MatchPrefixParams {
                    key: &vec![10],
                    namespace: Default::default(),
                });
            }
            let result = if generic_entrypoint {
                auxiliary.insert(
                    FULL,
                    vec![PoolTransfer {
                        host_indices: Some(PageValue::from_vec(vec![101, 102])),
                        ..Default::default()
                    }],
                );
                tree.commit_hicache_transfers(
                    leaf,
                    CacheTransferPhase::BackupHost,
                    auxiliary,
                    &mut Vec::new(),
                    None,
                    None,
                )
            } else {
                tree.commit_backup(leaf, PageValue::from_vec(vec![101, 102]), auxiliary)
            };
            if split {
                assert!(matches!(
                    result,
                    Err(TreeCoreRuntimeError::InvalidBackupSpec {
                        component_type: FULL,
                        ..
                    })
                ));
                for id in tree.inspect_get_all_node_ids() {
                    let node = tree.arena.node(tree.arena.resolve(id).unwrap());
                    assert!(!node.has_host_value(FULL));
                    assert!(!node.has_host_value(MAMBA));
                }
                // The rejected allocation was never adopted. Re-plan and supply
                // a fresh host span sized to the surviving suffix.
                tree.commit_backup(leaf, PageValue::from_vec(vec![102]), HashMap::new())
                    .unwrap();
            } else {
                result.unwrap();
                let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
                assert_eq!(node.host_value(FULL).as_slice(), &[101, 102]);
                assert_eq!(node.host_value(MAMBA).as_slice(), &[201]);
            }
        }
    }
}

#[test]
fn scoped_review_swa_backup_checks_all_targets_before_adopting_full_or_swa() {
    for split in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                swa_sliding_window_size: Some(8),
                ..Default::default()
            },
            vec![FULL, SWA],
        );
        tree.set_has_swa_host_pool();
        let parent = insert(&mut tree, vec![5], &[1]);
        let key = vec![5, 10, 20];
        let leaf = insert(&mut tree, key.clone(), &[1, 2, 3]);
        tree.attach_swa_window(
            &key,
            Default::default(),
            0,
            3,
            &PageValue::from_vec(vec![11, 12, 13]),
        )
        .unwrap();
        let (_, mut auxiliary) = tree.build_backup_spec(leaf).unwrap();
        assert_eq!(
            auxiliary[&SWA][0].nodes_to_load.as_deref(),
            Some(&[parent, leaf][..])
        );
        auxiliary.get_mut(&SWA).unwrap()[0].host_indices =
            Some(PageValue::from_vec(vec![201, 202, 203]));
        if split {
            tree.match_prefix(&MatchPrefixParams {
                key: &vec![5, 10],
                namespace: Default::default(),
            });
        }
        // Full's current span is valid in both cases, so the SWA guard must
        // reject before Full or an earlier SWA target has adopted anything.
        let result = tree.commit_backup(
            leaf,
            PageValue::from_vec(if split { vec![103] } else { vec![102, 103] }),
            auxiliary,
        );
        if split {
            assert!(matches!(
                result,
                Err(TreeCoreRuntimeError::InvalidBackupSpec {
                    component_type: SWA,
                    ..
                })
            ));
            for id in tree.inspect_get_all_node_ids() {
                let node = tree.arena.node(tree.arena.resolve(id).unwrap());
                assert!(!node.has_host_value(FULL));
                assert!(!node.has_host_value(SWA));
            }
            let (_, mut auxiliary) = tree.build_backup_spec(leaf).unwrap();
            auxiliary.get_mut(&SWA).unwrap()[0].host_indices =
                Some(PageValue::from_vec(vec![301, 302, 303]));
            tree.commit_backup(leaf, PageValue::from_vec(vec![103]), auxiliary)
                .unwrap();
        } else {
            result.unwrap();
        }
        let paths = if split {
            vec![vec![5], vec![5, 10], vec![5, 10, 20]]
        } else {
            vec![vec![5], vec![5, 10, 20]]
        };
        let host_rows: Vec<_> = paths
            .iter()
            .map(|key| {
                let id = tree
                    .match_prefix(&MatchPrefixParams {
                        key,
                        namespace: Default::default(),
                    })
                    .best_match_node_id;
                tree.arena
                    .node(tree.arena.resolve(id).unwrap())
                    .host_value(SWA)
                    .as_slice()
                    .to_vec()
            })
            .collect();
        assert_eq!(
            host_rows,
            if split {
                vec![vec![301], vec![302], vec![303]]
            } else {
                vec![vec![201], vec![202, 203]]
            }
        );
    }
}

#[test]
fn scoped_review_host_input_validation_precedes_recency_splits_and_allocation() {
    for page_size in [1, 2] {
        for events in [false, true] {
            for populated in [false, true] {
                for invalid_kind in 0..3 {
                    if invalid_kind == 0 && page_size == 1 {
                        continue;
                    }
                    let mut tree = Core::new(
                        CacheInitParams {
                            page_size,
                            enable_hicache: true,
                            enable_kv_cache_events: events,
                            ..Default::default()
                        },
                        vec![FULL],
                    );
                    let root = tree.root_node_handle(None);
                    let namespace = KeyNamespaceRef::new(Some("tenant"), Some("salt"));
                    let key: Vec<_> = (1..=2 * page_size as i64).map(|i| i * 10).collect();
                    if populated {
                        tree.insert_host_in_namespace(
                            root,
                            namespace,
                            key.clone(),
                            PageValue::from_vec(vec![101; key.len()]),
                            vec!["11".repeat(32), "22".repeat(32)],
                        )
                        .unwrap();
                    }
                    tree.take_events();
                    let nodes = tree.inspect_get_all_node_ids();
                    let before: Vec<_> = nodes
                        .iter()
                        .map(|&id| {
                            let n = tree.arena.node(tree.arena.resolve(id).unwrap());
                            (
                                n.key.clone(),
                                n.last_access_counter,
                                n.hash_value.clone(),
                                n.try_host_value(FULL).map(|v| v.as_slice().to_vec()),
                            )
                        })
                        .collect();
                    let mut incoming = key.clone();
                    let mut hashes = vec!["33".repeat(32), "44".repeat(32)];
                    let mut rows = vec![201; key.len()];
                    match invalid_kind {
                        0 => {
                            incoming.pop();
                            hashes.pop();
                        }
                        1 => {
                            hashes[1].clear();
                            *incoming.last_mut().unwrap() += 1;
                        }
                        _ => {
                            rows.pop();
                            *incoming.last_mut().unwrap() += 1;
                        }
                    }
                    let result = tree.insert_host_in_namespace(
                        root,
                        namespace,
                        incoming,
                        PageValue::from_vec(rows),
                        hashes,
                    );
                    assert!(match invalid_kind {
                        0 => matches!(
                            result,
                            Err(TreeCoreRuntimeError::InsertHostKeyNotPageAligned { .. })
                        ),
                        1 => matches!(
                            result,
                            Err(TreeCoreRuntimeError::InsertHostEmptyHash { page_index: 1 })
                        ),
                        _ => matches!(
                            result,
                            Err(TreeCoreRuntimeError::InsertHostValueTooShort { .. })
                        ),
                    });
                    assert_eq!(tree.inspect_get_all_node_ids(), nodes);
                    for (&id, expected) in nodes.iter().zip(before) {
                        let n = tree.arena.node(tree.arena.resolve(id).unwrap());
                        assert_eq!(
                            (
                                n.key.clone(),
                                n.last_access_counter,
                                n.hash_value.clone(),
                                n.try_host_value(FULL).map(|v| v.as_slice().to_vec())
                            ),
                            expected
                        );
                    }
                    assert!(tree.take_events().is_empty());
                    // Valid aligned input still grafts every row and retains
                    // real digests as the chain seed, including in a namespace.
                    let graft = tree
                        .insert_host_in_namespace(
                            root,
                            namespace,
                            key.clone(),
                            PageValue::from_vec(vec![101; key.len()]),
                            vec!["11".repeat(32), "22".repeat(32)],
                        )
                        .unwrap()
                        .inserted_host_node
                        .unwrap();
                    let child = tree
                        .insert_host_in_namespace(
                            graft,
                            namespace,
                            vec![90; page_size],
                            PageValue::from_vec(vec![301; page_size]),
                            vec!["55".repeat(32)],
                        )
                        .unwrap()
                        .inserted_host_node
                        .unwrap();
                    let idx = tree.arena.resolve(child).unwrap();
                    assert_eq!(
                        tree.arena.compute_node_hash_values(idx, page_size),
                        super::node::get_hash_str::<Vec<i64>>(
                            &vec![90; page_size],
                            Some(&"22".repeat(32)),
                            page_size
                        )
                    );
                }
            }
        }
    }
}

#[test]
fn scoped_review_mamba_load_back_binds_tree_transfer_to_its_anchor() {
    for invalid_kind in 0..3 {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                mamba_cache_chunk_size: Some(1),
                enable_kv_cache_events: true,
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let mut leaves = Vec::new();
        for key in [vec![10, 20], vec![30, 40]] {
            let mut p = params(&key, &[1, 2]);
            p.mamba_value = Some(PageValue::from_vec(vec![11]));
            let leaf = tree.insert(&p).last_device_node_id.unwrap();
            backup_components(&mut tree, leaf, MAMBA);
            evict_auxiliary(&mut tree, leaf, MAMBA);
            leaves.push(leaf);
        }
        let [a, b] = leaves[..] else { unreachable!() };
        let request = Req {
            mamba_pool_idx: Some(PageValue::from_vec(vec![301])),
            ..Default::default()
        };
        let (full, mut auxiliary) = tree.build_load_back_spec(b, Some(&request)).unwrap();
        assert!(full.nodes_to_load.as_ref().unwrap().is_empty());
        let transfers = auxiliary.get_mut(&MAMBA).unwrap();
        transfers[0].device_indices = Some(PageValue::from_vec(vec![501]));
        match invalid_kind {
            0 => transfers[0].nodes_to_load = Some(vec![a]),
            1 => transfers.swap(0, 1),
            _ => {
                transfers.remove(0);
            }
        }
        tree.take_events();
        let sizes = (tree.evictable_size(), tree.protected_size());
        assert!(matches!(
            tree.commit_load_back(b, PageValue::empty(), full, auxiliary),
            Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
        ));
        for id in [a, b] {
            let node = tree.arena.node(tree.arena.resolve(id).unwrap());
            assert!(!node.has_device_value(MAMBA));
            assert_eq!(node.load_back_pending_id, None);
        }
        assert_eq!((tree.evictable_size(), tree.protected_size()), sizes);
        assert!(tree.take_events().is_empty());
        let (full, mut auxiliary) = tree.build_load_back_spec(b, Some(&request)).unwrap();
        auxiliary.get_mut(&MAMBA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![501]));
        tree.commit_load_back(b, PageValue::empty(), full, auxiliary)
            .unwrap();
        tree.finish_load_back(b).unwrap();
        assert_eq!(
            tree.arena
                .node(tree.arena.resolve(b).unwrap())
                .device_value(MAMBA)
                .as_slice(),
            &[501]
        );
        assert!(
            !tree
                .arena
                .node(tree.arena.resolve(a).unwrap())
                .has_device_value(MAMBA)
        );
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_host_hash_count_is_validated_before_mutation() {
    for page_size in [1, 2] {
        for enable_kv_cache_events in [false, true] {
            let mut tree = Core::new(
                CacheInitParams {
                    page_size,
                    enable_hicache: true,
                    enable_kv_cache_events,
                    ..Default::default()
                },
                vec![FULL],
            );
            let root = tree.root_node_handle(None);
            let key: Vec<_> = (1..=2 * page_size as i64).map(|i| 10 * i).collect();
            let rows: Vec<_> = (100..100 + 2 * page_size as u32).collect();
            let hashes = vec!["11".repeat(32), "22".repeat(32)];
            // Exercise an empty root, a walk that would split an existing edge,
            // and a non-root anchor whose recency must also remain unchanged.
            let mut anchor = root;
            for populated in [false, true] {
                if populated {
                    anchor = tree
                        .insert_host(
                            root,
                            None,
                            key.clone(),
                            PageValue::from_vec(rows.clone()),
                            hashes.clone(),
                        )
                        .unwrap()
                        .inserted_host_node
                        .unwrap();
                    tree.take_events();
                }
                let starts = if populated {
                    vec![root, anchor]
                } else {
                    vec![root]
                };
                for start in starts {
                    for hash_count in [0, 1, 3] {
                        let nodes = tree.inspect_get_all_node_ids();
                        let ticks: Vec<_> = nodes
                            .iter()
                            .map(|&id| tree.inspect_get_node_access_counter(id).unwrap())
                            .collect();
                        let mut incoming_key = key.clone();
                        if populated && start == root {
                            *incoming_key.last_mut().unwrap() += 1;
                        }
                        let result = tree.insert_host_in_namespace(
                            start,
                            KeyNamespaceRef::default(),
                            incoming_key,
                            PageValue::from_vec(rows.clone()),
                            vec!["hash".into(); hash_count],
                        );
                        assert!(matches!(
                            result,
                            Err(TreeCoreRuntimeError::InsertHostHashCount {
                                hash_count: actual,
                                page_count: 2,
                            }) if actual == hash_count
                        ));
                        assert_eq!(tree.inspect_get_all_node_ids(), nodes);
                        for (&id, tick) in nodes.iter().zip(ticks) {
                            assert_eq!(tree.inspect_get_node_access_counter(id).unwrap(), tick);
                        }
                        assert!(tree.take_events().is_empty());
                    }
                }
            }
            // Valid hashes survive the exact partial-match split that used to
            // panic after an empty hash vector was accepted with events off.
            let matched = tree.match_prefix(&MatchPrefixParams {
                key: &key[..page_size].to_vec(),
                namespace: KeyNamespaceRef::default(),
            });
            assert_eq!(matched.host_hit_length, page_size);
            assert_eq!(
                tree.inspect_get_component_host_value(matched.last_host_node_id, FULL)
                    .unwrap()
                    .unwrap()
                    .as_slice(),
                &rows[..page_size]
            );
            let prefix = tree.arena.resolve(matched.last_host_node_id).unwrap();
            assert_eq!(
                tree.arena.node(prefix).hash_value.as_ref().unwrap(),
                &hashes[..1]
            );
            let suffix = tree.arena.node(tree.arena.resolve(anchor).unwrap());
            assert_eq!(suffix.hash_value.as_ref().unwrap(), &hashes[1..]);
            tree.try_sanity_check(&[], &[]).unwrap();
        }
    }
}

#[test]
fn scoped_review_lru_and_mru_free_different_competing_pages() {
    for (policy, expected) in [("lru", 1), ("mru", 2)] {
        let mut tree = Core::new(
            CacheInitParams {
                eviction_policy: policy.into(),
                ..Default::default()
            },
            vec![FULL],
        );
        insert(&mut tree, vec![10], &[1]);
        insert(&mut tree, vec![20], &[2]);
        tree.evict_device_start(FULL, 1);
        let (victim, _) = tree.evict_device_next_node(FULL, &HashMap::new());
        let (_, result) = tree.evict_device_leaf(victim.unwrap(), false).unwrap();
        tree.evict_device_end(FULL);
        assert_eq!(
            result.device_frees[&FULL][0].as_slice(),
            &[expected],
            "{policy}"
        );
        assert_eq!(tree.full_evictable_size(), 1);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_tlru_integer_overflow_keeps_ordinary_priority() {
    use super::unified_lru_list::{TlruFloatConfig, TlruPromptEstimate};
    let config = TlruFloatConfig {
        next_prompt_estimate: TlruPromptEstimate::Integer(i128::MAX),
        threshold: 0.0,
    };
    assert!(!config.is_tel_safe(1, 0));
    // Cancellation must still happen in integer space when the sum fits.
    let precise = TlruFloatConfig {
        next_prompt_estimate: TlruPromptEstimate::Integer(-(1i128 << 53)),
        threshold: 0.0,
    };
    assert!(!precise.is_tel_safe((1usize << 53) + 1, 0));
    let mut tree = Core::new(
        CacheInitParams {
            eviction_policy: "tlru".into(),
            tlru_float_config: Some(config),
            ..Default::default()
        },
        vec![FULL],
    );
    let leaf = insert(&mut tree, vec![10], &[1]);
    let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
    assert_eq!(tree.eviction_strategy.get_priority(node).0, 0);
    tree.evict_device_start(FULL, 1);
    assert_eq!(
        tree.evict_device_next_node(FULL, &HashMap::new()).0,
        Some(leaf)
    );
    let (_, result) = tree.evict_device_leaf(leaf, false).unwrap();
    tree.evict_device_end(FULL);
    assert_eq!(result.device_frees[&FULL][0].as_slice(), &[1]);
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn scoped_review_swa_boundary_is_validated_before_full_insert() {
    let mut tree = Core::new(
        CacheInitParams {
            page_size: 2,
            swa_sliding_window_size: Some(2),
            enable_kv_cache_events: true,
            ..Default::default()
        },
        vec![FULL, SWA],
    );
    let key = vec![10, 20, 30, 40];
    let mut p = params(&key, &[1, 2, 3, 4]);
    p.swa_evicted_seqlen = 1;
    let nodes = tree.inspect_get_all_node_ids();
    assert!(matches!(
        tree.try_begin_insert(&p),
        Err(TreeCoreRuntimeError::InsertSwaBoundaryNotPageAligned {
            boundary: 1,
            page_size: 2
        })
    ));
    assert_eq!(tree.inspect_get_all_node_ids(), nodes);
    assert_eq!(tree.total_size(), (0, 0));
    assert!(!tree.has_ongoing_insert());
    assert!(tree.take_events().is_empty());
    p.swa_evicted_seqlen = 2;
    let mut step = tree.try_begin_insert(&p).unwrap();
    let inserted = loop {
        for action in step.actions {
            if let CacheAction::SwaRebuild {
                node_id,
                source_value,
            } = action
            {
                // Synthetic controller: use an identity Full-to-SWA mapping.
                tree.set_component_device_value(node_id, SWA, source_value)
                    .unwrap();
            }
        }
        if let Some(result) = step.result {
            break result;
        }
        step = tree.try_resume_insert().unwrap();
    };
    assert_eq!(inserted.total_len, 4);
    assert_eq!(tree.full_kv_prefix_len(&key, Default::default()), 4);
    assert!(!tree.take_events().is_empty());
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn scoped_review_suspended_insert_recovers_demoted_prefix_without_duplicate_frees() {
    for demote in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                write_through_threshold: 2,
                enable_kv_cache_events: true,
                ..Default::default()
            },
            vec![FULL],
        );
        let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
        let key = vec![10, 20, 30];
        let mut p = params(&key, &[5, 6, 7]);
        p.swa_evicted_seqlen = 0; // Exercise ordinary duplicate Full KV frees.
        let step = tree.try_begin_insert(&p).unwrap();
        assert!(step.result.is_none());
        // The controller has already returned these incoming duplicates to its
        // allocator when it executes the backup barrier.
        let freed: Vec<_> = step
            .actions
            .iter()
            .filter_map(|action| match action {
                CacheAction::FreeDeviceKV(value) => Some(value.as_slice()),
                _ => None,
            })
            .flatten()
            .flat_map(|value| value.as_slice())
            .copied()
            .collect();
        assert_eq!(freed, vec![5, 6]);
        complete_backup(&mut tree, leaf);
        if demote {
            assert_eq!(
                tree.demote(leaf).unwrap().device_frees[&FULL][0].as_slice(),
                &[1, 2]
            );
            tree.take_events();
            for _ in 0..2 {
                assert!(
                    matches!(tree.try_resume_insert(), Err(TreeCoreRuntimeError::InsertPrefixNotDeviceResident { node_id }) if node_id == leaf)
                );
                assert!(tree.has_ongoing_insert());
                assert!(tree.take_events().is_empty());
                assert_eq!(tree.inspect_get_all_node_ids().len(), 2);
            }
            let (full, auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
            tree.commit_load_back(leaf, PageValue::from_vec(vec![8, 9]), full, auxiliary)
                .unwrap();
            tree.finish_load_back(leaf).unwrap();
        }
        let resumed = tree.try_resume_insert().unwrap();
        assert!(resumed.result.is_some());
        assert!(resumed.actions.iter().all(|action| !matches!(
            action,
            CacheAction::FreeDeviceKV(_) | CacheAction::FreeDeviceKVFullOnly(_)
        )));
        let matched = tree.match_prefix(&MatchPrefixParams {
            key: &key,
            namespace: Default::default(),
        });
        assert_eq!(
            matched.device_indices.as_slice(),
            if demote { &[8, 9, 7] } else { &[1, 2, 7] }
        );
        let lock = tree
            .inc_lock_ref(matched.last_device_node_id, ComponentSet::EMPTY)
            .unwrap();
        tree.dec_lock_ref(matched.last_device_node_id, &lock.to_dec_params(), false)
            .unwrap();
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_load_back_rejects_reclaimed_and_replaced_host_rows() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            is_write_back: true,
            enable_kv_cache_events: true,
            ..Default::default()
        },
        vec![FULL],
    );
    let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
    complete_backup(&mut tree, leaf);
    tree.demote(leaf).unwrap();
    let (stale, stale_aux) = tree.build_load_back_spec(leaf, None).unwrap();
    let (full, auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
    tree.commit_load_back(leaf, PageValue::from_vec(vec![3, 4]), full, auxiliary)
        .unwrap();
    tree.finish_load_back(leaf).unwrap();
    let reclaim = tree.drive_host_eviction(FULL, 2);
    assert_eq!(reclaim.host_frees[&FULL][0].as_slice(), &[101, 102]);
    assert!(reclaim.device_frees.is_empty());
    tree.commit_backup(leaf, PageValue::from_vec(vec![201, 202]), HashMap::new())
        .unwrap();
    let pending = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
    tree.finish_write_through(pending, leaf).unwrap();
    tree.demote(leaf).unwrap();
    tree.take_events();
    assert!(matches!(
        tree.commit_load_back(leaf, PageValue::from_vec(vec![5, 6]), stale, stale_aux),
        Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
    ));
    let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
    assert!(!node.has_device_value(FULL));
    assert_eq!(node.host_value(FULL).as_slice(), &[201, 202]);
    assert_eq!(node.load_back_pending_id, None);
    assert_eq!(tree.full_evictable_size(), 0);
    assert!(tree.take_events().is_empty());
    let (full, auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
    tree.commit_load_back(leaf, PageValue::from_vec(vec![5, 6]), full, auxiliary)
        .unwrap();
    tree.finish_load_back(leaf).unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn eviction_walk_rechecks_a_leaf_refreshed_after_start() {
    let mut tree = Core::new(CacheInitParams::default(), vec![FULL]);
    let a = insert(&mut tree, vec![10, 20], &[1, 2]);
    let b = insert(&mut tree, vec![30, 40], &[3, 4]);
    tree.evict_device_start(FULL, 1);
    tree.match_prefix(&MatchPrefixParams {
        key: &vec![10, 20],
        namespace: Default::default(),
    });
    let (victim, _) = tree.evict_device_next_node(FULL, &HashMap::new());
    assert_eq!(victim, Some(b));
    assert_ne!(victim, Some(a));
    tree.evict_device_end(FULL);
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn eviction_walk_rechecks_a_leaf_unlocked_after_start() {
    for release_last_lock in [false, true] {
        let mut tree = Core::new(CacheInitParams::default(), vec![FULL]);
        let a = insert(&mut tree, vec![10, 20], &[1, 2]);
        let first = tree.inc_lock_ref(a, Default::default()).unwrap();
        let last = tree.inc_lock_ref(a, Default::default()).unwrap();
        let b = insert(&mut tree, vec![30, 40], &[3, 4]);
        tree.evict_device_start(FULL, 1);
        tree.dec_lock_ref(a, &first.to_dec_params(), false).unwrap();
        if release_last_lock {
            tree.dec_lock_ref(a, &last.to_dec_params(), false).unwrap();
        }
        let (victim, _) = tree.evict_device_next_node(FULL, &HashMap::new());
        assert_eq!(victim, Some(if release_last_lock { a } else { b }));
        tree.evict_device_end(FULL);
        if !release_last_lock {
            tree.dec_lock_ref(a, &last.to_dec_params(), false).unwrap();
        }
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn mamba_load_back_rechecks_an_independently_filled_destination() {
    for filled in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                enable_kv_cache_events: true,
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let key = vec![10, 20];
        let mut p = params(&key, &[1, 2]);
        p.mamba_value = Some(PageValue::from_vec(vec![11]));
        let leaf = tree.insert(&p).last_device_node_id.unwrap();
        backup_components(&mut tree, leaf, MAMBA);
        evict_auxiliary(&mut tree, leaf, MAMBA);
        let (full, mut auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
        assert!(full.node_row_counts.as_ref().unwrap().is_empty());
        auxiliary.get_mut(&MAMBA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![302]));
        if filled {
            p.mamba_value = Some(PageValue::from_vec(vec![401]));
            tree.insert(&p);
        }
        tree.take_events();
        let sizes = (
            tree.total_size(),
            tree.evictable_size(),
            tree.protected_size(),
        );
        let lru = tree.inspect_get_component_device_lru_node_ids(MAMBA);
        let result = tree.commit_load_back(leaf, PageValue::from_vec(vec![]), full, auxiliary);
        if filled {
            assert!(matches!(
                result,
                Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
            ));
            assert_eq!(
                (
                    tree.total_size(),
                    tree.evictable_size(),
                    tree.protected_size()
                ),
                sizes
            );
            assert_eq!(tree.inspect_get_component_device_lru_node_ids(MAMBA), lru);
        } else {
            assert!(result.unwrap().is_empty());
            tree.finish_load_back(leaf).unwrap();
        }
        let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
        assert_eq!(
            node.device_value(MAMBA).as_slice(),
            if filled { &[401] } else { &[302] }
        );
        assert_eq!(node.device_value(FULL).as_slice(), &[1, 2]);
        assert_eq!(node.load_back_pending_id, None);
        assert!(tree.take_events().is_empty());
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn swa_load_back_rechecks_device_rows_skipped_by_the_plan() {
    for write_back in [false, true] {
        for (restore_full, restore_child_swa) in [(false, false), (false, true), (true, true)] {
            for evict_parent in [false, true] {
                let mut tree = Core::new(
                    CacheInitParams {
                        enable_hicache: true,
                        enable_kv_cache_events: true,
                        is_write_back: write_back,
                        swa_sliding_window_size: Some(2),
                        has_swa_host_pool: true,
                        ..Default::default()
                    },
                    vec![FULL, SWA],
                );
                let parent = insert(&mut tree, vec![10], &[1]);
                let key = vec![10, 20];
                let child = insert(&mut tree, key.clone(), &[1, 2]);
                tree.attach_swa_window(
                    &key,
                    Default::default(),
                    0,
                    2,
                    &PageValue::from_vec(vec![11, 12]),
                )
                .unwrap();
                for (id, full_host, swa_host) in [(parent, 101, 201), (child, 102, 202)] {
                    tree.commit_backup(
                        id,
                        PageValue::from_vec(vec![full_host]),
                        [(
                            SWA,
                            vec![PoolTransfer {
                                name: PoolName::Swa,
                                host_indices: Some(PageValue::from_vec(vec![swa_host])),
                                nodes_to_load: Some(vec![id]),
                                ..Default::default()
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    )
                    .unwrap();
                    let pending = tree.mark_write_through_pending(vec![id], id).unwrap();
                    tree.finish_write_through(pending, id).unwrap();
                }
                if restore_child_swa {
                    // Make the child LRU, then evict only its SWA through the
                    // normal internal eviction path. Full and parent SWA stay.
                    tree.match_prefix(&MatchPrefixParams {
                        key: &vec![10],
                        namespace: Default::default(),
                    });
                    evict_auxiliary(&mut tree, child, SWA);
                }
                if restore_full {
                    tree.demote(child).unwrap();
                }
                let (full, mut auxiliary) = tree.build_load_back_spec(child, None).unwrap();
                assert_eq!(
                    full.node_row_counts.as_deref().unwrap().len(),
                    usize::from(restore_full)
                );
                if restore_child_swa {
                    assert_eq!(
                        auxiliary[&SWA][0].nodes_to_load.as_deref(),
                        Some(&[child][..])
                    );
                    auxiliary.get_mut(&SWA).unwrap()[0].device_indices =
                        Some(PageValue::from_vec(vec![302]));
                } else {
                    assert!(!auxiliary.contains_key(&SWA));
                }
                if evict_parent {
                    evict_auxiliary(&mut tree, parent, SWA);
                }
                tree.take_events();
                let sizes = (
                    tree.total_size(),
                    tree.evictable_size(),
                    tree.protected_size(),
                );
                let lru = tree.inspect_get_component_device_lru_node_ids(SWA);
                let result = tree.commit_load_back(
                    child,
                    PageValue::from_vec(if restore_full { vec![3] } else { vec![] }),
                    full,
                    auxiliary,
                );
                if evict_parent {
                    assert!(matches!(
                        result,
                        Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
                    ));
                    assert_eq!(
                        (
                            tree.total_size(),
                            tree.evictable_size(),
                            tree.protected_size()
                        ),
                        sizes
                    );
                    assert_eq!(tree.inspect_get_component_device_lru_node_ids(SWA), lru);
                    assert!(tree.take_events().is_empty());
                    for id in [parent, child] {
                        let node = tree.arena.node(tree.arena.resolve(id).unwrap());
                        assert_eq!(node.load_back_pending_id, None);
                        assert_eq!(node.has_device_value(FULL), id == parent || !restore_full);
                        if let Some(value) = node.try_device_value(FULL) {
                            assert_eq!(value.as_slice(), if id == parent { &[1] } else { &[2] });
                        }
                        assert_eq!(
                            node.host_value(FULL).as_slice(),
                            if id == parent { &[101] } else { &[102] }
                        );
                        assert_eq!(
                            node.host_value(SWA).as_slice(),
                            if id == parent { &[201] } else { &[202] }
                        );
                        assert_eq!(
                            node.has_device_value(SWA),
                            id == child && !restore_child_swa
                        );
                    }
                    tree.try_sanity_check(&[], &[]).unwrap();
                    // The caller can reclaim the rejected staging rows and
                    // retry with a fresh plan covering the newly missing span.
                    let (full, mut auxiliary) = tree.build_load_back_spec(child, None).unwrap();
                    let expected_nodes = if restore_child_swa {
                        vec![parent, child]
                    } else {
                        vec![parent]
                    };
                    assert_eq!(
                        auxiliary[&SWA][0].nodes_to_load.as_deref(),
                        Some(expected_nodes.as_slice())
                    );
                    auxiliary.get_mut(&SWA).unwrap()[0].device_indices =
                        Some(PageValue::from_vec(if restore_child_swa {
                            vec![301, 302]
                        } else {
                            vec![301]
                        }));
                    tree.commit_load_back(
                        child,
                        PageValue::from_vec(if restore_full { vec![3] } else { vec![] }),
                        full,
                        auxiliary,
                    )
                    .unwrap();
                } else {
                    result.unwrap();
                }
                tree.finish_load_back(child).unwrap();
                let matched = tree.match_prefix(&MatchPrefixParams {
                    key: &key,
                    namespace: Default::default(),
                });
                assert_eq!(matched.last_device_node_id, child);
                assert_eq!(
                    matched.device_indices.as_slice(),
                    if restore_full { &[1, 3] } else { &[1, 2] }
                );
                for (id, value) in [
                    (parent, if evict_parent { 301 } else { 11 }),
                    (child, if restore_child_swa { 302 } else { 12 }),
                ] {
                    let node = tree.arena.node(tree.arena.resolve(id).unwrap());
                    assert_eq!(node.device_value(SWA).as_slice(), &[value]);
                    assert_eq!(node.load_back_pending_id, None);
                }
                tree.try_sanity_check(&[], &[]).unwrap();
            }
        }
    }
}

#[test]
fn swa_load_back_rejects_split_rows_before_any_component_commits() {
    for write_back in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                enable_kv_cache_events: true,
                is_write_back: write_back,
                swa_sliding_window_size: Some(2),
                has_swa_host_pool: true,
                ..Default::default()
            },
            vec![FULL, SWA],
        );
        let key = vec![10, 20];
        let leaf = insert(&mut tree, key.clone(), &[1, 2]);
        tree.attach_swa_window(
            &key,
            Default::default(),
            0,
            2,
            &PageValue::from_vec(vec![11, 12]),
        )
        .unwrap();
        backup_components(&mut tree, leaf, SWA);
        evict_auxiliary(&mut tree, leaf, SWA);
        let (full, mut auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
        assert!(full.node_row_counts.as_ref().unwrap().is_empty());
        auxiliary.get_mut(&SWA).unwrap()[0].device_indices =
            Some(PageValue::from_vec(vec![301, 302]));
        let prefix = tree
            .match_prefix(&MatchPrefixParams {
                key: &vec![10],
                namespace: Default::default(),
            })
            .best_match_node_id;
        tree.take_events();
        let sizes = (
            tree.total_size(),
            tree.evictable_size(),
            tree.protected_size(),
        );
        let lru = tree.inspect_get_component_device_lru_node_ids(SWA);
        assert!(matches!(
            tree.commit_load_back(leaf, PageValue::from_vec(vec![]), full, auxiliary),
            Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
        ));
        assert_eq!(
            (
                tree.total_size(),
                tree.evictable_size(),
                tree.protected_size()
            ),
            sizes
        );
        assert_eq!(tree.inspect_get_component_device_lru_node_ids(SWA), lru);
        assert!(tree.take_events().is_empty());
        for id in [prefix, leaf] {
            let node = tree.arena.node(tree.arena.resolve(id).unwrap());
            assert!(!node.has_device_value(SWA));
            assert_eq!(node.load_back_pending_id, None);
        }
        tree.try_sanity_check(&[], &[]).unwrap();
        let (full, mut auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
        assert_eq!(
            auxiliary[&SWA][0].node_row_counts.as_deref(),
            Some(&[1, 1][..])
        );
        auxiliary.get_mut(&SWA).unwrap()[0].device_indices =
            Some(PageValue::from_vec(vec![301, 302]));
        let actions = tree
            .commit_load_back(leaf, PageValue::from_vec(vec![]), full, auxiliary)
            .unwrap();
        assert_eq!(actions.len(), 1);
        let CacheAction::RebuildFullToSwaMapping {
            full_indices,
            swa_indices,
        } = &actions[0]
        else {
            panic!("expected mapping rebuild after successful commit");
        };
        assert_eq!(PageValue::concat(full_indices).as_slice(), &[1, 2]);
        assert_eq!(PageValue::concat(swa_indices).as_slice(), &[301, 302]);
        for (id, value) in [(prefix, 301), (leaf, 302)] {
            let node = tree.arena.node(tree.arena.resolve(id).unwrap());
            assert_eq!(node.device_value(SWA).as_slice(), &[value]);
        }
        tree.finish_load_back(leaf).unwrap();
        assert!(tree.take_events().is_empty());
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn foreign_pin_load_back_deferral_commits_as_a_noop() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            is_write_back: true,
            enable_kv_cache_events: true,
            mamba_cache_chunk_size: Some(1),
            ..Default::default()
        },
        vec![FULL, MAMBA],
    );
    let key = vec![10, 20];
    let mut p = params(&key, &[1, 2]);
    p.mamba_value = Some(PageValue::from_vec(vec![11]));
    let parent = tree.insert(&p).last_device_node_id.unwrap();
    let key = vec![10, 20, 30, 40];
    let mut p = params(&key, &[1, 2, 3, 4]);
    p.mamba_value = Some(PageValue::from_vec(vec![12]));
    let child = tree.insert(&p).last_device_node_id.unwrap();
    backup_components(&mut tree, parent, MAMBA);
    backup_components(&mut tree, child, MAMBA);
    tree.demote(child).unwrap();
    tree.demote(parent).unwrap();
    // Loading the child also pins the parent's Full rows. Its independent
    // Mamba checkpoint remains host-only and must defer behind the child ack.
    let (full, mut auxiliary) = tree.build_load_back_spec(child, None).unwrap();
    auxiliary.get_mut(&MAMBA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![302]));
    tree.commit_load_back(
        child,
        PageValue::from_vec(vec![5, 6, 7, 8]),
        full,
        auxiliary,
    )
    .unwrap();
    tree.take_events();
    let (full, auxiliary) = tree.build_load_back_spec(parent, None).unwrap();
    assert_eq!(full.node_row_counts.as_deref(), Some(&[][..]));
    assert!(auxiliary.is_empty());
    let sizes = (
        tree.total_size(),
        tree.evictable_size(),
        tree.protected_size(),
    );
    assert!(
        tree.commit_load_back(parent, PageValue::from_vec(vec![]), full, auxiliary)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        (
            tree.total_size(),
            tree.evictable_size(),
            tree.protected_size()
        ),
        sizes
    );
    let node = tree.arena.node(tree.arena.resolve(parent).unwrap());
    assert_eq!(node.load_back_pending_id, Some(child));
    assert!(!node.has_device_value(MAMBA));
    assert!(tree.take_events().is_empty());
    tree.finish_load_back(child).unwrap();
    let (full, mut auxiliary) = tree.build_load_back_spec(parent, None).unwrap();
    auxiliary.get_mut(&MAMBA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![303]));
    tree.commit_load_back(parent, PageValue::from_vec(vec![]), full, auxiliary)
        .unwrap();
    tree.finish_load_back(parent).unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn foreign_pin_load_back_deferral_with_host_only_anchor_commits_as_a_noop() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            is_write_back: true,
            enable_kv_cache_events: true,
            swa_sliding_window_size: Some(2),
            has_swa_host_pool: true,
            ..Default::default()
        },
        vec![FULL, SWA],
    );
    let a = insert(&mut tree, vec![10], &[1]);
    let b = insert(&mut tree, vec![10, 20, 30], &[1, 2, 3]);
    let c = insert(&mut tree, vec![10, 40], &[1, 4]);
    for (id, key, start, device, host, full_host) in [
        (a, vec![10], 0, vec![11], vec![101], vec![201]),
        (
            b,
            vec![10, 20, 30],
            1,
            vec![12, 13],
            vec![102, 103],
            vec![202, 203],
        ),
        (c, vec![10, 40], 1, vec![14], vec![104], vec![204]),
    ] {
        tree.attach_swa_window(
            &key,
            Default::default(),
            start,
            key.len(),
            &PageValue::from_vec(device),
        )
        .unwrap();
        tree.commit_backup(
            id,
            PageValue::from_vec(full_host),
            [(
                SWA,
                vec![PoolTransfer {
                    name: PoolName::Swa,
                    host_indices: Some(PageValue::from_vec(host)),
                    nodes_to_load: Some(vec![id]),
                    ..Default::default()
                }],
            )]
            .into_iter()
            .collect(),
        )
        .unwrap();
        let pending = tree.mark_write_through_pending(vec![id], id).unwrap();
        tree.finish_write_through(pending, id).unwrap();
    }
    for id in [b, c, a] {
        tree.demote(id).unwrap();
    }
    let (full, mut auxiliary) = tree.build_load_back_spec(b, None).unwrap();
    auxiliary.get_mut(&SWA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![302, 303]));
    tree.commit_load_back(b, PageValue::from_vec(vec![5, 6, 7]), full, auxiliary)
        .unwrap();
    tree.take_events();
    let nodes = tree.inspect_get_all_node_ids();
    let sizes = (
        tree.total_size(),
        tree.evictable_size(),
        tree.protected_size(),
    );
    // C still needs Full, while its two-atom SWA window crosses A's foreign pin.
    let (full, auxiliary) = tree.build_load_back_spec(c, None).unwrap();
    assert_eq!(full.node_row_counts.as_deref(), Some(&[][..]));
    assert!(auxiliary.is_empty());
    assert!(
        tree.commit_load_back(c, PageValue::from_vec(vec![]), full, auxiliary)
            .unwrap()
            .is_empty()
    );
    assert_eq!(tree.inspect_get_all_node_ids(), nodes);
    assert_eq!(
        (
            tree.total_size(),
            tree.evictable_size(),
            tree.protected_size()
        ),
        sizes
    );
    for id in [a, b, c] {
        let node = tree.arena.node(tree.arena.resolve(id).unwrap());
        assert_eq!(
            node.load_back_pending_id,
            if id == c { None } else { Some(b) }
        );
        assert_eq!(node.has_device_value(FULL), id != c);
        assert_eq!(node.has_device_value(SWA), id == b);
    }
    assert!(tree.take_events().is_empty());
    // An ordinary empty plan must not bypass the missing Full restore.
    assert!(matches!(
        tree.commit_load_back(
            c,
            PageValue::from_vec(vec![]),
            PoolTransfer {
                host_indices: Some(PageValue::from_vec(vec![])),
                nodes_to_load: Some(vec![]),
                node_row_counts: Some(vec![]),
                ..Default::default()
            },
            HashMap::new()
        ),
        Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
    ));
    tree.finish_load_back(b).unwrap();
    let (full, mut auxiliary) = tree.build_load_back_spec(c, None).unwrap();
    assert_eq!(full.node_row_counts.as_deref(), Some(&[1][..]));
    assert_eq!(
        auxiliary[&SWA][0].node_row_counts.as_deref(),
        Some(&[1, 1][..])
    );
    auxiliary.get_mut(&SWA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![301, 304]));
    let actions = tree
        .commit_load_back(c, PageValue::from_vec(vec![8]), full, auxiliary)
        .unwrap();
    let [
        CacheAction::RebuildFullToSwaMapping {
            full_indices,
            swa_indices,
        },
    ] = actions.as_slice()
    else {
        panic!("expected SWA mapping for the replanned window");
    };
    assert_eq!(PageValue::concat(full_indices).as_slice(), &[5, 8]);
    assert_eq!(PageValue::concat(swa_indices).as_slice(), &[301, 304]);
    tree.finish_load_back(c).unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn auxiliary_load_back_rejects_replaced_host_rows_before_any_commit() {
    for component in [SWA, MAMBA] {
        for write_back in [false, true] {
            for replaced in [false, true] {
                let mut tree = Core::new(
                    CacheInitParams {
                        enable_hicache: true,
                        is_write_back: write_back,
                        enable_kv_cache_events: true,
                        swa_sliding_window_size: Some(2),
                        has_swa_host_pool: true,
                        mamba_cache_chunk_size: Some(1),
                        ..Default::default()
                    },
                    vec![FULL, component],
                );
                let key = vec![10, 20];
                let mut p = params(&key, &[1, 2]);
                p.mamba_value = Some(PageValue::from_vec(vec![11]));
                let leaf = tree.insert(&p).last_device_node_id.unwrap();
                if component == SWA {
                    tree.attach_swa_window(
                        &key,
                        Default::default(),
                        0,
                        2,
                        &PageValue::from_vec(vec![11, 12]),
                    )
                    .unwrap();
                }
                backup_components(&mut tree, leaf, component);
                evict_auxiliary(&mut tree, leaf, component);
                let (full, mut auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
                let rows = if component == SWA { 2 } else { 1 };
                auxiliary.get_mut(&component).unwrap()[0].device_indices =
                    Some(PageValue::from_vec((301..301 + rows).collect()));
                if replaced {
                    let reclaim = tree.drive_host_eviction(component, rows as usize);
                    assert_eq!(
                        reclaim.host_frees[&component][0].as_slice(),
                        if component == SWA {
                            &[201, 202][..]
                        } else {
                            &[201][..]
                        }
                    );
                    assert!(reclaim.device_frees.is_empty());
                    if component == SWA {
                        tree.attach_swa_window(
                            &key,
                            Default::default(),
                            0,
                            2,
                            &PageValue::from_vec(vec![21, 22]),
                        )
                        .unwrap();
                    } else {
                        p.mamba_value = Some(PageValue::from_vec(vec![21]));
                        tree.insert(&p);
                    }
                    tree.commit_backup(
                        leaf,
                        PageValue::from_vec(vec![]),
                        [(
                            component,
                            vec![PoolTransfer {
                                host_indices: Some(PageValue::from_vec(
                                    (401..401 + rows).collect(),
                                )),
                                nodes_to_load: Some(vec![leaf]),
                                ..Default::default()
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    )
                    .unwrap();
                    let pending = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
                    tree.finish_write_through(pending, leaf).unwrap();
                    evict_auxiliary(&mut tree, leaf, component);
                }
                tree.take_events();
                let sizes = (
                    tree.total_size(),
                    tree.evictable_size(),
                    tree.protected_size(),
                );
                let lru = tree.inspect_get_component_device_lru_node_ids(component);
                let result =
                    tree.commit_load_back(leaf, PageValue::from_vec(vec![]), full, auxiliary);
                if replaced {
                    assert!(matches!(
                        result,
                        Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
                    ));
                    assert_eq!(
                        (
                            tree.total_size(),
                            tree.evictable_size(),
                            tree.protected_size()
                        ),
                        sizes
                    );
                    assert_eq!(
                        tree.inspect_get_component_device_lru_node_ids(component),
                        lru
                    );
                    let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
                    assert!(!node.has_device_value(component));
                    assert_eq!(
                        node.host_value(component).as_slice(),
                        if component == SWA {
                            &[401, 402][..]
                        } else {
                            &[401][..]
                        }
                    );
                    assert_eq!(node.load_back_pending_id, None);
                    assert!(tree.take_events().is_empty());
                    let (full, mut auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
                    auxiliary.get_mut(&component).unwrap()[0].device_indices =
                        Some(PageValue::from_vec((301..301 + rows).collect()));
                    tree.commit_load_back(leaf, PageValue::from_vec(vec![]), full, auxiliary)
                        .unwrap();
                } else {
                    result.unwrap();
                }
                tree.finish_load_back(leaf).unwrap();
                let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
                assert_eq!(
                    node.device_value(component).as_slice(),
                    if component == SWA {
                        &[301, 302][..]
                    } else {
                        &[301][..]
                    }
                );
                assert_eq!(node.device_value(FULL).as_slice(), &[1, 2]);
                assert_eq!(node.load_back_pending_id, None);
                assert!(tree.take_events().is_empty());
                tree.try_sanity_check(&[], &[]).unwrap();
            }
        }
    }
}

#[test]
fn failed_backup_drops_subtrees_in_stable_postorder() {
    for _ in 0..8 {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                is_write_back: true,
                enable_kv_cache_events: true,
                ..Default::default()
            },
            vec![FULL],
        );
        let parent = insert(&mut tree, vec![10], &[1]);
        let a = insert(&mut tree, vec![10, 20], &[1, 2]);
        let b = insert(&mut tree, vec![10, 30], &[1, 3]);
        let aa = insert(&mut tree, vec![10, 20, 40], &[1, 2, 4]);
        let ab = insert(&mut tree, vec![10, 20, 50], &[1, 2, 5]);
        for (id, slot) in [(a, 102), (b, 103), (aa, 104), (ab, 105)] {
            tree.commit_backup(id, PageValue::from_vec(vec![slot]), HashMap::new())
                .unwrap();
            let nodes = tree.mark_write_through_pending(vec![id], id).unwrap();
            tree.finish_write_through(nodes, id).unwrap();
        }
        for id in [aa, ab, a, b] {
            tree.demote(id).unwrap();
        }
        // Assign explicit hashes so the ordered CPU removal payload is observable.
        for (id, hash) in [
            (a, "0000000000000002"),
            (b, "0000000000000003"),
            (aa, "0000000000000004"),
            (ab, "0000000000000005"),
        ] {
            tree.inspect_set_node_hash_values(id, Some(vec![hash.to_string()]))
                .unwrap();
        }
        let expected_hashes = [aa, ab, a, b].map(|id| {
            let hash = tree.get_hash_values(id).unwrap().remove(0);
            super::node::hash_str_to_int64(&hash)
        });
        tree.take_events();
        let (dropped, result) = tree.drop_subtree_no_host(parent).unwrap();
        assert!(dropped);
        let freed: Vec<_> = result.host_frees[&FULL]
            .iter()
            .flat_map(|v| v.as_slice())
            .copied()
            .collect();
        assert_eq!(freed, vec![104, 105, 102, 103]);
        assert_eq!(
            tree.take_events()
                .into_iter()
                .filter_map(|event| match event {
                    KvCacheEvent::BlockRemoved {
                        block_hashes,
                        medium: StorageMedium::Cpu,
                    } => Some(block_hashes),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![expected_hashes.to_vec()]
        );
        assert_eq!(tree.total_size(), (0, 0));
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn skipped_full_host_receipt_cannot_release_a_later_lock() {
    for write_back in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                is_write_back: write_back,
                ..Default::default()
            },
            vec![FULL],
        );
        let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
        let a = tree.inc_host_lock_ref(leaf).unwrap();
        assert_eq!(a.skipped_lock_components.contains(FULL), !write_back);
        complete_backup(&mut tree, leaf);
        let b = tree.inc_host_lock_ref(leaf).unwrap();
        assert!(!b.skipped_lock_components.contains(FULL));
        tree.dec_host_lock_ref(leaf, &a.to_dec_params()).unwrap();
        let idx = tree.arena.resolve(leaf).unwrap();
        assert_eq!(tree.arena.node(idx).host_lock_ref(FULL), 1);
        tree.dec_host_lock_ref(leaf, &b.to_dec_params()).unwrap();
        assert_eq!(tree.arena.node(idx).host_lock_ref(FULL), 0);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
#[should_panic(expected = "mamba_cache_chunk_size must be positive")]
fn mamba_rejects_zero_chunk_size_at_construction() {
    Core::new(
        CacheInitParams {
            mamba_cache_chunk_size: Some(0),
            ..Default::default()
        },
        vec![FULL, MAMBA],
    );
}

#[test]
fn mamba_adopts_compact_slots_on_new_and_existing_nodes() {
    for existing in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let donor = PageValue::from_vec(vec![101, 102, 103, 104]);
        if existing {
            let key = vec![10, 20, 30, 40];
            let mut p = params(&key, &[1, 2, 3, 4]);
            p.mamba_value = Some(donor.slice(3, 1));
            tree.insert(&p);
            tree.match_prefix(&MatchPrefixParams {
                key: &vec![10, 20],
                namespace: Default::default(),
            });
        }
        let key = vec![10, 20];
        let mut p = params(&key, &[1, 2]);
        p.mamba_value = Some(donor.slice(1, 1));
        let leaf = tree.insert(&p).last_device_node_id.unwrap();
        let idx = tree.arena.resolve(leaf).unwrap();
        let adopted = tree.arena.node(idx).device_value(MAMBA);
        assert_eq!(adopted.as_slice(), &[102]);
        assert_ne!(adopted.as_slice().as_ptr(), donor.as_slice()[1..].as_ptr());
        let stored_pointer = adopted.as_slice().as_ptr();
        // A duplicate donation is not adopted and cannot replace the checkpoint.
        p.mamba_value = Some(donor.slice(2, 1));
        assert!(tree.insert(&p).mamba_exist);
        assert_eq!(
            tree.arena.node(idx).device_value(MAMBA).as_slice().as_ptr(),
            stored_pointer
        );
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn mamba_rejects_invalid_donations_before_insert_mutation() {
    for mode in ["root", "resumable", "continuation", "duplicate"] {
        for donation in [Some(vec![101, 102]), Some(vec![]), None] {
            let mut tree = Core::new(
                CacheInitParams {
                    enable_kv_cache_events: true,
                    mamba_cache_chunk_size: Some(1),
                    ..Default::default()
                },
                vec![FULL, MAMBA],
            );
            let key = vec![10, 20];
            let mut p = params(&key, &[1, 2]);
            p.mamba_value = Some(PageValue::from_vec(vec![100]));
            let leaf = tree.insert(&p).last_device_node_id.unwrap();
            assert!(!tree.take_events().is_empty());
            let idx = tree.arena.resolve(leaf).unwrap();
            let nodes = tree.inspect_get_all_node_ids();
            let access = tree.arena.node(idx).last_access_counter;
            let hits = tree.arena.node(idx).hit_count;
            let lru = tree.inspect_get_component_device_lru_node_ids(MAMBA);
            let key = match mode {
                "continuation" => vec![10, 20, 30],
                "duplicate" => vec![10, 20],
                _ => vec![10, 30], // Would split the existing edge.
            };
            let mut p = params(
                &key,
                if mode == "continuation" {
                    &[3]
                } else {
                    &[3, 4]
                },
            );
            p.mamba_value = donation.map(PageValue::from_vec);
            p.priority = 99;
            let result = match mode {
                "resumable" => tree.try_begin_insert(&p).map(|_| ()),
                "continuation" => tree.try_insert_suffix_from_node(leaf, 2, &p).map(|_| ()),
                _ => tree.try_insert(&p).map(|_| ()),
            };
            assert!(matches!(
                result,
                Err(TreeCoreRuntimeError::InsertMambaValueLength { value_len })
                    if value_len == p.mamba_value.as_ref().map_or(0, RadixValue::len)
            ));
            assert_eq!(tree.inspect_get_all_node_ids(), nodes);
            assert_eq!(tree.arena.node(idx).last_access_counter, access);
            assert_eq!(tree.arena.node(idx).hit_count, hits);
            assert_eq!(tree.inspect_get_component_device_lru_node_ids(MAMBA), lru);
            assert_eq!(tree.total_size(), (2, 1));
            assert_eq!(tree.arena.node(idx).device_value(MAMBA).as_slice(), &[100]);
            assert!(tree.take_events().is_empty());
            tree.try_sanity_check(&[], &[]).unwrap();
            // Rejection must leave the single-flight insert available for retry.
            p.mamba_value = Some(PageValue::from_vec(vec![103]));
            if mode == "continuation" {
                tree.try_insert_suffix_from_node(leaf, 2, &p).unwrap();
            } else {
                tree.try_insert(&p).unwrap();
            }
            tree.try_sanity_check(&[], &[]).unwrap();
        }
    }
}

#[test]
fn empty_mamba_inserts_do_not_adopt_or_validate_unused_donations() {
    let mut tree = Core::new(
        CacheInitParams {
            mamba_cache_chunk_size: Some(1),
            ..Default::default()
        },
        vec![FULL, MAMBA],
    );
    let key = vec![];
    let mut p = params(&key, &[]);
    assert!(tree.try_insert(&p).unwrap().mamba_exist);
    p.mamba_value = Some(PageValue::from_vec(vec![101, 102]));
    assert!(tree.try_insert(&p).unwrap().mamba_exist);
    assert_eq!(tree.total_size(), (0, 0));

    let key = vec![10, 20];
    let mut p = params(&key, &[1, 2]);
    p.mamba_value = Some(PageValue::from_vec(vec![100]));
    let leaf = tree.insert(&p).last_device_node_id.unwrap();
    let mut p = params(&key, &[]);
    p.mamba_value = Some(PageValue::from_vec(vec![101, 102]));
    assert!(
        tree.try_insert_suffix_from_node(leaf, 2, &p)
            .unwrap()
            .mamba_exist
    );
    assert_eq!(tree.total_size(), (2, 1));
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn mamba_empty_continuation_adopts_only_a_missing_checkpoint() {
    for chunked in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_kv_cache_events: true,
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let key = vec![10, 20, 30, 40];
        let mut p = params(&key, &[1, 2, 3, 4]);
        p.mamba_value = Some(PageValue::from_vec(vec![100]));
        let tail = tree.insert(&p).last_device_node_id.unwrap();
        let key = vec![10, 20];
        tree.match_prefix(&MatchPrefixParams {
            key: &key,
            namespace: Default::default(),
        });
        let tail_idx = tree.arena.resolve(tail).unwrap();
        let anchor_idx = tree.arena.node(tail_idx).parent();
        let anchor = tree.arena.node(anchor_idx).id;
        assert!(!tree.arena.has_device_value(anchor_idx, MAMBA));
        tree.take_events();

        let mut p = params(&key, &[]);
        p.chunked = chunked;
        // A nonempty prefix needs a checkpoint even when no Full rows are new.
        for donation in [None, Some(vec![]), Some(vec![101, 102])] {
            let tick = tree.arena.node(anchor_idx).last_access_counter;
            p.mamba_value = donation.map(PageValue::from_vec);
            assert!(matches!(
                tree.try_insert_suffix_from_node(anchor, 2, &p),
                Err(TreeCoreRuntimeError::InsertMambaValueLength { .. })
            ));
            assert_eq!(tree.arena.node(anchor_idx).last_access_counter, tick);
            assert_eq!(tree.total_size(), (4, 1));
            assert!(tree.take_events().is_empty());
        }

        let donor = PageValue::from_vec(vec![101, 102, 103]);
        p.mamba_value = Some(donor.slice(1, 1));
        let result = tree.try_insert_suffix_from_node(anchor, 2, &p).unwrap();
        assert!(!result.mamba_exist);
        assert_eq!((result.prefix_len, result.total_len), (2, 2));
        assert_eq!(result.last_device_node_id, Some(anchor));
        // In particular, do not replay Full overlap frees without donated rows.
        assert!(result.cache_actions.is_empty());
        assert!(tree.take_events().is_empty());
        let adopted = tree.arena.node(anchor_idx).device_value(MAMBA);
        assert_eq!(adopted.as_slice(), &[102]);
        assert_ne!(adopted.as_slice().as_ptr(), donor.as_slice()[1..].as_ptr());
        assert_eq!(tree.total_size(), (4, 2));

        p.mamba_value = Some(PageValue::from_vec(vec![104]));
        assert!(
            tree.try_insert_suffix_from_node(anchor, 2, &p)
                .unwrap()
                .mamba_exist
        );
        assert_eq!(
            tree.arena.node(anchor_idx).device_value(MAMBA).as_slice(),
            &[102]
        );
        assert_eq!(
            tree.collect_full_device_indices(tail, tree.root_node_handle(None))
                .unwrap()
                .as_slice(),
            &[1, 2, 3, 4]
        );
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn unbacked_device_leaf_survives_until_its_final_host_receipt_is_released() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            mamba_cache_chunk_size: Some(1),
            ..Default::default()
        },
        vec![FULL, MAMBA],
    );
    let key = vec![10, 20];
    let mut p = params(&key, &[1, 2]);
    p.mamba_value = Some(PageValue::from_vec(vec![100]));
    let leaf = tree.insert(&p).last_device_node_id.unwrap();
    let first = tree.inc_host_lock_ref(leaf).unwrap();
    let second = tree.inc_host_lock_ref(leaf).unwrap();
    assert!(first.skipped_lock_components.contains(FULL));
    assert!(!first.skipped_lock_components.contains(MAMBA));
    for receipt in [first, second] {
        assert!(!tree.inspect_is_device_evictable_leaf(leaf));
        tree.evict_device_start(FULL, 1);
        let (victim, step) = tree.evict_device_next_node(FULL, &HashMap::new());
        assert_eq!(victim, None);
        assert!(step.device_frees.is_empty());
        tree.evict_device_end(FULL);
        tree.dec_host_lock_ref(leaf, &receipt.to_dec_params())
            .unwrap();
        tree.try_sanity_check(&[], &[]).unwrap();
    }
    assert!(tree.inspect_is_device_evictable_leaf(leaf));
    tree.evict_device_start(FULL, 1);
    let (victim, _) = tree.evict_device_next_node(FULL, &HashMap::new());
    assert_eq!(victim, Some(leaf));
    let (backup, step) = tree.evict_device_leaf(leaf, false).unwrap();
    assert!(backup.is_none());
    assert_eq!(step.device_frees[&FULL][0].as_slice(), &[1, 2]);
    assert!(tree.arena.resolve(leaf).is_err());
    tree.evict_device_end(FULL);
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn backed_device_leaf_can_demote_with_a_live_host_receipt() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            mamba_cache_chunk_size: Some(1),
            ..Default::default()
        },
        vec![FULL, MAMBA],
    );
    let key = vec![10, 20];
    let mut p = params(&key, &[1, 2]);
    p.mamba_value = Some(PageValue::from_vec(vec![100]));
    let leaf = tree.insert(&p).last_device_node_id.unwrap();
    backup_components(&mut tree, leaf, MAMBA);
    let receipt = tree.inc_host_lock_ref(leaf).unwrap();
    assert!(tree.inspect_is_device_evictable_leaf(leaf));
    let (backup, step) = tree.evict_device_leaf(leaf, false).unwrap();
    assert!(backup.is_none());
    assert_eq!(step.device_frees[&FULL][0].as_slice(), &[1, 2]);
    assert!(tree.arena.resolve(leaf).is_ok());
    tree.dec_host_lock_ref(leaf, &receipt.to_dec_params())
        .unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn write_back_can_request_backup_with_a_live_host_receipt() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            is_write_back: true,
            mamba_cache_chunk_size: Some(1),
            ..Default::default()
        },
        vec![FULL, MAMBA],
    );
    let key = vec![10, 20];
    let mut p = params(&key, &[1, 2]);
    p.mamba_value = Some(PageValue::from_vec(vec![100]));
    let leaf = tree.insert(&p).last_device_node_id.unwrap();
    let receipt = tree.inc_host_lock_ref(leaf).unwrap();
    assert!(tree.inspect_is_device_evictable_leaf(leaf));
    let (backup, step) = tree.evict_device_leaf(leaf, true).unwrap();
    assert!(backup.is_some());
    assert!(step.device_frees.is_empty());
    tree.dec_host_lock_ref(leaf, &receipt.to_dec_params())
        .unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn swa_continuation_reports_the_full_branch_boundary() {
    let mut tree = Core::new(
        CacheInitParams {
            swa_sliding_window_size: Some(8),
            ..Default::default()
        },
        vec![FULL, SWA],
    );
    let anchor = insert(&mut tree, (0..64).collect(), &(0..64).collect::<Vec<_>>());
    let key = (0..72).collect();
    let mut p = params(&key, &(64..72).collect::<Vec<_>>());
    p.prev_prefix_len = 64;
    p.swa_branching_seqlen = Some(72);
    let result = tree.insert_suffix_from_node(anchor, 64, &p);
    assert_eq!(result.total_len, 72);
    assert!(result.swa_branch_inserted);
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn scoped_review_empty_swa_continuation_reports_reached_boundaries() {
    for boundary in [None, Some(56), Some(64), Some(72)] {
        let mut tree = Core::new(
            CacheInitParams {
                swa_sliding_window_size: Some(8),
                ..Default::default()
            },
            vec![FULL, SWA],
        );
        let key = (0..64).collect();
        let anchor = insert(&mut tree, key, &(0..64).collect::<Vec<_>>());
        let key = (0..64).collect();
        let mut p = params(&key, &[]);
        p.prev_prefix_len = 64;
        p.swa_branching_seqlen = boundary;
        let result = tree.insert_suffix_from_node(anchor, 64, &p);
        assert_eq!(result.total_len, 64);
        assert_eq!(result.prefix_len, 64);
        assert_eq!(result.last_device_node_id, Some(anchor));
        assert_eq!(
            result.swa_branch_inserted,
            boundary.is_some_and(|length| length <= 64),
            "boundary {boundary:?}"
        );
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_empty_continuation_reports_a_branch_only_with_swa() {
    for with_swa in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                swa_sliding_window_size: with_swa.then_some(2),
                ..Default::default()
            },
            if with_swa {
                vec![FULL, SWA]
            } else {
                vec![FULL]
            },
        );
        let key = vec![10, 20];
        let anchor = insert(&mut tree, key.clone(), &[1, 2]);
        let mut p = params(&key, &[]);
        p.prev_prefix_len = 2;
        p.swa_branching_seqlen = Some(2);
        let result = tree.insert_suffix_from_node(anchor, 2, &p);
        assert_eq!(result.swa_branch_inserted, with_swa);
        assert_eq!(result.last_device_node_id, Some(anchor));
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn early_swa_release_honors_skip_and_releases_acquired_mamba() {
    let mut tree = Core::new(
        CacheInitParams {
            swa_sliding_window_size: Some(2),
            mamba_cache_chunk_size: Some(1),
            ..Default::default()
        },
        vec![FULL, SWA, MAMBA],
    );
    let key = vec![10, 20];
    let mut p = params(&key, &[1, 2]);
    p.mamba_value = Some(PageValue::from_vec(vec![101]));
    let leaf = tree.insert(&p).last_device_node_id.unwrap();
    let receipt = tree
        .inc_lock_ref(leaf, [SWA].into_iter().collect())
        .unwrap();
    assert_eq!(tree.mamba_protected_size(), 1);
    tree.dec_swa_lock_only(
        leaf,
        &receipt.to_dec_params(),
        &mut HashMap::new(),
        &mut HashMap::new(),
    )
    .unwrap();
    assert_eq!(tree.mamba_protected_size(), 0);
    assert_eq!(tree.full_protected_size(), 2);
    tree.dec_lock_ref(leaf, &receipt.to_dec_params(), true)
        .unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[rstest::rstest]
fn continuation_rejects_a_wrong_key_before_side_effects(#[values(false, true)] chunked: bool) {
    let mut tree = Core::new(CacheInitParams::default(), vec![FULL]);
    let anchor = insert(&mut tree, vec![10, 20], &[1, 2]);
    let tick = tree.inspect_get_node_access_counter(anchor).unwrap();
    let key = vec![30, 40, 50];
    let mut p = params(&key, &[3]);
    p.chunked = chunked;
    assert!(tree.try_insert_suffix_from_node(anchor, 2, &p).is_err());
    assert_eq!(tree.inspect_get_node_access_counter(anchor).unwrap(), tick);
    assert_eq!(tree.evictable_size(), 2);
    assert!(tree.inspect_get_child_node_ids(anchor).unwrap().is_empty());
    let valid_key = vec![10, 20, 50];
    let mut valid = params(&valid_key, &[3]);
    valid.chunked = chunked;
    let result = tree.try_insert_suffix_from_node(anchor, 2, &valid).unwrap();
    assert_eq!(result.total_len, 3);
    assert_eq!(tree.full_kv_prefix_len(&valid_key, Default::default()), 3);
    assert_eq!(tree.full_kv_prefix_len(&key, Default::default()), 0);
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn empty_continuation_returns_retained_prefix_backup_actions() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            write_through_threshold: 2,
            ..Default::default()
        },
        vec![FULL],
    );
    let anchor = insert(&mut tree, vec![10, 20], &[1, 2]);
    let key = vec![10, 20];
    let result = tree.insert_suffix_from_node(anchor, 2, &params(&key, &[]));
    let backups: Vec<_> = result
        .cache_actions
        .iter()
        .filter_map(|action| match action {
            CacheAction::BackupKV(backup) => Some(backup),
            _ => None,
        })
        .collect();
    assert_eq!(backups.len(), 1);
    assert_eq!(backups[0].node_ids, vec![anchor]);
}

#[test]
fn deferred_demote_waits_for_device_locks_then_completes() {
    // Exercise both Full-only and Full+Mamba lock receipts.
    for with_mamba in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            if with_mamba {
                vec![FULL, MAMBA]
            } else {
                vec![FULL]
            },
        );
        let key = vec![10, 20];
        let mut p = params(&key, &[1, 2]);
        p.mamba_value = with_mamba.then(|| PageValue::from_vec(vec![201]));
        let leaf = tree.insert(&p).last_device_node_id.unwrap();
        complete_backup(&mut tree, leaf);
        let lock = tree.inc_lock_ref(leaf, ComponentSet::EMPTY).unwrap();
        let deferred = tree.demote(leaf).unwrap();
        assert!(deferred.device_frees.is_empty());
        assert_eq!(tree.full_protected_size(), 2);
        assert_eq!(tree.total_size(), (2, usize::from(with_mamba)));
        tree.try_sanity_check(&[], &[]).unwrap();
        tree.dec_lock_ref(leaf, &lock.to_dec_params(), false)
            .unwrap();
        let completed = tree.demote(leaf).unwrap();
        assert_eq!(completed.device_frees[&FULL][0].as_slice(), &[1, 2]);
        assert_eq!(tree.total_size(), (0, 0));
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn demote_preserves_invalid_state_errors_when_locked() {
    let mut tree = Core::new(CacheInitParams::default(), vec![FULL]);
    let root = tree.root_node_handle(None);
    assert!(matches!(
        tree.demote(root),
        Err(TreeCoreRuntimeError::InvalidDemoteState { .. })
    ));
    let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
    let lock = tree.inc_lock_ref(leaf, ComponentSet::EMPTY).unwrap();
    assert!(matches!(
        tree.demote(leaf),
        Err(TreeCoreRuntimeError::InvalidDemoteState { .. })
    ));
    tree.dec_lock_ref(leaf, &lock.to_dec_params(), false)
        .unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn swa_attach_rejects_host_only_full_before_splitting() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            swa_sliding_window_size: Some(2),
            ..Default::default()
        },
        vec![FULL, SWA],
    );
    let key = vec![10, 20];
    let mut p = params(&key, &[1, 2]);
    p.swa_evicted_seqlen = 2;
    let leaf = tree.insert(&p).last_device_node_id.unwrap();
    complete_backup(&mut tree, leaf);
    tree.demote(leaf).unwrap();
    assert_eq!(
        tree.swa_tombstone_ranges(&key, Default::default(), 0, 2)
            .unwrap(),
        vec![(0, 2)]
    );
    let tick = tree.inspect_get_node_access_counter(leaf).unwrap();
    assert!(
        tree.attach_swa_window(
            &key,
            Default::default(),
            1,
            2,
            &PageValue::from_vec(vec![5])
        )
        .is_err()
    );
    assert_eq!(tree.inspect_get_node_key_length(leaf).unwrap(), 2);
    assert_eq!(tree.inspect_get_node_access_counter(leaf).unwrap(), tick);
    assert_eq!(tree.swa_evictable_size(), 0);
    tree.try_sanity_check(&[], &[]).unwrap();
}

fn backup_components(tree: &mut Core, leaf: NodeId, auxiliary: ComponentType) {
    let transfer = PoolTransfer {
        name: if auxiliary == MAMBA {
            PoolName::Mamba
        } else {
            PoolName::Swa
        },
        host_indices: Some(PageValue::from_vec(if auxiliary == MAMBA {
            vec![201]
        } else {
            vec![201, 202]
        })),
        nodes_to_load: Some(vec![leaf]),
        ..Default::default()
    };
    tree.commit_backup(
        leaf,
        PageValue::from_vec(vec![101, 102]),
        [(auxiliary, vec![transfer])].into_iter().collect(),
    )
    .unwrap();
    let nodes = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
    tree.finish_write_through(nodes, leaf).unwrap();
}

fn evict_auxiliary(tree: &mut Core, leaf: NodeId, auxiliary: ComponentType) {
    // A Full-only request lock leaves the auxiliary pool independently evictable.
    let full_lock = tree
        .inc_lock_ref(leaf, [auxiliary].into_iter().collect())
        .unwrap();
    tree.evict_device_start(auxiliary, 1);
    let (victim, step) = tree.evict_device_next_node(auxiliary, &HashMap::new());
    assert_eq!(victim, None);
    assert!(!step.device_frees[&auxiliary].is_empty());
    tree.evict_device_end(auxiliary);
    tree.dec_lock_ref(leaf, &full_lock.to_dec_params(), false)
        .unwrap();
}

#[test]
fn full_host_hits_equal_load_back_rows_with_a_host_only_mamba_checkpoint() {
    for full_host_only in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let key = vec![10, 20];
        let mut p = params(&key, &[1, 2]);
        p.mamba_value = Some(PageValue::from_vec(vec![11]));
        let leaf = tree.insert(&p).last_device_node_id.unwrap();
        backup_components(&mut tree, leaf, MAMBA);
        evict_auxiliary(&mut tree, leaf, MAMBA);
        if full_host_only {
            tree.demote(leaf).unwrap();
        }
        let matched = tree.match_prefix(&MatchPrefixParams {
            key: &key,
            namespace: Default::default(),
        });
        assert_eq!(matched.best_match_node_id, leaf);
        assert_eq!(matched.mamba_host_hit_length, 1);
        let request = Req {
            mamba_pool_idx: Some(PageValue::from_vec(vec![301])),
        };
        let (full_transfer, mut auxiliary) =
            tree.build_load_back_spec(leaf, Some(&request)).unwrap();
        let rows = full_transfer.host_indices.as_ref().unwrap().len();
        assert_eq!(rows, if full_host_only { 2 } else { 0 });
        assert_eq!(matched.host_hit_length, rows);
        assert_eq!(matched.mamba_branching_seqlen, None);
        assert_eq!(
            auxiliary[&MAMBA][0]
                .host_indices
                .as_ref()
                .unwrap()
                .as_slice(),
            &[201]
        );
        // The controller allocates a cache slot for the planned checkpoint
        // restore, separately from the request's preallocated CoW slot.
        auxiliary.get_mut(&MAMBA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![302]));
        let full_values = if full_host_only { vec![3, 4] } else { vec![] };
        tree.commit_load_back(
            leaf,
            PageValue::from_vec(full_values),
            full_transfer,
            auxiliary,
        )
        .unwrap();
        tree.finish_load_back(leaf).unwrap();
        let loaded = tree.match_prefix(&MatchPrefixParams {
            key: &key,
            namespace: Default::default(),
        });
        assert_eq!(loaded.device_indices.len(), 2);
        assert_eq!(loaded.host_hit_length, 0);
        assert_eq!(loaded.mamba_host_hit_length, 0);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn host_match_branch_boundaries_are_independent_of_component_order() {
    for auxiliary in [MAMBA, SWA] {
        for components in [vec![FULL, auxiliary], vec![auxiliary, FULL]] {
            let mut tree = Core::new(
                CacheInitParams {
                    enable_hicache: true,
                    mamba_cache_chunk_size: Some(1),
                    swa_sliding_window_size: Some(2),
                    has_swa_host_pool: true,
                    ..Default::default()
                },
                components,
            );
            let key = vec![10, 20];
            let mut p = params(&key, &[1, 2]);
            p.mamba_value = (auxiliary == MAMBA).then(|| PageValue::from_vec(vec![11]));
            let leaf = tree.insert(&p).last_device_node_id.unwrap();
            if auxiliary == SWA {
                // The controller materializes SWA after the insert's rebuild action.
                tree.attach_swa_window(
                    &key,
                    Default::default(),
                    0,
                    2,
                    &PageValue::from_vec(vec![11, 12]),
                )
                .unwrap();
            }
            backup_components(&mut tree, leaf, auxiliary);
            tree.demote(leaf).unwrap();
            let matched = tree.match_prefix(&MatchPrefixParams {
                key: &key,
                namespace: Default::default(),
            });
            assert_eq!(matched.best_match_node_id, leaf);
            assert!(matched.device_indices.is_empty());
            assert_eq!(matched.host_hit_length, 2);
            assert_eq!(matched.mamba_branching_seqlen, None);
            assert_eq!(matched.swa_branching_seqlen, None);
            let (full, auxiliary_transfers) = tree.build_load_back_spec(leaf, None).unwrap();
            assert_eq!(full.host_indices.unwrap().len(), 2);
            if auxiliary == SWA {
                assert_eq!(
                    auxiliary_transfers[&SWA][0]
                        .host_indices
                        .as_ref()
                        .unwrap()
                        .len(),
                    2
                );
            }
            tree.try_sanity_check(&[], &[]).unwrap();
        }
    }
}

#[test]
fn auxiliary_eviction_cursor_rechecks_locks_acquired_after_start() {
    for auxiliary in [MAMBA, SWA] {
        let mut tree = Core::new(
            CacheInitParams {
                mamba_cache_chunk_size: Some(1),
                swa_sliding_window_size: Some(2),
                ..Default::default()
            },
            vec![FULL, auxiliary],
        );
        let key = vec![10, 20];
        let mut p = params(&key, &[1, 2]);
        p.mamba_value = (auxiliary == MAMBA).then(|| PageValue::from_vec(vec![11]));
        let leaf = tree.insert(&p).last_device_node_id.unwrap();
        if auxiliary == SWA {
            tree.attach_swa_window(
                &key,
                Default::default(),
                0,
                2,
                &PageValue::from_vec(vec![11, 12]),
            )
            .unwrap();
        }
        tree.evict_device_start(auxiliary, 1);
        let lock = tree.inc_lock_ref(leaf, ComponentSet::EMPTY).unwrap();
        let (victim, step) = tree.evict_device_next_node(auxiliary, &HashMap::new());
        assert_eq!(victim, None);
        assert!(step.device_frees.is_empty());
        assert!(step.tracker.is_empty());
        assert_eq!(
            tree.component_protected_size(auxiliary),
            if auxiliary == MAMBA { 1 } else { 2 }
        );
        tree.evict_device_end(auxiliary);
        tree.dec_lock_ref(leaf, &lock.to_dec_params(), false)
            .unwrap();
        evict_auxiliary(&mut tree, leaf, auxiliary);
        assert_eq!(tree.component_evictable_size(auxiliary), 0);
        assert_eq!(tree.full_evictable_size(), 2);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn ring_layout_host_match_builds_a_full_only_load_back() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            swa_req_ring: true,
            has_swa_host_pool: true,
            swa_sliding_window_size: Some(2),
            ..Default::default()
        },
        vec![FULL, SWA],
    );
    let key = vec![10, 20];
    let mut p = params(&key, &[1, 2]);
    p.swa_evicted_seqlen = 2;
    let leaf = tree.insert(&p).last_device_node_id.unwrap();
    complete_backup(&mut tree, leaf);
    tree.demote(leaf).unwrap();
    let matched = tree.match_prefix(&MatchPrefixParams {
        key: &key,
        namespace: Default::default(),
    });
    assert_eq!(matched.best_match_node_id, leaf);
    assert_eq!(matched.host_hit_length, 2);
    let (full, auxiliary) = tree
        .build_load_back_spec(matched.best_match_node_id, None)
        .unwrap();
    assert_eq!(full.host_indices.as_ref().unwrap().as_slice(), &[101, 102]);
    assert!(!auxiliary.contains_key(&SWA));
    tree.commit_load_back(leaf, PageValue::from_vec(vec![3, 4]), full, auxiliary)
        .unwrap();
    tree.finish_load_back(leaf).unwrap();
    let matched = tree.match_prefix(&MatchPrefixParams {
        key: &key,
        namespace: Default::default(),
    });
    assert_eq!(matched.device_indices.as_slice(), &[3, 4]);
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn deferred_demote_rechecks_leaf_eligibility_after_an_insert() {
    for grows_child in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                ..Default::default()
            },
            vec![FULL],
        );
        let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
        complete_backup(&mut tree, leaf);
        tree.evict_device_start(FULL, 1);
        assert_eq!(
            tree.evict_device_next_node(FULL, &HashMap::new()).0,
            Some(leaf)
        );
        tree.evict_device_end(FULL);
        let child = grows_child.then(|| {
            let child = insert(&mut tree, vec![10, 20, 30, 40], &[1, 2, 3, 4]);
            let lock = tree.inc_lock_ref(child, ComponentSet::EMPTY).unwrap();
            tree.dec_lock_ref(child, &lock.to_dec_params(), false)
                .unwrap();
            child
        });
        let step = tree.demote(leaf).unwrap();
        if let Some(child) = child {
            assert!(step.device_frees.is_empty());
            assert_eq!(tree.full_evictable_size(), 4);
            let matched = tree.match_prefix(&MatchPrefixParams {
                key: &vec![10, 20, 30, 40],
                namespace: Default::default(),
            });
            assert_eq!(matched.device_indices.as_slice(), &[1, 2, 3, 4]);
            let lock = tree
                .inc_lock_ref(matched.last_device_node_id, ComponentSet::EMPTY)
                .unwrap();
            tree.dec_lock_ref(child, &lock.to_dec_params(), false)
                .unwrap();
            tree.evict_device_start(FULL, 1);
            assert_eq!(
                tree.evict_device_next_node(FULL, &HashMap::new()).0,
                Some(child)
            );
            tree.evict_device_end(FULL);
        } else {
            assert_eq!(step.device_frees[&FULL][0].as_slice(), &[1, 2]);
            assert_eq!(tree.full_evictable_size(), 0);
        }
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn mru_walk_refreshes_buried_improvements_before_selecting() {
    for touch_a in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                eviction_policy: "mru".into(),
                ..Default::default()
            },
            vec![FULL],
        );
        let a = insert(&mut tree, vec![10, 20], &[1, 2]);
        let b = insert(&mut tree, vec![30, 40], &[3, 4]);
        tree.evict_device_start(FULL, 1);
        if touch_a {
            tree.match_prefix(&MatchPrefixParams {
                key: &vec![10, 20],
                namespace: Default::default(),
            });
        }
        let (victim, _) = tree.evict_device_next_node(FULL, &HashMap::new());
        assert_eq!(victim, Some(if touch_a { a } else { b }));
        tree.evict_device_end(FULL);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_mamba_walk_uses_current_lru_after_a_match() {
    for touch_a in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let a_key = vec![10, 20];
        let mut p = params(&a_key, &[1, 2]);
        p.mamba_value = Some(PageValue::from_vec(vec![100]));
        let a = tree.insert(&p).last_device_node_id.unwrap();
        let b_key = vec![30, 40];
        let mut p = params(&b_key, &[3, 4]);
        p.mamba_value = Some(PageValue::from_vec(vec![101]));
        let b = tree.insert(&p).last_device_node_id.unwrap();
        tree.evict_device_start(MAMBA, 1);
        if touch_a {
            tree.match_prefix(&MatchPrefixParams {
                key: &a_key,
                namespace: Default::default(),
            });
        }
        let (victim, _) = tree.evict_device_next_node(MAMBA, &HashMap::new());
        let expected = if touch_a { b } else { a };
        assert_eq!(victim, Some(expected));
        let (_, step) = tree.evict_device_leaf(expected, false).unwrap();
        assert_eq!(
            step.device_frees[&MAMBA][0].as_slice(),
            &[if touch_a { 101 } else { 100 }]
        );
        tree.evict_device_end(MAMBA);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_write_back_lfu_counts_hits_without_scheduling_backups() {
    for write_back in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                is_write_back: write_back,
                eviction_policy: "lfu".into(),
                write_through_threshold: 2,
                ..Default::default()
            },
            vec![FULL],
        );
        let key = vec![10, 20];
        let a = insert(&mut tree, key.clone(), &[1, 2]);
        for _ in 0..2 {
            let result = tree.insert(&params(&key, &[3, 4]));
            assert_eq!(
                result
                    .cache_actions
                    .iter()
                    .any(|a| matches!(a, CacheAction::BackupKV(_))),
                !write_back
            );
        }
        assert_eq!(tree.inspect_get_node_hit_count(a).unwrap(), 3);
        let b = insert(&mut tree, vec![30, 40], &[5, 6]);
        tree.evict_device_start(FULL, 1);
        assert_eq!(
            tree.evict_device_next_node(FULL, &HashMap::new()).0,
            Some(b)
        );
        tree.evict_device_end(FULL);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_resumed_insert_refreshes_the_active_full_walk() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            write_through_threshold: 2,
            ..Default::default()
        },
        vec![FULL],
    );
    let parent = insert(&mut tree, vec![10, 20], &[1, 2]);
    let key = vec![10, 20, 30, 40];
    let step = tree.begin_insert(&params(&key, &[3, 4, 5, 6]));
    assert!(step.result.is_none());
    assert!(step.actions.iter().any(|action| matches!(action,
        CacheAction::BackupKV(backup) if backup.node_ids == vec![parent])));
    // A failed backup adopts no rows. Starting eviction while the insert is
    // suspended snapshots the parent, before the suffix leaf exists.
    tree.evict_device_start(FULL, 1);
    let result = tree.try_resume_insert().unwrap().result.unwrap();
    let leaf = result.last_device_node_id.unwrap();
    assert_ne!(leaf, parent);
    assert_eq!(
        tree.evict_device_next_node(FULL, &HashMap::new()).0,
        Some(leaf)
    );
    let (_, step) = tree.evict_device_leaf(leaf, false).unwrap();
    assert_eq!(step.device_frees[&FULL][0].as_slice(), &[5, 6]);
    tree.evict_device_end(FULL);
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn scoped_review_full_backup_refreshes_eligibility_with_a_live_host_receipt() {
    for generic_entrypoint in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let key = vec![10, 20];
        let mut p = params(&key, &[1, 2]);
        p.mamba_value = Some(PageValue::from_vec(vec![100]));
        let leaf = tree.insert(&p).last_device_node_id.unwrap();
        let receipt = tree.inc_host_lock_ref(leaf).unwrap();
        assert!(receipt.skipped_lock_components.contains(FULL));
        tree.evict_device_start(FULL, 1);
        assert_eq!(tree.evict_device_next_node(FULL, &HashMap::new()).0, None);
        commit_staged_backup(
            &mut tree,
            leaf,
            vec![101, 102],
            HashMap::new(),
            generic_entrypoint,
        )
        .unwrap();
        let nodes = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
        tree.finish_write_through(nodes, leaf).unwrap();
        assert_eq!(
            tree.evict_device_next_node(FULL, &HashMap::new()).0,
            Some(leaf)
        );
        let (backup, step) = tree.evict_device_leaf(leaf, false).unwrap();
        assert!(backup.is_none());
        assert_eq!(step.device_frees[&FULL][0].as_slice(), &[1, 2]);
        tree.evict_device_end(FULL);
        assert!(tree.arena.resolve(leaf).is_ok());
        tree.dec_host_lock_ref(leaf, &receipt.to_dec_params())
            .unwrap();
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn namespaces_round_trip_without_changing_child_lookup_hashes() {
    use std::hash::{Hash, Hasher};
    for extra_key in [None, Some("tenant"), Some("")] {
        for cache_salt in [None, Some("salt"), Some("")] {
            let namespace = KeyNamespaceRef::new(extra_key, cache_salt);
            assert_eq!(namespace.extra_key(), extra_key);
            assert_eq!(namespace.cache_salt(), cache_salt.filter(|s| !s.is_empty()));
            let owned = namespace.to_owned();
            let borrowed = owned.as_ref();
            assert_eq!(namespace, borrowed);
            let hash = |value: KeyNamespaceRef<'_>| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                value.hash(&mut h);
                h.finish()
            };
            assert_eq!(hash(namespace), hash(borrowed));
            let mut tree = Core::new(CacheInitParams::default(), vec![FULL]);
            let key = vec![10, 20];
            let mut p = params(&key, &[1, 2]);
            p.namespace = namespace;
            tree.insert(&p);
            assert_eq!(
                tree.match_prefix(&MatchPrefixParams {
                    key: &key,
                    namespace: borrowed
                })
                .device_indices
                .as_slice(),
                &[1, 2]
            );
        }
    }
}

#[test]
fn write_through_publishes_only_the_matching_ack_once() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            enable_kv_cache_events: true,
            ..Default::default()
        },
        vec![FULL],
    );
    let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
    tree.take_events();
    tree.commit_backup(leaf, PageValue::from_vec(vec![101, 102]), HashMap::new())
        .unwrap();
    let nodes = tree.mark_write_through_pending(vec![leaf], 7).unwrap();
    tree.finish_write_through(nodes.clone(), 6).unwrap();
    assert!(tree.take_events().is_empty());
    let idx = tree.arena.resolve(leaf).unwrap();
    assert_eq!(tree.arena.node(idx).write_through_pending_id, Some(7));
    tree.finish_write_through(nodes.clone(), 7).unwrap();
    let events = tree.take_events();
    assert_eq!(events.len(), 1);
    assert!(
        matches!(&events[0], KvCacheEvent::BlockStored { medium: StorageMedium::Cpu, block_hashes, .. } if block_hashes.len() == 2)
    );
    tree.finish_write_through(nodes, 7).unwrap();
    assert!(tree.take_events().is_empty());
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn load_back_rejects_a_split_plan_before_adopting_any_rows() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            is_write_back: true,
            ..Default::default()
        },
        vec![FULL],
    );
    let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
    complete_backup(&mut tree, leaf);
    tree.demote(leaf).unwrap();
    let lock = tree.inc_lock_ref(leaf, Default::default()).unwrap();
    let (plan, aux) = tree.build_load_back_spec(leaf, None).unwrap();
    assert_eq!(plan.node_row_counts.as_deref(), Some(&[2][..]));
    let prefix = tree
        .match_prefix(&MatchPrefixParams {
            key: &vec![10],
            namespace: Default::default(),
        })
        .best_match_node_id;
    assert!(matches!(
        tree.commit_load_back(leaf, PageValue::from_vec(vec![3, 4]), plan, aux),
        Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
    ));
    for id in [prefix, leaf] {
        let node = tree.arena.node(tree.arena.resolve(id).unwrap());
        assert!(!node.has_device_value(FULL));
        assert_eq!(node.load_back_pending_id, None);
    }
    // Rebuild after the split: all transferred rows must reach the right fragment.
    let (plan, aux) = tree.build_load_back_spec(leaf, None).unwrap();
    tree.commit_load_back(leaf, PageValue::from_vec(vec![3, 4]), plan, aux)
        .unwrap();
    tree.finish_load_back(leaf).unwrap();
    assert_eq!(
        tree.arena
            .node(tree.arena.resolve(prefix).unwrap())
            .device_value(FULL)
            .as_slice(),
        &[3]
    );
    assert_eq!(
        tree.arena
            .node(tree.arena.resolve(leaf).unwrap())
            .device_value(FULL)
            .as_slice(),
        &[4]
    );
    tree.dec_lock_ref(leaf, &lock.to_dec_params(), false)
        .unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn load_back_rejects_incomplete_and_excess_device_allocations() {
    for rows in [1, 3] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                ..Default::default()
            },
            vec![FULL],
        );
        let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
        complete_backup(&mut tree, leaf);
        tree.demote(leaf).unwrap();
        let (plan, aux) = tree.build_load_back_spec(leaf, None).unwrap();
        assert!(
            tree.commit_load_back(leaf, PageValue::from_vec(vec![3; rows]), plan, aux)
                .is_err()
        );
        assert!(tree.is_full_device_evicted(leaf).unwrap());
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_load_back_refreshes_every_restored_parent_before_eviction() {
    for write_back in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                is_write_back: write_back,
                ..Default::default()
            },
            vec![FULL],
        );
        let parent = insert(&mut tree, vec![10, 20], &[1, 2]);
        complete_backup(&mut tree, parent);
        let middle = insert(&mut tree, vec![10, 20, 30, 40], &[1, 2, 3, 4]);
        complete_backup(&mut tree, middle);
        let leaf = insert(&mut tree, vec![10, 20, 30, 40, 50, 60], &[1, 2, 3, 4, 5, 6]);
        complete_backup(&mut tree, leaf);
        tree.demote(leaf).unwrap();
        tree.demote(middle).unwrap();
        tree.try_sanity_check(&[], &[]).unwrap();

        let (full, auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
        assert_eq!(full.node_row_counts.as_deref(), Some(&[2, 2][..]));
        tree.commit_load_back(
            leaf,
            PageValue::from_vec(vec![31, 32, 41, 42]),
            full,
            auxiliary,
        )
        .unwrap();
        tree.finish_load_back(leaf).unwrap();
        assert_eq!(
            tree.full_device_eviction_candidates()
                .map(|candidate| candidate.node_id)
                .collect::<Vec<_>>(),
            vec![leaf]
        );
        tree.try_sanity_check(&[], &[]).unwrap();

        for (expected, pages) in [(leaf, [41, 42]), (middle, [31, 32]), (parent, [1, 2])] {
            tree.evict_device_start(FULL, 1);
            let (victim, _) = tree.evict_device_next_node(FULL, &HashMap::new());
            assert_eq!(victim, Some(expected));
            let (backup, step) = tree.evict_device_leaf(expected, write_back).unwrap();
            assert!(backup.is_none());
            assert_eq!(step.device_frees[&FULL][0].as_slice(), &pages);
            tree.evict_device_end(FULL);
            tree.try_sanity_check(&[], &[]).unwrap();
        }
        assert_eq!(tree.full_evictable_size(), 0);
    }
}

#[test]
fn scoped_review_suspended_insert_rejects_evicted_and_reused_cursor() {
    for reuse_slot in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                write_through_threshold: 2,
                ..Default::default()
            },
            vec![FULL],
        );
        let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
        let other = insert(&mut tree, vec![30, 40], &[3, 4]);
        let saved_slot = tree.arena.resolve(leaf).unwrap();
        let key = vec![10, 20, 50];
        let step = tree.try_begin_insert(&params(&key, &[5, 6, 7])).unwrap();
        assert!(step.result.is_none());
        assert!(step.actions.iter().any(
            |action| matches!(action, CacheAction::BackupKV(backup) if backup.node_ids == vec![leaf])
        ));
        assert!(tree.has_ongoing_insert());

        // Keep the other branch newer, so the ordinary walk selects the saved cursor.
        tree.match_prefix(&MatchPrefixParams {
            key: &vec![30, 40],
            namespace: Default::default(),
        });
        tree.evict_device_start(FULL, 1);
        assert_eq!(
            tree.evict_device_next_node(FULL, &HashMap::new()).0,
            Some(leaf)
        );
        let (backup, step) = tree.evict_device_leaf(leaf, false).unwrap();
        assert!(backup.is_none());
        assert_eq!(step.device_frees[&FULL][0].as_slice(), &[1, 2]);
        tree.evict_device_end(FULL);
        if reuse_slot {
            // A normal partial match allocates a split prefix into the freed slot.
            let split = tree
                .match_prefix(&MatchPrefixParams {
                    key: &vec![30],
                    namespace: Default::default(),
                })
                .best_match_node_id;
            assert_eq!(tree.arena.resolve(split).unwrap(), saved_slot);
            assert_ne!(split, leaf);
        }
        let tick = tree.inspect_get_node_access_counter(other).unwrap();
        assert!(matches!(
            tree.try_resume_insert(),
            Err(TreeCoreRuntimeError::NodeAccess(NodeAccessError { node_id })) if node_id == leaf
        ));
        assert_eq!(tree.inspect_get_node_access_counter(other).unwrap(), tick);
        assert_eq!(tree.full_kv_prefix_len(&key, Default::default()), 0);
        assert_eq!(
            tree.full_kv_prefix_len(&vec![30, 40, 50], Default::default()),
            2
        );
        tree.end_insert();
        let result = tree.try_insert(&params(&key, &[5, 6, 7])).unwrap();
        assert_eq!(result.total_len, 3);
        assert_eq!(tree.full_kv_prefix_len(&key, Default::default()), 3);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_load_back_ack_refreshes_the_active_full_walk() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            is_write_back: true,
            ..Default::default()
        },
        vec![FULL],
    );
    let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
    complete_backup(&mut tree, leaf);
    tree.demote(leaf).unwrap();
    let (full, auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
    tree.commit_load_back(leaf, PageValue::from_vec(vec![3, 4]), full, auxiliary)
        .unwrap();
    tree.evict_device_start(FULL, 1);
    assert_eq!(tree.evict_device_next_node(FULL, &HashMap::new()).0, None);
    tree.finish_load_back(leaf).unwrap();
    assert_eq!(
        tree.evict_device_next_node(FULL, &HashMap::new()).0,
        Some(leaf)
    );
    let (backup, result) = tree.evict_device_leaf(leaf, true).unwrap();
    assert!(backup.is_none());
    assert_eq!(result.device_frees[&FULL][0].as_slice(), &[3, 4]);
    assert_eq!(result.unbacked_tokens, 0);
    tree.evict_device_end(FULL);
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn scoped_review_final_auxiliary_host_receipt_refreshes_the_active_full_walk() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            mamba_cache_chunk_size: Some(1),
            ..Default::default()
        },
        vec![FULL, MAMBA],
    );
    let key = vec![10, 20];
    let mut p = params(&key, &[1, 2]);
    p.mamba_value = Some(PageValue::from_vec(vec![100]));
    let leaf = tree.insert(&p).last_device_node_id.unwrap();
    let first = tree.inc_host_lock_ref(leaf).unwrap();
    let second = tree.inc_host_lock_ref(leaf).unwrap();
    tree.evict_device_start(FULL, 1);
    assert_eq!(tree.evict_device_next_node(FULL, &HashMap::new()).0, None);
    tree.dec_host_lock_ref(leaf, &first.to_dec_params())
        .unwrap();
    assert_eq!(tree.evict_device_next_node(FULL, &HashMap::new()).0, None);
    tree.dec_host_lock_ref(leaf, &second.to_dec_params())
        .unwrap();
    assert_eq!(
        tree.evict_device_next_node(FULL, &HashMap::new()).0,
        Some(leaf)
    );
    let (_, result) = tree.evict_device_leaf(leaf, false).unwrap();
    assert_eq!(result.device_frees[&FULL][0].as_slice(), &[1, 2]);
    assert_eq!(result.unbacked_tokens, 2);
    tree.evict_device_end(FULL);
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn scoped_review_failed_backup_counts_only_unbacked_full_rows() {
    for backup_succeeds in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                is_write_back: true,
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let key = vec![10, 20];
        let mut p = params(&key, &[1, 2]);
        p.mamba_value = Some(PageValue::from_vec(vec![100]));
        let leaf = tree.insert(&p).last_device_node_id.unwrap();
        let (backup, pending) = tree.evict_device_leaf(leaf, true).unwrap();
        assert!(backup.is_some());
        assert!(pending.device_frees.is_empty());
        assert_eq!(pending.unbacked_tokens, 0);
        let result = if backup_succeeds {
            complete_backup(&mut tree, leaf);
            tree.demote(leaf).unwrap()
        } else {
            let (dropped, result) = tree.drop_subtree_no_host(leaf).unwrap();
            assert!(dropped);
            result
        };
        assert_eq!(result.device_frees[&FULL][0].as_slice(), &[1, 2]);
        assert_eq!(result.unbacked_tokens, if backup_succeeds { 0 } else { 2 });
        assert_eq!(tree.arena.resolve(leaf).is_ok(), backup_succeeds);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_empty_mamba_continuation_schedules_terminal_backup() {
    for already_backed in [false, true] {
        for chunked in [false, true] {
            for continuation in [false, true] {
                let mut tree = Core::new(
                    CacheInitParams {
                        enable_hicache: true,
                        write_through_threshold: 100,
                        mamba_cache_chunk_size: Some(1),
                        ..Default::default()
                    },
                    vec![FULL, MAMBA],
                );
                let full_key = vec![10, 20, 30, 40];
                let mut p = params(&full_key, &[1, 2, 3, 4]);
                p.mamba_value = Some(PageValue::from_vec(vec![100]));
                tree.insert(&p);
                // Splitting a real checkpoint-bearing edge leaves its prefix
                // without a Mamba checkpoint, ready for a later donation.
                let key = vec![10, 20];
                tree.match_prefix(&MatchPrefixParams {
                    key: &key,
                    namespace: Default::default(),
                });
                let anchor = tree.match_full_device_prefix(&key, Default::default()).1;
                complete_backup(&mut tree, anchor);
                let mut p = params(&key, &[]);
                p.chunked = chunked;
                p.mamba_value = Some(PageValue::from_vec(vec![101]));
                if already_backed {
                    tree.try_insert_suffix_from_node(anchor, 2, &p).unwrap();
                    tree.commit_backup(
                        anchor,
                        PageValue::from_vec(vec![]),
                        HashMap::from([(
                            MAMBA,
                            vec![PoolTransfer {
                                host_indices: Some(PageValue::from_vec(vec![201])),
                                ..Default::default()
                            }],
                        )]),
                    )
                    .unwrap();
                }
                if !continuation {
                    p.value = PageValue::from_vec(vec![7, 8]);
                    p.swa_evicted_seqlen = 0; // Exercise ordinary duplicate Full KV frees.
                }
                let result = if continuation {
                    tree.try_insert_suffix_from_node(anchor, 2, &p).unwrap()
                } else {
                    tree.try_insert(&p).unwrap()
                };
                let backups: Vec<_> = result
                    .cache_actions
                    .iter()
                    .filter_map(|action| match action {
                        CacheAction::BackupKV(backup) => Some(backup.node_ids.clone()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    backups,
                    if already_backed {
                        vec![]
                    } else {
                        vec![vec![anchor]]
                    }
                );
                let action_order: Vec<_> = result
                    .cache_actions
                    .iter()
                    .map(|action| match action {
                        CacheAction::BackupKV(_) => "backup",
                        CacheAction::FreeDeviceKV(values) => {
                            assert_eq!(
                                values
                                    .iter()
                                    .flat_map(|v| v.as_slice())
                                    .copied()
                                    .collect::<Vec<_>>(),
                                vec![7, 8]
                            );
                            "free_duplicates"
                        }
                        _ => panic!("unexpected insert action"),
                    })
                    .collect();
                let mut expected = Vec::new();
                if !continuation {
                    expected.push("free_duplicates");
                }
                if !already_backed {
                    expected.push("backup");
                }
                assert_eq!(action_order, expected);
                assert_eq!(
                    tree.arena
                        .node(tree.arena.resolve(anchor).unwrap())
                        .device_value(MAMBA)
                        .as_slice(),
                    &[101]
                );
                // The empty root remains a no-backup, no-adoption control.
                let empty = vec![];
                let mut p = params(&empty, &[]);
                p.mamba_value = Some(PageValue::from_vec(vec![102]));
                let root = tree.root_node_handle(None);
                let result = tree.try_insert_suffix_from_node(root, 0, &p).unwrap();
                assert!(result.mamba_exist);
                assert!(result.cache_actions.is_empty());
                tree.try_sanity_check(&[], &[]).unwrap();
            }
        }
    }
}

#[test]
fn scoped_review_load_back_rejects_overlapping_same_anchor_operations() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            is_write_back: true,
            ..Default::default()
        },
        vec![FULL],
    );
    let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
    complete_backup(&mut tree, leaf);
    // This zero-row plan would be valid again once the first load-back has
    // installed its device rows. It must not create a second ack for the anchor.
    let (empty, empty_aux) = tree.build_load_back_spec(leaf, None).unwrap();
    tree.demote(leaf).unwrap();
    let (full, auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
    tree.commit_load_back(leaf, PageValue::from_vec(vec![3, 4]), full, auxiliary)
        .unwrap();
    let idx = tree.arena.resolve(leaf).unwrap();
    assert_eq!(tree.arena.node(idx).load_back_pending_id, Some(leaf));
    assert!(matches!(tree.build_load_back_spec(leaf, None),
        Err(TreeCoreRuntimeError::LoadBackAlreadyPending { node_id }) if node_id == leaf));
    assert!(matches!(
        tree.commit_load_back(leaf, PageValue::from_vec(vec![]), empty, empty_aux),
        Err(TreeCoreRuntimeError::LoadBackAlreadyPending { node_id }) if node_id == leaf
    ));
    assert_eq!(tree.arena.node(idx).load_back_pending_id, Some(leaf));
    assert!(tree.drive_host_eviction(FULL, 2).host_frees.is_empty());
    tree.finish_load_back(leaf).unwrap();
    assert_eq!(tree.arena.node(idx).load_back_pending_id, None);
    let (full, auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
    tree.commit_load_back(leaf, PageValue::from_vec(vec![]), full, auxiliary)
        .unwrap();
    tree.finish_load_back(leaf).unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn scoped_review_write_through_restore_refreshes_an_active_eviction_walk() {
    for restore_during_walk in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                ..Default::default()
            },
            vec![FULL],
        );
        let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
        complete_backup(&mut tree, leaf);
        tree.demote(leaf).unwrap();
        let (full, auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
        if restore_during_walk {
            tree.evict_device_start(FULL, 2);
        }
        tree.commit_load_back(leaf, PageValue::from_vec(vec![3, 4]), full, auxiliary)
            .unwrap();
        if !restore_during_walk {
            tree.evict_device_start(FULL, 2);
        }
        assert!(tree.inspect_is_device_evictable_leaf(leaf));
        let (victim, _) = tree.evict_device_next_node(FULL, &HashMap::new());
        assert_eq!(victim, Some(leaf));
        let (_, result) = tree.evict_device_leaf(victim.unwrap(), false).unwrap();
        assert_eq!(result.device_frees[&FULL][0].as_slice(), &[3, 4]);
        assert!(
            tree.evict_device_next_node(FULL, &result.tracker)
                .0
                .is_none()
        );
        tree.evict_device_end(FULL);
        tree.finish_load_back(leaf).unwrap();
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn scoped_review_device_leaf_inspection_observes_load_back_pins() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            is_write_back: true,
            ..Default::default()
        },
        vec![FULL],
    );
    let leaf = insert(&mut tree, vec![10, 20], &[1, 2]);
    complete_backup(&mut tree, leaf);
    tree.demote(leaf).unwrap();
    let (full, auxiliary) = tree.build_load_back_spec(leaf, None).unwrap();
    tree.commit_load_back(leaf, PageValue::from_vec(vec![3, 4]), full, auxiliary)
        .unwrap();
    assert!(!tree.inspect_is_device_evictable_leaf(leaf));
    assert!(!tree.inspect_is_device_leaf(leaf).unwrap());
    tree.finish_load_back(leaf).unwrap();
    assert!(tree.inspect_is_device_evictable_leaf(leaf));
    assert!(tree.inspect_is_device_leaf(leaf).unwrap());
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn repair_duplicate_swa_backup_is_atomic_and_distinct_targets_succeed() {
    for generic in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                swa_sliding_window_size: Some(8),
                has_swa_host_pool: true,
                ..Default::default()
            },
            vec![FULL, SWA],
        );
        let a = insert(&mut tree, vec![10], &[1]);
        let b = insert(&mut tree, vec![10, 20], &[1, 2]);
        tree.attach_swa_window(
            &vec![10, 20],
            Default::default(),
            0,
            2,
            &PageValue::from_vec(vec![11, 12]),
        )
        .unwrap();
        let transfer = |nodes| {
            [(
                SWA,
                vec![PoolTransfer {
                    name: PoolName::Swa,
                    nodes_to_load: Some(nodes),
                    host_indices: Some(PageValue::from_vec(vec![201, 202])),
                    ..Default::default()
                }],
            )]
            .into_iter()
            .collect()
        };
        assert!(matches!(
            commit_staged_backup(&mut tree, b, vec![102], transfer(vec![a, a]), generic),
            Err(TreeCoreRuntimeError::InvalidBackupSpec {
                component_type: SWA,
                ..
            })
        ));
        for id in [a, b] {
            let node = tree.arena.node(tree.arena.resolve(id).unwrap());
            assert!(!node.has_host_value(FULL));
            assert!(!node.has_host_value(SWA));
        }
        tree.commit_backup(a, PageValue::from_vec(vec![101]), HashMap::new())
            .unwrap();
        commit_staged_backup(&mut tree, b, vec![102], transfer(vec![a, b]), generic).unwrap();
        let pending = tree.mark_write_through_pending(vec![a, b], b).unwrap();
        tree.finish_write_through(pending, b).unwrap();
        for (id, row) in [(a, 201), (b, 202)] {
            assert_eq!(
                tree.arena
                    .node(tree.arena.resolve(id).unwrap())
                    .host_value(SWA)
                    .as_slice(),
                &[row]
            );
        }
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn repair_mamba_backup_rejects_a_destination_filled_since_planning() {
    for generic in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let key = vec![10];
        let mut p = params(&key, &[1]);
        p.mamba_value = Some(PageValue::from_vec(vec![11]));
        let leaf = tree.insert(&p).last_device_node_id.unwrap();
        tree.commit_backup(leaf, PageValue::from_vec(vec![101]), HashMap::new())
            .unwrap();
        let (_, mut first) = tree.build_backup_spec(leaf).unwrap();
        let (_, mut second) = tree.build_backup_spec(leaf).unwrap();
        first.get_mut(&MAMBA).unwrap()[0].host_indices = Some(PageValue::from_vec(vec![201]));
        second.get_mut(&MAMBA).unwrap()[0].host_indices = Some(PageValue::from_vec(vec![202]));
        commit_staged_backup(&mut tree, leaf, vec![], first, generic).unwrap();
        assert!(matches!(
            commit_staged_backup(&mut tree, leaf, vec![], second, generic),
            Err(TreeCoreRuntimeError::InvalidBackupSpec {
                component_type: MAMBA,
                ..
            })
        ));
        assert_eq!(
            tree.arena
                .node(tree.arena.resolve(leaf).unwrap())
                .host_value(MAMBA)
                .as_slice(),
            &[201]
        );
        assert!(!tree.build_backup_spec(leaf).unwrap().1.contains_key(&MAMBA));
        let pending = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
        tree.finish_write_through(pending, leaf).unwrap();
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn repair_transfer_entrypoints_reject_invalid_dispatch_before_adoption() {
    for extra in [FULL, SWA] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                is_write_back: true,
                enable_kv_cache_events: true,
                ..Default::default()
            },
            vec![FULL],
        );
        let leaf = insert(&mut tree, vec![10], &[1]);
        tree.take_events();
        let extra_plan = if extra == FULL {
            vec![PoolTransfer {
                host_indices: Some(PageValue::from_vec(vec![999])),
                ..Default::default()
            }]
        } else {
            vec![]
        };
        assert!(matches!(
            tree.commit_backup(
                leaf,
                PageValue::from_vec(vec![101]),
                [(extra, extra_plan)].into_iter().collect()
            ),
            Err(TreeCoreRuntimeError::InvalidBackupSpec { .. })
        ));
        assert!(!tree.node_backuped(leaf).unwrap());
        assert!(tree.take_events().is_empty());
        tree.commit_backup(leaf, PageValue::from_vec(vec![101]), HashMap::new())
            .unwrap();
        tree.demote(leaf).unwrap();
        tree.take_events();
        let (full, aux) = tree.build_load_back_spec(leaf, None).unwrap();
        assert!(aux.is_empty());
        assert!(matches!(
            tree.commit_load_back(
                leaf,
                PageValue::from_vec(vec![2]),
                tree.build_load_back_spec(leaf, None).unwrap().0,
                [(extra, vec![])].into_iter().collect()
            ),
            Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
        ));
        assert!(tree.arena.node(tree.arena.resolve(leaf).unwrap()).evicted());
        assert_eq!(
            tree.arena
                .node(tree.arena.resolve(leaf).unwrap())
                .load_back_pending_id,
            None
        );
        assert!(tree.take_events().is_empty());
        let mut generic_full = tree.build_load_back_spec(leaf, None).unwrap().0;
        generic_full.device_indices = Some(PageValue::from_vec(vec![2]));
        assert!(matches!(
            tree.commit_hicache_transfers(
                leaf,
                CacheTransferPhase::LoadBack,
                [(FULL, vec![generic_full])].into_iter().collect(),
                &mut Vec::new(),
                None,
                None
            ),
            Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
        ));
        assert!(tree.arena.node(tree.arena.resolve(leaf).unwrap()).evicted());
        assert!(tree.take_events().is_empty());
        tree.commit_load_back(leaf, PageValue::from_vec(vec![2]), full, aux)
            .unwrap();
        tree.finish_load_back(leaf).unwrap();
        assert_eq!(
            tree.get_component_device_value(leaf, FULL)
                .unwrap()
                .unwrap()
                .as_slice(),
            &[2]
        );
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[rstest::rstest]
fn repair_swa_load_back_rejects_an_unrelated_anchor(#[values(false, true)] ring: bool) {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            swa_req_ring: ring,
            swa_sliding_window_size: Some(1),
            has_swa_host_pool: true,
            ..Default::default()
        },
        vec![FULL, SWA],
    );
    let mut leaves = Vec::new();
    for (k, row) in [(10, 1), (20, 2)] {
        let leaf = insert(&mut tree, vec![k], &[row]);
        tree.attach_swa_window(
            &vec![k],
            Default::default(),
            0,
            1,
            &PageValue::from_vec(vec![row + 10]),
        )
        .unwrap();
        let (_, mut aux) = tree.build_backup_spec(leaf).unwrap();
        aux.get_mut(&SWA).unwrap()[0].host_indices = Some(PageValue::from_vec(vec![row + 200]));
        tree.commit_backup(leaf, PageValue::from_vec(vec![row + 100]), aux)
            .unwrap();
        let pending = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
        tree.finish_write_through(pending, leaf).unwrap();
        let lock = tree.inc_lock_ref(leaf, ComponentSet::of(SWA)).unwrap();
        tree.evict_device_start(SWA, 1);
        let (_, step) = tree.evict_device_next_node(SWA, &HashMap::new());
        assert_eq!(step.tracker[&SWA], 1);
        tree.evict_device_end(SWA);
        tree.dec_lock_ref(leaf, &lock.to_dec_params(), false)
            .unwrap();
        leaves.push(leaf);
    }
    let (_, mut a) = tree.build_load_back_spec(leaves[0], None).unwrap();
    a.get_mut(&SWA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![301]));
    let (full, mut b) = tree.build_load_back_spec(leaves[1], None).unwrap();
    b.get_mut(&SWA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![302]));
    assert!(matches!(
        tree.commit_load_back(
            leaves[1],
            PageValue::from_vec(vec![]),
            tree.build_load_back_spec(leaves[1], None).unwrap().0,
            a
        ),
        Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
    ));
    for &id in &leaves {
        assert!(
            !tree
                .arena
                .node(tree.arena.resolve(id).unwrap())
                .has_device_value(SWA)
        );
    }
    // Every SWA tree destination must be validated, including extra descriptors.
    let (mixed_full, mut mixed) = tree.build_load_back_spec(leaves[1], None).unwrap();
    mixed.get_mut(&SWA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![302]));
    let (_, mut extra) = tree.build_load_back_spec(leaves[0], None).unwrap();
    extra.get_mut(&SWA).unwrap()[0].device_indices = Some(PageValue::from_vec(vec![301]));
    mixed
        .get_mut(&SWA)
        .unwrap()
        .append(extra.get_mut(&SWA).unwrap());
    assert!(matches!(
        tree.commit_load_back(leaves[1], PageValue::from_vec(vec![]), mixed_full, mixed),
        Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
    ));
    if ring {
        // Ring Full-only restores are valid, but cannot adopt another anchor's SWA.
        let key = vec![30];
        let mut p = params(&key, &[3]);
        p.swa_evicted_seqlen = 1;
        let c = tree.insert(&p).last_device_node_id.unwrap();
        let (c_full, c_aux) = tree.build_load_back_spec(c, None).unwrap();
        assert!(c_aux.is_empty());
        assert!(matches!(
            tree.commit_load_back(
                c,
                PageValue::from_vec(vec![]),
                tree.build_load_back_spec(c, None).unwrap().0,
                {
                    let (_, mut wrong) = tree.build_load_back_spec(leaves[0], None).unwrap();
                    wrong.get_mut(&SWA).unwrap()[0].device_indices =
                        Some(PageValue::from_vec(vec![301]));
                    wrong
                },
            ),
            Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
        ));
        assert!(
            tree.get_component_device_value(leaves[0], SWA)
                .unwrap()
                .is_none()
        );
        assert!(tree.get_component_device_value(c, SWA).unwrap().is_none());
        tree.commit_load_back(c, PageValue::from_vec(vec![]), c_full, c_aux)
            .unwrap();
    }
    tree.commit_load_back(leaves[1], PageValue::from_vec(vec![]), full, b)
        .unwrap();
    assert_eq!(
        tree.get_component_device_value(leaves[1], SWA)
            .unwrap()
            .unwrap()
            .as_slice(),
        &[302]
    );
    assert!(
        !tree
            .arena
            .node(tree.arena.resolve(leaves[0]).unwrap())
            .has_device_value(SWA)
    );
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn repair_rotation_validation_is_bidirectional_and_rechecks_host_grafts() {
    let mut tree = Core::new(CacheInitParams::default(), vec![FULL]);
    let key = vec![10];
    let mut p = params(&key, &[1]);
    p.rotation_base = Some(7);
    let a = tree.insert(&p).last_device_node_id.unwrap();
    let key = vec![10, 20];
    assert!(tree.insert(&params(&key, &[2, 3])).rotation_tail_declined);
    assert_eq!(tree.inspect_get_all_node_ids().len(), 2);
    let mut p = params(&key, &[2, 3]);
    p.rotation_base = Some(7);
    assert!(!tree.insert(&p).rotation_tail_declined);
    assert_eq!(tree.rotation_base_of(a).unwrap(), Some(7));
    assert!(
        !tree
            .insert(&params(&vec![30, 40], &[4, 5]))
            .rotation_tail_declined
    );
    for graft in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                write_through_threshold: 2,
                ..Default::default()
            },
            vec![FULL],
        );
        let key = vec![10];
        let mut p = params(&key, &[1]);
        p.rotation_base = Some(7);
        let a = tree.insert(&p).last_device_node_id.unwrap();
        let key = vec![10, 20];
        let mut p = params(&key, &[2, 3]);
        p.rotation_base = Some(7);
        assert!(tree.try_begin_insert(&p).unwrap().result.is_none());
        tree.commit_backup(a, PageValue::from_vec(vec![101]), HashMap::new())
            .unwrap();
        let pending = tree.mark_write_through_pending(vec![a], a).unwrap();
        tree.finish_write_through(pending, a).unwrap();
        let grafted = if graft {
            Some(
                tree.insert_host(
                    a,
                    None,
                    vec![20],
                    PageValue::from_vec(vec![102]),
                    vec!["a".repeat(64)],
                )
                .unwrap()
                .inserted_host_node
                .unwrap(),
            )
        } else {
            None
        };
        if let Some(b) = grafted {
            assert!(
                matches!(tree.try_resume_insert(),Err(TreeCoreRuntimeError::InsertRotationMismatch {node_id}) if node_id==b)
            );
            let b = tree.arena.node(tree.arena.resolve(b).unwrap());
            assert!(b.evicted());
            assert_eq!(b.rotation_base, None);
            tree.end_insert();
        } else {
            let result = tree.try_resume_insert().unwrap().result.unwrap();
            let b = result.last_device_node_id.unwrap();
            assert_eq!(tree.rotation_base_of(b).unwrap(), Some(7));
        }
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn repair_enabling_storage_backfills_the_parent_before_hashing_new_children() {
    let mut hashes = Vec::new();
    for storage_first in [false, true] {
        let mut tree = Core::new(CacheInitParams::default(), vec![FULL]);
        if storage_first {
            tree.set_enable_storage(true);
        }
        let a = insert(&mut tree, vec![10], &[1]);
        tree.set_enable_storage(true);
        let b = insert(&mut tree, vec![10, 20], &[1, 2]);
        assert_eq!(tree.backfill_missing_hash_values(), 0);
        hashes.push((
            tree.get_hash_values(a).unwrap(),
            tree.get_prefix_hash_values(b).unwrap(),
            tree.get_hash_values(b).unwrap(),
        ));
    }
    assert_eq!(hashes[0], hashes[1]);
    assert_eq!(hashes[0].0, hashes[0].1);
}

#[test]
fn repair_auxiliary_walk_reselects_after_an_initially_empty_cursor() {
    for auxiliary in [MAMBA, SWA] {
        for unlock in [false, true] {
            let mut tree = Core::new(
                CacheInitParams {
                    mamba_cache_chunk_size: Some(1),
                    swa_sliding_window_size: Some(1),
                    ..Default::default()
                },
                vec![FULL, auxiliary],
            );
            let key = vec![10];
            let mut p = params(&key, &[1]);
            p.mamba_value = (auxiliary == MAMBA).then(|| PageValue::from_vec(vec![11]));
            let leaf = tree.insert(&p).last_device_node_id.unwrap();
            if auxiliary == SWA {
                tree.attach_swa_window(
                    &key,
                    Default::default(),
                    0,
                    1,
                    &PageValue::from_vec(vec![11]),
                )
                .unwrap();
            }
            let lock = tree.inc_lock_ref(leaf, ComponentSet::EMPTY).unwrap();
            tree.evict_device_start(auxiliary, 1);
            if unlock {
                tree.dec_lock_ref(leaf, &lock.to_dec_params(), false)
                    .unwrap();
            }
            let (victim, step) = tree.evict_device_next_node(auxiliary, &HashMap::new());
            assert_eq!(victim, unlock.then_some(leaf));
            assert!(step.device_frees.is_empty());
            tree.evict_device_end(auxiliary);
            if !unlock {
                tree.dec_lock_ref(leaf, &lock.to_dec_params(), false)
                    .unwrap();
            }
            tree.try_sanity_check(&[], &[]).unwrap();
        }
    }
}

#[test]
fn repair_mamba_exact_insert_refreshes_the_saved_victim() {
    for touch in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, MAMBA],
        );
        let a_key = vec![10];
        let mut p = params(&a_key, &[1]);
        p.mamba_value = Some(PageValue::from_vec(vec![11]));
        let a = tree.insert(&p).last_device_node_id.unwrap();
        let key = vec![20];
        let mut b = params(&key, &[2]);
        b.mamba_value = Some(PageValue::from_vec(vec![12]));
        let b = tree.insert(&b).last_device_node_id.unwrap();
        tree.evict_device_start(MAMBA, 1);
        if touch {
            tree.insert(&p);
        }
        assert_eq!(
            tree.evict_device_next_node(MAMBA, &HashMap::new()).0,
            Some(if touch { b } else { a })
        );
        tree.evict_device_end(MAMBA);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn repair_full_walk_observes_the_last_auxiliary_unlock_in_either_order() {
    for components in [vec![FULL, SWA], vec![SWA, FULL]] {
        let mut tree = Core::new(
            CacheInitParams {
                swa_sliding_window_size: Some(1),
                ..Default::default()
            },
            components,
        );
        let leaf = insert(&mut tree, vec![10], &[1]);
        tree.attach_swa_window(
            &vec![10],
            Default::default(),
            0,
            1,
            &PageValue::from_vec(vec![11]),
        )
        .unwrap();
        let lock = tree.inc_lock_ref(leaf, ComponentSet::EMPTY).unwrap();
        tree.evict_device_start(FULL, 1);
        tree.dec_lock_ref(leaf, &lock.to_dec_params(), false)
            .unwrap();
        assert_eq!(
            tree.evict_device_next_node(FULL, &HashMap::new()).0,
            Some(leaf)
        );
        tree.evict_device_end(FULL);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn repair_failed_backup_drop_defers_a_new_request_lock() {
    for lock_after in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                is_write_back: true,
                ..Default::default()
            },
            vec![FULL],
        );
        let leaf = insert(&mut tree, vec![10], &[1]);
        let (backup, _) = tree.evict_device_leaf(leaf, true).unwrap();
        assert!(backup.is_some());
        let lock = lock_after.then(|| tree.inc_lock_ref(leaf, ComponentSet::EMPTY).unwrap());
        let (dropped, step) = tree.drop_subtree_no_host(leaf).unwrap();
        assert_eq!(dropped, !lock_after);
        if let Some(lock) = lock {
            assert!(step.device_frees.is_empty());
            assert_eq!(tree.full_protected_size(), 1);
            tree.dec_lock_ref(leaf, &lock.to_dec_params(), false)
                .unwrap();
            assert!(tree.drop_subtree_no_host(leaf).unwrap().0);
        }
        assert_eq!(tree.total_size(), (0, 0));
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn repair_swa_cascade_checks_mamba_locks_before_immediate_and_deferred_eviction() {
    for deferred in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: deferred,
                is_write_back: deferred,
                has_swa_host_pool: deferred,
                swa_sliding_window_size: Some(1),
                mamba_cache_chunk_size: Some(1),
                ..Default::default()
            },
            vec![FULL, SWA, MAMBA],
        );
        let key = vec![10];
        let mut p = params(&key, &[1]);
        p.mamba_value = Some(PageValue::from_vec(vec![21]));
        let a = tree.insert(&p).last_device_node_id.unwrap();
        tree.attach_swa_window(
            &key,
            Default::default(),
            0,
            1,
            &PageValue::from_vec(vec![11]),
        )
        .unwrap();
        let key = vec![10, 20];
        let mut p = params(&key, &[1, 2]);
        p.mamba_value = Some(PageValue::from_vec(vec![22]));
        tree.insert(&p);
        let lock = if deferred {
            None
        } else {
            Some(tree.inc_lock_ref(a, ComponentSet::of(SWA)).unwrap())
        };
        tree.evict_device_start(SWA, 1);
        let (_, step) = tree.evict_device_next_node(SWA, &HashMap::new());
        let lock = if deferred {
            assert_eq!(step.swa_backup_node_id, Some(a));
            let lock = tree.inc_lock_ref(a, ComponentSet::of(SWA)).unwrap();
            let result = tree.finish_swa_state_eviction(a);
            assert!(result.device_frees.is_empty());
            lock
        } else {
            assert!(step.device_frees.is_empty());
            lock.unwrap()
        };
        assert_eq!(
            tree.get_component_device_value(a, SWA)
                .unwrap()
                .unwrap()
                .as_slice(),
            &[11]
        );
        assert_eq!(
            tree.get_component_device_value(a, MAMBA)
                .unwrap()
                .unwrap()
                .as_slice(),
            &[21]
        );
        tree.dec_lock_ref(a, &lock.to_dec_params(), false).unwrap();
        // A new walk selects the same formerly blocked state. A failed backup
        // still permits its normal best-effort drop once the receipt is gone.
        tree.evict_device_end(SWA);
        tree.evict_device_start(SWA, 1);
        let (_, step) = tree.evict_device_next_node(SWA, &HashMap::new());
        let step = if deferred {
            assert_eq!(step.swa_backup_node_id, Some(a));
            tree.finish_swa_state_eviction(a)
        } else {
            step
        };
        assert_eq!(step.device_frees[&SWA][0].as_slice(), &[1]);
        assert_eq!(step.device_frees[&MAMBA][0].as_slice(), &[21]);
        tree.evict_device_end(SWA);
        tree.try_sanity_check(&[], &[]).unwrap();
    }
    // A blocked older internal state must not hide an eligible later leaf.
    let mut tree = Core::new(
        CacheInitParams {
            swa_sliding_window_size: Some(1),
            mamba_cache_chunk_size: Some(1),
            ..Default::default()
        },
        vec![FULL, SWA, MAMBA],
    );
    let key = vec![10];
    let mut p = params(&key, &[1]);
    p.mamba_value = Some(PageValue::from_vec(vec![21]));
    let a = tree.insert(&p).last_device_node_id.unwrap();
    tree.attach_swa_window(
        &key,
        Default::default(),
        0,
        1,
        &PageValue::from_vec(vec![11]),
    )
    .unwrap();
    let key = vec![10, 20];
    let mut p = params(&key, &[1, 2]);
    p.mamba_value = Some(PageValue::from_vec(vec![22]));
    tree.insert(&p);
    let key = vec![30];
    let mut p = params(&key, &[3]);
    p.mamba_value = Some(PageValue::from_vec(vec![23]));
    let b = tree.insert(&p).last_device_node_id.unwrap();
    tree.attach_swa_window(
        &key,
        Default::default(),
        0,
        1,
        &PageValue::from_vec(vec![13]),
    )
    .unwrap();
    let receipt = tree.inc_lock_ref(a, ComponentSet::of(SWA)).unwrap();
    tree.evict_device_start(SWA, 1);
    let (victim, step) = tree.evict_device_next_node(SWA, &HashMap::new());
    assert_eq!(victim, Some(b));
    assert!(step.device_frees.is_empty());
    assert!(tree.get_component_device_value(a, SWA).unwrap().is_some());
    tree.evict_device_end(SWA);
    tree.dec_lock_ref(a, &receipt.to_dec_params(), false)
        .unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn repair_suspended_swa_tombstone_insert_pins_consumed_full_until_resume_or_end() {
    for cancel in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                write_through_threshold: 2,
                swa_sliding_window_size: Some(1),
                ..Default::default()
            },
            vec![FULL, SWA],
        );
        let key = vec![10, 20];
        let mut p = params(&key, &[1, 2]);
        p.swa_evicted_seqlen = 2;
        let leaf = tree.insert(&p).last_device_node_id.unwrap();
        let key = vec![10, 20, 30];
        let mut p = params(&key, &[5, 6, 7]);
        p.swa_evicted_seqlen = 2;
        let step = tree.try_begin_insert(&p).unwrap();
        assert!(step.result.is_none());
        let frees: Vec<_> = step
            .actions
            .iter()
            .flat_map(|action| match action {
                CacheAction::FreeDeviceKVFullOnly(chunks) => chunks
                    .iter()
                    .flat_map(|c| c.as_slice().iter().copied())
                    .collect::<Vec<_>>(),
                _ => vec![],
            })
            .collect();
        assert_eq!(frees, vec![5, 6]);
        complete_backup(&mut tree, leaf);
        assert!(tree.demote(leaf).unwrap().device_frees.is_empty());
        assert_eq!(tree.full_protected_size(), 2);
        if cancel {
            tree.end_insert();
            assert_eq!(tree.full_protected_size(), 0);
            assert_eq!(
                tree.demote(leaf).unwrap().device_frees[&FULL][0].as_slice(),
                &[1, 2]
            );
        } else {
            let mut completed = false;
            for _ in 0..3 {
                let step = tree.try_resume_insert().unwrap();
                assert!(step.actions.iter().all(|a| !matches!(
                    a,
                    CacheAction::FreeDeviceKV(_) | CacheAction::FreeDeviceKVFullOnly(_)
                )));
                if step.result.is_some() {
                    completed = true;
                    break;
                }
                assert!(!step.actions.is_empty());
                for action in step.actions {
                    if let CacheAction::SwaRebuild {
                        node_id,
                        source_value,
                    } = action
                    {
                        tree.set_component_device_value(node_id, SWA, source_value)
                            .unwrap();
                    } else {
                        panic!("unexpected insert barrier");
                    }
                }
            }
            assert!(completed);
            assert_eq!(tree.full_kv_prefix_len(&key, Default::default()), 3);
            assert_eq!(tree.full_protected_size(), 0);
        }
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn repair_swa_reinsertion_applies_rebuild_before_incremental_backup() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            enable_kv_cache_events: true,
            has_swa_host_pool: true,
            swa_sliding_window_size: Some(1),
            ..Default::default()
        },
        vec![FULL, SWA],
    );
    let key = vec![10];
    let mut initial = params(&key, &[1]);
    initial.swa_evicted_seqlen = 1;
    let leaf = tree.insert(&initial).last_device_node_id.unwrap();
    tree.commit_backup(leaf, PageValue::from_vec(vec![101]), HashMap::new())
        .unwrap();
    let pending = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
    tree.finish_write_through(pending, leaf).unwrap();
    tree.take_events();
    let before = (
        tree.total_size(),
        tree.inspect_get_node_access_counter(leaf).unwrap(),
    );
    let mut p = params(&key, &[2]);
    p.swa_evicted_seqlen = 0;
    assert!(matches!(
        tree.try_insert(&p),
        Err(TreeCoreRuntimeError::InsertRequiresController)
    ));
    assert_eq!(
        before,
        (
            tree.total_size(),
            tree.inspect_get_node_access_counter(leaf).unwrap()
        )
    );
    assert!(tree.take_events().is_empty());

    let mut step = tree.try_begin_insert(&p).unwrap();
    let mut observed = Vec::new();
    loop {
        for action in step.actions {
            match action {
                CacheAction::SwaRebuild {
                    node_id,
                    source_value,
                } => {
                    assert_eq!(node_id, leaf);
                    assert_eq!(source_value.as_slice(), &[2]);
                    tree.set_component_device_value(node_id, SWA, PageValue::from_vec(vec![12]))
                        .unwrap();
                    observed.push("rebuild");
                }
                CacheAction::BackupKV(backup) => {
                    assert_eq!(backup.node_ids, vec![leaf]);
                    let (full, mut aux) = tree.build_backup_spec(leaf).unwrap();
                    assert!(full.is_empty());
                    assert_eq!(
                        aux[&SWA][0].device_indices.as_ref().unwrap().as_slice(),
                        &[12]
                    );
                    aux.get_mut(&SWA).unwrap()[0].host_indices =
                        Some(PageValue::from_vec(vec![201]));
                    tree.commit_backup(leaf, PageValue::from_vec(vec![]), aux)
                        .unwrap();
                    let pending = tree.mark_write_through_pending(vec![leaf], leaf).unwrap();
                    tree.finish_write_through(pending, leaf).unwrap();
                    observed.push("backup");
                }
                CacheAction::FreeDeviceKVFullOnly(chunks) => assert_eq!(chunks[0].as_slice(), &[1]),
                _ => panic!("unexpected controller action"),
            }
        }
        if step.result.is_some() {
            break;
        }
        step = tree.try_resume_insert().unwrap();
    }
    assert_eq!(observed, ["rebuild", "backup"]);
    let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
    assert_eq!(node.host_value(FULL).as_slice(), &[101]);
    assert_eq!(node.host_value(SWA).as_slice(), &[201]);
    tree.try_sanity_check(&[], &[]).unwrap();
}

#[test]
fn repair_swa_backup_is_bound_to_the_anchors_ordered_window() {
    for generic in [false, true] {
        let mut tree = Core::new(
            CacheInitParams {
                enable_hicache: true,
                enable_kv_cache_events: true,
                has_swa_host_pool: true,
                swa_sliding_window_size: Some(1),
                ..Default::default()
            },
            vec![FULL, SWA],
        );
        let mut leaves = Vec::new();
        for (key, row) in [(vec![10], 1), (vec![20], 2)] {
            let mut p = params(&key, &[row]);
            p.swa_evicted_seqlen = 1;
            let leaf = tree.insert(&p).last_device_node_id.unwrap();
            tree.attach_swa_window(
                &key,
                Default::default(),
                0,
                1,
                &PageValue::from_vec(vec![row + 10]),
            )
            .unwrap();
            leaves.push(leaf);
        }
        let (_, mut own) = tree.build_backup_spec(leaves[0]).unwrap();
        let (_, mut unrelated) = tree.build_backup_spec(leaves[1]).unwrap();
        own.get_mut(&SWA).unwrap()[0].host_indices = Some(PageValue::from_vec(vec![201]));
        unrelated.get_mut(&SWA).unwrap()[0].host_indices = Some(PageValue::from_vec(vec![202]));
        tree.take_events();
        let before = tree.total_size();
        assert!(matches!(
            commit_staged_backup(&mut tree, leaves[0], vec![101], unrelated, generic),
            Err(TreeCoreRuntimeError::InvalidBackupSpec {
                component_type: SWA,
                ..
            })
        ));
        for &leaf in &leaves {
            let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
            assert!(!node.has_host_value(FULL));
            assert!(!node.has_host_value(SWA));
        }
        assert_eq!(before, tree.total_size());
        assert!(tree.take_events().is_empty());
        commit_staged_backup(&mut tree, leaves[0], vec![101], own, generic).unwrap();
        let pending = tree
            .mark_write_through_pending(vec![leaves[0]], leaves[0])
            .unwrap();
        tree.finish_write_through(pending, leaves[0]).unwrap();
        let a = tree.arena.node(tree.arena.resolve(leaves[0]).unwrap());
        assert_eq!(a.host_value(FULL).as_slice(), &[101]);
        assert_eq!(a.host_value(SWA).as_slice(), &[201]);
        let b = tree.arena.node(tree.arena.resolve(leaves[1]).unwrap());
        assert!(!b.has_host_value(FULL));
        assert!(!b.has_host_value(SWA));
        tree.try_sanity_check(&[], &[]).unwrap();
    }
}

#[test]
fn repair_empty_swa_load_back_batch_is_rejected_before_mutation() {
    let mut tree = Core::new(
        CacheInitParams {
            enable_hicache: true,
            enable_kv_cache_events: true,
            has_swa_host_pool: true,
            swa_sliding_window_size: Some(1),
            ..Default::default()
        },
        vec![FULL, SWA],
    );
    let key = vec![10];
    let mut p = params(&key, &[1]);
    p.swa_evicted_seqlen = 1;
    let leaf = tree.insert(&p).last_device_node_id.unwrap();
    tree.attach_swa_window(
        &key,
        Default::default(),
        0,
        1,
        &PageValue::from_vec(vec![11]),
    )
    .unwrap();
    tree.take_events();
    let (full, mut aux) = tree.build_load_back_spec(leaf, None).unwrap();
    assert!(aux.is_empty());
    aux.insert(SWA, vec![]);
    let before = tree.total_size();
    assert!(matches!(
        tree.commit_load_back(leaf, PageValue::from_vec(vec![]), full, aux),
        Err(TreeCoreRuntimeError::InvalidLoadBackSpec { .. })
    ));
    assert_eq!(tree.total_size(), before);
    assert!(tree.take_events().is_empty());
    let node = tree.arena.node(tree.arena.resolve(leaf).unwrap());
    assert_eq!(node.device_value(FULL).as_slice(), &[1]);
    assert_eq!(node.device_value(SWA).as_slice(), &[11]);
    assert_eq!(node.load_back_pending_id, None);
    let (full, aux) = tree.build_load_back_spec(leaf, None).unwrap();
    tree.commit_load_back(leaf, PageValue::from_vec(vec![]), full, aux)
        .unwrap();
    tree.try_sanity_check(&[], &[]).unwrap();
}
