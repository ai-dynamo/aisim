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
        swa_evicted_seqlen: 0,
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

#[test]
fn continuation_rejects_a_wrong_key_before_side_effects() {
    let mut tree = Core::new(CacheInitParams::default(), vec![FULL]);
    let anchor = insert(&mut tree, vec![10, 20], &[1, 2]);
    let tick = tree.inspect_get_node_access_counter(anchor).unwrap();
    let key = vec![30, 40, 50];
    let p = params(&key, &[3]);
    assert!(tree.try_insert_suffix_from_node(anchor, 2, &p).is_err());
    assert_eq!(tree.inspect_get_node_access_counter(anchor).unwrap(), tick);
    assert_eq!(tree.evictable_size(), 2);
    assert!(tree.inspect_get_child_node_ids(anchor).unwrap().is_empty());
    let valid_key = vec![10, 20, 50];
    let result = tree
        .try_insert_suffix_from_node(anchor, 2, &params(&valid_key, &[3]))
        .unwrap();
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
