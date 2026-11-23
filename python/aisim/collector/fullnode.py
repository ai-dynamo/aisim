# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-node collector orchestration.

Collectives such as DeepEP own every visible GPU and cannot run in the normal
per-GPU worker pool. This module keeps their case selection, checkpointing, and
failure mapping out of the top-level backend entrypoint.
"""

import json
import os
import random
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from collector.case_generator import without_models
from collector.helper import collector_retained_path, create_test_case_id
from collector.op_plans import OP_PLANS
from collector.skips import filter_cases, models
from collector.version_resolver import _check_compat

SGLANG_FULLNODE_OPS = {
    "deepep_ll": "DEEPEP_LL_SHAPE_INDEX",
    "deepep_normal": "DEEPEP_NORMAL_SHAPE_INDEX",
    "kimi_k31_dsa_context_module": None,
    "kimi_k31_dsa_generation_module": None,
}


def _resume_errors(module_name, resume_tracker, resume_options, logger, known_errors=()) -> list[dict]:
    if not (resume_options and resume_options.get("resume")):
        return []
    unresolved = len(
        resume_tracker.unresolved_failed_ids() - {e["task_id"] for e in known_errors if e.get("task_params")}
    )
    if unresolved <= 0:
        return []
    logger.warning(
        f"{module_name}: checkpoint holds {unresolved} unresolved failed tasks "
        "(skipped on resume; rerun with --resume-retry-failed to retry)"
    )
    return [
        {
            "module": module_name,
            "task_id": "resume_unresolved",
            "error_type": "UnresolvedFailures",
            "error_message": f"checkpoint holds {unresolved} unresolved failed tasks",
            "classification": "unresolved_from_checkpoint",
            "timestamp": datetime.now().isoformat(),
        }
    ]


def select_cases(op: str, cases: list, limit: int | None) -> list:
    """Apply the explicit job-array shape selector or the common case limit."""
    shape_index_env = SGLANG_FULLNODE_OPS[op]
    shape_index = os.environ.get(shape_index_env) if shape_index_env else None
    if shape_index is not None and shape_index != "":
        idx = int(shape_index)
        if idx < 0 or idx >= len(cases):
            raise RuntimeError(f"{shape_index_env}={idx} out of range (0..{len(cases) - 1})")
        return [cases[idx]]
    if limit is not None:
        return cases[:limit]
    return cases


def collect_sglang_fullnode_op(
    collection: dict,
    *,
    runtime_version: str,
    limit: int | None,
    shuffle: bool,
    shuffle_seed: int,
    backend: str,
    resume_options: dict | None,
    model_path: str | None,
    case_plan,
    sm_version: int | None,
    case_filters: list[str] | None,
    get_test_cases_for_model,
    resume_checkpoint_cls,
    logger,
    replay=None,
) -> list[dict]:
    """Run one SGLang full-node op with normal V3 checkpoint evidence."""
    op = collection["type"]
    full_name = f"{collection['name']}.{op}"
    if case_plan is not None and not case_plan.has_op(op):
        logger.info(f"Skipping {full_name} — not in collector v2 case plan")
        return []

    # Full-node routes obey the same registry maturity contract as pool ops.
    # In particular, moving a native TP module here must not enable an
    # unqualified backend/SM merely because it owns its GPU group exclusively.
    unverified_here = collection.get("unverified") or (
        collection.get("unverified_sms") and (sm_version is None or sm_version in collection["unverified_sms"])
    )
    if unverified_here:
        scope = (
            "this backend"
            if collection.get("unverified")
            else ("unresolved SM" if sm_version is None else f"SM{sm_version}")
        )
        logger.warning(f"Skipping {full_name} — registry marks it unverified on {scope}")
        return [
            {
                "module": full_name,
                "error_type": "UnverifiedCollector",
                "error_message": f"registry marks this op unverified on {scope}; collection skipped",
                "classification": "unverified_skipped",
            }
        ]

    if op.startswith("kimi_k31_dsa_") and os.environ.get("COLLECTOR_MEASURE_POWER", "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise ValueError("native Kimi K3.1 does not support --measure_power; use latency-only collection")

    module_name = collection["module"]
    get_module = __import__(module_name, fromlist=[collection["get_func"]])
    run_module = __import__(module_name, fromlist=[collection["run_func"]])

    declared = getattr(get_module, "__compat__", None)
    if declared:
        try:
            if not _check_compat(declared, runtime_version):
                raise RuntimeError(
                    f"module {module_name} declares __compat__={declared!r}, runtime is v{runtime_version}"
                )
        except ValueError as e:
            raise RuntimeError(f"invalid __compat__ {declared!r}: {e}") from e

    get_func = getattr(get_module, collection["get_func"])
    run_func = getattr(run_module, collection["run_func"])
    run_func_name = collection["run_func"]

    with without_models(models(backend, op, sm_version)):
        cases = get_test_cases_for_model(get_func, model_path)
    cases, _skipped = filter_cases(cases, backend=backend, op=op, sm_version=sm_version)
    if case_filters:
        before_count = len(cases)
        cases = [case for case in cases if any(fragment in str(case) for fragment in case_filters)]
        logger.info(f"{op}: --case-filter kept {len(cases)}/{before_count} cases")
    if replay is not None:
        # same selection, in the same order, as the pool ops (collect.py get_func_with_limit)
        before_count = len(cases)
        cases = replay.select(op, cases, random.Random(shuffle_seed))
        logger.info(
            f"{op}: replay selected {len(replay.matched[op])} requested + "
            f"{replay.controls[op]} control of {before_count} cases"
        )
    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(cases)
    cases = select_cases(op, cases, limit)
    if not cases:
        return [
            {
                "module": full_name,
                "error_type": "ModuleCollectionFailure",
                "error_message": f"No test cases resolved for full-node op {op}",
                "classification": "unexpected",
                "timestamp": datetime.now().isoformat(),
            }
        ]

    raw_task_infos = [
        {
            "id": create_test_case_id(case, run_func_name, full_name),
            "params": case,
            "index": i,
        }
        for i, case in enumerate(cases)
    ]

    checkpoint_dir = (
        resume_options.get("checkpoint_dir", ".collector_checkpoint") if resume_options else ".collector_checkpoint"
    )
    resume_tracker = resume_checkpoint_cls(
        backend=backend,
        module_name=full_name,
        run_func_name=run_func_name,
        checkpoint_dir=checkpoint_dir,
        framework_version=runtime_version,
        sm_version=sm_version,
    )
    if resume_options and resume_options.get("resume"):
        resume_tracker.load_existing()
        task_infos = resume_tracker.filter_done(raw_task_infos, retry_failed=resume_options.get("retry_failed", False))
    else:
        task_infos = raw_task_infos

    error_path = Path(checkpoint_dir) / backend / f"errors_{full_name}.json"
    errors: list[dict] = []
    if resume_options and resume_options.get("resume") and error_path.exists():
        errors = json.loads(error_path.read_text())

    historical = [e for e in errors if e.get("error_type") == "FullNodeCollectionFailure"]
    errors = [e for e in errors if e.get("error_type") != "FullNodeCollectionFailure"]

    def persist_errors():
        if not errors and not error_path.exists():
            return
        error_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = error_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(historical + errors, indent=2))
        os.replace(temporary, error_path)
        log_dir = os.environ.get("COLLECTOR_LOG_DIR")
        if log_dir:
            report = Path(log_dir) / error_path.name
            try:
                report.parent.mkdir(parents=True, exist_ok=True)
                temporary = report.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(historical + errors, indent=2))
                os.replace(temporary, report)
            except OSError as exc:
                logger.warning(f"Failed to mirror collector diagnostics to {report}: {exc}")

    # A native worker can outlive a killed launcher. Give each attempt a
    # private staging file. Only the parent publishes its bytes, after the
    # checkpoint is durable. A receipt bridges the two atomic replacements.
    native = op.startswith("kimi_k31_dsa_")
    output = Path(collection["perf_filename"])
    receipt = output.with_name(f".{output.name}.pending")

    def finish_publication():
        if not receipt.exists():
            return
        transaction = json.loads(receipt.read_text())
        name = transaction["stage"]
        if Path(name).name != name or not name.startswith(f".{output.name}.stage-"):
            raise ValueError(f"invalid native staging filename in {receipt}")
        stage = output.with_name(name)
        if resume_tracker.has_passed(transaction["task_id"]) and stage.exists():
            if collector_retained_path(output).exists():
                # Reuse the retained inode through the same publication protocol
                # as the other collector writers after a closed event.
                from collector.collect import _atomic_write_bytes

                _atomic_write_bytes(output, stage.read_bytes())
                stage.unlink()
            else:
                os.replace(stage, output)
        else:
            stage.unlink(missing_ok=True)
        receipt.unlink()

    if native:
        if Path(f"{output}.lock").exists():
            raise RuntimeError(f"Native output already has a writer lock: {output}.lock")
        if (
            resume_options
            and resume_options.get("resume")
            and not receipt.exists()
            and output.exists()
            and resume_tracker.has_unfinished_attempts()
        ):
            raise RuntimeError(
                f"Unreconciled native staging rows in {output}; preserve and restart in a fresh directory"
            )
        finish_publication()

    if not task_infos:
        logger.info(f"{full_name}: no full-node tasks to run")
        return errors + _resume_errors(full_name, resume_tracker, resume_options, logger, errors)

    logger.info(f"Running {full_name} full-node collection (bypassing per-GPU worker pool)")
    task_by_id = {task_info["id"]: task_info["params"] for task_info in task_infos}
    completed = set()
    submitted = set()
    try:
        # DeepEP amortizes group setup across its batch. K3.1 already launches
        # a fresh native process per case: checkpoint each returned case before
        # launching the next so a job time limit cannot erase earlier progress.
        batches = ([task_info] for task_info in task_infos) if op.startswith("kimi_k31_dsa_") else (task_infos,)
        for batch in batches:
            batch_ids = {str(task_info["params"]): task_info["id"] for task_info in batch}
            filename = collection["perf_filename"]
            if native:
                with tempfile.NamedTemporaryFile(
                    dir=output.parent, prefix=f".{output.name}.stage-", delete=False
                ) as tmp:
                    filename = Path(tmp.name)
                if output.exists():
                    shutil.copyfile(output, filename)
                transaction = {"task_id": batch[0]["id"], "stage": filename.name}
                pending = receipt.with_suffix(".pending.tmp")
                pending.write_text(json.dumps(transaction))
                os.replace(pending, receipt)
            resume_tracker.mark_attempted_many(batch_ids.values())
            submitted.update(batch_ids.values())
            result = run_func(
                perf_filename=str(filename) if native else filename,
                limit=None,
                cases=[task_info["params"] for task_info in batch],
            )
            if not isinstance(result, dict) or "failed" not in result:
                raise RuntimeError(f"{full_name} runner must return a result containing its failed case list")
            # Native runners can retain per-case causes without changing the case
            # list/checkpoint protocol used by older full-node collectors.
            failed_cases = result["failed"]
            failure_details = result.get("failure_details", {})
            unknown_failed = [case for case in failed_cases if str(case) not in batch_ids]
            if unknown_failed:
                raise RuntimeError(
                    f"{full_name} runner reported failures outside the requested case plan: {unknown_failed}"
                )
            failed_ids = {batch_ids[str(case)] for case in failed_cases}
            errors[:] = [error for error in errors if error.get("task_id") not in batch_ids.values()]
            for task_id in sorted(failed_ids):
                case = task_by_id[task_id]
                detail = failure_details.get(str(case), {})
                # Use the plan's consumer dimensions, excluding swept history
                # coordinates, so repeated failures expose unavailable keys.
                spec = OP_PLANS.get(op)
                group = (
                    json.dumps({field: getattr(case, field) for field in spec.group_fields}, sort_keys=True)
                    if spec is not None and all(hasattr(case, field) for field in spec.group_fields)
                    else None
                )
                message = detail.get("error_message", "full-node runner reported failed case")
                errors.append(
                    {
                        "module": full_name,
                        "task_id": task_id,
                        "task_params": str(task_by_id[task_id]),
                        "group": group,
                        "error_type": detail.get("error_type", "FullNodeCaseFailure"),
                        "error_message": message,
                        "classification": "kernel_limit"
                        if message.lstrip().startswith("FIXME(kernel-limit)")
                        else "unexpected",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            persist_errors()
            # mark_failed/mark_passed may themselves flush: publish the exact
            # error records first, before any cumulative checkpoint can skip them.
            for task_id in batch_ids.values():
                completed.add(task_id)
                if task_id in failed_ids:
                    resume_tracker.mark_failed(task_id)
                else:
                    resume_tracker.mark_passed(task_id)
            resume_tracker.flush(force=True)
            if native:
                finish_publication()
    except Exception as e:
        logger.exception(f"{full_name} full-node collection failed")
        for task_id in submitted - completed:
            resume_tracker.mark_failed(task_id)
        errors.append(
            {
                "module": full_name,
                "task_id": "fullnode_collection",
                "error_type": "FullNodeCollectionFailure",
                "error_message": str(e),
                "classification": "unexpected",
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
        )
    finally:
        persist_errors()
        resume_tracker.flush(force=True)

    if native:
        finish_publication()
    errors.extend(_resume_errors(full_name, resume_tracker, resume_options, logger, errors))
    return errors
