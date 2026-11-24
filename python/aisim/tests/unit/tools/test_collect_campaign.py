# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tools/perf_database/collect_campaign.py — the invariants a hand-rolled launcher drifted on (sm120)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "perf_database" / "collect_campaign.py"


def _load():
    spec = importlib.util.spec_from_file_location("collect_campaign", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["collect_campaign"] = mod
    spec.loader.exec_module(mod)
    return mod


def _publish_native_table(directory, phase):
    """Synthetic committed publication; no GPU measurement evidence."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from collector.provenance import render_collection_meta

    table = f"dsa_{phase}_module_perf"
    pq.write_table(pa.table({"latency": [1.0]}), directory / f"{table}.parquet")
    (directory / "collection_meta.yaml").write_bytes(
        render_collection_meta(
            {"framework": "sglang", "version": "0.0.0.dev1+g41208373c.d20261104"},
            {
                table: {
                    "collector_ref": "0" * 40,
                    "collector_hash": "sha256:" + "1" * 64,
                    "case_plan_hash": "sha256:" + "2" * 64,
                    "collected_at": "2026-11-24",
                    "rows": 1,
                    "status": "complete",
                }
            },
        )
    )


def _plan(tmp_path, shards):
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({"backend": "vllm", "sm": 120, "shards": shards}))
    return p


def test_collector_argv_always_resumes_and_never_keeps_csv(tmp_path):
    c = _load()
    plan = c.load_plan(_plan(tmp_path, [{"op": "gemm", "name": "bf16", "case_filter": "['bfloat16', "}, {"op": "kda"}]))
    for shard in plan.shards:
        argv = c.collector_argv(plan, shard)
        assert "--resume" in argv and "--keep-csv" not in argv
        assert argv[argv.index("--sm") + 1] == "120"
        assert argv[argv.index("--ops") + 1] == shard.op
    assert "--case-filter" in c.collector_argv(plan, plan.shards[0])
    assert "--case-filter" not in c.collector_argv(plan, plan.shards[1])
    assert plan.shards[1].name == "all"


def test_one_namespace_per_shard_and_readonly_checkout(tmp_path):
    c = _load()
    plan = c.load_plan(_plan(tmp_path, [{"op": "gemm", "name": "bf16"}, {"op": "gemm", "name": "fp8"}]))
    dirs = {c.shard_dir(tmp_path, s) for s in plan.shards}
    assert len(dirs) == 2
    ref = "0123456789abcdef0123456789abcdef01234567"
    argv = c.docker_argv(
        plan,
        plan.shards[0],
        image="img",
        gpu=3,
        checkout=Path("/ck"),
        shard_dir=dirs.pop(),
        container_name="x",
        collector_ref=ref,
    )
    assert '"device=3"' in argv and "/ck:/ais:ro" in argv and argv[argv.index("-w") + 1] == "/out"
    # the mount has no .git and the images no git: provenance gets the commit from the launcher
    assert f"AISIM_COLLECTOR_REF={ref}" in argv


def test_targeted_model_and_long_context_reach_collector(tmp_path):
    c = _load()
    plan = c.load_plan(
        _plan(
            tmp_path,
            [{"op": "dsa_generation_module", "model_path": "nvidia/GLM-5.2-NVFP4", "include_long_context": True}],
        )
    )
    argv = c.collector_argv(plan, plan.shards[0])
    assert argv[argv.index("--model-path") + 1] == "nvidia/GLM-5.2-NVFP4"
    assert "--model-cases-full" not in argv
    assert "--include-long-context" in argv
    default_plan = c.load_plan(_plan(tmp_path, [{"op": "gemm"}]))
    default_argv = c.collector_argv(default_plan, default_plan.shards[0])
    assert "--model-cases-full" in default_argv
    assert "--include-long-context" not in default_argv


def test_long_context_requires_boolean(tmp_path):
    c = _load()
    with pytest.raises(ValueError, match="YAML boolean"):
        c.load_plan(_plan(tmp_path, [{"op": "gemm", "include_long_context": "false"}]))


def test_collector_ref_is_the_clean_checkout_commit(tmp_path):
    import subprocess

    c = _load()

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "collect.py").write_text("v1\n")
    git("add", "collect.py")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "v1")
    ref = c.collector_ref(tmp_path)
    assert len(ref) == 40 and int(ref, 16) >= 0
    (tmp_path / "collect.py").write_text("v2\n")  # uncommitted: the rows would name the wrong code
    with pytest.raises(SystemExit, match="uncommitted"):
        c.collector_ref(tmp_path)


def test_duplicate_shard_keys_are_rejected(tmp_path):
    c = _load()
    with pytest.raises(ValueError):
        c.load_plan(_plan(tmp_path, [{"op": "gemm", "name": "a"}, {"op": "gemm", "name": "a"}]))


def test_image_is_the_manifest_pin_plus_finalize_deps():
    c = _load()
    base = c.pinned_image("vllm", "gemm")
    assert base.startswith("vllm/vllm-openai:")  # framework_manifest default pin, never a hand-typed tag
    df = c.dockerfile(base)
    assert df.startswith(f"FROM {base}\n") and "pyarrow" in df and "pandas" in df
    assert c.collect_image_tag(base).startswith("aisim-collect:")


def test_digest_pins_have_distinct_build_and_campaign_tags(monkeypatch, tmp_path):
    from types import SimpleNamespace

    c = _load()
    bases = [f"registry.example/{'long-repository-' * 6}:same@sha256:{digit * 64}" for digit in ("a", "b")]
    builds = []
    monkeypatch.setattr(c.subprocess, "run", lambda argv, **kwargs: builds.append((argv, kwargs["input"])))
    tags = [c.build_image(base) for base in bases]
    assert tags[0] != tags[1]
    assert c.collect_image_tag(bases[0]) == tags[0]
    assert all(len(tag.split(":", 1)[1]) <= 128 for tag in tags)
    for (argv, recipe), tag, base in zip(builds, tags, bases, strict=True):
        assert argv[argv.index("-t") + 1] == tag
        assert recipe.startswith(f"FROM {base}\n")

    shards = tuple(c.Shard("gemm", str(index), model_path=base) for index, base in enumerate(bases))
    plan = c.Plan("sglang", 103, shards)
    monkeypatch.setattr(c, "collector_ref", lambda _: "0" * 40)
    monkeypatch.setattr(c, "pinned_image", lambda _backend, _op, model: model)
    inspected, commands = [], []
    monkeypatch.setattr(c, "image_digest", lambda image: inspected.append(image) or "sha256:present")

    def launch(argv):
        commands.append(argv)
        return SimpleNamespace(returncode=0, poll=lambda: 0, wait=lambda: 0)

    monkeypatch.setattr(c.subprocess, "Popen", launch)
    assert c.run_campaign(plan, root=tmp_path, gpus=[0], checkout=tmp_path, image_override=None, mode="run") == 0
    assert set(inspected) == set(tags)
    assert [argv[argv.index("--image") + 1] for argv in commands] == tags


def test_shard_state_reads_the_namespace(tmp_path):
    c = _load()
    plan = c.load_plan(_plan(tmp_path, [{"op": "gemm", "name": "bf16"}]))
    s = plan.shards[0]
    assert c.shard_state(tmp_path, s) == "pending"
    d = c.shard_dir(tmp_path, s)
    (d / ".ckpt").mkdir(parents=True)
    assert c.shard_state(tmp_path, s) == "started"
    (d / "DONE").touch()
    assert c.shard_state(tmp_path, s) == "done-no-parquet"  # the --keep-csv failure mode, picked up by `finalize`
    (d / "gemm_perf.parquet").write_bytes(b"")
    assert c.shard_state(tmp_path, s) == "finalized"


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_k31_campaign_uses_model_scoped_image_and_leaves_model_free_resolution_alone(phase):
    from collector.framework_manifest import require_collector_runtime, resolve_op_runtime
    from collector.sglang.collect_kimi_k31_dsa_v1 import MODEL, RUNTIME_VERSION

    c = _load()
    op = f"kimi_k31_dsa_{phase}_module"
    pinned = require_collector_runtime("sglang", RUNTIME_VERSION, requested_ops={op}, model_path=MODEL)
    assert c.pinned_image("sglang", op, MODEL) == pinned.image()
    assert "@sha256:" in c.pinned_image("sglang", op, MODEL)
    assert c.pinned_image("sglang", op) == resolve_op_runtime("sglang", op).image()
    assert c.pinned_image("sglang", op) != pinned.image()


def test_campaign_images_are_keyed_by_shard_model(monkeypatch, tmp_path):
    from types import SimpleNamespace

    c = _load()
    shards = (c.Shard("gemm", "one", model_path="one"), c.Shard("gemm", "two", model_path="two"))
    plan = c.Plan("sglang", 103, shards)
    monkeypatch.setattr(c, "collector_ref", lambda _checkout: "0" * 40)
    monkeypatch.setattr(c, "pinned_image", lambda _backend, _op, model: model)
    monkeypatch.setattr(c, "collect_image_tag", lambda base: "image-" + base)
    monkeypatch.setattr(c, "image_digest", lambda _image: "sha256:test")
    commands = []

    def launch(argv):
        commands.append(argv)
        return SimpleNamespace(returncode=0, poll=lambda: 0, wait=lambda: 0)

    monkeypatch.setattr(c.subprocess, "Popen", launch)
    assert (
        c.run_campaign(
            plan,
            root=tmp_path / "out",
            gpus=[0],
            checkout=tmp_path,
            image_override=None,
            mode="run",
        )
        == 0
    )
    (tmp_path / "plan.yaml").write_text("changed after admission")
    captured = []
    monkeypatch.setattr(c, "run_shard", lambda plan, shard, **kwargs: captured.append((plan.sm, shard)) or 0)
    for command in commands:
        monkeypatch.setattr(sys, "argv", command[1:])
        assert c.main() == 0
    assert captured == [(103, shard) for shard in shards]
    assert [cmd[cmd.index("--image") + 1] for cmd in commands] == ["image-one", "image-two"]


def test_build_image_uses_model_scoped_runtime(monkeypatch):
    from collector.sglang.collect_kimi_k31_dsa_v1 import MODEL

    c = _load()
    op = "kimi_k31_dsa_context_module"
    seen = []
    monkeypatch.setattr(
        c, "build_image", lambda base, *, model_path, **_kwargs: seen.append((base, model_path)) or "built"
    )
    monkeypatch.setattr(
        sys, "argv", ["collect_campaign", "build-image", "--backend", "sglang", "--op", op, "--model-path", MODEL]
    )
    assert c.main() == 0
    assert seen == [(c.pinned_image("sglang", op, MODEL), MODEL)]


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("case_filter", [None, ", 48, 'fp8',", ", 24, 'fp8',", ", 12, 'fp8',"])
@pytest.mark.parametrize("mode", ["run", "finalize"])
def test_k31_campaign_refuses_multi_gpu_cases_before_launch(monkeypatch, tmp_path, phase, case_filter, mode):
    c = _load()
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "other/model")
    shard = c.Shard(
        f"kimi_k31_dsa_{phase}_module", "native", case_filter=case_filter, model_path="moonshotai/Kimi-K3.1-Pro"
    )
    plan = c.Plan("sglang", 103, (shard,))
    if mode == "finalize":
        directory = c.shard_dir(tmp_path / "out", shard)
        directory.mkdir(parents=True)
        (directory / "DONE").touch()
    monkeypatch.setattr(c, "collector_ref", lambda _: pytest.fail("admission must precede launch preparation"))
    with pytest.raises(SystemExit, match=r"one GPU.*collector/collect.py.*full-node"):
        c.run_campaign(
            plan,
            root=tmp_path / "out",
            gpus=[0, 1, 2, 3],
            checkout=tmp_path,
            image_override=None,
            mode=mode,
        )
    assert c.os.environ["COLLECTOR_MODEL_PATH"] == "other/model"


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("mode,transaction", [("run", False), ("finalize", False), ("finalize", True)])
def test_k31_campaign_preserves_tp1_shards(monkeypatch, tmp_path, phase, mode, transaction):
    from types import SimpleNamespace

    c = _load()
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "other/model")
    shard = c.Shard(
        f"kimi_k31_dsa_{phase}_module", "tp1", case_filter=", 96, 'fp8',", model_path="moonshotai/Kimi-K3.1-Pro"
    )
    plan = c.Plan("sglang", 103, (shard,))
    monkeypatch.setattr(c, "collector_ref", lambda _: "0" * 40)
    monkeypatch.setattr(c, "image_digest", lambda _: "sha256:test")
    if mode == "finalize":
        checkpoint = c.shard_dir(tmp_path / "out", shard) / ".ckpt" / plan.backend / f"{plan.backend}.{shard.op}.json"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        if transaction:
            from collector.helper import SIDECAR_TRANSACTION_FILENAME

            directory = c.shard_dir(tmp_path / "out", shard)
            (directory / c._native_phase_table(shard.op)).touch()
            (directory / SIDECAR_TRANSACTION_FILENAME).touch()
            assert c.shard_state(tmp_path / "out", shard, plan.backend) == "pending-publication"

    commands = []

    def launch(argv):
        commands.append(argv)
        return SimpleNamespace(returncode=0, poll=lambda: 0, wait=lambda: 0)

    monkeypatch.setattr(c.subprocess, "Popen", launch)
    assert (
        c.run_campaign(
            plan,
            root=tmp_path / "out",
            gpus=[3],
            checkout=tmp_path,
            image_override="test-image",
            mode=mode,
        )
        == 0
    )
    assert len(commands) == 1
    assert c.os.environ["COLLECTOR_MODEL_PATH"] == "other/model"
    assert commands[0][commands[0].index("--gpu") + 1] == "3"
    submitted = c.Shard(**json.loads(commands[0][commands[0].index("--shard") + 1]))
    assert submitted == shard
    assert ", 96, 'fp8'," in c.collector_argv(plan, submitted)


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("mode", ["run", "finalize"])
@pytest.mark.parametrize("model_path", [None, "moonshotai/Kimi-K3.1-Pr"])
def test_k31_noncanonical_shard_is_rejected_before_launch(monkeypatch, tmp_path, phase, mode, model_path):
    c = _load()
    shard = c.Shard(f"kimi_k31_dsa_{phase}_module", "tp1", case_filter=", 96, 'fp8',", model_path=model_path)
    plan = c.Plan("sglang", 103, (shard,))
    monkeypatch.setattr(c, "collector_ref", lambda _: pytest.fail("admission must precede launch preparation"))
    if mode == "finalize":
        directory = c.shard_dir(tmp_path / "out", shard)
        directory.mkdir(parents=True)
        (directory / "DONE").touch()
    with pytest.raises(SystemExit, match=r"model_path: moonshotai/Kimi-K3.1-Pro.*pinned K3.1 runtime"):
        c.run_campaign(
            plan,
            root=tmp_path / "out",
            gpus=[0],
            checkout=tmp_path,
            image_override=None,
            mode=mode,
        )


def test_model_pin_does_not_bypass_unknown_op_validation(monkeypatch):
    from collector.sglang.collect_kimi_k31_dsa_v1 import MODEL

    c = _load()
    monkeypatch.setattr(c, "build_image", lambda *_a, **_k: pytest.fail("built image for unknown op"))
    monkeypatch.setattr(
        sys, "argv", ["collect_campaign", "build-image", "--backend", "sglang", "--op", "typo", "--model-path", MODEL]
    )
    with pytest.raises(KeyError, match="registry has no op"):
        c.main()


@pytest.mark.parametrize(
    "backend,op", [("sglang", "gemm"), ("vllm", "gemm"), ("sglang", "kimi_k31_dsa_context_module")]
)
def test_unpinned_models_keep_family_or_default_image(backend, op):
    from collector.framework_manifest import resolve_op_runtime

    c = _load()
    assert c.pinned_image(backend, op, "unlisted/model") == resolve_op_runtime(backend, op).image()


@pytest.mark.parametrize("model", [None, "other/model"])
def test_native_build_image_requires_canonical_model(model, monkeypatch):
    c = _load()
    argv = ["collector", "build-image", "--backend", "sglang", "--op", "kimi_k31_dsa_generation_module"]
    monkeypatch.setattr(sys, "argv", argv + (["--model-path", model] if model else []))
    with pytest.raises(SystemExit, match="require model_path"):
        c.main()


def test_native_shard_maturity_is_checked_before_scheduling():
    c = _load()
    shard = c.Shard("kimi_k31_dsa_generation_module", "one", model_path="moonshotai/Kimi-K3.1-Pro")
    with pytest.raises(SystemExit, match="unverified on SM100"):
        c._validate_shard_allocation(c.Plan("sglang", 100, (shard,)), shard)


def test_finalize_admits_existing_checkpoint_with_worker_launch_disabled(tmp_path):
    c = _load()
    shard = c.Shard(
        "kimi_k31_dsa_generation_module", "tp2", case_filter=", 48, 'fp8',", model_path="moonshotai/Kimi-K3.1-Pro"
    )
    plan = c.Plan("sglang", 103, (shard,))
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.touch()
    c._validate_shard_allocation(plan, shard, finalize_checkpoint=checkpoint)
    args = c.docker_argv(
        plan,
        shard,
        image="test",
        gpu=0,
        checkout=tmp_path,
        shard_dir=tmp_path,
        container_name="finalize",
        collector_ref="0" * 40,
        mode="finalize",
    )
    assert "COLLECTOR_K31_FINALIZE_ONLY=1" in args


@pytest.mark.parametrize("mode", ["run", "finalize"])
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_campaign_recovers_tp_checkpoint_and_validates_all_selected_cases(
    monkeypatch, tmp_path, mode, interrupted, complete
):
    import logging
    from types import SimpleNamespace

    from collector import fullnode
    from collector.collect import ResumeCheckpoint, _atomic_write_previous_path
    from collector.helper import atomic_write_reservation_path, create_test_case_id
    from collector.op_plans import ModuleCase
    from collector.sglang import collect_kimi_k31_dsa_v1 as native

    c = _load()
    shard = c.Shard("kimi_k31_dsa_context_module", "tp2", case_filter=", 48, 'fp8',", model_path=native.MODEL)
    plan = c.Plan("sglang", 103, (shard,))
    out = c.shard_dir(tmp_path, shard)
    checkpoints = out / ".ckpt"
    tracker = ResumeCheckpoint(
        "sglang", "sglang." + shard.op, "run_kimi_k31_dsa_fullnode", str(checkpoints), native.RUNTIME_VERSION, 103
    )
    fullnode.bind_native_checkpoint(tracker._path, out, create=True)
    cases = [ModuleCase(length, 1, 48, "fp8", "bfloat16", "bfloat16", native.MODEL, 0) for length in (1024, 4096)]
    for case in cases:
        task_id = create_test_case_id(case, "run_kimi_k31_dsa_fullnode", "sglang." + shard.op)
        tracker.mark_attempted(task_id)
        if complete or case == cases[0]:
            tracker.mark_passed(task_id)
    tracker.flush(force=True)
    original = tracker._path.read_bytes()
    if interrupted:
        tracker._path.rename(_atomic_write_previous_path(tracker._path))
        atomic_write_reservation_path(tracker._path).write_bytes(b"incomplete replacement")
    monkeypatch.setattr(c, "collector_ref", lambda _: "0" * 40)
    monkeypatch.setattr(c, "image_digest", lambda _: "sha256:present")
    commands = []

    def launch(argv):
        commands.append(argv)
        return SimpleNamespace(returncode=0, poll=lambda: 0, wait=lambda: 0)

    monkeypatch.setattr(c.subprocess, "Popen", launch)
    assert c.run_campaign(plan, root=tmp_path, gpus=[0], checkout=tmp_path, image_override="test", mode=mode) == 0
    assert len(commands) == 1
    assert commands[0][commands[0].index("--mode") + 1] == "finalize"
    assert tracker._path.exists() != interrupted  # admission is read-only
    monkeypatch.setenv("COLLECTOR_K31_FINALIZE_ONLY", "1")
    monkeypatch.setattr(native, "run_kimi_k31_dsa_fullnode", lambda **_: pytest.fail("must never launch TP work"))

    def collect():
        return fullnode.collect_sglang_fullnode_op(
            {
                "name": "sglang",
                "type": shard.op,
                "module": native.__name__,
                "get_func": "get_kimi_k31_dsa_context_test_cases",
                "run_func": "run_kimi_k31_dsa_fullnode",
                "perf_filename": str(out / "dsa_context_module_perf.txt"),
            },
            runtime_version=native.RUNTIME_VERSION,
            limit=None,
            shuffle=False,
            shuffle_seed=42,
            backend="sglang",
            resume_options={"checkpoint_dir": str(checkpoints), "resume": True},
            model_path=native.MODEL,
            case_plan=None,
            sm_version=103,
            case_filters=None,
            get_test_cases_for_model=lambda *_: cases,
            resume_checkpoint_cls=ResumeCheckpoint,
            logger=logging.getLogger(__name__),
        )

    if complete:
        assert collect() == []
    else:
        with pytest.raises(RuntimeError, match="passed checkpoint for every selected case"):
            collect()
    assert tracker._path.read_bytes() == original


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize(
    ("receipt_phase", "staged_event"),
    [(None, None), ("context", None), ("generation", None), (None, "open"), (None, "closed")],
)
@pytest.mark.parametrize("mode", ["run", "finalize"])
def test_campaign_recovers_pending_native_case_despite_older_parquet(
    monkeypatch, tmp_path, phase, receipt_phase, staged_event, mode
):
    from types import SimpleNamespace

    from collector.collect import ResumeCheckpoint
    from collector.helper import create_test_case_id
    from collector.op_plans import ModuleCase
    from collector.sglang import collect_kimi_k31_dsa_v1 as native

    c = _load()
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", native.MODEL, 0)
    shard = c.Shard(f"kimi_k31_dsa_{phase}_module", "tp1", case_filter=str(case), model_path=native.MODEL)
    plan = c.Plan("sglang", 103, (shard,))
    out = c.shard_dir(tmp_path, shard)
    tracker = ResumeCheckpoint(
        plan.backend,
        f"{plan.backend}.{shard.op}",
        "run_kimi_k31_dsa_fullnode",
        str(out / ".ckpt"),
        native.RUNTIME_VERSION,
        plan.sm,
    )
    task_id = create_test_case_id(case, "run_kimi_k31_dsa_fullnode", f"{plan.backend}.{shard.op}")
    if staged_event != "closed":
        tracker.mark_attempted(task_id)
    tracker.mark_passed(task_id)
    tracker.flush(force=True)
    checkpoint_bytes = tracker._path.read_bytes()
    _publish_native_table(out, phase)
    (out / "DONE").touch()
    if staged_event is not None:
        (out / f"dsa_{phase}_module_perf.txt").write_text("recovered row\n")
    if receipt_phase is not None:
        filename = f"dsa_{receipt_phase}_module_perf.txt"
        stage = out / f".{filename}.stage-retry"
        stage.write_text("recovered row\n")
        (out / f".{filename}.pending").write_text(json.dumps({"task_id": task_id, "stage": stage.name}))
    pending = receipt_phase == phase or staged_event == "open"
    assert c.shard_state(tmp_path, shard, plan.backend) == ("pending-publication" if pending else "finalized")

    monkeypatch.setattr(c, "collector_ref", lambda _: "0" * 40)
    monkeypatch.setattr(c, "image_digest", lambda _: "sha256:present")
    commands = []

    def launch(argv):
        commands.append(argv)
        return SimpleNamespace(returncode=0, poll=lambda: 0, wait=lambda: 0)

    monkeypatch.setattr(c.subprocess, "Popen", launch)
    assert c.run_campaign(plan, root=tmp_path, gpus=[0], checkout=tmp_path, image_override="test", mode=mode) == 0
    assert len(commands) == int(pending)
    if pending:
        assert commands[0][commands[0].index("--mode") + 1] == mode
    assert tracker._path.read_bytes() == checkpoint_bytes


@pytest.mark.parametrize("publication", ["valid", "missing-phase", "missing-meta", "other-phase-meta", "wrong-rows"])
def test_native_shard_done_requires_its_phase_table(tmp_path, monkeypatch, publication):
    from types import SimpleNamespace

    c = _load()
    shard = c.Shard("kimi_k31_dsa_generation_module", "tp1", model_path="moonshotai/Kimi-K3.1-Pro")
    plan = c.Plan("sglang", 103, (shard,))
    out = c.shard_dir(tmp_path, shard)
    out.mkdir(parents=True)
    (out / ".ckpt").mkdir()
    (out / "DONE").touch()
    _publish_native_table(out, "generation")
    if publication == "missing-phase":
        (out / "dsa_generation_module_perf.parquet").rename(out / "other.parquet")
    elif publication == "missing-meta":
        (out / "collection_meta.yaml").unlink()
    elif publication in ("other-phase-meta", "wrong-rows"):
        sidecar = out / "collection_meta.yaml"
        doc = yaml.safe_load(sidecar.read_text())
        if publication == "other-phase-meta":
            doc["tables"]["dsa_context_module_perf"] = doc["tables"].pop("dsa_generation_module_perf")
        else:
            doc["tables"]["dsa_generation_module_perf"]["rows"] = 2
        sidecar.write_text(yaml.safe_dump(doc))
    expected = {
        "valid": "finalized",
        "missing-phase": "done-no-parquet",
    }.get(publication, "incomplete-provenance")
    assert c.shard_state(tmp_path, shard, plan.backend) == expected
    monkeypatch.setattr(c, "image_digest", lambda _: "sha256:test")
    monkeypatch.setattr(c.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    rc = c.run_shard(plan, shard, root=tmp_path, gpu=0, checkout=tmp_path, image="test", collector_ref="0" * 40)
    assert (rc == 0) == (publication == "valid")
    assert (out / "DONE").exists() == (publication == "valid")
    assert (out / ".ckpt").is_dir()


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("version_kind", ["stock", "without-local", "different-local"])
@pytest.mark.parametrize("runtime_location", ["document", "collection"])
def test_native_publication_requires_exact_model_runtime(tmp_path, monkeypatch, phase, version_kind, runtime_location):
    from types import SimpleNamespace

    from collector.framework_manifest import resolve_op_runtime
    from collector.provenance import validate_collection_meta_for_update
    from collector.sglang.collect_kimi_k31_dsa_v1 import MODEL, RUNTIME_VERSION

    c = _load()
    shard = c.Shard(f"kimi_k31_dsa_{phase}_module", "tp1", case_filter=", 96, 'fp8',", model_path=MODEL)
    plan = c.Plan("sglang", 103, (shard,))
    out = c.shard_dir(tmp_path, shard)
    out.mkdir(parents=True)
    _publish_native_table(out, phase)
    (out / "DONE").touch()
    version = {
        "stock": resolve_op_runtime(plan.backend, shard.op).version,
        "without-local": RUNTIME_VERSION.split("+", 1)[0],
        "different-local": RUNTIME_VERSION.split("+", 1)[0] + "+other",
    }[version_kind]
    assert version != RUNTIME_VERSION
    runtime = {"framework": plan.backend, "version": version}
    sidecar = out / "collection_meta.yaml"
    document = yaml.safe_load(sidecar.read_text())
    table = f"dsa_{phase}_module_perf"
    if runtime_location == "document":
        document["runtime"] = runtime
    else:
        document["schema_version"] = 2
        event = {**document["tables"][table], "runtime": runtime}
        document["tables"][table] = {"rows": 1, "status": "complete", "collections": [event]}
    validate_collection_meta_for_update(document, tables_to_update={table})
    sidecar.write_text(yaml.safe_dump(document))

    assert c.shard_state(tmp_path, shard, plan.backend) == "incomplete-provenance"
    commands = []
    monkeypatch.setattr(c, "collector_ref", lambda _: "0" * 40)
    monkeypatch.setattr(c, "image_digest", lambda _: "sha256:test")

    def launch(argv):
        commands.append(argv)
        return SimpleNamespace(returncode=1, poll=lambda: 1, wait=lambda: 1)

    monkeypatch.setattr(c.subprocess, "Popen", launch)
    assert c.run_campaign(plan, root=tmp_path, gpus=[0], checkout=tmp_path, image_override="test", mode="run") == 1
    assert len(commands) == 1  # A stale DONE must not skip the mismatched publication.


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("checkpoint_state", ["absent", "closed"])
@pytest.mark.parametrize("publication", ["valid", "missing-meta", "other-phase-meta", "wrong-rows", "wrong-runtime"])
def test_finalize_reports_unrecoverable_native_provenance(
    tmp_path, monkeypatch, capsys, phase, checkpoint_state, publication
):
    from collector.collect import ResumeCheckpoint
    from collector.helper import create_test_case_id
    from collector.op_plans import ModuleCase
    from collector.sglang.collect_kimi_k31_dsa_v1 import MODEL, RUNTIME_VERSION

    c = _load()
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    shard = c.Shard(f"kimi_k31_dsa_{phase}_module", "tp1", case_filter=str(case), model_path=MODEL)
    plan = c.Plan("sglang", 103, (shard,))
    out = c.shard_dir(tmp_path, shard)
    out.mkdir(parents=True)
    if checkpoint_state == "closed":
        tracker = ResumeCheckpoint(
            plan.backend,
            f"{plan.backend}.{shard.op}",
            "run_kimi_k31_dsa_fullnode",
            str(out / ".ckpt"),
            RUNTIME_VERSION,
            103,
        )
        tracker.mark_passed(create_test_case_id(case, "run_kimi_k31_dsa_fullnode", f"{plan.backend}.{shard.op}"))
        tracker.flush(force=True)
        # A closed event's cumulative passes cannot attest staged rows.
        (out / f"dsa_{phase}_module_perf.txt").write_text("closed event row\n")
    _publish_native_table(out, phase)
    (out / "DONE").touch()
    sidecar = out / "collection_meta.yaml"
    if publication == "missing-meta":
        sidecar.unlink()
    elif publication != "valid":
        document = yaml.safe_load(sidecar.read_text())
        table = f"dsa_{phase}_module_perf"
        if publication == "other-phase-meta":
            other_phase = "generation" if phase == "context" else "context"
            document["tables"][f"dsa_{other_phase}_module_perf"] = document["tables"].pop(table)
        elif publication == "wrong-rows":
            document["tables"][table]["rows"] = 2
        else:
            document["runtime"]["version"] = "0.0.0.dev1+other"
        sidecar.write_text(yaml.safe_dump(document))
    before = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}
    monkeypatch.setattr(c, "collector_ref", lambda _: "0" * 40)
    monkeypatch.setattr(c, "image_digest", lambda _: pytest.fail("finalize must not inspect images"))
    monkeypatch.setattr(c.subprocess, "Popen", lambda _: pytest.fail("finalize must not launch a worker"))

    def finalize():
        return c.run_campaign(plan, root=tmp_path, gpus=[0], checkout=tmp_path, image_override="test", mode="finalize")

    if publication == "valid":
        assert finalize() == 0
    else:
        with pytest.raises(SystemExit, match=rf"{shard.key}: incomplete-provenance.*no recoverable publication"):
            finalize()
        c.status(plan, tmp_path)
        assert "incomplete-provenance" in capsys.readouterr().out
    assert {path: path.read_bytes() for path in out.rglob("*") if path.is_file()} == before


def test_finalize_rejects_empty_native_case_selection(tmp_path):
    c = _load()
    shard = c.Shard(
        "kimi_k31_dsa_generation_module", "empty", model_path="moonshotai/Kimi-K3.1-Pro", case_filter="missing-case"
    )
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.touch()
    with pytest.raises(SystemExit, match="no qualified native cases"):
        c._validate_shard_allocation(c.Plan("sglang", 103, (shard,)), shard, finalize_checkpoint=checkpoint)
