#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a perf-data collection campaign as declared shards, one container per shard.

Why this exists (sm120 campaign): every box so far re-invented the
same launcher by hand — an image with the finalize deps added, ``--case-filter``
shards, one cwd per shard so staging CSVs never collide, a GPU pool, and the
finalize step. Each re-invention drifted somewhere (the sm120 run passed
``--keep-csv`` and silently produced no parquet/provenance for 13 shards; the
official image lacks pyarrow so the smoke run died at finalize). This tool fixes
the invariants once:

* the image is the framework_manifest pin for the (framework, op) being run —
  never a hand-typed tag — plus the finalize deps baked on top (``build-image``);
* one output/checkpoint namespace per shard (``<root>/<op>/<shard>/``), exactly
  as the collector-upgrade playbook §5 asks;
* ``collect.py`` is always invoked with ``--resume`` and never ``--keep-csv``,
  so a finished shard always ends with parquet + collection_meta.yaml;
* every container launch is appended to ``<root>/campaign.jsonl`` (image digest,
  argv, exit code, parquet outputs) — the work log the playbook §4 wants.

Shard plan (YAML)::

    backend: vllm
    sm: 120
    shards:
      - {op: gemm, name: bf16, case_filter: "['bfloat16', "}
      - {op: gemm, name: fp8,  case_filter: "['fp8', "}
      - {op: kda,  name: all}

Usage::

    python3 tools/perf_database/collect_campaign.py build-image --backend vllm --op gemm
    python3 tools/perf_database/collect_campaign.py run   --plan shards.yaml --root /ws/collect --gpus 0,1,2,3
    python3 tools/perf_database/collect_campaign.py status --plan shards.yaml --root /ws/collect
    python3 tools/perf_database/collect_campaign.py finalize --plan shards.yaml --root /ws/collect --gpus 0

``run`` is re-entrant: shards with a ``DONE`` marker are skipped, unfinished
shards resume from their checkpoint. ``finalize`` re-enters finished shards that
have no parquet (e.g. runs made by hand with ``--keep-csv``) — every task is
already checkpointed, so collect.py only finalizes. Merging the shards of one op
into one table is the delivery step (playbook §8) and is out of scope here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]  # python/aisim
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

FINALIZE_DEPS = ("pyarrow", "pandas")  # collector/helper.py finalize_perf_files imports
CONTAINER_CHECKOUT = "/ais"
CONTAINER_OUT = "/out"


@dataclass(frozen=True)
class Shard:
    op: str
    name: str
    case_filter: str | None = None
    model_path: str | None = None
    include_long_context: bool = False

    @property
    def key(self) -> str:
        return f"{self.op}/{self.name}"


@dataclass(frozen=True)
class Plan:
    backend: str
    sm: int
    shards: tuple[Shard, ...]


def load_plan(path: Path) -> Plan:
    doc = yaml.safe_load(path.read_text())
    shards = []
    seen: set[str] = set()
    for raw in doc["shards"]:
        include_long_context = raw.get("include_long_context", False)
        if not isinstance(include_long_context, bool):
            raise ValueError("include_long_context must be a YAML boolean")
        s = Shard(
            op=str(raw["op"]),
            name=str(raw.get("name", "all")),
            case_filter=raw.get("case_filter") or None,
            model_path=raw.get("model_path") or None,
            include_long_context=include_long_context,
        )
        if s.key in seen:
            raise ValueError(f"duplicate shard {s.key}")
        seen.add(s.key)
        shards.append(s)
    return Plan(backend=str(doc["backend"]), sm=int(doc["sm"]), shards=tuple(shards))


def _native_k31(backend: str, op: str) -> bool:
    return backend == "sglang" and op in ("kimi_k31_dsa_context_module", "kimi_k31_dsa_generation_module")


def _require_native_model(op, model_path):
    from collector.sglang.collect_kimi_k31_dsa_v1 import MODEL

    if model_path != MODEL:
        raise SystemExit(f"{op}: native K3.1 shards require model_path: {MODEL} to select the pinned K3.1 runtime")


def pinned_image(backend: str, op: str, model_path: str | None = None) -> str:
    """The manifest-pinned image for (backend, op) — family pins included."""
    from collector.framework_manifest import _model_pinned_runtime, load_manifest, resolve_op_runtime

    runtime = resolve_op_runtime(backend, op)
    if model_path is not None:
        model_runtime = _model_pinned_runtime(load_manifest(), backend.lower(), model_path)
        if model_runtime is not None:
            return model_runtime.image()
    return runtime.image()


def collect_image_tag(base_image: str) -> str:
    """Deterministic local tag for <base image> + finalize deps."""
    name = base_image.split("@", 1)[0].replace("/", "-").replace(":", "-")
    return f"aisim-collect:{name}"


def dockerfile(base_image: str, model_path: str | None = None) -> str:
    recipe = (
        f"FROM {base_image}\n"
        "# collector finalize deps the framework image does not ship (collector/helper.py finalize_perf_files)\n"
        f"RUN pip install --no-cache-dir {' '.join(FINALIZE_DEPS)}\n"
    )

    if model_path == "moonshotai/Kimi-K3.1-Pro":
        from collector.sglang.collect_kimi_k31_dsa_v1 import MODEL, MODEL_REVISION, TOKENIZER_FILES

        files = ("config.json", *TOKENIZER_FILES)
        fetch = (
            "from huggingface_hub import hf_hub_download; "
            f"[hf_hub_download({MODEL!r}, name, revision={MODEL_REVISION!r}) for name in {files!r}]"
        )
        recipe += f"RUN python3 -c {shlex.quote(fetch)}\n"
    return recipe


def build_image(
    base_image: str, *, model_path: str | None = None, network: str = "host", extra_build_args: list[str] | None = None
) -> str:
    tag = collect_image_tag(base_image)
    cmd = ["docker", "build", "--network", network, "-t", tag, "-f", "-", "."] + (extra_build_args or [])
    subprocess.run(cmd, input=dockerfile(base_image, model_path), text=True, check=True)
    return tag


def image_digest(image: str) -> str | None:
    try:
        out = subprocess.run(
            # a locally built collect image has NO RepoDigests — `index .RepoDigests 0` fails and the launcher
            # refused its own build-image output (B200 2026-10-03); fall back to the image Id
            ["docker", "inspect", "--format", "{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}|{{.Id}}", image],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out
    except Exception:
        return None


def collector_argv(plan: Plan, shard: Shard) -> list[str]:
    """The exact collect.py argv for one shard. ``--resume`` always, ``--keep-csv`` never."""
    argv = [
        f"{CONTAINER_CHECKOUT}/collector/collect.py",
        "--backend",
        plan.backend,
        "--sm",
        str(plan.sm),
        "--ops",
        shard.op,
        "--checkpoint-dir",
        f"{CONTAINER_OUT}/.ckpt",
        "--resume",
    ]
    if shard.model_path:
        argv += ["--model-path", shard.model_path]
    else:
        argv += ["--model-cases-full"]
    if shard.include_long_context:
        argv += ["--include-long-context"]
    if shard.case_filter:
        argv += ["--case-filter", shard.case_filter]
    assert "--keep-csv" not in argv
    return argv


def collector_ref(checkout: Path) -> str:
    """The commit the mounted checkout runs, for every shard's provenance.

    The container mounts only python/aisim (no .git) and the framework images carry no
    git, so collect.py's own lookup would record "unknown"; it reads AISIM_COLLECTOR_REF
    instead. A dirty tree is refused: its rows would name a commit that did not produce them.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(checkout), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        ref = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".")
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"{checkout}: cannot resolve the collector commit ({error})") from error
    if dirty:
        raise SystemExit(f"{checkout}: uncommitted changes; commit them so the rows record the code that ran")
    if not re.fullmatch(r"[0-9a-f]{40,64}", ref):
        raise SystemExit(f"{checkout}: `git rev-parse HEAD` gave {ref!r}, not a full commit hash")
    return ref


def docker_argv(
    plan: Plan,
    shard: Shard,
    *,
    image: str,
    gpu: int,
    checkout: Path,
    shard_dir: Path,
    container_name: str,
    collector_ref: str,
    mode: str = "run",
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--gpus",
        f'"device={gpu}"',
        "--shm-size",
        "16g",
        "-e",
        "CUDA_MPS_PIPE_DIRECTORY=/nonexistent-no-mps",
        *(
            ["-e", "COLLECTOR_K31_FINALIZE_ONLY=1"]
            if mode == "finalize" and _native_k31(plan.backend, shard.op)
            else []
        ),
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        f"PYTHONPATH={CONTAINER_CHECKOUT}",
        "-e",
        f"AISIM_COLLECTOR_REF={collector_ref}",
        "-e",
        f"TRITON_CACHE_DIR={CONTAINER_OUT}/.cache/triton",
        "-e",
        f"DG_JIT_CACHE_DIR={CONTAINER_OUT}/.cache/deep_gemm",
        "-e",
        f"FLASHINFER_WORKSPACE_BASE={CONTAINER_OUT}/.cache",
        "-v",
        f"{checkout}:{CONTAINER_CHECKOUT}:ro",
        "-v",
        f"{shard_dir}:{CONTAINER_OUT}",
        "-w",
        CONTAINER_OUT,
        "--entrypoint",
        "python3",
        image,
        *collector_argv(plan, shard),
    ]


def shard_dir(root: Path, shard: Shard) -> Path:
    return root / shard.op / shard.name


def _native_phase_table(op: str) -> str:
    phase = "context" if op == "kimi_k31_dsa_context_module" else "generation"
    return f"dsa_{phase}_module_perf.parquet"


def shard_state(root: Path, shard: Shard, backend: str | None = None) -> str:
    d = shard_dir(root, shard)
    if not d.exists():
        return "pending"
    native = backend is not None and _native_k31(backend, shard.op)
    published = (d / _native_phase_table(shard.op)).is_file() if native else any(d.glob("*.parquet"))
    if published:
        return "finalized"
    if (d / "DONE").exists():
        return "done-no-parquet"
    if (d / ".ckpt").exists():
        return "started"
    return "pending"


def _log(root: Path, record: dict) -> None:
    with (root / "campaign.jsonl").open("a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}) + "\n")


def run_shard(
    plan: Plan,
    shard: Shard,
    *,
    root: Path,
    gpu: int,
    checkout: Path,
    image: str,
    collector_ref: str,
    mode: str = "run",
) -> int:
    d = shard_dir(root, shard)
    (d / ".cache").mkdir(parents=True, exist_ok=True)
    name = f"{'f' if mode == 'finalize' else 'c'}_{plan.backend}_{shard.op}_{shard.name}"
    cmd = docker_argv(
        plan,
        shard,
        image=image,
        gpu=gpu,
        checkout=checkout,
        shard_dir=d,
        container_name=name,
        collector_ref=collector_ref,
        mode=mode,
    )
    _log(
        root,
        {
            "event": "start",
            "mode": mode,
            "shard": shard.key,
            "gpu": gpu,
            "image": image,
            "image_digest": image_digest(image),
            "collector_ref": collector_ref,
            "argv": collector_argv(plan, shard),
        },
    )
    with (d / "run.log").open("a") as log:
        log.write(f"### {time.strftime('%FT%T')} {mode} gpu{gpu} {shlex.join(cmd)}\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"### {time.strftime('%FT%T')} exit={rc}\n")
    parquets = sorted(p.name for p in d.glob("*.parquet"))
    if rc == 0 and _native_k31(plan.backend, shard.op):
        if _native_phase_table(shard.op) not in parquets:
            rc = 1
            (d / "DONE").unlink(missing_ok=True)
            with (d / "run.log").open("a") as log:
                log.write("No native phase table was published. Inspect the checkpoint and case diagnostics.\n")
    if rc == 0:
        (d / "DONE").touch()
    _log(root, {"event": "end", "mode": mode, "shard": shard.key, "gpu": gpu, "exit": rc, "parquet": parquets})
    return rc


def _free_gpu(busy: dict[int, subprocess.Popen | None], gpus: list[int]) -> int | None:
    for g in gpus:
        p = busy.get(g)
        if p is None or p.poll() is not None:
            return g
    return None


def _validate_shard_allocation(plan: Plan, shard: Shard, *, finalize_checkpoint: Path | None = None) -> None:
    """Refuse a native TP group before launching a one-device container."""
    if not _native_k31(plan.backend, shard.op):
        return
    from collector.op_plans import plan as plan_cases
    from collector.sglang.collect_kimi_k31_dsa_v1 import validate_case
    from collector.sglang.registry import REGISTRY

    _require_native_model(shard.key, shard.model_path)
    entry = next(entry for entry in REGISTRY if entry.op == shard.op)
    if entry.unverified or plan.sm in entry.unverified_sms:
        raise SystemExit(f"{shard.key}: registry marks the native op unverified on SM{plan.sm}")
    # Match collect.py's --model-path selection, independent of the host shell.
    previous_model = os.environ.get("COLLECTOR_MODEL_PATH")
    try:
        os.environ["COLLECTOR_MODEL_PATH"] = shard.model_path
        cases = plan_cases(shard.op, plan.backend, plan.sm, include_long_context=shard.include_long_context).cases
    finally:
        if previous_model is None:
            os.environ.pop("COLLECTOR_MODEL_PATH", None)
        else:
            os.environ["COLLECTOR_MODEL_PATH"] = previous_model
    if not cases:
        raise SystemExit(f"{shard.key}: no qualified native cases on SM{plan.sm}")
    if finalize_checkpoint is not None and finalize_checkpoint.is_file():
        # The mounted collector checks every selected ID under the output lock
        # before publication; this flag forbids it from launching missing work.
        return
    for case in cases:
        if shard.model_path and case.model_path != shard.model_path:
            continue
        if shard.case_filter and shard.case_filter not in str(case):
            continue
        tp = validate_case(case)
        if tp > 1:
            raise SystemExit(
                f"{shard.key}: TP{tp} needs {tp} GPUs, but campaign shards expose one GPU; "
                "use collector/collect.py's full-node path (collector/fullnode.py) on a dedicated "
                "allocation exposing the full TP group, or select only TP1 cases with case_filter"
            )


def run_campaign(
    plan: Plan, *, root: Path, gpus: list[int], checkout: Path, image_override: str | None, mode: str
) -> int:
    """GPU pool over the plan. mode=run: skip DONE shards; mode=finalize: only done-no-parquet shards."""
    root.mkdir(parents=True, exist_ok=True)
    todo = []
    for s in plan.shards:
        st = shard_state(root, s, plan.backend)
        if mode == "run" and (shard_dir(root, s) / "DONE").exists():
            if st == "finalized" or not _native_k31(plan.backend, s.op):
                continue
        if mode == "finalize" and st != "done-no-parquet":
            continue
        checkpoint = shard_dir(root, s) / ".ckpt" / plan.backend / f"{plan.backend}.{s.op}.json"
        _validate_shard_allocation(plan, s, finalize_checkpoint=checkpoint if mode == "finalize" else None)
        todo.append(s)
    print(f"{mode}: {len(todo)}/{len(plan.shards)} shards on gpus {gpus}")
    ref = collector_ref(checkout)  # once, before any shard starts: every shard records the same commit
    images = {
        s.key: (image_override or collect_image_tag(pinned_image(plan.backend, s.op, s.model_path))) for s in todo
    }
    for img in sorted(set(images.values())):
        if image_digest(img) is None:
            raise SystemExit(f"image {img} not present — run `build-image` for its base first")
    busy: dict[int, subprocess.Popen | None] = {}
    failures = 0
    while todo:
        g = _free_gpu(busy, gpus)
        if g is None:
            time.sleep(15)
            continue
        finished = busy.get(g)
        if finished is not None and finished.returncode != 0:  # _free_gpu polled it
            failures += 1
        s = todo.pop(0)
        print(f"{time.strftime('%T')} gpu{g} <- {s.key}")
        busy[g] = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "_shard",
                "--backend",
                plan.backend,
                "--sm",
                str(plan.sm),
                "--root",
                str(root),
                "--gpu",
                str(g),
                "--shard",
                json.dumps(asdict(s)),
                "--checkout",
                str(checkout),
                "--image",
                images[s.key],
                "--collector-ref",
                ref,
                "--mode",
                mode,
            ]
        )
    for p in busy.values():
        if p is not None and p.wait() != 0:
            failures += 1
    print(f"{mode} finished, {failures} shard(s) exited non-zero")
    return 1 if failures else 0


def status(plan: Plan, root: Path) -> None:
    for s in plan.shards:
        d = shard_dir(root, s)
        last = ""
        log = d / "run.log"
        if log.exists():
            tail = log.read_text(errors="replace").replace("\r", "\n").splitlines()
            last = next(
                (ln for ln in reversed(tail) if "it/s" in ln or "s/it" in ln or "Completed" in ln or "exit=" in ln), ""
            )
        print(f"{s.op:28s} {s.name:20s} {shard_state(root, s, plan.backend):16s} {last[:100]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-image", help="build <manifest image for (backend, op)> + finalize deps")
    b.add_argument("--backend", required=True)
    b.add_argument("--op", required=True, help="any op of the backend; family pins may select a different image")
    b.add_argument("--model-path", default=None, help="model identity for model-scoped runtime pins")
    b.add_argument("--build-arg", action="append", default=[], help="extra docker build args, e.g. http_proxy=...")
    for name in ("run", "finalize", "status"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True, type=Path)
        p.add_argument("--root", required=True, type=Path, help="campaign root: one <op>/<shard>/ namespace per shard")
        if name != "status":
            p.add_argument("--gpus", required=True, help="comma list of GPU indices for the pool")
            p.add_argument("--checkout", type=Path, default=PKG_ROOT, help="python/aisim checkout to mount")
            p.add_argument("--image", default=None, help="override the manifest-derived collect image (A/B only)")
    h = sub.add_parser("_shard")  # internal: one shard in one container (spawned by run/finalize)
    h.add_argument("--backend", required=True)
    h.add_argument("--sm", required=True, type=int)
    for flag in ("--root", "--checkout"):
        h.add_argument(flag, required=True, type=Path)
    h.add_argument("--gpu", required=True, type=int)
    h.add_argument("--shard", required=True)
    h.add_argument("--image", required=True)
    h.add_argument("--collector-ref", required=True)
    h.add_argument("--mode", required=True)
    args = ap.parse_args()
    if args.cmd == "_shard":
        shard = Shard(**json.loads(args.shard))
        plan = Plan(backend=args.backend, sm=args.sm, shards=(shard,))
        return run_shard(
            plan,
            shard,
            root=args.root,
            gpu=args.gpu,
            checkout=args.checkout,
            image=args.image,
            collector_ref=args.collector_ref,
            mode=args.mode,
        )
    if args.cmd == "build-image":
        if _native_k31(args.backend, args.op):
            _require_native_model(args.op, args.model_path)
        base = pinned_image(args.backend, args.op, args.model_path)
        tag = build_image(
            base, model_path=args.model_path, extra_build_args=[f"--build-arg={a}" for a in args.build_arg]
        )
        print(f"built {tag} from {base}")
        return 0
    plan = load_plan(args.plan.resolve())
    if args.cmd == "status":
        status(plan, args.root.resolve())
        return 0
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    return run_campaign(
        plan,
        root=args.root.resolve(),
        gpus=gpus,
        checkout=args.checkout.resolve(),
        image_override=args.image,
        mode=args.cmd,
    )


if __name__ == "__main__":
    raise SystemExit(main())
