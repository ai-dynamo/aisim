# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the pinned OpenEngine pilot inside an existing two-GPU Slurm job."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Barrier
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .config import FPMCollectionOptions
from .database import aggregate_cell, validate_formal_database_commit, write_formal_database
from .native_artifact import TRTLLM_RUNTIME_VERSION
from .planner import _git_revision
from .runner import _sigterm_as_interrupt
from .runtime_instrumentation import freeze_instrumentation, load_instrumentation
from .trtllm_plan import BUNDLED_MANIFEST, DECODE_KV_TOLERANCE, IMAGE, IMAGE_SHA256, MODEL_REVISION, POINTS, build_plan


def _save(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextmanager
def _process(command, log, environment):
    with log.open("wb") as sink:
        process = subprocess.Popen(
            command, env=environment, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def _wait(probe, processes, label, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError(f"{label}: launch exited before readiness; inspect the campaign logs")
        try:
            if result := probe():
                return result
        except (OSError, URLError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{label}: timed out waiting for readiness")


def _completion(url, model, prompt, output_tokens, start=None):
    request = Request(
        f"{url}/v1/completions",
        json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": output_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
            }
        ).encode(),
        {"Content-Type": "application/json"},
    )
    if start is not None:
        start.wait(timeout=10)
    submitted = time.monotonic()
    frames = 0
    first = None
    with urlopen(request, timeout=90) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                if first is not None:
                    return {"submitted_s": submitted, "first_s": first, "last_s": time.monotonic(), "frames": frames}
                break
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            if data.get("choices"):
                if first is None:
                    first = time.monotonic()
                frames += 1
    raise RuntimeError("TRT-LLM pilot completion stream ended without a result")


def _drive(url, model, phase):
    rng = random.Random(18335)
    prefill = phase == "prefill"
    length, out = (2048, 1) if prefill else (8192, 384)
    prompts = [[rng.randrange(100, 10000) for _ in range(length)] for _ in range(80 if prefill else 4)]
    start = None if prefill else Barrier(4)
    with ThreadPoolExecutor(max_workers=20 if prefill else 4) as pool:
        futures = [pool.submit(_completion, url, model, prompt, out, start) for prompt in prompts]
        return [future.result() for future in futures]


@contextmanager
def _energy_counter(uuids):
    import pynvml

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        handles = {uuid: pynvml.nvmlDeviceGetHandleByUUID(uuid) for uuid in uuids}
    except pynvml.NVMLError:
        handles = {}

    def read():
        try:
            return {
                uuid: (time.monotonic(), pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
                for uuid, handle in handles.items()
            }
        except pynvml.NVMLError:
            return {}

    try:
        yield read
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass


def _rates(gpus, start, end, replies):
    duration = max(reply["last_s"] for reply in replies) - min(reply["submitted_s"] for reply in replies)
    if duration <= 0:
        raise ValueError("pilot baseline has invalid elapsed time")
    watts = {}
    if set(start) == set(end) == set(gpus):
        for uuid, (stamp, mj) in start.items():
            elapsed, delta = end[uuid][0] - stamp, end[uuid][1] - mj
            if elapsed <= 0 or delta <= 0:
                watts.clear()
                break
            watts[uuid] = delta / (1000 * elapsed)
    return {
        "wall_s": duration,
        "serving_frames_s": sum(reply["frames"] for reply in replies) / duration,
        "mean_w_per_gpu": sum(watts.values()) / len(watts) if watts else None,
        "gpu_watts": watts,
        "gpu_uuids": gpus,
        "energy_start": start,
        "energy_end": end,
        "replies": replies,
    }


def _cell_context(plan, cell, attempt, job_id):
    phase = cell.workload_kind
    return {
        "attempt_id": attempt,
        "collector_attempt_id": attempt,
        "configuration": "collection",
        "phase": phase,
        "cell_id": cell.cell_id,
        "bundle_sha256": plan.runtime_instrumentation.sha256,
        "plan_sha256": plan.sha256,
        "container_sha256": plan.container_sha256,
        "run_id": f"slurm-{job_id}-{attempt}",
        "grid_digest": plan.options.benchmark_points_sha256,
        "decode_total_kv_read_tokens": POINTS["decode"]["total_kv_read_tokens"],
        "decode_kv_tolerance": DECODE_KV_TOLERANCE,
    }


def _run_arm(plan, cell, attempt, root, *, bundle, snapshot, image, bindings, sidecar_python, enabled):
    raw = root / "raw" / "node-0"
    raw.mkdir(parents=True)
    context = _cell_context(plan, cell, attempt, os.environ["SLURM_JOB_ID"])
    _save(raw / "context.json", context)
    _save(
        raw / "engine-options.json",
        {
            "enable_iter_perf_stats": True,
            "print_iter_log": True,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "enable_chunked_prefill": False,
            "kv_cache_config": {"enable_block_reuse": False, "free_gpu_memory_fraction": 0.85},
        },
    )
    grpc, http, system = (_port() for _ in range(3))
    while len({grpc, http, system}) != 3:
        grpc, http, system = (_port() for _ in range(3))
    url = f"http://127.0.0.1:{http}"
    script = Path(__file__).with_name("trtllm_serve.py").resolve()
    mounts = (root.parent.parent, snapshot, bindings, script.parent)
    engine_env = {
        "HF_HUB_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((str(bundle), str(bindings))),
        "AISIMULATE_TRTLLM_FPM_OBSERVER": "1" if enabled else "0",
        "AISIMULATE_RUNTIME_CONTEXT": str(raw / "context.json"),
        "AISIMULATE_RUNTIME_INSTRUMENTATION": str(bundle / "manifest.json"),
        "AISIMULATE_RUNTIME_OBSERVATION_DIR": str(raw),
    }
    discovery = root / "discovery"
    discovery.mkdir()
    sidecar_env = os.environ | {
        "DYN_DISCOVERY_BACKEND": "file",
        "DYN_FILE_DISCOVERY_ROOT": str(discovery),
        "DYN_NAMESPACE": f"aisim-{attempt}",
        "DYN_HTTP_PORT": str(http),
        "DYN_SYSTEM_PORT": str(system),
    }
    launch = [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--gpus=h200:2",
        f"--container-image={image}",
        "--container-mounts=" + ",".join(f"{path}:{path}" for path in dict.fromkeys(mounts)),
        "--no-container-mount-home",
        "--container-workdir=/tmp",
        "env",
        "TMPDIR=/tmp",
        # Pyxis can keep image defaults for environment variables like
        # PYTHONPATH. Apply these assignments inside the container.
        *(f"{key}={value}" for key, value in engine_env.items()),
        "python3",
        # The collector directory contains types.py. Safe-path mode prevents
        # the script directory from shadowing Python's standard library.
        "-P",
        str(script),
        "--engine",
        "--snapshot",
        str(snapshot),
        "--port",
        str(grpc),
        "--gpus",
        str(raw / "gpus.json"),
        "--options",
        str(raw / "engine-options.json"),
    ]
    _save(raw / "launch.json", {"command": launch, "observer_enabled": enabled})
    with ExitStack() as stack:
        engine = stack.enter_context(_process(launch, raw / "engine.log", os.environ))
        frontend = stack.enter_context(
            _process(
                [str(sidecar_python), "-m", "dynamo.frontend", "--http-host", "127.0.0.1"],
                raw / "frontend.log",
                sidecar_env,
            )
        )

        def engine_ready():
            with socket.create_connection(("127.0.0.1", grpc), timeout=1):
                return True

        # A cold Pyxis import and TRT-LLM sampling prewarm both precede
        # gRPC readiness. The Slurm allocation still bounds the full run.
        _wait(engine_ready, (engine, frontend), "TRT-LLM gRPC", timeout=480)
        sidecar = stack.enter_context(
            _process(
                [
                    str(sidecar_python),
                    "-m",
                    "dynamo.trtllm.sidecar",
                    "--grpc-endpoint",
                    f"127.0.0.1:{grpc}",
                    "--model-path",
                    str(snapshot),
                ],
                raw / "sidecar.log",
                sidecar_env,
            )
        )
        processes = (engine, frontend, sidecar)

        def model_ready():
            with urlopen(f"{url}/v1/models", timeout=2) as response:
                models = json.load(response)["data"]
            return models[0]["id"] if models else None

        model = _wait(model_ready, processes, "Dynamo frontend")
        gpus = json.loads((raw / "gpus.json").read_text())["uuids"]
        _completion(url, model, [812] * 128, 8)
        with _energy_counter(gpus) as read:
            start = read()
            replies = _drive(url, model, cell.workload_kind)
            end = read()
        receipt = _rates(gpus, start, end, replies)
        _save(raw / "throughput-energy.json", receipt)
        if enabled:
            try:
                _wait(
                    lambda: (raw / "benchmark.json").is_file() and (raw / "collector-provenance.json").is_file(),
                    processes,
                    "TRT-LLM observer artifact",
                    timeout=12,
                )
            except TimeoutError:
                # The observer also seals a valid latency sample on shutdown
                # when the work ended before the power window could close.
                pass
    if enabled and not ((raw / "benchmark.json").is_file() and (raw / "collector-provenance.json").is_file()):
        raise RuntimeError("TRT-LLM observer finished without a latency artifact; see rank records and engine log")
    return receipt


def _validate_coordinates(cell, rows):
    expected = POINTS[cell.workload_kind]
    if len(rows) != 1:
        raise ValueError(f"{cell.cell_id} must report one achieved point")
    row = rows[0]
    if (
        row["batch_size"] != expected["batch_size"]
        or row["total_prefill_tokens"] != expected["total_prefill_tokens"]
        or abs(row["total_kv_read_tokens"] - expected["total_kv_read_tokens"])
        > (DECODE_KV_TOLERANCE if cell.workload_kind == "decode" else 0)
    ):
        raise ValueError(f"{cell.cell_id} reported a point outside the frozen serving workload")


def _validate_clock(raw, gpus):
    """Verify the energy counter brackets and the two completion boundaries."""
    rank_paths = sorted(raw.glob("trtllm-*/rank-*.json"))
    if len(rank_paths) != 2:
        return "rank_boundary_mismatch", None
    leader, peer = (json.loads(path.read_text()) for path in rank_paths)
    start, end = leader["energy_start"], leader["energy_end"]
    if not start or not end or set(start["energy_mj"]) != gpus or set(end["energy_mj"]) != gpus:
        return "gpu_mapping_unavailable", None
    repeats = len(leader["steps"])
    if not repeats or len(peer["steps"]) < repeats or not peer["anchor"]:
        return "rank_boundary_mismatch", None
    # The two ranks share the node's monotonic clock. Review how far the
    # complete NVML read brackets extend from the matched completions.
    opening = (leader["anchor"]["completed_s"], peer["anchor"]["completed_s"])
    closing = (leader["steps"][-1]["completed_s"], peer["steps"][repeats - 1]["completed_s"])
    displacement = max(
        max(abs(value - bound) for value in stamps for bound in (read["before_s"], read["after_s"]))
        for stamps, read in ((opening, start), (closing, end))
    )
    duration = end["before_s"] - start["after_s"]
    details = {
        "boundary_displacement_s": displacement,
        "common_energy_window_s": duration,
        "opening_rank_times_s": opening,
        "closing_rank_times_s": closing,
    }
    # Limit both-boundary displacement to 1% of the measured common interval.
    if duration < 2 or not math.isfinite(displacement) or 2 * displacement > 0.01 * duration:
        return "ambiguous_energy_boundaries", details
    return None, details


def publish_campaign(plan, arms, systems_root):
    rows = []
    summaries = {}
    matched_uuids = None
    for cell, attempt, observed, off, on in arms:
        if set(off["gpu_uuids"]) != set(on["gpu_uuids"]):
            raise ValueError(f"{cell.cell_id} baseline and observer used different GPUs")
        gpus = set(on["gpu_uuids"])
        if len(gpus) != 2 or (matched_uuids is not None and matched_uuids != gpus):
            raise ValueError("TRT-LLM pilot must use the same two H200s throughout")
        matched_uuids = gpus
        measured = aggregate_cell(plan, cell, observed, expected_attempt_id=attempt)
        _validate_coordinates(cell, measured)
        reason, details = (None, None)
        off_power, on_power = off["mean_w_per_gpu"], on["mean_w_per_gpu"]
        paired_power = off_power is not None and on_power is not None
        if measured[0]["power_w"] is not None:
            reason, details = _validate_clock(observed / "raw" / "node-0", gpus)
            if not paired_power and reason is None:
                reason = "matched_power_unavailable"
            if reason:
                measured[0]["power_w"] = None
        summaries[cell.workload_kind] = {
            "serving_rate_change_pct": 100 * (on["serving_frames_s"] / off["serving_frames_s"] - 1),
            "mean_power_change_pct": 100 * (on_power / off_power - 1) if paired_power else None,
            "clock_evidence": details,
            "power_unavailable_reason": reason,
            "power_w": measured[0]["power_w"],
        }
        rows.extend(measured)
    parquet, metadata, skipped = write_formal_database(plan, rows, systems_root=systems_root)
    if skipped:
        raise ValueError(f"publication must use a fresh database root, existing cells: {skipped}")
    validate_formal_database_commit(parquet, metadata, plan)
    return parquet, rows, summaries


def query_published_profile(plan, parquet, rows):
    import aisim_core
    from aisim_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    first = rows[0]
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=plan.model_path,
            system=plan.system,
            backend=plan.backend,
            backend_version=plan.backend_version,
            tp=first["tp"],
            pp=first["pp"],
            attention_dp=first["dp"],
            moe_tp_size=first["moe_tp"],
            moe_ep_size=first["moe_ep"],
            gemm_quant_mode=first["gemm_quant_mode"],
            moe_quant_mode=first["moe_quant_mode"],
            fmha_quant_mode=first["fmha_quant_mode"],
            fpm_fmha_quant_mode=first["fmha_quant_mode"],
            kvcache_quant_mode=first["kv_cache_dtype"],
            comm_quant_mode=first["comm_quant_mode"],
            worker_type="aggregated",
            estimation_mode="fpm_interpolation",
            fallback_policy="deny",
            systems_paths=(str(Path(aisim_core.__file__).parent / "systems"),),
            estimator_config={"fpm_interpolation": {"fpm_parquet_path": str(parquet)}},
        )
    )
    queried = {}
    for row in rows:
        phase = row["workload_kind"]
        batch, q, kv = row["batch_size"], row["total_prefill_tokens"], row["total_kv_read_tokens"]
        metrics = {
            "scheduled_requests": {
                "num_prefill_requests": batch if phase == "prefill" else 0,
                "sum_prefill_tokens": q,
                "num_decode_requests": batch if phase == "decode" else 0,
                "sum_decode_kv_tokens": kv if phase == "decode" else 0,
            }
        }
        result = model.estimate_forward_pass_detailed([metrics])
        (rank,) = result["ranks"]
        phase_latency = rank[f"{phase}_ms"]
        if not math.isclose(phase_latency, row["latency_ms"]) or any(
            query["resolution"] != "exact_lookup" for query in rank["queries"]
        ):
            raise ValueError(f"published {phase} profile did not recover the measured point")
        energy = rank.get(f"{phase}_energy_wms")
        power = energy / phase_latency if energy is not None else None
        expected = row["power_w"]
        if (power is None) != (expected is None) or (power is not None and not math.isclose(power, expected)):
            raise ValueError(f"published {phase} profile query disagrees with power_w: {power} vs {expected}")
        queried[phase] = {"power_w": power, "details": result}
    return queried


def _require_allocation():
    job = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if not job or not node_list or os.environ.get("SLURM_JOB_NUM_NODES") != "1":
        raise ValueError("run TRT-LLM FPM pilot inside a one-node Slurm allocation with two H200s")
    nodes = subprocess.check_output(["scontrol", "show", "hostnames", node_list], text=True).split()
    if nodes != [socket.gethostname().split(".")[0]]:
        raise ValueError("the pilot must run on the allocated Slurm node")
    jobs = subprocess.run(
        ["squeue", "-h", "-t", "RUNNING,COMPLETING", "-w", nodes[0], "-o", "%A"],
        capture_output=True,
        text=True,
        check=False,
    )
    other_jobs = len(set(jobs.stdout.split()) - {job}) if jobs.returncode == 0 else None
    return {"job_id": job, "node": nodes[0], "other_jobs_at_start": other_jobs}


def run_cli(args, parser):
    if args.resume or args.smoke or args.limit is not None or args.repeatability_source_campaign:
        parser.error("TRT-LLM FPM pilot requires a fresh explicit two-point campaign")
    if args.fpm_runtime_backend_version not in (None, TRTLLM_RUNTIME_VERSION) or args.fpm_backend_version not in (
        None,
        TRTLLM_RUNTIME_VERSION,
    ):
        parser.error("TRT-LLM FPM supports only v1.3.0rc29")
    plan = build_plan(
        args.model_path,
        args.gpu,
        FPMCollectionOptions.from_args(args),
        load_instrumentation(BUNDLED_MANIFEST, TRTLLM_RUNTIME_VERSION),
        _git_revision(),
        IMAGE_SHA256,
    )
    if args.plan_only:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0
    allocation = _require_allocation()
    paths = (args.fpm_trtllm_model_snapshot, args.fpm_trtllm_bindings, args.fpm_trtllm_sidecar_python)
    if not all(paths) or not args.fpm_artifact_root or not args.fpm_database_root:
        parser.error("TRT-LLM FPM requires snapshot, bindings, sidecar Python and explicit artifact/database roots")
    snapshot, bindings, sidecar_python = (Path(item).absolute() for item in paths)
    if not snapshot.is_dir() or not snapshot.name.endswith(f"@{MODEL_REVISION}"):
        parser.error(f"TRT-LLM FPM requires the local Qwen3-32B-FP8 snapshot at {MODEL_REVISION}")
    if not bindings.is_dir() or not sidecar_python.is_file():
        parser.error("TRT-LLM runtime bindings or sidecar Python are missing")
    model_bytes = (snapshot / "config.json").read_bytes()
    model_config = json.loads(model_bytes)
    quantization = model_config.get("quantization_config", {})
    if (
        model_config.get("model_type") != "qwen3"
        or model_config.get("torch_dtype") != "bfloat16"
        or quantization.get("quant_method") != "fp8"
        or quantization.get("weight_block_size") != [128, 128]
    ):
        parser.error("the snapshot does not describe the pinned FP8 Qwen3 checkpoint")
    root = Path(args.fpm_artifact_root).absolute()
    root.mkdir(parents=True, exist_ok=False)
    bundle = freeze_instrumentation(plan.runtime_instrumentation, root / "bundle")
    _save(root / "plan.json", plan.to_dict())
    _save(root / "allocation.json", allocation | {"model_config_sha256": hashlib.sha256(model_bytes).hexdigest()})
    database_root = Path(args.fpm_database_root).absolute()
    database_root.mkdir(parents=True, exist_ok=False)
    arms = []
    with _sigterm_as_interrupt():
        for cell in plan.cells:
            results = {}
            attempt = uuid4().hex
            for mode in ("off", "on"):
                arm = root / cell.workload_kind / mode
                results[mode] = _run_arm(
                    plan,
                    cell,
                    attempt,
                    arm,
                    bundle=bundle.root,
                    snapshot=snapshot,
                    image=IMAGE,
                    bindings=bindings,
                    sidecar_python=sidecar_python,
                    enabled=mode == "on",
                )
            arms.append((cell, attempt, root / cell.workload_kind / "on", results["off"], results["on"]))
    parquet, rows, summaries = publish_campaign(plan, arms, database_root)
    _save(root / "acceptance.json", {"parquet": str(parquet), "phases": summaries, "rows": rows})
    _save(root / "profile-query.json", query_published_profile(plan, parquet, rows))
    print(json.dumps(summaries, indent=2))
    return 0 if all(row["power_w"] is not None for row in rows) else 1
