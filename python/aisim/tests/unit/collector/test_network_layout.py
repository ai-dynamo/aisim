# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_ROOT = REPO_ROOT / "collector"
NETWORK_ROOT = COLLECTOR_ROOT / "network"

SUBMIT_MOE_A2A = NETWORK_ROOT / "slurm" / "submit_moe_a2a.sh"
SUBMIT_TRTLLM_ALLTOALL = NETWORK_ROOT / "slurm" / "submit_trtllm_alltoall.sh"


def test_network_collectors_are_grouped_under_network_folder():
    expected_paths = [
        NETWORK_ROOT / "collect_comm.sh",
        NETWORK_ROOT / "collect_nccl.py",
        NETWORK_ROOT / "collect_oneccl_xpu.py",
        NETWORK_ROOT / "collect_all_reduce.py",
        NETWORK_ROOT / "slurm" / "collect_allreduce.py",
        NETWORK_ROOT / "slurm" / "collect_trtllm_alltoall.py",
        SUBMIT_TRTLLM_ALLTOALL,
        SUBMIT_MOE_A2A,
    ]

    for path in expected_paths:
        assert path.exists(), f"missing network collector: {path}"

    old_top_level_paths = [
        COLLECTOR_ROOT / "collect_comm.sh",
        COLLECTOR_ROOT / "collect_nccl.py",
        COLLECTOR_ROOT / "collect_oneccl_xpu.py",
        COLLECTOR_ROOT / "collect_all_reduce.py",
        COLLECTOR_ROOT / "slurm_comm_collector",
    ]

    for path in old_top_level_paths:
        assert not path.exists(), f"network collector should live under collector/network: {path}"


def test_slurm_network_docs_use_new_folder_name():
    docs = [
        NETWORK_ROOT / "README.md",
        NETWORK_ROOT / "slurm" / "README.md",
        SUBMIT_TRTLLM_ALLTOALL,
        SUBMIT_MOE_A2A,
    ]

    for path in docs:
        assert "slurm_comm_collector" not in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Launcher scripts
# ---------------------------------------------------------------------------


def _default_container_image(script: Path) -> str:
    """The launcher's CONTAINER_IMAGE default (`${CONTAINER_IMAGE:-<value>}`)."""
    match = re.search(r'CONTAINER_IMAGE="\$\{CONTAINER_IMAGE:-([^}]+)\}"', script.read_text(encoding="utf-8"))
    assert match, f"{script} declares no CONTAINER_IMAGE default"
    return match.group(1)


def _manifest_frameworks() -> dict:
    with (COLLECTOR_ROOT / "framework_manifest.yaml").open(encoding="utf-8") as manifest_file:
        return yaml.safe_load(manifest_file)["frameworks"]


def test_submit_moe_a2a_default_image_matches_manifest():
    # The launcher default must be the manifest wideep_sglang pin
    # (grace_blackwell variant: the launcher's default partition is GB200) —
    # never a hardcoded stale ref, the precedent launcher's drift bug.
    manifest_image = _manifest_frameworks()["wideep_sglang"]["default"]["images"]["grace_blackwell"]
    assert _default_container_image(SUBMIT_MOE_A2A) == manifest_image


def test_submit_trtllm_alltoall_default_image_matches_manifest():
    # Same sync requirement, against the WideEP all-to-all build base
    # (trtllm_a2a: the rc20 image is the immutable base the rc11 source wheel is
    # built over), NOT the stock trtllm pin — stock moved to rc29
    # while WideEP stays on its own pins. The collector gates the installed
    # tensorrt_llm version against this pin, so drift here would fail every
    # job loudly.
    manifest_image = _manifest_frameworks()["trtllm_a2a"]["default"]["images"]["default"]
    assert _default_container_image(SUBMIT_TRTLLM_ALLTOALL) == manifest_image


@pytest.mark.parametrize("script", [SUBMIT_MOE_A2A, SUBMIT_TRTLLM_ALLTOALL], ids=lambda path: path.name)
def test_launcher_scripts_pass_bash_syntax_check(script):
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n {script.name} failed: {result.stderr}"


def test_submit_moe_a2a_invokes_the_collector_with_the_world_layout():
    text = SUBMIT_MOE_A2A.read_text(encoding="utf-8")
    # The collector module, launched with the divisor node_num derives from.
    assert "python -m collector.wideep.sglang.collect_moe_a2a" in text
    assert "--gpus-per-node" in text
    assert "--output-path" in text
    # Slurm rendezvous + full-rack segment pinning per the precedent.
    assert "scontrol show hostname" in text
    assert "MASTER_PORT" in text
    assert "--mpi=pmix" in text
    assert "--segment" in text


def test_submit_launchers_use_per_job_output_dirs():
    # Each job finalizes its own parquet and attests a world-specific
    # collection_meta.yaml, so jobs get per-world output dirs; within a job
    # all cases append one CSV through log_perf's lockfile. The job-id
    # subdirectory (resolved inside the job) additionally keeps a
    # resubmission from appending after a preempted attempt's stale rows —
    # the collectors refuse a directory that already holds owned artifacts.
    for script in (SUBMIT_MOE_A2A, SUBMIT_TRTLLM_ALLTOALL):
        text = script.read_text(encoding="utf-8")
        assert 'OUTPUT_DIR="${SCRIPT_DIR}/results/moe_a2a_' in text, script.name
        assert "${NUM_GPUS}gpu" in text, script.name
        assert "${OUTPUT_DIR}/job_\\${SLURM_JOB_ID}" in text, script.name


def test_submit_launchers_pass_the_launched_image_to_the_collector():
    # Launcher-to-sidecar identity: the collector attests --image-ref, so it
    # must receive exactly the ref srun --container-image runs (including an
    # operator's CONTAINER_IMAGE override).
    for script in (SUBMIT_MOE_A2A, SUBMIT_TRTLLM_ALLTOALL):
        text = script.read_text(encoding="utf-8")
        assert '--image-ref \\"${CONTAINER_IMAGE}\\"' in text, script.name


def _default_gpu_list(script: Path) -> list[int]:
    for line in script.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith('GPU_LIST="'):
            return [int(item) for item in stripped.split('"')[1].split(",")]
    raise AssertionError(f"no GPU_LIST default found in {script.name}")


def test_advertised_default_worlds_expand_to_at_least_one_case(monkeypatch):
    # A default world that expands to zero cases reserves nodes (up to a full
    # rack) only to fail: every advertised default must survive the plan's
    # structural drops. 72 (sglang: no declared expert count divides 72) and
    # 48/72 (trtllm: the sole 256-expert shape) are deliberately absent.
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    from collector.network.slurm import collect_trtllm_alltoall as ata
    from collector.wideep.sglang import collect_moe_a2a as a2a

    gpus_per_node = 4  # both launchers' default
    grid = a2a.get_moe_a2a_workload_grid()
    for world in _default_gpu_list(SUBMIT_MOE_A2A):
        shapes = a2a.get_moe_a2a_shapes(required_expert_parallel_size=world)
        cases = a2a.build_case_plan(shapes=shapes, grid=grid, ep_size=world, node_num=world // gpus_per_node)
        assert cases, f"moe_a2a default world {world} expanded to zero cases"

    for world in _default_gpu_list(SUBMIT_TRTLLM_ALLTOALL):
        assert ata.get_default_test_cases(world), f"trtllm alltoall default world {world} expanded to zero cases"


def _run_collect_comm_with_fake_tools(
    tmp_path: Path,
    cuda_visible_devices: str | None,
    *,
    cuda_device_count: int = 0,
    cuda_query_fails: bool = False,
    expected_returncode: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run collect_comm.sh with fake device counts and a recording torchrun."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "torchrun_calls.txt"
    (bin_dir / "nvidia-smi").write_text("#!/bin/bash\nfor i in $(seq 8); do echo 'NVIDIA B300'; done\n")
    (bin_dir / "torchrun").write_text(f'#!/bin/bash\necho "$1" >> "{calls}"\n')
    for tool in bin_dir.iterdir():
        tool.chmod(0o755)
    (bin_dir / "python3").symlink_to(sys.executable)
    query_body = "raise RuntimeError('CUDA count query failed')" if cuda_query_fails else f"return {cuda_device_count}"
    (tmp_path / "torch.py").write_text(
        f"from types import SimpleNamespace\ndef device_count():\n    {query_body}\n"
        "cuda = SimpleNamespace(device_count=device_count)\n"
    )

    # Keep the active interpreter's runtime paths (for example LD_LIBRARY_PATH).
    env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "PYTHONPATH": str(tmp_path)}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    result = subprocess.run(
        ["bash", str(NETWORK_ROOT / "collect_comm.sh"), "--skip-nccl", "--all_reduce_backend", "vllm"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == expected_returncode, result.stdout + result.stderr
    return result, calls.read_text().split() if calls.exists() else []


def test_collect_comm_counts_all_gpus_without_cuda_visible_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    result, calls = _run_collect_comm_with_fake_tools(tmp_path, None)
    # BSD wc pads its count with spaces.
    assert re.search(r"Found\s+8 GPUs\.", result.stdout)
    assert calls == ["--nproc_per_node=2", "--nproc_per_node=4", "--nproc_per_node=8"]


def test_collect_comm_honors_cuda_visible_devices(tmp_path):
    # nvidia-smi lists every GPU on the node regardless of CUDA_VISIBLE_DEVICES;
    # on a node shared with other jobs the collector must stay on its own GPUs.
    result, calls = _run_collect_comm_with_fake_tools(tmp_path, "4,5,6,7", cuda_device_count=4)
    assert re.search(r"Found\s+4 GPUs\.", result.stdout)
    assert calls == ["--nproc_per_node=2", "--nproc_per_node=4"]


def test_collect_comm_uses_effective_cuda_device_count(tmp_path):
    # An invalid ordinal stops CUDA enumeration before later mask entries.
    result, calls = _run_collect_comm_with_fake_tools(tmp_path, "0,1,-1,2", cuda_device_count=2)
    assert re.search(r"Found\s+2 GPUs\.", result.stdout)
    assert calls == ["--nproc_per_node=2"]


def test_collect_comm_preserves_python_runtime_environment(tmp_path, monkeypatch):
    # setup-python's shared interpreter needs LD_LIBRARY_PATH on ARM64 CI.
    # Model that requirement even when the local interpreter has an rpath.
    runtime_library_path = os.pathsep.join(
        filter(None, (str(tmp_path / "python-lib"), os.environ.get("LD_LIBRARY_PATH")))
    )
    launcher = tmp_path / "python-with-runtime-env"
    launcher.write_text(
        f'#!/bin/bash\nif [[ "${{LD_LIBRARY_PATH:-}}" != {shlex.quote(runtime_library_path)} ]]; then\n'
        "    echo 'Missing Python runtime library path' >&2\n"
        "    exit 127\n"
        "fi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    launcher.chmod(0o755)
    monkeypatch.setenv("LD_LIBRARY_PATH", runtime_library_path)
    monkeypatch.setattr(sys, "executable", str(launcher))

    result, calls = _run_collect_comm_with_fake_tools(tmp_path, "0,1", cuda_device_count=2)

    assert re.search(r"Found\s+2 GPUs\.", result.stdout)
    assert calls == ["--nproc_per_node=2"]


@pytest.mark.parametrize(("cuda_visible_devices", "cuda_device_count"), [("", 0), ("6", 1)])
def test_collect_comm_rejects_fewer_than_two_visible_gpus(tmp_path, cuda_visible_devices, cuda_device_count):
    result, calls = _run_collect_comm_with_fake_tools(
        tmp_path, cuda_visible_devices, cuda_device_count=cuda_device_count, expected_returncode=1
    )
    assert re.search(rf"Found\s+{cuda_device_count} GPUs\.", result.stdout)
    assert "Error: single GPU detected." in result.stdout
    assert calls == []


def test_collect_comm_aborts_when_cuda_count_query_fails(tmp_path):
    result, calls = _run_collect_comm_with_fake_tools(tmp_path, "4,5,6,7", cuda_query_fails=True, expected_returncode=1)
    assert "RuntimeError: CUDA count query failed" in result.stderr
    assert not re.search(r"Found\s+.* GPUs\.", result.stdout)
    assert calls == []
