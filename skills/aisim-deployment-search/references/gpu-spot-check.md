<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU spot-check of AISim picks

Source: `docs/sweeper/deployment-generation.md` in ai-dynamo/aisim (0.14.0).
Step 1 runs on CPU and was tested with aisim 0.14.0. Steps 2 to 4 need GPUs
and run only after the user approves the picks, GPU count and duration.
They were not run during testing.

## 1. Render serving artifacts for one pick (CPU)

`recommendations/*.yaml` are prediction inputs, not deployments. Render a
selected candidate into Dynamo launch scripts with the generator API that
ships in the `aisim` wheel. Save as `render.py` and run
`python3 render.py ./aisim-recommendation <candidate_id>`; take the
candidate ID from the summarizer output.

```python
import json
import sys
from dataclasses import replace
from pathlib import Path

from aisim.generator.api import generate_from_request
from aisim.generator.request import from_sweeper_candidate

out = Path(sys.argv[1])
candidate_id = sys.argv[2]
result = json.loads((out / "recommendation.json").read_text())
candidate = next(r for r in result["candidates"] if r["candidate_id"] == candidate_id)
assert candidate["status"] == "feasible"
request = from_sweeper_candidate(
    candidate,
    workload=candidate["provenance"]["workload"],
    deployment_target="dynamo-j2",   # or dynamo-python, llm-d-helm, llm-d-kustomize, slurm
    output_dir=str(out.parent / f"generated-{candidate_id}"),
    generator_overrides={
        "ServiceConfig": {
            "model_path": "/workspace/models/<model-dir>",   # path inside the serving container
            "served_model_name": "<hf-model-id>",
            "head_node_ip": "127.0.0.1",
        },
    },
)
# Set this to the Dynamo version of the runtime image you will launch.
request = replace(request, backend=replace(request.backend, dynamo_version="<dynamo version>"))
print(candidate_id, sorted(generate_from_request(request)))
```

It writes:

- `run_0.sh`: frontend and workers;
- `bench_run.sh`: an AIPerf run;
- `k8s_deploy.yaml` and `k8s_bench.yaml`;
- `sflow.yaml`;
- per-role CLI argument files.

It worked for both aggregated and disaggregated picks.

Show the user any warning such as "No exact CLI args template for 0.30.0,
using closest prior template". It means the launch flags come from an
older backend template; check them against the runtime image before
launching. Confirm the runtime image named in `k8s_deploy.yaml`.

The bridge keeps the evaluated parallelism, worker counts and token limits,
and rejects what it cannot lower. Token budgets must be multiples of 64.
Run `bash -n` on every generated script.

## 2. Launch (GPU, approval required)

- Single host: start etcd and NATS from the matching Dynamo release's
  `dev/docker-compose.yml`. Run the Dynamo runtime image with the model
  mounted read-only, check the Dynamo and backend versions inside the
  container, then `bash run_0.sh`. Confirm `GET /v1/models` before
  benchmarking.
- Kubernetes: install the matching Dynamo platform and CRDs, set the
  namespace, model cache and image in the generator overrides, and review
  `k8s_deploy.yaml` with the user before applying it.

## 3. Benchmark with the same arrival process as the simulation

The generated `bench_run.sh` is closed-loop: it runs AIPerf with
`--concurrency`. It validates a simulation that used
`traffic.load.type: concurrency`. It does not validate a request-rate
(Poisson or constant) simulation: matching token lengths is not enough when
the arrival process differs.

Closed loop (simulation used `concurrency: C`):

```bash
aiperf profile -m <served-model-name> -u http://127.0.0.1:8000 \
  --endpoint-type chat --streaming --tokenizer <tokenizer-path-or-id> \
  --isl <input_tokens> --osl <output_tokens> --extra-inputs ignore_eos:true \
  --concurrency <C> --num-requests <requests in the simulation> \
  --warmup-request-count <2 x C> --artifact-dir ./bench-<candidate_id>
```

Open loop (simulation used `poisson` or `constant_rate` at R requests per
second):

```bash
aiperf profile -m <served-model-name> -u http://127.0.0.1:8000 \
  --endpoint-type chat --streaming --tokenizer <tokenizer-path-or-id> \
  --isl <input_tokens> --osl <output_tokens> --extra-inputs ignore_eos:true \
  --request-rate <R> --arrival-pattern <poisson|constant> \
  --num-requests <requests in the simulation> \
  --artifact-dir ./bench-<candidate_id>
```

These AIPerf flags were checked against an AIPerf source checkout from
August 2026. Run `aiperf profile --help` on the benchmark host before use,
and keep AIPerf's raw outputs. Check the completed request count before
comparing latency.

## 4. Compare, then clean up

- Compare like with like: output tokens per second per GPU, mean TTFT,
  per-request TPOT (AIPerf's per-request average ITL matches AISim's TPOT)
  and E2E, completed requests.
- The simulation ran `--stack engine`, without Dynamo's frontend, router or
  planner; the measurement includes them. Report that as a limitation of
  the comparison.
- Stop the serving processes or delete the deployment when done, and tell
  the user what was removed.
