# sm90 path-gate inventory for Qwen3.8-Flash-Next

This directory records **what the path gates cover**, not that any gate passed.
Nothing here needs a GPU: `gates.py paths` reads the op plans that already ship
in the repository, and `gates.py plan` only renders shard plans.

## Why this exists

The onboarding PR adds a new model class (`Qwen4ExpForConditionalGeneration`)
and a new QSA op path. A reader is entitled to ask whether that path is
*gated* — whether a run of the model in serving produces the kernels the
collected rows claim. Answering it honestly needs a serving record and a
graded result, which this repository cannot produce without the maintainers'
collection pipeline (see [Not established](#not-established)).

What can be stated without a GPU is which paths the gates name, which are
derived for unnamed paths, and which have no gate at all. That is what the
files here contain.

## Commands

Run from `python/aisim/collector/opharness` on this branch:

```bash
python components/gates.py paths --sm 90 --fw vllm   > paths-vllm-sm90.txt
python components/gates.py paths --sm 90 --fw sglang > paths-sglang-sm90.txt
python components/gates.py paths --sm 90 --fw vllm   --op msa_context_module,msa_generation_module > paths-vllm-sm90-msa.txt
python components/gates.py paths --sm 90 --fw sglang --op msa_context_module,msa_generation_module > paths-sglang-sm90-msa.txt
python components/gates.py plan  --sm 90 --fw vllm --repo Qwen/Qwen3.8-Flash-Next --derive --out <dir>
```

`paths` prints one line per (op, dtype) path with its planned case count and
the gate that covers it, followed by a `sm:` summary line. Each file is the
verbatim stdout of the command above it.

## Result

| View | Paths | Gated | Derivable | No gate |
| --- | --- | --- | --- | --- |
| vLLM, all ops | 51 | 27 | 18 | 3 |
| SGLang, all ops | 69 | 31 | 13 | 10 |
| vLLM, QSA ops only | 8 | 2 | 6 | 0 |
| SGLang, QSA ops only | 4 | 2 | 2 | 0 |

The QSA rows are the ones this model adds. Both backends name the BF16 cell
(`msa_ctx_MiniMax-M3`, `msa_gen_MiniMax-M3`) and derive the FP8 / `fp8_block`
cells from that checkpoint, because `msa_context_module` and
`msa_generation_module` are shared ops, not model-specific ones. No QSA path at
sm90 is ungated.

`gates.py plan --repo Qwen/Qwen3.8-Flash-Next --derive` writes **zero** shards:
no entry in `components/gates.yaml` names this checkpoint, and `--derive` does
not mint a Qwen3.8-Flash-Next gate. This PR therefore adds no `gates.yaml`
row. Adding one without a serving record would name a gate nobody has graded.

## Not established

- **No serving record.** A path gate compares a collector case's kernels with
  what the framework deploys in a real serving run. Producing that record needs
  a GPU job in the framework image and the collection pipeline's
  `--url-base` upload, per the opharness README; those shard plans have not
  been run on a single GPU box.
- **No graded gate.** `gates.py grade` consumes unpacked pipeline results.
  There are none for this checkpoint, so nothing here is a pass.
- **No release-upgrade claim.** The inventory is sm90 at the releases
  `collector/framework_manifest.yaml` resolves on this branch: vLLM 0.31.0 for
  the `msa` family (`msa: *vllm_0310_runtime`) and SGLang 0.5.21 as the
  framework default. It says nothing about a future release.
