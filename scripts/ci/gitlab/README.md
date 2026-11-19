# Internal GitLab job: InferenceX ingestion (`aisim-hf-dataset-ci`)

`aisim-infx-ingest.gitlab-ci.yml` defines the weekly and manual InferenceX ingestion for the
private `aisim-infx-cache` (AISIM-2334, DEP #435). The repository has no other GitLab config, so this
directory holds only this definition. **Nothing here is registered**: a maintainer installs it in the
internal project by hand.

## Install checklist (human step)

1. In `aisim-hf-dataset-ci`, include this file (`include: project/ref/file`) at a reviewed ref.
2. Set the protected variables listed at the top of the file. `AISIM_REVIEWED_COMMIT` is a full
   40-hex SHA of this repository; bump it only through a reviewed merge request.
   Set the GitLab project policy `ci_pipeline_variables_minimum_override_role` to
   exclude everyone who must not override that reviewed pin (prefer `no_one_allowed`
   when pipeline variables are unnecessary). This control also applies to scheduled
   and manual runs. Protected project variables alone do not prevent higher-precedence
   pipeline variables from overriding `AISIM_REVIEWED_COMMIT` or `JOB_PY`.
3. Create a token that can write **only** `aisim-infx-cache` and store it as masked, protected
   `INFX_CACHE_WRITE_TOKEN`, with environment scope **`infx-cache-writer`**, never `*`.
   Only `publish_candidate` and `promote` select that environment. Add a separate masked,
   protected, read-only `INFX_CACHE_READ_TOKEN` for staleness. Acquisition (public GET) uses no credential.
4. Supply CI/CD inputs, not pipeline variables: weekly with `job=ingest`; daily with
   `job=staleness`. Manual refresh: run a web pipeline (default `job=ingest`). Promotion:
   web pipeline with `job=promote`, `candidate-snapshot-id`, and `candidate-generation`
   (the pipeline ID suffix printed by ingestion), started by the reviewing
   maintainer. Leave the generation empty for legacy `candidate/<snapshot-id>` branches.
   When including this definition, forward these inputs from the project's
   pipeline inputs through `include:inputs`; keep their enum/regex constraints.
   Use a GitLab version supporting pipeline and schedule inputs before installing.
5. Protect the `infx-cache-writer` environment and restrict **Allowed to deploy** to
   reviewing maintainers (including the account that owns the ingestion schedule).
   The manual `promote` job declares `environment.action: access`; GitLab permits
   listed users and administrators to play it. `GITLAB_USER_NAME` records the actor;
   it does not authorize reviewers on its own.
6. Select or provision a runner with a POSIX shell, Git, `uv`, and `python3`.
   On that runner, verify `sh -c 'git --version && uv --version && python3 --version'`
   succeeds before enabling either schedule. A Docker runner can supply these tools
   through its configured default image; this definition does not select an image.

After a successful promotion the qualification exists only on its candidate branch. Open a
reviewed merge request from that branch into the default branch and merge it: ingestion and
staleness jobs clone the default branch, so they keep seeing the previous qualified snapshot until
then. The jobs never push to the default branch.

## Policy (AISIM-2314)

| Item | Decision |
| --- | --- |
| Owner | Simone Chen |
| Failure alerts | Slack `#swdl-dynamo-aisim-daily` plus email to the owner |
| Staleness | warn at 8 days, stale at 14 days since the last qualified snapshot (alert only) |
| Unchanged refresh | no commit, no snapshot |
| Candidates | pushed to `candidate/<snapshot-id>-<pipeline-id>` branches; never auto-promoted, never pushed to the default branch |
| Failure | publication may leave immutable revisions in the checkout; the qualification pointer stays unchanged and the failed GitLab job does not push the checkout |

A `run --model ...` candidate limits observation membership to the selected models and replaces the qualified
snapshot when promoted. The raw evidence can still include sibling DB-key observations from a shared
benchmark page and full dated `workflow-info` responses. This filter is not a redaction boundary.
Its `changes.removed` and `change_counts.removed` include qualified
observations from unselected models. Review that removal count before promotion; use an
unfiltered run to retain the full model scope.

Ingestion skips an existing branch only when it still has the same snapshot pending against the
current qualified baseline and its snapshot and all member artifacts pass verification. A damaged
pending branch is preserved, and the healthy staged replacement is published as a new generation.
If content recurs after another snapshot was qualified, a new
pipeline-suffixed branch preserves the historical review and supports a fresh qualification.

Configuration shards are stored under `revisions/<content-sha256>/e2e/...`.
The revision covers the converted rows, manifest body and provenance, including producer metadata.
Original benchmark and workflow-info HTTP response bodies are retained unchanged as hash-linked
provenance alongside the normalized record bundles; removing `curve_*` context for deduplication does not erase source evidence.
Snapshots reference these immutable revisions, so a candidate that corrects a measurement or changes
configuration membership cannot alter earlier qualified truth. Unchanged revisions are verified and
reused. Existing snapshots at legacy `e2e/...` paths remain readable and their shards are retained;
the first refresh after upgrading may create a candidate solely to adopt versioned paths.
The run report counts changed revision paths as `change_counts.revised_shards`, separately from
raw-observation additions, corrections and removals. This includes producer-metadata-only revisions.
Producer SHA and attempt metadata from `workflow-info` are optional: absent values remain null,
and `warnings` lists each affected benchmark run URL and missing field (including unchanged runs).
Each HTTP page is limited to 64 MiB before parsing; larger pages fail with a structured run report.
An aggregated row reporting `decode_num_workers=0` is rejected before conversion stores a
configuration: zero cannot represent a serving replica count.

## Assumptions to confirm before install

- `aisim-infx-cache` is reachable by git over HTTPS with a bearer token (the DEP names the repository but
  not the host); adjust the clone/push lines if it is only reachable through the Hugging Face API.
- Slack alerts use an incoming webhook; email uses the internal SMTP relay named in `SMTP_HOST`.
  CI ingestion uses `run --no-notify` so the fallback job is its single failure-alert path;
  it includes the structured ingestion error when `run-report.json` exists. Standalone
  CLI runs still notify directly. Failure notification uses Python's standard library without cloning
  or installing AISim. The notify stage waits for earlier stages, including unexpected scheduled staleness failures; statuses 10/11
  remain allowed warnings and do not trigger the failure job.
- InferenceX data terms are unresolved (DEP #435 Risks): keep the cache private and never export it publicly.

The logic lives in `python/aisim/src/aisim_core/data/infx_ingest.py` (client, caches, snapshot)
and `infx_job.py` (state, promotion, staleness, alerts). The definition is checked by
`python/aisim/tests/unit/data/test_infx_job_definition.py`.
