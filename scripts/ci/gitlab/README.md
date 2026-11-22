# Internal GitLab job: InferenceX ingestion (`aisim-hf-dataset-ci`)

`aisim-infx-ingest.gitlab-ci.yml` defines the weekly and manual InferenceX ingestion for the
private `aisim-infx-cache` (DEP #435). The repository has no other GitLab config, so this
directory holds only this definition. **Nothing here is registered**: a maintainer installs it in the
internal project by hand.

## Install checklist (human step)

1. Copy this definition into the protected default branch of `aisim-hf-dataset-ci` through a reviewed
   merge request. In `.pinned.before_script`, replace `REPLACE_WITH_REVIEWED_40_HEX_AISIM_COMMIT`
   with the full 40-hex code-reviewed AISim commit and `REPLACE_WITH_APPROVED_HTTPS_CACHE_REPO_URL`
   with the approved private cache's exact HTTPS git URL. The source repository is fixed to
   `https://github.com/ai-dynamo/aisim.git`. The template fails before network access until configured.
   These values and `JOB_PY` are readonly shell assignments: pipeline, schedule and manual-job variables
   cannot override them. Change the pin or destination only through another reviewed merge request;
   never replace these assignments with CI/CD variables or inputs. Authenticated git commands disable
   redirects so the cache token cannot be forwarded to another URL. If another project includes the
   installed definition, pin its reviewed ref and restrict edits to that include and the definition.
2. Set the protected credentials and alert variables listed at the top of the file. Do not create
   CI/CD variables for `AISIM_REVIEWED_COMMIT`, `AISIM_REPO_URL`, `INFX_CACHE_REPO_URL` or `JOB_PY`.
   Set the GitLab project policy `ci_pipeline_variables_minimum_override_role` to
   `no_one_allowed` when pipeline variables are unnecessary, as additional protection for other
   job settings. The checkout and token-destination boundary is enforced by the shell assignments,
   independently of this policy; protected project variables alone do not provide that boundary.
3. Create a token that can write **only** `aisim-infx-cache` and store it as masked, protected
   `INFX_CACHE_WRITE_TOKEN`, with environment scope **`infx-cache-writer`**, never `*`.
   Only `publish_candidate` and `promote` select that environment. Add a separate masked,
   protected, read-only `INFX_CACHE_READ_TOKEN` for staleness with environment scope
   **`infx-cache-reader`**, never `*`; only `staleness` selects that environment, so no other job inherits it.
   Dependency setup and the staleness CLI run with the read token removed; only the authenticated clone uses it.
   Acquisition (public GET) uses no credential.
4. Supply CI/CD inputs, not pipeline variables: weekly with `job=ingest`; daily with
   `job=staleness`. Manual refresh: run a web pipeline (default `job=ingest`). Promotion:
   web pipeline with `job=promote`, `candidate-snapshot-id`, `candidate-generation`
   (the job ID suffix printed by ingestion; pipeline-ID suffixes of earlier branches remain valid) and
   `candidate-commit` (the full commit SHA of the branch tip the reviewer inspected, also printed by
   ingestion), started by the reviewing maintainer. Leave the generation empty for legacy
   `candidate/<snapshot-id>` branches.
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
   succeeds before enabling either schedule. `uv sync` also builds the local AISim Rust
   extension with Maturin, so the runner needs Rust/Cargo, a C/C++ compiler and a platform linker;
   run a cold `uv sync --frozen --project python/aisim` at `AISIM_REVIEWED_COMMIT` on the selected
   runner before enabling schedules. A Docker runner can supply these tools
   through its configured default image; this definition does not select an image.

After a successful promotion the qualification exists only on its candidate branch. Open a
reviewed merge request from that branch into the default branch and merge it: ingestion and
staleness jobs clone the default branch, so they keep seeing the previous qualified snapshot until
then. The jobs never push to the default branch.

## Policy

| Item | Decision |
| --- | --- |
| Owner | Simone Chen |
| Failure alerts | Slack `#swdl-dynamo-aisim-daily` plus email to the owner |
| Staleness | warn at 8 days, stale at 14 days since the last qualified snapshot (alert only) |
| Unchanged refresh | no commit, no snapshot |
| Candidates | pushed to `candidate/<snapshot-id>-<job-id>` branches; never auto-promoted, never pushed to the default branch |
| Failure | publication may leave immutable revisions in the checkout; the qualification pointer stays unchanged and the failed GitLab job does not push the checkout |

A `run --model ...` candidate limits observation membership to the selected models and replaces the qualified
snapshot when promoted. The raw evidence can still include sibling DB-key observations from a shared
benchmark page and full dated `workflow-info` responses. This filter is not a redaction boundary.
Its `changes.removed` and `change_counts.removed` include qualified
observations from unselected models. Review that removal count before promotion; use an
unfiltered run to retain the full model scope.

Ingestion decisions are tied to the default-branch commit it cloned: before it accepts an unchanged
refresh, skips a pending branch or publishes a candidate, it checks that the default branch still has that
commit and otherwise fails (exit 4) so a rerun starts from the new baseline. It also verifies the pending
snapshot and its member artifacts from the exact commit it pushes, so a Git ignore rule that omits a required
artifact stops the job before any branch is created.

Promotion refuses a candidate branch whose tip is not `candidate-commit`, and one whose recorded qualified
baseline differs from the default branch's current qualification (its change counts would be stale); rerun
ingestion from the new baseline in that case.

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
An aggregated row reporting `decode_num_workers=0` is accepted: the source uses zero as an irrelevant
placeholder there. The converter stores one effective replica, keeps the source value `0` in the
observation's source config and records an `evidence_gaps` entry naming the normalization; it is not
a reported replica count.

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
