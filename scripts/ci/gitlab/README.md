# Internal GitLab job: InferenceX ingestion (`aisim-hf-dataset-ci`)

`aisim-infx-ingest.gitlab-ci.yml` defines the weekly and manual InferenceX ingestion for the
private `aisim-infx-cache` (AISIM-2334, DEP #435). The repository has no other GitLab config, so this
directory holds only this definition. **Nothing here is registered**: a maintainer installs it in the
internal project by hand.

## Install checklist (human step)

1. In `aisim-hf-dataset-ci`, include this file (`include: project/ref/file`) at a reviewed ref.
2. Set the protected variables listed at the top of the file. `AISIM_REVIEWED_COMMIT` is a full
   40-hex SHA of this repository; bump it only through a reviewed merge request.
3. Create a token that can write **only** `aisim-infx-cache` and store it as masked, protected
   `INFX_CACHE_WRITE_TOKEN`, with environment scope **`infx-cache-writer`**, never `*`.
   Only `publish_candidate` and `promote` select that environment. Add a separate masked,
   protected, read-only `INFX_CACHE_READ_TOKEN` for staleness. Acquisition (public GET) uses no credential.
4. Schedules: weekly with `AISIM_JOB=ingest`; daily with `AISIM_JOB=staleness`. Manual refresh:
   run a web pipeline (default `ingest`). Promotion: web pipeline with `AISIM_JOB=promote` and
   `CANDIDATE_SNAPSHOT_ID`, started by the reviewing maintainer.

After a successful promotion the qualification exists only on `candidate/<snapshot-id>`. Open a
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
| Candidates | pushed to `candidate/<snapshot-id>` branches; never auto-promoted, never pushed to the default branch |
| Failure | writes nothing; the last qualified snapshot in `ingestion-state.json` is retained |

A `run --model ...` candidate contains only the selected models and replaces the qualified
snapshot when promoted. Its `changes.removed` and `change_counts.removed` include qualified
observations from unselected models. Review that removal count before promotion; use an
unfiltered run to retain the full model scope.

Configuration shards are stored under `revisions/<content-sha256>/e2e/...`.
The revision covers the converted rows, manifest body and provenance, including producer metadata.
Original benchmark HTTP response bodies are retained unchanged as hash-linked provenance alongside
the normalized record bundles; removing `curve_*` context for deduplication does not erase source evidence.
Snapshots reference these immutable revisions, so a candidate that corrects a measurement or changes
configuration membership cannot alter earlier qualified truth. Unchanged revisions are verified and
reused. Existing snapshots at legacy `e2e/...` paths remain readable and their shards are retained;
the first refresh after upgrading may create a candidate solely to adopt versioned paths.

## Assumptions to confirm before install

- `aisim-infx-cache` is reachable by git over HTTPS with a bearer token (the DEP names the repository but
  not the host); adjust the clone/push lines if it is only reachable through the Hugging Face API.
- Slack alerts use an incoming webhook; email uses the internal SMTP relay named in `SMTP_HOST`.
  Failure notification uses Python's standard library without cloning or installing AISim. The notify
  stage waits for earlier stages, including unexpected scheduled staleness failures; statuses 10/11
  remain allowed warnings and do not trigger the failure job.
- InferenceX data terms are unresolved (DEP #435 Risks): keep the cache private and never export it publicly.

The logic lives in `python/aisim/src/aisim_core/data/infx_ingest.py` (client, caches, snapshot)
and `infx_job.py` (state, promotion, staleness, alerts). The definition is checked by
`python/aisim/tests/unit/data/test_infx_job_definition.py`.
