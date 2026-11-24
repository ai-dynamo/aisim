# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face model metadata files (``config.json`` and friends) behind an on-disk cache.

AISim bundles the configs of a fixed set of models. Every other model falls back to the small JSON files in its
Hugging Face repository. This module is that fallback, and it guarantees four things:

* **A branch is resolved to one commit.** ``main`` (or any branch or tag) is resolved to a full commit SHA once; the
  SHA is recorded in the cache and every later run of that model reads the same commit until the pin is refreshed.
  A full SHA passed as the revision needs no resolution.
* **Nothing is downloaded twice.** A file is stored under ``<cache>/model-configs/<owner>/<name>/<sha>/`` after a
  successful download and read from there before any network call. A file the repository does not have is recorded
  too (``<file>.absent``), because a commit never changes.
* **Offline means cache.** ``AISIM_DATA_OFFLINE=1`` never opens a connection; a file that is not cached raises
  :class:`ModelFileMissingError` and names the command that fetches it.
* **Rate limits are retried.** HTTP 429 (and transient 5xx) is retried with backoff, honoring ``Retry-After``.

The cache lives inside the AISim data cache (``AISIM_DATA_CACHE``, default ``~/.cache/aisim/data``). Only files the
user's own session downloaded from Hugging Face are stored there; nothing here ships or copies model metadata into
the repository or the AISim dataset. The network goes through the injectable :class:`ModelFileTransport`; tests use
fakes. Credentials are the standard Hugging Face ones (:func:`hf_auth_headers`) and are never stored or printed.
"""

from __future__ import annotations

import email.utils
import json
import logging
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

logger = logging.getLogger(__name__)

HF_ENDPOINT = "https://huggingface.co"
DEFAULT_REVISION = "main"
#: The model metadata AISim reads. Which of them a model needs depends on its architecture.
MODEL_FILES = ("config.json", "hf_quant_config.json", "preprocessor_config.json", "video_preprocessor_config.json")
CACHE_SUBDIR = "model-configs"
OFFLINE_ENV = "AISIM_DATA_OFFLINE"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "hf_model_manifest.json"

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_REPO = re.compile(r"(?:[A-Za-z0-9][\w.-]*/)?[A-Za-z0-9][\w.-]*")
_FILENAME = re.compile(r"[A-Za-z0-9][\w.-]*\.json")
_REQUEST_TIMEOUT_S = 30


# --------------------------------------------------------------------------- errors


class HuggingFaceDownloadError(Exception):
    """A Hugging Face JSON file could not be obtained."""


class HuggingFaceHTTPError(HuggingFaceDownloadError):
    """Hugging Face answered with an HTTP error; ``status`` is its code."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class ModelFileMissingError(HuggingFaceDownloadError):
    """Offline mode needs a model file that is not in the AISim data cache."""

    code = "missing-data"


# --------------------------------------------------------------------------- credentials and environment


def offline_requested() -> bool:
    """``AISIM_DATA_OFFLINE`` set to ``1``, ``true`` or ``yes``: the same switch as the rest of the AISim data."""
    return os.environ.get(OFFLINE_ENV, "").lower() in {"1", "true", "yes"}


def hf_token_path() -> Path:
    """Where ``hf auth login`` (and ``aisim init``) store the token."""
    default_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
    hf_home = Path(os.environ.get("HF_HOME", default_home))
    return Path(os.environ.get("HF_TOKEN_PATH", hf_home / "token")).expanduser()


def hf_auth_headers() -> dict[str, str]:
    """The ``Authorization`` header for the user's Hugging Face token, or none for an anonymous request.

    Order: ``HF_TOKEN``, ``HUGGING_FACE_HUB_TOKEN``, then the saved login (``$HF_HOME/token``).
    ``HF_HUB_DISABLE_IMPLICIT_TOKEN=1`` disables all of them.
    """
    if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").upper() in ("1", "ON", "YES", "TRUE"):
        return {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        try:
            hf_token = hf_token_path().read_text().strip()
        except FileNotFoundError:
            hf_token = None
    headers: dict[str, str] = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    return headers


def _auth_hint() -> str:
    return (
        "authenticate via one of the following (in priority order): (1) set the HF_TOKEN environment variable, "
        "(2) set the HUGGING_FACE_HUB_TOKEN environment variable, or "
        f"(3) run `aisim init` or `hf auth login` (token stored at {hf_token_path()})"
    )


# --------------------------------------------------------------------------- transport


class ModelFileTransport(Protocol):
    """The only network surface. Both calls may be slow and raise :class:`HuggingFaceDownloadError`."""

    def resolve_revision(self, repo: str, revision: str) -> str:
        """The full 40-hex commit SHA of a branch, tag or commit of the model repository."""

    def read_file(self, repo: str, sha: str, filename: str) -> bytes | None:
        """The bytes of ``filename`` at commit ``sha``, or ``None`` when the repository has no such file (HTTP 404)."""


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry of rate limits and transient server errors."""

    attempts: int = 5  # total tries, the first included
    base_delay_s: float = 1.0  # doubled after every failed try
    max_delay_s: float = 30.0  # cap of the doubling
    max_retry_after_s: float = 60.0  # a longer ``Retry-After`` is reported, not slept
    statuses: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def parse_retry_after(value: str | None, *, now: Callable[[], float] = time.time) -> float | None:
    """Seconds named by a ``Retry-After`` header (delta-seconds or an HTTP date); ``None`` when absent or unusable."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = when.timestamp() - now()
    return max(0.0, seconds) if math.isfinite(seconds) else None


class UrllibTransport:
    """``https://huggingface.co`` over ``urllib`` with the user's token and :class:`RetryPolicy`."""

    def __init__(
        self,
        *,
        retry: RetryPolicy | None = None,
        urlopen: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        headers: Callable[[], dict[str, str]] | None = None,
        endpoint: str = HF_ENDPOINT,
    ):
        self.retry = retry or RetryPolicy()
        self._urlopen = urlopen or urllib.request.urlopen
        self._sleep = sleep
        self._headers = headers or hf_auth_headers
        self._endpoint = endpoint

    def resolve_revision(self, repo: str, revision: str) -> str:
        url = f"{self._endpoint}/api/models/{repo}/revision/{quote(revision, safe='')}"
        body = self._get(url, f"{repo}@{revision}")
        try:
            sha = json.loads(body).get("sha")
        except (ValueError, AttributeError) as exc:
            raise HuggingFaceDownloadError(f"Hugging Face returned an unreadable answer for {url}: {exc}") from exc
        if not isinstance(sha, str) or not _FULL_SHA.fullmatch(sha):
            raise HuggingFaceDownloadError(f"{repo}@{revision} did not resolve to a full commit SHA")
        return sha

    def read_file(self, repo: str, sha: str, filename: str) -> bytes | None:
        url = f"{self._endpoint}/{repo}/raw/{quote(sha, safe='')}/{filename}"
        try:
            return self._get(url, f"{repo}'s {filename}")
        except HuggingFaceHTTPError as exc:
            if exc.status == 404:
                return None
            raise

    def _get(self, url: str, what: str) -> bytes:
        for attempt in range(1, self.retry.attempts + 1):
            try:
                request = urllib.request.Request(url, headers=self._headers())
                with self._urlopen(request, timeout=_REQUEST_TIMEOUT_S) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                delay = None
                if exc.code in self.retry.statuses and attempt < self.retry.attempts:
                    delay = self._delay(exc, attempt)
                if delay is None:
                    raise self._error(what, url, exc) from exc
                logger.warning(
                    "Hugging Face returned HTTP %s for %s; retrying in %.0f s (try %d of %d)%s",
                    exc.code,
                    what,
                    delay,
                    attempt + 1,
                    self.retry.attempts,
                    "; a Hugging Face token raises the rate limit (run `aisim init`)" if exc.code == 429 else "",
                )
                self._sleep(delay)
            except Exception as exc:
                raise HuggingFaceDownloadError(f"Failed to download {what} from HuggingFace: {exc}") from exc
        raise AssertionError("unreachable: the last try raises")  # pragma: no cover

    def _delay(self, exc: urllib.error.HTTPError, attempt: int) -> float | None:
        """Seconds to wait before the next try, or ``None`` to stop because the server asks for too long a wait."""
        asked = parse_retry_after(exc.headers.get("Retry-After") if exc.headers is not None else None)
        if asked is None:
            return min(self.retry.base_delay_s * 2 ** (attempt - 1), self.retry.max_delay_s)
        return asked if asked <= self.retry.max_retry_after_s else None

    @staticmethod
    def _error(what: str, url: str, exc: urllib.error.HTTPError) -> HuggingFaceHTTPError:
        message = (
            f"Failed to download {what} from HuggingFace: HuggingFace returned HTTP error {exc.code}: {exc.reason}. "
            f"URL: {url}."
        )
        if exc.code == 429:
            asked = parse_retry_after(exc.headers.get("Retry-After") if exc.headers is not None else None)
            wait = f" Hugging Face asked to wait {asked:.0f} s." if asked is not None else ""
            message += f" Hugging Face is rate limiting this client.{wait} To raise the limit, {_auth_hint()}."
        elif exc.code in (401, 403):
            message += f" If this is a gated or private model, accept its terms on huggingface.co and {_auth_hint()}."
        return HuggingFaceHTTPError(message, exc.code)


def make_transport() -> ModelFileTransport:
    """Transport factory; tests patch this to inject a fake."""
    return UrllibTransport()


# --------------------------------------------------------------------------- cache


def default_cache_root() -> Path:
    """``<AISIM_DATA_CACHE or ~/.cache/aisim/data>/model-configs``."""
    from aisim_core.data.fetch import default_cache_dir

    return default_cache_dir() / CACHE_SUBDIR


def check_repo(repo: str) -> None:
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise HuggingFaceDownloadError(
            f"{repo!r} is neither an existing local model directory nor a Hugging Face model ID (owner/name)"
        )


def check_filename(filename: str) -> None:
    if not _FILENAME.fullmatch(filename):
        raise HuggingFaceDownloadError(f"{filename!r} is not a model metadata JSON file name")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class ModelFileCache:
    """``<root>/<owner>/<name>/<sha>/<file>`` plus ``<root>/<owner>/<name>/refs/<ref>`` holding a resolved SHA.

    Writes are atomic, so an interrupted run leaves nothing at the final path. A repository without an owner
    (``gpt2``) uses the owner directory ``_``, which no Hugging Face owner can be named.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_cache_root()

    def repo_dir(self, repo: str) -> Path:
        check_repo(repo)
        owner, _, name = repo.rpartition("/")
        return self.root / (owner or "_") / name

    def repos(self) -> list[str]:
        """The repositories with anything cached."""
        if not self.root.is_dir():
            return []
        return [
            name.name if owner.name == "_" else f"{owner.name}/{name.name}"
            for owner in sorted(self.root.iterdir())
            if owner.is_dir()
            for name in sorted(owner.iterdir())
            if name.is_dir()
        ]

    def pinned(self, repo: str, revision: str) -> str | None:
        """The commit ``revision`` was resolved to earlier, if recorded."""
        try:
            sha = (self.repo_dir(repo) / "refs" / quote(revision, safe="")).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return sha if _FULL_SHA.fullmatch(sha) else None

    def pin(self, repo: str, revision: str, sha: str) -> None:
        _atomic_write(self.repo_dir(repo) / "refs" / quote(revision, safe=""), sha.encode())

    def read(self, repo: str, sha: str, filename: str) -> tuple[str, Any]:
        """``("hit", parsed)``, ``("absent", None)`` (the commit has no such file) or ``("miss", None)``."""
        check_filename(filename)
        directory = self.repo_dir(repo) / sha
        path = directory / filename
        try:
            data = json.loads(path.read_bytes())
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable cached %s (%s); it will be downloaded again", path, exc)
        else:
            if not isinstance(data, dict):
                raise HuggingFaceDownloadError(f"Cached {path} must contain a JSON object")
            return "hit", data
        if (directory / f"{filename}.absent").is_file():
            return "absent", None
        return "miss", None

    def write(self, repo: str, sha: str, filename: str, raw: bytes) -> None:
        check_filename(filename)
        directory = self.repo_dir(repo) / sha
        _atomic_write(directory / filename, raw)
        (directory / f"{filename}.absent").unlink(missing_ok=True)

    def mark_absent(self, repo: str, sha: str, filename: str) -> None:
        check_filename(filename)
        _atomic_write(self.repo_dir(repo) / sha / f"{filename}.absent", b"")


# --------------------------------------------------------------------------- loading


class ModelFiles:
    """The files of one model repository at one resolved commit.

    ``get`` answers from the cache first, then (unless offline) from the transport, and records what it learned.
    Its second result says where the answer came from: ``"cached"``, ``"fetched"`` or ``"absent"`` (the commit has
    no such file; the answer is ``None`` when the file is optional).
    """

    def __init__(
        self,
        repo: str,
        revision: str = DEFAULT_REVISION,
        *,
        transport: ModelFileTransport | None = None,
        cache: ModelFileCache | None = None,
        offline: bool | None = None,
        refresh: bool = False,
    ):
        check_repo(repo)
        self.repo, self.revision = repo, revision
        self.cache = cache if cache is not None else ModelFileCache()
        self.offline = offline_requested() if offline is None else offline
        self.refresh = refresh
        self._transport = transport
        self._sha: str | None = revision if _FULL_SHA.fullmatch(revision) else None
        self.downloaded = False  # True once this object has asked Hugging Face anything
        self._unrecorded = False  # a freshly resolved commit that the cache does not know yet

    def _network(self) -> ModelFileTransport:
        if self._transport is None:
            self._transport = make_transport()
        return self._transport

    def _missing(self, filename: str, why: str) -> ModelFileMissingError:
        return ModelFileMissingError(
            f"offline ({OFFLINE_ENV}=1): {self.repo}'s {filename} at revision {self.revision!r} {why} in the AISim "
            f"data cache ({self.cache.root}). While online, run `aisim data fetch --category model-config "
            f"--model {self.repo}` (or `aisim init --all`), or unset {OFFLINE_ENV}."
        )

    @property
    def sha(self) -> str:
        """The commit this model's files come from: a recorded pin, else the revision resolved now."""
        if self._sha is None:
            if not self.refresh:
                self._sha = self.cache.pinned(self.repo, self.revision)
            if self._sha is None:
                if self.offline:
                    raise self._missing("config.json", "has no pinned commit")
                self.downloaded = True
                self._sha = self._network().resolve_revision(self.repo, self.revision)
                self._unrecorded = True  # recorded with the first file the commit yields, never for an unreadable repo
        return self._sha

    def _record_pin(self) -> None:
        if self._unrecorded:
            self._unrecorded = False
            self.cache.pin(self.repo, self.revision, self.sha)
            logger.info("Pinned %s@%s to commit %s", self.repo, self.revision, self.sha)

    def get(self, filename: str, *, required: bool = True) -> tuple[dict | None, str]:
        check_filename(filename)
        sha = self.sha
        state, data = self.cache.read(self.repo, sha, filename)
        if state == "hit":
            self._record_pin()
            return data, "cached"
        if state == "miss":
            if self.offline:
                raise self._missing(filename, "is not cached")
            self.downloaded = True
            raw = self._network().read_file(self.repo, sha, filename)
            if raw is not None:
                try:
                    data = json.loads(raw)
                except ValueError as exc:
                    raise HuggingFaceDownloadError(
                        f"Failed to download {self.repo}'s {filename} from HuggingFace: not valid JSON ({exc})"
                    ) from exc
                if not isinstance(data, dict):
                    raise HuggingFaceDownloadError(f"{self.repo}'s {filename} must contain a JSON object")
                self.cache.write(self.repo, sha, filename, raw)
                self._record_pin()
                return data, "fetched"
            self.cache.mark_absent(self.repo, sha, filename)
        if required:
            raise HuggingFaceHTTPError(
                f"Failed to download {self.repo}'s {filename} from HuggingFace: the repository has no {filename} "
                f"at commit {sha} (HTTP 404). URL: {HF_ENDPOINT}/{self.repo}/raw/{sha}/{filename}",
                404,
            )
        self._record_pin()
        return None, "absent"


def load_model_json(
    repo: str,
    filename: str,
    *,
    revision: str = DEFAULT_REVISION,
    required: bool = True,
    transport: ModelFileTransport | None = None,
    cache: ModelFileCache | None = None,
    offline: bool | None = None,
    refresh: bool = False,
) -> dict | None:
    """One metadata file of a Hugging Face model, from the cache or (when allowed) the network.

    ``required=False`` returns ``None`` for a file the repository does not have (HTTP 404); every other failure
    raises :class:`HuggingFaceDownloadError`, and offline mode raises :class:`ModelFileMissingError` for a file that
    is not cached.
    """
    files = ModelFiles(repo, revision, transport=transport, cache=cache, offline=offline, refresh=refresh)
    return files.get(filename, required=required)[0]


# --------------------------------------------------------------------------- manifest and prefetch


def load_manifest(path: Path | None = None) -> tuple[str, ...]:
    """The model IDs ``aisim init`` prefetches: a list of Hugging Face IDs only, never any metadata."""
    path = path or MANIFEST_PATH
    try:
        models = json.loads(path.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HuggingFaceDownloadError(f"{path}: unreadable model manifest: {type(exc).__name__}: {exc}") from exc
    if not isinstance(models, list) or not all(isinstance(m, str) and _REPO.fullmatch(m) for m in models):
        raise HuggingFaceDownloadError(f"{path}: 'models' must be a list of Hugging Face model IDs")
    return tuple(models)


@dataclass
class PrefetchEntry:
    """What prefetching one model did.

    ``status``: ``bundled`` (AISim ships its config, nothing to fetch), ``cached`` (already complete), ``fetched``
    (downloaded now), ``skipped`` (gated, denied or not found: a warning, not a failure), ``failed`` (rate limit,
    network/server error, invalid metadata or missing required processor) or ``missing`` (offline and not cached).
    """

    model: str
    status: str
    revision: str | None = None
    files: dict[str, str] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if v not in (None, {})}


def prefetch_model(
    model: str,
    *,
    revision: str | None = None,
    bundled: Iterable[str] = (),
    extra_files: Callable[[dict], dict[str, bool]] = lambda _config: {},
    transport: ModelFileTransport | None = None,
    cache: ModelFileCache | None = None,
    offline: bool | None = None,
    refresh: bool = False,
) -> PrefetchEntry:
    """Cache every metadata file AISim reads for ``model``: ``config.json``, ``hf_quant_config.json`` and the
    ``extra_files`` its config calls for (the loader's own file-to-requiredness plan).
    Never raises for a per-model problem."""
    if revision is None and model in set(bundled):
        return PrefetchEntry(model, "bundled")
    entry = PrefetchEntry(model, "cached")
    try:
        files = ModelFiles(
            model, revision or DEFAULT_REVISION, transport=transport, cache=cache, offline=offline, refresh=refresh
        )
        config, source = files.get("config.json")
        entry.revision = files.sha
        entry.files["config.json"] = source
        for name, required in {"hf_quant_config.json": False, **extra_files(config)}.items():
            try:
                entry.files[name] = files.get(name, required=required)[1]
            except HuggingFaceHTTPError as exc:
                if required and exc.status == 404:
                    entry.files[name] = "absent"
                    raise HuggingFaceDownloadError(f"Incomplete model config cache: {exc}") from exc
                raise
    except ModelFileMissingError as exc:
        entry.status, entry.reason = "missing", str(exc)
    except HuggingFaceHTTPError as exc:
        entry.status = "skipped" if exc.status in (401, 403, 404) else "failed"
        entry.reason = _skip_reason(exc.status, model) if entry.status == "skipped" else str(exc)
    except Exception as exc:  # one model never stops the rest
        entry.status, entry.reason = "failed", f"{type(exc).__name__}: {exc}"
    else:
        if files.downloaded:
            entry.status = "fetched"
    return entry


def _skip_reason(status: int, model: str) -> str:
    if status == 404:
        return f"not found on Hugging Face (HTTP 404): {model} may have been renamed or removed"
    return (
        f"HTTP {status}: {model} is gated or private; accept its terms on huggingface.co with the account of your "
        "token (run `aisim init` or `hf auth login`) and fetch it again"
    )
