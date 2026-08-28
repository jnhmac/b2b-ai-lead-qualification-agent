"""Raw and derived object storage: local filesystem or GCS, one interface.

Nothing above this module knows which backend is in use. The layout is
identical under a local root and a bucket:

    raw/{domain}/{crawl_date}/manifest.json
    raw/{domain}/{crawl_date}/{url_hash}.md
    extractions/pack={pack}/run={run_id}/part-*.jsonl
    classifications/pack={pack}/run={run_id}/part-*.jsonl

Two invariants the path helpers exist to protect:

`raw/` is keyed by crawl date only, never by pack version. Scraping and
prompting version independently, which is what turns a prompt revision from a
four-hour re-crawl into a three-minute re-run.

Writes are append-only. Overwriting a run silently rewrites history that an
earlier evaluation already cited, so `write_*` refuses to clobber an existing
object unless the caller passes `overwrite=True` and means it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from pydantic import BaseModel

from .config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._@=-]+$")


class StoreError(RuntimeError):
    """Base class for store failures."""


class StoreConflictError(StoreError):
    """A write would have overwritten an existing object."""


class ObjectNotFoundError(StoreError):
    """A read targeted an object that does not exist."""


# ---------------------------------------------------------------------------
# Path layout (backend-agnostic)
# ---------------------------------------------------------------------------


def url_hash(url: str) -> str:
    """Stable short hash of a URL, used as the raw page filename.

    Truncated sha256. Collisions at 16 hex chars are not a practical concern at
    ~10 pages per domain, and the full URL is recorded in the manifest anyway.
    """
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]


def _segment(value: str, label: str) -> str:
    text = str(value).strip()
    if not text or not _SAFE_SEGMENT.match(text) or text in {".", ".."}:
        # Dots are allowed inside a segment (domains need them) but a segment
        # that is only dots is traversal. `_normalize_key` catches it again on
        # the way to the backend; catching it here names the culprit field.
        raise ValueError(
            f"{label} must be a non-empty path-safe segment "
            f"(letters, digits, . _ @ = -), got {value!r}"
        )
    return text


def _crawl_date(value: date | str) -> str:
    return value.isoformat() if isinstance(value, date) else _segment(value, "crawl_date")


def raw_dir(domain: str, crawl_date: date | str) -> str:
    return f"raw/{_segment(domain, 'domain')}/{_crawl_date(crawl_date)}"


def raw_manifest_path(domain: str, crawl_date: date | str) -> str:
    return f"{raw_dir(domain, crawl_date)}/manifest.json"


def raw_page_path(domain: str, crawl_date: date | str, url: str) -> str:
    return f"{raw_dir(domain, crawl_date)}/{url_hash(url)}.md"


def _part_path(kind: str, pack: str, run_id: str, part: int = 0) -> str:
    return (
        f"{kind}/pack={_segment(pack, 'pack')}/run={_segment(run_id, 'run_id')}"
        f"/part-{part:05d}.jsonl"
    )


def extractions_path(pack: str, run_id: str, part: int = 0) -> str:
    return _part_path("extractions", pack, run_id, part)


def classifications_path(pack: str, run_id: str, part: int = 0) -> str:
    return _part_path("classifications", pack, run_id, part)


def _normalize_key(path: str) -> str:
    """Reject absolute paths and traversal, and settle on posix separators."""
    text = str(path).strip().replace("\\", "/")
    if not text:
        raise ValueError("store path must not be empty")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"store path must be relative and free of '..': {path!r}")
    return pure.as_posix()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _to_json_line(record: BaseModel | Mapping[str, Any]) -> str:
    if isinstance(record, BaseModel):
        return record.model_dump_json()
    if isinstance(record, Mapping):
        return json.dumps(record, ensure_ascii=False, default=_json_default)
    raise TypeError(f"JSONL records must be pydantic models or mappings, got {type(record).__name__}")


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def encode_jsonl(records: Iterable[BaseModel | Mapping[str, Any]]) -> str:
    """Render records as JSONL. Always ends with a newline, so parts concatenate."""
    lines = [_to_json_line(record) for record in records]
    return "".join(f"{line}\n" for line in lines)


def decode_jsonl(text: str, model: type[T] | None = None) -> list[Any]:
    """Parse JSONL, skipping blank lines. Reports the line number on bad JSON.

    Split on `\\n` only, NOT `str.splitlines()`. JSONL's delimiter is `\\n`
    and `encode_jsonl` never writes anything else between records. But a
    verbatim quote can legally contain U+2028/U+2029 (the JSON spec only
    requires escaping `"`, `\\` and control characters below U+0020), and
    `splitlines()` treats those as line breaks too — silently shredding a
    valid record into two fragments, one an "unterminated string". Real
    quotes hit this: scraped markdown renders a soft line break as U+2028.
    """
    out: list[Any] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise StoreError(f"malformed JSON on line {lineno}: {exc}") from exc
        out.append(model.model_validate(payload) if model else payload)
    return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Store(ABC):
    """Read/write JSONL and markdown, plus an existence check.

    Paths are relative and posix-style. The backend decides what that means.
    """

    backend: str

    @abstractmethod
    def uri(self, path: str) -> str:
        """Fully qualified location of `path`, for logs and manifests."""

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def _read(self, path: str) -> str: ...

    @abstractmethod
    def _write(self, path: str, text: str, content_type: str) -> None: ...

    @abstractmethod
    def list_paths(self, prefix: str = "") -> list[str]:
        """Every object under `prefix`, sorted."""

    # -- public API ---------------------------------------------------------

    def read_text(self, path: str) -> str:
        key = _normalize_key(path)
        if not self.exists(key):
            raise ObjectNotFoundError(f"no object at {self.uri(key)}")
        return self._read(key)

    def write_text(
        self,
        path: str,
        text: str,
        *,
        overwrite: bool = False,
        content_type: str = "text/plain; charset=utf-8",
    ) -> str:
        key = _normalize_key(path)
        if not overwrite and self.exists(key):
            raise StoreConflictError(
                f"{self.uri(key)} already exists; the store is append-only. "
                "Pass overwrite=True only when replacing a known-bad object."
            )
        self._write(key, text, content_type)
        log.debug("wrote %s (%d bytes)", self.uri(key), len(text.encode("utf-8")))
        return self.uri(key)

    def read_markdown(self, path: str) -> str:
        return self.read_text(path)

    def write_markdown(self, path: str, text: str, *, overwrite: bool = False) -> str:
        return self.write_text(
            path, text, overwrite=overwrite, content_type="text/markdown; charset=utf-8"
        )

    def read_jsonl(self, path: str, model: type[T] | None = None) -> list[Any]:
        return decode_jsonl(self.read_text(path), model)

    def write_jsonl(
        self,
        path: str,
        records: Iterable[BaseModel | Mapping[str, Any]],
        *,
        overwrite: bool = False,
    ) -> str:
        return self.write_text(
            path,
            encode_jsonl(records),
            overwrite=overwrite,
            content_type="application/x-ndjson; charset=utf-8",
        )

    def read_json(self, path: str) -> Any:
        return json.loads(self.read_text(path))

    def write_json(self, path: str, payload: Any, *, overwrite: bool = False) -> str:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        return self.write_text(
            path, text, overwrite=overwrite, content_type="application/json; charset=utf-8"
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.uri('')}>"


class LocalStore(Store):
    """Filesystem backend. The default for development."""

    backend = "local"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def uri(self, path: str) -> str:
        return (self.root / path).as_posix() if path else self.root.as_posix()

    def _full(self, path: str) -> Path:
        return self.root / path

    def exists(self, path: str) -> bool:
        return self._full(_normalize_key(path)).is_file()

    def _read(self, path: str) -> str:
        return self._full(path).read_text(encoding="utf-8")

    def _write(self, path: str, text: str, content_type: str) -> None:
        target = self._full(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write cannot leave a half-file that
        # exists() would report as a completed object.
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(target)

    def list_paths(self, prefix: str = "") -> list[str]:
        base = self.root / prefix if prefix else self.root
        if not base.exists():
            return []
        if base.is_file():
            return [base.relative_to(self.root).as_posix()]
        return sorted(
            item.relative_to(self.root).as_posix()
            for item in base.rglob("*")
            if item.is_file() and not item.name.endswith(".tmp")
        )


class GCSStore(Store):
    """Google Cloud Storage backend. Used for the full run.

    `client` is injectable so the backend can be exercised against a double
    without a network or credentials; production passes nothing and gets a real
    `google.cloud.storage.Client`.
    """

    backend = "gcs"

    def __init__(self, root: str, client: Any | None = None):
        self.bucket_name, self.prefix = parse_gs_uri(root)
        if client is None:
            try:
                from google.cloud import storage  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - depends on install
                raise StoreError(
                    "STORE_BACKEND=gcs needs google-cloud-storage installed"
                ) from exc
            client = storage.Client()
        self.client = client
        self._bucket = client.bucket(self.bucket_name)

    def _blob_name(self, path: str) -> str:
        return f"{self.prefix}/{path}" if self.prefix else path

    def uri(self, path: str) -> str:
        name = self._blob_name(path) if path else self.prefix
        return f"gs://{self.bucket_name}/{name}" if name else f"gs://{self.bucket_name}"

    def exists(self, path: str) -> bool:
        return bool(self._bucket.blob(self._blob_name(_normalize_key(path))).exists())

    def _read(self, path: str) -> str:
        return self._bucket.blob(self._blob_name(path)).download_as_text(encoding="utf-8")

    def _write(self, path: str, text: str, content_type: str) -> None:
        self._bucket.blob(self._blob_name(path)).upload_from_string(
            text, content_type=content_type
        )

    def list_paths(self, prefix: str = "") -> list[str]:
        full = self._blob_name(_normalize_key(prefix)) if prefix else self.prefix
        blobs = self.client.list_blobs(self.bucket_name, prefix=full)
        cut = len(self.prefix) + 1 if self.prefix else 0
        return sorted(blob.name[cut:] for blob in blobs)


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/optional/prefix`` into ``(bucket, prefix)``."""
    if not uri.startswith("gs://"):
        raise ValueError(f"not a gs:// URI: {uri!r}")
    body = uri[len("gs://") :].strip("/")
    if not body:
        raise ValueError(f"gs:// URI is missing a bucket: {uri!r}")
    bucket, _, prefix = body.partition("/")
    return bucket, prefix.strip("/")


def get_store(settings: Settings, *, client: Any | None = None) -> Store:
    """Build the store the settings ask for. The only place backends are chosen."""
    if settings.store_backend == "gcs":
        return GCSStore(settings.store_root, client=client)
    return LocalStore(settings.store_root)
