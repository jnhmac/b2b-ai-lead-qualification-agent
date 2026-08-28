"""Shared fixtures, including an in-memory stand-in for the GCS client.

The round-trip gate has to cover both backends, and the GCS backend must be
exercisable in CI with no bucket and no credentials. `FakeGCSClient` implements
the slice of `google.cloud.storage` that `GCSStore` actually calls, so the real
`GCSStore` code runs end to end: prefix joining, key normalization, existence
checks, UTF-8 encoding, JSONL framing.

What it does not cover, and what only a real bucket can: authentication,
network failures, eventual consistency, and object-level permissions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from cleanroom.config import Settings, load_pack
from cleanroom.store import GCSStore, LocalStore

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = REPO_ROOT / "packs"
GOLDSET_PATH = REPO_ROOT / "eval" / "goldset.jsonl"


# ---------------------------------------------------------------------------
# GCS double
# ---------------------------------------------------------------------------


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str):
        self._bucket = bucket
        self.name = name

    def exists(self) -> bool:
        return self.name in self._bucket.objects

    def upload_from_string(self, data: str | bytes, content_type: str | None = None) -> None:
        payload = data.encode("utf-8") if isinstance(data, str) else data
        self._bucket.objects[self.name] = payload
        self._bucket.content_types[self.name] = content_type

    def download_as_text(self, encoding: str = "utf-8") -> str:
        if self.name not in self._bucket.objects:
            raise KeyError(f"404 no such object: {self.name}")
        return self._bucket.objects[self.name].decode(encoding)


class FakeBucket:
    def __init__(self, name: str):
        self.name = name
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str | None] = {}

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)


class FakeGCSClient:
    def __init__(self) -> None:
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, name: str) -> FakeBucket:
        return self.buckets.setdefault(name, FakeBucket(name))

    def list_blobs(self, bucket_or_name: Any, prefix: str = "") -> list[FakeBlob]:
        name = bucket_or_name if isinstance(bucket_or_name, str) else bucket_or_name.name
        bucket = self.bucket(name)
        return [bucket.blob(key) for key in sorted(bucket.objects) if key.startswith(prefix)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_environment():
    """Undo every environment change a test makes, including dotenv's.

    `monkeypatch` can only restore what it set. `load_dotenv` writes straight
    into `os.environ`, so a test that exercises env-file loading leaks its
    fixture values into every test that runs after it. That is how a temp
    `STORE_ROOT=./project-data` ended up silently redirecting the Phase 4 gate
    away from the real store, which made it skip in the full suite and pass
    when run alone.
    """
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture
def pack():
    return load_pack("medical_v1", pack_dir=PACK_DIR)


@pytest.fixture
def goldset_rows() -> list[dict[str, Any]]:
    text = GOLDSET_PATH.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.fixture
def local_store(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / "data")


@pytest.fixture
def gcs_store() -> GCSStore:
    return GCSStore("gs://cleanroom-test/dev", client=FakeGCSClient())


@pytest.fixture(params=["local", "gcs"])
def store(request, tmp_path: Path):
    """Every store test runs against both backends. That is the Phase 1 gate."""
    if request.param == "local":
        return LocalStore(tmp_path / "data")
    return GCSStore("gs://cleanroom-test/dev", client=FakeGCSClient())


@pytest.fixture
def local_settings(tmp_path: Path) -> Settings:
    return Settings(
        store_backend="local",
        store_root=str(tmp_path / "data"),
        pack_dir=PACK_DIR,
    )
