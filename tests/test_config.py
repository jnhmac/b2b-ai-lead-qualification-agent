"""Pack loading, pack validation, and environment settings."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from cleanroom import config
from cleanroom.config import (
    SIGNAL_LISTS,
    ConfigError,
    PackValidationError,
    Weight,
    load_pack,
    load_settings,
)
from tests.conftest import PACK_DIR


# ---------------------------------------------------------------------------
# The provided pack
# ---------------------------------------------------------------------------


def test_medical_v1_loads(pack):
    assert pack.version == "medical_v1"
    assert pack.sector == "medical_devices"
    assert pack.total_page_cap == 10


@pytest.mark.parametrize("list_name", SIGNAL_LISTS)
def test_every_signal_list_is_populated(pack, list_name):
    assert len(pack.signal_list(list_name)) > 0


def test_page_types_are_complete(pack):
    assert pack.always_fetch_type == "homepage"
    assert [name for name, _ in pack.page_types_by_priority][0] == "homepage"
    # Investors sorts last: it only exists on SEC filers, so it must never win
    # a slot away from a page type every domain has a chance of matching.
    assert pack.page_types_by_priority[-1][0] == "investors"
    assert pack.page_types["careers"].priority > pack.page_types["quality"].priority


def test_caps_over_allocate_on_purpose(pack):
    """Not a bug. Over-allocation lets the scorer fill slots from what exists."""
    assert sum(spec.max_pages for spec in pack.page_types.values()) > pack.total_page_cap


def test_trap_terms_are_neutral_not_negative(pack):
    """Trap terms are discounted by the classifier, never subtracted.

    Their home is `trap_terms`, and `ignore` exists nowhere else, so no scoring
    list can pick them up with a direction attached.
    """
    trap_forms = {form.lower() for term in pack.trap_terms for form in term.surface_forms}
    assert "iso 13485" in trap_forms

    for list_name in SIGNAL_LISTS:
        for term in pack.signal_list(list_name):
            if term.weight is Weight.ignore:
                assert list_name == "trap_terms"


def test_aliases_are_reachable(pack):
    cgm = next(t for t in pack.positive_products if t.term == "continuous glucose monitors")
    assert "CGM" in cgm.surface_forms
    assert cgm.surface_forms[0] == cgm.term


def test_fda_cross_check_is_present(pack):
    assert pack.fda_operation_types is not None
    assert "Contract Manufacturer" in pack.fda_operation_types.supports_outsourced
    assert "Relabeler" in pack.fda_operation_types.ignore


def test_summary(pack):
    summary = pack.summary()
    assert summary["version"] == "medical_v1"
    # 10 since `investors` was added 2026-08-24 for SEC-filer facility disclosures.
    assert summary["page_types"] == 10
    assert all(summary[name] > 0 for name in SIGNAL_LISTS)


# ---------------------------------------------------------------------------
# A malformed pack must fail loud
# ---------------------------------------------------------------------------


def _write_pack(tmp_path: Path, mutate=None, name: str = "medical_v1") -> Path:
    data = yaml.safe_load((PACK_DIR / "medical_v1.yaml").read_text(encoding="utf-8"))
    if mutate:
        mutate(data)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_missing_pack_raises(tmp_path):
    with pytest.raises(PackValidationError, match="not found"):
        load_pack("no_such_pack", pack_dir=tmp_path)


def test_invalid_yaml_raises(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("version: [unclosed\n", encoding="utf-8")
    with pytest.raises(PackValidationError, match="not valid YAML"):
        load_pack(path)


def test_non_mapping_yaml_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(PackValidationError, match="mapping"):
        load_pack(path)


def test_unknown_key_raises(tmp_path):
    """A typo'd key is a silent no-op under permissive parsing. Not here."""
    path = _write_pack(tmp_path, lambda d: d.update(totl_page_cap=10))
    with pytest.raises(PackValidationError, match="totl_page_cap"):
        load_pack(path)


def test_typo_inside_a_page_type_raises(tmp_path):
    path = _write_pack(tmp_path, lambda d: d["page_types"]["quality"].update(max_page=3))
    with pytest.raises(PackValidationError, match="max_page"):
        load_pack(path)


def test_empty_signal_list_raises(tmp_path):
    path = _write_pack(tmp_path, lambda d: d.update(trap_terms=[]))
    with pytest.raises(PackValidationError, match="trap_terms"):
        load_pack(path)


def test_unknown_weight_raises(tmp_path):
    path = _write_pack(tmp_path, lambda d: d["evidence_terms"][0].update(weight="huge"))
    with pytest.raises(PackValidationError, match="weight"):
        load_pack(path)


def test_ignore_weight_outside_trap_terms_raises(tmp_path):
    """`ignore` on a scoring list would make a live signal vanish silently."""
    path = _write_pack(tmp_path, lambda d: d["evidence_terms"][0].update(weight="ignore"))
    with pytest.raises(PackValidationError, match="reserved for trap_terms"):
        load_pack(path)


def test_term_in_two_lists_raises(tmp_path):
    """Direction comes from list membership, so two homes means no direction."""

    def mutate(data):
        data["negative_products"].append({"term": "catheters", "weight": "strong"})

    path = _write_pack(tmp_path, mutate)
    with pytest.raises(PackValidationError, match="exactly one list"):
        load_pack(path)


def test_alias_colliding_across_lists_raises(tmp_path):
    def mutate(data):
        data["negative_products"][0]["aliases"].append("Cleanroom")

    path = _write_pack(tmp_path, mutate)
    with pytest.raises(PackValidationError, match="exactly one list"):
        load_pack(path)


def test_duplicate_priorities_raise(tmp_path):
    path = _write_pack(tmp_path, lambda d: d["page_types"]["quality"].update(priority=1))
    with pytest.raises(PackValidationError, match="priorities must be unique"):
        load_pack(path)


def test_no_always_fetch_type_raises(tmp_path):
    path = _write_pack(tmp_path, lambda d: d["page_types"]["homepage"].update(always_fetch=False))
    with pytest.raises(PackValidationError, match="always_fetch"):
        load_pack(path)


def test_patternless_page_type_raises(tmp_path):
    path = _write_pack(tmp_path, lambda d: d["page_types"]["quality"].update(patterns=[]))
    with pytest.raises(PackValidationError, match="at least one pattern"):
        load_pack(path)


def test_version_must_match_filename_when_loaded_by_name(tmp_path):
    _write_pack(tmp_path, lambda d: d.update(version="medical_v2"), name="medical_v1")
    with pytest.raises(PackValidationError, match="wrong pack"):
        load_pack("medical_v1", pack_dir=tmp_path)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_defaults():
    settings = load_settings(env={})
    assert settings.store_backend == "local"
    assert settings.store_root == "./data"
    assert settings.firecrawl_api_key is None


def test_keys_are_read_and_unquoted():
    settings = load_settings(env={"ANTHROPIC_API_KEY": '"sk-test"  '})
    assert settings.anthropic_api_key == "sk-test"
    assert settings.require_anthropic_api_key() == "sk-test"


@pytest.mark.parametrize(
    ("accessor", "name"),
    [
        ("require_firecrawl_api_key", "FIRECRAWL_API_KEY"),
        ("require_anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("require_openrouter_api_key", "OPENROUTER_API_KEY"),
        ("require_apollo_api_key", "APOLLO_API_KEY"),
    ],
)
def test_missing_key_names_the_variable(accessor, name):
    settings = load_settings(env={})
    with pytest.raises(ConfigError, match=name):
        getattr(settings, accessor)()


def test_project_scoped_keys_are_read():
    settings = load_settings(
        env={"OPENROUTER_API_KEY": "sk-or-project", "APOLLO_API_KEY": "apollo-1"}
    )
    assert settings.require_openrouter_api_key() == "sk-or-project"
    assert settings.require_apollo_api_key() == "apollo-1"


def test_project_env_wins_over_hub_env(tmp_path, monkeypatch):
    """The project OpenRouter key bills separately, so hub must not clobber it."""
    project_env = tmp_path / "project" / ".env"
    project_env.parent.mkdir()
    project_env.write_text(
        "OPENROUTER_API_KEY=sk-or-project\nSTORE_ROOT=./project-data\n", encoding="utf-8"
    )
    hub_env = tmp_path / "hub" / ".env"
    hub_env.parent.mkdir()
    hub_env.write_text(
        "OPENROUTER_API_KEY=sk-or-hub\nFIRECRAWL_API_KEY=fc-hub\n", encoding="utf-8"
    )

    for var in ("OPENROUTER_API_KEY", "FIRECRAWL_API_KEY", "STORE_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "HUB_ENV_PATH", hub_env)
    monkeypatch.chdir(project_env.parent)

    loaded = config.load_env_files()
    assert loaded == [project_env, hub_env]

    settings = load_settings()
    assert settings.openrouter_api_key == "sk-or-project"  # project wins
    assert settings.firecrawl_api_key == "fc-hub"  # hub fills the gap
    assert settings.store_root == "./project-data"


def test_a_missing_hub_env_is_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HUB_ENV_PATH", tmp_path / "nope" / ".env")
    monkeypatch.chdir(tmp_path)
    assert config.load_env_files() == []


def test_gcs_backend_requires_a_gs_root():
    with pytest.raises(ConfigError, match="gs:// URI"):
        load_settings(env={"STORE_BACKEND": "gcs", "STORE_ROOT": "./data"})


def test_local_backend_rejects_a_gs_root():
    with pytest.raises(ConfigError, match="cannot use a gs://"):
        load_settings(env={"STORE_BACKEND": "local", "STORE_ROOT": "gs://bucket"})


def test_unknown_backend_raises():
    with pytest.raises(ConfigError, match="must be 'local' or 'gcs'"):
        load_settings(env={"STORE_BACKEND": "s3"})


def test_gcs_settings_accepted():
    settings = load_settings(env={"STORE_BACKEND": "GCS", "STORE_ROOT": "gs://bucket/prefix"})
    assert settings.store_backend == "gcs"
    assert settings.store_root == "gs://bucket/prefix"
