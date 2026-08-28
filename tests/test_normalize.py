"""Registrable domain extraction, dedupe, and collision reporting."""

from __future__ import annotations

import pytest

from cleanroom.normalize import (
    InvalidDomainError,
    normalize_all,
    normalize_domain,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("example.com", "example.com"),
        ("https://www.example.com/", "example.com"),
        ("http://example.com", "example.com"),
        ("HTTPS://WWW.Example.COM/About-Us", "example.com"),
        ("  https://example.com/products?page=2#top  ", "example.com"),
        ("www.example.com", "example.com"),
        ("example.com.", "example.com"),
        ("example.com:8443", "example.com"),
        ("https://user:pw@example.com:8443/x", "example.com"),
        ("<https://example.com>", "example.com"),
        ('"example.com"', "example.com"),
        # Multi-part public suffixes must not be split at the last dot.
        ("https://shop.example.co.uk/a/b", "example.co.uk"),
        ("news.bbc.co.uk", "bbc.co.uk"),
        # Hyphens and digits, both present in the goldset.
        ("https://www.rpq-medical.com/", "rpq-medical.com"),
        ("qt7software.com", "qt7software.com"),
        # Deep subdomains collapse to the registrable domain.
        ("https://emea.diabetes.meditronex.com/en/home", "meditronex.com"),
    ],
)
def test_normalize_domain(raw, expected):
    assert normalize_domain(raw).domain == expected


def test_normalization_is_idempotent():
    once = normalize_domain("https://www.Example.com/path/").domain
    assert normalize_domain(once).domain == once


def test_host_keeps_subdomain_but_drops_www():
    entry = normalize_domain("https://www.diabetes.meditronex.com/")
    assert entry.domain == "meditronex.com"
    assert entry.host == "diabetes.meditronex.com"
    assert entry.subdomain == "www.diabetes"
    assert not entry.is_bare_registrable

    plain = normalize_domain("https://www.meditronex.com")
    assert plain.host == "meditronex.com"
    assert plain.is_bare_registrable


def test_unicode_domain_is_punycoded_not_dropped():
    # Non-English sites stay in the population. Only the encoding is normalised.
    assert normalize_domain("https://bücher.de/katalog").domain == "xn--bcher-kva.de"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not a domain",
        "localhost",
        "http://192.168.0.1/",
        "192.168.0.1",
        "http://[2001:db8::1]/",
        "mailto:info@example.com",
        "ftp://files.example.com",
        "https://example.invalidtld",
        "example.com:notaport",
    ],
)
def test_invalid_inputs_raise(raw):
    with pytest.raises(InvalidDomainError):
        normalize_domain(raw)


def test_non_string_input_raises():
    with pytest.raises(InvalidDomainError):
        normalize_domain(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Batch behaviour
# ---------------------------------------------------------------------------


def test_dedupe_preserves_first_seen_order():
    result = normalize_all(
        ["https://b.com", "a.com", "https://www.b.com/", "c.com"]
    )
    assert result.domains == ["b.com", "a.com", "c.com"]


def test_repeated_input_is_a_plain_duplicate():
    result = normalize_all(["example.com", "https://www.example.com/"])
    assert result.domains == ["example.com"]
    assert len(result.collisions) == 1
    assert result.collisions[0].kind == "duplicate"
    assert result.subdomain_collapses == []


def test_parent_subsidiary_collapse_is_reported_not_hidden(caplog):
    """The spec's watch item: log the collapse, do not merge silently."""
    with caplog.at_level("WARNING", logger="cleanroom.normalize"):
        result = normalize_all(
            ["https://meditronex.com", "https://diabetes.meditronex.com/products"]
        )

    assert result.domains == ["meditronex.com"]
    assert len(result.subdomain_collapses) == 1

    collision = result.subdomain_collapses[0]
    assert collision.domain == "meditronex.com"
    assert collision.hosts == ["diabetes.meditronex.com", "meditronex.com"]
    assert len(collision.inputs) == 2
    assert "meditronex.com" in caplog.text
    assert "parent/subsidiary" in collision.describe().lower()


def test_invalid_rows_are_collected_not_fatal():
    result = normalize_all(["good.com", "not a domain", "also-good.com"])
    assert result.domains == ["good.com", "also-good.com"]
    assert len(result.invalid) == 1
    assert result.invalid[0][0] == "not a domain"


def test_summary_counts():
    result = normalize_all(["a.com", "www.a.com", "sub.a.com", "bad input"])
    assert result.summary() == {
        "unique_domains": 1,
        "collisions": 1,
        "subdomain_collapses": 1,
        "invalid": 1,
    }
