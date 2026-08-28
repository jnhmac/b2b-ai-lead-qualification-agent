"""Pre-crawl warnings read off the domain list's own optional columns.

`cli.py` has no test module of its own; these cover only the new pure function
and its printer, matching the house pattern of testing logic and leaving
`main()` to manual verification.

The point of the feature: `thehamlincompany.com` reached the classifier, was
crawled, extracted, classified and reviewed, and is a Boston real-estate firm
that was never the intended company. Every signal needed to catch it before a
single credit was spent sat in `outputs/in_universe.csv` — `match_basis='name'`,
`industry='real estate'`, `listed_website` pointing elsewhere. 84 of 433 domains
(19.4%) share that shape. See outputs/2026-08-26_design-review.md F8.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cleanroom.cli import DomainWarnings, read_domain_warnings  # noqa: E402


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestReadDomainWarnings:
    def test_reads_both_flag_columns(self, tmp_path):
        path = write(tmp_path, "pool.csv", (
            "domain,company,identity_risk,junk_sector\n"
            "good.com,Good Medical,no,no\n"
            "risky.com,Risky Inc,yes,no\n"
            "junk.com,Junk Ltd,no,yes\n"
            "both.com,Both Co,yes,yes\n"
        ))
        warnings = read_domain_warnings(path)
        assert warnings.identity_risk == ["both.com", "risky.com"]
        assert warnings.junk_sector == ["both.com", "junk.com"]
        assert warnings.total == 3  # both.com counted once

    def test_a_list_without_the_columns_warns_about_nothing(self, tmp_path):
        """The 2026-08-11 batch files, the goldset, a hand-written scratch list.
        Absence of the columns is not a claim that the domains are clean."""
        path = write(tmp_path, "plain.csv", "domain,company\nacme.com,Acme\n")
        warnings = read_domain_warnings(path)
        assert warnings.total == 0
        assert warnings.columns_present is False

    def test_records_that_the_columns_were_present_even_when_all_clean(self, tmp_path):
        path = write(tmp_path, "clean.csv", "domain,identity_risk\nacme.com,no\n")
        warnings = read_domain_warnings(path)
        assert warnings.columns_present is True
        assert warnings.total == 0

    def test_non_csv_inputs_are_silently_unwarned(self, tmp_path):
        """JSONL and plain text carry no columns to read. Returning empty rather
        than raising keeps `crawl --domains goldset.jsonl` working unchanged."""
        for name, text in (("x.jsonl", '{"domain": "acme.com"}\n'),
                           ("x.txt", "acme.com\n")):
            warnings = read_domain_warnings(write(tmp_path, name, text))
            assert warnings.total == 0
            assert warnings.columns_present is False

    def test_a_missing_file_is_not_an_error_here(self, tmp_path):
        """read_domains already raises ConfigError on a missing list, with a
        better message. This must not pre-empt it with a worse one."""
        warnings = read_domain_warnings(tmp_path / "nope.csv")
        assert warnings.total == 0

    def test_truthiness_is_permissive_about_spelling(self, tmp_path):
        path = write(tmp_path, "pool.csv", (
            "domain,identity_risk\n"
            "a.com,yes\n"
            "b.com,YES\n"
            "c.com,true\n"
            "d.com,1\n"
            "e.com,no\n"
            "f.com,\n"
        ))
        assert read_domain_warnings(path).identity_risk == [
            "a.com", "b.com", "c.com", "d.com"
        ]

    def test_rows_with_no_domain_are_skipped(self, tmp_path):
        path = write(tmp_path, "pool.csv", "domain,identity_risk\n,yes\nok.com,yes\n")
        assert read_domain_warnings(path).identity_risk == ["ok.com"]


class TestDescribe:
    def test_describe_is_empty_when_there_is_nothing_to_say(self):
        assert DomainWarnings().describe() == []

    def test_describe_names_the_domains_and_says_it_is_not_a_filter(self):
        warnings = DomainWarnings(
            identity_risk=["a.com", "b.com"], junk_sector=["c.com"],
            columns_present=True,
        )
        text = "\n".join(warnings.describe())
        assert "2" in text and "a.com" in text and "b.com" in text
        assert "c.com" in text
        # BUILD.md locks "no exclusion filters" -- this warns, it never skips.
        assert "not skipped" in text.lower() or "warning" in text.lower()

    def test_describe_truncates_a_long_list(self):
        warnings = DomainWarnings(
            identity_risk=[f"d{i}.com" for i in range(40)], columns_present=True,
        )
        text = "\n".join(warnings.describe())
        assert "40" in text
        assert "..." in text
