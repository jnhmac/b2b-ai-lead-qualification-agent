"""Input domain cleanup: registrable domain, redirect resolution, dedupe.

The input list arrives as a mix of full URLs and bare domains, with the usual
spreadsheet damage: trailing slashes, stray whitespace, mixed case, the odd
`http://` on a site that has been https for a decade. Everything downstream
keys on the registrable domain, so it has to be derived once, here, and be
stable across runs.

Two things this deliberately does not do:

It does not skip anything. Enterprise multi-division companies and non-English
sites normalize like everything else and get flagged later, at classification
time. Filtering here would remove them from the population silently.

It does not silently merge parent and subsidiary. `meditronex.com` and
`diabetes.meditronex.com` share a registrable domain and will dedupe to one
entry, but that collapse is reported and logged rather than assumed correct.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import httpx
import tldextract

log = logging.getLogger(__name__)

# suffix_list_urls=() pins the extractor to the snapshot bundled with the
# installed tldextract, so normalization is offline, deterministic, and cannot
# change under a running batch because the upstream PSL was updated mid-crawl.
#
# include_psl_private_domains=True keeps a company hosted on a platform
# subdomain (`acme.github.io`) from collapsing into the platform itself.
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(),
    cache_dir=None,
    include_psl_private_domains=True,
)

_SCHEME_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*)://")
_BARE_SCHEME_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*):(?!\d)")
_ALLOWED_SCHEMES = frozenset({"http", "https"})

DEFAULT_USER_AGENT = "cleanroom-classifier/0.1 (+domain normalization)"

CollisionKind = Literal["duplicate", "subdomain_collapse"]


class InvalidDomainError(ValueError):
    """The input could not be reduced to a registrable domain."""


@dataclass(frozen=True)
class NormalizedDomain:
    """One input, reduced.

    `host` keeps what was actually given (minus scheme, port, path and www) so
    a subdomain collapse can be explained after the fact. `domain` is the key
    everything downstream uses.
    """

    raw: str
    domain: str
    host: str
    subdomain: str

    @property
    def is_bare_registrable(self) -> bool:
        """True when the input pointed at the registrable domain itself."""
        return self.host == self.domain


@dataclass(frozen=True)
class Collision:
    """Two or more inputs that reduced to the same registrable domain."""

    domain: str
    kind: CollisionKind
    inputs: list[str]
    hosts: list[str]

    def describe(self) -> str:
        if self.kind == "subdomain_collapse":
            return (
                f"{self.domain}: {len(self.hosts)} distinct hosts collapsed "
                f"({', '.join(self.hosts)}) from inputs {self.inputs}. "
                "Possible parent/subsidiary, not merged blindly."
            )
        return f"{self.domain}: repeated input {self.inputs}"


@dataclass
class NormalizeResult:
    """The deduped domain list plus everything that went wrong getting there."""

    entries: list[NormalizedDomain] = field(default_factory=list)
    collisions: list[Collision] = field(default_factory=list)
    invalid: list[tuple[str, str]] = field(default_factory=list)

    @property
    def domains(self) -> list[str]:
        return [entry.domain for entry in self.entries]

    @property
    def subdomain_collapses(self) -> list[Collision]:
        """The collisions that need a human, as opposed to plain duplicates."""
        return [c for c in self.collisions if c.kind == "subdomain_collapse"]

    def summary(self) -> dict[str, int]:
        return {
            "unique_domains": len(self.entries),
            "collisions": len(self.collisions),
            "subdomain_collapses": len(self.subdomain_collapses),
            "invalid": len(self.invalid),
        }


# ---------------------------------------------------------------------------
# Single-value normalization
# ---------------------------------------------------------------------------


def _strip_wrappers(raw: str) -> str:
    text = raw.strip()
    while text and text[0] in "<\"'" and text[-1] in ">\"'":
        text = text[1:-1].strip()
    return text


def _host_from(text: str) -> str:
    """Reduce a URL or bare domain to its hostname."""
    match = _SCHEME_RE.match(text)
    if match:
        scheme = match.group("scheme").lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise InvalidDomainError(f"unsupported scheme {scheme!r}")
        text = text[match.end() :]
    elif bare := _BARE_SCHEME_RE.match(text):
        # `mailto:`, `tel:` and friends. The negative lookahead on a digit keeps
        # `example.com:8080` out of this branch.
        raise InvalidDomainError(f"unsupported scheme {bare.group('scheme').lower()!r}")

    netloc = re.split(r"[/?#]", text, maxsplit=1)[0]
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    if not netloc:
        raise InvalidDomainError("no host found")

    if netloc.startswith("["):
        raise InvalidDomainError("IPv6 literal is not a domain")

    host, sep, port = netloc.rpartition(":")
    if sep:
        if not port.isdigit():
            raise InvalidDomainError(f"malformed port in {netloc!r}")
    else:
        host = netloc

    return host.strip().strip(".").lower()


def normalize_domain(raw: str) -> NormalizedDomain:
    """Reduce a URL or bare domain to its registrable domain.

    Raises `InvalidDomainError` rather than returning a sentinel: a domain that
    cannot be parsed is an input-data problem someone has to look at, and a
    silent empty string would crawl as an empty path.
    """
    if not isinstance(raw, str):
        raise InvalidDomainError(f"expected a string, got {type(raw).__name__}")

    text = _strip_wrappers(raw)
    if not text:
        raise InvalidDomainError("empty input")

    host = _host_from(text)
    if not host:
        raise InvalidDomainError("empty host")
    if " " in host:
        raise InvalidDomainError(f"host contains whitespace: {host!r}")

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise InvalidDomainError(f"IP literal is not a domain: {host!r}")

    try:
        # Punycode so a non-Latin domain has one canonical spelling. Non-English
        # sites stay in the population; only their encoding is normalised.
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        # Some valid hosts fail the strict codec (over-long labels, mixed
        # scripts). tldextract still handles them, so keep going with unicode.
        log.debug("IDNA encoding failed for %r, keeping unicode form", host)

    extracted = _EXTRACT(host)
    if not extracted.suffix:
        raise InvalidDomainError(f"no public suffix in {host!r}")
    if not extracted.domain:
        raise InvalidDomainError(f"no registrable name in {host!r}")

    registrable = extracted.top_domain_under_public_suffix
    if not registrable:
        raise InvalidDomainError(f"could not derive a registrable domain from {host!r}")

    return NormalizedDomain(
        raw=raw,
        domain=registrable,
        host=_strip_www(host),
        subdomain=extracted.subdomain,
    )


def _strip_www(host: str) -> str:
    """`www` is a convention, not a subsidiary. Drop it before comparing hosts."""
    return host[4:] if host.startswith("www.") else host


# ---------------------------------------------------------------------------
# Batch normalization and dedupe
# ---------------------------------------------------------------------------


def normalize_all(raws: Iterable[str], *, warn_on_collision: bool = True) -> NormalizeResult:
    """Normalize and dedupe a list, preserving first-seen order.

    Collisions are reported, never hidden. A repeated input is noise; two
    distinct hosts collapsing onto one registrable domain is the parent /
    subsidiary case the spec calls out, and it is logged at warning level
    because merging those two crawls into one verdict may be wrong.
    """
    result = NormalizeResult()
    seen: OrderedDict[str, NormalizedDomain] = OrderedDict()
    grouped: dict[str, list[NormalizedDomain]] = defaultdict(list)

    for raw in raws:
        try:
            entry = normalize_domain(raw)
        except InvalidDomainError as exc:
            result.invalid.append((raw, str(exc)))
            log.warning("dropping unparseable input %r: %s", raw, exc)
            continue

        grouped[entry.domain].append(entry)
        seen.setdefault(entry.domain, entry)

    result.entries = list(seen.values())

    for domain, entries in grouped.items():
        if len(entries) == 1:
            continue
        hosts = sorted({entry.host for entry in entries})
        inputs = [entry.raw for entry in entries]
        kind: CollisionKind = "subdomain_collapse" if len(hosts) > 1 else "duplicate"
        collision = Collision(domain=domain, kind=kind, inputs=inputs, hosts=hosts)
        result.collisions.append(collision)
        if kind == "subdomain_collapse" and warn_on_collision:
            log.warning("domain collision: %s", collision.describe())
        else:
            log.debug("domain collision: %s", collision.describe())

    result.collisions.sort(key=lambda c: (c.kind, c.domain))
    return result


# ---------------------------------------------------------------------------
# Redirect resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedirectResult:
    """Where a domain actually lands after following redirects."""

    source: str
    domain: str
    final_url: str | None = None
    status: int | None = None
    error: str | None = None

    @property
    def changed(self) -> bool:
        return self.domain != self.source

    @property
    def ok(self) -> bool:
        return self.error is None


def resolve_redirect(
    domain: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 10.0,
    scheme: str = "https",
) -> RedirectResult:
    """Follow redirects and report the registrable domain actually served.

    Rebrands and acquisitions show up here: `oldname.com` 301s to `newname.com`
    and the two must not be crawled as separate companies. A failure is
    returned, not raised, so one dead host cannot abort a batch, and the source
    domain is kept as the answer so the pipeline still has something to crawl.
    """
    owned = client is None
    if client is None:
        client = httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )

    url = f"{scheme}://{domain}"
    try:
        try:
            response = client.head(url)
            # Plenty of sites answer HEAD with 403/405 while serving GET fine.
            if response.status_code >= 400:
                response = client.get(url)
        except httpx.HTTPError:
            response = client.get(url)

        final_url = str(response.url)
        try:
            landed = normalize_domain(final_url).domain
        except InvalidDomainError as exc:
            return RedirectResult(
                source=domain, domain=domain, final_url=final_url,
                status=response.status_code, error=f"unparseable final URL: {exc}",
            )

        if landed != domain:
            log.info("redirect: %s -> %s (%s)", domain, landed, response.status_code)
        return RedirectResult(
            source=domain, domain=landed, final_url=final_url, status=response.status_code
        )
    except httpx.HTTPError as exc:
        log.warning("redirect resolution failed for %s: %s", domain, exc)
        return RedirectResult(source=domain, domain=domain, error=f"{type(exc).__name__}: {exc}")
    finally:
        if owned:
            client.close()


def resolve_redirects(
    domains: Sequence[str],
    *,
    timeout: float = 10.0,
    max_workers: int = 8,
    scheme: str = "https",
) -> list[RedirectResult]:
    """Resolve a batch over a shared connection pool, preserving input order."""
    if not domains:
        return []

    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    ) as client:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(
                pool.map(
                    lambda domain: resolve_redirect(domain, client=client, scheme=scheme),
                    domains,
                )
            )
