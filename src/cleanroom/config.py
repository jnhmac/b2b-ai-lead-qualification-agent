"""Environment settings and sector-pack loading.

The pack is config, not code. That only holds if a malformed pack fails at load
time with a message naming the offending entry, rather than degrading quietly
into a classifier that scores nothing. Every model here sets
``extra="forbid"``: a typo'd key such as ``max_page`` is a silent no-op under
permissive parsing, and a silent no-op in the discovery caps is invisible until
the crawl bill arrives.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_DIR = REPO_ROOT / "packs"

StoreBackend = Literal["local", "gcs"]


class ConfigError(RuntimeError):
    """Bad environment or unusable configuration."""


class PackValidationError(ConfigError):
    """A sector pack failed to load or failed validation."""


# ---------------------------------------------------------------------------
# Pack schema
# ---------------------------------------------------------------------------


class Weight(str, Enum):
    """Signal *reliability*, not direction.

    Direction comes from which list a term sits in: a `strong` negative product
    pushes toward `none` exactly as hard as a `strong` positive one pushes
    toward `in_house`. `ignore` is reserved for trap terms.
    """

    strong = "strong"
    medium = "medium"
    weak = "weak"
    ignore = "ignore"


class Term(BaseModel):
    """One vocabulary entry: a surface form, its aliases, and its reliability."""

    model_config = ConfigDict(extra="forbid")

    term: str = Field(min_length=1)
    weight: Weight
    aliases: list[str] = Field(default_factory=list)
    note: str | None = None

    @property
    def surface_forms(self) -> list[str]:
        """The term plus its aliases: everything that counts as a match."""
        return [self.term, *self.aliases]


class PageTypeSpec(BaseModel):
    """A discovery target: how many pages of this kind, and how to spot them.

    `patterns` are matched against both URL path and anchor text. A page at
    `/company/heritage` matches no path pattern, but a nav link reading "Who We
    Are" catches it.
    """

    model_config = ConfigDict(extra="forbid")

    priority: int = Field(ge=0, description="lower number wins when competing for slots")
    max_pages: int = Field(ge=1)
    always_fetch: bool = False
    patterns: list[str] = Field(default_factory=list)


class FdaOperationTypes(BaseModel):
    """openFDA establishment operation types, used as a cross-check only."""

    model_config = ConfigDict(extra="forbid")

    supports_outsourced: list[str] = Field(default_factory=list)
    ignore: list[str] = Field(default_factory=list)


class SectorPack(BaseModel):
    """A loaded, validated sector pack."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    total_page_cap: int = Field(ge=1)

    page_types: dict[str, PageTypeSpec] = Field(min_length=1)

    positive_products: list[Term] = Field(min_length=1)
    negative_products: list[Term] = Field(min_length=1)
    evidence_terms: list[Term] = Field(min_length=1)
    trap_terms: list[Term] = Field(min_length=1)
    outsourcing_cues: list[Term] = Field(min_length=1)

    fda_operation_types: FdaOperationTypes | None = None

    # -- cross-field validation ---------------------------------------------

    @model_validator(mode="after")
    def _validate_page_types(self) -> SectorPack:
        always = [name for name, spec in self.page_types.items() if spec.always_fetch]
        if len(always) != 1:
            raise ValueError(
                "exactly one page type must set always_fetch: true "
                f"(the homepage); found {always or 'none'}"
            )

        by_priority: dict[int, list[str]] = defaultdict(list)
        for name, spec in self.page_types.items():
            by_priority[spec.priority].append(name)
        clashes = {p: sorted(names) for p, names in by_priority.items() if len(names) > 1}
        if clashes:
            raise ValueError(f"page_types priorities must be unique; duplicated: {clashes}")

        patternless = sorted(
            name
            for name, spec in self.page_types.items()
            if not spec.always_fetch and not spec.patterns
        )
        if patternless:
            raise ValueError(
                "page types without always_fetch need at least one pattern, "
                f"otherwise they can never be selected: {patternless}"
            )

        # Deliberately NOT validated: sum(max_pages) exceeding total_page_cap.
        # The caps over-allocate on purpose so the scorer can fill slots from
        # whichever page types a given site actually has.
        return self

    @model_validator(mode="after")
    def _validate_weights(self) -> SectorPack:
        """`ignore` may only appear on trap terms."""
        offenders = [
            f"{list_name}:{term.term}"
            for list_name in SIGNAL_LISTS
            if list_name != "trap_terms"
            for term in getattr(self, list_name)
            if term.weight is Weight.ignore
        ]
        if offenders:
            raise ValueError(
                "weight 'ignore' is reserved for trap_terms; a scoring list "
                f"cannot contain an ignored term: {offenders}"
            )
        return self

    @model_validator(mode="after")
    def _validate_no_ambiguous_terms(self) -> SectorPack:
        """A surface form may appear in only one list.

        Direction comes from list membership, so a term sitting in two lists
        has no defined direction. This is a correctness bug in the pack, not a
        style preference.
        """
        seen: dict[str, list[str]] = defaultdict(list)
        for list_name in SIGNAL_LISTS:
            for term in getattr(self, list_name):
                for form in term.surface_forms:
                    key = " ".join(form.lower().split())
                    seen[key].append(f"{list_name}:{term.term}")

        ambiguous = {form: places for form, places in seen.items() if len(places) > 1}
        if ambiguous:
            raise ValueError(
                "each surface form must belong to exactly one list, otherwise "
                f"its direction is undefined: {ambiguous}"
            )
        return self

    # -- convenience --------------------------------------------------------

    @property
    def always_fetch_type(self) -> str:
        """The page type fetched for every domain regardless of score."""
        return next(name for name, spec in self.page_types.items() if spec.always_fetch)

    @property
    def page_types_by_priority(self) -> list[tuple[str, PageTypeSpec]]:
        return sorted(self.page_types.items(), key=lambda item: item[1].priority)

    def signal_list(self, name: str) -> list[Term]:
        if name not in SIGNAL_LISTS:
            raise KeyError(f"unknown signal list {name!r}; expected one of {SIGNAL_LISTS}")
        return getattr(self, name)

    def summary(self) -> dict[str, Any]:
        """Counts per list, for logging what a run actually loaded."""
        return {
            "version": self.version,
            "sector": self.sector,
            "total_page_cap": self.total_page_cap,
            "page_types": len(self.page_types),
            **{name: len(getattr(self, name)) for name in SIGNAL_LISTS},
        }


SIGNAL_LISTS: tuple[str, ...] = (
    "positive_products",
    "negative_products",
    "evidence_terms",
    "trap_terms",
    "outsourcing_cues",
)


# ---------------------------------------------------------------------------
# Pack loading
# ---------------------------------------------------------------------------


def pack_path(name: str, pack_dir: Path | None = None) -> Path:
    """Resolve a pack name such as ``medical_v1`` to its YAML file."""
    directory = pack_dir or DEFAULT_PACK_DIR
    return directory / f"{name}.yaml"


def load_pack(name_or_path: str | Path, pack_dir: Path | None = None) -> SectorPack:
    """Load and validate a sector pack. Raises `PackValidationError` on anything wrong.

    Accepts a pack name (``medical_v1``) or a path to a YAML file. When loaded
    by name, the file's declared ``version`` must match the name, which catches
    the copy-a-pack-and-forget-to-rename mistake before it silently tags a run
    with the wrong pack version.
    """
    by_name = isinstance(name_or_path, str) and not str(name_or_path).endswith((".yaml", ".yml"))
    path = pack_path(str(name_or_path), pack_dir) if by_name else Path(name_or_path)

    if not path.is_file():
        raise PackValidationError(f"sector pack not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PackValidationError(f"{path}: not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise PackValidationError(f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}")

    try:
        pack = SectorPack.model_validate(raw)
    except ValidationError as exc:
        raise PackValidationError(f"{path}: pack failed validation:\n{exc}") from exc

    if by_name and pack.version != str(name_or_path):
        raise PackValidationError(
            f"{path}: pack declares version {pack.version!r} but is filed as "
            f"{name_or_path!r}; run outputs would be tagged with the wrong pack"
        )

    log.info("loaded sector pack %s", pack.summary())
    return pack


@lru_cache(maxsize=8)
def get_pack(name: str = "medical_v1") -> SectorPack:
    """Cached `load_pack` for the common case of one pack per process."""
    return load_pack(name)


# ---------------------------------------------------------------------------
# LLM providers and models
# ---------------------------------------------------------------------------
#
# Two providers serve the same models. OpenRouter is OpenAI-compatible and
# takes `response_format`; the Anthropic API takes `output_config.format`. The
# request shapes differ, the models do not.
#
# The seam that keeps the switch cheap lives in `llm.py`: the client varies the
# request, but always returns raw JSON text that pydantic validates. Nothing
# above `llm.py` learns which provider ran, so moving the full run onto the
# Anthropic API is a `LLM_PROVIDER` change, not a rewrite.
#
# Model IDs are never inlined at a call site. The same model is spelled
# differently per provider, and a hardcoded slug is a silent coupling.

LLMProvider = Literal["openrouter", "anthropic"]
LLMStage = Literal["extract", "classify"]

LLM_STAGES: tuple[LLMStage, ...] = ("extract", "classify")

PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com",
}

PROVIDER_KEY_VARS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@dataclass(frozen=True)
class ModelSpec:
    """One model, its per-provider spelling, and what it costs to run.

    Prices are USD per million tokens, verified against OpenRouter's model
    catalogue on 2026-08-06. They exist here so a run can log its own spend
    without a second API call.
    """

    stage: LLMStage
    openrouter_id: str
    anthropic_id: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_read_usd_per_mtok: float
    cache_write_usd_per_mtok: float
    max_output_tokens: int
    effort: str | None = None
    # Prompt caching silently does nothing below this many tokens: no error,
    # just cache_read_tokens stuck at 0. It is not uniform across models, and
    # Haiku 4.5's threshold is the highest of the two used here, so the
    # extraction system prompt has to clear 4096 tokens before the pack prefix
    # is worth caching at all.
    cache_min_tokens: int = 1024
    note: str | None = None

    def id_for(self, provider: LLMProvider, *, batch: bool = False) -> str:
        if provider == "openrouter":
            # OpenRouter exposes the 50%-off batch tier as a slug suffix.
            return f"{self.openrouter_id}:batch" if batch else self.openrouter_id
        return self.anthropic_id

    def cost_usd(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        *,
        batch: bool = False,
    ) -> float:
        total = (
            input_tokens * self.input_usd_per_mtok
            + output_tokens * self.output_usd_per_mtok
            + cache_read_tokens * self.cache_read_usd_per_mtok
            + cache_write_tokens * self.cache_write_usd_per_mtok
        ) / 1_000_000
        return total * 0.5 if batch else total


# Haiku 4.5 extracts (reports what a page says, no judgment), Sonnet 5
# classifies (all judgment lives here). Sonnet 5 input/output are introductory
# rates through 2026-08-31, reverting to 3.00 / 15.00.
STAGE_MODELS: dict[LLMStage, ModelSpec] = {
    "extract": ModelSpec(
        stage="extract",
        openrouter_id="anthropic/claude-haiku-4.5",
        anthropic_id="claude-haiku-4-5",
        input_usd_per_mtok=1.00,
        output_usd_per_mtok=5.00,
        cache_read_usd_per_mtok=0.10,
        cache_write_usd_per_mtok=1.25,
        max_output_tokens=8_000,
        effort=None,
        cache_min_tokens=4_096,
        note="Haiku 4.5 has no effort parameter; sending one errors.",
    ),
    "classify": ModelSpec(
        stage="classify",
        openrouter_id="anthropic/claude-sonnet-5",
        anthropic_id="claude-sonnet-5",
        input_usd_per_mtok=2.00,
        output_usd_per_mtok=10.00,
        cache_read_usd_per_mtok=0.20,
        cache_write_usd_per_mtok=2.50,
        # Sonnet 5 thinks by default and max_tokens caps thinking plus output
        # together, so this is sized well above the record itself.
        max_output_tokens=16_000,
        effort="medium",
        cache_min_tokens=1_024,
        note="Introductory pricing through 2026-08-31, then 3.00 / 15.00.",
    ),
}


@dataclass(frozen=True)
class LLMSettings:
    """Which provider runs the calls, and how strictly the schema is requested.

    `enforce_schema` is request-side only. Turning it off changes what is sent,
    never how the response is parsed, so the pydantic validation path stays
    identical on both providers and remains the single place a malformed
    response is caught.
    """

    provider: LLMProvider
    base_url: str
    api_key: str | None
    enforce_schema: bool = True
    use_batch: bool = False
    max_attempts: int = 3
    timeout: float = 120.0
    models: dict[str, ModelSpec] = field(default_factory=lambda: dict(STAGE_MODELS))

    @property
    def key_var(self) -> str:
        return PROVIDER_KEY_VARS[self.provider]

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                f"{self.key_var} is not set, and LLM_PROVIDER={self.provider} needs it"
            )
        return self.api_key

    def spec(self, stage: LLMStage) -> ModelSpec:
        try:
            return self.models[stage]
        except KeyError:
            raise ConfigError(
                f"unknown LLM stage {stage!r}; expected one of {LLM_STAGES}"
            ) from None

    def model_id(self, stage: LLMStage) -> str:
        """The provider-specific model string for this stage."""
        return self.spec(stage).id_for(self.provider, batch=self.use_batch)

    def summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "batch": self.use_batch,
            "enforce_schema": self.enforce_schema,
            "key_present": bool(self.api_key),
            **{stage: self.model_id(stage) for stage in self.models},
        }


def load_llm_settings(env: dict[str, str]) -> LLMSettings:
    """Build LLM settings, validating the provider and its key variable."""
    provider = (_clean(env.get("LLM_PROVIDER")) or "openrouter").lower()
    if provider not in PROVIDER_BASE_URLS:
        raise ConfigError(
            f"LLM_PROVIDER must be one of {sorted(PROVIDER_BASE_URLS)}, got {provider!r}"
        )

    base_url = (_clean(env.get("LLM_BASE_URL")) or PROVIDER_BASE_URLS[provider]).rstrip("/")
    use_batch = _flag(env.get("LLM_USE_BATCH"), default=False)

    if use_batch and provider == "anthropic":
        # Anthropic's batch tier is a different endpoint (/v1/messages/batches)
        # with submit-and-poll semantics, not a model-ID variant. Failing here
        # beats silently running the expensive synchronous path.
        raise ConfigError(
            "LLM_USE_BATCH=true is not supported on LLM_PROVIDER=anthropic yet; "
            "Anthropic batching needs the /v1/messages/batches endpoint (Phase 8). "
            "Use LLM_PROVIDER=openrouter for batch, or unset LLM_USE_BATCH."
        )

    models = dict(STAGE_MODELS)
    for stage in LLM_STAGES:
        # Escape hatch for pinning a different model per stage without a code
        # edit, e.g. EXTRACT_MODEL_OPENROUTER=anthropic/claude-sonnet-5.
        override = _clean(env.get(f"{stage.upper()}_MODEL_{provider.upper()}"))
        if override:
            spec = models[stage]
            field_name = "openrouter_id" if provider == "openrouter" else "anthropic_id"
            models[stage] = replace(spec, **{field_name: override})
            log.info("model override: %s stage -> %s", stage, override)

    return LLMSettings(
        provider=provider,  # type: ignore[arg-type]
        base_url=base_url,
        api_key=_clean(env.get(PROVIDER_KEY_VARS[provider])),
        enforce_schema=_flag(env.get("LLM_ENFORCE_SCHEMA"), default=True),
        use_batch=use_batch,
        max_attempts=_int(env.get("LLM_MAX_ATTEMPTS"), default=3, minimum=1),
        timeout=_float(env.get("LLM_TIMEOUT"), default=120.0, minimum=1.0),
        models=models,
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Resolved environment. Keys are optional here and required at point of use.

    Phase 1 touches no API, so an absent key must not stop a normalize or a
    store round-trip. The `require_*` accessors name the missing variable
    exactly when something actually needs it.
    """

    store_backend: StoreBackend
    store_root: str
    pack_dir: Path
    firecrawl_api_key: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None
    apollo_api_key: str | None = None
    llm: LLMSettings = field(
        default_factory=lambda: LLMSettings(
            provider="openrouter",
            base_url=PROVIDER_BASE_URLS["openrouter"],
            api_key=None,
        )
    )

    def _require(self, value: str | None, name: str) -> str:
        if not value:
            raise ConfigError(f"{name} is not set")
        return value

    def require_firecrawl_api_key(self) -> str:
        return self._require(self.firecrawl_api_key, "FIRECRAWL_API_KEY")

    def require_anthropic_api_key(self) -> str:
        return self._require(self.anthropic_api_key, "ANTHROPIC_API_KEY")

    def require_openrouter_api_key(self) -> str:
        return self._require(self.openrouter_api_key, "OPENROUTER_API_KEY")

    def require_apollo_api_key(self) -> str:
        return self._require(self.apollo_api_key, "APOLLO_API_KEY")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().strip('"').strip("'")
    return text or None


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _flag(value: str | None, *, default: bool) -> bool:
    text = (_clean(value) or "").lower()
    if not text:
        return default
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"expected a boolean (true/false), got {value!r}")


def _int(value: str | None, *, default: int, minimum: int) -> int:
    text = _clean(value)
    if not text:
        return default
    try:
        parsed = int(text)
    except ValueError:
        raise ConfigError(f"expected an integer, got {value!r}") from None
    if parsed < minimum:
        raise ConfigError(f"expected an integer >= {minimum}, got {parsed}")
    return parsed


def _float(value: str | None, *, default: float, minimum: float) -> float:
    text = _clean(value)
    if not text:
        return default
    try:
        parsed = float(text)
    except ValueError:
        raise ConfigError(f"expected a number, got {value!r}") from None
    if parsed < minimum:
        raise ConfigError(f"expected a number >= {minimum}, got {parsed}")
    return parsed


HUB_ENV_PATH = Path.home() / "Hub" / ".env"


def load_env_files() -> list[Path]:
    """Load the project `.env`, then the shared hub `.env`. Nearest wins.

    Most keys are shared and live in the hub file. A key that has to bill
    against this project alone (the project-scoped OpenRouter key) lives in the
    project `.env` and must not be overwritten by the hub value.

    `load_dotenv` does not override variables that are already set, so loading
    nearest-first gives exactly that precedence: project, then hub, then
    whatever the process already had. Returns the files actually loaded.
    """
    loaded: list[Path] = []
    candidates: list[Path] = []

    nearest = find_dotenv(usecwd=True)
    if nearest:
        candidates.append(Path(nearest))
    candidates.append(HUB_ENV_PATH)

    for path in candidates:
        resolved = path.expanduser()
        if resolved.is_file() and resolved not in loaded:
            load_dotenv(resolved, override=False)
            loaded.append(resolved)
            log.debug("loaded environment from %s", resolved)

    return loaded


def load_settings(env: dict[str, str] | None = None, *, use_dotenv: bool = True) -> Settings:
    """Read settings from the environment, validating the store configuration.

    `env` overrides the process environment, which is what the tests use.
    """
    if env is None:
        if use_dotenv:
            load_env_files()
        env = dict(os.environ)

    backend = (_clean(env.get("STORE_BACKEND")) or "local").lower()
    if backend not in ("local", "gcs"):
        raise ConfigError(f"STORE_BACKEND must be 'local' or 'gcs', got {backend!r}")

    store_root = _clean(env.get("STORE_ROOT")) or "./data"
    is_gs_uri = store_root.startswith("gs://")
    if backend == "gcs" and not is_gs_uri:
        raise ConfigError(f"STORE_BACKEND=gcs requires STORE_ROOT to be a gs:// URI, got {store_root!r}")
    if backend == "local" and is_gs_uri:
        raise ConfigError(f"STORE_BACKEND=local cannot use a gs:// STORE_ROOT, got {store_root!r}")

    pack_dir = Path(_clean(env.get("PACK_DIR")) or DEFAULT_PACK_DIR)

    return Settings(
        store_backend=backend,  # type: ignore[arg-type]
        store_root=store_root,
        pack_dir=pack_dir,
        firecrawl_api_key=_clean(env.get("FIRECRAWL_API_KEY")),
        anthropic_api_key=_clean(env.get("ANTHROPIC_API_KEY")),
        openrouter_api_key=_clean(env.get("OPENROUTER_API_KEY")),
        apollo_api_key=_clean(env.get("APOLLO_API_KEY")),
        llm=load_llm_settings(env),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached `load_settings` for the process-wide default."""
    return load_settings()
