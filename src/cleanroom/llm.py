"""One LLM client, two providers, one parse path.

The trap this module exists to avoid: OpenRouter is OpenAI-compatible and takes
`response_format`; the Anthropic API takes `output_config.format`. Build the
extraction and classification calls against either one directly and swapping
providers means rewriting the plumbing at every call site.

The split that prevents it:

    request side   varies by provider (response_format | output_config.format)
    response side  ALWAYS raw text -> json -> pydantic

Schema enforcement is a property of the *request*. Validation is a property of
the *response*. Keeping them separate means `enforce_schema` can be toggled,
and the provider can be swapped, without a single line downstream changing —
`complete()` returns the same validated pydantic model either way.

Deliberately raw `httpx` rather than the `openai` or `anthropic` SDK. Both SDKs
would parse the response for us, which is exactly the layer that has to stay
identical across providers. One HTTP client and two request builders is smaller
than two SDKs and two response handlers.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Generic, Mapping, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import ConfigError, LLMSettings, LLMStage, ModelSpec, Settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

ANTHROPIC_VERSION = "2023-06-01"

# Retried with backoff. Everything else fails fast: a 400 means the request is
# wrong and will be wrong again.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMError(RuntimeError):
    """Base class for LLM call failures."""


class LLMTransportError(LLMError):
    """The provider could not be reached, or returned an unusable status."""


class LLMResponseError(LLMError):
    """The provider replied, but the body was not a valid record.

    Carries the raw text so a parsing failure can be inspected without a
    re-run. A failure here is a prompt or schema problem, not a network one.
    """

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


@dataclass(frozen=True)
class Usage:
    """Token counts and the resulting spend for one call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True)
class Completion(Generic[T]):
    """A validated record, plus what it cost and how it was produced."""

    parsed: T
    raw_text: str
    model: str
    provider: str
    usage: Usage
    attempts: int


# ---------------------------------------------------------------------------
# Response parsing, identical on every provider
# ---------------------------------------------------------------------------


def extract_json(text: str) -> str:
    """Pull the JSON object out of a model response.

    Even with `response_format` set, models occasionally wrap output in a
    markdown fence or add a sentence before it. Stripping that here keeps the
    tolerance in one place instead of scattered through the callers.
    """
    stripped = text.strip()
    if not stripped:
        raise LLMResponseError("model returned an empty response", text)

    fenced = _FENCE_RE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMResponseError("no JSON object found in the model response", text)
    return stripped[start : end + 1]


def parse_into(text: str, schema: type[T]) -> T:
    """Validate raw model text into `schema`. The one place records are built."""
    payload_text = extract_json(text)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"response was not valid JSON: {exc}", text) from exc

    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseError(f"response did not match {schema.__name__}:\n{exc}", text) from exc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Structured-output calls against whichever provider the settings name.

    `client` is injectable so the whole request-and-parse path can be exercised
    against a transport double, with no key and no spend.
    """

    def __init__(self, settings: LLMSettings, *, client: httpx.Client | None = None):
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=settings.timeout)
        self._cache_warned = False

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> LLMClient:
        return cls(settings.llm, **kwargs)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        *,
        stage: LLMStage,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int | None = None,
        cache_system: bool = True,
    ) -> Completion[T]:
        """One structured call. Returns a validated `schema` instance.

        `cache_system` marks the system prompt as cacheable. The pack vocabulary
        and label definitions are a large stable prefix reused across thousands
        of calls, and cache reads are a tenth of the input price, so this is on
        by default.

        Retries cover two different failures: a retryable HTTP status, and a
        response that did not validate. The second retry appends the validation
        error to the prompt, because a model told exactly what was wrong with
        its last attempt usually fixes it.
        """
        spec = self.settings.spec(stage)
        model = self.settings.model_id(stage)
        budget = max_tokens or spec.max_output_tokens

        if cache_system and not self._cache_warned:
            # Rough 4 chars/token. A prefix under the threshold caches nothing
            # and reports nothing, so the only symptom is a bill that never
            # drops. Say it once per client rather than per call.
            if len(system) / 4 < spec.cache_min_tokens:
                log.warning(
                    "%s stage: system prompt is ~%d tokens, below the %d-token "
                    "caching minimum for %s. Caching will not engage.",
                    stage, len(system) // 4, spec.cache_min_tokens, model,
                )
            self._cache_warned = True

        attempt = 0
        correction: str | None = None
        last_error: LLMError | None = None

        while attempt < self.settings.max_attempts:
            attempt += 1
            prompt = user if correction is None else f"{user}\n\n{correction}"
            body = self._build_request(spec, model, system, prompt, schema, budget, cache_system)

            try:
                payload = self._post(body)
            except LLMTransportError as exc:
                last_error = exc
                if attempt >= self.settings.max_attempts:
                    break
                self._backoff(attempt)
                continue

            text = self._response_text(payload)
            usage = self._usage(payload, spec)

            try:
                parsed = parse_into(text, schema)
            except LLMResponseError as exc:
                last_error = exc
                log.warning(
                    "%s stage attempt %d/%d did not validate: %s",
                    stage, attempt, self.settings.max_attempts, exc,
                )
                correction = (
                    "Your previous response could not be parsed into the required "
                    f"schema. The error was:\n{exc}\n"
                    "Return only a single valid JSON object matching the schema. "
                    "No prose, no markdown fence."
                )
                continue

            log.debug(
                "%s stage ok in %d attempt(s): %d in / %d out, $%.5f",
                stage, attempt, usage.input_tokens, usage.output_tokens, usage.cost_usd,
            )
            return Completion(
                parsed=parsed,
                raw_text=text,
                model=model,
                provider=self.settings.provider,
                usage=usage,
                attempts=attempt,
            )

        assert last_error is not None
        raise last_error

    # -- request building, the only provider-specific code ------------------

    def _build_request(
        self,
        spec: ModelSpec,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
        cache_system: bool,
    ) -> dict[str, Any]:
        if self.settings.provider == "openrouter":
            return self._openrouter_body(spec, model, system, user, schema, max_tokens, cache_system)
        return self._anthropic_body(spec, model, system, user, schema, max_tokens, cache_system)

    def _openrouter_body(
        self,
        spec: ModelSpec,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
        cache_system: bool,
    ) -> dict[str, Any]:
        system_content: Any = system
        if cache_system:
            # The content-parts form is what carries Anthropic's cache_control
            # through OpenRouter's OpenAI-compatible surface.
            system_content = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user},
            ],
        }
        if self.settings.enforce_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": _json_schema(schema),
                },
            }
        if spec.effort:
            body["reasoning_effort"] = spec.effort
        return body

    def _anthropic_body(
        self,
        spec: ModelSpec,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
        cache_system: bool,
    ) -> dict[str, Any]:
        system_block: dict[str, Any] = {"type": "text", "text": system}
        if cache_system:
            system_block["cache_control"] = {"type": "ephemeral"}

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [system_block],
            "messages": [{"role": "user", "content": user}],
        }
        if self.settings.enforce_schema:
            body["output_config"] = {
                "format": {"type": "json_schema", "schema": _json_schema(schema)}
            }
        if spec.effort:
            body.setdefault("output_config", {})["effort"] = spec.effort
        return body

    def _endpoint(self) -> str:
        if self.settings.provider == "openrouter":
            return f"{self.settings.base_url}/chat/completions"
        return f"{self.settings.base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        key = self.settings.require_api_key()
        if self.settings.provider == "openrouter":
            return {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": "cleanroom-classifier",
            }
        return {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    # -- transport ----------------------------------------------------------

    def _post(self, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(
                self._endpoint(),
                headers=self._headers(),
                json=dict(body),
                timeout=self.settings.timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMTransportError(f"{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:600]
            message = f"HTTP {response.status_code} from {self.settings.provider}: {detail}"
            if response.status_code in RETRYABLE_STATUS:
                raise LLMTransportError(message)
            raise LLMError(message)

        try:
            return response.json()
        except ValueError as exc:
            raise LLMTransportError(f"provider returned non-JSON body: {exc}") from exc

    def _backoff(self, attempt: int) -> None:
        delay = min(2.0 ** (attempt - 1), 30.0)
        log.info("retrying after %.1fs", delay)
        time.sleep(delay)

    # -- response reading ---------------------------------------------------

    def _response_text(self, payload: Mapping[str, Any]) -> str:
        """Reach the assistant text. The last provider-shaped step."""
        try:
            if self.settings.provider == "openrouter":
                return payload["choices"][0]["message"]["content"] or ""
            blocks = payload.get("content") or []
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"unexpected response shape from {self.settings.provider}: {exc}",
                json.dumps(payload)[:1000],
            ) from exc

    def _usage(self, payload: Mapping[str, Any], spec: ModelSpec) -> Usage:
        raw = payload.get("usage") or {}
        if self.settings.provider == "openrouter":
            details = raw.get("prompt_tokens_details") or {}
            cache_read = int(details.get("cached_tokens") or 0)
            # OpenRouter reports prompt_tokens inclusive of cached reads.
            input_tokens = max(int(raw.get("prompt_tokens") or 0) - cache_read, 0)
            output_tokens = int(raw.get("completion_tokens") or 0)
            cache_write = int(raw.get("cache_creation_input_tokens") or 0)
        else:
            input_tokens = int(raw.get("input_tokens") or 0)
            output_tokens = int(raw.get("output_tokens") or 0)
            cache_read = int(raw.get("cache_read_input_tokens") or 0)
            cache_write = int(raw.get("cache_creation_input_tokens") or 0)

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=spec.cost_usd(
                input_tokens, output_tokens, cache_read, cache_write,
                batch=self.settings.use_batch,
            ),
        )


# Anthropic's structured-output schema validator rejects these outright, and
# OpenRouter translates `response_format` into `output_config.format` before
# forwarding, so the restriction applies identically on both paths.
#
# Dropping them costs nothing: every one of these is a pydantic Field
# constraint, and pydantic re-checks all of them when `parse_into` validates
# the response. The constraint moves from request-side to response-side, which
# is where validation lives anyway.
UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def _json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema, made acceptable to strict structured-output modes.

    Two edits. Strip the constraint keywords the providers reject, and tighten
    every object: `additionalProperties: false` plus all properties required,
    neither of which pydantic emits by default and both of which strict mode
    demands.
    """
    generated = schema.model_json_schema()
    _sanitize(generated)
    return generated


def _sanitize(node: Any) -> None:
    if isinstance(node, dict):
        for key in UNSUPPORTED_SCHEMA_KEYS & node.keys():
            del node[key]
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            _sanitize(value)
    elif isinstance(node, list):
        for item in node:
            _sanitize(item)


def get_llm_client(settings: Settings, **kwargs: Any) -> LLMClient:
    """Build the client the settings ask for. The only place providers are chosen."""
    if settings.llm.provider not in ("openrouter", "anthropic"):  # pragma: no cover
        raise ConfigError(f"unsupported LLM provider {settings.llm.provider!r}")
    return LLMClient(settings.llm, **kwargs)
