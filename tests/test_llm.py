"""The provider seam: same result from both paths, one parse path.

Every request-shape test asserts against a recorded body rather than a live
call, so the whole suite runs with no key and no spend.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, Field

from cleanroom.config import STAGE_MODELS, ConfigError, LLMSettings, load_settings
from cleanroom.llm import (
    LLMClient,
    LLMError,
    LLMResponseError,
    LLMTransportError,
    Usage,
    extract_json,
    parse_into,
)


class Verdict(BaseModel):
    """A stand-in schema. Keeps these tests independent of Phase 5/6 churn."""

    domain: str
    status: str
    confidence: float = Field(ge=0.0, le=1.0)


PAYLOAD = {"domain": "example.com", "status": "in_house", "confidence": 0.9}


# ---------------------------------------------------------------------------
# Transport double
# ---------------------------------------------------------------------------


class Recorder:
    """Captures requests and replays canned provider responses."""

    def __init__(self, responses: list[httpx.Response]):
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return self.responses[index]

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


def openrouter_response(text: str = json.dumps(PAYLOAD), **usage) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 1000),
                "completion_tokens": usage.get("completion_tokens", 100),
                "prompt_tokens_details": {"cached_tokens": usage.get("cached_tokens", 0)},
            },
        },
    )


def anthropic_response(text: str = json.dumps(PAYLOAD), **usage) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": usage.get("input_tokens", 1000),
                "output_tokens": usage.get("output_tokens", 100),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            },
        },
    )


def make_client(provider: str, responses: list[httpx.Response], **overrides) -> tuple[LLMClient, Recorder]:
    from cleanroom.config import PROVIDER_BASE_URLS

    recorder = Recorder(responses)
    settings = LLMSettings(
        provider=provider,  # type: ignore[arg-type]
        base_url=PROVIDER_BASE_URLS[provider],
        api_key="test-key",
        **overrides,
    )
    transport = httpx.MockTransport(recorder.handler)
    return LLMClient(settings, client=httpx.Client(transport=transport)), recorder


# ---------------------------------------------------------------------------
# The seam: identical results, different requests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openrouter", "anthropic"])
def test_both_providers_return_the_same_validated_record(provider):
    response = openrouter_response() if provider == "openrouter" else anthropic_response()
    client, _ = make_client(provider, [response])

    result = client.complete(
        stage="classify", system="sys", user="usr", schema=Verdict
    )

    assert result.parsed == Verdict(**PAYLOAD)
    assert result.provider == provider
    assert result.attempts == 1


def test_provider_switch_changes_only_the_request():
    """The point of the seam: swapping providers is a config change."""
    or_client, or_rec = make_client("openrouter", [openrouter_response()])
    an_client, an_rec = make_client("anthropic", [anthropic_response()])

    kwargs = dict(stage="classify", system="sys", user="usr", schema=Verdict)
    or_result = or_client.complete(**kwargs)
    an_result = an_client.complete(**kwargs)

    # Same record out.
    assert or_result.parsed == an_result.parsed

    # Different request in.
    assert or_rec.requests[0].url.path.endswith("/chat/completions")
    assert an_rec.requests[0].url.path.endswith("/v1/messages")
    assert "response_format" in or_rec.body
    assert "output_config" in an_rec.body
    assert "response_format" not in an_rec.body
    assert "output_config" not in or_rec.body


def test_model_ids_differ_per_provider():
    or_client, or_rec = make_client("openrouter", [openrouter_response()])
    an_client, an_rec = make_client("anthropic", [anthropic_response()])

    or_client.complete(stage="extract", system="s", user="u", schema=Verdict)
    an_client.complete(stage="extract", system="s", user="u", schema=Verdict)

    assert or_rec.body["model"] == "anthropic/claude-haiku-4.5"
    assert an_rec.body["model"] == "claude-haiku-4-5"


def test_auth_headers_differ_per_provider():
    or_client, or_rec = make_client("openrouter", [openrouter_response()])
    an_client, an_rec = make_client("anthropic", [anthropic_response()])

    or_client.complete(stage="extract", system="s", user="u", schema=Verdict)
    an_client.complete(stage="extract", system="s", user="u", schema=Verdict)

    assert or_rec.requests[0].headers["authorization"] == "Bearer test-key"
    assert an_rec.requests[0].headers["x-api-key"] == "test-key"
    assert "anthropic-version" in an_rec.requests[0].headers


# ---------------------------------------------------------------------------
# Request-side enforcement is separable from response-side validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openrouter", "anthropic"])
def test_enforcement_off_still_validates(provider):
    """Turning off request-side schema must not weaken the parse path."""
    response = openrouter_response() if provider == "openrouter" else anthropic_response()
    client, rec = make_client(provider, [response], enforce_schema=False)

    result = client.complete(stage="classify", system="s", user="u", schema=Verdict)

    assert "response_format" not in rec.body
    assert "format" not in rec.body.get("output_config", {})
    assert result.parsed == Verdict(**PAYLOAD)  # still a validated model


def test_schema_is_tightened_for_strict_mode():
    client, rec = make_client("openrouter", [openrouter_response()])
    client.complete(stage="classify", system="s", user="u", schema=Verdict)

    schema = rec.body["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"domain", "status", "confidence"}
    assert rec.body["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("provider", ["openrouter", "anthropic"])
def test_unsupported_constraints_are_stripped_from_the_request(provider):
    """`Field(ge=0, le=1)` emits minimum/maximum, which both providers reject.

    Verified against a live OpenRouter call on 2026-08-06: the upstream error
    is `output_config.format.schema: For 'number' type, properties maximum,
    minimum are not supported`. Dropping them is safe because pydantic
    re-checks every constraint when the response is validated.
    """
    response = openrouter_response() if provider == "openrouter" else anthropic_response()
    client, rec = make_client(provider, [response])
    client.complete(stage="classify", system="s", user="u", schema=Verdict)

    sent = json.dumps(rec.body)
    for key in ("minimum", "maximum", "exclusiveMinimum", "multipleOf", "maxLength"):
        assert key not in sent


def test_stripped_constraints_are_still_enforced_on_the_response():
    """The constraint moves from request-side to response-side, not away."""
    out_of_range = json.dumps({"domain": "x.com", "status": "none", "confidence": 4.2})
    client, _ = make_client("openrouter", [openrouter_response(out_of_range)], max_attempts=1)

    with pytest.raises(LLMResponseError, match="confidence"):
        client.complete(stage="classify", system="s", user="u", schema=Verdict)


def test_nested_models_are_sanitized_too():
    """The real Phase 5 record nests Finding, which carries its own constraints."""
    from cleanroom.models import ExtractionRecord

    client, rec = make_client("openrouter", [openrouter_response()], max_attempts=1)
    with pytest.raises(LLMResponseError):  # canned body isn't an ExtractionRecord
        client.complete(stage="extract", system="s", user="u", schema=ExtractionRecord)

    schema = rec.body["response_format"]["json_schema"]["schema"]
    assert "$defs" in schema  # Finding is nested by reference
    assert "minimum" not in json.dumps(schema)
    assert schema["$defs"]["Finding"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Caching and effort
# ---------------------------------------------------------------------------


def test_system_prompt_is_marked_cacheable_on_both_paths():
    or_client, or_rec = make_client("openrouter", [openrouter_response()])
    an_client, an_rec = make_client("anthropic", [anthropic_response()])

    or_client.complete(stage="extract", system="big pack prefix", user="u", schema=Verdict)
    an_client.complete(stage="extract", system="big pack prefix", user="u", schema=Verdict)

    assert or_rec.body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert an_rec.body["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_caching_can_be_turned_off():
    client, rec = make_client("openrouter", [openrouter_response()])
    client.complete(stage="extract", system="s", user="u", schema=Verdict, cache_system=False)
    assert rec.body["messages"][0]["content"] == "s"


def test_effort_sent_only_where_the_model_supports_it():
    """Haiku 4.5 has no effort parameter; sending one errors upstream."""
    client, rec = make_client("openrouter", [openrouter_response()] * 2)
    client.complete(stage="extract", system="s", user="u", schema=Verdict)
    assert "reasoning_effort" not in rec.bodies[0]

    client.complete(stage="classify", system="s", user="u", schema=Verdict)
    assert rec.bodies[1]["reasoning_effort"] == "medium"


def test_classify_gets_headroom_for_thinking():
    """Sonnet 5 thinks by default and max_tokens caps thinking plus output."""
    client, rec = make_client("openrouter", [openrouter_response()])
    client.complete(stage="classify", system="s", user="u", schema=Verdict)
    assert rec.body["max_tokens"] == STAGE_MODELS["classify"].max_output_tokens >= 16_000


# ---------------------------------------------------------------------------
# Retries and failures
# ---------------------------------------------------------------------------


def test_invalid_json_retries_with_the_error_appended():
    client, rec = make_client(
        "openrouter", [openrouter_response("not json at all"), openrouter_response()]
    )

    result = client.complete(stage="classify", system="s", user="u", schema=Verdict)

    assert result.attempts == 2
    assert result.parsed == Verdict(**PAYLOAD)
    # The retry tells the model exactly what was wrong.
    assert "could not be parsed" in rec.bodies[1]["messages"][1]["content"]


def test_schema_violation_retries():
    bad = json.dumps({"domain": "example.com", "status": "in_house", "confidence": 9.9})
    client, _ = make_client("openrouter", [openrouter_response(bad), openrouter_response()])
    assert client.complete(stage="classify", system="s", user="u", schema=Verdict).attempts == 2


def test_exhausted_retries_raise_with_the_raw_text():
    client, _ = make_client("openrouter", [openrouter_response("nope")], max_attempts=2)
    with pytest.raises(LLMResponseError) as exc:
        client.complete(stage="classify", system="s", user="u", schema=Verdict)
    assert exc.value.raw_text == "nope"


def test_retryable_status_is_retried(monkeypatch):
    monkeypatch.setattr("cleanroom.llm.time.sleep", lambda _: None)
    client, rec = make_client(
        "openrouter", [httpx.Response(429, text="slow down"), openrouter_response()]
    )
    assert client.complete(stage="classify", system="s", user="u", schema=Verdict).attempts == 2
    assert len(rec.requests) == 2


def test_client_error_fails_fast():
    """A 400 will be a 400 again. Don't burn attempts on it."""
    client, rec = make_client("openrouter", [httpx.Response(400, text="bad model")])
    with pytest.raises(LLMError, match="400"):
        client.complete(stage="classify", system="s", user="u", schema=Verdict)
    assert len(rec.requests) == 1


def test_network_error_is_wrapped(monkeypatch):
    monkeypatch.setattr("cleanroom.llm.time.sleep", lambda _: None)

    def boom(request):
        raise httpx.ConnectError("no route to host")

    from cleanroom.config import PROVIDER_BASE_URLS

    settings = LLMSettings(
        provider="openrouter",
        base_url=PROVIDER_BASE_URLS["openrouter"],
        api_key="k",
        max_attempts=2,
    )
    client = LLMClient(settings, client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(LLMTransportError, match="ConnectError"):
        client.complete(stage="classify", system="s", user="u", schema=Verdict)


def test_missing_key_names_the_right_variable():
    from cleanroom.config import PROVIDER_BASE_URLS

    settings = LLMSettings(
        provider="openrouter", base_url=PROVIDER_BASE_URLS["openrouter"], api_key=None
    )
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        LLMClient(settings).complete(stage="extract", system="s", user="u", schema=Verdict)


# ---------------------------------------------------------------------------
# Usage and cost
# ---------------------------------------------------------------------------


def test_usage_and_cost_openrouter():
    client, _ = make_client(
        "openrouter",
        [openrouter_response(prompt_tokens=5000, completion_tokens=600, cached_tokens=2000)],
    )
    usage = client.complete(stage="extract", system="s", user="u", schema=Verdict).usage

    # prompt_tokens is inclusive of cached reads; they bill differently.
    assert usage.input_tokens == 3000
    assert usage.cache_read_tokens == 2000
    assert usage.output_tokens == 600
    # 3000@$1 + 600@$5 + 2000@$0.10 per Mtok
    assert usage.cost_usd == pytest.approx((3000 * 1.0 + 600 * 5.0 + 2000 * 0.10) / 1e6)


def test_usage_and_cost_anthropic():
    client, _ = make_client(
        "anthropic",
        [anthropic_response(input_tokens=3000, output_tokens=600, cache_read_input_tokens=2000)],
    )
    usage = client.complete(stage="extract", system="s", user="u", schema=Verdict).usage
    assert (usage.input_tokens, usage.cache_read_tokens) == (3000, 2000)
    assert usage.cost_usd == pytest.approx((3000 * 1.0 + 600 * 5.0 + 2000 * 0.10) / 1e6)


def test_batch_halves_the_cost():
    plain, _ = make_client("openrouter", [openrouter_response()])
    batch, rec = make_client("openrouter", [openrouter_response()], use_batch=True)

    a = plain.complete(stage="extract", system="s", user="u", schema=Verdict)
    b = batch.complete(stage="extract", system="s", user="u", schema=Verdict)

    assert rec.body["model"] == "anthropic/claude-haiku-4.5:batch"
    assert b.usage.cost_usd == pytest.approx(a.usage.cost_usd / 2)


def test_usage_adds():
    total = Usage(1, 2, 3, 4, 0.5) + Usage(10, 20, 30, 40, 1.5)
    assert (total.input_tokens, total.output_tokens, total.cost_usd) == (11, 22, 2.0)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '  {"a": 1}  ',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the result:\n{"a": 1}',
        '{"a": 1}\nLet me know if you need more.',
    ],
)
def test_extract_json_tolerates_wrapping(text):
    assert json.loads(extract_json(text)) == {"a": 1}


@pytest.mark.parametrize("text", ["", "   ", "no object here"])
def test_extract_json_rejects_junk(text):
    with pytest.raises(LLMResponseError):
        extract_json(text)


def test_parse_into_reports_the_schema_by_name():
    with pytest.raises(LLMResponseError, match="Verdict"):
        parse_into('{"domain": "x"}', Verdict)


# ---------------------------------------------------------------------------
# Settings wiring
# ---------------------------------------------------------------------------


def test_default_provider_is_openrouter():
    llm = load_settings(env={"OPENROUTER_API_KEY": "k"}).llm
    assert llm.provider == "openrouter"
    assert llm.api_key == "k"
    assert llm.model_id("extract") == "anthropic/claude-haiku-4.5"
    assert llm.model_id("classify") == "anthropic/claude-sonnet-5"


def test_switching_provider_switches_the_key_and_ids():
    llm = load_settings(
        env={"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-ant", "OPENROUTER_API_KEY": "or"}
    ).llm
    assert llm.api_key == "sk-ant"  # not the OpenRouter one
    assert llm.model_id("extract") == "claude-haiku-4-5"
    assert llm.model_id("classify") == "claude-sonnet-5"


def test_unknown_provider_raises():
    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        load_settings(env={"LLM_PROVIDER": "togetherai"})


def test_batch_on_anthropic_is_refused_loudly():
    """Anthropic batching is a different endpoint, not a model suffix."""
    with pytest.raises(ConfigError, match="messages/batches"):
        load_settings(env={"LLM_PROVIDER": "anthropic", "LLM_USE_BATCH": "true"})


def test_batch_on_openrouter_uses_the_slug_suffix():
    llm = load_settings(env={"LLM_USE_BATCH": "yes", "OPENROUTER_API_KEY": "k"}).llm
    assert llm.model_id("extract") == "anthropic/claude-haiku-4.5:batch"


def test_model_override_by_env():
    llm = load_settings(
        env={"OPENROUTER_API_KEY": "k", "CLASSIFY_MODEL_OPENROUTER": "anthropic/claude-opus-4.8"}
    ).llm
    assert llm.model_id("classify") == "anthropic/claude-opus-4.8"
    assert llm.model_id("extract") == "anthropic/claude-haiku-4.5"  # untouched


def test_missing_key_for_selected_provider_is_reported_by_name():
    llm = load_settings(env={"LLM_PROVIDER": "anthropic"}).llm
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        llm.require_api_key()


def test_bad_boolean_raises():
    with pytest.raises(ConfigError, match="boolean"):
        load_settings(env={"LLM_USE_BATCH": "maybe"})


def test_cost_table_matches_verified_openrouter_pricing():
    """Guards against a silent edit to the numbers a run bills itself against."""
    assert STAGE_MODELS["extract"].input_usd_per_mtok == 1.00
    assert STAGE_MODELS["extract"].output_usd_per_mtok == 5.00
    assert STAGE_MODELS["classify"].input_usd_per_mtok == 2.00
    assert STAGE_MODELS["classify"].output_usd_per_mtok == 10.00
