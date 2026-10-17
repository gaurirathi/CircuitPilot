"""Natural-language planning into the validated CircuitPilot graph IR.

Planner implementations return structured data only. No planner output is
treated as MATLAB, Python, or shell code.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import urlsplit, urlunsplit
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .ir import ModelSpec, validate_model
from .repair import RepairDiagnosis, RepairProposal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOGGER = logging.getLogger(__name__)


def load_project_dotenv(project_root: Path | None = None) -> bool:
    """Load the project's root .env without overriding process environment values."""
    env_path = (project_root or PROJECT_ROOT) / ".env"
    return load_dotenv(dotenv_path=env_path, override=False)


# Load before provider/model/key configuration is read by the CLI or planners.
load_project_dotenv()


class PlannerError(ValueError):
    """Planning failed or the proposed graph did not pass validation."""


class ClarificationRequired(PlannerError):
    """The request is ambiguous, incomplete, or outside the supported domain."""


class ProviderRequestError(PlannerError):
    """Safe provider failure details plus transient retry metadata."""

    def __init__(
        self,
        message: str,
        *,
        retry_count: int,
        transient: bool,
        status_code: int | None,
        provider_error: str,
        failure_type: Literal["provider", "timeout", "unexpected"],
    ):
        super().__init__(message)
        self.retry_count = retry_count
        self.transient = transient
        self.status_code = status_code
        self.provider_error = provider_error
        self.failure_type = failure_type
        self.error_category = "rate_limit" if status_code == 429 else failure_type


DEFAULT_LLM_MAX_RETRIES = 2
MAX_LLM_RETRIES = 5
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
MAX_LLM_TIMEOUT_SECONDS = 300.0
MAX_RETRY_DELAY_SECONDS = 30.0


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key_env: str
    base_url: str | None
    default_model: str
    sdk: str


PROVIDERS = {
    "openai": ProviderConfig("openai", "OPENAI_API_KEY", None, "gpt-4o-mini", "openai"),
    "grok": ProviderConfig("grok", "XAI_API_KEY", "https://api.x.ai/v1", "grok-4.7", "openai"),
    "gemini": ProviderConfig("gemini", "GEMINI_API_KEY", None, "gemini-3.8-flash", "google"),
}

# Pin the Interactions API version explicitly so the request route is visible
# and testable. The configured Gemini endpoint serves Interactions on v1beta.
GEMINI_INTERACTIONS_API_VERSION = "v1beta"


def get_provider_config(provider: str) -> ProviderConfig:
    normalized = provider.strip().lower()
    try:
        return PROVIDERS[normalized]
    except KeyError as exc:
        raise PlannerError(
            f"Unsupported planner provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}."
        ) from exc


def create_provider_client(
    provider: str,
    *,
    client_factory: Any | None = None,
    environ: dict[str, str] | None = None,
) -> Any:
    """Build a provider client from provider-specific environment settings."""
    config = get_provider_config(provider)
    environment = os.environ if environ is None else environ
    api_key = environment.get(config.api_key_env)
    if not api_key:
        raise PlannerError(f"Set {config.api_key_env} in the environment to use the {config.name} planner.")
    if client_factory is None:
        if config.sdk == "google":
            try:
                from google import genai
            except ImportError as exc:
                raise PlannerError(
                    "Gemini planning requires the Google Gen AI SDK. Install CircuitPilot with `pip install -e .[llm]`."
                ) from exc
            client_factory = genai.Client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise PlannerError(
                    "LLM planning requires the OpenAI Python SDK. Install CircuitPilot with `pip install -e .[llm]`."
                ) from exc
            client_factory = OpenAI
    timeout_seconds = _configured_timeout_seconds()
    arguments: dict[str, Any] = {"api_key": api_key}
    if config.base_url is not None:
        arguments["base_url"] = config.base_url
    if config.sdk == "google":
        # Start with the documented no-retries value. In google-genai 2.31.0,
        # client construction normalizes zero to one; the post-construction
        # adapter-specific override below corrects the value actually consumed.
        arguments["http_options"] = {
            "timeout": int(timeout_seconds * 1000),
            "retry_options": {"attempts": 0},
        }
    else:
        arguments["timeout"] = timeout_seconds
        arguments["max_retries"] = 0
    client = client_factory(**arguments)
    if config.sdk == "google":
        _disable_google_genai_retries(client)
    return client


def _disable_google_genai_retries(client: Any) -> None:
    """Work around google-genai 2.31.0 Interactions retry-option normalization.

    That SDK documents attempts=0/1 as no retries, but Client normalizes both to
    1 and the Interactions adapter translates it as one additional retry. Set
    the actual client option consumed by the adapter to zero and fail closed if
    an installed Google client no longer exposes the expected configuration.
    Test doubles without SDK internals are intentionally left alone.
    """
    if not type(client).__module__.startswith("google.genai"):
        return
    api_client = getattr(client, "_api_client", None)
    http_options = getattr(api_client, "_http_options", None)
    retry_options = getattr(http_options, "retry_options", None)
    if retry_options is None or not hasattr(retry_options, "attempts"):
        raise PlannerError(
            "This google-genai client does not expose retry configuration CircuitPilot can safely disable."
        )
    retry_options.attempts = 0
    if retry_options.attempts != 0:
        raise PlannerError("Could not disable hidden google-genai Interactions retries.")


class PlannerEnvelope(BaseModel):
    """Strict LLM response envelope; the embedded graph is validated separately."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "clarification"]
    graph_json: str
    message: str

    @model_validator(mode="after")
    def status_fields_agree(self) -> "PlannerEnvelope":
        if self.status == "ready" and not self.graph_json.strip():
            raise ValueError("A ready planner response must include graph_json")
        if self.status == "clarification" and not self.message.strip():
            raise ValueError("A clarification response must include a question or explanation")
        return self


class GeminiDiagnosticEnvelope(BaseModel):
    """Tiny schema used to distinguish basic Gemini structured output from planner schema issues."""

    status: str


class Planner(Protocol):
    def plan(self, prompt: str) -> ModelSpec:
        """Convert a natural-language request to a fully validated graph."""

    def propose_repair(self, spec: ModelSpec, diagnosis: RepairDiagnosis) -> RepairProposal:
        """Suggest a structured parameter or simulation-duration patch."""


def _parse_envelope(raw: str | dict[str, Any] | PlannerEnvelope) -> PlannerEnvelope:
    try:
        if isinstance(raw, PlannerEnvelope):
            return raw
        if isinstance(raw, str):
            return PlannerEnvelope.model_validate_json(raw)
        return PlannerEnvelope.model_validate(raw)
    except (ValidationError, ValueError) as exc:
        raise PlannerError(f"Planner returned malformed structured output: {exc}") from exc


def _validated_graph(envelope: PlannerEnvelope) -> ModelSpec:
    if envelope.status == "clarification":
        raise ClarificationRequired(envelope.message)
    try:
        graph = ModelSpec.model_validate_json(envelope.graph_json)
        validate_model(graph)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise PlannerError(f"Planner graph failed validation: {exc}") from exc
    return graph


def _safe_request_error(exc: Exception, *, api_key: str | None = None) -> str:
    """Format provider diagnostics without exposing credentials or request headers."""
    response = getattr(exc, "response", None)
    status = (
        getattr(exc, "status_code", None)
        or getattr(response, "status_code", None)
        or getattr(exc, "code", None)
    )
    endpoint = ""
    request = getattr(response, "request", None)
    raw_url = getattr(request, "url", None) or getattr(response, "url", None)
    if raw_url:
        parts = urlsplit(str(raw_url))
        endpoint = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    method = getattr(request, "method", None)
    detail = ""
    content_type = ""
    safe_headers: list[str] = []
    if response is not None:
        try:
            detail = response.text or ""
        except Exception:
            detail = ""
        try:
            headers = response.headers
            content_type = str(headers.get("content-type", ""))
            for name in ("date", "server", "x-request-id", "x-goog-request-id"):
                value = headers.get(name)
                if value:
                    safe_headers.append(f"{name}={value}")
        except Exception:
            pass
    if not detail:
        detail = str(getattr(exc, "message", None) or getattr(exc, "details", None) or exc)
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    detail = re.sub(
        r"(?i)(authorization|x-goog-api-key)(['\"]?\s*[:=]\s*['\"]?)[^\s,'\"}]+",
        r"\1\2[REDACTED]",
        detail,
    )
    is_html = "text/html" in content_type.lower() or bool(re.search(r"(?is)^\s*<!doctype\s+html|^\s*<html", detail))
    detail = unescape(re.sub(r"<[^>]*>", " ", detail))
    detail = re.sub(r"\s+", " ", detail).strip()
    if not detail:
        detail = "No response details were provided."
    detail = detail[:1000]
    prefix = f"HTTP {status}" if status else "HTTP status unavailable"
    if method and endpoint:
        prefix += f" for {method} {endpoint}"
    elif endpoint:
        prefix += f" for {endpoint}"
    metadata = []
    if content_type:
        metadata.append(f"content-type={content_type}")
    if safe_headers:
        metadata.append("headers: " + ", ".join(safe_headers))
    if metadata:
        prefix += " (" + "; ".join(metadata) + ")"
    if is_html:
        detail = "HTML response body: " + detail
    exception_info = type(exc).__name__
    api_status = getattr(exc, "status", None)
    if api_status:
        exception_info += f", API status={api_status}"
    return f"{prefix}: {detail} [{exception_info}]"


def _http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    value = (
        getattr(exc, "status_code", None)
        or getattr(response, "status_code", None)
        or getattr(exc, "code", None)
    )
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _configured_max_retries(override: int | None = None) -> int:
    if override is not None:
        limit = override
    else:
        raw = os.getenv("CIRCUITPILOT_LLM_MAX_RETRIES")
        try:
            limit = DEFAULT_LLM_MAX_RETRIES if raw is None else int(raw)
        except ValueError as exc:
            raise PlannerError("CIRCUITPILOT_LLM_MAX_RETRIES must be an integer from 0 to 5.") from exc
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= MAX_LLM_RETRIES:
        raise PlannerError("The LLM retry limit must be an integer from 0 to 5.")
    return limit


def _configured_timeout_seconds() -> float:
    raw = os.getenv("CIRCUITPILOT_LLM_TIMEOUT_SECONDS")
    try:
        timeout = DEFAULT_LLM_TIMEOUT_SECONDS if raw is None else float(raw)
    except ValueError as exc:
        raise PlannerError("CIRCUITPILOT_LLM_TIMEOUT_SECONDS must be a number from 1 to 300.") from exc
    if not 1 <= timeout <= MAX_LLM_TIMEOUT_SECONDS:
        raise PlannerError("CIRCUITPILOT_LLM_TIMEOUT_SECONDS must be a number from 1 to 300.")
    return timeout


def _is_timeout_exception(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or any(
        "timeout" in error_type.__name__.lower()
        for error_type in type(exc).__mro__
    )


def _request_with_retries(
    operation: Any,
    *,
    label: str,
    api_key: str | None = None,
    max_retries: int | None = None,
    sleep_fn: Any | None = None,
    jitter_fn: Any | None = None,
) -> Any:
    """Retry only HTTP 429 and 5xx request failures; parsing is done by callers."""
    sleep_fn = sleep_fn or time.sleep
    jitter_fn = jitter_fn or random.uniform
    retry_limit = _configured_max_retries(max_retries)
    retries = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            status = _http_status_code(exc)
            is_timeout = _is_timeout_exception(exc)
            transient = is_timeout or status == 429 or (status is not None and 500 <= status <= 599)
            failure_type: Literal["provider", "timeout", "unexpected"] = (
                "timeout" if is_timeout else "provider" if status is not None else "unexpected"
            )
            provider_error = _safe_request_error(exc, api_key=api_key)
            if not transient or retries >= retry_limit:
                raise ProviderRequestError(
                    f"{label} failed: {provider_error}",
                    retry_count=retries,
                    transient=transient,
                    status_code=status,
                    provider_error=provider_error,
                    failure_type=failure_type,
                ) from exc

            backoff = jitter_fn(0.5 * (2 ** retries), 1.5 * (2 ** retries))
            retry_after = _retry_after_seconds(exc)
            delay = min(max(float(backoff), retry_after or 0.0), MAX_RETRY_DELAY_SECONDS)
            sleep_fn(delay)
            retries += 1


class MockPlanner:
    """Deterministic prompt-to-envelope mapping for offline tests and demos."""

    def __init__(self, responses: dict[str, str | dict[str, Any] | PlannerEnvelope]):
        self._responses = dict(responses)

    def plan(self, prompt: str) -> ModelSpec:
        if not prompt.strip():
            raise ClarificationRequired("Enter a control-system description to plan.")
        if prompt not in self._responses:
            raise ClarificationRequired("No deterministic mock plan is configured for this prompt.")
        return _validated_graph(_parse_envelope(self._responses[prompt]))


SYSTEM_INSTRUCTIONS = """You convert natural-language descriptions into a CircuitPilot classical-control graph.
Return only the requested structured response. The graph_json field must contain JSON
matching this graph shape: name, components [{id,type,parameters}], connections
[{source,source_port,target,target_port}], stop_time, output_component, output_port,
and optional verification criteria. Include verification only for performance limits
the user explicitly requested. Supported keys are expected_final_value,
max_tracking_error, max_overshoot_percent, max_settling_time, and settling_band_percent.
The model name must be a valid identifier: start with a letter and use only letters,
digits, and underscores (no spaces or punctuation).
Supported component types are exactly: Step, Sum, Gain, Transfer Fcn, PID Controller,
Integrator, State-Space, Scope. Provide required parameters explicitly: Step needs
Time/Before/After; Sum needs Inputs signs; Gain needs Gain; Transfer Fcn needs Numerator
and Denominator coefficient arrays; PID Controller needs P/I/D; Integrator needs
InitialCondition; State-Space needs A/B/C/D matrices. Scope has no parameters.
Every signal input must have exactly one driver, ports are one-based, and connections
must describe the requested topology. Select a signal-producing block for
output_component (Scope is a sink and has no output ports); output_port must be one of
that block's actual output ports and must have a connection. If important topology or engineering parameters are missing, ask
for clarification instead of assuming values. For unsupported requests (including
electrical/power-electronics blocks), return status=clarification and explain the limit.
Never return MATLAB source, code, or executable instructions in graph_json."""


class OpenAIPlanner:
    """OpenAI-compatible Responses API adapter for OpenAI or xAI Grok."""

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
        provider: str | None = None,
        max_retries: int | None = None,
    ):
        selected_provider = provider or os.getenv("CIRCUITPILOT_PROVIDER", "openai")
        config = get_provider_config(selected_provider)
        self.provider = config.name
        self.model = model or os.getenv("CIRCUITPILOT_MODEL") or config.default_model
        self.max_retries = max_retries
        if client is None:
            client = create_provider_client(config.name)
        self._client = client

    def plan(self, prompt: str) -> ModelSpec:
        if not prompt.strip():
            raise ClarificationRequired("Enter a control-system description to plan.")
        response = _request_with_retries(
            lambda: self._client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                text_format=PlannerEnvelope,
            ),
            label="Structured planning request",
            api_key=os.getenv(get_provider_config(self.provider).api_key_env),
            max_retries=self.max_retries,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            detail = getattr(response, "output_text", "") or "The model returned no structured plan."
            raise PlannerError(f"Planner did not return a usable structured response: {detail}")
        return _validated_graph(_parse_envelope(parsed))

    def propose_repair(self, spec: ModelSpec, diagnosis: RepairDiagnosis) -> RepairProposal:
        return _request_repair(
            provider=self.provider,
            model=self.model,
            client=self._client,
            spec=spec,
            diagnosis=diagnosis,
            max_retries=self.max_retries,
        )


class GeminiPlanner:
    """Google Gemini Interactions API adapter with JSON-schema structured output."""

    def __init__(self, model: str | None = None, client: Any | None = None, max_retries: int | None = None):
        config = get_provider_config("gemini")
        self.provider = config.name
        self.model = model or os.getenv("CIRCUITPILOT_MODEL") or config.default_model
        self.max_retries = max_retries
        self.credential_source = config.api_key_env
        self.credential_configured = bool(os.getenv(config.api_key_env))
        self._client = client if client is not None else create_provider_client("gemini")
        _disable_google_genai_retries(self._client)

    @property
    def api_endpoint(self) -> str:
        """Return the Interactions API endpoint without inspecting auth headers."""
        api_client = getattr(self._client, "_api_client", None)
        options = getattr(api_client, "_http_options", None)
        base_url = getattr(options, "base_url", None)
        if base_url:
            return f"{str(base_url).rstrip('/')}/{GEMINI_INTERACTIONS_API_VERSION}"
        return f"https://generativelanguage.googleapis.com/{GEMINI_INTERACTIONS_API_VERSION}"

    def plan(self, prompt: str) -> ModelSpec:
        if not prompt.strip():
            raise ClarificationRequired("Enter a control-system description to plan.")
        response = _request_with_retries(
            lambda: self._client.interactions.create(
                model=self.model,
                input=prompt,
                api_version=GEMINI_INTERACTIONS_API_VERSION,
                system_instruction=SYSTEM_INSTRUCTIONS,
                response_format=[
                    {
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": PlannerEnvelope.model_json_schema(),
                    }
                ],
            ),
            label="Structured planning request",
            api_key=os.getenv("GEMINI_API_KEY"),
            max_retries=self.max_retries,
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise PlannerError("Planner did not return a usable structured response: The model returned no output text.")
        return _validated_graph(_parse_envelope(output_text))

    def propose_repair(self, spec: ModelSpec, diagnosis: RepairDiagnosis) -> RepairProposal:
        return _request_repair(
            provider=self.provider,
            model=self.model,
            client=self._client,
            spec=spec,
            diagnosis=diagnosis,
            max_retries=self.max_retries,
        )

    def diagnose_api(self) -> list[dict[str, Any]]:
        """Probe Interactions and, when needed, GenerateContent using this client/model."""
        results: list[dict[str, Any]] = []
        api_key = os.getenv("GEMINI_API_KEY")

        def record(name: str, operation: Any) -> bool:
            try:
                detail = operation()
                results.append({"probe": name, "success": True, "detail": detail})
                return True
            except Exception as exc:
                results.append({
                    "probe": name,
                    "success": False,
                    "detail": _safe_request_error(exc, api_key=api_key),
                })
                return False

        def plain_interaction() -> str:
            response = self._client.interactions.create(
                model=self.model,
                input="Reply with OK.",
                api_version=GEMINI_INTERACTIONS_API_VERSION,
            )
            output_text = getattr(response, "output_text", None)
            if not output_text:
                raise PlannerError("The API returned no output text.")
            return "Received plain-text output."

        def structured_interaction() -> str:
            response = self._client.interactions.create(
                model=self.model,
                input="Return status as the string 'ok'.",
                api_version=GEMINI_INTERACTIONS_API_VERSION,
                system_instruction=SYSTEM_INSTRUCTIONS,
                response_format=[
                    {
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": GeminiDiagnosticEnvelope.model_json_schema(),
                    }
                ],
            )
            output_text = getattr(response, "output_text", None)
            if not output_text:
                raise PlannerError("The API returned no structured output text.")
            result = GeminiDiagnosticEnvelope.model_validate_json(output_text)
            if result.status != "ok":
                raise PlannerError("The response did not contain status='ok'.")
            return "Received and validated minimal JSON output."

        plain_ok = record("interactions_plain_text", plain_interaction)
        structured_ok = record("interactions_structured_json", structured_interaction)
        if structured_ok:
            def planner_schema_interaction() -> str:
                response = self._client.interactions.create(
                    model=self.model,
                    input=(
                        "Return status='clarification', graph_json='', and message="
                        "'Please specify the plant transfer function.'"
                    ),
                    api_version=GEMINI_INTERACTIONS_API_VERSION,
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    response_format=[
                        {
                            "type": "text",
                            "mime_type": "application/json",
                            "schema": PlannerEnvelope.model_json_schema(),
                        }
                    ],
                )
                output_text = getattr(response, "output_text", None)
                if not output_text:
                    raise PlannerError("The API returned no planner-schema output text.")
                envelope = _parse_envelope(output_text)
                if envelope.status != "clarification":
                    raise PlannerError("The planner-schema probe did not return clarification status.")
                return "Received and validated CircuitPilot envelope output."

            record("interactions_circuitpilot_schema", planner_schema_interaction)
        if not plain_ok and not structured_ok:
            def generate_content() -> str:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents="Reply with OK.",
                )
                output_text = getattr(response, "text", None)
                if not output_text:
                    raise PlannerError("The API returned no GenerateContent text.")
                return "Received GenerateContent text."

            record("generate_content_plain_text", generate_content)
        return results


REPAIR_SYSTEM_INSTRUCTIONS = """You are proposing a bounded repair to an existing CircuitPilot classical-control model.
Return only a RepairProposal JSON object. Do not return a replacement graph, code, MATLAB,
or executable instructions. The only permitted edits are sparse parameter updates to
existing component ids and/or stop_time. Do not change component ids or types, connections,
output selection, verification criteria, or acceptance thresholds. Preserve the model's
intent and make the smallest targeted change justified by failed checks. Never change a
verification criterion to make a model pass. Use only existing supported component
parameters; local validation will reject invalid updates. If the diagnosis does not justify
a safe change, provide no proposal rather than guessing."""


def _request_repair(
    *, provider: str, model: str, client: Any, spec: ModelSpec,
    diagnosis: RepairDiagnosis, max_retries: int | None = None,
) -> RepairProposal:
    request = (
        "Current validated model specification:\n"
        + spec.model_dump_json(indent=2)
        + "\nFailed-check diagnosis:\n"
        + diagnosis.model_dump_json(indent=2)
        + "\nReturn only the smallest targeted structured repair proposal."
    )
    config = get_provider_config(provider)
    if config.sdk == "google":
        request_attempt = 0

        def request_gemini_repair() -> Any:
            nonlocal request_attempt
            request_attempt += 1
            started = time.monotonic()
            _LOGGER.info(
                "Gemini repair request started model=%s retry_attempt=%d",
                model,
                request_attempt,
            )
            try:
                response = client.interactions.create(
                    model=model,
                    input=request,
                    api_version=GEMINI_INTERACTIONS_API_VERSION,
                    system_instruction=REPAIR_SYSTEM_INSTRUCTIONS,
                    response_format=[{
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": RepairProposal.model_json_schema(),
                    }],
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                status = _http_status_code(exc)
                category = "timeout" if _is_timeout_exception(exc) else "provider" if status is not None else "request"
                _LOGGER.warning(
                    "Gemini repair request failed model=%s retry_attempt=%d "
                    "elapsed_seconds=%.3f exception_category=%s http_status=%s",
                    model,
                    request_attempt,
                    elapsed,
                    category,
                    status,
                )
                raise
            elapsed = time.monotonic() - started
            _LOGGER.info(
                "Gemini repair request succeeded model=%s retry_attempt=%d elapsed_seconds=%.3f",
                model,
                request_attempt,
                elapsed,
            )
            return response

        response = _request_with_retries(
            request_gemini_repair,
            label="Structured repair request",
            api_key=os.getenv(config.api_key_env),
            max_retries=max_retries,
        )
        parsed = getattr(response, "output_text", None)
        if not parsed:
            raise PlannerError("The model returned no structured repair proposal.")
        try:
            return RepairProposal.model_validate_json(parsed)
        except ValidationError as exc:
            raise PlannerError(f"Planner returned malformed structured repair output: {exc}") from exc
    response = _request_with_retries(
        lambda: client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": REPAIR_SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": request},
            ],
            text_format=RepairProposal,
        ),
        label="Structured repair request",
        api_key=os.getenv(config.api_key_env),
        max_retries=max_retries,
    )
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise PlannerError("The model returned no structured repair proposal.")
    try:
        return RepairProposal.model_validate(parsed)
    except ValidationError as exc:
        raise PlannerError(f"Planner returned malformed structured repair output: {exc}") from exc
