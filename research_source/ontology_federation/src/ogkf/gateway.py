from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import time
import uuid
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator

from .models import digest


class GatewayError(RuntimeError):
    pass


class Gateway:
    """One concurrent live call across processes sharing the existing MARA lock.

    Network/timeout outcomes are ambiguous and never retried automatically. Only
    received retryable statuses are retried. No SDK retries or silent repair.
    """

    def __init__(
        self, recorder, config, *, client=None, sleep=time.sleep, capabilities=None
    ):
        if recorder.mode not in {"live", "mock"}:
            raise ValueError("gateway mode must be explicitly live or mock")
        if recorder.mode == "mock" and client is None:
            raise ValueError("mock gateway requires an injected offline client")
        self.recorder, self.config, self.sleep = recorder, config, sleep
        self.key = os.getenv("MARA_API_KEY") if recorder.mode == "live" else "mock"
        self.base = os.getenv("MARA_BASE_URL", "https://api.cloud.mara.com/v1").rstrip(
            "/"
        )
        parsed = urlsplit(self.base)
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("HTTPS base URL without credentials/query required")
        self.lock_path = os.getenv("MARA_ADMISSION_LOCK")
        if recorder.mode == "live" and (not self.key or not self.lock_path):
            raise GatewayError(
                "MARA_API_KEY and MARA_ADMISSION_LOCK environment variables required"
            )
        self.client = client or httpx.Client(
            timeout=config.timeout_seconds, follow_redirects=False
        )
        self.capabilities = capabilities or {}
        self.logical_calls = 0
        self.auth_failed = False

    def close(self):
        self.client.close()

    def redact(self, value):
        if self.recorder.mode != "live":
            return value
        if isinstance(value, str):
            return value.replace(self.key, "[REDACTED]")
        if isinstance(value, dict):
            return {self.redact(k): self.redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        return value

    @contextmanager
    def admission(self):
        if self.recorder.mode != "live":
            yield
            return
        path = Path(self.lock_path)
        # Configure this to the existing finder_multi_instance parent lock.
        with path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if (
                    any(path.parent.glob("*/http-state/*.pending"))
                    or path.with_suffix(".pending").exists()
                ):
                    raise GatewayError(
                        "Unresolved MARA outcome; inspect pending receipts before admission"
                    )
                self.sleep(self.config.min_interval_seconds)
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def request(self, body, metadata, *, path="/chat/completions"):
        if self.auth_failed:
            raise GatewayError(
                "provider authorization/quota circuit open; inspect terminal response"
            )
        if self.logical_calls >= self.config.max_logical_calls:
            raise GatewayError("logical call budget exhausted")
        if path not in ("/chat/completions", "/models"):
            raise ValueError("unsupported API path")
        self.logical_calls += 1
        request_id = str(uuid.uuid4())
        trace_id = metadata.get("trace_id", str(uuid.uuid4()))
        with self.admission():
            pending = (
                Path(self.lock_path).with_suffix(".pending")
                if self.recorder.mode == "live"
                else None
            )
            for attempt in range(self.config.max_attempts):
                request_hash = digest({"path": path, "body": body})
                if pending:
                    with pending.open("x") as f:
                        json.dump(
                            {
                                "run_id": self.recorder.run_id,
                                "request_id": request_id,
                                "request_hash": request_hash,
                                "attempt": attempt,
                            },
                            f,
                        )
                start = time.perf_counter()
                status, raw, parsed, error, headers = None, "", None, None, {}
                try:
                    response = self.client.request(
                        "GET" if body is None else "POST",
                        self.base + path,
                        json=body,
                        headers={"Authorization": f"Bearer {self.key}"},
                        timeout=self.config.timeout_seconds,
                    )
                    status, raw = response.status_code, response.text
                    headers = {
                        k: v
                        for k, v in response.headers.items()
                        if k.lower().startswith(("x-ratelimit", "ratelimit"))
                        or k.lower() == "retry-after"
                    }
                    try:
                        parsed = response.json()
                    except ValueError:
                        error = "invalid_response_json"
                except httpx.TransportError as exc:
                    error = type(exc).__name__
                latency = (time.perf_counter() - start) * 1000
                # Redact possible credential reflection even in provider error bodies.
                if self.recorder.mode == "live":
                    raw = self.redact(raw)
                    parsed = self.redact(parsed)
                usage = parsed.get("usage") if isinstance(parsed, dict) else None
                usage = usage if isinstance(usage, dict) else None
                row = {
                    k: metadata.get(k)
                    for k in (
                        "parent_step_id",
                        "experiment_condition_id",
                        "model_id",
                        "prompt_version",
                        "ontology_version",
                        "document_id",
                        "source_publisher",
                    )
                }
                row.update(
                    trace_id=trace_id,
                    request_id=request_id,
                    attempt=attempt,
                    retry_count=attempt,
                    llm_provider="mara" if self.recorder.mode == "live" else "mock",
                    graph_provider=None,
                    request_hash=request_hash,
                    http_status=status,
                    input_tokens=(usage or {}).get("prompt_tokens"),
                    output_tokens=(usage or {}).get("completion_tokens"),
                    provider_reported_usage=json.dumps(usage)
                    if usage is not None
                    else None,
                    rate_limit_headers=json.dumps(headers),
                    http_latency_ms=latency,
                    error=error,
                    validation_status="transport_only",
                )
                self.recorder.attempt(
                    row,
                    {
                        "request": self.redact(body),
                        "response_text": raw,
                        "response": parsed,
                        "http_status": status,
                        "error": error,
                        "headers": headers,
                    },
                )
                if pending and status is not None:
                    pending.unlink()
                if status is None or (error and 200 <= status < 300):
                    raise GatewayError(error or "unknown transport outcome")
                if 200 <= status < 300:
                    return {
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "response": parsed,
                    }
                if (
                    status not in {408, 429, 500, 502, 503, 504}
                    or attempt + 1 == self.config.max_attempts
                ):
                    if status in {401, 402, 403}:
                        self.auth_failed = True
                    raise GatewayError(f"HTTP {status}; request_id={request_id}")
                try:
                    retry_after = float(headers.get("retry-after", "0"))
                except ValueError:
                    retry_after = 0
                if retry_after > 30:
                    raise GatewayError("Retry-After exceeds bounded wait; stop")
                self.sleep(max(0, retry_after, min(8, 2**attempt + random.random())))

    def validate_parameters(self, parameters):
        allowed = {"temperature", "max_tokens", "top_p", "seed", "reasoning_effort"}
        if set(parameters) - allowed:
            raise ValueError("unregistered decoding parameter")
        if parameters.get("reasoning_effort", "low") != "low":
            raise ValueError("only receipt-verified low reasoning effort is registered")
        output_cap = getattr(self.config, "allowed_output_token_cap", 4096)
        if not isinstance(output_cap, int) or not 1 <= output_cap <= 16384:
            raise ValueError("explicit output cap must be 1..16384")
        if (
            "max_tokens" in parameters
            and not 1 <= parameters["max_tokens"] <= output_cap
        ):
            raise ValueError(f"output token limit must be 1..{output_cap}")

    def generate(self, messages, model, parameters, run_metadata):
        self.validate_parameters(parameters)
        if "reasoning_effort" in parameters and not self.capabilities.get(
            model, {}
        ).get("reasoning_effort_low_accepted"):
            raise GatewayError("reasoning effort capability not verified")
        return self.request(
            {"model": model, "messages": messages, **parameters},
            {**run_metadata, "model_id": model},
        )

    def generate_structured(self, messages, schema, model, parameters, run_metadata):
        self.validate_parameters(parameters)
        capability = self.capabilities.get(model, {})
        if "reasoning_effort" in parameters and not capability.get(
            "reasoning_effort_low_accepted"
        ):
            raise GatewayError("reasoning effort capability not verified")
        body = {"model": model, "messages": messages, **parameters}
        # Fixed user-visible structural schema remains present in every arm.
        if capability.get("json_schema") is True:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": True, "schema": schema},
            }
        elif capability.get("json_object") is True:
            body["response_format"] = {"type": "json_object"}
        result = self.request(body, {**run_metadata, "model_id": model})
        try:
            value = json.loads(result["response"]["choices"][0]["message"]["content"])
            Draft202012Validator(schema).validate(value)
        except Exception as exc:
            # Preserve invalid responses, never repair/retry them implicitly.
            self.recorder.write(
                f"rejected/{result['request_id']}.json",
                {"reason": type(exc).__name__, "request_id": result["request_id"]},
            )
            self.recorder.write(
                f"validation/{result['request_id']}.json",
                {
                    "request_id": result["request_id"],
                    "trace_id": result["trace_id"],
                    "validation_status": "schema_rejected",
                },
            )
            raise GatewayError("structured_output_validation_failed") from exc
        self.recorder.write(f"accepted/{result['request_id']}.json", value)
        self.recorder.write(
            f"validation/{result['request_id']}.json",
            {
                "request_id": result["request_id"],
                "trace_id": result["trace_id"],
                "validation_status": "schema_accepted",
            },
        )
        return {**result, "value": value}

    def execute_with_tools(self, messages, tools, model, parameters, run_metadata):
        self.validate_parameters(parameters)
        if "reasoning_effort" in parameters and not self.capabilities.get(
            model, {}
        ).get("reasoning_effort_low_accepted"):
            raise GatewayError("reasoning effort capability not verified")
        if self.capabilities.get(model, {}).get("tool_calling") is not True:
            raise GatewayError("tool capability not verified")
        result = self.request(
            {"model": model, "messages": messages, "tools": tools, **parameters},
            {**run_metadata, "model_id": model},
        )
        try:
            calls = result["response"]["choices"][0]["message"]["tool_calls"]
            by_name = {
                t["function"]["name"]: t["function"]["parameters"] for t in tools
            }
            for call in calls:
                fn = call["function"]
                Draft202012Validator(by_name[fn["name"]]).validate(
                    json.loads(fn["arguments"])
                )
            if not calls:
                raise ValueError("empty tool calls")
        except Exception as exc:
            self.recorder.write(
                f"rejected/{result['request_id']}.json", {"reason": type(exc).__name__}
            )
            raise GatewayError("tool_output_validation_failed") from exc
        # This interface returns validated requests; it never invokes arbitrary tools.
        return {**result, "tool_calls": calls}
