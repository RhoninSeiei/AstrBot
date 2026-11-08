"""Bounded Decisions transport observations without request or response content."""

from __future__ import annotations

import json
import re
import time
import uuid
from contextvars import ContextVar
from typing import Any

from astrbot import logger

current_decision_observation_id: ContextVar[str | None] = ContextVar(
    "decision_observation_id", default=None
)

_TRACE_EVENTS = frozenset(
    f"{name}.{stage}"
    for name in (
        "connection.connect_tcp",
        "connection.connect_unix_socket",
        "connection.start_tls",
        "proxy.start_tls",
        "http11.send_request_headers",
        "http11.send_request_body",
        "http11.receive_response_headers",
        "http11.receive_response_body",
        "http11.response_closed",
        "http2.send_connection_init",
        "http2.send_request_headers",
        "http2.send_request_body",
        "http2.receive_response_headers",
        "http2.receive_response_body",
        "http2.response_closed",
    )
    for stage in ("started", "complete", "failed")
)
_EVENTS = _TRACE_EVENTS | {
    "queue_enter",
    "queue_acquired",
    "attempt_start",
    "attempt_failed",
    "body_complete",
    "terminal",
}
_ERROR_TYPES = frozenset(
    (
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadError",
        "WriteError",
        "ConnectError",
        "ProxyError",
        "RemoteProtocolError",
        "LocalProtocolError",
        "DecisionError",
        "_DecisionTransportError",
    )
)


class DecisionObservation:
    """Keep phase and correlation state local to one logical Decisions request."""

    max_events = 128

    def __init__(self, operation: str) -> None:
        from .usage_recorder import _current_usage_context

        candidate = current_decision_observation_id.get()
        self.observation_id = (
            candidate
            if isinstance(candidate, str)
            and re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                candidate,
            )
            else str(uuid.uuid4())
        )
        trace_id = _current_usage_context.get().trace_id
        self.trace_id = (
            trace_id.lower()
            if isinstance(trace_id, str)
            and re.fullmatch(
                r"[0-9a-fA-F]{16}|[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                trace_id,
            )
            else None
        )
        self.operation = (
            operation if operation in {"decisions", "models"} else "unknown"
        )
        self.started = time.monotonic()
        self.phase = "unknown"
        self.attempt = 0
        self.events = 0
        self.suppressed = 0
        self.finished = False
        self.last_status_code: int | None = None
        self.request_role = "unknown"

    def note(
        self,
        event: str,
        *,
        status_code: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Emit only allowlisted phases and safe scalar metadata.

        Args:
            event: Fixed phase name, never request-controlled text.
            status_code: Observed HTTP response status, when available.
            error: Exception whose allowlisted class name may be retained.
        """
        try:
            if event not in _EVENTS or self.finished:
                return
            previous = self.phase
            self.phase = event
            if type(status_code) is int and 100 <= status_code <= 599:
                self.last_status_code = status_code
            if self.events >= self.max_events:
                self.suppressed += 1
                return
            self.events += 1
            fields = {
                "event": event,
                "observation_id": self.observation_id,
                "trace_id": self.trace_id,
                "operation": self.operation,
                "attempt": self.attempt,
                "elapsed_ms": max(0, int((time.monotonic() - self.started) * 1000)),
                "transport_request": self.request_role,
            }
            if type(status_code) is int and 100 <= status_code <= 599:
                fields["status_code"] = status_code
            if event == "attempt_failed":
                fields["phase_at_error"] = previous
            if error is not None:
                name = type(error).__name__
                fields["error_type"] = name if name in _ERROR_TYPES else "other"
            try:
                logger.info(
                    "decision_observation %s", json.dumps(fields, separators=(",", ":"))
                )
            except Exception:
                pass

        except Exception:
            pass

    async def trace(self, event: str, info: dict[str, Any]) -> None:
        """Observe phases, reading only a fixed method discriminator.

        Args:
            event: httpcore's phase name, checked against an exact allowlist.
            info: Metadata read only for request.method, never headers or content.
        """
        try:
            if isinstance(event, str) and event in _TRACE_EVENTS:
                if event.startswith(("connection.", "proxy.")):
                    self.request_role = "connection"
                elif event.endswith("send_request_headers.started"):
                    try:
                        method = getattr(info.get("request"), "method", None)
                        self.request_role = (
                            "proxy_connect"
                            if method == b"CONNECT"
                            else "request"
                            if method in (b"GET", b"POST")
                            else "unknown"
                        )
                    except Exception:
                        self.request_role = "unknown"
                self.note(event)

        except Exception:
            pass

    def terminal(self, outcome: str, *, error: BaseException | None = None) -> None:
        """Emit one terminal record even after the normal event limit is reached.

        Args:
            outcome: Success, error, external cancellation, or own total timeout.
            error: Optional exception; only its allowlisted class name is retained.
        """
        try:
            if self.finished:
                return
            self.finished = True
            fields = {
                "event": "terminal",
                "observation_id": self.observation_id,
                "trace_id": self.trace_id,
                "operation": self.operation,
                "attempt": self.attempt,
                "elapsed_ms": max(0, int((time.monotonic() - self.started) * 1000)),
                "outcome": outcome
                if outcome in {"success", "error", "cancelled", "timeout"}
                else "error",
                "phase": self.phase,
                "events_suppressed": self.suppressed,
                "transport_request": self.request_role,
            }
            if self.last_status_code is not None:
                fields["last_status_code"] = self.last_status_code
            if outcome in {"cancelled", "timeout"}:
                fields[
                    "phase_at_cancel" if outcome == "cancelled" else "phase_at_timeout"
                ] = self.phase
                fields["reason"] = (
                    "external_cancel" if outcome == "cancelled" else "total_timeout"
                )
            if error is not None:
                name = type(error).__name__
                fields["error_type"] = name if name in _ERROR_TYPES else "other"
            try:
                logger.info(
                    "decision_observation %s", json.dumps(fields, separators=(",", ":"))
                )
            except Exception:
                pass

        except Exception:
            pass
