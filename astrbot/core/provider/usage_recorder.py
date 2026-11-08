"""Record observed provider calls with independent usage and attribution fields."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any

from astrbot import logger
from astrbot.core.provider.entities import DecisionUsage, TokenUsage
from astrbot.core.provider.provider import provider_stats_managed_by_agent


def _valid_token_usage(usage: Any) -> bool:
    """Validate optional side-channel counters before statistics arithmetic.

    Args:
        usage: Untrusted response or exception metadata.

    Returns:
        Whether this is a finite nonnegative TokenUsage supported by the ledger.
    """
    return isinstance(usage, TokenUsage) and all(
        type(value) is int and 0 <= value <= 2**63 - 1
        for value in (usage.input_other, usage.input_cached, usage.output)
    )


@dataclass(frozen=True, slots=True)
class UsageContext:
    session_umo: str | None = None
    trace_id: str | None = None
    plugin_id: str | None = None
    origin_type: str | None = None
    conversation_id: str | None = None
    request_kind: str | None = None


class UsageAttemptLedger:
    """Keep failed-attempt usage separate from the final prompt's token usage."""

    def __init__(self, provider_key: int | None = None) -> None:
        """Initialize an independent logical call ledger.

        Args:
            provider_key: Owning provider identity for nested-call isolation.
        """
        self.provider_key = provider_key
        self.usage: TokenUsage | None = None
        self.unknown = False
        self.errors: set[str] = set()
        self.observed_usage: TokenUsage | None = None
        self.snapshot_complete = False
        self.snapshot_terminal_received = False
        self.snapshot_response_id: str | None = None
        self.snapshot_reported_fields: frozenset[str] | None = None
        self.snapshot_last_usage: TokenUsage | None = None
        self.closed = False

    def begin_attempt(self) -> None:
        """Clear the current snapshot before a distinct backend attempt starts."""
        self.observed_usage = None
        self.snapshot_complete = False
        self.snapshot_terminal_received = False
        self.snapshot_response_id = None
        self.snapshot_reported_fields = None
        self.snapshot_last_usage = None

    def note_failure(self, error: BaseException, *, new_attempt: bool = False) -> None:
        """Observe an exception once, even when several retry layers see it.

        Args:
            error: Failure of one actual or possibly submitted attempt.
            new_attempt: Start a new identity even if a caller reused an exception.
        """
        from astrbot.core.provider.context_budget import ContextBudgetExceeded

        if isinstance(error, ContextBudgetExceeded):
            return
        if getattr(error, "_astrbot_usage_recorded", False):
            return
        identity = getattr(error, "_astrbot_usage_attempt_id", None)
        if (
            new_attempt
            or not isinstance(identity, str)
            or not identity
            or len(identity) > 128
        ):
            identity = str(uuid.uuid4())
            try:
                setattr(error, "_astrbot_usage_attempt_id", identity)
            except (AttributeError, TypeError):
                identity = f"exception:{id(error)}"
        if identity in self.errors:
            return
        self.errors.add(identity)
        usage = getattr(error, "_astrbot_token_usage", None)
        if not _valid_token_usage(usage):
            usage = self.observed_usage
        elif (
            usage.is_partial
            and self.observed_usage is not None
            and (
                self.snapshot_complete
                or (
                    self.snapshot_reported_fields is not None
                    and self.snapshot_last_usage is not None
                    and (usage.input_other, usage.input_cached, usage.output)
                    == (
                        self.snapshot_last_usage.input_other,
                        self.snapshot_last_usage.input_cached,
                        self.snapshot_last_usage.output,
                    )
                )
            )
        ):
            usage = self.observed_usage
        self.observed_usage = None
        self.snapshot_complete = False
        self.snapshot_response_id = None
        self.snapshot_reported_fields = None
        self.snapshot_last_usage = None
        if not _valid_token_usage(usage):
            self.unknown = True
        else:
            self.usage = (self.usage or TokenUsage()) + usage

    def combined(
        self, usage: TokenUsage | None, error: BaseException | None = None
    ) -> TokenUsage | None:
        """Return ledger totals without changing the response's usage object.

        Args:
            usage: Final attempt's reported usage, when available.
            error: Final exception, to avoid counting its observed usage twice.

        Returns:
            Known total marked partial for unknown attempts, or None if all unknown.
        """
        if not _valid_token_usage(usage) or (
            error is not None and getattr(error, "_astrbot_usage_recorded", False)
        ):
            usage = None
        elif (
            usage.is_partial
            and self.observed_usage is not None
            and (
                self.snapshot_complete
                or (
                    self.snapshot_reported_fields is not None
                    and self.snapshot_last_usage is not None
                    and (usage.input_other, usage.input_cached, usage.output)
                    == (
                        self.snapshot_last_usage.input_other,
                        self.snapshot_last_usage.input_cached,
                        self.snapshot_last_usage.output,
                    )
                )
            )
        ):
            usage = self.observed_usage
        identity = getattr(error, "_astrbot_usage_attempt_id", None)
        already_observed = isinstance(identity, str) and identity in self.errors
        if usage is None and not already_observed:
            usage = self.observed_usage
        total = self.usage
        if not already_observed and usage is not None:
            total = (total or TokenUsage()) + usage
        if total is None:
            return None
        return TokenUsage(
            input_other=total.input_other,
            input_cached=total.input_cached,
            output=total.output,
            is_partial=total.is_partial
            or self.unknown
            or (usage is None and not already_observed),
        )


current_attempt_ledger: ContextVar[UsageAttemptLedger | None] = ContextVar(
    "usage_attempt_ledger", default=None
)


def note_failed_attempt(error: BaseException, *, new_attempt: bool = False) -> None:
    """Add a failed attempt to the current request's independent ledger.

    Args:
        error: Exception carrying optional backend-reported token usage.
        new_attempt: Assign a new identity for a newly executed retry attempt.
    """
    ledger = current_attempt_ledger.get()
    if ledger is not None and not ledger.closed:
        ledger.note_failure(error, new_attempt=new_attempt)


def observe_usage_snapshot(
    usage: TokenUsage | None,
    *,
    response_id: str | None = None,
    complete: bool = False,
    reported_fields: Collection[str] | None = None,
) -> None:
    """Publish a request-local snapshot without recording or summing a new call.

    Args:
        usage: Host TokenUsage from actual backend fields, never an estimate.
        response_id: Optional upstream identity for the current response snapshot.
        complete: Whether this snapshot came from a terminal backend event.
        reported_fields: Optional input/output direction presence for partial updates.

    This is a no-op outside an instrumented request or after its cleanup. A
    producer task inherits the mutable request sink when created under the
    provider call. Producers must publish before their close operation returns.
    Repeated snapshots replace, never add. Different response IDs alone do not
    imply extra billable calls; retries use the independent attempt ledger.
    """
    ledger = current_attempt_ledger.get()
    if ledger is None or ledger.closed or not _valid_token_usage(usage):
        return
    fields = (
        frozenset(
            value
            for value in reported_fields
            if isinstance(value, str) and value in {"input", "output"}
        )
        if isinstance(reported_fields, Collection)
        and not isinstance(reported_fields, (str, bytes))
        else None
    )
    if fields == frozenset():
        return
    identity = (
        response_id
        if isinstance(response_id, str) and len(response_id) <= 256
        else None
    )
    same_response = identity == ledger.snapshot_response_id
    fully_reported = (
        bool(complete)
        and not usage.is_partial
        and (fields is None or fields == {"input", "output"})
    )
    if same_response and ledger.snapshot_complete and not fully_reported:
        return
    ledger.snapshot_last_usage = TokenUsage(
        input_other=usage.input_other,
        input_cached=usage.input_cached,
        output=usage.output,
        is_partial=usage.is_partial,
    )
    previous = ledger.observed_usage
    if (
        not fully_reported
        and same_response
        and fields is not None
        and previous is not None
    ):
        ledger.observed_usage = TokenUsage(
            input_other=usage.input_other
            if "input" in fields
            else previous.input_other,
            input_cached=usage.input_cached
            if "input" in fields
            else previous.input_cached,
            output=usage.output if "output" in fields else previous.output,
            is_partial=True,
        )
        ledger.snapshot_reported_fields = (
            ledger.snapshot_reported_fields or frozenset()
        ) | fields
    else:
        ledger.observed_usage = TokenUsage(
            input_other=usage.input_other,
            input_cached=usage.input_cached,
            output=usage.output,
            is_partial=not fully_reported,
        )
        ledger.snapshot_reported_fields = fields or (
            frozenset({"input", "output"}) if fully_reported else None
        )
    ledger.snapshot_complete = fully_reported
    ledger.snapshot_terminal_received = (
        ledger.snapshot_terminal_received if same_response else False
    ) or bool(complete)
    ledger.snapshot_response_id = identity


def observe_attempt_usage(
    usage: TokenUsage | None,
    *,
    complete: bool = False,
    response_id: str | None = None,
    reported_fields: Collection[str] | None = None,
) -> None:
    """Publish an internal OAuth snapshot through the stable request sink.

    Args:
        usage: Validated backend counters.
        complete: Whether the backend event terminated this attempt.
        response_id: Actual response identity when the raw event includes it.
        reported_fields: Input/output directions present in the raw event.
    """
    observe_usage_snapshot(
        usage,
        complete=complete,
        response_id=response_id,
        reported_fields=reported_fields,
    )


_current_usage_context: ContextVar[UsageContext] = ContextVar(
    "current_usage_context", default=UsageContext()
)
_active_providers: ContextVar[frozenset[tuple[int, int]]] = ContextVar(
    "active_usage_providers", default=frozenset()
)


@contextmanager
def usage_scope(**metadata: str | None) -> Iterator[UsageContext]:
    """Temporarily bind attribution to provider calls in the current task.

    Args:
        **metadata: UsageContext fields to set for nested calls.

    Yields:
        The effective immutable usage context.
    """
    context = replace(_current_usage_context.get(), **metadata)
    token = _current_usage_context.set(context)
    try:
        yield context
    finally:
        _current_usage_context.reset(token)


def event_usage_scope(
    event: Any,
    plugin_id: str | None = None,
    origin_type: str = "chat",
    conversation_id: str | None = None,
):
    """Bind a real message event to subsequent provider calls.

    Args:
        event: AstrBot message event with a unified origin and trace span.
        plugin_id: Plugin that initiated the call, if known.
        origin_type: Provenance of the call.
        conversation_id: Optional conversation identifier.

    Returns:
        A context manager for the event's usage scope.
    """
    current = _current_usage_context.get()
    trace = getattr(event, "trace", None)
    return usage_scope(
        session_umo=getattr(event, "unified_msg_origin", None) or current.session_umo,
        trace_id=getattr(trace, "trace_id", None)
        or getattr(trace, "span_id", None)
        or current.trace_id,
        plugin_id=plugin_id or current.plugin_id,
        origin_type=origin_type
        if event is not None
        else current.origin_type or origin_type,
        conversation_id=conversation_id or current.conversation_id,
    )


async def record_usage(
    provider: Any,
    db: Any,
    *,
    usage: TokenUsage | None,
    status: str,
    start_time: float,
    end_time: float,
    request_kind: str = "text",
    model: str | None = None,
    request_id: str | None = None,
    time_to_first_token: float = 0.0,
    usage_status: str | None = None,
    replace_snapshot: bool = False,
    usage_snapshot_fields: frozenset[str] | None = None,
    _context: UsageContext | None = None,
) -> None:
    """Write one observed call without changing its business result.

    Args:
        provider: Provider instance that made the call.
        db: AstrBot database instance.
        usage: Provider reported token usage, or None when unavailable.
        status: Completed, error, or aborted call status.
        start_time: Unix timestamp at invocation start.
        end_time: Unix timestamp at invocation end.
        request_kind: Provider operation kind.
        model: Actual response model, when known.
        request_id: Stable idempotency key for this call.
        time_to_first_token: Seconds until first streamed response.
        usage_status: Explicit usage completeness override.
        replace_snapshot: Opt in to the restricted Realtime terminal-snapshot update.
        usage_snapshot_fields: Input/output/model fields actually present in a snapshot.
    """
    context = _context or _current_usage_context.get()
    if not _valid_token_usage(usage):
        usage = None
    provider_config = getattr(provider, "provider_config", {}) or {}
    usage_status = (
        "missing"
        if usage is None
        else usage_status
        or ("partial" if getattr(usage, "is_partial", False) else "reported")
    )
    token_usage = usage or TokenUsage()
    try:
        await db.insert_provider_stat(
            umo=context.session_umo or "",
            conversation_id=context.conversation_id,
            provider_id=provider_config.get("id", ""),
            provider_model=model or provider.get_model(),
            status=status,
            stats={
                "request_id": request_id or str(uuid.uuid4()),
                "trace_id": context.trace_id,
                "session_umo": context.session_umo,
                "source_id": provider_config.get("provider_source_id")
                or (
                    provider_config.get("oauth_source_id")
                    if request_kind == "transcription"
                    else None
                ),
                "plugin_id": context.plugin_id,
                "request_kind": request_kind,
                "usage_status": usage_status,
                "origin_type": context.origin_type or "background",
                "stat_version": 1,
                "usage_snapshot": replace_snapshot,
                "usage_snapshot_fields": sorted(usage_snapshot_fields or ()),
                "token_usage": {
                    "input_other": token_usage.input_other,
                    "input_cached": token_usage.input_cached,
                    "output": token_usage.output,
                },
                "start_time": start_time,
                "end_time": end_time,
                "time_to_first_token": time_to_first_token,
            },
            agent_type="provider",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Persist provider usage failed: %s", exc, exc_info=True)


def instrument_decision_provider(provider: Any, db: Any) -> Any:
    """Observe direct and service-mediated calls on a managed decision provider.

    Args:
        provider: Decision provider instance to instrument in place.
        db: Database for the usage ledger.

    Returns:
        The same provider instance, preserving its answer and exception contracts.
    """
    original_ask = provider.ask
    if getattr(original_ask, "_astrbot_usage_wrapped", False) is True:
        return provider

    @wraps(original_ask)
    async def wrapped_ask(*args: Any, **kwargs: Any):
        provider_key = (id(asyncio.current_task()), id(provider))
        if provider_key in _active_providers.get():
            return await original_ask(*args, **kwargs)
        call_context = _current_usage_context.get()
        active_token = _active_providers.set(_active_providers.get() | {provider_key})
        # Native Decisions carries all attempts in its own request-local result.
        # Shared retry helpers must not reset or add to an enclosing chat ledger.
        ledger_token = current_attempt_ledger.set(None)
        start_time = time.time()
        response = None
        final_error = None
        status = "completed"
        try:
            response = await original_ask(*args, **kwargs)
            if response is None:
                status = "error"
            return response
        except BaseException as error:
            final_error = error
            status = "aborted" if isinstance(error, asyncio.CancelledError) else "error"
            raise
        finally:
            current_attempt_ledger.reset(ledger_token)
            _active_providers.reset(active_token)
            subject = response if response is not None else final_error
            attempts = getattr(subject, "attempt_usages", ())
            # A nonempty attempt sequence already includes the final response.
            observations = (
                attempts
                if isinstance(attempts, (tuple, list)) and attempts
                else (getattr(subject, "usage", None),)
            )
            total = None
            unknown = False
            for observed in observations:
                if not isinstance(observed, DecisionUsage) or any(
                    value is not None
                    and (type(value) is not int or not 0 <= value <= 2**63 - 1)
                    for value in (
                        observed.input_tokens,
                        observed.cached_tokens,
                        observed.cache_write_tokens,
                        observed.output_tokens,
                        observed.reasoning_tokens,
                        observed.total_tokens,
                    )
                ):
                    unknown = True
                    continue
                input_tokens = observed.input_tokens
                cached_tokens = observed.cached_tokens
                output_tokens = observed.output_tokens
                if (
                    input_tokens is None
                    and cached_tokens is None
                    and output_tokens is None
                ) or (
                    input_tokens is not None
                    and cached_tokens is not None
                    and cached_tokens > input_tokens
                ):
                    unknown = True
                    continue
                # Cache reads and writes are subsets of input; reasoning is a
                # subset of output. Never add those details to their totals.
                usage = TokenUsage(
                    input_other=(input_tokens or 0) - (cached_tokens or 0)
                    if input_tokens is not None
                    else 0,
                    input_cached=cached_tokens or 0,
                    output=output_tokens or 0,
                    is_partial=input_tokens is None
                    or cached_tokens is None
                    or output_tokens is None,
                )
                total = (total or TokenUsage()) + usage
            if total is not None and unknown:
                total.is_partial = True
            await record_usage(
                provider,
                db,
                usage=total,
                status=status,
                start_time=start_time,
                end_time=time.time(),
                request_kind="test"
                if call_context.request_kind == "test"
                else "decision",
                model=getattr(subject, "model", None),
                _context=call_context,
            )

    wrapped_ask._astrbot_usage_wrapped = True
    provider.ask = wrapped_ask
    provider._usage_recording_enabled = True
    provider._usage_recording_db = db
    return provider


def instrument_provider(provider: Any, db: Any) -> Any:
    """Observe public text calls on a managed provider instance.

    Args:
        provider: Provider instance to instrument in place.
        db: Database for the usage ledger.

    Returns:
        The same provider instance.
    """
    original_text_chat = getattr(provider, "text_chat", None)
    if (
        original_text_chat is not None
        and getattr(original_text_chat, "_astrbot_usage_wrapped", False) is not True
    ):

        @wraps(original_text_chat)
        async def wrapped_text_chat(*args: Any, **kwargs: Any):
            provider_key = (id(asyncio.current_task()), id(provider))
            if provider_key in _active_providers.get():
                return await original_text_chat(*args, **kwargs)
            call_context = _current_usage_context.get()
            active_token = _active_providers.set(
                _active_providers.get() | {provider_key}
            )
            legacy_token = provider_stats_managed_by_agent.set(True)
            start_time = time.time()
            ledger = UsageAttemptLedger(id(provider))
            ledger_token = current_attempt_ledger.set(ledger)
            final_error = None
            response = None
            usage = None
            status = "completed"
            try:
                response = await original_text_chat(*args, **kwargs)
                usage = getattr(response, "usage", None)
                if response is None or getattr(response, "role", None) == "err":
                    status = "error"
                return response
            except BaseException as exc:
                final_error = exc
                status = (
                    "aborted" if isinstance(exc, asyncio.CancelledError) else "error"
                )
                usage = (
                    None
                    if getattr(exc, "_astrbot_usage_recorded", False)
                    else getattr(exc, "_astrbot_token_usage", None)
                )
                raise
            finally:
                current_attempt_ledger.reset(ledger_token)
                provider_stats_managed_by_agent.reset(legacy_token)
                _active_providers.reset(active_token)
                raw_completion = getattr(response, "raw_completion", None)
                if response is None and ledger.snapshot_complete:
                    usage = ledger.observed_usage
                ledger.closed = True
                actual_model = (
                    getattr(response, "model", None)
                    or (
                        raw_completion.get("model")
                        if isinstance(raw_completion, dict)
                        else getattr(raw_completion, "model", None)
                    )
                    or getattr(final_error, "_astrbot_provider_model", None)
                    or kwargs.get("model")
                )
                await record_usage(
                    provider,
                    db,
                    usage=ledger.combined(usage, final_error),
                    status=status,
                    start_time=start_time,
                    end_time=time.time(),
                    request_kind=call_context.request_kind or "text",
                    model=actual_model,
                    usage_status="partial"
                    if status == "aborted"
                    and usage is not None
                    and not ledger.snapshot_terminal_received
                    else None,
                    _context=call_context,
                )

        wrapped_text_chat._astrbot_usage_wrapped = True
        provider.text_chat = wrapped_text_chat

    original_stream = getattr(provider, "text_chat_stream", None)
    if (
        original_stream is not None
        and getattr(original_stream, "_astrbot_usage_wrapped", False) is not True
    ):

        @wraps(original_stream)
        async def wrapped_stream(*args: Any, **kwargs: Any):
            provider_key = (id(asyncio.current_task()), id(provider))
            if provider_key in _active_providers.get():
                async for response in original_stream(*args, **kwargs):
                    yield response
                return
            call_context = _current_usage_context.get()
            stream = original_stream(*args, **kwargs)
            start_time = time.time()
            first_token_time = 0.0
            usage = None
            status = "completed"
            response = None
            received_final = False
            final_usage = None
            ledger = UsageAttemptLedger(id(provider))
            final_error = None
            try:
                while True:
                    provider_key = (id(asyncio.current_task()), id(provider))
                    scope_token = _current_usage_context.set(call_context)
                    active_token = _active_providers.set(
                        _active_providers.get() | {provider_key}
                    )
                    legacy_token = provider_stats_managed_by_agent.set(True)
                    ledger_token = current_attempt_ledger.set(ledger)
                    try:
                        response = await anext(stream)
                    except StopAsyncIteration:
                        break
                    finally:
                        current_attempt_ledger.reset(ledger_token)
                        provider_stats_managed_by_agent.reset(legacy_token)
                        _active_providers.reset(active_token)
                        _current_usage_context.reset(scope_token)
                    if first_token_time == 0.0:
                        first_token_time = time.time() - start_time
                    if _valid_token_usage(getattr(response, "usage", None)):
                        usage = response.usage
                    if not getattr(response, "is_chunk", False):
                        received_final = True
                        final_usage = (
                            response.usage
                            if _valid_token_usage(getattr(response, "usage", None))
                            else None
                        )
                    if getattr(response, "role", None) == "err":
                        status = "error"
                    yield response
                if not received_final:
                    status = "error"
            except GeneratorExit:
                if not received_final:
                    status = "aborted"
                raise
            except BaseException as exc:
                final_error = exc
                if isinstance(exc, asyncio.CancelledError):
                    if not received_final:
                        status = "aborted"
                else:
                    status = "error"
                usage = (
                    None
                    if getattr(exc, "_astrbot_usage_recorded", False)
                    else getattr(exc, "_astrbot_token_usage", None)
                ) or usage
                raise
            finally:
                provider_key = (id(asyncio.current_task()), id(provider))
                scope_token = _current_usage_context.set(call_context)
                active_token = _active_providers.set(
                    _active_providers.get() | {provider_key}
                )
                legacy_token = provider_stats_managed_by_agent.set(True)
                ledger_token = current_attempt_ledger.set(ledger)
                try:
                    await stream.aclose()
                finally:
                    current_attempt_ledger.reset(ledger_token)
                    provider_stats_managed_by_agent.reset(legacy_token)
                    _active_providers.reset(active_token)
                    _current_usage_context.reset(scope_token)
                    raw_completion = getattr(response, "raw_completion", None)
                    if final_usage is not None:
                        usage = final_usage
                    elif ledger.observed_usage is not None:
                        usage = ledger.observed_usage
                    ledger.closed = True
                    await record_usage(
                        provider,
                        db,
                        usage=ledger.combined(usage, final_error),
                        status=status,
                        start_time=start_time,
                        end_time=time.time(),
                        request_kind=call_context.request_kind or "text",
                        model=getattr(response, "model", None)
                        or (
                            raw_completion.get("model")
                            if isinstance(raw_completion, dict)
                            else getattr(raw_completion, "model", None)
                        )
                        or getattr(final_error, "_astrbot_provider_model", None)
                        or kwargs.get("model"),
                        time_to_first_token=first_token_time,
                        usage_status="partial"
                        if status in ("aborted", "error")
                        and not received_final
                        and usage is not None
                        and not ledger.snapshot_terminal_received
                        else None,
                        _context=call_context,
                    )

        wrapped_stream._astrbot_usage_wrapped = True
        provider.text_chat_stream = wrapped_stream

    provider._usage_recording_enabled = True
    provider._usage_recording_db = db
    return provider
