import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from astrbot.core.provider import decision_observation
from astrbot.core.provider.entities import DecisionBooleanQuestion, DecisionError
from astrbot.core.provider.sources.openai_decision_source import OpenAIDecisionProvider
from astrbot.core.provider.usage_recorder import (
    instrument_decision_provider,
    usage_scope,
)

QUESTIONS = {"ok": DecisionBooleanQuestion("CANARY_PRIVATE_PROMPT")}


@pytest.fixture
def records(monkeypatch):
    output = []
    monkeypatch.setattr(
        decision_observation.logger,
        "info",
        lambda template, body: output.append(json.loads(body)),
    )
    return output


def provider(handler=None, **kwargs):
    config = {
        "id": "CANARY_PRIVATE_PROVIDER",
        "type": "openai_decision",
        "model": "CANARY_PRIVATE_MODEL",
        "key": "CANARY_PRIVATE_KEY",
        "api_base": "https://example.test/CANARY_PRIVATE_PATH",
        "custom_headers": {"X-Secret": "CANARY_PRIVATE_HEADER"},
    }
    config.update(kwargs)
    return OpenAIDecisionProvider(
        config, {}, transport=httpx.MockTransport(handler) if handler else None
    )


def response():
    return httpx.Response(
        200,
        json={
            "model": "CANARY_PRIVATE_MODEL",
            "answers": [{"name": "ok", "type": "predicate", "probability": 0.8}],
            "usage": {"input_tokens": 4, "output_tokens": 0},
        },
    )


@pytest.mark.asyncio
async def test_mock_transport_logs_only_known_stages_and_safe_trace(records):
    p = provider(lambda _: response())
    try:
        with usage_scope(trace_id="a" * 32):
            result = await p.ask(QUESTIONS, state="CANARY_PRIVATE_STATE")
        assert result["ok"].probability == 0.8
        assert [r["event"] for r in records] == [
            "queue_enter",
            "queue_acquired",
            "attempt_start",
            "body_complete",
            "terminal",
        ]
        assert {r["trace_id"] for r in records} == {"a" * 32}
        assert len({r["observation_id"] for r in records}) == 1
        assert records[-1]["outcome"] == "success"
        assert records[-1]["attempt"] == 1
        assert "CANARY_PRIVATE" not in json.dumps(records)
        assert all(
            type(r["elapsed_ms"]) is int and r["elapsed_ms"] >= 0 for r in records
        )
    finally:
        await p.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_queue_cancellation_and_own_timeout_are_distinct(records, cancel):
    p = provider(lambda _: response(), max_concurrency=1, total_timeout=0.05)
    await p.request_semaphore.acquire()
    try:
        task = asyncio.create_task(p.ask(QUESTIONS))
        if cancel:
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(DecisionError):
                await task
        terminal = records[-1]
        assert terminal["event"] == "terminal" and terminal["attempt"] == 0
        assert terminal["outcome"] == ("cancelled" if cancel else "timeout")
        assert terminal["reason"] == ("external_cancel" if cancel else "total_timeout")
        assert (
            terminal["phase_at_cancel" if cancel else "phase_at_timeout"]
            == "queue_enter"
        )
    finally:
        p.request_semaphore.release()
        await p.terminate()


@pytest.mark.asyncio
async def test_network_cancel_preserves_phase_and_known_retry_usage(records):
    waiting = asyncio.Event()
    hits = 0

    async def handler(request):
        nonlocal hits
        hits += 1
        if hits == 1:
            return httpx.Response(
                429, json={"usage": {"input_tokens": 2, "output_tokens": 0}}
            )
        await request.extensions["trace"](
            "http11.receive_response_headers.started",
            {"request": "CANARY_PRIVATE_BODY"},
        )
        waiting.set()
        await asyncio.Future()

    p = provider(handler)
    try:
        task = asyncio.create_task(p.ask(QUESTIONS, request_max_retries=1))
        await asyncio.wait_for(waiting.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert len(caught.value.attempt_usages) == 2
        assert caught.value.attempt_usages[0].input_tokens == 2
        assert (
            records[-1]["phase_at_cancel"] == "http11.receive_response_headers.started"
        )
        assert records[-1]["attempt"] == 2
        assert "last_status_code" not in records[-1]
        assert "CANARY_PRIVATE" not in json.dumps(records)
    finally:
        await p.terminate()


@pytest.mark.asyncio
async def test_cancel_after_failed_attempt_does_not_claim_upload_is_still_running(
    records,
):
    failed = asyncio.Event()

    async def handler(request):
        await request.extensions["trace"]("http11.send_request_body.started", {})
        failed.set()
        raise httpx.WriteError("CANARY_PRIVATE_ERROR", request=request)

    p = provider(handler)
    try:
        task = asyncio.create_task(p.ask(QUESTIONS, request_max_retries=1))
        await failed.wait()
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        terminal = records[-1]
        assert terminal["phase_at_cancel"] == "attempt_failed"
        failed_entry = next(r for r in records if r["event"] == "attempt_failed")
        assert failed_entry["phase_at_error"] == "http11.send_request_body.started"
        assert failed_entry["error_type"] == "WriteError"
    finally:
        await p.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception",
    [
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ConnectError,
        httpx.PoolTimeout,
        httpx.ReadError,
        httpx.WriteError,
    ],
)
async def test_transport_errors_only_log_allowlisted_types(records, exception):
    def handler(request):
        raise exception("CANARY_PRIVATE_EXCEPTION_URL_AND_BODY", request=request)

    p = provider(handler)
    try:
        with pytest.raises(DecisionError):
            await p.ask(QUESTIONS)
        assert any(r.get("error_type") == exception.__name__ for r in records)
        assert records[-1]["outcome"] == "error"
        assert "CANARY_PRIVATE" not in json.dumps(records)
    finally:
        await p.terminate()


@pytest.mark.asyncio
async def test_invalid_answer_is_terminal_error_not_transport_success(records):
    p = provider(lambda _: httpx.Response(200, json={"model": "m", "answers": []}))
    try:
        with pytest.raises(DecisionError):
            await p.ask(QUESTIONS)
        assert [r["outcome"] for r in records if r["event"] == "terminal"] == ["error"]
    finally:
        await p.terminate()


@pytest.mark.asyncio
async def test_inner_timeout_does_not_claim_the_total_deadline_expired(records):
    def handler(_):
        raise TimeoutError("CANARY_PRIVATE_INNER_TIMEOUT")

    p = provider(handler, total_timeout=10)
    try:
        with pytest.raises(DecisionError):
            await p.ask(QUESTIONS)
        assert records[-1]["outcome"] == "error"
        assert "reason" not in records[-1]
        assert "CANARY_PRIVATE" not in json.dumps(records)
    finally:
        await p.terminate()


@pytest.mark.asyncio
async def test_concurrent_observations_and_log_failure_do_not_change_calls(
    records, monkeypatch
):
    p = provider(lambda _: response())
    try:

        async def call(trace):
            with usage_scope(trace_id=trace):
                return await p.ask(QUESTIONS)

        await asyncio.gather(call("1" * 32), call("2" * 32))
        groups = defaultdict(list)
        for record in records:
            groups[record["observation_id"]].append(record)
        assert len(groups) == 2
        assert {tuple({r["trace_id"] for r in group}) for group in groups.values()} == {
            ("1" * 32,),
            ("2" * 32,),
        }

        def broken_logger(*args, **kwargs):
            raise RuntimeError("CANARY_PRIVATE_LOGGER_FAILURE")

        monkeypatch.setattr(decision_observation.logger, "info", broken_logger)
        assert (await p.ask(QUESTIONS))["ok"].probability == 0.8
    finally:
        await p.terminate()


@pytest.mark.asyncio
async def test_trace_allowlist_and_per_request_limit(records):
    async def handler(request):
        trace = request.extensions["trace"]
        await trace("CANARY_PRIVATE_EVENT", {"headers": "CANARY_PRIVATE_HEADER"})
        await trace(
            {"CANARY_PRIVATE_EVENT": True}, {"headers": "CANARY_PRIVATE_HEADER"}
        )
        for _ in range(200):
            await trace(
                "http2.send_request_body.started", {"request": "CANARY_PRIVATE_BODY"}
            )
        return response()

    p = provider(handler)
    try:
        with usage_scope(trace_id="CANARY_PRIVATE_TRACE\nsecret"):
            await p.ask(QUESTIONS)
        assert len(records) <= 129
        assert records[-1]["event"] == "terminal"
        assert records[-1]["events_suppressed"] > 0
        assert "CANARY_PRIVATE" not in json.dumps(records)
    finally:
        await p.terminate()


@pytest.mark.asyncio
async def test_managed_concurrent_calls_use_exact_database_ids_and_restore_parent_context(
    records,
):
    p = provider(lambda _: response())
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_decision_provider(p, db)
    token = decision_observation.current_decision_observation_id.set(
        "CANARY_PRIVATE_PARENT"
    )
    try:
        with usage_scope(trace_id="3" * 32):
            await asyncio.gather(p.ask(QUESTIONS), p.ask(QUESTIONS))
        observed = {r["observation_id"] for r in records}
        stored = {
            call.kwargs["stats"]["request_id"]
            for call in db.insert_provider_stat.await_args_list
        }
        assert len(observed) == 2 and observed == stored
        assert (
            decision_observation.current_decision_observation_id.get()
            == "CANARY_PRIVATE_PARENT"
        )
        assert "CANARY_PRIVATE" not in json.dumps(records)
    finally:
        decision_observation.current_decision_observation_id.reset(token)
        await p.terminate()


@pytest.mark.asyncio
async def test_proxy_connect_is_not_labeled_as_application_request_body(records):
    async def handler(request):
        trace = request.extensions["trace"]
        await trace(
            "http11.send_request_headers.started",
            {"request": SimpleNamespace(method=b"CONNECT")},
        )
        await trace("http11.send_request_body.started", {"request": "CANARY_PRIVATE"})
        await trace(
            "http11.send_request_body.complete", {"return_value": "CANARY_PRIVATE"}
        )
        await trace(
            "connection.start_tls.started", {"server_hostname": "CANARY_PRIVATE"}
        )
        await trace("connection.start_tls.complete", {"return_value": "CANARY_PRIVATE"})
        await trace(
            "http11.send_request_headers.started",
            {"request": SimpleNamespace(method=b"POST")},
        )
        await trace("http11.send_request_body.started", {"request": "CANARY_PRIVATE"})
        return response()

    p = provider(handler)
    try:
        await p.ask(QUESTIONS)
        body_starts = [
            r for r in records if r["event"] == "http11.send_request_body.started"
        ]
        assert [r["transport_request"] for r in body_starts] == [
            "proxy_connect",
            "request",
        ]
        assert "CANARY_PRIVATE" not in json.dumps(records)
    finally:
        await p.terminate()


@pytest.mark.asyncio
async def test_late_attempt_trace_is_ignored_and_internal_observer_failure_is_nonfatal(
    records, monkeypatch
):
    old_trace = None

    async def handler(request):
        nonlocal old_trace
        if old_trace is None:
            old_trace = request.extensions["trace"]
            return httpx.Response(429, json={})
        await old_trace("http11.send_request_body.started", {})
        return response()

    p = provider(handler)
    try:
        await p.ask(QUESTIONS, request_max_retries=1)
        assert not any(
            r["event"] == "http11.send_request_body.started" for r in records
        )
        monkeypatch.setattr(
            decision_observation.DecisionObservation,
            "max_events",
            "invalid-observer-state",
        )
        assert (await p.ask(QUESTIONS))["ok"].probability == 0.8
    finally:
        await p.terminate()


@pytest.mark.asyncio
async def test_logger_failure_does_not_replace_external_cancellation(monkeypatch):
    waiting = asyncio.Event()

    async def handler(_):
        waiting.set()
        await asyncio.Future()

    def broken_logger(*args, **kwargs):
        raise RuntimeError("CANARY_PRIVATE_LOGGER_FAILURE")

    monkeypatch.setattr(decision_observation.logger, "info", broken_logger)
    p = provider(handler)
    try:
        task = asyncio.create_task(p.ask(QUESTIONS))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.attempt_usages == (None,)
    finally:
        await p.terminate()


@pytest.mark.asyncio
async def test_transport_callback_failure_does_not_replace_business_result(
    records, monkeypatch
):
    async def handler(request):
        await request.extensions["trace"]("http11.send_request_body.started", {})
        return response()

    monkeypatch.setattr(
        decision_observation.DecisionObservation,
        "trace",
        AsyncMock(side_effect=RuntimeError("CANARY_PRIVATE_CALLBACK_FAILURE")),
    )
    p = provider(handler)
    try:
        assert (await p.ask(QUESTIONS))["ok"].probability == 0.8
        assert records[-1]["outcome"] == "success"
        assert "CANARY_PRIVATE" not in json.dumps(records)
    finally:
        await p.terminate()


@pytest.mark.asyncio
async def test_local_connect_proxy_records_tunnel_and_tls_without_model_request(
    records,
):
    proxy_requests = []

    async def proxy(reader, writer):
        try:
            proxy_requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await writer.drain()
            await reader.read(4096)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(proxy, "127.0.0.1", 0)
    p = provider(
        api_base="https://unreachable.test/v1",
        proxy=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        timeout=1,
        total_timeout=2,
    )
    try:
        with pytest.raises(DecisionError):
            await p.ask(QUESTIONS)
        assert proxy_requests and proxy_requests[0].startswith(b"CONNECT ")
        body_events = [
            r for r in records if r["event"] == "http11.send_request_body.started"
        ]
        assert body_events and all(
            r["transport_request"] == "proxy_connect" for r in body_events
        )
        assert any(r["event"] == "proxy.start_tls.started" for r in records)
        assert not any(r["transport_request"] == "request" for r in records)
        assert "CANARY_PRIVATE" not in json.dumps(records)
        assert b"CANARY_PRIVATE_KEY" not in b"".join(proxy_requests)
    finally:
        await p.terminate()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_local_http_server_emits_real_httpcore_stages(records):
    requests = []

    async def handler(reader, writer):
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            requests.append(await reader.readexactly(length))
            body = json.dumps(
                {
                    "model": "local",
                    "answers": [
                        {"name": "ok", "type": "predicate", "probability": 0.8}
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    p = provider(api_base=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1")
    try:
        await p.ask(QUESTIONS, state="CANARY_PRIVATE_REAL_BODY")
        names = [r["event"] for r in records]
        for name in [
            "connection.connect_tcp.started",
            "connection.connect_tcp.complete",
            "http11.send_request_body.started",
            "http11.send_request_body.complete",
            "http11.receive_response_headers.started",
            "http11.receive_response_headers.complete",
            "body_complete",
            "terminal",
        ]:
            assert name in names
        assert names.index("http11.send_request_body.complete") < names.index(
            "http11.receive_response_headers.complete"
        )
        assert requests and b"CANARY_PRIVATE_REAL_BODY" in requests[0]
        assert "CANARY_PRIVATE" not in json.dumps(records)
    finally:
        await p.terminate()
        server.close()
        await server.wait_closed()
