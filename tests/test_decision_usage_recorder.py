import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from astrbot.core.provider.decision import DecisionService
from astrbot.core.provider.entities import (
    DecisionBooleanAnswer,
    DecisionBooleanQuestion,
    DecisionError,
    DecisionResult,
    DecisionUsage,
    LLMResponse,
    TokenUsage,
)
from astrbot.core.provider.manager import ProviderManager
from astrbot.core.provider.provider import DecisionProvider
from astrbot.core.provider.sources.openai_decision_source import OpenAIDecisionProvider
from astrbot.core.provider.usage_recorder import (
    current_attempt_ledger,
    instrument_decision_provider,
    instrument_provider,
    observe_usage_snapshot,
    usage_scope,
)

QUESTIONS = {"ready": DecisionBooleanQuestion("Is the service ready?")}


def native_provider(handler):
    return OpenAIDecisionProvider(
        {
            "id": "decisions/luna",
            "provider_source_id": "decisions",
            "type": "openai_decision",
            "api_base": "https://decision.test/v1",
            "key": "synthetic-key",
            "model": "luna",
        },
        {},
        transport=httpx.MockTransport(handler),
    )


def native_response(usage):
    return httpx.Response(
        200,
        json={
            "model": "actual-luna",
            "answers": [{"name": "ready", "type": "predicate", "probability": 0.8}],
            "usage": usage,
        },
    )


@pytest.mark.asyncio
async def test_direct_and_service_calls_record_attributed_usage_without_changing_result():
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    provider = native_provider(
        lambda _: native_response(
            {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 1},
                "total_tokens": 13,
            }
        )
    )
    instrument_decision_provider(provider, db)
    instrument_decision_provider(provider, db)
    manager = SimpleNamespace(get_using_provider_async=AsyncMock(return_value=provider))
    try:
        with usage_scope(
            session_umo="qq:GroupMessage:42", plugin_id="router", trace_id="trace-42"
        ):
            direct = await provider.ask(QUESTIONS, state="healthy")
            service = await DecisionService(manager).ask(QUESTIONS, state="healthy")
        assert isinstance(direct, DecisionResult) and isinstance(
            service, DecisionResult
        )
        assert direct.usage.input_tokens == 10 and direct.usage.output_tokens == 3
        assert direct["ready"] == DecisionBooleanAnswer(0.8)
        assert db.insert_provider_stat.await_count == 2
        first, second = [
            call.kwargs for call in db.insert_provider_stat.await_args_list
        ]
        assert first["umo"] == "qq:GroupMessage:42"
        assert first["stats"]["plugin_id"] == "router"
        assert first["stats"]["trace_id"] == "trace-42"
        assert first["stats"]["source_id"] == "decisions"
        assert first["stats"]["request_kind"] == "decision"
        assert first["stats"]["usage_status"] == "reported"
        assert first["provider_model"] == "actual-luna"
        assert first["stats"]["token_usage"] == {
            "input_other": 6,
            "input_cached": 4,
            "output": 3,
        }
        assert first["stats"]["request_id"] != second["stats"]["request_id"]
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage", "status", "counts"),
    [
        (None, "missing", (0, 0, 0)),
        (DecisionUsage(), "missing", (0, 0, 0)),
        (
            DecisionUsage(input_tokens=0, cached_tokens=0, output_tokens=0),
            "reported",
            (0, 0, 0),
        ),
        (DecisionUsage(input_tokens=10, cached_tokens=4), "partial", (6, 4, 0)),
        (DecisionUsage(input_tokens=10, output_tokens=2), "partial", (10, 0, 2)),
        (DecisionUsage(cached_tokens=3, output_tokens=2), "partial", (0, 3, 2)),
        (DecisionUsage(output_tokens=0), "partial", (0, 0, 0)),
        (DecisionUsage(total_tokens=10), "missing", (0, 0, 0)),
        (
            DecisionUsage(input_tokens=10, cached_tokens=20, output_tokens=0),
            "missing",
            (0, 0, 0),
        ),
    ],
)
async def test_nullable_usage_distinguishes_missing_partial_and_real_zero(
    usage, status, counts
):
    result = DecisionResult({"ready": DecisionBooleanAnswer(0.7)}, "actual", usage)
    provider = SimpleNamespace(
        provider_config={"id": "decision"},
        get_model=lambda: "configured",
        ask=AsyncMock(return_value=result),
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_decision_provider(provider, db)
    assert await provider.ask(QUESTIONS) is result
    entry = db.insert_provider_stat.await_args.kwargs["stats"]
    assert entry["usage_status"] == status
    assert entry["token_usage"] == dict(
        zip(("input_other", "input_cached", "output"), counts)
    )


@pytest.mark.asyncio
async def test_reported_retry_usage_is_counted_once_and_keeps_final_usage_single_attempt():
    hits = 0

    def handler(_):
        nonlocal hits
        hits += 1
        if hits == 1:
            return httpx.Response(
                429,
                json={
                    "usage": {
                        "input_tokens": 3,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 1,
                    }
                },
            )
        return native_response(
            {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens": 0,
            }
        )

    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    provider = native_provider(handler)
    instrument_decision_provider(provider, db)
    try:
        result = await provider.ask(QUESTIONS, request_max_retries=1)
        assert result.usage.input_tokens == 10 and len(result.attempt_usages) == 2
        stats = db.insert_provider_stat.await_args.kwargs["stats"]
        assert stats["token_usage"] == {
            "input_other": 9,
            "input_cached": 4,
            "output": 1,
        }
        assert stats["usage_status"] == "reported"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_decision_retry_does_not_reset_or_pollute_outer_chat_attempt_ledger():
    hits = 0

    def handler(_):
        nonlocal hits
        hits += 1
        if hits == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return native_response(
            {
                "input_tokens": 8,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
            }
        )

    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    decision = native_provider(handler)
    instrument_decision_provider(decision, db)

    async def text_chat():
        ledger = current_attempt_ledger.get()
        observe_usage_snapshot(
            TokenUsage(input_other=5, output=1), response_id="outer", complete=True
        )
        await decision.ask(QUESTIONS, request_max_retries=1)
        assert current_attempt_ledger.get() is ledger
        assert ledger.observed_usage.input_other == 5
        assert ledger.errors == set() and not ledger.unknown
        return LLMResponse(role="assistant", usage=TokenUsage(input_other=5, output=1))

    chat = SimpleNamespace(
        provider_config={"id": "chat"},
        get_model=lambda: "chat-model",
        text_chat=text_chat,
    )
    instrument_provider(chat, db)
    try:
        await chat.text_chat()
        rows = {
            call.kwargs["provider_id"]: call.kwargs
            for call in db.insert_provider_stat.await_args_list
        }
        assert rows["chat"]["stats"]["usage_status"] == "reported"
        assert rows["chat"]["stats"]["token_usage"] == {
            "input_other": 5,
            "input_cached": 0,
            "output": 1,
        }
        assert rows["decisions/luna"]["stats"]["usage_status"] == "partial"
        assert rows["decisions/luna"]["stats"]["token_usage"]["input_other"] == 8
        assert current_attempt_ledger.get() is None
    finally:
        await decision.terminate()


@pytest.mark.asyncio
async def test_cancellation_retains_known_retry_usage_and_propagates():
    second_started = asyncio.Event()
    hits = 0

    async def handler(_):
        nonlocal hits
        hits += 1
        if hits == 1:
            return httpx.Response(
                429,
                json={
                    "usage": {
                        "input_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 0,
                    }
                },
            )
        second_started.set()
        await asyncio.Future()

    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    provider = native_provider(handler)
    instrument_decision_provider(provider, db)
    try:
        task = asyncio.create_task(provider.ask(QUESTIONS, request_max_retries=1))
        await asyncio.wait_for(second_started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await task
        assert len(cancelled.value.attempt_usages) == 2
        record = db.insert_provider_stat.await_args.kwargs
        assert record["status"] == "aborted"
        assert record["stats"]["usage_status"] == "partial"
        assert record["stats"]["token_usage"] == {
            "input_other": 2,
            "input_cached": 0,
            "output": 0,
        }
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_error_retains_reported_usage_without_double_counting_final_attempt():
    provider = native_provider(
        lambda _: httpx.Response(
            400,
            json={
                "usage": {
                    "input_tokens": 7,
                    "input_tokens_details": {"cached_tokens": 1},
                    "output_tokens": 0,
                }
            },
        )
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_decision_provider(provider, db)
    try:
        with pytest.raises(DecisionError):
            await provider.ask(QUESTIONS)
        record = db.insert_provider_stat.await_args.kwargs
        assert record["status"] == "error"
        assert record["stats"]["token_usage"] == {
            "input_other": 6,
            "input_cached": 1,
            "output": 0,
        }
        assert record["stats"]["usage_status"] == "reported"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_concurrent_calls_keep_usage_and_plugin_context_separate():
    async def handler(request):
        state = json.loads(request.content)["input"]
        await asyncio.sleep(0)
        return native_response(
            {
                "input_tokens": {"left": 11, "right": 13}[state],
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
            }
        )

    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    provider = native_provider(handler)
    instrument_decision_provider(provider, db)

    async def call(name):
        with usage_scope(session_umo=f"qq:GroupMessage:{name}", plugin_id=name):
            return await provider.ask(QUESTIONS, state=name)

    try:
        await asyncio.gather(call("left"), call("right"))
        records = {
            call.kwargs["stats"]["plugin_id"]: call.kwargs
            for call in db.insert_provider_stat.await_args_list
        }
        assert records["left"]["umo"] == "qq:GroupMessage:left"
        assert records["right"]["umo"] == "qq:GroupMessage:right"
        assert records["left"]["stats"]["token_usage"]["input_other"] == 11
        assert records["right"]["stats"]["token_usage"]["input_other"] == 13
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_unmetered_decision_and_dashboard_test_have_explicit_provenance():
    class Unmetered(DecisionProvider):
        async def ask(self, questions, state=None, *, request_max_retries=0):
            return {name: DecisionBooleanAnswer(0.8) for name in questions}

    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    provider = Unmetered({"id": "system-one"}, {})
    provider.set_model("jev")
    instrument_decision_provider(provider, db)
    await provider.ask(QUESTIONS)
    await provider.test()
    direct, probe = [
        call.kwargs["stats"] for call in db.insert_provider_stat.await_args_list
    ]
    assert direct["request_kind"] == "decision" and direct["usage_status"] == "missing"
    assert probe["request_kind"] == "test" and probe["origin_type"] == "test"
    assert probe["usage_status"] == "missing"


@pytest.mark.asyncio
async def test_manager_instruments_managed_decision_before_service_or_direct_calls(
    monkeypatch,
):
    config = {"provider": [], "provider_sources": [], "provider_settings": {}}
    acm = MagicMock()
    acm.confs = {"default": config}
    acm.get_conf.return_value = config
    persona = MagicMock()
    persona.default_persona = ""
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    manager = ProviderManager(acm, db, persona)
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(
            **{
                **kwargs,
                "transport": httpx.MockTransport(
                    lambda _: native_response(
                        {
                            "input_tokens": 4,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 0,
                        }
                    )
                ),
            }
        ),
    )
    await manager.load_provider(
        {
            "id": "managed",
            "type": "openai_decision",
            "provider_type": "decision",
            "enable": True,
            "model": "luna",
            "key": "test",
        }
    )
    provider = manager.inst_map["managed"]
    try:
        await provider.ask(QUESTIONS)
        await DecisionService(manager).ask(QUESTIONS)
        assert db.insert_provider_stat.await_count == 2
    finally:
        await provider.terminate()
