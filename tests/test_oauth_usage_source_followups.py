"""Source usage survives failed attempts without changing response semantics."""

import asyncio
import json
import time
import wave
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import astrbot.core.provider.sources.openai_oauth_source as source
import astrbot.core.provider.sources.request_retry as retry
from astrbot.core.provider.entities import LLMResponse, TokenUsage
from astrbot.core.provider.usage_recorder import instrument_provider, usage_scope
from tests.test_openai_oauth_image_websocket import (
    Session,
    Socket,
    image_events,
    install_sessions,
)
from tests.test_openai_oauth_realtime import (
    SDP,
    FakeProvider,
    FakeWebSocket,
    _install_fakes,
)
from tests.test_openai_oauth_source import _make_provider


def prepared_provider(monkeypatch):
    provider = _make_provider({"provider_source_id": "source-test"})
    monkeypatch.setattr(provider, "_ensure_fresh_oauth_token", AsyncMock())
    state = provider._oauth_shared_state
    state.models_cache_key = (provider.base_url, provider.account_id)
    state.models_cache = [provider.get_model()]
    state.models_expires_at = time.monotonic() + 300
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_provider(provider, db)
    return provider, db


def usage_error(status=503, usage=None):
    error = retry.ProviderRequestError("safe upstream error", status_code=status)
    if usage is not None:
        error._astrbot_token_usage = usage
    return error


@pytest.mark.asyncio
async def test_generate_image_error_records_attached_usage(monkeypatch):
    provider, db = prepared_provider(monkeypatch)
    error = usage_error(429, TokenUsage(input_other=7, output=2))
    monkeypatch.setattr(
        provider, "_request_image_backend", AsyncMock(side_effect=error)
    )
    try:
        with pytest.raises(retry.ProviderRequestError):
            await provider.generate_image("cat")
        assert db.insert_provider_stat.await_count == 1
        stats = db.insert_provider_stat.await_args.kwargs["stats"]
        assert stats["token_usage"] == {
            "input_other": 7,
            "input_cached": 0,
            "output": 2,
        }
        assert stats["usage_status"] == "reported"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_ws_error_terminal_and_rewrapped_parse_failure_preserve_usage(
    monkeypatch,
):
    provider, db = prepared_provider(monkeypatch)
    session = Session(
        Socket(
            [
                {
                    "type": "response.failed",
                    "response": {
                        "usage": {"input_tokens": 7, "output_tokens": 2},
                        "error": {"status_code": 429},
                    },
                }
            ]
        )
    )
    install_sessions(monkeypatch, [session])
    try:
        with pytest.raises(source.OpenAIOAuthImageStreamError) as caught:
            await provider.generate_image("cat", transport="websocket")
        assert caught.value._astrbot_token_usage.total == 9
        assert (
            db.insert_provider_stat.await_args.kwargs["stats"]["token_usage"][
                "input_other"
            ]
            == 7
        )
        session = Session(Socket(image_events()))
        install_sessions(monkeypatch, [session])
        error = ValueError("safe parse error")
        error._astrbot_token_usage = TokenUsage(input_other=3, output=1)
        monkeypatch.setattr(
            provider, "_parse_backend_response", Mock(side_effect=error)
        )
        with pytest.raises(source.OpenAIOAuthImageStreamError) as caught:
            await provider._request_image_backend_websocket({})
        assert caught.value._astrbot_token_usage.total == 4
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "stream", "image"])
@pytest.mark.parametrize(
    "raw_usage,expected",
    [
        ({"input_tokens": 8, "output_tokens": 2}, 10),
        ({"input_tokens": 0, "output_tokens": 0}, 0),
        ({"duration": 3.2}, None),
    ],
)
async def test_non_2xx_body_usage_is_attached_without_raw_body(
    monkeypatch, mode, raw_usage, expected
):
    provider, _ = prepared_provider(monkeypatch)

    @asynccontextmanager
    async def client(**kwargs):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    429, json={"error": {"message": "SECRET"}, "usage": raw_usage}
                )
            )
        ) as opened:
            yield opened

    monkeypatch.setattr(provider, "_open_oauth_http_client", client)
    try:
        with pytest.raises(retry.ProviderRequestError) as caught:
            if mode == "stream":
                _ = [event async for event in provider._stream_backend_events({})]
            elif mode == "image":
                await provider._request_image_backend({})
            else:
                await provider._request_backend({})
        usage = getattr(caught.value, "_astrbot_token_usage", None)
        assert (usage.total if usage is not None else None) == expected
        assert "SECRET" not in str(caught.value)
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", ["reported", "partial", "missing"])
async def test_retry_usage_accumulates_only_in_ledger_and_preserves_response_usage(
    monkeypatch, prior
):
    provider, db = prepared_provider(monkeypatch)
    monkeypatch.setattr(retry, "REQUEST_RETRY_WAIT_MIN_S", 0)
    monkeypatch.setattr(retry, "REQUEST_RETRY_WAIT_MAX_S", 0)
    first = (
        None
        if prior == "missing"
        else TokenUsage(input_other=7, output=2, is_partial=prior == "partial")
    )
    error = usage_error(503, first)
    monkeypatch.setattr(
        provider,
        "_request_backend",
        AsyncMock(
            side_effect=[
                error,
                {
                    "output_text": "ok",
                    "output": [],
                    "usage": {"input_tokens": 4, "output_tokens": 1},
                },
            ]
        ),
    )
    try:
        with usage_scope(session_umo="qq:GroupMessage:1", plugin_id="plugin-test"):
            response = await provider.text_chat("hello", request_max_retries=2)
        assert response.usage.total == 5
        assert db.insert_provider_stat.await_count == 1
        record = db.insert_provider_stat.await_args.kwargs
        assert record["stats"]["token_usage"] == {
            "input_other": 4 + (7 if first is not None else 0),
            "input_cached": 0,
            "output": 1 + (2 if first is not None else 0),
        }
        assert record["stats"]["usage_status"] == (
            "reported" if prior == "reported" else "partial"
        )
        assert record["stats"]["plugin_id"] == "plugin-test"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_final_retried_error_is_not_added_twice_and_equal_attempts_are_distinct(
    monkeypatch,
):
    provider, db = prepared_provider(monkeypatch)
    monkeypatch.setattr(retry, "REQUEST_RETRY_WAIT_MIN_S", 0)
    monkeypatch.setattr(retry, "REQUEST_RETRY_WAIT_MAX_S", 0)
    errors = [usage_error(503, TokenUsage(input_other=7, output=2)) for _ in range(2)]
    monkeypatch.setattr(provider, "_request_backend", AsyncMock(side_effect=errors))
    monkeypatch.setattr(
        provider, "_handle_api_error", AsyncMock(side_effect=errors[-1])
    )
    try:
        with pytest.raises(retry.ProviderRequestError):
            await provider.text_chat("hello", request_max_retries=2)
        stats = db.insert_provider_stat.await_args.kwargs["stats"]
        assert stats["token_usage"] == {
            "input_other": 14,
            "input_cached": 0,
            "output": 4,
        }
        assert db.insert_provider_stat.await_count == 1
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_usage_survives_cancel_during_completion_parse(monkeypatch):
    provider, db = prepared_provider(monkeypatch)
    monkeypatch.setattr(
        provider,
        "_request_backend",
        AsyncMock(
            return_value={
                "output": [],
                "usage": {"input_tokens": 4, "output_tokens": 1},
            }
        ),
    )
    monkeypatch.setattr(
        provider,
        "_parse_responses_completion",
        AsyncMock(side_effect=asyncio.CancelledError),
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await provider.text_chat("hello")
        record = db.insert_provider_stat.await_args.kwargs
        assert record["status"] == "aborted"
        assert record["stats"]["token_usage"]["input_other"] == 4
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage,expected_status",
    [
        ({"input_tokens": 8, "output_tokens": 2}, "reported"),
        ({"type": "duration", "seconds": 1.2}, "missing"),
        ({"input_tokens": 8}, "partial"),
    ],
)
async def test_transcription_records_only_real_tokens_and_keeps_chat_separate(
    monkeypatch, tmp_path, usage, expected_status
):
    provider, db = prepared_provider(monkeypatch)
    path = tmp_path / "test.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 160)
    from astrbot.core.provider.oauth.openai_oauth_transcription import (
        OpenAIOAuthTranscriptionClient,
    )

    monkeypatch.setattr(
        OpenAIOAuthTranscriptionClient,
        "_post_transcription",
        AsyncMock(
            return_value=(
                200,
                {"text": "hello", "model": "actual-asr", "usage": usage},
                1,
            )
        ),
    )
    monkeypatch.setattr(
        provider,
        "_request_backend",
        AsyncMock(
            return_value={
                "output_text": "ok",
                "output": [],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }
        ),
    )
    try:
        with usage_scope(session_umo="wechat:friend", plugin_id="voice-plugin"):
            assert await provider.transcribe_audio(str(path)) == "hello"
            await provider.text_chat("hello")
        assert db.insert_provider_stat.await_count == 2
        rows = [call.kwargs for call in db.insert_provider_stat.await_args_list]
        assert rows[0]["stats"]["request_kind"] == "transcription"
        assert rows[0]["provider_model"] == "actual-asr"
        assert rows[0]["stats"]["usage_status"] == expected_status
        assert rows[0]["stats"]["plugin_id"] == "voice-plugin"
        assert rows[1]["stats"]["token_usage"]["input_other"] == 3
        assert rows[1]["stats"]["request_kind"] == "text"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_realtime_done_snapshots_are_deduplicated_and_keep_creation_attribution(
    monkeypatch, temp_db
):
    from astrbot.core.provider.oauth.openai_oauth_realtime import (
        OpenAIOAuthRealtimeClient,
    )

    provider = FakeProvider(
        [
            httpx.Response(
                200, text=SDP, headers={"Location": "/v1/realtime/calls/rtc_usage"}
            )
        ]
    )
    db = temp_db
    await db.initialize()
    provider.provider_config.update({"id": "rt", "provider_source_id": "source-test"})
    provider.get_model = lambda: "gpt-live-1-codex"
    provider._extract_response_usage = lambda usage: (
        source.ProviderOpenAIOAuth._extract_response_usage(provider, usage)
    )
    instrument_provider(provider, db)
    socket = FakeWebSocket()
    _install_fakes(monkeypatch, provider, [socket])
    client = OpenAIOAuthRealtimeClient(provider)
    try:
        with usage_scope(session_umo="voice:group", plugin_id="realtime-plugin"):
            session = await client.create_session(SDP)
        for event in [
            {
                "type": "response.done",
                "response": {
                    "id": "response-one",
                    "model": "actual-live",
                    "usage": {"input_tokens": 4},
                },
            },
            {
                "type": "response.done",
                "response": {
                    "id": "response-one",
                    "model": "actual-live",
                    "usage": {"input_tokens": 7, "output_tokens": 2},
                },
            },
            {"type": "turn.done", "turn": {"id": "turn-two", "duration": 2.0}},
        ]:
            await socket.incoming.put(json.dumps(event))
        from sqlmodel import select

        from astrbot.core.db.po import ProviderStat

        async def completed_rows():
            while True:
                async with db.get_db() as database:
                    rows = (
                        (
                            await database.execute(
                                select(ProviderStat).order_by(ProviderStat.id)
                            )
                        )
                        .scalars()
                        .all()
                    )
                if len(rows) == 2 and rows[0].usage_status == "reported":
                    return rows
                await asyncio.sleep(0.001)

        # Rows are visible while the session remains open, before cleanup.
        rows = await asyncio.wait_for(completed_rows(), 2)
        assert not session.closed
        assert rows[0].token_input_other == 7 and rows[0].token_output == 2
        assert rows[0].provider_model == "actual-live"
        assert rows[0].plugin_id == "realtime-plugin"
        assert rows[1].usage_status == "missing"
        assert rows[0].request_id != rows[1].request_id
        await session.close()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_nested_transcription_failure_is_not_billed_again_as_chat(
    monkeypatch, tmp_path
):
    from astrbot.core.provider.oauth.openai_oauth_transcription import (
        OpenAIOAuthTranscriptionClient,
    )

    provider, db = prepared_provider(monkeypatch)
    provider.provider_config["oauth_audio_transcription"] = True
    path = tmp_path / "nested.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 160)
    monkeypatch.setattr(
        OpenAIOAuthTranscriptionClient,
        "_post_transcription",
        AsyncMock(
            return_value=(
                200,
                {"text": "", "usage": {"input_tokens": 8, "output_tokens": 2}},
                1,
            )
        ),
    )
    backend = AsyncMock()
    monkeypatch.setattr(provider, "_request_backend", backend)
    try:
        with pytest.raises(ValueError):
            await provider.text_chat("hear", audio_urls=[str(path)])
        backend.assert_not_awaited()
        rows = [call.kwargs for call in db.insert_provider_stat.await_args_list]
        assert len(rows) == 2
        assert [row["stats"]["request_kind"] for row in rows] == [
            "transcription",
            "text",
        ]
        assert sum(row["stats"]["token_usage"]["input_other"] for row in rows) == 8
        assert sum(row["stats"]["token_usage"]["output"] for row in rows) == 2
        assert rows[1]["stats"]["usage_status"] == "missing"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_untrusted_usage_metadata_does_not_change_business_result(monkeypatch):
    provider, db = prepared_provider(monkeypatch)
    monkeypatch.setattr(retry, "REQUEST_RETRY_WAIT_MIN_S", 0)
    monkeypatch.setattr(retry, "REQUEST_RETRY_WAIT_MAX_S", 0)
    error = usage_error(503, {"input_other": "bad"})
    monkeypatch.setattr(
        provider,
        "_request_backend",
        AsyncMock(
            side_effect=[
                error,
                {
                    "output_text": "ok",
                    "output": [],
                    "usage": {"input_tokens": 4, "output_tokens": 1},
                },
            ]
        ),
    )
    try:
        response = await provider.text_chat("hello", request_max_retries=2)
        assert response.completion_text == "ok"
        assert response.usage.total == 5
        stats = db.insert_provider_stat.await_args.kwargs["stats"]
        assert stats["token_usage"]["input_other"] == 4
        assert stats["usage_status"] == "partial"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_auth_refresh_failure_usage_is_included_without_changing_last_usage(
    monkeypatch,
):
    provider, db = prepared_provider(monkeypatch)
    monkeypatch.setattr(
        provider, "_refresh_after_auth_failure", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        provider,
        "_request_backend_once",
        AsyncMock(
            side_effect=[
                (
                    401,
                    json.dumps({"usage": {"input_tokens": 7, "output_tokens": 2}}),
                    1,
                ),
                (
                    200,
                    json.dumps(
                        {
                            "output_text": "ok",
                            "usage": {"input_tokens": 4, "output_tokens": 1},
                        }
                    ),
                    2,
                ),
            ]
        ),
    )
    try:
        response = await provider.text_chat("hello")
        assert response.usage.total == 5
        assert (
            db.insert_provider_stat.await_args.kwargs["stats"]["token_usage"][
                "input_other"
            ]
            == 11
        )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("instrumented", [True, False])
async def test_stream_closed_after_terminal_text_chunk_keeps_observed_usage(
    monkeypatch, instrumented
):
    provider, db = prepared_provider(monkeypatch)
    if not instrumented:
        provider.text_chat_stream = provider.text_chat_stream.__wrapped__
        provider._usage_recording_enabled = False
        monkeypatch.setattr(source, "db_helper", db)

    async def events(params):
        yield {
            "type": "response.completed",
            "response": {
                "output_text": "ok",
                "usage": {"input_tokens": 4, "output_tokens": 1},
            },
        }

    monkeypatch.setattr(provider, "_stream_backend_events", events)
    try:
        stream = provider.text_chat_stream("hello")
        assert (await anext(stream)).is_chunk
        await stream.aclose()
        assert db.insert_provider_stat.await_count == 1
        assert (
            db.insert_provider_stat.await_args.kwargs["stats"]["token_usage"][
                "input_other"
            ]
            == 4
        )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_transcription_usage_survives_cancellation_during_resolver_cleanup(
    monkeypatch, tmp_path
):
    import astrbot.core.provider.oauth.openai_oauth_transcription as transcription

    provider, db = prepared_provider(monkeypatch)
    path = tmp_path / "voice.wav"
    path.write_bytes(b"audio-fixture")

    @asynccontextmanager
    async def as_wav():
        yield SimpleNamespace(path=str(path), mime_type="audio/wav")
        raise asyncio.CancelledError

    monkeypatch.setattr(
        transcription,
        "BoundedOAuthAudioResolver",
        lambda *args, **kwargs: SimpleNamespace(as_wav_path=as_wav),
    )
    monkeypatch.setattr(
        transcription.OpenAIOAuthTranscriptionClient,
        "_post_transcription",
        AsyncMock(
            return_value=(
                200,
                {"text": "ok", "usage": {"input_tokens": 7, "output_tokens": 2}},
                1,
            )
        ),
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await provider.transcribe_audio(str(path))
        record = db.insert_provider_stat.await_args.kwargs
        assert record["status"] == "aborted"
        assert record["stats"]["token_usage"]["input_other"] == 7
        assert db.insert_provider_stat.await_count == 1
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_reused_exception_object_still_counts_distinct_retry_attempts(
    monkeypatch,
):
    provider, db = prepared_provider(monkeypatch)
    monkeypatch.setattr(retry, "REQUEST_RETRY_WAIT_MIN_S", 0)
    monkeypatch.setattr(retry, "REQUEST_RETRY_WAIT_MAX_S", 0)
    error = usage_error(503, TokenUsage(input_other=7, output=2))
    monkeypatch.setattr(
        provider,
        "_request_backend",
        AsyncMock(
            side_effect=[
                error,
                error,
                {"output_text": "ok", "usage": {"input_tokens": 4, "output_tokens": 1}},
            ]
        ),
    )
    try:
        response = await provider.text_chat("hello", request_max_retries=3)
        assert response.usage.total == 5
        assert db.insert_provider_stat.await_args.kwargs["stats"]["token_usage"] == {
            "input_other": 18,
            "input_cached": 0,
            "output": 5,
        }
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_all_unknown_failed_attempts_remain_missing(monkeypatch):
    provider, db = prepared_provider(monkeypatch)
    monkeypatch.setattr(retry, "REQUEST_RETRY_WAIT_MIN_S", 0)
    monkeypatch.setattr(retry, "REQUEST_RETRY_WAIT_MAX_S", 0)
    error = usage_error(503)
    monkeypatch.setattr(provider, "_request_backend", AsyncMock(side_effect=error))
    monkeypatch.setattr(provider, "_handle_api_error", AsyncMock(side_effect=error))
    try:
        with pytest.raises(retry.ProviderRequestError):
            await provider.text_chat("hello", request_max_retries=2)
        assert (
            db.insert_provider_stat.await_args.kwargs["stats"]["usage_status"]
            == "missing"
        )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_nested_image_does_not_inherit_an_outer_chat_attempt_total(monkeypatch):
    from astrbot.core.provider.usage_recorder import (
        UsageAttemptLedger,
        current_attempt_ledger,
    )

    provider, db = prepared_provider(monkeypatch)
    outer = UsageAttemptLedger(id(provider))
    outer.note_failure(usage_error(503, TokenUsage(input_other=10, output=2)))
    token = current_attempt_ledger.set(outer)
    monkeypatch.setattr(
        provider,
        "_request_image_backend",
        AsyncMock(return_value={"usage": {"input_tokens": 3, "output_tokens": 1}}),
    )
    monkeypatch.setattr(
        provider, "_extract_generated_images", AsyncMock(return_value=[])
    )
    try:
        await provider.generate_image("cat")
        assert db.insert_provider_stat.await_args.kwargs["stats"]["token_usage"] == {
            "input_other": 3,
            "input_cached": 0,
            "output": 1,
        }
        assert outer.usage.total == 12
    finally:
        current_attempt_ledger.reset(token)
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_nonterminal_ws_usage_remains_partial_after_disconnect_or_cancel(
    monkeypatch, cancel
):
    provider, db = prepared_provider(monkeypatch)
    processed = asyncio.Event()

    class ObservedSocket(Socket):
        async def __anext__(self):
            if not self.events:
                processed.set()
            return await super().__anext__()

    socket = ObservedSocket(
        [
            {
                "type": "response.created",
                "response": {"usage": {"input_tokens": 7, "output_tokens": 2}},
            }
        ],
        wait_forever=cancel,
    )
    install_sessions(monkeypatch, [Session(socket)])
    task = asyncio.create_task(provider.generate_image("cat", transport="websocket"))
    try:
        if cancel:
            await processed.wait()
            task.cancel()
        with pytest.raises(
            asyncio.CancelledError if cancel else source.OpenAIOAuthImageStreamError
        ):
            await task
        stats = db.insert_provider_stat.await_args.kwargs["stats"]
        assert stats["token_usage"] == {
            "input_other": 7,
            "input_cached": 0,
            "output": 2,
        }
        assert stats["usage_status"] == "partial"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_nested_image_failure_propagates_without_double_billing(monkeypatch):
    provider, db = prepared_provider(monkeypatch)
    error = usage_error(429, TokenUsage(input_other=7, output=2))
    monkeypatch.setattr(
        provider, "_request_image_backend", AsyncMock(side_effect=error)
    )

    async def image_chat(self, **kwargs):
        return await self.generate_image("cat")

    monkeypatch.setattr(source.ProviderOpenAIOfficial, "text_chat", image_chat)
    try:
        with pytest.raises(retry.ProviderRequestError):
            await provider.text_chat("image")
        rows = [call.kwargs for call in db.insert_provider_stat.await_args_list]
        assert len(rows) == 2
        assert [row["stats"]["request_kind"] for row in rows] == ["image", "text"]
        assert sum(row["stats"]["token_usage"]["input_other"] for row in rows) == 7
        assert sum(row["stats"]["token_usage"]["output"] for row in rows) == 2
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_native_other_provider_row_does_not_merge_its_parent_ledger(monkeypatch):
    from astrbot.core.provider.usage_recorder import (
        UsageAttemptLedger,
        current_attempt_ledger,
    )

    parent = object()
    outer = UsageAttemptLedger(id(parent))
    outer.note_failure(usage_error(503, TokenUsage(input_other=10, output=2)))
    provider, db = prepared_provider(monkeypatch)
    provider.text_chat = provider.text_chat.__wrapped__
    provider._usage_recording_enabled = False
    monkeypatch.setattr(source, "db_helper", db)
    monkeypatch.setattr(
        provider,
        "_request_backend",
        AsyncMock(
            return_value={
                "output_text": "ok",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }
        ),
    )
    token = current_attempt_ledger.set(outer)
    try:
        await provider.text_chat("hello")
        assert (
            db.insert_provider_stat.await_args.kwargs["stats"]["token_usage"][
                "input_other"
            ]
            == 3
        )
        assert outer.usage.total == 12
    finally:
        current_attempt_ledger.reset(token)
        await provider.terminate()


@pytest.mark.asyncio
async def test_realtime_stats_backpressure_does_not_block_voice_and_close_is_bounded(
    monkeypatch,
):
    from astrbot.core.provider.oauth.openai_oauth_realtime import (
        OpenAIOAuthRealtimeClient,
    )

    provider = FakeProvider(
        [
            httpx.Response(
                200, text=SDP, headers={"Location": "/v1/realtime/calls/rtc_slow_usage"}
            )
        ]
    )
    provider.get_model = lambda: "gpt-live-1-codex"
    provider._extract_response_usage = lambda usage: (
        source.ProviderOpenAIOAuth._extract_response_usage(provider, usage)
    )
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def blocked(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    instrument_provider(
        provider, SimpleNamespace(insert_provider_stat=AsyncMock(side_effect=blocked))
    )
    socket = FakeWebSocket()
    _install_fakes(monkeypatch, provider, [socket])
    client = OpenAIOAuthRealtimeClient(provider, close_timeout=0.02)
    try:
        session = await client.create_session(SDP)
        await socket.incoming.put(
            json.dumps(
                {
                    "type": "response.done",
                    "response": {
                        "id": "usage-one",
                        "usage": {"input_tokens": 4, "output_tokens": 1},
                    },
                }
            )
        )
        await started.wait()
        await socket.incoming.put(
            json.dumps(
                {"type": "turn.done", "turn": {"id": "voice-next", "text": "next"}}
            )
        )
        events = session.events()
        assert (await asyncio.wait_for(anext(events), 0.2)).type == "turn_done"
        await asyncio.wait_for(session.close(), 0.5)
        await asyncio.sleep(0)
        assert stopped.is_set()
        assert not session._usage_tasks
        await events.aclose()
    finally:
        await client.close()


def snapshot_provider(
    *, complete=True, publish=True, final=False, duplicate=False, actual_final=None
):
    import astrbot.core.provider.usage_recorder as recorder

    async def stream(**kwargs):
        try:
            yield LLMResponse(role="assistant", is_chunk=True, completion_text="piece")
            if final:
                usage = TokenUsage(input_other=7, output=2)
                observe = getattr(
                    recorder, "observe_usage_snapshot", lambda *args, **kwargs: None
                )
                observe(usage, response_id="response-test", complete=True)
                yield LLMResponse(role="assistant", usage=actual_final or usage)
        finally:
            if publish:
                observe = getattr(
                    recorder, "observe_usage_snapshot", lambda *args, **kwargs: None
                )
                observe(
                    TokenUsage(input_other=7, output=2),
                    response_id="response-test",
                    complete=complete,
                )
                if duplicate:
                    observe(
                        TokenUsage(input_other=7, output=2),
                        response_id="response-test",
                        complete=complete,
                    )

    return SimpleNamespace(
        provider_config={"id": "snapshot-provider"},
        get_model=lambda: "model",
        text_chat_stream=stream,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "complete,publish,expected",
    [(True, True, "reported"), (False, True, "partial"), (False, False, "missing")],
)
async def test_request_snapshot_survives_generator_close_without_delivering_final(
    complete, publish, expected
):
    provider = snapshot_provider(complete=complete, publish=publish, duplicate=True)
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_provider(provider, db)
    stream = provider.text_chat_stream()
    assert (await anext(stream)).is_chunk
    await stream.aclose()
    assert db.insert_provider_stat.await_count == 1
    record = db.insert_provider_stat.await_args.kwargs
    assert record["status"] == "aborted"
    assert record["stats"]["usage_status"] == expected
    assert record["stats"]["token_usage"]["input_other"] == (7 if publish else 0)


@pytest.mark.asyncio
async def test_normal_snapshot_does_not_duplicate_or_override_confirmed_response_usage():
    provider = snapshot_provider(
        final=True, duplicate=True, actual_final=TokenUsage(input_other=4, output=1)
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_provider(provider, db)
    responses = [response async for response in provider.text_chat_stream()]
    assert responses[-1].usage.total == 5
    assert db.insert_provider_stat.await_count == 1
    assert db.insert_provider_stat.await_args.kwargs["stats"]["token_usage"] == {
        "input_other": 4,
        "input_cached": 0,
        "output": 1,
    }


@pytest.mark.asyncio
async def test_shared_provider_snapshot_sinks_are_isolated_across_driver_tasks():
    import astrbot.core.provider.usage_recorder as recorder

    async def stream(amount, **kwargs):
        try:
            yield LLMResponse(role="assistant", is_chunk=True, completion_text="piece")
        finally:
            getattr(recorder, "observe_usage_snapshot", lambda *args, **kwargs: None)(
                TokenUsage(input_other=amount, output=1),
                response_id="same-upstream-id",
                complete=True,
            )

    provider = SimpleNamespace(
        provider_config={"id": "shared"},
        get_model=lambda: "model",
        text_chat_stream=stream,
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_provider(provider, db)
    first, second = (
        provider.text_chat_stream(amount=7),
        provider.text_chat_stream(amount=17),
    )
    await asyncio.gather(anext(first), anext(second))
    # Closing uses different tasks from the initial drivers.
    await asyncio.gather(first.aclose(), second.aclose())
    rows = [call.kwargs["stats"] for call in db.insert_provider_stat.await_args_list]
    assert sorted(row["token_usage"]["input_other"] for row in rows) == [7, 17]
    assert all(row["usage_status"] == "reported" for row in rows)
    assert len({row["request_id"] for row in rows}) == 2


@pytest.mark.asyncio
async def test_producer_task_inherits_snapshot_sink_and_complete_cannot_downgrade():
    import astrbot.core.provider.usage_recorder as recorder

    publish = asyncio.Event()
    ready = asyncio.Event()

    async def stream(**kwargs):
        async def producer():
            await publish.wait()
            observe = getattr(
                recorder, "observe_usage_snapshot", lambda *args, **kwargs: None
            )
            observe(
                TokenUsage(input_other=7, output=2),
                response_id="response-one",
                complete=True,
            )
            observe(
                TokenUsage(input_other=99, output=99),
                response_id="response-one",
                complete=False,
            )
            ready.set()

        task = asyncio.create_task(producer())
        try:
            yield LLMResponse(role="assistant", is_chunk=True, completion_text="piece")
        finally:
            await task

    provider = SimpleNamespace(
        provider_config={"id": "producer"},
        get_model=lambda: "model",
        text_chat_stream=stream,
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_provider(provider, db)
    iterator = provider.text_chat_stream()
    await anext(iterator)
    publish.set()
    await ready.wait()
    await iterator.aclose()
    stats = db.insert_provider_stat.await_args.kwargs["stats"]
    assert stats["token_usage"] == {"input_other": 7, "input_cached": 0, "output": 2}
    assert stats["usage_status"] == "reported"


@pytest.mark.asyncio
@pytest.mark.parametrize("zero", [False, True])
async def test_generic_disjoint_partial_snapshots_use_presence_and_honor_explicit_zero(
    zero,
):
    from astrbot.core.provider.usage_recorder import observe_usage_snapshot

    async def stream(**kwargs):
        try:
            yield LLMResponse(role="assistant", is_chunk=True, completion_text="piece")
        finally:
            observe_usage_snapshot(
                TokenUsage(input_other=4, is_partial=True),
                response_id="same-response",
                complete=True,
                reported_fields={"input"},
            )
            observe_usage_snapshot(
                TokenUsage(output=2, is_partial=True),
                response_id="same-response",
                complete=True,
                reported_fields={"output"},
            )
            if zero:
                observe_usage_snapshot(
                    TokenUsage(input_other=0, is_partial=True),
                    response_id="same-response",
                    complete=True,
                    reported_fields={"input"},
                )

    provider = SimpleNamespace(
        provider_config={"id": "partial"},
        get_model=lambda: "model",
        text_chat_stream=stream,
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_provider(provider, db)
    iterator = provider.text_chat_stream()
    await anext(iterator)
    await iterator.aclose()
    stats = db.insert_provider_stat.await_args.kwargs["stats"]
    assert stats["token_usage"] == {
        "input_other": 0 if zero else 4,
        "input_cached": 0,
        "output": 2,
    }
    assert stats["usage_status"] == "partial"


@pytest.mark.asyncio
async def test_generic_complete_snapshots_replace_including_downward_corrections():
    from astrbot.core.provider.usage_recorder import observe_usage_snapshot

    async def stream(**kwargs):
        try:
            yield LLMResponse(role="assistant", is_chunk=True, completion_text="piece")
        finally:
            for amount in (4, 7, 2):
                observe_usage_snapshot(
                    TokenUsage(input_other=amount, output=1),
                    response_id="same-response",
                    complete=True,
                    reported_fields={"input", "output"},
                )

    provider = SimpleNamespace(
        provider_config={"id": "complete"},
        get_model=lambda: "model",
        text_chat_stream=stream,
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_provider(provider, db)
    iterator = provider.text_chat_stream()
    await anext(iterator)
    await iterator.aclose()
    assert db.insert_provider_stat.await_count == 1
    stats = db.insert_provider_stat.await_args.kwargs["stats"]
    assert stats["token_usage"] == {"input_other": 2, "input_cached": 0, "output": 1}
    assert stats["usage_status"] == "reported"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage,expected",
    [
        ({"input_tokens": 8, "output_tokens": 2}, "reported"),
        ({"input_tokens": 8}, "partial"),
        ({"type": "duration", "seconds": 1.2}, "missing"),
    ],
)
async def test_actual_stt_get_text_records_modern_usage_once(
    monkeypatch, tmp_path, usage, expected
):
    from astrbot.core.provider.oauth.openai_oauth_transcription import (
        OpenAIOAuthTranscriptionClient,
    )
    from astrbot.core.provider.sources.openai_oauth_stt_source import (
        ProviderOpenAIOAuthSTT,
    )

    config = {
        "id": "real-stt",
        "type": "openai_oauth_stt",
        "oauth_source_id": "shared-source",
        "model": "gpt-4o-transcribe",
        "oauth_access_token": "test-token",
        "oauth_account_id": "test-account",
    }
    provider = ProviderOpenAIOAuthSTT(config, {})
    monkeypatch.setattr(
        provider.oauth_provider, "_ensure_fresh_oauth_token", AsyncMock()
    )
    monkeypatch.setattr(
        OpenAIOAuthTranscriptionClient,
        "_post_transcription",
        AsyncMock(return_value=(200, {"text": "hello", "usage": usage}, 1)),
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    monkeypatch.setattr(source, "db_helper", db)
    path = tmp_path / "actual-stt.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 160)
    try:
        with usage_scope(
            session_umo="wechat:friend", plugin_id="stt-stage", trace_id="trace-asr"
        ):
            assert await provider.get_text(str(path)) == "hello"
        assert db.insert_provider_stat.await_count == 1
        row = db.insert_provider_stat.await_args.kwargs
        assert row["provider_id"] == "real-stt"
        assert row["provider_model"] == "gpt-4o-transcribe"
        assert row["stats"]["stat_version"] == 1
        assert row["stats"]["request_kind"] == "transcription"
        assert row["stats"]["usage_status"] == expected
        assert row["stats"]["source_id"] == "shared-source"
        assert row["stats"]["plugin_id"] == "stt-stage"
        assert row["umo"] == "wechat:friend"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_terminal_snapshot_beats_old_chunk_usage_when_final_has_no_usage():
    from astrbot.core.provider.usage_recorder import observe_usage_snapshot

    async def stream(**kwargs):
        yield LLMResponse(
            role="assistant",
            is_chunk=True,
            completion_text="piece",
            usage=TokenUsage(input_other=4, is_partial=True),
        )
        observe_usage_snapshot(
            TokenUsage(input_other=7, output=2), response_id="terminal", complete=True
        )
        yield LLMResponse(role="assistant", usage=None)

    provider = SimpleNamespace(
        provider_config={"id": "final-missing"},
        get_model=lambda: "model",
        text_chat_stream=stream,
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_provider(provider, db)
    responses = [response async for response in provider.text_chat_stream()]
    assert responses[-1].usage is None
    assert db.insert_provider_stat.await_count == 1
    stats = db.insert_provider_stat.await_args.kwargs["stats"]
    assert stats["token_usage"] == {"input_other": 7, "input_cached": 0, "output": 2}
    assert stats["usage_status"] == "reported"


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", ["partial", "reported"])
async def test_missing_usage_cannot_be_upgraded_by_a_status_override(explicit):
    from astrbot.core.provider.usage_recorder import record_usage

    provider = SimpleNamespace(
        provider_config={"id": "missing"}, get_model=lambda: "model"
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    await record_usage(
        provider,
        db,
        usage=None,
        usage_status=explicit,
        status="aborted",
        start_time=1,
        end_time=2,
    )
    assert (
        db.insert_provider_stat.await_args.kwargs["stats"]["usage_status"] == "missing"
    )


@pytest.mark.asyncio
async def test_invalid_cancelled_exception_usage_remains_missing():
    async def chat(**kwargs):
        error = asyncio.CancelledError()
        error._astrbot_token_usage = {"input_other": "invalid"}
        raise error

    provider = SimpleNamespace(
        provider_config={"id": "invalid-cancel"},
        get_model=lambda: "model",
        text_chat=chat,
    )
    db = SimpleNamespace(insert_provider_stat=AsyncMock())
    instrument_provider(provider, db)
    with pytest.raises(asyncio.CancelledError):
        await provider.text_chat()
    assert (
        db.insert_provider_stat.await_args.kwargs["stats"]["usage_status"] == "missing"
    )
