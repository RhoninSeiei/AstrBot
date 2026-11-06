import copy
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import astrbot.core.provider.sources.openai_oauth_source as source
from tests.test_openai_oauth_source import _make_provider


def _input_texts(params):
    texts = []
    for item in params["input"]:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                part["text"] for part in content if part.get("type") == "input_text"
            )
    return texts


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_default", [False, True])
@pytest.mark.parametrize(
    "shape", ["text", "image-first", "system-only", "assistant-only", "tool-result"]
)
async def test_json_object_has_safe_input_text_without_mutating_input(
    custom_default, shape
):
    messages = [{"role": "user", "content": "Return the answer."}]
    if shape == "image-first":
        messages[0]["content"] = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
            }
        ]
    elif shape == "system-only":
        messages = [{"role": "system", "content": "Return JSON."}]
    elif shape == "assistant-only":
        messages = [{"role": "assistant", "content": "An earlier answer."}]
    elif shape == "tool-result":
        messages += [
            {
                "role": "assistant",
                "content": "Looking it up.",
                "tool_calls": [
                    {
                        "id": "call_one",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_one",
                "content": "A JSON label from the tool is not an input_text instruction.",
            },
        ]
    custom = (
        {"text": {"format": {"type": "json_object"}, "verbosity": "low"}}
        if custom_default
        else {}
    )
    provider = _make_provider({"custom_extra_body": custom})
    payload = {"messages": messages}
    if not custom_default:
        payload["response_format"] = {"type": "json_object"}
    original_payload = copy.deepcopy(payload)
    original_custom = copy.deepcopy(custom)
    try:
        original_input = provider._convert_messages_to_backend_input(messages)[1]
        params = provider._build_responses_params(payload, None)
        assert params["text"]["format"]["type"] == "json_object"
        assert any("json" in text.lower() for text in _input_texts(params))
        for before, after in zip(original_input, params["input"]):
            if before.get("role") != "user":
                assert after == before
        for item in params["input"]:
            if isinstance(item.get("content"), list):
                assert all(
                    "text" not in part or part["type"] == "input_text"
                    for part in item["content"]
                )
        if shape == "image-first":
            assert params["input"][0]["content"][0] == original_input[0]["content"][0]
        if custom_default:
            assert params["text"]["verbosity"] == "low"
        assert payload == original_payload
        assert provider.provider_config["custom_extra_body"] == original_custom
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_json_keyword_already_in_input_is_not_duplicated():
    provider = _make_provider()
    payload = {
        "messages": [{"role": "user", "content": "Return jSoN."}],
        "response_format": {"type": "json_object"},
    }
    try:
        expected = provider._convert_messages_to_backend_input(payload["messages"])[1]
        assert provider._build_responses_params(payload, None)["input"] == expected
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("format_type", ["text", "json_schema"])
async def test_final_non_json_object_format_does_not_change_input(format_type):
    provider = _make_provider(
        {"custom_extra_body": {"text": {"format": {"type": "json_object"}}}}
    )
    fmt = {"type": format_type}
    if format_type == "json_schema":
        fmt["json_schema"] = {"name": "answer", "schema": {"type": "object"}}
    payload = {
        "messages": [{"role": "user", "content": "Return the answer."}],
        "response_format": fmt,
    }
    try:
        expected = provider._convert_messages_to_backend_input(payload["messages"])[1]
        assert provider._build_responses_params(payload, None)["input"] == expected
    finally:
        await provider.terminate()


def _frame(event, *, multiline=False):
    data = json.dumps(event, indent=2 if multiline else None)
    return "".join("data: " + line + "\n" for line in data.split("\n")) + "\n"


def _completed():
    return {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 7, "output_tokens": 2},
        },
    }


@pytest.mark.asyncio
async def test_buffered_multiline_events_rehydrate_image_usage_and_text(monkeypatch):
    provider = _make_provider()
    warning = Mock()
    monkeypatch.setattr(source.logger, "warning", warning)
    item = {
        "id": "image_one",
        "type": "image_generation_call",
        "result": "opaque-image-result",
    }
    body = (
        "\ufeff: keepalive\n"
        + _frame({"type": "response.output_item.done", "item": item}, multiline=True)
        + _frame(
            {"type": "response.output_text.done", "text": "generated"}, multiline=True
        )
        + _frame(_completed(), multiline=True)
    )
    try:
        response = provider._parse_backend_response(body)
        assert response["output"] == [item]
        assert response["output_text"] == "generated"
        assert provider._extract_response_usage(response["usage"]).total == 9
        warning.assert_not_called()
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    ['data: {"SECRET":\ndata: invalid-SECRET\n\n', "data: []\n\n", 'data: {"SECRET":'],
)
async def test_buffered_bad_event_is_strict_and_safe(monkeypatch, bad):
    provider = _make_provider()
    warning = Mock()
    monkeypatch.setattr(source.logger, "warning", warning)
    try:
        with pytest.raises(ValueError) as caught:
            provider._parse_backend_response(
                bad + ("" if not bad.endswith("\n\n") else _frame(_completed()))
            )
        assert "SECRET" not in str(caught.value)
        warning.assert_called_once()
        assert "SECRET" not in str(warning.call_args)
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal", ["error", "response.error", "response.failed", "response.incomplete"]
)
async def test_multiline_error_terminal_preserves_safe_usage(monkeypatch, terminal):
    provider = _make_provider()
    event = {
        "type": terminal,
        "response": {
            "status": "failed",
            "error": {
                "code": "rate_limit_exceeded",
                "status_code": 429,
                "message": "SECRET",
            },
            "usage": {"input_tokens": 3, "output_tokens": 1},
        },
    }
    try:
        with pytest.raises(Exception) as caught:
            provider._parse_backend_response(
                _frame(event, multiline=True) + _frame(_completed())
            )
        assert "SECRET" not in str(caught.value)
        assert caught.value._astrbot_token_usage.total == 4
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_buffered_terminal_stops_before_bad_late_event():
    provider = _make_provider()
    try:
        assert (
            provider._parse_backend_response(
                _frame(_completed(), multiline=True) + "data: SECRET-invalid\n\n"
            )["status"]
            == "completed"
        )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
async def test_buffered_complete_eof_frame_preserves_standard_newlines(newline):
    provider = _make_provider()
    try:
        body = _frame(_completed(), multiline=True).rstrip("\n").replace("\n", newline)
        assert provider._parse_backend_response(body)["status"] == "completed"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_buffered_unicode_separator_is_not_an_sse_line_boundary():
    provider = _make_provider()
    event = _completed()
    event["response"]["output_text"] = "first\u2028second"
    try:
        body = "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        assert (
            provider._parse_backend_response(body)["output_text"] == "first\u2028second"
        )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_done_marker_is_not_dispatched_before_its_event_boundary(monkeypatch):
    provider = _make_provider()
    warning = Mock()
    monkeypatch.setattr(source.logger, "warning", warning)
    try:
        with pytest.raises(ValueError):
            provider._parse_backend_response("data: [DONE]\ndata: {}\n\n")
        warning.assert_called_once()
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_buffered_image_event_can_exceed_the_text_stream_default_limit():
    provider = _make_provider()
    event = _completed()
    image_result = "A" * (4 * 1024 * 1024 + 1)
    event["response"]["output"] = [
        {"type": "image_generation_call", "result": image_result}
    ]
    try:
        response = provider._parse_backend_response(_frame(event))
        assert response["output"][0]["result"] == image_result
    finally:
        await provider.terminate()


class _ImageEventStream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, reject_late_read=False):
        self.chunks = chunks
        self.reject_late_read = reject_late_read
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk.encode()
        if self.reject_late_read:
            raise AssertionError("Image collector read after terminal event")

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal",
    [
        "response.completed",
        "error",
        "response.error",
        "response.failed",
        "response.incomplete",
    ],
)
async def test_image_http_multiline_terminal_stops_and_closes(monkeypatch, terminal):
    provider = _make_provider()
    event = (
        _completed()
        if terminal == "response.completed"
        else {
            "type": terminal,
            "response": {
                "error": {"code": "rate_limit_exceeded", "status_code": 429},
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        }
    )
    stream = _ImageEventStream([_frame(event, multiline=True)], reject_late_read=True)
    monkeypatch.setattr(provider, "_ensure_fresh_oauth_token", AsyncMock())

    @asynccontextmanager
    async def open_client(**kwargs):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=stream)
            )
        ) as client:
            yield client

    monkeypatch.setattr(provider, "_open_oauth_http_client", open_client)
    try:
        if terminal == "response.completed":
            response = await provider._request_image_backend({})
            assert response["status"] == "completed"
            assert provider._extract_response_usage(response["usage"]).total == 9
        else:
            with pytest.raises(Exception) as caught:
                await provider._request_image_backend({})
            assert caught.value._astrbot_token_usage.total == 4
        assert stream.closed
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_image_http_bad_complete_event_fails_once_without_secret(monkeypatch):
    provider = _make_provider()
    stream = _ImageEventStream(
        ['data: {"SECRET":\ndata: bad-SECRET\n\n', _frame(_completed())]
    )
    warning = Mock()
    monkeypatch.setattr(source.logger, "warning", warning)
    monkeypatch.setattr(provider, "_ensure_fresh_oauth_token", AsyncMock())

    @asynccontextmanager
    async def open_client(**kwargs):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=stream)
            )
        ) as client:
            yield client

    monkeypatch.setattr(provider, "_open_oauth_http_client", open_client)
    try:
        with pytest.raises(ValueError) as caught:
            await provider._request_image_backend({})
        assert "SECRET" not in str(caught.value)
        warning.assert_called_once()
        assert "SECRET" not in str(warning.call_args)
        assert stream.closed
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_image_http_plain_json_response_remains_supported(monkeypatch):
    provider = _make_provider()
    response_data = _completed()["response"]
    monkeypatch.setattr(provider, "_ensure_fresh_oauth_token", AsyncMock())

    @asynccontextmanager
    async def open_client(**kwargs):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=response_data)
            )
        ) as client:
            yield client

    monkeypatch.setattr(provider, "_open_oauth_http_client", open_client)
    try:
        assert await provider._request_image_backend({}) == response_data
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_image_http_auth_error_body_is_preserved_for_one_refresh(
    monkeypatch, status
):
    provider = _make_provider()
    calls = []
    monkeypatch.setattr(provider, "_ensure_fresh_oauth_token", AsyncMock())
    refresh = AsyncMock(return_value=True)
    monkeypatch.setattr(provider, "_refresh_after_auth_failure", refresh)

    async def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(status, text="data: not-JSON-SECRET\n\n")
        return httpx.Response(200, text=_frame(_completed(), multiline=True))

    @asynccontextmanager
    async def open_client(**kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            yield client

    monkeypatch.setattr(provider, "_open_oauth_http_client", open_client)
    try:
        assert (await provider._request_image_backend({}))["status"] == "completed"
        assert len(calls) == 2
        refresh.assert_awaited_once()
    finally:
        await provider.terminate()
