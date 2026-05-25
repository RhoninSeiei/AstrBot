import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import astrbot.core.provider.sources.openai_oauth_source as oauth_source
from astrbot.core.provider.oauth.openai_oauth import parse_oauth_credential_json
from astrbot.core.provider.sources.openai_oauth_source import ProviderOpenAIOAuth


def _jwt_with_claims(claims: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def encode(value: dict) -> str:
        data = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    return f"{encode(header)}.{encode(claims)}."


def _make_provider(
    overrides: dict | None = None,
    *,
    persist_callback=None,
) -> ProviderOpenAIOAuth:
    provider_config = {
        "id": "test-openai-oauth",
        "type": "openai_oauth_chat_completion",
        "model": "gpt-5.4",
        "oauth_access_token": "test-token",
        "oauth_refresh_token": "test-refresh",
        "oauth_account_id": "test-account",
    }
    if overrides:
        provider_config.update(overrides)
    if persist_callback is not None:
        provider_config["oauth_persist_callback"] = persist_callback
    return ProviderOpenAIOAuth(
        provider_config=provider_config,
        provider_settings={},
    )


def test_oauth_provider_keeps_access_token_out_of_key_pool():
    provider = _make_provider()
    try:
        assert provider.get_keys() == ["__openai_oauth__"]
        assert provider.get_current_key() == "__openai_oauth__"
        assert provider.provider_config["oauth_access_token"] == "test-token"
        assert provider.chosen_api_key == ""
    finally:
        # AsyncOpenAI.close is async; this test does not perform network IO.
        pass


def test_parse_codex_auth_json_tokens_object():
    access_token = _jwt_with_claims(
        {
            "email": "codex@example.com",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acc_codex",
            },
        }
    )
    id_token = _jwt_with_claims({"email": "fallback@example.com"})
    raw = json.dumps(
        {
            "tokens": {
                "access_token": access_token,
                "refresh_token": "refresh-token",
                "id_token": id_token,
                "expires_at": "2026-05-07T12:00:00Z",
            }
        }
    )

    parsed = parse_oauth_credential_json(raw)

    assert parsed is not None
    assert parsed["access_token"] == access_token
    assert parsed["refresh_token"] == "refresh-token"
    assert parsed["expires_at"] == "2026-05-07T12:00:00+00:00"
    assert parsed["email"] == "codex@example.com"
    assert parsed["account_id"] == "acc_codex"


@pytest.mark.asyncio
async def test_ensure_fresh_oauth_token_refreshes_and_persists(monkeypatch):
    persisted: list[dict] = []

    async def persist_callback(patch: dict):
        persisted.append(patch)

    async def fake_refresh(refresh_token: str, proxy_url: str = ""):
        assert refresh_token == "test-refresh"
        assert proxy_url == "http://proxy.local"
        return {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
            "email": "new@example.com",
            "account_id": "new-account",
        }

    monkeypatch.setattr(oauth_source, "refresh_access_token", fake_refresh)
    provider = _make_provider(
        {
            "proxy": "http://proxy.local",
            "oauth_expires_at": (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat(),
        },
        persist_callback=persist_callback,
    )
    try:
        await provider._ensure_fresh_oauth_token()

        assert provider.provider_config["oauth_access_token"] == "new-access"
        assert provider.provider_config["oauth_refresh_token"] == "new-refresh"
        assert provider.account_id == "new-account"
        assert provider.chosen_api_key == ""
        assert persisted == [
            {
                "auth_mode": "openai_oauth",
                "oauth_provider": "openai",
                "oauth_access_token": "new-access",
                "oauth_refresh_token": "new-refresh",
                "oauth_expires_at": provider.provider_config["oauth_expires_at"],
                "oauth_account_email": "new@example.com",
                "oauth_account_id": "new-account",
            }
        ]
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_request_backend_refreshes_once_after_401(monkeypatch):
    calls = {"refresh": 0}

    async def fake_refresh(refresh_token: str, proxy_url: str = ""):
        calls["refresh"] += 1
        return {
            "access_token": "retried-access",
            "refresh_token": "retried-refresh",
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
            "email": "",
            "account_id": "retried-account",
        }

    class FakeResponse:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self._text = text

        async def aread(self):
            return self._text.encode()

    class FakeClient:
        responses = [
            FakeResponse(401, '{"error":{"message":"expired"}}'),
            FakeResponse(
                200,
                'data: {"type":"response.completed","response":{"id":"resp_ok","output_text":"OK","output":[]}}\n\n',
            ),
        ]

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aclose(self):
            pass

        async def post(self, *args, **kwargs):
            return self.responses.pop(0)

    monkeypatch.setattr(oauth_source, "refresh_access_token", fake_refresh)
    monkeypatch.setattr(oauth_source.httpx, "AsyncClient", FakeClient)
    provider = _make_provider()
    try:
        response = await provider._request_backend(
            {"model": "gpt-5.4", "input": "ping"}
        )

        assert calls["refresh"] == 1
        assert response["id"] == "resp_ok"
        assert provider.provider_config["oauth_access_token"] == "retried-access"
        assert provider.provider_config["oauth_account_id"] == "retried-account"
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_generate_image_extracts_base64_result(tmp_path):
    image_bytes = b"\x89PNG\r\n\x1a\nsample"
    requested_payloads: list[dict] = []
    provider = _make_provider(
        {
            "generated_image_dir": str(tmp_path),
        }
    )

    async def fake_request_image_backend(payload: dict):
        requested_payloads.append(payload)
        return {
            "output": [
                {
                    "type": "image_generation_call",
                    "result": base64.b64encode(image_bytes).decode(),
                    "revised_prompt": "revised",
                }
            ]
        }

    provider._request_image_backend = fake_request_image_backend
    try:
        results = await provider.generate_image(
            prompt="draw a small icon",
            model="gpt-5.4",
            size="1024x1024",
        )

        payload = requested_payloads[0]
        assert payload["instructions"] == "draw a small icon"
        assert payload["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "draw a small icon",
                    },
                ],
            },
        ]
        assert payload["stream"] is True
        assert payload["tools"] == [
            {
                "type": "image_generation",
                "action": "generate",
                "size": "1024x1024",
            }
        ]
        assert requested_payloads[0]["tool_choice"] == {"type": "image_generation"}
        assert len(results) == 1
        assert results[0].mime_type == "image/png"
        assert results[0].revised_prompt == "revised"
        assert Path(results[0].path).read_bytes() == image_bytes
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_generate_image_with_reference_file_builds_image_edit_payload(tmp_path):
    source_image_bytes = b"\x89PNG\r\n\x1a\nreference"
    output_image_bytes = b"\x89PNG\r\n\x1a\noutput"
    source_path = tmp_path / "reference.png"
    source_path.write_bytes(source_image_bytes)
    requested_payloads: list[dict] = []
    provider = _make_provider(
        {
            "generated_image_dir": str(tmp_path / "generated"),
        }
    )

    async def fake_request_image_backend(payload: dict):
        requested_payloads.append(payload)
        return {
            "output": [
                {
                    "type": "image_generation_call",
                    "result": base64.b64encode(output_image_bytes).decode(),
                }
            ]
        }

    provider._request_image_backend = fake_request_image_backend
    try:
        results = await provider.generate_image(
            prompt="keep the subject and change the background",
            model="gpt-5.4",
            size="1024x1024",
            reference_images=[str(source_path)],
        )

        payload = requested_payloads[0]
        assert (
            payload["instructions"] == "keep the subject and change the background"
        )
        assert payload["stream"] is True
        assert payload["tools"] == [
            {
                "type": "image_generation",
                "action": "edit",
                "size": "1024x1024",
            }
        ]
        assert payload["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "keep the subject and change the background",
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/png;base64,"
                            + base64.b64encode(source_image_bytes).decode()
                        ),
                    },
                ],
            }
        ]
        assert Path(results[0].path).read_bytes() == output_image_bytes
        assert provider.capabilities["image_edit"] is True
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_generate_image_with_data_url_reference_keeps_data_url(tmp_path):
    output_image_bytes = b"\x89PNG\r\n\x1a\noutput"
    data_url = "data:image/jpeg;base64," + base64.b64encode(b"jpeg").decode()
    requested_payloads: list[dict] = []
    provider = _make_provider({"generated_image_dir": str(tmp_path)})

    async def fake_request_image_backend(payload: dict):
        requested_payloads.append(payload)
        return {
            "output": [
                {
                    "type": "image_generation_call",
                    "result": base64.b64encode(output_image_bytes).decode(),
                }
            ]
        }

    provider._request_image_backend = fake_request_image_backend
    try:
        await provider.generate_image(
            prompt="turn this into a watercolor illustration",
            reference_images=[data_url],
            action="auto",
        )

        payload = requested_payloads[0]
        assert payload["instructions"] == "turn this into a watercolor illustration"
        assert payload["stream"] is True
        assert payload["tools"] == [
            {
                "type": "image_generation",
                "action": "auto",
            }
        ]
        assert payload["input"][0]["content"][0] == {
            "type": "input_text",
            "text": "turn this into a watercolor illustration",
        }
        assert payload["input"][0]["content"][1] == {
            "type": "input_image",
            "image_url": data_url,
        }
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_generate_image_reads_sse_incrementally(monkeypatch, tmp_path):
    image_bytes = b"\x89PNG\r\n\x1a\nstreamed"
    image_base64 = base64.b64encode(image_bytes).decode()
    sent_requests: list[dict] = []

    class FakeStreamResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            raise AssertionError("image generation should not read the full SSE body")

        async def aiter_lines(self):
            yield "event: response.output_item.done"
            yield (
                'data: {"type":"response.output_item.done","item":'
                '{"id":"ig_test","type":"image_generation_call",'
                f'"result":"{image_base64}","revised_prompt":"streamed prompt"}}'
                ',"output_index":0}'
            )
            yield ""
            yield "event: response.completed"
            yield (
                'data: {"type":"response.completed","response":'
                '{"id":"resp_img","output":[]}}'
            )
            yield ""

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aclose(self):
            pass

        async def post(self, *args, **kwargs):
            raise AssertionError("image generation should use an SSE stream")

        def stream(self, method, url, **kwargs):
            sent_requests.append(
                {
                    "method": method,
                    "url": url,
                    **kwargs,
                }
            )
            return FakeStreamResponse()

    monkeypatch.setattr(oauth_source.httpx, "AsyncClient", FakeClient)
    provider = _make_provider({"generated_image_dir": str(tmp_path)})
    try:
        results = await provider.generate_image("draw from streaming response")

        assert sent_requests[0]["method"] == "POST"
        assert sent_requests[0]["json"]["stream"] is True
        assert sent_requests[0]["json"]["instructions"] == "draw from streaming response"
        assert sent_requests[0]["json"]["tools"] == [
            {
                "type": "image_generation",
                "action": "generate",
            }
        ]
        assert len(results) == 1
        assert results[0].revised_prompt == "streamed prompt"
        assert Path(results[0].path).read_bytes() == image_bytes
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_parse_backend_response_rehydrates_sse_output_items():
    provider = _make_provider()
    try:
        text = """
event: response.output_text.done
data: {"type":"response.output_text.done","content_index":0,"item_id":"msg_test","output_index":0,"sequence_number":6,"text":"PONG"}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"id":"msg_test","type":"message","status":"completed","content":[{"type":"output_text","annotations":[],"logprobs":[],"text":"PONG"}],"phase":"final_answer","role":"assistant"},"output_index":0,"sequence_number":8}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_test","object":"response","created_at":1775575895,"status":"completed","background":false,"completed_at":1775575901,"error":null,"model":"gpt-5.4","output":[],"parallel_tool_calls":true,"reasoning":{"effort":"none","summary":null},"service_tier":"default","store":false,"temperature":1.0,"text":{"format":{"type":"text"},"verbosity":"medium"},"tool_choice":"auto","tool_usage":{"image_gen":{"input_tokens":0,"input_tokens_details":{"image_tokens":0,"text_tokens":0},"output_tokens":0,"output_tokens_details":{"image_tokens":0,"text_tokens":0},"total_tokens":0},"web_search":{"num_requests":0}},"tools":[],"top_logprobs":0,"top_p":0.98,"truncation":"disabled","usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":0},"output_tokens":6,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":18},"user":null,"metadata":{}},"sequence_number":9}
""".strip()

        response = provider._parse_backend_response(text)
        llm_response = await provider._parse_responses_completion(response, None)

        assert llm_response.completion_text == "PONG"
        assert response["output_text"] == "PONG"
        assert response["output"][0]["content"][0]["text"] == "PONG"
        assert llm_response.usage is not None
        assert llm_response.usage.output == 6
    finally:
        await provider.terminate()
