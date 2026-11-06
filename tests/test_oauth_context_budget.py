"""Context-window discovery and final OAuth request-boundary regressions."""

import asyncio
import copy
import hashlib
import json
import random
import string
import sys
import tempfile
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from tests.test_openai_oauth_source import _make_provider


async def resolve(provider, **kwargs):
    from astrbot.core.provider.context_budget import resolve_context_budget

    return await resolve_context_budget(provider, **kwargs)


def tokenizer_provider():
    return SimpleNamespace(
        provider_config={"type": "openai_oauth_chat_completion"},
        get_model=lambda: "test",
        get_model_context_metadata=AsyncMock(return_value={}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing", "corrupt", "disabled"])
async def test_offline_tokenizer_unavailable_uses_estimate_without_download(
    monkeypatch, tmp_path, mode
):
    import astrbot.core.provider.context_budget as module

    monkeypatch.setattr(module, "_TOKENIZERS", {})
    monkeypatch.setenv(
        "TIKTOKEN_CACHE_DIR", "" if mode == "disabled" else str(tmp_path)
    )
    if mode == "corrupt":
        (tmp_path / module._TOKENIZER_CACHE_KEY).write_bytes(b"invalid")
    download = Mock(side_effect=AssertionError("Tokenizer must stay offline"))
    monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(get_encoding=download))
    provider = tokenizer_provider()
    await resolve(provider)
    assert module.estimate_text_tokens("a" * 1000) == 424
    assert module.estimate_text_tokens("中文" * 100) >= 200
    assert module.estimate_text_tokens("😀" * 100) >= 200
    assert module.estimate_text_tokens("<|endoftext|>") > 0
    download.assert_not_called()


@pytest.mark.asyncio
async def test_verified_offline_tokenizer_counts_fixtures_and_special_text(
    monkeypatch, tmp_path
):
    pytest.importorskip("tiktoken")
    import tiktoken.load
    import tiktoken.registry

    import astrbot.core.provider.context_budget as module

    path = Path(tempfile.gettempdir()) / "data-gym-cache" / module._TOKENIZER_CACHE_KEY
    if not path.is_file():
        pytest.skip("Official offline o200k asset is not installed in this environment")
    cache_path = tmp_path / module._TOKENIZER_CACHE_KEY
    cache_path.write_bytes(path.read_bytes())
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
    monkeypatch.setattr(module, "_TOKENIZERS", {})
    monkeypatch.delitem(tiktoken.registry.ENCODINGS, "o200k_base", raising=False)
    network = Mock(side_effect=AssertionError("No tokenizer network reads"))
    monkeypatch.setattr(tiktoken.load, "read_file", network)
    monkeypatch.setattr(tiktoken, "get_encoding", network)
    provider = tokenizer_provider()
    await asyncio.gather(resolve(provider), resolve(provider))
    await resolve(provider)
    assert module.estimate_text_tokens("hello world") == 2
    assert module.estimate_text_tokens('{"a":1}') == 5
    assert module.estimate_text_tokens("你好，世界") <= 5
    assert module.estimate_text_tokens("<|endoftext|>") > 1
    assert module.estimate_text_tokens("hello world " * 1000) < 2500
    assert len(module._TOKENIZERS) == 1
    network.assert_not_called()


@pytest.mark.asyncio
async def test_missing_optional_tokenizer_dependency_uses_fallback(
    monkeypatch, tmp_path
):
    import astrbot.core.provider.context_budget as module

    asset = b"dependency-branch-fixture"
    (tmp_path / module._TOKENIZER_CACHE_KEY).write_bytes(asset)
    monkeypatch.setattr(module, "_TOKENIZER_SHA256", hashlib.sha256(asset).hexdigest())
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(module, "_TOKENIZERS", {})
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    await resolve(tokenizer_provider())
    assert module.estimate_text_tokens("a" * 1000) == 424


@pytest.mark.asyncio
async def test_verified_bytes_survive_cache_removal_without_downloading(
    monkeypatch, tmp_path
):
    tiktoken = pytest.importorskip("tiktoken")
    import astrbot.core.provider.context_budget as module

    asset = Path(tempfile.gettempdir()) / "data-gym-cache" / module._TOKENIZER_CACHE_KEY
    if not asset.is_file():
        pytest.skip("Official offline o200k asset is not installed in this environment")
    path = tmp_path / module._TOKENIZER_CACHE_KEY
    path.write_bytes(asset.read_bytes())
    original_read = Path.read_bytes

    def read_then_remove(self):
        data = original_read(self)
        if self == path:
            self.unlink()
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_remove)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(module, "_TOKENIZERS", {})
    network = Mock(side_effect=AssertionError("Do not download after verification"))
    monkeypatch.setattr(tiktoken, "get_encoding", network)
    await resolve(tokenizer_provider())
    assert not path.exists()
    assert module.estimate_text_tokens("hello world") == 2
    network.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_request_tokenizer_context_is_independent(
    monkeypatch, tmp_path
):
    import astrbot.core.provider.context_budget as module

    monkeypatch.setattr(module, "_TOKENIZERS", {})
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / module._TOKENIZER_CACHE_KEY).write_bytes(b"test")
    first_encoder = SimpleNamespace(encode=lambda text, **kwargs: [1, 2])
    monkeypatch.setattr(
        module,
        "_load_offline_tokenizer",
        lambda path, key: first_encoder if path.parent.name == "first" else None,
    )
    both_ready = asyncio.Event()
    ready = []

    async def request(name):
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / name))
        await resolve(tokenizer_provider())
        ready.append(name)
        if len(ready) == 2:
            both_ready.set()
        await both_ready.wait()
        await asyncio.sleep(0)
        return module.estimate_text_tokens("a" * 1000)

    assert await asyncio.gather(request("first"), request("second")) == [2, 424]


def catalog(monkeypatch, provider, entries, *, failure=False):
    async def read(headers):
        await asyncio.sleep(0)
        response = httpx.Response(
            503 if failure else 200,
            request=httpx.Request("GET", "https://catalog.invalid/models"),
        )
        return response, json.dumps({"models": entries}).encode()

    reader = AsyncMock(side_effect=read)
    monkeypatch.setattr(provider, "_read_model_catalog", reader)
    monkeypatch.setattr(provider, "_ensure_fresh_oauth_token", AsyncMock())
    return reader


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured,expected", [(64000, 64000), (0, 96000), (-1, 96000), (True, 96000)]
)
async def test_generic_resolution_is_bounded_immutable_and_does_not_mutate(
    monkeypatch, configured, expected
):
    from astrbot.core.utils.llm_metadata import LLM_METADATAS

    monkeypatch.setitem(LLM_METADATAS, "generic", {"limit": {"context": 96000}})
    provider = SimpleNamespace(
        provider_config={"max_context_tokens": configured}, get_model=lambda: "generic"
    )
    original = dict(provider.provider_config)
    budget = await resolve(provider)
    assert budget.context_window == expected
    assert budget.input_budget == expected
    assert budget.compression_threshold == 0.82
    assert not budget.enforce_input_limit
    with pytest.raises(FrozenInstanceError):
        budget.input_budget = 0
    assert provider.provider_config == original


@pytest.mark.asyncio
async def test_generic_invalid_metadata_and_fallback_never_become_unbounded(
    monkeypatch,
):
    from astrbot.core.utils.llm_metadata import LLM_METADATAS

    monkeypatch.setitem(LLM_METADATAS, "bad", {"limit": {"context": "infinite"}})
    provider = SimpleNamespace(provider_config={}, get_model=lambda: "bad")
    assert (
        await resolve(provider, fallback_max_context_tokens=0)
    ).input_budget == 128000


@pytest.mark.asyncio
async def test_catalog_numeric_metadata_and_model_override_are_cached_without_config_mutation(
    monkeypatch,
):
    provider = _make_provider({"model": "gpt-6.1-sol", "max_context_tokens": 0})
    original = copy.deepcopy(provider.provider_config)
    reader = catalog(
        monkeypatch,
        provider,
        [
            {
                "slug": "gpt-6.1-sol",
                "context_window": 500000,
                "max_context_window": 872000,
            },
            {
                "slug": "gpt-6-luna",
                "context_window": 272000,
                "max_context_window": True,
            },
            {"slug": "bad", "context_window": "900000", "max_context_window": 10**15},
        ],
    )
    try:
        sol, luna = await asyncio.gather(
            resolve(provider), resolve(provider, model="gpt-6-luna")
        )
        assert sol.context_window == 490000 and sol.input_budget == 452332
        assert luna.context_window == 272000 and luna.input_budget == 242080
        assert sol.compression_trigger_tokens == 441000
        assert sol.safety_margin == 4900
        assert sol.enforce_input_limit
        assert reader.await_count == 1
        assert await provider.get_models() == ["gpt-6.1-sol", "gpt-6-luna", "bad"]
        assert provider._oauth_shared_state.models_metadata == {
            "gpt-6.1-sol": {"context_window": 500000, "max_context_window": 872000},
            "gpt-6-luna": {"context_window": 272000},
        }
        assert provider.provider_config == original
    finally:
        await provider.terminate()


def test_native_guard_protects_all_system_messages_and_tool_pairing():
    from astrbot.core.provider.context_budget import (
        ContextBudget,
        ContextBudgetExceeded,
        trim_context_to_budget,
    )

    budget = ContextBudget("test", 500, 400, 0.9, 50, enforce_input_limit=True)
    protected = {"type": "message", "role": "developer", "content": "rules"}
    latest = {"type": "message", "role": "user", "content": "new"}
    old = {"type": "message", "role": "user", "content": "x" * 10000}
    call = {
        "type": "function_call",
        "call_id": "one",
        "name": "lookup",
        "arguments": "{}",
    }
    result = {"type": "function_call_output", "call_id": "one", "output": "ok"}
    request = {
        "instructions": "system",
        "input": [old, call, result, protected, latest],
    }
    original = copy.deepcopy(request)
    assert trim_context_to_budget(request, budget)["input"] == [protected, latest]
    assert request == original
    # An unfinished call cannot be discarded, even when it belongs to an old turn.
    request["input"] = [old, call, protected, latest]
    with pytest.raises(ContextBudgetExceeded, match="context budget"):
        trim_context_to_budget(request, budget)
    # Nor may a cut orphan the result or call on either side of a user boundary.
    request["input"] = [old, call, latest, result]
    with pytest.raises(ContextBudgetExceeded, match="context budget"):
        trim_context_to_budget(request, budget)
    # Non-head system messages are never removable input.
    request["input"] = [
        old,
        {**protected, "role": "system", "content": "x" * 10000},
        latest,
    ]
    with pytest.raises(ContextBudgetExceeded, match="context budget"):
        trim_context_to_budget(request, budget)


def test_media_counting_does_not_charge_encoded_image_as_text():
    from astrbot.core.provider.context_budget import estimate_context_tokens

    small = {"type": "input_image", "image_url": "data:image/png;base64,aA=="}
    large = {**small, "image_url": "data:image/png;base64," + "aA==" * 100000}
    assert estimate_context_tokens(small) == estimate_context_tokens(large)
    assert estimate_context_tokens(small) >= 16384
    # A schema/example describing media is still ordinary input text.
    assert estimate_context_tokens({"tools": [{"example": large}]}) > 100000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,window",
    [
        ("gpt-5.6-sol", 128000),
        ("gpt-6-luna", 272000),
        ("gpt-6-astra", 272000),
        ("gpt-6.1-terra", 490000),
    ],
)
async def test_catalog_failure_uses_conservative_known_window(
    monkeypatch, model, window
):
    provider = _make_provider({"model": model, "max_context_tokens": 0})
    catalog(monkeypatch, provider, [], failure=True)
    try:
        budget = await resolve(provider)
        assert budget.context_window == window
        assert budget.input_budget == window - min(32768, window // 10) - min(
            8192, window // 100
        )
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_output_reservation_can_increase_without_claiming_backend_limit(
    monkeypatch,
):
    provider = _make_provider(
        {"model": "gpt-6.1-sol", "custom_extra_body": {"max_output_tokens": 64000}}
    )
    catalog(monkeypatch, provider, [{"slug": "gpt-6.1-sol", "context_window": 500000}])
    try:
        budget = await resolve(provider)
        assert budget.output_reserve == 64000
        assert budget.input_budget == 490000 - 64000 - 4900
        assert "max_output_tokens" not in provider._build_responses_params(
            {"messages": []}, None
        )
        provider.provider_config["custom_extra_body"]["max_output_tokens"] = 500000
        with pytest.raises(ValueError, match="context budget"):
            await resolve(provider)
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("configured,expected", [(850000, 850000), (1050000, 872000)])
async def test_explicit_window_can_use_known_max_but_is_capped(
    monkeypatch, configured, expected
):
    provider = _make_provider(
        {"model": "gpt-6.1-sol", "max_context_tokens": configured}
    )
    catalog(
        monkeypatch,
        provider,
        [
            {
                "slug": "gpt-6.1-sol",
                "context_window": 500000,
                "max_context_window": 872000,
            }
        ],
    )
    try:
        budget = await resolve(provider)
        assert budget.context_window == expected
        assert budget.input_budget == expected - 32768 - min(8192, expected // 100)
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["account", "base"])
async def test_catalog_identity_change_clears_numeric_metadata(monkeypatch, change):
    provider = _make_provider({"model": "gpt-6.1-sol"})
    reader = catalog(
        monkeypatch, provider, [{"slug": "gpt-6.1-sol", "context_window": 500000}]
    )
    try:
        await resolve(provider)
        if change == "account":
            provider._oauth_shared_state.apply({"oauth_account_id": "other"})
        else:
            provider.base_url = "https://other.invalid/codex"
        reader.side_effect = RuntimeError("no catalog")
        budget = await resolve(provider)
        assert budget.context_window <= 500000
        assert not provider._oauth_shared_state.models_metadata
        assert reader.await_count == 2
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_catalog_ttl_refresh_failure_cancel_and_closed(monkeypatch):
    provider = _make_provider({"model": "unknown-codex", "max_context_tokens": 0})
    reader = catalog(monkeypatch, provider, [], failure=True)
    try:
        budget = await resolve(provider)
        assert 0 < budget.input_budget < 128000
        assert reader.await_count == 1
        await resolve(provider)
        assert reader.await_count == 1
        provider._oauth_shared_state.models_expires_at = time.monotonic() - 1
        reader.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await resolve(provider)
    finally:
        await provider.terminate()
    with pytest.raises(RuntimeError, match="closed"):
        await resolve(provider)


def payload(*, latest="new", system="rules"):
    return {
        "model": "gpt-6-luna",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "old " * 1000},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-old",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-old", "content": "result " * 1000},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": latest},
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_direct_oauth_request_trims_only_old_whole_turns(monkeypatch, stream):
    provider = _make_provider(
        {"max_context_tokens": 500, "custom_extra_body": {"max_output_tokens": 64}}
    )
    catalog(monkeypatch, provider, [{"slug": "gpt-6-luna", "context_window": 272000}])
    sent = []
    request = payload()
    original = copy.deepcopy(request)

    async def send(params):
        sent.append(params)
        return {"output_text": "ok", "output": []}

    async def events(params):
        sent.append(params)
        yield {
            "type": "response.completed",
            "response": {"output_text": "ok", "output": []},
        }

    monkeypatch.setattr(provider, "_request_backend", send)
    monkeypatch.setattr(provider, "_stream_backend_events", events)
    try:
        if stream:
            _ = [item async for item in provider._query_stream(request, None)]
        else:
            await provider._query(request, None)
        assert len(sent) == 1
        assert sent[0]["instructions"] == "rules"
        assert sent[0]["input"] == [
            {"type": "message", "role": "user", "content": "new"}
        ]
        assert request == original
        assert provider.provider_config["max_context_tokens"] == 500
    finally:
        await provider.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("oversize", ["system", "latest", "tools", "image"])
async def test_oversized_irreducible_input_is_rejected_before_any_request(
    monkeypatch, stream, oversize
):
    provider = _make_provider(
        {"max_context_tokens": 500, "custom_extra_body": {"max_output_tokens": 64}}
    )
    catalog(monkeypatch, provider, [{"slug": "gpt-6-luna", "context_window": 272000}])
    request = payload(
        latest="x" * 20000 if oversize == "latest" else "new",
        system="x" * 20000 if oversize == "system" else "rules",
    )
    tools = None
    if oversize == "tools":
        tools = SimpleNamespace(
            get_func_desc_openai_style=lambda **kwargs: [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "x" * 20000,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )
    if oversize == "image":
        request["messages"][-1]["content"] = [
            {"type": "image_url", "image_url": {"url": "https://image.invalid/a.png"}}
        ]
    send = AsyncMock(return_value={"output": []})
    monkeypatch.setattr(provider, "_request_backend", send)
    streamed = []

    async def events(params):
        streamed.append(params)
        yield {"type": "response.completed", "response": {"output": []}}

    monkeypatch.setattr(provider, "_stream_backend_events", events)
    original = copy.deepcopy(request)
    try:
        with pytest.raises(ValueError, match="context budget"):
            if stream:
                _ = [item async for item in provider._query_stream(request, tools)]
            else:
                await provider._query(request, tools)
        send.assert_not_awaited()
        assert not streamed
        assert request == original
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_concurrent_request_model_overrides_do_not_change_provider_or_inputs(
    monkeypatch,
):
    provider = _make_provider({"model": "gpt-6.1-sol", "max_context_tokens": 0})
    reader = catalog(
        monkeypatch,
        provider,
        [
            {"slug": "gpt-6.1-sol", "context_window": 500000},
            {"slug": "gpt-6-luna", "context_window": 272000},
        ],
    )
    sent = []

    async def send(params):
        await asyncio.sleep(0)
        sent.append(params)
        return {"output": [], "output_text": "ok"}

    monkeypatch.setattr(provider, "_request_backend", send)
    request = payload(
        latest="".join(
            random.Random(7).choices(string.ascii_letters + string.digits, k=600000)
        )
    )
    other = {**request, "model": "gpt-6.1-sol"}
    try:
        results = await asyncio.gather(
            provider._query(request, None),
            provider._query(other, None),
            return_exceptions=True,
        )
        assert isinstance(results[0], ValueError)
        assert not isinstance(results[1], Exception)
        assert [params["model"] for params in sent] == ["gpt-6.1-sol"]
        assert provider.get_model() == "gpt-6.1-sol"
        assert reader.await_count == 1
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_only_catalog_maximum_also_caps_automatic_window(monkeypatch):
    provider = _make_provider({"model": "gpt-6.1-sol", "max_context_tokens": 0})
    catalog(
        monkeypatch, provider, [{"slug": "gpt-6.1-sol", "max_context_window": 128000}]
    )
    try:
        assert (await resolve(provider)).context_window == 128000
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_non_oauth_resolution_clears_a_previous_request_encoder():
    import astrbot.core.provider.context_budget as module

    module._active_tokenizer.set(SimpleNamespace(encode=lambda *args, **kwargs: [1]))
    await resolve(SimpleNamespace(provider_config={}, get_model=lambda: "generic"))
    assert module._active_tokenizer.get() is None
    assert module.estimate_text_tokens("a" * 1000) == 424
