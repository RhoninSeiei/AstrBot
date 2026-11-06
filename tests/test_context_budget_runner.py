import asyncio
import copy
import json
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core import astr_main_agent as main_agent
from astrbot.core.agent.context.compressor import LLMSummaryCompressor
from astrbot.core.agent.context.config import ContextConfig
from astrbot.core.agent.context.manager import ContextManager
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import Message
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners import tool_loop_agent_runner as runner_module
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.provider.entities import LLMResponse, ProviderMeta, ProviderRequest
from astrbot.core.provider.provider import Provider
from astrbot.core.star.context import Context
from tests.unit import test_astr_main_agent as main_fixtures

main_context = main_fixtures.mock_context
main_event = main_fixtures.mock_event
main_provider = main_fixtures.mock_provider


class ProbeProvider(Provider):
    def __init__(self, provider_id="probe", replies=None):
        super().__init__({"id": provider_id, "max_context_tokens": 0}, {})
        self.set_model("unlisted-model")
        self.replies = list(replies or [])
        self.seen = []

    def get_current_key(self):
        return ""

    def set_key(self, key):
        pass

    async def get_models(self):
        return [self.get_model()]

    def meta(self):
        return ProviderMeta(
            id=self.provider_config["id"], model=self.get_model(), type="probe"
        )

    async def text_chat(self, **payload):
        self.seen.append(
            {
                "contexts": copy.deepcopy(payload["contexts"]),
                "model": payload.get("model"),
                "func_tool": payload.get("func_tool"),
            }
        )
        reply = (
            self.replies.pop(0)
            if self.replies
            else LLMResponse(role="assistant", completion_text="ok")
        )
        if isinstance(reply, Exception):
            raise reply
        return reply


class CharacterCounter:
    def count_tokens(self, messages, trusted_token_usage=0):
        return sum(
            len(message.content or "")
            if isinstance(message.content, str)
            else len(json.dumps(message.model_dump(), ensure_ascii=False))
            for message in messages
        )


def make_budget(size, *, strict=True, model="unlisted-model", threshold=0.9):
    return SimpleNamespace(
        model=model,
        context_window=size,
        input_budget=size,
        output_reserve=0,
        safety_margin=0,
        enforce_input_limit=strict,
        compression_threshold=threshold,
    )


@pytest.fixture
def resolver(monkeypatch):
    async def resolve(provider, *, model=None, fallback_max_context_tokens=128000):
        value = getattr(provider, "test_budget", fallback_max_context_tokens)
        return make_budget(value, model=model or provider.get_model())

    mock = AsyncMock(side_effect=resolve)
    monkeypatch.setattr(runner_module, "resolve_context_budget", mock, raising=False)
    return mock


async def reset_runner(provider, request=None, **kwargs):
    runner = ToolLoopAgentRunner()
    kwargs.setdefault("custom_token_counter", CharacterCounter())
    await runner.reset(
        provider=provider,
        request=request or ProviderRequest(prompt="hello"),
        run_context=ContextWrapper(context=SimpleNamespace()),
        tool_executor=MagicMock(),
        agent_hooks=BaseAgentRunHooks(),
        **kwargs,
    )
    return runner


@pytest.mark.asyncio
async def test_direct_runner_auto_budget_and_parallel_profile_fallback_are_local(
    resolver,
):
    provider = ProbeProvider()
    before = copy.deepcopy(provider.provider_config)
    first, second = await asyncio.gather(
        reset_runner(provider, fallback_max_context_tokens=1000),
        reset_runner(provider, fallback_max_context_tokens=3000),
    )
    assert first.request_context_manager_config.max_context_tokens == 1000
    assert second.request_context_manager_config.max_context_tokens == 3000
    assert provider.provider_config == before
    assert resolver.await_count == 2


@pytest.mark.asyncio
async def test_main_build_keeps_shared_auto_config_and_passes_profile_fallback(
    resolver, main_context, main_event, main_provider
):
    main_context.__class__ = Context
    main_provider.provider_config["max_context_tokens"] = 0
    main_provider.get_model.return_value = "unlisted-model"
    before = copy.deepcopy(main_provider.provider_config)
    result = await main_agent.build_main_agent(
        event=main_event,
        plugin_context=main_context,
        config=main_agent.MainAgentBuildConfig(
            tool_call_timeout=30, fallback_max_context_tokens=23456
        ),
        provider=main_provider,
        req=ProviderRequest(prompt="hello"),
        apply_reset=False,
    )
    assert result is not None
    pending = result.reset_coro
    try:
        assert main_provider.provider_config == before
        await pending
        assert resolver.await_args.kwargs["fallback_max_context_tokens"] == 23456
    finally:
        pending.close()


@pytest.mark.asyncio
async def test_actual_plugin_tool_loop_uses_same_auto_budget_without_shared_write(
    resolver,
):
    provider = ProbeProvider()
    context = SimpleNamespace(
        _db=SimpleNamespace(insert_provider_stat=AsyncMock()),
        provider_manager=SimpleNamespace(
            get_provider_by_id=AsyncMock(return_value=provider)
        ),
    )
    response = await Context.tool_loop_agent(
        context,
        event=SimpleNamespace(unified_msg_origin="test:group:budget"),
        chat_provider_id="probe",
        prompt="hello",
        max_steps=1,
        fallback_max_context_tokens=1000,
        agent_context=SimpleNamespace(
            context=context,
            event=SimpleNamespace(unified_msg_origin="test:group:budget"),
        ),
    )
    assert response.completion_text == "ok"
    assert resolver.await_count >= 1
    assert resolver.await_args.kwargs["fallback_max_context_tokens"] == 1000
    assert provider.provider_config["max_context_tokens"] == 0


@pytest.mark.asyncio
async def test_model_override_and_fallback_resolve_effective_budget_each_request(
    resolver,
):
    primary = ProbeProvider("primary", replies=[RuntimeError("primary unavailable")])
    primary.test_budget = 10000
    fallback = ProbeProvider("fallback")
    fallback.test_budget = 250
    request = ProviderRequest(
        prompt="latest",
        model="one-call-model",
        contexts=[
            {"role": "user", "content": "old" * 300},
            {"role": "assistant", "content": "history"},
        ],
    )
    runner = await reset_runner(primary, request, fallback_providers=[fallback])
    assert [item async for item in runner._iter_llm_responses_with_fallback()][
        -1
    ].completion_text == "ok"
    assert CharacterCounter().count_tokens(fallback.seen[0]["contexts"]) <= 250
    assert fallback.seen[0]["contexts"][-1].content == "latest"
    assert primary.seen[0]["model"] == "one-call-model"
    assert fallback.seen[0]["model"] is None
    assert [item async for item in runner._iter_llm_responses_with_fallback()][
        -1
    ].completion_text == "ok"
    assert fallback.seen[1]["model"] is None
    assert any(
        call.args[0] is primary and call.kwargs["model"] == "one-call-model"
        for call in resolver.await_args_list
    )
    assert any(
        call.args[0] is fallback and call.kwargs["model"] is None
        for call in resolver.await_args_list
    )


@pytest.mark.asyncio
async def test_default_runner_recounts_new_context_instead_of_prior_usage(resolver):
    provider = ProbeProvider()
    provider.test_budget = 100
    runner = await reset_runner(
        provider,
        ProviderRequest(
            prompt="new context " * 1000,
            conversation=SimpleNamespace(token_usage=1, cid="test"),
        ),
        custom_token_counter=None,
    )
    responses = [item async for item in runner._iter_llm_responses_with_fallback()]
    assert responses[-1].role == "err"
    assert "ContextBudgetExceeded" in responses[-1].completion_text
    assert provider.seen == []


def strict_config(size, *, reserve=0, summary_provider=None):
    config = ContextConfig(
        max_context_tokens=size,
        custom_token_counter=CharacterCounter(),
        llm_compress_provider=summary_provider,
    )
    config.reject_oversized_input = True
    config.reserved_context_tokens = reserve
    config.compression_threshold = 0.9
    return config


@pytest.mark.asyncio
async def test_latest_oversized_input_raises_instead_of_context_error_fallback():
    manager = ContextManager(strict_config(100))
    with pytest.raises(ValueError, match="(?i)budget"):
        await manager.process([Message(role="user", content="x" * 101)])


@pytest.mark.asyncio
async def test_schema_and_output_allowance_are_reserved_once(resolver):
    provider = ProbeProvider()
    provider.test_budget = 100
    tools = ToolSet(
        [
            FunctionTool(
                name="large",
                description="x" * 2000,
                parameters={"type": "object"},
                handler=AsyncMock(),
            )
        ]
    )
    runner = await reset_runner(
        provider, ProviderRequest(prompt="small", func_tool=tools)
    )
    responses = [item async for item in runner._iter_llm_responses_with_fallback()]
    assert responses[-1].role == "err"
    assert "ContextBudgetExceeded" in responses[-1].completion_text
    assert provider.seen == []


@pytest.mark.asyncio
async def test_new_large_tool_result_is_checked_despite_previous_conversation_usage(
    resolver,
):
    provider = ProbeProvider()
    provider.test_budget = 100
    request = ProviderRequest(
        prompt="latest", conversation=SimpleNamespace(token_usage=1, cid="test")
    )
    runner = await reset_runner(provider, request)
    runner.run_context.messages.extend(
        [
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            ),
            Message(role="tool", tool_call_id="call-1", content="x" * 101),
        ]
    )
    responses = [item async for item in runner._iter_llm_responses_with_fallback()]
    assert responses[-1].role == "err"
    assert "ContextBudgetExceeded" in responses[-1].completion_text
    assert provider.seen == []


@pytest.mark.asyncio
async def test_embedded_instructions_and_pending_old_tool_group_are_protected():
    manager = ContextManager(strict_config(100))
    messages = [
        Message(role="user", content="old"),
        Message(
            role="assistant",
            tool_calls=[
                {
                    "id": "pending",
                    "type": "function",
                    "function": {"name": "read", "arguments": "x" * 200},
                }
            ],
        ),
        Message(role="system", content="embedded instructions"),
        Message(role="user", content="latest"),
    ]
    with pytest.raises(ValueError, match="(?i)budget"):
        await manager.process(messages)


@pytest.mark.asyncio
async def test_summary_policy_preserves_latest_tool_round_and_embedded_system():
    summary = ProbeProvider(
        "summary", replies=[LLMResponse(role="assistant", completion_text="summary")]
    )
    manager = ContextManager(strict_config(250, summary_provider=summary))
    latest = [
        Message(role="user", content="active request"),
        Message(
            role="assistant",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", tool_call_id="call-1", content="exact evidence"),
    ]
    system = Message(role="system", content="embedded instructions")
    messages = [
        Message(role="user", content="old" * 250),
        Message(role="assistant", content="old answer"),
        system,
        *latest,
    ]
    result = await manager.process(messages)
    assert summary.seen
    assert all(item in result for item in latest)
    assert system in result
    assert result[-3:] == latest
    assert CharacterCounter().count_tokens(result) <= 250


@pytest.mark.asyncio
async def test_non_oauth_legacy_soft_limit_keeps_latest_input():
    manager = ContextManager(ContextConfig(max_context_tokens=10))
    messages = [Message(role="user", content="x" * 100)]
    assert await manager.process(messages) == messages


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [0, 7777])
async def test_real_non_oauth_metadata_override_and_explicit_compatibility(
    monkeypatch, explicit
):
    from astrbot.core.provider import context_budget

    monkeypatch.setattr(
        context_budget,
        "LLM_METADATAS",
        {
            "unlisted-model": {"limit": {"context": 9000}},
            "override": {"limit": {"context": 3000}},
        },
    )
    provider = ProbeProvider()
    provider.provider_config["max_context_tokens"] = explicit
    request = ProviderRequest(prompt="hello", model="override")
    runner = await reset_runner(provider, request)
    assert runner.context_budget.context_window == (explicit or 3000)
    assert runner.request_context_manager_config.compression_threshold == 0.82
    assert not runner.request_context_manager_config.reject_oversized_input
    assert provider.provider_config["max_context_tokens"] == explicit
    assert provider.get_model() == "unlisted-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_real_oauth_runner_uses_shared_offline_or_fallback_estimator(
    monkeypatch, tmp_path, cached
):
    if (
        cached
        and not Path(
            "/tmp/data-gym-cache/fb374d419588a4632f3f557e76b4b70aebbca790"
        ).is_file()
    ):
        pytest.skip("Optional verified offline tokenizer asset is not available")
    if cached:
        pytest.importorskip(
            "tiktoken", reason="Optional offline tokenizer dependency is not available"
        )
    monkeypatch.setenv(
        "TIKTOKEN_CACHE_DIR",
        "/tmp/data-gym-cache" if cached else str(tmp_path / "missing"),
    )
    provider = ProbeProvider()
    provider.provider_config["type"] = "openai_oauth_chat_completion"
    provider.set_model("gpt-6.1-sol")
    provider.get_model_context_metadata = AsyncMock(
        return_value={
            "context_window": 500000,
            "max_context_window": 1000000,
        }
    )
    text = "This is a realistic English request. " * 1000
    tools = ToolSet(
        [
            FunctionTool(
                name="offline",
                description=text,
                parameters={"type": "object"},
                handler=AsyncMock(),
            )
        ]
    )
    runner = await reset_runner(
        provider,
        ProviderRequest(prompt="hello", func_tool=tools),
        custom_token_counter=None,
    )
    assert runner.context_budget.context_window == 490000
    assert runner.context_budget.input_budget == 452332
    assert runner.context_budget.compression_trigger_tokens == 441000
    assert runner.request_context_manager_config.reject_oversized_input
    messages = [Message(role="user", content=text)]
    count = runner.request_context_manager.token_counter.count_tokens(
        messages, trusted_token_usage=1
    )
    assert count > 1000
    assert count < len(text)
    if cached:
        assert count < len(text) // 3
    else:
        assert count >= len(text) * 0.35
    schema_count = runner.request_context_manager_config.reserved_context_tokens
    responses = [item async for item in runner._iter_llm_responses_with_fallback()]
    assert responses[-1].completion_text == "ok"
    assert runner.request_context_manager_config.reserved_context_tokens == schema_count
    if cached:
        assert schema_count < len(text) // 3
    else:
        assert schema_count >= len(text) * 0.35
    assert provider.provider_config["max_context_tokens"] == 0


@pytest.mark.asyncio
async def test_new_tool_result_triggers_round_compression_without_losing_latest_evidence(
    resolver,
):
    provider = ProbeProvider()
    provider.test_budget = 500
    request = ProviderRequest(
        prompt="latest",
        conversation=SimpleNamespace(token_usage=1, cid="test"),
        contexts=[
            {"role": "user", "content": "old" * 100},
            {"role": "assistant", "content": "old answer"},
        ],
    )
    runner = await reset_runner(provider, request)
    latest = [
        runner.run_context.messages[-1],
        Message(
            role="assistant",
            tool_calls=[
                {
                    "id": "read",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        ),
        Message(role="tool", tool_call_id="read", content="evidence" * 20),
    ]
    runner.run_context.messages.extend(latest[1:])
    responses = [item async for item in runner._iter_llm_responses_with_fallback()]
    assert responses[-1].completion_text == "ok"
    sent = provider.seen[-1]["contexts"]
    assert sent == latest
    assert CharacterCounter().count_tokens(sent) <= 500


@pytest.mark.asyncio
async def test_pending_older_call_and_embedded_instruction_survive_removable_history():
    manager = ContextManager(strict_config(400))
    pending = [
        Message(role="user", content="pending request"),
        Message(
            role="assistant",
            tool_calls=[
                {
                    "id": "pending",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        ),
    ]
    system = Message(role="system", content="embedded instructions")
    latest = Message(role="user", content="latest")
    messages = [
        Message(role="user", content="old" * 200),
        Message(role="assistant", content="done"),
        *pending,
        system,
        latest,
    ]
    result = await manager.process(messages)
    assert result == [*pending, system, latest]


@pytest.mark.asyncio
async def test_custom_compressor_cannot_silently_remove_protected_input():
    class DestructiveCompressor:
        def should_compress(self, messages, current_tokens, max_tokens):
            return current_tokens > 200

        async def __call__(self, messages):
            return [Message(role="user", content="rewritten")]

    config = strict_config(300)
    config.custom_compressor = DestructiveCompressor()
    manager = ContextManager(config)
    with pytest.raises(ValueError, match="(?i)budget"):
        await manager.process(
            [
                Message(role="user", content="old" * 200),
                Message(role="assistant", content="done"),
                Message(role="user", content="latest"),
            ]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", [False, True])
@pytest.mark.parametrize("orphan", [False, True])
async def test_cross_round_calls_results_and_orphan_results_are_protected(
    summary, orphan
):
    summarizer = ProbeProvider("summary") if summary else None
    config = strict_config(500, summary_provider=summarizer)
    config.truncate_turns = 2
    config.llm_compress_keep_recent_ratio = 0
    manager = ContextManager(config)
    calls = (
        []
        if orphan
        else [
            Message(role="user", content="first"),
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "cross",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            ),
        ]
    )
    output = Message(role="tool", tool_call_id="cross", content="cross-round evidence")
    latest = Message(role="user", content="latest")
    messages = [
        Message(role="user", content="old" * 300),
        Message(role="assistant", content="done"),
        *calls,
        Message(role="user", content="second"),
        output,
        latest,
    ]
    result = await manager.process(messages)
    assert output in result
    assert all(message in result for message in calls)
    assert result[-1] is latest


@pytest.mark.asyncio
async def test_identical_old_user_content_is_still_removable_history():
    manager = ContextManager(strict_config(100))
    history = [
        message
        for _ in range(8)
        for message in (
            Message(role="user", content="same"),
            Message(role="assistant", content="old answer" * 3),
        )
    ]
    latest = Message(role="user", content="same")
    result = await manager.process([*history, latest])
    assert result[-1] is latest
    assert CharacterCounter().count_tokens(result) <= 100


@pytest.mark.asyncio
async def test_stop_interrupts_budget_resolution_before_provider_call(resolver):
    provider = ProbeProvider()
    runner = await reset_runner(provider)
    entered = asyncio.Event()

    async def wait_for_catalog(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    resolver.side_effect = wait_for_catalog

    async def collect():
        return [item async for item in runner._iter_llm_responses_with_fallback()]

    task = asyncio.create_task(collect())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        runner.request_stop()
        done, _ = await asyncio.wait({task}, timeout=0.2)
        assert task in done, "Stopping must finish without the test cancelling the task"
        assert task.result() == []
        assert provider.seen == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_turn_limit_keeps_both_sides_of_cross_round_tool_group():
    config = strict_config(500)
    config.enforce_max_turns = 1
    manager = ContextManager(config)
    calls = [
        Message(role="user", content="first"),
        Message(
            role="assistant",
            tool_calls=[
                {
                    "id": "cross",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        ),
    ]
    results = [
        Message(role="user", content="second"),
        Message(role="tool", tool_call_id="cross", content="exact result"),
    ]
    latest = Message(role="user", content="latest")
    messages = [
        Message(role="user", content="old" * 300),
        Message(role="assistant", content="done"),
        *calls,
        *results,
        latest,
    ]
    assert await manager.process(messages) == [*calls, *results, latest]


@pytest.mark.asyncio
async def test_fallback_retains_the_profile_selected_summary_provider(resolver):
    primary = ProbeProvider("primary", replies=[RuntimeError("unavailable")])
    primary.test_budget = 10000
    fallback = ProbeProvider("fallback")
    fallback.test_budget = 250
    summary = ProbeProvider(
        "summary", replies=[LLMResponse(role="assistant", completion_text="summary")]
    )
    runner = await reset_runner(
        primary,
        ProviderRequest(
            prompt="latest",
            contexts=[
                {"role": "user", "content": "old" * 300},
                {"role": "assistant", "content": "done"},
            ],
        ),
        fallback_providers=[fallback],
        llm_compress_provider=summary,
        llm_compress_keep_recent_ratio=0,
        llm_compress_instruction="PROFILE_SUMMARY_POLICY",
    )
    responses = [item async for item in runner._iter_llm_responses_with_fallback()]
    assert responses[-1].completion_text == "ok"
    assert len(summary.seen) == 1
    summary_prompt = summary.seen[0]["contexts"][-1]
    content = (
        summary_prompt["content"]
        if isinstance(summary_prompt, dict)
        else summary_prompt.content
    )
    assert "PROFILE_SUMMARY_POLICY" in content
    assert runner.request_context_manager.compressor.provider is summary
    assert fallback.seen[-1]["contexts"][-1].content == "latest"


@pytest.mark.asyncio
async def test_failed_small_candidate_does_not_shrink_large_fallback_history(resolver):
    primary = ProbeProvider("small", replies=[RuntimeError("unavailable")])
    primary.test_budget = 100
    fallback = ProbeProvider("large")
    fallback.test_budget = 10000
    request = ProviderRequest(
        prompt="latest",
        contexts=[
            {"role": "user", "content": "important old history" * 30},
            {"role": "assistant", "content": "old answer"},
        ],
    )
    original = copy.deepcopy(request.contexts)
    runner = await reset_runner(primary, request, fallback_providers=[fallback])
    responses = [item async for item in runner._iter_llm_responses_with_fallback()]
    assert responses[-1].completion_text == "ok"
    assert len(primary.seen[0]["contexts"]) == 1
    assert len(fallback.seen[0]["contexts"]) == 3
    assert fallback.seen[0]["contexts"][0].content == original[0]["content"]
    assert request.contexts == original
    assert runner.run_context.messages == fallback.seen[0]["contexts"]


@pytest.mark.asyncio
async def test_summary_provider_cannot_change_original_request_token_estimator(
    monkeypatch,
):
    from astrbot.core.provider import context_budget

    if not Path(
        "/tmp/data-gym-cache/fb374d419588a4632f3f557e76b4b70aebbca790"
    ).is_file():
        pytest.skip("Optional verified offline tokenizer asset is not available")
    pytest.importorskip(
        "tiktoken", reason="Optional offline tokenizer dependency is not available"
    )
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "/tmp/data-gym-cache")
    primary = ProbeProvider()
    primary.provider_config["type"] = "openai_oauth_chat_completion"
    primary.get_model_context_metadata = AsyncMock(
        return_value={"context_window": 500000}
    )
    await context_budget.resolve_context_budget(primary)

    class SwitchingSummaryProvider(ProbeProvider):
        async def text_chat(self, **kwargs):
            await context_budget.resolve_context_budget(self)
            return LLMResponse(role="assistant", completion_text="summary")

    config = strict_config(500, summary_provider=SwitchingSummaryProvider("summary"))
    config.custom_token_counter = None
    config.conservative_token_count = True
    manager = ContextManager(config)
    probe = [
        Message(role="user", content="This is a realistic English request. " * 1000)
    ]
    before = manager.token_counter.count_tokens(probe)
    assert before < len(probe[0].content) // 3
    await manager.process(
        [
            Message(role="user", content="old conversation " * 2000),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="latest"),
        ]
    )
    assert manager.token_counter.count_tokens(probe) == before


@pytest.mark.asyncio
async def test_summary_contextvars_are_isolated_without_optional_tokenizer_assets():
    encoding = ContextVar("test_summary_encoding", default="request")

    class ScopedCounter(CharacterCounter):
        def count_tokens(self, messages, trusted_token_usage=0):
            count = super().count_tokens(messages, trusted_token_usage)
            return count if encoding.get() == "request" else count * 2

    class SwitchingSummary(ProbeProvider):
        async def text_chat(self, **kwargs):
            encoding.set("summary")
            return LLMResponse(role="assistant", completion_text="summary")

    config = strict_config(300, summary_provider=SwitchingSummary("summary"))
    config.custom_token_counter = ScopedCounter()
    manager = ContextManager(config)
    probe = [Message(role="user", content="counter probe")]
    before = manager.token_counter.count_tokens(probe)
    await manager.process(
        [
            Message(role="user", content="old" * 300),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="latest"),
        ]
    )
    assert encoding.get() == "request"
    assert manager.token_counter.count_tokens(probe) == before


@pytest.mark.asyncio
async def test_shared_custom_summary_has_no_request_state_after_sequential_use():
    shared = LLMSummaryCompressor(
        ProbeProvider("summary"), keep_recent_ratio=0, token_counter=CharacterCounter()
    )
    config = strict_config(300)
    config.custom_compressor = shared
    manager = ContextManager(config)
    await manager.process(
        [
            Message(role="user", content="old" * 300),
            Message(role="assistant", content="done"),
            Message(role="user", content="latest"),
        ]
    )
    assert shared.preserve_message_ids is None
    assert shared.compression_threshold == 0.82


@pytest.mark.asyncio
async def test_shared_custom_summary_has_no_request_state_during_concurrent_use():
    both = asyncio.Event()

    class GatedSummary(ProbeProvider):
        count = 0

        async def text_chat(self, **kwargs):
            self.count += 1
            if self.count == 2:
                both.set()
            await both.wait()
            return LLMResponse(role="assistant", completion_text="summary")

    shared = LLMSummaryCompressor(
        GatedSummary("summary"), keep_recent_ratio=0, token_counter=CharacterCounter()
    )
    managers = []
    for _ in range(2):
        config = strict_config(300)
        config.custom_compressor = shared
        managers.append(ContextManager(config))
    results = await asyncio.gather(
        *(
            manager.process(
                [
                    Message(role="user", content="old" * 300),
                    Message(role="assistant", content="done"),
                    Message(role="user", content=f"latest-{index}"),
                ]
            )
            for index, manager in enumerate(managers)
        )
    )
    assert shared.preserve_message_ids is None
    assert [result[-1].content for result in results] == ["latest-0", "latest-1"]


@pytest.mark.asyncio
async def test_shared_custom_summary_has_no_request_state_after_cancellation():
    entered = asyncio.Event()

    class GatedSummary(ProbeProvider):
        async def text_chat(self, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    shared = LLMSummaryCompressor(
        GatedSummary("summary"), keep_recent_ratio=0, token_counter=CharacterCounter()
    )
    config = strict_config(300)
    config.custom_compressor = shared
    task = asyncio.create_task(
        ContextManager(config).process(
            [
                Message(role="user", content="old" * 300),
                Message(role="assistant", content="done"),
                Message(role="user", content="latest"),
            ]
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert shared.preserve_message_ids is None
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert shared.preserve_message_ids is None
