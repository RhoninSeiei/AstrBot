import json
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from tests.test_openai_oauth_source import _make_provider
from tests.test_tool_loop_agent_runner import MockHooks, MockProvider


def completion(arguments, *, valid=False, text=True):
    output = [
        {
            "type": "function_call",
            "name": "default_tool",
            "call_id": "bad",
            "arguments": arguments,
        }
    ]
    if valid:
        output.append(
            {
                "type": "function_call",
                "name": "default_tool",
                "call_id": "good",
                "arguments": '{"query":"intended"}',
            }
        )
    if text:
        output.append(
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Preserved text"}],
            }
        )
    return {
        "id": "response_test",
        "status": "completed",
        "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        '{"secret":"private',
        "[]",
        "null",
        "42",
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        None,
        [],
        {"x": float("nan")},
    ],
)
async def test_invalid_calls_never_enter_executable_lists(arguments):
    provider = _make_provider()
    try:
        result = await provider._parse_responses_completion(
            completion(arguments, valid=True), None
        )
        assert result.completion_text == "Preserved text"
        assert result.tools_call_ids == ["good"]
        assert result.tools_call_args == [{"query": "intended"}]
        assert len(result.rejected_tool_calls) == 1
        assert "private" not in json.dumps(result.rejected_tool_calls)
        assert [call.id for call in result.to_openai_tool_calls_model()] == ["good"]
        assert result.usage is not None
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_missing_arguments_rejected_but_explicit_empty_object_valid():
    provider = _make_provider()
    try:
        raw = completion(None, text=False)
        del raw["output"][0]["arguments"]
        rejected = await provider._parse_responses_completion(raw, None)
        assert rejected.rejected_tool_calls
        assert rejected.tools_call_name == []
        assert rejected.completion_text
        accepted = await provider._parse_responses_completion(completion("{}"), None)
        assert accepted.tools_call_args == [{}]
        assert accepted.rejected_tool_calls == []
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_actual_provider_stream_preserves_text_and_rejects_final_bad_call():
    provider = _make_provider()

    async def events(_params):
        yield {"type": "response.output_text.delta", "delta": "Preserved text"}
        yield {
            "type": "response.completed",
            "response": completion('{"secret":"private', valid=True),
        }

    provider._stream_backend_events = events
    try:
        responses = [
            r
            async for r in provider._query_stream(
                {"messages": [{"role": "user", "content": "test"}]}, None
            )
        ]
        final = responses[-1]
        assert not final.is_chunk
        assert final.tools_call_ids == ["good"]
        assert final.rejected_tool_calls[0]["code"] == "invalid_json"
        assert (
            "".join(r.completion_text or "" for r in responses if r.is_chunk)
            == "Preserved text"
        )
    finally:
        await provider.terminate()


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    async def execute(self, tool, run_context, **kwargs):
        self.calls.append(kwargs)
        yield CallToolResult(content=[TextContent(type="text", text="ok")])


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mixed,budget", [(False, 3), (True, 3), (True, 1)])
async def test_runner_retries_rejected_calls_without_executing_defaults(
    streaming, mixed, budget
):
    parser = _make_provider()
    first = await parser._parse_responses_completion(
        completion('{"secret":"private', valid=mixed), None
    )

    class SequenceProvider(MockProvider):
        async def text_chat(self, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return first
            self.repair_context = kwargs.get("contexts")
            return LLMResponse(role="assistant", completion_text="Repaired answer")

    provider = SequenceProvider()
    executor = RecordingExecutor()
    tool = FunctionTool(
        name="default_tool",
        description="test",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=lambda query="UNSAFE_DEFAULT": query,
    )
    runner = ToolLoopAgentRunner()
    try:
        await runner.reset(
            provider=provider,
            request=ProviderRequest(
                prompt="test", func_tool=ToolSet(tools=[tool]), contexts=[]
            ),
            run_context=ContextWrapper(context=None),
            tool_executor=executor,
            agent_hooks=MockHooks(),
            streaming=streaming,
        )
        async for _ in runner.step_until_done(budget):
            pass
        assert provider.call_count == 2
        assert executor.calls == ([{"query": "intended"}] if mixed else [])
        context = str(provider.repair_context)
        assert "tool_argument_validation" in context
        assert "private" not in context
        assert runner.get_final_llm_resp().completion_text == "Repaired answer"
    finally:
        await parser.terminate()


def test_rejection_metadata_not_serialized_as_executable_calls():
    rejected = LLMResponse(
        role="assistant",
        rejected_tool_calls=[
            {"name": "default_tool", "call_id": "bad", "code": "invalid_json"}
        ],
    )
    fresh = LLMResponse(role="assistant")
    assert fresh.rejected_tool_calls == []
    assert rejected.to_openai_tool_calls_model() == []
    assert rejected.to_openai_tool_calls() == []
    assert rejected.to_openai_to_calls_model() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "with_tools,budget,expected_calls", [(True, 1, 1), (True, 2, 2), (False, 3, 1)]
)
async def test_repair_obeys_budget_and_disabled_tools(
    with_tools, budget, expected_calls
):
    class RejectingProvider(MockProvider):
        async def text_chat(self, **kwargs):
            self.call_count += 1
            return LLMResponse(
                role="assistant",
                completion_text="Rejected call",
                rejected_tool_calls=[
                    {"name": "default_tool", "code": "invalid_json", "call_id": "bad"}
                ],
            )

    provider = RejectingProvider()
    executor = RecordingExecutor()
    tools = (
        ToolSet(
            tools=[
                FunctionTool(
                    name="default_tool",
                    description="test",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda: "default",
                )
            ]
        )
        if with_tools
        else None
    )
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=ProviderRequest(prompt="test", func_tool=tools, contexts=[]),
        run_context=ContextWrapper(context=None),
        tool_executor=executor,
        agent_hooks=MockHooks(),
    )
    async for _ in runner.step_until_done(budget):
        pass
    assert provider.call_count == expected_calls
    assert executor.calls == []
    assert runner.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("initial_rejection", [False, True])
async def test_skills_like_requery_keeps_rejections_separate(mixed, initial_rejection):
    class SequenceProvider(MockProvider):
        async def text_chat(self, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return LLMResponse(
                    role="tool",
                    tools_call_name=["default_tool"],
                    tools_call_args=[{}],
                    tools_call_ids=["discovery"],
                    rejected_tool_calls=[
                        {
                            "name": "default_tool",
                            "call_id": "bad",
                            "code": "invalid_json",
                        }
                    ]
                    if initial_rejection
                    else [],
                )
            self.repair_context = kwargs.get("contexts")
            return LLMResponse(role="assistant", completion_text="Final")

    tool = FunctionTool(
        name="default_tool",
        description="test",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=lambda query="UNSAFE_DEFAULT": query,
    )
    tools = ToolSet(tools=[tool])
    provider, executor, runner = (
        SequenceProvider(),
        RecordingExecutor(),
        ToolLoopAgentRunner(),
    )
    await runner.reset(
        provider=provider,
        request=ProviderRequest(prompt="test", func_tool=tools, contexts=[]),
        run_context=ContextWrapper(context=None),
        tool_executor=executor,
        agent_hooks=MockHooks(),
    )
    runner.tool_schema_mode = "skills_like"
    runner._skill_like_raw_tool_set = tools
    response = LLMResponse(
        role="assistant",
        completion_text="Retained",
        tools_call_name=["default_tool"] if mixed else [],
        tools_call_args=[{"query": "intended"}] if mixed else [],
        tools_call_ids=["good"] if mixed else [],
        rejected_tool_calls=[]
        if initial_rejection
        else [{"name": "default_tool", "call_id": "bad", "code": "invalid_json"}],
    )
    runner._resolve_tool_exec = AsyncMock(return_value=(response, None))
    async for _ in runner.step_until_done(3):
        pass
    assert provider.call_count == 2
    assert executor.calls == ([{"query": "intended"}] if mixed else [])
    assert "tool_argument_validation" in str(provider.repair_context)
