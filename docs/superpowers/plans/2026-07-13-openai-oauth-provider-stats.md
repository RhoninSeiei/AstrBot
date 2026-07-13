# OpenAI OAuth Provider Usage Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Include direct OpenAI OAuth text calls and all OpenAI OAuth image calls in Dashboard provider-token statistics without duplicating built-in Agent usage.

**Architecture:** The main Agent marks only its active provider text awaits with a task-local ownership flag. The OAuth provider records calls outside that scope as `agent_type="provider"`, while image calls are always recorded. The Dashboard reads existing `internal` rows and new `provider` rows.

**Tech Stack:** Python 3.10+, asyncio, ContextVar, SQLModel, pytest, pytest-asyncio, Ruff, Docker Compose

## Global Constraints

1. Preserve existing provider source configuration, OAuth credential sharing, model capabilities, image result types, and public method signatures.
2. Add no dependency and no database migration.
3. Preserve existing internal Agent records, including conversation ID, fallback aggregation, duration, and TTFT.
4. A statistics write failure must not replace a provider result or provider exception.
5. A public OAuth text or image call creates at most one provider record.
6. Production deployment derives a new image from the currently healthy AstrBot image and recreates only the `astrbot` compose service.

---

### Task 1: Define and propagate provider-stat ownership

**Files:**

- Modify: `astrbot/core/provider/provider.py`
- Modify: `astrbot/core/agent/runners/tool_loop_agent_runner.py`
- Modify: `astrbot/core/astr_main_agent.py`
- Test: `tests/test_tool_loop_agent_runner.py`
- Test: `tests/unit/test_astr_main_agent.py`

**Interfaces:**

- Produces: `provider_stats_managed_by_agent: ContextVar[bool]`
- Produces: `ToolLoopAgentRunner.reset(..., provider_stats_managed_by_agent: bool = False)`
- Consumes: the main Agent passes `provider_stats_managed_by_agent=True`

- [ ] **Step 1: Add failing runner ownership tests**

Add a probe provider whose text methods read
`provider_module.provider_stats_managed_by_agent.get()`. Parameterize the test for
streaming and non-streaming calls. Assert that the probe observes `True`, while the
test body observes `False` after each yielded response. Add a failure test asserting
that the value is restored after a provider exception.

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_stats_ownership_is_scoped_to_provider_await(
    runner,
    provider_request,
    mock_tool_executor,
    mock_hooks,
    streaming,
):
    provider = ProviderStatsContextProbe(streaming=streaming)
    await runner.reset(
        provider=provider,
        request=provider_request,
        run_context=ContextWrapper(context=None),
        tool_executor=mock_tool_executor,
        agent_hooks=mock_hooks,
        streaming=streaming,
        provider_stats_managed_by_agent=True,
    )

    responses = []
    async for response in runner._iter_llm_responses():
        responses.append(response)
        assert provider_module.provider_stats_managed_by_agent.get() is False

    assert responses
    assert provider.observed_values == [True]
    assert provider_module.provider_stats_managed_by_agent.get() is False
```

- [ ] **Step 2: Add a failing main Agent configuration assertion**

Extend `test_build_main_agent_passes_max_context_length_to_runner`:

```python
assert (
    mock_runner.reset.await_args.kwargs["provider_stats_managed_by_agent"] is True
)
```

- [ ] **Step 3: Run the ownership tests and verify failure**

Run:

```powershell
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
uv run pytest tests/test_tool_loop_agent_runner.py -k provider_stats_ownership -v
uv run pytest tests/unit/test_astr_main_agent.py -k passes_max_context_length -v
```

Expected: the context variable or runner behavior is absent, and the main Agent reset
keyword assertion fails.

- [ ] **Step 4: Implement the ownership context**

Define the context in `provider.py`:

```python
from contextvars import ContextVar

provider_stats_managed_by_agent: ContextVar[bool] = ContextVar(
    "provider_stats_managed_by_agent",
    default=False,
)
```

Import it into the runner, store the reset argument, and add a reusable context
manager:

```python
@contextmanager
def _provider_stats_scope(self) -> T.Iterator[None]:
    """Apply provider-stat ownership while awaiting one provider operation.

    Yields:
        Control while the provider operation uses this runner's ownership mode.
    """
    token = provider_stats_managed_by_agent.set(
        self.provider_stats_managed_by_agent
    )
    try:
        yield
    finally:
        provider_stats_managed_by_agent.reset(token)
```

Wrap non-streaming provider awaits and both skills-like requery awaits with this
context manager. Iterate streaming providers one item at a time so the context is
reset before yielding:

```python
stream = self.provider.text_chat_stream(**payload)
try:
    while True:
        try:
            with self._provider_stats_scope():
                response = await anext(stream)
        except StopAsyncIteration:
            break
        yield response
finally:
    await stream.aclose()
```

Pass `provider_stats_managed_by_agent=True` from `build_main_agent()`.

- [ ] **Step 5: Run the ownership tests and verify success**

Run the two commands from Step 3. Expected: all selected tests pass.

- [ ] **Step 6: Commit ownership support**

```powershell
git add astrbot/core/provider/provider.py astrbot/core/agent/runners/tool_loop_agent_runner.py astrbot/core/astr_main_agent.py tests/test_tool_loop_agent_runner.py tests/unit/test_astr_main_agent.py
git commit -m "feat(agent): mark provider statistics owned by main agent"
```

### Task 2: Record direct OAuth text and image calls

**Files:**

- Modify: `astrbot/core/provider/sources/openai_oauth_source.py`
- Test: `tests/test_openai_oauth_source.py`

**Interfaces:**

- Consumes: `provider_stats_managed_by_agent.get() -> bool`
- Produces: provider rows through `db_helper.insert_provider_stat(..., agent_type="provider")`
- Produces: one row per public `text_chat`, `text_chat_stream`, or `generate_image` call

- [ ] **Step 1: Isolate provider-stat persistence in existing OAuth tests**

Add an autouse fixture that installs an `AsyncMock` writer even before the production
module defines `db_helper`:

```python
@pytest.fixture(autouse=True)
def provider_stat_writer(monkeypatch):
    writer = AsyncMock()
    monkeypatch.setattr(
        oauth_source,
        "db_helper",
        SimpleNamespace(insert_provider_stat=writer),
        raising=False,
    )
    return writer
```

- [ ] **Step 2: Add failing text accounting tests**

Patch `ProviderOpenAIOfficial.text_chat` to return a controlled `LLMResponse` or raise
a controlled exception. Cover usage and session persistence, missing usage, role
errors, raised exceptions, ownership suppression, and streaming single-record
behavior.

```python
call = provider_stat_writer.await_args.kwargs
assert call["umo"] == "platform:message:session"
assert call["provider_id"] == "test-openai-oauth"
assert call["provider_model"] == "gpt-5.4"
assert call["status"] == "completed"
assert call["agent_type"] == "provider"
assert call["stats"]["token_usage"] == {
    "input_other": 3,
    "input_cached": 2,
    "output": 4,
}
assert call["stats"]["end_time"] >= call["stats"]["start_time"]
```

- [ ] **Step 3: Add failing image accounting tests**

Use the existing fake image backend pattern. Cover response usage, missing usage,
`n=2` aggregation into one row, backend failure, image extraction failure, and a
statistics writer failure that leaves the generated image result intact.

```python
assert provider_stat_writer.await_count == 1
call = provider_stat_writer.await_args.kwargs
assert call["umo"] == "provider:test-openai-oauth:image"
assert call["stats"]["token_usage"] == {
    "input_other": 6,
    "input_cached": 2,
    "output": 10,
}
```

- [ ] **Step 4: Run the OAuth accounting tests and verify failure**

Run:

```powershell
uv run pytest tests/test_openai_oauth_source.py -k "provider_stat or records_usage or aggregates_usage" -v
```

Expected: persistence mocks remain uncalled because the OAuth adapter has no provider
accounting implementation.

- [ ] **Step 5: Implement statistics persistence and text wrapping**

Import `time`, `db_helper`, and the ownership context. Add one helper because the
same persistence operation is used by text success, text failure, image success, and
image failure:

```python
async def _record_provider_stat(
    self,
    *,
    request_kind: str,
    status: str,
    usage: TokenUsage | None,
    start_time: float,
    end_time: float,
    model: str | None = None,
    session_id: str | None = None,
) -> None:
    """Persist one OAuth provider call without affecting its caller.

    Args:
        request_kind: Logical call type used by the synthetic UMO.
        status: Provider call status stored in the database.
        usage: Parsed token usage, or None when the backend omitted it.
        start_time: Epoch time immediately before the public call.
        end_time: Epoch time immediately after the public call.
        model: Explicit request model when supplied.
        session_id: Session identifier when supplied by the caller.
    """
    provider_id = str(self.provider_config.get("id") or self.meta().id)
    try:
        await db_helper.insert_provider_stat(
            umo=session_id or f"provider:{provider_id}:{request_kind}",
            provider_id=provider_id,
            provider_model=model or self.get_model(),
            status=status,
            stats={
                "token_usage": {
                    "input_other": usage.input_other if usage else 0,
                    "input_cached": usage.input_cached if usage else 0,
                    "output": usage.output if usage else 0,
                },
                "start_time": start_time,
                "end_time": end_time,
                "time_to_first_token": 0.0,
            },
            agent_type="provider",
        )
    except Exception:
        logger.warning(
            "Failed to record OpenAI OAuth provider statistics.",
            exc_info=True,
        )
```

Override `text_chat()` with the inherited signature. Capture ownership before calling
`super().text_chat()`. For unmanaged calls, write a success or error row. Preserve the
original response and raised exception.

- [ ] **Step 6: Add image usage aggregation**

Move the whole image method body under one `try` block initialized with:

```python
start_time = time.time()
total_usage = TokenUsage()
```

After each backend response, parse and add usage:

```python
response_usage = self._extract_response_usage(response.get("usage"))
if response_usage is not None:
    total_usage = total_usage + response_usage
```

Write one completed row before returning. In `except Exception`, write one error row
with accumulated usage and re-raise.

- [ ] **Step 7: Run the OAuth tests and verify success**

Run:

```powershell
uv run pytest tests/test_openai_oauth_source.py -v
```

Expected: every OAuth source test passes.

- [ ] **Step 8: Commit OAuth accounting**

```powershell
git add astrbot/core/provider/sources/openai_oauth_source.py tests/test_openai_oauth_source.py
git commit -m "feat(provider): account for OpenAI OAuth usage"
```

### Task 3: Include provider rows in Dashboard totals

**Files:**

- Modify: `astrbot/dashboard/services/stat_service.py`
- Modify: `astrbot/core/db/po.py`
- Create: `tests/unit/test_stat_service.py`

**Interfaces:**

- Consumes: `ProviderStat.agent_type` values `internal` and `provider`
- Produces: unchanged `/api/v1/stats/provider-tokens` response fields with expanded data

- [ ] **Step 1: Add a failing Dashboard aggregation test**

Insert one internal record, one provider record, and one unrelated record into
`temp_db`. Assert that only the first two contribute to calls, tokens, provider
groups, and success rate.

```python
stats = await StatService(temp_db, SimpleNamespace(), {}).get_provider_token_stats(1)

assert stats["range_total_calls"] == 2
assert stats["range_total_tokens"] == 21
assert stats["today_total_calls"] == 2
assert stats["today_total_tokens"] == 21
assert stats["range_by_provider"] == [
    {"provider_id": "oauth", "tokens": 13},
    {"provider_id": "standard", "tokens": 8},
]
```

- [ ] **Step 2: Run the Dashboard test and verify failure**

Run:

```powershell
uv run pytest tests/unit/test_stat_service.py -v
```

Expected: only the internal record is included.

- [ ] **Step 3: Expand the Dashboard filter**

Replace the equality predicate with:

```python
col(ProviderStat.agent_type).in_(("internal", "provider"))
```

Update the `ProviderStat` class docstring to describe Dashboard provider usage records
rather than internal Agent runs only.

- [ ] **Step 4: Run the Dashboard test and verify success**

Run the command from Step 2. Expected: the test passes.

- [ ] **Step 5: Commit Dashboard aggregation**

```powershell
git add astrbot/dashboard/services/stat_service.py astrbot/core/db/po.py tests/unit/test_stat_service.py
git commit -m "feat(dashboard): include direct provider usage"
```

### Task 4: Run repository verification

**Files:**

- Verify all modified Python and Markdown files

**Interfaces:**

- Consumes: implementation from Tasks 1 through 3
- Produces: formatted, lint-clean, test-verified branch

- [ ] **Step 1: Format modified Python files**

```powershell
uv run ruff format astrbot/core/provider/provider.py astrbot/core/agent/runners/tool_loop_agent_runner.py astrbot/core/astr_main_agent.py astrbot/core/provider/sources/openai_oauth_source.py astrbot/dashboard/services/stat_service.py astrbot/core/db/po.py tests/test_tool_loop_agent_runner.py tests/test_openai_oauth_source.py tests/unit/test_astr_main_agent.py tests/unit/test_stat_service.py
```

- [ ] **Step 2: Run focused tests together**

```powershell
uv run pytest tests/test_openai_oauth_source.py tests/test_tool_loop_agent_runner.py tests/unit/test_astr_main_agent.py tests/unit/test_provider_stats.py tests/unit/test_stat_service.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Run Ruff and diff checks**

```powershell
uv run ruff check astrbot/core/provider/provider.py astrbot/core/agent/runners/tool_loop_agent_runner.py astrbot/core/astr_main_agent.py astrbot/core/provider/sources/openai_oauth_source.py astrbot/dashboard/services/stat_service.py astrbot/core/db/po.py tests/test_tool_loop_agent_runner.py tests/test_openai_oauth_source.py tests/unit/test_astr_main_agent.py tests/unit/test_stat_service.py
git diff --check HEAD~3
```

Expected: both commands exit with status zero.

- [ ] **Step 4: Commit formatting changes only when present**

```powershell
git status --short
git add astrbot tests
git commit -m "style: format OpenAI OAuth statistics changes"
```

Skip this commit when formatting created no changes.

### Task 5: Publish and deploy the production image increment

**Files:**

- Modify on production host: `/volume1/docker/astrbot/compose.yaml`
- Copy into derived image: the six modified runtime Python files

**Interfaces:**

- Consumes: verified branch commits
- Produces: a healthy `astrbot` service using a new immutable local image
- Produces: a Dashboard provider-token response containing a newly recorded provider call

- [ ] **Step 1: Push the self-maintained branch**

```powershell
git push origin dev/upgrade-v4.26.4-local
```

- [ ] **Step 2: Verify the current production baseline**

```powershell
ssh -p 44012 wty1996@192.168.1.17 "cd /volume1/docker/astrbot && docker compose ps astrbot && docker inspect astrbot --format '{{.Config.Image}} {{.State.Health.Status}}'"
```

Expected: the service is running and healthy. Preserve the reported image as the
base image.

- [ ] **Step 3: Validate the source in a disposable container**

Create a stopped temporary container from the current image, copy the modified source
files to their matching `/AstrBot` locations, and run `python -m compileall` plus the
focused tests available in the image. Remove only that temporary container after a
successful check.

- [ ] **Step 4: Commit a derived image and update compose**

Commit the validated temporary container under the next
`astrbot-rhonin:4.26.4-local-YYYYMMDD-prodNN` tag. Update only the `astrbot.image`
value in `/volume1/docker/astrbot/compose.yaml`.

- [ ] **Step 5: Recreate only the AstrBot service**

```bash
cd /volume1/docker/astrbot
docker compose up -d --no-deps --force-recreate --no-build astrbot
```

- [ ] **Step 6: Verify health and compose identity**

```bash
docker compose ps astrbot
docker inspect astrbot --format '{{.Config.Image}} {{.State.Health.Status}}'
curl -fsS http://127.0.0.1:16185/
```

Expected: container name remains `astrbot`, the new image is active, health is
`healthy`, and the Dashboard responds.

- [ ] **Step 7: Verify live accounting**

Record the newest `provider_stats.id`, perform one controlled direct OAuth text call,
then query rows with larger IDs. Assert one new row with `agent_type="provider"`, the
expected provider ID and model, a completed status, nonzero duration, and returned
token values. Fetch `/api/v1/stats/provider-tokens?days=1` with Dashboard
authentication and assert that its call and token totals include the new row.

- [ ] **Step 8: Inspect the deployment window**

```bash
docker logs --since 10m astrbot
```

Expected: no import error, database error, repeated OAuth refresh error, or statistics
write failure appears during startup and the controlled request.
