# OpenAI OAuth Provider Usage Statistics Design

**Status:** Approved for implementation on 2026-07-13

## Problem

AstrBot records model usage after the built-in Agent finishes. Those records use
`agent_type="internal"`, and the Dashboard provider-token service only reads that
type. Calls made by plugins through `Provider.text_chat()` do not pass through the
built-in Agent recorder. OpenAI OAuth image generation also bypasses it.

The OAuth adapter already parses text response usage, and Codex image responses can
carry the same usage object. The missing behavior is persistence at the provider
boundary, with explicit ownership rules that prevent built-in Agent calls from being
counted twice.

## Goals

1. Count every logical OpenAI OAuth text call made outside the built-in Agent.
2. Count every logical OpenAI OAuth image generation call.
3. Preserve the existing built-in Agent summary record for normal conversations.
4. Include both existing Agent summaries and new OAuth provider records in the
   Dashboard totals.
5. Record failed calls without changing the exception observed by callers.
6. Keep statistics failures isolated from provider results.

## Non-goals

1. Backfill historical direct OAuth calls.
2. Add provider-boundary accounting for other provider adapters.
3. Add cost estimates or a text-versus-image Dashboard breakdown.
4. Change the `provider_stats` database schema or Dashboard components.

## Considered Approaches

### Provider accounting with task-local ownership

The OAuth provider records its public calls. A `ContextVar` marks text calls whose
statistics are already owned by the built-in Agent. The main Agent enables that mark
only while awaiting a provider text operation.

This preserves conversation identifiers, time-to-first-token values, fallback usage,
and accumulated multi-step Agent usage in the existing summary row. It also covers
plugin and standalone provider calls without requiring plugin changes.

### Replace built-in Agent summaries for OAuth

The OAuth provider could record every request while the internal stage skips its
summary for OAuth models. This would remove duplicate rows with fewer control-flow
changes, but it would discard the existing conversation-level timing and aggregation
semantics. Mixed primary and fallback providers would also become harder to describe
accurately.

### Deduplicate in the Dashboard query

The Dashboard service could read both record types and try to identify duplicates by
time and token values. The table has no request identifier, so this method cannot
distinguish a duplicate from two legitimate equal-sized calls.

The task-local ownership approach is selected because it defines duplicate
prevention at the call site and preserves current Agent behavior.

## Architecture

### Accounting ownership context

`astrbot/core/provider/provider.py` defines a boolean `ContextVar` named
`provider_stats_managed_by_agent`, with a default value of `False`.

`ToolLoopAgentRunner.reset()` accepts a matching boolean parameter. The main Agent
passes `True`; direct `Context.tool_loop_agent()` calls and third-party runners keep
the default value.

The runner sets the context value only while a provider text operation is actively
being awaited, then resets it in `finally`. Streaming calls set the value around each
`__anext__()` operation and reset it before yielding a response to runner consumers.
This keeps tool execution outside the marked interval. The same scope is applied to
the skills-like requery and repair requests.

### OAuth text records

`ProviderOpenAIOAuth.text_chat()` wraps the inherited OpenAI implementation.

When `provider_stats_managed_by_agent` is `False`, one `agent_type="provider"` row is
written for the public call. Internal retries remain part of that single logical
call. A response with role `err` is recorded with status `error`; other responses use
`completed`. Missing usage still produces a completed row with zero tokens so the
call count remains accurate.

When the context value is `True`, the provider skips its row and the existing
internal Agent recorder writes the accumulated summary. The OAuth streaming method
continues to delegate to `text_chat()`, so it produces one row rather than one row per
chunk.

The record uses `session_id` as `umo` when available. Calls without a session use a
synthetic value containing the provider ID and request kind.

### OAuth image records

`ProviderOpenAIOAuth.generate_image()` always writes one provider row because image
usage is absent from `AgentStats`. When `n` is greater than one, usage from all
backend responses is summed into one row matching the public method invocation.

Successful responses without usage still increment the call count with zero tokens.
Backend, parsing, validation, and file extraction failures write an error row with
usage accumulated before the failure, then re-raise the original exception.

### Dashboard aggregation

`StatService.get_provider_token_stats()` reads rows whose `agent_type` is either
`internal` or `provider`. Existing historical Agent rows remain visible. New direct
OAuth text and image rows join the same daily and range totals. Conversation-specific
statistics continue to use internal rows only.

## Failure Handling

Statistics persistence is secondary to the provider operation. Database exceptions
are logged with a traceback and suppressed. A successful model or image result stays
successful, and an original provider exception remains the exception returned to the
caller.

The ownership context is reset in `finally`, including cancellation and provider
failure cases. This prevents a failed internal request from changing the accounting
mode of later plugin calls in the same task.

## Compatibility

The change adds no dependency and requires no database migration. Existing rows,
Dashboard response fields, plugin interfaces, provider source configuration, OAuth
credentials, model capabilities, and image result objects remain compatible.

## Verification

Automated verification covers:

1. Direct OAuth text success, missing usage, role errors, raised exceptions, and
   streaming single-record behavior.
2. Built-in Agent ownership suppression and context restoration for non-streaming and
   streaming requests.
3. Image usage extraction, `n` aggregation, missing usage, and failed calls.
4. Isolation when database persistence fails.
5. Dashboard inclusion of `internal` and `provider` rows while excluding unrelated
   agent types.
6. Main Agent configuration of the ownership flag.

Production verification uses an image derived from the current working production
image. It checks container health, the compose image target, the Dashboard API, and a
controlled OAuth call that creates a new `agent_type="provider"` record.
