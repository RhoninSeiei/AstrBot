# Codex OAuth extensions

These extensions reuse a configured ChatGPT/Codex OAuth source. Backend access to
search, transcription and GPT-Live is separate from text-model access. An enabled
adapter is not evidence that a particular account is entitled to the endpoint.

## Text and hosted search

The model selector discovers visible, API-enabled models from the authenticated
Codex model catalog, including `gpt-6.1-terra`. Catalog reads share the source's
credentials, refresh lock, and persistence callback with its chat providers.
Successful catalogs are cached for five minutes. Discovery failures retain the
last successful catalog, or use the built-in model list when none exists, and
retry after 30 seconds. Each catalog read is limited to ten seconds and 4 MiB;
an account or endpoint change discards an in-flight result. Existing configured models are not removed when the
discovered list changes. The Terra adapter accepts the documented reasoning
efforts `none`, `low`, `medium`, `high`, `xhigh`, and `max`.

The provider implements incremental text, reasoning and function-call streams.
Chunks have `is_chunk=True`; the final response contains complete text, tools,
raw response and usage. Consumers must not append the final full text to chunks.

`oauth_web_search` accepts `disabled` (default), `cached`, or `live`.
`oauth_web_search_domains` optionally restricts domains. Function tools and custom
tools are merged; conflicting definitions fail before sending. Request-level
`tool_choice` is honored. Citations come from actual backend URL annotations;
raw hosted-tool output is preserved in `raw_completion`.

Plugins can override search and rate-limit behavior for one request without
changing shared provider configuration. The defaults preserve the provider and
agent behavior used before these fields were added:

- `oauth_web_search="inherit"` uses the provider configuration. `disabled`
  removes hosted search from both provider configuration and
  `custom_extra_body.tools`, while ordinary function tools remain available.
  `cached` and `live` select the corresponding hosted-search mode.
- `retry_rate_limits=False` disables only HTTP 429 retries. Connection failures
  and other retryable statuses keep the configured retry behavior.
- `fallback_on_rate_limit=False` stops the agent after a structured HTTP 429
  instead of trying a fallback provider.

Set the fields directly when returning a request from an event:

```python
yield event.request_llm(
    prompt="Answer with current information.",
    tool_set=tools,
    oauth_web_search="live",
    retry_rate_limits=False,
    fallback_on_rate_limit=False,
)
```

The same names are accepted by the two plugin SDK helpers:

```python
response = await context.llm_generate(
    chat_provider_id=provider_id,
    prompt="One model call",
    oauth_web_search="disabled",
    retry_rate_limits=False,
)

response = await context.tool_loop_agent(
    event=event,
    chat_provider_id=provider_id,
    prompt="Run tools if needed",
    tools=tools,
    oauth_web_search="disabled",
    retry_rate_limits=False,
    fallback_on_rate_limit=False,
)
```

These controls stay attached to every model call in the agent run, including
function-result rounds, schema repair calls and fallback providers. A 429 error
retains `status_code=429` on the final `LLMResponse` when fallback is disabled.

## Context budgets

`max_context_tokens=0` selects an automatic request-local budget. For Codex OAuth,
the target window is 490,000 tokens, capped by the authenticated catalog's default
`context_window`. The optional `max_context_window` is not used to enlarge an
automatic budget. An explicit positive value takes precedence, subject to the
known model maximum. Catalog failures use conservative Codex fallback windows;
public API model metadata does not override the OAuth account's window.

The window, input allowance, and compression trigger are separate. The default
output reservation is `min(32768, window // 10)` and the estimation margin is
`min(8192, window // 100)`. Compression starts at the smaller of 90% of the window
and the input allowance. For a 490,000-token window, these are 452,332 input tokens
and a 441,000-token compression trigger. A 272,000-token window has a 242,080-token
input allowance and trigger. Larger configured output limits can increase the
local reservation; this does not impose a new output cap on the Codex backend.

Main chat and plugin-created tool loops use the same resolver on each actual
provider/model candidate. Existing turn limits and summary providers remain in
effect. The final OAuth request also checks complete instructions, messages,
function schemas and media, including for direct `text_chat` and streaming calls.
That final check does not call a summary model: it only removes old complete
turns if needed. System instructions, the latest request and unfinished or
cross-boundary tool pairs are protected. If they cannot fit, the request fails
locally with a safe budget error. Callers' histories and shared provider settings
are not rewritten.

Text allowance uses a locally available, hash-verified `o200k_base` vocabulary
where available, with a conservative fallback when it is absent. The vocabulary
is an approximation for these models, not an exact backend token meter. Media
allowances are estimates, and budget estimates never replace reported usage.
There is no request-time vocabulary download. Other provider families retain
their existing 82% soft compression policy.

The offline encoder uses the `o200k_base` pattern and vocabulary format from
OpenAI's tiktoken project. Its [MIT license](third-party/tiktoken-license.txt)
is retained with this source.

## Read-only quota query

Quota configuration is intentionally hidden from general AI settings and other
provider pages. It selects a Codex OAuth source (account), never a chat model.
With one enabled built-in Codex OAuth source, the source is selected automatically,
even when no chat model has been added for it. With multiple sources, set the
source ID explicitly in the active profile's configuration:

```json
{
  "provider_settings": {
    "codex_oauth_usage": {
      "enabled": true,
      "source_id": "openai_oauth"
    }
  }
}
```

An empty `source_id` uses automatic selection only when the source is unambiguous.
`enabled: false` disables quota queries for the profile. Existing `provider_id`
settings are supported by resolving the saved model's exact `provider_source_id`;
new configuration should use `source_id`. The calling chat model is never used
as an account fallback. Quota reads use source credentials and network settings
without invoking a model or refreshing credentials.

The built-in `/codex_oauth_usage` command and `codex_oauth_usage` LLM tool
read the configured Codex OAuth account's usage. An administrator in the
current configuration profile can use them in private chats or any group;
ordinary members cannot query the account. Authorization requires both the
event's administrator role and its sender ID in the current profile's
`admins_id` list, checked again after provider lookup and after the usage
request. The legacy
`group_allowlist` value is retained in existing configuration files but no
longer controls this quota query. These rules do not change image generation
or editing permissions.

## Experimental image generation and editing model request

The ChatGPT/Codex OAuth source has an optional, experimental `oauth_image_model` setting.
It sends a model request in the `image_generation` tool and does not change the
main `model` used for the Responses request. New sources and sources without an
`oauth_image_model` setting default to `gpt-image-2.6-sunburst`. An explicitly empty
setting leaves the tool model unset. Accepted request values are:

| Image model | Request value |
| --- | --- |
| Images 2.6 Sunburst, default | `gpt-image-2.6-sunburst` |
| Images 2.6 Flare | `gpt-image-2.6-flare` |
| Images 2.5 Sunburst | `gpt-image-2.5-sunburst` |
| Images 2.5 Flare | `gpt-image-2.5-flare` |
| Images 2.0 | `gpt-image-2` |

Plugins calling the provider can override the source setting for one image
request. Passing an empty string explicitly restores backend selection for
that request. The same selection applies to HTTP and WebSocket requests and
to both generation and reference-image editing.

```python
images = await provider.generate_image(
    "Illustrate a mountain at sunrise",
    model="gpt-6.1-sol",
    image_model="gpt-image-2.6-sunburst",
)
```

An unsupported nonempty image model fails before sending the request. Existing
calls without the new argument keep the previous payload when the source
setting is empty. The backend may ignore the requested image model or fall
back, and a successful image response does not confirm which model was used.
Availability depends on the account's backend permissions. HTTP generation
requests for 2.6 Sunburst, 2.5 Sunburst, and 2.0 succeeded in a November 6, 2026
account probe, but the returned image tool items did not identify the actual
image model. Those results verify accepted requests rather than model routing.

### GIF references and output

Local GIF references retain their original bytes and animation frames. GIF
content is recognized by its signature even when the local file has an incorrect
suffix. Returned GIF, JPEG, and WebP bytes are saved with their detected MIME type
and extension instead of always being labeled PNG. The existing result contract
remains a list of objects containing `path`, `mime_type`, `revised_prompt`, and
`raw`; plugins should forward the file without converting it to a still image.

Plugins may request `output_format="gif"`, with optional `frame_count` from 2 to
16 and `frame_duration_ms` from 20 to 2000. These are optional keyword arguments
and do not alter existing positional arguments:

```python
images = await provider.generate_image(
    "Animate a bouncing dot",
    image_model="gpt-image-2.6-sunburst",
    output_format="gif",
    frame_count=9,
    frame_duration_ms=200,
)
```

The Responses API schema advertises these controls, but the tested Codex OAuth
backend rejected GIF output with HTTP 400 `invalid_value`, both with and without
the frame controls. GIF generation and GIF editing are therefore not verified
OAuth capabilities. Backend rejection is propagated; a PNG response to an
explicit GIF request raises an error before writing output files. No silent
conversion, API-key fallback, or claim of successful animation is made.
Ordinary chat's existing animation previews are unchanged; image-edit plugins
should pass original files through `reference_images`.

References:

- [OpenAI image generation guide](https://developers.openai.com/api/docs/guides/image-generation)
- [Responses tool schema](https://developers.openai.com/api/reference/resources/responses/methods/retrieve)
- [GPT-6.1 Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

## Ordinary voice messages

Existing AstrBot STT and TTS providers remain usable with the Codex text model.
The optional `openai_oauth_stt` adapter references an existing source using
`oauth_source_id`. It shares refresh state and follows source updates and removal.
The adapter is disabled by default because the transcription endpoint may require
separate account credits. No API-key fallback occurs.

To transcribe attachments passed directly to an OAuth chat provider, opt in using
`oauth_audio_transcription: true`, with optional
`oauth_transcription_model: gpt-4o-transcribe`. Transcription failures propagate;
audio is never silently discarded. Plugins can explicitly call
`await provider.transcribe_audio(audio_url, model="gpt-4o-transcribe")`.

This extension does not claim OAuth access to ordinary `/audio/speech` synthesis.
Use an existing AstrBot TTS provider for ordinary synthesized voice replies.

## Plugin realtime entry

The plugin owns its WebRTC peer connection, microphone input and playback. AstrBot
brokers an SDP offer and keeps a sideband control connection. Credentials remain
on the server. No Dashboard route, UI, or raw PCM transport is added.

Use an OAuth provider obtained with `context.get_provider_by_id(provider_id)`.
Create a realtime session from the plugin's SDP offer, apply the returned answer
to that peer connection, consume session events and close the session when the
plugin unloads or the peer disconnects. The implementation also bounds lifetime,
idle time, event queues and concurrent pending/active sessions. Text-model names
such as `gpt-6-astra` are not realtime model names.

The protocol uses OAuth `realtime/calls?intent=quicksilver&architecture=avas`
and a fixed GPT-Live sideband endpoint. Tool work uses client delegation events,
not an advertised native function schema. Plugins decide how to execute a
delegation and return its result. A failed creation whose response is lost cannot
prove remote cleanup; no unverified HTTP hangup endpoint is substituted.

```python
provider = context.get_provider_by_id("openai_oauth/gpt-6-astra")
session = await provider.create_realtime_session(
    offer_sdp,
    voice="cove",
    instructions="Respond briefly.",
)
async with session:
    await peer.set_remote_answer(session.answer_sdp)  # Implement in the plugin.
    async for event in session.events():
        if event.type == "delegation":
            result = await handle_plugin_delegation(event.text)
            await session.submit_delegation_result(event.delegation_id, result)
        elif event.type in {"input_transcript", "output_transcript", "turn_done"}:
            await handle_plugin_transcript(event)
```

The peer and delegation handlers above are plugin-owned examples. Never send
OAuth tokens to the peer. Close the session and peer together when either fails.
`session.send_text()` and delegation results use `speakable` or `commentary`
channels; `session.touch()` can report active media when sideband transcripts are
temporarily absent. Ping traffic does not reset the idle timer.

Transcription and realtime remain explicit opt-ins. Search and rate-limit fields
do not activate either feature, and neither feature falls back to an API key.

Protocol references:

- https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/codex-api/src/endpoint/realtime_call.rs
- https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/codex-api/src/endpoint/realtime_websocket/methods_frameless_bidi.rs
- https://github.com/openclaw/openclaw/blob/15c96969943da24fe52b92ee94c7083d534ee6df/extensions/openai/realtime-quicksilver-session.ts

End-to-end voice acceptance requires a real plugin WebRTC peer and audio input
and output. Unit tests, SDP negotiation and sideband readiness each establish
only their corresponding layer.

## Usage reporting

The usage ledger records backend-reported tokens, independently of context
estimates. A missing direction remains partial; no usage remains missing.
Explicit zero values are retained. Failed attempts with reported usage contribute
to the logical call's ledger, while `LLMResponse.usage` remains the usage of its
single response. Separately recorded image and transcription calls are not added
again to the enclosing chat call.

Streaming providers can publish a request-local snapshot before yielding the
response that contains it:

```python
from astrbot.core.provider.usage_recorder import observe_usage_snapshot

observe_usage_snapshot(
    usage,
    response_id=response_id,
    complete=True,  # Only after observing a terminal backend response.
    reported_fields={"input", "output"},
)
```

Publish cumulative snapshots for the current response, not token deltas.
`reported_fields` identifies directions actually present in a partial response,
including explicit zeros. Repeated snapshots replace known fields instead of
adding them. A full snapshot may correct a previous value downward. Directional
partial snapshots remain partial until an explicitly complete usage report is
received. The hook is a no-op outside a managed request; providers supporting
older AstrBot versions should import it optionally. A complete final usage report
takes precedence, preventing duplicate accounting. A matching partial report can
retain directions already observed for that response without changing the
returned `LLMResponse.usage`. Early stream closure can retain a snapshot that the
producer already observed.

Realtime response snapshots are keyed by session and response identity and
persisted asynchronously. Writes and close-time draining are bounded so a slow
database cannot block audio handling. Queue exhaustion or storage timeouts are
logged; this is not a durable outbox. Session totals without a per-response
identity are not interpreted as additional response usage.

Coverage describes the fraction of recorded calls that include usage. Calls
made through an uninstrumented private SDK are outside that denominator.
Historical missing reports cannot be reconstructed. Audio duration, local token
estimates and OAuth subscription allowance changes are not substituted for
backend-reported token usage.
