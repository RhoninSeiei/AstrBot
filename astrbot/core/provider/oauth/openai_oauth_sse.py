"""Small, bounded parser for JSON Server-Sent Events."""

import json
from collections.abc import AsyncIterator, Iterator
from io import StringIO
from typing import Any

DEFAULT_MAX_EVENT_BYTES = 4 * 1024 * 1024


class _JsonSSEDecoder:
    """Share bounded event framing across buffered and asynchronous readers."""

    def __init__(self, max_event_bytes: int) -> None:
        """Initialize event state for a single SSE transcript.

        Args:
            max_event_bytes: Maximum UTF-8 byte size of a joined data event.

        Raises:
            ValueError: If the event-size limit is not positive.
        """
        if max_event_bytes <= 0:
            raise ValueError("SSE 事件大小上限必须大于零。")
        self.max_event_bytes = max_event_bytes
        self.data_lines: list[str] = []
        self.data_bytes = 0
        self.first_line = True
        self.done = False

    def feed_line(self, line: str) -> dict[str, Any] | None:
        """Consume one physical line and decode only at an event boundary.

        Args:
            line: One SSE line, optionally including its newline terminator.

        Returns:
            A decoded object when a complete data event is available.

        Raises:
            ValueError: If the event exceeds the limit or contains invalid JSON.
        """
        if self.done:
            return None
        if self.first_line:
            line = line.removeprefix("\ufeff")
            self.first_line = False
        line = line.removesuffix("\n").removesuffix("\r")
        if line == "":
            if self.data_lines:
                payload = "\n".join(self.data_lines)
                self.data_lines = []
                self.data_bytes = 0
                if payload.strip() == "[DONE]":
                    self.done = True
                    return None
                try:
                    event = json.loads(payload)
                except (json.JSONDecodeError, UnicodeError) as exc:
                    raise ValueError("OpenAI OAuth SSE 事件包含无效 JSON。") from exc
                if not isinstance(event, dict):
                    raise ValueError("OpenAI OAuth SSE 事件必须是 JSON 对象。")
                return event
            return None
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field != "data":
            return None
        encoded_size = len(value.encode("utf-8"))
        if self.data_lines:
            encoded_size += 1
        self.data_bytes += encoded_size
        if self.data_bytes > self.max_event_bytes:
            raise ValueError("OpenAI OAuth SSE 事件超过大小上限。")
        self.data_lines.append(value)
        return None


def iter_json_sse_text(
    text: str,
    *,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
) -> Iterator[dict[str, Any]]:
    """Yield complete JSON SSE events from a buffered transcript.

    Args:
        text: The buffered SSE transcript.
        max_event_bytes: Maximum UTF-8 byte size of a joined data event.

    Yields:
        Decoded objects until EOF or a DONE marker.

    Raises:
        ValueError: If the limit or a complete data event is invalid.
    """
    decoder = _JsonSSEDecoder(max_event_bytes)
    for line in StringIO(text, newline=None):
        event = decoder.feed_line(line)
        if event is not None:
            yield event
        if decoder.done:
            return
    event = decoder.feed_line("")
    if event is not None:
        yield event


async def iter_json_sse_events(
    lines: AsyncIterator[str],
    *,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
) -> AsyncIterator[dict[str, Any]]:
    """Yield complete JSON SSE events from an asynchronous line stream.

    Args:
        lines: SSE lines with or without trailing newline characters.
        max_event_bytes: Maximum UTF-8 byte size of a joined data event.

    Yields:
        Decoded objects until EOF or a DONE marker.

    Raises:
        ValueError: If the limit or a complete data event is invalid.
    """
    decoder = _JsonSSEDecoder(max_event_bytes)
    async for line in lines:
        event = decoder.feed_line(line)
        if event is not None:
            yield event
        if decoder.done:
            return
    event = decoder.feed_line("")
    if event is not None:
        yield event
