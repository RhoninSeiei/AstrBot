"""Immutable request-local context budgets and deterministic boundary checks."""

import asyncio
import base64
import copy
import hashlib
import json
import math
import os
import tempfile
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot.core.utils.llm_metadata import LLM_METADATAS

_TOKENIZER_CACHE_KEY = "fb374d419588a4632f3f557e76b4b70aebbca790"
_TOKENIZER_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
_TOKENIZERS: dict[tuple[str, int, int], Any] = {}
_TOKENIZER_LOCK = threading.Lock()
_active_tokenizer: ContextVar[Any] = ContextVar(
    "astrbot_context_tokenizer", default=None
)


def _load_offline_tokenizer(path: Path, key: tuple[str, int, int]) -> Any:
    """Load the optional approximation once, only from a verified local asset.

    Args:
        path: Existing tiktoken cache asset.
        key: Path and file identity used to coalesce concurrent initialization.

    Returns:
        The optional encoding, or None for missing/corrupt/unsupported assets.
    """
    with _TOKENIZER_LOCK:
        if key in _TOKENIZERS:
            return _TOKENIZERS[key]
        encoding = None
        try:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() == _TOKENIZER_SHA256:
                import tiktoken

                # Static o200k_base pattern and special IDs from tiktoken's
                # openai_public.py; never invoke its remote-loading factory.
                pattern = "|".join(
                    [
                        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                        r"\p{N}{1,3}",
                        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
                        r"\s*[\r\n]+",
                        r"\s+(?!\S)",
                        r"\s+",
                    ]
                )
                ranks = {
                    base64.b64decode(token): int(rank)
                    for token, rank in (
                        line.split() for line in data.splitlines() if line
                    )
                }
                encoding = tiktoken.Encoding(
                    name="astrbot_o200k_base_offline",
                    pat_str=pattern,
                    mergeable_ranks=ranks,
                    special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018},
                )
        except (ImportError, OSError, ValueError):
            pass
        _TOKENIZERS[key] = encoding
        return encoding


class ContextBudgetExceeded(ValueError):
    """The protected request content cannot fit the model's input budget."""


@dataclass(frozen=True)
class ContextBudget:
    """Effective limits for one provider request, without changing configuration."""

    model: str
    context_window: int
    input_budget: int
    compression_threshold: float
    output_reserve: int
    safety_margin: int = 0
    enforce_input_limit: bool = False

    @property
    def compression_trigger_tokens(self) -> int:
        """Return the proactive compression trigger in input tokens."""
        return round(self.input_budget * self.compression_threshold)


async def resolve_context_budget(
    provider: Any,
    *,
    model: str | None = None,
    fallback_max_context_tokens: int = 128000,
) -> ContextBudget:
    """Resolve a finite budget for an effective model without mutating a provider.

    Args:
        provider: The provider selected for this request.
        model: A request-local model override.
        fallback_max_context_tokens: Generic-provider fallback window.

    Returns:
        Immutable effective window, input allowance, and compression trigger.

    Raises:
        RuntimeError: If the OAuth provider is closed.
        ContextBudgetExceeded: If output reservation leaves no input allowance.
    """
    model = str(model or provider.get_model())
    config = provider.provider_config
    configured = config.get("max_context_tokens")
    explicit = type(configured) is int and configured > 0
    if config.get("type") != "openai_oauth_chat_completion":
        _active_tokenizer.set(None)
        metadata = LLM_METADATAS.get(model, {})
        metadata_limit = metadata.get("limit", {}) if isinstance(metadata, dict) else {}
        if not isinstance(metadata_limit, dict):
            metadata_limit = {}
        window = configured if explicit else metadata_limit.get("context")
        if type(window) is not int or window <= 0:
            window = fallback_max_context_tokens
        if type(window) is not int or window <= 0:
            window = 128000
        return ContextBudget(model, window, window, 0.82, 0)

    if getattr(provider, "_oauth_media_closed", False):
        raise RuntimeError("OAuth provider is closed")
    cache_dir = os.environ.get(
        "TIKTOKEN_CACHE_DIR",
        os.environ.get(
            "DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache")
        ),
    )
    encoding = None
    if cache_dir:
        path = Path(cache_dir) / _TOKENIZER_CACHE_KEY
        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
            if key in _TOKENIZERS:
                encoding = _TOKENIZERS[key]
            else:
                encoding = await asyncio.to_thread(_load_offline_tokenizer, path, key)
        except OSError:
            pass
    _active_tokenizer.set(encoding)

    metadata = await provider.get_model_context_metadata(model)
    # These are conservative known Codex windows, not the optional extended window.
    known_window = 128000
    if model.startswith(("gpt-6.1-sol", "gpt-6.1-terra")):
        known_window = 500000
    elif model.startswith(("gpt-6-luna", "gpt-6-astra", "gpt-6-sol")):
        known_window = 272000
    advertised_window = metadata.get("context_window", known_window)
    maximum = metadata.get("max_context_window", advertised_window)
    window = (
        min(configured, maximum)
        if explicit
        else min(490000, advertised_window, maximum)
    )
    output_reserve = min(32768, window // 10)
    extra = config.get("custom_extra_body", {})
    for values in (config, extra if isinstance(extra, dict) else {}):
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = values.get(key)
            if type(value) is int and value > output_reserve:
                output_reserve = value
    safety_margin = min(8192, window // 100)
    input_budget = window - output_reserve - safety_margin
    if input_budget <= 0:
        raise ContextBudgetExceeded("OAuth context budget has no input allowance.")
    trigger = min(math.floor(window * 0.90), input_budget)
    return ContextBudget(
        model,
        window,
        input_budget,
        trigger / input_budget,
        output_reserve,
        safety_margin,
        True,
    )


def estimate_text_tokens(text: str) -> int:
    """Count with an offline approximation, or a conservative language estimate.

    Args:
        text: Literal text, excluding encoded binary media.

    Returns:
        Approximate input tokens, never reported backend usage or an exact GPT-6 count.
    """
    encoding = _active_tokenizer.get()
    if encoding is not None:
        return len(encoding.encode(text, disallowed_special=()))
    if not text:
        return 0
    ascii_count = sum(ord(char) < 128 for char in text)
    emoji_count = sum(
        0x1F000 <= ord(char) <= 0x1FAFF or 0x2600 <= ord(char) <= 0x27BF
        for char in text
    )
    other_count = len(text) - ascii_count - emoji_count
    return math.ceil((ascii_count * 0.35 + other_count + emoji_count * 2) * 1.20) + 4


def estimate_context_tokens(params: Any) -> int:
    """Estimate the complete serialized input, treating media as media.

    Args:
        params: Responses request fields, a schema, or a nested content value.

    Returns:
        Conservative allowance including system text, schemas and media parts.
    """
    if isinstance(params, str):
        return estimate_text_tokens(params) + 2
    if isinstance(params, dict):
        media_type = params.get("type")
        media_key = None
        if media_type in {"input_image", "image_url"}:
            media_key = "image_url"
        elif media_type in {"input_audio", "audio_url"}:
            media_key = "input_audio" if "input_audio" in params else "audio_url"
        total = 2
        if media_key and media_key in params:
            total += 16384 if "image" in str(media_type) else 32768
        for key, value in params.items():
            if key == media_key:
                continue
            if key in {"tools", "text"}:
                # Schemas and JSON examples are text, even if they describe images.
                total += estimate_text_tokens(str(key)) + 3
                total += estimate_text_tokens(json.dumps(value, ensure_ascii=False))
                continue
            total += estimate_text_tokens(str(key)) + 3 + estimate_context_tokens(value)
        return total
    if isinstance(params, (list, tuple)):
        return 2 + sum(estimate_context_tokens(item) + 1 for item in params)
    return len(str(params)) + 1


def trim_context_to_budget(
    params: dict[str, Any], budget: ContextBudget
) -> dict[str, Any]:
    """Remove only old whole turns when a final OAuth request exceeds its budget.

    Args:
        params: Final Responses request, including instructions and tools.
        budget: Limits resolved for this request's effective model.

    Returns:
        An independent request, with protected input unchanged.

    Raises:
        ContextBudgetExceeded: If system, schemas or the latest turn cannot fit.
    """
    result = copy.deepcopy(params)
    if estimate_context_tokens(result) <= budget.input_budget:
        return result
    items = result.get("input")
    if not isinstance(items, list):
        raise ContextBudgetExceeded("OAuth context budget exceeded by protected input.")
    # A user turn includes all following assistant calls and their tool results.
    boundaries = [
        index
        for index, item in enumerate(items)
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    if not boundaries:
        raise ContextBudgetExceeded("OAuth context budget exceeded by protected input.")
    for boundary in boundaries[1:]:
        removed = items[:boundary]
        kept = items[boundary:]
        # Preserve system/developer instructions even if embedded in input.
        protected = [
            item
            for item in removed
            if isinstance(item, dict) and item.get("role") in {"system", "developer"}
        ]
        removed = [item for item in removed if item not in protected]
        removed_ids = {
            item.get("call_id")
            for item in removed
            if isinstance(item, dict)
            and item.get("type") in {"function_call", "function_call_output"}
        }
        kept_ids = {
            item.get("call_id")
            for item in [*protected, *kept]
            if isinstance(item, dict)
            and item.get("type") in {"function_call", "function_call_output"}
        }
        if (removed_ids & kept_ids) - {None}:
            continue
        removed_calls = {
            item.get("call_id")
            for item in removed
            if isinstance(item, dict) and item.get("type") == "function_call"
        }
        removed_results = {
            item.get("call_id")
            for item in removed
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        }
        if removed_calls != removed_results:
            continue
        if None in removed_calls or None in removed_results:
            continue
        result["input"] = [*protected, *kept]
        if estimate_context_tokens(result) <= budget.input_budget:
            return result
    raise ContextBudgetExceeded("OAuth context budget exceeded by protected input.")
