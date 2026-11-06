import copy
from collections import Counter

from astrbot import logger
from astrbot.core.provider.context_budget import ContextBudgetExceeded

from ..message import Message
from .compressor import LLMSummaryCompressor, TruncateByTurnsCompressor
from .config import ContextConfig
from .round_utils import split_into_rounds
from .token_counter import EstimateTokenCounter
from .truncator import ContextTruncator


class ContextManager:
    """Context compression manager."""

    def __init__(
        self,
        config: ContextConfig,
    ) -> None:
        """Initialize the context manager.

        There are two strategies to handle context limit reached:
        1. Truncate by turns: remove older messages by turns.
        2. LLM-based compression: use LLM to summarize old messages.

        Args:
            config: The context configuration.
        """
        self.config = config

        self.token_counter = config.custom_token_counter or EstimateTokenCounter(
            conservative=config.conservative_token_count
        )
        self.truncator = ContextTruncator()

        if config.custom_compressor:
            self.compressor = config.custom_compressor
        elif config.llm_compress_provider:
            self.compressor = LLMSummaryCompressor(
                provider=config.llm_compress_provider,
                keep_recent_ratio=config.llm_compress_keep_recent_ratio,
                instruction_text=config.llm_compress_instruction,
                token_counter=self.token_counter,
                compression_threshold=config.compression_threshold,
            )
        else:
            self.compressor = TruncateByTurnsCompressor(
                truncate_turns=config.truncate_turns,
                compression_threshold=config.compression_threshold,
            )

    async def process(
        self, messages: list[Message], trusted_token_usage: int = 0
    ) -> list[Message]:
        """Process the messages.

        Args:
            messages: The original message list.

        Returns:
            The processed message list.
        """
        try:
            if self.config.reject_oversized_input:
                return await self._process_with_hard_limit(messages)
            result = messages

            # 1. 基于轮次的截断 (Enforce max turns)
            if self.config.enforce_max_turns != -1:
                result = self.truncator.truncate_by_turns(
                    result,
                    keep_most_recent_turns=self.config.enforce_max_turns,
                    drop_turns=self.config.truncate_turns,
                )

            # 2. 基于 token 的压缩
            if self.config.max_context_tokens > 0:
                total_tokens = self.token_counter.count_tokens(
                    result, trusted_token_usage
                )
                total_tokens += self.config.reserved_context_tokens

                if self.compressor.should_compress(
                    result, total_tokens, self.config.max_context_tokens
                ):
                    result = await self._run_compression(result, total_tokens)

            return result
        except ContextBudgetExceeded:
            raise
        except Exception as e:
            logger.error(f"Error during context processing: {e}", exc_info=True)
            if self.config.reject_oversized_input:
                raise ContextBudgetExceeded(
                    "Unable to validate the context budget."
                ) from e
            return messages

    async def _process_with_hard_limit(self, messages: list[Message]) -> list[Message]:
        """Apply the selected policy without removing protected OAuth input.

        Args:
            messages: Current messages, including newly appended tool results.

        Returns:
            Messages fitting the hard input allowance, with protected input intact.

        Raises:
            ContextBudgetExceeded: If protected input cannot fit or is removed.
        """
        limit = self.config.max_context_tokens
        reserve = max(0, self.config.reserved_context_tokens)
        if limit <= 0:
            raise ContextBudgetExceeded("The context budget has no input allowance.")
        instruction_ids = {
            id(message)
            for message in messages
            if message.role in {"system", "developer"}
        }
        rounds = split_into_rounds(
            [message for message in messages if id(message) not in instruction_ids]
        )
        protected_ids = set(instruction_ids)
        call_rounds: dict[str | None, set[int]] = {}
        result_rounds: dict[str | None, set[int]] = {}
        for index, current in enumerate(rounds):
            for message in current:
                for call in message.tool_calls or []:
                    call_id = call.get("id") if isinstance(call, dict) else call.id
                    call_rounds.setdefault(call_id, set()).add(index)
                if message.role == "tool":
                    result_rounds.setdefault(message.tool_call_id, set()).add(index)
        protected_rounds = {len(rounds) - 1} if rounds else set()
        for call_id in call_rounds.keys() | result_rounds.keys():
            starts = call_rounds.get(call_id, set())
            finishes = result_rounds.get(call_id, set())
            linked = starts | finishes
            # Keep both ends of a cross-round group, and preserve incomplete or
            # orphan groups so compression cannot conceal malformed input.
            if call_id is None or not starts or not finishes or len(linked) > 1:
                protected_rounds.update(linked)
        protected_ids.update(
            id(message) for index in protected_rounds for message in rounds[index]
        )
        protected = [message for message in messages if id(message) in protected_ids]
        protected_keys = Counter(message.model_dump_json() for message in protected)
        if self.token_counter.count_tokens(protected) + reserve > limit:
            raise ContextBudgetExceeded("Protected input exceeds the context budget.")

        # Enforce the profile's turn policy using whole user/tool rounds. Pending
        # call groups and the latest request may outlive an older-turn limit.
        max_turns = self.config.enforce_max_turns
        if max_turns != -1 and len(rounds) > max_turns:
            keep = max(1, max_turns - self.config.truncate_turns + 1)
            removable = [
                current
                for current in rounds
                if not any(id(message) in protected_ids for message in current)
            ]
            removed = {
                id(message)
                for current in removable[: max(0, len(rounds) - keep)]
                for message in current
            }
            messages = [message for message in messages if id(message) not in removed]
            rounds = split_into_rounds(
                [message for message in messages if id(message) not in instruction_ids]
            )

        tokens = self.token_counter.count_tokens(messages) + reserve
        if tokens <= limit and not self.compressor.should_compress(
            messages, tokens, limit
        ):
            return messages
        if isinstance(self.compressor, TruncateByTurnsCompressor):
            removable = [
                current
                for current in rounds
                if not any(id(message) in protected_ids for message in current)
            ]
            removed = {
                id(message)
                for current in removable[: max(0, self.config.truncate_turns)]
                for message in current
            }
            messages = [message for message in messages if id(message) not in removed]
        else:
            compressor = self.compressor
            if isinstance(compressor, LLMSummaryCompressor):
                # SDK callers may reuse a custom compressor across runners.
                # Request identities must never be written onto that shared object.
                compressor = copy.copy(compressor)
                compressor.preserve_message_ids = protected_ids
            try:
                messages = await compressor(messages)
            except ContextBudgetExceeded:
                raise
            except Exception:
                logger.warning(
                    "Context compression failed; dropping old complete rounds.",
                    exc_info=True,
                )

        remaining_keys = Counter(message.model_dump_json() for message in messages)
        if protected_keys - remaining_keys:
            raise ContextBudgetExceeded(
                "Context budget compression removed protected input."
            )
        surviving_protected_ids = {
            id(message) for message in messages if id(message) in protected_ids
        }
        needed = protected_keys - Counter(
            message.model_dump_json()
            for message in messages
            if id(message) in surviving_protected_ids
        )
        # Custom compressors may reconstruct equivalent objects. Prefer original
        # identities, then only the needed most-recent equivalents, not every old
        # user message that happens to contain identical text.
        for message in reversed(messages):
            key = message.model_dump_json()
            if id(message) not in surviving_protected_ids and needed[key] > 0:
                surviving_protected_ids.add(id(message))
                needed[key] -= 1
        # A failed/oversized summary only falls back to removing whole old rounds.
        # No second, hidden summary request is issued by this deterministic pass.
        tokens = self.token_counter.count_tokens(messages) + reserve
        current_rounds = split_into_rounds(
            [
                message
                for message in messages
                if message.role not in {"system", "developer"}
            ]
        )
        for current in current_rounds:
            if tokens <= limit and not self.compressor.should_compress(
                messages, tokens, limit
            ):
                break
            if any(id(message) in surviving_protected_ids for message in current):
                continue
            calls = {
                call.get("id") if isinstance(call, dict) else call.id
                for message in current
                for call in (message.tool_calls or [])
            }
            results = {
                message.tool_call_id for message in current if message.role == "tool"
            }
            if None in calls or calls - results:
                continue
            removed = {id(message) for message in current}
            messages = [message for message in messages if id(message) not in removed]
            tokens = self.token_counter.count_tokens(messages) + reserve
        if tokens > limit:
            raise ContextBudgetExceeded(
                "Protected input exceeds the context budget after compression."
            )
        return messages

    async def _run_compression(
        self, messages: list[Message], prev_tokens: int
    ) -> list[Message]:
        """
        Compress/truncate the messages.

        If the context still exceeds the limit after compression (for example
        because the summary request failed), whole rounds are dropped, oldest
        first, until it fits. The latest round is always kept, even when it
        alone exceeds the limit.

        Args:
            messages: The original message list.
            prev_tokens: The token count before compression.

        Returns:
            The compressed/truncated message list.
        """
        logger.debug("Compress triggered, starting compression...")

        messages = await self.compressor(messages)

        # double check
        tokens_after_summary = (
            self.token_counter.count_tokens(messages)
            + self.config.reserved_context_tokens
        )

        # calculate compress rate
        compress_rate = (tokens_after_summary / self.config.max_context_tokens) * 100
        logger.info(
            f"Compress completed."
            f" {prev_tokens} -> {tokens_after_summary} tokens,"
            f" compression rate: {compress_rate:.2f}%.",
        )

        # last check
        if not self.compressor.should_compress(
            messages, tokens_after_summary, self.config.max_context_tokens
        ):
            return messages

        logger.info(
            "Context still exceeds max tokens after compression, dropping the oldest rounds..."
        )
        # Drop whole rounds, oldest first, so that tool calls stay with their
        # results, and recount after each one. The latest round is the current
        # request and is always kept.
        first_non_system = next(
            (i for i, msg in enumerate(messages) if msg.role != "system"),
            len(messages),
        )
        system_messages = messages[:first_non_system]
        rounds = [
            [seg for seg in rnd if isinstance(seg, Message)]
            for rnd in split_into_rounds(messages[first_non_system:])
        ]
        tokens = tokens_after_summary
        dropped_rounds = 0
        while len(rounds) > 1 and self.compressor.should_compress(
            messages, tokens, self.config.max_context_tokens
        ):
            rounds.pop(0)
            dropped_rounds += 1
            messages = system_messages + [msg for rnd in rounds for msg in rnd]
            tokens = (
                self.token_counter.count_tokens(messages)
                + self.config.reserved_context_tokens
            )
        logger.info(f"Dropped {dropped_rounds} oldest round(s).")

        if self.compressor.should_compress(
            messages, tokens, self.config.max_context_tokens
        ):
            logger.warning(
                "Context still exceeds max tokens with only the latest round left; "
                "sending it as is."
            )
        return messages
