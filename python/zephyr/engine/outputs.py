"""Immutable output snapshots and per-request accumulation of native increments."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from .. import _C  # pyright: ignore[reportMissingModuleSource]


@dataclass(slots=True, frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(slots=True, frozen=True)
class ChoiceOutput:
    index: int
    tokens: tuple[_C.SamplingResult, ...]
    finish_reason: _C.FinishReason | None = None


@dataclass(slots=True, frozen=True)
class RequestOutput:
    request_id: int
    result: tuple[ChoiceOutput, ...] | tuple[float, ...]
    status: _C.RequestStatus | None = None
    usage: Usage = Usage()
    error_message: str = ""


@dataclass(slots=True)
class _ChoiceBuffer:
    tokens: list[_C.SamplingResult] = field(default_factory=list)
    finish_reason: _C.FinishReason | None = None


class OutputBuffer:
    """Accumulate unread increments; callers synchronize access.

    The byte budget counts payload: 64-bit IDs/counts, 32-bit tokens/floats,
    byte enums, and UTF-8 errors. It excludes Python object/container overhead
    and is not an RSS limit. Taking a snapshot clears only unread data;
    finished remains true after the terminal output has been taken.
    """

    def __init__(self, request_id: int, generation: bool) -> None:
        self._request_id = request_id
        self._generation = generation
        self._choices: dict[int, _ChoiceBuffer] = {}
        self._embedding: tuple[float, ...] = ()
        self._status: _C.RequestStatus | None = None
        self._usage = Usage()
        self._error_message = ""
        self._has_output = False
        self._finished = False
        self._size_bytes = 0

    @property
    def empty(self) -> bool:
        return not self._has_output

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def finished(self) -> bool:
        return self._finished

    def add(self, output: _C.RequestOutput | RequestOutput) -> None:
        if output.request_id != self._request_id:
            raise ValueError("Output belongs to another request")
        if self._finished:
            raise RuntimeError("Request has already finished")
        if not self._has_output:
            self._size_bytes = 8
            self._has_output = True

        if self._generation:
            for choice in cast(Sequence[_C.ChoiceOutput | ChoiceOutput], output.result):
                pending = self._choices.get(choice.index)
                if pending is None:
                    pending = _ChoiceBuffer()
                    self._choices[choice.index] = pending
                    self._size_bytes += 8
                tokens = choice.tokens
                pending.tokens.extend(tokens)
                for token in tokens:
                    self._size_bytes += 4
                    logprobs = token.logprobs
                    if logprobs is not None:
                        self._size_bytes += 4 + 8 * len(logprobs.top_logprobs)
                if choice.finish_reason is not None:
                    if pending.finish_reason is None:
                        self._size_bytes += 1
                    pending.finish_reason = choice.finish_reason
        else:
            embedding = tuple(cast(Sequence[float], output.result))
            self._size_bytes += 4 * (len(embedding) - len(self._embedding))
            self._embedding = embedding

        if output.status is not None:
            self._status = output.status
            usage = output.usage
            self._usage = Usage(usage.prompt_tokens, usage.completion_tokens)
            self._error_message = output.error_message
            self._size_bytes += 17 + len(self._error_message.encode("utf-8"))
            self._finished = True

    def take(self) -> RequestOutput:
        if self.empty:
            raise RuntimeError("No unread output")
        result: tuple[ChoiceOutput, ...] | tuple[float, ...]
        if self._generation:
            result = tuple(
                ChoiceOutput(index, tuple(choice.tokens), choice.finish_reason)
                for index, choice in sorted(self._choices.items())
            )
        else:
            result = self._embedding
        output = RequestOutput(
            self._request_id, result, self._status, self._usage, self._error_message
        )
        self.clear()
        return output

    def clear(self) -> None:
        """Discard unread data without changing the request's finished state."""
        self._choices.clear()
        self._embedding = ()
        self._status = None
        self._usage = Usage()
        self._error_message = ""
        self._has_output = False
        self._size_bytes = 0
