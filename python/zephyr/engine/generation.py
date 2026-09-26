"""Convert native increments to text before delivering them to a consumer."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from .. import _C  # pyright: ignore[reportMissingModuleSource]
from ..tokenizer import Tokenizer
from ..tokenizer.detokenizer import IncrementalDecoder
from .inputs import GenerationParams
from .outputs import (
    ChoiceOutput,
    GenerationBuffer,
    GenerationOutput,
    OutputBuffer,
    RequestOutput,
    TextChoice,
    Usage,
)


@dataclass(slots=True)
class _ChoiceState:
    decoder: IncrementalDecoder
    pending: str = ""
    num_tokens: int = 0
    num_characters: int = 0
    finished: bool = False


class GenerationProcessor:
    """Each choice owns its decoder and stop state, independently of delivery speed.

    Output token counts stop at the token that first completes a stop string.
    Native execution may already have sampled further tokens; those remain native
    execution statistics, not part of the reported completion or its logprobs.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        prompt: Sequence[int],
        params: GenerationParams,
        *,
        stop: Sequence[str],
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt = tuple(prompt)
        self._params = params
        self._stops = tuple(stop)
        self._hold_back = max((len(stop) - 1 for stop in self._stops), default=0)
        self._choices: dict[int, _ChoiceState] = {}

    def _drain(self, state: _ChoiceState, *, final: bool) -> tuple[str, str | None]:
        matches = [
            (position, index, stop)
            for index, stop in enumerate(self._stops)
            if (position := state.pending.find(stop)) >= 0
        ]
        if matches:
            position, _, stop = min(matches)
            end = position + (
                len(stop) if self._params.include_stop_str_in_output else 0
            )
            text, state.pending = state.pending[:end], ""
            return text, stop
        end = (
            len(state.pending)
            if final
            else max(0, len(state.pending) - self._hold_back)
        )
        text, state.pending = state.pending[:end], state.pending[end:]
        return text, None

    def process(self, output: _C.RequestOutput | RequestOutput) -> GenerationOutput:
        choices = []
        for choice in cast(Sequence[_C.ChoiceOutput | ChoiceOutput], output.result):
            state = self._choices.get(choice.index)
            if state is None:
                state = _ChoiceState(
                    self._tokenizer.decoder(
                        self._prompt,
                        skip_special_tokens=self._params.skip_special_tokens,
                    )
                )
                self._choices[choice.index] = state
            if state.finished:
                continue
            text: list[str] = []
            tokens: list[_C.SamplingResult] = []
            offsets: list[int] = []
            reason = None
            stop_reason = None
            new_tokens = choice.tokens
            for index, token in enumerate(new_tokens):
                tokens.append(token)
                offsets.append(state.num_characters)
                state.num_tokens += 1
                last = index == len(new_tokens) - 1
                token_stop = last and choice.finish_reason in (
                    _C.FinishReason.EOS,
                    _C.FinishReason.STOP_TOKEN,
                )
                if not token_stop or self._params.include_stop_str_in_output:
                    decoded = state.decoder.push(token.token_id)
                    state.pending += decoded
                    state.num_characters += len(decoded)
                chunk, matched = self._drain(state, final=False)
                text.append(chunk)
                if matched is not None:
                    reason, stop_reason = _C.FinishReason.STOP_STRING, matched
                    break
                if token_stop:
                    stop_reason = token.token_id
            if reason is None and choice.finish_reason is not None:
                state.pending += state.decoder.finish()
                chunk, matched = self._drain(state, final=True)
                text.append(chunk)
                reason = (
                    _C.FinishReason.STOP_STRING if matched else choice.finish_reason
                )
                if matched:
                    stop_reason = matched
            state.finished = reason is not None
            choices.append(
                TextChoice(
                    choice.index,
                    "".join(text),
                    tuple(tokens),
                    reason,
                    stop_reason,
                    tuple(offsets),
                )
            )
        return GenerationOutput(
            output.request_id,
            self._prompt,
            tuple(choices),
            output.status,
            Usage(
                len(self._prompt),
                sum(state.num_tokens for state in self._choices.values()),
            ),
            output.error_message,
        )


def update_output(
    native: _C.Engine,
    output: _C.RequestOutput,
    buffer: "OutputBuffer | GenerationBuffer",
    processor: GenerationProcessor | None,
) -> None:
    """Process stops before buffering so a slow reader cannot delay native retirement."""
    if processor is None:
        cast(OutputBuffer, buffer).add(output)
    else:
        converted = processor.process(output)
        for choice in converted.choices:
            if choice.finish_reason == _C.FinishReason.STOP_STRING:
                native.stop_choice(output.request_id, choice.index)
        cast(GenerationBuffer, buffer).add(converted)
