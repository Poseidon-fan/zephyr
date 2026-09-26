"""Resolve text-generation policy into the native engine's token contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .. import _C  # pyright: ignore[reportMissingModuleSource]
from ..tokenizer import Tokenizer

Prompt = str | Sequence[int]


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationParams:
    """Request overrides; omitted sampling values inherit checkpoint defaults.

    ``max_tokens`` limits generated tokens, including a terminating token. String
    stops affect visible text; their triggering token is still counted in usage.
    ``stop=None`` inherits model stop strings; an empty sequence disables them.
    """

    max_tokens: int | None = None
    n: int = 1
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    logit_bias: Mapping[int, float] | None = None
    top_logprobs: int | None = None
    seed: int | None = None
    stop: str | Sequence[str] | None = None
    stop_token_ids: Sequence[int] = ()
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False
    skip_special_tokens: bool = True

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("n", self.n, 1),
            ("max_tokens", self.max_tokens, 1),
            ("top_logprobs", self.top_logprobs, 0),
            ("seed", self.seed, 0),
        ):
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.seed is not None and self.seed >= 2**64:
            raise ValueError("seed must fit in an unsigned 64-bit integer")
        if self.stop is not None:
            object.__setattr__(self, "stop", self.stop_strings({}))
        object.__setattr__(self, "stop_token_ids", tuple(self.stop_token_ids))
        if self.logit_bias is not None:
            object.__setattr__(self, "logit_bias", dict(self.logit_bias))

    def stop_strings(self, defaults: Mapping[str, Any]) -> tuple[str, ...]:
        """Resolve text stops without treating an explicit empty list as omitted."""
        value = self.stop if self.stop is not None else defaults.get("stop_strings")
        if value is None:
            return ()
        if not isinstance(value, (str, Sequence)):
            raise ValueError("stop must be a string or a sequence of strings")
        stops = (value,) if isinstance(value, str) else tuple(value)
        if any(not isinstance(stop, str) or not stop for stop in stops):
            raise ValueError("stop must contain nonempty strings")
        return stops

    def request(
        self, token_ids: Sequence[int], defaults: Mapping[str, Any], info: _C.EngineInfo
    ) -> _C.GenerationRequest:
        if info.task != _C.ModelTask.GENERATION:
            raise ValueError("The loaded model does not support text generation")
        remaining = info.max_seq_len - len(token_ids)
        if remaining <= 0:
            raise ValueError(
                "The prompt must leave room for generation within max_seq_len"
            )
        maximum = self.max_tokens
        if maximum is None:
            maximum = defaults.get("max_new_tokens")
            if maximum is None and defaults.get("max_length") is not None:
                maximum = defaults["max_length"] - len(token_ids)
            maximum = min(remaining, maximum) if maximum is not None else remaining
        if type(maximum) is not int or not 0 < maximum <= remaining:
            raise ValueError(
                f"max_tokens must be between 1 and {remaining} for this prompt"
            )
        names = (
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
        )
        sampling = {}
        for name in names:
            value = getattr(self, name)
            if value is None:
                value = defaults.get(name)
            if value is not None:
                sampling[name] = value
        if self.temperature is None and defaults.get("do_sample") is False:
            sampling["temperature"] = 0.0
        if any(not math.isfinite(value) for value in sampling.values()):
            raise ValueError("Sampling parameters must be finite")
        if self.top_logprobs is not None and self.top_logprobs > info.vocab_size:
            raise ValueError("top_logprobs exceeds the model vocabulary size")
        return _C.GenerationRequest(
            token_ids,
            sampling=_C.SamplingParams(
                **sampling,
                logits_bias={
                    token: bias for token, bias in (self.logit_bias or {}).items()
                },
                top_logprobs=self.top_logprobs,
            ),
            max_new_tokens=maximum,
            stop_token_ids=self.stop_token_ids,
            seed=self.seed,
            num_choices=self.n,
            ignore_eos=self.ignore_eos,
        )


def encode_prompt(
    tokenizer: Tokenizer, prompt: Prompt, info: _C.EngineInfo
) -> list[int]:
    tokens = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    if not tokens or len(tokens) > info.max_seq_len:
        raise ValueError(f"Input must contain between 1 and {info.max_seq_len} tokens")
    # The model vocabulary can include padded IDs beyond the tokenizer's size.
    # Input validity follows the model's embedding table bounds.
    if any(
        type(token) is not int or not 0 <= token < info.vocab_size for token in tokens
    ):
        raise ValueError("Input contains a token ID outside the model vocabulary")
    return tokens
