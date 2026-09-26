"""Incremental decoding preserves tokenizer context and unfinished UTF-8."""

from collections.abc import Sequence
from os.path import commonprefix

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream


class IncrementalDecoder:
    """A single choice's decode state; never shared between requests.

    DecodeStream retains the context needed by space-sensitive and byte-fallback
    tokenizers. A final full decode flushes any incomplete byte suffix because
    DecodeStream does not expose a flush operation.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        prompt_token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> None:
        self._tokenizer = tokenizer
        self._skip_special_tokens = skip_special_tokens
        self._ids = list(prompt_token_ids)
        self._prompt_length = len(self._ids)
        self._stream = DecodeStream(
            ids=self._ids, skip_special_tokens=skip_special_tokens
        )
        prompt = tokenizer.decode(self._ids, skip_special_tokens=skip_special_tokens)
        self._prefix = prompt
        # A prompt ending inside a UTF-8 character has no stable DecodeStream
        # prefix. Its first returned chunk includes the stable prompt text.
        self._unprimed_prefix = prompt.endswith("\ufffd")
        self._chunks: list[str] = []
        self._finished = False

    def push(self, token_id: int) -> str:
        if self._finished:
            raise RuntimeError("Cannot append tokens after finishing a decoder")
        self._ids.append(token_id)
        text = self._stream.step(self._tokenizer, token_id) or ""
        if text and self._unprimed_prefix:
            self._prefix = commonprefix([self._prefix, text])
            text = text[len(self._prefix) :]
            self._unprimed_prefix = False
        if text:
            self._chunks.append(text)
        return text

    def finish(self) -> str:
        """Return the unreported tail, including terminal replacement characters."""
        if self._finished:
            return ""
        self._finished = True
        if len(self._ids) == self._prompt_length:
            return ""
        text = self._tokenizer.decode(
            self._ids, skip_special_tokens=self._skip_special_tokens
        )
        if self._unprimed_prefix:
            self._prefix = commonprefix([self._prefix, text])
        emitted = self._prefix + "".join(self._chunks)
        if not text.startswith(emitted):
            raise ValueError(
                "Tokenizer decoding rewrote previously emitted text; this decoder "
                "is incompatible with append-only streaming"
            )
        return text[len(emitted) :]
