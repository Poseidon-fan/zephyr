"""Hugging Face tokenization without model execution or request state."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer as BackendTokenizer
from transformers import AutoTokenizer

from .detokenizer import IncrementalDecoder


class Tokenizer:
    """One immutable tokenizer shared by an engine's requests.

    Paths are resolved before construction. Remote code is disabled unless the
    caller explicitly enables it; encoding never changes padding or truncation.
    """

    def __init__(
        self,
        model_dir: str | Path,
        tokenizer_dir: str | Path | None = None,
        *,
        trust_remote_code: bool = False,
        chat_template: str | None = None,
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir or model_dir),
            use_fast=True,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        backend = getattr(self._tokenizer, "backend_tokenizer", None)
        if not isinstance(backend, BackendTokenizer):
            raise ValueError(
                "Zephyr requires a Hugging Face fast tokenizer for incremental "
                "decoding. Supply a compatible tokenizer with tokenizer.json."
            )
        self._backend = backend
        self._chat_template = chat_template
        self._vocab_size = max(self._tokenizer.get_vocab().values()) + 1
        decoder = json.loads(backend.to_str()).get("decoder") or {}
        decoders = decoder.get("decoders", [decoder])
        self._byte_fallback = any(
            item.get("type") == "ByteFallback" for item in decoders
        )
        self._byte_decoder: dict[str, int] | None = None
        if decoder.get("type") == "ByteLevel":
            # ByteLevel maps printable Latin-1 directly and the remaining bytes
            # to consecutive Unicode code points starting at 256.
            visible = (
                list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
            )
            missing = [value for value in range(256) if value not in visible]
            self._byte_decoder = {chr(value): value for value in visible}
            self._byte_decoder.update(
                (chr(256 + index), value) for index, value in enumerate(missing)
            )
        self._added_ids = set(backend.get_added_tokens_decoder())

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        eos = self._tokenizer.eos_token_id
        return () if eos is None else (eos,)

    @property
    def vocab_size(self) -> int:
        """Exclusive upper bound of tokenizer IDs, including added tokens."""
        return self._vocab_size

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **template_kwargs: Any,
    ) -> list[int]:
        """Apply the checkpoint's template and tokenize it exactly once."""
        reserved = {
            "conversation",
            "chat_template",
            "tokenize",
            "return_dict",
            "return_tensors",
            "return_assistant_tokens_mask",
            "padding",
            "truncation",
            "max_length",
            "tokenizer_kwargs",
        }
        conflicts = reserved.intersection(template_kwargs)
        if conflicts:
            raise ValueError(
                f"Reserved chat template arguments: {', '.join(sorted(conflicts))}"
            )
        return self._tokenizer.apply_chat_template(
            list(messages),
            chat_template=self._chat_template,
            tokenize=True,
            return_dict=False,
            return_tensors=None,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            **template_kwargs,
        )

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        # Cleanup can rewrite previously emitted spaces, so ordinary and streaming
        # decoding both use the backend's unmodified text.
        return self._backend.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_text(self, token_id: int) -> str:
        """Display one vocabulary item; incomplete UTF-8 uses replacement text."""
        raw = self.token_bytes(token_id)
        if raw is not None:
            return bytes(raw).decode("utf-8", errors="replace")
        return self.decode([token_id], skip_special_tokens=False)

    def token_bytes(self, token_id: int) -> list[int] | None:
        """Return exact bytes where the backend exposes a byte-level encoding.

        Context-dependent decoders have no general per-token byte representation.
        They return None instead of reporting bytes for a replacement character.
        """
        token = self._backend.id_to_token(token_id)
        if token is None:
            return None
        if token_id in self._added_ids:
            return list(token.encode("utf-8"))
        if self._byte_decoder is not None:
            return [self._byte_decoder[character] for character in token]
        if (
            self._byte_fallback
            and len(token) == 6
            and token.startswith("<0x")
            and token.endswith(">")
            and all(character in "0123456789abcdefABCDEF" for character in token[3:5])
        ):
            return [int(token[3:5], 16)]
        return None

    def decoder(
        self,
        prompt_token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> IncrementalDecoder:
        return IncrementalDecoder(
            self._backend, prompt_token_ids, skip_special_tokens=skip_special_tokens
        )
