"""Resolve model assets and defaults before constructing the native engine."""

import json
import logging
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from huggingface_hub import snapshot_download

from .. import _C  # pyright: ignore[reportMissingModuleSource]
from ..tokenizer import Tokenizer

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    options: _C.EngineOptions
    tokenizer: Tokenizer | None
    generation_defaults: dict[str, Any]
    model_name: str
    task: _C.ModelTask | None
    kv_cache: _C.KvCacheOptions | None


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Python model loading options; compute and capacity remain native concerns.

    ``model`` and ``tokenizer`` accept local directories or Hub repository IDs.
    An explicit EOS list, including an empty list, overrides checkpoint defaults.
    """

    model: str | Path
    _: KW_ONLY
    devices: Sequence[int] = (0,)
    dtype: str = "bfloat16"
    max_seq_len: int = 0
    gpu_memory_utilization: float = 0.9
    scheduler: _C.LengthBucketSchedulerConfig | _C.PagedSchedulerConfig | None = None
    max_outstanding_sequences: int = 256
    max_buffered_output_bytes: int = 16 * 1024 * 1024
    eos_token_ids: Sequence[int] | None = None
    task: _C.ModelTask | None = None
    kv_cache: _C.KvCacheOptions | None = field(default_factory=_C.KvCacheOptions)
    revision: str | None = None
    tokenizer: str | Path | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    chat_template: str | None = None
    cache_dir: str | Path | None = None

    def resolve(self) -> ResolvedConfig:
        """Resolve weights and tokenizer to fixed local snapshots and read defaults."""
        started = perf_counter()
        _LOGGER.info(
            "Resolving model and tokenizer assets: model=%s revision=%s tokenizer=%s",
            self.model,
            self.revision or "default",
            self.tokenizer or self.model,
        )
        model_dir = self._resolve_directory(self.model, self.revision, weights=True)
        tokenizer_dir = model_dir
        if self.tokenizer_revision is not None or self.tokenizer not in (
            None,
            self.model,
        ):
            tokenizer_dir = self._resolve_directory(
                self.tokenizer or self.model,
                self.tokenizer_revision,
                weights=False,
            )
        _LOGGER.info(
            "Assets ready: model_dir=%s tokenizer_dir=%s elapsed_s=%.2f",
            model_dir,
            tokenizer_dir,
            perf_counter() - started,
        )
        started = perf_counter()
        _LOGGER.info("Loading tokenizer: path=%s", tokenizer_dir)
        tokenizer = Tokenizer(
            model_dir,
            tokenizer_dir,
            trust_remote_code=self.trust_remote_code,
            chat_template=self.chat_template,
        )
        _LOGGER.info(
            "Tokenizer ready: vocab_size=%d elapsed_s=%.2f",
            tokenizer.vocab_size,
            perf_counter() - started,
        )
        model_config = self._read_config(model_dir / "config.json")
        generation = self._read_config(model_dir / "generation_config.json")
        unsupported = {
            "num_beams": 1,
            "num_beam_groups": 1,
            "num_return_sequences": 1,
            "typical_p": 1.0,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
            "encoder_repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "encoder_no_repeat_ngram_size": 0,
            "min_length": 0,
            "min_new_tokens": 0,
            "guidance_scale": 1.0,
            "penalty_alpha": 0.0,
        }
        for key, default in unsupported.items():
            if generation.get(key) not in (None, default):
                raise ValueError(
                    f"Unsupported generation_config setting: {key}={generation[key]!r}"
                )
        for key in (
            "bad_words_ids",
            "suppress_tokens",
            "begin_suppress_tokens",
            "constraints",
            "force_words_ids",
            "forced_bos_token_id",
            "forced_eos_token_id",
            "forced_decoder_ids",
            "sequence_bias",
            "watermarking_config",
        ):
            if generation.get(key) not in (None, [], {}):
                raise ValueError(
                    f"Unsupported generation_config setting: {key}={generation[key]!r}"
                )
        task = self.task
        if task is None and (model_dir / "config_sentence_transformers.json").exists():
            task = _C.ModelTask.EMBEDDING
        scheduler = self.scheduler
        if scheduler is None:
            scheduler = (
                _C.LengthBucketSchedulerConfig()
                if task == _C.ModelTask.EMBEDDING
                else _C.PagedSchedulerConfig()
            )
        eos = self.eos_token_ids
        if eos is None:
            eos = generation.get("eos_token_id")
        if eos is None:
            eos = model_config.get("eos_token_id")
        if eos is None:
            eos = tokenizer.eos_token_ids
        if isinstance(eos, int):
            eos = [eos]

        # Do not instantiate GenerationConfig: its synthetic max_length=20 and
        # other defaults are not model-provided generation policy.
        supported = {
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repetition_penalty",
            "max_new_tokens",
            "max_length",
            "do_sample",
            "stop_strings",
        }
        defaults = {key: value for key, value in generation.items() if key in supported}
        options = _C.EngineOptions(
            model_dir,
            devices=self.devices,
            dtype=self.dtype,
            max_seq_len=self.max_seq_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            scheduler=scheduler,
            max_outstanding_sequences=self.max_outstanding_sequences,
            max_buffered_output_bytes=self.max_buffered_output_bytes,
            eos_token_ids=eos,
        )
        return ResolvedConfig(
            options,
            tokenizer,
            defaults,
            str(self.model),
            task,
            None if task == _C.ModelTask.EMBEDDING else self.kv_cache,
        )

    def _resolve_directory(
        self, model: str | Path, revision: str | None, *, weights: bool
    ) -> Path:
        path = Path(model).expanduser()
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"Model path must be a directory: {path}")
            return path.resolve()
        if isinstance(model, Path) or str(model).startswith(("/", "./", "../", "~")):
            raise FileNotFoundError(f"Model directory does not exist: {path}")

        # Both weights and tokenizer resolve in one snapshot call, preventing a
        # moving Hub branch from pairing different checkpoint revisions.
        patterns = [
            "config.json",
            "generation_config.json",
            "config_sentence_transformers.json",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab*",
            "merges.txt",
            "*.model",
            "*.tiktoken",
            "*.jinja",
            "chat_templates/*",
        ]
        if weights:
            patterns.extend(["*.safetensors", "*.safetensors.index.json"])
        if self.trust_remote_code:
            patterns.append("*.py")
        return Path(
            snapshot_download(
                str(model),
                revision=revision,
                cache_dir=self.cache_dir,
                allow_patterns=patterns,
            )
        )

    @staticmethod
    def _read_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError(f"Model configuration must be a JSON object: {path}")
        return config


def resolve_config(
    model: EngineConfig | str | Path | _C.EngineOptions, **kwargs: Any
) -> ResolvedConfig:
    """Normalize the text API's model options or preserve raw token-only options."""
    if isinstance(model, _C.EngineOptions):
        task = kwargs.pop("task", None)
        kv_cache = kwargs.pop("kv_cache", _C.KvCacheOptions())
        if kwargs:
            raise TypeError(
                f"Unexpected options with EngineOptions: {', '.join(kwargs)}"
            )
        return ResolvedConfig(model, None, {}, model.model_dir, task, kv_cache)
    if isinstance(model, EngineConfig):
        if kwargs:
            raise TypeError("Pass options in EngineConfig, not alongside it")
        return model.resolve()
    return EngineConfig(model, **kwargs).resolve()
