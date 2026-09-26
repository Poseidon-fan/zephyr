"""Synchronous inference over the native engine's single output consumer."""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Any, cast

from .. import _C  # pyright: ignore[reportMissingModuleSource]
from ..tokenizer import Tokenizer
from .config import EngineConfig, ResolvedConfig, resolve_config
from .generation import GenerationProcessor, update_output
from .inputs import GenerationParams, Prompt, encode_prompt
from .outputs import GenerationBuffer, GenerationOutput, OutputBuffer, RequestOutput

Request = _C.GenerationRequest | _C.EmbeddingRequest
EngineOutput = RequestOutput | GenerationOutput
Model = EngineConfig | str | Path | _C.EngineOptions
_LOGGER = logging.getLogger(__name__)


def _load_engine(
    model: Model, kwargs: dict[str, Any]
) -> tuple[_C.Engine, ResolvedConfig]:
    config = resolve_config(model, **kwargs)
    _LOGGER.info(
        "Initializing native engine: model=%s devices=%s dtype=%s max_seq_len=%s",
        config.model_name,
        config.options.devices,
        config.options.dtype,
        config.options.max_seq_len or "model default",
    )
    started = perf_counter()
    native = _C.Engine(config.options, task=config.task, kv_cache=config.kv_cache)
    _LOGGER.info(
        "Native engine ready: task=%s max_seq_len=%d elapsed_s=%.2f",
        native.info.task.name,
        native.info.max_seq_len,
        perf_counter() - started,
    )
    return native, config


class Engine:
    """One model, synchronous batch generation, and deterministic stream cleanup.

    A model path or EngineConfig enables text APIs and checkpoint defaults.
    Native EngineOptions retains the explicit, token-only integration contract.
    Only one execute/generate/stream operation may consume outputs at a time.
    """

    def __init__(self, model: Model, **kwargs: Any) -> None:
        self._native, self._config = _load_engine(model, kwargs)
        self._consumer_lock = threading.Lock()
        self._closed = False
        self._failed = False

    @property
    def info(self) -> _C.EngineInfo:
        return self._native.info

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def tokenizer(self) -> Tokenizer:
        if self._config.tokenizer is None:
            raise ValueError("Text APIs require a model path or EngineConfig")
        return self._config.tokenizer

    def execute(self, requests: Iterable[Request]) -> list[RequestOutput]:
        """Execute token requests in batches and preserve their input order."""
        return cast(
            list[RequestOutput], self._execute((request, None) for request in requests)
        )

    def generate(
        self,
        prompts: Prompt | Iterable[Prompt],
        params: GenerationParams | None = None,
        **kwargs: Any,
    ) -> GenerationOutput | list[GenerationOutput]:
        """Generate one completion or a batch; keyword arguments override params."""
        params = (
            replace(params, **kwargs)
            if params is not None
            else GenerationParams(**kwargs)
        )
        single = isinstance(prompts, str) or (
            isinstance(prompts, Sequence)
            and bool(prompts)
            and isinstance(prompts[0], int)
        )
        inputs = [cast(Prompt, prompts)] if single else cast(Iterable[Prompt], prompts)

        def prepared() -> Iterator[tuple[Request, GenerationProcessor]]:
            for prompt in inputs:
                tokens = encode_prompt(self.tokenizer, prompt, self.info)
                yield (
                    params.request(tokens, self._config.generation_defaults, self.info),
                    GenerationProcessor(
                        self.tokenizer,
                        tokens,
                        params,
                        stop=params.stop_strings(self._config.generation_defaults),
                    ),
                )

        outputs = cast(list[GenerationOutput], self._execute(prepared()))
        return outputs[0] if single else outputs

    def stream_generate(
        self,
        prompt: Prompt,
        params: GenerationParams | None = None,
        **kwargs: Any,
    ) -> Generator[GenerationOutput, None, None]:
        params = (
            replace(params, **kwargs)
            if params is not None
            else GenerationParams(**kwargs)
        )
        tokens = encode_prompt(self.tokenizer, prompt, self.info)
        request = params.request(tokens, self._config.generation_defaults, self.info)
        processor = GenerationProcessor(
            self.tokenizer,
            tokens,
            params,
            stop=params.stop_strings(self._config.generation_defaults),
        )
        yield from cast(
            Generator[GenerationOutput, None, None], self._stream(request, processor)
        )

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        params: GenerationParams | None = None,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> GenerationOutput:
        template_options = dict(chat_template_kwargs or {})
        template_options.update(
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        tokens = self.tokenizer.encode_chat(messages, **template_options)
        return cast(GenerationOutput, self.generate(tokens, params, **kwargs))

    def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        params: GenerationParams | None = None,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> Generator[GenerationOutput, None, None]:
        template_options = dict(chat_template_kwargs or {})
        template_options.update(
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        tokens = self.tokenizer.encode_chat(messages, **template_options)
        yield from self.stream_generate(tokens, params, **kwargs)

    def embed(self, prompt: Prompt) -> RequestOutput:
        if self.info.task != _C.ModelTask.EMBEDDING:
            raise ValueError("The loaded model does not support embeddings")
        tokens = encode_prompt(self.tokenizer, prompt, self.info)
        return self.execute([_C.EmbeddingRequest(tokens)])[0]

    def _execute(
        self,
        requests: Iterable[tuple[Request, GenerationProcessor | None]],
    ) -> list[EngineOutput]:
        with self._operation() as pending:
            iterator = iter(requests)
            buffers: dict[
                int,
                tuple[int, OutputBuffer | GenerationBuffer, GenerationProcessor | None],
            ] = {}
            results: list[EngineOutput | None] = []
            exhausted = False
            waiting = None
            while not exhausted or pending:
                while not exhausted:
                    if waiting is None:
                        try:
                            waiting = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                    request, processor = waiting
                    try:
                        request_id = self._native.submit(request)
                    except _C.OverloadedError:
                        if not pending:
                            raise
                        break
                    pending.add(request_id)
                    buffer = (
                        GenerationBuffer()
                        if processor
                        else OutputBuffer(
                            request_id, isinstance(request, _C.GenerationRequest)
                        )
                    )
                    buffers[request_id] = (len(results), buffer, processor)
                    results.append(None)
                    waiting = None
                if not pending:
                    continue
                for output in self._wait(pending):
                    index, buffer, processor = buffers[output.request_id]
                    update_output(self._native, output, buffer, processor)
                    if buffer.finished:
                        results[index] = buffer.take()
                        del buffers[output.request_id]
            return cast(list[EngineOutput], results)

    def stream(self, request: Request) -> Generator[RequestOutput, None, None]:
        """Yield token increments; close the iterator early to cancel unfinished work."""
        yield from cast(
            Generator[RequestOutput, None, None], self._stream(request, None)
        )

    def _stream(
        self,
        request: Request,
        processor: GenerationProcessor | None,
    ) -> Generator[EngineOutput, None, None]:
        with self._operation() as pending:
            request_id = self._native.submit(request)
            pending.add(request_id)
            buffer = (
                GenerationBuffer()
                if processor
                else OutputBuffer(request_id, isinstance(request, _C.GenerationRequest))
            )
            while pending:
                for output in self._wait(pending):
                    update_output(self._native, output, buffer, processor)
                if not buffer.empty:
                    yield buffer.take()

    def close(self) -> None:
        """Wake any waiting consumer and release native workers and GPU resources."""
        self._closed = True
        self._native.close()

    def __enter__(self) -> Engine:
        self._check_open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException:
            if exception_type is None:
                raise
            _LOGGER.exception(
                "Engine shutdown failed while propagating an earlier exception"
            )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Engine is closed")
        if self._failed:
            raise RuntimeError("Engine has failed")

    @contextmanager
    def _operation(self) -> Iterator[set[int]]:
        if not self._consumer_lock.acquire(blocking=False):
            raise RuntimeError(
                "Engine already has an active execute or stream consumer"
            )
        pending: set[int] = set()
        try:
            self._check_open()
            try:
                yield pending
            except BaseException:
                try:
                    self._cancel_and_drain(pending)
                except BaseException:
                    _LOGGER.exception(
                        "Request cleanup failed while propagating an earlier exception"
                    )
                raise
            else:
                self._cancel_and_drain(pending)
        finally:
            self._consumer_lock.release()

    def _wait(self, pending: set[int]) -> list[_C.RequestOutput]:
        try:
            outputs = self._native.wait_for_outputs()
        except Exception:
            self._failed = True
            raise
        if not outputs:
            self._failed = True
            raise RuntimeError(
                "Native engine stopped before all submitted requests completed"
            )
        # Native delivery has already released all terminal admission credits.
        # Retire the complete batch before result processing can fail.
        for output in outputs:
            if output.status is not None:
                pending.discard(output.request_id)
        return outputs

    def _cancel_and_drain(self, pending: set[int]) -> None:
        if self._failed:
            return
        for request_id in pending:
            self._native.cancel(request_id)
        while pending:
            self._wait(pending)
