"""Async request delivery over the native engine's single output stream."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from functools import partial
from types import TracebackType
from typing import Any, TypeVar, cast

from .. import _C  # pyright: ignore[reportMissingModuleSource]
from ..tokenizer import Tokenizer
from .config import ResolvedConfig
from .engine import Model, _load_engine
from .generation import GenerationProcessor, update_output
from .inputs import GenerationParams, Prompt, encode_prompt
from .outputs import GenerationBuffer, GenerationOutput, OutputBuffer, RequestOutput

_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


class OutputBufferError(RuntimeError):
    """A request's unread output exhausted the frontend's payload budget."""


@dataclass(slots=True)
class _RequestState:
    output: OutputBuffer | GenerationBuffer
    num_choices: int
    processor: GenerationProcessor | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    error: Exception | None = None
    abandoned: bool = False


async def _finish_cleanup(task: asyncio.Task[None]) -> None:
    """Finish resource cleanup even when the caller is canceled repeatedly."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()


class AsyncEngine:
    """Concurrent text and token inference bound to one asyncio event loop.

    Construct with ``await AsyncEngine.create(...)`` and close with ``aclose``
    or an async context manager. Requests and sampling settings are native value
    snapshots. Text encoding runs in a bounded CPU lane; native waiting and
    shutdown use independent threads so neither can starve the other.

    A single receiver owns native output consumption. The native admission and
    payload limits also bound frontend unread requests and output independently;
    one received batch and Python object overhead are outside the payload budget.
    On overflow, the largest unread request is canceled with OutputBufferError.
    """

    def __init__(
        self,
        native: _C.Engine,
        config: ResolvedConfig,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._native = native
        self._config = config
        self._loop = asyncio.get_running_loop()
        self._executor = executor
        self._tokenizer_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="zephyr-text"
        )
        self._preparing = 0
        self._requests: dict[int, _RequestState] = {}
        self._num_choices = 0
        self._buffered_bytes = 0
        self._max_choices = config.options.max_outstanding_sequences
        self._max_bytes = config.options.max_buffered_output_bytes
        self._closing = False
        self._failure: Exception | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._receiver = self._loop.create_task(self._receive())

    @classmethod
    async def create(
        cls,
        model: Model,
        **kwargs: Any,
    ) -> AsyncEngine:
        loop = asyncio.get_running_loop()
        # Closing must be able to run while the receiver blocks in native code.
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="zephyr")
        loading = loop.run_in_executor(executor, partial(_load_engine, model, kwargs))
        try:
            native, config = await asyncio.shield(loading)
            return cls(native, config, executor)
        except BaseException:

            async def cleanup() -> None:
                try:
                    loaded, _ = await loading
                except Exception:
                    return  # Native construction cleans up a failed initialization.
                try:
                    await loop.run_in_executor(executor, loaded.close)
                except Exception:
                    _LOGGER.exception(
                        "Engine cleanup failed after interrupted creation"
                    )

            try:
                await _finish_cleanup(loop.create_task(cleanup()))
            finally:
                executor.shutdown(wait=False)
            raise

    async def stream(
        self, request: _C.GenerationRequest | _C.EmbeddingRequest
    ) -> AsyncGenerator[RequestOutput, None]:
        """Yield token increments or one embedding, including the native terminal status.

        Cancellation or explicit iterator ``aclose()`` cancels unfinished native
        work. Use ``contextlib.aclosing`` when breaking out of iteration early.
        Request ERROR/REJECTED statuses remain results; engine failures raise.
        """
        async with aclosing(self._stream(request)) as stream:
            async for output in stream:
                yield cast(RequestOutput, output)

    def _check_open(self) -> None:
        self._check_loop()
        if self._failure is not None:
            raise self._failure
        if self._closing:
            raise RuntimeError("Engine is closing")

    def _admit(self, choices: int) -> None:
        if (
            0 < choices <= self._max_choices
            and choices > self._max_choices - self._num_choices - self._preparing
        ):
            raise _C.OverloadedError(
                "Consume outstanding outputs before submitting more"
            )

    async def _stream(
        self,
        request: _C.GenerationRequest | _C.EmbeddingRequest,
        processor: GenerationProcessor | None = None,
    ) -> AsyncGenerator[RequestOutput | GenerationOutput, None]:
        self._check_open()
        generation = isinstance(request, _C.GenerationRequest)
        choices = (
            request.num_choices if isinstance(request, _C.GenerationRequest) else 1
        )
        # Invalid choice counts still go through native argument validation.
        self._admit(choices)

        request_id = self._native.submit(request)
        buffer = (
            GenerationBuffer() if processor else OutputBuffer(request_id, generation)
        )
        state = _RequestState(buffer, choices, processor)
        # No await between submit and registration: the receiver cannot race it.
        self._requests[request_id] = state
        self._num_choices += choices
        try:
            while True:
                await state.ready.wait()
                if state.error is not None:
                    raise state.error
                size = state.output.size_bytes
                output = state.output.take()
                self._buffered_bytes -= size
                state.ready.clear()
                if output.status is not None:
                    self._retire(request_id)
                yield output
                if output.status is not None:
                    return
        finally:
            self._abandon(request_id, state)

    async def execute(
        self, request: _C.GenerationRequest | _C.EmbeddingRequest
    ) -> RequestOutput:
        """Collect one request's complete result, preserving every choice and logprob."""
        result: OutputBuffer | None = None
        async with aclosing(self.stream(request)) as stream:
            async for output in stream:
                if result is None:
                    result = OutputBuffer(
                        output.request_id, isinstance(request, _C.GenerationRequest)
                    )
                result.add(output)
        if result is None:
            raise RuntimeError("Engine ended a request without an output")
        return result.take()

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

    @property
    def is_running(self) -> bool:
        return not self._closing and self._failure is None and not self._receiver.done()

    async def _prepare(self, choices: int, function: Callable[[], _T]) -> _T:
        self._check_open()
        self._admit(choices)
        if choices > self._max_choices:
            raise ValueError("n exceeds max_outstanding_sequences")
        future = self._loop.run_in_executor(self._tokenizer_executor, function)
        self._preparing += choices

        def finished(completed: asyncio.Future[_T]) -> None:
            self._preparing -= choices
            if not completed.cancelled():
                completed.exception()  # Also retrieve failures after a caller disconnects.

        future.add_done_callback(finished)
        # A running tokenizer job cannot be canceled. Keep its admission credit
        # until it really finishes, even when the request no longer awaits it.
        return await asyncio.shield(future)

    def _generation_input(
        self,
        prompt: Prompt | Sequence[Mapping[str, Any]],
        params: GenerationParams,
        template_options: dict[str, Any] | None = None,
    ) -> tuple[_C.GenerationRequest, GenerationProcessor]:
        if template_options is not None:
            prompt = self.tokenizer.encode_chat(
                cast(Sequence[Mapping[str, Any]], prompt), **template_options
            )
        tokens = encode_prompt(self.tokenizer, cast(Prompt, prompt), self.info)
        request = params.request(tokens, self._config.generation_defaults, self.info)
        return request, GenerationProcessor(
            self.tokenizer,
            tokens,
            params,
            stop=params.stop_strings(self._config.generation_defaults),
        )

    async def stream_generate(
        self,
        prompt: Prompt,
        params: GenerationParams | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[GenerationOutput, None]:
        params = (
            replace(params, **kwargs)
            if params is not None
            else GenerationParams(**kwargs)
        )
        request, processor = await self._prepare(
            params.n, partial(self._generation_input, prompt, params)
        )
        async with aclosing(self._stream(request, processor)) as stream:
            async for output in stream:
                yield cast(GenerationOutput, output)

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        params: GenerationParams | None = None,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[GenerationOutput, None]:
        params = (
            replace(params, **kwargs)
            if params is not None
            else GenerationParams(**kwargs)
        )
        template_options = dict(chat_template_kwargs or {})
        template_options.update(
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        request, processor = await self._prepare(
            params.n,
            partial(self._generation_input, messages, params, template_options),
        )
        async with aclosing(self._stream(request, processor)) as stream:
            async for output in stream:
                yield cast(GenerationOutput, output)

    @staticmethod
    async def _collect(
        stream: AsyncGenerator[GenerationOutput, None],
    ) -> GenerationOutput:
        result = GenerationBuffer()
        async with aclosing(stream):
            async for output in stream:
                result.add(output)
        return result.take()

    async def generate(
        self,
        prompt: Prompt,
        params: GenerationParams | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        return await self._collect(self.stream_generate(prompt, params, **kwargs))

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        params: GenerationParams | None = None,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> GenerationOutput:
        return await self._collect(
            self.stream_chat(
                messages,
                params,
                chat_template_kwargs=chat_template_kwargs,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                **kwargs,
            )
        )

    async def embed(self, prompt: Prompt) -> RequestOutput:
        if self.info.task != _C.ModelTask.EMBEDDING:
            raise ValueError("The loaded model does not support embeddings")
        tokens = await self._prepare(
            1, partial(encode_prompt, self.tokenizer, prompt, self.info)
        )
        return await self.execute(_C.EmbeddingRequest(tokens))

    def _check_loop(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("AsyncEngine must be used on its creating event loop")

    def _retire(self, request_id: int) -> None:
        state = self._requests.pop(request_id)
        self._num_choices -= state.num_choices

    def _abandon(self, request_id: int, state: _RequestState) -> None:
        if request_id not in self._requests:
            return
        state.abandoned = True
        self._buffered_bytes -= state.output.size_bytes
        state.output.clear()
        if not state.output.finished:
            self._native.cancel(request_id)
        if state.output.finished or self._receiver.done():
            self._retire(request_id)

    def _limit_outputs(self) -> None:
        while self._buffered_bytes > self._max_bytes:
            request_id, state = max(
                self._requests.items(), key=lambda entry: entry[1].output.size_bytes
            )
            _LOGGER.warning(
                "Canceling request %d: unread output exceeded the payload budget "
                "(buffered_bytes=%d request_bytes=%d budget_bytes=%d)",
                request_id,
                self._buffered_bytes,
                state.output.size_bytes,
                self._max_bytes,
            )
            state.error = OutputBufferError(
                f"Unread output exceeded the {self._max_bytes}-byte payload budget"
            )
            self._abandon(request_id, state)
            state.ready.set()

    async def _receive(self) -> None:
        try:
            while True:
                outputs = await self._loop.run_in_executor(
                    self._executor, self._native.wait_for_outputs
                )
                if not outputs:
                    if not self._closing:
                        raise RuntimeError("Native engine stopped unexpectedly")
                    break
                for output in outputs:
                    state = self._requests.get(output.request_id)
                    if state is None:
                        continue
                    if state.abandoned:
                        if output.status is not None:
                            self._retire(output.request_id)
                        continue
                    size = state.output.size_bytes
                    update_output(self._native, output, state.output, state.processor)
                    self._buffered_bytes += state.output.size_bytes - size
                    state.ready.set()
                self._limit_outputs()
                # Let consumers run even when native output is immediately ready.
                await asyncio.sleep(0)
        except Exception as error:
            self._failure = error
            self._closing = True
        finally:
            for request_id, state in list(self._requests.items()):
                if state.abandoned:
                    self._retire(request_id)
                elif not state.output.finished:
                    state.error = self._failure or RuntimeError(
                        "Engine closed without a terminal request output"
                    )
                    state.ready.set()

    async def _shutdown(self) -> None:
        try:
            await self._loop.run_in_executor(self._executor, self._native.close)
        finally:
            try:
                await self._receiver
            finally:
                try:
                    # Encoding already running cannot be canceled; wait off the
                    # event loop so close returns with no background text jobs.
                    await self._loop.run_in_executor(
                        self._executor,
                        partial(
                            self._tokenizer_executor.shutdown,
                            wait=True,
                            cancel_futures=True,
                        ),
                    )
                finally:
                    self._executor.shutdown(wait=False)
        if self._failure is not None:
            raise self._failure

    async def aclose(self) -> None:
        """Close native execution, drain terminal outputs, then release receiver resources."""
        self._check_loop()
        if self._close_task is None:
            self._closing = True
            self._close_task = self._loop.create_task(self._shutdown())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            try:
                await _finish_cleanup(self._close_task)
            except Exception:
                _LOGGER.exception("Engine shutdown failed during cancellation")
            raise

    async def __aenter__(self) -> AsyncEngine:
        self._check_loop()
        if self._closing:
            raise RuntimeError("Engine is closing")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await self.aclose()
        except Exception:
            if exc_type is None:
                raise
            _LOGGER.exception("Engine shutdown failed while propagating an exception")
