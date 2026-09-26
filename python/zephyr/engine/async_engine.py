"""Async request delivery over the native engine's single output stream."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from dataclasses import dataclass, field
from functools import partial
from types import TracebackType

from .. import _C  # pyright: ignore[reportMissingModuleSource]
from .outputs import OutputBuffer, RequestOutput

_LOGGER = logging.getLogger(__name__)


class OutputBufferError(RuntimeError):
    """A request's unread output exhausted the frontend's payload budget."""


@dataclass(slots=True)
class _RequestState:
    output: OutputBuffer
    num_choices: int
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
    """Token-based inference bound to one asyncio event loop.

    Construct with ``await AsyncEngine.create(...)`` and close with ``aclose``
    or an async context manager. Requests and sampling settings are native value
    snapshots; tokenizer and protocol processing belong to their callers.

    A single receiver owns native output consumption. The native admission and
    payload limits also bound frontend unread requests and output independently;
    one received batch and Python object overhead are outside the payload budget.
    On overflow, the largest unread request is canceled with OutputBufferError.
    """

    def __init__(
        self,
        native: _C.Engine,
        options: _C.EngineOptions,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._native = native
        self._loop = asyncio.get_running_loop()
        self._executor = executor
        self._requests: dict[int, _RequestState] = {}
        self._num_choices = 0
        self._buffered_bytes = 0
        self._max_choices = options.max_outstanding_sequences
        self._max_bytes = options.max_buffered_output_bytes
        self._closing = False
        self._failure: Exception | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._receiver = self._loop.create_task(self._receive())

    @classmethod
    async def create(
        cls,
        options: _C.EngineOptions,
        *,
        task: _C.ModelTask | None = None,
        kv_cache: _C.KvCacheOptions | None = _C.KvCacheOptions(),
    ) -> AsyncEngine:
        loop = asyncio.get_running_loop()
        # Closing must be able to run while the receiver blocks in native code.
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="zephyr")
        loading = loop.run_in_executor(
            executor, partial(_C.Engine, options, task=task, kv_cache=kv_cache)
        )
        try:
            native = await asyncio.shield(loading)
            return cls(native, options, executor)
        except BaseException:

            async def cleanup() -> None:
                try:
                    loaded = await loading
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
        self._check_loop()
        if self._failure is not None:
            raise self._failure
        if self._closing:
            raise RuntimeError("Engine is closing")
        generation = isinstance(request, _C.GenerationRequest)
        choices = (
            request.num_choices if isinstance(request, _C.GenerationRequest) else 1
        )
        # Invalid choice counts still go through native argument validation.
        if (
            0 < choices <= self._max_choices
            and choices > self._max_choices - self._num_choices
        ):
            raise _C.OverloadedError(
                "Consume outstanding outputs before submitting more"
            )

        request_id = self._native.submit(request)
        state = _RequestState(OutputBuffer(request_id, generation), choices)
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
                    state.output.add(output)
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
