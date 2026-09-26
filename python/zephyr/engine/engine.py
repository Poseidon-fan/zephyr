from __future__ import annotations

import logging
import threading
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import cast

from .. import _C  # pyright: ignore[reportMissingModuleSource]
from .outputs import OutputBuffer, RequestOutput

Request = _C.GenerationRequest | _C.EmbeddingRequest

_LOGGER = logging.getLogger(__name__)


class Engine:
    """Synchronous native engine with one active execute or stream consumer."""

    def __init__(
        self,
        options: _C.EngineOptions,
        *,
        task: _C.ModelTask | None = None,
        kv_cache: _C.KvCacheOptions | None = _C.KvCacheOptions(),
    ) -> None:
        self._native = _C.Engine(options, task=task, kv_cache=kv_cache)
        self._consumer_lock = threading.Lock()
        self._closed = False
        self._failed = False

    def execute(self, requests: Iterable[Request]) -> list[RequestOutput]:
        """Submit within native admission capacity and return results in input order."""
        with self._operation() as pending:
            iterator = iter(requests)
            buffers: dict[int, tuple[int, OutputBuffer]] = {}
            results: list[RequestOutput | None] = []
            exhausted = False
            waiting: Request | None = None
            while not exhausted or pending:
                while not exhausted:
                    if waiting is None:
                        try:
                            waiting = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                    try:
                        request_id = self._native.submit(waiting)
                    except _C.OverloadedError:
                        if not pending:
                            raise
                        break
                    pending.add(request_id)
                    buffers[request_id] = (
                        len(results),
                        OutputBuffer(
                            request_id, isinstance(waiting, _C.GenerationRequest)
                        ),
                    )
                    results.append(None)
                    waiting = None
                if not pending:
                    continue
                for output in self._wait(pending):
                    if output.request_id not in buffers:
                        self._failed = True
                        raise RuntimeError(
                            "Native engine returned an unknown request ID"
                        )
                    index, buffer = buffers[output.request_id]
                    buffer.add(output)
                    if buffer.finished:
                        results[index] = buffer.take()
                        del buffers[output.request_id]
            return cast(list[RequestOutput], results)

    def stream(self, request: Request) -> Generator[RequestOutput, None, None]:
        """Yield token increments or one embedding result; close early to cancel."""
        with self._operation() as pending:
            request_id = self._native.submit(request)
            pending.add(request_id)
            buffer = OutputBuffer(request_id, isinstance(request, _C.GenerationRequest))
            while pending:
                for output in self._wait(pending):
                    if output.request_id != request_id:
                        self._failed = True
                        raise RuntimeError(
                            "Native engine returned an unknown request ID"
                        )
                    buffer.add(output)
                if not buffer.empty:
                    yield buffer.take()

    def close(self) -> None:
        """Stop the native engine and wake any consumer waiting in another thread."""
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
        # Native admission credits are already released for every terminal output in this batch.
        # Record all of them before Python result processing can fail and enter cancellation cleanup.
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
