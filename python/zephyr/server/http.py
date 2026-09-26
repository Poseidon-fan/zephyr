"""HTTP limits, errors, and cancellation shared by protocol routers."""

import asyncio
import hmac
import logging
from collections.abc import Awaitable, Callable, Collection
from typing import TypeVar

from fastapi import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


class APIError(Exception):
    def __init__(
        self, message: str, status: int = 400, *, param: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.param = param


def as_api_error(error: Exception, *, running: bool = True) -> APIError:
    if isinstance(error, APIError):
        return error
    if isinstance(error, ValueError):
        return APIError(str(error))
    from ..engine import OverloadedError

    if isinstance(error, OverloadedError):
        return APIError(str(error), 429)
    if not running:
        return APIError("The inference engine is unavailable", 503)
    _LOGGER.error("Inference request failed", exc_info=error)
    return APIError("Inference request failed", 500)


class RequestLimitsMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int,
        api_key: str | None,
        error_response: Callable[[APIError], Response],
        public_paths: Collection[str] = (),
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.api_key = api_key
        self.error_response = error_response
        self.public_paths = frozenset(public_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        if self.api_key is not None and scope["path"] not in self.public_paths:
            supplied = headers.get(b"authorization", b"").split()
            valid = (
                len(supplied) == 2
                and supplied[0].lower() == b"bearer"
                and hmac.compare_digest(supplied[1], self.api_key.encode())
            )
            if not valid:
                response = self.error_response(APIError("Invalid API key", 401))
                await response(scope, receive, send)
                return
        length = headers.get(b"content-length")
        try:
            oversized = length is not None and int(length) > self.max_bytes
        except ValueError:
            response = self.error_response(APIError("Invalid Content-Length", 400))
            await response(scope, receive, send)
            return
        body = bytearray()
        while not oversized:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            oversized = len(body) + len(chunk) > self.max_bytes
            if oversized:
                break
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        if oversized:
            response = self.error_response(APIError("Request body is too large", 413))
            await response(scope, receive, send)
            return
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


async def run_until_disconnect(request: Request, operation: Awaitable[_T]) -> _T:
    async def disconnected() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    work = asyncio.ensure_future(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait(
            (work, watcher), return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return work.result()
        raise APIError("Client disconnected", 499)
    finally:
        work.cancel()
        watcher.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)


async def gather_requests(operations: list[Awaitable[_T]]) -> list[_T]:
    tasks = [asyncio.ensure_future(operation) for operation in operations]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
