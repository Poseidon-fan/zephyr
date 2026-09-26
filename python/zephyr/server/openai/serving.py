"""OpenAI wire formats over the shared asynchronous engine."""

import asyncio
import base64
import json
import struct
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, aclosing
from typing import TYPE_CHECKING, cast

import anyio
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from ..http import (
    APIError,
    as_api_error,
    gather_requests,
    run_until_disconnect,
)
from .errors import error_body
from .protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    split_inputs,
)

if TYPE_CHECKING:
    from ...engine.outputs import GenerationOutput

router = APIRouter(prefix="/v1")
GenerationRequest = ChatCompletionRequest | CompletionRequest


def _engine(request: Request, model: str | None = None, task: str | None = None):
    engine = request.app.state.engine
    if not engine.is_running:
        raise APIError("The inference engine is unavailable", 503)
    if model is not None and model != request.app.state.model_name:
        raise APIError(f"Model {model!r} is not served", 404, param="model")
    if task is not None and engine.info.task.name != task:
        raise APIError(f"The loaded model does not support {task.lower()} requests")
    return engine


def _check_batch(request: Request, count: int, n: int = 1) -> None:
    limit = request.app.state.config.max_batch_size
    if count * n > limit:
        raise APIError(f"A request may contain at most {limit} total sequences")


def _params(body: GenerationRequest):
    from ...engine.inputs import GenerationParams

    max_tokens = body.max_tokens
    if isinstance(body, ChatCompletionRequest):
        top_logprobs = (body.top_logprobs or 0) if body.logprobs else None
        if body.max_completion_tokens is not None:
            max_tokens = body.max_completion_tokens
    else:
        top_logprobs = body.logprobs
    bias = None
    if body.logit_bias is not None:
        bias = {int(token): value for token, value in body.logit_bias.items()}
    return GenerationParams(
        max_tokens=max_tokens,
        n=body.n,
        temperature=body.temperature,
        top_k=body.top_k,
        top_p=body.top_p,
        min_p=body.min_p,
        frequency_penalty=body.frequency_penalty,
        presence_penalty=body.presence_penalty,
        repetition_penalty=body.repetition_penalty,
        logit_bias=bias,
        top_logprobs=top_logprobs,
        seed=body.seed,
        stop=body.stop,
        stop_token_ids=body.stop_token_ids,
        ignore_eos=body.ignore_eos,
        include_stop_str_in_output=body.include_stop_str_in_output,
        skip_special_tokens=body.skip_special_tokens,
    )


def _check_output(output) -> None:
    status = output.status.name if output.status is not None else None
    if status == "REJECTED":
        raise APIError(output.error_message or "Request rejected", 400)
    if status == "ERROR":
        raise APIError(output.error_message or "Inference request failed", 500)
    if status == "CANCELED":
        raise APIError(output.error_message or "Inference request was canceled", 503)


def _finish_reason(reason) -> str | None:
    if reason is None:
        return None
    if reason.name in ("EOS", "STOP_TOKEN", "STOP_STRING"):
        return "stop"
    if reason.name in ("LENGTH", "MODEL_LENGTH"):
        return "length"
    raise APIError("Generation terminated without a successful completion", 500)


def _usage(outputs) -> dict[str, int]:
    prompt = sum(output.usage.prompt_tokens for output in outputs)
    completion = sum(output.usage.completion_tokens for output in outputs)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _token_logprob(tokenizer, token_id: int, logprob: float) -> dict:
    return {
        "token": tokenizer.token_text(token_id),
        "logprob": max(logprob, -9999.0),
        "bytes": tokenizer.token_bytes(token_id),
    }


def _logprobs(tokenizer, tokens, *, chat: bool, token_offsets=()):
    content = []
    completion = {
        "tokens": [],
        "token_logprobs": [],
        "top_logprobs": [],
        "text_offset": [],
    }
    for index, token in enumerate(tokens):
        values = token.logprobs
        if values is None:
            raise APIError("The engine did not return requested log probabilities", 500)
        sampled = _token_logprob(tokenizer, token.token_id, values.logprob)
        candidates = [
            _token_logprob(tokenizer, candidate.token_id, candidate.logprob)
            for candidate in values.top_logprobs
        ]
        if chat:
            sampled["top_logprobs"] = candidates
            content.append(sampled)
        else:
            top = {candidate["token"]: candidate["logprob"] for candidate in candidates}
            top[sampled["token"]] = sampled["logprob"]
            completion["tokens"].append(sampled["token"])
            completion["token_logprobs"].append(sampled["logprob"])
            completion["top_logprobs"].append(top)
            completion["text_offset"].append(token_offsets[index])
    return {"content": content} if chat else completion


def _choice(engine, choice, index: int, body: GenerationRequest, *, stream=False):
    chat = isinstance(body, ChatCompletionRequest)
    wants_logprobs = body.logprobs if chat else body.logprobs is not None
    result = {
        "index": index,
        "finish_reason": _finish_reason(choice.finish_reason),
        "logprobs": (
            _logprobs(
                engine.tokenizer,
                choice.tokens,
                chat=chat,
                token_offsets=() if chat else choice.token_offsets,
            )
            if wants_logprobs
            else None
        ),
    }
    if chat:
        result["delta" if stream else "message"] = (
            {"content": choice.text}
            if stream
            else {"role": "assistant", "content": choice.text}
        )
    else:
        result["text"] = choice.text
    return result


def _envelope(request: Request, *, chat: bool, stream: bool) -> dict:
    return {
        "id": ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex,
        "object": "chat.completion.chunk"
        if chat and stream
        else ("chat.completion" if chat else "text_completion"),
        "created": int(time.time()),
        "model": request.app.state.model_name,
    }


def _event(body: dict) -> dict[str, str]:
    return {
        "data": json.dumps(
            body, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    }


class _OwnedEventSourceResponse(EventSourceResponse):
    def __init__(
        self,
        content: AsyncGenerator[dict[str, str], None],
        stack: AsyncExitStack,
        **kwargs,
    ):
        super().__init__(content, **kwargs)
        self._stack = stack

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Covers a disconnect or failed header send before the iterator starts.
            async def cleanup():
                with anyio.CancelScope(shield=True):
                    try:
                        await cast(AsyncGenerator, self.body_iterator).aclose()
                    finally:
                        await self._stack.aclose()

            closing = asyncio.create_task(cleanup())
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    continue
            closing.result()


async def _prepare_streams(streams, stack: AsyncExitStack):
    async def first(stream):
        output = await anext(stream)
        _check_output(output)
        return output

    try:
        for stream in streams:
            await stack.enter_async_context(aclosing(stream))
        return await gather_requests([first(stream) for stream in streams])
    except BaseException:
        await stack.aclose()
        raise


async def _merge_streams(
    streams, first_outputs
) -> AsyncGenerator[tuple[int, "GenerationOutput"], None]:
    queue = asyncio.Queue(maxsize=max(1, len(streams) * 2))

    async def pump(index, stream):
        try:
            async for output in stream:
                await queue.put((index, output))
        except Exception as error:
            await queue.put((index, error))
        await queue.put((index, None))

    tasks = []
    try:
        for index, output in enumerate(first_outputs):
            yield index, output
        tasks = [
            asyncio.create_task(pump(index, stream))
            for index, stream in enumerate(streams)
        ]
        remaining = len(tasks)
        while remaining:
            index, item = await queue.get()
            if item is None:
                remaining -= 1
            elif isinstance(item, Exception):
                raise item
            else:
                yield index, item
    finally:
        with anyio.CancelScope(shield=True):
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def _stream_response(request: Request, body: GenerationRequest, engine, streams):
    stack = AsyncExitStack()
    try:
        first = await run_until_disconnect(request, _prepare_streams(streams, stack))
        envelope = _envelope(
            request, chat=isinstance(body, ChatCompletionRequest), stream=True
        )
        include_usage = (
            body.stream_options is not None and body.stream_options.include_usage
        )

        async def events():
            finished = {}
            started = set()
            try:
                async with aclosing(_merge_streams(streams, first)) as merged:
                    async for prompt_index, output in merged:
                        _check_output(output)
                        for choice in output.choices:
                            index = prompt_index * body.n + choice.index
                            if (
                                isinstance(body, ChatCompletionRequest)
                                and index not in started
                            ):
                                started.add(index)
                                initial = {
                                    **envelope,
                                    "choices": [
                                        {
                                            "index": index,
                                            "delta": {
                                                "role": "assistant",
                                                "content": "",
                                            },
                                            "logprobs": None,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                if include_usage:
                                    initial["usage"] = None
                                yield _event(initial)
                            converted = _choice(
                                engine, choice, index, body, stream=True
                            )
                            chunk = {**envelope, "choices": [converted]}
                            if include_usage:
                                chunk["usage"] = None
                            yield _event(chunk)
                        if output.status is not None:
                            finished[prompt_index] = output
                if len(finished) != len(streams):
                    raise APIError("The engine ended a stream without final usage", 500)
                if include_usage:
                    yield _event(
                        {**envelope, "choices": [], "usage": _usage(finished.values())}
                    )
            except Exception as error:
                api_error = as_api_error(error, running=engine.is_running)
                yield _event(error_body(api_error))
            finally:
                with anyio.CancelScope(shield=True):
                    await stack.aclose()
            yield {"data": "[DONE]"}

        return _OwnedEventSourceResponse(
            events(),
            stack,
            send_timeout=request.app.state.config.send_timeout,
            headers={"X-Accel-Buffering": "no"},
        )
    except BaseException:
        await stack.aclose()
        raise


@router.get("/models")
async def models(request: Request):
    _engine(request)
    return {
        "object": "list",
        "data": [
            {
                "id": request.app.state.model_name,
                "object": "model",
                "created": request.app.state.created,
                "owned_by": "zephyr",
            }
        ],
    }


@router.post("/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    engine = _engine(request, body.model, "GENERATION")
    _check_batch(request, 1, body.n)
    params = _params(body)
    messages = [message.as_message() for message in body.messages]
    kwargs = {
        "chat_template_kwargs": body.chat_template_kwargs,
        "add_generation_prompt": body.add_generation_prompt,
        "continue_final_message": body.continue_final_message,
    }
    if body.stream:
        return await _stream_response(
            request, body, engine, [engine.stream_chat(messages, params, **kwargs)]
        )
    output = await run_until_disconnect(
        request, engine.chat(messages, params, **kwargs)
    )
    _check_output(output)
    return {
        **_envelope(request, chat=True, stream=False),
        "choices": [
            _choice(engine, choice, choice.index, body) for choice in output.choices
        ],
        "usage": _usage([output]),
    }


@router.post("/completions")
async def completions(request: Request, body: CompletionRequest):
    engine = _engine(request, body.model, "GENERATION")
    prompts = split_inputs(body.prompt)
    _check_batch(request, len(prompts), body.n)
    params = _params(body)
    if body.stream:
        return await _stream_response(
            request,
            body,
            engine,
            [engine.stream_generate(prompt, params) for prompt in prompts],
        )
    outputs = await run_until_disconnect(
        request,
        gather_requests([engine.generate(prompt, params) for prompt in prompts]),
    )
    for output in outputs:
        _check_output(output)
    return {
        **_envelope(request, chat=False, stream=False),
        "choices": [
            _choice(engine, choice, prompt_index * body.n + choice.index, body)
            for prompt_index, output in enumerate(outputs)
            for choice in output.choices
        ],
        "usage": _usage(outputs),
    }


@router.post("/embeddings")
async def embeddings(request: Request, body: EmbeddingRequest):
    engine = _engine(request, body.model, "EMBEDDING")
    prompts = split_inputs(body.input)
    _check_batch(request, len(prompts))
    outputs = await run_until_disconnect(
        request, gather_requests([engine.embed(prompt) for prompt in prompts])
    )
    data = []
    for index, output in enumerate(outputs):
        _check_output(output)
        vector = output.result
        if body.encoding_format == "base64":
            vector = base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode(
                "ascii"
            )
        data.append({"object": "embedding", "embedding": vector, "index": index})
    usage = _usage(outputs)
    return {
        "object": "list",
        "data": data,
        "model": request.app.state.model_name,
        "usage": {
            "prompt_tokens": usage["prompt_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    }
