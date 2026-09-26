"""
Zephyr native bindings
"""

from __future__ import annotations
import collections.abc
import os
import typing

__all__: list[str] = [
    "ChoiceOutput",
    "EmbeddingRequest",
    "Engine",
    "EngineInfo",
    "EngineOptions",
    "FinishReason",
    "GenerationRequest",
    "KvCacheOptions",
    "LengthBucketSchedulerConfig",
    "LogLevel",
    "ModelTask",
    "OverloadedError",
    "PagedSchedulerConfig",
    "RequestOutput",
    "RequestStatus",
    "SamplingLogprobs",
    "SamplingParams",
    "SamplingResult",
    "TokenLogprob",
    "Usage",
    "VectorAdder",
    "set_log_level",
]

class ChoiceOutput:
    """
    Token increments for one choice, including any terminating EOS or stop token.
    """
    @property
    def finish_reason(self) -> FinishReason | None: ...
    @property
    def index(self) -> int: ...
    @property
    def tokens(self) -> list[SamplingResult]: ...

class EmbeddingRequest:
    def __init__(
        self,
        token_ids: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex],
    ) -> None: ...
    @property
    def token_ids(self) -> list[int]: ...

class Engine:
    """
    Native engine owning its runtime and worker threads. Exactly one caller consumes outputs.
    """
    def __enter__(self) -> Engine: ...
    def __exit__(
        self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any
    ) -> bool: ...
    def __init__(
        self,
        options: EngineOptions,
        *,
        task: ModelTask | None = None,
        kv_cache: KvCacheOptions | None = ...,
    ) -> None:
        """
        Resolve a supported model and its task from the checkpoint; task=None selects automatically. kv_cache applies to generation models only; None disables KV storage.
        """
    def cancel(self, request_id: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Cancel at the next safe execution boundary. Unknown or finished IDs are harmless.
        """
    def close(self) -> None:
        """
        Stop admission, finish in-flight work, cancel remaining requests, and release GPU resources. Safe to repeat.
        """
    def stop_choice(
        self,
        request_id: typing.SupportsInt | typing.SupportsIndex,
        choice_index: typing.SupportsInt | typing.SupportsIndex,
    ) -> None:
        """
        Normally finish one choice after a text stop, leaving other choices running. Unknown or finished choices are harmless; already published outputs remain unchanged.
        """
    def submit(self, request: GenerationRequest | EmbeddingRequest) -> int:
        """
        Validate and enqueue a request, returning its ID. Admission overload raises OverloadedError.
        """
    def wait_for_outputs(self) -> list[RequestOutput]:
        """
        Block for CPU-owned increments or final results. Single consumer only; an empty list means shutdown. Engine failures raise after queued outputs have been drained.
        """
    @property
    def info(self) -> EngineInfo: ...

class EngineInfo:
    """
    Effective capabilities of the loaded model.
    """
    @property
    def max_seq_len(self) -> int: ...
    @property
    def task(self) -> ModelTask: ...
    @property
    def vocab_size(self) -> int: ...

class EngineOptions:
    """
    Devices define tensor-parallel rank order; max_seq_len=0 inherits the model context limit.
    Constructing Engine profiles the selected GPUs.
    """
    def __init__(
        self,
        model_dir: os.PathLike | str | bytes,
        *,
        devices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex] = [
            0
        ],
        dtype: str = "bfloat16",
        max_seq_len: typing.SupportsInt | typing.SupportsIndex = 0,
        gpu_memory_utilization: typing.SupportsFloat | typing.SupportsIndex = 0.9,
        scheduler: LengthBucketSchedulerConfig | PagedSchedulerConfig = ...,
        max_outstanding_sequences: typing.SupportsInt | typing.SupportsIndex = 256,
        max_buffered_output_bytes: typing.SupportsInt | typing.SupportsIndex = 16777216,
        eos_token_ids: collections.abc.Sequence[
            typing.SupportsInt | typing.SupportsIndex
        ] = [],
    ) -> None: ...
    @property
    def devices(self) -> list[int]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def eos_token_ids(self) -> list[int]: ...
    @property
    def gpu_memory_utilization(self) -> float: ...
    @property
    def max_buffered_output_bytes(self) -> int: ...
    @property
    def max_outstanding_sequences(self) -> int: ...
    @property
    def max_seq_len(self) -> int: ...
    @property
    def model_dir(self) -> str: ...
    @property
    def scheduler(self) -> LengthBucketSchedulerConfig | PagedSchedulerConfig: ...

class FinishReason:
    """
    Members:

      EOS

      STOP_TOKEN

      STOP_STRING

      LENGTH

      MODEL_LENGTH

      CANCELED

      ERROR
    """

    CANCELED: typing.ClassVar[FinishReason]  # value = <FinishReason.CANCELED: 5>
    EOS: typing.ClassVar[FinishReason]  # value = <FinishReason.EOS: 0>
    ERROR: typing.ClassVar[FinishReason]  # value = <FinishReason.ERROR: 6>
    LENGTH: typing.ClassVar[FinishReason]  # value = <FinishReason.LENGTH: 3>
    MODEL_LENGTH: typing.ClassVar[
        FinishReason
    ]  # value = <FinishReason.MODEL_LENGTH: 4>
    STOP_STRING: typing.ClassVar[FinishReason]  # value = <FinishReason.STOP_STRING: 2>
    STOP_TOKEN: typing.ClassVar[FinishReason]  # value = <FinishReason.STOP_TOKEN: 1>
    __members__: typing.ClassVar[
        dict[str, FinishReason]
    ]  # value = {'EOS': <FinishReason.EOS: 0>, 'STOP_TOKEN': <FinishReason.STOP_TOKEN: 1>, 'STOP_STRING': <FinishReason.STOP_STRING: 2>, 'LENGTH': <FinishReason.LENGTH: 3>, 'MODEL_LENGTH': <FinishReason.MODEL_LENGTH: 4>, 'CANCELED': <FinishReason.CANCELED: 5>, 'ERROR': <FinishReason.ERROR: 6>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(
        self, state: typing.SupportsInt | typing.SupportsIndex
    ) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class GenerationRequest:
    """
    Tokenized generation input; max_new_tokens=None permits generation until a stop condition or the context limit.
    """
    def __init__(
        self,
        token_ids: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex],
        *,
        sampling: SamplingParams = ...,
        max_new_tokens: typing.SupportsInt | typing.SupportsIndex | None = None,
        stop_token_ids: collections.abc.Sequence[
            typing.SupportsInt | typing.SupportsIndex
        ] = [],
        seed: typing.SupportsInt | typing.SupportsIndex | None = None,
        num_choices: typing.SupportsInt | typing.SupportsIndex = 1,
        ignore_eos: bool = False,
    ) -> None: ...
    @property
    def ignore_eos(self) -> bool: ...
    @property
    def max_new_tokens(self) -> int | None: ...
    @property
    def num_choices(self) -> int: ...
    @property
    def sampling(self) -> SamplingParams: ...
    @property
    def seed(self) -> int | None: ...
    @property
    def stop_token_ids(self) -> list[int]: ...
    @property
    def token_ids(self) -> list[int]: ...

class KvCacheOptions:
    """
    KV cache settings; memory_bytes=None automatically estimates the per-device budget.
    """
    def __init__(
        self,
        *,
        block_size: typing.SupportsInt | typing.SupportsIndex = 16,
        memory_bytes: typing.SupportsInt | typing.SupportsIndex | None = None,
    ) -> None: ...
    @property
    def block_size(self) -> int: ...
    @property
    def memory_bytes(self) -> int | None: ...

class LengthBucketSchedulerConfig:
    def __init__(
        self, *, max_num_seqs: typing.SupportsInt | typing.SupportsIndex = 8
    ) -> None: ...
    @property
    def max_num_seqs(self) -> int: ...

class LogLevel:
    """
    Members:

      TRACE

      DEBUG

      INFO

      WARN

      ERROR

      OFF
    """

    DEBUG: typing.ClassVar[LogLevel]  # value = <LogLevel.DEBUG: 1>
    ERROR: typing.ClassVar[LogLevel]  # value = <LogLevel.ERROR: 4>
    INFO: typing.ClassVar[LogLevel]  # value = <LogLevel.INFO: 2>
    OFF: typing.ClassVar[LogLevel]  # value = <LogLevel.OFF: 5>
    TRACE: typing.ClassVar[LogLevel]  # value = <LogLevel.TRACE: 0>
    WARN: typing.ClassVar[LogLevel]  # value = <LogLevel.WARN: 3>
    __members__: typing.ClassVar[
        dict[str, LogLevel]
    ]  # value = {'TRACE': <LogLevel.TRACE: 0>, 'DEBUG': <LogLevel.DEBUG: 1>, 'INFO': <LogLevel.INFO: 2>, 'WARN': <LogLevel.WARN: 3>, 'ERROR': <LogLevel.ERROR: 4>, 'OFF': <LogLevel.OFF: 5>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(
        self, state: typing.SupportsInt | typing.SupportsIndex
    ) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ModelTask:
    """
    Members:

      GENERATION

      EMBEDDING
    """

    EMBEDDING: typing.ClassVar[ModelTask]  # value = <ModelTask.EMBEDDING: 1>
    GENERATION: typing.ClassVar[ModelTask]  # value = <ModelTask.GENERATION: 0>
    __members__: typing.ClassVar[
        dict[str, ModelTask]
    ]  # value = {'GENERATION': <ModelTask.GENERATION: 0>, 'EMBEDDING': <ModelTask.EMBEDDING: 1>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(
        self, state: typing.SupportsInt | typing.SupportsIndex
    ) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class OverloadedError(RuntimeError):
    pass

class PagedSchedulerConfig:
    def __init__(
        self,
        *,
        max_num_seqs: typing.SupportsInt | typing.SupportsIndex = 8,
        max_num_batched_tokens: typing.SupportsInt | typing.SupportsIndex = 512,
        max_prefill_chunk_tokens: typing.SupportsInt | typing.SupportsIndex = 512,
        max_decode_steps_before_prefill: typing.SupportsInt | typing.SupportsIndex = 8,
    ) -> None: ...
    @property
    def max_decode_steps_before_prefill(self) -> int: ...
    @property
    def max_num_batched_tokens(self) -> int: ...
    @property
    def max_num_seqs(self) -> int: ...
    @property
    def max_prefill_chunk_tokens(self) -> int: ...

class RequestOutput:
    """
    CPU-owned generation increments or a complete embedding; status is present only on the final request output.
    """
    @property
    def error_message(self) -> str: ...
    @property
    def request_id(self) -> int: ...
    @property
    def result(self) -> list[ChoiceOutput] | list[float]: ...
    @property
    def status(self) -> RequestStatus | None: ...
    @property
    def usage(self) -> Usage: ...

class RequestStatus:
    """
    Members:

      COMPLETED

      CANCELED

      REJECTED

      ERROR
    """

    CANCELED: typing.ClassVar[RequestStatus]  # value = <RequestStatus.CANCELED: 1>
    COMPLETED: typing.ClassVar[RequestStatus]  # value = <RequestStatus.COMPLETED: 0>
    ERROR: typing.ClassVar[RequestStatus]  # value = <RequestStatus.ERROR: 3>
    REJECTED: typing.ClassVar[RequestStatus]  # value = <RequestStatus.REJECTED: 2>
    __members__: typing.ClassVar[
        dict[str, RequestStatus]
    ]  # value = {'COMPLETED': <RequestStatus.COMPLETED: 0>, 'CANCELED': <RequestStatus.CANCELED: 1>, 'REJECTED': <RequestStatus.REJECTED: 2>, 'ERROR': <RequestStatus.ERROR: 3>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(
        self, state: typing.SupportsInt | typing.SupportsIndex
    ) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SamplingLogprobs:
    @property
    def logprob(self) -> float: ...
    @property
    def top_logprobs(self) -> list[TokenLogprob]: ...

class SamplingParams:
    """
    Effective sampling settings; logprobs are measured after penalties and temperature, before filtering.
    """
    def __init__(
        self,
        *,
        temperature: typing.SupportsFloat | typing.SupportsIndex = 1.0,
        top_k: typing.SupportsInt | typing.SupportsIndex = -1,
        top_p: typing.SupportsFloat | typing.SupportsIndex = 1.0,
        min_p: typing.SupportsFloat | typing.SupportsIndex = 0.0,
        frequency_penalty: typing.SupportsFloat | typing.SupportsIndex = 0.0,
        presence_penalty: typing.SupportsFloat | typing.SupportsIndex = 0.0,
        repetition_penalty: typing.SupportsFloat | typing.SupportsIndex = 1.0,
        logits_bias: collections.abc.Mapping[
            typing.SupportsInt | typing.SupportsIndex,
            typing.SupportsFloat | typing.SupportsIndex,
        ] = {},
        top_logprobs: typing.SupportsInt | typing.SupportsIndex | None = None,
    ) -> None: ...
    @property
    def frequency_penalty(self) -> float: ...
    @property
    def logits_bias(self) -> dict[int, float]: ...
    @property
    def min_p(self) -> float: ...
    @property
    def presence_penalty(self) -> float: ...
    @property
    def repetition_penalty(self) -> float: ...
    @property
    def temperature(self) -> float: ...
    @property
    def top_k(self) -> int: ...
    @property
    def top_logprobs(self) -> int | None: ...
    @property
    def top_p(self) -> float: ...

class SamplingResult:
    @property
    def logprobs(self) -> SamplingLogprobs | None: ...
    @property
    def token_id(self) -> int: ...

class TokenLogprob:
    @property
    def logprob(self) -> float: ...
    @property
    def token_id(self) -> int: ...

class Usage:
    @property
    def completion_tokens(self) -> int: ...
    @property
    def prompt_tokens(self) -> int: ...

class VectorAdder:
    def __init__(self) -> None: ...
    def add(
        self,
        left: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
        right: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    ) -> list[float]: ...

def set_log_level(level: LogLevel) -> None:
    """
    Set the process-wide logging threshold for the native engine.
    """
