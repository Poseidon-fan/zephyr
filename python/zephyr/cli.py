"""Command-line entry point, without importing inference dependencies for help."""

import argparse
import os
from copy import deepcopy
from pathlib import Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use comma-separated CUDA device ordinals"
        ) from error
    if (
        not devices
        or any(device < 0 for device in devices)
        or len(set(devices)) != len(devices)
    ):
        raise argparse.ArgumentTypeError(
            "devices must be distinct nonnegative ordinals"
        )
    return devices


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zephyr", description="Zephyr CUDA inference engine"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve a model with the OpenAI HTTP API")
    serve.add_argument("model", help="Local model directory or Hugging Face repository")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_positive_int, default=8000)
    serve.add_argument("--served-model-name")
    devices = serve.add_mutually_exclusive_group()
    devices.add_argument("--tensor-parallel-size", type=_positive_int)
    devices.add_argument(
        "--devices", type=_devices, help="CUDA ordinals in tensor-parallel rank order"
    )
    serve.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    serve.add_argument("--max-model-len", type=_positive_int)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    serve.add_argument("--max-num-seqs", type=_positive_int)
    serve.add_argument("--max-num-batched-tokens", type=_positive_int)
    serve.add_argument("--max-prefill-chunk-tokens", type=_positive_int)
    serve.add_argument("--max-outstanding-sequences", type=_positive_int, default=256)
    serve.add_argument(
        "--max-buffered-output-bytes", type=_positive_int, default=16 * 1024 * 1024
    )
    serve.add_argument("--revision")
    serve.add_argument("--tokenizer")
    serve.add_argument("--tokenizer-revision")
    serve.add_argument("--trust-remote-code", action="store_true")
    serve.add_argument("--chat-template", type=Path, help="UTF-8 Jinja template file")
    serve.add_argument(
        "--api-key",
        default=os.environ.get("ZEPHYR_API_KEY"),
        help="Bearer key; defaults to ZEPHYR_API_KEY",
    )
    serve.add_argument("--task", choices=("auto", "generate", "embed"), default="auto")
    serve.add_argument(
        "--max-request-bytes", type=_positive_int, default=8 * 1024 * 1024
    )
    serve.add_argument(
        "--max-batch-size",
        type=_positive_int,
        default=32,
        help="Maximum total sequences per HTTP request",
    )
    serve.add_argument(
        "--send-timeout", type=float, default=30.0, help="SSE send timeout in seconds"
    )
    serve.add_argument("--shutdown-timeout", type=_positive_int, default=30)
    serve.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.port > 65535:
        parser.error("--port must be at most 65535")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.task == "embed" and (
        args.max_num_batched_tokens is not None
        or args.max_prefill_chunk_tokens is not None
    ):
        parser.error(
            "Embedding tasks do not support generation scheduler token budgets"
        )
    try:
        import uvicorn
        from uvicorn.config import LOG_LEVELS, LOGGING_CONFIG

        from . import _C  # pyright: ignore[reportMissingModuleSource]
        from .engine import LengthBucketSchedulerConfig, ModelTask, PagedSchedulerConfig
        from .engine.config import EngineConfig
        from .server import ServerConfig, create_app
    except ImportError as error:
        parser.error(
            f"Serving dependencies are unavailable; install zephyr[server]: {error}"
        )

    scheduler_args = {
        key: value
        for key, value in (
            ("max_num_seqs", args.max_num_seqs),
            ("max_num_batched_tokens", args.max_num_batched_tokens),
            ("max_prefill_chunk_tokens", args.max_prefill_chunk_tokens),
        )
        if value is not None
    }
    scheduler = None
    if scheduler_args:
        if (
            args.max_num_batched_tokens is not None
            and args.max_prefill_chunk_tokens is None
        ):
            scheduler_args["max_prefill_chunk_tokens"] = min(
                args.max_num_batched_tokens,
                PagedSchedulerConfig().max_prefill_chunk_tokens,
            )
        scheduler = (
            LengthBucketSchedulerConfig(**scheduler_args)
            if args.task == "embed"
            else PagedSchedulerConfig(**scheduler_args)
        )
    try:
        chat_template = (
            args.chat_template.read_text(encoding="utf-8")
            if args.chat_template
            else None
        )
        engine_config = EngineConfig(
            args.model,
            devices=args.devices or tuple(range(args.tensor_parallel_size or 1)),
            dtype=args.dtype,
            max_seq_len=args.max_model_len or 0,
            gpu_memory_utilization=args.gpu_memory_utilization,
            scheduler=scheduler,
            max_outstanding_sequences=args.max_outstanding_sequences,
            max_buffered_output_bytes=args.max_buffered_output_bytes,
            revision=args.revision,
            tokenizer=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
            trust_remote_code=args.trust_remote_code,
            chat_template=chat_template,
            task={
                "auto": None,
                "generate": ModelTask.GENERATION,
                "embed": ModelTask.EMBEDDING,
            }[args.task],
        )
        config = ServerConfig(
            served_model_name=args.served_model_name,
            api_key=args.api_key,
            max_request_bytes=args.max_request_bytes,
            max_batch_size=args.max_batch_size,
            send_timeout=args.send_timeout,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    log_config = deepcopy(LOGGING_CONFIG)
    log_config["loggers"]["zephyr"] = {
        "handlers": ["default"],
        "level": LOG_LEVELS[args.log_level],
        "propagate": False,
    }
    _C.set_log_level(
        {
            "critical": _C.LogLevel.ERROR,
            "error": _C.LogLevel.ERROR,
            "warning": _C.LogLevel.WARN,
            "info": _C.LogLevel.INFO,
            "debug": _C.LogLevel.DEBUG,
            "trace": _C.LogLevel.TRACE,
        }[args.log_level]
    )
    uvicorn.run(
        create_app(engine_config, config),
        host=args.host,
        port=args.port,
        workers=1,
        log_config=log_config,
        log_level=args.log_level,
        timeout_graceful_shutdown=args.shutdown_timeout,
    )
