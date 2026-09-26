"""Supported OpenAI request fields and strict request validation."""

from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

TokenId = Annotated[int, Field(ge=0)]
Prompt = str | list[TokenId]
BatchInput = str | list[TokenId] | list[str] | list[list[TokenId]]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class StreamOptions(RequestModel):
    include_usage: bool = False


class GenerationRequest(RequestModel):
    model: str = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    n: int = Field(default=1, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0, le=1)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    repetition_penalty: float | None = Field(default=None, gt=0)
    logit_bias: dict[str, float] | None = None
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    stop: str | list[str] | None = None
    stop_token_ids: list[TokenId] = Field(default_factory=list)
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False
    skip_special_tokens: bool = True
    stream: bool = False
    stream_options: StreamOptions | None = None
    user: str | None = None

    @model_validator(mode="after")
    def validate_generation(self):
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        stops = [self.stop] if isinstance(self.stop, str) else self.stop or []
        if any(not stop for stop in stops):
            raise ValueError("stop strings must not be empty")
        if len(stops) > 4:
            raise ValueError("At most four stop strings are supported")
        if self.logit_bias is not None:
            for token, bias in self.logit_bias.items():
                if not token.isascii() or not token.isdecimal():
                    raise ValueError("logit_bias keys must be nonnegative token IDs")
                if not -100 <= bias <= 100:
                    raise ValueError("logit_bias values must be between -100 and 100")
        return self


class TextPart(RequestModel):
    type: Literal["text"]
    text: str


class ChatMessage(RequestModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart]
    name: str | None = None

    def as_message(self) -> dict[str, str]:
        content = self.content
        if not isinstance(content, str):
            content = "".join(part.text for part in content)
        message = {"role": self.role, "content": content}
        if self.name is not None:
            message["name"] = self.name
        return message


class ChatCompletionRequest(GenerationRequest):
    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    chat_template_kwargs: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_chat(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("Specify only one of max_tokens and max_completion_tokens")
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs=true")
        if self.add_generation_prompt and self.continue_final_message:
            raise ValueError(
                "continue_final_message and add_generation_prompt are mutually exclusive"
            )
        reserved = {
            "add_generation_prompt",
            "continue_final_message",
        }
        conflicts = reserved.intersection(self.chat_template_kwargs or {})
        if conflicts:
            raise ValueError(
                "chat_template_kwargs cannot override request fields: "
                + ", ".join(sorted(conflicts))
            )
        return self


class CompletionRequest(GenerationRequest):
    prompt: BatchInput
    max_tokens: int | None = Field(default=16, ge=1)
    logprobs: int | None = Field(default=None, ge=0, le=20)


class EmbeddingRequest(RequestModel):
    model: str = Field(min_length=1)
    input: BatchInput
    encoding_format: Literal["float", "base64"] = "float"
    user: str | None = None


def split_inputs(value: BatchInput) -> list[Prompt]:
    if isinstance(value, str):
        return [value]
    if not value:
        raise ValueError("Input must not be empty")
    if isinstance(value[0], int):
        return [cast(list[int], value)]
    batch = cast(list[str] | list[list[int]], value)
    if any(isinstance(prompt, list) and not prompt for prompt in batch):
        raise ValueError("Token prompts must not be empty")
    return list(batch)
