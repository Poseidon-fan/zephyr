"""OpenAI error envelopes for HTTP responses and streaming events."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from ..http import APIError, as_api_error


def error_body(error: APIError) -> dict:
    kind = {
        401: "authentication_error",
        404: "not_found_error",
        429: "rate_limit_error",
    }.get(
        error.status, "server_error" if error.status >= 500 else "invalid_request_error"
    )
    return {
        "error": {
            "message": str(error),
            "type": kind,
            "param": error.param,
            "code": error.status,
        }
    }


def error_response(error: APIError) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else None
    return JSONResponse(error_body(error), status_code=error.status, headers=headers)


async def exception_response(request: Request, error: Exception) -> JSONResponse:
    if isinstance(error, RequestValidationError):
        first = error.errors()[0]
        param = ".".join(str(part) for part in first["loc"] if part != "body")
        message = first["msg"]
        if first["type"] == "extra_forbidden":
            message = "Unsupported parameter"
        api_error = APIError(f"{param}: {message}" if param else message, param=param)
    elif isinstance(error, HTTPException):
        api_error = APIError(str(error.detail), error.status_code)
    else:
        engine = getattr(request.app.state, "engine", None)
        api_error = as_api_error(
            error, running=engine is not None and engine.is_running
        )
    return error_response(api_error)
