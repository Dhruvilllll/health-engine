"""
===================================================
Health Engine - Error Handling Middleware
===================================================

Custom exception handling for API.
"""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from typing import Union


logger = logging.getLogger(__name__)


class HealthEngineException(Exception):
    """Base exception for Health Engine."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = None
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail or message
        super().__init__(self.message)


class ValidationException(HealthEngineException):
    """Validation error."""

    def __init__(self, message: str, detail: str = None):
        super().__init__(
            message,
            status.HTTP_400_BAD_REQUEST,
            detail
        )


class ModelException(HealthEngineException):
    """Model loading/prediction error."""

    def __init__(self, message: str, detail: str = None):
        super().__init__(
            message,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail
        )


class NotFoundExcept (HealthEngineException):
    """Resource not found."""

    def __init__(self, message: str, detail: str = None):
        super().__init__(
            message,
            status.HTTP_404_NOT_FOUND,
            detail
        )


async def health_engine_exception_handler(
    request: Request,
    exc: HealthEngineException
) -> JSONResponse:
    """Handle HealthEngineException."""
    logger.error(f"{exc.message}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )


async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "status_code": 500
        }
    )
