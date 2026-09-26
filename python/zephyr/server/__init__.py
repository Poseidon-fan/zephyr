"""HTTP serving; install zephyr[server] to use this package."""

from .app import ServerConfig, create_app

__all__ = ["ServerConfig", "create_app"]
