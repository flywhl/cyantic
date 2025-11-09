"""Hook system for Cyantic external build.

Hooks allow references in config like @import:, @env:, @value:, @asset:, @call:
to be resolved during the build process.
"""

import importlib
import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

HOOK_PREFIX = "@"


class HookRegistry:
    """Global registry for hooks."""

    _hooks: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, handler: Callable):
        """Register a hook handler function."""
        if name in cls._hooks:
            raise ValueError(f"Hook already registered: {name}")
        cls._hooks[name] = handler
        logger.debug(f"Registered hook: {name}")

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a hook handler by name."""
        if name not in cls._hooks:
            raise ValueError(f"Unknown hook: @{name}:")
        return cls._hooks[name]

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a hook is registered."""
        return name in cls._hooks

    @classmethod
    def clear(cls):
        """Clear all registered hooks."""
        cls._hooks.clear()


def hook(name: str):
    """Decorator to register a hook handler function.

    Example:
        @hook("myvalue")
        def my_custom_hook(path: str, built_assets: dict, root_data: dict, current_path: str = "") -> Any:
            return root_data.get(path)
    """

    def decorator(handler: Callable):
        HookRegistry.register(name, handler)
        return handler

    return decorator


def navigate_path(obj: Any, path: str) -> Any:
    """Navigate a path like 'a.b.c' using either dict keys or object attributes.

    Args:
        obj: The object to navigate
        path: Dot-separated path

    Returns:
        The value at the path

    Raises:
        ValueError: If path cannot be navigated
    """
    if not path:
        return obj

    current = obj
    for segment in path.split("."):
        try:
            # Try dict-style access first
            if hasattr(current, "__getitem__"):
                current = current[segment]
            else:
                # Fall back to attribute access
                current = getattr(current, segment)
        except (KeyError, AttributeError):
            # Try the other way
            try:
                if hasattr(current, "__getitem__"):
                    current = getattr(current, segment)
                else:
                    current = current[segment]
            except (KeyError, AttributeError, TypeError):
                raise ValueError(f"Cannot navigate to '{segment}' in path '{path}'")
    return current


# Built-in hooks


@hook("import")
def import_hook(
    path: str,
    built_assets: dict[str, Any],
    root_data: dict[str, Any],
    current_path: str = "",
) -> Any:
    """Handle @import:module.path.Class references.

    Example: @import:torch.optim.Adam
    """
    module_path, attr = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(f"Cannot import {path}") from e
    return getattr(module, attr)


@hook("env")
def env_hook(
    path: str,
    built_assets: dict[str, Any],
    root_data: dict[str, Any],
    current_path: str = "",
) -> str:
    """Handle @env:VARIABLE_NAME references.

    Example: @env:DATABASE_URL
    """
    try:
        return os.environ[path]
    except KeyError as e:
        raise ValueError(f"Environment variable {path} not found") from e


@hook("value")
def value_hook(
    path: str,
    built_assets: dict[str, Any],
    root_data: dict[str, Any],
    current_path: str = "",
) -> Any:
    """Handle @value:path.to.value - reference original config values.

    Example: @value:common.learning_rate
    """
    return navigate_path(root_data, path)


@hook("asset")
def asset_hook(
    path: str,
    built_assets: dict[str, Any],
    root_data: dict[str, Any],
    current_path: str = "",
) -> Any:
    """Handle @asset:path - reference built objects.

    Example: @asset:services.database
    """
    if path not in built_assets:
        raise ValueError(
            f"No asset found for @asset:{path}. "
            f"Available: {list(built_assets.keys())}"
        )
    return built_assets[path]


@hook("call")
def call_hook(
    path: str,
    built_assets: dict[str, Any],
    root_data: dict[str, Any],
    current_path: str = "",
) -> Any:
    """Handle @call:path.method - call methods on built objects.

    Example: @call:network.parameters
    """
    parts = path.split(".")

    # Try progressively longer prefixes to find the asset
    asset = None
    remaining_parts = []

    for i in range(1, len(parts) + 1):
        asset_path = ".".join(parts[:i])
        if asset_path in built_assets:
            asset = built_assets[asset_path]
            remaining_parts = parts[i:]
            break

    if asset is None:
        raise ValueError(
            f"No asset found for @call:{path}. "
            f"Available: {list(built_assets.keys())}"
        )

    # Navigate to the method
    current = asset
    for part in remaining_parts:
        current = getattr(current, part)

    # Call it
    if callable(current):
        try:
            return current()
        except Exception as e:
            raise ValueError(f"Error calling {path}: {str(e)}") from e
    else:
        raise ValueError(f"'{path}' is not callable")
