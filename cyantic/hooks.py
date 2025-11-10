"""Hook system for Cyantic external build.

Hooks allow references in config like @import:, @env:, @value:, @asset:, @call:, @include:
to be resolved during the build process.
"""

import importlib
import logging
import os
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

HOOK_PREFIX = "@"


class HookRegistry:
    """Global registry for hooks."""

    _hooks: dict[str, Callable] = {}
    _before_hooks: set[str] = set()  # Hooks that should be processed before discovery

    @classmethod
    def register(cls, name: str, handler: Callable, before: bool = False):
        """Register a hook handler function.

        Args:
            name: Hook name (without @ prefix)
            handler: Hook handler function
            before: If True, process this hook before discovery (e.g., @include, @value, @env)
                   If False, process after building (e.g., @asset, @call)
        """
        if name in cls._hooks:
            raise ValueError(f"Hook already registered: {name}")
        cls._hooks[name] = handler
        if before:
            cls._before_hooks.add(name)
        logger.debug(f"Registered hook: {name} (before={before})")

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
    def is_before_hook(cls, name: str) -> bool:
        """Check if a hook should be processed before discovery."""
        return name in cls._before_hooks

    @classmethod
    def clear(cls):
        """Clear all registered hooks."""
        cls._hooks.clear()
        cls._before_hooks.clear()


def hook(name: str, before: bool = False):
    """Decorator to register a hook handler function.

    Args:
        name: Hook name (without @ prefix)
        before: If True, this hook is processed before discovery (e.g., @include, @value, @env)
               If False, this hook is processed during/after building (e.g., @asset, @call)

    Example:
        @hook("include", before=True)
        def include_hook(path: str, built_assets: dict, root_data: dict, current_path: str = "") -> Any:
            return load_yaml(path)
    """

    def decorator(handler: Callable):
        HookRegistry.register(name, handler, before=before)
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


@hook("import", before=True)  # Process before discovery so imported types are available
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


@hook("env", before=True)  # Process before discovery so env values are available
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


@hook("value", before=True)  # Process before discovery so values can be expanded
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


@hook(
    "include", before=True
)  # Process before discovery so included config is discovered
def include_hook(
    path: str,
    built_assets: dict[str, Any],
    root_data: dict[str, Any],
    current_path: str = "",
) -> dict[str, Any]:
    """Include and parse a YAML file, returning its contents as a dict.

    Usage in YAML:
        some_field: "@include:path/to/file.yaml"

    Args:
        path: Path to the YAML file to include (relative to current working directory)
        built_assets: Dictionary of already-built assets
        root_data: Original root configuration data
        current_path: Path of the current node being built

    Returns:
        The parsed YAML file contents as a dictionary

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Include file not found: {path}")

    try:
        with open(file_path) as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file {path}: {e}")

    if not isinstance(content, dict):
        raise ValueError(
            f"Include file {path} must contain a YAML dict at the top level, "
            f"got {type(content).__name__}"
        )

    logger.debug(f"@include:{path} loaded with keys: {list(content.keys())}")
    return content


@hook("asset", before=False)  # Process after building since it references built objects
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


@hook("call", before=False)  # Process after building since it references built objects
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
