import importlib
import logging
import os
from typing import Any, Callable

from pydantic import BaseModel

from .context import ValidationContext

logger = logging.getLogger(__name__)

HOOK_PREFIX = "@"


class Hook(BaseModel):
    handler: Callable
    before: bool


class HookRegistry:
    """Global registry mapping reference hooks to their handler functions."""

    _hooks: dict[str, Hook] = {}

    @classmethod
    def register(cls, name: str, handler: Callable, before: bool = False):
        """Register a handler function for a reference hook."""
        if name in cls._hooks:
            raise ValueError(f"Handler already registered for hook: {name}")

        hook = Hook(handler=handler, before=before)
        cls._hooks[name] = hook
        logger.debug(f"Registered handler for hook: {hook}")

    @classmethod
    def get_hook(cls, name: str) -> Hook:
        """Get the handler for a reference hook."""
        if name not in cls._hooks:
            raise ValueError(f"No hook registered with name '{name}'")
        return cls._hooks[name]

    @classmethod
    def clear(cls):
        """Clear all registered handlers."""
        cls._hooks.clear()


def hook(hook: str, *, before: bool):
    """Decorator to register a reference hook handler function."""

    def decorator(handler: Callable):
        HookRegistry.register(hook, handler, before=before)
        return handler

    return decorator


# Implement built-in hooks
@hook("import", before=True)
def import_hook(path: str, _: ValidationContext) -> Any:
    """Handle @import:module.path.to.thing references."""
    module_path, attr = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(path) from e
    return getattr(module, attr)


@hook("value", before=True)
def value(path: str, context: ValidationContext) -> Any:
    """Handle @value:path.to.value references."""
    return context.get_nested_value(path)


@hook("env", before=True)
def env(name: str, _: ValidationContext) -> str:
    """Handle @env:VARIABLE_NAME references."""
    try:
        return os.environ[name]
    except KeyError as e:
        raise ValueError(f"Environment variable {name} not found") from e


@hook("asset", before=False)
def asset(path: str, ctx: ValidationContext) -> Any:
    """Handle @asset:path references with elegant hierarchical resolution."""
    return ctx.resolve_asset_reference(path)


@hook("call", before=False)
def call(path: str, ctx: ValidationContext) -> Any:
    """Handle @call:path.method references with elegant navigation."""
    current_context = ctx.get_current_path()

    from .context import navigate_path

    # For @call:services.service.get_value, we need to:
    # 1. Find where in the hierarchy we can start navigating (e.g., application.services)
    # 2. Navigate from there to the method (service.get_value)
    # 3. Call the method

    path_parts = path.split(".")

    # Try to find a starting point in the asset hierarchy
    # We'll try progressively longer prefixes until we find an asset
    for i in range(1, len(path_parts) + 1):
        prefix = ".".join(path_parts[:i])

        # Special case: if prefix is the full path, it means we're looking for a method on a direct asset
        if i == len(path_parts) and i > 1:
            # This is a method call like "error_raiser.raise_error"
            asset_name = path_parts[0]
            method_name = path_parts[-1]

            try:
                # Try to get the asset directly
                asset = navigate_path(ctx._context.built_assets, asset_name)
                method = getattr(asset, method_name)
                if callable(method):
                    try:
                        return method()
                    except Exception as e:
                        raise ValueError(
                            f"Error calling method '{path}': {str(e)}"
                        ) from e
                else:
                    raise ValueError(f"'{path}' is not callable")
            except (ValueError, AttributeError):
                pass

        # No need for model context anymore since we don't use model name prefix

        # Try direct resolution without model context
        try:
            asset = navigate_path(ctx._context.built_assets, prefix)

            # Navigate the remaining path from this asset
            remaining_parts = path_parts[i:]
            current = asset
            for part in remaining_parts:
                current = getattr(current, part)

            # Call if it's callable
            if callable(current):
                try:
                    return current()
                except Exception as e:
                    raise ValueError(f"Error calling method '{path}': {str(e)}") from e
            else:
                raise ValueError(f"'{path}' is not callable")

        except (ValueError, AttributeError):
            pass

    # If we get here, we couldn't resolve the path
    available_keys = ctx._get_all_keys(ctx._context.built_assets)
    raise ValueError(
        f"No callable found at path '{path}'. Available assets: {available_keys}"
    )
