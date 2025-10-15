import logging
from contextlib import contextmanager
from threading import local
from typing import Any, Optional

logger = logging.getLogger(__name__)


def navigate_path(obj: Any, path: str) -> Any:
    """Navigate a path like 'a.b.c' using either dict keys or object attributes."""
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


class ValidationContext:
    """Thread-local storage for validation context with hierarchical asset storage."""

    _context = local()

    @classmethod
    @contextmanager
    def root_data(cls, data: dict):
        """Initialize validation context with root data."""
        # Initialize depth counter if needed
        if not hasattr(cls._context, "depth"):
            cls._context.depth = 0

        # Set data only at top level
        if cls._context.depth == 0 and not hasattr(cls._context, "data"):
            cls._context.data = data
            cls._context.built_assets = {}
            cls._context.current_path = []

        cls._context.depth += 1
        try:
            yield
        finally:
            cls._context.depth -= 1
            # Only clean up data when unwinding the top level
            if cls._context.depth == 0 and hasattr(cls._context, "data"):
                del cls._context.data
                if hasattr(cls._context, "built_assets"):
                    del cls._context.built_assets
                if hasattr(cls._context, "current_path"):
                    del cls._context.current_path

    @classmethod
    def get_root_data(cls) -> Optional[dict]:
        """Get the current root data."""
        return getattr(cls._context, "data", None)

    @classmethod
    def get_nested_value(cls, path: str) -> Any:
        """Get a value from the root data using agnostic navigation."""
        data = cls.get_root_data()
        if not data:
            raise ValueError(
                f"Cannot get value at {path}, because there is no validation context data."
            )
        return navigate_path(data, path)

    @classmethod
    def get_current_path(cls) -> str:
        """Get the current path as a dot-notation string."""
        if not hasattr(cls._context, "current_path"):
            raise ValueError("No validation context available")
        return ".".join(cls._context.current_path)

    @classmethod
    def push_path(cls, segment: str) -> None:
        """Add a segment to the current path."""
        if not hasattr(cls._context, "current_path"):
            raise ValueError("No validation context available")
        cls._context.current_path.append(segment)

    @classmethod
    def pop_path(cls) -> None:
        """Remove the last segment from the current path."""
        if not hasattr(cls._context, "current_path"):
            raise ValueError("No validation context available")
        if cls._context.current_path:
            cls._context.current_path.pop()

    @classmethod
    def store_built_asset(cls, key: str, value: Any) -> None:
        """Store a built asset in the hierarchical structure."""
        if not hasattr(cls._context, "built_assets"):
            raise ValueError("No validation context available")

        # Navigate to the right spot in the hierarchy and store
        current = cls._context.built_assets
        path_segments = key.split(".")

        # Create nested structure as needed
        for segment in path_segments[:-1]:
            if segment not in current:
                current[segment] = {}
            elif not isinstance(current[segment], dict):
                # If there's already an object here, we can't nest further
                # This means we're trying to store nested attributes of an object
                # In this case, just return - the object itself will handle attribute access
                return
            current = current[segment]

        # Store the value
        current[path_segments[-1]] = value

    @classmethod
    def has_built_asset(cls, path: str) -> bool:
        """Check if a built asset exists at the given path."""
        try:
            cls.resolve_asset_reference(path)
            return True
        except ValueError:
            return False

    @classmethod
    def resolve_asset_reference(cls, path: str) -> Any:
        """Resolve an asset reference with intelligent path resolution."""
        if not hasattr(cls._context, "built_assets"):
            raise ValueError("No validation context available")

        # Simply try the path as specified
        try:
            return navigate_path(cls._context.built_assets, path)
        except ValueError:
            available_keys = cls._get_all_keys(cls._context.built_assets)
            raise ValueError(
                f"Asset not found at path: {path}. Available: {available_keys}"
            )

    @classmethod
    def _get_all_keys(cls, obj: Any, prefix: str = "") -> list[str]:
        """Get all available keys in the hierarchical structure for debugging."""
        keys = []
        if hasattr(obj, "keys"):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                keys.append(full_key)
                if hasattr(value, "keys"):
                    keys.extend(cls._get_all_keys(value, full_key))
        return keys
