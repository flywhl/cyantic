"""Built-in blueprints for Cyantic."""

from typing import Any

from pydantic import Field

from .core import Blueprint, BlueprintRegistry


class Classifier(Blueprint[Any]):
    """Blueprint for creating a class from a module path and kwargs.

    This is registered for `object` type, making it a fallback blueprint
    for any non-Pydantic class.

    Example:
        {
            "cls": "my.module.MyClass",
            "kwargs": {"arg1": "value1", "arg2": 42}
        }
    """

    cls: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def build(self) -> Any:
        """Import the class and instantiate it with kwargs."""
        module_path, class_name = self.cls.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        target_class = getattr(module, class_name)
        return target_class(**self.kwargs)


# Register Classifier as the first (and only) blueprint for object type
# We do this manually instead of using @blueprint to ensure it's at index 0
BlueprintRegistry.register(object, Classifier)
