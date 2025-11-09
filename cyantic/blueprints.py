"""Built-in blueprints for Cyantic."""

from typing import Any, Type

from pydantic import Field

from .core import Blueprint, BlueprintRegistry


def classifier(target_type: Type):
    """Register a Classifier blueprint for the given type.

    This allows building instances from a dict with 'cls' and 'kwargs' keys.

    Args:
        target_type: The type to register the classifier for

    Example:
        classifier(MyClass)

        # Now you can build MyClass instances like:
        config = {
            "my_field": {
                "cls": "my.module.MyClass",
                "kwargs": {"arg1": "value1"}
            }
        }
    """

    class Classifier(Blueprint[target_type]):
        """Blueprint for creating a class from a module path and kwargs.

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

    BlueprintRegistry.register(target_type, Classifier)
