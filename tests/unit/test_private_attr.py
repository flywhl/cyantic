"""Test that PrivateAttr fields are properly ignored by Cyantic."""

import pytest
from pydantic import PrivateAttr

from cyantic import Blueprint, CyanticModel, blueprint


class SimpleService:
    """A simple service class."""

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name


@blueprint(SimpleService)
class SimpleServiceBlueprint(Blueprint[SimpleService]):
    """Blueprint for SimpleService."""

    name: str

    def build(self) -> SimpleService:
        """Build the service."""
        return SimpleService(self.name)


class ModelWithPrivateAttr(CyanticModel):
    """Model with both public and private attributes."""

    public_field: str
    _private_service: SimpleService | None = PrivateAttr(default=None)
    _counter: int = PrivateAttr(default=0)

    def increment(self):
        """Method that uses private attribute."""
        self._counter += 1
        return self._counter


def test_private_attrs_ignored_in_config():
    """Test that private attributes in config are ignored."""
    config = {
        "public_field": "hello",
        "_private_service": {
            "name": "my-service",
        },
        "_counter": 42,
    }

    # This should build successfully, ignoring the private fields
    model = ModelWithPrivateAttr.build(config)

    # Check public field works
    assert model.public_field == "hello"

    # Check private fields retain their defaults (weren't populated from config)
    assert model._private_service is None
    assert model._counter == 0

    # Verify private attrs still work programmatically
    assert model.increment() == 1
    assert model._counter == 1


def test_extra_private_fields_not_validated():
    """Test that extra fields starting with _ don't cause validation errors."""
    config = {
        "public_field": "world",
        "_some_random_field": "should be ignored",
        "_another_one": {"nested": "data"},
    }

    # Should not raise validation error for extra _ fields
    model = ModelWithPrivateAttr.build(config)
    assert model.public_field == "world"


def test_private_attrs_not_in_model_dump():
    """Test that private attributes don't appear in model serialization."""
    model = ModelWithPrivateAttr.build({"public_field": "test"})

    # Set a private attribute
    model._private_service = SimpleService("internal")

    # Check it's not in the dumped model
    dumped = model.model_dump()
    assert "public_field" in dumped
    assert "_private_service" not in dumped
    assert "_counter" not in dumped


def test_underscore_fields_are_private_by_default():
    """Test that fields starting with _ are automatically private in Pydantic V2."""

    class ModelWithUnderscore(CyanticModel):
        public_field: str
        _implicit_private: str  # In Pydantic V2, this is automatically private

    config = {
        "public_field": "hello",
        "_implicit_private": "I'm automatically private",
    }

    model = ModelWithUnderscore.build(config)
    assert model.public_field == "hello"

    # In Pydantic V2, underscore fields are private by default
    # So this should have the default value (undefined), not the config value
    with pytest.raises(AttributeError):
        # This will raise because private attrs without defaults are not accessible
        _ = model._implicit_private

    # Should not appear in model_dump since it's automatically private
    dumped = model.model_dump()
    assert "public_field" in dumped
    assert "_implicit_private" not in dumped
