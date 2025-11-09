import os
import random
import statistics
from typing import Any, Sequence, Union, overload
from pydantic import BaseModel, Field, ValidationError

from cyantic import Blueprint, blueprint, CyanticModel


class SimpleModel(CyanticModel):
    """A simple model for testing hooks."""

    name: str


class Tensor(BaseModel):
    """A simple mock tensor class that wraps a list of numbers."""

    data: list[float]

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "Tensor":
        return cls(data=list(values))

    def __len__(self) -> int:
        return len(self.data)

    def mean(self) -> float:
        return statistics.mean(self.data)

    def std(self) -> float:
        return statistics.stdev(self.data)

    @overload
    def __getitem__(self, idx: int) -> float: ...

    @overload
    def __getitem__(self, idx: slice) -> "Tensor": ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[float, "Tensor"]:
        if isinstance(idx, slice):
            return Tensor(data=self.data[idx])
        return self.data[idx]

    def __hash__(self) -> int:
        """Make Tensor hashable for use in sets."""
        return hash(tuple(self.data))

    def __eq__(self, other: object) -> bool:
        """Define equality for hash consistency."""
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.data == other.data


@blueprint(Tensor)
class NormalTensor(Blueprint[Tensor]):
    """Blueprint for creating tensors from a normal distribution."""

    mean: float
    std_dev: float
    size: int = Field(gt=0, description="Size must be a positive integer")

    def build(self) -> Tensor:
        return Tensor(
            data=[random.gauss(self.mean, self.std_dev) for _ in range(self.size)]
        )


@blueprint(Tensor)
class UniformTensor(Blueprint[Tensor]):
    """Blueprint for creating tensors with values from a uniform distribution."""

    low: float
    high: float
    size: int = Field(gt=0, description="Size must be a positive integer")

    def build(self) -> Tensor:
        return Tensor(
            data=[random.uniform(self.low, self.high) for _ in range(self.size)]
        )


class DataContainer(CyanticModel):
    """Example model using blueprint-enabled tensor."""

    values: Tensor


class NestedConfig(BaseModel):
    """A regular Pydantic model for configuration."""

    name: str
    scale: float


class NestedTensorContainer(CyanticModel):
    """A CyanticModel that will be nested inside another CyanticModel."""

    config: NestedConfig
    tensor: Tensor


class ComplexDataContainer(CyanticModel):
    """A CyanticModel containing both regular fields and nested CyanticModels."""

    name: str
    primary: Tensor
    secondary: NestedTensorContainer


def test_value_reference():
    """Test the @value reference functionality."""
    # Test basic value reference
    data = {
        "stuff": {"tensor": {"mean": 0.0, "std_dev": 1.0, "size": 50}},
        "values": "@value:stuff.tensor",
    }

    model = DataContainer.build(data)
    assert isinstance(model.values, Tensor)
    assert len(model.values) == 50

    # Test nested value reference
    nested_data = {
        "config": {
            "tensors": {
                "primary": {"mean": 0.0, "std_dev": 1.0, "size": 30},
                "secondary": {"low": -1.0, "high": 1.0, "size": 20},
            }
        },
        "name": "foo",
        "primary": "@value:config.tensors.primary",
        "secondary": {
            "config": {"name": "nested", "scale": 2.0},
            "tensor": "@value:config.tensors.secondary",
        },
    }

    model = ComplexDataContainer.build(nested_data)
    assert len(model.primary) == 30
    assert len(model.secondary.tensor) == 20

    # Test invalid path
    invalid_data = {"values": "@value:nonexistent.path"}
    try:
        DataContainer.build(invalid_data)
        assert False, "Should have raised ValueError for invalid path"
    except ValueError:
        pass


def test_import_reference(mocker):
    """Test the @import reference functionality."""
    # Mock module with test object
    mock_module = mocker.Mock()
    mock_module.test_value = Tensor.from_list([1.0, 2.0, 3.0])

    # Setup mock for importlib.import_module
    mock_import = mocker.patch("importlib.import_module")

    def _side_effect(name: str):
        if name == "test.module":
            return mock_module
        raise ImportError(f"No module named '{name}'")

    mock_import.side_effect = _side_effect

    # Test successful import
    data = {"values": "@import:test.module.test_value"}
    model = DataContainer.build(data)
    assert isinstance(model.values, Tensor)
    assert model.values.data == [1.0, 2.0, 3.0]

    # Test invalid module
    data = {"values": "@import:nonexistent.module.value"}
    try:
        DataContainer.build(data)
        assert False, "Should have raised ValueError for invalid import"
    except ValueError:
        pass


def test_env_reference():
    """Test the @env reference functionality."""
    # Test successful env var reference
    os.environ["TEST_VAR"] = "test_value"
    data = {"name": "@env:TEST_VAR"}
    model = SimpleModel.build(data)
    assert model.name == "test_value"

    # Test missing env var raises ValidationError
    data = {"name": "@env:NONEXISTENT_VAR"}
    try:
        SimpleModel.build(data)
        assert False, "Should have raised ValidationError"
    except ValueError as e:
        assert "Environment variable NONEXISTENT_VAR not found" in str(e)


def test_asset_reference():
    """Test the @asset reference functionality."""
    # Test that we can reference a previously built field using @asset
    data = {
        "config": {
            "tensor1": {"mean": 0.0, "std_dev": 1.0, "size": 10},
        },
        "name": "model_name",
        "primary": "@value:config.tensor1",
        "secondary": {
            "config": {"name": "nested", "scale": 2.0},
            "tensor": "@asset:primary",  # Reference the built primary tensor
        },
    }

    model = ComplexDataContainer.build(data)

    # Verify both tensors exist and are the same object
    assert isinstance(model.primary, Tensor)
    assert isinstance(model.secondary.tensor, Tensor)
    assert len(model.primary) == 10
    assert model.secondary.tensor is model.primary  # Should be the same object


def test_call_hook():
    """Test the @call hook functionality for calling methods on built assets."""

    # Create a class with methods we want to call
    class MethodProvider:
        def __init__(self, value: str):
            self.value = value

        def get_value(self) -> str:
            return self.value

        def get_uppercase(self) -> str:
            return self.value.upper()

    # Create a blueprint for building the MethodProvider
    @blueprint(MethodProvider)
    class MethodProviderBlueprint(Blueprint[MethodProvider]):
        value: str

        def build(self) -> MethodProvider:
            return MethodProvider(self.value)

    # Create a model with a nested MethodProvider
    class ServiceContainer(CyanticModel):
        name: str
        service: MethodProvider

    # Create a model that will call methods on the MethodProvider
    class ServiceConsumer(CyanticModel):
        name: str
        original_value: str
        uppercase_value: str

    # Create a parent application model that contains both components
    class Application(CyanticModel):
        app_name: str
        services: ServiceContainer
        client: ServiceConsumer

    # Build the application with @call references
    app_data = {
        "app_name": "Test Application",
        "services": {
            "name": "Provider Service",
            "service": {"value": "hello world"},
        },
        "client": {
            "name": "Consumer Service",
            "original_value": "@call:services.service.get_value",
            "uppercase_value": "@call:services.service.get_uppercase",
        },
    }

    app = Application.build(app_data)

    # Verify the application was built correctly
    assert app.app_name == "Test Application"
    assert isinstance(app.services.service, MethodProvider)
    assert app.services.service.get_value() == "hello world"

    # Verify the @call references worked
    assert app.client.original_value == "hello world"
    assert app.client.uppercase_value == "HELLO WORLD"


def test_nested_dict_hooks():
    """Test that hooks are processed in nested dict fields."""

    class Service:
        def __init__(self, name: str):
            self.name = name

        def get_name(self) -> str:
            return self.name

    @blueprint(Service)
    class ServiceBlueprint(Blueprint[Service]):
        name: str

        def build(self) -> Service:
            return Service(self.name)

    class Config(CyanticModel):
        service: Service
        # Dict field with nested hooks
        settings: dict[str, Any]

    config_data = {
        "service": {"name": "TestService"},
        "settings": {
            "service_name": "@call:service.get_name",
            "nested": {
                "also_name": "@call:service.get_name",
            },
            "in_list": ["@call:service.get_name", "static_value"],
        },
    }

    config = Config.build(config_data)

    # Verify nested hooks were processed
    assert config.settings["service_name"] == "TestService"
    assert config.settings["nested"]["also_name"] == "TestService"
    assert config.settings["in_list"][0] == "TestService"
    assert config.settings["in_list"][1] == "static_value"
