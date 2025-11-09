"""Tests for external build system."""

import pytest
from cyantic.core import CyanticModel
from cyantic.blueprints import classifier


# Mock classes for testing
class Optimizer:
    """Mock optimizer base class."""

    params: list[list[int]]

    def step(self):
        pass


class Adam(Optimizer):
    """Mock Adam optimizer."""

    def __init__(self, params=None, lr=0.001):
        if params is None:
            params = []
        self.params = list(params) if not isinstance(params, list) else params
        self.lr = lr

    def step(self):
        pass


class Network:
    """Mock network class."""

    def parameters(self):
        """Return mock parameters."""
        return [[1.0, 2.0, 3.0]]


# Register classifiers for these types
classifier(Optimizer)
classifier(Network)


class ExperimentConfig(CyanticModel):
    """Test model with network and optimizer."""

    network: Network
    optimizer: Optimizer

    model_config = {"arbitrary_types_allowed": True}


class Database:
    """Mock database class."""

    def __init__(self, host="localhost"):
        self.host = host

    def get_connection_string(self):
        return f"postgres://{self.host}/db"


class Cache:
    """Mock cache class."""

    def __init__(self, db_connection=None):
        self.db_connection = db_connection


class ServiceContainer(CyanticModel):
    """Nested model with its own fields."""

    database: Database
    cache: Cache

    model_config = {"arbitrary_types_allowed": True}


class Monitor:
    """Mock monitor class."""

    def __init__(self, cache_ref=None):
        self.cache_ref = cache_ref


# Register classifiers for Database, Cache, and Monitor
classifier(Database)
classifier(Cache)
classifier(Monitor)


class Application(CyanticModel):
    """Root model with nested services."""

    services: ServiceContainer
    monitor: Monitor

    model_config = {"arbitrary_types_allowed": True}


def test_simple_build_with_call_hook():
    """Test basic build with @call: hook referencing sibling field."""
    config = {
        "network": {
            "cls": f"{__name__}.Network",
            "kwargs": {},
        },
        "optimizer": {
            "cls": f"{__name__}.Adam",
            "kwargs": {"params": "@call:network.parameters"},
        },
    }

    result = ExperimentConfig.build(config)

    # Check types by name to avoid module import issues
    assert type(result.network).__name__ == "Network"
    assert type(result.optimizer).__name__ == "Adam"
    # Type checker doesn't know optimizer is Adam, but we verified it above
    assert result.optimizer.params == [[1.0, 2.0, 3.0]]


def test_nested_models_with_full_path_references():
    """Test nested ECyanticModels with @call: hook using full paths."""
    config = {
        "services": {
            "database": {
                "cls": f"{__name__}.Database",
                "kwargs": {"host": "prod-db.example.com"},
            },
            "cache": {
                "cls": f"{__name__}.Cache",
                "kwargs": {
                    "db_connection": "@call:services.database.get_connection_string"
                },
            },
        },
        "monitor": {
            "cls": f"{__name__}.Monitor",
            "kwargs": {"cache_ref": "@asset:services"},
        },
    }

    result = Application.build(config)

    # Check services container built correctly
    assert type(result.services).__name__ == "ServiceContainer"
    assert type(result.services.database).__name__ == "Database"
    assert type(result.services.cache).__name__ == "Cache"

    # Check database configured correctly
    assert result.services.database.host == "prod-db.example.com"

    # Check @call: hook resolved correctly
    assert result.services.cache.db_connection == "postgres://prod-db.example.com/db"

    # Check monitor built correctly
    assert type(result.monitor).__name__ == "Monitor"

    # Check @asset: hook resolved correctly
    assert result.monitor.cache_ref is result.services


def test_asset_hook():
    """Test @asset: hook for referencing built objects."""
    config = {
        "services": {
            "database": {
                "cls": f"{__name__}.Database",
                "kwargs": {"host": "localhost"},
            },
            "cache": {
                "cls": f"{__name__}.Cache",
                "kwargs": {"db_connection": "@asset:services.database"},
            },
        },
        "monitor": {
            "cls": f"{__name__}.Monitor",
            "kwargs": {},
        },
    }

    result = Application.build(config)

    # Check that cache got the database object directly
    assert result.services.cache.db_connection is result.services.database


def test_import_hook():
    """Test @import: hook for dynamic imports."""

    class Container(CyanticModel):
        optimizer_class: type

        model_config = {"arbitrary_types_allowed": True}

    config = {
        "optimizer_class": f"@import:{__name__}.Adam",
    }

    result = Container.build(config)

    assert result.optimizer_class.__name__ == "Adam"
    assert callable(result.optimizer_class)


def test_env_hook(monkeypatch):
    """Test @env: hook for environment variables."""
    monkeypatch.setenv("TEST_HOST", "test.example.com")

    class Config(CyanticModel):
        host: str

    config = {
        "host": "@env:TEST_HOST",
    }

    result = Config.build(config)

    assert result.host == "test.example.com"


def test_missing_asset_error():
    """Test that missing asset references raise clear errors."""
    config = {
        "services": {
            "database": {
                "cls": f"{__name__}.Database",
                "kwargs": {},
            },
            "cache": {
                "cls": f"{__name__}.Cache",
                "kwargs": {"db_connection": "@asset:nonexistent"},
            },
        },
        "monitor": {
            "cls": f"{__name__}.Monitor",
            "kwargs": {},
        },
    }

    with pytest.raises(ValueError, match="no node exists with that path"):
        Application.build(config)


def test_missing_call_target_error():
    """Test that missing call targets raise clear errors."""
    config = {
        "network": {
            "cls": f"{__name__}.Network",
            "kwargs": {},
        },
        "optimizer": {
            "cls": f"{__name__}.Adam",
            "kwargs": {"params": "@call:nonexistent.method"},
        },
    }

    with pytest.raises(ValueError):
        ExperimentConfig.build(config)


def test_circular_dependency_error():
    """Test that circular dependencies are detected."""

    class StringHolder:
        def __init__(self, value: str):
            self.value = value

    # Register classifier for StringHolder so it gets discovered as a buildable node
    classifier(StringHolder)

    class CircularConfig(CyanticModel):
        a: StringHolder
        b: StringHolder

        model_config = {"arbitrary_types_allowed": True}

    # This creates a circular dependency: a depends on b, b depends on a
    config = {
        "a": {
            "cls": "tests.unit.external_build_test.test_circular_dependency_error.<locals>.StringHolder",
            "kwargs": {"value": "@asset:b"},
        },
        "b": {
            "cls": "tests.unit.external_build_test.test_circular_dependency_error.<locals>.StringHolder",
            "kwargs": {"value": "@asset:a"},
        },
    }

    with pytest.raises(ValueError, match="Circular dependency detected"):
        CircularConfig.build(config)


def test_progressive_prefix_matching():
    """Test that @call: hook tries progressively longer prefixes."""
    # This tests that @call:services.database.get_connection_string
    # correctly finds 'services.database' in built_assets and then
    # navigates to .get_connection_string method

    config = {
        "services": {
            "database": {
                "cls": f"{__name__}.Database",
                "kwargs": {"host": "example.com"},
            },
            "cache": {
                "cls": f"{__name__}.Cache",
                "kwargs": {
                    # This should match 'services.database', not 'services'
                    "db_connection": "@call:services.database.get_connection_string"
                },
            },
        },
        "monitor": {
            "cls": f"{__name__}.Monitor",
            "kwargs": {},
        },
    }

    result = Application.build(config)

    # The connection string should be the result of calling the method
    assert result.services.cache.db_connection == "postgres://example.com/db"
