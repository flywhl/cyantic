import pytest
import random
import statistics
from typing import Sequence, Union, overload

from pydantic import BaseModel, Field

from cyantic import Blueprint, blueprint, CyanticModel, classifier


class SimpleModel(CyanticModel):
    """A simple model for testing hooks."""

    name: str


class CounterModel(CyanticModel):
    """A model for testing stateful hooks."""

    first: int
    second: int
    third: int


class Tensor(BaseModel):
    """A simple mock tensor class that wraps a list of numbers.

    We use this because we don't want to take torch as a dependency.
    """

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


@blueprint(Tensor)
class ScalarTensor(Blueprint[Tensor]):
    """Blueprint for creating a single-element tensor from a scalar value."""

    value: float

    def build(self) -> Tensor:
        return Tensor(data=[self.value])


class DataContainer(CyanticModel):
    """Example model using cast-enabled tensor."""

    values: Tensor


def test_cast_build():
    """Test the cast building functionality."""
    # Test direct tensor assignment
    direct_tensor = Tensor.from_list([1.0, 2.0, 3.0])
    model1 = DataContainer(values=direct_tensor)
    assert len(model1.values) == 3
    assert list(model1.values.data) == [1.0, 2.0, 3.0]

    # Test normal distribution cast - verify only size and type
    cast_dict = {"values": {"mean": 0.0, "std_dev": 1.0, "size": 10}}
    model2 = DataContainer.build(cast_dict)
    assert len(model2.values) == 10
    assert isinstance(model2.values, Tensor)
    assert all(isinstance(x, float) for x in model2.values.data)

    # Test scalar blueprint - int scalar
    scalar_int = {"values": 5}
    model3 = DataContainer.build(scalar_int)
    assert len(model3.values) == 1
    assert isinstance(model3.values, Tensor)
    assert model3.values.data == [5.0]

    # Test scalar blueprint - float scalar
    scalar_float = {"values": 3.14}
    model4 = DataContainer.build(scalar_float)
    assert len(model4.values) == 1
    assert model4.values.data == [3.14]

    # Test validation error for missing required fields
    try:
        DataContainer.build({"values": {"mean": 0.0}})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


class TensorContainers(CyanticModel):
    """A model containing tensors in various container types."""

    list_tensors: list[Tensor]
    dict_tensors: dict[str, Tensor]
    set_tensors: set[Tensor]
    tuple_tensors: tuple[Tensor, ...]


@pytest.mark.skip(reason="Blueprint building in containers not yet supported")
def test_container_of_blueprints():
    """Test handling various container types with blueprinted fields."""
    # Test data with different container types
    container_data = {
        "list_tensors": [
            {"mean": 0.0, "std_dev": 1.0, "size": 10},
            {"low": 0.0, "high": 1.0, "size": 20},
        ],
        "dict_tensors": {
            "normal": {"mean": 0.0, "std_dev": 1.0, "size": 15},
            "uniform": {"low": -1.0, "high": 1.0, "size": 25},
        },
        "set_tensors": [  # Sets can be initialized from lists
            {"mean": 1.0, "std_dev": 0.5, "size": 30},
            {"low": -2.0, "high": 2.0, "size": 40},
        ],
        "tuple_tensors": [  # Tuples can be initialized from lists
            {"mean": -1.0, "std_dev": 2.0, "size": 45},
            {"low": 0.0, "high": 5.0, "size": 55},
        ],
    }

    model = TensorContainers.build(container_data)

    # Test list container
    assert isinstance(model.list_tensors, list)
    assert len(model.list_tensors) == 2
    assert all(isinstance(t, Tensor) for t in model.list_tensors)
    assert len(model.list_tensors[0]) == 10
    assert len(model.list_tensors[1]) == 20

    # Test dict container
    assert isinstance(model.dict_tensors, dict)
    assert set(model.dict_tensors.keys()) == {"normal", "uniform"}
    assert all(isinstance(t, Tensor) for t in model.dict_tensors.values())
    assert len(model.dict_tensors["normal"]) == 15
    assert len(model.dict_tensors["uniform"]) == 25

    # Test set container
    assert isinstance(model.set_tensors, set)
    assert len(model.set_tensors) == 2
    assert all(isinstance(t, Tensor) for t in model.set_tensors)
    assert any(len(t) == 30 for t in model.set_tensors)
    assert any(len(t) == 40 for t in model.set_tensors)

    # Test tuple container
    assert isinstance(model.tuple_tensors, tuple)
    assert len(model.tuple_tensors) == 2
    assert all(isinstance(t, Tensor) for t in model.tuple_tensors)
    assert len(model.tuple_tensors[0]) == 45
    assert len(model.tuple_tensors[1]) == 55

    # Test validation with invalid items
    invalid_data = {
        "list_tensors": [{"mean": 0.0}],  # Missing required fields
        "dict_tensors": {"bad": {"std_dev": 1.0}},  # Missing required fields
        "set_tensors": [{"size": -1}],  # Invalid size
        "tuple_tensors": [{"low": 1.0, "high": 0.0}],  # Invalid range
    }

    try:
        TensorContainers.build(invalid_data)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test uniform distribution cast - verify size and types
    model3 = DataContainer.build({"values": {"low": -1.0, "high": 1.0, "size": 50}})
    assert len(model3.values) == 50
    assert isinstance(model3.values, Tensor)
    assert all(isinstance(x, float) for x in model3.values.data)


def test_blueprints_nested_in_dict():
    """Test blueprints nested inside dict fields - minimal reproduction of lab_test issue."""

    class Parameters(CyanticModel):
        """Model with Tensor fields, similar to Voltage/Conductance in lab_test."""

        tau: Tensor
        threshold: Tensor

    class Container(CyanticModel):
        """Model with dict containing Parameters, similar to Layer.populations."""

        params: dict[str, Parameters]

    config = {
        "params": {
            "cell_a": {
                "tau": {"mean": 0.0, "std_dev": 1.0, "size": 5},
                "threshold": {"value": -55.0},  # Uses ScalarTensor blueprint
            },
            "cell_b": {
                "tau": {"low": 0.0, "high": 100.0, "size": 3},
                "threshold": {"value": -70.0},
            },
        }
    }

    model = Container.build(config)

    # Verify the dicts built correctly
    assert isinstance(model.params, dict)
    assert len(model.params) == 2

    # Verify the Parameters models built
    assert isinstance(model.params["cell_a"], Parameters)
    assert isinstance(model.params["cell_b"], Parameters)

    # Verify the Tensors were built from blueprints
    assert isinstance(model.params["cell_a"].tau, Tensor)
    assert len(model.params["cell_a"].tau) == 5
    assert isinstance(model.params["cell_a"].threshold, Tensor)
    assert len(model.params["cell_a"].threshold) == 1
    assert model.params["cell_a"].threshold.data[0] == -55.0

    assert isinstance(model.params["cell_b"].tau, Tensor)
    assert len(model.params["cell_b"].tau) == 3
    assert model.params["cell_b"].threshold.data[0] == -70.0


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


def test_nested_blueprint_models():
    """Test nested CyanticModels with mixed BaseModel types."""
    # Test nested structure with both normal and blueprint fields

    nested_data = {
        "name": "test_complex",
        "primary": {"mean": 0.0, "std_dev": 1.0, "size": 50},
        "secondary": {
            "config": {"name": "nested", "scale": 2.0},
            "tensor": {"low": -1.0, "high": 1.0, "size": 30},
        },
    }

    model = ComplexDataContainer.build(nested_data)

    # Verify top level fields
    assert model.name == "test_complex"
    assert len(model.primary) == 50

    # Verify nested structure
    assert model.secondary.config.name == "nested"
    assert model.secondary.config.scale == 2.0
    assert len(model.secondary.tensor) == 30


def test_mixed_model_validation():
    """Test validation behavior with mixed model types."""
    # Test validation with missing nested fields
    invalid_data = {
        "name": "test_invalid",
        "primary": {"mean": 0.0, "std_dev": 1.0, "size": 50},
        "secondary": {
            "config": {"name": "nested"},  # Missing scale
            "tensor": {"low": -1.0, "high": 1.0, "size": 30},
        },
    }

    try:
        ComplexDataContainer.build(invalid_data)
        assert False, "Should have raised ValueError for missing scale"
    except ValueError:
        pass

    # Test validation with invalid parameters
    invalid_data = {
        "name": "test_invalid",
        "primary": {"mean": 0.0, "std_dev": 1.0, "size": -50},  # Invalid size
        "secondary": {
            "config": {"name": "nested", "scale": 2.0},
            "tensor": {"low": -1.0, "high": 1.0, "size": 30},
        },
    }

    try:
        ComplexDataContainer.build(invalid_data)
        assert False, "Should have raised ValueError for negative size"
    except ValueError:
        pass


def test_blueprint_type_inference():
    """Test that we correctly infer and apply different blueprints."""
    # Test that both NormalTensor and UniformTensor blueprints work in the same model
    mixed_data = {
        "name": "test_mixed",
        "primary": {"mean": 0.0, "std_dev": 1.0, "size": 100},
        "secondary": {
            "config": {"name": "nested", "scale": 2.0},
            "tensor": {"low": 0.0, "high": 1.0, "size": 100},
        },
    }

    model = ComplexDataContainer.build(mixed_data)

    # Verify that different blueprints were correctly applied
    assert len(model.primary) == 100
    assert len(model.secondary.tensor) == 100

    # Verify types and sizes only
    assert isinstance(model.primary, Tensor)
    assert isinstance(model.secondary.tensor, Tensor)
    assert len(model.primary) == 100
    assert len(model.secondary.tensor) == 100
    assert all(isinstance(x, float) for x in model.primary.data)
    assert all(isinstance(x, float) for x in model.secondary.tensor.data)


def test_list_of_cyantic_models():
    """Test that we can handle list[CyanticModel] fields."""

    class Item(CyanticModel):
        name: str
        value: int

    class Container(CyanticModel):
        items: list[Item]

    config = {
        "items": [
            {"name": "first", "value": 1},
            {"name": "second", "value": 2},
            {"name": "third", "value": 3},
        ]
    }

    result = Container.build(config)

    assert isinstance(result.items, list)
    assert len(result.items) == 3
    assert all(isinstance(item, Item) for item in result.items)
    assert result.items[0].name == "first"
    assert result.items[0].value == 1
    assert result.items[1].name == "second"
    assert result.items[2].value == 3


def test_dict_of_cyantic_models():
    """Test that we can handle dict[str, CyanticModel] fields."""

    class Config(CyanticModel):
        host: str
        port: int

    class ServiceMap(CyanticModel):
        services: dict[str, Config]

    config = {
        "services": {
            "api": {"host": "api.example.com", "port": 8080},
            "db": {"host": "db.example.com", "port": 5432},
        }
    }

    result = ServiceMap.build(config)

    assert isinstance(result.services, dict)
    assert len(result.services) == 2
    assert all(isinstance(cfg, Config) for cfg in result.services.values())
    assert result.services["api"].host == "api.example.com"
    assert result.services["api"].port == 8080
    assert result.services["db"].port == 5432


def test_nested_basemodels_under_cyanticmodel():
    """Test that nested models can be plain BaseModel, not CyanticModel."""

    # These are plain Pydantic BaseModels, not CyanticModels
    class Address(BaseModel):
        street: str
        city: str
        zip_code: str

    class Person(BaseModel):
        name: str
        age: int
        address: Address

    # Only the top-level needs to be CyanticModel
    class Company(CyanticModel):
        name: str
        employees: list[Person]

    config = {
        "name": "Acme Corp",
        "employees": [
            {
                "name": "Alice",
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "Springfield",
                    "zip_code": "12345",
                },
            },
            {
                "name": "Bob",
                "age": 25,
                "address": {
                    "street": "456 Oak Ave",
                    "city": "Shelbyville",
                    "zip_code": "67890",
                },
            },
        ],
    }

    result = Company.build(config)

    # Verify everything built correctly
    assert result.name == "Acme Corp"
    assert len(result.employees) == 2

    # Verify nested BaseModels work
    assert isinstance(result.employees[0], Person)
    assert result.employees[0].name == "Alice"
    assert result.employees[0].age == 30

    assert isinstance(result.employees[0].address, Address)
    assert result.employees[0].address.street == "123 Main St"
    assert result.employees[0].address.city == "Springfield"

    assert result.employees[1].name == "Bob"
    assert result.employees[1].address.city == "Shelbyville"


def test_deeply_nested_collections():
    """Test deeply nested collection structures like dict[str, dict[str, Model]]."""

    class Metric(BaseModel):
        value: float
        unit: str

    class Dashboard(CyanticModel):
        # Deeply nested: dict -> dict -> Model
        metrics: dict[str, dict[str, Metric]]

    config = {
        "metrics": {
            "server1": {
                "cpu": {"value": 45.5, "unit": "percent"},
                "memory": {"value": 8.2, "unit": "GB"},
            },
            "server2": {
                "cpu": {"value": 72.3, "unit": "percent"},
                "memory": {"value": 12.1, "unit": "GB"},
                "disk": {"value": 512.0, "unit": "GB"},
            },
        }
    }

    result = Dashboard.build(config)

    # Verify deeply nested structure
    assert isinstance(result.metrics, dict)
    assert len(result.metrics) == 2

    assert isinstance(result.metrics["server1"], dict)
    assert len(result.metrics["server1"]) == 2

    assert isinstance(result.metrics["server1"]["cpu"], Metric)
    assert result.metrics["server1"]["cpu"].value == 45.5
    assert result.metrics["server1"]["cpu"].unit == "percent"

    assert result.metrics["server2"]["memory"].value == 12.1
    assert result.metrics["server2"]["disk"].unit == "GB"


def test_mixed_nested_collections():
    """Test mixed nested collections: list[dict[str, Model]]."""

    class Task(BaseModel):
        name: str
        priority: int

    class Project(CyanticModel):
        # List of dicts of models
        milestones: list[dict[str, Task]]

    config = {
        "milestones": [
            {
                "design": {"name": "Design phase", "priority": 1},
                "prototype": {"name": "Build prototype", "priority": 2},
            },
            {
                "testing": {"name": "QA testing", "priority": 3},
                "launch": {"name": "Production launch", "priority": 4},
            },
        ]
    }

    result = Project.build(config)

    assert len(result.milestones) == 2
    assert isinstance(result.milestones[0], dict)
    assert isinstance(result.milestones[0]["design"], Task)
    assert result.milestones[0]["design"].name == "Design phase"
    assert result.milestones[0]["design"].priority == 1
    assert result.milestones[1]["launch"].priority == 4


def test_complex_deeply_nested_structure():
    """Test a complex, deeply nested structure with multiple levels of nesting."""

    # Deeply nested BaseModels
    class Tag(BaseModel):
        name: str
        color: str

    class Metadata(BaseModel):
        created_by: str
        tags: list[Tag]
        properties: dict[str, str]

    class Attachment(BaseModel):
        filename: str
        size_bytes: int
        metadata: Metadata

    class Comment(BaseModel):
        author: str
        text: str
        attachments: list[Attachment]
        replies: list["Comment"]  # Recursive!

    class Issue(BaseModel):
        title: str
        description: str
        comments: list[Comment]
        labels: dict[str, Tag]
        related_issues: dict[str, list[str]]

    # Only top-level is CyanticModel
    class ProjectTracker(CyanticModel):
        # dict[repo_name, dict[issue_id, Issue]]
        repositories: dict[str, dict[str, Issue]]

    config = {
        "repositories": {
            "backend": {
                "ISS-001": {
                    "title": "Fix login bug",
                    "description": "Users cannot login",
                    "comments": [
                        {
                            "author": "alice",
                            "text": "Investigating...",
                            "attachments": [
                                {
                                    "filename": "logs.txt",
                                    "size_bytes": 1024,
                                    "metadata": {
                                        "created_by": "alice",
                                        "tags": [
                                            {"name": "urgent", "color": "red"},
                                            {"name": "backend", "color": "blue"},
                                        ],
                                        "properties": {
                                            "environment": "production",
                                            "server": "web-01",
                                        },
                                    },
                                }
                            ],
                            "replies": [
                                {
                                    "author": "bob",
                                    "text": "Found the issue!",
                                    "attachments": [],
                                    "replies": [],
                                }
                            ],
                        }
                    ],
                    "labels": {
                        "priority": {"name": "high", "color": "red"},
                        "type": {"name": "bug", "color": "orange"},
                    },
                    "related_issues": {
                        "depends_on": ["ISS-002"],
                        "blocks": ["ISS-003", "ISS-004"],
                    },
                },
                "ISS-002": {
                    "title": "Database migration",
                    "description": "Migrate to v2 schema",
                    "comments": [],
                    "labels": {
                        "priority": {"name": "medium", "color": "yellow"},
                    },
                    "related_issues": {},
                },
            },
            "frontend": {
                "ISS-100": {
                    "title": "Update UI",
                    "description": "New design system",
                    "comments": [
                        {
                            "author": "charlie",
                            "text": "Starting work",
                            "attachments": [
                                {
                                    "filename": "mockup.png",
                                    "size_bytes": 2048,
                                    "metadata": {
                                        "created_by": "charlie",
                                        "tags": [{"name": "design", "color": "purple"}],
                                        "properties": {"format": "png"},
                                    },
                                },
                                {
                                    "filename": "colors.css",
                                    "size_bytes": 512,
                                    "metadata": {
                                        "created_by": "charlie",
                                        "tags": [],
                                        "properties": {},
                                    },
                                },
                            ],
                            "replies": [],
                        }
                    ],
                    "labels": {},
                    "related_issues": {"related_to": ["ISS-001"]},
                },
            },
        }
    }

    result = ProjectTracker.build(config)

    # Verify top-level structure
    assert isinstance(result.repositories, dict)
    assert len(result.repositories) == 2
    assert "backend" in result.repositories
    assert "frontend" in result.repositories

    # Verify second level (dict[str, Issue])
    assert isinstance(result.repositories["backend"], dict)
    assert len(result.repositories["backend"]) == 2

    # Verify Issue model
    issue = result.repositories["backend"]["ISS-001"]
    assert isinstance(issue, Issue)
    assert issue.title == "Fix login bug"

    # Verify deeply nested comments
    assert len(issue.comments) == 1
    comment = issue.comments[0]
    assert isinstance(comment, Comment)
    assert comment.author == "alice"

    # Verify attachments in comments
    assert len(comment.attachments) == 1
    attachment = comment.attachments[0]
    assert isinstance(attachment, Attachment)
    assert attachment.filename == "logs.txt"
    assert attachment.size_bytes == 1024

    # Verify metadata in attachments
    metadata = attachment.metadata
    assert isinstance(metadata, Metadata)
    assert metadata.created_by == "alice"

    # Verify tags list in metadata
    assert len(metadata.tags) == 2
    assert isinstance(metadata.tags[0], Tag)
    assert metadata.tags[0].name == "urgent"
    assert metadata.tags[0].color == "red"

    # Verify properties dict in metadata
    assert isinstance(metadata.properties, dict)
    assert metadata.properties["environment"] == "production"

    # Verify recursive comments (replies)
    assert len(comment.replies) == 1
    reply = comment.replies[0]
    assert isinstance(reply, Comment)
    assert reply.author == "bob"
    assert len(reply.replies) == 0

    # Verify labels dict with Tag values
    assert isinstance(issue.labels, dict)
    assert isinstance(issue.labels["priority"], Tag)
    assert issue.labels["priority"].name == "high"

    # Verify related_issues dict with list values
    assert isinstance(issue.related_issues, dict)
    assert isinstance(issue.related_issues["depends_on"], list)
    assert issue.related_issues["depends_on"][0] == "ISS-002"
    assert len(issue.related_issues["blocks"]) == 2

    # Verify second repository
    frontend_issue = result.repositories["frontend"]["ISS-100"]
    assert frontend_issue.title == "Update UI"
    assert len(frontend_issue.comments[0].attachments) == 2
    assert frontend_issue.comments[0].attachments[1].filename == "colors.css"


class ArbitraryClass:
    """A simple class for testing the Classifier blueprint."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value


# Register classifier for ArbitraryClass
classifier(ArbitraryClass)


class ClassifierContainer(CyanticModel):
    """Container for testing Classifier blueprint."""

    obj: ArbitraryClass


def test_classifier_blueprint():
    """Test the Classifier blueprint with explicit registration."""
    data = {
        "obj": {
            "cls": "core_test.ArbitraryClass",  # Use pytest's relative module path
            "kwargs": {"name": "test", "value": 42},
        }
    }

    model = ClassifierContainer.build(data)
    assert isinstance(model.obj, ArbitraryClass)
    assert model.obj.name == "test"
    assert model.obj.value == 42
