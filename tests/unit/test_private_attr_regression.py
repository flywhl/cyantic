"""Test regression where PrivateAttr affects building of other fields."""

from pydantic import PrivateAttr

from cyantic import Blueprint, CyanticModel, blueprint


class Tensor:
    """Simple tensor class."""

    def __init__(self, value: int | float):
        self.value = value


@blueprint(Tensor)
class TensorBlueprint(Blueprint[Tensor]):
    """Blueprint for Tensor."""

    value: int | float

    def build(self) -> Tensor:
        return Tensor(self.value)


class Voltage(CyanticModel):
    """Voltage with tau parameter."""

    tau: Tensor


class Population(CyanticModel):
    """Population with voltages."""

    voltages: list[Voltage]


class Layer(CyanticModel):
    """Layer with populations."""

    populations: dict[str, Population]


class NetworkWithoutPrivate(CyanticModel):
    """Network without private attribute."""

    layers: list[Layer]


class NetworkWithPrivate(CyanticModel):
    """Network with private attribute."""

    layers: list[Layer]
    _cached_states: list[dict] | None = PrivateAttr(default=None)


def test_network_builds_without_private_attr():
    """Test that network builds correctly without private attr."""
    config = {
        "layers": [
            {
                "populations": {
                    "e": {
                        "voltages": [
                            {"tau": 30}  # This should be built into a Tensor
                        ]
                    }
                }
            }
        ]
    }

    # This should work
    network = NetworkWithoutPrivate.build(config)
    assert isinstance(network.layers[0].populations["e"].voltages[0].tau, Tensor)
    assert network.layers[0].populations["e"].voltages[0].tau.value == 30


def test_network_builds_with_private_attr():
    """Test that network still builds correctly with private attr."""
    config = {
        "layers": [
            {
                "populations": {
                    "e": {
                        "voltages": [
                            {"tau": 30}  # This should still be built into a Tensor
                        ]
                    }
                }
            }
        ]
    }

    # This should also work - private attr shouldn't affect other fields
    network = NetworkWithPrivate.build(config)
    assert isinstance(network.layers[0].populations["e"].voltages[0].tau, Tensor)
    assert network.layers[0].populations["e"].voltages[0].tau.value == 30
    assert network._cached_states is None  # Private attr has default value