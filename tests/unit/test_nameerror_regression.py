"""Test regression where NameError in get_type_hints breaks discovery."""

from typing import TYPE_CHECKING

from pydantic import PrivateAttr

from cyantic import Blueprint, CyanticModel, blueprint

if TYPE_CHECKING:
    # Type that's only available during type checking
    class UndefinedAtRuntime:
        pass


class Tensor:
    """Simple tensor class."""

    def __init__(self, value: int):
        self.value = value


@blueprint(Tensor) 
class TensorBlueprint(Blueprint[Tensor]):
    """Blueprint for Tensor."""

    value: int

    def build(self) -> Tensor:
        return Tensor(self.value)


class Voltage(CyanticModel):
    """Voltage model."""

    tau: Tensor


class Population(CyanticModel):
    """Population model.""" 

    voltages: list[Voltage]


class Layer(CyanticModel):
    """Layer model."""

    populations: dict[str, Population]


class Network(CyanticModel):
    """Network with private attr using undefined type."""

    layers: list[Layer]
    reuse_state: bool = False
    _cached_states: list["UndefinedAtRuntime"] | None = PrivateAttr(default=None)


def test_network_with_undefined_private_attr_type():
    """Test that undefined types in private attrs don't break field discovery."""
    config = {
        "layers": [
            {
                "populations": {
                    "e": {
                        "voltages": [
                            {"tau": 30}
                        ]
                    }
                }
            }
        ],
        "reuse_state": True
    }

    # This should work even though UndefinedAtRuntime doesn't exist
    network = Network.build(config)
    
    # Verify the tensor was built correctly
    tau = network.layers[0].populations["e"].voltages[0].tau
    assert isinstance(tau, Tensor)
    assert tau.value == 30
    
    # Verify other fields
    assert network.reuse_state is True
    assert network._cached_states is None