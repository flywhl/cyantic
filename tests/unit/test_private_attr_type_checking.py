"""Test private attr with TYPE_CHECKING imports."""

from typing import TYPE_CHECKING

from pydantic import PrivateAttr

from cyantic import Blueprint, CyanticModel, blueprint

if TYPE_CHECKING:
    # This simulates LayerState being behind TYPE_CHECKING
    from typing import Any as LayerState
else:
    # At runtime, LayerState doesn't exist
    LayerState = None


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


class NetworkWithTypeCheckingPrivate(CyanticModel):
    """Network with private attr using TYPE_CHECKING import."""

    layers: list[Layer]
    _cached_states: list["LayerState"] | None = PrivateAttr(default=None)  # Forward ref


def test_network_with_type_checking_private_attr():
    """Test that TYPE_CHECKING imports in private attrs don't break discovery."""
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

    # This might fail if TYPE_CHECKING affects type hint resolution
    network = NetworkWithTypeCheckingPrivate.build(config)
    assert isinstance(network.layers[0].populations["e"].voltages[0].tau, Tensor)
    assert network.layers[0].populations["e"].voltages[0].tau.value == 30


def test_type_hints_with_type_checking():
    """Check what happens to type hints with TYPE_CHECKING imports."""
    from typing import get_type_hints
    
    try:
        hints = get_type_hints(NetworkWithTypeCheckingPrivate)
        print(f"Type hints resolved: {hints}")
    except NameError as e:
        print(f"Type hint resolution failed: {e}")
        # Try without evaluating strings
        hints = get_type_hints(NetworkWithTypeCheckingPrivate, include_extras=True, globalns={"LayerState": object})
        print(f"Type hints with globalns: {hints}")