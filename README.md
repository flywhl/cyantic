<div align="center">
  
  ![logo](https://github.com/user-attachments/assets/4ce903ef-ae04-4c9f-9a17-ad6932472de7)

  [Discord](https://discord.gg/kTkF2e69fH) • [Website](https://flywhl.dev) • [Installation](#installation)
  <br/>
  <br/>
</div>

> The [cyanotype](https://en.wikipedia.org/wiki/Cyanotype) process, which gave birth to the "blueprint", uses iron salts and UV light to transform simple negatives into deep Prussian blue artwork.

## Features

Cyantic lets you build complex types from simple blueprints, using Pydantic.

* Using intermediate Pydantic models as `Blueprints`, giving type safety and validation for parameters.
* Reference other values using `@value:x.y.z`.
* Import objects using `@import:x.y.z`.
* Inject environment variables using `@env:MY_VAR`.
* ...and define custom `@hook` handlers using a simple API.

## Installation

* `uv add cyantic`

## Usage

```python
from cyantic import Blueprint, blueprint, CyanticModel, hook
from torch import Tensor
import torch
import yaml

# 1. Create and register some useful parameterisations
#       (or soon install from PyPi, i.e. `rye add cyantic-torch`)

@blueprint(Tensor)
class NormalTensor(Blueprint[Tensor]):

    mean: float
    std: float
    size: tuple[int, ...]

    def build(self) -> Tensor:
        return torch.normal(self.mean, self.std, size=self.size)

@blueprint(Tensor)
class UniformTensor(Blueprint[Tensor]):
    low: float
    high: float
    size: tuple[int, ...]

    def build(self) -> Tensor:
      return torch.empty(self.size).uniform_(self.low, self.high)


# 2. Write pydantic models using `CyanticModel` base class

class MyModel(CyanticModel):
    normal_tensor: Tensor
    uniform_tensor: Tensor


# 3. (optional) define custom hooks

@hook("env")
def env_hook(name: str, _: ValidationContext) -> str:
    """Handle @env:VARIABLE_NAME references."""
    try:
        return os.environ[name]
    except KeyError as e:
        raise ValueError(f"Environment variable {name} not found") from e


# 4. Validate from YAML files that specify the parameterisation

some_yaml = """common:
    size: [3, 5]
    foo: @env:MY_VARIABLE
normal_tensor:
    mean: 0.0
    std: 0.1
    size: @value:common.size
uniform_tensor:
    low: -1.0
    std: 1.0
    size: @value:common.size
"""

# 5. Receive objects built from the parameterisations.

my_model = MyModel.model_validate(yaml.safe_load(some_yaml))
assert isinstance(my_model.normal_tensor, Tensor)
assert isinstance(my_model.uniform_tensor, Tensor)
```


## Development

* `git clone https://github.com/flywhl/cyantic.git`
* `cd cyantic`
* `uv sync`
* `just test`

## Flywheel

Science needs humble software tools. [Flywheel](https://flywhl.dev/) is an open source collective building simple tools to preserve scientific momentum, inspired by devtools and devops culture. Join our Discord [here](discord.gg/fd37MFZ7RS).
