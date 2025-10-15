import logging
from collections import defaultdict, deque
from typing import TypeVar, Generic, Any, get_type_hints, get_args
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, ValidationError
from pydantic_core import core_schema

from .context import ValidationContext
from .hooks import HOOK_PREFIX, HookRegistry

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DependencyAnalyzer:
    """Analyzes field dependencies for topological sorting."""

    @staticmethod
    def extract_dependencies(data: dict[str, Any]) -> dict[str, set[str]]:
        """Extract all dependencies between fields.

        Returns a dict mapping field_name -> set of fields it depends on.
        """
        dependencies = defaultdict(set)

        for field_name, field_value in data.items():
            deps = DependencyAnalyzer._extract_field_dependencies(field_value)
            dependencies[field_name].update(deps)

        return dict(dependencies)

    @staticmethod
    def _extract_field_dependencies(value: Any) -> set[str]:
        """Extract dependencies from a single field value."""
        deps = set()

        if isinstance(value, str) and value.startswith(HOOK_PREFIX):
            # Extract dependencies from hook references
            hook_deps = DependencyAnalyzer._extract_hook_dependencies(value)
            deps.update(hook_deps)
        elif isinstance(value, dict):
            # Recursively check nested dictionaries
            for nested_value in value.values():
                nested_deps = DependencyAnalyzer._extract_field_dependencies(
                    nested_value
                )
                deps.update(nested_deps)
        elif isinstance(value, (list, tuple)):
            # Check list/tuple items
            for item in value:
                item_deps = DependencyAnalyzer._extract_field_dependencies(item)
                deps.update(item_deps)

        return deps

    @staticmethod
    def _extract_hook_dependencies(reference: str) -> set[str]:
        """Extract dependencies from a hook reference like @value:path or @call:path.method."""
        if not reference.startswith(HOOK_PREFIX):
            return set()

        hook_name, path = reference[1:].split(":", 1)

        if hook_name == "value":
            # @value:path.to.field -> depends on the root field
            # E.g., @value:common.size -> depends on "common"
            root_field = path.split(".")[0]
            return {root_field}
        elif hook_name in ("asset", "call"):
            # @asset:path.to.built or @call:path.to.built.method
            # E.g., @asset:model.tensor -> depends on "model"
            # E.g., @call:model.tensor.mean -> depends on "model"
            if "." in path:
                root_field = path.split(".")[0]
                return {root_field}
            else:
                return {path}
        elif hook_name in ("env", "import"):
            # These don't depend on other fields
            return set()
        else:
            # Unknown hook, assume no dependencies for now
            return set()


class TopologicalSorter:
    """Performs topological sorting with cycle detection."""

    @staticmethod
    def sort(dependencies: dict[str, set[str]], all_fields: set[str]) -> list[str]:
        """Sort fields in topological order.

        Args:
            dependencies: Dict mapping field -> set of fields it depends on
            all_fields: All field names to sort

        Returns:
            List of field names in topological order

        Raises:
            ValueError: If circular dependencies are detected
        """
        # Build adjacency list (reverse of dependencies - who depends on me)
        graph = defaultdict(set)
        in_degree = defaultdict(int)

        # Initialize all fields
        for field in all_fields:
            in_degree[field] = 0

        # Build graph and calculate in-degrees
        for field, deps in dependencies.items():
            for dep in deps:
                if dep in all_fields:  # Only consider dependencies within this model
                    graph[dep].add(field)
                    in_degree[field] += 1

        # Kahn's algorithm for topological sorting
        queue = deque([field for field in all_fields if in_degree[field] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # Remove this node and update in-degrees
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(all_fields):
            remaining = all_fields - set(result)
            raise ValueError(f"Circular dependency detected among fields: {remaining}")

        return result


class BlueprintRegistry:
    """Global registry mapping types to their blueprints."""

    _blueprints: dict[type, list[type["Blueprint"]]] = {}

    @classmethod
    def register(cls, target_type: type, blueprint_type: type["Blueprint"]):
        """Register a blueprint for a target type."""
        if target_type not in cls._blueprints:
            cls._blueprints[target_type] = []
        cls._blueprints[target_type].append(blueprint_type)

    @classmethod
    def get_blueprints(cls, target_type: type) -> list[type["Blueprint"]]:
        """Get all registered blueprints for a type."""
        return cls._blueprints.get(target_type, [])


def blueprint(target_type: type):
    """Decorator to register a blueprint for a given type."""

    def decorator(blueprint_type: type["Blueprint"]):
        BlueprintRegistry.register(target_type, blueprint_type)
        return blueprint_type

    return decorator


class CyanticModel(BaseModel):
    """Base model class that automatically builds fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # def __init__(self, **data: Any) -> None:
    #     """Initialize the model with validation context support."""
    #     # Support direct construction with validation context
    #     root_context_missing = ValidationContext.get_root_data() is None
    #     if root_context_missing and data:
    #         with ValidationContext.root_data(data):
    #             super().__init__(**data)
    #     else:
    #         super().__init__(**data)

    @staticmethod
    def _get_fields_requiring_validation(hints: dict[str, Any]) -> set[str]:
        """Get fields that need construction."""

        def needs_validation(type_):
            if BlueprintRegistry.get_blueprints(type_):
                return True
            if not hasattr(type_, "__origin__"):
                return False
            # Check if it's a container type
            origin = type_.__origin__
            if origin in (list, dict, set, tuple):
                # For dict, only check values
                if origin is dict:
                    value_type = get_args(type_)[1]
                    return needs_validation(value_type)
                # For other containers, check their contained type
                contained_type = get_args(type_)[0]
                return needs_validation(contained_type)
            return False

        return {name for name, type_ in hints.items() if needs_validation(type_)}

    @staticmethod
    def _validate_container_field(
        field_name: str, field_value: Any, container_type: type, value_type: type
    ) -> Any:
        """Validate and build container field (list, set, tuple, dict)."""
        try:
            if container_type is dict:
                return {
                    k: (
                        CyanticModel.try_build(value_type, v)
                        if isinstance(v, dict)
                        else v
                    )
                    for k, v in field_value.items()
                }
            elif container_type in (list, set, tuple):
                validated = [
                    CyanticModel.try_build(value_type, item)
                    if isinstance(item, dict)
                    else item
                    for item in field_value
                ]
                return container_type(validated)
        except ValueError as e:
            raise ValueError(f"Error building item in {field_name}: {str(e)}")

    @staticmethod
    def _get_raw_type(field_type: type) -> type:
        """Get the raw type without generic parameters."""
        return (
            field_type.__origin__ if hasattr(field_type, "__origin__") else field_type
        )

    @classmethod
    def _process_reference(cls, reference: str, before: bool | None = None) -> Any:
        """Process a reference like @value:path or @asset:path.

        Args:
            reference: The reference string (like "@value:path.to.value")
            before: If set, only process hooks matching this phase.
                   If None, process regardless of phase.

        Returns:
            The processed value, or the original reference if skipped due to phase.
        """
        logger.debug(f"Processing reference: {reference}")
        assert reference.startswith(HOOK_PREFIX)
        hook_name, value = reference[1:].split(":")  # slice out the @-prefix
        hook = HookRegistry.get_hook(hook_name)

        # If a phase is specified, skip hooks that don't match
        if before is not None and before != hook.before:
            return reference  # Return original reference to be processed in the other phase

        # Make sure the validation context has root data for the nested references
        if ValidationContext.get_root_data() is None:
            # Create a dummy context for initialization
            with ValidationContext.root_data({}):
                return hook.handler(value, ValidationContext)
        else:
            return hook.handler(value, ValidationContext)

    @classmethod
    def _store_nested_cyantic_structure(
        cls, obj: Any, base_path: str, has_context: bool
    ) -> None:
        """Store nested structure of CyanticModel objects for asset resolution."""
        if not has_context or not isinstance(obj, CyanticModel):
            return

        # Store each field of the CyanticModel
        for field_name, field_value in obj.model_dump().items():
            field_path = f"{base_path}.{field_name}"
            try:
                ValidationContext.store_built_asset(field_path, field_value)
                # Recursively store nested structures
                cls._store_nested_cyantic_structure(
                    field_value, field_path, has_context
                )
            except ValueError:
                pass  # Skip if can't store

    @classmethod
    def _is_before_hook(cls, hook_name: str) -> bool:
        """Check if a hook is a 'before' hook."""
        try:
            hook = HookRegistry.get_hook(hook_name)
            return hook.before
        except ValueError:
            # Unknown hook, assume it's a before hook for safety
            return True

    @classmethod
    def validate_cyantic_fields(
        cls, v: Any, fields_requiring_validation: set[str], hints: dict[str, Any]
    ) -> Any:
        """Validate and build cyantic fields in a model using topological ordering."""
        if not isinstance(v, dict):
            return v

        # Check if we have a validation context
        has_context = True
        try:
            # Just check if context exists without pushing model name
            ValidationContext.get_current_path()
        except ValueError:
            # No validation context available
            has_context = False

        try:
            # Analyze dependencies and get topological order
            dependencies = DependencyAnalyzer.extract_dependencies(v)
            all_fields = set(field for field in v.keys() if field in hints)

            if not all_fields:
                return v

            field_order = TopologicalSorter.sort(dependencies, all_fields)

            # Single pass: Process fields in dependency order
            for field_name in field_order:
                field_value = v[field_name]
                field_type = hints[field_name]

                # Push field context for nested validation
                if has_context:
                    ValidationContext.push_path(field_name)

                try:
                    # Step 1: Process before hooks (@value, @env, @import)
                    if isinstance(field_value, str) and field_value.startswith("@"):
                        hook_name = field_value[1:].split(":")[0]
                        if cls._is_before_hook(hook_name):
                            field_value = v[field_name] = cls._process_reference(
                                field_value, before=True
                            )

                    # Step 2: Build the object if needed
                    if field_name in fields_requiring_validation:
                        raw_type = cls._get_raw_type(field_type)

                        # Skip if value is already of the target type
                        if isinstance(field_value, raw_type):
                            pass  # Already the right type, no building needed
                        elif hasattr(field_type, "__origin__"):
                            # Handle container types
                            origin = field_type.__origin__
                            if origin in (list, dict, set, tuple):
                                type_args = get_args(field_type)
                                value_type = (
                                    type_args[1] if origin is dict else type_args[0]
                                )
                                if BlueprintRegistry.get_blueprints(value_type):
                                    field_value = v[field_name] = (
                                        cls._validate_container_field(
                                            field_name, field_value, origin, value_type
                                        )
                                    )
                        elif isinstance(field_value, dict):
                            # Handle direct blueprint types with dict input
                            try:
                                field_value = v[field_name] = cls.try_build(
                                    field_type, field_value
                                )
                            except ValueError as e:
                                raise ValueError(
                                    f"Error building {field_name}: {str(e)}"
                                )
                        elif BlueprintRegistry.get_blueprints(field_type):
                            # Handle scalar blueprint types (non-dict input)
                            # Wrap the scalar value as {"value": <scalar>} and try building
                            try:
                                field_value = v[field_name] = cls.try_build(
                                    field_type, {"value": field_value}
                                )
                            except ValueError:
                                # No blueprint matched the wrapped scalar, let Pydantic handle it
                                pass

                    # Step 3: Process after hooks (@call, @asset)
                    if isinstance(field_value, str) and field_value.startswith("@"):
                        hook_name = field_value[1:].split(":")[0]
                        if not cls._is_before_hook(hook_name):
                            field_value = v[field_name] = cls._process_reference(
                                field_value, before=False
                            )

                    # Store the built asset in the hierarchical structure
                    if has_context:
                        try:
                            current_path = ValidationContext.get_current_path()
                            ValidationContext.store_built_asset(
                                current_path, field_value
                            )

                            # If this is a pre-built CyanticModel, also store its nested structure
                            cls._store_nested_cyantic_structure(
                                field_value, current_path, has_context
                            )
                        except ValueError:
                            # No validation context available, skip storing
                            pass

                finally:
                    # Pop field path when done with this field
                    if has_context:
                        ValidationContext.pop_path()

            return v
        except Exception:
            # Re-raise any exceptions
            raise

    @classmethod
    def try_build(cls, target_type: type, value: dict) -> Any:
        """Try each registered blueprint in order until one works."""
        blueprints = BlueprintRegistry.get_blueprints(target_type)
        if not blueprints:
            raise ValueError(f"No blueprint registered for type {target_type}")

        errors = []
        for blueprint_type in blueprints:
            try:
                blueprint = blueprint_type.model_validate(value)
                return blueprint.build()
            except (ValidationError, ValueError) as e:
                errors.append(f"{blueprint_type.__name__}: {str(e)}")
                continue
            except Exception as e:
                logger.error(e)
                raise

        error_msg = "\n".join(f"- {err}" for err in errors)
        raise ValueError(
            f"No compatible blueprint found for {target_type}. Tried:\n{error_msg}"
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        schema = super().__get_pydantic_core_schema__(_source_type, _handler)

        hints = get_type_hints(cls)
        fields_requiring_validation = cls._get_fields_requiring_validation(hints)

        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(
                    lambda v: cls.validate_cyantic_fields(
                        v, fields_requiring_validation, hints
                    )
                ),
                schema,
            ]
        )

    @classmethod
    def model_validate(cls, obj: Any, *args, **kwargs):
        """Validate and build the model."""
        if not isinstance(obj, dict):
            return super().model_validate(obj, *args, **kwargs)

        with ValidationContext.root_data(obj):
            model = super().model_validate(obj, *args, **kwargs)
            # ValidationContext.store_built_asset(model)

        return model


class Blueprint(CyanticModel, Generic[T]):
    """Base class for parameter specifications that can be built into instances."""

    def build(self) -> T:
        raise NotImplementedError

    @property
    def fields(self):
        raise NotImplementedError()
