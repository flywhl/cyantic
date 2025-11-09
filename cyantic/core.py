"""Core Cyantic build system.

This module implements a bottom-up build strategy:
1. Discover all buildable nodes in the config tree
2. Analyze dependencies between nodes
3. Topologically sort nodes
4. Build in dependency order
5. Assemble final nested structure
6. Validate with Pydantic
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Generic, Type, TypeVar, get_type_hints, get_origin, get_args
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict

from .hooks import HOOK_PREFIX, HookRegistry

logger = logging.getLogger(__name__)

T = TypeVar("T")
CyanticModelT = TypeVar("CyanticModelT", bound="CyanticModel")


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
        """Get all registered blueprints for a type.

        Returns type-specific blueprints first, then falls back to blueprints
        registered for `object` (which apply to all types).

        Note: CyanticModel subclasses don't get the object fallback, since they
        should use normal Pydantic validation for dict inputs.
        """
        type_specific = cls._blueprints.get(target_type, [])

        # If we're already looking up object, don't add fallback (would be duplicate)
        if target_type is object:
            return type_specific

        fallback = cls._blueprints.get(object, [])

        # Don't apply object blueprints to Pydantic models (BaseModel or CyanticModel)
        # They should be validated normally by Pydantic
        try:
            if issubclass(target_type, BaseModel):
                return type_specific
        except TypeError:
            # target_type might not be a class (e.g., typing constructs)
            pass

        return type_specific + fallback


def blueprint(target_type: type):
    """Decorator to register a blueprint for a given type."""

    def decorator(blueprint_type: type["Blueprint"]):
        BlueprintRegistry.register(target_type, blueprint_type)
        return blueprint_type

    return decorator


def get_container_value_type(field_type: Type) -> Type | None:
    """Extract the value type from a container type annotation.

    Args:
        field_type: Type annotation like dict[str, Foo], list[Bar], etc.

    Returns:
        The value type (Foo, Bar, etc.) or None if not a recognized container
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is None or not args:
        return None

    # dict[K, V] -> return V (last arg)
    if origin is dict:
        return args[-1] if args else None

    # list[T], set[T], tuple[T, ...] -> return T (first arg)
    if origin in (list, set, tuple):
        return args[0] if args else None

    return None


def is_buildable_type(target_type: Type) -> bool:
    """Check if a type needs building (has blueprints or is a CyanticModel).

    Args:
        target_type: The type to check

    Returns:
        True if this type should be discovered and built
    """
    # Unwrap generics
    raw_type = get_origin(target_type) if get_origin(target_type) else target_type

    # If raw_type is None or not a type, it's not buildable
    if not isinstance(raw_type, type):
        return False

    # Skip plain container types
    if raw_type in (dict, list, tuple, set):
        return False

    # Check if it has blueprints
    if BlueprintRegistry.get_blueprints(raw_type):
        return True

    # Check if it's a CyanticModel or BaseModel subclass
    if issubclass(raw_type, (BaseModel, CyanticModel)):
        return True

    return False


@dataclass
class BuildNode:
    """Represents a node in the config tree that needs to be built."""

    path: str  # Dot-separated path like "network" or "services.database"
    field_name: str  # Just the field name like "database"
    target_type: Type  # The type to build
    config: Any  # The config data (dict, primitive, etc.)
    parent_path: str  # Parent path for assembly


class CyanticModel(BaseModel):
    """Base model class that uses explicit .build() instead of validation hooks."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    @classmethod
    def build(cls: Type[Self], config: dict) -> Self:
        """Build a model instance from config using external build system."""
        return build(cls, config)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> Self:
        """Override to use construct instead of validation for already-built objects."""
        # If obj is a dict with built objects, use model_construct to skip validation
        if isinstance(obj, dict):
            # Check if values are already built (not dicts)
            has_built_objects = any(
                not isinstance(v, (dict, list, str, int, float, bool, type(None)))
                for v in obj.values()
            )
            if has_built_objects:
                return cls.model_construct(**obj)
        return super().model_validate(obj, **kwargs)


class Blueprint(BaseModel, Generic[T]):
    """Base class for parameter specifications that can be built into instances."""

    def build(self) -> T:
        raise NotImplementedError


def discover_in_container(
    container_config: Any,
    value_type: Type,
    container_path: str,
    parent_path: str,
) -> list[BuildNode]:
    """Discover buildable nodes within a container (dict, list, etc.).

    Args:
        container_config: The container data (dict or list)
        value_type: The type of values in the container
        container_path: Path to the container field
        parent_path: Parent path for the container

    Returns:
        List of discovered nodes within the container
    """
    nodes = []

    # Check if value_type is itself a container with buildable values
    # This handles cases like dict[str, dict[str, Foo]]
    nested_value_type = get_container_value_type(value_type)

    if nested_value_type and is_buildable_type(nested_value_type):
        # Value type is a container - need to recurse into nested containers
        # Pass nested_value_type (not value_type) to discover the actual buildable nodes
        if isinstance(container_config, dict):
            for key, value_config in container_config.items():
                value_path = f"{container_path}.{key}"
                # Recurse into the nested container with nested_value_type
                # so we eventually reach the buildable nodes
                nested_nodes = discover_in_container(
                    value_config, nested_value_type, value_path, parent_path
                )
                nodes.extend(nested_nodes)
        elif isinstance(container_config, (list, tuple)):
            for idx, item_config in enumerate(container_config):
                item_path = f"{container_path}.{idx}"
                # Recurse into the nested container with nested_value_type
                # so we eventually reach the buildable nodes
                nested_nodes = discover_in_container(
                    item_config, nested_value_type, item_path, parent_path
                )
                nodes.extend(nested_nodes)
    else:
        # Value type is buildable (not a container) - discover normally
        if isinstance(container_config, dict):
            # For dict[K, V], discover nodes in each value
            for key, value_config in container_config.items():
                value_path = f"{container_path}.{key}"
                value_nodes = discover_buildable_nodes(
                    value_config, value_type, value_path, parent_path
                )
                nodes.extend(value_nodes)

        elif isinstance(container_config, (list, tuple)):
            # For list[T], discover nodes in each item
            for idx, item_config in enumerate(container_config):
                item_path = f"{container_path}.{idx}"
                item_nodes = discover_buildable_nodes(
                    item_config, value_type, item_path, parent_path
                )
                nodes.extend(item_nodes)

    # Note: set not supported for now since we can't preserve order/index

    return nodes


def discover_buildable_nodes(
    config: Any, target_type: Type, path: str = "", parent_path: str = ""
) -> list[BuildNode]:
    """Discover all nodes in the config tree that need building.

    Args:
        config: The configuration data (dict, list, or primitive)
        target_type: The target type we're trying to build
        path: Current dot-separated path (e.g., "services.database")
        parent_path: Path to parent node for assembly

    Returns:
        List of BuildNode objects in discovery order (depth-first)
    """
    nodes = []

    # Get raw type (unwrap generics)
    raw_type = (
        target_type.__origin__ if hasattr(target_type, "__origin__") else target_type
    )

    # Check if this type needs building
    needs_building = False

    # Skip plain container types - they don't need building even if they have blueprints
    if raw_type in (dict, list, tuple, set):
        needs_building = False
    # Check if it has blueprints
    elif BlueprintRegistry.get_blueprints(raw_type):
        needs_building = True
    # Check if it's a CyanticModel subclass
    elif isinstance(raw_type, type) and issubclass(raw_type, (BaseModel, CyanticModel)):
        needs_building = True

    # If this node itself needs building, we'll add it after discovering children
    # This gives us bottom-up ordering

    if isinstance(config, dict) and isinstance(raw_type, type):
        # Try to get type hints to discover nested buildable fields
        try:
            if issubclass(raw_type, BaseModel):
                try:
                    hints = get_type_hints(raw_type)
                except NameError:
                    # Forward reference that can't be resolved (e.g., self-referential types
                    # defined in local scope). Skip field discovery for this type.
                    hints = {}

                # Discover buildable fields recursively
                for field_name, field_type in hints.items():
                    if field_name not in config:
                        continue

                    field_config = config[field_name]
                    field_path = f"{path}.{field_name}" if path else field_name

                    # Check if field_type is a container with buildable values
                    value_type = get_container_value_type(field_type)

                    if value_type:
                        # This is a container - check if its values are buildable
                        # OR if it's a nested container (like dict[str, dict[str, Foo]])
                        nested_value_type = get_container_value_type(value_type)

                        if is_buildable_type(value_type) or (
                            nested_value_type and is_buildable_type(nested_value_type)
                        ):
                            # Container with buildable values (possibly nested)
                            # Recursively discover within the container
                            container_nodes = discover_in_container(
                                field_config, value_type, field_path, path
                            )
                            nodes.extend(container_nodes)
                        else:
                            # Container but values not buildable - skip
                            pass
                    else:
                        # Not a container - regular field, recursively discover nodes
                        child_nodes = discover_buildable_nodes(
                            field_config, field_type, field_path, path
                        )
                        nodes.extend(child_nodes)
        except (TypeError, AttributeError):
            # Not a class or doesn't have type hints
            pass

    # Add this node AFTER its children (bottom-up)
    if needs_building and path:  # Don't add root as a node
        field_name = path.split(".")[-1] if "." in path else path
        nodes.append(
            BuildNode(
                path=path,
                field_name=field_name,
                target_type=target_type,
                config=config,
                parent_path=parent_path,
            )
        )

    return nodes


def extract_dependencies_from_value(
    value: Any, valid_paths: set[str] | None = None
) -> set[str]:
    """Extract field dependencies from a config value (handles hooks and nested structures).

    Args:
        value: The config value to extract dependencies from
        valid_paths: Set of valid node paths for matching hook references

    Returns:
        Set of node paths that this value depends on
    """
    deps = set()

    if isinstance(value, str) and value.startswith(HOOK_PREFIX):
        # Extract dependencies from hook references
        hook_name, hook_path = value[1:].split(":", 1)

        # @value: references raw config data, not built nodes, so no dependency
        # @asset: and @call: reference built nodes, so they create dependencies
        if hook_name in ("asset", "call"):
            # These reference other fields
            # For full absolute paths, we need to find the longest matching prefix
            # E.g., @call:services.database.get_connection_string
            # Could be "services" or "services.database" - try all prefixes

            if valid_paths is not None:
                # Try progressively longer prefixes to find the actual node path
                parts = hook_path.split(".")
                matched = False
                for i in range(len(parts), 0, -1):
                    potential_path = ".".join(parts[:i])
                    if potential_path in valid_paths:
                        deps.add(potential_path)
                        matched = True
                        break

                if not matched:
                    # Can't match yet - just take first segment as placeholder
                    # Will be validated later in analyze_dependencies
                    deps.add(parts[0])
            else:
                # No valid paths provided - just take first segment
                root_field = hook_path.split(".")[0]
                deps.add(root_field)

    elif isinstance(value, dict):
        # Recursively check nested dicts
        for nested_value in value.values():
            deps.update(extract_dependencies_from_value(nested_value, valid_paths))
    elif isinstance(value, (list, tuple)):
        # Check list/tuple items
        for item in value:
            deps.update(extract_dependencies_from_value(item, valid_paths))

    return deps


def analyze_dependencies(nodes: list[BuildNode]) -> dict[str, set[str]]:
    """Analyze dependencies between nodes.

    Only exact full path matches are allowed - no relative references.

    Returns:
        Dict mapping node path -> set of node paths it depends on

    Raises:
        ValueError: If a dependency references a non-existent node path
    """
    dependencies = defaultdict(set)

    # Build a set of all valid node paths for validation
    valid_paths = {node.path for node in nodes}

    for node in nodes:
        # Skip CyanticModel nodes - they're assembled from children, not built from config
        raw_type = (
            node.target_type.__origin__
            if hasattr(node.target_type, "__origin__")
            else node.target_type
        )
        if isinstance(raw_type, type) and issubclass(raw_type, CyanticModel):
            # CyanticModel nodes implicitly depend on all their children
            children = [n for n in nodes if n.parent_path == node.path]
            for child in children:
                dependencies[node.path].add(child.path)
            continue

        # Extract dependencies from the node's config, passing valid paths for smart matching
        deps = extract_dependencies_from_value(node.config, valid_paths)

        # Only exact full path matches are allowed
        for dep_path in deps:
            if dep_path not in valid_paths:
                raise ValueError(
                    f"Node '{node.path}' has dependency on '{dep_path}', but no node exists with that path. "
                    f"Available paths: {sorted(valid_paths)}. "
                    f"Hint: Use full absolute paths like 'services.database' instead of just 'database'"
                )
            dependencies[node.path].add(dep_path)

    return dict(dependencies)


def topological_sort_nodes(
    nodes: list[BuildNode], dependencies: dict[str, set[str]]
) -> list[BuildNode]:
    """Sort nodes in topological order based on dependencies.

    Args:
        nodes: List of all nodes
        dependencies: Dict mapping node path -> set of paths it depends on

    Returns:
        Nodes sorted in build order (dependencies first)

    Raises:
        ValueError: If circular dependencies detected
    """
    # Build lookup
    node_by_path = {node.path: node for node in nodes}

    # Build graph and calculate in-degrees
    graph = defaultdict(set)
    in_degree = defaultdict(int)

    # Initialize all nodes
    for node in nodes:
        in_degree[node.path] = 0

    # Build graph
    for node_path, deps in dependencies.items():
        for dep_path in deps:
            if dep_path in node_by_path:
                graph[dep_path].add(node_path)
                in_degree[node_path] += 1

    # Kahn's algorithm
    queue = deque([node.path for node in nodes if in_degree[node.path] == 0])
    result = []

    while queue:
        current_path = queue.popleft()
        result.append(node_by_path[current_path])

        # Remove this node and update in-degrees
        for neighbor_path in graph[current_path]:
            in_degree[neighbor_path] -= 1
            if in_degree[neighbor_path] == 0:
                queue.append(neighbor_path)

    # Check for cycles
    if len(result) != len(nodes):
        remaining = set(node.path for node in nodes) - set(node.path for node in result)
        raise ValueError(f"Circular dependency detected among nodes: {remaining}")

    return result


def process_hooks(
    config: Any,
    built_assets: dict[str, Any],
    root_data: dict[str, Any],
    current_node_path: str = "",
) -> Any:
    """Process hooks in config, replacing them with resolved values.

    Args:
        config: Config value (can be dict, list, string, etc.)
        built_assets: Dict mapping paths to built objects
        root_data: Original config dict before any building
        current_node_path: Path of the node being built

    Returns:
        Config with hooks replaced by their resolved values
    """
    if isinstance(config, str) and config.startswith(HOOK_PREFIX):
        hook_name, hook_path = config[1:].split(":", 1)

        # Look up hook in registry
        if not HookRegistry.has(hook_name):
            raise ValueError(f"Unknown hook: @{hook_name}:")

        hook_handler = HookRegistry.get(hook_name)
        return hook_handler(hook_path, built_assets, root_data, current_node_path)

    elif isinstance(config, dict):
        return {
            k: process_hooks(v, built_assets, root_data, current_node_path)
            for k, v in config.items()
        }

    elif isinstance(config, list):
        return [
            process_hooks(item, built_assets, root_data, current_node_path)
            for item in config
        ]

    elif isinstance(config, tuple):
        return tuple(
            process_hooks(item, built_assets, root_data, current_node_path)
            for item in config
        )

    else:
        return config


def reconstruct_nested_value(
    path_parts: list[str],
    value: Any,
    root_config: dict[str, Any],
    current_result: dict[str, Any] | list[Any],
) -> None:
    """Recursively reconstruct nested container structure.

    Args:
        path_parts: Remaining path parts (e.g., ['ff_dendrites', 'e', 'stimulus'])
        value: The value to place at the end of the path
        root_config: Original config to determine container types
        current_result: Current reconstruction target (modified in place)
    """
    if len(path_parts) == 1:
        # Base case: direct assignment
        key = path_parts[0]
        if isinstance(current_result, list):
            # For lists, the key should be an index - just append
            current_result.append(value)
        else:
            # For dicts, use the key directly
            current_result[key] = value
        return

    # Recursive case: need to go deeper
    field_name = path_parts[0]
    remaining_parts = path_parts[1:]

    if isinstance(current_result, list):
        # Current level is a list - field_name should be an index
        # This shouldn't happen in typical usage but handle it gracefully
        idx = int(field_name)
        # Extend list if needed
        while len(current_result) <= idx:
            current_result.append({})
        # Recurse into the list element
        reconstruct_nested_value(
            remaining_parts, value, root_config, current_result[idx]
        )
    else:
        # Current level is a dict
        # Initialize the field if it doesn't exist
        if field_name not in current_result:
            # Check the config to determine what type of container to create
            # Walk down the config following the path
            config_value = root_config
            for part in path_parts[: len(path_parts) - len(remaining_parts)]:
                if isinstance(config_value, dict) and part in config_value:
                    config_value = config_value[part]
                elif isinstance(config_value, list):
                    # Try to convert part to int for list indexing
                    try:
                        idx = int(part)
                        if idx < len(config_value):
                            config_value = config_value[idx]
                        else:
                            config_value = {}
                            break
                    except (ValueError, IndexError):
                        config_value = {}
                        break
                else:
                    config_value = {}
                    break

            # Determine container type from config
            if isinstance(config_value, list):
                current_result[field_name] = []
            else:
                current_result[field_name] = {}

        # Recurse into the nested structure
        reconstruct_nested_value(
            remaining_parts, value, root_config, current_result[field_name]
        )


def build_node(
    node: BuildNode,
    built_assets: dict[str, Any],
    all_nodes: list[BuildNode],
    root_data: dict[str, Any],
) -> Any:
    """Build a single node.

    Args:
        node: The node to build
        built_assets: Already-built assets for hook resolution
        all_nodes: All nodes for finding children
        root_data: Original config dict before any building

    Returns:
        The built object
    """
    raw_type = (
        node.target_type.__origin__
        if hasattr(node.target_type, "__origin__")
        else node.target_type
    )

    # Special case: if this is a CyanticModel, assemble from children
    if isinstance(raw_type, type) and issubclass(raw_type, CyanticModel):
        # Find children nodes (nodes whose parent_path is this node's path)
        children = [n for n in all_nodes if n.parent_path == node.path]

        # Build dict from built children, reconstructing containers
        assembled = {}
        built_field_names = set()

        for child in children:
            # Get relative path from this node to the child
            path_parts = child.path.split(".")
            parent_parts = node.path.split(".") if node.path else []
            relative_parts = (
                path_parts[len(parent_parts) :] if parent_parts else path_parts
            )

            if len(relative_parts) == 1:
                # Direct child - simple assignment
                assembled[relative_parts[0]] = built_assets[child.path]
                built_field_names.add(relative_parts[0])
            else:
                # Nested child - need to recursively reconstruct container structure
                reconstruct_nested_value(
                    relative_parts,
                    built_assets[child.path],
                    node.config,
                    assembled,
                )
                built_field_names.add(relative_parts[0])

        # Also include non-built fields from config (primitives, etc.)
        if isinstance(node.config, dict):
            for field_name, field_value in node.config.items():
                if field_name not in built_field_names:
                    # Process hooks in non-built fields
                    assembled[field_name] = process_hooks(
                        field_value,
                        built_assets,
                        root_data,
                        f"{node.path}.{field_name}",
                    )

        # Validate with the model
        result = raw_type.model_validate(assembled)
        logger.debug(
            f"Built {node.path} (CyanticModel) from {len(children)} children and {len(assembled) - len(children)} non-built fields"
        )
        return result

    # For non-CyanticModel nodes, process hooks and build normally
    processed_config = process_hooks(node.config, built_assets, root_data, node.path)

    # If after processing hooks, the config is already the correct type, just return it
    try:
        if isinstance(processed_config, raw_type):
            logger.debug(f"Node {node.path} already correct type after hook processing")
            return processed_config
    except TypeError:
        # raw_type might not be valid for isinstance check
        pass

    # Try blueprints first
    blueprints = BlueprintRegistry.get_blueprints(raw_type)
    if blueprints:
        # If config is not a dict, try wrapping it as {"value": config} for scalar blueprints
        if not isinstance(processed_config, dict):
            processed_config = {"value": processed_config}

        # Try each blueprint
        for blueprint_type in blueprints:
            try:
                blueprint = blueprint_type.model_validate(processed_config)
                result = blueprint.build()
                logger.debug(
                    f"Built {node.path} using blueprint {blueprint_type.__name__}"
                )
                return result
            except Exception as e:
                logger.debug(f"Blueprint {blueprint_type.__name__} failed: {e}")
                continue

        raise ValueError(f"No compatible blueprint found for {node.path}")

    # Fall back to direct Pydantic validation (for regular BaseModel)
    if isinstance(raw_type, type) and issubclass(raw_type, BaseModel):
        # But only if it's not a plain container type
        if raw_type not in (dict, list, tuple, set):
            result = raw_type.model_validate(processed_config)
            logger.debug(f"Built {node.path} using Pydantic validation")
            return result

    # If not buildable, just return the processed config
    return processed_config


def assemble_nested_dict(
    nodes: list[BuildNode], built_assets: dict[str, Any]
) -> dict[str, Any]:
    """Assemble built objects into a nested dict structure.

    Args:
        nodes: All build nodes
        built_assets: Built objects keyed by path

    Returns:
        Nested dict ready for final model validation
    """
    result = {}

    # Group nodes by parent path
    by_parent = defaultdict(list)
    for node in nodes:
        by_parent[node.parent_path].append(node)

    # Process root-level nodes (parent_path == "")
    for node in by_parent[""]:
        result[node.field_name] = built_assets[node.path]

    # For nested nodes, we need to check if parent expects a dict or object
    # This is simplified - we'll just put top-level items in result
    # More complex nesting would require walking the structure

    return result


def build(target_type: Type[CyanticModelT], config: dict) -> CyanticModelT:
    """Build a model instance from config using external build system.

    Args:
        target_type: The type to build (should be CyanticModel subclass)
        config: Configuration dict

    Returns:
        Built and validated model instance
    """
    logger.debug(f"Building {target_type.__name__} with external build system")

    # Keep reference to original config as root_data for @value: hooks
    root_data = config

    # 1. Discover all buildable nodes
    nodes = discover_buildable_nodes(config, target_type)
    logger.debug(f"Discovered {len(nodes)} buildable nodes: {[n.path for n in nodes]}")

    # 2. Analyze dependencies
    dependencies = analyze_dependencies(nodes)
    logger.debug(f"Dependencies: {dependencies}")

    # 3. Topological sort
    sorted_nodes = topological_sort_nodes(nodes, dependencies)
    logger.debug(f"Build order: {[n.path for n in sorted_nodes]}")

    # 4. Build each node in order
    built_assets = {}
    for node in sorted_nodes:
        logger.debug(f"Building node: {node.path}")
        built_obj = build_node(node, built_assets, nodes, root_data)
        built_assets[node.path] = built_obj
        logger.debug(f"  -> {type(built_obj).__name__}")

    # 5. Assemble nested structure from built nodes
    # Get all root-level nodes (nodes whose parent_path is "")
    root_nodes = [node for node in sorted_nodes if not node.parent_path]

    assembled = {}
    built_field_names = set()

    for node in root_nodes:
        # Check if this is a direct field or container item
        path_parts = node.path.split(".")

        if len(path_parts) == 1:
            # Direct field - simple assignment
            assembled[path_parts[0]] = built_assets[node.path]
            built_field_names.add(path_parts[0])
        else:
            # Nested container item - use recursive reconstruction
            reconstruct_nested_value(
                path_parts,
                built_assets[node.path],
                config,
                assembled,
            )
            built_field_names.add(path_parts[0])

    # Add non-built fields from config (primitives, plain dicts, etc.)
    for field_name, field_value in config.items():
        if field_name not in built_field_names:
            # Process hooks in non-built fields
            assembled[field_name] = process_hooks(
                field_value, built_assets, root_data, field_name
            )

    logger.debug(f"Assembled structure: {list(assembled.keys())}")

    # 6. Final validation with Pydantic
    return target_type.model_validate(assembled)
