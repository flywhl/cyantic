# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Cyantic Development Guide

## Commands
- Build: `uv sync`
- Test: `just test` 
- Test with debug: `just test-s`
- Single test: `uv run pytest tests/unit/path_to_test.py::TestClass::test_method -v`
- Lint: `just lint`
- Typecheck: `just pyright`
- Format: `just ruff-fix`
- Lint single file: `just lint-file path/to/file.py`

## Architecture Overview

Cyantic uses Pydantic to build complex types from simple parameter specifications:

### Core Components
- **Blueprint**: Base class (`cyantic/core.py:310`) for parameter specs that build into instances via `build()` method
- **CyanticModel**: Base model class (`cyantic/core.py:42`) that automatically validates and builds fields during Pydantic validation
- **ValidationContext**: Thread-local context (`cyantic/context.py:9`) that tracks root data, built assets, and current path during validation
- **HookRegistry**: Global registry (`cyantic/hooks.py:20`) for reference handlers like `@value:`, `@env:`, `@import:`

### Hook System
Two-phase processing model:
1. **Before hooks** (`before=True`): Process during initial validation - `@value:`, `@env:`, `@import:`
2. **After hooks** (`before=False`): Process after objects are built - `@asset:`, `@call:`

Built-in hooks registered in `cyantic/hooks.py:58-104`:
- `@value:path.to.value` - Reference values from root data
- `@env:VAR_NAME` - Environment variables  
- `@import:module.path.Class` - Dynamic imports
- `@asset:path.to.built` - Reference built objects
- `@call:path.to.built.method` - Call methods on built objects

### Blueprint Registry
Global mapping (`cyantic/core.py:14`) from target types to their blueprint classes, populated via `@blueprint(TargetType)` decorator.

## Code Style
- **Imports**: Group by stdlib → third-party → local, alphabetically within groups
- **Types**: Type hints required for all parameters and returns
- **Naming**: 
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
- **Error handling**: Catch specific exceptions, raise with context
- **Documentation**: Docstrings for all public classes and methods
- **Formatting**: Managed by ruff
- **Max line length**: 100 characters
- **Test organization**: Test files mirror the source structure in tests/unit/

## Recent Development Work

### Dependency-Ordered Field Processing (Current Branch)
**Goal**: Support `@call:path.to.built.object.method` feature by ensuring objects are built before being referenced.

**Implementation**: 
- **Dependency Analysis** (`cyantic/core.py:15-82`): Extracts dependencies from `@value:`, `@asset:`, and `@call:` references in field values, including nested dict traversal
- **Topological Sorting** (`cyantic/core.py:84-135`): Uses Kahn's algorithm to order fields so dependencies are built before references  
- **Single-Pass Validation** (`cyantic/core.py:295-390`): Replaced 3-pass approach with dependency-ordered processing of before hooks → building → after hooks per field

### Elegant Hierarchical Asset Storage
**Problem**: Complex path resolution logic was error-prone and hard to debug.

**Solution**: Hierarchical storage with agnostic navigation:
- **Storage Structure**: `built_assets["model"]["field"] = object` instead of flat keys
- **Agnostic Navigation** (`cyantic/context.py:9-32`): `navigate_path()` uses both dict-style and attr-style access transparently
- **Smart Resolution** (`cyantic/context.py:133-144`): Simple names like `@asset:primary` try current model scope first, then fall back to direct paths
- **Asset Storage** (`cyantic/context.py:105-126`): Creates nested dict structure that mirrors model hierarchy

### Current Status
- ✅ Dependency analysis and topological sorting working correctly
- ✅ Single-pass validation replaces old 3-pass approach  
- ✅ Hierarchical storage with nested structure creation
- ✅ Agnostic navigation for mixed dict/object access
- 🔄 Working on: Cross-branch asset references (`@call:services.service.method` from nested contexts)
- 📋 Remaining: Handle pre-built CyanticModel object navigation in asset references

### Latest Work Session (Just Completed)
**Problem**: Tests failing on asset/call references despite correct dependency ordering and hierarchical storage.

**Investigation**: 
- `@asset:primary` from nested context (`complexdatacontainer.secondary.tensor`) needs to find `complexdatacontainer.primary`
- `@call:services.service.get_value` needs to navigate: `application.services` (object) → `.service` (attr) → `.get_value()` (method)

**Debug Findings**:
- Hierarchical storage working perfectly: assets stored as `application.services`, `complexdatacontainer.primary`, etc.
- Issue: When pre-built CyanticModel objects (like `ServiceContainer`) are passed in, their nested attributes need to be accessible via agnostic navigation
- Added `_store_nested_cyantic_structure()` to recursively store CyanticModel field structure
- Fixed storage conflicts where trying to store nested paths in already-occupied object slots

**Current State**: 
- Storage structure is correct and debuggable
- `navigate_path()` function handles dict/object access agnostically  
- Last issue: `@call:services.service.get_value` should resolve `services` → `application.services` → navigate `.service.get_value` → call result
- Hook logic in `cyantic/hooks.py:92-131` attempts this but still failing on final navigation step

**Next Steps**: Need to debug why agnostic navigation isn't finding methods on pre-built objects in hierarchical structure.