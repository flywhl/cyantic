default:
    @just --list

test:
    @uv run pytest

test-s:
    @uv run pytest -s -o log_cli=True -o log_cli_level=DEBUG

fix dir=".":
    uv run ruff format {{dir}}

ruff dir=".":
    uv run ruff check {{dir}}

pyright dir=".":
    uv run pyright {{dir}}

lint dir=".":
    just ruff {{dir}}
    just pyright {{dir}}

