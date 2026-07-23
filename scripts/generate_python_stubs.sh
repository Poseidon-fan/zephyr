#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stub_path="${project_root}/python/zephyr/_C.pyi"

cd "${project_root}"

format_stub() {
  local generated_stub="$1"

  uv run --no-sync ruff check --fix --quiet --config "${project_root}/pyproject.toml" "${generated_stub}"
  uv run --no-sync ruff format --quiet --config "${project_root}/pyproject.toml" "${generated_stub}"
}

case "${1:-}" in
  "")
    uv run --no-sync pybind11-stubgen -o python zephyr._C
    format_stub "${stub_path}"
    ;;
  --check)
    temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/zephyr-stubs.XXXXXX")"
    trap 'rm -rf -- "${temporary_directory}"' EXIT

    uv run --no-sync pybind11-stubgen -o "${temporary_directory}" zephyr._C
    generated_stub="${temporary_directory}/zephyr/_C.pyi"
    format_stub "${generated_stub}"
    diff -u "${stub_path}" "${generated_stub}"
    ;;
  *)
    echo "Usage: $0 [--check]" >&2
    exit 2
    ;;
esac
