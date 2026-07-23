#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stub_path="${project_root}/python/zephyr/_C.pyi"

cd "${project_root}"

case "${1:-}" in
  "")
    uv run --no-sync pybind11-stubgen -o python zephyr._C
    ;;
  --check)
    temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/zephyr-stubs.XXXXXX")"
    trap 'rm -rf -- "${temporary_directory}"' EXIT

    uv run --no-sync pybind11-stubgen -o "${temporary_directory}" zephyr._C
    diff -u "${stub_path}" "${temporary_directory}/zephyr/_C.pyi"
    ;;
  *)
    echo "Usage: $0 [--check]" >&2
    exit 2
    ;;
esac
