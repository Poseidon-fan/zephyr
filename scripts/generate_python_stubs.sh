#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stub_path="${project_root}/python/zephyr/_C.pyi"

cd "${project_root}"

case "${1:-}" in
  ""|--check) ;;
  *)
    echo "Usage: $0 [--check]" >&2
    exit 2
    ;;
esac

temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/zephyr-stubs.XXXXXX")"
trap 'rm -rf -- "${temporary_directory}"' EXIT

uv run --no-sync python - "${temporary_directory}" <<'PY'
import importlib.util
import os
from pathlib import Path
import sys
import types

import pybind11_stubgen as stubgen

# An explicit artifact avoids importing the Python wrapper or triggering an editable rebuild.
if extension := os.environ.get("ZEPHYR_PYTHON_EXTENSION"):
    package = types.ModuleType("zephyr")
    package.__path__ = []
    sys.modules["zephyr"] = package
    spec = importlib.util.spec_from_file_location("zephyr._C", Path(extension).resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules["zephyr._C"] = module
    spec.loader.exec_module(module)
    package._C = module

args = stubgen.arg_parser().parse_args(
    ["--ignore-invalid-expressions", r"^<.* object at 0x[0-9a-f]+>$", "--exit-code", "zephyr._C"],
    namespace=stubgen.CLIArgs(),
)
parser = stubgen.stub_parser_from_args(args)
parse_annotation = parser.parse_annotation_str


def normalize_union(annotation):
    # stubgen 2.5.5 skips import/prefix fixes for PEP 604 union branches. Its generic
    # type path recursively applies those fixes, so normalize unions before parsing.
    alternatives = parser._split_type_union_str(annotation)
    if alternatives is not None and len(alternatives) > 1:
        annotation = f"typing.Union[{', '.join(alternatives)}]"
    return parse_annotation(annotation)


parser.parse_annotation_str = normalize_union
stubgen.run(
    parser,
    stubgen.Printer(invalid_expr_as_ellipses=True),
    "zephyr._C",
    Path(sys.argv[1]),
    sub_dir=None,
    dry_run=False,
    writer=stubgen.Writer(),
)
PY

generated_stub="${temporary_directory}/_C.pyi"
uv run --no-sync ruff format --quiet --config "${project_root}/pyproject.toml" "${generated_stub}"
if [[ "${1:-}" == --check ]]; then
  diff -u "${stub_path}" "${generated_stub}"
else
  cp -- "${generated_stub}" "${stub_path}"
fi
