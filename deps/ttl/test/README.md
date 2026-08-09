# TTL Test Architecture

TTL tests are organized by responsibility first and subsystem second. A test belongs to the lowest layer that can
verify the behavior without hiding a required dependency.

## Layers

- `contract/` verifies public-header self-containment, compile-time traits, public type mappings, and API contracts.
- `unit/` verifies one public or private component in isolation. CUDA unit tests may use one real GPU when the
  component is inherently device-side; fault injection is preferred for host-side CUDA/NCCL error handling.
- `integration/` verifies ownership, asynchronous execution, allocation retirement, graph capture, CUDA libraries,
  and communication across multiple TTL components.
- `system/` verifies complete public workflows across contexts, streams, captured graphs, and devices without using
  private interfaces.
- `support/` contains shared fixtures, independent host references, data conversion, and assertions. Support code
  must not duplicate the implementation under test.

Within a layer, tests are grouped by TTL subsystem: `common`, `tensor`, `ops`, `runtime`, and `distributed`.

## Required dimensions

Every public operation is tested across the dimensions that affect its contract or dispatch:

- allocating and caller-provided `*Out` forms;
- supported and rejected dtypes;
- scalar, empty, singleton, ordinary, large, and maximum-rank shapes where applicable;
- contiguous, strided, broadcast, zero-stride, aligned, and deliberately misaligned layouts where legal;
- exact alias, partial overlap, disjoint storage, cross-device ownership, and moved-from handles;
- synchronous validation errors and asynchronous device errors at their documented observation boundary;
- primary and auxiliary streams, explicit event dependencies, capture/replay, and resource lifetime after producer
  handle destruction;
- one GPU and all required multi-GPU rank/root/peer orderings;
- finite values, signed/unsigned limits, modular overflow, NaN, infinities, ties, and invalid indices.

Tests must use independent expected results. Calling another TTL operation as the sole oracle is not sufficient.

## CTest labels

All tests carry the `ttl` label and one layer label (`contract`, `unit`, `integration`, or `system`). Hardware labels
are `host`, `cuda`, and `multi_gpu`; subsystem labels include `tensor`, `ops`, `runtime`, and `distributed`.

`multi_gpu` suites run serially because they own all selected local devices. A CUDA CI job must treat an unexpected
GoogleTest skip as a configuration failure rather than a passing test.

## Commands

```sh
cmake -S deps/ttl -B build/ttl-tests -G Ninja \
  -DTTL_BUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc
cmake --build build/ttl-tests --target ttl_build_tests
ctest --test-dir build/ttl-tests --output-on-failure -L ttl
```

Coverage gates apply to host code and CUDA host wrappers. gcov source attribution for `.cu` files is not a substitute
for device-path coverage; kernel dispatch matrices and Compute Sanitizer runs are required independently.
