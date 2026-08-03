#pragma once

#include <cstdint>
#include <source_location>
#include <utility>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

struct TopKOptions final {
  int64_t axis_{-1};
  int64_t k_{0};
  bool largest_{true};
  bool sorted_{true};
};

void TopKOut(ExecutionContext &context, Tensor &values, Tensor &indices, const Tensor &input,
             const TopKOptions &options, std::source_location location = std::source_location::current());
[[nodiscard]] auto TopK(ExecutionContext &context, const Tensor &input, const TopKOptions &options,
                        std::source_location location = std::source_location::current()) -> std::pair<Tensor, Tensor>;

}  // namespace ttl
