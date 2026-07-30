#pragma once

#include <cstdint>
#include <source_location>
#include <vector>

#include "ttl/execution_context.hpp"
#include "ttl/tensor.hpp"

namespace ttl {

struct ReductionOptions final {
  std::vector<int64_t> axes_;
  bool keep_dimensions_{false};
};

void SumOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
            std::source_location location = std::source_location::current());
[[nodiscard]] auto Sum(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                       std::source_location location = std::source_location::current()) -> Tensor;

void MeanOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
             std::source_location location = std::source_location::current());
[[nodiscard]] auto Mean(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                        std::source_location location = std::source_location::current()) -> Tensor;

void MinimumOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
                std::source_location location = std::source_location::current());
[[nodiscard]] auto Minimum(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                           std::source_location location = std::source_location::current()) -> Tensor;

void MaximumOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
                std::source_location location = std::source_location::current());
[[nodiscard]] auto Maximum(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                           std::source_location location = std::source_location::current()) -> Tensor;

void ArgMinOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis,
               bool keep_dimension = false, std::source_location location = std::source_location::current());
[[nodiscard]] auto ArgMin(ExecutionContext &context, const Tensor &input, int64_t axis, bool keep_dimension = false,
                          std::source_location location = std::source_location::current()) -> Tensor;

void ArgMaxOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis,
               bool keep_dimension = false, std::source_location location = std::source_location::current());
[[nodiscard]] auto ArgMax(ExecutionContext &context, const Tensor &input, int64_t axis, bool keep_dimension = false,
                          std::source_location location = std::source_location::current()) -> Tensor;

void AnyOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
            std::source_location location = std::source_location::current());
[[nodiscard]] auto Any(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                       std::source_location location = std::source_location::current()) -> Tensor;

void AllOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
            std::source_location location = std::source_location::current());
[[nodiscard]] auto All(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                       std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
