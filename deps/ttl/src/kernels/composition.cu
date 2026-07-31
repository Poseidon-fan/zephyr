#include "ttl/internal/composition.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/cuda_dtype.hpp"
#include "ttl/internal/index_width.hpp"
#include "ttl/shape.hpp"

namespace ttl::internal {
namespace {

constexpr uint32_t COMPOSITION_THREADS_PER_BLOCK = 256;
constexpr uint32_t MAXIMUM_COMPOSITION_BLOCKS = 65535;

template <typename Destination>
[[nodiscard]] auto ConvertParameters(const CompositionParameters64 &source, std::source_location location)
    -> Destination {
  using Index = std::remove_cvref_t<decltype(Destination::num_elements_)>;
  Destination destination{
      .output_ = source.output_,
      .input_ = source.input_,
      .num_elements_ = CheckedNarrow<Index>(source.num_elements_, "composition element count", location),
      .rank_ = source.rank_,
  };
  for (size_t axis = 0; axis < TTL_MAX_RANK; ++axis) {
    destination.shape_[axis] = CheckedNarrow<Index>(source.shape_[axis], "composition shape", location);
    destination.output_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.output_strides_bytes_[axis], "composition output stride", location);
    destination.input_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.input_strides_bytes_[axis], "composition input stride", location);
  }
  return destination;
}

template <typename Index>
[[nodiscard]] auto GetBlockCount(Index num_elements) noexcept -> uint32_t {
  const auto blocks =
      (static_cast<uint64_t>(num_elements) + COMPOSITION_THREADS_PER_BLOCK - 1) / COMPOSITION_THREADS_PER_BLOCK;
  return static_cast<uint32_t>(blocks < MAXIMUM_COMPOSITION_BLOCKS ? blocks : MAXIMUM_COMPOSITION_BLOCKS);
}

template <CudaStorageType T, typename Parameters>
__global__ void CompositionCopyKernel(Parameters parameters) {
  using Index = std::remove_cvref_t<decltype(parameters.num_elements_)>;
  auto linear_index =
      (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  const auto step = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);
  while (linear_index < parameters.num_elements_) {
    auto remaining = linear_index;
    auto output_offset = Index{0};
    auto input_offset = Index{0};
    for (size_t remaining_rank = parameters.rank_; remaining_rank > 0; --remaining_rank) {
      const auto axis = remaining_rank - 1;
      const auto coordinate = static_cast<Index>(remaining % parameters.shape_[axis]);
      remaining = static_cast<Index>(remaining / parameters.shape_[axis]);
      output_offset = static_cast<Index>(output_offset + (coordinate * parameters.output_strides_bytes_[axis]));
      input_offset = static_cast<Index>(input_offset + (coordinate * parameters.input_strides_bytes_[axis]));
    }
    *reinterpret_cast<T *>(parameters.output_ + output_offset) =
        *reinterpret_cast<const T *>(parameters.input_ + input_offset);
    if (step >= parameters.num_elements_ - linear_index) {
      break;
    }
    linear_index = static_cast<Index>(linear_index + step);
  }
}

template <CudaStorageType T, typename Parameters>
void LaunchTyped(cudaStream_t stream, const Parameters &parameters) {
  CompositionCopyKernel<T>
      <<<GetBlockCount(parameters.num_elements_), COMPOSITION_THREADS_PER_BLOCK, 0, stream>>>(parameters);
}

}  // namespace

auto GetCompositionIndexWidth(const CompositionParameters64 &parameters) noexcept -> IndexWidth {
  constexpr auto maximum = uint64_t{std::numeric_limits<uint32_t>::max()};
  if (parameters.num_elements_ > maximum) {
    return IndexWidth::UINT64;
  }
  auto output_offset = uint64_t{0};
  auto input_offset = uint64_t{0};
  for (size_t axis = 0; axis < parameters.rank_; ++axis) {
    if (parameters.shape_[axis] > maximum || parameters.output_strides_bytes_[axis] > maximum ||
        parameters.input_strides_bytes_[axis] > maximum) {
      return IndexWidth::UINT64;
    }
    const auto extent = parameters.shape_[axis] == 0 ? uint64_t{0} : parameters.shape_[axis] - 1;
    if ((parameters.output_strides_bytes_[axis] != 0 &&
         extent > (maximum - output_offset) / parameters.output_strides_bytes_[axis]) ||
        (parameters.input_strides_bytes_[axis] != 0 &&
         extent > (maximum - input_offset) / parameters.input_strides_bytes_[axis])) {
      return IndexWidth::UINT64;
    }
    output_offset += extent * parameters.output_strides_bytes_[axis];
    input_offset += extent * parameters.input_strides_bytes_[axis];
  }
  return IndexWidth::UINT32;
}

void LaunchCompositionCopy(cudaStream_t stream, DType dtype, IndexWidth index_width,
                           const CompositionParameters64 &parameters, std::source_location location) {
  if (stream == nullptr || parameters.output_ == nullptr || parameters.input_ == nullptr ||
      parameters.num_elements_ == 0) {
    throw InternalError("invalid composition launch parameters", location);
  }
  DispatchCudaDType(dtype, "CompositionOut", [&]<CudaStorageType T>(std::type_identity<T>) {
    if (index_width == IndexWidth::UINT32) {
      LaunchTyped<T>(stream, ConvertParameters<CompositionParameters32>(parameters, location));
    } else {
      LaunchTyped<T>(stream, parameters);
    }
  });
}

}  // namespace ttl::internal
