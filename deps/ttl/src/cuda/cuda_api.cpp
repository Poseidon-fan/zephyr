#include "ttl/internal/cuda_api.hpp"

#include <cuda_runtime_api.h>

namespace ttl::internal {
namespace {

constinit const CudaApi CUDA_API{
    .get_device_ = cudaGetDevice,
    .set_device_ = cudaSetDevice,
};

}  // namespace

auto GetCudaApi() noexcept -> const CudaApi & { return CUDA_API; }

}  // namespace ttl::internal
