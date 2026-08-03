#include "ttl/internal/runtime/cuda_api.hpp"

#include <atomic>
#include <exception>

#include <cuda_runtime_api.h>

namespace ttl::internal {
namespace {

constinit const CudaApi CUDA_API{
    .get_device_count_ = cudaGetDeviceCount,
    .get_device_properties_ = cudaGetDeviceProperties,
    .get_device_ = cudaGetDevice,
    .set_device_ = cudaSetDevice,
    .get_last_error_ = cudaGetLastError,
    .get_stream_priority_range_ = cudaDeviceGetStreamPriorityRange,
    .create_stream_with_priority_ = cudaStreamCreateWithPriority,
    .destroy_stream_ = cudaStreamDestroy,
    .create_event_with_flags_ = cudaEventCreateWithFlags,
    .record_event_ = cudaEventRecord,
    .record_event_with_flags_ = cudaEventRecordWithFlags,
    .query_event_ = cudaEventQuery,
    .synchronize_event_ = cudaEventSynchronize,
    .destroy_event_ = cudaEventDestroy,
    .stream_wait_event_ = cudaStreamWaitEvent,
    .synchronize_stream_ = cudaStreamSynchronize,
    .begin_stream_capture_ = cudaStreamBeginCapture,
    .end_stream_capture_ = cudaStreamEndCapture,
    .is_stream_capturing_ = cudaStreamIsCapturing,
    .get_graph_nodes_ = cudaGraphGetNodes,
    .instantiate_graph_ = cudaGraphInstantiateWithFlags,
    .launch_graph_ = cudaGraphLaunch,
    .destroy_graph_executable_ = cudaGraphExecDestroy,
    .destroy_graph_ = cudaGraphDestroy,
    .debug_graph_dot_print_ = cudaGraphDebugDotPrint,
    .create_memory_pool_ = cudaMemPoolCreate,
    .destroy_memory_pool_ = cudaMemPoolDestroy,
    .set_memory_pool_attribute_ = cudaMemPoolSetAttribute,
    .get_memory_pool_attribute_ = cudaMemPoolGetAttribute,
    .set_memory_pool_access_ = cudaMemPoolSetAccess,
    .trim_memory_pool_ = cudaMemPoolTrimTo,
    .malloc_from_pool_async_ = static_cast<CudaApi::MallocFromPoolAsync>(cudaMallocFromPoolAsync),
    .free_async_ = cudaFreeAsync,
    .memset_async_ = cudaMemsetAsync,
    .memcpy_async_ = cudaMemcpyAsync,
    .memcpy_peer_async_ = cudaMemcpyPeerAsync,
    .host_alloc_ = static_cast<CudaApi::HostAlloc>(cudaHostAlloc),
    .free_host_ = cudaFreeHost,
    .get_memory_info_ = cudaMemGetInfo,
    .get_pointer_attributes_ = cudaPointerGetAttributes,
    .can_access_peer_ = cudaDeviceCanAccessPeer,
};

constinit std::atomic<const CudaApi *> active_cuda_api{&CUDA_API};

}  // namespace

auto GetCudaApi() noexcept -> const CudaApi & { return *active_cuda_api.load(std::memory_order_acquire); }

ScopedCudaApiOverride::ScopedCudaApiOverride(const CudaApi &cuda_api) noexcept
    : cuda_api_(&cuda_api), previous_cuda_api_(active_cuda_api.exchange(cuda_api_, std::memory_order_acq_rel)) {}

ScopedCudaApiOverride::~ScopedCudaApiOverride() noexcept {
  if (active_cuda_api.exchange(previous_cuda_api_, std::memory_order_acq_rel) != cuda_api_) {
    std::terminate();
  }
}

}  // namespace ttl::internal
