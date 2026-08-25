#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ttl/common/device.hpp>
#include <ttl/tensor/dtype.hpp>

#include "common/types.h"
#include "planner/buffer.h"
#include "planner/instruction.h"

namespace zephyr::planner {

/** Describes one logical K/V storage entry used by a decoder attention instruction. */
struct KVCacheEntry final {
  /** Stable entry identity referenced by SelfAttention instructions. */
  kv_layer_id_t kv_layer_id_;

  /** Element type of the K/V storage. */
  ttl::DType dtype_;

  /** Number of K/V heads resident in this plan. */
  int64_t kv_head_count_;

  /** Width of one attention head. */
  int64_t head_dimension_;
};

/** Static paged-KV layout for one plan instance. */
struct KVCachePlan final {
  /** Number of tokens stored by one logical block. */
  int32_t block_size_;

  /** Entries in stable layer-id order. */
  std::vector<KVCacheEntry> entries_;
};

/** Names one rank-ordered communicator used by collective instructions. */
struct CommunicatorSpec final {
  /** Stable identifier referenced by collective instructions. */
  communicator_id_t id_;

  /** Global rank order, which also defines each communicator's local rank order. */
  std::vector<rank_t> ranks_;
};

/** Ordered executable actions and the communicators they reference. */
struct ExecutionPlan final {
  /** Instructions are submitted in this exact order. */
  std::vector<std::unique_ptr<Instruction>> instructions_;

  /** Communicator specifications referenced by instructions_. */
  std::vector<CommunicatorSpec> communicators_;
};

/** One checkpoint slice copied into one physical weight buffer slice. */
struct WeightSourcePart final {
  /** Safetensors/checkpoint tensor name. */
  std::string source_name_;

  /** Source tensor slice read in row-major order. */
  TensorSlice source_slice_;

  /** Destination slice written in the target WEIGHT buffer. */
  TensorSlice target_slice_;
};

/** Describes complete initialization of one physical WEIGHT buffer. */
struct WeightTarget final {
  /** Target buffer; it must have kind WEIGHT. */
  buffer_id_t buffer_;

  /** Non-overlapping source parts that completely cover the target. */
  std::vector<WeightSourcePart> sources_;
};

/** All checkpoint-to-device mappings for one plan. */
struct WeightPlan final {
  std::vector<WeightTarget> targets_;
};

/** Connects one logical model input to a Worker-local INPUT view. */
struct InputBinding final {
  /** Zero-based Model input index. */
  size_t input_index_;

  /** Worker-local destination view. */
  BufferView target_;
};

/** Exposes one Worker-local view as a logical model output. */
struct OutputBinding final {
  /** Zero-based Model output index. */
  size_t output_index_;

  /** Worker-local source view. */
  BufferView source_;
};

/** Rank-neutral serial execution template produced before parallel expansion. */
struct TemplatePlan final {
  /** All contiguous buffers in the serial template. */
  std::vector<BufferSpec> buffers_;

  /** Instructions in serial execution order. */
  std::vector<std::unique_ptr<Instruction>> instructions_;

  /** Complete checkpoint-to-buffer mappings. */
  WeightPlan weights_;

  /** Logical KV layout used by the template. */
  KVCachePlan kv_cache_;

  /** Template input bindings. */
  std::vector<InputBinding> inputs_;

  /** Template output bindings. */
  std::vector<OutputBinding> outputs_;
};

/** Complete executable description for one process-wide rank. */
struct WorkerPlan final {
  /** Global process rank. */
  rank_t rank_;

  /** CUDA device assigned to this rank. */
  ttl::Device device_;

  /** Number of TP ranks in each DP replica. */
  int32_t tensor_parallel_size_;

  /** Number of independent DP replicas. */
  int32_t data_parallel_size_;

  /** This Worker's coordinate inside its TP group. */
  int32_t tensor_parallel_rank_;

  /** This Worker's DP replica coordinate. */
  int32_t data_parallel_rank_;

  /** All contiguous device buffers owned by this Worker. */
  std::vector<BufferSpec> buffers_;

  /** Ordered instructions and rank-local communicators. */
  ExecutionPlan execution_;

  /** Weight initialization mapping for this rank. */
  WeightPlan weights_;

  /** Persistent KV layout for this rank. */
  KVCachePlan kv_cache_;

  /** Worker-local model input bindings. */
  std::vector<InputBinding> inputs_;

  /** Worker-local model output bindings. */
  std::vector<OutputBinding> outputs_;
};

}  // namespace zephyr::planner
