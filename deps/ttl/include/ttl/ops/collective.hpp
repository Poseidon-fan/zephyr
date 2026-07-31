#pragma once

#include <cstdint>
#include <source_location>
#include <span>

#include "ttl/communicator.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/tensor.hpp"

namespace ttl {

enum class ReduceOp : uint8_t {
  SUM,
  MINIMUM,
  MAXIMUM,
};

/** One rank in a single-thread process-local collective submission. */
struct LocalCollectiveCall final {
  ExecutionContext *context_;
  Tensor *output_;
  const Tensor *input_;
  NcclCommunicator *communicator_;
};

/** Optional send and receive issued by one rank inside one process-local NCCL group. */
struct LocalPointToPointCall final {
  ExecutionContext *context_;
  const Tensor *send_;
  int32_t send_peer_;
  Tensor *receive_;
  int32_t receive_peer_;
  NcclCommunicator *communicator_;
};

/** One rank in a single-thread process-local barrier submission. */
struct LocalBarrierCall final {
  ExecutionContext *context_;
  NcclCommunicator *communicator_;
};

/**
 * Rank-local asynchronous collective operators.
 *
 * Every rank must submit the same collective sequence from a distinct host thread. Inputs may use any legal strided
 * layout and outputs must be non-overlapping dense; TTL materializes them around NCCL as needed. A successful return
 * means NCCL has enqueued the operation, not that GPU work has completed. Use the Local variants below when one host
 * thread owns every local rank.
 */
/** Elementwise reduction to every rank; input and output have equal shape and dtype. */
void AllReduceOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                  ReduceOp operation, std::source_location location = std::source_location::current());
/** Elementwise reduction to root; non-root output contents are unspecified. */
void ReduceOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
               ReduceOp operation, int32_t root, std::source_location location = std::source_location::current());
/** Concatenate equal input chunks in rank order into every output. */
void AllGatherOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                  std::source_location location = std::source_location::current());
/** Reduce equal rank-ordered input chunks and return one chunk per rank. */
void ReduceScatterOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                      ReduceOp operation, std::source_location location = std::source_location::current());
/** Copy root input to every output; non-root input contents are ignored. */
void BroadcastOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                  int32_t root, std::source_location location = std::source_location::current());
/** Exchange equal rank-ordered chunks between every pair of ranks. */
void AllToAllOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                 std::source_location location = std::source_location::current());
/** Concatenate equal input chunks at root; non-root output contents are unspecified. */
void GatherOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
               int32_t root, std::source_location location = std::source_location::current());
/** Distribute equal rank-ordered root input chunks; non-root input contents are ignored. */
void ScatterOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                int32_t root, std::source_location location = std::source_location::current());
/** Enqueue one contiguous point-to-point send. */
void Send(ExecutionContext &context, const Tensor &input, NcclCommunicator &communicator, int32_t peer,
          std::source_location location = std::source_location::current());
/** Enqueue one contiguous point-to-point receive. */
void ReceiveOut(ExecutionContext &context, Tensor &output, NcclCommunicator &communicator, int32_t peer,
                std::source_location location = std::source_location::current());
/** Enqueue a send and receive together in one NCCL group to avoid bidirectional ordering deadlock. */
void SendReceiveOut(ExecutionContext &context, const Tensor &send, int32_t send_peer, Tensor &receive,
                    int32_t receive_peer, NcclCommunicator &communicator,
                    std::source_location location = std::source_location::current());
/** Establish an NCCL execution barrier without synchronizing a host thread or CUDA device. */
void Barrier(ExecutionContext &context, NcclCommunicator &communicator,
             std::source_location location = std::source_location::current());

/**
 * Single-host-thread submissions containing exactly one rank-ordered call for every communicator rank.
 *
 * TTL validates all local signatures and completes every NCCL group before translating native errors to exceptions.
 */
void AllReduceLocal(std::span<const LocalCollectiveCall> calls, ReduceOp operation,
                    std::source_location location = std::source_location::current());
void ReduceLocal(std::span<const LocalCollectiveCall> calls, ReduceOp operation, int32_t root,
                 std::source_location location = std::source_location::current());
void AllGatherLocal(std::span<const LocalCollectiveCall> calls,
                    std::source_location location = std::source_location::current());
void ReduceScatterLocal(std::span<const LocalCollectiveCall> calls, ReduceOp operation,
                        std::source_location location = std::source_location::current());
void BroadcastLocal(std::span<const LocalCollectiveCall> calls, int32_t root,
                    std::source_location location = std::source_location::current());
void AllToAllLocal(std::span<const LocalCollectiveCall> calls,
                   std::source_location location = std::source_location::current());
void GatherLocal(std::span<const LocalCollectiveCall> calls, int32_t root,
                 std::source_location location = std::source_location::current());
void ScatterLocal(std::span<const LocalCollectiveCall> calls, int32_t root,
                  std::source_location location = std::source_location::current());
void SendReceiveLocal(std::span<const LocalPointToPointCall> calls,
                      std::source_location location = std::source_location::current());
void BarrierLocal(std::span<const LocalBarrierCall> calls,
                  std::source_location location = std::source_location::current());

}  // namespace ttl
