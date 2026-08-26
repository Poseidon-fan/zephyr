#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "common/types.hpp"
#include "planner/buffer.hpp"
#include "planner/instruction.hpp"

namespace zephyr::planner {

/** Reduces a tensor with SUM and returns the result to every rank. */
class AllReduce final : public CloneableInstruction<AllReduce> {
 public:
  AllReduce(communicator_id_t communicator, BufferView input, BufferView output)
      : communicator_(communicator), input_(std::move(input)), output_(std::move(output)) {}

  communicator_id_t communicator_;
  BufferView input_;
  BufferView output_;
};

/** Concatenates equal rank chunks and returns the result to every rank. */
class AllGather final : public CloneableInstruction<AllGather> {
 public:
  AllGather(communicator_id_t communicator, BufferView input, BufferView output)
      : communicator_(communicator), input_(std::move(input)), output_(std::move(output)) {}

  communicator_id_t communicator_;
  BufferView input_;
  BufferView output_;
};

/** Reduces equal rank chunks and returns one chunk to every rank. */
class ReduceScatter final : public CloneableInstruction<ReduceScatter> {
 public:
  ReduceScatter(communicator_id_t communicator, BufferView input, BufferView output)
      : communicator_(communicator), input_(std::move(input)), output_(std::move(output)) {}

  communicator_id_t communicator_;
  BufferView input_;
  BufferView output_;
};

/** Exchanges equal rank-ordered chunks between all ranks. */
class AllToAll final : public CloneableInstruction<AllToAll> {
 public:
  AllToAll(communicator_id_t communicator, BufferView input, BufferView output)
      : communicator_(communicator), input_(std::move(input)), output_(std::move(output)) {}

  communicator_id_t communicator_;
  BufferView input_;
  BufferView output_;
};

/** Exchanges variable-size rows; counts are expressed in logical rows. */
class AllToAllV final : public CloneableInstruction<AllToAllV> {
 public:
  AllToAllV(communicator_id_t communicator, std::vector<BufferView> send_payloads, BufferView send_counts,
            std::vector<BufferView> receive_payloads, BufferView receive_counts)
      : communicator_(communicator),
        send_payloads_(std::move(send_payloads)),
        send_counts_(std::move(send_counts)),
        receive_payloads_(std::move(receive_payloads)),
        receive_counts_(std::move(receive_counts)) {}

  communicator_id_t communicator_;
  std::vector<BufferView> send_payloads_;
  BufferView send_counts_;
  std::vector<BufferView> receive_payloads_;
  BufferView receive_counts_;
};

/** Gathers variable-sized rows along one dimension to a coordinator rank. */
class GatherV final : public CloneableInstruction<GatherV> {
 public:
  GatherV(communicator_id_t communicator, BufferView input, BufferView output, int32_t dimension, rank_t root_rank)
      : communicator_(communicator),
        input_(std::move(input)),
        output_(std::move(output)),
        dimension_(dimension),
        root_rank_(root_rank) {}

  communicator_id_t communicator_;
  BufferView input_;
  BufferView output_;
  int32_t dimension_;
  rank_t root_rank_;
};

/** Copies one planned contiguous view to another. */
class MemoryCopy final : public CloneableInstruction<MemoryCopy> {
 public:
  MemoryCopy(BufferView source, BufferView target) : source_(std::move(source)), target_(std::move(target)) {}

  BufferView source_;
  BufferView target_;
};

}  // namespace zephyr::planner
