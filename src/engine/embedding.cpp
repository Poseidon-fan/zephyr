#include "engine/engine.hpp"

#include <span>
#include <utility>

#include <ttl/ops/cast.hpp>
#include <ttl/ops/copy.hpp>
#include <ttl/tensor/shape.hpp>

#include "common/macros.hpp"

namespace zephyr::engine {

void Engine::ExecuteEmbedding(const scheduler::ScheduledBatch &batch) {
  executor::EmbeddingBatch input;
  input.token_ids_.reserve(batch.sequences_.size());
  for (const auto &scheduled : batch.sequences_) {
    input.token_ids_.push_back(sequences_.at(scheduled.sequence_id_).token_ids_);
  }
  auto result = ExecuteBatch(input, batch);
  if (!result.has_value()) {
    return;
  }
  const auto &embeddings = std::get<executor::EmbeddingResult>(*result).embeddings_;
  ZEPHYR_ENSURE(embeddings.size() == batch.sequences_.size(), "Execution must preserve the scheduled row count");
  for (size_t index = 0; index < embeddings.size(); ++index) {
    const auto id = batch.sequences_[index].sequence_id_;
    auto &sequence = sequences_.at(id);
    auto &request = requests_.at(sequence_contexts_.at(id).request_id_);
    const auto &embedding = embeddings[index];
    const auto contiguous = embedding.GetDType() == ttl::DType::FLOAT32
                                ? ttl::Contiguous(*context_, embedding)
                                : ttl::Cast(*context_, embedding, ttl::DType::FLOAT32);
    ZEPHYR_ENSURE(contiguous.GetShape() == ttl::Shape{embedding_spec_->embedding_size_},
                  "Embedding execution must return one complete vector per input");
    std::vector<float> values(static_cast<size_t>(contiguous.GetNumElements()));
    ttl::CopyToHostBlocking(*context_, std::as_writable_bytes(std::span{values}), contiguous);
    request.output_.result_ = std::move(values);
    sequence.state_ = scheduler::SequenceState::FINISHED;
  }
}

}  // namespace zephyr::engine
