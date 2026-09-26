#include "engine/engine.hpp"

#include <algorithm>
#include <limits>
#include <span>
#include <utility>

#include "common/exception.hpp"
#include "common/logger.hpp"

namespace zephyr::engine {

Engine::RequestState::RequestState(request_id_t id, Request request, size_t num_sequences)
    : request_(std::move(request)),
      remaining_sequences_(num_sequences),
      finish_reasons_(num_sequences),
      output_indices_(num_sequences, std::numeric_limits<size_t>::max()) {
  output_.request_id_ = id;
  usage_.prompt_tokens_ = std::visit([](const auto &input) { return input.token_ids_.size(); }, request_);
  if (std::holds_alternative<EmbeddingRequest>(request_)) {
    output_.result_ = std::vector<float>{};
  }
}

auto Engine::RequestState::GetChoiceOutput(size_t index) -> ChoiceOutput & {
  auto &choices = std::get<std::vector<ChoiceOutput>>(output_.result_);
  auto &position = output_indices_[index];
  if (position == std::numeric_limits<size_t>::max()) {
    choices.push_back(ChoiceOutput{.index_ = index, .tokens_ = {}, .finish_reason_ = std::nullopt});
    position = choices.size() - 1;
  }
  return choices[position];
}

Engine::Engine(ttl::Runtime &runtime, EngineOptions options) : runtime_(runtime), options_(std::move(options)) {
  if (options_.max_outstanding_sequences_ == 0 || options_.max_buffered_output_bytes_ == 0) {
    throw ConfigurationException("Engine admission and output limits must be positive");
  }
}

auto Engine::Create(ttl::Runtime &runtime, EngineOptions options, executor::ExecutionFactory factory)
    -> std::unique_ptr<Engine> {
  auto engine = std::unique_ptr<Engine>{new Engine{runtime, std::move(options)}};
  engine->control_thread_ =
      std::jthread{[owner = engine.get(), factory = std::move(factory)]() mutable { owner->Run(std::move(factory)); }};
  std::unique_lock lock{engine->latch_};
  engine->changed_.wait(lock, [&] { return engine->initialized_ || engine->stopped_; });
  if (!engine->initialized_) {
    std::rethrow_exception(engine->failure_);
  }
  return engine;
}

Engine::~Engine() noexcept {
  try {
    Close();
  } catch (const std::exception &error) {
    ZEPHYR_LOG_ERROR("engine stopped with an error: {}", error.what());
  } catch (...) {
    ZEPHYR_LOG_ERROR("engine stopped with an unknown error");
  }
}

void Engine::Initialize(executor::ExecutionFactory factory) {
  executor_ = executor::Executor::Create(runtime_, options_.executor_, std::move(factory));
  const auto &spec = executor_->GetSpec();
  generation_spec_ = dynamic_cast<const executor::CausalLMExecutionSpec *>(&spec);
  embedding_spec_ = dynamic_cast<const executor::EmbeddingExecutionSpec *>(&spec);
  if (generation_spec_ == nullptr && embedding_spec_ == nullptr) {
    throw ConfigurationException("Engine requires generation or embedding execution capabilities");
  }
  if (generation_spec_ != nullptr) {
    for (const auto token : options_.eos_token_ids_) {
      if (token < 0 || token >= generation_spec_->vocab_size_) {
        throw ConfigurationException("Engine EOS IDs must belong to the model vocabulary");
      }
    }
    if (generation_spec_->cache_capacity_.has_value()) {
      cache_manager_ = std::make_unique<kv_cache::KVCacheManager>(*generation_spec_->cache_capacity_);
    }
  }
  scheduler_ = scheduler::Scheduler::Create(options_.scheduler_, sequences_, cache_manager_.get(),
                                            generation_spec_ != nullptr && generation_spec_->supports_packed_prefill_);
  context_.emplace(runtime_.CreateExecutionContext(spec.device_));
}

auto Engine::Submit(Request request) -> request_id_t {
  const auto *generation = std::get_if<GenerationRequest>(&request);
  if ((generation != nullptr) != (generation_spec_ != nullptr)) {
    throw InvalidArgumentException("Request task does not match the loaded execution capabilities");
  }
  const auto max_length = generation_spec_ != nullptr ? generation_spec_->max_seq_len_ : embedding_spec_->max_seq_len_;
  const auto vocabulary = generation_spec_ != nullptr ? generation_spec_->vocab_size_ : embedding_spec_->vocab_size_;
  const auto tokens =
      std::visit([](const auto &input) { return std::span<const token_id_t>{input.token_ids_}; }, request);
  if (tokens.empty() || std::cmp_greater(tokens.size(), max_length)) {
    throw InvalidArgumentException("Request must contain a nonempty prompt within the model context limit");
  }
  for (const auto token : tokens) {
    if (token < 0 || token >= vocabulary) {
      throw InvalidArgumentException("Request token IDs must belong to the model vocabulary");
    }
  }
  const auto num_sequences = generation != nullptr ? generation->num_choices_ : size_t{1};
  if (num_sequences == 0 || num_sequences > options_.max_outstanding_sequences_) {
    throw InvalidArgumentException("Request choice count exceeds the engine admission limit");
  }
  if (generation != nullptr) {
    sampler::ValidateSamplingParams(generation->sampling_, vocabulary);
    if (generation->max_new_tokens_ == 0) {
      throw InvalidArgumentException("Maximum new tokens must be positive when specified");
    }
    for (const auto token : generation->stop_token_ids_) {
      if (token < 0 || token >= vocabulary) {
        throw InvalidArgumentException("Stop token IDs must belong to the model vocabulary");
      }
    }
  }
  const std::scoped_lock lock{latch_};
  if (closing_ || stopped_) {
    throw InvalidArgumentException("Cannot submit to a closed or failed engine");
  }
  if (num_sequences > options_.max_outstanding_sequences_ - outstanding_sequences_) {
    throw OutOfMemoryException("Engine admission limit reached; consume outstanding outputs before submitting more");
  }
  if (next_request_id_ == std::numeric_limits<request_id_t>::max()) {
    throw InternalException("Engine request identifiers exhausted");
  }
  const auto id = next_request_id_;
  outstanding_.emplace(id, OutstandingRequest{.num_sequences_ = num_sequences, .prompt_tokens_ = tokens.size()});
  try {
    pending_.emplace_back(id, std::move(request));
  } catch (...) {
    outstanding_.erase(id);
    throw;
  }
  ++next_request_id_;
  outstanding_sequences_ += num_sequences;
  changed_.notify_all();
  return id;
}

void Engine::Cancel(request_id_t request_id) {
  const std::scoped_lock lock{latch_};
  const auto found = outstanding_.find(request_id);
  if (found != outstanding_.end() && !found->second.finished_) {
    cancellations_.insert(request_id);
    changed_.notify_all();
  }
}

auto Engine::WaitForOutputs() -> std::vector<RequestOutput> {
  std::unique_lock lock{latch_};
  changed_.wait(lock, [&] { return !outputs_.empty() || stopped_; });
  if (outputs_.empty() && failure_ != nullptr) {
    std::rethrow_exception(failure_);
  }
  for (const auto &output : outputs_) {
    if (output.status_.has_value()) {
      outstanding_sequences_ -= outstanding_.at(output.request_id_).num_sequences_;
      outstanding_.erase(output.request_id_);
      cancellations_.erase(output.request_id_);
    }
  }
  auto outputs = std::exchange(outputs_, {});
  buffered_output_bytes_ = 0;
  lock.unlock();
  changed_.notify_all();
  return outputs;
}

void Engine::PublishOutputs() {
  const std::scoped_lock lock{latch_};
  for (auto it = requests_.begin(); it != requests_.end();) {
    auto &request = it->second;
    auto &output = request.output_;
    if (!output.status_.has_value() && std::visit([](const auto &values) { return values.empty(); }, output.result_)) {
      ++it;
      continue;
    }
    size_t bytes = sizeof(RequestOutput) + output.error_message_.capacity();
    if (const auto *choices = std::get_if<std::vector<ChoiceOutput>>(&output.result_); choices != nullptr) {
      bytes += choices->capacity() * sizeof(ChoiceOutput);
      for (const auto &choice : *choices) {
        bytes += choice.tokens_.capacity() * sizeof(sampler::SamplingResult);
        for (const auto &token : choice.tokens_) {
          if (token.logprobs_.has_value()) {
            bytes += token.logprobs_->top_logprobs_.capacity() * sizeof(sampler::TokenLogprob);
          }
        }
      }
    } else {
      bytes += std::get<std::vector<float>>(output.result_).capacity() * sizeof(float);
    }
    outputs_.push_back(std::move(output));
    buffered_output_bytes_ += bytes;
    if (outputs_.back().status_.has_value()) {
      outstanding_.at(it->first).finished_ = true;
      it = requests_.erase(it);
    } else {
      for (const auto &choice : std::get<std::vector<ChoiceOutput>>(outputs_.back().result_)) {
        request.output_indices_[choice.index_] = std::numeric_limits<size_t>::max();
      }
      output = RequestOutput{.request_id_ = it->first,
                             .result_ = std::vector<ChoiceOutput>{},
                             .status_ = std::nullopt,
                             .usage_ = {},
                             .error_message_ = {}};
      ++it;
    }
  }
  changed_.notify_all();
}

void Engine::FinishRequest(RequestState &request) {
  request.output_.status_ = request.status_;
  request.output_.usage_ = request.usage_;
  request.output_.error_message_ = request.error_message_;
}

auto Engine::ProcessRequests() -> bool {
  std::deque<std::pair<request_id_t, Request>> pending;
  std::unordered_set<request_id_t> cancellations;
  bool closing;
  {
    const std::scoped_lock lock{latch_};
    pending.swap(pending_);
    cancellations.swap(cancellations_);
    closing = closing_;
  }
  if (pending.empty() && cancellations.empty() && !closing) {
    return false;
  }
  for (auto &[request_id, input] : pending) {
    const auto *generation = std::get_if<GenerationRequest>(&input);
    const auto num_sequences = generation != nullptr ? generation->num_choices_ : size_t{1};
    auto &request = requests_.try_emplace(request_id, request_id, std::move(input), num_sequences).first->second;
    if (closing || cancellations.contains(request_id)) {
      request.status_ = RequestStatus::CANCELED;
      if (std::holds_alternative<GenerationRequest>(request.request_)) {
        for (size_t index = 0; index < num_sequences; ++index) {
          request.GetChoiceOutput(index).finish_reason_ = FinishReason::CANCELED;
        }
      }
      FinishRequest(request);
      continue;
    }
    auto &tokens =
        std::visit([](auto &value) -> std::vector<token_id_t> & { return value.token_ids_; }, request.request_);
    generation = std::get_if<GenerationRequest>(&request.request_);
    for (size_t index = 0; index < num_sequences; ++index) {
      if (next_sequence_id_ == std::numeric_limits<sequence_id_t>::max()) {
        throw InternalException("Engine sequence identifiers exhausted");
      }
      const auto id = next_sequence_id_++;
      auto history = index + 1 == num_sequences ? std::exchange(tokens, {}) : tokens;
      const auto step = generation != nullptr ? scheduler::SequenceStepType::PROMPT_AND_DECODE
                                              : scheduler::SequenceStepType::ONE_SHOT;
      sequences_.try_emplace(id, id, std::move(history), step);
      auto &state =
          sequence_contexts_
              .emplace(id, SequenceContext{.request_id_ = request_id, .choice_index_ = index, .rng_ = std::nullopt})
              .first->second;
      if (generation != nullptr) {
        state.rng_.emplace(generation->seed_.has_value() ? *generation->seed_ + static_cast<uint64_t>(index) : rng_());
      }
      scheduler_->Add(id);
    }
  }
  for (auto &[id, state] : sequence_contexts_) {
    if (closing || cancellations.contains(state.request_id_)) {
      auto &sequence = sequences_.at(id);
      auto &request = requests_.at(state.request_id_);
      request.canceled_ = true;
      if (!sequence.IsTerminal()) {
        sequence.state_ = scheduler::SequenceState::FINISHED;
      }
    }
  }
  RetireSequences();
  PublishOutputs();
  return closing;
}

void Engine::RetireSequences() {
  for (auto it = sequences_.begin(); it != sequences_.end();) {
    auto &sequence = it->second;
    if (!sequence.IsTerminal()) {
      ++it;
      continue;
    }
    const auto &state = sequence_contexts_.at(it->first);
    auto &request = requests_.at(state.request_id_);
    auto &finish_reason = request.finish_reasons_[state.choice_index_];
    const auto choice_finished = finish_reason.has_value();
    scheduler_->Remove(it->first);
    if (sequence.state_ == scheduler::SequenceState::ERROR || sequence.state_ == scheduler::SequenceState::REJECTED) {
      if (request.status_ != RequestStatus::ERROR) {
        request.status_ =
            sequence.state_ == scheduler::SequenceState::ERROR ? RequestStatus::ERROR : RequestStatus::REJECTED;
      }
      if (request.error_message_.empty()) {
        request.error_message_ = sequence.error_message_;
      }
      finish_reason = FinishReason::ERROR;
    } else if (!choice_finished && request.canceled_) {
      finish_reason = FinishReason::CANCELED;
      if (request.status_ == RequestStatus::COMPLETED) {
        request.status_ = RequestStatus::CANCELED;
      }
    }
    if (!choice_finished && std::holds_alternative<GenerationRequest>(request.request_)) {
      request.GetChoiceOutput(state.choice_index_).finish_reason_ = finish_reason;
    }
    --request.remaining_sequences_;
    if (request.remaining_sequences_ == 0) {
      FinishRequest(request);
    }
    sequence_contexts_.erase(it->first);
    it = sequences_.erase(it);
  }
}

auto Engine::ExecuteBatch(const executor::ExecutionBatch &input, const scheduler::ScheduledBatch &batch)
    -> std::optional<executor::ExecutionResult> {
  try {
    return executor_->Execute(input);
  } catch (const std::exception &error) {
    if (executor_->IsFailed()) {
      throw;
    }
    FailBatch(batch, error.what());
    return std::nullopt;
  }
}

void Engine::FailBatch(const scheduler::ScheduledBatch &batch, const std::string &message) {
  for (const auto &scheduled : batch.sequences_) {
    auto &sequence = sequences_.at(scheduled.sequence_id_);
    sequence.state_ = scheduler::SequenceState::ERROR;
    sequence.error_message_ = message;
  }
}

void Engine::FailAllRequests(const std::string &message) {
  const std::scoped_lock lock{latch_};
  // Include accepted submissions not yet registered, and preserve already published choice endings.
  for (auto &[id, admission] : outstanding_) {
    if (admission.finished_) {
      continue;
    }
    RequestOutput output{.request_id_ = id,
                         .result_ = std::vector<ChoiceOutput>{},
                         .status_ = RequestStatus::ERROR,
                         .usage_ = {.prompt_tokens_ = admission.prompt_tokens_},
                         .error_message_ = message};
    const auto found = requests_.find(id);
    if (found != requests_.end()) {
      auto &request = found->second;
      if (generation_spec_ != nullptr) {
        for (size_t index = 0; index < admission.num_sequences_; ++index) {
          if (!request.finish_reasons_[index].has_value()) {
            request.GetChoiceOutput(index).finish_reason_ = FinishReason::ERROR;
          }
        }
      }
      request.status_ = RequestStatus::ERROR;
      request.error_message_ = message;
      FinishRequest(request);
      output = std::move(request.output_);
    } else if (generation_spec_ != nullptr) {
      auto &choices = std::get<std::vector<ChoiceOutput>>(output.result_);
      choices.reserve(admission.num_sequences_);
      for (size_t index = 0; index < admission.num_sequences_; ++index) {
        choices.push_back({.index_ = index, .tokens_ = {}, .finish_reason_ = FinishReason::ERROR});
      }
    } else {
      output.result_ = std::vector<float>{};
    }
    outputs_.push_back(std::move(output));
    admission.finished_ = true;
  }
  pending_.clear();
}

void Engine::Run(executor::ExecutionFactory factory) noexcept {
  try {
    Initialize(std::move(factory));
    {
      const std::scoped_lock lock{latch_};
      initialized_ = true;
    }
    changed_.notify_all();
    while (true) {
      {
        std::unique_lock lock{latch_};
        changed_.wait(lock, [&] {
          return closing_ || !pending_.empty() || !cancellations_.empty() ||
                 (!sequences_.empty() && buffered_output_bytes_ < options_.max_buffered_output_bytes_);
        });
      }
      if (ProcessRequests()) {
        break;
      }
      {
        const std::scoped_lock lock{latch_};
        if (closing_ || buffered_output_bytes_ >= options_.max_buffered_output_bytes_) {
          continue;
        }
      }
      const auto plan = scheduler_->Schedule();
      for (const auto &batch : plan.batches_) {
        if (generation_spec_ != nullptr) {
          ExecuteGeneration(batch);
        } else {
          ExecuteEmbedding(batch);
        }
      }
      // Schedule can reject requests without returning a batch; those terminal states still need delivery.
      RetireSequences();
      PublishOutputs();
    }
  } catch (...) {
    failure_ = std::current_exception();
    const std::scoped_lock lock{latch_};
    closing_ = true;
  }

  // GPU users must stop before logical pages are discarded, including failures outside the executor (sampling/copies).
  try {
    if (context_.has_value()) {
      context_->Synchronize();
    }
  } catch (...) {
    if (failure_ == nullptr) {
      failure_ = std::current_exception();
    }
  }
  try {
    if (executor_ != nullptr) {
      executor_->Close();
    }
  } catch (...) {
    if (failure_ == nullptr) {
      failure_ = std::current_exception();
    }
  }
  if (failure_ != nullptr) {
    try {
      try {
        std::rethrow_exception(failure_);
      } catch (const std::exception &error) {
        FailAllRequests(error.what());
      } catch (...) {
        FailAllRequests("Engine execution failed with an unknown error");
      }
    } catch (...) {
      ZEPHYR_LOG_ERROR("engine could not allocate failure outputs");
    }
  }
  scheduler_.reset();
  cache_manager_.reset();
  sequence_contexts_.clear();
  requests_.clear();
  sequences_.clear();
  context_.reset();
  {
    const std::scoped_lock lock{latch_};
    stopped_ = true;
  }
  changed_.notify_all();
}

void Engine::Close() {
  std::call_once(close_once_, [&] {
    {
      const std::scoped_lock lock{latch_};
      closing_ = true;
    }
    changed_.notify_all();
    if (control_thread_.joinable()) {
      control_thread_.join();
    }
  });
  if (failure_ != nullptr) {
    std::rethrow_exception(failure_);
  }
}

}  // namespace zephyr::engine
