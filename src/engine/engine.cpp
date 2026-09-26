#include "engine/engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include <ttl/common/error_sink.hpp>

#include "common/exception.hpp"
#include "common/logger.hpp"
#include "executor/factory.hpp"

namespace zephyr::engine {

/** Retain the first runtime failure without blocking or allocating in a cleanup callback. */
class Engine::RuntimeErrors final : public ttl::ErrorSink {
 public:
  void Report(ttl::ErrorRecord error) noexcept override {
    static_assert(std::is_nothrow_move_assignable_v<ttl::ErrorRecord>);
    if (!claimed_.test_and_set(std::memory_order_relaxed)) {
      error_ = std::move(error);
      ready_.store(true, std::memory_order_release);
    }
  }

  [[nodiscard]] auto GetError() const noexcept -> const ttl::ErrorRecord * {
    return ready_.load(std::memory_order_acquire) ? &error_ : nullptr;
  }

 private:
  std::atomic_flag claimed_;
  std::atomic<bool> ready_{false};
  ttl::ErrorRecord error_{};
};

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

Engine::Engine(EngineOptions options) : options_(std::move(options)) {
  if (options_.max_outstanding_sequences_ == 0 || options_.max_buffered_output_bytes_ == 0) {
    throw ConfigurationException("Engine admission and output limits must be positive");
  }
}

auto Engine::Create(EngineOptions options, std::optional<model::ModelTask> task, executor::CausalLMOptions causal_lm)
    -> std::unique_ptr<Engine> {
  auto loader = model::ResolveModelLoader(options.executor_.model_dir_, task);
  return Create(std::move(options), executor::CreateExecutionFactory(std::move(loader), causal_lm));
}

auto Engine::Create(EngineOptions options, executor::ExecutionFactory factory) -> std::unique_ptr<Engine> {
  auto engine = std::unique_ptr<Engine>{new Engine{std::move(options)}};
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
  const auto started = std::chrono::steady_clock::now();
  ZEPHYR_LOG_INFO("Initializing engine: {} CUDA devices, GPU memory target {:.0f}%", options_.executor_.devices_.size(),
                  options_.executor_.gpu_memory_utilization_ * 100.0);
  runtime_errors_ = std::make_shared<RuntimeErrors>();
  ttl::RuntimeOptions runtime_options;
  runtime_options.devices_ = options_.executor_.devices_;
  runtime_options.error_sink_ = runtime_errors_;
  runtime_ = std::make_unique<ttl::Runtime>(std::move(runtime_options));
  ZEPHYR_LOG_INFO("CUDA runtime initialized; starting model workers");
  auto execution_options = options_.executor_;
  auto &limits = execution_options.execution_limits_;
  limits.max_num_seqs_ =
      std::min(options_.max_outstanding_sequences_,
               std::visit([](const auto &config) { return config.max_num_seqs_; }, options_.scheduler_));
  const auto *paged = std::get_if<scheduler::PagedSchedulerConfig>(&options_.scheduler_);
  limits.max_num_batched_tokens_ = paged != nullptr ? paged->max_num_batched_tokens_ : 0;
  limits.max_num_output_tokens_ = limits.max_num_seqs_;
  executor_ = executor::Executor::Create(*runtime_, execution_options, std::move(factory));
  const auto &spec = executor_->GetSpec();
  generation_spec_ = dynamic_cast<const executor::CausalLMExecutionSpec *>(&spec);
  embedding_spec_ = dynamic_cast<const executor::EmbeddingExecutionSpec *>(&spec);
  if (generation_spec_ == nullptr && embedding_spec_ == nullptr) {
    throw ConfigurationException("Engine requires generation or embedding execution capabilities");
  }
  info_ = {.task_ = generation_spec_ != nullptr ? model::ModelTask::GENERATION : model::ModelTask::EMBEDDING,
           .max_seq_len_ = spec.limits_.max_seq_len_,
           .vocab_size_ = generation_spec_ != nullptr ? generation_spec_->vocab_size_ : embedding_spec_->vocab_size_};
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
  context_.emplace(runtime_->CreateExecutionContext(spec.device_));
  if (const auto *error = runtime_errors_->GetError(); error != nullptr) {
    throw ttl::Error(error->code_, error->message_, error->location_);
  }
  ZEPHYR_LOG_INFO(
      "Engine ready in {:.2f}s: task={}, scheduler={}, max context={}, max batch sequences={}, "
      "max batch tokens={}, admission limit={}, output buffer={} MiB",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
      generation_spec_ != nullptr ? "generation" : "embedding", cache_manager_ != nullptr ? "paged" : "length bucket",
      info_.max_seq_len_, spec.limits_.max_num_seqs_, spec.limits_.max_num_batched_tokens_,
      options_.max_outstanding_sequences_, options_.max_buffered_output_bytes_ / (1024 * 1024));
}

auto Engine::Submit(Request request) -> request_id_t {
  const auto *generation = std::get_if<GenerationRequest>(&request);
  if ((generation != nullptr) != (generation_spec_ != nullptr)) {
    throw InvalidArgumentException("Request task does not match the loaded execution capabilities");
  }
  const auto max_length = info_.max_seq_len_;
  const auto vocabulary = info_.vocab_size_;
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
    throw OverloadedException("Engine admission limit reached; consume outstanding outputs before submitting more");
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

void Engine::StopChoice(request_id_t request_id, size_t choice_index) {
  const std::scoped_lock lock{latch_};
  const auto found = outstanding_.find(request_id);
  if (info_.task_ == model::ModelTask::GENERATION && found != outstanding_.end() && !found->second.finished_ &&
      choice_index < found->second.num_sequences_) {
    choice_stops_[request_id].insert(choice_index);
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
      choice_stops_.erase(output.request_id_);
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
  if (request.status_ == RequestStatus::ERROR) {
    ZEPHYR_LOG_ERROR("Request {} failed: {}", request.output_.request_id_, request.error_message_);
  } else if (request.status_ == RequestStatus::REJECTED) {
    ZEPHYR_LOG_WARN("Request {} rejected: {}", request.output_.request_id_, request.error_message_);
  } else {
    ZEPHYR_LOG_DEBUG("Request {} {}: prompt tokens={}, generated tokens={}", request.output_.request_id_,
                     request.status_ == RequestStatus::COMPLETED ? "completed" : "canceled",
                     request.usage_.prompt_tokens_, request.usage_.completion_tokens_);
  }
}

auto Engine::ProcessRequests() -> bool {
  std::deque<std::pair<request_id_t, Request>> pending;
  std::unordered_set<request_id_t> cancellations;
  std::unordered_map<request_id_t, std::unordered_set<size_t>> choice_stops;
  bool closing;
  {
    const std::scoped_lock lock{latch_};
    pending.swap(pending_);
    cancellations.swap(cancellations_);
    choice_stops.swap(choice_stops_);
    closing = closing_;
  }
  if (pending.empty() && cancellations.empty() && choice_stops.empty() && !closing) {
    return false;
  }
  for (auto &[request_id, input] : pending) {
    const auto *generation = std::get_if<GenerationRequest>(&input);
    const auto num_sequences = generation != nullptr ? generation->num_choices_ : size_t{1};
    auto &request = requests_.try_emplace(request_id, request_id, std::move(input), num_sequences).first->second;
    ZEPHYR_LOG_DEBUG("Registering request {}: {} prompt tokens, {} sequences", request_id,
                     request.usage_.prompt_tokens_, num_sequences);
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
    auto &sequence = sequences_.at(id);
    auto &request = requests_.at(state.request_id_);
    if (closing || cancellations.contains(state.request_id_)) {
      request.canceled_ = true;
      if (!sequence.IsTerminal()) {
        sequence.state_ = scheduler::SequenceState::FINISHED;
      }
    } else if (const auto found = choice_stops.find(state.request_id_);
               found != choice_stops.end() && found->second.contains(state.choice_index_) && !sequence.IsTerminal()) {
      request.finish_reasons_[state.choice_index_] = FinishReason::STOP_STRING;
      request.GetChoiceOutput(state.choice_index_).finish_reason_ = FinishReason::STOP_STRING;
      sequence.state_ = scheduler::SequenceState::FINISHED;
      ZEPHYR_LOG_DEBUG("Request {} choice {} stopped by the text frontend", state.request_id_, state.choice_index_);
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

void Engine::LogRuntimeStatistics(bool force) {
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration<double>(now - log_statistics_.last_log_).count();
  if (elapsed <= 0.0 || (!force && elapsed < 5.0)) {
    return;
  }
  size_t pending_requests;
  size_t outstanding_requests;
  size_t unread_bytes;
  {
    const std::scoped_lock lock{latch_};
    pending_requests = pending_.size();
    outstanding_requests = outstanding_.size();
    unread_bytes = buffered_output_bytes_;
  }
  const auto statistics = scheduler_->GetStatistics();
  const auto active = log_statistics_.input_tokens_ != 0 || log_statistics_.generated_tokens_ != 0 ||
                      statistics.num_running_sequences_ != 0 || statistics.num_waiting_sequences_ != 0 ||
                      outstanding_requests != 0;
  if (active || log_statistics_.was_active_) {
    const auto total_pages = cache_manager_ != nullptr ? cache_manager_->GetNumGpuBlocks() - 1 : 0;
    const auto active_pages = cache_manager_ != nullptr ? total_pages - cache_manager_->GetNumFreeBlocks() : 0;
    const auto unreserved_pages = cache_manager_ != nullptr ? cache_manager_->GetNumUnreservedBlocks() : 0;
    // Free prefix-cache pages are reusable; active pages exclude those pages and the permanent null page.
    ZEPHYR_LOG_INFO(
        "Throughput: {} {:.1f} tokens/s, generation {:.1f} tokens/s; sequences: {} running, {} waiting; "
        "requests: {} pending, {} outstanding; KV: {}/{} pages active, {} unreserved; "
        "unread output: {:.2f} MiB",
        generation_spec_ != nullptr ? "prefill" : "embedding",
        static_cast<double>(log_statistics_.input_tokens_) / elapsed,
        static_cast<double>(log_statistics_.generated_tokens_) / elapsed, statistics.num_running_sequences_,
        statistics.num_waiting_sequences_, pending_requests, outstanding_requests, active_pages, total_pages,
        unreserved_pages, static_cast<double>(unread_bytes) / (1024.0 * 1024.0));
  }
  log_statistics_ = {.last_log_ = now, .was_active_ = active};
}

void Engine::Run(executor::ExecutionFactory factory) noexcept {
  try {
    Initialize(std::move(factory));
    log_statistics_.last_log_ = std::chrono::steady_clock::now();
    {
      const std::scoped_lock lock{latch_};
      initialized_ = true;
    }
    changed_.notify_all();
    bool output_paused = false;
    while (true) {
      bool has_work;
      size_t unread_bytes;
      {
        std::unique_lock lock{latch_};
        // Runtime callbacks cannot take this latch. Periodic polls advance errors and retirement while idle or paused.
        has_work = changed_.wait_for(lock, std::chrono::milliseconds{100}, [&] {
          return closing_ || runtime_errors_->GetError() != nullptr || !pending_.empty() || !cancellations_.empty() ||
                 !choice_stops_.empty() ||
                 (!sequences_.empty() && buffered_output_bytes_ < options_.max_buffered_output_bytes_);
        });
        unread_bytes = buffered_output_bytes_;
      }
      runtime_->Poll();
      if (const auto *error = runtime_errors_->GetError(); error != nullptr) {
        throw ttl::Error(error->code_, error->message_, error->location_);
      }
      const auto paused = unread_bytes >= options_.max_buffered_output_bytes_;
      if (paused != output_paused) {
        if (paused) {
          ZEPHYR_LOG_WARN("Scheduling paused: unread output {} bytes reached the {}-byte watermark", unread_bytes,
                          options_.max_buffered_output_bytes_);
        } else {
          ZEPHYR_LOG_INFO("Output backpressure cleared; scheduling resumed");
        }
        output_paused = paused;
      }
      LogRuntimeStatistics();
      if (!has_work) {
        continue;
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
        // Length buckets have a soft admission threshold. Execute large buckets in bounded slices without changing
        // their selection or order, so queued request count cannot enlarge the profiled GPU working set.
        const auto max_rows = executor_->GetSpec().limits_.max_num_seqs_;
        for (size_t start = 0; start < batch.sequences_.size(); start += max_rows) {
          const auto rows =
              std::span{batch.sequences_}.subspan(start, std::min(max_rows, batch.sequences_.size() - start));
          const scheduler::ScheduledBatch execution_batch{.phase_ = batch.phase_,
                                                          .is_final_prompt_chunk_ = batch.is_final_prompt_chunk_,
                                                          .sequences_ = {rows.begin(), rows.end()}};
          ZEPHYR_LOG_DEBUG("Executing {} batch: {} sequences, final prompt chunk={}",
                           batch.phase_ == scheduler::BatchPhase::PREFILL ? "prefill" : "decode", rows.size(),
                           batch.is_final_prompt_chunk_);
          if (generation_spec_ != nullptr) {
            ExecuteGeneration(execution_batch);
          } else {
            ExecuteEmbedding(execution_batch);
          }
        }
      }
      // Schedule can reject requests without returning a batch; those terminal states still need delivery.
      RetireSequences();
      PublishOutputs();
    }
    LogRuntimeStatistics(true);
  } catch (...) {
    failure_ = std::current_exception();
    const std::scoped_lock lock{latch_};
    closing_ = true;
  }

  const auto shutdown_started = std::chrono::steady_clock::now();
  ZEPHYR_LOG_INFO("Stopping engine and releasing device resources");
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
  context_.reset();
  try {
    if (runtime_ != nullptr) {
      runtime_->Shutdown();
      if (const auto *error = runtime_errors_->GetError(); error != nullptr) {
        throw ttl::Error(error->code_, error->message_, error->location_);
      }
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
        ZEPHYR_LOG_ERROR("Engine failed: {}", error.what());
        FailAllRequests(error.what());
      } catch (...) {
        ZEPHYR_LOG_ERROR("Engine failed with an unknown error");
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
  {
    const std::scoped_lock lock{latch_};
    stopped_ = true;
  }
  ZEPHYR_LOG_INFO("Engine stopped in {:.2f}s{}",
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - shutdown_started).count(),
                  failure_ != nullptr ? " after an error" : "");
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
