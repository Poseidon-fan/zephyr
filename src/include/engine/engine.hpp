#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/runtime/runtime.hpp>

#include "engine/output.hpp"
#include "engine/request.hpp"
#include "executor/causal_lm.hpp"
#include "executor/embedding.hpp"
#include "executor/executor.hpp"
#include "kv_cache/manager.hpp"
#include "model/loader.hpp"
#include "scheduler/scheduler.hpp"

namespace zephyr::engine {

struct EngineOptions final {
  /** Engine derives execution batch limits from scheduler_; context and memory settings remain user-configurable. */
  executor::ExecutorOptions executor_;
  scheduler::SchedulerConfig scheduler_{scheduler::PagedSchedulerConfig{.max_num_seqs_ = 8,
                                                                        .max_num_batched_tokens_ = 512,
                                                                        .max_prefill_chunk_tokens_ = 512,
                                                                        .max_decode_steps_before_prefill_ = 8}};
  /** Includes queued, running, and completed-but-unread choices; independent of GPU residency. */
  size_t max_outstanding_sequences_{256};
  /** Pause scheduling at this unread-output watermark; the current plan and terminal outputs may exceed it. */
  size_t max_buffered_output_bytes_{16U * 1024U * 1024U};
  /** Resolved model EOS IDs. Tokenization and text-level stop processing belong to the caller. */
  std::vector<token_id_t> eos_token_ids_;
};

/** Effective model capabilities, available after initialization and independent of runtime lifetime. */
struct EngineInfo final {
  model::ModelTask task_;
  int64_t max_seq_len_;
  int64_t vocab_size_;
};

/**
 * One model and a single control thread owning scheduling, sequences, and logical KV state.
 * Submission and stop controls may run concurrently; WaitForOutputs has one consumer. No user callback runs on the
 * control thread. The engine owns its GPU runtime; returned outputs own only CPU data.
 * Initialization profiles the selected devices; exclude concurrent allocations on those devices until
 * Create returns. Independent device groups may initialize concurrently.
 */
class Engine final {
 public:
  /** Resolve the model and task before starting workers. Causal options apply only to generation models. */
  [[nodiscard]] static auto Create(EngineOptions options, std::optional<model::ModelTask> task = std::nullopt,
                                   executor::CausalLMOptions causal_lm = {}) -> std::unique_ptr<Engine>;
  /** Native integration point: the supplied factory owns model selection and task-specific configuration. */
  [[nodiscard]] static auto Create(EngineOptions options, executor::ExecutionFactory factory)
      -> std::unique_ptr<Engine>;

  Engine(const Engine &) = delete;
  auto operator=(const Engine &) -> Engine & = delete;
  Engine(Engine &&) = delete;
  auto operator=(Engine &&) -> Engine & = delete;
  ~Engine() noexcept;

  /** Validate and queue a request; invalid arguments or exhausted admission capacity throw synchronously. */
  [[nodiscard]] auto Submit(Request request) -> request_id_t;
  /** Request cancellation at the next safe execution boundary. Unknown/finished IDs are harmless. */
  void Cancel(request_id_t request_id);
  /** Finish one choice after a caller-detected string stop. Unknown/finished choices are harmless. */
  void StopChoice(request_id_t request_id, size_t choice_index);
  [[nodiscard]] auto GetInfo() const -> EngineInfo { return info_; }
  /** Wait for increments or final results; empty means normal shutdown, failures rethrow after outputs drain. */
  [[nodiscard]] auto WaitForOutputs() -> std::vector<RequestOutput>;
  /** Stop admission, finish in-flight work, cancel remaining requests, and shut down the runtime. Rethrows failures. */
  void Close();

 private:
  class RuntimeErrors;

  struct RequestState final {
    RequestState(request_id_t id, Request request, size_t num_sequences);

    /** Find or create this choice's increment for the current control-loop iteration. */
    auto GetChoiceOutput(size_t index) -> ChoiceOutput &;

    Request request_;
    size_t remaining_sequences_;
    std::vector<std::optional<FinishReason>> finish_reasons_;
    /** Dense choice IDs map to the sparse output without searching the output vector. */
    std::vector<size_t> output_indices_;
    Usage usage_;
    RequestStatus status_{RequestStatus::COMPLETED};
    std::string error_message_;
    bool canceled_{false};
    RequestOutput output_{};
  };

  struct SequenceContext final {
    request_id_t request_id_;
    size_t choice_index_;
    std::optional<std::mt19937_64> rng_;
  };

  /** Admission credits remain charged until the consumer receives the final output. */
  struct OutstandingRequest final {
    size_t num_sequences_;
    size_t prompt_tokens_;
    bool finished_{false};
  };

  /** Control-thread counters for the current logging interval, excluding initialization probes. */
  struct LogStatistics final {
    std::chrono::steady_clock::time_point last_log_;
    size_t input_tokens_{0};
    size_t generated_tokens_{0};
    bool was_active_{false};
  };

  explicit Engine(EngineOptions options);
  void Run(executor::ExecutionFactory factory) noexcept;
  void Initialize(executor::ExecutionFactory factory);
  /** Consume queued commands at a safe execution boundary; returns whether shutdown was requested. */
  auto ProcessRequests() -> bool;
  void RetireSequences();
  void ExecuteGeneration(const scheduler::ScheduledBatch &batch);
  void ExecuteEmbedding(const scheduler::ScheduledBatch &batch);
  auto ExecuteBatch(const executor::ExecutionBatch &input, const scheduler::ScheduledBatch &batch)
      -> std::optional<executor::ExecutionResult>;
  void FailBatch(const scheduler::ScheduledBatch &batch, const std::string &message);
  void FailAllRequests(const std::string &message);
  /** Move complete increments into the output queue without waiting for a consumer. */
  void PublishOutputs();
  /** Complete a request after every sequence has been retired. */
  void FinishRequest(RequestState &request);
  void LogRuntimeStatistics(bool force = false);

  const EngineOptions options_;
  EngineInfo info_{};
  std::shared_ptr<RuntimeErrors> runtime_errors_;
  std::unique_ptr<ttl::Runtime> runtime_;
  std::unique_ptr<executor::Executor> executor_;
  const executor::CausalLMExecutionSpec *generation_spec_{nullptr};
  const executor::EmbeddingExecutionSpec *embedding_spec_{nullptr};
  std::optional<ttl::ExecutionContext> context_;
  scheduler::SequenceTable sequences_;
  std::unique_ptr<kv_cache::KVCacheManager> cache_manager_;
  std::unique_ptr<scheduler::Scheduler> scheduler_;
  std::unordered_map<sequence_id_t, SequenceContext> sequence_contexts_;
  std::unordered_map<request_id_t, RequestState> requests_;
  std::mt19937_64 rng_{0};
  sequence_id_t next_sequence_id_{0};
  LogStatistics log_statistics_{};

  std::mutex latch_;
  std::condition_variable changed_;
  // Only command transport, admission credits, and output delivery are shared with calling threads.
  std::deque<std::pair<request_id_t, Request>> pending_;
  std::unordered_set<request_id_t> cancellations_;
  std::unordered_map<request_id_t, std::unordered_set<size_t>> choice_stops_;
  std::unordered_map<request_id_t, OutstandingRequest> outstanding_;
  std::vector<RequestOutput> outputs_;
  size_t outstanding_sequences_{0};
  size_t buffered_output_bytes_{0};
  request_id_t next_request_id_{0};
  bool initialized_{false};
  bool closing_{false};
  bool stopped_{false};
  std::exception_ptr failure_;
  std::once_flag close_once_;
  std::jthread control_thread_;
};

}  // namespace zephyr::engine
