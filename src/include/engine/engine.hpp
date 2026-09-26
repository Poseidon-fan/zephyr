#pragma once

#include <condition_variable>
#include <cstddef>
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
#include "scheduler/scheduler.hpp"

namespace zephyr::engine {

struct EngineOptions final {
  executor::ExecutorOptions executor_;
  scheduler::SchedulerConfig scheduler_;
  /** Includes queued, running, and completed-but-unread choices; independent of GPU residency. */
  size_t max_outstanding_sequences_{256};
  /** Pause scheduling at this unread-output watermark; the current plan and terminal outputs may exceed it. */
  size_t max_buffered_output_bytes_{16U * 1024U * 1024U};
  /** Resolved model EOS IDs. Tokenization and text-level stop processing belong to the caller. */
  std::vector<token_id_t> eos_token_ids_;
};

/**
 * One model and a single control thread owning scheduling, sequences, and logical KV state.
 * Submit/Cancel may run concurrently; WaitForOutputs has one consumer. No user callback runs on the
 * control thread. Runtime is borrowed and must outlive the engine. Returned outputs own only CPU data.
 */
class Engine final {
 public:
  [[nodiscard]] static auto Create(ttl::Runtime &runtime, EngineOptions options, executor::ExecutionFactory factory)
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
  /** Wait for increments or final results; empty means normal shutdown, failures rethrow after outputs drain. */
  [[nodiscard]] auto WaitForOutputs() -> std::vector<RequestOutput>;
  /** Stop admission, finish in-flight work, terminate remaining requests, and join. Rethrows fatal failures. */
  void Close();

 private:
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

  Engine(ttl::Runtime &runtime, EngineOptions options);
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

  ttl::Runtime &runtime_;
  const EngineOptions options_;
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

  std::mutex latch_;
  std::condition_variable changed_;
  // Only command transport, admission credits, and output delivery are shared with calling threads.
  std::deque<std::pair<request_id_t, Request>> pending_;
  std::unordered_set<request_id_t> cancellations_;
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
