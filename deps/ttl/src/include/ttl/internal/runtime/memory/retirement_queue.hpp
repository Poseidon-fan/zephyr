#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <source_location>
#include <type_traits>
#include <vector>

#include "ttl/internal/runtime/memory/retirement_ticket.hpp"

namespace ttl::internal {

/** Pre-reserved queue for records published from noexcept allocation-owner destructors. */
template <typename Record>
class RetirementQueue final {
  static_assert(std::is_nothrow_move_constructible_v<Record>, "retirement records must be nothrow move constructible");

 public:
  class Checkout final {
   public:
    Checkout(const Checkout &) = delete;
    auto operator=(const Checkout &) -> Checkout & = delete;
    Checkout(Checkout &&other) noexcept;
    auto operator=(Checkout &&) -> Checkout & = delete;
    ~Checkout() noexcept;

    [[nodiscard]] auto Get() noexcept -> Record &;
    [[nodiscard]] auto Get() const noexcept -> const Record &;
    void Complete() noexcept;

   private:
    friend class RetirementQueue;

    Checkout(RetirementQueue *queue, Record record) noexcept;

    RetirementQueue *queue_;
    std::optional<Record> record_;
  };

  explicit RetirementQueue(size_t initial_capacity = 16);

  RetirementQueue(const RetirementQueue &) = delete;
  auto operator=(const RetirementQueue &) -> RetirementQueue & = delete;
  RetirementQueue(RetirementQueue &&) = delete;
  auto operator=(RetirementQueue &&) -> RetirementQueue & = delete;

  [[nodiscard]] auto Reserve(std::source_location location) -> RetirementTicket;
  void Enqueue(RetirementTicket ticket, Record record) noexcept;
  [[nodiscard]] auto CheckoutFront() noexcept -> std::optional<Checkout>;
  [[nodiscard]] auto GetAvailableCount() const noexcept -> size_t;
  [[nodiscard]] auto GetPendingCount() const noexcept -> uint64_t;
  [[nodiscard]] auto GetTotalCount() const noexcept -> uint64_t;
  [[nodiscard]] auto Empty() const noexcept -> bool;

 private:
  [[nodiscard]] static auto Advance(size_t index, size_t offset, size_t capacity) noexcept -> size_t;
  static void Increment(std::atomic<uint64_t> &counter) noexcept;
  static void CancelTicket(void *owner) noexcept;
  void Cancel() noexcept;
  void Return(Record record) noexcept;
  void CompleteCheckout() noexcept;

  size_t initial_capacity_;
  mutable std::mutex latch_;
  std::vector<std::optional<Record>> slots_;
  size_t head_{0};
  size_t available_count_{0};
  size_t reserved_count_{0};
  size_t checked_out_count_{0};
  std::atomic<uint64_t> pending_count_{0};
  std::atomic<uint64_t> total_count_{0};
};

}  // namespace ttl::internal
