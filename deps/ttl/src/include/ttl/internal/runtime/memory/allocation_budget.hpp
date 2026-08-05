#pragma once

#include <atomic>
#include <cstdint>
#include <optional>

namespace ttl::internal {

enum class BudgetFailure : uint8_t {
  NONE,
  LIMIT,
  OVERFLOW,
};

class AllocationBudget final {
 public:
  class Reservation final {
   public:
    Reservation() noexcept = default;
    Reservation(const Reservation &) = delete;
    auto operator=(const Reservation &) -> Reservation & = delete;
    Reservation(Reservation &&other) noexcept;
    auto operator=(Reservation &&other) noexcept -> Reservation &;
    ~Reservation() noexcept;

    [[nodiscard]] auto GetBytes() const noexcept -> uint64_t;
    [[nodiscard]] explicit operator bool() const noexcept;

   private:
    friend class AllocationBudget;

    Reservation(AllocationBudget *budget, uint64_t bytes) noexcept;
    void Reset() noexcept;

    AllocationBudget *budget_{nullptr};
    uint64_t bytes_{0};
  };

  struct ReserveResult final {
    BudgetFailure failure_;
    uint64_t previous_bytes_;
    std::optional<Reservation> reservation_;
  };

  explicit AllocationBudget(uint64_t maximum_bytes) noexcept;

  AllocationBudget(const AllocationBudget &) = delete;
  auto operator=(const AllocationBudget &) -> AllocationBudget & = delete;
  AllocationBudget(AllocationBudget &&) = delete;
  auto operator=(AllocationBudget &&) -> AllocationBudget & = delete;

  [[nodiscard]] auto Reserve(uint64_t bytes) noexcept -> ReserveResult;
  [[nodiscard]] auto GetCurrentBytes() const noexcept -> uint64_t;
  [[nodiscard]] auto GetPeakBytes() const noexcept -> uint64_t;
  [[nodiscard]] auto GetMaximumBytes() const noexcept -> uint64_t;

 private:
  void Release(uint64_t bytes) noexcept;
  void UpdatePeak(uint64_t value) noexcept;

  uint64_t maximum_bytes_;
  std::atomic<uint64_t> current_bytes_{0};
  std::atomic<uint64_t> peak_bytes_{0};
};

}  // namespace ttl::internal
