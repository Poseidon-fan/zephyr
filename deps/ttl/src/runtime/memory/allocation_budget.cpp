#include "ttl/internal/runtime/memory/allocation_budget.hpp"

#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <utility>

namespace ttl::internal {

AllocationBudget::Reservation::Reservation(AllocationBudget *budget, uint64_t bytes) noexcept
    : budget_(budget), bytes_(bytes) {}

AllocationBudget::Reservation::Reservation(Reservation &&other) noexcept
    : budget_(std::exchange(other.budget_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

auto AllocationBudget::Reservation::operator=(Reservation &&other) noexcept -> Reservation & {
  if (this != &other) {
    Reset();
    budget_ = std::exchange(other.budget_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

AllocationBudget::Reservation::~Reservation() noexcept { Reset(); }

auto AllocationBudget::Reservation::GetBytes() const noexcept -> uint64_t { return bytes_; }

AllocationBudget::Reservation::operator bool() const noexcept { return budget_ != nullptr; }

void AllocationBudget::Reservation::Reset() noexcept {
  if (budget_ == nullptr) {
    return;
  }
  budget_->Release(bytes_);
  budget_ = nullptr;
  bytes_ = 0;
}

AllocationBudget::AllocationBudget(uint64_t maximum_bytes) noexcept : maximum_bytes_(maximum_bytes) {}

auto AllocationBudget::Reserve(uint64_t bytes) noexcept -> ReserveResult {
  auto current = current_bytes_.load(std::memory_order_relaxed);
  while (true) {
    if (bytes > std::numeric_limits<uint64_t>::max() - current) {
      return {.failure_ = BudgetFailure::OVERFLOW, .previous_bytes_ = current, .reservation_ = std::nullopt};
    }
    if (maximum_bytes_ != 0 && (current > maximum_bytes_ || bytes > maximum_bytes_ - current)) {
      return {.failure_ = BudgetFailure::LIMIT, .previous_bytes_ = current, .reservation_ = std::nullopt};
    }
    if (current_bytes_.compare_exchange_weak(current, current + bytes, std::memory_order_acq_rel,
                                             std::memory_order_relaxed)) {
      UpdatePeak(current + bytes);
      return {
          .failure_ = BudgetFailure::NONE,
          .previous_bytes_ = current,
          .reservation_ = Reservation{this, bytes},
      };
    }
  }
}

auto AllocationBudget::GetCurrentBytes() const noexcept -> uint64_t {
  return current_bytes_.load(std::memory_order_relaxed);
}

auto AllocationBudget::GetPeakBytes() const noexcept -> uint64_t { return peak_bytes_.load(std::memory_order_relaxed); }

auto AllocationBudget::GetMaximumBytes() const noexcept -> uint64_t { return maximum_bytes_; }

void AllocationBudget::ResetPeakBytes() noexcept { peak_bytes_.store(GetCurrentBytes(), std::memory_order_relaxed); }

void AllocationBudget::Release(uint64_t bytes) noexcept {
  if (current_bytes_.fetch_sub(bytes, std::memory_order_relaxed) < bytes) {
    std::terminate();
  }
}

void AllocationBudget::UpdatePeak(uint64_t value) noexcept {
  auto peak = peak_bytes_.load(std::memory_order_relaxed);
  while (value > peak &&
         !peak_bytes_.compare_exchange_weak(peak, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
}

}  // namespace ttl::internal
