#include "ttl/internal/runtime/memory/retirement_queue.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <optional>
#include <source_location>
#include <utility>

#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/runtime/memory/device/retirement.hpp"
#include "ttl/internal/runtime/memory/pinned/retirement.hpp"
#include "ttl/internal/runtime/memory/retirement_ticket.hpp"

namespace ttl::internal {

RetirementTicket::RetirementTicket(void *owner, Cancel cancel) noexcept : owner_(owner), cancel_(cancel) {}

RetirementTicket::RetirementTicket(RetirementTicket &&other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), cancel_(std::exchange(other.cancel_, nullptr)) {}

auto RetirementTicket::operator=(RetirementTicket &&other) noexcept -> RetirementTicket & {
  if (this != &other) {
    CancelReservation();
    owner_ = std::exchange(other.owner_, nullptr);
    cancel_ = std::exchange(other.cancel_, nullptr);
  }
  return *this;
}

RetirementTicket::~RetirementTicket() noexcept { CancelReservation(); }

RetirementTicket::operator bool() const noexcept { return owner_ != nullptr; }

void RetirementTicket::CancelReservation() noexcept {
  if (owner_ == nullptr) {
    return;
  }
  cancel_(owner_);
  owner_ = nullptr;
  cancel_ = nullptr;
}

void RetirementTicket::Consume(void *expected_owner) noexcept {
  if (owner_ != expected_owner || cancel_ == nullptr) {
    std::terminate();
  }
  owner_ = nullptr;
  cancel_ = nullptr;
}

template <typename Record>
RetirementQueue<Record>::Checkout::Checkout(RetirementQueue *queue, Record record) noexcept
    : queue_(queue), record_(std::move(record)) {}

template <typename Record>
RetirementQueue<Record>::Checkout::Checkout(Checkout &&other) noexcept
    : queue_(std::exchange(other.queue_, nullptr)), record_(std::move(other.record_)) {}

template <typename Record>
RetirementQueue<Record>::Checkout::~Checkout() noexcept {
  if (queue_ != nullptr) {
    queue_->Return(std::move(*record_));
  }
}

template <typename Record>
auto RetirementQueue<Record>::Checkout::Get() noexcept -> Record & {
  return *record_;
}

template <typename Record>
auto RetirementQueue<Record>::Checkout::Get() const noexcept -> const Record & {
  return *record_;
}

template <typename Record>
void RetirementQueue<Record>::Checkout::Complete() noexcept {
  if (queue_ == nullptr) {
    std::terminate();
  }
  queue_->CompleteCheckout();
  queue_ = nullptr;
}

template <typename Record>
RetirementQueue<Record>::RetirementQueue(size_t initial_capacity) : initial_capacity_(initial_capacity) {}

template <typename Record>
auto RetirementQueue<Record>::Advance(size_t index, size_t offset, size_t capacity) noexcept -> size_t {
  if (capacity == 0 || index >= capacity || offset > capacity) {
    std::terminate();
  }
  const auto remaining = capacity - index;
  return offset < remaining ? index + offset : offset - remaining;
}

template <typename Record>
void RetirementQueue<Record>::Increment(std::atomic<uint64_t> &counter) noexcept {
  auto current = counter.load(std::memory_order_relaxed);
  while (true) {
    if (current == std::numeric_limits<uint64_t>::max()) {
      std::terminate();
    }
    if (counter.compare_exchange_weak(current, current + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
      return;
    }
  }
}

template <typename Record>
auto RetirementQueue<Record>::Reserve(std::source_location location) -> RetirementTicket {
  std::scoped_lock lock{latch_};
  const auto occupied = CheckedAdd(available_count_, checked_out_count_, "retirement occupied slot count", location);
  const auto required =
      CheckedAdd(occupied, CheckedAdd(reserved_count_, size_t{1}, "retirement reservation count", location),
                 "retirement queue capacity", location);
  if (slots_.size() < required) {
    auto capacity = std::max(required, initial_capacity_);
    if (!slots_.empty()) {
      capacity = std::max(capacity, CheckedMultiply(slots_.size(), size_t{2}, "retirement queue growth", location));
    }

    std::vector<std::optional<Record>> grown(capacity);
    for (size_t index = 0; index < available_count_; index++) {
      const auto source = Advance(head_, index, slots_.size());
      if (!slots_[source].has_value()) {
        std::terminate();
      }
      grown[index].emplace(std::move(*slots_[source]));
    }
    slots_.swap(grown);
    head_ = 0;
  }
  reserved_count_++;
  return RetirementTicket{this, CancelTicket};
}

template <typename Record>
void RetirementQueue<Record>::Enqueue(RetirementTicket ticket, Record record) noexcept {
  ticket.Consume(this);
  std::scoped_lock lock{latch_};
  if (reserved_count_ == 0 || available_count_ >= slots_.size() ||
      checked_out_count_ > slots_.size() - available_count_) {
    std::terminate();
  }
  const auto tail = Advance(head_, available_count_, slots_.size());
  if (slots_[tail].has_value()) {
    std::terminate();
  }
  reserved_count_--;
  slots_[tail].emplace(std::move(record));
  available_count_++;
  Increment(pending_count_);
  Increment(total_count_);
}

template <typename Record>
auto RetirementQueue<Record>::CheckoutFront() noexcept -> std::optional<Checkout> {
  std::scoped_lock lock{latch_};
  if (available_count_ == 0) {
    return std::nullopt;
  }
  if (!slots_[head_].has_value()) {
    std::terminate();
  }
  Record record{std::move(*slots_[head_])};
  slots_[head_].reset();
  head_ = Advance(head_, 1, slots_.size());
  available_count_--;
  checked_out_count_++;
  return Checkout{this, std::move(record)};
}

template <typename Record>
auto RetirementQueue<Record>::GetAvailableCount() const noexcept -> size_t {
  std::scoped_lock lock{latch_};
  return available_count_;
}

template <typename Record>
auto RetirementQueue<Record>::GetPendingCount() const noexcept -> uint64_t {
  return pending_count_.load(std::memory_order_relaxed);
}

template <typename Record>
auto RetirementQueue<Record>::GetTotalCount() const noexcept -> uint64_t {
  return total_count_.load(std::memory_order_relaxed);
}

template <typename Record>
auto RetirementQueue<Record>::Empty() const noexcept -> bool {
  return GetPendingCount() == 0;
}

template <typename Record>
void RetirementQueue<Record>::CancelTicket(void *owner) noexcept {
  static_cast<RetirementQueue *>(owner)->Cancel();
}

template <typename Record>
void RetirementQueue<Record>::Cancel() noexcept {
  std::scoped_lock lock{latch_};
  if (reserved_count_ == 0) {
    std::terminate();
  }
  reserved_count_--;
}

template <typename Record>
void RetirementQueue<Record>::Return(Record record) noexcept {
  std::scoped_lock lock{latch_};
  if (checked_out_count_ == 0 || available_count_ >= slots_.size() ||
      checked_out_count_ > slots_.size() - available_count_) {
    std::terminate();
  }
  const auto tail = Advance(head_, available_count_, slots_.size());
  if (slots_[tail].has_value()) {
    std::terminate();
  }
  slots_[tail].emplace(std::move(record));
  available_count_++;
  checked_out_count_--;
}

template <typename Record>
void RetirementQueue<Record>::CompleteCheckout() noexcept {
  std::scoped_lock lock{latch_};
  if (checked_out_count_ == 0 || pending_count_.fetch_sub(1, std::memory_order_relaxed) == 0) {
    std::terminate();
  }
  checked_out_count_--;
}

template class RetirementQueue<DeviceRetirement>;
template class RetirementQueue<PinnedRetirement>;

}  // namespace ttl::internal
