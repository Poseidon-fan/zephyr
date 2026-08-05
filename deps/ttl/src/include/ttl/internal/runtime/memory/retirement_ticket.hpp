#pragma once

namespace ttl::internal {

/** Move-only reservation that guarantees one future noexcept retirement enqueue. */
class RetirementTicket final {
 public:
  RetirementTicket() noexcept = default;
  RetirementTicket(const RetirementTicket &) = delete;
  auto operator=(const RetirementTicket &) -> RetirementTicket & = delete;
  RetirementTicket(RetirementTicket &&other) noexcept;
  auto operator=(RetirementTicket &&other) noexcept -> RetirementTicket &;
  ~RetirementTicket() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;

 private:
  template <typename Record>
  friend class RetirementQueue;

  using Cancel = void (*)(void *) noexcept;

  RetirementTicket(void *owner, Cancel cancel) noexcept;
  void CancelReservation() noexcept;
  void Consume(void *expected_owner) noexcept;

  void *owner_{nullptr};
  Cancel cancel_{nullptr};
};

}  // namespace ttl::internal
