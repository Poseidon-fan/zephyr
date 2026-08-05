#include <atomic>
#include <cstdint>
#include <source_location>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/runtime/memory/allocation_budget.hpp"
#include "ttl/internal/runtime/memory/allocator_lifecycle.hpp"
#include "ttl/internal/runtime/memory/pinned/retirement.hpp"
#include "ttl/internal/runtime/memory/retirement_queue.hpp"

namespace ttl::internal {
namespace {

void EnqueuePinnedRetirement(RetirementQueue<PinnedRetirement> &queue, AllocationBudget &budget,
                             size_t capacity_bytes) {
  auto budget_result = budget.Reserve(static_cast<uint64_t>(capacity_bytes));
  ASSERT_EQ(budget_result.failure_, BudgetFailure::NONE);
  ASSERT_TRUE(budget_result.reservation_.has_value());
  auto ticket = queue.Reserve(std::source_location::current());
  queue.Enqueue(std::move(ticket), PinnedRetirement{
                                       {.pointer_ = nullptr, .capacity_bytes_ = capacity_bytes},
                                       PinnedStreamUsageSnapshot{},
                                       std::move(*budget_result.reservation_),
                                       std::source_location::current(),
                                   });
}

TEST(AllocationBudgetTest, ReservationTracksCurrentPeakAndMoveLifetime) {
  AllocationBudget budget{16};
  auto first_result = budget.Reserve(6);
  ASSERT_EQ(first_result.failure_, BudgetFailure::NONE);
  ASSERT_TRUE(first_result.reservation_.has_value());
  EXPECT_EQ(first_result.previous_bytes_, 0);
  EXPECT_EQ(budget.GetCurrentBytes(), 6);

  auto first = std::move(*first_result.reservation_);
  auto moved = std::move(first);
  EXPECT_EQ(moved.GetBytes(), 6);

  {
    auto second_result = budget.Reserve(10);
    ASSERT_EQ(second_result.failure_, BudgetFailure::NONE);
    ASSERT_TRUE(second_result.reservation_.has_value());
    EXPECT_EQ(budget.GetCurrentBytes(), 16);
    EXPECT_EQ(budget.GetPeakBytes(), 16);
  }
  EXPECT_EQ(budget.GetCurrentBytes(), 6);
}

TEST(AllocationBudgetTest, ReportsLimitAndOverflowWithoutChangingAccounting) {
  AllocationBudget limited{8};
  auto reservation = limited.Reserve(8);
  ASSERT_EQ(reservation.failure_, BudgetFailure::NONE);
  EXPECT_EQ(limited.Reserve(1).failure_, BudgetFailure::LIMIT);
  EXPECT_EQ(limited.GetCurrentBytes(), 8);

  AllocationBudget unlimited{0};
  auto maximum = unlimited.Reserve(UINT64_MAX);
  ASSERT_EQ(maximum.failure_, BudgetFailure::NONE);
  EXPECT_EQ(unlimited.Reserve(1).failure_, BudgetFailure::OVERFLOW);
  EXPECT_EQ(unlimited.GetCurrentBytes(), UINT64_MAX);
}

TEST(AllocatorLifecycleTest, EnforcesTransitionsAndPreservesFailureUntilClosing) {
  AllocatorLifecycle lifecycle;
  EXPECT_EQ(lifecycle.GetStatus(), AllocatorStatus::RUNNING);
  EXPECT_NO_THROW(lifecycle.RequireRunning("allocator", std::source_location::current()));

  lifecycle.MarkFailed();
  EXPECT_EQ(lifecycle.GetStatus(), AllocatorStatus::FAILED);
  EXPECT_THROW(lifecycle.RequireRunning("allocator", std::source_location::current()), InvalidArgumentError);
  EXPECT_TRUE(lifecycle.BeginClosing());
  EXPECT_EQ(lifecycle.GetStatus(), AllocatorStatus::CLOSING);
  EXPECT_TRUE(lifecycle.BeginClosing());
  lifecycle.MarkClosed();
  EXPECT_FALSE(lifecycle.BeginClosing());
  lifecycle.MarkClosed();
}

TEST(RetirementQueueTest, BindsTicketAndRequeuesIncompleteCheckoutWithoutAllocation) {
  RetirementQueue<PinnedRetirement> queue{1};
  {
    auto cancelled = queue.Reserve(std::source_location::current());
    EXPECT_TRUE(cancelled);
  }

  AllocationBudget budget{16};
  EnqueuePinnedRetirement(queue, budget, 8);
  EXPECT_EQ(queue.GetAvailableCount(), 1);
  EXPECT_EQ(queue.GetPendingCount(), 1);
  EXPECT_EQ(queue.GetTotalCount(), 1);
  EXPECT_EQ(budget.GetCurrentBytes(), 8);

  {
    auto checkout = queue.CheckoutFront();
    ASSERT_TRUE(checkout.has_value());
    EXPECT_EQ(queue.GetAvailableCount(), 0);
  }
  EXPECT_EQ(queue.GetAvailableCount(), 1);
  EXPECT_EQ(queue.GetPendingCount(), 1);

  auto checkout = queue.CheckoutFront();
  ASSERT_TRUE(checkout.has_value());
  checkout->Complete();
  checkout.reset();
  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(budget.GetCurrentBytes(), 0);
}

TEST(RetirementQueueTest, EnqueueDoesNotDisplaceSnapshotRecords) {
  RetirementQueue<PinnedRetirement> queue{1};
  AllocationBudget budget{16};
  EnqueuePinnedRetirement(queue, budget, 1);
  EnqueuePinnedRetirement(queue, budget, 2);

  const auto snapshot_count = queue.GetAvailableCount();
  ASSERT_EQ(snapshot_count, 2);
  {
    auto checkout = queue.CheckoutFront();
    ASSERT_TRUE(checkout.has_value());
    EXPECT_EQ(checkout->Get().allocation_.capacity_bytes_, 1);
  }

  EnqueuePinnedRetirement(queue, budget, 3);
  {
    auto checkout = queue.CheckoutFront();
    ASSERT_TRUE(checkout.has_value());
    EXPECT_EQ(checkout->Get().allocation_.capacity_bytes_, 2);
    checkout->Complete();
  }

  std::vector<size_t> remaining;
  while (auto checkout = queue.CheckoutFront()) {
    remaining.push_back(checkout->Get().allocation_.capacity_bytes_);
    checkout->Complete();
  }
  EXPECT_EQ(remaining, (std::vector<size_t>{1, 3}));
  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(budget.GetCurrentBytes(), 0);
}

}  // namespace
}  // namespace ttl::internal
