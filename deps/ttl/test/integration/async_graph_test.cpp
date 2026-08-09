#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/runtime/graph.hpp"

namespace ttl::test {
namespace {

class AsyncGraphTest : public SingleDeviceTest {};

TEST_F(AsyncGraphTest, ExplicitEventDependencyMakesCrossContextCopySafeAfterProducerLifetimeEnds) {
  ExecutionContext consumer = GetRuntime().CreateExecutionContext(GetDevice());
  Tensor output = Empty(consumer, Shape{4}, DType::INT32);
  std::optional<Event> producer_ready;

  {
    Tensor input = Empty(GetContext(), Shape{4}, DType::INT32);
    FillOut(GetContext(), input, Scalar{int64_t{17}});
    producer_ready.emplace(GetContext().RecordEvent());
    consumer.Wait(*producer_ready);
    CopyOut(consumer, output, input);
  }

  consumer.Synchronize();
  ExpectValues<int32_t>(consumer, output, {17, 17, 17, 17});
  EXPECT_TRUE(producer_ready->Query());
}

TEST_F(AsyncGraphTest, CaptureFinishReplayAndMetadataFollowTheOwningContext) {
  Tensor input = TensorFromValues<int32_t>(GetContext(), Shape{4}, {1, 2, 3, 4});
  Tensor output = Empty(GetContext(), Shape{4}, DType::INT32);
  const auto baseline = GetRuntime().GetStatistics();

  CaptureSession session = GetContext().BeginCapture({.name_ = "increment"});
  EXPECT_TRUE(session.IsActive());
  AddOut(GetContext(), output, input, Scalar{int64_t{1}});
  CapturedGraph graph = session.Finish();

  EXPECT_FALSE(session.IsActive());
  EXPECT_EQ(graph.GetDevice(), GetDevice());
  EXPECT_EQ(graph.GetStreamId(), GetContext().GetStream().GetId());
  EXPECT_EQ(graph.GetName(), "increment");
  EXPECT_GT(graph.GetNodeCount(), 0U);
  EXPECT_EQ(graph.GetLaunchCount(), 0U);

  FillOut(GetContext(), output, Scalar{int64_t{0}});
  graph.Launch(GetContext());
  graph.Launch(GetContext());
  GetContext().Synchronize();
  ExpectValues<int32_t>(GetContext(), output, {2, 3, 4, 5});
  EXPECT_EQ(graph.GetLaunchCount(), 2U);

  const auto path = std::filesystem::temp_directory_path() / "ttl-captured-graph.dot";
  graph.DebugDumpDot(path.string());
  EXPECT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);

  const auto statistics = GetRuntime().GetStatistics();
  EXPECT_EQ(statistics.captured_graph_count_, baseline.captured_graph_count_ + 1U);
}

TEST_F(AsyncGraphTest, CaptureAbortIsNoThrowAndForbiddenOperationsAreRejected) {
  Tensor output = Empty(GetContext(), Shape{4}, DType::INT32);
  {
    CaptureSession session = GetContext().BeginCapture();
    EXPECT_TRUE(session.IsActive());
    session.Abort();
    EXPECT_FALSE(session.IsActive());
  }

  CaptureSession active = GetContext().BeginCapture();
  EXPECT_THROW(static_cast<void>(GetContext().RecordEvent()), CaptureError);
  EXPECT_THROW(GetRuntime().TrimMemory(GetDevice(), 0), CaptureError);
  active.Abort();
  EXPECT_NO_THROW(GetContext().Synchronize());
  static_cast<void>(output);
}

TEST_F(AsyncGraphTest, RuntimeRejectsShutdownWhileGraphOrContextIsAliveAndGraphRejectsWrongContext) {
  Tensor input = TensorFromValues<int32_t>(GetContext(), Shape{1}, {4});
  Tensor output = Empty(GetContext(), Shape{1}, DType::INT32);
  CaptureSession session = GetContext().BeginCapture();
  AddOut(GetContext(), output, input, Scalar{int64_t{1}});
  CapturedGraph graph = session.Finish();
  ExecutionContext other = GetRuntime().CreateExecutionContext(GetDevice());
  EXPECT_THROW(graph.Launch(other), InvalidArgumentError);
  other.Synchronize();
  EXPECT_THROW(GetRuntime().Shutdown(), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl::test
