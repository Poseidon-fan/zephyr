#include <array>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/runtime/device_properties.hpp"

namespace ttl::test {
namespace {

TEST(DeviceTest, PreservesOrdinalAndProvidesStableFormattingAndHashing) {
  const Device first{0};
  const Device same{0};
  const Device second{2};

  EXPECT_EQ(first.GetOrdinal(), 0);
  EXPECT_EQ(first, same);
  EXPECT_NE(first, second);
  EXPECT_EQ(first.ToString(), "cuda:0");

  std::ostringstream stream;
  stream << second;
  EXPECT_EQ(stream.str(), "cuda:2");

  const std::unordered_set<Device> devices{first, same, second};
  EXPECT_EQ(devices.size(), 2U);
}

TEST(DeviceTest, RejectsNegativeOrdinalAtRuntimeAndConstantEvaluationBoundary) {
  EXPECT_THROW(static_cast<void>(Device{-1}), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Device{INT32_MIN}), InvalidArgumentError);
}

TEST(DevicePropertiesTest, ComputeCapabilityOrdersLexicographicallyAndFormatsSmVersion) {
  constexpr ComputeCapability sm80{.major_ = 8, .minor_ = 0};
  constexpr ComputeCapability sm89{.major_ = 8, .minor_ = 9};
  constexpr ComputeCapability sm90{.major_ = 9, .minor_ = 0};
  static_assert(sm80.GetSmVersion() == 80);
  static_assert(sm89.GetSmVersion() == 89);
  EXPECT_LT(sm80, sm89);
  EXPECT_LT(sm89, sm90);
}

TEST(ErrorTest, EveryStableCodeHasASymbolicNameAndUnknownValuesAreHandled) {
  constexpr std::array<std::pair<ErrorCode, std::string_view>, 10> cases{{
      {ErrorCode::INVALID_ARGUMENT, "INVALID_ARGUMENT"},
      {ErrorCode::OVERFLOW, "OVERFLOW"},
      {ErrorCode::NOT_SUPPORTED, "NOT_SUPPORTED"},
      {ErrorCode::OUT_OF_MEMORY, "OUT_OF_MEMORY"},
      {ErrorCode::CUDA, "CUDA"},
      {ErrorCode::CUBLAS, "CUBLAS"},
      {ErrorCode::NCCL, "NCCL"},
      {ErrorCode::CAPTURE, "CAPTURE"},
      {ErrorCode::ASYNC_EXECUTION, "ASYNC_EXECUTION"},
      {ErrorCode::INTERNAL, "INTERNAL"},
  }};

  for (const auto &[code, name] : cases) {
    EXPECT_EQ(ErrorCodeToString(code), name);
  }
  // Exercise the defensive default branch for an ABI value from outside this build.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(ErrorCodeToString(static_cast<ErrorCode>(UINT8_MAX)), "UNKNOWN");
}

TEST(ErrorTest, PreservesStructuredFieldsAndFormatsWhat) {
  const auto location = std::source_location::current();
  const InvalidArgumentError error{"bad argument", location};

  EXPECT_EQ(error.GetCode(), ErrorCode::INVALID_ARGUMENT);
  EXPECT_EQ(error.GetMessage(), "bad argument");
  EXPECT_EQ(error.GetLocation().line(), location.line());
  EXPECT_EQ(error.GetLocation().file_name(), location.file_name());
  EXPECT_NE(std::string_view{error.what()}.find("[INVALID_ARGUMENT] bad argument"), std::string_view::npos);
  EXPECT_NE(std::string_view{error.what()}.find(location.file_name()), std::string_view::npos);
}

TEST(ErrorTest, DerivedErrorsMapToTheirDocumentedCodes) {
  EXPECT_EQ(OverflowError{"x"}.GetCode(), ErrorCode::OVERFLOW);
  EXPECT_EQ(NotSupportedError{"x"}.GetCode(), ErrorCode::NOT_SUPPORTED);
  EXPECT_EQ(OutOfMemoryError{"x"}.GetCode(), ErrorCode::OUT_OF_MEMORY);
  EXPECT_EQ(CudaError{"x"}.GetCode(), ErrorCode::CUDA);
  EXPECT_EQ(CublasError{"x"}.GetCode(), ErrorCode::CUBLAS);
  EXPECT_EQ(NcclError{"x"}.GetCode(), ErrorCode::NCCL);
  EXPECT_EQ(CaptureError{"x"}.GetCode(), ErrorCode::CAPTURE);
  EXPECT_EQ(DeviceError{"x"}.GetCode(), ErrorCode::ASYNC_EXECUTION);
  EXPECT_EQ(InternalError{"x"}.GetCode(), ErrorCode::INTERNAL);
}

TEST(ErrorSinkTest, RecordingSinkAcceptsConcurrentDestructorSafeReports) {
  auto sink = std::make_shared<RecordingErrorSink>();
  constexpr size_t thread_count = 8;
  constexpr size_t reports_per_thread = 64;
  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (size_t thread = 0; thread < thread_count; ++thread) {
    threads.emplace_back([sink, thread] {
      for (size_t report = 0; report < reports_per_thread; ++report) {
        sink->Report(ErrorRecord{.code_ = ErrorCode::INTERNAL,
                                 .message_ = std::to_string(thread) + ":" + std::to_string(report),
                                 .device_ = Device{static_cast<int32_t>(thread)},
                                 .stream_id_ = report,
                                 .location_ = std::source_location::current()});
      }
    });
  }
  for (std::thread &thread : threads) {
    thread.join();
  }

  const auto records = sink->GetRecords();
  EXPECT_EQ(records.size(), thread_count * reports_per_thread);
}

}  // namespace
}  // namespace ttl::test
