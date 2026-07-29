#include "ttl/error_sink.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <source_location>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"

namespace ttl {
namespace {

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override {
    record_.emplace(std::move(error));
    report_count_++;
  }

  [[nodiscard]] auto GetRecord() const noexcept -> const std::optional<ErrorRecord> & { return record_; }
  [[nodiscard]] auto GetReportCount() const noexcept -> size_t { return report_count_; }

 private:
  std::optional<ErrorRecord> record_;
  size_t report_count_{0};
};

TEST(ErrorSinkTest, TransfersOwningRecordToImplementation) {
  RecordingErrorSink error_sink;
  std::string message{"cleanup failed"};
  const auto location = std::source_location::current();

  error_sink.Report(ErrorRecord{
      .code_ = ErrorCode::CUDA,
      .message_ = message,
      .device_ = Device{2},
      .stream_id_ = uint64_t{17},
      .location_ = location,
  });

  ASSERT_TRUE(error_sink.GetRecord().has_value());
  const auto &record = *error_sink.GetRecord();
  EXPECT_EQ(record.code_, ErrorCode::CUDA);
  EXPECT_EQ(record.message_, message);
  EXPECT_EQ(record.device_, Device{2});
  EXPECT_EQ(record.stream_id_, uint64_t{17});
  EXPECT_EQ(record.location_.line(), location.line());
  EXPECT_EQ(error_sink.GetReportCount(), 1);
}

TEST(ErrorSinkTest, RepresentsProcessWideErrorWithoutDeviceOrStream) {
  RecordingErrorSink error_sink;

  error_sink.Report(ErrorRecord{
      .code_ = ErrorCode::INTERNAL,
      .message_ = "runtime shutdown failed",
      .device_ = std::nullopt,
      .stream_id_ = std::nullopt,
      .location_ = std::source_location::current(),
  });

  ASSERT_TRUE(error_sink.GetRecord().has_value());
  EXPECT_FALSE(error_sink.GetRecord()->device_.has_value());
  EXPECT_FALSE(error_sink.GetRecord()->stream_id_.has_value());
}

}  // namespace
}  // namespace ttl
