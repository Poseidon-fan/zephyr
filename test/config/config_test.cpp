#include "config/config.h"
#include "common/exception.h"
#include "gtest/gtest.h"

namespace zephyr {
namespace {

auto MakeDecoderConfig() -> Config {
  auto config = Config{};
  config.model_directory_ = "/models/example";
  config.runtime_.devices_ = {ttl::Device{0}, ttl::Device{1}};
  config.plan_.parallel_.tensor_parallel_size_ = 2;
  config.plan_.parallel_.data_parallel_size_ = 1;
  config.plan_.limits_.max_tokens_per_worker_ = 1024;
  config.plan_.limits_.max_sequences_per_worker_ = 8;
  config.plan_.limits_.max_sequence_length_ = 4096;
  config.plan_.limits_.kv_cache_bytes_per_worker_ = 1U << 30U;
  config.plan_.kv_cache_block_size_ = 16;
  config.plan_.dynamic_dimension_capacities_.emplace("tokens", 1024);
  config.scheduler_.max_batch_tokens_ = 1024;
  config.scheduler_.max_batch_sequences_ = 8;
  config.scheduler_.prefill_chunk_size_ = 512;
  return config;
}

TEST(ConfigTest, ValidatesDecoderConfiguration) { EXPECT_NO_THROW(MakeDecoderConfig().Validate()); }

TEST(ConfigTest, RequiresModelDirectory) {
  auto config = MakeDecoderConfig();
  config.model_directory_.clear();
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RequiresDeviceCountToMatchParallelism) {
  auto config = MakeDecoderConfig();
  config.plan_.parallel_.tensor_parallel_size_ = 1;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RequiresPositiveParallelSizes) {
  auto config = MakeDecoderConfig();
  config.plan_.parallel_.data_parallel_size_ = 0;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsDuplicateDevices) {
  auto config = MakeDecoderConfig();
  config.runtime_.devices_.push_back(ttl::Device{0});
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsInvalidWorkerLimits) {
  auto config = MakeDecoderConfig();
  config.plan_.limits_.max_tokens_per_worker_ = 0;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsInvalidDynamicDimensionCapacity) {
  auto config = MakeDecoderConfig();
  config.plan_.dynamic_dimension_capacities_.at("tokens") = 0;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsEmptyDynamicDimensionName) {
  auto config = MakeDecoderConfig();
  config.plan_.dynamic_dimension_capacities_.emplace("", 1);
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsBatchLimitsAboveWorkerCapacity) {
  auto config = MakeDecoderConfig();
  config.scheduler_.max_batch_tokens_ = 2049;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsSequenceBatchLimitsAboveWorkerCapacity) {
  auto config = MakeDecoderConfig();
  config.scheduler_.max_batch_sequences_ = 9;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsPrefillChunkAboveWorkerCapacity) {
  auto config = MakeDecoderConfig();
  config.scheduler_.prefill_chunk_size_ = 1025;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, EmbeddingDoesNotUseSchedulerOrKvSettings) {
  auto config = MakeDecoderConfig();
  config.runner_mode_ = RunnerMode::EMBEDDING;
  config.plan_.limits_.kv_cache_bytes_per_worker_ = 0;
  config.plan_.kv_cache_block_size_ = 0;
  config.scheduler_ = {};

  EXPECT_NO_THROW(config.Validate());
}

TEST(ConfigTest, RejectsSchedulerSettingsForEmbedding) {
  auto config = MakeDecoderConfig();
  config.runner_mode_ = RunnerMode::EMBEDDING;
  config.scheduler_.max_batch_tokens_ = 1;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsUnknownRunnerMode) {
  auto config = MakeDecoderConfig();
  config.runner_mode_ = static_cast<RunnerMode>(255);
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

}  // namespace
}  // namespace zephyr
