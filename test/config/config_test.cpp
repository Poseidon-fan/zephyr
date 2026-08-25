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
  config.plan_.model_.max_model_len_ = 4096;
  config.plan_.kv_cache_.kv_cache_memory_bytes_ = 1U << 30U;
  config.plan_.kv_cache_.block_size_ = 16;
  config.scheduler_.max_num_batched_tokens_ = 1024;
  config.scheduler_.max_num_seqs_ = 8;
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

TEST(ConfigTest, RejectsInvalidSchedulerLimits) {
  auto config = MakeDecoderConfig();
  config.scheduler_.max_num_batched_tokens_ = 0;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsMoreSequencesThanTokens) {
  auto config = MakeDecoderConfig();
  config.scheduler_.max_num_batched_tokens_ = 7;
  config.scheduler_.max_num_seqs_ = 8;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsInvalidModelLength) {
  auto config = MakeDecoderConfig();
  config.plan_.model_.max_model_len_ = 0;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsPrefillChunkAboveTokenBudget) {
  auto config = MakeDecoderConfig();
  config.scheduler_.prefill_chunk_size_ = 1025;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, EmbeddingUsesSchedulerWithoutKvCache) {
  auto config = MakeDecoderConfig();
  config.runner_mode_ = RunnerMode::EMBEDDING;
  config.plan_.kv_cache_.kv_cache_memory_bytes_ = 0;
  config.plan_.kv_cache_.block_size_ = 0;
  config.scheduler_.prefill_chunk_size_ = 0;

  EXPECT_NO_THROW(config.Validate());
}

TEST(ConfigTest, RejectsPrefillChunkForEmbedding) {
  auto config = MakeDecoderConfig();
  config.runner_mode_ = RunnerMode::EMBEDDING;
  config.plan_.kv_cache_ = {};
  config.scheduler_.prefill_chunk_size_ = 1;
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

TEST(ConfigTest, RejectsUnknownRunnerMode) {
  auto config = MakeDecoderConfig();
  config.runner_mode_ = static_cast<RunnerMode>(255);
  EXPECT_THROW(config.Validate(), ConfigurationException);
}

}  // namespace
}  // namespace zephyr
