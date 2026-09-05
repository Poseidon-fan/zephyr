#include "weight/checkpoint.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <ranges>
#include <regex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <ttl/common/error.hpp>

#include "common/exception.hpp"

namespace zephyr::weight {
namespace {

constexpr size_t SAFETENSORS_PREAMBLE_SIZE = sizeof(uint64_t);
constexpr std::string_view SAFETENSORS_SUFFIX = ".safetensors";
constexpr std::string_view SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json";

/** Parse a JSON document stored in a mapped checkpoint file. */
[[nodiscard]] auto ParseJson(std::span<const std::byte> bytes, const std::filesystem::path &path) -> nlohmann::json {
  if (bytes.empty()) {
    throw ConfigurationException(path.string() + ": JSON document is empty");
  }

  try {
    const auto *begin = reinterpret_cast<const char *>(bytes.data());
    return nlohmann::json::parse(begin, begin + bytes.size());
  } catch (const nlohmann::json::exception &error) {
    std::string message{path.string()};
    message.append(": invalid JSON: ");
    message.append(error.what());
    throw ConfigurationException(message);
  }
}

/** Parse one non-negative integer from a safetensors JSON field. */
[[nodiscard]] auto ParseNonNegativeInteger(const nlohmann::json &value, std::string_view field,
                                           const std::filesystem::path &path) -> uint64_t {
  if (value.is_number_unsigned()) {
    try {
      return value.get<uint64_t>();
    } catch (const nlohmann::json::exception &) {
      std::string message{path.string()};
      message.append(": ");
      message.append(field);
      message.append(" is outside uint64 range");
      throw ConfigurationException(message);
    }
  }
  if (value.is_number_integer()) {
    const auto signed_value = value.get<int64_t>();
    if (signed_value >= 0) {
      return static_cast<uint64_t>(signed_value);
    }
  }

  std::string message{path.string()};
  message.append(": ");
  message.append(field);
  message.append(" must be a non-negative integer");
  throw ConfigurationException(message);
}

/** Convert the JSON shape of one parameter to a TTL shape. */
[[nodiscard]] auto ParseShape(const nlohmann::json &value, const std::filesystem::path &path,
                              std::string_view parameter_name) -> ttl::Shape {
  if (!value.is_array()) {
    std::string message{path.string()};
    message.append(": parameter '");
    message.append(parameter_name);
    message.append("' shape must be an array");
    throw ConfigurationException(message);
  }

  std::vector<int64_t> dimensions;
  dimensions.reserve(value.size());
  for (const auto &dimension : value) {
    const auto raw_dimension = ParseNonNegativeInteger(dimension, "shape dimension", path);
    if (raw_dimension > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      std::string message{path.string()};
      message.append(": parameter '");
      message.append(parameter_name);
      message.append("' shape dimension exceeds int64 range");
      throw ConfigurationException(message);
    }
    dimensions.push_back(static_cast<int64_t>(raw_dimension));
  }

  try {
    return ttl::Shape{std::span<const int64_t>{dimensions.data(), dimensions.size()}};
  } catch (const ttl::Error &error) {
    std::string message{path.string()};
    message.append(": parameter '");
    message.append(parameter_name);
    message.append("' has an invalid shape: ");
    message.append(error.what());
    throw ConfigurationException(message);
  }
}

/** Convert a safetensors dtype spelling to a TTL dtype. */
[[nodiscard]] auto ParseDType(const nlohmann::json &value, const std::filesystem::path &path,
                              std::string_view parameter_name) -> ttl::DType {
  if (!value.is_string()) {
    std::string message{path.string()};
    message.append(": parameter '");
    message.append(parameter_name);
    message.append("' dtype must be a string");
    throw ConfigurationException(message);
  }

  const auto dtype = value.get<std::string>();
  if (dtype == "BOOL") {
    return ttl::DType::BOOL;
  }
  if (dtype == "U8") {
    return ttl::DType::UINT8;
  }
  if (dtype == "I32") {
    return ttl::DType::INT32;
  }
  if (dtype == "I64") {
    return ttl::DType::INT64;
  }
  if (dtype == "F16") {
    return ttl::DType::FLOAT16;
  }
  if (dtype == "BF16") {
    return ttl::DType::BFLOAT16;
  }
  if (dtype == "F32") {
    return ttl::DType::FLOAT32;
  }

  std::string message{path.string()};
  message.append(": safetensors dtype '");
  message.append(dtype);
  message.append("' for parameter '");
  message.append(parameter_name);
  message.append("' is not supported by the weight loader");
  throw NotImplementedException(message);
}

}  // namespace

Checkpoint::MappedFile::MappedFile(const std::filesystem::path &path) : path_(path) {
  const auto file_descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (file_descriptor < 0) {
    const auto error = errno;
    std::string message{"open '"};
    message.append(path_.string());
    message.append("' failed: ");
    message.append(std::strerror(error));
    if (error == ENOMEM) {
      throw OutOfMemoryException(message);
    }
    throw ConfigurationException(message);
  }

  struct stat file_status{};
  if (::fstat(file_descriptor, &file_status) != 0) {
    const auto error = errno;
    static_cast<void>(::close(file_descriptor));
    std::string message{"fstat '"};
    message.append(path_.string());
    message.append("' failed: ");
    message.append(std::strerror(error));
    throw ConfigurationException(message);
  }
  if (!S_ISREG(file_status.st_mode)) {
    static_cast<void>(::close(file_descriptor));
    throw ConfigurationException(path_.string() + ": checkpoint entry is not a regular file");
  }
  if (file_status.st_size < 0 || static_cast<uintmax_t>(file_status.st_size) > std::numeric_limits<size_t>::max()) {
    static_cast<void>(::close(file_descriptor));
    throw ConfigurationException(path_.string() + ": file size cannot be represented by the host");
  }

  size_bytes_ = static_cast<size_t>(file_status.st_size);
  if (size_bytes_ != 0) {
    mapping_ = ::mmap(nullptr, size_bytes_, PROT_READ, MAP_PRIVATE, file_descriptor, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      const auto error = errno;
      static_cast<void>(::close(file_descriptor));
      std::string message{"mmap '"};
      message.append(path_.string());
      message.append("' failed: ");
      message.append(std::strerror(error));
      if (error == ENOMEM) {
        throw OutOfMemoryException(message);
      }
      throw ConfigurationException(message);
    }
  }

  // The mapping keeps the file contents alive; retaining a descriptor per
  // shard would needlessly consume the process descriptor limit.
  static_cast<void>(::close(file_descriptor));
}

Checkpoint::MappedFile::~MappedFile() noexcept {
  if (mapping_ != nullptr) {
    static_cast<void>(::munmap(mapping_, size_bytes_));
  }
}

Checkpoint::MappedFile::MappedFile(MappedFile &&other) noexcept
    : path_(std::move(other.path_)), mapping_(other.mapping_), size_bytes_(other.size_bytes_) {
  other.mapping_ = nullptr;
  other.size_bytes_ = 0;
}

Checkpoint::Checkpoint(std::filesystem::path model_dir) : model_dir_(std::move(model_dir)) {
  std::error_code error;
  if (!std::filesystem::is_directory(model_dir_, error)) {
    if (error) {
      throw ConfigurationException(model_dir_.string() + ": " + error.message());
    }
    throw ConfigurationException(model_dir_.string() + ": model path is not a directory");
  }

  std::vector<std::filesystem::path> index_paths;
  std::filesystem::directory_iterator entries{model_dir_, error};
  if (error) {
    throw ConfigurationException(model_dir_.string() + ": " + error.message());
  }
  for (const auto &entry : entries) {
    const auto &entry_path = entry.path();
    const auto filename = entry_path.filename().string();
    if (!filename.ends_with(SAFETENSORS_INDEX_SUFFIX)) {
      continue;
    }
    const auto regular = entry.is_regular_file(error);
    if (error) {
      throw ConfigurationException(entry_path.string() + ": " + error.message());
    }
    if (regular) {
      index_paths.push_back(entry_path);
    }
  }
  if (error) {
    throw ConfigurationException(model_dir_.string() + ": " + error.message());
  }
  if (index_paths.size() > 1) {
    throw ConfigurationException(model_dir_.string() + ": multiple safetensors index files are ambiguous");
  }

  if (index_paths.empty()) {
    LoadWithoutIndex();
  } else {
    LoadWithIndex(index_paths.front());
  }
  if (parameter_index_.empty()) {
    throw ConfigurationException(model_dir_.string() + ": checkpoint contains no parameters");
  }
}

auto Checkpoint::GetParameterInfo(std::string_view full_name) const -> std::optional<ParameterInfo> {
  const auto *record = FindParameter(full_name);
  if (record == nullptr) {
    return std::nullopt;
  }
  return record->info_;
}

void Checkpoint::LoadWithIndex(const std::filesystem::path &index_path) {
  const MappedFile index_file{index_path};
  const auto index = ParseJson(index_file.GetBytes(), index_path);
  if (!index.is_object() || !index.contains("weight_map") || !index.at("weight_map").is_object()) {
    throw ConfigurationException(index_path.string() + ": index must contain an object named weight_map");
  }

  const auto &weight_map = index.at("weight_map");
  if (weight_map.empty()) {
    throw ConfigurationException(index_path.string() + ": index has an empty weight_map");
  }

  std::unordered_map<std::string, std::filesystem::path> expected_files;
  std::vector<std::filesystem::path> files;
  expected_files.reserve(weight_map.size());
  files.reserve(weight_map.size());
  // Keep the index routing by parameter name, while collecting the referenced
  // paths separately so several parameters can share one mapped shard.
  for (const auto &[name, value] : weight_map.items()) {
    if (name.empty()) {
      throw ConfigurationException(index_path.string() + ": index contains an empty parameter name");
    }
    if (!value.is_string()) {
      std::string message{"index entry for parameter '"};
      message.append(name);
      message.append("' must contain a string filename");
      throw ConfigurationException(index_path.string() + ": " + message);
    }

    const auto relative_name = value.get<std::string>();
    const std::filesystem::path relative_path{relative_name};
    if (relative_name.empty() || relative_path.is_absolute() || relative_path.has_root_name() ||
        relative_path.has_root_directory()) {
      std::string message{"index references an absolute or empty checkpoint path '"};
      message.append(relative_name);
      message.push_back('\'');
      throw ConfigurationException(index_path.string() + ": " + message);
    }
    for (const auto &component : relative_path) {
      if (component == "..") {
        std::string message{"index path escapes the model directory: '"};
        message.append(relative_name);
        message.push_back('\'');
        throw ConfigurationException(index_path.string() + ": " + message);
      }
    }
    if (!relative_name.ends_with(SAFETENSORS_SUFFIX)) {
      std::string message{"index references non-safetensors file '"};
      message.append(relative_name);
      message.push_back('\'');
      throw ConfigurationException(index_path.string() + ": " + message);
    }
    const auto file_path = (model_dir_ / relative_path).lexically_normal();
    expected_files.emplace(name, file_path);
    files.push_back(file_path);
  }

  // Parse each referenced file once; weight_map normally names the same shard
  // for many parameters.
  std::ranges::sort(files);
  files.erase(std::ranges::unique(files).begin(), files.end());
  mapped_files_.reserve(files.size());
  for (const auto &file : files) {
    ParseSafetensorsFile(file);
  }

  // The index is authoritative about routing. Checking every entry catches a
  // stale index that happens to point at an existing but different shard.
  for (const auto &[name, expected_file] : expected_files) {
    const auto *record = FindParameter(name);
    if (record == nullptr) {
      std::string message{"index parameter '"};
      message.append(name);
      message.append("' is absent from its referenced file");
      throw ConfigurationException(index_path.string() + ": " + message);
    }
    if (mapped_files_[record->file_index_].GetPath() != expected_file) {
      std::string message{"index parameter '"};
      message.append(name);
      message.append("' is routed to a different file than its header");
      throw ConfigurationException(index_path.string() + ": " + message);
    }
  }
}

void Checkpoint::LoadWithoutIndex() {
  // Discovery state is needed only for this pass; the Checkpoint retains the
  // mappings and parameter records, not filename classification machinery.
  size_t file_count = 0;
  bool has_unrecognized = false;
  std::filesystem::path unrecognized_file;
  std::vector<std::pair<std::filesystem::path, size_t>> shards;
  std::optional<std::filesystem::path> singleton;
  std::optional<bool> family_is_model;
  size_t shard_total = 0;
  const std::regex shard_pattern{"^(model|consolidated)-([0-9]+)-of-([0-9]+)\\.safetensors$"};

  std::error_code error;
  std::filesystem::directory_iterator entries{model_dir_, error};
  if (error) {
    throw ConfigurationException(model_dir_.string() + ": " + error.message());
  }
  // Discover the conventional singleton or complete shard family before
  // mapping anything, so an ambiguous directory fails without partial loads.
  for (const auto &entry : entries) {
    const auto &entry_path = entry.path();
    if (!entry_path.filename().string().ends_with(SAFETENSORS_SUFFIX)) {
      continue;
    }
    if (!entry.is_regular_file(error)) {
      if (error) {
        throw ConfigurationException(entry_path.string() + ": " + error.message());
      }
      continue;
    }
    ++file_count;

    const auto filename = entry_path.filename().string();
    std::optional<bool> filename_is_model;
    if (filename == "model.safetensors") {
      filename_is_model = true;
    } else if (filename == "consolidated.safetensors") {
      filename_is_model = false;
    }

    std::smatch match;
    if (!filename_is_model.has_value() && std::regex_match(filename, match, shard_pattern)) {
      filename_is_model = match[1].str() == "model";
      size_t index = 0;
      size_t total = 0;
      try {
        const auto parsed_index = std::stoull(match[2].str());
        const auto parsed_total = std::stoull(match[3].str());
        if (parsed_index > std::numeric_limits<size_t>::max() || parsed_total > std::numeric_limits<size_t>::max()) {
          throw std::out_of_range{"shard number does not fit size_t"};
        }
        index = static_cast<size_t>(parsed_index);
        total = static_cast<size_t>(parsed_total);
      } catch (const std::exception &) {
        throw ConfigurationException(entry_path.string() + ": shard filename contains an invalid number");
      }
      if (index == 0 || total == 0 || index > total) {
        throw ConfigurationException(entry_path.string() + ": shard filename has an invalid index or total");
      }
      if (shard_total == 0) {
        shard_total = total;
      } else if (shard_total != total) {
        throw ConfigurationException(model_dir_.string() + ": safetensors shards disagree about their total count");
      }
      shards.emplace_back(entry_path, index);
    } else if (filename_is_model.has_value()) {
      if (singleton.has_value()) {
        throw ConfigurationException(model_dir_.string() +
                                     ": both model and consolidated safetensors families are present");
      }
      singleton = entry_path;
    } else {
      has_unrecognized = true;
      unrecognized_file = entry_path;
    }

    if (filename_is_model.has_value() && family_is_model.has_value() && *filename_is_model != *family_is_model) {
      throw ConfigurationException(model_dir_.string() +
                                   ": both model and consolidated safetensors families are present");
    }
    if (filename_is_model.has_value()) {
      family_is_model = filename_is_model;
    }
  }
  if (error) {
    throw ConfigurationException(model_dir_.string() + ": " + error.message());
  }
  if (file_count == 0) {
    throw ConfigurationException(model_dir_.string() + ": no safetensors file found");
  }

  if (!family_is_model.has_value()) {
    if (file_count != 1) {
      throw ConfigurationException(model_dir_.string() + ": multiple unindexed safetensors files are ambiguous");
    }
    mapped_files_.reserve(1);
    ParseSafetensorsFile(unrecognized_file);
    return;
  }
  if (has_unrecognized) {
    throw ConfigurationException(model_dir_.string() + ": unindexed directory contains unrelated safetensors files");
  }
  if (singleton.has_value()) {
    if (!shards.empty()) {
      throw ConfigurationException(model_dir_.string() +
                                   ": a singleton and a sharded safetensors family cannot be combined");
    }
    mapped_files_.reserve(1);
    ParseSafetensorsFile(*singleton);
    return;
  }
  if (shards.empty()) {
    throw ConfigurationException(model_dir_.string() + ": safetensors family contains no files");
  }
  if (shards.size() != shard_total) {
    throw ConfigurationException(model_dir_.string() + ": safetensors shard family has a missing shard");
  }
  // Numeric ordering makes the file index stored in ParameterRecord stable
  // regardless of the directory iterator's order.
  std::ranges::sort(shards, [](const auto &left, const auto &right) { return left.second < right.second; });
  for (size_t expected_index = 1; expected_index <= shard_total; ++expected_index) {
    if (shards[expected_index - 1].second != expected_index) {
      throw ConfigurationException(model_dir_.string() + ": safetensors shard family has a duplicate or missing index");
    }
  }
  mapped_files_.reserve(shards.size());
  for (const auto &shard : shards) {
    ParseSafetensorsFile(shard.first);
  }
}

void Checkpoint::ParseSafetensorsFile(const std::filesystem::path &file_path) {
  const auto file_index = mapped_files_.size();
  mapped_files_.emplace_back(file_path);
  const auto &mapped_file = mapped_files_.back();
  const auto bytes = mapped_file.GetBytes();
  if (bytes.size() < SAFETENSORS_PREAMBLE_SIZE) {
    throw ConfigurationException(file_path.string() + ": safetensors file is shorter than its header length prefix");
  }

  // The first eight bytes are a little-endian JSON header length. All offsets
  // below are relative to the data region that follows that header.
  uint64_t header_length = 0;
  for (size_t index = 0; index < sizeof(uint64_t); ++index) {
    header_length |= static_cast<uint64_t>(std::to_integer<uint8_t>(bytes[index])) << (index * 8);
  }
  if (header_length > bytes.size() - SAFETENSORS_PREAMBLE_SIZE) {
    throw ConfigurationException(file_path.string() + ": safetensors header extends beyond the file");
  }
  const auto header_length_size = static_cast<size_t>(header_length);
  // The preceding bounds check proves this addition cannot overflow size_t.
  const auto data_offset = SAFETENSORS_PREAMBLE_SIZE + header_length_size;
  const auto header = ParseJson(bytes.subspan(SAFETENSORS_PREAMBLE_SIZE, header_length_size), file_path);
  if (!header.is_object()) {
    throw ConfigurationException(file_path.string() + ": safetensors header must be a JSON object");
  }

  const auto data_size = bytes.size() - data_offset;
  std::vector<std::pair<size_t, size_t>> ranges;
  ranges.reserve(header.size());
  // Convert each header entry into a compact record used by WeightBuilder.
  for (const auto &[name, description] : header.items()) {
    if (name == "__metadata__") {
      if (!description.is_object()) {
        throw ConfigurationException(file_path.string() + ": safetensors __metadata__ must be an object");
      }
      for (const auto &[metadata_name, metadata_value] : description.items()) {
        if (!metadata_value.is_string()) {
          std::string message{"safetensors metadata '"};
          message.append(metadata_name);
          message.append("' must be a string");
          throw ConfigurationException(file_path.string() + ": " + message);
        }
      }
      continue;
    }
    if (name.empty() || !description.is_object()) {
      throw ConfigurationException(file_path.string() +
                                   ": each safetensors parameter must have a non-empty object description");
    }
    if (!description.contains("dtype") || !description.contains("shape") || !description.contains("data_offsets")) {
      std::string message{"parameter '"};
      message.append(name);
      message.append("' is missing dtype, shape, or data_offsets");
      throw ConfigurationException(file_path.string() + ": " + message);
    }

    const auto dtype = ParseDType(description.at("dtype"), file_path, name);
    const auto shape = ParseShape(description.at("shape"), file_path, name);
    const auto &offsets = description.at("data_offsets");
    if (!offsets.is_array() || offsets.size() != 2) {
      std::string message{"parameter '"};
      message.append(name);
      message.append("' data_offsets must contain exactly two integers");
      throw ConfigurationException(file_path.string() + ": " + message);
    }
    const auto relative_start = ParseNonNegativeInteger(offsets[0], "data_offsets start", file_path);
    const auto relative_end = ParseNonNegativeInteger(offsets[1], "data_offsets end", file_path);
    if (relative_end < relative_start || relative_end > data_size) {
      std::string message{"parameter '"};
      message.append(name);
      message.append("' data_offsets are outside the data buffer");
      throw ConfigurationException(file_path.string() + ": " + message);
    }
    const auto relative_start_size = static_cast<size_t>(relative_start);
    const auto relative_end_size = static_cast<size_t>(relative_end);
    const auto payload_size = relative_end_size - relative_start_size;
    const auto dtype_size = ttl::GetDTypeInfo(dtype).size_bytes_;
    const auto element_count = shape.GetNumElements();
    if (element_count < 0 || static_cast<uint64_t>(element_count) > std::numeric_limits<size_t>::max() / dtype_size) {
      std::string message{"parameter '"};
      message.append(name);
      message.append("' tensor byte size overflows the host size type");
      throw ConfigurationException(file_path.string() + ": " + message);
    }
    const auto expected_size = static_cast<size_t>(element_count) * dtype_size;
    if (payload_size != expected_size) {
      std::string message{"parameter '"};
      message.append(name);
      message.append("' payload size does not match its shape and dtype");
      throw ConfigurationException(file_path.string() + ": " + message);
    }

    // relative_start is bounded by data_size, so this addition is within the
    // mapped file and cannot overflow size_t.
    const auto absolute_offset = data_offset + relative_start_size;
    const auto inserted =
        parameter_index_
            .emplace(
                name,
                ParameterRecord{.info_ = ParameterInfo{.shape_ = shape, .dtype_ = dtype, .size_bytes_ = payload_size},
                                .file_index_ = file_index,
                                .file_offset_ = absolute_offset})
            .second;
    if (!inserted) {
      std::string message{"duplicate parameter '"};
      message.append(name);
      message.push_back('\'');
      throw ConfigurationException(file_path.string() + ": " + message);
    }
    ranges.emplace_back(relative_start_size, relative_end_size);
  }

  // The format permits empty tensors at the same offset. Non-empty ranges,
  // however, must not overlap: accepting overlap would let one parameter alias
  // another parameter's bytes and makes a corrupt checkpoint nondeterministic.
  std::ranges::sort(ranges, [](const auto &left, const auto &right) {
    if (left.first != right.first) {
      return left.first < right.first;
    }
    return left.second < right.second;
  });
  size_t previous_end = 0;
  for (const auto &[start, end] : ranges) {
    if (start < previous_end && end != start) {
      throw ConfigurationException(file_path.string() + ": safetensors parameter ranges overlap");
    }
    previous_end = std::max(previous_end, end);
  }
}

auto Checkpoint::FindParameter(std::string_view full_name) const -> const ParameterRecord * {
  const auto iterator = parameter_index_.find(std::string{full_name});
  return iterator == parameter_index_.end() ? nullptr : &iterator->second;
}

auto Checkpoint::GetParameterBytes(const ParameterRecord &record) const -> std::span<const std::byte> {
  if (record.file_index_ >= mapped_files_.size()) {
    throw InternalException("parameter record refers to an unknown mapped file");
  }
  const auto bytes = mapped_files_[record.file_index_].GetBytes();
  if (record.file_offset_ > bytes.size() || record.info_.size_bytes_ > bytes.size() - record.file_offset_) {
    throw InternalException("parameter record extends beyond its mapped file");
  }
  return bytes.subspan(record.file_offset_, record.info_.size_bytes_);
}

}  // namespace zephyr::weight
