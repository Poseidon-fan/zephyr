#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ttl/tensor/dtype.hpp>
#include <ttl/tensor/shape.hpp>

namespace zephyr::weight {

/** Metadata needed to validate and materialize one checkpoint parameter. */
struct ParameterInfo {
  /** Logical row-major dimensions recorded by safetensors. */
  ttl::Shape shape_;
  /** Storage dtype recorded by safetensors. */
  ttl::DType dtype_;
  /** Number of payload bytes occupied by the parameter in its source file. */
  size_t size_bytes_;

  [[nodiscard]] auto operator==(const ParameterInfo &) const noexcept -> bool = default;
};

class WeightBuilder;

/**
 * Read-only, memory-mapped safetensors files belonging to one model directory.
 *
 * Construction discovers either one safetensors index or one unambiguous set of
 * standard safetensors files. The object owns every mapping for its lifetime;
 * callers can therefore build asynchronous device transfers without copying the
 * source files into an intermediate heap buffer.
 */
class Checkpoint {
 public:
  explicit Checkpoint(std::filesystem::path model_dir);

  ~Checkpoint() noexcept = default;

  Checkpoint(const Checkpoint &) = delete;
  auto operator=(const Checkpoint &) -> Checkpoint & = delete;
  Checkpoint(Checkpoint &&) = delete;
  auto operator=(Checkpoint &&) -> Checkpoint & = delete;

  /** Return metadata for a fully qualified parameter name, or nullopt if absent. */
  [[nodiscard]] auto GetParameterInfo(std::string_view full_name) const -> std::optional<ParameterInfo>;

 private:
  friend class WeightBuilder;

  /** One read-only POSIX mapping. The type is private because mappings are checkpoint-owned resources. */
  class MappedFile {
   public:
    explicit MappedFile(const std::filesystem::path &path);

    ~MappedFile() noexcept;

    MappedFile(const MappedFile &) = delete;
    auto operator=(const MappedFile &) -> MappedFile & = delete;
    MappedFile(MappedFile &&other) noexcept;
    auto operator=(MappedFile &&other) noexcept -> MappedFile & = delete;

    /** Return the complete mapped file as immutable bytes. */
    [[nodiscard]] auto GetBytes() const noexcept -> std::span<const std::byte> {
      return {static_cast<const std::byte *>(mapping_), size_bytes_};
    }
    /** Return the path used to open this mapping for diagnostics and index checks. */
    [[nodiscard]] auto GetPath() const noexcept -> const std::filesystem::path & { return path_; }

   private:
    /** Path retained for diagnostics and index routing checks. */
    std::filesystem::path path_;
    /** Address returned by mmap, or nullptr for an empty file. */
    void *mapping_{nullptr};
    /** Number of bytes in the mapping. */
    size_t size_bytes_{0};
  };

  /** Locate one parameter's bytes inside one of the owned mappings. */
  struct ParameterRecord {
    /** Shape, dtype, and payload size from the safetensors header. */
    ParameterInfo info_;
    /** Position in mapped_files_. */
    size_t file_index_;
    /** Absolute byte offset from the beginning of the mapped file. */
    size_t file_offset_;
  };

  /** Parse an index and load exactly the files named by its weight_map. */
  void LoadWithIndex(const std::filesystem::path &index_path);
  /** Discover and load a single file or a complete standard shard family. */
  void LoadWithoutIndex();
  /** Parse one safetensors header and add its records to parameter_index_. */
  void ParseSafetensorsFile(const std::filesystem::path &file_path);

  [[nodiscard]] auto FindParameter(std::string_view full_name) const -> const ParameterRecord *;
  [[nodiscard]] auto GetParameterBytes(const ParameterRecord &record) const -> std::span<const std::byte>;

  /** Directory containing the checkpoint and optional index. */
  std::filesystem::path model_dir_;
  /** All source files kept mapped for the lifetime of the checkpoint. */
  std::vector<MappedFile> mapped_files_;
  /** Fully qualified parameter name to its source location and metadata. */
  std::unordered_map<std::string, ParameterRecord> parameter_index_;
};

}  // namespace zephyr::weight
