#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>

#include <cublasLt.h>

namespace ttl::internal {

struct MatrixLayoutSignature final {
  uint64_t rows_;
  uint64_t columns_;
  int64_t leading_dimension_;
  int64_t batch_stride_;
  int32_t order_;

  [[nodiscard]] auto operator==(const MatrixLayoutSignature &) const noexcept -> bool = default;
};

struct MatmulAlgorithmKey final {
  MatrixLayoutSignature lhs_;
  MatrixLayoutSignature rhs_;
  MatrixLayoutSignature output_;
  int64_t batch_count_;
  int32_t data_type_;
  int32_t compute_type_;
  int32_t lhs_operation_;
  int32_t rhs_operation_;
  int32_t epilogue_;
  uint32_t lhs_alignment_;
  uint32_t rhs_alignment_;
  uint32_t output_alignment_;
  uint32_t bias_alignment_;
  size_t workspace_limit_bytes_;

  [[nodiscard]] auto operator==(const MatmulAlgorithmKey &) const noexcept -> bool = default;
};

struct MatmulAlgorithmChoice final {
  cublasLtMatmulAlgo_t algorithm_;
  size_t workspace_bytes_;
  bool supported_;
};

/** Per-device bounded LRU of immutable cuBLASLt heuristic choices. */
class MatmulAlgorithmCache final {
 public:
  [[nodiscard]] auto Find(const MatmulAlgorithmKey &key) -> std::optional<MatmulAlgorithmChoice>;
  void Insert(const MatmulAlgorithmKey &key, const MatmulAlgorithmChoice &choice);

 private:
  struct KeyHash final {
    [[nodiscard]] auto operator()(const MatmulAlgorithmKey &key) const noexcept -> size_t;
  };

  struct Entry final {
    MatmulAlgorithmKey key_;
    MatmulAlgorithmChoice choice_;
  };

  static constexpr size_t MAXIMUM_ENTRY_COUNT = 256;

  using Entries = std::list<Entry>;
  using Index = std::unordered_map<MatmulAlgorithmKey, Entries::iterator, KeyHash>;

  std::mutex latch_;
  Entries entries_;
  Index index_;
};

}  // namespace ttl::internal
