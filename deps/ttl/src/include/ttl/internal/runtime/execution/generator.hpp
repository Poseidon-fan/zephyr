#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <source_location>
#include <utility>

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/internal/runtime/memory/device/storage.hpp"
#include "ttl/runtime/generator.hpp"

namespace ttl::internal {

/**
 * @brief Shared implementation of a Generator bound to one execution-context stream.
 *
 * storage contains the stream-ordered Philox state. seed is an atomic host-side snapshot for observers, while in_use
 * rejects concurrent mutation that would make reservation order nondeterministic.
 */
class GeneratorImpl final {
 public:
  GeneratorImpl(std::shared_ptr<Storage> storage, Device device, uint64_t stream_id, uint64_t seed) noexcept
      : storage_(std::move(storage)), device_(device), stream_id_(stream_id), seed_(seed) {}

  std::shared_ptr<Storage> storage_;
  Device device_;
  uint64_t stream_id_;
  std::atomic<uint64_t> seed_;
  std::atomic_flag in_use_ = ATOMIC_FLAG_INIT;
};

/** @brief Exclusive host-side lease for a generator submission or reseed operation. */
class GeneratorUseGuard final {
 public:
  GeneratorUseGuard(GeneratorImpl &impl, std::source_location location) : impl_(impl) {
    if (impl_.in_use_.test_and_set(std::memory_order_acquire)) {
      throw InvalidArgumentError("Generator is already in use", location);
    }
  }

  GeneratorUseGuard(const GeneratorUseGuard &) = delete;
  auto operator=(const GeneratorUseGuard &) -> GeneratorUseGuard & = delete;
  ~GeneratorUseGuard() noexcept { impl_.in_use_.clear(std::memory_order_release); }

 private:
  GeneratorImpl &impl_;
};

class GeneratorAccess final {
 public:
  [[nodiscard]] static auto GetImpl(Generator &generator, std::source_location location) -> GeneratorImpl & {
    if (generator.impl_ == nullptr) {
      throw InvalidArgumentError("Generator is in a moved-from state", location);
    }
    return *generator.impl_;
  }
};

}  // namespace ttl::internal
