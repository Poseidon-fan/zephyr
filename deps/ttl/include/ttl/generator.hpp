#pragma once

#include <concepts>
#include <cstdint>
#include <memory>
#include <source_location>
#include <type_traits>

#include "ttl/execution_context.hpp"

namespace ttl::internal {

class GeneratorAccess;
class GeneratorImpl;

}  // namespace ttl::internal

namespace ttl {

/** Move-only Philox generator bound to one execution context stream. */
class Generator final {
 public:
  explicit Generator(ExecutionContext &context, uint64_t seed = 0,
                     std::source_location location = std::source_location::current());

  Generator(const Generator &) = delete;
  auto operator=(const Generator &) -> Generator & = delete;
  Generator(Generator &&) noexcept;
  auto operator=(Generator &&) noexcept -> Generator &;
  ~Generator() noexcept;

  [[nodiscard]] auto GetSeed() const noexcept -> uint64_t;
  void SetSeed(ExecutionContext &context, uint64_t seed,
               std::source_location location = std::source_location::current());

 private:
  friend class internal::GeneratorAccess;

  std::unique_ptr<internal::GeneratorImpl> impl_;
};

static_assert(!std::default_initializable<Generator>);
static_assert(!std::copy_constructible<Generator>);
static_assert(std::is_nothrow_move_constructible_v<Generator>);
static_assert(std::is_nothrow_move_assignable_v<Generator>);

}  // namespace ttl
