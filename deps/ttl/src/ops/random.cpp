#include "ttl/ops/random.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/generator.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/device_allocator.hpp"
#include "ttl/internal/elementwise_iterator.hpp"
#include "ttl/internal/execution_context.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/random.hpp"
#include "ttl/internal/scratch_arena.hpp"
#include "ttl/internal/storage.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/shape.hpp"
#include "ttl/stream.hpp"
#include "ttl/tensor.hpp"

namespace ttl::internal {

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

namespace ttl {
namespace {

void ValidateRandomDType(DType dtype, std::source_location location) {
  if (!IsFloating(dtype)) {
    throw NotSupportedError("random operators support only floating-point outputs", location);
  }
}

[[nodiscard]] auto CheckedFloatParameter(double value, std::string_view description, std::source_location location)
    -> float {
  const auto converted = static_cast<float>(value);
  if (!std::isfinite(converted)) {
    throw InvalidArgumentError(std::string{description} + " must be representable as finite FLOAT32", location);
  }
  return converted;
}

void ValidateUniformOptions(const UniformOptions &options, std::source_location location) {
  if (!std::isfinite(options.low_) || !std::isfinite(options.high_) || options.low_ >= options.high_) {
    throw InvalidArgumentError("Uniform requires finite low < high", location);
  }
  const auto low = CheckedFloatParameter(options.low_, "Uniform low", location);
  const auto high = CheckedFloatParameter(options.high_, "Uniform high", location);
  if (low >= high || !std::isfinite(high - low)) {
    throw InvalidArgumentError("Uniform bounds must define a non-empty finite FLOAT32 interval", location);
  }
}

void ValidateNormalOptions(const NormalOptions &options, std::source_location location) {
  if (!std::isfinite(options.mean_) || !std::isfinite(options.standard_deviation_) ||
      options.standard_deviation_ < 0.0) {
    throw InvalidArgumentError("Normal requires a finite mean and non-negative finite standard deviation", location);
  }
  static_cast<void>(CheckedFloatParameter(options.mean_, "Normal mean", location));
  static_cast<void>(CheckedFloatParameter(options.standard_deviation_, "Normal standard deviation", location));
}

void ValidateGeneratorContext(const internal::GeneratorImpl &impl, const ExecutionContext &context,
                              std::source_location location) {
  if (context.GetDevice() != impl.device_ || context.GetStream().GetId() != impl.stream_id_) {
    throw InvalidArgumentError("Generator must be used with the execution context stream that created it", location);
  }
}

[[nodiscard]] auto BuildRandomParameters(Tensor &output, float first_parameter, float second_parameter,
                                         std::source_location location) -> internal::RandomParameters {
  const auto element_size = GetDTypeSize(output.GetDType(), location);
  auto parameters = internal::RandomParameters{
      .output_ = static_cast<std::byte *>(internal::TensorAccess::GetMutableData(output, location)),
      .num_elements_ = static_cast<uint64_t>(output.GetNumElements()),
      .rank_ = static_cast<uint8_t>(output.GetRank()),
      .first_parameter_ = first_parameter,
      .second_parameter_ = second_parameter,
  };
  for (size_t axis = 0; axis < output.GetRank(); ++axis) {
    parameters.shape_[axis] = static_cast<uint64_t>(output.GetShape().GetDimension(axis, location));
    parameters.output_strides_bytes_[axis] =
        internal::CheckedBytes(output.GetStrides().GetStride(axis, location), element_size, location);
  }
  return parameters;
}

void RandomOutImpl(ExecutionContext &context, Tensor &output, Generator &generator,
                   internal::RandomDistribution distribution, double first_parameter, double second_parameter,
                   std::string_view operation, std::source_location location) {
  internal::OpGuard guard{context, operation, location};
  if (distribution == internal::RandomDistribution::UNIFORM) {
    ValidateUniformOptions(UniformOptions{.low_ = first_parameter, .high_ = second_parameter}, location);
  } else {
    ValidateNormalOptions(NormalOptions{.mean_ = first_parameter, .standard_deviation_ = second_parameter}, location);
  }
  guard.ValidateTensor(output);
  ValidateRandomDType(output.GetDType(), location);
  internal::ValidateWritableOutput(output, operation, location);
  auto &generator_impl = internal::GeneratorAccess::GetImpl(generator, location);
  ValidateGeneratorContext(generator_impl, context, location);
  internal::GeneratorUseGuard generator_guard{generator_impl, location};
  if (output.GetNumElements() == 0) {
    return;
  }

  auto scratch = guard.MakeScratchScope();
  auto *base_counter =
      static_cast<uint64_t *>(scratch.AllocateBytes(sizeof(uint64_t), alignof(uint64_t), location).GetData());
  const auto parameters = BuildRandomParameters(output, static_cast<float>(first_parameter),
                                                static_cast<float>(second_parameter), location);
  const auto error_context = guard.RegisterDeviceError(DType::INT64, DType::INT64);
  generator_impl.storage_->RecordUsage(context.GetStream());
  guard.RecordTensor(output);
  internal::LaunchRandom(guard.GetNativeStream(), output.GetDType(), distribution, parameters,
                         static_cast<internal::GeneratorState *>(generator_impl.storage_->GetBasePointer()),
                         base_counter, error_context, location);
  guard.CheckLaunch();
}

}  // namespace

Generator::Generator(ExecutionContext &context, uint64_t seed, std::source_location location) {
  internal::OpGuard guard{context, "Generator", location};
  const auto &allocator = internal::ContextAccess::GetAllocator(context, location);
  auto storage =
      allocator->Allocate(context.GetStream(), sizeof(internal::GeneratorState), alignof(internal::GeneratorState),
                          internal::AllocationContext{
                              .operation_ = "Generator state",
                              .output_shape_ = std::nullopt,
                              .dtype_ = std::nullopt,
                              .location_ = location,
                          });
  internal::LaunchInitializeGenerator(
      guard.GetNativeStream(), static_cast<internal::GeneratorState *>(storage->GetBasePointer()), seed, location);
  guard.CheckLaunch();
  impl_ = std::make_unique<internal::GeneratorImpl>(std::move(storage), context.GetDevice(),
                                                    context.GetStream().GetId(), seed);
}

Generator::Generator(Generator &&) noexcept = default;

auto Generator::operator=(Generator &&) noexcept -> Generator & = default;

Generator::~Generator() noexcept = default;

auto Generator::GetSeed() const noexcept -> uint64_t {
  return impl_ == nullptr ? uint64_t{0} : impl_->seed_.load(std::memory_order_acquire);
}

void Generator::SetSeed(ExecutionContext &context, uint64_t seed, std::source_location location) {
  internal::OpGuard guard{context, "Generator::SetSeed", location};
  auto &impl = internal::GeneratorAccess::GetImpl(*this, location);
  ValidateGeneratorContext(impl, context, location);
  internal::GeneratorUseGuard generator_guard{impl, location};
  impl.storage_->RecordUsage(context.GetStream());
  internal::LaunchInitializeGenerator(guard.GetNativeStream(),
                                      static_cast<internal::GeneratorState *>(impl.storage_->GetBasePointer()), seed,
                                      location);
  guard.CheckLaunch();
  impl.seed_.store(seed, std::memory_order_release);
}

void UniformOut(ExecutionContext &context, Tensor &output, Generator &generator, const UniformOptions &options,
                std::source_location location) {
  RandomOutImpl(context, output, generator, internal::RandomDistribution::UNIFORM, options.low_, options.high_,
                "UniformOut", location);
}

auto Uniform(ExecutionContext &context, const Shape &shape, DType dtype, Generator &generator,
             const UniformOptions &options, std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, "Uniform", location};
    ValidateUniformOptions(options, location);
    ValidateRandomDType(dtype, location);
    const auto &generator_impl = internal::GeneratorAccess::GetImpl(generator, location);
    ValidateGeneratorContext(generator_impl, context, location);
  }
  auto output = Empty(context, shape, dtype, location);
  UniformOut(context, output, generator, options, location);
  return output;
}

void NormalOut(ExecutionContext &context, Tensor &output, Generator &generator, const NormalOptions &options,
               std::source_location location) {
  RandomOutImpl(context, output, generator, internal::RandomDistribution::NORMAL, options.mean_,
                options.standard_deviation_, "NormalOut", location);
}

auto Normal(ExecutionContext &context, const Shape &shape, DType dtype, Generator &generator,
            const NormalOptions &options, std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, "Normal", location};
    ValidateNormalOptions(options, location);
    ValidateRandomDType(dtype, location);
    const auto &generator_impl = internal::GeneratorAccess::GetImpl(generator, location);
    ValidateGeneratorContext(generator_impl, context, location);
  }
  auto output = Empty(context, shape, dtype, location);
  NormalOut(context, output, generator, options, location);
  return output;
}

}  // namespace ttl
