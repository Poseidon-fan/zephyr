#include "executor/factory.hpp"

#include <type_traits>
#include <utility>
#include <variant>

#include "executor/embedding.hpp"

namespace zephyr::executor {

auto CreateExecutionFactory(model::ResolvedModelLoader loader, CausalLMOptions options) -> ExecutionFactory {
  return std::visit(
      [options](auto typed_loader) -> ExecutionFactory {
        if constexpr (std::is_same_v<decltype(typed_loader), model::ModelLoader<model::causal_lm::CausalLM>>) {
          return CreateCausalLMFactory(std::move(typed_loader), options);
        } else {
          return CreateEmbeddingFactory(std::move(typed_loader));
        }
      },
      std::move(loader));
}

}  // namespace zephyr::executor
