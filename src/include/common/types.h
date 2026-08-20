#pragma once

#include <cstdint>

namespace zephyr {

/** Identifies one Worker in the process-wide rank space. */
using rank_t = int32_t;

/** Identifies one independent data-parallel replica. */
using replica_id_t = int32_t;

/** Identifies one sequence throughout scheduling, execution, and KV-cache management. */
using sequence_id_t = uint64_t;

/** Identifies one logical K/V storage entry in an executable model. */
using kv_layer_id_t = uint32_t;

}  // namespace zephyr
