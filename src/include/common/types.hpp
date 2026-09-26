#pragma once

#include <cstdint>

namespace zephyr {

/** Token vocabulary index, matching INT32 model inputs. */
using token_id_t = int32_t;

/** Identifies a live inference sequence across the engine. */
using sequence_id_t = uint64_t;

/** Identifies one submitted request, which may own several inference sequences. */
using request_id_t = uint64_t;

}  // namespace zephyr
