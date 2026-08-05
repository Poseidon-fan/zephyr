# TTL Memory Management

## Ownership

Device allocations are owned by `Storage`, which retains the per-device allocator and a
`DeviceStreamUsage` tracker. A non-empty pool allocation also retains a budget reservation and a
retirement ticket before the `Storage` becomes visible to callers. `Storage::~Storage()` moves the
allocation and its strongly typed stream snapshot into a `DeviceRetirement` record; it never calls
CUDA free directly.

Pinned allocations are owned by `PinnedBlock`, behind one or more copyable `PinnedBuffer` handles.
The block retains the process-wide pinned allocator, a capacity budget reservation, a retirement
ticket, and a `PinnedStreamUsage` tracker. The last block owner moves entries that bind every
distinct stream to one optional completion-event slot into `PinnedRetirement`.

Borrowed external device memory has no retirement ticket and is never released by TTL. Shared-owner
external memory has a retirement ticket, waits for recorded stream usage, and releases its
`shared_ptr<void>` only after retirement.

## Common invariants

`AllocationBudget::Reservation` is move-only. Its destructor releases exactly the capacity reserved
by the allocation. The reservation remains alive while a resource is live or retiring and is
released only after retirement completes. Budget release checks for underflow and terminates on an
internal accounting violation.

`RetirementQueue<Record>` reserves ring slots before publishing an owner. `RetirementTicket` binds
that reservation to one queue, so an owner-construction rollback cancels its own reservation and a
`noexcept` destructor can consume only the corresponding ticket. `Checkout` temporarily owns the
oldest available record; destruction appends it to the ring without allocation, while `Complete`
removes it from pending accounting. A poll processes a count snapshot, so records enqueued
concurrently cannot displace older snapshot records.

The queue mutex protects only record-container state. Native CUDA queries, event synchronization,
host free, and cache operations run after checkout, outside that mutex. A failed native operation
poisons the record and changes allocator state to `FAILED`; the record remains reachable until
explicit `Shutdown` performs a conservative synchronized drain.

`AllocatorLifecycle` is the common state machine. New work requires `RUNNING`; `FAILED` rejects
new work but permits polling and shutdown; `CLOSING` rejects admission while allowing a retryable
shutdown; `CLOSED` is terminal.

## Device path

Each registered device owns one `CudaMemoryPool` and one `DeviceAllocator`. The pool wrapper owns
native pool creation, reuse attributes, peer access, trim, pool statistics, and destruction. The
allocator owns TTL budget admission, `Storage` construction, OOM retry policy, side-stream
dependency submission, reclaim-stream free submission, completion-event polling, maintenance
worker, and external ownership semantics.

When side streams exist, TTL records a dependency event on every usage stream, makes the reclaim
stream wait for them, submits `cudaFreeAsync` on the reclaim stream, and records a completion event
after the free. When no side stream exists, the allocation stream is the free and completion stream.
Polling is nonblocking; only shutdown synchronizes poisoned usage streams or an ambiguous free
submission.

## Pinned path

`PinnedMemoryCache` owns the process-wide size-class map and all
`cudaHostAllocPortable`/`cudaFreeHost` calls. Requests are rounded to 4 KiB and then to a power-of-two
class up to 64 MiB. Larger requests remain page aligned. The cache is keyed by physical capacity,
while `PinnedBlock` preserves the caller's logical byte count.

Cache trim swaps the map under the cache mutex, frees native allocations outside the mutex, and
restores remaining map nodes without allocation if a native free fails. Cache misses retry once
after a trim on `cudaErrorMemoryAllocation`.

Pinned usage can span devices. Each usage entry owns its stream and optional completion event,
preventing stream/event index drift. Retirement fills one event per distinct usage stream, queries
every event, and caches or frees the host allocation only when all events report completion. A
poisoned record synchronizes every usage stream during explicit shutdown rather than trusting an
incomplete event set.

The device-error host mirror is an ordinary runtime-managed `PinnedBuffer`. The D2H snapshot records
the primary stream, so context teardown and synchronization failures use the same retirement and
quarantine protocol as user pinned buffers.

## Statistics

Pinned `logical_live_bytes_` counts bytes requested by live `PinnedBuffer` owners, while
`live_capacity_bytes_` counts their rounded size-class capacity. `retiring_capacity_bytes_` remains
protected by asynchronous work, and `cached_capacity_bytes_` is immediately reusable.
`budgeted_capacity_bytes_` is live plus retiring capacity; `physical_bytes_` additionally includes
cached native host allocations.

Device `logical_live_bytes_` counts live pool requests, `retiring_bytes_` counts capacity awaiting
completion, and `pool_reserved_bytes_` includes pages retained by CUDA below the release threshold.

## Synchronization boundaries

Allocation, retirement submission, polling, and trimming do not synchronize running GPU work.
`Poll` only queries native completion state. `Shutdown` is the explicit blocking boundary: it rejects
new admission, verifies all public owners are released, drains pending records, trims native caches
and pools, and closes dependent event resources in runtime order.

External native streams remain caller-owned. A caller that records usage through an external stream
must keep the native stream alive until TTL retirement completes; `StreamState` retains only the TTL
side-stream metadata and optional owner.
