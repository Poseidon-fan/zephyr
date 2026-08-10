#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/distributed/collective.hpp"
#include "ttl/distributed/communicator.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/runtime/graph.hpp"
#include "ttl/tensor/layout.hpp"

namespace ttl::test {
namespace {

class DistributedTest : public MultiDeviceTest {};

TEST_F(DistributedTest, PeerCopyHonorsExplicitProducerEventAcrossDevices) {
  ASSERT_GE(GetDevices().size(), 2U);
  if (!GetRuntime().CanAccessPeer(GetDevices()[1], GetDevices()[0])) {
    GTEST_SKIP() << "requires destination-to-source peer access";
  }
  ExecutionContext producer = GetRuntime().CreateExecutionContext(GetDevices()[0]);
  ExecutionContext consumer = GetRuntime().CreateExecutionContext(GetDevices()[1]);
  Tensor source = TensorFromValues<int32_t>(producer, Shape{4}, {3, 1, 4, 1});
  Event source_ready = producer.RecordEvent();
  Tensor destination = Empty(consumer, Shape{4}, DType::INT32);

  CopyPeerOut(consumer, destination, source, source_ready);
  consumer.Synchronize();
  ExpectValues<int32_t>(consumer, destination, {3, 1, 4, 1});

  Tensor wrong_destination = Empty(consumer, Shape{3}, DType::INT32);
  EXPECT_THROW(CopyPeerOut(consumer, wrong_destination, source, source_ready), InvalidArgumentError);

  Tensor wrong_dtype = Empty(consumer, Shape{4}, DType::FLOAT32);
  EXPECT_THROW(CopyPeerOut(consumer, wrong_dtype, source, source_ready), InvalidArgumentError);

  Tensor destination_storage = Empty(consumer, Shape{8}, DType::INT32);
  Tensor non_contiguous = Slice(destination_storage, 0, 0, 8, 2);
  ASSERT_FALSE(non_contiguous.IsContiguous());
  EXPECT_THROW(CopyPeerOut(consumer, non_contiguous, source, source_ready), InvalidArgumentError);

  Tensor same_device_destination = Empty(producer, Shape{4}, DType::INT32);
  EXPECT_THROW(CopyPeerOut(producer, same_device_destination, source, source_ready), InvalidArgumentError);

  Event wrong_device_ready = consumer.RecordEvent();
  EXPECT_THROW(CopyPeerOut(consumer, destination, source, wrong_device_ready), InvalidArgumentError);
}

TEST_F(DistributedTest, PeerCopyRejectsUnavailablePeerAccessWhenExposedByTheTopology) {
  std::optional<std::pair<Device, Device>> unsupported_pair;
  for (Device destination : GetDevices()) {
    for (Device source : GetDevices()) {
      if (destination != source && !GetRuntime().CanAccessPeer(destination, source)) {
        unsupported_pair.emplace(destination, source);
        break;
      }
    }
    if (unsupported_pair.has_value()) {
      break;
    }
  }
  if (!unsupported_pair.has_value()) {
    GTEST_SKIP() << "all discovered device pairs support peer access";
  }

  ExecutionContext destination_context = GetRuntime().CreateExecutionContext(unsupported_pair->first);
  ExecutionContext source_context = GetRuntime().CreateExecutionContext(unsupported_pair->second);
  Tensor source = TensorFromValues<int32_t>(source_context, Shape{2}, {1, 2});
  Event source_ready = source_context.RecordEvent();
  Tensor destination = Empty(destination_context, Shape{2}, DType::INT32);
  EXPECT_THROW(CopyPeerOut(destination_context, destination, source, source_ready), NotSupportedError);
}

[[nodiscard]] auto MakeRankTensor(ExecutionContext &context, int32_t rank, int64_t length = 2) -> Tensor {
  std::vector<float> values(static_cast<size_t>(length));
  for (int64_t index = 0; index < length; ++index) {
    values[static_cast<size_t>(index)] = static_cast<float>((rank * 10) + index);
  }
  return FloatingTensorFromValues(context, Shape{length}, DType::FLOAT32, values);
}

TEST_F(DistributedTest, CommunicatorGroupLifecycleAndRankMetadataAreAllOrNothing) {
  LocalCommunicatorGroup group = LocalCommunicatorGroup::Create(GetRuntime(), GetDevices());
  ASSERT_EQ(group.GetWorldSize(), GetDevices().size());
  const auto rank_order = group.GetRankOrder();
  ASSERT_EQ(rank_order.size(), GetDevices().size());
  for (size_t rank = 0; rank < rank_order.size(); ++rank) {
    EXPECT_EQ(rank_order[rank], GetDevices()[rank]);
  }
  EXPECT_EQ(group.GetStatus(), CommunicatorStatus::READY);
  for (size_t rank = 0; rank < GetDevices().size(); ++rank) {
    NcclCommunicator &communicator = group.GetCommunicator(rank);
    EXPECT_EQ(communicator.GetRank(), static_cast<int32_t>(rank));
    EXPECT_EQ(communicator.GetWorldSize(), static_cast<int32_t>(GetDevices().size()));
    EXPECT_EQ(communicator.GetDevice(), GetDevices()[rank]);
    EXPECT_EQ(communicator.GetStatus(), CommunicatorStatus::READY);
  }
  EXPECT_THROW(static_cast<void>(group.GetCommunicator(GetDevices().size())), InvalidArgumentError);
  group.Poll();
  EXPECT_NO_THROW(group.Close());
  EXPECT_EQ(group.GetStatus(), CommunicatorStatus::CLOSED);
  EXPECT_NO_THROW(group.Close());
}

TEST_F(DistributedTest, ReductionCollectivesCoverEverySupportedDTypeAndOperationAcrossTheDiscoveredWorld) {
  const size_t world = GetDevices().size();
  LocalCommunicatorGroup group = LocalCommunicatorGroup::Create(GetRuntime(), GetDevices());

  const auto run_integer_all_reduce = [&]<TensorStorageType T>() {
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    std::vector<LocalCollectiveCall> calls;
    inputs.reserve(world);
    outputs.reserve(world);
    calls.reserve(world);
    for (size_t rank = 0; rank < world; ++rank) {
      const auto rank_value = static_cast<T>(rank + 1);
      inputs.push_back(TensorFromValues<T>(GetContexts()[rank], Shape{2}, {rank_value, static_cast<T>(2)}));
      outputs.push_back(Empty(GetContexts()[rank], Shape{2}, DTYPE_OF<T>));
    }
    for (size_t rank = 0; rank < world; ++rank) {
      calls.push_back(LocalCollectiveCall{.context_ = &GetContexts()[rank],
                                          .output_ = &outputs[rank],
                                          .input_ = &inputs[rank],
                                          .communicator_ = &group.GetCommunicator(rank)});
    }
    const auto verify = [&](ReduceOp operation, T expected_first, T expected_second) {
      AllReduceLocal(calls, operation);
      for (size_t rank = 0; rank < world; ++rank) {
        GetContexts()[rank].Synchronize();
        EXPECT_EQ(TensorToValues<T>(GetContexts()[rank], outputs[rank]),
                  (std::vector<T>{expected_first, expected_second}));
      }
    };
    verify(ReduceOp::SUM, static_cast<T>((world * (world + 1)) / 2), static_cast<T>(world * 2));
    verify(ReduceOp::MINIMUM, static_cast<T>(1), static_cast<T>(2));
    verify(ReduceOp::MAXIMUM, static_cast<T>(world), static_cast<T>(2));
  };

  run_integer_all_reduce.template operator()<uint8_t>();
  run_integer_all_reduce.template operator()<int32_t>();
  run_integer_all_reduce.template operator()<int64_t>();

  for (DType dtype : {DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    std::vector<LocalCollectiveCall> calls;
    inputs.reserve(world);
    outputs.reserve(world);
    calls.reserve(world);
    for (size_t rank = 0; rank < world; ++rank) {
      inputs.push_back(
          FloatingTensorFromValues(GetContexts()[rank], Shape{2}, dtype, {static_cast<float>(rank + 1), 2.0F}));
      outputs.push_back(Empty(GetContexts()[rank], Shape{2}, dtype));
      calls.push_back(LocalCollectiveCall{.context_ = &GetContexts()[rank],
                                          .output_ = &outputs[rank],
                                          .input_ = &inputs[rank],
                                          .communicator_ = &group.GetCommunicator(rank)});
    }
    const auto verify = [&](ReduceOp operation, float expected_first, float expected_second) {
      AllReduceLocal(calls, operation);
      for (size_t rank = 0; rank < world; ++rank) {
        GetContexts()[rank].Synchronize();
        ExpectFloatValues(GetContexts()[rank], outputs[rank], {expected_first, expected_second}, 0.01F, 0.01F);
      }
    };
    verify(ReduceOp::SUM, (static_cast<float>(world) * static_cast<float>(world + 1)) / 2.0F,
           static_cast<float>(world * 2));
    verify(ReduceOp::MINIMUM, 1.0F, 2.0F);
    verify(ReduceOp::MAXIMUM, static_cast<float>(world), 2.0F);
  }

  std::vector<Tensor> bool_inputs;
  std::vector<Tensor> bool_outputs;
  std::vector<LocalCollectiveCall> bool_calls;
  bool_inputs.reserve(world);
  bool_outputs.reserve(world);
  bool_calls.reserve(world);
  for (size_t rank = 0; rank < world; ++rank) {
    bool_inputs.push_back(BoolTensorFromValues(GetContexts()[rank], Shape{2},
                                               {static_cast<uint8_t>(rank != 0), static_cast<uint8_t>(rank == 0)}));
    bool_outputs.push_back(Empty(GetContexts()[rank], Shape{2}, DType::BOOL));
    bool_calls.push_back(LocalCollectiveCall{.context_ = &GetContexts()[rank],
                                             .output_ = &bool_outputs[rank],
                                             .input_ = &bool_inputs[rank],
                                             .communicator_ = &group.GetCommunicator(rank)});
  }
  EXPECT_THROW(AllReduceLocal(bool_calls, ReduceOp::SUM), InvalidArgumentError);
  AllReduceLocal(bool_calls, ReduceOp::MINIMUM);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
    ExpectBoolValues(GetContexts()[rank], bool_outputs[rank], {0, 0});
  }
  AllReduceLocal(bool_calls, ReduceOp::MAXIMUM);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
    ExpectBoolValues(GetContexts()[rank], bool_outputs[rank], {1, 1});
  }
  group.Close();
}

TEST_F(DistributedTest, LocalCollectivesImplementReductionGatherBroadcastScatterAndAllToAll) {
  const size_t world = GetDevices().size();
  LocalCommunicatorGroup group = LocalCommunicatorGroup::Create(GetRuntime(), GetDevices());
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  inputs.reserve(world);
  outputs.reserve(world);
  for (size_t rank = 0; rank < world; ++rank) {
    inputs.push_back(MakeRankTensor(GetContexts()[rank], static_cast<int32_t>(rank)));
    outputs.push_back(Empty(GetContexts()[rank], Shape{2}, DType::FLOAT32));
  }

  std::vector<LocalCollectiveCall> calls;
  calls.reserve(world);
  for (size_t rank = 0; rank < world; ++rank) {
    calls.push_back(LocalCollectiveCall{.context_ = &GetContexts()[rank],
                                        .output_ = &outputs[rank],
                                        .input_ = &inputs[rank],
                                        .communicator_ = &group.GetCommunicator(rank)});
  }

  AllReduceLocal(calls, ReduceOp::SUM);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
    const size_t rank_sum = (world * (world - 1)) / 2;
    const auto first = static_cast<float>(10 * rank_sum);
    ExpectFloatValues(GetContexts()[rank], outputs[rank], {first, first + static_cast<float>(world)});
  }

  for (size_t rank = 0; rank < world; ++rank) {
    FillOut(GetContexts()[rank], outputs[rank], Scalar{0.0});
  }
  BroadcastLocal(calls, 0);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
    ExpectFloatValues(GetContexts()[rank], outputs[rank], {0, 1});
  }

  std::vector<Tensor> gather_inputs;
  std::vector<Tensor> gather_outputs;
  gather_inputs.reserve(world);
  gather_outputs.reserve(world);
  for (size_t rank = 0; rank < world; ++rank) {
    gather_inputs.push_back(MakeRankTensor(GetContexts()[rank], static_cast<int32_t>(rank)));
    gather_outputs.push_back(Empty(GetContexts()[rank], Shape{static_cast<int64_t>(world * 2)}, DType::FLOAT32));
  }
  for (size_t rank = 0; rank < world; ++rank) {
    calls[rank] = LocalCollectiveCall{.context_ = &GetContexts()[rank],
                                      .output_ = &gather_outputs[rank],
                                      .input_ = &gather_inputs[rank],
                                      .communicator_ = &group.GetCommunicator(rank)};
  }
  AllGatherLocal(calls);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
    std::vector<float> expected;
    expected.reserve(world * 2);
    for (size_t source = 0; source < world; ++source) {
      expected.push_back(static_cast<float>(source * 10));
      expected.push_back(static_cast<float>((source * 10) + 1));
    }
    ExpectFloatValues(GetContexts()[rank], gather_outputs[rank], expected);
  }

  std::vector<Tensor> alltoall_inputs;
  std::vector<Tensor> alltoall_outputs;
  alltoall_inputs.reserve(world);
  alltoall_outputs.reserve(world);
  for (size_t rank = 0; rank < world; ++rank) {
    std::vector<float> values(world);
    for (size_t peer = 0; peer < world; ++peer) {
      values[peer] = static_cast<float>((rank * 10) + peer);
    }
    alltoall_inputs.push_back(
        FloatingTensorFromValues(GetContexts()[rank], Shape{static_cast<int64_t>(world)}, DType::FLOAT32, values));
    alltoall_outputs.push_back(Empty(GetContexts()[rank], Shape{static_cast<int64_t>(world)}, DType::FLOAT32));
  }
  for (size_t rank = 0; rank < world; ++rank) {
    calls[rank] = LocalCollectiveCall{.context_ = &GetContexts()[rank],
                                      .output_ = &alltoall_outputs[rank],
                                      .input_ = &alltoall_inputs[rank],
                                      .communicator_ = &group.GetCommunicator(rank)};
  }
  AllToAllLocal(calls);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
    std::vector<float> expected;
    expected.reserve(world);
    for (size_t source = 0; source < world; ++source) {
      expected.push_back(static_cast<float>((source * 10) + rank));
    }
    ExpectFloatValues(GetContexts()[rank], alltoall_outputs[rank], expected);
  }

  EXPECT_NO_THROW(group.Close());
}

TEST_F(DistributedTest, LocalPointToPointAndValidationCoverPeerOrdering) {
  const size_t world = GetDevices().size();
  LocalCommunicatorGroup group = LocalCommunicatorGroup::Create(GetRuntime(), GetDevices());
  std::vector<Tensor> sends;
  std::vector<Tensor> receives;
  std::vector<LocalPointToPointCall> calls;
  sends.reserve(world);
  receives.reserve(world);
  calls.reserve(world);
  for (size_t rank = 0; rank < world; ++rank) {
    sends.push_back(MakeRankTensor(GetContexts()[rank], static_cast<int32_t>(rank)));
    receives.push_back(Empty(GetContexts()[rank], Shape{2}, DType::FLOAT32));
  }
  for (size_t rank = 0; rank < world; ++rank) {
    const auto send_peer = static_cast<int32_t>((rank + 1) % world);
    const auto receive_peer = static_cast<int32_t>((rank + world - 1) % world);
    calls.push_back(LocalPointToPointCall{.context_ = &GetContexts()[rank],
                                          .send_ = &sends[rank],
                                          .send_peer_ = send_peer,
                                          .receive_ = &receives[rank],
                                          .receive_peer_ = receive_peer,
                                          .communicator_ = &group.GetCommunicator(rank)});
  }
  SendReceiveLocal(calls);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
    const size_t source = (rank + world - 1) % world;
    ExpectFloatValues(GetContexts()[rank], receives[rank],
                      {static_cast<float>(source * 10), static_cast<float>((source * 10) + 1)});
  }

  EXPECT_THROW(static_cast<void>(group.GetCommunicator(world)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(AllReduceLocal(std::span<const LocalCollectiveCall>{}, ReduceOp::SUM)),
               InvalidArgumentError);
  EXPECT_NO_THROW(group.Abort());
}

TEST_F(DistributedTest, LocalReductionGatherScatterAndBarrierCoverRootAndChunkSemantics) {
  const size_t world = GetDevices().size();
  LocalCommunicatorGroup group = LocalCommunicatorGroup::Create(GetRuntime(), GetDevices());
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  std::vector<LocalCollectiveCall> calls;
  inputs.reserve(world);
  outputs.reserve(world);
  calls.reserve(world);

  for (size_t rank = 0; rank < world; ++rank) {
    inputs.push_back(MakeRankTensor(GetContexts()[rank], static_cast<int32_t>(rank), static_cast<int64_t>(world)));
    outputs.push_back(Empty(GetContexts()[rank], Shape{1}, DType::FLOAT32));
    calls.push_back(LocalCollectiveCall{.context_ = &GetContexts()[rank],
                                        .output_ = &outputs[rank],
                                        .input_ = &inputs[rank],
                                        .communicator_ = &group.GetCommunicator(rank)});
  }

  ReduceScatterLocal(calls, ReduceOp::SUM);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
    const size_t rank_sum = (world * (world - 1)) / 2;
    const auto expected = static_cast<float>((10 * rank_sum) + (rank * world));
    ExpectFloatValues(GetContexts()[rank], outputs[rank], {expected});
  }

  for (size_t rank = 0; rank < world; ++rank) {
    inputs[rank] = MakeRankTensor(GetContexts()[rank], static_cast<int32_t>(rank));
    outputs[rank] = Empty(GetContexts()[rank], Shape{2}, DType::FLOAT32);
  }
  ReduceLocal(calls, ReduceOp::MAXIMUM, 1);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
  }
  ExpectFloatValues(GetContexts()[1], outputs[1],
                    {static_cast<float>((world - 1) * 10), static_cast<float>(((world - 1) * 10) + 1)});

  std::vector<Tensor> gather_inputs;
  std::vector<Tensor> gather_outputs;
  gather_inputs.reserve(world);
  gather_outputs.reserve(world);
  for (size_t rank = 0; rank < world; ++rank) {
    gather_inputs.push_back(MakeRankTensor(GetContexts()[rank], static_cast<int32_t>(rank)));
    gather_outputs.push_back(Empty(GetContexts()[rank], Shape{static_cast<int64_t>(world * 2)}, DType::FLOAT32));
    calls[rank] = LocalCollectiveCall{.context_ = &GetContexts()[rank],
                                      .output_ = &gather_outputs[rank],
                                      .input_ = &gather_inputs[rank],
                                      .communicator_ = &group.GetCommunicator(rank)};
  }
  GatherLocal(calls, 0);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
  }
  std::vector<float> gathered;
  for (size_t source = 0; source < world; ++source) {
    gathered.push_back(static_cast<float>(source * 10));
    gathered.push_back(static_cast<float>((source * 10) + 1));
  }
  ExpectFloatValues(GetContexts()[0], gather_outputs[0], gathered);

  std::vector<Tensor> scatter_inputs;
  std::vector<Tensor> scatter_outputs;
  scatter_inputs.reserve(world);
  scatter_outputs.reserve(world);
  std::vector<float> root_values(world * 2);
  std::iota(root_values.begin(), root_values.end(), 0.0F);
  for (size_t rank = 0; rank < world; ++rank) {
    scatter_inputs.push_back(rank == 0
                                 ? FloatingTensorFromValues(GetContexts()[rank], Shape{static_cast<int64_t>(world * 2)},
                                                            DType::FLOAT32, root_values)
                                 : Empty(GetContexts()[rank], Shape{static_cast<int64_t>(world * 2)}, DType::FLOAT32));
    scatter_outputs.push_back(Empty(GetContexts()[rank], Shape{2}, DType::FLOAT32));
    calls[rank] = LocalCollectiveCall{.context_ = &GetContexts()[rank],
                                      .output_ = &scatter_outputs[rank],
                                      .input_ = &scatter_inputs[rank],
                                      .communicator_ = &group.GetCommunicator(rank)};
  }
  ScatterLocal(calls, 0);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
    ExpectFloatValues(GetContexts()[rank], scatter_outputs[rank],
                      {static_cast<float>(rank * 2), static_cast<float>((rank * 2) + 1)});
  }

  std::vector<LocalBarrierCall> barriers;
  barriers.reserve(world);
  for (size_t rank = 0; rank < world; ++rank) {
    barriers.push_back(
        LocalBarrierCall{.context_ = &GetContexts()[rank], .communicator_ = &group.GetCommunicator(rank)});
  }
  EXPECT_NO_THROW(BarrierLocal(barriers));
  for (auto &context : GetContexts()) {
    context.Synchronize();
  }
  group.Close();
}

TEST_F(DistributedTest, LocalAllToAllVValidatesAndPreservesVariablePeerChunks) {
  const size_t world = GetDevices().size();
  LocalCommunicatorGroup group = LocalCommunicatorGroup::Create(GetRuntime(), GetDevices());
  std::vector<std::vector<int64_t>> send_counts(world, std::vector<int64_t>(world));
  std::vector<std::vector<int64_t>> receive_counts(world, std::vector<int64_t>(world));
  for (size_t source = 0; source < world; ++source) {
    for (size_t destination = 0; destination < world; ++destination) {
      const size_t diagonal_count = source == destination ? 1 : 0;
      send_counts[source][destination] = static_cast<int64_t>(((source + destination) % 2) + diagonal_count);
      receive_counts[destination][source] = send_counts[source][destination];
    }
  }
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  std::vector<LocalAllToAllVCall> calls;
  inputs.reserve(world);
  outputs.reserve(world);
  calls.reserve(world);
  for (size_t rank = 0; rank < world; ++rank) {
    const int64_t input_size = std::accumulate(send_counts[rank].begin(), send_counts[rank].end(), int64_t{0});
    const int64_t output_size = std::accumulate(receive_counts[rank].begin(), receive_counts[rank].end(), int64_t{0});
    std::vector<float> values;
    for (size_t peer = 0; peer < world; ++peer) {
      for (int64_t index = 0; index < send_counts[rank][peer]; ++index) {
        values.push_back(static_cast<float>((rank * 100) + (peer * 10) + static_cast<size_t>(index)));
      }
    }
    ASSERT_EQ(values.size(), static_cast<size_t>(input_size));
    inputs.push_back(FloatingTensorFromValues(GetContexts()[rank], Shape{input_size}, DType::FLOAT32, values));
    outputs.push_back(Empty(GetContexts()[rank], Shape{output_size}, DType::FLOAT32));
    calls.push_back(LocalAllToAllVCall{.context_ = &GetContexts()[rank],
                                       .output_ = &outputs[rank],
                                       .input_ = &inputs[rank],
                                       .send_counts_ = send_counts[rank],
                                       .receive_counts_ = receive_counts[rank],
                                       .communicator_ = &group.GetCommunicator(rank)});
  }
  AllToAllVLocal(calls);
  for (size_t rank = 0; rank < world; ++rank) {
    GetContexts()[rank].Synchronize();
    std::vector<float> expected;
    for (size_t source = 0; source < world; ++source) {
      for (int64_t index = 0; index < send_counts[source][rank]; ++index) {
        expected.push_back(static_cast<float>((source * 100) + (rank * 10) + static_cast<size_t>(index)));
      }
    }
    ExpectFloatValues(GetContexts()[rank], outputs[rank], expected);
  }
  auto invalid_send = send_counts;
  invalid_send[0][1] += 1;
  calls[0].send_counts_ = invalid_send[0];
  EXPECT_THROW(AllToAllVLocal(calls), InvalidArgumentError);
  group.Abort();
}

TEST_F(DistributedTest, CapturedGraphGroupCapturesAndReplaysEveryRankConcurrently) {
  const size_t world = GetDevices().size();
  std::vector<ExecutionContext> contexts;
  contexts.reserve(world);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  inputs.reserve(world);
  outputs.reserve(world);
  for (size_t rank = 0; rank < world; ++rank) {
    contexts.push_back(GetRuntime().CreateExecutionContext(GetDevices()[rank]));
    inputs.push_back(TensorFromValues<int32_t>(contexts.back(), Shape{2},
                                               {static_cast<int32_t>(rank), static_cast<int32_t>(rank + 1)}));
    outputs.push_back(Empty(contexts.back(), Shape{2}, DType::INT32));
  }

  CapturedGraphGroup graph_group = CapturedGraphGroup::Capture(
      std::move(contexts),
      [&](size_t rank, ExecutionContext &context) { AddOut(context, outputs[rank], inputs[rank], Scalar{int64_t{5}}); },
      {.name_ = "rank_increment"});
  EXPECT_EQ(graph_group.GetWorldSize(), world);
  EXPECT_EQ(graph_group.GetName(), "rank_increment");
  for (size_t rank = 0; rank < world; ++rank) {
    EXPECT_EQ(graph_group.GetDevice(rank), GetDevices()[rank]);
    EXPECT_GT(graph_group.GetNodeCount(rank), 0U);
  }
  graph_group.Launch();
  graph_group.Synchronize();
  for (size_t rank = 0; rank < world; ++rank) {
    ExpectValues<int32_t>(graph_group.GetContext(rank), outputs[rank],
                          {static_cast<int32_t>(rank + 5), static_cast<int32_t>(rank + 6)});
  }
  EXPECT_EQ(graph_group.GetLaunchCount(), 1U);
}

}  // namespace
}  // namespace ttl::test
