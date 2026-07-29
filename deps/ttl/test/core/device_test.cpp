#include "ttl/device.hpp"

#include <concepts>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_set>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ttl/error.hpp"

namespace ttl {

using testing::HasSubstr;

constexpr Device DEVICE_ZERO{0};
constexpr Device DEVICE_SEVEN{7};

static_assert(DEVICE_ZERO.GetOrdinal() == 0);
static_assert(DEVICE_SEVEN.GetOrdinal() == 7);
static_assert(Device{7} == Device{7});
static_assert(Device{7} != Device{8});
static_assert(std::constructible_from<Device, int32_t>);
static_assert(!std::convertible_to<int32_t, Device>);
static_assert(std::is_nothrow_copy_constructible_v<Device>);
static_assert(std::is_nothrow_move_constructible_v<Device>);

TEST(DeviceTest, StoresAnyNonNegativeOrdinal) {
  EXPECT_EQ(Device{0}.GetOrdinal(), 0);
  EXPECT_EQ(Device{17}.GetOrdinal(), 17);
  EXPECT_EQ(Device{std::numeric_limits<int32_t>::max()}.GetOrdinal(), std::numeric_limits<int32_t>::max());
}

TEST(DeviceTest, ComparesAndHashesByOrdinal) {
  const Device first{3};
  const Device same{3};
  const Device other{4};

  EXPECT_EQ(first, same);
  EXPECT_NE(first, other);
  EXPECT_EQ(std::hash<Device>{}(first), std::hash<Device>{}(same));

  const std::unordered_set<Device> devices{first, same, other};
  EXPECT_EQ(devices.size(), 2);
  EXPECT_TRUE(devices.contains(Device{3}));
  EXPECT_TRUE(devices.contains(Device{4}));
}

TEST(DeviceTest, FormatsConsistently) {
  const Device device{12};
  EXPECT_EQ(device.ToString(), "cuda:12");

  std::ostringstream stream;
  stream << std::hex << device;
  EXPECT_EQ(stream.str(), "cuda:12");
}

TEST(DeviceTest, RejectsNegativeOrdinalAtTheCallSite) {
  uint_least32_t expected_line = 0;
  try {
    expected_line = __LINE__ + 1;
    static_cast<void>(Device{-1});
  } catch (const InvalidArgumentError &error) {
    EXPECT_EQ(error.GetCode(), ErrorCode::INVALID_ARGUMENT);
    EXPECT_THAT(error.GetMessage(), HasSubstr("device ordinal"));
    EXPECT_THAT(error.GetMessage(), HasSubstr("-1"));
    EXPECT_EQ(error.GetLocation().line(), expected_line);
    EXPECT_THAT(std::string_view(error.GetLocation().file_name()), HasSubstr("device_test.cpp"));
    return;
  }
  FAIL() << "expected InvalidArgumentError";
}

TEST(DeviceTest, RejectsTheMinimumOrdinal) {
  EXPECT_THROW(static_cast<void>(Device{std::numeric_limits<int32_t>::min()}), InvalidArgumentError);
}

}  // namespace ttl
