/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/codec.h>

#include <gtest/gtest.h>

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <new>
#include <type_traits>
#include <utility>

namespace spdl::core {
namespace {

static_assert(!std::is_nothrow_constructible_v<
              VideoCodec,
              const AVCodecParameters*,
              Rational,
              Rational>);

void expect_rational_eq(Rational actual, Rational expected) {
  const std::pair actual_value{actual.num, actual.den};
  const std::pair expected_value{expected.num, expected.den};
  EXPECT_EQ(actual_value, expected_value);
}

VideoCodec make_codec(int width, Rational time_base, Rational frame_rate) {
  AVCodecParameters* parameters = avcodec_parameters_alloc();
  if (!parameters) {
    throw std::bad_alloc();
  }
  parameters->codec_id = AV_CODEC_ID_H264;
  parameters->width = width;
  parameters->height = 360;
  VideoCodec codec(parameters, time_base, frame_rate);
  avcodec_parameters_free(&parameters);
  return codec;
}

TEST(CodecTest, DefaultAndMovedFromCodecsCanBeCopiedSafely) {
  VideoCodec empty;
  EXPECT_EQ(empty.get_parameters(), nullptr);
  expect_rational_eq(empty.get_time_base(), Rational{});
  expect_rational_eq(empty.get_frame_rate(), Rational{});

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  VideoCodec empty_copy(empty);
  EXPECT_EQ(empty_copy.get_parameters(), nullptr);

  VideoCodec populated = make_codec(640, Rational{1, 90'000}, Rational{30, 1});
  VideoCodec moved(std::move(populated));
  ASSERT_NE(moved.get_parameters(), nullptr);
  EXPECT_EQ(
      populated.get_parameters(), nullptr); // NOLINT(bugprone-use-after-move)

  VideoCodec moved_from_copy(populated); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(moved_from_copy.get_parameters(), nullptr);
}

TEST(CodecTest, CopyAssignmentReplacesParametersAndPreservesMetadata) {
  VideoCodec source = make_codec(640, Rational{1, 90'000}, Rational{30, 1});
  VideoCodec destination = make_codec(320, Rational{1, 1'000}, Rational{24, 1});

  destination = source;

  ASSERT_NE(destination.get_parameters(), nullptr);
  EXPECT_NE(destination.get_parameters(), source.get_parameters());
  EXPECT_EQ(destination.get_parameters()->width, 640);
  expect_rational_eq(destination.get_time_base(), Rational{1, 90'000});
  expect_rational_eq(destination.get_frame_rate(), Rational{30, 1});

  VideoCodec* const destination_alias = &destination;
  destination = *destination_alias;
  EXPECT_EQ(destination.get_parameters()->width, 640);
}

} // namespace
} // namespace spdl::core
