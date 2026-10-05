/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/packets.h>

#include <gtest/gtest.h>

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <memory>
#include <new>
#include <utility>

namespace spdl::core {
namespace {

AVPacket* make_key_packet(int64_t pts) {
  AVPacket* packet = av_packet_alloc();
  if (!packet) {
    throw std::bad_alloc();
  }
  packet->pts = pts;
  packet->dts = pts;
  packet->flags = AV_PKT_FLAG_KEY;
  return packet;
}

TEST(PacketsTest, DefaultConstructionInitializesStreamIndex) {
  VideoPackets packets;
  EXPECT_EQ(packets.stream_index, 0);
}

TEST(PacketsTest, CopyPreservesStreamIndex) {
  VideoPackets source("memory://video", 7, Rational{1, 90'000});
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  VideoPackets copied(source);
  EXPECT_EQ(copied.stream_index, 7);

  VideoPackets assigned;
  assigned = source;
  EXPECT_EQ(assigned.stream_index, 7);
}

TEST(PacketsTest, MovePreservesStreamIndex) {
  VideoPackets source("memory://video", 7, Rational{1, 90'000});

  VideoPackets moved(std::move(source));
  EXPECT_EQ(moved.stream_index, 7);

  VideoPackets assigned("memory://other", 3, Rational{1, 1'000});
  assigned = std::move(moved);
  EXPECT_EQ(assigned.stream_index, 7);
}

TEST(PacketsTest, ExtractionPreservesDecodeMetadata) {
  auto source = std::make_unique<VideoPackets>(
      "memory://video",
      7,
      Rational{1, 90'000},
      TimeWindow{Rational{1, 10}, Rational{2, 10}});
  source->id = 42;
  source->pkts.push(make_key_packet(9'000));

  auto extracted = extract_packets_at_indices(source, {0});

  ASSERT_EQ(extracted.size(), 1);
  const auto& [packets, indices] = extracted[0];
  ASSERT_NE(packets, nullptr);
  ASSERT_EQ(indices.size(), 1);
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(packets->id, 42);
  EXPECT_EQ(packets->src, "memory://video");
  EXPECT_EQ(packets->stream_index, 7);
  EXPECT_EQ(packets->time_base.num, 1);
  EXPECT_EQ(packets->time_base.den, 90'000);
  EXPECT_FALSE(packets->timestamp.has_value());
}

} // namespace
} // namespace spdl::core
