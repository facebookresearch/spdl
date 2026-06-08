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

#include <cstring>
#include <stdexcept>
#include <vector>

namespace spdl::core {
namespace {

// Build a standalone AVPacket with owned data and the given metadata.
AVPacket* make_packet(
    const std::vector<uint8_t>& data,
    int64_t pts,
    int64_t dts,
    int flags,
    int64_t duration) {
  AVPacket* pkt = av_packet_alloc();
  if (!pkt || av_new_packet(pkt, static_cast<int>(data.size())) < 0) {
    throw std::runtime_error("Failed to allocate test packet");
  }
  std::memcpy(pkt->data, data.data(), data.size());
  pkt->pts = pts;
  pkt->dts = dts;
  pkt->flags = flags;
  pkt->duration = duration;
  return pkt;
}

void expect_packet_eq(const AVPacket* a, const AVPacket* b) {
  EXPECT_EQ(a->pts, b->pts);
  EXPECT_EQ(a->dts, b->dts);
  EXPECT_EQ(a->flags, b->flags);
  EXPECT_EQ(a->duration, b->duration);
  ASSERT_EQ(a->size, b->size);
  EXPECT_EQ(0, std::memcmp(a->data, b->data, a->size));
  ASSERT_EQ(a->side_data_elems, b->side_data_elems);
  for (int i = 0; i < a->side_data_elems; ++i) {
    EXPECT_EQ(a->side_data[i].type, b->side_data[i].type);
    ASSERT_EQ(a->side_data[i].size, b->side_data[i].size);
    EXPECT_EQ(
        0,
        std::memcmp(
            a->side_data[i].data, b->side_data[i].data, a->side_data[i].size));
  }
}

TEST(PacketsSerializationTest, RoundTripPreservesMetadataAndPayload) {
  Packets<MediaType::Video> packets;
  packets.src = "test://video";
  packets.stream_index = 2;
  packets.time_base = Rational{1, 1000};
  packets.timestamp = TimeWindow{Rational{1, 10}, Rational{5, 10}};

  packets.pkts.push(make_packet(
      {1, 2, 3, 4, 5},
      /*pts=*/100,
      /*dts=*/90,
      AV_PKT_FLAG_KEY,
      /*duration=*/10));
  packets.pkts.push(make_packet(
      {9, 8, 7},
      /*pts=*/110,
      /*dts=*/100,
      /*flags=*/0,
      /*duration=*/10));

  auto data = serialize_packets(packets);
  auto restored = deserialize_packets<MediaType::Video>(data);

  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->src, packets.src);
  EXPECT_EQ(restored->stream_index, packets.stream_index);
  EXPECT_EQ(restored->time_base.num, packets.time_base.num);
  EXPECT_EQ(restored->time_base.den, packets.time_base.den);

  ASSERT_TRUE(restored->timestamp.has_value());
  auto [start, end] = *restored->timestamp;
  EXPECT_EQ(start.num, 1);
  EXPECT_EQ(start.den, 10);
  EXPECT_EQ(end.num, 5);
  EXPECT_EQ(end.den, 10);

  const auto& orig = packets.pkts.get_packets();
  const auto& got = restored->pkts.get_packets();
  ASSERT_EQ(got.size(), orig.size());
  for (size_t i = 0; i < orig.size(); ++i) {
    expect_packet_eq(orig[i], got[i]);
  }
}

TEST(PacketsSerializationTest, RoundTripPreservesSideData) {
  Packets<MediaType::Audio> packets;
  packets.src = "test://audio";
  packets.stream_index = 0;
  packets.time_base = Rational{1, 44100};

  AVPacket* pkt = make_packet(
      {10, 20, 30, 40}, /*pts=*/0, /*dts=*/0, /*flags=*/0, /*duration=*/1024);
  const std::vector<uint8_t> sd = {0xde, 0xad, 0xbe, 0xef};
  uint8_t* dst =
      av_packet_new_side_data(pkt, AV_PKT_DATA_SKIP_SAMPLES, sd.size());
  ASSERT_NE(dst, nullptr);
  std::memcpy(dst, sd.data(), sd.size());
  packets.pkts.push(pkt);

  auto data = serialize_packets(packets);
  auto restored = deserialize_packets<MediaType::Audio>(data);

  ASSERT_NE(restored, nullptr);
  const auto& got = restored->pkts.get_packets();
  ASSERT_EQ(got.size(), 1u);
  expect_packet_eq(pkt, got[0]);
  ASSERT_EQ(got[0]->side_data_elems, 1);
  EXPECT_EQ(got[0]->side_data[0].type, AV_PKT_DATA_SKIP_SAMPLES);
}

TEST(PacketsSerializationTest, RoundTripEmpty) {
  Packets<MediaType::Audio> packets;
  packets.src = "test://empty";
  packets.stream_index = 1;
  packets.time_base = Rational{1, 1000};

  auto data = serialize_packets(packets);
  auto restored = deserialize_packets<MediaType::Audio>(data);

  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->src, "test://empty");
  EXPECT_EQ(restored->stream_index, 1);
  EXPECT_EQ(restored->pkts.get_packets().size(), 0u);
  EXPECT_FALSE(restored->timestamp.has_value());
}

TEST(PacketsSerializationTest, MediaTypeMismatchThrows) {
  Packets<MediaType::Audio> packets;
  packets.src = "test://mismatch";
  packets.time_base = Rational{1, 1000};

  auto data = serialize_packets(packets);
  EXPECT_THROW(deserialize_packets<MediaType::Video>(data), std::runtime_error);
}

} // namespace
} // namespace spdl::core
