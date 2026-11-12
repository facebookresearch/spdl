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
#include <libavutil/buffer.h>
#include <libavutil/mem.h>
}

#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
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

AVPacket* make_owned_packet() {
  AVPacket* packet = av_packet_alloc();
  if (!packet) {
    throw std::bad_alloc();
  }
  if (av_new_packet(packet, 1) < 0) {
    av_packet_free(&packet);
    throw std::bad_alloc();
  }
  return packet;
}

AVPacket* make_unclonable_packet(uint8_t& data) {
  AVPacket* packet = av_packet_alloc();
  if (!packet) {
    throw std::bad_alloc();
  }
  packet->data = &data;
  packet->size = std::numeric_limits<int>::max();
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

TEST(PacketsTest, H264ExtractionStartsOnlyAtIdrPackets) {
  for (int length_size = 0; length_size <= 4; ++length_size) {
    SCOPED_TRACE(length_size);
    auto* parameters = avcodec_parameters_alloc();
    ASSERT_NE(parameters, nullptr);
    parameters->codec_id = AV_CODEC_ID_H264;
    if (length_size) {
      parameters->extradata =
          static_cast<uint8_t*>(av_mallocz(5 + AV_INPUT_BUFFER_PADDING_SIZE));
      ASSERT_NE(parameters->extradata, nullptr);
      parameters->extradata_size = 5;
      parameters->extradata[0] = 1;
      parameters->extradata[4] = length_size - 1;
    }
    auto source = std::make_unique<VideoPackets>(
        "memory://h264",
        0,
        VideoCodec(parameters, Rational{1, 10}, Rational{10, 1}));
    avcodec_parameters_free(&parameters);
    for (int i = 0; i < 4; ++i) {
      std::vector<uint8_t> bytes(length_size ? length_size : 3, 0);
      bytes.back() = length_size ? 2 : 1;
      bytes.push_back(i == 0 || i == 3 ? 0x65 : 0x61);
      bytes.push_back(0x80);
      auto* packet = make_key_packet(i);
      ASSERT_GE(av_new_packet(packet, bytes.size()), 0);
      std::memcpy(packet->data, bytes.data(), bytes.size());
      packet->pts = packet->dts = i;
      packet->flags = AV_PKT_FLAG_KEY;
      source->pkts.push(packet);
    }
    auto splits = extract_packets_at_indices(source, {2, 3});
    ASSERT_EQ(splits.size(), 2);
    EXPECT_EQ(std::get<0>(splits[0])->pkts.get_packets().size(), 3);
    EXPECT_EQ(std::get<1>(splits[0]), (std::vector<size_t>{2}));
    EXPECT_EQ(std::get<0>(splits[1])->pkts.get_packets().size(), 1);
    EXPECT_EQ(std::get<1>(splits[1]), (std::vector<size_t>{0}));
  }
}

TEST(PacketsTest, UnrecognizedH264PacketsKeepDependencies) {
  const std::vector<std::vector<uint8_t>> payloads{
      {},
      {0},
      {0, 0, 0, 0},
      {0, 0, 0, 8, 0x65},
      {0, 0, 0, 1, 0x67},
      {0, 0, 0, 1, 0x65},
      {0, 0, 0, 2, 0x85, 0x80},
      {0, 0, 0, 2, 0x05, 0x80},
      {0, 0, 0, 2, 0x6f, 0x80},
      {0, 0, 0, 2, 0x65, 0x80, 0},
      {0, 0, 0, 2, 0x61, 0x80, 0, 0, 0, 2, 0x65, 0x80},
      {0, 0, 1, 0x61, 0x80, 0, 0, 1, 0x65, 0x80}};
  for (bool avcc : {false, true}) {
    for (const auto& bytes : payloads) {
      SCOPED_TRACE(avcc);
      SCOPED_TRACE(bytes.size());
      auto* parameters = avcodec_parameters_alloc();
      ASSERT_NE(parameters, nullptr);
      parameters->codec_id = AV_CODEC_ID_H264;
      if (avcc) {
        parameters->extradata =
            static_cast<uint8_t*>(av_mallocz(5 + AV_INPUT_BUFFER_PADDING_SIZE));
        ASSERT_NE(parameters->extradata, nullptr);
        parameters->extradata_size = 5;
        parameters->extradata[0] = 1;
        parameters->extradata[4] = 3;
      }
      auto source = std::make_unique<VideoPackets>(
          "memory://h264",
          0,
          VideoCodec(parameters, Rational{1, 10}, Rational{10, 1}));
      avcodec_parameters_free(&parameters);
      for (int i = 0; i < 3; ++i) {
        auto* packet = make_key_packet(i);
        if (!bytes.empty()) {
          ASSERT_GE(av_new_packet(packet, bytes.size()), 0);
          std::memcpy(packet->data, bytes.data(), bytes.size());
        }
        packet->pts = packet->dts = i;
        packet->flags = AV_PKT_FLAG_KEY;
        source->pkts.push(packet);
      }
      auto splits = extract_packets_at_indices(source, {2});
      ASSERT_EQ(splits.size(), 1);
      EXPECT_EQ(std::get<0>(splits[0])->pkts.get_packets().size(), 3);
      EXPECT_EQ(std::get<1>(splits[0]), (std::vector<size_t>{2}));
    }
  }
}

TEST(PacketsTest, H264ConfigurationChangesKeepFollowingIdrsInTheGroup) {
  for (int change = 0; change < 3; ++change) {
    SCOPED_TRACE(change);
    auto* parameters = avcodec_parameters_alloc();
    ASSERT_NE(parameters, nullptr);
    parameters->codec_id = AV_CODEC_ID_H264;
    parameters->extradata =
        static_cast<uint8_t*>(av_mallocz(5 + AV_INPUT_BUFFER_PADDING_SIZE));
    ASSERT_NE(parameters->extradata, nullptr);
    parameters->extradata_size = 5;
    parameters->extradata[0] = 1;
    parameters->extradata[4] = 3;
    auto source = std::make_unique<VideoPackets>(
        "memory://h264",
        0,
        VideoCodec(parameters, Rational{1, 10}, Rational{10, 1}));
    avcodec_parameters_free(&parameters);
    for (int i = 0; i < 3; ++i) {
      auto* packet = make_key_packet(i);
      std::vector<uint8_t> bytes{0, 0, 0, 2, 0x65, 0x80};
      if (i == 1 && change == 1) {
        bytes[4] = 0x67;
      } else if (i == 1 && change == 2) {
        bytes = {2, 0x65, 0x80};
      }
      ASSERT_GE(av_new_packet(packet, bytes.size()), 0);
      std::memcpy(packet->data, bytes.data(), bytes.size());
      packet->pts = packet->dts = i;
      packet->flags = AV_PKT_FLAG_KEY;
      if (i == 1 && change == 0) {
        auto* extra =
            av_packet_new_side_data(packet, AV_PKT_DATA_NEW_EXTRADATA, 5);
        ASSERT_NE(extra, nullptr);
        std::memset(extra, 0, 5);
        extra[0] = 1;
      }
      source->pkts.push(packet);
    }
    auto splits = extract_packets_at_indices(source, {2});
    ASSERT_EQ(splits.size(), 1);
    EXPECT_EQ(std::get<0>(splits[0])->pkts.get_packets().size(), 3);
    EXPECT_EQ(std::get<1>(splits[0]), (std::vector<size_t>{2}));
  }
}

TEST(PacketsTest, CopyFailureReleasesPreviouslyClonedPackets) {
  uint8_t invalid_data = 0;
  PacketSeries source;
  AVPacket* first = make_owned_packet();
  AVBufferRef* first_buffer = first->buf;
  source.push(first);
  source.push(make_unclonable_packet(invalid_data));
  ASSERT_EQ(av_buffer_get_ref_count(first_buffer), 1);

  EXPECT_THROW((void)PacketSeries{source}, std::runtime_error);

  EXPECT_EQ(av_buffer_get_ref_count(first_buffer), 1);
}

} // namespace
} // namespace spdl::core
