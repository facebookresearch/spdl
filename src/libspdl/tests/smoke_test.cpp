/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/conversion.h>
#include <libspdl/core/decoder.h>
#include <libspdl/core/demuxing.h>
#include <libspdl/core/frames.h>
#include <libspdl/core/types.h>
#include <libspdl/core/utils.h>

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <memory>
#include <string>

#ifdef BUCK_BUILD
#include <libspdl/tests/fb/test_utils.h>
#else
namespace spdl::core::test {
std::string getTestDataPath(const std::string& filename) {
  return std::string(TEST_DATA_DIR) + "/" + filename;
}
} // namespace spdl::core::test
#endif

namespace spdl::core {
namespace {

using test::getTestDataPath;

TEST(SmokeTest, DecodeAudio) {
  std::string audioPath = getTestDataPath("test_audio.aac");

  auto demuxer = make_demuxer(audioPath);
  ASSERT_NE(demuxer, nullptr);
  EXPECT_TRUE(demuxer->has_audio());

  auto codec = demuxer->get_default_codec<MediaType::Audio>();
  auto packets = demuxer->demux_window<MediaType::Audio>();
  ASSERT_NE(packets, nullptr);
  EXPECT_GT(packets->pkts.get_packets().size(), 0);

  Decoder<MediaType::Audio> decoder(codec, std::nullopt, std::nullopt);
  auto frames = decoder.decode_packets(std::move(packets));
  ASSERT_NE(frames, nullptr);
  EXPECT_GT(frames->get_num_frames(), 0);

  // Convert frames to buffer
  auto buffer = convert_frames<MediaType::Audio>(frames.get());
  ASSERT_NE(buffer, nullptr);
  EXPECT_GT(buffer->shape.size(), 0);
}

TEST(SmokeTest, DecodeImage) {
  std::string imagePath = getTestDataPath("test_image.jpg");

  auto demuxer = make_demuxer(imagePath);
  ASSERT_NE(demuxer, nullptr);

  auto codec = demuxer->get_default_codec<MediaType::Image>();
  auto packets = demuxer->demux_window<MediaType::Image>();
  ASSERT_NE(packets, nullptr);
  EXPECT_GT(packets->pkts.get_packets().size(), 0);

  Decoder<MediaType::Image> decoder(codec, std::nullopt, std::nullopt);
  auto frames = decoder.decode_packets(std::move(packets));
  ASSERT_NE(frames, nullptr);
  EXPECT_GT(frames->get_num_frames(), 0);

  // Convert frames to buffer
  auto buffer = convert_frames<MediaType::Image>(frames.get());
  ASSERT_NE(buffer, nullptr);
  EXPECT_GT(buffer->shape.size(), 0);
}

TEST(SmokeTest, DecodeVideo) {
  std::string videoPath = getTestDataPath("test_video.mp4");

  auto demuxer = make_demuxer(videoPath);
  ASSERT_NE(demuxer, nullptr);

  auto codec = demuxer->get_default_codec<MediaType::Video>();
  auto packets = demuxer->demux_window<MediaType::Video>();
  ASSERT_NE(packets, nullptr);
  EXPECT_GT(packets->pkts.get_packets().size(), 0);

  Decoder<MediaType::Video> decoder(codec, std::nullopt, std::nullopt);
  auto frames = decoder.decode_packets(std::move(packets));
  ASSERT_NE(frames, nullptr);
  EXPECT_GT(frames->get_num_frames(), 0);

  // Convert frames to buffer
  auto buffer = convert_frames<MediaType::Video>(frames.get());
  ASSERT_NE(buffer, nullptr);
  EXPECT_GT(buffer->shape.size(), 0);
}

TEST(SmokeTest, DecodeAudioWithFormatFilter) {
  std::string audioPath = getTestDataPath("test_audio.aac");

  auto demuxer = make_demuxer(audioPath);
  ASSERT_NE(demuxer, nullptr);
  EXPECT_TRUE(demuxer->has_audio());

  auto codec = demuxer->get_default_codec<MediaType::Audio>();
  auto packets = demuxer->demux_window<MediaType::Audio>();
  ASSERT_NE(packets, nullptr);
  EXPECT_GT(packets->pkts.get_packets().size(), 0);

  // Construct proper filter graph with abuffer source and abuffersink
  std::string abuffer = fmt::format(
      "abuffer=sample_rate={}:sample_fmt={}:channel_layout={}c:time_base={}/{}",
      codec.get_sample_rate(),
      codec.get_sample_fmt(),
      codec.get_num_channels(),
      codec.get_time_base().num,
      codec.get_time_base().den);
  std::string filterDesc =
      fmt::format("{},aformat=sample_fmts=fltp,abuffersink", abuffer);

  Decoder<MediaType::Audio> decoder(codec, std::nullopt, filterDesc);
  auto frames = decoder.decode_packets(std::move(packets));
  ASSERT_NE(frames, nullptr);
  EXPECT_GT(frames->get_num_frames(), 0);

  // Convert frames to buffer
  auto buffer = convert_frames<MediaType::Audio>(frames.get());
  ASSERT_NE(buffer, nullptr);
  EXPECT_GT(buffer->shape.size(), 0);
}

TEST(SmokeTest, DecodeVideoWithFormatFilter) {
  std::string videoPath = getTestDataPath("test_video.mp4");

  auto demuxer = make_demuxer(videoPath);
  ASSERT_NE(demuxer, nullptr);

  auto codec = demuxer->get_default_codec<MediaType::Video>();
  auto packets = demuxer->demux_window<MediaType::Video>();
  ASSERT_NE(packets, nullptr);
  EXPECT_GT(packets->pkts.get_packets().size(), 0);

  // Construct proper filter graph with buffer source and buffersink
  auto sar = codec.get_sample_aspect_ratio();
  std::string buffer = fmt::format(
      "buffer=video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
      codec.get_width(),
      codec.get_height(),
      codec.get_pix_fmt(),
      codec.get_time_base().num,
      codec.get_time_base().den,
      sar.num,
      sar.den);
  std::string filterDesc =
      fmt::format("{},format=pix_fmts=rgb24,buffersink", buffer);

  Decoder<MediaType::Video> decoder(codec, std::nullopt, filterDesc);
  auto frames = decoder.decode_packets(std::move(packets));
  ASSERT_NE(frames, nullptr);
  EXPECT_GT(frames->get_num_frames(), 0);

  // Convert frames to buffer
  auto buf = convert_frames<MediaType::Video>(frames.get());
  ASSERT_NE(buf, nullptr);
  EXPECT_GT(buf->shape.size(), 0);
}

TEST(SmokeTest, DecodeImageWithFormatFilter) {
  std::string imagePath = getTestDataPath("test_image.jpg");

  auto demuxer = make_demuxer(imagePath);
  ASSERT_NE(demuxer, nullptr);

  auto codec = demuxer->get_default_codec<MediaType::Image>();
  auto packets = demuxer->demux_window<MediaType::Image>();
  ASSERT_NE(packets, nullptr);
  EXPECT_GT(packets->pkts.get_packets().size(), 0);

  // Construct proper filter graph with buffer source and buffersink
  auto sar = codec.get_sample_aspect_ratio();
  std::string buffer = fmt::format(
      "buffer=video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
      codec.get_width(),
      codec.get_height(),
      codec.get_pix_fmt(),
      codec.get_time_base().num,
      codec.get_time_base().den,
      sar.num,
      sar.den);
  std::string filterDesc =
      fmt::format("{},format=pix_fmts=yuv420p,buffersink", buffer);

  Decoder<MediaType::Image> decoder(codec, std::nullopt, filterDesc);
  auto frames = decoder.decode_packets(std::move(packets));
  ASSERT_NE(frames, nullptr);
  EXPECT_GT(frames->get_num_frames(), 0);

  // Convert frames to buffer
  auto buf = convert_frames<MediaType::Image>(frames.get());
  ASSERT_NE(buf, nullptr);
  EXPECT_GT(buf->shape.size(), 0);
}

TEST(SmokeTest, DecodeVideoWithChainedFilters) {
  std::string videoPath = getTestDataPath("test_video.mp4");

  auto demuxer = make_demuxer(videoPath);
  ASSERT_NE(demuxer, nullptr);

  auto codec = demuxer->get_default_codec<MediaType::Video>();
  auto packets = demuxer->demux_window<MediaType::Video>();
  ASSERT_NE(packets, nullptr);
  EXPECT_GT(packets->pkts.get_packets().size(), 0);

  // Construct proper filter graph with buffer source, chained filters, and
  // buffersink
  auto sar = codec.get_sample_aspect_ratio();
  std::string buffer = fmt::format(
      "buffer=video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
      codec.get_width(),
      codec.get_height(),
      codec.get_pix_fmt(),
      codec.get_time_base().num,
      codec.get_time_base().den,
      sar.num,
      sar.den);
  std::string filterDesc =
      fmt::format("{},format=pix_fmts=rgb24,hflip,buffersink", buffer);

  Decoder<MediaType::Video> decoder(codec, std::nullopt, filterDesc);
  auto frames = decoder.decode_packets(std::move(packets));
  ASSERT_NE(frames, nullptr);
  EXPECT_GT(frames->get_num_frames(), 0);

  // Convert frames to buffer
  auto buf = convert_frames<MediaType::Video>(frames.get());
  ASSERT_NE(buf, nullptr);
  EXPECT_GT(buf->shape.size(), 0);
}

TEST(SmokeTest, ConfigInitialization) {
  // Test DemuxConfig initialization with all fields
  DemuxConfig demux_cfg;
  demux_cfg.format = "mp4";
  demux_cfg.format_options = OptionDict{{"key1", "value1"}, {"key2", "value2"}};
  demux_cfg.buffer_size = 16192;
  EXPECT_EQ(demux_cfg.format.value(), "mp4");
  EXPECT_EQ(demux_cfg.format_options.value().at("key1"), "value1");
  EXPECT_EQ(demux_cfg.buffer_size, 16192);

  // Test DemuxConfig default initialization
  DemuxConfig demux_default;
  EXPECT_FALSE(demux_default.format.has_value());
  EXPECT_FALSE(demux_default.format_options.has_value());
  EXPECT_EQ(demux_default.buffer_size, SPDL_DEFAULT_BUFFER_SIZE);

  // Test DecodeConfig initialization with all fields
  DecodeConfig decode_cfg;
  decode_cfg.decoder = "h264_cuvid";
  decode_cfg.decoder_options = OptionDict{{"opt1", "val1"}};
  EXPECT_EQ(decode_cfg.decoder.value(), "h264_cuvid");
  EXPECT_EQ(decode_cfg.decoder_options.value().at("opt1"), "val1");

  // Test DecodeConfig default initialization
  DecodeConfig decode_default;
  EXPECT_FALSE(decode_default.decoder.has_value());
  EXPECT_FALSE(decode_default.decoder_options.has_value());

  // Test VideoEncodeConfig initialization
  VideoEncodeConfig video_encode_cfg{
      .height = 1080,
      .width = 1920,
      .pix_fmt = "yuv420p",
      .frame_rate = Rational{30, 1},
      .bit_rate = 5000000,
      .compression_level = 6,
      .qscale = 23,
      .gop_size = 30,
      .max_b_frames = 2,
      .colorspace = "bt709",
      .color_primaries = "bt709",
      .color_trc = "bt709"};
  EXPECT_EQ(video_encode_cfg.height, 1080);
  EXPECT_EQ(video_encode_cfg.width, 1920);
  EXPECT_EQ(video_encode_cfg.pix_fmt.value(), "yuv420p");
  EXPECT_EQ(video_encode_cfg.frame_rate.value().num, 30);
  EXPECT_EQ(video_encode_cfg.frame_rate.value().den, 1);
  EXPECT_EQ(video_encode_cfg.bit_rate, 5000000);
  EXPECT_EQ(video_encode_cfg.compression_level, 6);
  EXPECT_EQ(video_encode_cfg.qscale, 23);
  EXPECT_EQ(video_encode_cfg.gop_size, 30);
  EXPECT_EQ(video_encode_cfg.max_b_frames, 2);
  EXPECT_EQ(video_encode_cfg.colorspace.value(), "bt709");
  EXPECT_EQ(video_encode_cfg.color_primaries.value(), "bt709");
  EXPECT_EQ(video_encode_cfg.color_trc.value(), "bt709");

  // Test AudioEncodeConfig initialization
  AudioEncodeConfig audio_encode_cfg{
      .num_channels = 2,
      .sample_fmt = "fltp",
      .sample_rate = 44100,
      .bit_rate = 192000,
      .compression_level = 5,
      .qscale = 4};
  EXPECT_EQ(audio_encode_cfg.num_channels, 2);
  EXPECT_EQ(audio_encode_cfg.sample_fmt.value(), "fltp");
  EXPECT_EQ(audio_encode_cfg.sample_rate.value(), 44100);
  EXPECT_EQ(audio_encode_cfg.bit_rate, 192000);
  EXPECT_EQ(audio_encode_cfg.compression_level, 5);
  EXPECT_EQ(audio_encode_cfg.qscale, 4);
}

TEST(SmokeTest, UtilityFunctions) {
  // Test FFmpeg log level get/set
  int original_level = get_ffmpeg_log_level();
  EXPECT_GE(original_level, -8);

  set_ffmpeg_log_level(0);
  EXPECT_EQ(get_ffmpeg_log_level(), 0);

  // Restore original level
  set_ffmpeg_log_level(original_level);
  EXPECT_EQ(get_ffmpeg_log_level(), original_level);

  // Test get_ffmpeg_versions - should return version info
  auto versions = get_ffmpeg_versions();
  EXPECT_GT(versions.size(), 0);
  // Should contain at least libavcodec, libavformat, libavutil
  EXPECT_TRUE(versions.find("libavcodec") != versions.end());
  EXPECT_TRUE(versions.find("libavformat") != versions.end());
  EXPECT_TRUE(versions.find("libavutil") != versions.end());

  // Verify version tuples have valid values
  for (const auto& [lib_name, version] : versions) {
    auto [major, minor, micro] = version;
    EXPECT_GT(major, 0);
    EXPECT_GE(minor, 0);
    EXPECT_GE(micro, 0);
  }
}

} // namespace
} // namespace spdl::core
