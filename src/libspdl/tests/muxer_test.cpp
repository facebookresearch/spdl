/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/compat.h"
#include "libspdl/core/detail/ffmpeg/ctx_utils.h"
#include "libspdl/core/detail/ffmpeg/wrappers.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/mem.h>
}

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace spdl::core::detail {
namespace {

class ScopedWarningCapture final : public google::LogSink {
 public:
  ScopedWarningCapture() {
    google::AddLogSink(this);
  }

  ~ScopedWarningCapture() override {
    google::RemoveLogSink(this);
  }

  void send(
      google::LogSeverity severity,
      const char*,
      const char*,
      int,
      const struct ::tm*,
      const char* message,
      size_t message_len) override {
    if (severity < google::GLOG_WARNING) {
      return;
    }
    std::lock_guard lock(mutex_);
    warnings_.emplace_back(message, message_len);
  }

  bool contains(const std::string& needle) {
    std::lock_guard lock(mutex_);
    return std::any_of(
        warnings_.begin(), warnings_.end(), [&](const std::string& warning) {
          return warning.find(needle) != std::string::npos;
        });
  }

 private:
  std::mutex mutex_;
  std::vector<std::string> warnings_;
};

std::string get_temp_output_path(const std::string& filename) {
  const auto path = std::filesystem::path(testing::TempDir()) / filename;
  std::filesystem::remove(path);
  return path.string();
}

void add_pcm_audio_stream(AVFormatContext* format_ctx) {
  AVStream* stream = avformat_new_stream(format_ctx, nullptr);
  ASSERT_NE(stream, nullptr);

  AVCodecParameters* params = stream->codecpar;
  params->codec_type = AVMEDIA_TYPE_AUDIO;
  params->codec_id = AV_CODEC_ID_PCM_S16LE;
  params->format = AV_SAMPLE_FMT_S16;
  params->sample_rate = 8000;
  params->bits_per_coded_sample = 16;
  params->bit_rate = 128000;
  params->block_align = 2;
  SET_CHANNELS(params, 1);
  stream->time_base = {1, params->sample_rate};
}

#if defined(__linux__)
bool is_file_open(const std::string& path) {
  for (const auto& entry :
       std::filesystem::directory_iterator{"/proc/self/fd"}) {
    std::error_code error;
    const auto target = std::filesystem::read_symlink(entry.path(), error);
    if (!error && std::filesystem::equivalent(target, path, error) && !error) {
      return true;
    }
  }
  return false;
}
#endif

struct WriteTracker {
  int bytes_written{};
};

#if LIBAVFORMAT_VERSION_MAJOR >= 61
using WriteBuffer = const uint8_t*;
#else
using WriteBuffer = uint8_t*;
#endif

int track_write(void* opaque, WriteBuffer, int size) {
  static_cast<WriteTracker*>(opaque)->bytes_written += size;
  return size;
}

void expect_io_is_usable(AVIOContext* io_ctx, WriteTracker* tracker) {
  // Flush any muxer bytes first so the assertion only observes this probe.
  avio_flush(io_ctx);
  const int bytes_written = tracker->bytes_written;
  const uint8_t byte{};
  avio_write(io_ctx, &byte, 1);
  avio_flush(io_ctx);
  EXPECT_EQ(tracker->bytes_written, bytes_written + 1);
}

int fail_trailer(AVFormatContext*) {
  return AVERROR(EIO);
}

TEST(MuxerCleanupTest, OutputContextDestructionClosesOwnedIo) {
#if defined(__linux__)
  const std::string path = get_temp_output_path("spdl_muxer_destructor.wav");
  {
    auto format_ctx = get_output_format_ctx(path, "wav");
    add_pcm_audio_stream(format_ctx.get());
    open_format(format_ctx.get());
    ASSERT_TRUE(is_file_open(path));
  }

  EXPECT_FALSE(is_file_open(path));
  std::filesystem::remove(path);
#else
  GTEST_SKIP() << "File descriptor introspection requires procfs";
#endif
}

TEST(MuxerCleanupTest, OutputContextDestructionLogsIoCloseFailure) {
#if defined(__linux__)
  if (!std::filesystem::exists("/dev/full")) {
    GTEST_SKIP() << "/dev/full is not available";
  }

  ScopedWarningCapture logs;
  auto format_ctx = get_output_format_ctx("/dev/full", "wav");
  ASSERT_GE(
      avio_open2(
          &format_ctx->pb, "/dev/full", AVIO_FLAG_WRITE, nullptr, nullptr),
      0);

  const uint8_t byte{0};
  avio_write(format_ctx->pb, &byte, 1);
  ASSERT_EQ(format_ctx->pb->error, 0);

  format_ctx.reset();

  EXPECT_TRUE(logs.contains("Failed to close output I/O during cleanup."));
#else
  GTEST_SKIP() << "/dev/full is only available on Linux";
#endif
}

TEST(MuxerCleanupTest, OpenAndCloseFormatPreserveCustomIo) {
  WriteTracker tracker;
  auto* buffer = static_cast<unsigned char*>(av_malloc(64));
  ASSERT_NE(buffer, nullptr);
  AVIOContextPtr custom_io{avio_alloc_context(
      buffer, 64, 1, &tracker, nullptr, track_write, nullptr)};
  ASSERT_NE(custom_io, nullptr);
  AVIOContext* const supplied_io = custom_io.get();

  {
    auto format_ctx = get_output_format_ctx("unused.wav", "wav");
    add_pcm_audio_stream(format_ctx.get());
    format_ctx->pb = supplied_io;
    format_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;

    ASSERT_NO_THROW(open_format(format_ctx.get()));
    EXPECT_EQ(format_ctx->pb, supplied_io);

    // Calling the low-level cleanup helper directly must be a no-op for custom
    // I/O, including leaving the pointer installed on the format context.
    EXPECT_EQ(close_output_io(format_ctx.get()), 0);
    EXPECT_EQ(format_ctx->pb, supplied_io);
    expect_io_is_usable(supplied_io, &tracker);

    ASSERT_NO_THROW(close_format(format_ctx.get()));
    EXPECT_EQ(format_ctx->pb, supplied_io);
    expect_io_is_usable(supplied_io, &tracker);
  }

  // The AVFormatContext deleter must not claim ownership either.
  expect_io_is_usable(supplied_io, &tracker);
}

TEST(MuxerCleanupTest, OpenFormatClosesIoWhenHeaderFails) {
  const std::string path =
      get_temp_output_path("spdl_muxer_header_failure.wav");
  auto format_ctx = get_output_format_ctx(path, "wav");

  try {
    open_format(format_ctx.get());
    FAIL() << "Expected a muxer without streams to reject its header";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string{error.what()}.find("Failed to write header"),
        std::string::npos);
  }

  EXPECT_EQ(format_ctx->pb, nullptr);
  std::filesystem::remove(path);
}

TEST(MuxerCleanupTest, OpenFormatClosesIoWhenOptionValidationFails) {
  const std::string path =
      get_temp_output_path("spdl_muxer_option_failure.wav");
  auto format_ctx = get_output_format_ctx(path, "wav");
  add_pcm_audio_stream(format_ctx.get());

  try {
    open_format(
        format_ctx.get(), OptionDict{{"spdl_unknown_muxer_option", "1"}});
    FAIL() << "Expected an unknown muxer option to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string{error.what()}.find("Unexpected options"),
        std::string::npos);
  }

  EXPECT_EQ(format_ctx->pb, nullptr);
  std::filesystem::remove(path);
}

TEST(MuxerCleanupTest, CloseFormatClosesIoWhenTrailerFails) {
  const std::string path =
      get_temp_output_path("spdl_muxer_trailer_failure.wav");
  auto format_ctx = get_output_format_ctx(path, "wav");
  add_pcm_audio_stream(format_ctx.get());
  open_format(format_ctx.get());
  ASSERT_NE(format_ctx->pb, nullptr);

  try {
    close_format(format_ctx.get(), fail_trailer);
    FAIL() << "Expected the injected output error to fail the trailer";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string{error.what()}.find("Failed to write trailer"),
        std::string::npos);
  }

  EXPECT_EQ(format_ctx->pb, nullptr);
  std::filesystem::remove(path);
}

TEST(MuxerCleanupTest, CloseFormatLogsIoFailureWhenTrailerAlsoFails) {
#if defined(__linux__)
  if (!std::filesystem::exists("/dev/full")) {
    GTEST_SKIP() << "/dev/full is not available";
  }

  ScopedWarningCapture logs;
  auto format_ctx = get_output_format_ctx("/dev/full", "wav");
  ASSERT_GE(
      avio_open2(
          &format_ctx->pb, "/dev/full", AVIO_FLAG_WRITE, nullptr, nullptr),
      0);

  const uint8_t byte{0};
  avio_write(format_ctx->pb, &byte, 1);
  ASSERT_EQ(format_ctx->pb->error, 0);

  try {
    close_format(format_ctx.get(), fail_trailer);
    FAIL() << "Expected the injected output error to fail the trailer";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string{error.what()}.find("Failed to write trailer"),
        std::string::npos);
  }

  EXPECT_EQ(format_ctx->pb, nullptr);
  EXPECT_TRUE(
      logs.contains("Failed to close output after the trailer also failed"));
#else
  GTEST_SKIP() << "/dev/full is only available on Linux";
#endif
}

} // namespace
} // namespace spdl::core::detail
