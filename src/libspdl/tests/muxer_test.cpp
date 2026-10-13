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

} // namespace
} // namespace spdl::core::detail
