/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libspdl/core/detail/ffmpeg/encoder.h"

#include <gtest/gtest.h>

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <limits>
#include <stdexcept>
#include <string>

namespace spdl::core::detail {
namespace {

TEST(EncoderTest, UnsupportedSampleRateFormattingStopsAtZeroSentinel) {
  const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
  ASSERT_NE(codec, nullptr);

  const int* supported_sample_rates = nullptr;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(61, 13, 100)
  supported_sample_rates = codec->supported_samplerates;
#else
  ASSERT_GE(
      avcodec_get_supported_config(
          nullptr,
          codec,
          AV_CODEC_CONFIG_SAMPLE_RATE,
          0,
          reinterpret_cast<const void**>(&supported_sample_rates),
          nullptr),
      0);
#endif
  if (!supported_sample_rates) {
    GTEST_SKIP() << "AAC encoder does not report supported sample rates";
  }

  const AudioEncodeConfig config{
      .num_channels = 2,
      .sample_rate = std::numeric_limits<int>::max(),
  };
  try {
    (void)make_encoder<MediaType::Audio>(codec, config, std::nullopt, 0);
    FAIL() << "Expected unsupported sample rate to be rejected";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("Supported values are"), std::string::npos);
    EXPECT_EQ(message.find(", 0"), std::string::npos);
  }
}

} // namespace
} // namespace spdl::core::detail
