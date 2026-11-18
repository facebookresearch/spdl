/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/frames.h>

#include <gtest/gtest.h>

#include <utility>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
}

namespace spdl::core {
namespace {

TEST(FramesTest, MoveConstructionPreservesTimeBase) {
  VideoFrames source(17, Rational{1, 90'000});

  VideoFrames destination(std::move(source));

  EXPECT_EQ(destination.get_id(), 17);
  EXPECT_EQ(destination.get_time_base().num, 1);
  EXPECT_EQ(destination.get_time_base().den, 90'000);
}

TEST(FramesTest, MoveAssignmentPreservesTimeBase) {
  VideoFrames source(17, Rational{1, 90'000});
  VideoFrames destination(23, Rational{1, 1'000});

  destination = std::move(source);

  EXPECT_EQ(destination.get_id(), 17);
  EXPECT_EQ(destination.get_time_base().num, 1);
  EXPECT_EQ(destination.get_time_base().den, 90'000);
}

std::vector<int64_t> get_pts(const VideoFrames& frames) {
  std::vector<int64_t> ret;
  ret.reserve(frames.get_frames().size());
  for (size_t i = 0; i < frames.get_frames().size(); ++i) {
    ret.push_back(frames.get_pts(i));
  }
  return ret;
}

void push_frames(VideoFrames& frames, int64_t start, int64_t stop) {
  for (int64_t pts = start; pts < stop; ++pts) {
    AVFrame* frame = av_frame_alloc();
    frame->pts = pts;
    frames.push_back(frame);
  }
}

TEST(FramesTest, ExtendMovesFrames) {
  VideoFrames frames(17, Rational{1, 30});
  VideoFrames other(17, Rational{1, 30});
  push_frames(frames, 0, 2);
  push_frames(other, 2, 5);

  frames.extend(std::move(other));

  EXPECT_EQ(get_pts(frames), (std::vector<int64_t>{0, 1, 2, 3, 4}));
}

TEST(FramesTest, TakeFrontMovesFirstFrames) {
  VideoFrames frames(17, Rational{1, 30});
  push_frames(frames, 0, 5);

  auto front = frames.take_front(3);
  auto rest = frames.take_front(10);

  EXPECT_EQ(get_pts(*front), (std::vector<int64_t>{0, 1, 2}));
  EXPECT_EQ(get_pts(*rest), (std::vector<int64_t>{3, 4}));
  EXPECT_TRUE(frames.get_frames().empty());
  EXPECT_EQ(front->get_id(), 17);
  EXPECT_EQ(front->get_time_base().den, 30);
}

} // namespace
} // namespace spdl::core
