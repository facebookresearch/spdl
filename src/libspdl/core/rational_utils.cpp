/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <libspdl/core/rational_utils.h>

#include <glog/logging.h>

extern "C" {
#include <libavutil/rational.h>
}

namespace spdl::core {

bool is_within_window(
    const Rational& val,
    const Rational& start,
    const Rational& end) {
  return (av_cmp_q(start, val) <= 0) && (av_cmp_q(val, end) < 0);
}

Rational to_rational(int64_t val, const Rational tb) {
  Rational ret;
  if (!av_reduce(&ret.num, &ret.den, val * tb.num, tb.den, INT32_MAX)) {
    // Warn once that reduced PTS may be inexact due to rational reduction
    // constraints.
    static bool warned_inexact_pts = false;
    if (!warned_inexact_pts) {
      LOG(WARNING) << "PTS conversion was not exact during rational reduction; "
                      "timestamp might be slightly inaccurate.";
      warned_inexact_pts = true;
    }
  }
  return ret;
}

Rational make_rational(const std::tuple<int64_t, int64_t>& val) {
  auto& [num, den] = val;
  Rational ret;
  if (!av_reduce(&ret.num, &ret.den, num, den, INT32_MAX)) {
    LOG(WARNING)
        << "Timestamp conversion was not exact during rational reduction; "
           "timestamp might be slightly inaccurate.";
  }
  return ret;
}

double to_double(const Rational& val) {
  return av_q2d(val);
}

} // namespace spdl::core
