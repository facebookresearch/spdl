/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "libspdl/core/detail/logging.h"

extern "C" {
#include <libavutil/rational.h>
}

namespace spdl::core::detail {
// Check if a given AVRational value falls within a specified window [start,
// end). Returns true if start <= val < end (half-open interval).
inline bool is_within_window(
    const AVRational& val,
    const AVRational& start,
    const AVRational& end) {
  return (av_cmp_q(start, val) <= 0) && (av_cmp_q(val, end) < 0);
}

inline AVRational to_rational(int64_t val, const AVRational time_base) {
  AVRational ret;
  if (av_reduce(
          &ret.num, &ret.den, val * time_base.num, time_base.den, INT32_MAX)) {
    // Warn once that reduced PTS may be inexact due to rational reduction
    // constraints.
    static bool warned_inexact_pts = false;
    if (!warned_inexact_pts) {
      LOG(WARNING) << "PTS estimation was not exact during rational reduction; "
                      "timestamps might be slightly inaccurate.";
      warned_inexact_pts = true;
    }
  }
  return ret;
}
} // namespace spdl::core::detail
