/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/types.h>

#include <cstdint>
#include <tuple>

namespace spdl::core {

/// Check if a rational value falls within a time window.
///
/// Tests if a value is within the half-open interval [start, end).
///
/// @param val Value to test.
/// @param start Window start (inclusive).
/// @param end Window end (exclusive).
/// @return true if start <= val < end, false otherwise.
bool is_within_window(
    const Rational& val,
    const Rational& start,
    const Rational& end);

/// Convert an integer value to rational using a time base.
///
/// @param val Integer value (typically a timestamp).
/// @param time_base Time base for conversion.
/// @return Rational representation.
Rational to_rational(int64_t val, const Rational time_base);

/// Create a rational from a tuple.
///
/// @param val Tuple of (numerator, denominator).
/// @return Rational representation.
Rational make_rational(const std::tuple<int64_t, int64_t>& val);

/// Convert a rational to a double.
///
/// @param val Rational value to convert.
/// @return Double representation.
double to_double(const Rational& val);

} // namespace spdl::core
