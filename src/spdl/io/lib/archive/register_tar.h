/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spdl::archive {
void register_tar(nb::module_& m);
}
