/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <nanobind/nanobind.h>

#ifdef SPDL_HOLD_GIL
#define RELEASE_GIL()
#else
#define RELEASE_GIL() nb::gil_scoped_release __g;
#endif
