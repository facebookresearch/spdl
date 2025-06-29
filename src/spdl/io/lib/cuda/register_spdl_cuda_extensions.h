/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spdl::cuda {
void register_types(nb::module_&);
void register_buffers(nb::module_&);
void register_decoding_nvdec(nb::module_&);
void register_decoding_nvjpeg(nb::module_&);
void register_storage(nb::module_&);
void register_transfer(nb::module_&);
void register_utils(nb::module_&);
void register_color_conversion(nb::module_&);
} // namespace spdl::cuda
