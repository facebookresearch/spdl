/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spdl::core {
void register_types(nb::module_&);
void register_packets(nb::module_&);
void register_frames(nb::module_&);
void register_storage(nb::module_&);
void register_buffers(nb::module_&);
void register_tracing(nb::module_&);
void register_utils(nb::module_&);
void register_demuxing(nb::module_&);
void register_decoding(nb::module_&);
void register_conversion(nb::module_&);
void register_encoding(nb::module_&);
void register_filtering(nb::module_&);
void register_bsf(nb::module_&);
} // namespace spdl::core
