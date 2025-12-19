/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("demuxing").SetDescription("Demuxing events"),
    perfetto::Category("decoding").SetDescription("Decoding events"),
    perfetto::Category("nvdec").SetDescription("Decoding events using NVDEC"),
    perfetto::Category("other").SetDescription(
        "Other events used for testing"));

namespace spdl::core::detail {
void init_perfetto();
void configure_perfetto(const std::string& process_name);

std::unique_ptr<perfetto::TracingSession> start_tracing_session(
    int fd,
    int buffer_size_in_kb);
void stop_tracing_session(std::unique_ptr<perfetto::TracingSession> sess);

} // namespace spdl::core::detail
