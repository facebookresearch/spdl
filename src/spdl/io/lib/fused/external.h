/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <libspdl/core/buffer.h>
#include <libspdl/core/packets.h>

#include <nanobind/nanobind.h>

#include <memory>
#include <variant>
#include <vector>

namespace spdl::fused {

// Implemented by the extension that owns the value's native type.
struct ExternalValue {
  virtual ~ExternalValue() = default;
  // Called with the GIL held. Transfers the value to Python.
  virtual nanobind::object to_python() = 0;
};

using ExternalValuePtr = std::shared_ptr<ExternalValue>;

// References to registers, valid only during ExternalOp::run. An operation
// may move from a core pointer when its Python API consumes that input.
using ExternalInput = std::variant<
    core::CPUBufferPtr*,
    core::VideoPacketsPtr*,
    core::ImagePacketsPtr*,
    ExternalValuePtr*>;

struct ExternalOp {
  virtual ~ExternalOp() = default;
  // Called without the GIL. Implementations must be safe for concurrent runs
  // and must acquire the GIL themselves if they invoke Python callbacks.
  virtual ExternalValuePtr run(
      const std::vector<ExternalInput>& inputs) const = 0;
};

using ExternalOpPtr = std::shared_ptr<ExternalOp>;
inline constexpr char kExternalOpCapsule[] = "spdl.fused.ExternalOp.v1";

inline nanobind::capsule to_capsule(ExternalOpPtr op) {
  return nanobind::capsule(
      new ExternalOpPtr{std::move(op)},
      kExternalOpCapsule,
      [](void* ptr) noexcept { delete static_cast<ExternalOpPtr*>(ptr); });
}

inline ExternalOpPtr from_capsule(const nanobind::capsule& capsule) {
  auto* ptr = static_cast<ExternalOpPtr*>(capsule.data(kExternalOpCapsule));
  if (!ptr || !*ptr) {
    throw nanobind::value_error("Invalid fused external operation capsule.");
  }
  return *ptr;
}

} // namespace spdl::fused
