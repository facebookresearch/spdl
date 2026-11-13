/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "register_spdl_core_extensions.h"

#include <libspdl/core/bsf.h>
#include <libspdl/core/conversion.h>
#include <libspdl/core/decoder.h>
#include <libspdl/core/demuxing.h>
#include <libspdl/core/rational_utils.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "memoryview_utils.h"

#include <algorithm>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace nb = nanobind;

namespace spdl::core {
namespace {

// Metadata produced by queries. Converted to Python objects after execution.
using Meta = std::variant<
    std::monostate,
    bool,
    int64_t,
    double,
    std::string,
    std::vector<int64_t>,
    std::vector<double>,
    std::tuple<int64_t, int64_t>,
    std::tuple<double, double>>;

// A value held by a register of the fused program.
// Sources are borrowed from Python for the duration of `FusedPlan::run`.
using Value = std::variant<
    std::monostate,
    std::string,
    std::string_view,
    DemuxerPtr,
    AudioPacketsPtr,
    VideoPacketsPtr,
    ImagePacketsPtr,
    AudioFramesPtr,
    VideoFramesPtr,
    ImageFramesPtr,
    AudioCodec,
    VideoCodec,
    ImageCodec,
    CPUBufferPtr,
    Meta>;

using Registers = std::vector<Value>;

using PyWindow =
    std::tuple<std::tuple<int64_t, int64_t>, std::tuple<int64_t, int64_t>>;

// Python slice. `std::nullopt` stands for `None`.
using PySlice = std::tuple<
    std::optional<int64_t>,
    std::optional<int64_t>,
    std::optional<int64_t>>;

using SliceKey = std::variant<int64_t, PySlice, std::vector<int64_t>>;

struct OpenOp {
  int src;
  int out;
  std::optional<DemuxConfig> demux_config;
  std::optional<std::string> name;
};

struct DemuxOp {
  MediaType media;
  int src;
  int out;
  std::optional<TimeWindow> window;
  std::optional<std::string> bsf;
};

struct DropOp {
  std::vector<int> regs;
};

struct BSFOp {
  MediaType media;
  int src;
  int out;
  std::string bsf;
};

struct DecodeOp {
  MediaType media;
  int src;
  int out;
  // The filter body between the buffer source and the sink.
  // The source part depends on the codec, so it is attached at run time.
  std::optional<std::string> filter_desc;
  std::optional<DecodeConfig> decode_config;
  int num_frames;
};

struct SampleDecodeOp {
  int src;
  std::vector<int> outs;
  std::vector<int64_t> indices;
  std::optional<std::string> filter_desc;
  std::optional<DecodeConfig> decode_config;
  bool optimized;
};

struct ConvertOp {
  MediaType media;
  std::vector<int> srcs;
  int out;
  bool batch;
  std::shared_ptr<CPUStorage> storage;
};

struct CloneOp {
  int src;
  int out;
};

struct SliceOp {
  int src;
  int out;
  SliceKey key;
};

struct QueryOp {
  int src;
  int out;
  std::string name;
  bool raw;
};

using Op = std::variant<
    OpenOp,
    DemuxOp,
    DropOp,
    BSFOp,
    DecodeOp,
    SampleDecodeOp,
    ConvertOp,
    CloneOp,
    SliceOp,
    QueryOp>;

MediaType parse_media(const std::string& media) {
  if (media == "audio") {
    return MediaType::Audio;
  }
  if (media == "video") {
    return MediaType::Video;
  }
  if (media == "image") {
    return MediaType::Image;
  }
  throw std::invalid_argument(fmt::format("Unexpected media type: {}", media));
}

template <typename F>
auto dispatch(MediaType media, F&& f) {
  switch (media) {
    case MediaType::Audio:
      return f(std::integral_constant<MediaType, MediaType::Audio>{});
    case MediaType::Video:
      return f(std::integral_constant<MediaType, MediaType::Video>{});
    case MediaType::Image:
      return f(std::integral_constant<MediaType, MediaType::Image>{});
  }
  throw std::runtime_error("Unexpected media type.");
}

template <typename T>
T& get(Registers& regs, int i) {
  auto* v = std::get_if<T>(&regs.at(i));
  if (!v) {
    throw std::runtime_error(
        fmt::format(
            "Register {} does not hold the expected value. "
            "(It was released already, or the program is malformed.)",
            i));
  }
  return *v;
}

template <typename T>
T take(Registers& regs, int i) {
  T ret = std::move(get<T>(regs, i));
  regs.at(i) = std::monostate{};
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Demux
////////////////////////////////////////////////////////////////////////////////
DemuxerPtr open_source(const Value& src, const OpenOp& op) {
  if (const auto* path = std::get_if<std::string>(&src)) {
    return make_demuxer(*path, nullptr, op.demux_config, op.name);
  }
  if (const auto* data = std::get_if<std::string_view>(&src)) {
    return make_demuxer(*data, op.demux_config, op.name);
  }
  throw std::runtime_error("Demuxer input is not a source.");
}

void execute(Registers& regs, const OpenOp& op) {
  regs.at(op.out) = open_source(regs.at(op.src), op);
}

void execute(Registers& regs, const DemuxOp& op) {
  auto& demuxer = get<DemuxerPtr>(regs, op.src);
  regs.at(op.out) = dispatch(op.media, [&](auto media) -> Value {
    return demuxer->demux_window<media.value>(op.window, op.bsf);
  });
}

void execute(Registers& regs, const DropOp& op) {
  for (const int i : op.regs) {
    regs.at(i) = std::monostate{};
  }
}

void execute(Registers& regs, const BSFOp& op) {
  regs.at(op.out) = dispatch(op.media, [&](auto media) -> Value {
    auto packets = take<PacketsPtr<media.value>>(regs, op.src);
    if (!packets->codec) {
      throw std::runtime_error("The packets do not have codec.");
    }
    BSF<media.value> bsf{*packets->codec, op.bsf};
    auto ret = bsf.filter(std::move(packets), true);
    if (!ret) {
      throw std::runtime_error("Bit stream filtering produced no packets.");
    }
    return std::move(*ret);
  });
}

////////////////////////////////////////////////////////////////////////////////
// Decode
////////////////////////////////////////////////////////////////////////////////
template <MediaType media>
std::string get_source_desc(const Codec<media>& codec) {
  const auto tb = codec.get_time_base();
  if constexpr (media == MediaType::Audio) {
    return fmt::format(
        "abuffer=time_base={}/{}:sample_rate={}:sample_fmt={}:channel_layout={}",
        tb.num,
        tb.den,
        codec.get_sample_rate(),
        codec.get_sample_fmt(),
        codec.get_channel_layout());
  } else {
    const auto sar = codec.get_sample_aspect_ratio();
    return fmt::format(
        "buffer=video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}",
        codec.get_width(),
        codec.get_height(),
        codec.get_pix_fmt(),
        tb.num,
        tb.den,
        sar.num,
        sar.den);
  }
}

template <MediaType media>
std::optional<std::string> get_filter_graph(
    const Codec<media>& codec,
    const std::optional<std::string>& body) {
  if (!body) {
    return std::nullopt;
  }
  constexpr auto sink =
      media == MediaType::Audio ? "abuffersink" : "buffersink";
  return fmt::format("{},{},{}", get_source_desc(codec), *body, sink);
}

template <MediaType media>
FramesPtr<media> decode(
    PacketsPtr<media> packets,
    const std::optional<DecodeConfig>& decode_config,
    const std::optional<std::string>& filter_desc,
    int num_frames) {
  if (!packets->codec) {
    throw std::runtime_error("Packets does not have codec info.");
  }
  Decoder<media> decoder{
      *packets->codec,
      decode_config,
      get_filter_graph(*packets->codec, filter_desc)};
  return decoder.decode_packets(std::move(packets), num_frames);
}

void execute(Registers& regs, const DecodeOp& op) {
  regs.at(op.out) = dispatch(op.media, [&](auto media) -> Value {
    return decode<media.value>(
        take<PacketsPtr<media.value>>(regs, op.src),
        op.decode_config,
        op.filter_desc,
        op.num_frames);
  });
}

// Mirrors `spdl.io._composite._decode_selected` and `_decode_partial`.
void decode_group(
    VideoPacketsPtr packets,
    const std::vector<size_t>& indices,
    const SampleDecodeOp& op,
    std::vector<ImageFramesPtr>& out) {
  const auto n = indices.size();
  const auto max = *std::max_element(indices.begin(), indices.end());
  if (op.optimized && op.filter_desc == "format=pix_fmts=rgb24" && n <= 32 &&
      n * 2 < max + 1) {
    std::vector<std::string> conds;
    conds.reserve(n);
    for (const auto i : indices) {
      conds.push_back(fmt::format("eq(n,{})", i));
    }
    auto frames = decode<MediaType::Video>(
        std::move(packets),
        op.decode_config,
        fmt::format(
            "select=expr='{}',{}", fmt::join(conds, "+"), *op.filter_desc),
        static_cast<int>(n));
    for (size_t i = 0; i < n; ++i) {
      out.push_back(frames->slice(static_cast<int64_t>(i)));
    }
    return;
  }
  auto frames = decode<MediaType::Video>(
      std::move(packets),
      op.decode_config,
      op.filter_desc,
      static_cast<int>(max + 1));
  for (const auto i : indices) {
    out.push_back(frames->slice(static_cast<int64_t>(i)));
  }
}

void execute(Registers& regs, const SampleDecodeOp& op) {
  const auto& packets = get<VideoPacketsPtr>(regs, op.src);
  const auto num_packets = get_timestamps(*packets).size();
  std::vector<size_t> indices;
  for (const auto i : op.indices) {
    if (i < 0 || static_cast<size_t>(i) >= num_packets) {
      throw std::out_of_range(
          fmt::format("Frame index must be [0, {}).", num_packets));
    }
    indices.push_back(static_cast<size_t>(i));
  }
  std::vector<ImageFramesPtr> frames;
  for (auto& [split, idxes] :
       extract_packets_at_indices(packets, indices, op.optimized)) {
    decode_group(std::move(split), idxes, op, frames);
  }
  if (frames.size() != op.outs.size()) {
    throw std::runtime_error(
        fmt::format(
            "Expected {} frames but decoded {}.",
            op.outs.size(),
            frames.size()));
  }
  for (size_t i = 0; i < frames.size(); ++i) {
    regs.at(op.outs[i]) = std::move(frames[i]);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Convert
////////////////////////////////////////////////////////////////////////////////
template <MediaType media>
CPUBufferPtr convert(Registers& regs, const ConvertOp& op) {
  std::vector<const Frames<media>*> refs;
  refs.reserve(op.srcs.size());
  for (const int src : op.srcs) {
    refs.push_back(get<FramesPtr<media>>(regs, src).get());
  }
  if (!op.batch) {
    return convert_frames<media>(refs.at(0), op.storage);
  }
  return convert_frames<media>(refs, op.storage);
}

void execute(Registers& regs, const ConvertOp& op) {
  regs.at(op.out) = dispatch(op.media, [&](auto media) -> Value {
    return convert<media.value>(regs, op);
  });
}

////////////////////////////////////////////////////////////////////////////////
// Clone / slice
////////////////////////////////////////////////////////////////////////////////
void execute(Registers& regs, const CloneOp& op) {
  regs.at(op.out) = std::visit(
      [](const auto& v) -> Value {
        using T = std::decay_t<decltype(v)>;
        if constexpr (
            std::is_same_v<T, AudioPacketsPtr> ||
            std::is_same_v<T, VideoPacketsPtr> ||
            std::is_same_v<T, ImagePacketsPtr>) {
          return std::make_unique<typename T::element_type>(*v);
        } else if constexpr (
            std::is_same_v<T, AudioFramesPtr> ||
            std::is_same_v<T, VideoFramesPtr> ||
            std::is_same_v<T, ImageFramesPtr>) {
          return v->clone();
        } else {
          throw std::runtime_error("Only packets and frames can be cloned.");
        }
      },
      regs.at(op.src));
}

// Mirrors `PySlice_Unpack` followed by `PySlice_AdjustIndices`.
// The results fit in `int`: start and stop are adjusted to [-1, len], and
// a step larger than `len` in magnitude selects the same frames as `len`.
std::tuple<int, int, int> adjust_slice(const PySlice& s, int64_t len) {
  const int64_t requested_step = std::get<2>(s).value_or(1);
  if (requested_step == 0) {
    throw std::invalid_argument("slice step cannot be zero");
  }
  const int64_t max_step = std::max<int64_t>(len, 1);
  const int64_t step = std::clamp(requested_step, -max_step, max_step);
  constexpr auto kMax = std::numeric_limits<int64_t>::max();
  constexpr auto kMin = std::numeric_limits<int64_t>::min();
  int64_t start = std::get<0>(s).value_or(step < 0 ? kMax : 0);
  int64_t stop = std::get<1>(s).value_or(step < 0 ? kMin : kMax);
  auto adjust = [&](int64_t& v) {
    if (v < 0) {
      v += len;
      if (v < 0) {
        v = step < 0 ? -1 : 0;
      }
    } else if (v >= len) {
      v = step < 0 ? len - 1 : len;
    }
  };
  adjust(start);
  adjust(stop);
  return {
      static_cast<int>(start), static_cast<int>(stop), static_cast<int>(step)};
}

void execute(Registers& regs, const SliceOp& op) {
  const auto& frames = get<VideoFramesPtr>(regs, op.src);
  regs.at(op.out) = std::visit(
      [&](const auto& key) -> Value {
        using T = std::decay_t<decltype(key)>;
        if constexpr (std::is_same_v<T, PySlice>) {
          auto [start, stop, step] =
              adjust_slice(key, frames->get_num_frames());
          return frames->slice(start, stop, step);
        } else {
          return frames->slice(key);
        }
      },
      op.key);
}

////////////////////////////////////////////////////////////////////////////////
// Query
////////////////////////////////////////////////////////////////////////////////
[[noreturn]] void unsupported_query(std::string_view type, const QueryOp& op) {
  throw std::invalid_argument(
      fmt::format("`{}` does not support query `{}`.", type, op.name));
}

std::tuple<int64_t, int64_t> to_tuple(const Rational& r) {
  return {r.num, r.den};
}

template <MediaType media>
Value query_codec(const Codec<media>& c, const QueryOp& op) {
  const auto& n = op.name;
  if (n == "name") {
    return Meta{c.get_name()};
  }
  if (n == "time_base") {
    return Meta{to_tuple(c.get_time_base())};
  }
  if constexpr (media == MediaType::Audio) {
    if (n == "sample_rate") {
      return Meta{int64_t{c.get_sample_rate()}};
    }
    if (n == "num_channels") {
      return Meta{int64_t{c.get_num_channels()}};
    }
    if (n == "sample_fmt") {
      return Meta{c.get_sample_fmt()};
    }
    if (n == "channel_layout") {
      return Meta{c.get_channel_layout()};
    }
  } else {
    if (n == "width") {
      return Meta{int64_t{c.get_width()}};
    }
    if (n == "height") {
      return Meta{int64_t{c.get_height()}};
    }
    if (n == "pix_fmt") {
      return Meta{c.get_pix_fmt()};
    }
    if (n == "sample_aspect_ratio") {
      return Meta{to_tuple(c.get_sample_aspect_ratio())};
    }
    if constexpr (media == MediaType::Video) {
      if (n == "frame_rate") {
        return Meta{to_tuple(c.get_frame_rate())};
      }
    }
  }
  unsupported_query("Codec", op);
}

template <MediaType media>
const Codec<media>& get_codec(const Packets<media>& p) {
  if (!p.codec) {
    throw std::runtime_error("Packet does not have codec info.");
  }
  return *p.codec;
}

template <MediaType media>
Value query_packets(const Packets<media>& p, const QueryOp& op) {
  const auto& n = op.name;
  if (n == "codec") {
    if (!p.codec) {
      return Meta{};
    }
    return Value{std::in_place_type<Codec<media>>, *p.codec};
  }
  if constexpr (media != MediaType::Image) {
    if (n == "timestamp") {
      if (!p.timestamp) {
        return Meta{};
      }
      auto [s, e] = *p.timestamp;
      return Meta{std::tuple<double, double>{to_double(s), to_double(e)}};
    }
  }
  if constexpr (media == MediaType::Audio) {
    if (n == "sample_rate") {
      return Meta{int64_t{get_codec(p).get_sample_rate()}};
    }
    if (n == "num_channels") {
      return Meta{int64_t{get_codec(p).get_num_channels()}};
    }
  } else {
    if (n == "width") {
      return Meta{int64_t{get_codec(p).get_width()}};
    }
    if (n == "height") {
      return Meta{int64_t{get_codec(p).get_height()}};
    }
    if (n == "pix_fmt") {
      return Meta{get_codec(p).get_pix_fmt()};
    }
  }
  if constexpr (media != MediaType::Image) {
    if (n == "__len__") {
      return Meta{static_cast<int64_t>(get_timestamps(p).size())};
    }
  }
  if constexpr (media == MediaType::Video) {
    if (n == "frame_rate") {
      return Meta{to_tuple(get_codec(p).get_frame_rate())};
    }
    if (n == "get_timestamps") {
      return Meta{get_timestamps(p, op.raw)};
    }
  }
  unsupported_query("Packets", op);
}

template <MediaType media>
Value query_frames(const Frames<media>& f, const QueryOp& op) {
  const auto& n = op.name;
  if constexpr (media != MediaType::Image) {
    if (n == "__len__") {
      return Meta{int64_t{f.get_num_frames()}};
    }
  }
  if constexpr (media == MediaType::Audio) {
    if (n == "num_frames") {
      return Meta{int64_t{f.get_num_frames()}};
    }
    if (n == "sample_rate") {
      return Meta{int64_t{f.get_sample_rate()}};
    }
    if (n == "num_channels") {
      return Meta{int64_t{f.get_num_channels()}};
    }
    if (n == "sample_fmt") {
      return Meta{std::string{f.get_media_format_name()}};
    }
  } else {
    if (n == "num_planes") {
      return Meta{int64_t{f.get_num_planes()}};
    }
    if (n == "width") {
      return Meta{int64_t{f.get_width()}};
    }
    if (n == "height") {
      return Meta{int64_t{f.get_height()}};
    }
    if (n == "pix_fmt") {
      return Meta{std::string{f.get_media_format_name()}};
    }
  }
  if constexpr (media == MediaType::Video) {
    const int num = f.get_num_frames();
    if (n == "num_frames") {
      return Meta{int64_t{num}};
    }
    if (n == "time_base") {
      return Meta{to_tuple(f.get_time_base())};
    }
    if (n == "get_pts") {
      std::vector<int64_t> ret;
      ret.reserve(num);
      for (int i = 0; i < num; ++i) {
        ret.push_back(f.get_pts(i));
      }
      return Meta{std::move(ret)};
    }
    if (n == "get_timestamps") {
      std::vector<double> ret;
      ret.reserve(num);
      for (int i = 0; i < num; ++i) {
        ret.push_back(f.get_timestamp(i));
      }
      return Meta{std::move(ret)};
    }
  }
  if constexpr (media == MediaType::Image) {
    if (n == "pts") {
      return Meta{f.get_timestamp()};
    }
  }
  unsupported_query("Frames", op);
}

Value query_demuxer(const Demuxer& d, const QueryOp& op) {
  const auto& n = op.name;
  if (n == "has_audio") {
    return Meta{d.has_audio()};
  }
  if (n == "audio_stream_index") {
    return Meta{int64_t{d.get_default_stream_index<MediaType::Audio>()}};
  }
  if (n == "video_stream_index") {
    return Meta{int64_t{d.get_default_stream_index<MediaType::Video>()}};
  }
  if (n == "audio_codec") {
    return d.get_default_codec<MediaType::Audio>();
  }
  if (n == "video_codec") {
    return d.get_default_codec<MediaType::Video>();
  }
  if (n == "image_codec") {
    return d.get_default_codec<MediaType::Image>();
  }
  unsupported_query("Demuxer", op);
}

void execute(Registers& regs, const QueryOp& op) {
  regs.at(op.out) = std::visit(
      [&](const auto& v) -> Value {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, DemuxerPtr>) {
          return query_demuxer(*v, op);
        } else if constexpr (
            std::is_same_v<T, AudioPacketsPtr> ||
            std::is_same_v<T, VideoPacketsPtr> ||
            std::is_same_v<T, ImagePacketsPtr>) {
          return query_packets(*v, op);
        } else if constexpr (
            std::is_same_v<T, AudioFramesPtr> ||
            std::is_same_v<T, VideoFramesPtr> ||
            std::is_same_v<T, ImageFramesPtr>) {
          return query_frames(*v, op);
        } else if constexpr (
            std::is_same_v<T, AudioCodec> || std::is_same_v<T, VideoCodec> ||
            std::is_same_v<T, ImageCodec>) {
          return query_codec(v, op);
        } else {
          unsupported_query("The value", op);
        }
      },
      regs.at(op.src));
}

////////////////////////////////////////////////////////////////////////////////
// Python conversion
////////////////////////////////////////////////////////////////////////////////
Value to_source(nb::handle obj) {
  if (nb::isinstance<nb::str>(obj)) {
    return nb::cast<std::string>(obj);
  }
  if (PyMemoryView_Check(obj.ptr())) {
    return ::spdl::detail::memoryview_to_sv(nb::borrow<nb::memoryview>(obj));
  }
  throw nb::type_error("Fused function sources must be `str` or `memoryview`.");
}

nb::object to_python(Meta&& meta) {
  return std::visit(
      [](auto&& v) -> nb::object {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return nb::none();
        } else {
          return nb::cast(std::move(v));
        }
      },
      std::move(meta));
}

nb::object to_python(Value&& value) {
  return std::visit(
      [](auto&& v) -> nb::object {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, Meta>) {
          return to_python(std::move(v));
        } else if constexpr (
            std::is_same_v<T, std::monostate> ||
            std::is_same_v<T, std::string> ||
            std::is_same_v<T, std::string_view> ||
            std::is_same_v<T, DemuxerPtr>) {
          throw std::runtime_error("The value cannot be returned.");
        } else {
          return nb::cast(std::move(v));
        }
      },
      std::move(value));
}

// Convert a register to Python for an intermediate flush of a lazy run.
// Metadata and codecs are copied, so that they remain usable by later
// operations. Other values are moved.
nb::object export_value(Value& value) {
  if (const auto* meta = std::get_if<Meta>(&value)) {
    return to_python(Meta{*meta});
  }
  if (const auto* c = std::get_if<AudioCodec>(&value)) {
    return nb::cast(AudioCodec{*c});
  }
  if (const auto* c = std::get_if<VideoCodec>(&value)) {
    return nb::cast(VideoCodec{*c});
  }
  if (const auto* c = std::get_if<ImageCodec>(&value)) {
    return nb::cast(ImageCodec{*c});
  }
  auto ret = to_python(std::move(value));
  value = std::monostate{};
  return ret;
}

// Registers that persist across multiple plans, used when a fused function
// needs intermediate values in Python (lazy execution).
struct FusedState {
  Registers regs;

  FusedState() = default;
  FusedState(const FusedState&) = delete;
  FusedState& operator=(const FusedState&) = delete;
  FusedState(FusedState&&) = delete;
  FusedState& operator=(FusedState&&) = delete;
  ~FusedState() = default;

  void bind(const std::vector<int>& inputs, const nb::list& srcs) {
    if (inputs.size() != nb::len(srcs)) {
      throw std::invalid_argument("The number of inputs and sources differ.");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (static_cast<size_t>(inputs[i]) >= regs.size()) {
        regs.resize(inputs[i] + 1);
      }
      regs.at(inputs[i]) = to_source(srcs[i]);
    }
  }

  // The values of the dead symbols are released in the native calls (see
  // `_LazyGraph`), so the remaining values are cheap to release. Releasing
  // them without releasing the GIL avoids waiting for the GIL once more.
  void close() {
    regs.clear();
  }

  size_t num_values() const {
    return std::count_if(regs.begin(), regs.end(), [](const Value& v) {
      return !std::holds_alternative<std::monostate>(v);
    });
  }
};

std::optional<TimeWindow> to_window(const std::optional<PyWindow>& window) {
  if (!window) {
    return std::nullopt;
  }
  return std::make_tuple(
      make_rational(std::get<0>(*window)), make_rational(std::get<1>(*window)));
}

////////////////////////////////////////////////////////////////////////////////
// Plan
////////////////////////////////////////////////////////////////////////////////

// A straight-line program of SPDL IO operations executed with one call
// from Python. Registers are allocated per call, so a plan can be run
// concurrently from multiple threads.
class FusedPlan {
  int num_registers_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<Op> ops_;

  void check(int i) const {
    if (i < 0 || i >= num_registers_) {
      throw std::out_of_range(fmt::format("Invalid register: {}", i));
    }
  }

  void check(const std::vector<int>& regs) const {
    for (const int i : regs) {
      check(i);
    }
  }

  void execute_ops(Registers& regs) const {
    for (const auto& op : ops_) {
      std::visit([&](const auto& o) { execute(regs, o); }, op);
    }
  }

  std::vector<Value> execute_all(Registers& regs) const {
    execute_ops(regs);
    std::vector<Value> ret;
    ret.reserve(outputs_.size());
    for (const int o : outputs_) {
      ret.push_back(std::move(regs.at(o)));
    }
    regs.clear();
    return ret;
  }

 public:
  FusedPlan(
      int num_registers,
      std::vector<int> inputs,
      std::vector<int> outputs)
      : num_registers_{num_registers},
        inputs_{std::move(inputs)},
        outputs_{std::move(outputs)} {
    check(inputs_);
    check(outputs_);
  }

  void add_open(
      int src,
      int out,
      const std::optional<DemuxConfig>& demux_config,
      const std::optional<std::string>& name) {
    check({src, out});
    ops_.emplace_back(
        OpenOp{
            .src = src,
            .out = out,
            .demux_config = demux_config,
            .name = name});
  }

  void add_demux(
      const std::string& media,
      int src,
      int out,
      const std::optional<PyWindow>& window,
      const std::optional<std::string>& bsf) {
    check({src, out});
    ops_.emplace_back(
        DemuxOp{
            .media = parse_media(media),
            .src = src,
            .out = out,
            .window = to_window(window),
            .bsf = bsf});
  }

  void add_drop(const std::vector<int>& regs) {
    check(regs);
    ops_.emplace_back(DropOp{.regs = regs});
  }

  void
  add_bsf(const std::string& media, int src, int out, const std::string& bsf) {
    check({src, out});
    ops_.emplace_back(
        BSFOp{.media = parse_media(media), .src = src, .out = out, .bsf = bsf});
  }

  void add_decode(
      const std::string& media,
      int src,
      int out,
      const std::optional<std::string>& filter_desc,
      const std::optional<DecodeConfig>& decode_config,
      int num_frames) {
    check({src, out});
    ops_.emplace_back(
        DecodeOp{
            .media = parse_media(media),
            .src = src,
            .out = out,
            .filter_desc = filter_desc,
            .decode_config = decode_config,
            .num_frames = num_frames});
  }

  void add_sample_decode(
      int src,
      const std::vector<int>& outs,
      const std::vector<int64_t>& indices,
      const std::optional<std::string>& filter_desc,
      const std::optional<DecodeConfig>& decode_config,
      bool optimized) {
    check(src);
    check(outs);
    if (indices.empty() || indices.size() != outs.size()) {
      throw std::invalid_argument("Invalid indices for sample decoding.");
    }
    ops_.emplace_back(
        SampleDecodeOp{
            .src = src,
            .outs = outs,
            .indices = indices,
            .filter_desc = filter_desc,
            .decode_config = decode_config,
            .optimized = optimized});
  }

  void add_convert(
      const std::string& media,
      const std::vector<int>& srcs,
      int out,
      bool batch,
      std::shared_ptr<CPUStorage> storage) {
    if (srcs.empty() || (!batch && srcs.size() != 1)) {
      throw std::invalid_argument("Invalid number of inputs for conversion.");
    }
    check(srcs);
    check(out);
    ops_.emplace_back(
        ConvertOp{
            .media = parse_media(media),
            .srcs = srcs,
            .out = out,
            .batch = batch,
            .storage = std::move(storage)});
  }

  void add_clone(int src, int out) {
    check({src, out});
    ops_.emplace_back(CloneOp{.src = src, .out = out});
  }

  void add_slice(int src, int out, SliceKey key) {
    check({src, out});
    ops_.emplace_back(SliceOp{.src = src, .out = out, .key = std::move(key)});
  }

  void add_query(int src, int out, const std::string& name, bool raw) {
    check({src, out});
    ops_.emplace_back(
        QueryOp{.src = src, .out = out, .name = name, .raw = raw});
  }

  size_t num_ops() const {
    return ops_.size();
  }

  // Execute the plan on registers that persist across plans.
  // The caller must keep the sources bound to `state` alive.
  nb::list run_on(FusedState& state, const std::vector<int>& exports) const {
    if (state.regs.size() < static_cast<size_t>(num_registers_)) {
      state.regs.resize(num_registers_);
    }
    for (const int i : exports) {
      check(i);
    }
    {
      nb::gil_scoped_release __g;
      execute_ops(state.regs);
    }
    nb::list ret;
    for (const int i : exports) {
      ret.append(export_value(state.regs.at(i)));
    }
    return ret;
  }

  // The caller must keep `srcs` alive until this function returns.
  nb::list run(const nb::list& srcs) const {
    if (nb::len(srcs) != inputs_.size()) {
      throw nb::value_error(
          fmt::format(
              "Expected {} sources. Found {}.", inputs_.size(), nb::len(srcs))
              .c_str());
    }
    Registers regs(num_registers_);
    for (size_t i = 0; i < inputs_.size(); ++i) {
      regs[inputs_[i]] = to_source(srcs[i]);
    }
    std::vector<Value> outputs;
    {
      nb::gil_scoped_release __g;
      outputs = execute_all(regs);
    }
    nb::list ret;
    for (auto& v : outputs) {
      ret.append(to_python(std::move(v)));
    }
    return ret;
  }
};

} // namespace

void register_fused(nb::module_& m) {
  nb::class_<FusedState>(m, "FusedState")
      .def(nb::init<>())
      .def("bind", &FusedState::bind, nb::arg("inputs"), nb::arg("srcs"))
      .def("close", &FusedState::close)
      .def_prop_ro("num_values", &FusedState::num_values);

  nb::class_<FusedPlan>(m, "FusedPlan")
      .def(
          nb::init<int, std::vector<int>, std::vector<int>>(),
          nb::arg("num_registers"),
          nb::arg("inputs"),
          nb::arg("outputs"))
      .def(
          "add_open",
          &FusedPlan::add_open,
          nb::arg("src"),
          nb::arg("out"),
          nb::kw_only(),
          nb::arg("demux_config") = nb::none(),
          nb::arg("name") = nb::none())
      .def(
          "add_demux",
          &FusedPlan::add_demux,
          nb::arg("media"),
          nb::arg("src"),
          nb::arg("out"),
          nb::kw_only(),
          nb::arg("window") = nb::none(),
          nb::arg("bsf") = nb::none())
      .def("add_drop", &FusedPlan::add_drop, nb::arg("regs"))
      .def(
          "add_bsf",
          &FusedPlan::add_bsf,
          nb::arg("media"),
          nb::arg("src"),
          nb::arg("out"),
          nb::kw_only(),
          nb::arg("bsf"))
      .def(
          "add_decode",
          &FusedPlan::add_decode,
          nb::arg("media"),
          nb::arg("src"),
          nb::arg("out"),
          nb::kw_only(),
          nb::arg("filter_desc") = nb::none(),
          nb::arg("decode_config") = nb::none(),
          nb::arg("num_frames") = -1)
      .def(
          "add_sample_decode",
          &FusedPlan::add_sample_decode,
          nb::arg("src"),
          nb::arg("outs"),
          nb::kw_only(),
          nb::arg("indices"),
          nb::arg("filter_desc") = nb::none(),
          nb::arg("decode_config") = nb::none(),
          nb::arg("optimized") = true)
      .def(
          "add_convert",
          &FusedPlan::add_convert,
          nb::arg("media"),
          nb::arg("srcs"),
          nb::arg("out"),
          nb::kw_only(),
          nb::arg("batch"),
          nb::arg("storage") = nullptr)
      .def("add_clone", &FusedPlan::add_clone, nb::arg("src"), nb::arg("out"))
      .def(
          "add_slice_index",
          [](FusedPlan& self, int src, int out, int64_t index) {
            self.add_slice(src, out, index);
          },
          nb::arg("src"),
          nb::arg("out"),
          nb::kw_only(),
          nb::arg("index"))
      .def(
          "add_slice_range",
          [](FusedPlan& self,
             int src,
             int out,
             std::optional<int64_t> start,
             std::optional<int64_t> stop,
             std::optional<int64_t> step) {
            self.add_slice(src, out, PySlice{start, stop, step});
          },
          nb::arg("src"),
          nb::arg("out"),
          nb::kw_only(),
          nb::arg("start").none(),
          nb::arg("stop").none(),
          nb::arg("step").none())
      .def(
          "add_slice_indices",
          [](FusedPlan& self, int src, int out, std::vector<int64_t> indices) {
            self.add_slice(src, out, std::move(indices));
          },
          nb::arg("src"),
          nb::arg("out"),
          nb::kw_only(),
          nb::arg("indices"))
      .def(
          "add_query",
          &FusedPlan::add_query,
          nb::arg("src"),
          nb::arg("out"),
          nb::kw_only(),
          nb::arg("name"),
          nb::arg("raw") = false)
      .def_prop_ro("num_ops", &FusedPlan::num_ops)
      .def_static(
          "window_to_timestamp",
          [](const PyWindow& window) {
            auto [start, end] = *to_window(window);
            return std::make_tuple(to_double(start), to_double(end));
          },
          nb::arg("window"),
          "The value of ``packets.timestamp`` for packets demuxed with ``window``.")
      .def_static(
          "filter_graph",
          &get_filter_graph<MediaType::Audio>,
          nb::arg("codec"),
          nb::arg("filter_desc").none(),
          "The filter graph that decoding builds for the codec and the filter body.")
      .def_static(
          "filter_graph",
          &get_filter_graph<MediaType::Video>,
          nb::arg("codec"),
          nb::arg("filter_desc").none())
      .def_static(
          "filter_graph",
          &get_filter_graph<MediaType::Image>,
          nb::arg("codec"),
          nb::arg("filter_desc").none())
      .def("run", &FusedPlan::run, nb::arg("srcs"))
      .def("run_on", &FusedPlan::run_on, nb::arg("state"), nb::arg("exports"));
}
} // namespace spdl::core
