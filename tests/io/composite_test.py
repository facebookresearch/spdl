# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import builtins
import inspect
import unittest
from typing import Any
from unittest import mock

import spdl.io
from spdl.io import _composite


class CompositeSignatureTest(unittest.TestCase):
    def test_optional_extension_absence_keeps_oss_enabled(self) -> None:
        """The generic fallback works when the optional extension is not shipped."""
        tree = ast.parse(inspect.getsource(_composite))
        guard = next(
            node
            for node in tree.body
            if isinstance(node, ast.Try)
            and any(
                isinstance(item, ast.ImportFrom) and item.module == "fb"
                for item in node.body
            )
        )
        module = ast.Module(body=[guard], type_ignores=[])
        namespace: dict[str, Any] = {
            "__name__": "spdl.io._composite",
            "__package__": "spdl.io",
        }
        with mock.patch.object(
            builtins, "__import__", side_effect=ImportError("extension not shipped")
        ):
            exec(compile(module, "<optional-extension-fallback>", "exec"), namespace)
        self.assertTrue(namespace["_is_sample_decode_enabled"]())
        self.assertTrue(namespace["_is_sample_decode_enabled"]())

    def test_import_time_policy_is_snapshotted_once(self) -> None:
        """The module stores one boolean instead of calling policy during decode."""
        tree = ast.parse(inspect.getsource(_composite))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_SAMPLE_DECODE_ENABLED"
                for target in node.targets
            )
        )
        module = ast.Module(body=[assignment], type_ignores=[])
        for value in (True, False):
            with self.subTest(value=value):
                policy = mock.Mock(return_value=value)
                namespace: dict[str, Any] = {"_is_sample_decode_enabled": policy}
                exec(compile(module, "<import-time-policy>", "exec"), namespace)
                policy.assert_called_once_with()
                policy.side_effect = AssertionError("lookup after import")
                for _ in range(3):
                    self.assertIs(namespace["_SAMPLE_DECODE_ENABLED"], value)
                policy.assert_called_once_with()

    def test_loaders_do_not_accept_arbitrary_kwargs(self) -> None:
        """Composite loaders expose only supported keyword arguments."""
        loaders = (
            _composite.load_audio,
            _composite.load_video,
            _composite.load_image,
            _composite.load_image_batch,
        )

        for loader in loaders:
            with self.subTest(loader=loader.__name__):
                parameter_kinds = {
                    parameter.kind
                    for parameter in inspect.signature(loader).parameters.values()
                }
                self.assertNotIn(inspect.Parameter.VAR_KEYWORD, parameter_kinds)


class SampleDecodeSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(_composite, "_SAMPLE_DECODE_ENABLED", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_sparse_rgb_uses_compact_native_outputs(self) -> None:
        """Sparse RGB selection precedes conversion and retains only requested outputs."""
        output = [mock.sentinel.first, mock.sentinel.middle, mock.sentinel.last]
        with mock.patch.object(
            _composite._core, "decode_packets", return_value=output
        ) as decode:
            result = _composite._decode_selected(
                mock.sentinel.packets, [0, 2, 19], None, "format=pix_fmts=rgb24"
            )
        self.assertEqual(result, output)
        decode.assert_called_once_with(
            mock.sentinel.packets,
            decode_config=None,
            filter_desc="select=expr='eq(n,0)+eq(n,2)+eq(n,19)',format=pix_fmts=rgb24",
            num_frames=3,
        )

    def test_rollback_uses_original_split_and_prefix(self) -> None:
        """Disabling the internal policy restores original grouping and decoding."""
        packets = mock.MagicMock()
        packets.__len__.return_value = 50
        output = list(range(20))
        with (
            mock.patch.object(_composite, "_SAMPLE_DECODE_ENABLED", False),
            mock.patch.object(
                _composite._libspdl,
                "_extract_packets_at_indices",
                return_value=[(mock.sentinel.split, [19])],
            ) as extract,
            mock.patch.object(
                _composite._core, "decode_packets", return_value=output
            ) as decode,
            mock.patch.object(_composite, "_decode_selected") as optimized,
        ):
            result = spdl.io.sample_decode_video(packets, [49])
        self.assertEqual(result, [19])
        extract.assert_called_once_with(packets, [49], h264_idr_split=False)
        decode.assert_called_once_with(
            mock.sentinel.split,
            decode_config=None,
            filter_desc="format=pix_fmts=rgb24",
            num_frames=20,
        )
        optimized.assert_not_called()

    def test_enabled_policy_and_validation_order(self) -> None:
        """Decoding reads the cached flag without consulting policy on any request."""
        packets = mock.MagicMock()
        packets.__len__.return_value = 50
        with (
            mock.patch.object(
                _composite,
                "_is_sample_decode_enabled",
                side_effect=AssertionError("hot-path policy lookup"),
            ) as policy,
            mock.patch.object(
                _composite._libspdl,
                "_extract_packets_at_indices",
                return_value=[(mock.sentinel.split, [19])],
            ) as extract,
            mock.patch.object(
                _composite._core, "decode_packets", return_value=[mock.sentinel.last]
            ),
        ):
            for indices in ([49, 0], [49, 49]):
                with self.assertRaises(ValueError):
                    spdl.io.sample_decode_video(packets, indices)
            extract.assert_not_called()
            for _ in range(3):
                self.assertEqual(
                    spdl.io.sample_decode_video(packets, [49]), [mock.sentinel.last]
                )
            policy.assert_not_called()
            self.assertEqual(
                extract.call_args_list,
                [mock.call(packets, [49], h264_idr_split=True)] * 3,
            )

    def test_default_filter_and_local_indices(self) -> None:
        """The public helper resolves default RGB and uses extracted local ordinals."""
        packets = mock.MagicMock()
        packets.__len__.return_value = 50
        selected = [mock.sentinel.last]
        with (
            mock.patch.object(
                _composite._libspdl,
                "_extract_packets_at_indices",
                return_value=[(mock.sentinel.split, [19])],
            ),
            mock.patch.object(
                _composite._core, "decode_packets", return_value=selected
            ) as decode,
        ):
            result = spdl.io.sample_decode_video(packets, [49])
        self.assertEqual(result, selected)
        self.assertEqual(decode.call_args.args, (mock.sentinel.split,))
        self.assertEqual(decode.call_args.kwargs["num_frames"], 1)
        self.assertIn("eq(n,19)", decode.call_args.kwargs["filter_desc"])

    def test_dense_large_and_non_rgb_filters_use_prefix(self) -> None:
        """Dense requests and arbitrary filter semantics retain the original path."""
        cases: list[tuple[list[int], str | None]] = [
            (list(range(10)), "format=pix_fmts=rgb24")
        ]
        cases.append((list(range(0, 100, 3)), "format=pix_fmts=rgb24"))
        cases.extend(
            ([0, 2, 19], desc)
            for desc in (
                None,
                "format=pix_fmts=rgb8",
                "format=pix_fmts=rgba",
                "format=pix_fmts=rgb24|rgba",
                "format=pix_fmts=rgb24:color_spaces=bt709",
                "fps=10,format=pix_fmts=rgb24",
                "setpts=PTS+1/TB,format=pix_fmts=rgb24",
                "trim=start_frame=2,format=pix_fmts=rgb24",
                "tpad=stop=2,format=pix_fmts=rgb24",
                "scale=32:24,format=pix_fmts=rgb24",
                "crop=16:16,format=pix_fmts=rgb24",
                " format=pix_fmts=rgb24",
                "format=pix_fmts='rgb24'",
                "split[a][b];[a][b]hstack",
            )
        )
        for indices, desc in cases:
            with self.subTest(indices=indices, desc=desc):
                output = list(range(max(indices) + 1))
                with mock.patch.object(
                    _composite._core, "decode_packets", return_value=output
                ) as decode:
                    result = _composite._decode_selected(
                        mock.sentinel.packets, indices, None, desc
                    )
                self.assertEqual(result, indices)
                self.assertEqual(decode.call_args.kwargs["filter_desc"], desc)
                self.assertEqual(
                    decode.call_args.kwargs["num_frames"], max(indices) + 1
                )

    def test_short_selected_output_is_not_silently_accepted(self) -> None:
        """A missing requested frame raises instead of shortening sampled output."""
        with mock.patch.object(_composite._core, "decode_packets", return_value=[]):
            with self.assertRaises(IndexError):
                _composite._decode_selected(
                    mock.sentinel.packets, [19], None, "format=pix_fmts=rgb24"
                )

    def test_invalid_indices_keep_public_errors(self) -> None:
        """Invalid sample requests retain their existing validation errors."""
        packets = mock.MagicMock()
        packets.__len__.return_value = 20
        for indices, error in (
            ([], ValueError),
            ([1, 1], ValueError),
            ([2, 1], ValueError),
            ([-1], IndexError),
            ([20], IndexError),
        ):
            with self.subTest(indices=indices):
                with self.assertRaises(error):
                    spdl.io.sample_decode_video(packets, indices)


class LoadImageBatchNvjpegTest(unittest.TestCase):
    def test_forwards_memoryview_source(self) -> None:
        """The public loader normalizes and forwards borrowed image data."""
        source = memoryview(b"image")
        device_config = mock.sentinel.device_config

        with mock.patch.object(
            _composite._core,
            "decode_image_nvjpeg",
        ) as decode_image:
            spdl.io.load_image_batch_nvjpeg(
                [source],
                device_config=device_config,
                width=4,
                height=2,
            )

        decode_image.assert_called_once()
        forwarded_sources = decode_image.call_args.args[0]
        self.assertEqual(len(forwarded_sources), 1)
        self.assertIsInstance(forwarded_sources[0], memoryview)
        self.assertIs(forwarded_sources[0].obj, source.obj)
        self.assertEqual(
            decode_image.call_args.kwargs,
            {
                "scale_width": 4,
                "scale_height": 2,
                "device_config": device_config,
                "pix_fmt": "rgb",
            },
        )
