# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory

from spdl.dataloader import ImageNet, LocalDirectory


def _make_files(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_LocalDirectory():
    """LocalDirectory can traverse specified files"""
    with TemporaryDirectory() as root_dir:
        root_dir = Path(root_dir)

        targets = {
            root_dir / "foo.txt",
            root_dir / "dir" / "bar.txt",
            root_dir / "dir" / "dir" / "bazz.txt",
        }
        others = {
            root_dir / "foo.dat",
            root_dir / "dir" / "bar.dat",
            root_dir / "dir" / "dir" / "bazz.dat",
        }
        _make_files(targets | others)

        src = LocalDirectory(root=root_dir, pattern="**/*.txt")

        vals1 = list(src)
        vals2 = list(src)
        src.shuffle(seed=0)
        vals3 = list(src)

        assert vals1 == vals2 != vals3
        assert set(vals1) == set(vals2) == set(vals3) == targets


def test_ImageNet():
    """ImageNet returns image path and class ID"""
    with TemporaryDirectory() as root_dir:
        root_dir = Path(root_dir)

        vals = {
            (root_dir / "val" / "n02110958" / "FOO.JPEG", 254),
            (root_dir / "val" / "n02027492" / "FOO.JPEG", 140),
            (root_dir / "val" / "n02071294" / "FOO.JPEG", 148),
            (root_dir / "val" / "n02088632" / "FOO.JPEG", 164),
        }
        trains = {
            (root_dir / "train" / "n02066245" / "FOO.JPEG", 147),
            (root_dir / "train" / "n02277742" / "FOO.JPEG", 322),
            (root_dir / "train" / "n02965783" / "FOO.JPEG", 475),
            (root_dir / "train" / "n03240683" / "FOO.JPEG", 540),
        }
        _make_files([v for v, _ in vals])
        _make_files([v for v, _ in trains])

        src = ImageNet(root=root_dir, split="val")
        v1 = list(src)
        src.shuffle(0)
        v2 = list(src)
        assert v1 != v2
        assert set(v1) == set(v2) == vals

        src = ImageNet(root=root_dir, split="train")
        v1 = list(src)
        src.shuffle(0)
        v2 = list(src)
        assert v1 != v2
        assert set(v1) == set(v2) == trains
