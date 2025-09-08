# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import subprocess
import sys
import sysconfig

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

THIS_DIR = Path(__file__).parent
SRC_DIR = str(THIS_DIR / "src")
TP_DIR = str(THIS_DIR / "src" / "third_party")


def _env(var, default=False):
    if var not in os.environ:
        return default

    val = os.environ.get(var, "0")
    trues = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]
    falses = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]
    if val in trues:
        return True
    if val not in falses:
        print(
            f"WARNING: Unexpected environment variable value `{var}={val}`. "
            f"Expected one of {trues + falses}"
        )
    return False


_SPDL_USE_FFMPEG_VERSION = os.environ.get("SPDL_USE_FFMPEG_VERSION", "all")
_SPDL_USE_NVCODEC = _env("SPDL_USE_NVCODEC")
_SPDL_USE_NVJPEG = _env("SPDL_USE_NVJPEG")
_SPDL_USE_CUDA = _env("SPDL_USE_CUDA", _SPDL_USE_NVCODEC or _SPDL_USE_NVJPEG)
_SPDL_BUILD_STUB = _env("SPDL_BUILD_STUB", _SPDL_USE_CUDA)


def _is_gil_enabled():
    try:
        return sys._is_gil_enabled()
    except AttributeError:
        return True


def _get_ext_modules():
    ext_modules = []
    for v in ["4", "5", "6", "7", "8"]:
        if _SPDL_USE_FFMPEG_VERSION == "all" or _SPDL_USE_FFMPEG_VERSION == v:
            ext_modules.extend(
                [
                    Extension(f"spdl.io.lib.libspdl_ffmpeg{v}", sources=[]),
                    Extension(f"spdl.io.lib._spdl_ffmpeg{v}", sources=[]),
                ]
            )
    if ext_modules and _SPDL_BUILD_STUB:
        ext_modules.append(
            Extension("spdl.io.lib.__STUB__", sources=[]),
        )

    return ext_modules


def _get_cmake_commands(build_dir, install_dir, debug):
    def _b(var):
        return "ON" if var else "OFF"

    cfg = "Release"
    deps_build_dir = os.path.join(build_dir, "deps_first_stage")
    main_build_dir = os.path.join(build_dir, "main")
    deps_cmd = [
        # fmt: off
        [
            "cmake",
            "-S", str(TP_DIR),
            "-B", deps_build_dir,
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DCMAKE_VERBOSE_MAKEFILE=OFF",
            "-DCMAKE_INSTALL_MESSAGE=NEVER",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            f"-DCMAKE_INSTALL_PREFIX={build_dir}",
            "-GNinja",
        ],
        [
            "cmake",
            "--build", deps_build_dir,
            "--target", "install",
            "--config", cfg,
        ],
        # fmt: on
    ]
    main_build_cmd = [
        # fmt: off
        [
            "cmake",
            "-LAH",
            "-S", str(SRC_DIR),
            "-B", main_build_dir,
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            f"-DCMAKE_VERBOSE_MAKEFILE={'ON' if debug else 'OFF'}",
            f"-DCMAKE_INSTALL_MESSAGE={'ALWAYS' if debug else 'LAZY'}",
            f"-DCMAKE_INSTALL_PREFIX={build_dir}",
            f"-DCMAKE_PREFIX_PATH={build_dir}",
            "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=false",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DSPDL_BUILD_PYTHON_BINDING=ON",
            f"-DSPDL_PYTHON_BINDING_INSTALL_PREFIX={install_dir}",
            f"-DSPDL_IS_GIL_ENABLED={_b(_is_gil_enabled())}",
            ###################################################################
            # Options based on env vars
            ###################################################################
            f"-DSPDL_USE_TRACING={_b(_env('SPDL_USE_TRACING'))}",
            f"-DSPDL_USE_CUDA={_b(_SPDL_USE_CUDA)}",
            f"-DSPDL_USE_NVCODEC={_b(_SPDL_USE_NVCODEC)}",
            f"-DSPDL_USE_NVJPEG={_b(_SPDL_USE_NVJPEG)}",
            f"-DSPDL_USE_NPPI={_b(_env('SPDL_USE_NPPI'))}",
            f"-DSPDL_LINK_STATIC_NVJPEG={_b(_env('SPDL_LINK_STATIC_NVJPEG'))}",
            f"-DSPDL_USE_FFMPEG_VERSION={_SPDL_USE_FFMPEG_VERSION}",
            f"-DSPDL_DEBUG_REFCOUNT={_b(_env('SPDL_DEBUG_REFCOUNT'))}",
            f"-DSPDL_BUILD_STUB={_b(_SPDL_BUILD_STUB)}",
            ###################################################################
            f"-DPython_INCLUDE_DIR={sysconfig.get_paths()['include']}",
            "-GNinja",
        ],
        [
            "cmake",
            "--build", main_build_dir,
            "--target", "install",
            "--config", cfg,
        ],
        # fmt: on
    ]
    if _env("SPDL_SKIP_DEPS"):
        return main_build_cmd
    return deps_cmd + main_build_cmd


BUILT_ONCE = False


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        if _SPDL_USE_CUDA:
            try:
                subprocess.check_output(["nvcc", "--version"])
            except OSError:
                raise RuntimeError("`nvcc` is not available.") from None
        super().run()

    def build_extension(self, ext):
        global BUILT_ONCE
        if BUILT_ONCE:
            return
        BUILT_ONCE = True

        install_dir = os.path.abspath(os.path.join(self.build_lib, "spdl", "io"))

        cmds = _get_cmake_commands(self.build_temp, install_dir, self.debug)

        for cmd in cmds:
            print(f"Running \"{' '.join(cmd)}\"", flush=True)
            subprocess.check_call(cmd)

        # Copy public header files
        # src = os.path.abspath(
        #     os.path.join(self.build_lib, "spdl", "include", "libspdl")
        # )
        # build_py = self.get_finalized_command("build_py")
        # dst = build_py.get_package_dir("spdl.include.libspdl")
        # print(src, dst)
        # shutil.copytree(src, dst, dirs_exist_ok=True)

    def get_ext_filename(self, filename):
        ext_filename = super().get_ext_filename(filename)

        # Fix the library name for libspdl
        # linux: spdl/lib/libspdl_ffmpeg4.cpython-310-x86_64-linux-gnu.so -> spdl/lib/libspdl_ffmpeg4.so
        # macOS: spdl/lib/libspdl_ffmpeg4.cpython-310-x86_64-linux-gnu.so -> spdl/lib/libspdl_ffmpeg4.dylib
        if "libspdl_ffmpeg" in ext_filename:
            parts = ext_filename.split(".")
            del parts[-2]
            if sys.platform == "darwin":
                parts[-1] = "dylib"
            ext_filename = ".".join(parts)
        elif "__STUB__" in ext_filename:
            parts = ext_filename.split(".")
            parts = [parts[0].replace("__STUB__", "_libspdl.pyi")]
            ext_filename = ".".join(parts)
            print(ext_filename)
        return ext_filename


def main():
    with open(THIS_DIR / "VERSION", "r") as f:
        version = f.read().strip()

    exclude = find_packages(where="src", exclude=["spdl.io*"])
    packages = find_packages(where="src", exclude=exclude)

    if "bdist_wheel" in sys.argv:
        print("*" * 40)
        print("* BUILDING SPDL IO (EXPERIMENTAL)")
        for pkg in packages:
            print(f"* {pkg}")
        print("*" * 40)

    setup(
        name="spdl_io",
        version=version,
        packages=packages,
        package_dir={"": "src"},
        ext_modules=_get_ext_modules(),
        cmdclass={"build_ext": CMakeBuild},
    )


if __name__ == "__main__":
    main()
