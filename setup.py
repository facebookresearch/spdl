import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = os.path.dirname(__file__)
TP_DIR = os.path.join(ROOT_DIR, "third_party")

ext_modules = [
    Extension("spdl.lib.libdouble-conversion", sources=[]),
    Extension("spdl.lib.libevent_core", sources=[]),
    Extension("spdl.lib.libfmt", sources=[]),
    Extension("spdl.lib.libgflags", sources=[]),
    Extension("spdl.lib.libglog", sources=[]),
    Extension("spdl.lib.libspdl_ffmpeg4", sources=[]),
    Extension("spdl.lib.libspdl_ffmpeg5", sources=[]),
    Extension("spdl.lib.libspdl_ffmpeg6", sources=[]),
    Extension("spdl.lib._spdl_ffmpeg4", sources=[]),
    Extension("spdl.lib._spdl_ffmpeg5", sources=[]),
    Extension("spdl.lib._spdl_ffmpeg6", sources=[]),
]


def _get_build_env(var, default=False):
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


_USE_CUDA = _get_build_env("USE_CUDA")
_DEBUG_REFCOUNT = _get_build_env("DEBUG_REFCOUNT")
_SKIP_FOLLY_DEPS = _get_build_env("SKIP_FOLLY_DEPS")


def _get_cmake_commands(build_dir, install_dir, debug):
    def _b(var):
        return "ON" if var else "OFF"

    cfg = "Debug" if debug else "Release"
    deps_build_dir = os.path.join(build_dir, "folly-deps")
    main_build_dir = os.path.join(build_dir, "main")
    deps_cmd = [
        # fmt: off
        [
            "cmake",
            "-S", TP_DIR,
            "-B", deps_build_dir,
            "-DCMAKE_VERBOSE_MAKEFILE=OFF",
            "-DCMAKE_INSTALL_MESSAGE=NEVER",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
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
            "-S", ROOT_DIR,
            "-B", main_build_dir,
            f"-DCMAKE_VERBOSE_MAKEFILE={'ON' if debug else 'OFF'}",
            f"-DCMAKE_INSTALL_MESSAGE={'ALWAYS' if debug else 'LAZY'}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_PREFIX_PATH={install_dir}",
            "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=false",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DSPDL_BUILD_PYTHON_BINDING=ON",
            f"-DSPDL_PYTHON_BINDING_INSTALL_PREFIX={install_dir}",
            f"-DSPDL_USE_CUDA={_b(_USE_CUDA)}",
            f"-DSPDL_DEBUG_REFCOUNT={_b(_DEBUG_REFCOUNT)}",
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
    if _SKIP_FOLLY_DEPS:
        return main_build_cmd
    return deps_cmd + main_build_cmd


def _fix_tp_library_name(path):
    parts = path.split(".")
    if sys.platform == "darwin":
        # remove ABI or replace it with version number
        if "gflags" in path:
            parts[-2] = "2.2"
        elif "glog" in path:
            parts[-2] = "0"
        elif "double-conversion" in path:
            parts[-2] = "3"
        elif "fmt" in path:
            parts[-2] = "10"
        elif "libevent" in path:
            del parts[-2]
            parts[-2] = f"{parts[-2]}-2.1.7"
        else:
            del parts[-2]
        # replace suffix
        parts[-1] = "dylib"
    elif sys.platform.startswith("linux"):
        # remove ABI
        del parts[-2]
        # append soversion
        if "gflags" in path:
            parts.append("2.2")
        elif "glog" in path:
            parts.append("0")
        elif "double-conversion" in path:
            parts.append("3")
        elif "fmt" in path:
            parts.append("10")
        elif "libevent" in path:
            parts[-2] += "-2.1"
            parts.append("7")
    return ".".join(parts)


BUILT_ONCE = False


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        super().run()

    def build_extension(self, ext):
        global BUILT_ONCE
        if BUILT_ONCE:
            return
        BUILT_ONCE = True

        extdir = os.path.dirname(self.get_ext_fullpath("foo"))
        install_dir = os.path.abspath(os.path.join(extdir, "spdl"))

        cmds = _get_cmake_commands(self.build_temp, install_dir, self.debug)

        for cmd in cmds:
            print(f"Running \"{' '.join(cmd)}\"", flush=True)
            subprocess.check_call(cmd)

    def get_ext_filename(self, fullname):
        ext_filename = super().get_ext_filename(fullname)
        if "_spdl_ffmpeg" not in ext_filename:
            ext_filename = _fix_tp_library_name(ext_filename)
        return ext_filename


def main():
    setup(
        name="spdl",
        version="0.0.1",
        author="Moto Hira",
        author_email="moto",
        description="SPDL: Scalable and Performant Data Loading.",
        long_description="Prototype data loader for fast multimedia processing.",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuild},
        python_requires=">=3.10",
        install_requires=[
            "numpy >= 1.21",
        ],
    )


if __name__ == "__main__":
    main()
