import os
import subprocess

from setuptools import find_packages, Extension, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = os.path.dirname(__file__)

ext_modules = [
    Extension("spdl.lib._spdl_ffmpeg4", sources=[]),
    Extension("spdl.lib._spdl_ffmpeg5", sources=[]),
    Extension("spdl.lib._spdl_ffmpeg6", sources=[]),
]

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

        cfg = "Debug" if self.debug else "Release"

        # fmt: off
        config_cmd = [
            "cmake",
            "-B", self.build_temp,
            "-S", ROOT_DIR,
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            f"-DSPDL_BUILD_SAMPLES:BOOL=OFF"
            "-GNinja",
        ]

        build_cmd = [
            "cmake",
            "--build", self.build_temp,
            "--target", "install",
        ]
        # fmt: on

        subprocess.check_call(config_cmd)
        subprocess.check_call(build_cmd)


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
        zip_safe=False,
        python_requires=">=3.9",
    )


if __name__ == "__main__":
    main()
