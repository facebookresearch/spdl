import logging

from griffe import Extension, get_logger, Module
from griffe.dataclasses import Alias

logger = get_logger(__name__)


def print_members(pkg):
    logger.info(pkg)
    for k, v in pkg.all_members.items():
        logger.info(f"    {k}: {v}")


class SPDLDynamicResolver(Extension):
    def on_package_loaded(self, pkg: Module) -> None:
        import spdl
        import spdl.io

        pkg["lib"].set_member("_libspdl", pkg["lib"]["_spdl_ffmpeg6"])

        print_members(pkg)
        print_members(pkg["lib"]["_libspdl"])

        # Populate dynamic attributes
        for attr in spdl.io._async.__all__:
            if attr.startswith("_"):
                continue
            logger.info("Setting: %s", attr)
            pkg["io"].set_member(attr, Alias(attr, pkg["io"]["_async"][attr]))

        for attr in spdl.io._convert.__all__:
            logger.info("Setting: %s", attr)
            pkg["io"].set_member(attr, Alias(attr, pkg["io"]["_convert"][attr]))

        super().on_package_loaded(pkg=pkg)


def _test():
    import logging

    import griffe

    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")

    extensions = griffe.load_extensions([SPDLDynamicResolver])
    spdl = griffe.load("spdl", extensions=extensions)


if __name__ == "__main__":
    _test()