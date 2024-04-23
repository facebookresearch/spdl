"""Custom resolver to make dynamic attributes available."""

from typing import List

import spdl
from griffe import Extension, get_logger, Module
from griffe.dataclasses import Alias

logger = get_logger(__name__)


def _print_members(pkg):
    logger.info(pkg)
    for k, v in pkg.all_members.items():
        logger.info(f"    {k}: {v}")


def _get_module(pkg: Module, path: List[str]):
    for submod in path:
        pkg = pkg[submod]
    return pkg


def _get_spdl_module(path: List[str]):
    mod = spdl
    for submod in path:
        mod = getattr(mod, submod)
    return mod


def _assign(pkg: Module, src: List[str], tgt: List[str]):
    src_pkg = _get_module(pkg, src)
    tgt_pkg = _get_module(pkg, tgt)

    logger.info(f"Copying attributes from: {'.'.join(src)} to {'.'.join(tgt)}")
    for attr in _get_spdl_module(src).__all__:
        if attr.startswith("_"):
            continue
        logger.info(f"  {attr}")
        tgt_pkg.set_member(attr, Alias(attr, src_pkg[attr]))


class SPDLDynamicResolver(Extension):
    """Add the entries for dynamically retrieved attributes."""

    def on_package_loaded(self, pkg: Module) -> None:
        """Add the entries for dynamically retrieved attributes."""
        import spdl.io
        import spdl.utils

        _print_members(pkg)

        # Populate dynamic attributes
        for mod in spdl.io._doc_submodules:
            _assign(pkg, ["io", mod], ["io"])

        for mod in spdl.utils._doc_submodules:
            _assign(pkg, ["utils", mod], ["utils"])

        super().on_package_loaded(pkg=pkg)


def _test():
    import logging

    import griffe

    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")

    extensions = griffe.load_extensions([SPDLDynamicResolver])
    griffe.load("spdl", extensions=extensions)


if __name__ == "__main__":
    _test()