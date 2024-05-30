"""Custom resolver to make dynamic attributes available."""

from griffe import Extension, get_logger, Module
from griffe.dataclasses import Alias

logger = get_logger(__name__)


def _print_members(pkg):
    logger.info(pkg)
    for k, v in pkg.all_members.items():
        logger.info(f"    {k}: {v}")


def _get_module(pkg: Module, path: list[str]):
    for submod in path:
        pkg = pkg[submod]
    return pkg


def _get_spdl_module(path: list[str]):
    import spdl

    mod = spdl
    for submod in path:
        mod = getattr(mod, submod)
    return mod


def _assign(pkg: Module, src: list[str], tgt: list[str]):
    src_pkg = _get_module(pkg, src)
    tgt_pkg = _get_module(pkg, tgt)

    logger.info(f"Copying attributes from spdl.{'.'.join(src)} to spdl.{'.'.join(tgt)}")
    for attr in _get_spdl_module(src).__all__:
        if attr.startswith("_"):
            continue
        logger.info(f"  {attr}")
        tgt_pkg.set_member(attr, Alias(attr, src_pkg[attr]))


def _process_spdl(pkg):
    _print_members(pkg)

    if pkg.name != "spdl":
        return

    import spdl.io
    import spdl.utils

    # Populate dynamic attributes
    for mod in spdl.io._doc_submodules:
        _assign(pkg, ["io", mod], ["io"])

    for mod in spdl.utils._doc_submodules:
        _assign(pkg, ["utils", mod], ["utils"])


class SPDLDynamicResolver(Extension):
    """Add the entries for dynamically retrieved attributes."""

    def on_package_loaded(self, pkg: Module) -> None:
        """Add the entries for dynamically retrieved attributes."""
        try:
            _process_spdl(pkg)
        except Exception:
            logger.exception("Failed to import spdl.")
        finally:
            super().on_package_loaded(pkg=pkg)

    def on_module_instance(self, *, node, mod) -> None:

        super().on_module_instance(node=node, mod=mod)
        if mod.name in ["io", "dataset"]:
            if mod.parent.name == "spdl":
                mod.docstring = None


def _test():
    import logging

    import griffe

    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")

    extensions = griffe.load_extensions([SPDLDynamicResolver])
    griffe.load("spdl", extensions=extensions)


if __name__ == "__main__":
    _test()
