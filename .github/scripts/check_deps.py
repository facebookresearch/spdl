#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import importlib.metadata


def check_package_meta():
    print("Verifying spdl_core package lists no dependency.")
    if importlib.metadata.requires("spdl_core"):
        raise ValueError("The spdl_core package must have no dependency.")

    print("Verifying spdl_io package lists only NumPy as dependency")
    deps = importlib.metadata.requires("spdl_io")
    if not (len(deps) == 1 and deps[0].startswith("numpy")):
        raise ValueError("Only NumPy is allowed as spdl_io package dependency.")


def _get_imported_3rd_party_modules():
    ret = set()
    for name, mod in sys.modules.items():
        if hasattr(mod, "__path__"):
            if any("site-packages" in p for p in mod.__path__):
                ret.add(name)
    return ret


def check_imported_modules():
    # --------------------------------------------------------------------------
    # Pre-condition
    # --------------------------------------------------------------------------
    print("Checking the target modules are not yet imported.")
    base_mods = _get_imported_3rd_party_modules()
    if violation := base_mods & set(("torch", "numpy", "jax", "spdl")):
        raise RuntimeError(
            "The following modules must not be imported before testing: " f"{violation}"
        )

    # --------------------------------------------------------------------------
    # Import must success without third party packages installed
    # --------------------------------------------------------------------------
    print("Testing the spdl_core module import")

    import spdl.pipeline

    mods = _get_imported_3rd_party_modules()
    assert "spdl.pipeline" in mods
    mods = {m for m in mods - base_mods if not m.startswith("spdl")}
    print(f"Modules imported with spdl.pipeline: {mods}")
    if violation := mods & set(("torch", "numpy", "jax")):
        raise RuntimeError(
            "`import spdl.pipeline` must not import third-party libraries. "
            f"Found: {violation}"
        )

    # --------------------------------------------------------------------------
    # Import (Same as above but NumPy is allowed
    # --------------------------------------------------------------------------
    print("Testing the spdl_io module import")

    import spdl.io

    mods = _get_imported_3rd_party_modules()
    assert "spdl.io" in mods
    mods = {m for m in mods - base_mods if not m.startswith("spdl")}
    print(f"Modules imported with spdl.io: {mods}")
    if violation := mods & set(("torch", "jax")):
        raise RuntimeError(
            "`import spdl.io` must not import third-party libraries except NumPy."
            f"Found: {violation}"
        )


def main():
    check_package_meta()
    check_imported_modules()
    print("OK")


if __name__ == "__main__":
    main()
