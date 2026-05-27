# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for ``python -m spdl.autoresearch``.

Importing this module is intentionally a no-op — the framework
dispatcher is loaded only when this module is executed as a script
(i.e. by ``python -m``). This keeps the import surface cheap for
tooling that may introspect the package without running the CLI.
"""

if __name__ == "__main__":
    from spdl.autoresearch._app._main import main

    main()
