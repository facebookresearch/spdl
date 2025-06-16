# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import importlib.metadata
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../../examples"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SPDL"
copyright = f"{datetime.today().strftime('%Y')}, Meta Platforms, Inc."
author = "Moto Hira"
release = importlib.metadata.version("spdl")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "breathe",
    "exhale",
]
autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False
autoclass_content = "class"
add_module_names = False
autodoc_default_options = {
    "special-members": ",".join(["__len__", "__getitem__", "__iter__", "__aiter__"]),
    "undoc-members": True,
    "exclude-members": ",".join(
        [
            "__init__",
        ]
    ),
}

templates_path = ["_templates"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
}
exclude_patterns = []


breathe_projects = {"libspdl": "generated/doxygen/xml/"}
breathe_default_project = "libspdl"
mermaid_version = "11.4.1"


def _get_source():
    import os

    base = "../../src/libspdl/core"
    return [f"{base}/{f}" for f in os.listdir(base) if f.endswith(".h")]


exhale_args = {
    # These arguments are required
    "containmentFolder": "./generated/libspdl",
    "rootFileName": "root.rst",
    "doxygenStripFromPath": "..",
    # Heavily encouraged optional argument (see docs)
    "rootFileTitle": "Libspdl API",
    # Suggested optional arguments
    "createTreeView": True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": f"INPUT = {','.join(_get_source())}",
    "contentsDirectives": False,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_theme_options = {
    "navigation_with_keys": True,
}
html_context = {
    "doc_versions": [
        ("dev", "/spdl/main"),
        ("latest-release", "/spdl/latest"),
        ("0.1.1", "/spdl/main"),
        ("0.1.0", "/spdl/0.1.0"),
        ("0.0.14", "/spdl/0.0.14"),
        ("0.0.13", "/spdl/0.0.13"),
        ("0.0.12", "/spdl/0.0.12"),
        ("0.0.11", "/spdl/0.0.11"),
        ("0.0.10", "/spdl/0.0.10"),
        ("0.0.9", "/spdl/0.0.9"),
        ("0.0.8", "/spdl/0.0.8"),
    ]
}


def linkcode_resolve(domain, info):
    import dataclasses
    import importlib
    import inspect

    if domain != "py":
        return None
    if not info["module"]:
        return None

    base = "https://github.com/facebookresearch/spdl/tree/main/"

    if info["module"].startswith("spdl"):
        base = f"{base}/src"
    else:
        base = f"{base}/examples"

    mod = importlib.import_module(info["module"])

    parts = info["fullname"].split(".")
    obj = getattr(mod, parts[0])
    filename = obj.__module__.replace(".", "/")
    if dataclasses.is_dataclass(obj):
        if len(parts) > 1:
            return None

    for part in parts[1:]:
        obj = getattr(obj, part)

    try:
        src, ln = inspect.getsourcelines(obj)
        return f"{base}/{filename}.py?#L{ln}-L{ln + len(src) - 1}"
    except Exception:
        pass

    # Fallback for property
    try:
        src, ln = inspect.getsourcelines(obj.fget)
        return f"{base}/{filename}.py?#L{ln}-L{ln + len(src) - 1}"
    except Exception:
        return None


# -- Options for HTML output -------------------------------------------------
# Custom directives

from custom_directives import CustomAutoSummary


def setup(app):
    import furo
    import sphinx_basic_ng

    sphinx_basic_ng.setup(app)
    furo.setup(app)
    app.add_directive("autosummary", CustomAutoSummary, override=True)
