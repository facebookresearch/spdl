# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.append(os.path.abspath("."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SPDL"
copyright = "2024, Moto Hira"
author = "Moto Hira"
release = "0.0.6"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    # "breathe",
    # "exhale",
]
autosummary_generate = True
autosummary_imported_members = True
autoclass_content = "class"
autodoc_default_options = {
    "special-members": ",".join(["__len__", "__getitem__", "__iter__"]),
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
}
exclude_patterns = []


breathe_projects = {"libspdl": "generated/doxygen/xml/"}
breathe_default_project = "libspdl"
mermaid_version = "10.9.1"


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

# -- Options for HTML output -------------------------------------------------
# Custom directives

from custom_directives import CustomAutoSummary


def setup(app):
    import furo
    import sphinx_basic_ng

    sphinx_basic_ng.setup(app)
    furo.setup(app)
    app.add_directive("autosummary", CustomAutoSummary, override=True)
