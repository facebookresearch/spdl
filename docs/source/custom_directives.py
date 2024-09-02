# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# isort: skip_file

import logging
import posixpath
import re

import docutils.nodes

from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.ext.autodoc.directive import DocumenterBridge, Options

from sphinx.ext.autosummary import Autosummary
from sphinx.locale import __
from sphinx.util.matching import Matcher

logger = logging.getLogger(__name__)


class CustomAutoSummary(Autosummary):
    """Custome autosummary

    1. Do not add entries to the left bar if `hide_from_toctree` is given

    """

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = True
    option_spec = {
        "caption": directives.unchanged_required,
        "toctree": directives.unchanged,
        "nosignatures": directives.flag,
        "recursive": directives.flag,
        "template": directives.unchanged,
        "hide_from_toctree": directives.flag,
    }

    def run(self):
        self.bridge = DocumenterBridge(
            self.env, self.state.document.reporter, Options(), self.lineno, self.state
        )

        names = [
            x.strip().split()[0]
            for x in self.content
            if x.strip() and re.search(r"^[~a-zA-Z_]", x.strip()[0])
        ]
        items = self.get_items(names)
        nodes = self.get_table(items)

        if "toctree" in self.options:
            dirname = posixpath.dirname(self.env.docname)

            tree_prefix = self.options["toctree"].strip()
            docnames = []
            excluded = Matcher(self.config.exclude_patterns)
            filename_map = self.config.autosummary_filename_map
            for _name, _sig, _summary, real_name in items:
                real_name = filename_map.get(real_name, real_name)
                docname = posixpath.join(tree_prefix, real_name)
                docname = posixpath.normpath(posixpath.join(dirname, docname))
                if docname not in self.env.found_docs:
                    if excluded(self.env.doc2path(docname, False)):
                        msg = __(
                            "autosummary references excluded document %r. Ignored."
                        )
                    else:
                        msg = __(
                            "autosummary: stub file not found %r. "
                            "Check your autosummary_generate setting."
                        )

                    logger.warning(msg, real_name)
                    continue

                docnames.append(docname)

            if docnames and "hide_from_toctree" not in self.options:
                tocnode = addnodes.toctree()
                tocnode["includefiles"] = docnames
                tocnode["entries"] = [(None, docn) for docn in docnames]
                tocnode["maxdepth"] = 1
                tocnode["glob"] = None
                tocnode["caption"] = self.options.get("caption")

                _nodes = docutils.nodes.comment("", "", tocnode)

                nodes.append(_nodes)

        if "toctree" not in self.options and "caption" in self.options:
            logger.warning(
                __("A captioned autosummary requires :toctree: option. ignored."),
                location=nodes[-1],
            )

        return nodes
