#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from unittest.mock import MagicMock, Mock

sys.path.insert(0, os.path.abspath('..'))


class MMock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        m = MagicMock()
        m.__repr__ = Mock(return_value=f"<{name}>")
        return m


MOCK_MODULES = ['scipy', 'scipy.ndimage',
                'scipy.ndimage.filters', 'scipy.sparse', 'scipy.spatial',
                'scipy.interpolate', 'scipy.stats', 'scipy.sparse.linalg',
                'shapely.geometry']
modules = []
for mod_name in MOCK_MODULES:
    mock = MMock()
    modules.append((mod_name, mock))

sys.modules.update(m for m in modules)


# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.extlinks', 'sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'clasp.sphinx_click_ext', 'sphinx.ext.napoleon', 'sphinx.ext.todo']
autodoc_member_order = 'bysource'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'raytools'
copyright = u"2022, Stephen Wasilewski"
author = u"Stephen Wasilewski"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = "0.1.4"
# The full version, including alpha/beta/rc tags.
release = "0.1.4"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'cdocs']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'page_width': '1050px',
    'fixed_sidebar': True
}

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html'
    ]}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'raytoolsedoc'


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    'papersize': 'a4paper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    'preamble': "\n\\usepackage{listings}\n\\lstset{ language=Python, title=\\lstname }",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'raytools.tex',
     u'raytools Documentation',
     u'Stephen Wasilewski', 'manual'),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'raytools',
     u'raytools Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'raytools',
     u'raytools Documentation',
     author,
     'raytools',
     'One line description of project.',
     'Miscellaneous'),
]



