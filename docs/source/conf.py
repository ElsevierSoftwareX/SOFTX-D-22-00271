# conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../../rosenpy'))  # Adiciona a pasta rosenpy ao caminho
sys.path.insert(0, os.path.abspath('../../examples'))  # Adiciona a pasta examples ao caminho

# -- Project information -----------------------------------------------------
project = 'rosenpy'
copyright = '2024'
author = 'Ariadne A. Cruz, Kayol S. Mayer, Dalton S. Arantes'
release = '2.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.githubpages'
]

autodoc_default_options = {
    'members': True,
    'inherited-members': True
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects htmlstatic_path and html_extra_path.
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    #'canonical_url': '',
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#343131',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_logo = 'img/logo_rosenpy.png'

html_favicon = 'img/ico_rosenpy.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
htmlstatic_path = ['static']