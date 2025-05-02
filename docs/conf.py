import sys
from pathlib import Path
import sphinx_pdj_theme

sys.path.insert(0, str(Path('..', 'src').resolve()))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Leaf Toolkit'
copyright = '2025, Radek Zenkl'
author = 'Radek Zenkl'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Add Napoleon for Google style docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
    'autoapi.extension',
]
autoapi_dirs = ['../src']
autoapi_python_class_content = 'both'
autoapi_options = [
    "members",
    "undoc-members",
    "special-members",
    "show-inheritance",
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_static_path = ['_static']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

extensions.append("sphinx_wagtail_theme")
html_theme = 'sphinx_wagtail_theme'
html_static_path = ['_static']

# This is used by Sphinx in many places, such as page title tags.
project = "Leaf Toolkit"

# These are options specifically for the Wagtail Theme.

html_theme_options = dict(
    logo = "img/logo_placeholder.png",
    logo_alt = "",
    logo_height = 5,
    logo_url = "home.html",
    logo_width = 5,
    project_name = "Leaf Toolkit",
    github_url = "https://github.com/RadekZenkl/leaf-toolkit/blob/main/docs/",
    # header_links = "Top 1|http://example.com/one, Top 2|http://example.com/two",
    footer_links = ",".join([
        # "About Us|http://example.com/",
        # "Contact|http://example.com/contact",
        # "Legal|http://example.com/dev/null",
    ]),
 )

# html_sidebars = {
#     '**': [
#         'globaltoc.html',
#         'relations.html',  # Next/prev links
#         'sourcelink.html',
#         'searchbox.html',
#     ]
# }