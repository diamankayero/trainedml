# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'trainedml'
copyright = '2026, Yéro Diamanka'
author = 'Yéro Diamanka'

# La version affichée est lue depuis pyproject.toml : une seule source de vérité.
import tomllib
from pathlib import Path

with open(Path(__file__).resolve().parents[2] / 'pyproject.toml', 'rb') as f:
    release = tomllib.load(f)['project']['version']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
]
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

templates_path = ['_templates']
exclude_patterns = []

language = 'fr'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_title = f'trainedml {release}'
html_static_path = ['_static']
html_logo = '_static/logo.svg'
html_favicon = '_static/logo.svg'
html_css_files = ['custom.css']

html_theme_options = {
    # Accent assorti au vert du logo (préréglage Shibuya le plus proche)
    'accent_color': 'grass',
    'github_url': 'https://github.com/diamankayero/trainedml',
}

# -- Extension configuration ------------------------------------------------
extensions += [
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
]

# Les titres de section des docstrings se répètent d'un module à l'autre
# (Examples, Mathematical context...) : on préfixe les labels par le document
# pour éviter les collisions.
autosectionlabel_prefix_document = True

