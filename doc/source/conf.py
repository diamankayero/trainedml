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
    'sphinx_gallery.gen_gallery',
]

# Les titres de section des docstrings se répètent d'un module à l'autre
# (Examples, Mathematical context...) : on préfixe les labels par le document
# pour éviter les collisions.
autosectionlabel_prefix_document = True

# Galerie d'exemples exécutables (façon scikit-learn/matplotlib) : les
# scripts source vivent dans des sous-dossiers de examples/ à la racine du
# dépôt ; la galerie générée (HTML, miniatures, notebooks) vit dans
# doc/source/auto_examples/.
sphinx_gallery_conf = {
    'examples_dirs': [
        '../../examples/01_bases',
        '../../examples/02_donnees_et_modeles',
        '../../examples/03_production',
    ],
    'gallery_dirs': [
        'auto_examples/01_bases',
        'auto_examples/02_donnees_et_modeles',
        'auto_examples/03_production',
    ],
    # r'/plot_' (le defaut sphinx-gallery) ne matche jamais sur des chemins
    # Windows a antislashs : motif independant du separateur de chemin.
    'filename_pattern': r'plot_',
    'download_all_examples': False,
    'remove_config_comments': True,
    'default_thumb_file': os.path.join(os.path.dirname(__file__), '_static', 'logo_thumb.png'),
    'backreferences_dir': None,
}

