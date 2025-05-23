# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MLOps Demo Project - Parakeet STT API'
copyright = '2025, Dean van Aswegen'
author = 'Dean van Aswegen'
release = 'beta'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',  # Add links to the source code
    'sphinx.ext.githubpages',  # Support for GitHub Pages
    'sphinx_rtd_theme',  # Read the Docs theme
    'sphinx.ext.autosummary',  # Auto-generate module summaries
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autosummary settings - automatically generate module docs
autosummary_generate = True
autosummary_imported_members = True

# Auto-generate API docs
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Mock imports for autodoc ------------------------------------------------
# Mock heavy dependencies that might not be available during doc build
autodoc_mock_imports = [
    'torch',
    'torchaudio', 
    'nemo',
    'nemo.collections',
    'nemo.collections.asr',
    'fastapi',
    'uvicorn',
]
