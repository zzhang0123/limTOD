"""Sphinx configuration for the limTOD documentation (ReadTheDocs)."""

import os
import sys

# The package is pip-installed on ReadTheDocs (see .readthedocs.yaml), but
# a source checkout on the path makes local `sphinx-build docs docs/_build`
# work without an install.
sys.path.insert(0, os.path.abspath(".."))

project = "limTOD"
author = "Zheng Zhang"
copyright = "2026, Zheng Zhang"

try:
    from importlib.metadata import version as _dist_version

    release = _dist_version("limTOD")
except Exception:  # pragma: no cover - bare checkout
    release = "1.5.3"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

# No autodoc mocks: every optional dependency (mpi4py, pygdsm, pyuvdata,
# joblib) is imported lazily or with a serial fallback, so the modules
# import cleanly from the base install. Mocking an INSTALLED optional
# actually breaks imports (mpiutil's `size > 1` guard vs a Mock).
# jax/s2fft are real on RTD (the build installs [jax]) so limtod_jax
# renders true signatures.
autodoc_mock_imports: list = []
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Both styles: the numpy package uses numpydoc sections, limtod_jax uses
# Google-style Args blocks.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

templates_path = []
exclude_patterns = ["_build", "README.md"]

html_theme = "sphinx_rtd_theme"
html_title = f"limTOD {release}"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_context = {
    "display_github": True,
    "github_user": "zzhang0123",
    "github_repo": "limTOD",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
