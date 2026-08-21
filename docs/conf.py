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
    release = "1.8.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # Grid cards on the drift-scan page. Self-contained CSS, so it renders on
    # sphinx_rtd_theme as well as on the themes it is usually seen with.
    "sphinx_design",
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
# One stylesheet, `_static/custom.css`: see the file for what it fixes (the
# theme's 800px content cap against two-column sphinx-design cards).
html_static_path = ["_static"]
html_css_files = ["custom.css"]
exclude_patterns = ["_build", "README.md", "superpowers/**"]

# TRIS is unreleased (see the Unreleased section of the changelog), so its two
# pages are NOT published. Build them locally with:
#
#     python -m sphinx -b html docs docs/_build -t tris
#
# The switch is opt-IN rather than opt-out on purpose. Keying it off an
# environment variable that Read the Docs happens to set would publish the
# pages the moment that variable changed name or failed to reach the build;
# this way the default -- which is what Read the Docs runs -- excludes them,
# and only a deliberate `-t tris` brings them back.
#
# `toc.excluded` is the warning for the two toctree entries that then point at
# excluded documents. It is suppressed only on the non-TRIS build, so a
# genuinely broken toctree entry elsewhere still fails to go unnoticed.
suppress_warnings: list[str] = []
if not tags.has("tris"):  # noqa: F821 -- `tags` is injected by Sphinx
    exclude_patterns += ["tris.md", "api/tris.md"]
    suppress_warnings.append("toc.excluded")

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
