"""
limTOD: Time-Ordered Data Simulator
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    # Single-sourced from pyproject.toml via the installed distribution
    # metadata (the hardcoded copy went stale once — see CHANGELOG 1.3.0).
    __version__ = _dist_version("limTOD")
except PackageNotFoundError:  # pragma: no cover — running from a bare checkout
    __version__ = "0.0.0.dev0"

__author__ = "Zheng Zhang"
__email__ = "zheng.zhang@manchester.ac.uk"
__license__ = "MIT"

from .simulator import (
    TODSim,
    example_scan,
    generate_LSTs_deg,
    zyzyz2zyz,
    zyz_of_pointing,
    generate_TOD_sky,
    pointing_beam_in_eq_sys,
    example_beam_map,
    example_symm_beam_map,
)

from .sky_model import (
    GDSM_sky_model,
    generate_gaussian_field,
)

from .HPW_filter import (
    get_filtfilt_matrix,
    HP_filter_TOD,
    wiener_filter_map,
    simple_wiener_map,
    HPW_mapmaking
)

__all__ = [
    "TODSim",
    "example_scan",
    "generate_LSTs_deg",
    "zyzyz2zyz",
    "zyz_of_pointing",
    "generate_TOD_sky",
    "pointing_beam_in_eq_sys",
    "GDSM_sky_model",
    "example_beam_map",
    "example_symm_beam_map",
    "generate_gaussian_field",
    "get_filtfilt_matrix",
    "HP_filter_TOD",
    "wiener_filter_map",
    "simple_wiener_map",
    "HPW_mapmaking",
]
