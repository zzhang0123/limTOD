"""
limTOD: Time-Ordered Data Simulator for MeerKLASS
"""

__version__ = "1.0.0"
__author__ = "Zheng Zhang"
__email__ = "zheng.zhang@manchester.ac.uk"
__license__ = "MIT"

from .simulator import (
    TODsim,
    example_scan,
    generate_LSTs_deg,
    zyzy2zyz,
    zyz_of_pointing,
    generate_TOD_sky,
    pointing_beam_in_eq_sys,
    GDSM_sky_model,
    example_beam_map,
)

from .sky_model import (
    GDSM_sky_model,
    generate_gaussian_field,
)

__all__ = [
    "TODsim",
    "example_scan",
    "generate_LSTs_deg",
    "zyzy2zyz",
    "zyz_of_pointing",
    "generate_TOD_sky",
    "pointing_beam_in_eq_sys",
    "GDSM_sky_model",
    "example_beam_map",
    "generate_gaussian_field",
]
