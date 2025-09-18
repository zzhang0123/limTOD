"""
MeerTOD: Time-Ordered Data Simulator for MeerKLASS
"""

__version__ = "1.0.0"
__author__ = "Zheng Zhang"
__email__ = "zheng.zhang@manchester.ac.uk"
__license__ = "MIT"

from .tod_simulator import (
    meerTODsim,
    example_scan,
    generate_LSTs_deg,
    zyzy2zyz,
    zyz_of_pointing,
    generate_TOD_sky,
    pointing_beam_in_eq_sys,
    GDSM_sky_model,
    example_beam_map,
)

__all__ = [
    'meerTODsim',
    'example_scan',
    'generate_LSTs_deg',
    'zyzy2zyz',
    'zyz_of_pointing',
    'generate_TOD_sky',
    'pointing_beam_in_eq_sys',
    'GDSM_sky_model',
    'example_beam_map',
]
