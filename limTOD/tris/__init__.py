"""Offline bridge between the public TRIS archive and limTOD.

The four public LAMBDA products are final, beam-convolved profiles and points
-- not raw time-ordered data.  This package parses them without touching the
network, converts the TRIS conventions into limTOD's, supplies a beam built
from the archive's own principal-plane cuts, and assembles the operator, noise
and prior objects that limTOD's Wiener/MAP map-makers consume.

``docs/tris.md`` is the canonical convention report and records which
statements are archive-confirmed, which were derived from the data here, and
which are explicit approximations.

Layout
------
``archive``
    Strict readers and typed models for the four text products.
``sky``
    What a TRIS temperature contains (Rayleigh-Jeans, CMB monopole included).
``beam``
    The cut-interpolated beam, the legacy Gaussian, and the horizon mask.
``geometry``
    Parked-zenith pointing, including the 7-degree E-plane roll.
``noise``
    Diagonal statistical noise plus the common zero level, applied exactly.
``inference``
    Rank-gated low-dimensional Fourier/template fits.
``mapmaking``
    Sky-to-sample operator and prior assembly for regularized reconstruction.
"""

from .archive import (
    AsymmetricUncertainty,
    TRISPointSet,
    TRISPrincipalPlaneCuts,
    TRISRing,
    parse_tris_ra,
    read_tris_beam_cuts,
    read_tris_point_set,
    read_tris_ring,
)
from .beam import (
    approximate_tris_gaussian_beam_map,
    tris_beam_func,
    tris_cut_beam_func,
    tris_cut_beam_map,
    tris_cut_beam_response,
    tris_horizon_mask,
)
from .geometry import (
    TRIS_DECLINATION_LABEL_DEG,
    TRIS_E_PLANE_EAST_OF_MERIDIAN_DEG,
    TRIS_SITE_LATITUDE_DEG,
    TRISZenithGeometry,
    tris_zenith_geometry,
)
from .inference import (
    TRISLinearFit,
    TRISRankDiagnostic,
    build_tris_fourier_design,
    fit_tris_linear_model,
)
from .mapmaking import (
    TRISMapMakingInputs,
    build_tris_mapmaking_inputs,
    tris_prior_from_template,
    tris_ring_pixels,
)
from .noise import TRISNoiseModel
from .sky import (
    CMB_T0_K,
    cmb_monopole_rj_k,
    galactic_spectral_index,
    to_tris_temperature_convention,
)

__all__ = [
    # archive
    "AsymmetricUncertainty",
    "TRISRing",
    "TRISPointSet",
    "TRISPrincipalPlaneCuts",
    "parse_tris_ra",
    "read_tris_ring",
    "read_tris_point_set",
    "read_tris_beam_cuts",
    # sky convention
    "CMB_T0_K",
    "cmb_monopole_rj_k",
    "to_tris_temperature_convention",
    "galactic_spectral_index",
    # beam
    "tris_cut_beam_map",
    "tris_cut_beam_func",
    "tris_cut_beam_response",
    "tris_horizon_mask",
    "approximate_tris_gaussian_beam_map",
    "tris_beam_func",
    # geometry
    "TRISZenithGeometry",
    "tris_zenith_geometry",
    "TRIS_SITE_LATITUDE_DEG",
    "TRIS_DECLINATION_LABEL_DEG",
    "TRIS_E_PLANE_EAST_OF_MERIDIAN_DEG",
    # noise
    "TRISNoiseModel",
    # inference
    "TRISRankDiagnostic",
    "TRISLinearFit",
    "build_tris_fourier_design",
    "fit_tris_linear_model",
    # map-making
    "TRISMapMakingInputs",
    "build_tris_mapmaking_inputs",
    "tris_ring_pixels",
    "tris_prior_from_template",
]
