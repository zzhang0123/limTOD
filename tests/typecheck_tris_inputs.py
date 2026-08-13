"""Positive static-typing fixtures for runtime-supported TRIS inputs."""

import numpy as np

from limTOD.tris import (
    AsymmetricUncertainty,
    TRISPointSet,
    TRISRankDiagnostic,
    TRISRing,
    approximate_tris_gaussian_beam_map,
    fit_tris_linear_model,
    tris_beam_func,
    tris_zenith_geometry,
)


def accepted_tris_inputs() -> None:
    """Type-check accepted zero-dimensional scalars and nested-list matrices."""
    scalar = np.array(0.3)
    uncertainty = AsymmetricUncertainty(positive_k=np.float64(0.43), negative_k=scalar)
    ring = TRISRing(
        nominal_frequency_mhz=np.float64(600.0),
        effective_frequency_mhz=np.array(600.5),
        bandwidth_mhz=scalar,
        ra_text=("0h00m", "1h00m", "2h00m"),
        ra_deg=np.array([0.0, 15.0, 30.0]),
        temperature_k=np.array([1.0, 2.0, 3.0]),
        statistical_uncertainty_k=np.array([0.1, 0.2, 0.3]),
        zero_level_uncertainty_k=uncertainty,
        declination_label_deg=np.array(42.0),
    )
    TRISPointSet(
        nominal_frequency_mhz=np.array(2500.0),
        effective_frequency_mhz=np.float64(2427.8),
        bandwidth_mhz=np.array(3.0),
        ra_text=("0h00m",),
        ra_deg=np.array([0.0]),
        temperature_k=np.array([2.3]),
        statistical_uncertainty_k=None,
        zero_level_uncertainty_k=np.array(0.284),
        declination_label_deg=np.float64(42.0),
    )
    TRISRankDiagnostic(
        singular_values=np.array([2.0]),
        numerical_rank=1,
        parameter_count=1,
        tolerance=np.array(1e-12),
        rank_rtol=np.float64(1e-10),
        condition_number=np.array(1.0),
    )

    fit_tris_linear_model(
        ring,
        [[1.0], [0.5], [-0.25]],
        uncertainty_floor_k=np.array(0.01),
        common_mode_sigma_k=np.float64(0.02),
        rank_rtol=np.array(1e-10),
    )
    approximate_tris_gaussian_beam_map(
        nside=1, fwhm_e_deg=np.array(18.0), fwhm_h_deg=np.float64(23.0)
    )
    beam_func = tris_beam_func(fwhm_e_deg=np.array(18.0), fwhm_h_deg=np.float64(23.0))
    beam_func(freq=np.array(600.0), nside=1)
    tris_zenith_geometry(
        [0.0],
        latitude_deg=np.array(42.0),
        e_plane_east_of_meridian_deg=np.float64(7.0),
    )
