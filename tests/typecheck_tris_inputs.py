"""Positive static-typing fixtures for runtime-supported TRIS inputs."""

import numpy as np

from limTOD.tris import (
    AsymmetricUncertainty,
    TRISLinearFit,
    TRISPointSet,
    TRISPrincipalPlaneCuts,
    TRISRankDiagnostic,
    TRISRing,
    approximate_tris_gaussian_beam_map,
    fit_tris_linear_model,
    tris_beam_func,
    tris_zenith_geometry,
)


def _requires_float(value: float) -> None:
    pass


def _requires_array(value: np.ndarray) -> None:
    pass


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
    points = TRISPointSet(
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
    cuts = TRISPrincipalPlaneCuts(
        angle_deg=[0.0, 3.0],
        h_plane_db=[0.0, -0.2],
        e_plane_db=[0.0, -0.3],
    )
    diagnostic = TRISRankDiagnostic(
        singular_values=np.array([2.0]),
        numerical_rank=1,
        parameter_count=1,
        tolerance=np.array(1e-12),
        rank_rtol=np.float64(1e-10),
        condition_number=np.array(1.0),
    )

    fit = fit_tris_linear_model(
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
    geometry = tris_zenith_geometry(
        [0.0],
        latitude_deg=np.array(42.0),
        e_plane_east_of_meridian_deg=np.float64(7.0),
    )

    for scalar_value in (
        uncertainty.positive_k,
        uncertainty.negative_k,
        ring.nominal_frequency_mhz,
        ring.effective_frequency_mhz,
        ring.bandwidth_mhz,
        ring.declination_label_deg,
        points.nominal_frequency_mhz,
        points.effective_frequency_mhz,
        points.bandwidth_mhz,
        points.zero_level_uncertainty_k,
        points.declination_label_deg,
        geometry.latitude_deg,
        diagnostic.tolerance,
        diagnostic.rank_rtol,
        diagnostic.condition_number,
    ):
        _requires_float(scalar_value)

    for array_value in (
        ring.ra_deg,
        ring.temperature_k,
        ring.statistical_uncertainty_k,
        points.ra_deg,
        points.temperature_k,
        cuts.angle_deg,
        cuts.h_plane_db,
        cuts.e_plane_db,
        geometry.lst_deg,
        diagnostic.singular_values,
        fit.coefficients,
        fit.coefficient_covariance,
    ):
        _requires_array(array_value)

    linear_fit: TRISLinearFit = fit
    _requires_array(linear_fit.prediction_k)


if __name__ == "__main__":
    accepted_tris_inputs()
