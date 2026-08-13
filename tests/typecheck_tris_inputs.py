"""Positive static-typing fixtures for runtime-supported TRIS inputs."""

import healpy as hp
import numpy as np

from limTOD.tris import (
    AsymmetricUncertainty,
    TRISLinearFit,
    TRISMapMakingInputs,
    TRISNoiseModel,
    TRISPointSet,
    TRISPrincipalPlaneCuts,
    TRISRankDiagnostic,
    TRISRing,
    approximate_tris_gaussian_beam_map,
    build_tris_mapmaking_inputs,
    cmb_monopole_rj_k,
    fit_tris_linear_model,
    galactic_spectral_index,
    to_tris_temperature_convention,
    tris_beam_func,
    tris_cut_beam_func,
    tris_cut_beam_map,
    tris_cut_beam_response,
    tris_horizon_mask,
    tris_prior_from_template,
    tris_ring_pixels,
    tris_zenith_geometry,
)


def _requires_float(value: float) -> None:
    pass


def _requires_int(value: int) -> None:
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
    _requires_float(linear_fit.chi_square)
    _requires_float(linear_fit.reduced_chi_square)
    _requires_int(linear_fit.degrees_of_freedom)


def accepted_beam_sky_and_mapmaking_inputs() -> None:
    """Type-check the cut beam, sky-convention and map-making entry points."""
    cuts = TRISPrincipalPlaneCuts(
        angle_deg=np.array([0.0, 10.0, 90.0, 176.0]),
        h_plane_db=np.array([0.0, -2.0, -40.0, -48.6]),
        e_plane_db=np.array([0.0, -3.3, -42.0, -47.8]),
    )
    e_fwhm, h_fwhm = cuts.half_power_full_width_deg()
    _requires_float(e_fwhm)
    _requires_float(h_fwhm)

    _requires_array(
        tris_cut_beam_response(
            cuts, np.array([0.0, 5.0]), np.array([0.0, 90.0]), blend="power"
        )
    )
    _requires_array(tris_cut_beam_map(cuts, nside=4, blend="db", normalization="sum"))
    _requires_array(tris_cut_beam_func(cuts)(freq=np.float64(600.0), nside=4))
    _requires_array(tris_horizon_mask(4, min_elevation_deg=np.array(5.0)))

    _requires_float(cmb_monopole_rj_k(np.float64(600.5)))
    _requires_array(to_tris_temperature_convention(np.array([1.0, 2.0]), 600.5))
    _requires_array(
        galactic_spectral_index(
            np.array([9.4, 28.2]), 600.5, np.array([5.4, 13.6]), 817.8
        )
    )

    noise = TRISNoiseModel(
        np.array([0.0, 0.01, 0.02]),
        common_mode_sigma_k=np.float64(0.066),
        uncertainty_floor_k=np.array(0.004),
    )
    _requires_array(noise.whiten(np.eye(3)))
    _requires_array(noise.inverse_apply(np.array([1.0, 2.0, 3.0])))
    _requires_array(noise.variance_k2)
    _requires_int(noise.size)

    ra_deg = np.array([0.0, 90.0, 180.0, 270.0])
    ring = TRISRing(
        nominal_frequency_mhz=600.0,
        effective_frequency_mhz=600.5,
        bandwidth_mhz=0.3,
        ra_text=("0h00m", "6h00m", "12h00m", "18h00m"),
        ra_deg=ra_deg,
        temperature_k=np.array([15.0, 16.0, 14.0, 17.0]),
        statistical_uncertainty_k=np.array([0.01, 0.01, 0.01, 0.01]),
        zero_level_uncertainty_k=0.066,
    )
    inputs: TRISMapMakingInputs = build_tris_mapmaking_inputs(
        ring,
        nside=4,
        cuts=cuts,
        dec_half_width_deg=np.float64(50.0),
        uncertainty_floor_k=np.array(0.004),
        zero_level_sigma_k=0.066,
    )
    _requires_array(inputs.operator)
    _requires_array(inputs.beam_coverage)
    _requires_array(inputs.data_k)
    _requires_int(inputs.sky_parameter_count)
    _requires_int(inputs.parameter_count)
    _requires_float(inputs.monopole_degeneracy)
    _requires_float(inputs.implied_monopole_prior_sigma_k(1.0))

    pixels = tris_ring_pixels(4, dec_deg=np.float64(42.43), half_width_deg=50.0)
    _requires_array(pixels)
    guess, sigma = tris_prior_from_template(
        np.ones(hp.nside2npix(4)) * 15.0, inputs.pixel_indices, relative_sigma=0.1
    )
    _requires_array(guess)
    _requires_array(sigma)

    simulated = inputs.predict(guess, 0.01)
    _requires_array(simulated)
    solution = inputs.with_data(simulated).solve(prior_map=guess, prior_sigma_k=sigma)
    _requires_array(solution.sky_k)
    _requires_array(solution.sky_uncertainty_k)
    _requires_array(solution.healpix_map())
    _requires_array(solution.healpix_uncertainty())


if __name__ == "__main__":
    accepted_tris_inputs()
    accepted_beam_sky_and_mapmaking_inputs()
