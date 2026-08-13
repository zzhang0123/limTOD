"""Beam, sky-convention, noise and map-making contracts for ``limTOD.tris``.

These tests pin the *physics* the TRIS bridge claims, not just its signatures:
the beam must reproduce the archive's own cuts, the temperature convention must
be the one the two rings jointly imply, the noise model must stay exact where a
dense factorization silently fails, and the sky-to-sample operator must agree
with limTOD's own forward model to machine precision.
"""

import numpy as np
import healpy as hp
import pytest

from limTOD.simulator import generate_TOD_sky
from limTOD.tris import (
    TRISNoiseModel,
    TRISPrincipalPlaneCuts,
    TRISRing,
    approximate_tris_gaussian_beam_map,
    build_tris_fourier_design,
    build_tris_mapmaking_inputs,
    cmb_monopole_rj_k,
    fit_tris_linear_model,
    galactic_spectral_index,
    to_tris_temperature_convention,
    tris_cut_beam_func,
    tris_cut_beam_map,
    tris_cut_beam_response,
    tris_horizon_mask,
    tris_prior_from_template,
    tris_ring_pixels,
    tris_zenith_geometry,
)

HALF_POWER_DB = -10.0 * np.log10(2.0)


def _gaussian_cuts(e_fwhm=19.155, h_fwhm=23.366, stop=90.0, step=0.5):
    """Principal-plane cuts of an exactly elliptical-Gaussian beam."""
    angle = np.arange(0.0, stop + step, step)
    return TRISPrincipalPlaneCuts(
        angle_deg=angle,
        h_plane_db=HALF_POWER_DB * (2.0 * angle / h_fwhm) ** 2,
        e_plane_db=HALF_POWER_DB * (2.0 * angle / e_fwhm) ** 2,
    )


def _tabulated_cuts():
    """A compact stand-in for the archive: narrow E, wide H, real sidelobes."""
    angle = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 90.0, 176.0])
    return TRISPrincipalPlaneCuts(
        angle_deg=angle,
        h_plane_db=np.array([0.0, -0.5, -2.0, -7.5, -22.0, -40.0, -48.6]),
        e_plane_db=np.array([0.0, -0.9, -3.3, -9.0, -25.0, -42.0, -47.8]),
    )


def _ring(ra_deg, temperature_k, uncertainty_k):
    ra_deg = np.asarray(ra_deg, dtype=float)
    return TRISRing(
        nominal_frequency_mhz=600.0,
        effective_frequency_mhz=600.5,
        bandwidth_mhz=0.3,
        ra_text=tuple("{}h00m".format(index) for index in range(ra_deg.size)),
        ra_deg=ra_deg,
        temperature_k=np.asarray(temperature_k, dtype=float),
        statistical_uncertainty_k=np.asarray(uncertainty_k, dtype=float),
        zero_level_uncertainty_k=0.066,
    )


# ---------------------------------------------------------------------------
# Beam built from the archive's own cuts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blend", ["db", "power"])
def test_cut_beam_reproduces_the_archive_cuts_on_the_principal_planes(blend):
    """Any blend that is not exact on the measured planes is discarding data."""
    cuts = _tabulated_cuts()

    on_e = tris_cut_beam_response(
        cuts, cuts.angle_deg, np.zeros_like(cuts.angle_deg), blend=blend
    )
    on_h = tris_cut_beam_response(
        cuts, cuts.angle_deg, np.full_like(cuts.angle_deg, 90.0), blend=blend
    )

    np.testing.assert_allclose(on_e, 10.0 ** (cuts.e_plane_db / 10.0), rtol=1e-12)
    np.testing.assert_allclose(on_h, 10.0 ** (cuts.h_plane_db / 10.0), rtol=1e-12)


def test_db_blend_of_gaussian_cuts_reproduces_the_elliptical_gaussian():
    """The dB blend is the exact generalization of the legacy Gaussian beam."""
    nside = 32
    e_fwhm, h_fwhm = 19.155, 23.366

    blended = tris_cut_beam_map(
        _gaussian_cuts(e_fwhm, h_fwhm, stop=180.0, step=0.25),
        nside=nside,
        blend="db",
        normalization="peak",
    )
    elliptical = approximate_tris_gaussian_beam_map(
        nside=nside, fwhm_e_deg=e_fwhm, fwhm_h_deg=h_fwhm, normalization="peak"
    )

    inside = np.rad2deg(hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))[0]) <= 60.0
    np.testing.assert_allclose(blended[inside], elliptical[inside], atol=2e-4)


def test_cut_beam_keeps_the_narrow_axis_on_the_e_plane():
    """Swapping E and H would rotate the beam 90 deg against the archive."""
    cuts = _gaussian_cuts()
    beam = tris_cut_beam_map(cuts, nside=256, normalization="peak")

    assert hp.get_interp_val(beam, np.deg2rad(19.155 / 2.0), 0.0) == pytest.approx(
        0.5, abs=0.01
    )
    assert hp.get_interp_val(
        beam, np.deg2rad(23.366 / 2.0), np.pi / 2.0
    ) == pytest.approx(0.5, abs=0.01)


def test_measured_half_power_widths_come_from_the_cuts_not_from_prose():
    """The archive's E cut is 19.155 deg wide; the ring headers round it to 18."""
    e_fwhm, h_fwhm = _gaussian_cuts().half_power_full_width_deg()

    assert e_fwhm == pytest.approx(19.155, abs=0.02)
    assert h_fwhm == pytest.approx(23.366, abs=0.02)


def test_half_power_width_rejects_a_cut_that_never_reaches_half_power():
    """Silently returning the last tabulated angle would invent a beam width."""
    shallow = TRISPrincipalPlaneCuts(
        angle_deg=[0.0, 1.0, 2.0],
        h_plane_db=[0.0, -0.1, -0.2],
        e_plane_db=[0.0, -0.1, -0.2],
    )

    with pytest.raises(ValueError, match="half-power"):
        shallow.half_power_full_width_deg()


def test_cut_beam_retains_below_horizon_response_that_the_gaussian_discards():
    """The ground term the Gaussian sets to zero is comparable to the zero level."""
    nside = 32
    cut_beam = tris_cut_beam_map(_tabulated_cuts(), nside=nside, normalization="sum")
    gaussian = approximate_tris_gaussian_beam_map(nside=nside, normalization="sum")
    below_horizon = tris_horizon_mask(nside) == 0.0

    cut_fraction = float(cut_beam[below_horizon].sum())
    gaussian_fraction = float(gaussian[below_horizon].sum())

    assert cut_fraction > 1e-5
    assert gaussian_fraction < 1e-15
    # against a 300 K ground this is a real, sub-tenth-kelvin term
    assert 0.001 < cut_fraction * 300.0 < 1.0


def test_horizon_mask_is_a_zenith_polar_cap_split():
    """A mask defined about the wrong pole would tip the beam onto the horizon."""
    nside = 32
    mask = tris_horizon_mask(nside)
    theta, _phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    assert set(np.unique(mask)) == {0.0, 1.0}
    assert mask[theta < np.pi / 2.0].min() == 1.0
    assert mask[theta > np.pi / 2.0].max() == 0.0
    assert tris_horizon_mask(nside, min_elevation_deg=30.0).sum() < mask.sum()


@pytest.mark.parametrize("normalization", ["peak", "sum", "none"])
def test_cut_beam_func_forwards_its_arguments(normalization):
    """The callable must not quietly ignore the blend or the normalization."""
    cuts = _tabulated_cuts()
    beam_func = tris_cut_beam_func(cuts, blend="power", normalization=normalization)

    np.testing.assert_array_equal(
        beam_func(freq=820.0, nside=16),
        tris_cut_beam_map(cuts, nside=16, blend="power", normalization=normalization),
    )
    np.testing.assert_array_equal(
        beam_func(freq=600.0, nside=16), beam_func(freq=2500.0, nside=16)
    )
    with pytest.raises(ValueError, match="freq"):
        beam_func(freq=0.0, nside=16)


# ---------------------------------------------------------------------------
# Temperature convention
# ---------------------------------------------------------------------------


def test_cmb_monopole_is_rayleigh_jeans_not_thermodynamic():
    """The 0.014-0.020 K difference exceeds the ring's statistical error."""
    assert cmb_monopole_rj_k(600.5) == pytest.approx(2.7111, abs=1e-4)
    assert cmb_monopole_rj_k(817.8) == pytest.approx(2.7059, abs=1e-4)
    assert cmb_monopole_rj_k(600.5) < 2.72548
    # the RJ limit is approached from below as the frequency drops
    assert cmb_monopole_rj_k(1.0) == pytest.approx(2.72548, abs=1e-4)
    assert cmb_monopole_rj_k(1.0) < 2.72548


def test_spectral_index_is_synchrotron_only_once_the_cmb_is_removed():
    """This is how the temperature convention was established from the data."""
    nu_low, nu_high = 600.5, 817.8
    galactic_low = np.array([6.7, 25.5])
    beta_true = -2.9
    galactic_high = galactic_low * (nu_high / nu_low) ** beta_true
    tris_low = galactic_low + cmb_monopole_rj_k(nu_low)
    tris_high = galactic_high + cmb_monopole_rj_k(nu_high)

    recovered = galactic_spectral_index(tris_low, nu_low, tris_high, nu_high)
    np.testing.assert_allclose(recovered, beta_true, rtol=1e-12)

    # leaving the monopole in flattens the index to a non-synchrotron value
    naive = np.log(tris_low / tris_high) / np.log(nu_low / nu_high)
    assert np.all(naive > beta_true + 0.4)


def test_to_tris_convention_adds_the_monopole_at_the_effective_frequency():
    """A Galactic-only template is not comparable with TRIS until this is done."""
    galactic = np.array([1.0, 2.0, 3.0])

    converted = to_tris_temperature_convention(galactic, 600.5)

    np.testing.assert_allclose(converted - galactic, cmb_monopole_rj_k(600.5))


# ---------------------------------------------------------------------------
# Noise model: exact where a dense factorization is not
# ---------------------------------------------------------------------------


_SIGMA = np.array([0.004, 0.012, 0.02, 0.007, 0.031, 0.004])


@pytest.mark.parametrize("common_mode", [0.0, 1e-4, 1e-3, 0.066, 1.0, 10.0])
def test_noise_whitening_matches_a_dense_solve_where_that_is_trustworthy(
    common_mode,
):
    """Pin the formula against the textbook route in its regime of validity.

    Below about ``sigma_c = 1e2`` the dense covariance is still well
    conditioned for these errors, so a Cholesky solve is a valid independent
    oracle; beyond it, it is not (see the companion test).  The comparison is
    on the Gram matrix rather than element-wise: this model uses the
    *symmetric* factor of ``C``, LAPACK returns the lower-triangular one, and
    the two differ by an orthogonal transform that no downstream quantity can
    see.
    """
    rng = np.random.default_rng(3)
    noise = TRISNoiseModel(_SIGMA, common_mode_sigma_k=common_mode)
    vectors = rng.standard_normal((_SIGMA.size, 4))

    dense = noise.dense_covariance()
    reference = vectors.T @ np.linalg.solve(dense, vectors)
    whitened = noise.whiten(vectors)

    np.testing.assert_allclose(whitened.T @ whitened, reference, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(
        noise.inverse_apply(vectors),
        np.linalg.solve(dense, vectors),
        rtol=1e-7,
        atol=1e-9,
    )


@pytest.mark.parametrize("common_mode", [1e3, 1e4, 1e5, 1e6, 1e9, 1e12])
def test_noise_whitening_stays_finite_and_converges_at_extreme_common_mode(
    common_mode,
):
    """The regime where the dense route silently returns garbage.

    ``C`` has condition number ``~ sigma_c^2 * sum(1/sigma^2)``, which is
    1e28 by ``sigma_c = 1e12``: no float64 factorization can invert it, and
    reconstructing ``C (C^-1 x) = x`` is meaningless there.  What *is* well
    posed, and what this asserts, is that whitening stays finite, stays
    self-consistent with ``C^-1``, and converges to the limit in which the
    common mode is simply projected out.
    """
    rng = np.random.default_rng(3)
    noise = TRISNoiseModel(_SIGMA, common_mode_sigma_k=common_mode)
    vectors = rng.standard_normal((_SIGMA.size, 4))

    whitened = noise.whiten(vectors)
    assert np.all(np.isfinite(whitened))
    np.testing.assert_allclose(
        whitened.T @ whitened,
        vectors.T @ noise.inverse_apply(vectors),
        rtol=1e-9,
        atol=1e-12,
    )

    # The limit is a projection that removes the common component entirely.
    # It is approached like 1/(sigma_c * sqrt(sum 1/sigma^2)), so the residual
    # offset must shrink in proportion as sigma_c grows.
    residual = float(np.abs(noise.whiten(np.ones(_SIGMA.size))).max())
    scale = common_mode * np.sqrt(np.sum(_SIGMA**-2.0))
    assert residual * scale == pytest.approx(
        float(np.abs(np.ones(_SIGMA.size) / _SIGMA).max()), rel=1e-3
    )


def test_infinite_common_mode_degenerates_into_projecting_the_offset_out():
    """ "Marginalize the zero level" must mean exactly that, not overflow."""
    sigma = np.full(5, 0.01)
    huge = TRISNoiseModel(sigma, common_mode_sigma_k=1e12)

    whitened_constant = huge.whiten(np.ones(5))

    np.testing.assert_allclose(whitened_constant, 0.0, atol=1e-6)


@pytest.mark.parametrize("common_mode", [1e4, 1e5, 1e6, 1e9])
def test_fit_is_stable_where_a_dense_cholesky_silently_returns_garbage(common_mode):
    """Regression: at sigma_c = 1e5 the dense route was wrong by ~190 sigma.

    The independent oracle marginalizes the offset by *parameter augmentation*
    instead of through the covariance, which is exact in the large-sigma limit
    and shares no code with the noise model.
    """
    ra_deg = np.arange(24) * 15.0
    design = build_tris_fourier_design(ra_deg, m_max=2, include_constant=False)
    truth = np.array([2.0, -1.0, 0.5, 0.25])
    sigma = np.linspace(0.004, 0.02, ra_deg.size)
    ring = _ring(ra_deg, design @ truth + 3.0, sigma)

    fit = fit_tris_linear_model(ring, design, common_mode_sigma_k=common_mode)

    augmented = np.column_stack([design, np.ones(ra_deg.size)])
    weights = 1.0 / sigma
    solution, *_ = np.linalg.lstsq(
        augmented * weights[:, None], ring.temperature_k * weights, rcond=None
    )

    np.testing.assert_allclose(fit.coefficients, solution[:-1], rtol=1e-6, atol=1e-8)
    assert np.isfinite(fit.coefficient_covariance).all()


def test_noise_model_requires_a_floor_for_the_zero_entry_every_real_ring_has():
    """Both published rings contain exactly one zero statistical uncertainty."""
    with pytest.raises(ValueError, match="uncertainty_floor_k"):
        TRISNoiseModel(np.array([0.0, 0.01]))

    floored = TRISNoiseModel(np.array([0.0, 0.01]), uncertainty_floor_k=0.004)
    np.testing.assert_allclose(floored.statistical_sigma_k, [0.004, 0.01])


# ---------------------------------------------------------------------------
# Map-making inputs
# ---------------------------------------------------------------------------


def _small_inputs(nside=8, n_samples=12, **kwargs):
    ra_deg = np.arange(n_samples) * (360.0 / n_samples)
    ring = _ring(ra_deg, np.full(n_samples, 15.0), np.full(n_samples, 0.01))
    return ring, build_tris_mapmaking_inputs(
        ring, nside=nside, cuts=_tabulated_cuts(), **kwargs
    )


def test_operator_reproduces_limtod_forward_model_to_machine_precision():
    """If the operator and generate_TOD_sky disagree, one of them is wrong."""
    nside = 8
    ring, inputs = _small_inputs(
        nside=nside, pixel_indices=np.arange(hp.nside2npix(nside))
    )
    rng = np.random.default_rng(11)
    sky = 10.0 + rng.standard_normal(hp.nside2npix(nside))

    geometry = tris_zenith_geometry(ring.ra_deg)
    reference = generate_TOD_sky(
        tris_cut_beam_map(_tabulated_cuts(), nside=nside, normalization="peak"),
        sky,
        geometry.lst_deg,
        geometry.latitude_deg,
        geometry.azimuth_deg,
        geometry.elevation_deg,
        geometry.selfrot_deg,
        normalize_beam=True,
        horizontal_mask=tris_horizon_mask(nside),
    )

    np.testing.assert_allclose(inputs.operator @ sky, reference, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(inputs.beam_coverage, 1.0, atol=1e-12)


def test_declination_band_reports_the_beam_power_it_drops():
    """Pixels outside the band are removed from the model, not down-weighted."""
    _ring_data, narrow = _small_inputs(dec_half_width_deg=20.0)
    _ring_data, wide = _small_inputs(dec_half_width_deg=60.0)

    assert narrow.sky_parameter_count < wide.sky_parameter_count
    assert narrow.beam_coverage.max() < wide.beam_coverage.min()
    assert wide.beam_coverage.min() > 0.9


def test_ring_pixels_selects_a_declination_band_around_the_boresight():
    """A disc would be the wrong shape: the ring sweeps every right ascension."""
    nside = 16
    pixels = tris_ring_pixels(nside, dec_deg=42.43, half_width_deg=20.0)
    theta, phi = hp.pix2ang(nside, pixels)
    declination = 90.0 - np.rad2deg(theta)

    assert np.all(np.abs(declination - 42.43) <= 20.0)
    assert np.ptp(np.rad2deg(phi)) > 350.0  # spans all right ascensions
    with pytest.raises(ValueError, match="no pixels"):
        tris_ring_pixels(nside, dec_deg=-89.999, half_width_deg=1e-3)


def test_zero_level_column_is_opt_in_and_needs_its_own_prior():
    """A free offset without a prior is unidentifiable against the sky monopole."""
    _ring_data, plain = _small_inputs()
    _ring_data, with_offset = _small_inputs(zero_level_sigma_k=0.066)

    assert not plain.has_zero_level
    assert with_offset.has_zero_level
    assert with_offset.parameter_count == with_offset.sky_parameter_count + 1
    np.testing.assert_allclose(with_offset.operator[:, -1], 1.0)


def test_zero_level_is_all_but_exactly_degenerate_with_the_sky_monopole():
    """The headline caveat for anyone fitting an offset from a single ring."""
    _ring_data, inputs = _small_inputs(zero_level_sigma_k=0.066)

    assert inputs.monopole_degeneracy > 1.0 - 1e-6
    implied = inputs.implied_monopole_prior_sigma_k(
        np.full(inputs.sky_parameter_count, 2.0)
    )
    assert 0.0 < implied < 2.0  # the per-pixel width shrinks by ~sqrt(N)


def test_predict_and_with_data_round_trip_a_simulated_ring():
    """Closure: what the operator predicts is what the solver is handed."""
    _ring_data, inputs = _small_inputs(zero_level_sigma_k=0.066)
    rng = np.random.default_rng(5)
    sky = 12.0 + rng.standard_normal(inputs.sky_parameter_count)

    simulated = inputs.predict(sky, 0.05)
    replaced = inputs.with_data(simulated)

    np.testing.assert_allclose(replaced.data_k, simulated)
    assert replaced.operator is inputs.operator
    with pytest.raises(ValueError, match="data_k"):
        inputs.with_data(simulated[:-1])
    with pytest.raises(ValueError, match="sky_k"):
        inputs.predict(sky[:-1])


def test_prior_regularized_solve_recovers_the_measured_directions():
    """The ring constrains the beam-convolved sky; the prior supplies the rest."""
    nside = 8
    _ring_data, inputs = _small_inputs(
        nside=nside, dec_half_width_deg=50.0, zero_level_sigma_k=0.066
    )
    rng = np.random.default_rng(19)
    truth = 12.0 + 3.0 * rng.standard_normal(inputs.sky_parameter_count)
    offset = 0.04
    simulated = inputs.with_data(inputs.predict(truth, offset))

    template = truth + 1.5 * rng.standard_normal(truth.size)
    solution = simulated.solve(prior_map=template, prior_sigma_k=1.5)

    sky_operator = inputs.operator[:, : inputs.sky_parameter_count]
    prior_forward = sky_operator @ (template - truth)
    posterior_forward = sky_operator @ (solution.sky_k - truth)

    # the data pins the directions it measures far better than the prior did
    assert np.std(posterior_forward) < 0.1 * np.std(prior_forward)
    assert solution.sky_k.size == inputs.sky_parameter_count
    assert np.all(np.isfinite(solution.sky_uncertainty_k))
    assert solution.zero_level_k is not None


def test_solution_scatters_back_onto_a_full_sky_map():
    """Selected pixels must land where they came from, and nowhere else."""
    nside = 8
    _ring_data, inputs = _small_inputs(nside=nside, zero_level_sigma_k=0.066)
    rng = np.random.default_rng(2)
    truth = 11.0 + rng.standard_normal(inputs.sky_parameter_count)
    solution = inputs.with_data(inputs.predict(truth)).solve(
        prior_map=truth, prior_sigma_k=1.0
    )

    full = solution.healpix_map()

    np.testing.assert_allclose(full[inputs.pixel_indices], solution.sky_k)
    outside = np.setdiff1d(np.arange(hp.nside2npix(nside)), inputs.pixel_indices)
    assert np.all(full[outside] == hp.UNSEEN)
    assert solution.healpix_uncertainty().shape == (hp.nside2npix(nside),)


def test_prior_from_template_tracks_brightness_and_keeps_a_floor():
    """A relative prior must not collapse to a delta where the template is ~0."""
    template = np.array([0.0, 1.0, 100.0, 10.0])
    indices = np.array([0, 2, 3])

    guess, sigma = tris_prior_from_template(
        template, indices, relative_sigma=0.1, floor_sigma_k=0.05
    )

    np.testing.assert_allclose(guess, [0.0, 100.0, 10.0])
    assert sigma[0] == pytest.approx(0.05)  # floor, not zero
    assert sigma[1] == pytest.approx(10.0)
    assert sigma[2] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="relative_sigma"):
        tris_prior_from_template(template, indices)


def test_builder_rejects_ambiguous_or_mismatched_inputs():
    """Fail at the boundary rather than producing a quietly wrong operator."""
    ra_deg = np.arange(6) * 60.0
    ring = _ring(ra_deg, np.full(6, 15.0), np.full(6, 0.01))

    with pytest.raises(ValueError, match="exactly one of cuts or beam_map"):
        build_tris_mapmaking_inputs(ring, nside=8)
    with pytest.raises(ValueError, match="exactly one of cuts or beam_map"):
        build_tris_mapmaking_inputs(
            ring, nside=8, cuts=_tabulated_cuts(), beam_map=np.ones(hp.nside2npix(8))
        )
    with pytest.raises(TypeError, match="TRISRing"):
        build_tris_mapmaking_inputs(object(), nside=8, cuts=_tabulated_cuts())
    with pytest.raises(ValueError, match="same number of samples"):
        build_tris_mapmaking_inputs(
            ring,
            nside=8,
            cuts=_tabulated_cuts(),
            geometry=tris_zenith_geometry(np.arange(5) * 70.0),
        )


def test_solution_reports_whether_the_model_can_describe_the_data():
    """Without this, a fitted zero level absorbing template error looks fine."""
    nside = 8
    _ring_data, inputs = _small_inputs(nside=nside, zero_level_sigma_k=0.066)
    rng = np.random.default_rng(23)
    truth = 12.0 + rng.standard_normal(inputs.sky_parameter_count)
    consistent = inputs.with_data(inputs.predict(truth, 0.02))

    good = consistent.solve(prior_map=truth, prior_sigma_k=1.0)
    assert good.reduced_chi_square < 1.0
    assert good.residual_k.size == inputs.data_k.size

    # a ring the model genuinely cannot fit: a large sample-to-sample offset
    inconsistent = inputs.with_data(
        inputs.predict(truth, 0.02)
        + 0.5 * np.cos(np.deg2rad(5.0 * inputs.geometry.lst_deg))
    )
    bad = inconsistent.solve(prior_map=truth, prior_sigma_k=1e-4)
    assert bad.reduced_chi_square > 100.0
