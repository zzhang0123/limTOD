import numpy as np
import pytest

from limTOD.mmode import (
    LSTWindow, conditioning, design_matrix, dpss_kernel, leakage_bound, legendre_basis,
    noise_amplification, pack, project_onto_basis, solve, spline_basis, synthesise, unpack,
)

MMAX = 40


def test_pack_unpack_roundtrip(modes):
    assert np.allclose(unpack(pack(modes)), modes)
    assert pack(modes).shape == (7, 2 * MMAX + 1)
    assert unpack(pack(modes[0])).shape == (MMAX + 1,)
    with pytest.raises(ValueError, match="odd"):
        unpack(np.zeros(4))


def test_synthesis_is_the_real_cosine_sine_form(arc7h, modes):
    ph = arc7h.phase
    manual = modes[:, 0].real[None, :] + 2 * np.real(
        np.exp(1j * np.outer(ph, np.arange(1, MMAX + 1))) @ modes[:, 1:].T)
    assert np.allclose(synthesise(arc7h, modes), manual)


def test_closed_scan_solves_every_mode_exactly(closed, modes):
    res = solve(synthesise(closed, modes), closed, m_trunc=MMAX)
    assert res.n_valid == closed.n and res.cond < 1.5
    assert np.allclose(res.modes, modes, atol=1e-9)
    assert np.allclose(res.residual_rms, 0.0, atol=1e-9)


def test_closed_scan_with_a_kernel_is_exact_up_to_leakage(closed, modes):
    res = solve(synthesise(closed, modes), closed, m_trunc=20, kernel=dpss_kernel(181, 4.0))
    assert res.n_valid == closed.n and res.cond < 2.0
    assert np.abs(res.d0 - modes[:, 0].real).max() < 1e-3


def test_ols_on_smoothed_data_keeps_the_kernels_suppression_and_gls_undoes_it(arc300):
    """The reason the default is OLS: whitening with S S^T divides K(m) back out.

    GLS on the smoothed data is exactly the raw solve projected onto row(S):
    the identity is pinned to 1e-6, and the stop-band mode leaks into d_0 by
    the same amount as with no kernel at all, while OLS suppresses it ~500x.
    """
    k = dpss_kernel(61, 3.0)                       # band edge ~35 on this arc; K(40) ~ 2e-4
    d = np.zeros(41, complex); d[40] = 10 * np.exp(0.7j)          # stop-band contamination only
    y = synthesise(arc300, d)
    ols = solve(y, arc300, 8, kernel=k, weighting="ols")
    gls = solve(y, arc300, 8, kernel=k, weighting="gls")
    plain = solve(y, arc300, 8)
    assert abs(ols.d0[0]) < 0.01 * abs(plain.d0[0])
    assert abs(gls.d0[0]) > 0.5 * abs(plain.d0[0])
    s, e = k.smoothing_operator(arc300), design_matrix(arc300, 8)
    proj = s.T @ np.linalg.solve(s @ s.T, s)
    manual = np.linalg.solve(e.T @ proj @ e, e.T @ proj @ y[:, 0])
    assert np.allclose(pack(gls.modes[0]), manual, rtol=1e-6, atol=1e-6)


def test_partial_arc_is_ill_conditioned_with_d0_unsplit(arc7h):
    k = dpss_kernel(31, 2.0)
    e = design_matrix(k.valid(arc7h), 4, k.transfer(arc7h, 4))
    assert conditioning(e) > 1e3
    v = np.linalg.svd(e / np.linalg.norm(e, axis=0))[2][-1]
    assert abs(v[0]) > 0.1 and np.abs(v[[1, 3]]).max() > 0.1


def test_reported_noise_matches_monte_carlo(arc300):
    k = dpss_kernel(41, 2.0)
    res = solve(np.zeros((arc300.n, 1)), arc300, m_trunc=3, kernel=k, sigma=1.0)
    rng = np.random.default_rng(0)
    draws = [solve(rng.normal(size=(arc300.n, 1)), arc300, 3, kernel=k).d0[0] for _ in range(600)]
    assert np.isclose(np.std(draws), res.std_d0()[0], rtol=0.1)


def test_reported_noise_with_weights_and_a_kernel_matches_monte_carlo():
    """The combination the first version got wrong: weights change the input variances before smoothing."""
    w = np.where(np.arange(301) < 150, 1.0, 10.0)
    win = LSTWindow(np.linspace(0.0, 150.0, 301), weights=w)
    k = dpss_kernel(21, 2.0)
    res = solve(np.zeros(301), win, 3, kernel=k, sigma=1.0)
    rng = np.random.default_rng(5)
    draws = [solve(rng.normal(size=301) / np.sqrt(w), win, 3, kernel=k).d0[0] for _ in range(600)]
    assert np.isclose(np.std(draws), res.std_d0()[0], rtol=0.1)


def test_weights_act_as_inverse_variances(closed, modes):
    y = synthesise(closed, modes)
    w2 = LSTWindow(closed.lst_deg, weights=np.full(closed.n, 2.0))
    a, b = solve(y, closed, 8), solve(y, w2, 8)
    assert np.allclose(a.modes, b.modes)
    assert np.isclose(b.cov[0, 0], a.cov[0, 0] / 2)


def test_supplied_noise_cov_is_used_as_given(arc300):
    w = LSTWindow(arc300.lst_deg, weights=np.full(arc300.n, 3.0))
    c = np.eye(arc300.n) * 0.25
    a = solve(np.zeros(arc300.n), w, 2, noise_cov=c)
    b = solve(np.zeros(arc300.n), arc300, 2, noise_cov=c)
    assert np.isclose(a.cov[0, 0], b.cov[0, 0])         # same covariance, no re-weighting


def test_std_on_the_7h_arc_matches_the_extended_precision_reference(arc7h):
    """The float64 path must not lose the small singular values: this is the error bar the package exists for."""
    with pytest.warns(UserWarning, match="condition number"):
        res = solve(np.zeros(arc7h.n), arc7h, 8, sigma=1.0)
    ref = noise_amplification(arc7h, 8)
    assert ref > 1e7
    assert np.isclose(res.std_d0()[0], ref, rtol=1e-4)
    e = design_matrix(arc7h, 8)
    assert np.sqrt(np.linalg.pinv(e.T @ e)[0, 0]) < ref / 10     # what the normal-matrix pinv would say


def test_noise_amplification_is_the_solve_at_unit_sigma(closed, arc300):
    assert np.isclose(noise_amplification(closed, 12), 1 / np.sqrt(closed.n), rtol=1e-9)
    k = dpss_kernel(61, 3.0)
    res = solve(np.zeros(arc300.n), arc300, 6, kernel=k)
    assert np.isclose(noise_amplification(arc300, 6, kernel=k), res.std_d0()[0], rtol=1e-8)


def test_leakage_bound_covers_the_realised_bias_including_weights_and_kernels(closed, arc300, modes):
    env = np.abs(modes).max(axis=0)
    stacked = LSTWindow.stacked_nights(100, 338.816, 105.288)
    cases = [(closed, 10, dpss_kernel(61, 3.0)), (arc300, 6, dpss_kernel(21, 2.0)),
             (arc300, 20, dpss_kernel(181, 4.0)), (stacked, 6, None)]
    # not tested: a truncation inside the kernel's stop band (K(M) ~ 1e-6, cond > 1e12), where the
    # amplified round-off of the float64 data exceeds the leakage; solve() warns there
    for w, m_trunc, k in cases:
        res = solve(synthesise(w, modes), w, m_trunc, kernel=k)
        actual = np.abs(res.d0 - modes[:, 0].real).max()
        bound = leakage_bound(w, m_trunc, env, kernel=k)
        assert actual <= bound * (1 + 1e-6) + 1e-9, (w.n, m_trunc)
    assert leakage_bound(closed, MMAX, np.abs(modes[0])) == 0.0


def test_joint_solve_is_per_channel_solve_then_projection(arc300, modes):
    """The Kronecker identity: the frequency basis acts after the time solve."""
    freqs = np.linspace(55, 85, 7)
    k = dpss_kernel(41, 2.0)
    m_trunc, n_par = 3, 4
    ys = k.smooth(synthesise(arc300, modes), arc300)
    e = design_matrix(k.valid(arc300), m_trunc, k.transfer(arc300, m_trunc))
    res = solve(synthesise(arc300, modes), arc300, m_trunc, kernel=k)
    basis = legendre_basis(freqs, n_par)
    joint = np.linalg.lstsq(np.kron(e, basis), ys.ravel(), rcond=None)[0]
    d0_joint = basis @ joint.reshape(2 * m_trunc + 1, n_par)[0]
    assert np.allclose(d0_joint, project_onto_basis(res.d0, basis), rtol=1e-8, atol=1e-8)


def test_projection_handles_a_rank_deficient_basis():
    x = np.linspace(-1, 1, 20)
    s = 1 + 2 * x + 0.3 * x ** 3
    full = project_onto_basis(s, np.column_stack([np.ones(20), x]))
    dup = project_onto_basis(s, np.column_stack([np.ones(20), x, x, np.zeros(20)]))
    assert np.allclose(full, dup)


@pytest.mark.parametrize("n_par", [4, 8, 12])
def test_bases_have_the_requested_size(n_par):
    f = np.linspace(55, 85, 61)
    assert spline_basis(f, n_par).shape == (61, n_par)
    assert legendre_basis(f, n_par).shape == (61, n_par)


def test_result_accessors(arc300, modes):
    res = solve(synthesise(arc300, modes), arc300, 3)
    assert res.std_d0().shape == (7,)
    assert res.std_mode(2).shape == (7, 2) and res.std_mode(0).shape == (7, 1)
    with pytest.raises(ValueError, match="m must"):
        res.std_mode(4)


def test_solve_validates_inputs(arc7h):
    with pytest.raises(ValueError, match="samples"):
        solve(np.zeros(10), arc7h, 2)
    with pytest.raises(ValueError, match="unknowns"):
        solve(np.zeros(arc7h.n), arc7h, 120)
    with pytest.raises(ValueError, match="sigma"):
        solve(np.zeros(arc7h.n), arc7h, 2, sigma=0.0)
    with pytest.raises(ValueError, match="one value per channel"):
        solve(np.zeros((arc7h.n, 3)), arc7h, 2, sigma=np.ones(2))
    with pytest.raises(ValueError, match="weighting"):
        solve(np.zeros(arc7h.n), arc7h, 2, weighting="ridge")


def _null_kernel(window, m_null, length=41):
    """A symmetric unit-sum kernel whose transfer is nulled at m_null on this window."""
    dt = np.deg2rad(window.cadence_deg)
    offs = (np.arange(length) - (length - 1) // 2) * dt
    taps = np.ones(length)
    c = np.cos(m_null * offs)
    taps -= c * (taps @ c) / (c @ c)                    # Gram-Schmidt against cos(m_null tau)
    from limTOD.mmode.kernel import Kernel
    return Kernel(taps, label=f"null(m={m_null})")


def test_a_mode_the_kernel_nulls_gets_a_huge_error_not_a_tiny_one(closed):
    """Round-2 finding: the SVD floor must act on mode confusion, not on K(m) scaling."""
    k = _null_kernel(closed, 15)
    assert abs(k.transfer(closed, 15)[15]) < 1e-12
    modes = np.zeros(16, complex); modes[15] = 5.0
    with pytest.raises(ValueError, match="nulls mode m=15"):
        solve(synthesise(closed, modes), closed, 15, kernel=k, sigma=1e-6)
    k2 = _null_kernel(closed, 15)
    taps = k2.taps + 1e-9 * np.cos(15 * (np.arange(41) - 20) * np.deg2rad(closed.cadence_deg))
    from limTOD.mmode.kernel import Kernel
    k3 = Kernel(taps)                                   # K(15) ~ 1e-9: tiny but not zero
    res = solve(synthesise(closed, modes), closed, 15, kernel=k3, sigma=1e-6)
    assert res.cond < 2.0                               # the window itself separates everything
    assert np.isclose(res.modes[0, 15].real, 5.0, rtol=1e-3)
    # on a closed scan smoothing scales a mode's signal and its noise by the same K(m), so the
    # error on that mode is exactly what it is without the kernel -- neither ~0 nor huge
    plain = solve(synthesise(closed, modes), closed, 15, sigma=1e-6)
    assert np.all(np.isfinite(res.std_mode(15)))
    assert np.allclose(res.std_mode(15), plain.std_mode(15), rtol=1e-3)


def test_std_of_a_higher_mode_matches_monte_carlo_with_a_kernel(arc300):
    k = dpss_kernel(41, 2.0)
    res = solve(np.zeros(arc300.n), arc300, 3, kernel=k, sigma=1.0)
    rng = np.random.default_rng(7)
    draws = np.array([pack(solve(rng.normal(size=arc300.n), arc300, 3, kernel=k).modes[0]) for _ in range(600)])
    assert np.allclose(draws.std(axis=0)[[3, 4]], res.std_mode(2)[0], rtol=0.12)


def test_estimator_branches_agree_across_the_dispatch_threshold(closed):
    from limTOD.mmode import linear as L
    e = design_matrix(closed, 6)
    w = np.ones(closed.n)
    p_svd = L._estimator(e, w, 1.0)
    p_mp = L._estimator(e, w, 1e13)                     # force the extended-precision branch
    assert np.allclose(p_svd, p_mp, rtol=1e-10, atol=1e-14)


def test_gls_refuses_a_kernel_that_makes_the_covariance_singular(arc300, modes):
    with pytest.raises(np.linalg.LinAlgError, match="weighting='ols'"):
        solve(synthesise(arc300, modes), arc300, 20, kernel=dpss_kernel(181, 4.0), weighting="gls")


def test_nan_in_the_tod_is_refused(arc300):
    y = np.zeros(arc300.n); y[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        solve(y, arc300, 2)


def test_leakage_bound_and_amplification_warn_on_a_degenerate_window(arc7h):
    with pytest.warns(UserWarning, match="condition number"):
        leakage_bound(arc7h, 8, np.ones(41))
    with pytest.warns(UserWarning, match="condition number"):
        noise_amplification(arc7h, 8)


def test_modes_beyond_the_sampling_nyquist_are_refused(arc300):
    assert 350 < arc300.nyquist_m < 370
    with pytest.raises(ValueError, match="Nyquist"):
        design_matrix(arc300, 400)
    with pytest.raises(ValueError, match="Nyquist"):
        leakage_bound(arc300, 8, np.ones(800), kernel=dpss_kernel(61, 3.0))
    assert leakage_bound(arc300, 8, np.ones(300), kernel=dpss_kernel(61, 3.0)) < np.inf
