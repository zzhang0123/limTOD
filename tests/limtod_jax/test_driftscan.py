"""Drift-scan m-mode suite: convention locks, oracle equivalence, adjoint,
horizon mask, and jit/vmap/grad safety.

Layers (each lock is NUMERICAL — no convention is trusted on paper):

1. The drift phase law ``B_lm(lst) = B_lm(ref)·e^{−imΔ}`` and the TOD
   synthesis sign ``e^{+imΔ}`` — both signs are raced, the loser must fail
   catastrophically.
2. ``driftscan_tod`` == ``generate_tod_sky`` (the generic per-sample-rotation
   path) to float64 roundoff, on a grid that includes the extreme corners
   (zenith gimbal, equator/pole latitudes, LST wrap past 360°, Δ = 0).
3. ``driftscan_tod`` == numpy ``limTOD.simulator.generate_TOD_sky`` via the
   quadrature-alm exactness recipe (see test_oracle_equivalence.py).
4. m-modes are the TOD's Fourier coefficients (MmodeNote eqn 15): FFT lock.
5. Adjoint: dot-test AND direct equality with the generic adjoint.
6. Horizon mask: healpy-oracle equality of the masked beam-local alms, and
   masked drift TOD == masked generic TOD.
"""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from limtod_jax.alm import alm_dot, packed_lm_arrays
from limtod_jax.angles import zyz_of_pointing, zyzyz2zyz
from limtod_jax.core import generate_tod_sky, generate_tod_sky_adjoint, rotate_alm
from limtod_jax.driftscan import (
    DriftScanMmode,
    beam_alm_at_reference,
    driftscan_tod,
    driftscan_tod_adjoint,
    horizon_masked_beam_alm,
    horizon_weights,
    mmodes_from_sky,
    mmodes_from_tod,
    tod_from_mmodes,
    _pixel_thetas,
)
from limtod_jax.hpx import map2alm_iter, map2alm_quad, ones_quadrature_alm

pytestmark = pytest.mark.filterwarnings("ignore:Gimbal lock detected")

NSIDE, LMAX = 16, 32
NPIX = 12 * NSIDE * NSIDE

# One fixed drift-scan pointing plus corner variants used across the file.
LAT, AZ, EL, SELFROT = -30.71, 41.5, 52.5, 10.0
LST_REF = 12.0
# Includes Δ=0 (the reference itself), a wrap past 360°, and irregular steps.
LSTS = np.array([12.0, 61.3, 155.0, 262.9, 359.9, 401.6])


def _dphi(lsts=LSTS, ref=LST_REF):
    return jnp.deg2rad(jnp.asarray(lsts) - ref)


def _zyz_stack(lsts, lat=LAT, az=AZ, el=EL, sr=SELFROT):
    psi, theta, phi = zyz_of_pointing(
        jnp.asarray(lsts), lat, jnp.asarray(az), jnp.asarray(el), jnp.asarray(sr)
    )
    return jnp.stack([psi, theta, phi], axis=-1)


@pytest.fixture(scope="module")
def fields(rng):
    """(beam_alm_local, beam_ref_alm, sky_alm, ones_alm) at NSIDE/LMAX."""
    theta, _ = hp.pix2ang(NSIDE, np.arange(NPIX))
    beam_map = np.exp(-(theta**2) / (2 * np.deg2rad(9.0) ** 2))
    beam_alm = jnp.asarray(hp.map2alm(beam_map, lmax=LMAX, iter=3))
    sky_map = jnp.asarray(rng.standard_normal(NPIX))
    sky_alm = map2alm_quad(sky_map, nside=NSIDE, lmax=LMAX)
    beam_ref = beam_alm_at_reference(beam_alm, LST_REF, LAT, AZ, EL, SELFROT, lmax=LMAX)
    ones_alm = ones_quadrature_alm(nside=NSIDE, lmax=LMAX)
    return beam_alm, beam_ref, sky_alm, ones_alm


# ------------------------------------------------------------ phase law lock
def test_drift_phase_sign_locked(fields):
    """B_lm(lst) = B_lm(ref)·e^{−imΔ}; the +imΔ candidate must lose badly."""
    beam_alm, beam_ref, _, _ = fields
    lst1 = 87.0
    delta = np.deg2rad(lst1 - LST_REF)
    _, ms = packed_lm_arrays(LMAX)
    b1 = rotate_alm(
        beam_alm, *zyz_of_pointing(lst1, LAT, AZ, EL, SELFROT), lmax=LMAX
    )
    scale = float(jnp.max(jnp.abs(beam_ref)))
    err_minus = float(jnp.max(jnp.abs(b1 - beam_ref * jnp.exp(-1j * ms * delta))))
    err_plus = float(jnp.max(jnp.abs(b1 - beam_ref * jnp.exp(+1j * ms * delta))))
    assert err_minus < 1e-12 * scale, f"phase law broken: {err_minus:.3e}"
    assert err_plus > 1e-2 * scale, "sign race degenerate — test lost its teeth"


def test_tod_synthesis_sign_locked(fields):
    """TOD phase is e^{+imΔ} (conjugation flips the beam's −imΔ)."""
    _, beam_ref, sky_alm, _ = fields
    tod_generic = generate_tod_sky(*_generic_args(fields), lmax=LMAX)
    c = mmodes_from_sky(beam_ref, sky_alm, lmax=LMAX)
    good = tod_from_mmodes(c, _dphi())
    flipped = tod_from_mmodes(jnp.conj(c), _dphi())  # == e^{−imΔ} synthesis
    scale = float(jnp.max(jnp.abs(tod_generic)))
    assert float(jnp.max(jnp.abs(good - tod_generic))) < 1e-12 * scale
    assert float(jnp.max(jnp.abs(flipped - tod_generic))) > 1e-3 * scale


def _generic_args(fields):
    beam_alm, _, sky_alm, _ = fields
    return beam_alm, sky_alm, _zyz_stack(LSTS)


# ------------------------------------------- equivalence with the generic path
@pytest.mark.parametrize("normalize", [False, True])
def test_matches_generic_jax(fields, normalize):
    beam_alm, beam_ref, sky_alm, ones_alm = fields
    generic = generate_tod_sky(
        beam_alm, sky_alm, _zyz_stack(LSTS),
        lmax=LMAX, normalize=normalize, ones_alm=ones_alm if normalize else None,
    )
    fast = driftscan_tod(
        beam_ref, sky_alm, _dphi(),
        lmax=LMAX, normalize=normalize, ones_alm=ones_alm if normalize else None,
    )
    rel = float(jnp.max(jnp.abs(fast - generic)) / jnp.max(jnp.abs(generic)))
    assert rel < 1e-12, f"rel err {rel:.3e}"


@pytest.mark.parametrize(
    "lat,az,el,sr",
    [
        (53.24, 0.0, 90.0, 0.0),   # zenith: gimbal lock in the ref rotation
        (0.0, 180.0, 5.0, 30.0),   # equator site, low elevation
        (-90.0, 270.0, 41.0, 0.0), # pole site
    ],
)
def test_matches_generic_at_corners(fields, rng, lat, az, el, sr):
    beam_alm, _, sky_alm, _ = fields
    beam_ref = beam_alm_at_reference(beam_alm, LST_REF, lat, az, el, sr, lmax=LMAX)
    generic = generate_tod_sky(
        beam_alm, sky_alm, _zyz_stack(LSTS, lat, az, el, sr), lmax=LMAX
    )
    fast = driftscan_tod(beam_ref, sky_alm, _dphi(), lmax=LMAX)
    rel = float(jnp.max(jnp.abs(fast - generic)) / jnp.max(jnp.abs(generic)))
    assert rel < 1e-11, f"rel err {rel:.3e} at {(lat, az, el, sr)}"


def test_lmax_small_extreme(rng):
    """Boundary rule: the tiny-lmax corner (lmax=2) must agree too.

    Random valid packed alms directly (m=0 real) — both paths consume the
    same alms, so no map transform is needed (s2fft's healpix transforms
    reject lmax ≪ nside, which is irrelevant to this comparison).
    """
    lmax = 2
    _, ms = packed_lm_arrays(lmax)

    def _random_alm():
        a = rng.standard_normal(len(ms)) + 1j * rng.standard_normal(len(ms))
        a[ms == 0] = a[ms == 0].real
        return jnp.asarray(a)

    beam_alm, sky_alm = _random_alm(), _random_alm()
    beam_ref = beam_alm_at_reference(beam_alm, LST_REF, LAT, AZ, EL, 0.0, lmax=lmax)
    generic = generate_tod_sky(
        beam_alm, sky_alm, _zyz_stack(LSTS, sr=0.0), lmax=lmax
    )
    fast = driftscan_tod(beam_ref, sky_alm, _dphi(), lmax=lmax)
    rel = float(jnp.max(jnp.abs(fast - generic)) / jnp.max(jnp.abs(generic)))
    assert rel < 1e-12, f"rel err {rel:.3e}"


def test_dphi_zero_is_reference_sample(fields):
    """Δ ≡ 0: the TOD is constant and equals the reference-LST sample."""
    beam_alm, beam_ref, sky_alm, _ = fields
    fast = driftscan_tod(beam_ref, sky_alm, jnp.zeros(4), lmax=LMAX)
    ref = generate_tod_sky(
        beam_alm, sky_alm, _zyz_stack(np.array([LST_REF])), lmax=LMAX
    )
    np.testing.assert_allclose(np.asarray(fast), float(ref[0]), rtol=1e-12)


# ------------------------------------------------------- numpy limTOD oracle
@pytest.mark.parametrize("normalize", [False, True])
def test_matches_numpy_oracle(rng, quad_alm, beam_alm_iter3, oracle_tod, normalize):
    nside, lmax = 8, 23
    npix = hp.nside2npix(nside)
    beam_map, sky_map = rng.random(npix), rng.random(npix)
    n_t = len(LSTS)
    direct = oracle_tod(
        beam_map, sky_map, LSTS, LAT,
        np.full(n_t, AZ), np.full(n_t, EL), np.full(n_t, SELFROT),
        normalize_beam=normalize,
    )
    beam_alm = jnp.asarray(beam_alm_iter3(beam_map, lmax))
    beam_ref = beam_alm_at_reference(beam_alm, LST_REF, LAT, AZ, EL, SELFROT, lmax=lmax)
    ones_alm = jnp.asarray(quad_alm(np.ones(npix), lmax))
    fast = driftscan_tod(
        beam_ref, jnp.asarray(quad_alm(sky_map, lmax)), _dphi(),
        lmax=lmax, normalize=normalize, ones_alm=ones_alm if normalize else None,
    )
    rel = np.max(np.abs(np.asarray(fast) - direct)) / np.max(np.abs(direct))
    assert rel < 1e-6, f"rel err {rel:.3e}"


def test_ncp_gimbal_drift_matches_numpy_oracle(rng, quad_alm, beam_alm_iter3, oracle_tod):
    """Celestial-pole drift scan (az=0, el=lat): the full pointing chain
    collapses to a pure z-rotation, so EVERY sample takes the theta~0 gimbal
    branch of the jax angle extraction. The numpy oracle goes through scipy,
    NOT through limtod_jax.angles — so a broken gimbal branch cannot cancel
    between the two sides (it does cancel in the drift-vs-generic tests,
    which share the angle chain; review-confirmed gap)."""
    nside, lmax = 8, 23
    lat = 53.24
    az, el, sr = 0.0, lat, 0.0  # boresight on the NCP
    npix = hp.nside2npix(nside)
    beam_map, sky_map = rng.random(npix), rng.random(npix)
    n_t = len(LSTS)
    direct = oracle_tod(
        beam_map, sky_map, LSTS, lat,
        np.full(n_t, az), np.full(n_t, el), np.full(n_t, sr),
    )
    beam_alm = jnp.asarray(beam_alm_iter3(beam_map, lmax))
    beam_ref = beam_alm_at_reference(beam_alm, LST_REF, lat, az, el, sr, lmax=lmax)
    fast = driftscan_tod(
        beam_ref, jnp.asarray(quad_alm(sky_map, lmax)), _dphi(), lmax=lmax
    )
    rel = np.max(np.abs(np.asarray(fast) - direct)) / np.max(np.abs(direct))
    assert rel < 1e-6, f"rel err {rel:.3e} (gimbal branch broken?)"


# --------------------------------------------- m-modes are Fourier coefficients
def test_mmodes_are_fourier_coefficients(fields):
    """MmodeNote eqn (15): uniform full-circle TOD -> FFT == mmodes."""
    _, beam_ref, sky_alm, _ = fields
    n_t = 4 * (LMAX + 1)  # comfortably above Nyquist (n_t > 2·lmax)
    dphi = jnp.asarray(2.0 * np.pi * np.arange(n_t) / n_t)
    tod = driftscan_tod(beam_ref, sky_alm, dphi, lmax=LMAX)
    vm_fft = np.fft.fft(np.asarray(tod)) / n_t  # e^{−2πi·mt/n} convention
    vm = np.asarray(mmodes_from_sky(beam_ref, sky_alm, lmax=LMAX))
    scale = np.max(np.abs(vm))
    np.testing.assert_allclose(vm_fft[: LMAX + 1], vm, atol=1e-12 * scale)
    # negative-m redundancy for real fields: Ṽ_{−m} = conj(Ṽ_m)
    np.testing.assert_allclose(
        vm_fft[-LMAX:], np.conj(vm[1:][::-1]), atol=1e-12 * scale
    )
    # and the DFT estimator inverts the synthesis on this sampling
    vm_back = np.asarray(mmodes_from_tod(tod, dphi, lmax=LMAX))
    np.testing.assert_allclose(vm_back, vm, atol=1e-12 * scale)


def test_mmodes_from_tod_offset_and_irregular_grids(fields, rng):
    """Uniform grid with a nonzero offset still inverts exactly; on an
    irregular grid the estimator must equal the documented DFT formula
    (independent numpy evaluation — pins the e^{-imΔ} sign on its own)."""
    _, beam_ref, sky_alm, _ = fields
    vm = np.asarray(mmodes_from_sky(beam_ref, sky_alm, lmax=LMAX))
    scale = np.max(np.abs(vm))

    n_t = 4 * (LMAX + 1)
    offset = 0.7331
    dphi_off = jnp.asarray(2.0 * np.pi * np.arange(n_t) / n_t + offset)
    tod_off = driftscan_tod(beam_ref, sky_alm, dphi_off, lmax=LMAX)
    vm_off = np.asarray(mmodes_from_tod(tod_off, dphi_off, lmax=LMAX))
    np.testing.assert_allclose(vm_off, vm, atol=1e-11 * scale)

    dphi_irr = jnp.asarray(np.sort(rng.uniform(0.0, 2.0 * np.pi, 37)))
    tod_irr = np.asarray(driftscan_tod(beam_ref, sky_alm, dphi_irr, lmax=LMAX))
    got = np.asarray(mmodes_from_tod(jnp.asarray(tod_irr), dphi_irr, lmax=LMAX))
    m = np.arange(LMAX + 1)
    expected = (
        np.exp(-1j * np.outer(m, np.asarray(dphi_irr))) @ tod_irr
    ) / len(tod_irr)
    np.testing.assert_allclose(got, expected, atol=1e-12 * np.max(np.abs(expected)))


def test_mmodes_lmax_inference(fields):
    """The lmax=None branch must infer the SAME band-limit as explicit lmax
    (an off-by-one here silently truncates or crashes the m aggregation)."""
    _, beam_ref, sky_alm, _ = fields
    explicit = mmodes_from_sky(beam_ref, sky_alm, lmax=LMAX)
    inferred = mmodes_from_sky(beam_ref, sky_alm)
    assert inferred.shape == (LMAX + 1,)
    np.testing.assert_array_equal(np.asarray(inferred), np.asarray(explicit))


def test_single_m_sky_is_pure_tone(fields):
    """A single-(l,m) sky produces a pure sinusoid at frequency m."""
    _, beam_ref, _, _ = fields
    ls, ms = packed_lm_arrays(LMAX)
    m0, l0 = 5, 9
    idx = int(np.nonzero((ls == l0) & (ms == m0))[0][0])
    sky = jnp.zeros(beam_ref.shape, beam_ref.dtype).at[idx].set(1.3 - 0.4j)
    c = np.asarray(mmodes_from_sky(beam_ref, sky, lmax=LMAX))
    assert np.max(np.abs(np.delete(c, m0))) < 1e-14 * max(np.abs(c[m0]), 1.0)
    dphi = _dphi()
    tod = np.asarray(driftscan_tod(beam_ref, sky, dphi, lmax=LMAX))
    expected = 2.0 * (
        np.real(c[m0]) * np.cos(m0 * np.asarray(dphi))
        - np.imag(c[m0]) * np.sin(m0 * np.asarray(dphi))
    )
    np.testing.assert_allclose(tod, expected, rtol=0, atol=1e-13 * np.max(np.abs(expected)))


# ------------------------------------------------------------------- adjoint
@pytest.mark.parametrize("normalize", [False, True])
def test_adjoint_dot_test(fields, rng, normalize):
    _, beam_ref, sky_alm, ones_alm = fields
    y = jnp.asarray(rng.standard_normal(len(LSTS)))
    kw = dict(lmax=LMAX, normalize=normalize, ones_alm=ones_alm if normalize else None)
    lhs = float(jnp.sum(driftscan_tod(beam_ref, sky_alm, _dphi(), **kw) * y))
    rhs = float(alm_dot(driftscan_tod_adjoint(y, beam_ref, _dphi(), **kw), sky_alm))
    assert abs(lhs - rhs) / abs(lhs) < 1e-12


@pytest.mark.parametrize("normalize", [False, True])
def test_adjoint_matches_generic_adjoint(fields, rng, normalize):
    beam_alm, beam_ref, _, ones_alm = fields
    y = jnp.asarray(rng.standard_normal(len(LSTS)))
    generic = generate_tod_sky_adjoint(
        y, beam_alm, _zyz_stack(LSTS),
        lmax=LMAX, normalize=normalize, ones_alm=ones_alm if normalize else None,
    )
    fast = driftscan_tod_adjoint(
        y, beam_ref, _dphi(),
        lmax=LMAX, normalize=normalize, ones_alm=ones_alm if normalize else None,
    )
    scale = float(jnp.max(jnp.abs(generic)))
    assert float(jnp.max(jnp.abs(fast - generic))) < 1e-11 * scale


# -------------------------------------------------------------- horizon mask
@pytest.mark.parametrize("nside", [4, 16, 64])
def test_pixel_thetas_match_healpy(nside):
    theta_hp, _ = hp.pix2ang(nside, np.arange(12 * nside * nside))
    np.testing.assert_allclose(_pixel_thetas(nside), theta_hp, atol=1e-12)


@pytest.mark.parametrize("iterations", [0, 3])
def test_map2alm_iter_matches_healpy(rng, iterations):
    m = rng.standard_normal(NPIX)
    ours = np.asarray(
        map2alm_iter(jnp.asarray(m), nside=NSIDE, lmax=LMAX, iterations=iterations)
    )
    ref = hp.map2alm(m, lmax=LMAX, iter=iterations)
    np.testing.assert_allclose(ours, ref, atol=1e-10 * np.max(np.abs(ref)))


@pytest.mark.parametrize("apod_deg", [0.0, 5.0])
def test_horizon_masked_beam_healpy_oracle(fields, apod_deg):
    """JAX mask chain == the same chain built from healpy primitives."""
    beam_alm, _, _, _ = fields
    az, el, sr = 30.0, 12.0, 20.0  # low elevation: the mask actually bites
    ours = np.asarray(
        horizon_masked_beam_alm(
            beam_alm, az, el, sr, nside=NSIDE, lmax=LMAX, apod_deg=apod_deg
        )
    )
    # healpy oracle. Locked slot convention (limtod_jax.wigner):
    # jax rotate_alm(x, psi, theta, phi) == hp.rotate_alm(x, phi, theta, psi).
    psi, theta, phi = (
        float(a) for a in zyzyz2zyz(0.0, 0.0, -az, el - 90.0, sr)
    )
    a_h = np.array(beam_alm)  # hp.rotate_alm mutates in place — copy first
    hp.rotate_alm(a_h, phi, theta, psi)
    m_h = hp.alm2map(a_h, NSIDE) * horizon_weights(NSIDE, apod_deg)
    ref = hp.map2alm(m_h, lmax=LMAX, iter=3)
    hp.rotate_alm(ref, -psi, -theta, -phi)
    np.testing.assert_allclose(ours, ref, atol=1e-9 * np.max(np.abs(ref)))


def test_masked_drift_equals_masked_generic(fields):
    """The masked beam drops into both paths and they still agree."""
    beam_alm, _, sky_alm, _ = fields
    az, el, sr = 30.0, 12.0, 0.0
    masked = horizon_masked_beam_alm(
        beam_alm, az, el, sr, nside=NSIDE, lmax=LMAX, apod_deg=2.0
    )
    beam_ref = beam_alm_at_reference(masked, LST_REF, LAT, az, el, sr, lmax=LMAX)
    generic = generate_tod_sky(
        masked, sky_alm, _zyz_stack(LSTS, LAT, az, el, sr), lmax=LMAX
    )
    fast = driftscan_tod(beam_ref, sky_alm, _dphi(), lmax=LMAX)
    rel = float(jnp.max(jnp.abs(fast - generic)) / jnp.max(jnp.abs(generic)))
    assert rel < 1e-12, f"rel err {rel:.3e}"


def test_ringing_apodization_mitigates():
    """Compact pin of docs/driftscan_ringing_study.py: for a wide beam at
    low elevation, the hard cut leaves real below-horizon Gibbs leakage in
    the band-limited masked beam, and a 5-deg cosine apodization reduces it
    by a large factor. (Study values at nside=32/lmax=64: hard 7.2e-3,
    apod5 1.2e-3.)"""
    from limtod_jax.hpx import alm2map

    nside, lmax = 32, 64
    npix = 12 * nside * nside
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    sigma = np.deg2rad(25.0) / np.sqrt(8.0 * np.log(2.0))
    beam = jnp.asarray(
        hp.map2alm(np.exp(-(theta**2) / (2 * sigma**2)), lmax=lmax, iter=3)
    )
    az, el = 41.5, 10.0
    psi, th_, phi = zyzyz2zyz(0.0, 0.0, -az, el - 90.0, 0.0)
    bh_map = alm2map(rotate_alm(beam, psi, th_, phi, lmax=lmax), nside=nside, lmax=lmax)
    below = horizon_weights(nside, 0.0) == 0.0
    peak = float(jnp.max(bh_map))

    def leak(apod):
        masked = bh_map * jnp.asarray(horizon_weights(nside, apod))
        recon = alm2map(
            map2alm_iter(masked, nside=nside, lmax=lmax), nside=nside, lmax=lmax
        )
        return float(jnp.sqrt(jnp.mean(recon[below] ** 2))) / peak

    hard, apod5 = leak(0.0), leak(5.0)
    assert hard > 3e-3, f"hard-cut ringing vanished? leak={hard:.2e}"
    assert apod5 < hard / 3.0, f"apodization stopped helping: {apod5:.2e} vs {hard:.2e}"
    assert apod5 < 2.5e-3, f"apod-5deg leak regressed: {apod5:.2e}"


def test_horizon_weights_shape_and_range():
    w0 = horizon_weights(NSIDE, 0.0)
    w5 = horizon_weights(NSIDE, 5.0)
    assert w0.shape == w5.shape == (NPIX,)
    assert set(np.unique(w0)) <= {0.0, 1.0}
    assert np.all((w5 >= 0.0) & (w5 <= 1.0))
    # apodization only REMOVES power near the horizon, never adds
    assert np.all(w5 <= w0 + 1e-15)


# ------------------------------------------------------- operator + jax safety
def test_operator_equals_functions(fields):
    beam_alm, beam_ref, sky_alm, ones_alm = fields
    op = DriftScanMmode.from_pointing(
        beam_alm, LSTS, LAT, AZ, EL, SELFROT,
        lmax=LMAX, lst_ref_deg=LST_REF, normalize=True, nside=NSIDE,
    )
    manual = driftscan_tod(
        beam_ref, sky_alm, _dphi(), lmax=LMAX, normalize=True, ones_alm=ones_alm
    )
    np.testing.assert_allclose(np.asarray(op(sky_alm)), np.asarray(manual), rtol=1e-13)
    np.testing.assert_allclose(
        np.asarray(op.mmodes(sky_alm)),
        np.asarray(mmodes_from_sky(beam_ref, sky_alm, lmax=LMAX)),
        rtol=1e-13,
    )
    y = jnp.asarray(np.linspace(-1, 1, len(LSTS)))
    np.testing.assert_allclose(
        np.asarray(op.adjoint(y)),
        np.asarray(
            driftscan_tod_adjoint(
                y, beam_ref, _dphi(), lmax=LMAX, normalize=True, ones_alm=ones_alm
            )
        ),
        rtol=1e-13,
    )


def test_operator_explicit_reference_differs_from_first_sample(fields):
    """lst_ref_deg far from lst_deg[0]: the reference rotation and the dphi
    offsets must BOTH honor it (a from_pointing that measured dphi from
    lst_deg[0] while rotating to lst_ref_deg survived the old suite —
    review-confirmed mutation)."""
    beam_alm, _, sky_alm, _ = fields
    lst_ref = 237.5  # nowhere near LSTS[0] = 12.0
    op = DriftScanMmode.from_pointing(
        beam_alm, LSTS, LAT, AZ, EL, SELFROT, lmax=LMAX, lst_ref_deg=lst_ref
    )
    generic = generate_tod_sky(beam_alm, sky_alm, _zyz_stack(LSTS), lmax=LMAX)
    rel = float(jnp.max(jnp.abs(op(sky_alm) - generic)) / jnp.max(jnp.abs(generic)))
    assert rel < 1e-11, f"rel err {rel:.3e}"


def test_from_pointing_horizon_mask_matches_manual(fields):
    """horizon_mask=True must apply the mask with the SAME az/el/selfrot and
    forward apod_deg AND mask_iterations (nondefault value on purpose)."""
    beam_alm, _, sky_alm, _ = fields
    az, el, sr, apod, iters = 30.0, 12.0, 20.0, 2.0, 1
    op = DriftScanMmode.from_pointing(
        beam_alm, LSTS, LAT, az, el, sr,
        lmax=LMAX, lst_ref_deg=LST_REF, nside=NSIDE,
        horizon_mask=True, apod_deg=apod, mask_iterations=iters,
    )
    masked = horizon_masked_beam_alm(
        beam_alm, az, el, sr, nside=NSIDE, lmax=LMAX, apod_deg=apod, iterations=iters
    )
    manual_ref = beam_alm_at_reference(masked, LST_REF, LAT, az, el, sr, lmax=LMAX)
    np.testing.assert_allclose(
        np.asarray(op.beam_ref_alm), np.asarray(manual_ref), rtol=0, atol=1e-13
    )
    # and the masked operator differs from the unmasked one (mask not dropped)
    op_nomask = DriftScanMmode.from_pointing(
        beam_alm, LSTS, LAT, az, el, sr, lmax=LMAX, lst_ref_deg=LST_REF
    )
    diff = float(jnp.max(jnp.abs(op(sky_alm) - op_nomask(sky_alm))))
    assert diff > 1e-3 * float(jnp.max(jnp.abs(op_nomask(sky_alm))))


def test_mask_iterations_sensitivity(fields):
    """iterations must actually reach the analysis step: 0 and 3 iterations
    give measurably different masked beams (guards a hardcoded default)."""
    beam_alm, _, _, _ = fields
    kw = dict(nside=NSIDE, lmax=LMAX, apod_deg=0.0)
    m0 = horizon_masked_beam_alm(beam_alm, 30.0, 12.0, 0.0, iterations=0, **kw)
    m3 = horizon_masked_beam_alm(beam_alm, 30.0, 12.0, 0.0, iterations=3, **kw)
    rel = float(jnp.max(jnp.abs(m0 - m3)) / jnp.max(jnp.abs(m3)))
    # measured 6.4e-5 at nside=16/lmax=32; a hardcoded-iterations mutant
    # gives exactly 0, so any positive floor discriminates
    assert rel > 1e-5, f"iterations had no effect (rel {rel:.2e})"


def test_operator_rejects_batched_beam(fields):
    """The operator is single-beam by contract; a (n_freq, n_alm) beam must
    be rejected at construction, not fail later inside __call__."""
    _, beam_ref, _, _ = fields
    with pytest.raises(ValueError, match="single-beam"):
        DriftScanMmode(
            beam_ref_alm=jnp.stack([beam_ref, beam_ref]), dphi=_dphi(), lmax=LMAX
        )


def test_operator_default_reference_is_first_sample(fields):
    beam_alm, _, sky_alm, _ = fields
    op = DriftScanMmode.from_pointing(
        beam_alm, LSTS, LAT, AZ, EL, SELFROT, lmax=LMAX
    )
    assert float(op.dphi[0]) == 0.0
    ref = generate_tod_sky(
        beam_alm, sky_alm, _zyz_stack(np.array([LSTS[0]])), lmax=LMAX
    )
    np.testing.assert_allclose(float(op(sky_alm)[0]), float(ref[0]), rtol=1e-12)


def test_operator_is_jit_vmap_grad_safe(fields, rng):
    import equinox as eqx

    beam_alm, beam_ref, sky_alm, _ = fields
    op = DriftScanMmode(beam_ref_alm=beam_ref, dphi=_dphi(), lmax=LMAX)

    # jit through eqx.filter_jit (module as a traced pytree argument)
    tod_jit = eqx.filter_jit(lambda o, s: o(s))(op, sky_alm)
    np.testing.assert_allclose(np.asarray(tod_jit), np.asarray(op(sky_alm)), rtol=1e-13)

    # vmap over a frequency batch of skies
    skies = jnp.stack([sky_alm, 2.0 * sky_alm, sky_alm - 1.0j * 0.0])
    batched = jax.vmap(op)(skies)
    assert batched.shape == (3, len(LSTS))
    np.testing.assert_allclose(np.asarray(batched[1]), 2.0 * np.asarray(batched[0]), rtol=1e-12)

    # grad w.r.t. the sky is finite and input-independent (linear forward)
    def loss(s):
        return jnp.sum(op(s) ** 2)

    g1 = jax.grad(loss, holomorphic=False)(sky_alm)
    assert bool(jnp.all(jnp.isfinite(jnp.abs(g1))))

    def linear(s):
        return jnp.sum(op(s))

    ga = jax.grad(linear, holomorphic=False)(sky_alm)
    gb = jax.grad(linear, holomorphic=False)(3.0 * sky_alm + 2.0)
    np.testing.assert_allclose(np.asarray(ga), np.asarray(gb), rtol=1e-12)


def test_validation_errors(fields):
    beam_alm, beam_ref, sky_alm, _ = fields
    with pytest.raises(ValueError, match="lmax"):
        driftscan_tod(beam_ref[:-1], sky_alm, _dphi(), lmax=LMAX)
    with pytest.raises(ValueError, match="ones_alm"):
        driftscan_tod(beam_ref, sky_alm, _dphi(), lmax=LMAX, normalize=True)
    with pytest.raises(ValueError, match="1D"):
        driftscan_tod(beam_ref, sky_alm, _dphi().reshape(2, -1), lmax=LMAX)
    with pytest.raises(ValueError, match="n_time"):
        driftscan_tod_adjoint(jnp.zeros(3), beam_ref, _dphi(), lmax=LMAX)
    with pytest.raises(ValueError, match="nside"):
        DriftScanMmode.from_pointing(
            beam_alm, LSTS, LAT, AZ, EL, lmax=LMAX, normalize=True
        )
    with pytest.raises(ValueError, match="ones_alm"):
        DriftScanMmode(beam_ref_alm=beam_ref, dphi=_dphi(), lmax=LMAX, normalize=True)
