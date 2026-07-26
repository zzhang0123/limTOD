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
    dl_plane_for_pointing,
    driftscan_tod,
    driftscan_tod_adjoint,
    horizon_masked_beam_alm,
    horizon_weights,
    mmodes_from_sky,
    mmodes_from_tod,
    mmodes_from_tod_uniform,
    tod_from_mmodes,
    tod_from_mmodes_uniform,
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


# ------------------------------------------------------- uniform FFT fast path
def _uniform_dphi(n_t, phase0=0.0):
    return jnp.asarray(phase0 + 2.0 * np.pi * np.arange(n_t) / n_t)


@pytest.mark.parametrize("phase0", [0.0, 0.7331, -2.5])
@pytest.mark.parametrize("n_t", [2 * LMAX + 1, 4 * (LMAX + 1), 257])
def test_uniform_synthesis_equals_direct_sum(fields, n_t, phase0):
    """The FFT synthesis must reproduce the direct phase sum exactly, at the
    Nyquist boundary (n_t = 2·lmax+1), on odd and even grids, and with a
    nonzero reference phase."""
    _, beam_ref, sky_alm, _ = fields
    vm = mmodes_from_sky(beam_ref, sky_alm, lmax=LMAX)
    dphi = _uniform_dphi(n_t, phase0)
    direct = tod_from_mmodes(vm, dphi)
    fast = tod_from_mmodes_uniform(vm, n_t, phase0=phase0)
    scale = float(jnp.max(jnp.abs(direct)))
    err = float(jnp.max(jnp.abs(fast - direct)))
    assert err < 1e-11 * scale, f"n_t={n_t} phase0={phase0}: {err:.3e} vs {scale:.3e}"


@pytest.mark.parametrize("normalize", [False, True])
def test_uniform_flag_matches_direct_forward(fields, normalize):
    _, beam_ref, sky_alm, ones_alm = fields
    dphi = _uniform_dphi(4 * (LMAX + 1), 0.51)
    kw = dict(lmax=LMAX, normalize=normalize, ones_alm=ones_alm if normalize else None)
    direct = driftscan_tod(beam_ref, sky_alm, dphi, **kw)
    fast = driftscan_tod(beam_ref, sky_alm, dphi, uniform=True, **kw)
    rel = float(jnp.max(jnp.abs(fast - direct)) / jnp.max(jnp.abs(direct)))
    assert rel < 1e-11, f"rel err {rel:.3e}"


@pytest.mark.parametrize("normalize", [False, True])
def test_uniform_adjoint_matches_direct_and_dot_tests(fields, rng, normalize):
    """The FFT adjoint must equal the direct-sum adjoint AND remain an exact
    transpose of the FFT forward (a compensating sign error in both FFT
    halves would pass the dot-test alone, so both checks run)."""
    _, beam_ref, sky_alm, ones_alm = fields
    n_t = 4 * (LMAX + 1)
    dphi = _uniform_dphi(n_t, -0.37)
    y = jnp.asarray(rng.standard_normal(n_t))
    kw = dict(lmax=LMAX, normalize=normalize, ones_alm=ones_alm if normalize else None)
    direct = driftscan_tod_adjoint(y, beam_ref, dphi, **kw)
    fast = driftscan_tod_adjoint(y, beam_ref, dphi, uniform=True, **kw)
    scale = float(jnp.max(jnp.abs(direct)))
    assert float(jnp.max(jnp.abs(fast - direct))) < 1e-11 * scale

    lhs = float(jnp.sum(driftscan_tod(beam_ref, sky_alm, dphi, uniform=True, **kw) * y))
    rhs = float(alm_dot(fast, sky_alm))
    assert abs(lhs - rhs) / abs(lhs) < 1e-11


def test_uniform_mmode_estimator_round_trip(fields):
    """mmodes_from_tod_uniform inverts the uniform synthesis exactly and
    agrees with the general DFT estimator on the same grid."""
    _, beam_ref, sky_alm, _ = fields
    n_t = 4 * (LMAX + 1)
    phase0 = 0.42
    dphi = _uniform_dphi(n_t, phase0)
    vm = mmodes_from_sky(beam_ref, sky_alm, lmax=LMAX)
    tod = tod_from_mmodes_uniform(vm, n_t, phase0=phase0)
    scale = float(jnp.max(jnp.abs(vm)))
    back = mmodes_from_tod_uniform(tod, lmax=LMAX, phase0=phase0)
    np.testing.assert_allclose(np.asarray(back), np.asarray(vm), atol=1e-11 * scale)
    general = mmodes_from_tod(tod, dphi, lmax=LMAX)
    np.testing.assert_allclose(
        np.asarray(back), np.asarray(general), atol=1e-11 * scale
    )


def test_uniform_rejects_bad_grids(fields):
    """Misuse must fail loudly, not silently return a wrong TOD: the Nyquist
    guard is static, the non-uniformity check fires whenever dphi is
    concrete (the only time it CAN fire)."""
    _, beam_ref, sky_alm, _ = fields
    n_t = 4 * (LMAX + 1)
    # Each grid below must trip its OWN guard: keep n_t long enough that the
    # Nyquist check does not mask the uniformity check.
    with pytest.raises(ValueError, match="2\\*lmax < n_time"):
        driftscan_tod(beam_ref, sky_alm, _uniform_dphi(2 * LMAX), lmax=LMAX, uniform=True)
    jittered = np.array(_uniform_dphi(n_t))  # copy: np.asarray view is read-only
    jittered[n_t // 3] += 0.05  # long, nearly-uniform, ONE bad sample
    with pytest.raises(ValueError, match="uniform grid"):
        driftscan_tod(beam_ref, sky_alm, jnp.asarray(jittered), lmax=LMAX, uniform=True)
    with pytest.raises(ValueError, match="uniform grid"):  # uniform but a HALF turn
        driftscan_tod(
            beam_ref, sky_alm,
            jnp.asarray(np.pi * np.arange(n_t) / n_t),
            lmax=LMAX, uniform=True,
        )
    with pytest.raises(ValueError, match="2\\*lmax < n_time"):
        mmodes_from_tod_uniform(jnp.zeros(2 * LMAX), lmax=LMAX)


def test_uniform_violation_is_poisoned_under_trace(fields):
    """Under jit the grid VALUES are unavailable, so the contract is enforced
    in pure JAX: a violated grid yields NaN, never a finite wrong TOD.

    This is the fix for a confirmed defect — the eager-only check was
    bypassed by ANY jit wrapping (even a compile-time-constant grid, since
    arithmetic inside a trace produces tracers), and a uniform HALF-turn
    grid then returned a silently 74%-wrong TOD."""
    _, beam_ref, sky_alm, _ = fields
    n_t = 4 * (LMAX + 1)
    good = _uniform_dphi(n_t, 0.1)

    @jax.jit
    def run(b, s, d):
        return driftscan_tod(b, s, d, lmax=LMAX, uniform=True)

    fast = run(beam_ref, sky_alm, good)
    direct = driftscan_tod(beam_ref, sky_alm, good, lmax=LMAX)
    assert float(jnp.max(jnp.abs(fast - direct))) < 1e-11 * float(
        jnp.max(jnp.abs(direct))
    )

    half = jnp.asarray(np.pi * np.arange(n_t) / n_t)  # uniform, HALF a turn
    irregular = jnp.asarray(np.sort(np.random.default_rng(0).uniform(0, 2 * np.pi, n_t)))
    for bad in (half, irregular):
        out = run(beam_ref, sky_alm, bad)
        assert bool(jnp.all(jnp.isnan(out))), "violated contract must poison, not lie"

    # the adjoint is exposed identically and must poison too
    y = jnp.asarray(np.linspace(-1.0, 1.0, n_t))
    adj = jax.jit(
        lambda t, b, d: driftscan_tod_adjoint(t, b, d, lmax=LMAX, uniform=True)
    )
    assert bool(jnp.all(jnp.isfinite(jnp.abs(adj(y, beam_ref, good)))))
    assert bool(jnp.all(jnp.isnan(jnp.abs(adj(y, beam_ref, half)))))


def test_poison_is_per_row_under_vmap(fields):
    """vmap must poison only the offending row — a scalar host-side check
    could not do this, which is why the guard is pure JAX."""
    _, beam_ref, sky_alm, _ = fields
    n_t = 4 * (LMAX + 1)
    grids = jnp.stack([_uniform_dphi(n_t), jnp.asarray(np.pi * np.arange(n_t) / n_t)])
    out = jax.vmap(
        lambda d: driftscan_tod(beam_ref, sky_alm, d, lmax=LMAX, uniform=True)
    )(grids)
    assert bool(jnp.all(jnp.isfinite(out[0]))) and bool(jnp.all(jnp.isnan(out[1])))


def test_uniform_dtype_matches_direct_path(fields):
    """The FFT path must not be silently less precise than the sum it
    reproduces: phase0's dtype has to enter the promotion (regression — it
    was dropped, so a float64 phase0 with complex64 modes lost the time
    axis), while the weak-typed default must not force float64."""
    vm = mmodes_from_sky(*fields[1:3], lmax=LMAX)
    n_t = 4 * (LMAX + 1)
    for m_dt, p_dt in [
        (jnp.complex64, jnp.float32), (jnp.complex64, jnp.float64),
        (jnp.complex128, jnp.float32), (jnp.complex128, jnp.float64),
    ]:
        c, p0 = vm.astype(m_dt), jnp.asarray(0.31, p_dt)
        dphi = (p0 + 2.0 * np.pi * jnp.arange(n_t, dtype=p_dt) / n_t).astype(p_dt)
        assert (
            tod_from_mmodes_uniform(c, n_t, phase0=p0).dtype
            == tod_from_mmodes(c, dphi).dtype
        ), (m_dt, p_dt)
    # weak-typed default (phase0=0.0) must not promote a complex64 input
    assert tod_from_mmodes_uniform(vm.astype(jnp.complex64), n_t).dtype == jnp.float32


def test_uniform_tolerance_boundaries():
    """Pin the tolerance in the repo that OWNS it, on both sides and in both
    dtypes. Before this, every call in limTOD's (x64) suite resolved to the
    old flat 1e-9 floor, so a mutant returning a constant survived the whole
    suite — only the downstream f32 repo could see the f32 branch."""
    from limtod_jax.driftscan import _uniform_tolerance, check_uniform_grid

    n_t = 4 * (LMAX + 1)
    for dtype in (np.float32, np.float64):
        tol = float(_uniform_tolerance(dtype, 2.0 * np.pi, np))
        base = 2.0 * np.pi * np.arange(n_t) / n_t

        def perturbed(amplitude):
            # index 0 must stay put: the checker measures deviation relative
            # to dphi[0], so perturbing it would double the apparent error
            j = np.zeros(n_t)
            j[1::2] = amplitude
            return (base + j).astype(dtype)

        check_uniform_grid(perturbed(0.5 * tol))  # inside the bound
        with pytest.raises(ValueError, match="uniform grid"):
            check_uniform_grid(perturbed(3.0 * tol))

    # the dtype scaling is the point: f32 must be ~1e5x looser than f64
    ratio = float(_uniform_tolerance(np.float32, 2.0 * np.pi, np)) / float(
        _uniform_tolerance(np.float64, 2.0 * np.pi, np)
    )
    assert 1e4 < ratio < 1e10, f"tolerance stopped tracking dtype (ratio {ratio:.1e})"

    # a genuinely uniform grid built in f32 (deg2rad of degrees) must pass —
    # its ~3e-7 rad representation error is exactly what the f32 bound exists
    # for, and it must NOT be checked against the f64 bound
    lst = np.float32(12.3) + np.arange(n_t, dtype=np.float32) * np.float32(360.0 / n_t)
    check_uniform_grid(np.deg2rad(lst - np.float32(12.3)))

    # non-floating dphi used to give a silent zero tolerance
    with pytest.raises(TypeError, match="floating-point"):
        check_uniform_grid(np.arange(n_t, dtype=np.int64))


def test_uniform_grad_is_the_phase0_direction(fields):
    """The uniform contract pins dphi to the one-parameter family
    Δ_0 + 2πt/n, so its Jacobian is a SINGLE column at index 0 — and that
    column must equal the direct path's row sum (the derivative w.r.t. a
    global LST shift). Documents the semantics and kills a
    stop_gradient/static-phase0 refactor, which is forward-identical and
    otherwise invisible."""
    _, beam_ref, sky_alm, _ = fields
    n_t = 4 * (LMAX + 1)
    w = jnp.asarray(np.linspace(0.5, 1.5, n_t))  # asymmetric: no Parseval cancel

    def loss(dphi, uniform):
        return jnp.sum(w * driftscan_tod(beam_ref, sky_alm, dphi, lmax=LMAX, uniform=uniform))

    dphi = _uniform_dphi(n_t, 0.31)
    g_fft = jax.grad(lambda d: loss(d, True))(dphi)
    g_dir = jax.grad(lambda d: loss(d, False))(dphi)

    assert np.nonzero(np.asarray(g_fft))[0].tolist() == [0], "expected a single column"
    total = float(jnp.sum(g_dir))
    assert abs(total) > 1e-6, "degenerate probe — the true derivative vanished"
    assert abs(float(g_fft[0]) - total) < 1e-10 * abs(total), (
        f"phase0 derivative {float(g_fft[0]):.6e} != direct row sum {total:.6e}"
    )


def test_check_uniform_grid_is_callable_inside_a_trace(fields):
    """The public eager checker must be safe to call from inside somebody
    else's jit trace (it computes its tolerance with numpy, not jnp — a jnp
    op on concrete scalars still returns a tracer while a trace is active,
    which made float(tol) raise ConcretizationTypeError)."""
    from limtod_jax.driftscan import check_uniform_grid

    concrete = np.asarray(_uniform_dphi(4 * (LMAX + 1)))

    @jax.jit
    def run(x):
        check_uniform_grid(concrete)  # concrete numpy, inside an active trace
        return x * 2.0

    assert float(run(jnp.asarray(1.5))) == 3.0


def test_uniform_operator_and_grad(fields, rng):
    """The operator's static flag routes to the FFT path and stays
    grad/jit-safe (map-making differentiates through this)."""
    import equinox as eqx

    beam_alm, _, sky_alm, _ = fields
    n_t = 4 * (LMAX + 1)
    lsts = np.linspace(0.0, 360.0, n_t, endpoint=False) + 12.0
    common = dict(lmax=LMAX, lst_ref_deg=12.0)
    op_fft = DriftScanMmode.from_pointing(
        beam_alm, lsts, LAT, AZ, EL, SELFROT, uniform_sampling=True, **common
    )
    op_sum = DriftScanMmode.from_pointing(
        beam_alm, lsts, LAT, AZ, EL, SELFROT, **common
    )
    assert op_fft.uniform_sampling and not op_sum.uniform_sampling
    a, b = op_fft(sky_alm), op_sum(sky_alm)
    assert float(jnp.max(jnp.abs(a - b))) < 1e-11 * float(jnp.max(jnp.abs(b)))

    y = jnp.asarray(rng.standard_normal(n_t))
    np.testing.assert_allclose(
        np.asarray(op_fft.adjoint(y)), np.asarray(op_sum.adjoint(y)), atol=1e-11
    )
    out = eqx.filter_jit(lambda o, s: o(s))(op_fft, sky_alm)
    np.testing.assert_allclose(np.asarray(out), np.asarray(a), rtol=1e-12)
    g = jax.grad(lambda s: jnp.sum(op_fft(s) ** 2), holomorphic=False)(sky_alm)
    assert bool(jnp.all(jnp.isfinite(jnp.abs(g))))


def test_uniform_operator_rejects_nonuniform_dphi(fields, rng):
    """Construction-time rejection: a long but irregular LST list (so the
    Nyquist guard passes and the uniformity check is what fires)."""
    beam_alm, _, _, _ = fields
    n_t = 4 * (LMAX + 1)
    irregular = 12.0 + np.sort(rng.uniform(0.0, 360.0, n_t))
    with pytest.raises(ValueError, match="uniform grid"):
        DriftScanMmode.from_pointing(
            beam_alm, irregular, LAT, AZ, EL, SELFROT,
            lmax=LMAX, lst_ref_deg=12.0, uniform_sampling=True,
        )


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


# ---------------------------------------------------------------- dl hoisting
def test_dl_plane_is_lst_independent_and_bit_exact(fields):
    """The precomputed Wigner-d plane must reproduce the rotation EXACTLY.

    The plane is built from the pointing alone. That is only legitimate
    because LST enters the zyz composition in the first-applied slot and so
    shifts psi, never the polar angle the plane is a function of — so one
    plane has to serve every LST, bit for bit. Anything less than bitwise
    would mean the hoist is an approximation, and it is not.
    """
    beam_alm = fields[0]
    dl = dl_plane_for_pointing(LAT, AZ, EL, SELFROT, lmax=LMAX)
    assert dl.shape == (LMAX + 1, 2 * LMAX + 1, 2 * LMAX + 1)
    for lst in LSTS:
        recomputed = beam_alm_at_reference(
            beam_alm, lst, LAT, AZ, EL, SELFROT, lmax=LMAX
        )
        hoisted = beam_alm_at_reference(
            beam_alm, lst, LAT, AZ, EL, SELFROT, lmax=LMAX, dl_array=dl
        )
        assert jnp.array_equal(recomputed, hoisted), f"lst={lst} not bit-exact"


def test_dl_plane_dtype_follows_the_angles(fields):
    """float32 angles give a float32 plane — the caller's choice must win.

    It used to be floored at the session default (float64 under x64), which
    silently doubled the largest array in the rotation. The Risbo recursion
    is float32-stable, so the halved plane still agrees to f32 roundoff.
    """
    beam_alm = fields[0]
    dl64 = dl_plane_for_pointing(LAT, AZ, EL, SELFROT, lmax=LMAX)
    dl32 = dl_plane_for_pointing(
        np.float32(LAT), np.float32(AZ), np.float32(EL), np.float32(SELFROT),
        lmax=LMAX,
    )
    assert dl32.dtype == jnp.float32
    assert dl32.nbytes * 2 == dl64.nbytes if dl64.dtype == jnp.float64 else True

    ref = beam_alm_at_reference(beam_alm, 137.5, LAT, AZ, EL, SELFROT, lmax=LMAX)
    got = beam_alm_at_reference(
        beam_alm, 137.5, LAT, AZ, EL, SELFROT, lmax=LMAX, dl_array=dl32
    )
    rel = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
    assert rel < 1e-4, f"float32 plane drifted by {rel:.2e}"


def test_dl_plane_is_differentiable_and_jittable(fields):
    """Hoisting must not cost the gradient w.r.t. the beam — the whole point
    is to speed up the case where the BEAM is the fitted parameter."""
    beam_alm = fields[0]
    dl = dl_plane_for_pointing(LAT, AZ, EL, SELFROT, lmax=LMAX)

    @jax.jit
    def loss(alm):
        out = beam_alm_at_reference(
            alm, 137.5, LAT, AZ, EL, SELFROT, lmax=LMAX, dl_array=dl
        )
        return jnp.sum(jnp.abs(out) ** 2)

    grad = jax.grad(loss)(beam_alm)
    assert grad.shape == beam_alm.shape
    assert bool(jnp.all(jnp.isfinite(jnp.abs(grad)))) and bool(jnp.any(grad != 0))


# --------------------------------------------- phase-matrix / scan crossover
@pytest.mark.parametrize("n_time", [64, 257])
def test_phase_matrix_and_scan_branches_agree(rng, n_time):
    """The size threshold must be a pure time/memory trade, never a numeric one.

    Both branches of tod_from_mmodes and _zeta are exercised at the same size
    by moving the threshold, so any divergence is the implementations
    disagreeing rather than the problem changing underneath them.
    """
    from limtod_jax import driftscan as ds

    mm = jnp.asarray(rng.standard_normal(LMAX + 1) + 1j * rng.standard_normal(LMAX + 1))
    dphi = jnp.asarray(np.linspace(0.0, 2 * np.pi, n_time, endpoint=False))
    tod = jnp.asarray(rng.standard_normal(n_time))

    def under(threshold, fn, *args):
        old = ds._PHASE_MATRIX_MAX
        ds._PHASE_MATRIX_MAX = threshold
        try:
            return fn(*args)
        finally:
            ds._PHASE_MATRIX_MAX = old

    matmul = under(10**9, ds.tod_from_mmodes, mm, dphi)
    scan = under(0, ds.tod_from_mmodes, mm, dphi)
    assert jnp.allclose(matmul, scan, rtol=1e-12, atol=1e-12)

    z_matmul = under(10**9, ds._zeta, tod, dphi, LMAX)
    z_map = under(0, ds._zeta, tod, dphi, LMAX)
    assert jnp.allclose(z_matmul, z_map, rtol=1e-12, atol=1e-12)


def test_phase_matrix_branch_keeps_the_gradient(rng):
    """Reverse-mode must survive the branch: the fast path is for inference."""
    from limtod_jax import driftscan as ds

    n_time = 128
    dphi = jnp.asarray(np.linspace(0.0, 2 * np.pi, n_time, endpoint=False))
    re = jnp.asarray(rng.standard_normal(LMAX + 1))
    im = jnp.asarray(rng.standard_normal(LMAX + 1))

    def loss(r, i):
        return jnp.sum(ds.tod_from_mmodes(r + 1j * i, dphi) ** 2)

    def under(threshold):
        old = ds._PHASE_MATRIX_MAX
        ds._PHASE_MATRIX_MAX = threshold
        try:
            return jax.grad(loss, argnums=(0, 1))(re, im)
        finally:
            ds._PHASE_MATRIX_MAX = old

    g_matmul, g_scan = under(10**9), under(0)
    for a, b in zip(g_matmul, g_scan):
        assert bool(jnp.all(jnp.isfinite(a)))
        assert jnp.allclose(a, b, rtol=1e-10, atol=1e-10)


def test_adjoint_dot_identity_holds_in_both_branches(rng):
    """<A x, y> == <x, A^T y> must hold whichever branch each side takes —
    the property map-making depends on, and the reason forward and adjoint
    switch on the SAME condition."""
    from limtod_jax import driftscan as ds

    n_time = 96
    dphi = jnp.asarray(np.linspace(0.0, 2 * np.pi, n_time, endpoint=False))
    mm = jnp.asarray(rng.standard_normal(LMAX + 1) + 1j * rng.standard_normal(LMAX + 1))
    y = jnp.asarray(rng.standard_normal(n_time))

    for threshold in (10**9, 0):
        old = ds._PHASE_MATRIX_MAX
        ds._PHASE_MATRIX_MAX = threshold
        try:
            lhs = jnp.sum(ds.tod_from_mmodes(mm, dphi) * y)
            zeta = ds._zeta(y, dphi, LMAX)
            weights = jnp.where(jnp.arange(LMAX + 1) > 0, 2.0, 1.0)
            # zeta_m = Σ_t y_t cos(mΔ) − i·Σ_t y_t sin(mΔ), so Im(zeta) already
            # carries the minus sign the forward synthesis applies to Im(V_m):
            # the two cancel and the pairing is a plain sum.
            rhs = jnp.sum(
                weights
                * (jnp.real(mm) * jnp.real(zeta) + jnp.imag(mm) * jnp.imag(zeta))
            )
            assert jnp.allclose(lhs, rhs, rtol=1e-10), f"threshold={threshold}"
        finally:
            ds._PHASE_MATRIX_MAX = old
