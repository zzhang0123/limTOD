"""Full-Stokes suite: the drift-scan m-mode path against the generic paths.

The claim being tested is that polarisation costs the linear chain NOTHING
structural — that the whole thing is the Stokes-I chain applied to the packed
``T``/``E``/``B``/``V`` rows and contracted over them. Nothing here is trusted
on paper; each step of that claim is a numerical lock:

1. FOUNDATIONS — the three facts :mod:`limtod_jax.stokes` rests on.
   a. healpy's 3-row (T,E,B) ``rotate_alm`` IS the row-wise scalar rotation
      (bit-exact), so ``limtod_jax``'s scalar Wigner kernel needs no spin-2
      counterpart.
   b. the pixel Stokes dot ``Σ_p (I·I + Q·Q + U·U)`` equals the row-wise
      weighted alm dot against QUADRATURE alms, so the exactness contract of
      :mod:`limtod_jax.core` survives at spin 2.
   c. the drift phase ``e^{−imΔ}`` is the SAME for T, E and B (spin-independence
      of a z-rotation) — the reason the m-mode collapse works at all. The
      ``+imΔ`` candidate must lose catastrophically.

2. CROSS-CHECK, the acceptance criterion. For npol = 1, 3 and 4, with and
   without ``normalize``, on direct and FFT synthesis:
       driftscan m-mode TOD == generic JAX TOD == numpy limTOD generate_TOD_sky
   to float64 roundoff. Leg 3 matters twice over: it is an INDEPENDENT oracle
   (healpy's own polarised transforms, a different code path end to end), and
   it is the first one numpy limTOD's full-Stokes chain has ever had — see the
   header of ``tests/test_stokes_and_boundaries.py``.

3. STRUCTURE — adjoint transposition, jit/vmap/grad safety, the Stokes-I
   regression guarantee (npol=None and npol=1 must agree BIT for bit), and the
   guards that stop a frequency axis from being read as a Stokes axis.

Extreme corners are included deliberately (zenith gimbal el=90, low elevation,
lat in {0, ±90}, LST wrapping past 360°, Δ=0, a pure-E beam, a pure-B beam):
polarisation failure modes concentrate exactly where the scalar ones do.
"""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from limtod_jax.alm import alm_dot, packed_lm_arrays
from limtod_jax.angles import zyz_of_pointing, zyzyz2zyz
from limtod_jax.core import (
    beam_weighted_sum,
    generate_tod_sky,
    generate_tod_sky_adjoint,
    rotate_alm,
)
from limtod_jax.driftscan import (
    DriftScanMmode,
    beam_alm_at_reference,
    driftscan_tod,
    driftscan_tod_adjoint,
    horizon_beam_fraction,
    horizon_masked_beam_alm,
    horizon_truncated_beam,
    horizon_weights,
    mmodes_from_sky,
)
from limtod_jax.hpx import (
    alm2map,
    eb_to_qu,
    map2alm_iter,
    map2alm_quad,
    ones_quadrature_alm,
    qu_to_eb_quad,
)
from limtod_jax.stokes import STOKES_ALM_ROWS, validate_npol

pytestmark = pytest.mark.filterwarnings("ignore:Gimbal lock detected")

NSIDE, LMAX = 8, 23  # lmax = 3*nside-1: what generate_TOD_sky uses internally
NPIX = 12 * NSIDE**2
NPOLS = [1, 3, 4]

# One fixed drift-scan pointing plus the corner variants.
LAT, AZ, EL, SELFROT = -30.71, 41.5, 52.5, 10.0
LST_REF = 12.0
LSTS = np.array([12.0, 61.3, 155.0, 262.9, 359.9, 401.6])  # incl. Δ=0 and wrap


# --------------------------------------------------------------- helpers
def _stokes_maps(rng, npol):
    """Random Stokes maps in limTOD's map layout: (npix,) or (npol, npix)."""
    return rng.random(NPIX) if npol == 1 else rng.random((npol, NPIX))


def _beam_alm(beam_map):
    """Packed beam alms EXACTLY as generate_TOD_sky computes them internally.

    Note the 4-row split: V is spin-0 and must not ride along in the spin-2
    transform, so numpy limTOD analyses rows 0:3 and row 3 separately. Getting
    this wrong is invisible in Stokes I and wrong only in V.
    """
    beam_map = np.asarray(beam_map, dtype=np.float64)
    if beam_map.ndim == 1 or beam_map.shape[0] == 3:
        return np.atleast_2d(hp.map2alm(beam_map, lmax=LMAX))
    return np.vstack(
        (hp.map2alm(beam_map[:3], lmax=LMAX), hp.map2alm(beam_map[3], lmax=LMAX))
    )


def _quad_alm(sky_map):
    """Quadrature alms (npix/4π)·map2alm(·, iter=0), same 4-row split."""
    sky_map = np.asarray(sky_map, dtype=np.float64)
    q = NPIX / (4.0 * np.pi)
    if sky_map.ndim == 1 or sky_map.shape[0] == 3:
        return np.atleast_2d(q * hp.map2alm(sky_map, lmax=LMAX, iter=0))
    return np.vstack(
        (
            q * hp.map2alm(sky_map[:3], lmax=LMAX, iter=0),
            q * hp.map2alm(sky_map[3], lmax=LMAX, iter=0),
        )
    )


def _zyz_stack(lsts, lat=LAT, az=AZ, el=EL, sr=SELFROT):
    psi, theta, phi = zyz_of_pointing(
        jnp.asarray(lsts, dtype=float), lat, jnp.asarray(az, dtype=float),
        jnp.asarray(el, dtype=float), jnp.asarray(sr, dtype=float),
    )
    return jnp.stack([psi, theta, phi], axis=-1)


def _dphi(lsts=LSTS, ref=LST_REF):
    return jnp.deg2rad(jnp.asarray(lsts) - ref)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _full(v, n):
    """Scalar -> constant per-sample pointing array (the oracle takes lists)."""
    return np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, float)


@pytest.fixture(scope="module")
def fields(rng):
    """Per-npol (beam_map, sky_map, beam_alm, sky_quad_alm, beam_ref_alm)."""
    out = {}
    for npol in NPOLS:
        beam_map = _stokes_maps(rng, npol)
        sky_map = _stokes_maps(rng, npol)
        balm = jnp.asarray(_beam_alm(beam_map))
        salm = jnp.asarray(_quad_alm(sky_map))
        bref = beam_alm_at_reference(
            balm, LST_REF, LAT, AZ, EL, SELFROT, lmax=LMAX, npol=npol
        )
        out[npol] = (beam_map, sky_map, balm, salm, bref)
    return out


@pytest.fixture(scope="module")
def ones_alm():
    return ones_quadrature_alm(nside=NSIDE, lmax=LMAX)


@pytest.fixture(scope="module")
def oracle():
    """numpy limTOD full-Stokes generate_TOD_sky, truncation disabled."""
    sim = pytest.importorskip("limTOD.simulator")

    def _run(beam_map, sky_map, lsts, normalize=False,
             lat=LAT, az=AZ, el=EL, sr=SELFROT):
        n = len(np.atleast_1d(lsts))
        return sim.generate_TOD_sky(
            np.asarray(beam_map), np.asarray(sky_map),
            np.asarray(lsts, dtype=float), float(lat),
            _full(az, n), _full(el, n), _full(sr, n),
            normalize_beam=normalize, truncate_frac_thres=0.0,
        )

    return _run


# ============================================================ 1. FOUNDATIONS
class TestFoundations:
    """The three spin facts the whole extension rests on."""

    def test_3row_rotate_alm_is_rowwise_scalar(self, rng):
        """healpy's (T,E,B) rotation IS the scalar rotation, row by row.

        BIT-exact, not approximately: healpy applies one Wigner-D to each of
        the three rows because spin-weighted alms rotate with the same D^l as
        scalar ones. This is what makes limtod_jax's scalar kernel sufficient
        — there is no spin-2 rotation anywhere in the package.
        """
        alm3 = np.asarray(hp.map2alm(rng.standard_normal((3, NPIX)), lmax=LMAX))
        psi, theta, phi = 0.7, 1.1, -2.3

        joint = alm3.copy()
        hp.rotate_alm(joint, phi, theta, psi)
        rowwise = np.empty_like(alm3)
        for i in range(3):
            row = alm3[i].copy()
            hp.rotate_alm(row, phi, theta, psi)
            rowwise[i] = row

        assert np.array_equal(joint, rowwise), "healpy TEB rotation is not row-wise"
        # ... and the rotation is nontrivial, so this is not a vacuous pass.
        assert _rel(joint, alm3) > 0.1

    def test_jax_stack_rotation_matches_healpy_3row(self, rng):
        """limtod_jax.rotate_alm on a (3, n_alm) stack == healpy 3-row."""
        alm3 = np.asarray(hp.map2alm(rng.standard_normal((3, NPIX)), lmax=LMAX))
        psi, theta, phi = 0.7, 1.1, -2.3
        ref = alm3.copy()
        hp.rotate_alm(ref, phi, theta, psi)
        got = rotate_alm(jnp.asarray(alm3), psi, theta, phi, lmax=LMAX)
        assert got.shape == alm3.shape
        assert _rel(got, ref) < 1e-12

    def test_stokes_pixel_dot_equals_rowwise_alm_dot(self, rng):
        """Σ_p (I·I + Q·Q + U·U) == Σ_row alm_dot(row_b, row_s_quad).

        The exactness contract at spin 2. Also asserts that E and B genuinely
        contribute — a bug that dropped them would still pass on Stokes I.
        """
        beam_map = rng.standard_normal((3, NPIX))
        balm = np.asarray(hp.map2alm(beam_map, lmax=LMAX))
        band = hp.alm2map(balm, NSIDE, lmax=LMAX)  # exactly band-limited
        sky_map = rng.standard_normal((3, NPIX))
        salm = _quad_alm(sky_map)

        pixel = float(np.sum(band * sky_map))
        harmonic = float(
            jnp.sum(alm_dot(jnp.asarray(balm), jnp.asarray(salm), LMAX))
        )
        assert abs(pixel - harmonic) / abs(pixel) < 1e-12

        rows = np.asarray(alm_dot(jnp.asarray(balm), jnp.asarray(salm), LMAX))
        assert abs(rows[1]) > 0.05 * abs(rows[0]), "E row contributes nothing"
        assert abs(rows[2]) > 0.05 * abs(rows[0]), "B row contributes nothing"

    @pytest.mark.parametrize("row", [0, 1, 2])
    def test_drift_phase_is_spin_independent(self, rng, row):
        """B_row,lm(lst) = B_row,lm(ref)·e^{−imΔ} for EVERY Stokes row.

        The m-mode collapse exists because a z-rotation contributes
        δ_{m'm}·e^{−imα} whatever the spin. The +imΔ candidate must lose by
        orders of magnitude, or the test has no teeth.
        """
        alm3 = jnp.asarray(hp.map2alm(rng.standard_normal((3, NPIX)), lmax=LMAX))
        bref = beam_alm_at_reference(
            alm3, LST_REF, LAT, AZ, EL, SELFROT, lmax=LMAX, npol=3
        )
        lst1 = 87.0
        delta = np.deg2rad(lst1 - LST_REF)
        b1 = beam_alm_at_reference(
            alm3, lst1, LAT, AZ, EL, SELFROT, lmax=LMAX, npol=3
        )
        _, ms = packed_lm_arrays(LMAX)
        scale = float(jnp.max(jnp.abs(bref[row])))
        err_minus = float(
            jnp.max(jnp.abs(b1[row] - bref[row] * jnp.exp(-1j * ms * delta)))
        )
        err_plus = float(
            jnp.max(jnp.abs(b1[row] - bref[row] * jnp.exp(+1j * ms * delta)))
        )
        assert err_minus < 1e-12 * scale, f"phase law broken on row {row}"
        assert err_plus > 1e-2 * scale, "sign race degenerate — test lost its teeth"


# ======================================================= 2. THE CROSS-CHECK
class TestCrossCheckWithGeneric:
    """m-mode drift scan == generic JAX == numpy limTOD, for full Stokes."""

    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("npol", NPOLS)
    def test_mmode_matches_generic_jax(self, fields, ones_alm, npol, normalize):
        _, _, balm, salm, bref = fields[npol]
        generic = generate_tod_sky(
            balm, salm, _zyz_stack(LSTS), lmax=LMAX, npol=npol,
            normalize=normalize, ones_alm=ones_alm,
        )
        drift = driftscan_tod(
            bref, salm, _dphi(), lmax=LMAX, npol=npol,
            normalize=normalize, ones_alm=ones_alm,
        )
        assert drift.shape == generic.shape == (len(LSTS),)
        assert _rel(drift, generic) < 1e-12

    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("npol", NPOLS)
    def test_mmode_matches_numpy_oracle(
        self, fields, ones_alm, oracle, npol, normalize
    ):
        """The independent leg: healpy's own polarised transforms, end to end.

        This is also the first independent oracle numpy limTOD's full-Stokes
        chain has had — until now it was pinned only by physical invariants.
        """
        beam_map, sky_map, _, salm, bref = fields[npol]
        direct = oracle(beam_map, sky_map, LSTS, normalize=normalize)
        drift = driftscan_tod(
            bref, salm, _dphi(), lmax=LMAX, npol=npol,
            normalize=normalize, ones_alm=ones_alm,
        )
        assert _rel(drift, direct) < 1e-10

    @pytest.mark.parametrize("npol", NPOLS)
    def test_uniform_fft_synthesis_matches_direct(self, fields, ones_alm, npol):
        """The FFT fast path is untouched by polarisation: the Stokes rows are
        contracted BEFORE the synthesis, so it sees the same (lmax+1,) m-modes.
        """
        _, _, _, salm, bref = fields[npol]
        n_time = 2 * LMAX + 4
        lsts = LST_REF + np.linspace(0.0, 360.0, n_time, endpoint=False)
        dphi = jnp.deg2rad(jnp.asarray(lsts) - LST_REF)
        direct = driftscan_tod(bref, salm, dphi, lmax=LMAX, npol=npol)
        fast = driftscan_tod(bref, salm, dphi, lmax=LMAX, npol=npol, uniform=True)
        assert _rel(fast, direct) < 1e-12

    @pytest.mark.parametrize("npol", NPOLS)
    def test_operator_matches_oracle_end_to_end(self, rng, oracle, npol):
        """DriftScanMmode.from_pointing, npol INFERRED from the beam shape."""
        beam_map = _stokes_maps(rng, npol)
        sky_map = _stokes_maps(rng, npol)
        balm = jnp.asarray(_beam_alm(beam_map))
        salm = jnp.asarray(_quad_alm(sky_map))
        op = DriftScanMmode.from_pointing(
            balm, LSTS, LAT, AZ, EL, SELFROT, lmax=LMAX, lst_ref_deg=LST_REF
        )
        assert op.npol == npol, "npol was not inferred from beam_alm.ndim"
        direct = oracle(beam_map, sky_map, LSTS)
        assert _rel(op(salm), direct) < 1e-10

    @pytest.mark.parametrize(
        "lat,az,el", [(0.0, 0.0, 90.0), (-90.0, 0.0, 90.0), (53.24, 123.4, 5.0)]
    )
    def test_corner_pointings_full_stokes(self, rng, ones_alm, oracle, lat, az, el):
        """Zenith gimbal, pole latitude and a near-horizon pointing, npol=3.

        The zenith/pole corners are where the ZYZ chain degenerates; if the
        Stokes rows were being rotated with anything but the shared plane,
        this is where it would show.
        """
        beam_map = _stokes_maps(rng, 3)
        sky_map = _stokes_maps(rng, 3)
        balm = jnp.asarray(_beam_alm(beam_map))
        salm = jnp.asarray(_quad_alm(sky_map))
        bref = beam_alm_at_reference(
            balm, LST_REF, lat, az, el, SELFROT, lmax=LMAX, npol=3
        )
        direct = oracle(beam_map, sky_map, LSTS, lat=lat, az=az, el=el)
        drift = driftscan_tod(bref, salm, _dphi(), lmax=LMAX, npol=3)
        generic = generate_tod_sky(
            balm, salm, _zyz_stack(LSTS, lat=lat, az=az, el=el),
            lmax=LMAX, npol=3,
        )
        assert _rel(drift, direct) < 1e-10
        assert _rel(drift, generic) < 1e-12

    @pytest.mark.parametrize("pure", ["E", "B"])
    def test_e_dominated_and_b_dominated_beams(self, rng, oracle, pure):
        """A beam whose power is (almost) all E, or almost all B, and I = 0.

        These are the cases a wrong E/B handling passes silently on a generic
        random field: a swap or a sign flip is invisible when both rows are
        present and comparable, and a beam with no Stokes I would hide an
        accidental Stokes-I-only contraction entirely.

        "Almost": the beam the ORACLE sees is a map, and its internal
        ``hp.map2alm`` of that map is not the inverse of the ``alm2map`` that
        built it — HEALPix pixelisation leaks ~1.5% of E into B. The lock is
        against those internal alms (the standing exactness recipe); the
        fixture only asserts the intended row still dominates ~40x, which is
        what gives the test its teeth.
        """
        alm3 = np.zeros((3, hp.Alm.getsize(LMAX)), dtype=complex)
        row = 1 if pure == "E" else 2
        alm3[row] = hp.map2alm(rng.standard_normal(NPIX), lmax=LMAX)
        beam_map = hp.alm2map(alm3, NSIDE, lmax=LMAX)

        balm = jnp.asarray(_beam_alm(beam_map))  # what the oracle uses internally
        amp = np.max(np.abs(np.asarray(balm)), axis=1)
        other = 1 if row == 2 else 2
        assert amp[0] == 0.0, "Stokes I should be identically zero here"
        assert amp[row] > 20.0 * amp[other], f"{pure}-domination lost: {amp}"

        sky_map = _stokes_maps(rng, 3)
        salm = jnp.asarray(_quad_alm(sky_map))
        bref = beam_alm_at_reference(
            balm, LST_REF, LAT, AZ, EL, SELFROT, lmax=LMAX, npol=3
        )
        direct = oracle(beam_map, sky_map, LSTS)
        drift = driftscan_tod(bref, salm, _dphi(), lmax=LMAX, npol=3)
        assert np.max(np.abs(direct)) > 0, "degenerate fixture"
        assert _rel(drift, direct) < 1e-10


# =========================================================== 3. STRUCTURE
class TestStokesIRegression:
    """What the unpolarised path is guaranteed, stated exactly.

    The guarantee is about the CODE PATH, not about ``npol=1``: a 2-D ``flm``
    still takes the original ``"mn,m,n,n->m"`` contraction, so every existing
    unpolarised result is unchanged bit for bit (the rest of the suite pins
    those values). A ``npol=1`` STACK takes the ``"mn,m,n,pn->pm"``
    contraction, which is the same arithmetic in a different association
    order and therefore agrees only to roundoff — ~1 ulp, and data-dependent,
    so it is asserted as roundoff and not as equality.
    """

    def test_scalar_rotation_path_is_unchanged(self, rng):
        """The 2-D path still reproduces healpy exactly as it did before."""
        alm = jnp.asarray(hp.map2alm(rng.standard_normal(NPIX), lmax=LMAX))
        psi, theta, phi = 0.7, 1.1, -2.3
        ref = np.asarray(alm).copy()
        hp.rotate_alm(ref, phi, theta, psi)
        got = rotate_alm(alm, psi, theta, phi, lmax=LMAX)
        assert got.ndim == 1
        assert _rel(got, ref) < 1e-12

    def test_npol_none_vs_npol_1_agree_to_roundoff(self, fields, ones_alm):
        _, _, balm, salm, bref = fields[1]
        for kwargs in ({}, {"normalize": True, "ones_alm": ones_alm}):
            old = generate_tod_sky(
                balm[0], salm[0], _zyz_stack(LSTS), lmax=LMAX, **kwargs
            )
            new = generate_tod_sky(
                balm, salm, _zyz_stack(LSTS), lmax=LMAX, npol=1, **kwargs
            )
            assert _rel(new, old) < 1e-14

            # The drift path does not rotate, so there npol=1 IS bit-exact:
            # the only added op is a sum over a length-1 axis.
            old_d = driftscan_tod(bref[0], salm[0], _dphi(), lmax=LMAX, **kwargs)
            new_d = driftscan_tod(bref, salm, _dphi(), lmax=LMAX, npol=1, **kwargs)
            assert np.array_equal(np.asarray(old_d), np.asarray(new_d))

    def test_beam_weighted_sum_stokes_i_unchanged(self, fields):
        _, _, balm, salm, _ = fields[1]
        assert np.array_equal(
            np.asarray(beam_weighted_sum(balm[0], salm[0])),
            np.asarray(beam_weighted_sum(balm, salm, npol=1)),
        )


class TestAdjoint:
    @pytest.mark.parametrize("uniform", [False, True])
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("npol", NPOLS)
    def test_dot_test(self, fields, ones_alm, rng, npol, normalize, uniform):
        """⟨forward(x), y⟩ == ⟨x, adjoint(y)⟩_w, summed over Stokes rows."""
        _, _, _, salm, bref = fields[npol]
        n_time = 2 * LMAX + 4
        lsts = LST_REF + np.linspace(0.0, 360.0, n_time, endpoint=False)
        dphi = jnp.deg2rad(jnp.asarray(lsts) - LST_REF)
        kw = {"lmax": LMAX, "npol": npol, "normalize": normalize,
              "ones_alm": ones_alm, "uniform": uniform}
        y = jnp.asarray(rng.standard_normal(n_time))
        fwd = driftscan_tod(bref, salm, dphi, **kw)
        adj = driftscan_tod_adjoint(y, bref, dphi, **kw)
        assert adj.shape == salm.shape
        lhs = float(jnp.sum(fwd * y))
        rhs = float(jnp.sum(alm_dot(salm, adj, LMAX)))
        assert abs(lhs - rhs) / abs(lhs) < 1e-10

    @pytest.mark.parametrize("npol", NPOLS)
    def test_matches_generic_adjoint(self, fields, npol):
        """The m-mode adjoint equals the generic accumulation of rotated beams."""
        _, _, balm, _, bref = fields[npol]
        y = jnp.asarray(np.linspace(-1.0, 2.0, len(LSTS)))
        drift = driftscan_tod_adjoint(y, bref, _dphi(), lmax=LMAX, npol=npol)
        generic = generate_tod_sky_adjoint(
            y, balm, _zyz_stack(LSTS), lmax=LMAX, npol=npol
        )
        assert _rel(drift, generic) < 1e-12


class TestJitVmapGrad:
    @pytest.mark.parametrize("npol", NPOLS)
    def test_jit_and_grad(self, fields, npol):
        _, _, _, salm, bref = fields[npol]

        @jax.jit
        def loss(sky):
            return jnp.sum(driftscan_tod(bref, sky, _dphi(), lmax=LMAX, npol=npol) ** 2)

        g = jax.grad(loss)(salm)
        assert g.shape == salm.shape
        assert jnp.all(jnp.isfinite(jnp.real(g)))

    def test_vmap_over_frequency(self, fields):
        """A frequency axis is batched with vmap, OUTSIDE the Stokes axis."""
        _, _, _, salm, bref = fields[3]
        skies = jnp.stack([salm, 2.0 * salm, -0.5 * salm])  # (3, 3, n_alm)
        out = jax.vmap(
            lambda s: driftscan_tod(bref, s, _dphi(), lmax=LMAX, npol=3)
        )(skies)
        single = driftscan_tod(bref, salm, _dphi(), lmax=LMAX, npol=3)
        assert out.shape == (3, len(LSTS))
        assert _rel(out[1], 2.0 * single) < 1e-12
        assert _rel(out[2], -0.5 * single) < 1e-12


class TestMmodesArePolarised:
    @pytest.mark.parametrize("npol", [3, 4])
    def test_mmodes_sum_the_stokes_rows(self, fields, npol):
        """Ṽ_m is ONE series, the sum of the per-row series — and the Q/U rows
        are not silently dropped."""
        _, _, _, salm, bref = fields[npol]
        total = mmodes_from_sky(bref, salm, lmax=LMAX, npol=npol)
        rows = jnp.stack([
            mmodes_from_sky(bref[i], salm[i], lmax=LMAX) for i in range(npol)
        ])
        assert total.shape == (LMAX + 1,)
        assert _rel(total, jnp.sum(rows, axis=0)) < 1e-12
        i_only = mmodes_from_sky(bref[0], salm[0], lmax=LMAX)
        assert _rel(total, i_only) > 1e-3, "polarised m-modes collapsed to Stokes I"

    @pytest.mark.parametrize("npol", NPOLS)
    def test_mmodes_are_the_tod_fourier_coefficients(self, fields, npol):
        """FFT lock (MmodeNote eqn 15) still holds with the Stokes rows in."""
        _, _, _, salm, bref = fields[npol]
        n_time = 4 * (LMAX + 1)
        dphi = jnp.asarray(2.0 * np.pi * np.arange(n_time) / n_time)
        tod = driftscan_tod(bref, salm, dphi, lmax=LMAX, npol=npol)
        vm = mmodes_from_sky(bref, salm, lmax=LMAX, npol=npol)
        fft = jnp.fft.rfft(tod)[: LMAX + 1] / n_time
        assert _rel(fft, vm) < 1e-12


# ------------------------------------------------------------ horizon utils
class TestHorizonPolarisation:
    def test_truncated_beam_tapers_every_row_and_splits_on_stokes_i(self, rng):
        """The taper is a real scalar -> broadcasts over Stokes; f_sky is the
        Stokes-I solid-angle split, NOT a per-row quantity."""
        beam_map = jnp.asarray(rng.random((3, NPIX)))
        cut, frac = horizon_truncated_beam(beam_map, nside=NSIDE, npol=3)
        assert cut.shape == beam_map.shape
        assert np.ndim(np.asarray(frac)) == 0, "one beam has one sky fraction"
        cut_i, frac_i = horizon_truncated_beam(beam_map[0], nside=NSIDE)
        assert _rel(cut[0], cut_i) < 1e-15
        assert abs(float(frac) - float(frac_i)) < 1e-15
        # every row carries the SAME taper — the mask is a real scalar
        taper = np.asarray(cut[0]) / np.asarray(beam_map[0])
        for row in (1, 2):
            assert _rel(cut[row], jnp.asarray(taper) * beam_map[row]) < 1e-14
        # ... and it actually removed something (below-horizon pixels)
        assert float(jnp.sum(jnp.abs(cut[2]))) < float(jnp.sum(jnp.abs(beam_map[2])))

    def test_beam_fraction_uses_stokes_i_row(self, rng):
        balm = jnp.asarray(hp.map2alm(rng.random((3, NPIX)), lmax=LMAX))
        f_pol = horizon_beam_fraction(balm, AZ, EL, SELFROT, nside=NSIDE,
                                      lmax=LMAX, npol=3)
        f_i = horizon_beam_fraction(balm[0], AZ, EL, SELFROT, nside=NSIDE,
                                    lmax=LMAX)
        assert np.ndim(np.asarray(f_pol)) == 0
        assert abs(float(f_pol) - float(f_i)) < 1e-15

    @pytest.mark.parametrize("apod_deg", [0.0, 3.0])
    @pytest.mark.parametrize("npol", [3, 4])
    def test_masked_beam_alm_matches_healpy(self, rng, npol, apod_deg):
        """Polarised horizon mask == the same chain run entirely in healpy."""
        beam_map = _stokes_maps(rng, npol)
        balm = jnp.asarray(_beam_alm(beam_map))
        az, el, sr = 30.0, 12.0, 0.0
        ours = np.asarray(
            horizon_masked_beam_alm(
                balm, az, el, sr, nside=NSIDE, lmax=LMAX,
                apod_deg=apod_deg, npol=npol,
            )
        )
        # healpy oracle, locked slot convention (limtod_jax.wigner):
        # jax rotate_alm(x, psi, theta, phi) == hp.rotate_alm(x, phi, theta, psi)
        psi, theta, phi = (
            float(a) for a in zyzyz2zyz(0.0, 0.0, -az, el - 90.0, sr)
        )
        a_h = np.array(balm)
        hp.rotate_alm(a_h if npol == 3 else a_h[:3], phi, theta, psi)
        if npol == 4:
            v = a_h[3].copy()
            hp.rotate_alm(v, phi, theta, psi)
            a_h[3] = v
        w = horizon_weights(NSIDE, apod_deg)
        if npol == 3:
            m_h = hp.alm2map(a_h, NSIDE, lmax=LMAX) * w
            ref = np.asarray(hp.map2alm(m_h, lmax=LMAX, iter=3))
        else:
            m3 = hp.alm2map(a_h[:3], NSIDE, lmax=LMAX) * w
            mv = hp.alm2map(a_h[3], NSIDE, lmax=LMAX) * w
            ref = np.vstack((hp.map2alm(m3, lmax=LMAX, iter=3),
                             hp.map2alm(mv, lmax=LMAX, iter=3)))
        hp.rotate_alm(ref if npol == 3 else ref[:3], -psi, -theta, -phi)
        if npol == 4:
            v = ref[3].copy()
            hp.rotate_alm(v, -psi, -theta, -phi)
            ref[3] = v
        assert _rel(ours, ref) < 1e-9

    def test_masked_polarised_drift_equals_masked_generic(self, rng, oracle):
        """The masked polarised beam drops into BOTH paths and they agree."""
        beam_map = _stokes_maps(rng, 3)
        sky_map = _stokes_maps(rng, 3)
        balm = jnp.asarray(_beam_alm(beam_map))
        salm = jnp.asarray(_quad_alm(sky_map))
        az, el, sr = 30.0, 12.0, 0.0
        masked = horizon_masked_beam_alm(
            balm, az, el, sr, nside=NSIDE, lmax=LMAX, apod_deg=2.0, npol=3
        )
        bref = beam_alm_at_reference(
            masked, LST_REF, LAT, az, el, sr, lmax=LMAX, npol=3
        )
        generic = generate_tod_sky(
            masked, salm, _zyz_stack(LSTS, lat=LAT, az=az, el=el, sr=sr),
            lmax=LMAX, npol=3,
        )
        drift = driftscan_tod(bref, salm, _dphi(), lmax=LMAX, npol=3)
        assert _rel(drift, generic) < 1e-12
        # and the mask actually did something to the polarised rows
        assert _rel(masked[1], balm[1]) > 1e-3

    def test_operator_horizon_mask_polarised(self, rng):
        """from_pointing(horizon_mask=True) on a polarised beam."""
        balm = jnp.asarray(_beam_alm(_stokes_maps(rng, 3)))
        op = DriftScanMmode.from_pointing(
            balm, LSTS, LAT, 30.0, 12.0, 0.0, lmax=LMAX,
            lst_ref_deg=LST_REF, nside=NSIDE, horizon_mask=True, apod_deg=2.0,
        )
        assert op.npol == 3
        assert op.beam_ref_alm.shape == balm.shape


class TestSpin2Transforms:
    """The (Q,U) <-> (E,B) pair, against healpy — and the backend trap.

    These exist because the obvious implementation is silently wrong: s2fft's
    on-the-fly recursion drops whole multipoles at spin != 0 on HEALPix
    (log|d| = -inf where a Wigner-d vanishes exactly on a ring, then
    0*inf -> nan -> nansum). The precompute kernel with recursion="risbo" is
    the one that reproduces healpy. See the warning in limtod_jax.hpx.
    """

    def test_eb_to_qu_matches_healpy(self, rng):
        alm3 = np.asarray(hp.map2alm(rng.standard_normal((3, NPIX)), lmax=LMAX))
        ref = hp.alm2map(alm3, NSIDE, lmax=LMAX)
        q, u = eb_to_qu(
            jnp.asarray(alm3[1]), jnp.asarray(alm3[2]), nside=NSIDE, lmax=LMAX
        )
        scale = max(np.max(np.abs(ref[1])), np.max(np.abs(ref[2])))
        assert np.max(np.abs(np.asarray(q) - ref[1])) < 1e-12 * scale
        assert np.max(np.abs(np.asarray(u) - ref[2])) < 1e-12 * scale

    def test_qu_to_eb_quad_matches_healpy(self, rng):
        q, u = rng.standard_normal(NPIX), rng.standard_normal(NPIX)
        e, b = qu_to_eb_quad(jnp.asarray(q), jnp.asarray(u),
                             nside=NSIDE, lmax=LMAX)
        ref = (NPIX / (4 * np.pi)) * np.asarray(
            hp.map2alm(np.vstack([np.zeros(NPIX), q, u]), lmax=LMAX, iter=0)
        )
        scale = max(np.max(np.abs(ref[1])), np.max(np.abs(ref[2])))
        assert np.max(np.abs(np.asarray(e) - ref[1])) < 1e-12 * scale
        assert np.max(np.abs(np.asarray(b) - ref[2])) < 1e-12 * scale

    @pytest.mark.parametrize("npol", [3, 4])
    def test_map2alm_iter_matches_healpy(self, rng, npol):
        maps = _stokes_maps(rng, npol)
        ours = np.asarray(
            map2alm_iter(jnp.asarray(maps), nside=NSIDE, lmax=LMAX,
                         iterations=3, npol=npol)
        )
        ref = _beam_alm(maps)  # hp.map2alm(..., iter=3), split for V
        assert _rel(ours, ref) < 1e-9

    def test_roundtrip_alm2map_map2alm_quad(self, rng):
        """alm2map/map2alm_quad are a matched pair at spin 2 as at spin 0."""
        alm3 = jnp.asarray(hp.map2alm(rng.standard_normal((3, NPIX)), lmax=LMAX))
        maps = alm2map(alm3, nside=NSIDE, lmax=LMAX, npol=3)
        ref = hp.alm2map(np.asarray(alm3), NSIDE, lmax=LMAX)
        assert _rel(maps, ref) < 1e-12

    def test_spin2_exactness_contract(self, rng):
        """Σ_p (Q_b Q_s + U_b U_s) == ⟨E_b, Ẽ_s⟩ + ⟨B_b, B̃_s⟩ through the
        JAX transforms — the quadrature identity, at spin 2."""
        alm3 = jnp.asarray(hp.map2alm(rng.standard_normal((3, NPIX)), lmax=LMAX))
        band = alm2map(alm3, nside=NSIDE, lmax=LMAX, npol=3)  # band-limited beam
        sky = jnp.asarray(rng.standard_normal((3, NPIX)))
        squad = map2alm_quad(sky, nside=NSIDE, lmax=LMAX, npol=3)
        pixel = float(jnp.sum(band[1] * sky[1] + band[2] * sky[2]))
        harmonic = float(
            alm_dot(alm3[1], squad[1], LMAX) + alm_dot(alm3[2], squad[2], LMAX)
        )
        assert abs(pixel - harmonic) / abs(pixel) < 1e-11

    def test_onthefly_backend_really_is_broken(self):
        """REGRESSION PIN, not a test of our code: if this ever starts
        passing, s2fft has fixed its HEALPix spin recursion and hpx.py can
        drop the precompute kernel (and its O(nside·lmax²) memory).

        Until then, nobody should "simplify" eb_to_qu back to
        s2fft.inverse_jax — it is wrong by ~4-6 %, silently, on whole
        multipoles.
        """
        import s2fft

        from limtod_jax.alm import packed_to_2d

        rng = np.random.default_rng(5)
        alm3 = np.asarray(hp.map2alm(rng.standard_normal((3, NPIX)), lmax=LMAX))
        ref = hp.alm2map(alm3, NSIDE, lmax=LMAX)
        a_p2 = -(
            packed_to_2d(jnp.asarray(alm3[1]), LMAX)
            + 1j * packed_to_2d(jnp.asarray(alm3[2]), LMAX)
        )
        otf = np.asarray(
            s2fft.inverse_jax(a_p2, LMAX + 1, spin=2, nside=NSIDE,
                              sampling="healpix", reality=False)
        )
        scale = max(np.max(np.abs(ref[1])), np.max(np.abs(ref[2])))
        err = np.max(np.abs(otf.real - ref[1])) / scale
        assert err > 1e-3, (
            f"s2fft's on-the-fly spin-2 HEALPix recursion now agrees to "
            f"{err:.2e} — recheck limtod_jax.hpx, the precompute workaround "
            f"may no longer be needed"
        )
        # ... and ours does not have that problem
        q, _ = eb_to_qu(jnp.asarray(alm3[1]), jnp.asarray(alm3[2]),
                        nside=NSIDE, lmax=LMAX)
        assert np.max(np.abs(np.asarray(q) - ref[1])) < 1e-12 * scale

    @pytest.mark.parametrize(
        "nside,lmax,msg", [(1, 7, "nside >= 2"), (8, 7, r"lmax\+1 >= 2\*nside")]
    )
    def test_spin2_grid_limits_are_stated(self, nside, lmax, msg):
        """s2fft fails these with a bare AssertionError; we must not."""
        n = 12 * nside**2
        with pytest.raises(ValueError, match=msg):
            eb_to_qu(jnp.zeros(hp.Alm.getsize(lmax), dtype=complex),
                     jnp.zeros(hp.Alm.getsize(lmax), dtype=complex),
                     nside=nside, lmax=lmax)
        assert n > 0  # silence the unused-variable lint


# ------------------------------------------------------------------ guards
class TestGuards:
    @pytest.mark.parametrize("bad", [0, 2, 5, 64, -1])
    def test_npol_must_be_1_3_or_4(self, bad):
        with pytest.raises(ValueError, match="npol must be"):
            validate_npol(bad)

    def test_npol_rejects_non_int(self):
        with pytest.raises(TypeError):
            validate_npol(3.0)

    def test_shape_must_carry_the_stokes_axis(self, fields):
        _, _, balm, salm, _ = fields[3]
        with pytest.raises(ValueError, match=r"\(\.\.\., npol, n_alm\)"):
            generate_tod_sky(balm[0], salm[0], _zyz_stack(LSTS), lmax=LMAX, npol=3)

    def test_beam_and_sky_npol_must_match(self, fields):
        _, _, balm3, _, _ = fields[3]
        _, _, _, salm4, _ = fields[4]
        with pytest.raises(ValueError, match="disagree on the Stokes axis|npol"):
            generate_tod_sky(balm3, salm4, _zyz_stack(LSTS), lmax=LMAX, npol=3)

    def test_frequency_stack_is_rejected_by_the_operator(self, rng):
        """A (n_freq, n_alm) beam with n_freq not in {1,3,4} must fail LOUDLY
        rather than be contracted into one TOD."""
        stack = jnp.asarray(
            np.stack([hp.map2alm(rng.random(NPIX), lmax=LMAX) for _ in range(5)])
        )
        with pytest.raises(ValueError, match="1, 3 or 4 rows"):
            DriftScanMmode.from_pointing(
                stack, LSTS, LAT, AZ, EL, SELFROT, lmax=LMAX
            )

    def test_operator_rejects_3d_beam(self, fields):
        _, _, balm, _, _ = fields[3]
        with pytest.raises(ValueError, match="jax.vmap"):
            DriftScanMmode.from_pointing(
                balm[None], LSTS, LAT, AZ, EL, SELFROT, lmax=LMAX, npol=3
            )

    def test_stokes_row_tables_agree(self):
        for npol, rows in STOKES_ALM_ROWS.items():
            assert len(rows) == npol


class TestSilentFailureSurface:
    """Regression pins for the guard holes found in review.

    Every one of these previously returned a finite, plausible, WRONG number
    (or poisoned the process). The standard this codebase holds itself to is
    that a violated contract raises or NaNs — never returns something a reader
    would believe.
    """

    def test_frequency_stack_is_not_contracted_by_beam_weighted_sum(self, rng):
        """A 5-row FREQUENCY stack must not be summed into one number.

        `match_npol` used to compare the two arrays against EACH OTHER, so a
        pair of 5-row stacks agreed with itself perfectly and npol=1/3/4 all
        returned the same silent sum-over-frequency.
        """
        stack = jnp.asarray(
            np.stack([hp.map2alm(rng.random(NPIX), lmax=LMAX) for _ in range(5)])
        )
        for npol in NPOLS:
            with pytest.raises(ValueError, match="leading Stokes axis"):
                beam_weighted_sum(stack, stack, npol=npol)

    def test_beam_weighted_sum_rejects_1d_under_npol(self, fields):
        """Used to raise a bare IndexError from indexing [-2] on a 1-D array."""
        _, _, balm, salm, _ = fields[3]
        with pytest.raises(ValueError, match="leading Stokes axis"):
            beam_weighted_sum(balm[0], salm[0], npol=3)

    def test_npol_4_on_a_3_row_array_raises(self, fields):
        """JAX CLAMPS out-of-bounds indices, so ``alm[3]`` on a 3-row array is
        ``alm[2]`` — the transforms used to fabricate a V row from B."""
        _, _, balm, _, _ = fields[3]
        maps = jnp.asarray(_stokes_maps(np.random.default_rng(1), 3))
        with pytest.raises(ValueError, match="leading Stokes axis"):
            alm2map(balm, nside=NSIDE, lmax=LMAX, npol=4)
        with pytest.raises(ValueError, match="leading Stokes axis"):
            map2alm_quad(maps, nside=NSIDE, lmax=LMAX, npol=4)
        with pytest.raises(ValueError, match="leading Stokes axis"):
            alm2map(balm, nside=NSIDE, lmax=LMAX, npol=1)

    def test_generic_path_rejects_2d_alms_without_npol(self, fields):
        """`rotate_flm_2d` now accepts a leading axis, so the generic path has
        to say no itself — otherwise a (3, n_alm) Stokes stack passed WITHOUT
        npol returns three separate TODs instead of raising, and a frequency
        stack becomes an undocumented batch."""
        _, _, balm, salm, _ = fields[3]
        with pytest.raises(ValueError, match="jax.vmap"):
            generate_tod_sky(balm, salm, _zyz_stack(LSTS), lmax=LMAX)
        with pytest.raises(ValueError, match="jax.vmap"):
            generate_tod_sky_adjoint(
                jnp.zeros(len(LSTS)), balm, _zyz_stack(LSTS), lmax=LMAX
            )

    def test_polarised_beam_with_extra_leading_axis_rejected(self, fields):
        _, _, balm, salm, _ = fields[3]
        with pytest.raises(ValueError, match="jax.vmap"):
            generate_tod_sky(
                balm[None], salm[None], _zyz_stack(LSTS), lmax=LMAX, npol=3
            )

    def test_polarised_ones_alm_is_rejected(self, fields, ones_alm):
        """ones_alm is always ONE row (the normalizer is the Stokes-I pixel
        sum). A stack broadcasts instead of failing and silently gives the TOD
        an extra axis."""
        _, _, balm, salm, bref = fields[3]
        bad = jnp.stack([ones_alm] * 3)
        for call in (
            lambda: generate_tod_sky(balm, salm, _zyz_stack(LSTS), lmax=LMAX,
                                     npol=3, normalize=True, ones_alm=bad),
            lambda: beam_weighted_sum(balm, salm, npol=3, normalize=True,
                                      ones_alm=bad),
            lambda: driftscan_tod(bref, salm, _dphi(), lmax=LMAX, npol=3,
                                  normalize=True, ones_alm=bad),
            lambda: DriftScanMmode(beam_ref_alm=bref, dphi=_dphi(), lmax=LMAX,
                                   normalize=True, ones_alm=bad, npol=3),
        ):
            with pytest.raises(ValueError, match="must be 1-D"):
                call()

    def test_spin2_kernel_cache_survives_jit_first(self, rng):
        """THE ORDER-DEPENDENT ONE. The kernel cache used to hold a jax array;
        built first inside a jit trace it stored a TRACER, and every later call
        outside that trace died with UnexpectedTracerError for the rest of the
        process. Caching numpy makes the order irrelevant.

        This test only has teeth in a fresh cache state, so it exercises a
        (nside, lmax) combination no other test uses.
        """
        ns, lm = 4, 15
        n = 12 * ns**2
        alm3 = jnp.asarray(
            hp.map2alm(rng.standard_normal((3, n)), lmax=lm, iter=1)
        )
        jitted = jax.jit(
            lambda a: eb_to_qu(a[1], a[2], nside=ns, lmax=lm)[0]
        )
        first = jitted(alm3)                                   # cold, under jit
        eager = eb_to_qu(alm3[1], alm3[2], nside=ns, lmax=lm)[0]  # then eager
        again = jax.jit(
            lambda a: eb_to_qu(a[1], a[2], nside=ns, lmax=lm)[0] * 2.0
        )(alm3)                                                 # a DIFFERENT jit
        assert _rel(first, eager) < 1e-14
        assert _rel(again, 2.0 * np.asarray(eager)) < 1e-14

    def test_polarised_mask_is_jittable(self, rng):
        """The documented polarised-mask entry point, under jit."""
        balm = jnp.asarray(_beam_alm(_stokes_maps(rng, 3)))
        f = jax.jit(
            lambda a: horizon_masked_beam_alm(
                a, 30.0, 12.0, 0.0, nside=NSIDE, lmax=LMAX, npol=3
            )
        )
        assert _rel(
            f(balm),
            horizon_masked_beam_alm(balm, 30.0, 12.0, 0.0, nside=NSIDE,
                                    lmax=LMAX, npol=3),
        ) < 1e-13
