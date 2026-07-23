"""Phase-2: native projection-matrix builder vs numpy generate_sky2sys_projection.

Rows are the rotated beam sampled at selected pixels — pure synthesis on
both sides, so agreement is at roundoff (oracle runs truncate_frac_thres=0).
"""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from limtod_jax.angles import zyz_of_pointing
from limtod_jax.hpx import ones_quadrature_alm
from limtod_jax.projection import generate_projection_rows

sim = pytest.importorskip("limTOD.simulator")

pytestmark = pytest.mark.filterwarnings("ignore:Gimbal lock detected")

NSIDE, LMAX, LAT = 8, 23, 53.24


@pytest.fixture(scope="module")
def pointings():
    lst = np.asarray([0.0, 100.0, 179.9, 359.9, 250.0, 40.0])
    az = np.asarray([0.0, 45.0, 123.4, -42.3, 80.0, 0.0])
    el = np.asarray([90.0, 60.0, 5.0, 41.0, 30.0, 90.0])
    sr = np.asarray([0.0, 30.0, 0.0, -15.0, 0.0, 7.0])
    return lst, az, el, sr


@pytest.fixture(scope="module")
def beam():
    beam_map = np.random.default_rng(3).random(hp.nside2npix(NSIDE))
    return beam_map, jnp.asarray(hp.map2alm(beam_map, lmax=LMAX))


@pytest.fixture(scope="module")
def zyz(pointings):
    lst, az, el, sr = pointings
    psi, theta, phi = zyz_of_pointing(
        jnp.asarray(lst), LAT, jnp.asarray(az), jnp.asarray(el), jnp.asarray(sr)
    )
    return jnp.stack([psi, theta, phi], axis=-1)


PIXEL_INDICES = np.arange(3, hp.nside2npix(8), 7)


@pytest.mark.parametrize("normalize", [False, True])
def test_matches_numpy_oracle(beam, pointings, zyz, normalize):
    beam_map, beam_alm = beam
    lst, az, el, sr = pointings
    direct = sim.generate_sky2sys_projection(
        beam_map,
        lst,
        LAT,
        az,
        el,
        sr,
        PIXEL_INDICES,
        normalize_beam=normalize,
        nside_target=NSIDE,
        truncate_frac_thres=0.0,
    )
    native = generate_projection_rows(
        beam_alm,
        zyz,
        jnp.asarray(PIXEL_INDICES),
        lmax=LMAX,
        nside=NSIDE,
        normalize=normalize,
        ones_alm=ones_quadrature_alm(nside=NSIDE, lmax=LMAX) if normalize else None,
    )
    assert native.shape == direct.shape == (len(lst), len(PIXEL_INDICES))
    rel = np.max(np.abs(np.asarray(native) - direct)) / np.max(np.abs(direct))
    assert rel < 1e-6, f"rel err {rel:.3e}"  # expected ~1e-13


def test_vmap_over_frequency(beam, zyz, rng):
    _, beam_alm = beam
    scales = jnp.asarray(rng.random(2)) + 0.5
    beams = scales[:, None] * beam_alm[None, :]
    idx = jnp.asarray(PIXEL_INDICES)

    vm = jax.vmap(
        lambda b: generate_projection_rows(b, zyz, idx, lmax=LMAX, nside=NSIDE)
    )(beams)
    loop = jnp.stack(
        [
            generate_projection_rows(beams[f], zyz, idx, lmax=LMAX, nside=NSIDE)
            for f in range(2)
        ]
    )
    np.testing.assert_allclose(np.asarray(vm), np.asarray(loop), rtol=1e-12)


def test_grad_and_jit(beam, zyz):
    _, beam_alm = beam
    idx = jnp.asarray(PIXEL_INDICES)

    def loss(re):
        b = beam_alm.at[2].set(re + 1j * jnp.imag(beam_alm[2]))
        return jnp.sum(generate_projection_rows(b, zyz, idx, lmax=LMAX, nside=NSIDE))

    g = float(jax.grad(loss)(jnp.real(beam_alm[2])))
    assert np.isfinite(g) and abs(g) > 1e-12

    n_traces = 0

    def counted(b):
        nonlocal n_traces
        n_traces += 1
        return generate_projection_rows(b, zyz, idx, lmax=LMAX, nside=NSIDE)

    f = jax.jit(counted)
    _ = f(beam_alm)
    _ = f(2.0 * beam_alm)
    assert n_traces == 1
