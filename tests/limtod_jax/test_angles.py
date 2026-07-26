"""Pointing -> ZYZ Euler angles against the scipy-based numpy limTOD oracle.

All comparisons reconstruct rotation MATRICES rather than comparing raw Euler
angles: at gimbal lock (theta ~ 0 or pi — e.g. zenith pointings) the zyz
split is degenerate and scipy makes an arbitrary choice; only the net
rotation is contractual (the downstream alm rotation depends on nothing
else). The grid includes every extreme corner the port contract mandates.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sim = pytest.importorskip("limTOD.simulator")

from limtod_jax.angles import zyz_of_pointing, zyzyz2zyz  # noqa: E402

# The scipy oracle warns at exact zenith corners; the degeneracy is precisely
# what the matrix-level comparison is designed to absorb.
pytestmark = pytest.mark.filterwarnings("ignore:Gimbal lock detected")

LATS = [53.24, 0.0, -90.0]
LSTS = [0.0, 179.9, 359.9]
AZELS = [(0.0, 90.0), (123.4, 90.0), (0.0, 5.0), (-42.3, 41.0)]
SELFROTS = [0.0, 30.0]


def _matrix_of_zyz(psi, theta, phi):
    """R = Rz(phi)·Ry(theta)·Rz(psi), angles in radians (limTOD convention)."""
    return (
        R.from_euler("z", float(phi))
        * R.from_euler("y", float(theta))
        * R.from_euler("z", float(psi))
    ).as_matrix()


@pytest.mark.parametrize("selfrot", SELFROTS)
@pytest.mark.parametrize("azel", AZELS)
@pytest.mark.parametrize("lst", LSTS)
@pytest.mark.parametrize("lat", LATS)
def test_zyz_of_pointing_matches_scipy(lat, lst, azel, selfrot):
    az, el = azel
    ours = zyz_of_pointing(lst, lat, az, el, selfrot)
    ref = sim.zyz_of_pointing(lst, lat, az, el, selfrot)
    np.testing.assert_allclose(
        _matrix_of_zyz(*ours), _matrix_of_zyz(*ref), atol=1e-12
    )


def test_generic_point_angle_equality():
    """Away from gimbal lock the raw angles themselves agree (mod 2π)."""
    args = (100.0, 53.24, -42.3, 41.0, 7.0)
    ours = zyz_of_pointing(*args)
    ref = sim.zyz_of_pointing(*args)
    for a, b in zip(ours, ref):
        assert abs(((float(a) - float(b) + np.pi) % (2 * np.pi)) - np.pi) < 1e-12


def test_zyzyz2zyz_matches_scipy(rng):
    for _ in range(5):
        angs = rng.uniform(-180.0, 360.0, size=5)
        ours = zyzyz2zyz(*angs)
        ref = sim.zyzyz2zyz(*angs)
        np.testing.assert_allclose(
            _matrix_of_zyz(*ours), _matrix_of_zyz(*ref), atol=1e-12
        )


def test_theta_range():
    for lat, lst, (az, el), sr in [
        (53.24, 0.0, (0.0, 90.0), 0.0),
        (-90.0, 179.9, (123.4, 90.0), 30.0),
        (0.0, 359.9, (-42.3, 5.0), 0.0),
    ]:
        _, theta, _ = zyz_of_pointing(lst, lat, az, el, sr)
        assert 0.0 <= float(theta) <= np.pi


def test_batched_arrays_match_scalar_loop():
    lst = jnp.asarray(LSTS)
    lat = 53.24
    az = jnp.asarray([0.0, 123.4, -42.3])
    el = jnp.asarray([90.0, 41.0, 5.0])
    sr = jnp.zeros(3)
    psi, theta, phi = zyz_of_pointing(lst, lat, az, el, sr)
    assert psi.shape == theta.shape == phi.shape == (3,)
    for i in range(3):
        one = zyz_of_pointing(float(lst[i]), lat, float(az[i]), float(el[i]), 0.0)
        np.testing.assert_allclose(
            [float(psi[i]), float(theta[i]), float(phi[i])],
            [float(x) for x in one],
            atol=1e-15,
        )


def test_vmap_matches_batch():
    lst = jnp.asarray([10.0, 200.0, 350.0])
    vm = jax.vmap(lambda t: zyz_of_pointing(t, 53.24, 12.0, 41.0, 0.0))(lst)
    direct = zyz_of_pointing(lst, 53.24, 12.0, 41.0, 0.0)
    for a, b in zip(vm, direct):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-15)


def test_jit_and_dtype():
    f = jax.jit(zyz_of_pointing)
    out1 = f(10.0, 53.24, 12.0, 41.0, 0.0)
    out2 = f(20.0, 53.24, 12.0, 41.0, 0.0)  # same shapes, no retrace error
    assert all(o.dtype == jnp.float64 for o in out1)
    assert [float(x) for x in out1] != [float(x) for x in out2]


def test_ncp_pointing_takes_theta0_gimbal_branch():
    """az=0, el=lat points the boresight at the NCP: the net rotation is a
    PURE z-rotation, so the theta~0 gimbal branch is taken. The old corner
    grid never produced this case (it needs the el=lat pairing), which let
    an injected sign flip in that branch survive the whole suite
    (review-confirmed). Matrix-level scipy comparison, several selfrots."""
    for lat in (53.24, -30.7):
        for lst in (0.0, 100.0):
            for sr in (0.0, 30.0):
                ours = zyz_of_pointing(lst, lat, 0.0, lat, sr)
                assert float(ours[1]) < 1e-7  # really the gimbal branch
                ref = sim.zyz_of_pointing(lst, lat, 0.0, lat, sr)
                np.testing.assert_allclose(
                    _matrix_of_zyz(*ours), _matrix_of_zyz(*ref), atol=1e-12
                )


def test_gimbal_threshold_boundary():
    """Boundary-validation methodology: bypass the dispatcher and check the
    zyz extraction reconstructs the input matrix on BOTH sides of the
    sin(theta) ~ _GIMBAL_TOL dispatch threshold, for theta near 0 AND near
    pi, with a nonzero z-rotation to give a sign error something to break."""
    from limtod_jax.angles import _GIMBAL_TOL, _zyz_from_matrix

    z_angle = 1.3439  # rad; arbitrary nonzero phi+psi
    for base in (0.0, np.pi):
        for factor in (0.3, 0.999, 1.001, 3.0, 1e6):
            theta = _GIMBAL_TOL * factor
            th = base + theta if base == 0.0 else base - theta
            r_in = _matrix_of_zyz(0.4321, th, z_angle - 0.4321)
            psi, th_out, phi = _zyz_from_matrix(jnp.asarray(r_in))
            # Below theta ~ sqrt(eps) ~ 1.5e-8 the tilt is unrecoverable from
            # r22 = cos(theta) (rounds to 1.0), so reconstruction error ~theta
            # is intrinsic; the bug class this guards (branch sign flips) is
            # O(1), eight orders above the loosest tolerance used here.
            np.testing.assert_allclose(
                _matrix_of_zyz(float(psi), float(th_out), float(phi)),
                r_in,
                atol=max(1e-12, 3.0 * theta),
                err_msg=f"base={base} factor={factor}",
            )
