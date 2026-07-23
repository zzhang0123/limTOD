"""Pointing -> ZYZ Euler angles, pure JAX.

Port of ``limTOD.simulator.zyzyz2zyz`` / ``zyz_of_pointing`` (scipy-based).
Convention (matching numpy limTOD):

* inputs in DEGREES, outputs in RADIANS (the unit ``hp.rotate_alm`` expects);
* "zyz" means extrinsic ``R = Rz(phi) · Ry(theta) · Rz(psi)`` — psi applied
  first — returned in the order ``(psi, theta, phi)``;
* "zyzyz" means ``R = Rz(chi)·Ry(delta)·Rz(gamma)·Ry(beta)·Rz(alpha)``.

At gimbal lock (theta ~ 0 or pi, e.g. zenith pointings) the zyz split is
degenerate; this implementation picks ``psi = 0`` (scipy may split
differently). Only the NET rotation is contractual — downstream alm
rotation depends on nothing else — so tests compare rotation matrices.

All functions broadcast over leading batch dimensions and are
jit/vmap-safe (no value-dependent Python control flow).
"""

from __future__ import annotations

import jax.numpy as jnp

# Below this, sin(theta) is treated as exactly zero (gimbal lock). Products
# like cos(phi)·sin(theta) keep an accurate RATIO down to ~1e-300, so the
# threshold only needs to catch true zeros from el = ±90 corners.
_GIMBAL_TOL = 1e-14


def _rz(a: jnp.ndarray) -> jnp.ndarray:
    ca, sa = jnp.cos(a), jnp.sin(a)
    z, o = jnp.zeros_like(ca), jnp.ones_like(ca)
    return jnp.stack(
        [
            jnp.stack([ca, -sa, z], axis=-1),
            jnp.stack([sa, ca, z], axis=-1),
            jnp.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )


def _ry(b: jnp.ndarray) -> jnp.ndarray:
    cb, sb = jnp.cos(b), jnp.sin(b)
    z, o = jnp.zeros_like(cb), jnp.ones_like(cb)
    return jnp.stack(
        [
            jnp.stack([cb, z, sb], axis=-1),
            jnp.stack([z, o, z], axis=-1),
            jnp.stack([-sb, z, cb], axis=-1),
        ],
        axis=-2,
    )


def _zyz_from_matrix(r: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract (psi, theta, phi) with R = Rz(phi)·Ry(theta)·Rz(psi).

    For that composition: R[2,2] = cosθ; R[0,2] = cosφ·sinθ,
    R[1,2] = sinφ·sinθ; R[2,0] = −sinθ·cosψ, R[2,1] = sinθ·sinψ.
    """
    cos_theta = jnp.clip(r[..., 2, 2], -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    sin_theta_sq = r[..., 0, 2] ** 2 + r[..., 1, 2] ** 2

    psi_reg = jnp.arctan2(r[..., 2, 1], -r[..., 2, 0])
    phi_reg = jnp.arctan2(r[..., 1, 2], r[..., 0, 2])

    # theta ~ 0: R = Rz(phi+psi);  theta ~ pi: R = Rz(phi−psi)·Ry(pi).
    # Put the whole z-rotation into phi (psi := 0) in both cases.
    phi_g0 = jnp.arctan2(r[..., 1, 0], r[..., 0, 0])
    phi_gpi = jnp.arctan2(-r[..., 1, 0], -r[..., 0, 0])
    phi_gim = jnp.where(cos_theta > 0.0, phi_g0, phi_gpi)

    gimbal = sin_theta_sq < _GIMBAL_TOL**2
    psi = jnp.where(gimbal, jnp.zeros_like(psi_reg), psi_reg)
    phi = jnp.where(gimbal, phi_gim, phi_reg)
    return psi, theta, phi


def zyzyz2zyz(alpha, beta, gamma, delta, chi, output_degrees: bool = False):
    """Collapse a "zyzyz" rotation to effective "zyz" angles.

    ``R = Rz(chi)·Ry(delta)·Rz(gamma)·Ry(beta)·Rz(alpha)`` with inputs in
    degrees; returns ``(psi, theta, phi)`` in radians (degrees when
    ``output_degrees=True``, a static flag). Mirrors
    ``limTOD.simulator.zyzyz2zyz``.
    """
    a, b, g, d, c = (jnp.deg2rad(jnp.asarray(x)) for x in (alpha, beta, gamma, delta, chi))
    r = _rz(c) @ _ry(d) @ _rz(g) @ _ry(b) @ _rz(a)
    psi, theta, phi = _zyz_from_matrix(r)
    if output_degrees:
        return jnp.rad2deg(psi), jnp.rad2deg(theta), jnp.rad2deg(phi)
    return psi, theta, phi


def zyz_of_pointing(lst_deg, lat_deg, az_deg, el_deg, selfrot_deg):
    """Pointing parameters -> effective "zyz" angles (radians).

    Mirrors ``limTOD.simulator.zyz_of_pointing``: azimuth is east-of-north
    positive (hence the sign flip), latitude/elevation enter as colatitudes.
    All inputs in degrees; scalars or broadcastable arrays.
    """
    return zyzyz2zyz(
        lst_deg,
        90.0 - jnp.asarray(lat_deg),
        -jnp.asarray(az_deg),
        jnp.asarray(el_deg) - 90.0,
        selfrot_deg,
    )
