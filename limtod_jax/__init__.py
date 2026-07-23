"""
limtod_jax: pure-JAX port of limTOD's sky -> TOD machinery.

Differentiable, jit/vmap-safe reimplementation of the beam-rotation TOD
simulator (``limTOD.simulator``) in harmonic space, per the port contract
in e-RHINO ``docs/limtod-port-contract.md``:

* pointing -> ZYZ Euler angles      (:mod:`limtod_jax.angles`)
* Wigner rotation of packed alms    (:mod:`limtod_jax.wigner`, :mod:`limtod_jax.core`)
* beam-weighted harmonic dot        (:mod:`limtod_jax.core`)
* full TOD chain + exact adjoint    (:mod:`limtod_jax.core`)
* projection-matrix builder         (:mod:`limtod_jax.projection`)
* HEALPix map <-> alm inside JAX    (:mod:`limtod_jax.hpx`)

Conventions (matching numpy limTOD):

* Degrees in public pointing APIs, radians internally.
* HEALPix RING ordering; healpy packed alm layout (m >= 0, real fields).
* ``lmax``/``nside`` are static Python ints; angles/alms/maps are traced.
* healpy is never imported here — it remains the test-suite oracle only.
"""

__version__ = "0.1.0"

from limtod_jax.alm import (
    alm_dot,
    lmax_of_nalm,
    nalm_of_lmax,
    packed_from_2d,
    packed_to_2d,
)
from limtod_jax.angles import zyz_of_pointing, zyzyz2zyz
from limtod_jax.core import (
    beam_weighted_sum,
    generate_tod_sky,
    generate_tod_sky_adjoint,
    rotate_alm,
)
from limtod_jax.hpx import alm2map, map2alm_quad, ones_quadrature_alm
from limtod_jax.projection import generate_projection_rows

__all__ = [
    "__version__",
    "alm2map",
    "alm_dot",
    "beam_weighted_sum",
    "generate_projection_rows",
    "generate_tod_sky",
    "generate_tod_sky_adjoint",
    "lmax_of_nalm",
    "map2alm_quad",
    "nalm_of_lmax",
    "ones_quadrature_alm",
    "packed_from_2d",
    "packed_to_2d",
    "rotate_alm",
    "zyz_of_pointing",
    "zyzyz2zyz",
]
