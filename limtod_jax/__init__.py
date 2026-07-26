"""
limtod_jax: pure-JAX port of limTOD's sky -> TOD machinery.

Differentiable, jit/vmap-safe reimplementation of the beam-rotation TOD
simulator (``limTOD.simulator``) in harmonic space, per the port contract
in replicant-telescope (formerly e-RHINO) ``docs/limtod-port-contract.md``:

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

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    # limtod_jax ships inside the limTOD distribution; keep one version.
    __version__ = _dist_version("limTOD")
except PackageNotFoundError:  # pragma: no cover — running from a bare checkout
    __version__ = "0.0.0.dev0"

try:
    import equinox as _eqx  # noqa: F401
    import jax as _jax  # noqa: F401
    import s2fft as _s2fft  # noqa: F401
except ImportError as exc:  # pragma: no cover — depends on install extras
    raise ImportError(
        "limtod_jax needs the jax extra of the limTOD distribution: "
        'pip install "limTOD[jax]" (Python >= 3.11).'
    ) from exc

from limtod_jax.alm import (
    alm_dot,
    alm_weights,
    lmax_of_nalm,
    nalm_of_lmax,
    packed_from_2d,
    packed_lm_arrays,
    packed_to_2d,
)
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
    horizon_masked_beam_alm,
    horizon_weights,
    mmodes_from_sky,
    mmodes_from_tod,
    tod_from_mmodes,
)
from limtod_jax.hpx import alm2map, map2alm_iter, map2alm_quad, ones_quadrature_alm
from limtod_jax.projection import generate_projection_rows

__all__ = [
    "__version__",
    "DriftScanMmode",
    "alm2map",
    "alm_dot",
    "alm_weights",
    "beam_alm_at_reference",
    "beam_weighted_sum",
    "driftscan_tod",
    "driftscan_tod_adjoint",
    "generate_projection_rows",
    "generate_tod_sky",
    "generate_tod_sky_adjoint",
    "horizon_masked_beam_alm",
    "horizon_weights",
    "lmax_of_nalm",
    "map2alm_iter",
    "map2alm_quad",
    "mmodes_from_sky",
    "mmodes_from_tod",
    "nalm_of_lmax",
    "ones_quadrature_alm",
    "packed_from_2d",
    "packed_lm_arrays",
    "packed_to_2d",
    "rotate_alm",
    "tod_from_mmodes",
    "zyz_of_pointing",
    "zyzyz2zyz",
]
