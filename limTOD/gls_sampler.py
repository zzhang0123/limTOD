"""Posterior map samples for :class:`limTOD.gls_mapmaking.GLS_mapmaking`,
drawn by hydra-tod.

limTOD supplies the GLS solution ``p_hat`` (its own IRLS, with the Gaussian
prior inside the iteration), the per-TOD operators, the inverse noise
covariances ``N⁻¹`` (dense, or the diagonal for uncorrelated noise) and the
prior precision. ``hydra_tod.map_sampler.sample_map_posterior`` (Zhang et
al. 2026, RASTI, rzag024) draws the samples. It evaluates ``Σ`` at
``p_hat``. It builds each TOD's normal-equation terms with
``hydra_tod.linear_sampler.params_space_oper_and_data`` and draws each
sample with ``hydra_tod.linear_sampler.sample_p_v2``. The draws follow
``N(p_hat, (A + S⁻¹)⁻¹)``, the estimate and covariance ``GLS_mapmaking``
reports. This module checks the arguments, imports hydra-tod and packages
the draws; it draws no random numbers itself.

Posterior sampling loads hydra-tod and its MPI dependency on demand.
"""

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

_INSTALL_HINT = (
    "GLS_mapmaking.posterior_samples needs hydra_tod.map_sampler, which "
    "could not be imported. Install or update the sampler with "
    "pip install mpi4py 'git+https://github.com/zzhang0123/hydra-tod'. "
    "The chained error identifies any missing dependency."
)


def hydra_map_sampler() -> ModuleType:
    """Import ``hydra_tod.map_sampler``, or raise with install advice.

    Preserve the caller's RNG state across imports, including older
    hydra-tod versions which also load JAX."""
    state = np.random.get_state()
    try:
        return importlib.import_module("hydra_tod.map_sampler")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    finally:
        np.random.set_state(state)


@dataclass(frozen=True, eq=False)
class GLSPosteriorSamples:
    """Draws from the GLS map posterior.

    Attributes
    ----------
    sky : ndarray
        ``(n_draws, num_pixels)``, or ``(n_draws, npol, num_pixels)`` for a
        polarised beam.
    others : list of ndarray or None
        Per TOD, ``(n_draws, n_params_others)`` draws of the other-Tsys
        parameters; ``None`` without other-Tsys operators.
    params : ndarray
        ``(n_draws, n_params)`` joint draws ``[sky | others_0 | others_1 ...]``.
    mean : ndarray
        ``(n_params,)`` GLS estimate ``p_hat``, the centre of the draws and
        the estimate ``GLS_mapmaking.__call__`` returns.
    precision : ndarray
        ``(n_params, n_params)`` posterior precision ``A + S⁻¹`` at ``p_hat``
        that the draws follow; its inverse is the ``posterior_cov`` of
        ``return_full_cov=True``.
    """

    sky: np.ndarray
    others: Optional[List[np.ndarray]]
    params: np.ndarray
    mean: np.ndarray
    precision: np.ndarray


def check_n_draws(n_draws: int) -> None:
    if (isinstance(n_draws, bool)
            or not isinstance(n_draws, (int, np.integer)) or n_draws < 1):
        raise ValueError(f"n_draws must be a positive integer, got {n_draws!r}")


SeedLike = Union[None, int, np.integer, np.random.Generator]


def resolve_seed(seed: SeedLike) -> SeedLike:
    """Check ``seed`` before the solve. hydra-tod takes it as its ``rng``:
    an integer or a ``Generator`` supplies every random number, ``None``
    draws from NumPy's global state."""
    if seed is None or isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(
            "seed must be None, an integer or a numpy.random.Generator, "
            f"got {type(seed).__name__}"
        )
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return int(seed)


def _hydra_prior(S_inv: np.ndarray) -> Optional[np.ndarray]:
    """limTOD's dense prior precision in the form hydra-tod takes: none, the
    diagonal (1D), or the full matrix (2D, which hydra-tod requires to be
    positive definite). The prior mean is not passed: it enters the draws
    only through ``p_hat``."""
    if not np.any(S_inv):
        return None
    diag = np.diag(S_inv)
    if np.count_nonzero(S_inv - np.diag(diag)) == 0:
        return diag.copy()
    return S_inv


def draw_gls_posterior(
    map_sampler: ModuleType,
    *,
    d_list: Sequence[np.ndarray],
    ops: Sequence[np.ndarray],
    Ninv_list: Sequence[np.ndarray],
    mu_list: Sequence[Union[float, np.ndarray]],
    p_hat: np.ndarray,
    S_inv: np.ndarray,
    noise_model: str,
    n_draws: int,
    seed: SeedLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(n_draws, n_params)`` draws and the posterior precision, from
    ``hydra_tod.map_sampler.sample_map_posterior``."""
    return map_sampler.sample_map_posterior(
        list(d_list), list(ops), list(Ninv_list), p_hat,
        mu_list=list(mu_list),
        prior_cov_inv=_hydra_prior(S_inv),
        multiplicative=noise_model != "additive",
        n_samples=n_draws, rng=seed,
    )
