"""Posterior map samples for :class:`limTOD.gls_mapmaking.GLS_mapmaking`,
drawn with hydra-tod's linear sampler.

limTOD supplies the GLS solution ``p_hat`` (its own IRLS, with the Gaussian
prior inside the iteration) and the per-TOD inverse noise covariances
``N⁻¹``. hydra-tod supplies the sampling step of its sky/Tsys Gibbs
conditional (Zhang et al. 2026, RASTI, rzag024):

* ``hydra_tod.linear_sampler.params_space_oper_and_data``: per TOD, the
  normal-equation matrix ``U^T Σ⁻¹ U`` and the noise projection
  ``U^T Σ^{-1/2}`` at ``p_hat``, with
  ``Σ⁻¹ = diag(1/(U p_hat + mu)) N⁻¹ diag(1/(U p_hat + mu))``. Under the
  additive model limTOD forms the same products itself with ``Σ = N``.
* ``hydra_tod.linear_sampler.sample_p_v2``: one Gaussian constrained
  realisation per draw,
  ``(A + S⁻¹) x = (A + S⁻¹) p_hat + U^T Σ^{-1/2} ω + S^{-1/2} ω'``.

Here ``A = sum_i U_i^T Σ_i⁻¹ U_i``, ``S⁻¹`` is the prior precision,
``Σ^{-1/2}`` is any factor ``F`` with ``F F^T = Σ⁻¹`` and ``ω``, ``ω'`` are
standard normal. Each ``x`` is a draw from ``N(p_hat, (A + S⁻¹)⁻¹)``, the
covariance ``GLS_mapmaking`` reports. Under the multiplicative model this
Gaussian, with ``Σ`` held at ``p_hat``, approximates the posterior; see
``GLS_mapmaking.posterior_samples``.

Gains and the 1/f and white-noise parameters stay fixed. Temperatures
modelled by the other-Tsys operator (e.g. a receiver temperature) are drawn
with the sky; temperatures passed as ``known_injection_group`` stay fixed.

hydra-tod is not on PyPI and its ``linear_sampler`` imports ``mpi4py``; both
are imported only when samples are requested.
"""

import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

# Eigenvalues of N⁻¹ down to -1e-10 x the largest are round-off of a
# semi-definite matrix and are clipped to zero; below that N⁻¹ is rejected.
_PSD_RTOL = 1e-10

_INSTALL_HINT = (
    "GLS_mapmaking.posterior_samples needs hydra_tod.linear_sampler, which "
    "could not be imported. It requires hydra-tod and mpi4py; the chained "
    "error names the missing one. hydra-tod is not on PyPI: install it from "
    "https://github.com/zzhang0123/hydra-tod, and mpi4py from PyPI."
)


def hydra_linear_sampler() -> ModuleType:
    """Import ``hydra_tod.linear_sampler``, or raise with install advice.

    The import pulls in jax, whose own import draws from NumPy's global RNG;
    the caller's global state is restored afterwards."""
    state = np.random.get_state()
    try:
        return importlib.import_module("hydra_tod.linear_sampler")
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
        ``(n_params, n_params)`` posterior precision ``A + S⁻¹`` at ``p_hat``;
        its inverse is the ``posterior_cov`` of ``return_full_cov=True``.
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


def resolve_seed(seed: SeedLike) -> Optional[int]:
    """The integer seed for NumPy's legacy global RNG, which hydra-tod draws
    from: ``None`` stays ``None``; a ``Generator`` contributes one draw."""
    if seed is None:
        return None
    if isinstance(seed, np.random.Generator):
        return int(seed.integers(0, 2 ** 32))
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(
            "seed must be None, an integer or a numpy.random.Generator, "
            f"got {type(seed).__name__}"
        )
    if not 0 <= seed < 2 ** 32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    return int(seed)


def _hydra_prior(
    S_inv: np.ndarray, prior_mean: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """The prior in a form ``sample_p_v2`` accepts: none, a diagonal (1D),
    or a positive-definite matrix (2D), whose Cholesky factor it draws with."""
    if not np.any(S_inv):
        return None, None
    diag = np.diag(S_inv)
    if np.count_nonzero(S_inv - np.diag(diag)) == 0:
        return diag.copy(), prior_mean
    try:
        np.linalg.cholesky(S_inv)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "posterior_samples: a non-diagonal prior must be positive "
            "definite over every parameter, because hydra-tod's sample_p_v2 "
            "draws the prior perturbation through cholesky(S^-1). Give the "
            "sky a prior as well (Tsky_prior_inv_cov_diag), or pass the "
            "other-Tsys priors as 1D diagonals."
        ) from exc
    return S_inv, prior_mean


def _inv_cov_sqrt(N_inv: np.ndarray, index: int) -> np.ndarray:
    """A square root ``L`` with ``L L^T = N⁻¹``, the ``Ninv_sqrt`` hydra-tod
    expects. Cholesky when ``N⁻¹`` is positive definite; otherwise (flagged
    samples zeroed out of ``N⁻¹`` leave it semi-definite) ``V Λ^{1/2}`` from
    its eigendecomposition."""
    try:
        return np.linalg.cholesky(N_inv)
    except np.linalg.LinAlgError:
        pass
    lam, V = np.linalg.eigh(N_inv)
    # an all-zero N⁻¹ (a fully flagged TOD) passes and gives a zero root
    if lam[0] < -_PSD_RTOL * max(lam[-1], 0.0):
        raise ValueError(
            f"posterior_samples: the inverse noise covariance of TOD {index} "
            f"is not positive semi-definite (eigenvalues {lam[0]:.3g} to "
            f"{lam[-1]:.3g})"
        )
    return V * np.sqrt(np.clip(lam, 0.0, None))


def _data_terms(
    linear_sampler: ModuleType,
    d_list: Sequence[np.ndarray],
    ops: Sequence[np.ndarray],
    Ninv_list: Sequence[np.ndarray],
    mu_list: Sequence[Union[float, np.ndarray]],
    p_hat: np.ndarray,
    noise_model: str,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """``sum_i U_i^T Σ_i⁻¹ U_i``, ``sum_i U_i^T Σ_i⁻¹ (d_i - mu_i)`` and each
    TOD's ``U_i^T Σ_i^{-1/2}``, all at ``p_hat`` (no prior)."""
    A = np.zeros((p_hat.size, p_hat.size))
    b = np.zeros(p_hat.size)
    projections = []
    for i, (d_i, U_i, Ninv_i, mu_i) in enumerate(zip(d_list, ops, Ninv_list, mu_list)):
        Ninv_sqrt = _inv_cov_sqrt(Ninv_i, i)
        if noise_model == "additive":
            # hydra-tod's operator is multiplicative-only; with Σ = N it
            # reduces to these three products.
            aux = U_i.T @ Ninv_i
            A_i, b_i, T_i = aux @ U_i, aux @ (d_i - mu_i), U_i.T @ Ninv_sqrt
        else:
            A_i, b_i, T_i = linear_sampler.params_space_oper_and_data(
                d_i, U_i, p_hat, Ninv_i, mu=mu_i, Ninv_sqrt=Ninv_sqrt
            )
        A += A_i
        b += b_i
        projections.append(T_i)
    return A, b, projections


@contextmanager
def _global_seed(seed: Optional[int]) -> Iterator[None]:
    """hydra-tod draws from NumPy's global RNG. An integer seed runs the
    draws on a freshly seeded global stream and restores the caller's state
    afterwards; ``None`` uses, and advances, the current global stream."""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def draw_gls_posterior(
    linear_sampler: ModuleType,
    *,
    d_list: Sequence[np.ndarray],
    ops: Sequence[np.ndarray],
    Ninv_list: Sequence[np.ndarray],
    mu_list: Sequence[Union[float, np.ndarray]],
    p_hat: np.ndarray,
    S_inv: np.ndarray,
    prior_mean: np.ndarray,
    noise_model: str,
    n_draws: int,
    seed: Optional[int],
) -> np.ndarray:
    """``(n_draws, n_params)`` draws from ``N(p_hat, (A + S⁻¹)⁻¹)``, one
    ``sample_p_v2`` call per draw.

    The right-hand side handed to ``sample_p_v2`` is
    ``(A + S⁻¹) p_hat - S⁻¹ m`` rather than ``b = U^T Σ⁻¹ (d - mu)``, so the
    draws are centred on ``p_hat`` itself. The two agree at the IRLS fixed
    point; when the IRLS stopped short of it, ``b`` would centre the draws
    on one more IRLS step instead of on the estimate ``__call__`` reports.
    """
    prior_cov_inv, prior_mean_h = _hydra_prior(S_inv, prior_mean)
    A, _, projections = _data_terms(
        linear_sampler, d_list, ops, Ninv_list, mu_list, p_hat, noise_model
    )
    b_centred = A @ p_hat + S_inv @ (p_hat - prior_mean)
    draws = np.empty((n_draws, p_hat.size))
    with _global_seed(seed):
        for k in range(n_draws):
            # the white-noise term U^T Σ^{-1/2} ω, drawn per TOD as in
            # hydra-tod's params_space_oper_and_data_list
            wn = sum(T @ np.random.normal(0.0, 1.0, size=T.shape[1])
                     for T in projections)
            # sample_p_v2 adds the prior to its A in place: pass copies
            draws[k] = linear_sampler.sample_p_v2(
                A.copy(), b_centred.copy(), wn,
                prior_cov_inv=prior_cov_inv, prior_mean=prior_mean_h,
                solver=None, Est_mode=False,
            )
    return draws
