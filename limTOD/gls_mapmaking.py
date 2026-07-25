"""GLS map-making with the multiplicative (gain-noise) data model.

The classic :class:`limTOD.HPW_filter.HPW_mapmaking` solves the (optionally
high-pass-filtered) normal equations with a *diagonal* noise weight —
effectively an ordinary-least-squares estimate on filtered data. This module
adds a generalised-least-squares (GLS) map-maker that weights the data with
the inverse of the full time-time noise covariance of limTOD's own noise
model. The solver is ported from ``hydra_tod.linear_sampler``
(Zhang et al. 2026, RASTI, rzag024, Sec. 3.2), serial — without hydra-tod's
MPI or Toeplitz-extension dependencies.

Data model (matching :meth:`limTOD.TODSim.generate_TOD`)::

    d = g_bg * (1 + n_g) * (U p + mu) * (1 + n_w)
      ≈ g_bg * (U p + mu) * (1 + n),        n = n_g + n_w

* ``U p`` — the sky (+ other-Tsys) operator applied to the parameters,
* ``mu`` — known injected temperature (e.g. a noise diode), part of the
  multiplicative model, NOT subtracted from the data,
* ``n_g`` — 1/f gain noise with covariance ``toeplitz(flicker_corr)``,
* ``n_w`` — fractional white noise with variance ``white_noise_var``.

The second-order ``n_g * n_w`` cross term is neglected (~1e-8 relative for
the default noise parameters), so the fractional-noise covariance is::

    N = toeplitz(flicker_corr(tau; f0, fc, alpha)) + white_noise_var * I

— exactly the covariance :func:`limTOD.flicker_model.sim_noise` draws the
simulated noise from. Because the noise is multiplicative, the covariance
of the calibrated data ``d / g_bg`` depends on the signal itself::

    Sigma = diag(U p + mu) N diag(U p + mu)

so the GLS is iterated (IRLS): solve, re-weight with the updated
``U p + mu``, repeat until the parameters converge.

For externally calibrated data with *additive* noise
(``d = U p + mu + eps``, ``Cov[eps] = N``) pass
``noise_model="additive"``: a single non-iterative GLS solve.
"""

import logging
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve, toeplitz

from limTOD.HPW_filter import _MapmakingBase
from limTOD.flicker_model import flicker_corr
from limTOD.simulator import DEFAULT_GAIN_NOISE_PARAMS

logger = logging.getLogger(__name__)

_NOISE_MODELS = ("multiplicative", "additive")


def flicker_noise_cov(
    time_list: Union[Sequence[float], np.ndarray],
    gain_noise_params: Optional[Sequence[float]] = DEFAULT_GAIN_NOISE_PARAMS,
    white_noise_var: float = 2.5e-6,
) -> np.ndarray:
    """Fractional-noise covariance matrix of limTOD's TOD noise model.

    ``N = toeplitz(flicker_corr(tau; f0, fc, alpha)) + white_noise_var * I``
    over the lags of ``time_list`` — the same matrix
    :func:`limTOD.flicker_model.sim_noise` draws the 1/f realisation from
    (plus the white-noise diagonal that ``generate_TOD`` injects
    separately), so a GLS weighted with its inverse is exactly matched to
    limTOD-simulated TOD.

    Parameters
    ----------
    time_list : array-like of float
        Sample times in seconds. Like ``sim_noise``, the Toeplitz
        construction indexes lags relative to the first sample, which is
        exact for uniformly spaced times.
    gain_noise_params : (f0, fc, alpha) or None
        Flicker parameters in the limTOD convention (``f0``/``fc`` are
        angular frequencies; ``alpha != 1``). ``None`` disables the 1/f
        term (white-only covariance). Default matches
        :meth:`limTOD.TODSim.generate_TOD`.
    white_noise_var : float
        Variance of the fractional white noise (``generate_TOD`` default
        ``2.5e-6``).

    Returns
    -------
    N : ndarray, shape (n_time, n_time)
    """
    time_arr = np.asarray(time_list, dtype=float)
    if time_arr.ndim != 1 or time_arr.size == 0:
        raise ValueError("time_list must be a non-empty 1D array of sample times")
    if white_noise_var < 0:
        raise ValueError(f"white_noise_var must be >= 0, got {white_noise_var}")
    if gain_noise_params is None:
        if white_noise_var == 0:
            raise ValueError(
                "gain_noise_params=None with white_noise_var=0 gives a zero "
                "(singular) noise covariance; provide at least one component"
            )
        return white_noise_var * np.eye(time_arr.size)
    f0, fc, alpha = gain_noise_params
    lags = time_arr - time_arr[0]
    corr = [flicker_corr(tau, f0, fc, alpha, var_w=white_noise_var) for tau in lags]
    return toeplitz(corr)


def flicker_noise_inv_cov(
    time_list: Union[Sequence[float], np.ndarray],
    gain_noise_params: Optional[Sequence[float]] = DEFAULT_GAIN_NOISE_PARAMS,
    white_noise_var: float = 2.5e-6,
) -> np.ndarray:
    """Inverse of :func:`flicker_noise_cov` (via Cholesky).

    Raises a ``LinAlgError`` with an actionable message if the covariance
    is not positive definite (e.g. ``white_noise_var=0`` with a nearly
    singular flicker correlation).
    """
    N = flicker_noise_cov(time_list, gain_noise_params, white_noise_var)
    try:
        cho = cho_factor(N)
    except LinAlgError as e:
        raise LinAlgError(
            "flicker_noise_inv_cov: the noise covariance is not positive "
            f"definite ({e}); increase white_noise_var or check the flicker "
            "parameters"
        ) from e
    return cho_solve(cho, np.eye(N.shape[0]))


def _sigma_inv_weighted(
    U: np.ndarray,
    d: np.ndarray,
    N_inv: np.ndarray,
    model: np.ndarray,
    mu: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """One block's normal-equation contributions ``(U^T Σ⁻¹ U, U^T Σ⁻¹ (d-mu))``
    with ``Σ⁻¹ = diag(1/model) N⁻¹ diag(1/model)``."""
    with np.errstate(divide="ignore"):
        D_p_inv = 1.0 / model
    if not np.all(np.isfinite(D_p_inv)):
        raise FloatingPointError(
            "iterative GLS: the model TOD (U p + mu) crossed zero, so the "
            "multiplicative noise weights diverged; check the operator/prior "
            "or use noise_model='additive'"
        )
    sigma_inv = N_inv * np.outer(D_p_inv, D_p_inv)
    aux = U.T @ sigma_inv
    return aux @ U, aux @ (d - mu)


def iterative_gls(
    d: np.ndarray,
    U: np.ndarray,
    N_inv: np.ndarray,
    mu: Union[float, np.ndarray] = 0.0,
    tol: float = 1e-10,
    min_iter: int = 5,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively-reweighted GLS for the multiplicative noise model.

    Solves for ``p`` in ``d = (U p + mu)(1 + n)`` with ``Cov[n] = N``.
    Faithful serial port of ``hydra_tod.linear_sampler.iterative_gls``
    (Zhang et al. 2026, RASTI, rzag024): initialise with OLS, then repeat

    ``(U^T Σ⁻¹ U) p = U^T Σ⁻¹ (d - mu)``,
    ``Σ⁻¹ = diag(1/(U p + mu)) N⁻¹ diag(1/(U p + mu))``

    until the parameter norm changes by less than ``tol`` (after at least
    ``min_iter`` iterations).

    Returns
    -------
    p_gls : ndarray, shape (n_params,)
        Converged parameter estimate.
    Sigma_inv : ndarray, shape (n_time, n_time)
        Inverse data covariance evaluated at the converged estimate (as in
        hydra-tod, for downstream Gaussian sampling).
    """
    d = np.asarray(d, dtype=float)
    U = np.asarray(U, dtype=float)
    p = np.linalg.lstsq(U, d - mu, rcond=None)[0]

    p_new = p
    for iteration in range(1, max_iter + 1):
        A, b = _sigma_inv_weighted(U, d, N_inv, U @ p + mu, mu)
        p_new = solve(A, b, assume_a="sym")
        if (
            np.linalg.norm(p_new - p) < tol * np.linalg.norm(p)
            and iteration >= min_iter
        ):
            logger.info("iterative_gls converged in %d iterations.", iteration)
            break
        p = p_new
    else:
        logger.warning(
            "iterative_gls did not reach tol=%g within max_iter=%d iterations.",
            tol, max_iter,
        )

    D_p_inv = 1.0 / (U @ p_new + mu)
    Sigma_inv = N_inv * np.outer(D_p_inv, D_p_inv)
    return p_new, Sigma_inv


class GLS_mapmaking(_MapmakingBase):
    """GLS map-maker sharing :class:`limTOD.HPW_mapmaking`'s operator build.

    Construction (scan geometry -> truncated pixel set -> per-TOD system
    operators) is identical to ``HPW_mapmaking`` — the ``__init__``
    signature is the same, see that class. Only the solve differs: instead
    of high-pass filtering + diagonally-weighted Wiener solution, the data
    are weighted by the inverse of the full 1/f + white noise covariance
    under the multiplicative model of :meth:`limTOD.TODSim.generate_TOD`
    (iteratively reweighted, per the hydra-tod GLS), or under a plain
    additive-noise model (single GLS solve).

    Example
    -------
    >>> mm = GLS_mapmaking(beam_map=..., LST_deg_list_group=..., ...)
    >>> sky, unc = mm(TOD_group=tod, dtime=2.0,
    ...               gain_noise_params=(1.335e-5, 1.099e-3, 2),
    ...               white_noise_var=2.5e-6)
    """

    def __call__(
        self,
        *,
        TOD_group: Union[np.ndarray, Sequence[np.ndarray]],
        dtime: Optional[float] = None,
        time_list_group: Optional[Sequence[np.ndarray]] = None,
        gain_group: Optional[Union[float, np.ndarray, Sequence[Union[float, np.ndarray]]]] = None,
        known_injection_group: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        gain_noise_params: Optional[Sequence[float]] = DEFAULT_GAIN_NOISE_PARAMS,
        white_noise_var: float = 2.5e-6,
        noise_inv_cov_group: Optional[Sequence[np.ndarray]] = None,
        noise_model: str = "multiplicative",
        Tsky_prior_mean: Optional[np.ndarray] = None,
        Tsky_prior_inv_cov_diag: Optional[np.ndarray] = None,
        Tsys_other_prior_mean_group: Optional[Sequence[np.ndarray]] = None,
        Tsys_other_prior_inv_cov_group: Optional[Sequence[np.ndarray]] = None,
        regularization: float = 1e-12,
        return_full_cov: bool = False,
        tol: float = 1e-10,
        min_iter: int = 5,
        max_iter: int = 100,
    ) -> Tuple[Any, ...]:
        """Run the GLS map-maker on one frequency channel's TOD(s).

        Parameters
        ----------
        TOD_group : array or list of arrays
            Raw TOD(s), same convention as ``HPW_mapmaking.__call__``.
        dtime : float, optional
            Uniform sample spacing in seconds; used to build each TOD's
            noise covariance. Required unless ``time_list_group`` or
            ``noise_inv_cov_group`` is given.
        time_list_group : list of arrays, optional
            Per-TOD sample times in seconds (overrides ``dtime``); pass the
            same ``time_list`` the TOD was simulated/observed with.
        gain_group : scalar/array or list, optional
            Known background gain divided out first (default 1: calibrated).
        known_injection_group : array or list of arrays, optional
            Known injected temperature ``mu`` per TOD (e.g. noise diode).
            Under the multiplicative model this is part of the signal model
            ``(U p + mu)(1 + n)`` — it is NOT subtracted from the data
            (subtracting would mis-weight the noise); under the additive
            model it enters the residual ``d - mu``.
        gain_noise_params : (f0, fc, alpha), optional
            Flicker parameters used to build the fractional-noise
            covariance via :func:`flicker_noise_cov`. Default matches
            ``generate_TOD``. Ignored if ``noise_inv_cov_group`` is given.
        white_noise_var : float
            Fractional white-noise variance for the covariance build
            (``generate_TOD`` default ``2.5e-6``).
        noise_inv_cov_group : list of (n_time_i, n_time_i) arrays, optional
            Explicit per-TOD inverse noise covariances (fractional noise
            for the multiplicative model, additive noise otherwise).
        noise_model : {"multiplicative", "additive"}
            ``"multiplicative"`` (default): IRLS on ``d = (U p + mu)(1+n)``,
            the limTOD/hydra-tod model. ``"additive"``: one GLS solve of
            ``d = U p + mu + eps``.
        Tsky_prior_mean : array, optional
            Gaussian prior mean on the sky parameters, as in
            ``HPW_mapmaking.__call__``.
        Tsky_prior_inv_cov_diag : array, optional
            Diagonal prior inverse covariance on the sky parameters.
        Tsys_other_prior_mean_group : list of arrays, optional
            Prior means for the other-Tsys parameters, per TOD.
        Tsys_other_prior_inv_cov_group : list of arrays, optional
            Prior inverse covariances for the other-Tsys parameters.
        regularization : float
            Added to the normal-equation diagonal (default 1e-12).
        return_full_cov : bool
            Also return the full posterior parameter covariance.
        tol : float
            IRLS relative convergence tolerance (multiplicative only).
        min_iter : int
            Minimum IRLS iterations before convergence is checked.
        max_iter : int
            Maximum IRLS iterations.

        Returns
        -------
        outputs : tuple
            Same structure as ``HPW_mapmaking.__call__``:
            ``(sky_estimation, sky_uncertainty)``, extended with the
            other-Tsys estimation/uncertainty groups when other-Tsys
            operators were supplied, and with ``posterior_cov`` when
            ``return_full_cov=True``.
        """
        if noise_model not in _NOISE_MODELS:
            raise ValueError(
                f"noise_model must be one of {_NOISE_MODELS}, got {noise_model!r}"
            )

        tod_list = self._as_tod_list(TOD_group)
        ops: List[np.ndarray] = (
            list(self.Tsys_operators) if self.num_tods > 1
            else [cast(np.ndarray, self.Tsys_operators)]
        )
        for i, (d_i, U_i) in enumerate(zip(tod_list, ops)):
            if d_i.shape[0] != U_i.shape[0]:
                raise ValueError(
                    f"TOD {i} has {d_i.shape[0]} samples but its system "
                    f"operator has {U_i.shape[0]} rows"
                )

        gain_list = self._as_per_tod(gain_group, default=1.0)
        mu_list = self._as_per_tod(known_injection_group, default=0.0)
        d_list = [
            np.asarray(d, dtype=float) / g for d, g in zip(tod_list, gain_list)
        ]

        Ninv_list = self._noise_inv_list(
            noise_inv_cov_group, tod_list, dtime, time_list_group,
            gain_noise_params, white_noise_var, noise_model,
        )

        self.nparams = ops[0].shape[1]
        prior_mean, S_inv = self._build_priors(
            Tsky_prior_mean, Tsys_other_prior_mean_group,
            Tsky_prior_inv_cov_diag, Tsys_other_prior_inv_cov_group,
        )

        p, A_final = self._solve_blocks(
            d_list, ops, Ninv_list, mu_list,
            S_inv=S_inv, prior_mean=prior_mean,
            regularization=regularization, noise_model=noise_model,
            tol=tol, min_iter=min_iter, max_iter=max_iter,
        )

        return self._assemble_outputs(p, A_final, return_full_cov)

    # ------------------------------------------------------------------ #
    # Input normalisation                                                #
    # ------------------------------------------------------------------ #
    def _as_tod_list(
        self, TOD_group: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> List[np.ndarray]:
        if self.num_tods == 1:
            tod = (
                TOD_group
                if isinstance(TOD_group, np.ndarray) and TOD_group.ndim == 1
                else TOD_group[0]
            )
            return [np.asarray(tod, dtype=float)]
        if len(TOD_group) != self.num_tods:
            raise ValueError(
                f"TOD_group has {len(TOD_group)} TODs but the geometry was "
                f"built for {self.num_tods}"
            )
        return [np.asarray(t, dtype=float) for t in TOD_group]

    def _as_per_tod(
        self,
        group: Optional[Union[float, np.ndarray, Sequence[Any]]],
        default: float,
    ) -> List[Union[float, np.ndarray]]:
        """Normalise a scalar / flat array / per-TOD list into a per-TOD list."""
        if group is None:
            return [default] * self.num_tods
        if np.isscalar(group):
            return [float(cast(float, group))] * self.num_tods
        if isinstance(group, np.ndarray) and group.ndim == 1 and self.num_tods == 1:
            return [group]
        if len(cast(Sequence[Any], group)) != self.num_tods:
            raise ValueError(
                f"per-TOD group has {len(cast(Sequence[Any], group))} entries "
                f"but there are {self.num_tods} TODs"
            )
        return [
            float(cast(float, g)) if np.isscalar(g)
            else np.asarray(g, dtype=float)
            for g in cast(Sequence[Any], group)
        ]

    def _noise_inv_list(
        self,
        noise_inv_cov_group: Optional[Sequence[np.ndarray]],
        tod_list: List[np.ndarray],
        dtime: Optional[float],
        time_list_group: Optional[Sequence[np.ndarray]],
        gain_noise_params: Optional[Sequence[float]],
        white_noise_var: float,
        noise_model: str = "multiplicative",
    ) -> List[np.ndarray]:
        if noise_inv_cov_group is not None:
            if len(noise_inv_cov_group) != self.num_tods:
                raise ValueError(
                    f"noise_inv_cov_group has {len(noise_inv_cov_group)} "
                    f"entries but there are {self.num_tods} TODs"
                )
            out = []
            for i, (Ninv, d_i) in enumerate(zip(noise_inv_cov_group, tod_list)):
                Ninv = np.asarray(Ninv, dtype=float)
                n_i = d_i.shape[0]
                if Ninv.shape != (n_i, n_i):
                    raise ValueError(
                        f"noise_inv_cov_group[{i}] has shape {Ninv.shape}, "
                        f"expected ({n_i}, {n_i})"
                    )
                out.append(Ninv)
            return out

        if time_list_group is not None:
            times = [np.asarray(t, dtype=float) for t in (
                [time_list_group] if isinstance(time_list_group, np.ndarray)
                and np.asarray(time_list_group).ndim == 1 and self.num_tods == 1
                else time_list_group
            )]
            if len(times) != self.num_tods:
                raise ValueError(
                    f"time_list_group has {len(times)} entries but there are "
                    f"{self.num_tods} TODs"
                )
        elif dtime is not None:
            times = [np.arange(d.shape[0]) * float(dtime) for d in tod_list]
        else:
            raise ValueError(
                "provide dtime, time_list_group, or noise_inv_cov_group so "
                "the noise covariance can be built"
            )
        if noise_model == "additive":
            # The parametric builder produces the FRACTIONAL-noise
            # covariance of the multiplicative model (its defaults are
            # tuned for generate_TOD's gain noise). Reusing it as an
            # absolute additive covariance fixes the relative
            # time-weighting but generally mis-scales the reported
            # uncertainties — make that explicit.
            logger.warning(
                "noise_model='additive' without noise_inv_cov_group: building "
                "the additive noise covariance from the flicker/white "
                "parameters, whose defaults describe FRACTIONAL noise in the "
                "multiplicative model. The point estimate is unaffected by "
                "the overall scale, but sky_uncertainty/posterior_cov are "
                "only meaningful if these parameters really describe your "
                "additive noise; otherwise pass noise_inv_cov_group."
            )
        return [
            flicker_noise_inv_cov(t, gain_noise_params, white_noise_var)
            for t in times
        ]

    # ------------------------------------------------------------------ #
    # Solver                                                             #
    # ------------------------------------------------------------------ #
    def _solve_blocks(
        self,
        d_list: List[np.ndarray],
        ops: List[np.ndarray],
        Ninv_list: List[np.ndarray],
        mu_list: List[Union[float, np.ndarray]],
        *,
        S_inv: np.ndarray,
        prior_mean: np.ndarray,
        regularization: float,
        noise_model: str,
        tol: float,
        min_iter: int,
        max_iter: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulate the per-TOD normal equations and solve.

        Multiplicative: IRLS (serial analogue of hydra-tod's
        ``iterative_gls_mpi_list``), with the Gaussian prior added inside
        each iteration. Additive: single accumulation with ``Σ⁻¹ = N⁻¹``.
        """
        n_par = ops[0].shape[1]
        A_prior = S_inv + regularization * np.eye(n_par)
        b_prior = S_inv @ prior_mean

        def accumulate(p: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            A = A_prior.copy()
            b = b_prior.copy()
            for d_i, U_i, Ninv_i, mu_i in zip(d_list, ops, Ninv_list, mu_list):
                if noise_model == "additive":
                    aux = U_i.T @ Ninv_i
                    A += aux @ U_i
                    b += aux @ (d_i - mu_i)
                else:
                    A_i, b_i = _sigma_inv_weighted(
                        U_i, d_i, Ninv_i, U_i @ cast(np.ndarray, p) + mu_i, mu_i
                    )
                    A += A_i
                    b += b_i
            return A, b

        if noise_model == "additive":
            A, b = accumulate(None)
            return solve(A, b, assume_a="sym"), A

        # OLS initialisation over the stacked system (as in hydra-tod).
        U_stack = np.concatenate(ops, axis=0)
        r_stack = np.concatenate(
            [d - mu for d, mu in zip(d_list, mu_list)]
        )
        p = np.linalg.lstsq(U_stack, r_stack, rcond=None)[0]

        p_new = p
        for iteration in range(1, max_iter + 1):
            A, b = accumulate(p)
            p_new = solve(A, b, assume_a="sym")
            if (
                np.linalg.norm(p_new - p) < tol * np.linalg.norm(p)
                and iteration >= min_iter
            ):
                logger.info("GLS_mapmaking converged in %d iterations.", iteration)
                break
            p = p_new
        else:
            logger.warning(
                "GLS_mapmaking did not reach tol=%g within max_iter=%d "
                "iterations.", tol, max_iter,
            )
        # Re-evaluate the normal equations AT the returned estimate (as
        # hydra-tod's iterative_gls recomputes Sigma_inv at p_new): the
        # in-loop A lags p_new by one iteration, which is bounded by tol
        # after convergence but can skew the reported uncertainties at the
        # percent level when max_iter is hit without convergence.
        A, _ = accumulate(p_new)
        return p_new, A

    # ------------------------------------------------------------------ #
    # Output assembly (mirrors HPW_mapmaking.__call__'s return shape)    #
    # ------------------------------------------------------------------ #
    def _assemble_outputs(
        self, p: np.ndarray, A_final: np.ndarray, return_full_cov: bool
    ) -> Tuple[Any, ...]:
        posterior_cov: Optional[np.ndarray] = None
        try:
            posterior_cov = np.linalg.inv(A_final)
            uncertainty = np.sqrt(np.diag(posterior_cov))
        except (LinAlgError, np.linalg.LinAlgError):
            logger.warning(
                "Could not invert the normal-equation matrix; using the "
                "diagonal approximation for uncertainties."
            )
            uncertainty = 1.0 / np.sqrt(np.diag(A_final))

        sky_estimation = p[: self.nsky_params]
        sky_uncertainty = uncertainty[: self.nsky_params]
        if self.npol > 1:
            sky_estimation = sky_estimation.reshape(self.npol, self.num_pixels)
            sky_uncertainty = sky_uncertainty.reshape(self.npol, self.num_pixels)

        outputs: Tuple[Any, ...] = (sky_estimation, sky_uncertainty)
        if self.Tsys_others:
            est_group, unc_group = [], []
            counter = self.nsky_params
            for _ in range(self.num_tods):
                est_group.append(p[counter:counter + self.n_params_others])
                unc_group.append(uncertainty[counter:counter + self.n_params_others])
                counter += self.n_params_others
            outputs = outputs + (est_group, unc_group)
        if return_full_cov:
            if posterior_cov is None:
                raise np.linalg.LinAlgError(
                    "return_full_cov=True but the posterior covariance could "
                    "not be computed (the normal-equations matrix is "
                    "numerically singular); increase regularization/priors."
                )
            outputs = outputs + (posterior_cov,)
        return outputs
