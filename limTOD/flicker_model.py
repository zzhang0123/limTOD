import numpy as np
from mpmath import gammainc
from scipy.linalg import toeplitz


def aux_int(mu, u):
    """Auxiliary incomplete-gamma integral for the flicker-noise correlation.

    Raises RuntimeError (with the offending mu/u) if mpmath cannot evaluate
    the incomplete gamma function, instead of silently fabricating a value.
    """
    try:
        aux = gammainc(mu, 1j * u)
        ang = np.pi / 2 * mu
        return float(aux.real) * np.cos(ang) + float(aux.imag) * np.sin(ang)
    except ValueError as e:
        raise RuntimeError(
            f"aux_int failed for mu={mu}, u={u}: {e}; the flicker-noise "
            "correlation cannot be evaluated for these parameters"
        ) from e


def flicker_corr(tau, f0, fc, alpha, var_w=0.0):
    """Flicker (1/f) noise autocorrelation at lag ``tau``.

    Note that f0 and fc are in unit of angular frequency, differently from
    that of FFT frequency convention by a factor of 2pi. The closed form is
    singular at ``alpha == 1`` (the pure-1/f exponent): the zero-lag variance
    carries a ``1/(alpha-1)`` factor, so that exponent is rejected explicitly
    rather than returning inf/raising a type-dependent ZeroDivisionError.
    """
    if np.isclose(float(alpha), 1.0):
        raise ValueError(
            "flicker_corr/sim_noise: the closed-form correlation is singular "
            f"at alpha=1 (got alpha={alpha}); use an exponent away from 1 "
            "(e.g. 1.001) or a different noise model."
        )
    if tau == 0:
        return fc / np.pi * (f0 / fc) ** alpha / (alpha - 1) + var_w
    tau = np.abs(tau)
    theta_c = fc * tau
    theta_0 = f0 * tau
    norm = 1 / (np.pi * tau)
    mu = 1 - alpha
    result = theta_0**alpha * aux_int(mu, theta_c)
    return result * norm


def sim_noise(f0, fc, alpha, time_list, n_samples=1, white_n_variance=5e-6):
    """Draw flicker-noise realizations with autocorrelation ``flicker_corr``.

    Returns an array of shape ``(n_samples, len(time_list))``. ``f0``/``fc``
    are angular frequencies; ``alpha`` must differ from 1 (see flicker_corr).
    """
    lags = time_list - time_list[0]
    corr_list = [flicker_corr(t, f0, fc, alpha, var_w=white_n_variance) for t in lags]
    covmat = toeplitz(corr_list)
    if n_samples == 1:
        return np.random.multivariate_normal(np.zeros_like(time_list), covmat).reshape(
            1, -1
        )
    else:
        return np.random.multivariate_normal(
            np.zeros_like(time_list), covmat, n_samples
        )
