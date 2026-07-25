import logging
from typing import Any, List, Optional, Sequence, SupportsFloat, Tuple, Union, cast

import numpy as np
from scipy import signal
from scipy.linalg import solve, LinAlgError
from limTOD.simulator import truncate_stacked_beam, generate_sky2sys_projection

logger = logging.getLogger(__name__)


def get_filtfilt_matrix(n_samples: int, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    More accurate matrix representation of filtfilt operation.
    """
    
    # Create matrix by applying filtfilt to each standard basis vector
    H = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        e_i = np.zeros(n_samples)
        e_i[i] = 1.0
        H[:, i] = signal.filtfilt(b, a, e_i)
    
    return H

def HP_filter_TOD(n_samples: int, dtime: float, cutoff_freq: float = 0.001,
                  filter_order: int = 4,
                  preserve_dc: bool = False) -> np.ndarray:
    """
    Apply high-pass Butterworth filter to the TOD.
    Parameters:
    -----------
    n_samples : int
        Number of samples in the TOD
    dtime : float
        Time interval between samples in seconds
    cutoff_freq : float, default=0.001 Hz
        Cutoff frequency for high-pass filter in unit of Hz
    filter_order : int, default=4
        Order of the Butterworth filter (typical range: 2-8)
        Higher order = sharper cutoff but more edge effects
    preserve_dc : bool, default=False
        If True, add back the DC projection so the filter has unit gain
        at ℓ=0 while still rejecting drifts between DC and the cutoff.
        Constructed as H' = H (I - P) + P with P = J/n (the DC
        projection). Useful only when the per-chunk mean is dominated
        by sky (low-noise regime). In the realistic 1/f regime the
        chunk mean is dominated by drift realisation, so preserve_dc
        would let drift DC leak into the recovered map — keep the
        default False unless you've verified this for your noise model.

    Returns:
    --------
    HP_operator : array-like, shape (n_time, n_params)
        High-pass filtered system temperature operator

    """
    # Design a high-pass Butterworth filter
    fs = 1.0 / dtime
    nyquist = fs / 2.0
    normalized_cutoff = cutoff_freq / nyquist # Normalized cutoff frequency for high-pass filter

    # Validate normalized cutoff frequency
    if normalized_cutoff <= 0:
        raise ValueError(f"Cutoff frequency must be positive. Got cutoff_freq={cutoff_freq} Hz")
    if normalized_cutoff >= 1:
        raise ValueError(
            f"Cutoff frequency ({cutoff_freq} Hz) must be less than Nyquist frequency ({nyquist} Hz). "
            f"Normalized cutoff = {normalized_cutoff:.3f} >= 1.0"
        )

    b, a = signal.butter(filter_order, normalized_cutoff, btype='high', analog=False)

    H_exact = get_filtfilt_matrix(n_samples, b, a) # Exact matrix representation of filtfilt operation
    if preserve_dc:
        # H' x = H (I - P) x + P x = H (x - mean(x) 1) + mean(x) 1.
        # Constants pass through unchanged; everything faster than the
        # Butterworth cutoff is still attenuated.
        P = np.ones((n_samples, n_samples)) / n_samples
        H_exact = H_exact @ (np.eye(n_samples) - P) + P
    return H_exact


def wiener_filter_map(
    TOD: np.ndarray,
    operator: np.ndarray,
    noise_variance: Optional[Union[float, np.floating, np.ndarray]] = None,
    prior_inv_cov: Optional[Union[float, np.ndarray]] = None,
    guess: Optional[np.ndarray] = None,
    regularization: float = 1e-12,
    return_full_cov: bool = False,
    rolling_variance: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Apply Wiener filtering for mapmaking from time-ordered data.
    
    The Wiener filter solves: (A^T N^-1 A + S^-1)^-1 A^T N^-1 d
    where A is the operator, N is noise covariance, S is signal covariance, d is data
    
    Parameters:
    -----------
    TOD : array-like, shape (n_time,)
        Time-ordered data to be mapped
    operator : array-like, shape (n_time, n_pixels)
        Pointing/beam operator mapping sky pixels to TOD samples
    noise_variance : float or array-like, optional
        Noise variance. If None, estimated from TOD
    prior_inv_cov : float or array-like, optional
        Inverse of Prior covariance for the parameters. If None, uses uninformative prior
    regularization : float, default=1e-12
        Regularization parameter to ensure matrix invertibility
        
    Returns:
    --------
    sky_map : array, shape (n_pixels,)
        Reconstructed sky map
    uncertainty : array, shape (n_pixels,)
        Per-pixel uncertainty (diagonal of covariance matrix)
    """    
    
    # Convert inputs to numpy arrays
    TOD = np.asarray(TOD)
    operator = np.asarray(operator)
    
    n_time, n_pixels = operator.shape
    
    # Estimate noise variance if not provided
    if noise_variance is None:
        # Simple estimate: variance of high-pass filtered residuals
        residual = TOD - operator @ np.linalg.pinv(operator) @ TOD
        if rolling_variance:
            # Cap the window at the TOD length: with the fixed window of 100
            # samples, shorter TODs used to produce a truncated variance
            # vector and an opaque matmul shape error downstream.
            window_size = min(100, n_time)
            half_window = window_size // 2

            # Pad with first N and last N samples (reflected)
            # This provides smoother boundaries than just repeating edge value
            left_pad = residual[:half_window][::-1]  # First N samples, reversed
            right_pad = residual[-half_window:][::-1]  # Last N samples, reversed
            padded_residual = np.concatenate([left_pad, residual, right_pad])

            # Apply rolling window to squared residuals
            noise_variance = np.convolve(
                padded_residual**2,
                np.ones(window_size)/window_size,
                mode='valid'
            )

            # Trim to exact length if needed
            if len(noise_variance) > len(residual):
                # Take the middle portion
                excess = len(noise_variance) - len(residual)
                start = excess // 2
                noise_variance = noise_variance[start:start+len(residual)]
        else:
            noise_variance = np.var(residual)
            logger.info("Estimated noise variance: %.6f", noise_variance)


    # Create noise inverse covariance matrix (assume diagonal)
    if np.isscalar(noise_variance):
        N_inv = np.eye(n_time) / cast(float, noise_variance)
    else:
        noise_variance = np.asarray(noise_variance)
        if len(noise_variance) != n_time:
            raise ValueError(
                f"noise_variance has length {len(noise_variance)} but the TOD "
                f"has {n_time} samples"
            )
        # Convert to dense diagonal matrix for consistent matrix operations
        N_inv = np.diag(1.0 / noise_variance)
    
    # Create signal inverse covariance matrix
    if prior_inv_cov is None:
        S_inv: np.ndarray = np.zeros((n_pixels, n_pixels))  # Uninformative prior
    elif np.isscalar(prior_inv_cov):
        S_inv = np.eye(n_pixels) * cast(float, prior_inv_cov)
    elif cast(np.ndarray, prior_inv_cov).ndim == 1:
        # Convert to dense diagonal matrix for consistent matrix operations
        S_inv = np.diag(np.asarray(prior_inv_cov))
    elif cast(np.ndarray, prior_inv_cov).ndim == 2:
        S_inv = np.asarray(prior_inv_cov)
    else:
        raise ValueError("prior_inv_cov must be a scalar, 1D array, or 2D array.")

    if guess is None:
        guess = np.zeros(n_pixels)
        # Only override prior if none was provided
        if prior_inv_cov is None:
            S_inv = np.zeros((n_pixels, n_pixels))  # Uninformative prior
    else:
        guess = np.asarray(guess)
        if len(guess) != n_pixels:
            raise ValueError("Length of guess must match number of pixels.")

    # Compute Wiener filter components
    # If float type, transpose; if complex, conjugate transpose
    if np.iscomplexobj(operator):
        AtN = operator.conj().T @ N_inv  # A^H N^-1
    else:
        AtN = operator.T @ N_inv  # A^T N^-1
    AtNA = AtN @ operator     # A^dagger N^-1 A
    
    # Add signal prior and regularization
    covariance_inv = AtNA + S_inv + regularization * np.eye(n_pixels)
    
    # Right-hand side: A^T N^-1 d +  S^-1 mu
    rhs = AtN @ TOD + S_inv @ guess 

    posterior_cov = None
    try:
        # Solve the linear system: (A^T N^-1 A + S^-1) x = A^T N^-1 d +  S^-1 mu
        sky_map = solve(covariance_inv, rhs, assume_a='pos')

        # Compute uncertainties (diagonal of posterior covariance)
        try:
            posterior_cov = np.linalg.inv(covariance_inv)
            uncertainty = np.sqrt(np.diag(posterior_cov))
        except (LinAlgError, np.linalg.LinAlgError):
            logger.warning(
                "Could not compute full covariance matrix; using diagonal approximation."
            )
            uncertainty = 1.0 / np.sqrt(np.diag(covariance_inv))

    except (LinAlgError, np.linalg.LinAlgError) as e:
        logger.warning("Linear algebra error: %s; falling back to pseudo-inverse solution.", e)
        sky_map = np.linalg.pinv(operator) @ TOD
        uncertainty = np.ones(n_pixels) * np.nan

    if return_full_cov:
        # posterior_cov stayed None on the degraded paths (inv failure or the
        # pseudo-inverse fallback) — returning it unbound used to NameError.
        if posterior_cov is None:
            raise np.linalg.LinAlgError(
                "return_full_cov=True but the posterior covariance could not "
                "be computed (the normal-equations matrix is numerically "
                "singular); rerun with return_full_cov=False or increase "
                "regularization/priors."
            )
        return sky_map, uncertainty, posterior_cov
    else:
        return sky_map, uncertainty




# Alternative simplified version for quick mapmaking
def simple_wiener_map(
    TOD: np.ndarray,
    operator: np.ndarray,
    noise_var: Optional[Union[float, np.floating]] = None,
) -> np.ndarray:
    """
    Simplified Wiener filter assuming uninformative signal prior.
    Equivalent to: (A^T A + lambda*I)^-1 A^T d
    """    
    if noise_var is None:
        # Estimate from residuals
        residual = TOD - operator @ np.linalg.pinv(operator) @ TOD
        noise_var = np.var(residual)
    
    AtA = operator.T @ operator
    regularization = noise_var * 1e-6  # Small regularization
    
    # Regularized normal equation
    lhs = AtA + regularization * np.eye(AtA.shape[0])
    rhs = operator.T @ TOD
    
    sky_map = np.linalg.solve(lhs, rhs)
    
    return sky_map



class HPW_mapmaking:
    """
    Map-making class for Time-Ordered Data (TOD) using high-pass filtering and Wiener filtering.
    """

    def __init__(
        self,
        *,
        beam_map: np.ndarray,
        LST_deg_list_group: Union[np.ndarray, Sequence[np.ndarray]],
        lat_deg: float,
        azimuth_deg_list_group: Union[np.ndarray, Sequence[np.ndarray]],
        elevation_deg_list_group: Union[np.ndarray, Sequence[np.ndarray]],
        selfrot_deg_list_group: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        threshold: float = 0.01,
        Tsys_others_operator_group: Optional[Sequence[np.ndarray]] = None,
        nside_hires: Optional[int] = None,
        nside_target: Optional[int] = None,
        beam_truncate_frac_thres: Optional[float] = None
    ) -> None:
        """
        Initialize the HPW_mapmaking class.

        Parameters:
        beam_map : array
            The Healpix map of the beam pattern for a single frequency.
            Input map can be:
                a single array is considered I,
                array with 3 rows:[I,Q,U]
                array with 4 rows:[I,Q,U,V]

        LST_deg_list_group : a LST list or a list of LST lists corresponding to each TOD in TOD_group.
            e.g. [LST_deg_list_1, LST_deg_list_2, ...]
            Note that it can be generated by limTOD.simulator.generate_LSTs_deg function. For example:
                LST_deg_list = generate_LSTs_deg(
                    ant_latitude_deg,
                    ant_longitude_deg,
                    ant_height_m,
                    time_list,
                    start_time_utc=start_time_utc,
                )

        lat_deg : float
            The latitude of the observation site in degrees.

        azimuth_deg_list_group : an azimuth array or a list of azimuth lists corresponding to each TOD in TOD_group.
            e.g. [azimuth_deg_list_1, azimuth_deg_list_2, ...]

        elevation_deg_list_group : an elevation array or a list of elevation lists corresponding to each TOD in TOD_group.
            e.g. [elevation_deg_list_1, elevation_deg_list_2, ...]

        threshold : float
            The threshold to cut off the fractional beam response np.abs(beam[pixel])/beam_max, default is 0.01.
            e.g., if threshold=0.01, only pixels with beam response larger than 1% of the maximum will be considered.
            Note that this is the threshold for singling out pixels.

        Tsys_others_operator_group : an array or a list of arrays, optional
            The operator for other system temperature components (e.g., Trec and Tdiode) mapping to TOD.

        nside_hires : int, optional
            If provided, upgrade the beam map to this nside before processing.
            This can help improve accuracy when the beam is narrow.

        beam_truncate_frac_thres : float, optional
            The fractional threshold value for beam truncation. 
            If specified, set all pixels with values below this fraction of the maximum pixel value to zero. 
            If None, use the other key word "threshold" as the value.

        Note the difference between "threshold" and "beam_truncate_frac_thres":
            "threshold" is used to determine which pixels to include in the mapmaking process based on their beam response.
            "beam_truncate_frac_thres" is used to truncate the beam map itself before processing.
        
        """
        self.nside_hires = nside_hires
        self.nside_target = nside_target

        if beam_truncate_frac_thres is None:
            beam_truncate_frac_thres = threshold

        # If LST_deg_list_group[0] is a list, flatten it.
        if isinstance(LST_deg_list_group[0], (list, np.ndarray)):
            self.num_tods = len(LST_deg_list_group)
            if not (self.num_tods == len(azimuth_deg_list_group) == len(elevation_deg_list_group)):
                raise ValueError(
                    "Length of LST_deg_list_group, azimuth_deg_list_group, "
                    "elevation_deg_list_group must be the same."
                )
            LST_deg_list = np.concatenate(LST_deg_list_group)
            azimuth_deg_list = np.concatenate(azimuth_deg_list_group)
            elevation_deg_list = np.concatenate(elevation_deg_list_group)
            if selfrot_deg_list_group is not None:
                selfrot_deg_list = np.concatenate(selfrot_deg_list_group) 
            else:
                selfrot_deg_list = np.zeros_like(LST_deg_list)
        else:
            self.num_tods = 1
            # In this branch the groups are flat per-sample arrays (their
            # first element is a scalar); the casts only narrow the static
            # Union type and are identity operations at runtime.
            LST_deg_list = cast(np.ndarray, LST_deg_list_group)
            azimuth_deg_list = cast(np.ndarray, azimuth_deg_list_group)
            elevation_deg_list = cast(np.ndarray, elevation_deg_list_group)
            if selfrot_deg_list_group is not None:
                selfrot_deg_list = cast(np.ndarray, selfrot_deg_list_group)
            else:
                selfrot_deg_list = np.zeros_like(LST_deg_list)

        if beam_map.ndim == 1:
            self.npol = 1
        elif beam_map.ndim == 2:
            self.npol = beam_map.shape[0]
        else:
            raise ValueError("beam_map must be a 1D or 2D array.")

        if Tsys_others_operator_group is not None:
            # Canonicalize to a list of per-TOD 2D operators: the docstring
            # accepts "an array or a list of arrays", but a bare 2D array
            # used to crash ([0] indexed a row) and the single-TOD
            # concatenation mishandled the list form.
            if isinstance(Tsys_others_operator_group, np.ndarray) and Tsys_others_operator_group.ndim == 2:
                Tsys_others_operator_group = [Tsys_others_operator_group]
            if len(Tsys_others_operator_group) != self.num_tods:
                raise ValueError(
                    f"Tsys_others_operator_group has {len(Tsys_others_operator_group)} "
                    f"entries but there are {self.num_tods} TODs."
                )
            self.Tsys_others = True
            self.n_params_others = Tsys_others_operator_group[0].shape[1]
        else:
            self.Tsys_others = False
            self.n_params_others = 0

        horizontal_mask = None # Not used in current implementation, but can be added as a feature later if needed.

        self.pixel_indices = truncate_stacked_beam(
            beam_map, LST_deg_list, lat_deg, azimuth_deg_list, elevation_deg_list, selfrot_deg_list,
            horizontal_mask=horizontal_mask,
            threshold=threshold, 
            nside_hires=self.nside_hires,
            nside_target=self.nside_target
        )

        self.num_pixels = len(self.pixel_indices)
        self.nsky_params = self.npol * self.num_pixels

        
        if self.num_tods > 1:

            # List[np.ndarray] here (one operator per TOD); a single stacked
            # ndarray in the num_tods == 1 branch below — hence Any.
            self.Tsys_operators: Any = []

            for i in range(self.num_tods):
                LST_deg_list_i = LST_deg_list_group[i]
                azimuth_deg_list_i = azimuth_deg_list_group[i]
                elevation_deg_list_i = elevation_deg_list_group[i]

                selfrot_deg_list = np.zeros_like(LST_deg_list_i) if selfrot_deg_list_group is None else selfrot_deg_list_group[i]

                sky_operator_i = generate_sky2sys_projection(
                    beam_map, LST_deg_list_i, lat_deg, azimuth_deg_list_i, elevation_deg_list_i, selfrot_deg_list,
                    self.pixel_indices, 
                    nside_hires=self.nside_hires,
                    nside_target=self.nside_target,
                    normalize_beam=False,
                    horizontal_mask=horizontal_mask,
                    truncate_frac_thres=beam_truncate_frac_thres
                )
                if Tsys_others_operator_group is not None:
                    other_operators = [np.zeros_like(item) for item in Tsys_others_operator_group]
                    other_operators[i] = Tsys_others_operator_group[i]
                    Tsys_operator_i = np.concatenate([sky_operator_i] + other_operators, axis=1) 
                else:
                    Tsys_operator_i = sky_operator_i

                self.Tsys_operators.append(Tsys_operator_i)

        else:
            sky_operators = generate_sky2sys_projection(
                beam_map, LST_deg_list, lat_deg, azimuth_deg_list, elevation_deg_list, selfrot_deg_list,
                self.pixel_indices, 
                horizontal_mask=horizontal_mask,
                normalize_beam=False,
                nside_hires=self.nside_hires,
                nside_target=self.nside_target,
                truncate_frac_thres=beam_truncate_frac_thres
            )
            if Tsys_others_operator_group is not None:
                self.Tsys_operators = np.concatenate(
                    [sky_operators] + list(Tsys_others_operator_group), axis=1
                )
            else:
                self.Tsys_operators = sky_operators

    def _filter_and_stack(
        self,
        TOD_group: Union[np.ndarray, Sequence[np.ndarray]],
        dtime: float,
        cutoff_freq_group: Optional[Union[float, Sequence[float]]],
        gain_group: Union[float, np.ndarray, Sequence[Union[float, np.ndarray]]],
        known_injection_group: Optional[Union[np.ndarray, Sequence[np.ndarray]]],
        filter_order: int,
        preserve_dc: bool,
        use_high_pass: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate, optionally high-pass filter, and stack every TOD and
        its system operator into one overall linear system.

        Returns ``(HP_cal_TOD_overall, HP_Tsys_operator_overall)`` and
        records the per-TOD filter matrices in ``self.HP_exact``.
        """

        def make_tod_filter(n_samples: int, cutoff_freq: Optional[float]) -> np.ndarray:
            if not use_high_pass:
                return np.eye(n_samples)
            if cutoff_freq is None:
                raise ValueError("cutoff_freq_group must be provided when use_high_pass=True.")
            return HP_filter_TOD(
                n_samples,
                dtime,
                cutoff_freq=cutoff_freq,
                filter_order=filter_order,
                preserve_dc=preserve_dc,
            )

        self.HP_exact: List[np.ndarray] = []
        if self.num_tods > 1:

            for i in range(self.num_tods):
                # The multi-TOD path requires per-TOD sequences; the casts
                # narrow the scalar-or-sequence Unions (identity at runtime).
                cutoff_freq = None if cutoff_freq_group is None else cast(Sequence[float], cutoff_freq_group)[i]
                hp_filter_mat = make_tod_filter(len(TOD_group[i]), cutoff_freq)
                self.HP_exact.append(hp_filter_mat)
                calibrated_TOD_i = np.asarray(TOD_group[i]) / cast(Sequence[Any], gain_group)[i]
                if known_injection_group is not None:
                    calibrated_TOD_i -= known_injection_group[i]
                hp_cal_TOD_i = hp_filter_mat @ calibrated_TOD_i
                hp_Tsys_operator_i = hp_filter_mat @ self.Tsys_operators[i]

                if i == 0:
                    HP_Tsys_operator_overall = hp_Tsys_operator_i
                    HP_cal_TOD_overall = hp_cal_TOD_i
                else:
                    HP_Tsys_operator_overall = np.concatenate([HP_Tsys_operator_overall, hp_Tsys_operator_i])
                    HP_cal_TOD_overall = np.concatenate([HP_cal_TOD_overall, hp_cal_TOD_i])

        elif self.num_tods == 1:
            TOD = TOD_group if isinstance(TOD_group, np.ndarray) and TOD_group.ndim == 1 else TOD_group[0]
            cutoff_freq = cutoff_freq_group if isinstance(cutoff_freq_group, (int, float)) else (
                None if cutoff_freq_group is None else cutoff_freq_group[0]
            )
            gain = gain_group if isinstance(gain_group, (int, float)) else gain_group[0]
            calibrated_TOD = np.asarray(TOD) / gain
            if known_injection_group is not None:
                known_injection = known_injection_group if isinstance(known_injection_group, np.ndarray) and known_injection_group.ndim == 1 else known_injection_group[0]
                calibrated_TOD -= known_injection
            hp_filter_mat = make_tod_filter(len(TOD), cutoff_freq)
            HP_cal_TOD_overall = hp_filter_mat @ calibrated_TOD
            HP_Tsys_operator_overall = hp_filter_mat @ self.Tsys_operators

        return HP_cal_TOD_overall, HP_Tsys_operator_overall

    def _build_priors(
        self,
        Tsky_prior_mean: Optional[np.ndarray],
        Tsys_other_prior_mean_group: Optional[Sequence[np.ndarray]],
        Tsky_prior_inv_cov_diag: Optional[np.ndarray],
        Tsys_other_prior_inv_cov_group: Optional[Sequence[np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble the joint prior mean vector and prior inverse-covariance
        matrix over (sky pixels + other-Tsys parameters). Requires
        ``self.nparams`` to be set."""
        Tsys_prior_mean = np.zeros(self.nparams)
        if Tsky_prior_mean is not None:
            if len(Tsky_prior_mean) != self.nsky_params:
                raise ValueError("Length of Tsky_prior_mean must match number of sky parameters.")
            Tsys_prior_mean[:self.nsky_params] = Tsky_prior_mean
        counter = self.nsky_params
        if Tsys_other_prior_mean_group is not None:
            if not self.Tsys_others:
                raise ValueError("Tsys_others_operator must be provided in initialization if Tsys_other_prior_mean_group is provided.")
            if len(Tsys_other_prior_mean_group) != self.num_tods:
                raise ValueError("Length of Tsys_other_prior_mean_group must match number of TODs.")
            for Tsys_other_prior_mean_i in Tsys_other_prior_mean_group:
                Tsys_prior_mean[counter:counter+len(Tsys_other_prior_mean_i)] = Tsys_other_prior_mean_i
                counter += len(Tsys_other_prior_mean_i)

        Tsys_prior_inv_cov = np.zeros((self.nparams, self.nparams))
        if Tsky_prior_inv_cov_diag is not None:
            Tsky_prior_inv_cov_diag = np.asarray(Tsky_prior_inv_cov_diag).reshape(-1) # flatten
            if len(Tsky_prior_inv_cov_diag) != self.nsky_params:
                raise ValueError("Length of Tsky_prior_inv_cov_diag must match number of sky parameters.")
            Tsys_prior_inv_cov[:self.nsky_params, :self.nsky_params] = np.diag(Tsky_prior_inv_cov_diag)

        counter = self.nsky_params
        if Tsys_other_prior_inv_cov_group is not None:
            if not self.Tsys_others:
                raise ValueError("Tsys_others_operator must be provided in initialization if Tsys_other_prior_inv_cov_group is provided.")
            if len(Tsys_other_prior_inv_cov_group) != self.num_tods:
                raise ValueError("Length of Tsys_other_prior_inv_cov_group must match number of TODs.")

            for Tsys_other_prior_inv_cov_i in Tsys_other_prior_inv_cov_group:
                # Branch on THIS element's ndim: checking group[0] here used to
                # misroute mixed 1D/2D groups, silently np.diag-ing a 2D
                # covariance and discarding its off-diagonal entries.
                Tsys_other_prior_inv_cov_i = np.asarray(Tsys_other_prior_inv_cov_i)
                if Tsys_other_prior_inv_cov_i.ndim == 1:
                    n_others = len(Tsys_other_prior_inv_cov_i)
                    Tsys_prior_inv_cov[counter:counter+n_others, counter:counter+n_others] = np.diag(Tsys_other_prior_inv_cov_i)
                    counter += n_others
                elif Tsys_other_prior_inv_cov_i.ndim == 2:
                    n_others = Tsys_other_prior_inv_cov_i.shape[0]
                    Tsys_prior_inv_cov[counter:counter+n_others, counter:counter+n_others] = Tsys_other_prior_inv_cov_i
                    counter += n_others
                else:
                    raise ValueError("Each element in Tsys_other_prior_inv_cov_group must be a 1D or 2D array.")

        return Tsys_prior_mean, Tsys_prior_inv_cov

    def _normalize_noise_variance(
        self,
        noise_variance: Optional[Union[float, np.ndarray, List[Union[float, np.ndarray]], Tuple[Union[float, np.ndarray], ...]]],
        TOD_group: Union[np.ndarray, Sequence[np.ndarray]],
        n_overall: int,
    ) -> Optional[Union[float, np.ndarray]]:
        """Normalise per-TOD noise_variance into the 1D/scalar/None form
        wiener_filter_map expects. Accepts None / scalar / 1D array /
        list-of-(scalar|1D-array)."""
        nv = noise_variance
        if isinstance(nv, (list, tuple)):
            if len(nv) != self.num_tods:
                raise ValueError(
                    f"noise_variance list length {len(nv)} != num_tods {self.num_tods}"
                )
            tod_lengths = [len(TOD_group[i]) for i in range(self.num_tods)] \
                if self.num_tods > 1 else [n_overall]
            pieces: List[np.ndarray] = []
            for i, nv_i in enumerate(nv):
                if np.isscalar(nv_i):
                    pieces.append(np.full(tod_lengths[i], float(cast(SupportsFloat, nv_i))))
                else:
                    nv_i = np.asarray(nv_i, dtype=float)
                    if nv_i.shape != (tod_lengths[i],):
                        raise ValueError(
                            f"noise_variance[{i}] shape {nv_i.shape} != ({tod_lengths[i]},)"
                        )
                    pieces.append(nv_i)
            nv = np.concatenate(pieces)
        return nv

    def __call__(
        self,
        *,
        TOD_group: Union[np.ndarray, Sequence[np.ndarray]],
        dtime: float,
        cutoff_freq_group: Optional[Union[float, Sequence[float]]] = None,
        gain_group: Optional[Union[float, np.ndarray, Sequence[Union[float, np.ndarray]]]] = None,
        known_injection_group: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        Tsky_prior_mean: Optional[np.ndarray] = None,
        Tsky_prior_inv_cov_diag: Optional[np.ndarray] = None,
        Tsys_other_prior_mean_group: Optional[Sequence[np.ndarray]] = None,
        Tsys_other_prior_inv_cov_group: Optional[Sequence[np.ndarray]] = None,
        noise_variance: Optional[Union[float, np.ndarray, List[Union[float, np.ndarray]], Tuple[Union[float, np.ndarray], ...]]] = None,
        regularization: float = 1e-12,
        return_full_cov: bool = False,
        filter_order: int = 4,
        preserve_dc: bool = False,
        use_high_pass: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]],
    ]:
        """
        TOD_group : a TOD array or a list of TOD arrays at the same frequency channel.
            e.g. [TOD_1, TOD_2, ...]

        gain_group : a gain array or a list of gain arrays corresponding to each TOD in TOD_group.
            e.g. [gain_1, gain_2, ...]
            gain_i can be a single value (constant gain) or an array with the same length as TOD_i.
            If None, assumed to be 1. (i.e., TOD is already calibrated)
            
        dtime : float
            Time interval between samples in seconds.

        cutoff_freq_group : list of float, optional
            Cutoff frequency for high-pass filter in Hz. Required when
            use_high_pass is True; ignored when use_high_pass is False.

        known_injection_group : a list of known system temperature components to be subtracted from Tsys (calibrated TOD),  each element corresponding to each TOD in TOD_group.
            e.g. [mu_1, mu_2, ...]
            A concrete example in MeerKLASS, mu_i can be time sequence of constant noise diode temperature, if we do not take it as a parameter.

        Tsky_prior_mean : array, optional
            Prior mean for the sky temperature map, the shape is (npol, num_pixels) for multi-polarization maps, or (num_pixels,) for single polarization map.
            If None, assumed to be zero.

        Tsky_prior_inv_cov_diag : array, optional
            Diagonal of the prior inverse covariance for the sky temperature map, the shape can be:
                (num_pixels,) : single polarization map.
                (npol, num_pixels) : multi-polarization map.
            If None, assumed to be uninformative prior (zero, i.e., infinite prior variance).

        Tsys_other_prior_mean_group : a list of prior means for other system temperature components, each element corresponding to each TOD in TOD_group.
            e.g. [Tsys_other_prior_mean_1, Tsys_other_prior_mean_2, ...]
            If None, assumed to be zero.

        noise_variance : float, 1D array, or list of (float | 1D array), optional
            Per-sample noise variance. Three forms accepted:
              * None (default): auto-estimate from the residual TOD - operator @ pinv(operator) @ TOD
                via a 100-sample rolling window. NOTE: this estimate can be heavily biased
                low when the operator does not span the projectable signal subspace —
                it conflates un-projectable signal with noise, which mis-weights data
                in the Wiener filter. Provide an explicit value when possible.
              * scalar: uniform variance applied to every concatenated sample.
              * 1D array of length sum(len(TOD_i)): per-sample variances over the
                concatenated TOD ordering.
              * list of length num_tods, each entry a scalar or 1D array of length
                len(TOD_i): per-TOD variance, concatenated internally.

        Tsys_other_prior_inv_cov_group : a list of prior inverse covariances for other system temperature components, each element corresponding to each TOD in TOD_group.
            e.g. [Tsys_other_prior_inv_cov_1, Tsys_other_prior_inv_cov_2, ...]
            The shape of each element can be:
                (num_other_params,) : Diagonal of the inverse covariance matrix.
                (num_other_params, num_other_params) : Full inverse covariance matrix. 
            But all elements must have the same shape.
            If None, assumed to be uninformative prior (zero, i.e., infinite prior variance).

        use_high_pass : bool, default=False
            If True, apply the Butterworth high-pass filter to the TOD and
            system operator. If False, use the identity matrix and solve the
            unfiltered map-making problem.


        Returns:        
        --------
        sky_estimation : array, the shape is (npol, num_pixels) for multi-polarization maps, or (num_pixels,) for single polarization map.
            Reconstructed sky map(s).
        sky_uncertainty : array, the shape is (npol, num_pixels) for multi-polarization maps, or (num_pixels,) for single polarization map.
            Uncertainty map(s) (diagonal of covariance matrix).
        Tsys_others_estimation_group : list of arrays, each with shape (num_other_params,)
            Reconstructed other system temperature components for each TOD, only returned if Tsys_others_operator is provided.
        Tsys_others_uncertainty_group : list of arrays, each with shape (num_other_params,)
            Per-parameter uncertainty (diagonal of covariance matrix) for other system temperature components, only returned if Tsys_others_operator is provided.
        """

        if gain_group is None:
            gain_group = [1.0]*self.num_tods

        HP_cal_TOD_overall, HP_Tsys_operator_overall = self._filter_and_stack(
            TOD_group, dtime, cutoff_freq_group, gain_group,
            known_injection_group, filter_order, preserve_dc, use_high_pass,
        )

        # # Debug: print the shape of the overall operator
        self.nparams = HP_Tsys_operator_overall.shape[1]

        Tsys_prior_mean, Tsys_prior_inv_cov = self._build_priors(
            Tsky_prior_mean, Tsys_other_prior_mean_group,
            Tsky_prior_inv_cov_diag, Tsys_other_prior_inv_cov_group,
        )

        nv = self._normalize_noise_variance(
            noise_variance, TOD_group, len(HP_cal_TOD_overall)
        )

        # Apply Wiener filter with the overall operator
        result = wiener_filter_map(
            HP_cal_TOD_overall,
            HP_Tsys_operator_overall,
            noise_variance=nv,  # explicit if provided, else auto-estimated inside
            prior_inv_cov=Tsys_prior_inv_cov,
            guess=Tsys_prior_mean,
            regularization=regularization,
            return_full_cov=return_full_cov,
        )
        # wiener_filter_map returns a 3-tuple when return_full_cov=True; the
        # posterior covariance is appended to this method's return values
        # (a 2-name unpacking here used to crash on that flag).
        if return_full_cov:
            estmation, uncertainty, posterior_cov = cast(
                Tuple[np.ndarray, np.ndarray, np.ndarray], result
            )
        else:
            estmation, uncertainty = cast(Tuple[np.ndarray, np.ndarray], result)
            posterior_cov = None

        sky_estimation = estmation[:self.nsky_params]
        sky_uncertainty = uncertainty[:self.nsky_params]

        if self.npol > 1:
            sky_estimation = sky_estimation.reshape(self.npol, self.num_pixels)
            sky_uncertainty = sky_uncertainty.reshape(self.npol, self.num_pixels)

        outputs: Tuple[Any, ...] = (sky_estimation, sky_uncertainty)
        if self.Tsys_others:
            Tsys_others_estimation_group = []
            Tsys_others_uncertainty_group = []
            counter = self.nsky_params
            for i in range(self.num_tods):
                Tsys_others_estimation_group.append(estmation[counter:counter+self.n_params_others])
                Tsys_others_uncertainty_group.append(uncertainty[counter:counter+self.n_params_others])
                counter += self.n_params_others
            outputs = outputs + (Tsys_others_estimation_group, Tsys_others_uncertainty_group)
        if return_full_cov:
            outputs = outputs + (posterior_cov,)
        return outputs
        
