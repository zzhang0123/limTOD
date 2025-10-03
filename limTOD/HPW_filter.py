from scipy import signal
import numpy as np
from limTOD.simulator import truncate_stacked_beam, generate_sky2sys_projection


def get_filtfilt_matrix(n_samples, b, a):
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

def HP_filter_TOD(TOD, dtime, Tsys_operator, cutoff_freq=0.001):
    """
    Apply high-pass Butterworth filter to the TOD.
    Parameters:
    -----------
    TOD : array-like, shape (n_time,)
        Time-ordered data to be filtered
    dtime : float
        Time interval between samples in seconds
    Tsys_operator : array-like, shape (n_time, n_params)
        System temperature operator mapping system parameters (e.g. sky pixels and linear coefficients of receiver temperatures) to TOD samples
    cutoff_freq : float, default=0.001 Hz
        Cutoff frequency for high-pass filter in unit of Hz

    Returns:
    --------
    HP_operator : array-like, shape (n_time, n_params)
        High-pass filtered system temperature operator

    """
    # Design a high-pass Butterworth filter
    fs = 1.0 / dtime
    nyquist = fs / 2.0
    normalized_cutoff = cutoff_freq / nyquist # Normalized cutoff frequency for high-pass filter

    b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False) # 4th order Butterworth filter
    n_samples = len(TOD)

    H_exact = get_filtfilt_matrix(n_samples, b, a) # Exact matrix representation of filtfilt operation
    HP_operator = H_exact @ Tsys_operator
    return HP_operator

# Define Wiener filter for mapmaking of the TOD
def wiener_filter_map(TOD, operator, noise_variance=None, prior_inv_cov=None, guess=None,
                      regularization=1e-12, return_full_cov=False):
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
    import numpy as np
    from scipy.sparse import diags
    from scipy.linalg import solve, LinAlgError
    
    # Convert inputs to numpy arrays
    TOD = np.asarray(TOD)
    operator = np.asarray(operator)
    
    n_time, n_pixels = operator.shape
    
    # Estimate noise variance if not provided
    if noise_variance is None:
        # Simple estimate: variance of high-pass filtered residuals
        residual = TOD - operator @ np.linalg.pinv(operator) @ TOD
        noise_variance = np.var(residual)
        print(f"Estimated noise variance: {noise_variance:.6f}")

    # Create noise inverse covariance matrix (assume diagonal)
    if np.isscalar(noise_variance):
        N_inv = np.eye(n_time) / noise_variance
    else:
        N_inv = diags(1.0 / np.asarray(noise_variance))
    
    # Create signal inverse covariance matrix
    if prior_inv_cov is None:
        S_inv = np.zeros((n_pixels, n_pixels))  # Uninformative prior
    elif np.isscalar(prior_inv_cov):
        S_inv = np.eye(n_pixels) * prior_inv_cov
    elif prior_inv_cov.ndim == 1:
        S_inv = diags(1.0 * np.asarray(prior_inv_cov))
    elif prior_inv_cov.ndim == 2:
        S_inv = np.asarray(prior_inv_cov)
    else:
        raise ValueError("prior_inv_cov must be a scalar, 1D array, or 2D array.")

    if guess is None:
        guess = np.zeros(n_pixels)
        S_inv = np.zeros((n_pixels, n_pixels))  # Uninformative prior
        if prior_inv_cov is not None:
            print("Warning: guess is None, ignoring provided prior_inv_cov and using uninformative prior.")
    else:
        guess = np.asarray(guess)
        if len(guess) != n_pixels:
            raise ValueError("Length of guess must match number of pixels.")

    # Compute Wiener filter components
    AtN = operator.T @ N_inv  # A^T N^-1
    AtNA = AtN @ operator     # A^T N^-1 A
    
    # Add signal prior and regularization
    covariance_inv = AtNA + S_inv + regularization * np.eye(n_pixels)
    
    # Right-hand side: A^T N^-1 d +  S^-1 mu
    rhs = AtN @ TOD + S_inv @ guess 

    try:
        # Solve the linear system: (A^T N^-1 A + S^-1) x = A^T N^-1 d +  S^-1 mu
        sky_map = solve(covariance_inv, rhs, assume_a='pos')
        
        # Compute uncertainties (diagonal of posterior covariance)
        try:
            posterior_cov = np.linalg.inv(covariance_inv)
            uncertainty = np.sqrt(np.diag(posterior_cov))
        except (LinAlgError, np.linalg.LinAlgError):
            print("Warning: Could not compute full covariance matrix. Using diagonal approximation.")
            uncertainty = 1.0 / np.sqrt(np.diag(covariance_inv))
            
    except (LinAlgError, np.linalg.LinAlgError) as e:
        print(f"Linear algebra error: {e}")
        print("Falling back to pseudo-inverse solution...")
        sky_map = np.linalg.pinv(operator) @ TOD
        uncertainty = np.ones(n_pixels) * np.nan
    
    if return_full_cov:
        return sky_map, uncertainty, posterior_cov
    else:
        return sky_map, uncertainty


# Alternative simplified version for quick mapmaking
def simple_wiener_map(TOD, operator, noise_var=None):
    """
    Simplified Wiener filter assuming uninformative signal prior.
    Equivalent to: (A^T A + lambda*I)^-1 A^T d
    """
    import numpy as np
    
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
        beam_map, 
        LST_deg_list_group, 
        lat_deg, 
        azimuth_deg_list_group, 
        elevation_deg_list_group, 
        threshold=0.01,
        Tsys_others_operator=None,
    ):
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

        lat_deg : float
            The latitude of the observation site in degrees.

        azimuth_deg_list_group : an azimuth list or a list of azimuth lists corresponding to each TOD in TOD_group.
            e.g. [azimuth_deg_list_1, azimuth_deg_list_2, ...]

        elevation_deg_list_group : an elevation list or a list of elevation lists corresponding to each TOD in TOD_group.
            e.g. [elevation_deg_list_1, elevation_deg_list_2, ...]

        threshold : float
            The threshold to cut off the fractional beam response np.abs(beam[pixel])/beam_max, default is 0.01.
            e.g., if threshold=0.01, only pixels with beam response larger than 1% of the maximum will be considered.

        Tsys_others_operator : array, optional
            The operator for other system temperature components (e.g., Trec and Tdiode) mapping to TOD.

        """

        # If LST_deg_list_group[0] is a list, flatten it.
        if isinstance(LST_deg_list_group[0], (list, np.ndarray)):
            self.num_tods = len(LST_deg_list_group)
            assert self.num_tods == len(azimuth_deg_list_group) == len(elevation_deg_list_group), \
                "Length of LST_deg_list_group, azimuth_deg_list_group, elevation_deg_list_group must be the same."
            LST_deg_list = np.concatenate(LST_deg_list_group)
            azimuth_deg_list = np.concatenate(azimuth_deg_list_group)
            elevation_deg_list = np.concatenate(elevation_deg_list_group)
        else:
            self.num_tods = 1
            LST_deg_list = LST_deg_list_group
            azimuth_deg_list = azimuth_deg_list_group
            elevation_deg_list = elevation_deg_list_group

        if beam_map.ndim == 1:
            self.npol = 1
        elif beam_map.ndim == 2:
            self.npol = beam_map.shape[0]
        else:
            raise ValueError("beam_map must be a 1D or 2D array.")

        if Tsys_others_operator is not None:
            self.Tsys_others = True
            self.n_params_others = Tsys_others_operator.shape[1]
        else:
            self.Tsys_others = False

        self.pixel_indices = truncate_stacked_beam(
            beam_map, LST_deg_list, lat_deg, azimuth_deg_list, elevation_deg_list, threshold=threshold
        )

        self.num_pixels = len(self.pixel_indices)
        self.nsky_params = self.npol * self.num_pixels

        
        if self.num_tods > 1:

            self.Tsys_operators = []
            null_operator = np.zeros_like(Tsys_others_operator) if Tsys_others_operator is not None else None

            for i in range(self.num_tods):
                LST_deg_list_i = LST_deg_list_group[i]
                azimuth_deg_list_i = azimuth_deg_list_group[i]
                elevation_deg_list_i = elevation_deg_list_group[i]

                sky_operator_i = generate_sky2sys_projection(
                    beam_map, LST_deg_list_i, lat_deg, azimuth_deg_list_i, elevation_deg_list_i, self.pixel_indices
                )
                other_operators = [null_operator]*self.num_tods 
                other_operators[i] = Tsys_others_operator 

                Tsys_operator_i = np.concatenate([sky_operator_i] + other_operators, axis=1) if Tsys_others_operator is not None else sky_operator_i
                self.Tsys_operators.append(Tsys_operator_i)

        else:
            sky_operators = generate_sky2sys_projection(
                beam_map, LST_deg_list, lat_deg, azimuth_deg_list, elevation_deg_list, self.pixel_indices
            )
            if Tsys_others_operator is not None:
                self.Tsys_operators = np.concatenate([sky_operators, Tsys_others_operator], axis=1)
            else:
                self.Tsys_operators = sky_operators

    def __call__(
        self,         
        TOD_group,
        gain_group,
        dtime,
        cutoff_freq_group,
        mu_group=None,
        Tsky_prior_mean=None,
        Tsky_prior_inv_cov_diag=None,
        Tsys_other_prior_mean_group=None,
        Tsys_other_prior_inv_cov_group=None,
        regularization=1e-12,
        return_full_cov=False,
    ):
        """
        TOD_group : a TOD array or a list of TOD arrays at the same frequency channel.
            e.g. [TOD_1, TOD_2, ...]

        gain_group : a gain array or a list of gain arrays corresponding to each TOD in TOD_group.
            e.g. [gain_1, gain_2, ...]
            gain_i can be a single value (constant gain) or an array with the same length as TOD_i.
            
        dtime : float
            Time interval between samples in seconds.

        cutoff_freq_group : list of float, 
            Cutoff frequency for high-pass filter in unit of the nyquist frequency.

        mu_group : a list of known system temperature components to be subtracted from Tsys (calibrated TOD),  each element corresponding to each TOD in TOD_group.
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

        Tsys_other_prior_inv_cov_group : a list of prior inverse covariances for other system temperature components, each element corresponding to each TOD in TOD_group.
            e.g. [Tsys_other_prior_inv_cov_1, Tsys_other_prior_inv_cov_2, ...]
            The shape of each element can be:
                (num_other_params,) : Diagonal of the inverse covariance matrix.
                (num_other_params, num_other_params) : Full inverse covariance matrix. 
            But all elements must have the same shape.
            If None, assumed to be uninformative prior (zero, i.e., infinite prior variance).


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


        for i in range(self.num_tods):
            hp_operator_i = HP_filter_TOD(TOD_group[i], dtime, self.Tsys_operators[i], cutoff_freq=cutoff_freq_group[i])
            calibrated_TOD_i = TOD_group[i] / gain_group[i] 
            if mu_group is not None:
                calibrated_TOD_i -= mu_group[i]
            hp_cal_TOD_i = hp_operator_i @ calibrated_TOD_i
            hp_Tsys_operator_i = hp_operator_i @ self.Tsys_operators[i]

            if i == 0:
                HP_Tsys_operator_overall = hp_Tsys_operator_i
                HP_cal_TOD_overall = hp_cal_TOD_i
            else:
                HP_Tsys_operator_overall = np.concatenate([HP_Tsys_operator_overall, hp_Tsys_operator_i])
                HP_cal_TOD_overall = np.concatenate([HP_cal_TOD_overall, hp_cal_TOD_i])

        self.nparams = HP_Tsys_operator_overall.shape[1]

        # Construct prior mean for all parameters
        Tsys_prior_mean = np.zeros(self.nparams)
        if Tsky_prior_mean is not None:
            assert len(Tsky_prior_mean) == self.nsky_params, "Length of Tsky_prior_mean must match number of sky parameters."
            Tsys_prior_mean[:self.nsky_params] = Tsky_prior_mean
        counter = self.nsky_params
        if Tsys_other_prior_mean_group is not None:
            assert self.Tsys_others, "Tsys_others_operator must be provided in initialization if Tsys_other_prior_mean_group is provided."
            assert len(Tsys_other_prior_mean_group) == self.num_tods, "Length of Tsys_other_prior_mean_group must match number of TODs."
            for Tsys_other_prior_mean_i in Tsys_other_prior_mean_group:
                Tsys_prior_mean[counter:counter+len(Tsys_other_prior_mean_i)] = Tsys_other_prior_mean_i
                counter += len(Tsys_other_prior_mean_i)

        # Construct prior inverse covariance matrix for all parameters
        Tsys_prior_inv_cov = np.zeros((self.nparams, self.nparams)) 
        if Tsky_prior_inv_cov_diag is not None:
            Tsky_prior_inv_cov_diag = np.asarray(Tsky_prior_inv_cov_diag).reshape(-1) # flatten
            assert len(Tsky_prior_inv_cov_diag) == self.nsky_params, "Length of Tsky_prior_inv_cov_diag must match number of sky parameters."
            Tsys_prior_inv_cov[:self.nsky_params, :self.nsky_params] = np.diag(Tsky_prior_inv_cov_diag)

        counter = self.nsky_params
        if Tsys_other_prior_inv_cov_group is not None:
            assert self.Tsys_others, "Tsys_others_operator must be provided in initialization if Tsys_other_prior_inv_cov_group is provided."
            assert len(Tsys_other_prior_inv_cov_group) == self.num_tods, "Length of Tsys_other_prior_inv_cov_group must match number of TODs."
            
            for Tsys_other_prior_inv_cov_i in Tsys_other_prior_inv_cov_group:
                if Tsys_other_prior_inv_cov_group[0].ndim == 1:
                    n_others = len(Tsys_other_prior_inv_cov_i)
                    Tsys_prior_inv_cov[counter:counter+n_others, counter:counter+n_others] = np.diag(Tsys_other_prior_inv_cov_i)
                    counter += n_others
                elif Tsys_other_prior_inv_cov_group[0].ndim == 2:
                    n_others = Tsys_other_prior_inv_cov_i.shape[0]
                    Tsys_prior_inv_cov[counter:counter+n_others, counter:counter+n_others] = Tsys_other_prior_inv_cov_i
                    counter += n_others
                else:
                    raise ValueError("Each element in Tsys_other_prior_inv_cov_group must be a 1D or 2D array.")

    
        # Apply Wiener filter with the overall operator
        estmation, uncertainty = wiener_filter_map(
            HP_cal_TOD_overall, 
            HP_Tsys_operator_overall, 
            noise_variance=None, # estimated from TOD, rather than provided
            prior_inv_cov=Tsys_prior_inv_cov, 
            guess=Tsys_prior_mean,
            regularization=regularization,
            return_full_cov=return_full_cov,
        )

        sky_estimation = estmation[:self.nsky_params]
        sky_uncertainty = uncertainty[:self.nsky_params]

        if self.npol > 1:
            sky_estimation = sky_estimation.reshape(self.npol, self.num_pixels)
            sky_uncertainty = sky_uncertainty.reshape(self.npol, self.num_pixels)

        if self.Tsys_others:
            Tsys_others_estimation_group = []
            Tsys_others_uncertainty_group = []
            counter = self.nsky_params
            for i in range(self.num_tods):
                Tsys_others_estimation_group.append(estmation[counter:counter+self.n_params_others])
                Tsys_others_uncertainty_group.append(uncertainty[counter:counter+self.n_params_others])
                counter += self.n_params_others
            return sky_estimation, sky_uncertainty, Tsys_others_estimation_group, Tsys_others_uncertainty_group
        else:
            return sky_estimation, sky_uncertainty
        
