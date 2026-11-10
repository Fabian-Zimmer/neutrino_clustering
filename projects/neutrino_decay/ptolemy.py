"""PTOLEMY beta-decay spectra and capture rates, including neutrino decay."""


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid as trap
import chex

from cnb.physics import Physics


@chex.dataclass
class Ptolemy:

    """
    The Equations are referring to the Akita et al. (2022) paper:
    http://arxiv.org/abs/2109.02900
    """

    @staticmethod
    @jax.jit
    def Gamma_i_to_j(m_i, lambda_ij, args):
        """Eqn. (3.3)."""
        return 1/args.t0 * m_i/(50*args.meV) * (lambda_ij/6.2e-16)**2


    @staticmethod
    @jax.jit
    def n_i_today(m_i, lambda_ij, f_c, n0_i, args):
        """Eqn. (3.4)"""
        return jnp.exp(- m_i/(50*args.meV) * (lambda_ij/6.2e-16)**2) * f_c * n0_i
    

    @staticmethod
    @jax.jit
    def p_j_z(p_j, z):
        """Upper eqn. of (4.6)."""
        return p_j*(1.+z)
    

    @staticmethod
    @jax.jit
    def E_j_z(p_j, m_j, z):
        """Lower eqn. of (4.6)."""
        return jnp.sqrt(p_j**2*(1.+z)**2 + m_j**2)


    @staticmethod
    @jax.jit
    def p_star(m_i, m_j, m_x):
        """Upper eqn. of (4.11)."""
        return 1/2/m_i*jnp.sqrt((m_i**2 - (m_j + m_x)**2)*(m_i**2 - (m_j - m_x)**2))


    @staticmethod
    @jax.jit
    def E_star(m_i, m_j, m_x):
        """Lower eqn. of (4.11)."""
        return (m_i**2 + m_j**2 - m_x**2)/(2*m_i)


    @staticmethod
    @jax.jit
    def E_j_from_E_e(E_e, args):
        """Upper eqn. of (5.5)."""
        return E_e - args.K0_end - args.m_e


    @staticmethod
    @jax.jit
    def E_e_from_E_j(E_j, args):
        """Inverse of upper eqn. of (5.5)."""
        return E_j + args.K0_end + args.m_e


    @staticmethod
    @jax.jit
    def Ei_end(m_i, args):
        """Eqn. (5.9)"""
        return args.K0_end + args.m_e - m_i


    @staticmethod
    @jax.jit
    def E_end(m_lightest, args):
        """Eqn. (5.10)"""
        return args.K0_end + args.m_e - m_lightest

    @staticmethod
    @jax.jit
    def Q_value(m_nu, args):
        return args.m_3H - args.m_3He - args.m_e - m_nu


    @staticmethod
    @jax.jit
    def K_end(m_nu, args):
        return ((args.m_3H - args.m_e)**2 - (m_nu - args.m_3He)**2)/(2*args.m_3H)


    @staticmethod
    @jax.jit
    def H_factor(E_e, m_i, args):
        """Eqn. (5.8)"""
        
        numer1 = 1 - (args.m_e**2/(E_e*args.m_3H))
        denom1 = (1 - (2*E_e/args.m_3H) + (args.m_e**2/args.m_3H**2))**2
        term1 = numer1/denom1
        
        # set negative values to zero
        y_i = jnp.clip(Ptolemy.Ei_end(m_i, args) - E_e, 0., None)
        
        term2 = jnp.sqrt(y_i * (y_i + (2*m_i*args.m_3He/args.m_3H)))
        term3 = y_i + m_i/args.m_3H*(args.m_3He + m_i)
        
        return term1*term2*term3
        

    @staticmethod
    def Beta_decay_spectrum(E_e, nu_masses, ordering, args):
        """Eqn. (5.7)"""

        # H factors for each neutrino mass.
        H_vals = jnp.array([Ptolemy.H_factor(E_e, m_i, args) for m_i in nu_masses])

        # dGamma_beta/dE_e.
        def normal_ordering(_):
            return jnp.tile(args.U_ei_AbsSq_NO, (len(E_e),1)).T
        
        def inverted_ordering(_):
            return jnp.tile(args.U_ei_AbsSq_IO, (len(E_e),1)).T

        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")

        branches = [normal_ordering, inverted_ordering]
        U_ei_tiled = jax.lax.switch(ordering_index[ordering], branches, None)

        return args.sigma_avg/args.Pi**2*args.N_T * jnp.sum(U_ei_tiled*H_vals, axis=0)


    @staticmethod
    @jax.jit
    def dndE_decay(E_j, m_j, m_i, m_x, tau_i, n0_i, args):
        """Eqn. (4.13)."""

        def true_branch(_):
            return jnp.array(0.0)

        def false_branch(_):
            zE = (Ptolemy.p_star(m_i, m_j, m_x) / jnp.sqrt(E_j**2 - m_j**2)) - 1
            tE = Physics.time_z(zE, args)
            const_term = n0_i * jnp.exp(-tE / tau_i) / (Physics.Hubble_z(zE, args) * tau_i)
            energy_term = E_j / (E_j**2 - m_j**2)
            return const_term * energy_term

        E_star = Ptolemy.E_star(m_i, m_j, m_x)
        condition = (E_j <= m_j) | (E_j > E_star)

        return jax.lax.cond(condition, true_branch, false_branch, None)


    @staticmethod
    def dGammaTilde_CNB_dE_e(E_e, m_j, m_i, m_x, tau_i, n0_i, ordering, args):
        """Eqn. (5.6)"""

        def normal_ordering(_):
            return 2*args.U_ei_AbsSq_NO[0]*args.sigma_avg*args.N_T
        
        def inverted_ordering(_):
            return 2*args.U_ei_AbsSq_IO[0]*args.sigma_avg*args.N_T
        
        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")
        
        branches = [normal_ordering, inverted_ordering]
        pre = jax.lax.switch(ordering_index[ordering], branches, None)

        E_j = Ptolemy.E_j_from_E_e(E_e, args)
        dnj_dEj = jnp.array([
            Ptolemy.dndE_decay(E, m_j, m_i, m_x, tau_i, n0_i, args) for E in E_j
        ])

        return pre*dnj_dEj
    

    @staticmethod
    def dndE_Fermi_Dirac(nu_masses, p_start, p_stop, p_num, args):

        # Momentum range determined by input parameters
        p_FD = jnp.geomspace(p_start, p_stop, p_num)
        
        # Compute Fermi-Dirac values
        f_FD = Physics.Fermi_Dirac(p_FD, args)

        # Neutrino number per momentum interval
        dndp_FD = p_FD**2 * f_FD

        # Get corresponding energy range
        E_FD = Physics.Energy_Momentum_conversion(p_FD, nu_masses, 'p_to_E')

        # Get velocities via momentum-energy relation
        v_FD = p_FD / E_FD

        # Neutrino number per energy interval
        E_sqrt = jnp.sqrt(E_FD**2 - nu_masses[:,None]**2)
        dndE_FD = E_FD * E_sqrt * f_FD

        return p_FD, dndp_FD, E_FD, dndE_FD, v_FD


    @staticmethod
    def dGamma_CNB_dE_e_FD(nu_masses, ordering, nature, p_start, p_stop, p_num, args):

        def normal_ordering(_):
            return nature*args.U_ei_AbsSq_NO*args.sigma_avg*args.N_T
        
        def inverted_ordering(_):
            return nature*args.U_ei_AbsSq_IO*args.sigma_avg*args.N_T
        
        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")
        
        # Choose ordering and determine appropriate prefactors
        branches = [normal_ordering, inverted_ordering]
        pre = jax.lax.switch(ordering_index[ordering], branches, None)
        phase_space_factor = args.g_nu / (2 * args.Pi**2)

        # Get Fermi-Dirac energy spectra
        *_, E_FD, dndE_FD, _ = Ptolemy.dndE_Fermi_Dirac(
            nu_masses, p_start, p_stop, p_num, args)

        return E_FD, phase_space_factor * pre[:,None] * dndE_FD
    

    @staticmethod
    def load_neutrino_velocities(halo_ID:str):

        # Neutrino vectors of one halo, stored in 8 batch files in data/.
        file_name = f'neutrino_vectors_numerical_{halo_ID}'

        nu_vels = jnp.array([
            jnp.load(f'data/{file_name}_batch{i+1}.npy')[...,3:6]
            for i in range(8)
        ]).reshape(-1,100,3)

        return nu_vels


    @staticmethod
    def dndE_simulation(halo_ID, nu_masses, args):

        # Load velocity data, units get taken care of in conversion to momenta
        v_sim = Ptolemy.load_neutrino_velocities(halo_ID)

        # Convert velocities to momenta
        p_sim, y_sim = Physics.velocities_to_momenta(v_sim, nu_masses, args)

        # Select neutrino with highest Fermi-Dirac value of each momentum batch
        p_back, y_init = Physics.filter_momenta(p_sim, y_sim, len(nu_masses), args)
        p_init = y_init*args.T_CNB

        # Compute Fermi-Dirac values of today, using p_back and Liouville's theorem
        f_today = Physics.Fermi_Dirac(p_back, args)

        # Neutrino number per momentum interval
        dndp_init = p_init**2 * f_today

        # Get corresponding energy range
        E_init = jnp.sqrt(p_init**2 + nu_masses[:, None]**2)

        # Neutrino number per energy interval
        E_sqrt = jnp.sqrt(E_init**2 - nu_masses[:, None]**2)
        dndE_init = E_init * E_sqrt * f_today

        return p_init, dndp_init, E_init, dndE_init
    

    @staticmethod
    def dGamma_CNB_dE_e_simulation(halo_ID, nu_masses, ordering, nature, args):

        def normal_ordering(_):
            return nature*args.U_ei_AbsSq_NO*args.sigma_avg*args.N_T
        
        def inverted_ordering(_):
            return nature*args.U_ei_AbsSq_IO*args.sigma_avg*args.N_T
        
        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")
        
        # Choose ordering and determine appropriate prefactors
        branches = [normal_ordering, inverted_ordering]
        pre = jax.lax.switch(ordering_index[ordering], branches, None)
        phase_space_factor = args.g_nu / (2 * args.Pi**2)

        # Get CNB simulation energy spectra
        *_, E_sim, dndE_sim = Ptolemy.dndE_simulation(
            halo_ID, nu_masses, args)

        return E_sim, phase_space_factor * pre[:,None] * dndE_sim


    @staticmethod
    def analyze_peak_for_X_range_resolution(data, points_in_FWHM):
        # Finding the peak
        peak_value = jnp.max(data)
        peak_idx = jnp.argmax(data)

        # Calculating the Full Width at Half Maximum (FWHM)
        half_max = peak_value / 2
        left_idx = jnp.where(data[:peak_idx] < half_max)[0][-1]
        right_idx = jnp.where(data[peak_idx:] < half_max)[0][0] + peak_idx
        FWHM = right_idx - left_idx

        # Determining the resolution for the linear x-axis
        total_range = len(data)
        resolution = int((total_range / FWHM) * points_in_FWHM) + 1

        return resolution, peak_idx, FWHM


    @staticmethod
    def analyze_peak_for_energy_step(data_x, data_y, points_in_FWHM):
        """
        Analyzes the peak in the spectrum to determine the energy step size for a given resolution.

        :param data_x: The original energy values (x-axis).
        :param data_y: The corresponding spectrum values (y-axis).
        :param points_in_FWHM: The number of points desired within the FWHM.
        :return: The necessary energy step size.
        """
        # Finding the peak
        peak_value = jnp.max(data_y)
        peak_idx = jnp.argmax(data_y)

        # Calculating the Full Width at Half Maximum (FWHM)
        half_max = peak_value / 2
        left_idx = jnp.where(data_y[:peak_idx] < half_max)[0][-1]
        right_idx = jnp.where(data_y[peak_idx:] < half_max)[0][0] + peak_idx

        # Energy values at FWHM
        E_FWHM_left = data_x[left_idx]
        E_FWHM_right = data_x[right_idx]

        # Determining the energy step size
        FWHM_energy_range = E_FWHM_right - E_FWHM_left
        energy_step = FWHM_energy_range / points_in_FWHM

        return energy_step, E_FWHM_left, E_FWHM_right


    @staticmethod
    def analyze_peak_and_determine_extended_resolution(data, points_in_FWHM, E_start_orig, E_stop_orig, E_min, E_max):
        """
        Determines the resolution for an extended linear energy range based on the FWHM of the original data.

        :param data: The array containing the data.
        :param points_in_FWHM: The number of points desired within the FWHM.
        :param E_start_orig: The starting energy of the original data range.
        :param E_stop_orig: The stopping energy of the original data range.
        :param E_min: The minimum energy of the extended range.
        :param E_max: The maximum energy of the extended range.
        :return: The number of points for the extended linear energy range.
        """
        # Finding the peak and FWHM in the original data
        peak_value = jnp.max(data)
        peak_idx = jnp.argmax(data)
        half_max = peak_value / 2
        left_idx = jnp.where(data[:peak_idx] < half_max)[0][-1]
        right_idx = jnp.where(data[peak_idx:] < half_max)[0][0] + peak_idx
        FWHM = right_idx - left_idx

        # Original energy range and step size
        original_range = E_stop_orig - E_start_orig
        step_size = original_range / FWHM

        # Number of points in the original range to achieve the desired resolution
        num_points_original = int((original_range / step_size) * points_in_FWHM)

        # Applying the same step size to the extended range
        extended_range = E_max - E_min
        num_points_extended = int(extended_range / step_size) + 1

        return num_points_extended, num_points_original, FWHM


    @staticmethod
    def gpr_interpolation(x_original, y_original, x_interpolated):
        """
        Interpolates irregularly spaced data using Gaussian Process Regression.
        """
        
        # Reshaping for compatibility with GPR
        x_original_reshaped = jnp.array(x_original).reshape(-1, 1)
        x_interpolated_reshaped = jnp.array(x_interpolated).reshape(-1, 1)

        # Matérn kernel with a specified nu
        kernel = Matern(
            nu=1.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # Fitting the model
        gp.fit(x_original_reshaped, y_original)

        # Predicting the interpolated data
        y_interpolated, sigma = gp.predict(
            x_interpolated_reshaped, return_std=True)
        
        return y_interpolated, sigma
    

    @staticmethod
    @jax.jit
    def Gaussian_kernel(X_range, mu, sigma, args):

        # Normalization for Gaussian
        norm = jnp.sqrt(2*args.Pi)*sigma  

        return jnp.exp(-(X_range - mu)**2 / (2 * sigma**2)) / norm


    @staticmethod
    @jax.jit
    def Gaussian_convolution_1D(X_range, array_to_convolve, Gaussian_sigma, args):

        # X-range differences for the Gaussian in the convolution integral
        X_diff = X_range[:, None] - X_range[None, :]

        # Functions to be convolved
        f1 = array_to_convolve[:, None]
        g1 = Ptolemy.Gaussian_kernel(X_diff, Gaussian_sigma, args)

        # Convolution by integration
        array_convolved = trap(f1 * g1, x=X_range, axis=0)
        
        # # Find peak values
        # peak_original = jnp.max(array_to_convolve)
        # peak_convolved = jnp.max(array_convolved)

        # # Adjust, i.e. normalize, convolved spectrum based on peak values
        # adjustment_factor = peak_original / peak_convolved
        # array_convolved_normalized = array_convolved * adjustment_factor
        

        # # Find areas under curves
        # area_original = trap(array_to_convolve, x=X_range)
        # area_convolved = trap(array_convolved, x=X_range)

        # # # Adjust, i.e. normalize, convolved spectrum based on areas
        # adjustment_factor = area_original / area_convolved
        # array_convolved_normalized = array_convolved * adjustment_factor

        array_convolved_normalized = array_convolved

        return array_convolved_normalized


    @staticmethod
    @jax.jit
    def fourier_convolution(signal, kernel):
        """
        Convolve a signal with a kernel via Fourier Transforms.
        Assumes kernel is already prepared (centered and optionally padded).
        """
        # Shift the kernel to align the zero-frequency component (representing
        # the average) with the first array element
        shifted_kernel = jnp.fft.ifftshift(kernel)

        # indices = jnp.arange(shifted_kernel.size)
        # middle_index = shifted_kernel.size // 2
        # shifted_kernel = jnp.where(indices >= middle_index, 0.0, shifted_kernel)

        # Fourier Transform of the signal and the shifted kernel
        signal_fft = jnp.fft.fft(signal)
        kernel_fft = jnp.fft.fft(shifted_kernel, n=len(signal))

        # Convolution in the frequency domain
        convolved_fft = signal_fft * kernel_fft

        # Inverse Fourier Transform
        convolved_signal = jnp.fft.ifft(convolved_fft)

        # Taking the real part of the result and normalize
        normalized_signal = jnp.real(convolved_signal) / jnp.sum(kernel)

        return normalized_signal


    @staticmethod
    @jax.jit
    def ZhangZhang2018_overdensity(m_nu):
        return 76.5 * m_nu**2.21
