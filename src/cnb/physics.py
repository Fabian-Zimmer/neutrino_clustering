"""Physics of relic neutrinos: Hubble rate, momenta, Fermi-Dirac phase space,
number densities."""


import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import healpy as hp
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.integrate import trapezoid as trap
import chex

from cnb.units import Params


@chex.dataclass
class Physics:

    """
    Cosmological functions.
    """

    @staticmethod
    @jax.jit
    def Hubble_z(z, args):
        """Evolution of Hubble parameter valid at low redshift."""
        return args.H0*jnp.sqrt(args.Omega_M*(1.+z)**3 + args.Omega_L)
    

    @staticmethod
    @jax.jit
    def time_z(z, args):
        """
        Time of the universe at redshift z, when dark matter and dark energy are 
        dominating and rest is negligible. Eqn. (4.9).
        """
        r_z = args.Omega_L/args.Omega_M * 1/(1.+z)**3
        return 2/(3*args.H0*jnp.sqrt(args.Omega_L)) * jnp.log(jnp.sqrt(r_z) + jnp.sqrt(1+r_z))


    @staticmethod
    def s_of_z(z, h, Omega_M, Omega_L):
        """
        Convert redshift to time variable s with eqn. 4.1 in Mertsch et al.
        (2020), keeping only Omega_M and Omega_L in the Hubble eqn. for H(z).
        Returns s in seconds (as a plain number). The simulation box uses its
        own cosmology (box_parameters.yaml), solar runs use Params (Planck18).
        """

        # We need value of H0 in units of 1/s.
        H0_val = h*100*Params.km/Params.s/Params.Mpc / (1/Params.s)

        def s_integrand(z):
            a_dot = np.sqrt(Omega_M*(1.+z)**3 + Omega_L)/(1.+z)*H0_val
            return 1./a_dot

        s_of_z, _ = quad(s_integrand, 0., z)

        return np.float64(s_of_z)


    @staticmethod
    def neutrino_masses(m_lightest, ordering, args):
        """Returns the 3 neutrino masses with given ordering and lightest mass."""

        def normal_ordering(_):
            m1 = m_lightest
            m2 = jnp.sqrt(m1**2 + args.Del_m21_Sq)
            m3 = jnp.sqrt(m1**2 + args.Del_m31_Sq_NO)
            return jnp.array([m1, m2, m3])

        def inverted_ordering(_):
            m3 = m_lightest
            m2 = jnp.sqrt(m3**2 + args.Del_m32_Sq_IO)
            m1 = jnp.sqrt(m2**2 - args.Del_m21_Sq)
            return jnp.array([m3, m1, m2])

        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")

        # Switch based on the ordering
        branches = [normal_ordering, inverted_ordering]
        return jax.lax.switch(ordering_index[ordering], branches, None)
    

    @jax.jit
    def velocities_to_momenta_ND_halo_0th_axis(v_arr, m_arr, args):
        """
        Converts velocities to magnitude of momentum [eV] and ratio y=p/T_CNB, 
        according to desired target mass (and mass used in simulation).
        """

        # Magnitudes of velocity
        mags_arr = jnp.linalg.norm(v_arr, axis=-1)
        
        # Reshape m_arr to have the right number of singleton dimensions
        new_shape = (1,) + (m_arr.shape[0],) + (1,) * (jnp.ndim(mags_arr)-1)
        m_sync = jnp.reshape(m_arr, new_shape)

        # From velocity (magnitude) in kpc/s to momentum in eV
        v_unit = args.kpc/args.s
        p_dim = mags_arr[:, None, ...]*v_unit * m_sync

        # p/T_CNB ratio
        y = p_dim / args.T_CNB

        return p_dim, y


    @jax.jit
    def velocities_to_momentum_vec_ND_halo_0th_axis(v_arr, m_arr, args):
        
        # Reshape m_arr to have the right number of singleton dimensions
        new_shape = (1,) + (m_arr.shape[0],) + (1,) * (jnp.ndim(v_arr)-1)
        m_sync = jnp.reshape(m_arr, new_shape)

        # From velocity (vector) in kpc/s to momentum in eV
        v_unit = args.kpc/args.s

        # Scaling the velocity vector by mass and unit conversion
        p_vec = v_arr[:, None, ...]*v_unit * m_sync  

        # Compute magnitude of momentum vectors
        p_dim = jnp.linalg.norm(p_vec, axis=-1)

        # p/T_CNB ratio
        y = p_dim / args.T_CNB

        return p_vec, p_dim, y


    @jax.jit
    def velocities_to_momenta(v_arr, m_arr, args):
        """
        Converts velocities to magnitude of momentum [eV] and ratio y=p/T_CNB, 
        according to desired target mass (and mass used in simulation).
        """

        # Magnitudes of velocity.
        mags_arr = jnp.linalg.norm(v_arr, axis=-1)
        
        # From velocity (magnitude) in kpc/s to momentum in eV.
        p_dim = mags_arr[None, :, :]*(args.kpc/args.s) * m_arr[:, None, None]

        # p/T_CNB ratio.
        y = p_dim / args.T_CNB

        return p_dim, y
    

    @jax.jit
    def velocities_to_momenta_all_sky(v_arr, m_arr, args):
        """
        Converts velocities to magnitude of momentum [eV] and ratio y=p/T_CNB, 
        according to desired target mass (and mass used in simulation). But now assuming the input array to have starting dimentions of (Npix, nu_per_pix), instead of just the total number of neutrinos (N,).
        """

        # Magnitudes of velocity.
        mags_arr = jnp.linalg.norm(v_arr, axis=-1)
        
        # From velocity (magnitude) in kpc/s to momentum in eV.
        p_dim = mags_arr[None,...]*(args.kpc/args.s) * m_arr[:,None,None,None]

        # p/T_CNB ratio.
        y = p_dim / args.T_CNB

        return p_dim, y


    @jax.jit
    def velocities_to_momenta_1D(v_arr, m_arr, args):

        # Magnitudes of velocity.
        mags_arr = jnp.linalg.norm(v_arr, axis=-1)
        
        # From velocity (magnitude) in kpc/s to momentum in eV.
        p_dim = mags_arr[None, :]*(args.kpc/args.s) * m_arr[:, None]

        # p/T_CNB ratio.
        y = p_dim / args.T_CNB

        return p_dim, y


    @jax.jit
    def velocities_to_momenta_individual(v_arr, m_arr, args):
        
        # Reshape m_arr to have the right number of singleton dimensions
        new_shape = (1,) + (m_arr.shape[0],) + (1,) * (jnp.ndim(v_arr)-1)
        m_sync = jnp.reshape(m_arr, new_shape)

        # From velocity (magnitude) in kpc/s to momentum in eV
        v_unit = args.kpc/args.s
        p_dim = v_arr[:, None, ...]*v_unit * m_sync

        # p/T_CNB ratio.
        y = p_dim / args.T_CNB

        return p_dim, y


    @staticmethod
    def filter_momenta(p_arr, y_arr, m_len, args):

        # Select necessary sub-arrays
        p_init = p_arr[...,0]
        p_back = p_arr[...,-1]
        y_init = y_arr[...,0]

        # Sort
        sort_indices = p_init.argsort(axis=-1)
        p_back_sort = jnp.take_along_axis(p_back, sort_indices, axis=-1)
        y_init_sort = jnp.take_along_axis(y_init, sort_indices, axis=-1)

        # Select "most likely", i.e. most clustered neutrinos, of each velocity batch
        p_back_blocks = p_back_sort.reshape((m_len, args.p_num, args.phis*args.thetas))
        p_back_select = jnp.min(p_back_blocks, axis=-1)
        y_init_blocks = y_init_sort.reshape((m_len, args.p_num, args.phis*args.thetas))
        y_init_select = y_init_blocks[...,0]

        return p_back_select, y_init_select
    

    @staticmethod
    def SHM_PSD(v_arr, v_esc_MW, v_0, args):

        v_for_f_mag = jnp.linalg.norm(v_arr, axis=-1)

        # Create mask for velocity magnitudes smaller than escape velocity
        mask = v_for_f_mag < v_esc_MW

        # Compute standard halo model (SHM) distribution
        exp_term = jnp.exp(-v_for_f_mag**2/v_0**2)
        f_v = jnp.where(
            mask, 
            jnp.power(jnp.pi*v_0**2, -3/2) * exp_term, 
            jnp.zeros_like(exp_term))
        
        # Normalisation constant, and normalised velocity distribution
        z = v_esc_MW/v_0
        N_esc = jsp.special.erf(z) - 2/jnp.sqrt(jnp.pi)*z*jnp.exp(-z**2)
        f_v_normalized = f_v / N_esc

        return args.N0 * f_v_normalized

    @staticmethod
    @jax.jit
    def Fermi_Dirac(p, args):
        return 1/(jnp.exp(p/(args.T_CNB)) + 1)
    

    @jax.jit
    def number_density(p_init, p_back, pix_sr, args):
      
        """Neutrino number density obtained by integration over initial momenta.

        Args:
            p_init (array): neutrino momentum today
            p_back (array): neutrino momentum at z_back (final redshift in sim.)

        Returns:
            array: Value of relic neutrino number density in (1/cm^3).
        """    

        # note: trapz integral method needs sorted (ascending) "x-axis" array.
        ind = p_init.argsort(axis=-1)
        p_init_sort = jnp.take_along_axis(p_init, ind, axis=-1)
        p_back_sort = jnp.take_along_axis(p_back, ind, axis=-1)

        # Fermi-Dirac values with momenta at end of sim
        FD_arr = Physics.Fermi_Dirac(p_back_sort, args)

        # Calculate number density
        y = p_init_sort**3 * FD_arr  # dlog integrand
        x = p_init_sort
        n_raw = trap(y, jnp.log(x), axis=-1)

        # Multiply by constants and/or solid angles and convert to 1/cm**3.
        n_cm3 = pix_sr * args.g_nu/((2*args.Pi)**3) * n_raw / (1/args.cm**3)

        return jnp.array(n_cm3)
    

    @staticmethod
    @jax.jit
    def Fermi_Dirac_boosted(p_vec, boost_vec, args):
        """Fermi-Dirac with direction-dependent effective temperature"""
        v_mag = jnp.sqrt(jnp.sum(boost_vec**2))
        gamma = 1/jnp.sqrt(1 - v_mag**2)
        
        # Compute cos(theta) between momentum and boost vectors
        p_mag = jnp.sqrt(jnp.sum(p_vec**2, axis=-1))
        cos_theta = jnp.sum(p_vec * boost_vec, axis=-1)/(p_mag * v_mag)
        
        # Effective temperature factor
        T_factor = gamma * (1 - v_mag * cos_theta)
        
        return 1/(jnp.exp(p_mag/(args.T_CNB / T_factor)) + 1)


    @jax.jit
    def number_density_boosted(p_init_mag, p_back_vec, boost_vec, pix_sr, args):
      
        """Neutrino number density obtained by integration over initial momenta.

        Args:
            p_init (array): neutrino momentum today
            p_back (array): neutrino momentum at z_back (final redshift in sim.)

        Returns:
            array: Value of relic neutrino number density in (1/cm^3).
        """

        # Fermi-Dirac values with momenta at end of sim
        FD_arr = Physics.Fermi_Dirac_boosted(p_back_vec, boost_vec, args)

        # Calculate number density
        y = p_init_mag**3 * FD_arr  # dlog integrand
        x = p_init_mag
        n_raw = trap(y, jnp.log(x), axis=-1)

        # Multiply by constants and/or solid angles and convert to 1/cm**3.
        n_cm3 = pix_sr * args.g_nu/((2*args.Pi)**3) * n_raw / (1/args.cm**3)

        return jnp.array(n_cm3)


    @jax.jit
    def Fermi_Dirac_Delta(p_arr, Deltas, args):
        temp_Deltas = args.T_CNB * (1 + Deltas)
        return 1 / (jnp.exp(p_arr / temp_Deltas) + 1)


    @jax.jit
    def number_density_Delta(p_init, p_back, Deltas, pix_sr, args):
      
        """Neutrino number density obtained by integration over initial momenta.

        Args:
            p_init (array): neutrino momentum today
            p_back (array): neutrino momentum at z_back (final redshift in sim.)

        Returns:
            array: Value of relic neutrino number density in (1/cm^3).
        """    

        # note: trapz integral method needs sorted (ascending) "x-axis" array.
        ind = p_init.argsort(axis=-1)
        p_init_sort = jnp.take_along_axis(p_init, ind, axis=-1)
        p_back_sort = jnp.take_along_axis(p_back, ind, axis=-1)

        # Also sort temp. perturbation array to match new order of momenta
        Deltas_sort = jnp.take_along_axis(Deltas, ind, axis=-1)

        # Fermi-Dirac values with momenta at end of sim
        FD_arr_Delta = Physics.Fermi_Dirac_Delta(p_back_sort, Deltas_sort, args)

        # Calculate number density
        y = p_init_sort**3 * FD_arr_Delta  # dlog integrand
        x = p_init_sort
        n_raw = trap(y, jnp.log(x), axis=-1)

        # Multiply by constants and/or solid angles and convert to 1/cm**3.
        n_cm3 = pix_sr * args.g_nu/((2*args.Pi)**3) * n_raw / (1/args.cm**3)

        return n_cm3




    @jax.jit
    def transform_momenta_to_orig_frame(p_vec, boost_vec, masses=None):
        """
        Transform momentum vectors to original frame using proper Lorentz transformation
        
        Args:
            p_vec: Momentum vectors with shape (..., 3)
            boost_vec: Boost velocity vector (3,)
            masses: Mass values for the second dimension
        """

        # Calculate boost magnitude squared
        boost_mag_squared = jnp.sum(boost_vec**2)
        
        # Check if boost is effectively zero
        # For jax, we can use jax.lax.cond for conditional execution
        def zero_boost_case(_):
            p_vec_mag = jnp.sqrt(jnp.sum(p_vec**2, axis=-1))
            return p_vec_mag, p_vec / p_vec_mag[..., None]
        
        def normal_boost_case(_):
            # Calculate Lorentz factor
            boost_mag = jnp.sqrt(jnp.sum(boost_vec**2))
            Lf = 1/jnp.sqrt(1 - boost_mag**2)  # Lorentz factor
            
            # Calculate momentum magnitude
            p_mag = jnp.sqrt(jnp.sum(p_vec**2, axis=-1))
            
            # Calculate energy
            if masses is not None:
                # Ensure masses is properly dimensioned for this specific p_vec
                # If p_vec has shape (1, 5, 768, 1000, 3) and masses has shape (5,)
                # We want to create energy with shape (1, 5, 768, 1000)
                E = jnp.sqrt(p_mag**2 + jnp.reshape(masses, (1, -1, 1, 1))**2)
            else:
                # Massless case (E = p)
                E = p_mag
            
            # Dot product of momentum and boost velocity
            # p_dot_v = jnp.sum(p_vec * boost_vec, axis=-1)

            # Calculate p⃗' = p⃗ + (γ-1)(p⃗·v̂)v̂/v² - γEv⃗
            # term1 = p_vec
            # term2 = (Lf-1) * (p_dot_v / boost_mag**2)[..., None] * boost_vec
            # term3 = -Lf * E[..., None] * boost_vec
            # p_orig_vec = term1 + term2 + term3

            # Calculate p⃗' = γ(p⃗ - Ev⃗)
            p_orig_vec = Lf * (p_vec - E[..., None]*boost_vec)
            
            # Get magnitude and unit vectors
            p_orig_mag = jnp.sqrt(jnp.sum(p_orig_vec**2, axis=-1))
            p_orig_unit = p_orig_vec / p_orig_mag[..., None]
            
            return p_orig_mag, p_orig_unit
        
        # Use jax.lax.cond to handle the special case
        return jax.lax.cond(
            boost_mag_squared < 1e-10,  # Threshold for "effectively zero"
            zero_boost_case,
            normal_boost_case,
            operand=None
        )


    def get_p_vec_pixels(p_unit, nside):
        """
        Convert momentum vectors to HEALPix pixel indices
        
        Args:
            p_unit: unit momentum vectors, shape (H, M, Npix, p_num, 3)
            nside: HEALPix nside parameter
        
        Returns:
            pixel indices, shape (H, M, Npix, p_num)

        Usage example:
            # Pixels (indices) that 1yr momenta in GC frame point at
            # pixel_indices = Physics.get_p_vec_pixels(
            #     p_unit=p_GC_unit, nside=simdata.Nside)
            # (1, masses, Npix, p_num)
        """
        
        # Convert to spherical coordinates
        theta = np.arccos(p_unit[..., 2])
        phi = np.arctan2(p_unit[..., 1], p_unit[..., 0])
        
        # Get pixels
        pixels = hp.ang2pix(nside, theta, phi)
        
        return jnp.array(pixels)


    @jax.jit
    def interpolate_fd_values(p_GC_mag, pixel_indices, p_grid, fd_vals):
        """
        Args:
            p_GC_mag: shape (1, M, Npix, p_num) - transformed momenta magnitudes
            pixel_indices: shape (1, M, Npix, p_num) - pixel indices these momenta point to
            p_grid: shape (H, M, Npix, p_num) - original momentum grid for FD values
            fd_vals: shape (H, M, Npix, p_num) - FD values on grid
        """
        # Process each halo and mass configuration separately to avoid broadcasting issues
        def halo_mass_fun(h, m, pixel_pos):
            # Get the values for this (halo, mass, pixel position) combination
            p_interp = p_GC_mag[h, m, pixel_pos]  # shape (p_num,)
            pixels = pixel_indices[h, m, pixel_pos]  # shape (p_num,)
            
            # Get the grid values for these pixels
            p0_pixels = p_grid[h, m, pixels]  # shape (p_num, p_num)
            fd_pixels = fd_vals[h, m, pixels]  # shape (p_num, p_num)
            
            # Find indices where each p_interp value sits in its corresponding grid
            idx = jnp.sum(p0_pixels <= p_interp[:, None], axis=1) - 1
            idx = jnp.clip(idx, 0, p_grid.shape[-1] - 2)
            
            # Get interpolation points
            x0 = jnp.take_along_axis(p0_pixels, idx[:, None], axis=1)[:, 0]
            x1 = jnp.take_along_axis(p0_pixels, (idx+1)[:, None], axis=1)[:, 0]
            y0 = jnp.take_along_axis(fd_pixels, idx[:, None], axis=1)[:, 0]
            y1 = jnp.take_along_axis(fd_pixels, (idx+1)[:, None], axis=1)[:, 0]
            
            # Linear interpolation
            slope = (y1 - y0) / (x1 - x0)
            return y0 + slope * (p_interp - x0)
        
        def pixel_fun(i, result):
            H, M = fd_vals.shape[:2]
            for h in range(H):
                for m in range(M):
                    vals = halo_mass_fun(h, m, i)
                    result = result.at[h, m, i].set(vals)
            return result
        
        result = jnp.zeros_like(fd_vals)
        return jax.lax.fori_loop(0, fd_vals.shape[-2], pixel_fun, result)


    # @jax.jit
    def average_velocity(v_init, p_init, p_back, args):

        # Syncronize velocity array to add dimension for masses
        v_sync = v_init[:, None, ...]

        # Magnitudes of momenta
        p_mags_init = jnp.linalg.norm(p_init, axis=-1)
        p_mags_back = jnp.linalg.norm(p_back, axis=-1)

        option = 3


        if option == 1:

            ### ============ ###
            ### With sorting ### #? broken? see plot...
            ### ============ ###
            sort_indices_p = p_mags_init.argsort(axis=-1)
            # sort_indices_v = jnp.repeat(sort_indices_p[..., None], 3, axis=-1)
            sort_indices_v = sort_indices_p[..., None]

            p_init_sort = jnp.take_along_axis(p_mags_init, sort_indices_p, axis=-1)
            p_back_sort = jnp.take_along_axis(p_mags_back, sort_indices_p, axis=-1)
            v_sync_sort = jnp.take_along_axis(v_sync, sort_indices_v, axis=-2)

            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac(p_back_sort, args)

            # Calculate average velocity integral, log integration
            integrand_v = p_init_sort[..., None] * FD_arr[..., None] * v_sync_sort
            int_array_v = p_init_sort[..., None]
            v_int = trap(integrand_v, x=jnp.log(int_array_v), axis=-2)

            print(int_array_v.shape, integrand_v.shape)

            h_pick = 0
            m_pick = 0

            x_axis = jnp.squeeze(int_array_v)[h_pick,m_pick,:]
            y_axis = jnp.linalg.norm(integrand_v, axis=-1)[h_pick,m_pick,:]
            print(x_axis.shape, y_axis.shape)

            plt.plot(x_axis, y_axis)
            plt.show(); plt.close()

            # Calculate normalization
            integrand_n = p_init_sort[..., None] * FD_arr[..., None]
            int_array_n = p_init_sort[..., None]
            n_int = trap(integrand_n, x=jnp.log(int_array_n), axis=-2)


        elif option == 2:

            ### =============== ###
            ### Without sorting ### #? correct?
            ### =============== ###
            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac(p_mags_back, args)

            # Calculate average velocity integral, log integration
            integrand_v = p_mags_init[..., None] * FD_arr[..., None] * v_sync
            int_array_v = p_mags_init[..., None]
            v_int = trap(integrand_v, x=jnp.log(int_array_v), axis=-2)

            print(int_array_v.shape, integrand_v.shape)

            x_axis = jnp.squeeze(int_array_v)[0,0,:]
            y_axis = jnp.linalg.norm(integrand_v, axis=-1)[0,0,:]
            print(x_axis.shape, y_axis.shape)

            plt.plot(x_axis, y_axis)
            plt.show(); plt.close()

            # Calculate normalization
            integrand_n = p_mags_init[..., None] * FD_arr[..., None]
            int_array_n = p_mags_init[..., None]
            n_int = trap(integrand_n, x=jnp.log(int_array_n), axis=-2)


        elif option == 3:

            ### =========== ###
            ### Simple sums ###
            ### =========== ###
            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac(p_mags_back, args)

            # Calculate average velocity integral, log integration
            v_int = jnp.sum(FD_arr[..., None] * v_sync, axis=-2)

            # Calculate normalization
            n_int = jnp.sum(FD_arr[..., None], axis=-2)


        return v_int / n_int


    @jax.jit
    def average_velocity_Delta(v_init, p_init, p_back, Deltas, args):

        # Syncronize velocity array to add dimension for masses
        v_sync = v_init[:, None, ...]

        # Magnitudes of momenta
        p_mags_init = jnp.linalg.norm(p_init, axis=-1)
        p_mags_back = jnp.linalg.norm(p_back, axis=-1)

        option = 3


        if option == 1:

            ### ============ ###
            ### With sorting ### #? broken? see plot...
            ### ============ ###
            sort_indices_p = p_mags_init.argsort(axis=-1)
            # sort_indices_v = jnp.repeat(sort_indices_p[..., None], 3, axis=-1)
            sort_indices_v = sort_indices_p[..., None]

            p_init_sort = jnp.take_along_axis(p_mags_init, sort_indices_p, axis=-1)
            p_back_sort = jnp.take_along_axis(p_mags_back, sort_indices_p, axis=-1)
            v_sync_sort = jnp.take_along_axis(v_sync, sort_indices_v, axis=-2)

            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac(p_back_sort, args)

            # Calculate average velocity integral, log integration
            integrand_v = p_init_sort[..., None] * FD_arr[..., None] * v_sync_sort
            int_array_v = p_init_sort[..., None]
            v_int = trap(integrand_v, x=jnp.log(int_array_v), axis=-2)

            print(int_array_v.shape, integrand_v.shape)

            x_axis = jnp.squeeze(int_array_v)[0,0,:]
            y_axis = jnp.linalg.norm(integrand_v, axis=-1)[0,0,:]
            print(x_axis.shape, y_axis.shape)

            plt.plot(x_axis, y_axis)
            plt.show(); plt.close()

            # Calculate normalization
            integrand_n = p_init_sort[..., None] * FD_arr[..., None]
            int_array_n = p_init_sort[..., None]
            n_int = trap(integrand_n, x=jnp.log(int_array_n), axis=-2)


        elif option == 2:

            ### =============== ###
            ### Without sorting ### #? correct?
            ### =============== ###
            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac(p_mags_back, args)

            # Calculate average velocity integral, log integration
            integrand_v = p_mags_init[..., None] * FD_arr[..., None] * v_sync
            int_array_v = p_mags_init[..., None]
            v_int = trap(integrand_v, x=jnp.log(int_array_v), axis=-2)

            print(int_array_v.shape, integrand_v.shape)

            x_axis = jnp.squeeze(int_array_v)[0,0,:]
            y_axis = jnp.linalg.norm(integrand_v, axis=-1)[0,0,:]
            print(x_axis.shape, y_axis.shape)

            plt.plot(x_axis, y_axis)
            plt.show(); plt.close()

            # Calculate normalization
            integrand_n = p_mags_init[..., None] * FD_arr[..., None]
            int_array_n = p_mags_init[..., None]
            n_int = trap(integrand_n, x=jnp.log(int_array_n), axis=-2)


        elif option == 3:

            ### =========== ###
            ### Simple sums ###
            ### =========== ###
            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac_Delta(p_mags_back, Deltas, args)
            print(FD_arr.shape)

            # Calculate average velocity integral, log integration
            v_int = jnp.sum(FD_arr[..., None] * v_sync, axis=-2)

            # Calculate normalization
            n_int = jnp.sum(FD_arr[..., None], axis=-2)


        return v_int / n_int


    @staticmethod
    def Energy_Momentum_conversion(X_arr, mass_arr, mode='E_to_p'):

        def Energy_to_momentum(_):
            signs = jnp.sign(X_arr[None,:])
            diffs = jnp.sqrt(X_arr[None, :]**2 - mass_arr[:, None]**2)
            return signs * diffs
        
        def Momentum_to_energy(_):
            signs = jnp.sign(X_arr[None,:])
            diffs = jnp.sqrt(X_arr[None, :]**2 + mass_arr[:, None]**2)
            return signs * diffs

        # Map modes to indices
        mode_index = {'E_to_p': 0, 'p_to_E': 1}
        if mode not in mode_index:
            raise ValueError("Invalid 'ordering' value. It should be 'E_to_p' or 'p_to_E'.")

        # Switch based on the mode
        branches = [Energy_to_momentum, Momentum_to_energy]
        return jax.lax.switch(mode_index[mode], branches, None)


    ### ---------------------------------------------- ###
    ### Functions using "base" number_density function ###
    ### ---------------------------------------------- ###

    @staticmethod
    @jax.jit
    def number_densities_all_sky(v_arr, m_arr, pix_sr, args):

        # Convert velocities to momenta.
        p_arr, _ = Physics.velocities_to_momenta_all_sky(v_arr, m_arr, args)
        # (masses, Npix, neutrinos per pixel, 2 (first and last time step))
        
        # Compute number densities
        nu_dens = Physics.number_density(
            p_arr[..., 0], p_arr[..., -1], pix_sr, args)
        # (masses, Npix)

        return nu_dens


    @staticmethod
    @jax.jit
    def number_densities_mass_range(v_arr, m_arr, pix_sr, args):
        
        # Convert velocities to momenta.
        p_arr, _ = Physics.velocities_to_momenta(v_arr, m_arr, args)

        nu_dens = Physics.number_density(
            p_arr[...,0], p_arr[...,-1], pix_sr, args)

        return nu_dens


    @staticmethod
    def number_densities_z_evolution(
        sim_vels, nu_masses, z_arr, out_file=None, pix_sr=4*Params.Pi,
        average=False, m_start=0.01, z_start=0., sim_type='single_halos'
    ):
        
        # Convert velocities to momenta.
        p_arr, _ = Physics.velocities_to_momenta(sim_vels, nu_masses, Params)

        if average:
            inds = np.array(np.where(z_arr >= z_start)).flatten()
            temp = [
                Physics.number_density(p_arr[...,0], p_arr[...,k], pix_sr, Params) for k in inds
            ]
            num_densities = np.mean(np.array(temp.T), axis=-1)
        else:
            num_densities = Physics.number_density(p_arr[...,0], p_arr[...,-1], pix_sr, Params)

        if 'all_sky' in sim_type:
            return num_densities
        else:
            np.save(f'{out_file}', num_densities)


    @staticmethod
    def CDF_Fermi_Dirac_normed(p, T, p_norm):
        CDF = p - T*np.log(np.exp(p/T)+1) + T*np.log(2)
        norm = p_norm - T*np.log(np.exp(p_norm/T)+1) + T*np.log(2)
        return CDF/norm


    @staticmethod
    def sample_Fermi_Dirac_momentum(N, T, p_norm):
        p_samples = np.zeros(N)
        
        for i in range(N):
            u = np.random.uniform(0, 1)
            
            try:
                # Use Brent's method for root finding
                root = root_scalar(
                    lambda p: Physics.CDF_Fermi_Dirac_normed(p, T, p_norm) - u, 
                    bracket=[0.01 * T, p_norm], 
                    method='brentq'
                )
                p_samples[i] = root.root

            except ValueError:
                # print(f"Skipped iteration {i} due to bracketing issues.")
                continue
        
        return p_samples
