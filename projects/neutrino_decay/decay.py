"""Neutrino decay: parent number densities, decay sampling and daughter densities.

Class Decay, used by the decay simulation scripts and notebooks. Restored from
the Shared repo (specific_CNB_decay.py, deleted there on 2024-11-04); a cleaned
version of decay_maria.py.
"""

# Additional imports
from jax.scipy.integrate import trapezoid as trap
import jax
import jax.numpy as jnp
import numpy as np

from cnb.physics import Physics
from cnb.utils import Utils


# Note: CHANGES
# - Capitalize class name
# - Only one Hubble rate function, and was missing Omega_L
# - all trapezoid calls replaced with trap (from jax.scipy)
#   - regarding the trap function, I used log integration everywhere
# - Overall: removed all unused functions, comments, imports, etc.
# - replaced decay_neutrinos function with other version, couldn't run yours


class Decay():

    @jax.jit
    def SPEC_n_prefactor(p, z, args):
        '''z and p are both arrays'''
        return p**2/(2*jnp.pi**2*(1+z)**3) * 1/(jnp.exp(p/(args.T_CNB*(1+z))) + 1)
    

    @jax.jit
    def CALC_Hubble_rate(z, args):
        # Hubble rate in "numerical" units.
        return args.H0*jnp.sqrt(args.Omega_R*(1+z)**4 + args.Omega_M*(1+z)**3 + args.Omega_L)    


    @jax.jit
    def integrand_rd(p0, m, z):

        H_z = Decay.CALC_Hubble_rate(z)
        denominator = H_z*jnp.sqrt(p0**2 + m**2/(1+z)**2)
        
        return (p0/denominator)
    
    
    @jax.jit
    def integrand(z, redshifted_p,  m_h, args):

        p = redshifted_p.T  # necessary operations for broadcasting arrays with different sizes
        z = z.T
        Hubble_factor = 1/(Decay.CALC_Hubble_rate(z, args)*(1+z))
        Sqrt_factor = 1/jnp.sqrt((1+z[:, :, jnp.newaxis])**2 + (m_h/p[:, jnp.newaxis])**2)
        integrand = Hubble_factor[:, :, jnp.newaxis]*Sqrt_factor

        return integrand.T

    
    @jax.jit
    def Xi(z, redshifted_p, m_h, args):
        """Auxiliary integral, representing effective time between z_end and z where z is an array."""
        
        # Integrand in xi function
        # Create redshift integration range.

        # integration range for each new z as given in the input variables
        z_int_range = Utils.make_logz_arr(z, z[-1], 100)

        # array of integrands in order to integrate in Xi_array
        integrand_array = Decay.integrand(z_int_range, redshifted_p, m_h, args=args)

        Xi_array = trap(integrand_array, x=z_int_range[jnp.newaxis, :, :], axis=1)
        
        return Xi_array

    
    @jax.jit
    def n3_p_range(p, z , Gamma, m_h, args): 
        '''gamma, z_end, m passed on as args'''
        """Number density for m_3 per momentum range, prefactor, xi_factor and decay probability."""
        # Prefactor
        n_pre = Decay.SPEC_n_prefactor(p, z, args=args)

        # Exp-factor with Xi integral.
        Xi_factor = Decay.Xi(z, p/(1+z), m_h, args=args)
        
        n_exp = jnp.exp(-(m_h*Gamma*(1+z)*Xi_factor)/p)

        return n_pre*n_exp, n_pre, Xi_factor
   

    def montecarlo(z_range, p_range, m_val, neutrinos, Gamma, args):

        # Create 2D redshifted momentum array
        redshifted_p = p_range[:,jnp.newaxis]*(1+z_range)
        
        # Compute parent number densities
        n3_raw, *_ = Decay.n3_p_range(
            p=redshifted_p, z=z_range, 
            Gamma=Gamma, m_h=m_val, args=args)
      
        # log integration over momentum axis
        n3_redshifts = trap(n3_raw*redshifted_p, x=jnp.log(redshifted_p), axis=0)
        
        # Starting number density, at high redshifts (i.e. at CNB decoupling)
        n3_initial = n3_redshifts[-1]
        
        # Normalized array
        integrated_distribution = n3_redshifts / n3_initial
     
        random_numbers = jax.random.uniform(args.key, shape=(neutrinos,))
        occurences_redshifts = z_range[jnp.searchsorted(integrated_distribution[:-1], random_numbers)]

        return occurences_redshifts
    

    def inverse_transform_sampling(z_range, p_range, m_val, neutrinos, Gamma, args):

        # Create 2D redshifted momentum array
        redshifted_p = p_range[:,jnp.newaxis]*(1+z_range)
        
        # Compute parent number densities
        n3_raw, *_ = Decay.n3_p_range(
            p=redshifted_p, z=z_range, 
            Gamma=Gamma, m_h=m_val, args=args)
      
        # log integration over momentum axis
        n3_redshifts = trap(n3_raw*redshifted_p, x=jnp.log(redshifted_p), axis=0)

        # Compute the derivative of n3_redshifts with respect to z_range using finite differences
        dn3_dz = np.zeros_like(n3_redshifts)
        dn3_dz[1:-1] = (n3_redshifts[2:] - n3_redshifts[:-2]) / (z_range[2:] - z_range[:-2])
        dn3_dz[0] = (n3_redshifts[1] - n3_redshifts[0]) / (z_range[1] - z_range[0])
        dn3_dz[-1] = (n3_redshifts[-1] - n3_redshifts[-2]) / (z_range[-1] - z_range[-2])
        
        # Normalize to create a probability density function (PDF)
        pdf = dn3_dz / jnp.sum(dn3_dz)
        
        # Compute the cumulative distribution function (CDF)
        cdf = jnp.cumsum(pdf)

        # Generate uniform random samples
        Us = jax.random.uniform(args.key, shape=(neutrinos,))
        
        # Use the searchsorted function to do inverse transform sampling
        occurences_redshifts = z_range[jnp.searchsorted(cdf[:-1], Us)]

        # # Use inverse transform sampling to get samples from z_range
        # occurences_redshifts = jnp.interp(uniform_samples, cdf, z_range)
        
        return pdf, cdf, occurences_redshifts

    
    # @jax.jit
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
        
        zero_mask_init = (p_init_sort == 0)
        p_init_sort_y = jnp.copy(p_init_sort)
        p_init_sort = p_init_sort.at[zero_mask_init].set(1)
        # shape should be (masses, npix, neutrinos per pixel)
      
        # Fermi-Dirac values with momenta at end of sim
        FD_arr = Physics.Fermi_Dirac(p_back_sort, args)
        
        # Calculate number density
        y = p_init_sort_y**3 * FD_arr  # dlog integrand
        y = y.at[zero_mask_init].set(0) 
        x = p_init_sort

        n_raw = trap(y, jnp.log(x), axis=-1)

        # Multiply by constants and/or solid angles and convert to 1/cm**3.
        n_cm3 = pix_sr * args.g_nu/((2*args.Pi)**3) * n_raw / (1/args.cm**3)

        return jnp.array(n_cm3)
    

    def number_densities_mass_range_decay(v_arr, m_arr, pix_sr, args):
        
        # Convert velocities to momenta.
        p_arr, _ = Physics.velocities_to_momenta(v_arr, m_arr, args)

        nu_dens = Decay.number_density(
            p_arr[...,0], p_arr[...,-1], pix_sr, args)

        return nu_dens
    

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
    

    def number_densities_all_sky(v_arr, m_arr, pix_sr, args):

        # Convert velocities to momenta.
        p_arr, _ = Decay.velocities_to_momenta_all_sky(v_arr, m_arr, args)
        # (masses, Npix, neutrinos per pixel, 2 (first and last time step))
        
        # Compute number densities
        nu_dens = Decay.number_density(
            p_arr[..., 0], p_arr[..., -1], pix_sr, args)
        # (masses, Npix)

        return nu_dens
