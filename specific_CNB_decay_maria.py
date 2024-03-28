from Shared.shared import *
import jax.random as random
from scipy.optimize import fsolve
from scipy.optimize import minimize


class decay():
    @jax.jit
    def SPEC_n_prefactor(p, z, args):
        '''z and p are both arrays'''
        # Assuming T_CNB is passed as an argument
        return p**2 / (2*jnp.pi**2 * (1+z)**3) * (1 / (jnp.exp(p / (args.T_CNB * (1+z))) + 1))

    @jax.jit
    def CALC_Hubble_rate(z, args):
        # Hubble rate in "numerical" units.
        return args.H0 * jnp.sqrt(args.Omega_R * (1 + z)**4 + args.Omega_M * (1 + z)**3)


    def integrand(z, redshifted_p,  m_h, args):
        p = redshifted_p.T  # necessary operations for broadcasting arrays with different sizes
        z = z.T
        factor_1 = 1 / (decay.CALC_Hubble_rate(z, args=args) * (1 + z))
        factor_2 = 1 / jnp.sqrt(jnp.square(1 + z[:, :, jnp.newaxis]) + jnp.square(m_h/ p[:, jnp.newaxis]))
        integrand = factor_1[:, :, jnp.newaxis] * factor_2
        return integrand.T

    def Xi(z, redshifted_p, m_h, args):
        """Auxiliary integral, representing effective time between z_end and z where z is an array."""
        int_steps = 100
        
        # Integrand in xi function
        # Create redshift integration range.
        z_int_range = jnp.logspace(jnp.log10(z + 1), jnp.log10(args.Z_LSS_CMB + 1), int_steps) - 1  # integration range for each new z as given in the input variables
        integrand_array = decay.integrand(z_int_range, redshifted_p, m_h,args=args)  # array of integrands in order to integrate in Xi_array
        Xi_array =jax.scipy.integrate.trapezoid(integrand_array, x=z_int_range[jnp.newaxis, :, :], axis=1)
        return Xi_array    

    #@jax.jit
    def n3_p_range(p, z , Gamma, m_h, args): 
        '''gamma, z_end, m passed on as args'''
        """Number density for m_3 per momentum range, prefactor, xi_factor and decay probability."""
        # Prefactor
        n_pre = decay.SPEC_n_prefactor(p,z,args = args)

        # Exp-factor with Xi integral.
        Xi_factor = decay.Xi(z, p/(1+z),m_h, args = args)
        n_exp = jnp.exp(-((m_h*Gamma*(1+z)* Xi_factor)/p) )
        
        n3_deriv_p_range = n_pre * n_exp  * (-3 / (1 + z)+ 
                    (p / (args.T_CNB * (1 + z) ** 2)) * 1/(jnp.exp(p/(args.T_CNB *(1+z))) + 1) *jnp.exp(p / (args.T_CNB  * (1 + z))) - 
                    (Gamma* m_h/ p)*(1+z) * (Xi_factor/(1+z) -(1/(decay.CALC_Hubble_rate(z,args=args)))* (1/(jnp.sqrt((1+z)**2 + (m_h**2/p**2) )))))

        return n_pre * n_exp, n_pre, Xi_factor, jnp.abs(n3_deriv_p_range)
    #@jax.jit
    def decay_neutrinos(histogram_data, args):
        total_neutrinos = jnp.sum(histogram_data)

        # Create an array representing individual neutrinos (all ones)
        neutrinos_array =jnp.ones(total_neutrinos)
        key = jax.random.PRNGKey(0)
        # List to store neutrinos_array for each redshift step
        decayed_redshift = []
        #for statement 
        # Iterate over redshift steps
        for i, num_decayed in enumerate(histogram_data):
        
            # Reset decayed indices
            decay_indices = []
            
            # Randomly select neutrinos to decay
            while len(decay_indices) < num_decayed:
                key, subkey = jax.random.split(key)
                index = jax.random.randint(subkey, minval=0, maxval=total_neutrinos, shape=(1,))
                if neutrinos_array[index] == 1:  # Check if neutrino is still active
                    decay_indices.append(index)
                    neutrinos_array = neutrinos_array.at[index].set(0)
            
            decayed_redshift.append(jnp.copy(neutrinos_array))
        
        return decayed_redshift

    def montecarlo(z,p,m_h,number_neutrinos,Gamma,args):
    # z_array = np.logspace(np.log10(0+1), np.log10(z_dec+1), z_steps)-1 #array of redshifts for testing # neutrino decays at each point
    # np.save('z_array.npy', np.array(z_array))
        number_neutrinos = 768000
        #distances_arr = redshift_distance(Params.p0, m_h, z) #translation into distance
        redshifted_p = p[:, None] * (1 + z)  # Broadcasting multiplication

        n3_raw, prefac, xi, decay_prob = decay.n3_p_range(p=redshifted_p, z=z, Gamma=Gamma, m_h=m_h, args=args)

        n3_deriv_redshifts =jax.scipy.integrate.trapezoid(decay_prob, redshifted_p, axis=0)
        n3_redshifts = jax.scipy.integrate.trapezoid(n3_raw, redshifted_p, axis=0)
        n3_initial = n3_redshifts[-1]
        
        prob_dec = n3_deriv_redshifts / n3_initial
        
        integrated_distribution = n3_redshifts / n3_initial
        
        prob_surv = 1 - prob_dec
        # Generate random numbers using JAX's PRNG
    
        # You can change the seed if needed
        random_numbers = jax.random.uniform(jax.random.PRNGKey(0), shape=(768000,))
        occurences_redshifts = z[jnp.searchsorted(integrated_distribution[:-1], random_numbers)]
        histogram_data, bin_edges = jnp.histogram(jnp.log10(occurences_redshifts + 1), bins=100)
        return histogram_data,occurences_redshifts
    
    def find_nearest(array, value):
        idx = jnp.argmin(jnp.abs(array - value))
        return array[idx]
    
        
    def equation_l(p_l, m_l,m_h,m_phi, cos_theta_l, p_h, E_h):
        # Calculate E_l and E_h based on the given relation
        E_l = jnp.sqrt(p_l**2 + m_l**2)
        # Calculate cos_theta_daughter using the given equation
        cos_theta_daughter_calculated = (E_h * E_l - (m_h**2 + m_l**2 - m_phi**2) / 2) / (p_h * p_l)
        # Return the difference between the calculated and given values
        return cos_theta_daughter_calculated - cos_theta_l

    def equation_phi(p_phi, m_phi,m_h, m_l, cos_theta_phi, p_h, E_h):
        # Calculate E_phi and E_h based on the given relation
        E_phi = jnp.sqrt(p_phi**2 + m_phi**2)
        # Calculate cos_theta_daughter using the given equation
        cos_theta_daughter_calculated = (E_h * E_phi - (m_h**2 + m_phi**2 - m_l**2) / 2) / (p_h * p_phi)

        # Return the difference between the calculated and given values
        return cos_theta_daughter_calculated - cos_theta_phi


    def daughter_momentum(p_h, m_h,m_l,m_phi, angle, args):
        E_h = jnp.sqrt(p_h**2 + m_h**2)
        initial_guess = p_h  
        solution_l = fsolve(decay.equation_l, initial_guess, args=( m_l,m_h,m_phi,jnp.cos(jnp.radians(angle)), p_h,E_h))
        solution_phi = fsolve(decay.equation_phi, initial_guess, args=( m_phi,m_h, m_l, jnp.cos(jnp.radians(angle)), p_h,E_h))
        return solution_l, solution_phi
