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


    #! FZ: changed Hubble rate to more complete one including Omega_L, which 
    #! FZ: might is relevant at late times (low redshifts), may be altering numerical values
    @jax.jit
    def CALC_Hubble_rate(z, args):
        # Hubble rate in "numerical" units.
        # return args.H0*jnp.sqrt(args.Omega_R*(1+z)**4 + args.Omega_M*(1+z)**3)
        return args.H0*jnp.sqrt(args.Omega_R*(1+z)**4 + args.Omega_M*(1+z)**3 + args.Omega_L)


    def CALC_Hubble_rate_2(z, args):
        # Hubble rate in "numerical" units.
        return args.H0*jnp.sqrt(args.Omega_R*(1+z)**6+ args.Omega_M*(1+z)**5+ args.Omega_L*(1+z)**2)
    
    
    def integrand_rd(p0, m, z):
        H_z = decay.CALC_Hubble_rate(z)
        denominator =   H_z*jnp.sqrt(p0**2+ m**2/(1+z)**2)
        return (p0/denominator)
    

    def redshift_distance(p0, m, z, int_steps = 100):
        z_final = jnp.zeros(jnp.shape(z)) #final redshift, so our redshift z=0
        z_initial = z #array of possitble initial redshifts
        z_range_2=jnp.logspace(jnp.log10(z_final+1), jnp.log10(z_initial+1),int_steps) - 1 #define an integration range

        # Calculate the integrand values
        integrand_values = decay.integrand_rd(p0,m, z_range_2)
        # Perform the numerical integration using np.trapz
        distance = np.trapz(integrand_values, z_range_2, axis = 0)
        distance = distance[-1]-distance #Correct values
        return distance
    

    def integrand(z, redshifted_p, m_h, args):
        p = redshifted_p.T  # necessary operations for broadcasting arrays with different sizes
        z = z.T
        factor_1 = 1 / (decay.CALC_Hubble_rate(z, args=args) * (1 + z))
        factor_2 = 1 / jnp.sqrt(jnp.square(1 + z[:, :, jnp.newaxis]) + jnp.square(m_h*(1+z[:, :, jnp.newaxis])/p[:, jnp.newaxis]))
        integrand = factor_1[:, :, jnp.newaxis] * factor_2
        return integrand.T


    def Xi(z, redshifted_p, m_h, args):
        """Auxiliary integral, representing effective time between z_end and z where z is an array."""
        int_steps = 100
        
        # Integrand in xi function
        # Create redshift integration range.
        z_int_range = jnp.logspace(jnp.log10(z + 1), jnp.log10(z[-1] + 1), int_steps) - 1  # integration range for each new z as given in the input variables
        integrand_array = decay.integrand(z_int_range, redshifted_p, m_h,args=args)  # array of integrands in order to integrate in Xi_array
        Xi_array =jax.scipy.integrate.trapezoid(integrand_array, x=z_int_range[jnp.newaxis, :, :], axis=1)
        return Xi_array    


    def n3_p_range_only(p, z, Gamma, m_h, args):

        n_pre = decay.SPEC_n_prefactor(p, z, args=args)
        
        Xi_factor = decay.Xi(z, p/(1+z), m_h, args=args)
        n_exp = jnp.exp(-((m_h*Gamma*(1+z)*Xi_factor)/p) )
        
        return n_pre*n_exp


    #@jax.jit
    def n3_p_range(p, z, Gamma, m_h, args): 
        '''gamma, z_end, m passed on as args'''
        """Number density for m_3 per momentum range, prefactor, xi_factor and decay probability."""
        # Prefactor
        n_pre = decay.SPEC_n_prefactor(p, z, args=args)

        # Exp-factor with Xi integral.
        Xi_factor = decay.Xi(z, p, m_h, args=args)
        
        n_exp = jnp.exp(-((m_h*Gamma*(1+z)* Xi_factor)/p) )
        n3_deriv_p_range = n_pre * n_exp  * (-3 / (1 + z)+ 
                    (p / (args.T_CNB * (1 + z) ** 2)) * 1/(jnp.exp(p/(args.T_CNB *(1+z))) + 1) *jnp.exp(p / (args.T_CNB  * (1 + z))) - 
                    (Gamma* m_h/ p) * (Xi_factor -((1+z)/(decay.CALC_Hubble_rate_2(z,args=args)))* (1/(jnp.sqrt((1+z)**2 + (m_h**2/p**2) )))))
        
        return n_pre * n_exp, n_pre, Xi_factor, jnp.abs(n3_deriv_p_range)
   

    def montecarlo(z, p, m_h, number_neutrinos, Gamma, args):

        number_neutrinos = 768000
        redshifted_p = p[:, jnp.newaxis] * (1 + z)  # Broadcasting multiplication
        
        n3_raw, prefac, xi, n3_deriv_p_range = decay.n3_p_range(p=redshifted_p, z=z, Gamma=Gamma, m_h=m_h, args=args)
      
        n3_deriv_redshifts = jax.scipy.integrate.trapezoid(n3_deriv_p_range, redshifted_p, axis=0)
        n3_redshifts = jax.scipy.integrate.trapezoid(n3_raw, redshifted_p, axis=0)
        n3_initial = n3_redshifts[-1]
        
        prob_dec = n3_deriv_redshifts / n3_initial
        prob_dec_max = np.max(prob_dec)
        
        '''plt.plot(z,prob_dec)
           plt.show()'''

        prob_dec = prob_dec/prob_dec_max #scaling
       
        integrated_distribution = n3_redshifts / n3_initial
    
        prob_surv = 1 - prob_dec
        
        random_numbers = jax.random.uniform(jax.random.PRNGKey(0), shape=(768000,))
        occurences_redshifts = z[jnp.searchsorted(integrated_distribution[:-1], random_numbers)]
        histogram_data, bins, bars = plt.hist((np.log10(1+occurences_redshifts[occurences_redshifts != 0])),bins=100,histtype='step',color='mediumpurple',label ='# decays')
        '''#Plotting:
        
        plt.plot(np.log10(1+z), np.max(histogram_data)*prob_dec,label='Prob of decay')
        plt.xscale('linear')
        plt.grid()
        plt.legend()
        plt.title('Number of Decayed Neutrinos and Scaled Probability of Decay')
        plt.xlabel('1+z')
        plt.ylabel('Ocurrences')
        plt.yscale('linear')
        plt.show()'''

        return histogram_data, occurences_redshifts
    

    #Not JAX compatible
    #@jax.jit
    def decay_neutrinos(histogram_data, gamma_str, args):
        total_neutrinos = 768000

        # Create an array representing individual neutrinos (all ones)
        neutrinos_array =np.ones(total_neutrinos)
        # List to store neutrinos_array for each redshift step
        decayed_redshift = []
        neutrinos_decayed_index =[]
        neutrinos_decayed_theta =[]
        neutrinos_decayed_phi = []
        #for statement 
        # Iterate over redshift steps
        for i, num_decayed in enumerate(histogram_data):
        
            # Reset decayed indices
            decay_indices = []
            neutrinos_decayed_index_0 =[]
            # Randomly select neutrinos to decay
            while len(decay_indices) < num_decayed:
                index = np.random.randint(total_neutrinos)
                if neutrinos_array[index] == 1:  # Check if neutrino is still active
                    decay_indices.append(index)
                    theta = np.random.randint(0,180)
                    phi = np.random.randint(0,360)    
                    neutrinos_decayed_theta.append(theta)
                    neutrinos_decayed_phi.append(phi)
                    neutrinos_decayed_index_0.append(index)
                    neutrinos_array[index] = 0
            neutrinos_decayed_index.append(np.array(neutrinos_decayed_index_0))
            decayed_redshift.append(np.copy(neutrinos_array))
        
        neutrinos_decayed_index_array = np.empty(len(neutrinos_decayed_index), dtype=object)
        neutrinos_decayed_index_array[:] = neutrinos_decayed_index
        
        np.save(f'decayed_neutrinos_index_z_{gamma_str}.npy', neutrinos_decayed_index_array)
        np.save(f"decayed_neutrinos_z_{gamma_str}.npy",decayed_redshift)

        np.save(f"decayed_neutrinos_theta_{gamma_str}.npy",neutrinos_decayed_theta)
        np.save(f"decayed_neutrinos_phi_{gamma_str}.npy",neutrinos_decayed_phi)

        return decayed_redshift
    

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
        solution_l = fsolve(
            decay.equation_l, initial_guess, 
            args=(m_l, m_h, m_phi, jnp.cos(jnp.radians(angle)), p_h, E_h))
        solution_phi = fsolve(
            decay.equation_phi, initial_guess, 
            args=(m_phi, m_h, m_l, jnp.cos(jnp.radians(angle)), p_h, E_h))
        return solution_l, solution_phi
    

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

        non_zero_mask_init = p_init_sort != 0
        non_zero_elements_init = p_init_sort[non_zero_mask_init]
        p_init_sort = non_zero_elements_init.reshape(p_init_sort.shape[0], -1)
        
        non_zero_mask_back = p_back_sort != 0
        non_zero_elements_back = p_back_sort[non_zero_mask_back]
        p_back_sort = non_zero_elements_back.reshape(p_back_sort.shape[0], -1)
        
        # Fermi-Dirac values with momenta at end of sim
        FD_arr = Physics.Fermi_Dirac(p_back_sort, args)

        # Calculate number density
        y = p_init_sort**3 * FD_arr  # dlog integrand
        x = p_init_sort
        n_raw = trap(y, jnp.log(x), axis=-1)

        # Multiply by constants and/or solid angles and convert to 1/cm**3.
        n_cm3 = pix_sr * args.g_nu/((2*args.Pi)**3) * n_raw / (1/args.cm**3)

        return jnp.array(n_cm3)
    

    def number_densities_mass_range_decay(v_arr, m_arr, pix_sr, args):
        
        # Convert velocities to momenta.
        p_arr, _ = Physics.velocities_to_momenta(v_arr, m_arr, args)

        nu_dens = decay.number_density(
            p_arr[...,0], p_arr[...,-1], pix_sr, args)

        return nu_dens