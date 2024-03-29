from Shared.specific_CNB_sim import *
import jax.random as random
from scipy.optimize import fsolve
from scipy.optimize import minimize

total_start = time.perf_counter()

angle_momentum_decay = jnp.load(f'sim_output/no_gravity/neutrino_angle_momentum_decay.npy') 
decayed_neutrinos_z = jnp.load(f'sim_output/no_gravity/decayed_neutrinos_z.npy')
z_array = jnp.load(f'sim_output/no_gravity/z_int_steps.npy')
neutrino_momenta = jnp.load(f'sim_output/no_gravity/neutrino_momenta.npy')

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('--directory', required=True)
parser.add_argument(
    '--pixel_densities', required=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    '--total_densities', required=True, action=argparse.BooleanOptionalAction)
pars = parser.parse_args()

# Create halo batch, files and other simulation setup parameters and arrays
DM_mass, CPUs_sim, neutrinos, init_dis, zeds_snaps, z_int_steps, s_int_steps, nu_massrange = SimData.simulation_setup(
    sim_dir=pars.directory,
    m_lower=None,
    m_upper=None,
    m_gauge=None,
    halo_num_req=None,
    no_gravity=True)

def find_nearest(array, value):
    idx = jnp.argmin(jnp.abs(array - value))
    return array[idx]


@jax.jit
#specify mass as an argument too
def EOMs_no_gravity(s_val, y, args):
    Nr_index, angle_momentum_decay,decayed_neutrinos_z, z_array, neutrino_momenta = args
    _, u_i_p = y
    # Find z corresponding to s via interpolation.
    z =  Utils.jax_interpolate(s_val, s_int_steps, z_int_steps)
    z_nearest = find_nearest(z_array, z)

   # check index of z in our z array
    z_index = jnp.argmax(jnp.equal(z_nearest, z_array)) # Need to access the first element of the array returned by where function
    neutrino_number = jnp.int16(decayed_neutrinos_z[z_index,Nr_index])
    prev_neutrino_number = jnp.int16(decayed_neutrinos_z[z_index - 1,Nr_index])
    #if z_index>0:
    #   prev_neutrino_number = jnp.int16(decayed_neutrinos_z[z_index - 1,Nr_index][0])
   # else:
    #    prev_neutrino_number = 1
    
 
    def true_func(y):
       
        p_i = find_nearest(neutrino_momenta, 0.06 * jnp.linalg.norm(y[1], axis=-1)) #neutrino momenta pass as arg 
        p_index = jnp.argmax(jnp.equal(neutrino_momenta, p_i))  # Check which index it corresponds to
        
        angle_decay_theta = random.randint(jax.random.PRNGKey(0),(1,), 0, 179)
        angle_decay_phi = random.randint(jax.random.PRNGKey(0),(1,), 0, 360)

        momentum_daughter = angle_momentum_decay[angle_decay_theta,p_index][0]
      
        u_i = jnp.array( [(1 / 0.05) * momentum_daughter * jnp.sin(angle_decay_theta) * jnp.cos(angle_decay_phi),
            (1 / 0.05) * momentum_daughter * jnp.sin(angle_decay_theta) * jnp.sin(angle_decay_phi),
            (1 / 0.05) * momentum_daughter * jnp.cos(angle_decay_theta)]) #mass should be an argument eventually

        return jnp.squeeze(u_i)

    def false_func(y):
        return jnp.squeeze(y[1])

    u_i = jax.lax.cond((neutrino_number == 0) & (prev_neutrino_number == 1), y,  true_func, y, false_func)
  
    dyds = -jnp.array([
        u_i, jnp.zeros(3)
    ])

    return dyds


@jax.jit
def backtrack_1_neutrino(init_vector, s_int_steps,angle_momentum_decay,decayed_neutrinos_z,z_array,neutrino_momenta):

    

    """
    Simulate trajectory of 1 neutrino. Input is 6-dim. vector containing starting positions and velocities of neutrino. Solves ODEs given by the EOMs function with an jax-accelerated integration routine, using the diffrax library. Output are the positions and velocities at each timestep, which was specified with diffrax.SaveAt. 
    """
    y0_r, Nr = init_vector[0:-1], init_vector[-1]
  
    # Initial vector in correct shape for EOMs function
    y0 = y0_r.reshape(2,3)
    
    # ODE solver setup
    term = diffrax.ODETerm(EOMs_no_gravity)
    t0 = s_int_steps[0]
    t1 = s_int_steps[-1]
    dt0 = (s_int_steps[0] + s_int_steps[1]) / 1000
    

    ### ------------- ###
    ### Dopri5 Solver ###
    ### ------------- ###
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    # note: no change for tighter rtol and atol, e.g. rtol=1e-5, atol=1e-9


    # Specify timesteps where solutions should be saved
    saveat = diffrax.SaveAt(ts=jnp.array(s_int_steps))
    # Solve the coupled ODEs, i.e. the EOMs of the neutrino
    sol = diffrax.diffeqsolve(
        term, solver, 
        t0=t0, t1=t1, dt0=dt0, y0=y0, max_steps=10000,
        saveat=saveat, stepsize_controller=stepsize_controller, args=(Nr.astype(int),angle_momentum_decay,decayed_neutrinos_z,z_array, neutrino_momenta))
    
    trajectory = sol.ys.reshape(100,6)

    # Only return the initial [0] and last [-1] positions and velocities
    return jnp.stack([trajectory[0], trajectory[-1]])


def simulate_neutrinos_1_pix(init_xyz, init_vels, s_int_steps, Nr_column, angle_momentum_decay,decayed_neutrinos_z,z_array,neutrino_momenta):
    """
    Function for the multiprocessing routine below, which simulates all neutrinos for 1 pixel on the healpix skymap.
    """

    # Neutrinos per pixel
    nus = init_vels.shape[0]
    # Make vector with same starting position but different velocities
    init_vectors_0 = jnp.array(
        [jnp.concatenate((init_xyz, init_vels[k])) for k in range(nus)])

    Nr_column_reshaped = jnp.expand_dims(Nr_column, axis=1)
    # Concatenate the additional column to the original array
    init_vectors = jnp.hstack((init_vectors_0, Nr_column_reshaped))

    trajectories = jnp.array([
        backtrack_1_neutrino(vec, s_int_steps, angle_momentum_decay,decayed_neutrinos_z,z_array,neutrino_momenta) for vec in init_vectors])

    return trajectories  # shape = (neutrinos, 2, 3)


# Lists for pixel and total number densities
pix_dens_l = []
tot_dens_l = []

# File name ending
end_str = f'halo1'

# Initial position (Earth)
init_xyz = np.array([float(init_dis), 0., 0.])
jnp.save(f'{pars.directory}/init_xyz_{end_str}.npy', init_xyz)


### ============== ###
### Run Simulation ###
### ============== ###

print(f"*** Simulation for no_gravity ***")

sim_start = time.perf_counter()

with open(f'{pars.directory}/sim_parameters.yaml', 'r') as file:
    sim_setup = yaml.safe_load(file)

pix_sr_sim = sim_setup['pix_sr']        # Size (in sr) of all-sky healpixels
Npix = sim_setup["Npix"]                # Number of healpixels 
nu_per_pix = sim_setup["momentum_num"]  # Number of neutrinos per healpixel

init_vels = np.load(f'{pars.directory}/initial_velocities.npy')  
# shape = (Npix, neutrinos per pixel, 3)
Nr_column= jnp.arange(768000).reshape(768, 1000)
# Use ProcessPoolExecutor to distribute the simulations across processes:
# 1 process (i.e. CPU) simulates all neutrinos for one healpixel.
with ProcessPoolExecutor(CPUs_sim) as executor:
    futures = [
        executor.submit(
            simulate_neutrinos_1_pix, init_xyz, init_vels[pixel], s_int_steps, Nr_column[pixel], angle_momentum_decay,decayed_neutrinos_z,z_array,neutrino_momenta) for pixel in range(Npix)]
    
    # Wait for all futures to complete and collect results in order
    nu_vectors = jnp.array([future.result() for future in futures])

# Save all sky neutrino vectors for current halo
jnp.save(f'{pars.directory}/vectors_{end_str}.npy', nu_vectors)

sim_time = time.perf_counter()-sim_start
print(f"Simulation time: {sim_time/60.:.2f} min, {sim_time/(60**2):.2f} h")


### ======================== ###
### Compute number densities ###
### ======================== ###

if pars.pixel_densities:

    # Compute individual number densities for each healpixel
    pix_start = time.perf_counter()

    nu_allsky_masses = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV
    pix_dens = Physics.number_densities_all_sky(
        v_arr=nu_vectors[..., 3:],
        m_arr=nu_allsky_masses,
        pix_sr=pix_sr_sim,
        args=Params())
    pix_dens_l.append(jnp.squeeze(pix_dens))

    pix_time = time.perf_counter() - pix_start

    jnp.save(f"{pars.directory}/pixel_densities.npy", jnp.array(pix_dens_l))
    print(f"Analysis time: {pix_time/60.:.2f} min, {pix_time/(60**2):.2f} h\n")

if pars.total_densities:

    # Compute total number density, by using all neutrino vectors for integral
    tot_start = time.perf_counter()

    tot_dens = Physics.number_densities_mass_range(
        v_arr=nu_vectors.reshape(-1, 2, 6)[..., 3:], 
        m_arr=nu_massrange, 
        pix_sr=4*Params.Pi,
        args=Params())
    tot_dens_l.append(jnp.squeeze(tot_dens))

    tot_time = time.perf_counter() - tot_start

    jnp.save(f"{pars.directory}/total_densities.npy", jnp.array(tot_dens_l))
    print(f"Analysis time: {tot_time/60.:.2f} min, {tot_time/(60**2):.2f} h\n")



total_time = time.perf_counter() - total_start
print(f"Total time: {total_time/60.:.2f} min, {total_time/(60**2):.2f} h")
