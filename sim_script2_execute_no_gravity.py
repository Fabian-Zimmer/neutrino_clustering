from Shared.specific_CNB_sim import *


total_start = time.perf_counter()

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


@jax.jit
def EOMs_no_gravity(s_val, y, args):

    # Initialize vector.
    _, u_i = y

    # Hamilton eqns. for integration (global minus, s.t. we go back in time).
    dyds = -jnp.array([
        u_i, jnp.zeros(3)
    ])

    return dyds


@jax.jit
def backtrack_1_neutrino(init_vector, s_int_steps):

    """
    Simulate trajectory of 1 neutrino. Input is 6-dim. vector containing starting positions and velocities of neutrino. Solves ODEs given by the EOMs function with an jax-accelerated integration routine, using the diffrax library. Output are the positions and velocities at each timestep, which was specified with diffrax.SaveAt. 
    """

    # Initial vector in correct shape for EOMs function
    y0 = init_vector.reshape(2,3)

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
        saveat=saveat, stepsize_controller=stepsize_controller, args=None)
    
    trajectory = sol.ys.reshape(100,6)

    # Only return the initial [0] and last [-1] positions and velocities
    return jnp.stack([trajectory[0], trajectory[-1]])


def simulate_neutrinos_1_pix(init_xyz, init_vels, s_int_steps):

    """
    Function for the multiprocessing routine below, which simulates all neutrinos for 1 pixel on the healpix skymap.
    """

    # Neutrinos per pixel
    nus = init_vels.shape[0]

    # Make vector with same starting position but different velocities
    init_vectors = jnp.array(
        [jnp.concatenate((init_xyz, init_vels[k])) for k in range(nus)])


    trajectories = jnp.array([
        backtrack_1_neutrino(vec, s_int_steps) for vec in init_vectors])
    
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

# Use ProcessPoolExecutor to distribute the simulations across processes:
# 1 process (i.e. CPU) simulates all neutrinos for one healpixel.
with ProcessPoolExecutor(CPUs_sim) as executor:
    futures = [
        executor.submit(
            simulate_neutrinos_1_pix, init_xyz, init_vels[pixel], s_int_steps) for pixel in range(Npix)]
    
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
