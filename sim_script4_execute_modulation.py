from Shared.specific_CNB_sim import *


total_start = time.perf_counter()

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('--directory', required=True)
parser.add_argument(
    '--pixel_densities', required=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    '--total_densities', required=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    '--testing', required=True, action=argparse.BooleanOptionalAction)
pars = parser.parse_args()

# Simulation parameters.
with open(f'{pars.directory}/sim_parameters.yaml', 'r') as file:
    sim_setup = yaml.safe_load(file)

CPUs_sim = 128
neutrinos = 1000

# Load Earth-Sun distances
df = pd.read_excel('Data/Earth-Sun_distances.xlsx')
ES_distances = jnp.array(df.iloc[:, 1::2].apply(pd.to_numeric, errors='coerce')\
                                  .stack().reset_index(drop=True).tolist())[:-1]
ES_distances *= Params.AU/Params.kpc
init_dis = ES_distances[0]  # day1

z_int_steps = jnp.load(f'{pars.directory}/z_int_steps_1year.npy')
s_int_steps = jnp.load(f'{pars.directory}/s_int_steps_1year.npy')
nu_massrange = jnp.load(f'{pars.directory}/neutrino_massrange_eV.npy')*Params.eV


@jax.jit
def EOMs_sun(s_val, y, args):

    # Unpack the input data
    s_int_steps, z_int_steps, kpc, s = args

    # Initialize vector.
    x_i, u_i = y

    # Switch to "numerical reality" here.
    x_i *= kpc
    u_i *= (kpc/s)

    # Find z corresponding to s via interpolation.
    z = Utils.jax_interpolate(s_val, s_int_steps, z_int_steps)

    # note: sun implementation causes nan densities 
    # note: (commenting this part out restores working no_gravity densities)
    # """
    # Compute gradient of sun.
    eps = (696_340/(3.086e16))*kpc
    grad_sun = SimExec.sun_gravity(x_i, jnp.array([0.,0.,0.]), eps)

    # Replace NaNs with zeros and apply cutoff
    grad_sun = jnp.nan_to_num(
        grad_sun, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    cutoff = 1e-50
    grad_sun = jnp.where(jnp.abs(grad_sun) < cutoff, 0.0, grad_sun)

    # Switch to "physical reality" here.
    grad_sun /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)
    # """
    # grad_sun = jnp.zeros(3)

    # Hamilton eqns. for integration (global minus, s.t. we go back in time).
    dyds = -jnp.array([
        u_i, 1./(1.+z)**2 * grad_sun
    ])

    return dyds


@jax.jit
def backtrack_1_neutrino(init_vector, s_int_steps, z_int_steps, kpc, s):

    """
    Simulate trajectory of 1 neutrino. Input is 6-dim. vector containing starting positions and velocities of neutrino. Solves ODEs given by the EOMs function with an jax-accelerated integration routine, using the diffrax library. Output are the positions and velocities at each timestep, which was specified with diffrax.SaveAt. 
    """

    # Initial vector in correct shape for EOMs function
    y0 = init_vector.reshape(2,3)

    # ODE solver setup
    term = diffrax.ODETerm(EOMs_sun)
    t0 = s_int_steps[0]
    t1 = s_int_steps[-1]
    dt0 = (s_int_steps[2]) / 50
    

    ### ------------------ ###
    ### Integration Solver ###
    ### ------------------ ###
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.ConstantStepSize()

    args = (s_int_steps, z_int_steps, kpc, s)

    # Specify timesteps where solutions should be saved
    saveat = diffrax.SaveAt(ts=jnp.array(s_int_steps))
    
    # Solve the coupled ODEs, i.e. the EOMs of the neutrino
    sol = diffrax.diffeqsolve(
        term, solver, 
        t0=t0, t1=t1, 
        dt0=dt0, 
        y0=y0, max_steps=10000,
        saveat=saveat, 
        stepsize_controller=stepsize_controller,
        args=args)
    
    trajectory = sol.ys.reshape(365,6)

    # Only return the initial [0] and last [-1] positions and velocities
    return jnp.stack([trajectory[0], trajectory[-1]])


def simulate_neutrinos_1_pix(init_xyz, init_vels, common_args):

    """
    Function for the multiprocessing routine below, which simulates all neutrinos for 1 pixel on the healpix skymap.
    """

    # Neutrinos per pixel
    nus = init_vels.shape[0]

    # Make vector with same starting position but different velocities
    init_vectors = jnp.array(
        [jnp.concatenate((init_xyz, init_vels[k])) for k in range(nus)])


    trajectories = jnp.array([
        backtrack_1_neutrino(vec, *common_args) for vec in init_vectors])
    
    return trajectories  # shape = (neutrinos, 2, 6)



# Lists for pixel and total number densities
pix_dens_l = []
tot_dens_l = []

# File name ending
end_str = f'day1'

# Initial position (Earth)
init_xyz = np.array([float(init_dis), 0., 0.])
jnp.save(f'{pars.directory}/init_xyz_{end_str}.npy', init_xyz)


### ============== ###
### Run Simulation ###
### ============== ###

print(f"*** Simulation for modulation ***")

sim_start = time.perf_counter()

with open(f'{pars.directory}/sim_parameters.yaml', 'r') as file:
    sim_setup = yaml.safe_load(file)

pix_sr_sim = sim_setup['pix_sr']        # Size (in sr) of all-sky healpixels
Npix = sim_setup["Npix"]                # Number of healpixels 
nu_per_pix = sim_setup["momentum_num"]  # Number of neutrinos per healpixel

init_vels = np.load(f'{pars.directory}/initial_velocities.npy')  
# shape = (Npix, neutrinos per pixel, 3)

common_args = (s_int_steps, z_int_steps, Params.kpc, Params.s)

if pars.testing:

    # Simulate all neutrinos along 1 pixel, without multiprocessing
    nu_vectors = simulate_neutrinos_1_pix(init_xyz, init_vels[0], common_args)

else:

    # Use ProcessPoolExecutor to distribute the simulations across processes:
    # 1 process (i.e. CPU) simulates all neutrinos for one healpixel.
    with ProcessPoolExecutor(CPUs_sim) as executor:
        futures = [
            executor.submit(
                simulate_neutrinos_1_pix, init_xyz, init_vels[pixel], common_args) for pixel in range(Npix)]
        
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

    jnp.save(
        f"{pars.directory}/pixel_densities_{end_str}.npy", jnp.array(pix_dens_l))
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

    jnp.save(
        f"{pars.directory}/total_densities_{end_str}.npy", jnp.array(tot_dens_l))
    print(f"Analysis time: {tot_time/60.:.2f} min, {tot_time/(60**2):.2f} h\n")


total_time = time.perf_counter() - total_start
print(f"Total time: {total_time/60.:.2f} min, {total_time/(60**2):.2f} h")
