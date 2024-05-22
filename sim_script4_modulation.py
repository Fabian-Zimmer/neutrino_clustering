from Shared.specific_CNB_sim import *
import pandas as pd

total_start = time.perf_counter()

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('--directory', required=True)
pars = parser.parse_args()


# @jax.jit
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

    # Compute gradient of sun.
    grad_sun = SimExec.sun_gravity(x_i, jnp.array([0.,0.,0.]))


    # Switch to "physical reality" here.
    grad_sun /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)


    # Hamilton eqns. for integration (global minus, s.t. we go back in time).
    dyds = -jnp.array([
        u_i, 1./(1.+z)**2 * grad_sun
    ])

    return dyds


def save_to_file(data, filename):
    np.savetxt(filename, data, fmt='%s')


@jax.jit
def compute_trajectory(init_vector, s_int_steps, z_int_steps, kpc, s):
    """
    Core computation function to be JIT compiled. This function performs the ODE solving without error handling.
    """
    # Initial vector in correct shape for EOMs function
    y0 = init_vector.reshape(2, 3)

    # ODE solver setup
    term = diffrax.ODETerm(EOMs_sun)
    t0 = s_int_steps[0]
    t1 = s_int_steps[-1]
    dt0 = (s_int_steps[4] - s_int_steps[3]) / 10  # Adjusting initial step size

    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-5)

    saveat = diffrax.SaveAt(ts=jnp.array(s_int_steps))
    max_steps = 10_000

    sol = diffrax.diffeqsolve(
        term, solver,
        t0=t0, t1=t1, dt0=dt0, y0=y0, max_steps=max_steps,
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=(s_int_steps, z_int_steps, kpc, s), throw=False)

    trajectory = sol.ys.reshape(len(s_int_steps), 6)

    return sol, jnp.stack([trajectory[0], trajectory[-1]])



def backtrack_1_neutrino(init_vector, s_int_steps, z_int_steps, kpc, s, day, pixel, directory):
    try:
        sol, result = compute_trajectory(init_vector, s_int_steps, z_int_steps, kpc, s)
        return result
    except Exception as e:
        print("Solver failed to converge within the max steps limit.")

        ts_filename = os.path.join(
            directory, f"sol_ts_day{day}_pixel{pixel}.txt")
        stats_filename = os.path.join(
            directory, f"sol_stats_day{day}_pixel{pixel}.txt")

        save_to_file(sol.ts, ts_filename)
        save_to_file(sol.stats, stats_filename)

        raise e



"""
# @jax.jit
def backtrack_1_neutrino(
    init_vector, s_int_steps, z_int_steps, kpc, s):

    # Initial vector in correct shape for EOMs function
    y0 = init_vector.reshape(2,3)

    # ODE solver setup
    term = diffrax.ODETerm(EOMs_sun)
    t0 = s_int_steps[0]
    t1 = s_int_steps[-1]
    dt0 = (s_int_steps[4] - s_int_steps[3]) / 10
    

    ### ------------- ###
    ### Dopri5 Solver ###
    ### ------------- ###
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-5)
    # stepsize_controller = diffrax.PIDController(rtol=1e-1, atol=1e-3)
    # note: no change for tighter rtol and atol, e.g. rtol=1e-5, atol=1e-9


    ### ------------- ###
    ### Alter. Solver ###
    ### ------------- ###
    # solver = diffrax.Tsit5()
    # stepsize_controller = diffrax.PIDController(rtol=1e-1, atol=1e-2)


    # Specify timesteps where solutions should be saved
    saveat = diffrax.SaveAt(ts=jnp.array(s_int_steps))
    
    try:
        # Solve the coupled ODEs, i.e. the EOMs of the neutrino
        sol = diffrax.diffeqsolve(
            term, solver, 
            t0=t0, t1=t1, dt0=dt0, y0=y0, max_steps=100_000,
            saveat=saveat, stepsize_controller=stepsize_controller,
            args=(s_int_steps, z_int_steps, kpc, s))
    
        trajectory = sol.ys.reshape(len(s_int_steps),6)
        return jnp.stack([trajectory[0], trajectory[-1]]) 
    
    # Print relevant information when the solver fails
    ts_filename = f"{pars.directory}/sol_ts.txt"
    stats_filename = f"{pars.directory}/sol_stats.txt"
    
    save_to_file(sol.ts, ts_filename)
    save_to_file(sol.stats, stats_filename)

    trajectory = sol.ys.reshape(len(s_int_steps),6)
    return jnp.stack([trajectory[0], trajectory[-1]]) 
"""

def simulate_neutrinos_1_pix(init_xyz, init_vels, common_args, day, directory):
    nus = init_vels.shape[0]
    init_vectors = jnp.array(
        [jnp.concatenate((init_xyz, init_vels[k])) for k in range(nus)]
    )

    trajectories = jnp.array([
        backtrack_1_neutrino(vec, *common_args, day=day, pixel=k, directory=directory) for k, vec in enumerate(init_vectors)
    ])

    return trajectories  # shape = (nus, 2, 3)



# Integration steps and massrange
z_int_steps = jnp.load(f'{pars.directory}/z_int_steps_1year.npy')
s_int_steps = jnp.load(f'{pars.directory}/s_int_steps_1year.npy')
nu_massrange = jnp.load(f'{pars.directory}/neutrino_massrange_eV.npy')*Params.eV

# Earth distances from file, and days array
df = pd.read_excel('Data/Earth-Sun_distances.xlsx')
ES_distances = jnp.array(
    df.iloc[:, 1::2].apply(pd.to_numeric, errors='coerce').stack().reset_index(drop=True).tolist())[:-1]
days = jnp.arange(1,len(ES_distances)+1)

# Lists for pixel and total number densities
pix_dens_l = []
tot_dens_l = []

for day, ES_dist in zip(days, ES_distances):

    init_xyz = jnp.array([ES_dist*Params.AU/Params.kpc, 0., 0.])

    if day == days[0]:
        print(init_xyz)
        print(s_int_steps[0], s_int_steps[-1])
        print(z_int_steps[0], z_int_steps[-1])

    print(f"*** Simulation for day={day}/{len(days)} ***")


    ### ============== ###
    ### Run Simulation ###
    ### ============== ###

    sim_start = time.perf_counter()

    with open(f'{pars.directory}/sim_parameters.yaml', 'r') as file:
        sim_setup = yaml.safe_load(file)

    # Size (in sr) of all-sky healpixels
    pix_sr_sim = sim_setup['pix_sr']

    # Number of healpixels 
    Npix = sim_setup["Npix"]

    # Number of neutrinos per healpixel
    nu_per_pix = sim_setup["momentum_num"]

    #TODO: this should be changing each day as Earth orientation changes w.r.t.
    #TODO: phase-space results from main sim
    init_vels = np.load(f'{pars.directory}/initial_velocities.npy')
    # shape = (Npix, neutrinos per pixel, 3)

    # Common arguments for simulation
    common_args = (s_int_steps, z_int_steps, Params.kpc, Params.s)

    # Use ProcessPoolExecutor to distribute the simulations across processes
    with ProcessPoolExecutor(128) as executor:
        futures = [
            executor.submit(
                simulate_neutrinos_1_pix, init_xyz, init_vels[pixel], common_args, day, pars.directory) for pixel in range(Npix)
        ]

        # Wait for all futures to complete and collect results in order
        nu_vectors = jnp.array([future.result() for future in futures])

    # Save all sky neutrino vectors for current halo
    jnp.save(f'{pars.directory}/vectors_day{day}.npy', nu_vectors)

    sim_time = time.perf_counter()-sim_start
    print(f"Simulation time: {sim_time/60.:.2f} min, {sim_time/(60**2):.2f} h")


    ### ======================== ###
    ### Compute number densities ###
    ### ======================== ###

    ana_start = time.perf_counter()

    # Compute individual number densities for each healpixel
    nu_allsky_masses = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV
    pix_dens = Physics.number_densities_all_sky(
        v_arr=nu_vectors[..., 3:],
        m_arr=nu_allsky_masses,
        pix_sr=pix_sr_sim,
        args=Params())
    pix_dens_l.append(jnp.squeeze(pix_dens))
    
    # Compute total number density, by using all neutrino vectors for integral
    tot_dens = Physics.number_densities_mass_range(
        v_arr=nu_vectors.reshape(-1, 2, 6)[..., 3:], 
        m_arr=nu_massrange, 
        pix_sr=4*Params.Pi,
        args=Params())
    tot_dens_l.append(jnp.squeeze(tot_dens))
    
    ana_time = time.perf_counter() - ana_start
    print(f"Analysis time: {ana_time/60.:.2f} min, {ana_time/(60**2):.2f} h\n")


# Save number density arrays for all halos
jnp.save(f"{pars.directory}/total_densities_days.npy", jnp.array(tot_dens_l))
jnp.save(f"{pars.directory}/pixel_densities_days.npy", jnp.array(pix_dens_l))

tot_time = time.perf_counter() - total_start
print(f"Total time: {tot_time/60.:.2f} min, {tot_time/(60**2):.2f} h")