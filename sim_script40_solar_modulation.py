from Shared.specific_CNB_sim import *


total_start = time.perf_counter()

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('--directory', required=True)
parser.add_argument('-hn', '--halo_num', required=True)
parser.add_argument(
    '--testing', required=True, action=argparse.BooleanOptionalAction)
pars = parser.parse_args()

# Instead of SimData.simulation_setup, define only parameters you need
CPUs_sim = 128
neutrinos = 1000

nu_massrange = jnp.load(f'{pars.directory}/neutrino_massrange_eV.npy')*Params.eV
simdata = SimData(pars.directory)


@jax.jit
def EOMs_sun(s_val, y, args):

    # Unpack the input data
    s_int_steps, z_int_steps, t_int_steps, sun_pos, sun_vel, kpc, km, s = args

    # Initialize vector.
    x_i, u_i = y

    # Switch to "numerical reality" here.
    x_i *= kpc
    u_i *= (kpc/s)

    # Find z (redshift) corresponding to s_val via interpolation.
    z = Utils.jax_interpolate(s_val, s_int_steps, z_int_steps)

    # Find t (lookback time) corresponding to s_val via interpolation.
    t = Utils.jax_interpolate(s_val, s_int_steps, t_int_steps)*s

    # Compute current position of Sun w.r.t. CNB(==CMB) frame, given starting 
    # position and velocity vectors of current day being simulated
    sun_pos_t = sun_pos - sun_vel*t

    # Compute gradient of sun.
    eps = 696_340*km  # solar radius in numerical units
    grad_sun = SimExec.sun_gravity(x_i, eps, sun_pos_t)

    # Switch to "physical reality" here.
    grad_sun /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)

    # Relativistic EOMs for integration (global minus, s.t. we go back in time).
    dyds = -jnp.array([
        u_i/(1+z)**2 / jnp.sqrt(jnp.sum(u_i**2) + (1+z)**-2), 
        1/(1+z)**2 * grad_sun
    ])

    return dyds


@jax.jit
def backtrack_1_neutrino(
    init_vector, s_int_steps, z_int_steps, t_int_steps, sun_pos, sun_vel, 
    kpc, km, s):

    """
    Simulate trajectory of 1 neutrino. Input is 6-dim. vector containing starting positions and velocities of neutrino. Solves ODEs given by the EOMs function with an jax-accelerated integration routine, using the diffrax library. Output are the positions and velocities at each timestep, which was specified with diffrax.SaveAt. 
    """

    # Initial vector in correct shape for EOMs function
    y0 = init_vector.reshape(2,3)

    # ODE solver setup
    term = diffrax.ODETerm(EOMs_sun)
    t0 = s_int_steps[0]
    t1 = s_int_steps[-1]
    dt0 = jnp.median(jnp.diff(s_int_steps)) / 1000

    ### ------------------ ###
    ### Integration Solver ###
    ### ------------------ ###

    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

    # Specify timesteps where solutions should be saved
    # saveat = diffrax.SaveAt(steps=True, ts=jnp.array(s_int_steps))
    saveat = diffrax.SaveAt(ts=jnp.array(s_int_steps))

    # Common arguments for solver
    args = (
        s_int_steps, z_int_steps, t_int_steps, sun_pos, sun_vel, 
        kpc, km, s)
    
    # Solve the coupled ODEs, i.e. the EOMs of the neutrino
    sol = diffrax.diffeqsolve(
        term, solver, 
        t0=t0, t1=t1, 
        dt0=dt0, 
        y0=y0, max_steps=5000,
        saveat=saveat, 
        stepsize_controller=stepsize_controller,
        args=args, throw=False)
    
    # Full neutrino trajectory (positions and velocities) for 1 year
    trajectory = sol.ys.reshape(-1,6)

    # note: integration stops close to end_point and timesteps then suddenly
    # note: switch to inf values. So we take [-2] (last finite) values.
    return jnp.stack([trajectory[0], trajectory[-2]])


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



# Load Sun's positions and velocity vectors (w.r.t. CNB==CMB) in Earth-GC frame
sun_positions, sun_velocities = SimPlot.calculate_sun_position_and_velocity(
    2024)
sun_positions *= Params.AU
sun_velocities *= Params.km/Params.s

# Repeat sun positions and velocities arrays to match length of special 2year+1
# sun_pos_tiled = jnp.repeat(sun_positions, 2, axis=0)
# sun_pos_ext = jnp.append(sun_pos_tiled, sun_pos_tiled[0][None,:], axis=0)
# sun_vel_tiled = jnp.repeat(sun_velocities_CNB, 2, axis=0)
# sun_vel_ext = jnp.append(sun_vel_tiled, sun_vel_tiled[0][None,:], axis=0)

# Redshift from today until (365*2)+1 days ago
z_int_steps_all = jnp.load(f'{pars.directory}/z_int_steps_2years.npy')
# z_int_steps = jnp.load(f'{pars.directory}/z_int_steps_1year.npy')

# Corresponding integration variable for EOMs
s_int_steps_all = jnp.load(f'{pars.directory}/s_int_steps_2years.npy')
# s_int_steps = jnp.load(f'{pars.directory}/s_int_steps_1year.npy')

# Corresponding lookback time
t_int_steps_all = jnp.load(f'{pars.directory}/t_int_steps_2years.npy')
# t_int_steps = jnp.load(f'{pars.directory}/t_int_steps_1year.npy')

# Initial position (Earth) is at (0,0,0) always, we move the Sun around in 
# special Earth-GC frame
init_xyz = np.array([0., 0., 0.])
jnp.save(f'{pars.directory}/init_xyz_modulation.npy', init_xyz)

# Lists for pixel and total number densities
tot_dens_days_l = []
pix_dens_days_l = []
# for day, day in enumerate(range(365)):
for day in range(10):  #note: testing

    # Select 1 years worth of redshift/time steps, +1 because we select second 
    # last time step in integration routine due to infinities issue (see above)
    z_int_steps = z_int_steps_all[day:day+365+1]
    s_int_steps = s_int_steps_all[day:day+365+1]
    t_int_steps = t_int_steps_all[day:day+365+1]

    # File name ending
    end_str = f"day{day+1}"


    ### ============== ###
    ### Run Simulation ###
    ### ============== ###

    print(f"*** Simulation for day {day+1}/365 ***")
    sim_start = time.perf_counter()

    # Load initial velocities, same for each day due to Earth-GC frame
    init_vels = np.load(f'{pars.directory}/initial_velocities.npy')  
    # shape = (Npix, neutrinos per pixel, 3)

    common_args = (
        s_int_steps, z_int_steps, t_int_steps, 
        sun_positions[day], sun_velocities[day], 
        Params.kpc, Params.km, Params.s)

    # """
    if pars.testing:
        # Simulate all neutrinos along 1 pixel, without multiprocessing
        nu_vectors = simulate_neutrinos_1_pix(
            init_xyz, init_vels[0], common_args)
    else:
        # Use ProcessPoolExecutor to distribute the simulations across processes:
        # 1 process (i.e. CPU) simulates all neutrinos for one healpixel.
        with ProcessPoolExecutor(CPUs_sim) as executor:
            futures = [
                executor.submit(
                    simulate_neutrinos_1_pix, init_xyz, init_vels[pixel], common_args) for pixel in range(simdata.Npix)]
            
            # Wait for all futures to complete and collect results in order
            nu_vectors = jnp.array([future.result() for future in futures])
    # """


    # Save all sky neutrino vectors for current halo
    if pars.testing:
        jnp.save(f'{pars.directory}/vectors_{end_str}_TEST.npy', nu_vectors)
    else:
        jnp.save(f'{pars.directory}/vectors_{end_str}.npy', nu_vectors)


    sim_time = time.perf_counter()-sim_start
    print(f"Simulation time: {sim_time/60.:.2f} min, {sim_time/(60**2):.2f} h")


    ### ======================== ###
    ### Compute number densities ###
    ### ======================== ###

    """
    ana_start = time.perf_counter()

    # Compute individual number densities for each healpixel
    pix_start = time.perf_counter()

    nu_allsky_masses = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV
    pix_dens = Physics.number_densities_all_sky(
        v_arr=nu_vectors[..., 3:],
        m_arr=nu_allsky_masses,
        pix_sr=simdata.pix_sr,
        args=Params())
    pix_dens_l.append(jnp.squeeze(pix_dens))

    pix_time = time.perf_counter() - pix_start

    jnp.save(
        f"{pars.directory}/pixel_densities_{end_str}.npy", jnp.array(pix_dens_l))

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

    ana_time = time.perf_counter() - ana_start
    print(f"Analysis time: {ana_time/60.:.2f} min, {ana_time/(60**2):.2f} h\n")
    """


total_time = time.perf_counter() - total_start
print(f"Total time: {total_time/60.:.2f} min, {total_time/(60**2):.2f} h")
