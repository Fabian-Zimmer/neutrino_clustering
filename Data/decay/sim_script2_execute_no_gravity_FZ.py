from Shared.specific_CNB_sim import *
from Shared.specific_CNB_decay import *

total_start = time.perf_counter()

# Note: CHANGES
# - gamma and m_p, m_d as parser input
# - load simulation specific quantities (e.g. Npix) with simdata
# - changed find_nearest function to return idx, val
# - renaming: EOMs_no_gravity_decay
#   - renaming variables & restructuring of EOMs function content
#   - m_p and m_d were used inversely (quite sure but not 100% certain...)
# - Added common_args routine
# - Added numerical/physical reality switches, since we manipulate velocities
#   - For this I added kpc and s units as input in common_args
# - Changed random selection routine, since jax does it differently
# - Decay might happen multiple times, since s_val can change little:
#   - requires new "dummy" integration variable that is used to track if decay happened


# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('--directory', required=True)
parser.add_argument('--m_p', required=True)
parser.add_argument('--m_d', required=True)
parser.add_argument('--gamma', required=True)
parser.add_argument(
    '--pixel_densities', required=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    '--total_densities', required=True, action=argparse.BooleanOptionalAction)
pars = parser.parse_args()

# Instantiate simulation parameter class
simdata = SimData(pars.directory)

#! Chosen masses for parent and daughter neutrinos
m_parent = pars.m_parent
m_daughter = pars.m_daughter

### --------------------------------------- ###
### Load arrays generated before simulation ###
### --------------------------------------- ###

# From original simulation code
z_array = jnp.load(
    f'{pars.directory}/z_int_steps.npy')
neutrino_momenta = jnp.load(
    f'{pars.directory}/neutrino_momenta.npy')

# Created in Decay.decay_neutrinos function
decayed_neutrinos_index_z = jnp.load(
    f'decays_gamma/decayed_neutrinos_index_z_{pars.gamma}.npy', allow_pickle=True)
decayed_neutrinos_z = jnp.load(
    f'decays_gamma/decayed_neutrinos_z_{pars.gamma}.npy')
decayed_neutrinos_theta = jnp.load(
    f'{pars.directory}/decayed_neutrinos_theta_{pars.gamma}.npy')
decayed_neutrinos_phi = jnp.load(
    f'{pars.directory}/decayed_neutrinos_phi_{pars.gamma}.npy')

# note: old predetermined angle-parent_momenta rray
""" 
# Created in Decay.daughter_momentum_4 function
angle_momentum_decay = jnp.load(
    f'{pars.directory}/neutrino_angle_momentum_decay.npy') 
"""

# note: loading combined anlge-parent_momenta array from new routinge
angle_momentum_decay = jnp.load(
    f'{pars.directory}/allowed_decay_angles_and_momenta.npy')
decay_angles = angle_momentum_decay[..., 0]
parent_momenta = angle_momentum_decay[..., 1]
parent_momenta = parent_momenta.at[parent_momenta <= 0.0].set(np.nan)


# Create halo batch, files and other simulation setup parameters and arrays
DM_mass, CPUs_sim, neutrinos, init_dis, zeds_snaps, z_int_steps, s_int_steps, nu_massrange = SimData.simulation_setup(
    sim_dir=pars.directory,
    m_lower=None,
    m_upper=None,
    m_gauge=None,
    halo_num_req=None,
    no_gravity=True)


@jax.jit
def find_nearest(array, value):
    idx = jnp.argmin(jnp.abs(array - value))
    return idx, array[idx]


#@jax.jit
def EOMs_no_gravity_decay(s_val, y, args):

    # Load arguments
    Nr_index, decay_angles, parent_momenta, decayed_neutrinos_z, z_array, nu_momenta, m_p, m_d, kpc, s = args

    # Read velocity from input vector
    _, v_in, decay_tracker = y

    # Switch to "numerical reality"
    v_in *= (kpc/s)

    # Find z corresponding to s via interpolation.
    z =  Utils.jax_interpolate(s_val, s_int_steps, z_int_steps)

    # Find correct index for z_array, that corresponds to current z
    z_index, _ = find_nearest(z_array, z)
    
    # Read current and previous decay flag number (1 = not decayed, 0 = decayed)
    # Combination of pre = 1 and now = 0 is unique, and is condition for decay
    now_nu_number = jnp.int16(decayed_neutrinos_z[z_index, Nr_index])
    pre_nu_number = jnp.int16(decayed_neutrinos_z[z_index-1, Nr_index])


    ### --------------------------------- ###
    ### Only relevant if neutrino decayed ###
    ### --------------------------------- ###

    # Integration backwards in time => neutrino goes from daughter to parent!
    # So current velocity is that of daughter neutrino, and we compute velocity of parent

    # Convert to momentum
    p_in = m_d*jnp.linalg.norm(v_in, axis=-1)

    # Find index in nu_momenta array of value closest to current momentum
    # (nu_momenta array already has numerical units attached)
    p_index, _ = find_nearest(nu_momenta, p_in)
    
    # Random key works differently with jax. Make new key for each s_val
    s_key = jax.random.key(jnp.int64(s_val))

    # Randomly selecting from new (combined) angle-parent_momentum array
    valid_mask = ~jnp.isnan(parent_momenta[p_index])
    valid_indices = jnp.where(valid_mask)[0]
    random_idx = jax.random.choice(s_key, valid_indices)
    decay_theta = decay_angles[p_index, random_idx]
    p_parent = parent_momenta[p_index, random_idx]

    # Also randomly select phi angle between 0 and 360
    decay_phi = jax.random.uniform(s_key)*360

    # Compute parent velocity vector in cartesian coordinates
    v_parent = jnp.squeeze(
        jnp.array(
            [
                (1/m_p)*p_parent*jnp.sin(decay_theta)*jnp.cos(decay_phi),  # x
                (1/m_p)*p_parent*jnp.sin(decay_theta)*jnp.sin(decay_phi),  # y
                (1/m_p)*p_parent*jnp.cos(decay_theta)                      # z
            ]
        )
    )
    ### --------------------------------- ###
    

    # If neutrino has decayed: Assign new velocity
    def true_func(v_parent):
        decay_tracker = decay_tracker.at[:].set(1)
        return v_parent
    
    # If neutrino has not decayed: Keep current velocity
    def false_func(v_in):
        return v_in

    # Get new/current velocity depending on decay condition being True/False
    v_out = jax.lax.cond(
        (now_nu_number == 0) & (pre_nu_number == 1) & jnp.all(decay_tracker == 0), 
        v_parent, true_func, 
        v_in, false_func
    )
   
    # Switch to "physical reality"
    v_out /= (kpc/s)

    dyds = -jnp.array([
        v_out, jnp.zeros(3), decay_tracker
    ])
    
    return dyds


#@jax.jit
def backtrack_1_neutrino(
    init_vector, s_int_steps, decay_angles, parent_momenta, decayed_neutrinos_z, 
    z_array, neutrino_momenta, m_parent, m_daughter, kpc, s):

    """
    Simulate trajectory of 1 neutrino. Input is 6-dim. vector containing starting positions and velocities of neutrino. Solves ODEs given by the EOMs function with an jax-accelerated integration routine, using the diffrax library. Output are the positions and velocities at each timestep, which was specified with diffrax.SaveAt. 
    """

    # Read initial neutrino vector (positions and momenta) and neutrino tracking number
    y0_r, Nr = init_vector[0:-1], init_vector[-1]

    # Create array used as "dummy" integration vector to keep decay to occur only once
    decay_tracker = jnp.zeros(3)

    # 
    y0 = jnp.concatenate((y0_r, decay_tracker)).reshape(3,3)

    # Initial vector in correct shape for EOMs function
    # y0 = y0_r.reshape(2,3)
    
    # ODE solver setup
    term = diffrax.ODETerm(EOMs_no_gravity_decay)
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
        saveat=saveat, stepsize_controller=stepsize_controller, 
        args=(
            Nr.astype(int), decay_angles, parent_momenta, decayed_neutrinos_z, z_array, 
            neutrino_momenta, m_parent, m_daughter, kpc, s)
    )
    
    # Total integration vector has 9 elements (3 from decay_tracker)
    # We only need first 6
    trajectory = sol.ys.reshape(100, 9)[:, :6]

    # Only return the initial [0] and last [-1] positions and velocities
    return jnp.stack([trajectory[0], trajectory[-1]])


def simulate_neutrinos_1_pix(init_xyz, init_vels, Nr_column, common_args):

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
        backtrack_1_neutrino(vec, *common_args) for vec in init_vectors])

    return trajectories  # shape = (neutrinos, 2, 3)


# Lists for pixel and total number densities: d=daughter, p=parent, nd=no_decay
pix_dens_l_d = []
tot_dens_l_d = []
pix_dens_l_p = []
tot_dens_l_p = []
pix_dens_l_nd = []
tot_dens_l_nd = []

# File name ending
end_str = f'halo1'

# Initial position (Earth)
init_xyz = np.array([float(init_dis), 0., 0.])
jnp.save(f'{pars.directory}/init_xyz_{end_str}.npy', init_xyz)


### ============== ###
### Run Simulation ###
### ============== ###

print(f"*** Simulation for no_gravity with decay ***")

sim_start = time.perf_counter()

pix_sr_sim = simdata.pix_sr
Npix = int(simdata.Npix)
nu_per_pix = int(simdata.p_num)
nu_total = int(simdata.nus_in_sim)

init_vels = np.load(f'{pars.directory}/initial_velocities.npy')
# shape = (Npix, neutrinos per pixel, 3)

Nr_column= jnp.arange(nu_total).reshape(Npix, nu_per_pix)

# Common arguments for simulation
common_args = (
    s_int_steps, decay_angles, parent_momenta, decayed_neutrinos_z, z_array, 
    neutrino_momenta, m_parent, m_daughter, Params.kpc, Params.s)

# Use ProcessPoolExecutor to distribute the simulations across processes:
# 1 process (i.e. CPU) simulates all neutrinos for one healpixel.
with ProcessPoolExecutor(CPUs_sim) as executor:
    futures = [
        executor.submit(
            simulate_neutrinos_1_pix, init_xyz, init_vels[pixel], 
            Nr_column[pixel], common_args) for pixel in range(Npix)
    ]
    
    # Wait for all futures to complete and collect results in order
    nu_vectors = jnp.array([future.result() for future in futures])


### Manipulate array values for number density computations
#? Setting elements to zero, where neutrinos have decayed
setter = np.concatenate([arr for arr in  decayed_neutrinos_index_z if arr.size > 0])
nu_vectors = nu_vectors.reshape((nu_total, 2, 6))
nu_vectors_p = nu_vectors.at[setter, 0, 3:6].set(0)
nu_vectors_p = nu_vectors_p.at[setter, 1, 3:6].set(0)
nu_vectors_p = nu_vectors_p.reshape((Npix, nu_per_pix, 2, 6))

#? 
nu_vectors_d = jnp.zeros(np.shape(nu_vectors))
nu_vectors_d = nu_vectors_d.at[setter,0,3:6].set(nu_vectors[setter,0,3:6])
nu_vectors_d = nu_vectors_d.at[setter,1,3:6].set(nu_vectors[setter,1,3:6])
nu_vectors_d = nu_vectors_d.reshape((Npix, nu_per_pix, 2, 6))

#? Set elements in nu_vectors_p and nu_vectors_d based on setter
jnp.save(f'{pars.directory}/vectors_{end_str}_p_{pars.gamma}.npy', nu_vectors_p)
jnp.save(f'{pars.directory}/vectors_{end_str}_d_{pars.gamma}.npy', nu_vectors_d)


# Save all sky neutrino vectors for current halo
#jnp.save(f'{pars.directory}/vectors_{end_str}.npy', nu_vectors)
sim_time = time.perf_counter()-sim_start
print(f"Simulation time: {sim_time/60.:.2f} min, {sim_time/(60**2):.2f} h")


### ======================== ###
### Compute number densities ###
### ======================== ###

if pars.pixel_densities:

    # Compute individual number densities for each healpixel
    pix_start = time.perf_counter()

    # Selected neutrino masses
    nu_allsky_masses = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV

    # Pixel densities: daughter neutrinos
    pix_dens_d = Decay.number_densities_all_sky(
        v_arr=nu_vectors_d[..., 3:],
        m_arr=nu_allsky_masses,
        pix_sr=pix_sr_sim,
        args=Params())
    pix_dens_l_d.append(jnp.squeeze(pix_dens_d))

    # Pixel densities: parent neutrinos
    pix_dens_p = Decay.number_densities_all_sky(
        v_arr=nu_vectors_p[..., 3:],
        m_arr=nu_allsky_masses,
        pix_sr=pix_sr_sim,
        args=Params())
    pix_dens_l_p.append(jnp.squeeze(pix_dens_p))
    
    # Pixel densities: no_decay
    pix_dens_nd = Decay.number_densities_all_sky(
        v_arr=nu_vectors[..., 3:],
        m_arr=nu_allsky_masses,
        pix_sr=pix_sr_sim,
        args=Params())
    pix_dens_l_nd.append(jnp.squeeze(pix_dens_nd))

    pix_time = time.perf_counter() - pix_start
    print(f"Analysis time: {pix_time/60.:.2f} min, {pix_time/(60**2):.2f} h\n")

    # Save pixel densities
    jnp.save(f"{pars.directory}/pixel_densities_{pars.gamma}_d.npy", jnp.array(pix_dens_l_d))
    jnp.save(f"{pars.directory}/pixel_densities_{pars.gamma}_p.npy", jnp.array(pix_dens_l_p))
    jnp.save(f"{pars.directory}/pixel_densities_{pars.gamma}_nd.npy", jnp.array(pix_dens_l_nd))


if pars.total_densities:

    # Compute total number density, by using all neutrino vectors for integral
    tot_start = time.perf_counter()

    # Total densities: daughter neutrinos
    tot_dens_d = Decay.number_densities_mass_range_decay(
        v_arr=nu_vectors_d.reshape(-1, 2, 6)[..., 3:], 
        m_arr=nu_massrange, 
        pix_sr=4*Params.Pi,
        args=Params())
    tot_dens_l_d.append(jnp.squeeze(tot_dens_d))

    # Total densities: parent neutrinos
    tot_dens_p = Decay.number_densities_mass_range_decay(
        v_arr=nu_vectors_p.reshape(-1, 2, 6)[..., 3:], 
        m_arr=nu_massrange, 
        pix_sr=4*Params.Pi,
        args=Params())
    tot_dens_l_p.append(jnp.squeeze(tot_dens_p))

    # Total densities: no_decay
    tot_dens_nd = Decay.number_densities_mass_range_decay(
        v_arr=nu_vectors.reshape(-1, 2, 6)[..., 3:], 
        m_arr=nu_massrange, 
        pix_sr=4*Params.Pi,
        args=Params())
    tot_dens_l_nd.append(jnp.squeeze(tot_dens_nd))

    tot_time = time.perf_counter() - tot_start
    print(f"Analysis time: {tot_time/60.:.2f} min, {tot_time/(60**2):.2f} h\n")
    
    # Save total densities
    jnp.save(f"{pars.directory}/total_densities_{pars.gamma}_d.npy", jnp.array(tot_dens_l_d))
    jnp.save(f"{pars.directory}/total_densities_{pars.gamma}_p.npy", jnp.array(tot_dens_l_p))
    jnp.save(f"{pars.directory}/total_densities_{pars.gamma}_nd.npy", jnp.array(tot_dens_l_nd))
    

total_time = time.perf_counter() - total_start
print(f"Total time: {total_time/60.:.2f} min, {total_time/(60**2):.2f} h")
