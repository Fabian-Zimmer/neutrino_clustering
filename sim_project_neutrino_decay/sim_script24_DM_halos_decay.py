from Shared.specific_CNB_sim import *
from Shared.specific_CNB_decay import *
total_start = time.perf_counter()


#note 1 (Fabian): single dash, e.g. -d, is just alternative, unnecessary for us.
#note 2 (Fabian): pixel/total densities were originally computed without asking, implemented this later for testing and/or giving choice to user (e.g. in case user is debugging sim only)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', required=True)
parser.add_argument('-mg', '--mass_gauge', required=True)
parser.add_argument('-ml', '--mass_lower', required=True)
parser.add_argument('-mu', '--mass_upper', required=True)
parser.add_argument('-hn', '--halo_num', required=True)
parser.add_argument('--halo_num', required=True)
parser.add_argument('--m_lightest', required=True)
parser.add_argument('--gamma', required=True)
parser.add_argument(
    '--pixel_densities', required=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    '--total_densities', required=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    '--benchmark', required=True, action=argparse.BooleanOptionalAction)
pars = parser.parse_args()

# Instantiate simulation parameter class
simdata = SimData(pars.directory)

# Neutrino masses
m_daughter, _, m_parent = Physics.neutrino_masses(
    m_lightest=float(pars.m_lightest)*Params.eV, ordering="NO", args=Params())

### --------------------------------------- ###
### Load arrays generated before simulation ###
### --------------------------------------- ###

# Arrays from original simulation code
neutrino_momenta = jnp.load(
    f'{pars.directory}/neutrino_momenta.npy')

# Arrays created specifically for decay simulation
decayed_neutrinos_index_z = jnp.load(
    f'{pars.directory}/decayed_neutrinos_index_z_{pars.gamma}.npy', allow_pickle=True)
decayed_neutrinos_z = jnp.load(
    f'{pars.directory}/decayed_neutrinos_z_{pars.gamma}.npy')
z_array = jnp.load(
    f'{pars.directory}/decayed_neutrinos_bin_z_{pars.gamma}.npy')

# note: loading combined angle-parent_momenta array from new routine
angle_momentum_decay = jnp.load(
    f'{pars.directory}/allowed_decay_angles_and_momenta.npy')
decay_angles = angle_momentum_decay[..., 0]
parent_momenta = angle_momentum_decay[..., 1]

### Find first negative indices ###
# Create a mask where the condition (array < 0) is True
negative_mask = parent_momenta < 0

# Find the index of the first negative value in each row
first_negative_indices = jnp.argmax(negative_mask, axis=-1)

# If no negative values, set index to the last element in the row
row_has_negative = jnp.any(negative_mask, axis=-1)
first_negative_indices = jnp.where(
    row_has_negative, first_negative_indices, parent_momenta.shape[1] - 1)


# Create halo batch, files and other simulation setup parameters and arrays
DM_mass, CPUs_sim, neutrinos, init_dis, zeds_snaps, z_int_steps, s_int_steps, nu_massrange, data_dir, halo_batch_IDs, halo_num = SimData.simulation_setup(
    sim_dir=pars.directory,
    m_lower=pars.mass_lower,
    m_upper=pars.mass_upper,
    m_gauge=pars.mass_gauge,
    halo_num_req=pars.halo_num)

@jax.jit
def find_nearest(array, value):
    idx = jnp.argmin(jnp.abs(array - value))
    return idx, array[idx]


@jax.jit
def EOMs(s_val, y, args):
    """
    Solves the Equations of Motion (EOMs) for a particle within a cosmological simulation space. This involves determining the particle's position and velocity in relation to the simulation grid and applying the appropriate gravitational forces based on its location. It supports dynamic adaptation to different epochs by interpolating across snapshots of the universe at various redshifts, and employs conditional logic to distinguish between gravitational influences inside and outside the simulation's voxel cells. The function outputs the rate of change in position and velocity for the particle, used for the backwards in time integration method we use.
    """

    # Unpack the simulation grid data
    Nr_index, s_int_steps, decay_angles, parent_momenta, first_negative_indices, decayed_neutrinos_z, z_array, nu_momenta, m_p, m_d, z_int_steps, zeds_snaps, snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, dPsi_grid_data, cell_grid_data, cell_gens_data, DM_mass, kpc, s = args

    # Initialize vector.
    x_i, v_in, decay_tracker = y
  
    # Switch to "numerical reality"
    x_i *= kpc
    v_in *= (kpc/s)
    decay_tracker *= kpc
    
    # Find z corresponding to s via interpolation.
    z =  Utils.jax_interpolate(s_val, s_int_steps, z_int_steps)

    # Find correct index for z_array, that corresponds to current z
    z_index, _ = find_nearest(z_array, z)
    
    # Read current and previous decay flag number (1 = not decayed, 0 = decayed)
    # Combination of pre = 1 and now = 0 is unique, and is condition for decay
    # note: See below for first redshift bin condition
    now_nu_nr = jnp.int16(decayed_neutrinos_z[z_index,   Nr_index])
    pre_nu_nr = jnp.int16(decayed_neutrinos_z[z_index-1, Nr_index])


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
    valid_idx_end = first_negative_indices[p_index]
    random_idx = jax.random.randint(
        s_key, shape=(), minval=0, maxval=valid_idx_end + 1)
    decay_theta = jnp.deg2rad(decay_angles[p_index, random_idx])
    p_parent = parent_momenta[p_index, random_idx]

    # Also randomly select phi angle between 0 and 360
    decay_phi = jnp.deg2rad(jax.random.uniform(s_key)*360)

    # Compute parent velocity vector in cartesian coordinates
    v_parent = jnp.squeeze(
        jnp.array(
            [
                (1/m_p)*p_parent*jnp.sin(decay_theta)*jnp.cos(decay_phi),
                (1/m_p)*p_parent*jnp.sin(decay_theta)*jnp.sin(decay_phi),
                (1/m_p)*p_parent*jnp.cos(decay_theta)
            ]
        )
    )
    ### --------------------------------- ###


    # If neutrino has decayed: Assign new velocity, set decay_tracker to ones
    def true_func(v_parent, decay_tracker):
        decay_tracker += 1.*kpc
        return v_parent, decay_tracker
    
    # If neutrino has not decayed: Keep current velocity and decay_tracker
    def false_func(v_in, decay_tracker):
        return v_in, decay_tracker

    # Conditions for first (redshift) bin and rest of bins
    cond_0bin = (now_nu_nr==0)&(z_index==0)&jnp.all(decay_tracker==8.*kpc)
    cond_rest = (now_nu_nr==0)&(pre_nu_nr==1)&jnp.all(decay_tracker==8.*kpc)
    cond_comb = jnp.logical_or(cond_0bin, cond_rest)

    # Get new/current velocity depending on decay condition(s) being True/False
    v_out, decay_tracker = jax.lax.cond(
        cond_comb, 
        lambda _: true_func(v_parent, decay_tracker),
        lambda _: false_func(v_in, decay_tracker),
        operand=None
    )
   

    # Snapshot specific parameters.
    idx = jnp.abs(zeds_snaps - z).argmin()
    snap_GRID_L = snaps_GRID_L[idx]

    def inside_cell(_):

        # Load files for current z, to find in which cell neutrino is. 
        # Then load gravity for that cell.
        dPsi_grid = dPsi_grid_data[idx]
        cell_grid = cell_grid_data[idx]
        cell_gens = cell_gens_data[idx]
        
        cell_idx, *_ = SimExec.nu_in_which_cell(
            x_i, cell_grid, cell_gens, snap_GRID_L)
        grad_tot = dPsi_grid[cell_idx, :]

        return grad_tot

    def outside_cell(_):

        # Apply long range force (incl. quadrupole) of whole grid content.
        DM_com = snaps_DM_com[idx]
        DM_num = snaps_DM_num[idx]
        QJ_abs = snaps_QJ_abs[idx]
        grad_tot = SimExec.outside_gravity_quadrupole(
            x_i, DM_com, DM_mass, DM_num, QJ_abs)

        return grad_tot

    grad_tot = jax.lax.cond(
        jnp.all(jnp.abs(x_i) < snap_GRID_L), inside_cell, outside_cell, None)

    # Switch to "physical reality" here.
    grad_tot /= (kpc/s**2)
    v_out /= (kpc/s)
    decay_tracker /= kpc

    # Hamilton eqns. for integration (global minus, s.t. we go back in time).
    dyds = -jnp.array([
        v_out, 1./(1.+z)**2 * grad_tot, jnp.zeros(3)
    ])
 
    return dyds


@jax.jit
def backtrack_1_neutrino(
    init_vector, s_int_steps, decay_angles, parent_momenta, first_negative_indices, decayed_neutrinos_z, z_array, neutrino_momenta, 
    m_parent, m_daughter, z_int_steps, zeds_snaps, 
    snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, 
    dPsi_grid_data, cell_grid_data, cell_gens_data, DM_mass, kpc, s):

    """
    Simulate trajectory of 1 neutrino. Input is 6-dim. vector containing starting positions and velocities of neutrino. Solves ODEs given by the EOMs function with an jax-accelerated integration routine, using the diffrax library. Output are the positions and velocities at each timestep, which was specified with diffrax.SaveAt. 
    """

    # Initial vector in correct shape for EOMs function
    y0_r, Nr = init_vector[0:-1], init_vector[-1]
  
    # Initial vector in correct shape for EOMs function
    decay_tracker = jnp.ones(3)*8.
    y0 = jnp.concatenate((y0_r, decay_tracker)).reshape(3,3)

    # ODE solver setup
    term = diffrax.ODETerm(EOMs)
    t0 = s_int_steps[0]
    t1 = s_int_steps[-1]
    dt0 = (s_int_steps[0] + s_int_steps[1]) / 1000
    

    ### ------------- ###
    ### Dopri5 Solver ###
    ### ------------- ###
    solver = diffrax.Kvaerno3()
    stepsize_controller = diffrax.PIDController(rtol=1e-1, atol=1e-3)
    # note (Fabian): we will need to run tests for solver/rtol/atol/etc.

    # Specify timesteps where solutions should be saved
    saveat = diffrax.SaveAt(ts=jnp.array(s_int_steps))
    
    # Solve the coupled ODEs, i.e. the EOMs of the neutrino
    sol = diffrax.diffeqsolve(
        term, solver, 
        t0=t0, t1=t1, dt0=dt0, y0=y0, max_steps=10000,
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=( 
            Nr.astype(int), s_int_steps,  decay_angles, parent_momenta, first_negative_indices, decayed_neutrinos_z,
            z_array, neutrino_momenta, m_parent, m_daughter, z_int_steps, zeds_snaps, snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, dPsi_grid_data, cell_grid_data, cell_gens_data, DM_mass, kpc, s)
    )
    
    trajectory =  sol.ys.reshape(100, 9)[:, :6]

    return jnp.stack([trajectory[0], trajectory[-1]])


def simulate_neutrinos_1_pix(init_xyz, init_vels, Nr_column,common_args):
    nus = init_vels.shape[0]

    init_vectors_0 = jnp.array(
        [jnp.concatenate((init_xyz, init_vels[k])) for k in range(nus)])

    Nr_column_reshaped = jnp.expand_dims(Nr_column, axis=1)

    # Concatenate the additional column to the original array
    init_vectors = jnp.hstack((init_vectors_0, Nr_column_reshaped))


    trajectories = jnp.array([
        backtrack_1_neutrino(vec, *common_args) for vec in init_vectors
    ])
    
    return trajectories  # shape = (nus, 2, 3)


# Lists for pixel and total number densities
pix_dens_l_used = []
tot_dens_l_used = []
pix_dens_l_rest = []
tot_dens_l_rest = []

for halo_j, halo_ID in enumerate(halo_batch_IDs):

    # note: "Broken" halo, no DM position data at snapshot 0012.
    if halo_j == 19:
        continue

    ### ======================== ###
    ### Load files and grid data ###
    ### ======================== ###

    if pars.benchmark:
        end_str = f'halo{halo_j}'
    else:
        end_str = f'halo{halo_j+1}'
    
    #! Important:
    # The loop ran from the earliest snapshot (z~4 for us) to the latest (z=0).
    # So the below arrays are in this order. Even though our simulation runs 
    # backwards in time, we can leave them like this, since the correct element 
    # gets picked with the idx routine in the EOMs function above.
    snaps_GRID_L = jnp.load(f'{data_dir}/snaps_GRID_L_{end_str}.npy')
    snaps_DM_num = jnp.load(f'{data_dir}/snaps_DM_num_{end_str}.npy')
    snaps_CC_num = jnp.load(f'{data_dir}/snaps_CC_num_{end_str}.npy')
    snaps_progID = jnp.load(f'{data_dir}/snaps_progID_{end_str}.npy')
    snaps_DM_com = jnp.load(f'{data_dir}/snaps_DM_com_{end_str}.npy')
    snaps_QJ_abs = jnp.load(f'{data_dir}/snaps_QJ_abs_{end_str}.npy')

    # Load grav. forces, coordinates and generation/lengths of cells in grid.
    dPsi_grids, cell_grids, cell_gens = SimGrid.grid_data(halo_ID, data_dir)


    ### ===================== ###
    ### Finding starting cell ###
    ### ===================== ###

    # Load grid data and compute radial distances from center of cell centers.
    cell_ccs = cell_grids[-1]
    cell_ccs_kpc = cell_ccs/Params.kpc
    cell_dis = jnp.linalg.norm(cell_ccs_kpc, axis=-1)

    # Get rid of zeros we appended when loading data (see in SimGrid.grid_data)
    cell_dis = cell_dis[cell_dis != 0.]

    # Take first cell, which is in Earth-like position (there can be multiple).
    # Needs to be without kpc units (thus doing /kpc) for simulation start.
    init_xyz = cell_ccs[jnp.abs(cell_dis - init_dis).argsort()][0]/Params.kpc.flatten()
    jnp.save(f'{pars.directory}/init_xyz_{end_str}.npy', init_xyz)
    # note: for some reason this (now jax) routine chooses a different cell to those in the paper, ergo the all-sky maps look different for the same halos

    # Display parameters for simulation.
    print(f"*** Simulation for halo={halo_j+1}/{halo_num} ***")


    ### ============== ###
    ### Run Simulation ###
    ### ============== ###

    sim_start = time.perf_counter()

    pix_sr_sim = simdata.pix_sr
    Npix = int(simdata.Npix)
    nu_per_pix = int(simdata.p_num)
    nu_total = int(simdata.nus_in_sim)

    init_vels = np.load(f'{pars.directory}/initial_velocities.npy')  
    # shape = (Npix, neutrinos per pixel, 3)

    # note (Fabian): let's keep all arrays modular, s.t. there's no crash if we change sim parameters like amount of neutrinos or pixels
    # Nr_column = jnp.arange(768000).reshape(768, 1000)
    Nr_column = jnp.arange(nu_total).reshape(Npix, nu_per_pix)

    # Common arguments for simulation
    common_args = (
        s_int_steps,decay_angles, parent_momenta, first_negative_indices, 
        decayed_neutrinos_z, z_array, neutrino_momenta, m_parent, m_daughter, 
        z_int_steps, zeds_snaps, snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, 
        dPsi_grids, cell_grids, cell_gens, DM_mass, Params.kpc, Params.s)

    # Use ProcessPoolExecutor to distribute the simulations across processes
    # note (Fabian): test a single trajectory by commenting out the multiprocessing routine like I did below, then call the function a single time.
    # with ProcessPoolExecutor(CPUs_sim) as executor:
    #     futures = [
    #         executor.submit(
    #             simulate_neutrinos_1_pix, init_xyz, init_vels[pixel],Nr_column[pixel], common_args) for pixel in range(Npix)
    #     ]
        
    #     # Wait for all futures to complete and collect results in order
    #     nu_vectors = jnp.array([future.result() for future in futures])
    
    # jnp.save(f'{pars.directory}/vectors_{end_str}_{pars.gamma}.npy', nu_vectors)
    
    # note (Fabian): single function call I use to debug scripts fast
    nu_vectors = simulate_neutrinos_1_pix(
        init_xyz, init_vels[0], Nr_column[0], common_args)   

    # note (Fabian): let's stop sim time here
    sim_time = time.perf_counter()-sim_start
    print(f"Simulation time: {sim_time/60.:.2f} min, {sim_time/(60**2):.2f} h")

    ### ------------------------------------------------------- ###
    ### Manipulate array values for number density computations ###
    ### ------------------------------------------------------- ###
    
    # note (Fabian): let's start analysis time here
    ana_start = time.perf_counter()
    
    # note: Those that (anti-)decayed in the sim are daughter neutrinos at z=0,
    # note: rest are all parent neutrinos that survived until today.

    # Load percentage values of total decayed neutrinos for specific gamma
    gammas = np.array(["0.1T", "0.5T", "1T", "2T"])
    g_idx = np.argwhere(gammas == pars.gamma).flatten()[0]
    decay_perc_data = np.load(f"{pars.directory}/decayed_neutrinos_perc.npy")

    # Amount of daughter and parent neutrinos
    daughter_neutrinos = int(decay_perc_data[g_idx] / 100 * simdata.nus_in_sim)
    parent_neutrinos = simdata.nus_in_sim - daughter_neutrinos

    # Indices of daughter and parent neutrinos
    daughter_indices = np.concatenate(
        [arr for arr in  decayed_neutrinos_index_z if arr.size > 0])
    parent_indices = np.setdiff1d(np.arange(simdata.nus_in_sim), daughter_indices) #! wrong before, was daughter_neutrinos

    # Load history of all decayed neutrinos, i.e. total of decays beyond z_sim = 4
    # hist_data = np.load(f"{pars.directory}/histogram_data.npy")[g_idx].sum()

    # Determine more abundant neutrino type: parents or daughter
    # Then use those to compute the number density, and the other is 1 - that value
    nu_vecs_pre = nu_vectors.reshape((nu_total, 2, 6))
    # note (Fabian): I made a mistake here previously, we must use _indices arrays!
    if parent_neutrinos >= daughter_neutrinos:
        # nu_vecs_set = nu_vecs_pre.at[daughter_neutrinos, :, 3:].set(0)
        nu_vecs_set = nu_vecs_pre.at[daughter_indices, :, 3:].set(0)
        nu_vecs_use = nu_vecs_set.reshape((Npix, nu_per_pix, 2, 6))
        used_str = "parents"
        rest_str = "daughters"
    else:
        # nu_vecs_set = nu_vecs_pre.at[parent_neutrinos, :, 3:].set(0)
        nu_vecs_set = nu_vecs_pre.at[parent_indices, :, 3:].set(0)
        nu_vecs_use = nu_vecs_set.reshape((Npix, nu_per_pix, 2, 6))
        used_str = "daughters"
        rest_str = "parents"

### ======================== ###
### Compute number densities ###
### ======================== ###

    # note (Fabian): let's first make total_densities work, then make skymaps, so pixel_densities we look at later
    if pars.pixel_densities:

        # Compute individual number densities for each healpixel
        pix_start = time.perf_counter()

        # Selected neutrino masses
        nu_allsky_masses = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV

        # Pixel densities: daughter neutrinos
        pix_dens_d = Decay.number_densities_all_sky(
            v_arr=nu_vecs_use[..., 3:],
            m_arr=nu_allsky_masses,
            pix_sr=pix_sr_sim,
            args=Params())
        pix_dens_l_d.append(jnp.squeeze(pix_dens_d))

        # Pixel densities: parent neutrinos
        pix_dens_p = Decay.number_densities_all_sky(
            v_arr=nu_vecs_use[..., 3:],
            m_arr=nu_allsky_masses,
            pix_sr=pix_sr_sim,
            args=Params())
        pix_dens_l_p.append(jnp.squeeze(pix_dens_p))
        
        # Pixel densities: no_decay
        # pix_dens_nd = Decay.number_densities_all_sky(
        #     v_arr=nu_vectors[..., 3:],
        #     m_arr=nu_allsky_masses,
        #     pix_sr=pix_sr_sim,
        #     args=Params())
        # pix_dens_l_nd.append(jnp.squeeze(pix_dens_nd))

        pix_time = time.perf_counter() - pix_start
        print(f"Analysis time: {pix_time/60.:.2f} min, {pix_time/(60**2):.2f} h\n")

        # Save pixel densities
        jnp.save(f"{pars.directory}/pixel_densities_{pars.gamma}_d.npy", jnp.array(pix_dens_l_d))
        jnp.save(f"{pars.directory}/pixel_densities_{pars.gamma}_p.npy", jnp.array(pix_dens_l_p))
        jnp.save(f"{pars.directory}/pixel_densities_{pars.gamma}_nd.npy", jnp.array(pix_dens_l_nd))


    if pars.total_densities:

        # Compute total number density, by using all neutrino vectors for integral
        tot_start = time.perf_counter()

        # Total densities for neutrino type (parents or daughter) with larger amount
        tot_dens_used = Decay.number_densities_mass_range_decay(
            v_arr=nu_vecs_use.reshape(-1, 2, 6)[..., 3:], 
            m_arr=nu_massrange, 
            pix_sr=4*Params.Pi,
            args=Params())
        tot_dens_l_used.append(jnp.squeeze(tot_dens_used))

        # Total densities for other neutrino type, not used above
        # For this we load number densities from seperate simulation without decay
        # note (Fabian): here we must change the folder to the gravity sim without decay (not the no_gravity folder), and select the current halo to compare with the correct number densities
        # tot_dens_no_decay = jnp.load(
        #     f"{pars.directory}/../no_gravity/total_densities.npy")
        tot_dens_no_decay = jnp.load(
            f"{pars.directory}/../Dopri5_1k/total_densities.npy")[halo_j]
        tot_dens_rest = tot_dens_no_decay - jnp.array(tot_dens_l_used)
        tot_dens_l_rest.append(jnp.squeeze(tot_dens_rest))

        # note (Fabian): instead of saving here already, we just append (since we're) looping over halos, and then save the arrays outside the loop as one file
        
    ana_time = time.perf_counter() - ana_start
    print(f"Analysis time: {ana_time/60.:.2f} min, {ana_time/(60**2):.2f} h\n")
        
    # Benchmark (NFW-like) halo is considered halo 0, so break off loop here
    if pars.benchmark:
        break


# Save number density arrays for all halos
# note (Fabian): now we save everything (see comment above also)
jnp.save(
    f"{pars.directory}/total_densities_{pars.gamma}_{used_str}.npy", 
    jnp.array(tot_dens_l_used))
jnp.save(
    f"{pars.directory}/total_densities_{pars.gamma}_{rest_str}.npy", 
    jnp.array(tot_dens_l_rest))

tot_time = time.perf_counter() - total_start
print(f"Total time: {tot_time/60.:.2f} min, {tot_time/(60**2):.2f} h")
