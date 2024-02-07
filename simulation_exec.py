from Shared.specific_CNB_sim import *


total_start = time.perf_counter()

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', required=True)
parser.add_argument('-st', '--sim_type', required=True)
parser.add_argument('-mg', '--mass_gauge', required=True)
parser.add_argument('-ml', '--mass_lower', required=True)
parser.add_argument('-mu', '--mass_upper', required=True)
parser.add_argument('-hn', '--halo_num', required=True)
parser.add_argument(
    '--upto_Rvir', required=True, action=argparse.BooleanOptionalAction
)
pars = parser.parse_args()

# Create halo batch, files and other simulation setup parameters and arrays
DM_mass, CPUs_sim, neutrinos, init_dis, zeds_snaps, z_int_steps, s_int_steps, nu_massrange, data_dir, halo_batch_IDs, halo_num = SimData.simulation_setup(
    sim_dir=pars.directory,
    m_lower=pars.mass_lower,
    m_upper=pars.mass_upper,
    m_gauge=pars.mass_gauge,
    halo_num_req=pars.halo_num,
    sim_type=pars.sim_type)


@jax.jit
def EOMs(s_val, y, args):

    """
    Solves the Equations of Motion (EOMs) for a particle within a cosmological simulation space. This involves determining the particle's position and velocity in relation to the simulation grid and applying the appropriate gravitational forces based on its location. It supports dynamic adaptation to different epochs by interpolating across snapshots of the universe at various redshifts, and employs conditional logic to distinguish between gravitational influences inside and outside the simulation's voxel cells. The function outputs the rate of change in position and velocity for the particle, used for the backwards in time integration method we use.
    """

    # Unpack the simulation grid data
    s_int_steps, z_int_steps, zeds_snaps, snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, dPsi_grid_data, cell_grid_data, cell_gens_data, DM_mass, kpc, s = args

    # Initialize vector.
    x_i, u_i = y

    # Switch to "numerical reality" here.
    x_i *= kpc
    u_i *= (kpc/s)

    # Find z corresponding to s via interpolation.
    z = Utils.jax_interpolate(s_val, s_int_steps, z_int_steps)

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
    x_i /= kpc
    u_i /= (kpc/s)

    # Hamilton eqns. for integration (global minus, s.t. we go back in time).
    dyds = -jnp.array([
        u_i, 1./(1.+z)**2 * grad_tot
    ])

    return dyds


@jax.jit
def backtrack_1_neutrino(
    init_vector, s_int_steps, z_int_steps, zeds_snaps, 
    snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, 
    dPsi_grid_data, cell_grid_data, cell_gens_data, DM_mass, kpc, s):

    """
    Simulate trajectory of 1 neutrino. Input is 6-dim. vector containing starting positions and velocities of neutrino. Solves ODEs given by the EOMs function with an jax-accelerated integration routine, using the diffrax library. Output are the positions and velocities at each timestep, which was specified with diffrax.SaveAt. 
    """

    # Initial vector in correct shape for EOMs function
    y0 = init_vector.reshape(2,3)

    # ODE solver setup
    term = diffrax.ODETerm(EOMs)
    solver = diffrax.Dopri5()
    t0 = s_int_steps[0]
    t1 = s_int_steps[-1]
    dt0 = (s_int_steps[0] + s_int_steps[1]) / 2
    # stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-1) # before
    stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)

    # Specify timesteps where solutions should be saved
    saveat = diffrax.SaveAt(ts=jnp.array(s_int_steps))
    
    # Solve the coupled ODEs, i.e. the EOMs of the neutrino
    sol = diffrax.diffeqsolve(
        term, solver, 
        t0=t0, t1=t1, dt0=dt0, y0=y0, #max_steps=10000,
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=( 
            s_int_steps, z_int_steps, zeds_snaps, snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, dPsi_grid_data, cell_grid_data, cell_gens_data, DM_mass, kpc, s)
    )
    
    trajectory = sol.ys.reshape(100,6)

    return jnp.stack([trajectory[0], trajectory[-1]])


for halo_j, halo_ID in enumerate(halo_batch_IDs):

    # note: "Broken" halo, no DM position data at snapshot 0012.
    if halo_j == 19:
        continue

    # ========================================= #
    # Run simulation for current halo in batch. #
    # ========================================= #

    if 'benchmark' in pars.sim_type:
        end_str = 'benchmark_halo'
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


    ### --------------------- ###
    ### Finding starting cell ###
    ### --------------------- ###

    # def find_starting_cell():

    # Load grid data and compute radial distances from center of cell centers.
    cell_ccs = cell_grids[-1]
    cell_ccs_kpc = cell_ccs/Params.kpc
    cell_dis = jnp.linalg.norm(cell_ccs_kpc, axis=-1)

    # Get rid of zeros, since we appended some in a previous routine
    cell_dis = cell_dis[cell_dis != 0.]

    # Take first cell, which is in Earth-like position (there can be multiple).
    # Needs to be without kpc units (thus doing /kpc) for simulation start.
    #? is this broken because I filled cell_grid_snaps with zeros?
    #! yes, something is not right now, different initial starting cells...
    # init_xyz = cell_ccs[jnp.abs(cell_dis - init_dis).argsort()][0]/Params.kpc.flatten()
    init_xyz = jnp.load(f'{pars.directory}/init_xyz_{end_str}.npy')

    # jnp.save(f'{pars.directory}/init_xyz_{end_str}.npy', init_xyz)

    # Display parameters for simulation.
    print(f'***Running simulation: mode = {pars.sim_type}***')
    print(f'halo={halo_j+1}/{halo_num}, CPUs={CPUs_sim}')

    sim_start = time.perf_counter()

    #! to overhaul...
    if pars.sim_type in ('single_halos', 'benchmark'):

        # Load initial velocities.
        ui = np.load(f'{pars.directory}/initial_velocities.npy')

        # Combine vectors and append neutrino particle number.
        y0_Nr = np.array(
            [np.concatenate((init_xyz, ui[i], [i+1])) for i in range(neutrinos)]
            )

        # Run simulation on multiple cores.
        with ProcessPoolExecutor(CPUs_sim) as ex:
            ex.map(backtrack_1_neutrino, y0_Nr)

        # Compactify all neutrino vectors into 1 file.
        neutrino_vectors = np.array(
            [np.load(f'{data_dir}/nu_{i+1}.npy') for i in range(neutrinos)]
        )

        # For these modes (i.e. not all_sky), save all neutrino vectors.
        # Split velocities and positions into 10k neutrino batches.
        # For reference: ndarray with shape (10_000, 100, 6) is  48 MB.
        batches = math.ceil(neutrinos/10_000)
        split = np.array_split(neutrino_vectors, batches, axis=0)
        vname = f'neutrino_vectors_numerical_{end_str}'
        for i, elem in enumerate(split):
            np.save(
                f'{pars.directory}/{vname}_batch{i+1}.npy', elem
            )

        # Compute the number densities.
        dname = f'number_densities_numerical_{end_str}'
        out_file = f'{pars.directory}/{dname}.npy'
        Physics.number_densities_mass_range(
            neutrino_vectors[...,3:6], nu_massrange, out_file
        )

    else:

        with open(f'{pars.directory}/sim_parameters.yaml', 'r') as file:
            sim_setup = yaml.safe_load(file)

        # Size (in sr) of all-sky healpixels
        pix_sr_sim = sim_setup['pix_sr']

        # Number of healpixels 
        Npix = sim_setup["Npix"]

        init_vels = np.load(f'{pars.directory}/initial_velocities.npy')  
        # shape = (Npix, neutrinos_per_pix, 3)


        def simulate_neutrinos_1_pix(init_xyz, init_vels, common_args):

            # Neutrinos per pixel
            nus = init_vels.shape[0]

            init_vectors = jnp.array(
                [jnp.concatenate((init_xyz, init_vels[k])) for k in range(nus)]
            )

            trajectories = jnp.array([
                backtrack_1_neutrino(vec, *common_args) for vec in init_vectors
            ])
            
            return trajectories  # shape = (nus, 2, 3)
        

        # Common arguments for simulation
        common_args = (
            s_int_steps, z_int_steps, zeds_snaps, snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, dPsi_grids, cell_grids, cell_gens, DM_mass, Params.kpc, Params.s)

        # Use ProcessPoolExecutor to distribute the simulations across processes
        with ProcessPoolExecutor(CPUs_sim) as executor:
            futures = [
                executor.submit(
                    simulate_neutrinos_1_pix, init_xyz, init_vels[pixel], common_args) for pixel in range(Npix)
            ]
            
            # Wait for all futures to complete and collect results in order
            results = [future.result() for future in futures]
            
        # Combine results into final array
        nu_vectors = jnp.vstack(results)

        # Save all sky neutrino vectors for current halo
        # note: Max. GitHub possible is nside=8, shape=(1_000*768,2,6), ~75 MB
        vname = f'neutrino_vectors_numerical_{end_str}_all_sky'
        jnp.save(f'{pars.directory}/{vname}.npy', nu_vectors)
        

        ################################
        ### Compute number densities ###
        ################################

        dname = f'number_densities_numerical_{end_str}_all_sky'
        out_path = f'{pars.directory}/{dname}.npy'

        nu_density = Physics.number_densities_all_sky(
            v_arr=nu_vectors[..., 3:6],
            m_arr=nu_massrange,
            Npix=Npix,
            pix_sr=pix_sr_sim,
            out_path=out_path,
            args=Params())


    sim_time = time.perf_counter()-sim_start
    print(f'Sim time: {sim_time/60.} min, {sim_time/(60**2)} h.')
    
    if 'benchmark' in pars.sim_type:
        break

    # '''

# Remove nu_* files, s.t. when testing it will show me if not produced.
# delete_temp_data(f'{temp_dir}/nu_*.npy')

# if pars.sim_type == 'all_sky':
    # Delete arrays not compatible with github file limit size.
    # delete_temp_data(f'{pars.directory}/initial_velocities.npy')

# Remove temporary folder.
# shutil.rmtree(temp_dir)

total_time = time.perf_counter()-total_start
print(f'Total time: {total_time/60.} min, {total_time/(60**2)} h.')
