# Save how much memory is used by OS and not available for script.
import psutil
MB_UNIT = 1024**2
OS_MEM = (psutil.virtual_memory().used)
total_start = time.perf_counter()

from shared.preface import *

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', required=True)
parser.add_argument('-st', '--sim_type', required=True)
parser.add_argument('-hn', '--halo_num', required=True)
parser.add_argument('-mg', '--mass_gauge', required=True)
parser.add_argument('-mr', '--mass_range', required=True)
args = parser.parse_args()


### ================================= ###
### Load box & simulation parameters. ###
### ================================= ###

# Box parameters.
with open(f'{args.directory}/box_parameters.yaml', 'r') as file:
    box_setup = yaml.safe_load(file)

box_file_dir = box_setup['File Paths/Box File Directory']
DM_mass = box_setup['Content/DM Mass [Msun]']*Msun
Smooth_L = box_setup['Content/Smoothening Length [pc]']*pc
z0_snap_4cif = box_setup['Content/z=0 snapshot']

# Simulation parameters.
with open(f'{args.directory}/sim_parameters.yaml', 'r') as file:
    sim_setup = yaml.safe_load(file)

CPUs_pre = sim_setup['CPUs_precalculations']
CPUs_sim = sim_setup['CPUs_simulations']
mem_lim_GB = sim_setup['memory_limit_GB']
DM_lim = sim_setup['DM_in_cell_limit']
integration_solver = sim_setup['integration_solver']
init_x_dis = sim_setup['initial_haloGC_distance']
init_xyz = np.array([init_x_dis, 0., 0.])
neutrinos = sim_setup['neutrinos']

# Load arrays.
nums_snaps = np.save(f'{args.directory}/nums_snaps.npy')
zeds_snaps = np.save(f'{args.directory}/zeds_snaps.npy')

z_int_steps = np.save(f'{args.directory}/z_int_steps.npy')
s_int_steps = np.save(f'{args.directory}/s_int_steps.npy')
neutrino_massrange = np.save(f'{args.directory}/neutrino_massrange_eV.npy')
DM_shell_edges = np.save(f'{args.directory}/DM_shell_edges.npy')
shell_multipliers = np.save(f'{args.directory}/shell_multipliers.npy')


# Load constants and arrays, which the functions.py script needs.
FCT_h = box_setup['Cosmology']['h']
FCT_H0 = FCT_h*100*km/s/Mpc
FCT_Omega_M = box_setup['Cosmology']['Omega_M']
FCT_Omega_L = box_setup['Cosmology']['Omega_L']
FCT_DM_shell_edges = np.copy(DM_shell_edges)
FCT_shell_multipliers = np.copy(shell_multipliers)
FCT_init_xys = np.copy(init_xyz)
FCT_neutrino_simulation_mass_eV = sim_setup['neutrino_simulation_mass_eV']*eV
FCT_zeds = np.copy(z_int_steps)

# note: now that variables are loaded into memory, the function.py will work.
#? probably not a good final solution, perhaps scripts have functions above, 
#? which they will use? they should be unique between the analytical and 
#? numerical simulation types.
import shared.functions as fct






# Make temporary folder to store files, s.t. parallel runs don't clash.
rand_code = ''.join(
    random.choices(string.ascii_uppercase + string.digits, k=4)
)
temp_dir = f'{args.directory}/temp_data_{rand_code}'
os.makedirs(temp_dir)



# note: Halo parameters used so far for MW-type halos.
# args.mass_gauge = 12.0
# args.mass_range = 0.6
# args.halo_num = 1

hname = f'1e+{args.mass_gauge}_pm{args.mass_range}Msun'
fct.halo_batch_indices(
    z0_snap_4cif, args.mass_gauge, args.mass_range, 'halos', args.halo_num, 
    hname, box_file_dir, args.directory
)
halo_batch_IDs = np.load(f'{args.directory}/halo_batch_{hname}_indices.npy')
halo_batch_params = np.load(f'{args.directory}/halo_batch_{hname}_params.npy')
halo_num = len(halo_batch_params)

print(f'********Numerical Simulation: Mode={args.sim_type}********')
print('Halo batch params (Rvir,Mvir,cNFW):')
print(halo_batch_params)
print('***********************************')


def EOMs(s_val, y):

    # Initialize vector.
    x_i, u_i = np.reshape(y, (2,3))

    # Switch to "numerical reality" here.
    x_i *= kpc
    u_i *= (kpc/s)

    # Find z corresponding to s via interpolation.
    z = np.interp(s_val, s_int_steps, z_int_steps)

    # Snapshot specific parameters.
    idx = np.abs(zeds_snaps - z).argmin()
    snap = nums_snaps[idx]
    snap_GRID_L = snaps_GRID_L[idx]

    # Neutrino inside cell grid.
    if np.all(np.abs(x_i)) <= snap_GRID_L:

        # Find which (pre-calculated) derivative grid to use at current z.
        simname = f'origID{halo_ID}_snap_{snap}'
        dPsi_grid = fct.load_grid(temp_dir, 'derivatives', simname)
        cell_grid = fct.load_grid(temp_dir, 'positions',   simname)

        cell_idx = fct.nu_in_which_cell(x_i, cell_grid)  # index of cell
        grad_tot = dPsi_grid[cell_idx,:]                 # derivative of cell

    # Neutrino outside cell grid.
    else:
        # NrDM = snaps_DM_num[idx]
        # grad_tot = fct.outside_gravity(x_i, NrDM, DM_mass)

        # With quadrupole.
        DM_com = snaps_DM_com[idx]
        DM_num = snaps_DM_num[idx]
        QJ_abs = snaps_QJ_abs[idx]
        grad_tot = fct.outside_gravity_quadrupole(
            x_i, DM_com, DM_mass, DM_num, QJ_abs
        )

    # Switch to "physical reality" here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)

    # Hamilton eqns. for integration (global minus, s.t. we go back in time).
    dyds = -np.array([
        u_i, 1./(1.+z)**2 * grad_tot
    ])

    return dyds


def backtrack_1_neutrino(y0_Nr):
    """Simulate trajectory of 1 neutrino."""

    # Split input into initial vector and neutrino number.
    y0, Nr = y0_Nr[0:-1], y0_Nr[-1]

    # Solve all 6 EOMs.
    sol = solve_ivp(
        fun=EOMs, t_span=[s_int_steps[0], s_int_steps[-1]], t_eval=s_int_steps,
        y0=y0, method=integration_solver, vectorized=True,
        args=()
        )
    
    np.save(f'{temp_dir}/nu_{int(Nr)}.npy', np.array(sol.y.T))


for halo_j, halo_ID in enumerate(halo_batch_IDs):

    # ============================================== #
    # Run precalculations for current halo in batch. #
    # ============================================== #

    # Generate progenitor index array for current halo.
    with h5py.File(f'{args.directory}/MergerTree.hdf5') as tree:
        prog_IDs = tree['Assembly_history/Progenitor_index'][halo_ID,:]
        prog_IDs_np = np.array(np.expand_dims(prog_IDs, axis=1), dtype=int)


    # Create empty arrays to save specifics of each loop.
    save_GRID_L = np.zeros(len(nums_snaps))
    save_num_DM = np.zeros(len(nums_snaps))
    save_DM_com = []
    save_QJ_abs = []

    #! before
    # for j, (snap, prog_ID) in enumerate(
    #     zip(nums_snaps[::-1], prog_IDs_np)
    # ):
        
    #! after
    for j, (snap, prog_ID) in enumerate(
        zip(nums_snaps, prog_IDs_np)
    ):
        print(f'halo {halo_j+1}/{halo_num} ; snapshot {snap}')
        prog_ID = int(prog_ID)


        # --------------------------- #
        # Read and load DM positions. #
        # --------------------------- #

        # Name for file identification of current halo for current snapshot.
        IDname = f'origID{halo_ID}_snap_{snap}'

        if args.sim_type in ('single_halos', 'all_sky'):
            
            fct.read_DM_halo_index(
                snap, prog_ID, IDname, box_file_dir, temp_dir
            )
            DM_raw = np.load(f'{temp_dir}/DM_pos_{IDname}.npy')
            DM_com = np.load(f'{temp_dir}/DM_com_coord_{IDname}.npy')*kpc
            DM_particles = len(DM_raw)

        else:

            # Define how many shells are used, out of len(DM_SHELL_EDGES)-1.
            shells = 1
            DM_shell_edges = DM_shell_edges[:shells+1]
            
            # Load DM from all used shells.
            DM_pre = []
            for shell_i in range(shells):
                DM_pre.append(
                    np.load(f'{temp_dir}/DM_pos_{IDname}_shell{shell_i}.npy')
                )
            DM_raw = np.array(list(chain.from_iterable(DM_pre)))
            DM_particles = len(DM_raw)
            DM_com = np.sum(DM_raw, axis=0)/len(DM_raw)
            del DM_pre


        # ---------------------- #
        # Cell division process. #
        # ---------------------- #

        # Initialize grid.
        snap_GRID_L = (int(np.abs(DM_raw).max()) + 1)*kpc
        raw_grid = fct.grid_3D(snap_GRID_L, snap_GRID_L)
        init_grid = np.expand_dims(raw_grid, axis=1)

        # Prepare arrays for cell division.
        DM_raw *= kpc
        DM_pos = np.expand_dims(DM_raw, axis=0)
        DM_pos_for_cell_division = np.repeat(DM_pos, len(init_grid), axis=0)

        
        ### Interlude: Calculate QJ_aa and QJ_ab for complete halo. ###
        
        # Center all DM particles of halo on c.o.m. of halo and get distances.
        DM_raw -= DM_com
        DM_raw_dis = np.expand_dims(np.sqrt(np.sum(DM_raw**2, axis=1)), axis=1)

        # Permute order of coords by one, i.e. (x,y,z) -> (z,x,y).
        DM_raw_roll = np.roll(DM_raw, 1)

        # Terms appearing in the quadrupole term.
        QJ_aa = np.sum(3*DM_raw**2 - DM_raw_dis**2, axis=0)
        QJ_ab = np.sum(3*DM_raw*DM_raw_roll, axis=0)
        del DM_raw
        save_QJ_abs.append(np.array([QJ_aa, QJ_ab]))


        # Cell division.
        cell_division_count = fct.cell_division(
            init_grid, DM_pos_for_cell_division, snap_GRID_L, DM_lim, None, temp_dir, IDname
        )
        del DM_pos_for_cell_division

        # Load files from cell division.
        fin_grid = np.load(f'{temp_dir}/fin_grid_{IDname}.npy')
        DM_count = np.load(f'{temp_dir}/DM_count_{IDname}.npy')
        cell_com = np.load(f'{temp_dir}/cell_com_{IDname}.npy')
        cell_gen = np.load(f'{temp_dir}/cell_gen_{IDname}.npy')
        
        # Save snapshot specific parameters.
        save_GRID_L[j] = snap_GRID_L
        save_num_DM[j] = np.sum(DM_count)
        save_DM_com.append(DM_com)


        # --------------------------------------------- #
        # Calculate gravity grid (in batches of cells). #
        # --------------------------------------------- #
        cell_coords = np.squeeze(fin_grid, axis=1)
        cells = len(cell_coords)


        # -------------------- #
        # Short-range gravity. #
        # -------------------- #

        # Calculate available memory per core.
        mem_so_far = (psutil.virtual_memory().used - OS_MEM)/MB_UNIT
        mem_left = mem_lim_GB*1e3 - mem_so_far
        core_mem_MB = mem_left / CPUs_pre

        # Determine short-range chuncksize based on available memory and cells.
        chunksize_sr = fct.chunksize_short_range(
            cells, DM_particles, DM_lim*shell_multipliers[-1], core_mem_MB
        )

        # Split workload into batches (if necessary).
        batch_arr, cell_chunks, cgen_chunks = fct.batch_generators_short_range(
            cell_coords, cell_gen, chunksize_sr
        )

        with ProcessPoolExecutor(CPUs_pre) as ex:
            ex.map(
                fct.cell_gravity_short_range, 
                cell_chunks, cgen_chunks, repeat(snap_GRID_L), 
                repeat(DM_pos), repeat(DM_lim), repeat(DM_mass), 
                repeat(Smooth_L), repeat(temp_dir), batch_arr, 
                repeat(chunksize_sr)
            )

        # Combine short-range batch files.
        dPsi_short_range_batches = [
            np.load(f'{temp_dir}/batch{b}_short_range.npy') for b in batch_arr
        ]
        dPsi_short_range = np.array(
            list(chain.from_iterable(dPsi_short_range_batches))
        )

        # Combine DM_in_cell_IDs batches (needed for long-range gravity).
        DM_in_cell_IDs_l = []
        for b_id in batch_arr:
            DM_in_cell_IDs_l.append(
                np.load(f'{temp_dir}/batch{b_id}_DM_in_cell_IDs.npy')
            )
        DM_in_cell_IDs_np = np.array(
            list(chain.from_iterable(DM_in_cell_IDs_l)))
        np.save(f'{temp_dir}/DM_in_cell_IDs_{IDname}.npy', DM_in_cell_IDs_np)
        

        # ------------------- #
        # Long-range gravity. #
        # ------------------- #
        
        # Calculate available memory per core.
        mem_so_far = (psutil.virtual_memory().used - OS_MEM)/MB_UNIT
        mem_left = mem_lim_GB*1e3 - mem_so_far
        core_mem_MB = mem_left / CPUs_pre

        # Determine long-range chuncksize based on available memory and cells.
        # chunksize_lr = chunksize_long_range(cells, core_mem_MB)
        chunksize_lr = 501

        # Split workload into batches (if necessary).
        DM_in_cell_IDs = np.load(f'{temp_dir}/DM_in_cell_IDs_{IDname}.npy')
        batch_IDs, cellC_rep, cellC_cc, gen_rep, cib_IDs_gens, count_gens, com_gens, gen_gens = fct.batch_generators_long_range(
            cell_coords, cell_com, cell_gen, DM_count, chunksize_lr
        )

        with ProcessPoolExecutor(CPUs_pre) as ex:
            ex.map(
                fct.cell_gravity_long_range_quadrupole, 
                cellC_rep, cib_IDs_gens, batch_IDs, 
                cellC_cc, com_gens, gen_gens, repeat(snap_GRID_L),
                repeat(np.squeeze(DM_pos, axis=0)), count_gens, 
                repeat(DM_in_cell_IDs), repeat(DM_mass), 
                repeat(temp_dir), repeat(chunksize_lr), gen_rep
            )

        # Combine long-range batch files.
        c_labels = np.unique(cellC_rep)
        b_labels = np.unique(batch_IDs)
        with ProcessPoolExecutor(CPUs_pre) as ex:
            ex.map(
                fct.load_dPsi_long_range, c_labels, 
                repeat(b_labels), repeat(temp_dir)
            )

        dPsi_long_range = np.array(
            [np.load(f'{temp_dir}/cell{c}_long_range.npy') for c in c_labels])

        # Combine short- and long-range forces.
        dPsi_grid = dPsi_short_range + dPsi_long_range
        np.save(f'{temp_dir}/dPsi_grid_{IDname}.npy', dPsi_grid)


    # ========================================= #
    # Run simulation for current halo in batch. #
    # ========================================= #

    #! Important:
    # The loop ran from the earliest snapshot (z~4 for us) to the latest (z=0).
    # So these arrays are in this order. Even though our simulation runs 
    # backwards in time, we can leave them like this, since the correct element 
    # gets picked with the idx routine in the EOMs function above.
    snaps_GRID_L = np.array(save_GRID_L)
    snaps_DM_num = np.array(save_num_DM)
    snaps_DM_com = np.array(save_DM_com)
    snaps_QJ_abs = np.array(save_QJ_abs)

    # Display parameters for simulation.
    print(f'***Running simulation: mode = {args.sim_type}***')
    print(f'halo={halo_j+1}/{halo_num}, CPUs={CPUs_sim}')

    sim_start = time.perf_counter()

    if args.sim_type in ('single_halos', 'spheres'):
    
        # Load initial velocities.
        ui = np.load(f'{args.directory}/initial_velocities.npy')

        # Combine vectors and append neutrino particle number.
        y0_Nr = np.array(
            [np.concatenate((init_xyz, ui[i], [i+1])) for i in range(neutrinos)]
            )

        # Run simulation on multiple cores.
        with ProcessPoolExecutor(CPUs_sim) as ex:
            ex.map(backtrack_1_neutrino, y0_Nr)

        # Compactify all neutrino vectors into 1 file.
        neutrino_vectors = np.array(
            [np.load(f'{temp_dir}/nu_{i+1}.npy') for i in range(neutrinos)]
        )

        # For these modes (i.e. not all_sky), save all neutrino vectors.
        # Split velocities and positions into 10k neutrino batches.
        # For reference: ndarray with shape (10_000, 100, 6) is  48 MB.
        split = np.array_split(neutrino_vectors, 10_000, axis=0)
        for i, elem in enumerate(split):
            np.save(
                f'{args.directory}/neutrino_vectors_batch{i+1}.npy', elem
            )

        # Compute the number densities.
        out_file = f'{args.directory}/number_densities.npy'
        fct.number_densities_mass_range(
            neutrino_vectors[:,:,3:6], neutrino_massrange, out_file
        )

    else:

        # Load initial velocities for all_sky mode. Note that this array is not 
        # github compatible, and will be deleted afterwards.
        ui = np.load(f'{args.directory}/initial_velocities.npy')

        # Empty list to append number densitites of each angle coord. pair.
        number_densities_pairs = []

        for i, ui_elem in enumerate(ui):

            print(f'Coord. pair {i+1}/{len(ui)}')

            # Combine vectors and append neutrino particle number.
            y0_Nr = np.array([np.concatenate(
                (init_xyz, ui_elem[k], [k+1])) for k in range(len(ui_elem))
            ])
            
            # Run simulation on multiple cores.
            with ProcessPoolExecutor(CPUs_sim) as ex:
                ex.map(backtrack_1_neutrino, y0_Nr)

            # Compactify all neutrino vectors into 1 file.
            neutrino_vectors = np.array(
                [np.load(f'{temp_dir}/nu_{i+1}.npy') for i in range(neutrinos)]
            )

            # Compute the number densities.
            number_densities_pairs.append(
                fct.number_densities_mass_range(
                    neutrino_vectors[:,:,3:6], 
                    neutrino_massrange, 
                    sim_type=args.sim_type
                )
            )


        # Combine number densities with angle pairs: First 2 entries are angles.
        nu_dens_pairs = np.array(number_densities_pairs)
        angle_pairs = np.load(f'{args.directory}/all_sky_angles.npy')
        nu_final = np.concatenate((angle_pairs, nu_dens_pairs), axis=2)
        np.save(f'{args.directory}/number_densities.npy', nu_final)

        # Delete arrays not compatible with github file limit size.
        fct.delete_temp_data(f'{args.directory}/initial_velocities.npy')


    sim_time = time.perf_counter()-sim_start
    print(f'Sim time: {sim_time/60.} min, {sim_time/(60**2)} h.')

# Remove temporary folder with all individual neutrino files.
shutil.rmtree(temp_dir)

total_time = time.perf_counter()-total_start
print(f'Total time: {total_time/60.} min, {total_time/(60**2)} h.')