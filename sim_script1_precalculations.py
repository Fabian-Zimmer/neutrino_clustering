from Shared.specific_CNB_sim import *

# Save how much memory is used by OS and not available for script.
OS_MEM = (psutil.virtual_memory().used)

total_start = time.perf_counter()

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', required=True)
parser.add_argument('-st', '--sim_type', required=True)
parser.add_argument('-mg', '--mass_gauge', required=True)
parser.add_argument('-ml', '--mass_lower', required=True)
parser.add_argument('-mu', '--mass_upper', required=True)
parser.add_argument('-hn', '--halo_num', required=True)
pars = parser.parse_args()


### ================================= ###
### Load box & simulation parameters. ###
### ================================= ###

# Box parameters.
with open(f'{pars.directory}/box_parameters.yaml', 'r') as file:
    box_setup = yaml.safe_load(file)

box_file_dir = box_setup['File Paths']['Box File Directory']
DM_mass = box_setup['Content']['DM Mass [Msun]']*Params.Msun
Smooth_L = box_setup['Content']['Smoothening Length [pc]']*Params.pc
z0_snap_4cif = box_setup['Content']['z=0 snapshot']

# Simulation parameters.
with open(f'{pars.directory}/sim_parameters.yaml', 'r') as file:
    sim_setup = yaml.safe_load(file)

CPUs_pre = sim_setup['CPUs_precalculations']
mem_lim_GB = sim_setup['memory_limit_GB']
DM_lim = sim_setup['DM_in_cell_limit']


# Load arrays.
nums_snaps = np.load(f'{pars.directory}/nums_snaps.npy')
shell_multipliers = np.load(f'{pars.directory}/shell_multipliers.npy')


# Make temporary folder to store files, s.t. parallel runs don't clash.
# rand_code = ''.join(
#     random.choices(string.ascii_uppercase + string.digits, k=4)
# )
# data_dir = f'{pars.directory}/temp_data_{rand_code}'
# os.makedirs(data_dir)

# Parent and root directory of current sim folder
sim_output_dir = str(pathlib.Path(pars.directory).parent)
root_dir = str(pathlib.Path(sim_output_dir).parent)

# All precalculations are stored here
if pars.sim_type in ('single_halos', 'all_sky'):
    data_dir = f"{root_dir}/Data/halo_grids"
elif 'benchmark' in pars.sim_type:
    data_dir = f"{root_dir}/Data/benchmark_halo_files"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# hname = f'1e+{pars.mass_gauge}_pm{pars.mass_range}Msun'
hname = f'{pars.mass_lower}-{pars.mass_upper}x1e+{pars.mass_gauge}_Msun'
mass_neg = np.abs(float(pars.mass_gauge) - SimData.M12_to_M12X(float(pars.mass_lower)))
mass_pos = np.abs(float(pars.mass_gauge) - SimData.M12_to_M12X(float(pars.mass_upper)))
SimData.halo_batch_indices(
    z0_snap_4cif, 
    float(pars.mass_gauge), mass_neg, mass_pos,
    'halos', int(pars.halo_num), 
    hname, box_file_dir, pars.directory
)
halo_batch_IDs = np.load(f'{pars.directory}/halo_batch_{hname}_indices.npy')
halo_batch_params = np.load(f'{pars.directory}/halo_batch_{hname}_params.npy')
halo_num = len(halo_batch_params)

print(f'********Numerical Simulation: Mode={pars.sim_type}********')
print('Halo batch params (Rvir,Mvir,cNFW):')
print(halo_batch_params)
print('***********************************')



for halo_j, halo_ID in enumerate(halo_batch_IDs):
    grav_time = time.perf_counter()

    # note: "Broken" halo, no DM position data at snapshot 0012.
    if halo_j == 19:
        continue

    # ============================================== #
    # Run precalculations for current halo in batch. #
    # ============================================== #

    # Generate progenitor index array for current halo.
    with h5py.File(f'{pars.directory}/MergerTree.hdf5') as tree:
        prog_IDs = tree['Assembly_history/Progenitor_index'][halo_ID,:]
        prog_IDs_np = np.array(np.expand_dims(prog_IDs, axis=1), dtype=int)


    # Create empty arrays to save specifics of each loop.
    save_GRID_L = np.zeros(len(nums_snaps))
    save_DM_num = np.zeros(len(nums_snaps))
    save_CC_num = np.zeros(len(nums_snaps))
    save_progID = np.zeros(len(nums_snaps), dtype=int)
    save_DM_com = []
    save_QJ_abs = []


    # Generate the gravity grids from the earliest snapshot to the latest, i.e. 
    # from z=4 to z=0 in our case.
    for j, (snap, prog_ID) in enumerate(
        zip(nums_snaps, prog_IDs_np[::-1])
    ):

        save_progID[j] = int(prog_ID)
        print(f'halo {halo_j+1}/{halo_num} (ID={halo_ID}); snapshot {snap}')


        # --------------------------- #
        # Read and load DM positions. #
        # --------------------------- #

        
        if 'all_sky' in pars.sim_type:
            # Name for file identification of current halo for current snapshot.
            IDname = f'origID{halo_ID}_snap_{snap}'
            
            SimData.read_DM_halo_index(
                snap, int(prog_ID), IDname, box_file_dir, data_dir
            )

            DM_raw = np.load(f'{data_dir}/DM_pos_{IDname}.npy')
            DM_com = np.load(f'{data_dir}/DM_com_coord_{IDname}.npy')*Params.kpc
            DM_particles = len(DM_raw)

        elif 'benchmark' in pars.sim_type:
            # Name for file identification of current halo for current snapshot.
            IDname = f'benchmark_halo_snap_{snap}'

            DM_raw = np.load(f'{data_dir}/{IDname}.npy')
            DM_particles = len(DM_raw)
            DM_com = np.sum(DM_raw, axis=0)/len(DM_raw)*Params.kpc


        # ---------------------- #
        # Cell division process. #
        # ---------------------- #

        # Initialize grid.
        snap_GRID_L = (int(np.abs(DM_raw).max()) + 1)*Params.kpc
        raw_grid = SimGrid.grid_3D(snap_GRID_L, snap_GRID_L)
        init_grid = np.expand_dims(raw_grid, axis=1)

        # Prepare arrays for cell division.
        DM_raw *= Params.kpc
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
        cell_division_count = SimGrid.cell_division(
            init_grid, DM_pos_for_cell_division, snap_GRID_L, DM_lim, None, data_dir, IDname
        )
        del DM_pos_for_cell_division

        # Load files from cell division.
        fin_grid = np.load(f'{data_dir}/fin_grid_{IDname}.npy')
        DM_count = np.load(f'{data_dir}/DM_count_{IDname}.npy')
        cell_com = np.load(f'{data_dir}/cell_com_{IDname}.npy')
        cell_gen = np.load(f'{data_dir}/cell_gen_{IDname}.npy')
        
        # Save snapshot specific parameters.
        save_GRID_L[j] = snap_GRID_L
        save_DM_num[j] = np.sum(DM_count)
        save_CC_num[j] = DM_particles
        save_DM_com.append(DM_com)


        # --------------------------------------------- #
        # Calculate gravity grid (in batches of cells). #
        # --------------------------------------------- #
        cell_coords = np.squeeze(fin_grid, axis=1)
        cells = len(cell_coords)

        print(f'DM Particles = {DM_particles}, cells = {cells}')

        # -------------------- #
        # Short-range gravity. #
        # -------------------- #

        # Calculate available memory per core.
        mem_so_far = (psutil.virtual_memory().used - OS_MEM)/Params.MB_UNIT
        mem_left = mem_lim_GB*1e3 - mem_so_far
        core_mem_MB = mem_left / CPUs_pre

        # Determine short-range chuncksize based on available memory and cells.
        chunksize_sr = SimGrid.chunksize_short_range(
            cells, DM_particles, DM_lim*shell_multipliers[-1], core_mem_MB
        )

        # Split workload into batches (if necessary).
        batch_arr, cell_chunks, cgen_chunks = SimGrid.batch_generators_short_range(
            cell_coords, cell_gen, chunksize_sr
        )

        with ProcessPoolExecutor(CPUs_pre) as ex:
            ex.map(
                SimGrid.cell_gravity_short_range, 
                cell_chunks, cgen_chunks, repeat(snap_GRID_L), 
                repeat(DM_pos), repeat(DM_lim), repeat(DM_mass), 
                repeat(Smooth_L), repeat(data_dir), batch_arr, 
                repeat(chunksize_sr)
            )

        # Combine short-range batch files.
        dPsi_short_range_batches = [
            np.load(f'{data_dir}/batch{b}_short_range.npy') for b in batch_arr
        ]
        dPsi_short_range = np.array(
            list(chain.from_iterable(dPsi_short_range_batches))
        )

        # Combine DM_in_cell_IDs batches (needed for long-range gravity).
        DM_in_cell_IDs_l = []
        for b_id in batch_arr:
            DM_in_cell_IDs_l.append(
                np.load(f'{data_dir}/batch{b_id}_DM_in_cell_IDs.npy')
            )
        DM_in_cell_IDs_np = np.array(
            list(chain.from_iterable(DM_in_cell_IDs_l)))
        np.save(f'{data_dir}/DM_in_cell_IDs_{IDname}.npy', DM_in_cell_IDs_np)
        

        # ------------------- #
        # Long-range gravity. #
        # ------------------- #
        
        # Calculate available memory per core.
        mem_so_far = (psutil.virtual_memory().used - OS_MEM)/Params.MB_UNIT
        mem_left = mem_lim_GB*1e3 - mem_so_far
        core_mem_MB = mem_left / CPUs_pre

        # Determine long-range chuncksize based on available memory and cells.
        # chunksize_lr = chunksize_long_range(cells, core_mem_MB)
        #! adjust/reduce manually to avoid out-of-memory errors.
        #note: 2000 for all_sky or single_halos on thin node works
        chunksize_lr = 2000

        # Split workload into batches (if necessary).
        DM_in_cell_IDs = np.load(f'{data_dir}/DM_in_cell_IDs_{IDname}.npy')
        batch_IDs, cellC_rep, cellC_cc, gen_rep, cib_IDs_gens, count_gens, com_gens, gen_gens = SimGrid.batch_generators_long_range(
            cell_coords, cell_com, cell_gen, DM_count, chunksize_lr
        )

        with ProcessPoolExecutor(CPUs_pre) as ex:
            ex.map(
                SimGrid.cell_gravity_long_range_quadrupole, 
                cellC_rep, cib_IDs_gens, batch_IDs, 
                cellC_cc, com_gens, gen_gens, repeat(snap_GRID_L),
                repeat(np.squeeze(DM_pos, axis=0)), count_gens, 
                repeat(DM_in_cell_IDs), repeat(DM_mass), 
                repeat(data_dir), repeat(chunksize_lr), gen_rep
            )

        # Combine long-range batch files.
        c_labels = np.unique(cellC_rep)
        b_labels = np.unique(batch_IDs)
        with ProcessPoolExecutor(CPUs_pre) as ex:
            ex.map(
                SimData.load_dPsi_long_range, c_labels, 
                repeat(b_labels), repeat(data_dir)
            )

        dPsi_long_range = np.array(
            [np.load(f'{data_dir}/cell{c}_long_range.npy') for c in c_labels])

        # Combine short- and long-range forces.
        dPsi_grid = dPsi_short_range + dPsi_long_range
        np.save(f'{data_dir}/dPsi_grid_{IDname}.npy', dPsi_grid)


        if 'benchmark' in pars.sim_type:
            break


    grav_time_tot = time.perf_counter()-grav_time
    print(f'Grid time: {grav_time_tot/60.} min, {grav_time_tot/(60**2)} h.')


    if 'benchmark' in pars.sim_type:
        end_str = 'benchmark_halo'
    else:
        end_str = f'halo{halo_j+1}'

    np.save(f'{data_dir}/snaps_GRID_L_{end_str}.npy', np.array(save_GRID_L))
    np.save(f'{data_dir}/snaps_DM_num_{end_str}.npy', np.array(save_DM_num))
    np.save(f'{data_dir}/snaps_CC_num_{end_str}.npy', np.array(save_CC_num))
    np.save(f'{data_dir}/snaps_progID_{end_str}.npy', np.array(save_progID))
    np.save(f'{data_dir}/snaps_DM_com_{end_str}.npy', np.array(save_DM_com))
    np.save(f'{data_dir}/snaps_QJ_abs_{end_str}.npy', np.array(save_QJ_abs))


total_time = time.perf_counter() - total_start
print(f'Total time: {total_time/60.} min, {total_time/(60**2)} h.')
