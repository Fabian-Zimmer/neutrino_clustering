# Save how much memory is used by OS and not available for script.
import psutil
MB_UNIT = 1024**2
OS_MEM = (psutil.virtual_memory().used)


from shared.preface import *
import shared.functions as fct

# --------------------------------- #
# Find starting IDs for halo batch. #
# --------------------------------- #

total_start = time.perf_counter()


# Initialize parameters and files.
PRE = PRE(
    sim='L025N752', 
    z0_snap=36, z4_snap=13, DM_lim=10000,
    sim_dir=SIM_ROOT, sim_ver=SIM_TYPE,
    phis=10, thetas=10, vels=100,
    pre_CPUs=128, sim_CPUs=128, mem_lim_GB=224
)

# Make temporary folder to store files, s.t. parallel runs don't clash.
rand_code = ''.join(
    random.choices(string.ascii_uppercase + string.digits, k=4)
)
TEMP_DIR = f'{PRE.OUT_DIR}/temp_data_{rand_code}'
os.makedirs(TEMP_DIR)


def EOMs(s_val, y):

    # Initialize vector.
    x_i, u_i = np.reshape(y, (2,3))

    # Switch to "numerical reality" here.
    x_i *= kpc
    u_i *= (kpc/s)

    # Find z corresponding to s via interpolation.
    z = np.interp(s_val, S_STEPS, ZEDS)

    # Snapshot specific parameters.
    idx = np.abs(PRE.ZEDS_SNAPS - z).argmin()
    snap = PRE.NUMS_SNAPS[idx]
    snap_GRID_L = snaps_GRID_L[idx]

    # Neutrino inside cell grid.
    if np.all(np.abs(x_i)) <= snap_GRID_L:

        # Find which (pre-calculated) derivative grid to use at current z.
        simname = f'benchmark_halo_snap_{snap}'
        dPsi_grid = fct.load_grid(TEMP_DIR, 'derivatives', simname)
        cell_grid = fct.load_grid(TEMP_DIR, 'positions',   simname)

        cell_idx = fct.nu_in_which_cell(x_i, cell_grid)  # index of cell
        grad_tot = dPsi_grid[cell_idx,:]                 # derivative of cell

    # Neutrino outside cell grid.
    else:
        NrDM = snaps_DM_num[idx]
        grad_tot = fct.outside_gravity(x_i, NrDM, PRE.DM_SIM_MASS)

        # With quadrupole.
        # DM_com = snaps_DM_com[idx]
        # DM_num = snaps_DM_num[idx]
        # QJ_abs = snaps_QJ_abs[idx]
        # grad_tot = fct.outside_gravity_quadrupole(
        #     x_i, DM_com, PRE.DM_SIM_MASS, DM_num, QJ_abs
        # )

    # Switch to "physical reality" here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)

    # Hamilton eqns. for integration.
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
        fun=EOMs, t_span=[S_STEPS[0], S_STEPS[-1]], t_eval=S_STEPS,
        y0=y0, method=SOLVER, vectorized=True,
        args=()
        )
    
    np.save(f'{TEMP_DIR}/nu_{int(Nr)}.npy', np.array(sol.y.T))


# ======================================= #
# Run precalculations for benchmark halo. #
# ======================================= #

# Create empty arrays to save specifics of each loop.
snaps_GRID_L = np.zeros(len(PRE.NUMS_SNAPS))
snaps_DM_num = np.zeros(len(PRE.NUMS_SNAPS))

for j, snap in enumerate(PRE.NUMS_SNAPS[::-1]):
    
    print(f'benchmark halo ; snapshot {snap}')

    # ------------------ #
    # Load DM positions. #
    # ------------------ #
    IDname = f'benchmark_halo_snap_{snap}'
    DM_raw = np.load(f'{PRE.OUT_DIR}/{IDname}.npy')
    DM_particles = len(DM_raw)


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
    del DM_raw

    # Cell division.
    cell_division_count = fct.cell_division(
        init_grid, DM_pos_for_cell_division, snap_GRID_L, PRE.DM_LIM, None, TEMP_DIR, IDname
    )
    del DM_pos_for_cell_division

    # Load files from cell division.
    fin_grid = np.load(f'{TEMP_DIR}/fin_grid_{IDname}.npy')
    DM_count = np.load(f'{TEMP_DIR}/DM_count_{IDname}.npy')
    cell_com = np.load(f'{TEMP_DIR}/cell_com_{IDname}.npy')
    cell_gen = np.load(f'{TEMP_DIR}/cell_gen_{IDname}.npy')
    
    # Save snapshot specific parameters.
    snaps_GRID_L[j] = snap_GRID_L
    snaps_DM_num[j] = np.sum(DM_count)


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
    mem_left = PRE.MEM_LIM_GB*1e3 - mem_so_far
    core_mem_MB = mem_left / PRE.PRE_CPUs

    # Determine short-range chuncksize based on available memory and cells.
    chunksize_sr = fct.chunksize_short_range(
        cells, DM_particles, PRE.DM_LIM*SHELL_MULTIPLIERS[-1], core_mem_MB
    )

    # Split workload into batches (if necessary).
    batch_arr, cell_chunks, cgen_chunks = fct.batch_generators_short_range(
        cell_coords, cell_gen, chunksize_sr
    )

    with ProcessPoolExecutor(PRE.PRE_CPUs) as ex:
        ex.map(
            fct.cell_gravity_short_range, 
            cell_chunks, cgen_chunks, repeat(snap_GRID_L), 
            repeat(DM_pos), repeat(PRE.DM_LIM), repeat(PRE.DM_SIM_MASS), 
            repeat(PRE.SMOOTH_L), repeat(TEMP_DIR), batch_arr, 
            repeat(chunksize_sr)
        )

    # Combine short-range batch files.
    dPsi_short_range_batches = [
        np.load(f'{TEMP_DIR}/batch{b}_short_range.npy') for b in batch_arr
    ]
    dPsi_short_range = np.array(
        list(chain.from_iterable(dPsi_short_range_batches))
    )

    # Combine DM_in_cell_IDs batches (needed for long-range gravity).
    DM_in_cell_IDs_l = []
    for b_id in batch_arr:
        DM_in_cell_IDs_l.append(
            np.load(f'{TEMP_DIR}/batch{b_id}_DM_in_cell_IDs.npy')
        )
    DM_in_cell_IDs_np = np.array(
        list(chain.from_iterable(DM_in_cell_IDs_l)))
    np.save(f'{TEMP_DIR}/DM_in_cell_IDs_{IDname}.npy', DM_in_cell_IDs_np)
    

    # ------------------- #
    # Long-range gravity. #
    # ------------------- #
    
    # Calculate available memory per core.
    mem_so_far = (psutil.virtual_memory().used - OS_MEM)/MB_UNIT
    mem_left = PRE.MEM_LIM_GB*1e3 - mem_so_far
    core_mem_MB = mem_left / PRE.PRE_CPUs

    # Determine long-range chuncksize based on available memory and cells.
    # chunksize_lr = chunksize_long_range(cells, core_mem_MB)
    chunksize_lr = 501

    # Split workload into batches (if necessary).
    DM_in_cell_IDs = np.load(f'{TEMP_DIR}/DM_in_cell_IDs_{IDname}.npy')
    batch_IDs, cellC_rep, cellC_cc, gen_rep, cib_IDs_gens, count_gens, com_gens, gen_gens = fct.batch_generators_long_range(
        cell_coords, cell_com, cell_gen, DM_count, chunksize_lr
    )

    with ProcessPoolExecutor(PRE.PRE_CPUs) as ex:
        ex.map(
            fct.cell_gravity_long_range_quadrupole, 
            cellC_rep, cib_IDs_gens, batch_IDs, 
            cellC_cc, com_gens, gen_gens, repeat(snap_GRID_L),
            repeat(np.squeeze(DM_pos, axis=0)), count_gens, 
            repeat(DM_in_cell_IDs), repeat(PRE.DM_SIM_MASS), 
            repeat(TEMP_DIR), repeat(chunksize_lr), gen_rep
        )

    # Combine long-range batch files.
    c_labels = np.unique(cellC_rep)
    b_labels = np.unique(batch_IDs)
    with ProcessPoolExecutor(PRE.PRE_CPUs) as ex:
        ex.map(
            fct.load_dPsi_long_range, c_labels, 
            repeat(b_labels), repeat(TEMP_DIR)
        )

    dPsi_long_range = np.array(
        [np.load(f'{TEMP_DIR}/cell{c}_long_range.npy') for c in c_labels])

    # Combine short- and long-range forces.
    dPsi_grid = dPsi_short_range + dPsi_long_range
    np.save(f'{TEMP_DIR}/dPsi_grid_{IDname}.npy', dPsi_grid)


# ========================================= #
# Run simulation for current halo in batch. #
# ========================================= #

# Display parameters for simulation.
print('***Running simulation***')
print(f'benchmark halo, CPUs={PRE.SIM_CPUs}')

start = time.perf_counter()

# Draw initial velocities.
ui = fct.init_velocities(PRE.PHIs, PRE.THETAs, PRE.MOMENTA)

# Combine vectors and append neutrino particle number.
y0_Nr = np.array(
    [np.concatenate((X_SUN, ui[i], [i+1])) for i in range(PRE.NUS)]
    )


# Run simulation on multiple cores.
with ProcessPoolExecutor(PRE.SIM_CPUs) as ex:
    ex.map(backtrack_1_neutrino, y0_Nr)


# Compactify all neutrino vectors into 1 file.
Ns = np.arange(PRE.NUS, dtype=int)            
nus = [np.load(f'{TEMP_DIR}/nu_{Nr+1}.npy') for Nr in Ns]
Bname = f'{PRE.NUS}nus_benchmark_halo'
np.save(f'{TEMP_DIR}/{Bname}.npy', np.array(nus))

# Calculate local overdensity.
vels = fct.load_sim_data(TEMP_DIR, Bname, 'velocities')

# note: The final number density is not stored in the temporary folder.
out_file = f'{PRE.OUT_DIR}/number_densities_band_{Bname}.npy'
fct.number_densities_mass_range(
    vels, NU_MRANGE, out_file
)

# Now delete velocities and distances.
fct.delete_temp_data(f'{TEMP_DIR}/{Bname}.npy')

seconds = time.perf_counter()-start
minutes = seconds/60.
hours = minutes/60.
print(f'Sim time min/h: {minutes} min, {hours} h.')


# Remove temporary folder with all individual neutrino files.
shutil.rmtree(TEMP_DIR)

total_time = time.perf_counter()-total_start
print(f'Total time: {total_time/60.} min, {total_time/(60**2)} h.')