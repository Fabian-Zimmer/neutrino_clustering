from shared.preface import *
import shared.functions as fct

# --------------------------------- #
# Find starting IDs for halo batch. #
# --------------------------------- #

# Halo batch parameters.
sim = 'L012N376'
snap = '0036'  # "starting" snapshot to search for halos
mass_gauge = 12.0  # in log10 Msun
mass_range = 0.8
size = 10
DM_lim_batch = 5000
snellius_CPUs = 4

hname = f'1e+{mass_gauge}_pm{mass_range}Msun'
fct.halo_batch_indices(sim, snap, mass_gauge, mass_range, 'halos', size, hname)
halo_batch_IDs = np.load(f'{sim}/halo_batch_{hname}_indices.npy')
halo_batch_params = np.load(f'{sim}/halo_batch_{hname}_params.npy')
halo_num = len(halo_batch_params)

print('********Number density band********')
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
    z = np.interp(s_val, S_STEPS, ZEDS)

    # Snapshot specific parameters.
    idx = np.abs(ZEDS_SNAPSHOTS - z).argmin()
    snap = NUMS_SNAPSHOTS[idx]
    snap_GRID_L = snaps_GRID_L[idx]

    # Neutrino inside cell grid.
    if np.all(np.abs(x_i)) <= snap_GRID_L:

        # Find which (pre-calculated) derivative grid to use at current z.
        simname = f'origID{halo_ID}_snap_{snap}'
        dPsi_grid = fct.load_grid(sim, 'derivatives', simname)
        cell_grid = fct.load_grid(sim, 'positions',   simname)

        cell_idx = fct.nu_in_which_cell(x_i, cell_grid)  # index of cell
        grad_tot = dPsi_grid[cell_idx,:]                 # derivative of cell

    # Neutrino outside cell grid.
    else:
        NrDM = NrDM_SNAPSHOTS[idx]
        com_DM = DM_COM_SNAPSHOTS[idx]
        grad_tot = fct.outside_gravity(x_i, com_DM, NrDM)

    # Switch to "physical reality" here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)

    # Hamilton eqns. for integration.
    dyds = TIME_FLOW * np.array([
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
    
    np.save(f'{sim}/nu_{int(Nr)}.npy', np.array(sol.y.T))


for halo_j, halo_ID in enumerate(halo_batch_IDs):

    try:
        # '''
        # ============================================== #
        # Run precalculations for current halo in batch. #
        # ============================================== #

        # Generate progenitor index array for current halo.
        proj_IDs = fct.read_MergerTree(sim, halo_ID)

        save_GRID_L = np.zeros(len(NUMS_SNAPSHOTS))
        save_num_DM = np.zeros(len(NUMS_SNAPSHOTS))
        save_DM_com = []
        for j, (snap, proj_ID) in enumerate(zip(
            NUMS_SNAPSHOTS[::-1], proj_IDs
        )):
            print(f'halo {halo_j+1}/{halo_num} ; snapshot {snap}')
            
            proj_ID = int(proj_ID)

            # --------------------------- #
            # Read and load DM positions. #
            # --------------------------- #

            IDname = f'origID{halo_ID}_snap_{snap}'
            fct.read_DM_halo_index(sim, snap, proj_ID, IDname)
            DM_raw = np.load(f'{sim}/DM_pos_{IDname}.npy')
            

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

            # Cell division.
            cell_division_count = fct.cell_division(
                init_grid, DM_pos_for_cell_division, snap_GRID_L, DM_lim_batch, None, 
                sim, IDname
            )

            # Load files from cell division.
            fin_grid = np.load(f'{sim}/fin_grid_{IDname}.npy')
            DM_count = np.load(f'{sim}/DM_count_{IDname}.npy')
            cell_com = np.load(f'{sim}/cell_com_{IDname}.npy')
            cell_gen = np.load(f'{sim}/cell_gen_{IDname}.npy')
            
            # Save snapshot specific parameters.
            save_GRID_L[j] = snap_GRID_L
            save_num_DM[j] = np.sum(DM_count)
            save_DM_com.append(np.load(f'{sim}/DM_com_coord_{IDname}.npy'))

            # Optional printout.
            # print(fin_grid.shape, DM_count.shape, cell_com.shape, cell_gen.shape)
            # print(DM_count.sum() - len(DM_raw))
            # note: 1 DM particle is "missing" in grid
            # print(f'snapshot {snap} : cell division rounds: {cell_division_count}')


            # --------------------------------------------- #
            # Calculate gravity grid (in batches of cells). #
            # --------------------------------------------- #

            def batch_gravity(
                grid_chunk, DMnr_chunk, com_chunk, gen_chunk, num_chunk
                ):

                b = int(num_chunk)
                b_cc = np.array(grid_chunk)
                b_com = np.array(com_chunk)
                b_gen = np.array(gen_chunk)
                b_count = np.array(DMnr_chunk)

                # Calculate gravity in each cell in current batch.
                b_DM = np.repeat(DM_pos, len(b_cc), axis=0)
                bname = f'batch{b}'
                fct.cell_gravity(
                    b_cc, b_com, b_gen, snap_GRID_L,
                    b_DM, b_count, DM_lim_batch,
                    sim, bname
                )

            chunk_size = 30
            grid_chunks = chunks(chunk_size, fin_grid)
            DMnr_chunks = chunks(chunk_size, DM_count)
            com_chunks = chunks(chunk_size, cell_com)
            gen_chunks = chunks(chunk_size, cell_gen)
            num_chunks = math.ceil(len(DM_count)/chunk_size)
            idx_chunks = np.arange(num_chunks)

            with ProcessPoolExecutor(snellius_CPUs) as ex:
                ex.map(
                    batch_gravity, grid_chunks, DMnr_chunks, 
                    com_chunks, gen_chunks, idx_chunks
                )

            # Combine and then delete batch files.
            dPsi_batches = [
                np.load(f'{sim}/dPsi_grid_batch{b}.npy') for b in idx_chunks
            ]
            dPsi_fin = np.array(list(chain.from_iterable(dPsi_batches)))
            np.save(f'{sim}/dPsi_grid_{IDname}.npy', dPsi_fin)
            fct.delete_temp_data(f'{sim}/dPsi_*batch*.npy')

        # Save snapshot and halo specific arrays.
        np.save(f'{sim}/snaps_GRID_L_origID{halo_ID}.npy', save_GRID_L)
        np.save(f'{sim}/NrDM_snaps_origID{halo_ID}.npy', save_num_DM)
        np.save(f'{sim}/DM_com_origID{halo_ID}.npy', np.array(save_DM_com))

        # Clean up.
        fct.delete_temp_data(f'{sim}/DM_pos_*halo*.npy')
        # '''

        # ========================================= #
        # Run simulation for current halo in batch. #
        # ========================================= #

        # These arrays will be used EOMs function above.
        snaps_GRID_L = np.load(f'{sim}/snaps_GRID_L_origID{halo_ID}.npy')
        NrDM_SNAPSHOTS = np.load(f'{sim}/NrDM_snaps_origID{halo_ID}.npy')
        DM_COM_SNAPSHOTS = np.load(f'{sim}/DM_com_origID{halo_ID}.npy')

        start = time.perf_counter()

        # Draw initial velocities.
        ui = fct.draw_ui(phi_points = PHIs, theta_points = THETAs)

        # Combine vectors and append neutrino particle number.
        y0_Nr = np.array(
            [np.concatenate((X_SUN, ui[i], [i+1])) for i in range(NUS)]
            )


        # Display parameters for simulation.
        CPUs = snellius_CPUs
        print('***Running simulation***')
        print(
            f'neutrinos={NUS}, halo={halo_j+1}/{halo_num}, CPUs={CPUs}, solver={SOLVER}'
        )

        sim_testing = False

        if sim_testing:
            # Test 1 neutrino only.
            backtrack_1_neutrino(y0_Nr[0])
            backtrack_1_neutrino(y0_Nr[1])

        else:
            # Run simulation on multiple cores.
            with ProcessPoolExecutor(CPUs) as ex:
                ex.map(backtrack_1_neutrino, y0_Nr)  
                #todo: maybe ex.map(backtrack_1_neutrino, y0_Nr, chunksize=???) 
                #todo: decreases time, where ??? could be e.g. 100 or 1000...  


            # Compactify all neutrino vectors into 1 file.
            Ns = np.arange(NUS, dtype=int)  # Nr. of neutrinos
            
            nus = np.array([np.load(f'{sim}/nu_{Nr+1}.npy') for Nr in Ns])
            np.save(
                f'{sim}/{NUS}nus_1e+{mass_gauge}_pm{mass_range}Msun_halo{halo_j}.npy', 
                nus
            )  

            # Delete all temporary files.
            fct.delete_temp_data(f'{sim}/nu_*.npy')
            fct.delete_temp_data(f'{sim}/fin_grid_*.npy')
            fct.delete_temp_data(f'{sim}/DM_count_*.npy')
            fct.delete_temp_data(f'{sim}/cell_com_*.npy')
            fct.delete_temp_data(f'{sim}/cell_gen_*.npy')

            seconds = time.perf_counter()-start
            minutes = seconds/60.
            hours = minutes/60.
            print(f'Sim time min/h: {minutes} min, {hours} h.')

    except ValueError:  # bad halo?
        traceback.print_exc()
        continue