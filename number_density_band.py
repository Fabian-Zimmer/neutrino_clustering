from shared.preface import *
import shared.functions as fct

# --------------------------------- #
# Find starting IDs for halo batch. #
# --------------------------------- #

# Halo batch parameters.
sim = 'L012N376'
snap = '0036'
mass_gauge = 12.4  # in log10 Msun
mass_range = 0.3

hname = f'1e+{mass_gauge}_pm{mass_range}Msun'
fct.halo_batch_indices(sim, snap, mass_gauge, mass_range, 'halos', 10, hname)
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

    # ID corresponding to current z.
    idx = np.abs(ZEDS_SNAPSHOTS - z).argmin()
    snap = NUMS_SNAPSHOTS[idx]
    NrDM = NrDM_SNAPSHOTS[idx]

    # Neutrino inside cell grid.
    if np.all(np.abs(x_i)) <= GRID_L:

        # Find which (pre-calculated) derivative grid to use at current z.
        simname = f'origID{halo_ID}_snap_{snap}'
        dPsi_grid = fct.load_grid(sim, 'derivatives', simname)
        cell_grid = fct.load_grid(sim, 'positions',   simname)

        cell_idx = fct.nu_in_which_cell(x_i, cell_grid)  # index of cell
        grad_tot = dPsi_grid[cell_idx,:]                 # derivative of cell

    # Neutrino outside cell grid.
    else:
        grad_tot = fct.outside_gravity(x_i, NrDM)

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

        # ============================================== #
        # Run precalculations for current halo in batch. #
        # ============================================== #

        # Generate progenitor index array for current halo.
        proj_IDs = fct.read_MergerTree(sim, halo_ID)

        NrDM_snaps = np.zeros(len(NUMS_SNAPSHOTS))
        for j, (snap, proj_ID) in enumerate(zip(NUMS_SNAPSHOTS[::-1], proj_IDs)):
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
            GRID_L = (int(np.abs(DM_raw).max()) + 1)*kpc
            raw_grid = fct.grid_3D(GRID_L, GRID_L)  # to get 8 all-covering cells
            init_grid = np.expand_dims(raw_grid, axis=1)

            # Prepare arrays for cell division.
            DM_raw *= kpc
            DM_pos = np.expand_dims(DM_raw, axis=0)
            DM_pos_for_cell_division = np.repeat(DM_pos, len(init_grid), axis=0)

            # Cell division.
            DM_lim_band = 5000
            cell_division_count = fct.cell_division(
                init_grid, DM_pos_for_cell_division, GRID_L, DM_lim_band, None, 
                sim, IDname
            )

            # Load files from cell division.
            fin_grid = np.load(f'{sim}/fin_grid_{IDname}.npy')
            DM_count = np.load(f'{sim}/DM_count_{IDname}.npy')
            cell_com = np.load(f'{sim}/cell_com_{IDname}.npy')
            cell_gen = np.load(f'{sim}/cell_gen_{IDname}.npy')
            NrDM_snaps[j] = np.sum(DM_count)

            # Optional printout.
            # print(fin_grid.shape, DM_count.shape, cell_com.shape, cell_gen.shape)
            # print(DM_count.sum() - len(DM_raw))
            # note: 1 DM particle is "missing" in grid
            # print(f'snapshot {snap} : cell division rounds: {cell_division_count}')


            # --------------------------------------------- #
            # Calculate gravity grid (in batches of cells). #
            # --------------------------------------------- #

            batch_size = 30
            bs_cc = chunks(batch_size, fin_grid)
            bs_count = chunks(batch_size, DM_count)
            bs_com = chunks(batch_size, cell_com)
            bs_gen = chunks(batch_size, cell_gen)

            b_nums = []
            for b, (b_cc,  b_gen,  b_com,  b_count) in enumerate(
                zip(bs_cc, bs_gen, bs_com, bs_count)
            ):
                b_nums.append(b)
                b_cc = np.array(b_cc)
                b_gen = np.array(b_gen)
                b_com = np.array(b_com)
                b_count = np.array(b_count)

                # Calculate gravity in each cell in current batch.
                b_DM = np.repeat(DM_pos, len(b_cc), axis=0)
                bname = f'batch{b}'
                fct.cell_gravity(
                    b_cc, b_com, b_gen, GRID_L,
                    b_DM, b_count, DM_lim_band,
                    sim, bname
                )
            bs_nums = np.array(b_nums)

            # Combine and then delete batch files.
            dPsi_batches = [
                np.load(f'{sim}/dPsi_grid_batch{b}.npy') for b in bs_nums
            ]
            dPsi_fin = np.array(list(chain.from_iterable(dPsi_batches)))
            np.save(f'{sim}/dPsi_grid_{IDname}.npy', dPsi_fin)

            # Delete intermediate data.
            fct.delete_temp_data(f'{sim}/dPsi_grid_batch*.npy') 
        fct.delete_temp_data(f'{sim}/DM_pos_*.npy')


        # ========================================= #
        # Run simulation for current halo in batch. #
        # ========================================= #

        # These arrays will be used EOMs function above.
        NrDM_SNAPSHOTS = NrDM_snaps

        start = time.perf_counter()

        # Draw initial velocities.
        ui = fct.draw_ui(phi_points = PHIs, theta_points = THETAs)

        # Combine vectors and append neutrino particle number.
        y0_Nr = np.array(
            [np.concatenate((X_SUN, ui[i], [i+1])) for i in range(NUS)]
            )


        # Display parameters for simulation.
        CPUs = 12
        print('***Running simulation***')
        print(
            f'neutrinos={NUS}, halo={halo_j+1}/{halo_num}, CPUs={CPUs}, solver={SOLVER}'
        )

        # Test 1 neutrino only.
        # backtrack_1_neutrino(y0_Nr[0])
        # backtrack_1_neutrino(y0_Nr[1])
        # backtrack_1_neutrino(y0_Nr[2])

        # '''
        # Run simulation on multiple cores.
        with ProcessPoolExecutor(CPUs) as ex:
            ex.map(backtrack_1_neutrino, y0_Nr)  


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
        # '''

        seconds = time.perf_counter()-start
        minutes = seconds/60.
        hours = minutes/60.
        print(f'Sim time min/h: {minutes} min, {hours} h.')

    except ValueError:  # bad halo?
        traceback.print_exc()
        continue