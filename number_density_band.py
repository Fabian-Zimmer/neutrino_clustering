from shared.preface import *
import shared.functions as fct


# Simulation and halo batch parameters.
sim = 'L006N188'
mass_gauge = 11  # in log10 Msun
mass_range = 1

NrDM_snaps = np.zeros(len(NUMS_SNAPSHOTS))
'''
for snap in NUMS_SNAPSHOTS[::-1]:

    # Read DM positions and halo parameters of halo batch for current snapshot.
    fct.read_DM_halo_batch(sim, snap, mass_gauge, mass_range, 'halos')
    hname = f'snap_{snap}_1e+{mass_gauge}Msun'
    halo_params = np.load(f'{sim}/halo_params_{hname}.npy')
    halo_num = len(halo_params)

    for j in range(halo_num):

        # Load DM positions of halo.
        DM_raw = np.load(f'{sim}/DM_pos_{hname}_halo{j}.npy')
        
        # ---------------------- #
        # Cell division process. #
        # ---------------------- #

        # Initialize grid.
        GRID_L = (int(np.abs(DM_raw).max()) + 1)*kpc
        raw_grid = fct.grid_3D(GRID_L, GRID_L)
        init_grid = np.expand_dims(raw_grid, axis=1)

        # Prepare arrays for cell division.
        DM_raw *= kpc
        DM_pos = np.expand_dims(DM_raw, axis=0)
        DM_pos_for_cell_division = np.repeat(DM_pos, len(init_grid), axis=0)

        # Cell division.
        DM_lim = 5000
        cell_division_count = fct.cell_division(
            init_grid, DM_pos_for_cell_division, GRID_L, DM_lim, None, 
            sim, snap, halo_num=j
        )

        # Load files from cell division.
        fname = f'snap_{snap}_halo{j}'
        fin_grid = np.load(f'{sim}/fin_grid_{fname}.npy')
        DM_count = np.load(f'{sim}/DM_count_{fname}.npy')
        cell_com = np.load(f'{sim}/cell_com_{fname}.npy')
        cell_gen = np.load(f'{sim}/cell_gen_{fname}.npy')
        NrDM_snaps[j] = np.sum(DM_count)
        # print(fin_grid.shape, DM_count.shape, cell_com.shape, cell_gen.shape)
        
        # print(DM_count.sum() - len(DM_raw))
        # note: 1 DM particle is "missing" in grid

        print(f'snapshot {snap} : cell division rounds: {cell_division_count}')


        # --------------------------------------------- #
        # Calculate gravity grid (in batches of cells). #
        # --------------------------------------------- #

        batch_size = 10
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
            fct.cell_gravity(
                b_cc, b_com, b_gen,
                b_DM, b_count, sim, snap, batch_num=b
            )
        bs_nums = np.array(b_nums)

        # Combine and then delete batch files.
        dPsi_batches = [
            np.load(f'{sim}/dPsi_grid_snap_{snap}_batch{b}.npy') for b in bs_nums
        ]
        dPsi_fin = np.array(list(chain.from_iterable(dPsi_batches)))
        np.save(f'{sim}/dPsi_grid_{fname}.npy', dPsi_fin)

        # Delete intermediate data.
        fct.delete_temp_data(f'{sim}/dPsi_*batch*.npy') 
    fct.delete_temp_data(f'{sim}/DM_pos_*halo*.npy')
'''

# -------------------------------------- #
# Run simulation for each halo in batch. #
# -------------------------------------- #

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
        dPsi_grid = fct.load_grid(snap, sim, 'derivatives', halo_j)
        cell_grid = fct.load_grid(snap, sim, 'positions',   halo_j)

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
    
    np.save(f'temp_data/nu_{int(Nr)}.npy', np.array(sol.y.T))


hname = f'snap_0036_1e+{mass_gauge}Msun'
halo_params = np.load(f'{sim}/halo_params_{hname}.npy')
halo_num = len(halo_params)
for halo_j in range(halo_num):

    start = time.perf_counter()

    # Draw initial velocities.
    ui = fct.draw_ui(phi_points = PHIs, theta_points = THETAs)

    # Combine vectors and append neutrino particle number.
    y0_Nr = np.array(
        [np.concatenate((X_SUN, ui[i], [i+1])) for i in range(NUS)]
        )

    # Number of DM particles used for each snapshot.
    NrDM_SNAPSHOTS = NrDM_snaps[::-1]

    CPUs = 6

    # Display parameters for simulation.
    print('***Running simulation***')
    print(
        f'neutrinos={NUS}, halo={halo_j+1}/{halo_num}, CPUs={CPUs}, solver={SOLVER}'
    )

    # Test 1 neutrino only.
    # backtrack_1_neutrino(y0_Nr[0])

    # '''
    # Run simulation on multiple cores.
    with ProcessPoolExecutor(CPUs) as ex:
        ex.map(backtrack_1_neutrino, y0_Nr)  


    # Compactify all neutrino vectors into 1 file.
    Ns = np.arange(NUS, dtype=int)  # Nr. of neutrinos
    
    nus = np.array([np.load(f'temp_data/nu_{Nr+1}.npy') for Nr in Ns])
    np.save(f'{sim}/{NUS}nus_{hname}.npy', nus)  
    fct.delete_temp_data('temp_data/nu_*.npy')    
    # '''

    seconds = time.perf_counter()-start
    minutes = seconds/60.
    hours = minutes/60.
    print(f'Time sec/min/h: {seconds} sec, {minutes} min, {hours} h.')