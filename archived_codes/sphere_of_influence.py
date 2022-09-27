from shared.preface import *
import shared.functions as fct

### Script for running precalculations and simulation for different radii. ###

def main_radius_pre(DM_radius):
    start = time.perf_counter()

    # Generate progenitor index list.
    m0, prog_idx = fct.read_MergerTree(init_halo=HALO_INDEX)

    # Display script parameters.
    print('*************************************')
    print(f'Simulation box: {SIM_ID}')
    print(f'Mass of selected halo: {m0}')
    print(f'DM particle limit for cells: {DM_LIM}')
    print(f'DM inclusion radius: {DM_radius} kpc')
    print('*************************************')

    # ---------------------------------- #
    # Precalculations for all snapshots. #
    # ---------------------------------- #

    # Initial grid always the same, hence outside of loop over snapshots.
    grid = fct.grid_3D(GRID_L, GRID_S)
    init_cc = np.expand_dims(grid, axis=1)

    # Save number of DM particles used for each snapshot.
    NrDM_snaps = np.zeros(len(NUMS_SNAPSHOTS))

    for j, (snap, proj) in enumerate(zip(NUMS_SNAPSHOTS[::-1], prog_idx)):

        # Generate files with positions of DM particles
        fct.read_DM_positions(
            snap_num=snap, sim=SIM_ID, 
            halo_index=int(proj), init_m=m0, DM_radius_kpc=DM_radius
        )

        # Initial grid and DM positions.
        DM_raw = np.load(
            f'CubeSpace/DM_positions_{SIM_ID}_snapshot_{snap}_{m0}Msun_{DM_radius}kpc.npy'
        )*kpc
        DM_pos = np.expand_dims(DM_raw, axis=0)
        DM_pos_for_cell_division = np.repeat(DM_pos, len(init_cc), axis=0)

        cell_division_count = fct.cell_division(
            init_cc, DM_pos_for_cell_division, GRID_S, DM_LIM, None,
            sim=SIM_ID, snap_num=snap, m0=m0, DM_incl_radius=DM_radius
            )

        # Arrays produced by cell division algorithm.
        fin_grid = np.load(
            f'CubeSpace/fin_grid_{SIM_ID}_snapshot_{snap}_{m0}Msun_{DM_radius}kpc.npy')
        cell_gen = np.load(
            f'CubeSpace/cell_gen_{SIM_ID}_snapshot_{snap}_{m0}Msun_{DM_radius}kpc.npy')
        cell_com = np.load(
            f'CubeSpace/cell_com_{SIM_ID}_snapshot_{snap}_{m0}Msun_{DM_radius}kpc.npy')
        DM_count = np.load(
            f'CubeSpace/DM_count_{SIM_ID}_snapshot_{snap}_{m0}Msun_{DM_radius}kpc.npy')
        NrDM_snaps[j] = np.sum(DM_count)
        # print(fin_grid.shape, cell_gen.shape, cell_com.shape, DM_count.shape)

        # Generate gravity grid, in batches of cells, due to memory intensity.
        batch_size = 10  
        #! snapshot 14 memory crash?
        
        bs_cc = chunks(batch_size, fin_grid)
        bs_gen = chunks(batch_size, cell_gen)
        bs_com = chunks(batch_size, cell_com)
        bs_count = chunks(batch_size, DM_count)

        b_nums = []
        for b,  (b_cc,  b_gen,  b_com,  b_count) in enumerate(
            zip(bs_cc, bs_gen, bs_com, bs_count)
        ):
            b_nums.append(b)
            b_cc = np.array(b_cc)
            b_gen = np.array(b_gen)
            b_com = np.array(b_com)
            b_count = np.array(b_count)

            # Calculate gravity in each cell in current batch.
            b_DM = np.repeat(DM_pos, len(b_cc), axis=0)
            fct.cell_gravity_3D(
                b_cc, b_com, b_gen,
                b_DM, b_count, DM_SIM_MASS, snap, m0,
                batches=True, batch_num=b
            )
        bs_nums = np.array(b_nums)

        # Combine and then delete batch files.
        dPsi_batches = [
            np.load(
                f'CubeSpace/dPsi_grid_snapshot_{snap}_batch{b}.npy'
            ) for b in bs_nums
        ]
        dPsi_combined = np.array(
            list(chain.from_iterable(dPsi_batches))
        )
        np.save(
            f'CubeSpace/dPsi_grid_snapshot_{snap}_{m0}Msun_{DM_radius}kpc.npy', dPsi_combined
        )

        # Delete temp. files.
        fct.delete_temp_data('CubeSpace/DM_positions_*.npy')
        fct.delete_temp_data('CubeSpace/dPsi_*batch*.npy') 

        print(f'snapshot {snap} : cell division rounds: {cell_division_count}')

    np.save(f'CubeSpace/NrDM_snapshots_{m0}Msun_{DM_radius}kpc.npy', NrDM_snaps[::-1])

    seconds = time.perf_counter()-start
    minutes = seconds/60.
    hours = minutes/60.
    print(f'Time sec/min/h: {seconds} sec, {minutes} min, {hours} h.')


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
        dPsi_grid = fct.load_grid(
            snap, SIM_ID, HALO_MASS, DM_radius, 'derivatives'
        )
        cell_grid = fct.load_grid(
            snap, SIM_ID, HALO_MASS, DM_radius, 'positions'
        )

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
        y0=y0, method=SOLVER, vectorized=True
        )
    
    np.save(f'neutrino_vectors/nu_{int(Nr)}_CubeSpace.npy', np.array(sol.y.T))


def main_radius_sim(DM_radius):
    start = time.perf_counter()

    # Draw initial velocities.
    ui = fct.draw_ui(
        phi_points   = PHIs,
        theta_points = THETAs
        )
    
    # Combine vectors and append neutrino particle number.
    y0_Nr = np.array(
        [np.concatenate((X_SUN, ui[i], [i+1])) for i in range(NUS)]
        )

    CPUs = 6

    # Display parameters for simulation.
    print(
        '***Running simulation*** \n',
        f'neutrinos={NUS} ; DM_radius={DM_radius} ; CPUs={CPUs} ; solver={SOLVER}'
    )

    # Test 1 neutrino only.
    # backtrack_1_neutrino(y0_Nr[0])

    # '''
    # Run simulation on multiple cores.
    with ProcessPoolExecutor(CPUs) as ex:
        ex.map(backtrack_1_neutrino, y0_Nr)  


    # Compactify all neutrino vectors into 1 file.
    Ns = np.arange(NUS, dtype=int)  # Nr. of neutrinos
    nus = np.array(
        [np.load(f'neutrino_vectors/nu_{Nr+1}_CubeSpace.npy') for Nr in Ns]
    )
    np.save(
        f'neutrino_vectors/nus_{NUS}_CubeSpace_{m0}Msun_{DM_radius}kpc.npy',
        nus
        )  
    fct.delete_temp_data('neutrino_vectors/nu_*CubeSpace.npy')    
    # '''

    seconds = time.perf_counter()-start
    minutes = seconds/60.
    hours = minutes/60.
    print(f'Time sec/min/h: {seconds} sec, {minutes} min, {hours} h.')


if __name__ == '__main__':
    
    DM_radii = np.linspace(260, 800, 5)

    # Integration steps.
    S_STEPS = np.array([fct.s_of_z(z) for z in ZEDS])

    m0 = HALO_MASS

    # Precalculations.
    # for DM_radius in DM_radii:
        # main_radius_pre(DM_radius)


    # Simulations.
    # for DM_radius in DM_radii:

    #     # Number of DM particles used for each snapshot.
    #     NrDM_SNAPSHOTS = np.load(
    #         f'CubeSpace/NrDM_snapshots_{m0}Msun_{DM_radius}kpc.npy'
    #     )

    #     main_radius_sim(DM_radius)


    DM_radius = DM_radii[-1]

    main_radius_pre(DM_radius)

    # Number of DM particles used for each snapshot.
    NrDM_SNAPSHOTS = np.load(
        f'CubeSpace/NrDM_snapshots_{m0}Msun_{DM_radius}kpc.npy'
    )

    main_radius_sim(DM_radius)