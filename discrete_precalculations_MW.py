from shared.preface import *
import shared.functions as fct


def main():
    start = time.perf_counter()

    # ------------------------------- #
    # Generate progenitor index list. #
    # ------------------------------- #

    # Path to merger_tree file.
    tree_path = f'{pathlib.Path.cwd().parent}/neutrino_clustering_output_local/MergerTree/MergerTree_{SIM_ID}.hdf5'

    with h5py.File(tree_path) as tree:
        # Choice of index in snapshot_0036.
        choice = 2  #note: 0 is ~1e12Msun, 1 & 2 are ~1e11Msun
        masses = tree['Assembly_history/Mass'][choice,:]
        zeds = tree['Assembly_history/Redshift']

        # Initial mass of traced halo.
        m0 = f'{masses[0]:.2e}'

        # Progenitor index list.
        prog_idx = tree['Assembly_history/Progenitor_index'][choice,:]
        prog_idx = np.array(np.expand_dims(prog_idx, axis=1), dtype=int)


    # Display script parameters.
    print('*************************************')
    print(f'Simulation box: {SIM_ID}')
    print(f'Mass of selected halo: {m0}')
    print(f'DM particle limit for cells: {DM_LIM}')
    print('*************************************')

    # ---------------------------------- #
    # Precalculations for all snapshots. #
    # ---------------------------------- #

    # Initial grid always the same, hence outside of loop over snapshots.
    grid = fct.grid_3D(GRID_L, GRID_S)
    init_cc = np.expand_dims(grid, axis=1)

    for snap, proj in zip(NUMS_SNAPSHOTS[::-1], prog_idx):
        
        # Generate files with positions of DM particles
        fct.read_DM_positions(
            which_halos='halos', mass_select=12,  # outdated but necessary...
            random=False, snap_num=snap, sim=SIM_ID, 
            halo_index=int(proj), init_m=m0
        )

        # Initial grid and DM positions.
        DM_raw = np.load(
            f'CubeSpace/DM_positions_{SIM_ID}_snapshot_{snap}_{m0}Msun.npy'
        )*kpc
        DM_pos = np.expand_dims(DM_raw, axis=0)
        DM_pos_for_cell_division = np.repeat(DM_pos, len(init_cc), axis=0)

        cell_division_count = fct.cell_division(
            init_cc, DM_pos_for_cell_division, GRID_S, DM_LIM, None,
            sim=SIM_ID, snap_num=snap
            )

        # Arrays produced by cell division algorithm.
        adapted_cc = np.load(
            f'CubeSpace/adapted_cc_{SIM_ID}_snapshot_{snap}.npy')
        cell_gen = np.load(
            f'CubeSpace/cell_gen_{SIM_ID}_snapshot_{snap}.npy')
        cell_com = np.load(
            f'CubeSpace/cell_com_{SIM_ID}_snapshot_{snap}.npy')
        DM_count = np.load(
            f'CubeSpace/DM_count_{SIM_ID}_snapshot_{snap}.npy')

        # Calculate gravity in each cell.
        adapted_DM = np.repeat(DM_pos, len(adapted_cc), axis=0)
        fct.cell_gravity_3D(
            adapted_cc, cell_com, cell_gen,
            adapted_DM, DM_count, DM_SIM_MASS, snap
        )

        print(f'snapshot {snap} : cell division rounds: {cell_division_count}')
        

    seconds = time.perf_counter()-start
    minutes = seconds/60.
    hours = minutes/60.
    print(f'Time sec/min/h: {seconds} sec, {minutes} min, {hours} h.')


if __name__ == "__main__":
    main()