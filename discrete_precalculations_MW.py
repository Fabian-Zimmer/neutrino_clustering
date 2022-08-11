from shared.preface import *
import shared.functions as fct


start = time.perf_counter()


# ------------------------------- #
# Generate progenitor index list. #
# ------------------------------- #

# Path to merger_tree file.
#note: for now this file is generated in Ch.5 cell of CubeSpace.ipynb ...
tree_path = '/home/fabian/ownCloud/snellius/MergerTree/Tree_data_Centrals_MergerTree_test_93_97.hdf5'

with h5py.File(tree_path) as tree:
    # Choice of index in snapshot_0036.
    choice = 2  #note: 0 or 1 overloads memory, too 
    masses = tree['Assembly_history/Mass'][choice,:]
    zeds = tree['Assembly_history/Redshift']

    # Initial mass of traced halo.
    m0 = f'{masses[0]:.2e}'

    # Progenitor index list.
    prog_idx = tree['Assembly_history/Progenitor_index'][choice,:]
    prog_idx = np.array(np.expand_dims(prog_idx, axis=1), dtype=int)


# ---------------------------------- #
# Precalculations for all snapshots. #
# ---------------------------------- #

sim_ID = 'L006N188'
for snap, proj in zip(NUMS_SNAPSHOTS[::-1], prog_idx):
    
    # Generate files with positions of DM particles
    fct.read_DM_positions(
        which_halos='halos', mass_select=12,  # unnecessary when giving index...
        random=False, snap_num=snap, sim=sim_ID, 
        halo_index=int(proj), init_m=m0
    )

    # Initial grid and DM positions.
    DM_raw = np.load(
        f'CubeSpace/DM_positions_{sim_ID}_snapshot_{snap}_{m0}Msun.npy'
    )*kpc
    grid = fct.grid_3D(GRID_L, GRID_S)
    init_cc = np.expand_dims(grid, axis=1)
    DM_pos = np.expand_dims(DM_raw, axis=0)
    DM_pos_for_cell_division = np.repeat(DM_pos, len(init_cc), axis=0)

    DM_lim = 10000

    fct.cell_division.count = -1  # initialize the recursion counter
    fct.cell_division(
        init_cc, DM_pos_for_cell_division, GRID_S, DM_lim, None,
        sim=sim_ID, snap_num=snap
        )
    adapted_cc = np.load(f'CubeSpace/adapted_cc_{sim_ID}_snapshot_{snap}.npy')
    adapted_DM = np.repeat(DM_pos, len(adapted_cc), axis=0)

    # Calculate gravity in each cell.
    fct.cell_gravity_3D(adapted_cc, adapted_DM, GRAV_RANGE, DM_SIM_MASS, snap)
    dPsi_grid = np.load(f'CubeSpace/dPsi_grid_snapshot_{snap}.npy')
    
    # Display and check order of magnitude of gravity.
    dPsi_grid /= (kpc/s**2) 
    mags = np.sqrt(np.sum(dPsi_grid**2, axis=1))
    print(mags[0])


seconds = time.perf_counter()-start
minutes = seconds/60.
hours = minutes/60.
print(f'Time sec/min/h: {seconds} sec, {minutes} min, {hours} h.')