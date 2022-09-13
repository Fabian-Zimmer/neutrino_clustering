from sqlite3 import adapt
from discrete_simulation_MW import NrDM_SNAPSHOTS
from shared.preface import *
import shared.functions as fct


start = time.perf_counter()

# Generate progenitor index list.
m0, prog_idx = fct.read_MergerTree(init_halo=HALO_INDEX)

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

# Save number of DM particles used for each snapshot.
NrDM_snaps = np.zeros(len(NUMS_SNAPSHOTS))

for j, (snap, proj) in enumerate(zip(NUMS_SNAPSHOTS[::-1], prog_idx)):

# For single snapshot test.
# snap = NUMS_SNAPSHOTS[::-1][0]
# proj = prog_idx[0]

    # Generate files with positions of DM particles
    fct.read_DM_positions(
        which_halos='halos', random=False, 
        snap_num=snap, sim=SIM_ID, 
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
        sim=SIM_ID, snap_num=snap, m0=m0
        )

    # Arrays produced by cell division algorithm.
    adapted_cc = np.load(
        f'CubeSpace/adapted_cc_{SIM_ID}_snapshot_{snap}_{m0}Msun.npy')
    cell_gen = np.load(
        f'CubeSpace/cell_gen_{SIM_ID}_snapshot_{snap}_{m0}Msun.npy')
    cell_com = np.load(
        f'CubeSpace/cell_com_{SIM_ID}_snapshot_{snap}_{m0}Msun.npy')
    DM_count = np.load(
        f'CubeSpace/DM_count_{SIM_ID}_snapshot_{snap}_{m0}Msun.npy')
    NrDM_snaps[j] = np.sum(DM_count)
    # print(adapted_cc.shape, cell_gen.shape, cell_com.shape, DM_count.shape)

    # Generate gravity grid, in batches of cells, due to memory intensity.
    batch_size = 70
    bs_cc = chunks(batch_size, adapted_cc)
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
        list(itertools.chain.from_iterable(dPsi_batches))
    )
    np.save(f'CubeSpace/dPsi_grid_snapshot_{snap}_{m0}Msun.npy', dPsi_combined)
    fct.delete_temp_data('CubeSpace/dPsi_*batch*.npy') 

    '''
    # Calculate gravity in each cell.
    adapted_DM = np.repeat(DM_pos, len(adapted_cc), axis=0)
    fct.cell_gravity_3D(
        adapted_cc, cell_com, cell_gen,
        adapted_DM, DM_count, DM_SIM_MASS, snap
    )
    '''

    print(f'snapshot {snap} : cell division rounds: {cell_division_count}')

np.save('shared/NrDM_SNAPSHOTS.npy', NrDM_snaps)

seconds = time.perf_counter()-start
minutes = seconds/60.
hours = minutes/60.
print(f'Time sec/min/h: {seconds} sec, {minutes} min, {hours} h.')