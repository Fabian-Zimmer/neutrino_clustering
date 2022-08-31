from shared.preface import *
import shared.functions as fct

@profile
def check_grid(init_cc, DM_pos, parent_GRID_S, DM_lim, gen_count):
    """
    Determine which cells have DM above threshold and thus need division.
    """

    # Center all DM positions w.r.t. center, for all cells.
    DM_pos -= init_cc

    # Cell length of current grid generation, used to limit DM particles.
    cell_len = np.ones((len(init_cc),1), dtype=np.float64)*parent_GRID_S/2.

    # Select DM particles inside each cell based on cube length generation.
    DM_in_cell_IDs = np.asarray(
        (np.abs(DM_pos[:,:,0]) <= cell_len) & 
        (np.abs(DM_pos[:,:,1]) <= cell_len) & 
        (np.abs(DM_pos[:,:,2]) <= cell_len)
    )

    # Set DM outside cell to nan values.
    DM_pos[~DM_in_cell_IDs] = np.nan

    # Sort all nan values to the bottom of axis 1.
    DM_sort = np.sort(DM_pos, axis=1)

    # Drop "rows" common to all cells, which contain only nan values. This is 
    # determined by the cell with the most non-nan entries.
    max_DM_rank = np.max(np.count_nonzero(~np.isnan(DM_sort[:,:,0]), axis=1))
    DM_compact = np.delete(DM_sort, np.s_[max_DM_rank:], axis=1)
    del DM_sort

    # Drop all cells containing an amount of DM below the given threshold, 
    # from the DM positions array.
    DM_count_all = np.count_nonzero(~np.isnan(DM_compact[:,:,0]), axis=1)
    stable_cells = DM_count_all <= DM_lim
    DM_cc_minimal = np.delete(DM_compact, stable_cells, axis=0)
    thresh = np.size(DM_cc_minimal, axis=0)
    del DM_count_all

    # Count the number of DM particles in stable cells.
    DM_stable_cells = np.delete(DM_compact, ~stable_cells, axis=0)
    DM_count_stable = np.count_nonzero(
        ~np.isnan(DM_stable_cells[:,:,0]), axis=1
    )
    del DM_compact

    # Calculate c.o.m coords for stable cells.
    DM_count_sync = np.expand_dims(DM_count_stable, axis=1)
    DM_count_sync[DM_count_sync==0] = 1  # to avoid divide by zero
    cell_com = np.nansum(DM_stable_cells, axis=1)/DM_count_sync
    del DM_count_stable

    # Count again, where zeros are present (not ones).
    DM_count_final = np.count_nonzero(
        ~np.isnan(DM_stable_cells[:,:,0]), axis=1
    )

    # Free up memory just in case.
    del DM_stable_cells

    return DM_count_final, cell_com, stable_cells, DM_cc_minimal, thresh


def cell_division(
    init_cc, DM_pos, parent_GRID_S, DM_lim, stable_cc, sim, snap_num
    ):

    # Initiate counters.
    thresh = 1
    cell_division_count = 0

    DM_count_arr = []
    cell_com_arr = []
    cell_gen_arr = []

    while thresh > 0:

        DM_count, cell_com, stable_cells, DM_cc_minimal, thresh = check_grid(
            init_cc, DM_pos, parent_GRID_S, DM_lim, cell_division_count
        )

        # Save DM count and c.o.m coords of stable cells.
        DM_count_arr.append(DM_count)
        cell_com_arr.append(cell_com)

        # If no cells are in need of division -> return final coords.
        if thresh == 0:

            # Append cell generation number of last iteration.
            cell_gen_arr.append(
                np.zeros(len(init_cc), int) + cell_division_count
            )
            
            # Convert nested lists to ndarrays.
            cell_gen_np = np.array(
                list(itertools.chain.from_iterable(cell_gen_arr))
            )
            DM_count_np = np.array(
                list(itertools.chain.from_iterable(DM_count_arr))
            )
            cell_com_np = np.array(
                list(itertools.chain.from_iterable(cell_com_arr))
            )

            if cell_division_count > 0:

                # The final iteration is a concatenation of the survival cells 
                # from the previous iteration and the newest sub8 cell coords 
                # (which are now init_cc).
                final_cc = np.concatenate((stable_cc, init_cc), axis=0)
                np.save(
                    f'CubeSpace/adapted_cc_{sim}_snapshot_{snap_num}.npy', 
                    final_cc
                )
                np.save(
                    f'CubeSpace/DM_count_{sim}_snapshot_{snap_num}.npy',
                    DM_count_np
                )
                np.save(
                    f'CubeSpace/cell_com_{sim}_snapshot_{snap_num}.npy',
                    cell_com_np
                )
                np.save(
                    f'CubeSpace/cell_gen_{sim}_snapshot_{snap_num}.npy',
                    cell_gen_np
                )
                return cell_division_count
            else:

                # Return initial grid itself, if it's fine-grained already.
                np.save(
                    f'CubeSpace/adapted_cc_{sim}_snapshot_{snap_num}.npy', 
                    init_cc
                )
                np.save(
                    f'CubeSpace/DM_count_{sim}_snapshot_{snap_num}.npy',
                    DM_count_np
                )
                np.save(
                    f'CubeSpace/cell_com_{sim}_snapshot_{snap_num}.npy',
                    cell_com_np
                )
                np.save(
                    f'CubeSpace/cell_gen_{sim}_snapshot_{snap_num}.npy',
                    cell_gen_np
                )
                return cell_division_count

        else:
            del DM_count, cell_com
            
            # Array containing all cells (i.e. their coords. ), which need to
            # be divided into 8 "child cells", hence the name "parent cells".
            parent_cc = np.delete(init_cc, stable_cells, axis=0)

            # -------------------------------------------------- #
            # Replace each parent cell by the 8 new child cells. #
            # -------------------------------------------------- #

            # note: 
            # There is no need for recentering the DM particles or the newly 
            # created child cells on the parent cells, etc. Since the DM 
            # particles are centered on their respective parent cells, they see 
            # the origin as (0,0,0) already. The child cells are also created 
            # around (0,0,0). The only crucial centering is inside the 
            # check_grid function with "DM_pos -= init_cc" !

            # Repeat each DM "column" 8 times. This will be the DM position 
            # array in the next iteration.
            DM_raw8 = np.repeat(DM_cc_minimal, repeats=8, axis=0)

            # Create 8 new cells around origin of (0,0,0). The length and size 
            # of the new cells is determined by the previous length of the 
            # parent cell.
            sub8_GRID_S = parent_GRID_S/2.
            sub8_raw = fct.grid_3D(sub8_GRID_S, sub8_GRID_S)

            # Match dimensions of child-array(s) to parent-array(s).
            sub8_coords = np.repeat(
                np.expand_dims(sub8_raw, axis=0), len(parent_cc), axis=0
            )

            # Reshape array to match repeated DM position array.
            sub8_coords = np.expand_dims(
                np.reshape(sub8_coords, (len(parent_cc)*8, 3)), axis=1
            )

            # Delete all cells in initial cell coords array, corresponding to 
            # the cells in need of division, i.e. the parent cells.
            no_parents_cc = np.delete(init_cc, ~stable_cells, axis=0)

            # Save generation index of stable cells.
            cell_gen_arr.append(
                np.zeros(len(no_parents_cc), int) + cell_division_count
            )

            if cell_division_count > 0:
                stable_cc_so_far = np.concatenate((stable_cc, no_parents_cc), axis=0)
            else:  # ending of first division loop
                stable_cc_so_far = no_parents_cc

            # Overwrite variables for next loop.
            init_cc       = sub8_coords
            DM_pos        = DM_raw8
            parent_GRID_S = sub8_GRID_S
            stable_cc     = stable_cc_so_far

            cell_division_count += 1

# Parameters.
snap = '0036'
m0 = '2.59e+11'
f = 'CubeSpace'
DM_LIM = 100000 # 0 rounds
# DM_LIM = 50000 # 1 round
# DM_LIM = 40000 # 2 rounds
# DM_LIM = 10000 # 4 rounds
# DM_LIM = 1000 # 6 rounds

# Input data.
grid = fct.grid_3D(GRID_L, GRID_S)
init_cc = np.expand_dims(grid, axis=1)
DM_raw = np.load(f'{f}/DM_positions_{SIM_ID}_snapshot_{snap}_{m0}Msun.npy')*kpc
DM_pos = np.expand_dims(DM_raw, axis=0)
DM_ready = np.repeat(DM_pos, len(init_cc), axis=0)
print('Input data shapes', init_cc.shape, DM_ready.shape)


cell_division_count = cell_division(
    init_cc, DM_ready, GRID_S, DM_LIM, 
    stable_cc=None, sim=SIM_ID, snap_num=snap
)
print(f'cell division rounds: {cell_division_count}')

# Output.
adapted_cc = np.load(
    f'CubeSpace/adapted_cc_{SIM_ID}_snapshot_{snap}.npy')
cell_gen = np.load(
    f'CubeSpace/cell_gen_{SIM_ID}_snapshot_{snap}.npy')
cell_com = np.load(
    f'CubeSpace/cell_com_{SIM_ID}_snapshot_{snap}.npy')
DM_count = np.load(
    f'CubeSpace/DM_count_{SIM_ID}_snapshot_{snap}.npy')

print('Shapes of output files:', adapted_cc.shape, cell_gen.shape, cell_com.shape, DM_count.shape)

print('Total DM count across all cells:', DM_count.sum())