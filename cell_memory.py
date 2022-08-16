from shared.preface import *
import shared.functions as fct


def check_grid(init_cc, DM_pos, parent_GRID_S, DM_lim):
    """
    Determine which cells have DM above threshold and thus need division.
    """

    # -------------- #
    # Preliminaries. #
    # -------------- #

    # Range within DM particles are "part of" the cell:
    # Circumscribed circle.
    # DM_range = parent_GRID_S/np.sqrt(2)
    # Inscribed circle.
    DM_range = parent_GRID_S/2.

    # Center all DM positions w.r.t. center, for all cells.
    DM_pos -= init_cc

    # Calculate distances of all DM to center, for all cells.
    DM_dis = np.sqrt(np.sum(DM_pos**2, axis=2))

    # Indices to order other arrays according to DM dist in ascending order.
    ind_2D = DM_dis.argsort(axis=1)
    ind_3D = np.repeat(np.expand_dims(ind_2D, axis=2), 3, axis=2)

    # Sort DM arrays according to DM distance with these indices.
    DM_pos_sort = np.take_along_axis(DM_pos, ind_3D, axis=1)
    DM_dis_sort = np.take_along_axis(DM_dis, ind_2D, axis=1)
    del DM_dis, ind_2D, ind_3D

    # This array is a bit confusing: It has the shape (X,2) and contains pairs 
    # of indices, where the 1st entry is the cell number and the 2nd the index 
    # of a DM particle inside the given range for that cell. X then counts all 
    # DM particles inside range across all cells.
    DM_IDs = np.argwhere(DM_dis_sort <= DM_range)
    del DM_dis_sort

    # Find the index unique to each cell, up to which the DM particles should
    # be kept, while the rest is outside range and no longer relevant.
    # Due to the method chosen, the last cell is not included and is attached 
    # seperately with np.vstack(...).
    row = DM_IDs.T[1]
    fw_diff = row[:-1] - row[1:]
    max_DM_rows_no_last_cell = np.argwhere(fw_diff.flatten() >= 0).flatten()
    max_DM_rows = np.vstack((DM_IDs[max_DM_rows_no_last_cell,:], DM_IDs[-1,:]))

    # Replace all entries beyond these indices (for each cell) with nan values.
    for i in range(len(init_cc)):

        if i in DM_IDs.T[0]:
            cell_ID = np.where(max_DM_rows[:,0] == i)[0][0]
            DM_pos_sort[i, max_DM_rows[cell_ID,1]+1:, :] = np.nan

        else:
            DM_pos_sort[i, ...] = np.nan

    # Drop "rows" common to all cells, which contain only nan values. This 
    # "maximum row" is determined by the highest value in the array, which 
    # contains the index of the row for each cell, beyond which the values are 
    # replaced by nan values.
    max_DM_rank = np.max(max_DM_rows[:,1])
    DM_cc_compact = np.delete(DM_pos_sort, np.s_[max_DM_rank+1:], axis=1)
    del DM_pos_sort

    # Count the number of DM particles in each cell, after all the filtering.
    DM_count = np.count_nonzero(~np.isnan(DM_cc_compact[:,:,0]), axis=1)

    # Drop all cells containing an amount of DM below the given threshold, 
    # from the DM positions array...
    cell_cut_IDs = DM_count <= DM_lim
    DM_cc_minimal = np.delete(DM_cc_compact, cell_cut_IDs, axis=0)
    del DM_cc_compact
    thresh = np.size(DM_cc_minimal, axis=0)

    return cell_cut_IDs, DM_cc_minimal, thresh


def cell_division_iterative(
    init_cc, DM_pos, parent_GRID_S, DM_lim, stable_cc, sim, snap_num
    ):

    # Initiate counters.
    thresh = 1
    cell_division_count = 0

    while thresh > 0:

        cell_cut_IDs, DM_cc_minimal, thresh = check_grid(
            init_cc, DM_pos, parent_GRID_S, DM_lim
        )

        #! If no cells are in need of division -> return final coords.
        if thresh == 0:

            if cell_division_count > 0:
                # The final iteration is a concatenation of the survival cells 
                # from the previous iteration and the newest sub8 cell coords...
                final_cc = np.concatenate((stable_cc, init_cc), axis=0)
                np.save(
                    f'CubeSpace/adapted_cc_{sim}_snapshot_{snap_num}.npy', 
                    final_cc
                )
                return cell_division_count
            else:
                # ...or the initial grid itself, if it's fine-grained already.
                np.save(
                    f'CubeSpace/adapted_cc_{sim}_snapshot_{snap_num}.npy', 
                    init_cc
                )
                return cell_division_count

        else:

            # ...and the initial cell coordinates grid. This array contains all 
            # cells (i.e. their coords. ), which need to be divided into 8 
            # "child cells", hence the name "parent cells".
            parent_cc = np.delete(init_cc, cell_cut_IDs, axis=0)

            # "Reset" the DM coords, s.t. all DM positions are w.r.t. the 
            # origin of (0,0,0) again. This way we can easily center them on 
            # the new child cells again, as done in later steps below.
            DM_cc_reset = DM_cc_minimal + parent_cc


            # -------------------------------------------------- #
            # Replace each parent cell by the 8 new child cells. #
            # -------------------------------------------------- #

            # Repeat each DM "column" 8 times. This will be the DM position 
            # array in the next iteration.
            DM_raw8 = np.repeat(DM_cc_reset, repeats=8, axis=0)

            # Create 8 new cells around origin of (0,0,0). The length and size 
            # of the new cells is determined by the previous length of the 
            # parent cell.
            sub8_GRID_S = parent_GRID_S/2.
            sub8_raw = fct.grid_3D(sub8_GRID_S, sub8_GRID_S)

            # Match dimensions of child-array(s) to parent-array(s).
            sub8 = np.expand_dims(sub8_raw, axis=0)
            sub8 = np.repeat(sub8, len(parent_cc), axis=0)

            # Center child-array(s) on parent cell coords.
            sub8_coords = sub8 - parent_cc

            # Reshape array to match repeated DM position array.
            sub8_coords = np.reshape(sub8_coords, (len(parent_cc)*8, 3))
            sub8_coords = np.expand_dims(sub8_coords, axis=1)

            # Delete all cells in initial cell coords array, corresponding to 
            # the cells in need of division, i.e. the parent cells.
            no_parents_cc = np.delete(init_cc, ~cell_cut_IDs, axis=0)

            if cell_division_count > 0:
                stable_cc_so_far = np.concatenate((stable_cc, no_parents_cc), axis=0)
            else:
                stable_cc_so_far = no_parents_cc

            # Overwrite variables for next loop.
            init_cc       = sub8_coords
            DM_pos        = DM_raw8
            parent_GRID_S = sub8_GRID_S
            stable_cc     = stable_cc_so_far

            cell_division_count += 1


# Values for file reading.
sim_ID = 'L006N188'
snap = '0036'
m0 = '2.59e+11'

# Initial grid and DM positions.
DM_raw = np.load(
    f'CubeSpace/DM_positions_{sim_ID}_snapshot_{snap}_{m0}Msun.npy'
)*kpc
grid = fct.grid_3D(GRID_L, GRID_S)
init_cc = np.expand_dims(grid, axis=1)
DM_raw = np.expand_dims(DM_raw, axis=0)
DM_pos_for_cell_division = np.repeat(DM_raw, len(init_cc), axis=0)*kpc

DM_lim = 1000

cell_division_count = cell_division_iterative(
    init_cc, DM_pos_for_cell_division, GRID_S, DM_lim, None,
    sim=sim_ID, snap_num=snap
    )
adapted_cc = np.load(f'CubeSpace/adapted_cc_{sim_ID}_snapshot_{snap}.npy')

print(init_cc.shape)
print(adapted_cc.shape)

print(f'Total cell division rounds: {cell_division_count}')