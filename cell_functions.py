
from shared.preface import *
import shared.functions as fct

def cell_gravity_short_range(
    cell_coords, cell_gen, init_GRID_S,
    DM_pos, DM_lim, DM_sim_mass, smooth_l,
    out_dir, fname
):
    # Center all DM positions w.r.t. cell center.
    DM_pos -= cell_coords

    # Cell lengths to limit DM particles. Limit for the largest cell is 
    # GRID_S/2, not just GRID_S, therefore the cell_gen+1 !
    cell_len = np.expand_dims(init_GRID_S/(2**(cell_gen+1)), axis=1)

    # Select DM particles inside each cell based on cube length generation.
    DM_in_cell_IDs = np.asarray(
        (np.abs(DM_pos[:,:,0]) < cell_len) & 
        (np.abs(DM_pos[:,:,1]) < cell_len) & 
        (np.abs(DM_pos[:,:,2]) < cell_len)
    )

    # Set DM outside cell to nan values.
    DM_pos[~DM_in_cell_IDs] = np.nan

    # Sort all nan values to the bottom of axis 1, i.e. the DM-in-cell-X axis 
    # and truncate array based on DM_lim parameter. This simple way works since 
    # each cell cannot have more than DM_lim.

    ind_2D = DM_pos[:,:,0].argsort(axis=1)
    ind_3D = np.repeat(np.expand_dims(ind_2D, axis=2), 3, axis=2)
    DM_sort = np.take_along_axis(DM_pos, ind_3D, axis=1)
    DM_in = DM_sort[:,:DM_lim,:]
    del ind_2D, ind_3D, DM_sort

    # Calculate distances of DM and adjust array dimensionally.
    DM_dis = np.expand_dims(np.sqrt(np.sum(DM_in**2, axis=2)), axis=2)

    # ------------------------------ #
    # Calculate short-range gravity. #
    # ------------------------------ #

    # Offset DM positions by smoothening length of Camila's simulations.
    eps = smooth_l / 2.

    # nan values to 0 for numerator, and 1 for denominator to avoid infinities.
    quot = np.nan_to_num(cell_coords - DM_in, copy=False, nan=0.0) / \
        np.nan_to_num(
            np.power((DM_dis**2 + eps**2), 3./2.), copy=False, nan=1.0
        )
    del DM_in, DM_dis
    derivative= G*DM_sim_mass*np.sum(quot, axis=1)
    del quot
    
    # note: Minus sign, s.t. velocity changes correctly (see GoodNotes).
    dPsi_short = np.asarray(-derivative, dtype=np.float64)
    np.save(f'{out_dir}/dPsi_grid_{fname}_short_range.npy', dPsi_short)


def cell_gravity_long_range(
    c_id, b_id, cellX_coords, 
    DM_count, cell_com, 
    DM_sim_mass, smooth_l, out_dir
):

    # Distances between cell centers and cell c.o.m. coords.
    com_dis = np.sqrt(np.sum((cellX_coords-cell_com)**2, axis=1))

    # Adjust dimensionally for later division.
    com_dis_sync = np.expand_dims(com_dis, axis=1)


    # Offset DM positions by smoothening length of Camila's simulations.
    eps = smooth_l / 2.

    # Long-range gravity component for each cell (without including itself).
    quot = (cellX_coords-cell_com)/np.power((com_dis_sync**2 + eps**2), 3./2.)
    DM_count_sync = np.expand_dims(DM_count, axis=1)
    derivative = G*DM_sim_mass*np.sum(DM_count_sync*quot, axis=0)
    del quot

    # note: Minus sign, s.t. velocity changes correctly (see GoodNotes).
    dPsi_long = np.asarray(-derivative, dtype=np.float64)
    np.save(f'{out_dir}/cell{c_id}_batch{b_id}_long_range.npy', dPsi_long)


def batch_generators_long_range(
    cell_coords, com_coords, DM_counts,
    chunk_size 
):
    cells = len(cell_coords)
    cell_nums = np.arange(cells)

    num = math.ceil(cells/chunk_size)

    # Arrays used for naming files.
    id_arr = np.array([idx+1 for idx in cell_nums for _ in range(num)])
    batch_arr = np.array([b+1 for _ in cell_nums for b in range(num)])

    # Coord of cell, for which long-range gravity gets calculated.
    coord_arr = np.array([cc for cc in cell_coords for _ in range(num)])

    # Chunks for DM_count array, as a generator for all cells.
    count_gens = (c for _ in cell_nums for c in chunks(chunk_size, DM_counts))
    count_chain = chain(gen for gen in count_gens)

    # Chunks for cell_com array, as a generator for all cells.
    com_gens = (c for _ in cell_nums for c in chunks(chunk_size, com_coords))
    com_chain = chain(gen for gen in com_gens)

    return id_arr, batch_arr, coord_arr, count_chain, com_chain
