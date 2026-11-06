"""Precalculation of the adaptive cell grid around a DM halo and of the
gravitational force (short-range + long-range quadrupole) in each cell."""

import math
from itertools import chain, zip_longest

import numpy as np
from funcy import chunks
import jax.numpy as jnp

from cnb.units import Params
from cnb.utils import Utils


class SimGrid():

    @staticmethod
    def grid_3D(l, s, origin_coords=[0.,0.,0.,]):
        """
        Generate 3D cell center coordinate grid (built around origin_coords) 
        extending from center until l in all 3 axes, spaced apart by s.
        If l=s then 8 cells will be generated (used for dividing 1 cell into 8).
        """

        # Generate edges of 3D grid.
        l_eps = l/10.
        x, y, z = np.mgrid[-l:l+l_eps:s, -l:l+l_eps:s, -l:l+l_eps:s]

        # Calculate centers of each axis.
        x_centers = (x[1:,...] + x[:-1,...])/2.
        y_centers = (y[:,1:,:] + y[:,:-1,:])/2.
        z_centers = (z[...,1:] + z[...,:-1])/2.

        # Create center coord.-pairs. and truncating outermost "layer".
        centers3D = np.array([
            x_centers[:,:-1,:-1], 
            y_centers[:-1,:,:-1], 
            z_centers[:-1,:-1,:]
        ])

        cent_coordPairs3D = centers3D.reshape(3,-1).T 

        # Shift center of creation from (0,0,0) to other coords.
        cent_coordPairs3D += origin_coords

        return cent_coordPairs3D


    @staticmethod
    def check_grid(init_grid, DM_pos, parent_GRID_S, DM_lim, FCT_DM_shell_edges, FCT_shell_multipliers):
        """
        Determine which cells have DM above threshold and thus need division.
        """

        # Center all DM positions w.r.t. center, for all cells.
        DM_pos -= init_grid

        # Cell length of current grid generation, used to limit DM particles.
        cell_len = np.ones((len(init_grid),1), dtype=np.float64)*parent_GRID_S/2.

        # Select DM particles inside each cell based on cube length generation.
        DM_in_cell_IDs = np.asarray(
            (np.abs(DM_pos[:,:,0]) < cell_len) & 
            (np.abs(DM_pos[:,:,1]) < cell_len) & 
            (np.abs(DM_pos[:,:,2]) < cell_len)
        )

        # Set DM outside cell to nan values.
        DM_pos[~DM_in_cell_IDs] = np.nan

        # Sort all nan values to the bottom of axis 1, i.e. the DM-in-cell-X axis.
        ind_2D = DM_pos[:,:,0].argsort(axis=1)
        ind_3D = np.repeat(np.expand_dims(ind_2D, axis=2), 3, axis=2)
        DM_sort = np.take_along_axis(DM_pos, ind_3D, axis=1)
        del ind_2D, ind_3D

        # Drop "rows" common to all cells, which contain only nan values. This is 
        # determined by the cell with the most non-nan entries.
        DM_count_cells = np.count_nonzero(~np.isnan(DM_sort[:,:,0]), axis=1)
        DM_compact = np.delete(
            DM_sort, np.s_[np.max(DM_count_cells):], axis=1
        )
        cells = len(DM_compact)
        del DM_sort


        ### Adjust DM threshold to the distance of the cell from the center.
        ### (cells farther away have higher DM threshold, i.e. less division)

        # Distance from center for each cell.
        grid_dis = np.sum(np.sqrt(init_grid**2), axis=2) # shape = (cells, 1)

        # Radial distance of each shell center, adjust array dimensionally.
        shell_cents = (FCT_DM_shell_edges[:-1] + FCT_DM_shell_edges[1:])/2.
        shells_sync = np.repeat(np.expand_dims(shell_cents, axis=0), cells, axis=0)

        # Find shell center, which each cell is closest to.
        which_shell = np.abs(grid_dis - shells_sync).argmin(axis=1)

        # Final DM limit for each cell.
        cell_DMlims = np.array(
            [FCT_shell_multipliers[k] for k in which_shell]
        )*DM_lim


        ### Drop all cells containing an amount of DM below the given threshold, 
        ### from the DM positions array.

        # stable_cells = DM_count_cells <= DM_lim  # note: original
        stable_cells = DM_count_cells <= cell_DMlims
        DM_unstable_cells = np.delete(DM_compact, stable_cells, axis=0)
        thresh = np.size(DM_unstable_cells, axis=0)
        del DM_count_cells

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
        del DM_count_stable, DM_count_sync

        # Reset cell_com coords. of stable cells to proper coordinates.
        good_grid = np.squeeze(np.delete(init_grid, ~stable_cells, axis=0), axis=1)
        cell_com += good_grid
        del good_grid
        
        # note: 
        # cell_com can have (0,0,0) for a cell. Doesn't matter, since DM_count in 
        # cell is then 0, which will set term in long-range gravity to zero.

        # Count again, where zeros are present (i.e. not replaced by ones).
        DM_count_final = np.count_nonzero(
            ~np.isnan(DM_stable_cells[:,:,0]), axis=1
        )

        # Free up memory just in case.
        del DM_stable_cells

        return DM_count_final, cell_com, stable_cells, DM_unstable_cells, thresh


    @staticmethod
    def cell_division(
        init_grid, DM_pos, parent_GRID_S, DM_lim, stable_grid, FCT_DM_shell_edges, FCT_shell_multipliers, out_dir, fname
        ):


        # Initiate counters.
        thresh = 1
        cell_division_count = 0

        DM_count_l = []
        cell_com_l = []
        cell_gen_l = []

        while thresh > 0:

            DM_count, cell_com, stable_cells, DM_parent_cells, thresh = SimGrid.check_grid(
                init_grid, DM_pos, parent_GRID_S, DM_lim, FCT_DM_shell_edges, FCT_shell_multipliers
            )

            # Save DM count and c.o.m coords of stable cells.
            DM_count_l.append(DM_count)
            cell_com_l.append(cell_com)

            # If no cells are in need of division -> return final coords.
            if thresh == 0:

                # Append cell generation number of last iteration.
                cell_gen_l.append(
                    np.zeros(len(init_grid), int) + cell_division_count
                )
                
                # Convert nested lists to ndarrays.
                cell_gen_np = np.array(list(chain.from_iterable(cell_gen_l)))
                DM_count_np = np.array(list(chain.from_iterable(DM_count_l)))
                cell_com_np = np.array(list(chain.from_iterable(cell_com_l)))

                if cell_division_count > 0:

                    # The final iteration is a concatenation of the survival cells 
                    # from the previous iteration and the newest sub8 cell coords 
                    # (which are now init_grid).
                    final_cc = np.concatenate((stable_grid, init_grid), axis=0)
                    np.save(f'{out_dir}/fin_grid_{fname}.npy', final_cc)

                else:
                    # Return initial grid itself, if it's fine-grained already.
                    np.save(f'{out_dir}/fin_grid_{fname}.npy', init_grid)

                # Save DM count, c.o.m. coord, and generation, for all cells.
                np.save(f'{out_dir}/DM_count_{fname}.npy', DM_count_np)
                np.save(f'{out_dir}/cell_com_{fname}.npy', cell_com_np)
                np.save(f'{out_dir}/cell_gen_{fname}.npy', cell_gen_np)

                return cell_division_count


            else:
                del DM_count, cell_com
                
                # Array containing all cells (i.e. their coords.), which need to
                # be divided into 8 "child cells", hence the name "parent cells".
                parent_cells = np.delete(init_grid, stable_cells, axis=0)
                pcs = len(parent_cells)

                # Array containing all cells, which are "stable" and need no 
                # division, so excluding all parent cells.
                no_parent_cells = np.delete(init_grid, ~stable_cells, axis=0)

                # Save generation index of stable cells.
                cell_gen_l.append(
                    np.zeros(len(no_parent_cells), int) + cell_division_count
                )

                # -------------------------------------------------- #
                # Replace each parent cell by the 8 new child cells. #
                # -------------------------------------------------- #

                # Reset DM on parent cells, such that they can be centered on new 
                # child cells again later.
                DM_reset = DM_parent_cells + parent_cells
                # note: Array doesn't contain duplicate DM, each cell has unique DM.

                # Repeat each DM "column" 8 times. This will be the DM position 
                # array in the next iteration.
                DM_rep8 = np.repeat(DM_reset, repeats=8, axis=0)

                # Create 8 new cells around origin of (0,0,0). The length and size 
                # of the new cells is determined by the previous length of the 
                # parent cell, i.e. half of it.
                sub8_GRID_S = parent_GRID_S/2.
                sub8_raw = SimGrid.grid_3D(sub8_GRID_S, sub8_GRID_S)

                # Temporarily reshape to center on parent cells.
                sub8_temp = np.tile(sub8_raw, (pcs,1)).reshape((pcs, 8, 3))

                # Center each new 8-batch of child cells on a different parent cell.
                sub8_coords = np.reshape(sub8_temp + parent_cells, (pcs*8, 1, 3))
                del sub8_raw, sub8_temp, parent_cells

                if cell_division_count > 0:
                    stable_grid_so_far = np.concatenate(
                        (stable_grid, no_parent_cells), axis=0
                    )
                else:  # ending of first division loop
                    stable_grid_so_far = no_parent_cells

                # Overwrite variables for next loop.
                init_grid     = sub8_coords
                DM_pos        = DM_rep8
                parent_GRID_S = sub8_GRID_S
                stable_grid   = stable_grid_so_far

                cell_division_count += 1


    @staticmethod
    def chunksize_short_range(cells, DM_tot, max_DM_lim, core_mem_MB):

        # note: mem_MB specific to peak memory usage in cell_gravity_short_range.
        # -> Peak memory after calculation of ind_2D,ind_3D,etc. sorting arrays.

        elem = 8                               # 8 bytes for standard np.float64
        mem_type0 = cells*3 * elem             # for list to ndarray of cell_coords
        mem_type1 = cells*DM_tot * elem        # for ind_2D
        mem_type2 = cells*DM_tot*3 * elem      # for DM_pos_sync, ind_3D, DM_sort
        mem_type3 = cells*max_DM_lim*3 * elem  # for DM_in

        mem_MB = (mem_type0+mem_type1+(3*mem_type2)+mem_type3)/1.e6

        batches = 1
        while mem_MB >= 0.90*core_mem_MB:
            mem_MB *= batches
            batches += 1
            mem_MB /= batches

        chunksize = math.ceil(cells/batches)

        return chunksize


    @staticmethod
    def batch_generators_short_range(cell_coords, cell_gen, chunksize):

        cells = len(cell_coords)

        batches = math.ceil(cells/chunksize)
        batch_arr = np.arange(batches)

        cell_chunks = list(chunks(chunksize, cell_coords))
        cgen_chunks = list(chunks(chunksize, cell_gen))
        
        return batch_arr, cell_chunks, cgen_chunks


    @staticmethod
    def cell_gravity_short_range(
        cell_coords_in, cell_gen, init_GRID_S,
        DM_pos, DM_lim, DM_sim_mass, FCT_shell_multipliers, smooth_l,
        out_dir, b_id, max_b_len
    ):

        cell_coords = np.expand_dims(np.array(cell_coords_in), axis=1)
        cell_gen = np.array(cell_gen)

        # Center all DM positions w.r.t. cell center.
        # DM_pos already in shape = (1, DM_particles, 3)
        DM_pos_sync = np.repeat(DM_pos, len(cell_coords), axis=0)
        DM_pos_sync -= cell_coords

        # Cell lengths to limit DM particles. Limit for the largest cell is 
        # GRID_S/2, not just GRID_S, therefore the cell_gen+1 !
        cell_len = np.expand_dims(init_GRID_S/(2**(cell_gen+1)), axis=1)
        
        # Select DM particles inside each cell based on cube length generation.
        DM_in_cell_IDs = np.asarray(
            (np.abs(DM_pos_sync[...,0]) < cell_len) & 
            (np.abs(DM_pos_sync[...,1]) < cell_len) & 
            (np.abs(DM_pos_sync[...,2]) < cell_len)
        )
        #? < results in 1 missing DM particle. Using <= though overcounts
        #? is there a way to get every DM particle by adjusting rtol and atol ?
        # note: -> not pressing for now however

        # Set DM outside cell to nan values.
        DM_pos_sync[~DM_in_cell_IDs] = np.nan

        # Save the DM IDs, such that we know which particles are in which cell.
        # This will be used in the long-range gravity calculations.
        DM_in_cell_IDs_compact = np.argwhere(DM_in_cell_IDs==True)
        DM_in_cell_IDs_compact[:,0] += (max_b_len*b_id)    
        np.save(f'{out_dir}/batch{b_id}_DM_in_cell_IDs.npy', DM_in_cell_IDs_compact)

        # Sort all nan values to the bottom of axis 1, i.e. the DM-in-cell-X axis 
        # and truncate array based on DM_lim parameter. This simple way works since 
        # each cell cannot have more than DM_lim (times the last shell multiplier).
        ind_2D = DM_pos_sync[:,:,0].argsort(axis=1)
        ind_3D = np.repeat(np.expand_dims(ind_2D, axis=2), 3, axis=2)
        DM_sort = np.take_along_axis(DM_pos_sync, ind_3D, axis=1)
        DM_in = DM_sort[:,:DM_lim*FCT_shell_multipliers[-1],:]
        
        # Calculate distances of DM and adjust array dimensionally.
        DM_dis = np.expand_dims(np.sqrt(np.sum(DM_in**2, axis=2)), axis=2)

        # Offset DM positions by smoothening length of Camila's simulations.
        eps = smooth_l / 2.

        # Quotient in sum (see formula). Can contain nan values, thus the np.nansum 
        # for the derivative, s.t. these values don't contribute. We center DM on 
        # c.o.m. of cell C, so we only need DM_in in numerator.
        quot = (-DM_in)/np.power((DM_dis**2 + eps**2), 3./2.)
        
        # Gradient of potential.
        gradient = Params.G*DM_sim_mass*np.nansum(quot, axis=1)

        # Acceleration is negative value of (grav. pot.) gradient.
        np.save(f'{out_dir}/batch{b_id}_short_range.npy', -gradient)


    @staticmethod
    def chunksize_long_range(cells, core_mem_MB):
        
        # note: mem_MB specific to peak memory usage in cell_gravity_long_range.
        # -> Peak memory after calculation of derivative.

        elem = 8                          # 8 bytes for standard np.float64
        mem_type1 = 3*elem                # for derivative
        mem_type2 = cells*3*elem          # for quot
        mem_type3 = cells*elem            # for DM_count_sync

        mem_MB = (mem_type1+mem_type2+mem_type3)/1.e6

        batches = 1
        while mem_MB >= 0.90*core_mem_MB:
            mem_MB *= batches
            batches += 1
            mem_MB /= batches

        chunksize = math.ceil(cells/batches)

        return chunksize
        

    @staticmethod
    def batch_generators_long_range(
        cell_coords, com_coords, cell_gen, 
        DM_counts, chunksize 
    ):

        # Cell number, chunksize and resulting number of batches.
        cells = len(cell_coords)
        cell_nums = np.arange(cells)
        batches = math.ceil(cells/chunksize)


        ### ------- ###
        ### Arrays. ###
        ### ------- ###

        # Arrays are repeated/tiled to match length & number of batches.

        batch_IDs = np.tile(np.arange(batches), cells)  # batch labels/indices
        # [0,...,batches,0,...,batches,...]
        
        cellC_rep = np.repeat(cell_nums, batches)  # cell labels/indices
        # [0,...,0,1,...,1,...,cells-1,...,cells-1]

        cellC_cc = np.repeat(cell_coords, batches, axis=0)  # cell coordinates
        # [[x1,y1,z1],...,[x1,y1,z1],...]

        cellC_gen = np.repeat(cell_gen, batches)  # cell generations
        # e.g. [0,...,0,0,...,0,1,...1,...], ... repeated to match nr of batches

        ### ----------- ###
        ### Generators. ###
        ### ----------- ###

        # We split up all arrays into smaller chunks, s.t. each batch gets a 
        # smaller set of cells. Generators are used to benefit from the chunks() 
        # function, then converted to lists (map() will do this anyway).
        
        # We do this for:
        # 1. the cell_IDs of cells present in a batch, 
        # 2. the DM counts of all cells in a batch, 
        # 3. the c.o.m. coords of all cells in a batch,
        # 4. the (cell division) generation index for all cells in a batch:

        # 1. 
        cib_IDs = list(
            (c for _ in cell_nums for c in chunks(chunksize, cell_nums)))

        # 2.
        counts = list(
            (c for _ in cell_nums for c in chunks(chunksize, DM_counts)))

        # 3.
        coms = list(
            (c for _ in cell_nums for c in chunks(chunksize, com_coords)))

        # 4.
        gens = list(
            (c for _ in cell_nums for c in chunks(chunksize, cell_gen)))

        return batch_IDs, cellC_rep, cellC_cc, cellC_gen, \
            cib_IDs, counts, coms, gens


    @staticmethod
    def cell_gravity_long_range_quadrupole(
        c_id, cib_ids, b_id, cellC_cc, cell_com, cell_gen, init_GRID_S,
        DM_pos, DM_count, DM_in_cell_IDs, DM_sim_mass, out_dir, 
        max_b_len, cellC_gen
    ):
        # Prefix "cellC" denotes the cell to calculate the long-range forces for.

        # Convert the list inputs to numpy arrays.
        cib_ids = np.array(cib_ids)
        cell_com = np.array(cell_com)
        cell_gen = np.array(cell_gen)
        DM_count = np.array(DM_count)

        # Array, where cell C is centered on c.o.m. coords. of all cells.
        cellC_sync = np.repeat(
            np.expand_dims(cellC_cc, axis=0), len(cell_com), axis=0
        ) 
        cellC_sync -= cell_com

        # Get (complete, not half) cell length of all cells and current cell C.
        cell_len = init_GRID_S/(2**(cell_gen))
        cellC_len = init_GRID_S/(2**(cellC_gen))

        # Distance of cellC to all cell_com's.
        cellC_dis = np.sqrt(np.sum(cellC_sync**2, axis=1))

        # Cells with distance bewlow critical we compute quadrupole moments for.
        r_crit = 1.5*cellC_len*((cellC_gen+1)**0.6)  # set to -1 for no quadrupoles
        multipole_IDs = np.argwhere(cellC_dis <= r_crit).flatten()

        if c_id in cib_ids:
            # Overwrite element corresponding to cell C from all arrays, to avoid
            # self-gravity of the cell (taken care of by short-range gravity).

            # First find the correct index for the current cell. We have to take 
            # this akin to a "modulo" according to the maximum batch length. E.g. 
            # if the max. batch length is 100, but we are at cell label/index 100, 
            # then this would be index '0' again for arrays in this function.
            cellC_idx = (c_id - (max_b_len*b_id))

            # Set cell C array element(s) to 0 or nan, s.t. it just adds 0 later on.
            cell_com[cellC_idx] = np.nan  # c.o.m. of cell C to [nan,nan,nan]
            cell_len[cellC_idx] = 0  # 0 cell length
            DM_count[cellC_idx] = 0  # 0 DM count

            # If cell C has 0 DM, then its c.o.m == coords., so set cellC_dis to 1.
            # (avoids divide by zero in these (edge) cases later on)
            if cellC_dis[cellC_idx] == 0:
                cellC_dis[cellC_idx] = 1

        # DM count for multipole cells.
        DM_count_mpoles = DM_count[multipole_IDs]

        # All DM particles (their positions), which are in multipole cells.
        # Adjust/match multipole_IDs first, since they are limited to size of batch.
        comparison_IDs = multipole_IDs + (max_b_len*b_id)
        DM_IDs = DM_in_cell_IDs[np.in1d(DM_in_cell_IDs[:,0], comparison_IDs)][:,1]
        DM_pos_mpoles = np.take(DM_pos, DM_IDs, axis=0)

        # Split this total DM array into DM chunks present in each multipole cell:
        # We can do this by breaking each axis into sub-arrays, with length 
        # depending on how many DM particles are in each cell (which is stored in 
        # DM_count of multipole cells, i.e. DM_count_mpoles).
        
        # Special case, where all multipole cells have no DM.
        # (This can happen e.g. if cell C is an outermost cell)
        if np.all(DM_count_mpoles==0):
            DM_mpoles = np.full(shape=(len(multipole_IDs),1,3), fill_value=np.nan)

        # "Normal" case, where some or all multipole cells contain DM.
        else:
            breaks = np.cumsum(DM_count_mpoles[:-1])
            ax0_split = np.split(DM_pos_mpoles[:,0], breaks)
            ax1_split = np.split(DM_pos_mpoles[:,1], breaks)
            ax2_split = np.split(DM_pos_mpoles[:,2], breaks)

            # Fill each axis with nans to obtain dimensionally valid ndarray.
            DM_axis0 = np.array(list(zip_longest(*ax0_split, fillvalue=np.nan))).T
            DM_axis1 = np.array(list(zip_longest(*ax1_split, fillvalue=np.nan))).T
            DM_axis2 = np.array(list(zip_longest(*ax2_split, fillvalue=np.nan))).T

            # Recombine all axes into one final DM positions array.
            DM_mpoles = np.stack((DM_axis0, DM_axis1, DM_axis2), axis=2)
            del ax0_split, ax1_split, ax2_split, DM_axis0, DM_axis1, DM_axis2    


        # Select c.o.m. of multipole cells and adjust dimensionally.
        mpoles_com = np.expand_dims(cell_com[multipole_IDs], axis=1)

        # Center DM in multipole cells (labeled with "J") on their c.o.m. coords.
        DM_mpoles -= mpoles_com

        # Calculate distances of all DM in J cells from their c.o.m. coord.
        DMj_dis = np.expand_dims(np.sqrt(np.sum(DM_mpoles**2, axis=2)), axis=2)
        
        # Array, where cell C is centered on c.o.m. coords. of J cells.
        cellC_Jcoms = np.repeat(
            np.expand_dims(cellC_cc, axis=(0,1)), 
            len(mpoles_com), axis=0
        )
        cellC_Jcoms -= mpoles_com

        # Distance of cell C to c.o.m. of multipole cells.
        cellC_Jdis = np.sqrt(np.sum(cellC_Jcoms**2, axis=2))


        ### -------------------- ###
        ### Quadrupole formulas. ###
        ### -------------------- ###
        # See GoodNotes for notation.

        # Terms appearing in the quadrupole term.
        QJ_aa = np.nansum(3*DM_mpoles**2 - DMj_dis**2, axis=1)
        QJ_ab = np.nansum(3*DM_mpoles*np.roll(DM_mpoles, 1, axis=2), axis=1)

        # Reduce dimensions.
        cellC_Jcoms = np.squeeze(cellC_Jcoms, axis=1)
        cellC_Jdis_1D = np.squeeze(cellC_Jdis, axis=1)
        
        # Permute order of coords by one, i.e. (x,y,z) -> (z,x,y).
        cellC_Jcoms_roll = np.roll(cellC_Jcoms, 1, axis=1)

        # Factors of 2 are for the symmetry of QJ_ab elements.
        term1_aa = np.nansum(QJ_aa*cellC_Jcoms, axis=1)
        term1_ab = np.nansum(2*QJ_ab*cellC_Jcoms_roll, axis=1)
        term1 = np.expand_dims((term1_aa+term1_ab)/cellC_Jdis_1D**5, axis=1)

        term2_pre = 5*cellC_Jcoms/(2*cellC_Jdis**7)
        term2_aa = np.nansum(QJ_aa*cellC_Jcoms**2, axis=1)
        term2_ab = np.nansum(2*QJ_ab*cellC_Jcoms*cellC_Jcoms_roll, axis=1)
        term2 = term2_pre*np.expand_dims(term2_aa+term2_ab, 1)

        dPsi_multipole_cells = Params.G*DM_sim_mass*np.nansum(-term1+term2, axis=0)
        

        ### ---------------------- ###
        ### Monopole of all cells. ###
        ### ---------------------- ###

        mono_com = np.expand_dims(cell_com, axis=1)

        # Array, where cell C is centered on c.o.m. coords. of monopole cells.
        cellC_Jcoms_mono = np.repeat(
            np.expand_dims(np.expand_dims(cellC_cc, axis=0), axis=0), 
            len(mono_com), axis=0
        )
        cellC_Jcoms_mono -= mono_com

        # Distance of cell C to c.o.m. of monopole cells.
        cellC_dis_mono = np.sqrt(np.sum(cellC_Jcoms_mono**2, axis=2))
        
        # Long-range force of all cells due to monopole.
        cellC_Jcoms_mono_2D = np.squeeze(cellC_Jcoms_mono, axis=1)
        DM_count_mono_sync = np.expand_dims(DM_count, axis=1)
        dPsi_monopole_cells = Params.G*DM_sim_mass*np.nansum(
            DM_count_mono_sync*cellC_Jcoms_mono_2D/(cellC_dis_mono**3), axis=0)

        # Total gradient.
        gradient_lr = dPsi_multipole_cells + dPsi_monopole_cells
        
        # Acceleration is negative value of (grav. pot.) gradient.
        np.save(f'{out_dir}/cell{c_id}_batch{b_id}_long_range.npy', -gradient_lr)


    @staticmethod
    def grid_data(origID, data_dir, start_snap=12, end_snap=36, benchmark=False):
        
        # Initialize lists to store the data
        dPsi_grid_list = []
        cell_grid_list = []
        cell_gens_list = []

        # Loop over the snapshot numbers
        for snap_num in range(start_snap, end_snap + 1):
            snap = str(snap_num).zfill(4)  # Format the number as a 4-digit string

            # Load the data files
            # The 3 arrays have shapes (N,3), (N,1,3) and (N,), respectively
            if benchmark:
                dPsi_grid_f = f'{data_dir}/dPsi_grid_benchmark_halo_snap_{snap}.npy'
                cell_grid_f = f'{data_dir}/fin_grid_benchmark_halo_snap_{snap}.npy'
                cell_gens_f = f'{data_dir}/cell_gen_benchmark_halo_snap_{snap}.npy'
            else:
                dPsi_grid_f = f'{data_dir}/dPsi_grid_origID{origID}_snap_{snap}.npy'
                cell_grid_f = f'{data_dir}/fin_grid_origID{origID}_snap_{snap}.npy'
                cell_gens_f = f'{data_dir}/cell_gen_origID{origID}_snap_{snap}.npy'

            # Append arrays to list, squeezing (N,1,3) cell_grid to (N,3) shape
            dPsi_grid_list.append(jnp.load(dPsi_grid_f))
            cell_grid_list.append(jnp.squeeze(jnp.load(cell_grid_f)))
            cell_gens_list.append(jnp.load(cell_gens_f))

        # Pad with zeros and convert lists to jnp arrays
        dPsi_grid = Utils.fill_2d_arrays_with_zeros_jax(dPsi_grid_list)
        cell_grid = Utils.fill_2d_arrays_with_zeros_jax(cell_grid_list)
        cell_gens = Utils.fill_1d_arrays_with_zeros_jax(cell_gens_list)

        return dPsi_grid, cell_grid, cell_gens
