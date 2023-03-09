# Save how much memory is used by OS and not available for script.
import psutil
MB_UNIT = 1024**2
OS_MEM = (psutil.virtual_memory().used)

from shared.preface import *
from shared.shared_functions import Fermi_Dirac, number_density, velocity_to_momentum, delete_temp_data
total_start = time.perf_counter()

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', required=True)
parser.add_argument('-st', '--sim_type', required=True)
parser.add_argument('-mg', '--mass_gauge', required=True)
parser.add_argument('-mr', '--mass_range', required=True)
parser.add_argument('-hn', '--halo_num', required=True)
args = parser.parse_args()


### ================================= ###
### Load box & simulation parameters. ###
### ================================= ###

# Box parameters.
with open(f'{args.directory}/box_parameters.yaml', 'r') as file:
    box_setup = yaml.safe_load(file)

box_file_dir = box_setup['File Paths']['Box File Directory']
DM_mass = box_setup['Content']['DM Mass [Msun]']*Msun
Smooth_L = box_setup['Content']['Smoothening Length [pc]']*pc
z0_snap_4cif = box_setup['Content']['z=0 snapshot']

# Simulation parameters.
with open(f'{args.directory}/sim_parameters.yaml', 'r') as file:
    sim_setup = yaml.safe_load(file)

CPUs_pre = sim_setup['CPUs_precalculations']
CPUs_sim = sim_setup['CPUs_simulations']
mem_lim_GB = sim_setup['memory_limit_GB']
DM_lim = sim_setup['DM_in_cell_limit']
integration_solver = sim_setup['integration_solver']
init_x_dis = sim_setup['initial_haloGC_distance']
init_xyz = np.array([init_x_dis, 0., 0.])
neutrinos = sim_setup['neutrinos']

# Load arrays.
nums_snaps = np.load(f'{args.directory}/nums_snaps.npy')
zeds_snaps = np.load(f'{args.directory}/zeds_snaps.npy')

z_int_steps = np.load(f'{args.directory}/z_int_steps.npy')
s_int_steps = np.load(f'{args.directory}/s_int_steps.npy')
neutrino_massrange = np.load(f'{args.directory}/neutrino_massrange_eV.npy')*eV
DM_shell_edges = np.load(f'{args.directory}/DM_shell_edges.npy')
shell_multipliers = np.load(f'{args.directory}/shell_multipliers.npy')


# Load constants and arrays, which the functions.py script needs.
FCT_h = box_setup['Cosmology']['h']
FCT_H0 = FCT_h*100*km/s/Mpc
FCT_Omega_M = box_setup['Cosmology']['Omega_M']
FCT_Omega_L = box_setup['Cosmology']['Omega_L']
FCT_DM_shell_edges = np.copy(DM_shell_edges)
FCT_shell_multipliers = np.copy(shell_multipliers)
FCT_init_xys = np.copy(init_xyz)
FCT_zeds = np.copy(z_int_steps)

### ==================================== ###
### Define all necessary functions here. ###
### ==================================== ###
# Defined in order of usage.

def halo_batch_indices(
    snap, mass_gauge, mass_range, 
    halo_type, halo_limit, fname, sim_dir, out_dir
):

    # ---------------------------------- #
    # Read in parameters of halo in sim. #
    # ---------------------------------- #

    props = h5py.File(f'{sim_dir}/subhalo_{snap}.properties')

    cNFW = props['cNFW_200crit'][:]  # NFW concentration.
    rvir = props['R_200crit'][:]*1e3 # Virial radius (to kpc with *1e3)
    m200c = props['Mass_200crit'][:] *1e10  # Crit. M_200 (to Msun with *1e10)
    m200c[m200c <= 0] = 1
    m200c = np.log10(m200c)  


    # -------------------------------------------------- #
    # Select a halo sample based on mass and mass range. #
    # -------------------------------------------------- #

    select_halos = np.where(
        (m200c >= mass_gauge-mass_range) & (m200c <= mass_gauge+mass_range)
    )[0]

    # Selecting subhalos or (host/main) halos.
    subtype = props["Structuretype"][:]
    if halo_type == 'subhalos':
        select = np.where(subtype[select_halos] > 10)[0]
        select_halos = select_halos[select]
    else:
        select = np.where(subtype[select_halos] == 10)[0]
        select_halos = select_halos[select]

    # Limit amount of halos to given size.
    halo_number = len(select_halos)
    if halo_number >= halo_limit:

        # Fix pseudo-random choice of halos.
        # np.random.seed(1)
        random.seed(1)
        
        # Select non-repeating indices for halos.
        # rand_IDs = np.random.randint(0, halo_number-1, size=(halo_limit))
        rand_IDs = random.sample(list(np.arange(halo_number)), halo_limit)
        select_halos = select_halos[rand_IDs]

    # Save cNFW, rvir and Mvir of halos in batch.
    halo_params = np.zeros((len(select_halos), 3))
    for j, halo_idx in enumerate(select_halos):
        halo_params[j, 0] = rvir[halo_idx]
        halo_params[j, 1] = m200c[halo_idx]
        halo_params[j, 2] = cNFW[halo_idx]

    np.save(f'{out_dir}/halo_batch_{fname}_indices.npy', select_halos)
    np.save(f'{out_dir}/halo_batch_{fname}_params.npy', halo_params)


def read_DM_halo_index(snap, halo_ID, fname, sim_dir, out_dir):

    # ---------------- #
    # Open data files. #
    # ---------------- #

    snaps = h5py.File(f'{sim_dir}/snapshot_{snap}.hdf5')
    group = h5py.File(f'{sim_dir}/subhalo_{snap}.catalog_groups')
    parts = h5py.File(f'{sim_dir}/subhalo_{snap}.catalog_particles')
    props = h5py.File(f'{sim_dir}/subhalo_{snap}.properties')

    # Positions.
    a = snaps["/Header"].attrs["Scale-factor"]
    pos = snaps['PartType1/Coordinates'][:][:] * a
    
    # Critical M_200.
    m200c = props['Mass_200crit'][:] * 1e10  # now in Msun
    m200c[m200c <= 0] = 1
    m200c = np.log10(m200c)  


    # ------------------------------------------------ #
    # Find DM particles gravitationally bound to halo. #
    # ------------------------------------------------ #

    halo_start_pos = group["Offset"][halo_ID]
    halo_end_pos = group["Offset"][halo_ID + 1]

    particle_ids_in_halo = parts["Particle_IDs"][halo_start_pos:halo_end_pos]
    particle_ids_from_snapshot = snaps["PartType1/ParticleIDs"][...]

    # Get indices of elements, which are present in both arrays.
    _, _, indices_p = np.intersect1d(
        particle_ids_in_halo, particle_ids_from_snapshot, 
        assume_unique=True, return_indices=True
    )


    # ------------------------------------------ #
    # Save DM positions centered on CoP of halo. #
    # ------------------------------------------ #

    CoP = np.zeros((len(m200c), 3))
    CoP[:, 0] = props["Xcminpot"][:]
    CoP[:, 1] = props["Ycminpot"][:]
    CoP[:, 2] = props["Zcminpot"][:]
    CoP_halo = CoP[halo_ID, :]

    DM_pos = pos[indices_p, :]  # x,y,z of each DM particle
    DM_pos -= CoP_halo
    DM_pos *= 1e3  # to kpc
    np.save(f'{out_dir}/DM_pos_{fname}.npy', DM_pos)

    # Save c.o.m. coord of all DM particles (used for outside_gravity fct.).
    DM_com_coord = np.sum(DM_pos, axis=0)/len(DM_pos)
    np.save(f'{out_dir}/DM_com_coord_{fname}.npy', DM_com_coord)


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


def check_grid(init_grid, DM_pos, parent_GRID_S, DM_lim):
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


def cell_division(
    init_grid, DM_pos, parent_GRID_S, DM_lim, stable_grid, out_dir, fname
    ):


    # Initiate counters.
    thresh = 1
    cell_division_count = 0

    DM_count_l = []
    cell_com_l = []
    cell_gen_l = []

    while thresh > 0:

        DM_count, cell_com, stable_cells, DM_parent_cells, thresh = check_grid(
            init_grid, DM_pos, parent_GRID_S, DM_lim
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
            sub8_raw = grid_3D(sub8_GRID_S, sub8_GRID_S)

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
    while mem_MB >= 0.95*core_mem_MB:
        mem_MB *= batches
        batches += 1
        mem_MB /= batches

    chunksize = math.ceil(cells/batches)

    return chunksize


def batch_generators_short_range(cell_coords, cell_gen, chunksize):

    cells = len(cell_coords)

    batches = math.ceil(cells/chunksize)
    batch_arr = np.arange(batches)

    cell_chunks = list(chunks(chunksize, cell_coords))
    cgen_chunks = list(chunks(chunksize, cell_gen))
    
    return batch_arr, cell_chunks, cgen_chunks


def cell_gravity_short_range(
    cell_coords_in, cell_gen, init_GRID_S,
    DM_pos, DM_lim, DM_sim_mass, smooth_l,
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
    del cell_gen, cell_len

    # Set DM outside cell to nan values.
    DM_pos_sync[~DM_in_cell_IDs] = np.nan

    # Save the DM IDs, such that we know which particles are in which cell.
    # This will be used in the long-range gravity calculations.
    DM_in_cell_IDs_compact = np.argwhere(DM_in_cell_IDs==True)
    DM_in_cell_IDs_compact[:,0] += (max_b_len*b_id)
    
    del DM_in_cell_IDs
    np.save(f'{out_dir}/batch{b_id}_DM_in_cell_IDs.npy', DM_in_cell_IDs_compact)
    del DM_in_cell_IDs_compact

    # Sort all nan values to the bottom of axis 1, i.e. the DM-in-cell-X axis 
    # and truncate array based on DM_lim parameter. This simple way works since 
    # each cell cannot have more than DM_lim (times the last shell multiplier).
    ind_2D = DM_pos_sync[:,:,0].argsort(axis=1)
    ind_3D = np.repeat(np.expand_dims(ind_2D, axis=2), 3, axis=2)
    DM_sort = np.take_along_axis(DM_pos_sync, ind_3D, axis=1)
    DM_in = DM_sort[:,:DM_lim*FCT_shell_multipliers[-1],:]
    
    # note: Visual to see, if DM_in is grouped into the right cells.
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ss = 5
    # for elem in (DM_in+cell_coords)/kpc:
    #     x, y, z = elem[:,0][::ss], elem[:,1][::ss], elem[:,2][::ss]
    #     ax.scatter(x,y,z, alpha=0.5, s=0.5)

    # ss = 100
    # x, y, z = DM_pos[...,0][::ss], DM_pos[...,1][::ss], DM_pos[...,2][::ss]
    # ax.scatter(x/kpc,y/kpc,z/kpc, alpha=0.1, c='blueviolet', marker='x', s=0.001)
    # ax.view_init(elev=90, azim=20)
    # plt.show()

    # note: Memory peaks here, due to these arrays:
    # print(DM_pos_sync.shape, ind_2D.shape, ind_3D.shape, DM_sort.shape, DM_in.shape)
    # mem_inc = gso(cell_coords)+gso(DM_pos_sync)+gso(ind_2D)+gso(ind_3D)+gso(DM_sort)+gso(DM_in)
    # print('MEM_PEAK:', mem_inc/1e6)
    del DM_pos_sync, ind_2D, ind_3D, DM_sort

    # Calculate distances of DM and adjust array dimensionally.
    DM_dis = np.expand_dims(np.sqrt(np.sum(DM_in**2, axis=2)), axis=2)

    # Offset DM positions by smoothening length of Camila's simulations.
    eps = smooth_l / 2.

    # Quotient in sum (see formula). Can contain nan values, thus the np.nansum 
    # for the derivative, s.t. these values don't contribute. We center DM on 
    # c.o.m. of cell C, so we only need DM_in in numerator.
    #? does DM_in also need to be shifted by eps?
    quot = (-DM_in)/np.power((DM_dis**2 + eps**2), 3./2.)
    
    # note: Minus sign, s.t. velocity changes correctly (see GoodNotes).
    derivative = -G*DM_sim_mass*np.nansum(quot, axis=1)    
    np.save(f'{out_dir}/batch{b_id}_short_range.npy', derivative)


def chunksize_long_range(cells, core_mem_MB):
    
    # note: mem_MB specific to peak memory usage in cell_gravity_long_range.
    # -> Peak memory after calculation of derivative.

    elem = 8                          # 8 bytes for standard np.float64
    mem_type1 = 3*elem                # for derivative
    mem_type2 = cells*3*elem          # for quot
    mem_type3 = cells*elem            # for DM_count_sync

    mem_MB = (mem_type1+mem_type2+mem_type3)/1.e6

    batches = 1
    while mem_MB >= 0.95*core_mem_MB:
        mem_MB *= batches
        batches += 1
        mem_MB /= batches

    chunksize = math.ceil(cells/batches)

    return chunksize
    

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

    dPsi_multipole_cells = G*DM_sim_mass*np.nansum(-term1+term2, axis=0)
    

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
    dPsi_monopole_cells = G*DM_sim_mass*np.nansum(
        DM_count_mono_sync*cellC_Jcoms_mono_2D/(cellC_dis_mono**3), axis=0)

    # Minus sign, s.t. velocity changes correctly (see GoodNotes).
    derivative_lr = -(dPsi_multipole_cells + dPsi_monopole_cells)
    
    np.save(f'{out_dir}/cell{c_id}_batch{b_id}_long_range.npy', derivative_lr)


def load_dPsi_long_range(c_id, batches, out_dir):

    # Load all batches for current cell.
    dPsi_raw = np.array(
        [np.load(f'{out_dir}/cell{c_id}_batch{b}_long_range.npy') for b in batches]
    )

    # Combine into one array by summing and save.
    dPsi_for_cell = np.sum(dPsi_raw, axis=0)
    np.save(f'{out_dir}/cell{c_id}_long_range.npy', dPsi_for_cell)  


def load_grid(root_dir, which, fname):

    if which == 'derivatives':
        grid = np.load(f'{root_dir}/dPsi_grid_{fname}.npy')

    elif which == 'positions':
        grid = np.load(f'{root_dir}/fin_grid_{fname}.npy')

    return grid


@nb.njit
def nu_in_which_cell(nu_coords, cell_coords):

    # For now, just subtract nu_coords from all cell_coords, then take min.
    dist = np.sqrt(np.sum((np.abs(cell_coords-nu_coords)**2), axis=2))
    cell_idx = dist.argmin()

    return cell_idx


def outside_gravity_quadrupole(x_i, com_halo, DM_sim_mass, DM_num, QJ_abs):

    ### ----------- ###
    ### Quadrupole. ###
    ### ----------- ###

    # Center neutrino on c.o.m. of halo and get distance.
    x_i -= com_halo
    r_i = np.sqrt(np.sum(x_i**2))
    
    # Permute order of coords by one, i.e. (x,y,z) -> (z,x,y).
    x_i_roll = np.roll(x_i, 1)

    # Terms appearing in the quadrupole term.
    QJ_aa = QJ_abs[0]
    QJ_ab = QJ_abs[1]

    # Factors of 2 are for the symmetry of QJ_ab elements.
    term1_aa = np.sum(QJ_aa*x_i, axis=0)
    term1_ab = np.sum(2*QJ_ab*x_i_roll, axis=0)
    term1 = (term1_aa+term1_ab)/r_i**5

    term2_pre = 5*x_i/(2*r_i**7)
    term2_aa = np.sum(QJ_aa*x_i**2, axis=0)
    term2_ab = np.sum(2*QJ_ab*x_i*x_i_roll, axis=0)
    term2 = term2_pre*(term2_aa+term2_ab)

    dPsi_multipole_cells = G*DM_sim_mass*(-term1+term2)


    ### --------- ###
    ### Monopole. ###
    ### --------- ###
    dPsi_monopole_cells = G*DM_num*DM_sim_mass*x_i/r_i**3

    # Minus sign, s.t. velocity changes correctly (see GoodNotes).
    derivative_lr = -(dPsi_multipole_cells + dPsi_monopole_cells)

    return derivative_lr


def number_densities_mass_range(
    sim_vels, nu_masses, out_file, pix_sr=4*Pi,
    average=False, m_start=0.01, z_start=0., sim_type='single_halos'
):
    
    # Convert velocities to momenta.
    p_arr, _ = velocity_to_momentum(sim_vels, nu_masses)

    if average:
        inds = np.array(np.where(FCT_zeds >= z_start)).flatten()
        temp = [
            number_density(p_arr[...,0], p_arr[...,k], pix_sr) for k in inds
        ]
        num_densities = np.mean(np.array(temp.T), axis=-1)
    else:
        num_densities = number_density(p_arr[...,0], p_arr[...,-1], pix_sr)

    if sim_type == 'all_sky':
        return num_densities
    else:
        np.save(f'{out_file}', num_densities)



# Make temporary folder to store files, s.t. parallel runs don't clash.
rand_code = ''.join(
    random.choices(string.ascii_uppercase + string.digits, k=4)
)
temp_dir = f'{args.directory}/temp_data_{rand_code}'
os.makedirs(temp_dir)


hname = f'1e+{args.mass_gauge}_pm{args.mass_range}Msun'
halo_batch_indices(
    z0_snap_4cif, float(args.mass_gauge), float(args.mass_range), 'halos', int(args.halo_num), 
    hname, box_file_dir, args.directory
)
halo_batch_IDs = np.load(f'{args.directory}/halo_batch_{hname}_indices.npy')
halo_batch_params = np.load(f'{args.directory}/halo_batch_{hname}_params.npy')
halo_num = len(halo_batch_params)

print(f'********Numerical Simulation: Mode={args.sim_type}********')
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
    z = np.interp(s_val, s_int_steps, z_int_steps)

    # Snapshot specific parameters.
    idx = np.abs(zeds_snaps - z).argmin()
    snap = nums_snaps[idx]
    snap_GRID_L = snaps_GRID_L[idx]

    # Neutrino inside cell grid.
    if np.all(np.abs(x_i)) <= snap_GRID_L:

        # Find which (pre-calculated) derivative grid to use at current z.
        simname = f'origID{halo_ID}_snap_{snap}'
        dPsi_grid = load_grid(temp_dir, 'derivatives', simname)
        cell_grid = load_grid(temp_dir, 'positions',   simname)

        cell_idx = nu_in_which_cell(x_i, cell_grid)  # index of cell
        grad_tot = dPsi_grid[cell_idx,:]             # derivative of cell

    # Neutrino outside cell grid.
    else:
        # NrDM = snaps_DM_num[idx]
        # grad_tot = fct.outside_gravity(x_i, NrDM, DM_mass)

        # With quadrupole.
        DM_com = snaps_DM_com[idx]
        DM_num = snaps_DM_num[idx]
        QJ_abs = snaps_QJ_abs[idx]
        grad_tot = outside_gravity_quadrupole(
            x_i, DM_com, DM_mass, DM_num, QJ_abs
        )

    # Switch to "physical reality" here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)

    # Hamilton eqns. for integration (global minus, s.t. we go back in time).
    dyds = -np.array([
        u_i, 1./(1.+z)**2 * grad_tot
    ])

    return dyds


def backtrack_1_neutrino(y0_Nr):
    """Simulate trajectory of 1 neutrino."""

    # Split input into initial vector and neutrino number.
    y0, Nr = y0_Nr[0:-1], y0_Nr[-1]

    # Solve all 6 EOMs.
    sol = solve_ivp(
        fun=EOMs, t_span=[s_int_steps[0], s_int_steps[-1]], t_eval=s_int_steps,
        y0=y0, method=integration_solver, vectorized=True,
        args=()
        )
    
    np.save(f'{temp_dir}/nu_{int(Nr)}.npy', np.array(sol.y.T))


for halo_j, halo_ID in enumerate(halo_batch_IDs):

    # ============================================== #
    # Run precalculations for current halo in batch. #
    # ============================================== #

    # Generate progenitor index array for current halo.
    with h5py.File(f'{args.directory}/MergerTree.hdf5') as tree:
        prog_IDs = tree['Assembly_history/Progenitor_index'][halo_ID,:]
        prog_IDs_np = np.array(np.expand_dims(prog_IDs, axis=1), dtype=int)


    # Create empty arrays to save specifics of each loop.
    save_GRID_L = np.zeros(len(nums_snaps))
    save_num_DM = np.zeros(len(nums_snaps))
    save_DM_com = []
    save_QJ_abs = []


    # Generate the gravity grids from the earliest snapshot to the latest, i.e. 
    # from z=4 to z=0 in our case.
    for j, (snap, prog_ID) in enumerate(
        zip(nums_snaps, prog_IDs_np[::-1])
    ):
        print(f'halo {halo_j+1}/{halo_num} ; snapshot {snap}')
        prog_ID = int(prog_ID)


        # --------------------------- #
        # Read and load DM positions. #
        # --------------------------- #

        # Name for file identification of current halo for current snapshot.
        IDname = f'origID{halo_ID}_snap_{snap}'

        if args.sim_type in ('single_halos', 'all_sky'):
            
            read_DM_halo_index(
                snap, prog_ID, IDname, box_file_dir, temp_dir
            )
            DM_raw = np.load(f'{temp_dir}/DM_pos_{IDname}.npy')
            DM_com = np.load(f'{temp_dir}/DM_com_coord_{IDname}.npy')*kpc
            DM_particles = len(DM_raw)

        else:

            # Define how many shells are used, out of len(DM_SHELL_EDGES)-1.
            shells = 1
            DM_shell_edges = DM_shell_edges[:shells+1]
            
            # Load DM from all used shells.
            DM_pre = []
            for shell_i in range(shells):
                DM_pre.append(
                    np.load(f'{temp_dir}/DM_pos_{IDname}_shell{shell_i}.npy')
                )
            DM_raw = np.array(list(chain.from_iterable(DM_pre)))
            DM_particles = len(DM_raw)
            DM_com = np.sum(DM_raw, axis=0)/len(DM_raw)
            del DM_pre


        # ---------------------- #
        # Cell division process. #
        # ---------------------- #

        # Initialize grid.
        snap_GRID_L = (int(np.abs(DM_raw).max()) + 1)*kpc
        raw_grid = grid_3D(snap_GRID_L, snap_GRID_L)
        init_grid = np.expand_dims(raw_grid, axis=1)

        # Prepare arrays for cell division.
        DM_raw *= kpc
        DM_pos = np.expand_dims(DM_raw, axis=0)
        DM_pos_for_cell_division = np.repeat(DM_pos, len(init_grid), axis=0)

        
        ### Interlude: Calculate QJ_aa and QJ_ab for complete halo. ###
        
        # Center all DM particles of halo on c.o.m. of halo and get distances.
        DM_raw -= DM_com
        DM_raw_dis = np.expand_dims(np.sqrt(np.sum(DM_raw**2, axis=1)), axis=1)

        # Permute order of coords by one, i.e. (x,y,z) -> (z,x,y).
        DM_raw_roll = np.roll(DM_raw, 1)

        # Terms appearing in the quadrupole term.
        QJ_aa = np.sum(3*DM_raw**2 - DM_raw_dis**2, axis=0)
        QJ_ab = np.sum(3*DM_raw*DM_raw_roll, axis=0)
        del DM_raw
        save_QJ_abs.append(np.array([QJ_aa, QJ_ab]))


        # Cell division.
        cell_division_count = cell_division(
            init_grid, DM_pos_for_cell_division, snap_GRID_L, DM_lim, None, temp_dir, IDname
        )
        del DM_pos_for_cell_division

        # Load files from cell division.
        fin_grid = np.load(f'{temp_dir}/fin_grid_{IDname}.npy')
        DM_count = np.load(f'{temp_dir}/DM_count_{IDname}.npy')
        cell_com = np.load(f'{temp_dir}/cell_com_{IDname}.npy')
        cell_gen = np.load(f'{temp_dir}/cell_gen_{IDname}.npy')
        
        # Save snapshot specific parameters.
        save_GRID_L[j] = snap_GRID_L
        save_num_DM[j] = np.sum(DM_count)
        save_DM_com.append(DM_com)


        # --------------------------------------------- #
        # Calculate gravity grid (in batches of cells). #
        # --------------------------------------------- #
        cell_coords = np.squeeze(fin_grid, axis=1)
        cells = len(cell_coords)


        # -------------------- #
        # Short-range gravity. #
        # -------------------- #

        # Calculate available memory per core.
        mem_so_far = (psutil.virtual_memory().used - OS_MEM)/MB_UNIT
        mem_left = mem_lim_GB*1e3 - mem_so_far
        core_mem_MB = mem_left / CPUs_pre

        # Determine short-range chuncksize based on available memory and cells.
        chunksize_sr = chunksize_short_range(
            cells, DM_particles, DM_lim*shell_multipliers[-1], core_mem_MB
        )

        # Split workload into batches (if necessary).
        batch_arr, cell_chunks, cgen_chunks = batch_generators_short_range(
            cell_coords, cell_gen, chunksize_sr
        )

        with ProcessPoolExecutor(CPUs_pre) as ex:
            ex.map(
                cell_gravity_short_range, 
                cell_chunks, cgen_chunks, repeat(snap_GRID_L), 
                repeat(DM_pos), repeat(DM_lim), repeat(DM_mass), 
                repeat(Smooth_L), repeat(temp_dir), batch_arr, 
                repeat(chunksize_sr)
            )

        # Combine short-range batch files.
        dPsi_short_range_batches = [
            np.load(f'{temp_dir}/batch{b}_short_range.npy') for b in batch_arr
        ]
        dPsi_short_range = np.array(
            list(chain.from_iterable(dPsi_short_range_batches))
        )

        # Combine DM_in_cell_IDs batches (needed for long-range gravity).
        DM_in_cell_IDs_l = []
        for b_id in batch_arr:
            DM_in_cell_IDs_l.append(
                np.load(f'{temp_dir}/batch{b_id}_DM_in_cell_IDs.npy')
            )
        DM_in_cell_IDs_np = np.array(
            list(chain.from_iterable(DM_in_cell_IDs_l)))
        np.save(f'{temp_dir}/DM_in_cell_IDs_{IDname}.npy', DM_in_cell_IDs_np)
        

        # ------------------- #
        # Long-range gravity. #
        # ------------------- #
        
        # Calculate available memory per core.
        mem_so_far = (psutil.virtual_memory().used - OS_MEM)/MB_UNIT
        mem_left = mem_lim_GB*1e3 - mem_so_far
        core_mem_MB = mem_left / CPUs_pre

        # Determine long-range chuncksize based on available memory and cells.
        # chunksize_lr = chunksize_long_range(cells, core_mem_MB)
        chunksize_lr = 501

        # Split workload into batches (if necessary).
        DM_in_cell_IDs = np.load(f'{temp_dir}/DM_in_cell_IDs_{IDname}.npy')
        batch_IDs, cellC_rep, cellC_cc, gen_rep, cib_IDs_gens, count_gens, com_gens, gen_gens = batch_generators_long_range(
            cell_coords, cell_com, cell_gen, DM_count, chunksize_lr
        )

        with ProcessPoolExecutor(CPUs_pre) as ex:
            ex.map(
                cell_gravity_long_range_quadrupole, 
                cellC_rep, cib_IDs_gens, batch_IDs, 
                cellC_cc, com_gens, gen_gens, repeat(snap_GRID_L),
                repeat(np.squeeze(DM_pos, axis=0)), count_gens, 
                repeat(DM_in_cell_IDs), repeat(DM_mass), 
                repeat(temp_dir), repeat(chunksize_lr), gen_rep
            )

        # Combine long-range batch files.
        c_labels = np.unique(cellC_rep)
        b_labels = np.unique(batch_IDs)
        with ProcessPoolExecutor(CPUs_pre) as ex:
            ex.map(
                load_dPsi_long_range, c_labels, 
                repeat(b_labels), repeat(temp_dir)
            )

        dPsi_long_range = np.array(
            [np.load(f'{temp_dir}/cell{c}_long_range.npy') for c in c_labels])

        # Combine short- and long-range forces.
        dPsi_grid = dPsi_short_range + dPsi_long_range
        np.save(f'{temp_dir}/dPsi_grid_{IDname}.npy', dPsi_grid)


    # ========================================= #
    # Run simulation for current halo in batch. #
    # ========================================= #

    #! Important:
    # The loop ran from the earliest snapshot (z~4 for us) to the latest (z=0).
    # So these arrays are in this order. Even though our simulation runs 
    # backwards in time, we can leave them like this, since the correct element 
    # gets picked with the idx routine in the EOMs function above.
    snaps_GRID_L = np.flip(np.array(save_GRID_L), axis=0)
    snaps_DM_num = np.flip(np.array(save_num_DM), axis=0)
    snaps_DM_com = np.flip(np.array(save_DM_com), axis=0)
    snaps_QJ_abs = np.flip(np.array(save_QJ_abs), axis=0)

    # Display parameters for simulation.
    print(f'***Running simulation: mode = {args.sim_type}***')
    print(f'halo={halo_j+1}/{halo_num}, CPUs={CPUs_sim}')

    sim_start = time.perf_counter()

    if args.sim_type in ('single_halos', 'spheres'):
    
        # Load initial velocities.
        ui = np.load(f'{args.directory}/initial_velocities.npy')

        # Combine vectors and append neutrino particle number.
        y0_Nr = np.array(
            [np.concatenate((init_xyz, ui[i], [i+1])) for i in range(neutrinos)]
            )

        # Run simulation on multiple cores.
        with ProcessPoolExecutor(CPUs_sim) as ex:
            ex.map(backtrack_1_neutrino, y0_Nr)

        # Compactify all neutrino vectors into 1 file.
        neutrino_vectors = np.array(
            [np.load(f'{temp_dir}/nu_{i+1}.npy') for i in range(neutrinos)]
        )

        # For these modes (i.e. not all_sky), save all neutrino vectors.
        # Split velocities and positions into 10k neutrino batches.
        # For reference: ndarray with shape (10_000, 100, 6) is  48 MB.
        batches = math.ceil(neutrinos/10_000)
        split = np.array_split(neutrino_vectors, batches, axis=0)
        for i, elem in enumerate(split):
            np.save(
                f'{args.directory}/neutrino_vectors_numerical_batch{i+1}.npy', elem
            )

        # Compute the number densities.
        out_file = f'{args.directory}/number_densities_numerical.npy'
        number_densities_mass_range(
            neutrino_vectors[:,:,3:6], neutrino_massrange, out_file
        )

    else:

        # Load initial velocities for all_sky mode. Note that this array is not 
        # github compatible, and will be deleted afterwards.
        ui = np.load(f'{args.directory}/initial_velocities.npy')

        # Empty list to append number densitites of each angle coord. pair.
        number_densities_pairs = []

        for i, ui_elem in enumerate(ui):

            print(f'Coord. pair {i+1}/{len(ui)}')

            # Combine vectors and append neutrino particle number.
            y0_Nr = np.array([np.concatenate(
                (init_xyz, ui_elem[k], [k+1])) for k in range(len(ui_elem))
            ])
            
            # Run simulation on multiple cores.
            with ProcessPoolExecutor(CPUs_sim) as ex:
                ex.map(backtrack_1_neutrino, y0_Nr)

            # Compactify all neutrino vectors into 1 file.
            neutrino_vectors = np.array(
                [np.load(f'{temp_dir}/nu_{i+1}.npy') for i in range(neutrinos)]
            )

            # Compute the number densities.
            number_densities_pairs.append(
                number_densities_mass_range(
                    neutrino_vectors[:,:,3:6], 
                    neutrino_massrange, 
                    sim_type=args.sim_type
                )
            )


        # Combine number densities with angle pairs: First 2 entries are angles.
        nu_dens_pairs = np.array(number_densities_pairs)
        angle_pairs = np.load(f'{args.directory}/all_sky_angles.npy')
        nu_final = np.concatenate((angle_pairs, nu_dens_pairs), axis=2)
        np.save(f'{args.directory}/number_densities_numerical.npy', nu_final)

        # Delete arrays not compatible with github file limit size.
        delete_temp_data(f'{args.directory}/initial_velocities.npy')


    sim_time = time.perf_counter()-sim_start
    print(f'Sim time: {sim_time/60.} min, {sim_time/(60**2)} h.')

# Remove temporary folder with all individual neutrino files.
shutil.rmtree(temp_dir)

total_time = time.perf_counter()-total_start
print(f'Total time: {total_time/60.} min, {total_time/(60**2)} h.')