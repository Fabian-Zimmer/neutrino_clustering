from shared.preface import *


###########################################
### Functions used in SMOOTH simulation ###
###########################################
# region

def NFW_profile(r, rho_0, r_s):
    """NFW density profile.

    Args:
        r (array): radius from center
        rho_0 (array): normalisation 
        r_s (array): scale radius

    Returns:
        array: density at radius r
    """    

    rho = rho_0 / (r/r_s) / np.power(1.+(r/r_s), 2.)

    return rho


def scale_density_NFW(z, c):
    """Eqn. (2) from arXiv:1302.0288. c=r_200/r_s."""
    numer = 200 * c**3
    denom = 3 * (np.log(1+c) - (c/(1+c)))
    delta_c = numer/denom

    rho_c = rho_crit(z)

    return rho_c*delta_c


@nb.njit
def c_vir_avg(z, M_vir):
    """Intermediate/helper function for c_vir function below."""

    # Functions for eqn. (5.5) in Mertsch et al. (2020), from their Ref. [40].
    a_of_z = 0.537 + (0.488)*np.exp(-0.718*np.power(z, 1.08))
    b_of_z = -0.097 + 0.024*z

    # Calculate average c_vir.
    arg_in_log = (M_vir / (1.e12 / h * Msun))
    c_vir_avg = np.power(a_of_z + b_of_z*np.log10(arg_in_log), 10.)

    return np.float64(c_vir_avg)
    

@nb.njit
def c_vir(z, M_vir, R_vir, R_s):
    """Concentration parameter defined as r_vir/r_s, i.e. the ratio of virial 
    radius to the scale radius of the halo according to eqn. 5.5 of 
    Mertsch et al. (2020). 

    Args:
        z (array): redshift
        M_vir (float): virial mass, treated as fixed in time

    Returns:
        array: concentration parameters at each given redshift [dimensionless]
    """

    # The "beta" in eqn. (5.5) is obtained from c_vir_avg(0, M_vir)
    # and c_vir(0, M_vir) (c0_vir variable below) and the values in Table 1.
    # (See Methods section of Zhang & Zhang (2018), but their beta is different)
    c0_vir = R_vir / R_s 
    beta = c0_vir / c_vir_avg(0, M_vir)

    c = beta * c_vir_avg(z, M_vir)

    return np.float64(c)


@nb.njit
def rho_crit(z):
    """Critical density of the universe as a function of redshift, assuming
    matter domination, only Omega_m and Omega_Lambda in Friedmann equation. See 
    notes for derivation.

    Args:
        z (array): redshift

    Returns:
        array: critical density at redshift z
    """    
    
    H_squared = H0**2 * (Omega_M*(1.+z)**3 + Omega_L) 
    rho_crit = 3.*H_squared / (8.*Pi*G)

    return np.float64(rho_crit)


@nb.njit
def Omega_M_z(z):
    """Matter density parameter as a function of redshift, assuming matter
    domination, only Omega_M and Omega_L in Friedmann equation. See notes
    for derivation.

    Args:
        z (array): redshift

    Returns:
        array: matter density parameter at redshift z [dimensionless]
    """    

    Omega_M_of_z = (Omega_M*(1.+z)**3) / (Omega_M*(1.+z)**3 + Omega_L)

    return np.float64(Omega_M_of_z)


@nb.njit
def Delta_vir(z):
    """Function as needed for their eqn. (5.7).

    Args:
        z (array): redshift

    Returns:
        array: value as specified just beneath eqn. (5.7) [dimensionless]
    """    

    Delta_vir = 18.*Pi**2 + 82.*(Omega_M_z(z)-1.) - 39.*(Omega_M_z(z)-1.)**2

    return np.float64(Delta_vir)


@nb.njit
def R_vir_fct(z, M_vir):
    """Virial radius according to eqn. 5.7 in Mertsch et al. (2020).

    Args:
        z (array): redshift
        M_vir (float): virial mass

    Returns:
        array: virial radius
    """

    R_vir = np.power(3.*M_vir / (4.*Pi*Delta_vir(z)*rho_crit(z)), 1./3.)

    return np.float64(R_vir)
# endregion


#############################################
### Functions used in DISCRETE simulation ###
#############################################


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


def halo_DM(halo_idx, snap, pos, snap_Particle_IDs, sim_dir, out_dir):

    # Open data files.
    group = h5py.File(f'{sim_dir}/subhalo_{snap}.catalog_groups')
    parts = h5py.File(f'{sim_dir}/subhalo_{snap}.catalog_particles')

    # Start and stop index for current halo.
    halo_init = group["Offset"][halo_idx]
    halo_stop = group["Offset"][halo_idx + 1]

    # Particle IDs of halo and snapshot.
    halo_Particle_IDs = parts["Particle_IDs"][halo_init:halo_stop]

    # Particle IDs present in both above arrays.
    _, _, indices_p = np.intersect1d(
        halo_Particle_IDs, snap_Particle_IDs, 
        assume_unique=True, return_indices=True
    )

    # Save DM positions.
    DM_pos = pos[indices_p, :]  # x,y,z of each DM particle
    np.save(f'{out_dir}/DM_of_haloID{halo_idx}.npy', DM_pos)


def read_DM_halos_inRange(
    snap, halo_ID, DM_range, halo_limit, fname, sim_dir, out_dir, CPUs
):

    # --------------- #
    # Initialize data #
    # --------------- #

    snaps = h5py.File(f'{sim_dir}/snapshot_{snap}.hdf5')
    props = h5py.File(f'{sim_dir}/subhalo_{snap}.properties')

    # Positions.
    a = snaps["/Header"].attrs["Scale-factor"]
    pos = snaps['PartType1/Coordinates'][:][:] * a
    
    # DM_range from physical to comoving.
    DM_range *= (a/kpc/1e3)

    # Masses of all halos in sim.
    m200c = props['Mass_200crit'][:]

    # Center of Potential coordinates, for all halos.
    CoP = np.zeros((len(m200c), 3))
    del m200c
    CoP[:, 0] = props["Xcminpot"][:]
    CoP[:, 1] = props["Ycminpot"][:]
    CoP[:, 2] = props["Zcminpot"][:]
    CoP_halo = CoP[halo_ID, :]


    # --------------------------------- #
    # Combine DM of all halos in range. #
    # --------------------------------- #

    CoP_cent = CoP - CoP_halo
    halo_dis = np.sqrt(np.sum(CoP_cent**2, axis=1))
    select_halos = np.where(halo_dis <= DM_range)[0]

    # Limit amount of halos in range, select by mass.
    if halo_limit is not None:
        halo_number = len(select_halos)
        if halo_number >= halo_limit:
            lim_IDs = select_halos[:halo_limit]
            fin_IDs = np.delete(lim_IDs, np.s_[lim_IDs==halo_ID])

    halo_IDs = np.insert(fin_IDs, 0, halo_ID)

    # Arrays to only load once.
    snap_Particle_IDs = snaps["PartType1/ParticleIDs"][...]

    with Pool(CPUs) as pool:
        pool.starmap(halo_DM, zip(
            halo_IDs,
            repeat(snap), repeat(pos), repeat(snap_Particle_IDs),
            repeat(sim_dir), repeat(out_dir)
        ))
        
    # Combine DM from all halos into 1 file.
    DM_halos = [np.load(f'{out_dir}/DM_of_haloID{i}.npy') for i in halo_IDs]

    # note: This I needed for plotting I think...maybe still useful
    DM_lengths = np.zeros(len(DM_halos))
    for i, DM_elem in enumerate(DM_halos):
        DM_lengths[i] = len(DM_elem)
    np.save(f'{out_dir}/DM_lengths_{fname}.npy', DM_lengths)

    # Combine all DM particles from selected halos into one file.
    DM_total = np.concatenate(DM_halos, axis=0)
    DM_total -= CoP_halo
    DM_total *= 1e3
    np.save(f'{out_dir}/DM_pos_{fname}.npy', DM_total) 
    delete_temp_data(f'{out_dir}/DM_of_haloID*.npy')

    # Save c.o.m. coord of all DM particles (used for outside_gravity fct.).
    DM_com_coord = np.sum(DM_total, axis=0)/len(DM_total)
    np.save(f'{out_dir}/DM_com_coord_{fname}.npy', DM_com_coord)


def read_DM_all_inRange(
    snap, halo_ID, DM_shell_edges, fname, sim_dir, out_dir
):

    # --------------- #
    # Initialize data #
    # --------------- #

    snaps = h5py.File(f'{sim_dir}/snapshot_{snap}.hdf5')
    props = h5py.File(f'{sim_dir}/subhalo_{snap}.properties')

    # Positions.
    a = snaps["/Header"].attrs["Scale-factor"]
    pos = snaps['PartType1/Coordinates'][:][:] * a

    # Masses of all halos in sim.
    m200c = props['Mass_200crit'][:]

    # Center of Potential coordinates, for all halos.
    CoP = np.zeros((len(m200c), 3))
    del m200c
    CoP[:, 0] = props["Xcminpot"][:]
    CoP[:, 1] = props["Ycminpot"][:]
    CoP[:, 2] = props["Zcminpot"][:]
    CoP_halo = CoP[halo_ID, :]
    del CoP

    # Center all DM on selected halo and calculate distance from halo center.
    pos -= CoP_halo
    DM_dis = np.sqrt(np.sum(pos**2, axis=1))

    # ----------------------------------- #
    # Save DM in spherical shell batches. #
    # ----------------------------------- #

    # DM_shell_edges from physical to comoving.
    DM_shell_edges_com = DM_shell_edges*(a/kpc/1e3)

    for i, (shell_start, shell_end) in enumerate(
        zip(DM_shell_edges_com[:-1], DM_shell_edges_com[1:])
    ):

        DM_pos = pos[(shell_start < DM_dis) & (DM_dis <= shell_end), :]*1e3
        np.save(f'{out_dir}/DM_pos_{fname}_shell{i}.npy', DM_pos)


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
    shell_cents = (DM_SHELL_EDGES[:-1] + DM_SHELL_EDGES[1:])/2.
    shells_sync = np.repeat(np.expand_dims(shell_cents, axis=0), cells, axis=0)

    # Find shell center, which each cell is closest to.
    which_shell = np.abs(grid_dis - shells_sync).argmin(axis=1)

    # Final DM limit for each cell.
    cell_DMlims = np.array([SHELL_MULTIPLIERS[k] for k in which_shell])*DM_lim


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


def halo_to_grid(DM_pos_kpc, DM_lim, fname, out_dir):

    # ---------------------- #
    # Cell division process. #
    # ---------------------- #

    # Initialize grid.
    snap_GRID_L = (int(np.abs(DM_pos_kpc).max()) + 1)*kpc
    raw_grid = grid_3D(snap_GRID_L, snap_GRID_L)
    init_grid = np.expand_dims(raw_grid, axis=1)

    # Prepare arrays for cell division.
    DM_pos = np.expand_dims(DM_pos_kpc*kpc, axis=0)
    DM_pos_for_cell_division = np.repeat(DM_pos, len(init_grid), axis=0)
    del DM_pos

    # Cell division.
    cell_division_count = cell_division(
        init_grid, DM_pos_for_cell_division, snap_GRID_L, DM_lim, None, out_dir, fname
    )
    del DM_pos_for_cell_division

    # Load files from cell division and return output.
    cell_ccs = np.squeeze(np.load(f'{out_dir}/fin_grid_{fname}.npy'), axis=1)
    DM_count = np.load(f'{out_dir}/DM_count_{fname}.npy')
    cell_com = np.load(f'{out_dir}/cell_com_{fname}.npy')
    cell_gen = np.load(f'{out_dir}/cell_gen_{fname}.npy')

    return cell_ccs, DM_count, cell_com, cell_gen, snap_GRID_L


@nb.njit
def outside_gravity(x_i, DM_count_tot, DM_sim_mass):
    pre = G*DM_count_tot*DM_sim_mass
    denom = np.sqrt(np.sum(x_i**2))**3

    return -pre*x_i/denom



@nb.njit
def outside_gravity_com(x_i, com_DM, DM_tot, DM_sim_mass):
    pre = G*DM_tot*DM_sim_mass
    denom = np.sqrt(np.sum((x_i-com_DM)**2))**3

    return -pre*(x_i-com_DM)/denom


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
    DM_in = DM_sort[:,:DM_lim*SHELL_MULTIPLIERS[-1],:]
    
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


def load_dPsi_long_range(c_id, batches, out_dir):

    # Load all batches for current cell.
    dPsi_raw = np.array(
        [np.load(f'{out_dir}/cell{c_id}_batch{b}_long_range.npy') for b in batches]
    )

    # Combine into one array by summing and save.
    dPsi_for_cell = np.sum(dPsi_raw, axis=0)
    np.save(f'{out_dir}/cell{c_id}_long_range.npy', dPsi_for_cell)  


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


#########################
### Utility functions ###
#########################

def delete_temp_data(path_to_wildcard_files):

    temp_files = glob.glob(path_to_wildcard_files, recursive=True)

    for f in temp_files:
        try:
            os.remove(f)
        except OSError:
            print("Error while deleting file (file not found")


def load_sim_data(out_dir, fname, which):
    """Loads neutrino positions or velocities of simulation."""

    sim = np.load(f'{out_dir}/{fname}.npy')

    # Return positions or velocities of neutrinos.
    if which == 'positions':
        data = sim[:,:,0:3]
    elif which == 'velocities':
        data = sim[:,:,3:6]

    # data.shape = (nr_of_nus, len(ZEDS), 3) ; u_all.ndim = 3
    return data


def velocity_to_momentum(sim_vels, m_arr):
    """
    Converts velocities (x,y,z from simulation) to 
    magnitude of momentum [eV] and ratio y=p/T_nu, according to desired
    target mass (and mass used in simulation).
    """

    # Magnitudes of velocity.
    mags_sim = np.sqrt(np.sum(sim_vels**2, axis=-1))
    mags_dim = np.repeat(np.expand_dims(mags_sim, axis=0), len(m_arr), axis=0)

    # Adjust neutrino target mass array dimensionally.
    m_dim = np.expand_dims(
        np.repeat(
            np.expand_dims(m_arr, axis=1), mags_dim.shape[1], axis=1),
        axis=2
    )

    # From velocity (magnitude) in kpc/s to momentum in eV.
    # p_dim = 1/np.sqrt(1-mags_dim**2) * mags_dim*(kpc/s) * m_dim
    p_dim = mags_dim*(kpc/s) * m_dim

    # p/T_CNB ratio.
    y = p_dim/T_CNB

    return p_dim, y


def y_fmt(value, tick_number):

    if value == 1e-3:
        return r'1+$10^{-3}$'
    elif value == 1e-2:
        return r'1+$10^{-2}$'
    elif value == 1e-1:
        return r'1+$10^{-1}$'
    elif value == 1e0:
        return r'1+$10^0$'
    elif value == 1e1:
        return r'1+$10^1$'



######################
### Main functions ###
######################


@nb.njit
def halo_pos(glat, glon, d):
    """Calculates halo position in x,y,z coords. from gal. lat. b and lon. l,
    relative to positon of earth (i.e. sun) at (x,y,z) = (8.5,0,0) [kpc].
    See notes for derivation."""

    # Galactic latitude and longitude in radian.
    b = np.deg2rad(glat)
    l = np.deg2rad(glon)

    # Relative z-axis coordinate.
    z_rel = np.sin(b)*d

    # Relative x-axis and y-axis coordinates.
    if l in (0., Pi):
        x_rel = np.sqrt(d**2-z_rel**2)
        y_rel = 0.
    else:

        # Angle between l and x-axis.
        # Both the x-axis and y-axis coord. are based on this angle.
        if 0. < l < Pi:
            alpha = np.abs(l-(Pi/2.))
        elif Pi < l < 2.*Pi:
            alpha = np.abs(l-(3.*Pi/2.))

        x_rel = np.sqrt( (d**2-z_rel**2) / (1+np.tan(alpha)**2) )
        y_rel = np.sqrt( (d**2-z_rel**2) / (1+1/np.tan(alpha)**2) )


    # x,y,z coords. w.r.t. GC (instead of sun), treating each case seperately.
    x_sun, y_sun, z_sun = X_SUN[0], X_SUN[1], X_SUN[2]
    z_GC = z_sun + z_rel

    if l in (0., 2.*Pi):
        x_GC = x_sun - x_rel
        y_GC = y_sun
    if 0. < l < Pi/2.:
        x_GC = x_sun - x_rel
        y_GC = y_sun - y_rel
    if l == Pi/2.:
        x_GC = x_sun
        y_GC = y_sun - y_rel
    if Pi/2. < l < Pi:
        x_GC = x_sun + x_rel
        y_GC = y_sun - y_rel
    if l == Pi:
        x_GC = x_sun + x_rel
        y_GC = y_sun
    if Pi < l < 3.*Pi/2.:
        x_GC = x_sun + x_rel
        y_GC = y_sun + y_rel
    if l == 3.*Pi/2.:
        x_GC = x_sun
        y_GC = y_sun + y_rel
    if 3.*Pi/2. < l < 2.*Pi:
        x_GC = x_sun - x_rel
        y_GC = y_sun + y_rel

    return np.asarray([x_GC, y_GC, z_GC], dtype=np.float64)


@nb.njit
def dPsi_dxi_NFW(x_i, z, rho_0, M_vir, R_vir, R_s, halo:str):
    """Derivative of NFW gravity of a halo w.r.t. any axis x_i.

    Args:
        x_i (array): spatial position vector
        z (array): redshift
        rho_0 (float): normalization
        M_vir (float): virial mass

    Returns:
        array: Derivative vector of grav. potential. for all 3 spatial coords.
               with units of acceleration.
    """    

    if halo in ('MW', 'VC', 'AG'):
        # Compute values dependent on redshift.
        r_vir = R_vir_fct(z, M_vir)
        c_NFW = c_vir(z, M_vir, R_vir, R_s)
        r_s = r_vir / c_NFW
        f_NFW = np.log(1+c_NFW) - (c_NFW/(1+c_NFW))
        rho_0_NEW = M_vir / (4*Pi*r_s**3*f_NFW)
    else:
        # If function is used to calculate NFW gravity for arbitrary halo.
        r_vir = R_vir
        r_s = R_s
        x_i_cent = x_i
        r = np.sqrt(np.sum(x_i_cent**2))
        rho_0_NEW = rho_0
        

    # Distance from respective halo center with current coords. x_i.
    if halo == 'MW':
        x_i_cent = x_i  # x_i - X_GC, but GC is coord. center, i.e. [0,0,0].
        r = np.sqrt(np.sum(x_i_cent**2))
    elif halo == 'VC':
        x_i_cent = x_i - (X_VC*kpc)  # center w.r.t. Virgo Cluster
        r = np.sqrt(np.sum(x_i_cent**2))
    elif halo == 'AG':
        x_i_cent = x_i - (X_AG*kpc)  # center w.r.t. Andromeda Galaxy
        r = np.sqrt(np.sum(x_i_cent**2))
    
    # Derivative in compact notation with m and M.
    m = np.minimum(r, r_vir)
    M = np.maximum(r, r_vir)
    # prefactor = 4.*Pi*G*rho_0*r_s**2*x_i_cent/r**2
    prefactor = 4.*Pi*G*rho_0_NEW*r_s**2*x_i_cent/r**2
    term1 = np.log(1. + (m/r_s)) / (r/r_s)
    term2 = (r_vir/M) / (1. + (m/r_s))
    derivative = prefactor * (term1 - term2)

    #NOTE: Minus sign, s.t. velocity changes correctly (see GoodNotes).
    return np.asarray(-derivative, dtype=np.float64)


@nb.njit
def grav_pot(x_i, z, rho_0, M_vir, R_vir, R_s, halo:str):
    
    if halo in ('MW', 'VC', 'AG'):
        # Compute values dependent on redshift.
        r_vir = R_vir_fct(z, M_vir)
        r_s = r_vir / c_vir(z, M_vir, R_vir, R_s)
    else:
        # If function is used to calculate NFW gravity for arbitrary halo.
        r_vir = R_vir
        r_s = R_s
        x_i_cent = x_i
        r = np.sqrt(np.sum(x_i_cent**2))

    # Distance from respective halo center with current coords. x_i.
    if halo == 'MW':
        x_i_cent = x_i  # x_i - X_GC, but GC is coord. center, i.e. [0,0,0].
        r = np.sqrt(np.sum(x_i_cent**2))
    elif halo == 'VC':
        x_i_cent = x_i - (X_VC*kpc)  # center w.r.t. Virgo Cluster
        r = np.sqrt(np.sum(x_i_cent**2))
    elif halo == 'AG':
        x_i_cent = x_i - (X_AG*kpc)  # center w.r.t. Andromeda Galaxy
        r = np.sqrt(np.sum(x_i_cent**2))

    # Gravitational potential in compact notation with m and M.
    m = np.minimum(r, r_vir)
    M = np.maximum(r, r_vir)
    prefactor = -4.*Pi*G*rho_0*r_s**2
    term1 = np.log(1. + (m/r_s)) / (r/r_s)
    term2 = (r_vir/M) / (1. + (r_vir/r_s))
    potential = prefactor * (term1 - term2)

    return np.asarray(potential, dtype=np.float64)


def escape_momentum(x_i, z, rho_0, M_vir, R_vir, R_s, m_nu, halo:str):

    # Gravitational potential at position x_i.
    grav = grav_pot(x_i, z, rho_0, M_vir, R_vir, R_s, halo)

    # Escape momentum formula from Ringwald & Wong (2004).
    p_esc = np.sqrt(2*np.abs(grav)) * m_nu/NU_MASS
    y_esc = p_esc/T_CNB

    return p_esc, y_esc


def Fermi_Dirac(p):
    """Fermi-Dirac phase-space distribution for CNB neutrinos. 
    Zero chem. potential and temp. T_CNB (CNB temp. today). 

    Args:
        p (array): magnitude of momentum, must be in eV!

    Returns:
        array: Value of Fermi-Dirac distr. at p.
    """

    # Function expit from scipy equivalent to 1/(np.exp(-X)+1).
    # (thus the minus sign)
    return expit(-p/T_CNB) 


def number_density(p0, p1, pix_sr=4*Pi):
    """Neutrino number density obtained by integration over initial momenta.

    Args:
        p0 (array): neutrino momentum today
        p1 (array): neutrino momentum at z_back (final redshift in sim.)

    Returns:
        array: Value of relic neutrino number density in (1/cm^3).
    """    

    g = 2.  # 2 degrees of freedom per flavour: particle and anti-particle
    
    # note: trapz integral method needs sorted (ascending) "x-axis" array.
    ind = p0.argsort(axis=-1)
    p0_sort = np.take_along_axis(p0, ind, axis=-1)
    p1_sort = np.take_along_axis(p1, ind, axis=-1)

    # Fermi-Dirac value with momentum at end of sim.
    FDvals = Fermi_Dirac(p1_sort)

    # Calculate number density.
    y = p0_sort**2 * FDvals
    x = p0_sort
    n_raw = np.trapz(y, x, axis=-1)

    # Multiply by constants and/or solid angles and convert to 1/cm**3.
    n_cm3 = pix_sr * g/((2*Pi)**3) * n_raw / (1/cm**3)

    return n_cm3


def number_densities_mass_range(
    sim_vels, nu_masses, out_file, pix_sr=4*Pi,
    average=False, m_start=0.01, z_start=0.
):
    
    # Convert velocities to momenta.
    p_arr, _ = velocity_to_momentum(sim_vels, nu_masses)

    if average:
        inds = np.array(np.where(ZEDS >= z_start)).flatten()
        temp = [
            number_density(p_arr[...,0], p_arr[...,k], pix_sr) for k in inds
        ]
        num_densities = np.mean(np.array(temp.T), axis=-1)
    else:
        num_densities = number_density(p_arr[...,0], p_arr[...,-1], pix_sr)

    np.save(f'{out_file}', num_densities)


def plot_eta_band(
    etas_sim, etas_smooth, 
    m_nu_range, fig_dir, fname, show=False, Mertsch=False, BenchHaloEtas=None,
    ylims=(1e-3, 1e1)
):

    fig, ax = plt.subplots(1,1)
    fig.patch.set_facecolor('cornflowerblue')

    # Plot smooth simulation.
    ax.plot(
        m_nu_range*1e3, (etas_smooth-1), color='red', ls='solid', 
        label='Analytical simulation'
    )

    # Plot dicrete simulation.
    if etas_sim.ndim <= 1:
        ax.plot(
            m_nu_range*1e3, (etas_sim-1), color='blue', 
            label='medians'
        )
    else:
        nus_median = np.median(etas_sim, axis=0)
        nus_perc2p5 = np.percentile(etas_sim, q=2.5, axis=0)
        nus_perc97p5 = np.percentile(etas_sim, q=97.5, axis=0)
        nus_perc16 = np.percentile(etas_sim, q=16, axis=0)
        nus_perc84 = np.percentile(etas_sim, q=84, axis=0)
        ax.plot(
            m_nu_range*1e3, (nus_median-1), color='blue', 
            label='Halo sample medians'
        )
        ax.fill_between(
            m_nu_range*1e3, (nus_perc2p5-1), (nus_perc97p5-1), 
            color='blue', alpha=0.2, label='2.5-97.5 %'
        )
        ax.fill_between(
            m_nu_range*1e3, (nus_perc16-1), (nus_perc84-1), 
            color='blue', alpha=0.3, label='16-84 %'
        )

    if Mertsch:
        # Plot endpoint values from Mertsch et al.
        x_ends = [1e1, 3*1e2]
        y_ends = [3*1e-3, 4]
        ax.scatter(x_ends, y_ends, marker='x', s=15, color='orange')

    if BenchHaloEtas is not None:
        # Plot (sampled) benchmark halo simulation.
        ax.plot(
            m_nu_range*1e3, (BenchHaloEtas-1), color='blueviolet', ls='solid', 
            label='Benchmark Halo simulation'
        )


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Overdensity band')
    ax.set_xlabel(r'$m_{\nu}$ [meV]')
    ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$')
    ax.set_ylim(ylims[0], ylims[1])
    plt.grid(True, which="both", ls="-")
    plt.legend(loc='lower right')

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))

    fig_out = f'{fig_dir}/etas_band_{fname}.pdf'
    plt.savefig(
        fig_out, facecolor=fig.get_facecolor(), edgecolor='none', 
        bbox_inches='tight'
    )
    if show:
        plt.show()
    else:
        plt.close()


def plot_eta_z_back_1Halo(sim_vels, nu_masses, fig_dir, fname, show=False):
    
    # Convert to momenta. 
    p_arr, _ = velocity_to_momentum(sim_vels, nu_masses)

    # Overdensities for each z_back.
    inds = np.arange(p_arr.shape[-1])
    etas_zeds = np.array(
        [number_density(p_arr[...,0], p_arr[...,z]) for z in inds]
    ).T/N0

    fig, ax = plt.subplots(1,1, figsize=(8,12))
    fig.patch.set_facecolor('cornflowerblue')


    colors = ['blue', 'orange', 'green', 'red']
    for j, m in enumerate(nu_masses):
        ax.semilogy(ZEDS, etas_zeds[j]-1, c=colors[j], label=f'{m:.3f} eV')

    ax.set_title('Overdensities evolution')
    ax.set_xlabel('z')
    ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$')
    ax.set_ylim(1e-3, 3e1)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
    plt.legend()

    fig_out = f'{fig_dir}/eta_z_back_{fname}.pdf'
    plt.savefig(
        fig_out, facecolor=fig.get_facecolor(), edgecolor='none', 
        bbox_inches='tight'
    )
    if show:
        plt.show()
    else:
        plt.close()


def plot_phase_space_1Halo(
    sim_vels, nu_masses, halo_param_arr, 
    vels, phis, thetas, lower, upper,
    fig_dir, fname, show=False, curtain=False
):

    # Convert to momenta.
    p_arr, y_arr = velocity_to_momentum(sim_vels, nu_masses)
    p0_arr, p1_arr, y0_arr = p_arr[...,0], p_arr[...,-1], y_arr[...,0]

    # Sort.
    ind = p0_arr.argsort(axis=-1)
    p1_sort = np.take_along_axis(p1_arr, ind, axis=-1)
    y0_sort = np.take_along_axis(y0_arr, ind, axis=-1)

    if curtain:
        p1_final = p1_sort
        y0_final = y0_sort
    else:
        # Each velocity has a batch of neutrinos.
        # (min. of each to represent most clustered ones)
        m_len = (len(nu_masses))
        p1_blocks = p1_sort.reshape((m_len, vels, phis*thetas))
        p1_final = np.min(p1_blocks, axis=-1)
        y0_blocks = y0_sort.reshape((m_len, vels, phis*thetas))
        y0_final = y0_blocks[...,0]

    # Fermi Dirac of the smoothed final momenta.
    FDvals = Fermi_Dirac(p1_final)

    fig, axs = plt.subplots(2,2, figsize=(12,12))
    fig.suptitle(
        'Phase-space distr. "today" compared to Fermi-Dirac' ,
        fontsize=18
    )

    for j, m_nu in enumerate(nu_masses):

        k = j
        i = 0
        if j in (2,3):
            i = 1
            j -= 2

        # Simulation phase-space distr. of neutrinos today.
        axs[i,j].loglog(
            y0_final[k], FDvals[k], label='PS today (from sim)', c='red', alpha=0.9
        )

        # Fermi-Dirac phase-space distr.
        pOG = np.geomspace(lower, upper, FDvals.shape[-1])
        FDvalsOG = Fermi_Dirac(pOG)
        yOG = pOG/T_CNB
        axs[i,j].loglog(yOG, FDvalsOG, label='PS Fermi-Dirac', c='blue', alpha=0.7)

        # Escape momentum.
        Rvir_halo = halo_param_arr[0]*kpc
        Mvir_halo = 10**(halo_param_arr[1])*Msun
        cNFW_halo = halo_param_arr[2]
        rho0_halo = scale_density_NFW(0., cNFW_halo)*(Msun/kpc**3)
        rs_halo = Rvir_halo/cNFW_halo
        _, y_esc = escape_momentum(
            X_SUN, 0., rho0_halo, Mvir_halo, Rvir_halo, rs_halo, m_nu, 'none'
        )
        axs[i,j].axvline(y_esc, c='k', ls='-.', label='y_esc')

        # Plot styling.
        axs[i,j].set_title(f'{m_nu} eV')
        axs[i,j].set_ylabel('FD(p)')
        axs[i,j].set_xlabel(r'$y = p / T_{\nu,0}$')
        axs[i,j].legend(loc='lower left')
        axs[i,j].set_ylim(1e-5, 1e0)
        axs[i,j].set_xlim(lower/T_CNB, 1e2)


    fig_out = f'{fig_dir}/phase_space_{fname}.pdf'
    plt.savefig(
        fig_out, facecolor=fig.get_facecolor(), edgecolor='none', 
        bbox_inches='tight'
    )
    if show:
        plt.show()
    else:
        plt.close()


def plot_number_density_integral(
    sim_vels, nu_masses, 
    vels, phis, thetas, lower, upper,
    fig_dir, fname, show=False, curtain=False
):

    # Convert to first and last momenta (of each neutrino).
    p_arr, _ = velocity_to_momentum(sim_vels, nu_masses)
    p0_arr, p1_arr = p_arr[...,0], p_arr[...,-1]

    # Sort.
    ind = p0_arr.argsort(axis=-1)
    p0_sort = np.take_along_axis(p0_arr, ind, axis=-1)
    p1_sort = np.take_along_axis(p1_arr, ind, axis=-1)
    
    if curtain:
        p0_final = p0_sort
        p1_final = p1_sort
    else:
        # Each velocity has a batch of neutrinos.
        # Take median as statistic of how much clustering shifts it.
        m_len = (len(nu_masses))
        p0_blocks = p0_sort.reshape((m_len, vels, phis*thetas))
        p0_final = p0_blocks[...,0]
        p1_blocks = p1_sort.reshape((m_len, vels, phis*thetas))
        p1_final = np.median(p1_blocks, axis=-1)


    print(p0_final.shape, p1_final.shape)

    # Fermi-Dirac value with momentum at end of sim.
    FDvals = Fermi_Dirac(p1_final)

    # What number density function integrates.
    y_axis_int = FDvals * p0_final**2 
    x_axis_int = p0_final



    fig, axs = plt.subplots(2,2, figsize=(12,12))
    fig.suptitle(
        'Inegral for number density visualized' ,
        fontsize=18
    )

    for j, m_nu in enumerate(nu_masses):

        k = j
        i = 0
        if j in (2,3):
            i = 1
            j -= 2

        # Using all velocities of sim.
        axs[i,j].semilogx(
            x_axis_int[k]/T_CNB, y_axis_int[k], 
            alpha=0.7, c='red')

        # Fermi-Dirac phase-space distr.
        pOG = np.geomspace(lower, upper, FDvals.shape[-1])
        integrand_FD = Fermi_Dirac(pOG) * pOG**2
        axs[i,j].semilogx(
            pOG/T_CNB, integrand_FD, 
            label='PS Fermi-Dirac', c='blue', alpha=0.7, ls=':'
        )

        # Plot settings.
        axs[i,j].set_title(f'{m_nu} eV')
        axs[i,j].set_xlabel(r'y = $p_0/T_{CNB}$')
        axs[i,j].set_ylabel(r'$FD(p_1) \cdot p_0^2$')
        
        if k==3:
            axs[i,j].set_ylim(0, 0.9*1e-7)
        else:
            axs[i,j].set_ylim(0, 2*1e-8)


    fig_out = f'{fig_dir}/number_density_integral_{fname}.pdf'
    plt.savefig(
        fig_out, facecolor=fig.get_facecolor(), edgecolor='none', 
        bbox_inches='tight'
    )
    if show:
        plt.show()
    else:
        plt.close()



#########################
### Old code snippets ###
#########################
'''
    # note: until further understood, this the angle criterium doesn't work.
    # Determine which cells are close enough and need multipole expansion:    
    # Cells with angle larger than the critical angle are multipole cells.
    # The critical angle is dependent on the cell length of cell C.
    # (see GoodNotes for why 0.3 exponent for now)
    # theta = cell_len / cellC_dis
    # theta_crit = 1/(1.5*((cellC_gen+1)**0.3))
    # theta_crit = 1/(1.5*cellC_len*((cellC_gen+1)**0.3))
    # theta_crit = 1e10  # large, s.t. only monopole cells

    # Find IDs for cells, for which we additionally calculate the quadrupole.
    # Cell C will be a monopole-only cell, but with its DM count set to 0, so 
    # it won't contribute later.
    # multipole_IDs = np.argwhere(theta >= theta_crit).flatten()
    # monopole_IDs = np.argwhere(theta < theta_crit).flatten()
'''