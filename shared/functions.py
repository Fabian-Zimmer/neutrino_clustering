from shared.preface import *


###########################################
### Functions used in SMOOTH simulation ###
###########################################

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



#############################################
### Functions used in DISCRETE simulation ###
#############################################

def read_MergerTree(init_halo):
    # Path to merger_tree file.
    tree_path = f'{pathlib.Path.cwd().parent}/neutrino_clustering_output_local/MergerTree/MergerTree_{SIM_ID}.hdf5'

    with h5py.File(tree_path) as tree:
        # Choice of index in snapshot_0036.
        masses = tree['Assembly_history/Mass'][init_halo,:]
        zeds = tree['Assembly_history/Redshift']

        # Initial mass of traced halo.
        m0 = f'{masses[0]:.2e}'

        # Progenitor index list.
        prog_idx = tree['Assembly_history/Progenitor_index'][init_halo,:]
        prog_idx = np.array(np.expand_dims(prog_idx, axis=1), dtype=int)

    return m0, prog_idx


def read_DM_positions(
    which_halos='halos', mass_select=12, mass_range=0.2, 
    random=True, snap_num='0036', sim='L___N___', halo_index=0, init_m=0,
    save_params=False
    ):

    # Open data files.
    snaps = h5py.File(str(next(
        pathlib.Path(
            f'{SIM_DATA}/{sim}').glob(f'**/snapshot_{snap_num}.hdf5'
        )
    )))
    group = h5py.File(str(next(
        pathlib.Path(
            f'{SIM_DATA}/{sim}').glob(f'**/subhalo_{snap_num}.catalog_groups'
        )
    )))
    parts = h5py.File(str(next(
        pathlib.Path(
            f'{SIM_DATA}/{sim}').glob(f'**/subhalo_{snap_num}.catalog_particles'
        )
    )))
    props = h5py.File(str(next(
        pathlib.Path(
            f'{SIM_DATA}/{sim}').glob(f'**/subhalo_{snap_num}.properties'
        )
    )))

    ### Properties of DM particles.

    # Positions.
    a = snaps["/Header"].attrs["Scale-factor"]
    pos = snaps['PartType1/Coordinates'][:][:] * a  
    #! comoving to physical (pc) with a, then *1e3 to go to kpc (later)

    # Masses.
    # mass = snaps['PartType1/Masses'][:] * 1e10  
    #! In Camila's sims: *1e10 to get to Msun & all DM particles have same mass.

    # Velocities.
    # vel = snaps['PartType1/Velocities'][:][:]  #! in km/s, physical

    # NFW concentration parameter.
    cNFW = props['cNFW_200crit'][:]

    # Virial radius.
    rvir = props['R_200crit'][:] *1e3 # now in kpc
    
    # Critical M_200.
    m200c = props['Mass_200crit'][:] * 1e10  # now in Msun

    # Set neg. values to 1, i.e. 0 in np.log10.
    m200c[m200c <= 0] = 1

    # This gives exponents of 10^x, which reproduces m200c vals.
    m200c = np.log10(m200c)  

    # Center of Potential coordinates, for all halos.
    CoP = np.zeros((len(m200c), 3))
    CoP[:, 0] = props["Xcminpot"][:]
    CoP[:, 1] = props["Ycminpot"][:]
    CoP[:, 2] = props["Zcminpot"][:]


    if random:
        # Select halos based on exponent, i.e. mass_select input parameter.
        select_halos = np.where(
            (m200c >= mass_select-mass_range) & (m200c <= mass_select+mass_range)
        )[0]

        # Selecting subhalos or halos.
        subtype = props["Structuretype"][:]
        if which_halos == 'subhalos':
            select = np.where(subtype[select_halos] > 10)[0]
            select_halos = select_halos[select]
        else:
            select = np.where(subtype[select_halos] == 10)[0]
            select_halos = select_halos[select]

        # Select 1 random halo.
        np.random.seed(1)
        select_random = np.random.randint(len(select_halos) - 1, size=(1))
        halo_index = select_halos[select_random]

        # Grab the start position in the particles file to read from
        halo_start_pos = group["Offset"][halo_index][0]
        halo_end_pos = group["Offset"][halo_index + 1][0]

    else:
        halo_start_pos = group["Offset"][halo_index]#[0]
        halo_end_pos = group["Offset"][halo_index + 1]#[0]

    particle_ids_in_halo = parts["Particle_IDs"][halo_start_pos:halo_end_pos]
    particle_ids_from_snapshot = snaps["PartType1/ParticleIDs"][...]

    # Get indices of elements, which are present in both arrays.
    _, _, indices_p = np.intersect1d(
        particle_ids_in_halo, particle_ids_from_snapshot, 
        assume_unique=True, return_indices=True
    )

    # particles_mass = mass[indices_p]
    particles_pos = pos[indices_p, :]  # : grabs all 3 spatial positions.
    particles_pos -= CoP[halo_index, :]  # centering, w.r.t halo they're part of
    particles_pos *= 1e3  # to kpc

    # Save positions relative to CoP (center of halo potential).
    if random:
        np.save(
            f'CubeSpace/DM_positions_{which_halos}_M{mass_select}.npy',
            particles_pos,
        )
    else:
        np.save(
            f'CubeSpace/DM_positions_{sim}_snapshot_{snap_num}_{init_m}Msun.npy',
            particles_pos
        )

    if save_params:
        # Select corresponding cNFW, rvir and Mvir of chosen halo.
        halo_cNFW = cNFW[halo_index]
        halo_rvir = rvir[halo_index]
        halo_Mvir = m200c[halo_index]
        return halo_cNFW, halo_rvir, halo_Mvir


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
    # print(DM_in_cell_IDs.shape, DM_in_cell_IDs)

    # Set DM outside cell to nan values.
    DM_pos[~DM_in_cell_IDs] = np.nan

    # Sort all nan values to the bottom of axis 1.
    DM_sort = np.sort(DM_pos, axis=1)

    # Drop "rows" common to all cells, which contain only nan values. This is 
    # determined by the cell with the most non-nan entries.
    DM_count_cells = np.count_nonzero(~np.isnan(DM_sort[:,:,0]), axis=1)
    DM_compact = np.delete(
        DM_sort, np.s_[np.max(DM_count_cells):], axis=1
    )
    del DM_sort

    # Drop all cells containing an amount of DM below the given threshold, 
    # from the DM positions array.
    stable_cells = DM_count_cells <= DM_lim
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
    # note: 
    # cell_com can have (0,0,0) for a cell. Doesn't matter, since DM_count in 
    # cell is then 0, which will set term in long-range to zero.

    # Count again, where zeros are present (not ones).
    DM_count_final = np.count_nonzero(
        ~np.isnan(DM_stable_cells[:,:,0]), axis=1
    )

    # Free up memory just in case.
    del DM_stable_cells

    return DM_count_final, cell_com, stable_cells, DM_unstable_cells, thresh


def cell_division(
    init_cc, DM_pos, parent_GRID_S, DM_lim, stable_cc, sim, snap_num, 
    test_names=False
    ):

    if test_names:
        sim = 'TestFile'

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
            
            # Array containing all cells (i.e. their coords.), which need to
            # be divided into 8 "child cells", hence the name "parent cells".
            parents_cc = np.delete(init_cc, stable_cells, axis=0)
            pcs = len(parents_cc)

            # Delete all cells in initial cell coords array, corresponding to 
            # the cells in need of division, i.e. the parent cells.
            no_parents_cc = np.delete(init_cc, ~stable_cells, axis=0)

            # Save generation index of stable cells.
            cell_gen_arr.append(
                np.zeros(len(no_parents_cc), int) + cell_division_count
            )

            # -------------------------------------------------- #
            # Replace each parent cell by the 8 new child cells. #
            # -------------------------------------------------- #

            # Reset DM on parent cells, such that they can be centered on new 
            # child cells again later.
            DM_reset = DM_cc_minimal + parents_cc

            nan_count = np.count_nonzero(np.isnan(DM_reset))

            unique_DM = np.unique(DM_reset, axis=1)
            print(unique_DM.shape)

            print(
                len(unique_DM) == (len(DM_reset.flatten()) - (nan_count-1))
            )

            #? why is this statement not True? The DM_reset should be unique

            # nan_count = np.count_nonzero(np.isnan(DM_reset))

            # print(len(np.unique(DM_reset)))
            # print((len(DM_reset.flatten()) - (nan_count-1)))

            # print(
            #     len(np.unique(DM_reset)) == (len(DM_reset.flatten()) - (nan_count-1))
            # )

            #! check: DM_reset should be completely unique array, i.e. no DM
            #! particle should be present twice. (before repetition below...)

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
            sub8_coords = np.reshape(sub8_temp + parents_cc, (pcs*8, 1, 3))
            del sub8_raw, sub8_temp, parents_cc

            if cell_division_count > 0:
                stable_cc_so_far = np.concatenate(
                    (stable_cc, no_parents_cc), axis=0
                )
            else:  # ending of first division loop
                stable_cc_so_far = no_parents_cc.copy()

            # Overwrite variables for next loop.
            init_cc       = sub8_coords.copy()
            del sub8_coords
            DM_pos        = DM_rep8.copy()
            del DM_rep8
            parent_GRID_S = sub8_GRID_S.copy()
            stable_cc     = stable_cc_so_far.copy()
            del stable_cc_so_far
            
            cell_division_count += 1


def manual_cell_division(
    sim_id, snap_num, DM_raw,
    DM_lim_manual, GRID_L_manual, GRID_S_manual
):
    
    # Initial grid and DM positions.
    grid = grid_3D(GRID_L_manual, GRID_S_manual)
    init_cc = np.expand_dims(grid, axis=1)
    DM_pos = np.expand_dims(DM_raw, axis=0)
    DM_ready = np.repeat(DM_pos, len(init_cc), axis=0)
    print('Grid and DM shapes before division:', init_cc.shape, DM_ready.shape)

    cell_division_count = cell_division(
        init_cc, DM_ready, GRID_S_manual, DM_lim_manual, 
        stable_cc=None, sim=sim_id, snap_num=snap_num, 
        test_names=True  #! s.t. important files don't get changed
    )
    print(f'cell division rounds: {cell_division_count}')

    # Output.
    adapted_cc = np.load(
        f'CubeSpace/adapted_cc_TestFile_snapshot_{snap_num}.npy')
    cell_gen = np.load(
        f'CubeSpace/cell_gen_TestFile_snapshot_{snap_num}.npy')
    cell_com = np.load(
        f'CubeSpace/cell_com_TestFile_snapshot_{snap_num}.npy')
    DM_count = np.load(
        f'CubeSpace/DM_count_TestFile_snapshot_{snap_num}.npy')

    print('Shapes of output files:', adapted_cc.shape, cell_gen.shape, cell_com.shape, DM_count.shape)

    print('Total DM count across all cells:', DM_count.sum())

    return adapted_cc, cell_gen, cell_com, DM_count


def simple_gravity(x_i, M_halo, X_halo):
    pre = G*M_halo

    numer = x_i - X_halo

    # Sum along some axis...
    denom = np.sqrt(np.sum((x_i - X_halo)**2))**3

    return pre*numer/denom


def cell_gravity_3D(
    cell_coords, cell_com, cell_gen, 
    DM_pos, DM_count, m_DM, snap_num, long_range=True, test_names=False):
    
    # Center all DM positions w.r.t. cell center.
    DM_pos -= cell_coords

    # Cell lengths to limit DM particles. Limit for the largest cell is 
    # GRID_S/2, not just GRID_S, therefore the cell_gen+1 !
    cell_len = np.expand_dims(GRID_S/(2**(cell_gen+1)), axis=1)

    # Select DM particles inside each cell based on cube length generation.
    DM_in_cell_IDs = np.asarray(
        (np.abs(DM_pos[:,:,0]) <= cell_len) & 
        (np.abs(DM_pos[:,:,1]) <= cell_len) & 
        (np.abs(DM_pos[:,:,2]) <= cell_len)
    )

    # Set DM outside cell to nan values.
    DM_pos[~DM_in_cell_IDs] = np.nan

    # Sort (nan values to bottom) and truncate array based on DM_LIM parameter.
    # This simple way works since each cell cannot have more than DM_LIM.
    DM_sort = np.sort(DM_pos, axis=1)
    DM_in = DM_sort[:,:DM_LIM,:]
    del DM_pos, DM_sort

    # Calculate distances of DM and adjust array dimensionally.
    DM_dis = np.expand_dims(np.sqrt(np.sum(DM_in**2, axis=2)), axis=2)

    # ------------------------------ #
    # Calculate short-range gravity. #
    # ------------------------------ #

    # Cells can be as small as ~1kpc, offset of resolution of Camila's sim 
    # (650 pc) is too high then, thus division by 10.
    eps = 650*pc / 2.

    # nan values to 0 for numerator, and 1 for denominator to avoid infinities.
    quot = np.nan_to_num(cell_coords - DM_in, copy=False, nan=0.0) / \
        np.nan_to_num(
            np.power((DM_dis**2 + eps**2), 3./2.), copy=False, nan=1.0
        )
    del DM_in, DM_dis
    dPsi_short = G*m_DM*np.sum(quot, axis=1)
    del quot

    if long_range:
        # ----------------------------- #
        # Calculate long-range gravity. #
        # ----------------------------- #
        
        # Number of cells.
        cs = cell_coords.shape[0]
        
        # Adjust c.o.m cell cords. and DM count arrays dimensionally.
        com_rep = np.repeat(
            np.expand_dims(cell_com, axis=1), cs, axis=1
        )
        DM_count_rep = np.repeat(
            np.expand_dims(DM_count, axis=1), cs, axis=1
        )

        # Create mask to drop cell, for which long-range gravity is being computed.
        # Otherwise each cell will get its own c.o.m. gravity additionally.
        mask_raw = np.zeros((cs, cs), int)
        np.fill_diagonal(mask_raw, 1)

        # Before mask changes dimensionally, filter DM count array, 
        # then adjust it dimensionally.
        DM_count_del = DM_count_rep[~mask_raw.astype(dtype=bool)].reshape(cs,cs-1)
        DM_count_sync = np.expand_dims(DM_count_del, axis=2)

        # Adjust mask dimensionally and filter c.o.m. cell coords.
        mask = np.repeat(np.expand_dims(mask_raw, axis=2), 3, axis=2)
        com_del = com_rep[~mask.astype(dtype=bool)].reshape(cs, cs-1, 3)
        del com_rep

        # Distances between cell centers and cell c.o.m. coords.
        com_dis = np.sqrt(np.sum((cell_coords-com_del)**2, axis=2))
        com_dis_sync = np.expand_dims(com_dis, axis=2)

        # Long-range gravity component for each cell (without including itself).
        quot_long = (cell_coords-com_del)/np.power(com_dis_sync, 3)
        dPsi_long = G*m_DM*np.sum(DM_count_sync*quot_long, axis=1)
        del quot_long

        # Total derivative as short+long range.
        derivative = dPsi_short + dPsi_long
    else:
        derivative = dPsi_short

    # note: Minus sign, s.t. velocity changes correctly (see GoodNotes).
    dPsi_grid = np.asarray(-derivative, dtype=np.float64)

    # Rename files when testing to not overwrite important files for sims.
    if test_names:
        snap_num = 'TestFile'

    np.save(f'CubeSpace/dPsi_grid_snapshot_{snap_num}', dPsi_grid)


def load_grid(z, sim, which):

    # ID corresponding to current z.
    idx = np.abs(ZEDS_SNAPSHOTS - z).argmin()
    snap = NUMS_SNAPSHOTS[idx]

    if which == 'derivatives':
        # Load file with derivative grid of ID.
        grid = np.load(f'CubeSpace/dPsi_grid_snapshot_{snap}.npy')

    elif which == 'positions':
        # Load file with position grid of ID.
        grid = np.load(f'CubeSpace/adapted_cc_{sim}_snapshot_{snap}.npy')

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


def load_u_sim(nr_of_nus, halos='MW', discrete=False):
    """Loads neutrino velocities of simulation."""

    if discrete:
        sim = np.load(f'neutrino_vectors/nus_{nr_of_nus}_CubeSpace.npy')
        u_all = sim[:,:,3:6]

    else:
        sim = np.load(f'neutrino_vectors/nus_{nr_of_nus}_halos_{halos}.npy')
        u_all = sim[:,:,3:6]

    # u_all.shape = (nr_of_nus, len(ZEDS), 3) ; u_all.ndim = 3

    return u_all


def load_x_sim(nr_of_nus, halos='MW', discrete=False):
    """Loads neutrino positions of simulation."""

    if discrete:
        sim = np.load(f'neutrino_vectors/nus_{nr_of_nus}_CubeSpace.npy')
        x_all = sim[:,:,0:3]

    else:
        sim = np.load(f'neutrino_vectors/nus_{nr_of_nus}_halos_{halos}.npy')
        x_all = sim[:,:,0:3]

    # x_all.shape = (nr_of_nus, len(ZEDS), 3) ; x_all.ndim = 3

    return x_all


def u_to_p_eV(u_sim, m_target):
    """Converts velocities (x,y,z from simulation) to 
    magnitude of momentum [eV] and ratio y=p/T_nu, according to desired
    target mass (and mass used in simulation)."""

    # Magnitude of velocity
    if u_sim.ndim in (0,1):
        mag_sim = np.sqrt(np.sum(u_sim**2))
    elif u_sim.ndim == 3:
        mag_sim = np.sqrt(np.sum(u_sim**2, axis=2))
    else:
        mag_sim = np.sqrt(np.sum(u_sim**2, axis=1))

    # From velocity (magnitude) in kpc/s to momentum in eV.

    # p_sim = mag_sim*(kpc/s) * NU_MASS #! only for non-rel neutrinos...

    #! The correct treatment is:
    gamma_L = 1/np.sqrt(1-mag_sim**2)
    p_sim = gamma_L * mag_sim*(kpc/s) * NU_MASS

    #note: as max. velocity is ~20% of c, the difference is not significant!

    # From p_sim to p_target.
    p_target = p_sim * m_target/NU_MASS

    # p/T_CNB ratio.
    y = p_target/T_CNB

    return p_target, y


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

def draw_ui(phi_points, theta_points):
    """Get initial velocities for the neutrinos."""

    # Convert momenta to initial velocity magnitudes, in units of [kpc/s].
    # note: since the max. velocity in the sim is ~20% of c, the difference
    # note: between the non-rel. and rel. treatment is negligible (~1%)
    # v_kpc = MOMENTA / NU_MASS / (kpc/s)  # non-rel formula
    v_kpc = 1/np.sqrt(NU_MASS**2/MOMENTA**2 + 1) / (kpc/s)  # rel. formula

    # Split up this magnitude into velocity components.
    # note: Done by using spher. coords. trafos, which act as "weights".
    eps = 0.01  # shift in theta, so poles are not included
    ps = np.linspace(0., 2.*Pi, phi_points)
    ts = np.linspace(0.+eps, Pi-eps, theta_points)

    # Minus signs due to choice of coord. system setup (see notes/drawings).
    #                              (<-- outer loops, --> inner loops)
    uxs = [-v*np.cos(p)*np.sin(t) for p in ps for t in ts for v in v_kpc]
    uys = [-v*np.sin(p)*np.sin(t) for p in ps for t in ts for v in v_kpc]
    uzs = [-v*np.cos(t) for _ in ps for t in ts for v in v_kpc]

    ui_array = np.array([[ux, uy, uz] for ux,uy,uz in zip(uxs,uys,uzs)])        

    return ui_array 


def s_of_z(z):
    """Convert redshift to time variable s with eqn. 4.1 in Mertsch et al.
    (2020), keeping only Omega_m0 and Omega_Lambda0 in the Hubble eqn. for H(z).

    Args:
        z (float): redshift

    Returns:
        float: time variable s (in [seconds] if 1/H0 factor is included)
    """    

    def s_integrand(z):        

        # We need value of H0 in units of 1/s.
        H0_val = H0/(1/s)
        a_dot = np.sqrt(Omega_M*(1.+z)**3 + Omega_L)/(1.+z)*H0_val
        s_int = 1./a_dot

        return s_int

    s_of_z, _ = quad(s_integrand, 0., z)

    return np.float64(s_of_z)


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
    
    # Derivative in compact notation with m and M.
    m = np.minimum(r, r_vir)
    M = np.maximum(r, r_vir)
    prefactor = 4.*Pi*G*rho_0*r_s**2*x_i_cent/r**2
    term1 = np.log(1. + (m/r_s)) / (r/r_s)
    term2 = (r_vir/M) / (1. + (m/r_s))
    derivative = prefactor * (term1 - term2)

    #NOTE: Minus sign, s.t. velocity changes correctly (see GoodNotes).
    return np.asarray(-derivative, dtype=np.float64)


@nb.njit
def grav_pot(x_i, z, rho_0, M_vir, R_vir, R_s):

    # Compute values dependent on redshift.
    r_vir = R_vir_fct(z, M_vir)
    r_s = r_vir / c_vir(z, M_vir, R_vir, R_s)
    
    # Distance from halo center with current coords. x_i.
    r = np.sqrt(np.sum(x_i**2))

    m = np.minimum(r, r_vir)
    M = np.maximum(r, r_vir)

    # Gravitational potential in compact notation with m and M.
    prefactor = -4.*Pi*G*rho_0*r_s**2
    term1 = np.log(1. + (m/r_s)) / (r/r_s)
    term2 = (r_vir/M) / (1. + (r_vir/r_s))
    potential = prefactor * (term1 - term2)

    return np.asarray(potential, dtype=np.float64)


def escape_momentum(x_i, z, rho_0, M_vir, R_vir, R_s, m):

    # Gravitational potential at position x_i.
    grav = grav_pot(x_i, z, rho_0, M_vir, R_vir, R_s)

    # Escape momentum formula from Ringwald & Wong (2004).
    p_esc = np.sqrt(2*np.abs(grav)) * m/NU_MASS
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


def number_density(p0, p1):
    """Neutrino number density obtained by integration over initial momenta.

    Args:
        p0 (array): neutrino momentum today
        p1 (array): neutrino momentum at z_back (final redshift in sim.)

    Returns:
        array: Value of relic neutrino number density.
    """    

    g = 2.  # 2 degrees of freedom per flavour: particle and anti-particle
    
    #NOTE: trapz integral method needs sorted (ascending) arrays
    ind = p0.argsort()
    p0_sort, p1_sort = p0[ind], p1[ind]

    # Fermi-Dirac value with momentum at end of sim.
    FDvals = Fermi_Dirac(p1_sort)

    # Calculate number density.
    y = p0_sort**2 * FDvals
    x = p0_sort
    n_raw = np.trapz(y, x)

    # Multiply by remaining g/(2*Pi^2) and convert to 1/cm**3
    n_cm3 = g/(2*Pi**2) * n_raw / (1/cm**3)

    return n_cm3


#####################
### Archived Code ###
#####################
# Perhaps useful as older functions contain other algorithms and methods. 

'''
def check_grid_OLD(init_cc, DM_pos, parent_GRID_S, DM_lim):
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

    # Count the number of DM particles in each cell, before deleting cells.
    DM_count_raw = np.count_nonzero(~np.isnan(DM_cc_compact[:,:,0]), axis=1)

    # Calculate c.o.m coords. for each cell, before deleting cells.
    # DM_count = np.repeat(np.expand_dims(DM_count_raw, axis=1), 3, axis=1)
    DM_count = np.expand_dims(DM_count_raw, axis=1)
    DM_count[DM_count==0] = 1  # to avoid divide by zero
    cell_com = np.nansum(DM_cc_compact, axis=1)/DM_count
    del DM_count

    # Drop all cells containing an amount of DM below the given threshold, 
    # from the DM positions array.
    cell_cut_IDs = DM_count_raw <= DM_lim
    DM_cc_minimal = np.delete(DM_cc_compact, cell_cut_IDs, axis=0)
    del DM_cc_compact
    thresh = np.size(DM_cc_minimal, axis=0)

    return DM_count_raw, cell_com, cell_cut_IDs, DM_cc_minimal, thresh
'''

'''
def cell_gravity(cell_coords, DM_coords, grav_range, m_DM):
    
    # Center all DM positions w.r.t. cell center.
    DM_cc = DM_coords*kpc - cell_coords

    # Calculate distances of DM to cc, sorted in ascending order.
    DM_dist = np.sqrt(np.sum(DM_cc**2, axis=1))

    # Ascending order indices.
    ind = DM_dist.argsort()

    # Truncate DM positions depending on distance to cc.
    DM_pos_sort = DM_cc[ind]
    DM_dist_sort = DM_dist[ind]

    if grav_range is None:
        DM_pos_inRange = DM_pos_sort
        DM_dist_inRange = DM_dist_sort
    else:
        DM_pos_inRange = DM_pos_sort[DM_dist_sort <= grav_range]
        DM_dist_inRange = DM_dist_sort[DM_dist_sort <= grav_range]

    # Adjust the distances array to make it compatible with DM positions array.
    DM_dist_inRange_sync = DM_dist_inRange.reshape(len(DM_dist_inRange),1)
    DM_dist_inRange_rep = np.repeat(DM_dist_inRange_sync, 3, axis=1)

    ### Calculate superposition gravity.
    pre = G*m_DM
    quotient = (cell_coords-DM_pos_inRange)/(DM_dist_inRange_rep**3)
    derivative = pre*np.sum(quotient, axis=0)

    #NOTE: Minus sign, s.t. velocity changes correctly (see GoodNotes).
    return np.asarray(-derivative, dtype=np.float64)
'''

'''
def cell_gravity_3D_OLD(
    cell_coords, cell_com, DM_pos, DM_count, grav_range, m_DM, snap_num):
    
    # Center all DM positions w.r.t. cell center.
    DM_pos -= cell_coords

    # Calculate distances of DM to cc.
    DM_dis = np.sqrt(np.sum(DM_pos**2, axis=2))

    # Ascending order indices.
    ind_2D = DM_dis.argsort(axis=1)

    #! gravity range has to be dynamic to work with long range gravity! 
    #! currently, small cells in center get duplicate gravity regions...
    if grav_range is not None:
        # Truncate DM in each cell, based on cell with most DM in range.
        # (each cell should have unique truncation according to DM_dis array, 
        # but that would make ndarray irregular, i.e. not a hyper-triangle)
        # note: this means each cell has a diff. grav. range, s.t. the range  
        # note: for each cell ultimately encompasses the same amount of DM
        diff = grav_range-DM_dis
        max_ID = np.max(np.sum(diff>=0, axis=1))
        ind_2D = ind_2D[:, :max_ID]
    

    ind_3D = np.repeat(np.expand_dims(ind_2D, axis=2), 3, axis=2)

    # Sort DM positions according to dist.
    DM_pos_sort = np.take_along_axis(DM_pos, ind_3D, axis=1)
    DM_dis_sort = np.take_along_axis(DM_dis, ind_2D, axis=1)
    del DM_dis, ind_2D, ind_3D

    # Adjust the distances array to make it compatible with DM positions array.
    DM_dis_sync = np.expand_dims(DM_dis_sort, axis=2)

    # ------------------------------ #
    # Calculate short-range gravity. #
    # ------------------------------ #

    pre = G*m_DM
    eps = 650*pc  # offset = resolution floor of Camila's sim
    quot = (cell_coords-DM_pos_sort)/np.power((DM_dis_sync**2 + eps**2), 3./2.)
    del DM_pos_sort, DM_dis_sync
    dPsi_short = pre*np.sum(quot, axis=1)
    del quot


    #! this long-range code breaks the number (over)density plot. FIX!
    # ----------------------------- #
    # Calculate long-range gravity. #
    # ----------------------------- #
    
    # Number of cells.
    cells = cell_coords.shape[0]
    
    # Adjust c.o.m cell cords. and DM count arrays dimensionally.
    com_rep = np.repeat(
        np.expand_dims(cell_com, axis=1), cells, axis=1
    )
    DM_count_rep = np.repeat(
        np.expand_dims(DM_count, axis=1), cells, axis=1
    )

    # Create mask to drop cell, for which long-range gravity is being computed.
    # Otherwise each cell will get its own c.o.m. gravity additionally.
    mask_raw = np.zeros((cells, cells), int)
    np.fill_diagonal(mask_raw, 1)

    # Before mask changes dimensionally, filter DM count array and adjust.
    DM_count_del = DM_count_rep[
        ~mask_raw.astype(dtype=bool)
    ].reshape(cells, cells-1)
    DM_count_sync = np.expand_dims(DM_count_del, axis=2)

    # Adjust mask dimensionally and filter c.o.m. cell coords.
    mask = np.repeat(np.expand_dims(mask_raw, axis=2), 3, axis=2)
    com_del = com_rep[~mask.astype(dtype=bool)].reshape(cells, cells-1, 3)
    del com_rep

    # Distances between cell centers and cell c.o.m. coords.
    com_dis = np.sqrt(np.sum((cell_coords-com_del)**2, axis=2))
    com_dis_sync = np.expand_dims(com_dis, axis=2)

    # Long-range gravity component for each cell (without including itself).
    quot_long = (cell_coords-com_del)/np.power(com_dis_sync, 3)
    dPsi_long = pre*np.sum(DM_count_sync*quot_long, axis=1)
    del quot_long

    # Total derivative as short+long range.
    derivative = dPsi_short + dPsi_long
    

    derivative = dPsi_short

    # note: Minus sign, s.t. velocity changes correctly (see GoodNotes).
    dPsi_grid = np.asarray(-derivative, dtype=np.float64)

    np.save(f'CubeSpace/dPsi_grid_snapshot_{snap_num}', dPsi_grid)
''' 