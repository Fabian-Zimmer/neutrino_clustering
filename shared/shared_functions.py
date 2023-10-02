from shared.preface import *


def halo_batch_indices(
    snap, mass_gauge, mass_lower, mass_upper, 
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
        (m200c >= mass_gauge-mass_lower) & (m200c <= mass_gauge+mass_upper)
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
    # if halo_number >= halo_limit:

    #     # Fix pseudo-random choice of halos.
    #     # np.random.seed(1)
    #     random.seed(1)
        
    #     # Select non-repeating indices for halos.
    #     # rand_IDs = np.random.randint(0, halo_number-1, size=(halo_limit))
    #     rand_IDs = random.sample(list(np.arange(halo_number)), halo_limit)
    #     select_halos = select_halos[rand_IDs]


    # Save cNFW, rvir and Mvir of halos in batch.
    halo_params = np.zeros((len(select_halos), 3))
    for j, halo_idx in enumerate(select_halos):
        halo_params[j, 0] = rvir[halo_idx]
        halo_params[j, 1] = m200c[halo_idx]
        halo_params[j, 2] = cNFW[halo_idx]

    # Sort arrays by concentration (in descending order)
    order = halo_params[:,2].argsort()[::-1]
    select_halos_sorted = select_halos[order]
    halo_params_sorted = halo_params[order]

    # Delete entries with 0. concentration (erroneous halos)
    not0c = ~np.any(halo_params_sorted==0, axis=1)
    select_halos_trimmed = select_halos_sorted[not0c]
    halo_params_trimmed = halo_params_sorted[not0c]

    if halo_number >= halo_limit:
        select_halos_lim = select_halos_trimmed[:halo_limit]
        halo_params_lim = halo_params_trimmed[:halo_limit]

    np.save(f'{out_dir}/halo_batch_{fname}_indices.npy', select_halos_lim)
    np.save(f'{out_dir}/halo_batch_{fname}_params.npy', halo_params_lim)


def read_column_from_file(filename, column_name):
    df = pd.read_csv(filename)
    return df[column_name].tolist()


def rotation_matrix(theta1, theta2, theta3, order='xyz'):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        order = rotation order of x,y,z e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])

    return matrix


def rotation_angles(matrix, order):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == 'xzx':
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == 'xyx':
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 *np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == 'yxy':
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == 'yzy':
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 *np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == 'zyz':
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == 'zxz':
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 *np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == 'xzy':
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == 'xyz':
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == 'yxz':
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == 'yzx':
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == 'zyx':
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == 'zxy':
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return (theta1, theta2, theta3)


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


@nb.njit
def fct_rho_crit(z, H0, Omega_M, Omega_L):
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


def scale_density_NFW(c, z, H0, Omega_M, Omega_L):
    """Eqn. (2) from arXiv:1302.0288. c=r_200/r_s."""
    numer = 200 * c**3
    denom = 3 * (np.log(1+c) - (c/(1+c)))
    delta_c = numer/denom

    rho_crit = fct_rho_crit(z, H0, Omega_M, Omega_L)

    return rho_crit*delta_c


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
    y = p0_sort**3 * FDvals
    x = p0_sort
    n_raw = np.trapz(y, np.log(x), axis=-1)

    # Multiply by constants and/or solid angles and convert to 1/cm**3.
    n_cm3 = pix_sr * g/((2*Pi)**3) * n_raw / (1/cm**3)

    return n_cm3


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


def escape_momentum(x_i, R_vir, R_s, rho_0, m_nu_eV):
    
    # Calculate gravitational potential at coords. x_i.
    r = np.linalg.norm(x_i, axis=-1)
    m = np.minimum(r, R_vir)
    M = np.maximum(r, R_vir)
    prefactor = -4.*Pi*G*rho_0*R_s**2
    term1 = np.log(1. + (m/R_s)) / (r/R_s)
    term2 = (R_vir/M) / (1. + (R_vir/R_s))
    potential = prefactor * (term1 - term2)

    # Escape momentum formula from Ringwald & Wong (2004).
    if m_nu_eV is None:
        v_esc = np.sqrt(2*np.abs(potential))
        return v_esc
    else:
        p_esc = np.sqrt(2*np.abs(potential)) * m_nu_eV
        y_esc = p_esc/T_CNB
        return p_esc, y_esc


def read_DM_halo_index(
    snap, halo_ID, fname, box_file_dir, out_dir, direct=False
):

    # ---------------- #
    # Open data files. #
    # ---------------- #

    snaps = h5py.File(f'{box_file_dir}/snapshot_{snap}.hdf5')
    group = h5py.File(f'{box_file_dir}/subhalo_{snap}.catalog_groups')
    parts = h5py.File(f'{box_file_dir}/subhalo_{snap}.catalog_particles')
    props = h5py.File(f'{box_file_dir}/subhalo_{snap}.properties')

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

    # Save c.o.m. coord of all DM particles (used for outside_gravity).
    DM_com_coord = np.sum(DM_pos, axis=0)/len(DM_pos)

    if direct:
        return DM_pos, DM_com_coord
    else:
        np.save(f'{out_dir}/DM_pos_indices_{fname}', indices_p)
        np.save(f'{out_dir}/DM_pos_{fname}.npy', DM_pos)
        np.save(f'{out_dir}/DM_com_coord_{fname}.npy', DM_com_coord)


def read_DM_all_inRange(
    snap, halo_ID, mode, DM_shell_edges,
    fname, box_file_dir, out_dir, direct=False
):

    # --------------- #
    # Initialize data #
    # --------------- #

    snaps = h5py.File(f'{box_file_dir}/snapshot_{snap}.hdf5')
    props = h5py.File(f'{box_file_dir}/subhalo_{snap}.properties')

    # Positions.
    a = snaps["/Header"].attrs["Scale-factor"]
    pos = snaps['PartType1/Coordinates'][:][:]*a

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

    # DM_shell_edges from physical to comoving, sync to Camilas Gpc units
    DM_dis_lims = DM_shell_edges*(a/kpc/1e3)

    if mode == 'single_halos':
        # ----------------------------------- #
        # Save DM content up to Rvir of halo. #
        # ----------------------------------- #
        
        DM_pos = pos[DM_dis <= DM_dis_lims, :]*1e3
        DM_com_coord = np.sum(DM_pos, axis=0)/len(DM_pos)

        if direct:
            return DM_pos, DM_com_coord
        else:        
            np.save(f'{out_dir}/DM_pos_{fname}.npy', DM_pos)
            np.save(f'{out_dir}/DM_com_coord_{fname}.npy', DM_com_coord)

    elif mode == 'spheres':
        # ----------------------------------- #
        # Save DM in spherical shell batches. #
        # ----------------------------------- #

        for i, (shell_start, shell_end) in enumerate(
            zip(DM_dis_lims[:-1], DM_dis_lims[1:])
        ):

            DM_pos = pos[(shell_start < DM_dis) & (DM_dis <= shell_end), :]*1e3
            np.save(f'{out_dir}/DM_pos_{fname}_shell{i}.npy', DM_pos)


def halo_DM(halo_idx, snap, pos, snap_Particle_IDs, sim_dir, out_dir):

    # Open data files.
    group = h5py.File(f'{sim_dir}/subhalo_{snap}.catalog_groups')
    parts = h5py.File(f'{sim_dir}/subhalo_{snap}.catalog_particles')

    # Start and stop index for current halo.
    halo_init = group["Offset"][int(halo_idx)]
    halo_stop = group["Offset"][int(halo_idx) + 1]

    # Particle IDs of halo and snapshot.
    halo_Particle_IDs = parts["Particle_IDs"][int(halo_init):int(halo_stop)]

    # Particle IDs present in both above arrays.
    _, _, indices_p = np.intersect1d(
        halo_Particle_IDs, snap_Particle_IDs, 
        assume_unique=True, return_indices=True
    )

    # Save DM positions.
    DM_pos = pos[indices_p, :]  # x,y,z of each DM particle
    np.save(f'{out_dir}/DM_of_haloID{halo_idx}.npy', DM_pos)


def read_DM_halos_shells(
    snap, halo_ID, DM_shell_edges, halo_limits, fname, sim_dir, out_dir, CPUs
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


    # --------------------------------- #
    # Combine DM of all halos in range. #
    # --------------------------------- #

    # Center all center-of-potential coords. on selected (host) halo c.o.p.
    CoP_cent = CoP - CoP_halo
    halo_dis = np.sqrt(np.sum(CoP_cent**2, axis=1))

    # DM_shell_edges from physical to comoving, sync to TangoSIDM Gpc units
    DM_dis_lims = DM_shell_edges*(a/kpc/1e3)


    # Select all halos in shell segments.
    shells = len(DM_shell_edges) - 1
    halo_IDs = [np.array([halo_ID])]
    for i, (shell_start, shell_end, shell_lim) in enumerate(
        zip(DM_dis_lims[:-1], DM_dis_lims[1:], halo_limits[:shells])
    ):
        shell_cond = (halo_dis > shell_start) & (halo_dis <= shell_end)
        halos_in_shells = np.where(shell_cond)[0]

        # Keep only most massive ones (determined by halo_limit) in each shell
        lim_IDs = halos_in_shells[:shell_lim]

        # For first shell, remove ID of selected (host) halo
        fin_IDs = np.delete(lim_IDs, np.s_[lim_IDs==halo_ID])

        halo_IDs.append(fin_IDs)

    halo_IDs_for_starmap = np.array(list(chain.from_iterable(halo_IDs)))
    
    # Arrays to only load once.
    snap_Particle_IDs = snaps["PartType1/ParticleIDs"][...]

    with Pool(CPUs) as pool:
        pool.starmap(halo_DM, zip(
            halo_IDs_for_starmap,
            repeat(snap), repeat(pos), repeat(snap_Particle_IDs),
            repeat(sim_dir), repeat(out_dir)
        ))
        
    # Combine DM from all halos into 1 file.
    DM_halos = [
        np.load(f'{out_dir}/DM_of_haloID{i}.npy') for i in halo_IDs_for_starmap
    ]

    # note: Something for plotting...maybe still useful
    # DM_lengths = np.zeros(len(DM_halos))
    # for i, DM_elem in enumerate(DM_halos):
    #     DM_lengths[i] = len(DM_elem)
    # np.save(f'{out_dir}/DM_lengths_{fname}.npy', DM_lengths)

    # Combine all DM particles from selected halos into one file.
    DM_total = np.concatenate(DM_halos, axis=0)
    DM_total -= CoP_halo
    DM_total *= 1e3
    np.save(f'{out_dir}/DM_pos_{fname}.npy', DM_total) 
    delete_temp_data(f'{out_dir}/DM_of_haloID*.npy')

    # Save c.o.m. coord of all DM particles (used for outside_gravity fct.).
    DM_com_coord = np.sum(DM_total, axis=0)/len(DM_total)
    np.save(f'{out_dir}/DM_com_coord_{fname}.npy', DM_com_coord)


def bin_volumes(radial_bins):
    """Returns the volumes of the bins. """

    single_vol = lambda x: (4.0 / 3.0) * np.pi * x ** 3
    outer = single_vol(radial_bins[1:])
    inner = single_vol(radial_bins[:-1])
    return outer - inner


def bin_centers(radial_bins):
    """Returns the centers of the bins. """

    outer = radial_bins[1:]
    inner = radial_bins[:-1]
    return 0.5 * (outer + inner)


def analyse_halo(mass, pos):
    # Define radial bins [log scale, kpc units]
    radial_bins = np.arange(0, 5, 0.1)
    radial_bins = 10 ** radial_bins

    # Radial coordinates [kpc units]
    r = np.sqrt(np.sum(pos ** 2, axis=1))

    SumMasses, _, _ = stat.binned_statistic(
        x=r, values=np.ones(len(r)) * mass, 
        statistic="sum", bins=radial_bins
    )
    density = (SumMasses / bin_volumes(radial_bins))  # Msun/kpc^3
    return density


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


def delete_temp_data(path_to_wildcard_files):

    temp_files = glob.glob(path_to_wildcard_files, recursive=True)

    for f in temp_files:
        try:
            os.remove(f)
        except OSError:
            print("Error while deleting file (file not found")
