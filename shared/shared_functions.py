from shared.preface import *


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


def escape_momentum_analytical(x_i, z, R_vir, R_s, rho_0, m_nu_eV):
    
    # Calculate gravitational potential at coords. x_i.
    r = np.sqrt(np.sum(x_i**2))
    m = np.minimum(r, R_vir)
    M = np.maximum(r, R_vir)
    prefactor = -4.*Pi*G*rho_0*R_s**2
    term1 = np.log(1. + (m/R_s)) / (r/R_s)
    term2 = (R_vir/M) / (1. + (R_vir/R_s))
    potential = prefactor * (term1 - term2)

    # Escape momentum formula from Ringwald & Wong (2004).
    p_esc = np.sqrt(2*np.abs(potential)) * m_nu_eV
    y_esc = p_esc/T_CNB

    return p_esc, y_esc


def read_DM_halo_index(snap, halo_ID, fname, box_file_dir, out_dir, direct=False):

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
        np.save(f'{out_dir}/DM_pos_{fname}.npy', DM_pos)
        np.save(f'{out_dir}/DM_com_coord_{fname}.npy', DM_com_coord)


def read_DM_all_inRange(
    snap, halo_ID, DM_shell_edges, fname, box_file_dir, out_dir
):

    # --------------- #
    # Initialize data #
    # --------------- #

    snaps = h5py.File(f'{box_file_dir}/snapshot_{snap}.hdf5')
    props = h5py.File(f'{box_file_dir}/subhalo_{snap}.properties')

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
    DM_shell_edges_com = DM_shell_edges*(a/kpc/1e3)  # sync to Camilas Gpc units

    for i, (shell_start, shell_end) in enumerate(
        zip(DM_shell_edges_com[:-1], DM_shell_edges_com[1:])
    ):

        DM_pos = pos[(shell_start < DM_dis) & (DM_dis <= shell_end), :]*1e3
        np.save(f'{out_dir}/DM_pos_{fname}_shell{i}.npy', DM_pos)



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