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
def read_DM_positions_randomHalo(which_halos, mass_select):

    # Open data files.
    folder = SIM_DATA
    snaps = h5py.File(f'{folder}/snapshot_0036.hdf5')
    group = h5py.File(f'{folder}/subhalo_0036.catalog_groups')
    parts = h5py.File(f'{folder}/subhalo_0036.catalog_particles')
    props = h5py.File(f'{folder}/subhalo_0036.properties')

    ### Properties of DM particles.

    # Positions.
    a = snaps["/Header"].attrs["Scale-factor"]
    pos = snaps['PartType1/Coordinates'][:][:] * a  
    #! comoving to physical (pc) with a, then *1e3 to go to kpc

    # Masses.
    mass = snaps['PartType1/Masses'][:] * 1e10  
    #! some choice of Camila, *1e10 to get to Msun. All DM particles have same mass.

    # Velocities.
    vel = snaps['PartType1/Velocities'][:][:]  #! in km/s, physical

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

    # Select halos based on exponent, i.e. mass_select input parameter.
    select_halos = np.where(
        (m200c >= mass_select-0.2) & (m200c <= mass_select+0.2)
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
    np.random.seed(SEED)
    select_random = np.random.randint(len(select_halos) - 1, size=(1))
    rand_halo = select_halos[select_random]

    # Grab the start position in the particles file to read from
    halo_start_pos = group["Offset"][rand_halo][0]
    halo_end_pos = group["Offset"][rand_halo + 1][0]

    particle_ids_in_halo = parts["Particle_IDs"][halo_start_pos:halo_end_pos]
    particle_ids_from_snapshot = snaps["PartType1/ParticleIDs"][...]

    # Get indices of elements, which are present in both arrays.
    _, _, indices_p = np.intersect1d(
        particle_ids_in_halo, particle_ids_from_snapshot, 
        assume_unique=True, return_indices=True
    )

    particles_mass = mass[indices_p]
    particles_pos = pos[indices_p, :]  # : grabs all 3 spatial positions.
    particles_pos -= CoP[rand_halo, :]  # centering, w.r.t halo they're part of
    particles_pos *= 1e3  # to kpc


    # Save positions relative to CoP (center of halo potential).
    np.save(
        f'sim_data/DM_positions_{which_halos}_M{mass_select}.npy',
        particles_pos,
    )


def grid_3D(l, s):

    # Generate edges of 3D grid.
    x, y, z = np.mgrid[-l:l+0.1:s, -l:l+0.1:s, -l:l+0.1:s]
    
    # Calculate centers of each axis.
    x_centers = (x[1:,...] + x[:-1,...])/2.
    y_centers = (y[:,1:,:] + y[:,:-1,:])/2.
    z_centers = (z[...,1:] + z[...,:-1])/2.

    # Create center coord.-pairs., truncate redundant points.
    centers3D = np.array([
        x_centers[:,:-1,:-1], 
        y_centers[:-1,:,:-1], 
        z_centers[:-1,:-1,:]
    ])

    cent_coordPairs3D = centers3D.reshape(3,-1).T 

    return cent_coordPairs3D


# @nb.njit
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

    # print(f'{len(DM_pos_trunc)} DM particles inside range')

    ### Calculate superposition gravity.
    pre = G*m_DM
    quotient = (cell_coords-DM_pos_inRange)/(DM_dist_inRange_rep**3)
    derivative = pre*np.sum(quotient, axis=0)

    #NOTE: Minus sign, s.t. velocity changes correctly (see notes).
    return np.asarray(-derivative, dtype=np.float64)


def load_derivative_grid(z):

    '''
    # ID corresponding to current z.
    idx = np.abs(Z_SNAPSHOTS - z).argmin()

    # Load file with derivative grid of ID.
    dPsi_grid = np.load(f'~/sim_data/derivative_grid_{idx}')
    '''

    #NOTE: for now while testing
    dPsi_grid = np.load(f'{os.getcwd()}/sim_data/dPsi_grid.npy')

    return dPsi_grid
    

def load_cell_coordinates(z):

    '''
    # ID corresponding to current z.
    idx = np.abs(Z_SNAPSHOTS - z).argmin()

    # Load file with position grid of ID.
    cell_coords = np.load(f'~/sim_data/position_grid_{idx}')
    '''

    #NOTE: for now while testing
    cell_coords = np.load(f'{os.getcwd()}/sim_data/cell_coords.npy')

    return cell_coords


def nu_in_which_cell(nu_coords, cell_coords):

    # For now, just subtract nu_coords from all cell_coords, then take min.
    dist = np.sqrt(np.sum((np.abs(cell_coords-nu_coords)**2), axis=1))
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
        sim = np.load(f'neutrino_vectors/nus_{nr_of_nus}_SpaceCubes.npy')
        u_all = sim[:,:,3:6]

    else:
        sim = np.load(f'neutrino_vectors/nus_{nr_of_nus}_halos_{halos}.npy')
        u_all = sim[:,:,3:6]

    # u_all.shape = (nr_of_nus, len(ZEDS), 3) ; u_all.ndim = 3

    return u_all


def load_x_sim(nr_of_nus, halos:str, discrete=False):
    """Loads neutrino positions of simulation."""

    sim = np.load(f'neutrino_vectors/nus_{nr_of_nus}_halos_{halos}.npy')
    x_all = sim[:,:,0:3]

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
    p_sim = mag_sim*(kpc/s) * NU_MASS

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

def draw_ui(phi_points, theta_points, method):
    """Get initial velocities for the neutrinos."""

    if method == 'P':
        # Convert momenta to initial velocity magnitudes, in units of [kpc/s].
        v_kpc = MOMENTA / NU_MASS / (kpc/s)
    elif method == 'V':
        # Run simulation covering total velocity range for all masses.
        v_kpc = VELOCITIES_KPC


    # Split up this magnitude into velocity components.
    #NOTE: Done by using spher. coords. trafos, which act as "weights".

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
def grav_pot(x_i, z, rho_0, M_vir):

    # Compute values dependent on redshift.
    r_vir = R_vir_fct(z, M_vir)
    r_s = r_vir / c_vir(z, M_vir)
    
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
    """Derivative of MW NFW grav. potential w.r.t. any axis x_i.

    Args:
        x_i (array): spatial position vector
        z (array): redshift
        rho_0 (float): normalization
        M_vir (float): virial mass

    Returns:
        array: Derivative vector of grav. potential. for all 3 spatial coords.
               with units of acceleration.
    """    


    # Compute values dependent on redshift.
    r_vir = R_vir_fct(z, M_vir)
    r_s = r_vir / c_vir(z, M_vir, R_vir, R_s)
    

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


    #NOTE: Minus sign, s.t. velocity changes correctly (see notes).
    return np.asarray(-derivative, dtype=np.float64)


def escape_momentum(x_i, z, rho_0, M_vir, masses):

    # Gravitational potential at position x_i.
    grav = grav_pot(x_i, z, rho_0, M_vir)

    # Escape momentum formula from Ringwald & Wong (2004).
    p_esc = np.sqrt(2*np.abs(grav)) * masses
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