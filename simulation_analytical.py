from shared.preface import *
from shared.shared_functions import Fermi_Dirac, number_density, velocity_to_momentum

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', required=True)
parser.add_argument('-st', '--sim_type', required=True)
parser.add_argument(
    '--MW_halo', required=True, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    '--VC_halo', required=True, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    '--AG_halo', required=True, action=argparse.BooleanOptionalAction
)
args = parser.parse_args()


### ================================= ###
### Load box & simulation parameters. ###
### ================================= ###

# Box parameters.
with open(f'{args.directory}/box_parameters.yaml', 'r') as file:
    box_setup = yaml.safe_load(file)

# Simulation parameters.
with open(f'{args.directory}/sim_parameters.yaml', 'r') as file:
    sim_setup = yaml.safe_load(file)

CPUs_sim = sim_setup['CPUs_simulations']
integration_solver = sim_setup['integration_solver']
init_x_dis = sim_setup['initial_haloGC_distance']
init_xyz = np.array([init_x_dis, 0., 0.])
neutrinos = sim_setup['neutrinos']

# Load arrays.
z_int_steps = np.load(f'{args.directory}/z_int_steps.npy')
s_int_steps = np.load(f'{args.directory}/s_int_steps.npy')
neutrino_massrange = np.load(f'{args.directory}/neutrino_massrange_eV.npy')*eV
DM_shell_edges = np.load(f'{args.directory}/DM_shell_edges.npy')
shell_multipliers = np.load(f'{args.directory}/shell_multipliers.npy')


# Load constants and arrays, which some functions below need.
FCT_h = box_setup['Cosmology']['h']
FCT_H0 = FCT_h*100*km/s/Mpc
FCT_Omega_M = box_setup['Cosmology']['Omega_M']
FCT_Omega_L = box_setup['Cosmology']['Omega_L']
FCT_init_xys = np.copy(init_xyz)
FCT_zeds = np.copy(z_int_steps)


### ==================================== ###
### Define all necessary functions here. ###
### ==================================== ###

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
    arg_in_log = (M_vir / (1.e12 / FCT_h * Msun))
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
    
    H_squared = FCT_H0**2 * (FCT_Omega_M*(1.+z)**3 + FCT_Omega_L) 
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

    numer = FCT_Omega_M*(1.+z)**3
    denom = FCT_Omega_M*(1.+z)**3 + FCT_Omega_L
    Omega_M_of_z = numer/denom

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
    x_sun, y_sun, z_sun = FCT_init_xys[0], FCT_init_xys[1], FCT_init_xys[2]
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


def number_densities_mass_range(
    sim_vels, nu_masses, out_file=None, pix_sr=4*Pi,
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


def EOMs(s_val, y):
    """Equations of motion for all x_i's and u_i's in terms of s."""

    # Initialize vector.
    x_i, u_i = np.reshape(y, (2,3))

    # Switch to "numerical reality" here.
    x_i *= kpc
    u_i *= (kpc/s)

    # Find z corresponding to s via interpolation.
    z = np.interp(s_val, s_int_steps, z_int_steps)

    # Sum gradients of each halo. Seperate if statements, for adding any halos.
    grad_tot = np.zeros(len(x_i))
    if args.MW_halo:
        grad_tot += dPsi_dxi_NFW(
            x_i, z, rho0_MW, Mvir_MW, Rvir_MW, Rs_MW, 'MW'
            )
    if args.VC_halo:
        grad_tot += dPsi_dxi_NFW(
            x_i, z, rho0_VC, Mvir_VC, Rvir_VC, Rs_VC, 'VC'
            )
    if args.AG_halo:
        grad_tot += dPsi_dxi_NFW(
            x_i, z, rho0_AG, Mvir_AG, Rvir_AG, Rs_AG, 'AG'
            )

    # Switch to "physical reality" here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)

    # Hamilton eqns. for integration.
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
        y0=y0, method=integration_solver, vectorized=True
        )
    
    np.save(f'{temp_dir}/nu_{int(Nr)}.npy', np.array(sol.y.T))


### ========================== ###
### Run analytical simulation. ###
### ========================== ###

sim_start = time.perf_counter()


# Draw initial velocities.
ui_array = np.load(f'{args.directory}/initial_velocities.npy')


if ui_array.ndim == 2:
    ui_array = np.expand_dims(ui_array, axis=0)
    key_str = 'single_halos'
else:
    key_str = 'all_sky'
    all_sky_number_densities = []


for cp, ui in enumerate(ui_array):
    # Combine vectors and append neutrino particle number.
    y0_Nr = np.array(
        [np.concatenate((init_xyz, ui[k], [k+1])) for k in range(len(ui))]
        )

    if key_str == 'all_sky':
        print(f'***Coord. pair {cp+1}/{len(ui_array)} ***')

    # Run simulation on multiple cores, in batches.
    # (important for other solvers (e.g. Rk45), due to memory increase)
    batch_size = 10_000
    ticks = np.arange(0, len(ui)/batch_size, dtype=int)
    for i in ticks:

        id_min = (i*batch_size) + 1
        id_max = ((i+1)*batch_size) + 1
        print(f'From {id_min} to and incl. {id_max-1}')

        if i == 0:
            id_min = 0

        with ProcessPoolExecutor(CPUs_sim) as ex:
            ex.map(backtrack_1_neutrino, y0_Nr[id_min:id_max])

        print(f'Batch {i+1}/{len(ticks)} done!')


    # Compactify all neutrino vectors into 1 file.
    neutrino_vectors = np.array(
        [np.load(f'{temp_dir}/nu_{i+1}.npy') for i in range(len(ui))]
    )

    # For these modes (i.e. not all_sky), save all neutrino vectors.
    # Split velocities and positions into 10k neutrino batches.
    # For reference: ndarray with shape (10_000, 100, 6) is  48 MB.
    batches = math.ceil(len(ui)/10_000)
    split = np.array_split(neutrino_vectors, batches, axis=0)
    for i, elem in enumerate(split):
        np.save(
            f'{args.directory}/neutrino_vectors_analytical_batch{i+1}.npy', elem
        )

    # Compute the number densities.
    if key_str == 'single_halos':
        out_file = f'{args.directory}/number_densities_analytical_{key_str}.npy'
        number_densities_mass_range(
            neutrino_vectors[...,3:6], neutrino_massrange, out_file,
            sim_type=key_str
        )
    elif key_str == 'all_sky':
        pix_sr_sim = sim_setup['pix_sr']
        all_sky_number_densities.append(
            number_densities_mass_range(
                neutrino_vectors[...,3:6], neutrino_massrange, 
                sim_type=key_str,
                pix_sr=pix_sr_sim
            )
        )


if key_str == 'all_sky':
    np.save(
        f'{args.directory}/number_densities_analytical_all_sky.npy', 
        np.array(all_sky_number_densities)
    )


# Remove temporary folder with all individual neutrino files.
shutil.rmtree(temp_dir)   

sim_time = time.perf_counter()-sim_start
print(f'Sim time: {sim_time/60.} min, {sim_time/(60**2)} h.')