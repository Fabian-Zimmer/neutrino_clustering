from shared.preface import *


def make_sim_parameters(
        sim_dir, sim_type,
        nu_mass_start, nu_mass_stop, nu_mass_num, nu_sim_mass,
        p_start, p_stop, p_num,
        phis, thetas,
        init_x_dis,
        z_int_shift, z_int_stop, z_int_num, 
        int_solver,
        CPUs_precalculations, CPUs_simulations, memory_limit_GB,
        DM_in_cell_limit, 
        Nside=None, Npix=None, pix_sr=None
    ):

    # Load simulation box parameters.
    with open(f'{sim_dir}/box_parameters.yaml', 'r') as file:
        box_params = yaml.safe_load(file)

    h = box_params['Cosmology']['h']
    H0 = h * 100 * km/s/Mpc
    Omega_M = box_params['Cosmology']['Omega_M']
    Omega_L = box_params['Cosmology']['Omega_L']


    ### ============================== ###
    ### Simulation control parameters. ###
    ### ============================== ###

    def s_of_z(z):
        """
        Convert redshift to time variable s with eqn. 4.1 in Mertsch et al.
        (2020), keeping only Omega_M and Omega_L in the Hubble eqn. for H(z).

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


    # Number of simulated neutrinos.
    if isinstance(phis, int):
        neutrinos = phis*thetas*p_num
    else:
        neutrinos = len(thetas)*p_num

    # Neutrino masses.
    nu_mass_eV = nu_sim_mass*eV
    nu_mass_kg = nu_mass_eV/kg
    neutrino_massrange_eV = np.geomspace(
        nu_mass_start, nu_mass_stop, nu_mass_num
    )*eV

    # Neutrino momentum range.
    neutrino_momenta = np.geomspace(p_start*T_CNB, p_stop*T_CNB, p_num)

    # Neutrino + antineutrino number density of 1 flavor in [1/cm**3],
    # using the analytical expression for Fermions.
    n0 = 2*zeta(3.)/Pi**2 *T_CNB**3 *(3./4.) /(1/cm**3)

    # Logarithmic redshift spacing, and conversion to integration variable s.
    z_int_steps = np.geomspace(z_int_shift, z_int_stop+z_int_shift, z_int_num)
    z_int_steps -= z_int_shift
    s_int_steps = np.array([s_of_z(z) for z in z_int_steps])


    # For sphere of incluence tests: Divide inclusion region into shells.
    DM_shell_edges = np.array([0,5,10,15,20,40,100])*100*kpc

    # Multiplier for DM limit for each shell.
    shell_multipliers = np.array([1,3,6,9,12,15])


    ### ============================================ ###
    ### Create .yaml file for simulation parameters. ###
    ### ============================================ ###

    # Only save nr. of angles, in all_sky simulation type.
    if sim_type == 'all_sky':
        phis_nr = len(phis)
        thetas_nr = len(thetas)
    else:
        phis_nr = phis
        thetas_nr = thetas

    sim_parameters = {
        "simulation type": sim_type,
        "neutrinos": neutrinos,
        "initial_haloGC_distance": init_x_dis,
        "neutrino_mass_start": nu_mass_start,
        "neutrino_mass_stop": nu_mass_stop,
        "neutrino_mass_num": nu_mass_num,
        "neutrino_simulation_mass_eV": nu_mass_eV,
        "neutrino_simulation_mass_kg": float(nu_mass_kg),
        "phis": phis_nr,
        "thetas": thetas_nr,
        "Nside": Nside,
        "Npix": Npix,
        "pix_sr": pix_sr,
        "momentum_start": p_start,
        "momentum_stop": p_stop,
        "momentum_num": p_num,
        "z_inegration_start": 0,
        "z_inegration_stop": z_int_stop,
        "z_inegration_num": z_int_num,
        "integration_solver": int_solver,
        "CPUs_precalculations": CPUs_precalculations,
        "CPUs_simulations": CPUs_simulations,
        "memory_limit_GB": memory_limit_GB,
        "DM_in_cell_limit": DM_in_cell_limit,
        "cosmo_neutrino_density [cm^-3]": float(n0)
    }

    with open(f'{sim_dir}/sim_parameters.yaml', 'w') as file:
        yaml.dump(sim_parameters, file)

    # Save arrays as .npy files.
    np.save(f'{sim_dir}/neutrino_massrange_eV.npy', neutrino_massrange_eV)
    np.save(f'{sim_dir}/neutrino_momenta.npy', neutrino_momenta)
    np.save(f'{sim_dir}/z_int_steps.npy', z_int_steps)
    np.save(f'{sim_dir}/s_int_steps.npy', s_int_steps)
    np.save(f'{sim_dir}/DM_shell_edges.npy', DM_shell_edges)
    np.save(f'{sim_dir}/shell_multipliers.npy', shell_multipliers)


    ### ================================ ###
    ### Initial velocities of neutrinos. ###
    ### ================================ ###
    
    # Convert momenta to initial velocity magnitudes, in units of [kpc/s].
    u_i = neutrino_momenta/nu_mass_eV / (kpc/s)

    if sim_type == 'all_sky':

        # Each coord. pair gets whole momentum, i.e. velocity range.
        glat = np.deg2rad(thetas)
        glon = np.deg2rad(phis)
        uxs = np.array([u_i*np.cos(b)*np.cos(l) for b, l in zip(glat, glon)])
        uys = np.array([u_i*np.cos(b)*np.sin(l) for b, l in zip(glat, glon)])
        uzs = np.array([u_i*np.sin(b) for b in glat])
        u_i_array = np.stack((uxs, uys, uzs), axis=2)

        # Save the theta and phi angles as numpy arrays.
        np.save(f'{sim_dir}/all_sky_angles.npy', np.transpose((thetas, phis)))

    else:

        # Split up this magnitude into velocity components, by using spher. 
        # coords. trafos, which act as "weights" for each direction.
        cts = np.linspace(-1, 1, thetas)  # cos(thetas)
        ps = np.linspace(0, 2*Pi, phis)   # normal phi angles

        # Minus signs due to choice of coord. system setup (see notes/drawings).
        #                              (<-- outer loops, --> inner loops)
        uxs = [
            u*np.cos(p)*np.sqrt(1-ct**2) for ct in cts for p in ps for u in u_i
        ]
        uys = [
            u*np.sin(p)*np.sqrt(1-ct**2) for ct in cts for p in ps for u in u_i
        ]
        uzs = [
            u*ct for ct in cts for _ in ps for u in u_i
        ]

        u_i_array = np.array(
            [[ux, uy, uz] for ux,uy,uz in zip(uxs,uys,uzs)]
        )

    # note: sim goes backwards in time, hence minus sign (see GoodNotes)'
    np.save(f'{sim_dir}/initial_velocities.npy', -u_i_array)


# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-sd', '--sim_dir', required=True)
parser.add_argument('-st', '--sim_type', required=True)

# If sim_type is all_sky, phi and theta angles are determined by Nside.
parser.add_argument('-hn', '--healpix_nside', required=False)

parser.add_argument('-ni', '--nu_mass_start', required=True)
parser.add_argument('-nf', '--nu_mass_stop', required=True)
parser.add_argument('-nn', '--nu_mass_num', required=True)
parser.add_argument('-nm', '--nu_sim_mass', required=True)
parser.add_argument('-pi', '--p_start', required=True)
parser.add_argument('-pf', '--p_stop', required=True)
parser.add_argument('-pn', '--p_num', required=True)

# Only required for modes other than all_sky.
parser.add_argument('-ph', '--phis', required=False)
parser.add_argument('-th', '--thetas', required=False)

parser.add_argument('-xi', '--init_x_dis', required=True)
parser.add_argument('-zi', '--z_int_shift', required=True)
parser.add_argument('-zf', '--z_int_stop', required=True)
parser.add_argument('-zn', '--z_int_num', required=True)
parser.add_argument('-is', '--int_solver', required=True)
parser.add_argument('-cp', '--CPUs_precalculations', required=True)
parser.add_argument('-cs', '--CPUs_simulations', required=True)
parser.add_argument('-mem', '--memory_limit_GB', required=True)
parser.add_argument('-dl', '--DM_in_cell_limit', required=True)
args = parser.parse_args()


# Adjust phi and theta angles for different modes.
if args.sim_type == 'all_sky':
    
    Nside = int(args.healpix_nside)  # Specified nside parameter, power of 2
    Npix = 12 * Nside**2  # Number of pixels
    pix_sr = (4*np.pi)/Npix  # Pixel size  [sr]

    # Galactic coordinates.
    phi_angles, theta_angles = np.array(
        hp.pixelfunc.pix2ang(Nside, np.arange(Npix), lonlat=True)
    )
else:

    phi_angles = int(args.phis)
    theta_angles = int(args.thetas)

    Nside = None
    Npix = None
    pix_sr = None


make_sim_parameters(
    sim_dir=args.sim_dir,
    sim_type=args.sim_type,
    nu_mass_start=float(args.nu_mass_start),
    nu_mass_stop=float(args.nu_mass_stop),
    nu_mass_num=int(args.nu_mass_num),
    nu_sim_mass=float(args.nu_sim_mass),
    p_start=float(args.p_start),
    p_stop=float(args.p_stop),
    p_num=int(args.p_num),
    phis=phi_angles,
    thetas=theta_angles,
    init_x_dis=float(args.init_x_dis),
    z_int_shift=float(args.z_int_shift),
    z_int_stop=float(args.z_int_stop),
    z_int_num=int(args.z_int_num),
    int_solver=args.int_solver,
    CPUs_precalculations=int(args.CPUs_precalculations),
    CPUs_simulations=int(args.CPUs_simulations),
    memory_limit_GB=int(args.memory_limit_GB),
    DM_in_cell_limit=int(args.DM_in_cell_limit),
    Nside=Nside,
    Npix=Npix,
    pix_sr=pix_sr

)