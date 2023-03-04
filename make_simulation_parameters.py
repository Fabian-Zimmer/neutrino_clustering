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
    ):

    # Load simulation box parameters.
    with open(f'{sim_dir}/box_parameters.yaml', 'r') as file:
        box_params = yaml.safe_load(file)

    h = box_params['Cosmology']['h']
    H0 = h*100*km/s/Mpc
    Omega_M = box_params['Cosmology']['Omega_M']
    Omega_L = 1.-Omega_M  # since we don't use Omega_R


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
        neutrinos = len(phis)*len(thetas)*p_num

    # Neutrino masses.
    nu_mass_eV = nu_sim_mass*eV
    nu_mass_kg = nu_mass_eV/kg
    neutrino_massrange_eV = np.geomspace(
        nu_mass_start, nu_mass_stop, nu_mass_num
    )*eV

    # Neutrino momentum range.
    neutrino_momenta = np.geomspace(p_start, p_stop, p_num)

    # Neutrino + antineutrino number density of 1 flavor in [1/cm**3],
    # using the analytical expression for Fermions.
    n0 = 2*zeta(3.)/Pi**2 *T_CNB**3 *(3./4.) /(1/cm**3)

    # Starting position of neutrinos.
    neutrino_init_position = np.array([init_x_dis, 0., 0.])

    # Logarithmic redshift spacing, and conversion to integration variable s.
    z_int_steps = np.geomspace(z_int_shift, z_int_stop+z_int_shift, z_int_num)
    z_int_steps -= z_int_shift
    s_int_steps = np.array([s_of_z(z) for z in z_int_steps])

    #? these two figure out later.
    # For sphere of incluence tests: Divide inclusion region into shells.
    DM_shell_edges = np.array([0,5,10,15,20,40,100])*100*kpc
    # Multiplier for DM limit for each shell.
    shell_multipliers = np.array([1,3,6,9,12,15])


    ### ============================================ ###
    ### Create .yaml file for simulation parameters. ###
    ### ============================================ ###

    sim_parameters = {
        "neutrinos": neutrinos,
        "initial_haloGC_distance": init_x_dis,
        "neutrino_mass_start": nu_mass_start,
        "neutrino_mass_stop": nu_mass_stop,
        "neutrino_mass_num": nu_mass_num,
        "neutrino_simulation_mass_eV": nu_mass_eV,
        "neutrino_simulation_mass_kg": float(nu_mass_kg),
        "phis": phis,
        "thetas": thetas,
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
        "cosmo_neutrino_density": float(n0)
    }

    with open(f'{sim_dir}/sim_parameters.yaml', 'w') as file:
        yaml.dump(sim_parameters, file)

    # Save arrays as .npy files.
    np.save(f'{sim_dir}/neutrino_massrange_eV.npy', neutrino_massrange_eV)
    np.save(f'{sim_dir}/neutrino_momenta.npy', neutrino_momenta)
    np.save(f'{sim_dir}/neutrino_init_position.npy', neutrino_init_position)
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
        # For all-sky script, input is just one coord. pair.
        t, p = thetas, phis

        # Each coord. pair gets whole momentum, i.e. velocity range.
        uxs = [-u*np.cos(p)*np.sin(t) for u in u_i]
        uys = [-u*np.sin(p)*np.sin(t) for u in u_i]
        uzs = [-u*np.cos(t) for u in u_i]

        u_i_array = np.array(
            [[ux, uy, uz] for ux,uy,uz in zip(uxs,uys,uzs)]
        )

    else:
        # Split up this magnitude into velocity components, by using spher. 
        # coords. trafos, which act as "weights" for each direction.
        cts = np.linspace(-1, 1, thetas)  # cos(thetas)
        ps = np.linspace(0, 2*Pi, phis)   # normal phi angles

        # Minus signs due to choice of coord. system setup (see notes/drawings).
        #                              (<-- outer loops, --> inner loops)
        uxs = [
            -u*np.cos(p)*np.sqrt(1-ct**2) for ct in cts for p in ps for u in u_i
        ]
        uys = [
            -u*np.sin(p)*np.sqrt(1-ct**2) for ct in cts for p in ps for u in u_i
        ]
        uzs = [
            -u*ct for ct in cts for _ in ps for u in u_i
        ]

        u_i_array = np.array(
            [[ux, uy, uz] for ux,uy,uz in zip(uxs,uys,uzs)]
        )

    np.save(f'{sim_dir}/initial_velocities.npy', u_i_array)


# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-sd', '--sim_dir', required=True)
parser.add_argument('-st', '--sim_type', required=True)
parser.add_argument('-ni', '--nu_mass_start', required=True)
parser.add_argument('-nf', '--nu_mass_stop', required=True)
parser.add_argument('-nn', '--nu_mass_num', required=True)
parser.add_argument('-nm', '--nu_sim_mass', required=True)
parser.add_argument('-pi', '--p_start', required=True)
parser.add_argument('-pf', '--p_stop', required=True)
parser.add_argument('-pn', '--p_num', required=True)
parser.add_argument('-ph', '--phis', required=True)
parser.add_argument('-th', '--thetas', required=True)
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


make_sim_parameters(
    sim_dir=args.sim_dir,
    sim_type=args.sim_type,
    nu_mass_start=args.nu_mass_start,
    nu_mass_stop=args.nu_mass_stop,
    nu_mass_num=args.nu_mass_num,
    nu_sim_mass=args.nu_sim_mass,
    p_start=args.p_start,
    p_stop=args.p_stop,
    p_num=args.p_num,
    phis=args.phis,
    thetas=args.thetas,
    init_x_dis=args.init_x_dis,
    z_int_shift=args.z_int_shift,
    z_int_stop=args.z_int_stop,
    z_int_num=args.z_int_num,
    int_solver=args.int_solver,
    CPUs_precalculations=args.CPUs_precalculations,
    CPUs_simulations=args.CPUs_simulations,
    memory_limit_GB=args.memory_limit_GB,
    DM_in_cell_limit=args.DM_in_cell_limit
)

# make_sim_parameters(
#     sim_dir='L025N752/DMONLY/SigmaConstant00', 
#     sim_type='halo_batch',
#     nu_mass_start=0.01, 
#     nu_mass_stop=0.3, 
#     nu_mass_num=100, 
#     nu_sim_mass=0.3,
#     p_start=0.01, 
#     p_stop=400, 
#     p_num=100,
#     phis=10, 
#     thetas=10,
#     init_x_dis=8.5,
#     z_int_shift=1e-1, 
#     z_int_stop=4, 
#     z_int_num=100, 
#     int_solver='RK23',
#     CPUs_precalculations=128, 
#     CPUs_simulations=128, 
#     memory_limit_GB=224,
#     DM_in_cell_limit=10_000, 
# )