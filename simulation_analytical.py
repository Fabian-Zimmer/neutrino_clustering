from shared.preface import *

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', required=True)
parser.add_argument('-st', '--sim_type', required=True)
parser.add_argument('-hn', '--halo_num', required=True)
parser.add_argument('-mg', '--mass_gauge', required=True)
parser.add_argument('-mr', '--mass_range', required=True)
args = parser.parse_args()


### ================================= ###
### Load box & simulation parameters. ###
### ================================= ###

# Box parameters.
with open(f'{args.directory}/box_parameters.yaml', 'r') as file:
    box_setup = yaml.safe_load(file)

box_file_dir = box_setup['File Paths/Box File Directory']
DM_mass = box_setup['Content/DM Mass [Msun]']*Msun
Smooth_L = box_setup['Content/Smoothening Length [pc]']*pc
z0_snap_4cif = box_setup['Content/z=0 snapshot']

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
nums_snaps = np.save(f'{args.directory}/nums_snaps.npy')
zeds_snaps = np.save(f'{args.directory}/zeds_snaps.npy')

z_int_steps = np.save(f'{args.directory}/z_int_steps.npy')
s_int_steps = np.save(f'{args.directory}/s_int_steps.npy')
neutrino_massrange = np.save(f'{args.directory}/neutrino_massrange_eV.npy')
DM_shell_edges = np.save(f'{args.directory}/DM_shell_edges.npy')
shell_multipliers = np.save(f'{args.directory}/shell_multipliers.npy')


# Load constants and arrays, which the functions.py script needs.
FCT_h = box_setup['Cosmology']['h']
FCT_H0 = FCT_h*100*km/s/Mpc
FCT_Omega_M = box_setup['Cosmology']['Omega_M']
FCT_Omega_L = box_setup['Cosmology']['Omega_L']
FCT_DM_shell_edges = np.copy(DM_shell_edges)
FCT_shell_multipliers = np.copy(shell_multipliers)
FCT_init_xys = np.copy(init_xyz)
FCT_neutrino_simulation_mass_eV = sim_setup['neutrino_simulation_mass_eV']*eV
FCT_zeds = np.copy(z_int_steps)

# note: now that variables are loaded into memory, the function.py will work.
#? probably not a good final solution, perhaps scripts have functions above, 
#? which they will use? they should be unique between the analytical and 
#? numerical simulation types.
import shared.functions as fct



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
    if PRE.MW_HALO:
        grad_tot += fct.dPsi_dxi_NFW(
            x_i, z, rho0_MW, Mvir_MW, Rvir_MW, Rs_MW, 'MW'
            )
    if PRE.VC_HALO:
        grad_tot += fct.dPsi_dxi_NFW(
            x_i, z, rho0_VC, Mvir_VC, Rvir_VC, Rs_VC, 'VC'
            )
    if PRE.AG_HALO:
        grad_tot += fct.dPsi_dxi_NFW(
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
        fun=EOMs, t_span=[S_STEPS[0], S_STEPS[-1]], t_eval=S_STEPS,
        y0=y0, method=SOLVER, vectorized=True
        )
    
    np.save(f'{TEMP_DIR}/nu_{int(Nr)}.npy', np.array(sol.y.T))


if __name__ == '__main__':
    start = time.perf_counter()

    # Draw initial velocities.
    ui = fct.init_velocities(PRE.PHIs, PRE.THETAs, PRE.MOMENTA)
    
    # Combine vectors and append neutrino particle number.
    y0_Nr = np.array(
        [np.concatenate((X_SUN, ui[k], [k+1])) for k in range(PRE.NUS)]
        )

    # Run simulation on multiple cores, in batches.
    # (important for other solvers (e.g. Rk45), due to memory increase)
    batch_size = 10000
    ticks = np.arange(0, PRE.NUS/batch_size, dtype=int)
    for i in ticks:

        id_min = (i*batch_size) + 1
        id_max = ((i+1)*batch_size) + 1
        print(f'From {id_min} to and incl. {id_max-1}')

        if i == 0:
            id_min = 0

        with ProcessPoolExecutor(PRE.SIM_CPUs) as ex:
            ex.map(backtrack_1_neutrino, y0_Nr[id_min:id_max])

        print(f'Batch {i+1}/{len(ticks)} done!')


    # Compactify all neutrino vectors into 1 file.
    Ns = np.arange(PRE.NUS, dtype=int)
    nus = np.array([np.load(f'{TEMP_DIR}/nu_{Nr+1}.npy') for Nr in Ns])
    oname = f'{PRE.NUS}nus_smooth_{PRE.HALOS}_{SOLVER}'
    np.save(f'{PRE.OUT_DIR}/{oname}.npy', nus)

    # Remove temporary folder with all individual neutrino files.
    shutil.rmtree(TEMP_DIR)   

    seconds = time.perf_counter()-start
    minutes = seconds/60.
    hours = minutes/60.
    print(f'Time min/h: {minutes} min, {hours} h.')