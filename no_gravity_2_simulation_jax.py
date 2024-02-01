# Save how much memory is used by OS and not available for script.
import psutil
MB_UNIT = 1024**2
OS_MEM = (psutil.virtual_memory().used)

from shared.preface import *
from shared.shared_functions import *
total_start = time.perf_counter()

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', required=True)
parser.add_argument('-st', '--sim_type', required=True)
parser.add_argument('-mg', '--mass_gauge', required=True)
parser.add_argument('-ml', '--mass_lower', required=True)
parser.add_argument('-mu', '--mass_upper', required=True)
parser.add_argument('-hn', '--halo_num', required=True)
parser.add_argument('-sh', '--shells', required=False)
parser.add_argument(
    '--upto_Rvir', required=True, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    '--gravity', required=True, action=argparse.BooleanOptionalAction
)
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
neutrinos = sim_setup['neutrinos']

# Initial distance from center (Earth-GC) cell must approximate.
init_dis = sim_setup['initial_haloGC_distance']


# Load arrays.
z_int_steps = np.load(f'{args.directory}/z_int_steps.npy')
s_int_steps = np.load(f'{args.directory}/s_int_steps.npy')
neutrino_massrange = np.load(f'{args.directory}/neutrino_massrange_eV.npy')*eV
DM_shell_edges = np.load(f'{args.directory}/DM_shell_edges.npy')  # *kpc already
shell_multipliers = np.load(f'{args.directory}/shell_multipliers.npy')


# Load constants and arrays, which some functions below need.
FCT_zeds = np.copy(z_int_steps)


### ==================================== ###
### Define all necessary functions here. ###
### ==================================== ###
# Defined in order of usage.

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

    if 'all_sky' in sim_type:
        return num_densities
    else:
        np.save(f'{out_file}', num_densities)


# Make temporary folder to store files, s.t. parallel runs don't clash.
# rand_code = ''.join(
#     random.choices(string.ascii_uppercase + string.digits, k=4)
# )
# temp_dir = f'{args.directory}/temp_data_{rand_code}'
# os.makedirs(temp_dir)

temp_dir = f'{args.directory}/temp_data'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


print(f'********Numerical Simulation: Mode={args.sim_type}********')
print(f'NO GRAVITY')
print('***********************************')

@jax.jit
def EOMs_noGravity(s_val, y, args):

    # Initialize vector.
    _, u_i = y

    # Hamilton eqns. for integration (global minus, s.t. we go back in time).
    dyds = -jnp.array([
        u_i, jnp.zeros(3)
    ])

    return dyds


# ODE solver setup
term = diffrax.ODETerm(EOMs_noGravity)
# solver = diffrax.Tsit5()
solver = diffrax.Dopri5()
t0 = s_int_steps[0]
t1 = s_int_steps[-1]

dt0 = (s_int_steps[0] + s_int_steps[1])/2
saveat = diffrax.SaveAt(ts=jnp.array(s_int_steps))
stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-1)

@jax.jit
def backtrack_1_neutrino(y0_Nr):
    """Simulate trajectory of 1 neutrino."""

    # Initial vector
    y0 = jnp.array([y0_Nr[0:3], y0_Nr[3:6]])

    # Solutions to coupled EOMs
    sol = diffrax.diffeqsolve(
        term, solver, 
        t0=t0, t1=t1, dt0=dt0, y0=y0, max_steps=10000,
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=None
    )
    
    sol_vector = sol.ys.reshape(100,6)

    # np.save(f'{temp_dir}/nu_{int(Nr)}.npy', np.array(sol.y.T))
    return jnp.array(sol_vector)


halo_num = int(args.halo_num)
for halo_j in range(halo_num):
    grav_time = time.perf_counter()

    # ========================================= #
    # Run simulation for current halo in batch. #
    # ========================================= #

    if 'benchmark' in args.sim_type:
        end_str = 'benchmark_halo'
    elif args.gravity is False:
        end_str = f'no_gravity'
    else:
        end_str = f'halo{halo_j+1}'
    
 
    # Initial position (Earth).
    # Needs to be without kpc units (thus doing /kpc) for simulation start.
    init_xyz = np.array([float(init_dis), 0., 0.])
    np.save(f'{args.directory}/init_xyz_{end_str}.npy', init_xyz)

    # Display parameters for simulation.
    print(f'***Running simulation: mode = {args.sim_type}***')
    print(f'halo={halo_j+1}/{halo_num}, CPUs={CPUs_sim}')

    sim_start = time.perf_counter()

    if args.sim_type in ('single_halos', 'benchmark'):
        
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
        vname = f'neutrino_vectors_numerical_{end_str}'
        for i, elem in enumerate(split):
            np.save(
                f'{args.directory}/{vname}_batch{i+1}.npy', elem
            )

        # Compute the number densities.
        dname = f'number_densities_numerical_{end_str}'
        out_file = f'{args.directory}/{dname}.npy'
        number_densities_mass_range(
            neutrino_vectors[...,3:6], neutrino_massrange, out_file
        )

    else:
        # note: change manually if you want to save vectors
        all_sky_small = True

        pix_sr_sim = sim_setup['pix_sr']

        # Load initial velocities for all_sky mode.
        # Deleted later due to size, unless simulation is manually downscaled.
        ui = np.load(f'{args.directory}/initial_velocities.npy')
        # ui = ui[::10,...]

        # Empty list to append number densitites of each angle coord. pair.
        nu_densities = []

        # Empty list to append first and last vectors for all coord. pairs
        if all_sky_small:
            # Pre-allocate array
            nus_per_pix = int(sim_setup['momentum_num'])
            Npix = int(sim_setup['Npix'])
            final_shape = (nus_per_pix*Npix, 2, 6)
            nu_vectors = np.empty(final_shape)

        if args.gravity:

            for cp, ui_elem in enumerate(ui):

                print(f'Coord. pair {cp+1}/{len(ui)}')

                # Combine vectors and append neutrino particle number.
                y0_Nr = np.array([np.concatenate(
                    (init_xyz, ui_elem[k], [k+1])) for k in range(len(ui_elem))
                ])

                # Run simulation on multiple cores.
                with ProcessPoolExecutor(CPUs_sim) as ex:
                    ex.map(backtrack_1_neutrino, y0_Nr)

                # Compactify all neutrino vectors into 1 file.
                neutrino_vectors = np.array(
                    [
                        np.load(f'{temp_dir}/nu_{i+1}.npy') 
                        for i in range(len(ui_elem))
                    ]
                )

                # Compute the number densities.
                nu_densities.append(
                    number_densities_mass_range(
                        neutrino_vectors[...,3:6], 
                        neutrino_massrange, 
                        sim_type=args.sim_type,
                        pix_sr=pix_sr_sim

                    )
                )

                # Save first and last vector elements for all_sky_small version
                if all_sky_small:
                    
                    # Select and combine first and last sim vectors
                    z0_elems = neutrino_vectors[:,0,:].reshape(-1,1,6)
                    z4_elems = neutrino_vectors[:,-1,:].reshape(-1,1,6)
                    combined = np.concatenate((z0_elems, z4_elems), axis=1)

                    # Fill elements of pre-allocated neutrino vectors array
                    start_idx = cp * nus_per_pix
                    end_idx = start_idx + nus_per_pix
                    nu_vectors[start_idx:end_idx,:,:] = combined


        else:

            def process_cp(cp_ui_tuple, all_sky_small=False):
                cp, ui_elem = cp_ui_tuple
                y0_Nr = np.array(
                    [np.concatenate((init_xyz, ui_elem[k], [k+1])) for k in range(len(ui_elem))]
                )
                # Single core simulation for all neutrinos for this cp.
                results = list(map(backtrack_1_neutrino, y0_Nr))

                # Compactify results
                neutrino_vectors = np.array(results)

                # Compute number densities
                nu_density = number_densities_mass_range(
                    neutrino_vectors[..., 3:6],
                    neutrino_massrange,
                    sim_type=args.sim_type,
                    pix_sr=pix_sr_sim,
                )

                if all_sky_small:
                    z0_elems = neutrino_vectors[:, 0, :].reshape(-1, 1, 6)
                    z4_elems = neutrino_vectors[:, -1, :].reshape(-1, 1, 6)
                    combined = np.concatenate((z0_elems, z4_elems), axis=1)
                    return cp, nu_density, combined

                return cp, nu_density, None
            
            
            with ProcessPoolExecutor(CPUs_sim) as ex:
                if all_sky_small:
                    ordered_results = list(ex.map(partial(process_cp, all_sky_small=all_sky_small), enumerate(ui)))
                else:
                    ordered_results = list(ex.map(process_cp, enumerate(ui)))


            # Sort results based on cp and extract nu_densities and combined vectors
            ordered_results.sort(key=lambda x: x[0])
            nu_densities = [res[1] for res in ordered_results]


            if all_sky_small:
                for cp, res in enumerate(ordered_results):
                    start_idx = cp * nus_per_pix
                    end_idx = start_idx + nus_per_pix
                    nu_vectors[start_idx:end_idx, :, :] = res[2]


        if all_sky_small:

            # Save all sky neutrino vectors for current halo
            # note: Max. possible is nside=8, shape (1_000*768, 2, 6), ~70 MB
            vname = f'neutrino_vectors_numerical_{end_str}_all_sky'
            np.save(f'{args.directory}/{vname}.npy', np.array(nu_vectors))


        # Save number densities for current halo
        dname = f'number_densities_numerical_{end_str}_all_sky'
        np.save(f'{args.directory}/{dname}.npy', np.array(nu_densities))


    sim_time = time.perf_counter()-sim_start
    print(f'Sim time: {sim_time/60.} min, {sim_time/(60**2)} h.')
    
    if 'benchmark' in args.sim_type:
        break


# Remove temporary folder.
# shutil.rmtree(temp_dir)

total_time = time.perf_counter()-total_start
print(f'Total time: {total_time/60.} min, {total_time/(60**2)} h.')