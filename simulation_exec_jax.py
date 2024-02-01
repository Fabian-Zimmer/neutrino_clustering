from Shared.specific_CNB_sim import *


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
pars = parser.parse_args()


### ================================= ###
### Load box & simulation parameters. ###
### ================================= ###

# Box parameters.
with open(f'{pars.directory}/box_parameters.yaml', 'r') as file:
    box_setup = yaml.safe_load(file)

box_file_dir = box_setup['File Paths']['Box File Directory']
DM_mass = box_setup['Content']['DM Mass [Msun]']*Params.Msun
z0_snap_4cif = box_setup['Content']['z=0 snapshot']

# Simulation parameters.
with open(f'{pars.directory}/sim_parameters.yaml', 'r') as file:
    sim_setup = yaml.safe_load(file)

CPUs_sim = sim_setup['CPUs_simulations']
integration_solver = sim_setup['integration_solver']
neutrinos = sim_setup['neutrinos']

# Initial distance from center (Earth-GC) cell must approximate.
init_dis = sim_setup['initial_haloGC_distance']


# Load arrays.
nums_snaps = np.load(f'{pars.directory}/nums_snaps.npy')
zeds_snaps = np.load(f'{pars.directory}/zeds_snaps.npy')

z_int_steps = np.load(f'{pars.directory}/z_int_steps.npy')
s_int_steps = np.load(f'{pars.directory}/s_int_steps.npy')
neutrino_massrange = np.load(f'{pars.directory}/neutrino_massrange_eV.npy')*Params.eV

# Load constants and arrays, which some functions below need.
FCT_zeds = np.copy(z_int_steps)


# @jax.jit(static_argnums=(3,))
def number_densities_for_mass_range(v_arr, m_arr, pix_sr, sim_type, fname, args):
    # Convert velocities to momenta.
    p_arr, _ = Physics.velocities_to_momenta(v_arr, m_arr, Params)

    nu_dens = Physics.number_density(p_arr[...,0], p_arr[...,-1], pix_sr, args)

    if sim_type == 'all_sky':
        return nu_dens
    else:
        jnp.save(f"{fname}", nu_dens)


def number_densities_mass_range(
    sim_vels, nu_masses, out_file=None, pix_sr=4*Params.Pi,
    average=False, m_start=0.01, z_start=0., sim_type='single_halos'
):
    
    # Convert velocities to momenta.
    p_arr, _ = Physics.velocities_to_momenta(sim_vels, nu_masses, Params)

    if average:
        inds = np.array(np.where(FCT_zeds >= z_start)).flatten()
        temp = [
            Physics.number_density(p_arr[...,0], p_arr[...,k], pix_sr, Params) for k in inds
        ]
        num_densities = np.mean(np.array(temp.T), axis=-1)
    else:
        num_densities = Physics.number_density(p_arr[...,0], p_arr[...,-1], pix_sr, Params)

    if 'all_sky' in sim_type:
        return num_densities
    else:
        np.save(f'{out_file}', num_densities)


# Make temporary folder to store files, s.t. parallel runs don't clash.
# rand_code = ''.join(
#     random.choices(string.ascii_uppercase + string.digits, k=4)
# )
# data_dir = f'{pars.directory}/temp_data_{rand_code}'
# os.makedirs(data_dir)

# Parent directory of current sim folder
parent_dir = str(pathlib.Path(pars.directory).parent)

# All precalculations are stored here
data_dir = f'{parent_dir}/data_precalculations'

# hname = f'1e+{pars.mass_gauge}_pm{pars.mass_range}Msun'
hname = f'{pars.mass_lower}-{pars.mass_upper}x1e+{pars.mass_gauge}_Msun'
mass_neg = np.abs(float(pars.mass_gauge) - SimData.M12_to_M12X(float(pars.mass_lower)))
mass_pos = np.abs(float(pars.mass_gauge) - SimData.M12_to_M12X(float(pars.mass_upper)))
SimData.halo_batch_indices(
    z0_snap_4cif, 
    float(pars.mass_gauge), mass_neg, mass_pos,
    'halos', int(pars.halo_num), 
    hname, box_file_dir, pars.directory
)
halo_batch_IDs = np.load(f'{pars.directory}/halo_batch_{hname}_indices.npy')
halo_batch_params = np.load(f'{pars.directory}/halo_batch_{hname}_params.npy')
halo_num = len(halo_batch_params)

print(f'********Numerical Simulation: Mode={pars.sim_type}********')
print('Halo batch params (Rvir,Mvir,cNFW):')
print(halo_batch_params)
print('***********************************')


def EOMs(s_val, y):

    # Initialize vector.
    x_i, u_i = np.reshape(y, (2,3))

    # Switch to "numerical reality" here.
    x_i *= Params.kpc
    u_i *= (Params.kpc/Params.s)

    # Find z corresponding to s via interpolation.
    z = np.interp(s_val, s_int_steps, z_int_steps)

    # Snapshot specific parameters.
    idx = np.abs(zeds_snaps - z).argmin()
    snap = nums_snaps[idx]
    snap_GRID_L = snaps_GRID_L[idx]

    # Neutrino inside cell grid.
    if np.all(np.abs(x_i) < snap_GRID_L):

        # Load files for current z, to find in which cell neutrino is. Then 
        # load gravity for that cell.
        snap_data = preloaded_data.get_data_for_snap(snap)
        dPsi_grid = snap_data['dPsi_grid']
        cell_grid = snap_data['cell_grid']
        cell_gens = snap_data['cell_gens']
        
        cell_idx, cell_len0, cell_cc0 = SimExec.nu_in_which_cell(
            x_i, cell_grid, cell_gens, snap_GRID_L
        )
        grad_tot = dPsi_grid[cell_idx,:]

    # Neutrino outside cell grid.
    else:

        DM_com = snaps_DM_com[idx]
        DM_num = snaps_DM_num[idx]
        QJ_abs = snaps_QJ_abs[idx]
        grad_tot = SimExec.outside_gravity_quadrupole(
            x_i, DM_com, DM_mass, DM_num, QJ_abs
        )

    # Switch to "physical reality" here.
    grad_tot /= (Params.kpc/Params.s**2)
    x_i /= Params.kpc
    u_i /= (Params.kpc/Params.s)

    # Hamilton eqns. for integration (global minus, s.t. we go back in time).
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
        y0=y0, method=integration_solver, vectorized=True,
        args=()
        )
    
    # np.save(f'{data_dir}/nu_{int(Nr)}.npy', np.array(sol.y.T))
    return np.array(sol.y.T)


for halo_j, halo_ID in enumerate(halo_batch_IDs):
    grav_time = time.perf_counter()

    # note: "Broken" halo, no DM position data at snapshot 0012.
    if halo_j == 19:
        continue

    # ========================================= #
    # Run simulation for current halo in batch. #
    # ========================================= #

    if 'benchmark' in pars.sim_type:
        end_str = 'benchmark_halo'
    else:
        end_str = f'halo{halo_j+1}'
    
    #! Important:
    # The loop ran from the earliest snapshot (z~4 for us) to the latest (z=0).
    # So the below arrays are in this order. Even though our simulation runs 
    # backwards in time, we can leave them like this, since the correct element 
    # gets picked with the idx routine in the EOMs function above.
    snaps_GRID_L = np.load(f'{data_dir}/snaps_GRID_L_{end_str}.npy')
    snaps_DM_num = np.load(f'{data_dir}/snaps_DM_num_{end_str}.npy')
    snaps_CC_num = np.load(f'{data_dir}/snaps_CC_num_{end_str}.npy')
    snaps_progID = np.load(f'{data_dir}/snaps_progID_{end_str}.npy')
    snaps_DM_com = np.load(f'{data_dir}/snaps_DM_com_{end_str}.npy')
    snaps_QJ_abs = np.load(f'{data_dir}/snaps_QJ_abs_{end_str}.npy')


    class PreloadedData:
        def __init__(self, halo_ID, nums_snaps, data_dir):
            self.halo_ID = halo_ID
            self.nums_snaps = nums_snaps
            self.data_dir = data_dir
            self.data_files = {}

            # Preload data
            for snap in self.nums_snaps:
                fname = f'origID{self.halo_ID}_snap_{snap}'
                self.data_files[fname] = {
                    'dPsi_grid': np.load(f'{self.data_dir}/dPsi_grid_{fname}.npy'),
                    'cell_grid': np.load(f'{self.data_dir}/fin_grid_{fname}.npy'),
                    'cell_gens': np.load(f'{self.data_dir}/cell_gen_{fname}.npy'),
                }

        def get_data_for_snap(self, snap):
            fname = f'origID{self.halo_ID}_snap_{snap}'
            data = self.data_files[fname]
            return {
                'dPsi_grid': data['dPsi_grid'].copy(),
                'cell_grid': data['cell_grid'].copy(),
                'cell_gens': data['cell_gens'].copy(),
            }

    # Create an instance of the PreloadedData class
    preloaded_data = PreloadedData(halo_ID, nums_snaps, data_dir)

    # Find a cell fitting initial distance criterium, then get (x,y,z) of that 
    # cell for starting position.

    # Load grid data and compute radial distances from center of cell centers.
    cell_ccs = np.squeeze(preloaded_data.get_data_for_snap('0036')['cell_grid'])
    cell_ccs_kpc = cell_ccs/Params.kpc
    cell_dis = np.linalg.norm(cell_ccs_kpc, axis=-1)

    # Take first cell, which is in Earth-like position (there can be multiple).
    # Needs to be without kpc units (thus doing /kpc) for simulation start.
    init_xyz = cell_ccs[np.abs(cell_dis - init_dis).argsort()][0]/Params.kpc.flatten()
    np.save(f'{pars.directory}/init_xyz_{end_str}.npy', init_xyz)

    # Display parameters for simulation.
    print(f'***Running simulation: mode = {pars.sim_type}***')
    print(f'halo={halo_j+1}/{halo_num}, CPUs={CPUs_sim}')

    sim_start = time.perf_counter()

    if pars.sim_type in ('single_halos', 'benchmark'):

        # Load initial velocities.
        ui = np.load(f'{pars.directory}/initial_velocities.npy')

        # Combine vectors and append neutrino particle number.
        y0_Nr = np.array(
            [np.concatenate((init_xyz, ui[i], [i+1])) for i in range(neutrinos)]
            )

        # Run simulation on multiple cores.
        with ProcessPoolExecutor(CPUs_sim) as ex:
            ex.map(backtrack_1_neutrino, y0_Nr)

        # Compactify all neutrino vectors into 1 file.
        neutrino_vectors = np.array(
            [np.load(f'{data_dir}/nu_{i+1}.npy') for i in range(neutrinos)]
        )

        # For these modes (i.e. not all_sky), save all neutrino vectors.
        # Split velocities and positions into 10k neutrino batches.
        # For reference: ndarray with shape (10_000, 100, 6) is  48 MB.
        batches = math.ceil(neutrinos/10_000)
        split = np.array_split(neutrino_vectors, batches, axis=0)
        vname = f'neutrino_vectors_numerical_{end_str}'
        for i, elem in enumerate(split):
            np.save(
                f'{pars.directory}/{vname}_batch{i+1}.npy', elem
            )

        # Compute the number densities.
        dname = f'number_densities_numerical_{end_str}'
        out_file = f'{pars.directory}/{dname}.npy'
        number_densities_for_mass_range(
            neutrino_vectors[...,3:6], neutrino_massrange, out_file
        )

    else:
        # note: change manually if you want to save vectors
        all_sky_small = True

        pix_sr_sim = sim_setup['pix_sr']

        # Load initial velocities for all_sky mode. Note that this array is 
        # (mostly, if Nside is larger than 2**1) not github compatible, and 
        # will be deleted afterwards.
        ui = np.load(f'{pars.directory}/initial_velocities.npy')

        # Empty list to append number densitites of each angle coord. pair.
        nu_densities = []

        # Empty list to append first and last vectors for all coord. pairs
        if all_sky_small:
            # Pre-allocate array
            nus_per_pix = int(sim_setup['momentum_num'])
            Npix = int(sim_setup['Npix'])
            final_shape = (nus_per_pix*Npix, 2, 6)
            nu_vectors = np.empty(final_shape)


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
            nu_density = number_densities_for_mass_range(
                neutrino_vectors[..., 3:6],
                neutrino_massrange,
                sim_type=pars.sim_type,
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

            # Save all sky neutrino vectors for current halo
            # note: Max. possible is nside=8, shape (1_000*768, 2, 6), ~70 MB
            vname = f'neutrino_vectors_numerical_{end_str}_all_sky'
            np.save(f'{pars.directory}/{vname}.npy', np.array(nu_vectors))

        '''
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
                    np.load(f'{data_dir}/nu_{i+1}.npy') 
                    for i in range(len(ui_elem))
                ]
            )

            # Compute the number densities.
            nu_densities.append(
                number_densities_for_mass_range(
                    neutrino_vectors[...,3:6], 
                    neutrino_massrange, 
                    sim_type=pars.sim_type,
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
                start_idx = cp*1_000
                end_idx = start_idx+1_000
                nu_vectors[start_idx:end_idx,:,:] = combined
        '''

        # Save number densities for current halo
        dname = f'number_densities_numerical_{end_str}_all_sky'
        np.save(f'{pars.directory}/{dname}.npy', np.array(nu_densities))


    sim_time = time.perf_counter()-sim_start
    print(f'Sim time: {sim_time/60.} min, {sim_time/(60**2)} h.')
    
    if 'benchmark' in pars.sim_type:
        break

    # '''

# Remove nu_* files, s.t. when testing it will show me if not produced.
# delete_temp_data(f'{temp_dir}/nu_*.npy')

# if pars.sim_type == 'all_sky':
    # Delete arrays not compatible with github file limit size.
    # delete_temp_data(f'{pars.directory}/initial_velocities.npy')

# Remove temporary folder.
# shutil.rmtree(temp_dir)

total_time = time.perf_counter()-total_start
print(f'Total time: {total_time/60.} min, {total_time/(60**2)} h.')
