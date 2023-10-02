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
nums_snaps = np.load(f'{args.directory}/nums_snaps.npy')
zeds_snaps = np.load(f'{args.directory}/zeds_snaps.npy')

z_int_steps = np.load(f'{args.directory}/z_int_steps.npy')
s_int_steps = np.load(f'{args.directory}/s_int_steps.npy')
neutrino_massrange = np.load(f'{args.directory}/neutrino_massrange_eV.npy')*eV
DM_shell_edges = np.load(f'{args.directory}/DM_shell_edges.npy')  # *kpc already
shell_multipliers = np.load(f'{args.directory}/shell_multipliers.npy')


# Load constants and arrays, which some functions below need.
FCT_DM_shell_edges = np.copy(DM_shell_edges)
FCT_shell_multipliers = np.copy(shell_multipliers)
FCT_zeds = np.copy(z_int_steps)


### ==================================== ###
### Define all necessary functions here. ###
### ==================================== ###
# Defined in order of usage.


@nb.njit
def nu_in_which_cell(x_i, cell_coords, cell_gens, init_GRID_S):

    # Number of cells.
    num_cells = cell_coords.shape[0]

    # "Center" the neutrino coordinates on the cell coordinates.
    x_cent = x_i - cell_coords.reshape(num_cells, 3)

    # All cell lengths. Limit for the largest cell is GRID_S/2, not just 
    # GRID_S, therefore the cell_gen+1 !
    cell_lens = init_GRID_S/(2**(cell_gens+1))

    # Find index of cell in which neutrino is enclosed.
    in_cell = np.asarray(
        (np.abs(x_cent[...,0]) < cell_lens) & 
        (np.abs(x_cent[...,1]) < cell_lens) & 
        (np.abs(x_cent[...,2]) < cell_lens)
    )
    cell_idx = np.argwhere(in_cell).flatten()[0]
    cell_len = cell_lens[cell_idx]
    cell_ccs = cell_coords[cell_idx, :]

    return cell_idx, cell_len, cell_ccs


@nb.njit
def outside_gravity_quadrupole(x_i, com_halo, DM_sim_mass, DM_num, QJ_abs):

    ### ----------- ###
    ### Quadrupole. ###
    ### ----------- ###

    # Center neutrino on c.o.m. of halo and get distance.
    x_i -= com_halo
    r_i = np.sqrt(np.sum(x_i**2))

    # Permute order of coords by one, i.e. (x,y,z) -> (z,x,y).
    x_i_roll = np.empty_like(x_i)
    x_i_roll[0] = x_i[2]
    x_i_roll[1] = x_i[0]
    x_i_roll[2] = x_i[1]

    # Terms appearing in the quadrupole term.
    QJ_aa = QJ_abs[0]
    QJ_ab = QJ_abs[1]

    # Factors of 2 are for the symmetry of QJ_ab elements.
    term1_aa = np.sum(QJ_aa*x_i, axis=0)
    term1_ab = np.sum(2*QJ_ab*x_i_roll, axis=0)
    term1 = (term1_aa+term1_ab)/r_i**5

    term2_pre = 5*x_i/(2*r_i**7)
    term2_aa = np.sum(QJ_aa*x_i**2, axis=0)
    term2_ab = np.sum(2*QJ_ab*x_i*x_i_roll, axis=0)
    term2 = term2_pre*(term2_aa+term2_ab)

    dPsi_multipole_cells = G*DM_sim_mass*(-term1+term2)

    ### --------- ###
    ### Monopole. ###
    ### --------- ###

    dPsi_monopole_cells = G*DM_num*DM_sim_mass*x_i/r_i**3

    # Total outside gravity gradient.
    gradient_out = dPsi_multipole_cells+dPsi_monopole_cells

    # Acceleration is negative value of (grav. pot.) gradient.
    return -gradient_out


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
# data_dir = f'{args.directory}/temp_data_{rand_code}'
# os.makedirs(data_dir)

# Parent directory of current sim folder
parent_dir = str(pathlib.Path(args.directory).parent)

# All precalculations are stored here
data_dir = f'{parent_dir}/data_precalculations'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def M12_to_M12X(M12_val):
    return np.log(M12_val*10.**12)/np.log(10.)

# hname = f'1e+{args.mass_gauge}_pm{args.mass_range}Msun'
hname = f'{args.mass_lower}-{args.mass_upper}x1e+{args.mass_gauge}_Msun'
mass_neg = np.abs(float(args.mass_gauge) - M12_to_M12X(float(args.mass_lower)))
mass_pos = np.abs(float(args.mass_gauge) - M12_to_M12X(float(args.mass_upper)))
halo_batch_indices(
    z0_snap_4cif, 
    float(args.mass_gauge), mass_neg, mass_pos,
    'halos', int(args.halo_num), 
    hname, box_file_dir, args.directory
)
halo_batch_IDs = np.load(f'{args.directory}/halo_batch_{hname}_indices.npy')
halo_batch_params = np.load(f'{args.directory}/halo_batch_{hname}_params.npy')
halo_num = len(halo_batch_params)

print(f'********Numerical Simulation: Mode={args.sim_type}********')
print('Halo batch params (Rvir,Mvir,cNFW):')
print(halo_batch_params)
print('***********************************')

def EOMs(s_val, y):

    # Initialize vector.
    x_i, u_i = np.reshape(y, (2,3))

    # Switch to "numerical reality" here.
    x_i *= kpc
    u_i *= (kpc/s)

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
        
        cell_idx, cell_len0, cell_cc0 = nu_in_which_cell(
            x_i, cell_grid, cell_gens, snap_GRID_L
        )
        grad_tot = dPsi_grid[cell_idx,:]

    # Neutrino outside cell grid.
    else:

        DM_com = snaps_DM_com[idx]
        DM_num = snaps_DM_num[idx]
        QJ_abs = snaps_QJ_abs[idx]
        grad_tot = outside_gravity_quadrupole(
            x_i, DM_com, DM_mass, DM_num, QJ_abs
        )

    # Switch to "physical reality" here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)

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
    
    np.save(f'{data_dir}/nu_{int(Nr)}.npy', np.array(sol.y.T))


for halo_j, halo_ID in enumerate(halo_batch_IDs):
    grav_time = time.perf_counter()

    # ========================================= #
    # Run simulation for current halo in batch. #
    # ========================================= #

    if 'benchmark' in args.sim_type:
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
    cell_ccs_kpc = cell_ccs/kpc
    cell_dis = np.linalg.norm(cell_ccs_kpc, axis=-1)

    # Take first cell, which is in Earth-like position (there can be multiple).
    # Needs to be without kpc units (thus doing /kpc) for simulation start.
    init_xyz = cell_ccs[np.abs(cell_dis - init_dis).argsort()][0]/kpc.flatten()
    np.save(f'{args.directory}/init_xyz_{end_str}.npy', init_xyz)

    # Display parameters for simulation.
    print(f'***Running simulation: mode = {args.sim_type}***')
    print(f'halo={halo_j+1}/{halo_num}, CPUs={CPUs_sim}')

    sim_start = time.perf_counter()

    if args.sim_type in ('single_halos', 'spheres', 'benchmark'):
        
        # Special case for spheres, such that files get named appropriately.
        if args.sim_type == 'spheres':
            end_str = f'{end_str}_{args.shells}shells'

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

        # Load initial velocities for all_sky mode. Note that this array is 
        # (mostly, if Nside is larger than 2**1) not github compatible, and 
        # will be deleted afterwards.
        ui = np.load(f'{args.directory}/initial_velocities.npy')

        # Empty list to append number densitites of each angle coord. pair.
        nu_densities = []

        # Empty list to append first and last vectors for all coord. pairs
        if all_sky_small:
            # Pre-allocate array
            final_shape = (1_000*768, 2, 6)
            nu_vectors = np.empty(final_shape)


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
                start_idx = cp*1_000
                end_idx = start_idx+1_000
                nu_vectors[start_idx:end_idx,:,:] = combined

                # Save all sky neutrino vectors for current halo
                # For nside=8 has shape (1_000*768, 2, 6) and is ~70 MB
                vname = f'neutrino_vectors_numerical_{end_str}_all_sky'
                np.save(f'{args.directory}/{vname}.npy', np.array(nu_vectors))


        # Save number densities for current halo
        dname = f'number_densities_numerical_{end_str}_all_sky'
        np.save(f'{args.directory}/{dname}.npy', np.array(nu_densities))


    sim_time = time.perf_counter()-sim_start
    print(f'Sim time: {sim_time/60.} min, {sim_time/(60**2)} h.')
    
    if 'benchmark' in args.sim_type:
        break


total_time = time.perf_counter()-total_start
print(f'Total time: {total_time/60.} min, {total_time/(60**2)} h.')