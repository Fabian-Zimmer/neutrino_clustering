from shared.preface import *
from shared.shared_functions import *
from shared.plot_class import analyze_simulation_outputs


# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-sd', '--sim_directory', required=True)
args = parser.parse_args()

# Sim things.
with open(f'{args.sim_directory}/sim_parameters.yaml', 'r') as file:
    sim_setup = yaml.safe_load(file)
init_dis = sim_setup['initial_haloGC_distance']
init_xyz = np.array([init_dis, 1e-6, 1e-6])

# Box things.
with open(f'{args.sim_directory}/box_parameters.yaml', 'r') as file:
    box_setup = yaml.safe_load(file)
box_file_dir = box_setup['File Paths']['Box File Directory']
halo_batch_IDs = np.load(
    glob.glob(f'{args.sim_directory}/halo_batch*indices.npy')[0]
)

objects = (
    # 'NFW_halo', 
    'box_halos', 
    # 'analytical_halo'
)

# Initialize Analysis class.
Analysis = analyze_simulation_outputs(
    sim_dir = args.sim_directory,
    objects = objects,
    sim_type = 'all_sky',
    shells = ''
)

# Generate DM projection for all halos in sim directory.
for halo_j, halo_ID in enumerate(halo_batch_IDs):

    halo_j += 1
    print(f'Plot for halo {halo_j}')

    # Generate progenitor index array for current halo.
    with h5py.File(f'{args.sim_directory}/MergerTree.hdf5') as tree:
        prog_IDs = tree['Assembly_history/Progenitor_index'][halo_ID,:]
        prog_IDs_np = np.array(np.expand_dims(prog_IDs, axis=1), dtype=int)

    # Load a DM particles of halo at z=0.
    DM_pos_orig, DM_com = read_DM_halo_index(
        '0036', int(prog_IDs_np[0]), '', box_file_dir, '', direct=True
    )

    # Generate plot.
    Analysis.plot_all_sky_map(
        halo=halo_j, DM_pos_orig=DM_pos_orig, Obs_pos_orig=init_xyz, Nside=2**4,
        nu_mass_eV=0.3
    )