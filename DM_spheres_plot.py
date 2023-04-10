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

# Initialize Analysis class.
Analysis = analyze_simulation_outputs(
    sim_dir = args.sim_directory,
    objects = '',
    sim_type = '',
    shells = ''
)

def make_snap_range(snap_start, snap_stop):
    snap_range = [f'{s:04d}' for s in range(snap_start, snap_stop+1)]
    return np.array(snap_range)

snap_range = make_snap_range(
    snap_start = 12,
    snap_stop  = 36
)

# Generate DM projection for all halos in sim directory.
for halo_j, halo_ID in enumerate(halo_batch_IDs):

    halo_j += 1
    print(f'3D plot for halo {halo_j}')


    # Generate plot.
    Analysis.plot_DM_3D(
        halo=halo_j, snap_range=snap_range, 
        zoom_lim=None, view_angle=None, shells=1
    )