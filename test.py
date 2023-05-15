from shared.preface import *
from shared.shared_functions import *

snap = '0012'
prog_ID = 55
IDname = f'origID{prog_ID}_snap_{snap}'

sim_dir = f'L025N752/DMONLY/SigmaConstant00/single_halos'
with open(f'{sim_dir}/box_parameters.yaml', 'r') as file:
    box_setup = yaml.safe_load(file)
box_file_dir = box_setup['File Paths']['Box File Directory']


DM_pos, DM_com = read_DM_halo_index(
    snap, int(prog_ID), IDname, box_file_dir, '', direct=True
)

print(type(DM_pos), DM_pos.shape)