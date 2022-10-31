from shared.preface import *
import shared.functions as fct

folder = f'{os.getcwd()}/L012N376/DMONLY/CDM_TF50'

fct.delete_temp_data(f'{folder}/nu_*.npy')
fct.delete_temp_data(f'{folder}/NrDM_*.npy')
fct.delete_temp_data(f'{folder}/fin_grid_*.npy')
fct.delete_temp_data(f'{folder}/DM_count_*.npy')
fct.delete_temp_data(f'{folder}/DM_pos_*.npy')
fct.delete_temp_data(f'{folder}/cell_com_*.npy')
fct.delete_temp_data(f'{folder}/cell_gen_*.npy')
fct.delete_temp_data(f'{folder}/CoP_*.npy')
fct.delete_temp_data(f'{folder}/snaps_GRID_L_*.npy')
fct.delete_temp_data(f'{folder}/halo_params_*.npy')
fct.delete_temp_data(f'{folder}/halo_batch_*.npy')
fct.delete_temp_data(f'{folder}/dPsi_grid_*.npy')


folder = f'{os.getcwd()}/L012N376/DMONLY/SigmaConstant00'

fct.delete_temp_data(f'{folder}/nu_*.npy')
fct.delete_temp_data(f'{folder}/NrDM_*.npy')
fct.delete_temp_data(f'{folder}/fin_grid_*.npy')
fct.delete_temp_data(f'{folder}/DM_count_*.npy')
fct.delete_temp_data(f'{folder}/DM_pos_*.npy')
fct.delete_temp_data(f'{folder}/cell_com_*.npy')
fct.delete_temp_data(f'{folder}/cell_gen_*.npy')
fct.delete_temp_data(f'{folder}/CoP_*.npy')
fct.delete_temp_data(f'{folder}/snaps_GRID_L_*.npy')
fct.delete_temp_data(f'{folder}/halo_params_*.npy')
fct.delete_temp_data(f'{folder}/halo_batch_*.npy')
fct.delete_temp_data(f'{folder}/dPsi_grid_*.npy')

folder = f'{os.getcwd()}/L006N188/DMONLY'

fct.delete_temp_data(f'{folder}/dPsi_grid_*.npy')