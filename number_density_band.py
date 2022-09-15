from shared.preface import *
import shared.functions as fct

sim = 'L006N188'
snap = '0036'
mass_gauge = 11

fct.read_DM_halo_batch(sim, snap, mass_gauge, 1, 'halos')

halo_params = np.load(
    f'{sim}/halo_params_snap_{snap}_1e+{mass_gauge}Msun.npy'
)
print(halo_params)


for j in range(len(halo_params)):
    DM_positions = np.load(
        f'{sim}/DM_pos_snap_{snap}_1e+{mass_gauge}Msun_halo{j}.npy'
    ) 
    print(DM_positions.shape)



# Steps:
# 1. Generate all DM position files for whole batch with read_DM_halo_batch.
# 2. Start to loop over function, which takes in DM positions and initial grid.
    # 2.1 In loop: 