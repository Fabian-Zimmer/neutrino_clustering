from shared.preface import *
from shared.shared_functions import *

halo_indices = np.load('L025N752/DMONLY/SigmaConstant00/single_halos/halo_batch_0.6-2.0x1e+12.0_Msun_indices.npy')

halo_params = np.load('L025N752/DMONLY/SigmaConstant00/single_halos/halo_batch_0.6-2.0x1e+12.0_Msun_params.npy')

print(halo_indices)
print(halo_params)