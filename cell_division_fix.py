from shared.preface import *
import shared.functions as fct


# Parameters.
snap = '0036'
m0 = '2.59e+11'

# Overwrite global DM_LIM parameter, to manually set rounds of cell division.
# DM_LIM = 100000  # round(s): 0
# DM_LIM = 50000   # round(s): 1
# DM_LIM = 40000   # round(s): 2
# DM_LIM = 10000   # round(s): 4
# DM_LIM = 1000    # round(s): 8

# Initial grid and DM positions.
grid = fct.grid_3D(GRID_L, GRID_S)
init_cc = np.expand_dims(grid, axis=1)
DM_raw = np.load(
    f'CubeSpace/DM_positions_{SIM_ID}_snapshot_{snap}_{m0}Msun.npy'
)*kpc
DM_pos = np.expand_dims(DM_raw, axis=0)
DM_ready = np.repeat(DM_pos, len(init_cc), axis=0)
print('Input data shapes', init_cc.shape, DM_ready.shape)

cell_division_count = fct.cell_division(
    init_cc, DM_ready, GRID_S, DM_LIM, 
    stable_cc=None, sim=SIM_ID, snap_num=snap
)
print(f'cell division rounds: {cell_division_count}')

# Output.
adapted_cc = np.load(
    f'CubeSpace/adapted_cc_{SIM_ID}_snapshot_{snap}.npy')
cell_gen = np.load(
    f'CubeSpace/cell_gen_{SIM_ID}_snapshot_{snap}.npy')
cell_com = np.load(
    f'CubeSpace/cell_com_{SIM_ID}_snapshot_{snap}.npy')
DM_count = np.load(
    f'CubeSpace/DM_count_{SIM_ID}_snapshot_{snap}.npy')

print('Shapes of output files:', adapted_cc.shape, cell_gen.shape, cell_com.shape, DM_count.shape)

print('Total DM count across all cells:', DM_count.sum())