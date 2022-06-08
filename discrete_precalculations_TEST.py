from shared.preface import *
import shared.functions as fct


### Do the following for all snapshots:

## See SpaceCubes.ipynb cells for each step.

# 1. Read snapshot.
folder = SIM_DATA
snaps = h5py.File(f'{folder}/snapshot_0036.hdf5')

# 2. Create file with all DM particle positions, then load it.
fct.read_DM_positions_randomHalo(which_halos='halos', mass_select=12)
DM_pos = np.load('sim_data/DM_positions_halos_M12.npy')

# 3. Construct spatial grid based on DM distribution (uniform for now).
cell_coords = fct.grid_3D(GRID_L, GRID_S)
#NOTE: Grid should extend until maximum distance neutrinos travel,
#NOTE: but it should still work with min. grid. for now

start = time.perf_counter()

# 4. Calculate gravity in each cell.
dPsi_grid = np.empty(cell_coords.shape)
for i, coords in enumerate(cell_coords):
    dPsi_grid[i] = fct.cell_gravity(coords, DM_pos, GRAV_RANGE, DM_SIM_MASS)

seconds = time.perf_counter()-start
minutes = seconds/60.
hours = minutes/60.
print(f'Time sec/min/h: {seconds} sec, {minutes} min, {hours} h.')

# 5. Save: 
#   - Cell (center) x,y,z positions.
np.save('sim_data/cell_coords.npy', cell_coords)
#   - Derivative grid.
np.save('sim_data/dPsi_grid', dPsi_grid)