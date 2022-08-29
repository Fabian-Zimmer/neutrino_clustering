from shared.preface import *
import shared.functions as fct

snap = '0036'
sim_ID = 'L006N188'
m0s = ['1.89e+12', '4.32e+11', '2.59e+11']
projs = [0,1,2]

for m0, proj in zip(m0s, projs):

    # Comment out to iterate over all halos in m0s.
    if proj != 2:
        continue

    # Generate files with positions of DM particles
    fct.read_DM_positions(
        which_halos='halos', mass_select=12,  # unnecessary when giving index...
        random=False, snap_num=snap, sim=sim_ID, 
        halo_index=int(proj), init_m=m0
    )

    # Read in DM particle positions.
    DM_raw = np.load(
        f'CubeSpace/DM_positions_{sim_ID}_snapshot_{snap}_{m0}Msun.npy'
    )[::1]*kpc
    # print(len(DM))


adapted_cc = np.load(
    f'CubeSpace/adapted_cc_{SIM_ID}_snapshot_{snap}.npy'
)
cell_com = np.load(
    f'CubeSpace/cell_com_{SIM_ID}_snapshot_{snap}.npy'
)
DM_count = np.load(
    f'CubeSpace/DM_count_{SIM_ID}_snapshot_{snap}.npy'
)
DM_pos = np.expand_dims(DM_raw, axis=0)
adapted_DM = np.repeat(DM_pos, len(adapted_cc), axis=0)


fig, ax = plt.subplots(1,1, figsize=(8,8))


ranges = [GRID_S/2., GRID_S/np.sqrt(2)]
labels = ['Inscribed sphere', 'Circumscribed sphere']
for rangeX, l0 in zip(ranges, labels):

    fct.cell_gravity_3D(
        adapted_cc, cell_com, adapted_DM, DM_count, 
        None, DM_SIM_MASS, snap
    )
    dPsi_grid_None = np.load(f'CubeSpace/dPsi_grid_snapshot_{snap}.npy')
    mags_None = np.sqrt(np.sum(dPsi_grid_None**2, axis=1))

    fct.cell_gravity_3D(
        adapted_cc, cell_com, adapted_DM, DM_count, 
        rangeX, DM_SIM_MASS, snap
    )
    dPsi_grid_rangeX = np.load(f'CubeSpace/dPsi_grid_snapshot_{snap}.npy')
    mags_rangeX = np.sqrt(np.sum(dPsi_grid_rangeX**2, axis=1))

    # Sort cells by distance from center (0,0,0).
    grid_dis = np.sqrt(np.sum(grid**2, axis=2)).flatten()
    dis_ind = grid_dis.argsort()
    grid_dis = grid_dis[dis_ind]
    mags_None = mags_None[dis_ind]
    mags_rangeX = mags_rangeX[dis_ind]

    diff = (mags_None-mags_rangeX)/mags_None
    ax.scatter(
        grid_dis/kpc, diff, s=5, alpha=0.8, 
        label=f'grav_range {np.round(rangeX/kpc,1)} kpc ({l0})'
        )

ax.set_title(
    f'Difference of limited vs. unlimited grav_range (1 good, 0 bad)'
    '\n'
    'Multiple dots for each x-axis point, since multiple cells share same distance'
    )
ax.set_xlabel(f'Cell distance from center (kpc)')
ax.set_ylabel(f'grav. strength w.r.t "infinite" (all DM particles) range')

def y_fmt_here(value, tick_number):
    return np.round(1-value,1)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt_here))

plt.legend(loc='lower right')
plt.show()