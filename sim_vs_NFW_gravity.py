from shared.preface import *
import shared.functions as fct


DM = np.load(f'sim_data/DM_positions_halos_M12.npy')[::1]
print(f'DM particles: {len(DM)}')
snap = '0036'

# Initialize grid and DM positions.
grid = fct.grid_3D(GRID_L, GRID_S)

z = 0
NFW_grav = np.array([
    fct.dPsi_dxi_NFW(x_i, z, rho0_MW, Mvir_MW, Rvir_MW, Rs_MW, 'MW')
    for x_i in grid
])
NFW_grav_mag = np.sqrt(np.sum(NFW_grav**2, axis=1))


grid = np.expand_dims(grid, axis=1)
DM = np.expand_dims(DM, axis=0)
DM = np.repeat(DM, len(grid), axis=0)

fig, ax = plt.subplots(1,1, figsize=(8,8))

ranges = [GRID_S/2., GRID_S/np.sqrt(2), None]
labels = ['Inscribed sphere', 'Circumscribed sphere', 'Inf. sphere']
for rangeX, l0 in zip(ranges, labels):

    fct.cell_gravity_3D(grid, DM*kpc, rangeX, DM_SIM_MASS, snap)
    dPsi_grid_rangeX = np.load(f'CubeSpace/dPsi_grid_snapshot_{snap}.npy')
    mags_rangeX = np.sqrt(np.sum(dPsi_grid_rangeX**2, axis=1))

    # Sort cells by distance from center (0,0,0).
    grid_dis = np.sqrt(np.sum(grid**2, axis=2)).flatten()
    dis_ind = grid_dis.argsort()
    grid_dis = grid_dis[dis_ind]
    mags_rangeX = mags_rangeX[dis_ind]
    NFW_grav_sort = NFW_grav_mag[dis_ind]

    diff = (NFW_grav_sort-mags_rangeX)/NFW_grav_sort

    if rangeX is None:
        rangeX = np.inf

    ax.scatter(
        grid_dis/kpc, diff, s=5, alpha=0.8, 
        label=f'grav_range {np.round(rangeX/kpc,1)} kpc ({l0})'
        )

ax.set_title(
    f'Difference of sim vs. NFW (>1 stronger, 1 same, 0 bad)'
    '\n'
    'Multiple dots for each x-axis point, since multiple cells share same distance'
    )
ax.set_xlabel(f'Cell distance from center (kpc)')
ax.set_ylabel(f'sim grav. strength w.r.t NFW')

def y_fmt_here(value, tick_number):
    return np.round(1-value,1)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt_here))

# ax.set_ylim(-0.5,1)
plt.legend(loc='lower right')
plt.savefig('figures/sim_vs_NFW_gravity.pdf')