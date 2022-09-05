from shared.preface import *
import shared.functions as fct


# Parameters.
sim_id = 'L006N188'
snap_num = '0036'
z = 0
halo_type = 'halos'
mass_gauge = 11


if halo_type == 'halos':
    # Generate progenitor index list.
    # note on init_halo for L006N188 sim: 0 is ~1e12Msun, 1 & 2 are ~1e11Msun.
    init_halo = 0
    m0, prog_idx = fct.read_MergerTree(init_halo) 

    # Generate file for DM particles of chosen halo and get parameters.
    halo_cNFW, halo_rvir, halo_Mvir = fct.read_DM_positions(
        random=False, snap_num=snap_num, sim=sim_id, 
        halo_index=int(init_halo), init_m=m0, save_params=True
    )
    halo_rvir *= kpc
    halo_Mvir = 10**halo_Mvir * Msun

    DM_raw = np.load(
        f'CubeSpace/DM_positions_{sim_id}_snapshot_{snap_num}_{m0}Msun.npy'
    )*kpc

elif halo_type == 'subhalos':
    # Generate file for DM particles of chosen halo and get parameters.
    halo_cNFW, halo_rvir, halo_Mvir = fct.read_DM_positions(
        which_halos=halo_type, mass_select=mass_gauge, mass_range=1,
        random=True, snap_num=snap_num, sim=sim_id, 
        save_params=True
    )
    halo_rvir *= kpc
    halo_Mvir = 10**halo_Mvir * Msun

    DM_raw = np.load(
        f'CubeSpace/DM_positions_{halo_type}_M{mass_gauge}.npy'
    )*kpc

print(
    f'Halo parameters:',
    '\n', 
    f'cNFW={halo_cNFW:.2f}',
    f'rvir={halo_rvir/kpc:.2f} kpc ; Mvir={halo_Mvir/Msun:.2e} Msun'
)

# DM_lim_custom = 300000
# DM_lim_custom = 50000
# DM_lim_custom = 30000
# DM_lim_custom = 10000
# DM_lim_custom = 8000
DM_lim_custom = 1000

GRID_L_custom = 800*kpc
GRID_S_custom= 600*kpc

adapted_cc, cell_gen, cell_com, DM_count = fct.manual_cell_division(
    sim_id, snap_num, DM_raw,
    DM_lim_custom, GRID_L_custom, GRID_S_custom
)


################################################
### Plotting the outcome after iteration(s). ###
################################################

# Build grid around Milky Way.
trimmed_cc = np.delete(adapted_cc, np.s_[DM_count==0], axis=0)
print(trimmed_cc.shape)
new_grid = np.squeeze(adapted_cc, axis=1) / kpc

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

DM_raw /= kpc
x_DM, y_DM, z_DM = DM_raw[:,0], DM_raw[:,1], DM_raw[:,2]
cut = 10
x, y, z = x_DM[1::cut], y_DM[1::cut], z_DM[1::cut]

ax.scatter(x, y, z, alpha=0.05, c='rebeccapurple', s=0.2)

# Draw sphere around GC with radius=Rvir_MW.
rGC = halo_rvir/kpc
uGC, vGC = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
xGC = rGC * np.cos(uGC) * np.sin(vGC)
yGC = rGC * np.sin(uGC) * np.sin(vGC)
zGC = rGC * np.cos(vGC)

xg, yg, zg = new_grid[:,0], new_grid[:,1], new_grid[:,2] 
ax.scatter(xg, yg, zg, s=0.2, marker='x', color='black', alpha=0.5)

# Can't make it show up with all the DM particles.
# ax.scatter(
#     X_SUN[0], X_SUN[1], X_SUN[2], s=10, color='blue', marker='o',
#     label='Earth', zorder=0
# )

ax.plot_surface(
    xGC, yGC, zGC, alpha=0.1, 
    cmap=plt.cm.coolwarm, vmin=-1, vmax=1,# antialiased=False,
    rstride=1, cstride=1
)

zero_cells = np.count_nonzero(DM_count==0.)
print(zero_cells)

plt.show()