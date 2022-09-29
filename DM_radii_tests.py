from shared.preface import *
import shared.functions as fct

sim = 'L012N376'
snap = '0036'
halo_ID = 3
# DM_range_kpc = 298.73*kpc  # rvir for halo_ID = 2
DM_range_kpc = 800*kpc

'''
fname = f'Test_all_inRange'
halo_rvir = fct.read_DM_all_inRange(sim, snap, halo_ID, DM_range_kpc, fname)
print('Virial radius of selected halo:', halo_rvir)
DM_raw = np.load(f'{sim}/DM_pos_{fname}.npy')
print('All DM in range: ', len(DM_raw))

fname2 = f'Test_gravBound'
fct.read_DM_halo_index(sim, snap, halo_ID, fname2)
DM_raw2 = np.load(f'{sim}/DM_pos_{fname2}.npy')
print('DM grav. bound to halo:', len(DM_raw2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
DM_fig = DM_raw
x_DM, y_DM, z_DM = DM_fig[:,0], DM_fig[:,1], DM_fig[:,2]
cut = 10
x, y, z = x_DM[1::cut], y_DM[1::cut], z_DM[1::cut]
ax.scatter(x, y, z, alpha=0.9, c='rebeccapurple', s=0.001)
plt.show()
'''

fname3 = f'Test_halos_inRange'
time1 = time.perf_counter()
fct.read_DM_halos_inRange(sim, snap, halo_ID, DM_range_kpc, 3, fname3)
time2 = time.perf_counter()
print('time in min.:', (time2-time1)/60)

DM_raw3 = np.load(f'{sim}/DM_pos_{fname3}.npy')
print('DM grav. bound to halos in range:', len(DM_raw3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
DM_fig = DM_raw3
x_DM, y_DM, z_DM = DM_fig[:,0], DM_fig[:,1], DM_fig[:,2]
cut = 10
x, y, z = x_DM[1::cut], y_DM[1::cut], z_DM[1::cut]
ax.scatter(x, y, z, alpha=0.9, c='rebeccapurple', s=0.001)
plt.savefig(f'DM_radii_tests.pdf')
plt.show()