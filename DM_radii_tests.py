from shared.preface import *
import shared.functions as fct

sim = 'L012N376'
snap = '0036'
halo_ID = 2
DM_range_kpc = 5000*kpc


fname3 = f'Test_halos_inRange'
time1 = time.perf_counter()
fct.read_DM_halos_inRange(sim, snap, halo_ID, DM_range_kpc, fname3)
time2 = time.perf_counter()
print('time in min.:', (time2-time1)/60)

DM_raw3 = np.load(f'{sim}/DM_pos_{fname3}.npy')
print('DM grav. bound to halos in range:', len(DM_raw3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
DM_fig = DM_raw3
x_DM, y_DM, z_DM = DM_fig[:,0], DM_fig[:,1], DM_fig[:,2]
cut = 5
x, y, z = x_DM[1::cut], y_DM[1::cut], z_DM[1::cut]
ax.scatter(x, y, z, alpha=0.9, c='rebeccapurple', s=0.001)
plt.savefig(f'DM_radii_tests.pdf')
plt.show()