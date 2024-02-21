from shared.preface import *
import commah

sim_dir = f'L025N752/DMONLY/SigmaConstant00/single_halos'

# Box parameters and arrays.
with open(f'{sim_dir}/box_parameters.yaml', 'r') as file:
    box_setup = yaml.safe_load(file)
DM_mass = box_setup['Content']['DM Mass [Msun]']*Msun
nums_snaps = np.load(f'{sim_dir}/nums_snaps.npy')
zeds_snaps = np.load(f'{sim_dir}/zeds_snaps.npy')

FCT_h = box_setup['Cosmology']['h']
FCT_H0 = FCT_h*100*km/s/Mpc
FCT_Omega_M = box_setup['Cosmology']['Omega_M']
FCT_Omega_L = box_setup['Cosmology']['Omega_L']

def rho_crit(z):
    """Critical density of the universe as a function of redshift, assuming
    matter domination, only Omega_m and Omega_Lambda in Friedmann equation. See 
    notes for derivation.

    Args:
        z (array): redshift

    Returns:
        array: critical density at redshift z
    """    
    
    H_squared = FCT_H0**2 * (FCT_Omega_M*(1.+z)**3 + FCT_Omega_L) 
    rho_crit = 3.*H_squared / (8.*Pi*G)

    return np.float64(rho_crit)


def halo_sample_z(z, snap, Mvir_z0, DM_mass, out_dir, origin_offset=0.):

    # z = np.float64(z)
    # snap = str(snap)
    # Mvir_z0 = np.float64(Mvir_z0)
    # DM_mass = np.float64(DM_mass)

    # Get the DM halo mass (and the number of DM particles for sample).
    commah_output = commah.run('Planck13', zi=0, Mi=Mvir_z0, z=z)
    Mz = commah_output['Mz'][0,0]*Msun
    num_DM = math.floor(Mz / DM_mass)

    # Get the concentration of the halo.
    c_200 = commah_output['c'][0,0]

    # Calculate R_200 and R_s ("virial" radius and scale radius).
    R_200 = np.power(Mz / (200*rho_crit(z)*4/3*Pi), 1./3.)
    R_s = R_200 / c_200

    # Construct projection function.
    def Proj(r, r_s, norm):
        x = r/r_s
        return (np.log(1+x) - (x/(1+x)))/norm

    # Construct inverse function. Needs to be without numerical units.
    f_200 = np.log(1+c_200) - (c_200/(1+c_200))
    invf = inversefunc(Proj, args=(R_s/kpc, f_200))  

    # Sample uniformly between [0,1] and project to corresponding radius.
    sample = np.sort(np.random.uniform(size=num_DM))
    r_sample = invf(sample)

    # Sample for angles and convert to cartesian DM_coords.
    phis = np.random.uniform(0, 2*Pi, num_DM)  # uniform [0,2pi)
    cos_thetas = 2.*np.random.uniform(0, 1, num_DM) - 1  # uniform [-1,1)

    # Convert to cartesian coordinates.
    x = r_sample*np.cos(phis)*np.sqrt(1. - cos_thetas**2)
    y = r_sample*np.sin(phis)*np.sqrt(1. - cos_thetas**2)
    z = r_sample*cos_thetas
    coords = np.column_stack((x,y,z)) + origin_offset

    print(np.column_stack((x,y,z)).shape)

    np.save(f'{out_dir}/benchmark_halo_snap_{snap}.npy', coords)


benchmark_outdir = f'L025N752/DMONLY/SigmaConstant00/benchmark_halo_files'
if not os.path.exists(benchmark_outdir):
    os.makedirs(benchmark_outdir)

# Get median Mvir of box halo sample from numerical simulation.
halo_params = np.load(glob.glob(f'{sim_dir}/halo*params.npy')[0])
Rvir_med = halo_params[:,0]
Mvir_med = (10**np.median(halo_params[:,1]))
cNFW_med = halo_params[:,2]
Rs_med = Rvir_med/cNFW_med
print(f'Median Mvir of box halo sample: {Mvir_med:.2e} Msun')
print(
    f'Min/Max value of Rvir of box halo sample: {np.min(Rvir_med):.2f}/{np.max(Rvir_med):.2f} kpc'
)
print(f'Min/Max value of concentration of box halo sample: {np.min(cNFW_med):.2f}/{np.max(cNFW_med):.2f}')

# with ProcessPoolExecutor(25) as ex:
#     ex.map(
#         halo_sample_z, zeds_snaps, nums_snaps,
#         repeat(Mvir_med), repeat(DM_mass), repeat(benchmark_outdir)
#     )

# offset = X_VC*kpc
# halo_sample_z(
#     zeds_snaps[0], nums_snaps[0], Mvir_VC/Msun, DM_mass, benchmark_outdir, offset
# )

benchmark_DM = np.array([
    len(np.load(f'{benchmark_outdir}/benchmark_halo_snap_{num}.npy')) 
    for num in nums_snaps
])

print(np.log10(Mvir_MW/Msun), np.log10(DM_mass*benchmark_DM[-1]/Msun))

nums_proxy = np.arange(12, 36+1)
plt.plot(nums_proxy, benchmark_DM); plt.show()
