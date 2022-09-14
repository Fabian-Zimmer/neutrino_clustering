from shared.preface import *
import shared.functions as fct

### Script for generating number density plot for different DM radii. ###


def number_density_1_mass(m_nu_eV, average:bool, DM_rad):

    n_nus = np.zeros(len(m_nu_eV))
    for i, m_eV in enumerate(m_nu_eV):

        # Get momenta.
        p, _ = fct.u_to_p_eV(u_all, m_eV)

        if average and m_eV >= 0.01:
            # Calculate number density, averaged values for z in [3.5, 4].
            # note: for CubeSpace, there are no squiggles in z_back,
            # note: so averaging does nothing
            #! re-evaluate this
            idx = np.array(np.where(ZEDS >= 3.5)).flatten()

            temp = np.zeros(len(idx))
            for j,k in enumerate(idx):
                val = fct.number_density(p[:,0], p[:,k])
                temp[j] = val

            n_nus[i] = np.mean(temp)

        else:
            n_nus[i] = fct.number_density(p[:,0], p[:,-1])

    np.save(
        f'neutrino_data/CubeSpace_overdensities_{nus}nus_{DM_rad}kpc.npy', 
        n_nus
        )


# Parameters.
nus = 10000
m0 = HALO_MASS
DM_radii = np.linspace(260, 800, 5)

# Plotting:
fig, ax = plt.subplots(1,1)

# 10 to 300 meV like in the paper.
mass_range_eV = np.geomspace(0.01, 0.3, 50)*eV

for DM_radius in DM_radii[:-1]:

    u_all = np.load(
        f'neutrino_vectors/nus_{nus}_CubeSpace_{m0}Msun_{DM_radius}kpc.npy'
    )[:,:,3:6]

    number_density_1_mass(mass_range_eV, average=True, DM_rad=DM_radius)

    n_nus = np.load(
        f'neutrino_data/CubeSpace_overdensities_{nus}nus_{DM_radius}kpc.npy'
        )/N0

    ax.plot(mass_range_eV*1e3, (n_nus-1), label=f'DM radius: {DM_radius}')

x_ends = [1e1, 3*1e2]
y_ends = [3*1e-3, 4]
ax.plot(x_ends, y_ends, marker='x', ls=':', c='r', alpha=0.6)

for m in NU_MASSES:
    ax.axvline(m*1e3, c='r', ls='-.')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(f'CubeSpace!, Neutrinos: {nus}')
ax.set_xlabel(r'$m_{\nu}$ [meV]')
ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$')
# ax.set_ylim(1e-3, 1e1)
plt.grid(True, which="both", ls="-")

ax.yaxis.set_major_formatter(ticker.FuncFormatter(fct.y_fmt))

plt.legend(loc='lower right')
plt.savefig(f'figures/CubeSpace_overdensities_{nus}nus_DM_radii.pdf')
# plt.show()

# print('Max value:', np.max(n_nus), np.max(n_nus-1))