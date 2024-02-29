from Shared.specific_CNB_sim import *

# This script solely computes the pixel and total densities,
# when including the primordial temperature fluctuations.

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('--sim_dir', required=True)
parser.add_argument('--shared_dir', required=True)
parser.add_argument('--halo_num', required=True)
pars = parser.parse_args()

# Load simulation parameters
with open(f"{pars.sim_dir}/sim_parameters.yaml", "r") as file:
    sim_setup = yaml.safe_load(file)
Nside = sim_setup["Nside"]
Npix = sim_setup["Npix"]
nu_per_pix = sim_setup["momentum_num"]
pix_sr = sim_setup["pix_sr"]

nu_allsky_masses = jnp.array([
    0.01, 0.05, 
    # 0.1, 0.2, 0.3
])*Params.eV
m_num = len(nu_allsky_masses)

# Momenta of the Cl maps
Cl_qs = jnp.geomspace(0.01, 100, 50)

# Microkelvin unit
uK = 1e-6*Params.K


### ============= ###
### Create Deltas ###
### ============= ###


Cl_folder = f"Shared/Cls"
Delta_folder = f"Shared/Deltas"

Deltas_z4_m_l = []
Deltas_z0_m_l = []
for m_Cl in nu_allsky_masses:

    # Load Cl's for current neutrino mass
    Cls_z4_raw = jnp.load(f"{Cl_folder}/Cls_z=4_m={m_Cl}eV.npy")
    Cls_z0_raw = jnp.load(f"{Cl_folder}/Cls_z=0_m={m_Cl}eV.npy")
    # (q momentum bins, l multipoles) = (50, 20)

    # Add monopole
    Cl_z4 = jnp.insert(Cls_z4_raw, 0, 0, axis=1)
    Cl_z0 = jnp.insert(Cls_z0_raw, 0, 0, axis=1)
    # (q momentum bins, l multipoles + 1) = (50, 21)

    # Create temp. flucts. skymaps from Cls, fixed with a seed, for each q
    Deltas_z4_q_l = []
    Deltas_z0_q_l = []
    for qi in range(Primordial.n_qbins):

        # note: seed needs to be initiated before each use for some reason...
        np.random.seed(5)  # fixed seed for skymaps
        Tmap_z4 = hp.sphtfunc.synfast(
            Cl_z4[qi], nside=Nside, lmax=None, pol=False)
        np.random.seed(5)  # fixed seed for skymaps
        Tmap_z0 = hp.sphtfunc.synfast(
            Cl_z0[qi], nside=Nside, lmax=None, pol=False)

        #? Convert temp. maps to correct units and conventions
        # Tmap_z4 += Params.T_CNB*(1+4)  #?
        # Tmap_z0 += Params.T_CNB        #?

        Deltas_z4 = Tmap_z4 * uK / (Params.T_CNB*(1+4))
        Deltas_z0 = Tmap_z0 * uK / (Params.T_CNB)
        # (Npix)

        Deltas_z4_q_l.append(Deltas_z4)
        Deltas_z0_q_l.append(Deltas_z0)

    Deltas_z4_m_l.append(jnp.array(Deltas_z4_q_l))
    Deltas_z0_m_l.append(jnp.array(Deltas_z0_q_l))


Deltas_z4_matrix = jnp.array(Deltas_z4_m_l)
Deltas_z0_matrix = jnp.array(Deltas_z0_m_l)
# (masses, q momentum bins, Npix)
jnp.save(f"{Delta_folder}/Delta_matrix_z4.npy", Deltas_z4_matrix)
jnp.save(f"{Delta_folder}/Delta_matrix_z0.npy", Deltas_z0_matrix)



# Loop over all halos in directory
Deltas_halos_l = []
pix_dens_incl_PFs_l = []
tot_dens_incl_PFs_l = []
for halo_j in range(int(pars.halo_num)):

    # Load neutrino vectors from simulation
    nu_vectors = jnp.load(f'{pars.sim_dir}/vectors_halo{halo_j+1}.npy')

    # Convert to momenta
    v_arr = nu_vectors[..., 3:]
    p_arr, y_arr = Physics.velocities_to_momenta_all_sky(
        v_arr, nu_allsky_masses, Params())
    # (masses, Npix, neutrinos per pixel, 2 (first and last time step))

    # Momenta at z=0 and z=4
    p_z0 = p_arr[..., 0]
    p_z4 = p_arr[...,-1]
    # (masses, Npix, neutrinos per pixel)

    q_z0 = y_arr[..., 0]
    q_z4 = y_arr[...,-1] / (1+4)  #? in terms of T_CNB(z=0) or T_CNB(z=4) ?
    # (masses, Npix, neutrinos per pixel)

    # Pixel indices for all neutrinos
    # (looks like [0,...,0,1,...,1,...,Npix-1,...,Npix-1])
    p_idx = jnp.repeat(jnp.arange(Npix), nu_per_pix)[None, :]

    # Find indices to match neutrino momenta to Cl momenta
    q_idx = jnp.abs(Cl_qs[None,None,None,:] - q_z4[...,None]).argmin(axis=-1)
    q_idx = jnp.reshape(q_idx, (m_num, -1))
    # (masses, Npix, neutrinos per pixel)

    # Mass indices adjusted for broadcasting / fancy indexing of Delta matrix
    m_idx = jnp.arange(m_num)[:, None]

    # Select corresponding pixels, i.e. temp. perturbations, for all neutrinos
    Deltas_halo = jnp.reshape(
        Deltas_z4_matrix[m_idx, q_idx, p_idx], (m_num, Npix, nu_per_pix))
    Deltas_halos_l.append(Deltas_halo)
    # (masses, Npix, neutrinos per pixel)


    ### ------------------------------------------------------------------ ###
    ### Compute number densities incl. primordial temperature fluctuations ###
    ### ------------------------------------------------------------------ ###

    # For individual allsky map pixels
    pix_dens_halo = Physics.number_density_Delta(
        p_z0, p_z4, Deltas_halo, pix_sr, Params())
    pix_dens_incl_PFs_l.append(pix_dens_halo)
    # (masses, Npix)

    # For total local value
    tot_dens_halo = Physics.number_density_Delta(
        p_z0.reshape(m_num, -1), 
        p_z4.reshape(m_num, -1), 
        Deltas_halo.reshape(m_num, -1), 
        pix_sr, Params())
    tot_dens_incl_PFs_l.append(tot_dens_halo)


jnp.save(
    f"{pars.sim_dir}/Deltas_halos.npy", jnp.array(Deltas_halos_l))
jnp.save(
    f"{pars.sim_dir}/pixel_densities_incl_PFs", jnp.array(pix_dens_incl_PFs_l))
jnp.save(
    f"{pars.sim_dir}/total_densities_incl_PFs", jnp.array(tot_dens_incl_PFs_l))


# note: then can download to local repo and make skymaps, power spectra, etc.