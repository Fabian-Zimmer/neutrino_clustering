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

nu_allsky_masses = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV
m_num = len(nu_allsky_masses)

# Momenta of the Cl maps
Cl_qs = jnp.geomspace(0.01, 100, 50)

# Microkelvin unit
uK = 1e-6*Params.K

Deltas_halos_l = []
pix_dens_incl_PFs_l = []
tot_dens_incl_PFs_l = []

# Loop over all halos in directory
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

    # Load temperature fluctuations
    Delta_matrix_l = []
    data_dir = f"{pars.shared_dir}/Deltas"
    for m_nu in nu_allsky_masses:
        for qi in range(Primordial.n_qbins):
            Delta_matrix_l.append(jnp.load(
                f"{data_dir}/Deltas_Nside={Nside}_qi={qi}_z=4_m={m_nu}eV.npy")) 

    Delta_matrix = jnp.array(Delta_matrix_l) * uK / (Params.T_CNB*(1+4))
    Deltas_sync = jnp.reshape(Delta_matrix, (m_num, Primordial.n_qbins, Npix))
    # (masses, q bins, Npix)

    # Pixel indices for all neutrinos (looks like [0,...,0,1,...,1,...])
    p_idx = jnp.repeat(jnp.arange(Npix), nu_per_pix)[None, :]

    # Find indices to match neutrino momenta to Cl momenta
    q_idx = jnp.abs(Cl_qs[None,None,None,:] - q_z4[...,None]).argmin(axis=-1)
    q_idx = jnp.reshape(q_idx, (m_num, -1))
    # (masses, Npix, neutrinos per pixel)

    m_idx = jnp.arange(m_num)[:, None]

    # Select corresponding pixels, i.e. temp. perturbations, for all neutrinos
    Deltas_halo = jnp.reshape(
        Deltas_sync[m_idx, q_idx, p_idx], (m_num, Npix, nu_per_pix))
    Deltas_halos_l.append(Deltas_halo)
    # (masses, Npix, neutrinos per pixel)


    ### ------------------------------------------------------------------ ###
    ### Compute number densities incl. primordial temperature fluctuations ###
    ### ------------------------------------------------------------------ ###

    # For individual allsky map pixels
    pix_dens_halo = Physics.number_density_Delta(
        p_z0, p_z4, Deltas_halo, pix_sr, Params())
    pix_dens_incl_PFs_l.append(pix_dens_halo / (Params.N0/Npix/Params.cm**-3))
    # (masses, Npix)

    # For total local value
    tot_dens_halo = Physics.number_density_Delta(
        p_z0.reshape(m_num, -1), 
        p_z4.reshape(m_num, -1), 
        Deltas_halo.reshape(m_num, -1), 
        pix_sr, Params())
    tot_dens_incl_PFs_l.append(tot_dens_halo / (Params.N0/Params.cm**-3))


jnp.save(
    f"{pars.sim_dir}/Deltas_halos.npy", jnp.array(Deltas_halos_l))
jnp.save(
    f"{pars.sim_dir}/pixel_densities_incl_PFs", jnp.array(pix_dens_incl_PFs_l))
jnp.save(
    f"{pars.sim_dir}/total_densities_incl_PFs", jnp.array(tot_dens_incl_PFs_l))


# note: then can download to local repo and make skymaps, power spectra, etc.