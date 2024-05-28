from Shared.specific_CNB_sim import *

# This script solely computes the pixel and total densities,
# when including the primordial temperature fluctuations, 
# drawn from different seeds.

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('--sim_dir', required=True)
parser.add_argument('--shared_dir', required=True)
parser.add_argument('--halo_num', required=True)
parser.add_argument('--redshift_cut', required=True)
pars = parser.parse_args()

# Load simulation parameters
simdata = SimData(pars.sim_dir)

nu_m_picks = jnp.array([
    0.01, 
    # 0.05, 
    0.1, 
    # 0.2, 
    # 0.3
])*Params.eV
m_num = len(nu_m_picks)
seeds = jnp.arange(10)

### ============= ###
### Create Deltas ###
### ============= ###


Cl_folder = f"Shared/Cls"
Delta_folder = f"Shared/Deltas"


Deltas_z0_matrix, Deltas_z_cut_matrix = SimUtil.generate_DeltaTs_seeds(
    m_arr=nu_m_picks,
    Cl_dir=Cl_folder,
    Delta_dir=Delta_folder,
    seeds=seeds,
    simdata=simdata,
    z_cut=int(pars.redshift_cut),
    args=Params())


### ============================= ###
### Do calculations for all halos ###
### ============================= ###

for halo_j in range(int(pars.halo_num)):

    # note: "Broken" halo, no DM position data at snapshot 0012.
    if halo_j == 19:
        continue

    # Load neutrino vectors from simulation
    nu_vectors = jnp.load(f'{pars.sim_dir}/vectors_halo{halo_j+1}.npy')

    # Convert to momenta
    v_arr = nu_vectors[..., 3:]
    p_arr, y_arr = Physics.velocities_to_momenta_all_sky(
        v_arr, nu_m_picks, Params())
    # (masses, Npix, neutrinos per pixel, 2 (first and last time step))

    # Momenta at z=0 and z=z_cut
    p_z0 = p_arr[..., 0]
    p_z_cut = p_arr[...,-1]
    # (masses, Npix, neutrinos per pixel)

    # Cl momenta are expressed in terms of T_CNB(z=0), but for proper matching 
    # we need momenta at z=z_cut in terms of T_CNB(z=z_cut).
    q_z_cut = y_arr[...,-1]/(1+int(pars.redshift_cut))
    # (masses, Npix, neutrinos per pixel)

    # Find indices to match neutrino momenta to Cl momenta
    q_idx = jnp.abs(Primordial.Cl_qs[None,None,None,:] - q_z_cut[...,None]).argmin(axis=-1)
    q_idx = jnp.reshape(q_idx, (m_num, -1))
    # (masses, Npix, neutrinos per pixel)

    # Pixel indices for all neutrinos
    # (looks like [0,...,0,1,...,1,...,Npix-1,...,Npix-1])
    p_idx = jnp.repeat(jnp.arange(simdata.Npix), simdata.p_num)[None, :]

    # Mass indices adjusted for broadcasting / fancy indexing of Delta matrix
    m_idx = jnp.arange(m_num)[:, None]


    ### ====================================== ###
    ### Compute number densities for all seeds ###
    ### ====================================== ###

    # Loop over all seeds for one fixed halo
    Deltas_seeds_l = []
    pix_dens_incl_PFs_seeds_l = []
    tot_dens_incl_PFs_seeds_l = []
    for i, _ in enumerate(seeds):

        # Choose submatrix for current seed
        Deltas_z_cut_seed = Deltas_z_cut_matrix[i]

        # Select corresponding pixels, i.e. temp. perturbations, for all neutrinos
        Deltas_seed = jnp.reshape(
            Deltas_z_cut_seed[m_idx, q_idx, p_idx], 
            (m_num, simdata.Npix, simdata.p_num))
        Deltas_seeds_l.append(Deltas_seed)
        # (masses, Npix, neutrinos per pixel)

        # For individual allsky map pixels
        pix_dens_seed = Physics.number_density_Delta(
            p_z0, p_z_cut, Deltas_seed, simdata.pix_sr, Params())
        pix_dens_incl_PFs_seeds_l.append(pix_dens_seed)
        # (masses, Npix)

        # For total local value
        tot_dens_seed = Physics.number_density_Delta(
            p_z0.reshape(m_num, -1), 
            p_z_cut.reshape(m_num, -1), 
            Deltas_seed.reshape(m_num, -1), 
            4*Params.Pi, Params())
        tot_dens_incl_PFs_seeds_l.append(tot_dens_seed)


    jnp.save(
        f"{pars.sim_dir}/Deltas_seeds_halo{halo_j+1}.npy", 
        jnp.array(Deltas_seeds_l))
    jnp.save(
        f"{pars.sim_dir}/pixel_densities_incl_PFs_seeds_halo{halo_j+1}.npy", 
        jnp.array(pix_dens_incl_PFs_seeds_l))
    jnp.save(
        f"{pars.sim_dir}/total_densities_incl_PFs_seeds_halo{halo_j+1}.npy", 
        jnp.array(tot_dens_incl_PFs_seeds_l))


# note: then can download to local repo and make skymaps, power spectra, etc.