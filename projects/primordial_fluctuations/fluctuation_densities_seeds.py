"""Number densities including primordial fluctuations, for several random seeds (paper 2).

Same as fluctuation_densities.py, but draws the fluctuation maps for 10 seeds
(Primordial.generate_DeltaTs_seeds) to estimate the scatter between
realisations. Saves *_seeds_halo<N>[_random_phases].npy in run_dir.

Usage: python projects/primordial_fluctuations/fluctuation_densities_seeds.py projects/primordial_fluctuations/primordial.yaml
"""

import jax.numpy as jnp

from cnb.config import config_from_cli
from cnb.physics import Physics
from cnb.primordial import Primordial
from cnb.sim_data import SimData
from cnb.units import Params

cfg = config_from_cli(__doc__)
pf = cfg["primordial"]
run_dir, Cl_folder, Delta_folder = cfg["run_dir"], cfg["cl_dir"], cfg["delta_dir"]
skip = set(cfg["halos"].get("skip", []))

# Redshift where the simulation ends (z~4 for full runs).
z_cut = 4 if pf["z_cut"] is None else pf["z_cut"]

# Load simulation parameters
simdata = SimData(run_dir)

nu_m_picks = jnp.array(pf["masses"])*Params.eV
m_num = len(nu_m_picks)
seeds = jnp.arange(pf["seeds"])

# note: False for synchronized harmonic maps for all q bins
random_phases = pf["random_phases"]

### ============= ###
### Create Deltas ###
### ============= ###



Deltas_z0_matrix, Deltas_z_cut_matrix = Primordial.generate_DeltaTs_seeds(
    m_arr=nu_m_picks,
    Cl_dir=Cl_folder,
    Delta_dir=Delta_folder,
    seeds=seeds,
    simdata=simdata,
    z_cut=z_cut,
    phases=random_phases,
    args=Params())


### ============================= ###
### Do calculations for all halos ###
### ============================= ###

for halo_j in range(int(cfg["halos"]["halo_num"])):

    # note: "Broken" halos (e.g. no DM position data at snapshot 0012).
    if halo_j+1 in skip:
        continue

    # Load neutrino vectors from simulation
    nu_vectors = jnp.load(f'{run_dir}/vectors_halo{halo_j+1}.npy')

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
    q_z_cut = y_arr[...,-1]/(1+z_cut)
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


    if random_phases:
        jnp.save(
            f"{run_dir}/Deltas_seeds_halo{halo_j+1}_random_phases.npy", 
            jnp.array(Deltas_seeds_l))
        jnp.save(
            f"{run_dir}/pixel_densities_incl_PFs_seeds_halo{halo_j+1}_random_phases.npy", 
            jnp.array(pix_dens_incl_PFs_seeds_l))
        jnp.save(
            f"{run_dir}/total_densities_incl_PFs_seeds_halo{halo_j+1}_random_phases.npy", 
            jnp.array(tot_dens_incl_PFs_seeds_l))
    else:
        jnp.save(
            f"{run_dir}/Deltas_seeds_halo{halo_j+1}.npy", 
            jnp.array(Deltas_seeds_l))
        jnp.save(
            f"{run_dir}/pixel_densities_incl_PFs_seeds_halo{halo_j+1}.npy", 
            jnp.array(pix_dens_incl_PFs_seeds_l))
        jnp.save(
            f"{run_dir}/total_densities_incl_PFs_seeds_halo{halo_j+1}.npy", 
            jnp.array(tot_dens_incl_PFs_seeds_l))
