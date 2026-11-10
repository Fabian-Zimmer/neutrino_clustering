"""Number densities including primordial temperature fluctuations (paper 2).

Draws healpix maps of the CNB temperature fluctuations Delta from the Cl's of
CLASS (one map per neutrino mass and momentum bin), assigns each simulated
neutrino the Delta of its pixel and momentum at the end of its trajectory, and
recomputes pixel and total number densities. Uses the vectors_halo*.npy of a
run from scripts/simulate.py; that run must end at z_cut (or z~4 if null).

Replaces the old sim_script30 (z_cut: null, random_phases: true) and the
sim_script3_primordial_fluctuations_z* scripts (z_cut 0.15/0.5/1/2,
random_phases: false).

Usage: python projects/primordial_fluctuations/fluctuation_densities.py projects/primordial_fluctuations/primordial.yaml
"""

import healpy as hp
import jax.numpy as jnp
import numpy as np

from cnb.config import config_from_cli
from cnb.physics import Physics
from cnb.primordial import Primordial
from cnb.sim_data import SimData
from cnb.units import Params


def z_label(z):
    """File name label of a redshift: 4 -> '4', 0.5 -> '0p5', 0.15 -> '0p15'."""
    return f"{z:g}".replace(".", "p")


def fluctuation_densities(cfg):

    run_dir, Cl_folder, Delta_folder = cfg["run_dir"], cfg["cl_dir"], cfg["delta_dir"]
    pf = cfg["primordial"]
    skip = set(cfg["halos"].get("skip", []))

    # Load simulation parameters
    simdata = SimData(run_dir)

    nu_m_picks = jnp.array(pf["masses"])*Params.eV
    m_num = len(nu_m_picks)

    # Redshift where the simulation ends. Full runs end at the box's last
    # snapshot (z~4); the Cl's and the momentum matching use z=4 then.
    z_cut = 4 if pf["z_cut"] is None else pf["z_cut"]
    z_Delta = simdata.z4_sim_val if pf["z_cut"] is None else z_cut
    z_str = z_label(z_cut)

    # Different random seed per momentum bin, or the same seed (5) for all.
    random_phases = pf["random_phases"]
    end_str = "_random_phases" if random_phases else ""


    ### ============= ###
    ### Create Deltas ###
    ### ============= ###

    Deltas_z_m_l = []
    Deltas_z0_m_l = []
    for m_Cl in nu_m_picks:

        # Load Cl's for current neutrino mass
        Cls_z_raw = jnp.load(f"{Cl_folder}/Cls_z={z_str}_m={m_Cl}eV.npy")
        Cls_z0_raw = jnp.load(f"{Cl_folder}/Cls_z=0_m={m_Cl}eV.npy")
        # (q momentum bins, l multipoles) = (50, 20)

        # Add monopole
        Cl_z = jnp.insert(Cls_z_raw, 0, 0, axis=1)
        Cl_z0 = jnp.insert(Cls_z0_raw, 0, 0, axis=1)
        # (q momentum bins, l multipoles + 1) = (50, 21)

        # Create temp. flucts. skymaps from Cls, fixed with a seed, for each q
        Deltas_z_q_l = []
        Deltas_z0_q_l = []
        for qi in range(Primordial.n_qbins):

            seed = qi if random_phases else 5

            np.random.seed(seed)
            Tmap_z = hp.sphtfunc.synfast(
                Cl_z[qi], nside=simdata.Nside, lmax=None, pol=False)

            np.random.seed(seed)  #note: seed needs to reset
            Tmap_z0 = hp.sphtfunc.synfast(
                Cl_z0[qi], nside=simdata.Nside, lmax=None, pol=False)

            Deltas_z = Tmap_z * Primordial.uK / (Params.T_CNB*(1+z_Delta))
            Deltas_z0 = Tmap_z0 * Primordial.uK / (Params.T_CNB)
            # (Npix)

            Deltas_z_q_l.append(Deltas_z)
            Deltas_z0_q_l.append(Deltas_z0)

        Deltas_z_m_l.append(jnp.array(Deltas_z_q_l))
        Deltas_z0_m_l.append(jnp.array(Deltas_z0_q_l))

    Deltas_z_matrix = jnp.array(Deltas_z_m_l)
    Deltas_z0_matrix = jnp.array(Deltas_z0_m_l)
    # (masses, q momentum bins, Npix)
    jnp.save(f"{Delta_folder}/Delta_matrix_z{z_str}{end_str}.npy", Deltas_z_matrix)
    jnp.save(f"{Delta_folder}/Delta_matrix_z0{end_str}.npy", Deltas_z0_matrix)


    # Loop over all halos in directory
    Deltas_halos_l = []
    pix_dens_incl_PFs_l = []
    tot_dens_incl_PFs_l = []
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
        p_z = p_arr[...,-1]
        # (masses, Npix, neutrinos per pixel)

        # Cl momenta are expressed in terms of T_CNB(z=0), but for proper
        # matching we need momenta at z=z_cut in terms of T_CNB(z=z_cut).
        q_z = y_arr[...,-1]/(1+z_cut)
        # (masses, Npix, neutrinos per pixel)

        # Find indices to match neutrino momenta to Cl momenta
        q_idx = jnp.abs(
            Primordial.Cl_qs[None,None,None,:] - q_z[...,None]).argmin(axis=-1)
        q_idx = jnp.reshape(q_idx, (m_num, -1))
        # (masses, Npix, neutrinos per pixel)

        # Pixel indices for all neutrinos
        # (looks like [0,...,0,1,...,1,...,Npix-1,...,Npix-1])
        p_idx = jnp.repeat(jnp.arange(simdata.Npix), simdata.p_num)[None, :]

        # Mass indices adjusted for broadcasting / fancy indexing of Delta matrix
        m_idx = jnp.arange(m_num)[:, None]

        # Select corresponding pixels, i.e. temp. perturbations, for all neutrinos
        Deltas_halo = jnp.reshape(
            Deltas_z_matrix[m_idx, q_idx, p_idx],
            (m_num, simdata.Npix, simdata.p_num))
        Deltas_halos_l.append(Deltas_halo)
        # (masses, Npix, neutrinos per pixel)


        ### ------------------------------------------------------------------ ###
        ### Compute number densities incl. primordial temperature fluctuations ###
        ### ------------------------------------------------------------------ ###

        # For individual allsky map pixels
        pix_dens_halo = Physics.number_density_Delta(
            p_z0, p_z, Deltas_halo, simdata.pix_sr, Params())
        pix_dens_incl_PFs_l.append(pix_dens_halo)
        # (masses, Npix)

        # For total local value
        tot_dens_halo = Physics.number_density_Delta(
            p_z0.reshape(m_num, -1),
            p_z.reshape(m_num, -1),
            Deltas_halo.reshape(m_num, -1),
            4*Params.Pi, Params())
        tot_dens_incl_PFs_l.append(tot_dens_halo)


    jnp.save(f"{run_dir}/Deltas_halos{end_str}.npy", jnp.array(Deltas_halos_l))
    jnp.save(f"{run_dir}/pixel_densities_incl_PFs{end_str}.npy", jnp.array(pix_dens_incl_PFs_l))
    jnp.save(f"{run_dir}/total_densities_incl_PFs{end_str}.npy", jnp.array(tot_dens_incl_PFs_l))


if __name__ == "__main__":
    fluctuation_densities(config_from_cli(__doc__))
