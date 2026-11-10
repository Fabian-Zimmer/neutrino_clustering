"""Neutrino density across a whole DM halo (the "neutrino halo").

Instead of one Earth-like starting position, neutrinos start from ~300 cells
spread over the halo (CellSelection.reduce_cells keeps more cells where the DM
is denser). For each start cell the all-sky simulation of scripts/simulate.py
runs and pixel/total densities are computed. Old sim_script50.

Needs the grids of the run (scripts/precalculate_grids.py).
Outputs in run_dir: vectors_halo<N>_cell<C>.npy (last z step only),
pixel_densities.npy and total_densities.npy with shape (halos, cells, ...).

Usage: python projects/neutrino_halo/simulate_neutrino_halo.py projects/neutrino_halo/neutrino_halo.yaml
"""

import time

import jax.numpy as jnp
import numpy as np
import yaml

from cell_selection import CellSelection
from cnb.config import config_from_cli
from cnb.sim_data import SimData
from cnb.simulation import Simulation, SolverOptions
from cnb.units import Params


def simulate_neutrino_halo(cfg):

    run_dir, data_dir = cfg["run_dir"], cfg["data_dir"]
    sim, halos = cfg["simulate"], cfg["halos"]
    skip = set(halos.get("skip", []))
    opts = SolverOptions(force="halo", relativistic=sim.get("relativistic", True))

    DM_mass, CPUs_sim, _, _, zeds_snaps, z_int_steps, s_int_steps, nu_massrange, halo_batch_IDs, halo_num = SimData.simulation_setup(
        sim_dir=run_dir,
        m_lower=halos["mass_lower"],
        m_upper=halos["mass_upper"],
        m_gauge=halos["mass_gauge"],
        halo_num_req=halos["halo_num"])

    with open(f'{run_dir}/sim_parameters.yaml', 'r') as file:
        pix_sr = yaml.safe_load(file)['pix_sr']
    init_vels = np.load(f'{run_dir}/initial_velocities.npy')

    pix_dens_halo_l = []
    tot_dens_halo_l = []

    for halo_j, halo_ID in enumerate(halo_batch_IDs):

        if halo_j+1 in skip:
            continue

        end_str = f'halo{halo_j+1}'
        force_args, cell_grids = Simulation.halo_force_args(
            data_dir, halo_ID, end_str, zeds_snaps, DM_mass)

        # Reduce cell count while respecting DM density
        # note: CellSelection.select_spherical_grid (Fibonacci spiral) didn't
        # give cells close enough to the center (min was ~40 kpc away).
        cell_ccs = cell_grids[-1]
        reduced_ccs = CellSelection.reduce_cells(
            cell_ccs, target_n=sim["target_cells"], k_neighbors=len(cell_ccs))

        # Starting positions in kpc, without the zeros appended in SimGrid.grid_data
        reduced_ccs_kpc = reduced_ccs/Params.kpc
        cell_dis = jnp.linalg.norm(reduced_ccs_kpc, axis=-1)
        reduced_ccs_kpc = reduced_ccs_kpc[cell_dis != 0.]

        print(f"*** Simulation for halo={halo_j+1}/{halo_num} ***")
        sim_start = time.perf_counter()

        pix_dens_cell_l = []
        tot_dens_cell_l = []
        for c, init_xyz in enumerate(reduced_ccs_kpc):

            nu_vectors = Simulation.simulate_all_sky(
                init_xyz, init_vels, s_int_steps, z_int_steps, force_args, opts, CPUs_sim)

            # Only save vectors at z_sim boundary
            jnp.save(f'{run_dir}/vectors_{end_str}_cell{c+1}.npy', nu_vectors[..., -1, :])

            pix_dens, tot_dens = Simulation.number_densities(
                nu_vectors, sim["skymap_masses"], nu_massrange, pix_sr)
            pix_dens_cell_l.append(pix_dens)
            tot_dens_cell_l.append(tot_dens)

        pix_dens_halo_l.append(pix_dens_cell_l)
        tot_dens_halo_l.append(tot_dens_cell_l)

        t = time.perf_counter() - sim_start
        print(f"Simulation time: {t/60.:.2f} min, {t/(60**2):.2f} h")

    jnp.save(f"{run_dir}/pixel_densities.npy", jnp.array(pix_dens_halo_l))
    jnp.save(f"{run_dir}/total_densities.npy", jnp.array(tot_dens_halo_l))


if __name__ == "__main__":
    simulate_neutrino_halo(config_from_cli(__doc__))
