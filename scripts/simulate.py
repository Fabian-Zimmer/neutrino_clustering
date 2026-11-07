"""Step 4: simulate relic neutrinos back in time and compute number densities.

Neutrinos start at an Earth-like position with the velocities from
initial_velocities.npy (one set per healpix pixel) and are integrated back to
z~4 (cnb.simulation). From their final momenta the number density today follows
via Liouville's theorem (cnb.physics.Physics).

simulate.mode in the config selects the experiment:

    halos             DM halos of the box (paper 1); old sim_script23
    benchmark         NFW-like benchmark halo; old sim_script23 --benchmark
    no_gravity        no forces at all (reference); old sim_script21
    z_evolution       stop the integration at different redshifts; old sim_script25
    solar_modulation  one year of Earth orbit with the Sun's gravity (paper 3);
                      old sim_script40 (sun_gravity: false) / sim_script41

Outputs in run_dir: vectors_<halo|day>.npy (positions and velocities, first and
last z step), pixel_densities.npy, total_densities.npy (and init_xyz_*.npy).

Usage: python scripts/simulate.py configs/<run>.yaml
"""

import time
from datetime import datetime

import jax.numpy as jnp
import numpy as np
import yaml

from cnb.astro import Astro
from cnb.config import config_from_cli
from cnb.physics import Physics
from cnb.sim_data import SimData
from cnb.simulation import Simulation, SolverOptions
from cnb.units import Params


def solver_options(sim, **defaults):
    """SolverOptions from the `simulate` section, falling back to mode defaults."""
    keys = SolverOptions._fields
    values = {**defaults, **{k: v for k, v in sim.items() if k in keys}}
    if "save_all_z" in sim:
        values["save_all"] = sim["save_all_z"]
    return SolverOptions(**values)


def load_sky(run_dir):
    """Pixel size, and initial velocities of shape (Npix, neutrinos per pixel, 3)."""
    with open(f'{run_dir}/sim_parameters.yaml', 'r') as file:
        sim_setup = yaml.safe_load(file)
    init_vels = np.load(f'{run_dir}/initial_velocities.npy')
    return sim_setup['pix_sr'], init_vels


def timed(label, start):
    t = time.perf_counter() - start
    print(f"{label}: {t/60.:.2f} min, {t/(60**2):.2f} h")


### =============== ###
### DM halos (23)   ###
### =============== ###

def simulate_halos(cfg, benchmark=False):

    run_dir, data_dir = cfg["run_dir"], cfg["data_dir"]
    sim, halos = cfg["simulate"], cfg["halos"]
    skip = set(halos.get("skip", []))
    opts = solver_options(sim, force="halo")
    testing = sim.get("testing", False)

    # Create halo batch, files and other simulation setup parameters and arrays
    DM_mass, CPUs_sim, _, init_dis, zeds_snaps, z_int_steps, s_int_steps, nu_massrange, halo_batch_IDs, halo_num = SimData.simulation_setup(
        sim_dir=run_dir,
        m_lower=halos["mass_lower"],
        m_upper=halos["mass_upper"],
        m_gauge=halos["mass_gauge"],
        halo_num_req=halos["halo_num"])

    pix_sr, init_vels = load_sky(run_dir)

    # Lists for pixel and total number densities
    pix_dens_l = []
    tot_dens_l = []

    for halo_j, halo_ID in enumerate(halo_batch_IDs):

        # note: "Broken" halos (e.g. no DM position data at snapshot 0012).
        if halo_j+1 in skip:
            continue

        end_str = 'benchmark_halo' if benchmark else f'halo{halo_j+1}'

        force_args, cell_grids = Simulation.halo_force_args(
            data_dir, halo_ID, end_str, zeds_snaps, DM_mass, benchmark=benchmark)

        init_xyz = Simulation.earth_like_start(cell_grids, init_dis)
        jnp.save(f'{run_dir}/init_xyz_{end_str}.npy', init_xyz)

        if benchmark:
            print("*** Simulation for benchmark halo ***")
        else:
            print(f"*** Simulation for halo={halo_j+1}/{halo_num} ***")

        sim_start = time.perf_counter()
        nu_vectors = Simulation.simulate_all_sky(
            init_xyz, init_vels, s_int_steps, z_int_steps, force_args, opts, CPUs_sim, testing)

        # Save all sky neutrino vectors for current halo
        jnp.save(f'{run_dir}/vectors_{end_str}.npy', nu_vectors)
        timed("Simulation time", sim_start)

        # With save_all_z the vectors hold every z step; densities need only
        # the first and last, so they are computed later in the analysis.
        if not opts.save_all and not testing:
            ana_start = time.perf_counter()
            pix_dens, tot_dens = Simulation.number_densities(
                nu_vectors, sim["skymap_masses"], nu_massrange, pix_sr)
            pix_dens_l.append(pix_dens)
            tot_dens_l.append(tot_dens)
            timed("Analysis time", ana_start)

        # Benchmark (NFW-like) halo is considered halo 0, so break off loop here
        if benchmark:
            break

    # Save number density arrays for all halos
    if tot_dens_l:
        jnp.save(f"{run_dir}/total_densities.npy", jnp.array(tot_dens_l))
        jnp.save(f"{run_dir}/pixel_densities.npy", jnp.array(pix_dens_l))


### ================= ###
### No gravity (21)   ###
### ================= ###

def simulate_no_gravity(cfg):

    run_dir, sim = cfg["run_dir"], cfg["simulate"]
    opts = solver_options(sim, force="none", relativistic=False)
    testing = sim.get("testing", False)

    _, CPUs_sim, _, init_dis, _, z_int_steps, s_int_steps, nu_massrange = SimData.simulation_setup(
        sim_dir=run_dir, m_lower=None, m_upper=None, m_gauge=None, halo_num_req=None,
        no_gravity=True)

    pix_sr, init_vels = load_sky(run_dir)

    # Initial position (Earth); file name ending as for a halo.
    end_str = 'halo1'
    init_xyz = np.array([float(init_dis), 0., 0.])
    jnp.save(f'{run_dir}/init_xyz_{end_str}.npy', init_xyz)

    print("*** Simulation for no_gravity ***")
    sim_start = time.perf_counter()
    nu_vectors = Simulation.simulate_all_sky(
        init_xyz, init_vels, s_int_steps, z_int_steps, (), opts, CPUs_sim, testing)
    jnp.save(f'{run_dir}/vectors_{end_str}.npy', nu_vectors)
    timed("Simulation time", sim_start)

    if not testing:
        pix_dens, tot_dens = Simulation.number_densities(
            nu_vectors, sim["skymap_masses"], nu_massrange, pix_sr)
        jnp.save(f"{run_dir}/pixel_densities.npy", jnp.array([pix_dens]))
        jnp.save(f"{run_dir}/total_densities.npy", jnp.array([tot_dens]))


### ================== ###
### z evolution (25)   ###
### ================== ###

def simulate_z_evolution(cfg):
    """Same as `halos`, but the integration stops at 20 different redshifts z_back."""

    run_dir, data_dir = cfg["run_dir"], cfg["data_dir"]
    sim, halos, par = cfg["simulate"], cfg["halos"], cfg["parameters"]
    skip = set(halos.get("skip", []))
    opts = solver_options(sim, force="halo", relativistic=False)

    DM_mass, CPUs_sim, _, init_dis, zeds_snaps, z_int_steps_orig, _, nu_massrange, halo_batch_IDs, halo_num = SimData.simulation_setup(
        sim_dir=run_dir,
        m_lower=halos["mass_lower"],
        m_upper=halos["mass_upper"],
        m_gauge=halos["mass_gauge"],
        halo_num_req=halos["halo_num"])

    simdata = SimData(run_dir)
    _, init_vels = load_sky(run_dir)
    tot_dens_l = []

    for halo_j, halo_ID in enumerate(halo_batch_IDs):

        if halo_j+1 in skip:
            continue

        end_str = f'halo{halo_j+1}'
        force_args, cell_grids = Simulation.halo_force_args(
            data_dir, halo_ID, end_str, zeds_snaps, DM_mass)
        init_xyz = Simulation.earth_like_start(cell_grids, init_dis)
        jnp.save(f'{run_dir}/init_xyz_{end_str}.npy', init_xyz)

        print(f"*** Simulation for halo={halo_j+1}/{halo_num} ***")
        sim_start = time.perf_counter()

        tot_dens_zed_l = []
        for z_back in z_int_steps_orig[jnp.arange(1,99,5)]:

            # Logarithmic redshift spacing up to z_back, and conversion to s.
            z_int_shift = par["z_int_shift"]
            z_int_steps = np.geomspace(z_int_shift, z_back+z_int_shift, par["z_int_num"])
            z_int_steps -= z_int_shift
            s_int_steps = np.array([
                Physics.s_of_z(z, simdata.h, simdata.Omega_M, simdata.Omega_L)
                for z in z_int_steps])

            nu_vectors = Simulation.simulate_all_sky(
                init_xyz, init_vels, s_int_steps, z_int_steps, force_args, opts, CPUs_sim)

            tot_dens = Physics.number_densities_mass_range(
                v_arr=nu_vectors.reshape(-1, 2, 6)[..., 3:],
                m_arr=nu_massrange,
                pix_sr=4*Params.Pi,
                args=Params())
            tot_dens_zed_l.append(jnp.squeeze(tot_dens))

            # Save all sky neutrino vectors for current halo at current z_back
            jnp.save(f'{run_dir}/vectors_{end_str}_z={z_back:.2f}.npy', nu_vectors[..., -1, :])

        tot_dens_l.append(tot_dens_zed_l)
        timed("Simulation time", sim_start)

    jnp.save(f"{run_dir}/total_densities_zeds.npy", jnp.array(tot_dens_l))


### ======================== ###
### Solar modulation (40/41) ###
### ======================== ###

def simulate_solar_modulation(cfg):
    """
    Earth stays at the origin (Earth-GC frame) and the Sun moves. Each simulated day integrates one year back in time. Needs z/s/t_int_steps_*year(s).npy from scripts/make_solar_time_steps.py.
    """

    run_dir, sim = cfg["run_dir"], cfg["simulate"]
    sun_gravity = sim.get("sun_gravity", True)
    opts = solver_options(
        sim, force="sun" if sun_gravity else "none", relativistic=True,
        throw=False, initial_step="median")
    testing = sim.get("testing", False)

    _, CPUs_sim, *_ = SimData.simulation_setup(
        sim_dir=run_dir, m_lower=None, m_upper=None, m_gauge=None, halo_num_req=None,
        no_gravity=True)

    print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))

    # Sun's positions and velocities (w.r.t. CNB==CMB) in Earth-GC frame for
    # 1 year. This function agrees with Safdi et al.
    _, sun_positions, sun_velocities, _ = Astro.get_sun_earth_galactic_data(
        year=2024, rel_vel="MW", Earth_rel_Sun=False)
    sun_positions *= Params.AU
    sun_velocities *= Params.km/Params.s

    # Redshift from today until (365*2)+1 days ago, and for one year the
    # corresponding integration variable s and lookback time t.
    z_int_steps_all = jnp.load(f'{run_dir}/z_int_steps_2years.npy')
    s_int_steps_1year = jnp.load(f'{run_dir}/s_int_steps_1year.npy')
    t_int_steps_1year = jnp.load(f'{run_dir}/t_int_steps_1year.npy')

    # Earth is at (0,0,0) always in Earth-GC frame, and we move the Sun
    init_xyz = np.array([0., 0., 0.])
    jnp.save(f'{run_dir}/init_xyz_modulation.npy', init_xyz)

    # Initial velocities, same for each day due to Earth-GC frame
    _, init_vels = load_sky(run_dir)

    # note: the DM cell force at Earth's position was tried as an extra,
    # constant force in earlier versions, but not used for the results.

    # Days to simulate. The final runs used days 125 and 305, since we go back
    # in time: 365-240=125 and 365-60=305. Every 12th day shows the trends.
    days = sim.get("days") or list(range(0, 365, sim.get("day_step", 12)))

    for day in days:

        # One year of redshift steps, +1 because earlier versions picked the
        # second last time step (see note in Simulation.backtrack_1_neutrino).
        z_int_steps_1year = z_int_steps_all[day:day+365+1]
        end_str = f"day{day+1}"

        print(f"*** Simulation for day {day+1}/365 ***")
        sim_start = time.perf_counter()

        force_args = Simulation.sun_force_args(
            t_int_steps_1year, sun_positions[day], sun_velocities[day]) if sun_gravity else ()

        nu_vectors = Simulation.simulate_all_sky(
            init_xyz, init_vels, s_int_steps_1year, z_int_steps_1year, force_args, opts,
            CPUs_sim, testing)

        suffix = "_TEST" if testing else ""
        jnp.save(f'{run_dir}/vectors_{end_str}{suffix}.npy', nu_vectors)
        timed("Simulation time", sim_start)


MODES = {
    "halos": simulate_halos,
    "benchmark": lambda cfg: simulate_halos(cfg, benchmark=True),
    "no_gravity": simulate_no_gravity,
    "z_evolution": simulate_z_evolution,
    "solar_modulation": simulate_solar_modulation,
}


if __name__ == "__main__":
    total_start = time.perf_counter()
    cfg = config_from_cli(__doc__)
    MODES[cfg["simulate"]["mode"]](cfg)
    timed("Total time", total_start)
