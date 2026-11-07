"""Equations of motion and the ODE solver for neutrinos travelling back in time.

All simulation variants (DM halos, benchmark halo, no gravity, z evolution,
solar modulation, neutrino halo) use this module. They differ only in:

    - the force on the neutrino:   SolverOptions.force = "none" | "halo" | "sun"
    - the equations of motion:     SolverOptions.relativistic
    - what is saved:               SolverOptions.save_all

The `force_args` tuple holds the arrays the chosen force needs, see
`Simulation.halo_force_args` and `Simulation.sun_force_args`.

Units: the solver works in "physical reality" (positions in kpc, velocities in
kpc/s). Forces are computed in "numerical reality" (Params units) and converted
back, as in the original scripts.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp

from cnb.forces import Forces
from cnb.grid import SimGrid
from cnb.physics import Physics
from cnb.units import Params
from cnb.utils import Utils


class SolverOptions(NamedTuple):
    """Static settings of one simulation (hashable, so jax.jit treats them as static)."""

    force: str = "halo"           # "none", "halo" (precalculated DM grid) or "sun"
    relativistic: bool = True     # relativistic EOMs (default since the JAX version)
    save_all: bool = False        # keep every z step, not only the first and last
    rtol: float = 1e-3            # note: no change for tighter rtol and atol,
    atol: float = 1e-6            # e.g. rtol=1e-5, atol=1e-9
    max_steps: int = 10_000
    throw: bool = True            # raise if the solver fails (False for solar runs)
    initial_step: str = "first"   # dt0 = (s[0]+s[1])/1000, or "median": median spacing/1000
                                  # (solar runs, where s[0] == s[1] == 0)


class Simulation():
    """Backtracking of neutrinos: forces, equations of motion and the solver."""


    ### ====== ###
    ### Forces ###
    ### ====== ###
    # Each force takes the neutrino position x_i in numerical units and returns
    # the gradient of the potential in numerical units.

    @staticmethod
    def gravity_none(x_i, s_val, z, s_int_steps, force_args):
        return jnp.zeros(3)


    @staticmethod
    def gravity_halo(x_i, s_val, z, s_int_steps, force_args):
        """
        Gravity of the DM halo of a cosmological simulation. Inside the grid, the precalculated force of the cell the neutrino is in; outside, the long-range force (incl. quadrupole) of the whole grid content. The snapshot closest to the current redshift is used.
        """

        zeds_snaps, snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, dPsi_grid_data, cell_grid_data, cell_gens_data, DM_mass = force_args

        # Snapshot specific parameters.
        idx = jnp.abs(zeds_snaps - z).argmin()
        snap_GRID_L = snaps_GRID_L[idx]

        def inside_cell(_):

            # Load files for current z, to find in which cell neutrino is.
            # Then load gravity for that cell.
            dPsi_grid = dPsi_grid_data[idx]
            cell_grid = cell_grid_data[idx]
            cell_gens = cell_gens_data[idx]

            cell_idx, *_ = Forces.nu_in_which_cell(
                x_i, cell_grid, cell_gens, snap_GRID_L)
            grad_tot = dPsi_grid[cell_idx, :]

            return grad_tot

        def outside_cell(_):

            # Apply long range force (incl. quadrupole) of whole grid content.
            DM_com = snaps_DM_com[idx]
            DM_num = snaps_DM_num[idx]
            QJ_abs = snaps_QJ_abs[idx]
            grad_tot = Forces.outside_gravity_quadrupole(
                x_i, DM_com, DM_mass, DM_num, QJ_abs)

            return grad_tot

        return jax.lax.cond(
            jnp.all(jnp.abs(x_i) < snap_GRID_L), inside_cell, outside_cell, None)


    @staticmethod
    def gravity_sun(x_i, s_val, z, s_int_steps, force_args):
        """
        Gravity of the Sun, moving with its velocity w.r.t. the CNB(==CMB) frame, in the Earth-GC frame (Earth fixed at the origin).
        """

        t_int_steps, sun_pos, sun_vel = force_args

        # Find t (lookback time) corresponding to s_val via interpolation.
        t = Utils.jax_interpolate(s_val, s_int_steps, t_int_steps)*Params.s

        # Current position of Sun, given starting position and velocity vectors
        # of the day being simulated.
        sun_pos_t = sun_pos - sun_vel*t

        eps = 696_340*Params.km  # solar radius in numerical units
        return Forces.sun_gravity(x_i, eps, sun_pos_t)


    ### ================== ###
    ### Equations of motion ###
    ### ================== ###

    @staticmethod
    def EOMs(s_val, y, args, gravity, relativistic):
        """
        Equations of motion of a neutrino in an expanding universe, in the time variable s (Mertsch et al. 2020). Returns the rate of change of position and velocity, with a global minus sign so the integration goes back in time.
        """

        s_int_steps, z_int_steps, force_args = args

        # Initialize vector.
        x_i, u_i = y

        # Find z corresponding to s via interpolation.
        z = Utils.jax_interpolate(s_val, s_int_steps, z_int_steps)

        # Force in "numerical reality", then switch back to "physical reality".
        grad_tot = gravity(x_i*Params.kpc, s_val, z, s_int_steps, force_args)
        grad_tot /= (Params.kpc/Params.s**2)

        if relativistic:
            dxds = u_i/(1+z)**2 / jnp.sqrt(jnp.sum(u_i**2) + (1+z)**-2)
        else:
            dxds = u_i

        # Hamilton eqns. for integration (global minus, s.t. we go back in time).
        return -jnp.array([dxds, 1./(1.+z)**2 * grad_tot])


    @staticmethod
    @partial(jax.jit, static_argnames=("opts",))
    def backtrack_1_neutrino(init_vector, s_int_steps, z_int_steps, force_args, opts):
        """
        Simulate trajectory of 1 neutrino. Input is 6-dim. vector containing starting positions and velocities of neutrino. Solves the EOMs with the Dopri5 solver of diffrax. Returns the first and last positions and velocities, shape (2, 6), or all of them if opts.save_all, shape (z steps, 6).
        """

        gravity = {
            "none": Simulation.gravity_none,
            "halo": Simulation.gravity_halo,
            "sun": Simulation.gravity_sun,
        }[opts.force]

        def EOMs(s_val, y, args):
            return Simulation.EOMs(s_val, y, args, gravity, opts.relativistic)

        # Initial vector in correct shape for EOMs function
        y0 = init_vector.reshape(2,3)

        # ODE solver setup
        term = diffrax.ODETerm(EOMs)
        t0 = s_int_steps[0]
        t1 = s_int_steps[-1]
        if opts.initial_step == "median":
            dt0 = jnp.median(jnp.diff(s_int_steps)) / 1000
        else:
            dt0 = (s_int_steps[0] + s_int_steps[1]) / 1000

        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=opts.rtol, atol=opts.atol)
        # Alternative: diffrax.Dopri8() with rtol=1e-7, atol=1e-9.

        # Specify timesteps where solutions should be saved
        saveat = diffrax.SaveAt(ts=jnp.array(s_int_steps))

        # Solve the coupled ODEs, i.e. the EOMs of the neutrino
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=t0, t1=t1, dt0=dt0, y0=y0, max_steps=opts.max_steps,
            saveat=saveat, stepsize_controller=stepsize_controller,
            args=(s_int_steps, z_int_steps, force_args), throw=opts.throw)

        trajectory = sol.ys.reshape(-1,6)

        if opts.save_all:
            return trajectory

        # note: for solar runs (throw=False) the last steps can be inf, if the
        # solver stops close to the end point. Earlier versions then used [-2].
        return jnp.stack([trajectory[0], trajectory[-1]])


    @staticmethod
    def simulate_neutrinos_1_pix(init_xyz, init_vels, s_int_steps, z_int_steps, force_args, opts):
        """
        Simulate all neutrinos of 1 healpix pixel: same starting position, different velocities. One process runs this for one pixel.
        """

        # Neutrinos per pixel
        nus = init_vels.shape[0]

        # Make vector with same starting position but different velocities
        init_vectors = jnp.array(
            [jnp.concatenate((init_xyz, init_vels[k])) for k in range(nus)])

        trajectories = jnp.array([
            Simulation.backtrack_1_neutrino(vec, s_int_steps, z_int_steps, force_args, opts)
            for vec in init_vectors])

        return trajectories  # shape = (neutrinos, 2, 6)


    @staticmethod
    def simulate_all_sky(init_xyz, init_vels, s_int_steps, z_int_steps, force_args, opts, CPUs, testing=False):
        """
        Simulate all pixels of the sky, one process per pixel. init_vels has shape (Npix, neutrinos per pixel, 3). With testing=True only the first pixel is simulated, without multiprocessing.
        """

        if testing:
            return Simulation.simulate_neutrinos_1_pix(
                init_xyz, init_vels[0], s_int_steps, z_int_steps, force_args, opts)

        with ProcessPoolExecutor(CPUs) as executor:
            futures = [
                executor.submit(
                    Simulation.simulate_neutrinos_1_pix,
                    init_xyz, init_vels[pixel], s_int_steps, z_int_steps, force_args, opts)
                for pixel in range(len(init_vels))
            ]

            # Wait for all futures to complete and collect results in order
            return jnp.array([future.result() for future in futures])


    ### ======================= ###
    ### Setup and post-processing ###
    ### ======================= ###

    @staticmethod
    def halo_force_args(data_dir, halo_ID, end_str, zeds_snaps, DM_mass, benchmark=False):
        """
        Load the precalculated grid of one halo (from scripts/precalculate_grids.py) as force_args for force="halo". Also returns the cell grids, to find the starting cell.
        """

        #! Important:
        # The precalculation ran from the earliest snapshot (z~4 for us) to the
        # latest (z=0). So the below arrays are in this order. Even though our
        # simulation runs backwards in time, we can leave them like this, since
        # the correct element gets picked with the idx routine in gravity_halo.
        snaps_GRID_L = jnp.load(f'{data_dir}/snaps_GRID_L_{end_str}.npy')
        snaps_DM_num = jnp.load(f'{data_dir}/snaps_DM_num_{end_str}.npy')
        snaps_DM_com = jnp.load(f'{data_dir}/snaps_DM_com_{end_str}.npy')
        snaps_QJ_abs = jnp.load(f'{data_dir}/snaps_QJ_abs_{end_str}.npy')

        # Load grav. forces, coordinates and generation/lengths of cells in grid.
        dPsi_grids, cell_grids, cell_gens = SimGrid.grid_data(
            halo_ID, data_dir, benchmark=benchmark)

        force_args = (
            zeds_snaps, snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs,
            dPsi_grids, cell_grids, cell_gens, DM_mass)

        return force_args, cell_grids


    @staticmethod
    def earth_like_start(cell_grids, init_dis):
        """
        Centre of the z=0 cell whose distance from the halo centre is closest to init_dis (in kpc), i.e. an Earth-like position. Returned in kpc, without units, as starting position of the simulation.
        """

        # Load grid data and compute radial distances from center of cell centers.
        cell_ccs = cell_grids[-1]
        cell_ccs_kpc = cell_ccs/Params.kpc
        cell_dis = jnp.linalg.norm(cell_ccs_kpc, axis=-1)

        # Get rid of zeros we appended when loading data (see SimGrid.grid_data)
        cell_dis = cell_dis[cell_dis != 0.]

        # Take first cell, which is in Earth-like position (there can be multiple).
        # note: for some reason this routine chooses a different cell to those in
        # the paper, ergo the all-sky maps look different for the same halos.
        return cell_ccs[jnp.abs(cell_dis - init_dis).argsort()][0]/Params.kpc.flatten()


    @staticmethod
    def sun_force_args(t_int_steps, sun_pos, sun_vel):
        """force_args for force="sun": lookback times, Sun position and velocity of one day."""
        return (t_int_steps, sun_pos, sun_vel)


    @staticmethod
    def number_densities(nu_vectors, skymap_masses, nu_massrange, pix_sr):
        """
        Number densities from simulated vectors of shape (Npix, neutrinos, 2, 6): per healpix pixel for the masses in skymap_masses (eV), and in total (all pixels) for the masses in nu_massrange (numerical units).
        """

        pix_dens = Physics.number_densities_all_sky(
            v_arr=nu_vectors[..., 3:],
            m_arr=jnp.array(skymap_masses)*Params.eV,
            pix_sr=pix_sr,
            args=Params())

        # Compute total number density, by using all neutrino vectors for integral
        tot_dens = Physics.number_densities_mass_range(
            v_arr=nu_vectors.reshape(-1, 2, 6)[..., 3:],
            m_arr=nu_massrange,
            pix_sr=4*Params.Pi,
            args=Params())

        return jnp.squeeze(pix_dens), jnp.squeeze(tot_dens)
