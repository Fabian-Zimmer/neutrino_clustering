"""Gravitational forces felt by a neutrino during the simulation: lookup of the
precalculated cell force, quadrupole force outside the grid, the Sun."""


import jax
import jax.numpy as jnp

from cnb.units import Params


class Forces():

    @jax.jit
    def nu_in_which_cell(x_i, cell_coords, cell_gens, init_GRID_S):

        # Number of cells.
        num_cells = cell_coords.shape[0]

        # "Center" the neutrino coordinates on the cell coordinates.
        x_cent = x_i - cell_coords.reshape(num_cells, 3)

        # All cell lengths. Limit for the largest cell is GRID_S/2, not just 
        # GRID_S, therefore the cell_gen+1 !
        cell_lens = init_GRID_S / (2**(cell_gens + 1))

        # Find index of cell in which neutrino is enclosed.
        in_cell = (
            (jnp.abs(x_cent[..., 0]) < cell_lens) & 
            (jnp.abs(x_cent[..., 1]) < cell_lens) & 
            (jnp.abs(x_cent[..., 2]) < cell_lens)
        )
        cell_idx = jnp.argmax(in_cell)
        cell_len = cell_lens[cell_idx]
        cell_ccs = cell_coords[cell_idx, :]

        return cell_idx, cell_len, cell_ccs


    @jax.jit
    def outside_gravity_quadrupole(x_i, com_halo, DM_sim_mass, DM_num, QJ_abs):

        ### ----------- ###
        ### Quadrupole. ###
        ### ----------- ###

        # Center neutrino on c.o.m. of halo and get distance.
        x_i = x_i - com_halo
        r_i = jnp.linalg.norm(x_i)

        # Permute order of coords by one, i.e. (x,y,z) -> (z,x,y).
        x_i_roll = jnp.roll(x_i, shift=1)

        # Terms appearing in the quadrupole term.
        QJ_aa = QJ_abs[0]
        QJ_ab = QJ_abs[1]

        # Factors of 2 are for the symmetry of QJ_ab elements.
        term1_aa = jnp.sum(QJ_aa * x_i, axis=0)
        term1_ab = jnp.sum(2 * QJ_ab * x_i_roll, axis=0)
        term1 = (term1_aa + term1_ab) / r_i**5

        term2_pre = 5 * x_i / (2 * r_i**7)
        term2_aa = jnp.sum(QJ_aa * x_i**2, axis=0)
        term2_ab = jnp.sum(2 * QJ_ab * x_i * x_i_roll, axis=0)
        term2 = term2_pre * (term2_aa + term2_ab)

        dPsi_multipole_cells = Params.G * DM_sim_mass * (-term1 + term2)

        ### --------- ###
        ### Monopole. ###
        ### --------- ###

        dPsi_monopole_cells = Params.G * DM_num * DM_sim_mass * x_i / r_i**3

        # Total outside gravity gradient.
        gradient_out = dPsi_multipole_cells + dPsi_monopole_cells

        # Acceleration is negative value of (grav. pot.) gradient.
        return -gradient_out


    @jax.jit
    def update_sun_position_at_origin(t, km, s):

        # Sun's velocity vector w.r.t. CNB(==CMB) from Planck 2018
        # (e.g. paper I, table 3)
        # Originally in galactic coordinates (l,b), now in caretesian frame with
        # x-axis defined by Sun-GC line in this case.
        v_CNB = jnp.array([-0.06935679, -0.66222051,  0.74609224])*369.82*km/s

        # Minus sign for Sun to move in opposite direction (back in time as sim)
        return - v_CNB * t


    @jax.jit
    def sun_gravity(x_i, eps, sun_position):

        # Distance from sun
        r_i = jnp.linalg.norm(x_i - sun_position)

        # Newtonion gravity for point mass with eps offset to avoid singularity
        quot = (x_i - sun_position) / jnp.power((r_i**2 + eps**2), 3./2.)
        gradient_sun = Params.G * Params.Msun * quot
        
        # Acceleration is negative value of (grav. pot.) gradient.
        return -gradient_sun
