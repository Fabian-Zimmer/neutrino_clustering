#!/bin/bash

sim_fullname=all_sky_final

# python make_box_parameters_and_merger_tree.py \
# --box_directory /projects/0/einf180/Tango_sims \
# --box_name L025N752 \
# --box_version DMONLY/SigmaConstant00 \
# --sim_fullname $sim_fullname \
# --initial_snap_z0 36 \
# --final_snap_z4 12


python make_simulation_parameters.py \
--sim_dir L025N752/DMONLY/SigmaConstant00/$sim_fullname \
--sim_type all_sky \
--healpix_nside 8 \
--nu_mass_start 0.01 \
--nu_mass_stop 0.3 \
--nu_mass_num 100 \
--nu_sim_mass 0.3 \
--p_start 0.01 \
--p_stop 400 \
--p_num 10_000 \
--init_x_dis 8.178 \
--z_int_shift 0.1 \
--z_int_stop 4 \
--z_int_num 200 \
--int_solver RK23 \
--CPUs_precalculations 128 \
--CPUs_simulations 128 \
--memory_limit_GB 224 \
--DM_in_cell_limit 1_000