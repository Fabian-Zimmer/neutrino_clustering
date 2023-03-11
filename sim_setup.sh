#!/bin/bash

sim_suffix=high_res

python make_box_parameters_and_merger_tree.py \
--box_directory /projects/0/einf180/Tango_sims \
--box_name L025N752 \
--box_version DMONLY/SigmaConstant00/$sim_suffix \
--initial_snap_z0 36 \
--final_snap_z4 12


python make_simulation_parameters.py \
--sim_dir L025N752/DMONLY/SigmaConstant00/$sim_suffix \
--sim_type single_halos \
--nu_mass_start 0.01 \
--nu_mass_stop 0.3 \
--nu_mass_num 100 \
--nu_sim_mass 0.3 \
--p_start 0.01 \
--p_stop 400 \
--p_num 100 \
--phis 10 \
--thetas 10 \
--init_x_dis 8.5 \
--z_int_shift 0.1 \
--z_int_stop 4 \
--z_int_num 100 \
--int_solver RK23 \
--CPUs_precalculations 128 \
--CPUs_simulations 128 \
--memory_limit_GB 224 \
--DM_in_cell_limit 10_000