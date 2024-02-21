#!/bin/bash

sim_fullname=Dopri8_1k_per_pix

python make1_simulation_box.py \
--box_directory /projects/0/prjs0781/simulation_data/Tango_sims \
--sim_fullname $sim_fullname \
--initial_snap_z0 36 \
--final_snap_z4 12

python make2_simulation_parameters.py \
--sim_dir sim_output/$sim_fullname \
--healpix_nside 8 \
--nu_mass_start 0.01 \
--nu_mass_stop 0.3 \
--nu_mass_num 100 \
--nu_sim_mass 0.3 \
--p_start 0.01 \
--p_stop 400 \
--p_num 1_000 \
--init_x_dis 8.178 \
--z_int_shift 0.1 \
--z_int_stop 4 \
--z_int_num 100 \
--CPUs_precalculations 128 \
--CPUs_simulations 128 \
--memory_limit_GB 224 \
--DM_in_cell_limit 1_000