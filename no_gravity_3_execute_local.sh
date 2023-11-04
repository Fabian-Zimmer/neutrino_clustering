#!/bin/bash

sim_boxtype=L025N752/DMONLY/SigmaConstant00
sim_fullname=no_gravity
local_folder=neutrino_clustering_V2

python $HOME/$local_folder/no_gravity_2_simulation.py \
--directory $HOME/$local_folder/$sim_boxtype/$sim_fullname \
--sim_type all_sky \
--mass_gauge 12.0 \
--mass_lower 0.6 \
--mass_upper 2.0 \
--halo_num 1 \
--no-upto_Rvir \
--no-gravity