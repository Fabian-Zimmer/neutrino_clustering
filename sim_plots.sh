#!/bin/bash

sim_fullname=low_res_spheres

python sim_plots.py \
--sim_directory L025N752/DMONLY/SigmaConstant00/$sim_fullname \
--sim_type 'spheres' \
--no-NFW_halo \
--box_halos \
--analytical_halo \