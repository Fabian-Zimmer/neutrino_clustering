#!/bin/bash

sim_fullname=spheres_high_res

python sim_plots.py \
--sim_directory L025N752/DMONLY/SigmaConstant00/$sim_fullname \
--sim_type 'spheres' \
--shells 1 \
--no-NFW_halo \
--box_halos \
--no-analytical_halo \