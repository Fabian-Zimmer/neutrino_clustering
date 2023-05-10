#!/bin/bash

sim_fullname=single_halos

python sim_plots.py \
--sim_directory L025N752/DMONLY/SigmaConstant00/$sim_fullname \
--sim_type 'single_halos' \
--NFW_halo \
--box_halos \
--analytical_halo \