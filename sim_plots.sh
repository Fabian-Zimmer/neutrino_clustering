#!/bin/bash

sim_fullname=low_res_upto_Rvir

python sim_plots.py \
--sim_directory L025N752/DMONLY/SigmaConstant00/$sim_fullname \
--sim_type 'single_halos' \
--no-NFW_halo \
--box_halos \
--no-analytical_halo \