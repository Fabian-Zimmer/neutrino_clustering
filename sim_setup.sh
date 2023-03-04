#!/bin/bash

python make_box_parameters_and_merger_tree.py \
-bd /projects/0/einf180/Tango_sims \
-bn L025N752 \
-bv DMONLY/SigmaConstant00 \
-zi 36 \
-zf 12

python make_simulation_parameters.py \
-sd L025N752/DMONLY/SigmaConstant00 \
-st halo_batch \
-ni 0.01 \
-nf 0.3 \
-nn 100 \
-nm 0.3 \
-pi 0.01 \
-pf 400 \
-pn 100 \
-ph 10 \
-th 10 \
-xi 8.5 \
-zi 0.1 \
-zf 4 \
-zn 100 \
-is RK23 \
-cp 128 \
-cs 128 \
-mem 224 \
-dl 10_000