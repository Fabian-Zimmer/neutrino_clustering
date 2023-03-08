#!/bin/bash

python simulation_analytical.py \
-d L025N752/DMONLY/SigmaConstant00 \
--MW_halo \
--no-VC_halo \
--no-AG_halo


python simulation_numerical.py \
-d L025N752/DMONLY/SigmaConstant00 \
-st single_halos \
-mg 12.0 \
-mr 0.6 \
-hn 3