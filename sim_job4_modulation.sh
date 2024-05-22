#!/bin/bash
#SBATCH --job-name=DaysSim
#SBATCH -p rome
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112000
#SBATCH -o jobs/DaysSim.out
#SBATCH -e jobs/DaysSim.err
#SBATCH -v


# rome  node- full: 128/224 ; half: 64/112 ; quarter: 32/56 ; CPUs/GB.
# genoa node- full: 192/336 ; half: 96/168 ; quarter: 48/84 ; CPUs/GB.


module purge

# Load .bashrc if it exists
if [ -f "$HOME/.bashrc" ]; then
    source $HOME/.bashrc
fi

conda activate neutrino_clustering


sim_fullname=Dopri5_1k

python $HOME/neutrino_clustering/sim_script4_modulation.py \
--directory $HOME/neutrino_clustering/sim_output/$sim_fullname