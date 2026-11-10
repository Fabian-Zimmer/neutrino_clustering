# Side projects

Each folder uses the `cnb` package and the run folders of the main pipeline
(see the root README). These projects are not finished or not maintained:
expect to adjust paths, parameters or code before they run. Run their scripts
from the repo root, e.g. `uv run python projects/neutrino_halo/simulate_neutrino_halo.py projects/neutrino_halo/neutrino_halo.yaml`.

## primordial_fluctuations

Adds primordial temperature fluctuations of the CNB to the initial conditions
(paper 2, [arXiv:2407.14582](https://doi.org/10.48550/arXiv.2407.14582)).
The fluctuation maps are drawn from angular power spectra (Cl's) computed with
CLASS; each simulated neutrino gets the fluctuation of its sky pixel and
momentum at the end of its trajectory, and the densities are recomputed.

- `fluctuation_densities.py`: one realisation of the fluctuations; replaces
  the old `sim_script30` and the four `sim_script3_..._z*` copies.
- `fluctuation_densities_seeds.py`: 10 realisations (old `sim_script31`).
- `primordial.yaml`: run folder, Cl folder, redshift cut, masses, seeding.
- `notebooks/z_cut_*.ipynb`: analyses of runs that stop at z = 0.15, 0.5, 1, 2.
- Needs: a `halos` run in `runs/` and the Cl files `data/primordial/Cls/Cls_z=<z>_m=<m>eV.npy`
  (made with CLASS, not in this repo). For a redshift cut, make a run with
  `parameters.z_int_stop` set to that redshift.
- The class `cnb.primordial.Primordial` and the paper-2 plots in `cnb.plotting`
  are part of the main package, because the main notebooks use them too.

## neutrino_halo

Densities for ~300 starting cells spread across one halo, instead of only the
Earth-like position: a 3D picture of the "neutrino halo".

- `simulate_neutrino_halo.py` + `neutrino_halo.yaml`: the simulation (old `sim_script50`).
- `cell_selection.py`: picks the start cells (`reduce_cells`; the Fibonacci
  sphere method `select_spherical_grid` gave cells too far from the centre).
- `neutrino_halo.ipynb`: grid construction and density profiles; reads
  `runs/neutrino_halo/` (results of the earlier run are there).
- Needs the grids of the halo in `data/halo_grids` (`scripts/precalculate_grids.py`).

## neutrino_decay

Neutrino decay (heavier neutrinos decaying into lighter ones) during the
backwards simulation (with Maria), and the signal in a
PTOLEMY-like experiment.

- `decay.py` (class `Decay`): decay rates, sampling of decays, parent and
  daughter number densities. It was deleted from the old Shared repo in 2024
  while the scripts still imported it, and is restored here.
- `decay_maria.py` (class `decay`): Maria's original version, used by
  `maria_simulate_no_gravity.py` and some archived notebooks.
- `simulate_no_gravity_decay.py`, `simulate_dm_halos_decay.py`: simulations
  with decays, own EOMs (old `sim_script22`, `sim_script24`). They still take
  command-line arguments; the old job files ran, for example:

  ```bash
  uv run python projects/neutrino_decay/simulate_no_gravity_decay.py \
    --directory runs/no_gravity_decay --m_lightest 0.05 --gamma 2T \
    --no-pixel_densities --total_densities
  uv run python projects/neutrino_decay/simulate_dm_halos_decay.py \
    --directory runs/Dopri5_1k_decay --data_dir data/halo_grids \
    --mass_gauge 12.0 --mass_lower 0.6 --mass_upper 2.0 --halo_num 10 \
    --m_lightest 0.05 --gamma 2T --no-pixel_densities --total_densities --no-benchmark
  ```

  The run folder needs the decay input arrays made in
  `notebooks/no_gravity_decay.ipynb` (`decayed_neutrinos_*.npy`). The scripts
  use multiprocessing without a `__main__` guard: run them on Linux.
- `compute_allowed_decay_angles_and_momenta.py`: decay kinematics, reads
  Maria's files from `data/decay`.
- `ptolemy.py` (class `Ptolemy`): beta-decay spectrum and capture rates for
  PTOLEMY, including decays. Not used by any script.
- `decay_plots.py` (class `DecayPlot`): plots used by `notebooks/dm_halos_decay.ipynb`.
- `notebooks/`: analyses; `notebooks/archive/` holds the temporary sketch notebooks.
