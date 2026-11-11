# N-1-body simulations of the cosmic neutrino background

**Authors:** Fabian Zimmer, Camila Correa, Guillermo Franco Abellán, Shin'ichiro Ando

This code computes how many relic neutrinos (the cosmic neutrino background,
CNB) there are on Earth today, and from which directions they come. It
follows neutrinos **backwards in time**: each neutrino starts at Earth's
position today with a given momentum and direction, and the code integrates
its equations of motion back to redshift z ≈ 4 through the gravity of a dark
matter (DM) halo. Phase space is conserved along the path (Liouville), so the
density today is the Fermi-Dirac value at the momentum the neutrino had back
then. Integrating over all momenta and directions gives the number density
and all-sky maps.

The DM halos come from a cosmological N-body simulation (the "box", SWIFT +
VELOCIraptor, on Snellius). The code loads the DM particles of one halo per
snapshot, divides space into cells (smaller where there is more DM) and
precalculates the gravitational force in each cell. During the simulation it
looks up the force of the cell the neutrino is in.

| Paper | What | Code |
| --- | --- | --- |
| [2306.16444](https://doi.org/10.48550/arXiv.2306.16444) Influence of local structure on relic neutrino abundances and anisotropies | MW-like halos of the box, benchmark NFW halo | version 1.0 (before the JAX rewrite); same pipeline as below |
| [2407.14582](https://doi.org/10.48550/arXiv.2407.14582) Effects of primordial fluctuations on relic neutrino simulations | primordial temperature fluctuations in the initial conditions | `configs/dm_halos.yaml`, [projects/primordial_fluctuations](projects/primordial_fluctuations), notebooks `dm_halos`, `no_gravity`, `benchmark_halo` |
| [2507.01505](https://doi.org/10.48550/arXiv.2507.01505) Numerical treatment of annual modulation of relic neutrinos | the Sun's gravity over one year | `configs/solar_modulation*.yaml`, [scripts/annual_modulation](scripts/annual_modulation), notebooks `annual_modulation_*` |

Version 2.0 uses JAX, diffrax and chex, which made the simulations much faster.
This research code is designed to be executed from a linux-based HPC cluster (publication results were obtained via the Snellius supercomputer array in the Netherlands), and the background dark matter simulation data that was used was archived on the accompanying storage servers. Contact fabiantje.zimmertje@gmail.com for questions.


## Install

The environment is pinned to the versions that produced the published results
(Python 3.10, JAX 0.4.23, diffrax 0.5.0, ...). Install [uv](https://docs.astral.sh/uv/)
once, then in the repo root:

```bash
uv sync                    # main dependencies + Jupyter kernel
uv sync --extra box        # + reading the raw simulation box (scripts/make_box.py)
uv sync --extra benchmark  # + sampling the benchmark halo (Linux only: commah does not build on macOS)
uv run nbstripout --install   # keep notebook outputs out of git
```

Run anything with `uv run python ...`, or select `.venv` as the Jupyter kernel.
The package `cnb` (in `src/cnb`) is installed in editable mode.


## Repository layout

```
src/cnb/        the library: physics, grid, forces, solver, plotting (see "Code map")
scripts/        one script per pipeline step; each takes a run config
configs/        one YAML file per simulation run
slurm/run.job   the only SLURM job script
notebooks/      analysis and figures (outputs stripped; see "Notebooks")
projects/       side projects built on the same code (see projects/README.md)
data/           input data and precalculated grids           (local)
runs/           one folder per simulation run: parameters + outputs (local)
figures/        figures saved by the notebooks               (local)
```

`data/`, `runs/` and `figures/` serve as local folder (i.e. not in git) that store input/output data. On Snellius the box lies in
`/projects/0/prjs0781/simulation_data/Tango_sims`; the runs and grids were in
`$HOME/neutrino_clustering/sim_output` and `.../Data`. Copy or symlink them to
`runs/` and `data/`, or set absolute paths in the configs.


## Pipeline

Every step is a script that takes one config: `uv run python scripts/<step>.py configs/<run>.yaml`.
On the cluster, submit the same command with `sbatch slurm/run.job scripts/<step>.py configs/<run>.yaml`.

| Step | Script | Reads | Writes (in `run_dir` unless noted) | Main code |
| --- | --- | --- | --- | --- |
| 1. Box | [make_box.py](scripts/make_box.py) | raw box (`box_directory`) | `box_parameters.yaml`, `zeds_snaps.npy`, `nums_snaps.npy`, `MergerTree.hdf5` | `cnb.box.SimInfo`, `cnb.box.build_tree` |
| 2. Parameters | [make_parameters.py](scripts/make_parameters.py) | `box_parameters.yaml` | `sim_parameters.yaml`, `initial_velocities.npy`, `z_int_steps.npy`, `s_int_steps.npy`, `neutrino_massrange_eV.npy`, ... | `Physics.s_of_z` |
| (2b. Benchmark halo) | [make_benchmark_halo.py](scripts/make_benchmark_halo.py) | `box_parameters.yaml`, halo sample of another run | `data_dir/benchmark_halo_snap_*.npy` | `halo_sample_z` (NFW + COMMAH) |
| (2c. Solar time steps) | [make_solar_time_steps.py](scripts/make_solar_time_steps.py) | - | `z/s/t_int_steps_1year.npy`, `..._2years.npy` | `Physics.s_of_z` |
| 3. Grids | [precalculate_grids.py](scripts/precalculate_grids.py) | box, `MergerTree.hdf5` | `data_dir/`: `dPsi_grid_*`, `fin_grid_*`, `cell_gen_*` per halo and snapshot, `snaps_*_halo<N>.npy` | `SimData.read_DM_halo_index`, `SimGrid.cell_division`, `SimGrid.cell_gravity_short_range`, `SimGrid.cell_gravity_long_range_quadrupole` |
| 4. Simulate | [simulate.py](scripts/simulate.py) | all of the above | `vectors_<halo\|day>.npy`, `pixel_densities.npy`, `total_densities.npy` | `cnb.simulation.Simulation`, `Physics.number_densities_*` |
| 5. Analyse | notebooks, [scripts/annual_modulation](scripts/annual_modulation) | `runs/<run>/` | figures, `annual_densities_*` | `cnb.plotting.SimPlot`, `cnb.analysis` |

`simulate.mode` in the config selects the experiment:

| Mode | Force on the neutrinos | Start | Old script |
| --- | --- | --- | --- |
| `halos` | precalculated DM grid of each halo in the sample | Earth-like cell, 8.178 kpc from the halo centre | `sim_script23_DM_halos.py` |
| `benchmark` | grid of the NFW benchmark halo | same | `sim_script23 --benchmark` |
| `no_gravity` | none | 8.178 kpc on the x axis | `sim_script21_no_gravity.py` |
| `z_evolution` | DM grid, integration stops at 20 redshifts | Earth-like cell | `sim_script25_z_evolution.py` |
| `solar_modulation` | the Sun, one year back, for chosen days | Earth (origin) | `sim_script40/41_solar_modulation*.py` |

Which runs belong to which config is written at the top of each file in
[configs/](configs). Configs list the commands and `sbatch` time limits that were
used. Keep `CPUs_simulations` in the config equal to `--cpus-per-task`.

### What happens in a simulation

[`cnb/simulation.py`](src/cnb/simulation.py) holds all of it:

1. `Simulation.halo_force_args` loads the grids of one halo (`SimGrid.grid_data`)
   and `Simulation.earth_like_start` picks the starting cell.
2. `Simulation.simulate_all_sky` runs one process per healpix pixel
   (`ProcessPoolExecutor`). Each process calls `Simulation.simulate_neutrinos_1_pix`,
   which solves `Simulation.backtrack_1_neutrino` for every momentum of that pixel.
3. `backtrack_1_neutrino` integrates `Simulation.EOMs` with diffrax (Dopri5,
   rtol 1e-3, atol 1e-6) in the time variable s (Mertsch et al. 2020, eq. 4.1).
   The EOMs get the redshift from s, ask the force function for the gradient
   of the potential, and return dx/ds and du/ds with a minus sign (backwards in time).
4. The forces: `gravity_halo` takes the snapshot closest in redshift; inside
   the grid it returns the precalculated force of the neutrino's cell
   (`Forces.nu_in_which_cell`), outside the long-range quadrupole force of
   the whole halo (`Forces.outside_gravity_quadrupole`). `gravity_sun` moves
   the Sun along its orbit (`Forces.sun_gravity`). `gravity_none` is zero.
5. The vectors (position and velocity at z = 0 and at the end) are saved, and
   `Simulation.number_densities` calls `Physics.number_densities_all_sky`
   (per pixel) and `Physics.number_densities_mass_range` (total). These convert
   velocities to momenta for each neutrino mass and integrate the Fermi-Dirac
   distribution of the end momenta over today's momenta.

All simulation variants share this code; `SolverOptions` (force, relativistic,
solver settings) is the only difference.

**Units.** `cnb.units.Params` defines natural units (eV, c = ħ = k_B = 1).
Multiply by a unit to get a number in code units, divide to get a plain number:
`8.178*Params.kpc`, `x/Params.kpc`. Most functions take `args=Params()`. The
solver works in kpc and kpc/s ("physical reality"); forces are computed in code
units ("numerical reality") and converted back.


## Code map

| Module | Class | What it does | Key functions |
| --- | --- | --- | --- |
| [units](src/cnb/units.py) | `Params` | units, constants, Planck18 cosmology, T_CNB, neutrino and PTOLEMY parameters | |
| [physics](src/cnb/physics.py) | `Physics` | momenta, Fermi-Dirac, number densities | `velocities_to_momenta_all_sky`, `Fermi_Dirac`, `number_density`, `number_densities_all_sky`, `number_densities_mass_range`, `number_density_Delta`, `neutrino_masses`, `s_of_z` |
| [simulation](src/cnb/simulation.py) | `Simulation` | EOMs, solver, forces, all-sky runs | see above |
| [forces](src/cnb/forces.py) | `Forces` | force lookups used by the EOMs | `nu_in_which_cell`, `outside_gravity_quadrupole`, `sun_gravity` |
| [grid](src/cnb/grid.py) | `SimGrid` | cell grid and force precalculation | `grid_3D`, `cell_division`, `cell_gravity_short_range`, `cell_gravity_long_range_quadrupole`, `grid_data` |
| [sim_data](src/cnb/sim_data.py) | `SimData` | run parameters, halo sample, DM particles, loading outputs | `SimData(run_dir)`, `simulation_setup`, `halo_batch_indices`, `read_DM_halo_index`, `load_velocities` |
| [primordial](src/cnb/primordial.py) | `Primordial` | primordial fluctuation parameters (Cl momenta, redshift cuts) | `generate_DeltaTs_seeds` |
| [astro](src/cnb/astro.py) | `Astro` | Sun and Earth position and velocity over a year | `get_sun_earth_galactic_data`, `SunEarthGC_frame_coords_posvel`, `calculate_earth_position`, `calculate_earth_velocity` |
| [halos](src/cnb/halos.py) | `Halos` | NFW profile and escape momenta | `get_halo_params`, `get_p_esc` |
| [skymaps](src/cnb/skymaps.py) | `Skymaps` | rotations and healpix maps | `rotate_skymap`, `get_rotation_matrix_euler`, `DM_pos_to_skymap` |
| [utils](src/cnb/utils.py) | `Utils` | interpolation, random directions, sorting momenta | `jax_interpolate`, `sim_vels_to_sorted_z0z4(_vec)`, `v_mag_to_xyz_biased` |
| [cosmology](src/cnb/cosmology.py) | `Cosmo` | distance to the neutrino last-scattering surface | `get_LSS_distance` |
| [plotting](src/cnb/plotting.py) | `SimPlot` | paper figures; importing it sets the matplotlib style (needs LaTeX) | `overdensity_band`, `skymaps_CNBxDM`, `power_spectra_FDxPF`, `phase_space_2x2_FD_vs_PF`, `plot_annual_modulations` |
| [analysis](src/cnb/analysis.py) | - | helpers the notebooks share | `bulk_velocity_simple`, `bulk_velocity_new`, `Delta_vs_z_cut_show`, `momentum_distribution`, `age_to_redshift` |
| [config](src/cnb/config.py) | - | loading run configs | `load_config`, `ROOT` |
| [box](src/cnb/box.py), [tree_dataset](src/cnb/tree_dataset.py) | `SimInfo`, `TreeCatalogue` | reading the raw box and merger trees (by Camila Correa) | `build_tree` |


## Notebooks

The notebooks show how the analyses and figures were made. They are a
reference, not a polished pipeline: some cells are exploratory, and their
outputs were removed (the original notebooks with outputs are in the first
commit, `git show f390b3a:sim_notebooks/...`).

| Notebook | Run(s) it reads (`sim_name`) | Topic |
| --- | --- | --- |
| [benchmark_halo](notebooks/benchmark_halo.ipynb) | `benchmark` | NFW benchmark halo; monopole, FD vs. primordial fluctuations (PF) |
| [no_gravity](notebooks/no_gravity.ipynb) | `no_gravity` | reference without gravity; skymaps, PF skymaps |
| [dm_halos](notebooks/dm_halos.ipynb) | `Dopri5_1k` | main analysis of the box halos: perturbative breakdown, monopole/dipole, FD vs. PF, power spectra (paper 2) |
| [relativistic_eoms](notebooks/relativistic_eoms.ipynb) | `rel_EOMs` | same analysis with relativistic EOMs |
| [annual_modulation_background](notebooks/annual_modulation_background.ipynb) | `SunMod_1k` | Sun vs. DM cell gravity, Earth orbit, time steps (paper 3) |
| [annual_modulation_main](notebooks/annual_modulation_main.ipynb) | `SunMod_1k`, `SunMod_PSD_Nside_128` | momentum boosts, annual modulation curves (paper 3) |
| [annual_modulation_no_gravity](notebooks/annual_modulation_no_gravity.ipynb) | `SunNoG` | modulation without the Sun's gravity (paper 3) |
| [thesis_cnb](notebooks/thesis_cnb.ipynb) | `SunMod_2k` | thesis figures: cosmology, distances to the last-scattering surface |
| [thesis_neutrino_plots](notebooks/thesis_neutrino_plots.ipynb) | `Dopri5_1k` | thesis figures: travelled distances, momentum distributions |
| [archive/sketchbook](notebooks/archive/sketchbook.ipynb) | - | scratch work |

**Using a notebook with your data.** The first cell imports from `cnb` and
changes to the repo root, so all paths are relative to it:

1. Put the run folder (the `run_dir` of a config, with `sim_parameters.yaml`,
   `vectors_halo*.npy`, `total_densities.npy`, ...) at `runs/<sim_name>`.
2. Set `sim_name` in the notebook's first cells to that folder name.
3. Grids and DM positions are read from `data/halo_grids` and
   `data/benchmark_halo_files`, the CLASS power spectra from `data/primordial/Cls`.
4. Figures are saved to `figures/<sim_name>/` (create it first).

The notebooks expect halo 20 to be missing and leave out halos 23 and 24
(`exclude_nums` in `dm_halos`): halo 20 has no DM data at snapshot 0012, halos
23 and 24 give anomalous densities.


## Reproducing earlier runs

- `scripts/make_parameters.py` reproduces `runs/SunNoG` exactly. It uses the
  CNB temperature of the original setup (`T_CNB_SETUP`, no N_eff correction),
  which is 0.36% lower than `Params.T_CNB`.
- The `*_1k` runs used log-spaced momenta, the `SunMod_PSD*` runs linear
  spacing: `parameters.p_spacing` in the config.
- `Simulation.backtrack_1_neutrino` gives the same trajectories as the old
  scripts (checked for halo gravity, no gravity and the Sun; differences ≤ 2e-12).
- The relativistic EOMs are the default (as in the last version of
  `sim_script23`); `z_evolution` and `no_gravity` use the Newtonian ones, as before.


## Potential Issues (AI-discovered)

- **Old Snellius results of the annual modulation:** the Snellius copy of the
  old density script (`sim_plots41_..._make_data_snellius.py`, run by
  `sim_job43`) converted angles with `* Params.deg` instead of `jnp.rad2deg`
  in `transform_pixel_indices_no_rot`, so the pixels it assigned were wrong.
  [make_density_data.py](scripts/annual_modulation/make_density_data.py) uses
  the correct conversion. Results made with the old Snellius script should be checked.
- `make_density_data.py` crashes for `with_DM_gravity: true` with
  `interp_grav_psd: false`, and for `bound: true/false`: `p_z4` and `p_z0` are
  not defined on those paths.
- Solar runs (`throw: false`): in a local test the last saved step was `inf`
  for some neutrinos, as in the original code. An old comment suggests using
  the second-last step instead.
- `Simulation.earth_like_start` picks a different starting cell than the one
  used in paper 1, so all-sky maps of the same halo look different.
- Some notebook cells use names that are not defined anywhere in the notebook
  (left over from older versions): `SimUtil.CNB_to_Earth_frame_trafo`
  (`dm_halos`, `annual_modulation_no_gravity`), `pix_dens_PF_halos`
  (`benchmark_halo`, `relativistic_eoms`), bare unit names like `kpc` and
  `T_CNB` (`thesis_neutrino_plots`).
