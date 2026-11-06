"""cnb: N-1-body simulations of the cosmic neutrino background (CNB).

Modules (each holds one class of static methods, as in the original code):

    units       Params      natural units and constants (pass `args=Params()`)
    physics     Physics     momenta, Fermi-Dirac phase space, number densities
    cosmology   Cosmo       distances to the neutrino last-scattering surface
    primordial  Primordial  primordial fluctuation parameters (arXiv:2407.14582)
    utils       Utils       interpolation, smoothing, random directions
    sim_data    SimData     run parameters, halo catalogue, loading outputs
    grid        SimGrid     cell grid and force precalculation per halo
    forces      Forces      forces on a neutrino during the simulation
    simulation  Simulation  equations of motion and the ODE solver
    skymaps     Skymaps     rotations and healpix skymaps
    halos       Halos       NFW helpers and escape momenta
    astro       Astro       Sun/Earth positions and velocities (arXiv:2507.01505)
    plotting    SimPlot     paper figures (importing it sets the plot style)
    config      load_config YAML run configs in configs/
    box         -           reading the raw SWIFT/VELOCIraptor box (extra: box)
"""

import warnings

import jax

# Simulations need float64. This must run before any module creates jax arrays.
jax.config.update("jax_enable_x64", True)

# Ignore UserWarnings issued by healpy and matplotlib.
warnings.filterwarnings("ignore", category=UserWarning, module="healpy")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
