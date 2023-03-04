from swiftsimio import load
from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
from matplotlib.pyplot import imsave
from matplotlib.colors import LogNorm

from shared.preface import NUMS_SNAPSHOTS

sim = 'L012N376'
snap = '0036'

# Generate projected DM maps for all snapshots of sim.
for snap in NUMS_SNAPSHOTS[::-1][::5]:
    data = load(f'/home/fabian/ownCloud/snellius/{sim}/snapshot_{snap}.hdf5')

    # Generate smoothing lengths for the dark matter
    data.dark_matter.smoothing_length = generate_smoothing_lengths(
        data.dark_matter.coordinates,
        data.metadata.boxsize,
        kernel_gamma=1.8,
        neighbours=57,
        speedup_fac=2,
        dimension=3,
    )

    # Project the dark matter mass
    dm_mass = project_pixel_grid(
        # Notice here that we pass in the dark matter dataset not the whole
        # data object, to specify what particle type we wish to visualise
        data=data.dark_matter,
        boxsize=data.metadata.boxsize,
        resolution=1024,
        project="masses",
        parallel=True,
        region=None
    )

    # Everyone knows that dark matter is purple
    imsave(f'figures/{sim}_projectedDM_{snap}.png', LogNorm()(dm_mass), cmap="inferno")