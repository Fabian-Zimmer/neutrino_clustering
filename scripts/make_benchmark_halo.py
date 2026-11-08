"""Sample DM particles of the NFW-like benchmark halo for every snapshot.

The halo has the median virial mass of the halo sample of another run
(benchmark.halo_sample_run) and grows with redshift following COMMAH. Saves
data_dir/benchmark_halo_snap_<snap>.npy (positions in kpc), which
scripts/precalculate_grids.py (sim_type: benchmark) turns into a force grid.

Needs run_dir/box_parameters.yaml (scripts/make_box.py) and
`uv sync --extra benchmark` (commah builds on Linux; not on macOS).

Usage: python scripts/make_benchmark_halo.py configs/benchmark.yaml
"""

import glob
import math
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import commah
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pynverse import inversefunc

from cnb.config import config_from_cli, resolve
from cnb.units import Params


def rho_crit(z, H0, Omega_M, Omega_L, args):
    """Critical density of the universe as a function of redshift, assuming
    matter domination, only Omega_m and Omega_Lambda in Friedmann equation.
    """
    H_squared = H0**2 * (Omega_M*(1.+z)**3 + Omega_L)
    return np.float64(3.*H_squared / (8.*args.Pi*args.G))


def halo_sample_z(z, snap, Mvir_z0, DM_mass, out_dir, cosmology, args, origin_offset=0.):

    # Get the DM halo mass (and the number of DM particles for sample).
    commah_output = commah.run('Planck13', zi=0, Mi=Mvir_z0, z=z)
    Mz = commah_output['Mz'][0,0]*args.Msun
    num_DM = math.floor(Mz / DM_mass)

    # Get the concentration of the halo.
    c_200 = commah_output['c'][0,0]

    # Calculate R_200 and R_s ("virial" radius and scale radius).
    R_200 = np.power(Mz / (200*rho_crit(z, *cosmology, args)*4/3*args.Pi), 1./3.)
    R_s = R_200 / c_200

    # Construct projection function.
    def Proj(r, r_s, norm):
        x = r/r_s
        return (np.log(1+x) - (x/(1+x)))/norm

    # Construct inverse function. Needs to be without numerical units.
    f_200 = np.log(1+c_200) - (c_200/(1+c_200))
    invf = inversefunc(Proj, args=(R_s/args.kpc, f_200))

    # Sample uniformly between [0,1] and project to corresponding radius.
    sample = np.sort(np.random.uniform(size=num_DM))
    r_sample = invf(sample)

    # Sample for angles and convert to cartesian DM_coords.
    phis = np.random.uniform(0, 2*args.Pi, num_DM)  # uniform [0,2pi)
    cos_thetas = 2.*np.random.uniform(0, 1, num_DM) - 1  # uniform [-1,1)

    # Convert to cartesian coordinates.
    x = r_sample*np.cos(phis)*np.sqrt(1. - cos_thetas**2)
    y = r_sample*np.sin(phis)*np.sqrt(1. - cos_thetas**2)
    z = r_sample*cos_thetas
    coords = np.column_stack((x,y,z)) + origin_offset

    np.save(f'{out_dir}/benchmark_halo_snap_{snap}.npy', coords)


def make_benchmark_halo(cfg):

    run_dir, out_dir = cfg["run_dir"], cfg["data_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Box parameters and arrays.
    with open(f'{run_dir}/box_parameters.yaml', 'r') as file:
        box_setup = yaml.safe_load(file)
    DM_mass = box_setup['Content']['DM Mass [Msun]']*Params.Msun
    nums_snaps = np.load(f'{run_dir}/nums_snaps.npy')
    zeds_snaps = np.load(f'{run_dir}/zeds_snaps.npy')

    H0 = box_setup['Cosmology']['h']*100*Params.km/Params.s/Params.Mpc
    cosmology = (H0, box_setup['Cosmology']['Omega_M'], box_setup['Cosmology']['Omega_L'])

    # Get median Mvir of box halo sample from numerical simulation.
    sample_run = resolve(cfg["benchmark"]["halo_sample_run"])
    halo_params = np.load(glob.glob(f'{sample_run}/halo*params.npy')[0])
    Rvir_med = halo_params[:,0]
    Mvir_med = (10**np.median(halo_params[:,1]))
    cNFW_med = halo_params[:,2]
    print(f'Median Mvir of box halo sample: {Mvir_med:.2e} Msun')
    print(f'Min/Max value of Rvir of box halo sample: {np.min(Rvir_med):.2f}/{np.max(Rvir_med):.2f} kpc')
    print(f'Min/Max value of concentration of box halo sample: {np.min(cNFW_med):.2f}/{np.max(cNFW_med):.2f}')

    missing = [
        (z, num) for z, num in zip(zeds_snaps, nums_snaps)
        if not os.path.exists(f'{out_dir}/benchmark_halo_snap_{num}.npy')]
    if missing:
        zs, nums = zip(*missing)
        with ProcessPoolExecutor(25) as ex:
            list(ex.map(
                halo_sample_z, zs, nums,
                repeat(Mvir_med), repeat(DM_mass), repeat(out_dir),
                repeat(cosmology), repeat(Params())))

    # Check: DM mass of the benchmark halo for each snapshot.
    benchmark_DM = np.array([
        len(np.load(f'{out_dir}/benchmark_halo_snap_{num}.npy')) for num in nums_snaps])
    print(np.log10(Params.Mvir_MW/Params.Msun), np.log10(DM_mass*benchmark_DM[-1]/Params.Msun))

    plt.plot(nums_snaps.astype(int), benchmark_DM)
    plt.savefig(f"{run_dir}/mass_check.pdf")


if __name__ == "__main__":
    make_benchmark_halo(config_from_cli(__doc__))
