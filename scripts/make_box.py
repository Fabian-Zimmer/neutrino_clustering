"""Step 1: read the simulation box and make the run folder.

Reads cosmology, DM particle mass, softening length and snapshot redshifts of
the SWIFT simulation box, writes them to <run_dir>/box_parameters.yaml (plus
zeds_snaps.npy and nums_snaps.npy) and builds the merger tree of the halos
(<run_dir>/MergerTree.hdf5).

Needs the raw box (config: box_directory) and `uv sync --extra box`.

Usage: python scripts/make_box.py configs/<run>.yaml
"""

import os

import h5py
import numpy as np
import yaml

from cnb.box import SimInfo, make_tree_data
from cnb.config import config_from_cli
from cnb.units import Params


def make_box(box_dir, out_dir, z0_snap, z4_snap):
    """
    Reads cosmological and other global parameters of the specified simulation box. Output is stored in the run folder, where all simulation outputs (e.g. neutrino densities) will be stored.
    """

    os.makedirs(out_dir, exist_ok=True)

    # The snapshots to read, determined by inputs z0_snap and z4_snap.
    snap_nums = np.arange(z4_snap, z0_snap+1)
    nums_4cif = []
    zeds = np.zeros(len(snap_nums))

    # Loop over selected snapshots and save outputs.
    for i, num in enumerate(snap_nums):
        num_4cif = f'{num:04d}'
        nums_4cif.append(num_4cif)

        with h5py.File(f'{box_dir}/snapshot_{num_4cif}.hdf5') as snap:

            # Store redshifts.
            zeds[i] = snap['Cosmology'].attrs['Redshift'][0]

            # Save global parameters only once.
            if i == 0:

                # Cosmology.
                cosmo = snap['Cosmology']
                Omega_R = cosmo.attrs['Omega_r'][0]
                Omega_M = cosmo.attrs['Omega_m'][0]
                Omega_L = cosmo.attrs['Omega_lambda'][0]
                h_dimless = cosmo.attrs['h'][0]

                # Dark matter "particle" mass of box.
                DM_mass = np.unique(snap['PartType1/Masses'][:]*1e10*Params.Msun)[0]

                # Gravity smoothening length of box.
                smooth_len = snap['GravityScheme'].attrs[
                    'Maximal physical DM softening length (Plummer equivalent) [internal units]'
                ][0]*1e6*Params.pc

                # Create .yaml file for global parameters of box.
                box_parameters = {
                    "File Paths": {
                        "Box File Directory": box_dir,
                        "Output Directory": out_dir
                    },
                    "Cosmology": {
                        "Omega_R": float(Omega_R),
                        "Omega_M": float(Omega_M),
                        "Omega_L": float(Omega_L),
                        "h": float(h_dimless),
                    },
                    "Content": {
                        "DM Mass [Msun]": float(DM_mass/Params.Msun),
                        "Smoothening Length [pc]": float(smooth_len/Params.pc),
                        "z=0 snapshot": f'{z0_snap:04d}',
                        "z=4 snapshot": f'{z4_snap:04d}',
                        "initial redshift": float(zeds[-1]),
                        "final redshift": float(zeds[0]),
                    }
                }
                with open(f'{out_dir}/box_parameters.yaml', 'w') as file:
                    yaml.dump(box_parameters, file)

    # Save 4cifer codes and redshift for snapshots, as array files.
    np.save(f'{out_dir}/zeds_snaps.npy', zeds)
    np.save(f'{out_dir}/nums_snaps.npy', nums_4cif)


    ### ================= ###
    ### Make merger tree. ###
    ### ================= ###

    print('***Making merger tree***')
    z0_snap_4cif = f'{z0_snap:04d}'
    sim_info = SimInfo(
        directory=box_dir,
        snapshot=f'snapshot_{z0_snap_4cif}.hdf5',
        catalogue=f'subhalo_{z0_snap_4cif}.properties',
        name='MergerTree',
        output=out_dir,
        simtype='DMONLY'
    )
    make_tree_data(sim_info)


if __name__ == "__main__":
    cfg = config_from_cli(__doc__)
    make_box(
        box_dir=cfg["box_directory"],
        out_dir=cfg["run_dir"],
        z0_snap=int(cfg["box"]["initial_snap_z0"]),
        z4_snap=int(cfg["box"]["final_snap_z4"]))
