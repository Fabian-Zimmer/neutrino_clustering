"""
Description here
"""

import os
import h5py
import glob
from scatter_rate import plot_cosmic_scatter_rate
from HMF import make_HMF
from plotter import plot_relations, plot_halo_profiles, plot_individual_profiles

class SimInfo:
    def __init__(self,folder,snap,name):
        self.name = name
        self.folder = folder
        self.halo_properties = os.path.join(folder,"halo_%04i.properties"%snap)
        self.catalog_groups = os.path.join(folder,"halo_%04i.catalog_groups"%snap)
        self.catalog_particles = os.path.join(folder, "halo_%04i.catalog_particles" % snap)
        #self.latest_snapshot = os.path.join(self.folder,"eagle_%04i.hdf5"%snap)
        self.latest_snapshot = os.path.join(self.folder,"snapshot_%04i.hdf5"%snap)
        self.n_snapshots = int(snap)

        snapshot_file = h5py.File(self.latest_snapshot, "r")
        self.boxSize = snapshot_file["/Header"].attrs["BoxSize"][0] # Mpc
        self.a = snapshot_file["/Cosmology"].attrs["Scale-factor"]
        self.h = snapshot_file["/Cosmology"].attrs["h"]
        self.Om = snapshot_file["/Cosmology"].attrs["Omega_m"]
        self.Ol = snapshot_file["/Cosmology"].attrs["Omega_lambda"]
        self.rhocrit0 = 2.7754e11 * self.h**2 # Msun / Mpc^3
        self.sigma = snapshot_file["/SIDMScheme"].attrs["SIDM cross section [cgs units]"]

    def snapshot(self,snap):
        #return os.path.join(self.folder,"eagle_%04i.hdf5"%snap)
        return os.path.join(self.folder,"snapshot_%04i.hdf5"%snap)

if __name__ == '__main__':
    from utils import *

    output_path = args.output
    siminfo = SimInfo(args.directory, args.snapshot, args.name)

    #plot_cosmic_scatter_rate(siminfo,output_path)
    make_HMF(siminfo,output_path)
    plot_relations(siminfo,output_path)
    plot_halo_profiles(siminfo, output_path)
    plot_individual_profiles(siminfo, output_path)

