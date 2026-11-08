"""Reading the raw SWIFT/VELOCIraptor simulation box and building merger trees.

Used by scripts/make_box.py. Needs the optional `box` dependencies
(uv sync --extra box). Adapted from code by Camila Correa.
"""

import glob
import os
from typing import List, Union

import h5py
import numpy as np
import swiftsimio
import unyt
import velociraptor

from cnb.tree_dataset import TreeCatalogue

Camila_kpc = 3.08567758e21
Camila_Msun = 1.9891e33
Camila_yr = 3.1556926e7
Camila_Myr = Camila_yr * 1e6


class HaloCatalogue:
    """
    General class containing halo properties
    """

    def __init__(
        self, path_to_catalogue: str, dm_particle_mass: float, simulation_type: str,
    ):
        """
        Parameters
        ----------
        path_to_catalogue: str
        Path to the catalogue with halo properties
        dm_particle_mass: unyt.array.unyt_quantity
        Minimum dark matter particle mass in units of Msun. Haloes that contain less than
        1000 dark mattter particles are disregarded
        """

        self.path_to_catalogue = path_to_catalogue

        # Load catalogue using velociraptor python library
        catalogue = velociraptor.load(self.path_to_catalogue)

        # Selecting haloes that contain at less 1000 DM particles
        mask = np.where(
            catalogue.masses.mass_200crit.to("Msun").value >= unyt.unyt_quantity(1e3 * dm_particle_mass, "Msun")
        )[0]

        # Compute the number of haloes following the selection mask
        self.number_of_haloes = len(mask)

        # Structure type
        self.structure_type = catalogue.structure_type.structuretype[mask]

        # Log10 halo mass in units of Msun
        self.log10_halo_mass = np.log10(
            catalogue.masses.mass_200crit.to("Msun").value[mask]
        )

        self.concentration = catalogue.concentration.cnfw.value[mask]
        self.virial_radius = catalogue.radii.r_200crit.to("Mpc").value[mask]

        self.scale_radius = self.virial_radius / self.concentration

        # Ids of haloes satisfying the selection criterion
        self.halo_index = mask.copy()

        self.xminpot = catalogue.positions.xcminpot.to("Mpc").value[mask]
        self.yminpot = catalogue.positions.ycminpot.to("Mpc").value[mask]
        self.zminpot = catalogue.positions.zcminpot.to("Mpc").value[mask]

        self.vxminpot = catalogue.velocities.vxcminpot.to("km/s").value[mask]
        self.vyminpot = catalogue.velocities.vycminpot.to("km/s").value[mask]
        self.vzminpot = catalogue.velocities.vzcminpot.to("km/s").value[mask]

        self.vmax = catalogue.velocities.vmax.to("km/s").value[mask]

        self.xcom = catalogue.positions.xc.to("Mpc").value[mask]
        self.ycom = catalogue.positions.yc.to("Mpc").value[mask]
        self.zcom = catalogue.positions.zc.to("Mpc").value[mask]

        if simulation_type == 'Hydro':

            # Log10 stellar mass in units of Msun
            self.log10_stellar_mass = np.log10(
                catalogue.apertures.mass_star_30_kpc.to("Msun").value[mask]
            )

            # Log10 gas mass in units of Msun
            self.log10_gas_mass = np.log10(
                catalogue.apertures.mass_gas_30_kpc.to("Msun").value[mask]
            )

            self.galaxy_size = catalogue.apertures.rhalfmass_star_30_kpc.to("kpc").value[mask]

            # Half mass radius in units of kpc (stars)
            self.half_mass_radius_star = catalogue.radii.r_halfmass_star.to("kpc").value[mask]

            # Half mass radius in units of kpc (gas)
            self.half_mass_radius_gas = catalogue.radii.r_halfmass_gas.to("kpc").value[mask]

            # Star formation rate in units of Msun/yr
            self.sfr = catalogue.apertures.sfr_gas_30_kpc.to("Msun/yr").value[mask]

            # Metallicity of star-forming gas
            self.metallicity_gas_sfr = catalogue.apertures.zmet_gas_sf_30_kpc.to("dimensionless").value[mask]

            # Metallicity of all gas
            self.metallicity_gas = catalogue.apertures.zmet_gas_30_kpc.to("dimensionless").value[mask]

            self.metallicity_stars = catalogue.apertures.zmet_star_30_kpc.to("dimensionless").value[mask]


class SimInfo:

    def __init__(
        self,
        directory: str,
        snapshot: str,
        catalogue: str,
        name: Union[str, None],
        output: str,
        simtype: str,
    ):
        """
        Parameters
        ----------
        directory: str
        Run directory
        snapshot: str
        Name of the snapshot file
        catalogue: str
        Name of the catalogue file
        name:
        Name of the run
        galaxy_min_stellar_mass: unyt.array.unyt_quantity
        """

        self.directory = directory
        self.snapshot_name = snapshot
        self.catalogue_name = catalogue
        self.output_path = output
        self.simulation_type = simtype

        # Find the group and particle catalogue files
        self.__find_groups_and_particles_catalogues()

        # Load snapshot via swiftsimio
        self.snapshot = swiftsimio.load(f"{self.directory}/{self.snapshot_name}")

        # Fetch the run name if not provided
        if name is not None:
            self.simulation_name = name
        else:
            self.simulation_name = self.snapshot.metadata.run_name

        # Conversion from internal units to kpc
        self.to_kpc_units = (
            self.snapshot.metadata.internal_code_units["Unit length in cgs (U_L)"][0]
            / Camila_kpc
        )

        # Conversion from internal units to Msun
        self.to_Msun_units = (
            self.snapshot.metadata.internal_code_units["Unit mass in cgs (U_M)"][0]
            / Camila_Msun
        )

        # Conversion from internal units to Myr
        self.to_Myr_units = (
            self.snapshot.metadata.internal_code_units["Unit time in cgs (U_t)"][0]
            / Camila_Myr
        )

        # Conversion from internal units to yr
        self.to_yr_units = (
            self.snapshot.metadata.internal_code_units["Unit time in cgs (U_t)"][0]
            / Camila_yr
        )

        # Box size of the simulation in kpc
        self.boxSize = self.snapshot.metadata.boxsize.to("kpc").value[0]

        # Cosmic scale factor
        self.a = self.snapshot.metadata.scale_factor

        self.hubble_time_Gyr = self.snapshot.metadata.time.to("Gyr").value

        self.Omega_m = self.snapshot.metadata.cosmology.Om0

        self.Omega_l = 1 - self.Omega_m

        self.h = self.snapshot.metadata.cosmology.H0.value / 100.

        self.z = self.snapshot.metadata.redshift

        self.rhocrit0 = 2.7754e11 * self.h ** 2  # Msun / Mpc^3

        # Maximum softening
        self.softening = (
            self.snapshot.metadata.gravity_scheme[
                "Maximal physical DM softening length (Plummer equivalent) [internal units]"
            ][0] * self.to_kpc_units
        )

        # Maximum softening for baryons
        self.baryon_max_soft = (
            self.snapshot.metadata.gravity_scheme[
                "Maximal physical baryon softening length  [internal units]"
            ][0]
            * self.to_kpc_units
        )

        self.dm_particle_mass = 1e6 # Need to fix, table mass is empty

        self.num_dm_particles = self.snapshot.metadata.n_dark_matter

        if self.simulation_type == 'Hydro':
            self.num_star_particles = self.snapshot.metadata.n_stars
            self.num_gas_particles = self.snapshot.metadata.n_gas

        snapshot_file = f"{self.directory}/{self.snapshot_name}"

        self.cross_section = 0.
        SIDM = "/SIDMScheme"
        with h5py.File(snapshot_file, "r") as sim:
            if SIDM in sim.keys():
                check = sim["/SIDMScheme"].attrs.get("SIDM cross section [cgs units]")
                if check != None: self.cross_section = check

        # Object containing halo properties (from halo catalogue)
        self.halo_data = HaloCatalogue(
           path_to_catalogue=f"{self.directory}/{self.catalogue_name}",
           dm_particle_mass=self.dm_particle_mass,
           simulation_type=self.simulation_type
        )
        catalogue_base_name = "".join([s for s in self.catalogue_name if not s.isdigit() and s != "_"])
        catalogue_base_name = os.path.splitext(catalogue_base_name)[0]
        self.catalogue_base_name = catalogue_base_name

        base_name = "".join([s for s in self.snapshot_name if not s.isdigit() and s != "_"])
        base_name = os.path.splitext(base_name)[0]

        newest_snap_name = max(glob.glob(f"{self.directory}/{base_name}_*.hdf5"), key=os.path.getctime)

        #? why necessary
        # self.n_snapshots = int(newest_snap_name.replace(f"{self.directory}/{base_name}_", "").replace(".hdf5", "")) + 1

        num_snap = "".join([s for s in self.catalogue_name if s.isdigit()])
        self.initial_snap = int(num_snap)
        self.snapshot_base_name = base_name

        print(f"Data from run '{self.simulation_name}' has been loaded! \n")

        return

    def __find_groups_and_particles_catalogues(self) -> None:
        """
        Finds paths to the fields with particles catalogue and groups catalogue
        """

        catalogue_num = "".join([s for s in self.catalogue_name if s.isdigit()])
        catalogue_groups_paths: List[str] = glob.glob(
            f"{self.directory}/*{catalogue_num}.catalog_groups*"
        )
        catalogue_particles_paths: List[str] = glob.glob(
            f"{self.directory}/*{catalogue_num}.catalog_particles*"
        )

        # We expect one file for particle groups
        if len(catalogue_groups_paths) == 1:
           self.catalogue_groups = catalogue_groups_paths[0].split("/")[-1]
        else:
           raise IOError("Couldn't find catalogue_groups file")

        # We expect two files: one for bound and the other for unbound particles
        if len(catalogue_particles_paths) == 2:
           for path in catalogue_particles_paths:
               if path.find("unbound") == -1:
                   self.catalogue_particles = path.split("/")[-1]
        else:
           raise IOError("Couldn't find catalogue_particles file")

        return


class snapshot_info:

    def __init__(self, sim_info, snap):

        self.directory = sim_info.directory
        self.catalogue_name = sim_info.catalogue_base_name + "_%04i.properties" % snap
        self.snapshot_name = sim_info.snapshot_base_name + "_%04i.hdf5" % snap
        self.output_path = sim_info.output_path

        # Find the group and particle catalogue files
        self.__find_groups_and_particles_catalogues()

        # Load snapshot via swiftsimio
        self.snapshot = swiftsimio.load(f"{self.directory}/{self.snapshot_name}")

        # Load conversion units
        self.__load_conversion()

        # Load cosmological params
        self.__load_cosmology()
        return

    def __load_conversion(self) -> None:

        # Conversion from internal units to kpc
        kpc = 3.08567758e21
        Msun = 1.9891e33
        yr = 3.1556926e7
        Myr = yr * 1e6

        self.to_kpc_units = (
                self.snapshot.metadata.internal_code_units["Unit length in cgs (U_L)"][0]
                / kpc
        )

        # Conversion from internal units to Msun
        self.to_Msun_units = (
                self.snapshot.metadata.internal_code_units["Unit mass in cgs (U_M)"][0]
                / Msun
        )

        # Conversion from internal units to Myr
        self.to_Myr_units = (
                self.snapshot.metadata.internal_code_units["Unit time in cgs (U_t)"][0]
                / Myr
        )

        # Conversion from internal units to yr
        self.to_yr_units = (
                self.snapshot.metadata.internal_code_units["Unit time in cgs (U_t)"][0]
                / yr
        )

    def __load_cosmology(self) -> None:

        # Box size of the simulation in kpc
        self.boxSize = self.snapshot.metadata.boxsize.to("kpc").value[0]

        # Cosmic scale factor
        self.a = self.snapshot.metadata.scale_factor

        self.hubble_time_Gyr = self.snapshot.metadata.time.to("Gyr").value

        self.Omega_m = self.snapshot.metadata.cosmology.Om0

        self.Omega_l = 1 - self.Omega_m

        self.h = self.snapshot.metadata.cosmology.H0.value / 100.

        self.z = self.snapshot.metadata.redshift

        self.rhocrit0 = 2.7754e11 * self.h ** 2  # Msun / Mpc^3

        # Maximum softening
        self.softening = (
                self.snapshot.metadata.gravity_scheme[
                    "Maximal physical DM softening length (Plummer equivalent) [internal units]"
                ][0] * self.to_kpc_units
        )

        # Maximum softening for baryons
        self.baryon_max_soft = (
                self.snapshot.metadata.gravity_scheme[
                    "Maximal physical baryon softening length  [internal units]"
                ][0]
                * self.to_kpc_units
        )

        self.dm_particle_mass = 1e6 # Need to fix, table mass is empty

        self.num_dm_particles = self.snapshot.metadata.n_dark_matter


    def __find_groups_and_particles_catalogues(self) -> None:
        """
        Finds paths to the fields with particles catalogue and groups catalogue
        """

        catalogue_num = "".join([s for s in self.catalogue_name if s.isdigit()])
        catalogue_groups_paths: List[str] = glob.glob(
            f"{self.directory}/*{catalogue_num}.catalog_groups*"
        )
        catalogue_particles_paths: List[str] = glob.glob(
            f"{self.directory}/*{catalogue_num}.catalog_particles*"
        )

        # We expect one file for particle groups
        if len(catalogue_groups_paths) == 1:
            self.catalogue_groups = catalogue_groups_paths[0].split("/")[-1]
        else:
            raise IOError("Couldn't find catalogue_groups file")

        # We expect two files: one for bound and the other for unbound particles
        if len(catalogue_particles_paths) == 2:
            for path in catalogue_particles_paths:
                if path.find("unbound") == -1:
                    self.catalogue_particles = path.split("/")[-1]
        else:
            raise IOError("Couldn't find catalogue_particles file")

        return

    def load_halo_catalogue(self, mask):

        # Load catalogue using velociraptor python library
        catalogue = velociraptor.load(f"{self.directory}/{self.catalogue_name}")

        # Compute the number of haloes following the selection mask
        # self.number_of_haloes = mask.sum()
        self.number_of_haloes = len(mask)

        mask = mask.astype('int')

        # Structure type
        self.structure_type = catalogue.structure_type.structuretype[mask]

        # Log10 halo mass in units of Msun
        self.log10_halo_mass = np.log10(
            catalogue.masses.mass_200crit.to("Msun").value[mask]
        )

        self.concentration = catalogue.concentration.cnfw.value[mask]
        self.virial_radius = catalogue.radii.r_200crit.to("Mpc").value[mask]

        self.scale_radius = self.virial_radius / self.concentration

        self.halo_index = mask.copy()

        self.xminpot = catalogue.positions.xcminpot.to("Mpc").value[mask]
        self.yminpot = catalogue.positions.ycminpot.to("Mpc").value[mask]
        self.zminpot = catalogue.positions.zcminpot.to("Mpc").value[mask]

        self.vxminpot = catalogue.velocities.vxcminpot.to("km/s").value[mask]
        self.vyminpot = catalogue.velocities.vycminpot.to("km/s").value[mask]
        self.vzminpot = catalogue.velocities.vzcminpot.to("km/s").value[mask]

        self.vmax = catalogue.velocities.vmax.to("km/s").value[mask]

        self.xcom = catalogue.positions.xc.to("Mpc").value[mask]
        self.ycom = catalogue.positions.yc.to("Mpc").value[mask]
        self.zcom = catalogue.positions.zc.to("Mpc").value[mask]


def look_for_progenitor_index(progenitor_offset, num_progenitors, progenitors_ID, ID, m200c):

    num_haloes = len(num_progenitors)
    pro_indx = np.ones(num_haloes) * (-1)

    for i in range(num_haloes):

        if num_progenitors[i] == 0:
            continue

        num = num_progenitors[i]
        progenitor_list = np.arange(num) + progenitor_offset[i]
        proID = progenitors_ID[progenitor_list]

        _, indx_ID, _ = np.intersect1d(ID, proID, assume_unique=True, return_indices=True, )

        if len(indx_ID) == 0:
            continue

        largest_mass_progenitor = np.argmax(m200c[indx_ID])
        pro_indx[i] = indx_ID[largest_mass_progenitor]

    return pro_indx


def build_tree(sim_info, halo_index, output_file):

    # Builds the merger tree of your halo selection and stores it in a file.

    initial_snap = sim_info.initial_snap
    final_snap = 11  #! final_snap+1 is "final" snapshot we simulate back to

    # How many haloes are we tracking?
    num_haloes = len(halo_index)

    # Let's collect some data from the halo that we are following,
    progenitor_index = np.zeros((num_haloes,initial_snap - final_snap))
    progenitor_index[:, 0] = halo_index
    z = np.zeros(initial_snap - final_snap)

    # Related to mass assembly history
    mass = np.zeros((num_haloes,initial_snap - final_snap))
    type = np.zeros((num_haloes,initial_snap - final_snap))

    catalogue_file = f"{sim_info.directory}/{sim_info.catalogue_base_name}" + "_%04i.properties" % initial_snap
    catalogue = velociraptor.load(catalogue_file)
    m200c = catalogue.masses.mass_200crit.to("Msun").value
    z[0] = catalogue.z
    mass[:, 0] = m200c[halo_index]
    type[:, 0] = catalogue.structure_type.structuretype[halo_index]

    tree_file = f"{sim_info.directory}/merger_tree/MergerTree.snapshot_0%i.VELOCIraptor.tree" % initial_snap
    tree_data = TreeCatalogue(tree_file)

    halo_offset = tree_data.catalogue.ProgenOffsets.value[halo_index]
    num_progenitors = tree_data.catalogue.NumProgen.value[halo_index]
    progenitors_ID = tree_data.catalogue.Progenitors.value

    for snap in range(initial_snap-1,final_snap,-1):

        i = initial_snap - snap
        snapshot_data = snapshot_info(sim_info, snap)
        path_to_catalogue_file = f"{snapshot_data.directory}/{snapshot_data.catalogue_name}"
        catalogue = velociraptor.load(path_to_catalogue_file)
        m200c = catalogue.masses.mass_200crit.to("Msun").value
        z[i] = catalogue.z

        tree_file = f"{sim_info.directory}/merger_tree/MergerTree.snapshot_0%i.VELOCIraptor.tree" % snap
        tree_data = TreeCatalogue(tree_file)

        pro_indx = look_for_progenitor_index(
            halo_offset,
            num_progenitors,
            progenitors_ID,
            tree_data.catalogue.ID.value,
            m200c,
        )

        halo_offset = tree_data.catalogue.ProgenOffsets.value[pro_indx.astype('int')]
        num_progenitors = tree_data.catalogue.NumProgen.value[pro_indx.astype('int')]
        progenitors_ID = tree_data.catalogue.Progenitors.value

        progenitor_index[:, i] = pro_indx
        mass[:, i] = m200c[pro_indx.astype('int')]
        type[:, i] = catalogue.structure_type.structuretype[pro_indx.astype('int')]

    # Write data to file
    data_file = h5py.File(output_file, 'w')
    f = data_file.create_group('Assembly_history')
    f.create_dataset('ID', data=halo_index)
    f.create_dataset('Mass', data=mass)
    f.create_dataset('Redshift', data=z)
    f.create_dataset('Structure_Type', data=type)
    f.create_dataset('Progenitor_index', data=progenitor_index)
    data_file.close()

    return progenitor_index


def make_tree_data(sim_info):

    # Some selection example:
    select_sub_sample = np.where(
        (sim_info.halo_data.log10_halo_mass >= 10) &
        (sim_info.halo_data.log10_halo_mass < 12.5))[0]

    select_type = np.where(sim_info.halo_data.structure_type[select_sub_sample] == 10)[0]

    sample = select_sub_sample[select_type]
    halo_index = sim_info.halo_data.halo_index[sample]

    # Output data
    output_file = f"{sim_info.output_path}/{sim_info.simulation_name}.hdf5"
    progenitor_index = build_tree(sim_info, halo_index, output_file)
