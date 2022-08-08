from typing import List, Union, Tuple, Dict
import unyt
import glob
import os
import h5py
from .utilities import constants
from .halo_catalogue import HaloCatalogue

from swiftsimio import load

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
        self.snapshot = load(f"{self.directory}/{self.snapshot_name}")

        # Fetch the run name if not provided
        if name is not None:
            self.simulation_name = name
        else:
            self.simulation_name = self.snapshot.metadata.run_name

        # Conversion from internal units to kpc
        self.to_kpc_units = (
            self.snapshot.metadata.internal_code_units["Unit length in cgs (U_L)"][0]
            / constants.kpc
        )

        # Conversion from internal units to Msun
        self.to_Msun_units = (
            self.snapshot.metadata.internal_code_units["Unit mass in cgs (U_M)"][0]
            / constants.Msun
        )

        # Conversion from internal units to Myr
        self.to_Myr_units = (
            self.snapshot.metadata.internal_code_units["Unit time in cgs (U_t)"][0]
            / constants.Myr
        )

        # Conversion from internal units to yr
        self.to_yr_units = (
            self.snapshot.metadata.internal_code_units["Unit time in cgs (U_t)"][0]
            / constants.yr
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
        self.n_snapshots = int(newest_snap_name.replace(f"{self.directory}/{base_name}_", "").replace(".hdf5", "")) + 1

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

