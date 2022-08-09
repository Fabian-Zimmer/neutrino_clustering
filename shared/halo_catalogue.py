import numpy as np
import unyt
import velociraptor


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
