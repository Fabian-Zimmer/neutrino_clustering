from shared.preface import *
from shared.shared_functions import *
from shared.plot_class import analyze_simulation_outputs
import matplotlib.colors as mcolors
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mycolorpy import colorlist as mcp

class analyze_simulation_outputs_test(object):

    def __init__(self, sim_dir, objects, sim_type):

        # Required:
        self.sim_dir = sim_dir
        self.objects = objects
        self.sim_type = sim_type
        
        self.fig_dir = f'figures/{sim_dir}'
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)

        # Neccessary arrays.
        self.mrange = np.load(f'{self.sim_dir}/neutrino_massrange_eV.npy')*eV
        self.mpicks = np.array([0.01, 0.05, 0.1, 0.3])

        # Find all integers for halos in files (some are broken).
        paths = glob.glob(f'{self.sim_dir}/init_xyz_halo*.npy')
        nums = np.array(
            [list(map(int, re.findall(r'\d+', path)))[-1] for path in paths]
        ).flatten()
        self.halo_glob_order = nums.argsort()
        self.final_halos = nums[self.halo_glob_order]

        # note: besides the broken halo 20, halos 24 & 25 are anomalies.
        for ex in (24,25):
            self.final_halos = np.delete(self.final_halos, self.final_halos==ex)

        self.halo_num = len(self.final_halos)

        # Halo indices and parameters for the final halos.
        self.halo_indices = np.load(
            glob.glob(f'{self.sim_dir}/halo_batch*indices.npy')[0]
        )[self.final_halos-1]
        self.halo_params = np.load(
            glob.glob(f'{self.sim_dir}/halo_batch*params.npy')[0]
        )[self.final_halos-1]


        if self.sim_type == 'single_halos':

            if 'NFW_halo' in self.objects:

                batch_paths = glob.glob(
                    f'{self.sim_dir}/neutrino_vectors_numerical_benchmark_halo_batch*.npy'
                )
                
                self.vectors_benchmark = []
                for batch_path in batch_paths:
                    self.vectors_benchmark.append(np.load(batch_path))
                self.vectors_benchmark = np.squeeze(
                    np.array(self.vectors_benchmark)
                )
                self.vectors_benchmark = np.array(self.vectors_benchmark)

                self.etas_benchmark = np.load(
                    f'{self.sim_dir}/number_densities_numerical_benchmark_halo.npy'
                )/N0


            if 'box_halos' in self.objects:
                
                self.etas_numerical = []
                self.vectors_numerical = []

                for halo in self.final_halos: 
                    

                    # Find all batch paths belonging to current halo.
                    batch_paths = glob.glob(
                        f'{self.sim_dir}/neutrino_vectors_numerical_halo{halo}_batch*.npy'
                    )

                    # Concatenate all vector batches into one array.
                    vectors_halo = []
                    for batch_path in batch_paths:
                        vectors_halo.append(np.load(batch_path))
                    vectors_halo = np.squeeze(np.array(vectors_halo))

                    # Append vectors.
                    self.vectors_numerical.append(vectors_halo)

                    # Append overdensities.
                    self.etas_numerical.append(
                        np.load(
                            f'{self.sim_dir}/number_densities_numerical_halo{halo}.npy'
                        )/N0
                    )

                self.etas_numerical = np.array(self.etas_numerical)
                self.vectors_numerical = np.array(self.vectors_numerical)


            if 'analytical_halo' in self.objects:

                batch_paths = glob.glob(
                    f'{self.sim_dir}/neutrino_vectors_analytical_batch*.npy'
                )
                
                self.vectors_analytical = []
                for batch_path in batch_paths:
                    self.vectors_analytical.append(np.load(batch_path))
                self.vectors_analytical = np.squeeze(
                    np.array(self.vectors_analytical)
                )
                self.vectors_analytical = np.array(self.vectors_analytical)

                self.etas_analytical = np.load(
                    f'{self.sim_dir}/number_densities_analytical_single_halos.npy'
                )/N0


        elif self.sim_type == 'all_sky':

            ### ============================ ###
            ### Parameters and dictionaries. ###
            ### ============================ ###

            with open(f'{self.sim_dir}/sim_parameters.yaml', 'r') as file:
                sim_setup = yaml.safe_load(file)
            self.Nside = sim_setup['Nside']
            self.Npix = sim_setup['Npix']
            self.pix_sr = sim_setup['pix_sr']
            self.N0_pix = N0/self.Npix
            self.healpy_dict = dict(
                coord=['G'],
                graticule=True,
                graticule_labels=True,
                xlabel="longitude",
                ylabel="latitude",
                cb_orientation="horizontal",
                projection_type="mollweide",
                flip='astro',
                latitude_grid_spacing=45,
                longitude_grid_spacing=45,
                phi_convention='counterclockwise',
            )


            ### ===================== ###
            ### Load simulation data. ###
            ### ===================== ###

            if 'box_halos' in self.objects:

                self.number_densities_numerical_all_sky = []

                for halo in self.final_halos:
                    
                    # Append overdensities.
                    self.number_densities_numerical_all_sky.append(
                        np.load(
                            f'{self.sim_dir}/number_densities_numerical_halo{halo}_all_sky.npy'
                        )
                    )

                self.number_densities_numerical_all_sky = np.array(
                    self.number_densities_numerical_all_sky
                )


            if 'NFW_halo' in self.objects:

                self.number_densities_numerical_all_sky = np.load(
                    f'{self.sim_dir}/number_densities_numerical_benchmark_halo_all_sky.npy'
                )[np.newaxis,...]


            if 'analytical_halo' in self.objects:

                self.number_densities_analytical_all_sky = np.load(
                    f'{self.sim_dir}/number_densities_analytical_all_sky.npy'
                )


    def rotate_DM(self, DM_pos, obs_pos):

        # Calculate the Euler angles, to match frame of Earth. This will be the 
        # new frame in which DM particles get projected to an all sky map.
        zAngle = np.arctan2(obs_pos[1], -obs_pos[0])
        yAngle = np.arctan2(-obs_pos[2], np.linalg.norm(obs_pos[:2]))
        # print('*** Angles: ***')
        # print(np.rad2deg(zAngle), np.rad2deg(yAngle))

        cz = np.cos(zAngle)
        sz = np.sin(zAngle)
        cy = np.cos(yAngle)
        sy = np.sin(yAngle)

        R_z = np.array([
            [cz, -sz, 0],
            [sz, cz,  0],
            [0,  0,   1]
        ])
        R_y = np.array([
            [cy,  0, sy],
            [0,   1,  0],
            [-sy, 0, cy]
        ])

        rot_mat = np.matmul(R_y, R_z)

        # Observer position in nex frame (aligned with x-axis).
        obs_orig_in_rot_frame = np.matmul(rot_mat, obs_pos.T).T

        # Dark matter positions in observer frame.
        DM_orig_in_rot_frame = np.matmul(rot_mat, DM_pos.T).T

        return DM_orig_in_rot_frame, obs_orig_in_rot_frame


    def rot_int_smooth_healpix_map(self, healpix_map, cell0):
        
        # Angles to compensate for location of starting cell (cell0).
        zAngle = np.rad2deg(np.arctan2(cell0[1], -cell0[0]))
        yAngle = np.rad2deg(np.arctan2(-cell0[2], np.linalg.norm(cell0[:2])))

        # Create rotator object, to place c.o.p. of halo in the map center.
        rot = hp.rotator.Rotator(
            rot=[-zAngle, yAngle, 0], deg=True, eulertype='ZYX', inv=False
        )

        # Create a grid of (theta, phi) coordinates for the map.
        Nside = hp.npix2nside(len(healpix_map))
        theta, phi = hp.pix2ang(Nside, np.arange(len(healpix_map)))

        # Apply rotation to the grid.
        theta_rot, phi_rot = rot(theta, phi)

        # Find the pixel indices of the rotated grid.
        pix_rot = hp.ang2pix(Nside, theta_rot, phi_rot)

        # Create a new map with rotated pixel values.
        rotated_map = np.zeros_like(healpix_map)
        rotated_map[pix_rot] = healpix_map

        # Interpolate zero pixels with values from 4 neighbours.
        zero_pixels = np.where(rotated_map == 0)[0]
        for pix in zero_pixels:
            theta_zero, phi_zero = hp.pix2ang(Nside, pix)
            interp_val = hp.get_interp_val(
                rotated_map, theta_zero, phi_zero
            )
            rotated_map[pix] = interp_val

        # Smooth the rotated map using a Gaussian kernel.
        reso_1pix_deg = hp.nside2resol(Nside, arcmin=True) / 60
        sigma_rad = np.deg2rad(reso_1pix_deg)
        smooth_rot_map = hp.smoothing(rotated_map, sigma=sigma_rad)

        return smooth_rot_map


    def DM_pos_to_healpix(self, Nside_map, DM_pos_in, obs_pos_in):

        # Center on observer position.
        DM_pos = DM_pos_in - obs_pos_in
        xDM, yDM, zDM = DM_pos[:,0], DM_pos[:,1], DM_pos[:,2]

        # Convert x,y,z to angles.
        proj_xy_plane_dis = np.sqrt(xDM**2 + yDM**2)

        thetas = np.arctan2(zDM, proj_xy_plane_dis)
        phis = np.arctan2(yDM, xDM)

        # To galactic latitude and longitude (in degrees) for healpy.
        hp_glon, hp_glat = np.rad2deg(phis), np.rad2deg(thetas)

        # Convert angles to pixel indices using ang2pix.
        pixel_indices = hp.ang2pix(
            Nside_map, hp_glon, hp_glat, lonlat=True
        )

        # Create a Healpix map and increment the corresponding pixels
        DM_healpix_map = np.zeros(hp.nside2npix(Nside_map))
        np.add.at(DM_healpix_map, pixel_indices, 1)

        return DM_healpix_map


    def syncronize_all_sky_maps(self, halo, nu_mass_eV, apply=True):

        if halo == 0:
            end_str = f'benchmark_halo'
        else:
            halo_label = self.final_halos[halo-1]
            end_str = f'halo{halo_label}'

        # Get number densities and convert to clustering factors.
        nu_mass_idx = (np.abs(self.mrange-nu_mass_eV)).argmin()
        dens_nu = self.number_densities_numerical_all_sky[...,nu_mass_idx]
        etas_nu = dens_nu/self.N0_pix

        etas_halo = etas_nu[halo-1,...]  # select input halo

        eta_min, eta_max = np.min(etas_halo), np.max(etas_halo)
        factor = np.round(eta_max/eta_min, 2)
        print(f'Halo {halo_label} original values: min={eta_min}, max={eta_max}, factor={factor}')
        
        # DM halo directory.
        parent = str(pathlib.Path(f'{self.sim_dir}').parent)
        DM_dir = f'{parent}/final_halo_data'
        NFW_dir = f'{parent}/benchmark_halo_files'

        if halo == 0:
            # Load benchmark NFW halo positions.
            pos_origin = np.load(f'{NFW_dir}/benchmark_halo_snap_0036.npy')
        else:
            # Load Halo DM positions.
            haloID = self.halo_indices[halo-1]
            pos_origin = np.load(
                f'{DM_dir}/DM_pos_origID{haloID}_snap_0036.npy'
            )
    
        if apply:

            # Rotate (depending on initial cell) and (gaussian) smooth the map.
            init_xyz = np.load(f'{self.sim_dir}/init_xyz_{end_str}.npy')
            healpix_map = self.rot_int_smooth_healpix_map(etas_halo, init_xyz)

            # Rotate DM to bring CoP of halo to origin (middle) in plot.
            DM_rot, obs_rot = self.rotate_DM(pos_origin, init_xyz)

            # Get healpix map from cartesian DM coords.
            DM_healpix_map = self.DM_pos_to_healpix(self.Nside, DM_rot, obs_rot)

        else:
            healpix_map = etas_halo
            DM_healpix_map = self.DM_pos_to_healpix(self.Nside, pos_origin, 0)

        return healpix_map, DM_healpix_map, end_str, eta_min, eta_max, factor


    def plot_all_sky_map(self, sim_method, halo, nu_mass_eV):

        nu_mass_idx = (np.abs(self.mrange-nu_mass_eV)).argmin()

        if sim_method == 'analytical':

            dens_nu = self.number_densities_analytical_all_sky[...,nu_mass_idx]
            etas_nu = dens_nu/self.N0_pix

            healpix_map = etas_nu

            # Create a grid of (theta, phi) coordinates for the map
            theta, phi = hp.pix2ang(self.Nside, np.arange(len(healpix_map)))

            # Rotation for Virgo Cluster.
            # VCzAngle = np.rad2deg(
            #     np.arctan2(X_VC[1], -X_VC[0])
            # )
            # VCyAngle = np.rad2deg(
            #     np.arctan2(-X_VC[2], np.linalg.norm(X_VC[:2]))
            # )
            # rot = hp.rotator.Rotator(
            #     rot=[0, 90, 0], deg=True, eulertype='ZYX', inv=False
            # )

            # Rotation for Milky Way only.
            rot = hp.rotator.Rotator(
                rot=[-180, 0, 0], deg=True, eulertype='ZYX', inv=False
            )

            # Apply rotation to the grid
            theta_rot, phi_rot = rot(theta, phi)

            # Find the pixel indices of the rotated grid
            pix_rot = hp.ang2pix(self.Nside, theta_rot, phi_rot)

            # Create a new map with rotated pixel values
            rotated_map = np.zeros_like(healpix_map)
            rotated_map[pix_rot] = healpix_map

            # Find zero-valued pixels and interpolate their values from 4 neighbors
            zero_pixels = np.where(rotated_map == 0)[0]
            for pix in zero_pixels:
                theta_zero, phi_zero = hp.pix2ang(self.Nside, pix)
                interp_val = hp.get_interp_val(
                    healpix_map, theta_zero, phi_zero
                )
                rotated_map[pix] = interp_val

            hp.newvisufunc.projview(
                rotated_map,
                coord=['G'],
                title=f'Analytical (Npix={self.Npix})',
                unit=r'$n_{\nu, pix} / n_{\nu, pix, 0}$',
                cmap=cc.cm.CET_D1A,
                graticule=True,
                graticule_labels=True,
                xlabel="longitude",
                ylabel="latitude",
                cb_orientation="horizontal",
                projection_type="mollweide",
                flip='astro',
                # cbar_ticks=[],
                show_tickmarkers=True,
                latitude_grid_spacing=45,
                longitude_grid_spacing=45,
                alpha=1,
                phi_convention='counterclockwise',
                # min=0.7,
                # max=5.5
            )

            plt.savefig(
                f'{self.fig_dir}/All_sky_map_analytical_{nu_mass_eV}eV.pdf', 
                bbox_inches='tight'
            )
            plt.show()
            plt.close()

            return healpix_map

        if sim_method == 'numerical':

            # Load synchronized maps.
            healpix_map, DM_healpix_map, end_str, *_ = self.syncronize_all_sky_maps(
                halo, nu_mass_idx
            )

            fig = plt.figure(figsize =(12, 4.4))
            fig.tight_layout()


            ### ----------------------------------- ###
            ### Plot clustering factor all_sky map. ###
            ### ----------------------------------- ###
            
            ax1 = fig.add_subplot(121)

            # Remove all ticks, labels and frames, to only show mollview plot.
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax1.get_xaxis().set_ticks([])
            ax1.get_yaxis().set_ticks([])

            # Make center value to be 1 (no overdensity).
            mid = 1.
            divnorm = mcolors.TwoSlopeNorm(vcenter=mid)

            # All-sky map in mollview projection.
            hp.newvisufunc.projview(
                healpix_map, unit=r'$n_{\nu, pix} / n_{\nu, pix, 0}$',
                cmap=cc.cm.CET_D1,
                override_plot_properties={
                    "cbar_pad": 0.1,
                    # "cbar_label_pad": 15,
                },
                cbar_ticks=[np.min(healpix_map), mid, np.max(healpix_map)],
                sub=121, norm=divnorm,
                **self.healpy_dict
            )


            ### ----------------------------------- ###
            ### Plot clustering factor all_sky map. ###
            ### ----------------------------------- ###

            # Plot DM line-of-sight content as healpix map.
            ax2 = fig.add_subplot(122)

            # Remove all ticks, labels and frames, to only show mollview plot.
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])

            # All-sky map in mollview projection.
            hp.newvisufunc.projview(
                DM_healpix_map,
                cmap='Purples', alpha=1, norm='log',
                unit='DM particles in line-of-sight', cbar=True, 
                override_plot_properties={
                    "cbar_pad": 0.1,
                    "cbar_label_pad": 1,
                },
                sub=122, 
                **self.healpy_dict
            )

            # Adjust and save figure.
            plt.subplots_adjust(wspace=0.15)
            plt.savefig(
                f'{self.fig_dir}/All_sky_maps_{end_str}.pdf', 
                bbox_inches='tight'
            )
            plt.close()


    def plot_all_sky_power_spectra(self, halo, nu_mass_idx):

        # Load synchronized maps.
        healpix_map, DM_healpix_map, end_str = self.syncronize_all_sky_maps(
            halo, nu_mass_idx
        )

        # Unit to compare to similar figures in literature.
        micro_Kelvin_unit = 1e12

        # Compute power spectrum of number density all-sky map.
        cl = hp.sphtfunc.anafast(healpix_map)
        ell = np.arange(len(cl))
        power_spectrum = ell*(ell+1)*cl/(2*Pi)*micro_Kelvin_unit

        # Compute cross-correlation spectrum of n_nu and DM maps.
        cross = hp.sphtfunc.anafast(healpix_map, DM_healpix_map)
        cross_power_spectrum = ell*(ell+1)*cross/(2*Pi)*micro_Kelvin_unit

        # Pearsons correlation coefficient for total map information.
        pearson_r, _ = pearsonr(healpix_map, DM_healpix_map)

        fig = plt.figure(figsize =(12, 4))
        fig.tight_layout()

        ax1 = fig.add_subplot(121)
        ell = np.arange(len(cl))
        ax1.loglog(ell, power_spectrum)
        ax1.set_xlabel("$\ell$")
        ax1.set_xlim(1,)
        ax1.set_ylabel("$\ell(\ell+1)C_{\ell}$")
        ax1.grid()

        ax2 = fig.add_subplot(122)
        ell = np.arange(len(cross))
        ax2.plot(ell, cross_power_spectrum)
        ax2.set_xlabel("$\ell$")
        ax2.set_ylabel("$\ell(\ell+1)C_{\ell}$")
        ax2.grid()

        plt.savefig(
            f'{self.fig_dir}/power_spectra_{end_str}.pdf', 
            bbox_inches='tight'
        )
        plt.close()

        return pearson_r
    

    def plot_all_spectra_1plot(self, halo_array, nu_mass_idx):

        fig = plt.figure(figsize =(12, 4))
        fig.tight_layout()

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Unit to compare to similar figures in literature.
        micro_Kelvin_unit = 1e12

        eta_mins = np.zeros(len(halo_array))
        eta_maxs = np.zeros(len(halo_array))
        factors = np.zeros(len(halo_array))
        for i, halo in enumerate(halo_array):

            # Load synchronized maps.
            healpix_map, DM_healpix_map, _, eta_min, eta_max, factor = self.syncronize_all_sky_maps(
                halo, nu_mass_idx, apply=False
            )
            eta_mins[i] = eta_min
            eta_maxs[i] = eta_max
            factors[i] = factor

            # Convert maps of neutrino densities to temperatures.
            healpix_map = np.cbrt(healpix_map*2*Pi**2/3/zeta(3))

            # Compute power spectrum of number density all-sky map.
            cl = hp.sphtfunc.anafast(healpix_map)
            ell = np.arange(len(cl))
            power_spectrum = ell*(ell+1)*cl/(2*Pi)*micro_Kelvin_unit

            # Compute cross-correlation spectrum of n_nu and DM maps.
            cross = hp.sphtfunc.anafast(healpix_map, DM_healpix_map)
            cross_power_spectrum = ell*(ell+1)*cross/(2*Pi)*micro_Kelvin_unit

            # Pearsons correlation coefficient for total map information.
            # pearson_r, _ = pearsonr(healpix_map, DM_healpix_map)

            ax1.semilogy(ell, power_spectrum)
            ax2.plot(ell, cross_power_spectrum)
        
        ax1.set_xlabel("$\ell$")
        ax1.set_xlim(1,)
        ax1.set_ylabel("$\ell(\ell+1)C_{\ell} [\mu K^2]$")
        ax1.grid()

        ax2.set_xlabel("$\ell$")
        ax2.set_ylabel("$\ell(\ell+1)C_{\ell}$")
        ax2.grid()

        plt.savefig(
            f'{self.fig_dir}/all_power_spectra.pdf', 
            bbox_inches='tight'
        )
        plt.close()

        # Save arrays.
        def write_arrays_to_file(
                arr1, arr2, arr3, head1, head2, head3, filename
        ):
            df = pd.DataFrame({head1: arr1, head2: arr2, head3: arr3})
            df = df.round(2)
            df.to_csv(filename, index=False)

        write_arrays_to_file(
            eta_mins, eta_maxs, factors, 
            'Min', 'Max', 'Factor', 
            f'{self.sim_dir}/all_sky_values_original.csv'
        )


    def plot_pixel_correlation(self, halo, nu_mass_idx):
        """Scatterplot of number densities (sorted) pixel values vs. corresponding DM l.o.s. amount of same pixel. Correlation should be what pearson's r measures."""
        #? Is it visible? Is there significant spread/noise?

        # Load synchronized maps.
        healpix_map, DM_healpix_map, end_str = self.syncronize_all_sky_maps(
            halo, nu_mass_idx
        )

        sort_idx = healpix_map.argsort()
        healpix_map_sort = healpix_map[sort_idx]
        DM_healpix_map_sync = DM_healpix_map[sort_idx]

        plt.scatter(healpix_map_sort, DM_healpix_map_sync)
        plt.savefig(
            f'{self.fig_dir}/pix_corr_{end_str}.pdf', 
            bbox_inches='tight'
        )
        plt.close()


    def plot_neutrinos_inside_Rvir(self):
        
        # All positions and velocities across redshifts.
        pos = self.vectors_numerical[...,0:3]
        vel = self.vectors_numerical[...,3:6]
        # (halos, neutrinos, z_int_steps, 3)

        # Radii and velocity magnitudes.
        rad = np.sqrt(np.sum(pos**2, axis=-1))
        mag = np.sqrt(np.sum(vel**2, axis=-1))

        if mag.shape[0] < 100:
            nu_axis = 1
        else:
            nu_axis = 0
        neutrinos = mag.shape[nu_axis]
        
        # Read R_vir of halo sample.
        halo_params = np.load(
            glob.glob(f'{self.sim_dir}/halo*params.npy')[0]
        )

        # Halo sample starting (z=0) parameters.
        R_vir_sample_z0 = halo_params[:,0]
        M_vir_sample = 10**halo_params[:,1]

        
        tree_path = f'{self.sim_dir}/MergerTree.hdf5'
        halo_IDs = np.load(glob.glob(f'{self.sim_dir}/halo*indices.npy')[0])

        with h5py.File(tree_path) as tree:

            inds = list(halo_IDs.flatten())
            Masses_box = np.take(
                np.array(tree['Assembly_history/Mass']), inds, axis=0)
            Masses = np.asarray(Masses_box)*Msun

            zeds_box = tree['Assembly_history/Redshift']
            zeds = np.asarray(zeds_box)


            # Box parameters.
            with open(f'{self.sim_dir}/box_parameters.yaml', 'r') as file:
                box_setup = yaml.safe_load(file)
            H0 = box_setup['Cosmology']['h']*100*km/s/Mpc
            Omega_M = box_setup['Cosmology']['Omega_M']
            Omega_L = box_setup['Cosmology']['Omega_L']

            # Calculate evolution of R_vir of halo sample.
            R_vir_l = []
            for z, Mz in zip(zeds, Masses):
                rho_c = fct_rho_crit(z, H0, Omega_M, Omega_L)
                R_vir_l.append(np.power(Mz / (200*rho_c*4/3*Pi), 1./3.))
        
        R_vir_sample = np.array(R_vir_l)

        # Select only values at box snapshots
        z_int_steps = np.load(f'{self.sim_dir}/z_int_steps.npy')
        snap_ids = [np.abs(z_int_steps - z).argmin() for z in zeds]
        rad_snaps = rad[...,snap_ids]

        cond = (rad_snaps <= R_vir_sample[:, np.newaxis, :]/kpc)
        perc_in_Rvir = np.count_nonzero(cond, axis=nu_axis)/neutrinos*100

        '''
        # Escape velocity at redshift z, dependent on halo parameters.
        # Read in R_vir, R_s and rho_0. x_i will be R_vir

        v_esc = escape_momentum_analytical(x_i, R_vir, R_s, rho_0, None)

        # MW_esc = 550  # km/s         
        
        esc_cond = (mag*(kpc/s)/(km/s) <= MW_esc)
        # vals, esc_num = np.unique(esc_cond, return_counts=True, axis=0)
        perc_in_Rvir = np.count_nonzero(esc_cond, axis=nu_axis)/neutrinos*100
        print(perc_in_Rvir.shape)
        # '''


        for i, perc_curve in enumerate(perc_in_Rvir):
            plt.semilogx(
                1+zeds, perc_curve, label=f'Mvir = {M_vir_sample[i]:.2e}'
            )

        # Plot specific redshift markers.
        plt.axvline(1+0.5, c='red', ls='-.', label='z=0.5')
        plt.axvline(1+1., c='orange', ls='-.', label='z=1')

        # plt.ylim(50, 60)
        plt.xlabel('z')
        plt.ylabel('%')
        plt.title(r'% of $\nu$ inside $R_{vir}$')
        plt.legend()
        plt.show()


    def get_halo_params(self):

        # Get halo parameters.
        Rvir = self.halo_params[:,0]
        Mvir = 10**self.halo_params[:,1]
        conc = self.halo_params[:,2]

        # Get initial distances (of cells at z=0).
        init_xyz_paths = glob.glob(f'{self.sim_dir}/init_xyz_halo*.npy')
        init_xyz_halos = np.array([np.load(path) for path in init_xyz_paths])
        init_dis_halos = np.linalg.norm(init_xyz_halos, axis=-1)
        init_dis_halos = init_dis_halos[self.halo_glob_order]

        return Rvir, Mvir, conc, init_dis_halos


    def plot_eta_vs_halo_params(self):
        
        Rvir, Mvir, conc, init_dis_halos = self.get_halo_params()

        # Get number densities and convert to clustering factors.
        etas_nu = np.flip(self.etas_numerical, axis=-1)
        
        # Show relative change to median for each mass.
        etas_0 = np.median(etas_nu, axis=0)
        etas_nu = (etas_nu-etas_0)/etas_0

        fig = plt.figure(figsize=(4, 6))
        cmap = np.flip(np.array(mcp.gen_color(cmap='RdYlBu', n=etas_nu.shape[-1])))
        size = 5

        # Virial mass.
        ax1 = fig.add_subplot(111)
        for etas, color in zip(etas_nu[Mvir.argsort()].T, cmap):
            ax2.scatter(
                Mvir[Mvir.argsort()], 
                etas, c=color, s=size)
        ax2.set_xlabel(r'$R_{vir}$')
        ax2.set_yscale('symlog', linthresh=0.00001)

        # ax1 = fig.add_subplot(211)
        # for etas, color in zip(etas_nu[Mvir.argsort()].T, cmap):
        #     pos_cond = np.argwhere(etas > 0)
        #     ax1.scatter(
        #         Mvir[Mvir.argsort()][pos_cond], 
        #         etas[pos_cond], c=color, s=size)
        # ax1.set_ylabel(r'$(f-\bar{f})/\bar{f}$')
        # ax1.set_xlabel(r'$M_{vir}$')
        # ax1.set_yscale('log')

        # ax2 = fig.add_subplot(212, sharey=ax1)
        # for etas, color in zip(etas_nu[Mvir.argsort()].T, cmap):
        #     neg_cond = np.argwhere(etas < 0)
        #     ax2.scatter(
        #         Mvir[Mvir.argsort()][neg_cond], 
        #         np.abs(etas[neg_cond]), c=color, s=size)
        # ax2.set_xlabel(r'$R_{vir}$')
        # ax2.set_yscale('log')
        # ax2.invert_yaxis()


        # Virial radius.
        # ax2 = fig.add_subplot(142, sharey=ax1)
        # for etas, color in zip(etas_nu[Rvir.argsort()].T, cmap):
        #     ax2.scatter(
        #         # np.arange(len(conc)), 
        #         Rvir[Rvir.argsort()], 
        #         etas, c=color, s=size)
        # ax2.set_xlabel(r'$R_{vir}$')

        # Concentration.
        # ax3 = fig.add_subplot(132, sharey=ax1)
        # for etas, color in zip(etas_nu[conc.argsort()].T, cmap):
        #     ax3.scatter(
        #         # np.arange(len(conc)), 
        #         conc[conc.argsort()],
        #         etas, c=color, s=size)
        # ax3.set_xlabel('concentration')

        # Initial distance.
        # ax4 = fig.add_subplot(133, sharey=ax1)
        # for etas, color in zip(etas_nu[init_dis_halos.argsort()].T, cmap):
        #     ax4.scatter(
        #         # np.arange(len(conc)), 
        #         init_dis_halos[init_dis_halos.argsort()],
        #         etas, c=color, s=size)
        # ax4.set_xlabel('initial distance')

        # ax1.set_ylim(-0.3, 0.5)
        # ax2.set_ylim(-0.3, 0.5)
        # ax3.set_ylim(-0.3, 0.5)
        # ax4.set_ylim(-0.3, 0.5)

        fig.tight_layout()
        plt.savefig(
            f'{self.fig_dir}/Clustering_halo_params_dependence.pdf', 
            bbox_inches='tight'
        )
        plt.show(); plt.close()
        

    def plot_2d_params(self, nu_mass_eV):
        
        # Make scatterplot with Mvir vs. conc. (for example).
        # Coloring is then from lowest to highest (f-\bar{f})/\bar{f},
        # i.e. the relative clustering factor.
        # (choose different colorscheme than blue to red to not confuse with other plot idea)

        Rvir, Mvir, conc, init_dis_halos = self.get_halo_params()

        # Closest mass (index) for chosen neutrino mass.
        nu_mass_idx = (np.abs(self.mrange-nu_mass_eV)).argmin()

        # Get clustering factors.
        etas_nu = self.etas_numerical[...,nu_mass_idx].flatten()

        # Show relative change (choice of median, mean, etc.) for each mass.
        # etas_0 = np.median(etas_nu, axis=0)
        # etas_nu = (etas_nu-etas_0)/etas_0

        fig = plt.figure(figsize=(11, 5))
        # cmap_plot = np.array(mcp.gen_color(cmap='bwr', n=len(etas_nu)))
        cmap_plot = np.array(mcp.gen_color(cmap='coolwarm', n=len(etas_nu)))
        size = 30

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharey=ax1)

        # Sort arrays by (relative) clustering factors.
        plot_ind = etas_nu.argsort()
        Mvir_plot = Mvir[plot_ind]
        conc_plot = conc[plot_ind]
        init_plot = init_dis_halos[plot_ind]
        # etas_plot = etas_nu[plot_ind]
        # cmap_plot = cmap_custom[plot_ind]

        sct1 = ax1.scatter(conc_plot, Mvir_plot, c=cmap_plot, s=size)
        ax1.set_ylabel(r'$M_{vir}$')
        ax1.set_xlabel('concentration')
        
        sct2 = ax2.scatter(init_plot, Mvir_plot, c=cmap_plot, s=size)
        ax2.set_xlabel('initial distance (kpc)')

        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Create and remove the colorbar for the first subplot.
        # cbar1 = fig.colorbar(sct1, cax = cax1)
        # fig.delaxes(fig.axes[2])


        # for col, ax in enumerate(axes):
        # im = ax.pcolormesh(data, vmin=data.min(), vmax=data.max())
        # if col > 0:
        #     ax.yaxis.set_visible(False)



        norm = mpl.colors.Normalize(vmin=np.min(etas_nu), vmax=np.max(etas_nu))
        cmap = mpl.cm.coolwarm
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cax,
                    orientation='vertical', 
                    label=r'$n_\nu / n_{\nu,0}$',
                    # pad=0.05
                )
        
        ax1.grid()
        ax2.grid()
        plt.savefig(f'{self.fig_dir}/2D_params_box_halos.pdf', bbox_inches='tight')



# ======================== #
# '''
sim_dir = f'L025N752/DMONLY/SigmaConstant00/all_sky_final'

objects = (
    # 'NFW_halo', 
    'box_halos', 
    # 'analytical_halo'
)
Analysis = analyze_simulation_outputs_test(
    sim_dir = sim_dir, 
    objects = objects,
    sim_type = 'all_sky',
)

print(Analysis.final_halos)
print(Analysis.halo_num)
halo_array = np.arange(Analysis.halo_num)+1

# Generate power spectra plots.
Analysis.plot_all_spectra_1plot(halo_array, 0.3)

# Generate all all-sky anisotropy maps.
# for halo in halo_array:
#     Analysis.plot_all_sky_map('numerical', halo, 0.3)
# '''
# ======================== #



# ======================== #
'''
sim_dir = f'L025N752/DMONLY/SigmaConstant00/single_halos'

objects = (
    # 'NFW_halo', 
    'box_halos', 
    # 'analytical_halo'
)
Analysis = analyze_simulation_outputs_test(
    sim_dir = sim_dir, 
    objects = objects,
    sim_type = 'single_halos',
)

# Analysis.plot_eta_vs_halo_params()
Analysis.plot_2d_params(nu_mass_eV=0.3)
# '''
# ======================== #