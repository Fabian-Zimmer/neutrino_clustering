from shared.preface import *
from shared.shared_functions import *
import matplotlib.colors as mcolors
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mycolorpy import colorlist as mcp

class analyze_simulation_outputs(object):

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
        paths_all = glob.glob(f'{self.sim_dir}/init_xyz_halo*.npy')
        nums_all = np.array(
            [list(map(int, re.findall(r'\d+', path)))[-1] for path in paths_all]
        ).flatten()

        # note: besides the broken halo 20, halos 24 & 25 are anomalies.
        # Remove anomalous halos from arrays.
        anomalies = [24,25]
        del_ids = np.array(
            [i for i, x in enumerate(nums_all) if x in anomalies])
        nums = np.array(
            [x for i, x in enumerate(nums_all) if i not in del_ids])
        paths = np.array(
            [x for i, x in enumerate(paths_all) if i not in del_ids])

        limit_halos = True

        if limit_halos:
            until = 3
        else:
            until = -1

        self.halo_glob_order = nums.argsort()
        self.final_halos = nums[self.halo_glob_order][:until]
        self.halo_num= len(self.final_halos)

        # Halo indices and parameters for the final halos.
        self.halo_indices = np.load(
            glob.glob(f'{self.sim_dir}/halo_batch*indices.npy')[0]
        )[self.final_halos-1]
        self.halo_params = np.load(
            glob.glob(f'{self.sim_dir}/halo_batch*params.npy')[0]
        )[self.final_halos-1]

        # Initial distances (of starting cell at z=0) for the final halos.
        init_xyz_pre = np.array([np.load(path) for path in paths])
        self.init_xyz = init_xyz_pre[self.halo_glob_order][:until]
        self.init_dis = np.linalg.norm(self.init_xyz, axis=-1)[:until]


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

                # Milky Way params as in Mertsch et al.
                batch_paths = glob.glob(
                    f'{self.sim_dir}/neutrino_vectors_analytical_batch*.npy'
                )
                
                self.vectors_analytical = []
                for batch_path in batch_paths:
                    self.vectors_analytical.append(np.load(batch_path))
                self.vectors_analytical = np.squeeze(
                    np.array(self.vectors_analytical)
                )
                self.vectors_analytical_Mertsch = np.array(self.vectors_analytical)

                self.etas_analytical_Mertsch = np.load(
                    f'{self.sim_dir}/number_densities_analytical_single_halos.npy'
                )/N0

                # Using median parameters of box halo sample. 
                batch_paths = glob.glob(
                    f'{self.sim_dir}/neutrino_vectors_analytical_median_batch*.npy'
                )
                
                self.vectors_analytical = []
                for batch_path in batch_paths:
                    self.vectors_analytical.append(np.load(batch_path))
                self.vectors_analytical = np.squeeze(
                    np.array(self.vectors_analytical)
                )
                self.vectors_analytical = np.array(self.vectors_analytical)

                self.etas_analytical = np.load(
                    f'{self.sim_dir}/number_densities_analytical_median_single_halos.npy'
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

                self.number_densities_all_sky_benchmark = np.load(
                    f'{self.sim_dir}/number_densities_numerical_benchmark_halo_all_sky.npy'
                )[np.newaxis,...]


            if 'analytical_halo' in self.objects:

                self.number_densities_analytical_all_sky = np.load(
                    f'{self.sim_dir}/number_densities_analytical_all_sky.npy'
                )


    def get_halo_params(self):

        # Get halo parameters.
        Rvir = self.halo_params[:,0]
        Mvir = 10**self.halo_params[:,1]
        conc = self.halo_params[:,2]

        return Rvir, Mvir, conc
    

    def plot_overdensity_band(self, plot_ylims):

        ### ------------- ###
        ### Setup figure. ###
        ### ------------- ###
        fig, ax = plt.subplots(1,1)

        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_title(f'Overdensity band')
        ax.set_xlabel(r'$m_{\nu}$ [meV]')
        ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$')

        if plot_ylims is not None:
            ax.set_ylim(plot_ylims[0], plot_ylims[1])
        
        plt.grid(True, which="both", ls="-")

        savefig_args = dict(
            bbox_inches='tight'
        )


        ### ------------- ###
        ### Plot objects. ###
        ### ------------- ###

        if 'NFW_halo' in self.objects:

            plt.plot(
                self.mrange*1e3, self.etas_benchmark-1, 
                color='green', label='(Benchmark) NFW Halo'
            )

        if 'box_halos' in self.objects:

            etas_median = np.median(
                self.etas_numerical, axis=0)
            etas_perc2p5 = np.percentile(
                self.etas_numerical, q=2.5, axis=0)
            etas_perc97p5 = np.percentile(
                self.etas_numerical, q=97.5, axis=0)
            etas_perc16 = np.percentile(
                self.etas_numerical, q=16, axis=0)
            etas_perc84 = np.percentile(
                self.etas_numerical, q=84, axis=0)
            
            ax.plot(
                self.mrange*1e3, (etas_median-1), color='blue', 
                label='Box Halos: medians'
            )
            ax.fill_between(
                self.mrange*1e3, (etas_perc2p5-1), (etas_perc97p5-1), 
                color='blue', alpha=0.2, label='Box Halos: 2.5-97.5 % C.L.'
            )
            ax.fill_between(
                self.mrange*1e3, (etas_perc16-1), (etas_perc84-1), 
                color='blue', alpha=0.3, label='Box Halos: 16-84 % C.L.'
            )

        if 'analytical_halo' in self.objects:

            plt.plot(
                self.mrange*1e3, self.etas_analytical-1, 
                color='red', ls='solid', label='Analytical Halo'
            )

            # Plot endpoint values from Mertsch et al (2020).
            # x_ends = [1e1, 3*1e2]
            # y_ends = [3*1e-3, 4]
            # ax.scatter(x_ends, y_ends, marker='x', s=15, color='orange')
            
            # Plot reproduction of Mertsch et al., using their MW params.
            plt.plot(
                self.mrange*1e3, self.etas_analytical_Mertsch-1, 
                color='red', ls='dashed', label='Parameters as in Mertsch+(2020)'
            )

        plt.legend(loc='lower right')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
        plt.savefig(f'{self.fig_dir}/overdensity_band.pdf', **savefig_args)


    def plot_overdensity_evolution(self, plot_ylims:tuple):

        fig, ax = plt.subplots(1,1, figsize=(8,12))

        z_int_steps = np.load(f'{self.sim_dir}/z_int_steps.npy')
        colors = ['blue', 'orange', 'green', 'red']


        ### ---------------- ###
        ### Analytical halo. ###
        ### ---------------- ###

        if 'analytical_halo' in self.objects:

            # Convert velocities to momenta.
            p_ana, _ = velocity_to_momentum(
                self.vectors_analytical[...,3:6], self.mpicks)

            # Overdensities for each redshift as if it was the last in the sim.
            inds = np.arange(p_ana.shape[-1])
            ns_ana_zeds = np.array(
                [number_density(p_ana[...,0], p_ana[...,z]) for z in inds]).T
            etas_ana_zeds = ns_ana_zeds/N0

            for j, m in enumerate(self.mpicks):
                ax.plot(
                    z_int_steps, etas_ana_zeds[j]-1, 
                    c=colors[j], ls='dashed', 
                    label=f'Analytical Halo: {m:.3f} eV')


        ### ---------------------- ###
        ### Box (numerical) halos. ###
        ### ---------------------- ###

        if 'box_halos' in self.objects:

            ns_num_zeds = []
            halo_num = len(self.vectors_numerical)
            for halo_j in range(halo_num):

                # Load and combine velocity vectors (batches).
                vels_batches = self.vectors_numerical[halo_j,...,3:6]
                vels_in = vels_batches.reshape(-1, 100, 3)

                # Convert velocities to momenta.
                p_num, _ = velocity_to_momentum(vels_in, self.mpicks)

                inds = np.arange(p_num.shape[-1])
                ns_num_zeds.append(
                    np.array(
                    [number_density(p_num[...,0], p_num[...,z]) for z in inds]).T)
            
            etas_num_zeds = np.array(ns_num_zeds)/N0

            etas_median = np.median(etas_num_zeds, axis=0)
            etas_perc2p5 = np.percentile(etas_num_zeds, q=2.5, axis=0)
            etas_perc97p5 = np.percentile(etas_num_zeds, q=97.5, axis=0)
            etas_perc16 = np.percentile(etas_num_zeds, q=16, axis=0)
            etas_perc84 = np.percentile(etas_num_zeds, q=84, axis=0)

            for j, m in enumerate(self.mpicks):
                ax.plot(
                    1+z_int_steps, etas_median[j]-1, 
                    color=colors[j], 
                    label=f'{m:.3f} eV')
                ax.fill_between(
                    1+z_int_steps, (etas_perc2p5[j]-1), (etas_perc97p5[j]-1),
                    color=colors[j], alpha=0.2, 
                    label='Box Halos: 2.5-97.5 % C.L.')
                ax.fill_between(
                    1+z_int_steps, (etas_perc16[j]-1), (etas_perc84[j]-1),
                    color=colors[j], alpha=0.3, 
                    label='Box Halos: 16-84 % C.L.')


        # Invert ordering of items in legend (looks better since the curves 
        # of higher masses are higher up in the plot).
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='lower right')

        # ax.set_title('Overdensities (redshift) evolution')
        ax.set_xlabel('z')
        ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(plot_ylims[0], plot_ylims[1])

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
        plt.savefig(
            f'{self.fig_dir}/overdensity_evolution.pdf', bbox_inches='tight')
        plt.close()


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
            # theta_zero, phi_zero = hp.pix2ang(Nside, pix)
            # interp_val = hp.get_interp_val(
            #     rotated_map, theta_zero, phi_zero
            # )

            neighbours_pix = hp.get_all_neighbours(Nside, pix)
            interp_val = np.mean(rotated_map[neighbours_pix])

            rotated_map[pix] = interp_val

        # Smooth the rotated map using a Gaussian kernel.
        reso_1pix_deg = hp.nside2resol(Nside, arcmin=True) / 60
        sigma_rad = np.deg2rad(reso_1pix_deg)/2
        smooth_rot_map = hp.smoothing(rotated_map, sigma=sigma_rad)

        # return smooth_rot_map
        return rotated_map


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

        # Shared between NFW and box halos.
        parent = str(pathlib.Path(f'{self.sim_dir}').parent)
        nu_mass_idx = (np.abs(self.mrange-nu_mass_eV)).argmin()

        if halo == 0:
            end_str = f'benchmark_halo'

            # Load benchmark NFW halo positions.
            NFW_dir = f'{parent}/benchmark_halo_files'
            pos_origin = np.load(f'{NFW_dir}/benchmark_halo_snap_0036.npy')

            # Get number densities and convert to clustering factors.
            dens_nu = self.number_densities_all_sky_benchmark[...,nu_mass_idx]
            etas_halo = dens_nu[0]/self.N0_pix

            eta_min, eta_max = np.min(etas_halo), np.max(etas_halo)
            factor = np.round(eta_max/eta_min, 2)

        else:
            halo_label = self.final_halos[halo-1]
            end_str = f'halo{halo_label}'

            # Get number densities and convert to clustering factors.
            dens_nu = self.number_densities_numerical_all_sky[...,nu_mass_idx]
            etas_nu = dens_nu/self.N0_pix

            etas_halo = etas_nu[halo-1,...]  # select input halo

            eta_min, eta_max = np.min(etas_halo), np.max(etas_halo)
            factor = np.round(eta_max/eta_min, 2)
            print(f'Halo {halo_label} original values: min={eta_min}, max={eta_max}, factor={factor}')
        
            # Load Halo DM positions.
            DM_dir = f'{parent}/final_halo_data'
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


    def syncronize_all_sky_maps_analytical(self):

        synced_maps = []
        for nu_mass_eV in (0.01, 0.3):

            nu_mass_idx = (np.abs(self.mrange-nu_mass_eV)).argmin()
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
                    rotated_map, theta_zero, phi_zero
                )
                rotated_map[pix] = interp_val

            synced_maps.append(rotated_map)

        return np.array(synced_maps)


    def plot_all_sky_map(self, sim_method, halo, nu_mass_eV):

        if sim_method == 'analytical':

            # Get maps for lighest and heaviest neutrino masses.
            synced_maps = self.syncronize_all_sky_maps_analytical()
            heavy_healpix_map = synced_maps[1]
            light_healpix_map = synced_maps[0]

            ### ------------------------------------ ###
            ### Plot clustering factor all_sky maps. ###
            ### ------------------------------------ ###

            fig = plt.figure(figsize =(12, 4.4))
            fig.tight_layout()
            
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            # Remove all ticks, labels and frames, to only show mollview plot.
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax1.get_xaxis().set_ticks([])
            ax1.get_yaxis().set_ticks([])
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])

            # Create original colormap
            cmap = plt.get_cmap('coolwarm')
            colors = cmap(np.linspace(0, 1, cmap.N))

            # Create a TwoSlopeNorm instance for each map
            norm1 = mcolors.TwoSlopeNorm(vmin=1, vcenter=5, vmax=10)
            norm2 = mcolors.TwoSlopeNorm(vmin=1, vcenter=5, vmax=10)

            # Split colormap
            mid_point = int(0.5 * cmap.N)
            left_end = int(0.8*cmap.N)
            cmap_left = mcolors.LinearSegmentedColormap.from_list(
                'cmap_left', colors[mid_point:left_end], N=256)
            right_start = int(0.7 * cmap.N)
            cmap_right = mcolors.LinearSegmentedColormap.from_list(
                'cmap_right', colors[right_start:], N=256)


            # Plot the lighest mass.
            hp.newvisufunc.projview(
                light_healpix_map,
                unit=r'$n_{\nu, pix} / n_{\nu, pix, 0}$',
                # cmap='Reds',
                cmap=cmap_left,
                override_plot_properties={
                    "cbar_pad": 0.1,
                    # "cbar_label_pad": 15,
                },
                sub=121,
                # norm=norm1,
                # norm='log',
                **self.healpy_dict
            )

            diff = np.max(heavy_healpix_map) - np.min(heavy_healpix_map)
            mid = np.min(heavy_healpix_map)
            divnorm = mcolors.TwoSlopeNorm(vcenter=mid)

            # Plot the heaviest mass.
            hp.newvisufunc.projview(
                heavy_healpix_map,
                unit=r'$n_{\nu, pix} / n_{\nu, pix, 0}$',
                cmap=cmap_right,
                override_plot_properties={
                    "cbar_pad": 0.1,
                    # "cbar_label_pad": 15,
                },
                sub=122,
                # norm=norm2,
                # norm='log',
                **self.healpy_dict
            )

            plt.savefig(
                f'{self.fig_dir}/All_sky_maps_analytical.pdf', 
                bbox_inches='tight'
            )
            plt.show()
            plt.close()


        if sim_method == 'numerical':

            # Load synchronized maps.
            healpix_map, DM_healpix_map, end_str, *_ = self.syncronize_all_sky_maps(halo, nu_mass_eV, apply=True)

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

            if halo == 0:
                # Create original colormap
                cmap = plt.get_cmap('coolwarm')
                colors = cmap(np.linspace(0, 1, cmap.N))

                # Split colormap
                three_quarter_point = int(0.5 * cmap.N)
                cmap_right = mcolors.LinearSegmentedColormap.from_list(
                    'cmap_right', colors[three_quarter_point:], N=256)

                # All-sky map in mollview projection.
                hp.newvisufunc.projview(
                    healpix_map, unit=r'$n_{\nu, pix} / n_{\nu, pix, 0}$',
                    cmap='coolwarm',
                    # cmap=cmap_right,
                    override_plot_properties={
                        "cbar_pad": 0.1,
                        # "cbar_label_pad": 15,
                    },
                    sub=121, 
                    # norm=divnorm,
                    **self.healpy_dict)
                
            else:
                # Make center value to be 1 (no overdensity).
                mid = 1.
                divnorm = mcolors.TwoSlopeNorm(vcenter=mid)

                # All-sky map in mollview projection.
                hp.newvisufunc.projview(
                    healpix_map, unit=r'$n_{\nu, pix} / n_{\nu, pix, 0}$',
                    cmap='coolwarm',
                    override_plot_properties={
                        "cbar_pad": 0.1},
                    cbar_ticks=[
                        np.min(healpix_map), mid, np.max(healpix_map)],
                    sub=121, 
                    norm=divnorm,
                    **self.healpy_dict)


            ### ---------------------------------- ###
            ### Plot DM line-of-sight all_sky map. ###
            ### ---------------------------------- ###

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


    def plot_all_sky_power_spectra(self, halo, nu_mass_eV):

        # Load synchronized maps.
        healpix_map, DM_healpix_map, end_str = self.syncronize_all_sky_maps(
            halo, nu_mass_eV
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
    

    def plot_all_spectra_1plot(self, halo_array, nu_mass_eV):

        fig = plt.figure(figsize =(12, 4))
        fig.tight_layout()

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Unit to compare to similar figures in literature.
        micro_Kelvin_unit = 1e12

        eta_mins = np.zeros(len(halo_array))
        eta_maxs = np.zeros(len(halo_array))
        factors = np.zeros(len(halo_array))
        power_spectra = []
        cross_spectra = []
        for i, halo in enumerate(halo_array):

            # Load synchronized maps.
            healpix_map, DM_healpix_map, _, eta_min, eta_max, factor = self.syncronize_all_sky_maps(
                halo, nu_mass_eV, apply=False
            )
            eta_mins[i] = eta_min
            eta_maxs[i] = eta_max
            factors[i] = factor

            # Convert maps of neutrino densities to temperatures.
            temperature_map = np.cbrt((healpix_map*2*Pi**2)/(3*zeta(3)))

            # Compute power spectrum of number density all-sky map.
            cl = hp.sphtfunc.anafast(temperature_map)
            ell = np.arange(len(cl))
            power_spectrum = ell*(ell+1)*cl/(2*Pi)*micro_Kelvin_unit
            power_spectra.append(power_spectrum)

            # Compute cross-correlation spectrum of n_nu and DM maps.
            cross = hp.sphtfunc.anafast(temperature_map, DM_healpix_map)
            cross_spectrum = ell*(ell+1)*cross/(2*Pi)*micro_Kelvin_unit
            cross_spectra.append(cross_spectrum)

            # Pearsons correlation coefficient for total map information.
            # pearson_r, _ = pearsonr(healpix_map, DM_healpix_map)

            # ax1.semilogy(ell, power_spectrum)
            # ax2.plot(ell, cross_spectrum)

        # '''
        power_spectra = np.array(power_spectra)
        power_median = np.median(power_spectra, axis=0)
        power_perc2p5 = np.percentile(power_spectra, q=2.5, axis=0)
        power_perc97p5 = np.percentile(power_spectra, q=97.5, axis=0)
        power_perc16 = np.percentile(power_spectra, q=16, axis=0)
        power_perc84 = np.percentile(power_spectra, q=84, axis=0)

        cross_spectra = np.array(cross_spectra)
        cross_median = np.median(cross_spectra, axis=0)
        cross_perc2p5 = np.percentile(cross_spectra, q=2.5, axis=0)
        cross_perc97p5 = np.percentile(cross_spectra, q=97.5, axis=0)
        cross_perc16 = np.percentile(cross_spectra, q=16, axis=0)
        cross_perc84 = np.percentile(cross_spectra, q=84, axis=0)

        ax1.semilogy(
            ell, power_median, color='blue', 
            label='Power spectra median')
        ax1.fill_between(
            ell, power_perc2p5, power_perc97p5, 
            color='blue', alpha=0.2, label='2.5-97.5 % C.L.')
        ax1.fill_between(
            ell, power_perc16, power_perc84, 
            color='blue', alpha=0.3, label='16-84 % C.L.')
        
        ax2.plot(
            ell, cross_median, color='blue', 
            label='Cross spectra median')
        ax2.fill_between(
            ell, cross_perc2p5, cross_perc97p5, 
            color='blue', alpha=0.2, label='2.5-97.5 % C.L.')
        ax2.fill_between(
            ell, cross_perc16, cross_perc84, 
            color='blue', alpha=0.3, label='16-84 % C.L.')      
        # '''

        Tully_x_axis = read_column_from_file(
            f'{self.sim_dir}/all_sky_power_spectra_Tully.csv', 'x_axis')
        Tully_y_axis = read_column_from_file(
            f'{self.sim_dir}/all_sky_power_spectra_Tully.csv', 'y_axis')

        ax1.semilogy(
            Tully_x_axis, Tully_y_axis, color='magenta', label='Tully+(2021)')

        ax1.set_xlabel("$\ell$")
        ax1.set_xlim(1,np.max(ell))
        # ax1.set_ylabel("$\ell(\ell+1)C_{\ell} [\mu K^2]$")
        ax1.grid()

        ax2.set_xlabel("$\ell$")
        # ax2.set_ylabel("$\ell(\ell+1)C_{\ell} [\mu K^2]$")
        ax2.grid()

        fig.text(
            0.04, 0.5, "$\ell(\ell+1)C_{\ell} [\mu K^2]$", 
            va='center', rotation='vertical')

        plt.savefig(
            f'{self.fig_dir}/all_power_spectra_{nu_mass_eV}eV.pdf', 
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


    #? use for alt determination of clustered neutrinos?
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

        plt.savefig(
            f'{self.fig_dir}/neutrinos_inside_Rvir.pdf', bbox_inches='tight')
        plt.close()


    def plot_phase_space(self, most_likely:bool):

        # Load necessary box and sim info.
        with open(f'{self.sim_dir}/sim_parameters.yaml', 'r') as file:
            sim_setup = yaml.safe_load(file)

        p_num = sim_setup['momentum_num']
        p_start = sim_setup['momentum_start']
        p_stop = sim_setup['momentum_stop']
        phis = sim_setup['phis']
        thetas = sim_setup['thetas']

        with open(f'{self.sim_dir}/box_parameters.yaml', 'r') as file:
            box_setup = yaml.safe_load(file)

        # Box Cosmology.
        box_H0 =  box_setup['Cosmology']['h']*100*km/s/Mpc
        box_Omega_M = box_setup['Cosmology']['Omega_M']
        box_Omega_L = box_setup['Cosmology']['Omega_L']


        ### ---------------- ###
        ### Analytical halo. ###
        ### ---------------- ###

        if 'analytical_halo' in self.objects:

            # Convert velocities to mementa.
            vels_in = self.vectors_analytical[...,3:6]
            p_arr, y_arr = velocity_to_momentum(vels_in, self.mpicks)
            p0_arr, p1_arr, y0_arr = p_arr[...,0], p_arr[...,-1], y_arr[...,0]

            # Sort.
            ind = p0_arr.argsort(axis=-1)
            p1_sort = np.take_along_axis(p1_arr, ind, axis=-1)
            y0_sort = np.take_along_axis(y0_arr, ind, axis=-1)

            if most_likely:
                # Each velocity has a batch of neutrinos.
                # (min. of each to represent most (likely) clustered ones)
                m_len = (len(self.mpicks))
                p1_blocks = p1_sort.reshape((m_len, p_num, phis*thetas))
                p1_final = np.min(p1_blocks, axis=-1)
                y0_blocks = y0_sort.reshape((m_len, p_num, phis*thetas))
                y0_final = y0_blocks[...,0]
            else:
                p1_final = p1_sort
                y0_final = y0_sort

            # Fermi Dirac of the final momenta.
            FDvals = Fermi_Dirac(p1_final)

            fig, axs = plt.subplots(2,2, figsize=(12,12))
            fig.suptitle(
                'Phase-space distr. "today" compared to Fermi-Dirac' ,
                fontsize=18
            )

            savefig_args = dict(
                bbox_inches='tight'
            )

            for j, m_nu in enumerate(self.mpicks):

                k = j
                i = 0
                if j in (2,3):
                    i = 1
                    j -= 2

                # Simulation phase-space distr. of neutrinos today.
                axs[i,j].loglog(
                    y0_final[k], FDvals[k], label='PS today (from sim)', c='red', alpha=0.9
                )

                # Fermi-Dirac phase-space distr.
                pOG = np.geomspace(
                    p_start*T_CNB, p_stop*T_CNB, FDvals.shape[-1])
                FDvalsOG = Fermi_Dirac(pOG)
                yOG = pOG/T_CNB
                axs[i,j].loglog(yOG, FDvalsOG, label='PS Fermi-Dirac', c='black', alpha=0.7)

                # Escape momentum.
                _, y_esc = escape_momentum(
                    x_i=self.init_xyz, z=0., 
                    R_vir=Rvir_MW, R_s=Rs_MW, rho_0=rho0_MW, m_nu_eV=m_nu
                )
                axs[i,j].axvline(y_esc, c='k', ls='-.', label='y_esc')

                # Plot styling.
                axs[i,j].set_title(f'{m_nu} eV')
                axs[i,j].set_ylabel('FD(p)')
                axs[i,j].set_xlabel(r'$y = p / T_{\nu,0}$')
                axs[i,j].legend(loc='lower left')
                axs[i,j].set_ylim(1e-5, 1e0)
                axs[i,j].set_xlim(p_start, 1e2)

            plt.savefig(
                f'{self.fig_dir}/phase_space_analytical.pdf', **savefig_args)
            plt.close()


        ### ---------------------- ###
        ### Box (numerical) halos. ###
        ### ---------------------- ###

        if 'box_halos' in self.objects:

            box_halos_p1_final = []
            box_halos_y0_final = []
            for halo_j in range(self.halo_num):

                vels_batches = self.vectors_numerical[halo_j,...,3:6]
                vels_in = vels_batches.reshape(-1, 100, 3)

                # Convert velocities to momenta.
                p_arr, y_arr = velocity_to_momentum(vels_in, self.mpicks)
                p0_arr = p_arr[...,0]
                p1_arr = p_arr[...,-1]
                y0_arr = y_arr[...,0]

                # Sort.
                ind = p0_arr.argsort(axis=-1)
                p1_sort = np.take_along_axis(p1_arr, ind, axis=-1)
                y0_sort = np.take_along_axis(y0_arr, ind, axis=-1)

                if most_likely:
                    # Each velocity has a batch of neutrinos.
                    # (min. of each to represent most (likely) clustered ones)
                    m_len = (len(self.mpicks))
                    p1_blocks = p1_sort.reshape((m_len, p_num, phis*thetas))
                    p1_final = np.min(p1_blocks, axis=-1)
                    y0_blocks = y0_sort.reshape((m_len, p_num, phis*thetas))
                    y0_final = y0_blocks[...,0]
                else:
                    p1_final = p1_sort
                    y0_final = y0_sort

                box_halos_p1_final.append(p1_final)
                box_halos_y0_final.append(y0_final)

            box_halos_p1_final = np.array(box_halos_p1_final)
            box_halos_y0_final = np.array(box_halos_y0_final)

            p1_median = np.median(box_halos_p1_final, axis=0)
            p1_perc2p5 = np.percentile(box_halos_p1_final, q=2.5, axis=0)
            p1_perc97p5 = np.percentile(box_halos_p1_final, q=97.5, axis=0)
            p1_perc16 = np.percentile(box_halos_p1_final, q=16, axis=0)
            p1_perc84 = np.percentile(box_halos_p1_final, q=84, axis=0)

            y0_median = np.median(box_halos_y0_final, axis=0)
            y0_perc2p5 = np.percentile(box_halos_y0_final, q=2.5, axis=0)
            y0_perc97p5 = np.percentile(box_halos_y0_final, q=97.5, axis=0)
            y0_perc16 = np.percentile(box_halos_y0_final, q=16, axis=0)
            y0_perc84 = np.percentile(box_halos_y0_final, q=84, axis=0)

            # Fermi Dirac of the final momenta.
            FDvals_median = Fermi_Dirac(p1_median)
            FDvals_perc2p5 = Fermi_Dirac(p1_perc2p5)
            FDvals_perc97p5 = Fermi_Dirac(p1_perc97p5)
            FDvals_perc16 = Fermi_Dirac(p1_perc16)
            FDvals_perc84 = Fermi_Dirac(p1_perc84)

            fig, axs = plt.subplots(2,2, figsize=(12,12))
            # fig.suptitle(
            #     'Phase-space distr. "today" compared to Fermi-Dirac\n Box (numerical) halos',
            #     fontsize=18
            # )

            percentages_med = np.zeros(len(self.mpicks))
            percentages_max = np.zeros(len(self.mpicks))
            for j, m_nu in enumerate(self.mpicks):

                k = j  # k selects the element(s) for the current neutrino mass

                # Indices for axes of 2x2 subplots.
                i = 0
                if j in (2,3):
                    i = 1
                    j -= 2

                individual = False

                if individual:
                    for xs, ys in zip(box_halos_y0_final, box_halos_p1_final):
                        FDs = Fermi_Dirac(ys)

                        axs[i,j].plot(
                            xs[k], FDs[k], c='dodgerblue', alpha=0.8
                        )

                else:
                    # Simulation phase-space distr. of neutrinos today.
                    axs[i,j].plot(
                        y0_median[k], FDvals_median[k], label='PS today (from sim)', c='blue', alpha=0.9)

                    axs[i,j].fill_between(
                        y0_median[k], FDvals_perc2p5[k], FDvals_perc97p5[k],
                        color='blue', alpha=0.2, 
                        label='Box Halos: 2.5-97.5 % C.L.')
                    
                    axs[i,j].fill_between(
                        y0_median[k], FDvals_perc16[k], FDvals_perc84[k],
                        color='blue', alpha=0.3, 
                        label='Box Halos: 16-84 % C.L.')
                
                # Fermi-Dirac phase-space distr.
                pOG = np.geomspace(
                    p_start*T_CNB, p_stop*T_CNB, FDvals_median.shape[-1])
                FDvalsOG = Fermi_Dirac(pOG)
                yOG = pOG/T_CNB
                axs[i,j].plot(
                    yOG, FDvalsOG, 
                    label='PS Fermi-Dirac', c='black', alpha=0.7)

                # Escape momentum for each box halo.
                Rvir, Mvir, conc = self.get_halo_params()
                Rs = Rvir/conc
                rho0 = scale_density_NFW(
                    c=conc, z=0.,
                    H0=box_H0, Omega_M=box_Omega_M, Omega_L=box_Omega_L)

                _, y_esc = escape_momentum(
                    x_i=self.init_xyz*kpc, 
                    R_vir=Rvir*kpc, R_s=Rs*kpc, rho_0=rho0, 
                    m_nu_eV=m_nu)

                y_esc_med = np.median(y_esc, axis=0)
                y_esc_perc2p5 = np.percentile(y_esc, q=2.5, axis=0)
                y_esc_perc97p5 = np.percentile(y_esc, q=97.5, axis=0)
                y_esc_perc16 = np.percentile(y_esc, q=16, axis=0)
                y_esc_perc84 = np.percentile(y_esc, q=84, axis=0)

                axs[i,j].axvline(
                    y_esc_med, c='black', ls='-.', label='y_esc median')
                
                y_min, y_max = axs[i, j].get_ylim()

                axs[i, j].fill_betweenx(
                    [y_min, y_max], x1=y_esc_perc2p5, x2=y_esc_perc97p5, 
                    color='black', alpha=0.2, 
                    label='y_esc: 2.5-97.5 % C.L.')
                
                axs[i, j].fill_betweenx(
                    [y_min, y_max], x1=y_esc_perc16, x2=y_esc_perc84, 
                    color='black', alpha=0.3, 
                    label='y_esc: 16-84 % C.L.')

                # Plot styling.
                axs[i,j].set_title(f'{m_nu} eV')
                axs[i,j].set_ylabel('FD(p)')
                axs[i,j].set_xlabel(r'$y = p / T_{\nu,0}$')

                log_scale = True

                if log_scale:
                    lin_or_log = 'log'
                    axs[i,j].set_ylim(1e-3, 1e0)
                    axs[i,j].set_xlim(p_start, 1e2)
                    axs[i,j].set_xscale('log')
                    axs[i,j].set_yscale('log')
                else:
                    lin_or_log = 'linear'
                    x_max_lims = [0.5, 1.5, 2.5, 7]
                    axs[i,j].set_xlim(np.min(y0_median[k]), x_max_lims[k])
                    

                ### --------------------------------------- ###
                ### Excourse: Percentages of distributions. ###
                ### --------------------------------------- ###

                ### Normal Fermi-Dirac distribution.

                # Total area (for normalization).
                orig_mom = np.geomspace(0.001*T_CNB, 1000*T_CNB, 100_000)
                FD_orig = Fermi_Dirac(orig_mom)
                FD1_norm = np.trapz(
                    orig_mom**3*FD_orig, x=np.log(orig_mom))
                
                # Percentage covered withing momentum interval.
                p_interval = np.geomspace(0.01*T_CNB, y_esc_med*T_CNB, 10_000)
                FD_interval = Fermi_Dirac(p_interval)
                perc_interval = np.trapz(
                    p_interval**3*FD_interval, x=np.log(p_interval))/FD1_norm
                ic(np.round(perc_interval*100, 2))


                ### "Distorted" Fermi-Dirac distribution due to clustering.
                
                # Total area (for normalization).
                all_mom = y0_median[k]*T_CNB
                FD2_norm = np.trapz(
                    all_mom**3*FDvals_median[k], x=np.log(all_mom))

                #! (visually) find momentum threshold of median curve.
                esc_cond_med = np.array([0.15, 0.7, 1.45, 3.92])
                axs[i,j].axvline(
                    esc_cond_med[k], 
                    c='green', ls='-.', label='esc_p median')
                
                #! (visually) find momentum threshold of max curve.
                esc_cond_max = np.array([0.2, 1, 2, 5])
                axs[i,j].axvline(
                    esc_cond_max[k], 
                    c='red', ls='-.', label='esc_p max')

                # Clustered neutrino percentage using median curve.
                below_esc_med = (y0_median[k] <= esc_cond_med[k])
                # below_esc = (y0_median[k] <= y_esc_med)
                x_interval = y0_median[k][below_esc_med]*T_CNB
                y_interval = FDvals_median[k][below_esc_med]
                perc_esc = np.trapz(
                    x_interval**3*y_interval, x=np.log(x_interval))/FD2_norm
                percentages_med[k] = np.round(perc_esc*100, 2)
                percentages_med[k] = np.round(perc_esc*100, 2)
                ic(np.round(perc_esc*100, 2))

                # Clustered neutrino percentage using max curve.
                below_esc_max = (y0_median[k] <= esc_cond_max[k])
                # below_esc = (y0_median[k] <= y_esc_med)
                x_interval = y0_median[k][below_esc_max]*T_CNB
                y_interval = FDvals_median[k][below_esc_max]
                perc_esc = np.trapz(
                    x_interval**3*y_interval, x=np.log(x_interval))/FD2_norm
                percentages_max[k] = np.round(perc_esc*100, 2)
                ic(np.round(perc_esc*100, 2))

                if m_nu == self.mpicks[-1]:
                    axs[i,j].legend(
                        loc='lower left', borderpad=0.25, labelspacing=0.25, 
                        fontsize='small')

            np.savetxt(
                f'clustered_neutrinos_percentages_med.txt', percentages_med)
            np.savetxt(
                f'clustered_neutrinos_percentages_max.txt', percentages_max)

            plt.savefig(
                f'{self.fig_dir}/phase_space_numerical_{lin_or_log}.pdf', 
                bbox_inches='tight')
            plt.close()


    def plot_eta_vs_halo_params(self):
        
        Rvir, Mvir, conc = self.get_halo_params()

        # Get number densities and convert to clustering factors.
        etas_nu = np.flip(self.etas_numerical, axis=-1)
        
        # Show relative change to median for each mass.
        etas_0 = np.median(etas_nu, axis=0)
        etas_nu = (etas_nu-etas_0)/etas_0

        fig = plt.figure(figsize=(15, 5))
        cmap = np.flip(
            np.array(mcp.gen_color(cmap='RdYlBu', n=etas_nu.shape[-1])))
        size = 5

        # Virial mass.
        ax1 = fig.add_subplot(131)
        for etas, color in zip(etas_nu[Mvir.argsort()].T, cmap):
            ax1.scatter(
                Mvir[Mvir.argsort()], 
                etas, c=color, s=size)
        ax1.set_xlabel(r'$M_{vir}$')
        ax1.set_ylabel(r'$(f-\bar{f})/\bar{f}$')
        # ax1.set_yscale('symlog', linthresh=0.0001)

        # Add abscissa y-axis to show virial radius (without data).
        ax1b = ax1.twiny()
        ax1b.scatter(
            Rvir[Rvir.argsort()], 
            etas, 
            alpha=0)  # Set transparency to 100% (i.e., invisible)
        ax1b.set_xlabel(r'$R_{vir}$')

        # Concentration.
        ax2 = fig.add_subplot(132, sharey=ax1)
        for etas, color in zip(etas_nu[conc.argsort()].T, cmap):
            ax2.scatter(
                conc[conc.argsort()],
                etas, c=color, s=size)
        ax2.set_xlabel('concentration')

        # Initial distance.
        ax3 = fig.add_subplot(133, sharey=ax1)
        for etas, color in zip(etas_nu[self.init_dis.argsort()].T, cmap):
            ax3.scatter(
                self.init_dis[self.init_dis.argsort()],
                etas, c=color, s=size)
        ax3.set_xlabel('initial distance')

        # Customized colorbar.
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        norm = mpl.colors.Normalize(vmin=0.01, vmax=0.3)
        cmap = mpl.cm.RdYlBu
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation='vertical', 
            label='$m_\nu$ [eV]',
            # pad=0.05
        )

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
        
        # Get halo parameters.
        Rvir, Mvir, conc = self.get_halo_params()

        # Closest mass (index) for chosen neutrino mass.
        nu_mass_idx = (np.abs(self.mrange-nu_mass_eV)).argmin()

        # Get clustering factors.
        etas_nu = self.etas_numerical[...,nu_mass_idx].flatten()

        fig = plt.figure(figsize=(11, 5))
        cmap_plot = np.array(mcp.gen_color(cmap='coolwarm', n=len(etas_nu)))
        size = 30

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, sharey=ax1)

        # Sort arrays by (relative) clustering factors.
        plot_ind = etas_nu.argsort()
        Mvir_plot = Mvir[plot_ind]
        conc_plot = conc[plot_ind]
        init_plot = self.init_dis[plot_ind]

        ax1.scatter(conc_plot, Mvir_plot, c=cmap_plot, s=size)
        ax1.set_ylabel(r'$M_{vir}$')
        ax1.set_xlabel('concentration')
        ax1.grid()
        
        ax2.scatter(init_plot, Mvir_plot, c=cmap_plot, s=size)
        ax2.set_xlabel('initial distance (kpc)')
        ax2.grid()

        # Customized colorbar.
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        norm = mpl.colors.Normalize(vmin=np.min(etas_nu), vmax=np.max(etas_nu))
        cmap = mpl.cm.coolwarm
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation='vertical', 
            label=r'$n_\nu / n_{\nu,0}$',
            # pad=0.05
        )
    
        plt.savefig(
            f'{self.fig_dir}/2D_params_box_halos.pdf', bbox_inches='tight')


    def plot_density_profiles(self, NFW_orig=False):

        fig, ax = plt.subplots(1,1)

        ### ============= ###
        ### Plot objects. ###
        ### ============= ###

        # Box and sim parameters.
        with open(f'{self.sim_dir}/box_parameters.yaml', 'r') as file:
            box_setup = yaml.safe_load(file)

        DM_mass = box_setup['Content']['DM Mass [Msun]']*Msun
        nums_snaps = np.load(f'{self.sim_dir}/nums_snaps.npy')

        # File Paths.
        box_file_dir = box_setup['File Paths']['Box File Directory']
        sim_parent = str(pathlib.Path(self.sim_dir).parent)
        benchmark_file_dir = f'{sim_parent}/benchmark_halo_files'

        # Box Cosmology.
        box_H0 =  box_setup['Cosmology']['h']*100*km/s/Mpc
        box_Omega_M = box_setup['Cosmology']['Omega_M']
        box_Omega_L = box_setup['Cosmology']['Omega_L']

        # Bin centers in kpc for plot.
        radial_bins = 10**np.arange(0, 5, 0.1)
        centers = bin_centers(radial_bins)

        if 'NFW_halo' in self.objects:

            NFW_halo_coords = np.load(
                f'{benchmark_file_dir}/benchmark_halo_snap_{nums_snaps[-1]}.npy'
            )

            # Calculate density for each bin center.
            NFW_halo_densities = analyse_halo(DM_mass/Msun, NFW_halo_coords)
            
            ax.plot(
                centers, NFW_halo_densities, 
                    c='green', label=f'NFW Halo'
            )

        if 'box_halos' in self.objects:

            densities = []
            for halo_ID in self.halo_indices:

                # Read the DM positions of all halos at redshift z=0.
                DM_coords = np.load(glob.glob(
                    f'{sim_parent}/final_halo_data/DM_pos_origID{halo_ID}_snap_0036.npy')[0]
                )

                # Calculate density for each bin center.
                densities.append(analyse_halo(DM_mass/Msun, DM_coords))

            densities = np.array(densities)
            densities_median = np.median(densities, axis=0)
            densities_perc2p5 = np.percentile(densities, q=2.5, axis=0)
            densities_perc97p5 = np.percentile(densities, q=97.5, axis=0)
            densities_perc16 = np.percentile(densities, q=16, axis=0)
            densities_perc84 = np.percentile(densities, q=84, axis=0)

            ax.plot(
                centers, densities_median, 
                c='blue', label=f'Box Halos Median'
            )
            ax.fill_between(
                centers, densities_perc2p5, densities_perc97p5,
                color='blue', alpha=0.2, 
                label='Box Halos: 2.5-97.5 % C.L.'
            )
            ax.fill_between(
                centers, densities_perc16, densities_perc84,
                color='blue', alpha=0.3, 
                label='Box Halos: 16-84 % C.L.'
            )


        if 'analytical_halo' in self.objects:

            NFW = NFW_profile(
                centers*kpc, rho0_MW, Rs_MW
            )
            ax.plot(
                centers, NFW/(Msun/kpc**3), 
                c='red', ls='-.', label='Analytical halo NFW'
            )


        if NFW_orig:

            Rvir, Mvir, conc = self.get_halo_params()
            Rvir_med = np.median(Rvir)
            Mvir_med = np.median(Mvir)
            conc_med = np.median(conc)

            # NFW density profile with parameters of sampled halo.
            Rs_med = Rvir_med/conc_med
            rho_NFW = scale_density_NFW(
                c=conc_med, z=0.,
                 H0=box_H0, Omega_M=box_Omega_M, Omega_L=box_Omega_L)
            NFW = NFW_profile(
                centers*kpc, rho_NFW, Rs_med*kpc)
            ax.plot(
                centers, NFW/(Msun/kpc**3), 
                c='black', ls='-.', label='Box halos median NFW')

        # ax.set_title('Density Profiles')
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel(r'$\rho$ [$M_{\odot}/\mathrm{kpc}^3$]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e0, 1e3)
        ax.set_ylim(5*1e2, 5*1e8)

        plt.legend(loc='lower left')
        plt.savefig(f'{self.fig_dir}/density_profiles.pdf', bbox_inches='tight')
        plt.close()


    # note: nothing interesting
    def plot_factor_vs_halo_params(self):

        # Load halo parameters.
        Rvir, Mvir, conc = self.get_halo_params()

        # Load factors.
        factors = np.array(read_column_from_file(
            f'{self.sim_dir}/all_sky_values_original.csv', 'Factor'))
        # factors = np.delete(factors, factors>=300)

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        size = 30

        sct1 = ax1.scatter(factors, Mvir, s=size)
        ax1.set_ylabel(r'$M_{vir}$')
        ax1.set_xlabel('factor')
        
        sct2 = ax2.scatter(factors, conc, s=size)
        ax2.set_ylabel('concentration')
        ax2.set_xlabel('factor')

        ax1.set_xlim(0, 100)
        ax2.set_xlim(0, 100)

        ax1.grid()
        ax2.grid()
        plt.savefig(
            f'{self.fig_dir}/factor_vs_halo_params.pdf', 
            bbox_inches='tight')


    def electron_flavor_number_densities(self, m_lightest, ordering):


        def nu_masses(m_lightest, ordering):
            """Returns the 3 neutrino masses with given ordering and lightest mass."""
            if ordering == 'NO':
                m1 = m_lightest
                m2 = np.sqrt(m1**2 + Del_m21_Sq)
                m3 = np.sqrt(m1**2 + Del_m3l_Sq)
                return np.array([m1, m2, m3])
            if ordering == 'IO':
                m3 = m_lightest
                m2 = np.sqrt(m3**2 + Del_m3l_Sq)
                m1 = np.sqrt(m2**2 - Del_m21_Sq)
                return np.array([m3, m1, m2])

        
        # 1. Pick m_lightest and ordering -> establish m_nu1, m_nu2, m_nu3.
        masses_eV = nu_masses(m_lightest, ordering)
        m_min, m_med, m_max = masses_eV[0], masses_eV[1], masses_eV[2]

        # 2. Read the numberdensities for these masses -> n_nu1, n_nu2, n_nu3.
        ns_nu_masses = np.zeros(3)
        for i, m_nu_eV in enumerate(masses_eV):
            
            # Get mass index.
            nu_mass_idx = (np.abs(self.mrange-m_nu_eV)).argmin()

            # Get number densities (median).
            ns_nu_masses[i] = np.median(
                self.etas_numerical[...,nu_mass_idx].flatten()*N0)


        # 3. Convert to n_nu_e.
        if ordering == 'NO':
            n_nu_e_flavor = np.sum(U_ei_AbsSq*ns_nu_masses/2)
        if ordering == 'IO':
            n_nu_e_flavor = np.sum(np.roll(U_ei_AbsSq, 1)*ns_nu_masses/2)

        ic(n_nu_e_flavor)
        ic(n_nu_e_flavor/(N0/2))


# ======================== #
# '''
sim_dir = f'L025N752/DMONLY/SigmaConstant00/all_sky_final'

objects = (
    'NFW_halo', 
    # 'box_halos', 
    # 'analytical_halo'
)
Analysis = analyze_simulation_outputs(
    sim_dir = sim_dir, 
    objects = objects,
    sim_type = 'all_sky',
)

print(Analysis.final_halos)
print(Analysis.halo_num)
halo_array = np.arange(Analysis.halo_num)+1

### Generate power spectra plots.
# Analysis.plot_all_spectra_1plot(halo_array, 0.3)

### Generate all all-sky anisotropy maps.

# For numerical:
# for halo in halo_array:
#     Analysis.plot_all_sky_map('numerical', halo, 0.3)

# For analytical:
# Analysis.plot_all_sky_map('analytical', 0, 0)

# For benchmark NFW:
Analysis.plot_all_sky_map('numerical', 0, 0.3)

### Generate correlation plots.
# Analysis.plot_factor_vs_halo_params()
# '''
# ======================== #



# ======================== #
'''
sim_dir = f'L025N752/DMONLY/SigmaConstant00/single_halos'

objects = (
    'NFW_halo', 
    'box_halos', 
    'analytical_halo'
)
Analysis = analyze_simulation_outputs(
    sim_dir = sim_dir, 
    objects = objects,
    sim_type = 'single_halos',
)

### Generate correlation plots.
# Analysis.plot_2d_params(nu_mass_eV=0.3)
# Analysis.plot_eta_vs_halo_params()

# Analysis.plot_overdensity_band(plot_ylims=None)

# Analysis.plot_phase_space(most_likely=True)

Analysis.plot_density_profiles(NFW_orig=True)

# Analysis.plot_overdensity_evolution(plot_ylims=(1e-4,1e1))

# Analysis.electron_flavor_number_densities(m_lightest=0.1*eV, ordering='IO')
# Analysis.electron_flavor_number_densities(m_lightest=0.05*eV, ordering='NO')
# Analysis.electron_flavor_number_densities(m_lightest=0.1*eV, ordering='NO')
# '''
# ======================== #