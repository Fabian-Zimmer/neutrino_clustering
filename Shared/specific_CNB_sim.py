from Shared.shared import *

import psutil
import h5py
from itertools import chain, zip_longest, repeat
from funcy import chunks
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors
import jax.ops


class SimData():

    def __init__(self, sim_dir):
        self.sim_dir = sim_dir
        self.load_parameters()


    def load_parameters(self):
        with open(f'{self.sim_dir}/sim_parameters.yaml', 'r') as file:
            sim_setup = yaml.safe_load(file)

        self.CPUs_pre = sim_setup['CPUs_precalculations']
        self.CPUs_sim = sim_setup['CPUs_simulations']
        self.DM_in_cell_limit = sim_setup['DM_in_cell_limit']
        self.Npix = sim_setup['Npix']
        self.Nside = sim_setup['Nside']
        self.init_haloGC_dist = sim_setup['initial_haloGC_distance']
        self.mem_limit_GB = sim_setup['memory_limit_GB']
        self.p_num = sim_setup['momentum_num']
        self.p_start = sim_setup['momentum_start']
        self.p_stop = sim_setup['momentum_stop']
        self.nu_mass_num = sim_setup['neutrino_mass_num']
        self.nu_mass_start = sim_setup['neutrino_mass_start']
        self.nu_mass_stop = sim_setup['neutrino_mass_stop']
        self.nu_sim_mass_eV = sim_setup['neutrino_simulation_mass_eV']
        self.nu_sim_mass_kg = sim_setup['neutrino_simulation_mass_kg']
        self.nus_in_sim = sim_setup['neutrinos']
        self.phis = sim_setup['phis']
        self.pix_sr = sim_setup['pix_sr']
        self.thetas = sim_setup['thetas']
        self.z_int_num = sim_setup['z_inegration_num']
        self.z_int_start = sim_setup['z_inegration_start']
        self.z_int_stop = sim_setup['z_inegration_stop']

        with open(f'{self.sim_dir}/box_parameters.yaml', 'r') as file:
            sim_setup = yaml.safe_load(file)

        self.DM_mass_Msun = sim_setup['Content']['DM Mass [Msun]']
        self.smoothening_length_pc = sim_setup['Content']['Smoothening Length [pc]']
        self.z0_sim_val = sim_setup['Content']['initial redshift']
        self.z4_sim_val = sim_setup['Content']['final redshift']
        self.z0_snap_num = sim_setup['Content']['z=0 snapshot']
        self.z4_snap_num = sim_setup['Content']['z=4 snapshot']
        self.Omega_L = sim_setup['Cosmology']['Omega_L']
        self.Omega_M = sim_setup['Cosmology']['Omega_M']
        self.Omega_R = sim_setup['Cosmology']['Omega_R']
        self.h = sim_setup['Cosmology']['h']
        self.box_dir = sim_setup['File Paths']['Box File Directory']
        self.out_dir = sim_setup['File Paths']['Output Directory']


    @staticmethod
    def load_vectors_OLD(sim_dir, halo_num):

        halo_vecs_l = []
        for i in range(halo_num):
            halo_vecs_l.append(jnp.load(f"{sim_dir}/vectors_halo{i+1}.npy"))

        return jnp.array(halo_vecs_l)


    @staticmethod
    def load_vectors(sim_dir, halo_num):
        halo_vels_l = []
        for i in range(halo_num):
            file_path = f"{sim_dir}/vectors_halo{i+1}.npy"
            if os.path.exists(file_path):
                try:
                    halo_vecs = jnp.load(file_path)
                    halo_vels_l.append(halo_vecs)
                except:
                    pass  # Silently skip any loading errors
        
        return jnp.array(halo_vels_l)


    @staticmethod
    def load_velocities(sim_dir, halo_num):
        halo_vels_l = []
        for i in range(halo_num):
            file_path = f"{sim_dir}/vectors_halo{i+1}.npy"
            if os.path.exists(file_path):
                try:
                    halo_vecs = jnp.load(file_path)
                    halo_vels_l.append(halo_vecs[..., 3:])
                except:
                    pass  # Silently skip any loading errors
        
        return jnp.array(halo_vels_l)

    @staticmethod
    def load_boosted_velocities(sim_dir, halo_num, end_str):
        halo_vels_l = []
        for i in range(halo_num):
            file_path = f"{sim_dir}/vectors_halo{i+1}_{end_str}.npy"
            if os.path.exists(file_path):
                try:
                    halo_vecs = jnp.load(file_path)
                    halo_vels_l.append(halo_vecs[..., 3:])
                except:
                    pass  # Silently skip any loading errors
        
        return jnp.array(halo_vels_l)



    @staticmethod
    def load_positions(sim_dir, halo_num):
        halo_pos_l = []
        for i in range(halo_num):
            file_path = f"{sim_dir}/vectors_halo{i+1}.npy"
            if os.path.exists(file_path):
                try:
                    halo_vecs = jnp.load(file_path)
                    halo_pos_l.append(halo_vecs[..., :3])
                except:
                    pass  # Silently skip any loading errors
        
        return jnp.array(halo_pos_l)


    @staticmethod
    def halo_batch_indices(
        snap, mass_gauge, mass_lower, mass_upper, 
        halo_type, halo_limit, fname, sim_dir, out_dir
    ):

        # ---------------------------------- #
        # Read in parameters of halo in sim. #
        # ---------------------------------- #

        props = h5py.File(f'{sim_dir}/subhalo_{snap}.properties', 'r')

        cNFW = props['cNFW_200crit'][:]  # NFW concentration.
        rvir = props['R_200crit'][:]*1e3 # Virial radius (to kpc with *1e3)
        m200c = props['Mass_200crit'][:]*1e10  # Crit. M_200 (to Msun with *1e10)
        m200c[m200c <= 0] = 1
        m200c = np.log10(m200c)  


        # -------------------------------------------------- #
        # Select a halo sample based on mass and mass range. #
        # -------------------------------------------------- #

        select_halos = np.where(
            (m200c >= mass_gauge-mass_lower) & (m200c <= mass_gauge+mass_upper)
        )[0]

        # Selecting subhalos or (host/main) halos.
        subtype = props["Structuretype"][:]
        if halo_type == 'subhalos':
            select = np.where(subtype[select_halos] > 10)[0]
            select_halos = select_halos[select]
        else:
            select = np.where(subtype[select_halos] == 10)[0]
            select_halos = select_halos[select]

        # Limit amount of halos to given size.
        halo_number = len(select_halos)

        # Save cNFW, rvir and Mvir of halos in batch.
        halo_params = np.zeros((len(select_halos), 3))
        for j, halo_idx in enumerate(select_halos):
            halo_params[j, 0] = rvir[halo_idx]
            halo_params[j, 1] = m200c[halo_idx]
            halo_params[j, 2] = cNFW[halo_idx]

        # Sort arrays by concentration (in descending order)
        order = halo_params[:,2].argsort()[::-1]
        select_halos_sorted = select_halos[order]
        halo_params_sorted = halo_params[order]

        # Delete entries with 0. concentration (erroneous halos)
        not0c = ~np.any(halo_params_sorted==0, axis=1)
        select_halos_trimmed = select_halos_sorted[not0c]
        halo_params_trimmed = halo_params_sorted[not0c]

        if halo_number >= halo_limit:
            select_halos_lim = select_halos_trimmed[:halo_limit]
            halo_params_lim = halo_params_trimmed[:halo_limit]

        np.save(f'{out_dir}/halo_batch_{fname}_indices.npy', select_halos_lim)
        np.save(f'{out_dir}/halo_batch_{fname}_params.npy', halo_params_lim)


    @staticmethod
    def M12_to_M12X(M12_val):
        return np.log(M12_val*10.**12)/np.log(10.)


    @staticmethod
    def simulation_setup(
        sim_dir, m_lower, m_upper, m_gauge, halo_num_req, no_gravity=False, benchmark=False):

        ### ================================= ###
        ### Load box & simulation parameters. ###
        ### ================================= ###

        # Box parameters.
        with open(f'{sim_dir}/box_parameters.yaml', 'r') as file:
            box_setup = yaml.safe_load(file)

        box_file_dir = box_setup['File Paths']['Box File Directory']
        DM_mass = box_setup['Content']['DM Mass [Msun]']*Params.Msun
        z0_snap_4cif = box_setup['Content']['z=0 snapshot']

        # Simulation parameters.
        with open(f'{sim_dir}/sim_parameters.yaml', 'r') as file:
            sim_setup = yaml.safe_load(file)

        CPUs_sim = sim_setup['CPUs_simulations']
        neutrinos = sim_setup['neutrinos']

        # Initial distance from center (Earth-GC) cell must approximate.
        init_dis = sim_setup['initial_haloGC_distance']

        # Load arrays.
        zeds_snaps = jnp.load(f'{sim_dir}/zeds_snaps.npy')

        z_int_steps = jnp.load(f'{sim_dir}/z_int_steps.npy')
        s_int_steps = jnp.load(f'{sim_dir}/s_int_steps.npy')
        nu_massrange = jnp.load(f'{sim_dir}/neutrino_massrange_eV.npy')*Params.eV

        if no_gravity:

            return DM_mass, CPUs_sim, neutrinos, init_dis, zeds_snaps, z_int_steps, s_int_steps, nu_massrange

        else:

            ### =========================== ###
            ### Create halo batch and files ###
            ### =========================== ###

            # Parent and root directory of current sim folder
            sim_output_dir = str(pathlib.Path(sim_dir).parent)
            root_dir = str(pathlib.Path(sim_output_dir).parent)

            # All precalculations are stored here
            if benchmark:
                data_dir = f"{root_dir}/Data/benchmark_halo_files"
            else:
                data_dir = f"{root_dir}/Data/halo_grids"

            hname = f'{m_lower}-{m_upper}x1e+{m_gauge}_Msun'
            mass_neg = jnp.abs(
                float(m_gauge) - SimData.M12_to_M12X(float(m_lower)))
            mass_pos = jnp.abs(
                float(m_gauge) - SimData.M12_to_M12X(float(m_upper)))
            SimData.halo_batch_indices(
                z0_snap_4cif, 
                float(m_gauge), mass_neg, mass_pos,
                'halos', int(halo_num_req), 
                hname, box_file_dir, sim_dir
            )
            halo_batch_IDs = jnp.load(
                f'{sim_dir}/halo_batch_{hname}_indices.npy')
            halo_batch_params = jnp.load(
                f'{sim_dir}/halo_batch_{hname}_params.npy')
            halo_num_avail = len(halo_batch_params)

            print(f'******** CNB Simulation ********')
            print('Halo batch params (Rvir,Mvir,cNFW):')
            print(halo_batch_params)
            print('*********************************\n')

            return DM_mass, CPUs_sim, neutrinos, init_dis, zeds_snaps, z_int_steps, s_int_steps, nu_massrange, data_dir, halo_batch_IDs, halo_num_avail


    @staticmethod
    def read_DM_halo_index(
        snap, halo_ID, fname, box_file_dir, out_dir, direct=False
    ):

        # ---------------- #
        # Open data files. #
        # ---------------- #

        snaps = h5py.File(f'{box_file_dir}/snapshot_{snap}.hdf5')
        group = h5py.File(f'{box_file_dir}/subhalo_{snap}.catalog_groups')
        parts = h5py.File(f'{box_file_dir}/subhalo_{snap}.catalog_particles')
        props = h5py.File(f'{box_file_dir}/subhalo_{snap}.properties')

        # Positions.
        a = snaps["/Header"].attrs["Scale-factor"]
        pos = snaps['PartType1/Coordinates'][:][:] * a
        
        # Critical M_200.
        m200c = props['Mass_200crit'][:] * 1e10  # now in Msun
        m200c[m200c <= 0] = 1
        m200c = np.log10(m200c)  


        # ------------------------------------------------ #
        # Find DM particles gravitationally bound to halo. #
        # ------------------------------------------------ #

        halo_start_pos = group["Offset"][halo_ID]
        halo_end_pos = group["Offset"][halo_ID + 1]

        particle_ids_in_halo = parts["Particle_IDs"][halo_start_pos:halo_end_pos]
        particle_ids_from_snapshot = snaps["PartType1/ParticleIDs"][...]

        # Get indices of elements, which are present in both arrays.
        _, _, indices_p = np.intersect1d(
            particle_ids_in_halo, particle_ids_from_snapshot, 
            assume_unique=True, return_indices=True
        )


        # ------------------------------------------ #
        # Save DM positions centered on CoP of halo. #
        # ------------------------------------------ #

        CoP = np.zeros((len(m200c), 3))
        CoP[:, 0] = props["Xcminpot"][:]
        CoP[:, 1] = props["Ycminpot"][:]
        CoP[:, 2] = props["Zcminpot"][:]
        CoP_halo = CoP[halo_ID, :]

        DM_pos = pos[indices_p, :]  # x,y,z of each DM particle
        DM_pos -= CoP_halo
        DM_pos *= 1e3  # to kpc

        # Save c.o.m. coord of all DM particles (used for outside_gravity).
        DM_com_coord = np.sum(DM_pos, axis=0)/len(DM_pos)

        if direct:
            return DM_pos, DM_com_coord
        else:
            np.save(f'{out_dir}/DM_pos_indices_{fname}', indices_p)
            np.save(f'{out_dir}/DM_pos_{fname}.npy', DM_pos)
            np.save(f'{out_dir}/DM_com_coord_{fname}.npy', DM_com_coord)


    @staticmethod
    def read_DM_all_inRange(
        snap, halo_ID, mode, DM_shell_edges,
        fname, box_file_dir, out_dir, direct=False
    ):

        # --------------- #
        # Initialize data #
        # --------------- #

        snaps = h5py.File(f'{box_file_dir}/snapshot_{snap}.hdf5')
        props = h5py.File(f'{box_file_dir}/subhalo_{snap}.properties')

        # Positions.
        a = snaps["/Header"].attrs["Scale-factor"]
        pos = snaps['PartType1/Coordinates'][:][:]*a

        # Masses of all halos in sim.
        m200c = props['Mass_200crit'][:]

        # Center of Potential coordinates, for all halos.
        CoP = np.zeros((len(m200c), 3))
        del m200c
        CoP[:, 0] = props["Xcminpot"][:]
        CoP[:, 1] = props["Ycminpot"][:]
        CoP[:, 2] = props["Zcminpot"][:]
        CoP_halo = CoP[halo_ID, :]
        del CoP

        # Center all DM on selected halo and calculate distance from halo center.
        pos -= CoP_halo
        DM_dis = np.sqrt(np.sum(pos**2, axis=1))

        # DM_shell_edges from physical to comoving, sync to Camilas Gpc units
        DM_dis_lims = DM_shell_edges*(a/kpc/1e3)

        if mode == 'single_halos':
            # ----------------------------------- #
            # Save DM content up to Rvir of halo. #
            # ----------------------------------- #
            
            DM_pos = pos[DM_dis <= DM_dis_lims, :]*1e3
            DM_com_coord = np.sum(DM_pos, axis=0)/len(DM_pos)

            if direct:
                return DM_pos, DM_com_coord
            else:        
                np.save(f'{out_dir}/DM_pos_{fname}.npy', DM_pos)
                np.save(f'{out_dir}/DM_com_coord_{fname}.npy', DM_com_coord)

        elif mode == 'spheres':
            # ----------------------------------- #
            # Save DM in spherical shell batches. #
            # ----------------------------------- #

            for i, (shell_start, shell_end) in enumerate(
                zip(DM_dis_lims[:-1], DM_dis_lims[1:])
            ):

                DM_pos = pos[(shell_start < DM_dis) & (DM_dis <= shell_end), :]*1e3
                np.save(f'{out_dir}/DM_pos_{fname}_shell{i}.npy', DM_pos)


    @staticmethod
    def load_dPsi_long_range(c_id, batches, out_dir):

        # Load all batches for current cell.
        dPsi_raw = np.array(
            [np.load(f'{out_dir}/cell{c_id}_batch{b}_long_range.npy') for b in batches]
        )

        # Combine into one array by summing and save.
        dPsi_for_cell = np.sum(dPsi_raw, axis=0)
        np.save(f'{out_dir}/cell{c_id}_long_range.npy', dPsi_for_cell)


class SimGrid():

    @staticmethod
    def grid_3D(l, s, origin_coords=[0.,0.,0.,]):
        """
        Generate 3D cell center coordinate grid (built around origin_coords) 
        extending from center until l in all 3 axes, spaced apart by s.
        If l=s then 8 cells will be generated (used for dividing 1 cell into 8).
        """

        # Generate edges of 3D grid.
        l_eps = l/10.
        x, y, z = np.mgrid[-l:l+l_eps:s, -l:l+l_eps:s, -l:l+l_eps:s]

        # Calculate centers of each axis.
        x_centers = (x[1:,...] + x[:-1,...])/2.
        y_centers = (y[:,1:,:] + y[:,:-1,:])/2.
        z_centers = (z[...,1:] + z[...,:-1])/2.

        # Create center coord.-pairs. and truncating outermost "layer".
        centers3D = np.array([
            x_centers[:,:-1,:-1], 
            y_centers[:-1,:,:-1], 
            z_centers[:-1,:-1,:]
        ])

        cent_coordPairs3D = centers3D.reshape(3,-1).T 

        # Shift center of creation from (0,0,0) to other coords.
        cent_coordPairs3D += origin_coords

        return cent_coordPairs3D


    @staticmethod
    def check_grid(init_grid, DM_pos, parent_GRID_S, DM_lim, FCT_DM_shell_edges, FCT_shell_multipliers):
        """
        Determine which cells have DM above threshold and thus need division.
        """

        # Center all DM positions w.r.t. center, for all cells.
        DM_pos -= init_grid

        # Cell length of current grid generation, used to limit DM particles.
        cell_len = np.ones((len(init_grid),1), dtype=np.float64)*parent_GRID_S/2.

        # Select DM particles inside each cell based on cube length generation.
        DM_in_cell_IDs = np.asarray(
            (np.abs(DM_pos[:,:,0]) < cell_len) & 
            (np.abs(DM_pos[:,:,1]) < cell_len) & 
            (np.abs(DM_pos[:,:,2]) < cell_len)
        )

        # Set DM outside cell to nan values.
        DM_pos[~DM_in_cell_IDs] = np.nan

        # Sort all nan values to the bottom of axis 1, i.e. the DM-in-cell-X axis.
        ind_2D = DM_pos[:,:,0].argsort(axis=1)
        ind_3D = np.repeat(np.expand_dims(ind_2D, axis=2), 3, axis=2)
        DM_sort = np.take_along_axis(DM_pos, ind_3D, axis=1)
        del ind_2D, ind_3D

        # Drop "rows" common to all cells, which contain only nan values. This is 
        # determined by the cell with the most non-nan entries.
        DM_count_cells = np.count_nonzero(~np.isnan(DM_sort[:,:,0]), axis=1)
        DM_compact = np.delete(
            DM_sort, np.s_[np.max(DM_count_cells):], axis=1
        )
        cells = len(DM_compact)
        del DM_sort


        ### Adjust DM threshold to the distance of the cell from the center.
        ### (cells farther away have higher DM threshold, i.e. less division)

        # Distance from center for each cell.
        grid_dis = np.sum(np.sqrt(init_grid**2), axis=2) # shape = (cells, 1)

        # Radial distance of each shell center, adjust array dimensionally.
        shell_cents = (FCT_DM_shell_edges[:-1] + FCT_DM_shell_edges[1:])/2.
        shells_sync = np.repeat(np.expand_dims(shell_cents, axis=0), cells, axis=0)

        # Find shell center, which each cell is closest to.
        which_shell = np.abs(grid_dis - shells_sync).argmin(axis=1)

        # Final DM limit for each cell.
        cell_DMlims = np.array(
            [FCT_shell_multipliers[k] for k in which_shell]
        )*DM_lim


        ### Drop all cells containing an amount of DM below the given threshold, 
        ### from the DM positions array.

        # stable_cells = DM_count_cells <= DM_lim  # note: original
        stable_cells = DM_count_cells <= cell_DMlims
        DM_unstable_cells = np.delete(DM_compact, stable_cells, axis=0)
        thresh = np.size(DM_unstable_cells, axis=0)
        del DM_count_cells

        # Count the number of DM particles in stable cells.
        DM_stable_cells = np.delete(DM_compact, ~stable_cells, axis=0)
        DM_count_stable = np.count_nonzero(
            ~np.isnan(DM_stable_cells[:,:,0]), axis=1
        )
        del DM_compact

        # Calculate c.o.m coords for stable cells.
        DM_count_sync = np.expand_dims(DM_count_stable, axis=1)
        DM_count_sync[DM_count_sync==0] = 1  # to avoid divide by zero
        cell_com = np.nansum(DM_stable_cells, axis=1)/DM_count_sync
        del DM_count_stable, DM_count_sync

        # Reset cell_com coords. of stable cells to proper coordinates.
        good_grid = np.squeeze(np.delete(init_grid, ~stable_cells, axis=0), axis=1)
        cell_com += good_grid
        del good_grid
        
        # note: 
        # cell_com can have (0,0,0) for a cell. Doesn't matter, since DM_count in 
        # cell is then 0, which will set term in long-range gravity to zero.

        # Count again, where zeros are present (i.e. not replaced by ones).
        DM_count_final = np.count_nonzero(
            ~np.isnan(DM_stable_cells[:,:,0]), axis=1
        )

        # Free up memory just in case.
        del DM_stable_cells

        return DM_count_final, cell_com, stable_cells, DM_unstable_cells, thresh


    @staticmethod
    def cell_division(
        init_grid, DM_pos, parent_GRID_S, DM_lim, stable_grid, FCT_DM_shell_edges, FCT_shell_multipliers, out_dir, fname
        ):


        # Initiate counters.
        thresh = 1
        cell_division_count = 0

        DM_count_l = []
        cell_com_l = []
        cell_gen_l = []

        while thresh > 0:

            DM_count, cell_com, stable_cells, DM_parent_cells, thresh = SimGrid.check_grid(
                init_grid, DM_pos, parent_GRID_S, DM_lim, FCT_DM_shell_edges, FCT_shell_multipliers
            )

            # Save DM count and c.o.m coords of stable cells.
            DM_count_l.append(DM_count)
            cell_com_l.append(cell_com)

            # If no cells are in need of division -> return final coords.
            if thresh == 0:

                # Append cell generation number of last iteration.
                cell_gen_l.append(
                    np.zeros(len(init_grid), int) + cell_division_count
                )
                
                # Convert nested lists to ndarrays.
                cell_gen_np = np.array(list(chain.from_iterable(cell_gen_l)))
                DM_count_np = np.array(list(chain.from_iterable(DM_count_l)))
                cell_com_np = np.array(list(chain.from_iterable(cell_com_l)))

                if cell_division_count > 0:

                    # The final iteration is a concatenation of the survival cells 
                    # from the previous iteration and the newest sub8 cell coords 
                    # (which are now init_grid).
                    final_cc = np.concatenate((stable_grid, init_grid), axis=0)
                    np.save(f'{out_dir}/fin_grid_{fname}.npy', final_cc)

                else:
                    # Return initial grid itself, if it's fine-grained already.
                    np.save(f'{out_dir}/fin_grid_{fname}.npy', init_grid)

                # Save DM count, c.o.m. coord, and generation, for all cells.
                np.save(f'{out_dir}/DM_count_{fname}.npy', DM_count_np)
                np.save(f'{out_dir}/cell_com_{fname}.npy', cell_com_np)
                np.save(f'{out_dir}/cell_gen_{fname}.npy', cell_gen_np)

                return cell_division_count


            else:
                del DM_count, cell_com
                
                # Array containing all cells (i.e. their coords.), which need to
                # be divided into 8 "child cells", hence the name "parent cells".
                parent_cells = np.delete(init_grid, stable_cells, axis=0)
                pcs = len(parent_cells)

                # Array containing all cells, which are "stable" and need no 
                # division, so excluding all parent cells.
                no_parent_cells = np.delete(init_grid, ~stable_cells, axis=0)

                # Save generation index of stable cells.
                cell_gen_l.append(
                    np.zeros(len(no_parent_cells), int) + cell_division_count
                )

                # -------------------------------------------------- #
                # Replace each parent cell by the 8 new child cells. #
                # -------------------------------------------------- #

                # Reset DM on parent cells, such that they can be centered on new 
                # child cells again later.
                DM_reset = DM_parent_cells + parent_cells
                # note: Array doesn't contain duplicate DM, each cell has unique DM.

                # Repeat each DM "column" 8 times. This will be the DM position 
                # array in the next iteration.
                DM_rep8 = np.repeat(DM_reset, repeats=8, axis=0)

                # Create 8 new cells around origin of (0,0,0). The length and size 
                # of the new cells is determined by the previous length of the 
                # parent cell, i.e. half of it.
                sub8_GRID_S = parent_GRID_S/2.
                sub8_raw = SimGrid.grid_3D(sub8_GRID_S, sub8_GRID_S)

                # Temporarily reshape to center on parent cells.
                sub8_temp = np.tile(sub8_raw, (pcs,1)).reshape((pcs, 8, 3))

                # Center each new 8-batch of child cells on a different parent cell.
                sub8_coords = np.reshape(sub8_temp + parent_cells, (pcs*8, 1, 3))
                del sub8_raw, sub8_temp, parent_cells

                if cell_division_count > 0:
                    stable_grid_so_far = np.concatenate(
                        (stable_grid, no_parent_cells), axis=0
                    )
                else:  # ending of first division loop
                    stable_grid_so_far = no_parent_cells

                # Overwrite variables for next loop.
                init_grid     = sub8_coords
                DM_pos        = DM_rep8
                parent_GRID_S = sub8_GRID_S
                stable_grid   = stable_grid_so_far

                cell_division_count += 1


    @staticmethod
    def chunksize_short_range(cells, DM_tot, max_DM_lim, core_mem_MB):

        # note: mem_MB specific to peak memory usage in cell_gravity_short_range.
        # -> Peak memory after calculation of ind_2D,ind_3D,etc. sorting arrays.

        elem = 8                               # 8 bytes for standard np.float64
        mem_type0 = cells*3 * elem             # for list to ndarray of cell_coords
        mem_type1 = cells*DM_tot * elem        # for ind_2D
        mem_type2 = cells*DM_tot*3 * elem      # for DM_pos_sync, ind_3D, DM_sort
        mem_type3 = cells*max_DM_lim*3 * elem  # for DM_in

        mem_MB = (mem_type0+mem_type1+(3*mem_type2)+mem_type3)/1.e6

        batches = 1
        while mem_MB >= 0.90*core_mem_MB:
            mem_MB *= batches
            batches += 1
            mem_MB /= batches

        chunksize = math.ceil(cells/batches)

        return chunksize


    @staticmethod
    def batch_generators_short_range(cell_coords, cell_gen, chunksize):

        cells = len(cell_coords)

        batches = math.ceil(cells/chunksize)
        batch_arr = np.arange(batches)

        cell_chunks = list(chunks(chunksize, cell_coords))
        cgen_chunks = list(chunks(chunksize, cell_gen))
        
        return batch_arr, cell_chunks, cgen_chunks


    @staticmethod
    def cell_gravity_short_range(
        cell_coords_in, cell_gen, init_GRID_S,
        DM_pos, DM_lim, DM_sim_mass, FCT_shell_multipliers, smooth_l,
        out_dir, b_id, max_b_len
    ):

        cell_coords = np.expand_dims(np.array(cell_coords_in), axis=1)
        cell_gen = np.array(cell_gen)

        # Center all DM positions w.r.t. cell center.
        # DM_pos already in shape = (1, DM_particles, 3)
        DM_pos_sync = np.repeat(DM_pos, len(cell_coords), axis=0)
        DM_pos_sync -= cell_coords

        # Cell lengths to limit DM particles. Limit for the largest cell is 
        # GRID_S/2, not just GRID_S, therefore the cell_gen+1 !
        cell_len = np.expand_dims(init_GRID_S/(2**(cell_gen+1)), axis=1)
        
        # Select DM particles inside each cell based on cube length generation.
        DM_in_cell_IDs = np.asarray(
            (np.abs(DM_pos_sync[...,0]) < cell_len) & 
            (np.abs(DM_pos_sync[...,1]) < cell_len) & 
            (np.abs(DM_pos_sync[...,2]) < cell_len)
        )
        #? < results in 1 missing DM particle. Using <= though overcounts
        #? is there a way to get every DM particle by adjusting rtol and atol ?
        # note: -> not pressing for now however

        # Set DM outside cell to nan values.
        DM_pos_sync[~DM_in_cell_IDs] = np.nan

        # Save the DM IDs, such that we know which particles are in which cell.
        # This will be used in the long-range gravity calculations.
        DM_in_cell_IDs_compact = np.argwhere(DM_in_cell_IDs==True)
        DM_in_cell_IDs_compact[:,0] += (max_b_len*b_id)    
        np.save(f'{out_dir}/batch{b_id}_DM_in_cell_IDs.npy', DM_in_cell_IDs_compact)

        # Sort all nan values to the bottom of axis 1, i.e. the DM-in-cell-X axis 
        # and truncate array based on DM_lim parameter. This simple way works since 
        # each cell cannot have more than DM_lim (times the last shell multiplier).
        ind_2D = DM_pos_sync[:,:,0].argsort(axis=1)
        ind_3D = np.repeat(np.expand_dims(ind_2D, axis=2), 3, axis=2)
        DM_sort = np.take_along_axis(DM_pos_sync, ind_3D, axis=1)
        DM_in = DM_sort[:,:DM_lim*FCT_shell_multipliers[-1],:]
        
        # Calculate distances of DM and adjust array dimensionally.
        DM_dis = np.expand_dims(np.sqrt(np.sum(DM_in**2, axis=2)), axis=2)

        # Offset DM positions by smoothening length of Camila's simulations.
        eps = smooth_l / 2.

        # Quotient in sum (see formula). Can contain nan values, thus the np.nansum 
        # for the derivative, s.t. these values don't contribute. We center DM on 
        # c.o.m. of cell C, so we only need DM_in in numerator.
        quot = (-DM_in)/np.power((DM_dis**2 + eps**2), 3./2.)
        
        # Gradient of potential.
        gradient = Params.G*DM_sim_mass*np.nansum(quot, axis=1)

        # Acceleration is negative value of (grav. pot.) gradient.
        np.save(f'{out_dir}/batch{b_id}_short_range.npy', -gradient)


    @staticmethod
    def chunksize_long_range(cells, core_mem_MB):
        
        # note: mem_MB specific to peak memory usage in cell_gravity_long_range.
        # -> Peak memory after calculation of derivative.

        elem = 8                          # 8 bytes for standard np.float64
        mem_type1 = 3*elem                # for derivative
        mem_type2 = cells*3*elem          # for quot
        mem_type3 = cells*elem            # for DM_count_sync

        mem_MB = (mem_type1+mem_type2+mem_type3)/1.e6

        batches = 1
        while mem_MB >= 0.90*core_mem_MB:
            mem_MB *= batches
            batches += 1
            mem_MB /= batches

        chunksize = math.ceil(cells/batches)

        return chunksize
        

    @staticmethod
    def batch_generators_long_range(
        cell_coords, com_coords, cell_gen, 
        DM_counts, chunksize 
    ):

        # Cell number, chunksize and resulting number of batches.
        cells = len(cell_coords)
        cell_nums = np.arange(cells)
        batches = math.ceil(cells/chunksize)


        ### ------- ###
        ### Arrays. ###
        ### ------- ###

        # Arrays are repeated/tiled to match length & number of batches.

        batch_IDs = np.tile(np.arange(batches), cells)  # batch labels/indices
        # [0,...,batches,0,...,batches,...]
        
        cellC_rep = np.repeat(cell_nums, batches)  # cell labels/indices
        # [0,...,0,1,...,1,...,cells-1,...,cells-1]

        cellC_cc = np.repeat(cell_coords, batches, axis=0)  # cell coordinates
        # [[x1,y1,z1],...,[x1,y1,z1],...]

        cellC_gen = np.repeat(cell_gen, batches)  # cell generations
        # e.g. [0,...,0,0,...,0,1,...1,...], ... repeated to match nr of batches

        ### ----------- ###
        ### Generators. ###
        ### ----------- ###

        # We split up all arrays into smaller chunks, s.t. each batch gets a 
        # smaller set of cells. Generators are used to benefit from the chunks() 
        # function, then converted to lists (map() will do this anyway).
        
        # We do this for:
        # 1. the cell_IDs of cells present in a batch, 
        # 2. the DM counts of all cells in a batch, 
        # 3. the c.o.m. coords of all cells in a batch,
        # 4. the (cell division) generation index for all cells in a batch:

        # 1. 
        cib_IDs = list(
            (c for _ in cell_nums for c in chunks(chunksize, cell_nums)))

        # 2.
        counts = list(
            (c for _ in cell_nums for c in chunks(chunksize, DM_counts)))

        # 3.
        coms = list(
            (c for _ in cell_nums for c in chunks(chunksize, com_coords)))

        # 4.
        gens = list(
            (c for _ in cell_nums for c in chunks(chunksize, cell_gen)))

        return batch_IDs, cellC_rep, cellC_cc, cellC_gen, \
            cib_IDs, counts, coms, gens


    @staticmethod
    def cell_gravity_long_range_quadrupole(
        c_id, cib_ids, b_id, cellC_cc, cell_com, cell_gen, init_GRID_S,
        DM_pos, DM_count, DM_in_cell_IDs, DM_sim_mass, out_dir, 
        max_b_len, cellC_gen
    ):
        # Prefix "cellC" denotes the cell to calculate the long-range forces for.

        # Convert the list inputs to numpy arrays.
        cib_ids = np.array(cib_ids)
        cell_com = np.array(cell_com)
        cell_gen = np.array(cell_gen)
        DM_count = np.array(DM_count)

        # Array, where cell C is centered on c.o.m. coords. of all cells.
        cellC_sync = np.repeat(
            np.expand_dims(cellC_cc, axis=0), len(cell_com), axis=0
        ) 
        cellC_sync -= cell_com

        # Get (complete, not half) cell length of all cells and current cell C.
        cell_len = init_GRID_S/(2**(cell_gen))
        cellC_len = init_GRID_S/(2**(cellC_gen))

        # Distance of cellC to all cell_com's.
        cellC_dis = np.sqrt(np.sum(cellC_sync**2, axis=1))

        # Cells with distance bewlow critical we compute quadrupole moments for.
        r_crit = 1.5*cellC_len*((cellC_gen+1)**0.6)  # set to -1 for no quadrupoles
        multipole_IDs = np.argwhere(cellC_dis <= r_crit).flatten()

        if c_id in cib_ids:
            # Overwrite element corresponding to cell C from all arrays, to avoid
            # self-gravity of the cell (taken care of by short-range gravity).

            # First find the correct index for the current cell. We have to take 
            # this akin to a "modulo" according to the maximum batch length. E.g. 
            # if the max. batch length is 100, but we are at cell label/index 100, 
            # then this would be index '0' again for arrays in this function.
            cellC_idx = (c_id - (max_b_len*b_id))

            # Set cell C array element(s) to 0 or nan, s.t. it just adds 0 later on.
            cell_com[cellC_idx] = np.nan  # c.o.m. of cell C to [nan,nan,nan]
            cell_len[cellC_idx] = 0  # 0 cell length
            DM_count[cellC_idx] = 0  # 0 DM count

            # If cell C has 0 DM, then its c.o.m == coords., so set cellC_dis to 1.
            # (avoids divide by zero in these (edge) cases later on)
            if cellC_dis[cellC_idx] == 0:
                cellC_dis[cellC_idx] = 1

        # DM count for multipole cells.
        DM_count_mpoles = DM_count[multipole_IDs]

        # All DM particles (their positions), which are in multipole cells.
        # Adjust/match multipole_IDs first, since they are limited to size of batch.
        comparison_IDs = multipole_IDs + (max_b_len*b_id)
        DM_IDs = DM_in_cell_IDs[np.in1d(DM_in_cell_IDs[:,0], comparison_IDs)][:,1]
        DM_pos_mpoles = np.take(DM_pos, DM_IDs, axis=0)

        # Split this total DM array into DM chunks present in each multipole cell:
        # We can do this by breaking each axis into sub-arrays, with length 
        # depending on how many DM particles are in each cell (which is stored in 
        # DM_count of multipole cells, i.e. DM_count_mpoles).
        
        # Special case, where all multipole cells have no DM.
        # (This can happen e.g. if cell C is an outermost cell)
        if np.all(DM_count_mpoles==0):
            DM_mpoles = np.full(shape=(len(multipole_IDs),1,3), fill_value=np.nan)

        # "Normal" case, where some or all multipole cells contain DM.
        else:
            breaks = np.cumsum(DM_count_mpoles[:-1])
            ax0_split = np.split(DM_pos_mpoles[:,0], breaks)
            ax1_split = np.split(DM_pos_mpoles[:,1], breaks)
            ax2_split = np.split(DM_pos_mpoles[:,2], breaks)

            # Fill each axis with nans to obtain dimensionally valid ndarray.
            DM_axis0 = np.array(list(zip_longest(*ax0_split, fillvalue=np.nan))).T
            DM_axis1 = np.array(list(zip_longest(*ax1_split, fillvalue=np.nan))).T
            DM_axis2 = np.array(list(zip_longest(*ax2_split, fillvalue=np.nan))).T

            # Recombine all axes into one final DM positions array.
            DM_mpoles = np.stack((DM_axis0, DM_axis1, DM_axis2), axis=2)
            del ax0_split, ax1_split, ax2_split, DM_axis0, DM_axis1, DM_axis2    


        # Select c.o.m. of multipole cells and adjust dimensionally.
        mpoles_com = np.expand_dims(cell_com[multipole_IDs], axis=1)

        # Center DM in multipole cells (labeled with "J") on their c.o.m. coords.
        DM_mpoles -= mpoles_com

        # Calculate distances of all DM in J cells from their c.o.m. coord.
        DMj_dis = np.expand_dims(np.sqrt(np.sum(DM_mpoles**2, axis=2)), axis=2)
        
        # Array, where cell C is centered on c.o.m. coords. of J cells.
        cellC_Jcoms = np.repeat(
            np.expand_dims(cellC_cc, axis=(0,1)), 
            len(mpoles_com), axis=0
        )
        cellC_Jcoms -= mpoles_com

        # Distance of cell C to c.o.m. of multipole cells.
        cellC_Jdis = np.sqrt(np.sum(cellC_Jcoms**2, axis=2))


        ### -------------------- ###
        ### Quadrupole formulas. ###
        ### -------------------- ###
        # See GoodNotes for notation.

        # Terms appearing in the quadrupole term.
        QJ_aa = np.nansum(3*DM_mpoles**2 - DMj_dis**2, axis=1)
        QJ_ab = np.nansum(3*DM_mpoles*np.roll(DM_mpoles, 1, axis=2), axis=1)

        # Reduce dimensions.
        cellC_Jcoms = np.squeeze(cellC_Jcoms, axis=1)
        cellC_Jdis_1D = np.squeeze(cellC_Jdis, axis=1)
        
        # Permute order of coords by one, i.e. (x,y,z) -> (z,x,y).
        cellC_Jcoms_roll = np.roll(cellC_Jcoms, 1, axis=1)

        # Factors of 2 are for the symmetry of QJ_ab elements.
        term1_aa = np.nansum(QJ_aa*cellC_Jcoms, axis=1)
        term1_ab = np.nansum(2*QJ_ab*cellC_Jcoms_roll, axis=1)
        term1 = np.expand_dims((term1_aa+term1_ab)/cellC_Jdis_1D**5, axis=1)

        term2_pre = 5*cellC_Jcoms/(2*cellC_Jdis**7)
        term2_aa = np.nansum(QJ_aa*cellC_Jcoms**2, axis=1)
        term2_ab = np.nansum(2*QJ_ab*cellC_Jcoms*cellC_Jcoms_roll, axis=1)
        term2 = term2_pre*np.expand_dims(term2_aa+term2_ab, 1)

        dPsi_multipole_cells = Params.G*DM_sim_mass*np.nansum(-term1+term2, axis=0)
        

        ### ---------------------- ###
        ### Monopole of all cells. ###
        ### ---------------------- ###

        mono_com = np.expand_dims(cell_com, axis=1)

        # Array, where cell C is centered on c.o.m. coords. of monopole cells.
        cellC_Jcoms_mono = np.repeat(
            np.expand_dims(np.expand_dims(cellC_cc, axis=0), axis=0), 
            len(mono_com), axis=0
        )
        cellC_Jcoms_mono -= mono_com

        # Distance of cell C to c.o.m. of monopole cells.
        cellC_dis_mono = np.sqrt(np.sum(cellC_Jcoms_mono**2, axis=2))
        
        # Long-range force of all cells due to monopole.
        cellC_Jcoms_mono_2D = np.squeeze(cellC_Jcoms_mono, axis=1)
        DM_count_mono_sync = np.expand_dims(DM_count, axis=1)
        dPsi_monopole_cells = Params.G*DM_sim_mass*np.nansum(
            DM_count_mono_sync*cellC_Jcoms_mono_2D/(cellC_dis_mono**3), axis=0)

        # Total gradient.
        gradient_lr = dPsi_multipole_cells + dPsi_monopole_cells
        
        # Acceleration is negative value of (grav. pot.) gradient.
        np.save(f'{out_dir}/cell{c_id}_batch{b_id}_long_range.npy', -gradient_lr)


    @staticmethod
    def grid_data(origID, data_dir, start_snap=12, end_snap=36, benchmark=False):
        
        # Initialize lists to store the data
        dPsi_grid_list = []
        cell_grid_list = []
        cell_gens_list = []

        # Loop over the snapshot numbers
        for snap_num in range(start_snap, end_snap + 1):
            snap = str(snap_num).zfill(4)  # Format the number as a 4-digit string

            # Load the data files
            # The 3 arrays have shapes (N,3), (N,1,3) and (N,), respectively
            if benchmark:
                dPsi_grid_f = f'{data_dir}/dPsi_grid_benchmark_halo_snap_{snap}.npy'
                cell_grid_f = f'{data_dir}/fin_grid_benchmark_halo_snap_{snap}.npy'
                cell_gens_f = f'{data_dir}/cell_gen_benchmark_halo_snap_{snap}.npy'
            else:
                dPsi_grid_f = f'{data_dir}/dPsi_grid_origID{origID}_snap_{snap}.npy'
                cell_grid_f = f'{data_dir}/fin_grid_origID{origID}_snap_{snap}.npy'
                cell_gens_f = f'{data_dir}/cell_gen_origID{origID}_snap_{snap}.npy'

            # Append arrays to list, squeezing (N,1,3) cell_grid to (N,3) shape
            dPsi_grid_list.append(jnp.load(dPsi_grid_f))
            cell_grid_list.append(jnp.squeeze(jnp.load(cell_grid_f)))
            cell_gens_list.append(jnp.load(cell_gens_f))

        # Pad with zeros and convert lists to jnp arrays
        dPsi_grid = Utils.fill_2d_arrays_with_zeros_jax(dPsi_grid_list)
        cell_grid = Utils.fill_2d_arrays_with_zeros_jax(cell_grid_list)
        cell_gens = Utils.fill_1d_arrays_with_zeros_jax(cell_gens_list)

        return dPsi_grid, cell_grid, cell_gens


class SimExec():

    @jax.jit
    def nu_in_which_cell(x_i, cell_coords, cell_gens, init_GRID_S):

        # Number of cells.
        num_cells = cell_coords.shape[0]

        # "Center" the neutrino coordinates on the cell coordinates.
        x_cent = x_i - cell_coords.reshape(num_cells, 3)

        # All cell lengths. Limit for the largest cell is GRID_S/2, not just 
        # GRID_S, therefore the cell_gen+1 !
        cell_lens = init_GRID_S / (2**(cell_gens + 1))

        # Find index of cell in which neutrino is enclosed.
        in_cell = (
            (jnp.abs(x_cent[..., 0]) < cell_lens) & 
            (jnp.abs(x_cent[..., 1]) < cell_lens) & 
            (jnp.abs(x_cent[..., 2]) < cell_lens)
        )
        cell_idx = jnp.argmax(in_cell)
        cell_len = cell_lens[cell_idx]
        cell_ccs = cell_coords[cell_idx, :]

        return cell_idx, cell_len, cell_ccs


    @jax.jit
    def outside_gravity_quadrupole(x_i, com_halo, DM_sim_mass, DM_num, QJ_abs):

        ### ----------- ###
        ### Quadrupole. ###
        ### ----------- ###

        # Center neutrino on c.o.m. of halo and get distance.
        x_i = x_i - com_halo
        r_i = jnp.linalg.norm(x_i)

        # Permute order of coords by one, i.e. (x,y,z) -> (z,x,y).
        x_i_roll = jnp.roll(x_i, shift=1)

        # Terms appearing in the quadrupole term.
        QJ_aa = QJ_abs[0]
        QJ_ab = QJ_abs[1]

        # Factors of 2 are for the symmetry of QJ_ab elements.
        term1_aa = jnp.sum(QJ_aa * x_i, axis=0)
        term1_ab = jnp.sum(2 * QJ_ab * x_i_roll, axis=0)
        term1 = (term1_aa + term1_ab) / r_i**5

        term2_pre = 5 * x_i / (2 * r_i**7)
        term2_aa = jnp.sum(QJ_aa * x_i**2, axis=0)
        term2_ab = jnp.sum(2 * QJ_ab * x_i * x_i_roll, axis=0)
        term2 = term2_pre * (term2_aa + term2_ab)

        dPsi_multipole_cells = Params.G * DM_sim_mass * (-term1 + term2)

        ### --------- ###
        ### Monopole. ###
        ### --------- ###

        dPsi_monopole_cells = Params.G * DM_num * DM_sim_mass * x_i / r_i**3

        # Total outside gravity gradient.
        gradient_out = dPsi_multipole_cells + dPsi_monopole_cells

        # Acceleration is negative value of (grav. pot.) gradient.
        return -gradient_out


    @jax.jit
    def update_sun_position_at_origin(t, km, s):

        # Sun's velocity vector w.r.t. CNB(==CMB) from Planck 2018
        # (e.g. paper I, table 3)
        # Originally in galactic coordinates (l,b), now in caretesian frame with
        # x-axis defined by Sun-GC line in this case.
        v_CNB = jnp.array([-0.06935679, -0.66222051,  0.74609224])*369.82*km/s

        # Minus sign for Sun to move in opposite direction (back in time as sim)
        return - v_CNB * t


    @jax.jit
    def sun_gravity(x_i, eps, sun_position):

        # Distance from sun
        r_i = jnp.linalg.norm(x_i - sun_position)

        # Newtonion gravity for point mass with eps offset to avoid singularity
        quot = (x_i - sun_position) / jnp.power((r_i**2 + eps**2), 3./2.)
        gradient_sun = Params.G * Params.Msun * quot
        
        # Acceleration is negative value of (grav. pot.) gradient.
        return -gradient_sun


@chex.dataclass
class SimUtil():


    ### ============================ ###
    ### Frame manipulation & skymaps ###
    ### ============================ ###

    @staticmethod
    def Galactic_velocity_to_Cartesian_vector(velocity, l, b):
        """
        Convert velocity from galactic coordinates (l, b) to a Cartesian vector.

        Args:
        velocity (float): Magnitude of the velocity in km/s.
        l (float): Galactic longitude in degrees.
        b (float): Galactic latitude in degrees.

        Returns:
        jnp.ndarray: 3D velocity vector in Cartesian coordinates.
        """
        # Create a galactic coordinate object with a unit distance
        galactic_coord = Galactic(
            l=l*apu.degree, b=b*apu.degree, distance=1*apu.kpc)

        # Convert to Cartesian and normalize the direction vector
        x, y, z = galactic_coord.cartesian.xyz
        unit_vector = jnp.array([x.value, y.value, z.value]) / jnp.linalg.norm(jnp.array([x.value, y.value, z.value]))

        # Scale the unit vector by the velocity magnitude
        velocity_vector = unit_vector * velocity

        return velocity_vector

    @staticmethod
    def S_to_Sprime_frame_trafo(u, v, args=None):
        """
        Transform velocity from CNB frame (S) to Earth frame (S') using relativistic velocity addition formula.
        See https://en.wikipedia.org/wiki/Velocity-addition_formula, under "General configuration", 2nd formula (v positive, S -> S').
        
        Parameters:
        u: velocity of object (neutrinos) in frame S
        v: velocity of frame S' as measured in frame S
        
        Returns:
        u_prime: velocity of object in frame S'
        """

        if args is None:
            args = Params()

        u_dot_v = jnp.sum(u*v, axis=-1, keepdims=True)
        v_sq = jnp.sum(v**2, axis=-1, keepdims=True)
        
        # Lorentz factor
        gamma = 1/jnp.sqrt(1 - v_sq**2)
        
        numer = u/gamma - v + (gamma/(1+gamma))*u_dot_v*v
        denom = 1 - u_dot_v
        u_prime = numer/denom
        
        return u_prime

    @staticmethod
    def rotate_particles(particles_pos, obs_pos):

        # Calculate the Euler angles, to match frame of observer. This will be
        # the new frame in which DM particles get projected to an all sky map.
        zAngle = jnp.arctan2(obs_pos[1], -obs_pos[0])
        yAngle = jnp.arctan2(-obs_pos[2], jnp.linalg.norm(obs_pos[:2]))
        # print('*** Angles: ***')
        # print(np.rad2deg(zAngle), np.rad2deg(yAngle))

        cz = jnp.cos(zAngle)
        sz = jnp.sin(zAngle)
        cy = jnp.cos(yAngle)
        sy = jnp.sin(yAngle)

        # Rotation matrix around z-axis
        R_z = jnp.array([
            [cz, -sz, 0],
            [sz, cz,  0],
            [0,  0,   1]
        ])

        # Rotation matrix around y-axis
        R_y = jnp.array([
            [cy,  0, sy],
            [0,   1,  0],
            [-sy, 0, cy]
        ])

        # Combined rotation matrix
        rot_mat = jnp.matmul(R_y, R_z)

        # Observer position in new frame (aligned with x-axis).
        obs_orig_in_rot_frame = jnp.matmul(rot_mat, obs_pos.T).T

        # Dark matter positions in observer frame.
        DM_orig_in_rot_frame = jnp.matmul(rot_mat, particles_pos.T).T

        return DM_orig_in_rot_frame, obs_orig_in_rot_frame


    @staticmethod
    def get_rotation_matrix_euler(cell_pos):
        """
        Calculate rotation matrix using Euler angles to transform between reference frames.
        """
        # Calculate Euler angles for rotation
        # First angle: rotation around z-axis
        zAngle = jnp.arctan2(cell_pos[1], -cell_pos[0])
        
        # Second angle: rotation around y-axis
        yAngle = jnp.arctan2(-cell_pos[2], jnp.linalg.norm(cell_pos[:2]))
        
        # Compute trigonometric values
        cz = jnp.cos(zAngle)
        sz = jnp.sin(zAngle)
        cy = jnp.cos(yAngle)
        sy = jnp.sin(yAngle)
        
        # Rotation matrix around z-axis
        R_z = jnp.array([
            [cz, -sz, 0],
            [sz, cz,  0],
            [0,  0,   1]
        ])
        
        # Rotation matrix around y-axis
        R_y = jnp.array([
            [cy,  0, sy],
            [0,   1,  0],
            [-sy, 0, cy]
        ])
        
        # Combined rotation matrix (y-rotation after z-rotation)
        rot_mat = jnp.matmul(R_y, R_z)
        
        return rot_mat


    @staticmethod
    def get_direct_rotation(earth_position, cell_position):
        """
        Directly construct a rotation matrix that rotates earth_position to cell_position.
        """
        # Normalize both vectors
        earth_norm = earth_position / jnp.linalg.norm(earth_position, axis=-1)
        cell_norm = cell_position / jnp.linalg.norm(cell_position, axis=-1)
        
        # Calculate the rotation axis
        axis = jnp.cross(earth_norm, cell_norm)
        axis_norm = jnp.linalg.norm(axis, axis=-1)
        
        # Handle parallel or antiparallel vectors
        if axis_norm < 1e-10:
            # If vectors are parallel, no rotation needed
            if jnp.dot(earth_norm, cell_norm) > 0:
                return jnp.eye(3)
            # If vectors are antiparallel, rotate 180° around z-axis
            else:
                return jnp.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        
        # Normalize the axis
        axis = axis / axis_norm
        
        # Calculate the rotation angle
        cos_angle = jnp.dot(earth_norm, cell_norm)
        sin_angle = axis_norm
        
        # Build rotation matrix using Rodrigues' formula
        K = jnp.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = jnp.eye(3) + sin_angle * K + (1 - cos_angle) * jnp.matmul(K, K)
        return R


    @staticmethod
    def rotate_skymap(healpix_map, cell0, lonlat=True):

        # Input map related quantities
        Nside = hp.pixelfunc.get_nside(healpix_map)
        pix_idx = jnp.arange(len(healpix_map))

        # Angles to compensate for location of starting cell (cell0)
        zAngle = jnp.rad2deg(jnp.arctan2(cell0[1], -cell0[0]))
        yAngle = jnp.rad2deg(jnp.arctan2(-cell0[2], jnp.linalg.norm(cell0[:2])))

        # Create rotator object, to place c.o.p. of halo in the map center
        rot = hp.rotator.Rotator(
            rot=[-zAngle, yAngle, 0], deg=True, eulertype='ZYX', inv=False)

        # Create a grid of (theta, phi) coordinates for the map
        phi, theta = hp.pix2ang(
            Nside, pix_idx, lonlat=lonlat)
        
        # Apply rotation to the grid
        phi_rot, theta_rot = rot(phi, theta, lonlat=lonlat)
        
        # Find the pixel indices of the rotated grid
        pix_rot = hp.ang2pix(Nside, phi_rot, theta_rot, lonlat=lonlat)

        # Create a new map with rotated pixel values
        zeros_map = jnp.zeros_like(healpix_map)
        # rotated_map[pix_rot] = healpix_map
        rotated_map = zeros_map.at[pix_rot].set(healpix_map)
        #? pix_rot contains duplicates, and duplicated pixel will get the last healpix_map value...change to mean of all the healpix_map values, which fall onto the new pixel in rotated_map?
        
        return rotated_map
    
    @staticmethod
    def interpolate_skymap_OLD(healpix_map):

        Nside = hp.pixelfunc.get_nside(healpix_map)

        # Interpolate zero pixels with values from neighbours
        zero_pixels = jnp.where(healpix_map == 0)[0]
        for pix in zero_pixels:
            neighbours_pix = hp.get_all_neighbours(Nside, pix)
            interp_val = jnp.mean(healpix_map[neighbours_pix])
            healpix_map[pix] = interp_val

        return healpix_map

    def interpolate_skymap(healpix_map):

        Nside = hp.pixelfunc.get_nside(healpix_map)
        zero_pixels = jnp.where(healpix_map == 0)[0]
        
        for pix in zero_pixels:

            # Get neighbouring pixels or pixel without value
            neighbours_pix = hp.get_all_neighbours(Nside, pix)

            # Use median of neighbours to fill in missing value
            interp_val = jnp.median(healpix_map.at[neighbours_pix].get())
            
            # Update the pixel value using .at[].set() for immutability
            healpix_map = healpix_map.at[pix].set(interp_val)
        
        return healpix_map

    @staticmethod
    def smooth_skymap(healpix_map):

        Nside = hp.pixelfunc.get_nside(healpix_map)

        # Smooth using a Gaussian kernel.
        reso_1pix_deg = hp.nside2resol(Nside, arcmin=True) / 60
        sigma_rad = jnp.deg2rad(reso_1pix_deg)/4
        smooth_rot_map = hp.smoothing(healpix_map, sigma=sigma_rad)

        return smooth_rot_map

    @staticmethod
    def DM_pos_to_skymap(Nside, DM_pos_in, obs_pos_in):

        # Center on observer position.
        DM_pos = DM_pos_in - obs_pos_in
        xDM, yDM, zDM = DM_pos[:,0], DM_pos[:,1], DM_pos[:,2]

        # Convert x,y,z to angles.
        proj_xy_plane_dis = jnp.sqrt(xDM**2 + yDM**2)

        thetas = jnp.arctan2(zDM, proj_xy_plane_dis)
        phis = jnp.arctan2(yDM, xDM)

        # To galactic latitude and longitude (in degrees) for healpy.
        hp_glon, hp_glat = jnp.rad2deg(phis), jnp.rad2deg(thetas)

        # Convert angles to pixel indices using ang2pix.
        pixel_indices = hp.ang2pix(
            Nside, hp_glon, hp_glat, lonlat=True
        )

        # Create a Healpix map and increment the corresponding pixels
        DM_zeros = jnp.zeros(hp.nside2npix(Nside))
        # jnp.add.at(DM_healpix_map, pixel_indices, 1)
        DM_healpix_map = DM_zeros.at[pixel_indices].add(1)

        return DM_healpix_map

    ### ======================= ###
    ### Primordial Fluctuations ###
    ### ======================= ###

    @staticmethod
    def generate_DeltaTs_seeds(
        m_arr, Cl_dir, Delta_dir, seeds, simdata, z_cut, phases, args):

        Deltas_z_cut_s_l = []
        Deltas_z0_s_l = []
        for seed in seeds:

            Deltas_z_cut_m_l = []
            Deltas_z0_m_l = []
            for m_Cl in m_arr:

                # Load Cl's for current neutrino mass
                if z_cut == 0.5:
                    Cls_z_cut_raw = jnp.load(f"{Cl_dir}/Cls_z=0p5_m={m_Cl}eV.npy")
                elif z_cut == 0.15:
                    Cls_z_cut_raw = jnp.load(f"{Cl_dir}/Cls_z=0p15_m={m_Cl}eV.npy")
                else:
                    Cls_z_cut_raw = jnp.load(f"{Cl_dir}/Cls_z={z_cut}_m={m_Cl}eV.npy")
                Cls_z0_raw = jnp.load(f"{Cl_dir}/Cls_z=0_m={m_Cl}eV.npy")
                # (q momentum bins, l multipoles) = (50, 20)

                # Add monopole
                Cl_z_cut = jnp.insert(Cls_z_cut_raw, 0, 0, axis=1)
                Cl_z0 = jnp.insert(Cls_z0_raw, 0, 0, axis=1)
                # (q momentum bins, l multipoles + 1) = (50, 21)

                Deltas_z_cut_q_l = []
                Deltas_z0_q_l = []
                # Create temp. flucts. skymaps from Cls for each q-momentum
                for seed_int, qi in enumerate(range(Primordial.n_qbins)):

                    # np.random.seed(seed)  # original
                    np.random.seed(seed + seed_int)
                    Tmap_z_cut = hp.sphtfunc.synfast(
                        Cl_z_cut[qi], nside=simdata.Nside, lmax=None, pol=False)
                    # np.random.seed(seed)  # original  #note: seed needs to reset
                    np.random.seed(seed + seed_int)
                    Tmap_z0 = hp.sphtfunc.synfast(
                        Cl_z0[qi], nside=simdata.Nside, lmax=None, pol=False)

                    Deltas_z_cut = Tmap_z_cut * args.muK / (args.T_CNB*(1+z_cut))
                    Deltas_z0 = Tmap_z0 * args.muK / (args.T_CNB)
                    # (Npix)

                    Deltas_z_cut_q_l.append(Deltas_z_cut)
                    Deltas_z0_q_l.append(Deltas_z0)

                Deltas_z_cut_m_l.append(Deltas_z_cut_q_l)
                Deltas_z0_m_l.append(Deltas_z0_q_l)

            Deltas_z_cut_s_l.append(Deltas_z_cut_m_l)
            Deltas_z0_s_l.append(Deltas_z0_m_l)

        Deltas_z_cut_matrix = jnp.array(Deltas_z_cut_s_l)
        Deltas_z0_matrix = jnp.array(Deltas_z0_s_l)

        if z_cut == 0.5:
            z_cut = "0p5"
        if z_cut == 0.15:
            z_cut = "0p15"

        if phases:
            if len(seeds) == 1:
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z{z_cut}_random_phases.npy", 
                    Deltas_z_cut_matrix)
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z0_random_phases.npy", 
                    Deltas_z0_matrix)
            else:
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z{z_cut}_seeds_random_phases.npy", 
                    Deltas_z_cut_matrix)
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z0_seeds_random_phases.npy", 
                    Deltas_z0_matrix)
        else:
            if len(seeds) == 1:
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z{z_cut}.npy", 
                    Deltas_z_cut_matrix)
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z0.npy", 
                    Deltas_z0_matrix)
            else:
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z{z_cut}_seeds.npy", 
                    Deltas_z_cut_matrix)
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z0_seeds.npy", 
                    Deltas_z0_matrix)

        return Deltas_z0_matrix, Deltas_z_cut_matrix

    ### ============= ###
    ### Neutrino halo ###
    ### ============= ###

    @staticmethod
    def fibonacci_sphere(samples, radius=1):
        """Generate points on a sphere using the Fibonacci spiral method."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius_at_y = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            points.append((x * radius, y * radius, z * radius))
        return np.array(points)

    @staticmethod
    def select_spherical_grid(cell_centers, target_n, n_shells=10):
        """
        Select cells from the original grid to form a spherically symmetric grid.
        
        Parameters:
        cell_centers (np.array): Array of shape (N, 3) containing cell center coordinates.
        target_n (int): Target number of cells in the final grid.
        n_shells (int): Number of spherical shells to use.
        
        Returns:
        np.array: Selected cell centers forming a spherically symmetric grid.
        """
        # Find the center and maximum radius of the original grid
        center = np.mean(cell_centers, axis=0)
        max_radius = np.max(np.linalg.norm(cell_centers - center, axis=1))
        
        # Create KD-Tree for efficient nearest neighbor search
        tree = cKDTree(cell_centers)
        
        selected_cells = []
        cells_per_shell = target_n // n_shells
        
        for i in range(n_shells):
            radius = max_radius * (i + 1) / n_shells
            # Generate points on the sphere for this shell
            sphere_points = SimUtil.fibonacci_sphere(cells_per_shell, radius) + center
            
            # Find the nearest cells to these points
            _, indices = tree.query(sphere_points)
            selected_cells.extend(cell_centers[indices])
        
        # If we haven't reached target_n, add more cells from the last shell
        while len(selected_cells) < target_n:
            extra_points = SimUtil.fibonacci_sphere(target_n - len(selected_cells), max_radius) + center
            _, indices = tree.query(extra_points)
            selected_cells.extend(cell_centers[indices])
        
        return np.array(selected_cells[:target_n])

    @staticmethod
    def reduce_cells(cell_centers, target_n, k_neighbors=10):
        """
        Reduce the number of cells while preserving regions of higher cell density.
        
        Parameters:
        cell_centers (np.array): Array of shape (N, 3) containing cell center coordinates.
        target_n (int): Target number of cells after reduction.
        k_neighbors (int): Number of neighbors to consider for density estimation.
        
        Returns:
        np.array: Reduced array of cell centers with shape (target_n, 3).
        """
        N = len(cell_centers)
        if target_n >= N:
            return cell_centers[:target_n]  # Return all cells if target_n is greater or equal
        
        # Create a KD-tree for efficient nearest neighbor search
        tree = cKDTree(cell_centers)
        
        # Estimate local density for each cell
        distances, _ = tree.query(cell_centers, k=k_neighbors)
        local_density = 1 / np.mean(distances, axis=1)
        
        # Normalize densities to use as weights
        weights = local_density / np.sum(local_density)
        
        # Use weights to select cells
        np.random.seed(0)
        selected_indices = np.random.choice(N, size=target_n, replace=False, p=weights)
        
        return cell_centers[selected_indices]

    ### =============== ###
    ### Escape momentum ###
    ### =============== ###

    @staticmethod
    def get_halo_params(sim_folder):

        # Load halo files
        halo_params = jnp.load(
            f"{sim_folder}/halo_batch_0.6-2.0x1e+12.0_Msun_params.npy")

        # Extract halo parameters
        Rvir = halo_params[:,0]  # in kpc
        Mvir = 10**halo_params[:,1]  # in Msun
        conc = halo_params[:,2]  # unitless

        return Rvir, Mvir, conc

    @staticmethod
    def rho_crit(z, args):

        H_squared = args.H0**2 * (args.Omega_M*(1.+z)**3 + args.Omega_L) 
        rho_crit = 3.*H_squared / (8.*args.Pi*args.G)

        return rho_crit

    @staticmethod
    def scale_density_NFW(c, z, args):
        """Eqn. (2) from arXiv:1302.0288. c=r_200/r_s."""
        numer = 200 * c**3
        denom = 3 * (np.log(1+c) - (c/(1+c)))
        delta_c = numer/denom

        rho_critical = SimUtil.rho_crit(z, args)

        return rho_critical*delta_c

    @staticmethod
    def compute_p_esc_1D(x_i, R_vir, R_s, rho_0, m_nu, args):
        
        # Calculate gravitational potential at coords. x_i.
        r = np.linalg.norm(x_i, axis=-1)
        m = np.minimum(r, R_vir)
        M = np.maximum(r, R_vir)
        prefactor = -4.*args.Pi*args.G*rho_0*R_s**2
        term1 = np.log(1. + (m/R_s)) / (r/R_s)
        term2 = (R_vir/M) / (1. + (R_vir/R_s))
        potential = prefactor * (term1 - term2)

        # Escape momentum formula from Ringwald & Wong (2004).
        p_esc = np.sqrt(2*np.abs(potential)) * m_nu
        y_esc = p_esc/args.T_CNB

        return p_esc, y_esc

    @staticmethod
    def get_p_esc_1D(sim_folder, x_i, m_nu_arr, args):

        # Extract parameters from halos used in simulation folder
        Rvir, Mvir, conc = SimUtil.get_halo_params(sim_folder)
        
        # Only select DM halos used in papers
        exclude_nums = jnp.array([
            20,  # halo with missing/broken snapshot info
            23,  # halo with anomalous number densities (~0s on almost all pixels)
            24,  # anomalous "compactified into 1 cell" DM halo
            25,  # anomalous "compactified into 1 cell" DM halo
        ])

        halo_nums = jnp.array(
            [x for x in range(1, 31) if x not in exclude_nums]) - 1  # subtract 1 for 0-based indexing
        
        Rvir = Rvir[halo_nums]
        Mvir = Mvir[halo_nums]
        conc = conc[halo_nums]

        # Using get_halo_params function, so x_200 values.
        Rs = Rvir/conc
        rho0 = SimUtil.scale_density_NFW(c=conc, z=0., args=args)

        # Compute escape momentum for each halo
        p_escs_l, y_escs_l = [], []
        for m in m_nu_arr:
            p_esc, y_esc = SimUtil.compute_p_esc_1D(
                x_i, 
                Rvir*args.kpc, 
                Rs*args.kpc, 
                rho0, m, args)
            p_escs_l.append(p_esc)
            y_escs_l.append(y_esc)

        return jnp.array(p_escs_l).flatten(), jnp.array(y_escs_l).flatten()

    @staticmethod
    def compute_p_esc(x_i, R_vir, R_s, rho_0, m_nu, args):
        """
        Parameters:
        x_i: [..., 3] coordinates
        R_vir: [n_halos] virial radii
        R_s: [n_halos] scale radii
        rho_0: [n_halos] scale densities
        m_nu: [n_masses] neutrino masses
        """
        # Add axes for broadcasting
        R_vir = R_vir[:, None]  # [n_halos, 1]
        R_s = R_s[:, None]      # [n_halos, 1]
        rho_0 = rho_0[:, None]  # [n_halos, 1]
        m_nu = m_nu[None, :]    # [1, n_masses]
        
        # Calculate gravitational potential at coords. x_i
        r = jnp.linalg.norm(x_i, axis=-1)  # [...] 
        m = jnp.minimum(r[..., None, None], R_vir)  # [..., n_halos, 1]
        M = jnp.maximum(r[..., None, None], R_vir)  # [..., n_halos, 1]
        
        prefactor = -4.*args.Pi*args.G*rho_0*R_s**2  # [n_halos, 1]
        term1 = jnp.log(1. + (m/R_s)) / (r[..., None, None]/R_s)  # [..., n_halos, 1]
        term2 = (R_vir/M) / (1. + (R_vir/R_s))  # [..., n_halos, 1]
        potential = prefactor * (term1 - term2)  # [..., n_halos, 1]

        # Escape momentum formula from Ringwald & Wong (2004)
        # Broadcasting: [..., n_halos, 1] * [1, n_masses] -> [..., n_halos, n_masses]
        p_esc = jnp.sqrt(2*jnp.abs(potential)) * m_nu
        y_esc = p_esc/args.T_CNB

        return p_esc, y_esc

    @staticmethod
    def get_p_esc(sim_folder, x_i, m_nu_arr, args):
        # Extract parameters from halos used in simulation folder
        Rvir, Mvir, conc = SimUtil.get_halo_params(sim_folder)
        
        # Only select DM halos used in papers
        exclude_nums = jnp.array([20, 23, 24, 25])
        halo_nums = jnp.array([x for x in range(1, 31) if x not in exclude_nums]) - 1
        
        Rvir = Rvir[halo_nums]  # [n_halos]
        Mvir = Mvir[halo_nums]  # [n_halos]
        conc = conc[halo_nums]  # [n_halos]

        # Using get_halo_params function, so x_200 values
        Rs = Rvir/conc  # [n_halos]
        rho0 = SimUtil.scale_density_NFW(c=conc, z=0., args=args)  # [n_halos]

        # Compute escape momentum for all halos and masses at once
        p_esc, y_esc = SimUtil.compute_p_esc(
            x_i, 
            Rvir*args.kpc, 
            Rs*args.kpc, 
            rho0, 
            m_nu_arr, 
            args)
        
        return p_esc, y_esc

    ### ================== ###
    ### Earth-Sun dynamics ###
    ### ================== ###

    @staticmethod
    def calculate_fractional_day_numbers(year):
        """
        Calculate the fractional day number n for each day of the given year.
        
        :param year: The year for which to calculate the fractional day numbers
        :return: A list of tuples, each containing (date, fractional day number)
        """
        results = []
        start_date = datetime(year, 1, 1)
        
        for day in range(
            366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365):
            current_date = start_date + timedelta(days=day)
            Y, M, D = current_date.year, current_date.month, current_date.day
            
            Y_tilde = Y - 1 if M <= 2 else Y
            M_tilde = M + 12 if M <= 2 else M
            
            n = np.floor(365.25 * Y_tilde) + np.floor(30.61 * (M_tilde + 1)) + D - 730563.5
            
            results.append((current_date.strftime('%Y-%m-%d'), n))
        
        return results

    @staticmethod
    def calculate_average_ecliptic_vectors(year):
        """
        Calculate the average ecliptic unit vectors ϵ_x and ϵ_y for the given year.
        
        :param year: The year for which to calculate the average ecliptic vectors
        :return: A tuple containing (year, average ϵ_x, average ϵ_y)
        """
        # Constants for ϵ_x and ϵ_y calculations
        eps_x_0 = np.array([0.054876, -0.494109, 0.867666])
        eps_x_T = np.array([-0.024232, -0.002689, 1.546e-6])
        eps_y_0 = np.array([0.993821, 0.110992, 0.000352])
        eps_y_T = np.array([0.001316, -0.011851, 0.021267])
        
        eps_x_sum = np.zeros(3)
        eps_y_sum = np.zeros(3)
        
        # Get fractional day numbers for the year
        fractional_days = SimUtil.calculate_fractional_day_numbers(year)
        
        for _, n in fractional_days:
            # Calculate T
            T = n / 36525
            
            # Calculate ϵ_x and ϵ_y for this day
            eps_x = eps_x_0 + eps_x_T * T
            eps_y = eps_y_0 + eps_y_T * T
            
            # Add to the sum
            eps_x_sum += eps_x
            eps_y_sum += eps_y
        
        # Calculate averages
        days_in_year = len(fractional_days)
        eps_x_avg = eps_x_sum / days_in_year
        eps_y_avg = eps_y_sum / days_in_year
        
        # Normalize the average vectors
        eps_x_avg /= np.linalg.norm(eps_x_avg)
        eps_y_avg /= np.linalg.norm(eps_y_avg)
        
        return eps_x_avg, eps_y_avg

    @staticmethod
    def calculate_earth_position(year, mode="ecliptic"):
        """
        Calculate Earth's position relative to the Sun in ecliptic or galactic coordinates for each day of the year.
        
        :param year: The year for which to calculate Earth's position
        :return: A list of tuples, each containing (date, position vector)
        """
        # Constants
        e = 0.9574  # eccentricity in degrees
        e_rad = np.radians(e)

        # Get fractional day numbers and ecliptic vectors for the year
        fractional_days = SimUtil.calculate_fractional_day_numbers(year)
        eps_x, eps_y = SimUtil.calculate_average_ecliptic_vectors(year)

        # For mode == "galactic"
        b_gal_0 = np.array([5.536, -59.574, -29.811])
        b_gal_T = np.array([0.013, 0.002, 0.001])
        lambda_gal_0 = np.array([266.840, 347.340, 180.023])
        lambda_gal_T = np.array([1.397, 1.375, 1.404])

        results = []

        for date, n in fractional_days:

            # Calculate L and g
            L = np.radians(280.460 + 0.9856474 * n)
            g = np.radians(357.528 + 0.9856003 * n)

            # Calculate ecliptic longitude
            l = L + 2 * e_rad * np.sin(g) + 5/4 * e_rad**2 * np.sin(2*g)

            # Calculate Earth-Sun distance (in AU)
            r = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2*g)

            # Calculate position vector
            
            if mode == "ecliptic":
                position = r * (np.cos(l) * eps_x + np.sin(l) * eps_y)

            if mode == "galactic":

                # Calculate T
                T = n / 36525
                
                b_gal = b_gal_0 + b_gal_T * T
                l_gal = (lambda_gal_0 + lambda_gal_T * T) - 180

                position = r * np.cos(b_gal) * np.cos(l - l_gal)
            
            results.append((date, position))

        return results

    @staticmethod
    def calculate_earth_velocity(year, mode="ecliptic"):
        """
        Calculate Earth's velocity for each day of the year.
        
        :param year: The year for which to calculate Earth's velocity
        :return: A list of tuples, each containing (date, velocity vector)
        """
        # Constants
        e = 0.9574  # eccentricity in degrees
        e_rad = np.radians(e)
        u_E_avg = 29.79  # Average Earth velocity in km/s

        # Get fractional day numbers and ecliptic vectors for the year
        fractional_days = SimUtil.calculate_fractional_day_numbers(year)
        eps_x, eps_y = SimUtil.calculate_average_ecliptic_vectors(year)

        # For mode == "galactic"
        b_gal_0 = np.array([5.536, -59.574, -29.811])
        b_gal_T = np.array([0.013, 0.002, 0.001])

        results = []

        for date, n in fractional_days:
            # Calculate L and varpi
            L = np.radians(280.460 + 0.9856474 * n)
            varpi = np.radians(282.932 + 0.0000471 * n)
            
            if mode == "ecliptic":
                # Calculate velocity components
                v_x = -u_E_avg * (np.sin(L) + e_rad * np.sin(2*L - varpi))
                v_y = u_E_avg * (np.cos(L) + e_rad * np.cos(2*L - varpi))

                # Calculate velocity vector
                velocity = v_x * eps_x + v_y * eps_y
                
            if mode == "galactic":
                
                # Calculate T
                T = n / 36525
                
                b_gal = b_gal_0 + b_gal_T * T

                velocity = u_E_avg * np.cos(b_gal)

            results.append((date, velocity))

        return results

    @staticmethod
    def SunEarthGC_frame_coords_posvel(
            year, rel_vel="MW", Earth_rel_Sun=False):
        """
        Calculate coordinates and velocities in a frame aligned with the Galactic Center,
        using get_body_barycentric_posvel for more accurate Earth velocities.
        """
        # Define the Galactic Center coordinates
        gc = SkyCoord(l=0*apu.degree, b=0*apu.degree, frame='galactic')
        
        if rel_vel == "CNB":
            # Sun's velocity w.r.t. CNB in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([-0.06935679,-0.66222051,0.74609224])*369.82

        if rel_vel == "MW":
            # Sun's velocity w.r.t. MWGC in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([0.04420196,0.99859217,0.0293235])*251.19

        # Lists for Sun's positions and velocities, and Earth's velocities
        sun_positions = []
        sun_vels = []
        earth_vels = []
        earth_positions = []
        
        # Loop backwards through each day of the year
        for day in range(365, 0, -1):  
            # Time object of current day at noon UTC
            time = Time(f'{year}-12-31 12:00:00') - (365-day)*apu.day
            
            # Get Earth's position and velocity (ICRS)
            earth_pos, earth_vel = get_body_barycentric_posvel('earth', time)
            
            # Extract Earth's position values, leave in AU
            earth_xyz = np.array([
                earth_pos.x.value,
                earth_pos.y.value,
                earth_pos.z.value
            ])

            # Convert Earth's velocity from AU/day to km/s
            earth_vel_icrs = np.array([
                earth_vel.x.value,
                earth_vel.y.value,
                earth_vel.z.value
            ]) * 1731.45683633  # conversion factor from AU/day to km/s
            
            # Get Sun's position in GCRS coordinates (relative to Earth) in km
            sun_gcrs = SkyCoord(
                get_body_barycentric('sun', time), 
                frame='icrs').transform_to(GCRS(obstime=time))
            sun_vector = sun_gcrs.cartesian.xyz.to(apu.km).value
            
            # Transform the Galactic Center to GCRS
            gc_gcrs = gc.transform_to(GCRS(obstime=time))
            
            # Calculate the unit vector pointing from Earth to Galactic Center
            gc_vector = gc_gcrs.cartesian.xyz.value
            gc_unit_vector = gc_vector / np.linalg.norm(gc_vector)
            
            # Define the new coordinate system
            x_axis = gc_unit_vector
            z_axis = np.cross(x_axis, [0, 0, 1])
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Create rotation matrix
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
            # Rotate Sun's position and velocity into new coordinate system
            sun_rotated = np.dot(rotation_matrix.T, sun_vector)
            sun_vel_rotated = np.dot(rotation_matrix.T, sun_vel_Gal)
            
            # Rotate Earth's velocity into the new coordinate system (km/s)
            earth_vel_rotated = np.dot(rotation_matrix.T, earth_vel_icrs)
            
            # Calculate Earth's velocity relative to CMB in the rotated frame (km/s)
            if not Earth_rel_Sun:
                earth_vel_rotated = sun_vel_rotated + earth_vel_rotated
            
            # Append the rotated position and velocities to the lists
            sun_positions.append(sun_rotated)
            sun_vels.append(sun_vel_rotated)
            earth_vels.append(earth_vel_rotated)
            earth_positions.append(earth_xyz)
        
        #/ sun_positions: km units ; velocities: km/s units
        return np.array(sun_positions), np.array(sun_vels), np.array(earth_vels), np.array(earth_positions)

    @staticmethod
    def get_sun_pos_vel_earth_gc_frame(
            year, rel_vel="MW", Earth_rel_Sun=False):
        """
        Calculate the daily position and velocity of the Sun relative to Earth
        in a coordinate frame where:
        - Earth is at the origin (0,0,0)
        - The x-axis points AWAY from the Galactic Center (GC is in negative x direction)
        - The Galactic Center is 8 kpc away along the negative x-axis
        
        Parameters:
        -----------
        year : int
            The year for which to calculate positions
        rel_vel : str, optional
            Reference frame for Sun velocity, either "MW" (Milky Way Galactic Center) 
            or "CNB" (Cosmic Neutrino Background), default is "MW"
        Earth_rel_Sun : bool, optional
            If True, don't add Sun's velocity to Earth's velocity, default is False
            
        Returns:
        --------
        tuple: (times, positions, velocities)
            times: list of Time objects for each day (in reverse order, Dec 31 first)
            positions: array of shape (n_days, 3) with x,y,z positions in AU
                    x: away from Galactic Center
                    y: perpendicular to x in the Galactic plane
                    z: perpendicular to the Galactic plane
            velocities: array of shape (n_days, 3) with x,y,z velocities in km/s
        """
        # Determine if it's a leap year
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        n_days = 366 if is_leap_year else 365
        
        # Initialize arrays to store results
        times = []
        positions = np.zeros((n_days, 3))
        velocities = np.zeros((n_days, 3))
        
        # Define Sun's velocity vectors in the Galaxy
        if rel_vel == "CNB":
            # Sun's velocity w.r.t. CNB in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([-0.06935679, -0.66222051, 0.74609224]) * 369.82
        elif rel_vel == "MW":
            # Sun's velocity w.r.t. MWGC in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([0.04420196, 0.99859217, 0.0293235]) * 251.19
        else:
            raise ValueError("rel_vel must be either 'CNB' or 'MW'")
        
        # Define the Galactic Center direction in Galactic coordinates
        galactic_center = SkyCoord(l=0*apu.degree, b=0*apu.degree, frame='galactic')
        
        # Define the north Galactic pole
        galactic_north_pole = SkyCoord(l=0*apu.degree, b=90*apu.degree, frame='galactic')
        
        # Loop backwards through each day of the year
        for i, day in enumerate(range(n_days, 0, -1)):
            # Time object of current day at noon UTC
            t = Time(f'{year}-12-31 12:00:00') - (n_days-day)*apu.day
            times.append(t)
            
            # Get Sun and Earth barycentric positions and velocities
            sun_pos_vel = get_body_barycentric_posvel('sun', t)
            earth_pos_vel = get_body_barycentric_posvel('earth', t)
            
            # Calculate Sun's position and velocity relative to Earth
            sun_pos_rel = sun_pos_vel[0] - earth_pos_vel[0]  # Position
            sun_vel_rel = sun_pos_vel[1] - earth_pos_vel[1]  # Velocity
            
            # Convert to Earth-centered Cartesian coordinates
            sun_pos_earth = sun_pos_rel.xyz.to(apu.au).value
            sun_vel_earth = sun_vel_rel.xyz.to(apu.km/apu.s).value
            
            # Get Galactic Center direction in GCRS for this date
            gc_gcrs = galactic_center.transform_to(GCRS(obstime=t))
            gnp_gcrs = galactic_north_pole.transform_to(GCRS(obstime=t))
            
            # Define the new coordinate axes
            # X-axis points AWAY from the Galactic Center
            x_axis = -gc_gcrs.cartesian.xyz.value
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Z-axis is perpendicular to the Galactic plane
            z_axis = gnp_gcrs.cartesian.xyz.value
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # Y-axis is perpendicular to both X and Z (using cross product)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            # Ensure orthogonality by recalculating z_axis
            z_axis = np.cross(x_axis, y_axis)
            
            # Create rotation matrix to transform from GCRS to our custom frame
            rotation_matrix = np.array([x_axis, y_axis, z_axis])
            
            # Transform position and velocity to our custom Earth-centered, 
            # Galactic-Center-oriented frame
            sun_pos_custom = np.dot(rotation_matrix, sun_pos_earth)
            
            # Rotate Sun's reference velocity (CNB or MW) into this frame
            # sun_vel_rotated = np.dot(rotation_matrix, sun_vel_Gal)
            sun_vel_rotated = sun_vel_Gal
            
            # Rotate Earth's velocity into the new coordinate system (km/s)
            # earth_vel_rotated = np.dot(rotation_matrix, sun_vel_earth)
            earth_vel_rotated = sun_vel_earth
            
            # Calculate Earth's velocity relative to CNB or MW in the rotated frame (km/s)
            if not Earth_rel_Sun:
                earth_vel_rotated = sun_vel_rotated + earth_vel_rotated
            
            velocities[i] = earth_vel_rotated
            positions[i] = sun_pos_custom
        
        return times, positions, velocities

    @staticmethod
    def get_sun_earth_galactic_data(year, rel_vel="MW", Earth_rel_Sun=False):
        """
        Calculate Sun and Earth positions and velocities in the Galactic coordinate system.
        
        Parameters:
        -----------
        year : int
            The year for which to calculate data
        rel_vel : str, optional
            Reference frame for velocities, either "MW" (Milky Way Galactic Center) 
            or "CNB" (Cosmic Neutrino Background), default is "MW"
        Earth_rel_Sun : bool, optional
            If False, add Sun's velocity to Earth's relative velocity, default is False
            
        Returns:
        --------
        tuple: (times, sun_positions, sun_velocities, earth_velocities)
            times: list of Time objects for each day (in reverse order, Dec 31 first)
            sun_positions: array of shape (n_days, 3) with x,y,z positions in AU
                        Positions are relative to Earth in Galactic coordinates
                        x: toward Galactic Center (l=0°)
                        y: toward l=90° (left when facing GC)
                        z: toward North Galactic Pole (b=90°)
            sun_velocities: array of shape (n_days, 3) with x,y,z velocities in km/s
                        Sun's velocity relative to MW/CNB (same each day)
            earth_velocities: array of shape (n_days, 3) with x,y,z velocities in km/s
                        By default: Earth's velocity relative to the Sun in Galactic coordinates
                        If not Earth_rel_Sun: Earth's velocity with Sun's MW/CNB motion added
        """
        # Determine if it's a leap year
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        n_days = 366 if is_leap_year else 365
        
        # Initialize arrays to store results
        times = []
        sun_positions = np.zeros((n_days, 3))
        earth_velocities = np.zeros((n_days, 3))
        
        # Define Sun's velocity vectors in the Galaxy
        if rel_vel == "CNB":
            # Sun's velocity w.r.t. CNB in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([-0.06935679, -0.66222051, 0.74609224]) * 369.82
        elif rel_vel == "MW":
            # Sun's velocity w.r.t. MWGC in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([0.04420196, 0.99859217, 0.0293235]) * 251.19
        else:
            raise ValueError("rel_vel must be either 'CNB' or 'MW'")
        
        # Create sun_velocities array with the same velocity for each day
        sun_velocities = np.tile(sun_vel_Gal, (n_days, 1))
        
        # Loop backwards through each day of the year
        for i, day in enumerate(range(n_days, 0, -1)):
            # Time object of current day at noon UTC
            t = Time(f'{year}-12-31 12:00:00') - (n_days-day)*apu.day
            times.append(t)
            
            # Get Sun and Earth barycentric positions and velocities
            sun_pos_vel = get_body_barycentric_posvel('sun', t)
            earth_pos_vel = get_body_barycentric_posvel('earth', t)
            
            # Calculate Sun's position and velocity relative to Earth in ICRS
            sun_pos_rel = sun_pos_vel[0] - earth_pos_vel[0]
            sun_vel_rel = sun_pos_vel[1] - earth_pos_vel[1]
            
            # Create SkyCoord object for Sun's position in ICRS
            sun_icrs = SkyCoord(
                x=sun_pos_rel.x,
                y=sun_pos_rel.y,
                z=sun_pos_rel.z,
                frame='icrs',
                representation_type='cartesian'
            )
            
            # Transform to Galactic coordinates
            sun_gal = sun_icrs.transform_to('galactic')
            
            # Get Cartesian representation in Galactic frame
            sun_pos_gal = sun_gal.cartesian.xyz.to(apu.au).value
            
            # Store Sun's position relative to Earth in Galactic coordinates
            sun_positions[i] = sun_pos_gal
            
            # Calculate Earth's velocity relative to the Sun
            # Earth_vel = -Sun_vel_rel (in the Earth-Sun system)
            earth_sun_vel_rel = -(sun_vel_rel)
            
            # We need to create a SkyCoord with both position and velocity
            # First, temporarily use the Earth's position (at origin in ICRS)
            earth_icrs = SkyCoord(
                x=0*apu.au, y=0*apu.au, z=0*apu.au,
                v_x=earth_sun_vel_rel.x,
                v_y=earth_sun_vel_rel.y,
                v_z=earth_sun_vel_rel.z,
                frame='icrs',
                representation_type='cartesian',
                differential_type='cartesian'
            )
            
            # Transform to Galactic frame
            earth_gal = earth_icrs.transform_to('galactic')
            
            # Extract just the velocity components in Galactic coordinates
            earth_vel_gal_xyz = earth_gal.velocity.d_xyz.to(apu.km/apu.s).value
            
            # Add Sun's galactic motion if requested
            if not Earth_rel_Sun:
                earth_vel_gal_xyz = earth_vel_gal_xyz + sun_vel_Gal
            
            earth_velocities[i] = earth_vel_gal_xyz
        
        return times, sun_positions, sun_velocities, earth_velocities




class SimPlot():


    @staticmethod
    def healpix_side_by_side(
            map_left, map_right, dict_left, dict_right, fig_paths, args):

        fig = plt.figure(figsize=(12, 4.4))
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

        hp.newvisufunc.projview(
            map_left,
            **dict_left,
            **args.healpy_dict)

        hp.newvisufunc.projview(
            map_right,
            **dict_right,
            **args.healpy_dict)
        
        for fig_path in fig_paths:
            plt.savefig(f"{fig_path}", bbox_inches="tight")
        plt.show(); plt.close()

    @staticmethod
    def overdensity_band(
            dens_FD, dens_PF, m_arr_FD, m_arr_PF, colors, labels,
            plot_xlims, plot_ylims, plot_log, fig_paths, args, save=False):

        fig = plt.figure()
        # fig = plt.figure(figsize=(8,3))
        fig.tight_layout()
        ax = fig.add_subplot(111)

        densities = [
            dens_FD, 
            dens_PF
        ]
        mass_arr = [
            m_arr_FD, 
            m_arr_PF
        ]

        for dens, masses, color, label in zip(densities, mass_arr, colors, labels):

            # Convert to overdensities
            etas = dens / (args.N0 / args.cm**-3)

            if label == r"$f_\Delta$" or label == r"$f_{\Delta,5}$":
                etas += 1

            # Compute median and percentile ranges
            etas_med = np.median(etas, axis=0)
            etas_16 = np.percentile(etas, q=16, axis=0)
            etas_84 = np.percentile(etas, q=84, axis=0)
            etas_2p5 = np.percentile(etas, q=2.5, axis=0)
            etas_97p5 = np.percentile(etas, q=97.5, axis=0)

            # if save and color == "mediumblue":
            #     np.save(f"sim_output/SunMod_2k/etas_med.npy", etas_med)
            #     np.save(f"sim_output/SunMod_2k/etas_16.npy", etas_16)
            #     np.save(f"sim_output/SunMod_2k/etas_84.npy", etas_84)

            # print(etas.shape)

            # print(masses.shape)
            # mass_sync = np.tile(masses, (len(etas))).reshape(len(etas), 50)
            # print(mass_sync.shape)
            # for mass_arr, eta_arr in zip(mass_sync[21:22], etas[21:22]):
            # for i, (mass_arr, eta_arr) in enumerate(zip(mass_sync, etas)):
            #     ax.plot(mass_arr, eta_arr-1, alpha=0.7)
            # print(i+1)

            # """
            # Plot median of halo batch
            ax.plot(
                masses, etas_med-1, color=color, label=label, alpha=0.7)
            
            # Plot 84% C.L. band
            ax.fill_between(
                masses, etas_16-1, etas_84-1, 
                color=color, alpha=0.3, label=r"$68\%$ C.L.")
            
            # if label == r"MW + $f_\Delta$":
            # Plot 97.5% C.L. band
                # ax.fill_between(
                #     masses, etas_2p5-1, etas_97p5-1, 
                #     color=color, alpha=0.2, label=r"2.5 $-$ 97.5 % C.L.")
            # """


        ### ============ ###
        ### Plot styling ###
        ### ============ ###

        if plot_xlims is not None:
            ax.set_xlim(plot_xlims[0], plot_xlims[1])

        if plot_ylims is not None:
            ax.set_ylim(plot_ylims[0], plot_ylims[1])

        if plot_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$')

            # 1+10^X styled y-axis labels, to easily see order of overdensities
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(Utils.y_fmt))

        else:
            ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0} - 1$')

        # Define a function to round the tick labels
        def round_tick_labels(x, pos):
            return f'{x:.2f}'

        # Apply the formatter to the x-axis and y-axis
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(round_tick_labels))


        # ax.set_title(f"Monopole perturbation")
        ax.set_xlabel(r'$m_{\nu}$ [eV]')
        plt.grid(**args.grid_dict)
        # ax.legend(loc='upper left', prop={"size":18})
        ax.legend(
            loc='upper left', 
            bbox_to_anchor=(0.025, 0.475, 1.0, 0.5), 
            prop={"size":20}, 
            borderaxespad=0.)

        for fig_path in fig_paths:
            plt.savefig(f"{fig_path}", bbox_inches='tight')


    @staticmethod
    def overdensity_band_inset(
        dens_FD, dens_PF, m_arr_FD, m_arr_PF, colors, labels,
        plot_xlims, plot_ylims, plot_log, fig_paths,
        inset_xlims, inset_ylims, inset_position, inset_size,
        args):

        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(111)

        # Create the inset axes with manual positioning and sizing
        axins = ax.inset_axes(inset_position + inset_size)

        densities = [
            dens_FD, 
            dens_PF
        ]
        mass_arr = [
            m_arr_FD, 
            m_arr_PF
        ]

        for dens, masses, color, label in zip(densities, mass_arr, colors, labels):
            # Convert to overdensities
            etas = dens / (args.N0 / args.cm**-3)

            if label == r"$f_\Delta$":
                etas += 1

            # Compute median and percentile ranges
            etas_med = np.median(etas, axis=0)
            etas_16 = np.percentile(etas, q=16, axis=0)
            etas_84 = np.percentile(etas, q=84, axis=0)

            # Plot on main axes
            ax.plot(masses, etas_med-1, color=color, label=label, alpha=0.7)
            ax.fill_between(
                masses, etas_16-1, etas_84-1, 
                color=color, alpha=0.3, label=r"$68\%$ C.L.")

        ### Plot styling ###
        if plot_xlims is not None:
            ax.set_xlim(plot_xlims[0], plot_xlims[1])

        if plot_ylims is not None:
            ax.set_ylim(plot_ylims[0], plot_ylims[1])

        if plot_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(Utils.y_fmt))
        else:
            ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0} - 1$')

        def round_tick_labels(x, pos):
            return f'{x:.2f}'

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(round_tick_labels))

        ax.set_xlabel(r'$m_{\nu}$ [eV]')
        plt.grid(**args.grid_dict)

        # Move legend to top right
        ax.legend(loc='upper right', prop={"size":18})


        ### Inset plot ###
        
        # Convert to overdensities
        seed_nr = 5
        dens_PF_seed5 = dens_PF.reshape(26, 10, 5)[:,seed_nr,:]
        print(dens_PF_seed5.shape)
        etas_seed5 = (dens_PF_seed5 / (args.N0 / args.cm**-3))

        # Compute median and percentile ranges
        etas_med_seed5 = np.median(etas_seed5, axis=0)
        etas_16_seed5 = np.percentile(etas_seed5, q=16, axis=0)
        etas_84_seed5 = np.percentile(etas_seed5, q=84, axis=0)

        etas_FD = dens_FD / (args.N0 / args.cm**-3)
        etas_FD_med = np.median(etas_FD, axis=0)
        etas_16_FD = np.percentile(etas_FD, q=16, axis=0)
        etas_84_FD = np.percentile(etas_FD, q=84, axis=0)

        # Plot FD
        axins.plot(m_arr_FD, etas_FD_med-1, color=colors[0])
        axins.fill_between(
            m_arr_FD, etas_16_FD-1, etas_84_FD-1, 
            color=colors[0], alpha=0.3, label=r"$68\%$ C.L.")

        # Plot PF
        axins.plot(
            m_arr_PF, etas_med_seed5-1, 
            color=colors[1], label=labels[1], alpha=0.7)
        axins.fill_between(
            m_arr_PF, etas_16_seed5-1, etas_84_seed5-1, 
            color=colors[1], alpha=0.3, label=r"$68\%$ C.L.")
        
        axins.text(0.05, 0.6, r"Specific $f_{\Delta,5}$", color="black")

        axins.set_xlim(inset_xlims[0], inset_xlims[1])
        axins.set_ylim(inset_ylims[0], inset_ylims[1])
        # axins.set_xscale('linear')
        # axins.set_yscale('linear')
        axins.grid(**args.grid_dict)
        # axins.legend(loc='lower right', prop={"size":8})
        # ax.indicate_inset_zoom(axins, edgecolor="black")

        for fig_path in fig_paths:
            plt.savefig(f"{fig_path}", bbox_inches='tight')

    @staticmethod
    def overdensity_band_decay(
        dens_FD, m_arr_FD, colors, labels,
        plot_xlims, plot_ylims, plot_log, fig_paths, args):

        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(111)

        densities = [
            dens_FD, 
        ]
        mass_arr = [
            m_arr_FD, 
        ]

        for dens, masses, color, label in zip(densities, mass_arr, colors, labels):

            # Convert to overdensities
            etas = dens / (args.N0 / args.cm**-3)

            # Compute median and percentile ranges
            etas_med = np.median(etas, axis=0)
            etas_16 = np.percentile(etas, q=16, axis=0)
            etas_84 = np.percentile(etas, q=84, axis=0)
            etas_2p5 = np.percentile(etas, q=2.5, axis=0)
            etas_97p5 = np.percentile(etas, q=97.5, axis=0)

            # Plot median of halo batch
            ax.plot(
                masses, etas_med-1, color=color, label=label, alpha=0.7)
            
            # Plot 84% C.L. band
            ax.fill_between(
                masses, etas_16-1, etas_84-1, 
                color=color, alpha=0.3, label=r"16 $-$ 84 % C.L.")
            
            """
            # Plot 97.5% C.L. band
            ax.fill_between(
                masses, etas_2p5-1, etas_97p5-1, 
                color=color, alpha=0.2, label=r"2.5 $-$ 97.5 % C.L.")
            """


        ### ============ ###
        ### Plot styling ###
        ### ============ ###

        if plot_xlims is not None:
            ax.set_xlim(plot_xlims[0], plot_xlims[1])

        if plot_ylims is not None:
            ax.set_ylim(plot_ylims[0], plot_ylims[1])

        if plot_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$', fontsize=16)

            # 1+10^X styled y-axis labels, to easily see order of overdensities
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(Utils.y_fmt))

        else:
            ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0} - 1$', fontsize=16)


        ax.set_title(f"Monopole perturbation")
        ax.set_xlabel(r'$m_{\nu}$ [eV]', fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        plt.grid(**args.grid_dict)
        ax.legend(loc='upper left', prop={"size":12})

        
        for fig_path in fig_paths:
            plt.savefig(f"{fig_path}", bbox_inches='tight')

        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(111)

    @staticmethod
    def overdensity_evolution():
        ...

        # Compute number densities for different final z
        Physics.number_density

    @staticmethod
    def bulk_velocity_band_Guillermo(
        sim_dir, halo_num, m_arr, pix_sr, Deltas, colors, labels,
        plot_xlims, plot_ylims, fig_paths, args):

        # Load velocities for halos up to halo_num
        vectors = SimData.load_vectors(sim_dir=sim_dir, halo_num=halo_num)
        vels = vectors[..., 3:]

        # Momenta
        p_arr, _ = Physics.velocities_to_momenta_ND_halo_0th_axis(vels, m_arr, args)
        p_vecs, p_arr, _ = Physics.velocities_to_momentum_vec_ND_halo_0th_axis(vels, m_arr, args)
        # (halo_num, len(m_arr), 768, 1000, 2, 3)
        p_init = p_arr[..., 0].reshape(halo_num, len(m_arr), -1)
        p_back = p_arr[...,-1].reshape(halo_num, len(m_arr), -1)

        # Sort momenta in ascending order
        ind = p_init.argsort(axis=-1)
        p_init_sort = jnp.take_along_axis(p_init, ind, axis=-1)
        p_back_sort = jnp.take_along_axis(p_back, ind, axis=-1)
        Deltas_sort = jnp.take_along_axis(Deltas, ind, axis=-1)

        # Neutrino densities
        n_nus = Physics.number_density(p_init_sort, p_back_sort, pix_sr, args)

        # Phase-space quantities
        f_FD = Physics.Fermi_Dirac(p_init_sort, args)
        f_Delta = Physics.Fermi_Dirac(p_back_sort, args)
        f_PF = Physics.Fermi_Dirac_Delta(p_back_sort, Deltas_sort, args)
        f_diffs = [(f_Delta-f_FD), (f_PF-f_FD)]

        # Relative velocities from Planck (2018)
        v_Sun_CMB = 369.82*args.km/args.s  # velocity of Sun w.r.t. CMB
        l_Sun_CMB = 264.021
        b_Sun_CMB = 48.253
        v_GC_CMB = 565*args.km/args.s  # velocity of Galactic Center w.r.t. CMB
        vmag_Sun_CMB = jnp.linalg.norm(SimUtil.Galactic_velocity_to_Cartesian_vector(
            v_Sun_CMB, l_Sun_CMB, b_Sun_CMB))/(args.km/args.s)
        

        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(111)

        for f_diff, color, label in zip(f_diffs, colors, labels):

            # Compute bulk velocity
            y = p_init_sort**4 * f_diff  # dlog integrand
            x = p_init_sort
            v_raw = trap(y, jnp.log(x), axis=-1)

            # Normalisation with energy density
            rho_nu = m_arr[None, :] * args.N0
            v_norm = rho_nu*(args.km/args.s)

            # Multiply by constants, solid angles and convert to km/s.
            v_km_div_s = pix_sr * args.g_nu/((2*args.Pi)**3) * v_raw / v_norm

            # Accounting for Sun's motion w.r.t. CMB (i.e. CNB) rest frame
            v_Sun_frame = v_km_div_s# - vmag_Sun_CMB

            # Compute median and percentile ranges
            pert_mags = v_Sun_frame
            pert_med = np.median(pert_mags, axis=0)
            pert_16 = np.percentile(pert_mags, q=16, axis=0)
            pert_84 = np.percentile(pert_mags, q=84, axis=0)
            pert_2p5 = np.percentile(pert_mags, q=2.5, axis=0)
            pert_97p5 = np.percentile(pert_mags, q=97.5, axis=0)

            # Plot median of halo batch
            ax.plot(
                m_arr, pert_med, color=color, label=label, alpha=0.7)

            # Plot 84% C.L. band
            ax.fill_between(
                m_arr, pert_16, pert_84, 
                color=color, alpha=0.3, label=r"16 $-$ 84 % C.L.")

            # Plot 97.5% C.L. band
            # ax.fill_between(
            #     m_arr, pert_2p5, pert_97p5, 
            #     color='blue', alpha=0.2, label=r"2.5 $-$ 97.5 % C.L.")


        ### ============ ###
        ### Plot styling ###
        ### ============ ###

        if plot_xlims is not None:
            ax.set_xlim(plot_xlims[0], plot_xlims[1])

        if plot_ylims is not None:
            ax.set_ylim(plot_ylims[0], plot_ylims[1])

        ax.set_xlabel(r'$m_{\nu}$ [eV]', fontsize=16)
        ax.set_ylabel(r'$v_{\nu}$ [km/s]', fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        plt.grid(**args.grid_dict)
        ax.legend(loc='upper left', prop={'size':12})

        for fig_path in fig_paths:
            plt.savefig(f"{fig_path}", bbox_inches='tight')

    @staticmethod
    def bulk_velocity_band(
        sim_dir, halo_num, m_arr, pix_sr, Deltas, colors, labels,
        plot_xlims, plot_ylims, fig_dir, args):

        # Velocity unit conversion to km/s
        v_unit = (Params.kpc/Params.s)/(Params.km/Params.s)

        # Load velocities for halos up to halo_num
        vectors = SimData.load_vectors(sim_dir=sim_dir, halo_num=halo_num)
        vels = vectors[..., 3:]
        v_init = jnp.reshape(vels[..., 0, :], (halo_num, -1, 3))*v_unit

        # Momenta
        p_arr, _ = Physics.velocities_to_momenta_ND_halo_0th_axis(vels, m_arr, args)
        p_init = p_arr[..., 0]
        p_back = p_arr[...,-1]
        print(p_init.shape)

        # Neutrino densities to cross-check methods
        n_norm = args.N0/args.cm**-3
        n_nus = Physics.number_density(p_init, p_back, pix_sr, args)/n_norm
        # print(jnp.sum(n_nus, axis=-1))

        # \bar{f}) and g distributions as they defined it
        g_dist = Physics.Fermi_Dirac(p_init, args)
        f_dist = Physics.Fermi_Dirac(p_back, args)

        # Phase-space weight
        weight = jnp.reshape(jnp.abs(f_dist-g_dist)/g_dist, (halo_num, len(m_arr), -1))

        # Compute mean weight
        w_mean = jnp.mean(weight, axis=-1)
        print(1+w_mean)

        # Without abs
        # weight = jnp.reshape((f_dist-g_dist)/g_dist, (halo_num, len(m_arr), -1))
        # w_mean = jnp.mean(weight, axis=-1)
        # print(1+w_mean)

        # Compute weighted velocity
        v_mean = jnp.mean(
            jnp.swapaxes(v_init[:, None, ...], 2, 3) * weight[..., None, :], 
            axis=-1)

        # Velocity perturbations as momentum density divided by the density
        v_pert = jnp.sqrt(jnp.sum(v_mean**2, axis=-1))/(1+w_mean)
        print(v_pert.shape)

        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(111)

        pert_mags = v_pert

        # Compute median and percentile ranges
        pert_med = np.median(pert_mags, axis=0)
        pert_16 = np.percentile(pert_mags, q=16, axis=0)
        pert_84 = np.percentile(pert_mags, q=84, axis=0)
        pert_2p5 = np.percentile(pert_mags, q=2.5, axis=0)
        pert_97p5 = np.percentile(pert_mags, q=97.5, axis=0)

        # Plot median of halo batch
        ax.plot(
            m_arr, pert_med, color='blue', label="test", alpha=0.7)

        # Plot 84% C.L. band
        ax.fill_between(
            m_arr, pert_16, pert_84, 
            color='blue', alpha=0.3, label=r"16 $-$ 84 % C.L.")

        # Plot 97.5% C.L. band
        ax.fill_between(
            m_arr, pert_2p5, pert_97p5, 
            color='blue', alpha=0.2, label=r"2.5 $-$ 97.5 % C.L.")


        ### ============ ###
        ### Plot styling ###
        ### ============ ###

        if plot_xlims is not None:
            ax.set_xlim(plot_xlims[0], plot_xlims[1])

        if plot_ylims is not None:
            ax.set_ylim(plot_ylims[0], plot_ylims[1])

        ax.set_title(f"Dipole perturbation")
        ax.set_xlabel(r'$m_{\nu}$ [eV]', fontsize=16)
        ax.set_ylabel(r'$v_{\nu}$ [km/s]', fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        plt.grid(**args.grid_dict)
        ax.legend(loc='upper left', prop={'size':12})

        plt.savefig(f"{fig_dir}/dipole_perturbation.pdf", bbox_inches="tight")
        plt.savefig(f"{fig_dir}/dipole_perturbation.png", bbox_inches="tight")
        """
        """

    @staticmethod
    def skymaps_CNBxDM(
        nu_mass, CNB_dens, DM_halo_pos, init_xyz, fig_paths, args, plot=True):

        Nside = hp.pixelfunc.get_nside(CNB_dens)
        Npix = hp.pixelfunc.nside2npix(Nside)
        
        # Rotate DM halo (based on observer position), s.t. CoP is at origin
        # (middle) in healpy skymap
        DM_halo_rot, obs_rot = SimUtil.rotate_particles(DM_halo_pos, init_xyz)

        # Create healpix skymap with these new DM coords
        DM_halo_skymap = SimUtil.DM_pos_to_skymap(Nside, DM_halo_rot, obs_rot)

        # Rotate neutrino density skymap
        CNB_skymap_rot = SimUtil.rotate_skymap(CNB_dens, init_xyz)

        # Interpolate pixels without values due to rotation
        CNB_skymap_int = SimUtil.interpolate_skymap(CNB_skymap_rot)

        # Convert to overdensities
        CNB_skymap = CNB_skymap_int / (args.N0/Npix/args.cm**-3)


        ### ------------------------------ ###
        ### Plot parameters for CNB skymap ###
        ### ------------------------------ ###

        # Make center value to be 1 (no overdensity).
        mid = 1.
        divnorm = mcolors.TwoSlopeNorm(vcenter=mid)
        
        # Plot style healpy dictionary for left skymap
        eta_cmap = "coolwarm"
        dict_CNB = dict(
            title=fr"Overdensities of $m_\nu={nu_mass}$ eV at z=0", 
            unit=r"$n_{\nu, pix} / n_{\nu, pix, 0}$",
            cmap=eta_cmap,
            override_plot_properties={"cbar_pad": 0.1},
            cbar_ticks=[np.min(CNB_skymap), mid, np.max(CNB_skymap)],
            norm=divnorm,
            sub=121)


        ### ---------------------------------- ###
        ### Plot parameters for DM halo skymap ###
        ### ---------------------------------- ###

        # Plot style healpy dictionary for right skymap
        DM_cmap = "Purples"
        dict_DM = dict(
            title=fr"Dark matter halo", 
            unit=f"particle count",
            cmap=DM_cmap,
            cbar=True,
            norm="log",
            override_plot_properties={"cbar_pad": 0.1, "cbar_label_pad": 1},
            sub=122)


        ### ------------ ###
        ### Plot skymaps ###
        ### ------------ ###

        if plot:
            SimPlot.healpix_side_by_side(
                map_left=CNB_skymap,
                map_right=DM_halo_skymap,
                dict_left=dict_CNB,
                dict_right=dict_DM,
                fig_paths=fig_paths,
                args=args)  
        

        return CNB_skymap, DM_halo_skymap

    @staticmethod
    def skymaps_CNBxDM_decay(
        nu_mass, CNB_dens, DM_halo_pos, init_xyz, fig_paths, args, plot=True):

        Nside = hp.pixelfunc.get_nside(CNB_dens)
        Npix = hp.pixelfunc.nside2npix(Nside)
        
        # Rotate DM halo (based on observer position), s.t. CoP is at origin
        # (middle) in healpy skymap
        DM_halo_rot, obs_rot = SimUtil.rotate_particles(DM_halo_pos, init_xyz)

        # Create healpix skymap with these new DM coords
        DM_halo_skymap = SimUtil.DM_pos_to_skymap(Nside, DM_halo_rot, obs_rot)

        # Rotate neutrino density skymap
        CNB_skymap_rot = SimUtil.rotate_skymap(CNB_dens, init_xyz)

        # Interpolate pixels without values due to rotation
        CNB_skymap_int = SimUtil.interpolate_skymap(CNB_skymap_rot)

        # Convert to overdensities
        CNB_skymap = CNB_skymap_int / (args.N0/Npix/args.cm**-3)


        ### ------------------------------ ###
        ### Plot parameters for CNB skymap ###
        ### ------------------------------ ###

        # Make center value to be 1 (no overdensity).
        mid = 1.
        divnorm = mcolors.TwoSlopeNorm(vcenter=mid)
        
        # Plot style healpy dictionary for left skymap
        eta_cmap = "magma"
        dict_CNB = dict(
            title=r"Overdensities of $n_{\nu_3}$ at z=0", 
            unit=r"$n_{\nu, pix} / n_{\nu, pix, 0}$",
            cmap=eta_cmap,
            override_plot_properties={"cbar_pad": 0.1},
            cbar_ticks=[0., 1., 10.],
            norm=divnorm,
            sub=121)


        ### ---------------------------------- ###
        ### Plot parameters for DM halo skymap ###
        ### ---------------------------------- ###

        # Plot style healpy dictionary for right skymap
        DM_cmap = "Purples"
        dict_DM = dict(
            title=fr"Dark matter halo", 
            unit=f"particle count",
            cmap=DM_cmap,
            cbar=True,
            norm="log",
            override_plot_properties={"cbar_pad": 0.1, "cbar_label_pad": 1},
            sub=122)


        ### ------------ ###
        ### Plot skymaps ###
        ### ------------ ###

        if plot:
            SimPlot.healpix_side_by_side(
                map_left=CNB_skymap,
                map_right=DM_halo_skymap,
                dict_left=dict_CNB,
                dict_right=dict_DM,
                fig_paths=fig_paths,
                args=args)  
        

        return CNB_skymap, DM_halo_skymap

    @staticmethod
    def skymaps_percXdm(
        nu_mass, perc_map, DM_halo_pos, init_xyz, fig_paths, args, plot=True):

        Nside = hp.pixelfunc.get_nside(perc_map)

        # Rotate DM halo (based on observer position), s.t. CoP is at origin
        # (middle) in healpy skymap
        DM_halo_rot, obs_rot = SimUtil.rotate_particles(DM_halo_pos, init_xyz)

        # Create healpix skymap with these new DM coords
        DM_halo_skymap = SimUtil.DM_pos_to_skymap(Nside, DM_halo_rot, obs_rot)


        ### ------------------------------ ###
        ### Plot parameters for CNB skymap ###
        ### ------------------------------ ###
        
        # Plot style healpy dictionary for left skymap
        dict_perc = dict(
            title=fr"Bound neutrino percentage of $m_\nu={nu_mass}$ eV", 
            unit='%',
            cmap=cc.cm.CET_L7,
            sub=121)


        ### ---------------------------------- ###
        ### Plot parameters for DM halo skymap ###
        ### ---------------------------------- ###

        # Plot style healpy dictionary for right skymap
        dict_DM = dict(
            title=fr"Dark matter halo", 
            unit=f"particle count",
            cmap="Purples",
            cbar=True,
            norm="log",
            override_plot_properties={"cbar_pad": 0.1, "cbar_label_pad": 1},
            sub=122)


        ### ------------ ###
        ### Plot skymaps ###
        ### ------------ ###

        if plot:
            SimPlot.healpix_side_by_side(
                map_left=perc_map,
                map_right=DM_halo_skymap,
                dict_left=dict_perc,
                dict_right=dict_DM,
                fig_paths=fig_paths,
                args=args)  
        

        return perc_map, DM_halo_skymap

    @staticmethod
    def temp_deltas_skymaps(
        m_Cl, q_select, Cl_qi, seed, Nside, z4_val, Cl_dir, fig_paths, args):

        # Load Cl's
        Cls_z4 = jnp.load(f"{Cl_dir}/Cls_z=4_m={m_Cl}eV.npy")
        Cls_z0 = jnp.load(f"{Cl_dir}/Cls_z=0_m={m_Cl}eV.npy")
        # (q momentum bins, l multipoles) = (50, 20)

        # Add the monopole term as 0 and choose a momentum to display
        Cl_qi_z4 = jnp.insert(Cls_z4, 0, 0, axis=1)[Cl_qi, :]
        Cl_qi_z0 = jnp.insert(Cls_z0, 0, 0, axis=1)[Cl_qi, :]
        # (q momentum bins, l multipoles + 1) = (50, 21)

        # Create temp. flucts. skymaps from Cls, fixed with a seed
        np.random.seed(seed)
        Deltas_z4 = hp.sphtfunc.synfast(Cl_qi_z4, nside=Nside, lmax=None, pol=False)
        np.random.seed(seed)  #note: seed needs to reset
        Deltas_z0 = hp.sphtfunc.synfast(Cl_qi_z0, nside=Nside, lmax=None, pol=False)

        # Show Deltas in percentages of T_CNB(z)
        z4_unit = Primordial.uK/(Params.T_CNB*(1+z4_val))
        z0_unit = Primordial.uK/Params.T_CNB
        # z4_unit = Primordial.uK/Params.K
        # z0_unit = Primordial.uK/Params.K
        Deltas_z4 *= z4_unit
        Deltas_z0 *= z0_unit
        # (Npix)

        Delta_cmap = "inferno"
        dict_z4 = dict(
            title=fr"Temp. flucts. skymap: $m_\nu={m_Cl}$ eV, q={q_select}, z=4", 
            unit=r"$\delta T_\nu / T_\nu (z=4)$",
            cmap=Delta_cmap,
            sub=121)
        dict_z0 = dict(
            title=fr"Temp. flucts. skymap: $m_\nu={m_Cl}$ eV, q={q_select}, z=0", 
            unit=r"$\delta T_\nu / T_\nu (z=0)$",
            cmap=Delta_cmap,
            sub=122)

        SimPlot.healpix_side_by_side(
            Deltas_z4, Deltas_z0, dict_z4, dict_z0, 
            fig_paths, args)
        
        return Deltas_z4, Deltas_z0

    @staticmethod
    def temp_deltas_skymaps_qavg(
        m_Cl, seed, Nside, z4_val, Cl_dir, fig_paths, args, plot=True):

        # Load q-averaged Cl's for z=4 and z=0 (closest values)
        idx_z4 = jnp.argmin(jnp.abs(Primordial.z_cuts - 0.0))
        idx_z0 = jnp.argmin(jnp.abs(Primordial.z_cuts - 4.0))
        z_cut_z4 = Primordial.z_cuts[idx_z4]
        z_cut_z0 = Primordial.z_cuts[idx_z0]

        Cls_q_avg_pre_z4 = jnp.array(
            jnp.load(f"{Cl_dir}/Cls_q_averaged_z_cut={z_cut_z4:.2f}_m={m_Cl}eV.npy"))
        Cls_q_avg_pre_z0 = jnp.array(
            jnp.load(f"{Cl_dir}/Cls_q_averaged_z_cut={z_cut_z0:.2f}_m={m_Cl}eV.npy"))
        # (l multipoles,) = (20,)

        # Add the monopole term as 0 and choose a momentum to display
        Cl_q_avg_z4 = jnp.insert(Cls_q_avg_pre_z4, 0, 0, axis=-1)
        Cl_q_avg_z0 = jnp.insert(Cls_q_avg_pre_z0, 0, 0, axis=-1)
        # (l multipoles + 1,) = (21,)

        # Create temp. flucts. skymaps from Cls, fixed with a seed
        np.random.seed(seed)
        Deltas_z4 = hp.sphtfunc.synfast(Cl_q_avg_z4, nside=Nside, lmax=None, pol=False)
        np.random.seed(seed)  #note: seed needs to reset
        Deltas_z0 = hp.sphtfunc.synfast(Cl_q_avg_z0, nside=Nside, lmax=None, pol=False)

        # Show Deltas in percentages of T_CNB(z)
        z4_unit = Primordial.uK/(Params.T_CNB*(1+z4_val))
        z0_unit = Primordial.uK/Params.T_CNB
        Deltas_z4 *= z4_unit
        Deltas_z0 *= z0_unit
        # (Npix,)

        Delta_cmap = "inferno"
        dict_z4 = dict(
            title=fr"Temp. flucts. skymap: $m_\nu={m_Cl}$ eV, q-averaged, z=4", 
            unit=r"$\delta T_\nu / T_\nu (z=4)$",
            cmap=Delta_cmap,
            sub=121)
        dict_z0 = dict(
            title=fr"Temp. flucts. skymap: $m_\nu={m_Cl}$ eV, q-averaged, z=0", 
            unit=r"$\delta T_\nu / T_\nu (z=0)$",
            cmap=Delta_cmap,
            sub=122)

        if plot:
            SimPlot.healpix_side_by_side(
                Deltas_z4, Deltas_z0, dict_z4, dict_z0, 
                fig_paths, args)
        
        return Deltas_z4, Deltas_z0

    @staticmethod
    def Delta_vs_z_cut(
            m_arr, massless, m_colors, seed, Nside, Cl_dir, fig_dir, args):
        
        fig = plt.figure(figsize=(8,4))
        fig.tight_layout()
        ax = fig.add_subplot(111)

        for i, (m_val, m_col) in enumerate(zip(m_arr, m_colors)):
            
            if massless and i == 4:
                # Load Cl for "massless" neutrino, i.e. 1e-5 eV
                Cls_q_avg_pre = jnp.array([
                    jnp.load(f"{Cl_dir}/Cls_q_averaged_z_cut={z_cut:.2f}_m=1e-05eV.npy")
                    for z_cut in Primordial.z_cuts])
                # (z_cuts, 20)
                label = fr"$m_\nu = 10^{{-5}}$ eV"
            else:
                # Load q-averaged Cl's for all z_cuts
                Cls_q_avg_pre = jnp.array([
                    jnp.load(f"{Cl_dir}/Cls_q_averaged_z_cut={z_cut:.2f}_m={m_val}eV.npy")
                    for z_cut in Primordial.z_cuts])
                # (z_cuts, 20)
                label = fr"$m_\nu = {m_val}$ eV"

            # Add the monopole term as 0 and choose a momentum to display
            Cls_q_avg = jnp.insert(Cls_q_avg_pre, 0, 0, axis=1)
            # (z_cuts, 21)

            # Generate temp. fluctuations (map) from Cls
            T_del_avg_l = []
            for Cls, z_cut in zip(Cls_q_avg, Primordial.z_cuts):

                np.random.seed(seed)
                Deltas = hp.sphtfunc.synfast(Cls, nside=Nside, lmax=None, pol=False)
                Deltas *= Primordial.uK/(Params.T_CNB * (1+z_cut))

                T_del_avg = (jnp.abs(jnp.min(Deltas)) + jnp.max(Deltas)) / 2
                T_del_avg_l.append(T_del_avg)

            ax.semilogx(
                1+Primordial.z_cuts, 3*jnp.array(T_del_avg_l), 
                label=label, color=m_col)
            ax.set_xlabel(fr"$1 + z$")
            # ax.set_ylabel(fr"$\delta_\nu = 3 \Delta T \, / \, T_\nu (z)$")
            ax.set_ylabel(fr"$\delta_\nu (z)$")

        # Our original z_back=4
        ax.axvline(4+1, color="red", ls="dotted")
        ax.text(4.9, 0.8, r"$z_{\rm sim}=4$", color='red', rotation=90, verticalalignment='center', fontsize=18)

        # ax.axvline(0.15+1, color="red", ls="dotted")
        # ax.text(1.14, 0.9, 'z=0.15', color='red', rotation=90, verticalalignment='center')

        # Elbers z_back=31
        # ax.axvline(31+1, color="red", ls="dotted")
        # ax.text(31+1e-1, 0.4, 'z=31', color='red', rotation=90, verticalalignment='center')

        # Major and minor locators
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(Utils.int_labels))
        ax.xaxis.set_minor_formatter(ticker.FuncFormatter(Utils.int_labels))

        ax.invert_xaxis()
        ax.set_xlim(33, 1)

        plt.grid(**args.grid_dict)
        plt.legend(loc="upper left", prop={"size":14})
        plt.savefig(f"{fig_dir}/Delta_vs_z_cuts.pdf", bbox_inches="tight")
        plt.savefig(f"{fig_dir}/Delta_vs_z_cuts.png", bbox_inches="tight")
        plt.show(); plt.close()

    @staticmethod
    def power_spectra_FDxPF(
        m_pick, pix_dens_FD, pix_dens_PF, y_lims, sim_dir, fig_dir, args, _1Halo=False, Elbers=False, Tully=True):
        
        m_specs = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*args.eV
        m_range = jnp.load(f"{sim_dir}/neutrino_massrange_eV.npy")

        mPF_i = jnp.abs(m_specs - m_pick).argmin()
        # mFD_i = jnp.abs(m_range - m_pick).argmin()
        mFD_i = jnp.abs(m_specs - m_pick).argmin()

        halo_num = len(pix_dens_FD)

        power_spectra_FD_l = []
        power_spectra_PF_l = []
        for halo_i in range(halo_num):

            m_nu = m_specs[mFD_i]

            FD_skymap = pix_dens_FD[halo_i, mFD_i]
            PF_skymap = pix_dens_PF[halo_i, mPF_i]

            # Convert maps of neutrino densities to temperatures.
            cm3 = args.cm**3
            FD_Tmap = np.cbrt((FD_skymap*(1/cm3)*4*args.Pi**2) / (args.g_nu*3*zeta(3)))
            PF_Tmap = np.cbrt((PF_skymap*(1/cm3)*4*args.Pi**2) / (args.g_nu*3*zeta(3)))

            T_unit = args.muK
            FD_Tmap /= T_unit
            PF_Tmap /= T_unit

            # Compute power spectrum of number density all-sky map.
            cl_FD = hp.sphtfunc.anafast(FD_Tmap, lmax=None)
            cl_PF = hp.sphtfunc.anafast(PF_Tmap, lmax=None)

            ell = jnp.arange(len(cl_FD))

            # Unit to compare to similar figures in literature.
            # scaling = Primordial.Cl_scaling
            scaling = 1  # if Tmaps already in muK
            
            power_spectrum_PF = ell * (ell+1) * cl_PF * scaling / (2*args.Pi)
            power_spectrum_FD = ell * (ell+1) * cl_FD * scaling / (2*args.Pi)
            
            power_spectra_FD_l.append(power_spectrum_FD)
            power_spectra_PF_l.append(power_spectrum_PF)


        # Convert lists to arrays
        power_spectra_FD = jnp.array(power_spectra_FD_l)
        power_spectra_PF = jnp.array(power_spectra_PF_l)
        # power_spectra_FD = savgol_filter(jnp.array(power_spectra_FD_l), 4, 2)
        # power_spectra_PF = savgol_filter(jnp.array(power_spectra_PF_l), 4, 2)

        # Compute median and percentile ranges for Fermi-Dirac spectra
        power_median_FD = jnp.median(power_spectra_FD, axis=0)
        power_perc2p5_FD = jnp.percentile(power_spectra_FD, q=2.5, axis=0)
        power_perc97p5_FD = jnp.percentile(power_spectra_FD, q=97.5, axis=0)
        power_perc16_FD = jnp.percentile(power_spectra_FD, q=16, axis=0)
        power_perc84_FD = jnp.percentile(power_spectra_FD, q=84, axis=0)

        # Compute median and percentile ranges for Primordial Fluctuations spectra
        power_median_PF = jnp.median(power_spectra_PF, axis=0)
        power_perc2p5_PF = jnp.percentile(power_spectra_PF, q=2.5, axis=0)
        power_perc97p5_PF = jnp.percentile(power_spectra_PF, q=97.5, axis=0)
        power_perc16_PF = jnp.percentile(power_spectra_PF, q=16, axis=0)
        power_perc84_PF = jnp.percentile(power_spectra_PF, q=84, axis=0)


        fig = plt.figure(figsize =(6, 4))
        fig.tight_layout()
        ax1 = fig.add_subplot(111)

        FD_color = "mediumblue"
        ax1.semilogy(
            ell, power_median_FD, color=FD_color, 
            label=r"MW + $f_{\rm FD}$")
        # ax1.fill_between(
        #     ell, power_perc2p5_FD, power_perc97p5_FD, 
        #     color=FD_color, alpha=0.15, label='2.5-97.5 % C.L.')

        if not _1Halo:
            ax1.fill_between(
                ell, power_perc16_FD, power_perc84_FD, 
                color=FD_color, alpha=0.25, label=r"$68\%$ C.L.")

        PF_color = "red"
        ax1.semilogy(
            ell, power_median_PF, color=PF_color, 
            label=r"MW + $f_{\Delta,{\rm all}}$", ls="solid")
        ax1.fill_between(
            ell, power_perc16_PF, power_perc84_PF, 
            color=PF_color, alpha=0.3, label=r"$68\%$ C.L.")
            # color=PF_color, alpha=0.3, label=r'$16\% - 84\%$ C.L.')
        if _1Halo:
            ax1.fill_between(
                ell, power_perc2p5_PF, power_perc97p5_PF, 
                color=PF_color, alpha=0.2, label=r"$95\%$ C.L.")

        if Elbers and m_nu == 0.1*args.eV:

            nu_100meV_Elbers_spectrum = jnp.array([
                0.0,
                79554042252.58139,
                18973423272.205936,
                4389187219.65975,
                2047662034.4722998,
                1147103616.5831783,
                748466187.2841997,
                309076670.4748975,
                144191744.89476708,
                109580959.62711476,
                78350132.47241905,
                40054287.11881511,
                34389340.010659076,
                28638722.571206003,
                28638722.571206003,
                19861554.89837982,
                13360656.9941187,
                11126475.498076187,
                9848667.679721419,
                8201764.305105058,
                6233069.749206187,
                2209891.0393051817])

            ax1.semilogy(
                ell[:16], nu_100meV_Elbers_spectrum[:16], 
                color="magenta", ls="dashed", label="Elbers+(2023)")
            ax1.semilogy(
                ell[15:22], nu_100meV_Elbers_spectrum[15:22], 
                color="magenta", ls="dotted",)


            print(f"Max power at l=1: {power_spectra_PF[:,1].max():.2e}")
            print(f"Max power of Elbers: {nu_100meV_Elbers_spectrum[1]:.2e}")


        if Tully and m_nu == 0.1*args.eV:

            # Read data
            fname = "Data/all_sky_power_spectra_Tully_100meV.csv"
            df = pd.read_csv(fname, header=None, names=['x_axis', 'y_axis'])

            # Plot data
            ax1.semilogy(
                df['x_axis'], df['y_axis'], 
                color='magenta', label='Tully+(2021)', alpha=0.8)


        # ax1.set_title(fr"Power spectra comparison $m_\nu = ${m_pick} eV")
        ax1.set_xlabel("$\ell$")
        # ax1.set_xlim(1,np.max(ell))
        ax1.set_xlim(1,21)
        ax1.set_ylabel(r"$\ell(\ell+1)C_{\ell} \, / \, 2\pi \,\, [\mu K^2]$")

        if y_lims is None:
            ax1.set_ylim(1e4, 1e11)
        else:
            ax1.set_ylim(y_lims[0], y_lims[1])

        ax1.grid(visible=True, which="major", ls="dashed", alpha=0.5)
        ax1.legend(loc='lower left', prop={'size':14})

        if _1Halo:
            plt.savefig(
                f"{fig_dir}/power_spectra_FD_vs_PF_seeds_{m_pick}eV.pdf", 
                bbox_inches="tight")
            plt.savefig(
                f"{fig_dir}/power_spectra_FD_vs_PF_seeds_{m_pick}eV.png", 
                bbox_inches="tight")
        else:
            plt.savefig(
                f"{fig_dir}/power_spectra_FD_vs_PF_halos_{m_pick}eV.pdf", 
                bbox_inches="tight")
            plt.savefig(
                f"{fig_dir}/power_spectra_FD_vs_PF_halos_{m_pick}eV.png", 
                bbox_inches="tight")
        plt.show(); plt.close()

    @staticmethod
    def power_spectra_FDxPF_ALL(
        m_pick, pix_dens_FD, pix_dens_PF, y_lims, sim_dir, fig_dir, args, Elbers=False, Tully=True):
        
        m_specs = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*args.eV
        m_range = jnp.load(f"{sim_dir}/neutrino_massrange_eV.npy")

        mPF_i = jnp.abs(m_specs - m_pick).argmin()
        # mFD_i = jnp.abs(m_range - m_pick).argmin()
        mFD_i = jnp.abs(m_specs - m_pick).argmin()
        m_nu = m_specs[mFD_i]

        halo_num = len(pix_dens_FD)

        power_spectra_FD_l = []
        power_spectra_PF_l = []
        T_unit = args.muK
        cm3 = args.cm**3
        for halo_i in range(halo_num):

            ### ========== ###
            ### FD spectra ###
            ### ========== ###

            # Pick current map
            FD_skymap = pix_dens_FD[halo_i, mFD_i]

            # Convert maps of neutrino densities to temperatures.
            FD_Tmap = np.cbrt((FD_skymap*(1/cm3)*4*args.Pi**2) / (args.g_nu*3*zeta(3)))
            FD_Tmap /= T_unit

            # Compute power spectrum of number density all-sky map.            
            cl_FD = hp.sphtfunc.anafast(FD_Tmap, lmax=None)
            ell = jnp.arange(len(cl_FD))

            # Unit to compare to similar figures in literature.
            # scaling = Primordial.Cl_scaling
            scaling = 1  # if Tmaps already in muK
            
            # Compute power spectrum of number density all-sky map.
            power_spectrum_FD = ell * (ell+1) * cl_FD * scaling / (2*args.Pi)
            power_spectra_FD_l.append(power_spectrum_FD)


            for seed_i in range(10):

                ### ========== ###
                ### PF spectra ###
                ### ========== ###

                # Pick current map
                PF_skymap = pix_dens_PF[halo_i, mPF_i, seed_i]

                # Convert maps of neutrino densities to temperatures.
                PF_Tmap = np.cbrt((PF_skymap*(1/cm3)*4*args.Pi**2) / (args.g_nu*3*zeta(3)))
                PF_Tmap /= T_unit

                # Compute power spectrum of number density all-sky map.
                cl_PF = hp.sphtfunc.anafast(PF_Tmap, lmax=None)

                # Compute power spectrum of number density all-sky map.
                power_spectrum_PF = ell * (ell+1) * cl_PF * scaling / (2*args.Pi)
                power_spectra_PF_l.append(power_spectrum_PF)
                


        # Convert lists to arrays
        power_spectra_FD = jnp.array(power_spectra_FD_l)
        power_spectra_PF = jnp.array(power_spectra_PF_l)
        # power_spectra_FD = savgol_filter(jnp.array(power_spectra_FD_l), 4, 2)
        # power_spectra_PF = savgol_filter(jnp.array(power_spectra_PF_l), 4, 2)

        # Compute median and percentile ranges for Fermi-Dirac spectra
        power_median_FD = jnp.median(power_spectra_FD, axis=0)
        power_perc2p5_FD = jnp.percentile(power_spectra_FD, q=2.5, axis=0)
        power_perc97p5_FD = jnp.percentile(power_spectra_FD, q=97.5, axis=0)
        power_perc16_FD = jnp.percentile(power_spectra_FD, q=16, axis=0)
        power_perc84_FD = jnp.percentile(power_spectra_FD, q=84, axis=0)

        # Compute median and percentile ranges for Primordial Fluctuations spectra
        power_median_PF = jnp.median(power_spectra_PF, axis=0)
        power_perc2p5_PF = jnp.percentile(power_spectra_PF, q=2.5, axis=0)
        power_perc97p5_PF = jnp.percentile(power_spectra_PF, q=97.5, axis=0)
        power_perc16_PF = jnp.percentile(power_spectra_PF, q=16, axis=0)
        power_perc84_PF = jnp.percentile(power_spectra_PF, q=84, axis=0)


        fig = plt.figure(figsize =(6, 4))
        fig.tight_layout()
        ax1 = fig.add_subplot(111)

        FD_color = "mediumblue"
        ax1.semilogy(
            ell, power_median_FD, color=FD_color, 
            label=r"MW + $f_{\rm FD}$")
        # ax1.fill_between(
        #     ell, power_perc2p5_FD, power_perc97p5_FD, 
        #     color=FD_color, alpha=0.15, label='2.5-97.5 % C.L.')
        ax1.fill_between(
            ell, power_perc16_FD, power_perc84_FD, 
            color=FD_color, alpha=0.25, label=r"$68\%$ C.L.")

        PF_color = "red"
        ax1.semilogy(
            ell, power_median_PF, color=PF_color, 
            label=r"MW + $f_{\Delta,{\rm all}}$", ls="solid")
        ax1.fill_between(
            ell, power_perc16_PF, power_perc84_PF, 
            color=PF_color, alpha=0.3, label=r"$68\%$ C.L.")
            # color=PF_color, alpha=0.3, label=r'$16\% - 84\%$ C.L.')
        ax1.fill_between(
            ell, power_perc2p5_PF, power_perc97p5_PF, 
            color=PF_color, alpha=0.2, label=r"$95\%$ C.L.")

        if Elbers and m_nu == 0.1*args.eV:

            nu_100meV_Elbers_spectrum = jnp.array([
                0.0,
                79554042252.58139,
                18973423272.205936,
                4389187219.65975,
                2047662034.4722998,
                1147103616.5831783,
                748466187.2841997,
                309076670.4748975,
                144191744.89476708,
                109580959.62711476,
                78350132.47241905,
                40054287.11881511,
                34389340.010659076,
                28638722.571206003,
                28638722.571206003,
                19861554.89837982,
                13360656.9941187,
                11126475.498076187,
                9848667.679721419,
                8201764.305105058,
                6233069.749206187,
                2209891.0393051817])

            ax1.semilogy(
                ell[:16], nu_100meV_Elbers_spectrum[:16], 
                color="magenta", ls="dashed", label="Elbers+(2023)")
            ax1.semilogy(
                ell[15:22], nu_100meV_Elbers_spectrum[15:22], 
                color="magenta", ls="dotted",)


            print(f"Max power at l=1: {power_spectra_PF[:,1].max():.2e}")
            print(f"Max power of Elbers: {nu_100meV_Elbers_spectrum[1]:.2e}")


        if Tully and m_nu == 0.1*args.eV:

            # Read data
            fname = "Data/all_sky_power_spectra_Tully_100meV.csv"
            df = pd.read_csv(fname, header=None, names=['x_axis', 'y_axis'])

            # Plot data
            ax1.semilogy(
                df['x_axis'], df['y_axis'], 
                color='magenta', label='Tully+(2021)', alpha=0.8)


        # ax1.set_title(fr"Power spectra comparison $m_\nu = ${m_pick} eV")
        ax1.set_xlabel("$\ell$")
        # ax1.set_xlim(1,np.max(ell))
        ax1.set_xlim(1,21)
        ax1.set_ylabel(r"$\ell(\ell+1)C_{\ell} \, / \, 2\pi \,\, [\mu K^2]$")

        if y_lims is None:
            ax1.set_ylim(1e4, 1e11)
        else:
            ax1.set_ylim(y_lims[0], y_lims[1])

        ax1.grid(visible=True, which="major", ls="dashed", alpha=0.5)

        if m_pick == 0.1:
            ax1.legend(loc='lower left', prop={'size':12})

        plt.savefig(
            f"{fig_dir}/power_spectra_FD_vs_PF_halos_ALL_{m_pick}eV.pdf", 
            bbox_inches="tight")
        plt.savefig(
            f"{fig_dir}/power_spectra_FD_vs_PF_halos_ALL_{m_pick}eV.png", 
            bbox_inches="tight")
        plt.show(); plt.close()

    @staticmethod
    def phase_space_2x2_FD_vs_PF(sim_dir, m_picks, halo_num, fig_dir, args):

        # Load neutrino vectors for selected halos (via halo_num)
        vels = SimData.load_velocities(sim_dir=sim_dir, halo_num=halo_num)

        # Convert velocities to momenta
        p_arr, y_arr = Physics.velocities_to_momenta_ND_halo_0th_axis(
            v_arr=vels, 
            m_arr=m_picks,
            args=Params())
        del vels

        p_z0 = p_arr[...,0]
        p_z4 = p_arr[...,-1]
        y_z0 = y_arr[...,0]

        # Sort in ascending order of momentum array today
        ind = p_z0.argsort(axis=-1)
        p_z4_sort = jnp.take_along_axis(p_z4, ind, axis=-1)
        y_z0_sort = jnp.take_along_axis(y_z0, ind, axis=-1)
        # Deltas_sort = jnp.take_along_axis(
        #     jnp.load(f"{sim_dir}/Deltas_halos.npy")[:halo_num, ...], ind, axis=-1)

        # x-axis
        y_z0_select = jnp.swapaxes(y_z0_sort, -1, -2)[..., 0]
        y_z0_median = jnp.median(y_z0_select, axis=0)

        # PSD using Fermi-Dirac assumption
        FD_vals = Physics.Fermi_Dirac(p_z4_sort, Params())

        # Select most (likely) clustered ones
        # FD_select = jnp.max(jnp.swapaxes(FD_vals, -1, -2), axis=-1)
        FD_select = jnp.max(FD_vals, axis=-2)
        print(FD_select.shape)

        # Compute medians and percentile ranges
        FD_median   = jnp.median(FD_select, axis=0)
        FD_perc2p5  = jnp.percentile(FD_select, q=2.5,  axis=0)
        FD_perc97p5 = jnp.percentile(FD_select, q=97.5, axis=0)
        FD_perc16   = jnp.percentile(FD_select, q=16,   axis=0)
        FD_perc84   = jnp.percentile(FD_select, q=84,   axis=0)

        """
        # PSD including primordial fluctuations
        PF_vals = Physics.Fermi_Dirac_Delta(p_z4_sort, Deltas_sort, Params())
        
        # Select most (likely) clustered ones
        PF_select = jnp.max(jnp.swapaxes(PF_vals, -1, -2), axis=-1)

        # Compute medians and percentile ranges
        PF_median   = jnp.median(PF_select, axis=0)
        PF_perc2p5  = jnp.percentile(PF_select, q=2.5,  axis=0)
        PF_perc97p5 = jnp.percentile(PF_select, q=97.5, axis=0)
        PF_perc16   = jnp.percentile(PF_select, q=16,   axis=0)
        PF_perc84   = jnp.percentile(PF_select, q=84,   axis=0)
        """


        plt.figure()
        for i in range(4):
            ax = plt.subplot(221 + i)


            ### =========== ###
            ### Fermi-Dirac ###
            ### =========== ###
            FD_color = "mediumblue"

            # Median of all halos
            ax.plot(
                y_z0_median[i], FD_median[i], 
                color=FD_color, alpha=0.7, label="Sim w/ Fermi-Dirac")

            # Percentile ranges of all halos
            ax.fill_between(
                y_z0_median[i], FD_perc2p5[i], FD_perc97p5[i],
                color=FD_color, alpha=0.2, label=r"$2.5-97.5$ % C.L.")
            ax.fill_between(
                y_z0_median[i], FD_perc16[i], FD_perc84[i],
                color=FD_color, alpha=0.3, label=r"$16-84$ % C.L.")


            """
            ### ======================= ###
            ### Primordial Fluctuations ###
            ### ======================= ###
            PF_color = "orangered"

            # Median of all halos
            ax.plot(
                y_z0_median[i], PF_median[i], 
                color=PF_color, alpha=0.7, label="Sim w/ Primordial Fluctuations")

            # Percentile ranges of all halos
            ax.fill_between(
                y_z0_median[i], PF_perc2p5[i], PF_perc97p5[i],
                color=PF_color, alpha=0.2, label=r"$2.5-97.5$ % C.L.")
            ax.fill_between(
                y_z0_median[i], PF_perc16[i], PF_perc84[i],
                color=PF_color, alpha=0.3, label=r"$16-84$ % C.L.")
            # """


            ### ================================= ###
            ### Original Fermi-Dirac distribution ###
            ### ================================= ###

            p_FD = jnp.geomspace(
                args.p_start*args.T_CNB, 
                args.p_stop*args.T_CNB, 
                FD_median.shape[-1])
            y_FD = p_FD / args.T_CNB
            vals_FD = Physics.Fermi_Dirac(p_FD, args)
            ax.plot(
                y_FD, vals_FD, 
                color="magenta", ls=":", alpha=0.85, 
                label="Original Fermi-Dirac")


            ### ============ ###
            ### Plot Styling ###
            ### ============ ###

            ax.set_title(fr"PSD for $m_\nu =$ {m_picks[i]} eV")
            ax.set_ylim(1e-2, 1.5*1e0)
            ax.set_xlim(Params.p_start, 1e1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$p/T_{\nu,0}$")
            ax.set_ylabel(r"$f_{\oplus,\mathrm{today}}$")
            ax.grid(**args.grid_dict)

            if i==3:
                ax.legend(
                    loc='lower left', borderpad=0.5, labelspacing=0.5, 
                    fontsize='medium', prop={"size":10})

            plt.tight_layout()

        # plt.savefig(f"{fig_dir}/PSD_FD_vs_PF.png", bbox_inches="tight")
        # plt.savefig(f"{fig_dir}/PSD_FD_vs_PF.pdf", bbox_inches="tight")
        plt.show(); plt.close()

        return y_z0_median, FD_select

    @staticmethod
    def momentum_distributions_2x2_FD_vs_PF(
            sim_dir, m_picks, halo_num, fig_dir, args):

        # Load neutrino vectors for all halos
        halo_vectors = SimData.load_vectors(sim_dir=sim_dir, halo_num=halo_num)

        # Convert velocities to momenta
        p_arr, y_arr = Physics.velocities_to_momenta_ND_halo_0th_axis(
            v_arr=halo_vectors[...,3:], 
            m_arr=m_picks,
            args=Params())

        p_z0 = p_arr[...,0]
        p_z4 = p_arr[...,-1]
        y_z0 = y_arr[...,0]

        # Sort in ascending order of momentum array today
        ind = p_z0.argsort(axis=-1)
        p_z4_sort = jnp.take_along_axis(p_z4, ind, axis=-1)
        y_z0_sort = jnp.take_along_axis(y_z0, ind, axis=-1)
        Deltas_sort = jnp.take_along_axis(
            jnp.load(f"{sim_dir}/Deltas_halos.npy"), ind, axis=-1)

        # x-axis
        y_z0_select = jnp.swapaxes(y_z0_sort, -1, -2)[..., 0]
        y_z0_median = jnp.median(y_z0_select, axis=0)

        # PSD using Fermi-Dirac assumption
        FD_vals = Physics.Fermi_Dirac(p_z4_sort, Params())

        # Select most (likely) clustered ones
        FD_select = jnp.max(jnp.swapaxes(FD_vals, -1, -2), axis=-1)
        
        # Compute medians and percentile ranges
        FD_median   = jnp.median(FD_select, axis=0)
        FD_perc2p5  = jnp.percentile(FD_select, q=2.5,  axis=0)
        FD_perc97p5 = jnp.percentile(FD_select, q=97.5, axis=0)
        FD_perc16   = jnp.percentile(FD_select, q=16,   axis=0)
        FD_perc84   = jnp.percentile(FD_select, q=84,   axis=0)

        # PSD including primordial fluctuations
        PF_vals = Physics.Fermi_Dirac_Delta(p_z4_sort, Deltas_sort, Params())
        
        # Select most (likely) clustered ones
        PF_select = jnp.max(jnp.swapaxes(PF_vals, -1, -2), axis=-1)

        # Compute medians and percentile ranges
        PF_median   = jnp.median(PF_select, axis=0)
        PF_perc2p5  = jnp.percentile(PF_select, q=2.5,  axis=0)
        PF_perc97p5 = jnp.percentile(PF_select, q=97.5, axis=0)
        PF_perc16   = jnp.percentile(PF_select, q=16,   axis=0)
        PF_perc84   = jnp.percentile(PF_select, q=84,   axis=0)


        plt.figure()
        for i in range(4):
            ax = plt.subplot(221 + i)


            ### =========== ###
            ### Fermi-Dirac ###
            ### =========== ###
            FD_color = "mediumblue"

            mult = (y_z0_median[i])**3

            # Median of all halos
            ax.plot(
                y_z0_median[i], FD_median[i]*mult, 
                color=FD_color, alpha=0.7, label="Sim w/ Fermi-Dirac")

            # Percentile ranges of all halos
            ax.fill_between(
                y_z0_median[i], FD_perc2p5[i]*mult, FD_perc97p5[i]*mult,
                color=FD_color, alpha=0.2, label=r"$2.5-97.5$ % C.L.")
            ax.fill_between(
                y_z0_median[i], FD_perc16[i]*mult, FD_perc84[i]*mult,
                color=FD_color, alpha=0.3, label=r"$16-84$ % C.L.")


            # """
            ### ======================= ###
            ### Primordial Fluctuations ###
            ### ======================= ###
            PF_color = "orangered"

            # Median of all halos
            ax.plot(
                y_z0_median[i], PF_median[i]*mult, 
                color=PF_color, alpha=0.7, label="Sim w/ Primordial Fluctuations")

            # Percentile ranges of all halos
            ax.fill_between(
                y_z0_median[i], PF_perc2p5[i]*mult, PF_perc97p5[i]*mult,
                color=PF_color, alpha=0.2, label=r"$2.5-97.5$ % C.L.")
            ax.fill_between(
                y_z0_median[i], PF_perc16[i]*mult, PF_perc84[i]*mult,
                color=PF_color, alpha=0.3, label=r"$16-84$ % C.L.")
            # """


            ### ================================= ###
            ### Original Fermi-Dirac distribution ###
            ### ================================= ###

            p_FD = jnp.geomspace(
                args.p_start*args.T_CNB, 
                args.p_stop*args.T_CNB, 
                FD_median.shape[-1])
            y_FD = p_FD / args.T_CNB
            vals_FD = Physics.Fermi_Dirac(p_FD, args)
            ax.plot(
                y_FD, vals_FD * y_FD**3, 
                color="magenta", ls=":", alpha=0.85, 
                label="Original Fermi-Dirac")


            ### ============ ###
            ### Plot Styling ###
            ### ============ ###

            ax.set_title(fr"PSD for $m_\nu=${m_picks[i]} eV")
            ax.set_ylim(1e-2, 3*1e1)
            ax.set_xlim(args.p_start, args.p_stop)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$p/T_{\nu,0}$")
            ax.set_ylabel(r"$f_{\oplus,\mathrm{today}}$")
            ax.grid(**args.grid_dict)

            if i==3:
                ax.legend(
                    loc='lower left', borderpad=0.5, labelspacing=0.5, 
                    fontsize='medium', prop={"size":10})

            plt.tight_layout()

        plt.savefig(f"{fig_dir}/PSD_FD_vs_PF.png", bbox_inches="tight")
        plt.savefig(f"{fig_dir}/PSD_FD_vs_PF.pdf", bbox_inches="tight")
        plt.show(); plt.close()

    @staticmethod
    def SunEarthGC_frame_coords(year):

        # Define the Galactic Center coordinates
        gc = SkyCoord(l=0*apu.degree, b=0*apu.degree, frame='galactic')
        
        # Sun's velocity w.r.t. CNB in galactic coordinates/frame (km/s)
        sun_vel_CNB = np.array([-0.06935679, -0.66222051, 0.74609224]) * 369.82

        # Sun's velocity w.r.t. MWGC in galactic coordinates/frame (km/s)
        sun_vel_MW = np.array([11.1029257, 250.8323298, 7.3656565])
        
        # Choose which relative velocity:
        #? I think vel_CNB for unbound, vel_MW for bound
        # sun_vel_Gal = sun_vel_CNB
        sun_vel_Gal = sun_vel_MW

        # Lists for Sun's positions and velocities, and Earth's velocities
        sun_positions = []
        sun_vels = []
        earth_vels = []
        
        # Loop backwards through each day of the year
        for day in range(365, 0, -1):  
            
            # Time objects of current and "next" (in the past) day at noon UTC
            time = Time(f'{year}-12-31 12:00:00') - (365-day)*apu.day
            time_next = time - 1*apu.day
            
            # Earth's position in ICRS coords. at current and "next" day (in km)
            earth_icrs = SkyCoord(
                get_body_barycentric('earth', time), 
                frame='icrs').cartesian.xyz
            earth_icrs_next = SkyCoord(
                get_body_barycentric('earth', time_next), 
                frame='icrs').cartesian.xyz
            
            # Calculate Earth's velocity in ICRS frame (km/s)
            earth_vel_icrs = (earth_icrs_next - earth_icrs).to(apu.km).value / apu.day.to(apu.s)
            
            # Get Sun's position in GCRS coordinates (relative to Earth) in km
            sun_gcrs = SkyCoord(
                get_body_barycentric('sun', time), 
                frame='icrs').transform_to(GCRS(obstime=time))
            sun_vector = sun_gcrs.cartesian.xyz.to(apu.km).value
            
            # Transform the Galactic Center to GCRS
            gc_gcrs = gc.transform_to(GCRS(obstime=time))
            
            # Calculate the unit vector pointing from Earth to Galactic Center
            gc_vector = gc_gcrs.cartesian.xyz.value
            gc_unit_vector = gc_vector / np.linalg.norm(gc_vector)
            
            # Define the new coordinate system
            x_axis = gc_unit_vector
            z_axis = np.cross(x_axis, [0, 0, 1])
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Create rotation matrix
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
            # Rotate Sun's position and velocity into new coordinate system
            sun_rotated = np.dot(rotation_matrix.T, sun_vector)
            sun_vel_rotated = np.dot(rotation_matrix.T, sun_vel_Gal)
            
            # Rotate Earth's velocity into the new coordinate system (km/s)
            earth_vel_rotated = np.dot(rotation_matrix.T, earth_vel_icrs)
            
            # Calculate Earth's velocity relative to CMB in the rotated frame (km/s)
            earth_vel_rotated = sun_vel_rotated + earth_vel_rotated
            
            # Append the rotated position and velocities to the lists
            sun_positions.append(sun_rotated)
            sun_vels.append(sun_vel_rotated)
            earth_vels.append(earth_vel_rotated)
        
        # note: sun_positions: km units ; velocities: km/s units
        return np.array(sun_positions), np.array(sun_vels), np.array(earth_vels)

    @staticmethod
    def SunEarthGC_frame_coords_circular(year):
        """
        Calculate coordinates and velocities in a frame aligned with the Galactic Center,
        approximating Earth's orbit as perfectly circular.
        """
        # Define the Galactic Center coordinates
        gc = SkyCoord(l=0*apu.degree, b=0*apu.degree, frame='galactic')
        
        # Sun's velocity w.r.t. CNB in galactic coordinates/frame (km/s)
        sun_vel_CNB = np.array([-0.06935679, -0.66222051, 0.74609224]) * 369.82

        # Sun's velocity w.r.t. MWGC in galactic coordinates/frame (km/s)
        sun_vel_MW = np.array([11.1029257, 250.8323298, 7.3656565])

        # Choose which relative velocity:
        #? I think vel_CNB for unbound, vel_MW for bound
        # sun_vel_Gal = sun_vel_CNB
        sun_vel_Gal = sun_vel_MW

        # Constants
        earth_orbital_period = 365.256363004  # days
        avg_earth_sun_distance = 149597870.7  # km (1 AU)
        orbital_speed = 29.78  # km/s (Earth's average orbital speed)

        sun_positions = []
        sun_vels = []
        earth_vels = []
        
        # Loop backwards through each day of the year
        for day in range(365, 0, -1):
            # Time object of current day at noon UTC
            time = Time(f'{year}-12-31 12:00:00') - (365-day)*apu.day
            
            # Calculate angle in the orbital plane (in radians)
            angle = 2 * np.pi * day / earth_orbital_period
            
            # Calculate simplified circular position
            # Using the ecliptic plane approximation
            x = avg_earth_sun_distance * np.cos(angle)
            y = avg_earth_sun_distance * np.sin(angle)
            z = 0  # Assuming orbit is in x-y plane
            #? I think we can do this because the paper which computes the 
            #? Earth-Sun orbital dynamics said that the orbit lies to a very 
            #? good approximation in the ecliptic plane
            sun_vector = np.array([x, y, z])
            
            # Calculate simplified circular velocity
            # Velocity is perpendicular to position in circular motion
            vel_x = -orbital_speed * np.sin(angle)
            vel_y = orbital_speed * np.cos(angle)
            vel_z = 0
            earth_vel_circular = np.array([vel_x, vel_y, vel_z])
            
            # Transform the Galactic Center to GCRS
            gc_gcrs = gc.transform_to(GCRS(obstime=time))
            
            # Calculate the unit vector pointing from Earth to Galactic Center
            gc_vector = gc_gcrs.cartesian.xyz.value
            gc_unit_vector = gc_vector / np.linalg.norm(gc_vector)
            
            # Define the new coordinate system
            x_axis = gc_unit_vector
            z_axis = np.cross(x_axis, [0, 0, 1])
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Create rotation matrix
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
            # Rotate Sun's position and velocity into new coordinate system
            sun_rotated = np.dot(rotation_matrix.T, sun_vector)
            sun_vel_rotated = np.dot(rotation_matrix.T, sun_vel_Gal)
            
            # Rotate Earth's velocity into the new coordinate system
            earth_vel_rotated = np.dot(rotation_matrix.T, earth_vel_circular)
            
            # Calculate Earth's total velocity relative to CMB in the rotated frame
            earth_vel_rotated = sun_vel_rotated + earth_vel_rotated
            
            # Append the rotated position and velocities to the lists
            sun_positions.append(sun_rotated)
            sun_vels.append(sun_vel_rotated)
            earth_vels.append(earth_vel_rotated)
        
        return np.array(sun_positions), np.array(sun_vels), np.array(earth_vels)

    @staticmethod
    def plot_enhanced_sun_trajectory(sun_positions, elev, azim):
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the Sun's trajectory
        x, y, z = sun_positions.T
        ax.plot(x, y, z, color='yellow', label='Sun trajectory', linewidth=2)

        # Plot Earth as a blue sphere at the center
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        earth_radius = 0.05 * np.max(np.abs(sun_positions))
        ex = earth_radius * np.outer(np.cos(u), np.sin(v))
        ey = earth_radius * np.outer(np.sin(u), np.sin(v))
        ez = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(ex, ey, ez, color='blue', alpha=0.7)

        # Calculate the maximum range for equal scaling
        max_range = np.max([np.ptp(x), np.ptp(y), np.ptp(z)]) / 2

        # Add Earth-GC line (x-axis)
        ax.plot([-max_range, max_range], [0, 0], [0, 0], color='red', linestyle='--', label='Earth-GC line')

        # Find the point with maximum z-coordinate
        max_z_index = np.argmax(z)
        max_z_point = sun_positions[max_z_index]

        # Plot line from origin to max_z_point
        ax.plot([0, max_z_point[0]], [0, max_z_point[1]], [0, max_z_point[2]], color='green', linestyle=':', label='Max height line')

        # Plot the ecliptic plane
        xx, yy = np.meshgrid(np.linspace(-max_range, max_range, 10), np.linspace(-max_range, max_range, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='cyan', label='Ecliptic plane')

        # Add labels and title
        ax.set_xlabel('X (Earth-GC direction)')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Sun's Trajectory Relative to Earth (Earth-GC Frame)")

        # Add legend
        ax.legend()

        # Set equal aspect ratio for all axes
        mid_x = np.mean(x)
        mid_y = np.mean(y)
        mid_z = np.mean(z)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add text annotations
        ax.text(mid_x + max_range, 0, 0, 'To Galactic Center', color='red')
        ax.text(0, 0, mid_z + max_range, 'North Ecliptic Pole', color='cyan')

        # Set the view angle
        ax.view_init(elev=elev, azim=azim)

        plt.show()


    @staticmethod
    @jax.jit
    def v_infinity(v_s, r_s):
        """
        Calculate the initial Solar-frame velocity for particles, s.t. they have velocity v_s at Earth's location.
        """
        
        v_GM = 2*Params.G*Params.Msun/jnp.linalg.norm(r_s, axis=-1)
        v_inf2 = jnp.linalg.norm(v_s, axis=-1)**2 - v_GM
        v_inf = jnp.sqrt(jnp.maximum(0, v_inf2))
        # v_inf = jnp.sqrt(v_inf2)
        r_s_unit = r_s / jnp.linalg.norm(r_s, axis=-1)
        vr_s_dot = jnp.dot(v_s, r_s_unit)

        numer = v_inf2[..., None]*v_s + v_inf[..., None]*v_GM/2*r_s_unit - v_inf[..., None]*v_s*vr_s_dot[..., None]
        denom = v_inf2 + v_GM/2 - v_inf*vr_s_dot
        
        return numer / denom[..., None]


    @staticmethod
    def plot_modulations_Safdi_arrays(
            which, density_arrays, cases, fig_folder, start_date_str='09-11', 
            y_lims=None, mode="modulation", pdf_str='mod.pdf'):
        """
        Plot the modulations using pre-computed density arrays.
        
        Parameters:
        -----------
        which : str
            'unbound', 'bound', or 'both'
        density_arrays : dict
            Dictionary containing the density arrays, with keys matching the cases
            For unbound: f"annual_densities_{m_nu}eV_unbound.npy"
            For bound: f"annual_densities_{v_0}kms_bound.npy"
        cases : list
            List of tuples containing (m_nu, bound, v_0, label, linestyle, color, alpha)
        start_date_str : str
            Start date in format 'MM-DD'
        """

        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()
        ax = fig.add_subplot(111)

        # Convert start_date_str to datetime object
        start_date = datetime.strptime(start_date_str, '%m-%d')
        start_day_of_year = start_date.timetuple().tm_yday

        # Filter cases based on which
        if which == "unbound":
            scenarios = [case for case in cases if not case[1]]  # bound is False
        elif which == "bound":
            scenarios = [case for case in cases if case[1]]      # bound is True
        elif which == "both":
            scenarios = cases

        # Generate time points
        days = jnp.linspace(0, Params.yr, 365)

        for m_nu, bound, v_0, label, linestyle, color, alpha in scenarios:
            # Load the appropriate density array
            if not bound:
                key = f"annual_densities_{m_nu}eV_unbound"
            else:
                key = f"annual_densities_{v_0/(Params.km/Params.s)}kms_{m_nu}eV_bound"
            
            densities = density_arrays[key]

            # Calculate modulation
            min_density = jnp.min(densities)
            mod = (densities - min_density) / (densities + min_density) * 100

            if mode == 'densities':
                mod = densities
                plt.title("Number densities")
                plt.ylabel(r'$n_\nu$ [${\rm cm}^{-3}$]')
            else:
                plt.title("Annual modulation")
                plt.ylabel(r'Modulation ($\%$)')

            # Check if all elements are zero or nan
            if jnp.all(mod == 0):
                print("all zeros")
            if jnp.all(jnp.isnan(mod)):
                print("all nan")

            # Shift days to start from the specified date
            shifted_days = (days/Params.yr*365 - start_day_of_year) % 365 + 1
            sorted_indices = jnp.argsort(shifted_days)
            shifted_days = shifted_days[sorted_indices]
            mod = mod[sorted_indices]

            ax.plot(
                shifted_days, mod, 
                linestyle=linestyle, color=color, label=label, alpha=alpha)

        # Calculate tick positions and labels
        tick_dates = [datetime_date(2000, 11, 1), datetime_date(2000, 2, 1), 
                    datetime_date(2000, 5, 1), datetime_date(2000, 8, 1)]
        tick_days = [(d - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1 
                    for d in tick_dates]
        tick_labels = ['Nov 1', 'Feb 1', 'May 1', 'Aug 1']
        plt.xticks(tick_days, tick_labels)
        plt.grid(True, which="major", linestyle="dashed")

        # Axis limits
        if y_lims is not None:
            plt.ylim(y_lims[0], y_lims[1])
        
        if which == "unbound" or which == "both":
            march_12 = (datetime_date(2000, 3, 12) - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1
            sept_11 = (datetime_date(2000, 9, 11) - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1
            plt.axvline(x=march_12, color='dodgerblue', linestyle=':', 
                    label=r'$\textbf{Min. Unbound}$ (Mar 12th)')
            plt.axvline(x=sept_11, color='magenta', linestyle=':', 
                    label=r'$\textbf{Max. Unbound}$ (Sep 11th)')
            
        if which == "bound" or which == "both":
            march_1 = (datetime_date(2000, 3, 1) - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1
            sept_1 = (datetime_date(2000, 9, 1) - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1
            plt.axvline(x=march_1, color='red', linestyle=':', 
                    label=r'$\textbf{Max. Bound}$ (Mar 1st)')
            plt.axvline(x=sept_1, color='purple', linestyle=':', 
                    label=r'$\textbf{Min. Bound}$ (Sep 1st)')

        plt.legend(prop={"size":10})
        plt.savefig(f"{fig_folder}/{pdf_str}", bbox_inches="tight")
        plt.show()
        plt.close()

    @staticmethod
    def plot_annual_modulations(
            nu_dens_days, simulated_days, nu_m_picks, 
            start_date_str='09-11', mode='modulation', mod_method="", log_yaxis=False, y_lims=None, pdf_path="", etas_tuple=None, args=None):
        
        if args is None:
            args = Params()

        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()
        ax = fig.add_subplot(111)

        # Convert start_date_str to datetime object
        start_date = datetime.strptime(start_date_str, '%m-%d')

        # Calculate the day of the year for the start date
        start_day_of_year = start_date.timetuple().tm_yday

        # Shift days to start from the specified date
        shifted_days = (simulated_days - start_day_of_year) % 365 + 1
        sorted_indices = jnp.argsort(shifted_days)
        shifted_days = shifted_days[sorted_indices]

        violet_colors = [
            # "lightpink",
            # "hotpink", 
            "mediumvioletred", 
            # "blueviolet",
            "rebeccapurple",
        ]

        for m_idx, (m_val, color) in enumerate(zip(nu_m_picks, violet_colors)):

            # Select current mass
            nu_dens_m = nu_dens_days[..., m_idx]
            # shape: (days, halos)

            # Since we started from Dec 31st to Jan 1st, we flip array
            nu_dens_m = jnp.flip(nu_dens_m, axis=0)

            if mode == 'densities':
                # Display number over-densities
                mod = (nu_dens_m / args.N0_cm3) - 1

                plt.title("Number densities")

                if log_yaxis:
                    ax.set_yscale("log")
                    ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$')
                    # 1+10^X styled y-axis labels, to easily see order of overdensities
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(Utils.y_fmt))

                    if m_idx in (0,1,3):
                        plt.axhline(
                            y=etas_tuple[0,m_idx], color=color, 
                            linestyle="dashed", alpha=1)
                        # plt.axhline(
                        #     y=etas_tuple[1,m_idx], color=color, 
                        #     linestyle=":", alpha=1)
                        # plt.axhline(
                        #     y=etas_tuple[2,m_idx], color=color, 
                        #     linestyle=":", alpha=1)

                else:
                    if m_idx in (0,1,3):
                        plt.axhline(
                            y=etas_tuple[0,m_idx], color=color, 
                            linestyle="dashed", alpha=1)
                        # plt.axhline(
                        #     y=etas_tuple[1,m_idx], color=color, 
                        #     linestyle=":", alpha=1)
                        # plt.axhline(
                        #     y=etas_tuple[2,m_idx], color=color, 
                        #     linestyle=":", alpha=1)
                    
                    ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$ - 1')
                
                # if m_idx == 0:
                #     plt.axhline(y=Params.N0_cm3, color='k', linestyle=":", label=r"$\Lambda$CDM $n_\nu$")

            else:
                # Compute fractional modulation for each halo
                if mod_method == "rel":
                    nu_min = jnp.min(nu_dens_m, axis=0)
                    mod = (nu_dens_m - nu_min) / (nu_dens_m + nu_min) * 100
                    ax.set_ylabel(r'Modulation ($\%$)')
                if mod_method == "avg":
                    nu_avg = jnp.mean(nu_dens_m, axis=0)
                    mod = (nu_dens_m - nu_avg) / nu_avg * 100
                    ax.set_ylabel(r'Difference from $\bar{n}_\nu$ ($\%$)')

                plt.title("Annual modulation")


            # Compute median and 1-sigma band
            median = jnp.median(mod, axis=-1)
            lower = jnp.percentile(mod, 16, axis=-1)
            upper = jnp.percentile(mod, 84, axis=-1)

            # Apply sorting to match shifted_days
            median = median[sorted_indices]
            lower = lower[sorted_indices]
            upper = upper[sorted_indices]

            # Plot median and 1-sigma band
            ax.plot(shifted_days, median, '-', color=color, label=fr"$m_\nu = {m_val}$ eV")
            ax.fill_between(shifted_days, lower, upper, color=color, alpha=0.2)


        # Calculate tick positions and labels
        tick_dates = [datetime_date(2000, 11, 1), datetime_date(2000, 2, 1), 
                    datetime_date(2000, 5, 1), datetime_date(2000, 8, 1)]
        tick_days = [(d - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1 for d in tick_dates]
        tick_labels = ['Nov 1', 'Feb 1', 'May 1', 'Aug 1']

        plt.xticks(tick_days, tick_labels)

        # March 12th (2000,3,12) and September 11th (2000,9,11) (Safdi, bound)
        # January 4th (2000,1,4) and Juli 5th (2000,7,5) (Max/Min Earth vel.)
        markday1 = (datetime_date(2000,1,4) - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1
        markday2 = (datetime_date(2000,7,5) - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1

        # if mode != 'densities':
        #     plt.axvline(
        #         x=markday1, color='r', linestyle=':', 
        #         # label='March 12th')
        #         label='v_max Earth (closest to Sun)')
        #     plt.axvline(
        #         x=markday2, color='g', linestyle=':', 
        #         # label='September 11th')
        #         label='v_min Earth (furthest from Sun)')

        if y_lims is not None:
            plt.ylim(y_lims[0], y_lims[1])
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, which="major", linestyle="dashed")
        plt.legend(prop={"size":12})

        plt.savefig(f"{pdf_path}.pdf", bbox_inches="tight")

        plt.show()
        plt.close()
