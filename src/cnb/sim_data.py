"""Reading run parameters, halo catalogues, DM particles and simulated neutrino
vectors from disk."""

import os
import yaml

import numpy as np
import h5py
import jax.numpy as jnp

from cnb.units import Params


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
        sim_dir, m_lower, m_upper, m_gauge, halo_num_req, no_gravity=False):
        """
        Load the parameters of a run (from scripts/make_box.py and scripts/make_parameters.py). Unless no_gravity, also select the halo sample from the box's halo catalogue and save its IDs and (Rvir, Mvir, cNFW) in sim_dir.
        """

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

            return DM_mass, CPUs_sim, neutrinos, init_dis, zeds_snaps, z_int_steps, s_int_steps, nu_massrange, halo_batch_IDs, halo_num_avail


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
        DM_dis_lims = DM_shell_edges*(a/Params.kpc/1e3)

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
