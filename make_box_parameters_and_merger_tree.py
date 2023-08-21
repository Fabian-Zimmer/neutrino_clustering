from shared.preface import *

import velociraptor
from shared.snapshot_data import snapshot_info
from shared.tree_dataset import TreeCatalogue
from shared.argumentparser import ArgumentParser
from shared import simulation_data
import pathlib


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


def main(config: ArgumentParser):

    # Loop over simulation list
    for sim in range(config.number_of_inputs):

        # Fetch relevant input parameters from lists
        directory = config.directory_list
        snapshot = config.snapshot_list[sim]
        catalogue = config.catalogue_list[sim]
        sim_name = config.name_list
        output = config.output_directory

        # Load all data and save it in SimInfo class
        sim_info = simulation_data.SimInfo(
            directory=directory,
            snapshot=snapshot,
            catalogue=catalogue,
            name=sim_name,
            output=output,
            simtype='DMONLY'
        )

        make_tree_data(sim_info)


def make_box_parameters_and_merger_tree(
    box_dir, box_name, box_ver, sim_fullname, z0_snap, z4_snap
):
    """
    Reads cosmological and other global parameters of the specified simulation box. Output is stored in corresponding output folder, where all simulation outputs (e.g. neutrino densities) will be stored.
    """

    # Box input/output paths.
    file_dir = f'{box_dir}/{box_name}/{box_ver}'
    out_dir = f'{os.getcwd()}/{box_name}/{box_ver}/{sim_fullname}'

    # Create output director, if it doesn't exist yet.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # The snapshots to read, determined by inputs z0_snap and z4_snap.
    snap_nums = np.arange(z4_snap, z0_snap+1)
    nums_4cif = []
    zeds = np.zeros(len(snap_nums))

    # Loop over selected snapshots and save outputs.
    for i, num in enumerate(snap_nums):
        num_4cif = f'{num:04d}'
        nums_4cif.append(num_4cif)

        with h5py.File(f'{file_dir}/snapshot_{num_4cif}.hdf5') as snap:

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
                DM_mass = np.unique(snap['PartType1/Masses'][:]*1e10*Msun)[0]

                # Gravity smoothening length of box.
                smooth_len = snap['GravityScheme'].attrs[
                    'Maximal physical DM softening length (Plummer equivalent) [internal units]'
                ][0]*1e6*pc



                # Create .yaml file for global parameters of box.
                box_parameters = {
                    "File Paths": {
                        "Box Root Directory": box_dir,
                        "Box Name": box_name,
                        "Box Version": box_ver,
                        "Box File Directory": file_dir,
                        "Output Directory": out_dir
                    },
                    "Cosmology": {
                        "Omega_R": float(Omega_R),
                        "Omega_M": float(Omega_M),
                        "Omega_L": float(Omega_L),
                        "h": float(h_dimless),
                    },
                    "Content": {
                        "DM Mass [Msun]": float(DM_mass/Msun),
                        "Smoothening Length [pc]": float(smooth_len/pc),
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

    class Mock_ArgumentParser():
        def __init__(
                self, box_file_dir, output_dir, file_name, z0_snap_4cif
            ):
            self.directory_list = box_file_dir
            self.snapshot_list = [f'snapshot_{z0_snap_4cif}.hdf5',]
            self.catalogue_list = [f'subhalo_{z0_snap_4cif}.properties',]
            self.name_list = file_name
            self.output_directory = output_dir
            self.number_of_inputs = len(self.snapshot_list)

    print(f'***Making merger tree for box name/version {box_name}/{box_ver}***')
    config_parameters = Mock_ArgumentParser(
        box_file_dir=file_dir, 
        output_dir=out_dir, 
        file_name='MergerTree',
        z0_snap_4cif=f'{z0_snap:04d}'
    )
    main(config_parameters)


# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-bd', '--box_directory', required=True)
parser.add_argument('-bn', '--box_name', required=True)
parser.add_argument('-bv', '--box_version', required=True)
parser.add_argument('-sf', '--sim_fullname', required=True)
parser.add_argument('-z0', '--initial_snap_z0', required=True)
parser.add_argument('-z4', '--final_snap_z4', required=True)
args = parser.parse_args()


make_box_parameters_and_merger_tree(
    box_dir=args.box_directory, 
    box_name=args.box_name, 
    box_ver=args.box_version, 
    sim_fullname=args.sim_fullname,
    z0_snap=int(args.initial_snap_z0), 
    z4_snap=int(args.final_snap_z4)
)
