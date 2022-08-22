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

    # This function builds the merger tree of your halo selection and stores it in a file

    # Starting snapshot is 36 (z=0).
    initial_snap = sim_info.initial_snap
    final_snap = 10

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


class Mock_ArgumentParser:

    sim_ID = 'L006N188'
    name_list = f'MergerTree_{sim_ID}'

    snapshot_list = [
        'snapshot_0036.hdf5', 
        # 'snapshot_0035.hdf5', 
        # 'snapshot_0034.hdf5'
    ]
    catalogue_list = [
        'subhalo_0036.properties', 
        # 'subhalo_0035.properties', 
        # 'subhalo_0034.properties'
    ]

    # Paths for FZ_snellius.
    if str(pathlib.Path.home()) == '/home/zimmer':
        home = '/home/zimmer'
        root = '/projects/0/einf180/Tango_sims'
        directory_list = f'{root}/{sim_ID}/DMONLY/SigmaConstant00/'
        output_directory = f'{home}/neutrino_clustering_output_local/MergerTree/'
        number_of_inputs = len(snapshot_list)

    # Paths for FZ_desktop.
    elif str(pathlib.Path.home()) == '/home/fabian':
        root = '/home/fabian'
        home = f'{root}/my_github_projects'
        directory_list = f'{root}/ownCloud/snellius/L006N188/'
        output_directory = f'{home}/neutrino_clustering_output_local/MergerTree/'
        number_of_inputs = len(snapshot_list)

config_parameters = Mock_ArgumentParser()
main(config_parameters)