from shared.preface import *
# from shared.plot_class import analyze_simulation_outputs
from analysis_testground import analyze_simulation_outputs_test

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('-sd', '--sim_directory', required=True)
parser.add_argument('-st', '--sim_type', required=True)
parser.add_argument(
    '--NFW_halo', required=True, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    '--box_halos', required=True, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    '--analytical_halo', required=True, action=argparse.BooleanOptionalAction
)
args = parser.parse_args()

# All objects to analyze.
objects = ()
objects = objects \
    + ('NFW_halo',)*args.NFW_halo \
    + ('box_halos',)*args.box_halos \
    + ('analytical_halo',)*args.analytical_halo

# Create Analysis class.
Analysis = analyze_simulation_outputs_test(
    sim_dir = args.sim_directory,
    objects = objects,
    sim_type = args.sim_type,
)

# Generate suite of plots.
# Analysis.plot_overdensity_band(plot_ylims=(3*1e-4,1e1))
# Analysis.plot_overdensity_band(plot_ylims=None)
# Analysis.plot_overdensity_evolution(plot_ylims=(1e-4,1e1))
# Analysis.plot_phase_space(mass_gauge=12.0, mass_range=0.6, most_likely=True)
# Analysis.plot_density_profiles(mass_gauge=12.0, mass_range=0.6, NFW_orig=True)
# Analysis.plot_2d_params(nu_mass_eV=0.3)
print(Analysis.final_halos)
print(Analysis.halo_num)
halo_array = np.arange(Analysis.halo_num)
Analysis.plot_all_spectra_1plot(halo_array, 0.3)