from shared.preface import *
from shared.shared_functions import *

def transform_simulation_outputs(out_dir, sim_fullname):
    """
    Input sim_fullname specifies the simulation mode with additional information: e.g. "single_halos_40k_test1" or "spheres_5shells_10k_final".
    """

    # Try loading the equivalent output for the analytical simulation method.
    # If it hasn't been run yet, raise error and inform user.
    try:

        # Load neutrino vectors (positions and velocities) of the 10k batches, and concatenate them into a single array.
        batch_paths = glob.glob(
            f'{out_dir}/neutrino_vectors_analytical_batch*.npy'
        )
        vectors_ana = []
        for batch_path in batch_paths:
            vectors_ana.append(np.load(batch_path))
        vectors_ana = np.squeeze(np.array(vectors_ana))

        etas_ana = np.load(
            f'{out_dir}/number_densities_analytical.npy'
        )/N0

        analytical_out = True

    except FileNotFoundError:
        print('! Analytical simulation output not found !')
        analytical_out = False


    if 'single_halos' or 'spheres' in sim_fullname:
        
        # Load neutrino vectors (positions and velocities) of the 10k batches, and concatenate them into a single array.
        batch_paths = glob.glob(
            f'{out_dir}/neutrino_vectors_numerical_batch*.npy'
        )
        vectors_num = []
        for batch_path in batch_paths:
            vectors_num.append(np.load(batch_path))
        vectors_num = np.squeeze(np.array(vectors_num))

        etas_num = np.load(
            f'{out_dir}/number_densities_numerical.npy'
        )/N0

        if analytical_out:
            return vectors_ana, etas_ana, vectors_num, etas_num
        else:
            return vectors_num, etas_num
    
    else:
        # Load angle pairs and calculate overdensities for the all_sky mode.
        all_sky_output = np.load(f'{out_dir}/number_densities_numerical.npy')
        angle_pairs = all_sky_output[:, :2]
        etas_numerical = all_sky_output[:, 2:]/N0

        return etas_numerical, angle_pairs


def analyze_simulation_outputs(
        etas, m_nu_range,
        box_name, box_ver,
        plots_to_make, ylims, Mertsch=False,
        # vectors, sim_fullname,
    ):

    # Figure directory.
    # fig_dir = f'figures/{box_name}/{box_ver}/{sim_fullname}'
    fig_dir = f'figures/{box_name}/{box_ver}'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    savefig_args = dict(
        bbox_inches='tight'
    )

    if Mertsch:
        etas_analytical = etas[0,:]
        etas_numerical = etas[1:,:]
    else:
        etas_numerical = etas[...]

    ### ================================== ###
    ### Figure of merit: overdensity band. ###
    ### ================================== ###

    fig, ax = plt.subplots(1,1)

    if 'overdensity_band' in plots_to_make:

        if Mertsch:
            # Plot smooth simulation.
            ax.plot(
                m_nu_range*1e3, (etas_analytical-1), color='red', ls='solid', 
                label='Analytical simulation'
            )

        # '''
        if etas_numerical.ndim <= 1:
            ax.plot(
                m_nu_range*1e3, (etas_numerical-1), color='blue', 
                label='medians'
            )
        else:
            nus_median = np.median(etas_numerical, axis=0)
            nus_perc2p5 = np.percentile(etas_numerical, q=2.5, axis=0)
            nus_perc97p5 = np.percentile(etas_numerical, q=97.5, axis=0)
            nus_perc16 = np.percentile(etas_numerical, q=16, axis=0)
            nus_perc84 = np.percentile(etas_numerical, q=84, axis=0)
            ax.plot(
                m_nu_range*1e3, (nus_median-1), color='blue', 
                label='Halo sample medians'
            )
            ax.fill_between(
                m_nu_range*1e3, (nus_perc2p5-1), (nus_perc97p5-1), 
                color='blue', alpha=0.2, label='2.5-97.5 %'
            )
            ax.fill_between(
                m_nu_range*1e3, (nus_perc16-1), (nus_perc84-1), 
                color='blue', alpha=0.3, label='16-84 %'
            )
        # '''

        if Mertsch:
            # Plot endpoint values from Mertsch et al.
            x_ends = [1e1, 3*1e2]
            y_ends = [3*1e-3, 4]
            ax.scatter(x_ends, y_ends, marker='x', s=15, color='orange')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'Overdensity band')
        ax.set_xlabel(r'$m_{\nu}$ [meV]')
        ax.set_ylabel(r'$n_{\nu} / n_{\nu, 0}$')
        # ax.set_ylim(ylims[0], ylims[1])
        plt.grid(True, which="both", ls="-")
        plt.legend(loc='lower right')


        plt.savefig(fig_dir, **savefig_args)
        plt.show()


box_name = 'L025N752'
box_ver = 'DMONLY/SigmaConstant00'
out_dir = f'{box_name}/{box_ver}'
vecs_ana, etas_ana, vecs_num, etas_num = transform_simulation_outputs(
    out_dir, sim_fullname='single_halos'
)
etas_combined = np.stack((etas_ana, etas_num), axis=0)

neutrino_massrange = np.load(f'{out_dir}/neutrino_massrange_eV.npy')*eV

analyze_simulation_outputs(
    etas=etas_combined, 
    m_nu_range=neutrino_massrange,
    box_name=box_name, 
    box_ver=box_ver,
    plots_to_make=('overdensity_band', 'vels'), 
    ylims=(1e-3, 1e2),
    # ylims=(0, 1e-2),
    Mertsch=True
)