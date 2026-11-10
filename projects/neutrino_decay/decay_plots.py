"""Plots for the neutrino decay project."""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import healpy as hp

from cnb.plotting import SimPlot
from cnb.skymaps import Skymaps
from cnb.utils import Utils


class DecayPlot():
    """Plots for the neutrino decay project."""


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
    def skymaps_CNBxDM_decay(
        nu_mass, CNB_dens, DM_halo_pos, init_xyz, fig_paths, args, plot=True):

        Nside = hp.pixelfunc.get_nside(CNB_dens)
        Npix = hp.pixelfunc.nside2npix(Nside)
        
        # Rotate DM halo (based on observer position), s.t. CoP is at origin
        # (middle) in healpy skymap
        DM_halo_rot, obs_rot = Skymaps.rotate_particles(DM_halo_pos, init_xyz)

        # Create healpix skymap with these new DM coords
        DM_halo_skymap = Skymaps.DM_pos_to_skymap(Nside, DM_halo_rot, obs_rot)

        # Rotate neutrino density skymap
        CNB_skymap_rot = Skymaps.rotate_skymap(CNB_dens, init_xyz)

        # Interpolate pixels without values due to rotation
        CNB_skymap_int = Skymaps.interpolate_skymap(CNB_skymap_rot)

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
