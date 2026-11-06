"""Plotting functions for the paper figures. Importing this module sets the
matplotlib style."""

from datetime import datetime, date as datetime_date

import numpy as np
from scipy.special import zeta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import healpy as hp
import pandas as pd
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid as trap
import colorcet as cc

from cnb.astro import Astro
from cnb.physics import Physics
from cnb.primordial import Primordial
from cnb.sim_data import SimData
from cnb.skymaps import Skymaps
from cnb.units import Params
from cnb.utils import Utils


def set_style():
    """Matplotlib style used for the paper figures (needs LaTeX)."""

    # For paper:
    # SMALL_SIZE = 14
    # MEDIUM_SIZE = 18
    # BIGGER_SIZE = 22

    # For testing
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    mpl.rc('figure', figsize=(8, 8))
    mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
    mpl.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    mpl.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    # mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    mpl.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    mpl.rc('xtick', direction='in', top=True)  # x ticks inside, top enabled
    mpl.rc('ytick', direction='in', right=True) # y ticks inside, right enabled
    mpl.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
    mpl.rc('text', usetex=True)
    mpl.rc('axes.formatter', use_mathtext=True)
    mpl.rcParams['xtick.minor.visible'] = True


set_style()


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
            #     np.save(f"runs/SunMod_2k/etas_med.npy", etas_med)
            #     np.save(f"runs/SunMod_2k/etas_16.npy", etas_16)
            #     np.save(f"runs/SunMod_2k/etas_84.npy", etas_84)

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
        vmag_Sun_CMB = jnp.linalg.norm(Astro.Galactic_velocity_to_Cartesian_vector(
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
    def skymaps_percXdm(
        nu_mass, perc_map, DM_halo_pos, init_xyz, fig_paths, args, plot=True):

        Nside = hp.pixelfunc.get_nside(perc_map)

        # Rotate DM halo (based on observer position), s.t. CoP is at origin
        # (middle) in healpy skymap
        DM_halo_rot, obs_rot = Skymaps.rotate_particles(DM_halo_pos, init_xyz)

        # Create healpix skymap with these new DM coords
        DM_halo_skymap = Skymaps.DM_pos_to_skymap(Nside, DM_halo_rot, obs_rot)


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
            fname = "data/all_sky_power_spectra_Tully_100meV.csv"
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
            fname = "data/all_sky_power_spectra_Tully_100meV.csv"
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
