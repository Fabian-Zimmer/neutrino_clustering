"""Analysis helpers shared by the notebooks: redshift-age conversion, bulk
velocity and dipole of the neutrino skymaps, temperature-fluctuation skymaps,
momentum distributions. Each notebook used to have its own copy.

The bulk velocity functions read the run's vectors_halo*.npy from sim_folder.
"""

import astropy.units as u
import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.integrate import trapezoid as trap

from cnb.physics import Physics
from cnb.plotting import SimPlot
from cnb.primordial import Primordial
from cnb.units import Params, cosmo


def age_to_redshift(age_gyr):
    """
    Converts the age of the Universe (in Gyr) to redshift.
    
    Parameters:
    age_gyr (float): Age of the Universe in gigayears (Gyr).
    
    Returns:
    float: Corresponding redshift.
    """
    # Convert age to redshift
    age = age_gyr * u.Gyr
    
    # Ensure the input age is within valid range
    if age_gyr < 0 or age_gyr > cosmo.age(0).value:
        raise ValueError("Age must be between 0 and the age of the Universe today.")
    
    # Create an array of redshifts to interpolate over
    z_vals = np.logspace(-5, 5, 10000)
    age_vals = cosmo.age(z_vals).value  # Get corresponding ages for these redshifts
    
    # Interpolate to find the redshift for the given age
    redshift = np.interp(age_gyr, age_vals[::-1], z_vals[::-1])
    return redshift


def redshift_to_age(z):
    """
    Converts redshift to the age of the Universe.
    
    Parameters:
    z (float): Redshift.
    
    Returns:
    float: Age of the Universe in gigayears (Gyr).
    """
    # Calculate the age of the Universe at the given redshift
    age = cosmo.age(z).to(u.Gyr).value
    return age


def bulk_velocity_simple(args, sim_folder, simdata, nu_m_picks):

    # Velocities
    v_unit = (Params.kpc/Params.s)/(Params.km/Params.s)
    vels = jnp.load(f"{sim_folder}/vectors_halo1.npy")[..., 3:6]
    # print(vels.shape)
    v_init = vels[..., 0, :]
    v_back = vels[...,-1, :]
    # print(v_init.shape)
    v_mags = jnp.linalg.norm(vels, axis=-1)

    # Momenta
    p_arr, _ = Physics.velocities_to_momenta_all_sky(vels, nu_m_picks, args)
    p_init = p_arr[..., 0]
    p_back = p_arr[...,-1]
    # print(p_arr.shape)
    # print(p_init.shape)

    # Sorting arrays in ascending order (for trap integral)
    ind = p_init.argsort(axis=-1)
    # print(v_init[..., 0][None, ...].shape)
    p_init_sort = jnp.take_along_axis(p_init, ind, axis=-1)
    p_back_sort = jnp.take_along_axis(p_back, ind, axis=-1)
    v_x_sort = jnp.take_along_axis(v_init[..., 0][None, ...], ind, axis=-1)
    v_y_sort = jnp.take_along_axis(v_init[..., 1][None, ...], ind, axis=-1)
    v_z_sort = jnp.take_along_axis(v_init[..., 2][None, ...], ind, axis=-1)
    # print(v_z_sort.shape)

    # Checking if number density is ok
    # n_nus = Physics.number_density(p_init, p_back, simdata.pix_sr, args)*(1/args.cm**3)
    n_nus = Physics.number_density(p_init_sort, p_back_sort, simdata.pix_sr, args)*(1/args.cm**3)
    print("n_nus:", jnp.sum(n_nus, axis=-1)/args.cm**-3)


    FD_vals = Physics.Fermi_Dirac(p_back_sort, args)
    print(FD_vals.shape)

    v_sort = jnp.stack((v_x_sort, v_y_sort, v_z_sort), axis=3)*v_unit
    print(v_sort.shape)

    weighted_vels = FD_vals[..., None]*v_sort

    summed_vels_xyz = jnp.sum(weighted_vels, axis=2)
    print(summed_vels_xyz.shape)

    v_bulk_vecs = jnp.sum(summed_vels_xyz, axis=1)
    print(v_bulk_vecs)

    norm = jnp.sum(FD_vals)

    v_bulk_mags = jnp.linalg.norm(v_bulk_vecs, axis=-1) / norm
    print(v_bulk_mags)


def bulk_velocity_new(args, sim_folder, simdata, nu_m_picks):

    # Velocities
    vels = jnp.load(f"{sim_folder}/vectors_halo1.npy")[..., 3:6]
    v_mags = jnp.linalg.norm(vels, axis=-1)
    v_init = v_mags[..., 0]
    v_back = v_mags[...,-1]
    # print(vels.shape)
    # print(v_init.shape)

    # Momenta
    p_arr, _ = Physics.velocities_to_momenta_all_sky(vels, nu_m_picks, args)
    p_init = p_arr[..., 0]
    p_back = p_arr[...,-1]
    # print(p_arr.shape)
    # print(p_init.shape)

    n_nus = Physics.number_density(p_init, p_back, simdata.pix_sr, args)*(1/args.cm**3)
    # print(n_nus.shape)

    # note: trapz integral method needs sorted (ascending) "x-axis" array.
    ind = p_init.argsort(axis=-1)
    p_init_sort = jnp.take_along_axis(p_init, ind, axis=-1)
    p_back_sort = jnp.take_along_axis(p_back, ind, axis=-1)
    v_init_sort = jnp.take_along_axis(v_init[None, ...], ind, axis=-1)
    v_back_sort = jnp.take_along_axis(v_back[None, ...], ind, axis=-1)

    # Deviation of phase-space today from Fermi-Dirac distribution
    f_diff = jnp.abs(Physics.Fermi_Dirac(p_back_sort, args) - Physics.Fermi_Dirac(p_init_sort, args))
    # print(f_diff.shape)

    y = p_init_sort**4 * f_diff  # dlog integrand
    x = p_init_sort
    integral = trap(y, jnp.log(x), axis=-1)

    # left hand side of equation
    rho_P_v = args.g_nu / (2*args.Pi**2) * integral
    print(rho_P_v.shape)

    # energy density
    # rho = nu_m_picks[:, None] * n_nus
    rho = nu_m_picks[:, None] * args.N0/(1/args.cm**3)
    print(rho.shape)

    # bulk velocity? dipole perturbation?
    v_bulk = rho_P_v / rho

    return v_bulk


def dipole_perturbation(nu_vel_skymap):

    cl, alm = hp.sphtfunc.anafast(np.array(nu_vel_skymap), alm=True, lmax=1)

    return cl, alm


def Delta_vs_z_cut_show(Cl_dir, m_arr, seed, Nside, args, fig_folder):

    Deltas_l = []
    for m_val in m_arr:
        
        # Load q-averaged Cl's for all z_cuts
        indices = jnp.array([0,9])
        z_cuts_select = Primordial.z_cuts[indices]
        Cls_q_avg_pre = jnp.array([
            jnp.load(f"{Cl_dir}/Cls_q_averaged_z_cut={z_cut:.2f}_m={m_val}eV.npy")
            for z_cut in z_cuts_select])
        # (z_cuts, 20)
        
        # Add the monopole term as 0 and choose a momentum to display
        Cls_q_avg = jnp.insert(Cls_q_avg_pre, 0, 0, axis=1)
        # (z_cuts, 21)
        
        # print(Cls_q_avg.shape)

        # Generate temp. fluctuations map
        for i in range(len(z_cuts_select)):

            np.random.seed(seed)
            Deltas = hp.sphtfunc.synfast(Cls_q_avg[i], nside=Nside, lmax=None, pol=False)

            z_cut_unit = Primordial.uK/(Params.T_CNB * (1+z_cuts_select[i]))
            Deltas *= z_cut_unit
            # print(Deltas.shape)
            Deltas_l.append(Deltas)

    Deltas_arr = jnp.array(Deltas_l)
    print(Deltas_arr.shape)

    Delta_cmap = "inferno"
    dict_z4 = dict(
        title=fr"Temp. flucts. skymap: $m_\nu={m_val}$ eV, q-averaged, z={Primordial.z_cuts[indices[-1]]}", 
        unit=r"$\delta T_\nu / T_\nu (z=4)$",
        cmap=Delta_cmap,
        sub=121)
    dict_z0 = dict(
        title=fr"Temp. flucts. skymap: $m_\nu={m_val}$ eV, q-averaged, z={Primordial.z_cuts[indices[0]]}", 
        unit=r"$\delta T_\nu / T_\nu (z=0)$",
        cmap=Delta_cmap,
        sub=122)

    SimPlot.healpix_side_by_side(
        Deltas_arr[-1], Deltas_arr[0], dict_z4, dict_z0, 
        f"{fig_folder}/Delta_skymaps_m={m_val}eV_q-averaged.pdf", args)


def compute_percentages(v_arr, v_min, v_max, m_arr, args):

    vels_z0 = v_arr[:, 0, :]

    # Condition for having neutrinos with velocities in range, today
    mags_z0 = jnp.linalg.norm(vels_z0, axis=-1)*args.kpc/args.s
    cond = (v_min <= mags_z0) & (mags_z0 <= v_max)

    vels_range = v_arr[cond, ...]

    # Compute number densities for neutrinos in velocity range
    dens_range = Physics.number_densities_mass_range(
        vels_range, m_arr, 4*args.Pi, args)

    dens_total = Physics.number_densities_mass_range(
        v_arr, m_arr, 4*args.Pi, args)

    # Percentages 
    perc_range = dens_range/dens_total*100
    percs = [f"{perc:.3f}" for perc in perc_range]
    print(percs)


def momentum_distribution(y_arr, nbins, args):
    """
    Plot a momentum histogram with logarithmically spaced bins.
    """

    # Generating logarithmically spaced bins
    log_bins = np.logspace(
            np.log10(np.min(y_arr)), np.log10(np.max(y_arr)), nbins)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot simulation distances
    ax.hist(
        y_arr, bins=log_bins, 
        color="blue", label="CNB simulation", alpha=0.8)
    
    ax.set_xscale("log")
    ax.set_title("Momentum distribution at z=4 (Log Bins)")
    ax.set_xlabel(r"$p_\mathrm{back}/T_\mathrm{CNB,0}$")
    ax.set_ylabel("Nr. of neutrinos")
    ax.legend(loc="upper left")

    plt.show(); plt.close()
