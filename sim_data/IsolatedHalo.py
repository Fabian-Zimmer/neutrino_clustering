import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy.optimize import curve_fit

def logfunc(x,rc):
    # Hernquist profile #
    M = 1e14
    a = 225
    beta = 4.
    f = np.log10(M)-np.log10(2.*np.pi)
    f += np.log10(a)-(1./beta)*np.log10((10**x)**beta + rc**beta)
    f -= 3.*np.log10(10**x+a)
    return f

def func(x,M,a):
    # Hernquist profile #
    f = M-np.log10(2.*np.pi)+a-x-3.*np.log10(10**x+10**a)
    return f

def bin_volumes(radial_bins):
    """Returns the volumes of the bins. """

    single_vol = lambda x: (4.0 / 3.0) * np.pi * x ** 3
    outer = single_vol(radial_bins[1:])
    inner = single_vol(radial_bins[:-1])
    return outer - inner


def bin_centers(radial_bins):
    """Returns the centers of the bins. """

    outer = radial_bins[1:]
    inner = radial_bins[:-1]
    return 0.5 * (outer + inner)


def fit_profile(file, radial_bins, centers):

    sim = h5py.File(file, "r")
    pos = sim["/PartType1/Coordinates"][:, :]
    mass = sim["/PartType1/Masses"][:] * 1e10
    num = len(mass)

    # Geometry info
    boxsize = sim["/Header"].attrs["BoxSize"]
    center = boxsize / 2.0

    # Radial coordinates [kpc units]
    r = np.sqrt(np.sum((pos - center) ** 2, axis=1))

    SumMasses, _, _ = stat.binned_statistic(x=r, values=np.ones(len(r)) * mass[0], statistic="sum", bins=radial_bins, )
    NumParts, _, _ = stat.binned_statistic(x=r, values=np.ones(len(r)), statistic="sum", bins=radial_bins, )
    density = (SumMasses / bin_volumes(radial_bins))  # Msun/kpc^3

    select = np.where((NumParts > 5) & (centers < 1e3))[0]
    xdata = np.log10(centers[select])
    ydata = np.log10(density[select])

    # Doing fit #
    popt, pcov = curve_fit(func, xdata, ydata)

    M = popt[0]
    rs = 10 ** popt[1]
    return M, rs, num


def read_simulation(folder, snap):

    snap_list = np.arange(0,snap,1)

    # Define radial bins [log scale, kpc units]
    radial_bins = np.arange(0, 5, 0.1)
    radial_bins = 10 ** radial_bins
    centers = bin_centers(radial_bins)  # kpc

    # Fit density profile
    M, rs, n_parts = fit_profile(folder+"/halo_0000.hdf5", radial_bins, centers)

    n_snaps = len(snap_list)
    density_in_bin_per_snap = np.zeros((len(centers), n_snaps))
    velocity_in_bin_per_snap = np.zeros((len(centers), n_snaps))

    time = np.zeros(n_snaps)
    core = np.zeros(n_snaps)
    old_time = 0

    # Only follow scatter, probability and radius for first ten snaps e.g.
    n_snap_s = 10

    # Table containing positions of particle's collisions and time
    n_collisions = np.zeros((n_parts, n_snap_s))
    n_positions = np.zeros((n_parts, n_snap_s))

    scatter_rate_per_bin = np.zeros((len(centers), n_snap_s))
    probability_per_bin = np.zeros((len(centers), n_snap_s))
    search_radius_per_bin = np.zeros((len(centers), n_snap_s))

    i = 1
    for ii in snap_list:

        sim = h5py.File(folder + "/halo_%04d.hdf5" % ii, "r")
        pos = sim["/PartType1/Coordinates"][:, :]
        mass = sim["/PartType1/Masses"][:] * 1e10
        vel = sim["/PartType1/Velocities"][:, :]
        if ii < n_snap_s:
            nsidm = sim["/PartType1/SIDM_events"][:]
            ids = sim["/PartType1/ParticleIDs"][:]
            prob = sim["/PartType1/SIDM_probability"][:]
            h = sim["/PartType1/SIDM_search_radius"][:]
            timestep = sim["/PartType1/Time_step_size"][:]
            prob *= timestep

        # Read units
        unit_length_in_cgs = sim["/Units"].attrs["Unit length in cgs (U_L)"]
        unit_time_in_cgs = sim["/Units"].attrs["Unit time in cgs (U_t)"]

        # Geometry info
        boxsize = sim["/Header"].attrs["BoxSize"]
        center = boxsize / 2.0

        #Time info
        t = sim["/Header"].attrs["Time"] * 0.979 #Gyr
        time[i-1] = t
        delta_time = t - old_time
        old_time = t

        # Radial coordinates [kpc units]
        r = np.sqrt(np.sum((pos - center) ** 2, axis=1))

        SumMasses, _, _ = stat.binned_statistic(x=r, values=np.ones(len(r)) * mass[0], statistic="sum",
                                                bins=radial_bins, )
        density = (SumMasses / bin_volumes(radial_bins))  # Msun/kpc^3
        density_in_bin_per_snap[:, i-1] = density

        # Doing fit #
        select = np.where( (centers>4) & (centers<1e4) )[0]
        popt, pcov = curve_fit(logfunc, np.log10(centers[select]), np.log10(density[select]))
        core[i-1] = popt[0]

        # Check 1D velocity dispersion
        vel *= unit_length_in_cgs / unit_time_in_cgs  # cm/s
        vel *= 1e-5  # km/s

        std_vel_x, _, _ = stat.binned_statistic(x=r, values=vel[:, 0], statistic="std", bins=radial_bins, )
        std_vel_y, _, _ = stat.binned_statistic(x=r, values=vel[:, 1], statistic="std", bins=radial_bins, )
        std_vel_z, _, _ = stat.binned_statistic(x=r, values=vel[:, 2], statistic="std", bins=radial_bins, )
        std_vel = np.sqrt(std_vel_x ** 2 + std_vel_y ** 2 + std_vel_z ** 2) / np.sqrt(3.)
        velocity_in_bin_per_snap[:, i-1] = std_vel

        if i < n_snap_s:
            ids_sorted = np.argsort(ids)
            r = r[ids_sorted]
            nsidm = nsidm[ids_sorted]
            prob = prob[ids_sorted]
            part_h = h[ids_sorted]

            n_collisions[:, i - 1] = nsidm
            n_positions[:, i - 1] = r

            for j in range(0, len(radial_bins) - 1):

                select = np.where((n_positions[:, i - 1] >= radial_bins[j]) &
                                  (n_positions[:, i - 1] < radial_bins[j + 1]))[0]

                num_parts = len(select)
                if num_parts > 1:

                    if i == 1: num_collisions = np.sum(n_collisions[select, i - 1])
                    if i > 1: num_collisions = np.sum(n_collisions[select, i - 1]) - np.sum(n_collisions[select, i - 2])
                    if num_collisions > 0: scatter_rate_per_bin[j, i - 1] = num_collisions / (delta_time * num_parts)
                    probability_per_bin[j, i - 1] = np.median(prob[select])
                    search_radius_per_bin[j, i - 1] = np.median(part_h[select])

        i += 1  # update counter


    density = np.zeros((len(centers), 5))
    velocity = np.zeros((len(centers), 5))
    density[:,0] = np.mean(density_in_bin_per_snap[:, 0:2], axis=1)
    density[:,1] = np.mean(density_in_bin_per_snap[:, 8:12], axis=1)
    density[:,2] = np.mean(density_in_bin_per_snap[:, 18:22], axis=1)
    density[:,3] = np.mean(density_in_bin_per_snap[:, 38:42], axis=1)
    #density[:,4] = np.mean(density_in_bin_per_snap[:, 78:82], axis=1)
    velocity[:,0] = np.mean(velocity_in_bin_per_snap[:, 0:2], axis=1)
    velocity[:,1] = np.mean(velocity_in_bin_per_snap[:, 8:12], axis=1)
    velocity[:,2] = np.mean(velocity_in_bin_per_snap[:, 18:22], axis=1)
    velocity[:,3] = np.mean(velocity_in_bin_per_snap[:, 38:42], axis=1)
    #velocity[:,4] = np.mean(velocity_in_bin_per_snap[:, 78:82], axis=1)

    scatter_rate = np.mean(scatter_rate_per_bin[:,  0:i], axis=1)
    probability = np.mean(probability_per_bin[:,  0:i], axis=1)
    search_radius = np.mean(search_radius_per_bin[:, 0:i], axis=1)

    return centers, density, velocity, scatter_rate, probability, search_radius, time, core


def plot_core_evolution(x, y):

    # Plot parameters
    params = {
        "font.size": 12,
      #  "font.family": "Times",
      #  "text.usetex": True,
        "figure.figsize": (4, 3),
        "figure.subplot.left": 0.18,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.45,
        "figure.subplot.hspace": 0.35,
        "lines.markersize": 2,
        "lines.linewidth": 2,
    }
    rcParams.update(params)

    figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid("True")

    plt.plot(x, y, '-', color='tab:blue')
    #plt.axis([1e9, 1e15, 1, 20])
    plt.xlabel("Time [Gyr]")
    plt.ylabel("Core Radius [kpc]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_path + "core_evolution.png", dpi=200)
    plt.close()


def plot_probability(x, y):

    # Plot parameters
    params = {
        "font.size": 12,
      #  "font.family": "Times",
      #  "text.usetex": True,
        "figure.figsize": (4, 3),
        "figure.subplot.left": 0.18,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.45,
        "figure.subplot.hspace": 0.35,
        "lines.markersize": 2,
        "lines.linewidth": 2,
    }
    rcParams.update(params)

    figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid("True")

    plt.plot(x, y, '-', color='tab:blue')
    plt.axis([1e0, 1e3, 1e-4,1e0])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Probability")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_path + "probability.png", dpi=200)
    plt.close()

def plot_scatter_rate(x, y):

    # Plot parameters
    params = {
        "font.size": 12,
      #  "font.family": "Times",
      #  "text.usetex": True,
        "figure.figsize": (4, 3),
        "figure.subplot.left": 0.18,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.45,
        "figure.subplot.hspace": 0.35,
        "lines.markersize": 2,
        "lines.linewidth": 2,
    }
    rcParams.update(params)

    figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid("True")

    plt.plot(x, y, '-', color='tab:blue')
    plt.axis([1e0, 1e3, 1e-4, 1e2])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Scatter rate [particle$^{-1}$ Gyr$^{-1}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_path + "scatter_rate.png", dpi=200)
    plt.close()

def plot_search_radius(x, y):

    # Plot parameters
    params = {
        "font.size": 12,
       # "font.family": "Times",
       # "text.usetex": True,
        "figure.figsize": (4, 3),
        "figure.subplot.left": 0.18,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.45,
        "figure.subplot.hspace": 0.35,
        "lines.markersize": 2,
        "lines.linewidth": 2,
    }
    rcParams.update(params)

    figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid("True")

    plt.plot(x, y, '-', color='tab:blue')
    plt.axis([1e0, 1e3, 1e-2,1e2])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Search radius [kpc]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_path + "search_radius.png", dpi=200)
    plt.close()


def plot_profile_evolution(x, rho, vel):

    # Plot parameters
    params = {
        "font.size": 12,
       # "font.family": "Times",
       # "text.usetex": True,
        "figure.figsize": (7, 3),
        "figure.subplot.left": 0.1,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.35,
        "figure.subplot.hspace": 0.25,
        "lines.markersize": 2,
        "lines.linewidth": 2,
    }
    rcParams.update(params)

    figure()
    ax = plt.subplot(1, 2, 1)
    plt.grid("True")

    plt.plot(x, rho[:,0], '-', color='tab:blue',label='t = 0 Gyr')
    plt.plot(x, rho[:,1], '-', color='tab:orange',label='t = 1 Gyr')
    plt.plot(x, rho[:,2], '-', color='tab:green',label='t = 2 Gyr')
    plt.plot(x, rho[:,3], '-', color='tab:red',label='t = 4 Gyr')
    #plt.plot(x, rho[:,4], '-', color='tab:purple',label='t = 8 Gyr')

    plt.axis([1e0,1e3, 1e5, 1e9])
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel("Radius [kpc]")
    plt.ylabel("Density [M$_{\odot}$/kpc$^{3}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(labelspacing=0.2, handlelength=1.5, handletextpad=0.4, frameon=False)

    ax = plt.subplot(1, 2, 2)
    plt.grid("True")

    plt.plot(x, vel[:,0], '-', color='tab:blue',label='t = 0 Gyr')
    plt.plot(x, vel[:,1], '-', color='tab:orange',label='t = 1 Gyr')
    plt.plot(x, vel[:,2], '-', color='tab:green',label='t = 2 Gyr')
    plt.plot(x, vel[:,3], '-', color='tab:red',label='t = 4 Gyr')
    #plt.plot(x, vel[:,4], '-', color='tab:purple',label='t = 8 Gyr')

    plt.axis([1e0, 1e3, 0, 600])
    plt.xscale('log')
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Velocity dispersion [km/s]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_path + "profile_evolution.png", dpi=200)
    plt.close()

if __name__ == '__main__':
    from utils import *

    output_path = args.output
    folder = args.directory
    snapshot = args.snapshot

    centers, density, velocity, \
    scatter_rate, probability, \
    search_radius, time, core = read_simulation(folder, snapshot)

    plot_profile_evolution(centers, density, velocity)
    plot_core_evolution(time, core)
    plot_probability(centers, probability)
    plot_scatter_rate(centers, scatter_rate)
    plot_search_radius(centers, search_radius)
