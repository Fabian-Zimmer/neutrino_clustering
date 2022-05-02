
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

from colossus.cosmology import cosmology
cosmology.setCosmology('planck13');
from colossus.lss import mass_function

def HMF(siminfo,halo):

    dlogm = 0.2
    bins = 10 ** (np.arange(6, 15, dlogm))
    V = siminfo.boxSize ** 3 # Mpc

    # Load the data
    g = h5py.File(siminfo.halo_properties, "r")
    mass = g["Mass_200crit"][:] * 1e10  # convert to Msun

    if halo == 'subhalo':
        subtype = g["Structuretype"][:]
        subhalo = subtype > 10
        mass = mass[subhalo]

    binnedmass, massrange = np.histogram(mass, bins=bins)

    massnlarger = np.zeros(len(binnedmass))
    for i in range(0, len(massnlarger)):
        massnlarger[i] = np.sum(binnedmass[i:])

    f = h5py.File(siminfo.latest_snapshot, "r")
    cosmo = f["Cosmology"]
    redshift = cosmo.attrs["Redshift"][0]
    mass = f["PartType1/Masses"][:]
    mass = mass[0] * 1e10 # convert to Msun
    limit = 100 * mass

    # Determine the HMF
    errormassn = massnlarger ** 0.5
    numbden = massnlarger / V
    numbdenerr = errormassn / V
    massplot = (massrange[:-1] + massrange[1:]) / 2
    dernumbden = -np.diff(numbden) / np.diff(np.log10(massplot))
    dererr = 2 ** 0.5 / dlogm * (numbdenerr[:-1] + numbdenerr[1:]) / 2

    M = (massplot[:-1] + massplot[1:]) / 2.
    dndM = dernumbden
    error = dererr

    return M, dndM, error, limit, redshift


def make_HMF(siminfo,output_path):

    M, dndM, error, limit, redshift = HMF(siminfo,'halo')

    # Plot parameters
    params = {
        "font.size": 14,
#        "font.family": "Times",
#        "text.usetex": True,
        "figure.figsize": (5, 4),
        "figure.subplot.left": 0.17,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.3,
        "figure.subplot.hspace": 0.3,
        "lines.markersize": 6,
        "lines.linewidth": 1.0,
        "figure.max_open_warning": 0,
    }
    rcParams.update(params)

    figure()
    ax = plt.subplot(1,1,1)
    plt.grid("True")

    plt.plot(M, dndM, label="SWIFT",lw=2,color='tab:green')
    plt.fill_between(M, dndM - error, dndM + error, alpha=0.4,color='tab:green')

    h = siminfo.h
    M = np.arange(7, 16, 0.2)
    M = 10**M
    mfunc = mass_function.massFunction(M / h, redshift, mdef = '200c', model = 'tinker08', q_out = 'dndlnM')
    plt.plot(M, mfunc / h**3, '-', label = 'Tinker et al. (2008)',color='black',zorder=1)

    plt.xscale("log")
    plt.ylim(1e-7, 1e0)
    plt.xlim(10 ** 9, 10 ** 15)
    plt.axvline(x=limit, linestyle="--", lw=1, color="grey")

    plt.xlabel("M${}_{200,\mathrm{crit}}$ ($M_\odot$)")
    plt.ylabel("dn/d($\log$10(M${}_{200}$) (Mpc$^{-3}$)")
    ax.tick_params(direction='in',axis='both',which='both',pad=4.5)
    plt.yscale("log")
    plt.legend(loc=[0.45,0.8],labelspacing=0.2,handlelength=1.5,handletextpad=0.4,frameon=False)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_path+"HMF_%04d.png" % siminfo.n_snapshots)#, dpi=200)
    plt.close()


    M, dndM, error, limit, redshift = HMF(siminfo,'subhalo')

    figure()
    ax = plt.subplot(1,1,1)
    plt.grid("True")

    plt.plot(M, dndM, label="SWIFT",lw=2,color='tab:blue')
    plt.fill_between(M, dndM - error, dndM + error, alpha=0.4,color='tab:blue')

    plt.xscale("log")
    plt.ylim(1e-7, 1e0)
    plt.xlim(10 ** 9, 10 ** 15)
    plt.axvline(x=limit, linestyle="--", lw=1, color="grey")

    h = siminfo.h
    M = np.arange(7, 16, 0.2)
    M = 10**M
    mfunc = mass_function.massFunction(M / h, redshift, mdef = '200c', model = 'tinker08', q_out = 'dndlnM')
    plt.plot(M, mfunc / h**3, '-', label = 'Tinker et al. (2008)',color='black',zorder=1)

    plt.xlabel("M${}_{200,\mathrm{crit}}$ ($M_\odot$)")
    plt.ylabel("(Subhalo) dn/d($\log$10(M${}_{200}$) (Mpc$^{-3}$)")
    ax.tick_params(direction='in',axis='both',which='both',pad=4.5)
    plt.yscale("log")
    plt.legend(loc=[0.45,0.8],labelspacing=0.2,handlelength=1.5,handletextpad=0.4,frameon=False)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_path+"SubHMF_%04d.png" % siminfo.n_snapshots)#, dpi=200)
    plt.close()
