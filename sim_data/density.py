###############################################################################
# This file is part of SWIFT.
# Copyright (c) 2021 Camila Correa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

import matplotlib

matplotlib.use("Agg")
from pylab import *
import scipy.stats as stat
from scipy.interpolate import interp1d
import h5py
import numpy as np
import glob
import os.path
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter


def func(x,M,a):
    # Hernquist profile #
    f = M-np.log10(2.*np.pi)+a-x-3.*np.log10(10**x+10**a)
    return f

def calc_density(x,M,a):
    # Hernquist profile #
    f = M * a / (2. * np.pi * x)
    f *= 1./(x + a)**3
    return f

def sigma_1D(x,M,a):
    # M in Msun
    # x,a in kpc
    G = 4.3e-6  #kpc km^2 Msun^-1 s^-2
    ff = ((12.*x*(x+a)**3)/a**4)*np.log((x+a)/x)-(x/(x+a))*(25.+52.*(x/a)+42.*(x/a)**2+12.*(x/a)**3)
    f = (G*M/(12.*a))*ff
    return np.sqrt(f)


# Plot parameters
params = {
    "font.size": 16,
    "font.family": "STIXGeneral",
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "figure.figsize": (9, 4),
    "figure.subplot.left": 0.1,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.18,
    "figure.subplot.top": 0.95,
    "figure.subplot.wspace": 0.25,
    "figure.subplot.hspace": 0.25,
    "lines.markersize": 6,
    "lines.linewidth": 1.5,
    "figure.max_open_warning": 0,
}
rcParams.update(params)

# Physical constants
Msun_in_cgs = 1.98848e33
kpc_in_cgs = 3.08567758e21
Msun_p_kpc2 = Msun_in_cgs / kpc_in_cgs ** 2


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

def read_simulation(file):
    
    # Define radial bins [log scale, kpc units]
    radial_bins = np.arange(-1, 5, 0.1)
    radial_bins = 10**radial_bins
    centers = bin_centers(radial_bins) #kpc

    sim = h5py.File(file, "r")
    mass = sim["/PartType1/Masses"][:]
    pos = sim["/PartType1/Coordinates"][:,:]
    vel = sim["/PartType1/Velocities"][:,:]

    # Read units
    unit_length_in_cgs = sim["/Units"].attrs["Unit length in cgs (U_L)"]
    unit_mass_in_cgs = sim["/Units"].attrs["Unit mass in cgs (U_M)"]
    unit_time_in_cgs = sim["/Units"].attrs["Unit time in cgs (U_t)"]

    # Gemoetry info
    boxsize = sim["/Header"].attrs["BoxSize"]
    center = boxsize / 2.0

    # Turn the mass into Msun
    mass *= unit_mass_in_cgs / Msun_in_cgs

    # Radial coordinates [kpc units]
    r = np.sqrt(np.sum((pos - center) ** 2, axis=1))

    SumMasses, _, _ = stat.binned_statistic(x = r, values= np.ones(len(r))*mass[0], statistic="sum", bins=radial_bins,)
    density = (SumMasses / bin_volumes(radial_bins)) #Msun/kpc^3

    # Check 1D velocity dispersion
    vel *= unit_length_in_cgs/unit_time_in_cgs  #cm/s
    vel *= 1e-5 #km/s

    std_vel_x, _, _ = stat.binned_statistic(x = r, values= vel[:,0], statistic="std", bins=radial_bins,)
    std_vel_y, _, _ = stat.binned_statistic(x = r, values= vel[:,1], statistic="std", bins=radial_bins,)
    std_vel_z, _, _ = stat.binned_statistic(x = r, values= vel[:,2], statistic="std", bins=radial_bins,)
    std_vel = np.sqrt(std_vel_x**2 + std_vel_y**2 + std_vel_z**2)/np.sqrt(3.)
    return centers, density, std_vel

########
file = 'HernquistHalo.hdf5'

### SIDM halo ###
M = 12
rs = 23.43

#######################
# Plot the interesting quantities
figure()
ax = plt.subplot(1,2,1)
grid(True)

centers, density, velocity = read_simulation(file)
plot(centers,10**func(np.log10(centers),M, np.log10(rs)), '--',lw=2,color='black')
plot(centers,density, '-',color='tab:blue')

yscale('log')
xscale('log')
xlabel('r [kpc]')
ylabel(r'$\rho$ [M$_{\odot}$ kpc$^{-3}$]')
axis([1e-1,1e2,1e5,5e9])
locmin = ticker.LogLocator(base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
locmaj = ticker.LogLocator(base=10,numticks=12)
ax.xaxis.set_major_locator(locmaj)
locmin = ticker.LogLocator(base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

ax.tick_params(direction='in',axis='both',which='both',pad=4.5)

text(2,2e9,'Hernquist initial profile',fontsize=13)
text(2,1e9,'Halo Mass $10^{12}M_{\odot}$',fontsize=13)
text(2,5e8,'Scale radius 23.4 kpc',fontsize=13)

#######################
# Plot the interesting quantities
ax = plt.subplot(1,2,2)
grid(True)

plot(centers,sigma_1D(centers,10**M,rs),'--',lw=2,color='black')
plot(centers,velocity, '-',color='tab:blue')

ylabel(r'$\sigma_{\mathrm{1D}}$ [km/s]')
xlabel('r [kpc]')

xscale('log')
axis([1e-1,1e3,0,200])
locmin = ticker.LogLocator(base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
locmaj = ticker.LogLocator(base=10,numticks=12)
ax.xaxis.set_major_locator(locmaj)
locmin = ticker.LogLocator(base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.tick_params(direction='in',axis='both',which='both',pad=4.5)

savefig("Density.png", dpi=200)
plt.clf()
