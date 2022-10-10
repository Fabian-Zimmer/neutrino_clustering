###############
### Imports ### #SNELLIUS TEST
###############
import sys, os
from sys import getsizeof
import time
import glob
import random
import gc
import argparse
import pathlib
from itertools import chain, repeat
from memory_profiler import profile
import traceback
import math

# arrays and data packages
import numpy as np
import pandas as pd
import re
import h5py
from funcy import chunks

# astrophysics
from astropy import units as unit
from astropy import constants as const
import natpy as nat
import healpy as hp

# speed improvement
import numba as nb
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool

# scipy packages
from scipy.integrate import solve_ivp, quad, simpson
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.special import expit, zeta

# plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
import colorcet as cc
# import imageio


# Fix random seeds.
SEED = 1



#######################
### Path Management ###
#######################

HOME = pathlib.Path.home()

# Paths for FZ_snellius.
if str(HOME) == '/home/zimmer':
    # SIM_DATA_ROOT = '/projects/0/einf180/Tango_sims'
    SIM_DATA_ROOT = '/archive/ccorrea/Tango_sims'
    SIM_DATA_NEST = 'DMONLY/SigmaConstant00'
    CPUs_FOR_PRE = 14
    CPUs_FOR_SIM = 128

# Paths for FZ_desktop.
elif str(HOME) == '/home/fabian':
    SIM_DATA_ROOT = f'{HOME}/ownCloud/snellius'
    SIM_DATA_NEST = 'DMONLY'
    CPUs_FOR_PRE = 4
    CPUs_FOR_SIM = 6

# Paths for FZ_laptop.
elif str(HOME) == '/home/fpc':
    SIM_DATA_ROOT = f'{HOME}/SURFdrive/snellius'
    SIM_DATA_NEST = 'DMONLY'
    CPUs_FOR_PRE = 4
    CPUs_FOR_SIM = 6


#############
### Plots ###
#############
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('figure', figsize=(8, 8))
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



#############
### Units ###
#############
Pi = np.pi
hc_val = (const.h/(2*Pi)*const.c).convert(unit.eV*unit.cm).value

eV     = 1.                         # Unit of energy: eV
meV    = 1.0e-3*eV
keV    = 1.0e3*eV
MeV    = 1.0e3*keV
GeV    = 1.0e3*MeV
TeV    = 1.0e3*GeV
erg    = TeV/1.602                  # erg
J      = 1.0e7*erg                  # joule
K      = 8.61732814974493e-5*eV     # Kelvin

cm     = (1/hc_val)/eV              # centi-meter
m      = 1.0e2*cm
km     = 1.0e3*m
pc     = 3.08567758128e18*cm        # parsec
kpc    = 1.0e3*pc
Mpc    = 1.0e3*kpc

# note: 
# Natural units defined via c=1, i.e. s/m = 299792458
s      = 2.99792458e10*cm           # second
yr     = 365*24*60*60*s
Gyr    = 1e9*yr
t0     = 13.787*Gyr
Hz     = 1.0/s

kg     = J/m**2*s**2
gram   = kg/1000.
Msun   = 1.98847e30*kg              # Mass of the Sun
G      = 6.6743e-11*m**3/kg/s**2    # Gravitational constant
Da     = 1.66053906660e-27*kg       # Dalton or atomic mass unit (u)

deg    = Pi/180.0                   # Degree
arcmin = deg/60.                    # Arcminute
arcsec = arcmin/60.                 # Arcsecond
sr     = 1.                         # Steradian



#################
### Constants ###
#################
h = 0.674
H0 = h * 100 * km/s/Mpc
Omega_R = 9.23640e-5  # not used in simulation
Omega_M = 0.3111
Omega_L = 1.-Omega_M  # since we don't use Omega_R

T_CMB = 2.72548*K
T_CNB = np.power(4/11, 1/3)*T_CMB



##########################################
### Parameters - Mertsch et al. (2020) ###
##########################################

# NFW parameters for Milky Way.
Mvir_MW  = 2.03e12*Msun                            # Virial mass
rho0_MW  = 1.06e7*(Msun/kpc**3)                    # density normalization
Rs_MW    = 19.9*kpc                                # scale radius 
Rvir_MW  = 333.5*kpc                               # virial radius

# NFW parameters for Virgo Cluster.
Mvir_VC  = 6.9e14*Msun                             # Virial mass
rho0_VC  = 8.08e5*(Msun/kpc**3)                    # density normalization
Rs_VC    = 399.1*kpc                               # scale radius 
Rvir_VC  = 2328.8*kpc                              # virial radius

# Coordinates.
GLAT_VC = 74.44                                    # Galactic latitude
GLON_VC = 283.81                                   # Galactic longitude
DIST_VC = 16.5e3*kpc                               # Distance

# Translated to cartesian coordinates [kpc] in our setup (from fct.halo_pos).
X_VC     = np.array([-4289.63477282, 1056.51861602, 15895.27621304])

# NFW parameters for Andromeda.
Mvir_AG  = 8.0e11*Msun                             # Virial mass
rho0_AG  = 3.89e6*(Msun/kpc**3)                    # density normalization
Rs_AG    = 21.8*kpc                                # scale radius 
Rvir_AG  = 244.7*kpc                               # virial radius

# Coordinates.
GLAT_AG = -21.573311                               # Galactic latitude
GLON_AG = 121.174322                               # Galactic longitude
DIST_AG = 0.784e3*kpc                              # Distance

# Translated to cartesian coordinates [kpc] in our setup (from fct.halo_pos).
X_AG    = np.array([632.29742673, -377.40315121, -288.27006757])



######################
### Control Center ###
######################
# note:
# Using heaviest mass in conjunction with high momentum range covers
# velocity range sufficient for whole mass range. And in this way, the fastest 
# neutrino travels at ~22% of the speed of light.
NU_MASS = 0.3*eV  
NU_MASS_KG = NU_MASS/kg
NU_MASSES = np.array([0.01, 0.05, 0.1, 0.3])*eV

# Neutrino + antineutrino number density of 1 flavor in [1/cm**3],
# using the analytical expression for Fermions.
N0 = 2*zeta(3.)/Pi**2 *T_CNB**3 *(3./4.) /(1/cm**3)

PHIs = 20
THETAs = 20
Vs = 100
NUS = PHIs*THETAs*Vs

LOWER = 0.01*T_CNB
UPPER = 400.*T_CNB

# Momentum range.
# MOMENTA = np.geomspace(LOWER, UPPER, Vs)
MOMENTA = np.linspace(LOWER, UPPER, Vs)


def s_of_z(z):
    """Convert redshift to time variable s with eqn. 4.1 in Mertsch et al.
    (2020), keeping only Omega_m0 and Omega_Lambda0 in the Hubble eqn. for H(z).

    Args:
        z (float): redshift

    Returns:
        float: time variable s (in [seconds] if 1/H0 factor is included)
    """    

    def s_integrand(z):        

        # We need value of H0 in units of 1/s.
        H0_val = H0/(1/s)
        a_dot = np.sqrt(Omega_M*(1.+z)**3 + Omega_L)/(1.+z)*H0_val
        s_int = 1./a_dot

        return s_int

    s_of_z, _ = quad(s_integrand, 0., z)

    return np.float64(s_of_z)


# Logarithmic redshift spacing.
Z_AMOUNT = 100
z_shift = 1e-1
ZEDS = np.geomspace(z_shift, 4.+z_shift, Z_AMOUNT) - z_shift
S_STEPS = np.array([s_of_z(z) for z in ZEDS])

# Control if simulation runs forwards (+1) or backwards (-1) in time. 
TIME_FLOW = -1

# Position of earth w.r.t Milky Way NFW halo center.
# note: Earth is placed on x axis of coord. system.
X_SUN = np.array([8.5, 0., 0.])

# Available halos.
MW_HALO = True
VC_HALO = False
AG_HALO = False

SOLVER = 'RK23'


######################################
### Discrete simulation parameters ###
######################################

# sim = 'L006N188'
sim = 'L012N376'

zeds = np.zeros(25)
nums = []
for j, i in enumerate(range(12,37)):
    snap_i = f'{i:04d}'
    nums.append(snap_i)

    with h5py.File(
        f'{SIM_DATA_ROOT}/{sim}/{SIM_DATA_NEST}/snapshot_{snap_i}.hdf5'
    ) as snap:
        zeds[j] = snap['Cosmology'].attrs['Redshift'][0]
        if snap_i == '0036':

            # Get DM mass used in simulation box.
            dm_mass = snap['PartType1/Masses'][:]*1e10*Msun
            DM_SIM_MASS = np.unique(dm_mass)[0]

            # Get gravity smoothening length used in simulation box.
            sl = snap['GravityScheme'].attrs[
                'Maximal physical DM softening length (Plummer equivalent) [internal units]'
            ][0]
            SMOOTHENING_LENGTH = sl*1e6*pc


# note:
# MergerTree script needs N+1 snapshots to trace halo back through 
# N snapshots. I forgot to download snapshot 0011 then, and thus I can only
# start at snapshot 0013, since it was traced back until then, and not 0012,
# hence the [1:].
ZEDS_SNAPSHOTS = np.asarray(zeds)[1:]
NUMS_SNAPSHOTS = np.asarray(nums)[1:]